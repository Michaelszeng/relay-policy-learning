"""
Episode recording for eval/evaluate_kitchen.py rollouts (--record-zarr).

Writes a plain zarr in the exact layout of
kitchen_demos_markovian_scripted_expert.zarr (see experts/record_demos.py):

    data/action  (N, 9)            float64   action handed to env.step
    data/state   (N, 60)           float64   noised env obs the policy saw
    data/scene   (N, 240, 320, 3)  uint8     scene camera (camera_id 2)
    data/wrist   (N, 240, 320, 3)  uint8     wrist camera (camera_id 3)
    data/current_subtask  (N, 7)   float64   all-zeros: the expert oracle's
                                             current-subtask label doesn't exist
                                             for a policy rollout (kept only for
                                             layout parity with the dataset)
    data/subtask_sequence (N, 28)  float64   the plan one-hot fed to the policy
                                             (all-zeros for unconditioned policies)
    meta/episode_ends (E,)         int64     cumulative episode end indices

plus two extra arrays holding the full MuJoCo state (model.na == 0, so
qpos+qvel is the complete physical state) at each recorded timestep, so the
sim can be reset back to any step later:

    data/qpos (N, nq=30) float64   ground-truth sim qpos at obs time
    data/qvel (N, nv=29) float64   ground-truth sim qvel at obs time

Per step we store (o_t, a_t): the obs/images/sim-state seen *before* stepping,
paired with the action taken -- the same convention as record_demos.py. `state`
is the *noised* observation (matching the training/eval distribution), while
qpos/qvel are the exact unnoised sim state that produced it.

Replay example:

    import zarr
    from episode_recorder import restore_sim_state

    root = zarr.open_group("rollouts.zarr", mode="r")
    t = ...  # global step index (use meta/episode_ends to find episode bounds)
    env.reset()  # build goal/obs machinery, then overwrite the sim state
    restore_sim_state(env, root["data/qpos"][t], root["data/qvel"][t])
"""

import shutil
from pathlib import Path

import numcodecs
import numpy as np
import zarr

# Same codec/chunking as experts/record_demos.py so the recorded zarr is
# structurally identical to the expert dataset (plus the qpos/qvel arrays).
_COMPRESSOR = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
_CHUNKS = {
    "action": (2048, 9),
    "state": (1024, 60),
    "scene": (128, 240, 320, 3),
    "wrist": (128, 240, 320, 3),
    "current_subtask": (1024, 7),
    "subtask_sequence": (1024, 28),
    "qpos": (1024, 30),
    "qvel": (1024, 29),
}


class _ZarrEpisodeWriter:
    """Incrementally write episodes in the ReplayBuffer zarr layout.

    Mirror of experts/record_demos.py's writer: per-key arrays under
    ``data/<key>`` plus ``meta/episode_ends``, appended episode-by-episode so
    the camera streams never sit in RAM.
    """

    def __init__(self, out_path, chunks, compressor):
        out_path = Path(out_path)
        if out_path.exists():
            raise FileExistsError(
                f"--record-zarr target already exists: {out_path} -- refusing to "
                "overwrite; pass a fresh path."
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(str(out_path), mode="w")
        self.data = self.root.create_group("data")
        self.meta = self.root.create_group("meta")
        self.chunks = chunks
        self.compressor = compressor
        self.episode_ends = []
        self.n_steps = 0

    def add_episode(self, episode):
        T = next(iter(episode.values())).shape[0]
        for key, value in episode.items():
            if key not in self.data:
                self.data.zeros(
                    key,
                    shape=(0,) + value.shape[1:],
                    chunks=self.chunks.get(key),
                    dtype=value.dtype,
                    compressor=self.compressor,
                )
            self.data[key].append(value)
        self.n_steps += T
        self.episode_ends.append(self.n_steps)
        # Rewritten after every episode (it's tiny) so the zarr stays valid even
        # if the run is killed before finalize() -- e.g. Ctrl+C mid-eval.
        self._write_episode_ends()

    def _write_episode_ends(self):
        ends = np.asarray(self.episode_ends, dtype=np.int64)
        self.meta.array("episode_ends", data=ends, chunks=ends.shape if ends.size else (1,), overwrite=True)

    def finalize(self):
        self._write_episode_ends()


class RecordingSingleEnv:
    """Drop-in wrapper around BatchedSingleEnv that records every episode.

    Exposes the same interface (n_envs / reset / step / render / close), so
    run_rollout needs no changes. Episode boundaries follow reset() calls; the
    final episode is flushed and the zarr finalized on close(). Requires the
    in-process single-env adapter (n_envs == 1): the recorder snapshots
    env.sim.data directly, which subprocess workers don't expose.
    """

    def __init__(self, inner, out_path, n_subtasks=7, seq_dim=28):
        assert inner.n_envs == 1, "RecordingSingleEnv requires a single in-process env"
        self.inner = inner
        self.env = inner.env
        self.n_envs = inner.n_envs
        self._n_subtasks = n_subtasks
        self._seq_onehot = np.zeros(seq_dim, dtype=np.float64)
        self._writer = _ZarrEpisodeWriter(out_path, _CHUNKS, _COMPRESSOR)
        self._out_path = Path(out_path)
        self._buf = []  # completed (o_t, a_t) rows of the current episode
        self._pending = None  # o_t awaiting its action
        self._closed = False

    def set_sequence_onehot(self, seq_onehot):
        """Set the (constant) plan one-hot recorded with the next episode's steps.

        Call before run_rollout each round with the same encoding fed to the
        policy; leave unset (all-zeros) for policies without plan conditioning.
        """
        self._seq_onehot = np.asarray(seq_onehot, dtype=np.float64).reshape(-1).copy()

    def _snapshot(self, state, cams):
        """Capture o_t: noised obs + frames + the ground-truth sim state behind them."""
        return {
            "state": np.asarray(state, dtype=np.float64).copy(),
            "scene": np.ascontiguousarray(cams["scene"], dtype=np.uint8),
            "wrist": np.ascontiguousarray(cams["wrist"], dtype=np.uint8),
            "qpos": np.asarray(self.env.sim.data.qpos, dtype=np.float64).copy(),
            "qvel": np.asarray(self.env.sim.data.qvel, dtype=np.float64).copy(),
        }

    def _flush_episode(self):
        if self._buf:
            episode = {k: np.stack([row[k] for row in self._buf]) for k in self._buf[0]}
            self._writer.add_episode(episode)
        self._buf = []
        self._pending = None

    def reset(self):
        self._flush_episode()
        s, c = self.inner.reset()
        self._pending = self._snapshot(s[0], {k: v[0] for k, v in c.items()})
        return s, c

    def step(self, actions):
        # Pair the held o_t with the action about to be taken, *before* stepping:
        # if the sim blows up mid-step we still keep the pair that triggered it.
        if self._pending is not None:
            row = self._pending
            self._pending = None
            row["action"] = np.asarray(actions[0], dtype=np.float64).copy()
            row["current_subtask"] = np.zeros(self._n_subtasks, dtype=np.float64)
            row["subtask_sequence"] = self._seq_onehot.copy()
            self._buf.append(row)
        s, c = self.inner.step(actions)
        self._pending = self._snapshot(s[0], {k: v[0] for k, v in c.items()})
        return s, c

    def render(self):
        self.inner.render()

    def close(self):
        if not self._closed:
            self._flush_episode()
            self._writer.finalize()
            self._closed = True
            print(
                f"[episode_recorder] wrote {len(self._writer.episode_ends)} episodes, "
                f"{self._writer.n_steps} steps -> {self._out_path}"
            )
        self.inner.close()


def restore_sim_state(env, qpos, qvel):
    """Reset the kitchen sim to a recorded (qpos, qvel) snapshot.

    Call env.reset() once beforehand (initializes goal/obs machinery), then this
    to overwrite the physical state. Writes the full state directly and refreshes
    the robot's observation cache -- the velocity-actuated controller integrates
    its next position target from the cached last obs, so a stale cache would
    corrupt the first replayed step. (MujocoEnv.set_state is avoided: it assumes
    the mujoco_py state-object API, which the dm_control backend doesn't have.)
    """
    env.sim.data.qpos[:] = np.asarray(qpos)
    env.sim.data.qvel[:] = np.asarray(qvel)
    env.sim.forward()
    env.robot._observation_cache_refresh(env)
