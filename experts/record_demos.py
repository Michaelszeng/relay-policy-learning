"""Record Markovian-expert rollouts into a diffusion_policy ReplayBuffer zarr.

Format (plain zarr):

    data/action  (N, 9)            float64   normalized joint-velocity action
    data/state   (N, 60)           float64   env obs = [qp(9), obj_qp(21), goal(30)]
    data/scene   (N, 240, 320, 3)  uint8     scene camera (camera_id 2)
    data/wrist   (N, 240, 320, 3)  uint8     wrist camera (camera_id 3)
    data/current_subtask  (N, 7)   float64   one-hot of the subtask being executed
    data/subtask_sequence (N, 28)  float64   flattened one-hot (4 slots x 7) of the
                                             fixed plan; constant within an episode
    meta/episode_ends (E,)         int64     cumulative episode end indices

The two subtask keys let the policy condition on the same information the
Markovian expert uses -- the current subtask (which changes through the episode)
and the fixed plan (constant) -- so the otherwise multi-modal state->action map
(the same state demands different moves for different subtasks) is disambiguated.
Encoding: one-hot over the 7 subtasks (SUBTASK_IDS in subtasks/base.py); the
sequence is 4 slots flattened to 28, with empty slots (chains shorter than 4)
left all-zeros. The training shape_meta keys are current_subtask=[7],
subtask_sequence=[28].

We record exactly the 9-D action (which are joint velocities) handed to env.step, after the
Diff IK conversion.

Per step we store (o_t, a_t): the obs/images seen *before* stepping paired with
the action taken.
"""

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))  # controller, expert
sys.path.insert(0, str(_REPO / "adept_envs"))  # adept_envs

import controller  # noqa: E402
import expert  # noqa: E402
import gym  # noqa: E402
import numcodecs  # noqa: E402
import zarr  # noqa: E402

import adept_envs  # noqa: E402,F401  (registers kitchen envs)

# Match kitchen_demos.zarr: blosc/lz4, clevel 5, byte shuffle, with the same
# per-array chunking. (Chunking only affects IO, not readability, but we mirror
# it so the datasets are byte-for-byte comparable in structure.)
_COMPRESSOR = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
_CHUNKS = {
    "action": (2048, 9),
    "state": (1024, 60),
    "scene": (128, 240, 320, 3),
    "wrist": (128, 240, 320, 3),
    "current_subtask": (1024, 7),
    "subtask_sequence": (1024, 28),
}
_IMG_W, _IMG_H = 320, 240


class _ZarrEpisodeWriter:
    """Incrementally write episodes to a zarr in the ReplayBuffer layout.

    Produces exactly the layout the kitchen dataset reads -- per-key arrays under
    ``data/<key>`` plus ``meta/episode_ends`` (cumulative episode end indices) --
    but appends each episode straight to the on-disk arrays, so the whole dataset
    (notably the two camera streams) never has to sit in RAM.
    """

    def __init__(self, out_path, chunks, compressor):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():  # if_exists="replace"
            shutil.rmtree(out_path)
        self.root = zarr.open_group(str(out_path), mode="w")
        self.data = self.root.create_group("data")
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
            self.data[key].append(value)  # grows the on-disk array along axis 0
        self.n_steps += T
        self.episode_ends.append(self.n_steps)

    def finalize(self):
        ends = np.asarray(self.episode_ends, dtype=np.int64)
        meta = self.root.create_group("meta")
        # episode_ends stored unchunked, matching the reference dataset
        meta.array("episode_ends", data=ends, chunks=ends.shape if ends.size else (1,))


def _rollout(env, sequence, max_steps):
    """Run one expert rollout, returning per-step arrays, success and osc flags.

    The runner plays the env oracle: it advances the subtask label only once the
    current subtask is complete AND the arm has returned to the home pose.

    Oscillation detection: in a clean rollout each subtask's FSM marches
    monotonically through its phases (move-to-pre-grasp -> ... -> return-to-home)
    and never re-enters a phase it has already left. We key each phase by
    (subtask_index, fsm_state) -- the index matters because phase names repeat
    across subtasks -- and flag the episode if we ever transition *into* a key we
    previously *left*. That re-entry means the FSM is dithering between phases
    (the rare residual jitter), so the episode is treated as a failure and kept
    out of the dataset even if the task technically still completes.
    """
    obs = env.reset()
    idx = 0
    state, action, scene, wrist = [], [], [], []
    current_subtask, subtask_sequence = [], []
    # The plan is fixed for the whole episode -> one constant vector, recorded at
    # every step so it routes as a per-step low-dim obs key like everything else.
    seq_onehot = expert.sequence_onehot(sequence)

    last_key = None  # (idx, fsm_state) of the previous step
    left_keys = set()  # (idx, fsm_state) phases we have entered and then left
    oscillated = False
    osc_detail = None

    for _ in range(max_steps):
        if idx >= len(sequence):
            break
        label = sequence[idx]

        cams = env.render_cameras(width=_IMG_W, height=_IMG_H)
        subtask, fsm_state = expert.compute_state(env, label)
        gripper, tpos, trot = expert.compute_action(env, subtask, fsm_state)
        if fsm_state == "return-to-home":
            act = controller.joint_pose_to_action(env, expert.RESET_ARM_QPOS, gripper)
        else:
            act = controller.pose_to_action(env, tpos, trot, gripper)

        # Phase-revisit (oscillation) check, on the collapsed transition sequence:
        # only act on an actual phase change, and flag re-entry of a left phase.
        key = (idx, fsm_state)
        if key != last_key:
            if last_key is not None:
                left_keys.add(last_key)
            if key in left_keys and not oscillated:
                oscillated = True
                osc_detail = f"{label}:{fsm_state}"
            last_key = key

        # (o_t, a_t): obs/images before the step, paired with the action taken.
        # IMPORTANT: `obs` is the *noised* observation (env.step -> _get_obs ->
        # get_obs applies robot_noise_ratio=0.1), the same noisy obs the policy
        # sees at eval. The expert's *control* (pose_to_action above) instead
        # reads ground-truth sim.data, so the expert acts perfectly while the
        # recorded state matches the eval distribution. Do NOT swap this for a
        # ground-truth state -- that would create a train/eval distribution shift.
        state.append(np.asarray(obs, dtype=np.float64))
        action.append(np.asarray(act, dtype=np.float64))
        scene.append(np.ascontiguousarray(cams["scene"], dtype=np.uint8))
        wrist.append(np.ascontiguousarray(cams["wrist"], dtype=np.uint8))
        # Subtask conditioning: one-hot of the subtask being executed this step
        # (the oracle's current label) and the flattened one-hot of the fixed
        # plan. Recording the same information the Markovian expert acts on lets
        # the policy condition on it too (resolves the otherwise-ambiguous
        # state->action mapping where one state maps to different subtasks' moves).
        current_subtask.append(expert.subtask_onehot(label))
        subtask_sequence.append(seq_onehot)

        obs, _, _, _ = env.step(act)

        # oracle: advance once the current subtask is done and back home
        if expert.is_done(env, label) and expert.at_home(env):
            idx += 1

    task_complete = all(expert.is_done(env, s) for s in sequence)
    success = task_complete and not oscillated
    episode = {
        "state": np.stack(state),
        "action": np.stack(action),
        "scene": np.stack(scene),
        "wrist": np.stack(wrist),
        "current_subtask": np.stack(current_subtask),
        "subtask_sequence": np.stack(subtask_sequence),
    }
    return episode, success, oscillated, osc_detail


def record(sequence, n_episodes, out_path, keep_failures=False, randomize=False, chain_len=4, seed=0):
    """Record n_episodes successful rollouts to a zarr.

    Two modes:
      * fixed (default): every episode runs the same `sequence`.
      * randomize=True: each *attempt* samples a fresh random chain of
        `chain_len` distinct subtasks (random subset AND random order) from the
        7 implemented subtasks, for diverse coverage across subtasks and
        sequences. Sampling per attempt (not per saved episode) means a rare
        hard sequence that keeps failing is simply replaced by a new draw rather
        than deadlocking the loop. Reproducible via `seed`.
    """
    env = gym.make("kitchen_relax-v1").unwrapped
    writer = _ZarrEpisodeWriter(out_path, _CHUNKS, _COMPRESSOR)
    rng = random.Random(seed)
    all_subtasks = list(expert.SUBTASKS.keys())

    subtask_coverage = Counter()  # subtask -> # saved episodes containing it
    n_saved, n_attempt = 0, 0
    while n_saved < n_episodes:
        n_attempt += 1
        if randomize:
            seq = rng.sample(all_subtasks, chain_len)
        else:
            seq = sequence
        episode, success, oscillated, osc_detail = _rollout(env, seq, 200 * len(seq))
        T = episode["action"].shape[0]
        if success or keep_failures:
            writer.add_episode(episode)
            n_saved += 1
            subtask_coverage.update(seq)
            tag = "ok" if success else ("FAIL-osc(kept)" if oscillated else "FAIL(kept)")
            print(
                f"  episode {n_saved}/{n_episodes} [{tag}] len={T} "
                f"seq={'+'.join(seq)} (attempt {n_attempt}) "
                f"[success rate: {n_saved}/{n_attempt} = {n_saved / n_attempt:.1%}]"
            )
        else:
            reason = f"OSCILLATION @ {osc_detail}" if oscillated else "incomplete"
            print(
                f"  attempt {n_attempt}: FAILED ({reason}, seq={'+'.join(seq)}), discarding (len={T}) "
                f"[success rate: {n_saved}/{n_attempt} = {n_saved / n_attempt:.1%}]"
            )

    env.close_env()
    writer.finalize()

    print(f"\nSaved {n_saved} episodes, {writer.n_steps} steps -> {out_path}")
    print(
        "  arrays: {k: (shape, dtype)} = "
        + ", ".join(f"{k}:{writer.data[k].shape}/{writer.data[k].dtype}" for k in writer.data)
    )
    print(f"  success rate: {n_saved}/{n_attempt} attempts")
    if randomize:
        print(
            "  subtask coverage (episodes containing each): "
            + ", ".join(f"{k}:{subtask_coverage[k]}" for k in all_subtasks)
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subtask", type=str, default=None, help="single subtask, e.g. microwave")
    p.add_argument("--sequence", type=str, default=None, help="comma-separated subtask sequence")
    p.add_argument(
        "--randomize",
        action="store_true",
        help="sample a fresh random chain per episode (diverse coverage); ignores --subtask/--sequence",
    )
    p.add_argument("--chain-len", type=int, default=4, help="subtasks per chain when --randomize")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --randomize")
    # 581 matches the size of the human teleop dataset
    p.add_argument("--n-episodes", type=int, default=581, help="number of episodes to KEEP")
    p.add_argument(
        "--keep-failures", action="store_true", help="also record episodes where the sequence did not complete"
    )
    p.add_argument("--out", type=str, default=str(_HERE / "data" / "kitchen_demos_markovian_expert.zarr"))
    args = p.parse_args()

    if args.sequence:
        seq = [s.strip() for s in args.sequence.split(",") if s.strip()]
    elif args.subtask:
        seq = [args.subtask]
    else:
        seq = ["microwave"]

    if args.randomize:
        print(f"Recording {args.n_episodes} episodes of random {args.chain_len}-subtask chains (seed={args.seed})")
    else:
        print(f"Recording {args.n_episodes} episodes of sequence {seq}")
    record(
        seq,
        args.n_episodes,
        args.out,
        keep_failures=args.keep_failures,
        randomize=args.randomize,
        chain_len=args.chain_len,
        seed=args.seed,
    )
