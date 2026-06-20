"""Record Markovian-expert rollouts into a diffusion_policy ReplayBuffer zarr.

The output matches the provided `kitchen_demos.zarr` exactly so a policy can be
trained on expert data with the *same* obs/action space and low-level controller
as one trained on `kitchen_demos_multitask` -- the only difference is the data
source. Format (diffusion_policy ReplayBuffer, zarr v2):

    data/action  (N, 9)            float64   normalized joint-velocity action
    data/state   (N, 60)           float64   env obs = [qp(9), obj_qp(21), goal(30)]
    data/scene   (N, 240, 320, 3)  uint8     scene camera (camera_id 2)
    data/wrist   (N, 240, 320, 3)  uint8     wrist camera (camera_id 3)
    meta/episode_ends (E,)         int64     cumulative episode end indices

We record exactly the 9-D action handed to env.step (the output of
controller.pose_to_action) -- that is the env's native action, identical to the
representation parse_demos.py writes for the human demos -- so no IK conversion
is needed; the EEF->joint mapping already happens inside pose_to_action.

Per step we store (o_t, a_t): the obs/images seen *before* stepping paired with
the action taken, matching parse_demos.gather_training_data's convention.

Usage (needs the repo venv + EGL offscreen GL):
    MUJOCO_GL=egl LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia \
      env/bin/python experts/record_demos.py --subtask microwave --n-episodes 50 \
        --out experts/data/microwave_demos.zarr
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))                # controller, expert
sys.path.insert(0, str(_REPO / "adept_envs"))  # adept_envs

import gym  # noqa: E402
import adept_envs  # noqa: E402,F401  (registers kitchen envs)
import numcodecs  # noqa: E402
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

import controller  # noqa: E402
import expert  # noqa: E402

# Match kitchen_demos.zarr: blosc/lz4, clevel 5, byte shuffle, with the same
# per-array chunking. (Chunking only affects IO, not readability, but we mirror
# it so the datasets are byte-for-byte comparable in structure.)
_COMPRESSOR = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
_CHUNKS = {
    "action": (2048, 9),
    "state": (1024, 60),
    "scene": (128, 240, 320, 3),
    "wrist": (128, 240, 320, 3),
}
_IMG_W, _IMG_H = 320, 240


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

    last_key = None       # (idx, fsm_state) of the previous step
    left_keys = set()     # (idx, fsm_state) phases we have entered and then left
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
    }
    return episode, success, oscillated, osc_detail


def record(sequence, n_episodes, out_path, max_steps, keep_failures=False):
    env = gym.make("kitchen_relax-v1").unwrapped
    rb = ReplayBuffer.create_empty_numpy()

    n_saved, n_attempt = 0, 0
    while n_saved < n_episodes:
        n_attempt += 1
        episode, success, oscillated, osc_detail = _rollout(env, sequence, max_steps)
        T = episode["action"].shape[0]
        if success or keep_failures:
            rb.add_episode(episode)
            n_saved += 1
            tag = "ok" if success else ("FAIL-osc(kept)" if oscillated else "FAIL(kept)")
            print(f"  episode {n_saved}/{n_episodes} [{tag}] len={T} "
                  f"(attempt {n_attempt})")
        else:
            reason = f"OSCILLATION @ {osc_detail}" if oscillated else "incomplete"
            print(f"  attempt {n_attempt}: FAILED ({reason}), discarding (len={T})")

    env.close_env()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rb.save_to_path(str(out_path), chunks=_CHUNKS, compressors=_COMPRESSOR, if_exists="replace")
    print(f"\nSaved {n_saved} episodes, {rb.n_steps} steps -> {out_path}")
    print(f"  arrays: {{k: (shape, dtype)}} = "
          + ", ".join(f"{k}:{v.shape}/{v.dtype}" for k, v in rb.data.items()))
    print(f"  success rate: {n_saved}/{n_attempt} attempts")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subtask", type=str, default=None, help="single subtask, e.g. microwave")
    p.add_argument("--sequence", type=str, default=None, help="comma-separated subtask sequence")
    p.add_argument("--n-episodes", type=int, default=50, help="number of episodes to KEEP")
    p.add_argument("--max-steps", type=int, default=None,
                   help="max steps per episode (default: 200 * len(sequence))")
    p.add_argument("--keep-failures", action="store_true",
                   help="also record episodes where the sequence did not complete")
    p.add_argument("--out", type=str, default=str(_HERE / "data" / "expert_demos.zarr"))
    args = p.parse_args()

    if args.sequence:
        seq = [s.strip() for s in args.sequence.split(",") if s.strip()]
    elif args.subtask:
        seq = [args.subtask]
    else:
        seq = ["microwave"]

    max_steps = args.max_steps if args.max_steps is not None else 200 * len(seq)
    print(f"Recording {args.n_episodes} episodes of sequence {seq} (max_steps={max_steps})")
    record(seq, args.n_episodes, args.out, max_steps, keep_failures=args.keep_failures)
