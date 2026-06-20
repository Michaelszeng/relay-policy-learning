"""Run / test the Markovian scripted expert, one subtask at a time.

The runner plays the role of the *environment oracle*: it holds the current
subtask index and advances it to the next subtask only once the current one is
complete AND the arm has returned to the fixed home pose. The expert itself
only ever sees the current subtask label (Markovian).

Examples:
    # test a single subtask
    python experts/run_expert.py --subtask microwave

    # run a 4-subtask sequence
    python experts/run_expert.py --sequence microwave,kettle,slide,hinge

Requires (see README): the repo's Python 3.9 venv, MUJOCO_GL set for offscreen
rendering, e.g.
    MUJOCO_GL=egl LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia \
      env/bin/python experts/run_expert.py --subtask microwave
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))  # controller, expert
sys.path.insert(0, str(_REPO / "adept_envs"))  # adept_envs

import controller  # noqa: E402
import expert  # noqa: E402
import gym  # noqa: E402

import adept_envs  # noqa: E402,F401  (registers kitchen envs)


def _save_video(frames, path, fps=10):
    import imageio

    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(f)
    print(f"  saved video: {path}")


def run(sequence, max_steps, record, out_dir, render=False, verbose=True):
    env = gym.make("kitchen_relax-v1").unwrapped
    env.reset()

    # Point the --render free camera at the kitchen. The env's default free-cam
    # settings (kitchen_multitask_v0.py: distance=4.5, azimuth=-66, elevation=-65)
    # stare down behind the kitchen, so the window shows nothing useful. The
    # renderer reads these once, when the viewer is first created, so override
    # them here before the first render (a front-left view, cf. the XML scene_cam).
    if render:
        env.sim_robot.renderer._camera_settings = dict(
            distance=2.8,
            azimuth=90.0,
            elevation=-25.0,
            lookat=np.array([-0.3, 0.4, 2.0]),
        )

    try:
        idx = 0  # oracle's current subtask index
        last_state = None
        frames = []
        transitions = []
        done_steps = {}  # subtask -> step it was first completed

        for step in range(max_steps):
            if idx >= len(sequence):
                break
            label = sequence[idx]

            subtask, fsm_state = expert.compute_state(env, label)
            gripper, tpos, trot = expert.compute_action(env, subtask, fsm_state)
            if fsm_state == "return-to-home":
                action = controller.joint_pose_to_action(env, expert.RESET_ARM_QPOS, gripper)
            else:
                action = controller.pose_to_action(env, tpos, trot, gripper)
            env.step(action)

            if render:
                env.render(mode="human")

            if verbose and fsm_state != last_state:
                ang = float(env.sim.data.qpos[22])
                if fsm_state == "return-to-home":
                    d = float(np.linalg.norm(env.sim.data.qpos[:7] - expert.RESET_ARM_QPOS))
                else:
                    d = float(np.linalg.norm(controller.ee_pos(env) - tpos))
                print(f"  step {step:4d}  [{label}] -> {fsm_state:18s} micro_angle={ang:+.3f}  dist_to_target={d:.3f}")
                transitions.append((step, label, fsm_state))
                last_state = fsm_state

            if expert.is_done(env, label) and label not in done_steps:
                done_steps[label] = step
                if verbose:
                    print(f"  step {step:4d}  *** {label} COMPLETE ***")

            # Oracle: advance only after completion AND return to home pose.
            if expert.is_done(env, label) and expert.at_home(env):
                if verbose:
                    print(f"  step {step:4d}  {label} done + home -> advancing")
                idx += 1
                last_state = None

            if record:
                cams = env.render_cameras()
                frame = np.concatenate([cams["scene"], cams["wrist"]], axis=1).astype(np.uint8)
                frames.append(frame)

        # report
        print("\n=== summary ===")
        for st in sequence:
            ok = expert.is_done(env, st)
            when = done_steps.get(st, None)
            print(
                f"  {st:12s}: {'DONE' if ok else 'FAIL'}"
                + (f" (first complete @ step {when})" if when is not None else "")
            )

        if record and frames:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            _save_video(frames, out_dir / ("_".join(sequence) + ".mp4"))

    finally:
        env.close_env()
        env.close()

    return done_steps


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subtask", type=str, default=None, help="run a single subtask (e.g. microwave)")
    p.add_argument("--sequence", type=str, default=None, help="comma-separated subtask sequence")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--render", action="store_true", help="render to a window (non-headless)")
    p.add_argument("--out", type=str, default=str(_HERE / "out"))
    args = p.parse_args()

    if args.sequence:
        seq = [s.strip() for s in args.sequence.split(",") if s.strip()]
    elif args.subtask:
        seq = [args.subtask]
    else:
        seq = ["microwave"]

    print(f"Running expert on sequence: {seq}")
    run(seq, args.max_steps, record=not args.no_video, out_dir=args.out, render=args.render)
