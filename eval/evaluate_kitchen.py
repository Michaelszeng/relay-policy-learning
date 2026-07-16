"""
Evaluate a diffusion_policy checkpoint on the Franka Kitchen relay env.

Success rule: a trial counts as a success if at least --n-required-subtasks
distinct kitchen subtasks (out of the 7 D4RL franka_kitchen subtasks) are
achieved at any point during the rollout. Order doesn't matter.

Usage:
    python eval/evaluate_kitchen.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --n-rollouts 10
"""

import argparse
import collections
import csv
import datetime
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

# Headless compute nodes have no X11 DISPLAY, so dm_control's default GLFW
# backend fails to create a GL context ("gladLoadGL error"). Default to EGL
# (offscreen GPU rendering) unless the user picked a backend explicitly. Must
# run before any dm_control/mujoco import. Spawned workers inherit os.environ.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# adept_envs lives at <repo_root>/adept_envs/adept_envs. Insert <repo_root>/adept_envs
# on sys.path so `import adept_envs` works without the user exporting PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "adept_envs"))

# Reuse the recorder's subtask encoding so eval-time conditioning matches the
# dataset bit-for-bit. Load experts/subtasks/base.py directly (importlib) rather
# than `import experts.subtasks`: that package's __init__ pulls in every FSM
# module (and the mujoco controller), whereas base.py only needs numpy.
import importlib.util as _ilu  # noqa: E402

_base_spec = _ilu.spec_from_file_location(
    "_expert_subtask_base", _REPO_ROOT / "experts" / "subtasks" / "base.py"
)
_subtask_base = _ilu.module_from_spec(_base_spec)
_base_spec.loader.exec_module(_subtask_base)
SUBTASK_IDS = _subtask_base.SUBTASK_IDS
MAX_SEQUENCE_LEN = _subtask_base.MAX_SEQUENCE_LEN
sequence_onehot = _subtask_base.sequence_onehot
ALL_SUBTASKS = list(SUBTASK_IDS)

import dill
import gym
import hydra
import imageio
import numpy as np
import torch
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa
from omegaconf import OmegaConf

import adept_envs  # noqa: F401 — registers kitchen envs with gym

# Required for `${eval:...}` resolvers inside the saved cfg.
OmegaConf.register_new_resolver("eval", eval, replace=True)


# ------------------------------------------------------------------------- #
# Subtask success criteria — indices/goals/threshold match D4RL franka_kitchen.
# Indices are into env.sim.data.qpos[:30] (equivalently the first 30 entries
# of the observation returned by KitchenTaskRelaxV1._get_obs).
# ------------------------------------------------------------------------- #
SUBTASK_INFO = {
    "bottomknob": {"indices": np.array([11, 12]), "goal": np.array([-0.88, -0.01])},
    "topknob": {"indices": np.array([15, 16]), "goal": np.array([-0.92, -0.01])},
    "light": {"indices": np.array([17, 18]), "goal": np.array([-0.69, -0.05])},
    "slide": {"indices": np.array([19]), "goal": np.array([0.37])},
    "hinge": {"indices": np.array([20, 21]), "goal": np.array([0.0, 1.45])},
    "microwave": {"indices": np.array([22]), "goal": np.array([-0.75])},
    "kettle": {
        "indices": np.array([23, 24, 25, 26, 27, 28, 29]),
        "goal": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
    },
}
BONUS_THRESH = 0.3
DEFAULT_TASK_TIMEOUT = 280  # matches D4RL franka_kitchen episode horizon


# ------------------------------------------------------------------------- #
# Obs preprocessing (batched)
# ------------------------------------------------------------------------- #
def preprocess_obs(
    state_batch: np.ndarray,
    cams_batch: dict,
    device: torch.device,
    obs_keys: set,
    seq_onehot_batch: np.ndarray = None,
) -> dict:
    """Pack a batched kitchen timestep into the dict the policy expects.

    state_batch      : (N, state_dim) numpy
    cams_batch       : {cam_name: (N, H, W, 3) uint8}
    obs_keys         : keys present in cfg.shape_meta.obs
    seq_onehot_batch : (N, MAX_SEQUENCE_LEN * N_SUBTASKS) one-hot of each env's
                       fixed plan, or None. Constant over time within a trial, so
                       the same array is injected at every step.

    MuJoCo offscreen frames have negative row strides (OpenGL→image flip);
    np.ascontiguousarray materializes them so torch.from_numpy will accept them.
    """
    result = {}
    if "agent_pos" in obs_keys:
        # The policy consumes only the robot proprioceptive state (first 9 dims:
        # 7 arm joints + 2 gripper), not the full kitchen state vector.
        agent_pos = np.ascontiguousarray(state_batch[..., :9])
        result["agent_pos"] = torch.from_numpy(agent_pos).float().to(device)
    for cam_name, frames in cams_batch.items():
        if cam_name in obs_keys:
            result[cam_name] = torch.from_numpy(np.ascontiguousarray(frames)).float().to(device)
    if "subtask_sequence" in obs_keys and seq_onehot_batch is not None:
        result["subtask_sequence"] = (
            torch.from_numpy(np.ascontiguousarray(seq_onehot_batch)).float().to(device)
        )
    return result


# ------------------------------------------------------------------------- #
# Env vectorization
# ------------------------------------------------------------------------- #
def _bootstrap_and_make(env_id: str, robot_noise_ratio: float = None):
    """Re-create the kitchen env in any process (main or subprocess worker).

    robot_noise_ratio : if not None, override the env's simulated observation
                        (sensor) noise scale. None keeps the env default (0.1);
                        0.0 disables observation noise entirely.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "adept_envs"))
    import adept_envs  # noqa: F401 — registers envs

    import gym as _gym

    env = _gym.make(env_id).unwrapped
    if robot_noise_ratio is not None:
        env.robot_noise_ratio = robot_noise_ratio
    return env


def _pack(state, cams):
    return (
        np.ascontiguousarray(state),
        {k: np.ascontiguousarray(v) for k, v in cams.items()},
    )


def _worker(env_id: str, conn, robot_noise_ratio: float = None):
    """Subprocess worker: holds one kitchen env, services reset/step/close.

    Replies are tagged tuples: ("ok", state, cams) or ("err", msg).
    """
    env = _bootstrap_and_make(env_id, robot_noise_ratio=robot_noise_ratio)
    while True:
        cmd, payload = conn.recv()
        if cmd == "reset":
            state = env.reset()
            cams = env.render_cameras()
            s, c = _pack(state, cams)
            conn.send(("ok", s, c))
        elif cmd == "step":
            try:
                state, _, _, _ = env.step(payload)
                cams = env.render_cameras()
            except Exception as e:
                conn.send(("err", f"{type(e).__name__}: {e}"))
                continue
            s, c = _pack(state, cams)
            conn.send(("ok", s, c))
        elif cmd == "close":
            conn.close()
            return


class BatchedSingleEnv:
    """In-process batch-of-1 adapter (no subprocess overhead for n_envs=1)."""

    def __init__(self, env):
        self.env = env
        self.n_envs = 1

    def reset(self):
        s, c = _pack(self.env.reset(), self.env.render_cameras())
        return s[None], {k: v[None] for k, v in c.items()}

    def step(self, actions):
        s, _, _, _ = self.env.step(actions[0])
        s, c = _pack(s, self.env.render_cameras())
        return s[None], {k: v[None] for k, v in c.items()}

    def render(self):
        self.env.mj_render()

    def close(self):
        pass


class BatchedParallelEnvs:
    """Subprocess pool: one kitchen env per worker, obs/actions stacked in main."""

    def __init__(self, env_id: str, n_envs: int, robot_noise_ratio: float = None):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")  # CUDA-safe; fork would deadlock if torch is imported
        self.conns, self.procs = [], []
        for _ in range(n_envs):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(env_id, child, robot_noise_ratio), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self.n_envs = n_envs

    def _gather(self):
        results = [c.recv() for c in self.conns]
        # Tagged tuples: ("ok", state, cams) or ("err", msg).
        for r in results:
            if r[0] == "err":
                raise RuntimeError(f"worker env failed: {r[1]}")
        states = np.stack([r[1] for r in results])
        keys = list(results[0][2].keys())
        cams = {k: np.stack([r[2][k] for r in results]) for k in keys}
        return states, cams

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return self._gather()

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", a))
        return self._gather()

    def render(self):
        # Live MuJoCo viewer doesn't compose with subprocess workers — no-op.
        pass

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2)


def build_obs_dict(obs_deque: collections.deque, device: torch.device) -> dict:
    """Stack obs deque into {'obs': {k: (1, T_obs, ...)}} for policy.predict_action."""
    keys = obs_deque[0].keys()
    obs_stacked = {k: torch.stack([o[k] for o in obs_deque], dim=1) for k in keys}
    return {"obs": obs_stacked}


# ------------------------------------------------------------------------- #
# Policy loading
# ------------------------------------------------------------------------- #
def load_policy(checkpoint_path: str, device: torch.device):
    """
    Load a diffusion_policy workspace + policy from a .ckpt file.

    Also loads (or generates) the paired normalizer.pt that lives next to the
    checkpoint's parent checkpoints/ directory.
    """
    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    # Exclude optimizer/scheduler state: not needed for inference, and DataParallel
    # training can leave param-group sizes that don't match the unwrapped model.
    workspace.load_payload(payload, exclude_keys=["optimizer", "lr_scheduler"], include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model

    # --- normalizer ---
    ckpt_path = Path(checkpoint_path)
    normalizer_path = ckpt_path.parent.parent / "normalizer.pt"
    if normalizer_path.exists():
        print(f"Loading normalizer from {normalizer_path}")
        normalizer = torch.load(normalizer_path, map_location="cpu")
    else:
        print(f"Normalizer not found at {normalizer_path}, generating from dataset…")
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        torch.save(normalizer, normalizer_path)
        print(f"Saved normalizer to {normalizer_path}")

    policy.set_normalizer(normalizer)
    policy.to(device).eval()
    # Explicit eval mode for the encoder — R3M's ResNet18 uses BatchNorm which
    # behaves badly in train mode at inference batch sizes (n_obs_steps=2 typical).
    if hasattr(policy, "obs_encoder"):
        policy.obs_encoder.eval()
    return policy, cfg


# ------------------------------------------------------------------------- #
# IO helpers (identical to reference)
# ------------------------------------------------------------------------- #
def _write_mp4(frames: list, path: Path, fps: int = 10) -> None:
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_summary(n_success: int, n_total: int, trial_records: list, summary_path: Path) -> None:
    n_failure = sum(1 for r in trial_records if r["result"] == "failure")
    n_timeout = sum(1 for r in trial_records if r["result"] == "timeout")
    rate = n_success / n_total if n_total > 0 else 0.0
    avg_steps = sum(r["trial_time"] for r in trial_records) / len(trial_records) if trial_records else 0.0
    with open(summary_path, "w") as f:
        f.write(f"Trials completed : {n_total} / {args.n_rollouts}\n")
        f.write(f"Successes        : {n_success}\n")
        f.write(f"Failures         : {n_failure}\n")
        f.write(f"Timeouts         : {n_timeout}\n")
        f.write(f"Success rate     : {rate:.1%}\n")
        f.write(f"Avg trial steps  : {avg_steps:.1f}\n")


def _repair_csv_and_summary_from_pkl_data(pkl_path: Path):
    saved = pickle.load(open(pkl_path, "rb"))
    out_dir, fields = pkl_path.parent, ["trial", "result", "reward", "trial_time"]
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(saved["trials"])
    _write_summary(saved["n_success"], saved["n_total"], saved["trials"], out_dir / "summary.txt")
    csv_file = open(csv_path, "a", newline="")
    return csv_file, csv.DictWriter(csv_file, fieldnames=fields)


# ------------------------------------------------------------------------- #
# Rollout
# ------------------------------------------------------------------------- #
def _completed_subtasks(state_vec: np.ndarray) -> set:
    """Subset of the 7 kitchen subtasks whose joints are within BONUS_THRESH of goal."""
    out = set()
    for name, info in SUBTASK_INFO.items():
        if np.linalg.norm(state_vec[info["indices"]] - info["goal"]) < BONUS_THRESH:
            out.add(name)
    return out


def _append_video_frames(frame_buffers, cams_batch, mask):
    """For each env i where mask[i] is True, append a side-by-side frame."""
    cam_keys = list(cams_batch.keys())
    for i, keep in enumerate(mask):
        if not keep:
            continue
        side = np.concatenate([cams_batch[k][i] for k in cam_keys], axis=1).astype(np.uint8)
        frame_buffers[i].append(side)


@torch.no_grad()
def run_rollout(
    batched_env,
    policy,
    n_obs_steps: int,
    rollout_max_steps: int,
    device: torch.device,
    obs_keys: set,
    n_required_subtasks: int,
    headless: bool = True,
    record_video: bool = False,
    n_action_steps: int = None,
    seq_onehot_batch: np.ndarray = None,
) -> list:
    """
    Run one round of parallel trials. Returns one dict per env in the batch.

    Success rule: at least ``n_required_subtasks`` distinct subtasks (out of
    the 7 D4RL franka_kitchen subtasks) must each be achieved (within
    ``BONUS_THRESH`` of their goal) at some point during the rollout. Order
    doesn't matter.

    Per-env dict fields match the reference per-env-slot format:
        success, total_reward, result, steps, steps_per_env,
        frames (list of (H, W_total, 3) uint8) — only when record_video=True
    """
    n_envs = batched_env.n_envs

    state_batch, cams_batch = batched_env.reset()
    if not headless:
        batched_env.render()
    preprocessed = preprocess_obs(state_batch, cams_batch, device, obs_keys, seq_onehot_batch)
    obs_deque = collections.deque([preprocessed] * n_obs_steps, maxlen=n_obs_steps)
    action_queue: collections.deque = collections.deque()

    completed = [set() for _ in range(n_envs)]
    done = np.zeros(n_envs, dtype=bool)
    done_step = np.full(n_envs, -1, dtype=np.int64)
    frame_buffers = [[] for _ in range(n_envs)] if record_video else None
    step = 0

    if record_video:
        _append_video_frames(frame_buffers, cams_batch, np.ones(n_envs, dtype=bool))

    while not done.all() and step < rollout_max_steps:
        if len(action_queue) == 0:
            obs_dict = build_obs_dict(obs_deque, device)
            try:
                result = policy.predict_action(obs_dict, use_DDIM=True)
            except TypeError:
                # Older diffusion-policy variants without the use_DDIM kwarg.
                result = policy.predict_action(obs_dict)
            start = n_obs_steps - 1
            actions = result["action_pred"][:, start:]
            n_steps = n_action_steps if n_action_steps is not None else policy.n_action_steps
            for t in range(n_steps):
                action_queue.append(actions[:, t, :])

        action_batch = action_queue.popleft().cpu().numpy()
        try:
            state_batch, cams_batch = batched_env.step(action_batch)
        except Exception as e:
            # Sim blow-up: end the round; envs still undone count as failures.
            print(
                f"[run_rollout] step raised {type(e).__name__} at step {step}: {e}. "
                "Marking unfinished envs as failures and ending rollout."
            )
            for i in range(n_envs):
                done[i] = True
            break

        if not headless:
            batched_env.render()

        for i in range(n_envs):
            if done[i]:
                continue
            completed[i] |= _completed_subtasks(state_batch[i])
            if len(completed[i]) >= n_required_subtasks:
                done[i] = True
                done_step[i] = step

        preprocessed = preprocess_obs(state_batch, cams_batch, device, obs_keys, seq_onehot_batch)
        obs_deque.append(preprocessed)

        if record_video:
            # Keep recording all envs (even after they finish) until rollout ends —
            # mirrors the reference, which lets per-env videos run to round length.
            _append_video_frames(frame_buffers, cams_batch, np.ones(n_envs, dtype=bool))

        step += 1

    out = []
    for i in range(n_envs):
        n_completed = len(completed[i])
        success = n_completed >= n_required_subtasks
        if success:
            result_str = "success"
        elif done_step[i] == -1:
            result_str = "timeout"
        else:
            result_str = "failure"
        steps_per_env_val = int(done_step[i] + 1) if done_step[i] >= 0 else step
        d = {
            "success": bool(success),
            "total_reward": float(n_completed),
            "result": result_str,
            "steps": step,
            "steps_per_env": steps_per_env_val,
        }
        if record_video:
            d["frames"] = frame_buffers[i]
        out.append(d)
    return out


# ------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a diffusion_policy checkpoint on Franka Kitchen.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--n-required-subtasks",
        type=int,
        default=4,
        help=(
            "Number of distinct subtasks (of the 7 D4RL franka_kitchen subtasks: "
            f"{', '.join(SUBTASK_INFO.keys())}) required for a trial to count "
            "as a success (default: 4)."
        ),
    )
    parser.add_argument("--env", type=str, default="kitchen_relax-v1", help="Gym env id (default: kitchen_relax-v1).")
    parser.add_argument(
        "--noise",
        choices=["on", "off"],
        default="on",
        help=(
            "Whether to inject the env's simulated observation (sensor) noise during "
            "rollouts. 'on' keeps the env default (robot_noise_ratio=0.1, matching data "
            "collection); 'off' sets robot_noise_ratio=0.0 for clean observations "
            "(default: on)."
        ),
    )
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Trials per round. Kitchen env is single-instance, so n_envs>1 runs serially.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help=(
            "Run without the onscreen MuJoCo viewer (default: show a viewer "
            "window that updates each step). Camera rendering for the policy is "
            "offscreen either way, so --headless is purely about the live window."
        ),
    )
    parser.add_argument(
        "--n-video-trials",
        type=int,
        default=20,
        help=(
            "Save mp4s for the first N trials (default: 20). Set to -1 to save all, "
            "or 0 to disable mp4 output entirely. Ignored when --record-failures is set."
        ),
    )
    parser.add_argument(
        "--record-failures",
        action="store_true",
        default=False,
        help="If set, save mp4s of all failed trials (overrides --n-video-trials).",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override action horizon (default: use value from checkpoint config)",
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=None,
        help=f"Max rollout steps per trial (default: {DEFAULT_TASK_TIMEOUT}).",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help=(
            "Comma-separated subtask plan to feed the policy as the (constant) "
            "subtask_sequence conditioning, e.g. 'microwave,kettle,slide,hinge'. "
            "If omitted, a fresh random chain is sampled per trial (matching "
            "record_demos.py --randomize). Only used when the checkpoint's "
            "shape_meta.obs contains subtask_sequence."
        ),
    )
    parser.add_argument(
        "--chain-len",
        type=int,
        default=MAX_SEQUENCE_LEN,
        help=f"Length of randomly-sampled plans when --sequence is omitted (default: {MAX_SEQUENCE_LEN}).",
    )
    parser.add_argument(
        "--seq-seed",
        type=int,
        default=0,
        help="Seed for the per-trial random plan sampler (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write results (default: outputs/<date>/<time>)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from an existing results.pkl in --output-dir (requires --output-dir)",
    )
    args = parser.parse_args()

    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir to be specified")

    if args.n_required_subtasks < 1 or args.n_required_subtasks > len(SUBTASK_INFO):
        parser.error(f"--n-required-subtasks={args.n_required_subtasks} must be in [1, {len(SUBTASK_INFO)}].")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"No CUDA GPUs are available (device={args.device}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})."
        )

    print(f"Loading policy from {args.checkpoint}")
    policy, cfg = load_policy(args.checkpoint, device)
    n_obs_steps: int = int(cfg.n_obs_steps)
    n_action_steps: int = args.n_action_steps  # None means use the value from the checkpoint config
    print(
        f"n_obs_steps={n_obs_steps}, "
        f"n_action_steps={'from_cfg' if n_action_steps is None else n_action_steps}, "
        f"n_required_subtasks={args.n_required_subtasks}/{len(SUBTASK_INFO)}, "
        f"n_envs={args.n_envs}"
    )

    policy_obs_keys = set(cfg.shape_meta.obs.keys())
    is_image_based = any(k in policy_obs_keys for k in ("scene", "wrist"))
    print(f"Policy type: {'image-based' if is_image_based else 'state-based'} (obs keys: {sorted(policy_obs_keys)})")

    # --- subtask conditioning ---------------------------------------------- #
    # current_subtask conditioning would require re-running the training oracle
    # (detect is_done + at_home, advance a per-env latch) on the policy's own
    # trajectory, which is fragile (the policy may never cleanly reach the home
    # pose). We only support the constant subtask_sequence conditioning here.
    if "current_subtask" in policy_obs_keys:
        raise NotImplementedError(
            "This checkpoint conditions on `current_subtask`, which is not supported "
            "at eval time: supplying it would require re-implementing the training-time "
            "oracle (is_done + at_home advancement) on the policy's own rollout. Retrain "
            "with subtask_sequence-only conditioning, or implement the eval oracle."
        )
    needs_sequence = "subtask_sequence" in policy_obs_keys

    fixed_sequence = None
    if args.sequence is not None:
        fixed_sequence = [s.strip() for s in args.sequence.split(",") if s.strip()]
        unknown = [s for s in fixed_sequence if s not in SUBTASK_IDS]
        if unknown:
            parser.error(f"--sequence has unknown subtask(s) {unknown}; valid: {ALL_SUBTASKS}")
        if len(fixed_sequence) > MAX_SEQUENCE_LEN:
            parser.error(f"--sequence has {len(fixed_sequence)} subtasks; max is MAX_SEQUENCE_LEN={MAX_SEQUENCE_LEN}.")

    if fixed_sequence is None and not 1 <= args.chain_len <= min(MAX_SEQUENCE_LEN, len(ALL_SUBTASKS)):
        parser.error(f"--chain-len={args.chain_len} must be in [1, {min(MAX_SEQUENCE_LEN, len(ALL_SUBTASKS))}].")

    seq_rng = random.Random(args.seq_seed)

    def make_sequences(n: int) -> list:
        """One subtask plan per env for a round: fixed if --sequence else random."""
        if fixed_sequence is not None:
            return [list(fixed_sequence) for _ in range(n)]
        return [seq_rng.sample(ALL_SUBTASKS, args.chain_len) for _ in range(n)]

    if needs_sequence:
        mode = f"fixed {fixed_sequence}" if fixed_sequence is not None else f"random chain-len={args.chain_len}"
        print(f"subtask_sequence conditioning: {mode} (seq-seed={args.seq_seed})")
    elif args.sequence is not None:
        print("[warn] --sequence given but checkpoint has no subtask_sequence obs key; ignoring.")

    if args.task_timeout is not None:
        rollout_max_steps = args.task_timeout
    else:
        rollout_max_steps = DEFAULT_TASK_TIMEOUT
    np.random.seed(43)
    # None keeps the env default (0.1); 0.0 disables observation noise entirely.
    robot_noise_ratio = None if args.noise == "on" else 0.0
    print(f"Observation noise: on (env default robot_noise_ratio=0.1)" if args.noise == "on" else "off (robot_noise_ratio=0.0)")
    if args.n_envs == 1:
        print(f"Creating env ({args.env}, max_steps={rollout_max_steps})")
        batched_env = BatchedSingleEnv(_bootstrap_and_make(args.env, robot_noise_ratio=robot_noise_ratio))
    else:
        if not args.headless:
            print(
                f"[main] n_envs={args.n_envs}>1: disabling live viewer "
                "(MuJoCo onscreen viewer doesn't compose with subprocess workers)."
            )
            args.headless = True
        print(f"Creating {args.n_envs} parallel envs ({args.env}, max_steps={rollout_max_steps})")
        batched_env = BatchedParallelEnvs(args.env, args.n_envs, robot_noise_ratio=robot_noise_ratio)

    # --- output directory ---
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        now = datetime.datetime.now()
        out_dir = Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve the video budget once. n_video_trials == 0 disables mp4s entirely
    # (unless --record-failures, which always saves failures).
    video_budget = args.n_video_trials if args.n_video_trials >= 0 else args.n_rollouts
    videos_enabled = args.record_failures or video_budget > 0
    videos_dir = out_dir / "videos"
    if videos_enabled:
        videos_dir.mkdir(parents=True, exist_ok=True)

    # --- state initialization (fresh or resumed) ---
    n_success = 0
    n_total = 0
    all_trial_records = []
    resuming = False

    if args.resume:
        pkl_path = out_dir / "results.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("n_total", 0) > 0:
                n_success = saved["n_success"]
                n_total = saved["n_total"]
                all_trial_records = saved["trials"]
                resuming = True
            else:
                print(f"Found results.pkl at {pkl_path} but no completed trials; starting fresh.")
        else:
            print(f"--resume set but no results.pkl found in {out_dir}; starting fresh.")

    n_rounds = max(1, math.ceil(args.n_rollouts / args.n_envs))
    i_start = math.ceil(n_total / args.n_envs)

    csv_path = out_dir / "results.csv"
    csv_fields = ["trial", "result", "reward", "trial_time"]
    summary_path = out_dir / "summary.txt"

    if resuming:
        print(
            f"Resuming: {n_total}/{args.n_rollouts} trials done"
            f" ({n_success} successes, {n_success / n_total:.1%});"
            f" starting from round {i_start + 1}/{n_rounds}"
        )
        csv_file, csv_writer = _repair_csv_and_summary_from_pkl_data(pkl_path)
    else:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()
        csv_file.flush()

    for i in range(i_start, n_rounds):
        t_start = time.time()
        if args.record_failures:
            record_this_round = True
        else:
            record_this_round = n_total < video_budget

        # Per-trial subtask plans (one per env in the batch). Constant within a
        # trial, so we encode once and feed the same one-hot at every step.
        seq_onehot_batch = None
        if needs_sequence:
            round_sequences = make_sequences(batched_env.n_envs)
            seq_onehot_batch = np.stack([sequence_onehot(s) for s in round_sequences])

        # One batched rollout per round: returns n_envs per-trial dicts.
        round_results = run_rollout(
            batched_env=batched_env,
            policy=policy,
            n_obs_steps=n_obs_steps,
            rollout_max_steps=rollout_max_steps,
            device=device,
            obs_keys=policy_obs_keys,
            n_required_subtasks=args.n_required_subtasks,
            headless=args.headless,
            record_video=record_this_round,
            n_action_steps=n_action_steps,
            seq_onehot_batch=seq_onehot_batch,
        )
        rollout_time = time.time() - t_start

        for env_idx, rr in enumerate(round_results):
            if n_total >= args.n_rollouts:
                break
            trial_num = n_total + 1
            result_str = rr["result"]
            record = {
                "trial": trial_num,
                "result": result_str,
                "reward": float(rr["total_reward"]),
                "trial_time": int(rr["steps_per_env"]),
            }
            all_trial_records.append(record)
            n_success += int(rr["success"])
            n_total += 1

            csv_writer.writerow(record)
            csv_file.flush()

            if record_this_round and "frames" in rr:
                save_this_video = False
                if args.record_failures:
                    if result_str != "success":
                        save_this_video = True
                elif trial_num <= video_budget:
                    save_this_video = True
                if save_this_video:
                    video_path = videos_dir / f"trial_{trial_num:04d}_{result_str}.mp4"
                    _write_mp4(rr["frames"], video_path)
                    print(f"  Saved video: {video_path.name}")

        _write_summary(n_success, n_total, all_trial_records, summary_path)

        pkl_path = out_dir / "results.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "trials": all_trial_records,
                    "n_success": n_success,
                    "n_total": n_total,
                    "success_rate": n_success / n_total if n_total > 0 else 0.0,
                    "checkpoint": args.checkpoint,
                    "n_required_subtasks": args.n_required_subtasks,
                    "env": args.env,
                    "n_obs_steps": n_obs_steps,
                    "rollout_max_steps": rollout_max_steps,
                },
                f,
            )

        success_rate = n_success / n_total if n_total > 0 else 0.0
        video_tag = "video=on" if record_this_round else "video=off"
        total_time = time.time() - t_start
        round_results_str = [r["result"] for r in round_results]
        print(
            f"Round {i + 1}/{n_rounds} [{video_tag}]: "
            f"rollout time={rollout_time:.1f}s, total time={total_time:.1f}s, "
            f"result={round_results_str}  running {n_success}/{n_total} ({success_rate:.1%})"
        )

    csv_file.close()
    batched_env.close()

    final_success_rate = n_success / n_total if n_total > 0 else 0.0
    print(f"\nFinal success rate: {n_success}/{n_total} ({final_success_rate:.1%})")
    print(f"Results written to {out_dir}/")
