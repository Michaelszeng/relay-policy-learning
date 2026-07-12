"""Render onion-skin motion overlay figures from recorded rollout zarrs.

Loads a rollout zarr (recorded by eval/evaluate_kitchen.py --record-zarr, or any
zarr in that layout that has data/qpos + data/qvel), resets a kitchen env to the
exact recorded sim state at each requested timestep of one episode, renders each
state from a fixed high-resolution camera, and alpha-composites the moving
pixels (the Franka arm + every movable kitchen object: kettle, doors, knobs,
slide, ...) into a single image where older poses are more translucent than
recent ones.

The overlay is done in image space: each pose is rendered separately along with
a segmentation mask of the moving bodies (static kitchen = background), and the
masked pixels are blended oldest-first with increasing opacity. The final pose
is drawn fully opaque on top.

Workflow (first run -- interactive camera framing):
    python eval/motion_overlay.py rollouts.zarr --episode 0 \
        --timesteps 0 40 80 120 -1 --alpha-min 0.2 --alpha-max 0.8

    A viewer window opens showing the last requested pose. Orbit/pan/zoom with
    the mouse to frame the shot, press N to cycle through the selected poses to
    check framing, then press C to capture the view and render. The camera pose
    is saved to <out>.camera.json so the shot can be re-rendered without the GUI.

Re-render headless with a saved camera (e.g. different timesteps/alphas):
    python eval/motion_overlay.py rollouts.zarr \
        --timesteps 0 100 200 -1 --camera-pose rollouts_ep0_overlay.camera.json

Or skip the GUI entirely and use the env's default scene framing:
    python eval/motion_overlay.py rollouts.zarr --timesteps 0 100 -1 --no-gui

The camera JSON uses the MuJoCo free-camera form and may also be hand-written:
    {"lookat": [x, y, z], "distance": 4.5, "azimuth": -66.0, "elevation": -65.0}
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def composite(colors, masks, alphas):
    """Blend older poses (per-frame boolean masks) onto the final pose's render.

    Ghosts are drawn oldest-first so newer poses occlude older ones; the final
    pose's masked pixels are re-pasted fully opaque on top so ghosts never
    bleed over it.
    """
    canvas = colors[-1].astype(np.float32)
    for color, mask, alpha in zip(colors[:-1], masks[:-1], alphas):
        mask = mask[..., None]
        canvas = np.where(mask, alpha * color + (1 - alpha) * canvas, canvas)
    final_mask = masks[-1][..., None]
    canvas = np.where(final_mask, colors[-1].astype(np.float32), canvas)
    return canvas.round().astype(np.uint8)


def moving_body_flags(model):
    """Bool array over bodies: True iff the body can move.

    A body moves iff there is at least one DOF somewhere in its kinematic chain
    (itself or an ancestor). This picks out the Franka arm and every articulated
    kitchen object, and leaves the welded/static kitchen as background.
    """
    moving = np.zeros(model.nbody, dtype=bool)
    for b in range(1, model.nbody):
        moving[b] = bool(model.body_dofnum[b] > 0) or moving[model.body_parentid[b]]
    return moving


def subtree_flags(model, body_ids):
    """Bool array over bodies: True for each body in body_ids or its descendants."""
    flags = np.zeros(model.nbody, dtype=bool)
    flags[list(body_ids)] = True
    for b in range(1, model.nbody):  # parentid < b, so one forward pass suffices
        flags[b] = flags[b] or flags[model.body_parentid[b]]
    return flags


def set_sim_state(env, qpos, qvel):
    """Kinematic reset to a recorded snapshot (forward only -- nothing integrates,
    so the rendered pose is exactly the recorded one)."""
    env.sim.data.qpos[:] = np.asarray(qpos)
    env.sim.data.qvel[:] = np.asarray(qvel)
    env.sim.forward()


def make_overlay_camera(env, dm_mujoco, width, height):
    """Create a free camera at the requested resolution.

    The model's offscreen framebuffer defaults to a small size; grow it before
    creating the camera or dm_control refuses to render at high resolution.
    """
    vis_global = env.sim.model.vis.global_
    vis_global.offwidth = max(vis_global.offwidth, width)
    vis_global.offheight = max(vis_global.offheight, height)
    return dm_mujoco.Camera(physics=env.sim, height=height, width=width, camera_id=-1)


def apply_camera_json(camera, cam_json):
    """Point the free camera using a saved/hand-written camera JSON."""
    required = ("lookat", "distance", "azimuth", "elevation")
    missing = [k for k in required if k not in cam_json]
    if missing:
        raise ValueError(f"Camera JSON missing key(s) {missing}; required: {list(required)}")
    render_cam = camera._render_camera  # MjvCamera; same access as DMRenderer._update_camera
    render_cam.lookat[:] = cam_json["lookat"]
    render_cam.distance = float(cam_json["distance"])
    render_cam.azimuth = float(cam_json["azimuth"])
    render_cam.elevation = float(cam_json["elevation"])


def render_state(camera, geom_ghosted, geom_type_id):
    """Render color + moving-body mask for the sim's current state.

    The segmentation render returns (H, W, 2) int32: [object id, object type],
    with -1/-1 for background. Mask = geom pixels whose geom belongs to a moving
    (and not excluded) body.
    """
    color = np.ascontiguousarray(camera.render())
    seg = camera.render(segmentation=True)
    ids, types = seg[..., 0], seg[..., 1]
    mask = (types == geom_type_id) & geom_ghosted[np.clip(ids, 0, len(geom_ghosted) - 1)]
    return color, mask


def pick_camera_interactively(env, module, states, timesteps):
    """Run the viewer loop until the user frames the shot and presses C.

    Returns a camera JSON dict (free-camera form). N cycles through the
    selected poses so the whole motion can be checked against the framing.
    """
    renderer = env.sim_robot.renderer
    pose_i = len(states) - 1
    set_sim_state(env, *states[pose_i])
    renderer.render_to_window()  # first call creates the window

    window = renderer._window
    glfw = module.get_dm_viewer().gui.glfw_gui.glfw
    glfw_window = window._window._context.window

    print(
        "\n=== Interactive camera framing ===\n"
        "  Orbit/pan/zoom with the mouse (viewer window must have focus).\n"
        "  [N]  cycle through the selected timesteps to check framing\n"
        f"       (showing timestep {timesteps[pose_i]})\n"
        "  [C]  capture this view and render the overlay\n"
        "  Close the window to abort.\n"
    )

    was_down = {glfw.KEY_C: True, glfw.KEY_N: True}  # True: ignore keys held at start
    while True:
        if glfw.window_should_close(glfw_window):
            print("Viewer closed -- aborting without rendering.")
            sys.exit(1)
        window.run_frame()

        for key in (glfw.KEY_C, glfw.KEY_N):
            down = glfw.get_key(glfw_window, key) == glfw.PRESS
            pressed, was_down[key] = down and not was_down[key], down
            if not pressed:
                continue
            if key == glfw.KEY_C:
                print("View captured.")
                cam = window.camera  # the viewer's MjvCamera
                return {
                    "lookat": [float(v) for v in cam.lookat],
                    "distance": float(cam.distance),
                    "azimuth": float(cam.azimuth),
                    "elevation": float(cam.elevation),
                }
            pose_i = (pose_i + 1) % len(states)
            set_sim_state(env, *states[pose_i])
            print(f"Showing timestep {timesteps[pose_i]}")


def zarr_stem(zarr_path: Path):
    name = zarr_path.name
    return name[: -len(".zarr")] if name.endswith(".zarr") else name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zarr", type=Path, help="Rollout zarr with data/qpos + data/qvel (--record-zarr output)")
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        required=True,
        help="Episode-relative timesteps to overlay (negative indices count from the end, e.g. -1 is the last)",
    )
    parser.add_argument(
        "--episode", type=int, default=0, help="Episode index within the zarr (negative counts from the end)"
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <zarr-stem>_ep<E>_overlay.png)")
    parser.add_argument(
        "--camera-pose",
        type=Path,
        default=None,
        help="Camera JSON from a previous run; if given, renders headless without the GUI",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=False,
        help="Skip interactive framing and render headless with the env's default scene camera",
    )
    parser.add_argument("--width", type=int, default=1920, help="Render width")
    parser.add_argument("--height", type=int, default=1080, help="Render height")
    parser.add_argument("--alpha-min", type=float, default=0.25, help="Opacity of the oldest ghost pose")
    parser.add_argument(
        "--alpha-max", type=float, default=0.7, help="Opacity of the newest ghost pose (final pose is always opaque)"
    )
    parser.add_argument(
        "--exclude-bodies",
        type=str,
        nargs="*",
        default=[],
        help="Body names (with their subtrees) to exclude from the ghost/overlay masks, "
        "e.g. a door that would cover older arm poses. See --list-bodies for names.",
    )
    parser.add_argument(
        "--list-bodies", action="store_true", help="Print the moving (maskable) body names and exit"
    )
    parser.add_argument("--save-frames", action="store_true", help="Also save per-timestep color/mask PNGs")
    parser.add_argument("--env", type=str, default="kitchen_relax-v1", help="Gym env id")
    args = parser.parse_args()

    # GL backend must be decided before dm_control is imported: EGL for headless
    # offscreen rendering, GLFW (the default) when the interactive viewer is needed.
    if args.camera_pose is not None or args.no_gui or args.list_bodies:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adept_envs"))
    import gym
    import imageio
    import zarr

    import adept_envs  # noqa: F401 -- registers kitchen envs
    from adept_envs.simulation import module

    dm_mujoco = module.get_dm_mujoco()
    geom_type_id = int(dm_mujoco.wrapper.mjbindings.enums.mjtObj.mjOBJ_GEOM)

    # --- load the requested episode's sim states --------------------------- #
    root = zarr.open_group(str(args.zarr), mode="r")
    if "qpos" not in root["data"] or "qvel" not in root["data"]:
        raise ValueError(
            f"{args.zarr} has no data/qpos + data/qvel arrays -- record it with "
            "evaluate_kitchen.py --record-zarr (the expert dataset doesn't store sim state)."
        )
    ends = root["meta/episode_ends"][:]
    n_eps = len(ends)
    episode = args.episode if args.episode >= 0 else n_eps + args.episode
    if not 0 <= episode < n_eps:
        raise ValueError(f"Episode {args.episode} out of range for zarr with {n_eps} episodes")
    ep_start = int(ends[episode - 1]) if episode > 0 else 0
    n_obs = int(ends[episode]) - ep_start

    timesteps = [t if t >= 0 else n_obs + t for t in args.timesteps]
    for t in timesteps:
        if not 0 <= t < n_obs:
            raise ValueError(f"Timestep {t} out of range for episode {episode} of length {n_obs}")
    timesteps = sorted(set(timesteps))
    states = [(root["data/qpos"][ep_start + t], root["data/qvel"][ep_start + t]) for t in timesteps]
    print(f"Loaded {args.zarr} episode {episode} ({n_obs} timesteps); rendering timesteps {timesteps}")

    env = gym.make(args.env).unwrapped
    model = env.sim.model

    # --- moving-body segmentation masks ------------------------------------ #
    moving = moving_body_flags(model)
    body_names = {model.id2name(b, "body") or f"<unnamed body {b}>": b for b in range(model.nbody)}

    if args.list_bodies:
        print("Moving (maskable) bodies:")
        for name, b in sorted(body_names.items()):
            if moving[b]:
                print(f"  {name}")
        return

    unknown = [name for name in args.exclude_bodies if name not in body_names]
    if unknown:
        raise ValueError(f"Unknown body name(s) {unknown}; see --list-bodies for available names")
    excluded = subtree_flags(model, [body_names[n] for n in args.exclude_bodies])
    if args.exclude_bodies:
        print(f"Excluding bodies (and their subtrees) from overlay masks: {args.exclude_bodies}")
    geom_ghosted = (moving & ~excluded)[np.asarray(model.geom_bodyid)]

    out_path = args.out or Path(f"{zarr_stem(args.zarr)}_ep{episode}_overlay.png")
    camera_json_path = out_path.with_suffix(".camera.json")

    # --- camera ------------------------------------------------------------- #
    if args.camera_pose is not None:
        with open(args.camera_pose) as f:
            cam_json = json.load(f)
    elif args.no_gui:
        # The env's default scene framing (what the live eval viewer opens with).
        cam_json = dict(env.sim_robot.renderer._camera_settings)
        cam_json["lookat"] = [float(v) for v in cam_json["lookat"]]
        print(f"Using default scene camera: {cam_json}")
    else:
        cam_json = pick_camera_interactively(env, module, states, timesteps)
        with open(camera_json_path, "w") as f:
            json.dump(cam_json, f, indent=2)
        print(f"Camera pose saved to {camera_json_path} (reuse with --camera-pose {camera_json_path})")

    camera = make_overlay_camera(env, dm_mujoco, args.width, args.height)
    apply_camera_json(camera, cam_json)

    # --- render + composite -------------------------------------------------- #
    colors, masks = [], []
    for t, (qpos, qvel) in zip(timesteps, states):
        set_sim_state(env, qpos, qvel)
        color, mask = render_state(camera, geom_ghosted, geom_type_id)
        colors.append(color)
        masks.append(mask)
        print(f"Rendered timestep {t} (mask covers {mask.mean():.1%} of the image)")
    camera._scene.free()

    if args.save_frames:
        frames_dir = out_path.parent / f"{out_path.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for t, color, mask in zip(timesteps, colors, masks):
            imageio.imwrite(frames_dir / f"t{t:05d}_color.png", color)
            imageio.imwrite(frames_dir / f"t{t:05d}_mask.png", (mask * 255).astype(np.uint8))
        print(f"Per-frame renders saved to {frames_dir}/")

    n_ghosts = len(states) - 1
    alphas = np.linspace(args.alpha_min, args.alpha_max, n_ghosts) if n_ghosts > 0 else []
    overlay = composite(colors, masks, alphas)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, overlay)
    print(f"Overlay saved to {out_path}")


if __name__ == "__main__":
    main()
