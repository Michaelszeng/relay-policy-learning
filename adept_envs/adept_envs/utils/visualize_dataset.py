"""Interactively view episodes from a processed zarr dataset.

Expects the zarr layout produced by `process_pickles.py`:
    /data/<key>            (T_total, ...)
    /meta/episode_ends     (N_episodes,)

Camera arrays (uint8, ndim >= 3 per timestep) are auto-detected and stitched
side-by-side. The remaining 1D-per-timestep arrays can be shown as time-series
plots in a panel below the cameras via --state.
"""

import argparse
import io
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # off-screen rendering; must be set before pyplot import
import matplotlib.pyplot as plt

import zarr


def _list_keys(group):
    return list(group.array_keys())


def _is_camera_arr(arr):
    return arr.dtype == np.uint8 and arr.ndim >= 4


def render_state_panel(
    series: dict,        # {name: (T, D) ndarray}
    frame_idx: int,
    panel_h: int = 480,
    panel_w: int = 1280,
) -> np.ndarray:
    """Render a matplotlib panel with a line plot per series; returns HxWx3 uint8 RGB."""
    n = len(series)
    fig = plt.figure(figsize=(panel_w / 100, panel_h / 100), dpi=100)
    fig.patch.set_facecolor("#1a1a1a")

    for i, (name, arr) in enumerate(series.items()):
        ax = fig.add_subplot(n, 1, i + 1)
        T, D = arr.shape
        t_axis = np.arange(T)
        cmap = plt.get_cmap("tab20" if D > 10 else "tab10")
        for d in range(D):
            ax.plot(t_axis, arr[:, d], color=cmap(d % cmap.N), linewidth=0.8)
        ax.axvline(frame_idx, color="yellow", linewidth=1.0, zorder=5)
        ax.set_xlim(0, max(T - 1, 1))
        ax.set_ylabel(name, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_facecolor("#1a1a1a")
        ax.grid(color="gray", linewidth=0.3)
        for spine in ax.spines.values():
            spine.set_color("gray")
        ax.title.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.tick_params(colors="white")

    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#1a1a1a", dpi=100)
    plt.close(fig)
    buf.seek(0)

    raw = np.frombuffer(buf.getvalue(), np.uint8)
    panel_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    if panel_rgb.shape[:2] != (panel_h, panel_w):
        panel_rgb = cv2.resize(panel_rgb, (panel_w, panel_h))
    return panel_rgb


def main():
    parser = argparse.ArgumentParser(description="Interactively view episodes from a processed zarr dataset.")
    parser.add_argument("zarr_path", help="Path to the .zarr dataset directory")
    parser.add_argument("--episode", "-e", type=int, default=0, help="Episode index to start at (default: 0)")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback speed in frames per second (default: 10)")
    parser.add_argument(
        "--state", action="store_true",
        help="Show a matplotlib time-series panel for non-camera data below the cameras.",
    )
    parser.add_argument(
        "--cam_h", type=int, default=480,
        help="Display height for each camera tile (default: 480).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.zarr_path) or not os.path.exists(os.path.join(args.zarr_path, ".zgroup")):
        raise SystemExit(
            f"'{args.zarr_path}' is not a valid zarr group "
            "(directory missing or no .zgroup). Run process_pickles.py first."
        )
    root = zarr.open(args.zarr_path, mode="r")
    data = root["data"]
    episode_ends = root["meta"]["episode_ends"][:]
    n_episodes = len(episode_ends)

    keys = _list_keys(data)
    camera_keys = [k for k in keys if _is_camera_arr(data[k])]
    other_keys = [k for k in keys if k not in camera_keys]
    if not camera_keys:
        raise SystemExit(f"No camera arrays found in {args.zarr_path}/data (need uint8, ndim>=4).")

    print(f"Dataset: {args.zarr_path}")
    print(f"Episodes: {n_episodes}  |  Total frames: {int(episode_ends[-1])}")
    print(f"Camera keys: {camera_keys}")
    print(f"Other keys:  {other_keys}")
    print(f"Structure:\n{root.tree()}")
    print()
    print("Controls:")
    print("  k / l     step 1 / 10 frames forward")
    print("  j / h     step 1 / 10 frames backward")
    print("  n / p     next / previous episode")
    print("  Space     toggle play/pause")
    print("  q         quit")
    print()

    ep_idx = max(0, min(args.episode, n_episodes - 1))
    frame_delay_ms = max(1, int(round(1000.0 / args.fps)))
    playing = False

    # Compute per-tile width: scale each camera so all tiles share `cam_h` height,
    # then horizontally concatenate.
    sample = {k: data[k][0] for k in camera_keys}  # (H, W, C) each
    per_tile_w = []
    for k in camera_keys:
        h, w = sample[k].shape[:2]
        per_tile_w.append(int(round(args.cam_h * w / h)))
    cam_h = args.cam_h
    cam_w = int(sum(per_tile_w))
    state_h = 720 if args.state and other_keys else 0
    win_w = cam_w
    win_h = cam_h + state_h

    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dataset Viewer", win_w, win_h)

    def load_episode(ep):
        start = 0 if ep == 0 else int(episode_ends[ep - 1])
        end = int(episode_ends[ep])
        cam_frames = {k: data[k][start:end] for k in camera_keys}
        series = {k: np.asarray(data[k][start:end]) for k in other_keys} if args.state else {}
        return cam_frames, series, end - start

    cam_frames, series, ep_len = load_episode(ep_idx)
    frame_idx = 0

    panel_cache: list = []

    def get_panel(fi, ep_len_):
        nonlocal panel_cache
        if not panel_cache:
            panel_cache = [None] * ep_len_
        if panel_cache[fi] is None:
            panel_cache[fi] = render_state_panel(series, fi, panel_h=state_h, panel_w=cam_w)
        return panel_cache[fi]

    def on_episode_change(ep):
        nonlocal cam_frames, series, ep_len, panel_cache
        cam_frames, series, ep_len = load_episode(ep)
        panel_cache = []

    while True:
        # Assemble the camera strip for this frame.
        tiles = []
        for k, tw in zip(camera_keys, per_tile_w):
            frame = cam_frames[k][frame_idx]
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bgr = cv2.resize(bgr, (tw, cam_h))
            tiles.append(bgr)
        cam_bgr = np.concatenate(tiles, axis=1) if len(tiles) > 1 else tiles[0]

        if state_h > 0:
            panel_rgb = get_panel(frame_idx, ep_len)
            panel_bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
            display = np.concatenate([cam_bgr, panel_bgr], axis=0)
        else:
            display = cam_bgr

        label = f"Ep {ep_idx + 1}/{n_episodes}  |  Frame {frame_idx}/{ep_len - 1}"
        cv2.putText(display, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Dataset Viewer", display)

        wait_ms = frame_delay_ms if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("k") or (playing and key == 255):
            if frame_idx < ep_len - 1:
                frame_idx += 1
            else:
                playing = False
        elif key == ord("l"):
            frame_idx = min(frame_idx + 10, ep_len - 1)
        elif key == ord("j"):
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord("h"):
            frame_idx = max(frame_idx - 10, 0)
        elif key == ord("n"):
            ep_idx = min(ep_idx + 1, n_episodes - 1)
            on_episode_change(ep_idx)
            frame_idx = 0
            playing = False
        elif key == ord("p"):
            ep_idx = max(ep_idx - 1, 0)
            on_episode_change(ep_idx)
            frame_idx = 0
            playing = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
