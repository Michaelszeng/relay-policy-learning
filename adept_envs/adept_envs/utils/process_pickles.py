#!/usr/bin/env python
"""Pack .pkl demos under a directory tree into a single zarr replay buffer.

Each .pkl file is treated as one episode and must contain a dict with at least:
    {
        "observations": dict[str, ndarray]  OR  ndarray (treated as "state"),
        "actions":      ndarray of shape (T, A),
    }
Camera observations are detected automatically (uint8 with per-step ndim >= 2).

Output layout matches the diffusion-policy replay buffer format:
    /data/<key>            (T_total, ...)
    /meta/episode_ends     (N_episodes,) cumulative timestep counts
"""
import pickle
from pathlib import Path

import click
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm


CAMERA_CHUNK_T = 128
STATE_CHUNK_T = 1024
ACTION_CHUNK_T = 2048


def _episode_array(ep, key):
    """Pull (T, ...) array for `key` out of an episode dict."""
    if key == "action":
        arr = np.asarray(ep["actions"])
    else:
        obs = ep["observations"]
        if isinstance(obs, dict):
            arr = np.asarray(obs[key])
        elif key == "state":
            arr = np.asarray(obs)
        else:
            raise KeyError(key)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _is_camera(arr):
    return arr.dtype == np.uint8 and arr.ndim >= 3


def _schema_keys(ep):
    keys = []
    obs = ep["observations"]
    if isinstance(obs, dict):
        keys.extend(obs.keys())
    else:
        keys.append("state")
    keys.append("action")
    return keys


def _chunks_for(key, per_step_shape, total, is_camera):
    if is_camera:
        t_chunk = CAMERA_CHUNK_T
    elif key == "action":
        t_chunk = ACTION_CHUNK_T
    else:
        t_chunk = STATE_CHUNK_T
    return (min(t_chunk, max(total, 1)),) + tuple(per_step_shape)


@click.command(help="Convert .pkl demo files in a directory tree into a zarr replay buffer.")
@click.option(
    "--input_dir", "-i", required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory to search recursively for .pkl files.",
)
@click.option(
    "--output", "-o", required=True, type=click.Path(),
    help="Output .zarr path.",
)
def main(input_dir, output):
    pkl_paths = sorted(str(p) for p in Path(input_dir).rglob("*.pkl"))
    if not pkl_paths:
        raise click.ClickException(f"No .pkl files found under {input_dir}")
    print(f"Found {len(pkl_paths)} episodes")

    # First pass: episode lengths + schema (keys, per-step shapes, dtypes).
    lengths = []
    schema = None
    for p in tqdm(pkl_paths, desc="Scanning pkls", unit="ep"):
        with open(p, "rb") as f:
            ep = pickle.load(f)
        T = int(np.asarray(ep["actions"]).shape[0])
        lengths.append(T)
        if schema is None:
            schema = []
            for key in _schema_keys(ep):
                arr = _episode_array(ep, key)
                schema.append((key, arr.shape[1:], arr.dtype, _is_camera(arr)))

    total = int(sum(lengths))
    print(f"Total timesteps: {total} across {len(pkl_paths)} episodes")
    for key, per_step_shape, dtype, is_cam in schema:
        kind = "camera" if is_cam else "vector"
        print(f"  {key:<24} {kind:<7} shape=(T,{','.join(map(str, per_step_shape))}) dtype={dtype}")

    # Create zarr layout.
    store = zarr.DirectoryStore(output)
    root = zarr.group(store=store, overwrite=True)
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

    arrays = {}
    for key, per_step_shape, dtype, is_cam in schema:
        shape = (total,) + tuple(per_step_shape)
        chunks = _chunks_for(key, per_step_shape, total, is_cam)
        arrays[key] = data_group.create_dataset(
            key, shape=shape, chunks=chunks, dtype=dtype, compressor=compressor,
        )

    episode_ends = np.cumsum(lengths).astype(np.int64)
    meta_group.create_dataset(
        "episode_ends",
        data=episode_ends,
        chunks=(max(len(episode_ends), 1),),
        dtype="i8",
        compressor=compressor,
    )

    # Second pass: stream episode data into the zarr arrays.
    offset = 0
    pbar = tqdm(pkl_paths, desc="Writing zarr", unit="ep", total=len(pkl_paths))
    written_steps = 0
    for i, p in enumerate(pbar):
        with open(p, "rb") as f:
            ep = pickle.load(f)
        T = lengths[i]
        for key, _, _, _ in schema:
            arr = _episode_array(ep, key)
            n = min(arr.shape[0], T)
            arrays[key][offset:offset + n] = arr[:n]
        offset += T
        written_steps += T
        pbar.set_postfix(steps=f"{written_steps}/{total}")

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
