"""Merge multiple zarr stores (same pyrite_flipup_sim schema) into one.

Episodes are renumbered sequentially (0, 1, 2, ...) across all source stores.
All per-episode arrays are copied verbatim; top-level metadata (schema_name,
schema_version, etc.) is taken from the first source store, then the
per-episode length arrays in 'meta' are rebuilt from the merged episode list.

Usage:
    python merge_zarr.py --inputs a.zarr b.zarr c.zarr --output combined.zarr
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import zarr
from tqdm import tqdm


def merge_zarrs(input_paths: list[str], output_path: str, min_len: int = 0):
    """Merge zarr stores, optionally skipping episodes shorter than min_len frames.

    min_len is checked against the length of 'ts_pose_fb_0' (= wrench_time_stamps_0
    = rgb_time_stamps_0 in properly-aligned sim data).  Set to 0 to keep all episodes.
    """
    # Collect source stores
    sources = [zarr.open(p, mode="r") for p in input_paths]

    ep_counts = []
    for p, root in zip(input_paths, sources):
        n = len(list(root["data"].group_keys()))
        if min_len > 0:
            kept = sum(
                1 for ep in root["data"].group_keys()
                if len(root["data"][ep]["ts_pose_fb_0"]) >= min_len
            )
            print(f"  {p}: {n} episodes, {kept} kept (min_len={min_len})")
        else:
            print(f"  {p}: {n} episodes")
            kept = n
        ep_counts.append(kept)

    total = sum(ep_counts)
    print(f"  Total: {total} episodes -> {output_path}")

    # Open output (overwrite if exists)
    try:
        shutil.rmtree(output_path)
    except FileNotFoundError:
        pass
    out = zarr.open(output_path, mode="w")

    # Copy top-level metadata attrs from first source
    for k, v in sources[0].attrs.items():
        out.attrs[k] = v

    out_data = out.require_group("data")
    out_meta = out.require_group("meta")

    out_ep_robot_lens = []
    out_ep_rgb_lens = []
    out_ep_wrench_lens = []

    global_ep = 0
    for src_path, src in zip(input_paths, sources):
        src_names = sorted(
            src["data"].group_keys(),
            key=lambda k: int(k.rsplit("_", 1)[-1]),
        )
        for name in tqdm(src_names, desc=f"Copying {src_path.split('/')[-1]}"):
            src_ep = src["data"][name]
            dst_name = f"episode_{global_ep}"
            dst_ep = out_data.require_group(dst_name)

            # Copy every array in the episode
            for key in src_ep.keys():
                arr = src_ep[key]
                dst_ep.create_dataset(
                    key,
                    data=arr[:],
                    chunks=arr.chunks,
                    dtype=arr.dtype,
                    compressor=arr.compressor,
                    overwrite=True,
                )

            # Per-episode attrs carry success / coverage / termination reason.
            # Without this the merged store reports every episode as a failure
            # with coverage 0, since zarr defaults a missing attr rather than
            # erroring -- silent, and it survives all the way into eval configs.
            for k, v in src_ep.attrs.items():
                dst_ep.attrs[k] = v

            out_ep_robot_lens.append(len(src_ep["ts_pose_fb_0"]))
            out_ep_wrench_lens.append(len(src_ep["wrench_0"]))
            # RGB is asynchronous (~38 Hz against 1 kHz control) in the sanding
            # teleop schema, so its length is NOT the robot length -- the old
            # code assumed the two matched, which held only for the older
            # per-control-step sim recordings.
            out_ep_rgb_lens.append(
                src_ep["rgb_0"].shape[0] if "rgb_0" in src_ep
                else len(src_ep["ts_pose_fb_0"]))
            global_ep += 1

    # Rebuild meta length arrays
    out_meta.create_dataset(
        "episode_robot0_len", data=np.array(out_ep_robot_lens, dtype=np.int64), overwrite=True
    )
    out_meta.create_dataset(
        "episode_rgb0_len", data=np.array(out_ep_rgb_lens, dtype=np.int64), overwrite=True
    )
    out_meta.create_dataset(
        "episode_wrench0_len", data=np.array(out_ep_wrench_lens, dtype=np.int64), overwrite=True
    )

    print(f"Done. Merged {global_ep} episodes into {output_path}")
    return global_ep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merge_zarrs(args.inputs, args.output)
