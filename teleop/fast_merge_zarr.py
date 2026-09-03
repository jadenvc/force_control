#!/usr/bin/env python3
"""Merge pyrite-schema zarr stores by copying episode directories verbatim.

``merge_zarr.py`` round-trips every array through zarr (decompress -> recompress),
which costs ~27 s per sanding episode, almost all of it re-encoding RGB that is
already stored exactly as we want it. Each episode is a self-contained group
directory, so the merge is really just a directory copy plus a renumber, and the
per-episode ``.zattrs`` come along for free instead of having to be re-copied.

Rebuilds ``meta/episode_{robot0,rgb0,wrench0}_len`` from the ``.zarray`` shapes
(metadata only -- no chunk reads), keeping rgb's own length rather than assuming
it equals the robot length.

    python fast_merge_zarr.py --inputs a.zarr b.zarr --output merged.zarr [--limit 100]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr


def ep_dirs(store):
    d = os.path.join(store, "data")
    if not os.path.isdir(d):
        return []
    names = [e for e in os.listdir(d) if e.startswith("episode_")]
    return [os.path.join(d, n) for n in
            sorted(names, key=lambda s: int(s.rsplit("_", 1)[-1]))]


def link_tree(src, dst, copy=False):
    """Hardlink an episode directory into place, falling back to a real copy.

    zarr keeps every chunk as its own file, so an episode is hundreds of small
    files and a byte copy over NFS runs ~90 s/episode. Source and destination
    live on the same filesystem here, so ``cp -al`` just creates directory
    entries -- effectively instant and it uses no extra space. The merged store
    is only ever read (and the ACP patch adds NEW arrays rather than rewriting
    existing chunks), so sharing inodes with the shards is safe.
    """
    if not copy:
        r = subprocess.run(["cp", "-al", src, dst], capture_output=True)
        if r.returncode == 0:
            return
    shutil.copytree(src, dst)


def array_len(ep_dir, key):
    z = os.path.join(ep_dir, key, ".zarray")
    if not os.path.exists(z):
        return None
    return int(json.load(open(z))["shape"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after N episodes (0 = all)")
    ap.add_argument("--copy", action="store_true",
                    help="real copy instead of hardlinks (use across filesystems)")
    ap.add_argument("--jobs", type=int, default=24,
                    help="parallel episode links; NFS link() latency, not CPU, is the limit")
    args = ap.parse_args()

    out = args.output
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "data"), exist_ok=True)

    # root group/attrs from the first input that has them
    for src in args.inputs:
        for f in (".zgroup", ".zattrs"):
            s = os.path.join(src, f)
            d = os.path.join(out, f)
            if os.path.exists(s) and not os.path.exists(d):
                shutil.copy2(s, d)
    if not os.path.exists(os.path.join(out, ".zgroup")):
        json.dump({"zarr_format": 2}, open(os.path.join(out, ".zgroup"), "w"))
    json.dump({"zarr_format": 2}, open(os.path.join(out, "data", ".zgroup"), "w"))

    # Build the whole (src, dst) plan first so the links can run concurrently.
    plan = []
    for src in args.inputs:
        for ep in ep_dirs(src):
            if args.limit and len(plan) >= args.limit:
                break
            plan.append((ep, os.path.join(out, "data", f"episode_{len(plan)}")))
        if args.limit and len(plan) >= args.limit:
            break

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(lambda a: link_tree(a[0], a[1], copy=args.copy), plan))

    robot, rgb, wrench = [], [], []
    for _, dst in plan:
        robot.append(array_len(dst, "ts_pose_fb_0") or 0)
        wrench.append(array_len(dst, "wrench_0") or 0)
        rgb.append(array_len(dst, "rgb_0") or robot[-1])
    n = len(plan)

    root = zarr.open(out, mode="a")
    meta = root.require_group("meta")
    for key, vals in (("episode_robot0_len", robot),
                      ("episode_rgb0_len", rgb),
                      ("episode_wrench0_len", wrench)):
        meta.create_dataset(key, data=np.array(vals, dtype=np.int64),
                            dtype="i8", overwrite=True)

    print(f"merged {n} episodes -> {out}")


if __name__ == "__main__":
    main()
