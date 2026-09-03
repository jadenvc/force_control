#!/usr/bin/env python3
"""Repair the rgb/robot clock mismatch in a sanding teleop zarr.

THE BUG (teleop_sanding.py, the record_sample call around line 895):

    timestamp_ms         = step_index * 1000.0 / args.control_freq   # session clock
    image_capture_time_s = shot["sim_time_s"]  (= env.data.time)     # per-episode clock

``step_index`` is a session-global control-tick counter that keeps counting
across episodes, so ``robot_time_stamps_0`` / ``wrench_time_stamps_0`` are
session-cumulative. ``env.data.time`` is reset by ``env.reset()`` at the start of
every episode, so ``rgb_time_stamps_0`` restarts near zero each episode. The two
streams therefore only share a clock for the FIRST recorded episode.

WHY IT MATTERS: PyriteML's sampler intersects the streams in time --

    start_time = max(rgb_t[0],  robot_t[0])
    end_time   = min(rgb_t[-1], robot_t[-1])

For every episode after the first, ``robot_t[0]`` is far larger than
``rgb_t[-1]``, so the window is empty and the episode silently contributes zero
training samples. On sanding_1.zarr this dropped 54 of 56 episodes (613 samples
survived, all from the 2 episodes whose clocks happened to line up).

THE FIX: within an episode both clocks advance at exactly 1 ms per control tick,
so they differ by a constant. Recover it from a stream recorded on BOTH clocks --
``sim_time_s`` is per-sample sim time (rgb's clock) and sits alongside
``robot_time_stamps_0`` (robot's clock):

    offset            = median(robot_time_stamps_0 - 1000 * sim_time_s)
    rgb_time_stamps_0 = rgb_time_stamps_0_raw + offset

The original array is preserved as ``rgb_time_stamps_0_raw`` before the first
overwrite, so this is reversible and idempotent.

Usage:
    python fix_sanding_rgb_timestamps.py /store/real/jvclark/sanding_1.zarr [--dry-run]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import zarr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr_path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = zarr.open(args.zarr_path, mode="r" if args.dry_run else "a")
    names = sorted(
        [k for k in root["data"].group_keys() if k.startswith("episode_")],
        key=lambda k: int(k.rsplit("_", 1)[-1]),
    )
    print(f"{len(names)} episodes in {args.zarr_path}"
          + ("   [DRY RUN]" if args.dry_run else ""))

    n_fixed = n_ok = n_bad = 0
    for nm in names:
        ep = root["data"][nm]
        raw_key = "rgb_time_stamps_0_raw"
        already = raw_key in ep

        rgb_raw = np.asarray(ep[raw_key][:] if already
                             else ep["rgb_time_stamps_0"][:], dtype=np.float64).ravel()
        robot = np.asarray(ep["robot_time_stamps_0"][:], dtype=np.float64).ravel()
        sim_ms = np.asarray(ep["sim_time_s"][:], dtype=np.float64).ravel() * 1000.0

        off_series = robot - sim_ms
        offset = float(np.median(off_series))
        spread = float(np.ptp(off_series))
        if spread > 5.0:
            print(f"  {nm}: WARNING offset not constant (spread {spread:.1f} ms) "
                  f"-- skipping, needs manual inspection")
            n_bad += 1
            continue

        rgb_new = rgb_raw + offset

        # sanity: aligned rgb must now sit inside the robot stream's window
        inside = (rgb_new[-1] > robot[0]) and (rgb_new[0] < robot[-1])
        mono = bool(np.all(np.diff(rgb_new) > 0))
        if not (inside and mono):
            print(f"  {nm}: WARNING post-fix check failed "
                  f"(inside={inside} monotonic={mono}) -- skipping")
            n_bad += 1
            continue

        if abs(offset) < 1e-9 and not already:
            n_ok += 1
            continue

        if args.dry_run:
            print(f"  {nm}: would shift rgb by {offset:+.0f} ms "
                  f"[{rgb_raw[0]:.0f}..{rgb_raw[-1]:.0f}] -> "
                  f"[{rgb_new[0]:.0f}..{rgb_new[-1]:.0f}] "
                  f"(robot [{robot[0]:.0f}..{robot[-1]:.0f}])")
        else:
            if not already:
                ep.array(raw_key, rgb_raw, chunks=rgb_raw.shape, dtype="f8",
                         overwrite=False)
            ep["rgb_time_stamps_0"][:] = rgb_new.astype(
                ep["rgb_time_stamps_0"].dtype)
        n_fixed += 1

    print(f"\nshifted: {n_fixed}   already aligned: {n_ok}   skipped: {n_bad}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
