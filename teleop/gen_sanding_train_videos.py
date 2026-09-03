#!/usr/bin/env python3
"""Render force-annotated videos of sanding TRAINING samples straight from a zarr.

Left  : the rgb_0 frame the policy actually sees (224x224, upscaled)
Right : normal contact force over time, with the dose floor / target / cap /
        break reference lines and every contact-loss sample marked in red,
        plus live coverage.

Works on the source teleop zarr and on any clean replay-generated one, so the
two can be eyeballed side by side.

    python gen_sanding_train_videos.py --zarr /store/real/jvclark/sanding_clean_smooth.zarr \
        --out /store/real/jvclark/eval_videos/sanding_clean_smooth --n 8
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import imageio
import numpy as np
import zarr

sys.path.insert(0, "/store/real/jvclark/force_control/teleop")
from eval_sanding1_policy import render_force_panel   # noqa: E402

HZ = 1000.0


def episodes(path):
    d = os.path.join(path, "data")
    return sorted([e for e in os.listdir(d) if e.startswith("episode_")],
                  key=lambda s: int(s.split("_")[1]))


def force_summary(F):
    t = F > 1.0
    if not t.any():
        return 0.0, 0.0, 0.0
    i0, i1 = int(np.argmax(t)), int(len(F) - np.argmax(t[::-1]))
    Fe, te = F[i0:i1], t[i0:i1]
    idx = np.flatnonzero(np.diff(np.concatenate(([0], (~te).astype(np.int8), [0]))))
    gaps = len(idx.reshape(-1, 2))
    dur = max((i1 - i0) / HZ, 1e-6)
    return float(Fe[te].mean()), gaps / dur, float(np.diff(Fe).std())


def render_episode(path, nm, out_path, fps, scale):
    g = zarr.open(os.path.join(path, "data", nm), mode="r")
    rgb = np.asarray(g["rgb_0"][:], dtype=np.uint8)
    F = np.asarray(g["normal_force_n"][:], float).ravel()
    cov = np.asarray(g["coverage_just_right"][:], float).ravel()
    rt = np.asarray(g["rgb_time_stamps_0"][:], float).ravel()
    bt = np.asarray(g["robot_time_stamps_0"][:], float).ravel()
    attrs = dict(g.attrs)

    # map each rgb frame onto the control-rate index it was captured at
    idx = np.clip(np.searchsorted(bt, rt), 0, len(F) - 1)
    n = len(rgb)
    h = w = int(224 * scale)
    panel_w = max(w, 460)

    f_max = float(max(50.0, np.max(F) * 1.15))
    F_in, gaps_s, rough = force_summary(F)
    outcome = "SUCCESS" if attrs.get("success") else ("BROKEN" if attrs.get("broken") else "FAIL")

    frames = []
    for k in range(n):
        j = int(idx[k])
        left = cv2.resize(rgb[k], (w, h), interpolation=cv2.INTER_NEAREST)
        panel = render_force_panel(F[:j + 1], cov[:j + 1], panel_w, h,
                                   window_s=4.0, fps=HZ, f_max=f_max)
        comp = np.hstack([cv2.cvtColor(left, cv2.COLOR_RGB2BGR), panel])
        bar = np.full((30, comp.shape[1], 3), 252, np.uint8)
        cv2.putText(bar, f"{nm}  {outcome}  cov {attrs.get('final_task_metric_value', 0):.2f}  "
                         f"{len(F)/HZ:.1f}s   F {F_in:.1f}N  losses {gaps_s:.2f}/s  "
                         f"rough {rough:.3f} N/ms",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (11, 11, 11), 1, cv2.LINE_AA)
        frames.append(cv2.cvtColor(np.vstack([bar, comp]), cv2.COLOR_BGR2RGB))

    imageio.mimwrite(out_path, frames, fps=fps, quality=7, macro_block_size=1)
    return outcome, F_in, gaps_s, rough, len(F) / HZ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    eps = episodes(args.zarr)
    rng = np.random.default_rng(args.seed)
    pick = list(eps) if args.n >= len(eps) else [
        eps[i] for i in sorted(rng.choice(len(eps), args.n, replace=False))]

    print(f"{os.path.basename(args.zarr)}: rendering {len(pick)} of {len(eps)} episodes -> {args.out}")
    for nm in pick:
        dst = os.path.join(args.out, f"{nm}.mp4")
        outcome, F_in, gaps_s, rough, dur = render_episode(
            args.zarr, nm, dst, args.fps, args.scale)
        print(f"  {nm:<14} {outcome:<8} {dur:5.1f}s  F {F_in:5.2f}N  "
              f"losses {gaps_s:6.2f}/s  rough {rough:.3f} N/ms  -> {os.path.basename(dst)}")


if __name__ == "__main__":
    main()
