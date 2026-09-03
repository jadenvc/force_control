#!/usr/bin/env python3
"""Compare a generated clean sanding dataset against the source teleop demos.

Checks the three things that matter before training on it:
  1. task success / coverage
  2. force smoothness (contact-loss rate, roughness, in-contact force level)
  3. demonstrator timing (episode duration, lateral sweep speed) vs the source

    python verify_sanding_clean.py --new /store/real/jvclark/sanding_clean_smooth.zarr
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import zarr

HZ = 1000.0
SRC_DEFAULT = "/store/real/jvclark/sanding_1.zarr"


def episodes(path):
    d = os.path.join(path, "data")
    return sorted([e for e in os.listdir(d) if e.startswith("episode_")],
                  key=lambda s: int(s.split("_")[1]))


def per_episode(path, nm):
    g = zarr.open(os.path.join(path, "data", nm), mode="r")
    a = dict(g.attrs)
    F = np.asarray(g["normal_force_n"][:], float).ravel()
    fb = np.asarray(g["ts_pose_fb_0"][:], float)
    n = len(F)
    t = F > 1.0
    out = dict(dur=n / HZ, success=bool(a.get("success", False)),
               broken=bool(a.get("broken", False)),
               cov=float(a.get("final_task_metric_value", 0.0)),
               n_rgb=int(g["rgb_0"].shape[0]))
    if not t.any():
        out.update(F_in=0, F_std=0, gaps_s=0, rough=0, lat=0, contact=0)
        return out
    i0, i1 = int(np.argmax(t)), int(n - np.argmax(t[::-1]))
    Fe, te = F[i0:i1], t[i0:i1]
    idx = np.flatnonzero(np.diff(np.concatenate(([0], (~te).astype(np.int8), [0]))))
    gaps = idx.reshape(-1, 2)
    dur = max((i1 - i0) / HZ, 1e-6)
    v = np.diff(fb[i0:i1, :3], axis=0) * HZ
    lat = np.linalg.norm(v[:, :2], axis=1)[te[:-1]]
    out.update(F_in=float(Fe[te].mean()), F_std=float(Fe[te].std()),
               gaps_s=len(gaps) / dur, rough=float(np.diff(Fe).std()),
               lat=float(np.median(lat)) * 1000 if len(lat) else 0.0,
               contact=float(te.mean()))
    return out


def summarize(path, label):
    rows = [per_episode(path, nm) for nm in episodes(path)]
    def C(k):
        return np.array([r[k] for r in rows], float)
    n = len(rows)
    print(f"\n=== {label}  ({n} episodes) ===")
    print(f"  success            {int(C('success').sum())}/{n} = {100*C('success').mean():.0f}%"
          f"   broken {int(C('broken').sum())}")
    for k, u, f in [("cov", "", "{:.3f}"), ("dur", "s", "{:.2f}"),
                    ("F_in", "N", "{:.2f}"), ("F_std", "N", "{:.2f}"),
                    ("gaps_s", "/s", "{:.2f}"), ("rough", "N/ms", "{:.3f}"),
                    ("lat", "mm/s", "{:.1f}"), ("contact", "", "{:.3f}"),
                    ("n_rgb", "", "{:.0f}")]:
        v = C(k)
        print(f"  {k:<18} " + f.format(v.mean()) + f" ± " + f.format(v.std())
              + f"   [{f.format(v.min())} .. {f.format(v.max())}] {u}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--src", default=SRC_DEFAULT)
    args = ap.parse_args()

    src = summarize(args.src, f"SOURCE  {os.path.basename(args.src)}")
    new = summarize(args.new, f"CLEAN   {os.path.basename(args.new)}")

    def m(rows, k):
        return float(np.mean([r[k] for r in rows]))

    print("\n=== HEAD TO HEAD (means) ===")
    print(f"  {'metric':<22} {'source':>10} {'clean':>10} {'change':>14}")
    for k, better in [("gaps_s", "lower"), ("rough", "lower"), ("F_std", "lower"),
                      ("cov", "higher"), ("dur", "match"), ("lat", "match"),
                      ("F_in", "match"), ("contact", "higher")]:
        a, b = m(src, k), m(new, k)
        if better == "match":
            chg = f"{100*(b-a)/max(abs(a),1e-9):+.1f}%"
        elif a > 0:
            chg = f"{a/b:.1f}x lower" if (better == "lower" and b > 0) else f"{100*(b-a)/a:+.1f}%"
        else:
            chg = "-"
        print(f"  {k:<22} {a:10.3f} {b:10.3f} {chg:>14}")
    sa = 100 * np.mean([r["success"] for r in src])
    sb = 100 * np.mean([r["success"] for r in new])
    print(f"  {'success rate %':<22} {sa:10.1f} {sb:10.1f} {sb-sa:>+13.1f}")


if __name__ == "__main__":
    main()
