#!/usr/bin/env python3
"""Relabel a sanding zarr's ACP fields so they carry real signal.

add_sanding_acp_fields.py writes ts_pose_virtual_target_0 = ts_pose_command_0 and
stiffness_0 = const. Both channels are then degenerate: the VT block is a
byte-duplicate of the command block and the stiffness channel has zero variance,
so ACP's 19-dim action is functionally DP with 10 dead dimensions.

This applies the same construction gen_cube_pick_arm_synthetic.py's
relabel_acp_with_known_k() uses:

    VT = cmd + F / k          stiffness_0 = k / STIFFNESS_REF

i.e. the virtual target is the spring anchor that, at stiffness k, would produce
the force actually measured. With k known (the sanding controller's fixed
tool_kp, recorded in the zarr's gen_env attr) this is well-posed, and it turns
the VT block into an implicit force command -- the policy commits to a desired
contact force, not just a position.

Two differences from the cube_pick version, both load-bearing:

  * NO rotation. cube_pick reads `wrench_filtered_0` in the TOOL frame and maps
    the displacement to world via R(cmd_quat). SandingEnv.pad_contact_force()
    already returns the reaction in the WORLD frame, so rotating again would
    apply a spurious transform.
  * Sign. pad_contact_force() returns the reaction ON THE PAD (pressing down
    reads +Fz), so the anchor that produces it sits along +F from the command;
    cube_pick negates because its stored wrench has the opposite convention.

Note the deflection cmd-fb does NOT equal F/k in this data (it is 4-6x larger
and often anti-correlated, being dominated by lateral tracking lag of a
compliant arm following a ~25 mm/s sweep). That is fine here -- this
construction only needs k and F -- but it does mean VT must not be read as
"where the tool actually is".

Stiffness stays constant because the sanding controller's kp is fixed, so this
buys a force-aware action space, NOT adaptive compliance. Varying stiffness
would require regenerating demos with a varying --kp.

    python relabel_sanding_acp.py /store/real/jvclark/sanding_clean_smooth.zarr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STIFFNESS_REF = 16000.0
F_LOW = 2.0            # below this the pad is not meaningfully loaded


def aggregate_force(F: np.ndarray, mode: str, hz: float,
                    tau: float, window: float) -> np.ndarray:
    """Causal aggregation of the (T,3) world-frame force before labelling.

    Both filters are causal on purpose: the label has to be something a policy
    could in principle commit to from past observations only, so a centred or
    look-ahead filter would leak future force into the action target.
    """
    if mode == "raw":
        return F
    T = len(F)
    if mode == "ema":
        alpha = 1.0 - np.exp(-1.0 / (hz * max(tau, 1e-6)))
        out = np.empty_like(F)
        acc = np.zeros(3, dtype=np.float64)
        for i in range(T):
            acc += alpha * (F[i] - acc)
            out[i] = acc
        return out
    if mode == "maxpool":
        # max of |F| over a trailing window, keeping that sample's direction
        w = max(int(round(window * hz)), 1)
        mag = np.linalg.norm(F, axis=1)
        out = np.empty_like(F)
        for i in range(T):
            lo = max(0, i - w + 1)
            out[i] = F[lo + int(np.argmax(mag[lo:i + 1]))]
        return out
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr_path")
    ap.add_argument("--tool-kp", type=float, default=None,
                    help="override; default reads gen_env, else 16000")
    ap.add_argument("--force-agg", choices=["raw", "ema", "maxpool"], default="ema",
                    help="how the force signal driving the VT label is aggregated. "
                         "ema = causal exponential moving average (smooth force "
                         "tracking, matches the Z-servo filter the demos were "
                         "generated with); maxpool = causal windowed maximum, so the "
                         "label tracks peak rather than mean load; raw = unfiltered.")
    ap.add_argument("--agg-tau", type=float, default=0.05,
                    help="ema time constant (s)")
    ap.add_argument("--agg-window", type=float, default=0.05,
                    help="maxpool window (s)")
    ap.add_argument("--hz", type=float, default=1000.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import zarr, numcodecs
    root = zarr.open(args.zarr_path, mode="r" if args.dry_run else "a")

    kp = args.tool_kp
    if kp is None:
        if "gen_env" in root.attrs:
            kp = float(json.loads(root.attrs["gen_env"])["tool_kp"])
        else:
            kp = STIFFNESS_REF
    stiff_norm = kp / STIFFNESS_REF
    print(f"{args.zarr_path}\n  tool_kp={kp:.0f}  stiffness_0={stiff_norm:.4f}"
          f"  force_agg={args.force_agg}"
          + ("   [DRY RUN]" if args.dry_run else ""))

    names = sorted([k for k in root["data"].group_keys() if k.startswith("episode_")],
                   key=lambda k: int(k.rsplit("_", 1)[-1]))
    cmp_ = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    offs = []
    for nm in names:
        ep = root["data"][nm]
        cmd = np.asarray(ep["ts_pose_command_0"][:], dtype=np.float64)   # (T,7)
        F = np.asarray(ep["wrench_0"][:], dtype=np.float64)[:, :3]       # world
        T = len(cmd)

        F = aggregate_force(F, args.force_agg, args.hz, args.agg_tau, args.agg_window)
        fn = np.linalg.norm(F, axis=1)
        disp = np.zeros((T, 3), dtype=np.float64)
        loaded = fn >= F_LOW
        disp[loaded] = F[loaded] / kp        # world frame already: no rotation

        vt = cmd.copy()
        vt[:, :3] += disp                    # VT = cmd + F/k
        qn = np.linalg.norm(vt[:, 3:], axis=1, keepdims=True)
        vt[:, 3:] /= np.where(qn > 0, qn, 1.0)

        offs.append(np.linalg.norm(disp[loaded], axis=1) * 1000 if loaded.any()
                    else np.array([0.0]))

        if args.dry_run:
            continue

        # keep the degenerate originals once, so this is reversible
        if "ts_pose_virtual_target_0_const" not in ep:
            ep.array("ts_pose_virtual_target_0_const",
                     np.asarray(ep["ts_pose_virtual_target_0"][:]),
                     chunks=(min(256, T), 7), compressor=cmp_, overwrite=False)
        if "stiffness_0_const" not in ep:
            ep.array("stiffness_0_const", np.asarray(ep["stiffness_0"][:]),
                     chunks=(min(256, T),), compressor=cmp_, overwrite=False)

        ep.array("ts_pose_virtual_target_0", vt, chunks=(min(256, T), 7),
                 compressor=cmp_, overwrite=True)
        ep.array("stiffness_0", np.full(T, stiff_norm, dtype=np.float64),
                 chunks=(min(256, T),), compressor=cmp_, overwrite=True)

    allo = np.concatenate(offs)
    print(f"  {len(names)} episodes relabelled")
    print(f"  |VT - cmd| while loaded: mean {allo.mean():.3f} mm  "
          f"p95 {np.percentile(allo, 95):.3f} mm  max {allo.max():.3f} mm")
    print(f"  (expected ~F/k = {9.5/kp*1000:.2f} mm at the ~9.5 N demo force)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
