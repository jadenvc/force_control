#!/usr/bin/env python3
"""Post-processing: add ts_pose_virtual_target_0 and stiffness_0 to sanding zarr.

Required by sanding_acp_stiff_fixed_s8h8.yaml before ACP training.

For the scripted sanding data:
  ts_pose_virtual_target_0 = ts_pose_command_0
      (the commanded position IS the virtual target — pure position controller)
  stiffness_0 = tool_kp / STIFFNESS_REF everywhere (constant, fixed per dataset)
      Read from the zarr's `gen_env` attr when present, so a dataset generated at
      a reduced tool_kp is not mislabelled as running at full stiffness. The clean
      replay datasets use tool_kp=4000 -> 0.25, NOT 1.0.

      NOTE: this channel is constant within a dataset, so it carries no
      information for the policy either as an input or as an action target, and
      eval_sanding1_policy.py ignores the predicted stiffness (--fixed-stiffness,
      real stiffness comes from --tool-kp). Getting the value right therefore
      matters for provenance and for any future variable-stiffness work, not for
      the current numbers.

Usage:
    python add_sanding_acp_fields.py \\
        /store/real/jvclark/PyriteML/data/sanding/sanding_synthetic.zarr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/store/real/jvclark/force_control/teleop")
sys.path.insert(0, "/store/real/jvclark/force_control/flipup_minimal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path", help="Path to sanding_synthetic.zarr")
    args = parser.parse_args()

    try:
        import zarr, numcodecs
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zarr", "numcodecs"])
        import zarr, numcodecs

    zarr_path = Path(args.zarr_path).expanduser().resolve()
    if not zarr_path.exists():
        print(f"ERROR: zarr not found: {zarr_path}", file=sys.stderr)
        sys.exit(1)

    STIFFNESS_REF = 16000.0
    root = zarr.open(str(zarr_path), mode="a")
    stiffness_value = 1.0
    if "gen_env" in root.attrs:
        try:
            kp = float(json.loads(root.attrs["gen_env"])["tool_kp"])
            stiffness_value = kp / STIFFNESS_REF
            print(f"gen_env tool_kp={kp:.0f} -> stiffness_0={stiffness_value:.4f}")
        except Exception as e:
            print(f"WARNING: could not read gen_env tool_kp ({e}); using 1.0")
    else:
        print("no gen_env attr; assuming tool_kp=16000 -> stiffness_0=1.0")
    episodes = sorted(
        [k for k in root["data"].group_keys() if k.startswith("episode_")],
        key=lambda k: int(k.rsplit("_", 1)[-1]),
    )
    print(f"Found {len(episodes)} episodes in {zarr_path}")

    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    for ep_name in episodes:
        ep = root["data"][ep_name]

        # Skip if already patched
        if "ts_pose_virtual_target_0" in ep and "stiffness_0" in ep:
            continue

        cmd_pose = np.asarray(ep["ts_pose_command_0"], dtype=np.float64)  # (T, 7)
        T = len(cmd_pose)

        # Virtual target = commanded pose
        if "ts_pose_virtual_target_0" not in ep:
            ep.array(
                name="ts_pose_virtual_target_0",
                data=cmd_pose,
                chunks=(min(256, T), 7),
                compressor=compressor,
                overwrite=True,
            )

        # Stiffness = 1.0 (normalised max), constant throughout episode.
        # Store as 1D (T,) — raw_to_action19 does [:, np.newaxis] to get (T, 1).
        if "stiffness_0" not in ep:
            stiffness = np.full(T, stiffness_value, dtype=np.float64)
            ep.array(
                name="stiffness_0",
                data=stiffness,
                chunks=(min(256, T),),
                compressor=compressor,
                overwrite=True,
            )

        print(f"  Patched {ep_name}  (T={T})")

    print("Done.")


if __name__ == "__main__":
    main()
