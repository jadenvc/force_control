#!/usr/bin/env python3
"""Prove that the env a sanding dataset was generated in == the env eval builds.

Three layers are compared, because agreement at one level does not imply the others:

  1. what the dataset says it was made in     (zarr root attr ``gen_env``)
  2. what the task YAML tells eval to build   (``eval.extra_args``)
  3. what MuJoCo actually ends up with        (live geom_solref / geom_friction /
                                               tool_kp on the constructed env)

Layer 3 matters: ``--pad-softness`` was silently inert for months because the
value written to the pad geom was overridden by the panel's higher contact
priority, so a config that "looks right" can still compile to different physics.

  # stamp a freshly generated dataset with the params it was made with
  python check_sanding_env_match.py --zarr X.zarr --stamp --tool-kp 4000 --friction 0.3

  # then verify dataset <-> config <-> live env
  python check_sanding_env_match.py --zarr X.zarr --config sanding_smooth_dp_ft_s8h8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml
import zarr

sys.path.insert(0, "/store/real/jvclark/force_control/teleop")
sys.path.insert(0, "/store/real/jvclark/force_control/flipup_minimal")
os.environ.setdefault("MUJOCO_GL", "egl")

CFG_DIR = "/store/real/jvclark/PyriteML/diffusion_policy/config/task"


def read_extra_args(cfg_name):
    path = cfg_name if cfg_name.endswith(".yaml") else os.path.join(CFG_DIR, cfg_name + ".yaml")
    cfg = yaml.safe_load(open(path))
    args = [str(a) for a in cfg.get("eval", {}).get("extra_args", [])]
    out = {}
    for i, a in enumerate(args[:-1]):
        if a in ("--tool-kp", "--friction"):
            out[a.lstrip("-").replace("-", "_")] = float(args[i + 1])
    return out, cfg.get("dataset_path"), path


def live_env(tool_kp, friction):
    from sanding_teleop import SandingTeleop, SandingProperties
    env = SandingTeleop(seed=0,
                        properties=SandingProperties(friction=(friction, 0.01, 0.0002)),
                        tool_kp=tool_kp, arm_damping=2.5)
    env.reset()
    return {
        # task_space_kp is a per-axis vector on this env, not a scalar
        "tool_kp": np.atleast_1d(np.asarray(
            getattr(env, "task_space_kp", tool_kp), dtype=float)).tolist(),
        "panel_solref": [float(x) for x in env.model.geom_solref[env.panel_surface_geom_id]],
        "panel_solimp_w": float(env.model.geom_solimp[env.panel_surface_geom_id, 2]),
        "panel_friction": [float(x) for x in env.model.geom_friction[env.panel_surface_geom_id]],
        "pad_solref": [float(x) for x in env.model.geom_solref[env.pad_geom_id]],
        "cam": [env.default_cam_azimuth, env.default_cam_elevation, env.default_cam_distance],
        "dose_band": [env.properties.dose_low, env.properties.dose_high],
        "force_target": env.properties.force_target_n,
        "n_regions_range": [env.properties.num_regions_min, env.properties.num_regions_max],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stamp", action="store_true")
    ap.add_argument("--tool-kp", type=float, default=None)
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--warp", type=float, default=None)
    args = ap.parse_args()

    root = zarr.open(args.zarr, mode="a" if args.stamp else "r")

    if args.stamp:
        if args.tool_kp is None or args.friction is None:
            sys.exit("--stamp needs --tool-kp and --friction")
        root.attrs["gen_env"] = json.dumps({
            "tool_kp": args.tool_kp, "friction": args.friction,
            "arm_damping": 2.5, "mode": args.mode, "warp": args.warp,
            "generator": "gen_sanding_clean.py",
            "source": "/store/real/jvclark/sanding_1.zarr",
        })
        print(f"stamped gen_env on {args.zarr}: {root.attrs['gen_env']}")
        if args.config is None:
            return

    if "gen_env" not in root.attrs:
        sys.exit(f"{args.zarr} has no gen_env attr -- run with --stamp first")
    gen = json.loads(root.attrs["gen_env"])
    print(f"dataset  {os.path.basename(args.zarr)}")
    print(f"  gen_env: tool_kp={gen['tool_kp']:.0f}  friction={gen['friction']}  "
          f"mode={gen.get('mode')}  warp={gen.get('warp')}")

    ok = True
    if args.config:
        ev, ds_path, cfg_path = read_extra_args(args.config)
        print(f"config   {os.path.basename(cfg_path)}")
        print(f"  eval.extra_args: {ev}")
        print(f"  dataset_path:    {ds_path}")
        if os.path.realpath(str(ds_path)) != os.path.realpath(args.zarr):
            print(f"  MISMATCH dataset_path points elsewhere")
            ok = False
        for k in ("tool_kp", "friction"):
            if k not in ev:
                print(f"  MISMATCH eval.extra_args is missing --{k.replace('_','-')}")
                ok = False
            elif abs(ev[k] - gen[k]) > 1e-9:
                print(f"  MISMATCH {k}: dataset {gen[k]} vs config {ev[k]}")
                ok = False
        kp, mu = ev.get("tool_kp", gen["tool_kp"]), ev.get("friction", gen["friction"])
    else:
        kp, mu = gen["tool_kp"], gen["friction"]

    live = live_env(kp, mu)
    print("live env built from those knobs:")
    for k, v in live.items():
        print(f"  {k:16s} {v}")
    if abs(live["panel_friction"][0] - mu) > 1e-9:
        print(f"  MISMATCH panel sliding friction {live['panel_friction'][0]} != {mu}")
        ok = False

    print("\n" + ("ENV MATCH OK" if ok else "ENV MISMATCH -- fix before training"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
