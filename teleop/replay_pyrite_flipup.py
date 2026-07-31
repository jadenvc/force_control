"""Validate or replay an episode written by ``--collect-dataset``.

Examples:
    python replay_pyrite_flipup.py ~/data/flipup_sim_20hz.zarr --list
    python replay_pyrite_flipup.py ~/data/flipup_sim_20hz.zarr --validate-only
    python replay_pyrite_flipup.py ~/data/flipup_sim_20hz.zarr --episode episode_0
    python replay_pyrite_flipup.py ~/data/flipup_sim_20hz.zarr --mode state \
        --output /tmp/replay.mp4 --no-view
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import cv2
import mujoco
import numpy as np
import zarr

from flipup_teleop import FlipUpTeleop
from flipup.physical_properties import PhysicalProperties
from pyrite_recorder import validate_pyrite_dataset


def _episode_names(root) -> list[str]:
    return sorted(
        root["data"].group_keys(),
        key=lambda key: int(key.rsplit("_", 1)[-1]),
    )


def _make_env(metadata: dict) -> FlipUpTeleop:
    controller = metadata["controller"]
    return FlipUpTeleop(
        seed=int(metadata["seed"]),
        physical_properties=PhysicalProperties(**metadata["physical_properties"]),
        tool_kp=float(controller["tool_kp"]),
        tool_rot_kp=float(controller["tool_rot_kp"]),
        tool_rot_kd=float(controller["tool_rot_kd"]),
        joint_kd=np.asarray(controller["joint_kd"], dtype=float),
        settle_s=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", default=None, help="default: latest episode")
    parser.add_argument("--mode", choices=["rgb", "state"], default="rgb",
                        help="replay stored RGB or render restored MuJoCo states")
    parser.add_argument("--output", type=Path, default=None, help="optional MP4 output")
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="list saved episodes and exit")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    summary = validate_pyrite_dataset(args.dataset)
    print(json.dumps(summary, indent=2))
    root = zarr.open(str(args.dataset.expanduser()), mode="r")
    names = _episode_names(root)
    if args.list:
        print("\nepisode       samples  duration  success  final angle  termination")
        print("------------  -------  --------  -------  -----------  -----------")
        for episode_name in names:
            item = root["data"][episode_name]
            count = int(item.attrs.get("sample_count", len(item["rgb_0"])))
            hz = float(item.attrs.get("sample_hz", summary["sample_hz"]))
            success = "yes" if bool(item.attrs.get("success", False)) else "no"
            angle = float(item.attrs.get("final_book_angle_deg", np.nan))
            reason = str(item.attrs.get("termination_reason", "unknown"))
            print(
                f"{episode_name:12s}  {count:7d}  {count / hz:7.2f}s  "
                f"{success:7s}  {angle:10.1f}°  {reason}"
            )
        return
    if args.validate_only:
        return

    name = args.episode or names[-1]
    if name not in root["data"]:
        raise SystemExit(f"no {name!r}; available episodes: {names}")
    episode = root["data"][name]
    fps = float(episode.attrs["sample_hz"])

    env = None
    render = None
    if args.mode == "state":
        metadata = json.loads(episode.attrs["metadata_json"])
        env = _make_env(metadata)
        camera = metadata["camera"]
        env.set_arm_visual(camera["arm_view"])
        render = env.make_camera(
            width=int(episode["rgb_0"].shape[2]),
            height=int(episode["rgb_0"].shape[1]),
            quality=camera["render_quality"],
            azimuth=float(camera["azimuth"]),
            elevation=float(camera["elevation"]),
            distance=float(camera["distance"]),
            camera=camera["name"],
        )

    writer = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        height, width = episode["rgb_0"].shape[1:3]
        writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open {args.output} for video writing")

    state_spec = int(episode.attrs["mujoco_state_spec"])
    max_qpos_error = 0.0
    try:
        for index in range(len(episode["rgb_0"])):
            if args.mode == "rgb":
                rgb = np.asarray(episode["rgb_0"][index])
            else:
                state = np.asarray(episode["mujoco_state"][index], dtype=np.float64)
                mujoco.mj_setState(env.model.ptr, env.data.ptr, state, state_spec)
                mujoco.mj_forward(env.model.ptr, env.data.ptr)
                max_qpos_error = max(
                    max_qpos_error,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(env.data.qpos)
                                - np.asarray(episode["qpos"][index])
                            )
                        )
                    ),
                )
                rgb = render()
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            if writer is not None:
                writer.write(bgr)
            if not args.no_view:
                cv2.imshow(f"Pyrite FlipUp replay: {name}", bgr)
                key = cv2.waitKey(max(1, int(round(1000.0 / fps)))) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        if writer is not None:
            writer.release()
            print(f"wrote {args.output}")
        if env is not None:
            env.close()
        cv2.destroyAllWindows()
    if args.mode == "state":
        print(f"maximum restored qpos error: {max_qpos_error:.3e}")


if __name__ == "__main__":
    main()
