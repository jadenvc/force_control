"""Same rich visualization as render_rich_episode.py (nice view + side view with
nominal/virtual-target traces + felt-force graph), but replaying a REAL recorded
demonstration episode instead of a policy rollout -- so you can see exactly what
forces/targets the model is being trained to imitate, with zero policy inference.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import imageio
import numpy as np
import zarr

sys.path.insert(0, "/store/real/jvclark/PyriteUtility")
sys.path.insert(0, "/store/real/jvclark/force_control/teleop")

from render_rich_episode import draw_trace, force_panel, NOMINAL_COLOR, VT_COLOR
from flipup_teleop import FlipUpTeleop


def replay_episode(env, nice_cam, side_cam, cmd, vt, wrench, out_path):
    env._teleop_ready = True
    env.reset()
    init_pose7 = cmd[0]
    target = env.tool_pos.copy()
    for _ in range(int(4.0 / env.timestep)):
        delta = init_pose7[:3] - target
        d = np.linalg.norm(delta)
        if d > 0.25 * env.timestep:
            target = target + delta * (0.25 * env.timestep / d)
        else:
            target = init_pose7[:3].copy()
        env.step_task_space(env.target_pose7(target))
        if np.linalg.norm(env.tool_pos - init_pose7[:3]) < 1e-4 and np.linalg.norm(env.data.qvel[env.joint_dof_ids]) < 1e-2:
            break

    T = len(cmd)
    nice_frames, side_frames, angles = [], [], []
    for i in range(T):
        for _ in range(50):
            env.step_task_space(cmd[i])
        nice_frames.append(nice_cam().copy())
        side_frames.append(side_cam().copy())
        angles.append(env.book_angle_deg())

    W = 640
    frames_out = []
    for t in range(T):
        nice_img = cv2.resize(nice_frames[t], (W // 2, 240))
        side_full = side_frames[t].copy()
        draw_trace(side_full, side_cam.camera, cmd[: t + 1, :3], NOMINAL_COLOR)
        draw_trace(side_full, side_cam.camera, vt[: t + 1, :3], VT_COLOR)
        side_img = cv2.resize(side_full, (W // 2, 240))
        top_row = np.hstack([nice_img, side_img])
        force_img = force_panel(wrench[:, :3], t, W)
        composite = np.vstack([top_row, force_img])
        cv2.putText(composite, f"tick {t}  book_angle={angles[t]:.1f}",
                    (5, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        frames_out.append(composite)

    imageio.mimwrite(out_path, frames_out, fps=20, quality=6)
    return angles[-1], bool(angles[-1] < 15.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="/local/real/jvclark/mujoco_data/try_2_flipup_sim_20hz.zarr")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    root = zarr.open(args.dataset_path, mode="r")
    names = sorted(root["data"].group_keys(), key=lambda k: int(k.rsplit("_", 1)[-1]))
    names = names[args.start_index: args.start_index + args.num_episodes]

    env = FlipUpTeleop(seed=args.seed, settle_s=2.5)
    env.set_arm_visual(mode="hidden")
    nice_cam = env.make_camera(width=640, height=480, quality="fast", azimuth=90.0, elevation=-40.0, distance=0.75)
    side_cam = env.make_camera(width=640, height=480, quality="fast", azimuth=90.0, elevation=-10.0, distance=0.75)

    os.makedirs(args.out_dir, exist_ok=True)
    for name in names:
        ep = root["data"][name]
        cmd = ep["ts_pose_command_0"][:]
        vt = ep["ts_pose_virtual_target_0"][:]
        wrench = ep["wrench_0"][:]
        out_path = f"{args.out_dir}/{name}.mp4"
        final_angle, success = replay_episode(env, nice_cam, side_cam, cmd, vt, wrench, out_path)
        recorded_success = bool(dict(ep.attrs)["success"])
        print(f"[demo-replay] {name}: replay_final_angle={final_angle:.1f} replay_success={success} "
              f"recorded_success={recorded_success} len={len(cmd)}")


if __name__ == "__main__":
    main()
