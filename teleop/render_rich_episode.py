"""Rich per-episode rollout visualization: nice-view + side-view (with virtual
target / nominal target traces overlaid, projected into the side camera) on
top, felt tool-frame force (Fx, Fy, Fz) graph on the bottom.

Colors: nominal target trace = categorical slot 1 (blue #2a78d6), virtual
target trace = slot 2 (orange #eb6834) -- consistent with plot_trajectories.py.
Force channels use slots 3/4/5 (aqua/yellow/magenta) to avoid colliding with
the trace colors in the same composite frame.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/store/real/jvclark/PyriteML")
sys.path.insert(0, "/store/real/jvclark/PyriteUtility")
sys.path.insert(0, "/store/real/jvclark/PyriteConfig")
sys.path.insert(0, "/store/real/jvclark/force_control/teleop")

import eval_flipup_policy as E
import PyriteUtility.spatial_math.spatial_utilities as su
from tasks.common.common_type_conversions import sparse_obs_to_obs_sample, action9_postprocess, action19_postprocess

# Everything here is composited with MuJoCo/dm_control renders, which are
# natively RGB -- so all cv2 drawing below uses RGB tuples directly (no BGR
# conversion, no final channel flip) to stay consistent with them.
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

NOMINAL_COLOR = hex_to_rgb("#2a78d6")
VT_COLOR = hex_to_rgb("#eb6834")
FX_COLOR = hex_to_rgb("#1baf7a")
FY_COLOR = hex_to_rgb("#eda100")
FZ_COLOR = hex_to_rgb("#e87ba4")
GRID_COLOR = (216, 216, 211)
TEXT_COLOR = (20, 20, 20)


def project_points(camera, points_world):
    hom = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    proj = (camera.matrix @ hom.T).T
    px = proj[:, 0] / proj[:, 2]
    py = proj[:, 1] / proj[:, 2]
    return np.stack([px, py], axis=1)


def draw_trace(img, camera, points_world, color, thickness=2):
    if len(points_world) < 2:
        return img
    pts2d = project_points(camera, np.array(points_world)).astype(np.int32)
    cv2.polylines(img, [pts2d], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.circle(img, tuple(pts2d[-1]), 5, color, -1, lineType=cv2.LINE_AA)
    return img


def _nice_scale(max_abs):
    """Round a max-abs force value up to a clean gridline scale."""
    if max_abs < 1e-6:
        return 10.0
    for step in (5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000):
        if max_abs <= step:
            return float(step)
    # beyond the table: round up to the nearest 500
    return float(np.ceil(max_abs / 500.0) * 500.0)


def force_panel(wrench_trace, tick, width, height=220, scale_n=None):
    wt_full = np.asarray(wrench_trace)
    if scale_n is None:
        scale_n = _nice_scale(np.abs(wt_full).max())
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    margin_l, margin_r = 50, 10
    plot_w = width - margin_l - margin_r
    n = len(wrench_trace)
    cv2.line(canvas, (margin_l, height // 2), (width - margin_r, height // 2), GRID_COLOR, 1)
    for frac, label in [(0.0, f"+{scale_n:.0f}N"), (1.0, f"-{scale_n:.0f}N"), (0.5, "0N")]:
        y = int(10 + frac * (height - 20))
        cv2.line(canvas, (margin_l, y), (width - margin_r, y), GRID_COLOR, 1)
        cv2.putText(canvas, label, (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, 1, cv2.LINE_AA)

    def to_xy(i, val):
        x = margin_l + int(plot_w * i / max(n - 1, 1))
        y = int(10 + (height - 20) * (0.5 - val / (2 * scale_n)))
        y = np.clip(y, 10, height - 10)
        return x, y

    wt = np.array(wrench_trace[: tick + 1])
    for ch, color, label in [(0, FX_COLOR, "Fx"), (1, FY_COLOR, "Fy"), (2, FZ_COLOR, "Fz")]:
        pts = np.array([to_xy(i, wt[i, ch]) for i in range(len(wt))], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
    cur_x = margin_l + int(plot_w * tick / max(n - 1, 1))
    cv2.line(canvas, (cur_x, 5), (cur_x, height - 5), (120, 120, 120), 1)
    for i, (label, color) in enumerate([("Fx", FX_COLOR), ("Fy", FY_COLOR), ("Fz", FZ_COLOR)]):
        cv2.putText(canvas, label, (margin_l + 10 + i * 50, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def run_rich_episode(env, policy, shape_meta, action_dim, init_pos, cfg, nice_cam, side_cam, obs_cam, out_path,
                      collision_cam=None):
    env._teleop_ready = True
    env.reset()
    target = env.tool_pos.copy()
    for _ in range(int(4.0 / env.timestep)):
        delta = init_pos - target
        distance = np.linalg.norm(delta)
        if distance > 0.25 * env.timestep:
            target = target + delta * (0.25 * env.timestep / distance)
        else:
            target = init_pos.copy()
        env.step_task_space(env.target_pose7(target))
        if np.linalg.norm(env.tool_pos - init_pos) < 1e-4 and np.linalg.norm(env.data.qvel[env.joint_dof_ids]) < 1e-2:
            break

    base_kp = env.task_space_kp.copy()
    env.task_space_cartesian_kd[:3] = getattr(cfg, "translational_damping", 250.0)
    pose_stride = shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["down_sample_steps"]
    pose_horizon = shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["horizon"]
    pose_len = (pose_horizon - 1) * pose_stride + 1
    has_wrench = "robot0_eef_wrench" in shape_meta["sample"]["obs"]["sparse"]
    wrench_len = 0
    if has_wrench:
        wc = shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]
        wrench_len = (wc["horizon"] - 1) * wc["down_sample_steps"] + 1
    action_stride = shape_meta["sample"]["action"]["sparse"]["down_sample_steps"]
    action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]

    hist = E.ObsHistory(pose_len=pose_len, wrench_len=wrench_len)
    nice_frames, side_frames, collision_frames = [], [], []
    nominal_trace, vt_trace, wrench_trace = [], [], []

    def capture():
        obs_full = obs_cam().copy()
        nice_frames.append(nice_cam().copy())
        side_frames.append(side_cam().copy())
        if collision_cam is not None:
            # Collision-only render (quality="collision" -> only geom group 3,
            # the actual contact meshes), re-centered on the tool every tick so
            # the small fingertip/book-edge contact stays framed as it moves.
            collision_cam.camera.set_pose(env.tool_pos.copy(), 0.12, 90.0, -10.0)
            collision_frames.append(collision_cam().copy())
        rgb = cv2.resize(obs_full, (224, 224), interpolation=cv2.INTER_AREA)
        pose9 = E.tool_pose9(env)
        # Always compute the real felt wrench for the force graph, regardless of
        # whether this policy actually consumes it as an observation (DP doesn't,
        # but we still want to show what the tool physically feels).
        wrench = env.wrist_wrench(frame="tool")
        return rgb, pose9, wrench

    rgb0, pose90, wrench0 = capture()
    hist.bootstrap(rgb0, pose90, wrench0)
    nominal_trace.append(env.tool_pos.copy())
    vt_trace.append(env.tool_pos.copy())
    wrench_trace.append(wrench0[:3].copy())

    active_targets = active_vts = active_stiffness = None
    ticks_since_replan = cfg.replan_every_ticks
    success = False
    for tick in range(cfg.max_ticks):
        if ticks_since_replan >= cfg.replan_every_ticks:
            obs_sparse = E.build_obs_sparse(hist, shape_meta)
            obs_sample_np, base_SE3_WT = sparse_obs_to_obs_sample(obs_sparse, shape_meta, reshape_mode="reshape", id_list=[0])
            obs_torch = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)).unsqueeze(0).to(cfg.device) for k, v in obs_sample_np.items()}
            with torch.no_grad():
                result = policy.predict_action({"sparse": obs_torch})
            raw_action = result["sparse"][0].detach().cpu().numpy()
            if action_dim == 9:
                se3_list = action9_postprocess(raw_action, base_SE3_WT, id_list=[0])
                active_targets = [su.SE3_to_pose7(m) for m in se3_list[0]]
                active_vts = active_targets
                active_stiffness = [None] * action_horizon
            else:
                se3_list, se3_vt_list, stiffness_list = action19_postprocess(raw_action, base_SE3_WT, id_list=[0])
                active_targets = [su.SE3_to_pose7(m) for m in se3_list[0]]
                active_vts = [su.SE3_to_pose7(m) for m in se3_vt_list[0]]
                active_stiffness = list(stiffness_list[0])
            ticks_since_replan = 0

        waypoint_idx = min(ticks_since_replan // action_stride, action_horizon - 2)
        alpha = (ticks_since_replan % action_stride) / action_stride
        nominal_pose7 = su.pose7_interp(active_targets[waypoint_idx], active_targets[waypoint_idx + 1], np.array([alpha]))[0]

        if action_dim == 19:
            vt_pose7 = su.pose7_interp(active_vts[waypoint_idx], active_vts[waypoint_idx + 1], np.array([alpha]))[0]
            stiffness_scalar = np.clip(active_stiffness[waypoint_idx], 500.0, 20000.0)
            K_tool = E.build_tool_stiffness(active_targets[waypoint_idx], vt_pose7, stiffness_scalar, default_perp=cfg.default_perp_stiffness)
            R_world_tool = su.pose7_to_SE3(np.concatenate([env.tool_pos, env.tool_quat]))[:3, :3]
            K_world = R_world_tool @ K_tool @ R_world_tool.T
            env.task_space_kp = base_kp.copy()
            env.task_space_kp[:3, :3] = K_world
            drive_target = vt_pose7
            active_K_pos = K_world
        else:
            vt_pose7 = nominal_pose7
            env.task_space_kp = base_kp
            drive_target = nominal_pose7
            active_K_pos = base_kp[:3, :3]

        # Cap the spring force a large VT/target jump would otherwise imply --
        # see clamp_spring_force's docstring in eval_flipup_policy.py.
        clamped_pos = E.clamp_spring_force(env.tool_pos, drive_target[:3], active_K_pos,
                                            getattr(cfg, "max_spring_force_n", 50.0))
        drive_target = np.concatenate([clamped_pos, drive_target[3:]])
        vt_pose7 = np.concatenate([clamped_pos, vt_pose7[3:]]) if action_dim == 19 else vt_pose7

        for _ in range(50):
            env.step_task_space(drive_target)

        ticks_since_replan += 1
        rgb, pose9, wrench = capture()
        hist.push(rgb, pose9, wrench)
        nominal_trace.append(nominal_pose7[:3].copy())
        vt_trace.append(vt_pose7[:3].copy())
        wrench_trace.append(wrench[:3].copy())

        if env.success():
            success = True
            break

    final_angle = float(env.book_angle_deg())

    # composite video
    has_collision = len(collision_frames) > 0
    ncols = 3 if has_collision else 2
    col_w = 300 if has_collision else 320
    W = col_w * ncols
    frames_out = []
    for t in range(len(nice_frames)):
        nice_img = cv2.resize(nice_frames[t], (col_w, 240))
        # draw traces on the ORIGINAL-res side frame before resize, for correct projection
        side_full = side_frames[t].copy()
        draw_trace(side_full, side_cam.camera, nominal_trace[: t + 1], NOMINAL_COLOR)
        draw_trace(side_full, side_cam.camera, vt_trace[: t + 1], VT_COLOR)
        side_img = cv2.resize(side_full, (col_w, 240))
        cols = [nice_img, side_img]
        if has_collision:
            collision_img = cv2.resize(collision_frames[t], (col_w, 240))
            cv2.putText(collision_img, "collision mesh (group 3 only)", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cols.append(collision_img)
        top_row = np.hstack(cols)
        force_img = force_panel(wrench_trace, t, W)
        composite = np.vstack([top_row, force_img])
        cv2.putText(composite, f"tick {t}  angle={env.book_angle_deg() if t == len(nice_frames)-1 else ''}",
                    (5, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        frames_out.append(composite)

    import imageio
    imageio.mimwrite(out_path, frames_out, fps=20, quality=6)
    return {"success": success, "final_book_angle_deg": final_angle, "ticks": len(nice_frames) - 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset-path", default="/local/real/jvclark/mujoco_data/try_2_flipup_sim_20hz.zarr")
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--replan-every-ticks", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--default-perp-stiffness", type=float, default=5000.0)
    parser.add_argument("--max-spring-force-n", type=float, default=50.0)
    parser.add_argument("--translational-damping", type=float, default=250.0)
    parser.add_argument("--show-collision-mesh", action="store_true",
                         help="add a third panel showing only the raw collision geometry "
                              "(geom group 3), zoomed and following the tool")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    class Cfg:
        pass
    cfg = Cfg()
    cfg.replan_every_ticks = args.replan_every_ticks
    cfg.max_ticks = args.max_ticks
    cfg.device = args.device
    cfg.default_perp_stiffness = args.default_perp_stiffness
    cfg.max_spring_force_n = args.max_spring_force_n
    cfg.translational_damping = args.translational_damping

    policy, shape_meta = E.load_policy(args.ckpt, args.device)
    action_dim = shape_meta["action"]["shape"][0]
    init_bank, init_bank_names = E.load_init_position_bank(args.dataset_path)

    env = E.FlipUpTeleop(seed=args.seed, settle_s=2.5)
    env.set_arm_visual(mode="hidden")
    obs_cam = env.make_camera(width=640, height=480, quality="fast", azimuth=0.0, elevation=-25.0, distance=0.75)
    nice_cam = env.make_camera(width=640, height=480, quality="fast", azimuth=90.0, elevation=-40.0, distance=0.75)
    side_cam = env.make_camera(width=640, height=480, quality="fast", azimuth=90.0, elevation=-10.0, distance=0.75)
    collision_cam = None
    if args.show_collision_mesh:
        collision_cam = env.make_camera(width=640, height=480, quality="collision",
                                         azimuth=90.0, elevation=-10.0, distance=0.12)

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.eval_seed)
    for ep in range(args.num_episodes):
        bank_idx = rng.randint(len(init_bank))
        init_pos = init_bank[bank_idx]
        out_path = f"{args.out_dir}/episode_{ep}.mp4"
        res = run_rich_episode(env, policy, shape_meta, action_dim, init_pos, cfg, nice_cam, side_cam, obs_cam, out_path,
                               collision_cam=collision_cam)
        print(f"[rich] episode {ep}: success={res['success']} final_angle={res['final_book_angle_deg']:.1f} "
              f"ticks={res['ticks']} demo={init_bank_names[bank_idx]}")


if __name__ == "__main__":
    main()
