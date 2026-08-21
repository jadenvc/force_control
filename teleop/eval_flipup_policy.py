"""Headless MuJoCo evaluation loop for trained FlipUp policies (ACP or vanilla DP).

Reuses the exact same env, wrench measurement, and rendering path as data
collection (FlipUpTeleop / pyrite_recorder.py) so the policy sees observations
identical in construction to what it was trained on. Two execution modes:

* Vanilla DP (9D action = commanded pose): the predicted pose is tracked
  directly by the env's existing fixed-gain task-space controller, unchanged
  from data collection (tool_kp=16000 isotropic, tool_rot_kp=3000).
* ACP (19D action = commanded pose + virtual target + scalar stiffness): the
  virtual target is tracked instead of the commanded pose, and the
  controller's translational stiffness is replaced every tick with a matrix
  built from the policy's learned stiffness along the (virtual target -
  commanded pose) direction in the tool frame, 5000 N/m along the two
  perpendicular directions, ported from
  UMI-FT/PyriteEnvSuites/env_runners/virtual_target_real_env_runner.py.
  Rotation is left rigid (unchanged tool_rot_kp), matching how the real
  controller's rotational stiffness is actually wired (force-controlled axes
  = translation only) and how the training labels were generated (ac_dim=3).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import cv2
import dill
import hydra
import imageio
import numpy as np
import torch
import zarr

sys.path.insert(0, "/store/real/jvclark/PyriteML")
sys.path.insert(0, "/store/real/jvclark/PyriteUtility")
sys.path.insert(0, "/store/real/jvclark/PyriteConfig")
sys.path.insert(0, "/store/real/jvclark/force_control/teleop")

import PyriteUtility.spatial_math.spatial_utilities as su  # noqa: E402
from tasks.common.common_type_conversions import (  # noqa: E402
    sparse_obs_to_obs_sample,
    action9_postprocess,
    action19_postprocess,
)
from flipup_teleop import FlipUpTeleop  # noqa: E402


# --------------------------------------------------------------------------- policy

def load_policy(ckpt_path: str, device: str):
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = cfg.policy.num_inference_steps
    policy.eval().to(device)
    return policy, cfg.task.shape_meta


# --------------------------------------------------------------------- init poses

def load_init_position_bank(dataset_path: str):
    """Bootstrap bank of real episode-start tool positions, for matching the
    init-pose distribution actually seen during data collection. Returns
    (positions, episode_names) so a sampled init pose can be traced back to
    the demonstration it came from, for trajectory comparison plots."""
    root = zarr.open(dataset_path, mode="r")
    names = sorted(root["data"].group_keys(), key=lambda k: int(k.rsplit("_", 1)[-1]))
    positions = np.stack([root["data"][n]["ts_pose_fb_0"][0][:3] for n in names])
    return positions, names


# --------------------------------------------------------------------- obs history

@dataclass
class ObsHistory:
    """Rolling raw-sample buffers, one entry per dataset tick (20Hz)."""

    pose_len: int  # (horizon-1)*down_sample_steps + 1 for rgb/pos/rot
    wrench_len: int  # same, for wrench (0 if policy has no wrench input)
    rgb: list = field(default_factory=list)
    pose9: list = field(default_factory=list)  # pos(3) + rot6d(6)
    wrench: list = field(default_factory=list)

    def bootstrap(self, rgb, pose9, wrench):
        self.rgb = [rgb.copy() for _ in range(self.pose_len)]
        self.pose9 = [pose9.copy() for _ in range(self.pose_len)]
        if self.wrench_len > 0:
            self.wrench = [wrench.copy() for _ in range(self.wrench_len)]

    def push(self, rgb, pose9, wrench):
        self.rgb.append(rgb.copy())
        self.pose9.append(pose9.copy())
        self.rgb = self.rgb[-self.pose_len:]
        self.pose9 = self.pose9[-self.pose_len:]
        if self.wrench_len > 0:
            self.wrench.append(wrench.copy())
            self.wrench = self.wrench[-self.wrench_len:]


def build_obs_sparse(hist: ObsHistory, shape_meta: dict) -> dict:
    sample_cfg = shape_meta["sample"]["obs"]["sparse"]
    pose_stride = sample_cfg["robot0_eef_pos"]["down_sample_steps"]
    obs_sparse = {
        "rgb_0": np.stack(hist.rgb[::pose_stride]),
        "robot0_eef_pos": np.stack([p[:3] for p in hist.pose9[::pose_stride]]),
        "robot0_eef_rot_axis_angle": np.stack([p[3:] for p in hist.pose9[::pose_stride]]),
    }
    if "robot0_eef_wrench" in sample_cfg:
        wrench_stride = sample_cfg["robot0_eef_wrench"]["down_sample_steps"]
        obs_sparse["robot0_eef_wrench"] = np.stack(hist.wrench[::wrench_stride])
    return obs_sparse


def tool_pose9(env) -> np.ndarray:
    pose7 = np.concatenate([env.tool_pos, env.tool_quat])
    return su.SE3_to_pose9(su.pose7_to_SE3(pose7))


# --------------------------------------------------------------------- compliance

def clamp_spring_force(tool_pos, target_pos, K_world_pos, max_force_n):
    """Cap the translational spring force implied by (K_world_pos @ position_error)
    to max_force_n, by scaling the commanded displacement down (direction
    preserved) -- the safety clamp the real admittance_controller.cpp applies
    (max_spring_force_magnitude) but that this MuJoCo port had been missing,
    letting a single bad replan's position jump imply hundreds of newtons."""
    err = target_pos - tool_pos
    force = K_world_pos @ err
    force_mag = np.linalg.norm(force)
    if force_mag <= max_force_n or force_mag < 1e-9:
        return target_pos
    return tool_pos + err * (max_force_n / force_mag)


def build_tool_stiffness(target_pose7, vt_pose7, stiffness_scalar, default_perp=5000.0):
    """Port of virtual_target_real_env_runner.py's compliance-direction /
    6x6-stiffness construction, restricted to the 3x3 translational block
    (rotation is left rigid -- see module docstring)."""
    direction = vt_pose7[:3] - target_pose7[:3]
    norm = np.linalg.norm(direction)
    if norm < 0.001:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / norm
    X = direction
    cross_z = np.cross(X, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(cross_z) < 1e-6:
        cross_z = np.cross(X, np.array([0.0, 1.0, 0.0]))
    Y = cross_z / np.linalg.norm(cross_z)
    Z = np.cross(X, Y)
    S = np.stack([X, Y, Z], axis=1)  # columns
    M = np.diag([float(stiffness_scalar), default_perp, default_perp])
    K = S @ M @ S.T  # S is orthonormal, so S^-1 == S.T
    return K


# --------------------------------------------------------------------- rollout

@dataclass
class EvalConfig:
    ckpt: str
    dataset_path: str
    num_episodes: int
    replan_every_ticks: int
    max_ticks: int
    device: str
    seed: int
    cam_azimuth: float
    cam_elevation: float
    cam_distance: float
    default_perp_stiffness: float = 5000.0
    acp_track_nominal: bool = False
    fixed_compliant_stiffness: float = None
    compliant_stiffness_clip: tuple = (500.0, 20000.0)
    max_spring_force_n: float = 50.0
    translational_damping: float = 250.0


def run_episode(env, policy, shape_meta, action_dim, init_pos, cfg: EvalConfig, obs_render_fn,
                 video_render_fn=None, video_path=None):
    env._teleop_ready = True
    env.reset()

    # Slew from the settled/tared tool_home to a sampled init position (matching
    # how the real dataset's ts_pose_fb_0[0] reflects wherever the human's
    # cursor was when they pressed start, not the settle target itself).
    # No re-tare: the real collection didn't re-tare after this move either.
    target = env.tool_pos.copy()
    settle_speed = 0.25
    for _ in range(int(4.0 / env.timestep)):
        delta = init_pos - target
        distance = np.linalg.norm(delta)
        if distance > settle_speed * env.timestep:
            target = target + delta * (settle_speed * env.timestep / distance)
        else:
            target = init_pos.copy()
        env.step_task_space(env.target_pose7(target))
        if (
            np.linalg.norm(env.tool_pos - init_pos) < 1e-4
            and np.linalg.norm(env.data.qvel[env.joint_dof_ids]) < 1e-2
        ):
            break

    base_kp = env.task_space_kp.copy()  # data-collection defaults, restored for DP / between ACP ticks
    # Translational Cartesian damping defaults to 0 in this env (only rotation
    # has damping, 90 N*m*s/rad) -- fine for a human-teleoperated, naturally
    # damped input, but a stiff-with-zero-damping spring rings/bounces whenever
    # it's displaced by real contact or a policy's discontinuous target jump.
    # Add translational damping so it behaves closer to critically damped.
    env.task_space_cartesian_kd[:3] = cfg.translational_damping

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

    hist = ObsHistory(pose_len=pose_len, wrench_len=wrench_len)
    video_frames = [] if video_path is not None else None

    def capture_raw_sample():
        # Observation camera must match training's viewpoint exactly -- the model
        # has never seen any other angle, so this is NOT interchangeable with
        # whatever camera is used for human-readable video.
        obs_full_res = obs_render_fn()
        if video_frames is not None:
            video_frames.append(video_render_fn().copy())
        rgb = cv2.resize(obs_full_res, (224, 224), interpolation=cv2.INTER_AREA)
        pose9 = tool_pose9(env)
        wrench = env.wrist_wrench(frame="tool") if has_wrench else np.zeros(6)
        return rgb, pose9, wrench

    rgb0, pose90, wrench0 = capture_raw_sample()
    hist.bootstrap(rgb0, pose90, wrench0)

    active_targets = None  # list of pose7, len action_horizon
    active_vts = None
    active_stiffness = None
    ticks_since_replan = cfg.replan_every_ticks  # force an immediate replan

    device = cfg.device
    success = False
    airborne_ticks = 0
    tool_xyz_trace = [env.tool_pos.copy()]
    for tick in range(cfg.max_ticks):
        if ticks_since_replan >= cfg.replan_every_ticks:
            obs_sparse = build_obs_sparse(hist, shape_meta)
            obs_sample_np, base_SE3_WT = sparse_obs_to_obs_sample(
                obs_sparse, shape_meta, reshape_mode="reshape", id_list=[0]
            )
            obs_torch = {
                k: torch.from_numpy(np.asarray(v, dtype=np.float32)).unsqueeze(0).to(device)
                for k, v in obs_sample_np.items()
            }
            with torch.no_grad():
                result = policy.predict_action({"sparse": obs_torch})
            raw_action = result["sparse"][0].detach().cpu().numpy()

            if action_dim == 9:
                se3_list = action9_postprocess(raw_action, base_SE3_WT, id_list=[0])
                active_targets = [su.SE3_to_pose7(m) for m in se3_list[0]]
                active_vts = active_targets
                active_stiffness = [None] * action_horizon
            else:
                se3_list, se3_vt_list, stiffness_list = action19_postprocess(
                    raw_action, base_SE3_WT, id_list=[0]
                )
                active_targets = [su.SE3_to_pose7(m) for m in se3_list[0]]
                active_vts = [su.SE3_to_pose7(m) for m in se3_vt_list[0]]
                active_stiffness = list(stiffness_list[0])
            ticks_since_replan = 0

        waypoint_idx = min(ticks_since_replan // action_stride, action_horizon - 2)
        alpha = (ticks_since_replan % action_stride) / action_stride
        target_pose7 = su.pose7_interp(
            active_targets[waypoint_idx], active_targets[waypoint_idx + 1], np.array([alpha])
        )[0]

        if action_dim == 19 and not cfg.acp_track_nominal:
            vt_pose7 = active_vts[waypoint_idx]
            if cfg.fixed_compliant_stiffness is not None:
                stiffness_scalar = cfg.fixed_compliant_stiffness
            else:
                stiffness_scalar = np.clip(
                    active_stiffness[waypoint_idx],
                    cfg.compliant_stiffness_clip[0], cfg.compliant_stiffness_clip[1],
                )
            K_tool = build_tool_stiffness(
                active_targets[waypoint_idx], vt_pose7, stiffness_scalar,
                default_perp=cfg.default_perp_stiffness,
            )
            R_world_tool = su.pose7_to_SE3(np.concatenate([env.tool_pos, env.tool_quat]))[:3, :3]
            K_world = R_world_tool @ K_tool @ R_world_tool.T
            env.task_space_kp = base_kp.copy()
            env.task_space_kp[:3, :3] = K_world
            drive_target = su.pose7_interp(
                active_vts[waypoint_idx], active_vts[waypoint_idx + 1], np.array([alpha])
            )[0]
            active_K_pos = K_world
        else:
            # DP, or ACP with --acp-track-nominal: track the commanded/nominal
            # pose directly with the env's fixed data-collection stiffness,
            # ignoring the virtual-target/stiffness heads entirely. Isolates
            # ACP's position-prediction quality from its compliance-execution
            # logic.
            env.task_space_kp = base_kp
            drive_target = target_pose7
            active_K_pos = base_kp[:3, :3]

        # Cap the spring force a large VT/target jump (e.g. right after a fresh,
        # independently-sampled replan that doesn't land near the tool's actual
        # current position) would otherwise imply -- matching the safety clamp
        # the real admittance_controller.cpp applies (max_spring_force_magnitude)
        # that this port had been missing.
        clamped_pos = clamp_spring_force(env.tool_pos, drive_target[:3], active_K_pos, cfg.max_spring_force_n)
        drive_target = np.concatenate([clamped_pos, drive_target[3:]])

        for _ in range(50):  # 50 physics ticks = 1 dataset tick at 20Hz/1000Hz
            env.step_task_space(drive_target)

        ticks_since_replan += 1
        rgb, pose9, wrench = capture_raw_sample()
        hist.push(rgb, pose9, wrench)
        tool_xyz_trace.append(env.tool_pos.copy())
        if env.data.ncon == 0:
            airborne_ticks += 1

        if env.success():
            success = True
            break

    if video_frames is not None:
        imageio.mimwrite(video_path, video_frames, fps=20, quality=6)

    return {
        "success": success,
        "final_book_angle_deg": float(env.book_angle_deg()),
        "ticks": tick + 1,
        "airborne_ticks": airborne_ticks,
        "tool_xyz_trace": np.stack(tool_xyz_trace),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset-path", default="/local/real/jvclark/mujoco_data/try_2_flipup_sim_20hz.zarr")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--replan-every-ticks", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--cam-azimuth", type=float, default=90.0,
                         help="VIDEO camera only -- never used for the policy's actual observation")
    parser.add_argument("--cam-elevation", type=float, default=-40.0,
                         help="VIDEO camera only")
    parser.add_argument("--cam-distance", type=float, default=0.75,
                         help="VIDEO camera only")
    parser.add_argument("--arm-visual", default="hidden", choices=["full", "hidden", "ghost"])
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--default-perp-stiffness", type=float, default=5000.0,
                         help="ACP only: N/m stiffness along the two directions perpendicular "
                              "to the compliance direction")
    parser.add_argument("--acp-track-nominal", action="store_true",
                         help="ACP only: bypass the compliance controller entirely -- track the "
                              "commanded/nominal pose with fixed data-collection stiffness, "
                              "same as DP, ignoring the virtual-target/stiffness heads")
    parser.add_argument("--fixed-compliant-stiffness", type=float, default=None,
                         help="ACP only: override the predicted compliant-direction stiffness "
                              "with this constant value instead of using the model's prediction")
    parser.add_argument("--compliant-stiffness-clip", type=float, nargs=2, default=[500.0, 20000.0],
                         help="ACP only: clip range applied to the predicted compliant-direction "
                              "stiffness (ignored if --fixed-compliant-stiffness is set)")
    parser.add_argument("--max-spring-force-n", type=float, default=50.0,
                         help="Cap on the translational spring force implied by the commanded "
                              "position error x active stiffness, matching the real admittance "
                              "controller's max_spring_force_magnitude safety clamp")
    parser.add_argument("--translational-damping", type=float, default=250.0,
                         help="Cartesian translational damping (N*s/m), 0 in the base env -- "
                              "raise to reduce bouncing/oscillation on contact or target jumps")
    parser.add_argument("--trajectory-dir", default=None,
                         help="if set, saves one .npz per episode with the tool xyz trace and "
                              "the source demo episode name, for policy-vs-demo plots")
    args = parser.parse_args()

    cfg = EvalConfig(
        ckpt=args.ckpt,
        dataset_path=args.dataset_path,
        num_episodes=args.num_episodes,
        replan_every_ticks=args.replan_every_ticks,
        max_ticks=args.max_ticks,
        device=args.device,
        seed=args.seed,
        cam_azimuth=args.cam_azimuth,
        cam_elevation=args.cam_elevation,
        cam_distance=args.cam_distance,
        default_perp_stiffness=args.default_perp_stiffness,
        acp_track_nominal=args.acp_track_nominal,
        fixed_compliant_stiffness=args.fixed_compliant_stiffness,
        compliant_stiffness_clip=tuple(args.compliant_stiffness_clip),
        max_spring_force_n=args.max_spring_force_n,
        translational_damping=args.translational_damping,
    )

    policy, shape_meta = load_policy(cfg.ckpt, cfg.device)
    action_dim = shape_meta["action"]["shape"][0]
    assert action_dim in (9, 19), f"unsupported action_dim {action_dim}"
    print(f"[eval] loaded policy, action_dim={action_dim}")

    init_bank, init_bank_names = load_init_position_bank(cfg.dataset_path)
    print(f"[eval] init-pose bank: {len(init_bank)} real episode starts")

    env = FlipUpTeleop(seed=cfg.seed, settle_s=2.5)
    env.set_arm_visual(mode=args.arm_visual)
    # Observation camera: fixed to match the training data's viewpoint exactly
    # (azimuth 0/15, elevation -25, distance 0.75 -- see episode metadata in the
    # dataset). This is what the policy actually sees and must never be changed
    # for viewing convenience.
    obs_render_fn = env.make_camera(
        width=640, height=480, quality="fast",
        azimuth=0.0, elevation=-25.0, distance=0.75,
    )
    # Video camera: whatever angle is convenient for a human to watch. Fully
    # decoupled from the observation camera above.
    video_render_fn = env.make_camera(
        width=640, height=480, quality="fast",
        azimuth=cfg.cam_azimuth, elevation=cfg.cam_elevation, distance=cfg.cam_distance,
    )

    if args.video_dir:
        import os
        os.makedirs(args.video_dir, exist_ok=True)
    if args.trajectory_dir:
        import os
        os.makedirs(args.trajectory_dir, exist_ok=True)

    rng = np.random.RandomState(args.eval_seed)
    results = []
    t0 = time.time()
    for ep in range(cfg.num_episodes):
        bank_idx = rng.randint(len(init_bank))
        init_pos = init_bank[bank_idx]
        demo_name = init_bank_names[bank_idx]
        video_path = None
        if args.video_dir and ep < args.video_episodes:
            video_path = f"{args.video_dir}/episode_{ep}.mp4"
        res = run_episode(env, policy, shape_meta, action_dim, init_pos, cfg, obs_render_fn,
                           video_render_fn=video_render_fn, video_path=video_path)
        results.append(res)
        print(f"[eval] episode {ep}: success={res['success']} "
              f"final_angle={res['final_book_angle_deg']:.1f} ticks={res['ticks']} "
              f"airborne_ticks={res['airborne_ticks']} demo={demo_name} "
              f"(elapsed {time.time()-t0:.0f}s)")
        if args.trajectory_dir:
            np.savez(
                f"{args.trajectory_dir}/episode_{ep}.npz",
                tool_xyz_trace=res["tool_xyz_trace"],
                demo_episode_name=demo_name,
                success=res["success"],
                final_book_angle_deg=res["final_book_angle_deg"],
                replan_every_ticks=cfg.replan_every_ticks,
                default_perp_stiffness=cfg.default_perp_stiffness,
                ckpt=cfg.ckpt,
            )

    success_rate = float(np.mean([r["success"] for r in results]))
    mean_angle = float(np.mean([r["final_book_angle_deg"] for r in results]))
    print(f"\n[eval] SUMMARY ckpt={cfg.ckpt}")
    print(f"[eval] success_rate={success_rate:.2f} mean_final_angle_deg={mean_angle:.1f} "
          f"n={cfg.num_episodes}")


if __name__ == "__main__":
    main()
