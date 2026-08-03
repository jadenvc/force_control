"""Teacher-forced (open-loop, ground-truth-history) prediction error diagnostic.

At each valid raw tick t in a REAL recorded episode, build the exact observation
sample a training-time query at t would have produced (real rgb/pose/wrench
history, no simulation involved), run predict_action, and compare the predicted
action horizon against the REAL recorded action labels at the same indices
sampler.py would have used. This isolates single-step prediction quality from
closed-loop compounding error: since the model never sees its own past outputs
here (every step is fed real ground-truth history), large errors here mean the
model itself doesn't know what to do even under ideal conditions, whereas small
errors here despite closed-loop rollout failure would point at compounding
drift instead.
"""
from __future__ import annotations

import argparse
import sys

import dill
import hydra
import numpy as np
import torch
import zarr

sys.path.insert(0, "/store/real/jvclark/PyriteML")
sys.path.insert(0, "/store/real/jvclark/PyriteUtility")
sys.path.insert(0, "/store/real/jvclark/PyriteConfig")

import PyriteUtility.spatial_math.spatial_utilities as su  # noqa: E402
from tasks.common.common_type_conversions import (  # noqa: E402
    sparse_obs_to_obs_sample,
    action9_postprocess,
    action19_postprocess,
)


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


def pose7_to_pose9(pose7: np.ndarray) -> np.ndarray:
    return su.SE3_to_pose9(su.pose7_to_SE3(pose7))


def rot_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R_rel = R_pred.T @ R_gt
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


def evaluate_episode(policy, shape_meta, action_dim, episode, device):
    fb = episode["ts_pose_fb_0"][:]
    cmd = episode["ts_pose_command_0"][:]
    wrench = episode["wrench_0"][:] if action_dim == 19 else None
    vt = episode["ts_pose_virtual_target_0"][:] if action_dim == 19 else None
    stiffness = episode["stiffness_0"][:] if action_dim == 19 else None
    T = len(fb)

    pose_cfg = shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]
    pose_horizon, pose_stride = pose_cfg["horizon"], pose_cfg["down_sample_steps"]
    pose_lookback = (pose_horizon - 1) * pose_stride

    wrench_lookback = 0
    if action_dim == 19:
        wc = shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]
        wrench_lookback = (wc["horizon"] - 1) * wc["down_sample_steps"]

    action_cfg = shape_meta["sample"]["action"]["sparse"]
    action_horizon, action_stride = action_cfg["horizon"], action_cfg["down_sample_steps"]
    lookahead = (action_horizon - 1) * action_stride

    t_start = max(pose_lookback, wrench_lookback)
    t_end = T - lookahead - 1
    if t_end <= t_start:
        return None

    pose9_all = np.stack([pose7_to_pose9(fb[i]) for i in range(T)])
    rgb_all = episode["rgb_0"]

    pos_errs = np.zeros((t_end - t_start, action_horizon))
    rot_errs = np.zeros((t_end - t_start, action_horizon))
    vt_pos_errs = np.zeros((t_end - t_start, action_horizon)) if action_dim == 19 else None
    stiff_errs = np.zeros((t_end - t_start, action_horizon)) if action_dim == 19 else None

    for row, t in enumerate(range(t_start, t_end)):
        pos_idx = np.arange(t - pose_lookback, t + 1, pose_stride)
        obs_sparse = {
            "rgb_0": np.stack([rgb_all[i] for i in pos_idx]),
            "robot0_eef_pos": pose9_all[pos_idx, :3],
            "robot0_eef_rot_axis_angle": pose9_all[pos_idx, 3:],
        }
        if action_dim == 19:
            w_idx = np.arange(t - wrench_lookback, t + 1, wc["down_sample_steps"])
            obs_sparse["robot0_eef_wrench"] = wrench[w_idx]

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

        act_idx = np.arange(t, t + lookahead + 1, action_stride)
        gt_cmd_SE3 = su.pose7_to_SE3(cmd[act_idx])

        if action_dim == 9:
            se3_list = action9_postprocess(raw_action, base_SE3_WT, id_list=[0])
            pred_SE3 = se3_list[0]
        else:
            se3_list, se3_vt_list, stiffness_list = action19_postprocess(
                raw_action, base_SE3_WT, id_list=[0]
            )
            pred_SE3 = se3_list[0]
            gt_vt_SE3 = su.pose7_to_SE3(vt[act_idx])
            pred_vt_SE3 = se3_vt_list[0]
            vt_pos_errs[row] = np.linalg.norm(
                pred_vt_SE3[:, :3, 3] - gt_vt_SE3[:, :3, 3], axis=-1
            )
            stiff_errs[row] = np.abs(np.asarray(stiffness_list[0]) - stiffness[act_idx])

        pos_errs[row] = np.linalg.norm(pred_SE3[:, :3, 3] - gt_cmd_SE3[:, :3, 3], axis=-1)
        for h in range(action_horizon):
            rot_errs[row, h] = rot_error_deg(pred_SE3[h, :3, :3], gt_cmd_SE3[h, :3, :3])

    out = {"pos_err_m": pos_errs, "rot_err_deg": rot_errs}
    if action_dim == 19:
        out["vt_pos_err_m"] = vt_pos_errs
        out["stiffness_err"] = stiff_errs
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset-path", default="/local/real/jvclark/mujoco_data/try_2_flipup_sim_20hz.zarr")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-indices", default=None, help="comma-separated episode indices, overrides --num-episodes")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy, shape_meta = load_policy(args.ckpt, args.device)
    action_dim = shape_meta["action"]["shape"][0]
    print(f"[teacher-forcing] loaded policy, action_dim={action_dim}")

    root = zarr.open(args.dataset_path, mode="r")
    all_names = sorted(root["data"].group_keys(), key=lambda k: int(k.rsplit("_", 1)[-1]))
    if args.episode_indices is not None:
        idxs = [int(x) for x in args.episode_indices.split(",")]
        names = [f"episode_{i}" for i in idxs]
    else:
        names = all_names[: args.num_episodes]

    all_pos, all_rot = [], []
    all_vt_pos, all_stiff = [], []
    for name in names:
        res = evaluate_episode(policy, shape_meta, action_dim, root["data"][name], args.device)
        if res is None:
            continue
        all_pos.append(res["pos_err_m"])
        all_rot.append(res["rot_err_deg"])
        print(f"[teacher-forcing] {name}: mean_pos_err={res['pos_err_m'].mean()*100:.1f}cm "
              f"mean_rot_err={res['rot_err_deg'].mean():.1f}deg "
              f"(h0={res['pos_err_m'][:,0].mean()*100:.1f}cm h7={res['pos_err_m'][:,-1].mean()*100:.1f}cm)")
        if action_dim == 19:
            all_vt_pos.append(res["vt_pos_err_m"])
            all_stiff.append(res["stiffness_err"])
            print(f"                 vt_pos_err={res['vt_pos_err_m'].mean()*100:.1f}cm "
                  f"stiffness_err={res['stiffness_err'].mean():.0f}N/m")

    pos = np.concatenate(all_pos, axis=0)
    rot = np.concatenate(all_rot, axis=0)
    print(f"\n[teacher-forcing] SUMMARY ckpt={args.ckpt}")
    print(f"[teacher-forcing] n_queries={pos.shape[0]}")
    print(f"[teacher-forcing] cmd pos err by horizon step (cm): "
          f"{[f'{x*100:.1f}' for x in pos.mean(axis=0)]}")
    print(f"[teacher-forcing] cmd rot err by horizon step (deg): "
          f"{[f'{x:.1f}' for x in rot.mean(axis=0)]}")
    if action_dim == 19:
        vt_pos = np.concatenate(all_vt_pos, axis=0)
        stiff = np.concatenate(all_stiff, axis=0)
        print(f"[teacher-forcing] vt pos err by horizon step (cm): "
              f"{[f'{x*100:.1f}' for x in vt_pos.mean(axis=0)]}")
        print(f"[teacher-forcing] stiffness err by horizon step (N/m): "
              f"{[f'{x:.0f}' for x in stiff.mean(axis=0)]}")


if __name__ == "__main__":
    main()
