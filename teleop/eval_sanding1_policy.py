#!/usr/bin/env python3
"""Evaluate a sanding policy in the SAME env the sanding_1 teleop demos were
collected in, and save force-annotated rollout videos.

Differences from ``eval_sanding_policy.py`` (which targets the older scripted
``sanding_synthetic.zarr``):

  * Stock ``SandingEnv`` + stock ``SandingProperties()`` — the teleop demos were
    collected on the plain env with randomized target lines, NOT the clustered
    ``GroupedSandingEnv`` with the loosened dose band used for the scripted data.
    Every default in ``SandingProperties`` already equals the value the
    collection CLI recorded in the zarr metadata, so no overrides are passed.
  * Observation camera is the main scene ``MovableCamera`` (azimuth 90,
    elevation -75, distance 0.55), rendered 520x390 then INTER_AREA-resized to
    224x224 — exactly what ``teleop_sanding.py`` fed ``SandingEpisodeRecorder``.
    The old script rendered the wrist camera, which the recorder never stored.
  * 1 kHz control (``n_substeps=1``), matching ``control_freq=1000``.
  * Each predicted action is HELD for ``sparse_action_down_sample_steps`` ticks
    instead of one tick, so the commanded trajectory advances at the same rate
    it does in the training data.
  * Per-stream obs history: rgb / proprio / wrench each keep their own ring
    buffer at their own native rate and are strided by their own
    ``down_sample_steps``.

CLI matches the contract in EVAL_SYSTEM.md so eval_watcher/eval_dashboard can
drive it unchanged. ``--replan-every-ticks N`` is the exec horizon in *actions*
(the watcher injects the exec horizon through this flag).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import cv2
import dill
import hydra
import imageio
import numpy as np
import torch
import zarr

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, "/store/real/jvclark/PyriteML")
sys.path.insert(0, "/store/real/jvclark/PyriteUtility")
sys.path.insert(0, "/store/real/jvclark/PyriteConfig")
sys.path.insert(0, "/store/real/jvclark/force_control/teleop")
sys.path.insert(0, "/store/real/jvclark/force_control/flipup_minimal")

os.environ.setdefault("MUJOCO_GL", "egl")

import PyriteUtility.spatial_math.spatial_utilities as su
from sanding_teleop import SandingTeleop, SandingProperties

# Collection-time constants (from sanding_1.zarr metadata_json / command_line).
CTRL_HZ = 1000.0
RENDER_W, RENDER_H = 520, 390     # teleop_sanding.py --render-width/--render-height
OBS_W, OBS_H = 224, 224           # --dataset-image-size
RGB_HZ = 37.8                     # measured from the zarr (397 frames / 10.5 s)
TOOL_KP = 16000.0                 # --tool-kp
ARM_DAMPING = 2.5                 # --arm-damping


# ── camera ────────────────────────────────────────────────────────────────────

def make_scene_camera(env, width: int, height: int):
    """The main MovableCamera view teleop recorded into the dataset."""
    from dm_control.mujoco.engine import MovableCamera

    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, height)
    cam = MovableCamera(env.physics, height=height, width=width)
    cam.set_pose(
        lookat=env.default_cam_lookat,
        distance=env.default_cam_distance,
        azimuth=env.default_cam_azimuth,
        elevation=env.default_cam_elevation,
    )

    def render():
        return np.asarray(cam.render(), dtype=np.uint8).copy()

    return render


def to_obs_image(frame: np.ndarray) -> np.ndarray:
    if frame.shape[:2] != (OBS_H, OBS_W):
        return cv2.resize(frame, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)
    return frame


# ── policy ────────────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, device: str):
    payload = torch.load(ckpt_path, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir="/tmp")
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    # The action/obs normalizer is NOT carried in the checkpoint -- the workspace
    # pickles it beside the run (sparse_normalizer.pkl) and calls set_normalizer()
    # at train time. Without restoring it here predict_action returns the raw
    # network output, which reads as ~(-0.04, 0.03, -0.02) instead of an absolute
    # world pose ~(0.33, 0.01, 0.30) -- i.e. the arm gets yanked off the panel.
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    sp = os.path.join(run_dir, "sparse_normalizer.pkl")
    dp = os.path.join(run_dir, "dense_normalizer.pkl")
    if not os.path.exists(sp):
        raise FileNotFoundError(f"sparse_normalizer.pkl not found next to run: {sp}")
    import pickle
    sparse_normalizer = pickle.load(open(sp, "rb"))
    dense_normalizer = pickle.load(open(dp, "rb")) if os.path.exists(dp) else None
    policy.set_normalizer(sparse_normalizer, dense_normalizer)
    print(f"[eval_sanding1] restored normalizer from {os.path.basename(run_dir)}")

    policy.num_inference_steps = cfg.policy.num_inference_steps
    policy.eval().to(device)
    return policy, cfg.task.shape_meta


# ── obs history (one ring buffer per stream, each at its own native rate) ─────

@dataclass
class StreamSpec:
    horizon: int
    stride: int

    @property
    def length(self) -> int:
        return self.stride * (self.horizon - 1) + 1


class ObsHistory:
    def __init__(self, shape_meta: dict):
        s = shape_meta["sample"]["obs"]["sparse"]
        self.rgb_spec = StreamSpec(s["rgb_0"]["horizon"], s["rgb_0"]["down_sample_steps"])
        self.pose_spec = StreamSpec(s["robot0_eef_pos"]["horizon"],
                                    s["robot0_eef_pos"]["down_sample_steps"])
        self.has_wrench = "robot0_eef_wrench" in s
        self.wrench_spec = (StreamSpec(s["robot0_eef_wrench"]["horizon"],
                                       s["robot0_eef_wrench"]["down_sample_steps"])
                            if self.has_wrench else None)
        self.rgb = deque(maxlen=self.rgb_spec.length)
        self.pose9 = deque(maxlen=self.pose_spec.length)
        self.wrench = deque(maxlen=self.wrench_spec.length) if self.has_wrench else None

    def bootstrap(self, rgb, pose9):
        for _ in range(self.rgb_spec.length):
            self.rgb.append(rgb.copy())
        for _ in range(self.pose_spec.length):
            self.pose9.append(pose9.copy())
        if self.has_wrench:
            for _ in range(self.wrench_spec.length):
                self.wrench.append(np.zeros(6, dtype=float))

    def push_rgb(self, rgb):
        self.rgb.append(rgb.copy())

    def push_robot(self, pose9, wrench):
        self.pose9.append(pose9.copy())
        if self.has_wrench:
            self.wrench.append(np.asarray(wrench, dtype=float).copy())

    def build(self):
        """Returns (obs_dict, base_SE3) in the SAME frame convention as training.

        sparse_obs_to_obs_sample() takes base = the LAST observed eef pose and
        reports every pose as SE3_inv(base) @ SE3, so the newest entry is exactly
        the identity (position [0,0,0]) and the policy never sees an absolute
        position. Actions come back in that same base frame, so the caller needs
        base_SE3 to map them back to world.

        The wrench needs no transform: its adjoint is built from SE3_inv(SE3_base_i)
        at the last index, which is the identity by construction.
        """
        rgb = list(self.rgb)[::self.rgb_spec.stride][-self.rgb_spec.horizon:]
        pose = list(self.pose9)[::self.pose_spec.stride][-self.pose_spec.horizon:]
        # Match common_type_conversions.py exactly: float32, /255, then
        # moveaxis(-1, 1) so the encoder sees (T, C, H, W), not (T, H, W, C).
        imgs = np.stack(rgb).astype(np.float32) / 255.0

        SE3_WT = su.pose9_to_SE3(np.stack(pose))          # (T, 4, 4)
        base_SE3 = SE3_WT[-1]
        rel = su.SE3_to_pose9(su.SE3_inv(base_SE3) @ SE3_WT)

        out = {
            "rgb_0": np.moveaxis(imgs, -1, 1),
            "robot0_eef_pos": rel[..., :3].astype(np.float32),
            "robot0_eef_rot_axis_angle": rel[..., 3:].astype(np.float32),
        }
        if self.has_wrench:
            w = list(self.wrench)[::self.wrench_spec.stride][-self.wrench_spec.horizon:]
            out["robot0_eef_wrench"] = np.stack(w).astype(np.float32)
        return out, base_SE3


def tool_pose9(env) -> np.ndarray:
    return su.SE3_to_pose9(su.pose7_to_SE3(env.get_tool_pose()))


# ── demo init bank ────────────────────────────────────────────────────────────

def load_init_bank(dataset_path: str, successful_only: bool = True):
    root = zarr.open(dataset_path, mode="r")
    names = sorted(root["data"].group_keys(), key=lambda k: int(k.rsplit("_", 1)[-1]))
    bank = []
    for name in names:
        ep = root["data"][name]
        attrs = dict(ep.attrs)
        if successful_only and not attrs.get("success", False):
            continue
        p7 = np.asarray(ep["ts_pose_fb_0"][0], dtype=float)
        bank.append({
            "init_pose9": su.SE3_to_pose9(su.pose7_to_SE3(p7)),
            "ep_name": name,
            "duration_s": len(ep["normal_force_n"]) / CTRL_HZ,
        })
    if not bank:  # fall back to all episodes
        return load_init_bank(dataset_path, successful_only=False)
    return bank


# ── force-annotated video ─────────────────────────────────────────────────────

FORCE_MIN, FORCE_TARGET, FORCE_CAP, FORCE_BREAK = 6.66, 18.0, 30.0, 45.0


def render_force_panel(force_hist, cov_hist, w, h, window_s, fps,
                       f_max=50.0, title="wrist normal force"):
    """cv2-drawn force strip chart (BGR)."""
    panel = np.full((h, w, 3), 252, dtype=np.uint8)
    pad_l, pad_r, pad_t, pad_b = 56, 12, 30, 34
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b

    def y_of(f):
        return int(pad_t + ph * (1.0 - np.clip(f, 0, f_max) / f_max))

    def x_of(i, n):
        return int(pad_l + pw * (i / max(n - 1, 1)))

    cv2.rectangle(panel, (pad_l, pad_t), (pad_l + pw, pad_t + ph), (225, 224, 219), 1)
    refs = [(FORCE_MIN, "6.7 dose floor", (138, 136, 128)),
            (FORCE_TARGET, "18 target", (52, 104, 235)),
            (FORCE_CAP, "30 cap", (138, 136, 128)),
            (FORCE_BREAK, "45 break", (72, 73, 227))]
    for val, _lab, col in refs:
        if val > f_max:
            continue
        y = y_of(val)
        for x in range(pad_l, pad_l + pw, 8):
            cv2.line(panel, (x, y), (min(x + 4, pad_l + pw), y), col, 1)

    n_win = max(int(window_s * fps), 2)
    hist = list(force_hist)
    seg = hist[-n_win:]
    base = len(hist) - len(seg)
    # Only flag contact LOSS, i.e. dropouts after the pad has first touched down --
    # otherwise the whole pre-contact approach reads as one solid red block.
    arr = np.asarray(hist, dtype=float)
    touched = int(np.argmax(arr > 1.0)) if (arr > 1.0).any() else None
    if len(seg) > 1:
        pts = np.array([[x_of(i, len(seg)), y_of(f)] for i, f in enumerate(seg)], np.int32)
        cv2.polylines(panel, [pts], False, (214, 120, 42), 2, cv2.LINE_AA)  # blue
        if touched is not None:
            for i, f in enumerate(seg):
                if f <= 1.0 and (base + i) > touched:
                    cv2.line(panel, (x_of(i, len(seg)), pad_t),
                             (x_of(i, len(seg)), pad_t + ph), (200, 200, 255), 1)
    # labels last so the data never hides them
    for val, lab, col in refs:
        if val > f_max:
            continue
        cv2.putText(panel, lab, (pad_l + 4, y_of(val) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, col, 1, cv2.LINE_AA)
    # y axis ticks
    for val in range(0, int(f_max) + 1, 10):
        y = y_of(val)
        cv2.putText(panel, f"{val:3d}", (6, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (82, 81, 78), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{title}  (N)", (pad_l, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, (11, 11, 11), 1, cv2.LINE_AA)
    cv2.putText(panel, f"last {window_s:.0f}s", (pad_l + pw - 62, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 128, 120), 1, cv2.LINE_AA)
    f_now = force_hist[-1] if len(force_hist) else 0.0
    cov_now = cov_hist[-1] if len(cov_hist) else 0.0
    cv2.putText(panel, f"F = {f_now:5.1f} N", (pad_l, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (214, 120, 42), 2, cv2.LINE_AA)
    cv2.putText(panel, f"coverage = {100*cov_now:5.1f}%", (pad_l + 150, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (122, 175, 27), 2, cv2.LINE_AA)
    return panel


def write_annotated_video(path, frames, forces, covs, fps, window_s=3.0):
    """frames: RGB uint8 list. forces/covs: same length."""
    if not frames:
        return
    fh, fw = frames[0].shape[:2]
    pw = max(fw, 420)
    out = []
    f_max = float(max(50.0, np.max(forces) * 1.15)) if len(forces) else 50.0
    for i, fr in enumerate(frames):
        panel = render_force_panel(forces[: i + 1], covs[: i + 1], pw, fh,
                                   window_s, fps, f_max=f_max)
        left = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        comp = np.hstack([left, panel])
        out.append(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    imageio.mimwrite(path, out, fps=fps, quality=6, macro_block_size=1)


# ── rollout ───────────────────────────────────────────────────────────────────

def run_episode(env, policy, shape_meta, action_dim, init_entry, *,
                exec_horizon, max_ticks, obs_render_fn, video_render_fn,
                device, ep_seed, want_video):
    env._rng = np.random.default_rng(ep_seed)
    env.reset()

    action_stride = int(shape_meta["sample"]["action"]["sparse"]["down_sample_steps"])
    rgb_every = max(int(round(CTRL_HZ / RGB_HZ)), 1)
    video_every = max(int(round(CTRL_HZ / 30.0)), 1)   # 30 fps video
    video_fps = CTRL_HZ / video_every

    # Slew to the demo's initial tool pose so the rollout starts in-distribution.
    #
    # The commanded target has to advance on its OWN, not be re-derived from
    # env.tool_pos each tick: with a compliant arm the tool always lags its
    # command, so "current position + 1 step" just chases the lag and the arm
    # crawls. That left the rollout starting at HOME, ~42 mm from the demo start,
    # and the policy's (absolute) first target then yanked the tool into the
    # panel at >200 N.
    init_target = np.asarray(init_entry["init_pose9"][:3], dtype=float)
    cmd_pos = env.tool_pos.copy()
    step_m = 0.15 / CTRL_HZ                      # 0.15 m/s command ramp
    for _ in range(12000):
        err = init_target - cmd_pos
        d = float(np.linalg.norm(err))
        if d < 1e-4:
            break
        cmd_pos = cmd_pos + err * (min(step_m, d) / d)
        env.step(cmd_pos, target_rotvec=None, n_substeps=1)
    # let the arm actually converge onto the held target before observing
    for _ in range(2000):
        env.step(init_target, target_rotvec=None, n_substeps=1)
        if float(np.linalg.norm(init_target - env.tool_pos)) < 0.002:
            break
    env._broken = False          # ignore any contact made while getting into pose
    env._break_streak = 0
    env._break_force_filtered = 0.0

    hist = ObsHistory(shape_meta)
    frame0 = obs_render_fn()
    hist.bootstrap(to_obs_image(frame0), tool_pose9(env))
    chunk_base_SE3 = np.eye(4)

    force_trace, cov_trace, tool_trace, cmd_trace, tick_trace = [], [], [], [], []
    vid_frames, vid_force, vid_cov = [], [], []

    action_chunk = None
    act_idx = 0            # which action in the chunk
    hold = 0               # ticks already spent on the current action
    n_exec = 0             # actions executed since last replan
    target_pos = env.tool_pos.copy()
    tick = 0
    peak_force = 0.0

    while tick < max_ticks:
        if action_chunk is None or n_exec >= exec_horizon:
            obs, chunk_base_SE3 = hist.build()
            # predict_action wants {"sparse": {...}} and returns
            # {"sparse": ..., "dense": ...} -- not a flat dict with an "action" key.
            obs_t = {"sparse": {k: torch.tensor(v[None], dtype=torch.float32).to(device)
                                for k, v in obs.items()}}
            with torch.no_grad():
                action_chunk = policy.predict_action(obs_t)["sparse"][0].cpu().numpy()
            act_idx, hold, n_exec = 0, 0, 0

        act = action_chunk[min(act_idx, len(action_chunk) - 1)]
        # DP action_dim 9 = pose9; ACP 19 = pose9_cmd + pose9_vt + stiffness.
        # act[:9] is a pose9 RELATIVE to the base of the chunk it came from, so
        # world target = base @ relative (the inverse of the training transform
        # SE3_relative = SE3_inv(base) @ SE3_absolute).
        SE3_target = chunk_base_SE3 @ su.pose9_to_SE3(np.asarray(act[:9], dtype=float))
        target_pos = np.asarray(SE3_target[:3, 3], dtype=float).copy()

        env.step(target_pos, target_rotvec=None, n_substeps=1)
        tick += 1
        hold += 1
        if hold >= action_stride:
            hold = 0
            act_idx += 1
            n_exec += 1

        f_n = float(env.normal_force_n())
        cov = float(env.coverage_fraction("just_right"))
        peak_force = max(peak_force, f_n)
        force_trace.append(f_n)
        cov_trace.append(cov)
        tool_trace.append(env.tool_pos.copy())
        cmd_trace.append(target_pos.copy())
        tick_trace.append(tick)

        # proprio + wrench stream runs at full control rate
        hist.push_robot(tool_pose9(env),
                        np.concatenate([env.pad_contact_force()[0], np.zeros(3)]))
        if tick % rgb_every == 0:
            hist.push_rgb(to_obs_image(obs_render_fn()))
        if want_video and tick % video_every == 0:
            vid_frames.append(video_render_fn())
            vid_force.append(f_n)
            vid_cov.append(cov)

        if env.success() or env.broken:
            break

    F = np.asarray(force_trace)
    touching = F > 1.0
    return {
        "success": bool(env.success()),
        "broken": bool(env.broken),
        "coverage_just_right": float(env.coverage_fraction("just_right")),
        "ticks": tick,
        "duration_s": tick / CTRL_HZ,
        "peak_force_n": float(peak_force),
        "mean_contact_force_n": float(F[touching].mean()) if touching.any() else 0.0,
        "contact_frac": float(touching.mean()) if len(F) else 0.0,
        "init_from": init_entry["ep_name"],
        "_traces": {
            "force": F.astype(np.float32),
            "coverage": np.asarray(cov_trace, dtype=np.float32),
            "tool_xyz": np.asarray(tool_trace, dtype=np.float32),
            "cmd_xyz": np.asarray(cmd_trace, dtype=np.float32),
        },
        "_video": (vid_frames, vid_force, vid_cov, video_fps),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset-path", default="/store/real/jvclark/sanding_1.zarr")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--replan-every-ticks", type=int, default=8,
                   help="exec horizon: number of predicted actions run before replanning")
    p.add_argument("--max-ticks", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-seed", type=int, default=123)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--video-episodes", type=int, default=0)
    p.add_argument("--fixed-stiffness", action="store_true", default=True)
    # Env knobs that MUST match whichever dataset the policy was trained on.
    # Stock (raw sanding_1 teleop demos): --tool-kp 16000 --friction 0.6
    # Clean replay-generated demos:       --tool-kp 4000  --friction 0.3
    p.add_argument("--tool-kp", type=float, default=TOOL_KP)
    p.add_argument("--friction", type=float, default=0.6)
    args = p.parse_args()

    print(f"[eval_sanding1] ckpt: {args.ckpt}")
    policy, shape_meta = load_policy(args.ckpt, args.device)
    action_dim = shape_meta["action"]["shape"][0]
    print(f"[eval_sanding1] action_dim={action_dim}  exec_horizon={args.replan_every_ticks}")

    # Defaults reproduce the collection config recorded in sanding_1.zarr;
    # --tool-kp/--friction retarget it at a clean-replay dataset's env.
    props = SandingProperties(friction=(args.friction, 0.01, 0.0002))
    print(f"[eval_sanding1] env: tool_kp={args.tool_kp:.0f} friction={args.friction}")
    env = SandingTeleop(seed=0, properties=props,
                        tool_kp=args.tool_kp, arm_damping=ARM_DAMPING)
    env.reset()

    # Size the offscreen buffer for the LARGEST camera before constructing any of
    # them: dm_control allocates the shared render context when the first Camera
    # is built, so a later max() on offwidth is too late. The model ships with
    # offwidth=600, which the 640-wide video camera overran -- the right 40
    # columns of every saved frame were uninitialised garbage. (The 520-wide obs
    # camera fits, so policy inputs were never affected.)
    VIDEO_W, VIDEO_H = 640, 480
    env.model.vis.global_.offwidth = max(int(env.model.vis.global_.offwidth),
                                         RENDER_W, VIDEO_W)
    env.model.vis.global_.offheight = max(int(env.model.vis.global_.offheight),
                                          RENDER_H, VIDEO_H)
    obs_render_fn = make_scene_camera(env, RENDER_W, RENDER_H)
    video_render_fn = make_scene_camera(env, VIDEO_W, VIDEO_H)

    bank = load_init_bank(args.dataset_path)
    print(f"[eval_sanding1] init bank: {len(bank)} successful demos")

    video_dir = None
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        if args.video_episodes > 0:
            video_dir = os.path.join(args.out_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)

    rng = np.random.default_rng(args.eval_seed)
    results, traces = [], {}
    t0 = time.time()

    for ep in range(args.num_episodes):
        entry = bank[int(rng.integers(0, len(bank)))]
        ep_seed = int(rng.integers(0, 2**31))
        want_video = video_dir is not None and ep < args.video_episodes

        res = run_episode(
            env, policy, shape_meta, action_dim, entry,
            exec_horizon=args.replan_every_ticks,
            max_ticks=args.max_ticks,
            obs_render_fn=obs_render_fn,
            video_render_fn=video_render_fn,
            device=args.device,
            ep_seed=ep_seed,
            want_video=want_video,
        )
        tr = res.pop("_traces")
        frames, vf, vc, vfps = res.pop("_video")
        res["episode"] = ep
        outcome = "success" if res["success"] else ("broken" if res["broken"] else "fail")

        if want_video and frames:
            base = os.path.join(video_dir, f"episode_{ep:03d}_{outcome}")
            imageio.mimwrite(base + ".mp4", frames, fps=vfps, quality=6,
                             macro_block_size=1)
            write_annotated_video(base + "_forces.mp4", frames, vf, vc, vfps)
            print(f"[eval_sanding1]   video → {base}_forces.mp4")

        for k, v in tr.items():
            traces[f"ep{ep:03d}_{k}"] = v
        results.append(res)
        print(f"[eval] ep={ep}: {outcome}  cov={res['coverage_just_right']:.2f}  "
              f"F_peak={res['peak_force_n']:.1f}N  F_contact={res['mean_contact_force_n']:.1f}N  "
              f"t={res['duration_s']:.1f}s  (elapsed {time.time()-t0:.0f}s)")

    n = len(results)
    summary = {
        "success_rate": float(np.mean([r["success"] for r in results])) if n else 0.0,
        "avg_ticks": float(np.mean([r["ticks"] for r in results])) if n else 0.0,
        "avg_duration_s": float(np.mean([r["duration_s"] for r in results])) if n else 0.0,
        "avg_coverage": float(np.mean([r["coverage_just_right"] for r in results])) if n else 0.0,
        "broken_rate": float(np.mean([r["broken"] for r in results])) if n else 0.0,
        "peak_force_max_N": float(np.max([r["peak_force_n"] for r in results])) if n else 0.0,
        "peak_force_mean_N": float(np.mean([r["peak_force_n"] for r in results])) if n else 0.0,
        "mean_contact_force_N": float(np.mean([r["mean_contact_force_n"] for r in results])) if n else 0.0,
        "exec_horizon": args.replan_every_ticks,
        "results": results,
    }
    print(f"\n[eval_sanding1] success {summary['success_rate']*100:.0f}%  "
          f"cov {summary['avg_coverage']:.2f}  peakF {summary['peak_force_mean_N']:.1f}N")

    if args.out_dir:
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        np.savez_compressed(os.path.join(args.out_dir, "episode_traces.npz"), **traces)


if __name__ == "__main__":
    main()
