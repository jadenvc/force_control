#!/usr/bin/env python3
"""Generate CLEAN sanding demos by augmented replay of the human teleop demos.

Why: in sanding_1.zarr the pad chatters in and out of contact ~26 times/second
(median contact fragment 9 ms, median gap 1 ms) because the operator sands at
~9.5 N -- roughly half the 18 N nominal -- which is right at the edge of stable
contact for this pad/panel pair.

Fix: replay each demo's recorded command trajectory through the same env with a
softer arm (``--kp``, default 4000 N/m instead of 16000) and a slow Z force servo
holding a setpoint high enough for contact to stay solidly closed. Measured over
10 episodes at 13 N: 0.14 contact losses/s and 0.094 N/ms force roughness, versus
26/s and 1.7 N/ms in the source demos.

The dose law couples force to episode duration -- dose = k*(F - 6.66)*dwell -- so
holding a smooth (higher) force means each cell needs proportionally LESS dwell.
Two modes expose that trade-off:

  --mode timing   keep the source episode's duration exactly; solve for the force
                  setpoint that lands mean target dose on --dose-target.
                  (Lands near ~9 N, so smoother than the source but not perfectly
                  smooth -- this is the timing-faithful set.)

  --mode smooth   stretch the episode by --warp (default 1.25x) and solve for the
                  force setpoint. Sweeping SLOWER is what actually kills the
                  chatter -- the XY->Z kinematic coupling that breaks contact
                  scales with lateral speed, so speeding up to allow a higher
                  force makes it much worse (measured: warp 0.44 -> 86 losses/s),
                  while slowing down and dropping the force to keep the same dose
                  drives it to zero. Costs ~25% episode duration.

Both modes pin the target line inside the span the source demo actually swept, so
coverage is recoverable, and both re-derive success from the live env.

Timestamps are written on ONE clock (see fix_sanding_rgb_timestamps.py for the bug
this avoids): robot/wrench and rgb both use per-episode milliseconds.

Usage (one shard; run several in parallel then merge_zarr.py):
    python gen_sanding_clean.py --mode fast --out /path/shard0.zarr \
        --sources episode_0,episode_1 --variants 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import zarr

sys.path.insert(0, "/store/real/jvclark/force_control/teleop")
sys.path.insert(0, "/store/real/jvclark/force_control/flipup_minimal")
os.environ.setdefault("MUJOCO_GL", "egl")

from sanding_teleop import SandingTeleop, SandingProperties, PANEL_TRANSFORM
from sanding_recorder import SandingEpisodeRecorder


class MonotonicRecorder(SandingEpisodeRecorder):
    """SandingEpisodeRecorder that never reuses an episode id.

    The base class derives the next id from the on-disk group listing on every
    commit. In practice that occasionally handed back an id that was already
    written, and the second commit's ``data_group.move()`` clobbered the first --
    silently, since commit still returned a name and reported success. Measured
    at roughly one lost episode per shard (a shard logging 15 commits held 14).
    Keeping a high-water mark alongside the on-disk maximum makes reuse
    impossible within a process, which is all a single-writer shard needs.
    """

    _high_water = -1

    def _next_episode_id(self) -> int:
        disk = super()._next_episode_id()
        nxt = max(disk, self._high_water + 1)
        self._high_water = nxt
        return nxt

SRC = "/store/real/jvclark/sanding_1.zarr"
HZ = 1000.0
RENDER_W, RENDER_H = 520, 390
RGB_EVERY = 26                      # ~38.5 Hz, matches the source demos' 37.8 Hz
PANEL_X = float(PANEL_TRANSFORM[0, 3])


# ── source demos ──────────────────────────────────────────────────────────────

def load_source(nm):
    g = zarr.open(os.path.join(SRC, "data", nm), mode="r")
    return dict(
        name=nm,
        cmd=np.asarray(g["ts_pose_command_0"][:], float)[:, :3],
        fb=np.asarray(g["ts_pose_fb_0"][:], float),
        F=np.asarray(g["normal_force_n"][:], float).ravel(),
        qpos=np.asarray(g["qpos"][0], float),
        attrs=dict(g.attrs),
    )


def moving_average(a, k):
    if k <= 1:
        return a.copy()
    ker = np.ones(k) / k
    pad = np.vstack([np.repeat(a[:1], k, 0), a, np.repeat(a[-1:], k, 0)])
    return np.stack([np.convolve(pad[:, i], ker, "same") for i in range(a.shape[1])], 1)[k:k + len(a)]


def time_warp(cmd, c):
    """Resample the command path in time. c<1 = shorter/faster episode."""
    if abs(c - 1.0) < 1e-6:
        return cmd.copy()
    n = len(cmd)
    m = max(int(round(n * c)), 64)
    src = np.linspace(0.0, n - 1, m)
    return np.stack([np.interp(src, np.arange(n), cmd[:, i]) for i in range(3)], 1)


def pick_line(d, props, rng):
    """(n_regions, start_x) placed inside the panel-local x span the demo pressed."""
    touch = d["F"] > 1.0
    if not touch.any():
        return None
    x = d["fb"][touch, 0] - PANEL_X
    lo_x, hi_x = float(np.percentile(x, 3)), float(np.percentile(x, 97))
    pitch = 1.8 * props.region_radius_m
    half = props.panel_length_m / 2.0
    margin = props.region_radius_m * 1.2
    lo_x = max(lo_x, -half + margin)
    hi_x = min(hi_x, half - margin)
    n_max = int(np.floor((hi_x - lo_x) / pitch)) + 1
    if n_max < props.num_regions_min:
        return None
    n = int(np.clip(n_max, props.num_regions_min, props.num_regions_max))
    line_len = (n - 1) * pitch
    hi_start = min(hi_x - line_len, half - margin - line_len)
    if hi_start < lo_x:
        return None
    return n, float(rng.uniform(lo_x, hi_start))


def make_env(props, kp, line):
    n_reg, start_x = line

    class Pinned(SandingTeleop):
        def _sample_target_regions(self):
            self._num_regions = int(n_reg)
            pitch = self._region_pitch_m()
            xs = start_x + np.arange(self._num_regions) * pitch
            return np.stack([xs, np.zeros_like(xs)], axis=1)

    return Pinned(seed=0, properties=props, tool_kp=kp, arm_damping=2.5)


# ── the augmented replay ──────────────────────────────────────────────────────

def replay(d, *, kp, line, props, f_target, warp=1.0, smooth_ms=25, y_off=0.0,
           servo_gain=4e-7, servo_tau=0.05, z_rate=2.0e-6, z_clip=0.012,
           recorder=None, render=None):
    env = make_env(props, kp, line)
    env.reset()
    env.data.qpos[env.joint_qpos_ids] = d["qpos"]
    env.data.qvel[env.joint_dof_ids] = 0.0
    env.physics.forward()

    cmd = time_warp(d["cmd"], warp)
    cmd = moving_average(cmd, smooth_ms)
    cmd[:, 1] += y_off          # shift the stroke across the target line
    n = len(cmd)

    alpha = 1.0 - np.exp(-1.0 / (HZ * max(servo_tau, 1e-4)))
    f_filt, z_corr = 0.0, 0.0
    F = np.empty(n)
    frame = None
    img_id = -1

    for i in range(n):
        want = cmd[i].copy()
        want[2] += z_corr
        env.step(want, target_rotvec=None, n_substeps=1)
        f = float(env.normal_force_n())
        f_filt += alpha * (f - f_filt)
        z_corr = float(np.clip(
            z_corr + np.clip(-servo_gain * (f_target - f_filt), -z_rate, z_rate),
            -z_clip, z_clip))
        F[i] = f

        if recorder is not None:
            if render is not None and i % RGB_EVERY == 0:
                frame = render()
                img_id += 1
            # ONE clock for every stream: per-episode milliseconds.
            recorder.record_sample(
                env,
                timestamp_ms=float(i),
                target_pos=want,
                target_rotvec=None,
                device_state={"pos": env.tool_pos, "vel": np.zeros(3),
                              "force_cmd": np.zeros(3), "force_meas": np.zeros(3)},
                sent_force=np.zeros(3),
                image_rgb=frame,
                image_capture_time_s=(i / HZ),
                image_id=img_id,
            )
        if env.broken:
            break

    m = i + 1
    tm = env._target_mask
    return dict(
        F=F[:m], n=m, duration_s=m / HZ,
        success=bool(env.success()), broken=bool(env.broken),
        cov=float(env.coverage_fraction("just_right")),
        cov_under=float(env.coverage_fraction("under")),
        cov_over=float(env.coverage_fraction("over")),
        dose=float(env._dose[tm].mean()),
        n_regions=int(env._num_regions),
    )


def force_stats(F):
    t = F > 1.0
    if not t.any():
        return dict(F_in=0.0, F_std=0.0, gaps=0, gaps_s=0.0, rough=0.0, contact=0.0)
    i0, i1 = int(np.argmax(t)), int(len(F) - np.argmax(t[::-1]))
    Fe, te = F[i0:i1], t[i0:i1]
    idx = np.flatnonzero(np.diff(np.concatenate(([0], (~te).astype(np.int8), [0]))))
    gaps = idx.reshape(-1, 2)
    dur = max((i1 - i0) / HZ, 1e-6)
    return dict(F_in=float(Fe[te].mean()), F_std=float(Fe[te].std()),
                gaps=len(gaps), gaps_s=len(gaps) / dur,
                rough=float(np.diff(Fe).std()), contact=float(te.mean()))


def solve(d, *, mode, kp, line, props, dose_target, f_target, warp=1.25, y_off=0.0,
          iters=7, verbose=False):
    """Search the free variable for the best COVERAGE, not just the right dose.

    Mean target dose is monotonic in both free variables, but landing the mean on
    1.0 is not sufficient: dose is spread unevenly along the target line (the ends
    of the line see fewer pad passes than the middle), so the mean can sit at 1.0
    while a third of the cells are outside the [0.5, 1.5] band. Coverage itself is
    unimodal in the free variable -- too little and cells stay under-sanded, too
    much and they tip over -- so a coarse grid plus a local refine around the peak
    is both cheaper and far more reliable than bisecting the proxy.
    """
    if mode == "timing":
        grid = np.linspace(7.5, 15.0, 8)
        var = lambda v: dict(f_target=float(v), warp=1.0, y_off=y_off)
    else:
        grid = np.linspace(7.2, 10.0, 8)
        var = lambda v: dict(f_target=float(v), warp=float(warp), y_off=y_off)

    evaluated = {}

    def probe(v):
        v = float(v)
        if v in evaluated:
            return evaluated[v]
        out = replay(d, kp=kp, line=line, props=props, **var(v))
        out["var"] = v
        evaluated[v] = out
        if verbose:
            print(f"      var={v:7.3f} cov={out['cov']:.2f} dose={out['dose']:.2f} "
                  f"succ={out['success']}", flush=True)
        return out

    results = [probe(v) for v in grid]
    best = max(results, key=lambda r: (r["cov"], -abs(r["dose"] - dose_target)))
    if best["success"]:
        return best, var(best["var"])

    # refine around the peak: bisect the bracket the peak sits in
    order = sorted(grid)
    i = order.index(best["var"])
    lo = order[max(i - 1, 0)]
    hi = order[min(i + 1, len(order) - 1)]
    for _ in range(iters):
        for v in (lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0):
            out = probe(v)
            if out["cov"] > best["cov"]:
                best = out
        if best["success"]:
            break
        span = (hi - lo) / 3.0
        lo, hi = max(best["var"] - span, lo), min(best["var"] + span, hi)
        if hi - lo < 1e-3:
            break
    return best, var(best["var"])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["timing", "smooth"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="", help="comma list; default = all")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--kp", type=float, default=4000.0)
    ap.add_argument("--friction", type=float, default=0.3,
                    help="sliding friction; 0.3 vs the stock 0.6 removes the "
                         "stick-slip that couples lateral motion into contact loss")
    ap.add_argument("--warp", type=float, default=1.25,
                    help="smooth mode: stretch episode duration by this factor")
    ap.add_argument("--f-target", type=float, default=13.0, help="unused in current modes")
    ap.add_argument("--dose-target", type=float, default=1.0)
    ap.add_argument("--smooth-ms", type=int, default=25)
    ap.add_argument("--max-episodes", type=int, default=10**9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--y-jitter", type=float, default=0.008,
                    help="per-variant lateral shift of the stroke (m)")
    ap.add_argument("--warp-jitter", type=float, default=0.10,
                    help="per-variant fractional jitter on --warp (smooth mode)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    props = SandingProperties(friction=(args.friction, 0.01, 0.0002))
    names = ([s for s in args.sources.split(",") if s] or
             sorted([e for e in os.listdir(os.path.join(SRC, "data"))
                     if e.startswith("episode_")],
                    key=lambda s: int(s.split("_")[1])))

    rec = MonotonicRecorder(args.out, sample_hz=HZ, image_size=(224, 224),
                            include_rgb=True, min_samples=20)
    rng = np.random.default_rng(args.seed)
    kept = 0
    t0 = time.time()
    for nm in names:
        if kept >= args.max_episodes:
            break
        d = load_source(nm)
        for v in range(args.variants):
            if kept >= args.max_episodes:
                break
            # Per-variant augmentation. Without this, two variants of the same
            # source collapse onto an identical demo whenever pick_line's legal
            # start range is degenerate (~25% of them did).
            y_off = float(rng.uniform(-args.y_jitter, args.y_jitter))
            warp_v = (1.0 if args.mode == "timing"
                      else args.warp * float(rng.uniform(1.0 - args.warp_jitter,
                                                         1.0 + args.warp_jitter)))
            line = pick_line(d, props, rng)
            if line is None:
                print(f"[{nm} v{v}] no target line fits the swept span -- skip", flush=True)
                continue

            probe, params = solve(d, mode=args.mode, kp=args.kp, line=line, props=props,
                                  dose_target=args.dose_target, f_target=args.f_target,
                                  warp=warp_v, y_off=y_off, verbose=args.verbose)
            if not probe["success"]:
                print(f"[{nm} v{v}] tuned but not successful "
                      f"(cov {probe['cov']:.2f} dose {probe['dose']:.2f}) -- skip", flush=True)
                continue

            # final pass, this time recording RGB + every logged field
            rec.start_episode()
            out = _record_pass(d, rec, args, props, line, params)
            if out is None or not out["success"]:
                rec.discard()
                print(f"[{nm} v{v}] record pass failed/unsuccessful -- skip", flush=True)
                continue
            fs = force_stats(out["F"])
            name = rec.commit(success=out["success"], broken=out["broken"],
                              termination_reason="generated_replay",
                              final_coverage_fraction=out["cov"],
                              final_task_metric_name="coverage_just_right",
                              final_task_metric_value=out["cov"])
            if name is None:
                continue
            kept += 1
            print(f"[{nm} v{v}] -> {name}  {out['duration_s']:5.2f}s  cov {out['cov']:.2f}  "
                  f"F {fs['F_in']:5.2f}±{fs['F_std']:4.2f}N  gaps {fs['gaps_s']:5.2f}/s  "
                  f"rough {fs['rough']:.3f}  [{kept} kept, {time.time()-t0:.0f}s]", flush=True)

    print(f"\nDONE: {kept} episodes -> {args.out}  ({time.time()-t0:.0f}s)")


def _record_pass(d, rec, args, props, line, params):
    """Re-run the solved parameters with RGB rendering into the recorder."""
    from eval_sanding1_policy import make_scene_camera
    env = make_env(props, args.kp, line)
    env.reset()
    env.data.qpos[env.joint_qpos_ids] = d["qpos"]
    env.data.qvel[env.joint_dof_ids] = 0.0
    env.physics.forward()
    render = make_scene_camera(env, RENDER_W, RENDER_H)

    cmd = moving_average(time_warp(d["cmd"], params["warp"]), args.smooth_ms)
    cmd[:, 1] += params.get("y_off", 0.0)
    n = len(cmd)
    alpha = 1.0 - np.exp(-1.0 / (HZ * 0.05))
    f_filt, z_corr = 0.0, 0.0
    F = np.empty(n)
    frame, img_id = None, -1
    for i in range(n):
        want = cmd[i].copy()
        want[2] += z_corr
        env.step(want, target_rotvec=None, n_substeps=1)
        f = float(env.normal_force_n())
        f_filt += alpha * (f - f_filt)
        z_corr = float(np.clip(
            z_corr + np.clip(-4e-7 * (params["f_target"] - f_filt), -2.0e-6, 2.0e-6),
            -0.012, 0.012))
        F[i] = f
        if i % RGB_EVERY == 0:
            frame = render()
            img_id += 1
        rec.record_sample(
            env, timestamp_ms=float(i), target_pos=want, target_rotvec=None,
            device_state={"pos": env.tool_pos, "vel": np.zeros(3),
                          "force_cmd": np.zeros(3), "force_meas": np.zeros(3)},
            sent_force=np.zeros(3), image_rgb=frame,
            image_capture_time_s=(i / HZ), image_id=img_id)
        if env.broken:
            break
    m = i + 1
    tm = env._target_mask
    return dict(F=F[:m], n=m, duration_s=m / HZ,
                success=bool(env.success()), broken=bool(env.broken),
                cov=float(env.coverage_fraction("just_right")),
                dose=float(env._dose[tm].mean()))


if __name__ == "__main__":
    main()
