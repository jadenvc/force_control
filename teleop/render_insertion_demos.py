"""Render N scripted insertion demos to video, with the scalar peg contact
force annotated as a scrolling strip-chart underneath the rendered scene.

Two passes per seed, both through the exact same ``run_scripted_demo`` used
by insertion_scripted_demo.py/test_insertion.py (no duplicated control
logic):

  1. A fast, render-free pass just to get ``DemoResult.peak_force_n`` so the
     force panel's vertical scale can be fixed for the whole clip (rather
     than rescaling frame-to-frame, which would make a perfectly smooth
     force trace look like it's jumping around).
  2. A second pass, with the same seed (so it reproduces the identical
     trajectory -- the env/state machine have no external randomness beyond
     the seeded landing-perturbation RNG in run_scripted_demo, see its
     ``rng = np.random.default_rng(seed)``), this time with a
     ``frame_callback`` that appends the running force trace every control
     step and grabs a rendered RGB frame every ``render_every`` steps.

Usage:
    python teleop/render_insertion_demos.py --out-dir teleop/insertion_demo_videos --num-demos 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from insertion_scripted_demo import ScriptedDemoConfig, run_scripted_demo
from insertion_teleop import InsertionEnv

WIDTH = 480  # <= default MuJoCo offscreen framebuffer (600x480); see ground.xml's <visual> if this needs raising
SCENE_H = 360
PANEL_H = 160
FPS = 30


def _force_panel(force_trace: list, scale_n: float, width: int, height: int) -> np.ndarray:
    """Scrolling force strip-chart, most-recent value on the right edge.

    Mirrors render_rich_episode.py's ``force_panel`` (cv2 polyline plot,
    fixed vertical scale, gridlines + numeric labels) but for a single
    scalar contact-force magnitude (``DemoResult.force_n`` /
    ``env.peg_contact_force()``'s norm) rather than a 3-axis wrench, and a
    fixed trailing time window instead of the whole-episode x-axis, since
    episodes here run long enough (thousands of control steps) that a
    whole-episode x-axis would squash the interesting contact/search/insert
    detail into a few pixels.
    """
    window = 400  # control steps of trailing history shown at once
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 55, 12, 14, 14
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    for frac, label in [(0.0, f"{scale_n:.0f}N"), (1.0, "0N")]:
        y = int(margin_t + frac * plot_h)
        cv2.line(canvas, (margin_l, y), (width - margin_r, y), (210, 210, 210), 1)
        cv2.putText(canvas, label, (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (margin_l, margin_t), (width - margin_r, height - margin_b), (170, 170, 170), 1)

    trace = force_trace[-window:]
    n = len(trace)
    if n >= 2:
        xs = margin_l + (np.arange(n) / (window - 1)) * plot_w
        ys = margin_t + plot_h * (1.0 - np.clip(np.asarray(trace), 0.0, scale_n) / scale_n)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, (200, 90, 20), 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts[-1]), 4, (20, 60, 200), -1, cv2.LINE_AA)

    current = force_trace[-1] if force_trace else 0.0
    cv2.putText(canvas, f"contact force: {current:5.1f} N", (margin_l + 6, height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def render_one(seed: int, out_path: Path, cfg: ScriptedDemoConfig, render_every: int, camera_id: int) -> dict:
    # Pass 1: force-only, no rendering, just to fix the force panel's scale.
    dry = run_scripted_demo(cfg=cfg, seed=seed)
    scale_n = max(5.0, float(dry.peak_force_n) * 1.15)

    # Pass 2: same seed, same cfg -> identical trajectory, this time capturing frames.
    env = InsertionEnv(seed=seed)
    running_force: list = []
    frames: list = []
    step_counter = {"i": 0}

    def _cb(env_, force_n, phase):
        running_force.append(force_n)
        step_counter["i"] += 1
        if step_counter["i"] % render_every == 0:
            scene = env_.physics.render(height=SCENE_H, width=WIDTH, camera_id=camera_id)
            panel = _force_panel(running_force, scale_n, WIDTH, PANEL_H)
            frame = np.vstack([scene, panel])
            cv2.putText(frame, f"seed={seed}  phase={phase}  t={env_.data.time:5.2f}s",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(frame)

    result = run_scripted_demo(env=env, cfg=cfg, seed=seed, frame_callback=_cb)
    env.close()

    if frames:
        # Hold the last frame for a beat so success/failure is readable before the clip cuts.
        frames.extend([frames[-1]] * (FPS // 2))
        imageio.mimwrite(str(out_path), frames, fps=FPS, quality=7, macro_block_size=None)

    return {
        "seed": seed,
        "success": result.success,
        "termination_reason": result.termination_reason,
        "peak_force_n": result.peak_force_n,
        "mean_force_n": result.mean_force_n,
        "std_force_n": result.std_force_n,
        "num_frames": len(frames),
        "video_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="teleop/insertion_demo_videos")
    parser.add_argument("--num-demos", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--render-every", type=int, default=8,
                         help="capture 1 frame every this many control steps (control runs at 1kHz; "
                              "8 -> 125Hz raw capture, still generous headroom over the 30fps output)")
    parser.add_argument("--camera-id", type=int, default=0,
                         help="fixed scene camera defined in ground.xml (0 or 1); -1 for the default free camera")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ScriptedDemoConfig()
    summaries = []
    for i in range(args.num_demos):
        seed = args.start_seed + i
        out_path = out_dir / f"insertion_demo_seed{seed}.mp4"
        print(f"[{i + 1}/{args.num_demos}] rendering seed={seed} -> {out_path}")
        summary = render_one(seed, out_path, cfg, args.render_every, args.camera_id)
        print(f"    success={summary['success']} reason={summary['termination_reason']} "
              f"peak={summary['peak_force_n']:.1f}N mean={summary['mean_force_n']:.1f}N "
              f"std={summary['std_force_n']:.1f}N frames={summary['num_frames']}")
        summaries.append(summary)

    print("\n=== summary ===")
    for s in summaries:
        print(f"seed={s['seed']:>3}  success={s['success']!s:>5}  "
              f"peak={s['peak_force_n']:6.1f}N  mean={s['mean_force_n']:5.1f}N  std={s['std_force_n']:5.1f}N  "
              f"-> {s['video_path']}")


if __name__ == "__main__":
    main()
