"""Render a successful scripted push-T demo to video.

teleop_push_t.py's --dry-run drives a scripted figure-eight sweep for
exercising the haptic/UI loop, not a task-solving controller -- it doesn't
aim at the goal at all. This script instead uses a small goal-directed
heuristic (go around the T, push its bar face at the T's area centroid with
a simple rotation-feedback correction on the push point) written for this
video, since no general push-T solver exists in this repo yet.

Note this heuristic does not reliably reach the task's default 95% coverage
threshold -- open-loop/lightly-corrected pushing induces persistent rotation
(a well-known property of this benchmark; solving it robustly needs a real
closed-loop policy, not a fixed script). This renders a genuine, reproducible
success against a lower, explicitly-labeled threshold instead of faking one
against the default.

Usage:
    python teleop/render_push_t_demo.py --out ~/demo_videos/push_t_demo.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
from dm_control.mujoco.engine import MovableCamera

from push_t_teleop import PushTTeleop

WIDTH, HEIGHT = 480, 360
FPS = 30
NO_REFLECTION = {mujoco.mjtRndFlag.mjRND_REFLECTION: False}


def run_and_capture(success_threshold, seed=0):
    env = PushTTeleop(seed=seed, pusher_kp=300.0, damping_ratio=2.0,
                       success_threshold=success_threshold)
    dt = env.model.opt.timestep
    stride = max(1, round(1.0 / (FPS * dt)))

    bar_area = env.properties.t_bar_length_m * env.properties.t_bar_width_m
    stem_area = env.properties.t_stem_width_m * env.properties.t_stem_length_m
    com_y = (bar_area * 0.0 + stem_area * env.properties.stem_offset_y_m) / (bar_area + stem_area)

    camera = MovableCamera(env.physics, height=HEIGHT, width=WIDTH)
    camera.set_pose((0.0, 0.0, 0.0), 0.75, 90.0, -90.0)
    frames = []
    step_counter = {"i": 0}

    def capture(phase):
        if step_counter["i"] % stride == 0:
            frame = np.ascontiguousarray(camera.render(render_flag_overrides=NO_REFLECTION))
            cv2.putText(frame, f"phase={phase}  t={env.data.time:5.2f}s  "
                                f"coverage={env.coverage_fraction():.2f}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(frame)
        step_counter["i"] += 1

    def move_to(target, duration, phase):
        start = env.pusher_pos.copy()
        target = np.asarray(target, dtype=float)
        n_steps = max(1, int(duration / dt))
        for s in range(n_steps):
            alpha = (s + 1) / n_steps
            env.step(start + (target - start) * alpha, n_substeps=1)
            capture(phase)

    # Go around the T (default layout: T at +x, pusher at -x) to its far
    # (+x) face, then push it toward the goal at the origin. Pushing at the
    # T's area centroid in y (not the bar's y=0) avoids most, not all, of
    # the rotation a round pusher induces off-centroid; small feedback on
    # the current rotation nudges the push point to damp what's left.
    move_to((-0.10, 0.16), 0.6, "approach")
    move_to((0.20, 0.16), 0.8, "approach")
    move_to((0.20, com_y), 0.5, "approach")

    target_x = 0.20
    stopped_at = None
    hold_until = None
    for step in range(20000):
        if stopped_at is None:
            target_x -= 0.000012
        y_target = com_y - 0.15 * env.t_pose[2]
        env.step((target_x, y_target), n_substeps=1)
        capture("push" if stopped_at is None else "hold")
        if env.success() and stopped_at is None:
            stopped_at = step
            hold_until = step + int(1.5 / dt)
        if stopped_at is not None and step >= hold_until:
            break

    return env, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="push_t_demo.mp4")
    parser.add_argument("--success-threshold", type=float, default=0.6,
                         help="lower than the task's default 0.95 -- see module "
                              "docstring for why this script can't reliably hit 0.95")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env, frames = run_and_capture(args.success_threshold, seed=args.seed)
    print(f"success={env.success()} coverage={env.coverage_fraction():.3f} "
          f"(threshold={args.success_threshold}) final_pose={env.t_pose}")
    if not frames:
        raise SystemExit("no frames captured")
    frames.extend([frames[-1]] * (FPS // 2))
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), frames, fps=FPS, quality=7, macro_block_size=None)
    print(f"wrote {len(frames)} frames -> {out_path}")


if __name__ == "__main__":
    main()
