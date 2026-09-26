"""Haptic teleoperation for the 2D Push-Circle task.

Push-T's simpler sibling: a circular puck instead of a T-shape, so there's
no orientation/rotation to reason about at all -- just get the puck's center
into the goal circle. See push_circle_teleop.py for the physics (closed-form
circle-overlap coverage, no rasterized grid needed).

The goal can relocate mid-episode, same as push-T's --goal-move-* (see
teleop_push_t.py), but here it can also be biased to appear BEHIND the
puck's current direction of travel (--goal-move-opposite-bias), so pushing
toward where the goal used to be is unreliable -- verified this actually
places new goals within the configured angular spread of "opposite the
current velocity" once the puck is moving.

Dataset collection, the keep/delete review workflow, and the single
composited OpenCV viewer all mirror teleop_push_t.py directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections import deque

import numpy as np

from push_circle_teleop import PushCircleProperties, PushCircleTeleop

# Same convention as teleop_push_t.py's --axes/--scale -- see that module
# for the full rationale (duplicated rather than imported so this script
# stays self-contained).
DEFAULT_AXES = "y,z"
DEVICE_WORKSPACE_HALF_M = np.array([0.045, 0.040, 0.048])
DEFAULT_WORKSPACE_HALF_M = 0.30
DEFAULT_SCALE = tuple(float(v) for v in DEFAULT_WORKSPACE_HALF_M / DEVICE_WORKSPACE_HALF_M)

# ---- composited view layout (all in pixels) --------------------------------
CAM_W, CAM_H = 640, 480
PLOT_H = 130
STATUS_H = 26
BUTTON_H = 46
WINDOW_NAME = "Push-Circle"
BUTTON_RECTS = {
    "record": (10, 6, 190, BUTTON_H - 6),
    "keep": (210, 6, 330, BUTTON_H - 6),
    "delete": (350, 6, 470, BUTTON_H - 6),
}


def build_pos_map_2d(spec):
    """Same signed 2-of-3-axis selection as teleop_push_t.py's."""
    idx = {"x": 0, "y": 1, "z": 2}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"--axes needs 2 entries, got {spec!r}")
    m = np.zeros((2, 3))
    used = set()
    for row, part in enumerate(parts):
        sign = -1.0 if part.startswith("-") else 1.0
        name = part.lstrip("+-").lower()
        if name not in idx:
            raise ValueError(f"--axes entry {part!r} must name x, y or z")
        if name in used:
            raise ValueError(
                f"--axes {spec!r} reuses device axis {name!r}; sim x and y "
                f"must be driven by two different device axes"
            )
        used.add(name)
        m[row, idx[name]] = sign
    return m


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Haptic teleoperation for the 2D Push-Circle task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- task physics ------------------------------------------------------
    parser.add_argument("--object-mass", type=float, default=PushCircleProperties.object_mass_kg,
                        help="mass of the puck (kg)")
    parser.add_argument("--object-radius", type=float, default=PushCircleProperties.object_radius_m,
                        help="puck radius (m)")
    parser.add_argument("--object-thickness", type=float, default=PushCircleProperties.object_thickness_m,
                        help="puck thickness (m)")
    parser.add_argument("--goal-radius", type=float, default=PushCircleProperties.goal_radius_m,
                        help="goal circle radius (m). Must be <= --object-radius "
                             "for --success-threshold to be reachable at all "
                             "(coverage caps at (object_radius/goal_radius)^2)")
    parser.add_argument("--table-friction", type=float, default=PushCircleProperties.table_friction,
                        help="object <-> table sliding friction coefficient. "
                             "Low (e.g. 0.05) lets it coast noticeably after a "
                             "push; high (e.g. 1.5+) stops it almost as soon as "
                             "the pusher lets go")
    parser.add_argument("--table-torsional-friction", type=float,
                        default=PushCircleProperties.table_torsional_friction,
                        help="currently a no-op (condim=3); see push_t's flag "
                             "of the same name for why")
    parser.add_argument("--table-rolling-friction", type=float,
                        default=PushCircleProperties.table_rolling_friction,
                        help="currently a no-op (condim=3); see push_t's flag "
                             "of the same name for why")
    parser.add_argument("--pusher-friction", type=float, default=PushCircleProperties.pusher_friction,
                        help="pusher <-> object sliding friction on side contact")
    parser.add_argument("--pusher-radius", type=float, default=PushCircleProperties.pusher_radius_m,
                        help="pusher disc radius (m)")
    parser.add_argument("--pusher-mass", type=float, default=PushCircleProperties.pusher_mass_kg,
                        help="pusher mass (kg)")

    # ---- controller ---------------------------------------------------------
    parser.add_argument("--pusher-kp", type=float, default=PushCircleTeleop.default_tool_kp,
                        help="Cartesian impedance stiffness (N/m) driving the "
                             "pusher toward the operator's target")
    parser.add_argument("--damping-ratio", type=float, default=1.2,
                        help="pusher damping ratio (1.0 = critically damped)")
    parser.add_argument("--pusher-softness", type=float, default=0.0,
                        help="[0,1] softens the pusher<->object contact solver "
                             "response. Raise this first if forces jitter/spike, "
                             "especially after raising --pusher-kp")
    parser.add_argument("--force-sensor-cutoff", type=float, default=0.0,
                        help="cutoff (Hz) of a causal two-pole F/T sensor model "
                             "applied to the RECORDED wrench_0 only (haptic "
                             "feedback always uses raw contact). 0 = raw")
    parser.add_argument("--pusher-joint-damping", type=float, default=1.0,
                        help="physical damping (N s/m) on the pusher's slide "
                             "joints, separate from the controller's task-space kd")
    parser.add_argument("--noslip-iterations", type=int, default=2,
                        help="extra PGS passes refining the friction-cone "
                             "solution; 0 disables it")
    parser.add_argument("--disturbance-force", type=float, default=0.0,
                        help="stationary-std magnitude (N) of an unforced, "
                             "unpredictable Ornstein-Uhlenbeck push applied "
                             "directly to the puck. 0 (default) disables it")
    parser.add_argument("--disturbance-tau", type=float, default=0.8,
                        help="correlation time (s) of the disturbance's random "
                             "walk; only matters if --disturbance-force is nonzero")
    parser.add_argument("--disturbance-seed", type=int, default=None,
                        help="seed for the disturbance's random stream; "
                             "default derives one from --seed")
    parser.add_argument("--goal-move-min-interval", type=float, default=0.0,
                        help="minimum seconds between goal relocations; the "
                             "actual interval is redrawn uniformly from "
                             "[this, --goal-move-max-interval] every time. 0 "
                             "with max also 0 (default) disables goal-moving")
    parser.add_argument("--goal-move-max-interval", type=float, default=0.0,
                        help="maximum seconds between goal relocations; try "
                             "0.5-2 for 'often'")
    parser.add_argument("--goal-move-xy-half", type=float, default=0.15,
                        help="half-extent (m) of the square region a new goal "
                             "position is drawn from when NOT using the "
                             "opposite-direction bias (or as a fallback clip "
                             "bound when it is)")
    parser.add_argument("--goal-move-skip-prob", type=float, default=0.0,
                        help="probability [0,1) a scheduled relocation check "
                             "does not move the goal after all")
    parser.add_argument("--goal-move-opposite-bias", type=float, default=0.0,
                        help="probability [0,1] that a relocation is placed "
                             "BEHIND the puck's current direction of travel "
                             "instead of drawn uniformly -- 0 (default) is "
                             "push-T-style pure-uniform relocation; try 0.7-0.9 "
                             "for 'usually opposite, so it usually has to move "
                             "backward'. Falls back to uniform whenever the "
                             "puck's speed is below --goal-move-velocity-threshold "
                             "(there's no 'direction of travel' to oppose "
                             "while it's essentially stationary)")
    parser.add_argument("--goal-move-opposite-spread-deg", type=float, default=50.0,
                        help="+/- degrees of randomness around the exact "
                             "opposite-of-travel direction, so the backward "
                             "bias isn't a perfectly predictable single point")
    parser.add_argument("--goal-move-distance", type=float, default=0.15,
                        help="how far (m) behind the puck's current position "
                             "an opposite-biased relocation is placed (drawn "
                             "uniformly from half this to this)")
    parser.add_argument("--goal-move-velocity-threshold", type=float, default=0.03,
                        help="puck speed (m/s) below which relocations fall "
                             "back to uniform regardless of --goal-move-opposite-bias")
    parser.add_argument("--goal-move-seed", type=int, default=None,
                        help="seed for the goal-relocation random stream; "
                             "default derives one from --seed")
    parser.add_argument("--workspace-half", type=float, default=DEFAULT_WORKSPACE_HALF_M,
                        help="half-extent (m) of the square workspace the "
                             "pusher target is clamped to")
    parser.add_argument("--max-speed", type=float, default=0.5,
                        help="cap on how fast the commanded pusher target may "
                             "travel (m/s), 0 = uncapped")

    # ---- task / goal --------------------------------------------------------
    parser.add_argument("--goal-xy", type=float, nargs=2, default=(0.0, 0.0),
                        metavar=("X", "Y"), help="fixed initial goal position (m)")
    parser.add_argument("--success-threshold", type=float, default=0.85,
                        help="fraction of the goal circle's area that must be "
                             "covered by the puck to count as success")
    parser.add_argument("--randomize-start", action="store_true",
                        help="randomize the puck and pusher start position "
                             "each episode instead of the fixed default layout")
    parser.add_argument("--start-center-prob", type=float, default=0.70,
                        help="with --randomize-start, probability of sampling "
                             "from a center-biased Gaussian instead of "
                             "uniformly over the full start range")
    parser.add_argument("--start-object-xy-half", type=float, default=0.15,
                        help="with --randomize-start, half-extent (m) of the "
                             "puck start position range")
    parser.add_argument("--start-pusher-xy-half", type=float, default=0.20,
                        help="with --randomize-start, half-extent (m) of the "
                             "pusher start position range")
    parser.add_argument("--seed", type=int, default=0)

    # ---- haptics -------------------------------------------------------------
    parser.add_argument("--stiffness", type=float, default=1200.0,
                        help="target stiffness AT THE HANDLE (N/m); force-gain "
                             "is derived as stiffness/(pusher_kp*scale) unless "
                             "--force-gain is given")
    parser.add_argument("--force-gain", type=float, default=None,
                        help="N of handle force per N of sim contact force")
    parser.add_argument("--force-clip", type=float, default=40.0,
                        help="ceiling on the reflected sim force (N) before gain")
    parser.add_argument("--max-force", type=float, default=8.0,
                        help="clamp on the handle force vector magnitude (N)")
    parser.add_argument("--force-tau", type=float, default=2.0,
                        help="handle-force smoothing time constant (ms)")
    parser.add_argument("--force-rate", type=float, default=80.0,
                        help="cap on how fast the handle force may change (N/s)")
    parser.add_argument("--damping", type=float, default=15.0,
                        help="handle velocity damping (N/(m/s))")
    parser.add_argument("--scale", type=float, nargs=3, default=DEFAULT_SCALE,
                        metavar=("SX", "SY", "SZ"),
                        help="handle-displacement-to-task-target scale, one "
                             "value per device axis; default maps the omega's "
                             "full comfortable range onto --workspace-half")
    parser.add_argument("--home", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"),
                        help="physical handle position (device m) mapped to origin")
    parser.add_argument("--axes", type=str, default=DEFAULT_AXES,
                        help="which 2 of the 3 device axes drive sim x/y; "
                             "default 'y,z' (left/right -> x, up/down -> y)")
    parser.add_argument("--auto-init", action="store_true",
                        help="auto-calibrate the omega on open (it will move)")

    # ---- dataset collection ---------------------------------------------------
    parser.add_argument("--collect-dataset", type=str, default=None,
                        help="path to a Zarr dataset; enables recording")
    parser.add_argument("--auto-finish", action="store_true",
                        help="automatically stop an episode the instant "
                             "coverage crosses --success-threshold")
    parser.add_argument("--dataset-hz", type=float, default=1000.0,
                        help="recording rate (Hz)")
    parser.add_argument("--dataset-min-samples", type=int, default=20,
                        help="episodes shorter than this are discarded automatically")

    # ---- view / plot ---------------------------------------------------------
    parser.add_argument("--no-view", action="store_true",
                        help="disable the viewer entirely (headless)")
    parser.add_argument("--no-plot", action="store_true",
                        help="hide the live force strip-chart panel")
    parser.add_argument("--plot-span", type=float, default=4.0,
                        help="seconds of force history shown in the live plot")
    parser.add_argument("--plot-smoothing-hz", type=float, default=8.0,
                        help="display-only low-pass cutoff (Hz) for the plotted "
                             "force trace; does not touch recorded data or haptics")
    parser.add_argument("--view-fps", type=float, default=30.0,
                        help="viewer redraw rate (Hz)")
    parser.add_argument("--record-video", type=str, default=None,
                        help="path to write an mp4 of the composited view")
    parser.add_argument("--video-fps", type=float, default=30.0,
                        help="frame rate of --record-video's output file")

    # ---- loop -----------------------------------------------------------------
    parser.add_argument("--control-freq", type=int, default=1000,
                        help="sim + control loop rate (Hz)")
    parser.add_argument("--dry-run", action="store_true",
                        help="run a scripted demo motion with no hardware attached")
    parser.add_argument("--dry-run-seconds", type=float, default=20.0,
                        help="how long the --dry-run scripted demo runs")
    parser.add_argument("--print-interval-s", type=float, default=1.0,
                        help="how often to print coverage/status while running")

    return parser


def _derive_force_gain(args, pos_map):
    if args.force_gain is not None:
        return float(args.force_gain)
    selected_scale = np.abs(pos_map) @ np.asarray(args.scale, dtype=float)
    scale = 0.5 * (selected_scale[0] + selected_scale[1])
    return float(args.stiffness / max(args.pusher_kp * scale, 1e-9))


def _properties_from_args(args):
    return PushCircleProperties(
        object_mass_kg=args.object_mass,
        object_radius_m=args.object_radius,
        object_thickness_m=args.object_thickness,
        goal_radius_m=args.goal_radius,
        table_friction=args.table_friction,
        table_torsional_friction=args.table_torsional_friction,
        table_rolling_friction=args.table_rolling_friction,
        pusher_friction=args.pusher_friction,
        pusher_radius_m=args.pusher_radius,
        pusher_mass_kg=args.pusher_mass,
    )


def _scripted_dry_run_target(t_s, workspace_half):
    radius = 0.5 * workspace_half
    x = radius * np.sin(0.5 * t_s)
    y = radius * np.sin(0.25 * t_s)
    return np.array([x, y])


def _draw_plot_panel(trace_t, trace_mag, trace_fx, trace_fy, width, height):
    import cv2

    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    if not trace_t:
        return panel
    t_arr = np.asarray(trace_t)
    mag, fx, fy = np.asarray(trace_mag), np.asarray(trace_fx), np.asarray(trace_fy)
    finite = np.concatenate([mag, fx, fy])
    if finite.size and np.any(np.isfinite(finite)):
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    else:
        lo, hi = -1.0, 1.0
    pad = 0.1 * max(hi - lo, 1.0)
    lo, hi = lo - pad, hi + pad
    span_t = max(t_arr[-1] - t_arr[0], 1e-6)

    def polyline(values, color):
        xs = ((t_arr - t_arr[0]) / span_t * (width - 1)).astype(np.int32)
        ys = (height - 1 - (values - lo) / max(hi - lo, 1e-6) * (height - 1)).astype(np.int32)
        pts = np.stack([xs, ys], axis=1)
        if len(pts) > 1:
            cv2.polylines(panel, [pts], False, color, 1, cv2.LINE_AA)

    zero_y = int(np.clip(height - 1 - (0.0 - lo) / max(hi - lo, 1e-6) * (height - 1), 0, height - 1))
    cv2.line(panel, (0, zero_y), (width, zero_y), (210, 210, 210), 1, cv2.LINE_AA)
    polyline(fx, (200, 120, 20))
    polyline(fy, (20, 140, 230))
    polyline(mag, (0, 0, 0))
    cv2.putText(
        panel, f"|F| black   Fx blue   Fy orange   range [{lo:.1f}, {hi:.1f}] N",
        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA,
    )
    return panel


def _draw_status_panel(state, width):
    import cv2

    panel = np.full((STATUS_H, width, 3), 255, dtype=np.uint8)
    label, color = {
        "disabled": ("", (0, 0, 0)),
        "idle": ("idle -- click START or long-press the handle", (60, 60, 60)),
        "recording": ("RECORDING", (30, 30, 220)),
        "review": ("choose KEEP or DELETE", (0, 140, 240)),
    }[state]
    cv2.putText(panel, label, (8, STATUS_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return panel


def _draw_button_panel(state, width):
    import cv2

    panel = np.full((BUTTON_H, width, 3), 235, dtype=np.uint8)
    recording = state == "recording"
    review = state == "review"
    buttons = {
        "record": (
            "STOP" if recording else "START",
            (110, 110, 230) if recording else (140, 200, 140),
            False if review else True,
        ),
        "keep": ("KEEP", (140, 200, 140), review),
        "delete": ("DELETE", (110, 110, 230), review),
    }
    for name, (label, active_color, active) in buttons.items():
        x0, y0, x1, y1 = BUTTON_RECTS[name]
        color = active_color if active else (210, 210, 210)
        cv2.rectangle(panel, (x0, y0), (x1, y1), color, -1)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (120, 120, 120), 1)
        text_color = (30, 30, 30) if active else (160, 160, 160)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(
            panel, label,
            (x0 + ((x1 - x0) - tw) // 2, y0 + ((y1 - y0) + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA,
        )
    return panel


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.control_freq <= 0:
        parser.error("--control-freq must be positive")
    if args.pusher_kp <= 0.0:
        parser.error("--pusher-kp must be positive")
    if not 0.0 < args.success_threshold <= 1.0:
        parser.error("--success-threshold must be in (0, 1]")
    if not 0.0 <= args.start_center_prob <= 1.0:
        parser.error("--start-center-prob must be in [0, 1]")
    if args.dataset_hz <= 0.0:
        parser.error("--dataset-hz must be positive")
    pos_map = build_pos_map_2d(args.axes)

    def do_randomize_start(*, force_center=False):
        env.randomize_start(
            object_xy_half=args.start_object_xy_half,
            pusher_xy_half=args.start_pusher_xy_half,
            center_probability=args.start_center_prob,
            force_center=force_center,
        )

    goal_xy = np.array(args.goal_xy, dtype=float)
    properties = _properties_from_args(args)
    env = PushCircleTeleop(
        seed=args.seed,
        properties=properties,
        goal_xy=goal_xy,
        pusher_kp=args.pusher_kp,
        damping_ratio=args.damping_ratio,
        pusher_softness=args.pusher_softness,
        force_sensor_cutoff_hz=args.force_sensor_cutoff,
        pusher_joint_damping=args.pusher_joint_damping,
        noslip_iterations=args.noslip_iterations,
        disturbance_force_n=args.disturbance_force,
        disturbance_tau_s=args.disturbance_tau,
        disturbance_seed=args.disturbance_seed,
        goal_move_min_interval_s=args.goal_move_min_interval,
        goal_move_max_interval_s=args.goal_move_max_interval,
        goal_move_xy_half_m=args.goal_move_xy_half,
        goal_move_skip_prob=args.goal_move_skip_prob,
        goal_move_opposite_bias=args.goal_move_opposite_bias,
        goal_move_opposite_spread_deg=args.goal_move_opposite_spread_deg,
        goal_move_distance_m=args.goal_move_distance,
        goal_move_velocity_threshold_mps=args.goal_move_velocity_threshold,
        goal_move_seed=args.goal_move_seed,
        workspace_half_m=args.workspace_half,
        success_threshold=args.success_threshold,
    )
    if args.randomize_start:
        do_randomize_start()

    force_gain = _derive_force_gain(args, pos_map)
    print(
        f"push-circle ready: table_friction={args.table_friction:.3f} "
        f"pusher_friction={args.pusher_friction:.3f} pusher_kp={args.pusher_kp:.0f} "
        f"force_gain={force_gain:.4f} N_handle/N_sim goal={goal_xy}"
    )
    print(f"[axes] device->sim mapping {args.axes} (sim x, sim y in that order)")

    device = None
    if not args.dry_run:
        from fd_omega import FDOmega

        device = FDOmega(
            auto_init=args.auto_init,
            read_orientation=False,
            spring_k=0.0,
            reflected_tau_s=args.force_tau / 1000.0,
            reflected_rate=args.force_rate,
            damping_b=args.damping,
            max_force=args.max_force,
            home_pos=np.array(args.home, dtype=float),
        ).open()

    # ------------------------------------------------------------- dataset
    recorder = None
    if args.collect_dataset:
        from push_circle_recorder import PushCircleEpisodeRecorder

        recorder = PushCircleEpisodeRecorder(
            args.collect_dataset, sample_hz=args.dataset_hz, min_samples=args.dataset_min_samples,
        )
        print(
            f"[dataset] recording to {recorder.dataset_path} "
            f"({recorder.sample_count} samples already committed across "
            f"{len(recorder.episode_names)} episodes)"
        )
        if args.dry_run:
            print("[dataset] --dry-run: recording starts immediately and "
                  "auto-keeps when the run ends")
        elif args.no_view:
            print("[dataset] --no-view: use the handle -- LONG-PRESS to "
                  "START/STOP an episode, then short-press (KEEP) or "
                  "long-press again (DELETE) to resolve it")
        else:
            print("[dataset] use the START/STOP/KEEP/DELETE buttons in the "
                  "viewer (the handle's short/long press also works)")
    sample_stride = max(1, int(round(args.control_freq / args.dataset_hz)))

    collection = {
        "state": "idle" if recorder is not None else "disabled",
        "reason": None,
        "success": False,
        "final_coverage_fraction": None,
        "episode_tick": 0,
    }

    def episode_metadata():
        return {
            "seed": args.seed,
            "goal_xy": goal_xy.tolist(),
            "properties": dataclasses.asdict(properties),
            "pusher_kp": env.pusher_kp,
            "damping_ratio": env.damping_ratio,
            "pusher_softness": env.pusher_softness,
            "force_sensor_cutoff_hz": env.force_sensor_cutoff_hz,
            "pusher_joint_damping": env.pusher_joint_damping,
            "noslip_iterations": env.noslip_iterations,
            "disturbance_force_n": env.disturbance_force_n,
            "disturbance_tau_s": env.disturbance_tau_s,
            "goal_move_min_interval_s": env.goal_move_min_interval_s,
            "goal_move_max_interval_s": env.goal_move_max_interval_s,
            "goal_move_xy_half_m": env.goal_move_xy_half_m,
            "goal_move_skip_prob": env.goal_move_skip_prob,
            "goal_move_opposite_bias": env.goal_move_opposite_bias,
            "goal_move_distance_m": env.goal_move_distance_m,
            "workspace_half_m": env.workspace_half_m,
            "success_threshold": env.success_threshold,
            "axes": args.axes,
            "scale": list(args.scale),
            "home": list(args.home),
            "force_gain": force_gain,
            "randomize_start": args.randomize_start,
            "start_center_prob": args.start_center_prob,
            "command_line": vars(args),
        }

    def start_recorded_episode():
        if collection["state"] != "idle":
            return
        recorder.start_episode(episode_metadata())
        collection["state"] = "recording"
        collection["episode_tick"] = 0
        print("[dataset] recording STARTED")

    def stop_recorded_episode(reason):
        if collection["state"] != "recording":
            return
        collection["success"] = bool(env.success())
        collection["final_coverage_fraction"] = float(env.coverage_fraction())
        collection["reason"] = reason
        collection["state"] = "review"
        if device is not None:
            device.clear_reflected_force()
        print(
            f"[dataset] episode stopped ({reason}); "
            f"success={collection['success']} "
            f"coverage={collection['final_coverage_fraction']:.3f} "
            f"samples={recorder.sample_count}"
        )
        if args.no_view:
            print("[dataset] KEEP: handle short press | DELETE: handle long press")
        else:
            print("[dataset] click KEEP or DELETE in the viewer, or handle short/long press")

    def finish_recorded_episode(save):
        pending_count = recorder.sample_count
        if save:
            name = recorder.commit(
                success=collection["success"],
                termination_reason=collection["reason"] or "unknown",
                final_coverage_fraction=collection["final_coverage_fraction"] or 0.0,
            )
            if name is None:
                print(
                    f"[dataset] episode had only {pending_count} samples "
                    f"(< --dataset-min-samples {args.dataset_min_samples}); discarded automatically"
                )
            else:
                print(f"[dataset] KEPT as {name} ({pending_count} samples)")
        else:
            recorder.discard()
            print(f"[dataset] DELETED ({pending_count} samples)")
        collection["state"] = "idle"
        collection["reason"] = None
        collection["success"] = False
        collection["final_coverage_fraction"] = None

    def resolve_recorded_episode(keep):
        finish_recorded_episode(save=keep)
        env.reset()
        if args.randomize_start:
            do_randomize_start()
        plot_smoothed[:] = 0.0

    # ------------------------------------------------------------ the view
    view = None
    plot_action = [None]
    plot_len = max(2, int(round(args.plot_span * args.control_freq)))
    trace_t = deque(maxlen=plot_len)
    trace_fx = deque(maxlen=plot_len)
    trace_fy = deque(maxlen=plot_len)
    trace_mag = deque(maxlen=plot_len)
    plot_smoothed = np.zeros(2)
    plot_smoothing_alpha = None
    if args.plot_smoothing_hz > 0.0:
        plot_tau = 1.0 / (2.0 * np.pi * args.plot_smoothing_hz)
        plot_smoothing_alpha = 1.0 - np.exp(-(1.0 / args.control_freq) / plot_tau)
    show_plot = not args.no_plot
    show_buttons = recorder is not None
    button_panel_y0 = CAM_H + (PLOT_H if show_plot else 0) + (STATUS_H if show_buttons else 0)

    need_render = (not args.no_view) or bool(args.record_video)
    video_frames = [] if args.record_video else None
    video_every = max(1, int(round(args.control_freq / max(args.video_fps, 1e-6))))
    if need_render:
        import cv2
        from dm_control.mujoco.engine import MovableCamera

        camera = MovableCamera(env.physics, height=CAM_H, width=CAM_W)
        camera.set_pose((0.0, 0.0, 0.0), 0.9, 90.0, -90.0)
        view = {"camera": camera, "cv2": cv2, "running": True}

        if not args.no_view:
            def on_mouse(event, x, y, _flags, _userdata):
                if event != cv2.EVENT_LBUTTONUP or not show_buttons:
                    return
                local_y = y - button_panel_y0
                for name, (x0, y0, x1, y1) in BUTTON_RECTS.items():
                    if x0 <= x <= x1 and y0 <= local_y <= y1:
                        plot_action[0] = "toggle_recording" if name == "record" else name
                        return

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    ui_ready_at = time.monotonic() + 0.25

    def poll_ui_action():
        action, plot_action[0] = plot_action[0], None
        if action is not None and time.monotonic() < ui_ready_at:
            return
        if action == "toggle_recording" and recorder is not None:
            if collection["state"] == "idle":
                start_recorded_episode()
            elif collection["state"] == "recording":
                stop_recorded_episode("ui_button")
        elif action == "keep" and collection["state"] == "review":
            resolve_recorded_episode(True)
        elif action == "delete" and collection["state"] == "review":
            resolve_recorded_episode(False)

    def build_canvas():
        cv2 = view["cv2"]
        frame = view["camera"].render()
        canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        canvas = np.ascontiguousarray(canvas)
        if show_plot:
            canvas = np.vstack(
                [canvas, _draw_plot_panel(trace_t, trace_mag, trace_fx, trace_fy, CAM_W, PLOT_H)]
            )
        if show_buttons:
            canvas = np.vstack(
                [canvas, _draw_status_panel(collection["state"], CAM_W),
                 _draw_button_panel(collection["state"], CAM_W)]
            )
        return canvas

    def capture_video_frame():
        if video_frames is not None:
            video_frames.append(build_canvas())

    def refresh_view():
        if view is None or args.no_view:
            return True
        canvas = build_canvas()
        cv2 = view["cv2"]
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("k"):
            plot_action[0] = "keep"
        elif key == ord("d"):
            plot_action[0] = "delete"
        elif key == ord("s"):
            plot_action[0] = "toggle_recording"
        try:
            still_open = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            still_open = False
        return still_open

    if recorder is not None and args.dry_run:
        start_recorded_episode()

    dt = 1.0 / float(args.control_freq)
    max_step = args.max_speed * dt if args.max_speed > 0.0 else float("inf")
    target = env.pusher_pos.copy()
    prev_success = False
    t_start = time.monotonic()
    last_print = t_start
    last_view = t_start
    tick_index = 0
    last_long = device.get_state()["long_press_count"] if device is not None else 0
    last_short = device.get_state()["short_press_count"] if device is not None else 0

    try:
        while True:
            now = time.monotonic()
            t_elapsed = now - t_start
            if args.dry_run and t_elapsed >= args.dry_run_seconds:
                break

            poll_ui_action()

            if collection["state"] == "review":
                if args.dry_run:
                    resolve_recorded_episode(True)
                    continue
                state = device.get_state()
                if state["short_press_count"] != last_short:
                    last_short = state["short_press_count"]
                    plot_action[0] = "keep"
                if state["long_press_count"] != last_long:
                    last_long = state["long_press_count"]
                    plot_action[0] = "delete"
                poll_ui_action()
                if not refresh_view():
                    break
                time.sleep(0.01)
                continue

            if args.dry_run:
                desired = _scripted_dry_run_target(t_elapsed, args.workspace_half)
            else:
                state = device.get_state()
                device_xyz = state["pos"][:3]
                home_xyz = np.array(args.home, dtype=float)
                scale_xyz = np.array(args.scale, dtype=float)
                desired = pos_map @ (scale_xyz * (device_xyz - home_xyz))

            delta = desired - target
            step_norm = np.linalg.norm(delta)
            if step_norm > max_step:
                delta *= max_step / step_norm
            target = target + delta

            env.step(target, n_substeps=1)

            contact_force_xy = env.pusher_contact_force_xy()
            reflected_sim = np.clip(
                contact_force_xy * force_gain,
                -args.force_clip * force_gain,
                args.force_clip * force_gain,
            )
            reflected = pos_map.T @ reflected_sim
            if device is not None:
                device.set_reflected_force(reflected)
                state = device.get_state()
                if state["long_press_count"] != last_long:
                    last_long = state["long_press_count"]
                    if recorder is None:
                        env.reset()
                        if args.randomize_start:
                            do_randomize_start()
                        device.clear_reflected_force()
                        plot_smoothed[:] = 0.0
                    elif collection["state"] == "idle":
                        start_recorded_episode()
                    elif collection["state"] == "recording":
                        stop_recorded_episode("device_long_press")
                last_short = state["short_press_count"]

            if recorder is not None and collection["state"] == "recording":
                if tick_index % sample_stride == 0:
                    recorder.record_sample(
                        env,
                        timestamp_ms=collection["episode_tick"] * (1000.0 / args.dataset_hz),
                        target_xy=target,
                        device_state=(device.get_state() if device is not None else {}),
                        sent_force=reflected,
                        wall_time_ns=time.perf_counter_ns(),
                    )
                    collection["episode_tick"] += 1
                if args.auto_finish and env.success() and not prev_success:
                    stop_recorded_episode("auto_success")

            success_now = env.success()
            if success_now and not prev_success:
                print(f"*** SUCCESS: coverage={env.coverage_fraction():.3f} ***")
            prev_success = success_now

            if plot_smoothing_alpha is None:
                plotted_force = contact_force_xy
            else:
                plot_smoothed += plot_smoothing_alpha * (contact_force_xy - plot_smoothed)
                plotted_force = plot_smoothed
            trace_t.append(t_elapsed)
            trace_fx.append(plotted_force[0])
            trace_fy.append(plotted_force[1])
            trace_mag.append(float(np.linalg.norm(plotted_force)))

            if video_frames is not None and tick_index % video_every == 0:
                capture_video_frame()

            if view is not None and now - last_view >= 1.0 / max(args.view_fps, 1e-6):
                last_view = now
                if not refresh_view():
                    break

            if now - last_print >= args.print_interval_s:
                x, y = env.object_pos
                state_label = f" [{collection['state']}]" if recorder is not None else ""
                print(
                    f"t={t_elapsed:6.1f}s{state_label}  "
                    f"coverage={env.coverage_fraction():.3f}  "
                    f"object=({x:+.3f},{y:+.3f})  goal=({env.goal_xy[0]:+.3f},{env.goal_xy[1]:+.3f})  "
                    f"contact_force={np.linalg.norm(contact_force_xy):5.2f} N"
                )
                last_print = now

            tick_index += 1
            elapsed_this_tick = time.monotonic() - now
            sleep_s = dt - elapsed_this_tick
            if sleep_s > 0.0 and not args.dry_run:
                time.sleep(sleep_s)

        if recorder is not None and collection["state"] in ("recording", "review"):
            if collection["state"] == "recording":
                stop_recorded_episode("run_ended")
            if args.dry_run:
                resolve_recorded_episode(True)
            else:
                print("[dataset] run ending with an unresolved episode; deleting it")
                resolve_recorded_episode(False)
    finally:
        if device is not None:
            device.close()
        if view is not None and not args.no_view:
            view["cv2"].destroyWindow(WINDOW_NAME)
        if video_frames:
            import imageio
            from pathlib import Path as _Path

            out_path = _Path(args.record_video).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_frames = [frame[:, :, ::-1] for frame in video_frames]
            imageio.mimwrite(str(out_path), rgb_frames, fps=args.video_fps,
                              quality=7, macro_block_size=None)
            print(f"[video] wrote {len(video_frames)} frames -> {out_path}")


if __name__ == "__main__":
    main()
