"""Haptic teleoperation for the 2D Push-T task.

Moves a Force Dimension omega handle to drive a circular pusher that shoves
a T-shaped block toward a fixed goal pose. By default the handle's left/right
motion drives sim x and its up/down motion drives sim y (see --axes); the
operator feels the pusher/T-block contact force reflected at the handle.
--dry-run runs a scripted demo motion with no hardware attached, for testing.

Dataset collection (--collect-dataset), the keep/delete episode review
workflow, and the center-biased randomized reset mirror teleop_flipup.py's
behavior, adapted to push-T's simpler 2D state -- see
push_t_teleop.PushTTeleop.randomize_start and push_t_recorder.py. The camera
view, live force plot, and START/STOP/KEEP/DELETE buttons are one composited
OpenCV window (same approach as teleop_flipup.py's own viewer), not separate
popups.

See push_t_teleop.py for the physics (friction, inertia, contact model, and
the area-coverage success metric).
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections import deque

import numpy as np

from push_t_teleop import PushTProperties, PushTTeleop

# Force Dimension's device frame is +x toward the operator, +y to the
# operator's right, +z up. The view here looks straight down, so which
# device axis should drive sim x vs. y is not self-evident the way it is for
# flip-up's oblique camera. Default maps the device's left/right (y) to sim x
# and up/down (z) to sim y, leaving the device's toward/away-from-operator
# axis (x) unused -- override with --axes if pushing/lifting the handle
# moves the pusher along the wrong screen axis, in the wrong direction, or
# you want the device's horizontal plane instead (--axes x,y). Same
# signed-axis-selection convention as teleop_flipup.py's --axes (duplicated
# rather than imported so this script does not pull in the rest of that
# module), generalized to pick 2 of the 3 device axes since only 2 are
# needed to drive this task.
DEFAULT_AXES = "y,z"

# Force Dimension omega comfortable/safe workspace half-extents, in device
# metres -- the same hardware-calibrated figures teleop_sanding.py uses
# (DEVICE_WORKSPACE_HALF_M there; duplicated rather than imported so this
# script doesn't pull in the sanding task). Used below to derive a default
# --scale that maps the device's full comfortable range onto the full
# push-T workspace, instead of an arbitrary flat multiplier that left the
# operator unable to reach the workspace edge.
DEVICE_WORKSPACE_HALF_M = np.array([0.045, 0.040, 0.048])
# workspace_half_m / device_half, per device axis -- matches --workspace-half's
# own default (0.30) below; if you raise --workspace-half, raise --scale by
# the same ratio or the edges will be out of reach again.
DEFAULT_WORKSPACE_HALF_M = 0.30
DEFAULT_SCALE = tuple(float(v) for v in DEFAULT_WORKSPACE_HALF_M / DEVICE_WORKSPACE_HALF_M)

# ---- composited view layout (all in pixels) --------------------------------
CAM_W, CAM_H = 640, 480
PLOT_H = 130
STATUS_H = 26
BUTTON_H = 46
WINDOW_NAME = "Push-T"
# Button rects in LOCAL button-panel coordinates (x0, y0, x1, y1).
BUTTON_RECTS = {
    "record": (10, 6, 190, BUTTON_H - 6),
    "keep": (210, 6, 330, BUTTON_H - 6),
    "delete": (350, 6, 470, BUTTON_H - 6),
}


def build_pos_map_2d(spec):
    """Signed selection matrix mapping 2 of the 3 omega axes (x, y, z) onto
    the 2 sim axes (x, y). ``spec`` is two comma-separated entries naming
    which device axis (optionally negated) drives sim x and y, e.g. "x,y"
    (device's horizontal plane), "x,z" (device up/down drives sim y instead),
    "-x,y" (x reversed). The device axis not named is not read and receives
    no reflected force. Its transpose maps sim contact force back onto all
    three device axes, keeping felt resistance opposed to the motion that
    caused it."""
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
        description="Haptic teleoperation for the 2D Push-T task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- task physics ------------------------------------------------------
    parser.add_argument("--t-mass", type=float, default=PushTProperties.t_mass_kg,
                        help="mass of the T-block (kg)")
    parser.add_argument("--table-friction", type=float,
                        default=PushTProperties.table_friction,
                        help="T-block <-> table sliding friction coefficient. "
                             "Low (e.g. 0.05) lets the T coast noticeably after "
                             "a push; high (e.g. 1.5+) stops it almost as soon "
                             "as the pusher lets go. This is a real Coulomb "
                             "friction coefficient, not a special-cased slide/"
                             "stick switch")
    parser.add_argument("--table-torsional-friction", type=float,
                        default=PushTProperties.table_torsional_friction,
                        help="CURRENTLY A NO-OP: the table geom compiles at "
                             "condim=3, which only solves normal + 2 "
                             "tangential friction dimensions -- MuJoCo needs "
                             "condim>=4 for the torsional friction dimension "
                             "to be used at all. Kept/exposed for when condim "
                             "is raised; verified empirically to have zero "
                             "effect on rotation right now regardless of value")
    parser.add_argument("--table-rolling-friction", type=float,
                        default=PushTProperties.table_rolling_friction,
                        help="CURRENTLY A NO-OP for the same reason as "
                             "--table-torsional-friction: needs condim=6, "
                             "compiled geoms use condim=3")
    parser.add_argument("--pusher-friction", type=float,
                        default=PushTProperties.pusher_friction,
                        help="pusher <-> T-block sliding friction on side "
                             "contact. Higher lets the round pusher drag the T "
                             "tangentially (not just shove it straight back "
                             "along the contact normal) when nudged against a "
                             "face at an angle")
    parser.add_argument("--pusher-radius", type=float,
                        default=PushTProperties.pusher_radius_m,
                        help="pusher disc radius (m)")
    parser.add_argument("--pusher-mass", type=float,
                        default=PushTProperties.pusher_mass_kg,
                        help="pusher mass (kg)")
    parser.add_argument("--t-bar-length", type=float,
                        default=PushTProperties.t_bar_length_m, help="T bar length (m)")
    parser.add_argument("--t-bar-width", type=float,
                        default=PushTProperties.t_bar_width_m, help="T bar width (m)")
    parser.add_argument("--t-stem-length", type=float,
                        default=PushTProperties.t_stem_length_m, help="T stem length (m)")
    parser.add_argument("--t-stem-width", type=float,
                        default=PushTProperties.t_stem_width_m, help="T stem width (m)")
    parser.add_argument("--t-thickness", type=float,
                        default=PushTProperties.t_thickness_m, help="T thickness (m)")

    # ---- controller ---------------------------------------------------------
    parser.add_argument("--pusher-kp", type=float, default=PushTTeleop.default_tool_kp,
                        help="Cartesian impedance stiffness (N/m) driving the "
                             "pusher toward the operator's target")
    parser.add_argument("--damping-ratio", type=float, default=1.2,
                        help="pusher damping ratio (1.0 = critically damped)")
    parser.add_argument("--pusher-softness", type=float, default=0.0,
                        help="[0,1] softens the pusher<->T contact solver "
                             "response (same convention as flip-up's "
                             "--tip-softness): 0 is the stiff compiled "
                             "default, 1 widens/slows the contact correction. "
                             "Raise this first if forces jitter/spike, "
                             "especially after raising --pusher-kp")
    parser.add_argument("--force-sensor-cutoff", type=float, default=0.0,
                        help="cutoff in Hz of each pole in a causal two-pole "
                             "F/T sensor model applied to the RECORDED "
                             "wrench_0 (not the live haptic feedback, which "
                             "always uses raw contact). 0 keeps wrench_0 "
                             "identical to the raw contact force; try 30 for "
                             "a realistic smooth measurement. The raw signal "
                             "is discretely noisy tick-to-tick as MuJoCo's "
                             "contact solver adds/drops contact points even "
                             "while the commanded target is smooth -- this "
                             "isn't a bug, see PushTTeleop.sensor_force_xy. "
                             "wrench_ground_truth_0 always keeps the raw value")
    parser.add_argument("--pusher-joint-damping", type=float, default=1.0,
                        help="physical damping (N s/m) on the pusher's slide "
                             "joints themselves, separate from the "
                             "controller's task-space kd -- passive "
                             "dissipation MuJoCo applies every substep, which "
                             "can settle inter-tick velocity oscillations the "
                             "1 kHz controller can't see between its own "
                             "samples. Was hardcoded to 0 before; raise "
                             "further if the raw contact force still chatters")
    parser.add_argument("--noslip-iterations", type=int, default=2,
                        help="extra PGS passes MuJoCo runs after the main "
                             "solve to refine the friction-cone (slip-"
                             "direction) solution -- reduces the sliding "
                             "pusher's contact flip-flopping between nearby "
                             "friction solutions tick to tick. 0 disables it "
                             "(the previous, unset default)")
    parser.add_argument("--workspace-half", type=float, default=DEFAULT_WORKSPACE_HALF_M,
                        help="half-extent (m) of the square workspace the "
                             "pusher target is clamped to. --scale's default "
                             "is calibrated against this value -- raise both "
                             "together or the device's comfortable range will "
                             "undershoot/overshoot the workspace edge")
    parser.add_argument("--max-speed", type=float, default=0.5,
                        help="cap on how fast the commanded pusher target may "
                             "travel (m/s), 0 = uncapped")

    # ---- task / goal --------------------------------------------------------
    parser.add_argument("--goal-pose", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "THETA_DEG"),
                        help="fixed goal T pose (m, m, degrees)")
    parser.add_argument("--success-threshold", type=float, default=0.95,
                        help="fraction of the goal T footprint that must be "
                             "covered by the T-block to count as success")
    parser.add_argument("--randomize-start", action="store_true",
                        help="randomize the T and pusher start pose each "
                             "episode instead of using the fixed default layout")
    parser.add_argument("--start-center-prob", type=float, default=0.70,
                        help="with --randomize-start, probability of sampling "
                             "from a center-biased Gaussian instead of "
                             "uniformly over the full start range -- same "
                             "70%% center-bias convention as flip-up/cube-lift")
    parser.add_argument("--start-t-xy-half", type=float, default=0.15,
                        help="with --randomize-start, half-extent (m) of the "
                             "T-block start position range")
    parser.add_argument("--start-t-theta-half-deg", type=float, default=180.0,
                        help="with --randomize-start, half-extent (deg) of "
                             "the T-block start orientation range")
    parser.add_argument("--start-pusher-xy-half", type=float, default=0.20,
                        help="with --randomize-start, half-extent (m) of the "
                             "pusher start position range")
    parser.add_argument("--seed", type=int, default=0)

    # ---- haptics -------------------------------------------------------------
    parser.add_argument("--stiffness", type=float, default=1200.0,
                        help="target stiffness AT THE HANDLE (N/m). force-gain "
                             "is derived as stiffness/(pusher_kp*scale) unless "
                             "--force-gain is given")
    parser.add_argument("--force-gain", type=float, default=None,
                        help="N of handle force per N of sim contact force. "
                             "Default is derived from --stiffness")
    parser.add_argument("--force-clip", type=float, default=40.0,
                        help="ceiling on the reflected sim force (N) before "
                             "force-gain is applied")
    parser.add_argument("--max-force", type=float, default=8.0,
                        help="clamp on the handle force vector magnitude (N)")
    parser.add_argument("--force-tau", type=float, default=2.0,
                        help="handle-force smoothing time constant (ms), "
                             "0 = raw. Filters INSIDE the feedback loop, so "
                             "raising it to cure buzzing lowers the stiffness "
                             "you may safely render")
    parser.add_argument("--force-rate", type=float, default=80.0,
                        help="cap on how fast the handle force may change "
                             "(N/s), 0 = uncapped. Softens first-touch onset")
    parser.add_argument("--damping", type=float, default=15.0,
                        help="handle velocity damping (N/(m/s))")
    parser.add_argument("--scale", type=float, nargs=3, default=DEFAULT_SCALE,
                        metavar=("SX", "SY", "SZ"),
                        help="handle-displacement-to-task-target scale, one "
                             "value per device axis (x, y, z), applied before "
                             "--axes selects which 2 drive the sim. Default "
                             "maps the omega's full comfortable range "
                             "(DEVICE_WORKSPACE_HALF_M) onto the full "
                             "--workspace-half square, so the operator can "
                             "reach every edge without leaving a safe range")
    parser.add_argument("--home", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"),
                        help="physical handle position (device m) mapped to "
                             "task-space origin (0, 0)")
    parser.add_argument("--axes", type=str, default=DEFAULT_AXES,
                        help="which 2 of the 3 device axes (x, y, z; "
                             "optionally negated) drive sim x and y, "
                             "comma-separated, e.g. 'y,z' (default: device "
                             "left/right drives sim x, device up/down drives "
                             "sim y), 'x,y' (device's horizontal plane), "
                             "'y,x' (swapped), '-y,z' (x reversed). Fixes a "
                             "handle motion that moves "
                             "the pusher along the wrong axis, backwards, or "
                             "lets you drive with up/down instead")
    parser.add_argument("--auto-init", action="store_true",
                        help="auto-calibrate the omega on open (it will move)")

    # ---- dataset collection ---------------------------------------------------
    parser.add_argument("--collect-dataset", type=str, default=None,
                        help="path to a Zarr dataset; enables recording. Use "
                             "the START/STOP/KEEP/DELETE buttons in the "
                             "viewer, or the handle (see --no-view)")
    parser.add_argument("--auto-finish", action="store_true",
                        help="automatically stop (not keep) an episode the "
                             "instant coverage crosses --success-threshold; "
                             "you still choose KEEP or DELETE afterward")
    parser.add_argument("--dataset-hz", type=float, default=1000.0,
                        help="recording rate (Hz); rounds --control-freq / "
                             "--dataset-hz to the nearest integer stride")
    parser.add_argument("--dataset-min-samples", type=int, default=20,
                        help="episodes shorter than this many samples are "
                             "discarded automatically, even if KEEP is chosen")

    # ---- view / plot ---------------------------------------------------------
    parser.add_argument("--no-view", action="store_true",
                        help="disable the viewer entirely (headless). With a "
                             "recorder active, falls back to the handle's "
                             "short/long press for KEEP/DELETE and START/STOP")
    parser.add_argument("--no-plot", action="store_true",
                        help="hide the live force strip-chart panel; the "
                             "camera view and START/STOP/KEEP/DELETE buttons "
                             "(if a recorder is active) still show")
    parser.add_argument("--plot-span", type=float, default=4.0,
                        help="seconds of force history shown in the live plot")
    parser.add_argument("--plot-smoothing-hz", type=float, default=8.0,
                        help="single-pole low-pass cutoff (Hz) applied ONLY "
                             "to the plotted force trace -- display-only, "
                             "does not touch haptic feedback or any recorded "
                             "array (wrench_0/wrench_ground_truth_0 are "
                             "unaffected). The raw contact force is "
                             "genuinely noisy tick-to-tick (MuJoCo's solver "
                             "adds/drops contact points at 1 kHz even under "
                             "smooth motion -- see --force-sensor-cutoff's "
                             "help), so plotting it raw looks jittery even "
                             "when nothing is wrong. 0 plots the raw value")
    parser.add_argument("--view-fps", type=float, default=30.0,
                        help="viewer redraw rate (Hz); force data is still "
                             "sampled every control tick regardless")
    parser.add_argument("--record-video", type=str, default=None,
                        help="path to write an mp4 of the composited view "
                             "(camera + force plot). Works with --no-view "
                             "(renders offscreen only, no window shown)")
    parser.add_argument("--video-fps", type=float, default=30.0,
                        help="frame rate of --record-video's output file")

    # ---- loop -----------------------------------------------------------------
    parser.add_argument("--control-freq", type=int, default=1000,
                        help="sim + control loop rate (Hz)")
    parser.add_argument("--dry-run", action="store_true",
                        help="run a scripted demo motion with no Force "
                             "Dimension hardware attached")
    parser.add_argument("--dry-run-seconds", type=float, default=20.0,
                        help="how long the --dry-run scripted demo runs")
    parser.add_argument("--print-interval-s", type=float, default=1.0,
                        help="how often to print coverage/status while running")

    return parser


def _derive_force_gain(args, pos_map):
    if args.force_gain is not None:
        return float(args.force_gain)
    # Average the magnitude of the two device-axis scales --axes actually
    # selects, not always scale[0]/scale[1] -- otherwise e.g. --axes x,z
    # with a distinct --scale entry for z would silently use the wrong pair.
    selected_scale = np.abs(pos_map) @ np.asarray(args.scale, dtype=float)
    scale = 0.5 * (selected_scale[0] + selected_scale[1])
    return float(args.stiffness / max(args.pusher_kp * scale, 1e-9))


def _properties_from_args(args):
    return PushTProperties(
        t_mass_kg=args.t_mass,
        t_bar_length_m=args.t_bar_length,
        t_bar_width_m=args.t_bar_width,
        t_stem_length_m=args.t_stem_length,
        t_stem_width_m=args.t_stem_width,
        t_thickness_m=args.t_thickness,
        table_friction=args.table_friction,
        table_torsional_friction=args.table_torsional_friction,
        table_rolling_friction=args.table_rolling_friction,
        pusher_friction=args.pusher_friction,
        pusher_radius_m=args.pusher_radius,
        pusher_mass_kg=args.pusher_mass,
    )


def _scripted_dry_run_target(t_s, workspace_half):
    """A slow figure-eight, biased to sweep across the default T/goal layout."""
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
    polyline(fx, (200, 120, 20))    # BGR: blue-ish
    polyline(fy, (20, 140, 230))    # BGR: orange-ish
    polyline(mag, (0, 0, 0))        # black
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
        "recording": ("RECORDING", (30, 30, 220)),       # BGR red
        "review": ("choose KEEP or DELETE", (0, 140, 240)),  # BGR orange
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
    start_t_theta_half = np.radians(args.start_t_theta_half_deg)

    def do_randomize_start(*, force_center=False):
        env.randomize_start(
            t_xy_half=args.start_t_xy_half,
            t_theta_half=start_t_theta_half,
            pusher_xy_half=args.start_pusher_xy_half,
            center_probability=args.start_center_prob,
            force_center=force_center,
        )

    goal_pose = np.array(
        [args.goal_pose[0], args.goal_pose[1], np.radians(args.goal_pose[2])]
    )
    properties = _properties_from_args(args)
    env = PushTTeleop(
        seed=args.seed,
        properties=properties,
        goal_pose=goal_pose,
        pusher_kp=args.pusher_kp,
        damping_ratio=args.damping_ratio,
        pusher_softness=args.pusher_softness,
        force_sensor_cutoff_hz=args.force_sensor_cutoff,
        pusher_joint_damping=args.pusher_joint_damping,
        noslip_iterations=args.noslip_iterations,
        workspace_half_m=args.workspace_half,
        success_threshold=args.success_threshold,
    )
    if args.randomize_start:
        # Unlike flip-up/cube-lift, do NOT force the very first layout to the
        # exact center: there, the forced-center prism is the tool's start
        # region, distinct from the task object. Here the randomized pose IS
        # the T-block, and its center normally coincides with the default
        # goal pose (both default to the origin) -- forcing it would make
        # episode 1 start already at ~success. Use the normal mixture from
        # the first episode on.
        do_randomize_start()

    force_gain = _derive_force_gain(args, pos_map)
    print(
        f"push-T ready: table_friction={args.table_friction:.3f} "
        f"pusher_friction={args.pusher_friction:.3f} pusher_kp={args.pusher_kp:.0f} "
        f"force_gain={force_gain:.4f} N_handle/N_sim goal={goal_pose}"
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
        from push_t_recorder import PushTEpisodeRecorder

        recorder = PushTEpisodeRecorder(
            args.collect_dataset,
            sample_hz=args.dataset_hz,
            min_samples=args.dataset_min_samples,
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
                  "long-press again (DELETE) to resolve it"
                  + (" (or pass --auto-finish to stop automatically on "
                     "success)" if not args.auto_finish else ""))
        else:
            print("[dataset] use the START/STOP/KEEP/DELETE buttons in the "
                  "viewer (the handle's short/long press also works)"
                  + (" -- or pass --auto-finish to stop automatically on "
                     "success" if not args.auto_finish else ""))
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
            "goal_pose": goal_pose.tolist(),
            "properties": dataclasses.asdict(properties),
            "pusher_kp": env.pusher_kp,
            "damping_ratio": env.damping_ratio,
            "pusher_softness": env.pusher_softness,
            "force_sensor_cutoff_hz": env.force_sensor_cutoff_hz,
            "pusher_joint_damping": env.pusher_joint_damping,
            "noslip_iterations": env.noslip_iterations,
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
            print("[dataset] click KEEP or DELETE in the viewer, "
                  "or handle short/long press")

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
                    f"(< --dataset-min-samples {args.dataset_min_samples}); "
                    f"discarded automatically"
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
    # One OpenCV window stacking: camera render, force strip chart (unless
    # --no-plot), status line + START/STOP/KEEP/DELETE buttons (if a
    # recorder is active). Buttons are real clickable regions via a mouse
    # callback on this same window, not a separate popup.
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
    camera = None
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
                        plot_action[0] = (
                            "toggle_recording" if name == "record" else name
                        )
                        return

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # Some window managers/backends deliver a stray click or keypress right
    # when a window is created/gains focus; ignore UI actions for a brief
    # window after startup so that can't be misread as an operator click.
    ui_ready_at = time.monotonic() + 0.25

    def poll_ui_action():
        """Drain one pending button click and dispatch it. Clicks made while
        they don't apply (e.g. KEEP during recording) are simply ignored."""
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
        """Append one composited frame to --record-video's output, on a
        control-tick stride (not wall-clock time) so playback speed is
        accurate even during an unthrottled --dry-run that runs faster than
        real time."""
        if video_frames is not None:
            video_frames.append(build_canvas())

    def refresh_view():
        """Render one composited frame for the interactive window, pump its
        event loop (so clicks/keys are processed even while physics is
        paused during review), and report whether it's still open (always
        True if there's no window, e.g. --no-view with --record-video)."""
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
        # No hardware button to trigger start -- record the whole dry run,
        # then auto-keep at the end (or on --auto-finish success) since
        # there's nobody present to press KEEP/DELETE.
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
                    # Nobody to press KEEP/DELETE in a scripted run.
                    resolve_recorded_episode(True)
                    continue
                state = device.get_state()
                if state["short_press_count"] != last_short:
                    last_short = state["short_press_count"]
                    plot_action[0] = "keep"
                if state["long_press_count"] != last_long:
                    last_long = state["long_press_count"]
                    plot_action[0] = "delete"
                poll_ui_action()  # apply a device press immediately rather
                # than waiting for the next loop's top-of-loop poll
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
            # Map sim-frame force back onto all three device axes (pos_map's
            # transpose) so felt resistance opposes the handle motion that
            # caused it; the unmapped device axis naturally gets 0 force.
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
                        timestamp_ms=collection["episode_tick"]
                        * (1000.0 / args.dataset_hz),
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
                x, y, theta = env.t_pose
                state_label = f" [{collection['state']}]" if recorder is not None else ""
                print(
                    f"t={t_elapsed:6.1f}s{state_label}  "
                    f"coverage={env.coverage_fraction():.3f}  "
                    f"T=({x:+.3f},{y:+.3f},{np.degrees(theta):+6.1f}deg)  "
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
            # build_canvas() produces BGR (for cv2.imshow); imageio wants RGB.
            rgb_frames = [frame[:, :, ::-1] for frame in video_frames]
            imageio.mimwrite(str(out_path), rgb_frames, fps=args.video_fps,
                              quality=7, macro_block_size=None)
            print(f"[video] wrote {len(video_frames)} frames -> {out_path}")


if __name__ == "__main__":
    main()
