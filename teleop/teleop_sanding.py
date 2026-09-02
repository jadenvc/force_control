"""Haptic teleoperation for the full-arm sanding task.

Moves a Force Dimension omega handle to drive a UR5e holding a compliant
sander pad. The operator feels the pad/panel contact force reflected at the
handle, watches the panel's color gradient update live as regions get
under-sanded (blue) / just-right (green) / over-sanded (red), and sees a
live cv2 HUD -- a force-over-time strip chart with reference lines, current
coverage/force/max-episode-force, and a SUCCESS/BROKEN flag -- the same
kind of interface teleop_flipup.py uses, ported here at a scope that matches
this task (one force signal, not FlipUp's raw/sensor/xyz breakdown). Episode
saving/deleting works both from the handle button and from the keyboard
(s/k/d/r/q), and each new episode's start position is randomized within a
small prism above the panel rather than always the exact same pose.
--dry-run runs a scripted demo motion with no hardware attached, for testing.

See sanding_teleop.py for the physics (dose accumulation, break condition,
success metric) and sanding_recorder.py for BC dataset recording.
"""

from __future__ import annotations

import argparse
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np

from sanding_teleop import (
    CONTACT_TOOL_Z,
    DEFAULT_SANDING_PROPERTIES,
    DEVICE_WORKSPACE_HALF_M,
    PANEL_TRANSFORM,
    SandingProperties,
    SandingTeleop,
)

# Map the device's full comfortable workspace (DEVICE_WORKSPACE_HALF_M) onto
# the full panel (half-length/half-width), so the operator can reach every
# corner of the sanding area without needing to leave the device's safe
# range. z keeps a much smaller scale on purpose: force control needs fine
# resolution in the pressing direction, and a large z-scale would make a
# tiny, unintentional hand tremor translate into a large, twitchy force swing.
DEFAULT_SCALE = (
    (DEFAULT_SANDING_PROPERTIES.panel_length_m / 2.0) / DEVICE_WORKSPACE_HALF_M[0],
    (DEFAULT_SANDING_PROPERTIES.panel_width_m / 2.0) / DEVICE_WORKSPACE_HALF_M[1],
    1.0,
)

# Force Dimension's device frame is +x toward the operator, +y to the
# operator's right, +z up. Same signed-permutation convention as
# teleop_flipup.py's --axes / teleop_push_t.py's --axes (duplicated rather
# than imported so this script stays standalone).
#
# Re-derived for the side-on camera (azimuth=90, elevation=-15): screen
# left/right is world x, screen up/down is world z, and screen depth
# (near/far from the camera) is world y. So:
#   device right (+y)      -> tool moves screen-right    (sim +x)
#   device away from you (-x) -> tool moves away into the screen (sim +y)
#   device up (+z)          -> tool moves up on screen     (sim +z, unchanged)
DEFAULT_AXES = "y,-x,z"

WINDOW_NAME = "Force Dimension -- Sanding"
AXIS_ORANGE = (60, 150, 230)  # BGR


def build_pos_map(spec):
    idx = {"x": 0, "y": 1, "z": 2}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--axes needs 3 entries, got {spec!r}")
    m = np.zeros((3, 3))
    for row, part in enumerate(parts):
        sign = -1.0 if part.startswith("-") else 1.0
        name = part.lstrip("+-").lower()
        if name not in idx:
            raise ValueError(f"--axes entry {part!r} must name x, y or z")
        m[row, idx[name]] = sign
    if abs(abs(np.linalg.det(m)) - 1.0) > 1e-9:
        raise ValueError(f"--axes {spec!r} reuses an axis; each of x,y,z must appear once")
    return m


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Haptic teleoperation for the full-arm sanding task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- task physics -------------------------------------------------------
    parser.add_argument("--sand-force-min", type=float, default=DEFAULT_SANDING_PROPERTIES.force_min_n,
                        help="below this normal force (N), no material removal at all")
    parser.add_argument("--sand-force-target", type=float, default=DEFAULT_SANDING_PROPERTIES.force_target_n,
                        help="desired/nominal sanding force (N)")
    parser.add_argument("--sand-force-cap", type=float, default=DEFAULT_SANDING_PROPERTIES.force_cap_n,
                        help="dose-rate saturates at/above this force (N)")
    parser.add_argument("--sand-break-force", type=float, default=DEFAULT_SANDING_PROPERTIES.force_break_n,
                        help="sustained (filtered) force above this breaks the panel")
    parser.add_argument("--dose-target-time", type=float, default=DEFAULT_SANDING_PROPERTIES.dose_target_time_s,
                        help="seconds at --sand-force-target to reach dose 1.0")
    parser.add_argument("--dose-low", type=float, default=DEFAULT_SANDING_PROPERTIES.dose_low,
                        help="dose below this is under-sanded")
    parser.add_argument("--dose-high", type=float, default=DEFAULT_SANDING_PROPERTIES.dose_high,
                        help="dose above this is over-sanded")
    parser.add_argument("--dose-max", type=float, default=DEFAULT_SANDING_PROPERTIES.dose_max,
                        help="dose accumulator clip ceiling")
    parser.add_argument("--pad-softness", type=float, default=DEFAULT_SANDING_PROPERTIES.pad_softness,
                        help="[0,1] sander pad contact compliance -- higher = softer/more "
                             "gradual contact onset, 1.0 is the softest available. See "
                             "SandingEnv._configure_pad_contact")
    parser.add_argument("--friction", type=float, nargs=3, default=DEFAULT_SANDING_PROPERTIES.friction,
                        metavar=("SLIDING", "TORSIONAL", "ROLLING"),
                        help="pad-vs-panel Coulomb friction. Lower sliding friction = less "
                             "resistance/drag as the pad moves across the panel. Written onto "
                             "the panel geom, not the pad -- MuJoCo resolves contact friction "
                             "from whichever geom has higher `priority`, and the panel wins")
    parser.add_argument("--grid-resolution", type=float, default=DEFAULT_SANDING_PROPERTIES.grid_resolution_m,
                        help="dose-accumulation grid cell size (m)")
    parser.add_argument("--vis-cell", type=float, default=DEFAULT_SANDING_PROPERTIES.vis_cell_m,
                        help="visual gradient grid cell size (m), coarser than --grid-resolution")
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_SANDING_PROPERTIES.success_threshold,
                        help="fraction of TARGET REGION area (not the whole panel) that "
                             "must be in the just-right dose band for success")
    parser.add_argument("--num-regions", type=int, default=DEFAULT_SANDING_PROPERTIES.num_regions,
                        help="pin an exact number of discrete square regions -- "
                             "highlighted amber until touched; the rest of the panel "
                             "doesn't count toward coverage/success. Leave unset "
                             "(default) to randomize the count every episode within "
                             "[--num-regions-min, --num-regions-max]; the line's "
                             "start position along the panel randomizes either way")
    parser.add_argument("--num-regions-min", type=int, default=DEFAULT_SANDING_PROPERTIES.num_regions_min,
                        help="randomized region-count lower bound, used unless "
                             "--num-regions pins an exact count")
    parser.add_argument("--num-regions-max", type=int, default=DEFAULT_SANDING_PROPERTIES.num_regions_max,
                        help="randomized region-count upper bound, used unless "
                             "--num-regions pins an exact count")
    parser.add_argument("--region-radius", type=float, default=DEFAULT_SANDING_PROPERTIES.region_radius_m,
                        help="radius (m) of each target region")
    parser.add_argument("--seed", type=int, default=0,
                        help="also determines the target regions' layout on the panel")

    # ---- controller -----------------------------------------------------------
    parser.add_argument("--tool-kp", type=float, default=SandingTeleop.default_tool_kp,
                        help="Cartesian task-space stiffness (N/m) driving the arm "
                             "toward the operator's target")
    parser.add_argument("--arm-damping", type=float, default=2.5,
                        help="multiplier on the default joint damping")
    parser.add_argument("--max-speed", type=float, default=0.30,
                        help="cap on how fast the commanded target may travel (m/s), "
                             "0 = uncapped. Matches FlipUp's full-arm default -- --scale "
                             "multiplies hand velocity too (especially in x/y here, "
                             "which are scaled ~4x to reach the whole panel), so a much "
                             "lower cap throttles ordinary hand motion into sluggishness")

    # ---- haptics ----------------------------------------------------------------
    parser.add_argument("--stiffness", type=float, default=SandingTeleop.default_haptic_stiffness,
                        help="target stiffness AT THE HANDLE (N/m); force-gain is "
                             "derived as stiffness/(tool_kp*scale) unless --force-gain is given")
    parser.add_argument("--force-gain", type=float, default=None,
                        help="N of handle force per N of simulated contact force")
    parser.add_argument("--force-clip", type=float, default=90.0,
                        help="ceiling on the reflected sim force (N) before force-gain. "
                             "Kept at 2x --sand-break-force so the operator still feels "
                             "the full ramp-up as force approaches breaking, not a "
                             "clipped/truncated version of it")
    parser.add_argument("--max-force", type=float, default=10.0,
                        help="clamp on the handle force vector magnitude (N)")
    parser.add_argument("--force-tau", type=float, default=2.0,
                        help="handle-force smoothing time constant (ms), 0 = raw")
    parser.add_argument("--force-rate", type=float, default=80.0,
                        help="cap on how fast handle force may change (N/s), 0 = uncapped")
    parser.add_argument("--damping", type=float, default=60.0,
                        help="handle velocity damping (N/(m/s)). Raised from FlipUp's "
                             "30 default -- the bigger x/y reach scale here means "
                             "faster free-space hand swings, and more damping keeps "
                             "those calm instead of overshooting/oscillating")
    parser.add_argument("--scale", type=float, nargs=3, default=DEFAULT_SCALE,
                        metavar=("SX", "SY", "SZ"),
                        help="sim metres per device metre, per axis. Default maps the "
                             "device's full comfortable range onto the full panel area")
    parser.add_argument("--home", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"),
                        help="physical handle position (device m) mapped to the panel-hover target")
    parser.add_argument("--axes", type=str, default=DEFAULT_AXES,
                        help="which device axis (optionally negated) drives sim x, y, z")
    parser.add_argument("--auto-init", action="store_true",
                        help="auto-calibrate the omega on open (it will move)")
    parser.add_argument("--arm-tolerance", type=float, default=0.02,
                        help="handle FORCE FEEDBACK stays silent (zero) until the sim "
                             "tool has tracked to within this many metres of the "
                             "commanded target -- same idea and default as "
                             "teleop_flipup.py's hardcoded 0.02 armed-check. Motion is "
                             "NOT gated on this (the --max-speed slew limiter already "
                             "prevents an instant jump; a device home mismatch just "
                             "becomes a gradual, visible catch-up instead of a snap)")
    parser.add_argument("--takeover-ramp-ms", type=float, default=400.0,
                        help="milliseconds right after arming (see --arm-tolerance) "
                             "during which reflected handle force ramps 0 -> full "
                             "instead of snapping on")

    # ---- reset position -----------------------------------------------------------
    parser.add_argument(
        "--episode-randomization", action=argparse.BooleanOptionalAction, default=True,
        help="vary the start hover position within --start-prism each new episode "
             "(default: enabled). Disable for a fixed, identical start every time",
    )
    parser.add_argument("--start-prism", type=float, nargs=3, default=(0.06, 0.06, 0.03),
                        metavar=("X", "Y", "Z"),
                        help="full size (m) of the box the start hover position is "
                             "sampled from, centered on the nominal panel-center hover")
    parser.add_argument("--start-center-prob", type=float, default=0.5,
                        help="probability a new episode starts at the exact nominal "
                             "hover instead of a random point in --start-prism")

    # ---- dataset collection -----------------------------------------------------
    parser.add_argument("--collect-dataset", type=str, default=None,
                        help="zarr path to record BC episodes to; omit to disable recording")
    parser.add_argument("--dataset-hz", type=float, default=1000.0)
    parser.add_argument("--dataset-image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--dataset-no-rgb", action="store_true")
    parser.add_argument("--dataset-min-samples", type=int, default=20)
    parser.add_argument("--auto-finish", action="store_true",
                        help="auto-stop a recording episode the instant success() or "
                             "broken becomes true")

    # ---- loop / view --------------------------------------------------------------
    parser.add_argument("--control-freq", type=int, default=1000)
    parser.add_argument("--view-fps", type=float, default=30.0)
    parser.add_argument("--render-width", type=int, default=520)
    parser.add_argument("--render-height", type=int, default=390)
    parser.add_argument("--no-wrist-cam", action="store_true",
                        help="hide the wrist-camera picture-in-picture inset")
    parser.add_argument("--wrist-cam-width", type=int, default=160)
    parser.add_argument("--wrist-cam-height", type=int, default=120)
    parser.add_argument("--viewer-scale", type=float, default=1.5,
                        help="resize factor for the displayed cv2 window (rendering "
                             "itself always happens at --render-width/height)")
    parser.add_argument("--plot-span", type=float, default=6.0,
                        help="seconds of force history shown in the strip chart")
    parser.add_argument("--plot-fixed-scale", action="store_true",
                        help="pin the force plot to [0, --sand-break-force] instead of "
                             "autoscaling to what is actually happening")
    parser.add_argument("--no-plot", action="store_true",
                        help="hide the live force strip chart (HUD text stays on)")
    parser.add_argument("--dry-run", action="store_true",
                        help="run a scripted demo motion with no Force Dimension hardware")
    parser.add_argument("--dry-run-seconds", type=float, default=20.0)
    parser.add_argument("--no-view", action="store_true",
                        help="disable the cv2 HUD window (rendering still runs if "
                             "recording RGB frames)")
    parser.add_argument("--no-readout", action="store_true", help="disable the console readout")
    parser.add_argument("--print-interval-s", type=float, default=0.5)
    parser.add_argument("--cam-azimuth", type=float, default=None)
    parser.add_argument("--cam-elevation", type=float, default=None)
    parser.add_argument("--cam-distance", type=float, default=None)
    parser.add_argument("--cam-lookat", type=float, nargs=3, default=None,
                        help="camera lookat point (m); default frames the whole arm + panel")

    return parser


def _derive_force_gain(args):
    """N of handle force per N of simulated contact force.

    FlipUp/push_t derive this from scale[0] alone, which is fine there
    because their --scale is always isotropic (same number on every axis).
    Sanding's --scale is deliberately anisotropic (large in x/y to reach the
    whole panel, small in z for fine press control), so scale[0] is the
    WRONG axis here -- it's the reach axis, not the axis that actually
    determines how a given contact force maps to hand displacement.
    """
    if args.force_gain is not None:
        return float(args.force_gain)
    return float(args.stiffness / max(args.tool_kp * args.scale[2], 1e-9))


def _scripted_dry_run_target(t_s, home_xy, contact_z, radius=0.05):
    """A slow lawnmower-ish sweep over the panel, dipping in and out of
    contact, for hardware-free testing. contact_z is the tool height at
    which the pad JUST touches the panel (0 penetration)."""
    x = home_xy[0] + radius * np.sin(0.3 * t_s)
    y = home_xy[1] + radius * np.sin(0.11 * t_s)
    # Cycles from 3mm clear of the panel down to ~4mm of penetration
    # (~25N at the default pad_softness=1.0's much-softened contact
    # stiffness -- NOT tool_kp*penetration, that assumes rigid contact;
    # see SOFT_PAD_SOLREF's comment), covering both free-space and
    # in-contact behavior across the actual force band in one scripted run.
    z = contact_z + 0.003 - 0.007 * (0.5 + 0.5 * np.sin(0.7 * t_s))
    return np.array([x, y, z])


def sample_start_offset(rng, prism_size, center_probability, force_center):
    """A small xyz offset around the nominal hover pose, or exactly zero.

    Mirrors teleop_flipup.py's sample_start_pose center-bias idea (a mix of
    "exactly nominal" and "uniform in a small box") but geometrically
    clamped rather than settle-and-reject: the sanding reset target is a
    fixed hover height above a static panel, not a pose relative to a
    randomized book/bookend layout, so there's no equivalent settle risk to
    check by simulating -- clamping the vertical offset to stay above a
    safe clearance is sufficient (done by the caller).
    """
    if force_center or rng.random() < center_probability:
        return np.zeros(3)
    half = np.asarray(prism_size, dtype=float) / 2.0
    return rng.uniform(-half, half)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.control_freq <= 0:
        parser.error("--control-freq must be positive")
    if args.tool_kp <= 0.0:
        parser.error("--tool-kp must be positive")
    pos_map = build_pos_map(args.axes)

    properties = SandingProperties(
        force_min_n=args.sand_force_min,
        force_target_n=args.sand_force_target,
        force_cap_n=args.sand_force_cap,
        force_break_n=args.sand_break_force,
        dose_target_time_s=args.dose_target_time,
        dose_low=args.dose_low,
        dose_high=args.dose_high,
        dose_max=args.dose_max,
        pad_softness=args.pad_softness,
        friction=tuple(args.friction),
        grid_resolution_m=args.grid_resolution,
        vis_cell_m=args.vis_cell,
        success_threshold=args.success_threshold,
        num_regions=args.num_regions,
        num_regions_min=args.num_regions_min,
        num_regions_max=args.num_regions_max,
        region_radius_m=args.region_radius,
    )
    env = SandingTeleop(
        seed=args.seed,
        properties=properties,
        tool_kp=args.tool_kp,
        arm_damping=args.arm_damping,
    )

    force_gain = _derive_force_gain(args)
    print(
        f"sanding ready: force band [{properties.force_min_n:.1f}, "
        f"{properties.force_target_n:.1f}, {properties.force_cap_n:.1f}, "
        f"BREAK={properties.force_break_n:.1f}] N, tool_kp={env.tool_kp:.0f} N/m, "
        f"force_gain={force_gain:.4f} N/N"
    )
    reach_xy = np.asarray(args.scale[:2]) * DEVICE_WORKSPACE_HALF_M[:2]
    print(
        f"[workspace] scale {tuple(round(s, 2) for s in args.scale)}: device's "
        f"comfortable range reaches +/-{reach_xy[0]*100:.1f}cm x, "
        f"+/-{reach_xy[1]*100:.1f}cm y around panel center (panel is "
        f"{properties.panel_length_m*100:.0f}x{properties.panel_width_m*100:.0f}cm)"
    )
    ctl_dt = 1.0 / args.control_freq
    tau = args.force_tau / 1000.0
    t_eff = ctl_dt + 2.0 * tau
    k_limit = 2.0 * args.damping / t_eff if t_eff > 0 else float("inf")
    k_handle = env.tool_kp * args.scale[2] * force_gain
    print(
        f"[haptics] handle stiffness <= {k_handle:.0f} N/m, passivity limit "
        f"~{k_limit:.0f} N/m"
        + ("" if k_handle < k_limit else "  <-- WARNING: over the limit, expect buzz")
    )

    recorder = None
    collection = {"state": "disabled", "started_monotonic": None, "reason": None,
                  "success": False, "broken": False, "final_coverage": 0.0}
    if args.collect_dataset:
        from sanding_recorder import SandingEpisodeRecorder

        recorder = SandingEpisodeRecorder(
            args.collect_dataset,
            sample_hz=args.dataset_hz,
            image_size=tuple(args.dataset_image_size),
            include_rgb=not args.dataset_no_rgb,
            min_samples=args.dataset_min_samples,
        )
        collection["state"] = "idle"
        print(f"[dataset] recording to {args.collect_dataset}")
        print(
            "[dataset] press S (or short-press the handle) to start/stop an episode, "
            "K/D (or click) to keep/delete, R to reset -- --dry-run starts one automatically"
        )

    # ---------------------------------------------------------------- render
    render_model_lock = threading.Lock()
    W, H = args.render_width, args.render_height
    need_rgb_for_dataset = recorder is not None and not args.dataset_no_rgb
    rendering = (not args.no_view) or need_rgb_for_dataset
    shot = {"frame": None, "wrist_frame": None, "sim_time_s": None, "n": 0, "run": True, "err": None}
    render_thread = None
    show_wrist_cam = rendering and not args.no_wrist_cam
    if rendering:
        lookat = args.cam_lookat if args.cam_lookat is not None else env.default_cam_lookat
        azimuth = args.cam_azimuth if args.cam_azimuth is not None else env.default_cam_azimuth
        elevation = args.cam_elevation if args.cam_elevation is not None else env.default_cam_elevation
        distance = args.cam_distance if args.cam_distance is not None else env.default_cam_distance
        view_period = 1.0 / args.view_fps if args.view_fps > 0 else 0.0

        def render_loop():
            # MovableCamera's constructor touches the GL context (to
            # validate framebuffer size), which makes it current on
            # whichever thread constructs it. Building it here, on the
            # render thread itself, rather than on the main thread before
            # this thread starts, is what keeps the context affinity
            # consistent -- constructing it on the main thread and then
            # calling .render() from here raised "Cannot make context
            # current on thread X: already current on thread Y" on every
            # single frame, which is why the cv2 window opened but never
            # updated (render_loop died on its first iteration, silently,
            # until the error-surfacing added alongside this fix).
            from dm_control.mujoco.engine import Camera, MovableCamera

            try:
                camera = MovableCamera(env.physics, height=H, width=W)
                camera.set_pose(lookat=lookat, distance=distance, azimuth=azimuth, elevation=elevation)
                wrist_camera = None
                if show_wrist_cam:
                    wrist_cam_id = env.model.camera("ur5e/sander/wrist_cam").id
                    wrist_camera = Camera(
                        env.physics, height=args.wrist_cam_height, width=args.wrist_cam_width,
                        camera_id=wrist_cam_id,
                    )
                while shot["run"]:
                    t_frame = time.time()
                    with render_model_lock:
                        shot["frame"] = camera.render()
                        if wrist_camera is not None:
                            shot["wrist_frame"] = wrist_camera.render()
                        shot["sim_time_s"] = float(env.data.time)
                    shot["n"] += 1
                    idle = view_period - (time.time() - t_frame)
                    if idle > 0:
                        time.sleep(idle)
            except Exception:  # keep the haptic loop alive if rendering dies
                shot["err"] = traceback.format_exc()

        render_thread = threading.Thread(target=render_loop, daemon=True)
        render_thread.start()

    # ----------------------------------------------------------------- plot
    plot_lock = threading.Lock()
    trace_force = deque(maxlen=W)
    plot_every = max(1, int(round(args.control_freq * args.plot_span / W)))
    next_plot_step = [plot_every]

    def draw_plot(frame):
        if args.no_plot or not trace_force:
            return frame
        h, w = frame.shape[:2]
        plot_h = 108
        top = h - plot_h
        frame[top:, :] = (frame[top:, :].astype(np.float32) * 0.25).astype(np.uint8)
        p = properties
        fmax = p.force_break_n * 1.15 if args.plot_fixed_scale else max(
            5.0, 1.25 * max(trace_force, default=0.0), p.force_break_n * 1.05
        )
        graph_top = top + 6
        graph_bottom = h - 4

        def y_of(force_value):
            return int(graph_bottom - np.clip(force_value / fmax, 0.0, 1.0) * (graph_bottom - graph_top))

        for value, colour, label in (
            (p.force_min_n, (140, 140, 140), "min"),
            (p.force_target_n, (90, 220, 120), "target"),
            (p.force_cap_n, AXIS_ORANGE, "cap"),
            (p.force_break_n, (70, 70, 245), "BREAK"),
        ):
            y = y_of(value)
            cv2.line(frame, (0, y), (w, y), colour, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{label} {value:.0f}N", (4, max(y - 3, graph_top + 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)

        n = len(trace_force)
        xs = np.arange(w - n, w)
        ys = np.array([y_of(v) for v in trace_force], dtype=np.int32)
        cv2.polylines(frame, [np.stack([xs, ys], 1).astype(np.int32)], False,
                      (110, 220, 235), 2, cv2.LINE_AA)
        now = trace_force[-1] if trace_force else 0.0
        cv2.putText(
            frame,
            f"FORCE {now:5.1f}N  (max this episode {env.episode_max_force_n:5.1f}N)  "
            f"{args.plot_span:.0f}s window",
            (6, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
        )
        return frame

    button_y0, button_y1 = (78, 108)
    button_gap = 8
    button_w = max(82, min(112, (W - 3 * button_gap) // 2))
    keep_rect = (button_gap, button_y0, button_gap + button_w, button_y1)
    delete_rect = (2 * button_gap + button_w, button_y0, 2 * (button_gap + button_w), button_y1)

    def draw_state(frame):
        cov = env.coverage_fraction("just_right")
        done = env.success()
        state_text = (
            f"sanded {100.0 * cov:5.1f}% of {env.num_regions} regions  "
            f"(need >= {100.0 * properties.success_threshold:.0f}%)"
        )
        cv2.putText(frame, state_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (120, 230, 140) if done else (230, 230, 230), 1, cv2.LINE_AA)
        if env.broken:
            cv2.putText(frame, "BROKEN", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (70, 70, 245), 2, cv2.LINE_AA)
        elif done:
            cv2.putText(frame, "SUCCESS", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (120, 230, 140), 2, cv2.LINE_AA)
        if recorder is not None:
            status = collection["state"]
            if status == "idle":
                text, colour = "IDLE -- press S to start episode", (80, 220, 240)
            elif status == "recording":
                text = f"REC {recorder.sample_count / args.dataset_hz:5.1f}s -- S to stop"
                colour = (80, 80, 245)
                cv2.circle(frame, (W - 17, 17), 7, colour, -1, cv2.LINE_AA)
            else:
                text, colour = "EPISODE FINISHED -- choose KEEP or DELETE", (80, 220, 240)
            cv2.putText(frame, text, (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour, 1, cv2.LINE_AA)
            if status == "review":
                x0, y0, x1, y1 = keep_rect
                cv2.rectangle(frame, (x0, y0), (x1, y1), (65, 150, 65), -1)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 240, 120), 2)
                cv2.putText(frame, "KEEP [K]", (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.44, (245, 245, 245), 1, cv2.LINE_AA)
                x0, y0, x1, y1 = delete_rect
                cv2.rectangle(frame, (x0, y0), (x1, y1), (65, 65, 165), -1)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (100, 100, 245), 2)
                cv2.putText(frame, "DELETE [D]", (x0 + 5, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.40, (245, 245, 245), 1, cv2.LINE_AA)
        return frame

    # ---------------------------------------------------------------- device
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
        print(
            f"[device] the arm starts tracking the handle immediately, relative to "
            f"--home {tuple(args.home)} (device m). If you saw 'drdMoveToPos failed', "
            f"the device did NOT auto-center there -- the arm will still move right "
            f"away, just from a possibly-offset starting point; --max-speed limits how "
            f"fast it can close that gap, so it'll be a gradual catch-up, not a jump. "
            f"Haptic force feedback stays silent until the tool has actually caught up "
            f"(--arm-tolerance)."
        )

    # 2cm clear of first touch, centered on the panel (not hardcoded -- this
    # was a literal 0.55 that silently went stale the first time the panel
    # moved; PANEL_TRANSFORM is now the one source of truth for its position).
    nominal_hover = np.array([PANEL_TRANSFORM[0, 3], PANEL_TRANSFORM[1, 3], CONTACT_TOOL_Z + 0.02])
    min_clearance_z = CONTACT_TOOL_Z + 0.01  # never sample a start pose closer than this
    rng = np.random.default_rng(args.seed)
    episode_attempt = [0]

    def sample_reset_target(attempt_index):
        if not args.episode_randomization:
            return nominal_hover.copy()
        offset = sample_start_offset(
            rng, args.start_prism, args.start_center_prob, force_center=(attempt_index == 0)
        )
        candidate = nominal_hover + offset
        candidate[2] = max(candidate[2], min_clearance_z)
        return candidate

    # Start the slew-limited target at wherever the arm actually is, not at
    # the reset target -- otherwise the very first tick commands a giant
    # instantaneous jump (home pose -> panel), producing exactly the
    # impact-transient false break this task's own dose/break design is
    # supposed to guard against.
    target = env.tool_pos.copy()
    reset_target = sample_reset_target(0)
    scale = np.array(args.scale, dtype=float)
    dt = 1.0 / float(args.control_freq)
    max_step = args.max_speed * dt if args.max_speed > 0.0 else float("inf")

    def do_reset(advance_episode=False):
        nonlocal target, reset_target, device_armed
        with render_model_lock:
            env.reset()
        if advance_episode:
            episode_attempt[0] += 1
        reset_target = sample_reset_target(episode_attempt[0])
        target = env.tool_pos.copy()
        # Same as teleop_flipup.py's do_reset: force feedback goes silent
        # again until the tool re-converges. dry-run has no physical device
        # to protect, so it stays armed (matches the initial-arming default
        # above) -- immediately true again anyway since target==tool_pos.
        device_armed = bool(args.dry_run)

    def start_episode():
        if collection["state"] != "idle":
            return
        recorder.start_episode(metadata={"command_line": vars(args)})
        collection["state"] = "recording"
        collection["started_monotonic"] = time.monotonic()
        print("\n[dataset] recording started")

    def stop_episode(reason):
        if collection["state"] != "recording":
            return
        collection.update(
            reason=reason, success=bool(env.success()), broken=bool(env.broken),
            final_coverage=env.coverage_fraction("just_right"),
        )
        collection["state"] = "review"
        print(
            f"\n[dataset] episode stopped ({reason}): success={collection['success']}, "
            f"broken={collection['broken']}, coverage={100*collection['final_coverage']:.1f}%, "
            f"max force {env.episode_max_force_n:.2f}N -- press K to keep or D to delete"
        )

    def resolve_episode(keep):
        if collection["state"] != "review":
            return
        if keep:
            name = recorder.commit(
                success=collection["success"],
                broken=collection["broken"],
                termination_reason=collection.get("reason") or "unknown",
                final_coverage_fraction=collection["final_coverage"],
            )
            print(f"[dataset] kept as {name}" if name else "[dataset] discarded (too few samples)")
        else:
            recorder.discard()
            print("[dataset] discarded")
        collection["state"] = "idle"
        do_reset(advance_episode=True)

    # ----------------------------------------------------------------- HUD
    viewer_keys = deque()
    viewer_key_lock = threading.Lock()
    viewer_state = {"run": True, "err": None}
    review_action = [None]
    shown = [0, -1]

    def on_mouse(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONUP or collection["state"] != "review":
            return
        x, y = int(x / args.viewer_scale), int(y / args.viewer_scale)
        kx0, ky0, kx1, ky1 = keep_rect
        dx0, dy0, dx1, dy1 = delete_rect
        if kx0 <= x <= kx1 and ky0 <= y <= ky1:
            review_action[0] = "keep"
        elif dx0 <= x <= dx1 and dy0 <= y <= dy1:
            review_action[0] = "delete"

    def draw_wrist_inset(frame):
        wrist_img = shot["wrist_frame"]
        if not show_wrist_cam or wrist_img is None:
            return frame
        inset = np.ascontiguousarray(wrist_img[:, :, ::-1])  # RGB -> BGR
        ih, iw = inset.shape[:2]
        margin = 6
        x1 = frame.shape[1] - margin
        x0 = x1 - iw
        y0 = margin
        y1 = y0 + ih
        if x0 < 0 or y1 > frame.shape[0]:
            return frame
        cv2.rectangle(frame, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (200, 200, 200), 1)
        frame[y0:y1, x0:x1] = inset
        cv2.putText(frame, "wrist", (x0 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (230, 230, 230), 1, cv2.LINE_AA)
        return frame

    def show():
        img = shot["frame"]
        if img is None:
            return 255
        frame = np.ascontiguousarray(img[:, :, ::-1])  # RGB -> BGR
        frame = draw_state(frame)
        with plot_lock:
            frame = draw_plot(frame)
        frame = draw_wrist_inset(frame)
        shown[0] += 1
        shown[1] = shot["n"]
        if args.no_view:
            return 255
        display = frame if np.isclose(args.viewer_scale, 1.0) else cv2.resize(
            frame, None, fx=args.viewer_scale, fy=args.viewer_scale, interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow(WINDOW_NAME, display)
        return cv2.waitKey(1) & 0xFF

    def pop_viewer_key():
        with viewer_key_lock:
            return viewer_keys.popleft() if viewer_keys else 255

    viewer_thread = None
    if rendering:
        display_period = 1.0 / max(args.view_fps, 1e-6)

        def viewer_loop():
            window_created = False
            try:
                if not args.no_view:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
                    window_created = True
                    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
                while viewer_state["run"]:
                    if shot["frame"] is not None and shot["n"] != shown[1]:
                        key = show()
                        if key != 255:
                            with viewer_key_lock:
                                viewer_keys.append(key)
                    else:
                        time.sleep(min(0.01, display_period / 4.0))
            except Exception:
                viewer_state["err"] = traceback.format_exc()
            finally:
                if window_created:
                    try:
                        cv2.destroyWindow(WINDOW_NAME)
                    except Exception:
                        pass

        viewer_thread = threading.Thread(target=viewer_loop, daemon=True)
        viewer_thread.start()

    t_start = time.monotonic()
    last_print = t_start
    step_index = 0
    last_short_press_count = 0
    # Matches teleop_flipup.py's own "armed" pattern exactly: motion is
    # NEVER gated on this (the --max-speed slew limiter is what prevents an
    # instant jump if the device's actual rest position doesn't match
    # --home -- e.g. after a failed auto-home -- turning that mismatch into
    # a gradual, visible catch-up instead of a snap). Only haptic force
    # reflection stays silent until the SIM tool has tracked to within
    # --arm-tolerance of wherever it's currently being commanded. This is
    # deliberately NOT a wait for the physical device to reach any specific
    # position -- an earlier version of this gate did that and took 8+
    # seconds every run for hardware that never successfully auto-homes;
    # this should arm in a similar ballpark to FlipUp's, normally well
    # under a second.
    device_armed = bool(args.dry_run)
    home_xyz = np.array(args.home, dtype=float)
    armed_at = None if not args.dry_run else t_start

    if recorder is not None and args.dry_run:
        # No device button to press in --dry-run, so start immediately --
        # this is what makes `--dry-run --collect-dataset ...` a useful
        # hardware-free smoke test of the recording path.
        start_episode()

    reported_render_err = False
    reported_viewer_err = False

    try:
        while True:
            now = time.monotonic()
            t_elapsed = now - t_start
            if args.dry_run and t_elapsed >= args.dry_run_seconds:
                break

            if rendering and not reported_render_err and shot["err"] is not None:
                reported_render_err = True
                print(f"\n[render] thread died, HUD/dataset RGB frames will stop:\n{shot['err']}")
            if rendering and not reported_viewer_err and viewer_state["err"] is not None:
                reported_viewer_err = True
                print(f"\n[viewer] thread died, window will stop updating:\n{viewer_state['err']}")

            if collection["state"] == "review":
                key = pop_viewer_key()
                action = review_action[0]
                review_action[0] = None
                if key in (ord("k"), 13) or action == "keep":
                    resolve_episode(keep=True)
                elif key in (ord("d"), ord("x"), 8, 127) or action == "delete":
                    resolve_episode(keep=False)
                elif key in (ord("q"), 27):
                    print("\n[dataset] choose KEEP or DELETE before quitting")
                time.sleep(0.005)
                continue

            key = pop_viewer_key()
            if key == ord("s") and recorder is not None:
                if collection["state"] == "idle":
                    start_episode()
                elif collection["state"] == "recording":
                    stop_episode("operator_stop")
            elif key == ord("r"):
                if collection["state"] == "recording":
                    print("\n[dataset] press S to stop, then KEEP or DELETE")
                else:
                    do_reset(advance_episode=True)
            elif key in (ord("q"), 27) and recorder is None:
                break

            if args.dry_run:
                desired = _scripted_dry_run_target(t_elapsed, reset_target[:2], CONTACT_TOOL_Z)
                device_state = {"pos": np.zeros(3), "vel": np.zeros(3)}
            else:
                state = device.get_state()
                device_state = state
                device_xyz = state["pos"][:3]
                # Always computed directly from the device -- NOT frozen
                # while unarmed. If the device's actual rest position
                # doesn't match --home (e.g. a failed auto-home), this is
                # just a larger-than-usual initial offset; --max-speed below
                # already slew-limits how fast target can approach it, so
                # it becomes a gradual, visible catch-up, not an instant jump.
                desired = reset_target + pos_map @ (scale * (device_xyz - home_xyz))
                if state["long_press_count"] > 0:
                    if collection["state"] == "recording":
                        stop_episode("operator_reset")
                    do_reset(advance_episode=True)
                # Single button click starts/stops an episode; there is no
                # gripper on this end effector, so the short-press signal
                # (normally used to toggle a gripper) is repurposed here.
                if recorder is not None and state["short_press_count"] != last_short_press_count:
                    last_short_press_count = state["short_press_count"]
                    if collection["state"] == "idle":
                        start_episode()
                    elif collection["state"] == "recording":
                        stop_episode("operator_stop")

            delta = desired - target
            step_norm = np.linalg.norm(delta)
            if step_norm > max_step:
                delta *= max_step / step_norm
            target = target + delta

            with render_model_lock:
                env.step(target)
            step_index += 1

            # Matches teleop_flipup.py's "armed" check exactly: the sim tool
            # tracking to within --arm-tolerance of its own commanded target
            # (not any physical-device-position requirement) is what allows
            # force feedback to turn on. Reset convergence isn't interaction
            # force -- while catching up to a fresh/mismatched target,
            # reflect silence instead of whatever transient contact force
            # that catch-up motion happens to produce.
            if not device_armed:
                if np.linalg.norm(env.tool_pos - target) < args.arm_tolerance:
                    device_armed = True
                    armed_at = now
                    print("\n[device] tool reached target -- haptic feedback engaged")

            force_ramp = 0.0
            if device_armed:
                since_armed_ms = (now - armed_at) * 1000.0
                force_ramp = (
                    np.clip(since_armed_ms / max(args.takeover_ramp_ms, 1e-9), 0.0, 1.0)
                    if args.takeover_ramp_ms > 0.0
                    else 1.0
                )

            force, _ = env.pad_contact_force()
            reflected = force_ramp * np.clip(
                force * force_gain, -args.force_clip * force_gain, args.force_clip * force_gain
            )
            if device is not None:
                device.set_reflected_force(pos_map.T @ reflected)

            if not args.no_plot and step_index >= next_plot_step[0]:
                while next_plot_step[0] <= step_index:
                    next_plot_step[0] += plot_every
                with plot_lock:
                    trace_force.append(env.normal_force_n())

            if step_index % max(1, int(round(args.control_freq / args.view_fps))) == 0:
                env.refresh_visual_gradient()

            if args.auto_finish and collection["state"] == "recording" and (env.success() or env.broken):
                stop_episode("auto_success" if env.success() else "auto_broken")

            if recorder is not None and collection["state"] == "recording":
                # record_sample dedups by image_id itself (only actually
                # appends a frame when it's genuinely new), so it's fine to
                # pass the latest async render every tick unconditionally --
                # same pattern as teleop_flipup.py's dataset block.
                recorder.record_sample(
                    env,
                    timestamp_ms=step_index * 1000.0 / args.control_freq,
                    target_pos=target,
                    target_rotvec=None,
                    device_state=device_state,
                    sent_force=reflected,
                    image_rgb=shot["frame"],
                    image_capture_time_s=shot["sim_time_s"],
                    image_id=int(shot["n"]),
                )

            if not args.no_readout and now - last_print >= args.print_interval_s:
                fn = env.normal_force_n()
                cov = env.coverage_fraction("just_right")
                under = env.coverage_fraction("under")
                over = env.coverage_fraction("over")
                flag = "   BROKEN" if env.broken else ("   SUCCESS" if env.success() else "")
                rec_flag = f"  [{collection['state'].upper()}]" if recorder is not None else ""
                print(
                    f"\rF={fn:5.1f}N  max={env.episode_max_force_n:5.1f}N  cov(ok/under/over)="
                    f"{cov*100:5.1f}/{under*100:5.1f}/{over*100:5.1f}%{rec_flag}{flag}   ",
                    end="",
                )
                last_print = now

            elapsed_this_tick = time.monotonic() - now
            sleep_s = dt - elapsed_this_tick
            if sleep_s > 0.0 and not args.dry_run:
                time.sleep(sleep_s)
    finally:
        if collection["state"] == "recording":
            stop_episode("shutdown")
        if collection["state"] == "review":
            resolve_episode(keep=True)
        viewer_state["run"] = False
        if viewer_thread is not None:
            viewer_thread.join(timeout=2.0)
        shot["run"] = False
        if render_thread is not None:
            render_thread.join(timeout=2.0)
        if device is not None:
            device.close()


if __name__ == "__main__":
    main()
