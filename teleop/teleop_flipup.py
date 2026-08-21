"""
Teleoperate the flipup_minimal FlipUp task (UR5e + closed WSG50 pivoting a book
upright against a bookend) with the Force Dimension omega.

Same haptic pipeline as teleop_ball.py -- absolute position mapping, slew-limited
target, true-contact-force reflection, one-pole force filter, passivity check at
the achieved loop rate, cv2 viewer with the force strip chart and per-axis panel,
CSV/video recording, --diagnose. The differences from teleop_ball, all forced by
this being an arm rather than a free-floating ball, are:

* The manipulator is the WSG50 tip. You command its POSITION; the wrist
  orientation is derived from the target exactly as the scripted heuristic
  derives it, so the omega's 3 translational axes are enough.
* The arm keeps the heuristic's stiff 16 kN/m task-space gains. Softening them
  to "renderable" values breaks the task -- see flipup_teleop.py. What makes the
  stiff arm renderable is that the force gain is ~10x smaller here, because this
  task's contact forces are ~10x BallPush's (22 N median while levering the book
  vs ~2 N sliding a light block). The felt force lands in the same 1-9 N band and
  the felt stiffness in the same few-kN/m band.
* --force-source defaults to ``contact`` (the true solver contact force, exactly
  0.00 N in free space). BallPush's actuator-side ``estimated`` is available but
  is the weaker choice here: the arm's free-space tracking lag times 16 kN/m is
  tens of newtons of phantom force.
* The viewer renders in a background thread. dm_control software-renders this
  scene at 13-20 fps because of the high-resolution meshes, and a blocking
  render would stall the force loop for ~50 ms at a time.

Examples
--------
    python teleop_flipup.py                       # seed 0, cv2 view
    python teleop_flipup.py --seed 1              # another scene (0,1,2,4 are solvable by the scripted arc)
    python teleop_flipup.py --stiffness 5000      # firmer contact (check the margin line)
    python teleop_flipup.py --scale 5 5 5         # more sim travel per cm of handle
    python teleop_flipup.py --render-quality collision   # 9 ms frames, ugly but smooth
    python teleop_flipup.py --no-episode-randomization  # fixed book and start pose
    python teleop_flipup.py --dry-run             # no device: scripted arc, checks the loop
    python teleop_flipup.py --collect-dataset ~/data/flipup_sim_1khz.zarr
    python teleop_floating_flipup.py --collect-dataset ~/data/flipup_float_1khz.zarr

Push the handle AWAY from you to drive the tool into the bookend, and UP to
lever the book over. Normally, 'r' resets and 'q'/ESC quits. During dataset
collection, 's' starts/stops an episode. After stopping, click KEEP or DELETE
(keyboard: 'k'/'d'); the simulation remains paused until that decision.
``--auto-finish`` stops the active episode when the book reaches success.
Before collection start, return the handle to fixed ``--home`` and hold it still;
the viewer changes to HANDLE READY when the start gate will accept ``s``.
"""

import argparse
import os
import sys
import threading
import time

# The cv2 viewer renders off-screen; osmesa needs no on-screen GLX/EGL context.
# Must be set before mujoco/dm_control are imported.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flipup_teleop import (  # noqa: E402
    BOOK_COLOR_PALETTE,
    DEFAULT_ARM_DAMPING,
    DEFAULT_FORCE_CLIP,
    DEFAULT_JOINT_KD,
    DEFAULT_SURFACE_FORCE_LIMIT,
    DEFAULT_TOOL_KP,
    DEFAULT_TOOL_ROT_KD,
    DEFAULT_TOOL_ROT_KP,
    FlipUpTeleop,
    flipup_scene,
    sample_book_color,
    sample_episode_properties,
    sample_start_pose,
)
from flipup.physical_properties import (  # noqa: E402
    DEFAULT_PHYSICAL_PROPERTIES,
    PhysicalProperties,
    sample_physical_properties,
)


# Force Dimension's device frame is +x toward the operator, +y to the operator's
# right, +z up. The default camera is the original left-oblique view (azimuth
# -30), so the push direction runs mostly INTO the screen, and the lift
# direction is straight up the screen. So the natural mapping is
#     handle away from you (dev -x) -> tool into the bookend (sim +x)
#     handle up             (dev +z) -> tool up                (sim +z)
# and sim y takes -y_dev to keep the mapping right-handed. Same default as
# teleop_ball.py. Override with --axes if a direction comes out reversed.
DEFAULT_AXES = "-x,-y,z"


def lower_background_thread_priority(nice_increment):
    """Best-effort Linux priority drop for rendering/UI worker threads.

    MuJoCo's OSMesa renderer can occupy several CPU cores at once.  Keeping the
    renderer asynchronous prevents a single 40 ms frame from blocking control,
    while lowering its scheduler priority makes the 1 kHz simulation/haptic
    work win when both become runnable at the same instant.  Linux accepts a
    native thread id for ``PRIO_PROCESS``; other platforms simply skip this.
    """
    if nice_increment <= 0 or not hasattr(os, "setpriority"):
        return
    try:
        thread_id = threading.get_native_id()
        current = os.getpriority(os.PRIO_PROCESS, thread_id)
        os.setpriority(os.PRIO_PROCESS, thread_id, current + int(nice_increment))
    except (AttributeError, OSError):
        # Scheduling priority is an optimization, never a requirement for
        # collection (containers and non-Linux hosts may reject it).
        pass


def build_pos_map(spec):
    """
    Signed permutation matrix mapping omega axes -> sim axes.

    ``spec`` is three comma-separated entries: which device axis (optionally
    negated) drives sim x, y and z. Its transpose maps sim forces back onto
    device axes, which is what keeps contact resistance opposing the motion that
    caused it. (Same helper as teleop_ball.py; duplicated rather than imported so
    this script does not pull robosuite in.)
    """
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
    if np.linalg.det(m) < 0:
        print(f"[axes] note: {spec} is a mirrored (left-handed) mapping -- "
              f"motion will feel reflected. Negate one entry if that is not intended.")
    return m


def collection_home_metrics(device_state, fixed_home):
    """Return handle distance/speed and whether the SDK velocity is usable."""
    position = np.asarray(device_state["pos"], dtype=float)
    velocity = np.asarray(device_state.get("vel", np.zeros(3)), dtype=float)
    velocity_valid = bool(
        device_state.get("velocity_valid", "vel" in device_state)
    )
    return (
        float(np.linalg.norm(position - np.asarray(fixed_home, dtype=float))),
        float(np.linalg.norm(velocity)),
        velocity_valid,
    )


class GripOpenCalibrator:
    """Learn the relaxed omega gripper opening without requiring its nominal limit.

    Force Dimension gap ranges vary slightly between devices and hand fixtures.
    During collection idle the gentle opening force is already active, so the
    largest stable observed gap is a better open endpoint than a hard-coded
    value.  The calibrated endpoint is retained across episode resets.
    """

    def __init__(
        self,
        *,
        closed_m,
        nominal_open_m,
        auto=True,
        stable_s=0.35,
        tolerance_m=0.002,
        minimum_span_m=0.006,
        increase_epsilon_m=0.0002,
    ):
        self.closed_m = float(closed_m)
        self.nominal_open_m = float(nominal_open_m)
        self.nominal_span_m = abs(self.nominal_open_m - self.closed_m)
        self.auto = bool(auto)
        self.stable_s = float(stable_s)
        self.tolerance_m = float(tolerance_m)
        self.minimum_span_m = float(minimum_span_m)
        self.increase_epsilon_m = float(increase_epsilon_m)
        self.observed_max_span_m = -np.inf
        self._change_anchor_span_m = -np.inf
        self.direction = (
            None
            if self.auto
            else float(np.sign(self.nominal_open_m - self.closed_m))
        )
        self.last_increase_s = None
        self.reference_m = None if self.auto else self.nominal_open_m

    def observe(self, gap_m, now_s):
        gap_m = float(gap_m)
        now_s = float(now_s)
        if not np.isfinite(gap_m):
            return False
        if self.auto:
            offset = gap_m - self.closed_m
            if self.direction is None:
                if abs(offset) < self.minimum_span_m:
                    return False
                self.direction = float(np.sign(offset))
            directed_span = self.direction * offset
            if directed_span < 0.0:
                return False
            candidate_span = min(directed_span, self.nominal_span_m)
            self.observed_max_span_m = max(
                self.observed_max_span_m, candidate_span
            )
            if (
                candidate_span
                > self._change_anchor_span_m + self.increase_epsilon_m
            ):
                self._change_anchor_span_m = candidate_span
                self.last_increase_s = now_s
            span_ok = (
                self.observed_max_span_m >= self.minimum_span_m
            )
            stable = (
                self.last_increase_s is not None
                and now_s - self.last_increase_s >= self.stable_s
            )
            if self.reference_m is None and span_ok and stable:
                self.reference_m = (
                    self.closed_m + self.direction * self.observed_max_span_m
                )
            elif (
                self.reference_m is not None
                and self.observed_max_span_m
                > abs(self.reference_m - self.closed_m) + self.increase_epsilon_m
            ):
                # Improve calibration only while idle; callers do not invoke
                # observe during recording, so an episode's mapping stays fixed.
                self.reference_m = (
                    self.closed_m + self.direction * self.observed_max_span_m
                )
        if self.reference_m is None:
            return False
        direction = float(np.sign(self.reference_m - self.closed_m))
        directed_gap = direction * (gap_m - self.closed_m)
        reference_span = abs(self.reference_m - self.closed_m)
        return bool(directed_gap >= reference_span - self.tolerance_m)

    @property
    def effective_open_m(self):
        return (
            self.nominal_open_m
            if self.reference_m is None
            else float(self.reference_m)
        )


def smooth_collection_force_gain(elapsed_s, hold_s, ramp_s):
    """Smoothstep gain for engaging haptics after a collection start."""
    ramp_elapsed = float(elapsed_s) - max(0.0, float(hold_s))
    if ramp_elapsed <= 0.0:
        return 0.0
    if ramp_s <= 0.0:
        return 1.0
    u = np.clip(ramp_elapsed / float(ramp_s), 0.0, 1.0)
    return float(u * u * (3.0 - 2.0 * u))


def episode_start_safety(settle_error_m, contact_force_n, *,
                         max_settle_error_m, max_contact_force_n):
    """Classify a reset pose before it can become a recorded episode."""
    settle_error_m = float(settle_error_m)
    contact_force_n = float(contact_force_n)
    safe = (
        np.isfinite(settle_error_m)
        and np.isfinite(contact_force_n)
        and settle_error_m <= float(max_settle_error_m)
        and contact_force_n <= float(max_contact_force_n)
    )
    return bool(safe)


def map_wrist_orientation(
    device_rotation,
    device_home_rotation,
    rotation_map,
    tool_home_rotvec,
    *,
    frame="world",
    scale=1.0,
    deadzone=0.0,
):
    """Map an omega wrist frame to an absolute simulated-tool rotation.

    Returns ``(tool_rotvec, wrist_delta_rotvec)``. Keeping this transformation
    outside the control loop makes it testable with a synthetic wrist sample.
    """
    from scipy.spatial.transform import Rotation

    R_dev = np.asarray(device_rotation, dtype=float).reshape(3, 3)
    R_home = np.asarray(device_home_rotation, dtype=float).reshape(3, 3)
    P = np.asarray(rotation_map, dtype=float).reshape(3, 3)
    if frame == "world":
        device_delta = R_dev @ R_home.T
    elif frame == "tool":
        device_delta = R_home.T @ R_dev
    else:
        raise ValueError(f"unknown rotation frame {frame!r}")
    sim_delta = P @ device_delta @ P.T
    delta_rotvec = Rotation.from_matrix(sim_delta).as_rotvec()
    delta_angle = np.linalg.norm(delta_rotvec)
    if delta_angle <= float(deadzone):
        delta_rotvec = np.zeros(3)
    else:
        # Radial soft deadzone: subtract the threshold instead of jumping from
        # zero to ``deadzone`` radians at the boundary.
        delta_rotvec = (
            delta_rotvec
            * ((delta_angle - float(deadzone)) / delta_angle)
            * float(scale)
        )

    home = Rotation.from_rotvec(np.asarray(tool_home_rotvec, dtype=float))
    delta = Rotation.from_rotvec(delta_rotvec)
    command = delta * home if frame == "world" else home * delta
    return command.as_rotvec(), delta_rotvec


def book_properties(args):
    """PhysicalProperties from --randomize-physics plus any explicit overrides."""
    base = (sample_physical_properties(args.seed) if args.randomize_physics
            else DEFAULT_PHYSICAL_PROPERTIES)
    overrides = {
        "mass_kg": args.book_mass,
        "sliding_friction": args.book_friction,
        "length_m": args.book_length,
        "width_m": args.book_width,
        "thickness_m": args.book_thickness,
    }
    kwargs = {
        "mass_kg": base.mass_kg,
        "sliding_friction": base.sliding_friction,
        "torsional_friction": base.torsional_friction,
        "rolling_friction": base.rolling_friction,
        "length_m": base.length_m,
        "width_m": base.width_m,
        "thickness_m": base.thickness_m,
    }
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return PhysicalProperties(**kwargs)


def main(env_class=None):
    env_class = FlipUpTeleop if env_class is None else env_class
    floating = getattr(env_class, "controller_kind", "joint_arm") == "floating_gripper"
    task_kind = getattr(env_class, "task_kind", "flipup")
    cube_lift = task_kind == "cube_lift"
    if cube_lift:
        from floating_cube_lift_teleop import (
            CUBE_COLOURS,
            CUBE_CENTER_XY,
            CubeProperties,
            DEFAULT_CUBE_PROPERTIES,
            cube_lift_scene,
            sample_cube_color,
            sample_cube_properties,
            sample_cube_start_pose,
        )
    default_tool_kp = float(getattr(env_class, "default_tool_kp", DEFAULT_TOOL_KP))
    default_haptic_stiffness = float(
        getattr(env_class, "default_haptic_stiffness", 1500.0)
    )
    parser = argparse.ArgumentParser()
    # ---- scene ------------------------------------------------------------
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "randomized cube/property/start sequence"
            if cube_lift
            else "scene seed: bookend pose, yaw and book offset. --dry-run "
                 "solves seeds 0, 1, 2 and 4 of 0-5 through this pipeline; "
                 "3 and 5 need a human to adapt the path"
        ),
    )
    parser.add_argument("--randomize-physics", action="store_true",
                        help="sample book mass/friction/size from flipup's ranges")
    parser.add_argument("--book-mass", type=float, default=None, help="kg (default 1.375)")
    parser.add_argument("--book-friction", type=float, default=None,
                        help="book sliding friction (default 0.12)")
    parser.add_argument("--book-length", type=float, default=None, help="m (default 0.15)")
    parser.add_argument("--book-width", type=float, default=None, help="m (default 0.10)")
    parser.add_argument("--book-thickness", type=float, default=None, help="m (default 0.025)")
    if cube_lift:
        parser.add_argument("--cube-mass", type=float, default=0.03125,
                            help="cube mass in kg")
        parser.add_argument("--cube-size", type=float, default=0.0275,
                            help="cube outer side length in m")
        parser.add_argument("--cube-corner-radius", type=float, default=0.003,
                            help="rounded-edge radius in m")
        parser.add_argument("--cube-friction", type=float, default=0.85,
                            help="cube sliding friction coefficient")
        parser.add_argument("--cube-position-jitter", type=float, nargs=2,
                            default=list(getattr(
                                env_class, "default_cube_position_jitter", [0.025, 0.025]
                            )), metavar=("X", "Y"),
                            help="full x/y range for per-episode cube placement")
    parser.add_argument(
        "--episode-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "randomize cube colour, size, mass, placement, and gripper start "
            "for each episode (default: enabled)"
            if cube_lift
            else "randomize book cover, height, width and mass independently for "
                 "each episode (default: enabled)"
        ),
    )
    parser.add_argument(
        "--cube-size-jitter" if cube_lift else "--book-size-jitter",
        dest="book_size_jitter",
        type=float,
        default=(0.0 if cube_lift else 0.20),
        help="independent +/- object size fraction per episode",
    )
    parser.add_argument(
        "--cube-mass-jitter" if cube_lift else "--book-mass-jitter",
        dest="book_mass_jitter",
        type=float,
        default=0.20,
        help="independent +/- object mass fraction per episode",
    )
    parser.add_argument(
        "--start-prism",
        type=float,
        nargs=3,
        default=list(getattr(env_class, "default_start_prism", [0.04, 0.06, 0.05])),
        metavar=("DEPTH", "LATERAL", "VERTICAL"),
        help="full size in metres of the fixed initial-tool sampling prism "
             "centred at the nominal pre-contact pose",
    )
    parser.add_argument("--start-center-prob", type=float, default=0.70,
                        help="probability of a tightly centre-biased start; remaining "
                             "episodes uniformly cover the full prism (default 0.70)")
    parser.add_argument("--start-max-contact-force", type=float, default=0.5,
                        help="reject and resample an initial tool pose if reset contact "
                             "on the robot exceeds this many N (default 0.5)")
    parser.add_argument("--start-max-settle-error", type=float, default=0.01,
                        help="reject and resample an initial tool pose if the settled "
                             "tool remains farther away than this many m (default 0.01)")
    parser.add_argument("--start-max-resamples", type=int, default=32,
                        help="maximum unsafe initial-pose resamples per episode")
    parser.add_argument(
        "--tool-kp",
        type=float,
        default=default_tool_kp,
        help=(
            "task-space translational stiffness (N/m). Floating-gripper default "
            "5000 gives more realistic contact force without arm dynamics. Full-arm "
            "default remains 16000 because lower values lose the book edge. This is "
            "the simulated controller; --stiffness separately controls handle feel."
        ),
    )
    parser.add_argument("--tool-rot-kp", type=float,
                        default=(300.0 if floating else DEFAULT_TOOL_ROT_KP),
                        help="task-space rotational stiffness (N m/rad). Arm default "
                             "3000; floating-gripper default 300 because its rotational "
                             "inertia is about two orders of magnitude smaller. "
                             "higher tracks faster but abrupt commands saturate the "
                             "28 N m wrist actuators and disturb position tracking.")
    parser.add_argument("--tool-rot-kd", type=float,
                        default=(None if floating else DEFAULT_TOOL_ROT_KD),
                        help="task-space rotational damping (N m s/rad). Default 90, "
                             "approximately critical at the start pose for rot-kp 3000. "
                             "The floating gripper derives its much smaller critical "
                             "value from gripper inertia. Set 0 to disable it.")
    parser.add_argument("--arm-damping", type=float, default=None,
                        help="multiplier on the arm's joint-space damping, relative to "
                             "the value flipup ships (64 N m s/rad on the arm joints, 16 "
                             f"on the wrist). Default {DEFAULT_ARM_DAMPING:.1f}: a 3 cm "
                             "step settles without overshoot and a scripted-path sweep "
                             "has fewer contact dropouts than at 2.0, for about 0.9 mm "
                             "more mean tracking lag. Use 2.0 for faster tracking; avoid "
                             "values above 4.0. Pass 1.0 to reproduce flipup exactly.")
    parser.add_argument("--settle", type=float, default=(0.0 if floating else 2.5),
                        help="seconds of sim time to slew the tool to the start pose "
                             "after each reset, before the operator takes over")
    parser.add_argument(
        "--standoff",
        type=float,
        default=float(getattr(env_class, "default_standoff", 0.05)),
        help=(
            "initial vertical clearance above the cube centre (m)"
            if cube_lift
            else "how far in front of the book edge the tool starts (m)"
        ),
    )
    parser.add_argument(
        "--tip-softness",
        type=float,
        default=float(getattr(env_class, "default_tip_softness", 0.0)),
        help="floating gripper only: interpolate fingertip contact from the "
             "unchanged model at 0 (10 ms, damping ratio 1, 3 mm impedance "
             "width) to a softer pad at 1 (20 ms, ratio 2, 5 mm). Try 0.5; "
             "default 0 preserves current behavior",
    )
    parser.add_argument(
        "--table-solref",
        type=float,
        nargs=2,
        default=None,
        metavar=("TIME_CONSTANT_S", "DAMPING_RATIO"),
        help="floating gripper only: override the visible table surface's raw "
             "MuJoCo solref [time_constant, damping_ratio], replacing table.xml's "
             "compiled (0.015, 2.0). Smaller time_constant makes the table "
             "resist penetration sooner (stiffer, higher peak force); larger "
             "makes it softer and slower to push back (more penetration, "
             "lower peak force). Leave unset to keep the compiled value",
    )
    parser.add_argument(
        "--table-solimp",
        type=float,
        nargs=5,
        default=None,
        metavar=("D0", "D_WIDTH", "WIDTH", "MIDPOINT", "POWER"),
        help="floating gripper only: override the visible table surface's raw "
             "MuJoCo solimp [d0, d_width, width, midpoint, power], replacing "
             "table.xml's compiled (0.85, 0.95, 0.004, 0.5, 2.0). Increasing "
             "width spreads the impedance transition over more penetration for "
             "a more gradual onset. Leave unset to keep the compiled value",
    )
    parser.add_argument(
        "--force-sensor-cutoff",
        type=float,
        default=float(getattr(env_class, "default_force_sensor_cutoff", 0.0)),
        help="floating gripper only: cutoff in Hz of each pole in a causal "
             "two-pole F/T sensor model, updated at 1 kHz in the tool frame. "
             "0 keeps the exact raw contact wrench; try 30 for a realistic "
             "smooth measurement while retaining raw ground truth",
    )
    parser.add_argument("--grasp-force-limit", type=float, default=25.0,
                        help="cube lift only: smooth simulated WSG50 closing-force "
                             "limit in N (default 25)")
    parser.add_argument("--gripper-speed", type=float, default=0.12,
                        help="cube lift only: maximum simulated finger command speed "
                             "in m/s")
    parser.add_argument("--success-height", type=float, default=0.08,
                        help="cube lift only: required clearance between cube bottom "
                             "and tabletop in m")
    parser.add_argument("--device-grip-closed", type=float, default=0.0,
                        help="omega.7 gap in m mapped to closed simulated fingers")
    parser.add_argument("--device-grip-open", type=float, default=0.025,
                        help="nominal omega.7 open gap in m; during dataset idle "
                             "the sign and reachable endpoint are inferred from "
                             "the device's stable relaxed opening")
    parser.add_argument(
        "--device-grip-auto-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="learn the physical open endpoint while the idle opening force is "
             "active; disable to use --device-grip-open exactly",
    )
    parser.add_argument("--device-grip-calibration-ms", type=float, default=350.0,
                        help="time the largest observed relaxed gap must remain "
                             "stable before S is accepted")
    parser.add_argument("--device-grip-open-tolerance", type=float, default=0.002,
                        help="allowed distance below the calibrated open gap at "
                             "collection start (m)")
    parser.add_argument("--device-grip-min-span", type=float, default=0.006,
                        help="minimum gap above --device-grip-closed accepted as "
                             "a valid automatic open calibration (m)")
    parser.add_argument("--grip-force-gain", type=float, default=0.08,
                        help="omega.7 squeeze feedback N per N of simulated grasp load")
    parser.add_argument("--grip-feedback-sign", type=float, choices=[-1.0, 1.0],
                        default=-1.0, help="sign of omega.7 grasp resistance; -1 "
                             "pushes the jaws open and normally resists squeezing")
    parser.add_argument("--grip-force-tau", type=float, default=10.0,
                        help="omega.7 grasp-feedback low-pass time constant in ms")
    parser.add_argument("--grip-force-rate", type=float, default=60.0,
                        help="omega.7 grasp-feedback slew limit in N/s; 0 disables")
    parser.add_argument("--max-grip-force", type=float, default=3.0,
                        help="absolute omega.7 gripper feedback limit in N")
    parser.add_argument("--grip-damping", type=float, default=6.0,
                        help="omega.7 gripper-axis damping in N/(m/s)")
    parser.add_argument("--collection-grip-open-force", type=float, default=0.35,
                        help="cube lift only: gentle opening force used while idle (N)")

    # ---- mapping ----------------------------------------------------------
    parser.add_argument("--scale", type=float, nargs=3,
                        default=list(getattr(env_class, "default_scale", [4.0, 4.0, 4.0])),
                        help="sim metres per device metre, per axis. The flip arc "
                             "spans ~15 cm of push and ~14 cm of lift, so 4 asks for "
                             "~3.7 cm of handle travel in each. Raising it also raises "
                             "the felt stiffness (= tool_kp * scale * force_gain)")
    parser.add_argument("--home", type=float, nargs=3,
                        default=list(getattr(env_class, "default_device_home", [0.02, 0.0, -0.02])),
                        help="device home position (m). Offset +x/-z by default so "
                             "there is room to push AWAY (dev -x = into the bookend) "
                             "and UP (dev +z) without hitting the workspace stops")
    parser.add_argument("--axes", type=str, default=DEFAULT_AXES,
                        help="which device axis drives each sim axis, as "
                             "'<simx>,<simy>,<simz>'. Default '-x,-y,z'. Negate an "
                             "entry if that direction is reversed (e.g. 'x,-y,z'); "
                             "reorder them if two axes are swapped (e.g. '-y,x,z')")
    parser.add_argument("--enable-rotation", action="store_true",
                        help="6-DoF: the omega's wrist drives the tool's roll/pitch/yaw "
                             "as well as xyz. Off by default because the flip does not "
                             "need it -- the wrist otherwise follows the heuristic's own "
                             "rule (pitched 30 deg down, yawed away from the base). NOTE "
                             "there is no torque feedback: the omega.6/.7 wrist is "
                             "passive, so rotation is open-loop while translation is not")
    parser.add_argument("--rot-scale", type=float, default=1.0,
                        help="wrist rotation amplification -- scales the ANGLE and keeps "
                             "the axis, like --scale does for position")
    parser.add_argument("--rot-axes", type=str, default=None,
                        help="signed permutation for ROTATION only, same syntax as "
                             "--axes; defaults to --axes. Use it when the wrist gimbal "
                             "reports pitch/yaw swapped independently of position (the "
                             "left-hand omega.7 needed that on the robosuite track)")
    parser.add_argument("--rot-frame", type=str, default="world",
                        choices=["world", "tool"],
                        help="world = turning the handle turns the tool about WORLD axes "
                             "(spatial delta, pre-multiplied); tool = about the tool's "
                             "own axes (body delta, post-multiplied). Mixing the two "
                             "conventions is what made the axes come out wrong in "
                             "teleop_ball, so they are kept paired here")
    parser.add_argument("--max-rot-speed", type=float, default=60.0,
                        help="cap on how fast the commanded tool ROTATION may travel "
                             "(deg/s), 0 = uncapped. The analogue of --max-speed, and it "
                             "matters more because the wrist actuators are limited to "
                             "28 N m; an uncapped orientation step can saturate them and "
                             "disturb position tracking.")
    parser.add_argument("--rot-deadzone", type=float, default=0.005,
                        help="radians of wrist rotation from the reset pose to ignore. "
                             "Default 0.005 (~0.3 deg), implemented as a smooth radial "
                             "deadzone around the home wrist pose.")
    parser.add_argument("--max-speed", type=float,
                        default=float(getattr(env_class, "default_max_speed", (0.60 if floating else 0.30))),
                        help="cap on how fast the tool TARGET may travel (m/s), "
                             "0 = uncapped. A speed limit, not a workspace limit: "
                             "--scale multiplies hand velocity too, so without it a "
                             "brisk hand drives the tool into the book at ~1 m/s")

    # ---- haptics ----------------------------------------------------------
    parser.add_argument("--force-source", type=str, default="contact",
                        choices=["contact", "estimated", "wrist", "none"],
                        help="contact (default) = true solver contact force on the "
                             "robot: exactly 0.00 N in free space, ground truth in "
                             "contact. wrist = the arm's tared WSG50 MuJoCo sensor, or "
                             "the floating gripper's --force-sensor-cutoff model. "
                             "estimated = BallPush's actuator-side "
                             "reconstruction, which is BROKEN for an arm this stiff: "
                             "measured 111 N of free-space phantom force. Kept only for "
                             "A/B comparison")
    parser.add_argument("--stiffness", type=float, default=default_haptic_stiffness,
                        help="target stiffness AT THE HANDLE (N/m) -- the thing that "
                             "decides whether contact feels solid or buzzes. force-gain "
                             "is derived as stiffness/(tool_kp*scale). Must stay under "
                             "2*damping/T_effective or the loop limit-cycles. Default "
                             "1800 for floating gripper, 1500 for the full arm")
    parser.add_argument("--force-gain", type=float, default=None,
                        help="N of handle force per N of sim contact force. Default is "
                             "derived from --stiffness; set this to override")
    parser.add_argument("--force-clip", type=float, default=DEFAULT_FORCE_CLIP,
                        help="ceiling on the reflected SIM force (N), the analogue of "
                             "ball_force_limit. The scripted flip itself peaks at "
                             "~145 N, so this only clips a hard jam")
    parser.add_argument(
        "--surface-force-limit",
        "--table-contact-force-limit",
        "--table-normal-force-limit",
        dest="surface_force_limit",
        type=float,
        default=float(getattr(env_class, "default_surface_force_limit", DEFAULT_SURFACE_FORCE_LIMIT)),
        help="maximum steady Cartesian spring force (N) pressing through the "
             "visible support-surface normal (table only for cube lift; table "
             "and bookend for flip-up). Tangential motion and object contact "
             "are unchanged; 0 disables support-surface anti-windup",
    )
    parser.add_argument(
        "--book-normal-force-limit",
        dest="book_normal_force_limit",
        type=float,
        default=float(getattr(env_class, "default_book_normal_force_limit", 0.0)),
        help="EXPERIMENTAL, floating gripper only. Maximum steady Cartesian "
             "spring force (N) pressing through the fingertip-book contact "
             "normal once contact is detected -- caps sustained wedging "
             "instead of impact. Tangential motion is unchanged; 0 (default) "
             "disables it. Try book_force / tool_kp for a starting deflection "
             "budget, e.g. 18/4000 ~= 4.5mm",
    )
    parser.add_argument(
        "--approach-compliance-distance",
        dest="approach_compliance_distance",
        type=float,
        default=0.0,
        metavar="METRES",
        help="floating gripper only: turn on variable compliance near the "
             "table/bookend. Above this many metres from the nearest guarded "
             "surface, --tool-kp is used unchanged; within it, kp (and its "
             "matching critically-damped kd) ramp linearly down to "
             "--approach-compliance-min-kp-ratio times tool_kp at, or below, "
             "the surface. 0 (default) disables the ramp -- current "
             "behavior. This targets the FIRST-CONTACT impact spike that the "
             "surface anti-windup cannot prevent (it only bounds sustained "
             "penetration after contact is already detected); try 0.03",
    )
    parser.add_argument(
        "--approach-compliance-min-kp-ratio",
        dest="approach_compliance_min_kp_ratio",
        type=float,
        default=0.2,
        help="floating gripper only: fraction of --tool-kp used once the "
             "tool is at or below the guarded surface, when "
             "--approach-compliance-distance > 0. Must be in (0, 1]; "
             "default 0.2 (i.e. 5x softer right at the surface)",
    )
    parser.add_argument(
        "--approach-compliance-max-speed",
        dest="approach_compliance_max_speed",
        type=float,
        default=0.0,
        metavar="M_PER_S",
        help="floating gripper only: the knob that actually limits impact "
             "energy. Requires --approach-compliance-distance > 0. Once "
             "within that distance of a guarded surface, clamps how fast "
             "the incoming target may move INTO the surface to this many "
             "m/s (tangential/outward motion is unaffected). 0 (default) "
             "disables it. The kp/kd ramp alone does NOT slow the tool down "
             "-- a critically damped spring tracking a constant-velocity "
             "target settles into a bigger lag at the SAME tool velocity as "
             "kp drops, so without this the tool still hits the surface at "
             "full --max-speed and the impact is absorbed entirely by the "
             "raw contact solver. Try 0.03-0.05",
    )
    parser.add_argument("--workspace-wall-stiffness", type=float,
                        default=float(getattr(env_class, "default_device_wall_stiffness", 0.0)),
                        help="physical-device workspace wall stiffness in N/m; "
                             "cube lift defaults to 800, flip-up to 0")
    parser.add_argument("--workspace-wall-half", type=float, nargs=3,
                        default=list(getattr(env_class, "default_device_wall_half", [0.0, 0.0, 0.0])),
                        help="half extents in device metres around --home for the "
                             "haptic workspace wall")
    parser.add_argument("--force-deadband", type=float, default=0.0,
                        help="N of sim force to ignore. 0 by default because the "
                             "contact source is already exactly zero in free space; "
                             "raise it for --force-source estimated/wrist")
    parser.add_argument("--tool-damping", type=float, default=0.0,
                        help="Cartesian damping (N/(m/s)) used only by --force-source "
                             "estimated. 0 by default because it does not help: a "
                             "least-squares fit over 7863 free-space samples returns "
                             "14 N/(m/s), i.e. the arm's lag is not velocity-"
                             "proportional (it is dominated by inertia and the "
                             "joint-space damping structure), and no value brings the "
                             "residual below ~70 N")
    parser.add_argument("--force-rate", type=float, default=120.0,
                        help="cap on how fast the handle force may change (N/s), "
                             "0 = uncapped. Aimed squarely at the thud when the "
                             "fingertip first meets the book: the sim force goes 0 -> 80 "
                             "N in a few ms there, and unlike --force-tau this softens "
                             "that onset WITHOUT lagging the steady force or the ripple "
                             "(measured: onset slope 0.35 -> 0.12 N/ms at 120 N/s, "
                             "0.06 at 60, with the same dwell force and tremor ripple)")
    parser.add_argument("--force-tau", type=float, default=2.0,
                        help="time constant of the smoothing on the handle force, in "
                             "MILLISECONDS (0 = raw). This filter sits INSIDE the "
                             "feedback loop, so its lag counts against stability twice "
                             "over -- raising it to cure buzzing lowers the stiffness "
                             "you may safely render. 2 ms is what puts this at the same "
                             "4x passivity margin teleop_ball's settled tuning has "
                             "(measured: tau 2 -> 4.0x and 38 mN/step of felt ripple, "
                             "tau 4 -> 2.2x and 23 mN/step)")
    parser.add_argument("--max-force", type=float, default=10.0,
                        help="clamp on the handle force vector magnitude (N)")
    parser.add_argument("--damping", type=float, default=30.0,
                        help="handle velocity damping (N/(m/s)). FDOmega ramps this in "
                             "with force, so free space stays effortless and it only "
                             "acts on contact -- where it buys stability headroom "
                             "(stable stiffness limit = 2*b/T)")
    parser.add_argument("--auto-init", action="store_true",
                        help="auto-calibrate the omega on open (it will move)")

    # ---- loop / view ------------------------------------------------------
    parser.add_argument("--control-freq", type=int, default=1000,
                        help="sim + force update rate (Hz). Directly sets the stability "
                             "limit, so higher is better; capped at 1/sim-timestep = "
                             "1000 Hz, which is the default (one 1 ms sim step per "
                             "control step) and holds real-time factor 1.00 here")
    parser.add_argument("--cam-res", type=int, nargs=2, default=[640, 480])
    parser.add_argument("--view-fps", type=float, default=30.0,
                        help="cap on the viewer's frame rate. Worth having: with "
                             "--render-quality collision the render thread otherwise "
                             "runs at ~290 fps and the main loop spends its budget "
                             "drawing plots instead of updating forces")
    parser.add_argument(
        "--viewer-scale",
        type=float,
        default=2.0,
        help="scale only the displayed window (default 2 = twice the width and "
             "height). Rendering, dataset RGB, video, and control timing retain "
             "their native resolution",
    )
    parser.add_argument("--arm-view", type=str, default="hidden",
                        choices=["hidden", "ghost", "full"],
                        help="hidden (default) draws only the WSG50, not the UR5e links: "
                             "looking down the robot's own reach direction is otherwise "
                             "dominated by the forearm (14-39%% of the frame at azimuth "
                             "0, and it covers the gripper too -- hiding it raises the "
                             "gripper's visible area from 13.5k to 22.8k px). ghost draws "
                             "them translucent, full restores them. Visualization only; "
                             "collisions are unchanged")
    parser.add_argument("--render-quality", type=str, default="fast",
                        choices=["full", "fast", "collision"],
                        help="full ~75 ms/frame, fast (no reflection/shadow/skybox) "
                             "~51 ms, collision (collision geoms only) ~9 ms. The cost "
                             "is the high-resolution meshes, not the resolution")
    parser.add_argument("--cam-azimuth", type=float,
                        default=float(getattr(env_class, "default_cam_azimuth", -30.0)),
                        help="0 is dead-on the way the robot reaches (world +x), 90 is "
                             "square on to the plane the book pivots in. Default -30 is "
                             "the left-oblique view from the original teleop setup: "
                             "enough side angle that the book's tilt "
                             "reads at all, which at exactly 0 it geometrically cannot "
                             "(the long axis projects to a vertical line for EVERY tilt "
                             "and elevation there). Measured apparent tilt at a true "
                             "35.4 deg: az 0 -> 89.9 deg (no information), az 15 -> 59.2, "
                             "az 30 -> 52.6, az 90 -> ~35. Raise it toward 45-90 to judge "
                             "the angle geometrically rather than off the overlay")
    parser.add_argument("--cam-elevation", type=float,
                        default=float(getattr(env_class, "default_cam_elevation", -25.0)),
                        help="degrees above the horizontal, negative = looking down. "
                             "Steeper hides more of the book behind the gripper "
                             "(measured 3.8%% occluded at -20, 13.6%% at -40)")
    parser.add_argument("--cam-name", type=str,
                        default=getattr(env_class, "default_cam_name", None),
                        help="render a fixed camera compiled into the scene instead of "
                             "the free one, e.g. 'wsg50/d435i/rgb' for the floating "
                             "gripper or 'ur5e/wsg50/d435i/rgb' for the arm -- the wrist "
                             "RealSense, literally the robot's own viewpoint. Note it "
                             "moves WITH the tool, so the book looks static while the "
                             "world rotates around it; fine for recording policy "
                             "observations, disorienting to drive from. Overrides "
                             "--cam-azimuth/--cam-elevation/--cam-distance")
    parser.add_argument(
        "--free-camera",
        dest="cam_name",
        action="store_const",
        const=None,
        help="use the external free camera even when the task defaults to a fixed "
             "wrist camera; configure it with --cam-azimuth, --cam-elevation, and "
             "--cam-distance",
    )
    parser.add_argument("--cam-distance", type=float,
                        default=float(getattr(env_class, "default_cam_distance", 0.75)),
                        help="metres from the middle of the flip arc. The default "
                             "frames the book, bookend and fingertips; back out to "
                             "0.9+ to see the whole arm")
    parser.add_argument("--no-readout", action="store_true")
    parser.add_argument("--no-plot", action="store_true",
                        help="hide the live force strip chart drawn in the viewer")
    parser.add_argument("--plot-fixed-scale", action="store_true",
                        help="pin the force plots to +/- --max-force instead of "
                             "autoscaling to what is actually happening")
    parser.add_argument("--plot-span", type=float, default=4.0,
                        help="seconds of force history shown in the strip chart")
    parser.add_argument("--record-video", type=str, default=None,
                        help="write the whole viewer to a video file. Extension picks "
                             "the codec: .mp4 = H.264, .webm = VP9 (use this if your "
                             "viewer is Chromium-based, e.g. the VS Code preview)")
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--record", type=str, default=None,
                        help="write a CSV log (time, handle, target, tool, contact "
                             "force, commanded force, book pose, success) on exit")
    parser.add_argument("--collect-dataset", type=str, default=None,
                        help="append demonstrations to a Pyrite-compatible Zarr dataset. "
                             "Stores state, action, sensed F/T, solver ground-truth "
                             "wrench, compatibility compliance labels, and full MuJoCo "
                             "replay state at --dataset-hz; RGB is asynchronous")
    parser.add_argument("--auto-finish", action="store_true",
                        help="while collecting, stop the active episode automatically "
                             "when the task succeeds; the episode still "
                             "waits for KEEP/DELETE confirmation")
    parser.add_argument("--dataset-hz", type=float, default=1000.0,
                        help="state/wrench recording rate (default 1000 Hz; RGB keeps "
                             "its asynchronous renderer timestamps)")
    parser.add_argument("--dataset-image-size", type=int, nargs=2, default=[224, 224],
                        metavar=("WIDTH", "HEIGHT"),
                        help="stored Pyrite RGB size (default 224 224)")
    parser.add_argument("--dataset-no-rgb", action="store_true",
                        help="do not render observations while collecting; store black "
                             "placeholder RGB frames for low-dimensional experiments")
    parser.add_argument("--dataset-min-samples", type=int, default=20,
                        help="discard episodes shorter than this many control samples")
    parser.add_argument("--collection-home-tolerance", type=float, default=0.005,
                        help="maximum handle distance from fixed --home before S can "
                             "start a dataset episode, in m (default 0.005)")
    parser.add_argument("--collection-home-speed", type=float, default=0.015,
                        help="maximum handle speed before S can start a dataset episode, "
                             "in m/s (default 0.015)")
    parser.add_argument("--collection-recenter-stiffness", type=float, default=100.0,
                        help="idle-only haptic spring pulling the handle toward fixed "
                             "--home, in N/m (default 100; 0 disables)")
    parser.add_argument("--collection-recenter-max-force", type=float, default=2.0,
                        help="independent force cap on the idle centering pull, in N "
                             "(default 2.0)")
    parser.add_argument("--collection-home-dwell-ms", type=float, default=250.0,
                        help="time the centered handle must remain slow before S is "
                             "accepted (default 250 ms)")
    parser.add_argument("--collection-takeover-hold-ms", type=float, default=100.0,
                        help="hold the sampled simulated start after S before handle "
                             "motion takes over (default 100 ms)")
    parser.add_argument("--collection-force-ramp-ms", type=float, default=400.0,
                        help="smooth haptic feedback engagement time after the takeover "
                             "hold (default 400 ms; 0 disables the ramp)")
    parser.add_argument("--dataset-wrench-filter", type=float, default=0.25,
                        help="seconds in the causal F/T moving average used to generate "
                             "adaptive-compliance labels")
    parser.add_argument("--ac-k-max", type=float, default=16000.0,
                        help="maximum translational stiffness label (N/m)")
    parser.add_argument("--ac-k-min", type=float, default=2000.0,
                        help="minimum translational stiffness label (N/m)")
    parser.add_argument("--ac-f-low", type=float, default=2.0,
                        help="force magnitude below which adaptive stiffness is maximal")
    parser.add_argument("--ac-f-high", type=float, default=100.0,
                        help="force magnitude above which adaptive stiffness is minimal")
    parser.add_argument("--diagnose", action="store_true",
                        help="record contact force and handle motion, then report their "
                             "dominant oscillation frequencies at exit. Tells a haptic "
                             "limit cycle apart from sim-side contact chatter")
    parser.add_argument("--dry-run", action="store_true",
                        help="skip the device and drive the scripted flip arc instead")
    parser.add_argument("--dry-speed", type=float,
                        default=float(getattr(env_class, "default_dry_speed", 0.05)),
                        help="how fast --dry-run walks the scripted arc (m/s). The "
                             "scripted flip needs ~0.05; faster loses the book edge")
    parser.add_argument("--no-view", action="store_true",
                        help="no cv2 window (for headless runs). --record-video still "
                             "works; --collect-dataset also renders RGB unless "
                             "--dataset-no-rgb is set")
    args = parser.parse_args()
    if args.tool_kp <= 0.0:
        parser.error("--tool-kp must be greater than zero")
    if args.viewer_scale <= 0.0:
        parser.error("--viewer-scale must be greater than zero")
    if args.surface_force_limit < 0.0:
        parser.error("--surface-force-limit cannot be negative")
    if args.book_normal_force_limit < 0.0:
        parser.error("--book-normal-force-limit cannot be negative")
    if args.book_normal_force_limit > 0.0 and not floating:
        parser.error("--book-normal-force-limit is only supported on the floating gripper")
    if args.approach_compliance_distance < 0.0:
        parser.error("--approach-compliance-distance cannot be negative")
    if not 0.0 < args.approach_compliance_min_kp_ratio <= 1.0:
        parser.error("--approach-compliance-min-kp-ratio must be in (0, 1]")
    if args.approach_compliance_distance > 0.0 and not floating:
        parser.error(
            "--approach-compliance-distance is available only for the floating gripper"
        )
    if args.approach_compliance_max_speed < 0.0:
        parser.error("--approach-compliance-max-speed cannot be negative")
    if args.approach_compliance_max_speed > 0.0 and not floating:
        parser.error(
            "--approach-compliance-max-speed is available only for the floating gripper"
        )
    if (
        args.approach_compliance_max_speed > 0.0
        and args.approach_compliance_distance <= 0.0
    ):
        parser.error(
            "--approach-compliance-max-speed requires --approach-compliance-distance > 0"
        )
    if args.grasp_force_limit <= 0.0:
        parser.error("--grasp-force-limit must be positive")
    if args.gripper_speed <= 0.0:
        parser.error("--gripper-speed must be positive")
    if args.success_height <= 0.0:
        parser.error("--success-height must be positive")
    if args.device_grip_open == args.device_grip_closed:
        parser.error("--device-grip-open must differ from --device-grip-closed")
    if args.device_grip_calibration_ms < 0.0:
        parser.error("--device-grip-calibration-ms cannot be negative")
    if args.device_grip_open_tolerance < 0.0:
        parser.error("--device-grip-open-tolerance cannot be negative")
    if args.device_grip_min_span <= 0.0:
        parser.error("--device-grip-min-span must be positive")
    if args.device_grip_min_span >= abs(
        args.device_grip_open - args.device_grip_closed
    ):
        parser.error("--device-grip-min-span must be smaller than the configured grip span")
    if args.grip_force_gain < 0.0:
        parser.error("--grip-force-gain cannot be negative")
    if args.grip_force_tau < 0.0 or args.grip_force_rate < 0.0:
        parser.error("grip force tau/rate cannot be negative")
    if args.max_grip_force < 0.0 or args.grip_damping < 0.0:
        parser.error("grip force/damping limits cannot be negative")
    if args.collection_grip_open_force < 0.0:
        parser.error("--collection-grip-open-force cannot be negative")
    if args.workspace_wall_stiffness < 0.0:
        parser.error("--workspace-wall-stiffness cannot be negative")
    if any(value < 0.0 for value in args.workspace_wall_half):
        parser.error("--workspace-wall-half values cannot be negative")
    if args.tool_rot_kp <= 0.0:
        parser.error("--tool-rot-kp must be greater than zero")
    if args.tool_rot_kd is not None and args.tool_rot_kd < 0.0:
        parser.error("--tool-rot-kd cannot be negative")
    if args.rot_scale <= 0.0:
        parser.error("--rot-scale must be greater than zero")
    if args.rot_deadzone < 0.0:
        parser.error("--rot-deadzone cannot be negative")
    if args.max_rot_speed < 0.0:
        parser.error("--max-rot-speed cannot be negative")
    if args.arm_damping is not None and args.arm_damping <= 0.0:
        parser.error("--arm-damping must be greater than zero")
    if not 0.0 <= args.tip_softness <= 1.0:
        parser.error("--tip-softness must be in [0, 1]")
    if not floating and not np.isclose(args.tip_softness, 0.0):
        parser.error("--tip-softness is available only for the floating gripper")
    if not floating and args.table_solref is not None:
        parser.error("--table-solref is available only for the floating gripper")
    if not floating and args.table_solimp is not None:
        parser.error("--table-solimp is available only for the floating gripper")
    if args.table_solref is not None and args.table_solref[0] <= 0.0:
        parser.error("--table-solref time constant must be positive")
    if args.table_solref is not None and args.table_solref[1] <= 0.0:
        parser.error("--table-solref damping ratio must be positive")
    if args.force_sensor_cutoff < 0.0:
        parser.error("--force-sensor-cutoff cannot be negative")
    if args.force_sensor_cutoff >= 0.5 * args.control_freq:
        parser.error("--force-sensor-cutoff must be below half --control-freq")
    if not floating and not np.isclose(args.force_sensor_cutoff, 0.0):
        parser.error(
            "--force-sensor-cutoff is available only for the floating gripper"
        )
    if args.dataset_hz <= 0.0:
        parser.error("--dataset-hz must be greater than zero")
    if args.dataset_min_samples <= 0:
        parser.error("--dataset-min-samples must be greater than zero")
    if args.dataset_wrench_filter < 0.0:
        parser.error("--dataset-wrench-filter cannot be negative")
    if not (0.0 <= args.book_size_jitter < 1.0):
        parser.error("--book-size-jitter must be in [0, 1)")
    if not (0.0 <= args.book_mass_jitter < 1.0):
        parser.error("--book-mass-jitter must be in [0, 1)")
    if cube_lift:
        if args.cube_size <= 0.0 or args.cube_mass <= 0.0:
            parser.error("--cube-size and --cube-mass must be positive")
        if not 0.0 < args.cube_corner_radius < 0.45 * args.cube_size:
            parser.error("--cube-corner-radius must be between 0 and 45% of size")
        if args.cube_friction < 0.0:
            parser.error("--cube-friction cannot be negative")
        if any(value < 0.0 for value in args.cube_position_jitter):
            parser.error("--cube-position-jitter values cannot be negative")
    if any(value < 0.0 for value in args.start_prism):
        parser.error("--start-prism values must be nonnegative")
    if not (0.0 <= args.start_center_prob <= 1.0):
        parser.error("--start-center-prob must be in [0, 1]")
    if args.start_max_contact_force < 0.0:
        parser.error("--start-max-contact-force cannot be negative")
    if args.start_max_settle_error < 0.0:
        parser.error("--start-max-settle-error cannot be negative")
    if args.start_max_resamples < 0:
        parser.error("--start-max-resamples cannot be negative")
    if args.collection_home_tolerance < 0.0:
        parser.error("--collection-home-tolerance cannot be negative")
    if args.collection_home_speed < 0.0:
        parser.error("--collection-home-speed cannot be negative")
    if args.collection_recenter_stiffness < 0.0:
        parser.error("--collection-recenter-stiffness cannot be negative")
    if args.collection_recenter_max_force < 0.0:
        parser.error("--collection-recenter-max-force cannot be negative")
    if args.collection_home_dwell_ms < 0.0:
        parser.error("--collection-home-dwell-ms cannot be negative")
    if args.collection_takeover_hold_ms < 0.0:
        parser.error("--collection-takeover-hold-ms cannot be negative")
    if args.collection_force_ramp_ms < 0.0:
        parser.error("--collection-force-ramp-ms cannot be negative")
    if any(value <= 0 for value in args.dataset_image_size):
        parser.error("--dataset-image-size values must be positive")
    if not (0.0 < args.ac_k_min <= args.ac_k_max):
        parser.error("--ac stiffness must satisfy 0 < --ac-k-min <= --ac-k-max")
    if not (0.0 <= args.ac_f_low < args.ac_f_high):
        parser.error("--ac force thresholds must satisfy 0 <= low < high")

    pos_map = build_pos_map(args.axes)
    # Rotation may need its own signed permutation: the wrist gimbal can report
    # pitch/yaw swapped independently of the position axes.
    rot_map = build_pos_map(args.rot_axes) if args.rot_axes else pos_map
    if args.enable_rotation and np.linalg.det(rot_map) < 0:
        raise SystemExit(f"--enable-rotation needs a right-handed rotation map; "
                         f"{args.rot_axes or args.axes} is mirrored (det -1). Negate one "
                         f"entry, or pass a proper rotation via --rot-axes.")
    scale = np.array(args.scale, dtype=float)
    W, H = args.cam_res
    if cube_lift:
        base_properties = CubeProperties(
            mass_kg=args.cube_mass,
            size_m=args.cube_size,
            corner_radius_m=args.cube_corner_radius,
            sliding_friction=args.cube_friction,
            torsional_friction=DEFAULT_CUBE_PROPERTIES.torsional_friction,
            rolling_friction=DEFAULT_CUBE_PROPERTIES.rolling_friction,
        )
    else:
        base_properties = book_properties(args)
    properties = base_properties
    size_envelope_scale = 1.0 + (
        args.book_size_jitter if args.episode_randomization else 0.0
    )
    collision_envelope_dimensions = (
        np.full(3, base_properties.size_m * size_envelope_scale, dtype=float)
        if cube_lift
        else np.array(
            [
                base_properties.length_m * size_envelope_scale,
                base_properties.width_m * size_envelope_scale,
                base_properties.thickness_m,
            ],
            dtype=float,
        )
    )

    env_kwargs = {
        "seed": args.seed,
        "tool_kp": args.tool_kp,
        "tool_rot_kp": args.tool_rot_kp,
        "tool_rot_kd": args.tool_rot_kd,
        "joint_kd": (
            None
            if args.arm_damping is None
            else DEFAULT_JOINT_KD * args.arm_damping
        ),
        "force_clip": args.force_clip,
        "surface_force_limit": args.surface_force_limit,
        "tool_damping": args.tool_damping,
        "physical_properties": properties,
        "standoff": args.standoff,
        "settle_s": args.settle,
        "offscreen": (max(W, 640), max(H, 480)),
        "collision_envelope_dimensions": collision_envelope_dimensions,
    }
    if floating:
        env_kwargs["tip_softness"] = args.tip_softness
        env_kwargs["force_sensor_cutoff_hz"] = args.force_sensor_cutoff
        env_kwargs["book_normal_force_limit"] = args.book_normal_force_limit
        env_kwargs["table_solref"] = args.table_solref
        env_kwargs["table_solimp"] = args.table_solimp
        env_kwargs["approach_compliance_distance_m"] = args.approach_compliance_distance
        env_kwargs["approach_compliance_min_kp_ratio"] = (
            args.approach_compliance_min_kp_ratio
        )
        env_kwargs["approach_max_speed_mps"] = args.approach_compliance_max_speed
    if cube_lift:
        env_kwargs.update(
            {
                "grasp_force_limit": args.grasp_force_limit,
                "gripper_speed": args.gripper_speed,
                "success_height": args.success_height,
            }
        )
    env = env_class(**env_kwargs)
    env.set_arm_visual(args.arm_view)
    render_model_lock = threading.Lock()
    episode_attempt = [0]
    episode_spec = {}

    def configure_episode_attempt(attempt_index):
        """Deterministically configure one collection attempt before recording."""
        nonlocal properties
        rng = np.random.default_rng(
            np.random.SeedSequence([int(args.seed), int(attempt_index), 0xF11F])
        )
        if args.episode_randomization and cube_lift:
            properties = sample_cube_properties(
                base_properties,
                rng,
                size_jitter=args.book_size_jitter,
                mass_jitter=args.book_mass_jitter,
            )
            color = sample_cube_color(rng)
            if attempt_index == 0:
                object_xy = CUBE_CENTER_XY.copy()
            else:
                object_xy = CUBE_CENTER_XY + np.clip(
                    rng.normal(0.0, 0.22, 2), -0.5, 0.5
                ) * np.asarray(args.cube_position_jitter, dtype=float)
        elif args.episode_randomization:
            properties = sample_episode_properties(
                base_properties,
                rng,
                size_jitter=args.book_size_jitter,
                mass_jitter=args.book_mass_jitter,
            )
            color = sample_book_color(rng)
            object_xy = None
        else:
            properties = base_properties
            color = (
                CUBE_COLOURS[1].copy()
                if cube_lift
                else BOOK_COLOR_PALETTE[5].copy()
            )
            object_xy = CUBE_CENTER_XY.copy() if cube_lift else None
        candidate_scene = (
            cube_lift_scene(
                args.seed,
                properties,
                standoff=args.standoff,
                object_xy=object_xy,
            )
            if cube_lift
            else flipup_scene(args.seed, properties, standoff=args.standoff)
        )
        rejected_starts = []
        start_position = None
        start_sample = None
        for resample_index in range(args.start_max_resamples + 1):
            sampler = sample_cube_start_pose if cube_lift else sample_start_pose
            start_position, start_sample = sampler(
                candidate_scene,
                rng,
                prism_size=args.start_prism,
                center_probability=args.start_center_prob,
                force_center=(attempt_index == 0 and resample_index == 0),
            )
            with render_model_lock:
                if cube_lift:
                    env.configure_episode(
                        properties, color, start_position, object_xy=object_xy
                    )
                else:
                    env.configure_episode(properties, color, start_position)
                settle_error = float(env.settle_error)
                contact_force = float(np.linalg.norm(env.contact_force()))
            if episode_start_safety(
                settle_error,
                contact_force,
                max_settle_error_m=args.start_max_settle_error,
                max_contact_force_n=args.start_max_contact_force,
            ):
                break
            rejected_starts.append(
                {
                    "position_world_m": np.asarray(start_position, dtype=float),
                    "settle_error_m": settle_error,
                    "contact_force_n": contact_force,
                }
            )
            print(
                f"[scene] rejecting start {resample_index + 1}: "
                f"settle {settle_error * 1000.0:.1f} mm, "
                f"contact {contact_force:.1f} N"
            )
        else:
            raise RuntimeError(
                f"could not sample a safe initial tool pose after "
                f"{args.start_max_resamples + 1} tries; reduce --start-prism or "
                "relax --start-max-contact-force/--start-max-settle-error"
            )
        start_sample = dict(start_sample)
        start_sample.update(
            {
                "resample_count": len(rejected_starts),
                "settle_error_m": settle_error,
                "initial_contact_force_n": contact_force,
                "rejected_starts": rejected_starts,
            }
        )
        episode_spec.clear()
        episode_spec.update(
            {
                "attempt_index": int(attempt_index),
                "random_seed_entropy": [int(args.seed), int(attempt_index), 0xF11F],
                "physical_properties": dict(properties.__dict__),
                "object_color_rgba": np.asarray(color, dtype=float),
                "book_color_rgba": np.asarray(color, dtype=float),
                "object_position_xy_m": object_xy,
                "start_sample": start_sample,
            }
        )
        return episode_spec

    max_cf = int(round(1.0 / env.timestep))
    if args.control_freq > max_cf:
        print(f"[warn] --control-freq {args.control_freq} exceeds 1/sim-timestep; "
              f"clamping to {max_cf}")
        args.control_freq = max_cf
    substeps = max(1, int(round((1.0 / args.control_freq) / env.timestep)))
    recorder = None
    dataset_stride = None
    if args.collect_dataset:
        ratio = args.control_freq / args.dataset_hz
        if not np.isclose(ratio, round(ratio), atol=1e-9):
            parser.error(
                "--control-freq must be an integer multiple of --dataset-hz "
                "for exact-rate collection"
            )
        dataset_stride = int(round(ratio))
        from pyrite_recorder import PyriteEpisodeRecorder

        recorder = PyriteEpisodeRecorder(
            args.collect_dataset,
            sample_hz=args.dataset_hz,
            image_size=tuple(args.dataset_image_size),
            include_rgb=not args.dataset_no_rgb,
            min_samples=args.dataset_min_samples,
            wrench_filter_seconds=args.dataset_wrench_filter,
            ac_k_max=args.ac_k_max,
            ac_k_min=args.ac_k_min,
            ac_f_low=args.ac_f_low,
            ac_f_high=args.ac_f_high,
        )
        # Continue the deterministic start/property sequence when appending to
        # an existing dataset instead of repeating attempt zero every launch.
        episode_attempt[0] = len(recorder.episode_names)
    configure_episode_attempt(episode_attempt[0])
    task_metric_name = "cube_lift_height_m" if cube_lift else "book_angle_deg"

    def task_metric_value():
        if hasattr(env, "task_metric_value"):
            return float(env.task_metric_value())
        return float(env.book_angle_deg())

    def task_metric_summary(value=None):
        value = task_metric_value() if value is None else float(value)
        if cube_lift:
            return (
                f"lift clearance={100.0 * value:.1f} cm "
                f"(need >= {100.0 * args.success_height:.1f})"
            )
        return f"book angle={value:.1f} deg from vertical (need < 15)"

    collection = {
        "state": "idle" if recorder is not None else "disabled",
        "reason": None,
        "success": False,
        "final_book_angle_deg": None,
        "final_task_metric": None,
        "started_monotonic": None,
        "started_sim_time_s": None,
    }
    collection_gate = {
        "ready": bool(args.dry_run),
        "within_since": None,
        "distance_m": 0.0 if args.dry_run else float("inf"),
        "speed_m_s": 0.0 if args.dry_run else float("inf"),
        "velocity_valid": bool(args.dry_run),
        "gripper_ready": bool(args.dry_run or not cube_lift),
        "gripper_gap_m": args.device_grip_open if args.dry_run else float("nan"),
        "gripper_open_reference_m": args.device_grip_open,
        "gripper_calibrated": bool(args.dry_run or not cube_lift),
    }
    force_monitor = {
        "grasp_current_n": 0.0,
        "grasp_max_n": 0.0,
        "table_current_n": 0.0,
        "table_max_n": 0.0,
    }

    def reset_force_monitor():
        for key in force_monitor:
            force_monitor[key] = 0.0

    review_action = [None]
    last_dataset_frame_id = [-1]

    # Derive the force gain from the stiffness the operator should feel, and check
    # it against the passivity limit for a sampled-data impedance display
    # (Colgate & Schenkel): the loop stays passive while k_handle < 2*b/T. Exceed
    # it and the hand-device-sim loop limit-cycles, which reads as the tool
    # bouncing on contact even when the sim itself is well behaved.
    if args.force_gain is None:
        args.force_gain = args.stiffness / max(env.tool_kp * scale[0], 1e-9)
    k_handle = env.tool_kp * scale[0] * args.force_gain
    ctl_dt = 1.0 / args.control_freq
    tau = args.force_tau / 1000.0
    force_alpha = 1.0 if tau <= 0 else ctl_dt / (tau + ctl_dt)
    t_eff = ctl_dt + 2.0 * tau
    k_limit = 2.0 * args.damping / t_eff

    print(f"[scene] seed {args.seed}: {properties.summary()}")
    kd_mult = DEFAULT_ARM_DAMPING if args.arm_damping is None else args.arm_damping
    if floating:
        print(
            f"[floating] direct gripper impedance kp/kd "
            f"{env.tool_kp:.0f}/{env.task_space_kd[0]:.1f} N/m and N s/m, "
            f"rotation {env.tool_rot_kp:.0f}/{env.tool_rot_kd:.1f} N m/rad and "
            f"N m s/rad; physical moving mass {env.gripper_mass_kg:.2f} kg, "
            f"gravity compensation {'ON' if env.gravity_compensation else 'OFF'}"
        )
        tip = env.tip_contact_parameters
        print(
            f"[contact] fingertip softness {tip['softness']:.2f}: "
            f"time constant {1000.0 * tip['solref_time_constant_s']:.1f} ms, "
            f"damping ratio {tip['solref_damping_ratio']:.2f}, impedance width "
            f"{1000.0 * tip['solimp_width_m']:.1f} mm"
        )
        sensor = env.force_sensor_parameters
        if sensor["enabled"]:
            print(
                f"[sensor] two-pole tool-frame F/T model: each pole "
                f"{sensor['pole_cutoff_hz']:.1f} Hz at "
                f"{sensor['sample_hz']:.0f} Hz; step t50 "
                f"{sensor['step_t50_ms']:.1f} ms, low-frequency group delay "
                f"{sensor['low_frequency_group_delay_ms']:.1f} ms. "
                "BC/plot use modeled force; raw solver truth is also recorded; "
                "default haptics stay raw"
            )
        else:
            print(
                "[sensor] force observation is raw solver contact "
                "(--force-sensor-cutoff 0); try 30 Hz for the two-pole model"
            )
        if cube_lift:
            print(
                f"[gripper] analogue WSG50 control: smooth closing-force limit "
                f"{env.grasp_force_limit:.1f} N, finger command speed "
                f"{env.gripper_speed:.3f} m/s; omega feedback gain "
                f"{args.grip_force_gain:.3f} N/N, tau "
                f"{args.grip_force_tau:.1f} ms, rate "
                f"{args.grip_force_rate:.1f} N/s, cap "
                f"{args.max_grip_force:.1f} N"
            )
            if recorder is not None and args.device_grip_auto_calibration:
                print(
                    f"[gripper] collection idle auto-calibrates the relaxed open "
                    f"gap within the nominal {1000.0 * abs(args.device_grip_open - args.device_grip_closed):.1f} mm "
                    f"span after {args.device_grip_calibration_ms:.0f} ms stable; "
                    "you do not need to reach the nominal mechanical limit"
                )
    else:
        print(f"[arm] task-space kp {env.tool_kp:.0f} N/m / {env.tool_rot_kp:.0f} N m/rad, "
              f"rotational kd {env.tool_rot_kd:.0f} N m s/rad, "
              f"joint damping {env.task_space_kd[0]:.0f} N m s/rad = {kd_mult:.2f}x what "
              f"flipup ships"
              + ("" if kd_mult >= 1.5 else
                 "  <-- below 1.5x the arm rings without settling under saturation"))
        if kd_mult > 4.0:
            print(f"[arm] WARNING: {kd_mult:.1f}x is very sluggish; at 6x the tool cannot "
                  f"follow the flip arc at all (contact duty fell to 2%, the flip failed).")
    if env.surface_force_limit > 0.0:
        print(
            f"[contact] {'table' if cube_lift else 'visible table + bookend surfaces'} "
            f"enabled; {'smooth ' if cube_lift else ''}normal anti-windup caps "
            f"steady controller deflection at "
            f"{1000.0 * env.surface_force_limit / env.tool_kp:.1f} mm "
            f"({env.surface_force_limit:.0f} N), while tangential sliding remains free"
        )
    else:
        print("[contact] surface anti-windup OFF")
    if floating:
        if env.book_force_limit > 0.0:
            print(
                f"[contact] fingertip-book normal anti-windup enabled; caps steady "
                f"controller deflection into the book at "
                f"{1000.0 * env.book_force_limit / env.tool_kp:.1f} mm "
                f"({env.book_force_limit:.0f} N), tangential sliding remains free"
            )
        else:
            print("[contact] fingertip-book normal anti-windup OFF")
    print(
        f"[scene] {task_metric_summary()}. Tool starts "
        f"{args.standoff * 100:.1f} cm from the "
        f"{'cube centre' if cube_lift else 'book edge'}, settled to "
        f"{env.settle_error * 1000:.2f} mm"
    )
    start_meta = episode_spec["start_sample"]
    print(
        f"[scene] episode attempt {episode_spec['attempt_index']} starts "
        f"{start_meta['component']} inside a "
        f"{args.start_prism[0]*100:.0f} x {args.start_prism[1]*100:.0f} x "
        f"{args.start_prism[2]*100:.0f} cm "
        f"{'x/y/z' if cube_lift else 'depth/lateral/vertical'} prism; "
        f"{args.start_center_prob*100:.0f}% of later starts are centre-biased"
    )
    arc = env.scene["waypoints"]
    if cube_lift:
        lift_span = abs(arc[-1][2] - env.scene["engage"][2])
        print(
            f"[scene] scripted pickup descends {args.standoff * 100:.1f} cm and "
            f"lifts {lift_span * 100:.1f} cm = "
            f"{args.standoff / scale[2] * 100:.1f} / "
            f"{lift_span / scale[2] * 100:.1f} cm handle travel at this --scale"
        )
        print(
            f"[workspace] sim tool xyz [{np.round(env.workspace_low, 3).tolist()}, "
            f"{np.round(env.workspace_high, 3).tolist()}]; device walls +/- "
            f"{np.round(args.workspace_wall_half, 3).tolist()} m at "
            f"{args.workspace_wall_stiffness:.0f} N/m"
        )
    else:
        print(f"[scene] the scripted flip arc spans {np.abs(arc[-1]-arc[0])[0]*100:.0f} cm of "
              f"push and {np.abs(arc[-1]-arc[0])[2]*100:.0f} cm of lift = "
              f"{np.abs(arc[-1]-arc[0])[0]/scale[0]*100:.1f} / "
              f"{np.abs(arc[-1]-arc[0])[2]/scale[2]*100:.1f} cm of handle travel at this --scale")
    view_desc = (f"camera '{args.cam_name}'" if args.cam_name else
                 f"free camera az {args.cam_azimuth:.0f} / el {args.cam_elevation:.0f} "
                 f"/ {args.cam_distance:.2f} m"
                 f"{' (over the robot shoulder)' if abs(args.cam_azimuth - 45) < 15 else ''}")
    print(f"[view] {'headless (--no-view)' if args.no_view else 'cv2 window'}, "
          f"off-screen {W}x{H} via MUJOCO_GL={os.environ.get('MUJOCO_GL')}, {view_desc}, "
          f"arm {args.arm_view}, quality '{args.render_quality}' in a background thread, capped at "
          f"{args.view_fps:.0f} fps; control {args.control_freq} Hz "
          f"({substeps} x {env.timestep*1000:.0f} ms sim step)")
    print(
        f"[timing] sim force target + numeric recording {args.control_freq} Hz; "
        f"Force Dimension servo 1000 Hz when connected; RGB/plot/UI <= "
        f"{args.view_fps:.0f} Hz "
        "on lower-priority workers"
    )
    if args.cam_name and args.cam_name not in env.camera_names():
        print(f"[view] WARNING: no camera named {args.cam_name!r}; scene has "
              f"{env.camera_names()}")
    print(f"[haptics] source '{args.force_source}', force gain {args.force_gain:.4f} N/N, "
          f"handle stiffness <= {k_handle:.0f} N/m (= tool_kp {env.tool_kp:.0f} x scale "
          f"{scale[0]:.1f} x gain {args.force_gain:.4f})")
    # On an arm the drive stiffness is an UPPER BOUND on what the hand feels: in
    # contact the series compliance is set by the contact and the object, not by
    # the task-space controller (the same caveat teleop_ball's --arm mode makes).
    # So treat the passivity number as conservative, and let --diagnose arbitrate.
    print(f"[haptics] passivity limit {k_limit:.0f} N/m  (damping {args.damping:.0f}, "
          f"effective delay {1000*t_eff:.1f} ms = {1000*ctl_dt:.1f} ms hold + "
          f"2 x {args.force_tau:.1f} ms filter)")
    print(f"[haptics] margin {k_limit/max(k_handle,1e-9):.1f}x -- teleop_ball's settled "
          f"anti-bounce tuning sits at 4.0x (1500 N/m against a 6000 N/m limit), and the "
          f"bound above is conservative for an arm, so >=4x should not bounce. "
          f"Confirm with --diagnose if it does.")
    sim_ceiling = args.force_clip * args.force_gain
    binding = "--force-clip * gain" if sim_ceiling < args.max_force else "--max-force"
    print(f"[haptics] peak renderable force {min(sim_ceiling, args.max_force):.1f} N "
          f"(sim ceiling {sim_ceiling:.1f} N vs --max-force {args.max_force:.1f} N "
          f"-> {binding} binds)")
    # Reference points measured from scripted flips, so the felt force can be
    # predicted before touching the device (teleop_ball's settled feel for
    # comparison: ~0.8 N sliding the block, ~8.7 N against the wall).
    def felt(sim_newtons):
        return min(sim_newtons * args.force_gain, args.max_force, sim_ceiling)
    # free-space reading of each source, measured over 4500 free-space samples
    free_ref = {"contact": 0.0, "wrist": 0.2, "estimated": 111.0, "none": 0.0}[
        args.force_source
    ]
    if floating and args.force_source == "wrist":
        free_ref = 0.0
    if cube_lift:
        nominal_force, heavy_force, peak_force = 6.0, 20.0, 40.0
    elif floating and np.isclose(env.tool_kp, 5000.0):
        # Default floating run: contact distribution was 15.3 N median,
        # 39.7 N p95 and 68.2 N maximum. Unlike the arm reference below, this
        # does not include force manufactured by a 16 kN/m arm controller.
        nominal_force, heavy_force, peak_force = 15.0, 40.0, 68.0
    else:
        nominal_force, heavy_force, peak_force = 30.0, 82.0, 157.0
    print(f"[haptics] expected feel: free space {felt(free_ref):.2f} N, "
          f"{'light object/table contact' if cube_lift else 'levering the book'} "
          f"{felt(nominal_force):.2f} N (sim {nominal_force:.0f} N median), "
          f"heavy contact {felt(heavy_force):.2f} N "
          f"(sim {heavy_force:.0f} N p95), contact peak "
          f"{felt(peak_force):.2f} N (sim {peak_force:.0f} N)")
    print(f"[haptics] contact onset softened by --force-rate {args.force_rate:.0f} N/s; "
          f"hand tremor of +/-0.5 mm renders as about "
          f"{0.84 * k_handle / 1500.0:.2f} N peak-to-peak (measured 0.84 N at 1500 N/m)")
    if k_handle > k_limit:
        print(f"[haptics] WARNING: {k_handle/k_limit:.1f}x over the limit -- the loop "
              f"will buzz/bounce on contact. Lower --stiffness or --scale, or raise "
              f"--damping.")
    if args.force_source == "estimated":
        print(f"[haptics] WARNING: 'estimated' reads ~111 N in FREE SPACE on this arm "
              f"(= {111*args.force_gain:.1f} N at the handle): it renders the arm's 3-7 mm "
              f"tracking lag times tool_kp, not contact. Measured cos to the true contact "
              f"force is only +0.77. Use --force-source contact unless you are "
              f"deliberately comparing.")
    elif args.force_source == "wrist":
        if floating and env.force_sensor_parameters["enabled"]:
            print(
                "[haptics] WARNING: --force-source wrist reflects the delayed "
                "modeled sensor. The default 'contact' source stays raw for "
                "passivity; lower --stiffness and verify with --diagnose if "
                "you intentionally close the loop through the sensor model"
            )
        elif args.force_deadband <= 0.0:
            print(f"[haptics] note: 'wrist' is ~0 in free space (p90 0.2 N) but spikes when "
                  f"the arm accelerates hard, because the sensor also reads the gripper's "
                  f"inertia. --force-deadband 1 removes that.")
    if args.enable_rotation:
        print(f"[axes] rotation ON: map {args.rot_axes or args.axes}, frame "
              f"'{args.rot_frame}', scale {args.rot_scale:.2f}, deadzone "
              f"{args.rot_deadzone:.3f} rad; tool gains kp/kd "
              f"{env.tool_rot_kp:.0f}/{env.tool_rot_kd:.0f}. No torque feedback "
              f"(passive omega wrist), so orientation is open-loop.")
    else:
        if cube_lift:
            print("[axes] rotation OFF: gripper stays vertical for a fast top-down grasp. "
                  "Pass --enable-rotation only when you need wrist variation.")
        else:
            print(f"[axes] rotation OFF: wrist follows the heuristic's rule (30 deg down, "
                  f"yawed away from the base). Pass --enable-rotation for 6 DoF.")
    if cube_lift:
        print(
            f"[axes] device->sim mapping {args.axes}: xyz is absolute about raised "
            "--home; squeeze the omega.7 to close the WSG50, descend onto the "
            "cube, then lift"
        )
    else:
        print(f"[axes] device->sim mapping {args.axes}: push the handle "
              f"{'AWAY from' if args.axes.split(',')[0].startswith('-') else 'TOWARD'} you "
              f"to drive the tool into the bookend, and "
              f"{'UP' if not args.axes.split(',')[2].startswith('-') else 'DOWN'} to lever "
              f"the book over")
    if recorder is not None:
        print(
            f"[dataset] {args.dataset_hz:g} Hz state/wrench + asynchronous RGB Pyrite Zarr -> "
            f"{recorder.dataset_path} ({args.dataset_image_size[0]}x"
            f"{args.dataset_image_size[1]} RGB"
            f"{' placeholders' if args.dataset_no_rgb else ''}); "
            f"{'centering pull guides' if args.collection_recenter_stiffness > 0.0 else 'return'} "
            f"handle within {args.collection_home_tolerance * 1000.0:.0f} mm "
            f"of --home and hold still for {args.collection_home_dwell_ms:.0f} ms, "
            "then S starts/stops; click KEEP/DELETE (or K/D)"
            + ("; auto-finish ON" if args.auto_finish else "")
        )
        if args.collection_recenter_stiffness > 0.0:
            print(
                f"[dataset] idle auto-centering: "
                f"{args.collection_recenter_stiffness:.0f} N/m, capped at "
                f"{args.collection_recenter_max_force:.1f} N; relax your grip while "
                "it settles, then hold normally after pressing S"
            )

    import cv2
    from collections import deque

    # ---- viewer: force strip chart + per-axis panel, as in teleop_ball ------
    PLOT_H = 190
    SIM_PLOT_H = 82
    SIDE_W = 210
    trace_felt = deque(maxlen=W)
    trace_raw = deque(maxlen=W)  # sim force x haptic gain, in handle N
    trace_sim_mag = deque(maxlen=W)  # true MuJoCo contact magnitude, in sim N
    trace_sim_xyz = [deque(maxlen=W) for _ in range(3)]  # true world-frame sim N
    trace_sensor_mag = deque(maxlen=W)  # modeled sensor magnitude, in sim N
    trace_sensor_xyz = [deque(maxlen=W) for _ in range(3)]  # modeled world F
    trace_xyz = [deque(maxlen=SIDE_W) for _ in range(3)]      # FELT (sent to the device)
    trace_xyz_raw = [deque(maxlen=SIDE_W) for _ in range(3)]  # sim force x gain
    plot_lock = threading.Lock()
    sensor_model_enabled = bool(
        floating and env.force_sensor_parameters["enabled"]
    )

    def draw_plot(frame):
        if args.no_plot or not trace_felt:
            return frame
        h, w = frame.shape[:2]
        top = h - PLOT_H
        sim_bottom = top + SIM_PLOT_H
        frame[top:, :] = (frame[top:, :].astype(np.float32) * 0.25).astype(np.uint8)

        # Upper panel: exact solver contact and the causal sensor observation.
        # Both use a SIM-NEWTON scale; haptic gain is shown only below.
        if args.plot_fixed_scale:
            sim_fmax = max(args.force_clip, 1.0)
        else:
            sim_peak = max(
                max(trace_sim_mag, default=0.0),
                max(trace_sensor_mag, default=0.0),
            )
            sim_fmax = max(5.0, 1.25 * sim_peak)
        sim_graph_top = top + 19
        sim_graph_bottom = sim_bottom - 4
        sim_mid = (sim_graph_top + sim_graph_bottom) // 2
        sim_amp = max(1, (sim_graph_bottom - sim_graph_top) // 2)
        cv2.line(frame, (0, sim_mid), (w, sim_mid), (75, 75, 75), 1)
        cv2.putText(frame, f"+{sim_fmax:.0f}", (4, sim_graph_top + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (145, 145, 145), 1, cv2.LINE_AA)
        cv2.putText(frame, "0", (4, sim_mid - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.33, (145, 145, 145), 1, cv2.LINE_AA)
        cv2.putText(frame, f"-{sim_fmax:.0f}", (4, sim_graph_bottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (145, 145, 145), 1, cv2.LINE_AA)

        def signed_poly(trace, scale_value, colour, thick, y_mid, amplitude):
            n = len(trace)
            if n < 2:
                return
            xs = np.arange(w - n, w)
            ys = y_mid - np.clip(
                np.asarray(trace) / max(scale_value, 1e-9), -1, 1
            ) * amplitude
            cv2.polylines(
                frame,
                [np.stack([xs, ys.astype(np.int32)], 1).astype(np.int32)],
                False,
                colour,
                thick,
                cv2.LINE_AA,
            )

        for axis in range(3):
            signed_poly(
                trace_sim_xyz[axis], sim_fmax,
                (
                    tuple(int(0.42 * c) for c in AXIS_COLOURS[axis])
                    if sensor_model_enabled
                    else AXIS_COLOURS[axis]
                ),
                1,
                sim_mid, sim_amp,
            )
            if sensor_model_enabled:
                signed_poly(
                    trace_sensor_xyz[axis], sim_fmax, AXIS_COLOURS[axis], 2,
                    sim_mid, sim_amp,
                )
        sim_now = [trace_sim_xyz[k][-1] if trace_sim_xyz[k] else 0.0 for k in range(3)]
        sim_mag_now = trace_sim_mag[-1] if trace_sim_mag else 0.0
        sensor_now = [
            trace_sensor_xyz[k][-1] if trace_sensor_xyz[k] else 0.0
            for k in range(3)
        ]
        sensor_mag_now = trace_sensor_mag[-1] if trace_sensor_mag else 0.0
        if sensor_model_enabled:
            force_title = (
                f"FORCE [N] raw thin |F| {sim_mag_now:.1f} / "
                f"sensor {args.force_sensor_cutoff:.0f}Hz thick "
                f"[{sensor_now[0]:+.1f} {sensor_now[1]:+.1f} "
                f"{sensor_now[2]:+.1f}] |F| {sensor_mag_now:.1f}"
            )
        else:
            force_title = (
                f"ACTUAL MuJoCo world force [N]  "
                f"x {sim_now[0]:+.1f}  y {sim_now[1]:+.1f}  "
                f"z {sim_now[2]:+.1f}  |F| {sim_mag_now:.1f}"
            )
        cv2.putText(
            frame,
            force_title,
            (42, top + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Lower panel: force requested/rendered at the haptic handle, in handle N.
        if args.plot_fixed_scale:
            fmax = max(args.max_force, 1.0)
        else:
            peak = max(max(trace_felt, default=0.0), max(trace_raw, default=0.0))
            fmax = min(max(1.0, 1.25 * peak), max(args.max_force, 1.0))
        handle_graph_top = sim_bottom + 19
        handle_graph_bottom = h - 1
        for frac in (0.0, 0.5, 1.0):
            y = int(handle_graph_bottom - frac * (
                handle_graph_bottom - handle_graph_top
            ))
            cv2.line(frame, (0, y), (w, y), (70, 70, 70), 1)
            cv2.putText(frame, f"{frac*fmax:.0f}N", (4, max(y - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

        def poly(trace, colour, thick):
            n = len(trace)
            if n < 2:
                return
            xs = np.arange(w - n, w)
            ys = handle_graph_bottom - np.clip(
                np.asarray(trace) / fmax, 0, 1
            ) * (handle_graph_bottom - handle_graph_top)
            cv2.polylines(frame, [np.stack([xs, ys.astype(np.int32)], 1).astype(np.int32)],
                          False, colour, thick, cv2.LINE_AA)

        poly(trace_raw, (90, 90, 220), 1)      # sim force x gain, red-ish
        poly(trace_felt, (90, 220, 120), 2)    # what the handle is commanded, green
        cv2.putText(frame, f"HANDLE |F|: felt green / sim x gain red   {args.plot_span:.0f}s   "
                           f"src={args.force_source} gain={args.force_gain:.3f} "
                           f"tau={args.force_tau:.0f}ms",
                    (42, sim_bottom + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (200, 200, 200), 1,
                    cv2.LINE_AA)
        return frame

    AXIS_COLOURS = ((90, 90, 230), (110, 220, 110), (240, 170, 80))   # x,y,z in BGR

    def draw_side(canvas):
        """Signed per-axis traces of the commanded force, stacked on the right."""
        h = canvas.shape[0]
        x0 = W
        cv2.rectangle(canvas, (x0, 0), (canvas.shape[1], h), (24, 24, 24), -1)
        cell = h // 3
        amp = cell // 2 - 18
        for k, name in enumerate("xyz"):
            top = k * cell
            mid = top + cell // 2 + 6
            if k:
                cv2.line(canvas, (x0, top), (canvas.shape[1], top), (60, 60, 60), 1)
            cv2.line(canvas, (x0, mid), (canvas.shape[1], mid), (75, 75, 75), 1)
            tr, tr_raw = trace_xyz[k], trace_xyz_raw[k]
            if args.plot_fixed_scale:
                fmax = max(args.max_force, 1.0)
            else:
                peak = max(max((abs(v) for v in tr), default=0.0),
                           max((abs(v) for v in tr_raw), default=0.0))
                fmax = min(max(0.5, 1.25 * peak), max(args.max_force, 1.0))

            def _line(data, colour, thick):
                if len(data) < 2:
                    return
                xs = np.arange(canvas.shape[1] - len(data), canvas.shape[1])
                ys = mid - np.clip(np.asarray(data) / fmax, -1, 1) * amp
                cv2.polylines(canvas, [np.stack([xs, ys.astype(np.int32)], 1).astype(np.int32)],
                              False, colour, thick, cv2.LINE_AA)

            _line(tr_raw, tuple(int(0.45 * c) for c in AXIS_COLOURS[k]), 1)
            _line(tr, AXIS_COLOURS[k], 2)
            now = tr[-1] if tr else 0.0
            sim_now = trace_sim_xyz[k][-1] if trace_sim_xyz[k] else 0.0
            sensor_now = (
                trace_sensor_xyz[k][-1] if trace_sensor_xyz[k] else sim_now
            )
            cv2.putText(canvas,
                        (
                            f"R/S F{name} {sim_now:+4.1f}/{sensor_now:+4.1f}N "
                            f"dev {now:+4.2f}N"
                            if sensor_model_enabled
                            else f"world F{name} {sim_now:+5.1f}N | dev {now:+4.2f}N"
                        ),
                        (x0 + 6, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33,
                        AXIS_COLOURS[k], 1, cv2.LINE_AA)
        return canvas

    # ---- video (see teleop_ball.py for the three bugs this shape avoids) ----
    vid = {"w": None, "t0": None, "written": 0, "kind": None}

    def _open_writer(w, h):
        """Pipe raw frames to our own ffmpeg started with start_new_session=True.

        A terminal Ctrl-C signals the whole process GROUP, so an ffmpeg child in
        the same group dies before writing the moov atom and the file fails with
        "moov atom not found" even though the frames are in it. Its own session
        shields it until we close its stdin.
        """
        try:
            import subprocess
            try:
                import imageio_ffmpeg
                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                exe = "ffmpeg"
            common = [exe, "-y", "-loglevel", "error",
                      "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                      "-r", str(args.video_fps), "-i", "-", "-an"]
            if args.record_video.lower().endswith(".webm"):
                enc = ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8",
                       "-row-mt", "1", "-b:v", "2M", "-pix_fmt", "yuv420p"]
                codec = "vp9/webm"
            else:
                enc = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                       "-crf", "20", "-movflags", "+faststart"]
                codec = "h264/mp4"
            proc = subprocess.Popen(common + enc + [args.record_video],
                                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, start_new_session=True)
            return proc, codec
        except Exception as exc:
            print(f"[video] ffmpeg unavailable ({exc}); falling back to cv2 mp4v, which "
                  f"some players cannot open")
            wr = cv2.VideoWriter(args.record_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(args.video_fps), (w, h))
            return (wr, "mp4v") if wr.isOpened() else (None, None)

    def record_frame(canvas):
        if not args.record_video:
            return
        if canvas.shape[0] % 2 or canvas.shape[1] % 2:
            canvas = canvas[: canvas.shape[0] // 2 * 2, : canvas.shape[1] // 2 * 2]
        now = time.time()
        if vid["w"] is None:
            h, w = canvas.shape[:2]
            wr, kind = _open_writer(w, h)
            if wr is None:
                print(f"[video] could not open {args.record_video} for writing")
                args.record_video = None
                return
            vid["w"], vid["kind"], vid["t0"] = wr, kind, now
            print(f"[video] {w}x{h} @ {args.video_fps} fps ({kind}) -> "
                  f"{os.path.abspath(args.record_video)}")
        want = int((now - vid["t0"]) * args.video_fps) + 1
        while vid["written"] < want:
            if vid["kind"] != "mp4v":      # our own ffmpeg pipe
                try:
                    vid["w"].stdin.write(canvas.tobytes())
                except (BrokenPipeError, ValueError):
                    print("[video] ffmpeg pipe closed early; stopping recording")
                    args.record_video = None
                    return
            else:
                vid["w"].write(canvas)
            vid["written"] += 1

    # ---- render thread ----------------------------------------------------
    # Rendering this scene costs 9-75 ms. Doing it inline would freeze the force
    # loop for that long every frame, so it runs alongside: it only reads
    # model/data, and a torn frame is harmless. Soak-tested over 145k steps.
    rendering = (
        (not args.no_view)
        or bool(args.record_video)
        or (recorder is not None and not args.dataset_no_rgb)
    )
    shot = {
        "frame": None,
        "sim_time_s": None,
        "n": 0,
        "run": True,
        "err": None,
    }
    if rendering:
        render_fn = env.make_camera(width=W, height=H, quality=args.render_quality,
                                   azimuth=args.cam_azimuth,
                                   elevation=args.cam_elevation,
                                   distance=args.cam_distance,
                                   camera=args.cam_name)

        # RGB is intentionally asynchronous from the 1 kHz state stream.  Do
        # not drive the renderer at --dataset-hz; that would create duplicate
        # images and steal the control loop's real-time budget.
        render_fps = args.view_fps
        view_period = 1.0 / render_fps if render_fps > 0 else 0.0

        def render_loop():
            lower_background_thread_priority(10)
            try:
                while shot["run"]:
                    t_frame = time.time()
                    with render_model_lock:
                        shot["frame"] = render_fn()
                        shot["sim_time_s"] = float(env.data.time)
                    shot["n"] += 1
                    idle = view_period - (time.time() - t_frame)
                    if idle > 0:
                        time.sleep(idle)
            except Exception as exc:   # keep the haptic loop alive if rendering dies
                shot["err"] = repr(exc)

        render_thread = threading.Thread(target=render_loop, daemon=True)
        render_thread.start()
    else:
        render_thread = None

    # The viewer starts before the device/control loop below and can draw its first
    # frame immediately.  Initialize the shared wrist display state before that
    # thread exists; otherwise --enable-rotation races with the later assignment.
    rot_home = [None]          # device wrist frame at start / after reset
    wrist_delta = [np.zeros(3)]

    shown = [0, -1]      # [frames shown, index of the last frame drawn]
    last_display_monotonic = [0.0]
    display_period = 1.0 / (args.view_fps if args.view_fps > 0.0 else 60.0)
    button_y0, button_y1 = ((112, 146) if cube_lift else (48, 82))
    button_gap = 8
    button_w = max(82, min(112, (W - 3 * button_gap) // 2))
    keep_rect = (button_gap, button_y0, button_gap + button_w, button_y1)
    delete_rect = (
        2 * button_gap + button_w,
        button_y0,
        2 * (button_gap + button_w),
        button_y1,
    )

    def draw_state(frame):
        """Task metric, collection state, and optional tool orientation."""
        metric = task_metric_value()
        done = env.success()
        state_text = (
            f"cube clear {100.0 * metric:4.1f}/{100.0 * args.success_height:.1f} cm  "
            f"jaw {1000.0 * env.gripper_opening:4.1f} mm"
            if cube_lift
            else f"book {metric:5.1f} deg from vertical  (need < 15)"
        )
        cv2.putText(frame, state_text,
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (120, 230, 140) if done else (230, 230, 230), 1, cv2.LINE_AA)
        if cube_lift:
            grasp_exceeded = (
                force_monitor["grasp_max_n"] > args.grasp_force_limit + 1e-6
            )
            table_exceeded = (
                force_monitor["table_max_n"] > args.surface_force_limit + 1e-6
            )
            grasp_text = (
                f"GRASP now {force_monitor['grasp_current_n']:5.1f} N  "
                f"MAX {force_monitor['grasp_max_n']:5.1f} N / "
                f"LIMIT {args.grasp_force_limit:.1f} N  "
                f"{'EXCEEDED' if grasp_exceeded else 'OK'}"
            )
            table_text = (
                f"TABLE now {force_monitor['table_current_n']:5.1f} N  "
                f"MAX {force_monitor['table_max_n']:5.1f} N / "
                f"LIMIT {args.surface_force_limit:.1f} N  "
                f"{'EXCEEDED' if table_exceeded else 'OK'}"
            )
            cv2.putText(
                frame, grasp_text, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (70, 70, 245) if grasp_exceeded else (120, 230, 140),
                1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, table_text, (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (70, 70, 245) if table_exceeded else (120, 230, 140),
                1, cv2.LINE_AA,
            )
        if done:
            cv2.putText(frame, "SUCCESS", (8, 84 if cube_lift else 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (120, 230, 140), 2, cv2.LINE_AA)
        if recorder is not None:
            status = collection["state"]
            if status == "idle":
                if collection_gate["ready"]:
                    text = "HANDLE READY - press S to start episode"
                    colour = (100, 230, 120)
                else:
                    text = (
                        "RELAX GRIP - HANDLE AUTO-CENTERING"
                        if args.collection_recenter_stiffness > 0.0
                        else "RETURN HANDLE TO FIXED HOME"
                    )
                    colour = (80, 220, 240)
            elif status == "recording":
                text = (
                    f"REC {recorder.sample_count / args.dataset_hz:5.1f}s "
                    f"({recorder.sample_count} samples) - S to stop"
                )
                colour = (80, 80, 245)
                cv2.circle(frame, (W - 17, 17), 7, colour, -1, cv2.LINE_AA)
            else:
                text = "EPISODE FINISHED - choose KEEP or DELETE"
                colour = (80, 220, 240)
            if cube_lift:
                text_y = 96 if status == "review" else (104 if done else 84)
            else:
                text_y = 102 if status == "review" else (62 if done else 42)
            cv2.putText(
                frame,
                text,
                (8, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                colour,
                1,
                cv2.LINE_AA,
            )
            if status == "idle":
                speed_text = (
                    f"distance {collection_gate['distance_m'] * 1000.0:4.1f} mm "
                    f"(need <= {args.collection_home_tolerance * 1000.0:.1f}), "
                    f"speed {collection_gate['speed_m_s'] * 1000.0:4.1f} mm/s"
                    if collection_gate["velocity_valid"]
                    else "waiting for device velocity sample"
                )
                if cube_lift and not collection_gate.get("gripper_ready", False):
                    gap_mm = 1000.0 * collection_gate.get("gripper_gap_m", 0.0)
                    if collection_gate.get("gripper_calibrated", False):
                        ref_mm = 1000.0 * collection_gate.get(
                            "gripper_open_reference_m", args.device_grip_open
                        )
                        speed_text += (
                            f"  |  relax gripper: {gap_mm:.1f}/{ref_mm:.1f} mm"
                        )
                    else:
                        speed_text += (
                            f"  |  relax gripper: calibrating open at {gap_mm:.1f} mm"
                        )
                cv2.putText(
                    frame,
                    speed_text,
                    (8, text_y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    colour,
                    1,
                    cv2.LINE_AA,
                )
            if status == "review":
                x0, y0, x1, y1 = keep_rect
                cv2.rectangle(frame, (x0, y0), (x1, y1), (65, 150, 65), -1)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 240, 120), 2)
                cv2.putText(
                    frame,
                    "KEEP [K]",
                    (x0 + 11, y0 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
                x0, y0, x1, y1 = delete_rect
                cv2.rectangle(frame, (x0, y0), (x1, y1), (65, 65, 165), -1)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (100, 100, 245), 2)
                cv2.putText(
                    frame,
                    "DELETE [D]",
                    (x0 + 7, y0 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
        if args.enable_rotation:
            from scipy.spatial.transform import Rotation
            rpy = Rotation.from_quat(env.tool_quat[[1, 2, 3, 0]]).as_euler(
                "xyz", degrees=True)
            if recorder is not None:
                rotation_text_y = 126 if collection["state"] == "review" else (
                    82 if done else 62
                )
            else:
                rotation_text_y = 62 if done else 42
            cv2.putText(frame, f"tool rpy {rpy[0]:+6.1f} {rpy[1]:+6.1f} {rpy[2]:+6.1f}",
                        (8, rotation_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 120), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"wrist delta {np.degrees(np.linalg.norm(wrist_delta[0])):5.1f} deg",
                (8, rotation_text_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 120),
                1,
                cv2.LINE_AA,
            )
        return frame

    def show():
        img = shot["frame"]
        if img is None:
            return 255
        frame = np.ascontiguousarray(img[:, :, ::-1])       # RGB -> BGR
        frame = draw_state(frame)
        if args.no_plot:
            canvas = frame
        else:
            with plot_lock:
                frame = draw_plot(frame)
                canvas = np.zeros((H, W + SIDE_W, 3), dtype=np.uint8)
                canvas[:, :W] = frame
                canvas = draw_side(canvas)
        record_frame(canvas)
        shown[0] += 1
        shown[1] = shot["n"]
        last_display_monotonic[0] = time.monotonic()
        if args.no_view:
            return 255
        if np.isclose(args.viewer_scale, 1.0):
            display_canvas = canvas
        else:
            display_canvas = cv2.resize(
                canvas,
                None,
                fx=args.viewer_scale,
                fy=args.viewer_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        cv2.imshow("Force Dimension -- FlipUp", display_canvas)
        return cv2.waitKey(1) & 0xFF

    def display_due():
        """Refresh force/UI overlays even while the RGB producer is late."""
        return (
            shot["frame"] is not None
            and (
                shot["n"] != shown[1]
                or time.monotonic() - last_display_monotonic[0] >= display_period
            )
        )

    def on_view_mouse(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONUP or collection["state"] != "review":
            return
        # Button rectangles live in the native canvas coordinate system; the
        # displayed window may be enlarged independently.
        x = int(x / args.viewer_scale)
        y = int(y / args.viewer_scale)
        kx0, ky0, kx1, ky1 = keep_rect
        dx0, dy0, dx1, dy1 = delete_rect
        if kx0 <= x <= kx1 and ky0 <= y <= ky1:
            review_action[0] = "keep"
        elif dx0 <= x <= dx1 and dy0 <= y <= dy1:
            review_action[0] = "delete"

    viewer_keys = deque()
    viewer_key_lock = threading.Lock()
    viewer_state = {"run": True, "err": None}

    def pop_viewer_key():
        with viewer_key_lock:
            return viewer_keys.popleft() if viewer_keys else 255

    def viewer_loop():
        lower_background_thread_priority(5)
        window_created = False
        try:
            if not args.no_view:
                cv2.namedWindow("Force Dimension -- FlipUp", cv2.WINDOW_AUTOSIZE)
                window_created = True
                cv2.setMouseCallback("Force Dimension -- FlipUp", on_view_mouse)
            while viewer_state["run"]:
                if display_due():
                    key = show()
                    if key != 255:
                        with viewer_key_lock:
                            viewer_keys.append(key)
                else:
                    time.sleep(min(0.002, display_period / 4.0))
        except Exception as exc:
            viewer_state["err"] = repr(exc)
        finally:
            # OpenCV's Qt backend owns timers on the thread that creates the
            # window.  Destroy it here as well to avoid cross-thread Qt warnings.
            if window_created:
                try:
                    cv2.destroyWindow("Force Dimension -- FlipUp")
                except cv2.error:
                    pass

    viewer_enabled = (not args.no_view) or bool(args.record_video)
    viewer_thread = (
        threading.Thread(target=viewer_loop, daemon=True)
        if viewer_enabled
        else None
    )
    if viewer_thread is not None:
        viewer_thread.start()

    # ---- device -----------------------------------------------------------
    device = None
    if not args.dry_run:
        from fd_omega import FDOmega

        device = FDOmega(
            auto_init=args.auto_init,
            read_orientation=args.enable_rotation,
            # Collection idle mode gently returns the handle to a repeatable
            # physical origin. It is disabled before the episode begins, so
            # demonstrated free-space motion remains effortless.
            spring_k=(
                args.collection_recenter_stiffness
                if recorder is not None
                else 0.0
            ),
            spring_max_force=args.collection_recenter_max_force,
            reflected_tau_s=args.force_tau / 1000.0,
            reflected_rate=args.force_rate,
            wall_k=args.workspace_wall_stiffness,
            wall_half=(
                np.asarray(args.workspace_wall_half, dtype=float)
                if args.workspace_wall_stiffness > 0.0
                else None
            ),
            damping_b=args.damping,
            grip_damping=args.grip_damping,
            grip_tau_s=args.grip_force_tau / 1000.0,
            grip_rate_n_s=args.grip_force_rate,
            max_grip_force=args.max_grip_force,
            max_force=args.max_force,
            home_pos=np.array(args.home, dtype=float),
        ).open()
        try:
            import ctypes
            import fdsdk
            fdsdk.libdrd.dhdGetMaxForce.restype = ctypes.c_double
            dev_max = float(fdsdk.libdrd.dhdGetMaxForce(device.id))
        except Exception:
            dev_max = -1.0
        cap = dev_max if dev_max > 0 else 12.0
        note = "SDK-configured" if dev_max > 0 else "omega.6/.7 datasheet continuous rating"
        if args.max_force > cap:
            print(f">>> WARNING: --max-force {args.max_force:.0f} N exceeds {cap:.0f} N "
                  f"({note}). The device cannot sustain that; it will saturate or "
                  f"thermally limit.")
        print(f">>> {device.system_name} (serial {device.serial})"
              f"{', active gripper' if device.has_gripper else ''}")
        if cube_lift and not device.has_gripper:
            device.close()
            raise RuntimeError(
                "cube lift requires an active omega.7 gripper for analogue grasp control"
            )
        if args.enable_rotation:
            orientation_state = device.get_state()
            if not orientation_state["orientation_valid"]:
                device.close()
                raise RuntimeError("Wrist orientation was enabled but no valid frame was read")
            print(f">>> Wrist orientation sensor ready "
                  f"({orientation_state['orientation_sample_count']} initial sample)")
        if cube_lift:
            print(">>> Start above the cube, descend, squeeze the omega.7 to grasp, "
                  "then lift. Grasp and table forces are smoothly limited.")
        else:
            print(">>> Push the handle AWAY from you and UP to lever the book upright. "
                  "The WSG50 stays closed -- this is a nonprehensile pivot.")

    fixed_device_home = np.asarray(args.home, dtype=float)
    grip_open_calibrator = GripOpenCalibrator(
        closed_m=args.device_grip_closed,
        nominal_open_m=args.device_grip_open,
        auto=args.device_grip_auto_calibration and recorder is not None,
        stable_s=args.device_grip_calibration_ms / 1000.0,
        tolerance_m=args.device_grip_open_tolerance,
        minimum_span_m=args.device_grip_min_span,
    )

    def effective_device_grip_open():
        return grip_open_calibrator.effective_open_m

    def update_collection_gate(device_state, now=None):
        """Require a centered, stationary handle before accepting episode start."""
        if recorder is None or args.dry_run:
            collection_gate.update(
                {
                    "ready": True,
                    "within_since": None,
                    "distance_m": 0.0,
                    "speed_m_s": 0.0,
                    "velocity_valid": True,
                    "gripper_ready": True,
                    "gripper_gap_m": args.device_grip_open,
                    "gripper_open_reference_m": args.device_grip_open,
                    "gripper_calibrated": True,
                }
            )
            return True
        now = time.monotonic() if now is None else float(now)
        distance, speed, velocity_valid = collection_home_metrics(
            device_state, fixed_device_home
        )
        gripper_gap = float(device_state.get("gripper", 0.0))
        previous_grip_reference = grip_open_calibrator.reference_m
        gripper_ready = (
            True
            if not cube_lift
            else grip_open_calibrator.observe(gripper_gap, now)
        )
        if (
            cube_lift
            and previous_grip_reference is None
            and grip_open_calibrator.reference_m is not None
        ):
            print(
                f"\n[gripper] calibrated relaxed open gap to "
                f"{1000.0 * grip_open_calibrator.reference_m:.1f} mm "
                f"(nominal upper bound {1000.0 * args.device_grip_open:.1f} mm)"
            )
        inside = (
            velocity_valid
            and distance <= args.collection_home_tolerance
            and speed <= args.collection_home_speed
            and gripper_ready
        )
        if inside:
            if collection_gate["within_since"] is None:
                collection_gate["within_since"] = now
            dwell_s = args.collection_home_dwell_ms / 1000.0
            ready = now - collection_gate["within_since"] >= dwell_s
        else:
            collection_gate["within_since"] = None
            ready = False
        collection_gate.update(
            {
                "ready": bool(ready),
                "distance_m": distance,
                "speed_m_s": speed,
                "velocity_valid": velocity_valid,
                "gripper_ready": bool(gripper_ready),
                "gripper_gap_m": gripper_gap,
                "gripper_open_reference_m": effective_device_grip_open(),
                "gripper_calibrated": bool(
                    not cube_lift or grip_open_calibrator.reference_m is not None
                ),
            }
        )
        return bool(ready)

    def start_recorded_episode():
        nonlocal target, target_rv, commanded_rv, armed, force_filt, t_start
        if recorder is None or collection["state"] != "idle":
            return False
        if device is not None:
            current_state = device.get_state()
            if not update_collection_gate(current_state):
                print(
                    "\n[dataset] start refused: relax your grip and let the handle "
                    "settle at fixed --home "
                    f"(distance {collection_gate['distance_m'] * 1000.0:.1f} mm, "
                    f"speed {collection_gate['speed_m_s'] * 1000.0:.1f} mm/s) "
                    + (
                        (
                            "and relax the omega gripper while its open gap "
                            f"calibrates ({collection_gate['gripper_gap_m'] * 1000.0:.1f} mm) "
                            if not collection_gate.get("gripper_calibrated", False)
                            else "and relax the omega gripper back to its calibrated open gap "
                        )
                        if cube_lift and not collection_gate.get("gripper_ready", False)
                        else ""
                    )
                    + f"and hold for {args.collection_home_dwell_ms:.0f} ms"
                )
                return False
        else:
            current_state = {
                "pos": fixed_device_home.copy(),
                "vel": np.zeros(3),
                "velocity_valid": True,
                "gripper": args.device_grip_open,
            }
        # Idle collection mode has already reset and continuously holds this
        # sampled state. Do not reset again here: arm settling can take hundreds
        # of wall-clock milliseconds, during which a previously ready handle may
        # move. The randomized sim pose always maps to the same physical --home.
        target = env.tool_home.copy()
        target_rv = commanded_rv = None
        armed = False
        if device is not None:
            device.set_centering_enabled(False)
        reset_haptic_feedback()
        rot_home[0] = None
        wrist_delta[0] = np.zeros(3)
        reset_force_monitor()
        recorder.start_episode(
            {
                "seed": args.seed,
                "episode_attempt": episode_spec,
                "physical_properties": dict(properties.__dict__),
                "task_kind": task_kind,
                "task_metric_name": task_metric_name,
                "book_color_rgba": env.book_color,
                "object_color_rgba": env.book_color,
                "initial_tool_position_world_m": env.tool_home,
                "scene": env.scene,
                "controller": {
                    "tool_kp": env.tool_kp,
                    "tool_rot_kp": env.tool_rot_kp,
                    "tool_rot_kd": env.tool_rot_kd,
                    "joint_kd": env.task_space_kd,
                    "cartesian_kd": env.task_space_cartesian_kd,
                    "control_freq_hz": args.control_freq,
                    "sim_timestep_s": env.timestep,
                    "substeps": substeps,
                    "surface_force_limit": env.surface_force_limit,
                    "book_normal_force_limit": getattr(env, "book_force_limit", None),
                    "workspace_low": getattr(env, "workspace_low", None),
                    "workspace_high": getattr(env, "workspace_high", None),
                    "grasp_force_limit": getattr(env, "grasp_force_limit", None),
                    "gripper_speed": getattr(env, "gripper_speed", None),
                    "success_height": getattr(env, "success_height", None),
                    "approach_compliance": {
                        "enabled": getattr(env, "_approach_compliance_enabled", False),
                        "distance_m": getattr(
                            env, "approach_compliance_distance_m", 0.0
                        ),
                        "min_kp_ratio": getattr(
                            env, "approach_compliance_min_kp_ratio", None
                        ),
                        "max_speed_mps": getattr(env, "approach_max_speed_mps", 0.0),
                    },
                },
                "mapping": {
                    "position_matrix": pos_map,
                    "rotation_matrix": rot_map,
                    "scale": scale,
                    "home": home.copy(),
                    "start_handle_position": np.asarray(
                        current_state["pos"], dtype=float
                    ),
                    "start_handle_velocity": np.asarray(
                        current_state.get("vel", np.zeros(3)), dtype=float
                    ),
                    "start_home_error_m": collection_gate["distance_m"],
                    "start_handle_speed_m_s": collection_gate["speed_m_s"],
                    "rotation_enabled": args.enable_rotation,
                    "rotation_frame": args.rot_frame,
                    "rotation_scale": args.rot_scale,
                    "rotation_deadzone": args.rot_deadzone,
                    "max_speed": args.max_speed,
                    "max_rotation_speed_deg_s": args.max_rot_speed,
                    "device_grip_closed_m": args.device_grip_closed,
                    "device_grip_open_m": effective_device_grip_open(),
                    "device_grip_open_nominal_m": args.device_grip_open,
                    "device_grip_auto_calibration": args.device_grip_auto_calibration,
                },
                "camera": {
                    "resolution": args.cam_res,
                    "azimuth": args.cam_azimuth,
                    "elevation": args.cam_elevation,
                    "distance": args.cam_distance,
                    "name": args.cam_name,
                    "arm_view": args.arm_view,
                    "render_quality": args.render_quality,
                },
                "haptics": {
                    "force_source": args.force_source,
                    "stiffness_n_m": args.stiffness,
                    "force_gain": args.force_gain,
                    "force_clip": args.force_clip,
                    "max_force": args.max_force,
                    "filter_tau_ms": args.force_tau,
                    "force_rate_n_s": args.force_rate,
                    "damping": args.damping,
                    "grip_force_gain": args.grip_force_gain,
                    "grip_feedback_sign": args.grip_feedback_sign,
                    "grip_filter_tau_ms": args.grip_force_tau,
                    "grip_force_rate_n_s": args.grip_force_rate,
                    "max_grip_force_n": args.max_grip_force,
                    "grip_damping": args.grip_damping,
                    "workspace_wall_stiffness_n_m": args.workspace_wall_stiffness,
                    "workspace_wall_half_m": args.workspace_wall_half,
                    "collection_home_tolerance_m": args.collection_home_tolerance,
                    "collection_home_speed_m_s": args.collection_home_speed,
                    "collection_recenter_stiffness_n_m": (
                        args.collection_recenter_stiffness
                    ),
                    "collection_recenter_max_force_n": (
                        args.collection_recenter_max_force
                    ),
                    "collection_home_dwell_ms": args.collection_home_dwell_ms,
                    "collection_takeover_hold_ms": args.collection_takeover_hold_ms,
                    "collection_force_ramp_ms": args.collection_force_ramp_ms,
                },
                "model": {
                    "controller_kind": env.controller_kind,
                    "nq": env.model.nq,
                    "nv": env.model.nv,
                    "nu": env.model.nu,
                    "na": env.model.na,
                    "nsensordata": env.model.nsensordata,
                    "tip_contact": getattr(env, "tip_contact_parameters", None),
                    "table_contact": getattr(env, "table_contact_parameters", None),
                    "force_sensor": getattr(env, "force_sensor_parameters", None),
                },
                "device": (
                    {"kind": "dry_run"}
                    if device is None
                    else {
                        "system_name": device.system_name,
                        "serial": device.serial,
                        "has_gripper": device.has_gripper,
                        "has_wrist": device.has_wrist,
                    }
                ),
                "command_line": vars(args),
            }
        )
        collection.update(
            {
                "state": "recording",
                "reason": None,
                "success": False,
                "final_book_angle_deg": None,
                "final_task_metric": None,
                "started_monotonic": None,
                "started_sim_time_s": float(env.data.time),
            }
        )
        episode[1] = step
        last_dataset_frame_id[0] = int(shot["n"])
        recorder.record_sample(
            env,
            timestamp_ms=0.0,
            target_pos=target,
            target_rotvec=target_rv,
            device_state=(device.get_state() if device is not None else state),
            sent_force=np.zeros(3),
            image_rgb=None,
            image_capture_time_s=None,
            image_id=None,
            wall_time_ns=time.perf_counter_ns(),
        )
        # Sample-zero allocation is deliberately outside live scheduling.
        collection["started_monotonic"] = time.monotonic()
        t_start = time.time() - step / args.control_freq
        print(
            f"\n[dataset] recording started -> {recorder.dataset_path} "
            f"from sampled pose {np.round(env.tool_home, 4).tolist()} (press S to stop)"
        )
        return True

    def stop_recorded_episode(reason):
        """Freeze an in-memory episode for an explicit keep/delete decision."""
        if (
            recorder is None
            or collection["state"] != "recording"
            or not recorder.active
        ):
            return False
        final_metric = task_metric_value()
        collection.update(
            {
                "state": "review",
                "reason": str(reason),
                "success": bool(env.success()),
                "final_book_angle_deg": float(env.book_angle_deg()),
                "final_task_metric": final_metric,
            }
        )
        reset_haptic_feedback()
        print(
            f"\n[dataset] episode stopped: {recorder.sample_count} samples, "
            f"success={collection['success']}, "
            f"{task_metric_summary(collection['final_task_metric'])}"
        )
        if cube_lift:
            print(
                f"[forces] grasp max {force_monitor['grasp_max_n']:.2f} / "
                f"{args.grasp_force_limit:.2f} N "
                f"({'EXCEEDED' if force_monitor['grasp_max_n'] > args.grasp_force_limit + 1e-6 else 'OK'}); "
                f"table max {force_monitor['table_max_n']:.2f} / "
                f"{args.surface_force_limit:.2f} N "
                f"({'EXCEEDED' if force_monitor['table_max_n'] > args.surface_force_limit + 1e-6 else 'OK'})"
            )
        print("[dataset] click KEEP or DELETE in the viewer (keyboard: K or D)")
        return True

    def finish_recorded_episode(save):
        """Resolve the episode currently held in the review state."""
        if recorder is None or not recorder.active:
            return False
        count = recorder.sample_count
        if save:
            name = recorder.commit(
                success=bool(collection["success"]),
                termination_reason=str(collection["reason"]),
                final_book_angle_deg=float(collection["final_book_angle_deg"]),
                final_task_metric_name=task_metric_name,
                final_task_metric_value=float(collection["final_task_metric"]),
            )
            if name is None:
                print(
                    f"\n[dataset] discarded {count} samples; minimum is "
                    f"{args.dataset_min_samples}"
                )
            else:
                print(
                    f"\n[dataset] saved {name}: {count} samples, "
                    f"success={collection['success']}, "
                    f"{task_metric_summary(collection['final_task_metric'])}"
                )
        else:
            discarded = recorder.discard()
            print(f"\n[dataset] deleted episode ({discarded} unsaved samples)")
        collection.update(
            {
                "state": "idle",
                "reason": None,
                "success": False,
                "final_book_angle_deg": None,
                "final_task_metric": None,
                "started_monotonic": None,
                "started_sim_time_s": None,
            }
        )
        return True

    # ---- loop -------------------------------------------------------------
    diag = {"f": [], "h": [], "c": []}
    rec_rows = []
    episode = [0, 0]
    force_filt = np.zeros(3)
    sent = [np.zeros(3)]

    def reset_haptic_feedback():
        """Hard-clear both control-loop and device-loop force history."""
        nonlocal force_filt
        force_filt = np.zeros(3)
        sent[0] = np.zeros(3)
        if device is not None:
            device.clear_reflected_force()
            device.clear_grip_force()

    def reflect(f):
        """Clamp and publish a handle-force target.

        Hardware filtering and slew limiting run in FDOmega's independently
        scheduled servo, using measured wall-clock dt. Dry runs retain the same
        filter locally so their recorded haptic channel remains representative.
        """
        nonlocal force_filt
        f = np.asarray(f, dtype=float)
        mag = np.linalg.norm(f)
        if mag > args.max_force:
            f = f * (args.max_force / mag)
        if device is not None:
            device.set_reflected_force(f)
            return
        if force_alpha < 1.0:
            force_filt += force_alpha * (f - force_filt)
            f = force_filt.copy()
        else:
            force_filt = f.copy()
        if args.force_rate > 0.0:
            # Slew-limit the whole VECTOR toward the new command, so a direction
            # change is limited the same way a magnitude change is.
            step_v = f - sent[0]
            step_mag = np.linalg.norm(step_v)
            budget = args.force_rate * period
            if step_mag > budget:
                f = sent[0] + step_v * (budget / step_mag)
        mag = np.linalg.norm(f)
        if mag > args.max_force:
            f = f * (args.max_force / mag)
        sent[0] = f

    # Wrist rotation -> absolute world rotvec for the tool. Convention copied from
    # teleop_ball.py, including the bug it documents: the delta convention and the
    # composition order MUST match, because a body-frame delta composed
    # extrinsically mixes two frames and the axes come out wrong.
    #   world frame: spatial delta  R_dev R_home^T , pre-multiplied
    #   tool  frame: body    delta  R_home^T R_dev , post-multiplied
    def orientation_command(state):
        R_dev = np.asarray(state["rot"], dtype=float).reshape(3, 3)
        if rot_home[0] is None:
            rot_home[0] = R_dev.copy()
        command, delta = map_wrist_orientation(
            R_dev,
            rot_home[0],
            rot_map,
            env.home_rotvec,
            frame=args.rot_frame,
            scale=args.rot_scale,
            deadzone=args.rot_deadzone,
        )
        wrist_delta[0] = delta
        return command

    home = fixed_device_home.copy()
    target = env.tool_home.copy()
    target_rv = None           # tool rotvec actually commanded (None = derived from pos)
    commanded_rv = None        # what the wrist asks for, before the slew limit
    armed = False
    last_long = 0 if device is None else device.get_state()["long_press_count"]
    step = 0
    t_start = time.time()
    period = 1.0 / args.control_freq
    MAX_CATCHUP = 16
    state = {"pos": home.copy(), "gripper": args.device_grip_open}
    gripper_command = float(getattr(env, "gripper_open_command", 0.0))
    dry_gripper_command = [gripper_command]

    def load_dry_waypoints():
        if "dry_waypoints" in env.scene:
            descriptors = env.scene["dry_waypoints"]
            return (
                [np.asarray(item["position"], dtype=float) for item in descriptors],
                [float(item.get("gripper", gripper_command)) for item in descriptors],
                [float(item.get("dwell_s", 0.0)) for item in descriptors],
            )
        positions = [env.scene["engage"]] + list(env.scene["waypoints"])
        return positions, [gripper_command] * len(positions), [0.0] * len(positions)

    # Dry run: walk the task script so physics, grasp, success, and recording
    # can all be regression-tested without the physical device.
    canned, canned_grip, canned_dwell = load_dry_waypoints()
    canned_i = [0]
    canned_arrival_step = [None]

    def do_reset(*, advance_episode=False):
        nonlocal target, armed, force_filt, target_rv, commanded_rv, t_start
        if advance_episode:
            episode_attempt[0] += 1
            configure_episode_attempt(episode_attempt[0])
        else:
            with render_model_lock:
                env.reset()
        target = env.tool_home.copy()
        target_rv = commanded_rv = None
        rot_home[0] = None     # the operator's current wrist pose becomes the reference
        wrist_delta[0] = np.zeros(3)
        armed = False
        reset_haptic_feedback()
        reset_force_monitor()
        if device is not None and recorder is not None:
            device.set_centering_enabled(True)
        collection["started_monotonic"] = None
        collection["started_sim_time_s"] = None
        collection_gate.update(
            {
                "ready": bool(args.dry_run),
                "within_since": None,
                "distance_m": 0.0 if args.dry_run else float("inf"),
                "speed_m_s": 0.0 if args.dry_run else float("inf"),
                "velocity_valid": bool(args.dry_run),
                "gripper_ready": bool(args.dry_run or not cube_lift),
                "gripper_gap_m": (
                    args.device_grip_open if args.dry_run else float("nan")
                ),
                "gripper_open_reference_m": effective_device_grip_open(),
                "gripper_calibrated": bool(
                    args.dry_run
                    or not cube_lift
                    or grip_open_calibrator.reference_m is not None
                ),
            }
        )
        canned_i[0] = 0
        canned_arrival_step[0] = None
        positions, grips, dwells = load_dry_waypoints()
        canned[:] = positions
        canned_grip[:] = grips
        canned_dwell[:] = dwells
        episode[0] += 1
        episode[1] = step
        t_start = time.time() - step / args.control_freq

    def resolve_recorded_episode(keep, *, reset=True):
        if collection["state"] != "review":
            return False
        if not finish_recorded_episode(save=keep):
            return False
        if reset:
            do_reset(advance_episode=True)
        return True

    # Dry-run is the noninteractive collection/test path, so start it
    # immediately. Hardware collection always waits for the operator's S key.
    if recorder is not None and args.dry_run:
        start_recorded_episode()

    def slew_position_target(current, requested):
        delta = requested - current
        distance = np.linalg.norm(delta)
        if args.max_speed > 0 and distance > args.max_speed * period:
            return current + delta * (args.max_speed * period / distance)
        return requested

    def slew_rotation_target(current, requested):
        if requested is None:
            return current
        if current is None or args.max_rot_speed <= 0:
            return requested
        from scipy.spatial.transform import Rotation as _R

        step_rotation = _R.from_rotvec(requested) * _R.from_rotvec(current).inv()
        angle = step_rotation.magnitude()
        cap = np.radians(args.max_rot_speed) * period
        if angle <= cap:
            return requested
        axis = step_rotation.as_rotvec() / max(angle, 1e-12)
        return (_R.from_rotvec(axis * cap) * _R.from_rotvec(current)).as_rotvec()

    def next_dry_command():
        goal = np.asarray(canned[min(canned_i[0], len(canned) - 1)], dtype=float)
        dry_gripper_command[0] = canned_grip[canned_i[0]]
        if np.linalg.norm(goal - target) < 2e-4:
            if canned_arrival_step[0] is None:
                canned_arrival_step[0] = step
            dwell_steps = int(round(canned_dwell[canned_i[0]] * args.control_freq))
            if (
                step - canned_arrival_step[0] >= dwell_steps
                and canned_i[0] < len(canned) - 1
            ):
                canned_i[0] += 1
                canned_arrival_step[0] = None
                goal = np.asarray(canned[canned_i[0]], dtype=float)
                dry_gripper_command[0] = canned_grip[canned_i[0]]
        else:
            canned_arrival_step[0] = None
        delta = goal - target
        distance = np.linalg.norm(delta)
        if distance <= 1e-9:
            return goal
        return target + delta * min(1.0, args.dry_speed * period / distance)

    def map_device_gripper(device_gap):
        open_gap = effective_device_grip_open()
        fraction = np.clip(
            (float(device_gap) - args.device_grip_closed)
            / (open_gap - args.device_grip_closed),
            0.0,
            1.0,
        )
        return float(fraction * getattr(env, "gripper_open_command", 0.0))

    def publish_haptic_grip(grasp_force):
        if device is None or not cube_lift:
            return
        if recorder is not None and collection["state"] == "idle":
            # Negative is the SDK direction that opens/resists closing on the
            # present omega.7. It also brings the grip axis to a repeatable
            # collection start without using squeeze-as-button emulation.
            force = -args.collection_grip_open_force
        elif collection.get("state") == "review":
            force = 0.0
        else:
            force = (
                args.grip_feedback_sign
                * args.grip_force_gain
                * float(grasp_force)
            )
        device.set_grip_force(
            float(np.clip(force, -args.max_grip_force, args.max_grip_force))
        )

    def publish_haptic_force(contact_force):
        """Publish the newest sim-force target after every physics tick."""
        nonlocal armed
        if args.force_source == "contact":
            sim_force = np.asarray(contact_force, dtype=float)
        else:
            sim_force = env.reflected_force(args.force_source, target)
        if recorder is not None and collection["state"] == "idle":
            sim_force = np.zeros(3)
        if args.force_deadband > 0.0:
            magnitude = np.linalg.norm(sim_force)
            sim_force = (
                sim_force * (1.0 - args.force_deadband / magnitude)
                if magnitude > args.force_deadband
                else np.zeros(3)
            )
        if args.force_source == "none":
            reflect(np.zeros(3))
            return
        if not armed:
            # Reset convergence is not interaction force. Arm only once the
            # physical tool has reached the requested starting target.
            if np.linalg.norm(env.tool_pos - target) < 0.02:
                armed = True
            reflect(np.zeros(3))
            return
        collection_gain = 1.0
        if (
            recorder is not None
            and collection["state"] == "recording"
            and collection["started_monotonic"] is not None
        ):
            collection_gain = smooth_collection_force_gain(
                time.monotonic() - collection["started_monotonic"],
                args.collection_takeover_hold_ms / 1000.0,
                args.collection_force_ramp_ms / 1000.0,
            )
        reflect(collection_gain * args.force_gain * (pos_map.T @ sim_force))

    plot_every = max(1, int(round(args.control_freq * args.plot_span / W)))
    next_plot_step = [plot_every]

    try:
        while True:
            if recorder is not None and collection["state"] == "review":
                # Hold the final physical state while the operator decides.
                # Rebase wall-clock scheduling throughout the pause so resuming
                # does not try to catch up thousands of missed control ticks.
                reflect(np.zeros(3))
                if device is not None and cube_lift:
                    device.set_grip_force(0.0)
                t_start = time.time() - step / args.control_freq
                key = pop_viewer_key()
                action = review_action[0]
                review_action[0] = None
                if key in (ord("k"), 13) or action == "keep":
                    resolve_recorded_episode(keep=True)
                    continue
                if key in (ord("d"), ord("x"), 8, 127) or action == "delete":
                    resolve_recorded_episode(keep=False)
                    continue
                if key in (ord("q"), 27):
                    print("\n[dataset] choose KEEP or DELETE before quitting")
                time.sleep(0.01)
                continue

            if device is not None:
                state = device.get_state()
                # This is the force the independent haptic servo actually sent,
                # not merely the latest simulator request.
                sent[0] = np.asarray(state["force_cmd"], dtype=float)
                if recorder is not None and collection["state"] == "idle":
                    update_collection_gate(state)
                takeover_hold = (
                    recorder is not None
                    and collection["state"] == "recording"
                    and collection["started_monotonic"] is not None
                    and time.monotonic() - collection["started_monotonic"]
                    < args.collection_takeover_hold_ms / 1000.0
                )
                if (
                    recorder is not None
                    and collection["state"] == "idle"
                ) or takeover_hold:
                    # Keep the sampled start pristine until recording begins.
                    commanded = env.tool_home.copy()
                    commanded_rv = None
                    if cube_lift:
                        gripper_command = env.gripper_open_command
                else:
                    # absolute, unclamped: handle offset from home -> tool world target
                    commanded = env.tool_home + pos_map @ ((state["pos"] - home) * scale)
                    if cube_lift:
                        gripper_command = map_device_gripper(state["gripper"])
                if args.enable_rotation and not (
                    (recorder is not None and collection["state"] == "idle")
                    or takeover_hold
                ):
                    commanded_rv = orientation_command(state)
                if state["long_press_count"] != last_long:
                    last_long = state["long_press_count"]
                    if recorder is None:
                        do_reset()
                    elif collection["state"] == "idle":
                        start_recorded_episode()
                    elif collection["state"] == "recording":
                        stop_recorded_episode("device_long_press")
                    continue
            else:
                # Canned task, including analogue jaw commands and contact dwell.
                commanded = next_dry_command()
                gripper_command = dry_gripper_command[0]
                if cube_lift:
                    state["gripper"] = (
                        args.device_grip_open
                        * gripper_command
                        / max(env.gripper_open_command, 1e-9)
                    )
            # slew-limit how fast the target may chase the handle. Position stays
            # unbounded; only the rate is capped, which keeps the tool from being
            # flung into the book faster than the contact can settle.
            target = slew_position_target(target, commanded)
            target_rv = slew_rotation_target(target_rv, commanded_rv)

            # Advance the sim to match WALL time, not one step per iteration:
            # time.sleep overshoots and a stall costs several periods, so
            # single-stepping runs the sim slow, which time-warps recorded demos
            # and scales their contact forces with it.
            due = int((time.time() - t_start) * args.control_freq) + 1
            n_steps = min(max(1, due - step), MAX_CATCHUP)
            auto_finish_ready = False
            for substep_index in range(n_steps):
                # A renderer/recorder stall may require several simulation ticks
                # to catch up. Advance rate-limited commands on every one of
                # those ticks; otherwise a nominal 0.6 m/s command silently
                # becomes much slower whenever RGB or Zarr work is enabled.
                if substep_index > 0:
                    if device is None:
                        commanded = next_dry_command()
                        gripper_command = dry_gripper_command[0]
                    target = slew_position_target(target, commanded)
                    target_rv = slew_rotation_target(target_rv, commanded_rv)
                if cube_lift:
                    env.set_gripper_command(gripper_command)
                env.step(target, n_substeps=substeps, target_rotvec=target_rv)
                step += 1
                contact = env.contact_force()
                sensor_force = (
                    env.sensor_wrench(frame="world")[:3]
                    if sensor_model_enabled
                    else contact
                )
                publish_haptic_force(contact)
                grasp_force = env.grasp_force() if cube_lift else 0.0
                table_force = env.table_contact_force() if cube_lift else 0.0
                if cube_lift:
                    force_monitor["grasp_current_n"] = grasp_force
                    force_monitor["table_current_n"] = table_force
                    force_monitor["grasp_max_n"] = max(
                        force_monitor["grasp_max_n"], grasp_force
                    )
                    force_monitor["table_max_n"] = max(
                        force_monitor["table_max_n"], table_force
                    )
                publish_haptic_grip(grasp_force)

                # Read the independently running haptic servo at the sample
                # boundary. Repeated sequence IDs are now explicit evidence
                # that a simulator catch-up batch reused one hardware sample.
                sample_device_state = (
                    device.get_state() if device is not None else state
                )
                if device is not None:
                    sent[0] = np.asarray(
                        sample_device_state["force_cmd"], dtype=float
                    )
                if args.record:
                    rec_rows.append((
                        episode[0], (step - episode[1]) / args.control_freq,
                        step / args.control_freq, *sample_device_state["pos"], *target,
                        *(target_rv if target_rv is not None else np.zeros(3)),
                        *env.tool_pos, *env.tool_quat, *env.tool_vel,
                        *contact, *sent[0],
                        *env.book_pos, *env.book_quat, env.book_angle_deg(),
                        int(env.success())))
                if recorder is not None and collection["state"] == "recording":
                    episode_step = step - episode[1]
                    if episode_step > 0 and episode_step % dataset_stride == 0:
                        sample_index = episode_step // dataset_stride
                        frame_id = int(shot["n"])
                        has_new_frame = (
                            shot["frame"] is not None
                            and frame_id != last_dataset_frame_id[0]
                        )
                        dataset_frame = (
                            np.array(shot["frame"], copy=True)
                            if has_new_frame
                            else None
                        )
                        image_capture_time_s = None
                        if (
                            has_new_frame
                            and shot["sim_time_s"] is not None
                            and collection["started_sim_time_s"] is not None
                        ):
                            # Pyrite timestamps for every modality share the
                            # episode-relative origin. MuJoCo data.time is not
                            # reset between attempts, so storing it directly
                            # would shift RGB by all prior reset/episode time.
                            image_capture_time_s = max(
                                0.0,
                                float(shot["sim_time_s"])
                                - float(collection["started_sim_time_s"]),
                            )
                        sample_recorded = recorder.record_sample(
                            env,
                            timestamp_ms=sample_index * 1000.0 / args.dataset_hz,
                            target_pos=target,
                            target_rotvec=target_rv,
                            device_state=sample_device_state,
                            sent_force=sent[0],
                            image_rgb=dataset_frame,
                            image_capture_time_s=image_capture_time_s,
                            image_id=(frame_id if has_new_frame else None),
                            wall_time_ns=time.perf_counter_ns(),
                            control_batch_size=n_steps,
                            control_batch_index=substep_index,
                            deadline_lateness_ms=max(
                                0.0,
                                (
                                    (time.time() - t_start)
                                    - step / args.control_freq
                                )
                                * 1000.0,
                            ),
                        )
                        if has_new_frame:
                            last_dataset_frame_id[0] = frame_id
                        if sample_recorded and args.auto_finish and env.success():
                            auto_finish_ready = True
                            break

                if args.diagnose:
                    force_magnitude = float(np.linalg.norm(contact))
                    diag["f"].append(force_magnitude)
                    diag["h"].append(float(sample_device_state["pos"][0]))
                    diag["c"].append(force_magnitude > 0.5)

                if not args.no_plot and step >= next_plot_step[0]:
                    while next_plot_step[0] <= step:
                        next_plot_step[0] += plot_every
                    contact_magnitude = float(np.linalg.norm(contact))
                    sensor_magnitude = float(np.linalg.norm(sensor_force))
                    with plot_lock:
                        trace_sim_mag.append(contact_magnitude)
                        trace_sensor_mag.append(sensor_magnitude)
                        trace_raw.append(args.force_gain * contact_magnitude)
                        trace_felt.append(float(np.linalg.norm(sent[0])))
                        raw_dev = args.force_gain * (pos_map.T @ contact)
                        for axis in range(3):
                            trace_sim_xyz[axis].append(float(contact[axis]))
                            trace_sensor_xyz[axis].append(float(sensor_force[axis]))
                            trace_xyz[axis].append(float(sent[0][axis]))
                            trace_xyz_raw[axis].append(float(raw_dev[axis]))

            if auto_finish_ready:
                stop_recorded_episode("auto_success")
                if args.dry_run:
                    final_metric = collection["final_task_metric"]
                    resolve_recorded_episode(keep=True, reset=False)
                    print(
                        f"\n[dry-run] auto-finished at step {step}, "
                        f"{task_metric_summary(final_metric)}"
                    )
                    break
                continue

            # ---- view ----------------------------------------------------
            key = pop_viewer_key()
            if key != 255:
                if key in (ord("q"), 27):
                    if recorder is not None and collection["state"] == "recording":
                        stop_recorded_episode("operator_quit")
                        continue
                    break
                if key == ord("s") and recorder is not None:
                    if collection["state"] == "idle":
                        start_recorded_episode()
                    elif collection["state"] == "recording":
                        stop_recorded_episode("operator_stop")
                    continue
                if key == ord("r"):
                    if recorder is not None and collection["state"] == "recording":
                        print("\n[dataset] press S to stop, then KEEP or DELETE")
                    else:
                        do_reset()
                    continue
            if shot["err"] is not None:
                print(f"\n[view] render thread died: {shot['err']}")
                shot["err"] = None
            if viewer_state["err"] is not None:
                print(f"\n[view] viewer thread died: {viewer_state['err']}")
                viewer_state["err"] = None

            if not args.no_readout and step % 50 == 0:
                line = ((f"lift {100.0 * task_metric_value():5.1f} cm  "
                         if cube_lift else f"angle {task_metric_value():5.1f} deg  ")
                        + f"lag {np.linalg.norm(target-env.tool_pos)*1000:5.1f} mm  "
                        + f"sim F [{contact[0]:+7.2f} {contact[1]:+7.2f} {contact[2]:+7.2f}] N")
                if cube_lift:
                    line += (
                        f"  jaw {1000.0 * env.gripper_opening:4.1f} mm"
                        f" grasp {grasp_force:4.1f} N"
                    )
                if sensor_model_enabled:
                    line += (
                        f"  sensor [{sensor_force[0]:+6.2f} "
                        f"{sensor_force[1]:+6.2f} {sensor_force[2]:+6.2f}] N"
                    )
                if args.enable_rotation and target_rv is not None:
                    from scipy.spatial.transform import Rotation
                    target_rot = Rotation.from_rotvec(target_rv)
                    tool_rot = Rotation.from_quat(env.tool_quat[[1, 2, 3, 0]])
                    rot_error = np.degrees((target_rot * tool_rot.inv()).magnitude())
                    line += (
                        f"  wrist d {np.degrees(np.linalg.norm(wrist_delta[0])):4.1f} deg"
                        f"  rot err {rot_error:4.1f} deg"
                    )
                if device is not None:
                    d = device.get_state()["force_cmd"]
                    line += f"  handle [{d[0]:+5.2f} {d[1]:+5.2f} {d[2]:+5.2f}] N"
                print("\r" + line + ("   SUCCESS" if env.success() else "        "), end="")

            dry_run_waiting_for_auto_sample = (
                recorder is not None
                and collection["state"] == "recording"
                and args.auto_finish
                and env.success()
            )
            if args.dry_run and (
                step > 20 * args.control_freq
                or (env.success() and not dry_run_waiting_for_auto_sample)
            ):
                print(
                    f"\n[dry-run] stopped at step {step}, "
                    f"{task_metric_summary()}, success={env.success()}"
                )
                if recorder is not None and collection["state"] == "recording":
                    stop_recorded_episode(
                        "dry_run_success" if env.success() else "dry_run_timeout"
                    )
                    resolve_recorded_episode(keep=True, reset=False)
                break

            ahead = (step / args.control_freq) - (time.time() - t_start)
            if ahead > 0.0005:
                time.sleep(ahead - 0.0003)
    except KeyboardInterrupt:
        pass
    finally:
        viewer_state["run"] = False
        if viewer_thread is not None:
            viewer_thread.join(timeout=2.0)
        shot["run"] = False
        if render_thread is not None:
            render_thread.join(timeout=2.0)
        if recorder is not None and recorder.active:
            # An episode reaches disk only through an explicit KEEP decision.
            # This includes Ctrl-C and unexpected errors, which must not silently
            # turn a partial demonstration into training data.
            discarded = recorder.discard()
            print(
                f"\n[dataset] exit discarded {discarded} unconfirmed samples; "
                "use KEEP before quitting"
            )
        dt = time.time() - t_start
        achieved = step / max(dt, 1e-6)
        sim_t = step / max(args.control_freq, 1)
        rtf = sim_t / max(dt, 1e-6)
        print(f"\n{step} control steps, {shown[0]} frames shown "
              f"({shot['n']} rendered) in {dt:.1f}s ({achieved:.0f} Hz control, "
              f"{shown[0]/max(dt,1e-6):.1f} fps)")
        print(f"[timing] {sim_t:.1f}s of sim time in {dt:.1f}s wall = real-time factor "
              f"{rtf:.2f}")
        if abs(rtf - 1.0) > 0.1:
            faster = 1.0 / max(rtf, 1e-6)
            print(f"[timing] WARNING: the sim ran at {rtf:.2f}x real time, so recorded "
                  f"demos replay {faster:.2f}x {'faster' if rtf < 1 else 'slower'} than "
                  f"you performed them (and contact forces scale with that). Set "
                  f"--control-freq near the achieved {achieved:.0f} Hz to fix it.")
        if step > 50:
            t_real = 1.0 / achieved + 2.0 * tau
            k_real = 2.0 * args.damping / t_real
            ratio = k_real / max(k_handle, 1e-9)
            verdict = ("OK" if ratio >= 4.0 else
                       "thin -- teleop_ball needed 4x to stop bouncing" if ratio >= 1.0
                       else "OVER -- expect buzz/bounce on contact")
            print(f"[haptics] at the achieved {achieved:.0f} Hz the real limit was "
                  f"{k_real:.0f} N/m vs stiffness {k_handle:.0f} N/m = {ratio:.1f}x: {verdict}")
        print(
            f"[task] final {task_metric_summary()}: "
            f"{'SUCCESS' if env.success() else 'not successful'}"
        )

        if args.record_video and vid["w"] is not None:
            if vid["kind"] != "mp4v":      # our own ffmpeg pipe
                try:
                    vid["w"].stdin.close()
                    vid["w"].wait(timeout=120)   # +faststart rewrites the file
                    if vid["w"].returncode:
                        err = (vid["w"].stderr.read() or b"").decode(errors="replace")
                        print(f"[video] ffmpeg exited {vid['w'].returncode}: "
                              f"{err.strip()[-500:]}")
                except Exception as exc:
                    print(f"[video] ffmpeg did not shut down cleanly: {exc}")
            else:
                vid["w"].release()
            print(f"[video] wrote {vid['written']} frames = "
                  f"{vid['written']/float(args.video_fps):.1f}s to "
                  f"{os.path.abspath(args.record_video)}")

        if args.record and rec_rows:
            import csv
            cols = (["episode", "t_episode", "t"]
                    + [f"handle_{a}" for a in "xyz"] + [f"target_{a}" for a in "xyz"]
                    + [f"target_rotvec_{a}" for a in "xyz"]
                    + [f"tool_{a}" for a in "xyz"] + [f"tool_quat_{a}" for a in "wxyz"]
                    + [f"tool_vel_{a}" for a in "xyz"]
                    + [f"contact_force_{a}" for a in "xyz"]
                    + [f"sent_force_{a}" for a in "xyz"]
                    + [f"book_{a}" for a in "xyz"] + [f"book_quat_{a}" for a in "wxyz"]
                    + ["book_angle_deg", "success"])
            with open(args.record, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(cols)
                writer.writerows(rec_rows)
            print(f"[record] {len(rec_rows)} rows x {len(cols)} cols -> "
                  f"{os.path.abspath(args.record)}")

        if args.diagnose and len(diag["f"]) > 256:
            fs = achieved
            f_all = np.array(diag["f"])
            h_all = np.array(diag["h"])
            c_all = np.array(diag["c"], dtype=bool)
            # Analyse the longest unbroken stretch of CONTACT: including free
            # space fills the record with zeros and swamps the spectrum.
            best_i = best_n = 0
            i = 0
            while i < len(c_all):
                if c_all[i]:
                    j = i
                    while j < len(c_all) and c_all[j]:
                        j += 1
                    if j - i > best_n:
                        best_i, best_n = i, j - i
                    i = j
                else:
                    i += 1
            if best_n < 128:
                print(f"[diagnose] only {best_n} samples of continuous contact -- push "
                      f"and hold against the book for a few seconds, then quit.")
            else:
                sl = slice(best_i, best_i + best_n)

                def dominant(sig):
                    sig = np.asarray(sig, float)
                    sig = sig - sig.mean()
                    if not np.any(sig):
                        return 0.0, 0.0
                    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
                    freq = np.fft.rfftfreq(len(sig), 1.0 / fs)
                    band = (freq > 2.0) & (freq < 100.0)
                    if not band.any():
                        return 0.0, 0.0
                    k = np.flatnonzero(band)[np.argmax(spec[band])]
                    return freq[k], spec[k] / (spec[band].sum() + 1e-9)

                ff, fp = dominant(f_all[sl])
                hf, hp = dominant(h_all[sl])
                print(f"[diagnose] {best_n} samples ({best_n/fs:.1f} s) of continuous contact")
                print(f"[diagnose] contact force  peaks at {ff:5.1f} Hz "
                      f"({100*fp:.0f}% of 2-100 Hz energy)")
                print(f"[diagnose] handle motion  peaks at {hf:5.1f} Hz "
                      f"({100*hp:.0f}% of 2-100 Hz energy)")
                # A limit cycle is a LOOP -- the hand itself has to be oscillating.
                # Sim chatter shows up in the force while the handle stays smooth.
                if hf > 8.0 and hp > 0.15:
                    print("[diagnose] -> HAPTIC LIMIT CYCLE: the handle itself is being "
                          "shaken. Lower --stiffness, shorten --force-tau, or raise "
                          "--damping.")
                elif ff > 8.0:
                    print("[diagnose] -> SIM CHATTER rendered faithfully: force "
                          "oscillates but your hand does not. Lower --stiffness to "
                          "render less of it.")
                else:
                    print("[diagnose] -> SIM CONTACT, slow: the book separating and "
                          "being re-tapped.")

        if device is not None:
            device.close()
        env.close()
        try:
            # Explicitly free dm_control's GL/MuJoCo contexts before Python's
            # executor shutdown.  Accessing ``contexts.free`` was a no-op (the
            # namedtuple has no such method) and produced noisy atexit traces.
            env.physics.free()
        except Exception:
            pass


if __name__ == "__main__":
    main()
