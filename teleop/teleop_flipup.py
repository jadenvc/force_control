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
    python teleop_flipup.py --randomize-physics   # sample book mass/friction/size
    python teleop_flipup.py --dry-run             # no device: scripted arc, checks the loop
    python teleop_flipup.py --collect-dataset ~/data/flipup_sim_20hz.zarr
    python teleop_flipup.py --collect-dataset ~/data/flipup_sim_20hz.zarr --auto-finish

Push the handle AWAY from you to drive the tool into the bookend, and UP to
lever the book over. Normally, 'r' resets and 'q'/ESC quits. During dataset
collection, 's' starts/stops an episode. After stopping, click KEEP or DELETE
(keyboard: 'k'/'d'); the simulation remains paused until that decision.
``--auto-finish`` stops the active episode when the book reaches success.
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
    DEFAULT_ARM_DAMPING,
    DEFAULT_FORCE_CLIP,
    DEFAULT_JOINT_KD,
    DEFAULT_TOOL_KP,
    DEFAULT_TOOL_ROT_KD,
    DEFAULT_TOOL_ROT_KP,
    FlipUpTeleop,
)
from flipup.physical_properties import (  # noqa: E402
    DEFAULT_PHYSICAL_PROPERTIES,
    PhysicalProperties,
    sample_physical_properties,
)


# Force Dimension's device frame is +x toward the operator, +y to the operator's
# right, +z up. The default camera uses a left-oblique view (azimuth -30), so the
# push direction runs mostly INTO the screen with a visible leftward lean, and
# the lift direction is straight up the screen. So the natural mapping is
#     handle away from you (dev -x) -> tool into the bookend (sim +x)
#     handle up             (dev +z) -> tool up                (sim +z)
# and sim y takes -y_dev to keep the mapping right-handed. Same default as
# teleop_ball.py. Override with --axes if a direction comes out reversed.
DEFAULT_AXES = "-x,-y,z"


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


def main():
    parser = argparse.ArgumentParser()
    # ---- scene ------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=0,
                        help="scene seed: bookend pose, yaw and book offset. --dry-run "
                             "solves seeds 0, 1, 2 and 4 of 0-5 through this pipeline; "
                             "3 and 5 need a human to adapt the path")
    parser.add_argument("--randomize-physics", action="store_true",
                        help="sample book mass/friction/size from flipup's ranges")
    parser.add_argument("--book-mass", type=float, default=None, help="kg (default 1.375)")
    parser.add_argument("--book-friction", type=float, default=None,
                        help="book sliding friction (default 0.12)")
    parser.add_argument("--book-length", type=float, default=None, help="m (default 0.15)")
    parser.add_argument("--book-width", type=float, default=None, help="m (default 0.10)")
    parser.add_argument("--book-thickness", type=float, default=None, help="m (default 0.025)")
    parser.add_argument("--tool-kp", type=float, default=DEFAULT_TOOL_KP,
                        help="task-space translational stiffness (N/m). Leave alone: "
                             "the flip needs the shipped 16000 because the fingertip "
                             "pad is an 8 mm-radius capsule on the book's top edge, and the "
                             "scripted flip fails at every value <= 8000. Lowering it "
                             "does NOT soften the felt force -- --stiffness does that.")
    parser.add_argument("--tool-rot-kp", type=float, default=DEFAULT_TOOL_ROT_KP,
                        help="task-space rotational stiffness (N m/rad). Default 3000; "
                             "higher tracks faster but abrupt commands saturate the "
                             "28 N m wrist actuators and disturb position tracking.")
    parser.add_argument("--tool-rot-kd", type=float, default=DEFAULT_TOOL_ROT_KD,
                        help="task-space rotational damping (N m s/rad). Default 90, "
                             "approximately critical at the start pose for rot-kp 3000. "
                             "Set 0 to reproduce the original joint-damping-only controller.")
    parser.add_argument("--arm-damping", type=float, default=None,
                        help="multiplier on the arm's joint-space damping, relative to "
                             "the value flipup ships (64 N m s/rad on the arm joints, 16 "
                             f"on the wrist). Default {DEFAULT_ARM_DAMPING:.1f}: a 3 cm "
                             "step settles without overshoot and a scripted-path sweep "
                             "has fewer contact dropouts than at 2.0, for about 0.9 mm "
                             "more mean tracking lag. Use 2.0 for faster tracking; avoid "
                             "values above 4.0. Pass 1.0 to reproduce flipup exactly.")
    parser.add_argument("--settle", type=float, default=2.5,
                        help="seconds of sim time to slew the tool to the start pose "
                             "after each reset, before the operator takes over")
    parser.add_argument("--standoff", type=float, default=0.05,
                        help="how far in front of the book edge the tool starts (m)")

    # ---- mapping ----------------------------------------------------------
    parser.add_argument("--scale", type=float, nargs=3, default=[4.0, 4.0, 4.0],
                        help="sim metres per device metre, per axis. The flip arc "
                             "spans ~15 cm of push and ~14 cm of lift, so 4 asks for "
                             "~3.7 cm of handle travel in each. Raising it also raises "
                             "the felt stiffness (= tool_kp * scale * force_gain)")
    parser.add_argument("--home", type=float, nargs=3, default=[0.02, 0.0, -0.02],
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
    parser.add_argument("--max-speed", type=float, default=0.30,
                        help="cap on how fast the tool TARGET may travel (m/s), "
                             "0 = uncapped. A speed limit, not a workspace limit: "
                             "--scale multiplies hand velocity too, so without it a "
                             "brisk hand drives the tool into the book at ~1 m/s")

    # ---- haptics ----------------------------------------------------------
    parser.add_argument("--force-source", type=str, default="contact",
                        choices=["contact", "estimated", "wrist", "none"],
                        help="contact (default) = true solver contact force on the "
                             "robot: exactly 0.00 N in free space, ground truth in "
                             "contact. wrist = the WSG50's MuJoCo force sensor, tared "
                             "(cos +0.99 to contact, free space p90 0.2 N but spikes on "
                             "hard acceleration). estimated = BallPush's actuator-side "
                             "reconstruction, which is BROKEN for an arm this stiff: "
                             "measured 111 N of free-space phantom force. Kept only for "
                             "A/B comparison")
    parser.add_argument("--stiffness", type=float, default=1500.0,
                        help="target stiffness AT THE HANDLE (N/m) -- the thing that "
                             "decides whether contact feels solid or buzzes. force-gain "
                             "is derived as stiffness/(tool_kp*scale). Must stay under "
                             "2*damping/T_effective or the loop limit-cycles")
    parser.add_argument("--force-gain", type=float, default=None,
                        help="N of handle force per N of sim contact force. Default is "
                             "derived from --stiffness; set this to override")
    parser.add_argument("--force-clip", type=float, default=DEFAULT_FORCE_CLIP,
                        help="ceiling on the reflected SIM force (N), the analogue of "
                             "ball_force_limit. The scripted flip itself peaks at "
                             "~145 N, so this only clips a hard jam")
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
    parser.add_argument("--cam-azimuth", type=float, default=-30.0,
                        help="0 is dead-on the way the robot reaches (world +x), 90 is "
                             "square on to the plane the book pivots in. Default -30 is "
                             "left-oblique: enough side angle that the book's tilt "
                             "reads clearly, which at exactly 0 it geometrically cannot "
                             "(the long axis projects to a vertical line for EVERY tilt "
                             "and elevation there). Measured apparent tilt at a true "
                             "35.4 deg: az 0 -> 89.9 deg (no information), |az| 15 -> 59.2, "
                             "|az| 30 -> 52.6, |az| 90 -> ~35. Increase |azimuth| to judge "
                             "the angle geometrically rather than off the overlay")
    parser.add_argument("--cam-elevation", type=float, default=-25.0,
                        help="degrees above the horizontal, negative = looking down. "
                             "Steeper hides more of the book behind the gripper "
                             "(measured 3.8%% occluded at -20, 13.6%% at -40)")
    parser.add_argument("--cam-name", type=str, default=None,
                        help="render a fixed camera compiled into the scene instead of "
                             "the free one, e.g. 'ur5e/wsg50/d435i/rgb' -- the wrist "
                             "RealSense, literally the robot's own viewpoint. Note it "
                             "moves WITH the tool, so the book looks static while the "
                             "world rotates around it; fine for recording policy "
                             "observations, disorienting to drive from. Overrides "
                             "--cam-azimuth/--cam-elevation/--cam-distance")
    parser.add_argument("--cam-distance", type=float, default=0.75,
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
                             "Samples aligned RGB, state, action, sensed F/T, solver "
                             "ground-truth wrench, adaptive-compliance labels, and full "
                             "MuJoCo replay state at --dataset-hz")
    parser.add_argument("--auto-finish", action="store_true",
                        help="while collecting, stop the active episode automatically "
                             "when the book is successfully flipped; the episode still "
                             "waits for KEEP/DELETE confirmation")
    parser.add_argument("--dataset-hz", type=float, default=20.0,
                        help="behavior-cloning sample rate (default 20 Hz)")
    parser.add_argument("--dataset-image-size", type=int, nargs=2, default=[224, 224],
                        metavar=("WIDTH", "HEIGHT"),
                        help="stored Pyrite RGB size (default 224 224)")
    parser.add_argument("--dataset-no-rgb", action="store_true",
                        help="do not render observations while collecting; store black "
                             "placeholder RGB frames for low-dimensional experiments")
    parser.add_argument("--dataset-min-samples", type=int, default=20,
                        help="discard episodes shorter than this many 20 Hz samples")
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
    parser.add_argument("--dry-speed", type=float, default=0.05,
                        help="how fast --dry-run walks the scripted arc (m/s). The "
                             "scripted flip needs ~0.05; faster loses the book edge")
    parser.add_argument("--no-view", action="store_true",
                        help="no cv2 window (for headless runs). --record-video still "
                             "works; --collect-dataset also renders RGB unless "
                             "--dataset-no-rgb is set")
    args = parser.parse_args()
    if args.tool_kp <= 0.0:
        parser.error("--tool-kp must be greater than zero")
    if args.tool_rot_kp <= 0.0:
        parser.error("--tool-rot-kp must be greater than zero")
    if args.tool_rot_kd < 0.0:
        parser.error("--tool-rot-kd cannot be negative")
    if args.rot_scale <= 0.0:
        parser.error("--rot-scale must be greater than zero")
    if args.rot_deadzone < 0.0:
        parser.error("--rot-deadzone cannot be negative")
    if args.max_rot_speed < 0.0:
        parser.error("--max-rot-speed cannot be negative")
    if args.arm_damping is not None and args.arm_damping <= 0.0:
        parser.error("--arm-damping must be greater than zero")
    if args.dataset_hz <= 0.0:
        parser.error("--dataset-hz must be greater than zero")
    if args.dataset_min_samples <= 0:
        parser.error("--dataset-min-samples must be greater than zero")
    if args.dataset_wrench_filter < 0.0:
        parser.error("--dataset-wrench-filter cannot be negative")
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
    properties = book_properties(args)

    env = FlipUpTeleop(
        seed=args.seed,
        tool_kp=args.tool_kp,
        tool_rot_kp=args.tool_rot_kp,
        tool_rot_kd=args.tool_rot_kd,
        joint_kd=(None if args.arm_damping is None
                  else DEFAULT_JOINT_KD * args.arm_damping),
        force_clip=args.force_clip,
        tool_damping=args.tool_damping,
        physical_properties=properties,
        standoff=args.standoff,
        settle_s=args.settle,
        offscreen=(max(W, 640), max(H, 480)),
    )
    env.set_arm_visual(args.arm_view)
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
    collection = {
        "state": "idle" if recorder is not None else "disabled",
        "reason": None,
        "success": False,
        "final_book_angle_deg": None,
    }
    review_action = [None]

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
    print(f"[arm] task-space kp {env.tool_kp:.0f} N/m / {env.tool_rot_kp:.0f} N m/rad, "
          f"rotational kd {env.tool_rot_kd:.0f} N m s/rad, "
          f"joint damping {env.task_space_kd[0]:.0f} N m s/rad = {kd_mult:.2f}x what "
          f"flipup ships"
          + ("" if kd_mult >= 1.5 else
             "  <-- below 1.5x the arm rings without settling under saturation"))
    if kd_mult > 4.0:
        print(f"[arm] WARNING: {kd_mult:.1f}x is very sluggish; at 6x the tool cannot "
              f"follow the flip arc at all (contact duty fell to 2%, the flip failed).")
    print(f"[scene] book angle {env.book_angle_deg():.1f} deg from vertical "
          f"(success < 15). Tool starts {args.standoff*100:.0f} cm from the book edge, "
          f"settled to {env.settle_error*1000:.2f} mm")
    arc = env.scene["waypoints"]
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
    # Reference points measured from the scripted flip on seed 1, so the felt
    # force can be predicted before touching the device (teleop_ball's settled
    # feel for comparison: ~0.8 N sliding the block, ~8.7 N against the wall).
    def felt(sim_newtons):
        return min(sim_newtons * args.force_gain, args.max_force, sim_ceiling)
    # free-space reading of each source, measured over 4500 free-space samples
    free_ref = {"contact": 0.0, "wrist": 0.2, "estimated": 111.0, "none": 0.0}[
        args.force_source]
    print(f"[haptics] expected feel: free space {felt(free_ref):.2f} N, levering the book "
          f"{felt(30):.2f} N (sim 30 N median), pressing it against the bookend "
          f"{felt(82):.2f} N (sim 82 N median), first-touch spike "
          f"{felt(157):.2f} N (sim 157 N peak)")
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
    elif args.force_source == "wrist" and args.force_deadband <= 0.0:
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
        print(f"[axes] rotation OFF: wrist follows the heuristic's rule (30 deg down, "
              f"yawed away from the base). Pass --enable-rotation for 6 DoF.")
    print(f"[axes] device->sim mapping {args.axes}: push the handle "
          f"{'AWAY from' if args.axes.split(',')[0].startswith('-') else 'TOWARD'} you "
          f"to drive the tool into the bookend, and "
          f"{'UP' if not args.axes.split(',')[2].startswith('-') else 'DOWN'} to lever "
          f"the book over")
    if recorder is not None:
        print(
            f"[dataset] {args.dataset_hz:g} Hz Pyrite Zarr -> "
            f"{recorder.dataset_path} ({args.dataset_image_size[0]}x"
            f"{args.dataset_image_size[1]} RGB"
            f"{' placeholders' if args.dataset_no_rgb else ''}); "
            "S starts/stops, then click KEEP/DELETE (or K/D)"
            + ("; auto-finish ON" if args.auto_finish else "")
        )

    import cv2
    from collections import deque

    # ---- viewer: force strip chart + per-axis panel, as in teleop_ball ------
    PLOT_H = 130
    SIDE_W = 210
    trace_felt = deque(maxlen=W)
    trace_raw = deque(maxlen=W)
    trace_xyz = [deque(maxlen=SIDE_W) for _ in range(3)]      # FELT (sent to the device)
    trace_xyz_raw = [deque(maxlen=SIDE_W) for _ in range(3)]  # sim force x gain

    def draw_plot(frame):
        if args.no_plot or not trace_felt:
            return frame
        h, w = frame.shape[:2]
        top = h - PLOT_H
        frame[top:, :] = (frame[top:, :].astype(np.float32) * 0.25).astype(np.uint8)
        if args.plot_fixed_scale:
            fmax = max(args.max_force, 1.0)
        else:
            peak = max(max(trace_felt, default=0.0), max(trace_raw, default=0.0))
            fmax = min(max(1.0, 1.25 * peak), max(args.max_force, 1.0))
        for frac in (0.0, 0.5, 1.0):
            y = int(h - 1 - frac * (PLOT_H - 12))
            cv2.line(frame, (0, y), (w, y), (70, 70, 70), 1)
            cv2.putText(frame, f"{frac*fmax:.0f}N", (4, max(y - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

        def poly(trace, colour, thick):
            n = len(trace)
            if n < 2:
                return
            xs = np.arange(w - n, w)
            ys = h - 1 - np.clip(np.asarray(trace) / fmax, 0, 1) * (PLOT_H - 12)
            cv2.polylines(frame, [np.stack([xs, ys.astype(np.int32)], 1).astype(np.int32)],
                          False, colour, thick, cv2.LINE_AA)

        poly(trace_raw, (90, 90, 220), 1)      # sim force x gain, red-ish
        poly(trace_felt, (90, 220, 120), 2)    # what the handle is commanded, green
        cv2.putText(frame, f"|F| felt (green) / sim x gain (red)   {args.plot_span:.0f}s   "
                           f"src={args.force_source} gain={args.force_gain:.3f} "
                           f"tau={args.force_tau:.0f}ms",
                    (60, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1,
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
            cv2.putText(canvas, f"F{name} felt {now:+5.2f}N  (+/-{fmax:.1f})",
                        (x0 + 6, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
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

        render_fps = max(
            args.view_fps,
            args.dataset_hz
            if recorder is not None and not args.dataset_no_rgb
            else 0.0,
        )
        view_period = 1.0 / render_fps if render_fps > 0 else 0.0

        def render_loop():
            try:
                while shot["run"]:
                    t_frame = time.time()
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

    shown = [0, -1]      # [frames shown, index of the last frame drawn]
    button_y0, button_y1 = 48, 82
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
        """Book angle (and tool RPY) as text.

        Not decoration: at --cam-azimuth 0 the book's long axis projects to a
        vertical line for EVERY tilt and elevation, so the flip angle is
        geometrically invisible in that view and this number is the only readout.
        """
        angle = env.book_angle_deg()
        done = angle < 15.0
        cv2.putText(frame, f"book {angle:5.1f} deg from vertical  (need < 15)",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (120, 230, 140) if done else (230, 230, 230), 1, cv2.LINE_AA)
        if done:
            cv2.putText(frame, "SUCCESS", (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (120, 230, 140), 2, cv2.LINE_AA)
        if recorder is not None:
            status = collection["state"]
            if status == "idle":
                text = "DATASET IDLE - press S to start episode"
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
        frame = draw_plot(frame)
        if args.no_plot:
            canvas = frame
        else:
            canvas = np.zeros((H, W + SIDE_W, 3), dtype=np.uint8)
            canvas[:, :W] = frame
            canvas = draw_side(canvas)
        record_frame(canvas)
        shown[0] += 1
        shown[1] = shot["n"]
        if args.no_view:
            return 255
        cv2.imshow("Force Dimension -- FlipUp", canvas)
        return cv2.waitKey(1) & 0xFF

    def on_view_mouse(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONUP or collection["state"] != "review":
            return
        kx0, ky0, kx1, ky1 = keep_rect
        dx0, dy0, dx1, dy1 = delete_rect
        if kx0 <= x <= kx1 and ky0 <= y <= ky1:
            review_action[0] = "keep"
        elif dx0 <= x <= dx1 and dy0 <= y <= dy1:
            review_action[0] = "delete"

    if not args.no_view:
        cv2.namedWindow("Force Dimension -- FlipUp", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Force Dimension -- FlipUp", on_view_mouse)

    # ---- device -----------------------------------------------------------
    device = None
    if not args.dry_run:
        from fd_omega import FDOmega

        device = FDOmega(
            auto_init=args.auto_init,
            read_orientation=args.enable_rotation,
            spring_k=0.0,                # effortless free space
            wall_k=0.0,                  # no workspace walls
            wall_half=None,
            damping_b=args.damping,
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
        if args.enable_rotation:
            orientation_state = device.get_state()
            if not orientation_state["orientation_valid"]:
                device.close()
                raise RuntimeError("Wrist orientation was enabled but no valid frame was read")
            print(f">>> Wrist orientation sensor ready "
                  f"({orientation_state['orientation_sample_count']} initial sample)")
        print(">>> Push the handle AWAY from you and UP to lever the book upright. "
              "The WSG50 stays closed -- this is a nonprehensile pivot.")

    def start_recorded_episode():
        if recorder is None or collection["state"] != "idle":
            return False
        recorder.start_episode(
            {
                "seed": args.seed,
                "physical_properties": dict(properties.__dict__),
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
                },
                "mapping": {
                    "position_matrix": pos_map,
                    "rotation_matrix": rot_map,
                    "scale": scale,
                    "home": args.home,
                    "rotation_enabled": args.enable_rotation,
                    "rotation_frame": args.rot_frame,
                    "rotation_scale": args.rot_scale,
                    "rotation_deadzone": args.rot_deadzone,
                    "max_speed": args.max_speed,
                    "max_rotation_speed_deg_s": args.max_rot_speed,
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
                    "force_gain": args.force_gain,
                    "force_clip": args.force_clip,
                    "max_force": args.max_force,
                    "filter_tau_ms": args.force_tau,
                    "damping": args.damping,
                },
                "model": {
                    "nq": env.model.nq,
                    "nv": env.model.nv,
                    "nu": env.model.nu,
                    "na": env.model.na,
                    "nsensordata": env.model.nsensordata,
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
            }
        )
        episode[1] = step
        print(
            f"\n[dataset] recording started -> {recorder.dataset_path} "
            f"(press S to stop)"
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
        collection.update(
            {
                "state": "review",
                "reason": str(reason),
                "success": bool(env.success()),
                "final_book_angle_deg": float(env.book_angle_deg()),
            }
        )
        reflect(np.zeros(3))
        print(
            f"\n[dataset] episode stopped: {recorder.sample_count} samples, "
            f"success={collection['success']}, "
            f"angle={collection['final_book_angle_deg']:.1f} deg"
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
                    f"angle={collection['final_book_angle_deg']:.1f} deg"
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
            }
        )
        return True

    # ---- loop -------------------------------------------------------------
    diag = {"f": [], "h": [], "c": []}
    rec_rows = []
    episode = [0, 0]
    force_filt = np.zeros(3)
    sent = [np.zeros(3)]

    def reflect(f):
        """Smooth, clamp, and send a handle force.

        The filter is kept to a few ms via --force-tau because it sits INSIDE the
        feedback loop, so its lag lowers the stable stiffness roughly twice over.
        Clamping after filtering: FDOmega clamps too, but feeding a 140 N command
        into a 10 N clamp would leave its own filter nothing to do.
        """
        nonlocal force_filt
        f = np.asarray(f, dtype=float)
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
        if device is not None:
            device.set_reflected_force(f)

    # Wrist rotation -> absolute world rotvec for the tool. Convention copied from
    # teleop_ball.py, including the bug it documents: the delta convention and the
    # composition order MUST match, because a body-frame delta composed
    # extrinsically mixes two frames and the axes come out wrong.
    #   world frame: spatial delta  R_dev R_home^T , pre-multiplied
    #   tool  frame: body    delta  R_home^T R_dev , post-multiplied
    rot_home = [None]          # device wrist frame at start / after reset
    wrist_delta = [np.zeros(3)]

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

    home = np.array(args.home, dtype=float)
    target = env.tool_home.copy()
    target_rv = None           # tool rotvec actually commanded (None = derived from pos)
    commanded_rv = None        # what the wrist asks for, before the slew limit
    armed = False
    last_long = 0 if device is None else device.get_state()["long_press_count"]
    step = 0
    t_start = time.time()
    period = 1.0 / args.control_freq
    MAX_CATCHUP = 16
    state = {"pos": home.copy()}
    # dry run: walk the scripted arc so the whole loop can be exercised headlessly
    canned = [env.scene["engage"]] + list(env.scene["waypoints"])
    canned_i = [0]

    def do_reset():
        nonlocal target, armed, force_filt, target_rv, commanded_rv
        env.reset()
        target = env.tool_home.copy()
        target_rv = commanded_rv = None
        rot_home[0] = None     # the operator's current wrist pose becomes the reference
        wrist_delta[0] = np.zeros(3)
        armed = False
        force_filt = np.zeros(3)
        canned_i[0] = 0
        episode[0] += 1
        episode[1] = step
        if device is not None:
            device.set_reflected_force(np.zeros(3))

    def resolve_recorded_episode(keep, *, reset=True):
        if collection["state"] != "review":
            return False
        if not finish_recorded_episode(save=keep):
            return False
        if reset:
            do_reset()
        return True

    # Dry-run is the noninteractive collection/test path, so start it
    # immediately. Hardware collection always waits for the operator's S key.
    if recorder is not None and args.dry_run:
        start_recorded_episode()

    try:
        while True:
            if recorder is not None and collection["state"] == "review":
                # Hold the final physical state while the operator decides.
                # Rebase wall-clock scheduling throughout the pause so resuming
                # does not try to catch up thousands of missed control ticks.
                reflect(np.zeros(3))
                t_start = time.time() - step / args.control_freq
                key = 255
                if shot["frame"] is not None and shot["n"] != shown[1]:
                    key = show()
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
                # absolute, unclamped: handle offset from home -> tool world target
                commanded = env.tool_home + pos_map @ ((state["pos"] - home) * scale)
                if args.enable_rotation:
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
                # canned arc, advanced at --dry-speed so the slew limiter stays out
                # of the way (the scripted flip needs ~0.05 m/s to keep the edge)
                goal = np.asarray(canned[min(canned_i[0], len(canned) - 1)], dtype=float)
                if np.linalg.norm(goal - target) < 2e-4 and canned_i[0] < len(canned) - 1:
                    canned_i[0] += 1
                    goal = np.asarray(canned[canned_i[0]], dtype=float)
                delta = goal - target
                dist = np.linalg.norm(delta)
                commanded = (target + delta * min(1.0, args.dry_speed * period / dist)
                             if dist > 1e-9 else goal)
            # slew-limit how fast the target may chase the handle. Position stays
            # unbounded; only the rate is capped, which keeps the tool from being
            # flung into the book faster than the contact can settle.
            delta = commanded - target
            dist = np.linalg.norm(delta)
            if args.max_speed > 0 and dist > args.max_speed * period:
                target = target + delta * (args.max_speed * period / dist)
            else:
                target = commanded
            if commanded_rv is not None:
                # same treatment for orientation, via the rotation between the two
                if target_rv is None or args.max_rot_speed <= 0:
                    target_rv = commanded_rv
                else:
                    from scipy.spatial.transform import Rotation as _R
                    step_rot = (_R.from_rotvec(commanded_rv)
                                * _R.from_rotvec(target_rv).inv())
                    ang = step_rot.magnitude()
                    cap = np.radians(args.max_rot_speed) * period
                    if ang > cap:
                        axis = step_rot.as_rotvec() / max(ang, 1e-12)
                        target_rv = (_R.from_rotvec(axis * cap)
                                     * _R.from_rotvec(target_rv)).as_rotvec()
                    else:
                        target_rv = commanded_rv

            # Advance the sim to match WALL time, not one step per iteration:
            # time.sleep overshoots and a stall costs several periods, so
            # single-stepping runs the sim slow, which time-warps recorded demos
            # and scales their contact forces with it.
            due = int((time.time() - t_start) * args.control_freq) + 1
            n_steps = min(max(1, due - step), MAX_CATCHUP)
            auto_finish_ready = False
            for _ in range(n_steps):
                env.step(target, n_substeps=substeps, target_rotvec=target_rv)
                step += 1
                if args.record:
                    rec_rows.append((
                        episode[0], (step - episode[1]) / args.control_freq,
                        step / args.control_freq, *state["pos"], *target,
                        *(target_rv if target_rv is not None else np.zeros(3)),
                        *env.tool_pos, *env.tool_quat, *env.tool_vel,
                        *env.contact_force(), *sent[0],
                        *env.book_pos, *env.book_quat, env.book_angle_deg(),
                        int(env.success())))
                if recorder is not None and collection["state"] == "recording":
                    episode_step = step - episode[1]
                    if episode_step > 0 and episode_step % dataset_stride == 0:
                        sample_index = episode_step // dataset_stride - 1
                        dataset_frame = (
                            None
                            if shot["frame"] is None
                            else np.array(shot["frame"], copy=True)
                        )
                        sample_recorded = recorder.record_sample(
                            env,
                            timestamp_ms=sample_index * 1000.0 / args.dataset_hz,
                            target_pos=target,
                            target_rotvec=target_rv,
                            device_state=state,
                            sent_force=sent[0],
                            image_rgb=dataset_frame,
                            image_capture_time_s=shot["sim_time_s"],
                        )
                        if sample_recorded and args.auto_finish and env.success():
                            auto_finish_ready = True
                            break

            if auto_finish_ready:
                stop_recorded_episode("auto_success")
                if args.dry_run:
                    final_angle = collection["final_book_angle_deg"]
                    resolve_recorded_episode(keep=True, reset=False)
                    print(
                        f"\n[dry-run] auto-finished at step {step}, angle "
                        f"{final_angle:.1f} deg"
                    )
                    break
                continue

            # ---- haptics -------------------------------------------------
            sim_force = env.reflected_force(args.force_source, target)
            if args.force_deadband > 0.0:
                mag = np.linalg.norm(sim_force)
                sim_force = (sim_force * (1.0 - args.force_deadband / mag)
                             if mag > args.force_deadband else np.zeros(3))
            if args.force_source != "none":
                if not armed:
                    # after a reset the tool sits at its start pose while the
                    # mapped target can be centimetres away; that slew is not
                    # contact, so hold off until it first converges
                    if np.linalg.norm(env.tool_pos - target) < 0.02:
                        armed = True
                    reflect(np.zeros(3))
                else:
                    reflect(args.force_gain * (pos_map.T @ sim_force))
            else:
                sent[0] = np.zeros(3)

            contact = env.contact_force()
            if args.diagnose:
                fmag = float(np.linalg.norm(contact))
                diag["f"].append(fmag)
                diag["h"].append(float(state["pos"][0]))
                diag["c"].append(fmag > 0.5)

            if not args.no_plot:
                plot_every = max(1, int(round(args.control_freq * args.plot_span / W)))
                if step % plot_every == 0:
                    trace_raw.append(args.force_gain * float(np.linalg.norm(contact)))
                    trace_felt.append(float(np.linalg.norm(sent[0])))
                    raw_dev = args.force_gain * (pos_map.T @ contact)
                    for k in range(3):
                        trace_xyz[k].append(float(sent[0][k]))
                        trace_xyz_raw[k].append(float(raw_dev[k]))

            # ---- view ----------------------------------------------------
            if shot["frame"] is not None and shot["n"] != shown[1]:
                key = show()
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

            if not args.no_readout and step % 50 == 0:
                line = (f"angle {env.book_angle_deg():5.1f} deg  "
                        f"lag {np.linalg.norm(target-env.tool_pos)*1000:5.1f} mm  "
                        f"sim F [{contact[0]:+7.2f} {contact[1]:+7.2f} {contact[2]:+7.2f}] N")
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
                print(f"\n[dry-run] stopped at step {step}, angle "
                      f"{env.book_angle_deg():.1f} deg, success={env.success()}")
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
        print(f"[task] final book angle {env.book_angle_deg():.1f} deg from vertical: "
              f"{'SUCCESS' if env.success() else 'not upright'}")

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
        cv2.destroyAllWindows()
        env.close()
        try:
            env.physics.contexts.free()
        except Exception:
            pass


if __name__ == "__main__":
    main()
