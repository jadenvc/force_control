"""
Adapter that makes the flipup_minimal FlipUp scene drivable by the omega, with
the same haptic pipeline BallPush/teleop_ball.py uses.

flipup_minimal contains the UR5e + closed WSG50 pivoting a book up against a
bookend. This adapter subclasses ``FlipUpEnv`` and re-derives scene geometry
from its conventions.

What this adds on top of it
---------------------------
* **A Cartesian-position interface.** The operator commands the WSG50 tip
  position only; wrist orientation is derived from the commanded position
  exactly as the scripted heuristic derives it, so the 3-DoF omega is enough.
  This is the direct analogue of BallPush's ball-on-slide-joints: a position
  target in, a spring pulling the manipulator toward it.
* **Force sources**, all reporting *the force the world applies to the robot*
  in world axes, so a positive reading always opposes the motion that caused it:
    - ``contact``   true contact force, summed from the solver's per-contact
                    forces over every contact with the robot. Exactly zero in
                    free space (measured 0.00 N), which is why it is the default
                    here -- unlike BallPush, where the actuator-side estimate won.
    - ``wrist``     the WSG50's own MuJoCo force sensor, tared at reset and
                    negated (see ``wrist_force``). cos +0.99 to the contact sum,
                    p90 0.2 N in free space; the outliers are the sensor also
                    reading the gripper's inertia under hard acceleration.
    - ``estimated`` ``-clip(tool_kp * (target - tool)) + tool_damping * tool_vel``,
                    i.e. BallPush's default reconstructed from the actuator side.
                    Measurably WRONG for an arm this stiff: it renders the 3-7 mm
                    free-space tracking lag times 16 kN/m, so it reads 111 N mean
                    with no contact at all, and only cos +0.77 to the truth. It is
                    also the smoothest of the three, which is the trap. No
                    tool_damping fixes it (a least-squares fit returns 14
                    N/(m/s)); kept only for A/B comparison.
* **2.5x joint damping** (``DEFAULT_ARM_DAMPING``): the shipped value rings
  without settling once the actuators saturate. Pass ``joint_kd=DEFAULT_JOINT_KD``
  to reproduce flipup exactly.
* **Success + observables** for logging: the book's long-axis angle from
  vertical (the heuristic's own criterion), tool pose, contact force.
* **A threaded-friendly renderer.** dm_control software rendering of this scene
  costs ~50-75 ms per frame -- the high-resolution UR5e/finray/book meshes, not
  the pixel count (640x480 with only collision geoms visible is 8.7 ms). The
  camera factory here is safe to call from a viewer thread while the sim steps.

Measured notes that shaped the defaults (see teleop/README.md for the rest):

* **Do not soften the task-space stiffness.** 16 kN/m looks unrenderable, but
  the flip needs it: the fingertip pad is an 8 mm capsule contacting the book
  7.5 mm below its top edge, so a few mm of sag loses the edge. Sweeping the
  shipped heuristic over tool_kp, the flip succeeds at 16000 and 12000 and fails
  at every value at or below 8000. What makes the stiff arm renderable anyway is
  that this task's contact forces are ~10x BallPush's (19 N median while
  levering, vs ~2 N sliding a light block), so the force gain that lands the
  felt force in the same 1-8 N band is ~10x smaller, and the felt stiffness
  ``tool_kp * scale * force_gain`` still comes out in the same few-kN/m band.
* The scene geometry (bookend/book placement, the contact points the flip pivots
  about) is re-derived here from flipup.heuristic's conventions rather than
  imported, because that module only exposes a whole scripted episode.
"""

import sys
from pathlib import Path

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from spatialmath import SE3, SO3
from spatialmath.base import q2r

_FLIPUP_DIR = Path(__file__).resolve().parent.parent / "flipup_minimal"
if str(_FLIPUP_DIR) not in sys.path:
    sys.path.insert(0, str(_FLIPUP_DIR))

from flipup.environment import FlipUpEnv  # noqa: E402
from flipup.physical_properties import (  # noqa: E402
    DEFAULT_PHYSICAL_PROPERTIES,
    PhysicalProperties,
    sample_physical_properties,
)

# Translational stiffness stays at the shipped heuristic's value because
# softening it breaks the task (see the module docstring). Rotational impedance
# is tuned separately for wrist teleoperation.
DEFAULT_TOOL_KP = 16000.0
# Calibrated against the recorded book-pivot/book-floor jam (see
# _configure_book_fixture_contact's docstring). Softening solref's time
# constant past the compiled 0.01s does cut the sustained jam force a lot
# (mean 103N compiled -> 25N at (0.02, 2.0) stacked with the widened
# solimp below, on the recorded jam replay) -- but every softer value
# tested has an unacceptable failure mode, and the failure gets WORSE, not
# better, the more carefully you look for a middle ground:
#
# - (0.06, 2.0) collapses gravity support outright at REST, no velocity
#   needed -- sends the book ~20m underground during the settle window.
# - (0.02, 2.0) holds at rest, but tunnels the book clean through
#   book_floor at 1.0+ m/s of impact velocity (a hard drop/bounce).
# - The real deal-breaker: below that full-tunnel threshold, (0.02, 2.0)
#   doesn't fail cleanly -- it WEDGES. At 0.6-0.9 m/s (an ordinary
#   jostle/release speed reached easily via --scale amplifying normal
#   hand motion, nowhere near "already dropped it") the book sinks 1-4m
#   into the floor and gets stuck there rather than tunneling through or
#   bouncing back, leaving the task physically unable to proceed at all --
#   not a lost episode, a stuck one. That's strictly worse than the
#   original jam-force chatter this was meant to fix.
#
# Net: there's no tested value above 0.01 that's actually usable, so
# solref stays at the compiled default -- the jam-force win isn't safely
# reachable through this lever. book_fixture_solimp's width and
# book_fixture_friction (below) are the two levers that ARE independently
# safe and still worth using.
DEFAULT_BOOK_FIXTURE_SOLREF = (0.01, 1.0)
# Gravity-safe across 60 random book sizes stacked with this solref value
# (see _configure_book_fixture_contact's docstring for the (0.03, 2.0)
# combination that wasn't safe).
DEFAULT_BOOK_FIXTURE_SOLIMP = (0.85, 0.95, 0.08, 0.5, 2.0)
# Sliding friction on book_wall/pivot/floor was tried as a lever too, and
# rejected for a subtler reason than the other two: it's not independently
# safe/beneficial, its effect flips sign depending on book_fixture_solref.
#
# Measured at the (now-abandoned) softened solref=(0.02, 2.0): lowering
# friction cut jam-force chatter cleanly, but going low enough to matter
# (down toward 0.1) also let the book slide off its own support under
# gravity -- these surfaces hold the book at an angle, so friction is
# partly load-bearing, not just drag (0.1 drifted the book 1.8m sideways
# in 3s of doing nothing; random-book-size settle failed 14/30). A safe
# floor around 0.5-0.6 still looked like a real, if modest, win in that
# context.
#
# But re-checked against the ACTUAL shipped solref (compiled 0.01, not
# 0.02 -- see book_fixture_solref's comment for why that changed), the
# same 0.6 friction value made the recorded jam replay WORSE, not better
# (mean 102.8N unmodified -> 132.9N at friction=0.6 alone). Friction was
# apparently providing tangential damping that helps stabilize the
# multi-contact chatter specifically when the normal contact is already
# soft; with the normal contact back at compiled stiffness, removing that
# damping just makes the chatter worse. Net: not a safe, portable lever on
# its own -- left at the compiled value.
DEFAULT_BOOK_FIXTURE_FRICTION = None
DEFAULT_TOOL_ROT_KP = 3000.0
# Critical-ish Cartesian angular damping for the measured home-pose rotational
# inertia (mean effective inertia 0.70 kg m^2 gives 2*sqrt(kp*I) ~= 91).
DEFAULT_TOOL_ROT_KD = 90.0
DEFAULT_JOINT_KD = np.array([64.0, 64.0, 64.0, 16.0, 16.0, 16.0])
# ...except the damping, which is raised by default. The shipped value is
# marginally unstable once the actuators saturate: a 3 cm position step keeps
# ringing instead of settling. The scripted trajectory rarely excites that mode,
# but an operator and contact transients do.
#
# A current 3 cm free-space step sweep settled without overshoot at 1.5x and
# above; settling time increased from 380 ms at 1.5x to 520/656/1064 ms at
# 2.0/2.5/4.0x. On the scripted contact path, 2.5x added about 0.9 mm of mean lag
# over 2.0x but cut contact dropouts from 1.0/s to 0.2/s across seeds 0 and 1.
# Both completed the flip. This makes 2.5x the conservative teleoperation default;
# 2.0x remains a useful lower-lag setting, while values above 4x are too sluggish.
# The multiplier scales kd=64 on the arm joints and kd=16 on the wrist.
DEFAULT_ARM_DAMPING = 2.5
# Ceiling on the reflected sim force (N) -- the analogue of ball_force_limit.
# The heuristic's own flip peaks at ~145 N, so this only clips a hard jam.
DEFAULT_FORCE_CLIP = 200.0
# Cap on the Cartesian force the task-space controller may command, N, as a smooth
# tanh saturation. 0 = off, which is the default, and the reason is worth recording:
# it looked like the obvious analogue of BallPush's ball_force_limit (nothing else
# bounds the interaction force except the joint torque limits, which saturate around
# 110-125 N and, once clipping per joint, return an erratic +/-13 N instead of a
# steady force). But at DEFAULT_ARM_DAMPING the arm already needs ~56 N just to drag
# itself through FREE SPACE at 10 cm/s, so a limit low enough to soften contact also
# starves free motion: measured, 90 N and below never reached the book at all and the
# scripted flip went 4/4 -> 0/4. A limit cannot separate "pushing the arm" from
# "pushing on the book" while the damping term is that large. Left available for
# anyone running much lower --arm-damping; to bound what the OPERATOR feels, use
# --stiffness and --force-clip instead, which act on the rendering, not the robot.
DEFAULT_TOOL_FORCE_LIMIT = 0.0
DEFAULT_SURFACE_FORCE_LIMIT = 80.0
DEFAULT_PAD_TIME_CONSTANT = 0.010

# Fingertip-pad geom names as compiled on the full arm: the WSG50 mjcf is
# attached under the UR5e's own tree, so every gripper element (including the
# tip pads) is double-prefixed relative to the floating gripper's unprefixed
# "wsg50/..." names -- see FloatingFlipUpTeleop.TIP_CONTACT_GEOM_NAMES.
ARM_TIP_CONTACT_GEOM_NAMES = (
    "ur5e/wsg50/right_tip_pad",
    "ur5e/wsg50/left_tip_pad",
)
# Same softened-contact endpoint used by the floating gripper's tip_softness
# knob (see floating_flipup_teleop.py); duplicated here rather than imported
# since FloatingFlipUpTeleop already imports from this module, not vice versa.
SOFT_TIP_SOLREF = (0.020, 2.0)
SOFT_TIP_SOLIMP_WIDTH = 0.005

# Teleoperation keeps the fixture in the better-conditioned placement used by
# the earlier Force Dimension setup.  The floating controller does not need the
# extra reach margin, but sharing one scene makes arm/floating demonstrations
# directly comparable.
TELEOP_BOOKEND_X_MIN = 0.4
TELEOP_BOOKEND_X_SPAN = 0.2

# Muted, high-contrast cover colours.  Sampling a palette instead of arbitrary
# RGB avoids nearly-black books and neon colours that are implausible in the
# real demonstrations this simulator is intended to approximate.
BOOK_COLOR_PALETTE = np.array(
    [
        [0.64, 0.16, 0.14, 1.0],  # brick red
        [0.13, 0.25, 0.48, 1.0],  # navy
        [0.16, 0.43, 0.27, 1.0],  # forest green
        [0.67, 0.43, 0.12, 1.0],  # ochre
        [0.12, 0.43, 0.46, 1.0],  # teal
        [0.42, 0.25, 0.16, 1.0],  # brown
        [0.43, 0.20, 0.43, 1.0],  # plum
        [0.32, 0.36, 0.40, 1.0],  # slate
    ],
    dtype=float,
)


def _wxyz_from_matrix(rotation_matrix):
    q_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return q_xyzw[[3, 0, 1, 2]]


def tool_orientation(tool_position, robot_base_xy=(-0.3, 0.0)):
    """Wrist orientation for a tool position -- the heuristic's convention.

    Pitched 30 degrees down and yawed to face away from the robot base, so the
    fingers meet the book edge from above. Mirrors flipup.heuristic so the
    operator's wrist behaves exactly like the scripted run's.
    """
    delta = np.asarray(tool_position, dtype=float)[:2] - np.asarray(
        robot_base_xy, dtype=float
    )
    if abs(delta[0]) <= 1e-5:
        raise ValueError("Tool target is too close to the robot base x-coordinate")
    yaw_deg = np.degrees(np.arctan(delta[1] / delta[0]))
    return SO3.RPY(0.0, -30.0, yaw_deg, unit="deg")


def flipup_scene(seed=0, properties=DEFAULT_PHYSICAL_PROPERTIES, standoff=0.05):
    """Scene transforms plus the tool positions the flip pivots about, for a seed.

    Re-derives what flipup.heuristic computes internally: where the bookend and
    book go, the tool position that meets the book's top edge (``engage``), a
    retracted start (``prepare``), and the scripted quarter-turn arc
    (``waypoints``), which is handy as a reference path and for tests.

    All tool positions are world-frame xyz.
    """
    rng = np.random.RandomState(seed)
    bookend = SE3.Rt(
        SO3.RPY(90.0, 0.0, 180.0 + rng.uniform(-10.0, 10.0), unit="deg"),
        [
            TELEOP_BOOKEND_X_MIN + rng.uniform(0.0, TELEOP_BOOKEND_X_SPAN),
            rng.uniform(-0.2, 0.2),
            0.2,
        ],
    )
    book_relative = SE3.Rt(SO3.RPY(90.0, 0.0, 0.0, unit="deg"), [0.015, 0.035, 0.03])
    book = bookend * book_relative

    # Contact geometry, adapted to the sampled book dimensions exactly as the
    # heuristic does it (including its 1 cm trajectory clearance).
    book_length = properties.length_m + 0.01
    upper_contact_height = properties.thickness_m * 0.30
    lower_contact_height = properties.thickness_m * 0.70
    thickness_offset = properties.thickness_m * 0.28
    unit_x = np.array([1.0, 0.0, 0.0])
    unit_y = np.array([0.0, 1.0, 0.0])

    book_tip_contact = np.array(
        [properties.length_m, properties.width_m / 2.0, upper_contact_height]
    )
    tip = np.asarray((book_relative * book_tip_contact).reshape(3), dtype=float)
    rotation_start = np.array([thickness_offset + book_length, tip[1], tip[2]])

    arc_local = [rotation_start]
    for waypoint_id in range(21):
        angle = waypoint_id / 20.0 * np.pi / 2.0
        arc_local.append(
            rotation_start
            + (
                upper_contact_height * np.sin(angle)
                + book_length * np.cos(angle)
                - book_length
            )
            * unit_x
            + (
                lower_contact_height * np.cos(angle)
                + book_length * np.sin(angle)
                - lower_contact_height
            )
            * unit_y
        )

    def to_world(local_point):
        return np.asarray((bookend * local_point).reshape(3), dtype=float)

    origin = to_world(np.zeros(3))
    return {
        "bookend_transform": np.asarray(bookend.data[0], dtype=float),
        "book_transform": np.asarray(book.data[0], dtype=float),
        "prepare": to_world(tip + standoff * unit_x),
        "engage": to_world(rotation_start),
        "waypoints": np.array([to_world(p) for p in arc_local]),
        # World directions of "push into the bookend" and "lift", for sanity
        # checks on the device axis mapping.
        "push_dir": to_world(-unit_x) - origin,
        "lift_dir": to_world(unit_y) - origin,
        # Columns map local bookend xyz to world xyz.  Local +x is the
        # front/standoff direction, +y is lift, and +z is lateral.
        "bookend_rotation": np.asarray(bookend.R, dtype=float),
    }


def sample_episode_properties(
    base_properties,
    rng,
    *,
    size_jitter=0.20,
    mass_jitter=0.20,
):
    """Independently vary book height, width and mass around a nominal book.

    ``length_m`` is the upright book height.  Thickness and contact friction
    deliberately remain fixed so this is a controlled three-factor domain
    randomization rather than the much broader legacy ``--randomize-physics``.
    """
    if not isinstance(base_properties, PhysicalProperties):
        raise TypeError("base_properties must be a PhysicalProperties")
    if not (0.0 <= size_jitter < 1.0 and 0.0 <= mass_jitter < 1.0):
        raise ValueError("size and mass jitter must be in [0, 1)")

    # Rejection only matters at the extreme corner where a -20% height and
    # +20% width would make the nominal 15 x 10 cm book square.
    for _ in range(100):
        length = base_properties.length_m * rng.uniform(
            1.0 - size_jitter, 1.0 + size_jitter
        )
        width = base_properties.width_m * rng.uniform(
            1.0 - size_jitter, 1.0 + size_jitter
        )
        if length > width + 1e-4:
            break
    else:  # pragma: no cover - impossible for the shipped nominal geometry
        raise RuntimeError("could not sample a book with length greater than width")
    mass = base_properties.mass_kg * rng.uniform(
        1.0 - mass_jitter, 1.0 + mass_jitter
    )
    return PhysicalProperties(
        mass_kg=float(mass),
        sliding_friction=base_properties.sliding_friction,
        torsional_friction=base_properties.torsional_friction,
        rolling_friction=base_properties.rolling_friction,
        length_m=float(length),
        width_m=float(width),
        thickness_m=base_properties.thickness_m,
    )


def sample_book_color(rng):
    """Return a reproducible realistic cover colour with slight variation."""
    base = BOOK_COLOR_PALETTE[int(rng.integers(len(BOOK_COLOR_PALETTE)))].copy()
    base[:3] = np.clip(base[:3] * rng.uniform(0.88, 1.12), 0.08, 0.82)
    return base


def sample_start_pose(
    scene,
    rng,
    *,
    prism_size=(0.04, 0.06, 0.05),
    center_probability=0.70,
    force_center=False,
):
    """Sample a tool start from a fixed prism in front of the book.

    ``prism_size`` is full depth/lateral/vertical size in metres.  With the
    default mixture, 70% of starts come from a tight clipped Gaussian around
    the head-on centre and 30% uniformly cover the full prism.  The first
    episode can use ``force_center`` so every collection begins with a known
    reference demonstration.
    """
    size = np.asarray(prism_size, dtype=float)
    if size.shape != (3,) or np.any(size < 0.0):
        raise ValueError("prism_size must contain three nonnegative values")
    if not 0.0 <= center_probability <= 1.0:
        raise ValueError("center_probability must be in [0, 1]")

    if force_center:
        normalized = np.zeros(3)
        component = "center"
    elif rng.random() < center_probability:
        # sigma 0.22 means most central samples stay within roughly the middle
        # half of the volume.  Clipping retains the advertised hard bounds.
        normalized = np.clip(rng.normal(0.0, 0.22, size=3), -0.5, 0.5)
        component = "center_gaussian"
    else:
        normalized = rng.uniform(-0.5, 0.5, size=3)
        component = "uniform"

    # User-facing order is depth/lateral/vertical.  Bookend local coordinates
    # are front(+x), lift(+y), lateral(+z).
    offset_dlv = normalized * size
    offset_local_xyz = np.array(
        [offset_dlv[0], offset_dlv[2], offset_dlv[1]], dtype=float
    )
    rotation = np.asarray(scene["bookend_rotation"], dtype=float)
    position = np.asarray(scene["prepare"], dtype=float) + rotation @ offset_local_xyz
    return position, {
        "component": component,
        "prism_size_depth_lateral_vertical_m": size,
        "normalized_depth_lateral_vertical": normalized,
        "offset_depth_lateral_vertical_m": offset_dlv,
        "position_world_m": position,
    }


class FlipUpTeleop(FlipUpEnv):
    """FlipUp with a Cartesian-position interface and contact-force readout."""

    controller_kind = "joint_arm"
    default_tip_softness = 0.0

    def __init__(
        self,
        seed=0,
        tool_kp=DEFAULT_TOOL_KP,
        tool_rot_kp=DEFAULT_TOOL_ROT_KP,
        tool_rot_kd=DEFAULT_TOOL_ROT_KD,
        joint_kd=None,
        force_clip=DEFAULT_FORCE_CLIP,
        tool_force_limit=DEFAULT_TOOL_FORCE_LIMIT,
        surface_force_limit=DEFAULT_SURFACE_FORCE_LIMIT,
        tool_damping=0.0,
        physical_properties=None,
        randomize_physics=False,
        standoff=0.05,
        settle_s=2.5,
        settle_speed=0.25,
        offscreen=(1024, 768),
        collision_envelope_dimensions=None,
        tip_softness=0.0,
        table_friction=None,
        approach_compliance_distance_m=0.0,
        approach_compliance_min_kp_ratio=0.2,
        approach_max_speed_mps=0.0,
        book_normal_force_limit=0.0,
        tip_softness_max_solref=None,
        tip_softness_max_width=None,
        bookend_solref=None,
        bookend_solimp=None,
        bookend_friction=None,
        book_fixture_solref=None,
        book_fixture_solimp=None,
        book_fixture_friction=None,
        tip_friction=None,
        tool_kp_axes=(1.0, 1.0, 1.0),
        tool_cartesian_kd=(0.0, 0.0, 0.0),
        noslip_iterations=0,
    ):
        if tip_softness_max_solref is not None and len(tip_softness_max_solref) != 2:
            raise ValueError(
                "tip_softness_max_solref must have exactly 2 values "
                "(time_constant, damping_ratio)"
            )
        if bookend_solref is not None and len(bookend_solref) != 2:
            raise ValueError(
                "bookend_solref must have exactly 2 values (time_constant, "
                "damping_ratio)"
            )
        if bookend_solimp is not None and len(bookend_solimp) != 5:
            raise ValueError(
                "bookend_solimp must have exactly 5 values (d0, d_width, "
                "width, midpoint, power)"
            )
        if bookend_friction is not None and len(bookend_friction) != 3:
            raise ValueError(
                "bookend_friction must have exactly 3 values (sliding, "
                "torsional, rolling)"
            )
        if book_fixture_solref is not None and len(book_fixture_solref) != 2:
            raise ValueError(
                "book_fixture_solref must have exactly 2 values "
                "(time_constant, damping_ratio)"
            )
        if book_fixture_solimp is not None and len(book_fixture_solimp) != 5:
            raise ValueError(
                "book_fixture_solimp must have exactly 5 values (d0, "
                "d_width, width, midpoint, power)"
            )
        if book_fixture_friction is not None and len(book_fixture_friction) != 3:
            raise ValueError(
                "book_fixture_friction must have exactly 3 values (sliding, "
                "torsional, rolling)"
            )
        if tip_friction is not None and len(tip_friction) != 3:
            raise ValueError(
                "tip_friction must have exactly 3 values (sliding, "
                "torsional, rolling)"
            )
        if len(tool_kp_axes) != 3:
            raise ValueError(
                "tool_kp_axes must have exactly 3 values -- per-axis "
                "WORLD-frame multipliers on tool_kp, (1,1,1) is the "
                "original isotropic behavior"
            )
        if any(float(v) <= 0.0 for v in tool_kp_axes):
            raise ValueError("tool_kp_axes values must be positive")
        if len(tool_cartesian_kd) != 3:
            raise ValueError(
                "tool_cartesian_kd must have exactly 3 values (WORLD xyz "
                "translational Cartesian damping, N/(m/s)) -- (0,0,0) is "
                "the original behavior (translation damping came only from "
                "the fixed joint-space term, task_space_kd)"
            )
        if any(float(v) < 0.0 for v in tool_cartesian_kd):
            raise ValueError("tool_cartesian_kd values cannot be negative")
        if int(noslip_iterations) < 0:
            raise ValueError("noslip_iterations cannot be negative")
        if float(approach_compliance_distance_m) < 0.0:
            raise ValueError("approach_compliance_distance_m cannot be negative")
        if not 0.0 < float(approach_compliance_min_kp_ratio) <= 1.0:
            raise ValueError("approach_compliance_min_kp_ratio must be in (0, 1]")
        if float(approach_max_speed_mps) < 0.0:
            raise ValueError("approach_max_speed_mps cannot be negative")
        if float(book_normal_force_limit) < 0.0:
            raise ValueError("book_normal_force_limit cannot be negative")
        if physical_properties is None:
            physical_properties = (
                sample_physical_properties(seed)
                if randomize_physics
                else DEFAULT_PHYSICAL_PROPERTIES
            )
        if not isinstance(physical_properties, PhysicalProperties):
            raise TypeError("physical_properties must be a PhysicalProperties")

        self.seed = int(seed)
        self.physical_properties = physical_properties
        self.scene = flipup_scene(seed, physical_properties, standoff=standoff)
        self.tool_kp = float(tool_kp)
        # Per-WORLD-axis multiplier on tool_kp, applied only to
        # task_space_kp's Cartesian translational diagonal (see
        # _compute_effective_translation_kp). Every OTHER formula that uses
        # tool_kp -- surface_force_limit/book_normal_force_limit deflection
        # caps, limited_target's force-squash, the "estimated" force source
        # -- keeps using the plain scalar self.tool_kp unmodified. That is
        # an approximation once tool_kp_axes is anisotropic (those formulas
        # implicitly assume the isotropic tool_kp*err relationship), not a
        # fully-general per-axis correction -- acceptable because those
        # limits are fail-safes, not the primary control law, and this
        # keeps the (default, isotropic) behavior of everything else in
        # this class byte-identical to before this parameter existed.
        self.tool_kp_axes = np.asarray(tool_kp_axes, dtype=float)
        # WORLD-frame translational Cartesian damping (N/(m/s)), i.e. the
        # missing D in task_space_cartesian_kd's F=K*e-D*xdot law -- see
        # that attribute's construction below. (0,0,0) preserves the
        # original behavior, where all translational damping came from the
        # fixed joint-space task_space_kd term instead.
        self.tool_cartesian_kd = np.asarray(tool_cartesian_kd, dtype=float)
        self.tool_rot_kp = float(tool_rot_kp)
        self.tool_rot_kd = float(tool_rot_kd)
        self.force_clip = float(force_clip)
        self.tool_force_limit = float(tool_force_limit)
        self.surface_force_limit = float(surface_force_limit)
        if self.surface_force_limit < 0.0:
            raise ValueError("surface_force_limit cannot be negative")
        self.tool_damping = float(tool_damping)
        self.standoff = float(standoff)
        self.settle_s = float(settle_s)
        self.settle_speed = float(settle_speed)
        self._teleop_ready = False

        if collision_envelope_dimensions is None:
            collision_envelope_dimensions = np.array(
                [
                    1.2 * physical_properties.length_m,
                    1.2 * physical_properties.width_m,
                    physical_properties.thickness_m,
                ],
                dtype=float,
            )

        super().__init__(
            self.scene["bookend_transform"],
            self.scene["book_transform"],
            show_viewer=False,
            physical_properties=physical_properties,
            collision_envelope_dimensions=collision_envelope_dimensions,
        )
        self.physical_properties = physical_properties

        if not 0.0 <= float(tip_softness) <= 1.0:
            raise ValueError("tip_softness must be in [0, 1]")
        self.tip_softness = float(tip_softness)
        self.tip_softness_max_solref = (
            SOFT_TIP_SOLREF
            if tip_softness_max_solref is None
            else tuple(float(v) for v in tip_softness_max_solref)
        )
        self.tip_softness_max_width = (
            SOFT_TIP_SOLIMP_WIDTH
            if tip_softness_max_width is None
            else float(tip_softness_max_width)
        )
        self.tip_contact_geom_ids = np.array(
            [self.model.geom(name).id for name in ARM_TIP_CONTACT_GEOM_NAMES],
            dtype=np.int32,
        )
        self.tip_friction = (
            None if tip_friction is None else tuple(float(v) for v in tip_friction)
        )
        self._configure_tip_contact()
        if table_friction is not None and len(table_friction) != 3:
            raise ValueError(
                "table_friction must have exactly 3 values (sliding, "
                "torsional, rolling), matching MuJoCo's geom_friction convention"
            )
        self.table_friction = (
            None if table_friction is None else tuple(float(v) for v in table_friction)
        )
        self._configure_table_friction()
        self.bookend_solref = (
            None if bookend_solref is None else tuple(float(v) for v in bookend_solref)
        )
        self.bookend_solimp = (
            None if bookend_solimp is None else tuple(float(v) for v in bookend_solimp)
        )
        self.bookend_friction = (
            None
            if bookend_friction is None
            else tuple(float(v) for v in bookend_friction)
        )
        self._configure_bookend_contact()
        self.book_fixture_solref = (
            None
            if book_fixture_solref is None
            else tuple(float(v) for v in book_fixture_solref)
        )
        self.book_fixture_solimp = (
            None
            if book_fixture_solimp is None
            else tuple(float(v) for v in book_fixture_solimp)
        )
        self.book_fixture_friction = (
            None
            if book_fixture_friction is None
            else tuple(float(v) for v in book_fixture_friction)
        )
        self._configure_book_fixture_contact()

        # Runtime book randomization keeps the compiled topology fixed, which
        # lets the camera/render thread and the recorder remain valid across
        # episodes.  Contact uses a box geom; the visual mesh is scaled from
        # this immutable reference copy.
        self.book_visual_geom_id = self.model.geom(
            "book2_blend/book_visual"
        ).id
        self.book_mesh_id = int(self.model.geom_dataid[self.book_visual_geom_id])
        mesh_start = int(self.model.mesh_vertadr[self.book_mesh_id])
        mesh_count = int(self.model.mesh_vertnum[self.book_mesh_id])
        self._book_mesh_slice = slice(mesh_start, mesh_start + mesh_count)
        self._book_mesh_reference = np.asarray(
            self.model.mesh_vert[self._book_mesh_slice], dtype=float
        ).copy()
        self._book_mesh_reference_dimensions = np.array(
            [
                physical_properties.thickness_m,
                physical_properties.width_m,
                physical_properties.length_m,
            ],
            dtype=float,
        )
        book_attachment_id = int(self.model.body_parentid[self.book_body_id])
        self.book_free_joint_id = int(self.model.body_jntadr[book_attachment_id])
        self.book_free_qpos_adr = int(
            self.model.jnt_qposadr[self.book_free_joint_id]
        )
        self.book_color = np.array([0.42, 0.25, 0.16, 1.0], dtype=float)
        self._book_mesh_version = 0

        # Geom size/position and collision-mesh vertices are not generally safe
        # runtime edits in MuJoCo.  This primitive was compiled at the largest
        # episode dimensions, so shrinking it stays inside these conservative
        # broad-phase bounds.  Preserve those bounds after mj_setConst updates
        # the randomized body mass/inertia.
        self._book_collision_envelope_dimensions = 2.0 * np.asarray(
            self.model.geom_size[self.book_collision_geom_id], dtype=float
        )
        self._book_collision_geom_rbound = float(
            self.model.geom_rbound[self.book_collision_geom_id]
        )
        self._book_collision_geom_aabb = np.asarray(
            self.model.geom_aabb[self.book_collision_geom_id], dtype=float
        ).copy()
        self._compiled_bvh_aabb = np.asarray(self.model.bvh_aabb, dtype=float).copy()

        # Off (0) by compiled default. MuJoCo's main constraint solver
        # (iterations=100 here) converges the overall contact/friction
        # problem, but with multiple simultaneous near-redundant contacts
        # (compiled contact_count is 3-5 during a real flip: fingertip
        # pads against the book plus 1-2 bookend surfaces at once) its
        # friction-force split across them isn't uniquely determined and
        # can still wobble step to step even though the TOTAL is converged
        # -- this shows up as near-Nyquist-frequency noise in the summed
        # contact wrench specifically, not in ncon/dropouts. noslip_iterations
        # runs MuJoCo's separate post-pass aimed exactly at this: refining
        # the friction split under the already-solved normal forces. Untested
        # against a real flip in this repo as of writing -- exposed here so
        # it can be tried live rather than guessed at.
        self.model.opt.noslip_iterations = int(noslip_iterations)

        self.task_space_kp = np.diag(
            list(self.tool_kp * self.tool_kp_axes) + [self.tool_rot_kp] * 3
        ).astype(float)
        self.task_space_kd = (
            DEFAULT_JOINT_KD * DEFAULT_ARM_DAMPING if joint_kd is None
            else np.broadcast_to(np.asarray(joint_kd, dtype=float), (6,)).copy()
        )
        self.task_space_cartesian_kd = np.concatenate(
            [self.tool_cartesian_kd, [self.tool_rot_kd] * 3]
        ).astype(float)

        # Bodies belonging to the robot (arm + gripper + wrist camera): every
        # body whose kinematic root is the UR5e's. Contacts with exactly one
        # side in this set are what the operator should feel.
        robot_root = int(self.model.body_rootid[self.model.body("ur5e/base").id])
        self._robot_bodies = frozenset(
            i for i in range(self.model.nbody)
            if int(self.model.body_rootid[i]) == robot_root
        )
        self._contact_buf = np.zeros(6, dtype=float)
        self._init_surface_safety()
        self.book_force_limit = float(book_normal_force_limit)
        self._init_book_safety()
        self.approach_compliance_distance_m = float(approach_compliance_distance_m)
        self.approach_compliance_min_kp_ratio = float(approach_compliance_min_kp_ratio)
        self._approach_compliance_enabled = self.approach_compliance_distance_m > 0.0
        self.approach_max_speed_mps = float(approach_max_speed_mps)
        # Recomputed every step in step(); the pre-step defaults below only
        # matter for code that reads them before the first step() call.
        # A 3-vector (WORLD xyz) since tool_kp_axes.
        self._effective_translation_kp = self.tool_kp * self.tool_kp_axes

        self.wrist_force_adr = None
        self.wrist_torque_adr = None
        self.wrist_site_id = None
        try:
            self.wrist_force_adr = int(
                np.asarray(self.model.sensor("ur5e/wsg50/wrist_force_sensor").adr).item()
            )
            self.wrist_torque_adr = int(
                np.asarray(self.model.sensor("ur5e/wsg50/wrist_torque_sensor").adr).item()
            )
            self.wrist_site_id = self.model.site("ur5e/wsg50/ft_sensor_site").id
        except (KeyError, ValueError):
            pass
        self._wrist_tare = np.zeros(6)

        # A bigger offscreen buffer than ground.xml's 600x480, so the viewer can
        # render at the size teleop_ball uses. Must be set before the first
        # render: the GL framebuffer is sized from it when the context is made.
        self.model.vis.global_.offwidth = max(
            int(offscreen[0]), int(self.model.vis.global_.offwidth)
        )
        self.model.vis.global_.offheight = max(
            int(offscreen[1]), int(self.model.vis.global_.offheight)
        )

        self.tool_home = self.scene["prepare"].copy()
        self.settle_error = float("nan")
        self._teleop_ready = True
        self.reset()

    def configure_episode(self, physical_properties, book_color, tool_home):
        """Apply per-episode book and starting-pose parameters, then reset.

        No model dimensions change, so complete MuJoCo state snapshots retain a
        stable shape across all episodes in one dataset.
        """
        if not isinstance(physical_properties, PhysicalProperties):
            raise TypeError("physical_properties must be a PhysicalProperties")
        color = np.asarray(book_color, dtype=float)
        if color.shape != (4,) or np.any(~np.isfinite(color)):
            raise ValueError("book_color must be finite RGBA")
        tool_home = np.asarray(tool_home, dtype=float)
        if tool_home.shape != (3,) or np.any(~np.isfinite(tool_home)):
            raise ValueError("tool_home must be a finite xyz position")

        self.physical_properties = physical_properties
        self.book_color = np.clip(color, 0.0, 1.0)
        self.scene = flipup_scene(
            self.seed, physical_properties, standoff=self.standoff
        )
        self.tool_home = tool_home.copy()

        half_size = np.array(
            [
                physical_properties.length_m,
                physical_properties.width_m,
                physical_properties.thickness_m,
            ],
            dtype=float,
        ) / 2.0
        if np.any(2.0 * half_size > self._book_collision_envelope_dimensions + 1e-12):
            raise ValueError(
                "episode book dimensions exceed the compiled collision envelope"
            )
        self.model.geom_size[self.book_collision_geom_id] = half_size
        self.model.geom_pos[self.book_collision_geom_id] = half_size
        self.model.geom_friction[
            self.book_collision_geom_id
        ] = physical_properties.friction

        # The compiled book mesh axes are thickness, width, length.  Its geom
        # frame rotates those axes into the book body frame.
        mesh_dimensions = np.array(
            [
                physical_properties.thickness_m,
                physical_properties.width_m,
                physical_properties.length_m,
            ],
            dtype=float,
        )
        self.model.mesh_vert[self._book_mesh_slice] = (
            self._book_mesh_reference
            * (mesh_dimensions / self._book_mesh_reference_dimensions)
        )
        self._book_mesh_version += 1
        self.model.geom_pos[self.book_visual_geom_id] = half_size
        # Disconnect the brick texture for this geom so RGBA is the actual
        # sampled cover colour rather than a weak tint over a fixed texture.
        self.model.geom_matid[self.book_visual_geom_id] = -1
        self.model.geom_rgba[self.book_visual_geom_id] = self.book_color

        mass = physical_properties.mass_kg
        length, width, thickness = 2.0 * half_size
        self.model.body_mass[self.book_body_id] = mass
        self.model.body_ipos[self.book_body_id] = half_size
        self.model.body_inertia[self.book_body_id] = mass / 12.0 * np.array(
            [
                length * length + width * width,
                length * length + thickness * thickness,
                width * width + thickness * thickness,
            ]
        )

        book_transform = np.asarray(self.scene["book_transform"], dtype=float)
        book_pose = np.concatenate(
            [book_transform[:3, 3], _wxyz_from_matrix(book_transform[:3, :3])]
        )
        self._initial_qpos[
            self.book_free_qpos_adr : self.book_free_qpos_adr + 7
        ] = book_pose
        self._configure_tool_home()
        mujoco.mj_setConst(self.model.ptr, self.data.ptr)
        self.model.geom_rbound[
            self.book_collision_geom_id
        ] = self._book_collision_geom_rbound
        self.model.geom_aabb[
            self.book_collision_geom_id
        ] = self._book_collision_geom_aabb
        self.model.bvh_aabb[:] = self._compiled_bvh_aabb
        self.reset()
        return self

    def _configure_tool_home(self):
        """Hook for the floating controller, whose free joint starts at home."""
        return None

    # ------------------------------------------------------------------ state
    @property
    def tool_pos(self):
        return np.array(self.data.site_xpos[self.tool_site_id], dtype=float)

    @property
    def tool_quat(self):
        return _wxyz_from_matrix(
            np.array(self.data.site_xmat[self.tool_site_id]).reshape(3, 3)
        )

    @property
    def tool_vel(self):
        """Linear velocity of the tool site, world axes (m/s)."""
        res = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model.ptr, self.data.ptr,
            mujoco.mjtObj.mjOBJ_SITE, self.tool_site_id, res, 0,
        )
        return res[3:].copy()          # mj_objectVelocity gives [angular, linear]

    @property
    def book_pos(self):
        return np.array(self.data.body(self.book_body_id).xpos, dtype=float)

    @property
    def book_quat(self):
        return np.array(self.data.body(self.book_body_id).xquat, dtype=float)

    def book_angle_deg(self):
        """Angle of the book's long axis from vertical -- the heuristic's metric."""
        book_x = q2r(self.book_quat)[:, 0]
        return float(np.degrees(np.arccos(np.clip(abs(book_x[2]), -1.0, 1.0))))

    def success(self, threshold_deg=15.0):
        return self.book_angle_deg() < threshold_deg

    # ----------------------------------------------------------------- forces
    def contact_force(self):
        """Net contact force the world applies to the robot, world axes (N).

        Zero in free space by construction. mj_contactForce reports the wrench
        in the contact frame acting on geom2's body, so contacts where the robot
        is geom1 get negated. Verified signed, not by magnitude: driving the
        tool into the bookend gives a force that opposes the tool velocity
        (dot product -0.36) and sits at cosine -0.99 to the raw wrist sensor,
        which is how the sign of ``wrist_force`` below was pinned down.
        """
        return self.contact_wrench(frame="world")[:3]

    def contact_wrench(self, frame="tool"):
        """Solver-exact contact wrench on the robot at the tool origin.

        Force and moment are accumulated from every robot contact. Moments are
        transported from each contact point to ``planner_tip_site``. The order
        is ``[Fx, Fy, Fz, Tx, Ty, Tz]``; ``frame`` may be ``tool`` (Pyrite's
        convention) or ``world``.
        """
        total = np.zeros(6)
        origin = self.tool_pos
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            robot1 = int(self.model.geom_bodyid[contact.geom1]) in self._robot_bodies
            robot2 = int(self.model.geom_bodyid[contact.geom2]) in self._robot_bodies
            if robot1 == robot2:       # neither side, or a robot self-contact
                continue
            mujoco.mj_contactForce(
                self.model.ptr, self.data.ptr, i, self._contact_buf
            )
            contact_to_world = np.asarray(contact.frame).reshape(3, 3).T
            force = contact_to_world @ self._contact_buf[:3]
            torque = contact_to_world @ self._contact_buf[3:]
            if not robot2:
                force = -force
                torque = -torque
            torque += np.cross(np.asarray(contact.pos) - origin, force)
            total[:3] += force
            total[3:] += torque

        if frame == "world":
            return total
        if frame == "tool":
            world_from_tool = np.asarray(
                self.data.site_xmat[self.tool_site_id]
            ).reshape(3, 3)
            return np.concatenate(
                [
                    world_from_tool.T @ total[:3],
                    world_from_tool.T @ total[3:],
                ]
            )
        raise ValueError(f"unknown wrench frame {frame!r}")

    def book_contact_force(self):
        """Net contact force between the fingertip pads and book, world axes (N).

        Unlike ``contact_force``/``contact_wrench``, which sum over EVERY
        contact where one side is a robot body -- an arm link, the gripper
        base, or the wrist camera mount brushing the table/bookend/book all
        count identically -- this isolates specifically the fingertip-pad vs.
        book contact pair, using the same geom filter ``_active_book_normal``
        uses for the anti-windup latch. Use this to check whether a reported
        ``contact_force()``/``force_monitor["contact_max_n"]`` spike is
        genuinely the fingertip touching the book, or an unrelated contact
        elsewhere on the robot being summed into the same total: if this
        number is much smaller than ``contact_force()`` at the same instant,
        the reported force is not coming from the fingertip.
        """
        tip_ids = frozenset(int(g) for g in self.tip_contact_geom_ids)
        book_id = int(self.book_collision_geom_id)
        force = np.zeros(3)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 == book_id and g2 in tip_ids:
                tip_is_geom2 = True
            elif g2 == book_id and g1 in tip_ids:
                tip_is_geom2 = False
            else:
                continue
            mujoco.mj_contactForce(self.model.ptr, self.data.ptr, i, self._contact_buf)
            contact_to_world = np.asarray(contact.frame).reshape(3, 3).T
            contact_force_vec = contact_to_world @ self._contact_buf[:3]
            # mj_contactForce reports the wrench acting on geom2's body; flip
            # when the tip pad is geom1, matching contact_force's convention
            # of "force the world applies to the robot" (see that docstring).
            if not tip_is_geom2:
                contact_force_vec = -contact_force_vec
            force += contact_force_vec
        return force

    def fingertip_contact_force(self):
        """Net contact force ON the fingertip pads, from ANY contact partner.

        Broader than ``book_contact_force``: a fingertip pressed against the
        table or a bookend surface (not just the book) is still force on the
        fingertip, and this includes it. Use this instead of
        ``book_contact_force`` when the question is "is this force actually
        at the fingertip at all", not specifically "is it the book".
        """
        tip_ids = frozenset(int(g) for g in self.tip_contact_geom_ids)
        force = np.zeros(3)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in tip_ids:
                tip_is_geom2 = False
            elif g2 in tip_ids:
                tip_is_geom2 = True
            else:
                continue
            mujoco.mj_contactForce(self.model.ptr, self.data.ptr, i, self._contact_buf)
            contact_to_world = np.asarray(contact.frame).reshape(3, 3).T
            contact_force_vec = contact_to_world @ self._contact_buf[:3]
            if not tip_is_geom2:
                contact_force_vec = -contact_force_vec
            force += contact_force_vec
        return force

    def contact_breakdown(self):
        """Per-contact-pair decomposition of what ``contact_wrench`` sums.

        Returns ``{"<robot geom> vs <other geom>": force_vector_N, ...}`` for
        every currently active robot/non-robot contact, world-frame force,
        signed so it's the force ON the robot side (matching
        ``contact_force``'s convention). This is the tool for actually
        answering "what is producing this number" instead of guessing: sum
        the values whose key starts with a fingertip pad name to reproduce
        ``fingertip_contact_force()``, and everything else is exactly what
        else is loaded and by how much.
        """
        tip_ids = frozenset(int(g) for g in self.tip_contact_geom_ids)
        breakdown = {}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            robot1 = int(self.model.geom_bodyid[g1]) in self._robot_bodies
            robot2 = int(self.model.geom_bodyid[g2]) in self._robot_bodies
            if robot1 == robot2:
                continue
            robot_geom, other_geom = (g2, g1) if robot2 else (g1, g2)
            mujoco.mj_contactForce(self.model.ptr, self.data.ptr, i, self._contact_buf)
            contact_to_world = np.asarray(contact.frame).reshape(3, 3).T
            force = contact_to_world @ self._contact_buf[:3]
            if not robot2:
                force = -force
            robot_name = (
                mujoco.mj_id2name(self.model.ptr, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
                or f"geom{robot_geom}"
            )
            other_name = (
                mujoco.mj_id2name(self.model.ptr, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
                or f"geom{other_geom}"
            )
            key = f"{robot_name} vs {other_name}"
            breakdown[key] = breakdown.get(key, np.zeros(3)) + force
        return breakdown

    def wrist_wrench_raw(self):
        """Untared MuJoCo wrist F/T output in the sensor site's local frame."""
        if self.wrist_force_adr is None or self.wrist_torque_adr is None:
            return np.zeros(6)
        force = np.asarray(
            self.data.sensordata[self.wrist_force_adr : self.wrist_force_adr + 3],
            dtype=float,
        )
        torque = np.asarray(
            self.data.sensordata[self.wrist_torque_adr : self.wrist_torque_adr + 3],
            dtype=float,
        )
        return np.concatenate([force, torque])

    def wrist_wrench(self, frame="tool"):
        """Tared simulated wrist F/T measurement on the robot.

        MuJoCo's sensor sign is negated so this and :meth:`contact_wrench` both
        report the wrench the world applies to the robot. Taring remains in the
        world frame, preserving the existing haptic force behavior.
        """
        if self.wrist_site_id is None:
            return np.zeros(6)
        world_from_sensor = np.asarray(
            self.data.site_xmat[self.wrist_site_id]
        ).reshape(3, 3)
        raw = self.wrist_wrench_raw()
        world = -np.concatenate(
            [
                world_from_sensor @ raw[:3],
                world_from_sensor @ raw[3:],
            ]
        )
        world -= self._wrist_tare
        if frame == "world":
            return world
        if frame == "tool":
            world_from_tool = np.asarray(
                self.data.site_xmat[self.tool_site_id]
            ).reshape(3, 3)
            return np.concatenate(
                [
                    world_from_tool.T @ world[:3],
                    world_from_tool.T @ world[3:],
                ]
            )
        raise ValueError(f"unknown wrench frame {frame!r}")

    def wrist_force(self):
        """WSG50 force sensor in world axes, tared at reset (N).

        Negated relative to the raw sensor so it shares the sign convention of
        ``contact_force`` (force on the robot, i.e. opposing the operator).
        """
        return self.wrist_wrench(frame="world")[:3]

    def estimated_force(self, target_pos):
        """Contact force reconstructed from the actuator side (BallPush's default).

        The task-space controller applies ``tool_kp * err`` at the tool, so
        Newton on the tool gives ``F_contact ~= -F_actuator + damping * v``. The
        minus matters: the actuator force points the way the tool is being
        driven, and reflecting it unnegated assists the operator instead of
        resisting -- that bug shipped once in teleop_ball and felt magnetic.
        """
        err = np.asarray(target_pos, dtype=float) - self.tool_pos
        f_act = np.clip(self.tool_kp * err, -self.force_clip, self.force_clip)
        return -f_act + self.tool_damping * self.tool_vel

    def reflected_force(self, source, target_pos):
        """Sim force to reflect, world axes, clipped to ``force_clip``."""
        if source == "contact":
            force = self.contact_force()
        elif source == "wrist":
            force = self.wrist_force()
        elif source == "estimated":
            force = self.estimated_force(target_pos)
        elif source == "none":
            return np.zeros(3)
        else:
            raise ValueError(f"unknown force source {source!r}")
        magnitude = np.linalg.norm(force)
        if magnitude > self.force_clip:
            force = force * (self.force_clip / magnitude)
        return force

    # ------------------------------------------------------------ tip contact
    def _configure_tip_contact(self):
        """Interpolate only the two fingertip pads toward a softer contact.

        A zero knob leaves the compiled XML values untouched. At one, the
        original 10 ms / damping-ratio 1 / 3 mm contact becomes
        ``tip_softness_max_solref``/``tip_softness_max_width`` (default
        ``SOFT_TIP_SOLREF``/``SOFT_TIP_SOLIMP_WIDTH``, the originally-tested
        20 ms / damping-ratio 2 / 5 mm endpoint). Intermediate values are
        linear. Per the sanding teleop task's investigation
        (SANDING_JITTER_FIX_SUMMARY.md): a longer solref time constant
        lowers the contact's effective bandwidth at DC, not just at the
        onset transient -- so pushing the ceiling past the originally-tested
        endpoint via ``--tip-softness-max-solref`` can reduce *sustained*
        wedging force too, not only impact sharpness, at the cost of some
        genuine steady-state force-sensitivity (this tradeoff is the whole
        point here, unlike sanding where it was an unwanted side effect).
        Mirrors FloatingFlipUpTeleop._configure_tip_contact, duplicated
        rather than shared since the two classes resolve different
        (differently-prefixed) geom names and don't otherwise share a
        constructor path.

        ``tip_friction`` is a separate, independent override (like
        ``bookend_friction`` is to ``bookend_solref``/``bookend_solimp``):
        it replaces the tip pads' compiled friction ``(1.2, 0.01, 0.0005)``
        -- notably high, and the ONE friction value that's actually
        reachable for fingertip-vs-book contact without any priority
        workaround, since the tip pads' priority (8) already wins over the
        book's (unset/0). Unlike ``book_fixture_friction``, there's no
        known load-bearing safety floor here: the fingertip is always
        actively driven by the task-space controller, never passively
        resting under gravity the way the book rests on book_floor, so
        lowering this doesn't risk anything sliding off under its own
        weight -- it only changes how much the fingertip grips/drags
        against the book's surface while sliding.
        """
        base_solref = np.asarray(
            self.model.geom_solref[self.tip_contact_geom_ids], dtype=float
        ).copy()
        base_width = np.asarray(
            self.model.geom_solimp[self.tip_contact_geom_ids, 2], dtype=float
        ).copy()
        softness = self.tip_softness
        if softness > 0.0:
            target_solref = np.broadcast_to(
                np.asarray(self.tip_softness_max_solref, dtype=float), base_solref.shape
            )
            self.model.geom_solref[self.tip_contact_geom_ids] = (
                (1.0 - softness) * base_solref + softness * target_solref
            )
            self.model.geom_solimp[self.tip_contact_geom_ids, 2] = (
                (1.0 - softness) * base_width
                + softness * self.tip_softness_max_width
            )
        if self.tip_friction is not None:
            self.model.geom_friction[self.tip_contact_geom_ids] = self.tip_friction

        resolved_solref = np.asarray(
            self.model.geom_solref[self.tip_contact_geom_ids], dtype=float
        )
        resolved_width = np.asarray(
            self.model.geom_solimp[self.tip_contact_geom_ids, 2], dtype=float
        )
        resolved_friction = np.asarray(
            self.model.geom_friction[self.tip_contact_geom_ids], dtype=float
        )
        self.tip_contact_parameters = {
            "softness": softness,
            "geom_names": list(ARM_TIP_CONTACT_GEOM_NAMES),
            "solref_time_constant_s": float(resolved_solref[0, 0]),
            "solref_damping_ratio": float(resolved_solref[0, 1]),
            "solimp_width_m": float(resolved_width[0]),
            "friction": resolved_friction[0].tolist(),
        }

    def _configure_bookend_contact(self):
        """Optionally override the bookend fixture surfaces' raw contact.

        The three bookend surfaces the robot can touch --
        ``robot_wall_surface``/``robot_pivot_surface``/``robot_floor_surface``
        -- compile at MuJoCo priority 10 or 20, both higher than the
        fingertip pads' priority 8. Since MuJoCo takes contact parameters
        from the higher-priority geom verbatim (not blended), this means
        ``--tip-softness`` has NO effect on fingertip-vs-bookend contact --
        same silent-scope gap the sanding task's ``--pad-softness`` bug
        exposed (there it was an unintended no-op; here it's simply outside
        ``--tip-softness``'s documented scope, which only ever covered the
        book). This is the lever that actually reaches bookend contact.
        Mirrors ``_configure_table_friction``/(floating's)
        ``_configure_table_contact``. Any of the three params may be left
        ``None`` to keep the compiled per-surface defaults.
        """
        names = (
            "bookend2_blender/robot_wall_surface",
            "bookend2_blender/robot_pivot_surface",
            "bookend2_blender/robot_floor_surface",
        )
        geom_ids = [self.model.geom(name).id for name in names]
        for geom_id in geom_ids:
            if self.bookend_solref is not None:
                self.model.geom_solref[geom_id] = self.bookend_solref
            if self.bookend_solimp is not None:
                self.model.geom_solimp[geom_id] = self.bookend_solimp
            if self.bookend_friction is not None:
                self.model.geom_friction[geom_id] = self.bookend_friction
        self.bookend_contact_parameters = {
            "geom_names": list(names),
            "solref": [
                np.asarray(self.model.geom_solref[g], dtype=float).tolist()
                for g in geom_ids
            ],
            "solimp": [
                np.asarray(self.model.geom_solimp[g], dtype=float).tolist()
                for g in geom_ids
            ],
            "friction": [
                np.asarray(self.model.geom_friction[g], dtype=float).tolist()
                for g in geom_ids
            ],
        }

    def _configure_book_fixture_contact(self):
        """Optionally override the bookend fixture's BOOK-facing surfaces.

        Distinct from ``_configure_bookend_contact``: that method softens
        ``robot_wall_surface``/``robot_pivot_surface``/``robot_floor_surface``
        (priority=20), the surfaces the *fingertip* touches. This method
        targets ``book_wall``/``book_pivot``/``book_floor`` (priority=10),
        the surfaces the *book itself* swings into as it approaches the
        fixture -- e.g. near the top of a flip, when the book's edge nears
        vertical and can contact multiple fixture surfaces within a few ms.

        These compile stiff (``solref=(0.01, 1.0)``, i.e. a 10ms time
        constant, the same order as the un-softened contact that caused the
        original sanding-task oscillation) and were not reachable by any
        prior flag: ``--book-friction`` writes onto the book geom itself,
        which is priority 0/unset and so loses to these priority-10 surfaces
        for book-vs-fixture contact, the same silent-scope gap
        ``--bookend-solref`` was added to close for fingertip-vs-fixture
        contact. A multi-contact impact here (several of these surfaces
        engaging within 1-2 physics steps while the book still has real
        angular velocity) produces a large transient spike that neither
        ``--tip-softness`` nor ``--book-normal-force-limit`` can prevent --
        the former doesn't reach this geom pair at all, and the latter only
        throttles the *operator's target* for sustained wedging, acting too
        slowly to catch a multi-contact impact that resolves within a
        control tick.

        DANGER, ``book_fixture_solref`` specifically: unlike the robot-facing
        bookend surfaces (which the fingertip actively pushes on, so
        softening them is always safe), ``book_pivot``/``book_floor``
        passively hold the book's static weight up against gravity at rest
        -- no active control compensates for it, and they also have to
        arrest real dynamic impacts, not just hold a static load.
        Lengthening solref's time constant looks like a good trade at
        first, then gets worse the more carefully it's checked:

        1. Steady-state stiffness (same mechanism as the sanding task's
           contact tuning) -- push it far enough and the support can't
           hold the book's weight at all; it sinks through and free-falls
           for the rest of the settle window at REST, no velocity needed
           ((0.06, 2.0) alone sends the book ~20m underground, matching
           ``0.5 * g * settle_s**2``).
        2. Tunneling under a hard impact -- (0.02, 2.0) holds at rest but
           punches clean through book_floor at 1.0+ m/s (vs. 3.0 m/s
           compiled). Tempting to accept, since that only bites on an
           already-dropped book.
        3. The actual deal-breaker, found by checking below the full-
           tunnel threshold rather than stopping at #2: (0.02, 2.0)
           doesn't fail cleanly at more moderate velocity, it WEDGES.
           Between 0.6 and 0.9 m/s -- an ordinary jostle/release speed,
           reached easily via --scale amplifying normal hand motion,
           nowhere near "already dropped it" -- the book sinks 1-4m into
           the floor and gets stuck there instead of tunneling through or
           bouncing back. That doesn't cost one bad episode, it makes the
           task physically unable to proceed at all until reset. Strictly
           worse than the jam-force chatter this was meant to fix.

        Net: there is no tested value above the compiled 0.01 that's
        actually usable -- the jam-force win (mean 103N compiled -> 25N at
        (0.02, 2.0) on the recorded jam replay, a real and large
        reduction) isn't safely reachable through this lever, so solref
        stays compiled. ``book_fixture_solimp``'s width is a smaller,
        genuinely safe lever on its own (~7% jam-force cut, no tunneling
        or wedging cost at all -- it matches the compiled solref's
        behavior across every velocity tested).

        ``book_fixture_friction`` was tried too and left at the compiled
        value -- not because it lacks a safety floor of its own (it does:
        these surfaces support the book at an angle, so sliding friction
        is partly load-bearing, and cutting it enough to matter, ~0.1,
        drifted the book 1.8m sideways in 3s of doing nothing and failed
        the random-book-size settle check 14/30), but because even a
        floor-respecting value (0.6) turned out to depend on
        ``book_fixture_solref`` also being softened to help at all.
        Checked against the solref value actually shipped above
        (compiled), the same 0.6 made the recorded jam replay WORSE (mean
        102.8N unmodified -> 132.9N at friction=0.6 alone) -- friction was
        providing tangential damping that stabilizes the multi-contact
        chatter specifically when normal contact is soft; with normal
        contact back at compiled stiffness, removing that damping just
        makes it worse. Not a safe, portable lever on its own.

        Any of the three params may be left ``None`` to keep the compiled
        per-surface defaults.
        """
        names = (
            "bookend2_blender/book_wall",
            "bookend2_blender/book_pivot",
            "bookend2_blender/book_floor",
        )
        geom_ids = [self.model.geom(name).id for name in names]
        for geom_id in geom_ids:
            if self.book_fixture_solref is not None:
                self.model.geom_solref[geom_id] = self.book_fixture_solref
            if self.book_fixture_solimp is not None:
                self.model.geom_solimp[geom_id] = self.book_fixture_solimp
            if self.book_fixture_friction is not None:
                self.model.geom_friction[geom_id] = self.book_fixture_friction
        self.book_fixture_contact_parameters = {
            "geom_names": list(names),
            "solref": [
                np.asarray(self.model.geom_solref[g], dtype=float).tolist()
                for g in geom_ids
            ],
            "solimp": [
                np.asarray(self.model.geom_solimp[g], dtype=float).tolist()
                for g in geom_ids
            ],
            "friction": [
                np.asarray(self.model.geom_friction[g], dtype=float).tolist()
                for g in geom_ids
            ],
        }

    def _configure_table_friction(self):
        """Optionally override the visible table surface's raw friction.

        MuJoCo combines two contacting geoms' friction by taking the
        higher-priority geom's values verbatim, not an average, when
        priorities differ. table.xml compiles the table surface at
        priority=20; the book geom is unset (priority 0). That means the
        TABLE's friction governs book-table sliding contact regardless of
        --book-friction -- this is the knob that actually controls it. Left
        at None (default), the compiled table.xml friction (0.15, 0.003,
        0.0001) is unchanged.
        """
        table_geom_id = self.model.geom("table/table_surface").id
        if self.table_friction is not None:
            self.model.geom_friction[table_geom_id] = self.table_friction
        self.table_friction_parameters = {
            "friction": np.asarray(
                self.model.geom_friction[table_geom_id], dtype=float
            ).tolist(),
        }

    # --------------------------------------------------------- surface safety
    def _init_surface_safety(self):
        names = (
            "table/table_surface",
            "bookend2_blender/robot_wall_surface",
            "bookend2_blender/robot_pivot_surface",
            "bookend2_blender/robot_floor_surface",
        )
        self._surface_guard_geom_ids = frozenset(
            self.model.geom(name).id for name in names
        )
        # Approach-compliance (speed cap + kp ramp) additionally covers the
        # book -- unlike _surface_guard_geom_ids, which the REACTIVE
        # anti-windup (surface_safe_target, and on the floating gripper
        # book_normal_safe_target) deliberately keeps book-free, since
        # capping sustained book-normal force there would fight the task.
        # This set only feeds the two approach-compliance geometry lookups
        # (_nearest_guarded_surface / _nearest_guarded_surface_distance),
        # used pre-contact to shape approach speed/stiffness, not to cap
        # force after contact. book_collision_geom_id is resolved before
        # this runs in both FlipUpTeleop and FloatingFlipUpTeleop.
        #
        # Caveat: the box-face math below assumes local +Z is the relevant
        # outward contact normal, which is only true for the book while it's
        # still resting near its start pose (confirmed against real data:
        # the impact spike this targets occurs while book_angle is still
        # ~89.5-90 deg, i.e. before any rotation). Once the flip is actually
        # underway the book's rotating orientation means +Z no longer tracks
        # the true contact face, so this approximation degrades through the
        # rest of the flip -- it is deliberately scoped to the first-contact
        # window, not a general per-instant book-normal admittance.
        self._approach_guard_geom_ids = self._surface_guard_geom_ids | frozenset(
            {int(self.book_collision_geom_id)}
        )
        self._surface_limit_normal = None
        self._surface_limit_boundary = None
        self._surface_contact_misses = 0
        self._surface_contact_grace_steps = max(
            1, int(round(0.020 / float(self.model.opt.timestep)))
        )
        self._requested_target = np.asarray(self.scene["prepare"], dtype=float).copy()
        self._drive_target = self._requested_target.copy()

    def _active_surface_normal(self):
        """Return the average outward normal of protected robot contacts."""
        normals = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if not self._surface_guard_geom_ids.intersection(
                (int(contact.geom1), int(contact.geom2))
            ):
                continue
            robot1 = int(self.model.geom_bodyid[contact.geom1]) in self._robot_bodies
            robot2 = int(self.model.geom_bodyid[contact.geom2]) in self._robot_bodies
            if robot1 == robot2:
                continue
            normal = np.asarray(contact.frame, dtype=float).reshape(3, 3)[0]
            normals.append(normal if robot2 else -normal)
        if not normals:
            return None
        normal = np.mean(normals, axis=0)
        magnitude = np.linalg.norm(normal)
        return None if magnitude < 1e-9 else normal / magnitude

    def surface_safe_target(self, target_pos):
        """Bound stored normal spring energy after visible-surface contact.

        The operator can still slide tangentially and can pull away immediately.
        Only the component continuing through the contacted table/support surface
        is capped, at ``surface_force_limit / tool_kp`` metres of deflection.
        """
        target = np.asarray(target_pos, dtype=float)
        if self.surface_force_limit <= 0.0:
            return target.copy()

        active_normal = self._active_surface_normal()
        if self._surface_limit_normal is not None:
            if active_normal is None:
                self._surface_contact_misses += 1
                if self._surface_contact_misses > self._surface_contact_grace_steps:
                    self._surface_limit_normal = None
                    self._surface_limit_boundary = None
                    self._surface_contact_misses = 0
            else:
                self._surface_contact_misses = 0
        if self._surface_limit_normal is None and active_normal is not None:
            if np.dot(target - self.tool_pos, active_normal) < 0.0:
                self._surface_limit_normal = active_normal
                self._surface_limit_boundary = float(
                    np.dot(self.tool_pos, active_normal)
                )
                self._surface_contact_misses = 0

        normal = self._surface_limit_normal
        if normal is None:
            return target.copy()
        target_coordinate = float(np.dot(target, normal))
        if target_coordinate >= self._surface_limit_boundary:
            self._surface_limit_normal = None
            self._surface_limit_boundary = None
            self._surface_contact_misses = 0
            return target.copy()

        normal_error = float(np.dot(target - self.tool_pos, normal))
        # Deliberately the static tool_kp, NOT the approach-compliance-
        # softened effective kp: this budget is a hard safety cap on
        # penetration, and letting it grow as kp is ramped down would let
        # sustained pushing drive the target far deeper before the force
        # limit engages (e.g. 15 N / (0.2*700 N/m) = 10.7 cm instead of
        # 15 N / 700 N/m = 2.1 cm).
        max_deflection = self.surface_force_limit / self.tool_kp
        if normal_error >= -max_deflection:
            return target.copy()
        return target + (-max_deflection - normal_error) * normal

    # ------------------------------------------------------ book-normal safety
    def _init_book_safety(self):
        """State for the opt-in fingertip-book normal anti-windup.

        Separate from ``_init_surface_safety`` because the visible-surface
        guard (table/bookend) deliberately excludes the book -- see
        FLOATING_FLIPUP_COMPLIANCE_TELEOP.md section 10.1. Reuses
        ``_surface_contact_grace_steps`` for the release debounce so both
        latches release on the same ~20 ms miss window. Identical to
        FloatingFlipUpTeleop's version of the same name; duplicated rather
        than shared since the two classes don't otherwise share a
        constructor path.
        """
        self._book_limit_normal = None
        self._book_limit_boundary = None
        self._book_contact_misses = 0

    def _active_book_normal(self):
        """Average outward normal of active fingertip-pad/book contacts.

        Mirrors ``_active_surface_normal``'s geom-pair scan and sign
        convention, but filters on ``tip_contact_geom_ids`` vs.
        ``book_collision_geom_id`` instead of the table/bookend allowlist.
        """
        tip_ids = frozenset(int(g) for g in self.tip_contact_geom_ids)
        book_id = int(self.book_collision_geom_id)
        normals = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 == book_id and g2 in tip_ids:
                tip_is_geom2 = True
            elif g2 == book_id and g1 in tip_ids:
                tip_is_geom2 = False
            else:
                continue
            normal = np.asarray(contact.frame, dtype=float).reshape(3, 3)[0]
            normals.append(normal if tip_is_geom2 else -normal)
        if not normals:
            return None
        normal = np.mean(normals, axis=0)
        magnitude = np.linalg.norm(normal)
        return None if magnitude < 1e-9 else normal / magnitude

    def book_normal_safe_target(self, target_pos):
        """Bound stored normal spring energy after fingertip-book contact.

        Same latch/debounce/release state machine as ``surface_safe_target``
        (see that docstring), applied to the book instead of the visible
        table/bookend surfaces. Off by default: 0 disables it, matching the
        ``surface_force_limit`` convention. Tangential motion, and any
        table/bookend contact handled by ``surface_safe_target``, are
        unaffected.
        """
        target = np.asarray(target_pos, dtype=float)
        if self.book_force_limit <= 0.0:
            return target.copy()

        active_normal = self._active_book_normal()
        if self._book_limit_normal is not None:
            if active_normal is None:
                self._book_contact_misses += 1
                if self._book_contact_misses > self._surface_contact_grace_steps:
                    self._book_limit_normal = None
                    self._book_limit_boundary = None
                    self._book_contact_misses = 0
            else:
                self._book_contact_misses = 0
        if self._book_limit_normal is None and active_normal is not None:
            if np.dot(target - self.tool_pos, active_normal) < 0.0:
                self._book_limit_normal = active_normal
                self._book_limit_boundary = float(
                    np.dot(self.tool_pos, active_normal)
                )
                self._book_contact_misses = 0

        normal = self._book_limit_normal
        if normal is None:
            return target.copy()
        target_coordinate = float(np.dot(target, normal))
        if target_coordinate >= self._book_limit_boundary:
            self._book_limit_normal = None
            self._book_limit_boundary = None
            self._book_contact_misses = 0
            return target.copy()

        normal_error = float(np.dot(target - self.tool_pos, normal))
        # Static tool_kp on purpose -- see the matching comment in
        # surface_safe_target above.
        max_deflection = self.book_force_limit / self.tool_kp
        if normal_error >= -max_deflection:
            return target.copy()
        return target + (-max_deflection - normal_error) * normal

    @property
    def book_limit_active(self):
        return self._book_limit_normal is not None

    # ------------------------------------------------------ approach safety
    def _nearest_guarded_surface(self, pos):
        """(distance, outward_normal) to the closest guarded surface.

        Guarded surfaces are the table/bookend/book box geoms tracked by
        ``_approach_guard_geom_ids`` (see ``_init_surface_safety``) -- unlike
        ``_surface_guard_geom_ids``, used by the REACTIVE anti-windup
        (``surface_safe_target``), this set includes the book so approach
        compliance can soften the first-contact impact. See
        ``_init_surface_safety`` for the caveat on why the book's inclusion
        here is only a good approximation near the book's start pose.
        Each is assumed to be a box whose local +Z axis is its outward
        contact normal -- true for the table and bookend fixtures modeled
        here. Non-box guarded geoms are skipped. Returns ``(None, None)`` if
        no guarded box geom exists yet or the allowlist contains no box
        geoms. Distance is positive above the surface, negative if already
        penetrating it. Identical to FloatingFlipUpTeleop's version of the
        same name; duplicated rather than shared since the two classes don't
        otherwise share a constructor path.
        """
        guard_ids = getattr(self, "_approach_guard_geom_ids", None)
        if not guard_ids:
            return None, None
        pos = np.asarray(pos, dtype=float)
        nearest_distance, nearest_normal = None, None
        for geom_id in guard_ids:
            if int(self.model.geom_type[geom_id]) != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            distance, normal = self._box_face_distance(geom_id, pos)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance, nearest_normal = distance, normal
        return nearest_distance, nearest_normal

    def _box_face_distance(self, geom_id, pos):
        """(distance, outward_normal) from ``pos`` to a box geom's nearest face.

        Picks whichever of the box's 6 faces actually faces ``pos``, rather
        than assuming local +Z. The table/bookend fixtures were authored so
        local +Z is always their real exposed top face, which is why the
        original version of this method (before the book was added to
        ``_approach_guard_geom_ids``) hardcoded +Z. That assumption breaks
        for the book: its orientation varies with episode randomization and
        rotates through the flip, and was measured to have local +Z pointing
        straight DOWN in world frame at one perfectly ordinary start pose --
        hardcoding +Z there silently picked the book's underside as
        "outward" and reported the tool as already penetrating it at rest.

        The correct rule (standard point-to-box distance): transform ``pos``
        into the box's local frame, and per axis compute how far the point's
        |local coordinate| protrudes past that axis's half-extent. The axis
        with the largest such protrusion is the face the point is furthest
        outside of, and that protrusion IS the signed distance to that face
        (positive outside, negative if the point is within that axis's span
        but still outside another). An earlier version of this method picked
        the face by raw alignment (dot of the center-to-point vector with
        each candidate normal) instead -- that is NOT a valid nearest-face
        rule for a point whose offset is dominated by an axis the box is
        actually narrow along (exactly the book's case, small and off to the
        side of the tool's home position): a large raw coordinate difference
        along an axis can win the alignment comparison even when that axis
        isn't the one the point is actually protruding past, producing
        spuriously large negative "distances" unrelated to real proximity.
        Normalizing by each axis's own half-extent (below) fixes that.
        """
        half_extents = np.asarray(self.model.geom_size[geom_id], dtype=float)
        center = np.asarray(self.data.geom_xpos[geom_id], dtype=float)
        rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        local = rotation.T @ (pos - center)
        excess = np.abs(local) - half_extents
        axis = int(np.argmax(excess))
        sign = 1.0 if local[axis] >= 0.0 else -1.0
        normal = sign * rotation[:, axis]
        return float(excess[axis]), normal

    def _nearest_guarded_surface_distance(self, pos):
        """Signed distance (m) from ``pos`` to the closest guarded surface.

        See ``_nearest_guarded_surface`` for the sign convention and box
        assumption.
        """
        distance, _ = self._nearest_guarded_surface(pos)
        return distance

    def _approach_speed_limited_target(self, target_pos):
        """Cap how fast the incoming target may move THROUGH a guarded surface.

        This is the mechanism that actually controls impact kinetic energy --
        see FloatingFlipUpTeleop._approach_speed_limited_target's docstring
        for why softening kp alone (below) does not. Off unless
        ``approach_max_speed_mps > 0``. Only the component of the requested
        step (relative to the previous drive target) pointing INTO the
        nearest guarded surface is clamped, and only once within
        ``approach_compliance_distance_m`` of it -- outward and tangential
        motion, and everything outside the band, are unaffected. Scope is
        the same table/bookend allowlist as ``surface_safe_target``: this
        does NOT slow an approach toward the book.
        """
        target = np.asarray(target_pos, dtype=float)
        if not self._approach_compliance_enabled or self.approach_max_speed_mps <= 0.0:
            return target
        distance, normal = self._nearest_guarded_surface(self.tool_pos)
        if distance is None or distance > self.approach_compliance_distance_m:
            return target
        max_step = self.approach_max_speed_mps * float(self.model.opt.timestep)
        delta = target - self._drive_target
        inward = float(np.dot(delta, -normal))
        if inward <= max_step:
            return target
        return self._drive_target + delta - (inward - max_step) * (-normal)

    def _compute_effective_translation_kp(self, pos):
        """Translational kp (WORLD xyz, a 3-vector), scheduled by approach
        compliance and tool_kp_axes.

        Base per-axis stiffness is ``tool_kp * tool_kp_axes`` -- (1,1,1) is
        the original isotropic scalar behavior. Off by default
        (``approach_compliance_distance_m == 0``), in which case this always
        returns that base vector unchanged. When enabled, EVERY axis is
        ramped by the same scalar ``ratio`` (distance-based only, not
        force/axis-aware) from 1.0 at ``approach_compliance_distance_m`` (or
        farther) above the nearest guarded surface down to
        ``approach_compliance_min_kp_ratio`` at, or below (already
        penetrating), the surface -- i.e. approach compliance scales
        whatever anisotropic profile tool_kp_axes already set up, it does
        not add its own axis-selectivity.

        Unlike FloatingFlipUpTeleop's version, this does not also recompute a
        matching kd: the arm's translational damping is a JOINT-space term
        (``task_space_kd``, tuned against arm inertia/actuator saturation),
        not a Cartesian velocity term derived in closed form from a moving
        mass, so there is no equivalent closed-form kd to re-derive here.
        Reducing kp while leaving joint damping fixed moves the system
        toward more overdamped, not less -- this is the safe direction, so
        leaving joint damping alone as kp ramps down does not introduce a new
        stability risk.
        """
        base = self.tool_kp * self.tool_kp_axes
        if not self._approach_compliance_enabled:
            return base
        distance = self._nearest_guarded_surface_distance(pos)
        if distance is None:
            ratio = 1.0
        else:
            clipped = float(np.clip(distance, 0.0, self.approach_compliance_distance_m))
            min_ratio = self.approach_compliance_min_kp_ratio
            ratio = min_ratio + (1.0 - min_ratio) * (
                clipped / self.approach_compliance_distance_m
            )
        return ratio * base

    @property
    def requested_target(self):
        return self._requested_target.copy()

    @property
    def drive_target(self):
        return self._drive_target.copy()

    @property
    def surface_limit_active(self):
        return self._surface_limit_normal is not None

    @property
    def home_rotvec(self):
        """Tool orientation at the start pose, as a rotation vector.

        The reference the operator's wrist rotations compose onto when 6-DoF
        control is enabled. Named to match PivotArm's property of the same name.
        """
        from scipy.spatial.transform import Rotation

        return Rotation.from_matrix(
            np.asarray(tool_orientation(self.tool_home))
        ).as_rotvec()

    # ------------------------------------------------------------------ drive
    def target_pose7(self, target_pos, target_rotvec=None):
        """xyz + wxyz pose for a commanded tool position.

        Orientation is derived from the position the way the scripted heuristic
        derives it, unless ``target_rotvec`` supplies one (6-DoF teleoperation).
        """
        from scipy.spatial.transform import Rotation

        target_pos = np.asarray(target_pos, dtype=float)
        if target_rotvec is None:
            rot = np.asarray(tool_orientation(target_pos))
        else:
            rot = Rotation.from_rotvec(
                np.asarray(target_rotvec, dtype=float)
            ).as_matrix()
        return np.concatenate([target_pos, _wxyz_from_matrix(rot)])

    def limited_target(self, target_pos):
        """Target with the commanded interaction force smoothly saturated.

        The controller applies ``tool_kp * (target - tool)``, so bounding that
        product bounds the force. Rewriting the target instead of the wrench keeps
        the parent's controller untouched. ``F = limit * tanh(kp*err/limit)`` is
        used rather than a hard clamp so the knee is differentiable -- a corner
        there is its own source of chatter as the operator rides the boundary.
        See DEFAULT_TOOL_FORCE_LIMIT for why this exists.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        if self.tool_force_limit <= 0.0:
            return target_pos
        err = target_pos - self.tool_pos
        raw = self.tool_kp * np.linalg.norm(err)
        if raw < 1e-9:
            return target_pos
        squashed = self.tool_force_limit * np.tanh(raw / self.tool_force_limit)
        return self.tool_pos + err * (squashed / raw)

    def step(self, target_pos, n_substeps=1, target_rotvec=None):
        """Advance ``n_substeps`` x 1 ms with the tool driven toward the target."""
        for _ in range(max(1, int(n_substeps))):
            self._requested_target = np.asarray(target_pos, dtype=float).copy()
            # Schedule translational kp off the tool's CURRENT position; this
            # only shapes tracking stiffness, not approach speed (see
            # _approach_speed_limited_target's docstring for why that
            # distinction matters). task_space_kp is read directly by
            # step_task_space, so mutate it in place before calling that.
            self._effective_translation_kp = self._compute_effective_translation_kp(
                self.tool_pos
            )
            self.task_space_kp[0, 0], self.task_space_kp[1, 1], self.task_space_kp[2, 2] = (
                self._effective_translation_kp
            )
            speed_limited_target = self._approach_speed_limited_target(
                self._requested_target
            )
            self._drive_target = self.limited_target(
                self.book_normal_safe_target(
                    self.surface_safe_target(speed_limited_target)
                )
            )
            self.step_task_space(
                self.target_pose7(self._drive_target, target_rotvec)
            )
        return self

    def reset(self):
        super().reset()
        if not self._teleop_ready:
            return
        self._surface_limit_normal = None
        self._surface_limit_boundary = None
        self._surface_contact_misses = 0
        if hasattr(self, "_book_limit_normal"):
            self._book_limit_normal = None
            self._book_limit_boundary = None
            self._book_contact_misses = 0
        self._requested_target = self.tool_home.copy()
        self._drive_target = self.tool_home.copy()
        self._wrist_tare = np.zeros(6)
        # step() mutates task_space_kp's translational diagonal in place for
        # the approach-compliance ramp (see _compute_effective_translation_kp)
        # and never restores it -- without resetting it here, an episode that
        # ends near a guarded surface (table/bookend/book) leaves the NEXT
        # episode's settle loop below running at a softened kp, which
        # converges weaker/slower and can inflate settle_error into spurious
        # start-pose resample rejections.
        self._effective_translation_kp = self.tool_kp * self.tool_kp_axes
        self.task_space_kp[0, 0], self.task_space_kp[1, 1], self.task_space_kp[2, 2] = (
            self._effective_translation_kp
        )
        # Slew the target from the arm's joint home to the operator's start pose
        # instead of stepping it there: a 17 cm jump commands ~2.7 kN through a
        # 16 kN/m spring, which saturates the joints and leaves the arm 45 mm
        # off target even after seconds of settling.
        if self.settle_s > 0.0:
            target = self.tool_pos.copy()
            for _ in range(int(self.settle_s / self.timestep)):
                delta = self.tool_home - target
                distance = np.linalg.norm(delta)
                if distance > self.settle_speed * self.timestep:
                    target = target + delta * (
                        self.settle_speed * self.timestep / distance
                    )
                else:
                    target = self.tool_home.copy()
                self.step_task_space(self.target_pose7(target))
                if distance < 1e-4 and np.linalg.norm(
                    self.data.qvel[self.joint_dof_ids]
                ) < 1e-2:
                    break
        self.settle_error = float(np.linalg.norm(self.tool_pos - self.tool_home))
        # Tare the wrist sensor at rest: it reads the gripper's own weight there,
        # which is not information about contact.
        if self.wrist_force_adr is not None:
            self._wrist_tare = np.zeros(6)
            self._wrist_tare = self.wrist_wrench(frame="world")
        self.data.time = 0.0

    # ----------------------------------------------------------------- render
    # A visualization group that MuJoCo's default mjvOption leaves switched off,
    # so moving a geom into it hides it from every render mode used here.
    _HIDDEN_GROUP = 5

    def set_arm_visual(self, mode="full", ghost_alpha=0.22):
        """Show, ghost or hide the UR5e's links, keeping the WSG50 visible.

        Looking down the robot's own reach direction is otherwise dominated by the
        forearm (measured: 39% of the frame at azimuth 0). This only touches
        visualization fields -- ``geom_group`` and ``geom_rgba`` play no part in
        collision detection, which keys off contype/conaffinity -- so the physics
        is bit-for-bit unchanged.
        """
        if not hasattr(self, "_arm_visual_saved"):
            self._arm_visual_saved = (self.model.geom_group.copy(),
                                     self.model.geom_rgba.copy())
        groups, rgba = self._arm_visual_saved
        self.model.geom_group[:] = groups          # start from the compiled state
        self.model.geom_rgba[:] = rgba
        if mode == "full":
            return self
        arm = [
            g for g in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model.ptr, mujoco.mjtObj.mjOBJ_BODY,
                                  int(self.model.geom_bodyid[g])) or "")
            .startswith("ur5e/")
            and "wsg50" not in (mujoco.mj_id2name(
                self.model.ptr, mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[g])) or "")
        ]
        if mode == "hidden":
            self.model.geom_group[arm] = self._HIDDEN_GROUP
        elif mode == "ghost":
            # Materials win over the compiled default geom rgba, so clear the
            # material link for these geoms; they render in their rgba colour,
            # which is what carries the alpha.
            if not hasattr(self, "_arm_matid_saved"):
                self._arm_matid_saved = self.model.geom_matid.copy()
            self.model.geom_matid[arm] = -1
            self.model.geom_rgba[arm, 3] = float(ghost_alpha)
        else:
            raise ValueError(f"unknown arm visual mode {mode!r}")
        self.arm_visual = mode
        return self

    def camera_names(self):
        """Fixed cameras compiled into the scene (the wrist RealSense, mostly)."""
        return [
            mujoco.mj_id2name(self.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(self.model.ncam)
        ]

    def make_camera(self, width=640, height=480, quality="fast",
                    azimuth=-30.0, elevation=-25.0, distance=0.75, lookat=None,
                    camera=None):
        """Return ``render() -> HxWx3 RGB`` for a camera on this scene.

        ``camera`` names a fixed camera (e.g. ``ur5e/wsg50/d435i/rgb``, the wrist
        RealSense -- literally the robot's own viewpoint, but it moves with the
        tool so the book appears static while the world rotates around it). When
        it is None a free camera is placed from azimuth/elevation/distance.

        Safe to call the returned function from a viewer thread while the main
        thread steps: it only reads model/data, and a torn frame is harmless.
        Soak-tested at 22.7 fps against 145k concurrent sim steps.

        ``quality``: ``full`` (~75 ms/frame), ``fast`` (reflections off, ~56 ms --
        the groundplane's 0.2 reflectance costs 20 ms, shadow and skybox only
        ~2 ms each, so those stay on), ``collision`` (collision geoms only, ~9 ms
        -- ugly, but the only mode that keeps up with a fast control loop; the
        cost is the high-resolution meshes, not the resolution).
        """
        from dm_control.mujoco.engine import Camera, MovableCamera
        from dm_control.mujoco import wrapper

        if camera is not None:
            view = Camera(self.physics, height=int(height), width=int(width),
                          camera_id=camera)
        else:
            if lookat is None:
                lookat = 0.5 * (self.scene["engage"] + self.scene["waypoints"][-1])
            view = MovableCamera(self.physics, height=int(height), width=int(width))
            view.set_pose(np.asarray(lookat, dtype=float), float(distance),
                          float(azimuth), float(elevation))

        # Leave geomgroup at MuJoCo's default (visual groups on, collision group
        # 3 off): turning every group on draws the collision boxes over the
        # meshes, which makes the scene unreadable.
        options = wrapper.MjvOption()
        overrides = {}
        if quality == "collision":
            options.geomgroup[:] = 0
            options.geomgroup[3] = 1
        elif quality not in ("fast", "full"):
            raise ValueError(f"unknown render quality {quality!r}")
        if quality in ("fast", "collision"):
            overrides = {mujoco.mjtRndFlag.mjRND_REFLECTION: False}

        def render():
            uploaded_version = getattr(render, "book_mesh_version", -1)
            if uploaded_version != self._book_mesh_version:
                contexts = self.physics.contexts
                with contexts.gl.make_current() as context:
                    context.call(
                        mujoco.mjr_uploadMesh,
                        self.model.ptr,
                        contexts.mujoco.ptr,
                        self.book_mesh_id,
                    )
                render.book_mesh_version = self._book_mesh_version
            return view.render(scene_option=options,
                              render_flag_overrides=overrides)

        render.camera = view
        return render
