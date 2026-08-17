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
    ):
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

        self.task_space_kp = np.diag(
            [self.tool_kp] * 3 + [self.tool_rot_kp] * 3
        ).astype(float)
        self.task_space_kd = (
            DEFAULT_JOINT_KD * DEFAULT_ARM_DAMPING if joint_kd is None
            else np.broadcast_to(np.asarray(joint_kd, dtype=float), (6,)).copy()
        )
        self.task_space_cartesian_kd = np.array(
            [0.0, 0.0, 0.0] + [self.tool_rot_kd] * 3,
            dtype=float,
        )

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
        max_deflection = self.surface_force_limit / self.tool_kp
        if normal_error >= -max_deflection:
            return target.copy()
        return target + (-max_deflection - normal_error) * normal

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
            self._drive_target = self.limited_target(
                self.surface_safe_target(self._requested_target)
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
        self._requested_target = self.tool_home.copy()
        self._drive_target = self.tool_home.copy()
        self._wrist_tare = np.zeros(6)
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
