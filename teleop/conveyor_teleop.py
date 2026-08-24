"""
Adapter that makes the conveyor_minimal pick-place scene drivable by the omega,
with the same haptic pipeline teleop_ball.py and flipup_teleop.py use.

conveyor_minimal contains the UR5e + WSG50 grasping a cube off a moving belt and
placing it in a bin. This adapter subclasses ``ConveyorEnv`` and adds the same
interface ``FlipUpTeleop`` adds to ``FlipUpEnv``, so the device code, the force
rendering, the viewer and the recorder carry over.

What this adds on top of it
---------------------------
* **A Cartesian-position interface.** The operator commands the WSG50 tip
  position; wrist orientation is the task's fixed top-down grasp
  (``GRASP_QUAT_WXYZ``) unless 6-DoF control supplies one, so the 3-DoF omega is
  enough. Same contract as ``FlipUpTeleop.target_pose7``.
* **A gripper channel**, which FlipUp does not have: FlipUp is a nonprehensile
  pivot with the gripper closed throughout, whereas this task is a grasp. The
  operator's grip axis (omega.7) or button (omega.6) drives the commanded finger
  width, and :meth:`grip_force` returns the fingertip normal load so a device
  with a force-reflecting grip axis can render it.
* **Force sources**, all reporting *the force the world applies to the robot* in
  world axes, so a positive reading always opposes the motion that caused it:
    - ``contact``   the solver's per-contact forces summed over every contact
                    involving the robot. Exactly zero in free space by
                    construction, which is why it is the default here as in
                    ``flipup_teleop``.
    - ``wrist``     the WSG50's own MuJoCo F/T sensor, tared at reset and negated
                    (see :meth:`wrist_wrench`). What a real wrist sensor reads.
    - ``estimated`` ``-clip(tool_kp * (target - tool)) + tool_damping * tool_vel``,
                    BallPush's default reconstructed from the actuator side.
                    Kept for A/B comparison only: for an arm this stiff it
                    renders the free-space tracking lag times tool_kp, which is
                    a large force with nothing touching anything.
    - ``none``      no force feedback; the decisive A/B when something feels off.
* **The belt as a force source in its own right.** Once the fingers hold the
  cube, the belt keeps pulling it at up to ``mu * normal_load`` (see
  ``ConveyorEnv._drive_conveyor``), and that pull arrives through the grasp as a
  +y force on the tool. It is the physical signal an operator needs in order to
  grasp off a moving belt, and it is why this environment models the belt with
  friction instead of the source task's kinematic teleport.
* **Randomization that survives a reset.** ``ConveyorEnv.reset`` draws a new belt
  speed, layout offset and cube spawn pose from ``(seed, episode_index)``, and
  this class advances the episode index on every reset, so an operator collecting
  a session sees a new speed every episode. :attr:`conveyor_speed_m_per_s` is the
  current one, and it is worth logging into every episode's metadata.
* **Success + observables** for logging, from ``ConveyorJudge``: the source
  task's own pick-and-place criteria, plus a miss when the cube reaches the belt
  end and a fall when it lands off the belt.
* **A threaded-friendly renderer**, the same camera factory as
  ``flipup_teleop.make_camera``.

Where the numbers come from
---------------------------
Unlike ``flipup_teleop``, the haptic constants here have **not** been through a
hardware tuning pass. They are carried over from the FlipUp/BallPush pipeline
and rescaled by this task's measured simulated contact forces, and the module
prints the resulting predicted feel and passivity margin so it can be checked
before an operator touches anything. Treat ``--stiffness`` as the knob to tune
first, and see ``README_conveyor.md`` for what was measured in simulation and
what is still an assumption.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

_CONVEYOR_DIR = Path(__file__).resolve().parent.parent / "conveyor_minimal"
if str(_CONVEYOR_DIR) not in sys.path:
    sys.path.insert(0, str(_CONVEYOR_DIR))

from conveyor.environment import (  # noqa: E402
    GRASP_QUAT_WXYZ,
    ConveyorEnv,
)
from conveyor.judge import ConveyorJudge  # noqa: E402
from conveyor.properties import (  # noqa: E402
    DEFAULT_BELT_SPEED_RANGE,
    CubeProperties,
    sample_cube_properties,
)

# Translational stiffness is the value conveyor_minimal ships, which is also
# flipup's. This task does not need it the way the book pivot does -- the
# fin-ray pads grasp a 5 cm cube with a 10 cm opening, not an edge -- so it is
# the first thing to try lowering if the arm feels harsh. Lowering it changes the
# sim's task dynamics, though; to change what the OPERATOR feels, use --stiffness.
DEFAULT_TOOL_KP = 16000.0
DEFAULT_TOOL_ROT_KP = 3000.0
# Critical-ish Cartesian angular damping for the home-pose rotational inertia,
# carried over from flipup_teleop's measurement on the same arm and gripper.
DEFAULT_TOOL_ROT_KD = 90.0
# conveyor_minimal's shipped joint damping.
DEFAULT_JOINT_KD = np.array([64.0, 64.0, 64.0, 16.0, 16.0, 16.0])
# ...raised for teleoperation, as in flipup_teleop: the shipped value is
# marginally underdamped once the actuators saturate, which an operator and
# contact transients both excite.
DEFAULT_ARM_DAMPING = 2.5
# Ceiling on the reflected sim force (N). This task's contact forces are far
# below FlipUp's book-levering forces, so this only clips a hard jam.
DEFAULT_FORCE_CLIP = 100.0
# Cap on the Cartesian force the task-space controller may command (N), as a
# smooth tanh saturation. 0 = off. See flipup_teleop.DEFAULT_TOOL_FORCE_LIMIT
# for why this defaults off: at DEFAULT_ARM_DAMPING the arm needs tens of newtons
# just to drag itself through free space, so a limit low enough to soften contact
# also starves free motion.
DEFAULT_TOOL_FORCE_LIMIT = 0.0


def _wxyz_from_matrix(rotation_matrix):
    q_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return q_xyzw[[3, 0, 1, 2]]


class ConveyorTeleop(ConveyorEnv):
    """Conveyor pick-place with a Cartesian-position interface and force readout."""

    def __init__(
        self,
        seed=0,
        episode_index=0,
        tool_kp=DEFAULT_TOOL_KP,
        tool_rot_kp=DEFAULT_TOOL_ROT_KP,
        tool_rot_kd=DEFAULT_TOOL_ROT_KD,
        joint_kd=None,
        arm_damping=DEFAULT_ARM_DAMPING,
        force_clip=DEFAULT_FORCE_CLIP,
        tool_force_limit=DEFAULT_TOOL_FORCE_LIMIT,
        tool_damping=0.0,
        belt_speed_m_per_s=None,
        belt_speed_range=DEFAULT_BELT_SPEED_RANGE,
        randomize_belt_speed=True,
        randomize_layout=True,
        cube_properties=None,
        randomize_cube=False,
        respawn_object=True,
        time_limit_s=60.0,
        # Longer than conveyor_minimal's own default: at DEFAULT_ARM_DAMPING the
        # arm takes longer to absorb the episode's layout jitter.
        settle_seconds=1.5,
        offscreen=(1024, 768),
    ):
        if cube_properties is not None and randomize_cube:
            raise ValueError(
                "Pass either cube_properties or randomize_cube=True, not both"
            )
        if cube_properties is None and randomize_cube:
            cube_properties = sample_cube_properties(seed)
        if cube_properties is not None and not isinstance(
            cube_properties, CubeProperties
        ):
            raise TypeError("cube_properties must be a CubeProperties")

        self.tool_kp = float(tool_kp)
        self.tool_rot_kp = float(tool_rot_kp)
        self.tool_rot_kd = float(tool_rot_kd)
        self.force_clip = float(force_clip)
        self.tool_force_limit = float(tool_force_limit)
        self.tool_damping = float(tool_damping)
        self._teleop_ready = False

        kwargs = {} if cube_properties is None else {"cube_properties": cube_properties}
        super().__init__(
            seed=seed,
            belt_speed_m_per_s=belt_speed_m_per_s,
            belt_speed_range=belt_speed_range,
            randomize_belt_speed=randomize_belt_speed,
            randomize_layout=randomize_layout,
            # Continuous operation: a cube that runs off the end or is dropped
            # comes back at the belt start so the operator keeps working instead
            # of waiting for a reset.
            respawn_object=respawn_object,
            settle_seconds=float(settle_seconds),
            show_viewer=False,
            **kwargs,
        )

        self.task_space_kp = np.diag(
            [self.tool_kp] * 3 + [self.tool_rot_kp] * 3
        ).astype(float)
        self.task_space_kd = (
            DEFAULT_JOINT_KD * float(arm_damping)
            if joint_kd is None
            else np.broadcast_to(np.asarray(joint_kd, dtype=float), (6,)).copy()
        )
        self.task_space_cartesian_kd = np.array(
            [0.0, 0.0, 0.0] + [self.tool_rot_kd] * 3,
            dtype=float,
        )

        # Bodies belonging to the robot (arm + gripper): every body whose
        # kinematic root is the UR5e's. Contacts with exactly one side in this
        # set are what the operator should feel.
        robot_root = int(self.model.body_rootid[self.model.body("ur5e/base").id])
        self._robot_bodies = frozenset(
            index
            for index in range(self.model.nbody)
            if int(self.model.body_rootid[index]) == robot_root
        )
        self._finger_pad_geoms = frozenset(
            self.model.geom(f"ur5e/wsg50/{side}_finger_pad").id
            for side in ("right", "left")
        )
        self._contact_buf = np.zeros(6, dtype=float)

        self.wrist_force_adr = None
        self.wrist_torque_adr = None
        self.wrist_site_id = None
        try:
            self.wrist_force_adr = int(
                np.asarray(self.model.sensor("ur5e/wsg50/wrist_force_sensor").adr).item()
            )
            self.wrist_torque_adr = int(
                np.asarray(
                    self.model.sensor("ur5e/wsg50/wrist_torque_sensor").adr
                ).item()
            )
            self.wrist_site_id = self.model.site("ur5e/wsg50/ft_sensor_site").id
        except (KeyError, ValueError):
            pass
        self._wrist_tare = np.zeros(6)

        # A bigger offscreen buffer than the scene asset's, so the viewer can
        # render at the size teleop_ball uses. Must be set before the first
        # render: the GL framebuffer is sized from it when the context is made.
        self.model.vis.global_.offwidth = max(
            int(offscreen[0]), int(self.model.vis.global_.offwidth)
        )
        self.model.vis.global_.offheight = max(
            int(offscreen[1]), int(self.model.vis.global_.offheight)
        )

        self.judge = ConveyorJudge(
            time_limit_s=float(time_limit_s),
            terminate_on_miss=False,
            terminate_on_fall=False,
        )
        self._teleop_ready = True
        self.reset(episode_index=episode_index)

    # ------------------------------------------------------------------ state
    @property
    def tool_home(self):
        """Start pose of the tool, and the reference an absolute mapping centres on."""
        return self.home_tool_pose[:3].copy()

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
        result = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model.ptr,
            self.data.ptr,
            mujoco.mjtObj.mjOBJ_SITE,
            self.tool_site_id,
            result,
            0,
        )
        return result[3:].copy()  # mj_objectVelocity gives [angular, linear]

    @property
    def object_pos(self):
        return np.array(self.data.body(self.object_body_id).xpos, dtype=float)

    @property
    def object_quat(self):
        return np.array(self.data.body(self.object_body_id).xquat, dtype=float)

    @property
    def object_twist_world(self):
        """Cube linear then angular velocity, world axes."""
        result = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model.ptr,
            self.data.ptr,
            mujoco.mjtObj.mjOBJ_BODY,
            self.object_body_id,
            result,
            0,
        )
        return np.r_[result[3:], result[:3]]

    def object_distance_to_bin(self):
        """Straight-line distance from the cube to the bin centre (m)."""
        return float(
            np.linalg.norm(self.object_pos - np.array(self.layout.target_bin_center_xyz))
        )

    def success(self):
        """The source task's criterion: cube lifted off the belt, then in the bin."""
        return bool(self.judge.success)

    @property
    def termination_reason(self):
        return self.judge.termination_reason

    def status_line(self):
        """One-line viewer overlay: what the operator needs to see."""
        states = self.judge.states()
        stage = "in bin" if states["object_placed_in_target_bin"] else (
            "held" if states["object_picked_up"] else "on belt"
        )
        return (
            f"belt {self.conveyor_speed_m_per_s * 100:.0f} cm/s | "
            f"cube {stage} | "
            f"{self.object_pos[1] - self.layout.conveyor_end_y:+.2f} m to belt end | "
            f"grip {self.gripper_width * 1000:.0f} mm"
        )

    # ----------------------------------------------------------------- forces
    def contact_force(self):
        """Net contact force the world applies to the robot, world axes (N).

        Zero in free space by construction. ``mj_contactForce`` reports the
        wrench in the contact frame acting on geom2's body, so contacts where the
        robot is geom1 get negated.
        """
        return self.contact_wrench(frame="world")[:3]

    def contact_wrench(self, frame="tool"):
        """Solver-exact contact wrench on the robot at the tool origin.

        Force and moment are accumulated from every robot contact, with moments
        transported from each contact point to the tool site. The order is
        ``[Fx, Fy, Fz, Tx, Ty, Tz]``; ``frame`` may be ``tool`` (Pyrite's
        convention) or ``world``.
        """
        total = np.zeros(6)
        origin = self.tool_pos
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            robot1 = int(self.model.geom_bodyid[contact.geom1]) in self._robot_bodies
            robot2 = int(self.model.geom_bodyid[contact.geom2]) in self._robot_bodies
            if robot1 == robot2:  # neither side, or a robot self-contact
                continue
            mujoco.mj_contactForce(
                self.model.ptr, self.data.ptr, index, self._contact_buf
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
                [world_from_tool.T @ total[:3], world_from_tool.T @ total[3:]]
            )
        raise ValueError(f"unknown wrench frame {frame!r}")

    def grip_force(self):
        """Normal load on the fingertip pads (N), summed over both fingers.

        The grasp channel FlipUp has no use for. A device with a force-reflecting
        grip axis can render this directly, so the operator feels the cube being
        squeezed and, once the belt starts slipping under it, feels the grasp
        take the belt's pull.
        """
        total = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geoms = (int(contact.geom1), int(contact.geom2))
            if not any(geom in self._finger_pad_geoms for geom in geoms):
                continue
            mujoco.mj_contactForce(
                self.model.ptr, self.data.ptr, index, self._contact_buf
            )
            total += abs(float(self._contact_buf[0]))
        return total

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
        report the wrench the world applies to the robot. Taring is done in the
        world frame at reset, where the sensor reads the gripper's own weight.
        """
        if self.wrist_site_id is None:
            return np.zeros(6)
        world_from_sensor = np.asarray(
            self.data.site_xmat[self.wrist_site_id]
        ).reshape(3, 3)
        raw = self.wrist_wrench_raw()
        world = -np.concatenate(
            [world_from_sensor @ raw[:3], world_from_sensor @ raw[3:]]
        )
        world -= self._wrist_tare
        if frame == "world":
            return world
        if frame == "tool":
            world_from_tool = np.asarray(
                self.data.site_xmat[self.tool_site_id]
            ).reshape(3, 3)
            return np.concatenate(
                [world_from_tool.T @ world[:3], world_from_tool.T @ world[3:]]
            )
        raise ValueError(f"unknown wrench frame {frame!r}")

    def wrist_force(self):
        """WSG50 force sensor in world axes, tared at reset (N)."""
        return self.wrist_wrench(frame="world")[:3]

    def estimated_force(self, target_pos):
        """Contact force reconstructed from the actuator side (BallPush's default).

        The minus sign matters: the actuator force points the way the tool is
        being driven, and reflecting it unnegated assists the operator instead of
        resisting -- that bug shipped once in teleop_ball and felt magnetic.
        """
        error = np.asarray(target_pos, dtype=float) - self.tool_pos
        actuator_force = np.clip(
            self.tool_kp * error, -self.force_clip, self.force_clip
        )
        return -actuator_force + self.tool_damping * self.tool_vel

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

    @property
    def home_rotvec(self):
        """Tool orientation at the start pose, as a rotation vector.

        The reference the operator's wrist rotations compose onto when 6-DoF
        control is enabled. This task's grasp orientation is constant, so unlike
        FlipUp's it does not depend on the commanded position at all.
        """
        return Rotation.from_quat(GRASP_QUAT_WXYZ[[1, 2, 3, 0]]).as_rotvec()

    # ------------------------------------------------------------------ drive
    def target_pose7(self, target_pos, target_rotvec=None):
        """xyz + wxyz pose for a commanded tool position.

        Orientation is the task's fixed top-down grasp unless ``target_rotvec``
        supplies one (6-DoF teleoperation).
        """
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        if target_rotvec is None:
            return np.concatenate([target_pos, GRASP_QUAT_WXYZ])
        rotation = Rotation.from_rotvec(
            np.asarray(target_rotvec, dtype=float)
        ).as_matrix()
        return np.concatenate([target_pos, _wxyz_from_matrix(rotation)])

    def limited_target(self, target_pos):
        """Target with the commanded interaction force smoothly saturated.

        The controller applies ``tool_kp * (target - tool)``, so bounding that
        product bounds the force. ``F = limit * tanh(kp*err/limit)`` keeps the
        knee differentiable, since a corner there is its own source of chatter as
        the operator rides the boundary.
        """
        target_pos = np.asarray(target_pos, dtype=float)
        if self.tool_force_limit <= 0.0:
            return target_pos
        error = target_pos - self.tool_pos
        raw = self.tool_kp * np.linalg.norm(error)
        if raw < 1e-9:
            return target_pos
        squashed = self.tool_force_limit * np.tanh(raw / self.tool_force_limit)
        return self.tool_pos + error * (squashed / raw)

    def gripper_width_from_fraction(self, fraction):
        """Map a device grip axis in [0, 1] to a commanded finger width.

        0 is fully closed, 1 fully open, matching the omega.7's grip angle after
        normalization. A device with only a button should latch between the two
        ends instead.
        """
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return self.CLOSE_GRIPPER_WIDTH_M + fraction * (
            self.OPEN_GRIPPER_WIDTH_M - self.CLOSE_GRIPPER_WIDTH_M
        )

    def step(self, target_pos, n_substeps=1, target_rotvec=None, gripper_width=None):
        """Advance ``n_substeps`` timesteps with the tool driven toward the target.

        ``gripper_width`` is held from the previous call when omitted, so a
        control loop that only moves the tool does not have to restate it.
        """
        for _ in range(max(1, int(n_substeps))):
            # limited_target is recomputed per substep: it saturates against the
            # current tool position, so freezing it would misstate the force.
            self.step_task_space(
                self.target_pose7(self.limited_target(target_pos), target_rotvec),
                gripper_width,
            )
            self.judge.update(self)
        return self

    def reset(self, *, episode_index=None):
        """New episode: fresh belt speed, layout and spawn pose, arm back home."""
        super().reset(episode_index=episode_index)
        if not self._teleop_ready:
            return
        self.judge.reset()
        # Tare the wrist sensor at rest: it reads the gripper's own weight there,
        # which is not information about contact.
        if self.wrist_force_adr is not None:
            self._wrist_tare = np.zeros(6)
            self._wrist_tare = self.wrist_wrench(frame="world")
        self.data.time = 0.0

    # ------------------------------------------------------------- recording
    def recorder_task_channels(self):
        """Task-specific channels for ``pyrite_recorder.PyriteEpisodeRecorder``.

        The recorder stores whatever this returns alongside its fixed schema. The
        belt speed is per-episode but is stored per sample so a slice of the
        dataset carries it without having to look up episode attributes.
        """
        return {
            "object_pose": np.r_[self.object_pos, self.object_quat],
            "object_twist_world": self.object_twist_world,
            "conveyor_speed_m_per_s": float(self.conveyor_speed_m_per_s),
            "object_distance_to_bin": self.object_distance_to_bin(),
            "gripper_width": float(self.gripper_width),
            "gripper_width_command": float(self.gripper_width_cmd),
            "grip_force": self.grip_force(),
            "object_picked_up": int(self.judge.object_picked_up),
            "success": int(self.success()),
        }

    def episode_metadata(self):
        """Attributes worth storing on every recorded episode."""
        return {
            "task": "conveyor_pick_place",
            "seed": int(self.seed),
            "episode_index": self.current_episode_index,
            "conveyor_speed_m_per_s": float(self.conveyor_speed_m_per_s),
            "belt_speed_range": [
                float(self.belt_speed_range.minimum),
                float(self.belt_speed_range.maximum),
            ],
            "layout_offset_xy": [float(value) for value in self.layout_offset_xy],
            "conveyor_center_xyz": list(self.layout.conveyor_center_xyz),
            "target_bin_center_xyz": list(self.layout.target_bin_center_xyz),
            "cube_mass_kg": float(self.cube_properties.mass_kg),
            "cube_half_extent_m": float(self.cube_properties.half_extent_m),
            "cube_friction": list(self.cube_properties.friction),
            "belt_drive_friction": float(self.belt_drive_friction),
            "tool_kp": float(self.tool_kp),
            "tool_rot_kp": float(self.tool_rot_kp),
            "tool_rot_kd": float(self.tool_rot_kd),
            "joint_kd": [float(value) for value in self.task_space_kd],
            "grasp_quat_wxyz": [float(value) for value in GRASP_QUAT_WXYZ],
            "timestep_s": float(self.timestep),
        }

    # ---------------------------------------------------------------- render
    # A visualization group MuJoCo's default mjvOption leaves switched off, so
    # moving a geom into it hides it from every render mode used here.
    _HIDDEN_GROUP = 5

    def set_arm_visual(self, mode="full", ghost_alpha=0.22):
        """Show, ghost or hide the UR5e's links, keeping the WSG50 visible.

        Visualization only: ``geom_group`` and ``geom_rgba`` play no part in
        collision detection, which keys off contype/conaffinity, so the physics
        is bit-for-bit unchanged.
        """
        if not hasattr(self, "_arm_visual_saved"):
            self._arm_visual_saved = (
                self.model.geom_group.copy(),
                self.model.geom_rgba.copy(),
            )
        groups, rgba = self._arm_visual_saved
        self.model.geom_group[:] = groups  # start from the compiled state
        self.model.geom_rgba[:] = rgba
        if mode == "full":
            return self

        def body_name(geom):
            return (
                mujoco.mj_id2name(
                    self.model.ptr,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(self.model.geom_bodyid[geom]),
                )
                or ""
            )

        arm = [
            geom
            for geom in range(self.model.ngeom)
            if body_name(geom).startswith("ur5e/") and "wsg50" not in body_name(geom)
        ]
        if mode == "hidden":
            self.model.geom_group[arm] = self._HIDDEN_GROUP
        elif mode == "ghost":
            # Materials win over the compiled default geom rgba, so clear the
            # material link for these geoms; they then render in their rgba
            # colour, which is what carries the alpha.
            if not hasattr(self, "_arm_matid_saved"):
                self._arm_matid_saved = self.model.geom_matid.copy()
            self.model.geom_matid[arm] = -1
            self.model.geom_rgba[arm, 3] = float(ghost_alpha)
        else:
            raise ValueError(f"unknown arm visual mode {mode!r}")
        self.arm_visual = mode
        return self

    def camera_names(self):
        """Fixed cameras compiled into the scene."""
        return [
            mujoco.mj_id2name(self.model.ptr, mujoco.mjtObj.mjOBJ_CAMERA, index)
            for index in range(self.model.ncam)
        ]

    def make_camera(
        self,
        width=640,
        height=480,
        quality="fast",
        azimuth=-125.0,
        elevation=-30.0,
        distance=1.3,
        lookat=None,
        camera=None,
    ):
        """Return ``render() -> HxWx3 RGB`` for a camera on this scene.

        ``camera`` names a fixed camera; ``third_person_camera`` is the one the
        source task's policies were trained from, and ``ur5e/wsg50/wrist_camera``
        is the robot's own viewpoint. When it is None a free camera is placed from
        azimuth/elevation/distance, defaulting to a view that has both the belt's
        working stretch and the bin in frame.

        Safe to call the returned function from a viewer thread while the main
        thread steps: it only reads model/data, and a torn frame is harmless.

        ``quality``: ``full``, ``fast`` (reflections off) or ``collision``
        (collision geoms only, much cheaper -- the cost is the high-resolution
        meshes, not the resolution).
        """
        from dm_control.mujoco import wrapper
        from dm_control.mujoco.engine import Camera, MovableCamera

        if camera is not None:
            view = Camera(
                self.physics, height=int(height), width=int(width), camera_id=camera
            )
        else:
            if lookat is None:
                # Midway between where a grasp happens and where the cube lands.
                lookat = 0.5 * (
                    np.array(
                        [
                            self.layout.conveyor_center_xyz[0],
                            self.layout.conveyor_center_xyz[1],
                            self.layout.conveyor_top_z,
                        ]
                    )
                    + np.array(self.layout.target_bin_center_xyz)
                )
            view = MovableCamera(self.physics, height=int(height), width=int(width))
            view.set_pose(
                np.asarray(lookat, dtype=float),
                float(distance),
                float(azimuth),
                float(elevation),
            )

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
            return view.render(scene_option=options, render_flag_overrides=overrides)

        render.camera = view
        return render


def predicted_feel(tool_kp, scale, force_gain, damping, control_freq, force_tau):
    """Rendered stiffness and the sampled-data passivity limit it must stay under.

    ``k_handle = tool_kp * scale * force_gain`` is what the operator's hand feels,
    and the loop is passive while ``k_handle < 2 * damping / T_effective`` with
    ``T_effective = 1/control_freq + 2 * force_tau``. This is the same criterion
    teleop_ball's anti-bounce tuning settled on, restated here so a conveyor run
    can print its own margin.
    """
    k_handle = float(tool_kp) * float(scale) * float(force_gain)
    effective_period = 1.0 / float(control_freq) + 2.0 * float(force_tau)
    limit = 2.0 * float(damping) / effective_period
    return {
        "k_handle_n_per_m": k_handle,
        "passivity_limit_n_per_m": limit,
        "margin": float("inf") if k_handle <= 0.0 else limit / k_handle,
    }
