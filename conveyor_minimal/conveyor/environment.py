from __future__ import annotations

from pathlib import Path
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt
from dm_control import mjcf
from dm_control.mujoco.engine import Physics
from scipy.spatial.transform import Rotation

from .properties import (
    DEFAULT_BELT_SPEED_M_PER_S,
    DEFAULT_BELT_SPEED_RANGE,
    DEFAULT_CUBE_PROPERTIES,
    CubeProperties,
    ValueRange,
)
from .scene import DEFAULT_LAYOUT, ConveyorLayout

FloatArray = npt.NDArray[np.float64]

ASSET_DIR: Final[Path] = Path(__file__).resolve().parent / "assets"


def _wxyz_from_matrix(rotation_matrix: FloatArray) -> FloatArray:
    quaternion_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return quaternion_xyzw[[3, 0, 1, 2]]


def matrix_to_pose7(transform: FloatArray) -> FloatArray:
    """Convert a 4x4 transform to xyz + MuJoCo wxyz quaternion."""
    return np.concatenate([transform[:3, 3], _wxyz_from_matrix(transform[:3, :3])])


# Tool orientation used for every top-down grasp on this belt. The WSG50's local
# +z points out of the fingers and its local +x is the closing direction, so a
# 180-degree turn about world x aims the fingers straight down and closes them
# across the belt's width. Closing across x rather than along y matters: the cube
# slides tangentially along the pads while the belt carries it, instead of being
# batted by the leading pad.
GRASP_QUAT_WXYZ: Final[FloatArray] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


def grasp_pose7(position_xyz: FloatArray) -> FloatArray:
    """xyz + wxyz tool pose for a top-down grasp at ``position_xyz``."""
    return np.concatenate(
        [np.asarray(position_xyz, dtype=np.float64).reshape(3), GRASP_QUAT_WXYZ]
    )


class ConveyorEnv:
    """UR5e + WSG50 picking a cube off a moving belt into a target bin.

    The belt drags whatever rests on it through surface friction (see
    :meth:`_drive_conveyor`), so grasping, carrying, placing and dropping are all
    ordinary MuJoCo contact and the operator feels the belt pull against a
    grasp.

    The belt speed is resampled on every :meth:`reset`, so an operator or a
    policy never sees the same speed twice unless one is pinned explicitly.
    """

    # Joint configuration that puts the tool exactly at the nominal
    # :attr:`home_tool_pose` with :data:`GRASP_QUAT_WXYZ`, so a reset does not
    # have to travel. Solved once by damped least squares against this model;
    # the residual pose error is 3e-16 m / 4e-15 rad, and the tool sits 0.61 m
    # from the base, well inside the UR5e's reach.
    _HOME_JOINTS: Final[FloatArray] = np.array(
        [-1.836662, -1.541864, 1.291586, -1.320519, -1.570796, 1.304930],
        dtype=np.float64,
    )
    _AXIS_NAMES: Final[tuple[str, ...]] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    )
    # Source task: config/task/env/table_conveyor_1robot_1object.yaml.
    _ROBOT_BASE_XYZ: Final[tuple[float, float, float]] = (0.82, 0.24, 0.2)
    _ROBOT_BASE_QUAT_WXYZ: Final[tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 1.0)

    OPEN_GRIPPER_WIDTH_M: Final[float] = 0.10
    CLOSE_GRIPPER_WIDTH_M: Final[float] = 0.02

    def __init__(
        self,
        *,
        layout: ConveyorLayout = DEFAULT_LAYOUT,
        cube_properties: CubeProperties = DEFAULT_CUBE_PROPERTIES,
        belt_speed_m_per_s: float | None = None,
        belt_speed_range: ValueRange = DEFAULT_BELT_SPEED_RANGE,
        randomize_belt_speed: bool = True,
        randomize_layout: bool = True,
        respawn_object: bool = False,
        object_picked_height_threshold: float = 0.06,
        object_respawn_y_margin: float = 0.03,
        belt_velocity_time_constant_s: float = 0.01,
        belt_drive_friction: float | None = None,
        settle_seconds: float = 0.5,
        seed: int = 0,
        show_viewer: bool = True,
    ) -> None:
        self.nominal_layout = layout
        self.layout = layout
        self.cube_properties = cube_properties
        self.belt_speed_range = belt_speed_range
        self.randomize_belt_speed = bool(randomize_belt_speed)
        self.randomize_layout = bool(randomize_layout)
        self.respawn_object = bool(respawn_object)
        self.object_picked_height_threshold = float(object_picked_height_threshold)
        self.object_respawn_y_margin = float(object_respawn_y_margin)
        if belt_velocity_time_constant_s <= 0.0:
            raise ValueError("belt_velocity_time_constant_s must be positive")
        self.belt_velocity_time_constant_s = float(belt_velocity_time_constant_s)
        self._belt_drive_friction_override = (
            None if belt_drive_friction is None else float(belt_drive_friction)
        )
        self.settle_seconds = float(settle_seconds)
        self.seed = int(seed)

        self._fixed_belt_speed_m_per_s = (
            None if belt_speed_m_per_s is None else float(belt_speed_m_per_s)
        )
        self.conveyor_speed_m_per_s = (
            DEFAULT_BELT_SPEED_M_PER_S
            if self._fixed_belt_speed_m_per_s is None
            else self._fixed_belt_speed_m_per_s
        )
        self.episode_index = 0
        self.layout_offset_xy = np.zeros(2, dtype=np.float64)
        self.respawn_count = 0

        self.physics = self._build_physics(layout, cube_properties)
        self.model = self.physics.model
        self.data = self.physics.data

        self._bind_indices()

        self.task_space_kp = 2.0 * np.diag(
            [8000.0, 8000.0, 8000.0, 2000.0, 2000.0, 2000.0]
        )
        self.task_space_kd = 8.0 * np.array(
            [8.0, 8.0, 8.0, 2.0, 2.0, 2.0], dtype=np.float64
        )
        # Optional Cartesian damping on the tool twist. Zero here so the shipped
        # controller is joint-damped only; the haptic bridge adds rotational
        # damping without changing this default.
        self.task_space_cartesian_kd = np.zeros(6, dtype=np.float64)
        self.jacobian = np.zeros((6, self.model.nv), dtype=np.float64)
        self.twist = np.zeros(6, dtype=np.float64)
        self.site_quaternion = np.zeros(4, dtype=np.float64)
        self.site_quaternion_conjugate = np.zeros(4, dtype=np.float64)
        self.error_quaternion = np.zeros(4, dtype=np.float64)

        self._initial_qpos = self.data.qpos.copy()
        self._gripper_width_cmd = self.OPEN_GRIPPER_WIDTH_M
        self._belt_drive_enabled = True
        self.viewer = None
        self.reset()

        if show_viewer:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(
                model=self.model.ptr,
                data=self.data.ptr,
            )
            self.viewer.sync()

    # ----------------------------------------------------------- model assembly
    @staticmethod
    def _is_collidable(geom: mjcf.Element) -> bool:
        """True for geoms that take part in contact once compiled.

        Visual-only geoms either switch contact off inline (the scene assets) or
        inherit a ``visual`` default class (the robot assets).
        """
        if geom.contype == 0 and geom.conaffinity == 0:
            return False
        identifier = getattr(geom.dclass, "full_identifier", "") or ""
        return not identifier.rstrip("/").endswith("visual")

    @classmethod
    def _collision_geoms(cls, model: mjcf.RootElement) -> tuple[mjcf.Element, ...]:
        return tuple(geom for geom in model.find_all("geom") if cls._is_collidable(geom))

    @staticmethod
    def _configure_cube_model(
        cube_model: mjcf.RootElement,
        properties: CubeProperties,
    ) -> None:
        """Resize the cube and set its contact/inertial properties before compiling."""
        collision_geom = cube_model.find("geom", "cube_collision")
        marker_geom = cube_model.find("geom", "yaw_marker")
        if collision_geom is None or marker_geom is None:
            raise RuntimeError("The cube collision or yaw-marker geom is missing")

        half_extent = properties.half_extent_m
        scale = half_extent / DEFAULT_CUBE_PROPERTIES.half_extent_m
        collision_geom.size = (half_extent, half_extent, half_extent)
        collision_geom.mass = properties.mass_kg
        collision_geom.friction = properties.friction

        # Keep the yaw marker sitting on the top face at the same relative spot,
        # so a resized cube still shows its orientation in camera views.
        marker_geom.size = tuple(
            float(value) * scale for value in (0.008, 0.004, 0.0012)
        )
        marker_geom.pos = (0.015 * scale, 0.0, half_extent + 0.0012 * scale)

        for site_name, sign in (("bottom_site", -1.0), ("top_site", 1.0)):
            site = cube_model.find("site", site_name)
            if site is not None:
                site.pos = (0.0, 0.0, sign * half_extent)

    @staticmethod
    def _configure_contact_channels(
        *,
        robot_geoms: tuple[mjcf.Element, ...],
        scene_geoms: tuple[mjcf.Element, ...],
        cube_geoms: tuple[mjcf.Element, ...],
    ) -> None:
        """Allow robot-world, robot-cube and cube-world contact, nothing else.

        Two channels are enough. Bit 0 is the "physical world" channel that both
        the fixtures and the cube emit; bit 1 is the robot's own. Because the
        robot only *accepts* bit 0 and only *emits* bit 1, robot self-contact is
        off, and because the fixtures only accept bit 1, static fixture pairs
        (the belt surface against its own frame, the bin walls against the bin
        floor) never generate contacts either.
        """
        for geom in cube_geoms:
            geom.contype = 1
            geom.conaffinity = 3
        for geom in scene_geoms:
            geom.contype = 1
            geom.conaffinity = 2
        for geom in robot_geoms:
            geom.contype = 2
            geom.conaffinity = 1

    @classmethod
    def _build_physics(
        cls,
        layout: ConveyorLayout,
        cube_properties: CubeProperties,
    ) -> Physics:
        world_model = mjcf.from_path(str(ASSET_DIR / "scenes" / "conveyor_pick_place.xml"))
        scene_collision_geoms = cls._collision_geoms(world_model)

        robot_model = mjcf.from_path(
            str(
                ASSET_DIR
                / "robots"
                / "mujoco_menagerie"
                / "universal_robots_ur5e"
                / "ur5e.xml"
            )
        )
        del robot_model.keyframe
        robot_model.worldbody.light.clear()
        robot_collision_geoms = cls._collision_geoms(robot_model)

        attachment_site = robot_model.find("site", "attachment_site")
        if attachment_site is None:
            raise RuntimeError("UR5e attachment site is missing")
        gripper_model = mjcf.from_path(str(ASSET_DIR / "robots" / "wsg50" / "wsg50.xml"))
        robot_collision_geoms += cls._collision_geoms(gripper_model)
        attachment_site.attach(gripper_model)

        robot_site = world_model.worldbody.add(
            "site",
            name="robot_attachment_site",
            pos=cls._ROBOT_BASE_XYZ,
            quat=cls._ROBOT_BASE_QUAT_WXYZ,
            group=3,
        )
        robot_site.attach(robot_model)

        cube_model = mjcf.from_path(
            str(ASSET_DIR / "objects" / "conveyor_cube.xml")
        )
        cls._configure_cube_model(cube_model, cube_properties)
        cube_collision_geoms = cls._collision_geoms(cube_model)
        cube_site = world_model.worldbody.add(
            "site",
            name="cube_attachment_site",
            pos=(
                layout.conveyor_center_xyz[0],
                layout.conveyor_start_y,
                layout.conveyor_top_z + cube_properties.half_extent_m,
            ),
            group=3,
        )
        cube_site.attach(cube_model).add("freejoint")

        cls._configure_contact_channels(
            robot_geoms=robot_collision_geoms,
            scene_geoms=scene_collision_geoms,
            cube_geoms=cube_collision_geoms,
        )

        return mjcf.Physics.from_mjcf_model(world_model)

    def _bind_indices(self) -> None:
        joint_names = tuple(f"ur5e/{name}_joint" for name in self._AXIS_NAMES)
        actuator_names = tuple(f"ur5e/{name}" for name in self._AXIS_NAMES)
        self.joint_qpos_ids = np.array(
            [
                int(np.asarray(self.model.joint(name).qposadr).item())
                for name in joint_names
            ],
            dtype=np.int32,
        )
        self.joint_dof_ids = np.array(
            [
                int(np.asarray(self.model.joint(name).dofadr).item())
                for name in joint_names
            ],
            dtype=np.int32,
        )
        self.actuator_ids = np.array(
            [self.model.actuator(name).id for name in actuator_names],
            dtype=np.int32,
        )
        self.gripper_qpos_ids = np.array(
            [
                int(
                    np.asarray(
                        self.model.joint(f"ur5e/wsg50/{side}_driver_joint").qposadr
                    ).item()
                )
                for side in ("right", "left")
            ],
            dtype=np.int32,
        )
        self.gripper_actuator_id = self.model.actuator("ur5e/wsg50/gripper").id
        self.tool_site_id = self.model.site("ur5e/wsg50/end_effector").id
        self.object_body_id = self.model.body("conveyor_cube/cube").id
        self.belt_surface_geom_id = self.model.geom("conveyor_belt_surface").id
        self.object_collision_geom_id = self.model.geom(
            "conveyor_cube/cube_collision"
        ).id
        self._object_geom_ids = frozenset(
            index
            for index in range(self.model.ngeom)
            if int(self.model.geom_bodyid[index]) == self.object_body_id
        )
        self._contact_buffer = np.zeros(6, dtype=np.float64)

        if self._belt_drive_friction_override is None:
            # The belt geom contributes no tangential friction on purpose (it
            # holds contact priority), so the grip the belt has on the cube is
            # the cube's own sliding friction -- the value MuJoCo would have used
            # for this pair without the priority override.
            self.belt_drive_friction = float(
                self.model.geom_friction[self.object_collision_geom_id][0]
            )
        else:
            self.belt_drive_friction = self._belt_drive_friction_override

        # Compiled positions of the movable fixtures, so per-episode layout
        # jitter is always applied relative to the nominal scene.
        self._default_body_pos = {
            name: np.asarray(self.model.body(name).pos, dtype=np.float64).copy()
            for name in ("conveyor_frame", "target_bin")
        }

        # The cube's freejoint lives on the attachment frame body, not on the
        # cube body itself, so walk up to whichever ancestor owns it rather than
        # relying on a generated name.
        body_id = self.object_body_id
        while self.model.body_jntnum[body_id] == 0:
            parent_id = int(self.model.body_parentid[body_id])
            if parent_id == body_id:
                raise RuntimeError("The cube has no freejoint")
            body_id = parent_id
        joint_id = int(self.model.body_jntadr[body_id])
        if int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError("The cube's root joint is not a freejoint")
        self.object_qpos_adr = int(self.model.jnt_qposadr[joint_id])
        self.object_dof_adr = int(self.model.jnt_dofadr[joint_id])

    # ------------------------------------------------------------------- state
    @property
    def current_time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def tool_pose(self) -> FloatArray:
        """Tool xyz + wxyz in world coordinates."""
        site_data = self.data.site(self.tool_site_id)
        return np.concatenate(
            [
                np.asarray(site_data.xpos, dtype=np.float64).copy(),
                _wxyz_from_matrix(
                    np.asarray(site_data.xmat, dtype=np.float64).reshape(3, 3)
                ),
            ]
        )

    @property
    def object_pose(self) -> FloatArray:
        """Cube xyz + wxyz in world coordinates."""
        body_data = self.data.body(self.object_body_id)
        return np.concatenate(
            [
                np.asarray(body_data.xpos, dtype=np.float64).copy(),
                np.asarray(body_data.xquat, dtype=np.float64).copy(),
            ]
        )

    @property
    def object_velocity(self) -> FloatArray:
        """Cube linear then angular velocity from its freejoint DOFs."""
        return np.asarray(
            self.data.qvel[self.object_dof_adr : self.object_dof_adr + 6],
            dtype=np.float64,
        ).copy()

    @property
    def gripper_width(self) -> float:
        """Measured opening between the fingers."""
        return float(np.sum(self.data.qpos[self.gripper_qpos_ids]))

    @property
    def gripper_width_cmd(self) -> float:
        return float(self._gripper_width_cmd)

    @property
    def object_is_lifted(self) -> bool:
        """True once the cube is clear of the belt by the pick threshold."""
        return bool(
            self.object_pose[2]
            >= self.layout.conveyor_top_z + self.object_picked_height_threshold
        )

    @property
    def object_on_conveyor(self) -> bool:
        return self.layout.is_on_conveyor_xy(self.object_pose[:3])

    @property
    def object_in_target_bin(self) -> bool:
        return self.layout.is_in_target_bin(self.object_pose[:3])

    @property
    def home_tool_pose(self) -> FloatArray:
        """Tool pose the arm settles at after a reset: above the belt, facing down."""
        return grasp_pose7(
            (
                self.layout.conveyor_center_xyz[0] + 0.11,
                self.layout.conveyor_center_xyz[1],
                self.layout.conveyor_top_z + 0.30,
            )
        )

    # ------------------------------------------------------------------- reset
    def reset(self, *, episode_index: int | None = None) -> None:
        """Re-randomize the episode and put the arm back at its home pose.

        Each reset draws a fresh belt speed, layout jitter and cube spawn pose
        from ``(seed, episode_index)``, so a given index is reproducible while
        successive resets differ.
        """
        if episode_index is None:
            episode_index = self.episode_index
        else:
            self.episode_index = int(episode_index)
        rng = np.random.default_rng([self.seed, int(episode_index)])

        if self._fixed_belt_speed_m_per_s is not None:
            self.conveyor_speed_m_per_s = self._fixed_belt_speed_m_per_s
        elif self.randomize_belt_speed:
            self.conveyor_speed_m_per_s = self.belt_speed_range.sample(rng)
        else:
            self.conveyor_speed_m_per_s = DEFAULT_BELT_SPEED_M_PER_S

        if self.randomize_layout:
            self.layout_offset_xy = self.nominal_layout.sample_layout_offset(rng)
        else:
            self.layout_offset_xy = np.zeros(2, dtype=np.float64)
        self.layout = self.nominal_layout.shifted(self.layout_offset_xy)
        self._apply_layout_to_model()

        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        self.data.qpos[self.joint_qpos_ids] = self._HOME_JOINTS
        self._set_gripper_width(self.OPEN_GRIPPER_WIDTH_M, instant=True)
        self._set_object_pose(self._sample_object_pose(rng))
        self.respawn_count = 0
        self.episode_index = int(episode_index) + 1

        self.physics.forward()
        self._settle()
        if self.viewer is not None:
            self.viewer.sync()

    def _settle(self) -> None:
        """Let the cube come to rest on the belt and the arm reach its home pose.

        The belt is held still here, matching the source task's
        ``wait_until_stable`` reset: the episode clock and the cube's travel both
        start from zero afterwards. Because :attr:`_HOME_JOINTS` already solves
        the nominal home pose, the arm only has to absorb the layout jitter.
        """
        if self.settle_seconds <= 0.0:
            return
        home_pose = self.home_tool_pose
        self._belt_drive_enabled = False
        try:
            for _ in range(int(round(self.settle_seconds / self.timestep))):
                self.step_task_space(home_pose, self.OPEN_GRIPPER_WIDTH_M)
        finally:
            self._belt_drive_enabled = True
        self.data.time = 0.0

    def _apply_layout_to_model(self) -> None:
        """Move the belt and bin bodies to match the episode's layout jitter."""
        offset_xyz = np.array(
            [self.layout_offset_xy[0], self.layout_offset_xy[1], 0.0],
            dtype=np.float64,
        )
        for body_name, default_pos in self._default_body_pos.items():
            self.model.body(body_name).pos[:] = default_pos + offset_xyz

    def _set_object_pose(self, pose_xyz_wxyz: FloatArray) -> None:
        pose = np.asarray(pose_xyz_wxyz, dtype=np.float64).reshape(7)
        self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 7] = pose
        self.data.qvel[self.object_dof_adr : self.object_dof_adr + 6] = 0.0

    def _sample_object_pose(self, rng: np.random.Generator) -> FloatArray:
        return self.layout.sample_spawn_pose(
            self.cube_properties.radius_m,
            self.cube_properties.half_extent_m,
            rng,
        )

    def _set_gripper_width(self, width_m: float, *, instant: bool = False) -> None:
        width_m = float(np.clip(width_m, 0.0, 2.0 * 0.055))
        self._gripper_width_cmd = width_m
        self.data.ctrl[self.gripper_actuator_id] = width_m / 2.0
        if instant:
            self.data.qpos[self.gripper_qpos_ids] = width_m / 2.0

    # -------------------------------------------------------------- belt drive
    def _belt_normal_load(self) -> float:
        """Normal force the belt surface currently carries from the cube (N)."""
        total = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geoms = (int(contact.geom1), int(contact.geom2))
            if self.belt_surface_geom_id not in geoms:
                continue
            if not any(geom in self._object_geom_ids for geom in geoms):
                continue
            mujoco.mj_contactForce(
                self.model.ptr, self.data.ptr, index, self._contact_buffer
            )
            total += abs(float(self._contact_buffer[0]))
        return total

    def _drive_conveyor(self) -> None:
        """Apply the Coulomb friction of a belt surface moving at the belt speed.

        The source PyriteEnvSuites task moves the cube kinematically: once per
        20 Hz control step it teleports the freejoint forward and pins its
        velocity. That does not survive a 1 kHz haptic loop -- pinned every
        millisecond the cube could never be lifted at all, and a 20 Hz teleport
        would reach the operator's hand as an impulse train.

        MuJoCo has no belt primitive and its contact solver treats the surface as
        stationary, so the tangential force is applied here instead. The belt
        geom carries contact priority with near-zero friction (see the scene
        asset) and this method supplies the real thing: a force opposing the
        cube's velocity *relative to the belt surface*, saturating at
        ``mu * normal_load``. Below saturation the cube sticks to the belt and
        rides at belt speed; above it, it slips -- which is what happens once the
        gripper holds it, and is the belt drag the operator feels.
        """
        self.data.xfrc_applied[self.object_body_id, :] = 0.0
        if not self._belt_drive_enabled:
            return

        normal_load = self._belt_normal_load()
        if normal_load <= 0.0:
            return

        # Planar velocity of the cube relative to the belt surface.
        relative_velocity = np.array(
            [
                float(self.data.qvel[self.object_dof_adr]),
                float(self.data.qvel[self.object_dof_adr + 1])
                - self.conveyor_speed_m_per_s,
            ]
        )
        mass = float(self.model.body_mass[self.object_body_id])
        force_xy = -mass * relative_velocity / self.belt_velocity_time_constant_s
        limit = self.belt_drive_friction * normal_load
        magnitude = float(np.linalg.norm(force_xy))
        if magnitude > limit:
            force_xy *= limit / magnitude
        self.data.xfrc_applied[self.object_body_id, 0] = force_xy[0]
        self.data.xfrc_applied[self.object_body_id, 1] = force_xy[1]

    def _should_respawn_object(self) -> bool:
        if not self.respawn_object:
            return False
        pose = self.object_pose
        if self.object_is_lifted or self.layout.is_in_target_bin(pose[:3]):
            return False
        if pose[1] > self.layout.conveyor_end_y + self.object_respawn_y_margin:
            return True
        # Recovery: the cube slipped off the side, or a place attempt missed the
        # bin, so put it back at the belt start and let the operator retry.
        return not self.layout.is_on_conveyor_xy(pose[:3])

    def _maybe_respawn_object(self) -> None:
        if not self._should_respawn_object():
            return
        rng = np.random.default_rng(
            [self.seed, int(self.episode_index), 1 + self.respawn_count]
        )
        self._set_object_pose(self._sample_object_pose(rng))
        self.data.qvel[self.object_dof_adr + 1] = self.conveyor_speed_m_per_s
        self.respawn_count += 1
        # Refresh kinematics so callers reading the pose on this same step see
        # the respawned cube rather than the one that just left the belt.
        self.physics.forward()

    # -------------------------------------------------------------------- step
    def step_task_space(
        self,
        target_pose: FloatArray,
        gripper_width_m: float | None = None,
    ) -> bool:
        """Advance one timestep. Returns False once the viewer has been closed."""
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (7,):
            raise ValueError(
                f"target_pose must have shape (7,), got {target_pose.shape}"
            )
        if gripper_width_m is not None:
            self._set_gripper_width(gripper_width_m)

        target_position = target_pose[:3]
        target_quaternion = target_pose[3:]
        site_data = self.data.site(self.tool_site_id)

        self.twist[:3] = target_position - site_data.xpos
        mujoco.mju_mat2Quat(self.site_quaternion, site_data.xmat)
        mujoco.mju_negQuat(self.site_quaternion_conjugate, self.site_quaternion)
        mujoco.mju_mulQuat(
            self.error_quaternion,
            target_quaternion,
            self.site_quaternion_conjugate,
        )
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quaternion, 1.0)

        mujoco.mj_jacSite(
            self.model.ptr,
            self.data.ptr,
            self.jacobian[:3],
            self.jacobian[3:],
            self.tool_site_id,
        )

        tool_velocity = self.jacobian @ self.data.qvel
        task_wrench = (
            self.task_space_kp @ self.twist
            - self.task_space_cartesian_kd * tool_velocity
        )
        generalized_force = self.jacobian.T @ task_wrench
        generalized_force[self.joint_dof_ids] -= (
            self.task_space_kd * self.data.qvel[self.joint_dof_ids]
        )
        generalized_force += self.data.qfrc_bias

        actuator_force = generalized_force[self.joint_dof_ids]
        force_ranges = np.asarray(self.model.actuator_forcerange)[self.actuator_ids]
        actuator_force = np.clip(
            actuator_force,
            force_ranges[:, 0],
            force_ranges[:, 1],
        )
        self.data.ctrl[self.actuator_ids] = actuator_force
        self.data.ctrl[self.gripper_actuator_id] = self._gripper_width_cmd / 2.0

        self._drive_conveyor()
        mujoco.mj_step(self.model.ptr, self.data.ptr)
        self._maybe_respawn_object()

        if self.viewer is not None:
            if not self.viewer.is_running():
                return False
            self.viewer.sync()
        return True

    # ------------------------------------------------------------------ render
    def render(
        self,
        *,
        camera: str = "third_person_camera",
        width: int = 256,
        height: int = 256,
    ) -> npt.NDArray[np.uint8]:
        """Render one RGB frame from a fixed scene camera."""
        return self.physics.render(
            height=int(height), width=int(width), camera_id=camera
        )

    # ------------------------------------------------------------------- close
    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __enter__(self) -> "ConveyorEnv":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
