from __future__ import annotations

from pathlib import Path
from typing import Final

import mujoco
import numpy as np
import numpy.typing as npt
from dm_control import mjcf
from dm_control.mujoco.engine import Physics
from scipy.spatial.transform import Rotation

from .physical_properties import (
    DEFAULT_PHYSICAL_PROPERTIES,
    PhysicalProperties,
)

FloatArray = npt.NDArray[np.float64]

ASSET_DIR: Final[Path] = Path(__file__).resolve().parent / "assets"


def _wxyz_from_matrix(rotation_matrix: FloatArray) -> FloatArray:
    quaternion_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return quaternion_xyzw[[3, 0, 1, 2]]


def matrix_to_pose7(transform: FloatArray) -> FloatArray:
    """Convert a 4x4 transform to xyz + MuJoCo wxyz quaternion."""
    return np.concatenate([transform[:3, 3], _wxyz_from_matrix(transform[:3, :3])])


def _quat_from_transform(transform: FloatArray) -> FloatArray:
    return _wxyz_from_matrix(transform[:3, :3])


class FlipUpEnv:
    """UR5e + closed WSG50 environment for pivoting a book upright."""

    _HOME_JOINTS: Final[FloatArray] = np.array(
        [-1.53, -2.14, 2.07, -1.7, -1.57, 0.0], dtype=np.float64
    )
    _AXIS_NAMES: Final[tuple[str, ...]] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    )

    def __init__(
        self,
        bookend_transform: FloatArray,
        book_transform: FloatArray,
        *,
        show_viewer: bool = True,
        physical_properties: PhysicalProperties = DEFAULT_PHYSICAL_PROPERTIES,
    ) -> None:
        self._validate_transform(bookend_transform, "bookend_transform")
        self._validate_transform(book_transform, "book_transform")
        self.physical_properties = physical_properties

        self.physics = self._build_physics(
            bookend_transform,
            book_transform,
            physical_properties,
        )
        self.model = self.physics.model
        self.data = self.physics.data

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
        self.tool_site_id = self.model.site("ur5e/wsg50/planner_tip_site").id
        self.book_body_id = self.model.body("book2_blend/book2_blend").id
        self.book_collision_geom_id = self.model.geom("book2_blend/book_collision").id

        self.task_space_kp = 2.0 * np.diag(
            [8000.0, 8000.0, 8000.0, 2000.0, 2000.0, 2000.0]
        )
        self.task_space_kd = 8.0 * np.array(
            [8.0, 8.0, 8.0, 2.0, 2.0, 2.0], dtype=np.float64
        )
        # Optional Cartesian damping applied to the tool twist. The original
        # controller has joint damping only; subclasses can add task-space
        # translational or rotational damping without changing its default.
        self.task_space_cartesian_kd = np.zeros(6, dtype=np.float64)
        self.jacobian = np.zeros((6, self.model.nv), dtype=np.float64)
        self.twist = np.zeros(6, dtype=np.float64)
        self.site_quaternion = np.zeros(4, dtype=np.float64)
        self.site_quaternion_conjugate = np.zeros(4, dtype=np.float64)
        self.error_quaternion = np.zeros(4, dtype=np.float64)

        self._initial_qpos = self.data.qpos.copy()
        self.viewer = None
        self.reset()

        if show_viewer:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(
                model=self.model.ptr,
                data=self.data.ptr,
            )

    @staticmethod
    def _validate_transform(transform: FloatArray, name: str) -> None:
        transform = np.asarray(transform)
        if transform.shape != (4, 4):
            raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")

    @staticmethod
    def _attach_model(
        world_model: mjcf.RootElement,
        child_model: mjcf.RootElement,
        transform: FloatArray,
        *,
        freejoint: bool,
    ) -> None:
        site = world_model.worldbody.add(
            "site",
            name=f"{child_model.model}_attachment_site",
            pos=transform[:3, 3],
            quat=_quat_from_transform(transform),
            group=3,
        )
        attached_body = site.attach(child_model)
        if freejoint:
            attached_body.add("freejoint")

    @staticmethod
    def _configure_book_model(
        book_model: mjcf.RootElement,
        properties: PhysicalProperties,
    ) -> None:
        """Resize the book and set contact/inertial properties before compilation."""
        book_mesh = book_model.find("mesh", "book2_blend")
        collision_geom = book_model.find("geom", "book_collision")
        if book_mesh is None or collision_geom is None:
            raise RuntimeError("The book mesh or collision geometry is missing")

        base = DEFAULT_PHYSICAL_PROPERTIES
        book_mesh.scale = (
            properties.length_m / base.length_m,
            properties.width_m / base.width_m,
            properties.thickness_m / base.thickness_m,
        )
        collision_geom.pos = (
            properties.length_m / 2.0,
            properties.width_m / 2.0,
            properties.thickness_m / 2.0,
        )
        collision_geom.size = (
            properties.length_m / 2.0,
            properties.width_m / 2.0,
            properties.thickness_m / 2.0,
        )
        collision_geom.mass = properties.mass_kg
        collision_geom.friction = properties.friction

    @staticmethod
    def _collision_geoms(model: mjcf.RootElement) -> tuple[mjcf.Element, ...]:
        """Return physical geoms, excluding geoms using the visual default."""
        return tuple(
            geom
            for geom in model.find_all("geom")
            if getattr(geom.dclass, "full_identifier", "") != "visual"
        )

    @staticmethod
    def _configure_contact_allowlist(
        world_model: mjcf.RootElement,
        *,
        robot_geoms: tuple[mjcf.Element, ...],
        object_geom: mjcf.Element,
        support_geoms: tuple[mjcf.Element, ...],
    ) -> None:
        """Allow only robot-object and object-support contact."""
        # Use one directed collision bit: the object emits it, while only the
        # robot and its physical support accept it. This removes robot
        # self-contact, robot-fixture contact, the floor, and the unused large
        # table while retaining MuJoCo's normal geom-derived friction and
        # solver parameters for the two allowed interaction classes.
        for geom in world_model.find_all("geom"):
            geom.contype = 0
            geom.conaffinity = 0

        object_geom.contype = 1
        for geom in (*robot_geoms, *support_geoms):
            geom.conaffinity = 1

    @classmethod
    def _build_physics(
        cls,
        bookend_transform: FloatArray,
        book_transform: FloatArray,
        physical_properties: PhysicalProperties,
    ) -> Physics:
        world_model = mjcf.from_path(str(ASSET_DIR / "ground.xml"))

        robot_model = mjcf.from_path(
            str(ASSET_DIR / "mujoco_menagerie" / "universal_robots_ur5e" / "ur5e.xml")
        )
        del robot_model.keyframe
        robot_model.worldbody.light.clear()
        robot_collision_geoms = cls._collision_geoms(robot_model)

        attachment_site = robot_model.find("site", "attachment_site")
        if attachment_site is None:
            raise RuntimeError("UR5e attachment site is missing")
        gripper_model = mjcf.from_path(str(ASSET_DIR / "wsg50" / "wsg50.xml"))
        robot_collision_geoms += cls._collision_geoms(gripper_model)
        attachment_site.attach(gripper_model)

        camera_mount_site = gripper_model.find("site", "cam_mount")
        if camera_mount_site is None:
            raise RuntimeError("WSG50 camera mount site is missing")
        camera_model = mjcf.from_path(
            str(
                ASSET_DIR
                / "mujoco_menagerie"
                / "realsense_d435i"
                / "d435i_with_cam.xml"
            )
        )
        robot_collision_geoms += cls._collision_geoms(camera_model)
        camera_mount_site.attach(camera_model)

        robot_site = world_model.worldbody.add(
            "site",
            name="robot_attachment_site",
            pos=(-0.3, 0.0, 0.05),
        )
        robot_site.attach(robot_model)

        table_model = mjcf.from_path(str(ASSET_DIR / "custom" / "table" / "table.xml"))
        table_site = world_model.worldbody.add(
            "site",
            name="table_attachment_site",
            pos=(0.5, 0.0, 0.0),
        )
        table_site.attach(table_model)

        bookend_model = mjcf.from_path(
            str(ASSET_DIR / "custom" / "bookend2_blender" / "bookend2_blender.xml")
        )
        support_collision_geoms = cls._collision_geoms(bookend_model)
        cls._attach_model(
            world_model,
            bookend_model,
            bookend_transform,
            freejoint=False,
        )

        book_model = mjcf.from_path(
            str(ASSET_DIR / "custom" / "book2_blend" / "book2_blend.xml")
        )
        cls._configure_book_model(book_model, physical_properties)
        book_collision_geom = book_model.find("geom", "book_collision")
        if book_collision_geom is None:
            raise RuntimeError("Flip object collision geom is missing")
        cls._attach_model(
            world_model,
            book_model,
            book_transform,
            freejoint=True,
        )

        frame_model = mjcf.from_path(str(ASSET_DIR / "custom" / "frame" / "frame.xml"))
        frame_transform = np.eye(4, dtype=np.float64)
        frame_transform[:3, 3] = (0.5, 1.0, 0.5)
        cls._attach_model(
            world_model,
            frame_model,
            frame_transform,
            freejoint=False,
        )

        # The fixture that physically supports the book is the bookend model's
        # floor/wall/corner, not the separate large decorative table asset.
        cls._configure_contact_allowlist(
            world_model,
            robot_geoms=robot_collision_geoms,
            object_geom=book_collision_geom,
            support_geoms=support_collision_geoms,
        )

        return mjcf.Physics.from_mjcf_model(world_model)

    @property
    def current_time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self) -> None:
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        self.data.qpos[self.joint_qpos_ids] = self._HOME_JOINTS
        self.data.qpos[self.gripper_qpos_ids] = 0.0
        self.data.ctrl[self.gripper_actuator_id] = 0.0
        self.physics.forward()
        if self.viewer is not None:
            self.viewer.sync()

    def get_tool_pose(self) -> FloatArray:
        site_data = self.data.site(self.tool_site_id)
        return np.concatenate(
            [
                site_data.xpos.copy(),
                _wxyz_from_matrix(site_data.xmat.copy().reshape(3, 3)),
            ]
        ).astype(np.float32)

    def get_book_pose(self) -> FloatArray:
        body_data = self.data.body(self.book_body_id)
        return np.concatenate([body_data.xpos.copy(), body_data.xquat.copy()]).astype(
            np.float32
        )

    def step_task_space(self, target_pose: FloatArray) -> bool:
        """Advance one 1 ms step. Returns False when the viewer was closed."""
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (7,):
            raise ValueError(
                f"target_pose must have shape (7,), got {target_pose.shape}"
            )

        target_position = target_pose[:3]
        target_quaternion = target_pose[3:]
        site_data = self.data.site(self.tool_site_id)

        self.twist[:3] = target_position - site_data.xpos
        mujoco.mju_mat2Quat(self.site_quaternion, site_data.xmat)
        mujoco.mju_negQuat(
            self.site_quaternion_conjugate,
            self.site_quaternion,
        )
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
        self.data.ctrl[self.gripper_actuator_id] = 0.0

        mujoco.mj_step(self.model.ptr, self.data.ptr)

        if self.viewer is not None:
            if not self.viewer.is_running():
                return False
            self.viewer.sync()
        return True

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __enter__(self) -> "FlipUpEnv":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
