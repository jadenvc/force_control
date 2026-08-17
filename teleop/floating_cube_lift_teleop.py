"""Floating WSG50 teleoperation task for rapidly grasping and lifting a cube.

The robot is the same gravity-compensated, inertial floating gripper used by
``FloatingFlipUpTeleop``.  This task adds an actuated analogue gripper, a
rounded-cube contact model, a bounded tabletop workspace, smooth grasp/table
force saturation, and a height-based success condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from dm_control import mjcf
from scipy.spatial.transform import Rotation

from floating_flipup_teleop import FloatingFlipUpTeleop
from flipup.environment import ASSET_DIR, FlipUpEnv


TABLE_TOP_Z = 0.05
CUBE_CENTER_XY = np.array([0.35, 0.0], dtype=float)
CUBE_COLOURS = np.array(
    [
        [0.78, 0.22, 0.16, 1.0],
        [0.16, 0.38, 0.72, 1.0],
        [0.18, 0.58, 0.33, 1.0],
        [0.88, 0.57, 0.12, 1.0],
        [0.48, 0.24, 0.62, 1.0],
        [0.12, 0.62, 0.64, 1.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class CubeProperties:
    mass_kg: float = 0.03125
    size_m: float = 0.0275
    corner_radius_m: float = 0.003
    sliding_friction: float = 0.85
    torsional_friction: float = 0.006
    rolling_friction: float = 0.0002

    def __post_init__(self):
        values = np.array(
            [
                self.mass_kg,
                self.size_m,
                self.corner_radius_m,
                self.sliding_friction,
                self.torsional_friction,
                self.rolling_friction,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("cube properties must be finite")
        if self.mass_kg <= 0.0 or self.size_m <= 0.0:
            raise ValueError("cube mass and size must be positive")
        if not 0.0 < self.corner_radius_m < 0.45 * self.size_m:
            raise ValueError("corner radius must be between 0 and 45% of cube size")
        if np.any(values[3:] < 0.0):
            raise ValueError("cube friction values cannot be negative")

    # Compatibility with the fixed-topology randomization machinery shared by
    # the flip-up collector.  All three outer dimensions are the cube size.
    @property
    def length_m(self):
        return self.size_m

    @property
    def width_m(self):
        return self.size_m

    @property
    def thickness_m(self):
        return self.size_m

    @property
    def friction(self):
        return (
            self.sliding_friction,
            self.torsional_friction,
            self.rolling_friction,
        )

    def summary(self):
        return (
            f"mass={self.mass_kg:.3f} kg, size={100.0 * self.size_m:.2f} cm, "
            f"corner radius={1000.0 * self.corner_radius_m:.1f} mm, "
            f"friction={self.sliding_friction:.3f}/"
            f"{self.torsional_friction:.4f}/{self.rolling_friction:.5f}"
        )


DEFAULT_CUBE_PROPERTIES = CubeProperties()


def sample_cube_properties(base, rng, *, size_jitter=0.10, mass_jitter=0.20):
    size_scale = float(rng.uniform(1.0 - size_jitter, 1.0 + size_jitter))
    mass_scale = float(rng.uniform(1.0 - mass_jitter, 1.0 + mass_jitter))
    return CubeProperties(
        mass_kg=base.mass_kg * mass_scale,
        size_m=base.size_m * size_scale,
        corner_radius_m=base.corner_radius_m * size_scale,
        sliding_friction=base.sliding_friction,
        torsional_friction=base.torsional_friction,
        rolling_friction=base.rolling_friction,
    )


def sample_cube_color(rng):
    return CUBE_COLOURS[int(rng.integers(0, len(CUBE_COLOURS)))].copy()


def cube_lift_scene(seed, properties, *, standoff=0.065, object_xy=None):
    del seed
    object_xy = CUBE_CENTER_XY if object_xy is None else np.asarray(object_xy, float)
    object_center = np.array(
        [object_xy[0], object_xy[1], TABLE_TOP_Z + 0.5 * properties.size_m]
    )
    prepare = object_center + np.array([0.0, 0.0, standoff])
    grasp = object_center.copy()
    lift = grasp + np.array([0.0, 0.0, 0.14])
    transform = np.eye(4)
    transform[:3, 3] = object_center
    return {
        "prepare": prepare,
        "engage": grasp,
        "waypoints": [grasp.copy(), lift],
        "dry_waypoints": [
            {"position": prepare, "gripper": 0.040, "dwell_s": 0.10},
            {"position": grasp, "gripper": 0.040, "dwell_s": 0.10},
            {"position": grasp, "gripper": 0.000, "dwell_s": 0.45},
            {"position": lift, "gripper": 0.000, "dwell_s": 0.10},
        ],
        "object_transform": transform,
        "object_center": object_center,
        "table_top_z": TABLE_TOP_Z,
    }


def sample_cube_start_pose(
    scene,
    rng,
    *,
    prism_size=(0.035, 0.035, 0.025),
    center_probability=0.70,
    force_center=False,
):
    size = np.asarray(prism_size, dtype=float)
    if size.shape != (3,) or np.any(size < 0.0):
        raise ValueError("prism_size must contain three nonnegative values")
    if force_center:
        normalized = np.zeros(3)
        component = "center"
    elif rng.random() < center_probability:
        normalized = np.clip(rng.normal(0.0, 0.20, 3), -0.5, 0.5)
        component = "center_gaussian"
    else:
        normalized = rng.uniform(-0.5, 0.5, 3)
        component = "uniform"
    offset = normalized * size
    position = np.asarray(scene["prepare"], dtype=float) + offset
    return position, {
        "component": component,
        "prism_size_xyz_m": size,
        "normalized_xyz": normalized,
        "offset_xyz_m": offset,
        "position_world_m": position,
    }


def _rounded_cube_mesh(size, radius, resolution=7):
    """Return a visually rounded cube as a triangulated Minkowski surface."""
    half = 0.5 * float(size)
    core = half - float(radius)
    vertices = []
    faces = []
    # (normal axis, sign, u axis, v axis), with du x dv pointing outward.
    configurations = (
        (0, 1.0, 1, 2),
        (0, -1.0, 2, 1),
        (1, 1.0, 2, 0),
        (1, -1.0, 0, 2),
        (2, 1.0, 0, 1),
        (2, -1.0, 1, 0),
    )
    grid = np.linspace(-half, half, int(resolution) + 1)
    for axis, sign, u_axis, v_axis in configurations:
        start = len(vertices)
        for u in grid:
            for v in grid:
                point = np.zeros(3)
                point[axis] = sign * half
                point[u_axis] = u
                point[v_axis] = v
                nearest = np.clip(point, -core, core)
                delta = point - nearest
                point = nearest + radius * delta / max(np.linalg.norm(delta), 1e-12)
                vertices.append(point)
        stride = len(grid)
        for iu in range(stride - 1):
            for iv in range(stride - 1):
                a = start + iu * stride + iv
                b = a + stride
                c = b + 1
                d = a + 1
                faces.extend(((a, b, c), (a, c, d)))
    return np.asarray(vertices, float), np.asarray(faces, np.int32)


class FloatingCubeLiftTeleop(FloatingFlipUpTeleop):
    """Direct-impedance floating WSG50 grasping and lifting a rounded cube."""

    task_kind = "cube_lift"
    requires_gripper_control = True
    default_device_home = np.array([0.02, 0.0, 0.015])
    default_scale = np.array([5.0, 5.0, 5.0])
    default_start_prism = np.array([0.10, 0.10, 0.05])
    default_cube_position_jitter = np.array([0.10, 0.10])
    default_tip_softness = 0.5
    default_force_sensor_cutoff = 30.0
    default_standoff = 0.10
    default_max_speed = 0.80
    default_dry_speed = 0.14
    default_cam_azimuth = -45.0
    default_cam_elevation = -32.0
    default_cam_distance = 0.70
    default_surface_force_limit = 40.0
    default_device_wall_half = np.array([0.045, 0.040, 0.048])
    default_device_wall_stiffness = 800.0

    def __init__(
        self,
        *args,
        grasp_force_limit=25.0,
        gripper_speed=0.12,
        success_height=0.08,
        gripper_open_command=0.040,
        **kwargs,
    ):
        self.grasp_force_limit = float(grasp_force_limit)
        self.gripper_speed = float(gripper_speed)
        self.success_height = float(success_height)
        self.gripper_open_command = float(gripper_open_command)
        if self.grasp_force_limit <= 0.0:
            raise ValueError("grasp_force_limit must be positive")
        if self.gripper_speed <= 0.0:
            raise ValueError("gripper_speed must be positive")
        if self.success_height <= 0.0:
            raise ValueError("success_height must be positive")
        self.gripper_command = self.gripper_open_command
        self._gripper_target = self.gripper_open_command
        physical_properties = kwargs.get("physical_properties")
        if physical_properties is None:
            kwargs["physical_properties"] = DEFAULT_CUBE_PROPERTIES
        super().__init__(*args, **kwargs)
        self.gripper_ctrl_range = np.asarray(
            self.model.actuator_ctrlrange[self.gripper_actuator_id], dtype=float
        ).copy()
        self.gripper_open_command = float(
            np.clip(
                self.gripper_open_command,
                self.gripper_ctrl_range[0],
                self.gripper_ctrl_range[1],
            )
        )
        self.gripper_command = self.gripper_open_command
        self._gripper_target = self.gripper_open_command
        self.gripper_actuator_kp = float(
            self.model.actuator_gainprm[self.gripper_actuator_id, 0]
        )
        self.model.actuator_forcerange[self.gripper_actuator_id] = (
            -self.grasp_force_limit,
            max(self.grasp_force_limit, 80.0),
        )
        self.workspace_low = np.array(
            [CUBE_CENTER_XY[0] - 0.20, -0.20, TABLE_TOP_Z - 0.025]
        )
        self.workspace_high = np.array(
            [CUBE_CENTER_XY[0] + 0.20, 0.20, TABLE_TOP_Z + 0.37]
        )
        self.workspace_limit_active = False
        self.configure_episode(
            self.physical_properties,
            CUBE_COLOURS[1],
            self.tool_home,
            object_xy=self.scene["object_center"][:2],
        )

    @classmethod
    def _valid_physical_properties(cls, physical_properties):
        del cls
        return isinstance(physical_properties, CubeProperties)

    @classmethod
    def _make_scene(cls, seed, physical_properties, *, standoff):
        del cls
        return cube_lift_scene(seed, physical_properties, standoff=standoff)

    @staticmethod
    def _desired_tool_transform(tool_position):
        transform = np.eye(4)
        # Identity at planner_tip_site makes the WSG50 fingers point down and
        # keeps its jaw axis fixed in the horizontal plane.
        transform[:3, 3] = np.asarray(tool_position, dtype=float)
        return transform

    @property
    def home_rotvec(self):
        return np.zeros(3)

    def target_pose7(self, position, rotation_vector=None):
        rotation = (
            Rotation.identity()
            if rotation_vector is None
            else Rotation.from_rotvec(np.asarray(rotation_vector, dtype=float))
        )
        quat_xyzw = rotation.as_quat()
        return np.r_[np.asarray(position, dtype=float), quat_xyzw[3], quat_xyzw[:3]]

    @classmethod
    def _build_floating_physics(
        cls,
        scene,
        physical_properties,
        *,
        collision_envelope_dimensions=None,
    ):
        world_model = mjcf.from_path(str(ASSET_DIR / "ground.xml"))

        gripper_model = mjcf.from_path(str(ASSET_DIR / "wsg50" / "wsg50.xml"))
        gripper_collision_geoms = FlipUpEnv._collision_geoms(gripper_model)
        robot_collision_geoms = gripper_collision_geoms
        camera_mount_site = gripper_model.find("site", "cam_mount")
        camera_model = mjcf.from_path(
            str(
                ASSET_DIR
                / "mujoco_menagerie"
                / "realsense_d435i"
                / "d435i_with_cam.xml"
            )
        )
        robot_collision_geoms += FlipUpEnv._collision_geoms(camera_model)
        camera_mount_site.attach(camera_model)
        gripper_transform = np.eye(4)
        gripper_transform[:3, 3] = scene["prepare"]
        FlipUpEnv._attach_model(
            world_model, gripper_model, gripper_transform, freejoint=True
        )

        table_model = mjcf.from_path(str(ASSET_DIR / "custom" / "table" / "table.xml"))
        table_surface = table_model.find("geom", "table_surface")
        table_site = world_model.worldbody.add(
            "site", name="table_attachment_site", pos=(0.5, 0.0, 0.0)
        )
        table_site.attach(table_model)

        cube_model = mjcf.RootElement(model="book2_blend")
        vertices, faces = _rounded_cube_mesh(
            physical_properties.size_m, physical_properties.corner_radius_m
        )
        mesh = cube_model.asset.add(
            "mesh",
            name="rounded_cube",
            vertex=vertices.ravel(),
            face=faces.ravel(),
        )
        body = cube_model.worldbody.add("body", name="book2_blend")
        body.add(
            "geom",
            name="book_visual",
            type="mesh",
            mesh=mesh,
            rgba=CUBE_COLOURS[1],
            contype=0,
            conaffinity=0,
            group=2,
        )
        if collision_envelope_dimensions is None:
            envelope_size = physical_properties.size_m
        else:
            envelope_size = float(np.max(collision_envelope_dimensions))
        radius_ratio = (
            physical_properties.corner_radius_m / physical_properties.size_m
        )
        envelope_radius = envelope_size * radius_ratio
        collision = body.add(
            "geom",
            name="book_collision",
            type="box",
            size=np.full(3, 0.5 * envelope_size - envelope_radius),
            margin=envelope_radius,
            mass=physical_properties.mass_kg,
            friction=physical_properties.friction,
            condim=4,
            # The half-scale cube needs proportionally smaller compliance than
            # the old 5.5 cm object. This remains critically damped and smooth,
            # but avoids several millimetres of penetration destabilizing the
            # much smaller opposing-finger grasp.
            solref=(0.005, 2.0),
            solimp=(0.90, 0.95, 0.001),
            priority=12,
            group=3,
        )
        object_transform = np.asarray(scene["object_transform"], dtype=float).copy()
        object_transform[2, 3] = TABLE_TOP_Z + 0.5 * envelope_size
        FlipUpEnv._attach_model(
            world_model, cube_model, object_transform, freejoint=True
        )

        for geom in world_model.find_all("geom"):
            geom.contype = 0
            geom.conaffinity = 0
        collision.contype = 1
        collision.conaffinity = 4
        for geom in robot_collision_geoms:
            geom.conaffinity = 5  # cube bit 1 | table bit 4
        table_surface.contype = 4
        table_surface.conaffinity = 1
        return mjcf.Physics.from_mjcf_model(world_model)

    def _init_surface_safety(self):
        self._surface_guard_geom_ids = frozenset(
            (self.model.geom("table/table_surface").id,)
        )
        self._surface_limit_normal = None
        self._surface_limit_boundary = None
        self._surface_contact_misses = 0
        self._surface_contact_grace_steps = max(
            1, int(round(0.020 / float(self.model.opt.timestep)))
        )
        self._requested_target = np.asarray(self.scene["prepare"], float).copy()
        self._drive_target = self._requested_target.copy()

    def surface_safe_target(self, target_pos):
        """Smoothly saturate only the spring component pushing through the table."""
        target = np.asarray(target_pos, dtype=float)
        # Let the inherited state machine acquire/release the active surface;
        # calculate the smooth saturation from the original requested target.
        super().surface_safe_target(target)
        normal = self._surface_limit_normal
        if normal is None or self.surface_force_limit <= 0.0:
            return target.copy()
        normal_error = float(np.dot(target - self.tool_pos, normal))
        if normal_error >= 0.0:
            return target.copy()
        max_deflection = self.surface_force_limit / self.tool_kp
        limited_error = -max_deflection * np.tanh(
            -normal_error / max(max_deflection, 1e-12)
        )
        return target + (limited_error - normal_error) * normal

    def limited_target(self, target_pos):
        requested = np.asarray(target_pos, dtype=float)
        bounded = np.clip(requested, self.workspace_low, self.workspace_high)
        self.workspace_limit_active = bool(
            np.any(np.abs(bounded - requested) > 1e-12)
        )
        return super().limited_target(bounded)

    def set_gripper_command(self, command):
        self.gripper_command = float(
            np.clip(command, self.gripper_ctrl_range[0], self.gripper_ctrl_range[1])
        )

    def reset(self):
        self.gripper_command = self.gripper_open_command
        self._gripper_target = self.gripper_open_command
        super().reset()

    def _reset_gripper_state(self):
        opening = float(self.gripper_command)
        self._gripper_target = opening
        self.data.qpos[self.gripper_qpos_ids] = opening
        self.data.qvel[self.gripper_dof_ids] = 0.0
        self.data.ctrl[self.gripper_actuator_id] = opening
        mujoco.mj_forward(self.model.ptr, self.data.ptr)

    def _apply_gripper_control(self):
        step = self.gripper_speed * float(self.model.opt.timestep)
        delta = np.clip(self.gripper_command - self._gripper_target, -step, step)
        self._gripper_target += float(delta)
        position = float(np.mean(self.data.qpos[self.gripper_qpos_ids]))
        raw_force = self.gripper_actuator_kp * (self._gripper_target - position)
        smooth_force = self.grasp_force_limit * np.tanh(
            raw_force / self.grasp_force_limit
        )
        effective_target = position + smooth_force / self.gripper_actuator_kp
        self.data.ctrl[self.gripper_actuator_id] = np.clip(
            effective_target,
            self.gripper_ctrl_range[0],
            self.gripper_ctrl_range[1],
        )

    def _after_gripper_step(self):
        # Unlike flip-up, the sliders remain dynamic so contacts can stop them.
        return None

    @property
    def gripper_opening(self):
        return float(np.mean(self.data.qpos[self.gripper_qpos_ids]))

    @property
    def gripper_controller_target(self):
        return float(self._gripper_target)

    @property
    def gripper_actuator_force(self):
        return float(self.data.actuator_force[self.gripper_actuator_id])

    def grasp_force(self):
        """Average cube contact load across the two fingertip pads (N)."""
        per_finger = np.zeros((2, 3), dtype=float)
        tip_lookup = {
            int(self.tip_contact_geom_ids[0]): 0,
            int(self.tip_contact_geom_ids[1]): 1,
        }
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in tip_lookup and geom2 == self.book_collision_geom_id:
                tip_geom, sign = geom1, -1.0
            elif geom2 in tip_lookup and geom1 == self.book_collision_geom_id:
                tip_geom, sign = geom2, 1.0
            else:
                continue
            mujoco.mj_contactForce(
                self.model.ptr, self.data.ptr, index, self._contact_buf
            )
            contact_to_world = np.asarray(contact.frame).reshape(3, 3).T
            per_finger[tip_lookup[tip_geom]] += (
                sign * contact_to_world @ self._contact_buf[:3]
            )
        active = np.linalg.norm(per_finger, axis=1)
        return float(np.mean(active)) if np.any(active > 0.0) else 0.0

    def table_contact_force(self):
        """Magnitude of the net solver contact force on the robot from the table."""
        table_geom = int(self.model.geom("table/table_surface").id)
        total = np.zeros(3, dtype=float)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if table_geom not in (geom1, geom2):
                continue
            robot1 = int(self.model.geom_bodyid[geom1]) in self._robot_bodies
            robot2 = int(self.model.geom_bodyid[geom2]) in self._robot_bodies
            if robot1 == robot2:
                continue
            mujoco.mj_contactForce(
                self.model.ptr, self.data.ptr, index, self._contact_buf
            )
            contact_to_world = np.asarray(contact.frame).reshape(3, 3).T
            force = contact_to_world @ self._contact_buf[:3]
            total += force if robot2 else -force
        return float(np.linalg.norm(total))

    @property
    def object_height_m(self):
        return float(self.book_pos[2])

    @property
    def lift_height_m(self):
        return max(
            0.0,
            self.object_height_m - 0.5 * self.physical_properties.size_m - TABLE_TOP_Z,
        )

    def task_metric_value(self):
        return self.lift_height_m

    def success(self, threshold_deg=None):
        del threshold_deg
        return self.lift_height_m >= self.success_height

    def book_angle_deg(self):
        # Compatibility field for older replay/CSV code; the generic task
        # metric and explicit object-height arrays carry the correct semantics.
        return self.lift_height_m

    def configure_episode(
        self,
        physical_properties,
        cube_color,
        tool_home,
        *,
        object_xy=None,
    ):
        if not isinstance(physical_properties, CubeProperties):
            raise TypeError("physical_properties must be CubeProperties")
        color = np.asarray(cube_color, dtype=float)
        tool_home = np.asarray(tool_home, dtype=float)
        if color.shape != (4,) or tool_home.shape != (3,):
            raise ValueError("cube color must be RGBA and tool_home must be xyz")
        object_xy = (
            CUBE_CENTER_XY.copy()
            if object_xy is None
            else np.asarray(object_xy, dtype=float)
        )
        self.physical_properties = physical_properties
        self.book_color = np.clip(color, 0.0, 1.0)
        self.scene = cube_lift_scene(
            self.seed,
            physical_properties,
            standoff=self.standoff,
            object_xy=object_xy,
        )
        self.tool_home = tool_home.copy()

        radius = physical_properties.corner_radius_m
        inner_half = 0.5 * physical_properties.size_m - radius
        self.model.geom_size[self.book_collision_geom_id] = inner_half
        self.model.geom_margin[self.book_collision_geom_id] = radius
        self.model.geom_friction[
            self.book_collision_geom_id
        ] = physical_properties.friction
        self.model.geom_rgba[self.book_visual_geom_id] = self.book_color
        self.model.mesh_vert[self._book_mesh_slice] = (
            self._book_mesh_reference
            * physical_properties.size_m
            / self._book_mesh_reference_dimensions[0]
        )
        self._book_mesh_version += 1

        mass = physical_properties.mass_kg
        self.model.body_mass[self.book_body_id] = mass
        self.model.body_ipos[self.book_body_id] = 0.0
        self.model.body_inertia[self.book_body_id] = (
            mass * physical_properties.size_m**2 / 6.0
        )
        transform = self.scene["object_transform"]
        self._initial_qpos[
            self.book_free_qpos_adr : self.book_free_qpos_adr + 7
        ] = np.r_[transform[:3, 3], 1.0, 0.0, 0.0, 0.0]
        self._configure_tool_home()
        mujoco.mj_setConst(self.model.ptr, self.data.ptr)
        self.model.geom_rbound[
            self.book_collision_geom_id
        ] = self._book_collision_geom_rbound
        self.model.geom_aabb[
            self.book_collision_geom_id
        ] = self._book_collision_geom_aabb
        self.model.bvh_aabb[:] = self._compiled_bvh_aabb
        self.gripper_command = self.gripper_open_command
        self._gripper_target = self.gripper_open_command
        self.reset()
        return self
