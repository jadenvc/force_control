"""Floating-gripper dynamics for Force Dimension flip-up teleoperation.

This scene keeps the real WSG50 geometry, mass, inertia, camera, contact model,
book and bookend, but removes the UR5e.  A Cartesian impedance wrench is applied
directly to the gripper rigid body, so there are no arm joints, singularities,
joint torque limits, or manipulator Jacobian in the controller.  Gravity is
cancelled body-by-body while physical inertia and contact dynamics remain.
"""

from __future__ import annotations

import numpy as np
import mujoco
from dm_control import mjcf
from scipy.spatial.transform import Rotation

from flipup_teleop import (
    DEFAULT_FORCE_CLIP,
    DEFAULT_SURFACE_FORCE_LIMIT,
    DEFAULT_TOOL_FORCE_LIMIT,
    FlipUpTeleop,
    _wxyz_from_matrix,
    flipup_scene,
    tool_orientation,
)
from flipup.environment import ASSET_DIR, FlipUpEnv
from flipup.physical_properties import DEFAULT_PHYSICAL_PROPERTIES, PhysicalProperties

DEFAULT_FLOATING_TOOL_ROT_KP = 300.0
DEFAULT_FLOATING_TOOL_KP = 5000.0
DEFAULT_FLOATING_HAPTIC_STIFFNESS = 1800.0
# 0 = off. Mirrors surface_force_limit's convention: the visible table/bookend
# anti-windup deliberately excludes the book (see FLOATING_FLIPUP_COMPLIANCE_
# TELEOP.md section 10.1), so this is a separate opt-in cap on fingertip-book
# normal deflection, off by default until validated against real pad data.
DEFAULT_BOOK_NORMAL_FORCE_LIMIT = 0.0
TIP_CONTACT_GEOM_NAMES = (
    "wsg50/right_tip_pad",
    "wsg50/left_tip_pad",
)
SOFT_TIP_SOLREF = (0.020, 2.0)
SOFT_TIP_SOLIMP_WIDTH = 0.005


def advance_two_pole_filter(stage1, stage2, target, alpha):
    """Advance two cascaded exact-discrete-time first-order low-pass poles."""
    stage1 += alpha * (np.asarray(target, dtype=float) - stage1)
    stage2 += alpha * (stage1 - stage2)
    return stage2


class FloatingFlipUpTeleop(FlipUpTeleop):
    """Dynamical floating WSG50 with direct Cartesian impedance control."""

    controller_kind = "floating_gripper"
    task_kind = "flipup"
    default_tool_kp = DEFAULT_FLOATING_TOOL_KP
    default_haptic_stiffness = DEFAULT_FLOATING_HAPTIC_STIFFNESS

    def __init__(
        self,
        seed=0,
        tool_kp=DEFAULT_FLOATING_TOOL_KP,
        tool_rot_kp=DEFAULT_FLOATING_TOOL_ROT_KP,
        tool_rot_kd=None,
        joint_kd=None,
        force_clip=DEFAULT_FORCE_CLIP,
        tool_force_limit=DEFAULT_TOOL_FORCE_LIMIT,
        surface_force_limit=DEFAULT_SURFACE_FORCE_LIMIT,
        tool_damping=0.0,
        physical_properties=None,
        randomize_physics=False,
        standoff=0.05,
        settle_s=0.0,
        settle_speed=0.25,
        offscreen=(1024, 768),
        damping_ratio=1.05,
        gravity_compensation=True,
        collision_envelope_dimensions=None,
        tip_softness=0.0,
        force_sensor_cutoff_hz=0.0,
        book_normal_force_limit=DEFAULT_BOOK_NORMAL_FORCE_LIMIT,
        table_solref=None,
        table_solimp=None,
        approach_compliance_distance_m=0.0,
        approach_compliance_min_kp_ratio=0.2,
        approach_max_speed_mps=0.0,
    ):
        del settle_s, settle_speed, joint_kd, randomize_physics
        if physical_properties is None:
            physical_properties = DEFAULT_PHYSICAL_PROPERTIES
        if not self._valid_physical_properties(physical_properties):
            raise TypeError(
                f"unsupported physical properties {type(physical_properties).__name__}"
            )
        if damping_ratio <= 0.0:
            raise ValueError("damping_ratio must be positive")
        if not 0.0 <= float(tip_softness) <= 1.0:
            raise ValueError("tip_softness must be in [0, 1]")
        if float(force_sensor_cutoff_hz) < 0.0:
            raise ValueError("force_sensor_cutoff_hz cannot be negative")
        if table_solref is not None and len(table_solref) != 2:
            raise ValueError(
                "table_solref must have exactly 2 values (time_constant, "
                "damping_ratio), matching MuJoCo's positive solref convention"
            )
        if table_solimp is not None and len(table_solimp) != 5:
            raise ValueError(
                "table_solimp must have exactly 5 values (d0, d_width, "
                "width, midpoint, power), matching MuJoCo's solimp convention"
            )
        if float(approach_compliance_distance_m) < 0.0:
            raise ValueError("approach_compliance_distance_m cannot be negative")
        if not 0.0 < float(approach_compliance_min_kp_ratio) <= 1.0:
            raise ValueError(
                "approach_compliance_min_kp_ratio must be in (0, 1]"
            )
        if float(approach_max_speed_mps) < 0.0:
            raise ValueError("approach_max_speed_mps cannot be negative")

        self.seed = int(seed)
        self.physical_properties = physical_properties
        self.standoff = float(standoff)
        self.scene = self._make_scene(
            seed, physical_properties, standoff=standoff
        )
        self.tool_kp = float(tool_kp)
        self.tool_rot_kp = float(tool_rot_kp)
        requested_rot_kd = tool_rot_kd
        self.force_clip = float(force_clip)
        self.tool_force_limit = float(tool_force_limit)
        self.surface_force_limit = float(surface_force_limit)
        if self.surface_force_limit < 0.0:
            raise ValueError("surface_force_limit cannot be negative")
        self.book_force_limit = float(book_normal_force_limit)
        if self.book_force_limit < 0.0:
            raise ValueError("book_normal_force_limit cannot be negative")
        self.tool_damping = float(tool_damping)
        self.damping_ratio = float(damping_ratio)
        self.tip_softness = float(tip_softness)
        self.force_sensor_cutoff_hz = float(force_sensor_cutoff_hz)
        self.table_solref = (
            None if table_solref is None else tuple(float(v) for v in table_solref)
        )
        self.table_solimp = (
            None if table_solimp is None else tuple(float(v) for v in table_solimp)
        )
        self.approach_compliance_distance_m = float(approach_compliance_distance_m)
        self.approach_compliance_min_kp_ratio = float(approach_compliance_min_kp_ratio)
        self._approach_compliance_enabled = self.approach_compliance_distance_m > 0.0
        self.approach_max_speed_mps = float(approach_max_speed_mps)
        # Recomputed every step in step(); the pre-step defaults below only
        # matter for code that reads them before the first step() call.
        self._effective_translation_kp = self.tool_kp
        self._effective_translation_kd = None
        self.gravity_compensation = bool(gravity_compensation)
        self._teleop_ready = False
        self.viewer = None

        if collision_envelope_dimensions is None:
            collision_envelope_dimensions = np.array(
                [
                    1.2 * physical_properties.length_m,
                    1.2 * physical_properties.width_m,
                    physical_properties.thickness_m,
                ],
                dtype=float,
            )

        self.physics = self._build_floating_physics(
            self.scene,
            physical_properties,
            collision_envelope_dimensions=collision_envelope_dimensions,
        )
        self.model = self.physics.model
        self.data = self.physics.data
        sensor_sample_hz = 1.0 / float(self.model.opt.timestep)
        if self.force_sensor_cutoff_hz >= 0.5 * sensor_sample_hz:
            raise ValueError(
                "force_sensor_cutoff_hz must be below the physics Nyquist frequency"
            )
        if self.force_sensor_cutoff_hz > 0.0:
            sensor_tau_s = 1.0 / (2.0 * np.pi * self.force_sensor_cutoff_hz)
            sensor_alpha = 1.0 - np.exp(
                -float(self.model.opt.timestep) / sensor_tau_s
            )
        else:
            sensor_tau_s = 0.0
            sensor_alpha = 1.0
        self._force_sensor_alpha = float(sensor_alpha)
        self._force_sensor_stage1_tool = np.zeros(6, dtype=float)
        self._force_sensor_stage2_tool = np.zeros(6, dtype=float)
        self._raw_contact_wrench_world = np.zeros(6, dtype=float)
        self._raw_contact_wrench_tool = np.zeros(6, dtype=float)
        self._sensor_wrench_world = np.zeros(6, dtype=float)
        self._sensor_wrench_tool = np.zeros(6, dtype=float)
        self.force_sensor_parameters = {
            "enabled": self.force_sensor_cutoff_hz > 0.0,
            "kind": (
                "cascaded_first_order"
                if self.force_sensor_cutoff_hz > 0.0
                else "identity"
            ),
            "frame": "tool",
            "pole_cutoff_hz": self.force_sensor_cutoff_hz,
            "alpha": self._force_sensor_alpha,
            "sample_hz": sensor_sample_hz,
            "step_t50_ms": (
                1000.0 * 1.67834699 * sensor_tau_s
                if self.force_sensor_cutoff_hz > 0.0
                else 0.0
            ),
            "low_frequency_group_delay_ms": (
                2000.0 * sensor_tau_s
                if self.force_sensor_cutoff_hz > 0.0
                else 0.0
            ),
        }

        self.gripper_qpos_ids = np.array(
            [
                int(
                    np.asarray(
                        self.model.joint(f"wsg50/{side}_driver_joint").qposadr
                    ).item()
                )
                for side in ("right", "left")
            ],
            dtype=np.int32,
        )
        self.gripper_dof_ids = np.array(
            [
                int(
                    np.asarray(
                        self.model.joint(f"wsg50/{side}_driver_joint").dofadr
                    ).item()
                )
                for side in ("right", "left")
            ],
            dtype=np.int32,
        )
        self.gripper_actuator_id = self.model.actuator("wsg50/gripper").id
        if not hasattr(self, "gripper_command"):
            self.gripper_command = 0.0
        self.tool_site_id = self.model.site("wsg50/planner_tip_site").id
        self.book_body_id = self.model.body("book2_blend/book2_blend").id
        self.book_collision_geom_id = self.model.geom(
            "book2_blend/book_collision"
        ).id
        self.tip_contact_geom_ids = np.array(
            [self.model.geom(name).id for name in TIP_CONTACT_GEOM_NAMES],
            dtype=np.int32,
        )
        self._configure_tip_contact()
        self._configure_table_contact()

        self.free_joint_id = self.model.joint("wsg50/").id
        self.free_qpos_adr = int(self.model.jnt_qposadr[self.free_joint_id])
        self.free_dof_adr = int(self.model.jnt_dofadr[self.free_joint_id])
        self.free_body_id = int(self.model.jnt_bodyid[self.free_joint_id])
        self.gripper_base_body_id = self.model.body("wsg50/base").id
        robot_root = int(self.model.body_rootid[self.free_body_id])
        self._robot_bodies = frozenset(
            i
            for i in range(self.model.nbody)
            if int(self.model.body_rootid[i]) == robot_root
        )
        self._contact_buf = np.zeros(6, dtype=float)
        self._init_surface_safety()
        self._init_book_safety()

        # Kept for recorder/controller compatibility with the arm environment.
        # For the floating body this is Cartesian damping, not joint damping.
        robot_mass = sum(float(self.model.body_mass[i]) for i in self._robot_bodies)
        self.gripper_mass_kg = robot_mass
        robot_com = sum(
            float(self.model.body_mass[i]) * np.asarray(self.data.xipos[i])
            for i in self._robot_bodies
        ) / max(robot_mass, 1e-9)
        inertia_world = np.zeros((3, 3))
        for body_id in self._robot_bodies:
            mass = float(self.model.body_mass[body_id])
            if mass <= 0.0:
                continue
            rotation = np.asarray(self.data.ximat[body_id]).reshape(3, 3)
            inertia_world += rotation @ np.diag(
                self.model.body_inertia[body_id]
            ) @ rotation.T
            offset = np.asarray(self.data.xipos[body_id]) - robot_com
            inertia_world += mass * (
                np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
            )
        self.gripper_inertia_kg_m2 = np.linalg.eigvalsh(inertia_world)
        critical_rot_kd = 2.0 * self.damping_ratio * np.sqrt(
            self.tool_rot_kp * max(float(np.max(self.gripper_inertia_kg_m2)), 1e-8)
        )
        self.tool_rot_kd = (
            float(critical_rot_kd)
            if requested_rot_kd is None
            else float(requested_rot_kd)
        )
        translation_kd = 2.0 * self.damping_ratio * np.sqrt(
            self.tool_kp * max(robot_mass, 1e-6)
        )
        self.task_space_kp = np.diag(
            [self.tool_kp] * 3 + [self.tool_rot_kp] * 3
        )
        self.task_space_kd = np.array([translation_kd] * 3 + [self.tool_rot_kd] * 3)
        self.task_space_cartesian_kd = self.task_space_kd.copy()
        self._effective_translation_kd = translation_kd

        self.wrist_force_adr = None
        self.wrist_torque_adr = None
        self.wrist_site_id = self.model.site("wsg50/ft_sensor_site").id
        self._wrist_tare = np.zeros(6)

        self.model.vis.global_.offwidth = max(
            int(offscreen[0]), int(self.model.vis.global_.offwidth)
        )
        self.model.vis.global_.offheight = max(
            int(offscreen[1]), int(self.model.vis.global_.offheight)
        )

        # Set up the same runtime book-randomization data as FlipUpTeleop.
        self.book_visual_geom_id = self.model.geom("book2_blend/book_visual").id
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
            ]
        )
        book_attachment_id = int(self.model.body_parentid[self.book_body_id])
        self.book_free_joint_id = int(self.model.body_jntadr[book_attachment_id])
        self.book_free_qpos_adr = int(self.model.jnt_qposadr[self.book_free_joint_id])
        self.book_color = np.array([0.42, 0.25, 0.16, 1.0])
        self._book_mesh_version = 0
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

        self._initial_qpos = self.data.qpos.copy()
        self.physics.forward()
        self._free_to_tool = self._relative_free_to_tool()
        self.tool_home = self.scene["prepare"].copy()
        self._configure_tool_home()
        self.settle_error = 0.0
        self._teleop_ready = True
        self.reset()

    @classmethod
    def _make_scene(cls, seed, physical_properties, *, standoff):
        del cls
        return flipup_scene(seed, physical_properties, standoff=standoff)

    @classmethod
    def _valid_physical_properties(cls, physical_properties):
        del cls
        return isinstance(physical_properties, PhysicalProperties)

    def _configure_tip_contact(self):
        """Interpolate only the two fingertip pads toward a softer contact.

        A zero knob leaves the compiled XML values untouched.  At one, the
        original 10 ms / damping-ratio 1 / 3 mm contact becomes the tested
        20 ms / damping-ratio 2 / 5 mm variant.  Intermediate values are linear,
        so 0.5 resolves to 15 ms / 1.5 / 4 mm.
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
                np.asarray(SOFT_TIP_SOLREF, dtype=float), base_solref.shape
            )
            self.model.geom_solref[self.tip_contact_geom_ids] = (
                (1.0 - softness) * base_solref + softness * target_solref
            )
            self.model.geom_solimp[self.tip_contact_geom_ids, 2] = (
                (1.0 - softness) * base_width
                + softness * SOFT_TIP_SOLIMP_WIDTH
            )

        resolved_solref = np.asarray(
            self.model.geom_solref[self.tip_contact_geom_ids], dtype=float
        )
        resolved_width = np.asarray(
            self.model.geom_solimp[self.tip_contact_geom_ids, 2], dtype=float
        )
        self.tip_contact_parameters = {
            "softness": softness,
            "geom_names": list(TIP_CONTACT_GEOM_NAMES),
            "solref_time_constant_s": float(resolved_solref[0, 0]),
            "solref_damping_ratio": float(resolved_solref[0, 1]),
            "solimp_width_m": float(resolved_width[0]),
        }

    def _configure_table_contact(self):
        """Apply optional direct overrides to the visible table's contact.

        Unlike ``tip_softness``, which interpolates the fingertip pads along a
        single normalized knob, ``table_solref``/``table_solimp`` set the
        table surface's raw MuJoCo contact parameters directly. Either may be
        left as ``None`` to keep the compiled ``table.xml`` values -- a
        positive solref of ``[time_constant, damping_ratio]`` and the default
        ``solimp`` impedance curve. Passing only one of the two leaves the
        other at its compiled default.
        """
        table_geom_id = self.model.geom("table/table_surface").id
        if self.table_solref is not None:
            self.model.geom_solref[table_geom_id] = self.table_solref
        if self.table_solimp is not None:
            self.model.geom_solimp[table_geom_id] = self.table_solimp
        self.table_contact_parameters = {
            "solref": np.asarray(
                self.model.geom_solref[table_geom_id], dtype=float
            ).tolist(),
            "solimp": np.asarray(
                self.model.geom_solimp[table_geom_id], dtype=float
            ).tolist(),
        }

    def _nearest_guarded_surface(self, pos):
        """(distance, outward_normal) to the closest guarded surface.

        Guarded surfaces are the table/bookend box geoms tracked by
        ``_surface_guard_geom_ids`` (see ``_init_surface_safety``). Each is
        assumed to be a box whose local +Z axis is its outward contact
        normal -- true for the table and bookend fixtures modeled here.
        Non-box guarded geoms are skipped. Returns ``(None, None)`` if no
        guarded box geom exists yet (e.g. before ``_init_surface_safety`` has
        run) or the allowlist contains no box geoms. Distance is positive
        above the surface, negative if already penetrating it.
        """
        guard_ids = getattr(self, "_surface_guard_geom_ids", None)
        if not guard_ids:
            return None, None
        pos = np.asarray(pos, dtype=float)
        nearest_distance, nearest_normal = None, None
        for geom_id in guard_ids:
            if int(self.model.geom_type[geom_id]) != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            half_extent_z = float(self.model.geom_size[geom_id][2])
            center = np.asarray(self.data.geom_xpos[geom_id], dtype=float)
            rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=float).reshape(
                3, 3
            )
            normal = rotation[:, 2]
            top = center + half_extent_z * normal
            distance = float(np.dot(pos - top, normal))
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance, nearest_normal = distance, normal
        return nearest_distance, nearest_normal

    def _nearest_guarded_surface_distance(self, pos):
        """Signed distance (m) from ``pos`` to the closest guarded surface.

        See ``_nearest_guarded_surface`` for the sign convention and box
        assumption.
        """
        distance, _ = self._nearest_guarded_surface(pos)
        return distance

    def _approach_speed_limited_target(self, target_pos):
        """Cap how fast the incoming target may move THROUGH a guarded surface.

        This is the mechanism that actually controls impact kinetic energy.
        The kp/kd ramp in ``_effective_translation_gains`` does NOT slow the
        tool down on approach: a critically damped spring tracking a
        constant-velocity target settles into a steady-state lag of
        ``2*damping_ratio*v*sqrt(mass/kp)`` while still MOVING AT THE SAME
        VELOCITY v, so softening kp near the surface only grows the lag --
        the tool still arrives at the surface at full commanded speed, and
        the entire impact then has to be absorbed by the raw contact solver
        instead of the controller. Confirmed against a real training run:
        tool velocity was unchanged (~0.17 m/s, essentially --max-speed) in
        every step immediately before a 67 N impact spike, despite the ramp
        being active for the full 3+ cm approach.

        Off unless ``approach_max_speed_mps > 0``. When enabled, only the
        component of the requested step (relative to the previous drive
        target) pointing INTO the nearest guarded surface is clamped, and
        only once within ``approach_compliance_distance_m`` of it -- outward
        and tangential motion, and everything outside the band, are
        unaffected.
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

    def _effective_translation_gains(self, pos):
        """Cartesian translational (kp, kd), scheduled by approach compliance.

        Off by default (``approach_compliance_distance_m == 0``), in which
        case this always returns ``(tool_kp, task_space_kd[0])`` -- current
        behavior is unchanged. When enabled, ``kp`` is ramped linearly from
        ``tool_kp`` at ``approach_compliance_distance_m`` (or farther) above
        the nearest guarded surface down to
        ``approach_compliance_min_kp_ratio * tool_kp`` at, or below (already
        penetrating), the surface. ``kd`` is recomputed at the same
        ``damping_ratio`` so the controller stays near-critically damped as
        it softens, instead of becoming underdamped and bouncy.
        """
        if not self._approach_compliance_enabled:
            return self.tool_kp, self.task_space_kd[0]
        distance = self._nearest_guarded_surface_distance(pos)
        if distance is None:
            ratio = 1.0
        else:
            clipped = float(np.clip(distance, 0.0, self.approach_compliance_distance_m))
            min_ratio = self.approach_compliance_min_kp_ratio
            ratio = min_ratio + (1.0 - min_ratio) * (
                clipped / self.approach_compliance_distance_m
            )
        kp = ratio * self.tool_kp
        kd = 2.0 * self.damping_ratio * np.sqrt(kp * max(self.gripper_mass_kg, 1e-6))
        return float(kp), float(kd)

    @staticmethod
    def _desired_tool_transform(tool_position):
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(tool_orientation(tool_position), dtype=float)
        transform[:3, 3] = np.asarray(tool_position, dtype=float)
        return transform

    def _relative_free_to_tool(self):
        world_from_free = np.eye(4)
        world_from_free[:3, :3] = np.asarray(
            self.data.xmat[self.free_body_id]
        ).reshape(3, 3)
        world_from_free[:3, 3] = self.data.xpos[self.free_body_id]
        world_from_tool = np.eye(4)
        world_from_tool[:3, :3] = np.asarray(
            self.data.site_xmat[self.tool_site_id]
        ).reshape(3, 3)
        world_from_tool[:3, 3] = self.data.site_xpos[self.tool_site_id]
        return np.linalg.inv(world_from_free) @ world_from_tool

    def _configure_tool_home(self):
        world_from_tool = self._desired_tool_transform(self.tool_home)
        world_from_free = world_from_tool @ np.linalg.inv(self._free_to_tool)
        self._initial_qpos[self.free_qpos_adr : self.free_qpos_adr + 7] = np.r_[
            world_from_free[:3, 3],
            _wxyz_from_matrix(world_from_free[:3, :3]),
        ]

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

        # The exact free-joint pose is overwritten from the desired tool pose
        # after compilation.  This finite initial transform merely keeps the
        # compiler's inferred inertias and bounds well behaved.
        gripper_transform = np.eye(4)
        gripper_transform[:3, 3] = scene["prepare"]
        FlipUpEnv._attach_model(
            world_model, gripper_model, gripper_transform, freejoint=True
        )

        table_model = mjcf.from_path(str(ASSET_DIR / "custom" / "table" / "table.xml"))
        table_surface_geom = table_model.find("geom", "table_surface")
        if table_surface_geom is None:
            raise RuntimeError("Robot-safe table surface geom is missing")
        table_site = world_model.worldbody.add(
            "site", name="table_attachment_site", pos=(0.5, 0.0, 0.0)
        )
        table_site.attach(table_model)

        bookend_model = mjcf.from_path(
            str(ASSET_DIR / "custom" / "bookend2_blender" / "bookend2_blender.xml")
        )
        robot_surface_names = {
            "robot_wall_surface",
            "robot_pivot_surface",
            "robot_floor_surface",
        }
        all_support_geoms = FlipUpEnv._collision_geoms(bookend_model)
        robot_surface_geoms = tuple(
            geom for geom in all_support_geoms if geom.name in robot_surface_names
        )
        support_collision_geoms = tuple(
            geom for geom in all_support_geoms if geom.name not in robot_surface_names
        )
        if len(robot_surface_geoms) != len(robot_surface_names):
            raise RuntimeError("Visible-aligned robot support surfaces are missing")
        FlipUpEnv._attach_model(
            world_model,
            bookend_model,
            np.asarray(scene["bookend_transform"]),
            freejoint=False,
        )

        book_model = mjcf.from_path(
            str(ASSET_DIR / "custom" / "book2_blend" / "book2_blend.xml")
        )
        FlipUpEnv._configure_book_model(
            book_model,
            physical_properties,
            collision_envelope_dimensions=collision_envelope_dimensions,
        )
        book_collision_geom = book_model.find("geom", "book_collision")
        FlipUpEnv._attach_model(
            world_model,
            book_model,
            np.asarray(scene["book_transform"]),
            freejoint=True,
        )

        frame_model = mjcf.from_path(str(ASSET_DIR / "custom" / "frame" / "frame.xml"))
        frame_transform = np.eye(4)
        frame_transform[:3, 3] = (0.5, 1.0, 0.5)
        FlipUpEnv._attach_model(
            world_model, frame_model, frame_transform, freejoint=False
        )

        FlipUpEnv._configure_contact_allowlist(
            world_model,
            robot_geoms=robot_collision_geoms,
            surface_robot_geoms=gripper_collision_geoms,
            object_geom=book_collision_geom,
            support_geoms=support_collision_geoms,
            robot_surface_geoms=robot_surface_geoms,
            robot_table_geom=table_surface_geom,
        )
        return mjcf.Physics.from_mjcf_model(world_model)

    # ------------------------------------------------------ book-normal safety
    def _init_book_safety(self):
        """State for the opt-in fingertip-book normal anti-windup.

        Separate from ``_init_surface_safety`` because the visible-surface
        guard (table/bookend) deliberately excludes the book -- see
        FLOATING_FLIPUP_COMPLIANCE_TELEOP.md section 10.1. Reuses
        ``_surface_contact_grace_steps`` for the release debounce so both
        latches release on the same ~20 ms miss window.
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
        # flipup_teleop.py's surface_safe_target.
        max_deflection = self.book_force_limit / self.tool_kp
        if normal_error >= -max_deflection:
            return target.copy()
        return target + (-max_deflection - normal_error) * normal

    @property
    def book_limit_active(self):
        return self._book_limit_normal is not None

    def reset(self):
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_command
        self.physics.forward()
        self._reset_gripper_state()
        self._refresh_force_sensor(reset=True)
        if hasattr(self, "_surface_limit_normal"):
            self._surface_limit_normal = None
            self._surface_limit_boundary = None
            self._surface_contact_misses = 0
            self._requested_target = self.tool_home.copy()
            self._drive_target = self.tool_home.copy()
        if hasattr(self, "_book_limit_normal"):
            self._book_limit_normal = None
            self._book_limit_boundary = None
            self._book_contact_misses = 0
        self.settle_error = float(np.linalg.norm(self.tool_pos - self.tool_home))
        self.data.time = 0.0

    def _lock_closed_gripper(self):
        """Project both WSG50 sliders to the permanently closed task state."""
        self.data.qpos[self.gripper_qpos_ids] = 0.0
        self.data.qvel[self.gripper_dof_ids] = 0.0
        self.data.ctrl[self.gripper_actuator_id] = 0.0
        # Recompute geom poses and contact forces at the projected state.  This
        # retains the original WSG model topology for old datasets/replay while
        # making the fingers a rigid part of the floating tool.
        mujoco.mj_forward(self.model.ptr, self.data.ptr)

    def _reset_gripper_state(self):
        self._lock_closed_gripper()

    def _apply_gripper_control(self):
        self.data.ctrl[self.gripper_actuator_id] = 0.0

    def _after_gripper_step(self):
        self._lock_closed_gripper()

    def _apply_gravity_compensation(self):
        if not self.gravity_compensation:
            return
        gravity = np.asarray(self.model.opt.gravity, dtype=float)
        for body_id in self._robot_bodies:
            mass = float(self.model.body_mass[body_id])
            if mass > 0.0:
                # xfrc_applied is a world-frame wrench applied at each body's
                # centre of mass.  Cancelling each link separately also cancels
                # gravity moments without an arm dynamics model.
                self.data.xfrc_applied[body_id, :3] -= mass * gravity

    def step(self, target_pos, n_substeps=1, target_rotvec=None):
        target_pos = np.asarray(target_pos, dtype=float)
        for _ in range(max(1, int(n_substeps))):
            self._requested_target = target_pos.copy()
            # Schedule translational (kp, kd) off the tool's CURRENT position;
            # this only shapes tracking stiffness/damping, not approach
            # speed (see _approach_speed_limited_target's docstring for why
            # that distinction matters).
            self._effective_translation_kp, self._effective_translation_kd = (
                self._effective_translation_gains(self.tool_pos)
            )
            speed_limited_target = self._approach_speed_limited_target(
                self._requested_target
            )
            self._drive_target = self.limited_target(
                self.book_normal_safe_target(
                    self.surface_safe_target(speed_limited_target)
                )
            )
            target_pose = self.target_pose7(self._drive_target, target_rotvec)
            target_rotation = Rotation.from_quat(target_pose[[4, 5, 6, 3]])
            tool_rotation = Rotation.from_quat(self.tool_quat[[1, 2, 3, 0]])
            rotation_error = (target_rotation * tool_rotation.inv()).as_rotvec()

            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model.ptr,
                self.data.ptr,
                mujoco.mjtObj.mjOBJ_SITE,
                self.tool_site_id,
                velocity,
                0,
            )
            omega_world = velocity[:3]
            linear_world = velocity[3:]
            force = (
                self._effective_translation_kp * (target_pose[:3] - self.tool_pos)
                - self._effective_translation_kd * linear_world
            )
            torque_at_tool = (
                self.tool_rot_kp * rotation_error
                - self.tool_rot_kd * omega_world
            )

            self.data.xfrc_applied[:] = 0.0
            self._apply_gravity_compensation()
            self.data.xfrc_applied[self.gripper_base_body_id, :3] += force
            # The controller directly commands the floating gripper's force and
            # moment at its COM.  Contact forces still act at their real geom
            # locations; omitting an artificial r x F controller moment avoids
            # turning a translational tip command into a very stiff rotational
            # mode on this short, low-inertia body.
            self.data.xfrc_applied[
                self.gripper_base_body_id, 3:
            ] += torque_at_tool
            self._apply_gripper_control()
            mujoco.mj_step(self.model.ptr, self.data.ptr)
            self._after_gripper_step()
            self._refresh_force_sensor()
        return self

    def _refresh_force_sensor(self, *, reset=False):
        """Cache exact contact truth and one causal 1 kHz sensor observation."""
        raw_world = super().contact_wrench(frame="world")
        world_from_tool = np.asarray(
            self.data.site_xmat[self.tool_site_id]
        ).reshape(3, 3)
        raw_tool = np.concatenate(
            [
                world_from_tool.T @ raw_world[:3],
                world_from_tool.T @ raw_world[3:],
            ]
        )
        self._raw_contact_wrench_world[:] = raw_world
        self._raw_contact_wrench_tool[:] = raw_tool
        if reset or self.force_sensor_cutoff_hz <= 0.0:
            self._force_sensor_stage1_tool[:] = raw_tool
            self._force_sensor_stage2_tool[:] = raw_tool
        else:
            advance_two_pole_filter(
                self._force_sensor_stage1_tool,
                self._force_sensor_stage2_tool,
                raw_tool,
                self._force_sensor_alpha,
            )
        self._sensor_wrench_tool[:] = self._force_sensor_stage2_tool
        self._sensor_wrench_world[:3] = (
            world_from_tool @ self._sensor_wrench_tool[:3]
        )
        self._sensor_wrench_world[3:] = (
            world_from_tool @ self._sensor_wrench_tool[3:]
        )

    def contact_wrench(self, frame="tool"):
        """Cached solver-exact contact truth, updated every physics tick."""
        if frame == "world":
            return self._raw_contact_wrench_world.copy()
        if frame == "tool":
            return self._raw_contact_wrench_tool.copy()
        raise ValueError(f"unknown wrench frame {frame!r}")

    def sensor_wrench(self, frame="tool"):
        """Modeled F/T sensor output; identity when the cutoff knob is zero."""
        if frame == "world":
            return self._sensor_wrench_world.copy()
        if frame == "tool":
            return self._sensor_wrench_tool.copy()
        raise ValueError(f"unknown wrench frame {frame!r}")

    def wrist_wrench_raw(self):
        # A floating base has no physical wrist cut.  Preserve the recorder API
        # with the exact interaction wrench instead of manufacturing a sensor
        # signal that would include the controller wrench.
        return self.contact_wrench(frame="tool")

    def wrist_wrench(self, frame="tool"):
        return self.sensor_wrench(frame=frame)

    def wrist_force(self):
        return self.sensor_wrench(frame="world")[:3]
