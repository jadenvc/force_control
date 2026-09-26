"""2D Push-Circle task: a floating disc pusher shoves a circular puck to a
target circle -- push-T's simpler sibling (no orientation DOF at all, so no
rotation technique/aiming question exists here the way it does for the T),
built to stress continuous re-tracking instead of careful final placement.

Coverage is the exact closed-form circle-circle overlap area (a standard
"circular segment" formula), not a rasterized grid like push-T's T-shape
needs -- exact, no resolution parameter, and cheap enough to call every tick.

The goal doesn't just relocate at unpredictable times (see push_t_teleop.py's
--goal-move-*, which this reuses almost verbatim): --goal-move-opposite-bias
usually places the new goal BEHIND the puck's current direction of travel,
anchored to the puck's current position rather than the origin, so finishing
a push often means the target relocates to somewhere that specifically
requires reversing course -- not just "somewhere else in the workspace".
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from dm_control import mjcf


TABLE_TOP_Z = 0.0
DEFAULT_GOAL_XY = (0.0, 0.0)

# Same convention/rationale as push_t_teleop.py's HARD/SOFT_PUSHER_* --
# pusher_softness=0 reproduces the compiled (stiff) defaults, 1 widens/slows
# the contact solver's correction to reduce chatter/spiking at a high
# --pusher-kp.
HARD_PUSHER_SOLREF = (0.006, 1.0)
SOFT_PUSHER_SOLREF = (0.020, 2.0)
HARD_PUSHER_SOLIMP_WIDTH = 0.0005
SOFT_PUSHER_SOLIMP_WIDTH = 0.005
HARD_PUSHER_SOLIMP_D0 = 0.9
SOFT_PUSHER_SOLIMP_D0 = 0.7


@dataclass(frozen=True)
class PushCircleProperties:
    """Physical properties of the puck and pusher (all independently tunable)."""

    object_mass_kg: float = 0.20
    object_radius_m: float = 0.05
    object_thickness_m: float = 0.02
    # Smaller than object_radius_m on purpose: coverage_fraction is
    # goal-overlap-area / goal-area, which caps at (object_radius/goal_radius)^2
    # < 1.0 if the object is EVER smaller than the goal -- making the default
    # --success-threshold (0.85) geometrically unreachable. With goal <
    # object, perfect centering gives full containment (ratio 1.0).
    goal_radius_m: float = 0.04
    # Object <-> table sliding friction. Low values let the puck coast
    # several centimetres after a push; high values stop it almost as soon
    # as the pusher lets go. Real Coulomb coefficient, not special-cased.
    table_friction: float = 0.4
    table_torsional_friction: float = 0.006
    table_rolling_friction: float = 0.0002
    # Pusher <-> object sliding friction on side contact.
    pusher_friction: float = 0.6
    pusher_radius_m: float = 0.018
    pusher_mass_kg: float = 0.05

    def __post_init__(self):
        positive = (
            self.object_mass_kg,
            self.object_radius_m,
            self.object_thickness_m,
            self.goal_radius_m,
            self.pusher_radius_m,
            self.pusher_mass_kg,
        )
        if any(v <= 0.0 for v in positive):
            raise ValueError("PushCircleProperties dimensions/masses must be positive")
        frictions = (
            self.table_friction,
            self.table_torsional_friction,
            self.table_rolling_friction,
            self.pusher_friction,
        )
        if any(v < 0.0 for v in frictions):
            raise ValueError("friction coefficients cannot be negative")


DEFAULT_PUSH_CIRCLE_PROPERTIES = PushCircleProperties()


def circle_overlap_area(center_distance, r1, r2):
    """Exact area of overlap between two circles of radius r1, r2 whose
    centers are ``center_distance`` apart (the "circular segment" formula).
    Vectorized: any argument may be an array."""
    d = np.asarray(center_distance, dtype=float)
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    out = np.zeros(np.broadcast_shapes(d.shape, r1.shape, r2.shape))
    no_overlap = d >= (r1 + r2)
    fully_contained = d <= np.abs(r1 - r2)
    partial = ~no_overlap & ~fully_contained
    out = np.where(fully_contained, np.pi * np.minimum(r1, r2) ** 2, out)
    if np.any(partial):
        dp = np.where(partial, d, 1.0)  # avoid divide-by-zero on masked-out entries
        r1p = np.where(partial, r1, 1.0)
        r2p = np.where(partial, r2, 1.0)
        alpha = np.arccos(np.clip((dp**2 + r1p**2 - r2p**2) / (2 * dp * r1p), -1.0, 1.0))
        beta = np.arccos(np.clip((dp**2 + r2p**2 - r1p**2) / (2 * dp * r2p), -1.0, 1.0))
        lens = (
            r1p**2 * (alpha - np.sin(2 * alpha) / 2.0)
            + r2p**2 * (beta - np.sin(2 * beta) / 2.0)
        )
        out = np.where(partial, lens, out)
    return out


class PushCircleTeleop:
    """Direct-impedance 2D pusher shoving a circular puck to a target circle."""

    task_kind = "push_circle"
    default_tool_kp = 400.0
    default_max_speed = 0.5

    def __init__(
        self,
        seed=0,
        properties=None,
        goal_xy=DEFAULT_GOAL_XY,
        pusher_kp=400.0,
        damping_ratio=1.2,
        workspace_half_m=0.30,
        success_threshold=0.85,
        settle_s=0.0,
        pusher_softness=0.0,
        force_sensor_cutoff_hz=0.0,
        pusher_joint_damping=1.0,
        noslip_iterations=2,
        disturbance_force_n=0.0,
        disturbance_tau_s=1.0,
        disturbance_seed=None,
        goal_move_min_interval_s=0.0,
        goal_move_max_interval_s=0.0,
        goal_move_xy_half_m=0.15,
        goal_move_skip_prob=0.0,
        goal_move_opposite_bias=0.0,
        goal_move_opposite_spread_deg=50.0,
        goal_move_distance_m=0.15,
        goal_move_velocity_threshold_mps=0.03,
        goal_move_seed=None,
    ):
        del settle_s
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.properties = properties or DEFAULT_PUSH_CIRCLE_PROPERTIES
        if not isinstance(self.properties, PushCircleProperties):
            raise TypeError("properties must be a PushCircleProperties instance")
        self.goal_xy = np.asarray(goal_xy, dtype=float).copy()
        if self.goal_xy.shape != (2,):
            raise ValueError("goal_xy must be (x, y)")
        self.pusher_kp = float(pusher_kp)
        if self.pusher_kp <= 0.0:
            raise ValueError("pusher_kp must be positive")
        self.damping_ratio = float(damping_ratio)
        if self.damping_ratio <= 0.0:
            raise ValueError("damping_ratio must be positive")
        if not 0.0 <= float(pusher_softness) <= 1.0:
            raise ValueError("pusher_softness must be in [0, 1]")
        self.pusher_softness = float(pusher_softness)
        if float(force_sensor_cutoff_hz) < 0.0:
            raise ValueError("force_sensor_cutoff_hz cannot be negative")
        self.force_sensor_cutoff_hz = float(force_sensor_cutoff_hz)
        if float(pusher_joint_damping) < 0.0:
            raise ValueError("pusher_joint_damping cannot be negative")
        self.pusher_joint_damping = float(pusher_joint_damping)
        if int(noslip_iterations) < 0:
            raise ValueError("noslip_iterations cannot be negative")
        self.noslip_iterations = int(noslip_iterations)

        # --- disturbance: force-only (the puck has no rotation DOF) --------
        if float(disturbance_force_n) < 0.0:
            raise ValueError("disturbance_force_n cannot be negative")
        if float(disturbance_tau_s) <= 0.0:
            raise ValueError("disturbance_tau_s must be positive")
        self.disturbance_force_n = float(disturbance_force_n)
        self.disturbance_tau_s = float(disturbance_tau_s)
        disturbance_seed_ = (
            self.seed + 100_003 if disturbance_seed is None else int(disturbance_seed)
        )
        self._disturbance_rng = np.random.default_rng(disturbance_seed_)
        self._disturbance_force = np.zeros(2, dtype=float)

        # --- goal relocation -------------------------------------------------
        if float(goal_move_min_interval_s) < 0.0 or float(goal_move_max_interval_s) < 0.0:
            raise ValueError("goal_move_min/max_interval_s cannot be negative")
        if goal_move_max_interval_s > 0.0 and goal_move_min_interval_s > goal_move_max_interval_s:
            raise ValueError("goal_move_min_interval_s must be <= goal_move_max_interval_s")
        if float(goal_move_xy_half_m) <= 0.0:
            raise ValueError("goal_move_xy_half_m must be positive")
        if not 0.0 <= float(goal_move_skip_prob) < 1.0:
            raise ValueError("goal_move_skip_prob must be in [0, 1)")
        if not 0.0 <= float(goal_move_opposite_bias) <= 1.0:
            raise ValueError("goal_move_opposite_bias must be in [0, 1]")
        if float(goal_move_distance_m) <= 0.0:
            raise ValueError("goal_move_distance_m must be positive")
        if float(goal_move_velocity_threshold_mps) < 0.0:
            raise ValueError("goal_move_velocity_threshold_mps cannot be negative")
        self.goal_move_min_interval_s = float(goal_move_min_interval_s)
        self.goal_move_max_interval_s = float(goal_move_max_interval_s)
        self.goal_move_xy_half_m = float(goal_move_xy_half_m)
        self.goal_move_skip_prob = float(goal_move_skip_prob)
        self.goal_move_opposite_bias = float(goal_move_opposite_bias)
        self.goal_move_opposite_spread_rad = np.radians(goal_move_opposite_spread_deg)
        self.goal_move_distance_m = float(goal_move_distance_m)
        self.goal_move_velocity_threshold_mps = float(goal_move_velocity_threshold_mps)
        goal_seed = self.seed + 200_003 if goal_move_seed is None else int(goal_move_seed)
        self._goal_move_rng = np.random.default_rng(goal_seed)
        self._goal_move_enabled = self.goal_move_max_interval_s > 0.0
        self._next_goal_move_s = float("inf")

        self.workspace_half_m = float(workspace_half_m)
        if self.workspace_half_m <= 0.0:
            raise ValueError("workspace_half_m must be positive")
        self.success_threshold = float(success_threshold)
        if not 0.0 < self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in (0, 1]")
        # coverage_fraction is goal-overlap-area / goal-area, which caps at
        # (object_radius/goal_radius)^2 whenever the object is smaller than
        # the goal -- catch a success_threshold that's silently unreachable
        # given the configured radii, rather than let someone discover it
        # only after collecting a dataset that can never record a success.
        max_achievable_coverage = min(
            1.0, (self.properties.object_radius_m / self.properties.goal_radius_m) ** 2
        )
        if self.success_threshold > max_achievable_coverage + 1e-9:
            raise ValueError(
                f"--success-threshold {self.success_threshold:.3f} is unreachable with "
                f"object_radius_m={self.properties.object_radius_m:.4f} and "
                f"goal_radius_m={self.properties.goal_radius_m:.4f} (max possible "
                f"coverage is {max_achievable_coverage:.3f} at perfect centering) -- "
                f"lower --success-threshold, raise --object-radius, or lower --goal-radius"
            )

        self.physics = self._build_physics(
            self.properties, self.goal_xy,
            pusher_joint_damping=self.pusher_joint_damping,
            noslip_iterations=self.noslip_iterations,
        )
        self.model = self.physics.model
        self.data = self.physics.data

        sensor_sample_hz = 1.0 / float(self.model.opt.timestep)
        if self.force_sensor_cutoff_hz >= 0.5 * sensor_sample_hz:
            raise ValueError(
                "force_sensor_cutoff_hz must be below the physics Nyquist frequency"
            )
        self._sensor_alpha = None
        if self.force_sensor_cutoff_hz > 0.0:
            tau = 1.0 / (2.0 * np.pi * self.force_sensor_cutoff_hz)
            dt = float(self.model.opt.timestep)
            self._sensor_alpha = 1.0 - np.exp(-dt / tau)
        self._sensor_stage1 = np.zeros(2, dtype=float)
        self._sensor_stage2 = np.zeros(2, dtype=float)

        dt = float(self.model.opt.timestep)
        self._disturbance_decay = float(np.exp(-dt / self.disturbance_tau_s))
        noise_factor = float(np.sqrt(1.0 - self._disturbance_decay**2))
        self._disturbance_noise_scale = self.disturbance_force_n * noise_factor

        self.object_qpos_ids = np.array(
            [int(np.asarray(self.model.joint(n).qposadr).item())
             for n in ("object_slide_x", "object_slide_y")],
            dtype=np.int32,
        )
        self.object_dof_ids = np.array(
            [int(np.asarray(self.model.joint(n).dofadr).item())
             for n in ("object_slide_x", "object_slide_y")],
            dtype=np.int32,
        )
        self.pusher_qpos_ids = np.array(
            [int(np.asarray(self.model.joint(n).qposadr).item())
             for n in ("pusher_slide_x", "pusher_slide_y")],
            dtype=np.int32,
        )
        self.pusher_dof_ids = np.array(
            [int(np.asarray(self.model.joint(n).dofadr).item())
             for n in ("pusher_slide_x", "pusher_slide_y")],
            dtype=np.int32,
        )
        self._object_z_qpos_id = int(
            np.asarray(self.model.joint("object_slide_z").qposadr).item()
        )
        self.object_body_id = self.model.body("object_block").id
        self.pusher_body_id = self.model.body("pusher").id
        self._goal_mocap_id = int(
            self.model.body_mocapid[self.model.body("goal_marker").id]
        )
        self.object_geom_id = self.model.geom("object_geom").id
        self.pusher_geom_id = self.model.geom("pusher_geom").id
        self.table_geom_id = self.model.geom("table_surface").id
        self._configure_pusher_contact()

        translation_kd = 2.0 * self.damping_ratio * np.sqrt(
            self.pusher_kp * self.properties.pusher_mass_kg
        )
        self.pusher_kd = float(translation_kd)

        self._contact_buf = np.zeros(6, dtype=float)
        # Sensible default start layout: object off-center, pusher on the
        # opposite side, matching push-T's own convention.
        self.default_object_start_xy = np.array([0.10, 0.0])
        self.default_pusher_start_xy = np.array([-0.10, 0.0])
        self.data.qpos[self.object_qpos_ids] = self.default_object_start_xy
        self.data.qpos[self.pusher_qpos_ids] = self.default_pusher_start_xy
        mujoco.mj_forward(self.model.ptr, self.data.ptr)
        for _ in range(400):
            mujoco.mj_step(self.model.ptr, self.data.ptr)
        self._object_rest_z = float(self.data.qpos[self._object_z_qpos_id])
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model.ptr, self.data.ptr)
        self._initial_qpos = self.data.qpos.copy()

        self._initial_goal_xy = self.goal_xy.copy()
        self._set_goal_xy(self.goal_xy)

        self._requested_target = np.zeros(2, dtype=float)
        self._drive_target = np.zeros(2, dtype=float)

        self.reset()

    # ------------------------------------------------------------- building
    @staticmethod
    def _build_physics(properties, goal_xy, pusher_joint_damping=1.0, noslip_iterations=2):
        world = mjcf.RootElement(model="push_circle_world")
        world.compiler.angle = "radian"
        world.option.timestep = 0.001
        world.option.gravity = (0.0, 0.0, -9.81)
        world.option.noslip_iterations = int(noslip_iterations)

        world.asset.add(
            "texture", name="grid", type="2d", builtin="checker",
            rgb1=(0.28, 0.28, 0.30), rgb2=(0.34, 0.34, 0.36), width=64, height=64,
        )
        world.asset.add("material", name="table_mat", texture="grid", texrepeat=(8, 8))

        world.worldbody.add(
            "light", pos=(0, 0, 1.2), dir=(0, 0, -1), diffuse=(0.8, 0.8, 0.8)
        )
        world.worldbody.add(
            "camera", name="topdown", pos=(0, 0, 0.9), xyaxes=(1, 0, 0, 0, 1, 0)
        )

        table_half = 0.5
        table_thickness = 0.02
        world.worldbody.add(
            "geom", name="table_surface", type="box",
            pos=(0, 0, TABLE_TOP_Z - table_thickness / 2.0),
            size=(table_half, table_half, table_thickness / 2.0),
            material="table_mat", contype=1, conaffinity=1, condim=3, priority=10,
            friction=(
                properties.table_friction,
                properties.table_torsional_friction,
                properties.table_rolling_friction,
            ),
            solref=(0.01, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        )

        # --- the puck: a free-sliding disc, no rotation DOF at all -- a
        # circle's coverage doesn't depend on orientation, so there's
        # nothing for a hinge joint to usefully add here. ------------------
        object_body = world.worldbody.add(
            "body", name="object_block",
            pos=(0.0, 0.0, TABLE_TOP_Z + properties.object_thickness_m / 2.0),
        )
        object_body.add("joint", name="object_slide_x", type="slide", axis=(1, 0, 0), damping=0.0)
        object_body.add("joint", name="object_slide_y", type="slide", axis=(0, 1, 0), damping=0.0)
        # Same out-of-plane compliance trick as push_t_teleop.py's t_slide_z:
        # without it gravity generates zero contact penetration against the
        # table, hence zero normal force, hence zero friction no matter what
        # table_friction is set to.
        object_body.add(
            "joint", name="object_slide_z", type="slide", axis=(0, 0, 1),
            damping=2.0 * np.sqrt(properties.object_mass_kg * 5000.0),
        )
        object_density = properties.object_mass_kg / (
            np.pi * properties.object_radius_m**2 * properties.object_thickness_m
        )
        object_body.add(
            "geom", name="object_geom", type="cylinder",
            size=(properties.object_radius_m, properties.object_thickness_m / 2.0),
            density=object_density, rgba=(0.85, 0.35, 0.2, 1.0),
            contype=1, conaffinity=1, condim=3, priority=0,
            friction=(0.5, 0.005, 0.0001),
            solref=(0.01, 1.0), solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        )

        # --- goal marker: a non-colliding circle outline, kinematic (mocap)
        # so --goal-move-* can reposition it live without recompiling. -----
        goal_body = world.worldbody.add(
            "body", name="goal_marker", mocap=True,
            pos=(goal_xy[0], goal_xy[1], TABLE_TOP_Z + 0.0005),
        )
        goal_body.add(
            "geom", name="goal_geom", type="cylinder",
            size=(properties.goal_radius_m, 0.0005),
            rgba=(0.2, 0.75, 0.35, 0.35), contype=0, conaffinity=0, group=2,
        )

        # --- pusher: identical to push_t_teleop.py's --------------------
        pusher_half_height = max(
            properties.object_thickness_m * 0.75, properties.object_thickness_m / 2.0 + 0.005
        )
        pusher_body = world.worldbody.add(
            "body", name="pusher", pos=(0.0, 0.0, TABLE_TOP_Z + pusher_half_height),
        )
        pusher_body.add(
            "joint", name="pusher_slide_x", type="slide", axis=(1, 0, 0),
            damping=pusher_joint_damping,
        )
        pusher_body.add(
            "joint", name="pusher_slide_y", type="slide", axis=(0, 1, 0),
            damping=pusher_joint_damping,
        )
        pusher_density = properties.pusher_mass_kg / (
            np.pi * properties.pusher_radius_m**2 * (2.0 * pusher_half_height)
        )
        pusher_body.add(
            "geom", name="pusher_geom", type="cylinder",
            size=(properties.pusher_radius_m, pusher_half_height),
            density=pusher_density, rgba=(0.2, 0.45, 0.85, 1.0),
            contype=1, conaffinity=1, condim=3, priority=20,
            friction=(properties.pusher_friction, 0.005, 0.0001),
            solref=(0.006, 1.0), solimp=(0.9, 0.95, 0.0005, 0.5, 2.0),
        )

        return mjcf.Physics.from_mjcf_model(world)

    # --------------------------------------------------------------- goal
    def _set_goal_xy(self, goal_xy):
        self.goal_xy = np.asarray(goal_xy, dtype=float).copy()
        self.data.mocap_pos[self._goal_mocap_id] = (
            self.goal_xy[0], self.goal_xy[1], TABLE_TOP_Z + 0.0005
        )

    def coverage_fraction(self):
        """Fraction of the goal circle's area currently covered by the puck."""
        distance = float(np.linalg.norm(self.object_pos - self.goal_xy))
        overlap = float(
            circle_overlap_area(distance, self.properties.object_radius_m, self.properties.goal_radius_m)
        )
        return overlap / (np.pi * self.properties.goal_radius_m**2)

    def success(self):
        return self.coverage_fraction() >= self.success_threshold

    def task_metric_value(self):
        return self.coverage_fraction()

    # ------------------------------------------------------------- state
    @property
    def object_pos(self):
        return np.asarray(self.data.qpos[self.object_qpos_ids], dtype=float).copy()

    @property
    def object_vel(self):
        return np.asarray(self.data.qvel[self.object_dof_ids], dtype=float).copy()

    @property
    def pusher_pos(self):
        return np.asarray(self.data.qpos[self.pusher_qpos_ids], dtype=float).copy()

    @property
    def pusher_vel(self):
        return np.asarray(self.data.qvel[self.pusher_dof_ids], dtype=float).copy()

    @property
    def requested_target(self):
        return self._requested_target.copy()

    @property
    def drive_target(self):
        return self._drive_target.copy()

    def pusher_contact_force_xy(self):
        """Signed (Fx, Fy) reaction on the pusher from the object, world frame."""
        total = np.zeros(3, dtype=float)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            pusher_is_1 = geom1 == self.pusher_geom_id
            pusher_is_2 = geom2 == self.pusher_geom_id
            obj_is_1 = geom1 == self.object_geom_id
            obj_is_2 = geom2 == self.object_geom_id
            if not ((pusher_is_1 and obj_is_2) or (pusher_is_2 and obj_is_1)):
                continue
            mujoco.mj_contactForce(self.model.ptr, self.data.ptr, index, self._contact_buf)
            contact_to_world = np.asarray(contact.frame, dtype=float).reshape(3, 3).T
            force = contact_to_world @ self._contact_buf[:3]
            total += force if pusher_is_2 else -force
        return total[:2].copy()

    def sensor_force_xy(self):
        """Causal two-pole low-pass of pusher_contact_force_xy -- same
        finite-bandwidth F/T sensor model as push_t_teleop.py's."""
        if self._sensor_alpha is None:
            return self.pusher_contact_force_xy()
        return self._sensor_stage2.copy()

    def _refresh_force_sensor(self):
        raw = self.pusher_contact_force_xy()
        if self._sensor_alpha is None:
            return
        self._sensor_stage1 += self._sensor_alpha * (raw - self._sensor_stage1)
        self._sensor_stage2 += self._sensor_alpha * (self._sensor_stage1 - self._sensor_stage2)

    @property
    def disturbance_force(self):
        """Current (Fx, Fy) disturbance applied to the object, ground truth
        -- not something an operator/policy could have anticipated."""
        return self._disturbance_force.copy()

    def _advance_disturbance(self):
        """One exact-discrete-time OU step, applied to the puck. See
        push_t_teleop.py's _advance_disturbance for the containment
        rationale (a sustained unopposed disturbance can walk the object
        off the table, where friction -- what it's fighting against -- goes
        to exactly zero, so the remaining force is genuinely unbounded)."""
        if self.disturbance_force_n <= 0.0:
            return
        self._disturbance_force = (
            self._disturbance_decay * self._disturbance_force
            + self._disturbance_noise_scale * self._disturbance_rng.standard_normal(2)
        )
        force_xy = self._disturbance_force.copy()
        boundary = 1.15 * self.workspace_half_m
        pos_xy = self.object_pos
        excess = pos_xy - np.clip(pos_xy, -boundary, boundary)
        if np.any(excess != 0.0):
            vel_xy = self.object_vel
            leash_kp = 300.0
            leash_kd = 2.0 * np.sqrt(leash_kp * self.properties.object_mass_kg)
            beyond = excess != 0.0
            force_xy = np.where(
                beyond, force_xy - leash_kp * excess - leash_kd * vel_xy, force_xy
            )
        self.data.qfrc_applied[self.object_dof_ids[0]] = force_xy[0]
        self.data.qfrc_applied[self.object_dof_ids[1]] = force_xy[1]

    @property
    def goal_move_active(self):
        return self._goal_move_enabled

    def _sample_goal_xy(self):
        """Usually (--goal-move-opposite-bias probability) anchored BEHIND
        the puck's current direction of travel -- reaching this goal means
        reversing course, not just continuing toward wherever it already
        was heading. Falls back to plain uniform-in-workspace whenever the
        puck is too slow for "direction of travel" to mean anything
        (--goal-move-velocity-threshold), or on the (1 - opposite_bias)
        fraction of draws that deliberately go uniform anyway, so the
        backward-bias itself isn't a perfectly reliable pattern either.
        """
        speed = float(np.linalg.norm(self.object_vel))
        use_opposite = (
            speed > self.goal_move_velocity_threshold_mps
            and self._goal_move_rng.random() < self.goal_move_opposite_bias
        )
        if use_opposite:
            heading = float(np.arctan2(self.object_vel[1], self.object_vel[0]))
            angle = (
                heading + np.pi
                + self._goal_move_rng.uniform(
                    -self.goal_move_opposite_spread_rad, self.goal_move_opposite_spread_rad
                )
            )
            distance = self._goal_move_rng.uniform(
                0.5 * self.goal_move_distance_m, self.goal_move_distance_m
            )
            candidate = self.object_pos + distance * np.array([np.cos(angle), np.sin(angle)])
        else:
            candidate = self._goal_move_rng.uniform(
                -self.goal_move_xy_half_m, self.goal_move_xy_half_m, size=2
            )
        return np.clip(candidate, -self.goal_move_xy_half_m, self.goal_move_xy_half_m)

    def _schedule_next_goal_move(self):
        if not self._goal_move_enabled:
            self._next_goal_move_s = float("inf")
            return
        interval = self._goal_move_rng.uniform(
            self.goal_move_min_interval_s, self.goal_move_max_interval_s
        )
        self._next_goal_move_s = float(self.data.time) + interval

    def _maybe_move_goal(self):
        if not self._goal_move_enabled or self.data.time < self._next_goal_move_s:
            return
        if self._goal_move_rng.random() >= self.goal_move_skip_prob:
            self._set_goal_xy(self._sample_goal_xy())
        self._schedule_next_goal_move()

    # ------------------------------------------------------------ control
    def limited_target(self, target_xy):
        target = np.asarray(target_xy, dtype=float)
        return np.clip(target, -self.workspace_half_m, self.workspace_half_m)

    def step(self, target_xy, n_substeps=1):
        target_xy = np.asarray(target_xy, dtype=float)
        for _ in range(max(1, int(n_substeps))):
            self._requested_target = target_xy.copy()
            self._drive_target = self.limited_target(self._requested_target)
            force_xy = self.pusher_kp * (
                self._drive_target - self.pusher_pos
            ) - self.pusher_kd * self.pusher_vel
            self.data.qfrc_applied[self.pusher_dof_ids] = force_xy
            self._advance_disturbance()
            mujoco.mj_step(self.model.ptr, self.data.ptr)
            self._refresh_force_sensor()
            self._maybe_move_goal()
        return self

    def reset(self, *, object_xy=None, pusher_xy=None):
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        if object_xy is not None:
            self.data.qpos[self.object_qpos_ids] = np.asarray(object_xy, dtype=float)
            self.data.qpos[self._object_z_qpos_id] = self._object_rest_z
        if pusher_xy is not None:
            self.data.qpos[self.pusher_qpos_ids] = np.asarray(pusher_xy, dtype=float)
        mujoco.mj_forward(self.model.ptr, self.data.ptr)
        self._requested_target = self.pusher_pos.copy()
        self._drive_target = self._requested_target.copy()
        self._sensor_stage1[:] = 0.0
        self._sensor_stage2[:] = 0.0
        self._disturbance_force[:] = 0.0
        self._set_goal_xy(self._initial_goal_xy)
        self._schedule_next_goal_move()
        return self

    def _configure_pusher_contact(self):
        t = self.pusher_softness
        time_constant = HARD_PUSHER_SOLREF[0] + t * (SOFT_PUSHER_SOLREF[0] - HARD_PUSHER_SOLREF[0])
        damping_ratio = HARD_PUSHER_SOLREF[1] + t * (SOFT_PUSHER_SOLREF[1] - HARD_PUSHER_SOLREF[1])
        width = HARD_PUSHER_SOLIMP_WIDTH + t * (SOFT_PUSHER_SOLIMP_WIDTH - HARD_PUSHER_SOLIMP_WIDTH)
        d0 = HARD_PUSHER_SOLIMP_D0 + t * (SOFT_PUSHER_SOLIMP_D0 - HARD_PUSHER_SOLIMP_D0)
        self.model.geom_solref[self.pusher_geom_id] = (time_constant, damping_ratio)
        self.model.geom_solimp[self.pusher_geom_id, 0] = d0
        self.model.geom_solimp[self.pusher_geom_id, 2] = width

    def _pusher_overlaps_object(self):
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 == self.pusher_geom_id and g2 == self.object_geom_id) or (
                g2 == self.pusher_geom_id and g1 == self.object_geom_id
            ):
                return True
        return False

    def randomize_start(
        self,
        *,
        object_xy_half=0.15,
        pusher_xy_half=0.20,
        center_probability=0.70,
        force_center=False,
        max_resamples=64,
    ):
        """Same center-biased mixture + overlap-rejection sampling as
        push_t_teleop.py's randomize_start, simplified to 2D positions only
        (no orientation to sample for a circle)."""
        clearance = (
            self.properties.object_radius_m + self.properties.pusher_radius_m + 0.01
        )

        def draw():
            if force_center:
                normalized = np.zeros(4)
            elif self._rng.random() < center_probability:
                normalized = np.clip(self._rng.normal(0.0, 0.22, size=4), -0.5, 0.5)
            else:
                normalized = self._rng.uniform(-0.5, 0.5, size=4)
            half_extents = np.array([object_xy_half, object_xy_half, pusher_xy_half, pusher_xy_half])
            scaled = normalized * 2.0 * half_extents
            return scaled[0:2], scaled[2:4]

        object_xy, pusher_xy = draw()
        for _ in range(max(0, int(max_resamples))):
            self.reset(object_xy=object_xy, pusher_xy=pusher_xy)
            if not self._pusher_overlaps_object():
                return self
            object_xy, pusher_xy = draw()

        offset = pusher_xy - object_xy
        distance = float(np.linalg.norm(offset))
        if distance < 1e-9:
            angle = self._rng.uniform(0.0, 2.0 * np.pi)
            offset = np.array([np.cos(angle), np.sin(angle)])
            distance = 1.0
        pusher_xy = object_xy + offset / distance * clearance
        self.reset(object_xy=object_xy, pusher_xy=pusher_xy)
        return self

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
