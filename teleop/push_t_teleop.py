"""2D Push-T task: a floating disc pusher shoves a T-shaped block to a target.

This is deliberately self-contained (no gripper, no arm, no attached asset
tree) -- just a flat table, a T-shaped block with real mass/inertia, and a
circular pusher, all built directly with ``dm_control.mjcf``. It models two
independent, configurable friction interfaces:

- T-block <-> table: controls whether the T keeps sliding after a push or
  stops immediately (``--table-friction`` in the CLI / ``table_friction``
  here). This is a real geom-friction coefficient, so the same push impulse
  travels farther at low friction and barely moves the block at high
  friction -- no special-cased "sliding" logic, it's just Coulomb friction.
- pusher <-> T-block: controls how much the pusher can "grip" the T's sides
  while pushing (``--pusher-friction`` / ``pusher_friction``). The pusher is
  a cylinder tall enough to contact the T's side faces (not just push it from
  directly above), so nudging it against any face -- bar or stem, from any
  side -- pushes and/or drags the T, matching the classic 2D push-T task.

Both interfaces are decoupled via MuJoCo's contact ``priority``: the T's own
geoms carry priority 0 (lowest), so in a T-vs-table contact the table geom's
friction wins, and in a T-vs-pusher contact the pusher geom's friction wins.

Success is the fraction of the FIXED GOAL T's footprint currently covered by
the T-block, computed by rasterizing both T footprints (each the union of two
axis-aligned-in-local-frame rectangles) on a fine grid and comparing boolean
masks -- exact for this shape, fully vectorized, and needs no extra
dependency (e.g. shapely) for polygon boolean ops.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from dm_control import mjcf


TABLE_TOP_Z = 0.0
DEFAULT_GOAL_POSE = (0.0, 0.0, 0.0)

# pusher_softness=0 reproduces the compiled pusher_geom defaults (a fairly
# stiff, fast-correcting contact -- solref (0.006, 1.0), solimp width
# 0.0005), which combined with a stiffened --pusher-kp can chatter/spike on
# the single-point cylinder-vs-box contact this task uses. softness=1 widens
# the impedance ramp and slows/damps the constraint correction, same
# time-constant/damping-ratio/width convention as floating_flipup_teleop's
# --tip-softness (not a calibrated material model there either -- just a
# controlled way to make the contact respond less like two rigid bodies).
HARD_PUSHER_SOLREF = (0.006, 1.0)
SOFT_PUSHER_SOLREF = (0.020, 2.0)
HARD_PUSHER_SOLIMP_WIDTH = 0.0005
SOFT_PUSHER_SOLIMP_WIDTH = 0.005
# solimp's d0 (impedance at zero penetration) also scales with softness now,
# not just width: at the compiled d0=0.9 the constraint is already almost
# fully enforced the instant a contact point activates, which is exactly
# the abrupt on/off behavior that makes the raw force flicker as MuJoCo
# adds/drops points tick to tick. Lowering d0 a bit gives contact points a
# genuine ease-in instead of being either off or nearly-fully-active.
HARD_PUSHER_SOLIMP_D0 = 0.9
SOFT_PUSHER_SOLIMP_D0 = 0.7


@dataclass(frozen=True)
class PushTProperties:
    """Physical properties of the T-block and pusher (all independently tunable)."""

    t_mass_kg: float = 0.20
    t_bar_length_m: float = 0.12
    t_bar_width_m: float = 0.03
    t_stem_length_m: float = 0.09
    t_stem_width_m: float = 0.03
    t_thickness_m: float = 0.02
    # T <-> table sliding friction. Low values (e.g. 0.05) let the T coast
    # several centimetres after a push is released; high values (e.g. 1.5)
    # stop it almost as soon as the pusher lets go.
    table_friction: float = 0.4
    table_torsional_friction: float = 0.006
    table_rolling_friction: float = 0.0002
    # pusher <-> T sliding friction on side contact. Governs how much the
    # round pusher can drag the T tangentially (vs. only shoving it straight
    # back along the contact normal) when nudged against a face at an angle.
    pusher_friction: float = 0.6
    pusher_radius_m: float = 0.018
    pusher_mass_kg: float = 0.05

    def __post_init__(self):
        positive = (
            self.t_mass_kg,
            self.t_bar_length_m,
            self.t_bar_width_m,
            self.t_stem_length_m,
            self.t_stem_width_m,
            self.t_thickness_m,
            self.pusher_radius_m,
            self.pusher_mass_kg,
        )
        if any(v <= 0.0 for v in positive):
            raise ValueError("PushTProperties dimensions/masses must be positive")
        frictions = (
            self.table_friction,
            self.table_torsional_friction,
            self.table_rolling_friction,
            self.pusher_friction,
        )
        if any(v < 0.0 for v in frictions):
            raise ValueError("friction coefficients cannot be negative")

    @property
    def t_area_m2(self):
        return (
            self.t_bar_length_m * self.t_bar_width_m
            + self.t_stem_width_m * self.t_stem_length_m
        )

    @property
    def stem_offset_y_m(self):
        """Distance from the T's local origin (bar center) to the stem center."""
        return -(0.5 * self.t_bar_width_m + 0.5 * self.t_stem_length_m)


DEFAULT_PUSH_T_PROPERTIES = PushTProperties()


class PushTTeleop:
    """Direct-impedance 2D pusher shoving a T-block toward a fixed goal pose."""

    task_kind = "push_t"
    default_tool_kp = 400.0
    default_max_speed = 0.5

    def __init__(
        self,
        seed=0,
        properties=None,
        goal_pose=DEFAULT_GOAL_POSE,
        pusher_kp=400.0,
        damping_ratio=1.2,
        workspace_half_m=0.30,
        success_threshold=0.95,
        coverage_grid_resolution_m=0.001,
        settle_s=0.0,
        pusher_softness=0.0,
        force_sensor_cutoff_hz=0.0,
        pusher_joint_damping=1.0,
        noslip_iterations=2,
        t_disturbance_force_n=0.0,
        t_disturbance_torque_n_m=0.0,
        t_disturbance_tau_s=1.0,
        t_disturbance_seed=None,
        goal_move_min_interval_s=0.0,
        goal_move_max_interval_s=0.0,
        goal_move_xy_half_m=0.15,
        goal_move_skip_prob=0.0,
        goal_move_seed=None,
    ):
        del settle_s
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.properties = properties or DEFAULT_PUSH_T_PROPERTIES
        if not isinstance(self.properties, PushTProperties):
            raise TypeError("properties must be a PushTProperties instance")
        self.goal_pose = np.asarray(goal_pose, dtype=float).copy()
        if self.goal_pose.shape != (3,):
            raise ValueError("goal_pose must be (x, y, theta_rad)")
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
        if float(t_disturbance_force_n) < 0.0:
            raise ValueError("t_disturbance_force_n cannot be negative")
        if float(t_disturbance_torque_n_m) < 0.0:
            raise ValueError("t_disturbance_torque_n_m cannot be negative")
        if float(t_disturbance_tau_s) <= 0.0:
            raise ValueError("t_disturbance_tau_s must be positive")
        self.t_disturbance_force_n = float(t_disturbance_force_n)
        self.t_disturbance_torque_n_m = float(t_disturbance_torque_n_m)
        self.t_disturbance_tau_s = float(t_disturbance_tau_s)
        # Separate stream from self._rng (start-pose sampling) so changing
        # one doesn't perturb the other's draw sequence; offset from the
        # main seed so seed=N doesn't silently reuse seed=N's disturbance
        # stream for some other purpose that happens to seed with N too.
        disturbance_seed = (
            self.seed + 100_003 if t_disturbance_seed is None else int(t_disturbance_seed)
        )
        self._disturbance_rng = np.random.default_rng(disturbance_seed)
        self._disturbance_force = np.zeros(2, dtype=float)
        self._disturbance_torque = 0.0
        if float(goal_move_min_interval_s) < 0.0 or float(goal_move_max_interval_s) < 0.0:
            raise ValueError("goal_move_min/max_interval_s cannot be negative")
        if goal_move_max_interval_s > 0.0 and goal_move_min_interval_s > goal_move_max_interval_s:
            raise ValueError("goal_move_min_interval_s must be <= goal_move_max_interval_s")
        if float(goal_move_xy_half_m) <= 0.0:
            raise ValueError("goal_move_xy_half_m must be positive")
        if not 0.0 <= float(goal_move_skip_prob) < 1.0:
            raise ValueError("goal_move_skip_prob must be in [0, 1)")
        self.goal_move_min_interval_s = float(goal_move_min_interval_s)
        self.goal_move_max_interval_s = float(goal_move_max_interval_s)
        self.goal_move_xy_half_m = float(goal_move_xy_half_m)
        self.goal_move_skip_prob = float(goal_move_skip_prob)
        # 0 (the default) disables goal-moving entirely -- next_goal_move_s
        # of +inf means "never fires" without needing an enabled/disabled
        # branch sprinkled through step().
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

        self.physics = self._build_physics(
            self.properties, self.goal_pose,
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

        # Exact discrete-time Ornstein-Uhlenbeck coefficients (stable for any
        # dt/tau ratio, unlike an Euler update): decay shrinks the current
        # disturbance each tick, noise_scale injects fresh randomness sized
        # so the process's STATIONARY std matches the requested magnitude
        # (t_disturbance_force_n/torque_n_m), not the per-tick noise itself.
        # A high tau gives a slow, smoothly-wandering push (genuinely hard to
        # predict a step ahead but not violent); a low tau gives fast,
        # buzzy jitter -- same "path" either way, just a different bandwidth.
        dt = float(self.model.opt.timestep)
        self._disturbance_decay = float(np.exp(-dt / self.t_disturbance_tau_s))
        noise_factor = float(np.sqrt(1.0 - self._disturbance_decay ** 2))
        self._disturbance_force_noise_scale = self.t_disturbance_force_n * noise_factor
        self._disturbance_torque_noise_scale = self.t_disturbance_torque_n_m * noise_factor

        self.t_free_joint_ids = np.array(
            [
                int(np.asarray(self.model.joint(name).qposadr).item())
                for name in ("t_slide_x", "t_slide_y", "t_hinge_theta")
            ],
            dtype=np.int32,
        )
        self.t_dof_ids = np.array(
            [
                int(np.asarray(self.model.joint(name).dofadr).item())
                for name in ("t_slide_x", "t_slide_y", "t_hinge_theta")
            ],
            dtype=np.int32,
        )
        self.pusher_qpos_ids = np.array(
            [
                int(np.asarray(self.model.joint(name).qposadr).item())
                for name in ("pusher_slide_x", "pusher_slide_y")
            ],
            dtype=np.int32,
        )
        self.pusher_dof_ids = np.array(
            [
                int(np.asarray(self.model.joint(name).dofadr).item())
                for name in ("pusher_slide_x", "pusher_slide_y")
            ],
            dtype=np.int32,
        )
        self._t_z_qpos_id = int(
            np.asarray(self.model.joint("t_slide_z").qposadr).item()
        )
        self.t_body_id = self.model.body("t_block").id
        self.pusher_body_id = self.model.body("pusher").id
        self._goal_mocap_id = int(
            self.model.body_mocapid[self.model.body("goal_marker").id]
        )
        self.t_geom_ids = np.array(
            [self.model.geom(name).id for name in ("t_bar", "t_stem")],
            dtype=np.int32,
        )
        self.pusher_geom_id = self.model.geom("pusher_geom").id
        self.table_geom_id = self.model.geom("table_surface").id
        self._configure_pusher_contact()

        translation_kd = 2.0 * self.damping_ratio * np.sqrt(
            self.pusher_kp * self.properties.pusher_mass_kg
        )
        self.pusher_kd = float(translation_kd)

        self._contact_buf = np.zeros(6, dtype=float)
        # Sensible default start layout: T off-center, pusher on the
        # opposite side so an operator has room to approach it.
        self.default_t_start_pose = np.array([0.10, 0.0, 0.0])
        self.default_pusher_start_pos = np.array([-0.10, 0.0])
        self.data.qpos[self.t_free_joint_ids] = self.default_t_start_pose
        self.data.qpos[self.pusher_qpos_ids] = self.default_pusher_start_pos
        mujoco.mj_forward(self.model.ptr, self.data.ptr)
        # Let the T settle onto the table under its own weight once, and
        # cache the resulting equilibrium t_slide_z penetration. Flat T on a
        # flat table means this equilibrium is the same regardless of
        # (x, y, theta), so every later reset() can just jump straight to it
        # instead of re-simulating a settle each episode.
        for _ in range(400):
            mujoco.mj_step(self.model.ptr, self.data.ptr)
        self._t_rest_z = float(self.data.qpos[self._t_z_qpos_id])
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        mujoco.mj_forward(self.model.ptr, self.data.ptr)
        self._initial_qpos = self.data.qpos.copy()

        # The goal pose passed in at construction -- reset() puts the goal
        # back here, distinct from wherever --goal-move-* last relocated it.
        self._initial_goal_pose = self.goal_pose.copy()
        self._build_goal_grid(float(coverage_grid_resolution_m))

        self._requested_target = np.zeros(2, dtype=float)
        self._drive_target = np.zeros(2, dtype=float)

        self.reset()

    # ------------------------------------------------------------- building
    @staticmethod
    def _build_physics(properties, goal_pose, pusher_joint_damping=1.0, noslip_iterations=2):
        world = mjcf.RootElement(model="push_t_world")
        world.compiler.angle = "radian"
        world.option.timestep = 0.001
        world.option.gravity = (0.0, 0.0, -9.81)
        # noslip_iterations>0 runs extra PGS passes on the friction cone after
        # the main solve, specifically to reduce slip-direction error --
        # helps the round pusher's sliding contact settle onto a consistent
        # friction solution instead of flip-flopping between nearby ones
        # tick to tick, one contributor to the jittery raw force signal.
        world.option.noslip_iterations = int(noslip_iterations)

        world.asset.add(
            "texture",
            name="grid",
            type="2d",
            builtin="checker",
            rgb1=(0.28, 0.28, 0.30),
            rgb2=(0.34, 0.34, 0.36),
            width=64,
            height=64,
        )
        world.asset.add(
            "material", name="table_mat", texture="grid", texrepeat=(8, 8)
        )

        world.worldbody.add(
            "light", pos=(0, 0, 1.2), dir=(0, 0, -1), diffuse=(0.8, 0.8, 0.8)
        )
        world.worldbody.add(
            "camera",
            name="topdown",
            pos=(0, 0, 0.9),
            xyaxes=(1, 0, 0, 0, 1, 0),
        )

        table_half = 0.5
        table_thickness = 0.02
        world.worldbody.add(
            "geom",
            name="table_surface",
            type="box",
            pos=(0, 0, TABLE_TOP_Z - table_thickness / 2.0),
            size=(table_half, table_half, table_thickness / 2.0),
            material="table_mat",
            contype=1,
            conaffinity=1,
            condim=3,
            priority=10,
            friction=(
                properties.table_friction,
                properties.table_torsional_friction,
                properties.table_rolling_friction,
            ),
            solref=(0.01, 1.0),
            solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        )

        # --- T-block: bar + stem, free to slide/spin on the table plane ---
        # Anchored at the origin (only z is fixed): the t_slide_x/t_slide_y
        # joint qpos values are then directly the T's absolute (x, y) world
        # position, matching how t_pose/reset/coverage_fraction read them.
        # (A nonzero anchor x/y here would silently offset qpos from world
        # position -- exactly the bug that first made the pusher sail
        # through the T with zero contact.)
        t_body = world.worldbody.add(
            "body",
            name="t_block",
            pos=(0.0, 0.0, TABLE_TOP_Z + properties.t_thickness_m / 2.0),
        )
        t_body.add("joint", name="t_slide_x", type="slide", axis=(1, 0, 0), damping=0.0)
        t_body.add("joint", name="t_slide_y", type="slide", axis=(0, 1, 0), damping=0.0)
        t_body.add(
            "joint", name="t_hinge_theta", type="hinge", axis=(0, 0, 1), damping=0.0
        )
        # A small out-of-plane compliance DOF -- NOT part of the public
        # (x, y, theta) pose. Without this, the T has no way to be pushed
        # DOWN into the table by its own weight, so gravity generates zero
        # contact penetration and therefore exactly zero normal force --
        # and friction (mu * normal_force) is then zero no matter what
        # table_friction is set to. This lets gravity load the contact for
        # real, while staying critically damped so it settles in a few ms
        # and never visibly bobs.
        t_body.add(
            "joint",
            name="t_slide_z",
            type="slide",
            axis=(0, 0, 1),
            damping=2.0 * np.sqrt(properties.t_mass_kg * 5000.0),
        )
        t_density = properties.t_mass_kg / (
            properties.t_area_m2 * properties.t_thickness_m
        )
        t_body.add(
            "geom",
            name="t_bar",
            type="box",
            pos=(0, 0, 0),
            size=(
                properties.t_bar_length_m / 2.0,
                properties.t_bar_width_m / 2.0,
                properties.t_thickness_m / 2.0,
            ),
            density=t_density,
            rgba=(0.85, 0.35, 0.2, 1.0),
            contype=1,
            conaffinity=1,
            condim=3,
            priority=0,
            friction=(0.5, 0.005, 0.0001),
            solref=(0.01, 1.0),
            solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        )
        t_body.add(
            "geom",
            name="t_stem",
            type="box",
            pos=(0, properties.stem_offset_y_m, 0),
            size=(
                properties.t_stem_width_m / 2.0,
                properties.t_stem_length_m / 2.0,
                properties.t_thickness_m / 2.0,
            ),
            density=t_density,
            rgba=(0.85, 0.35, 0.2, 1.0),
            contype=1,
            conaffinity=1,
            condim=3,
            priority=0,
            friction=(0.5, 0.005, 0.0001),
            solref=(0.01, 1.0),
            solimp=(0.9, 0.95, 0.001, 0.5, 2.0),
        )

        # --- goal marker: identical T footprint, non-colliding, kinematic --
        # A mocap body (no joint, no inertia -- purely kinematic), not a
        # static geom baked in at this pos/euler: --goal-move-* repositions
        # it live via data.mocap_pos/mocap_quat, which a compiled-in static
        # body could not do without recompiling the whole model.
        goal_body = world.worldbody.add(
            "body",
            name="goal_marker",
            mocap=True,
            pos=(goal_pose[0], goal_pose[1], TABLE_TOP_Z + 0.0005),
            euler=(0, 0, goal_pose[2]),
        )
        goal_body.add(
            "geom",
            name="goal_bar",
            type="box",
            pos=(0, 0, 0),
            size=(
                properties.t_bar_length_m / 2.0,
                properties.t_bar_width_m / 2.0,
                0.0005,
            ),
            rgba=(0.2, 0.75, 0.35, 0.35),
            contype=0,
            conaffinity=0,
            group=2,
        )
        goal_body.add(
            "geom",
            name="goal_stem",
            type="box",
            pos=(0, properties.stem_offset_y_m, 0),
            size=(
                properties.t_stem_width_m / 2.0,
                properties.t_stem_length_m / 2.0,
                0.0005,
            ),
            rgba=(0.2, 0.75, 0.35, 0.35),
            contype=0,
            conaffinity=0,
            group=2,
        )

        # --- pusher: a disc confined to the table plane, tall enough to --
        # --- contact the T's side faces, not just push from above -------
        pusher_half_height = max(
            properties.t_thickness_m * 0.75, properties.t_thickness_m / 2.0 + 0.005
        )
        # Anchored at the origin too, for the same reason as t_block above.
        pusher_body = world.worldbody.add(
            "body",
            name="pusher",
            pos=(0.0, 0.0, TABLE_TOP_Z + pusher_half_height),
        )
        # Physical joint damping, distinct from the controller's task-space
        # kd (pusher_kd): this is passive dissipation MuJoCo applies every
        # substep at the DOF itself, which can damp inter-tick velocity
        # oscillations the 1 kHz position controller can't see or correct
        # for between its own samples. Was 0.0 (relying entirely on the
        # controller's kd), which left the joints undamped for anything the
        # controller doesn't explicitly counteract.
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
            "geom",
            name="pusher_geom",
            type="cylinder",
            size=(properties.pusher_radius_m, pusher_half_height),
            density=pusher_density,
            rgba=(0.2, 0.45, 0.85, 1.0),
            contype=1,
            conaffinity=1,
            condim=3,
            priority=20,
            friction=(properties.pusher_friction, 0.005, 0.0001),
            solref=(0.006, 1.0),
            solimp=(0.9, 0.95, 0.0005, 0.5, 2.0),
        )

        return mjcf.Physics.from_mjcf_model(world)

    # --------------------------------------------------------------- goal
    def _build_goal_grid(self, resolution_m):
        """Cache the goal footprint's LOCAL-frame grid/mask once -- these
        depend only on the T's fixed geometry, never on where the goal
        currently is, so a goal move (``_set_goal_pose``) only needs a cheap
        rotate+translate of this cached grid, not a full rebuild."""
        if resolution_m <= 0.0:
            raise ValueError("coverage_grid_resolution_m must be positive")
        margin = 0.01
        half_x = self.properties.t_bar_length_m / 2.0 + margin
        top_y = self.properties.t_bar_width_m / 2.0 + margin
        bottom_y = (
            -self.properties.stem_offset_y_m
            + self.properties.t_stem_length_m / 2.0
            + margin
        )
        xs = np.arange(-half_x, half_x, resolution_m)
        ys = np.arange(-bottom_y, top_y, resolution_m)
        grid_x, grid_y = np.meshgrid(xs, ys)
        self._goal_local_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        self._goal_mask = self._t_footprint_mask(self._goal_local_points)
        self._goal_mask_count = int(self._goal_mask.sum())
        if self._goal_mask_count == 0:
            raise RuntimeError("goal T footprint grid is empty; lower resolution")
        self._set_goal_pose(self.goal_pose)

    def _set_goal_pose(self, goal_pose):
        """Move the goal (coverage target AND the visual marker) to
        ``goal_pose``, without touching the T-block, the pusher, or
        anything else -- used both for the initial placement and by
        --goal-move-* to relocate it mid-episode."""
        goal_pose = np.asarray(goal_pose, dtype=float)
        self.goal_pose = goal_pose.copy()
        cos_g, sin_g = np.cos(goal_pose[2]), np.sin(goal_pose[2])
        rotation = np.array([[cos_g, -sin_g], [sin_g, cos_g]])
        self._goal_grid_xy = self._goal_local_points @ rotation.T + goal_pose[:2]
        if self._goal_mocap_id is not None:
            self.data.mocap_pos[self._goal_mocap_id] = (
                goal_pose[0], goal_pose[1], TABLE_TOP_Z + 0.0005
            )
            half_theta = 0.5 * goal_pose[2]
            self.data.mocap_quat[self._goal_mocap_id] = (
                np.cos(half_theta), 0.0, 0.0, np.sin(half_theta)
            )

    def _t_footprint_mask(self, points_local):
        """Boolean mask of ``points_local`` (N, 2) inside the T's own footprint."""
        p = np.asarray(points_local, dtype=float)
        half_bar = np.array(
            [self.properties.t_bar_length_m / 2.0, self.properties.t_bar_width_m / 2.0]
        )
        half_stem = np.array(
            [self.properties.t_stem_width_m / 2.0, self.properties.t_stem_length_m / 2.0]
        )
        in_bar = (np.abs(p[:, 0]) <= half_bar[0]) & (np.abs(p[:, 1]) <= half_bar[1])
        in_stem = (np.abs(p[:, 0]) <= half_stem[0]) & (
            np.abs(p[:, 1] - self.properties.stem_offset_y_m) <= half_stem[1]
        )
        return in_bar | in_stem

    def coverage_fraction(self):
        """Fraction of the FIXED GOAL T's footprint currently covered by the T-block."""
        x, y, theta = self.t_pose
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        local_points = (self._goal_grid_xy - np.array([x, y])) @ rotation.T
        current_mask = self._t_footprint_mask(local_points)
        intersection = current_mask & self._goal_mask
        return float(intersection.sum()) / float(self._goal_mask_count)

    def success(self):
        return self.coverage_fraction() >= self.success_threshold

    def task_metric_value(self):
        return self.coverage_fraction()

    # ------------------------------------------------------------- state
    @property
    def t_pose(self):
        """(x, y, theta_rad) of the T-block in the world/table frame."""
        return np.asarray(self.data.qpos[self.t_free_joint_ids], dtype=float).copy()

    @property
    def t_twist(self):
        return np.asarray(self.data.qvel[self.t_dof_ids], dtype=float).copy()

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
        """Target actually driving the spring, after ``limited_target``.

        Differs from ``requested_target`` only while the workspace clamp is
        active -- kept separate so a recorded dataset can distinguish the
        operator's raw request from what the controller actually received.
        """
        return self._drive_target.copy()

    def pusher_contact_force(self):
        """Net contact force (N, magnitude) between the pusher and the T-block."""
        total = np.zeros(3, dtype=float)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            pusher_is_1 = geom1 == self.pusher_geom_id
            pusher_is_2 = geom2 == self.pusher_geom_id
            t_is_1 = geom1 in self.t_geom_ids
            t_is_2 = geom2 in self.t_geom_ids
            if not ((pusher_is_1 and t_is_2) or (pusher_is_2 and t_is_1)):
                continue
            mujoco.mj_contactForce(
                self.model.ptr, self.data.ptr, index, self._contact_buf
            )
            contact_to_world = np.asarray(contact.frame, dtype=float).reshape(3, 3).T
            force = contact_to_world @ self._contact_buf[:3]
            total += force if pusher_is_2 else -force
        return float(np.linalg.norm(total))

    def pusher_contact_force_xy(self):
        """Signed (Fx, Fy) reaction on the pusher from the T-block, world frame."""
        total = np.zeros(3, dtype=float)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            pusher_is_1 = geom1 == self.pusher_geom_id
            pusher_is_2 = geom2 == self.pusher_geom_id
            t_is_1 = geom1 in self.t_geom_ids
            t_is_2 = geom2 in self.t_geom_ids
            if not ((pusher_is_1 and t_is_2) or (pusher_is_2 and t_is_1)):
                continue
            mujoco.mj_contactForce(
                self.model.ptr, self.data.ptr, index, self._contact_buf
            )
            contact_to_world = np.asarray(contact.frame, dtype=float).reshape(3, 3).T
            force = contact_to_world @ self._contact_buf[:3]
            # Reaction felt BY the pusher, i.e. force pointing away from the T.
            # Same sign convention as pusher_contact_force() above and
            # flipup_teleop.py's contact_wrench() (validated against a real
            # wrist F/T sensor): keep sign when the special-role geom (here,
            # the pusher) is geom2, negate when it's geom1.
            total += force if pusher_is_2 else -force
        return total[:2].copy()

    def sensor_force_xy(self):
        """Causal two-pole low-pass of ``pusher_contact_force_xy``, updated
        every physics tick in ``step`` -- same finite-bandwidth F/T sensor
        model as floating_flipup_teleop.py's, at ``force_sensor_cutoff_hz``.

        The raw solver wrench changes discretely as MuJoCo's contact solver
        adds/drops contact points tick to tick (a round pusher against a
        boxy T can flicker between several simultaneous contacts even while
        the commanded target barely moves) -- this doesn't damp out that by
        smoothing a genuinely noisy signal, it mimics what a real F/T sensor's
        limited bandwidth would report instead of that raw discrete signal.
        With ``force_sensor_cutoff_hz == 0`` this returns the identical raw
        value as ``pusher_contact_force_xy`` (no filtering).
        """
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
    def t_disturbance_wrench(self):
        """Current (Fx, Fy, Mz) disturbance applied to the T, ground truth.

        Not something a live operator can see coming -- it's an unforced
        stochastic process, not a scripted path -- but recorded so an
        offline learner/analysis can tell "the T moved on its own" from
        "the operator's push did that".
        """
        return np.array(
            [self._disturbance_force[0], self._disturbance_force[1], self._disturbance_torque]
        )

    def _advance_disturbance(self):
        """One exact-discrete-time OU step, applied to the T's own DOFs.

        A no-op (returns the zero vector, same as if disabled) whenever both
        magnitudes are 0 -- the default -- so this task is unchanged unless
        --t-disturbance-force/--t-disturbance-torque are explicitly set.

        A sustained wandering force with no operator contact can walk the T
        off the table surface entirely -- once it's off, table friction
        (which is what the disturbance is fighting against) goes to exactly
        zero, so the remaining unopposed disturbance force is genuinely
        unbounded (measured: a T pushed with the pusher held away from it
        reached ~100m from origin in 6 simulated seconds under a modest 1.5N
        disturbance). This isn't a real task failure mode to design around --
        it's what happens if nobody is playing -- but it needs a floor so an
        idle/disconnected episode can't blow up the sim. A soft spring+damper
        activates only once the T strays beyond 1.15x the pusher's own
        workspace, pulling it back without adding any drag or resistance
        inside the actual play area.
        """
        if self.t_disturbance_force_n <= 0.0 and self.t_disturbance_torque_n_m <= 0.0:
            return
        self._disturbance_force = (
            self._disturbance_decay * self._disturbance_force
            + self._disturbance_force_noise_scale * self._disturbance_rng.standard_normal(2)
        )
        self._disturbance_torque = (
            self._disturbance_decay * self._disturbance_torque
            + self._disturbance_torque_noise_scale * self._disturbance_rng.standard_normal()
        )
        force_xy = self._disturbance_force.copy()
        boundary = 1.15 * self.workspace_half_m
        pos_xy = np.asarray(self.data.qpos[self.t_free_joint_ids[:2]], dtype=float)
        excess = pos_xy - np.clip(pos_xy, -boundary, boundary)
        if np.any(excess != 0.0):
            vel_xy = np.asarray(self.data.qvel[self.t_dof_ids[:2]], dtype=float)
            leash_kp = 300.0
            leash_kd = 2.0 * np.sqrt(leash_kp * self.properties.t_mass_kg)
            beyond = excess != 0.0
            force_xy = np.where(
                beyond, force_xy - leash_kp * excess - leash_kd * vel_xy, force_xy
            )
        self.data.qfrc_applied[self.t_dof_ids[0]] = force_xy[0]
        self.data.qfrc_applied[self.t_dof_ids[1]] = force_xy[1]
        self.data.qfrc_applied[self.t_dof_ids[2]] = self._disturbance_torque

    @property
    def goal_move_active(self):
        return self._goal_move_enabled

    def _sample_goal_pose(self):
        xy = self._goal_move_rng.uniform(
            -self.goal_move_xy_half_m, self.goal_move_xy_half_m, size=2
        )
        theta = self._goal_move_rng.uniform(-np.pi, np.pi)
        return np.array([xy[0], xy[1], theta])

    def _schedule_next_goal_move(self):
        """+inf when disabled -- lets _maybe_move_goal skip a branch, since
        ``sim_time < inf`` is always true and it'll just never fire."""
        if not self._goal_move_enabled:
            self._next_goal_move_s = float("inf")
            return
        interval = self._goal_move_rng.uniform(
            self.goal_move_min_interval_s, self.goal_move_max_interval_s
        )
        self._next_goal_move_s = float(self.data.time) + interval

    def _maybe_move_goal(self):
        """Relocate the goal (and hence what ``success``/``coverage_fraction``
        require) at an interval drawn fresh each time from
        [goal_move_min_interval_s, goal_move_max_interval_s] -- not a fixed
        period, so there's no reliable countdown for an operator to time
        against. ``goal_move_skip_prob`` additionally makes it uncertain
        whether a given wakeup actually relocates the goal at all, on top of
        never knowing exactly when the next wakeup is. The only strategy
        that's robust to both is to stop optimizing for the current goal
        pose specifically and just close the gap as fast as possible,
        continuously, since "finish carefully but slowly" can be invalidated
        by a relocation at any moment.
        """
        if not self._goal_move_enabled or self.data.time < self._next_goal_move_s:
            return
        if self._goal_move_rng.random() >= self.goal_move_skip_prob:
            self._set_goal_pose(self._sample_goal_pose())
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

    def reset(self, *, t_pose=None, pusher_pos=None):
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        if t_pose is not None:
            self.data.qpos[self.t_free_joint_ids] = np.asarray(t_pose, dtype=float)
            # Re-seat at the cached settled height rather than 0 -- see the
            # settle loop in __init__.
            self.data.qpos[self._t_z_qpos_id] = self._t_rest_z
        if pusher_pos is not None:
            self.data.qpos[self.pusher_qpos_ids] = np.asarray(pusher_pos, dtype=float)
        mujoco.mj_forward(self.model.ptr, self.data.ptr)
        self._requested_target = self.pusher_pos.copy()
        self._drive_target = self._requested_target.copy()
        self._sensor_stage1[:] = 0.0
        self._sensor_stage2[:] = 0.0
        # Zero the OU process's state, not its RNG stream -- each episode
        # starts from "no disturbance yet" but keeps drawing new values, so
        # repeated resets don't replay the identical disturbance path.
        self._disturbance_force[:] = 0.0
        self._disturbance_torque = 0.0
        # Put the goal back where it started (not wherever --goal-move-*
        # last relocated it to) and draw a fresh first relocation interval,
        # continuing the RNG stream rather than reseeding it -- same
        # "reset the state, not the randomness" convention as the
        # disturbance process above.
        self._set_goal_pose(self._initial_goal_pose)
        self._schedule_next_goal_move()
        return self

    def _configure_pusher_contact(self):
        """Interpolate the pusher/T contact's solref/solimp by pusher_softness.

        See HARD_PUSHER_SOLREF/SOFT_PUSHER_SOLREF for what this trades off.
        """
        t = self.pusher_softness
        time_constant = (
            HARD_PUSHER_SOLREF[0] + t * (SOFT_PUSHER_SOLREF[0] - HARD_PUSHER_SOLREF[0])
        )
        damping_ratio = (
            HARD_PUSHER_SOLREF[1] + t * (SOFT_PUSHER_SOLREF[1] - HARD_PUSHER_SOLREF[1])
        )
        width = (
            HARD_PUSHER_SOLIMP_WIDTH
            + t * (SOFT_PUSHER_SOLIMP_WIDTH - HARD_PUSHER_SOLIMP_WIDTH)
        )
        d0 = HARD_PUSHER_SOLIMP_D0 + t * (SOFT_PUSHER_SOLIMP_D0 - HARD_PUSHER_SOLIMP_D0)
        self.model.geom_solref[self.pusher_geom_id] = (time_constant, damping_ratio)
        self.model.geom_solimp[self.pusher_geom_id, 0] = d0
        self.model.geom_solimp[self.pusher_geom_id, 2] = width

    def _pusher_overlaps_t(self):
        """True if the just-reset pose has the pusher already touching the T.

        Requires ``mj_forward`` to have run (``reset`` always calls it) so
        ``self.data.contact`` reflects the pose just written.
        """
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 == self.pusher_geom_id and g2 in self.t_geom_ids) or (
                g2 == self.pusher_geom_id and g1 in self.t_geom_ids
            ):
                return True
        return False

    def randomize_start(
        self,
        *,
        t_xy_half=0.15,
        t_theta_half=np.pi,
        pusher_xy_half=0.20,
        center_probability=0.70,
        force_center=False,
        max_resamples=64,
    ):
        """Sample a start layout from the same mixture flip-up/cube-lift use.

        With probability ``center_probability`` (default 0.70), sample a
        clipped ``N(0, 0.22)`` in each axis's normalized [-0.5, 0.5] range
        ("center_gaussian" -- most samples stay within roughly the middle
        half of the range); otherwise sample uniformly over the full range
        ("uniform"). ``force_center`` collapses to the exact center, used
        for the very first episode so an operator's first attempt isn't an
        unusually hard corner case. See
        flipup_teleop.sample_start_pose for the source of this convention.

        The T-block and pusher are sampled independently, so -- unlike the
        fixed default layout, which places them by construction on opposite
        sides -- a random draw can land the pusher on top of the T,
        especially since both distributions are center-biased toward the
        same region. Resample (up to ``max_resamples`` times) whenever the
        drawn pose starts in contact; if every draw is unlucky (astronomically
        unlikely at the defaults), fall back to placing the pusher just
        outside the T's bounding radius along the direction it was drawn in.
        """
        t_bound_radius = 0.5 * float(
            np.hypot(
                max(self.properties.t_bar_length_m, self.properties.t_stem_length_m),
                max(self.properties.t_bar_width_m, self.properties.t_stem_width_m),
            )
        )
        clearance = t_bound_radius + self.properties.pusher_radius_m + 0.01

        def draw():
            if force_center:
                normalized = np.zeros(5)
            elif self._rng.random() < center_probability:
                normalized = np.clip(self._rng.normal(0.0, 0.22, size=5), -0.5, 0.5)
            else:
                normalized = self._rng.uniform(-0.5, 0.5, size=5)
            half_extents = np.array(
                [t_xy_half, t_xy_half, t_theta_half, pusher_xy_half, pusher_xy_half]
            )
            scaled = normalized * 2.0 * half_extents
            return scaled[0:2], scaled[2], scaled[3:5]

        t_xy, t_theta, pusher_xy = draw()
        for _ in range(max(0, int(max_resamples))):
            self.reset(t_pose=(t_xy[0], t_xy[1], t_theta), pusher_pos=pusher_xy)
            if not self._pusher_overlaps_t():
                return self
            t_xy, t_theta, pusher_xy = draw()

        # Every draw overlapped -- push the last-drawn pusher position
        # radially outward from the T center until it clears the T's
        # bounding radius, instead of giving up with an overlapping start.
        offset = pusher_xy - t_xy
        distance = float(np.linalg.norm(offset))
        if distance < 1e-9:
            angle = self._rng.uniform(0.0, 2.0 * np.pi)
            offset = np.array([np.cos(angle), np.sin(angle)])
            distance = 1.0
        pusher_xy = t_xy + offset / distance * clearance
        self.reset(t_pose=(t_xy[0], t_xy[1], t_theta), pusher_pos=pusher_xy)
        return self

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
