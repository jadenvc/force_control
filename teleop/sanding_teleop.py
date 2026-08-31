"""Full-arm haptic sanding task.

A UR5e holds a compliant sander pad and must sand an entire target area on a
flat panel to a desired force band. Force is read exactly (MuJoCo solver
contact force between the sander pad and the panel), the same way the
full-arm FlipUp task reads its fingertip contact. Every point on the panel
accumulates a "sanding dose" that grows with applied force -- more force
reaches the just-right dose faster, matching how a real orbital sander
removes more material per second the harder you press. Too much
instantaneous force is a hard, sticky failure ("the sander breaks through
the panel"), not just a discouraged state.

See flipup_teleop.py / floating_flipup_teleop.py for the sibling tasks this
one borrows patterns from:
  - the whole-robot UR5e + Jacobian-transpose task-space controller
    (flipup_minimal/flipup/environment.py: FlipUpEnv.step_task_space);
  - the compliant-contact interpolation knob
    (floating_flipup_teleop.py: _configure_tip_contact / tip_softness);
  - the rasterized-grid coverage metric, generalized from a boolean mask to
    a continuous per-cell dose accumulator
    (push_t_teleop.py: _build_goal_grid / coverage_fraction).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from dm_control import mjcf
from dm_control.mujoco.engine import Physics
from scipy.spatial.transform import Rotation

_FLIPUP_DIR = Path(__file__).resolve().parent.parent / "flipup_minimal"
if str(_FLIPUP_DIR) not in sys.path:
    sys.path.insert(0, str(_FLIPUP_DIR))

from flipup.environment import ASSET_DIR, FlipUpEnv, _wxyz_from_matrix  # noqa: E402

# ------------------------------------------------------------------- defaults

# Same order of magnitude as the full-arm FlipUp task's tool_kp (16000 N/m):
# the arm itself stays stiff/accurate, compliance comes from the pad contact
# (pad_softness below), exactly the same division of labor FlipUp uses
# between its Cartesian controller and the fingertip tip-softness knob.
DEFAULT_TOOL_KP = 16000.0
DEFAULT_TOOL_ROT_KP = 3000.0
DEFAULT_TOOL_ROT_KD = 90.0
DEFAULT_JOINT_KD = np.array([64.0, 64.0, 64.0, 16.0, 16.0, 16.0])
DEFAULT_ARM_DAMPING = 2.5
DEFAULT_HAPTIC_STIFFNESS = 1500.0

# Ported from floating_flipup_teleop.py's SOFT_TIP_SOLREF/SOFT_TIP_SOLIMP_WIDTH,
# slightly softer since a sanding pad should feel more compliant by default
# than FlipUp's rigid-by-default fingertip pad.
COMPILED_PAD_SOLREF = (0.010, 1.0)
COMPILED_PAD_SOLIMP_WIDTH = 0.003
SOFT_PAD_SOLREF = (0.025, 2.0)
SOFT_PAD_SOLIMP_WIDTH = 0.006

PANEL_TRANSFORM = np.eye(4, dtype=np.float64)
PANEL_TRANSFORM[:3, 3] = (0.30, 0.0, 0.25)
# Must match sanding_panel.xml's panel geom half-thickness and sander.xml's
# pad ellipsoid half-height -- these aren't exposed as tunable properties
# since changing them means editing the compiled geometry anyway.
PANEL_HALF_THICKNESS_M = 0.01
PAD_HALF_THICKNESS_M = 0.01
# The tool_site (control point) height at which the pad's lowest point JUST
# touches the panel's top surface with zero penetration. CLI drivers use
# this as the reference height for both a safe hover offset and for
# interpreting the operator's up/down motion as penetration into the panel.
CONTACT_TOOL_Z = (
    PANEL_TRANSFORM[2, 3] + PANEL_HALF_THICKNESS_M + PAD_HALF_THICKNESS_M
)

# Force Dimension omega comfortable/safe workspace half-extents, in device
# metres -- the real device's usable range (not a guess), reused from the
# figures already calibrated for it in floating_cube_lift_teleop.py's
# default_device_wall_half. Used to size --scale defaults so the device's
# full comfortable range of motion maps onto the full panel, not just a
# patch near its center.
DEVICE_WORKSPACE_HALF_M = np.array([0.045, 0.040, 0.048])


@dataclass(frozen=True)
class SandingProperties:
    """Panel size, force band, dose-rate law, and grid resolution.

    All of these are exposed as CLI flags in teleop_sanding.py; the exact
    numbers here are a starting point, not a calibrated material model.
    """

    # Sized/positioned (with PANEL_TRANSFORM and the UR5e base attachment
    # point below) so every corner sits in a comfortable ~0.3-0.65m band from
    # the base -- inside the UR5e's ~0.85m reach with real margin, and not so
    # close to the base that corner poses fold up/lose manipulability. The
    # original 0.40x0.30 panel centered 0.85m from the base put its far
    # corner past the arm's actual reach.
    panel_length_m: float = 0.34  # x-extent (world, before any panel rotation)
    panel_width_m: float = 0.26  # y-extent
    pad_radius_m: float = 0.045  # must match sander.xml's pad geom size

    # force_min_n and dose_target_time_s are calibrated together so that
    # dose = k_dose * (force - force_min_n) * time gives roughly equivalent
    # results for "10N for 1s" and "5N for 10s" -- i.e. it really is a
    # force/time trade-off, not just a force threshold: less force still
    # sands the same amount, it just needs proportionally longer dwell.
    # Solving (10 - Fmin)*1 = (5 - Fmin)*10 gives Fmin = 40/9 ~= 4.44.
    force_min_n: float = 4.44  # below this, no material removal at all
    force_target_n: float = 12.0  # the desired/nominal sanding force
    force_cap_n: float = 20.0  # dose-rate saturates at/above this
    force_break_n: float = 30.0  # sustained force above this breaks the panel
    break_debounce_steps: int = 3
    # Break-detection reads a short low-pass of the contact force, not the raw
    # per-step value -- MuJoCo's solver produces genuine few-ms impact spikes
    # on first contact (well above steady-state force) even for an approach
    # that settles to a perfectly reasonable force; without this the break
    # check false-triggers on ordinary first touch, not actual overload. This
    # mirrors FLOATING_FLIPUP_COMPLIANCE_TELEOP.md's finding that raw contact
    # force needs a modeled-sensor-style filter before it's meaningful as a
    # sustained-load signal.
    break_force_tau_s: float = 0.015

    # Time at force_target_n to reach dose 1.0. Paired with force_min_n
    # above: 0.735s here makes 10N reach dose 1.0 in ~1s and 5N in ~10s,
    # both via the same underlying rate law -- see force_min_n's comment.
    dose_target_time_s: float = 0.735
    dose_low: float = 0.7  # below this: under-sanded
    dose_high: float = 1.3  # above this: over-sanded
    dose_max: float = 2.0  # accumulator clip ceiling

    grid_resolution_m: float = 0.01  # dose-accumulation grid cell size
    vis_cell_m: float = 0.02  # visual gradient grid cell size (coarser, ok)

    pad_softness: float = 0.6  # [0, 1], see _configure_pad_contact
    success_threshold: float = 0.90  # fraction of area in the just-right band

    # A handful of discrete SQUARE target regions, not the whole panel --
    # coverage_fraction/success only look at cells within these; the rest of
    # the panel can still physically be sanded (dose still accumulates there
    # too) but doesn't count. Centers are spaced along the panel's long axis
    # (see _sample_target_regions); the actual rendered/counted square size
    # is clamped by _region_half_size to guarantee a visible gap between
    # adjacent squares, so region_radius_m here is a maximum, not a
    # guarantee -- with 7-10 regions on a typical panel it usually ends up
    # smaller than this.
    num_regions: int = 7  # [5, 10]
    region_radius_m: float = 0.025  # square half-side-length, see above

    def __post_init__(self):
        if self.panel_length_m <= 0.0 or self.panel_width_m <= 0.0:
            raise ValueError("panel dimensions must be positive")
        if self.pad_radius_m <= 0.0:
            raise ValueError("pad_radius_m must be positive")
        if not (0.0 < self.force_min_n < self.force_target_n < self.force_cap_n < self.force_break_n):
            raise ValueError(
                "forces must satisfy 0 < force_min_n < force_target_n < "
                "force_cap_n < force_break_n"
            )
        if self.break_debounce_steps < 1:
            raise ValueError("break_debounce_steps must be >= 1")
        if self.break_force_tau_s < 0.0:
            raise ValueError("break_force_tau_s cannot be negative")
        if self.dose_target_time_s <= 0.0:
            raise ValueError("dose_target_time_s must be positive")
        if not (0.0 < self.dose_low < self.dose_high < self.dose_max):
            raise ValueError("dose thresholds must satisfy 0 < dose_low < dose_high < dose_max")
        if self.grid_resolution_m <= 0.0 or self.vis_cell_m <= 0.0:
            raise ValueError("grid resolutions must be positive")
        if not 0.0 <= self.pad_softness <= 1.0:
            raise ValueError("pad_softness must be in [0, 1]")
        if not 0.0 < self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in (0, 1]")
        if not 5 <= self.num_regions <= 10:
            raise ValueError("num_regions must be in [5, 10]")
        if self.region_radius_m <= 0.0:
            raise ValueError("region_radius_m must be positive")


DEFAULT_SANDING_PROPERTIES = SandingProperties()


def dose_to_rgba(dose, properties, is_target=None):
    """Piecewise-linear dose -> RGBA color ramp: highlight -> blue -> green -> red.

    Pure numpy, no external colormap dependency, matching this codebase's
    existing convention of hand-rolled color logic (see flipup_teleop.py's
    book-color-at-episode-setup pattern).

    ``is_target`` marks which cells belong to one of the num_regions patches
    that actually count toward coverage_fraction/success -- non-target
    cells always render as plain panel wood (`inert`), regardless of dose,
    so the handful of regions that matter stand out from the rest of the
    panel. A target cell starts as `highlight` (amber, "needs sanding"),
    turns blue-toward-green as its dose approaches dose_low, flat green
    across the just-right band, then green-toward-red past dose_high.
    """
    dose = np.clip(np.asarray(dose, dtype=float), 0.0, properties.dose_max)
    n = dose.shape[0]
    is_target = np.ones(n, dtype=bool) if is_target is None else np.asarray(is_target, dtype=bool)
    rgba = np.empty((n, 4), dtype=float)
    rgba[:, 3] = 1.0

    inert = np.array([0.70, 0.60, 0.45])  # plain panel wood, non-target cells
    highlight = np.array([0.95, 0.75, 0.10])  # target cell, not yet touched
    blue = np.array([0.15, 0.25, 0.90])  # target cell, under-sanded
    green = np.array([0.15, 0.85, 0.20])  # target cell, just right
    red = np.array([0.90, 0.15, 0.10])  # target cell, over-sanded

    rgba[:, :3] = inert

    untouched = is_target & (dose < 1e-6)
    in_progress = is_target & (dose >= 1e-6) & (dose < properties.dose_low)
    just_right = is_target & (dose >= properties.dose_low) & (dose <= properties.dose_high)
    over = is_target & (dose > properties.dose_high)

    rgba[untouched, :3] = highlight

    t_under = np.clip(dose / max(properties.dose_low, 1e-9), 0.0, 1.0)
    under_color = (1.0 - t_under[:, None]) * blue + t_under[:, None] * green
    rgba[in_progress, :3] = under_color[in_progress]

    rgba[just_right, :3] = green

    span = max(properties.dose_max - properties.dose_high, 1e-9)
    t_over = np.clip((dose - properties.dose_high) / span, 0.0, 1.0)
    over_color = (1.0 - t_over[:, None]) * green + t_over[:, None] * red
    rgba[over, :3] = over_color[over]

    return rgba


class SandingEnv(FlipUpEnv):
    """UR5e + compliant sander pad sanding a flat panel to a target force.

    Deliberately does NOT call ``FlipUpEnv.__init__`` -- that constructor
    hardcodes WSG50 finger joints and book-specific geom lookups that don't
    exist here. This follows the exact precedent
    ``floating_flipup_teleop.FloatingFlipUpTeleop`` set: reuse ``FlipUpEnv``'s
    static helpers and its Jacobian-transpose control *pattern*, but build the
    scene and constructor state from scratch.
    """

    # Chosen so the pad starts already hovering ~5cm above the panel (about
    # 3mm settled tracking error at tool_kp=16000), not at some arbitrary
    # distant configuration. The previous placeholder home config put the
    # tool_site ~0.65m from the panel, meaning the sander was invisible in
    # any panel-framed camera view for the ~13s (at the 0.05 m/s default
    # --max-speed) it took to travel into frame after startup.
    _HOME_JOINTS = np.array(
        [-1.873, -1.577, 2.136, -2.13, -1.571, -1.873], dtype=np.float64
    )

    def __init__(
        self,
        seed=0,
        properties=None,
        tool_kp=DEFAULT_TOOL_KP,
        tool_rot_kp=DEFAULT_TOOL_ROT_KP,
        tool_rot_kd=DEFAULT_TOOL_ROT_KD,
        arm_damping=DEFAULT_ARM_DAMPING,
        show_viewer=False,
    ):
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.properties = properties or DEFAULT_SANDING_PROPERTIES
        if not isinstance(self.properties, SandingProperties):
            raise TypeError("properties must be a SandingProperties instance")
        # Fixed for v1 -- not yet exposed as a CLI knob (see README's open
        # questions); _build_dose_grid uses this same module-level constant so
        # the grid's world coordinates always match the compiled panel pose.
        self.panel_transform = PANEL_TRANSFORM

        self.tool_kp = float(tool_kp)
        self.tool_rot_kp = float(tool_rot_kp)
        self.tool_rot_kd = float(tool_rot_kd)

        self.physics = self._build_physics(self.properties)
        self.model = self.physics.model
        self.data = self.physics.data

        joint_names = tuple(f"ur5e/{name}_joint" for name in self._AXIS_NAMES)
        actuator_names = tuple(f"ur5e/{name}" for name in self._AXIS_NAMES)
        self.joint_qpos_ids = np.array(
            [int(np.asarray(self.model.joint(n).qposadr).item()) for n in joint_names],
            dtype=np.int32,
        )
        self.joint_dof_ids = np.array(
            [int(np.asarray(self.model.joint(n).dofadr).item()) for n in joint_names],
            dtype=np.int32,
        )
        self.actuator_ids = np.array(
            [self.model.actuator(n).id for n in actuator_names], dtype=np.int32
        )
        self.tool_site_id = self.model.site("ur5e/sander/planner_tip_site").id
        self.pad_geom_id = self.model.geom("ur5e/sander/sander_pad_collision").id
        self.panel_surface_geom_id = self.model.geom("sanding_panel/panel_surface").id
        self.panel_body_id = self.model.body("sanding_panel/panel").id

        self.task_space_kp = np.diag(
            [self.tool_kp] * 3 + [self.tool_rot_kp] * 3
        ).astype(np.float64)
        self.task_space_kd = DEFAULT_JOINT_KD * float(arm_damping)
        self.task_space_cartesian_kd = np.array(
            [0.0, 0.0, 0.0, self.tool_rot_kd, self.tool_rot_kd, self.tool_rot_kd],
            dtype=np.float64,
        )
        self.jacobian = np.zeros((6, self.model.nv), dtype=np.float64)
        self.twist = np.zeros(6, dtype=np.float64)
        self.site_quaternion = np.zeros(4, dtype=np.float64)
        self.site_quaternion_conjugate = np.zeros(4, dtype=np.float64)
        self.error_quaternion = np.zeros(4, dtype=np.float64)

        self._configure_pad_contact()
        self._build_dose_grid()

        self._contact_buf = np.zeros(6, dtype=float)
        self._break_streak = 0
        self._break_force_filtered = 0.0
        self._episode_max_force_n = 0.0

        self._initial_qpos = self.data.qpos.copy()
        self.viewer = None
        self.reset()

        if show_viewer:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(model=self.model.ptr, data=self.data.ptr)

    # ------------------------------------------------------------- building
    @classmethod
    def _build_physics(cls, properties: SandingProperties) -> Physics:
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
        sander_model = mjcf.from_path(str(ASSET_DIR / "sander" / "sander.xml"))
        robot_collision_geoms += cls._collision_geoms(sander_model)
        attachment_site.attach(sander_model)

        robot_site = world_model.worldbody.add(
            "site", name="robot_attachment_site", pos=(-0.15, 0.0, 0.05)
        )
        robot_site.attach(robot_model)

        panel_model = mjcf.from_path(
            str(ASSET_DIR / "custom" / "sanding_panel" / "sanding_panel.xml")
        )
        panel_surface_geom = panel_model.find("geom", "panel_surface")
        if panel_surface_geom is None:
            raise RuntimeError("Sanding panel surface geom is missing")
        cls._add_visual_grid(panel_model, properties)
        cls._attach_model(
            world_model, panel_model, PANEL_TRANSFORM, freejoint=False
        )

        # Only two directed bits are needed here (no book/bookend/table split
        # to disambiguate): every robot collision geom may contact the panel
        # surface, so nothing physically clips through it, but the dose grid
        # and haptic force in SandingEnv only ever look at the specific
        # sander-pad-vs-panel-surface contact, exactly like push_t only counts
        # pusher-vs-T contacts despite the pusher also touching the table.
        for geom in world_model.find_all("geom"):
            geom.contype = 0
            geom.conaffinity = 0
        for geom in robot_collision_geoms:
            geom.conaffinity = 1
        panel_surface_geom.contype = 1

        return mjcf.Physics.from_mjcf_model(world_model)

    @staticmethod
    def _add_visual_grid(panel_model, properties: SandingProperties):
        """Add a grid of small, non-colliding, individually-colored geoms
        tiling the panel for the live sanded-amount gradient (dose_to_rgba
        drives their geom_rgba every step). Reuses the geom_matid=-1 +
        geom_rgba trick already used once-per-episode at
        flipup_teleop.py:572-573, just applied per-cell and updated live."""
        cell = float(properties.vis_cell_m)
        half_x = properties.panel_length_m / 2.0
        half_y = properties.panel_width_m / 2.0
        xs = np.arange(-half_x + cell / 2.0, half_x, cell)
        ys = np.arange(-half_y + cell / 2.0, half_y, cell)
        panel_body = panel_model.find("body", "panel")
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                panel_body.add(
                    "geom",
                    name=f"vis_cell_{i}_{j}",
                    type="box",
                    size=(cell * 0.48, cell * 0.48, 0.0015),
                    pos=(float(x), float(y), 0.012),
                    contype=0,
                    conaffinity=0,
                    group=2,
                    rgba=(0.55, 0.50, 0.40, 1.0),
                )

    def _configure_pad_contact(self):
        """Interpolate the sander pad's solref/solimp between the compiled
        (rigid) values and a soft endpoint, by pad_softness in [0, 1].

        Ported directly from floating_flipup_teleop.py's
        _configure_tip_contact -- same interpolation, different geom.
        """
        s = float(self.properties.pad_softness)
        time_constant = (
            COMPILED_PAD_SOLREF[0] + s * (SOFT_PAD_SOLREF[0] - COMPILED_PAD_SOLREF[0])
        )
        damping_ratio = (
            COMPILED_PAD_SOLREF[1] + s * (SOFT_PAD_SOLREF[1] - COMPILED_PAD_SOLREF[1])
        )
        width = (
            COMPILED_PAD_SOLIMP_WIDTH
            + s * (SOFT_PAD_SOLIMP_WIDTH - COMPILED_PAD_SOLIMP_WIDTH)
        )
        self.model.geom_solref[self.pad_geom_id] = (time_constant, damping_ratio)
        self.model.geom_solimp[self.pad_geom_id, 2] = width

    # ------------------------------------------------------------- dose grid
    def _sample_target_regions(self):
        """num_regions square region centers (panel-local xy), spaced out
        along the panel's long axis (evenly spaced, with a small random
        jitter so it doesn't look robotically exact), deterministic from
        self._rng (seeded in __init__, so a given seed always gets the same
        layout). Positioned along a line for a legible, deliberate layout,
        but each square is a separate, gapped patch -- not one continuous
        sanded stripe -- see _region_half_size for the gap-guaranteeing size."""
        p = self.properties
        half_x = p.panel_length_m / 2.0
        margin = p.region_radius_m * 1.2
        xs = np.linspace(-half_x + margin, half_x - margin, p.num_regions)
        spacing = (xs[1] - xs[0]) if p.num_regions > 1 else p.panel_length_m
        jitter = min(p.region_radius_m * 0.4, spacing * 0.1)
        offsets = self._rng.uniform(-jitter, jitter, size=(p.num_regions, 2))
        centers = np.stack([xs, np.zeros_like(xs)], axis=1) + offsets
        return centers

    def _region_half_size(self):
        """Half-side-length of each square target region. Clamped to well
        under half the spacing between adjacent region centers so squares
        stay visibly separate, discrete patches -- --region-radius alone
        (e.g. sized for a sparse layout) could otherwise make adjacent
        squares touch or overlap into one continuous sanded strip once
        packed along the line in _sample_target_regions."""
        p = self.properties
        if p.num_regions > 1:
            half_x = p.panel_length_m / 2.0
            margin = p.region_radius_m * 1.2
            spacing = (2.0 * (half_x - margin)) / (p.num_regions - 1)
            return min(p.region_radius_m, spacing * 0.35)
        return p.region_radius_m

    def _build_dose_grid(self):
        p = self.properties
        cell = float(p.grid_resolution_m)
        half_x = p.panel_length_m / 2.0
        half_y = p.panel_width_m / 2.0
        xs = np.arange(-half_x + cell / 2.0, half_x, cell)
        ys = np.arange(-half_y + cell / 2.0, half_y, cell)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        local_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

        panel_rotation = self.panel_transform[:3, :3]
        panel_origin = self.panel_transform[:3, 3]
        world_xy = (local_xy @ panel_rotation[:2, :2].T) + panel_origin[:2]

        self._region_centers_local = self._sample_target_regions()
        self._region_half_size_m = self._region_half_size()
        # Chebyshev/L-inf distance -- membership test for a SQUARE region,
        # not a circle: max(|dx|, |dy|) <= half_size.
        chebyshev_to_regions = np.max(
            np.abs(local_xy[:, None, :] - self._region_centers_local[None, :, :]), axis=-1
        )
        # Only these num_regions square patches count toward
        # coverage_fraction/success -- the rest of the panel can still be
        # physically sanded (dose still accumulates there in
        # _update_dose_and_break), it just isn't part of the task.
        self._target_mask = np.any(chebyshev_to_regions <= self._region_half_size_m, axis=1)

        self._grid_xy_world = world_xy
        self._dose = np.zeros(world_xy.shape[0], dtype=np.float64)

        # Visual grid geom ids, same iteration order as _add_visual_grid, and
        # a nearest-dose-cell index per visual cell so a coarser visual grid
        # can be driven from the finer dose-accumulation grid.
        vis_cell = float(p.vis_cell_m)
        vxs = np.arange(-half_x + vis_cell / 2.0, half_x, vis_cell)
        vys = np.arange(-half_y + vis_cell / 2.0, half_y, vis_cell)
        vis_geom_ids = []
        vis_local_xy = []
        for i in range(len(vxs)):
            for j in range(len(vys)):
                name = f"sanding_panel/vis_cell_{i}_{j}"
                vis_geom_ids.append(self.model.geom(name).id)
                vis_local_xy.append((vxs[i], vys[j]))
        self._vis_geom_ids = np.array(vis_geom_ids, dtype=np.int32)
        vis_local_xy = np.asarray(vis_local_xy, dtype=float)
        # Nearest dose-grid cell for each visual cell (vectorized).
        diff = vis_local_xy[:, None, :] - local_xy[None, :, :]
        self._vis_to_dose_index = np.argmin(np.sum(diff * diff, axis=-1), axis=-1)
        vis_chebyshev_to_regions = np.max(
            np.abs(vis_local_xy[:, None, :] - self._region_centers_local[None, :, :]), axis=-1
        )
        self._vis_is_target = np.any(vis_chebyshev_to_regions <= self._region_half_size_m, axis=1)

    def refresh_visual_gradient(self):
        colors = dose_to_rgba(
            self._dose[self._vis_to_dose_index], self.properties, self._vis_is_target
        )
        self.model.geom_rgba[self._vis_geom_ids] = colors

    # ---------------------------------------------------------------- contact
    def pad_contact_force(self):
        """Signed (Fx, Fy, Fz) reaction on the sander pad from the panel,
        world frame, and the world-frame centroid of the contact point(s).
        Same iteration pattern as push_t_teleop.py's pusher_contact_force_xy,
        generalized to 3D and returning the contact point for the dose grid.

        mj_contactForce's normal component is the force acting on geom2 (not
        geom1) along contact.frame's normal axis, which itself points from
        geom1 toward geom2. Push_t's pusher happened to always be geom1 in
        its own contacts, so its "-force if pusher_is_2" convention silently
        matched; here the pad is geom1 against the panel (geom2), so that
        same formula was inverted -- pressing down measured a NEGATIVE (further
        into the panel) reaction instead of positive (pushing the pad away),
        which fed a backwards-feeling haptic reflection. Verified empirically:
        pressing straight down must read a positive world Fz.
        """
        total = np.zeros(3, dtype=float)
        points = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            pad_is_1 = geom1 == self.pad_geom_id
            pad_is_2 = geom2 == self.pad_geom_id
            surf_is_1 = geom1 == self.panel_surface_geom_id
            surf_is_2 = geom2 == self.panel_surface_geom_id
            if not ((pad_is_1 and surf_is_2) or (pad_is_2 and surf_is_1)):
                continue
            mujoco.mj_contactForce(self.model.ptr, self.data.ptr, index, self._contact_buf)
            contact_to_world = np.asarray(contact.frame, dtype=float).reshape(3, 3).T
            force = contact_to_world @ self._contact_buf[:3]
            total += force if pad_is_2 else -force
            points.append(np.asarray(contact.pos, dtype=float))
        centroid = np.mean(points, axis=0) if points else None
        return total, centroid

    def normal_force_n(self):
        force, _ = self.pad_contact_force()
        # Panel normal is world +z (flat, unrotated panel); the pad only ever
        # pushes down into it, so the normal component's magnitude is what
        # both the dose-rate law and the break check use.
        return float(abs(force[2]))

    # ------------------------------------------------------------------ step
    def step_task_space(self, target_pose):
        """Same Jacobian-transpose task-space controller as
        FlipUpEnv.step_task_space, minus the WSG50-specific gripper-actuator
        line (this end effector has no finger actuator)."""
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (7,):
            raise ValueError(f"target_pose must have shape (7,), got {target_pose.shape}")

        target_position = target_pose[:3]
        target_quaternion = target_pose[3:]
        site_data = self.data.site(self.tool_site_id)

        self.twist[:3] = target_position - site_data.xpos
        mujoco.mju_mat2Quat(self.site_quaternion, site_data.xmat)
        mujoco.mju_negQuat(self.site_quaternion_conjugate, self.site_quaternion)
        mujoco.mju_mulQuat(
            self.error_quaternion, target_quaternion, self.site_quaternion_conjugate
        )
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quaternion, 1.0)

        mujoco.mj_jacSite(
            self.model.ptr, self.data.ptr, self.jacobian[:3], self.jacobian[3:], self.tool_site_id
        )

        tool_velocity = self.jacobian @ self.data.qvel
        task_wrench = self.task_space_kp @ self.twist - self.task_space_cartesian_kd * tool_velocity
        generalized_force = self.jacobian.T @ task_wrench
        generalized_force[self.joint_dof_ids] -= (
            self.task_space_kd * self.data.qvel[self.joint_dof_ids]
        )
        generalized_force += self.data.qfrc_bias

        actuator_force = generalized_force[self.joint_dof_ids]
        force_ranges = np.asarray(self.model.actuator_forcerange)[self.actuator_ids]
        actuator_force = np.clip(actuator_force, force_ranges[:, 0], force_ranges[:, 1])
        self.data.ctrl[self.actuator_ids] = actuator_force

        mujoco.mj_step(self.model.ptr, self.data.ptr)

        self._update_dose_and_break()

        if self.viewer is not None:
            if not self.viewer.is_running():
                return False
            self.viewer.sync()
        return True

    def target_pose7(self, target_pos, target_rotvec=None):
        """xyz + wxyz pose for a commanded pad position.

        Orientation defaults to pad-facing-straight-down (pi rotation about
        world x) unless target_rotvec supplies one, matching flipup_teleop.py's
        target_pose7 shape/naming so the recorder can call it the same way.
        """
        if target_rotvec is None:
            rot = Rotation.from_euler("xyz", (np.pi, 0.0, 0.0)).as_matrix()
        else:
            rot = Rotation.from_rotvec(np.asarray(target_rotvec, dtype=float)).as_matrix()
        return np.concatenate([np.asarray(target_pos, dtype=float), _wxyz_from_matrix(rot)])

    def step(self, target_pos, target_rotvec=None, n_substeps=1):
        for _ in range(max(1, int(n_substeps))):
            target_pose = self.target_pose7(target_pos, target_rotvec)
            if not self.step_task_space(target_pose):
                return False
        return True

    def _update_dose_and_break(self):
        p = self.properties
        dt = self.timestep
        force = self.normal_force_n()
        self._episode_max_force_n = max(self._episode_max_force_n, force)

        if force < p.force_min_n:
            rate = 0.0
        else:
            f_eff = min(force, p.force_cap_n)
            dose_at_target = 1.0
            k_dose = dose_at_target / (p.dose_target_time_s * (p.force_target_n - p.force_min_n))
            rate = k_dose * (f_eff - p.force_min_n)

        if rate > 0.0:
            _, centroid = self.pad_contact_force()
            if centroid is not None:
                contact_xy_world = centroid[:2]
                covered = (
                    np.linalg.norm(self._grid_xy_world - contact_xy_world, axis=1)
                    <= p.pad_radius_m
                )
                self._dose[covered] = np.clip(
                    self._dose[covered] + dt * rate, 0.0, p.dose_max
                )

        tau = p.break_force_tau_s
        alpha = 1.0 if tau <= 0.0 else dt / (tau + dt)
        self._break_force_filtered += alpha * (force - self._break_force_filtered)

        if self._break_force_filtered > p.force_break_n:
            self._break_streak += 1
        else:
            self._break_streak = 0
        if self._break_streak >= p.break_debounce_steps:
            self._broken = True

    # ---------------------------------------------------------------- task
    def coverage_fraction(self, band="just_right"):
        p = self.properties
        target = self._dose[self._target_mask]
        if target.size == 0:
            return 0.0
        if band == "under":
            in_band = target < p.dose_low
        elif band == "just_right":
            in_band = (target >= p.dose_low) & (target <= p.dose_high)
        elif band == "over":
            in_band = target > p.dose_high
        else:
            raise ValueError(f"unknown band {band!r}")
        return float(in_band.sum()) / float(target.size)

    @property
    def broken(self):
        return bool(getattr(self, "_broken", False))

    def success(self, threshold=None):
        threshold = self.properties.success_threshold if threshold is None else threshold
        return (not self.broken) and self.coverage_fraction("just_right") >= threshold

    def task_metric_value(self):
        return self.coverage_fraction("just_right")

    # --------------------------------------------------------------- lifecycle
    def reset(self):
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        self.data.qpos[self.joint_qpos_ids] = self._HOME_JOINTS
        self.physics.forward()
        self._dose[:] = 0.0
        self._broken = False
        self._break_streak = 0
        self._break_force_filtered = 0.0
        self._episode_max_force_n = 0.0
        self.refresh_visual_gradient()
        if self.viewer is not None:
            self.viewer.sync()

    @property
    def episode_max_force_n(self):
        return float(self._episode_max_force_n)

    def get_tool_pose(self):
        site_data = self.data.site(self.tool_site_id)
        return np.concatenate(
            [site_data.xpos.copy(), _wxyz_from_matrix(site_data.xmat.copy().reshape(3, 3))]
        ).astype(np.float32)

    @property
    def tool_pos(self):
        return np.asarray(self.data.site(self.tool_site_id).xpos, dtype=float).copy()


class SandingTeleop(SandingEnv):
    """SandingEnv plus haptic/camera defaults for the CLI driver, mirroring
    how FlipUpTeleop sits on top of FlipUpEnv."""

    task_kind = "sanding"
    default_tool_kp = DEFAULT_TOOL_KP
    default_haptic_stiffness = DEFAULT_HAPTIC_STIFFNESS
    # A side-on view: azimuth=90 puts the camera off to the side (along y)
    # rather than above or in front, elevation is only a slight downward tilt
    # (mostly horizontal) so the arm's whole profile and the panel are both
    # visible side-by-side, not one occluding the other from overhead.
    default_cam_azimuth = 90.0
    default_cam_elevation = -15.0
    default_cam_distance = 0.75  # closer than the original 0.95 while still framing the
    # whole arm base-to-pad and the panel together (0.65 starts clipping the base)
    default_cam_lookat = (0.15, 0.0, 0.25)  # between the base and the panel
    default_cam_name = None
