"""Full-arm haptic peg-in-hole insertion task.

A UR5e holds a rigid peg and must find and insert it into a square socket
fixture. Unlike sanding (a compliant tool sliding over a flat surface),
insertion is a *contact-rich, force-sensitive* task: the peg must make and
maintain gentle contact with the fixture's top surface, search laterally
for the opening without excessive force, and then descend into the socket
under a small residual push -- exactly the "approach -> contact -> search ->
insert" structure used by the reference peg-insertion sim at
/home/yifan/git/force-insertion-sim (a Franka FR3 + SimCore sim; SimCore
itself is not installed here, so everything below is a from-scratch numpy/
MuJoCo reimplementation of that sim's *documented* algorithms/parameters,
not a binding to it -- see README_insertion.md's "What's a faithful port vs.
an approximation" section for the full list of where this deliberately
differs).

Structurally this file mirrors sanding_teleop.py (same UR5e + Jacobian-
transpose task-space controller pattern from
flipup_minimal/flipup/environment.py: FlipUpEnv.step_task_space, same
dataclass-of-tunables + Env + Teleop class layout, same
`ASSET_DIR / "custom" / ...` scene-composition pattern). Two controller
pieces are new relative to sanding/flipup, both ported from
force-insertion-sim per this task's brief:

  1. Non-zero *translational* Cartesian damping in the task-space PD law
     (sanding/flipup's ``task_space_cartesian_kd`` is zero on translation --
     see flipup_minimal/flipup/environment.py:110-119's comment -- damping
     there comes only from joint-velocity damping). Ported here as a
     per-axis ratio taken from
     /home/yifan/git/force-insertion-sim/configs/control/panda_arm.yaml's
     ``dynamic_impedance`` block (K_cart=[450,450,700,80,80,200],
     D_cart=[55,55,55,10,10,20]), applied to *this* controller's own
     (much stiffer, UR5e-scale) Kp rather than reusing the Franka's raw
     numbers -- see ``_cartesian_kd_from_ratio`` below.
  2. A critically-damped 2nd-order "dynamic filter" (``DynamicFilter``)
     smoothing the commanded feed-forward wrench before it enters the
     control law, ported from
     /home/yifan/git/force-insertion-sim/src/policy/dynamic_filter.py.

A filtered F/T sensor reading (``DynamicFilter``'s sibling,
``EMAFilter``/``ButterworthFilter``, ported from
/home/yifan/git/force-insertion-sim/src/utils/sensor_callback.py) is also
exposed for reporting/recording, alongside the existing exact-solver-force
helper the way sanding/flipup already do.

See teleop/insertion_scripted_demo.py for the min-jerk trajectory + phase
state machine (APPROACH/CONTACT/SEARCH/INSERT) that drives this env for
open-loop scripted demonstrations, and teleop/teleop_insertion.py for the
haptic teleoperation CLI.
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

# Deliberately MUCH softer than sanding's tool_kp (16000 N/m). Sanding puts
# essentially all of its contact compliance into the PAD geom's solref/solimp
# and keeps the ARM itself rigid/accurate (a flat pad sliding over a flat
# panel never needs the arm itself to yield). A rigid peg searching for and
# entering a rigid socket is different: any contact against the fixture's
# top surface needs the CONTROLLER to be soft enough that ordinary position
# tracking error (from sensing/approach uncertainty, or from the wiggle
# probing motion below) doesn't itself generate large forces. This mirrors
# why the reference sim's controller is an impedance controller with
# K_cart=[450,450,700,...] N/m -- two orders of magnitude softer than a
# typical rigid position controller -- rather than a stiff PD position
# controller with a compliant end effector. Empirically (see
# README_insertion.md's tuning log), reusing sanding's 16000 N/m here and
# then commanding even a few mm of position error against the rigid fixture
# produced 100-250N force spikes and, combined with non-zero Cartesian
# damping (see _CARTESIAN_KD_RATIO below), actuator saturation and loss of
# tracking; dropping tool_kp by ~6x is what made gentle, bounded-force
# contact achievable with this same Jacobian-transpose PD control law.
DEFAULT_TOOL_KP = 2500.0
DEFAULT_TOOL_ROT_KP = 400.0
DEFAULT_JOINT_KD = np.array([64.0, 64.0, 64.0, 16.0, 16.0, 16.0])
DEFAULT_ARM_DAMPING = 2.5
DEFAULT_HAPTIC_STIFFNESS = 3000.0

# ---- Cartesian (task-space) damping ratio, ported from force-insertion-sim
#
# /home/yifan/git/force-insertion-sim/configs/control/panda_arm.yaml:14-16
# (the `dynamic_impedance` block actually used by insertion_episode.py:58):
#   K_cart = [450, 450, 700,  80,  80, 200]   # N/m (xyz), N*m/rad (rxryrz)
#   D_cart = [ 55,  55,  55,  10,  10,  20]
#
# The absolute numbers are for a Franka impedance controller running at a
# totally different Kp scale (450-700 N/m) than this UR5e task-space
# controller (8000-16000 N/m, ~15-25x stiffer -- see DEFAULT_TOOL_KP above).
# Reusing D_cart's raw numbers here would be a negligible damping ratio
# relative to our much stiffer Kp. What *is* portable is the ratio D/K per
# axis, which sets a characteristic response time (D/K, in seconds, for a
# simple K-D system) independent of the absolute stiffness scale; applying
# that same ratio to our own Kp diagonal gives a Cartesian Kd with an
# analogous damping character at our stiffness. This is the "ratio/shape,
# not literal Franka numbers" porting called for in the task brief.
_CARTESIAN_KD_RATIO = np.array(
    [55.0 / 450.0, 55.0 / 450.0, 55.0 / 700.0, 10.0 / 80.0, 10.0 / 80.0, 20.0 / 200.0],
    dtype=np.float64,
)
# Global multiplier on top of the ratio above, exposed as a tunable because
# the ratio's absolute scale still needed empirical adjustment against this
# controller's actual (undamped, discretized, contact-coupled) dynamics --
# see insertion_scripted_demo.py's tuning iterations in README_insertion.md.
#
# This was originally tuned against a stiff tool_kp=16000 N/m (sanding's
# value), where the raw ratio (scale=1.0) saturated the UR5e's wrist
# actuators (+-28 N*m, far smaller than the shoulder/elbow's +-150 N*m --
# J^T maps Cartesian damping force into torque at every joint including the
# wrist) and left a persistent limit-cycle position error. Once tool_kp was
# lowered to 2500 N/m instead (see DEFAULT_TOOL_KP's comment -- the bigger
# and more physically-motivated fix), the raw ratio at scale=1.0 settles
# cleanly with no saturation, so the scale is kept at 1.0; this knob is left
# in place (rather than removed) as a documented escape hatch in case a
# future tool_kp/tool_rot_kp change reintroduces the saturation problem.
DEFAULT_CARTESIAN_DAMPING_SCALE = 1.0

# ---- DynamicFilter defaults, ported verbatim from
# /home/yifan/git/force-insertion-sim/src/policy/dynamic_filter.py:5
# (alpha=0.9, beta=0.3 there). Kept as the starting point; the *effective*
# filter dynamics differ from the reference regardless, because the
# reference's `step(F_df, dt)` is called at their 200 Hz control rate
# (dt=0.005s, see configs/scene_config.yaml:3's control_rate:200) while this
# controller filters every physics step at this repo's 1 kHz timestep
# (dt=0.001s, ground.xml's <option timestep="0.001">) -- a genuinely faster
# sample rate, not just a relabeled one. Re-tuned empirically for that reason
# (see README_insertion.md).
DEFAULT_DYNAMIC_FILTER_ALPHA = 0.9
DEFAULT_DYNAMIC_FILTER_BETA = 0.3

# ---- F/T sensor EMA filter default, ported from
# /home/yifan/git/force-insertion-sim/src/utils/sensor_callback.py:61
# (`filter_alpha=0.2` constructor default for SensorCallback's EMAFilter).
DEFAULT_FT_FILTER_ALPHA = 0.2

# Peg contact softness interpolation endpoints (ported convention from
# sanding_teleop.py's COMPILED_PAD_SOLREF/SOFT_PAD_SOLREF, re-derived from
# insertion_peg.xml's own compiled defaults rather than reusing sanding's
# numbers -- see _configure_peg_contact).
COMPILED_PEG_SOLREF = (0.010, 1.2)
COMPILED_PEG_SOLIMP_WIDTH = 0.002
SOFT_PEG_SOLREF = (0.020, 1.8)
SOFT_PEG_SOLIMP_WIDTH = 0.004

HOLE_TRANSFORM = np.eye(4, dtype=np.float64)
# z chosen empirically so InsertionEnv._HOME_JOINTS (reused from sanding's
# home config, since it happens to also produce a comfortable, non-singular
# UR5e pose) leaves the peg tip hovering ~5cm above the hole entrance rather
# than already overlapping the fixture -- sanding's home joints were tuned
# for the sander tool's very different planner_tip_site offset, so this
# tool's actual reach needed its own empirical check (see
# insertion_scripted_demo.py / the env sanity check in tests/test_insertion.py).
HOLE_TRANSFORM[:3, 3] = (0.30, 0.0, 0.21)
# Depth of the socket tunnel below the entrance plane, and how far above the
# entrance the fixture's "picture frame" lip extends -- must match
# insertion_hole.xml's socket geometry (see that file's long comment); no
# shared source of truth, kept in sync by hand like sanding_panel.xml's
# half-extents.
SOCKET_DEPTH_M = 0.05
SOCKET_LIP_HEIGHT_M = 0.015

DEVICE_WORKSPACE_HALF_M = np.array([0.045, 0.040, 0.048])


# --------------------------------------------------------------------------
# Ported filters
# --------------------------------------------------------------------------


class DynamicFilter:
    """Critically-damped 2nd-order shaping filter for a commanded wrench.

    Ported verbatim from
    /home/yifan/git/force-insertion-sim/src/policy/dynamic_filter.py:1-18::

        class DynamicFilter:
            def __init__(self, alpha=0.9, beta=0.3, dt=0.005):
                self.alpha = alpha
                self.beta  = beta
                self.dt    = dt
                self.reset()

            def reset(self):
                self.F_ff     = np.zeros(6)
                self.F_ff_dot = np.zeros(6)

            def step(self, F_df, dt):
                F_ff_ddot  = self.alpha * (self.beta * (F_df - self.F_ff) - self.F_ff_dot)
                self.F_ff_dot += F_ff_ddot * dt
                self.F_ff     += self.F_ff_dot * dt
                return self.F_ff.copy()

    Semantics unchanged: ``F_df`` is the raw/desired feed-forward wrench for
    this step, ``F_ff`` is the actually-applied (smoothed) wrench, and the
    smoothing dynamics are those of a mass-spring-damper being driven toward
    F_df by a "spring" of rate ``beta`` and critically-damped by ``alpha`` --
    a single scalar pair applied identically to all 6 wrench components (not
    per-axis), integrated with explicit forward Euler exactly as the
    reference does (velocity update then position update, both using the
    ``dt`` passed to ``step``, not the constructor's stored ``dt``).
    """

    def __init__(
        self,
        alpha: float = DEFAULT_DYNAMIC_FILTER_ALPHA,
        beta: float = DEFAULT_DYNAMIC_FILTER_BETA,
        dt: float = 0.005,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        self.F_ff = np.zeros(6, dtype=np.float64)
        self.F_ff_dot = np.zeros(6, dtype=np.float64)

    def step(self, F_df: np.ndarray, dt: float) -> np.ndarray:
        F_df = np.asarray(F_df, dtype=np.float64)
        F_ff_ddot = self.alpha * (self.beta * (F_df - self.F_ff) - self.F_ff_dot)
        self.F_ff_dot = self.F_ff_dot + F_ff_ddot * dt
        self.F_ff = self.F_ff + self.F_ff_dot * dt
        return self.F_ff.copy()


class EMAFilter:
    """Exponential moving average filter, ported from
    /home/yifan/git/force-insertion-sim/src/utils/sensor_callback.py:6-19::

        class EMAFilter:
            def __init__(self, alpha):
                self.alpha = alpha
                self._state = None
            def reset(self):
                self._state = None
            def __call__(self, x):
                if self._state is None:
                    self._state = x.copy()
                else:
                    self._state = self.alpha * x + (1 - self.alpha) * self._state
                return self._state.copy()
    """

    def __init__(self, alpha: float = DEFAULT_FT_FILTER_ALPHA) -> None:
        self.alpha = float(alpha)
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._state is None:
            self._state = x.copy()
        else:
            self._state = self.alpha * x + (1.0 - self.alpha) * self._state
        return self._state.copy()


class ButterworthFilter:
    """Low-pass Butterworth filter, streaming sample-by-sample.

    Same algorithm as
    /home/yifan/git/force-insertion-sim/src/utils/sensor_callback.py:22-56
    (design a `scipy.signal.butter(..., output="sos")` cascade, then filter
    one sample at a time carrying state forward), except this uses scipy's
    own stateful ``sosfilt(..., zi=...)`` per call instead of the reference's
    hand-rolled per-sample direct-form-II-transposed biquad stepper -- scipy
    is already a dependency of this codebase (see the ``Rotation`` import
    above), so there is no need to reimplement `sosfilt` by hand the way the
    reference apparently did (presumably for a non-Python/real-time
    deployment target this repo doesn't have). Numerically equivalent
    streaming behavior, simpler implementation.
    """

    def __init__(self, cutoff_hz: float, fs_hz: float, order: int = 2) -> None:
        from scipy.signal import butter

        self.sos = butter(order, cutoff_hz, btype="low", fs=fs_hz, output="sos")
        self._zi = None

    def reset(self) -> None:
        self._zi = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfilt, sosfilt_zi

        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        if self._zi is None:
            zi_single = sosfilt_zi(self.sos)
            self._zi = np.stack([zi_single * xi for xi in x], axis=-1)
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            y, self._zi[:, :, i] = sosfilt(self.sos, x[i : i + 1], zi=self._zi[:, :, i])
            out[i] = y[0]
        return out


class FTSensorFilter:
    """Filtered force/torque reporting, generalizing
    force-insertion-sim's ``SensorCallback``
    (/home/yifan/git/force-insertion-sim/src/utils/sensor_callback.py:58-126)
    to whichever of EMA/Butterworth/none this task is configured with.

    Deliberately narrower than the reference: no gravity-compensation term
    is ported (see README_insertion.md's gap list) because this env already
    exposes an exact, zero-in-free-space contact-force ground truth
    (``InsertionEnv.peg_contact_force``) as an alternative to the raw
    simulated F/T sensor -- gravity compensation on the *sensor* reading
    matters most when that ground truth isn't available (i.e. on real
    hardware), which isn't the situation here.
    """

    def __init__(self, filter_type: str = "ema", alpha: float = DEFAULT_FT_FILTER_ALPHA,
                 cutoff_hz: float = 20.0, fs_hz: float = 1000.0, order: int = 2) -> None:
        self.filter_type = filter_type
        if filter_type == "ema":
            self._force_filter = EMAFilter(alpha)
            self._torque_filter = EMAFilter(alpha)
        elif filter_type == "butterworth":
            self._force_filter = ButterworthFilter(cutoff_hz, fs_hz, order)
            self._torque_filter = ButterworthFilter(cutoff_hz, fs_hz, order)
        elif filter_type == "none":
            self._force_filter = None
            self._torque_filter = None
        else:
            raise ValueError(f"unknown filter_type {filter_type!r}")

    def reset(self) -> None:
        if self._force_filter is not None:
            self._force_filter.reset()
            self._torque_filter.reset()

    def __call__(self, wrench: np.ndarray) -> np.ndarray:
        wrench = np.asarray(wrench, dtype=np.float64)
        if self._force_filter is None:
            return wrench.copy()
        force = self._force_filter(wrench[:3])
        torque = self._torque_filter(wrench[3:])
        return np.concatenate([force, torque])


# --------------------------------------------------------------------------
# Task properties
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InsertionProperties:
    """Insertion task tunables: contact softness, force limits, controller
    damping/filter knobs, and success/failure thresholds.

    Mirrors sanding_teleop.py's SandingProperties in spirit (a frozen,
    validated dataclass of everything a CLI driver or scripted-demo script
    might want to sweep); hole/peg *geometry* itself lives in the compiled
    XML (insertion_peg.xml / insertion_hole.xml) rather than here, the same
    division sanding uses (panel_length_m/panel_width_m are the exception
    there because the dose grid needs them at Python level; insertion has no
    analogous grid).
    """

    insert_depth_target_m: float = 0.03  # peg-tip depth below hole_entrance counted as "inserted"
    success_hold_steps: int = 200  # steps the depth criterion must hold before declaring success

    # Force limits. force_contact_threshold_n is the "did we touch the
    # fixture yet" detector (ported in spirit from
    # /home/yifan/git/force-insertion-sim/src/task/insertion_episode.py's
    # contact.force_threshold=2.0N, contact.n_confirm=75 steps @200Hz =
    # 0.375s -- re-expressed here as a TIME duration rather than a step
    # count, contact_confirm_time_s, so it's portable across control rates).
    force_contact_threshold_n: float = 1.5
    contact_confirm_time_s: float = 0.375
    # force_break_n: sustained force above this is a hard task failure (peg
    # jammed/crashed), same "filtered, debounced" pattern as sanding's
    # force_break_n/break_force_tau_s.
    # 45N (not, say, 40N) specifically because insertion_scripted_demo.py's
    # spiral search deliberately drags the peg against the rigid fixture
    # frame at up to ~search_spiral_max_radius_m * tool_kp N (see that
    # file's search_spiral_max_radius_m comment) as part of NORMAL, intended
    # search behavior, not a fault condition -- 40N left too little headroom
    # between that expected peak and the break ceiling and produced sporadic
    # false-positive breaks during tuning (see README_insertion.md's log).
    force_break_n: float = 45.0
    break_force_tau_s: float = 0.015
    break_debounce_steps: int = 3

    # Feed-forward force magnitudes for the scripted demo's CONTACT/SEARCH/
    # INSERT phases (see insertion_scripted_demo.py). Ported in spirit from
    # /home/yifan/git/force-insertion-sim/configs/task_config.yaml's
    # contact.f_push=4.0 and search/insert wiggle az=3.0/5.0 -- these are
    # already force units (Newtons), and this task's peg/fixture contact
    # stiffness is the same order of magnitude as the reference's peg/hole
    # contact (both use MuJoCo solref/solimp on rigid-body primitives), so
    # unlike the Cartesian-damping ratio above there's no unit-scale gap to
    # bridge; still empirically re-tuned (see README_insertion.md).
    push_force_n: float = 4.0
    search_wiggle_amplitude_n: float = 2.0
    search_wiggle_freq_hz: float = 1.2
    insert_wiggle_amplitude_n: float = 1.2
    insert_wiggle_freq_hz: float = 1.8
    wiggle_ramp_steps: int = 50  # ported from insertion_episode.py's ramp_steps=50

    # Cartesian damping / dynamic-filter knobs, see module docstring.
    cartesian_damping_scale: float = DEFAULT_CARTESIAN_DAMPING_SCALE
    dynamic_filter_alpha: float = DEFAULT_DYNAMIC_FILTER_ALPHA
    dynamic_filter_beta: float = DEFAULT_DYNAMIC_FILTER_BETA
    ft_filter_type: str = "ema"
    ft_filter_alpha: float = DEFAULT_FT_FILTER_ALPHA

    # Peg contact softness, see COMPILED_PEG_SOLREF/SOFT_PEG_SOLREF and
    # _configure_peg_contact.
    peg_softness: float = 0.5  # [0, 1]
    friction: tuple = (0.3, 0.005, 0.0001)

    def __post_init__(self):
        if self.insert_depth_target_m <= 0.0:
            raise ValueError("insert_depth_target_m must be positive")
        if self.insert_depth_target_m >= SOCKET_DEPTH_M:
            raise ValueError("insert_depth_target_m must be less than the socket depth")
        if self.success_hold_steps < 1:
            raise ValueError("success_hold_steps must be >= 1")
        if self.force_contact_threshold_n <= 0.0:
            raise ValueError("force_contact_threshold_n must be positive")
        if self.contact_confirm_time_s <= 0.0:
            raise ValueError("contact_confirm_time_s must be positive")
        if not (0.0 < self.force_contact_threshold_n < self.force_break_n):
            raise ValueError("force_contact_threshold_n must be less than force_break_n")
        if self.break_force_tau_s < 0.0:
            raise ValueError("break_force_tau_s cannot be negative")
        if self.break_debounce_steps < 1:
            raise ValueError("break_debounce_steps must be >= 1")
        if self.push_force_n <= 0.0:
            raise ValueError("push_force_n must be positive")
        if self.search_wiggle_amplitude_n < 0.0 or self.insert_wiggle_amplitude_n < 0.0:
            raise ValueError("wiggle amplitudes cannot be negative")
        if self.search_wiggle_freq_hz <= 0.0 or self.insert_wiggle_freq_hz <= 0.0:
            raise ValueError("wiggle frequencies must be positive")
        if self.wiggle_ramp_steps < 1:
            raise ValueError("wiggle_ramp_steps must be >= 1")
        if self.cartesian_damping_scale < 0.0:
            raise ValueError("cartesian_damping_scale cannot be negative")
        if not 0.0 < self.dynamic_filter_alpha:
            raise ValueError("dynamic_filter_alpha must be positive")
        if not 0.0 < self.dynamic_filter_beta:
            raise ValueError("dynamic_filter_beta must be positive")
        if self.ft_filter_type not in ("ema", "butterworth", "none"):
            raise ValueError("ft_filter_type must be one of 'ema', 'butterworth', 'none'")
        if not 0.0 < self.ft_filter_alpha <= 1.0:
            raise ValueError("ft_filter_alpha must be in (0, 1]")
        if not 0.0 <= self.peg_softness <= 1.0:
            raise ValueError("peg_softness must be in [0, 1]")
        if len(self.friction) != 3 or any(v < 0.0 for v in self.friction):
            raise ValueError("friction must have exactly 3 non-negative values")


DEFAULT_INSERTION_PROPERTIES = InsertionProperties()


class InsertionEnv(FlipUpEnv):
    """UR5e + rigid peg inserting into a square socket fixture.

    Deliberately does NOT call ``FlipUpEnv.__init__`` for the same reason
    ``SandingEnv``/``FloatingFlipUpTeleop`` don't -- see sanding_teleop.py's
    class docstring; this class follows that exact precedent.
    """

    # Chosen (like sanding's _HOME_JOINTS) so the peg tip starts hovering a
    # few centimeters above the hole fixture, not at some distant
    # configuration -- see insertion_scripted_demo.py's APPROACH phase for
    # how the scripted demo bridges the remaining distance.
    _HOME_JOINTS = np.array(
        [-1.873, -1.577, 2.136, -2.13, -1.571, -1.873], dtype=np.float64
    )

    def __init__(
        self,
        seed=0,
        properties=None,
        tool_kp=DEFAULT_TOOL_KP,
        tool_rot_kp=DEFAULT_TOOL_ROT_KP,
        arm_damping=DEFAULT_ARM_DAMPING,
        show_viewer=False,
    ):
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.properties = properties or DEFAULT_INSERTION_PROPERTIES
        if not isinstance(self.properties, InsertionProperties):
            raise TypeError("properties must be an InsertionProperties instance")
        self.hole_transform = HOLE_TRANSFORM

        self.tool_kp = float(tool_kp)
        self.tool_rot_kp = float(tool_rot_kp)

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
        self.tool_site_id = self.model.site("ur5e/insertion_peg/planner_tip_site").id
        self.wrist_force_adr = int(self.model.sensor("ur5e/insertion_peg/wrist_force_sensor").adr[0])
        self.wrist_torque_adr = int(self.model.sensor("ur5e/insertion_peg/wrist_torque_sensor").adr[0])
        self.wrist_site_id = self.model.site("ur5e/insertion_peg/ft_sensor_site").id
        self.peg_geom_id = self.model.geom("ur5e/insertion_peg/peg_shaft_collision").id
        self.hole_geom_ids = np.array(
            [
                self.model.geom(f"insertion_hole/socket_wall_{name}").id
                for name in ("px", "nx", "py", "ny")
            ]
            + [self.model.geom("insertion_hole/socket_floor").id],
            dtype=np.int32,
        )
        self.hole_entrance_site_id = self.model.site("insertion_hole/hole_entrance").id
        self.hole_bottom_site_id = self.model.site("insertion_hole/hole_bottom").id

        self.task_space_kp = np.diag(
            [self.tool_kp] * 3 + [self.tool_rot_kp] * 3
        ).astype(np.float64)
        self.task_space_kd = DEFAULT_JOINT_KD * float(arm_damping)
        self._recompute_cartesian_damping()

        self.jacobian = np.zeros((6, self.model.nv), dtype=np.float64)
        self.twist = np.zeros(6, dtype=np.float64)
        self.site_quaternion = np.zeros(4, dtype=np.float64)
        self.site_quaternion_conjugate = np.zeros(4, dtype=np.float64)
        self.error_quaternion = np.zeros(4, dtype=np.float64)

        self._configure_peg_contact()

        self._dynamic_filter = DynamicFilter(
            alpha=self.properties.dynamic_filter_alpha,
            beta=self.properties.dynamic_filter_beta,
            dt=self.timestep,
        )
        self._ft_filter = FTSensorFilter(
            filter_type=self.properties.ft_filter_type,
            alpha=self.properties.ft_filter_alpha,
            fs_hz=1.0 / self.timestep,
        )

        self._contact_buf = np.zeros(6, dtype=float)
        self._break_streak = 0
        self._break_force_filtered = 0.0
        self._episode_max_force_n = 0.0
        self._depth_hold_steps = 0

        self._initial_qpos = self.data.qpos.copy()
        self.viewer = None
        self.reset()

        if show_viewer:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(model=self.model.ptr, data=self.data.ptr)

    def _recompute_cartesian_damping(self) -> None:
        """(Re)build task_space_cartesian_kd from _CARTESIAN_KD_RATIO applied
        to this controller's own Kp diagonal -- see module docstring / the
        _CARTESIAN_KD_RATIO comment for the porting rationale. This is the
        "extend the existing task-space PD law with non-zero Cartesian
        translational damping" piece the task brief calls for; sanding/flipup
        both leave translational Cartesian damping at exactly zero (see
        flipup_minimal/flipup/environment.py:110-119)."""
        kp_diag = np.array(
            [self.tool_kp] * 3 + [self.tool_rot_kp] * 3, dtype=np.float64
        )
        scale = float(self.properties.cartesian_damping_scale)
        self.task_space_cartesian_kd = kp_diag * _CARTESIAN_KD_RATIO * scale

    # ------------------------------------------------------------- building
    @classmethod
    def _build_physics(cls, properties: InsertionProperties) -> Physics:
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
        peg_model = mjcf.from_path(str(ASSET_DIR / "insertion_peg" / "insertion_peg.xml"))
        robot_collision_geoms += cls._collision_geoms(peg_model)
        attachment_site.attach(peg_model)

        robot_site = world_model.worldbody.add(
            "site", name="robot_attachment_site", pos=(-0.15, 0.0, 0.05)
        )
        robot_site.attach(robot_model)

        hole_model = mjcf.from_path(
            str(ASSET_DIR / "custom" / "insertion_hole" / "insertion_hole.xml")
        )
        hole_collision_geoms = cls._collision_geoms(hole_model)
        if not hole_collision_geoms:
            raise RuntimeError("insertion_hole socket collision geoms are missing")
        cls._attach_model(world_model, hole_model, HOLE_TRANSFORM, freejoint=False)

        # Same contype/conaffinity convention as sanding_teleop.py: nothing
        # collides by default, then the robot side (including the peg) gets
        # conaffinity=1 and the fixture's collision geoms get contype=1, so
        # the peg can contact the fixture but the fixture's non-colliding
        # visual geoms stay inert.
        for geom in world_model.find_all("geom"):
            geom.contype = 0
            geom.conaffinity = 0
        for geom in robot_collision_geoms:
            geom.conaffinity = 1
        for geom in hole_collision_geoms:
            geom.contype = 1

        return mjcf.Physics.from_mjcf_model(world_model)

    def _configure_peg_contact(self):
        """Interpolate the peg's solref/solimp between the compiled (rigid)
        values and a soft endpoint, by peg_softness in [0, 1].

        Ported pattern from sanding_teleop.py's _configure_pad_contact /
        floating_flipup_teleop.py's _configure_tip_contact. Unlike sanding
        (where softness had to be written onto the higher-priority PANEL
        geom because the pad's own priority lost), here the PEG geom is the
        higher-priority side (insertion_peg.xml's peg_collision class,
        priority=1 vs. the fixture's default priority=0) so writing only to
        the peg geom is sufficient -- still also written to the fixture's
        wall/floor geoms for documentation/robustness if priority is ever
        changed, same defensive habit sanding's fix established.
        """
        s = float(self.properties.peg_softness)
        time_constant = COMPILED_PEG_SOLREF[0] + s * (SOFT_PEG_SOLREF[0] - COMPILED_PEG_SOLREF[0])
        damping_ratio = COMPILED_PEG_SOLREF[1] + s * (SOFT_PEG_SOLREF[1] - COMPILED_PEG_SOLREF[1])
        width = COMPILED_PEG_SOLIMP_WIDTH + s * (SOFT_PEG_SOLIMP_WIDTH - COMPILED_PEG_SOLIMP_WIDTH)

        self.model.geom_solref[self.peg_geom_id] = (time_constant, damping_ratio)
        self.model.geom_solimp[self.peg_geom_id, 2] = width
        self.model.geom_friction[self.peg_geom_id] = self.properties.friction
        for geom_id in self.hole_geom_ids:
            self.model.geom_solref[geom_id] = (time_constant, damping_ratio)
            self.model.geom_solimp[geom_id, 2] = width
            self.model.geom_friction[geom_id] = self.properties.friction

    # ---------------------------------------------------------------- contact
    def peg_contact_force(self):
        """Signed (Fx, Fy, Fz) reaction on the peg from the fixture (any of
        the 4 walls or the floor), world frame, plus the world-frame
        centroid of the contact point(s) -- exact MuJoCo solver force, zero
        in free space by construction. Same pattern as sanding_teleop.py's
        pad_contact_force, generalized from a single panel_surface geom to
        the set of fixture geoms in self.hole_geom_ids.

        Sign convention verified the same way flipup_teleop.py's
        contact_force docstring describes: mj_contactForce's normal
        component acts on geom2 along contact.frame's normal axis (which
        points geom1 -> geom2); the peg's actual geom1/geom2 role vs. the
        fixture is not fixed (MuJoCo may order either geom first depending
        on ids), so both cases are handled explicitly below and the sign is
        flipped so this always reports the force the WORLD/fixture applies
        TO the peg (i.e. pressing the peg down into the fixture should read
        a positive world +z reaction pushing back up).
        """
        total = np.zeros(3, dtype=float)
        points = []
        hole_set = set(int(g) for g in self.hole_geom_ids)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            peg_is_1 = geom1 == self.peg_geom_id
            peg_is_2 = geom2 == self.peg_geom_id
            hole_is_1 = geom1 in hole_set
            hole_is_2 = geom2 in hole_set
            if not ((peg_is_1 and hole_is_2) or (peg_is_2 and hole_is_1)):
                continue
            mujoco.mj_contactForce(self.model.ptr, self.data.ptr, index, self._contact_buf)
            contact_to_world = np.asarray(contact.frame, dtype=float).reshape(3, 3).T
            force = contact_to_world @ self._contact_buf[:3]
            # Force acts on geom2's body; report force applied TO the peg.
            total += force if peg_is_2 else -force
            points.append(np.asarray(contact.pos, dtype=float))
        centroid = np.mean(points, axis=0) if points else None
        return total, centroid

    def normal_force_n(self):
        force, _ = self.peg_contact_force()
        return float(np.linalg.norm(force))

    def wrist_wrench_raw(self):
        """Raw (unfiltered) simulated wrist F/T sensor reading, tool frame."""
        force = np.array(self.data.sensordata[self.wrist_force_adr : self.wrist_force_adr + 3])
        torque = np.array(self.data.sensordata[self.wrist_torque_adr : self.wrist_torque_adr + 3])
        return np.concatenate([force, torque])

    def wrist_wrench_filtered(self):
        """EMA/Butterworth-filtered wrist F/T reading (see FTSensorFilter),
        the ported equivalent of
        /home/yifan/git/force-insertion-sim/src/utils/sensor_callback.py's
        SensorCallback output. This is the value insertion's teleop/recorder
        should prefer for haptic rendering and dataset logging; the raw
        wrist_wrench_raw() and exact peg_contact_force() remain available
        for anything that specifically wants unfiltered/ground-truth force
        (this is the "add a filtered variant" piece of the task brief --
        sanding/flipup's contact_force()/wrist_wrench() report raw,
        unfiltered force only, see flipup_teleop.py:904-1097).
        """
        return self._ft_filter(self.wrist_wrench_raw())

    def peg_tip_depth_m(self):
        """How far the peg tip is below the hole_entrance plane, in the
        fixture's local +z-down sense (positive means inserted; negative
        means still above the fixture). Uses the compiled hole_entrance
        site's world position/orientation so this stays correct even if
        HOLE_TRANSFORM changes."""
        tip_world = np.asarray(self.data.site(self.tool_site_id).xpos, dtype=float)
        entrance_world = np.asarray(self.data.site(self.hole_entrance_site_id).xpos, dtype=float)
        entrance_mat = np.asarray(
            self.data.site(self.hole_entrance_site_id).xmat, dtype=float
        ).reshape(3, 3)
        local = entrance_mat.T @ (tip_world - entrance_world)
        return float(-local[2])

    # ------------------------------------------------------------------ step
    def step_task_space(self, target_pose, feed_forward_wrench=None):
        """Jacobian-transpose task-space PD control, same structure as
        FlipUpEnv.step_task_space / SandingEnv.step_task_space
        (flipup_minimal/flipup/environment.py:412-470,
        teleop/sanding_teleop.py:675-736), extended with two pieces ported
        from force-insertion-sim per the task brief:

          - task_space_cartesian_kd now has non-zero TRANSLATIONAL entries
            (see _recompute_cartesian_damping), not just rotational like
            sanding's tool_rot_kd.
          - an optional ``feed_forward_wrench`` (raw/desired, 6,) is passed
            through this env's DynamicFilter instance every step and the
            smoothed result is added directly to task_wrench BEFORE the
            Jacobian-transpose conversion to generalized/actuator force --
            i.e. exactly the "commanded task-space force/wrench" the task
            brief calls out, per
            /home/yifan/git/force-insertion-sim/src/policy/dynamic_filter.py.
            When None, this behaves identically to a plain impedance
            controller (as if F_df were held at zero), still routed through
            the filter so its internal state stays consistent across calls.
        """
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (7,):
            raise ValueError(f"target_pose must have shape (7,), got {target_pose.shape}")
        F_df = np.zeros(6) if feed_forward_wrench is None else np.asarray(
            feed_forward_wrench, dtype=np.float64
        )
        if F_df.shape != (6,):
            raise ValueError(f"feed_forward_wrench must have shape (6,), got {F_df.shape}")

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
        F_ff = self._dynamic_filter.step(F_df, self.timestep)
        task_wrench = (
            self.task_space_kp @ self.twist
            - self.task_space_cartesian_kd * tool_velocity
            + F_ff
        )
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

        self._update_break_state()

        if self.viewer is not None:
            if not self.viewer.is_running():
                return False
            self.viewer.sync()
        return True

    def target_pose7(self, target_pos, target_rotvec=None):
        """xyz + wxyz pose for a commanded peg position.

        Orientation defaults to peg-pointing-straight-down (pi rotation
        about world x), same convention as sanding_teleop.py's
        target_pose7."""
        if target_rotvec is None:
            rot = Rotation.from_euler("xyz", (np.pi, 0.0, 0.0)).as_matrix()
        else:
            rot = Rotation.from_rotvec(np.asarray(target_rotvec, dtype=float)).as_matrix()
        return np.concatenate([np.asarray(target_pos, dtype=float), _wxyz_from_matrix(rot)])

    def step(self, target_pos, target_rotvec=None, feed_forward_wrench=None, n_substeps=1):
        for _ in range(max(1, int(n_substeps))):
            target_pose = self.target_pose7(target_pos, target_rotvec)
            if not self.step_task_space(target_pose, feed_forward_wrench=feed_forward_wrench):
                return False
        return True

    def _update_break_state(self):
        p = self.properties
        dt = self.timestep
        force = self.normal_force_n()
        self._episode_max_force_n = max(self._episode_max_force_n, force)

        tau = p.break_force_tau_s
        alpha = 1.0 if tau <= 0.0 else dt / (tau + dt)
        self._break_force_filtered += alpha * (force - self._break_force_filtered)

        if self._break_force_filtered > p.force_break_n:
            self._break_streak += 1
        else:
            self._break_streak = 0
        if self._break_streak >= p.break_debounce_steps:
            self._broken = True

        if self.peg_tip_depth_m() >= p.insert_depth_target_m:
            self._depth_hold_steps += 1
        else:
            self._depth_hold_steps = 0

    # ---------------------------------------------------------------- task
    @property
    def broken(self):
        return bool(getattr(self, "_broken", False))

    def success(self):
        return (not self.broken) and self._depth_hold_steps >= self.properties.success_hold_steps

    def task_metric_value(self):
        return self.peg_tip_depth_m()

    # --------------------------------------------------------------- lifecycle
    def reset(self):
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        self.data.qpos[self.joint_qpos_ids] = self._HOME_JOINTS
        self.physics.forward()
        self._broken = False
        self._break_streak = 0
        self._break_force_filtered = 0.0
        self._episode_max_force_n = 0.0
        self._depth_hold_steps = 0
        self._dynamic_filter.reset()
        self._ft_filter.reset()
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

    @property
    def hole_entrance_pos(self):
        return np.asarray(self.data.site(self.hole_entrance_site_id).xpos, dtype=float).copy()


class InsertionTeleop(InsertionEnv):
    """InsertionEnv plus haptic/camera defaults for the CLI driver, mirroring
    SandingTeleop sitting on top of SandingEnv."""

    task_kind = "insertion"
    default_tool_kp = DEFAULT_TOOL_KP
    default_haptic_stiffness = DEFAULT_HAPTIC_STIFFNESS
    default_cam_azimuth = 90.0
    default_cam_elevation = -35.0
    default_cam_distance = 0.55
    default_cam_lookat = HOLE_TRANSFORM[:3, 3].tolist()
    default_cam_name = None
