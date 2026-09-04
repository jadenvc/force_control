"""Open-loop scripted peg-in-hole insertion demo.

Drives InsertionEnv (insertion_teleop.py) through a min-jerk Cartesian
trajectory planner and an APPROACH -> CONTACT -> SEARCH -> INSERT -> DONE
phase state machine, ported (with the simplifications noted below) from the
reference peg-insertion sim:

  - min-jerk trajectory polynomial: ported verbatim from
    /home/yifan/git/force-insertion-sim/src/task/trajectory.py:82-87
    (``TrajectoryPlanner._minjerk``); this env's tool orientation is always
    "peg pointing straight down" and never changes, so the SLERP-based
    orientation trajectory in the reference (trajectory.py's ``step()``,
    lines ~36-56) is intentionally NOT ported -- there is nothing to
    interpolate.
  - phase state machine structure and thresholds: ported IN SPIRIT (not
    verbatim -- units, control rate, and controller stiffness all differ,
    see insertion_teleop.py's module docstring) from
    /home/yifan/git/force-insertion-sim/src/task/insertion_episode.py.
    The reference's more elaborate INSERT sub-state-machine
    (STUCK/UNSTUCK/ALIGNED hysteresis, insertion_episode.py:102-239) is NOT
    ported -- this demo uses a single, simpler INSERT behavior (a deep
    min-jerk re-target once the hole is found, plus continued wiggle for
    lateral micro-adjustment). See README_insertion.md's gap list.
  - sinusoidal "wiggle" feed-forward with a ramp_steps=50 linear blend on
    phase transition: ported from insertion_episode.py:150-156 (search) and
    :115-156 (the ramp -- see WiggleGenerator/PhaseController below).

Run directly for a single tuned demo with a force-profile plot:

    python teleop/insertion_scripted_demo.py --plot teleop/insertion_force_profile.png

Or import ``run_scripted_demo`` for use from a test or tuning script.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from insertion_teleop import InsertionEnv, InsertionProperties  # noqa: E402


# --------------------------------------------------------------------------
# Min-jerk trajectory (position only -- see module docstring)
# --------------------------------------------------------------------------


def _minjerk(tau: float):
    """s(tau), ds/dtau, d2s/dtau2 for the quintic min-jerk profile.

    Ported verbatim from
    /home/yifan/git/force-insertion-sim/src/task/trajectory.py:82-87::

        @staticmethod
        def _minjerk(tau):
            tau = np.clip(tau, 0.0, 1.0)
            s = 10*tau**3 - 15*tau**4 + 6*tau**5
            ds = 30*tau**2 - 60*tau**3 + 30*tau**4
            dds = 60*tau - 180*tau**2 + 120*tau**3
            return s, ds, dds
    """
    tau = float(np.clip(tau, 0.0, 1.0))
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


class MinJerkSegment:
    """Position-only min-jerk trajectory between two 3D points.

    Same ``plan_with_speed``-style duration derivation as
    /home/yifan/git/force-insertion-sim/src/task/trajectory.py:27-30
    (``duration = max(dist / max_speed, min_duration)``).
    """

    def __init__(self, p_start, p_end, max_speed: float, min_duration: float = 0.2):
        self.p_start = np.asarray(p_start, dtype=np.float64)
        self.p_end = np.asarray(p_end, dtype=np.float64)
        dist = float(np.linalg.norm(self.p_end - self.p_start))
        self.duration = max(dist / max(max_speed, 1e-6), min_duration)
        self.t = 0.0

    @property
    def done(self) -> bool:
        return self.t >= self.duration

    def step(self, dt: float):
        s, _, _ = _minjerk(self.t / self.duration)
        pos = self.p_start + s * (self.p_end - self.p_start)
        self.t = min(self.t + dt, self.duration)
        return pos


# --------------------------------------------------------------------------
# Phase state machine
# --------------------------------------------------------------------------


class Phase:
    APPROACH = "approach"
    CONTACT = "contact"
    SEARCH = "search"
    INSERT = "insert"
    DONE = "done"
    FAILED = "failed"


@dataclass
class ScriptedDemoConfig:
    """Tunables for the scripted demo's phase state machine, separate from
    InsertionProperties (which lives on the env/controller side). Defaults
    below are the result of the tuning pass described in
    README_insertion.md; see that file for the iteration log.
    """

    # ---- geometry helpers (must match insertion_hole.xml/insertion_peg.xml)
    peg_radius_m: float = 0.010
    socket_lip_height_m: float = 0.015  # insertion_teleop.SOCKET_LIP_HEIGHT_M
    socket_depth_m: float = 0.05  # insertion_teleop.SOCKET_DEPTH_M

    # ---- APPROACH
    hover_clearance_m: float = 0.05
    approach_gap_m: float = 0.01  # how far above contact_control_z APPROACH ends
    transit_speed_mps: float = 0.25
    descent_speed_mps: float = 0.04
    # Random XY landing perturbation (ported in spirit from
    # insertion_episode.py's episode.approach.pertubation.pos_std=[2mm]*3 --
    # see extraction notes; this is what makes SEARCH necessary instead of
    # every episode dropping straight in).
    landing_perturbation_std_m: float = 0.0035

    # ---- CONTACT / SEARCH shared virtual-target penetration.
    # Chosen (see README_insertion.md) so the peg resting against the
    # fixture's top surface (NOT yet through the hole) settles to a gentle,
    # bounded force rather than the reference's raw 20mm-below-touch target
    # (which would be ~50N+ at this controller's tool_kp -- see
    # insertion_teleop.py's DEFAULT_TOOL_KP comment for the scaling story).
    search_penetration_m: float = 0.006
    push_force_n: float = 4.0  # ported in spirit from task_config.yaml's contact.f_push=4.0
    contact_settle_speed_mps: float = 0.02

    # ---- CONTACT phase confirmation (ported in spirit from
    # insertion_episode.py's contact.force_threshold=2.0N / n_confirm=75
    # steps @200Hz=0.375s -- re-expressed as a TIME duration, see
    # InsertionProperties.contact_confirm_time_s).
    contact_timeout_s: float = 4.0
    # If the peg lands close enough to center to fall straight through
    # without ever registering frame contact, treat it as "hole already
    # found" instead of timing out waiting for a contact event that will
    # never happen. Expressed as meters PAST the touch height (contact_control_z
    # = entrance_z + socket_lip_height_m + peg_radius_m), NOT as an absolute
    # peg_tip_depth_m() value -- peg_tip_depth_m() is zero at the
    # hole_entrance REFERENCE PLANE, which sits socket_lip_height_m +
    # peg_radius_m (0.025m by default) ABOVE the touch height because of the
    # raised lip, so "resting on the frame at zero penetration" already
    # reads as peg_tip_depth_m() == -0.025, not 0. Getting this frame mixed
    # up was the first scripted-demo bug found during tuning (see
    # README_insertion.md): a fell_through/hole_found threshold expressed as
    # a small POSITIVE peg_tip_depth_m() is only reachable 25mm+ past the
    # touch height and never fires for a peg merely settling near the touch
    # line.
    fell_through_past_touch_m: float = 0.004

    # ---- SEARCH
    #
    # A pure force wiggle (as the reference sim uses -- see the module
    # docstring's wiggle citation) turned out to be far too weak an
    # exploration signal at this controller's tool_kp: a 2N force wiggle
    # against tool_kp=2500 N/m only produces ~0.8mm of actual motion, well
    # short of what's needed to reliably re-cover a landing offset with
    # landing_perturbation_std_m=3.5mm (measured empirically: only 6/10
    # seeds found the hole before search_timeout_s with force-wiggle-only
    # search, see README_insertion.md's tuning log). This is the SAME
    # "reference's absolute numbers don't transfer at this controller's much
    # higher stiffness" lesson as DEFAULT_TOOL_KP/DEFAULT_CARTESIAN_DAMPING_SCALE
    # in insertion_teleop.py, just showing up in the trajectory/search layer
    # instead of the controller layer.
    #
    # The fix is a POSITION-based expanding spiral search (a standard,
    # well-known peg-in-hole search strategy, distinct from -- and layered
    # on top of -- the force-only wiggle) instead of relying on force wiggle
    # alone: the xy target traces a spiral of growing radius around the
    # landed point while the z target stays fixed (same small penetration
    # used for CONTACT), reliably sweeping through the actual clearance
    # annulus. The residual sinusoidal FORCE wiggle (search_wiggle_amplitude_n)
    # is kept, layered on top of the spiral, mainly to help the peg jiggle
    # loose of static friction once the spiral brings it near the opening
    # (per-axis force still ported from insertion_episode.py's search wiggle
    # concept), not as the primary search mechanism anymore.
    search_wiggle_amplitude_n: float = 1.5
    search_wiggle_freq_hz: float = 3.0
    # Capped at 10mm (not, say, 16mm) specifically to bound the worst-case
    # lateral contact force the spiral itself can generate while dragging
    # the peg along the fixture's rigid frame: worst case force is roughly
    # ``tool_kp * search_spiral_max_radius_m`` (2500 N/m * 0.010m = 25N),
    # comfortably under InsertionProperties.force_break_n=40N. An earlier
    # 16mm radius put that same estimate right at the break threshold and
    # measurably tripped it (2/15 seeds broke during tuning, see
    # README_insertion.md) -- 10mm keeps real margin while still being
    # several times landing_perturbation_std_m=3.5mm, so it reliably
    # recovers a mislanded approach.
    search_spiral_max_radius_m: float = 0.013
    search_spiral_growth_time_s: float = 6.0
    # Slow enough that the compliant controller has time to actually settle
    #/sink into the opening during each pass near it, rather than sweeping
    # past too fast to react (tangential speed at max radius is
    # 2*pi*search_spiral_freq_hz*search_spiral_max_radius_m -- at the
    # originally-tried 0.7 Hz that's ~44mm/s, which measurably missed the
    # opening on some landings even when the spiral geometrically passed
    # within tolerance of hole center, see README_insertion.md's tuning log).
    search_spiral_freq_hz: float = 0.2
    search_timeout_s: float = 20.0
    hole_found_past_touch_m: float = 0.004  # depth past touch height at which SEARCH -> INSERT

    # ---- INSERT
    insert_wiggle_amplitude_n: float = 1.0
    insert_wiggle_freq_hz: float = 1.8
    # Deep re-target once the hole is found, measured (like
    # search_penetration_m) as meters BELOW contact_control_z (the touch
    # height), not below hole_entrance. hole_entrance itself sits
    # socket_lip_height_m + peg_radius_m (0.025m by default) below the touch
    # height (see touch_depth_m in run_scripted_demo), so to comfortably
    # clear InsertionProperties.insert_depth_target_m (0.03m past
    # hole_entrance, with margin) while staying short of the floor
    # (socket_depth_m=0.05m past hole_entrance), this needs to be at least
    # ~0.025+0.03=0.055m; 0.065m leaves the peg tip ~40mm past hole_entrance,
    # 10mm short of the floor.
    insert_target_depth_m: float = 0.065
    insert_descent_speed_mps: float = 0.03
    insert_timeout_s: float = 10.0
    wiggle_ramp_steps: int = 50  # ported from insertion_episode.py's ramp_steps=50
    # Admittance-style safety clamp: the commanded z target is never allowed
    # to run more than max_lead_m AHEAD (i.e. deeper) of the peg's actual
    # current position. Without this, a plain time-parameterized min-jerk
    # descent keeps advancing its target regardless of whether the peg is
    # actually following -- if the peg jams sideways against a wall inside
    # the tunnel (the reference sim handles this with an explicit
    # STUCK/UNSTUCK/ALIGNED hysteresis state machine, insertion_episode.py's
    # insert_state, which is NOT ported here, see README_insertion.md), the
    # tracking error (and therefore commanded force, force = kp * error)
    # grows without bound until either it un-jams or trips force_break_n.
    # Clamping the lead distance bounds the worst-case commanded force to
    # roughly ``tool_kp * max_lead_m`` regardless of how long the peg stays
    # stuck, trading a slower recovery for a hard safety guarantee -- this
    # is this port's (simpler) substitute for the reference's explicit
    # jam-recovery state machine.
    max_lead_m: float = 0.006


@dataclass
class DemoResult:
    success: bool
    termination_reason: str
    time_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    force_n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    depth_m: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phase: list = field(default_factory=list)
    peak_force_n: float = 0.0
    mean_force_n: float = 0.0
    std_force_n: float = 0.0
    search_mean_force_n: float = 0.0
    search_std_force_n: float = 0.0
    steps: int = 0


def _wiggle_force(t: float, amplitude_n: float, freq_hz: float, push_force_n: float) -> np.ndarray:
    """Per-axis sinusoidal wiggle in x/y, constant push bias in z.

    Ported in spirit from insertion_episode.py:269-270 / :150-151::

        Fff    = a * np.sin(2*np.pi*f*t + phi)
        Fff[2] = -az

    Simplified to a single shared amplitude/frequency/phase=0 for x and y
    (the reference uses distinct per-axis a/f/phi values, see
    insertion_episode.py's search.wiggle/insert.wiggle config in the
    extraction notes) -- this repo's much smaller hole clearance (2mm radial,
    vs. the reference's sub-mm-to-mm tolerances at a very different absolute
    scale) doesn't benefit from the extra per-axis richness for a first
    port, and a simple circular/elliptical wiggle already reliably finds the
    opening in testing (see README_insertion.md).
    """
    fx = amplitude_n * np.sin(2.0 * np.pi * freq_hz * t)
    fy = amplitude_n * np.sin(2.0 * np.pi * freq_hz * t + np.pi / 2.0)
    return np.array([fx, fy, -push_force_n, 0.0, 0.0, 0.0])


def run_scripted_demo(
    env: InsertionEnv | None = None,
    cfg: ScriptedDemoConfig | None = None,
    seed: int = 0,
    max_steps: int = 60000,
    record: bool = True,
    frame_callback=None,
):
    """Run one scripted insertion episode against ``env`` (a fresh
    InsertionEnv is created if not supplied), returning a DemoResult.

    ``frame_callback``, if given, is invoked as ``frame_callback(env,
    force_n, phase)`` once per main-loop step (same cadence as the force
    recording below) -- purely a video-rendering hook for
    render_insertion_demos.py; it has no effect on the control/state-machine
    logic above and is a no-op when ``record`` is False.
    """
    cfg = cfg or ScriptedDemoConfig()
    owns_env = env is None
    if owns_env:
        env = InsertionEnv(seed=seed)
    dt = env.timestep

    entrance = env.hole_entrance_pos.copy()
    surface_z = entrance[2] + cfg.socket_lip_height_m
    contact_control_z = surface_z + cfg.peg_radius_m  # planner_tip_site height at zero-penetration touch
    # peg_tip_depth_m() value at zero penetration (touching the lip's top
    # surface) -- see fell_through_past_touch_m's comment for why this isn't 0.
    touch_depth_m = entrance[2] - contact_control_z

    rng = np.random.default_rng(seed)
    landing_xy = entrance[:2] + rng.normal(0.0, cfg.landing_perturbation_std_m, size=2)

    times, forces, depths, phases = [], [], [], []

    phase = Phase.APPROACH
    start_pos = env.tool_pos.copy()
    hover_target = np.array([landing_xy[0], landing_xy[1], contact_control_z + cfg.hover_clearance_m])
    approach_target = np.array([landing_xy[0], landing_xy[1], contact_control_z + cfg.approach_gap_m])
    seg = MinJerkSegment(start_pos, hover_target, cfg.transit_speed_mps)
    landed_xy = landing_xy.copy()

    # CONTACT/SEARCH shared z target (reached via a bounded-rate slew, not a
    # single min-jerk segment, so the exact instant contact happens doesn't
    # matter for trajectory timing).
    search_target_z = contact_control_z - cfg.search_penetration_m
    cur_z = None  # set once APPROACH's second segment finishes

    contact_elapsed_s = 0.0
    contact_confirm_elapsed_s = 0.0
    search_elapsed_s = 0.0
    insert_elapsed_s = 0.0
    insert_seg = None
    wiggle_t = 0.0
    last_wiggle_ff = np.zeros(6)
    ramp_step = 0
    reason = "running"
    success = False

    def _record(force_vec, depth, ph):
        times.append(env.data.time)
        forces.append(float(np.linalg.norm(force_vec)))
        depths.append(depth)
        phases.append(ph)

    approach_stage = 0  # 0: transit to hover, 1: descend to approach_target

    for step_i in range(max_steps):
        if phase == Phase.APPROACH:
            pos = seg.step(dt)
            ok = env.step(pos)
            if not ok:
                reason = "viewer_closed"
                break
            if seg.done:
                if approach_stage == 0:
                    approach_stage = 1
                    seg = MinJerkSegment(hover_target, approach_target, cfg.descent_speed_mps)
                else:
                    phase = Phase.CONTACT
                    cur_z = approach_target[2]
        elif phase == Phase.CONTACT:
            if cur_z > search_target_z:
                cur_z = max(search_target_z, cur_z - cfg.contact_settle_speed_mps * dt)
            target = np.array([landed_xy[0], landed_xy[1], cur_z])
            ff = np.array([0.0, 0.0, -cfg.push_force_n, 0.0, 0.0, 0.0])
            ok = env.step(target, feed_forward_wrench=ff)
            if not ok:
                reason = "viewer_closed"
                break
            depth = env.peg_tip_depth_m()
            # Exact solver contact force (zero in free space by
            # construction, see InsertionEnv.peg_contact_force), NOT the
            # simulated wrist F/T sensor -- the raw sensor also reads the
            # peg's own held weight (~mass*g) even absent any contact, which
            # would trip a low force_contact_threshold_n immediately in free
            # space. This mirrors how sanding/flipup's own break/dose logic
            # reads the exact solver force rather than a raw sensor for
            # threshold decisions; wrist_wrench_filtered() remains available
            # for reporting/recording (it's what the CLI/recorder use), just
            # not for this state-machine's ground-truth contact decision.
            exact_force = env.normal_force_n()
            contact_elapsed_s += dt
            if depth >= touch_depth_m + cfg.fell_through_past_touch_m:
                phase = Phase.SEARCH
                wiggle_t = 0.0
                last_wiggle_ff = ff.copy()
            elif exact_force >= env.properties.force_contact_threshold_n:
                contact_confirm_elapsed_s += dt
                if contact_confirm_elapsed_s >= env.properties.contact_confirm_time_s:
                    phase = Phase.SEARCH
                    wiggle_t = 0.0
                    last_wiggle_ff = ff.copy()
            else:
                contact_confirm_elapsed_s = 0.0
            if contact_elapsed_s >= cfg.contact_timeout_s and phase == Phase.CONTACT:
                reason = "contact_timeout"
                break
        elif phase == Phase.SEARCH:
            # Expanding position spiral around the landed point -- see
            # search_spiral_max_radius_m's comment for why this (not a pure
            # force wiggle) is the primary search mechanism here.
            spiral_radius = cfg.search_spiral_max_radius_m * min(
                1.0, search_elapsed_s / cfg.search_spiral_growth_time_s
            )
            theta = 2.0 * np.pi * cfg.search_spiral_freq_hz * search_elapsed_s
            spiral_xy = landed_xy + spiral_radius * np.array([np.cos(theta), np.sin(theta)])
            target = np.array([spiral_xy[0], spiral_xy[1], cur_z])
            ff = _wiggle_force(
                wiggle_t, cfg.search_wiggle_amplitude_n, cfg.search_wiggle_freq_hz, cfg.push_force_n
            )
            wiggle_t += dt
            ok = env.step(target, feed_forward_wrench=ff)
            if not ok:
                reason = "viewer_closed"
                break
            last_wiggle_ff = ff.copy()
            depth = env.peg_tip_depth_m()
            search_elapsed_s += dt
            if depth >= touch_depth_m + cfg.hole_found_past_touch_m:
                phase = Phase.INSERT
                ramp_step = 0
                # Anchor INSERT's xy at wherever the peg actually is right
                # now (i.e. inside the found opening), NOT the original
                # landed_xy -- the spiral search has since moved the xy
                # target away from that original point.
                found_xy = env.tool_pos[:2].copy()
                insert_target_z = contact_control_z - cfg.insert_target_depth_m
                insert_seg = MinJerkSegment(
                    np.array([found_xy[0], found_xy[1], cur_z]),
                    np.array([found_xy[0], found_xy[1], insert_target_z]),
                    cfg.insert_descent_speed_mps,
                )
            elif search_elapsed_s >= cfg.search_timeout_s:
                reason = "search_timeout"
                break
        elif phase == Phase.INSERT:
            pos = insert_seg.step(dt)
            # Admittance-style lead clamp, see max_lead_m's docstring: never
            # command the peg tip more than max_lead_m below where it
            # actually currently is.
            pos[2] = max(pos[2], env.tool_pos[2] - cfg.max_lead_m)
            wiggle_ff = _wiggle_force(
                wiggle_t, cfg.insert_wiggle_amplitude_n, cfg.insert_wiggle_freq_hz, cfg.push_force_n
            )
            wiggle_t += dt
            # Ramp-in the new phase's wiggle from the previous phase's last
            # commanded feed-forward over wiggle_ramp_steps control steps --
            # ported from insertion_episode.py:115-116,153-156's
            # ramp_steps=50 linear blend on phase transitions.
            if ramp_step < cfg.wiggle_ramp_steps:
                alpha = ramp_step / cfg.wiggle_ramp_steps
                ff = (1.0 - alpha) * last_wiggle_ff + alpha * wiggle_ff
                ramp_step += 1
            else:
                ff = wiggle_ff
            ok = env.step(pos, feed_forward_wrench=ff)
            if not ok:
                reason = "viewer_closed"
                break
            depth = env.peg_tip_depth_m()
            insert_elapsed_s += dt
            if env.success():
                phase = Phase.DONE
                success = True
                reason = "success"
            elif env.broken:
                reason = "broken"
                break
            elif insert_elapsed_s >= cfg.insert_timeout_s:
                reason = "insert_timeout"
                break

        # Global break check, independent of which phase is active: a jam
        # hard enough to trip force_break_n is a failure regardless of
        # whether it happened during CONTACT/SEARCH's frame-sliding or
        # INSERT's descent (the per-phase loop above only explicitly checked
        # env.broken inside the INSERT branch, which silently missed
        # CONTACT/SEARCH-phase breaks during tuning -- see README_insertion.md).
        if env.broken and reason == "running":
            reason = "broken"

        if record:
            force_vec, _ = env.peg_contact_force()
            _record(force_vec, env.peg_tip_depth_m(), phase)
            if frame_callback is not None:
                frame_callback(env, float(np.linalg.norm(force_vec)), phase)

        if reason == "broken":
            break

        if phase == Phase.DONE:
            # Hold a little longer post-success so the recorded force trace
            # shows the settled final state, not just the instant of success.
            for _ in range(200):
                env.step(np.array([landed_xy[0], landed_xy[1], insert_seg.p_end[2]]))
                if record:
                    force_vec, _ = env.peg_contact_force()
                    _record(force_vec, env.peg_tip_depth_m(), phase)
                    if frame_callback is not None:
                        frame_callback(env, float(np.linalg.norm(force_vec)), phase)
            break

    if reason == "running":
        reason = "max_steps"

    times_arr = np.asarray(times)
    forces_arr = np.asarray(forces)
    depths_arr = np.asarray(depths)
    search_mask = np.array([p == Phase.SEARCH for p in phases])

    result = DemoResult(
        success=success,
        termination_reason=reason,
        time_s=times_arr,
        force_n=forces_arr,
        depth_m=depths_arr,
        phase=phases,
        peak_force_n=float(forces_arr.max()) if forces_arr.size else 0.0,
        mean_force_n=float(forces_arr.mean()) if forces_arr.size else 0.0,
        std_force_n=float(forces_arr.std()) if forces_arr.size else 0.0,
        search_mean_force_n=float(forces_arr[search_mask].mean()) if search_mask.any() else 0.0,
        search_std_force_n=float(forces_arr[search_mask].std()) if search_mask.any() else 0.0,
        steps=len(times),
    )
    if owns_env:
        env.close()
    return result


def _plot(result: DemoResult, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    phase_colors = {
        Phase.APPROACH: "#9ecae1",
        Phase.CONTACT: "#fdae6b",
        Phase.SEARCH: "#fd8d3c",
        Phase.INSERT: "#74c476",
        Phase.DONE: "#31a354",
    }
    phases_arr = np.array(result.phase)
    t = result.time_s
    ax = axes[0]
    for ph, color in phase_colors.items():
        mask = phases_arr == ph
        if mask.any():
            ax.scatter(t[mask], result.force_n[mask], s=2, color=color, label=ph)
    ax.set_ylabel("|contact force| (N)")
    ax.legend(loc="upper right", markerscale=4, fontsize=8)
    ax.set_title(
        f"Insertion scripted demo -- success={result.success} "
        f"peak={result.peak_force_n:.1f}N mean={result.mean_force_n:.2f}N "
        f"std={result.std_force_n:.2f}N"
    )

    ax2 = axes[1]
    for ph, color in phase_colors.items():
        mask = phases_arr == ph
        if mask.any():
            ax2.scatter(t[mask], 1000.0 * result.depth_m[mask], s=2, color=color)
    ax2.axhline(0.0, color="k", linewidth=0.5)
    ax2.set_ylabel("peg tip depth (mm)")
    ax2.set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved force-profile plot to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=str, default=None, help="Path to save a force-profile PNG")
    parser.add_argument("--max-steps", type=int, default=60000)
    args = parser.parse_args()

    result = run_scripted_demo(seed=args.seed, max_steps=args.max_steps)
    print(f"termination_reason={result.termination_reason} success={result.success} steps={result.steps}")
    print(
        f"peak_force_n={result.peak_force_n:.3f} mean_force_n={result.mean_force_n:.3f} "
        f"std_force_n={result.std_force_n:.3f}"
    )
    print(
        f"search_mean_force_n={result.search_mean_force_n:.3f} "
        f"search_std_force_n={result.search_std_force_n:.3f}"
    )
    if args.plot:
        _plot(result, args.plot)


if __name__ == "__main__":
    main()
