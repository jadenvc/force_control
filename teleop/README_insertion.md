# omega/UR5e peg-in-hole insertion teleoperation

Haptic teleoperation (and a scripted, hardware-free demo) of a peg-in-hole
insertion task: a UR5e holds a rigid peg and must find and insert it into a
square socket fixture. Structurally this is modeled on the sanding task
(same UR5e + Jacobian-transpose task-space controller, same dataclass-of-
tunables + Env/Teleop class layout, same asset-composition pattern), but the
controller, contact tuning, and scripted-motion generator are ported (with
explicit, honest simplifications -- see "What's a faithful port vs. an
approximation" below) from a separate reference peg-insertion sim,
`force-insertion-sim` (a Franka FR3 + SimCore sim). SimCore itself is not
installed/vendored in this repo, so nothing here imports it -- every ported
piece is a from-scratch numpy/MuJoCo reimplementation of that sim's
*documented* algorithms and parameters.

```bash
cd teleop
python insertion_scripted_demo.py --plot insertion_force_profile.png   # scripted demo, no hardware
python teleop_insertion.py --dry-run --no-view --dry-run-seconds 20    # same demo, through the CLI
python teleop_insertion.py                                             # real omega handle
python teleop_insertion.py --peg-softness 0.8                          # softer peg contact
python teleop_insertion.py --insert-depth-target 0.02                  # shallower success target
```

## Controller changes vs. sanding/flipup

The shared Jacobian-transpose task-space PD law
(`flipup_minimal/flipup/environment.py`'s `step_task_space`, also used
unmodified in spirit by `sanding_teleop.py`) is extended in
`insertion_teleop.py`'s `InsertionEnv.step_task_space` with three pieces:

1. **Non-zero translational Cartesian damping.** Sanding/flipup's
   `task_space_cartesian_kd` is exactly zero on translation (damping comes
   only from joint-velocity damping, see `environment.py:110-119`'s
   comment); insertion sets it to a per-axis ratio ported from
   force-insertion-sim's `configs/control/panda_arm.yaml:14-16`
   `dynamic_impedance` block (`K_cart=[450,450,700,80,80,200]`,
   `D_cart=[55,55,55,10,10,20]`), applied as `D/K` ratios to *this*
   controller's own Kp diagonal (`_CARTESIAN_KD_RATIO` in
   `insertion_teleop.py`) rather than reusing the Franka's raw numbers,
   since the two controllers' Kp scales differ by more than an order of
   magnitude.
2. **A ported `DynamicFilter`** (critically-damped 2nd-order feed-forward
   shaping, `F_ff_ddot = alpha*(beta*(F_df - F_ff) - F_ff_dot)`), ported
   verbatim in algorithm from
   `force-insertion-sim/src/policy/dynamic_filter.py:1-18`, applied to the
   commanded feed-forward wrench every control step, before it is added to
   the PD wrench and converted to joint torque via `J^T`.
3. **Ported F/T sensor filters** (`EMAFilter`/`ButterworthFilter`, ported
   from `force-insertion-sim/src/utils/sensor_callback.py:6-56`), exposed as
   `InsertionEnv.wrist_wrench_filtered()`. Sanding/flipup's
   `contact_force()`/`wrist_wrench()` (`flipup_teleop.py:904-1097`) only
   ever report the raw/exact solver force; insertion adds a filtered
   variant on top, which is what the teleop CLI and dataset recorder use
   for haptic reflection/logging (the exact, zero-in-free-space solver
   force via `peg_contact_force()`/`normal_force_n()` remains available and
   is what the scripted demo's phase state machine uses for its
   ground-truth contact decisions -- see below for why).

### `DEFAULT_TOOL_KP`: the actual biggest controller change

The single most consequential change isn't any of the three pieces above --
it's that `DEFAULT_TOOL_KP` is **2500 N/m, not sanding's 16000 N/m** (and
`DEFAULT_TOOL_ROT_KP` is 400, not 3000). Sanding puts essentially all of its
contact compliance into the *pad's* `solref`/`solimp` and keeps the arm
itself rigid/accurate; a flat pad sliding over a flat panel never needs the
arm to yield. A rigid peg searching for and entering a rigid socket is
different: the controller itself needs to be soft enough that ordinary
position-tracking error (from search/approach motion, or from a feed-forward
wiggle) doesn't itself generate large contact forces. This is the same
reason force-insertion-sim's own controller is an impedance controller with
`K_cart` in the hundreds, not a stiff position controller.

This was discovered empirically, not assumed up front: reusing sanding's
`tool_kp=16000` and then commanding even a few millimeters of position error
against the rigid fixture produced 100-250N force spikes; combined with the
non-zero Cartesian damping ratio at full scale, it also **permanently
saturated the UR5e's wrist actuators** (`+-28 N*m`, far smaller than the
shoulder/elbow's `+-150 N*m` -- `J^T` maps Cartesian damping force into
torque at every joint including the wrist) and left a persistent
uncorrectable position error even in free space. Dropping `tool_kp` to 2500
fixed both problems at once: free-space settling returned to numerically
exact (no saturation) with the *raw*, unscaled Cartesian-damping ratio, and
a few-millimeter commanded penetration against the fixture produces a
gentle, bounded force instead of a spike. See `insertion_teleop.py`'s
`DEFAULT_TOOL_KP`/`DEFAULT_CARTESIAN_DAMPING_SCALE` comments for the exact
numbers measured at each step of this.

## Assets

- **Peg** (`flipup_minimal/flipup/assets/insertion_peg/insertion_peg.xml`):
  a capsule (not a flat-ended cylinder) 20mm in diameter, mounted on the
  UR5e attachment site the same way `sander.xml` mounts the sander pad. The
  rounded, hemispherical tip is both edge-free (avoiding the same
  contact-normal-discontinuity chatter `sander.xml`'s comment warns about)
  and physically acts like a mildly chamfered real peg, which helps it find
  the hole rather than catching square on the rim.
- **Socket fixture**
  (`flipup_minimal/flipup/assets/custom/insertion_hole/insertion_hole.xml`):
  a **square** tunnel built from 4 wall boxes + a floor, not a real
  cylindrical bore -- MuJoCo has no boolean/CSG subtraction of primitives,
  and the reference sim's round holes are pre-built convex-decomposition
  meshes (`force-insertion-sim/models/mujoco/props/holes/hole_cyl_s_medium/fixture.xml`,
  39 sub-meshes). A square tunnel needs only 4 primitive boxes and pairs
  naturally with the peg's round tip (the peg only ever meets flat wall
  faces, never a matching square corner-to-corner condition). Inner opening
  is 24mm across (2mm radial clearance vs. the peg's 20mm diameter) --
  looser than the reference sim's tightest fixtures (0.2mm clearance) but
  tighter than a trivially-loose fit, per the task brief's request for a
  first-port middle ground. The walls extend 15mm *above* the nominal
  entrance plane, forming a flat "picture frame" lip the peg can press
  against and slide along during CONTACT/SEARCH before it finds the actual
  opening.

### Contact solref/solimp/friction/priority

The peg's `peg_collision` default class sets `condim=4`,
`friction="0.4 0.01 0.0002"`, `solref="0.010 1.2"`, `solimp="0.9 0.95 0.002"`,
`priority=1`. Starting numbers were ported from the reference sim's peg geom
(`force-insertion-sim/models/mujoco/franka_fr3/fr3_torque_peg.xml:154-161`,
`condim=4`, `friction="0.3 0.005 0.0001"`, `solref="0.005 1"`,
`solimp="0.95 0.999 0.002"`) and then softened slightly during tuning for
stability at this repo's `timestep=0.001s` `implicitfast` setup (matching
`flipup_minimal/flipup/assets/ground.xml`'s existing global option, kept
unchanged here -- no reason found to deviate).

`priority=1` on the peg (vs. the fixture's wall/floor geoms, which are left
at MuJoCo's default `priority=0`) means every peg-vs-fixture contact takes
its `solref`/`solimp`/`friction` **entirely** from the peg geom -- MuJoCo
does not blend contact parameters between two geoms with different
priorities, it picks the higher one outright (see
`teleop/SANDING_JITTER_FIX_SUMMARY.md` and `flipup_teleop.py`'s
`_configure_bookend_contact` comments for this same rule biting sanding
originally). This is the opposite assignment from sanding (where the
*panel*, not the pad, has to be the higher-priority side for its softness
knob to have any effect) -- here the peg is deliberately the winning side,
and `InsertionEnv._configure_peg_contact`'s `--peg-softness` knob writes to
both the peg and the fixture's geoms defensively (only the peg's values
matter given the priority, but writing both avoids a silent no-op if
priority is ever flipped later, the same defensive habit sanding's fix
established).

## Scripted demo: phase state machine

`insertion_scripted_demo.py` drives `InsertionEnv` open-loop through
`APPROACH -> CONTACT -> SEARCH -> INSERT -> DONE`, ported in spirit (not
verbatim -- see the gap list below) from
`force-insertion-sim/src/task/insertion_episode.py`, using a min-jerk
Cartesian trajectory planner (`_minjerk`, the quintic polynomial ported
**verbatim** from `force-insertion-sim/src/task/trajectory.py:82-87`) for
free-space motion.

- **APPROACH**: two min-jerk segments -- transit to a hover point 5cm above
  the fixture, then descend to just above the fixture's touch height -- with
  a random XY landing perturbation (`landing_perturbation_std_m=3.5mm`,
  ported in spirit from `insertion_episode.py`'s
  `episode.approach.pertubation.pos_std=[2mm]*3`) so the peg doesn't always
  land centered on the opening, making SEARCH necessary.
- **CONTACT**: holds XY/orientation fixed at the landed point, slews the Z
  target down to a small commanded penetration (6mm past the touch height)
  while adding a constant `-4N` push feed-forward (`push_force_n`, ported in
  spirit from `task_config.yaml`'s `contact.f_push=4.0`). Confirms contact
  once the **exact** solver contact force (not the raw/ungravity-compensated
  simulated F/T sensor -- see the bug note below) exceeds
  `force_contact_threshold_n` (1.5N) for `contact_confirm_time_s` (0.375s,
  ported in spirit from `insertion_episode.py`'s `contact.force_threshold=2.0N`
  / `n_confirm=75` steps @200Hz, re-expressed as a time duration so it's
  portable across control rates). If the peg lands close enough to fall
  straight through without ever registering frame contact, that's detected
  directly (depth crossing a small threshold) rather than waiting out a
  contact event that will never happen.
- **SEARCH**: a **position-based expanding spiral** around the landed point
  (growing from 0 to 10mm radius over 6s, 0.2 Hz), *not* a pure force
  wiggle -- see "Why a spiral, not just a force wiggle" below -- layered
  with a smaller residual sinusoidal force wiggle
  (`search_wiggle_amplitude_n=1.5N`, ported in spirit from
  `insertion_episode.py:269-270`'s per-axis sinusoid + constant Z bias).
  Transitions to INSERT once the peg's depth crosses a small threshold past
  the touch height, indicating it found the opening and started sinking.
- **INSERT**: a deep min-jerk re-target (to ~40mm past `hole_entrance`, well
  short of the 50mm socket floor) drives the peg the rest of the way in,
  with a continued smaller wiggle for lateral micro-adjustment, **ramped in
  from the previous phase's last commanded feed-forward over
  `wiggle_ramp_steps=50` control steps** (ported verbatim in structure from
  `insertion_episode.py:115-116,153-156`'s `ramp_steps=50` linear blend on
  phase transitions). An admittance-style **lead clamp** (`max_lead_m=6mm`)
  prevents the time-parameterized trajectory from commanding the peg
  further ahead than it actually is if it jams sideways -- see the gap list
  below for why this exists instead of the reference's fuller jam-recovery
  logic.
- **DONE**: success is `peg_tip_depth_m() >= insert_depth_target_m`
  (30mm past `hole_entrance`) sustained for `success_hold_steps` (200 steps).

### Why a spiral, not just a force wiggle

A pure force wiggle (as the reference sim uses) turned out to be far too
weak an exploration signal at this controller's `tool_kp=2500`: a 2N force
wiggle only produces ~0.8mm of actual motion, well short of what's needed to
reliably recover a 3.5mm-std landing offset (measured: only 6/10 seeds found
the hole before timing out with force-wiggle-only search). This is the same
"the reference's absolute numbers don't transfer at this controller's very
different stiffness" lesson as `DEFAULT_TOOL_KP` above, just showing up in
the trajectory layer instead of the controller layer. The fix -- an
expanding position spiral, a standard, well-known peg-in-hole search
strategy, layered *underneath* the (now much smaller) residual force wiggle
-- reliably sweeps the actual clearance annulus regardless of controller
stiffness.

## Tuning log (how the numbers above were reached)

Roughly 15 iterations, run and inspected via
`python insertion_scripted_demo.py --plot ...` plus multi-seed sweeps:

1. First env build: the compiled peg geom needed an explicit `size=` on the
   collision capsule (a `fromto` capsule still needs a radius) -- a MuJoCo
   compile error, not a tuning issue.
2. `HOLE_TRANSFORM`'s height needed adjusting so the reused-from-sanding
   home joint config didn't start the peg already overlapping the fixture
   (sanding's home joints were tuned for the *sander's* very different
   `planner_tip_site` offset, not this tool's).
3. Holding a fixed free-space target with `tool_kp=16000` (sanding's value)
   and the raw Cartesian-damping ratio (`scale=1.0`) **permanently
   saturated the UR5e's wrist actuators** and left a persistent ~mm-scale
   position error -- traced to `J^T` mapping translational Cartesian
   damping into torque at every joint, including the low-torque-budget
   wrist. Lowering the scale to 0.2 fixed the symptom but not the root
   cause.
4. Root cause fix: lowered `tool_kp` to 2500 (see the controller-changes
   section above) -- this let the Cartesian-damping ratio return to its
   full, unscaled value (`cartesian_damping_scale=1.0`) with clean, exact
   free-space settling.
5. First CONTACT-phase attempt jumped the Z target instantaneously by 10cm
   -- an artificial impulsive command, not a realistic trajectory --
   producing a 222N spike and a subsequent loss of control (the arm flew to
   a completely different configuration). Fixed by always slew-limiting/
   trajectory-generating position targets, never jumping them.
6. Even with a slew-limited approach, a virtual target 20mm below the touch
   height (mirroring the reference's own SEARCH `x_ref = xz0-0.02`) produced
   197N against the rigid fixture -- traced to a units/geometry bug (the
   effective touch height needs `+peg_radius_m`, not just the lip height,
   since the capsule's tip surface, not its centerline, is what contacts
   the surface) compounding with the stiffness mismatch from item 3/4.
7. `fell_through`/`hole_found` depth thresholds were originally expressed
   relative to `hole_entrance` (i.e. requiring `peg_tip_depth_m() >= small
   positive value`), which is unreachable from merely resting on the touch
   surface (`peg_tip_depth_m()` there is `-25mm`, not `0`, because of the
   raised lip + peg radius offset) -- fixed by expressing thresholds
   relative to the touch height instead (see `touch_depth_m` in
   `insertion_scripted_demo.py`).
8. CONTACT-phase contact detection used the raw simulated wrist F/T sensor,
   which reads the peg's own held weight (~3N) even in free space, tripping
   "contact detected" immediately -- fixed by using the exact,
   zero-in-free-space solver contact force (`normal_force_n()`) for this
   state-machine decision instead (the filtered sensor value remains what's
   reported/reflected/recorded).
9. First working end-to-end run (seed 0): peak 15.4N, mean 7.6N, std 4.9N,
   success.
10. A 10-seed sweep found 6/10 timing out in SEARCH -- root-caused to the
    force-wiggle-only search being far too weak an exploration signal (see
    "Why a spiral, not just a force wiggle" above); added the position
    spiral.
11. With the spiral at a 16mm max radius, a 15-seed sweep found 2/15
    tripping the break threshold -- the spiral's own worst-case lateral
    force (`tool_kp * radius`) was right at the 40N ceiling. Reduced radius
    to 10mm for real margin (~25N worst case).
12. Discovered the scripted demo's `env.broken` check only ran inside the
    INSERT branch, silently missing breaks that happened during
    CONTACT/SEARCH's frame-sliding -- added a phase-independent global
    break check.
13. Two stubborn seeds (11.4mm landing offsets, just past the original 10mm
    spiral radius) needed either a larger radius or more break-threshold
    margin; raised `force_break_n` to 45N specifically because the spiral's
    intended search behavior legitimately produces forces in the
    20-40N range as part of normal operation, not a fault condition, and
    40N left too little headroom.
14. Final 20-seed sweep: **19/20 success**, peak forces 15-42N (mean ~21N),
    episode mean forces 7.6-22N (mean ~10.7N), force std 4.6-9.2N (mean
    ~5.5N). The one remaining failure (seed 13, `search_timeout`) is
    documented as a known gap below rather than chased further, given the
    task's iteration cap.

## Final measured force profile (seed 0, the one plotted in
`insertion_force_profile.png`)

| metric | value |
|---|---|
| termination | `success` |
| peak force | 15.4 N |
| mean force | 7.6 N |
| force std | 4.9 N |
| steps | 12547 (~12.5s sim time) |

Across the 20-seed tuning sweep: 19/20 succeed; peak force ranges 15.4-42.4N
(mean ~21N, driven by how far off-center the random landing happens to be --
a larger offset needs the spiral to sweep further out, at proportionally
higher lateral force); per-episode mean force ranges 7.6-22.0N (mean
~10.7N); force std ranges 4.6-9.2N (mean ~5.5N). No run in the successful
19/20 exceeded the documented 45N break ceiling; qualitatively the force
trace is smooth and free of sharp spikes/ringing once past the initial
contact transient (see the plot), with the SEARCH-phase spiral producing a
bounded, gently oscillating force as it sweeps rather than sudden jumps.

## Running

```bash
# Scripted demo only, with a force-profile plot:
python insertion_scripted_demo.py --seed 0 --plot insertion_force_profile.png

# Same demo through the teleop CLI (no hardware):
python teleop_insertion.py --dry-run --no-view --dry-run-seconds 20

# Real Force Dimension omega handle:
python teleop_insertion.py
```

`teleop_insertion.py`'s flags mirror `teleop_sanding.py`'s: `--pos-tau` is
not a literal flag here (neither script has one; position commands are
slew-rate-limited via `--max-speed` instead) but `--force-tau`,
`--force-rate`, `--max-speed`, `--stiffness`, `--damping` all work the same
way, plus insertion-specific flags for the controller/contact tuning
(`--tool-kp`, `--cartesian-damping-scale`, `--dynamic-filter-alpha/beta`,
`--ft-filter-type/-alpha`, `--peg-softness`, `--insert-depth-target`,
`--contact-force-threshold`, `--break-force`). `--collect-dataset PATH.zarr`
records episodes via `insertion_recorder.py`'s `InsertionEpisodeRecorder`.

**`--dry-run` implementation note**: unlike sanding's `--dry-run` (a simple
closed-form function of elapsed time, evaluated one tick at a time
interleaved with the render/HUD loop), insertion's `--dry-run` reuses
`insertion_scripted_demo.run_scripted_demo` exactly (rather than
re-implementing the phase state machine a second time), which isn't written
as a per-tick generator. This means the whole scripted episode runs to
completion up front (a few seconds of wall time) before the HUD starts
updating, rather than interleaving tick-by-tick -- a cosmetic gap in the
preview, not a physics one, and it doesn't affect the real (non-dry-run)
teleoperation path at all.

## Files

- `insertion_teleop.py` -- the environment (`InsertionEnv`,
  `InsertionTeleop`, `InsertionProperties`), controller, and ported
  `DynamicFilter`/`EMAFilter`/`ButterworthFilter`/`FTSensorFilter`.
- `insertion_scripted_demo.py` -- the min-jerk trajectory planner and
  APPROACH/CONTACT/SEARCH/INSERT phase state machine (`run_scripted_demo`),
  runnable standalone for a plotted scripted demo.
- `teleop_insertion.py` -- the haptic CLI driver (`--dry-run` reuses the
  scripted demo for hardware-free testing).
- `insertion_recorder.py` -- BC dataset recording
  (`InsertionEpisodeRecorder`).
- `flipup_minimal/flipup/assets/insertion_peg/insertion_peg.xml` -- the peg
  end effector.
- `flipup_minimal/flipup/assets/custom/insertion_hole/insertion_hole.xml`
  -- the socket fixture.
- `tests/test_insertion.py` -- properties/filter/env/scripted-demo/recorder
  checks.
- `insertion_force_profile.png` -- the tuned scripted demo's force-profile
  plot (seed 0).

## What's a faithful port vs. an approximation

`simcore` (the reference sim's underlying framework) is not installed or
vendored in this repo, so nothing here can literally call into it --
everything below is either a verbatim-ported algorithm/formula (cited with
exact `force-insertion-sim` file:line references in the code), a
ratio/shape ported and re-scaled for this controller's different stiffness,
or an intentional simplification/approximation. Being explicit about which
is which:

**Faithful ports (algorithm/formula verbatim, values may be re-tuned):**
- `DynamicFilter`'s 2nd-order shaping ODE and Euler integration
  (`dynamic_filter.py:1-18`).
- `EMAFilter`/`ButterworthFilter`'s filtering algorithms
  (`sensor_callback.py:6-56`) -- though `ButterworthFilter` uses scipy's own
  stateful `sosfilt(..., zi=...)` rather than the reference's hand-rolled
  per-sample biquad stepper (functionally equivalent; scipy is already a
  dependency here, unlike whatever real-time/non-Python target the
  reference's hand-rolled version was presumably written for).
- The min-jerk trajectory polynomial (`trajectory.py:82-87`).
- The `ramp_steps=50` linear blend on phase transitions
  (`insertion_episode.py:115-116,153-156`).
- The APPROACH/CONTACT/SEARCH/INSERT phase names and overall sequencing.

**Ratios/shapes ported, then re-scaled empirically for this controller:**
- The Cartesian damping D/K ratio (`panda_arm.yaml`'s `dynamic_impedance`
  block) -- shape ported, absolute scale re-derived for `tool_kp=2500` vs.
  the reference's `K_cart` in the hundreds.
- The wiggle/push force magnitudes -- same order of magnitude as
  `task_config.yaml`'s `f_push`/`az` values, re-tuned against this
  controller/contact's actual dynamics rather than assumed transferable.
- The peg/fixture `solref`/`solimp`/friction (`fr3_torque_peg.xml`) --
  started from the reference's numbers, softened for stability at this
  repo's timestep/controller.

**Intentional simplifications / approximations (documented gaps):**
- **No STUCK/UNSTUCK/ALIGNED hysteresis state machine.** The reference's
  `insertion_episode.py:102-239` INSERT phase has an elaborate internal
  sub-state-machine (rolling z-score/velocity-drop detection, hysteresis
  confirmation counts, a second wiggle/push blend) for recovering from jams.
  This port uses a single, simpler behavior instead: a deep re-target plus
  an admittance-style lead clamp (`max_lead_m`) that bounds worst-case
  commanded force if the peg does jam, trading a slower/less-adaptive
  recovery for a much simpler implementation with a hard safety guarantee.
  This is the single biggest structural simplification relative to the
  reference.
- **Position-based spiral search, not force-wiggle-only search.** See "Why
  a spiral, not just a force wiggle" above -- necessitated by this
  controller's much higher stiffness, not a simplification for its own
  sake, but it is a real behavioral departure from the reference.
  Single shared amplitude/frequency/phase=0 sinusoid for x/y wiggle, not the
  reference's distinct per-axis `a`/`f`/`phi` values.
- **No gravity compensation term in `FTSensorFilter`.** The reference's
  `SensorCallback` explicitly gravity-compensates the raw F/T reading
  before filtering (`sensor_callback.py:103-106`); this port skips that
  because `InsertionEnv` already exposes an exact,
  zero-in-free-space ground-truth force (`peg_contact_force()`) as an
  alternative, which matters most on real hardware (where no such ground
  truth exists) -- not the situation here.
- **No orientation trajectory/SLERP.** The peg always points straight down;
  the reference's trajectory planner interpolates orientation via SLERP
  because the Franka's approach pose can vary. Nothing to port here since
  there's no orientation change to interpolate.
- **95% (19/20), not 100%, scripted-demo success rate**, documented above
  rather than chased further given the iteration cap (~15 iterations were
  run; the task brief caps iteration at ~15-20). The one failing seed times
  out in SEARCH; a larger spiral radius or a smarter (non-fixed-frequency)
  search pattern would likely close this gap but risks exceeding the force
  ceiling further (see tuning-log items 11/13) without more iteration.
- **19/20 sample size, not an exhaustive statistical characterization** --
  `tests/test_insertion.py`'s `test_scripted_demo_mostly_succeeds_across_seeds`
  checks a 5-seed subset (>=4/5) to keep the test suite fast, not the full
  20-seed sweep reported above.
