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

## Investigation: rounding the socket's inner-top edge (not shipped)

A follow-up pass targeted one specific, confirmed remaining discontinuity:
each of `insertion_hole.xml`'s 4 wall boxes has a real 90-degree corner
where its inner vertical face meets its horizontal top "picture frame"
face, and a peg sweeping across that corner sees the contact normal flip
discontinuously between "top face" and "inner face" in a single 1ms step.
Reproduction (sweep the peg across the +x wall's inner-top edge at a fixed
shallow penetration, `peg_softness=1.0`, `--peg-softness-max-solref 0.06
2.0`, `--peg-softness-max-solimp-width 0.006`, `noslip_iterations=15`,
`tool_kp=1200`):

| variant | max force | max single-step jump | 10-seed scripted-demo success |
|---|---|---|---|
| baseline (unmodified box corner) | 12.39 N | 3.40 N | 10/10 |

**Contact-parameter knobs, confirmed to have zero effect on the jump**
(consistent with this being a discontinuity in *which geom/normal is
nearest*, not a softness/convergence problem a continuous-signal knob can
reach): widening the peg's `solimp` width up to 0.05, sweeping `condim`
3/4/6, sweeping `noslip_iterations` 0/5/10/15/25/40, and sweeping
`model.opt.iterations` (50-300) and `model.opt.tolerance` (1e-8 to 1e-12)
all left `max_force`/`max_jump` unchanged to 3+ significant figures.

**Geometry variants tried, in order:**

1. *Small capsule (r=0.0008m) inset tangent to both faces, box unchanged.*
   No measurable effect on the jump (stayed 3.401N) - the box's own sharp
   corner is still an independent, still-discontinuous contact; adding a
   second, smoother contact alongside it doesn't remove the first one.
2. *Same capsule, but placed protruding exactly at the corner point instead
   of inset.* Peak force dropped (12.3N to 7.25N) but the worst single-step
   jump got **worse** (3.4N to 5.1N, now an abrupt 0-to-5.07N step) -
   contact turns on later but harder. Net: not a win.
3. *Split each wall into an outer bulk box (pulled back by the fillet
   radius `r`) + a thin recessed inner "strip" box (top lowered by `r`) +
   a capsule tangent to both the inner face and the top face, so the box's
   own corner is structurally removed and the capsule becomes the only
   nearby surface.* This is the geometrically "correct" fix and it does
   measurably shrink the jump (e.g. `r=0.001m`: 3.40N to 2.97N; `r=0.008m`:
   3.40N to 2.19N, with diminishing returns beyond `r~0.003`) - **but every
   variant of it that used the fixture's normal `margin=0.001` collapsed
   the 10-seed scripted-demo success rate to 0-1/10** (`insert_timeout`,
   not `broken`: the peg's descent measurably slows and stalls just short
   of the depth target). Root-caused to `margin` creating simultaneous,
   overlapping peg-vs-strip and peg-vs-capsule (and, in variants where the
   strip ran the wall's full height, peg-vs-outer) contact registrations
   near the boundary, effectively summing extra normal/friction force right
   at the region the SEARCH/INSERT transition has to cross. This reproduced
   identically whether the strip spanned the wall's full depth or was
   confined to a small band near the top, and regardless of `r` - i.e. it
   is `margin`, not the split's geometry or `r`, that breaks success.
   Setting `margin="0"` on just the strip/capsule geoms (leaving the outer
   box at the default 1mm margin) restored 10/10 success and kept a modest
   jump improvement (`r=0.002m`: 3.40N to 3.09N) with all 22 existing tests
   green - **but** it also produced a large, seed-dependent side effect in
   the full scripted demo that wasn't root-caused in the time available:
   several seeds (e.g. seed 0, seed 7) that show real double-digit-Newton
   contact forces at baseline instead register **zero contact force for
   the entire episode** with this change, i.e. the peg appears to glide
   straight into the hole without ever registering a fixture contact for
   those particular random search trajectories. That is too large and too
   unexplained a behavior change to ship without understanding it, so this
   variant is **not** merged - `insertion_hole.xml` and `insertion_teleop.py`
   are unchanged from before this investigation.
4. *Isolated check: does an inset, non-protruding capsule alone (no box
   change at all) break success the same way?* Yes - even with the box
   completely untouched, just adding one inert-looking capsule at the
   corner with the fixture's normal margin drops the scripted-demo success
   rate to 0/4. This was not caught by the original (jump-only) evaluation
   of that variant, and is worth remembering: *any* additional geom placed
   at this specific edge, even one intended to be a no-op improvement, can
   silently break the SEARCH-to-INSERT transition unless its `margin` is
   also addressed.

**Corner geometry** (the 4 points where two walls meet) was not attempted
given the above - fixing the straight-edge case first is a prerequisite,
and it isn't fixed yet.

**Net result (this pass):** the inner-top-edge discontinuity is real, well
characterized, and demonstrably NOT fixable by any contact-parameter or
solver-setting knob; a geometrically correct fix (box split + tangent
fillet) exists and reduces the jump by ~10-35% depending on radius, but
only survives the 10-seed success check with `margin=0` on the new geoms,
which then introduces an unresolved, seed-dependent "silent contact skip"
side effect. Given the ~25-30 iteration budget for this pass, the change
was reverted rather than shipped with an unexplained regression risk; the
next person picking this up should start from variant 3's `margin=0`
version (reproducible via the `r`/`strip_margin` sweep described above) and
root-cause the zero-contact seeds before merging it.

### Follow-up: the zero-contact mystery, root-caused (still not shipped)

A later session reconstructed variant 3's `margin=0` geometry from scratch
(no leftover stash/code survived; the `insertion_hole.xml`/`insertion_teleop.py`
in git were byte-identical to before the investigation, confirming the prior
session's revert was clean) and root-caused the zero-contact seeds. **It is
NOT the `hole_geom_ids`-omits-new-geoms reporting bug it looked like at
first glance** (that was the first hypothesis tested and it's a real,
separate footgun worth knowing about -- `InsertionEnv.hole_geom_ids` is a
hardcoded list of geom *names*, `peg_contact_force()` silently reports zero
for any contact against a geom not in that list, so adding new fixture
geoms without updating it would silently drop their contribution to every
force reading. But instrumenting `env.data.ncon`/`env.data.contact` directly
(bypassing `hole_geom_ids` entirely) showed `ncon` was genuinely **0 for the
entire episode**, not just mis-attributed -- a real physics difference, not
a reporting one).

The real cause: variant 3's "recessed inner strip" box, as specified (`u in
[0, r]`, `w` from the wall's full bottom up to `z_top - r`), spans **the
wall's entire ~65mm height minus a 2mm top band** -- i.e. it isn't a small
detail confined to the corner, it *is* almost the entire inner surface of
the wall the peg slides against throughout SEARCH/INSERT. Instrumenting
`contact.dist` directly during a baseline run showed why this matters:
baseline's sustained "in contact" reading throughout INSERT (`ncon=4`,
~11N, for thousands of consecutive steps) has `dist` around **+0.002m** --
i.e. the peg is NOT geometrically touching anything; MuJoCo's per-pair
contact `margin` (empirically the *sum* of both geoms' margins here, peg
0.001 + wall 0.001 = 0.002, matching the observed threshold) creates a real,
solver-applied restoring force *before* actual penetration, as an
intentional soft/anticipatory contact feature -- not a bug. At this
fixture's genuinely tight 2mm radial clearance, that 2mm combined margin
reaches essentially the entire clearance annulus, so baseline is *almost
always* in this soft/anticipatory contact throughout descent, regardless of
how small the landing offset is. Variant 3's strip -- covering nearly the
whole depth -- sets its own margin to 0 for nearly the whole depth, cutting
the reachable margin from 2mm to 1mm (just the peg's own). For any seed
whose landing+wiggle trajectory stays farther than 1mm from every wall for
its whole episode -- which turns out to be common, since the landing
perturbation (3.5mm std) is small relative to the 12mm opening half-width
and most seeds' search never drifts the peg that close to a wall -- contact
never triggers at all: concretely, **6 of 10 seeds never come within 1mm of
any wall for their whole episode**, so
`ncon` truly stays 0 and the peg just glides cleanly through -- the "10/10
success, several seeds silently zero-force" combination observed before is
exactly what falls out of that, not a sign error or geometric gap in the
fillet math (the fillet's tangent-point arithmetic itself was re-derived
independently this session and checked flush/gap-free by direct
coordinate computation).

A restructured 3-piece split was tried to fix this: keep the wall's
**original, full-depth box at its original margin (0.001) unchanged** for
everything except a small top `r`-tall band, and confine the
fillet treatment (a smaller recessed "corner box" + tangent capsule, both
`margin=0`) to only that band -- so the vast majority of the tunnel's
depth keeps baseline's exact soft-cushion behavior, and only the sharp
90-degree corner detail itself gets replaced. This **did** fix the
zero-contact anomaly: at `r=0.002m`, all 10/10 seeds succeed AND all 10
show nonzero, baseline-comparable contact force (e.g. seed 0: 15.42N peak
vs. baseline's 15.37N, seed 3: 31.16N vs. baseline's own high-offset seeds
in the 15-42N documented range) -- no silent zero-force episodes.

**However, re-measuring the actual jump this was all for (using a clean,
controller-free kinematic reproduction -- a standalone 2-geom MuJoCo model
[free peg capsule + one wall], directly setting `qpos` and calling
`mj_forward` at each swept x position with no arm/controller in the loop at
all, to isolate the pure geometric/solver discontinuity from any
settling-transient artifact a closed-loop controller sweep introduces --
several controller-based sweep methodologies were tried first and discarded
for exactly this contamination) showed the restructured split does NOT
reduce the worst-case single-step jump: baseline's sharp corner
gives **0.35N** at its corner (`x=0.012`), but the fillet variant gives
**1.21N** at the *seam between the new fillet capsule and corner-box
primitives* (`x=0.014`, i.e. NOT the original corner location) -- a real,
larger discontinuity introduced by the split itself, not a controller
artifact (confirmed with zero controller dynamics involved). Splitting a
single primitive into multiple primitives replaces one discontinuity
(sharp box corner) with at least one *new* seam discontinuity (where two
independently-computed nearest-contact primitives hand off), and there's no
guarantee that seam is smoother than the corner it replaced -- here it
measurably wasn't, this specific tangent placement made the worst case
*larger*.

**Net result (updated): not shipped, and not recommended without further
work.** The zero-contact anomaly is now fully understood (not a bug, not
shippable-with-caveats -- it was masking a genuinely different fix needed),
and a corrected split exists that resolves it while keeping 10/10 success.
But the core motivating goal (reduce the worst-case single-step contact
force jump at this edge) is not actually achieved by any box-split+fillet
variant tried across both sessions once measured without a
settling-dynamics-blurred methodology -- the jump is relocated, not
reduced, and by this specific tangent construction it gets worse at the new
seam. A genuinely smoother fix would need either a single smooth primitive
spanning the whole corner+adjacent-face region (no internal seam at all --
not expressible with MuJoCo's box/capsule primitives for a 3D box corner
without a custom mesh) or a principled way to blend contact response across
the seam (e.g. matching solimp curvature at the tangent point, not just
position) -- both larger efforts than this task's cap allows. No files
changed; `insertion_hole.xml`/`insertion_teleop.py` remain exactly as
before both investigation sessions.

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

## Data collection & live monitoring

Same on-disk dataset shape and CLI conventions as flipup/sanding, via
`insertion_recorder.py`'s `InsertionEpisodeRecorder`:

```bash
python teleop_insertion.py \
    --collect-dataset ~/data/insertion_v1.zarr \
    --tool-kp 1200 --cartesian-damping-scale 1.0 \
    --peg-softness 1.0 --peg-softness-max-solref 0.06 2.0 --peg-softness-max-solimp-width 0.006 \
    --noslip-iterations 15 \
    --max-speed 0.1 --force-tau 6 --force-rate 80 \
    --scale 1.0 1.0 1.0 \
    --enable-rotation --rot-scale 0.3 --max-rot-speed 10 --max-rot-lead-deg 5 \
    --max-lead-m 0.010 --max-force 5 --stiffness 2000 \
    --auto-finish
```

`--scale` at the shipped default (0.5, 0.5, 1.0) maps the device's
comfortable reach to only +-22.5mm x / +-20mm y around the hole -- *less*
than the fixture's own 27mm half-width, so the device physically can't
reach the fixture's corners. `--scale 1.0 1.0 1.0` doubles that to
+-45mm/+-40mm, comfortably covering it. `--rot-scale` trades the other
way: lower it (e.g. 0.3) to make the SAME wrist rotation move the peg's
angle less, if `--enable-rotation` feels too twitchy at 1.0.

### Finding what settings a past session/episode used

Two things get saved automatically, both readable with
`show_insertion_commands.py <dataset>`:

- **Every KEPT episode** carries the full args it was recorded with, in its
  own `metadata_json` attr (`InsertionEpisodeRecorder.start_episode`'s
  `metadata={"command_line": vars(args)}`) -- so two episodes in the same
  dataset can show different settings if you restarted with different
  flags mid-session.
- **Every session that used `--collect-dataset`** appends one line to a
  sibling `<dataset>.sessions.jsonl` log at startup (`_log_session_command`
  in `teleop_insertion.py`), regardless of whether any episode from that
  session ever got kept -- closes the gap the per-episode metadata has
  when everything gets discarded (or the script is closed before starting
  an episode at all).

```bash
python show_insertion_commands.py ~/data/insertion_v2.zarr
```

**Note on `--tool-kp`/`--peg-softness-max-solref`**: the first version of
this command shipped here used `--tool-kp 2500` and the shipped
`peg_softness=1` ceiling `(0.020, 1.8)`. Real teleop with a human operator
(rather than the scripted demo's clean, precisely-aimed single-axis
approach) exposed a genuinely stiff/elastic "always pushes back" feel on
contact with the fixture's flat top/frame -- pushing sideways into the
frame is not a scenario the scripted-demo tuning ever exercised. Measured
(synthetic sustained-push test, 3cm virtual penetration into the frame):
105.6N mean at the compiled default, 76.2N at the old `peg_softness=1`
ceiling, 43.7N at `--tool-kp 1200` + the widened `(0.06, 2.0)` ceiling.
`--tool-kp` is the bigger lever of the two (it directly multiplies
commanded position error into force); `--cartesian-damping-scale` has
essentially no effect on this *sustained*-push number (it only damps
transients, not steady-state error force) so raising it alone won't fix
this symptom.

- `--collect-dataset PATH.zarr` turns on recording; `S` (keyboard) or a
  short press of the handle button starts/stops an episode, `K`/`D` keep or
  delete it, matching flipup/sanding exactly. `--dataset-hz`,
  `--dataset-image-size`, `--dataset-no-rgb`, `--dataset-min-samples` all
  mean the same thing they do for sanding.
- **Live force plot**: on by default (drawn into the same cv2 HUD window as
  a scrolling strip chart, not a separate matplotlib window -- see
  `draw_plot`/`plot_lock` in `teleop_insertion.py`). `--plot-span` sets the
  trailing time window (default 6s), `--plot-fixed-scale` pins the y-axis
  to `[0, --break-force]` instead of autoscaling (steadier to watch during
  a long session), `--no-plot` disables it.
- `--noslip-iterations N` (default 0, try 10-25): MuJoCo's post-pass for
  refining the friction-force split across simultaneous contacts, ported
  from `FLIPUP_LOW_STIFFNESS_CONTROLLER.md` #6 -- the peg touching 2+
  socket walls at once during CONTACT/SEARCH is the same multi-contact
  friction-allocation-noise situation diagnosed there. Not validated
  against real recorded insertion episodes yet (that doc's version was
  validated against real flipup data); exposed here on the same reasoning,
  worth trying if force readings show high-frequency noise while
  `env.data.ncon` stays >1 through a contact window.

### Known environment issue, now fixed: SIGSEGV at exit under GLFW

`teleop_insertion.py`'s `render_loop` (mirroring sanding/flipup) builds and
uses a `MovableCamera` from a background thread so the live HUD/dataset RGB
capture don't block the control loop. dm_control's GLFW backend is
documented as main-thread-only
(`dm_control/_render/glfw_renderer.py`: "GLFWContext always uses
PassthroughRenderExecutor rather than offloading rendering calls to a
separate thread because GLFW can only be safely used from the main
thread"). In practice this reproduced as a real SIGSEGV at process exit
whenever the render thread had touched GL -- confirmed identical in
`teleop_sanding.py` too (not specific to this task), and independent of
`--collect-dataset`/`--no-view` (reproduces with either on or off, e.g. a
plain `--dry-run --auto-finish` run). `teleop_insertion.py` now sets
`os.environ.setdefault("MUJOCO_GL", "egl")` before any mujoco/dm_control
import, which is thread-safe for this exact pattern and was verified to
exit cleanly (code 0, no core dump) under the same test that segfaulted
under GLFW. `setdefault` leaves an already-exported `MUJOCO_GL` (e.g.
`osmesa` on a box with no GPU at all) alone. This is a repo-wide latent
issue, not fixed in `teleop_sanding.py`/`teleop_flipup.py` themselves.

## Safety: lead clamp (fixes unbounded jam force)

Root-caused from a real recorded episode (`~/data/insertion_v2.zarr`,
episode_1): the peg got tilted (~8 degrees, via `--peg-tilt-randomization-deg`/
`--enable-rotation`) and physically wedged in the fixture's tight 2mm-clearance
opening. While the operator was correctly trying to pull AWAY (`device_pos`/
`ts_pose_command_0`'s z target moving outward), the ACTUAL tool position stayed
frozen -- and `normal_force_n` climbed from 2.0N to 48.5N in 233ms anyway
(`force_break_n`'s default is 45N), because nothing capped how far the
commanded target could drift from the real, stuck tool position: `--max-speed`
only bounds the target's *speed*, not its *distance* from the actual peg. A
genuinely wedged peg (self-locking friction, the same class of failure as
`flipup_teleop.py`'s documented book-wedging problem) can't be un-stuck by
pulling harder, and the target just kept out-running the stuck actual position,
so the position error -- and therefore commanded force -- grew without any
built-in ceiling.

Fix: `--max-lead-m` (default 0.010m) and `--max-rot-lead-deg` (default 15
degrees), applied every control step in `teleop_insertion.py`'s main loop
(not scripted-demo-only), **gated on `env.data.ncon > 0` (actually in
contact)**: clamp the commanded target's translation/rotation to never sit
more than this far from the peg's ACTUAL current pose, in any direction, but
only while touching something. This generalizes `insertion_scripted_demo.py`'s
existing `max_lead_m=0.006` (which only applied along z, INSERT-phase-only)
to the whole live teleop path, all phases, all axes. Bounds worst-case
sustained force to roughly `tool_kp * max_lead_m` by construction, regardless
of how long the jam persists -- measured (synthetic sustained-wall-push
test):

| tool_kp | max_lead_m | peak | mean |
|---|---|---|---|
| 1200 | 0 (unclamped) | 43.7N | 42.3N |
| 1200 | 0.010 (gated) | **10.1N** | **8.7N** |
| 2500 | 0 (unclamped) | 78.8N | 76.2N |

**The gate matters, and its absence was a real regression I shipped and then
had to fix**: an earlier version applied the clamp unconditionally (contact or
not). At a low `--tool-kp`, the controller's own closed-loop settling is
slower than `max_lead_m` -- so during any ordinary free-space move bigger than
that (e.g. the initial reach toward the hole), the clamp stayed permanently
saturated, chasing the actual position like a carrot that's always exactly
`max_lead_m` ahead and never lets the error close. That's not just a
slowdown, it's its own oscillation source: measured, it turned a smooth,
never-drops-out **light** single-wall contact (force std 0.21N, `ncon` never
0) into one that **lost contact entirely 41% of the time** (force std
1.33N) -- i.e. exactly "the device bouncing all over, uncontrollable, even
with light contact". Gating on actual contact fixes both: free-space motion
is completely unaffected (back to the pre-clamp behavior), and the jam-force
cap above still holds, since a genuine jam is by definition in contact.

`0` disables the clamp entirely (reverts to fully unclamped). Note it bounds
*force*, not the underlying jam itself -- a hard clamp doesn't make a
genuinely wedged peg become unstuck, it just guarantees pulling on it harder
never ramps the force past a known, small ceiling instead of toward
`force_break_n`. If wedging itself (not just the force spike) is the problem,
reduce `--peg-tilt-randomization-deg`/turn off `--enable-rotation`, since the
fixture's 2mm clearance was only ever validated for a straight peg.

## Safety: rotation rate limit + anchored rotation clamp (fixes device bouncing while inserted)

Root-caused from the user's own recorded episode (`~/data/insertion_v2.zarr`,
episode_2, command: `--tool-kp 1200 --enable-rotation --rot-scale 1.0
--peg-tilt-randomization-deg 0 --noslip-iterations 15 ...`): "the haptic
device is bouncing all over the place with major displacements... even with
light contact." First test: is this sim or haptic? `haptic_force_sent`
correctly tracked `normal_force_n` (ruling out a signal-routing bug --
confirming the earlier gravity-comp fix still held) and
`device_force_measured` stayed properly capped at `--max-force` -- so the
device hardware itself was doing exactly what it was told. The bug was
sim/control-side: `normal_force_n` climbed smoothly 2N->40N over 313ms
during light single-wall contact, well above what the translation lead clamp
alone should have allowed at that `lead_norm` (~9.5mm, right at its cap).

Reproduced synthetically (single-wall contact + a wrist rotation, mirroring
the recorded episode's target_rotvec growing several degrees over the same
window): a rotation reaching the (old) 15-degree cap within ~1 control step
peaked at 166-177N, because **`--max-rot-lead-deg` bounded the STEADY-STATE
angle error, not the RATE of approach** -- the same class of gap
`--max-lead-m` had for translation before it was gated on contact. But
rate-limiting rotation alone (mirroring `--max-speed`) was NOT sufficient
either, and this is the more interesting part: even a *slow*, properly
rate-limited rotation still peaked at 112N before eventually decaying to a
safe steady value, because **continuing to twist a peg wedged against one
wall has no natural force ceiling the way pushing straight into a wall
does** (pushing harder into a flat wall reaches a stable equilibrium: more
penetration, proportionally more restoring force; twisting further just
keeps ratcheting the wedge tighter). Capping the rotation error relative to
the peg's *continuously-updated* actual orientation doesn't help, because
the arm has enough torque to keep dragging "actual" along -- the error
relative to it never grows past the cap even as the ABSOLUTE rotation keeps
climbing.

Fix, both required together:
- **`--max-rot-speed`** (default 10 degrees/second, deliberately much
  tighter than `teleop_flipup.py`'s 60 -- this fixture's 2mm clearance has
  far less room): slew-rate-limits `target_rotvec` the same way `--max-speed`
  already did for translation.
- **`--max-rot-lead-deg`** (default 5 degrees): now caps the angle from an
  **anchor frozen the moment contact begins** (`env.data.ncon` transitions
  0->1+), not from the continuously-updated actual orientation. Cleared
  whenever contact is lost, so the next contact gets a fresh anchor.

Measured (synthetic single-wall-contact test, both fixes together): 10
degrees/second + 5-degree cap -> **27.9N peak**, decaying to ~2N steady --
comfortably under `force_break_n=45`. Both default to values that pair
well together; loosening either one independently reproduces a large
transient (e.g. 60 deg/s + 15deg cap alone -> 111.9N peak).

## Orientation control & tilt randomization

Two independent knobs, both off by default (peg always exactly
straight-down, byte-identical to the original behavior):

- **`--enable-rotation`**: the omega's wrist (omega.6/.7 only -- raises at
  device-open on an omega.3 with no wrist) drives the peg's roll/pitch/yaw
  live, on top of whatever this episode's `home_rotvec` is. Open-loop (no
  torque feedback; the wrist is passive), same convention and
  `map_wrist_orientation` mapping as `teleop_flipup.py`'s
  `--enable-rotation` (duplicated into `teleop_insertion.py` rather than
  imported, same "stays standalone" convention as `build_pos_map`/
  `DEFAULT_AXES`). `--rot-scale` (lower = harder to move the peg's angle
  for the same wrist motion; try 0.3 if it feels too easy to tilt),
  `--rot-frame` (`world`/`tool`), `--rot-deadzone`, `--rot-axes` all mean
  what they do there.

  **The rotation mapping is ABSOLUTE, not relative to wherever the wrist
  happened to be at reset** (`ROT_HOME_FIXED = np.eye(3)` in
  `teleop_insertion.py`, deliberately not re-captured per episode the way
  an earlier version did). An earlier, relative version captured "wrist
  home" lazily from whatever pose the wrist was actually in on the first
  sample after each reset -- so if the wrist wasn't physically level at
  that instant, that arbitrary tilt silently became the new zero, and the
  peg could visually sit at `home_rotvec` (straight down) while the
  operator's wrist was tilted 20 degrees with no correction. With the
  fixed identity reference, "wrist level" and "peg straight down" are the
  same pose in every episode -- but that means the operator now needs to
  actually hold the wrist level at the start of each episode for the peg
  to sit at `home_rotvec`; `start_episode()` prints a reminder to do so
  when `--enable-rotation` is set.
- **`--peg-tilt-randomization-deg DEG`**: on each `reset()`, samples an
  independent roll and pitch (about the peg's own axes, each uniform in
  `[-DEG, DEG]`) and composes it onto `NOMINAL_HOME_ROTVEC`
  (peg-straight-down) to get that episode's `InsertionEnv.home_rotvec` --
  the orientation `target_pose7`/the scripted demo/live rotation control
  (as the base wrist-rotation-command is layered onto) all fall back to
  when no explicit `target_rotvec` is given. `0` (default) makes the
  sampled range collapse to a single point, so `home_rotvec` is always
  exactly `NOMINAL_HOME_ROTVEC` -- a true no-op, not an approximation of
  one (verified: `InsertionEnv(peg_tilt_randomization_deg=0.0)` after
  `reset()` is bit-identical to the pre-existing hardcoded
  straight-down rotation, and all 22 tests still pass with the default).
  No yaw term: the peg is an axisymmetric capsule, so yaw alone doesn't
  change the effective contact geometry.

  **Caveat**: `insertion_scripted_demo.py`'s phase state machine was not
  adapted for a tilted peg (its min-jerk descent + lateral spiral search
  assume straight-down) -- a nonzero `--peg-tilt-randomization-deg` measurably
  hurts *scripted*-demo success (a quick check at 8 degrees failed on
  `insert_timeout` where the same seed succeeds at 0). This knob is aimed at
  teleop data collection, where a human operator can see and correct for
  the tilt; it isn't meant to make the scripted demo itself tilt-robust.

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
