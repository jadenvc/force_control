# FlipUp Full-Arm Controller: Low-Stiffness / Anti-Chatter Work

Summary of the controller and contact-tuning work done on the full-arm FlipUp
pivot task (`flipup_teleop.py`/`teleop_flipup.py`) to get a genuinely soft,
low-force feel without the jam-force chatter that stiff `tool_kp` used to
require. Companion to `SANDING_JITTER_FIX_SUMMARY.md` (same investigation
style, different task).

## Background: why this needed real controller changes, not just tuning

The original module docstring's finding was: "16 kN/m looks unrenderable, but
the flip needs it -- the fingertip pad is an 8mm capsule contacting the book
7.5mm below its top edge, so a few mm of sag loses the edge. Sweeping the
shipped heuristic over `tool_kp`, the flip succeeds at 16000 and 12000 and
fails at every value at or below 8000." That constraint predates the
bookend/table contact-softening work below -- with that softening in place,
real recorded data now shows **`tool_kp=5000` (and even 4000) succeeding
reliably** (9/9 in one dataset). The lesson generalizes: contact softness and
controller stiffness interact, and a constraint measured under one contact
configuration doesn't necessarily hold once the contact itself changes.

## 1. Contact-priority parity fixes (bookend/table)

Same root cause as sanding's `--pad-softness` bug: MuJoCo takes
`solref`/`solimp`/`friction` entirely from whichever geom has the higher
`priority`, not a blend. The bookend surfaces
(`robot_wall_surface`/`robot_pivot_surface`/`robot_floor_surface`) compile at
priority 10-20, above the fingertip pads' priority 8, so `--tip-softness`
alone never reached bookend contact -- only fingertip-vs-book. Added dedicated
flags that write directly onto the higher-priority geoms:

- `--bookend-solref` / `--bookend-solimp` / `--bookend-friction`
- `--table-solref` / `--table-solimp` / `--table-friction` (floating-gripper
  variant)

Working combination found by testing: `--bookend-solref 0.06 2.0
--bookend-solimp 0.85 0.95 0.006 0.5 2.0` -- the same `(0.060, 2.0)` time
constant that worked for sanding's panel contact, reused here for the same
reason (kills sliding/jam oscillation, costs some steady-state stiffness).

## 2. Approach compliance (`_compute_effective_translation_kp`)

`--approach-compliance-distance` / `--approach-compliance-min-kp-ratio` /
`--approach-compliance-max-speed`: ramps `tool_kp` down smoothly only within a
configurable distance of a guarded surface (table/bookend/book), full
stiffness everywhere else. This is what makes homing/settling fast and
reliable even when the *in-contact* stiffness is soft -- the ramp only
engages near the book, not during the initial slew to the start pose.

## 3. `book_normal_force_limit` anti-windup

A Cartesian-deflection cap (`max_deflection = book_force_limit / tool_kp`)
that bounds sustained wedging force against the book without touching the
transient/onset behavior. Off (0.0) by default; the deflection budget scales
inversely with whatever `tool_kp` you're running, so lowering `tool_kp`
automatically loosens this cap for the same force limit.

## 4. New this round: `tool_kp_axes` -- anisotropic per-axis stiffness

`task_space_kp` was a single scalar `tool_kp` applied uniformly to all three
translation axes (`np.diag([tool_kp]*3 + ...)`). Motivated directly by
comparing against `force-insertion-sim`'s controller (a different repo, 7-DOF
Franka FR3 peg-insertion task): its `dynamic_impedance` mode runs
`K_cart=[450,450,700,80,80,200]` -- anisotropic, insertion axis stiffest --
20-35x softer overall than our `tool_kp=16000`/`tool_rot_kp=3000`.

`tool_kp_axes=(1.0, 1.0, 1.0)` (default, byte-identical to prior behavior) is
now a per-WORLD-axis multiplier on `tool_kp`, applied only to
`task_space_kp`'s translational diagonal. Lets you keep one axis stiff
(precision-critical) while softening another (implicated in chatter) instead
of an all-or-nothing scalar. CLI: `--tool-kp-axes X Y Z`.

Caveat, stated plainly in the code: every OTHER formula that uses `tool_kp`
(`surface_force_limit`/`book_normal_force_limit` deflection caps, the
tool-force-limit tanh squash) still uses the plain scalar -- an approximation
once this is anisotropic, not a fully-general per-axis correction. Acceptable
because those are fail-safes, not the primary control law.

## 5. New this round: `tool_cartesian_kd` -- the missing Cartesian damping term

`step_task_space()`'s law is `task_wrench = Kp@twist - Kd*tool_velocity` --
literally the same impedance law force-insertion-sim uses. But
`task_space_cartesian_kd` had **zero damping on every translation axis**;
all translational damping came from a fixed joint-space term, independent of
`tool_kp`. force-insertion-sim pairs its low `K_cart` with `D_cart=55-200` on
every axis, translation included.

`tool_cartesian_kd=(0.0, 0.0, 0.0)` (default, unchanged behavior) now
populates the translational part of `task_space_cartesian_kd` directly. CLI:
`--tool-cartesian-kd X Y Z`. Verified stable (finite state, small settle
error) up to `kd=600` at `tool_kp=5000` in a smoke test; live-tested by the
user at `--tool-kp 5000 --tool-cartesian-kd 200 200 200` with good results
(see the validated command below).

## 6. New this round: `noslip_iterations` -- multi-contact friction-force noise

Diagnosed from real recorded data (`flipup_arm_1khz_v12.zarr`): even after the
lighter-book + Cartesian-damping wins, the force reading still showed
oscillation. `contact_count` never dropped to 0 during any contact window
(ruling out the dropout/re-impact chatter mechanism found in the sanding
investigation) -- but 51-88% of the (detrended) force signal's power sat above
100Hz. That combination (no dropouts, near-Nyquist noise, `contact_count`
sitting at 3-5 throughout) points at **friction-force allocation noise across
multiple simultaneous contacts** (fingertip pads + 1-2 bookend surfaces at
once): MuJoCo's main solver converges the *total* constraint problem, but the
*split* of friction force across several near-redundant simultaneous contacts
isn't uniquely determined and can wobble step to step even while the sum is
right.

The model compiled with `noslip_iterations=0` (MuJoCo's dedicated post-pass
for refining exactly this friction split, off by default). Exposed as
`--noslip-iterations` (try 10-25). Two attempts at an offline replay harness
to validate this in isolation both failed to reproduce the real episode's
force magnitudes faithfully (4-5x too high), so this was shipped as a
CLI-exposed option to test live rather than validated numerically in this doc
-- see the validated command below for the result.

## 7. Two non-controller findings from this round

- **Lighter book helps, confirmed on real data.** `--book-mass` (down from
  the 1.375kg default, valid range [0.4, 2.0]) reduces the contact force
  needed to pivot the book at a given angular acceleration, which directly
  reduces how much position-tracking error a softer `tool_kp` needs to
  produce that force -- i.e. it reduces the "sag distance" that's the actual
  mechanism behind the original 7.5mm-margin failure. Measured:
  `book_mass=0.2` episodes averaged roughness ~1.75 vs ~6.3 for default-mass
  episodes in the same session, with lower mean/peak force too.
- **Softening `tool_kp` needs a bigger `--settle` budget, not a fundamentally
  different task.** Lowering `tool_kp` without raising `--settle` produces an
  apparent "instant failure on start" that looks like a hard incompatibility
  but isn't: `configure_episode()`/`reset()` slews to the start pose for a
  fixed `--settle` seconds (default 2.5s) then rejects/resamples if residual
  error exceeds `--start-max-settle-error` (default 10mm) -- and resampling
  is useless against a *convergence-speed* problem, so it burns all 32
  retries and raises. Measured settle error @ 2.5s: 0.29mm at `kp=16000`,
  9.44mm at `kp=4000` (borderline), 31.13mm at `kp=2000` (fails). Same
  `kp=2000` converges to 2.77mm given `--settle 8`. It's a slower time
  constant, not a steady-state failure -- raise `--settle` alongside any
  large `tool_kp` reduction.
- (Environment-only, not a code issue): a `teleop`-conda-env conflict between
  `opencv-python` and `opencv-python-headless` (both installed, the headless
  one winning the `cv2` import) broke the live viewer window
  (`cv2.namedWindow: function not implemented`). Fixed by uninstalling
  `opencv-python-headless`. Not specific to this task, but hit while testing
  it, so recorded here.

## Rejected / not-yet-validated

- Two offline jam-replay harnesses (reconstructing a recorded episode's env
  args and re-driving its target trajectory) both produced forces 4-5x higher
  than the real recording -- something about the reconstruction (missing
  params, or raw per-step target-jumps not matching the real slew-limited
  teleop loop) isn't faithful yet. Findings in this doc that came from live
  data (real recorded zarr episodes) are trustworthy; anything attributed to
  these replay attempts is explicitly flagged as unreliable above, not
  reported as a number.

## The validated command

```
python -u teleop/teleop_flipup.py \
    --auto-init \
    --tip-softness 1.0 \
    --bookend-solref 0.06 2.0 --bookend-solimp 0.85 0.95 0.006 0.5 2.0 \
    --bookend-friction 0.06 0.001 0.0001 \
    --pos-tau 8 \
    --stiffness 2000 --damping 35 --max-force 20 \
    --tool-kp 5000 --tool-cartesian-kd 200 200 200 \
    --scale 4 4 4 --max-speed 0.1 --force-tau 6 \
    --collect-dataset ~/data/flipup_arm_1khz_v12.zarr \
    --auto-finish --arm-view full --book-mass 0.2 --noslip-iterations 15
```

Reported by the user as feeling good after the OpenCV viewer fix. Real
recorded episodes at this general configuration (same `tool_kp`/`bookend_*`,
book mass jitter varying) show reliable success and materially lower
force/roughness than the pre-fix baseline.
