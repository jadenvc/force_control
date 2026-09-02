# omega/UR5e sanding teleoperation

Haptic teleoperation of a full-arm sanding task: a UR5e holds a compliant
sander pad and must sand an entire target panel to a desired force, with
strong reflected force feedback and a live color gradient showing which
regions are under-sanded, just right, or over-sanded. Unlike the floating-
gripper FlipUp variant, this reuses the full-arm task-space controller so
teleop feel matches the pivot task: soft/compliant contact at the pad, a
stiff and accurate arm underneath it.

```bash
conda activate teleop
cd teleop
python teleop_sanding.py --dry-run --no-view --dry-run-seconds 20   # no device: scripted demo
python teleop_sanding.py                                            # real omega handle
python teleop_sanding.py --pad-softness 0.9                         # softer pad, mushier onset
python teleop_sanding.py --sand-force-target 8 --sand-break-force 20  # lower-force task
```

## Physics

- The arm, controller, and gravity compensation are the same Jacobian-
  transpose task-space impedance controller as the full-arm FlipUp task
  (`flipup_minimal/flipup/environment.py`'s `step_task_space`, ported into
  `SandingEnv` since this end effector has no gripper actuator to skip). The
  arm itself stays stiff and accurate (`--tool-kp`, default 16000 N/m,
  matching FlipUp); compliance comes from the pad contact, not from
  softening the arm.
- **Sander end effector** (`flipup_minimal/flipup/assets/sander/sander.xml`)
  is built entirely from rounded primitives on purpose: a capsule housing
  and an oblate-ellipsoid pad, no flat-faced cylinders. A flat face meeting
  a round side is a real geometric corner, and MuJoCo's contact normal is
  discontinuous across it -- exactly the kind of feature that produces
  torque spikes/chatter during teleop. The ellipsoid pad has no rim at all.
- **`--pad-softness`** (`[0, 1]`, default `1.0`) interpolates contact
  `solref`/`solimp` between rigid and soft endpoints, ported from
  `floating_flipup_teleop.py`'s `_configure_tip_contact`/`tip_softness`.
  Two things worth knowing:
  - It's written onto `panel_surface`, not the pad. MuJoCo takes a
    contact's `solref`/`solimp` (and `friction`, see below) entirely from
    whichever of the two geoms has the higher `priority` -- the panel's
    (10) beats the pad's (8) -- so writing only to the pad (an earlier bug
    here) silently did nothing at all, the same way `--friction` was
    broken before it got the same fix.
  - The soft endpoint isn't just "settles more gradually" -- softening
    `solref`'s time constant measurably lowers the contact's real
    steady-state stiffness too (it's not merely a slower-but-equal
    spring), which is *why* it fixes fast-sliding oscillation: the
    contact's own natural frequency needs to sit well below the arm's
    closed-loop frequency to avoid the two resonating (trading energy back
    and forth) instead of decoupling. Measured directly: at the old
    `(0.025, 2.0)` endpoint, a fast sanding sweep dropped to literal zero
    contact force on ~27% of steps; `(0.060, 2.0)` (the new default)
    eliminates that entirely in the same test. Going further (`(0.150,
    3.0)`, tried and rejected) removes the oscillation completely but
    softens steady-state stiffness by ~25x instead of ~2.6x, needing
    impractically deep penetration to reach ordinary sanding forces --
    `(0.060, 2.0)` is the point that gets most of the stability win
    without wrecking the force calibration.
- **Sanding-dose grid**: the panel is rasterized into a fine grid (default
  1 cm cells, `--grid-resolution`). Each step, the pad's contact centroid
  selects which cells are under the pad footprint (radius `pad_radius_m`),
  and those cells accumulate `dose += dt * k_dose * (force - --sand-force-min)`
  above `--sand-force-min`, saturating at `--sand-force-cap`. This is a
  genuine force/time trade-off, not just a force threshold: `--sand-force-min`
  and `--dose-target-time` are calibrated together so that, e.g., 10N for
  ~1s and 5N for ~10s both reach dose 1.0 -- less force still sands the
  same amount, it just needs proportionally longer dwell. Dose only ever
  grows (no "un-sanding"), so a cell that's crossed into over-sanded can
  never fall back into the just-right band -- this is what makes each spot
  a one-shot attempt without needing separate bookkeeping.
- **Target regions**: `--num-regions` (5-10, default 7) small circular
  patches (`--region-radius`, default 2.5cm) actually count toward coverage/
  success -- the rest of the panel can still physically be sanded, it just
  isn't part of the task. Centers are evenly spaced along the panel's long
  axis with a small random jitter (`--seed` controls it), reading as one
  continuous stroke to sand rather than scattered unrelated spots.
- **Visual gradient**: a coarser grid of small flat, non-colliding geoms
  (`--vis-cell`, default 2 cm) tiles the panel, colored live from the dose
  grid via a piecewise-linear ramp. Non-target cells always render as plain
  panel wood; target cells go amber (untouched, "needs sanding") -> blue
  (in progress) -> green (just right) -> red (over). Same `geom_rgba`
  mutation trick already used once-per-episode for the FlipUp book's
  color, just applied per-cell and refreshed every view tick.
- **Break condition**: normal contact force, passed through a short
  low-pass filter (`break_force_tau_s`, default 15 ms) so genuine few-ms
  impact transients on first touch don't false-trigger it, is compared
  against `--sand-break-force` (default 30 N, 2.5x the target force) with a
  small debounce. Once tripped, `broken` is sticky for the rest of the
  episode -- pulling back out of contact does not clear it. This mirrors
  the finding in `FLOATING_FLIPUP_COMPLIANCE_TELEOP.md` that raw
  instantaneous contact force needs filtering before it's a meaningful
  signal for anything beyond an impact spike.

## Success metric

`success()` is `not broken and coverage_fraction("just_right") >= threshold`
(`--success-threshold`, default 0.90) -- the fraction of the TARGET REGIONS
(not the whole panel) whose accumulated dose sits in `[dose_low, dose_high]`
(defaults 0.7/1.3). "One try": the episode ends at the first `success()` or
`broken` (no continued sanding after either fires); per-cell, the monotonic
dose accumulator already makes an over-sanded spot permanently unrecoverable
without any extra state.

## Haptics

Reuses the same `fd_omega.FDOmega` bridge as every other task here.
`--stiffness` sets the requested handle stiffness in N/m; `--force-gain` is
derived from it (`stiffness / (tool_kp * scale[2])`, the PRESS axis, not
`scale[0]` -- sanding's `--scale` is deliberately anisotropic, unlike
FlipUp/push_t's) unless given directly. A passivity-margin estimate is
printed at startup, same formula as FlipUp's. There is no gripper on this
end effector, so the omega's single-button short-press signal (normally
used to toggle a gripper) is repurposed to start/stop a recording episode;
long-press resets to a new (possibly randomized, see below) start position.

**Arming (matches `teleop_flipup.py` exactly, not a device-homing wait)**:
the arm starts tracking the handle immediately from the first tick,
relative to `--home` -- motion is never gated on the physical device
reaching any specific position. If the device's actual rest position
doesn't match `--home` (e.g. a failed/skipped auto-home -- watch for
`drdMoveToPos failed` at startup), that's just a larger-than-usual initial
offset; `--max-speed`'s existing slew limit is what turns that into a
gradual, visible catch-up instead of an instant jump, exactly the same way
FlipUp already relies on it. What's actually gated is *haptic force
feedback*: it stays silent until the sim tool has tracked to within
`--arm-tolerance` (default 2cm, same default FlipUp hardcodes) of its own
commanded target, so a reset/catch-up motion's transient contact never
gets mistaken for real interaction force. `--takeover-ramp-ms` (default
400ms) then ramps that force 0 -> full once armed, rather than snapping it
on.

An earlier version of this made the arm wait up to several seconds for
the *physical device* to reach `--home` before moving at all, with a
timeout fallback if it didn't. That turned out to be the wrong fix for the
wrong problem: it made every run wait for something (`drdMoveToPos`
succeeding) that a genuinely broken/obstructed auto-home will never do,
and a hand still moving when the timeout fired could arm mid-gesture
anyway. FlipUp doesn't have or need any of that -- it just lets the arm
move (slew-limited) from t=0 and only delays *feeling* it.

If `drdMoveToPos` fails on every run, that's worth chasing at the hardware
level -- it's a robotic-move command failing for reasons unrelated to
calibration (workspace obstruction, a snagged cable, a torque/safety
limit, etc.), not something this script can or should work around further.
Try `bin/HapticInit` (or your SDK's equivalent) standalone to see if the
same move fails outside this script too, and check the physical workspace
is actually clear.

## HUD

Same kind of live interface as `teleop_flipup.py`, scoped down to what this
task needs (one force signal, not FlipUp's raw/sensor/xyz breakdown):
- A cv2 window with a force-over-time strip chart (`--plot-span` seconds of
  history, reference lines at `--sand-force-min/-target/-cap/-break-force`)
  plus text HUD: current coverage%, current force, **max force this
  episode**, and a SUCCESS/BROKEN flag. `--no-plot` hides just the strip
  chart; `--no-view` disables the window entirely (rendering still runs if
  `--collect-dataset` wants RGB frames).
- The console `\r`-refreshed readout (`--no-readout` to disable) is
  unchanged and runs independently of the cv2 window, same as FlipUp.
- Episode keep/delete: `K`/Enter to keep, `D`/`X`/Backspace/Delete to
  discard, clickable KEEP/DELETE buttons during review, `S` to start/stop
  recording, `R` to reset, `Q`/Esc to quit -- all in addition to the handle
  button (short-press start/stop, long-press reset).
- Needs a real display (same `cv2`/Qt requirement as FlipUp) -- use
  `--no-view` for headless testing.

## Reset positions

Each new episode's start hover position is randomized within
`--start-prism` (full size in metres, default `0.06 0.06 0.03`) centered on
the nominal panel-center hover, with probability `--start-center-prob`
(default 0.5) of starting at the exact nominal point instead. Disable with
`--no-episode-randomization` for a fixed, identical start every time. The
sampled height is geometrically clamped to a minimum safe clearance above
the panel -- simpler than FlipUp's settle-and-reject sampling, which exists
there to guard against a randomized book/bookend layout; sanding's start is
just a hover point above a static panel, so no equivalent risk exists.

## Camera

`--cam-azimuth`/`--cam-elevation`/`--cam-distance`/`--cam-lookat` (all
optional overrides) default to a side-on view (`azimuth=90, elevation=-15`,
distance `0.75`, mostly horizontal), framing the whole arm and panel
together as closely as possible without clipping either (`0.65` starts
cropping the UR5e's base out of frame).

A **wrist camera** (`ur5e/sander/wrist_cam` in `sander.xml`, mounted above
the pad and tilted ~45deg off straight-down) shows as a picture-in-picture
inset in the top-right of the HUD window -- `--no-wrist-cam` to hide it,
`--wrist-cam-width`/`--wrist-cam-height` (default 160x120) to resize it.
Its position/tilt account for the pad's default orientation (a 180deg
flip about x, pointing straight down), which flips the body's local y/z
axes relative to world -- worth knowing if you ever retune it directly in
the XML.

## Dataset collection

Unlike Push-T, BC dataset recording is wired in from the start:
`--collect-dataset PATH.zarr` uses a new, sanding-specific
`SandingEpisodeRecorder` (`sanding_recorder.py`), schema name
`pyrite_sanding_sim` -- a sibling of `pyrite_recorder.py`, not an edit to it
in place, since `PyriteEpisodeRecorder.record_sample`/`commit` hardcode
FlipUp fields (`book_angle_deg`, a required `final_book_angle_deg` kwarg,
...) directly in their bodies rather than accepting a generic per-step
field dict. The sanding recorder reuses `pyrite_recorder.py`'s generic
storage engine (`_NumericSampleBuffer`, the same growable-array/zarr-chunk
logic) and records, per step: pose, contact wrench, `coverage_under`/
`coverage_just_right`/`coverage_over`, `dose_mean`/`dose_max`, `broken`,
`success`. The full grid of dose cells is **not** recorded per step (at
1 kHz that would be hundreds of MB/episode uncompressed) -- a BC policy
would condition on the compact coverage/dose summary above, not the raw
grid. RGB frames come from the same async render thread that drives the
cv2 HUD (`--dataset-no-rgb` to skip).

In `--dry-run`, a recording episode starts automatically (there's no
device button to press); with real hardware or the cv2 window, use the
handle button or the `S`/`K`/`D` keys.

## Files

- `sanding_teleop.py` -- the environment (`SandingEnv`, `SandingTeleop`,
  `SandingProperties`).
- `teleop_sanding.py` -- the haptic CLI driver (`--dry-run` for
  hardware-free testing).
- `sanding_recorder.py` -- BC dataset recording (`SandingEpisodeRecorder`).
- `flipup_minimal/flipup/assets/sander/sander.xml` -- the sander end
  effector.
- `flipup_minimal/flipup/assets/custom/sanding_panel/sanding_panel.xml` --
  the workpiece.
- `tests/test_sanding.py` -- physics/dose-rate/break/coverage/recorder
  checks.

## Not yet implemented

- The panel's world pose (`PANEL_TRANSFORM` in `sanding_teleop.py`) is a
  fixed constant -- only the start hover position is randomized per
  episode (see Reset positions above), not the panel/robot layout itself.
- A `--dataset-dose-grid` opt-in flag for occasional (e.g. once/second, not
  per-step), heavily-decimated full-grid snapshots, if the raw per-cell
  history is ever needed for offline visualization/debugging beyond the
  per-step summary fields.
- Unlike FlipUp's settle-and-reject start-pose sampler, sanding's reset
  safety is a simple geometric clearance clamp (see Reset positions) --
  fine for a hover point above a static panel, but would need revisiting
  if the panel/workpiece layout is ever randomized too.
