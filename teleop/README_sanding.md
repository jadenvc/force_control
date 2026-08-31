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
- **`--pad-softness`** (`[0, 1]`, default `0.6`) interpolates the pad's
  `solref`/`solimp` between rigid and soft endpoints, ported directly from
  `floating_flipup_teleop.py`'s `_configure_tip_contact`/`tip_softness`.
  Softer settles contact more gradually but, per that same task's findings,
  does not by itself cap sustained force -- the controller will keep
  pushing until its commanded penetration is reached regardless.
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

**Home gate**: on real hardware, the arm will not move at all until the
physical handle is within `--home-tolerance` (default 1cm) of `--home`.
Without this, a failed or skipped auto-home (watch for `drdMoveToPos
failed` at startup) leaves the device's actual rest position mismatched
from the assumed `--home` reference -- the very first tick would then
command a large, unintended offset, potentially straight into the panel,
before the operator has touched anything. `[device] waiting for handle at
home: ...mm away` prints periodically until you move the handle there by
hand. `drdMoveToPos` failing is a hardware-level robotic-move issue, not a
calibration problem -- `--auto-init` won't fix it, and it gives no way to
know where the declared `--home` physically is. So after
`--home-timeout` seconds (default 8s, `0` to wait forever) with the handle
never reaching tolerance, the gate gives up and adopts wherever the handle
currently is as the working home reference instead of blocking forever.

**Takeover hold/ramp**: right after arming (either way -- reached home, or
the timeout fallback) and after every reset, target speed is capped well
below `--max-speed` for `--takeover-hold-ms` (default 150ms) and reflected
handle force ramps 0 -> full over `--takeover-ramp-ms` (default 400ms).
Without this, the instant control engages it snaps straight to full
speed/force -- if there's any mismatch between the (possibly just-adopted)
home reference and where the operator's hand actually is at that moment,
that mismatch gets applied at full authority on the very first tick instead
of being walked into gradually. Same idea as
`FLOATING_FLIPUP_COMPLIANCE_TELEOP.md`'s "100ms takeover hold and 400ms
force ramp" to avoid a reset/start haptic impulse.

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
