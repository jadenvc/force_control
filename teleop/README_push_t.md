# omega → 2D Push-T teleoperation

Haptic teleoperation of a 2D push-T task: a circular pusher, driven by the
Force Dimension omega handle in the horizontal plane, shoves a T-shaped block
toward a fixed goal pose. No gripper, no arm, no attached asset tree -- just
a table, the T, and the pusher, built directly in `push_t_teleop.py`.

```bash
conda activate teleop
cd teleop
python teleop_push_t.py --dry-run --no-view --dry-run-seconds 5   # no device: scripted demo
python teleop_push_t.py                                           # real omega handle
python teleop_push_t.py --table-friction 0.05 --pusher-friction 0.8   # coasts after a push
python teleop_push_t.py --table-friction 1.5  --pusher-friction 0.8   # stops almost instantly
```

A long button press on the handle resets the episode (randomizing the start
layout too if `--randomize-start` was passed).

## Physics

- **T-block <-> table friction** (`--table-friction`) is a real Coulomb
  friction coefficient on the T's bottom contact, not a special-cased
  "sliding" vs. "sticking" switch. The T has its own small out-of-plane
  compliance DOF (`t_slide_z`, not part of the public pose) purely so its
  weight actually loads that contact -- without it, MuJoCo reports exactly
  zero normal force (nothing pushes it down) and therefore exactly zero
  friction regardless of the coefficient. It's tuned to settle in a few
  milliseconds and is otherwise invisible.
- **Pusher <-> T friction** (`--pusher-friction`) governs the side-contact
  interface. The pusher is a cylinder tall enough to touch the T's side
  faces (bar or stem, from any direction), not just push down from above,
  so it can shove the T along the contact normal AND drag it tangentially
  when nudged against a face at an angle -- exactly the classic 2D push-T
  interaction.
- The two friction interfaces are independently tunable via MuJoCo's contact
  `priority`: the T's own geoms carry the lowest priority, so a T-vs-table
  contact resolves to the table geom's friction and a T-vs-pusher contact
  resolves to the pusher geom's friction, with no cross-coupling.
- **`--table-torsional-friction`/`--table-rolling-friction` are currently
  no-ops.** All geoms compile at `condim=3` (normal + 2 sliding-friction
  dimensions only); MuJoCo needs `condim>=4` for torsional friction and
  `condim=6` for rolling friction to be solved at all. Verified empirically:
  identical rotation regardless of the value passed. They're exposed for when
  `condim` is raised on the table geom, not because they do anything today --
  don't reach for them to tune how easily the T rotates.
- **To rotate the T more with less force, push off-center, not through the
  middle.** Torque is force times moment arm from the T's center of mass, so
  a push through the T's centroid only translates it (no matter how hard);
  a push near an edge of the bar (or the far side of the stem) with the same
  force produces real angular acceleration. Verified: a single slow, steady
  push (0.03 m/s, ~9 cm of travel, `--pusher-kp 800`) at the bar's edge (12mm
  off centerline, bar half-width is 15mm) rotated the T 24 degrees --
  `--pusher-kp`/speed did not need to be high. A fast/hard push through the
  center will translate the T further before any meaningful rotation and is
  also the more likely source of the bounce/jitter discussed above -- an
  off-center, unhurried push gets you rotation more reliably, not less.
- The T-block has real mass/inertia (`--t-mass`, plus `--t-bar-length` etc.
  for its geometry); MuJoCo derives its rotational inertia from the compiled
  box geometry and density, not a hand-tuned constant.
- **`--pusher-softness`** (`[0, 1]`, default 0) softens the pusher<->T
  contact solver response, same `solref`/`solimp` convention as flip-up's
  `--tip-softness`: 0 reproduces the compiled default (fairly stiff, 6 ms
  time constant), 1 widens/slows the correction. Raising `--pusher-kp` well
  above the 400 N/m default (e.g. for stronger haptic feedback) can make the
  single-point cylinder-vs-box contact chatter/spike, since the controller
  is now driving harder than the default contact solver settles -- if forces
  look jittery or spike far above what a push should produce, raise
  `--pusher-softness` before lowering `--pusher-kp` back down. Measured on a
  head-on push at `--pusher-kp 2500`: softness 0 peaks around 190 N with
  visible tick-to-tick jitter; softness 1 peaks around 6 N and is smooth.

## Success metric

Success is the fraction of the FIXED GOAL T's footprint currently covered by
the T-block (`--success-threshold`, default 0.95, matching the original
push-T task). Both T footprints are the union of two axis-aligned-in-local-
frame rectangles (bar + stem), so this is computed by rasterizing them on a
fine grid and comparing boolean masks -- exact for this shape, fully
vectorized in numpy, and needs no extra polygon-boolean-ops dependency
(e.g. shapely).

## Haptics

Reuses the same `fd_omega.FDOmega` bridge as the other tasks here: the
device's background thread applies the tau/rate-limited reflected-force
filter internally, so the main loop just sets the raw target force each
tick. `--stiffness` sets the requested handle stiffness in N/m; `--force-gain`
is derived from it (`stiffness / (pusher_kp * scale)`) unless given directly.

## Device axes and scale

By default `--scale` maps the omega's full comfortable workspace
(`DEVICE_WORKSPACE_HALF_M`, the same hardware-calibrated figures
`teleop_sanding.py` uses: 45/40/48 mm half-range in x/y/z) onto the full
`--workspace-half` square (default 0.30 m), so the operator can reach every
edge of the push-T workspace without leaving a safe range on the device.
This replaced an earlier flat `--scale 3 3 3` default that left the edges
unreachable. If you raise `--workspace-half`, raise `--scale` by the same
ratio (or just recompute it: `workspace_half / DEVICE_WORKSPACE_HALF_M`).

By default the handle's left/right motion drives sim x and its up/down
motion drives sim y (`--axes y,z`), leaving the device's toward/away axis
unused. Override with `--axes` (e.g. `--axes x,y` for the device's horizontal
plane, `-y,z` to reverse x) -- see `build_pos_map_2d` in `teleop_push_t.py`.
It's a signed *selection* of 2 of the device's 3 axes (not a full 3-axis
permutation like flip-up's `--axes`, since only 2 sim axes exist here), and
its transpose maps reflected force back onto the device consistently.

## Randomized start

`--randomize-start` samples the T-block and pusher start pose from the same
70%-center-biased mixture flip-up/cube-lift use (`--start-center-prob`,
default 0.70): most episodes land near the center of each axis's range
(`--start-t-xy-half`, `--start-t-theta-half-deg`, `--start-pusher-xy-half`),
occasionally sampling uniformly over the full range instead. Unlike flip-up,
the very first episode is NOT forced to the exact center: here the
randomized pose is the T-block itself, and its center normally coincides
with the default goal pose, so forcing it would start episode 1 already at
~success.

The T-block and pusher are drawn independently and can overlap by chance
(more likely than it sounds, since both distributions are center-biased
toward the same region) -- `randomize_start` resamples (up to
`max_resamples=64`) whenever a draw starts in contact, checked via a real
`mj_forward` + contact scan rather than a geometric estimate, so it's exact
for whatever T/pusher sizes you've configured. If every draw is unlucky it
falls back to placing the pusher just outside the T's bounding radius.

## Dataset collection

`--collect-dataset PATH` enables full-rate (`--dataset-hz`, default 1000 Hz)
recording via `push_t_recorder.PushTEpisodeRecorder`, an on-disk-compatible
sibling of `pyrite_recorder.PyriteEpisodeRecorder` (atomic Zarr write, same
`start_episode`/`record_sample`/`commit`/`discard` shape) with a schema
scoped to push-T's actual state (pusher/T pose and velocity, contact force,
coverage, device state) instead of forcing the gripper/wrench-at-tool-origin
shape flip-up/cube-lift use. **Not included yet:** the top-down camera image
described below is rendered live for the viewer but is NOT captured into the
dataset (no RGB array in the schema), and the adaptive-compliance
virtual-target labels (`stiffness_0`) are also omitted -- that scheme is
specific to flip-up/cube-lift's impedance-controlled tool and isn't
meaningful for push-T's rigid, ungripped contact.

Workflow, same states as flip-up (`idle` -> `recording` -> `review`):
- Click START in the viewer (or long-press the handle) to start recording,
  STOP (or long-press again) to stop -- or pass `--auto-finish` to stop
  automatically the instant coverage crosses `--success-threshold`; you
  still choose KEEP/DELETE afterward.
- In `review`, click KEEP or DELETE in the viewer, or short-press (keep) /
  long-press (delete) the handle.
- `--dry-run` runs headless: it starts recording immediately (no button to
  press) and auto-keeps the episode when the run ends, since there's nobody
  present to choose.
- Episodes shorter than `--dataset-min-samples` (default 20) are discarded
  automatically even if you choose KEEP.

### Why the force looks jittery, what actually helps, and what doesn't

The root cause: MuJoCo's contact solver adds/drops individual contact points
between the round pusher and the T's corners tick to tick (`contact_count`
measured cycling like 0→13→5→5→0→8→13→5 across consecutive 1 ms steps in a
real recording), so the force can swing or flip sign within a single step
even while the commanded target (`ts_pose_command_0`/`ts_pose_fb_0`) is smooth
to sub-millimeter precision. This is a genuine solver artifact, not a device
tremor or a recording bug.

What measurably helps (implemented):
- **`--force-sensor-cutoff 30`** -- a causal two-pole low-pass F/T sensor
  model (`PushTTeleop.sensor_force_xy`, same as flip-up's) applied to the
  *recorded* `wrench_0` only; `wrench_ground_truth_0` always keeps the raw
  signal. Verified: cut tick-to-tick jitter ~19x on a real recording.
- **`--plot-smoothing-hz 8`** (on by default) -- a separate, lighter
  single-pole filter applied ONLY to the live plot trace, so the display
  looks clean regardless of whether you've set `--force-sensor-cutoff`. Does
  not touch `wrench_0`, `wrench_ground_truth_0`, or haptic feedback at all --
  it's purely cosmetic. Set to `0` to plot the raw value.
- Haptic feedback always uses raw contact, unfiltered, on purpose (same as
  flip-up's default) -- adding sensor delay inside the bilateral force loop
  costs passivity margin.

What's implemented but, measured honestly, did NOT meaningfully reduce
jitter in either a straight push or a corner-grazing test (kept because
they're legitimate MuJoCo levers worth having exposed, not because they
solved this):
- **`--noslip-iterations`** (default 2) -- extra PGS passes refining the
  friction-cone/slip-direction solution after the main solve.
- **`--pusher-joint-damping`** (default 1.0 N·s/m) -- real passive damping on
  the pusher's slide joints (distinct from the controller's task-space kd,
  which was already nonzero via `--damping-ratio`).

Both target different failure modes (friction-slip convergence, underdamped
DOF oscillation) than what's actually happening here (a discrete change in
*how many* contact points are active, not their slip direction or DOF
velocity) -- which is exactly why filtering the *signal* (`--force-sensor-
cutoff`, `--plot-smoothing-hz`) works and tuning the *solver* didn't, in this
case.

## The viewer

Unless `--no-view`, a single OpenCV window (not the native MuJoCo viewer,
and not a separate popup) stacks three things top to bottom:
- a live top-down camera render of the scene (fixed viewpoint -- there's no
  camera orbiting to lose, and a top-down view is all a planar task needs);
- a force strip chart (`|F|` black, `Fx` blue, `Fy` orange) over the last
  `--plot-span` seconds, unless `--no-plot`;
- if a recorder is active, a status line and the START/STOP/KEEP/DELETE
  buttons, drawn directly in this window and clickable via a mouse callback
  on it (`k`/`d`/`s` keyboard shortcuts also work if the window has focus,
  but the buttons are the primary interface). Buttons visibly gray out when
  they don't apply to the current state (e.g. KEEP/DELETE before an episode
  is stopped).

Also drawn on the camera view unless `--no-com-marker`: a magenta cross at
the T's actual center of mass (offset from the bar's visual centerline
toward the stem -- not obvious by eye), a line to the pusher, and the
current offset distance in mm. Rotating the T needs an off-center push
(torque = force x lever arm from the CoM); a push through the CoM only
translates it no matter how hard. This exists because aiming, not physics,
turned out to be the actual bottleneck for rotating the T -- offset size,
pusher radius, T mass, and friction were all tested and barely change how
much rotation a given off-center push produces (16-29 degrees across a
5x range of each), but there was previously no way to see where the CoM
even was.

## Files

- `push_t_teleop.py` -- the environment (`PushTTeleop`, `PushTProperties`).
- `teleop_push_t.py` -- the haptic CLI driver (`--dry-run` for hardware-free testing).
- `push_t_recorder.py` -- the dataset recorder (`PushTEpisodeRecorder`).
- `tests/test_push_t.py` -- friction/coverage/stability checks.
