# omega → FlipUp (pivot a book upright) teleoperation

Haptic teleoperation of the `flipup_minimal` task: a UR5e with a **closed** WSG50
noses under a flat book's edge and levers it 90° upright against a bookend. It
reuses the omega bridge and the whole haptic pipeline from `teleop_ball.py`, so
the feel, the flags and the failure modes carry over.

```bash
conda activate teleop
cd teleop
python teleop_flipup.py                 # seed 0
python teleop_flipup.py --seed 1        # a scene the shipped scripted flip also solves
python teleop_flipup.py --dry-run --no-view      # no device: walks the scripted arc
python teleop_floating_flipup.py         # WSG50 rigid body, no UR5e joints
python teleop_floating_flipup.py --dry-run --no-view
python teleop_floating_cube_lift.py      # analogue grasp + rounded-cube lift
```

Keys in the cv2 window: **`r`** reset, **`q`/ESC** quit. On the omega.6 a long
button press also resets. Dataset collection adds explicit save/discard controls
described below. The gripper stays closed throughout — this is a nonprehensile
pivot, so there is nothing to grasp and no grip feedback.

Files:

- `flipup_teleop.py` — `FlipUpTeleop`, a `FlipUpEnv` subclass adding a Cartesian
  position interface, the three force sources, the book-angle success metric and
  a threaded-safe camera factory. Also `flipup_scene(seed)`, which re-derives the
  scene and the flip arc from `flipup.heuristic`'s conventions.
- `teleop_flipup.py` — the launcher: device, mapping, haptics, viewer, recording.
- `floating_flipup_teleop.py` — the dynamical floating-WSG50 environment: direct
  Cartesian wrench control, physical gripper mass/inertia, and gravity compensation.
- `teleop_floating_flipup.py` — the floating-controller launcher; it shares the
  collection, haptic, viewer, and command-line interface with the arm launcher.
- `pyrite_recorder.py` — exact-rate Pyrite Zarr writer, adaptive-compliance
  label generation, and schema validation.
- `replay_pyrite_flipup.py` — stored-RGB or MuJoCo-state episode replay.
- `../flipup_minimal/` — the scene and controller, including the strict contact
  allowlist and configurable physical properties.

## What you move, and what moves in the sim

You command the **WSG50 tip position**. By default wrist orientation is derived
from the commanded position exactly the way the scripted heuristic derives it
(pitched 30° down, yawed away from the robot base), so the omega's three
translational axes are enough and there is nothing to steer.

**`--enable-rotation` adds the other three DoF**: the omega's wrist then drives the
tool's roll/pitch/yaw, composed on top of the orientation the tool starts in.

- `--rot-frame world` (default) means turning the handle turns the tool about
  **world** axes (spatial delta, pre-multiplied); `tool` turns it about the tool's
  own axes (body delta, post-multiplied). The two are kept paired because mixing
  them — a body-frame delta composed extrinsically — is exactly the bug
  `teleop_ball.py` documents, and it makes the axes come out wrong.
- `--rot-scale` amplifies the angle and keeps the axis. `--rot-axes` takes the same
  signed-permutation syntax as `--axes` and defaults to it, for when the wrist
  gimbal reports pitch/yaw swapped independently of position. `--rot-deadzone`
  (0.005 rad, about 0.3°) is a smooth radial dead area around the home wrist
  pose—absolute, not per-poll, because the mapping itself is absolute.
- **There is no torque feedback.** The omega.6/.7 wrist is passive and
  `fd_omega.py` commands zero wrist torque, so orientation is open-loop while
  translation is force-reflected.
- The device wrapper verifies `dhdHasWrist()` and reads a valid wrist frame
  synchronously before the control loop starts. The viewer and terminal readout
  show `wrist delta`; if that number changes, the SDK input is reaching the
  controller.
- With rotation on the nominal orientation is **frozen** at the start pose rather
  than tracking the target position. Not worth worrying about: the derived yaw only
  moves ~2° across the whole arc, and measured side by side the flip lands
  identically either way (final angle 1.2°, position lag 3.8 mm, seeds 0/1/2/4).
- Rotational impedance defaults to **`--tool-rot-kp 3000` N·m/rad** and
  **`--tool-rot-kd 90` N·m·s/rad**. The damping is derived from the measured
  home-pose effective rotational inertia and removes the overshoot seen with an
  abrupt command. The previous 4000/0 controller drove the 28 N·m wrist actuators
  into saturation.
- **`--max-rot-speed 60` °/s still matters.** With the new gains, a swept 10°
  command about each world axis had no overshoot or actuator saturation and
  settled within 0.5° in at most 0.48 s after the sweep. An uncapped step can
  still briefly saturate a wrist actuator.

Verified end to end against a stub device that scripts a handle trajectory: a +20°
handle yaw about device z produces exactly +20° about world z on the tool (axis
[0,0,1]), tracked to 0.00°.

Default mapping is `--axes -x,-y,z`, the same as `teleop_ball.py`:

| handle motion | tool motion | what it does |
| --- | --- | --- |
| **away from you** (device −x) | sim +x | drives the tip into the book/bookend |
| **up** (device +z) | sim +z | levers the book over |
| left/right (device ±y) | sim ∓y | across the book's width; barely used |

The flip arc spans **~15 cm of push and ~14 cm of lift**, which at the default
`--scale 4 4 4` asks for ~3.8 cm of handle travel in each. `--home 0.02 0 -0.02`
puts the handle's rest position forward and low so that travel does not run into
the workspace stops. If a direction comes out reversed on the hardware, change
`--axes` (negate an entry to flip a sign, reorder to swap two axes) — fixing
motion direction automatically fixes force direction, because the force is
mapped back through the transpose of the same matrix.

## Simulated arm stability

The simulated arm defaults to **16,000 N/m Cartesian translational stiffness**
and **2.5× the original joint damping**: 160 N·m·s/rad on the first three joints
and 40 N·m·s/rad on the wrist joints. A free-space step and scripted-contact
sweep showed that 2.5× removes overshoot and reduces contact dropouts versus
2.0×, while adding about 0.9 mm of mean tracking lag.

Keep `--tool-kp 16000` for this task. The narrow fingertip contact needs that
stiffness to stay on the book edge. If the motion feels too sluggish, reduce only
the damping to `--arm-damping 2.0`; if the arm still rings after a fast input,
reduce `--max-speed` before increasing damping beyond 4×:

```bash
# Stable default
python teleop_flipup.py --tool-kp 16000 --arm-damping 2.5

# Lower-lag alternative
python teleop_flipup.py --tool-kp 16000 --arm-damping 2.0

# Conservative target slew for abrupt hand motion
python teleop_flipup.py --tool-kp 16000 --arm-damping 2.5 --max-speed 0.15

# Rotation tuning used by the default profile
python teleop_flipup.py --enable-rotation \
  --tool-rot-kp 3000 --tool-rot-kd 90 --max-rot-speed 60
```

`--stiffness` and `--damping` without the `arm-` prefix tune the **haptic
handle**, not the simulated robot controller.

## Behavior-cloning and adaptive-compliance data

Install the recorder dependency once in the `teleop` environment. PyriteML uses
the Zarr 2.x API, so the upper bound is intentional:

```bash
python -m pip install -r requirements_dataset.txt
```

Collect demonstrations directly into the dataset PyriteML will train from:

```bash
export PYRITE_DATASET_FOLDERS="$HOME/data/real_processed"
python teleop_flipup.py \
  --collect-dataset "$PYRITE_DATASET_FOLDERS/flipup_sim_1khz.zarr"
```

State, command, wrench, contact truth, controller state and complete MuJoCo state
contain one sample per **1 ms simulation tick (1000 Hz)** by default. RGB is
stored as 224×224 HWC `uint8` only when the asynchronous renderer produces a new
frame, with its own timestamps; the recorder does not duplicate the same image
33 times between 30 Hz frames. Use the timestamps to select any later training
rate. `--control-freq` must be an integer multiple of `--dataset-hz` if a lower
raw state rate is explicitly chosen.

The timing paths are deliberately separate: every physics tick publishes a new
sim-force target, the Force Dimension thread filters/rate-limits and sends that
target on its own 1 kHz schedule, numeric recording uses preallocated arrays,
and RGB rendering plus plot/window composition run in lower-priority background
threads at `--view-fps`. Thus a slow visual frame no longer freezes force
feedback. The timestamps in the numeric stream remain exactly 1 ms apart even
if software rendering makes simulation run slightly slower than wall time;
`wall_time_ns`, `deadline_lateness_ms`, and the control-batch fields retain that
wall-time behavior for auditing.

Collection controls:

- The collector starts **idle at the sampled initial tool pose**. The tool is
  held there and cannot drift while the operator moves the handle. In idle, a
  damped 100 N/m haptic spring gently pulls the physical handle toward fixed
  `--home`, independently capped at 2 N. Relax your grip and let it settle; the
  overlay turns green after it is within 3 mm and slower than 15 mm/s for 250 ms.
  Only then will **`s`** start. The centering spring switches off before data
  collection, so it is not part of the demonstrated motion or recorded contact.
  Every randomized simulated start therefore maps to the same physical center
  instead of rebasing to wherever the handle happened to be.
- Starting holds the simulated tool for 100 ms, clears all old reflected-force
  filter state, and smoothly engages haptic feedback over the next 400 ms.
  Tune these gates with `--collection-home-*`, `--collection-takeover-hold-ms`,
  and `--collection-force-ramp-ms`. Tune or disable the pull with
  `--collection-recenter-stiffness`, setting it to zero to disable, and cap it
  separately with `--collection-recenter-max-force`.
- Press **`s`** again to stop. The simulation pauses on the final frame.
- Click **KEEP** or **DELETE** in the viewer. Keyboard shortcuts are **`k`** and
  **`d`**. Only KEEP writes the episode into the Zarr dataset.
- After the decision, the next book properties, colour, and start pose are
  sampled; the task resets there and returns to idle.
- **`q`** or ESC quits while idle. If an episode is active, it first moves to
  the KEEP/DELETE review screen. Ctrl-C discards unconfirmed samples.
- A device long press also toggles idle → recording → review.

Automatically stop when the book reaches the task's success angle:

```bash
python teleop_flipup.py \
  --collect-dataset "$PYRITE_DATASET_FOLDERS/flipup_sim_1khz.zarr" \
  --auto-finish
```

`--auto-finish` still waits at the KEEP/DELETE screen; it never silently adds a
demonstration to the training set.

Every episode contains PyriteML's required raw fields:

- `rgb_0`, `ts_pose_fb_0`, `ts_pose_command_0`, and their timestamps.
- `wrench_0`: the sensor observation available to BC in the tool frame,
  `[Fx,Fy,Fz,Tx,Ty,Tz]`. For the floating gripper this is the tunable causal
  sensor model, or exact contact when `--force-sensor-cutoff 0`.
- `ts_pose_virtual_target_0` and `stiffness_0`: adaptive-compliance labels
  generated from a causal F/T filter using Pyrite's virtual-target rule.

Additional channels retain the information needed to audit or replay a demo:

- `ts_pose_controller_0` stores the pose actually sent to the impedance
  controller. `ts_pose_command_0` remains the operator's unsaturated request;
  `surface_limit_active` marks samples where visible-surface anti-windup makes
  those differ.
- `wrench_ground_truth_0`: solver-exact contact wrench transported to the tool
  origin; world-frame ground truth, world-frame sensed wrench, and untared raw
  wrist sensor values are stored separately.
- `wrench_sensor_model_0` and `wrench_sensor_model_world_0`: explicit copies of
  the sensed wrench in tool/world axes. These equal `wrench_0` (after the frame
  transform), while the ground-truth channels always retain the unfiltered
  MuJoCo constraint wrench for analysis and alternate preprocessing.
- Complete MuJoCo integration state plus `qpos`, `qvel`, `qacc`, controls,
  actuator/constraint forces, sensor data, book pose/twist, tool twist, contact
  count, device pose/rotation/forces, mapped command, and haptic force.
- `wall_time_ns`, `deadline_lateness_ms`, `control_batch_size`, and
  `control_batch_index` expose wall-clock stalls/catch-up explicitly. On real
  hardware, `device_servo_sequence`, `device_servo_timestamp_ns`, and
  `device_servo_dt_s` expose the independent Force Dimension servo cadence.
- Episode attributes contain scene seed, sampled book physics, controller gains,
  mappings, camera, haptic settings, model dimensions, termination reason, and
  success.

Validate or replay a dataset:

```bash
python replay_pyrite_flipup.py "$PYRITE_DATASET_FOLDERS/flipup_sim_1khz.zarr" \
  --validate-only

# List saved episode names, duration, success, angle, and termination reason
python replay_pyrite_flipup.py "$PYRITE_DATASET_FOLDERS/flipup_sim_1khz.zarr" \
  --list

# Replay stored observations
python replay_pyrite_flipup.py "$PYRITE_DATASET_FOLDERS/flipup_sim_1khz.zarr" \
  --episode episode_0

# Restore and render every MuJoCo state snapshot
python replay_pyrite_flipup.py "$PYRITE_DATASET_FOLDERS/flipup_sim_1khz.zarr" \
  --mode state
```

The data lives exactly at the path passed to `--collect-dataset`. Inside the
Zarr directory, kept demonstrations are stored as
`data/episode_0`, `data/episode_1`, and so on. Deleted or unconfirmed episodes
are never written.

Two task configurations are installed in PyriteML:

```bash
cd /path/to/PyriteML

# Standard diffusion behavior cloning: 9D pose actions
accelerate launch train.py --config-name=train_dp_workspace \
  task=flipup_sim_bc_20hz

# Adaptive compliance: pose9 + virtual-target pose9 + scalar stiffness = 19D
accelerate launch train.py --config-name=train_spec_workspace \
  task=flipup_sim_adaptive_compliance_20hz
```

Both training configurations expect data subsampled/aligned to 20 Hz; do that
from the stored 1 kHz streams before training. They use two RGB frames, three
pose frames, eight sensed-F/T frames, and a 16-step action horizon. The default
compatibility labels range from
16,000 N/m in light contact to 2,000 N/m under heavy load; tune with
`--ac-k-max`, `--ac-k-min`, `--ac-f-low`, and `--ac-f-high`.

## Contact allowlist

The simulation now permits four deliberately separated contact classes:

- robot/gripper ↔ book;
- book ↔ the physical bookend wall, pivot, and floor;
- every physical robot geom ↔ the complete visible wooden tabletop;
- WSG50 geoms ↔ low-friction collision copies aligned exactly with the visible
  bookend wall, pivot, and floor.

Independent collision bits prevent duplicate book/support contacts and keep
robot self-contact disabled. The old oversized invisible shelf guard is not
used. Fin-ray shafts use smooth ellipsoid envelopes and the distal pads use
8 mm capsules, avoiding edge snagging while covering the visible finger.

After a visible table/bookend contact, `--surface-force-limit 80` caps only the
continued inward spring deflection (5 mm at 16 kN/m). Sliding and pulling away
remain unconstrained, and book contact is not capped. Pass
`--surface-force-limit 0` for an unbounded A/B test.

## Haptics

Absolute position mapping → slew-limited target → sim contact force → gain →
`FDOmega.set_reflected_force`. The independent 1 kHz device servo then applies
the one-pole filter, vector slew-rate limit, damping and magnitude clamp before
sending the force. Damping ramps in with force so free space stays effortless.

### Why the arm stays stiff, and where the force gain comes from

The felt stiffness of any of these setups is

```
k_handle = tool_kp × scale × force_gain          (N/m at the handle)
```

and the loop is only passive while `k_handle < 2 × damping / T_effective`, where
`T_effective = 1/control_freq + 2 × force_tau`. The obvious move — soften the
arm until `tool_kp` is "renderable" — **does not work here**: the fingertip pad
is an 8 mm capsule bearing on the book 7.5 mm below its top edge, so a few mm of
sag loses the edge. Sweeping the shipped heuristic trajectory over `tool_kp`:

| `tool_kp` (N/m) | scripted flip on the seeds it solves | free-space lag |
| --- | --- | --- |
| **16000 (shipped, default)** | works | 3–7 mm |
| 12000 | 1 of 2 | — |
| ≤ 8000 | fails on every seed | 6–10 mm |

What makes the stiff arm renderable anyway is that this task's contact forces are
about **10× BallPush's** (22 N median while levering a 1.375 kg book, versus ~2 N
sliding a light block), so the gain that lands the felt force in the same 1–9 N
band is ~10× smaller, and `k_handle` still comes out in the same few-kN/m band.
For the full arm, `--stiffness` (default **1500 N/m**) derives

```
force_gain = stiffness / (tool_kp × scale) = 1500 / (16000 × 4) = 0.0234
```

**Do not use `--tool-kp` to soften the feel** — it changes the sim's task
dynamics, not the rendering. `--stiffness` is the knob.

### Measured feel at the defaults

From the scripted flip on seed 1 (`--dry-run --record`), sim contact force → what
the handle is commanded:

| phase | sim force | felt at `--stiffness 1500` | `teleop_ball` for comparison |
| --- | --- | --- | --- |
| free space | **0.00 N** | 0.00 N | 0.00 N |
| levering the book up | 30 N median | **0.86 N** median | ~0.8 N sliding the block |
| pressing it against the bookend | 82 N median | **1.50 N** | — |
| first-touch spike | 157 N peak | 2.90 N | ~8.7 N against the wall |

The startup banner prints this prediction for whatever settings you pass, plus the
passivity margin, and the exit line recomputes the margin at the **achieved** loop
rate so a shortfall cannot hide.

Want it firmer? `--stiffness 3000` doubles every number above (still a 4× margin),
at the cost of a harsher touch — see the next section for the measured trade.

### Smoothing the touch (what to change if contact feels bouncy)

Diagnosed 2026-07-30 from a report that the gripper "bounces/oscillates" when it
meets the book. **It is not the simulator.** An open-loop touch with no device in
the loop, at approach speeds from 0.05 to 0.30 m/s, gives:

- **zero contact make/break events** (`separations/s = 0.00` at every speed),
- tool-position ringing of only **23–37 µm**, and what there is sits at 2–4 Hz with
  5–18% band energy, i.e. broadband, not a resonance,
- impact force that *falls* as approach speed rises (128 N at 0.05 m/s → 99 N at
  0.30), which rules out an impulse effect: it is a quasi-static penetration force,
  so `--max-speed` is not the lever here.

What actually makes it feel bad is the **rendered stiffness**, and two measurements
pin it down. First, force sensitivity to hand position: at `--stiffness 3000`,
holding the hand 1 mm deeper changes the felt force by **~2 N**, so the ±0.5 mm a
human hand cannot help moving renders as **1.7 N of peak-to-peak ripple** at tremor
frequency. That ripple *is* the perceived bouncing. Second, the onset: the sim force
goes 0 → 80 N within a few milliseconds of touch, which arrived at the handle as a
**0.98 N/ms** step.

Both are rendering problems, so the fixes are on the rendering side:

| | before | after | change |
| --- | --- | --- | --- |
| onset peak | 2.87 N | 1.29 N | 2.2× lower |
| **max onset slope** | 0.98 N/ms | **0.12 N/ms** | **8.2× lower** |
| mean slew | 25.2 mN/step | 9.9 mN/step | 2.5× lower |
| tremor ripple (±0.5 mm) | 1.69 N | 0.84 N | 2.0× lower |
| levering / pressing | 1.58 / 2.86 N | 0.86 / 1.50 N | 1.8× lower |
| flip still succeeds | yes | yes | — |

1. **`--stiffness` 3000 → 1500** (the value the `teleop_ball` tuning settled on).
   This halves the ripple, the onset and every contact force at once. It is the
   dominant lever, because the ripple is exactly `k_handle × hand tremor`.
2. **`--force-rate 120` N/s**, new. A slew limit on the handle force vector, applied
   after the filter and before the clamp by the 1 kHz device servo using its
   measured elapsed time. It targets the onset step specifically and,
   unlike raising `--force-tau`, does **not** lag the steady force or the ripple —
   and unlike `--force-tau` it does not eat passivity margin. Measured onset slope
   0.35 → 0.12 N/ms at 120 N/s, 0.06 at 60 N/s, with the dwell force unchanged.
   Set 0 to disable.

If it still feels rough on hardware, `--diagnose` is the arbiter: it separates a
haptic limit cycle (your *hand* oscillating → lower `--stiffness`, raise
`--damping`) from sim chatter rendered faithfully. The margin is now **8×**, so a
limit cycle is unlikely; the remaining ripple is your own hand motion being
rendered honestly, and only lower `k_handle` reduces it further (`--stiffness 1000`
→ ~0.56 N, at the cost of a 0.47 N levering force).

**A fix that was tried and rejected:** capping the Cartesian force the task-space
controller may command (`--tool-force-limit`, the apparent analogue of BallPush's
`ball_force_limit`, implemented as a smooth tanh saturation). At
`--arm-damping 2.5` the arm already needs ~56 N just to drag itself through free
space at 10 cm/s, so any limit low enough to soften contact also starves free
motion: at 90 N and below the tool never reached the book and the scripted flip went
4/4 → 0/4. The flag is still there and defaults to 0 (off) for anyone running much
lower arm damping.

### Why it should not bounce

Bouncing on contact in this pipeline is a **haptic limit cycle**, not a sim
artefact — that was the conclusion of the `teleop_ball` tuning, and the fix there
was to keep the *rendered* stiffness well under the passivity limit of a
sampled-data impedance display,

```
k_handle = tool_kp × scale × force_gain   <   2 × damping / (1/control_freq + 2 × force_tau)
```

`teleop_ball`'s settled anti-bounce tuning sits at a **4.0× margin** (1500 N/m
against a 6000 N/m limit at 500 Hz, damping 30, τ 4 ms). This runs at 1000 Hz, so
the same margin is available at twice the stiffness — which is why `--force-tau`
defaults to **2 ms** here rather than 4. Measured on the scripted flip:

| `--force-tau` | limit | margin at 3000 N/m | felt ripple | levering force |
| --- | --- | --- | --- | --- |
| 0 ms | 60000 N/m | 20× | 112 mN/step | 1.20 N |
| **2 ms (default)** | **12000 N/m** | **4.0×** | **39 mN/step** | 1.13 N |
| 4 ms | 6667 N/m | 2.2× | 23 mN/step | 1.17 N |
| 6 ms | 4615 N/m | 1.5× | 17 mN/step | — |

Note the direction: **raising `--force-tau` to cure buzzing makes bouncing more
likely**, because the filter sits inside the feedback loop and its lag counts
roughly twice in the round trip.

Two caveats specific to an arm, both matching what `teleop_ball --arm` says about
`PivotArm`:

- `k_handle` computed from the *drive* stiffness is an **upper bound** on what the
  hand feels. In contact the series compliance is set by the contact and the book,
  not by the task-space controller, so the margin above is conservative.
- The honest knob on an arm is therefore the **force gain** itself (`--force-gain`,
  0.0234 by default here). It comes out ~10× smaller than `PivotArm`'s
  `--arm-force-gain 0.4` purely because this task's contact forces are ~10× that
  scene's — the *felt* newtons are the same.

The sim side is not the problem either: contact separations are only **1.0–1.4/s**
(the ball tuning was chasing 5.3/s), contact duty 83%, and the first-touch impact
peak of 80–97 N maps to 3.7–4.5 N at the handle, never reaching the clamp (0.00% of
steps). If it does buzz on hardware, `--diagnose` distinguishes a limit cycle (your
hand oscillating) from sim chatter rendered faithfully.

### Force sources

`--force-source` picks what gets reflected. All three report *the force the world
applies to the robot*, so a positive reading always opposes the motion that
caused it (validated by signed comparison, not magnitudes — a sign error here
once shipped in `teleop_ball` and felt like a magnet pulling you in).

Measured over ~4500 free-space samples (five directions × three speeds) and one
full scripted flip:

| source | free space | accuracy in contact | smoothness |
| --- | --- | --- | --- |
| **`contact`** (default) | **0.00 N** exactly | ground truth | 2.52 N/step |
| `wrist` | mean 6.8 N, **p90 0.2 N**, spikes to 218 N | cos **+0.99**, mean error 5.4 N | 2.40 N/step |
| `estimated` | **111 N mean** (= 5.2 N at the handle) | cos +0.77, mean error 41 N | 0.05 N/step |

- **`contact`** — the solver's true per-contact forces, summed over every contact
  involving the robot. Exactly zero in free space by construction, which is why it
  wins here even though BallPush preferred the actuator-side estimate.
- **`wrist`** — the WSG50's own MuJoCo force sensor, tared at reset. Nearly as
  good, and it is what a real wrist F/T sensor would read; the rare large values
  are the sensor also seeing the gripper's inertia when the arm accelerates hard.
  `--force-deadband 1` removes those.
- **`estimated`** — BallPush's `-clip(tool_kp × (target − tool)) + tool_damping ×
  tool_vel`. **Broken for an arm this stiff**: it renders the 3–7 mm tracking lag
  times 16 kN/m, so free space feels like 5 N of constant push. It is the
  *smoothest* signal of the three, which is exactly the trap — smooth and wrong.
  `--tool-damping` does not rescue it: least squares over those free-space samples
  returns 14 N/(m/s), i.e. the lag is not velocity-proportional for a 6-DoF arm
  (inertia and the joint-space damping structure dominate), and no value brings the
  residual under ~70 N. It defaults to 0 and the banner warns when you select this
  source. Kept only for A/B comparison.
- **`none`** — no force feedback; the decisive A/B when something feels wrong.

## Viewer

Off-screen osmesa render blitted into a cv2 window (this workstation cannot do
on-screen GL — see the main README). The lower force strip shows handle-scale
forces (green = force sent to the handle, red = sim force × gain). A separate
upper strip shows solver-exact MuJoCo world-frame `Fx/Fy/Fz` in actual simulator
newtons with its own scale and live numeric values. The per-axis panel on the
right retains the device-axis rendered/sent traces and prints the current actual
world force beside the felt device force.

The default camera restores the original **left-oblique** view: `--cam-azimuth -30
--cam-elevation -25 --cam-distance 0.75`, aimed at the middle of the flip arc, with
**`--arm-view hidden`** drawing only the WSG50 and not the UR5e links. Hiding the
arm is what makes that view usable — the forearm otherwise fills 14–39% of the frame
(depending on distance) and covers the gripper as well: hiding it raises the
gripper's own visible area from 13.5k to 22.8k pixels. `--arm-view ghost` draws the
links translucent instead, `full` restores them. It is a visualization change only
(`geom_group`/`geom_rgba` play no part in collision detection), so the physics is
bit-for-bit identical.

**Why an oblique view and not 0°.** At *exactly* azimuth 0 the flip angle is geometrically
invisible: the book's long axis rotates in the world x–z plane, that plane contains
the view direction, so the axis projects to a *vertical line for every tilt and
every elevation* — only its apparent length changes, and not even monotonically (at
elevation −25 the projected length runs 0.91 → 0.57 → 0.09 → 0.42 as the tilt goes
0° → 30° → 60° → 90°). The restored −30° view breaks that degeneracy and matches
the earlier Force Dimension setup.
Measured apparent tilt against a true 35.4°, arm hidden, at the default elevation
and distance:

| azimuth | apparent tilt | book occluded | robot pixels |
| --- | --- | --- | --- |
| 0 | 89.9° — carries no information | 7.4% | 22.8k |
| 10 | 63.2° | 7.1% | 21.9k |
| 15 | 59.2° | 6.9% | 21.3k |
| 20 | 56.4° | 6.7% | 20.5k |
| **−30 (default)** | **52.6°** | 6.0% | 18.5k |

The angle still reads exaggerated in an oblique view, so the viewer prints
**`book NN.N deg from vertical (need < 15)`** in the top-left, turning green on
success. Raise `--cam-azimuth` toward 45–90 to judge the angle geometrically rather
than off that overlay.

The visual and force rates are independent. With the normal visual meshes,
`--render-quality fast --view-fps 30` typically delivers about 25–30 fps; the
precise rate depends on CPU/software-renderer speed. For a responsive 30–60 fps
control display, use `--render-quality collision --view-fps 30` (or 60 on a fast
CPU). That choice also determines the RGB images saved in a dataset. `--no-view`
runs without a window (and still renders dataset RGB unless `--dataset-no-rgb`
is set). The on-screen window is enlarged to twice the native canvas width and
height by default without increasing render or dataset resolution; use
`--viewer-scale 1` to restore the old window size.

## Recording

- `--record run.csv` — 37 columns: episode, `t_episode`, `t`, handle xyz, target
  xyz, **target rotvec xyz** (zero unless `--enable-rotation`), tool xyz, **tool
  quaternion wxyz**, tool velocity xyz, sim contact force xyz, force sent to the
  handle xyz, book xyz, book quaternion wxyz, book angle from vertical, success.
  Orientation is logged unconditionally so one schema covers both 3- and 6-DoF
  runs — with rotation on, the orientation *is* part of the action. Written in a
  `finally` block, so Ctrl-C keeps it.
- `--record-video run.webm` — the whole composited viewer. **Use `.webm`** if you
  will preview it in VS Code or a browser (Chromium builds usually lack H.264);
  `.mp4` gives H.264 with `+faststart` for real players. ffmpeg runs in its own
  session so a terminal Ctrl-C cannot kill it before it writes the index.
- `--diagnose` — FFTs the contact force and the handle position over the longest
  continuous contact and says whether roughness is a **haptic limit cycle** (your
  hand is being shaken → lower `--stiffness`, shorten `--force-tau`, raise
  `--damping`), **sim chatter rendered faithfully**, or slow make/break contact.

## Scene randomization

`--seed` moves the bookend (position and ±10° of yaw) and the book on it.
Every episode independently samples book upright height, width, and mass within
±20% of nominal and chooses a realistic cover colour. Thickness and friction are
held fixed to avoid confounding the requested factors. Tune the ranges with
`--book-size-jitter` and `--book-mass-jitter`, or disable them with
`--no-episode-randomization`.

The tool starts inside a fixed prism centred at the nominal standoff in front of
the book. Its default full depth/lateral/vertical size is 4×6×5 cm. The first
attempt is exactly centred; afterward 70% use a tight centre distribution and
30% uniformly cover the prism, so head-on starts dominate without losing edge
coverage. Change this with `--start-prism D L V` and `--start-center-prob`.
Candidates that start with more than 0.5 N of robot contact or retain more than
10 mm of settling error are automatically rejected and resampled; this keeps
the broad distribution without turning an overlapping or unreachable reset into
a haptic impulse. The thresholds are configurable with `--start-max-contact-force`
and `--start-max-settle-error`.
Each attempt's component, normalized offset, world pose, dimensions, mass,
colour, and deterministic seed are stored in episode metadata.

The older `--randomize-physics` flag still samples a broad nominal book including
friction and thickness; the controlled per-episode ±20% variation is then
applied around it. Explicit `--book-*` values select the nominal values.

## Floating gripper

`teleop_floating_flipup.py` removes the UR5e but keeps the compiled WSG50,
RealSense, book, support, contact parameters, and the gripper's 1.89 kg physical
mass/inertia. It applies Cartesian force and moment directly at the gripper,
cancels gravity body-by-body, and retains real acceleration/contact dynamics.
Because flip-up is nonprehensile, both finger sliders are projected to zero
opening after every physics step; contact cannot back-drive or chatter them.
There is no manipulator Jacobian, arm singularity, joint damping, or arm torque
saturation. The default target speed is 0.6 m/s instead of 0.3 m/s. Rotational
stiffness is 300 N m/rad rather than 3000 because the free gripper has much less
rotational inertia; damping is derived from that inertia. Solver-exact contact
wrench remains the default haptic source and is always retained as dataset
ground truth; the recorder's sensed-wrench field can instead use the causal
sensor model below.

Its translational impedance defaults to **5,000 N/m**, with critically scaled
damping of about **204 N·s/m**, rather than inheriting the arm's 16,000 N/m.
On the same scripted flip this reduced simulated contact from 33.9 N median /
173.8 N peak to **15.3 N median / 68.2 N peak**, while still succeeding. The
floating handle target is **1,800 N/m** (versus 1,500 for the arm), giving a
default force gain of `1800 / (5000 × 4) = 0.09`. In the matched run the felt
contact changed from 0.81 N median / 3.57 N peak to **1.38 N median / 4.98 N
peak**, with a 6.7× passivity margin.

Fingertip compliance is an opt-in A/B knob. `--tip-softness 0` leaves the two
tip-pad contacts unchanged. `--tip-softness 0.5` uses a 15 ms contact time
constant, damping ratio 1.5, and 4 mm impedance width; `1` uses 20 ms, ratio 2,
and 5 mm. Only the two fingertip pads change. The resolved contact parameters
are printed at startup and stored in each dataset episode's model metadata.

Force-sensor bandwidth is a separate opt-in knob:

```bash
python teleop_floating_flipup.py \
  --tip-softness 0.5 --force-sensor-cutoff 30
```

`--force-sensor-cutoff 0` is an identity path and exactly preserves the old raw
measurement. A positive value runs two cascaded causal first-order poles in the
tool frame at every 1 ms physics tick; the value is the cutoff of each pole.
At 30 Hz the modeled step reaches 50% in about 8.9 ms and has about 10.6 ms of
low-frequency group delay. This removes much of the instantaneous
constraint/friction switching while remaining substantially faster than the
250 ms causal moving average used to make the optional adaptive-compliance
labels. The plot draws raw force as a thin muted trace and the sensor as a thick
trace, and both are stored at the full numeric sample rate.

The default `--force-source contact` deliberately keeps haptic reflection on
the immediate raw contact path even when the recorded sensor is enabled. Use
`--force-source wrist` only if you intentionally want to feel the modeled
sensor too; its added delay reduces haptic passivity margin, so lower
`--stiffness` and confirm the result with `--diagnose`.

## Rounded-cube lift

`teleop_floating_cube_lift.py` reuses the floating WSG50, gravity compensation,
1 kHz Force Dimension servo, force-sensor model, viewer, randomized reset gate,
KEEP/DELETE workflow, and Pyrite recorder. The new scene has a rounded 2.75 cm
cube on the complete visible tabletop and maps the omega.7 gap continuously to
the WSG50 finger opening.

```bash
python teleop_floating_cube_lift.py \
  --collect-dataset ~/data/cube_lift_floating_1khz.zarr \
  --auto-finish --render-quality fast
```

The task is set up for fast top-down pickups: the simulated gripper starts 10
cm above the cube centre, the device is homed at `0.02 0 0.015` (raised relative
to flip-up), translation scale is `5 5 5`, target speed is 0.8 m/s, and wrist
orientation stays vertical unless `--enable-rotation` is passed. Success occurs
when the cube bottom clears the table by `--success-height 0.08` m.

At the default scale, the 10 cm descent takes 2 cm of handle travel and the
14 cm lift takes 2.8 cm. The simulated safety workspace spans 40 cm in x/y, but
the physical device walls retain their conservative 9 x 8 x 9.6 cm box. Later
episodes sample the gripper throughout a 10 x 10 x 5 cm start prism and move the
cube within a separate 10 x 10 cm placement region; the existing 70% centred
Gaussian mixture still concentrates most demonstrations near the nominal pose.

Contact and workspace protection are independent:

- `--grasp-force-limit 25` smoothly saturates simulated closing force with a
  tanh law instead of allowing the position actuator to wind up through the
  cube. `--gripper-speed 0.12` also rate-limits the simulated jaw target.
- `--table-contact-force-limit 40` (an alias for `--surface-force-limit`)
  smoothly saturates only the Cartesian spring component pressing through the
  tabletop. Sliding and lifting remain free.
- The tool target is constrained to the printed task workspace. The physical
  handle has a matching conservative wall around the raised home position;
  tune it with `--workspace-wall-half` and `--workspace-wall-stiffness`.
- Cube-lift defaults to `--tip-softness 0.5` and
  `--force-sensor-cutoff 30`, unlike flip-up's opt-in defaults.

The omega.7 grasp return is its own 1 kHz feedback loop. Simulated two-finger
contact load is mapped with `--grip-force-gain 0.08`, then filtered by
`--grip-force-tau 10`, slew-limited by `--grip-force-rate 60`, damped by
`--grip-damping 6`, and capped by `--max-grip-force 3`. The normal sign is
`--grip-feedback-sign -1`, which pushes the physical jaws open to resist a
squeeze; switch it to `1` only if the particular hand configuration feels
assisting instead of resisting. During collection idle, a gentle opening force
and the Cartesian centering pull return both device channels to a repeatable
start before `S` is accepted.

By default, episodes independently randomize plausible cube colour, mass by
±20%, x/y placement, and the gripper's start pose while keeping the requested
2.75 cm cube size exact; attempt zero is centred. Alongside the normal
pose/wrench/MuJoCo channels, every 1 kHz sample
stores `object_pose`, `object_height_m`, `object_lift_height_m`, `task_metric`,
`gripper_command`, `gripper_controller_target`, `gripper_opening`,
`gripper_actuator_force`, `grasp_force`, physical device gap and applied grip
feedback, plus surface/workspace-limit flags. State replay automatically selects
the cube environment from episode metadata.

The viewer reports current and episode-maximum grasp and robot/table contact
forces beside their explicit configured limits. Each line changes from `OK` to
`EXCEEDED` when its solver-measured maximum crosses the threshold, and the same
`table_contact_force` plus per-sample exceedance flags are recorded at 1 kHz.

The omega.7's measured maximum gap is not assumed to be exactly 25 mm. During
the first idle reset, release the grip and let the gentle opening force settle;
the collector learns the direction and largest stable gap after 350 ms and uses
that endpoint for both the `S` start gate and analogue jaw mapping. Negative gap
readings, including approximately -26 mm on a left-hand omega.7, are supported.
The learned value persists
across resets and is saved in episode metadata. `--device-grip-open` is therefore
only a nominal upper bound by default. Use `--no-device-grip-auto-calibration`
only when supplying a measured endpoint explicitly.

## Environment notes

- Needs `dm_control`, `spatialmath-python` and `toppra`, installed into the
  existing `teleop` conda env. **`dm_control` must be 1.0.27**: 1.0.28+ expects
  `mjModel.flex_interp` from mujoco 3.3, and mujoco is pinned to 3.2.7 for
  robosuite (see the main README). Verified that `robosuite`/`teleop_ball.py`
  still run unchanged after the install.
- The `flipup_minimal` drop does **not** reproduce its own documented seed-0
  result in this environment: `python run_flipup.py --seed 0 --no-viewer` reports
  `FAILURE: final angle=89.96 deg` where its README shows `SUCCESS: final
  angle=1.14 deg`. This is not caused by anything here — the unmodified package
  behaves the same way, so it is a physics-version difference (most likely
  mujoco 3.2.7 versus whatever it was validated against). Of seeds 0–5 the
  shipped trajectory solves only 1 and 4, so it is not a reliable regression
  oracle here. It matters less than it sounds: `--dry-run`, which walks the same
  arc at a constant 0.05 m/s through this pipeline, solves **0, 1, 2 and 4**
  (seeds 3 and 5 need a human to adapt the path), so the teleop stack is not the
  limitation.
- Its own scripted runner also cuts the trajectory short: `termination_timestep =
  6000` steps = 6.0 s against a 6.38 s plan.
