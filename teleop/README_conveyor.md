# omega → conveyor pick-place teleoperation

Haptic teleoperation of the [`conveyor_minimal`](../conveyor_minimal) task: a
cube rides a belt toward a UR5e with a fin-ray WSG50, and the operator has to
grasp it off the moving belt and drop it in a bin before it reaches the belt's
far end. **The belt speed is drawn fresh on every episode** (0.01–0.30 m/s), so
a collected session covers the speed distribution the source
`conveyor_pick_place` task trains on.

It reuses the omega bridge, the force-rendering pipeline and the Pyrite recorder
from `teleop_ball.py` / `teleop_flipup.py`.

```bash
conda activate teleop
cd teleop

# No device: the scripted heuristic plays the operator and the whole path runs.
python collect_conveyor.py --dry-run --episodes 3 --auto-finish \
    --dataset "$PYRITE_DATASET_FOLDERS/conveyor_sim_20hz.zarr"

# With an omega attached.
python collect_conveyor.py --episodes 20 --auto-finish \
    --dataset "$PYRITE_DATASET_FOLDERS/conveyor_sim_20hz.zarr"
```

Files:

- `conveyor_teleop.py` — `ConveyorTeleop`, a `ConveyorEnv` subclass adding a
  Cartesian position interface, a gripper channel, the force sources, the
  success metric, the recorder channels and a threaded-safe camera factory. The
  direct analogue of `flipup_teleop.py`.
- `collect_conveyor.py` — the collector: device, mapping, haptics, recording.
- `pyrite_recorder.py` — shared with FlipUp; see *Dataset* below.
- `../conveyor_minimal/` — the scene, the belt model and the controller.

## What is different from FlipUp

| | FlipUp | conveyor |
| --- | --- | --- |
| gripper | closed throughout, nonprehensile | **the point of the task** |
| what moves on its own | nothing | the belt, at a new speed every episode |
| tool orientation | derived from commanded position | fixed top-down grasp |
| sim contact force | 22–82 N median while levering | ~1 N carrying, ~20 N transients |
| where the force information is | translational axes | **the grip axis** |

The last two rows are the ones that matter for tuning, and they are measured
below.

## What you move, and what moves in the sim

You command the **WSG50 tip position**; wrist orientation is the task's fixed
top-down grasp, so the omega's three translational axes are enough and there is
nothing to steer. (`ConveyorTeleop.target_pose7` accepts a rotation vector if you
want to drive orientation too, but `collect_conveyor.py` does not expose it —
`teleop_flipup.py --enable-rotation` is the reference implementation to copy if
you need it.)

Default mapping is `--axes -x,-y,z`, the same as `teleop_ball.py` and
`teleop_flipup.py`:

| handle motion | tool motion | what it does |
| --- | --- | --- |
| **away from you** (device −x) | sim +x | across the belt, toward the far rail |
| left/right (device ±y) | sim ∓y | **along the belt**: this is the tracking axis |
| **up** (device +z) | sim +z | lift clear of the belt, and carry to the bin |

The travel that matters is along the belt. A grasp needs the tool to follow the
cube for a few tenths of a second, and the belt-to-bin carry is about 0.3 m of
sim motion, which at `--scale 4` is 7.5 cm of handle travel. If a direction comes
out reversed on the hardware, change `--axes`: fixing motion direction
automatically fixes force direction, because the force is mapped back through the
transpose of the same matrix.

**The gripper.** On an omega.7 the grip axis drives the finger width directly:
fully open on the handle is a 10 cm opening, fully closed is 2 cm.
`--button-gripper` latches between those two on short button presses instead, for
an omega.6. `gripper_width_from_fraction` clamps rather than extrapolating, so a
device that reports slightly outside [0, 1] cannot command a width the actuator
would refuse.

## Where the force is, and where it is not

Measured in simulation (mujoco 3.3.5) over scripted picks at 0.05, 0.15 and
0.30 m/s, at the shipped `--tool-kp 16000 --arm-damping 2.5`. `contact` is the
solver's exact per-contact sum over robot contacts, `wrist` is the WSG50's own
F/T sensor tared at reset, `grip` is the fingertip normal load.

| phase | `contact` med / p99 | `wrist` med | `grip` med | cos(contact, wrist) |
| --- | --- | --- | --- | --- |
| free space, holding still | 0.00 / 0.00 N | 0.00 N | 0.0 N | — |
| tracking the cube down the belt | **0.00** / 0.00 N | 0.97 N | 0.0 N | — |
| descending onto it | 0.00 / 0.00 N | 7.4 N | 0.0 N | — |
| closing the fingers | 0.00 / 18.4 N | 9.8 N | 0.0 N | 0.54 |
| lifting it clear | 1.00 / 20.5 N | 3.1 N | 11.2 N | 0.92 |
| carrying to the bin | 0.96 / 1.8 N | 1.6 N | 11.3 N | 0.53 |
| lowering to place | 0.93 / 2.1 N | 5.7 N | 11.5 N | 0.65 |
| releasing | 0.00 / 1.1 N | 0.2 N | 0.0 N | 0.97 |

Two things follow, and they set the defaults:

**1. `contact` is the right default source, as in FlipUp.** It is exactly
0.00 N in free space by construction — verified per-step over a 0.6 s free-space
sweep in the tests. `wrist` reads 1–10 N with nothing touching anything, because
the sensor also sees the gripper's own inertia when the arm accelerates; in
FlipUp that was a small fraction of a 22 N signal (cos +0.99), but here the
signal itself is ~1 N, so the agreement drops to cos 0.53–0.65. `wrist` is for
studying what a real F/T sensor would give you, not for rendering. `estimated` is
unusable, as it is for FlipUp — see the next table.

**2. Most of what there is to feel is on the grip axis.** The sustained
translational force while carrying is the cube's weight, 0.96 N. Rendered at
FlipUp's `--stiffness 1500` (force gain 0.023) that is 0.023 N at the handle,
which is nothing. The grip channel carries 11 N while holding and 60 N+ during a
hard squeeze, which maps to a strong, informative grip signal. So
`collect_conveyor.py` reflects `grip_force()` into `FDOmega.set_grip_force`,
scaled by `--grip-force-full-scale` (60 N of fingertip load → `--grip-force-max`,
4 N at the handle by default).

If you want a firmer translational collision cue for hitting the belt or the bin
rim, raise `--stiffness` toward 3000–6000 and re-read the printed passivity
margin, rather than raising `--force-gain` blind. The margin is
`2 * damping / (1/control_freq + 2 * force_tau) / (tool_kp * scale * force_gain)`;
`predicted_feel()` computes it and the startup banner prints it, with a warning
below 2×. At the defaults (`--stiffness 1500 --scale 4 --damping 30
--control-freq 1000 --force-tau 0.002`) it is **8×**.

### The belt pull through the grasp

The belt keeps pulling the cube while the fingers hold it, bounded by
`mu * normal_load` (see `ConveyorEnv._drive_conveyor`). Measured over the window
where the fingers hold the cube and it is still on the belt:

| belt speed | belt pull on the cube, med / peak | resulting tool Fy peak | `mu * mg` |
| --- | --- | --- | --- |
| 0.05 m/s | 0.14 / 0.73 N | 1.36 N | 0.98 N |
| 0.15 m/s | 0.37 / 2.61 N | 3.70 N | 0.98 N |
| 0.30 m/s | 0.32 / 1.78 N | 2.91 N | 0.98 N |

It scales with speed and exceeds `mu * mg` on the peaks because the fingers press
down, which raises the normal load. This is the signal that makes a moving-belt
grasp feel different from a static one, and it lives on the same translational
axes as everything else, so it shares their gain.

### Free-space tracking lag, and why `estimated` cannot work

Tool commanded along the belt at a constant speed, belt stopped, median over the
steady window:

| `--arm-damping` | commanded speed | tool lag | `contact` | `estimated` |
| --- | --- | --- | --- | --- |
| 1.0 | 0.05 m/s | 1.1 mm | 0.00 N | 17 N |
| 1.0 | 0.10 m/s | 2.4 mm | 0.00 N | 39 N |
| 1.0 | 0.20 m/s | 6.2 mm | 0.00 N | 99 N |
| 1.0 | 0.40 m/s | 20.6 mm | 0.00 N | 173 N (clipped) |
| **2.5 (default)** | 0.05 m/s | 2.7 mm | 0.00 N | 43 N |
| 2.5 | 0.10 m/s | 5.8 mm | 0.00 N | 93 N |
| 2.5 | 0.20 m/s | 13.6 mm | 0.00 N | 154 N |
| 2.5 | 0.40 m/s | 37.2 mm | 0.00 N | 173 N (clipped) |

`estimated` is `tool_kp` times that lag, so it reads tens to hundreds of newtons
in free space and saturates `--force-clip` at ordinary hand speeds. It is kept
only for A/B comparison. The lag column is also the reason `--arm-damping`
matters: 2.5× removes the ringing the shipped 1.0× shows once the actuators
saturate, at the cost of roughly 2.5× the lag. Both values complete the scripted
pick.

## Where the grasp height came from

The source task's heuristic grasps at `object_z - 0.034`. On this gripper's
collider that drives the fin-ray pads into the belt surface on the way in.
Measured peak robot↔belt contact force over 9 episodes per depth:

| `grasp_height_over_object_m` | scripted success | peak robot↔belt force |
| --- | --- | --- |
| −0.034 (source value) | 9/9 | 173 N |
| −0.030 | 9/9 | 128 N |
| −0.026 | 9/9 | 85 N |
| −0.022 | 9/9 | 23 N |
| **−0.018 (shipped here)** | 9/9 | **0 N** |

`conveyor_minimal`'s heuristic ships −0.018: the pads never touch the belt, so
the only contact an operator feels during a grasp is the cube. Nothing else about
the grasp changed — success is 60/60 over seeds 0–5 with full randomization at
either depth.

## Simulated arm

`--tool-kp 16000` is what `conveyor_minimal` ships, which is also FlipUp's.
Unlike FlipUp, this task does **not** need it: there is no book edge to stay on,
just a 5 cm cube in a 10 cm opening. It is the first thing to try lowering if the
arm feels harsh — but note that lowering it changes the sim's task dynamics, not
the rendering. To change what the operator *feels*, use `--stiffness`.

The task-space controller saturates the UR5e's actuators if handed a large
position step, and can then stall in the wrong configuration. An absolute
position mapping never does that, because the handle moves continuously. A
scripted or replayed command must be slewed; `conveyor_minimal`'s README has the
details.

## Dataset

`collect_conveyor.py` writes the same Pyrite Zarr layout `teleop_flipup.py` does,
through the same `pyrite_recorder.py`, under schema name `pyrite_conveyor_sim`.
Install the recorder dependency once:

```bash
python -m pip install -r requirements_dataset.txt
```

Collection is exactly **20 Hz** by default; `--control-freq` must be an integer
multiple of `--dataset-hz`. RGB is stored as 224×224 HWC `uint8` from
`third_person_camera`, the camera the source task's policies were trained from.

Every episode carries PyriteML's required fields (`rgb_0`, `ts_pose_fb_0`,
`ts_pose_command_0`, `ts_pose_virtual_target_0`, `stiffness_0`, `wrench_0` and
their timestamps), the complete MuJoCo integration state, and the device state.
The task-specific channels replace FlipUp's book channels:

| channel | |
| --- | --- |
| `object_pose`, `object_twist_world` | the cube |
| `conveyor_speed_m_per_s` | **per sample**, so a slice of the dataset carries it |
| `object_distance_to_bin` | |
| `gripper_width`, `gripper_width_command` | measured and commanded |
| `grip_force` | fingertip normal load |
| `object_picked_up`, `success` | the judge's state |

Episode attributes additionally record the seed and episode index, the speed
range, the layout offset, the belt and bin positions, the cube's mass/size/
friction, the belt drive friction, the controller gains and the grasp
orientation.

`pyrite_recorder.py` became task-agnostic to support this: an env that defines
`recorder_task_channels()` gets those channels stored, and one that does not gets
FlipUp's book channels exactly as before, so existing FlipUp datasets and the
scripts that read them are unaffected.

Validate or list a dataset with the same tool FlipUp uses:

```bash
python -c "from pyrite_recorder import validate_pyrite_dataset as v; \
    print(v('$PYRITE_DATASET_FOLDERS/conveyor_sim_20hz.zarr'))"
```

`replay_pyrite_flipup.py` is FlipUp-specific and will not replay these episodes.

## Status: what has and has not been checked

Checked, in simulation, and covered by `tests/test_conveyor_teleop.py`:

- the whole collection path end to end via `--dry-run`, including a two-episode
  dataset at two different randomized belt speeds that passes
  `validate_pyrite_dataset`;
- `contact` is exactly zero in free space, per step;
- the force sources, wrench frames, force saturation, gripper mapping and
  gripper command latching;
- reset randomization, judge state, camera rendering, and that the arm
  visualization modes leave collision fields untouched;
- the passivity-margin arithmetic against FlipUp's documented 4× and 8× points.

**Not** checked, and worth doing before a long collection session:

- anything with the device attached. The axis mapping, `--scale`, `--home`, the
  grip-force scaling and the force gain are all carried over from FlipUp/BallPush
  and rescaled by the measurements above; none of them has been through a
  hardware pass on this task.
- whether `--stiffness 1500` feels too light. Given the 0.96 N sustained force it
  probably does; the fix is a higher `--stiffness` with the printed margin
  watched, or leaning on the grip channel.
- long-session stability of the 1 kHz loop with `--realtime` and RGB rendering
  on. `--no-rgb` runs much faster if the loop cannot keep up.
