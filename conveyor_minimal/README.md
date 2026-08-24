# Minimal MuJoCo Conveyor Pick-Place

This folder contains only the UR5e + WSG50 conveyor pick-and-place environment
and its scripted heuristic. Like [`flipup_minimal`](../flipup_minimal) it does
not depend on `PyriteEnvSuites`, `PyriteUtility`, `PyriteML`, Hydra, Ray, Torch,
Zarr, `mink`, or any dataset/recording code.

A cube rides a moving belt toward the robot. The task is to grasp it off the
moving belt and place it in the target bin before it reaches the belt's far end.
**The belt speed is drawn fresh on every reset** (0.01--0.30 m/s by default), as
is the belt/bin layout jitter and the cube's spawn pose and yaw.

The scene, the robot, the belt and bin geometry, and the success criteria are
the source `conveyor_pick_place` task's, so demonstrations collected here live
in the same workspace as that task's data.

## Install

Linux and Python 3.10 or newer are recommended.

```bash
cd conveyor_minimal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run

Run the scripted heuristic with the interactive MuJoCo viewer:

```bash
python run_conveyor.py --seed 0
```

Without a viewer, which is useful on a server:

```bash
python run_conveyor.py --seed 0 --no-viewer
```

After installation this equivalent command is also available:

```bash
conveyor-demo --seed 0
```

A successful run prints output similar to:

```text
Conveyor episode: seed=0 index=0 belt=0.195 m/s layout_offset=(-0.009, -0.009) m
Cube: mass=0.100 kg, friction=1.000/0.300/0.100, edge=5.00 cm
SUCCESS: success, belt=0.195 m/s (estimated 0.195), phase=retreating, simulated=4.02 s, wall=2.10 s
```

Run several consecutive episodes and get a tally. Each one draws a new belt
speed:

```bash
python run_conveyor.py --seed 0 --no-viewer --episodes 10
```

The shipped heuristic solves 60/60 episodes over seeds 0--5 with fully
randomized speed, layout and spawn pose, and 4/4 at every pinned speed from 0.01
to 0.30 m/s.

## Belt-speed randomization

Every `reset()` draws a new speed from `--conveyor-speed-range`. `--seed` and
`--episode-index` together select it, so any episode is reproducible while
successive resets differ:

```bash
# Reproduce one specific episode
python run_conveyor.py --seed 0 --episode-index 7 --no-viewer

# Restrict the speed distribution
python run_conveyor.py --conveyor-speed-range 0.20 0.30 --no-viewer --episodes 5

# Pin the speed and switch the layout jitter off, for A/B comparisons
python run_conveyor.py --conveyor-speed 0.15 --no-randomize-layout --no-viewer
```

From Python:

```python
from conveyor import ConveyorEnv, ValueRange

env = ConveyorEnv(show_viewer=False, seed=0)
for episode in range(5):
    env.reset(episode_index=episode)
    print(env.conveyor_speed_m_per_s, env.layout_offset_xy)

# Or narrow the distribution / pin it
fast = ConveyorEnv(show_viewer=False, belt_speed_range=ValueRange(0.25, 0.30))
fixed = ConveyorEnv(show_viewer=False, belt_speed_m_per_s=0.15)
```

Randomization can be switched off per axis with
`randomize_belt_speed=False` and `randomize_layout=False`.

## Cube physical properties

Cube mass, edge length and the three friction coefficients change the compiled
model, so unlike the belt speed they are fixed for the lifetime of an env
instance:

```bash
python run_conveyor.py --randomize-cube --no-viewer --episodes 5
python run_conveyor.py --cube-mass 0.25 --cube-edge 0.04 --cube-friction 0.7 --no-viewer
```

```python
from conveyor import CubeProperties, run_conveyor, sample_cube_properties

run_conveyor(seed=4, show_viewer=False,
             cube_properties=CubeProperties(mass_kg=0.25, half_extent_m=0.02))
run_conveyor(seed=4, show_viewer=False, randomize_cube=True)
```

Cube randomization uses a random stream separate from the belt speed, layout and
spawn pose, so enabling it does not change the episode an existing seed
produces. The cube's sliding friction is also the grip the belt has on it (see
below), so lowering it makes the cube slip on the belt.

## How the belt works

MuJoCo has no conveyor primitive, and its contact solver treats the belt surface
as stationary. The source task sidestepped this kinematically: once per 20 Hz
control step it teleported the cube's freejoint forward and pinned its velocity.
That does not port to this package, which runs a 1 kHz task-space torque
controller so the haptic bridge can share it:

- pinned every millisecond, the cube could never be lifted off the belt at all;
- a 20 Hz teleport would reach an operator's hand as an impulse train.

So the belt here is modelled as what it is. `conveyor_belt_surface` takes contact
priority and contributes no tangential friction of its own, and
`ConveyorEnv._drive_conveyor` supplies the Coulomb friction of a surface moving
at `conveyor_speed_m_per_s`: a planar force opposing the cube's velocity
*relative to the belt*, saturating at `mu * normal_load`, where `mu` is the
cube's own sliding friction and the normal load is read from the actual belt
contacts.

Consequences, all of them the physical ones:

- The cube rides at exactly the commanded speed (measured to within 0.01% from
  0.01 to 0.30 m/s) and reaches it within a few tens of milliseconds.
- Pressing down on the cube increases the belt's grip on it.
- Once the gripper holds the cube, the belt slips underneath and keeps pulling
  with a bounded force. That force is what an operator feels while grasping
  against a running belt, and it is what makes the grasp a race.
- The belt's end rollers are visual only. They stand 2 cm proud of the surface,
  so as collision geoms they would dam the belt and the cube could never run off
  the end -- which is the task's miss condition. The start-side back stop is a
  real wall and still collides.

## Contact channels

Two collision channels leave exactly the pairs the task needs: robot-world,
robot-cube and cube-world. The robot emits one channel and accepts the other, so
robot self-contact is off; the fixtures only accept the robot's channel, so
static fixture pairs (the belt surface against its own frame, the bin walls
against the bin floor) never generate contacts either.

## Commanded poses must be slewed

The task-space controller is the one from `flipup_minimal`: a stiff Cartesian
spring with joint damping and gravity compensation, writing torques directly.
Handed a large step it saturates the UR5e's actuators and can stall in the wrong
configuration. Every caller therefore slews its commanded target -- the
heuristic at 0.74 m/s, the test helper at 0.35 m/s. With a slewed target the
whole belt, the bin and every grasp height are reached to under 5 mm.

## Contents

- `conveyor/environment.py`: MuJoCo model assembly, task-space controller, belt
  drive, and per-reset randomization.
- `conveyor/scene.py`: belt and bin geometry, spawn sampling, layout jitter.
- `conveyor/properties.py`: validated cube values and randomization ranges.
- `conveyor/judge.py`: pick/place success, miss, fall and time-limit bookkeeping.
- `conveyor/heuristic.py`: the scripted agent and the episode runner.
- `conveyor/assets/`: the scene, cube, UR5e and WSG50 assets this task needs.
- `run_conveyor.py`: executable entry point.

Learning code, recording code, datasets, and unrelated environments are
deliberately omitted.

## Deviations from the source task

Recorded so a discrepancy against `PyriteEnvSuites` data is not a surprise:

| | source `conveyor_pick_place` | here |
| --- | --- | --- |
| timestep / integrator | 2 ms, Euler | 1 ms, `implicitfast` |
| arm control | position actuators + `mink` IK | task-space torque controller |
| belt | kinematic teleport at 20 Hz | surface friction at the belt speed |
| end rollers | collision geoms | visual only |
| tool orientation | free, from the policy | fixed top-down grasp |
| belt speed | per-episode from the task config | per-reset, in the env |

The first two are what `flipup_minimal` already does, and are what makes force
feedback possible. Geometry, spawn distribution, speed distribution, layout
jitter and success criteria are unchanged.

## Tests

```bash
python -m pip install pytest
python -m pytest tests -q
```

## Haptic teleoperation

The optional Force Dimension bridge lives in [`../teleop`](../teleop); see
`../teleop/README_conveyor.md`.

## Troubleshooting

If the viewer cannot open, check that the machine has a working desktop/OpenGL
session, or use `--no-viewer`.

On a headless machine, set `MUJOCO_GL=osmesa` (or `egl`) for `ConveyorEnv.render`
and for the tests.
