# omega → Push-Circle teleoperation

Push-T's simpler sibling: a circular puck instead of a T-shape, so there's
no orientation/rotation to reason about at all -- coverage is the exact
closed-form circle-overlap area (no rasterized grid, no resolution
parameter), and success is purely about getting the puck's center close
enough to the goal circle's center.

```bash
conda activate teleop
cd teleop
python teleop_push_circle.py --dry-run --no-view --dry-run-seconds 10   # no device: scripted demo
python teleop_push_circle.py                                            # real omega handle
```

## Success metric

`coverage_fraction()` is the fraction of the goal circle's area currently
covered by the puck (exact circular-segment formula, not a grid). This caps
at `(object_radius / goal_radius)^2` whenever the puck is smaller than the
goal -- `--object-radius`/`--goal-radius` default to `0.05`/`0.04` so full
coverage (perfect centering) is achievable, and the constructor raises a
clear error if you pick radii that make `--success-threshold` (default
`0.85`) geometrically unreachable rather than silently shipping an
uncompletable task.

## Randomized start

`--randomize-start` samples the puck and pusher start positions from the
same 70%-center-biased mixture push-T/flip-up use, with the same
overlap-rejection resampling (verified 0 overlapping starts across 200
resets even under a stress-tested 0.9 center-bias).

## Dynamic variants (same features as push-T)

- **`--disturbance-force`** -- an unforced Ornstein-Uhlenbeck random push
  applied directly to the puck (force-only; there's no rotation DOF to
  spin). Off by default.
- **`--goal-move-min/max-interval`** -- relocate the goal at an interval
  redrawn fresh each time, not a fixed period, so there's no reliable
  countdown to time against. `--goal-move-skip-prob` additionally makes a
  scheduled check sometimes do nothing.

### Opposite-direction goal bias (new here, not in push-T)

`--goal-move-opposite-bias [0,1]` makes a relocation land BEHIND the puck's
current direction of travel instead of drawn uniformly -- anchored to the
puck's current position (not the origin), at `--goal-move-distance` away,
within `--goal-move-opposite-spread-deg` of dead-opposite. Falls back to
uniform whenever the puck is slower than `--goal-move-velocity-threshold`
(there's no meaningful "direction of travel" to oppose while stationary) or
on the `(1 - opposite_bias)` fraction of draws that go uniform anyway, so
the backward bias itself isn't a perfectly reliable pattern either.

Verified: once the puck is moving, the sampled goal direction (relative to
the puck) lands within the configured spread of exactly opposite the
puck's velocity heading, across dozens of real relocations in a scripted
push test.

```bash
python teleop_push_circle.py \
  --randomize-start \
  --goal-move-min-interval 1 --goal-move-max-interval 2 \
  --goal-move-opposite-bias 0.85 --goal-move-distance 0.15 \
  --collect-dataset ~/data/push_circle_v1.zarr --auto-finish
```

Since the goal usually reappears behind wherever the puck is currently
heading, pushing carefully toward the current goal and then continuing that
way is a losing strategy -- the only thing that stays robust is stopping
and reversing as soon as the goal relocates, which is what "so it needs to
move backward" was asking for.

## Files

- `push_circle_teleop.py` -- the environment (`PushCircleTeleop`, `PushCircleProperties`).
- `teleop_push_circle.py` -- the haptic CLI driver.
- `push_circle_recorder.py` -- the dataset recorder (`PushCircleEpisodeRecorder`).
- `tests/test_push_circle.py` -- overlap-formula/friction/coverage/disturbance/goal-move checks.

## Not included (see README_push_t.md for why these were cut from push-T too)

RGB frames (no camera capture into the dataset) and the adaptive-compliance
virtual-target labels (`stiffness_0`) -- not meaningful for a rigid,
ungripped pusher task.
