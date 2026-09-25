# Teleop Control-Loop Latency Diagnostics

Adds a wall-clock breakdown of `teleop_flipup.py`'s control loop so a lagging
run can be diagnosed (device read? physics step? recorder? viewer/readout?
the pacing sleep overshooting?) instead of only observed via the pre-existing
`deadline_lateness_ms`/`control_batch_size` fields, which say a batch fell
behind but not *where* the time went.

## Usage

```
--latency-diagnostics                 # print a periodic breakdown
--latency-log-interval 2.0             # seconds between printouts (default 2.0)
```

Example printout:

```
[latency] prev_iter_ms=17.91/23.28ms avg/max  command_ms=0.01/0.01ms avg/max
          sim_batch_ms=17.90/23.25ms avg/max  prev_view_ms=0.02/0.02ms avg/max
          prev_sleep_ms=0.00/0.00ms avg/max  catchup-saturated 28/28 iters (100%)
```

**The printout is opt-in; the underlying data is not.** The per-stage timers
are cheap (`perf_counter()` calls, ~50-100ns) and run unconditionally, and
every sample written via `--collect-dataset` always carries the seven
`diag_*` fields below -- so a run collected without `--latency-diagnostics`
still has this breakdown available for post-hoc analysis; the flag only
controls whether you also see it live.

## Fields (always in the recorded dataset)

| field | meaning | lag |
|---|---|---|
| `diag_prev_iter_ms` | total wall time of one full outer control-loop pass | 1 iteration |
| `diag_command_ms` | device read + position filtering/gating + slew-limiting | none |
| `diag_step_ms` | the `env.step()` physics call, per substep | none |
| `diag_record_ms` | the `recorder.record_sample()` call, per substep | 1 substep |
| `diag_catchup_debt_ticks` | ticks the wall clock says are owed, BEFORE clamping to the 16-tick catch-up cap | none |
| `diag_prev_view_ms` | viewer/readout/plot-decision section | 1 iteration |
| `diag_prev_sleep_ms` | the deliberate end-of-loop pacing sleep | 1 iteration |

`sim_batch_ms` (printout only, not a separate recorded field -- it's the sum
of one batch's `step_ms`+`record_ms`+bookkeeping) is the whole catch-up
batch: potentially several physics ticks run back-to-back when the loop has
fallen behind wall time.

**Why some fields lag by one iteration/substep:** `prev_view_ms` and
`prev_sleep_ms` measure code that runs *after* this iteration's samples are
already recorded, so the earliest they can be attributed is the *next*
iteration's samples. `diag_record_ms` lags by one substep for the same
self-referential reason -- a call's own duration isn't known until it
returns. `diag_command_ms`, `diag_step_ms`, and `diag_catchup_debt_ticks` are
all same-iteration/same-substep (available before their samples are written),
no lag.

**`catchup-saturated N/M iters`** is the single most important number for
telling "occasionally jittery" apart from "structurally can't keep up":
`MAX_CATCHUP=16` caps how many ticks one outer-loop pass will pay off. If
`diag_catchup_debt_ticks` (the owed-ticks count *before* that cap) regularly
exceeds 16, each pass is paying down at most 16 ticks while wall-clock debt
keeps growing faster than that -- the loop never catches up, rather than
occasionally stumbling.

## What this immediately found

A `--dry-run --no-view --collect-dataset` smoke test (headless osmesa,
otherwise default settings) came back **100% catchup-saturated** the entire
run, and `diag_step_ms` (mean 0.19ms, pure physics) was smaller than
`diag_record_ms` (mean 0.56ms, spiking to 12ms) -- in this environment, **the
recorder cost more than the physics it was recording**. The run's own
existing real-time-factor check confirmed it: 0.88x real time, meaning
recorded demos replay 1.14x faster than they were actually performed and
contact forces scale with that speed-up. This matches the same signature
seen in real user-recorded episodes (`flipup_arm_1khz_v13.zarr`): every
episode hit `control_batch_size=16` (the hard cap) almost continuously, with
`deadline_lateness_ms` averaging 10-220ms and spiking past 500-700ms in
several episodes.

## Implementation notes

- `teleop_flipup.py`: a `latency` dict tracks the lagged/same-tick values
  described above across loop iterations (same style as the existing
  `force_monitor`/`collection` state dicts in this file; named `latency`,
  not `diag`, to avoid colliding with the pre-existing `--diagnose` feature's
  own `diag` dict -- an unrelated force/handle oscillation-correlation tool
  declared later in the same function). `diag_stats` + `_diag_track()`
  accumulate count/sum/max between printouts, cleared each time
  `maybe_print_latency_diagnostics()` fires. `latency`, `diag_stats`, and
  `MAX_CATCHUP` are declared early in `main()` (before `render_thread`/
  `viewer_thread` start), not next to the control loop that updates them --
  the always-on-screen HUD line below reads them from the viewer thread,
  which can start running before execution reaches a later declaration
  point, and closures resolve free variables at call time, not definition
  time, so an out-of-order declaration would race.
- `pyrite_recorder.py`: `record_sample()` gained seven new optional kwargs,
  all defaulting to `0.0`, unconditionally appended via the existing generic
  `_NumericSampleBuffer` (no schema-version bump needed -- new columns are
  additive, matching how every other field in this class is always written
  regardless of whether a given run's task/config exercises it).
- Verified: full-arm test suite unaffected (34/34 relevant tests pass, same
  before and after), and a `--collect-dataset` smoke test confirms all 7
  `diag_*` fields land correctly per-sample in the resulting zarr.

## Always-on-screen HUD line

`draw_state()` now draws a latency line at the bottom-left of the live
viewer, every frame, regardless of `--latency-diagnostics`:

```
loop 17.7ms  debt  +29 ticks  catchup-saturated 100%  LAGGING
```

Green when healthy, red with a `LAGGING` suffix when `catchup_debt_ticks`
exceeds `MAX_CATCHUP` (16) or the recent catchup-saturated fraction exceeds
10% -- meant to be watched continuously during a live run, not just
inspected after the fact.

## Sparse recording + offline replay

The diagnostics above traced the lag to a specific cause: `diag_record_ms`
(the full recorder's per-tick cost -- `mj_getState`'s full-state copy, plus
contact/wrist/sensor wrench extraction, more costly with the 3-5
simultaneous contacts a real flip produces) exceeded `diag_step_ms` (pure
physics) in testing. Two ways to remove the recorder from the real-time
budget were considered (see the design discussion in this repo's history):
async/background recording was rejected -- the per-tick cost is CPU-bound,
not I/O-bound, so a Python thread still contends for the GIL, and anything
truly deferred would need a synchronous copy of live-mutating MuJoCo state
first anyway. Recording a cheap stream live and reconstructing the
expensive one offline (no real-time deadline there) was adopted instead --
the same pattern already validated in this repo's `gen_sanding_clean.py`.

**`--dataset-mode {full,sparse}`** (default `full`, unchanged behavior).
`sparse` swaps in `sparse_flipup_recorder.SparseFlipUpRecorder`: cheap
fields only (target/achieved pose, device telemetry, the `diag_*` timing
fields, no RGB) -- see that module's docstring for the exact field list and
why each heavy call is skipped.

Measured (`--dry-run --collect-dataset`, otherwise identical settings):

| | `full` | `sparse` |
|---|---|---|
| catchup-saturated | 100% | **0%** |
| `prev_iter_ms` avg | ~17.7ms | **~1.0ms** |
| real-time factor | 0.88x | **0.99x** |

**`replay_flipup_sparse.py --src <sparse.zarr> --dst <dense.zarr> [--dataset-hz N] [--verify]`**
reconstructs the full dense state/wrench stream offline: rebuilds the exact
`FlipUpTeleop` env from each episode's recorded metadata (an explicit
CLI-flag whitelist for controller/contact-physics args, plus the exact
accepted book/start-pose from `episode_attempt` -- no RNG/resampling needed),
replays the recorded command stream tick-by-tick, and calls the FULL
`PyriteEpisodeRecorder.record_sample()` at every tick since there's no
real-time deadline offline. `--verify` compares the replayed achieved
trajectory against the source's own live-recorded one as a determinism
check -- measured **0.000mm** max deviation on a `--dry-run` round trip.

Known limitation, stated in the script's docstring: the env-reconstruction
whitelist is a hand-maintained list of CLI flags, not `teleop_flipup.py`'s
own `env_kwargs`-building code (embedded in a large stateful `main()`, not
factored out for reuse) -- a new physics/controller-relevant flag added to
`teleop_flipup.py` in the future must also be added to the whitelist, or
replay will silently fall back to that flag's default.

Example commands:

```
# current full-rate interface (unchanged) -- watch the on-screen "loop"/
# "debt" HUD line, or add --latency-diagnostics for a periodic console
# breakdown, to see whether/how badly this is lagging on your machine:
python teleop_flipup.py --collect-dataset ~/data/run.zarr --latency-diagnostics

# new low-latency interface -- cheap fields only, live; reconstruct the
# dense stream afterward with replay_flipup_sparse.py:
python teleop_flipup.py --collect-dataset ~/data/run_sparse.zarr \
    --dataset-mode sparse --latency-diagnostics

# offline: rebuild the full dense state/wrench stream at up to 1kHz
python replay_flipup_sparse.py --src ~/data/run_sparse.zarr \
    --dst ~/data/run_dense.zarr --verify
```
