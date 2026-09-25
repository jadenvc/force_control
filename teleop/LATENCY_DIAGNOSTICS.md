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

- `teleop_flipup.py`: a `diag` dict tracks the lagged/same-tick values
  described above across loop iterations (same style as the existing
  `force_monitor`/`collection` state dicts in this file); `diag_stats` +
  `_diag_track()` accumulate count/sum/max between printouts, cleared each
  time `maybe_print_latency_diagnostics()` fires.
- `pyrite_recorder.py`: `record_sample()` gained seven new optional kwargs,
  all defaulting to `0.0`, unconditionally appended via the existing generic
  `_NumericSampleBuffer` (no schema-version bump needed -- new columns are
  additive, matching how every other field in this class is always written
  regardless of whether a given run's task/config exercises it).
- Verified: full-arm test suite unaffected (34/34 relevant tests pass, same
  before and after), and a `--collect-dataset` smoke test confirms all 7
  `diag_*` fields land correctly per-sample in the resulting zarr.
