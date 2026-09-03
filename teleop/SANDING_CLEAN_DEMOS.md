# Sanding: cleaning the teleop demos, and evaluating policies in the same env

Covers the `sanding_1` human teleop dataset (56 episodes, 1 kHz, collected on
commit `f535145`): what was wrong with it, how the cleaned datasets were built,
how to reproduce an eval, and what the policies actually scored.

---

## 1. What the teleop demos look like

56 episodes, 1 kHz control, RGB asynchronous at ~37.8 Hz, mean 8.1 s/episode,
**43/56 (76.8%) success**, mean coverage 0.891.

The dominant problem is **contact chatter**: the pad repeatedly loses and
re-acquires contact during the sweep.

| metric | value |
|---|---|
| contact-loss events | **~26 /s** (up to 72 /s) |
| median contact fragment | 9 ms |
| median gap | 1 ms |
| `ncon == 0` during contact | 9.3% mean, 42% worst |
| force roughness (std of per-ms Δ) | 1.7 N/ms |
| in-contact force | **9.5 N** against an 18 N nominal target |

99% of gaps are shorter than 50 ms and the median tool lift during a gap is
0.01 mm, so these are genuine chatter, not the operator lifting between regions.

The chatter is **generated in the contact solver, not by the operator's hand**:
52–71% of the force signal's power is above 100 Hz, while the commanded and
measured tool positions are >90% below 10 Hz. The root cause is the dose law —
`dose = k·(F − 6.66)·t` with the `[0.5, 1.5]` band — which at the demonstrator's
sweep speed forces sanding at ~9 N, right at the edge of stable contact for this
pad/panel pair.

---

## 2. What removes it (measured, in order of leverage)

- **Sliding friction 0.6 → 0.3.** Biggest single lever; removes the stick-slip
  that couples lateral motion into contact loss. One episode went 25.7 → 0.00
  losses/s, roughness 2.12 → 0.021 N/ms.
- **`tool_kp` 16000 → 4000.** Best of {2000, 4000, 8000, 16000}: 0.094 N/ms
  roughness vs 0.43 at 16000, and 7/10 test episodes at literally zero
  contact loss.
- **Sweep SLOWER, not faster.** Chatter scales with lateral speed through the
  XY→Z kinematic coupling. Warping *faster* to permit a higher force made it far
  worse (warp 0.44 → 86 losses/s); warping to 1.25× and lowering the force to
  hold the same dose drives it to zero.
- `arm_damping` **must stay 2.5** — 8.0 and 20.0 both destabilise badly.
- Command low-pass filtering barely helps, which is consistent with the chatter
  not being in the command.

---

## 3. The clean datasets

`gen_sanding_clean.py` replays each demo's recorded command trajectory through
the same env with a softer arm and lower friction, regulating Z with a slow
force servo, and pins the target line inside the span the source demo actually
swept so coverage is recoverable. The force setpoint is solved per episode to
**maximise coverage** — not to hit mean dose 1.0, because dose is spread
unevenly along the target line and the mean can sit at 1.0 while a third of
cells fall outside the band.

Two modes:

| dataset | mode | warp | episodes |
|---|---|---|---|
| `sanding_clean_timing.zarr` | preserves the source episode's duration | 1.0 | 100 |
| `sanding_clean_smooth.zarr` | stretches it for the lowest chatter | 1.25 | 100 |

Both are **100/100 success**:

| metric | source teleop | timing | smooth |
|---|---|---|---|
| success | 43/56 (77%) | **100/100** | **100/100** |
| contact losses/s | 25.9 | 0.27 (**95× fewer**) | 0.16 (**167× fewer**) |
| roughness N/ms | 2.18 | 0.090 (24×) | 0.062 (35×) |
| coverage | 0.891 | 0.954 | 0.963 |
| duration | 8.11 s | 7.84 s (−3%) | 9.31 s (+15%) |
| lateral speed | 25.9 mm/s | 27.7 (+7%) | 22.1 (−15%) |

`verify_sanding_clean.py` regenerates that comparison from any pair of zarrs.

---

## 4. Reproducing an eval

**Use `eval_sanding1_policy.py`, not `eval_sanding_policy.py`.** The older script
targets the scripted `sanding_synthetic.zarr` and is wrong for this data in ways
that are silent rather than loud (see §5).

```bash
python eval_sanding1_policy.py \
    --ckpt   <run>/checkpoints/milestone_epoch=0500.ckpt \
    --dataset-path /store/real/jvclark/sanding_clean_smooth.zarr \
    --tool-kp 4000 --friction 0.3 \
    --replan-every-ticks 8 --num-episodes 50 --max-ticks 30000 \
    --video-episodes 3 --device cuda:0
```

Defaults are `--tool-kp 16000 --friction 0.6` (stock / raw teleop), so the flags
are **required** for the clean datasets. Everything else in `SandingProperties`
is left at the stock `f535145` values — the env code itself is unmodified, so
`git checkout f535145` reproduces the physics exactly.

Which values a dataset needs is recorded in the zarr itself as a `gen_env` root
attr, and `check_sanding_env_match.py` verifies dataset ↔ task YAML ↔ live
compiled MuJoCo (`geom_solref`, `geom_friction`, priority, camera, dose band):

```bash
python check_sanding_env_match.py --zarr <zarr> --config <task_cfg_name>
```

That third layer matters: `--pad-softness` was silently inert for months because
the panel geom's higher contact priority overrode the value written to the pad,
so a config that reads correctly can still compile to different physics.

| dataset | tool_kp | friction |
|---|---|---|
| `sanding_1.zarr` (raw teleop) | 16000 | 0.6 |
| `sanding_clean_timing.zarr` | 4000 | 0.3 |
| `sanding_clean_smooth.zarr` | 4000 | 0.3 |
| `sanding_acp_raw.zarr` | 16000 | 0.6 |
| `sanding_acp_smooth_{ema,maxpool}.zarr` | 4000 | 0.3 |

---

## 5. Why the old eval script was wrong

Each of these was an outright bug, and none of them announced itself:

- **Wrong env.** `GroupedSandingEnv` with scripted-data properties (force target
  12 N, dose band `[0.1, 30]`, `pad_softness` 0.6, 5 clustered regions) instead
  of the stock randomised-line env the demos were collected in.
- **Wrong camera.** It rendered the wrist camera; `SandingEpisodeRecorder` stores
  the main scene camera (az 90, el −75, dist 0.55) at 520×390, INTER_AREA-resized
  to 224×224. The policy was being shown a view it was never trained on.
- **Wrong control rate.** `n_substeps=2` (500 Hz) against 1 kHz data.
- **Wrong action timing.** One action per tick, so the commanded trajectory
  advanced 16× too fast; each action must be held for
  `sparse_action_down_sample_steps` (16) ticks.
- **Wrong action frame — the expensive one.** Actions are RELATIVE. Training
  applies `SE3_relative = SE3_inv(base) @ SE3_absolute` with `base` = the most
  recent observed eef pose, which is why `robot0_eef_pos` is identically
  `[0,0,0]` in a batch: the policy never sees absolute position. Eval must invert
  it, `world_target = base @ predicted`. Feeding the prediction as an absolute
  pose drove the tool into the panel at **>300 N** on the first step.
- **Wrong image layout.** RGB must be `float32/255` then `moveaxis(-1, 1)` to
  `(T, 3, 224, 224)`.
- **Missing normalizer.** It is not carried in the checkpoint; the workspace
  pickles it to `sparse_normalizer.pkl` beside the run and calls
  `set_normalizer()` at train time. Without restoring it, `predict_action`
  returns the raw network output — ~0.04 m instead of an absolute ~0.33 m pose.
- **Undersized offscreen buffer.** `model.vis.global_.offwidth` ships at 600 px
  and dm_control allocates the shared render context when the FIRST camera is
  built, so a later `max()` is too late: the 640-wide video camera rendered
  uninitialised garbage in its right 40 columns. Size the buffer for the largest
  camera before constructing any of them. (Observations were unaffected — the
  520-wide obs camera fits.)

---

## 6. Results

DP + force-torque input, 1000 epochs, 50-episode final evals, exec horizons
{4, 8}. 400 final-eval episodes total.

| | success | p vs raw |
|---|---|---|
| RAW `sanding_1` | 30/100 = 30% | — |
| clean-timing | 80/200 = 40% | 0.090 |
| clean-smooth | 43/100 = 43% | 0.056 |
| **all clean pooled** | **123/300 = 41%** | **0.050** |

Cleaning is worth roughly **+11 points of success, 30% → 41%**, sitting exactly
on the conventional significance threshold. Every one of the six clean cells
beats both raw cells, which is the reason to believe it; the formal test cannot
put much distance on it at 50 episodes per cell. Note the pooled figure
aggregates across two datasets, two checkpoints and two horizons — legitimate
for power, but not a pre-registered single hypothesis.

**Cleaning does NOT buy lower contact force or fewer breakages.** An apparent
force benefit showed up twice and reversed both times once the second exec
horizon landed:

```
exec8:  clean 15.4 N   raw 22.5 N    −7.1
exec4:  clean 18.7 N   raw 16.7 N    +2.0
```

Breakage is 2/300 vs 1/100 — the same rate. Clean policies also run ~40% longer
per episode (21–23 s vs 16 s).

---

## 7. ACP on this task

`add_sanding_acp_fields.py` originally wrote
`ts_pose_virtual_target_0 = ts_pose_command_0` and `stiffness_0 = 1.0`. Both ACP
channels were therefore degenerate — the VT block a byte-duplicate of the command
block, the stiffness channel zero-variance — so "ACP" was functionally DP with 10
dead action dimensions, and the stiffness label was additionally a factor of 4
wrong on the clean datasets (`tool_kp` 4000 → 0.25, not 1.0).

`relabel_sanding_acp.py` applies the construction from
`gen_cube_pick_arm_synthetic.py`'s `relabel_acp_with_known_k()`:

```
VT = cmd + F/k          stiffness_0 = k / 16000
```

With k known (the controller's fixed `tool_kp`) this is well-posed and turns the
VT block into an implicit force command. Two deliberate differences from the
cube_pick version:

- **No rotation.** cube_pick reads `wrench_filtered_0` in the TOOL frame and maps
  the displacement through `R(cmd_quat)`; `SandingEnv.pad_contact_force()` already
  returns world-frame, so rotating again applies a spurious transform.
- **Sign.** `pad_contact_force()` returns the reaction ON THE PAD (+Fz when
  pressing down), the opposite convention to cube_pick's, which negates.

Validated: `|VT − cmd|` = 2.360 mm at k=4000 against a predicted `F/k` of
2.38 mm, and 0.636 mm at k=16000 against 0.59 mm.

`--force-agg {ema,maxpool,raw}` selects how the force driving the label is
aggregated (both filters causal, so the label stays something a policy could
commit to from past observations alone).

**Caveat:** `stiffness_0` is still constant *within* a dataset, so this buys a
force-aware action space, not adaptive compliance. Genuinely varying stiffness
needs demos generated with a varying `--kp`, the way
`gen_cube_pick_arm_synthetic.py` does it by phase (`K_APPROACH=16000` →
`k_contact=200–600` → `K_HOLD`).

---

## 8. A live bug in the teleop driver

`teleop_sanding.py` (around the `record_sample` call, ~line 895) takes

```python
timestamp_ms         = step_index * 1000.0 / args.control_freq   # session-global
image_capture_time_s = shot["sim_time_s"]                        # env.data.time, per-episode
```

`step_index` keeps counting across episodes; `env.data.time` is reset by
`env.reset()`. So `robot_time_stamps_0`/`wrench_time_stamps_0` and
`rgb_time_stamps_0` only share a clock for the FIRST recorded episode.

PyriteML's sampler intersects the streams in time:

```
start = max(rgb_t[0],  robot_t[0])
end   = min(rgb_t[-1], robot_t[-1])
```

so for every later episode the window is empty and the episode contributes
**zero** training samples — silently. On `sanding_1` this dropped 54 of 56
episodes; the dataset was 613 samples instead of 16,266.

`fix_sanding_rgb_timestamps.py` repairs an existing zarr (offset recovered from
`sim_time_s`, originals preserved as `rgb_time_stamps_0_raw`). **The driver
itself is still unfixed** — worth doing before the next collection.

---

## 9. Files

| file | purpose |
|---|---|
| `eval_sanding1_policy.py` | eval in the collection env; force-annotated videos |
| `gen_sanding_clean.py` | clean demo generation by augmented replay |
| `verify_sanding_clean.py` | source vs generated head-to-head |
| `relabel_sanding_acp.py` | ACP VT/stiffness relabel (`--force-agg`) |
| `fix_sanding_rgb_timestamps.py` | repair the rgb/robot clock mismatch |
| `check_sanding_env_match.py` | dataset ↔ config ↔ live-MuJoCo env gate |
| `gen_sanding_train_videos.py` | force-annotated videos of training samples |
| `make_sanding_configs.py` | derive PyriteML task configs for a dataset |
| `fast_merge_zarr.py` | hardlink-based zarr merge (~3.6 s/episode) |
| `finish_sanding_clean.sh` | merge → trim → stamp → ACP fields → configs → verify |
| `launch_sanding_clean_training.sh` | launch training behind the env gate |
| `restart_eval_watchers.sh` | repoint watchers at hydra's checkpoint dir |
| `retrigger_final_evals.sh` | fire each run's final eval when that run is ready |
