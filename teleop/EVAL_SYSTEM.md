# Pyrite Eval System

Automated evaluation pipeline that runs sim rollouts at training milestones, schedules them
across the cluster, and logs results back to the same WandB run as training.

---

## Architecture

```
pyrite_train.sh
├── train.py              ← training loop; writes logs.json.txt + wandb_run_id.txt
└── eval_watcher.py       ← polls logs.json.txt, enqueues milestones
      └── eval_dashboard.py (daemon)
            ├── Scheduler loop (30s)  ← probes cluster, launches evals via SSH
            └── HTTP server (:8765)   ← live browser dashboard
```

**All three run concurrently on the training machine.** `pyrite_train.sh` starts them together.

---

## Quick Start

```bash
cd /store/real/jvclark/PyriteML
./pyrite_train.sh <task_config> <run_name> [device] [extra_hydra_overrides...]

# Examples
./pyrite_train.sh cube_pick_acp_v1  acp_run1   cuda:2
./pyrite_train.sh flipup_acp_v1     flipup_run1 cuda:5 task.action_pred_horizon=16
./pyrite_train.sh cube_pick_dp_v1   dp_run1    cuda:3 task.eval_exec_horizon=4
```

The script:
1. Reads `eval.script`, `eval.max_ticks`, `eval.extra_args` from the task YAML
2. Launches training with `train_pyrite_workspace.yaml` (3000 epochs, checkpoint every 50, milestone every 250)
3. Starts `eval_watcher.py` alongside
4. Writes `pids.txt` to the eval output dir

---

## File Paths

| Purpose | Path |
|---|---|
| Task configs | `/store/real/jvclark/PyriteML/diffusion_policy/config/task/<name>.yaml` |
| Workspace config | `/store/real/jvclark/PyriteML/diffusion_policy/config/train_pyrite_workspace.yaml` |
| Training outputs | `/local/real/jvclark/training_outputs/<ts>_<task>_<run>/` *(local disk, not NFS)* |
| Eval outputs | `/store/real/jvclark/PyriteML/evals/<ts>_<task>_<run>/epoch_XXXX/` |
| Queue file | `/store/real/jvclark/eval_queue/queue.json` |
| Perf history | `/store/real/jvclark/eval_queue/perf.json` |
| Scheduler log | `/store/real/jvclark/eval_queue/scheduler.log` |

> **Note:** Checkpoints live on `/local/` (machine-local SSD) and are **not** accessible from
> other nodes over NFS. `eval_watcher` passes `required_node=hostname` so the scheduler always
> runs the eval on the same machine where the checkpoint exists.

---

## Components

### 1. `pyrite_train.sh`

Located at `/store/real/jvclark/PyriteML/pyrite_train.sh`.

Parses the task YAML with Python to extract eval parameters, then runs:
```bash
nohup python train.py --config-name=train_pyrite_workspace task=<cfg> name=<run> training.device=<dev> ...
nohup python eval_watcher.py --checkpoint-dir <ckpt_dir> --out-dir <eval_dir> \
    --eval-every 250 --num-episodes 10 --max-ticks <N> \
    --eval-script <script> --dataset-path <zarr> <extra_args>
```

Outputs `pids.txt` with `train_pid`, `watcher_pid`, `ckpt_dir`, `task`, `run_name`, `device`.

#### Batch size: raise it

`train_pyrite_workspace.yaml` ships `dataloader.batch_size: 128`, which is well
below what these GPUs hold. A sanding DP/ACP run occupies **~12.4 GB of 98 GB**,
so batch size can go up roughly 4x before memory is the constraint:

```bash
./pyrite_train.sh <cfg> <run> cuda:0 dataloader.batch_size=512 val_dataloader.batch_size=512
```

Two things to adjust alongside it:

- **Scale the LR.** `lr=1e-4` was tuned at batch 128. Going to 512 wants roughly
  2-4x that (linear or sqrt scaling), or a warmup, otherwise the larger batch
  trains *worse* rather than faster.
- **Watch steps/epoch.** These datasets are small (16k samples -> 128 steps/epoch
  at bs=128). At bs=512 that is 32 steps/epoch, so an "epoch" is 4x cheaper but
  each does 4x fewer updates; scale `num_epochs` up or the run will finish
  undertrained at the same epoch count.

Raising batch size is the cheapest available speedup **after** GPU pinning (see
below) -- it does nothing if six runs are already contending for one card.

#### GPU pinning: `training.device` does NOT work

`train_diffusion_unet_image_workspace.py` drives device placement through
HuggingFace `Accelerator`, and its `device = torch.device(cfg.training.device)`
line and the `.to(device)` calls are **commented out**. `training.device=cuda:N`
is therefore dead code and every run lands on physical **GPU 0** regardless.
Verified: runs launched with `cuda:1/2/6/7` were all on GPU 0 simultaneously.

Symptoms are slow epochs (six runs sharing one card at 1.16 s/it) followed by
`torch.OutOfMemoryError` at ~94 GiB while other GPUs sit idle.

Pin with `CUDA_VISIBLE_DEVICES` and pass `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=3 PYRITE_NUM_EPOCHS=1000 \
  setsid ./pyrite_train.sh <cfg> <run> cuda:0 training.num_epochs=1000 \
  < /dev/null > /tmp/launch_<run>.log 2>&1
```

Measured effect: 0.86 it/s -> 2.2-3.3 it/s per run (~3-4x), taking 1000 epochs
from ~43 h to ~11 h. `launch_sanding_clean_training.sh` does this automatically.

`setsid` matters too: without it the run shares the launching shell's process
group and dies when that shell is torn down.

#### Two directories per run

Each launch creates **two** directories under `PYRITE_CHECKPOINT_FOLDERS`:

| dir | contains |
|---|---|
| `${ts}_${TASK_CFG}_${RUN_NAME}` | this script's redirected **stdout** (`train.log`) — tracebacks live here |
| `${ts}_${task_name}_${name}` | hydra's run dir: `logs.json.txt`, `checkpoints/`, `wandb_run_id.txt`, `.hydra/`, and its OWN `train.log` with only INFO lines |

`eval_watcher` needs the **second**. Pointing it at the first makes it wait on a
`logs.json.txt` that never appears, and no eval ever runs. Note both contain a
file called `train.log`, and hydra's ends after model construction — so reading
the wrong one makes a healthy run look like a silent crash. A relaunched run has
several of each; select the live one by content (`logs.json.txt` present, largest
`train.log`), not by mtime.

---

### 2. `eval_watcher.py`

Polls `logs.json.txt` every **120 seconds**. Log format (one JSON per line):
```json
{"global_step": 1234, "epoch": 50, "lr": 1e-4, "smoothness_loss": 0}
```

**Milestone logic:**
- Triggers at every multiple of `--eval-every` (default: 250) that is ≤ current epoch
- Sweeps upward from `eval_every` so past milestones are backfilled in order
- A milestone is "handled" when `.eval_queued` **or** `.eval_done` stamp exists in its output dir

**On trigger:**
1. Calls `enqueue(ckpt_path, out_dir, ..., required_node=hostname)` → appends job to `queue.json`
2. Writes `.eval_queued` stamp (prevents re-queuing)

**On completion detection:**
- Retroactively checks: if `.eval_queued` exists but no `.eval_done`, and `summary.json` appeared → logs to WandB + writes `.eval_done`
- Works even if the watcher was offline when the scheduler marked the job done

---

### 3. `eval_dashboard.py`

Daemon at `/store/real/jvclark/force_control/teleop/eval_dashboard.py`.

**Start/restart:**
```bash
cd /store/real/jvclark/force_control/teleop
PYTHONPATH=/store/real/jvclark:/store/real/jvclark/PyriteML:/store/real/jvclark/PyriteUtility:/store/real/jvclark/force_control/flipup_minimal:/store/real/jvclark/force_control/teleop \
nohup /local/real/jvclark/miniconda3/envs/imitation/bin/python eval_dashboard.py \
  --nodes real002,real005,real006,real007,real008,real009 \
  --poll 30 --port 8765 \
  >> /store/real/jvclark/eval_queue/scheduler.log 2>&1 &
```

**Access dashboard from laptop:**
```bash
ssh -L 8765:localhost:8765 real009
# then open: http://localhost:8765
```
*(Port 8765 is firewalled; SSH tunnel is required. The server binds to `0.0.0.0` but direct
access is blocked by the network.)*

**Text-only status (no daemon):**
```bash
python eval_dashboard.py --status
```

---

## Scheduler: GPU Selection

Every 30 seconds, the scheduler:
1. SSHes to all nodes and runs `nvidia-smi` to collect per-GPU metrics
2. Labels each process as **TRAIN** (>6 GB VRAM) or **OTHER**
3. Picks a GPU with `find_best_slot()`:

| Priority | Condition |
|---|---|
| **P1** | `real009`, no TRAIN process on GPU, ≥20 GB free |
| **P2** | Any other node, no TRAIN process on GPU, ≥20 GB free |
| **P3** | Any GPU, ≥20 GB free (fallback — may co-locate with training) |

- If a job has `required_node` set, only that node is considered (other jobs in the queue are not blocked — the scheduler uses `continue` to try them)
- Max 4 concurrent evals (`MAX_CONCURRENT = 4`)
- Launches via SSH: `conda run -n imitation python3 <eval_script> --ckpt ... --device cuda:N ...`
- Writes PID to `eval.pid`; polls `kill -0 <pid>` for liveness; declares done when `summary.json` appears

**nvidia-smi metrics collected** (per GPU):
- VRAM used/total/free
- GPU utilization %, memory bandwidth utilization %
- Power draw (W)
- GPU clock / memory clock (MHz)
- PCIe gen × width → theoretical bandwidth (GB/s)
- Per-process VRAM (for TRAIN/OTHER labeling)

---

## Eval Scripts

All scripts share this CLI interface (required by `launch_eval`):

```
--ckpt <path>             checkpoint file
--out-dir <path>          output directory (created if needed)
--num-episodes <N>        number of rollout episodes
--max-ticks <N>           max ticks per episode
--dataset-path <path>     zarr dataset (for env config + demo bank)
--device cuda:N           GPU
--video-episodes <N>      save first N episodes as mp4 (0 = no video)
```

Task-specific flags go in `eval.extra_args` in the task YAML and are appended by the watcher/scheduler.

| Script | Task | Action dim | Task-specific flags |
|---|---|---|---|
| `eval_cubelift_policy.py` | Cube pick (floating sim) | ACP 21 / DP 10 | `--acp-track-nominal`, `--replan-every-ticks`, `--history-bootstrap` |
| `eval_floating_flipup_policy.py` | Flipup / pivot (floating sim) | ACP 19 / DP 9 | `--restore-sim-state`, `--history-bootstrap`, `--t-init-override`, `--replan-every-ticks` |
| `eval_sanding_policy.py` | Sanding | — | — |
| `eval_arm_policy.py` | Arm (real robot or sim) | — | — |

---

## Video Output

Videos are saved to `<out_dir>/videos/` when `--video-episodes N > 0`.

Filename convention: `episode_NNN_success.mp4` or `episode_NNN_fail.mp4`

| Script | Plain video | Force-annotated video | VT/target side view |
|---|---|---|---|
| `eval_floating_flipup_policy.py` | ✅ | ✅ `*_forces.mp4` (Fx/Fy/Fz curves overlaid) | ✅ `*_annotated.mp4` via `eval_video_annotator.py` |
| `eval_cubelift_policy.py` | ✅ | — | ✅ `*_annotated.mp4` via `eval_video_annotator.py` |
| `eval_sanding_policy.py` | ✅ | — | ✅ `*_annotated.mp4` via `eval_video_annotator.py` |

**`eval_video_annotator.py`** (unified post-hoc annotation):
- Reads `episode_traces.npz` and the raw mp4
- Left panel: RGB video frame (obs camera)
- Center panel: side-view scatter showing tool position (blue), VT (red dashed), nominal/cmd target (green) — all in world-frame XZ plane
- Right panel: Fx, Fy, Fz time-series (wrist wrench) with 3-second rolling window
- Output: `episode_NNN_<outcome>_annotated.mp4`

---

## Eval Output Structure

```
epoch_0250/
  summary.json              # {"success_rate": 0.8, "avg_ticks": 45, "results": [...]}
  episode_traces.npz        # per-episode numpy arrays (tool_xyz, lift, stiffness, forces, ...)
  eval.log                  # stdout/stderr from eval script
  eval.pid                  # PID of eval process on the worker node
  .eval_queued              # stamp: job submitted to dashboard queue
  .eval_done                # stamp: completed + WandB results logged
  videos/
    episode_000_success.mp4
    episode_001_fail.mp4
    episode_000_success_forces.mp4    # flipup only
    episode_000_success_annotated.mp4 # unified annotator output
```

### summary.json schema
```json
{
  "success_rate": 0.8,
  "avg_ticks": 45.2,
  "results": [
    {
      "episode": 0,
      "success": true,
      "ticks": 42,
      "peak_total_wrist_F_N": 18.3,
      "peak_wrist_Fz_N": 14.1,
      ...
    }
  ]
}
```

---

## Queue Job Schema

`queue.json` is a JSON array. Each entry:

```json
{
  "id": "abc12345",
  "status": "pending | running | done | failed",
  "label": "run_name  epoch=0250",
  "ckpt_path": "/local/real/jvclark/training_outputs/.../checkpoints/latest.ckpt",
  "out_dir": "/store/real/jvclark/PyriteML/evals/.../epoch_0250",
  "eval_script": "/store/real/jvclark/force_control/teleop/eval_cubelift_policy.py",
  "dataset_path": "/store/real/jvclark/PyriteML/data/cube_pick/...",
  "num_episodes": 10,
  "max_ticks": 80,
  "extra_args": ["--video-episodes", "3", "--acp-track-nominal"],
  "required_node": "real009",
  "submitted_at": "2026-09-01T14:00:00+00:00",
  "node": "real009",
  "gpu": 2,
  "pid": 12345,
  "started_at": "2026-09-01T14:01:05+00:00",
  "completed_at": "2026-09-01T14:08:22+00:00",
  "result": "8/10"
}
```

The file is `fcntl`-locked during reads/writes to allow concurrent access from watcher and scheduler.

---

## WandB Integration

**Project:** `pyrite-force-control`  
**Group:** task name (e.g. `cube_pick`, `flipup`) — same-task runs cluster together in the UI  
**Tags:** `[model_type, task, pred{N}, exec{N}, run_name]` — filterable by any dimension

### Training → WandB
Training writes `wandb_run_id.txt` to the checkpoint dir immediately after `init_trackers`.
Config logged to WandB at startup includes:
- Hydra config (full)
- `task`, `model_type`, `action_pred_horizon`, `eval_exec_horizon`, `dataset`
- Dataset metadata from zarr: `cube_mass_kg`, `cube_friction`, `tip_softness`, `tool_kp`, `surface_force_limit`, `force_sensor_cutoff`, `settle_s`

### Eval → WandB
`eval_watcher._report_eval_to_wandb()` is called when an eval completes:
1. Reads `wandb_run_id.txt` from the checkpoint dir
2. Opens `summary.json` from the eval output dir
3. Calls `wandb.init(id=<run_id>, resume="allow")` — attaches to the **same** run
4. Logs at `step=milestone_epoch`:

| Key | Description |
|---|---|
| `eval/epoch` | Milestone epoch number |
| `eval/success_rate` | Fraction of episodes that succeeded |
| `eval/avg_ticks` | Mean ticks to success/failure |
| `eval/peak_force_max_N` | Max per-episode peak wrist force |
| `eval/peak_force_mean_N` | Mean per-episode peak wrist force |


Metrics are namespaced per horizon: `eval/exec4/success_rate`,
`eval/exec8/success_rate`, and `eval_final/exec{4,8}/...` for the 50-episode
final evals. Finals get their own prefix so a 50-episode point is never plotted
against an 8-episode milestone point on the same axis — the milestone cells are
noisy enough (a single episode moves them 12.5 points) that mixing the two is
actively misleading.

> **If `eval/success_rate` is missing from WandB**, it is almost certainly this:
> the watcher used `resume="must"`, but training writes `wandb_run_id.txt`
> locally **before/independently of the run being registered server-side**. A
> perfectly valid-looking id can therefore name a run WandB has never seen, and
> `"must"` refuses outright:
>
> ```
> WandB log failed (exec8): You provided an invalid value for the `resume`
> argument. The value 'must' is not a valid option for resuming the run
> (omq5rwc8) that has not been initialized.
> ```
>
> Every eval metric is then dropped, with only a line in `watcher.log` to show
> for it — the summaries are still on disk, so nothing looks broken until you go
> looking for the curves. Fixed by using `resume="allow"` (attach if present,
> create if not) and raising `init_timeout` to 300 s, since the 90 s default also
> times out from these nodes.
>
> A relaunched run has **several** hydra dirs, each with a different valid
> `wandb_run_id.txt`. Resolve which one the evals actually came from via
> `ckpt_dir` in the eval dir's `pids.txt` — globbing and taking the first match
> attributes results to a dead launch.
>
> To recover evals that already ran (their `.eval_done` stamps mean the watcher
> will never retry them):
>
> ```bash
> python backfill_eval_wandb.py --dry-run     # check run-id resolution first
> python backfill_eval_wandb.py
> ```

---

## Canonical Task Configs

Located at `/store/real/jvclark/PyriteML/diffusion_policy/config/task/`:

| Config | Task | Model | Action dim | pred_horizon | exec_horizon | Episodes | max_ticks |
|---|---|---|---|---|---|---|---|
| `cube_pick_acp_v1.yaml` | cube_pick | ACP | 21 | 16 | 8 | 10 | 80 |
| `cube_pick_dp_v1.yaml` | cube_pick | DP+FT | 10 | 16 | 8 | 10 | 80 |
| `flipup_acp_v1.yaml` | flipup | ACP | 19 | 8 | 8 | 10 | 300 |
| `flipup_dp_v1.yaml` | flipup | DP | 9 | 8 | 8 | 10 | 300 |

Each YAML has an `eval:` block:
```yaml
eval:
  script: /store/real/jvclark/force_control/teleop/eval_cubelift_policy.py
  max_ticks: 80
  extra_args:
    - --acp-track-nominal
    - --replan-every-ticks
    - "8"
    - --video-episodes
    - "3"
```

---

## Cluster Nodes

| Node | Priority | Notes |
|---|---|---|
| real009 | **P1** | Preferred for evals; training also runs here |
| real002, real005, real006, real007, real008 | P2/P3 | Used when real009 is busy |

All 6 nodes are probed in parallel every 30 seconds. Each node must be reachable via `ssh -F /store/real/jvclark/.ssh/config <node>` with no password prompt (BatchMode=yes).

---

## Performance Tracking

`perf.json` stores rolling history per node:
- **Eval speed:** duration per N-episode eval run, used to estimate how long a queued job will take
- **Train speed:** epochs/hour computed from `Δepoch / Δwall_time` between scheduler polls (from `logs.json.txt` mtime + last epoch)

Visible in the dashboard's CLUSTER and TRAINING RUNS panels.

---

## Troubleshooting

**Eval not triggering:**
- Check `watcher.log` in the eval output dir
- Verify `logs.json.txt` exists and is updating: `tail -f <ckpt_dir>/logs.json.txt`
- Check `.eval_queued` / `.eval_done` stamps in `epoch_XXXX/` subdirs

**Job stuck in `running`:**
- SSH to the assigned node, check `kill -0 <pid>` — if dead, job will be marked `failed` on next poll
- Check `eval.log` in the job's `out_dir` for the error

**Eval runs on wrong machine:**
- Ensure `eval_dashboard.py` accepts `required_node` in `enqueue()` (fixed 2026-09-01)
- Verify `eval_watcher.py` is passing `required_node=socket.gethostname().split(".")[0]`

**Dashboard unreachable:**
- Dashboard binds to `0.0.0.0:8765` but is firewalled — use SSH tunnel:
  ```bash
  ssh -L 8765:localhost:8765 real009
  ```
  then open `http://localhost:8765`
- Check process is running: `ps aux | grep eval_dashboard`
- Check port: `ss -tlnp | grep 8765`

**WandB eval metrics missing:**
- `wandb_run_id.txt` must exist in the checkpoint dir (written by training on first epoch)
- `eval_watcher.py` must still be running (it does the logging, not the scheduler)
- Check `watcher.log` for `[eval_watcher] Logged eval results to wandb` lines
