#!/usr/bin/env python3
"""Watch a training run and trigger sim rollout eval every N epochs.

Protocol
--------
- Polls logs.json.txt every --poll-interval seconds.
- At each multiple-of-eval_every milestone:
    submits one eval job per exec_horizon (e.g. [4, 8]):
      • 8 episodes each → 16 evals per milestone
      • cold-start: --history-bootstrap static (no warm buffer/force)
      • initial positions interpolated from demo starts (--restore-sim-state /
        --no-randomize-cube flags come from task YAML extra_args)
      • --replan-every-ticks N injected per horizon
- Stamps per horizon:  .eval_queued_execN  /  .eval_done_execN
- A milestone is fully "handled" when ALL horizon stamps exist.
- WandB: logs eval/execN/success_rate, eval/execN/avg_ticks, etc.
  Also logs exec_horizon to wandb config for sortability.
- End-of-training detection: when epoch >= --num-epochs (and log stops
  changing), find the milestone with the highest mean success rate across
  both horizons, then queue a --final-eval-episodes (default 50) run of
  that checkpoint. Stamp: final_eval/.final_eval_queued

See EVAL_SYSTEM.md for full documentation.

Usage:
    # Queue mode (default)
    python eval_watcher.py \\
        --checkpoint-dir /local/real/jvclark/training_outputs/<run> \\
        --out-dir /store/real/jvclark/PyriteML/evals/<task>/<date>_<run> \\
        --eval-every 250 --num-epochs 3000 \\
        --exec-horizons 4,8 --num-episodes 8 \\
        --eval-script /store/real/jvclark/force_control/teleop/eval_cubelift_policy.py \\
        --dataset-path /store/real/jvclark/PyriteML/data/cube_pick/...

    # Local mode (legacy)
    python eval_watcher.py --local --device cuda:0 ...
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable

# Import queue helper
sys.path.insert(0, str(Path(__file__).parent))
try:
    from eval_dashboard import enqueue as _queue_enqueue
    _QUEUE_AVAILABLE = True
except ImportError:
    try:
        from eval_scheduler import enqueue as _queue_enqueue
        _QUEUE_AVAILABLE = True
    except ImportError:
        _QUEUE_AVAILABLE = False


# ── log parsing ────────────────────────────────────────────────────────────────

def last_epoch_in_log(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    last_epoch = None
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "epoch" in entry:
                        ep = int(entry["epoch"])
                        if last_epoch is None or ep > last_epoch:
                            last_epoch = ep
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        return None
    return last_epoch


# ── stamp helpers ──────────────────────────────────────────────────────────────

def _ep_dir(out_dir: str, milestone: int) -> Path:
    return Path(out_dir) / f"epoch_{milestone:04d}"

def _queued_stamp(out_dir: str, milestone: int, exec_hz: int) -> Path:
    return _ep_dir(out_dir, milestone) / f".eval_queued_exec{exec_hz}"

def _done_stamp(out_dir: str, milestone: int, exec_hz: int) -> Path:
    return _ep_dir(out_dir, milestone) / f".eval_done_exec{exec_hz}"

def _horizon_out_dir(out_dir: str, milestone: int, exec_hz: int) -> str:
    return str(_ep_dir(out_dir, milestone) / f"exec{exec_hz}")


# ── per-horizon eval result ────────────────────────────────────────────────────

def _read_summary(horizon_out_dir: str) -> dict | None:
    p = Path(horizon_out_dir) / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# ── WandB reporting ────────────────────────────────────────────────────────────

def _report_horizon_to_wandb(
    checkpoint_dir: str,
    out_dir: str,
    milestone: int,
    exec_hz: int,
) -> None:
    """Log one horizon's eval results to the training wandb run."""
    run_id_file = Path(checkpoint_dir) / "wandb_run_id.txt"
    if not run_id_file.exists():
        return

    h_dir = _horizon_out_dir(out_dir, milestone, exec_hz)
    summary = _read_summary(h_dir)
    if summary is None:
        return

    try:
        import wandb
        run_id = run_id_file.read_text().strip()

        results = summary.get("results", [])
        peak_forces = []
        for r in results:
            force = (
                r.get("peak_total_contact_force_n")
                or r.get("peak_total_wrist_F_N")
                or r.get("peak_wrist_Fz_N")
                or 0.0
            )
            peak_forces.append(float(force))

        prefix = f"eval/exec{exec_hz}"
        log_dict = {
            f"{prefix}/epoch":        milestone,
            f"{prefix}/success_rate": summary.get("success_rate", 0.0),
            f"{prefix}/avg_ticks":    summary.get("avg_ticks", 0.0),
        }
        if peak_forces:
            log_dict[f"{prefix}/peak_force_max_N"]  = max(peak_forces)
            log_dict[f"{prefix}/peak_force_mean_N"] = sum(peak_forces) / len(peak_forces)

        # resume="allow", not "must": training writes wandb_run_id.txt locally
        # before/independently of the run being registered server-side, so a
        # perfectly valid-looking id can refer to a run wandb has never seen.
        # "must" then refuses outright ("not a valid option for resuming the run
        # (<id>) that has not been initialized") and every eval metric is
        # silently dropped -- which is why eval/success_rate was absent from
        # wandb for every sanding run despite the id files being present and
        # readable. "allow" attaches if the run exists and creates it if not, so
        # the metrics land either way.
        #
        # init_timeout is raised from the 90 s default because these nodes
        # regularly exceed it and the resulting failure is also silent.
        run = wandb.init(
            id=run_id, resume="allow", reinit=True,
            project=os.environ.get("WANDB_PROJECT", "pyrite-force-control"),
            settings=wandb.Settings(init_timeout=300),
        )
        # Log exec_horizon as a config field for table-view sortability
        run.config.update({
            f"eval_exec{exec_hz}_milestone": milestone,
        }, allow_val_change=True)
        wandb.log(log_dict, step=milestone)
        run.finish()
        print(
            f"[eval_watcher] WandB logged exec{exec_hz} epoch={milestone}  "
            f"success_rate={log_dict[f'{prefix}/success_rate']:.2f}",
            flush=True,
        )
    except Exception as exc:
        print(f"[eval_watcher] WandB log failed (exec{exec_hz}): {exc}", flush=True)


# ── milestone completeness ─────────────────────────────────────────────────────

def _horizon_handled(out_dir: str, milestone: int, exec_hz: int) -> bool:
    """True if this (milestone, exec_hz) pair is queued or done."""
    done    = _done_stamp(out_dir, milestone, exec_hz).exists()
    queued  = _queued_stamp(out_dir, milestone, exec_hz).exists()

    # Retroactive completion: if queued but summary.json arrived, log + stamp done
    if queued and not done:
        h_dir = _horizon_out_dir(out_dir, milestone, exec_hz)
        if (_read_summary(h_dir) is not None):
            _done_stamp(out_dir, milestone, exec_hz).parent.mkdir(parents=True, exist_ok=True)
            _done_stamp(out_dir, milestone, exec_hz).touch()
            # checkpoint_dir not available here; watcher main loop handles wandb
            done = True

    return done or queued


def _milestone_fully_handled(out_dir: str, milestone: int, exec_horizons: list[int]) -> bool:
    return all(_horizon_handled(out_dir, milestone, hz) for hz in exec_horizons)


# ── best milestone for final eval ─────────────────────────────────────────────

def _best_milestone(out_dir: str, exec_horizons: list[int], eval_every: int) -> tuple[int, float] | None:
    """Scan epoch_XXXX dirs, return (milestone, mean_success_rate) of best."""
    root = Path(out_dir)
    best_m, best_rate = None, -1.0
    for ep_dir in sorted(root.glob("epoch_*")):
        try:
            m = int(ep_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        rates = []
        for hz in exec_horizons:
            s = _read_summary(_horizon_out_dir(out_dir, m, hz))
            if s is not None:
                rates.append(s.get("success_rate", 0.0))
        if rates:
            mean_rate = sum(rates) / len(rates)
            if mean_rate > best_rate:
                best_rate, best_m = mean_rate, m
    return (best_m, best_rate) if best_m is not None else None


# ── local eval subprocess (legacy mode) ───────────────────────────────────────

def run_eval_local(
    ckpt_path: str,
    milestone: int,
    out_dir: str,
    exec_hz: int,
    *,
    num_episodes: int,
    max_ticks: int,
    device: str,
    eval_script: str,
    dataset_path: str,
    extra_args: list,
) -> tuple[subprocess.Popen, object]:
    h_dir = _horizon_out_dir(out_dir, milestone, exec_hz)
    os.makedirs(h_dir, exist_ok=True)

    cmd = [
        PYTHON, eval_script,
        "--ckpt", ckpt_path,
        "--out-dir", h_dir,
        "--num-episodes", str(num_episodes),
        "--max-ticks", str(max_ticks),
        "--dataset-path", dataset_path,
        "--device", device,
        "--replan-every-ticks", str(exec_hz),
    ] + (extra_args or [])

    eval_env = os.environ.copy()
    eval_env["MUJOCO_GL"] = "egl"

    log_file = open(os.path.join(h_dir, "eval.log"), "w")
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        env=eval_env, preexec_fn=os.setsid,
    )
    print(
        f"[eval_watcher] Launched local eval exec{exec_hz} epoch={milestone:04d} "
        f"pid={proc.pid}  out={h_dir}",
        flush=True,
    )
    return proc, log_file


# ── queue submission ───────────────────────────────────────────────────────────

def _submit_horizon(
    ckpt_path: str,
    out_dir: str,
    milestone: int,
    exec_hz: int,
    *,
    num_episodes: int,
    max_ticks: int,
    eval_script: str,
    dataset_path: str,
    extra_args: list,
    required_node: str | None,
    label_prefix: str,
    is_final: bool = False,
) -> str | None:
    h_dir = _horizon_out_dir(out_dir, milestone, exec_hz)
    os.makedirs(h_dir, exist_ok=True)
    # inject replan rate for this horizon
    full_extra = list(extra_args) + ["--replan-every-ticks", str(exec_hz)]
    suffix = "(FINAL 50-ep)" if is_final else f"epoch={milestone:04d}"
    label = f"{label_prefix}  exec{exec_hz}  {suffix}"
    job_id = _queue_enqueue(
        ckpt_path    = ckpt_path,
        out_dir      = h_dir,
        eval_script  = eval_script,
        dataset_path = dataset_path,
        num_episodes = num_episodes,
        max_ticks    = max_ticks,
        extra_args   = full_extra,
        label        = label,
        required_node= required_node,
    )
    # stamp so we don't re-queue
    stamp = _queued_stamp(out_dir, milestone, exec_hz)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()
    print(
        f"[eval_watcher] Queued exec{exec_hz} epoch={milestone:04d}  "
        f"job_id={job_id}  n={num_episodes}ep",
        flush=True,
    )
    return job_id


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-eval watcher for PyriteML training runs")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Training output dir (contains logs.json.txt and checkpoints/)")
    parser.add_argument("--out-dir", required=True,
                        help="Root dir for eval output (task-specific, date-named)")
    parser.add_argument("--eval-every", type=int, default=250,
                        help="Trigger eval at each multiple of this epoch count (default: 250)")
    parser.add_argument("--exec-horizons", default="4,8",
                        help="Comma-separated execution horizons to evaluate (default: 4,8)")
    parser.add_argument("--num-episodes", type=int, default=8,
                        help="Episodes per horizon eval (default: 8; 2 horizons → 16 total)")
    parser.add_argument("--max-ticks", type=int, default=400)
    parser.add_argument("--num-epochs", type=int, default=3000,
                        help="Total training epochs; triggers end-of-training final eval")
    parser.add_argument("--final-eval-episodes", type=int, default=50,
                        help="Episodes for end-of-training best-policy eval (default: 50)")
    parser.add_argument("--device", default="cuda",
                        help="GPU device for local mode (ignored in queue mode)")
    parser.add_argument("--poll-interval", type=float, default=120,
                        help="Seconds between log polls (default: 2 min)")
    parser.add_argument("--eval-script", required=True,
                        help="Eval script path (e.g. eval_cubelift_policy.py)")
    parser.add_argument("--dataset-path", required=True,
                        help="Dataset zarr path passed to eval script")
    parser.add_argument("--local", action="store_true",
                        help="Run evals locally instead of submitting to queue")
    args, extra_eval_args = parser.parse_known_args()

    exec_horizons = [int(x.strip()) for x in args.exec_horizons.split(",") if x.strip()]
    log_path  = Path(args.checkpoint_dir) / "logs.json.txt"
    ckpt_path = Path(args.checkpoint_dir) / "checkpoints" / "latest.ckpt"
    os.makedirs(args.out_dir, exist_ok=True)

    model_name    = Path(args.checkpoint_dir).name
    required_node = socket.gethostname().split(".")[0]
    use_queue     = _QUEUE_AVAILABLE and not args.local

    # Local-mode state (one proc per horizon)
    running_procs: dict[int, tuple[subprocess.Popen, object, int]] = {}  # exec_hz → (proc, log, milestone)

    # End-of-training tracking
    final_eval_done = Path(args.out_dir) / "final_eval" / ".final_eval_queued"
    _last_epoch_seen: list[int] = [0]

    print(
        f"[eval_watcher] Watching {log_path}\n"
        f"  checkpoint:    {ckpt_path}\n"
        f"  eval every:    {args.eval_every} epochs, horizons={exec_horizons}\n"
        f"  episodes/eval: {args.num_episodes} per horizon "
        f"({args.num_episodes * len(exec_horizons)} total per milestone)\n"
        f"  num_epochs:    {args.num_epochs}  final_eval: {args.final_eval_episodes} ep\n"
        f"  poll interval: {args.poll_interval}s\n"
        f"  mode:          {'queue' if use_queue else 'local'}",
        flush=True,
    )

    while True:
        # ── Poll local procs ───────────────────────────────────────────────────
        for hz in list(running_procs):
            proc, logf, m = running_procs[hz]
            rc = proc.poll()
            if rc is not None:
                if logf:
                    logf.close()
                status = "OK" if rc == 0 else f"FAILED(rc={rc})"
                print(f"[eval_watcher] Local exec{hz} epoch={m:04d} {status}", flush=True)
                if rc == 0:
                    _done_stamp(args.out_dir, m, hz).touch()
                    _report_horizon_to_wandb(args.checkpoint_dir, args.out_dir, m, hz)
                del running_procs[hz]

        # ── Retroactive wandb for any newly-completed queue jobs ───────────────
        epoch = last_epoch_in_log(log_path)
        if epoch is None:
            print("[eval_watcher] Waiting for training log …", flush=True)
            time.sleep(args.poll_interval)
            continue

        _last_epoch_seen[0] = epoch
        max_milestone = (epoch // args.eval_every) * args.eval_every

        for m in range(args.eval_every, max_milestone + 1, args.eval_every):
            for hz in exec_horizons:
                queued = _queued_stamp(args.out_dir, m, hz).exists()
                done   = _done_stamp(args.out_dir, m, hz).exists()
                if queued and not done:
                    h_dir = _horizon_out_dir(args.out_dir, m, hz)
                    if _read_summary(h_dir) is not None:
                        _done_stamp(args.out_dir, m, hz).touch()
                        _report_horizon_to_wandb(args.checkpoint_dir, args.out_dir, m, hz)

        # ── Find next milestone to eval ────────────────────────────────────────
        milestone = None
        for m in range(args.eval_every, max_milestone + 1, args.eval_every):
            if not _milestone_fully_handled(args.out_dir, m, exec_horizons):
                milestone = m
                break

        if milestone is None:
            print(
                f"[eval_watcher] epoch={epoch}  all milestones up to {max_milestone} handled",
                flush=True,
            )
        else:
            # Evaluate the checkpoint the milestone actually refers to. Using the
            # single `latest.ckpt` path for every milestone made the epoch labels
            # meaningless -- an "epoch_0250" job that got scheduled once training
            # had reached ~950 was really re-evaluating the current policy, so
            # epoch_0250 and epoch_0500 came back byte-identical and
            # _best_milestone() was comparing one checkpoint against itself.
            milestone_ckpt = (Path(args.checkpoint_dir) / "checkpoints"
                              / f"milestone_epoch={milestone:04d}.ckpt")
            eval_ckpt = milestone_ckpt if milestone_ckpt.exists() else ckpt_path
            for hz in exec_horizons:
                if _horizon_handled(args.out_dir, milestone, hz):
                    continue  # this horizon already queued/done

                if not eval_ckpt.exists():
                    print(f"[eval_watcher] Checkpoint not found yet: {eval_ckpt}", flush=True)
                    break

                if use_queue:
                    _submit_horizon(
                        ckpt_path    = str(eval_ckpt),
                        out_dir      = args.out_dir,
                        milestone    = milestone,
                        exec_hz      = hz,
                        num_episodes = args.num_episodes,
                        max_ticks    = args.max_ticks,
                        eval_script  = args.eval_script,
                        dataset_path = args.dataset_path,
                        extra_args   = extra_eval_args,
                        required_node= required_node,
                        label_prefix = model_name,
                    )
                else:
                    if hz in running_procs:
                        continue  # previous local run still going
                    proc, logf = run_eval_local(
                        str(eval_ckpt), milestone, args.out_dir, hz,
                        num_episodes = args.num_episodes,
                        max_ticks    = args.max_ticks,
                        device       = args.device,
                        eval_script  = args.eval_script,
                        dataset_path = args.dataset_path,
                        extra_args   = extra_eval_args,
                    )
                    running_procs[hz] = (proc, logf, milestone)

        # ── End-of-training: queue final 50-episode eval of best policy ────────
        # Epochs are 0-indexed: a --num-epochs 1000 run logs 0..999 and never
        # reports 1000, so `epoch >= num_epochs` never fires and the final
        # 50-episode eval silently never happens. Compare against the last epoch
        # the run will actually emit.
        if epoch >= args.num_epochs - 1 and not final_eval_done.exists():
            result = _best_milestone(args.out_dir, exec_horizons, args.eval_every)
            if result is not None:
                best_m, best_rate = result
                # Try milestone checkpoint; fall back to latest
                milestone_ckpt = (
                    Path(args.checkpoint_dir) / "checkpoints"
                    / f"milestone_epoch={best_m:04d}.ckpt"
                )
                final_ckpt = str(milestone_ckpt) if milestone_ckpt.exists() else str(ckpt_path)
                final_out  = str(Path(args.out_dir) / "final_eval")
                os.makedirs(final_out, exist_ok=True)

                print(
                    f"[eval_watcher] Training complete (epoch={epoch}/{args.num_epochs}). "
                    f"Best milestone: epoch={best_m} mean_success={best_rate:.2f}. "
                    f"Queueing {args.final_eval_episodes}-episode final eval …",
                    flush=True,
                )

                for hz in exec_horizons:
                    if use_queue:
                        _submit_horizon(
                            ckpt_path    = final_ckpt,
                            out_dir      = final_out,
                            milestone    = best_m,
                            exec_hz      = hz,
                            num_episodes = args.final_eval_episodes,
                            max_ticks    = args.max_ticks,
                            eval_script  = args.eval_script,
                            dataset_path = args.dataset_path,
                            extra_args   = extra_eval_args,
                            required_node= required_node,
                            label_prefix = f"{model_name} FINAL",
                            is_final     = True,
                        )

                # Mark so we don't re-queue on the next poll
                final_eval_done.parent.mkdir(parents=True, exist_ok=True)
                final_eval_done.write_text(
                    json.dumps({"best_epoch": best_m, "mean_success": best_rate,
                                "ckpt": final_ckpt})
                )
            else:
                print(
                    "[eval_watcher] Training complete but no completed evals found yet "
                    "to determine best milestone — will retry next poll.",
                    flush=True,
                )

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
