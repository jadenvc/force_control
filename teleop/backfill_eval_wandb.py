#!/usr/bin/env python3
"""Push already-completed eval results into their training wandb runs.

eval_watcher logged eval metrics with wandb.init(resume="must"). Training writes
wandb_run_id.txt locally before/independently of the run being registered
server-side, so a valid-looking id can name a run wandb has never seen; "must"
then refuses and the metrics are dropped with only a line in watcher.log. Result:
eval/success_rate was absent from wandb for every sanding run even though the id
files were present and the summaries were on disk.

The watcher is fixed going forward (resume="allow"), but the evals that already
ran will never be retried -- their .eval_done stamps are set. This walks the eval
tree and logs them.

    python backfill_eval_wandb.py --dry-run
    python backfill_eval_wandb.py
    python backfill_eval_wandb.py --runs smooth_dp_ft_1k,timing_dp_ft_1k
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

EVALS = "/store/real/jvclark/PyriteML/evals"
CKPT_ROOT = "/local/real/jvclark/training_outputs"


def find_run_id(run_name: str, eval_run_dir: str | None = None) -> str | None:
    """Resolve the wandb id of the training run these evals actually came from.

    wandb_run_id.txt lives in hydra's dir, not pyrite_train.sh's -- and a run that
    was relaunched has SEVERAL hydra dirs, each with a different, valid-looking
    id. Globbing and taking the first match silently attributes the results to a
    dead launch (dp_1k has enqoui94 from a killed run and jd6gy5ub from the live
    one). The eval dir's own pids.txt records the ckpt_dir the jobs ran against,
    which is the authoritative link; fall back to the dir with the most training
    logged.
    """
    if eval_run_dir:
        pids = Path(eval_run_dir) / "pids.txt"
        if pids.exists():
            for line in pids.read_text().splitlines():
                if line.startswith("ckpt_dir="):
                    f = Path(line.split("=", 1)[1].strip()) / "wandb_run_id.txt"
                    if f.exists():
                        return f.read_text().strip()

    best, best_sz = None, -1
    for d in glob.glob(os.path.join(CKPT_ROOT, f"*_{run_name}")):
        f, lg = Path(d) / "wandb_run_id.txt", Path(d) / "logs.json.txt"
        if f.exists() and lg.exists():
            sz = lg.stat().st_size
            if sz > best_sz:
                best, best_sz = f.read_text().strip(), sz
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default=EVALS)
    ap.add_argument("--runs", default="", help="comma list; default = all sanding runs")
    ap.add_argument("--project", default="pyrite-force-control")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    only = [r for r in args.runs.split(",") if r]
    pat = os.path.join(args.evals, "sanding*", "*", "**", "summary.json")
    found = {}
    eval_dirs: dict[str, str] = {}
    OBSOLETE_RUNS = ("acp_1k", "smooth_acp_1k", "timing_acp_1k")   # pre-relabel ACP
    for f in glob.glob(pat, recursive=True):
        if "obsolete" in f:
            continue
        parts = f.split(os.sep)
        run_dir = next((p for p in parts if p.startswith("2026-")), None)
        if run_dir is None:
            continue
        run = run_dir.split("_s8h8_")[-1]
        if run in OBSOLETE_RUNS:
            continue
        if only and run not in only:
            continue
        eval_run_dir = f.split(os.sep + run_dir + os.sep)[0] + os.sep + run_dir
        milestone = int(parts[-3].replace("epoch_", ""))
        hz = int(parts[-2].replace("exec", ""))
        is_final = "final_eval" in f
        found.setdefault(run, []).append((milestone, hz, is_final, f))
        eval_dirs[run] = eval_run_dir

    if not found:
        print("no summaries found")
        return 1

    for run, items in sorted(found.items()):
        rid = find_run_id(run, eval_dirs.get(run))
        print(f"\n{run}  wandb_run_id={rid or 'MISSING'}  ({len(items)} summaries)")
        if rid is None:
            print("  skip: no wandb_run_id.txt")
            continue
        items.sort()
        if args.dry_run:
            for m, hz, fin, f in items:
                s = json.load(open(f))
                n = len(s.get("results", []))
                sc = sum(1 for r in s.get("results", []) if r.get("success"))
                tag = "FINAL" if fin else "     "
                print(f"  {tag} epoch={m:<5} exec{hz}  {sc}/{n} = {100*sc/max(n,1):.0f}%")
            continue

        import wandb
        run_obj = wandb.init(id=rid, resume="allow", reinit=True, project=args.project,
                             settings=wandb.Settings(init_timeout=300))
        try:
            for m, hz, fin, f in items:
                s = json.load(open(f))
                res = s.get("results", [])
                n = len(res)
                sc = sum(1 for r in res if r.get("success"))
                nb = sum(1 for r in res if r.get("broken"))
                pf = [float(r.get("peak_force_n") or 0.0) for r in res]
                # finals go under their own prefix so a 50-episode point is never
                # mistaken for an 8-episode milestone point on the same axis
                pre = f"eval_final/exec{hz}" if fin else f"eval/exec{hz}"
                d = {
                    f"{pre}/epoch": m,
                    f"{pre}/success_rate": sc / max(n, 1),
                    f"{pre}/n_episodes": n,
                    f"{pre}/broken_rate": nb / max(n, 1),
                    f"{pre}/avg_coverage": s.get("avg_coverage", 0.0),
                    f"{pre}/avg_ticks": s.get("avg_ticks", 0.0),
                }
                if pf:
                    d[f"{pre}/peak_force_mean_N"] = sum(pf) / len(pf)
                    d[f"{pre}/peak_force_max_N"] = max(pf)
                wandb.log(d, step=m)
                print(f"  logged epoch={m:<5} exec{hz}  {sc}/{n}"
                      + ("  [FINAL]" if fin else ""))
        finally:
            run_obj.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
