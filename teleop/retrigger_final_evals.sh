#!/usr/bin/env bash
# Fire each run's 50-episode final eval as soon as THAT run is ready, rather than
# waiting for the whole cluster to go quiet.
#
# A run is ready when: its training process is gone AND it has no pending/running
# milestone jobs left. The stamp .final_eval_queued is then removed so its watcher
# re-picks the best milestone (with all milestone results now in) and queues the
# final against that checkpoint.
#
# Per-run on purpose: the earlier global version would have parked the finished DP
# runs' finals behind ~11 h of freshly-started ACP training.
set -uo pipefail

PY=/local/real/jvclark/miniconda3/envs/imitation/bin/python
TELEOP=/store/real/jvclark/force_control/teleop
Q=/store/real/jvclark/eval_queue/queue.json
E=/store/real/jvclark/PyriteML/evals

# Only these runs are live; anything else in evals/ is retired or empty.
LIVE=(dp_1k timing_dp_ft_1k smooth_dp_ft_1k acp_raw_1k acp_smooth_ema_1k acp_smooth_maxpool_1k)
done_runs=""

busy_for_run() {   # $1 = eval run dir basename
    $PY - "$Q" "$1" <<'EOF'
import json,sys
q=json.load(open(sys.argv[1])); run=sys.argv[2]
print(sum(1 for j in q
          if run in j.get("out_dir","")
          and j.get("status") in ("pending","running")
          and "final_eval" not in j.get("out_dir","")))
EOF
}

echo "[retrigger] per-run mode; polling every 120s"
for _ in $(seq 1 720); do          # ~24 h ceiling
    any_left=0
    for rn in "${LIVE[@]}"; do
        case " $done_runs " in *" $rn "*) continue ;; esac
        # locate this run's eval dir (exact trailing match on the run name)
        d=$(ls -d "$E"/*/2026-09-0*_"$rn" 2>/dev/null | tail -1)
        [ -n "$d" ] || { any_left=1; continue; }
        run=$(basename "$d")
        # still training?
        if pgrep -f "name=$rn " >/dev/null 2>&1; then any_left=1; continue; fi
        # milestones still in flight?
        if [ "$(busy_for_run "$run")" != "0" ]; then any_left=1; continue; fi

        st="$d/final_eval/.final_eval_queued"
        if [ -f "$st" ]; then
            rm -f "$st"
            echo "[retrigger] $run ready -> cleared stamp, restarting its watcher"
        else
            echo "[retrigger] $run ready (no stale stamp); restarting its watcher"
        fi
        # per-run: restarting ALL watchers here previously resurrected retired
        # runs' watchers and spawned duplicates (5 on one obsolete run)
        "$TELEOP/restart_eval_watchers.sh" "$rn" 2>/dev/null | grep -F "$rn" || true
        done_runs="$done_runs $rn"
    done
    [ "$any_left" = "0" ] && [ -n "$done_runs" ] && {
        echo "[retrigger] all known runs handled: $done_runs"; break; }
    sleep 120
done
echo "[retrigger] exit $(date +%H:%M:%S)"
