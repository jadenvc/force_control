#!/usr/bin/env bash
# Repoint the running runs' eval_watchers at hydra's checkpoint dir.
#
# pyrite_train.sh handed each watcher the ${TASK_CFG}_${RUN} directory, which only
# ever receives stdout; logs.json.txt and checkpoints/ live in hydra's
# ${task_name}_${name} directory next to it. The watchers therefore blocked on
# "Waiting for training log ..." for the whole run while checkpoints accumulated.
# Training is unaffected, so only the watchers need restarting.
set -uo pipefail

PY=/local/real/jvclark/miniconda3/envs/imitation/bin/python
TELEOP=/store/real/jvclark/force_control/teleop
PYRITE=/store/real/jvclark/PyriteML
ROOT=/local/real/jvclark/training_outputs
EPOCHS=1000

# run_name : task_cfg : eval_group
RUNS=(
  "smooth_dp_ft_1k:sanding_smooth_dp_ft_s8h8:sanding_smooth"
  "timing_dp_ft_1k:sanding_timing_dp_ft_s8h8:sanding_timing"
  "dp_1k:sanding1_dp_ft_s8h8:sanding1"
  "acp_raw_1k:sanding_acp_raw_acp_s8h8:sanding"
  "acp_smooth_ema_1k:sanding_acp_smooth_ema_acp_s8h8:sanding"
  "acp_smooth_maxpool_1k:sanding_acp_smooth_maxpool_acp_s8h8:sanding"
)

ONLY="${1:-}"
for spec in "${RUNS[@]}"; do
    [ -n "$ONLY" ] && [ "${spec%%:*}" != "$ONLY" ] && continue
    RUN="${spec%%:*}"; rest="${spec#*:}"; CFG="${rest%%:*}"; GROUP="${rest##*:}"
    YAML="$PYRITE/diffusion_policy/config/task/${CFG}.yaml"

    # newest dir ending in _$RUN that actually holds hydra's output
    CKPT=$(ls -td "$ROOT"/*_"$RUN"/ 2>/dev/null | sed 's|/$||' | while read -r d; do
               [ -f "$d/logs.json.txt" ] && echo "$d"; done | head -1)
    if [ -z "$CKPT" ]; then echo "[$RUN] no dir with logs.json.txt -- skip"; continue; fi

    OUT=$(ls -d "$PYRITE"/evals/"$GROUP"/*_"$CFG"_"$RUN" 2>/dev/null | tail -1)
    if [ -z "$OUT" ]; then echo "[$RUN] no eval out dir -- skip"; continue; fi

    # stop the stuck watcher
    OLD=$(grep -oP 'watcher_pid=\K[0-9]+' "$OUT/pids.txt" 2>/dev/null || true)
    [ -n "$OLD" ] && kill "$OLD" 2>/dev/null && echo "[$RUN] stopped stuck watcher $OLD"

    read -r SCRIPT DATASET MAXTICKS EXTRA <<EOF
$($PY - "$YAML" <<'PYEOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
e = c.get("eval", {})
print(e.get("script",""), c.get("dataset_path",""), e.get("max_ticks",20000),
      " ".join(str(a) for a in e.get("extra_args", [])))
PYEOF
)
EOF

    MUJOCO_GL=egl setsid nohup "$PY" "$TELEOP/eval_watcher.py" \
        --checkpoint-dir "$CKPT" --out-dir "$OUT" \
        --eval-every 250 --exec-horizons 4,8 --num-episodes 8 \
        --max-ticks "$MAXTICKS" --num-epochs "$EPOCHS" --final-eval-episodes 50 \
        --eval-script "$SCRIPT" --dataset-path "$DATASET" $EXTRA \
        < /dev/null >> "$OUT/watcher.log" 2>&1 &
    NEW=$!
    sed -i "s|^watcher_pid=.*|watcher_pid=$NEW|; s|^ckpt_dir=.*|ckpt_dir=$CKPT|" "$OUT/pids.txt" 2>/dev/null
    echo "[$RUN] watcher $NEW -> $(basename "$CKPT")  ($(ls "$CKPT/checkpoints" 2>/dev/null | wc -l) ckpts)"
done
