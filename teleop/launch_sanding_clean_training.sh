#!/usr/bin/env bash
# Launch the four clean-dataset training runs (timing/smooth x DP+FT/ACP).
#
# Refuses to start a run whose task config does not agree with the env its
# dataset was generated in -- the whole point of the clean regeneration is that
# eval reproduces the training env (tool_kp 4000, sliding friction 0.3), and a
# silent mismatch there is exactly the failure mode that made the previous
# sanding eval meaningless.
#
#   ./launch_sanding_clean_training.sh [EPOCHS]
set -uo pipefail

EPOCHS="${1:-1000}"
PY=/local/real/jvclark/miniconda3/envs/imitation/bin/python
TELEOP=/store/real/jvclark/force_control/teleop
PYRITE=/store/real/jvclark/PyriteML

# GPUs ordered by free memory, most free first (we are deliberately co-locating
# with other users' jobs, so memory headroom is the thing to optimise).
#
# IMPORTANT: pinning is done with CUDA_VISIBLE_DEVICES, not training.device.
# train_diffusion_unet_image_workspace.py drives placement through HuggingFace
# Accelerator and its `device = torch.device(cfg.training.device)` line is
# commented out, so `training.device=cuda:N` is dead code -- every run silently
# lands on physical GPU 0. That is what stacked six runs onto one card (slow
# epochs) and then OOMed this set at ~94 GiB.
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                    | sort -t, -k2 -n | cut -d, -f1 | tr -d ' ')
echo "[launch] GPU preference order (most free first): ${GPUS[*]}"

i=0
for MODE in smooth timing; do
    ZARR=/store/real/jvclark/sanding_clean_${MODE}.zarr
    for KIND in dp_ft acp; do
        CFG="sanding_${MODE}_${KIND}_s8h8"
        if [ ! -f "$PYRITE/diffusion_policy/config/task/${CFG}.yaml" ]; then
            echo "[launch] SKIP $CFG -- config missing"; continue
        fi
        if ! $PY "$TELEOP/check_sanding_env_match.py" --zarr "$ZARR" --config "$CFG" >/dev/null 2>&1; then
            echo "[launch] SKIP $CFG -- ENV MISMATCH:"
            $PY "$TELEOP/check_sanding_env_match.py" --zarr "$ZARR" --config "$CFG" 2>&1 | grep -i mismatch
            continue
        fi
        GPU=${GPUS[$(( i % ${#GPUS[@]} ))]}
        RUN="${MODE}_${KIND}_1k"
        echo "[launch] $CFG -> cuda:$GPU  (run $RUN, $EPOCHS epochs)  ENV MATCH OK"
        # setsid puts the run in its own session, so it survives the launching
        # shell's process group being torn down. Without it, an unrelated tool
        # call timing out took all four runs with it ~8 minutes in, after they
        # had already loaded the dataset and written sparse_normalizer.pkl.
        LAUNCH_LOG="/tmp/pyrite_launch_${RUN}.log"
        ( cd "$PYRITE" && \
          CONDA_PREFIX=/local/real/jvclark/miniconda3/envs/imitation \
          PYRITE_NUM_EPOCHS="$EPOCHS" \
          CUDA_VISIBLE_DEVICES="$GPU" \
          setsid ./pyrite_train.sh "$CFG" "$RUN" "cuda:0" "training.num_epochs=$EPOCHS" \
          < /dev/null > "$LAUNCH_LOG" 2>&1 )
        grep -E 'Training PID|Eval watcher PID|eval_out' "$LAUNCH_LOG" || true
        i=$((i+1))
        sleep 5
    done
done
echo "[launch] done"
