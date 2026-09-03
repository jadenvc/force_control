#!/usr/bin/env bash
# Waits for any in-flight finalize, then brings BOTH clean datasets to exactly
# 100 episodes, stamps the generation env onto each zarr, adds the ACP fields,
# derives the task configs, and verifies dataset <-> config <-> live env.
set -uo pipefail

PY=/local/real/jvclark/miniconda3/envs/imitation/bin/python
TELEOP=/store/real/jvclark/force_control/teleop
BASE=/store/real/jvclark/sanding_clean_shards
KP=4000
MU=0.3
N=100

# wait for the earlier finalize / merge / acp pass to drain
while [ "$(ps -eo cmd | grep -cE 'finalize_sanding[_]clean|merge[_]zarr|add_sanding[_]acp|gen_sanding[_]clean')" -gt 1 ]; do
    sleep 20
done
echo "[finish] in-flight jobs drained at $(date +%H:%M:%S)"

for MODE in timing smooth; do
    OUT=/store/real/jvclark/sanding_clean_${MODE}.zarr
    have=0
    [ -d "$OUT/data" ] && have=$(ls "$OUT/data" | grep -c episode_ || true)
    shard_total=0
    for p in $BASE/$MODE/*.zarr; do
        n=$(ls "$p/data" 2>/dev/null | grep -c episode_ || true); shard_total=$((shard_total+n))
    done
    echo "[finish] $MODE: merged=$have shards=$shard_total"

    # re-merge whenever the shards hold more than the merged copy (a top-up landed)
    if [ "$have" -ne "$N" ]; then
        echo "[finish] re-merging $MODE from $shard_total shard episodes"
        rm -rf "$OUT"
        $PY "$TELEOP/fast_merge_zarr.py" --inputs $BASE/$MODE/*.zarr --output "$OUT" --limit $N 2>&1 | tail -2
    fi

    $PY - "$OUT" "$N" <<'EOF'
import sys, zarr, numpy as np
out, n = sys.argv[1], int(sys.argv[2])
r = zarr.open(out, mode="a")
eps = sorted([e for e in r["data"].group_keys() if e.startswith("episode_")],
             key=lambda s: int(s.split("_")[1]))
if len(eps) > n:
    for e in eps[n:]:
        del r["data"][e]
    eps = eps[:n]
m = r.require_group("meta")
m.create_dataset("episode_robot0_len", overwrite=True, dtype="i8",
    data=np.array([len(r["data"][e]["ts_pose_fb_0"]) for e in eps], dtype=np.int64))
m.create_dataset("episode_rgb0_len", overwrite=True, dtype="i8",
    data=np.array([r["data"][e]["rgb_0"].shape[0] for e in eps], dtype=np.int64))
m.create_dataset("episode_wrench0_len", overwrite=True, dtype="i8",
    data=np.array([len(r["data"][e]["wrench_0"]) for e in eps], dtype=np.int64))
print(f"[finish] {out}: {len(eps)} episodes")
EOF

    WARP=1.0; [ "$MODE" = "smooth" ] && WARP=1.25
    $PY "$TELEOP/check_sanding_env_match.py" --zarr "$OUT" --stamp \
        --tool-kp $KP --friction $MU --mode "$MODE" --warp $WARP
    $PY "$TELEOP/add_sanding_acp_fields.py" "$OUT" | tail -1
    $PY "$TELEOP/make_sanding_configs.py" --dataset "$OUT" \
        --slug "sanding_${MODE}" --tool-kp $KP --friction $MU \
        --note "clean replay demos, mode=${MODE}, warp=${WARP}"
done

echo
echo "================ ENV MATCH ================"
for MODE in timing smooth; do
    OUT=/store/real/jvclark/sanding_clean_${MODE}.zarr
    for K in dp_ft acp; do
        $PY "$TELEOP/check_sanding_env_match.py" --zarr "$OUT" \
            --config "sanding_${MODE}_${K}_s8h8" 2>&1 | tail -3
    done
done

echo
echo "================ VERIFICATION ================"
for MODE in timing smooth; do
    $PY "$TELEOP/verify_sanding_clean.py" --new "/store/real/jvclark/sanding_clean_${MODE}.zarr"
done
echo "[finish] DONE $(date +%H:%M:%S)"
