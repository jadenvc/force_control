#!/usr/bin/env bash
# Merge the per-worker clean-demo shards into one zarr per mode, trim to N
# episodes, add the ACP fields, and derive the PyriteML task configs.
#
#   ./finalize_sanding_clean.sh [N_EPISODES]
set -euo pipefail

N="${1:-100}"
PY=/local/real/jvclark/miniconda3/envs/imitation/bin/python
TELEOP=/store/real/jvclark/force_control/teleop
BASE=/store/real/jvclark/sanding_clean_shards
KP=4000
MU=0.3

for MODE in timing smooth; do
    OUT=/store/real/jvclark/sanding_clean_${MODE}.zarr
    shards=$(ls -d $BASE/$MODE/shard_*.zarr 2>/dev/null || true)
    [ -z "$shards" ] && { echo "no shards for $MODE"; continue; }

    echo "=============== $MODE ==============="
    rm -rf "$OUT"
    $PY "$TELEOP/merge_zarr.py" --inputs $shards --output "$OUT"

    # trim to exactly N episodes (merge renumbers 0..M-1)
    $PY - "$OUT" "$N" <<'EOF'
import shutil, sys, zarr, numpy as np
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
print(f"  {out}: {len(eps)} episodes")
EOF

    $PY "$TELEOP/add_sanding_acp_fields.py" "$OUT" | tail -1
    $PY "$TELEOP/make_sanding_configs.py" --dataset "$OUT" \
        --slug "sanding_${MODE}" --tool-kp $KP --friction $MU \
        --note "clean replay demos, mode=${MODE}"
done

echo
echo "=============== verification ==============="
for MODE in timing smooth; do
    OUT=/store/real/jvclark/sanding_clean_${MODE}.zarr
    [ -d "$OUT" ] && $PY "$TELEOP/verify_sanding_clean.py" --new "$OUT"
done
