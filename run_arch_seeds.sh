#!/bin/bash
# =============================================================================
# Architecture arm: 3 seeds x {YOLOv5s, YOLOv5s+SingleSwin}
#
# Closes the exposure in docs/SUMMARY.md §4 — the architecture claim is currently
# n=1 and sits at 0.9-1.5x the 1.4% noise floor established in F-A, i.e. below the
# threshold this paper itself argues for.
#
# Also removes a confound in the original comparison: runs/train/yolov5_s_no_swin
# used batch 64 while runs/train/yolov5_s_swin used batch 16. Both arms here use
# batch 32 (VRAM-probed: 5.4 GB / 6.0 GB allocated, headroom for other tenants).
#
# NOTE: models/yolov5s_swin.yaml is RECONSTRUCTED — see its header. If these runs do
# not land near the reference 0.3484, the reconstruction is wrong and the
# YOLOv5s+SingleSwin row should be dropped rather than re-derived.
#
# Interleaved by seed, so stopping part way still leaves a complete comparison.
# =============================================================================
set -e
cd "$(dirname "$0")"
PY=/home/avcom/miniconda3/envs/yolov5/bin/python
mkdir -p logs

for SEED in 42 43 44; do
  for CFG in "A0_yolov5s:models/yolov5s.yaml" "A1_yolov5s_swin:models/yolov5s_swin.yaml"; do
    BASE="${CFG%%:*}"; MODEL="${CFG#*:}"
    NAME="${BASE}_s${SEED}"
    if [ -d "runs/arch/$NAME" ]; then echo "=== skip $NAME (exists) ==="; continue; fi
    echo "=== $NAME | seed=$SEED | cfg=$MODEL | $(date) ==="
    $PY train.py \
        --img 640 --batch 32 --epochs 300 \
        --data data/VisDrone_local.yaml \
        --cfg "$MODEL" \
        --hyp data/hyps/hyp.scratch-low.yaml \
        --seed "$SEED" --device 0 --workers 8 \
        --name "$NAME" --project runs/arch \
        > "logs/arch_${NAME}.log" 2>&1
    echo "=== finished $NAME : $(date) ==="
  done
done
echo "Architecture seed runs done -> runs/arch/"
