#!/bin/bash
# =============================================================================
# INDISCON revision plan §1.2 — A/B/C/D run matrix
#
#   A  YOLOv5s                          P3/P4/P5      CIoU          100 ep
#   B  + P2 head                        P2/P3/P4/P5   CIoU          100 ep
#   C  + Swin@P2 (neck branch)          P2/P3/P4/P5   CIoU          100 ep
#   D  = C + custom loss                P2/P3/P4/P5   a=1.5, 4-head beta
#
# ONE epoch budget (100) across all four, per §1.2. Batch 12, seed 42 throughout.
#
# Batch 12, not 16: arm C (P2 head + Swin@P2) OOMs at batch 16 on this 16 GB card —
# it reached 14.91 GiB allocated and died on the first backward pass. Batch 12 peaks
# at 12.8 GB and cleared a full epoch including validation. ALL arms use 12 so that
# B and C differ only by the SwinStage, per the plan pre-flight check.
# B and C differ by exactly one module (the SwinStage on the P2 neck branch).
# --save-period 10 so per-class AP can be evaluated at any 10-epoch interval.
#
# Order: B and C first (they carry the §1.6 decision rule), then A, then D.
#
# NOTE: A is RUN, not reused. The plan assumed Table I was 100 epochs; it is
# actually 300 (every opt.yaml reads epochs: 300, and each results.csv has 300
# rows). Reusing it would reintroduce the mixed-budget flaw the plan exists to fix.
# =============================================================================
set -e
cd "$(dirname "$0")"
PY=/home/avcom/miniconda3/envs/yolov5/bin/python
mkdir -p logs

BETA="3.0 2.0 1.0 0.4"   # beta-mid per §1.4, analogous ratio to the current 3-head [3.0,1.0,0.4]

run () {  # name cfg extra_flags
  local NAME=$1 MODEL=$2; shift 2
  if [ -d "runs/indiscon/$NAME" ]; then echo "=== skip $NAME (exists) ==="; return; fi
  echo "=== $NAME | cfg=$MODEL | flags='$*' | $(date) ==="
  $PY train.py \
      --img 640 --batch 12 --epochs 100 --save-period 10 \
      --data data/VisDrone_local.yaml \
      --cfg "$MODEL" \
      --hyp data/hyps/hyp.scratch-low.yaml \
      --seed 42 --device 0 --workers 8 \
      --loss-log-interval 200 \
      --name "$NAME" --project runs/indiscon \
      "$@" \
      > "logs/indiscon_${NAME}.log" 2>&1
  echo "=== finished $NAME : $(date) ==="
}

run B_p2head        models/yolov5s_p2.yaml
run C_swin_p2       models/yolov5s_p2_swinP2.yaml
run A_baseline      models/yolov5s.yaml
run D_swin_p2_loss  models/yolov5s_p2_swinP2.yaml \
    --scale-aware-loss --resolution-weighting --scale-alpha 1.5 --resolution-beta $BETA

echo "INDISCON A/B/C/D done -> runs/indiscon/"
