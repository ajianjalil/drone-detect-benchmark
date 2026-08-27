#!/bin/bash
# Resume the Step 1 control sequence after a manual pause.
#   C1_e4_loss  — resumed from weights/last.pt (was stopped at epoch 172/300)
#   C2_boxgain_control — fresh run, baseline CIoU with box=0.2182
# Logs append to the same files so the existing monitor keeps tracking.
set -e
cd "$(dirname "$0")"
PY=/home/avcom/miniconda3/envs/yolov5/bin/python
mkdir -p logs

echo "=== C1_e4_loss (RESUME from epoch 172) | $(date) ==="
$PY train.py --resume runs/control/C1_e4_loss/weights/last.pt >> logs/control_C1_e4_loss.log 2>&1
echo "=== finished C1_e4_loss : $(date) ==="

echo "=== C2_boxgain_control | hyp=hyp.boxgain-e4.yaml | flags='' | $(date) ==="
$PY train.py \
    --img 640 --batch 64 --epochs 300 \
    --data data/VisDrone_local.yaml \
    --cfg models/yolov5n.yaml \
    --hyp data/hyps/hyp.boxgain-e4.yaml \
    --seed 42 --device 0 --workers 8 \
    --loss-log-interval 200 \
    --name C2_boxgain_control --project runs/control \
    >> logs/control_C2_boxgain_control.log 2>&1
echo "=== finished C2_boxgain_control : $(date) ==="
echo "All control runs done -> runs/control/"
