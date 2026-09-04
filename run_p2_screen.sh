#!/bin/bash
# =============================================================================
# 50-epoch screening: does moving detection/attention to high resolution help?
#
# Motivation: at stride 32 a VisDrone pedestrian is 0.17x0.53 feature cells, so
# the existing P5 SwinStage cannot see small objects at all. This screens the two
# fixes: detect at stride 4 (P2 head), and attend at stride 8 after fusion (SwinP3).
#
#   S0  yolov5s                        baseline, batch-matched
#   S1  yolov5s + P2 head              high-res detection, no attention
#   S2  yolov5s + SwinP3               attention at P3 in the neck, window 8, depth 4
#   S3  yolov5s + P2 head + SwinP3     both
#
# Single run per config (screening only — no seeds). batch 16 throughout because
# S3 peaks at 14.1 GB at batch 32 and would OOM; a batch-matched baseline is
# included so every contrast has identical arms.
#
# If a config is positive here, it goes to 300 epochs x 3 seeds.
# =============================================================================
set -e
cd "$(dirname "$0")"
PY=/home/avcom/miniconda3/envs/yolov5/bin/python
mkdir -p logs

for CFG in \
    "S0_base:models/yolov5s.yaml" \
    "S1_p2head:models/yolov5s_p2.yaml" \
    "S2_swinP3:models/yolov5s_swinP3.yaml" \
    "S3_p2_swinP3:models/yolov5s_p2_swinP3.yaml" ; do
  NAME="${CFG%%:*}"; MODEL="${CFG#*:}"
  if [ -d "runs/p2screen/$NAME" ]; then echo "=== skip $NAME (exists) ==="; continue; fi
  echo "=== $NAME | cfg=$MODEL | $(date) ==="
  $PY train.py \
      --img 640 --batch 16 --epochs 50 --save-period 10 \
      --data data/VisDrone_local.yaml \
      --cfg "$MODEL" \
      --hyp data/hyps/hyp.scratch-low.yaml \
      --seed 42 --device 0 --workers 8 \
      --name "$NAME" --project runs/p2screen \
      > "logs/p2s_${NAME}.log" 2>&1
  echo "=== finished $NAME : $(date) ==="
done
echo "P2 screen done -> runs/p2screen/"
