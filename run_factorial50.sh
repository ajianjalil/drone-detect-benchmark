#!/bin/bash
# =============================================================================
# 2x2 factorial at 50 epochs: {no Swin, +Swin} x {baseline CIoU, E4 loss}
#
# Answers three things at once:
#   1. Does the loss x architecture COMBINATION do anything the parts do not?
#      (the paper's Table III tested this at n=1 and found net-zero)
#   2. Does the +4.8% appear in a *completed 50-epoch schedule*? The 300-epoch runs
#      only sampled epoch 50 of a 300-epoch LR schedule, which is a different regime
#      (lr0 still 0.0084 vs ~0.0002 annealed). This is the honest replication.
#   3. Per-class trend over training — --save-period 10 keeps checkpoints at
#      10/20/30/40/50 so per-class AP can be evaluated at each, which the earlier
#      runs cannot support (they used save_period: -1).
#
# 3 seeds x 4 configs = 12 runs, ~22 h. yolov5s scale, batch 32 throughout to match
# the architecture arm (runs/arch) and keep both arms of every contrast identical.
# =============================================================================
set -e
cd "$(dirname "$0")"
PY=/home/avcom/miniconda3/envs/yolov5/bin/python
mkdir -p logs

E4FLAGS="--scale-aware-loss --resolution-weighting --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4"

for SEED in 42 43 44; do
  for CFG in \
      "F0_s_base:models/yolov5s.yaml:" \
      "F1_s_e4:models/yolov5s.yaml:$E4FLAGS" \
      "F2_swin_base:models/yolov5s_swin.yaml:" \
      "F3_swin_e4:models/yolov5s_swin.yaml:$E4FLAGS" ; do
    BASE="${CFG%%:*}"; REST="${CFG#*:}"; MODEL="${REST%%:*}"; FLAGS="${REST#*:}"
    NAME="${BASE}_s${SEED}"
    if [ -d "runs/factorial50/$NAME" ]; then echo "=== skip $NAME (exists) ==="; continue; fi
    echo "=== $NAME | seed=$SEED | cfg=$MODEL | flags='${FLAGS:-<none>}' | $(date) ==="
    $PY train.py \
        --img 640 --batch 32 --epochs 50 --save-period 10 \
        --data data/VisDrone_local.yaml \
        --cfg "$MODEL" \
        --hyp data/hyps/hyp.scratch-low.yaml \
        --seed "$SEED" --device 0 --workers 8 \
        --name "$NAME" --project runs/factorial50 \
        $FLAGS \
        > "logs/f50_${NAME}.log" 2>&1
    echo "=== finished $NAME : $(date) ==="
  done
done
echo "Factorial-50 done -> runs/factorial50/"
