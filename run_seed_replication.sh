#!/bin/bash
# =============================================================================
# 3-seed replication of the Step 1 control (answers R2: statistical significance)
#
# Seed 42 already exists as runs/control/{C0_baseline,C1_e4_loss,C2_boxgain_control}.
# This adds seeds 43 and 44 for the same three configs -> n=3 per config, so the
# C0/C1/C2 gaps can be reported with a spread instead of as bare point estimates.
#
# 6 runs x ~6h = ~36h. Order is interleaved by seed so that if it is stopped part
# way, whatever finished is still a complete comparison at that seed rather than
# three runs of one config.
# =============================================================================
set -e
cd "$(dirname "$0")"
PY=/home/avcom/miniconda3/envs/yolov5/bin/python
mkdir -p logs

E4FLAGS="--scale-aware-loss --resolution-weighting --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4"

for SEED in 43 44; do
  for CFG in "C0_baseline:hyp.scratch-low.yaml:" "C1_e4_loss:hyp.scratch-low.yaml:$E4FLAGS" "C2_boxgain_control:hyp.boxgain-e4.yaml:"; do
    BASE="${CFG%%:*}"; REST="${CFG#*:}"; HYP="${REST%%:*}"; FLAGS="${REST#*:}"
    NAME="${BASE}_s${SEED}"
    if [ -d "runs/control/$NAME" ]; then echo "=== skip $NAME (exists) ==="; continue; fi
    echo "=== $NAME | seed=$SEED | hyp=$HYP | flags='${FLAGS:-<none>}' | $(date) ==="
    $PY train.py \
        --img 640 --batch 64 --epochs 300 \
        --data data/VisDrone_local.yaml \
        --cfg models/yolov5n.yaml \
        --hyp "data/hyps/$HYP" \
        --seed "$SEED" --device 0 --workers 8 \
        --loss-log-interval 200 \
        --name "$NAME" --project runs/control \
        $FLAGS \
        > "logs/control_${NAME}.log" 2>&1
    echo "=== finished $NAME : $(date) ==="
  done
done
echo "Seed replication done -> runs/control/"
