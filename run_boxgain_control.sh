#!/bin/bash
# =============================================================================
# Step 1 control experiment (docs/REVIEW_RESPONSE.md F6 / R1.6)
#
# Question: is E4's reported gain caused by scale/resolution *weighting*, or
# merely by the ~4.4x inflation of total box loss that the weighting causes?
#
# All three runs are identical apart from the loss configuration, and all run
# on the same GPU so the comparison is self-consistent (the 300-epoch numbers
# in docs/REVIEW_RESPONSE.md F1 were produced on a different machine).
#
#   C0  baseline CIoU,      box=0.05     -> reference
#   C1  E4 loss flags,      box=0.05     -> reproduces the paper's E4
#   C2  baseline CIoU,      box=0.2182   -> E4's box-loss magnitude, no weighting
#
# 0.2182 = 0.05 x 4.364, the inflation factor measured on real VisDrone batches
# (not the 5.64 estimated in F6 from an assumed 50/30/20 layer split; the
#  measured baseline split is ~33/33/34).
#
# Reading it:  C2 ~ C1  -> weighting adds nothing beyond magnitude
#              C2 ~ C0  -> weighting does real work
# =============================================================================
set -e
cd "$(dirname "$0")"
PY=/home/avcom/miniconda3/envs/yolov5/bin/python
mkdir -p logs

NAMES=(C0_baseline           C1_e4_loss                                                                       C2_boxgain_control)
HYPS=( hyp.scratch-low.yaml  hyp.scratch-low.yaml                                                             hyp.boxgain-e4.yaml)
FLAGS=(""                    "--scale-aware-loss --resolution-weighting --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4"  "")

for i in 0 1 2; do
    NAME="${NAMES[$i]}"
    echo "=== $NAME | hyp=${HYPS[$i]} | flags='${FLAGS[$i]}' | $(date) ==="
    $PY train.py \
        --img 640 --batch 64 --epochs 300 \
        --data data/VisDrone_local.yaml \
        --cfg models/yolov5n.yaml \
        --hyp "data/hyps/${HYPS[$i]}" \
        --seed 42 --device 0 --workers 8 \
        --loss-log-interval 200 \
        --name "$NAME" --project runs/control \
        ${FLAGS[$i]} \
        > "logs/control_${NAME}.log" 2>&1
    echo "=== finished $NAME : $(date) ==="
done
echo "All control runs done -> runs/control/"
