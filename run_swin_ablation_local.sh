#!/bin/bash
# =============================================================================
# Swin ablation — local run on RTX 4000 Pro Blackwell
# Runs 4 experiments sequentially, logging each to logs/swin_<name>.log
#
# Usage:
#   bash run_swin_ablation_local.sh            # run all 4 (E0–E3)
#   bash run_swin_ablation_local.sh 0          # run only E0
#   bash run_swin_ablation_local.sh 0 1        # run E0 and E1
#   nohup bash run_swin_ablation_local.sh &    # run in background
#
# Experiment map:
#   0 → E0  swin_small_baseline    (yolov5s_swin2, original loss)
#   1 → E1  swin_small_new_loss    (yolov5s_swin2, scale-aware + res strong)
#   2 → E2  swin_medium_baseline   (yolov5m_swin,  original loss)
#   3 → E3  swin_medium_new_loss   (yolov5m_swin,  scale-aware + res strong)
# Machine switch:
#   omen      → data/VisDrone.yaml          (path: /mnt/mydrive/ajith/data_set/VisDrone)
#   blackwell → data/VisDrone_blackwell.yaml (path: /home/avcom/Documents/ajith_personal/dataset/VisDrone)
# =============================================================================
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

mkdir -p logs

# ── Machine selection ─────────────────────────────────────────────────────────
# Set MACHINE=blackwell when running on the RTX 4000 Pro Blackwell machine.
# Set MACHINE=omen (or leave unset) for the local OMEN laptop.
MACHINE="${MACHINE:-omen}"

case "$MACHINE" in
    blackwell)
        DATA="data/VisDrone_blackwell.yaml"
        ;;
    omen)
        DATA="data/VisDrone.yaml"
        ;;
    cluster)
        DATA="data/VisDrone_cluster.yaml"
        ;;
    *)
        echo "ERROR: unknown MACHINE='$MACHINE'. Use: omen | blackwell | cluster"
        exit 1
        ;;
esac

echo "Machine : $MACHINE"
echo "Data    : $DATA"

# ── Experiment definitions ────────────────────────────────────────────────────

EXP_NAMES=(
    "E0_swin_small_baseline"
    "E1_swin_small_new_loss"
    "E2_swin_medium_baseline"
    "E3_swin_medium_new_loss"
)

EXP_MODELS=(
    "models/yolov5s_swin2.yaml"
    "models/yolov5s_swin2.yaml"
    "models/yolov5m_swin.yaml"
    "models/yolov5m_swin.yaml"
)

# Adjust if you hit OOM: try 32/16 for small/medium
EXP_BATCH=(
    16
    16
    16
    16
)

EXP_FLAGS=(
    ""
    "--scale-aware-loss --resolution-weighting --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4"
    ""
    "--scale-aware-loss --resolution-weighting --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4"
)

# ── Select which experiments to run ──────────────────────────────────────────

if [ $# -gt 0 ]; then
    INDICES=("$@")
else
    INDICES=(0 1 2 3)
fi

# ── Run ───────────────────────────────────────────────────────────────────────

for IDX in "${INDICES[@]}"; do
    NAME="${EXP_NAMES[$IDX]}"
    CFG="${EXP_MODELS[$IDX]}"
    BATCH="${EXP_BATCH[$IDX]}"
    FLAGS="${EXP_FLAGS[$IDX]}"
    LOG="logs/swin_${NAME}.log"

    echo "========================================"
    echo "Experiment : $NAME"
    echo "Model      : $CFG"
    echo "Batch      : $BATCH"
    echo "Flags      : ${FLAGS:-<none>}"
    echo "Log        : $LOG"
    echo "Started    : $(date)"
    echo "========================================"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

    python train.py \
        --img 640 \
        --batch "$BATCH" \
        --epochs 300 \
        --data "$DATA" \
        --cfg "$CFG" \
        --device 0 \
        --seed 42 \
        --loss-log-interval 50 \
        --name "$NAME" \
        --project runs/ablation_swin \
        $FLAGS \
        2>&1 | tee "$LOG"

    echo "========================================"
    echo "Finished $NAME : $(date)"
    echo "========================================"
done

echo "All experiments done."
