#!/bin/bash
# =============================================================================
# Local ablation runner — runs all 5 experiments sequentially on local machine
# Usage:  bash run_ablation_local.sh            # run E0–E4
#         bash run_ablation_local.sh 0 3        # run only E0 and E3
# Logs:   logs/local_ablation_<name>.log
# Results: runs/ablation_local/<name>/
# =============================================================================
set -e
mkdir -p logs

# ── Experiment definitions ────────────────────────────────────────────────────
EXP_NAMES=(
    "E0_baseline"
    "E1_scale_only"
    "E2_res_only"
    "E3_both"
    "E4_both_strong"
)

EXP_FLAGS=(
    ""
    "--scale-aware-loss"
    "--resolution-weighting"
    "--scale-aware-loss --resolution-weighting"
    "--scale-aware-loss --resolution-weighting --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4"
)

# Decide which indices to run (default: all)
if [ "$#" -gt 0 ]; then
    INDICES=("$@")
else
    INDICES=(0 1 2 3 4)
fi

# ── Run loop ──────────────────────────────────────────────────────────────────
for IDX in "${INDICES[@]}"; do
    NAME="${EXP_NAMES[$IDX]}"
    FLAGS="${EXP_FLAGS[$IDX]}"
    LOGFILE="logs/local_ablation_${NAME}.log"

    echo "========================================"
    echo "Starting experiment $IDX: $NAME"
    echo "Flags: ${FLAGS:-<none>}"
    echo "Log  : $LOGFILE"
    echo "Time : $(date)"
    echo "========================================"

    python train.py \
        --img 640 \
        --batch 16 \
        --epochs 50 \
        --data VisDrone.yaml \
        --cfg models/yolov5n.yaml \
        --seed 42 \
        --loss-log-interval 50 \
        --name "$NAME" \
        --project runs/ablation_local \
        $FLAGS \
        2>&1 | tee "$LOGFILE"

    echo "========================================"
    echo "Finished $NAME : $(date)"
    echo "========================================"
done

echo ""
echo "All experiments done. Results in runs/ablation_local/"
echo "Logs in logs/local_ablation_*.log"
