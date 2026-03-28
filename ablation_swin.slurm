#!/bin/bash
# =============================================================================
# Swin Transformer ablation — scale-aware + resolution-aware loss
# Runs 4 experiments as a SLURM job array, one at a time (queued sequentially).
#
# Usage:
#   sbatch ablation_swin.slurm                 # run all 4 (E0–E3)
#   sbatch --array=0   ablation_swin.slurm     # run only E0
#   sbatch --array=0,2 ablation_swin.slurm     # run E0 and E2
#
# Experiment index map:
#   0 → E0  swin_small_baseline    (Swin-Small, original loss)
#   1 → E1  swin_small_new_loss    (Swin-Small, scale-aware + resolution strong)
#   2 → E2  swin_medium_baseline   (Swin-Medium, original loss)
#   3 → E3  swin_medium_new_loss   (Swin-Medium, scale-aware + resolution strong)
#
# Queue behaviour:
#   --array=0-3%1 means all 4 jobs are submitted at once but only 1 runs at a
#   time.  Remove %1 to allow parallel execution if multiple GPUs are available.
# =============================================================================
#SBATCH --job-name=ablation_swin
#SBATCH --array=0-3%1
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/ablation_swin_%A_%a.out
#SBATCH --error=logs/ablation_swin_%A_%a.err

set -e
mkdir -p logs

# ── Experiment definitions ────────────────────────────────────────────────────

EXP_NAMES=(
    "E0_swin_small_baseline"
    "E1_swin_small_new_loss"
    "E2_swin_medium_baseline"
    "E3_swin_medium_new_loss"
)

# Model config per experiment
EXP_MODELS=(
    "models/yolov5s_swin2.yaml"
    "models/yolov5s_swin2.yaml"
    "models/yolov5m_swin.yaml"
    "models/yolov5m_swin.yaml"
)

# Batch size per experiment (A100 80GB — ample VRAM for both sizes)
EXP_BATCH=(
    64
    64
    32
    32
)

# Extra train.py loss flags per experiment (empty = original loss)
EXP_FLAGS=(
    ""
    "--scale-aware-loss --resolution-weighting --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4"
    ""
    "--scale-aware-loss --resolution-weighting --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4"
)

NAME="${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}"
CFG="${EXP_MODELS[$SLURM_ARRAY_TASK_ID]}"
BATCH="${EXP_BATCH[$SLURM_ARRAY_TASK_ID]}"
FLAGS="${EXP_FLAGS[$SLURM_ARRAY_TASK_ID]}"

# ── Environment info ──────────────────────────────────────────────────────────
echo "========================================"
echo "Job array ID : $SLURM_ARRAY_JOB_ID"
echo "Task index   : $SLURM_ARRAY_TASK_ID"
echo "Experiment   : $NAME"
echo "Model config : $CFG"
echo "Batch size   : $BATCH"
echo "Extra flags  : ${FLAGS:-<none>}"
echo "Node         : $(hostname)"
echo "Started      : $(date)"
echo "========================================"
nvidia-smi

# ── Conda environment ─────────────────────────────────────────────────────────
source /dist_home/ak_ajithkumar/miniconda3/etc/profile.d/conda.sh
conda activate yolov5

cd /dist_home/ak_ajithkumar/ondemand/ajith_work/drone-detect-benchmark
git pull origin main

# Confirm Python + GPU
which python
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
EOF

# ── Training ──────────────────────────────────────────────────────────────────
python train.py \
  --img 640 \
  --batch "$BATCH" \
  --epochs 300 \
  --data data/VisDrone_cluster.yaml \
  --cfg "$CFG" \
  --device 0 \
  --seed 42 \
  --loss-log-interval 50 \
  --name "$NAME" \
  --project runs/ablation_swin \
  $FLAGS

echo "========================================"
echo "Experiment $NAME finished : $(date)"
echo "========================================"
