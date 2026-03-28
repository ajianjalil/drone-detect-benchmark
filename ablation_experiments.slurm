#!/bin/bash
# =============================================================================
# Ablation experiments — Scale-aware + Resolution-aware loss
# Runs 5 experiments as a SLURM job array, one at a time (queued sequentially).
#
# Usage:
#   sbatch ablation_experiments.slurm            # run all 5 (E0–E4)
#   sbatch --array=0   ablation_experiments.slurm  # run only E0 (baseline)
#   sbatch --array=0,3 ablation_experiments.slurm  # run E0 and E3
#
# Experiment index map:
#   0 → E0  baseline          (original YOLOv5 loss, no modifications)
#   1 → E1  scale_only        (scale-aware weight only)
#   2 → E2  res_only          (resolution weighting only)
#   3 → E3  both              (scale-aware + resolution weighting, default beta)
#   4 → E4  both_strong       (scale-aware + resolution weighting, stronger weights)
#
# Queue behaviour:
#   --array=0-4%1  means all 5 jobs are submitted at once but only 1 runs at a
#   time.  SLURM queues them and starts the next as soon as the GPU is free.
#   Remove the %1 to allow parallel execution if multiple GPUs are available.
# =============================================================================
#SBATCH --job-name=ablation
#SBATCH --array=0-4%1
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err

set -e
mkdir -p logs

# ── Experiment definitions ────────────────────────────────────────────────────
# Each index corresponds to one ablation configuration.

EXP_NAMES=(
    "E0_baseline"
    "E1_scale_only"
    "E2_res_only"
    "E3_both"
    "E4_both_strong"
)

# Extra train.py flags per experiment (empty string = no extra flags)
EXP_FLAGS=(
    ""
    "--scale-aware-loss"
    "--resolution-weighting"
    "--scale-aware-loss --resolution-weighting"
    "--scale-aware-loss --resolution-weighting --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4"
)

NAME="${EXP_NAMES[$SLURM_ARRAY_TASK_ID]}"
FLAGS="${EXP_FLAGS[$SLURM_ARRAY_TASK_ID]}"

# ── Environment info ──────────────────────────────────────────────────────────
echo "========================================"
echo "Job array ID : $SLURM_ARRAY_JOB_ID"
echo "Task index   : $SLURM_ARRAY_TASK_ID"
echo "Experiment   : $NAME"
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
  --batch 64 \
  --epochs 50 \
  --data VisDrone.yaml \
  --cfg models/yolov5n.yaml \
  --device 0 \
  --seed 42 \
  --loss-log-interval 50 \
  --name "$NAME" \
  --project runs/ablation \
  $FLAGS

echo "========================================"
echo "Experiment $NAME finished : $(date)"
echo "========================================"
