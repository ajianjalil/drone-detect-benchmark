#!/bin/bash
# =============================================================================
# MODE 1 — Conda (miniconda) on GPU cluster
# Usage: sbatch drone_detect_gpu.slurm
# Setup: conda create -n yolov5 python=3.10 -y
#        conda activate yolov5
#        pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
#        pip install -r requirements-cluster.txt --no-deps
# =============================================================================
#SBATCH --job-name=drone_detect
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -e
mkdir -p logs

echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# ── Conda environment ─────────────────────────────────────────────────────────
source /dist_home/ak_ajithkumar/miniconda3/etc/profile.d/conda.sh
conda activate yolov5

cd /dist_home/ak_ajithkumar/ondemand/ajith_work/drone-detect-benchmark
git pull origin main   # always run latest code

# Confirm environment
which python
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))
EOF

# ── Training ──────────────────────────────────────────────────────────────────
python train.py \
  --img 640 \
  --batch 64 \
  --epochs 300 \
  --data VisDrone.yaml \
  --cfg models/yolov5m.yaml \
  --device 0 \
  --seed 42 \
  --scale-aware-loss \
  --resolution-weighting \
  --loss-log-interval 100 \
  --name yolov5_m_scale_aware \
  --project runs/slurm

echo "Training finished"
