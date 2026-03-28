#!/bin/bash
# =============================================================================
# MODE 2 — venv on GPU cluster
# Usage: sbatch yolov5_visdrone_train.slurm
# Setup: python3 -m venv venv
#        source venv/bin/activate
#        pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
#        pip install -r requirements-cluster.txt
# =============================================================================
#SBATCH --job-name=yolov5_visdrone
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --nodelist=asaicomputenode02
#SBATCH --output=logs/yolov5_%j.out
#SBATCH --error=logs/yolov5_%j.err

set -e
mkdir -p logs

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# ── venv environment ──────────────────────────────────────────────────────────
cd /dist_home/ak_ajithkumar/ondemand/ajith_work/drone-detect-benchmark
git pull origin main   # always run latest code

source venv/bin/activate

# Confirm environment
which python
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# ── Training ──────────────────────────────────────────────────────────────────
python train.py \
  --img 640 \
  --batch 64 \
  --epochs 300 \
  --data VisDrone.yaml \
  --cfg models/yolov5n.yaml \
  --device 0 \
  --seed 42 \
  --scale-aware-loss \
  --resolution-weighting \
  --loss-log-interval 100 \
  --name yolov5_n_scale_aware \
  --project runs/slurm

echo "Job finished"
