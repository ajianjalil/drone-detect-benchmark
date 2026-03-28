#!/bin/bash
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

# ===============================
# Safety & debug
# ===============================
set -e
echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# ===============================
# Move to project directory
# ===============================
cd /dist_home/ak_ajithkumar/ondemand/ajith_work/drone-detect-benchmark

# ===============================
# Activate virtual environment
# ===============================
source venv/bin/activate

# ===============================
# Confirm Python & GPU
# ===============================
which python
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# ===============================
# YOLOv5 Training
# ===============================
python train.py \
  --img 640 \
  --batch 64 \
  --epochs 300 \
  --data VisDrone.yaml \
  --cfg models/yolov5n.yaml \
  --device 0 \
  --name yolov5_n_no_swin \
  --project runs/slurm

echo "Job finished"

