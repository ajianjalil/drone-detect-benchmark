#!/bin/bash
#SBATCH --job-name=drone_detect
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
set -e

echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Activate conda
source /dist_home/ak_ajithkumar/miniconda3/etc/profile.d/conda.sh
conda activate yolov5

cd /dist_home/ak_ajithkumar/ondemand/ajith_work/drone-detect-benchmark

# Hard GPU checks
nvidia-smi

python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))
EOF

# Run training
python train.py \
  --img 640 \
  --batch 64 \
  --epochs 300 \
  --data VisDrone.yaml \
  --cfg models/yolov5m.yaml \
  --device 0 \
  --name yolov5_m_no_swin \
  --project runs/slurm

echo "Training finished"

