#!/usr/bin/env bash
# setup_env.sh — Fresh Ubuntu + NVIDIA setup for drone-detect-benchmark
#
# Tested on: Ubuntu 22.04, NVIDIA RTX 3050 Ti (driver 570.x), CUDA 12.1
# Run from the repo root after cloning.
#
# Usage:
#   bash setup_env.sh          # full setup (installs driver, miniconda, env)
#   bash setup_env.sh --no-driver  # skip NVIDIA driver install (already installed)

set -euo pipefail

CONDA_ENV="yolov5"
PYTHON_VERSION="3.10"
TORCH_VERSION="2.5.1+cu121"
TORCHVISION_VERSION="0.20.1+cu121"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
MINICONDA_DIR="$HOME/miniconda3"
MINICONDA_INSTALLER="$HOME/miniconda.sh"

SKIP_DRIVER=false
for arg in "$@"; do
  [[ "$arg" == "--no-driver" ]] && SKIP_DRIVER=true
done

echo "============================================"
echo " drone-detect-benchmark environment setup"
echo "============================================"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    wget curl git build-essential \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    ffmpeg

# ── 2. NVIDIA driver ──────────────────────────────────────────────────────────
if [[ "$SKIP_DRIVER" == false ]]; then
    echo "[2/6] Installing NVIDIA driver (570)..."
    sudo apt-get install -y --no-install-recommends \
        software-properties-common
    sudo add-apt-repository -y ppa:graphics-drivers/ppa
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-driver-570
    echo "  --> Reboot required after this script to load the driver."
    echo "      Re-run with --no-driver after rebooting."
else
    echo "[2/6] Skipping NVIDIA driver install (--no-driver)."
    nvidia-smi || { echo "ERROR: nvidia-smi failed. Is the driver installed?"; exit 1; }
fi

# ── 3. Miniconda ──────────────────────────────────────────────────────────────
echo "[3/6] Installing Miniconda..."
if [[ -d "$MINICONDA_DIR" ]]; then
    echo "  Miniconda already found at $MINICONDA_DIR, skipping download."
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O "$MINICONDA_INSTALLER"
    bash "$MINICONDA_INSTALLER" -b -p "$MINICONDA_DIR"
    rm "$MINICONDA_INSTALLER"
fi

# Init conda in shell (idempotent)
"$MINICONDA_DIR/bin/conda" init bash
# Make conda available in this script session
source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# Accept Anaconda ToS (required since conda 24.x)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ── 4. Create conda env ───────────────────────────────────────────────────────
echo "[4/6] Creating conda env '$CONDA_ENV' (Python $PYTHON_VERSION)..."
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Env '$CONDA_ENV' already exists, skipping creation."
else
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
fi
conda activate "$CONDA_ENV"

# ── 5. Install Python packages ────────────────────────────────────────────────
echo "[5/6] Installing PyTorch $TORCH_VERSION (CUDA 12.1)..."
pip install --quiet \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "$TORCH_INDEX"

echo "  Installing project requirements..."
pip install --quiet -r requirements.txt

echo "  Installing timm (Swin Transformer backbone)..."
pip install --quiet timm

echo "  Installing setuptools (provides pkg_resources, must be <81)..."
pip install --quiet "setuptools>=70.0.0,<81"

# ── 6. Verify ─────────────────────────────────────────────────────────────────
echo "[6/6] Verifying installation..."
python - <<'EOF'
import torch, timm
print(f"  torch       : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
print(f"  timm        : {timm.__version__}")
import cv2, numpy, pandas, scipy
print(f"  opencv      : {cv2.__version__}")
print(f"  numpy       : {numpy.__version__}")
print()
print("All checks passed.")
EOF

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " Activate env with:"
echo "   conda activate $CONDA_ENV"
echo ""
echo " Sanity-check training run:"
echo "   python train.py --img 640 --batch 4 --epochs 1 \\"
echo "     --data data/VisDrone.yaml --cfg models/yolov5s_swin2.yaml \\"
echo "     --device 0 --seed 42 --loss-log-interval 10 \\"
echo "     --name swin_small_sanity --project runs/sanity \\"
echo "     --scale-aware-loss --resolution-weighting \\"
echo "     --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4"
echo "============================================"
