# Use the official PyTorch base image with CUDA
FROM nvcr.io/nvidia/pytorch:25.03-py3

# Set a working directory
WORKDIR /app

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# Install necessary system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean

# Install Python packages
RUN python3 -m pip install numpy \
    && python3 -m pip install opencv-python-headless \
    && python3 -m pip install Pillow \
    && python3 -m pip install tqdm

RUN python3 -m pip install matplotlib
RUN python3 -m pip install notebook
# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python packages listed in requirements.txt
RUN python3 -m pip install -r requirements.txt

# Update package lists and install git
RUN apt-get update && apt-get install -y git
# Configure git to recognize the /app/yolov5 directory as safe
RUN git config --global --add safe.directory /app

