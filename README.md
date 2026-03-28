# Drone-Detect-Benchmark

**Enhancing Small Object Detection in UAV Imagery Using an Optimized Attention-Driven YOLO Model**  
A Benchmarking Study on VisDrone2019  
PhD Coursework — Deep Learning & Deep Learning for Visual Recognition  
Second Semester (2024–2025 Even)  
Group: 24DS611 & 24DS736

---

## 📌 Project Overview

This repository contains code and configuration for our benchmarking study focused on improving small object detection in UAV (Unmanned Aerial Vehicle) imagery. We extend the YOLOv5 model by integrating attention mechanisms (such as Swin Transformer blocks) and evaluate performance on the VisDrone2019 dataset.

The aim is to optimize detection accuracy for small and occluded objects commonly found in aerial surveillance footage using advanced feature representation techniques.

---

## 🧠 Key Contributions

- Benchmarking traditional YOLOv5 models against Swin-Transformer-augmented variants.
- Fine-tuned configurations for small object detection in the VisDrone2019 dataset.
- Comparative training results and performance analysis.

---

## 🗃️ Dataset

We use the **VisDrone2019** dataset. It must be preprocessed into a YOLOv5-compatible format before training.

### 🔧 Dataset Setup

1. Download VisDrone2019 from the [official website](https://github.com/VisDrone/VisDrone-Dataset).
2. Convert annotations to YOLO format. You can use available converters (e.g., [VisDrone-to-YOLO](https://github.com/Guolei1130/VisDrone-YOLO-Converter)) or scripts included in the repo (if any).
3. Organize the dataset as follows:
`datasets/ └── VisDrone/ ├── images/ │ ├── train/ │ ├── val/ │ └── test/ └── labels/ ├── train/ ├── val/ └── test/`

4. Update the `VisDrone.yaml` file accordingly:
```yaml
train: datasets/VisDrone/images/train
val: datasets/VisDrone/images/val
test: datasets/VisDrone/images/test

nc: 10  # Number of classes
names: [ignored regions, pedestrians, people, bicycles, cars, vans, trucks, tricycles, awning-tricycles, buses, motorcycles]
```
5. Training Instructions
```
python train.py --img 640 --batch 64 --epochs 300 --data VisDrone.yaml --cfg models/yolov5n.yaml --name yolov5_n_no_swin

python train.py --img 640 --batch 16 --epochs 300 --data VisDrone.yaml --cfg models/yolov5_swin.yaml --name yolov5_15_100

python train.py --img 640 --batch 16 --epochs 300 --data VisDrone.yaml --cfg models/yolov5s_swin.yaml --name yolov5_s_swin
```
Make sure you have all dependencies installed as per the YOLOv5 repository, including PyTorch, OpenCV, and other required libraries.

```
pip install torch==2.5.1 torchvision==0.20.1   --index-url https://download.pytorch.org/whl/cu121   --extra-index-url https://pypi.org/simple   --no-cache-dir
```