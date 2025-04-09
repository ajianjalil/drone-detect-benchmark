# drone-detect-benchmark
Code for the project: Enhancing Small Object Detection in UAV Imagery Using an Optimized Attention-Driven YOLO Model: A Benchmarking Study on VisDrone2019 24DS611 &amp; 24DS736 Deep Learning &amp; Deep Learning for Visual Recognition PhD Course Work Second Semester (2024 - 2025 (Even))
## How to run the code
1. you have to download visdrone19 dataset and preprocess it so that it is in the form of yolov5 expected
2. run the code as below
```
python train.py --img 640 --batch 64 --epochs 300 --data VisDrone.yaml --cfg models/yolov5n.yaml --name yolov5_n_no_swin
python train.py --img 640 --batch 16 --epochs 300 --data VisDrone.yaml --cfg models/yolov5_swin.yaml --name yolov5_15_100
python train.py --img 640 --batch 16 --epochs 300 --data VisDrone.yaml --cfg models/yolov5s_swin.yaml --name yolov5_s_swin
```
