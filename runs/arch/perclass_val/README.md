# Per-class validation output — architecture arm

`val.py --verbose` on each of the six architecture checkpoints (A0 = YOLOv5s,
A1 = YOLOv5s + SingleSwin; seeds 42/43/44), VisDrone2019 val, evaluated identically.

Evidence for F-H in ../../docs/SUMMARY.md. The `best.pt` checkpoints are gitignored,
so these outputs are the reproducible record.

NOTE: A1 uses models/yolov5s_swin.yaml, which is RECONSTRUCTED — see that file's header.
