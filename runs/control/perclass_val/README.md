# Per-class validation output

`val.py --verbose` on each of the nine control-run checkpoints, VisDrone2019 val
(548 images, 38759 instances), evaluated identically. Columns are the YOLOv5 default:

    Class  Images  Instances  P  R  mAP@0.5  mAP@0.5:0.95

These files are the evidence for F-D (frequency reallocation) and F-E (pedestrian
+9.6%) in ../../docs/SUMMARY.md. The `best.pt` checkpoints themselves are not
committed (see .gitignore), so these outputs are the reproducible record.

Regenerate with:

    python val.py --img 640 --batch 8 --data data/VisDrone_local.yaml \
      --weights runs/control/<run>/weights/best.pt --task val --verbose
