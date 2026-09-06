# Per-class validation output — INDISCON A/B/C/D matrix

`val.py --verbose` on the four run checkpoints, VisDrone2019 val (548 images,
38,759 instances), evaluated identically at batch 8.

Name mapping (use the right-hand column in the paper):

| file | name |
|---|---|
| `A_baseline.txt` | YOLOv5s |
| `B_p2head.txt` | YOLOv5s-P2 |
| `C_swin_p2.txt` | YOLOv5s-P2 + Swin |
| `D_swin_p2_loss.txt` | YOLOv5s-P2 + Swin + custom loss |

The `Speed:` line carries the inference timing used for the FPS column
(RTX 2000 Ada, 640x640, inference only, excludes NMS).

The `best.pt` checkpoints are gitignored, so these outputs are the reproducible record.
