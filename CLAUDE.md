# drone-detect-benchmark — Claude context

YOLOv5-based drone detection benchmark on VisDrone2019-DET.
Fork of Ultralytics YOLOv5, extended with Swin Transformer backbones and custom losses.

---

## Project goal

Benchmark YOLOv5 variants on VisDrone for small-object detection.
Current focus: ablation study comparing a custom scale-aware + resolution-aware box loss
against the original YOLOv5 CIoU loss, across both standard and Swin-backbone models.

---

## Custom loss changes (utils/loss.py)

Two optional modifications to the YOLOv5 box regression (CIoU) loss:

**Scale-aware weighting** (`--scale-aware-loss`)
- Formula: `scale_weight = alpha * (2 - norm_w * norm_h)`
- Upweights loss for small boxes, downweights large boxes
- Controlled by `--scale-alpha` (default 1.0, best result: 1.5)

**Resolution-aware per-layer weighting** (`--resolution-weighting`)
- Multiplies box loss per detection head layer by a fixed beta
- Default betas: `[2.0, 1.0, 0.5]` for P3/P4/P5 (emphasise small-object head)
- Best result: `--resolution-beta 3.0 1.0 0.4`

Both flags are off by default (original YOLOv5 loss). They are independent and additive.

**train.py flags summary:**
```
--scale-aware-loss              enable scale-aware weight
--resolution-weighting          enable per-layer resolution weight
--scale-alpha   FLOAT           multiplier for scale weight (default 1.0)
--resolution-beta FLOAT FLOAT FLOAT   per-layer betas P3 P4 P5 (default 2.0 1.0 0.5)
--loss-log-interval INT         print loss diagnostics every N steps (default 200, 0=off)
```

---

## Model configs (models/)

| File | Backbone | depth_multiple | width_multiple | Notes |
|---|---|---|---|---|
| yolov5n.yaml | standard | 0.33 | 0.25 | nano |
| yolov5s.yaml | standard | 0.33 | 0.50 | small |
| yolov5m.yaml | standard | 0.67 | 0.75 | medium |
| yolov5l.yaml | standard | 1.00 | 1.00 | large |
| yolov5_swin.yaml | Swin at P5 only | 0.33 | 0.25 | SwinStage replaces C3 at top |
| yolov5s_swin2.yaml | Swin at P2+P5 | 0.33 | 0.50 | small-sized, two SwinStages |
| yolov5m_swin.yaml | Swin at P2+P5 | 0.67 | 0.75 | medium-sized (created for ablation) |

`SwinStage` args: `[c2, depth, num_heads, window_size]` — channels are auto-scaled by `parse_model` via `width_multiple`.
Implementation: `models/swintransformer.py`.

---

## Data configs (data/)

| File | path | Use for |
|---|---|---|
| VisDrone.yaml | /mnt/mydrive/ajith/data_set/VisDrone | local machine |
| VisDrone_cluster.yaml | /dist_home/ak_ajithkumar/ondemand/ajith_work/VisDrone | SLURM cluster |

VisDrone has **10 classes**: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.
Always set `nc: 10` in model configs for VisDrone training.

---

## SLURM scripts

| File | Purpose | Array | Epochs | GPU |
|---|---|---|---|---|
| ablation_experiments.slurm | 5-exp loss ablation on yolov5n | 0-4%1 | 50 | A100 80GB |
| ablation_swin.slurm | 4-exp Swin ablation (small+medium, with/without new loss) | 0-3%1 | 300 | A100 80GB |
| drone_detect_gpu.slurm | single training run | — | — | — |
| yolov5_visdrone_train.slurm | single training run | — | — | — |

**Cluster paths:**
- conda env: `yolov5` at `/dist_home/ak_ajithkumar/miniconda3`
- repo: `/dist_home/ak_ajithkumar/ondemand/ajith_work/drone-detect-benchmark`
- dataset: `/dist_home/ak_ajithkumar/ondemand/ajith_work/VisDrone`

**ablation_swin.slurm experiment map:**

| Index | Name | Model | Loss |
|---|---|---|---|
| 0 | E0_swin_small_baseline | yolov5s_swin2 | original |
| 1 | E1_swin_small_new_loss | yolov5s_swin2 | scale-aware + res strong |
| 2 | E2_swin_medium_baseline | yolov5m_swin | original |
| 3 | E3_swin_medium_new_loss | yolov5m_swin | scale-aware + res strong |

Batch: small=64, medium=32. Wall time: 4 days.

---

## Completed ablation results (runs/ablation_local/ablation/)

5-experiment loss ablation on **yolov5n**, 50 epochs, local machine.
Best mAP@0.5 per experiment:

| Exp | Config | mAP@0.5 | mAP@0.5:0.95 | vs baseline |
|---|---|---|---|---|
| E0 | baseline | 0.1736 | 0.0798 | — |
| E1 | scale_only | 0.1745 | 0.0810 | +0.5% |
| E2 | res_only | 0.1775 | 0.0821 | +2.2% |
| E3 | both (default beta) | 0.1776 | 0.0824 | +2.3% |
| E4 | both_strong (alpha=1.5, beta=3.0,1.0,0.4) | **0.1819** | **0.0845** | **+4.8%** |

**Key finding:** resolution weighting drives most of the gain; stronger weights (E4) are best.
E4 flags are used as the "new loss" config in all subsequent experiments.

---

## Local sanity-check command

```bash
python train.py \
  --img 640 --batch 4 --epochs 1 \
  --data data/VisDrone.yaml \
  --cfg models/yolov5s_swin2.yaml \
  --device 0 --seed 42 --loss-log-interval 10 \
  --name swin_small_sanity --project runs/sanity \
  --scale-aware-loss --resolution-weighting \
  --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4
```

---

## Hardware

- Local: OMEN laptop GPU (limited VRAM — use batch 2–4 for sanity checks)
- Cluster: **A100 80GB** — batch 64 for small models, 32 for medium Swin
