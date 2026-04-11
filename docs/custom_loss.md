# Custom Loss Functions for Small-Object Detection

This document describes two modifications to the standard YOLOv5 bounding-box
regression loss introduced for the VisDrone2019 ablation study.
Both changes target the box (CIoU) loss term only; classification and
objectness losses are unchanged.

---

## 1. Background: YOLOv5 Box Loss

YOLOv5 uses a three-head detection architecture operating at three spatial
scales, referred to as P3, P4, and P5:

| Head | Feature-map size | Stride | Primary target size |
|------|-----------------|--------|---------------------|
| P3   | 80 × 80         | 8 px   | Small objects       |
| P4   | 40 × 40         | 16 px  | Medium objects      |
| P5   | 20 × 20         | 32 px  | Large objects       |

For a single detection layer $l$ containing $N_l$ matched targets, the
standard box loss is:

$$\mathcal{L}_{\text{box}}^{(l)} = \frac{1}{N_l} \sum_{i=1}^{N_l} \bigl(1 - \text{CIoU}(\hat{b}_i,\, b_i)\bigr)$$

where $\hat{b}_i$ is the predicted box and $b_i$ is the ground-truth box.
The total box loss accumulates across all layers equally:

$$\mathcal{L}_{\text{box}} = \sum_{l \in \{P3, P4, P5\}} \mathcal{L}_{\text{box}}^{(l)}$$

**Limitation for drone imagery:** VisDrone is dominated by small, densely
packed objects (pedestrians, cyclists, motorcycles).  Equal per-target and
per-layer weighting means large objects contribute disproportionate gradient
signal relative to their frequency and difficulty, and the high-resolution P3
head competes on equal footing with the coarser P4/P5 heads.

---

## 2. Scale-Aware Loss Weighting

### Motivation

A large car covering 25 % of the image and a pedestrian covering 0.1 % of
the image each contribute equally to the mean loss.  The model therefore
optimises primarily for whatever class is most common and spatially dominant,
which is not small objects in drone footage.

### Formulation

For each matched target $i$ in layer $l$, define a per-target scale weight:

$$w_i^{\text{scale}} = \alpha \cdot \bigl(2 - \tilde{w}_i \cdot \tilde{h}_i\bigr)$$

where:

- $\tilde{w}_i = w_i^{\text{grid}} / W_l$ — target width in grid-cell units
  divided by the feature-map width $W_l$, giving image-normalised width
  $\tilde{w}_i \in [0, 1]$
- $\tilde{h}_i = h_i^{\text{grid}} / H_l$ — analogous normalised height
- $\alpha \geq 0$ — global scale multiplier (hyperparameter)

The normalised area $\tilde{w}_i \cdot \tilde{h}_i$ is clamped to $[0, 1]$
to guard against augmentation artefacts, giving a weight range:

$$w_i^{\text{scale}} \in [\alpha,\; 2\alpha]$$

A point-like object (area $\to 0$) receives weight $2\alpha$; an object
filling the entire image (area $= 1$) receives weight $\alpha$.

The scale-aware layer loss becomes:

$$\mathcal{L}_{\text{box,SA}}^{(l)} = \frac{1}{N_l} \sum_{i=1}^{N_l} w_i^{\text{scale}} \cdot \bigl(1 - \text{CIoU}(\hat{b}_i,\, b_i)\bigr)$$

### Numerical Example

Consider three objects in the same batch with $\alpha = 1.5$:

| Object      | $\tilde{w}$ | $\tilde{h}$ | Area   | $w^{\text{scale}}$ |
|-------------|------------|------------|--------|---------------------|
| Pedestrian  | 0.02       | 0.05       | 0.001  | **2.998**           |
| Van         | 0.12       | 0.08       | 0.010  | **2.986**           |
| Car         | 0.25       | 0.20       | 0.050  | **2.925**           |

All three weights cluster near $3 \approx 2\alpha$ because VisDrone objects
are small relative to the image.  The formula has a larger discriminative
effect on mixed datasets where large objects (area $> 0.2$) are common.

### CLI flag

```
--scale-aware-loss        enable scale-aware weighting
--scale-alpha FLOAT       value of alpha (default 1.0, best result 1.5)
```

---

## 3. Resolution-Aware Per-Layer Weighting

### Motivation

Summing P3, P4, and P5 box losses with equal weight dilutes the gradient
signal from the high-resolution P3 head — the very head responsible for
small-object detection.  For VisDrone, where the vast majority of objects
fall in the P3 detection range, this is a systematic bias against the most
important detection scale.

### Formulation

Each layer's (potentially scale-aware) box loss is multiplied by a fixed
scalar $\beta_l$ before accumulation:

$$\mathcal{L}_{\text{box,RA}}^{(l)} = \beta_l \cdot \mathcal{L}_{\text{box}}^{(l)}$$

$$\mathcal{L}_{\text{box}} = \sum_{l \in \{P3, P4, P5\}} \beta_l \cdot \mathcal{L}_{\text{box}}^{(l)}$$

where $\boldsymbol{\beta} = [\beta_{P3},\, \beta_{P4},\, \beta_{P5}]$ is a
fixed vector of per-layer importance weights set before training.

### Numerical Example

Suppose a mini-batch produces these raw per-layer box losses:

$$\mathcal{L}_{P3} = 0.30, \quad \mathcal{L}_{P4} = 0.20, \quad \mathcal{L}_{P5} = 0.10$$

**Original loss** ($\boldsymbol{\beta} = [1.0, 1.0, 1.0]$):

$$\mathcal{L}_{\text{box}} = 0.30 + 0.20 + 0.10 = 0.60$$

P3 contributes $0.30 / 0.60 = \mathbf{50\%}$ of the gradient signal.

**Resolution-aware loss** ($\boldsymbol{\beta} = [3.0, 1.0, 0.4]$):

$$\mathcal{L}_{\text{box}} = 3.0 \times 0.30 + 1.0 \times 0.20 + 0.4 \times 0.10$$
$$= 0.90 + 0.20 + 0.04 = 1.14$$

P3 now contributes $0.90 / 1.14 = \mathbf{79\%}$ of the gradient signal —
shifting the backbone and neck updates to prioritise small-object
localisation.

### CLI flag

```
--resolution-weighting                  enable per-layer weighting
--resolution-beta FLOAT FLOAT FLOAT     beta for P3 P4 P5 (default 2.0 1.0 0.5)
```

---

## 4. Combined Loss

When both modifications are active, the full box loss is:

$$\boxed{
\mathcal{L}_{\text{box}} = \sum_{l} \beta_l \cdot \frac{1}{N_l} \sum_{i=1}^{N_l}
  \alpha \bigl(2 - \tilde{w}_i \tilde{h}_i\bigr) \cdot \bigl(1 - \text{CIoU}(\hat{b}_i, b_i)\bigr)
}$$

The two terms are **independent and multiplicative at the target level** (scale-aware)
and **additive at the layer level** (resolution-aware).  Either modification
can be used in isolation.

### Best configuration (from ablation)

| Hyperparameter | Value |
|---|---|
| $\alpha$ | 1.5 |
| $\beta_{P3}$ | 3.0 |
| $\beta_{P4}$ | 1.0 |
| $\beta_{P5}$ | 0.4 |

```bash
python train.py ... \
  --scale-aware-loss --resolution-weighting \
  --scale-alpha 1.5 --resolution-beta 3.0 1.0 0.4
```

---

## 5. Empirical Results

### 5.1 Loss Ablation on YOLOv5n (50 epochs, VisDrone val)

| Experiment | Config | mAP@0.5 | mAP@0.5:0.95 | vs. baseline |
|---|---|---|---|---|
| E0 | Baseline (CIoU only) | 0.1736 | 0.0798 | — |
| E1 | Scale-aware only ($\alpha=1.0$) | 0.1745 | 0.0810 | +0.5% |
| E2 | Resolution-aware only (default $\boldsymbol{\beta}$) | 0.1775 | 0.0821 | +2.2% |
| E3 | Both (default $\boldsymbol{\beta}$) | 0.1776 | 0.0824 | +2.3% |
| E4 | Both (strong: $\alpha=1.5$, $\boldsymbol{\beta}=[3.0,1.0,0.4]$) | **0.1819** | **0.0845** | **+4.8%** |

Resolution-aware weighting drives the majority of the overall gain.

### 5.2 Swin Backbone Ablation (300 epochs, VisDrone val)

Per-class AP50 for selected small-object classes:

| Class | E0 (small, base) | E1 (small, new) | E2 (med, base) | E3 (med, new) |
|---|---|---|---|---|
| **pedestrian** | 0.379 | **0.410** (+3.1%) | 0.436 | **0.468** (+3.2%) |
| people | 0.320 | 0.322 (+0.2%) | 0.346 | **0.369** (+2.3%) |
| motor | 0.382 | 0.389 (+0.7%) | 0.415 | **0.425** (+1.0%) |
| bicycle | 0.110 | 0.105 (−0.5%) | 0.142 | 0.132 (−1.0%) |
| tricycle | 0.176 | 0.172 (−0.4%) | 0.222 | 0.212 (−1.0%) |
| awning-tricycle | 0.106 | 0.087 (−1.9%) | 0.108 | 0.101 (−0.7%) |

**Interpretation:** the combined loss consistently improves high-frequency small
classes (pedestrian, people, motor) whose large instance counts produce stable
gradients under amplification.  Rare classes (bicycle: 1,287 instances;
tricycle: 1,045) show slight regression because amplifying noisy gradients
from few labelled examples increases localisation variance.  This trade-off
is a direction for future work (e.g., class-frequency-aware $\alpha$).

---

## 6. Relation to Prior Work

The scale-aware weight $\alpha(2 - \tilde{w}\tilde{h})$ follows the convention
introduced in **Scaled-YOLOv4** (Wang et al., 2021), which applies the same
formula inside the cross-stage partial bottleneck to reduce the contribution
of large, easy-to-detect objects.  Our resolution-aware layer weighting is
related to the **focal loss** (Lin et al., 2017) philosophy of directing
gradient signal toward harder, smaller targets, but operates at the
feature-pyramid level rather than the per-anchor confidence level.
