# Step 1 — local measurements (avcom box)

Companion to [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md). Everything here was measured
on this machine and can be re-run; nothing is carried over from the audit box.

**Environment.** conda env `yolov5` (cloned from `dygrid`, setuptools pinned <81 so
`pkg_resources` resolves), torch 2.5.1+cu124, NVIDIA RTX 2000 Ada 16GB.

**Data.** `data/VisDrone_local.yaml` → `/home/avcom/Documents/ajith/visdrone_yolov5`,
a symlink tree over the VisDrone2019 copy in `phd_research/dygrid_edgeyolo/data/`.
YOLOv5 writes its `labels.cache` into the symlink tree only; the ultralytics caches
in `phd_research` were verified byte-identical before and after (md5).
All three splits present: train 6471, val 548, **test-dev 1610 (labelled)**.

---

## M1 — The box-loss inflation factor, measured

F6 estimates ×2.80 (E3) and ×5.64 (E4) from an **assumed** P3/P4/P5 box-loss split
of 50/30/20. Measured over 640 augmented VisDrone training images through the real
`ComputeLoss` path (yolov5n at initialisation):

| Config | mean `lbox` | × baseline | F6 estimate |
|---|---|---|---|
| E0 baseline | 0.12488 | 1.000 | 1.00 |
| E1 scale only | 0.24930 | 1.996 | — |
| E2 res only | 0.14482 | 1.160 | — |
| E3 both, default β | 0.28931 | **2.317** | 2.80 |
| E4 both, strong | 0.54499 | **4.364** | 5.64 |

F6's direction is right and its central claim holds — **the loss is not a pure
re-allocation, it inflates total box loss ~4.4×** — but the estimates run ~20–30%
high. The control runs use the measured factors.

## M2 — The assumed layer split is wrong

Per-layer share, measured by isolating one layer at a time with one-hot β:

| | P3 | P4 | P5 |
|---|---|---|---|
| baseline | 33.1% | 32.8% | 34.2% |
| E4 loss | 68.1% | 22.5% | 9.4% |

The paper claims the P3 share moves **~50% → ~79%**. Measured: **33.1% → 68.1%**.
The baseline split is near-uniform, not 50/30/20. The *effect* the paper describes is
real and large; the two numbers quoted for it are both wrong.

*Caveat: measured at initialisation. The split shifts during training as IoU improves
per-layer, so treat these as the starting condition, not a training average.*

## M3 — YOLOv5's printed GFLOPs is wrong for the Swin variants

`train.py` prints GFLOPs by profiling at 32×32 and scaling by (640/32)². That
extrapolation assumes every op scales with H·W, which windowed attention does not.

| Model | Params | FP16 MB | **printed** | thop@640 | aten@640 |
|---|---|---|---|---|---|
| YOLOv5n | 1,777,447 | 3.6 | 4.3 | 4.3 | 4.2 |
| YOLOv5s | 7,046,599 | 14.1 | 16.0 | 16.0 | 15.8 |
| YOLOv5m | 20,907,687 | 41.8 | 48.3 | 48.3 | 48.0 |
| YOLOv5n + SingleSwin | 3,357,751 | 6.7 | **11.8** | 5.5 | **5.4** |
| YOLOv5s + DoubleSwin | 13,383,515 | 26.8 | **47.8** | 22.6 | **22.6** |
| YOLOv5m + DoubleSwin | 42,331,271 | 84.7 | **158.3** | 73.4 | **73.4** |

The three methods agree to within 1% on every pure-CNN model, which validates them;
they diverge by **~2.2×** on every Swin model. `aten@640`
(`torch.utils.flop_counter`, real ATEN ops including the attention matmuls) is the
number to quote.

If the paper cites the printed figure for the Swin rows it **overstates their cost by
~2.2×** — which understates the contribution, since the Swin variants are cheaper
than reported. The FP16 MB column reproduces Table I's size column exactly.

Answers R2 (computational complexity) and R3.2 (FLOPs).

---

## Control experiment (running)

`run_boxgain_control.sh` — three 300-epoch yolov5n runs, identical but for the loss:

| Run | Loss | `box` gain |
|---|---|---|
| C0_baseline | original CIoU | 0.05 |
| C1_e4_loss | scale-aware + resolution, α=1.5, β=[3,1,0.4] | 0.05 |
| C2_boxgain_control | original CIoU | 0.2182 (= 0.05 × 4.364, from M1) |

All three on the same GPU, so the comparison is self-consistent — the F1 300-epoch
numbers were produced on a different machine and are not directly comparable.

- **C2 ≈ C1** → the weighting adds nothing beyond box-loss magnitude.
- **C2 ≈ C0** → the weighting does real work.

### Results

| Run | Loss | `box` | mAP@0.5 | mAP@0.5:0.95 | wall |
|---|---|---|---|---|---|
| C0_baseline | original CIoU | 0.05 | **0.2612** | **0.1317** | 6.42 h |
| C1_e4_loss | scale+res, α=1.5, β=[3,1,0.4] | 0.05 | **0.2532** | **0.1284** | 6.6 h¹ |
| C2_boxgain_control | original CIoU | 0.2182 | **0.2467** | **0.1269** | 5.3 h |

¹ paused at epoch 172 and resumed from `last.pt`; the dataloader reseeds at resume,
so C1 is not bit-identical to an uninterrupted run.

**C1 vs C0: −0.0080 mAP@0.5 (−3.1% relative), −0.0033 mAP@0.5:0.95 (−2.5%).**

The paper's headline E4 configuration finishes **below** baseline at 300 epochs.
This independently reproduces F1 on a second machine — the audit box measured E4 at
−0.5% vs E0 (0.2560 vs 0.2572); here it is −3.1% (0.2532 vs 0.2612). Different
magnitudes, same sign, and both contradict the paper's reported +4.8%.

Trajectory (relative to C0):

| epoch | 26 | 51 | 101 | 151 | 201 | 251 | 300 |
|---|---|---|---|---|---|---|---|
| C1 vs C0 | −2.0% | −1.3% | −5.1% | −5.2% | −4.3% | −3.7% | −3.0% |

C1 never leads. The deficit peaks near epoch 150 and then narrows as both runs
converge, but does not close. Note what this rules out: the "+4.8% at 50 epochs"
cannot be reproduced here even *at* 50 epochs, where C1 sits 1.3% **behind**. So the
paper's result is not simply an early-training effect that later decays — under this
protocol (batch 64, 300-epoch LR schedule) it is absent throughout.

### The control settles F6

| Run | Config | mAP@0.5 | vs C0 |
|---|---|---|---|
| C0 | baseline CIoU, box=0.05 | 0.2612 | — |
| C1 | E4 weighting, box=0.05 | 0.2532 | **−3.1%** |
| C2 | baseline CIoU, box=0.2182 | 0.2467 | **−5.6%** |

`|C2 − C1| = 0.0065` versus `|C2 − C0| = 0.0146`. **C2 lands more than twice as close
to C1 as to C0**, which is F6's "C2 ≈ C1" branch: most of what the custom loss does is
explained by the box-loss magnitude alone, not by scale or resolution awareness.

The decomposition is slightly more interesting than a flat null:

- inflating the box gain to E4's magnitude, with **no** weighting: **−5.6%**
- adding the scale/resolution weighting back on top: **−3.1%**

So the weighting *does* something — it recovers ~2.5 points of the damage the
inflation causes — but it never turns positive. The mechanism the paper credits for
its gain is real; it is simply working against a larger penalty that the same code
introduces silently, and the net is negative.

The ordering **C2 < C1 < C0 holds at every one of the seven checkpoints sampled**
(epochs 26, 51, 101, 151, 201, 251, 300), which is considerably stronger evidence than
three endpoints would be.

**Caveat, stated for our own runs and not only the paper's.** One seed per config. The
absolute gaps (0.008 and 0.015 mAP@0.5) are small enough that seed variance could move
them, and R2's complaint about the paper's missing significance testing applies here
too. The stable ordering across the full trajectory is what carries the conclusion; the
exact percentages should not be quoted as precise without 3 seeds.

### What this means for the paper

R1.6 asked how sensitive the +4.8% is to α and β. The honest answer this produces is
that the question is mis-framed: under a 300-epoch schedule the configuration does not
beat baseline at any α/β tested, and its dominant effect is an unreported ~4.4× box-loss
inflation that is itself harmful. Section III-C's claim needs to become a description
of a confound rather than a contribution.

**Baseline sanity check.** C0's 0.2612 sits between the two independent 300-epoch
yolov5n baselines already on record — F1's E0 (0.2572, audit box) and
`runs/train/yolov5_n_no_swin` (0.2636, batch 64 / seed 0). Reproducing both to
within ~1% on a third machine means the local pipeline is sound and the C0/C1/C2
comparison rests on a baseline that is not an artifact of this box.

Run settings: batch 64, seed 42, `--weights yolov5s.pt` (train.py default, matching
the recorded `opt.yaml` of the reference architecture runs), 300 epochs, img 640.
