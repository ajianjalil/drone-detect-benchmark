# Paper 2887 — findings summary

**Consolidated reference for the revision.** Everything below was measured on the
avcom box between 2026-08-25 and 2026-08-28 and is reproducible from this repo.

Companion documents:
[REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) (reviewer audit) ·
[STEP1_MEASUREMENTS.md](STEP1_MEASUREMENTS.md) (control experiment) ·
[PERCLASS_FINDING.md](PERCLASS_FINDING.md) (per-class analysis) ·
[PAPER_NARRATIVE.md](PAPER_NARRATIVE.md) (framing)

---

## 1. What was run

| Experiment | Configs | Seeds | Epochs | Total |
|---|---|---|---|---|
| Box-gain control | C0 / C1 / C2 | 42, 43, 44 | 300 | 9 runs, ~48 h |
| Per-class evaluation | all 9 checkpoints | — | — | `val.py --verbose` |
| Loss instrumentation | E0–E4 | — | — | 640 real batches |
| Complexity | 6 model configs | — | — | 3 FLOP methods |

- **C0** baseline CIoU, `box=0.05`
- **C1** scale-aware + resolution-aware, α=1.5, β=[3.0, 1.0, 0.4], `box=0.05`
- **C2** baseline CIoU, `box=0.2182` — E4's box-loss *magnitude* with **no** weighting

C2 is the control that separates "the weighting does something" from "the weighting
just makes the box loss louder".

Environment: conda env `yolov5` (torch 2.5.1+cu124), RTX 2000 Ada 16 GB,
VisDrone2019-DET via `data/VisDrone_local.yaml`. yolov5n, img 640, batch 64,
`--weights yolov5s.pt` (matching the reference runs' recorded `opt.yaml`).

---

## 2. Headline findings

### F-A. A measured noise floor of 1.4%

Three seeds of the identical baseline:

| seed | 42 | 43 | 44 | mean | sd |
|---|---|---|---|---|---|
| C0 mAP@0.5 | 0.2612 | 0.2641 | 0.2605 | **0.2619** | 0.0019 |

Range 0.0036 = **1.4% of the mean**. Single-run deltas below roughly 1.5% relative on
this benchmark are not interpretable. This is the study's most reusable output and it
answers R2 (statistical significance) directly.

### F-B. The +4.8% aggregate gain does not reproduce

| | s42 | s43 | s44 | mean | sd |
|---|---|---|---|---|---|
| C0 baseline | 0.2612 | 0.2641 | 0.2605 | 0.2619 | 0.0019 |
| C1 E4 loss | 0.2532 | 0.2574 | 0.2571 | 0.2559 | 0.0024 |
| C2 box-gain only | 0.2467 | 0.2457 | 0.2447 | 0.2457 | 0.0010 |

Relative to each seed's own C0:

| | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| C1 | −2.3% (sd 0.9) | −1.0% (sd 1.4) |
| C2 | **−6.2%** (sd 0.7) | **−4.0%** (sd 0.5) |

The supported claim is **no aggregate benefit**, not harm: C1 is below C0 on mAP@0.5 at
all three seeds, but the mean deficit is only ~1.7× the floor and narrows across seeds
(−3.1% → −2.5% → −1.3%); on mAP@0.5:0.95 one seed is positive.

C2's harm, by contrast, is unambiguous on both metrics at every seed.

### F-C. The loss inflates total box loss 4.4×

Measured over 640 augmented VisDrone batches through the real `ComputeLoss` path:

| config | × baseline box loss |
|---|---|
| E3 (α=1.0, β=[2, 1, 0.5]) | ×2.32 |
| E4 (α=1.5, β=[3, 1, 0.4]) | **×4.36** |

No renormalisation exists anywhere in `utils/loss.py`. The formulation therefore raises
the effective box-branch learning rate as a side effect, and **C2 shows that side
effect alone is harmful** (−6.2%). The weighting recovers about two-thirds of that
damage, netting to roughly zero.

### F-D. The effect is a frequency reallocation, not a small-object prior

Per-class AP@0.5, mean of 3 seeds, sorted by instance count:

| class | instances | mean area | C0 | C1 | rel |
|---|---|---|---|---|---|
| car | 14064 | 24.44 | 0.680 | 0.691 | **+1.6%** |
| pedestrian | 8844 | 5.83 | 0.325 | 0.357 | **+9.6%** |
| people | 5125 | 5.11 | 0.261 | 0.274 | **+4.8%** |
| motor | 4886 | 7.41 | 0.322 | 0.336 | **+4.5%** |
| van | 1975 | 25.05 | 0.249 | 0.203 | −18.7% |
| bicycle | 1287 | 7.08 | 0.069 | 0.055 | −19.5% |
| tricycle | 1045 | 16.92 | 0.123 | 0.107 | −13.0% |
| truck | 750 | 53.46 | 0.190 | 0.179 | −6.1% |
| awning-tricycle | 532 | 17.41 | 0.085 | 0.068 | −20.1% |
| bus | 251 | 47.91 | 0.312 | 0.288 | −7.6% |

The four most frequent classes improve; the six rarest decline. Correlation of the
per-class change with **log instance count: r = +0.688**; with **log mean box area:
r = −0.365** (weak, and the wrong sign for a scale mechanism).

Two classes settle it: `car` has the largest area of any frequent class and gains;
`bicycle` is small (7.08, comparable to `motor` at 7.41) and loses 19.5%.

### F-E. Pedestrian AP is the study's most solid result

| | s42 | s43 | s44 | spread |
|---|---|---|---|---|
| C0 | 0.324 | 0.324 | 0.328 | 0.004 |
| C1 | 0.359 | 0.356 | 0.355 | 0.004 |

**Disjoint distributions** — every C1 seed exceeds every C0 seed, and the gap (0.032) is
eight times the within-group spread. The paper currently claims +3.1–3.2%; the measured
effect is **+9.6%**.

Decomposition via C2: magnitude alone gives +4.8%, the full loss +9.6%. **Roughly half
the pedestrian gain is attributable to the weighting itself** — a genuine, separable
contribution, unlike the aggregate result.

### F-F. YOLOv5's printed GFLOPs overstates Swin variants ~2.2×

`train.py` profiles at 32×32 and scales by (640/32)², which windowed attention does not
obey. Three methods agree within 1% on every pure-CNN model and diverge only on Swin:

| Model | Params | printed | **aten@640** |
|---|---|---|---|
| YOLOv5n | 1,777,447 | 4.3 | 4.2 |
| YOLOv5s | 7,046,599 | 16.0 | 15.8 |
| YOLOv5m | 20,907,687 | 48.3 | 48.0 |
| YOLOv5n + SingleSwin | 3,357,751 | 11.8 | **5.4** |
| YOLOv5s + DoubleSwin | 13,383,515 | 47.8 | **22.6** |
| YOLOv5m + DoubleSwin | 42,331,271 | 158.3 | **73.4** |

Quote `aten@640` (`torch.utils.flop_counter`). This cuts in the paper's favour — the
Swin variants are cheaper than reported. Answers R2-complexity and R3.2.

### F-G. Corrections to the manuscript's own numbers

| Paper says | Measured |
|---|---|
| Table I trained 100 epochs | **300** (all six `opt.yaml` and 300 rows in each `results.csv`) |
| "batch size 16 throughout" | Only `yolov5_s_swin` used 16; the rest used 64 |
| P3 loss share 50% → 79% | **33.1% → 68.1%** (baseline split is ~33/33/34, not 50/30/20) |
| SwinStage = Patch Merging + W-MSA + SW-MSA | **No Patch Merging exists.** `PatchMerging` is defined and imported but never instantiated; the stage is resolution-preserving |
| Pedestrian +3.1–3.2% | **+9.6%** |

The Patch Merging correction *refutes* R1.5 rather than conceding it: the architecture
never had the 20×20 → 10×10 downsampling the reviewer objected to.

---

## 3. Claim language for the manuscript

**Use:**

> The proposed weighting produces a consistent, seed-stable redistribution of detection
> performance across the class-frequency distribution — +9.6% AP@0.5 on pedestrian, the
> most frequent small class, at the cost of 6–20% on rare classes — leaving aggregate
> mAP unchanged. Approximately half the effect is attributable to an unintended 4.4×
> inflation of total box loss, isolated by a magnitude-matched control; the remainder to
> the weighting itself.

**Do not use:**

- "improves overall mAP" — it does not; aggregate is unchanged at best
- "E4 is worse than baseline" — not supported at n=3 on mAP@0.5:0.95
- "small-object prior" — the mechanism is frequency, not scale
- the +4.8% figure, in any section
- the YOLOv5n+DoubleSwin row, until `models/yolov5n_swin2.yaml` is restored

---

## 4. Stated exposures

Recorded so they are not discovered during review.

**The architecture claim is n=1 and sits at the noise floor this paper establishes.**
Measured against the 1.4% floor from F-A:

| variant | rel mAP@0.5 | × floor | rel mAP@0.5:0.95 | × floor |
|---|---|---|---|---|
| YOLOv5s + SingleSwin | +2.04% | 1.5 | +1.27% | **0.9** |
| YOLOv5n + SingleSwin | +1.17% | 0.9 | +1.76% | 1.3 |
| YOLOv5n + DoubleSwin | +4.30% | 3.1 | +4.41% | 3.2 |
| YOLOv5s + DoubleSwin | −3.30% | 2.4 | −4.61% | 3.4 |

The paper headlines "~2% mAP@0.5:0.95 for YOLOv5s + SingleSwin". That is **0.9× the
floor** — below the threshold the same paper argues for. A reviewer applying F-A to
Table I reaches this in one step. Options, in descending strength:

1. Run 3 seeds × {YOLOv5s, YOLOv5s+SingleSwin} (~60 h) and report with a spread.
2. Report the architecture results explicitly as within-noise point estimates, and let
   the loss analysis carry the paper.
3. Drop the architecture arm from the contributions and keep it as context.

**`models/yolov5n_swin2.yaml` does not exist** in the repo or its git history, yet
`runs/train/yolov5_n_swin2/opt.yaml` references it. The +4.30% YOLOv5n+DoubleSwin row —
the largest architecture effect in the paper — is currently unreproducible.

**Protocol difference from the original ablation.** These runs use batch 64 on a
300-epoch schedule; the original 50-epoch ablation used batch 16 on a 50-epoch
schedule. The +4.8% is absent here even at epoch 50, but the schedules are not
identical, so "fails to reproduce under a longer, larger-batch protocol" is the precise
statement.

**C1 seed 42 was paused at epoch 172 and resumed** from `last.pt`; the dataloader
reseeds at resume, so that run is not bit-identical to an uninterrupted one. Its result
sits between the other two seeds, so nothing suggests this mattered.

---

## 5. Remaining work before submission

| Task | Cost | Closes |
|---|---|---|
| `ref.bib` beside the `.tex`, BibTeX actually run | minutes | R1.1, R5, R7.6 — **fatal if missed** |
| test-dev evaluation (1610 labelled images, on disk) | minutes | R1.8, R3.3 |
| Per-class `val.py --save-json` for the Swin runs | minutes | R1.4, Table IV |
| Re-caption Figs. 3–4 as YOLOv5s+SingleSwin | minutes | R1.4 (proven by MD5) |
| Table I: 100 → 300 epochs, per-run batch sizes | minutes | R1.3, F-G |
| FPS / latency on stated hardware | ~1 h | R2, R3.2 |
| 3 seeds × {YOLOv5s, +SingleSwin} | ~60 h | the exposure in §4 |

Deferred by choice: WIoU/SIoU comparison rows (R1.7), UAVDT/AI-TOD generalisation
(R3.3), related-work comparison table (R2).

---

## 6. Reproducing everything here

```bash
conda activate yolov5

# the three measurements that need no checkpoints
python tools/measure_box_inflation.py    # F-C: x2.32 (E3), x4.36 (E4)
python tools/measure_layer_split.py      # F-G: P3 share 33.1% -> 68.1%
python tools/measure_complexity.py       # F-F: printed GFLOPs vs aten@640

# the control experiment (3 seeds x 3 configs)
bash run_boxgain_control.sh        # seed 42
bash run_seed_replication.sh       # seeds 43, 44

# per-class numbers
python val.py --img 640 --batch 8 --data data/VisDrone_local.yaml \
  --weights runs/control/C1_e4_loss/weights/best.pt --task val --verbose

# held-out test split
python val.py --img 640 --batch 8 --data data/VisDrone_local.yaml \
  --weights runs/control/C1_e4_loss/weights/best.pt --task test --verbose
```

### What is committed

`runs/control/` carries `results.csv`, `opt.yaml` and `hyp.yaml` for **all nine**
control runs (3 configs × seeds 42/43/44), and `runs/control/perclass_val/` carries the
`val.py --verbose` output for all nine checkpoints.

Every table in this document is therefore regenerable from the repo alone:

| Finding | Verifiable from |
|---|---|
| F-A noise floor | `runs/control/C0_baseline{,_s43,_s44}/results.csv` |
| F-B no aggregate gain | all nine `results.csv` |
| F-C 4.36× inflation | `tools/measure_box_inflation.py` |
| F-D frequency reallocation | `runs/control/perclass_val/*.txt` |
| F-E pedestrian +9.6% | `runs/control/perclass_val/C{0,1,2}*.txt` |
| F-F GFLOPs | `tools/measure_complexity.py` (no data or checkpoints needed) |
| F-G corrections | `runs/train/*/opt.yaml`, `models/swintransformer.py`, `tools/measure_layer_split.py` |

The `best.pt` checkpoints are excluded by `.gitignore` (they are ~14 MB each); the
per-class outputs above stand in for them.
