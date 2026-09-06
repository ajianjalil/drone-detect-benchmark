# Numbers-only errata — paper 2887

**Scope: change numbers, leave structure, claims and section order untouched.**
Every row gives old → new and the artifact the new value comes from. Nothing here
requires rewriting a paragraph *except* the two rows marked **SIGN**, which cannot be
swapped in place because the measured value has the opposite sign to the sentence
around it.

Locations are by section/table, not line number — line numbers in
`documentation/main.tex` (an earlier draft) will not match the submitted source.

---

## A. Pure number fixes — safe to substitute directly

| # | Where | Old | New | Source |
|---|---|---|---|---|
| A1 | Table I caption | "trained for 100 epochs" | **300 epochs** | all six `runs/train/*/opt.yaml` read `epochs: 300`; each `results.csv` has 300 rows |
| A2 | Sec. IV-A, batch size | "batch size 16 throughout" | **64 for all runs except `yolov5_s_swin`, which used 16** | per-run `opt.yaml` |
| A3 | Sec. IV-A, methodology | `[Specify LR, Optimizer, Batch Size, Epochs - Placeholder]` | **LR 0.01, SGD (momentum 0.937, wd 5e-4), batch per A2, 300 epochs, 640×640** | `runs/train/*/hyp.yaml` |
| A4 | Table I, YOLOv5s+SingleSwin | `19.1` | **0.191** | unit inconsistency — every other cell in that column is a fraction |
| A5 | Table III, E3 row | **0.3647** bolded | unbold — 0.3655 (E2) is higher | Table III's own values |
| A6 | Sec. III-C, P3 gradient share | "~50% → ~79%" | **33.1% → 68.1%** | `tools/measure_layer_split.py`, 640 augmented images |
| A7 | Δ columns, Tables II/IV/V | mixed relative % | **percentage points**, relative in parentheses | see §C |
| A8 | Figs. 3–4 captions | "yolov5s_swin2 … mAP@0.5 = 0.347" | **YOLOv5s+SingleSwin** (`runs/train/yolov5_s_swin`, mAP@0.5 = 0.3484) | MD5 of the manuscript PNGs matches that run's `PR_curve.png` and `confusion_matrix.png` byte-for-byte |
| A9 | Table IV / Fig. 4 / Table III | three different mAP@0.5 for "yolov5s_swin2": 0.3318, 0.319, 0.347 | 0.347 belongs to **YOLOv5s+SingleSwin** (A8). Keep 0.3318 (Table III) and reconcile 0.319 against the Table IV per-class mean | as A8 |

## B. Numbers that reverse sign — cannot be swapped in place

| # | Where | Paper says | Measured | n |
|---|---|---|---|---|
| **B1 SIGN** | Abstract, Contribution bullet 2, Sec. IV-B, Conclusion | loss improves mAP@0.5 **"by up to 4.8%"** | **−2.3%** at 300 ep; **−5.0%** in a completed 50-epoch schedule | 3 seeds |
| **B2 SIGN** | Abstract, Contribution bullet 1, Sec. IV-A, Conclusion | SwinStage gives **"~2% mAP@0.5:0.95"** | **−0.96% ± 1.58** mAP@0.5, **−2.27%** mAP@0.5:0.95 | 3 seeds |

**Why these two are different from §A.** "+4.8%" → "−2.3%" inside *"these loss
modifications improve overall mAP@0.5 by up to 4.8%"* yields a sentence that says the
improvement is negative. The number and the verb have to agree.

Three ways to handle each **without restructuring the paper**, in ascending honesty:

1. **Keep the original number, add the budget.** The paper's 0.1736 → 0.1819 is a real
   measurement from a real 50-epoch run; it is not fabricated. Add the plan's Fix-7
   sentence to Sec. IV-A so the reader knows the budget differs by experiment. Changes
   no claim, adds one sentence. **Weakest, but survives if nobody replicates.**
2. **Keep the number, add one clause**: "…by up to 4.8% at 50 epochs; this gain does not
   persist at 300 epochs." One clause per location, four locations. Reviewer #1 already
   predicted the non-persistence, so pre-empting it is worth more than it costs.
3. **Replace with the per-class result**, which is positive, larger than claimed, and
   reproducible: **pedestrian AP +9.6%** (3 seeds, disjoint distributions). This changes
   the sentence but not the section, table order, or argument.

Same three options apply to B2.

## C. Unit convention (plan §2, fix 1)

Use **percentage points** for all Δ, relative in parentheses where wanted.

| Quantity | Absolute | pp | Relative |
|---|---|---|---|
| Loss E4 vs E0 (YOLOv5n, 50 ep) | 0.1736 → 0.1819 | **+0.83 pp** | +4.8% |
| SwinStage on YOLOv5s (mAP@.5:.95) | 0.188 → 0.191 | **+0.3 pp** | +1.6% |
| Pedestrian AP (s_swin2) | 0.379 → 0.410 | **+3.1 pp** | +8.2% |
| Pedestrian AP (m_swin) | 0.436 → 0.468 | **+3.2 pp** | +7.3% |
| Bus AP (s_swin2) | 0.388 → 0.342 | **−4.6 pp** | −11.9% |

This alone fixes R1.2: the abstract's "+2%" and "~0.3% (1.6% relative)" are the same
number in two conventions.

## D. New numbers available if you want them (no structural change)

**Naming.** The run directories are `A_baseline` … `D_swin_p2_loss`, but use these
names in the paper — the plan's final checklist asks for a naming convention defined
once and used consistently:

| run dir | **name to use in the paper** | what it is |
|---|---|---|
| `A_baseline` | **YOLOv5s** | stock, detection heads at P3/P4/P5 (strides 8/16/32) |
| `B_p2head` | **YOLOv5s-P2** | adds a fourth detection head at P2 (stride 4). No attention. |
| `C_swin_p2` | **YOLOv5s-P2 + Swin** | YOLOv5s-P2 plus one SwinStage on the P2 *neck* branch |
| `D_swin_p2_loss` | **YOLOv5s-P2 + Swin + custom loss** | as above, with α=1.5, β=[3.0, 2.0, 1.0, 0.4] |

All four: **100 epochs, batch 12, seed 42, 640×640, SGD** — one budget, one batch, one
seed. This is the single-budget comparison the paper currently lacks. Drops into
Table I as extra rows without touching the surrounding text.

| Model | P | R | mAP@.5 | mAP@.5:.95 | Params | GFLOPs | Size (MB) | FPS |
|---|---|---|---|---|---|---|---|---|
| YOLOv5s | 0.448 | 0.333 | 0.3270 | 0.1784 | 7,046,599 | 15.8 | 14.1 | 244 |
| **YOLOv5s-P2** | 0.466 | 0.373 | **0.3708** | **0.2026** | 7,192,244 | 18.7 | 14.4 | **182** |
| YOLOv5s-P2 + Swin | 0.473 | 0.373 | 0.3719 | 0.2051 | 7,395,780 | 30.4 | 14.8 | 72 |
| YOLOv5s-P2 + Swin + loss | 0.435 | 0.370 | 0.3597 | 0.2019 | 7,395,780 | 30.4 | 14.8 | 71 |

FPS measured on RTX 2000 Ada, 640×640, batch 8, inference only (excludes NMS).

**The three comparisons:**

| | mAP@.5 | mAP@.5:.95 | cost |
|---|---|---|---|
| **P2 head** (YOLOv5s-P2 − YOLOv5s) | **+4.38 pp (+13.4%)** | **+2.42 pp (+13.5%)** | +145 K params (+2%), 244 → 182 FPS |
| **Swin** (+Swin − YOLOv5s-P2) | +0.12 pp (+0.3%) | +0.25 pp (+1.2%) | +204 K params, +11.7 GFLOPs (+62%), **182 → 72 FPS** |
| **custom loss** (+loss − +Swin) | −1.22 pp (−3.3%) | −0.32 pp (−1.6%) | free |

The gain is **recall**: 0.333 → 0.373 (+4.0 pp) for the P2 head, precision only
0.448 → 0.466. A stride-4 head finds small objects the P3/P4/P5 heads cannot resolve —
at stride 32 a VisDrone pedestrian is 0.17 × 0.53 feature cells, at stride 4 it is
1.37 × 4.27.

**Per-class AP@0.5** (sorted by instance count; last two columns in percentage points):

| class | instances | YOLOv5s | YOLOv5s-P2 | P2 gain | + Swin | Swin gain |
|---|---|---|---|---|---|---|
| car | 14064 | 0.715 | 0.774 | +5.9 | 0.774 | +0.0 |
| pedestrian | 8844 | 0.380 | 0.446 | **+6.6** | 0.445 | −0.1 |
| people | 5125 | 0.318 | 0.350 | +3.2 | 0.353 | +0.3 |
| motor | 4886 | 0.376 | 0.428 | +5.2 | 0.426 | −0.2 |
| van | 1975 | 0.346 | 0.385 | +3.9 | 0.381 | −0.4 |
| bicycle | 1287 | 0.108 | 0.129 | +2.1 | 0.121 | −0.8 |
| tricycle | 1045 | 0.189 | 0.217 | +2.8 | 0.215 | −0.2 |
| truck | 750 | 0.297 | 0.339 | +4.2 | 0.342 | +0.3 |
| awning-tricycle | 532 | 0.118 | 0.123 | +0.5 | 0.120 | −0.3 |
| bus | 251 | 0.356 | 0.446 | **+9.0** | 0.427 | −1.9 |

**Every class improves with the P2 head**, including the rarest (bus, 251 instances,
+9.0 pp — the largest single gain). This is a different signature from the custom loss,
which gained on frequent classes and lost 6–20% on rare ones: the P2 head **adds
capability rather than reallocating it**, which is a far easier claim to defend.

Attention changes nothing per-class — ±0.4 pp on nine of ten classes, −1.9 on bus.

Also correctable, and it helps the paper: **GFLOPs for every Swin variant is overstated
~2.2×** by YOLOv5's printed figure (it profiles at 32×32 and scales by (640/32)², which
windowed attention does not obey). Correct values from `tools/measure_complexity.py`:
YOLOv5n+SingleSwin **5.4** not 11.8 · YOLOv5s+DoubleSwin **22.6** not 47.8 ·
YOLOv5m+DoubleSwin **73.4** not 158.3.

## E. Not a number, but fatal

`ref.bib` must sit beside the `.tex` and BibTeX must actually run. Every citation
renders `[?]` without it; three reviewers raised this independently and one wrote that
the paper "cannot be accepted without a complete reference list."
