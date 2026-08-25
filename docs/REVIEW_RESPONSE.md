# INDISCON 2026 — Paper 2887 review response

**Paper ID** 2887
**Title** Small Object Detection Enhancement in UAV Imagery using an Optimized Attention-Driven YOLO Model: A Benchmarking Study on VisDrone2019
**Track** Track 6: Signal Processing, Computing & Data Science
**Venue** INDISCON 2026 (Microsoft CMT)
**Reviews received** #1, #2, #3, #5, #7 (no #4 or #6 in the CMT export)
**This document written** 2026-08-25

Manuscript source lives in a separate repo: `/home/ajith/work/indiscon/INDISCON_2026/`.
This file lives with the code because Findings F1–F6 below are facts about *this* repo.

> **Status of the headline claim.** The paper's primary contribution — "+4.8% mAP@0.5 from the
> custom loss" — is measured at 50 epochs. This repo already contains a 300-epoch rerun of the
> same ablation in which that configuration performs **worse than baseline**. See [F1](#f1).
> Reviewer #1 predicted this without access to the data.

---

## Contents

1. [Reviewer comments (verbatim)](#1-reviewer-comments-verbatim)
2. [Findings from the code and run artifacts](#2-findings-from-the-code-and-run-artifacts)
3. [Reviewer objection → finding → status](#3-reviewer-objection--finding--status)
4. [Remediation steps](#4-remediation-steps)
5. [Status log](#5-status-log)

---

## 1. Reviewer comments (verbatim)

Transcribed exactly from `Review_comments.docx`. Not paraphrased, not reordered.
All five reviewers answered the same prompt:

> *4. Please write detailed Comments for - Suitability of the title and quality of the abstract.
> - Adequacy of literature review and proposed methods -Quality of result analysis and conclusion*

### Reviewer #1

**Critical Submission Error - Missing References:**

> Every citation throughout the paper appears as "[?]" and the reference list is entirely absent. This applies to all citations in the introduction, related work, methodology, and results sections, including citations for VisDrone2019, YOLOv5, Swin Transformer, CIoU loss, WIoU, SAHI, and every other work referenced. This is almost certainly a LaTeX BibTeX compilation failure. The paper cannot be reviewed for citation accuracy or related-work completeness in its current state and cannot be accepted without a complete reference list. This must be corrected before the paper can be accepted at any venue.

**Contribution Magnitude Discrepancy:**

> The contributions bullet in the introduction claims "+2% mAP@0.5:0.95 on YOLOv5s" for the SwinStage integration. The data in Table I shows YOLOv5s improving from 0.188 to 0.191, a difference of 0.003 absolute percentage points, equivalent to approximately 1.6% relative gain. Neither the claimed "+2%" nor the abstract's "~0.3% improvement (1.6% relative)" framing is incorrect in isolation, but they describe the same result using incompatible presentations in adjacent sections of the paper. The contributions bullet should either state the absolute gain ("+0.3 percentage points") or the relative gain ("1.6% relative") consistently with the abstract's presentation.

**Training Epoch Inconsistency Across Ablations:**

> The three core result tables use dramatically different training durations. Table I (architecture ablation) trains for 100 epochs. Table II (loss ablation) trains for 50 epochs. Table III (combined ablation) trains for 300 epochs. The +4.8% mAP@0.5 gain from the E4 loss configuration (Table II) is measured over a baseline of 0.1736, while the same YOLOv5n model trained for 100 epochs achieves 0.263 mAP@0.5 (Table I). The 50-epoch baseline is substantially undertrained relative to the architecture ablation baseline, meaning the +4.8% gain may be an artifact of the short training duration rather than a property of the loss modification itself. Showing the loss ablation at 100 or 300 epochs would demonstrate whether the gain persists at convergence. Without this, the primary loss contribution claim is not reliably supported.

**Figure and Table Inconsistency:**

> Figure 4 is captioned as showing per-class precision-recall curves for "yolov5s_swin2" with "Overall mAP@0.5 = 0.347." Table III reports yolov5s_swin2 baseline mAP@0.5 = 0.3318 (E0). Table IV's "all" row for the same model shows E0 = 0.319. None of these match the 0.347 shown in Fig. 4. Similarly, Fig. 4 shows pedestrian AP = 0.395, while Table IV E0 shows pedestrian AP = 0.379 for the same nominal model. The PR curves and confusion matrix in Figs. 3 and 4 appear to be from a configuration or checkpoint that is not directly described by the numbers reported in Tables III and IV. This inconsistency is unresolved and undermines the reproducibility of the per-class analysis in Section IV-E, which draws conclusions about pedestrian and rare-class behavior from these same results.

**Patch Merging Resolution Impact Not Addressed:**

> The SwinStage insertion is described as "Patch Merging followed by one W-MSA block and one SW-MSA block." Standard Swin Transformer Patch Merging reduces spatial resolution by a factor of 2 (combining 2x2 patches). Inserted before the SPPF layer at the deepest backbone stage (stride 32, approximately 20x20 spatial resolution for a 640x640 input), Patch Merging would produce a 10x10 feature map entering SPPF. For a paper specifically targeting small object detection, a 2x spatial resolution reduction at the deepest feature scale is potentially harmful and should be explicitly analyzed. The paper does not discuss this architectural effect, making it unclear whether the SwinStage is operating with or without downsampling, and how this impacts the SPPF global pooling and subsequent PANet neck.

**Hyperparameter Selection Not Ablated:**

> The E4 configuration uses α=1.5 and β=[3.0, 1.0, 0.4]. These are stated to come from ablation, but Table II presents only five configurations (E0 through E4), where E4 jumps directly to the "strong" hyperparameters without showing intermediate values. No grid search or sensitivity analysis is shown. How sensitive is the +4.8% gain to these specific values? Would α=1.0 with β=[3.0, 1.0, 0.4] approach the same gain? The resolution-aware term alone (E2, +2.2%) accounts for most of the gain; understanding the marginal contribution of the specific α=1.5 tuning is important for practitioners seeking to reproduce or extend this work.

**No Comparison Against Existing Loss Modifications:**

> The related work explicitly discusses WIoU, CFIoU, SIOU, and Scale-Feedback Loss as directly related methods for loss-based small-object improvements. However, the results section contains no comparison against any of these methods under the same experimental setup. The paper argues in Section III-C that the proposed loss is distinguishable from these methods, but a direct empirical comparison would be far more convincing. Adding one row each for WIoU and SIOU in Table II would substantially strengthen the claim that the proposed formulation outperforms existing scale-aware loss approaches.

**Validation-Set-Only Evaluation:**

> All results are reported on the VisDrone2019 validation set. The VisDrone2019 challenge provides a test set (1,610 images) with a public evaluation server for blind assessment. No test set results are reported. While validation-set results are standard for ablation studies, the absence of any held-out test evaluation limits confidence that the reported gains are not the result of implicit hyperparameter overfitting to the validation set, particularly given that α and β were selected to maximize validation performance.

### Reviewer #2

> **bstract** *(sic)*
> Better highlight the novelty and practical contribution.
> Briefly mention the improvement over recent state-of-the-art methods.
>
> **Introduction**
> Clearly define the research gap with respect to recent UAV detection models.
> Explicitly summarize the paper's contributions.
>
> **Related Work**
> Include recent literature (2024–2026).
> Add a comparative summary table of existing approaches.
>
> **Methodology**
> Provide hyperparameter sensitivity analysis.
> Discuss computational complexity and resource requirements.
> Include algorithmic pseudocode or workflow summary.
>
> **Results**
> Include statistical significance analysis.
> Compare with additional state-of-the-art detectors.
> Add inference speed (FPS) and latency evaluation on edge hardware.
>
> **Discussion**
> Expand discussion on deployment challenges, scalability, and rare-class detection.
> Include practical implications for UAV surveillance applications.
>
> **Conclusion**
> Emphasize scientific contributions and practical deployment.
> Discuss future work on adaptive loss weighting and lightweight transformer integration.
>
> **References**
> Include more recent IEEE, Springer, ACM, and Elsevier publications (2024–2026).
> Ensure consistent IEEE reference formatting throughout the manuscript.

### Reviewer #3

> 1. The authors should test or formalize their proposed solution within this paper rather than leaving it purely as future work, as the current net-zero gain in Table III weakens the combined model claim.
> 2. Include latency (FPS / inference time in ms) and FLOPs comparisons across GPUs and edge hardware.
> 3. Include evaluation on the official VisDrone2019 test-dev set and optionally validate generalization on a secondary UAV dataset (e.g., UAVDT or AI-TOD).

### Reviewer #5

> It has a clear research question, identifiable methodological contributions, appropriate mathematical formulation. I don't know why References are missing.

### Reviewer #7

> 1. Suitability of the title and quality of the abstract. - OK
> 2. Adequacy of literature review and proposed methods - NO
> 3. Quality of result analysis and conclusion - OK
> 4. There is scope of LANGUAGE improvement.
> 5. Validation may add value to the work.
> 6. No references included.

---

## 2. Findings from the code and run artifacts

Every number below carries the path it came from, so nothing here has to be taken on trust.

<a name="f1"></a>
### F1 — The headline "+4.8%" does not survive to convergence; it reverses

`runs/ablation_local (1)/` holds a **300-epoch rerun of the identical 5-experiment loss
ablation** that the paper reports at 50 epochs from `runs/ablation_local/ablation/`.

The runs are a clean controlled comparison — `opt.yaml` differs **only** in the loss flags:

```bash
diff <(sed 's/name:.*//;s/save_dir:.*//' "runs/ablation_local (1)/E0_baseline5/opt.yaml") \
     <(sed 's/name:.*//;s/save_dir:.*//' "runs/ablation_local (1)/E4_both_strong/opt.yaml")
# only: scale_aware_loss, resolution_weighting, scale_alpha, resolution_beta
```

| Exp | Config | 50 ep (paper Table II) | **300 ep (on disk)** | rel. vs E0 @300 |
|---|---|---|---|---|
| E0 | baseline CIoU | 0.1736 | 0.2572 | — |
| E1 | scale only | 0.1745 (+0.5%) | 0.2581 | +0.4% |
| E2 | res only | 0.1775 (+2.2%) | 0.2596 | +0.9% |
| E3 | both, default β | 0.1776 (+2.3%) | **0.2610** | **+1.5%** |
| E4 | both, strong (α=1.5, β=[3,1,0.4]) | **0.1819 (+4.8%)** | **0.2560** | **−0.5%** |

Sources: `runs/ablation_local (1)/{E0_baseline5,E1_scale_only4,E2_res_only2,E3_both2,E4_both_strong}/results.csv`

At convergence the paper's headline configuration is **worse than baseline**, and the best
configuration is **E3, not E4**.

### F2 — Table I is labelled 100 epochs; every run in it was 300

`opt.yaml` for all six architecture-ablation runs reads `epochs: 300`. The values match the
paper's Table I to three decimals:

| Paper Table I row | Actual run | epochs | batch | mAP@.5 / .5:.95 |
|---|---|---|---|---|
| YOLOv5n | `runs/train/yolov5_n_no_swin` | 300 | 64 | 0.2636 / 0.1340 |
| YOLOv5s | `runs/train/yolov5_s_no_swin` | 300 | 64 | 0.3415 / 0.1885 |
| YOLOv5n+SingleSwin | `runs/train/yolov5_n_swin` | 300 | 64 | 0.2667 / 0.1363 |
| YOLOv5s+SingleSwin | `runs/train/yolov5_s_swin` | 300 | 16 | 0.3484 / 0.1910 |
| YOLOv5n+DoubleSwin | `runs/train/yolov5_n_swin2` | 300 | 64 | 0.2749 / 0.1399 |
| YOLOv5s+DoubleSwin | `runs/train/yolov5_s_swin2` | 300 | 64 | 0.3302 / 0.1799 |

The real epoch story is **300 / 50 / 300**, not 100 / 50 / 300. This *narrows* Reviewer #1's
objection to the loss ablation alone. The paper's `batch size 16 throughout` is also wrong —
only one run used 16.

### F3 — The SwinStage contains no Patch Merging

`models/swintransformer.py:298` — `SwinStage.__init__` builds **only** `SwinTransformerBlock`s.
`forward()` does `B,C,H,W → tokens → blocks → view(B, H, W, C)`: **spatial resolution is exactly
preserved**, and `assert dim == c2` forces channels unchanged.

`PatchMerging` is defined at `models/swintransformer.py:429` and imported at
`models/yolo.py:54`, but is **never instantiated** and no YAML references it — dead code.

```bash
grep -rn "PatchMerging" --include=*.py --include=*.yaml .
# models/swintransformer.py:429:class PatchMerging(nn.Module):     <- definition
# models/yolo.py:54:from models.swintransformer import SwinStage, PatchMerging, PatchEmbed
```

The paper (Section III-A) states the stage comprises "Patch Merging followed by one W-MSA block
and one SW-MSA block". **That is not what the code does.** Correcting the sentence dissolves
Reviewer #1's concern about 20×20 → 10×10 downsampling: the objection is valid given what the
paper claims, but the implementation never had the problem.

### F4 — "DoubleSwin" is not two consecutive SwinStages

`models/yolov5s_swin2.yaml` places SwinStages at **layer 2 (P2/4)** and **layer 9 (P5/32)** —
opposite ends of the backbone, not consecutive as the paper describes.
`models/yolov5m_swin.yaml` likewise has two (P2 depth 2, P5 depth 3) despite its name.
Only `models/yolov5_swin.yaml` matches the paper's "single stage before SPPF" description.

Separately, `runs/train/yolov5_n_swin2/opt.yaml` references `models/yolov5n_swin2.yaml`, which
**does not exist and is not in git history**. The YOLOv5n+DoubleSwin row is not reproducible.

### F5 — Figs. 3 and 4 come from the wrong run (byte-identical proof)

MD5 of the manuscript's `confusion_matrix.png` and `PR_curve.png` match `runs/train/yolov5_s_swin/`
**exactly** — the **YOLOv5s+SingleSwin** architecture run (mAP@0.5 = 0.3484, consistent with the
0.347 in the Fig. 4 caption), *not* `yolov5s_swin2` as both captions claim (Table III E0 = 0.3318).

```bash
md5sum /home/ajith/work/indiscon/INDISCON_2026/PR_curve.png runs/train/yolov5_s_swin/PR_curve.png
# 9aeadb1ee39ffb9d7362af51ddc7c0d8  (both)
```

Reviewer #1 deduced this from the numbers alone and is correct.

Related: `runs/ablation_swin_val/*` saved **only PNGs — no numeric per-class output** (no
`--save-json`, no captured console log). Table IV's per-class APs are currently unreproducible
from artifacts.

### F6 — The loss inflates total box loss ~5.6×; it is not a pure re-allocation

`utils/loss.py` computes `scale_w * (1 - iou)` per target, then `* beta[layer]`
(`apply_resolution_weight`, line 140), then `lbox *= self.hyp["box"]` (line 321) — with **no
renormalization**. For a small object the scale weight is ≈ 2α, so with a typical P3/P4/P5
box-loss split of 50/30/20:

| Config | total box loss vs baseline | P3 share of box gradient |
|---|---|---|
| baseline | ×1.00 | 50% |
| E3 (α=1.0, β=[2,1,0.5]) | **×2.80** | 71% |
| E4 (α=1.5, β=[3,1,0.4]) | **×5.64** | 80% |

The paper's "~50% → ~79%" P3-share claim checks out arithmetically. But it never mentions that
E4 also multiplies the **total** box loss by 5.6×, shifting the box/obj/cls balance and
effectively raising the box-branch learning rate. **The reported gain is confounded** — it may
be "louder box loss" rather than "scale awareness", and that is the most plausible mechanism
behind F1: over-weighting helps escape early, then over-fits localization by convergence.

---

## 3. Reviewer objection → finding → status

| # | Objection | Finding | Status |
|---|---|---|---|
| R1.1 | Missing references / `[?]` | — | **Fixed.** Build issue, not source. `.bbl` was absent or BibTeX never ran. Full reference audit completed separately; all 23 citations resolve. |
| R1.2 | "+2%" vs "+0.3 pp / 1.6% relative" | — | **Confirmed, open.** Contradiction appears in three places (contributions bullet, results text, conclusion) against the paper's own Table I. |
| R1.3 | Training epoch inconsistency | [F1](#f1), F2 | **Confirmed, and worse than stated.** Table I is 300 ep not 100; and the 300-ep loss ablation already on disk shows the gain *reverses*. |
| R1.4 | Fig. 4 vs Tables III/IV mismatch | F5 | **Confirmed.** Figures are from `yolov5_s_swin`, proven by MD5. |
| R1.5 | Patch Merging halves resolution | F3 | **Refuted.** No Patch Merging in the implementation; resolution is preserved. The paper's description is wrong, not the architecture. |
| R1.6 | Hyperparameters not ablated | F6 | **Confirmed, open.** Step 1 control run addresses the mechanism; a proper α/β grid is still missing. |
| R1.7 | No comparison vs WIoU / SIoU / CFIoU | — | **Open, deferred.** Needs new training runs. |
| R1.8 | Validation-set-only evaluation | — | **Addressable now.** Labelled test-dev split is on disk — see Step 2. |
| R2 | Complexity / resource requirements | — | **Addressable now.** Step 4. |
| R2 | Statistical significance | — | **Open.** Single run per config gives no variance estimate. Either 3 seeds or drop the significance language (the paper currently asserts "statistically meaningful" with no test). |
| R2 | FPS / latency on edge hardware | — | **Addressable.** Step 4, on the test box. |
| R2 | Recent literature 2024–2026, comparison table | — | **Open, deferred.** |
| R2 | Pseudocode / workflow summary | — | **Open.** Cheap to add. |
| R3.1 | Net-zero gain in Table III weakens the claim | [F1](#f1), F6 | **Confirmed and sharpened.** The net-zero result is consistent with F1: the loss does not help at convergence. |
| R3.2 | Latency / FLOPs across GPUs | — | **Addressable.** Step 4. |
| R3.3 | test-dev + secondary dataset | — | **Partly addressable.** test-dev now; UAVDT/AI-TOD deferred. |
| R5 | References missing | — | **Fixed.** Same as R1.1. |
| R7.4 | Language improvement | — | **Fixed.** Grammar/punctuation pass completed separately. |
| R7.6 | No references | — | **Fixed.** Same as R1.1. |

---

## 4. Remediation steps

### Step 1 — Control run that settles F6 *(the one experiment worth GPU time)*

Train baseline CIoU with **no** scale/resolution flags, but the `box` gain scaled by E4's
inflation factor: `0.05 × 5.64 ≈ 0.282`. Same seed, same 300 epochs, same `yolov5n.yaml`.

```bash
cp data/hyps/hyp.scratch-low.yaml data/hyps/hyp.boxgain-e4.yaml
# edit the copy:  box: 0.05  ->  box: 0.282

python train.py --img 640 --batch 64 --epochs 300 \
  --data data/VisDrone.yaml --cfg models/yolov5n.yaml \
  --hyp data/hyps/hyp.boxgain-e4.yaml --seed 42 \
  --name E5_boxgain_control --project "runs/ablation_local (1)"
```

Optionally repeat at E3's factor (`box: 0.14`).

**Reading the result:**
- E5 ≈ E4 → the weighting contributes nothing beyond box-loss magnitude; reframe the
  contribution accordingly.
- E5 ≈ E0 → the weighting does real work and E3's +1.5% is genuine, if modest.

Either outcome is publishable and both answer R1.6.

### Step 2 — Regenerate numbers that exist only as PNGs

No retraining — `val.py` on saved checkpoints. Makes Table IV reproducible and resolves F5.

```bash
python val.py --img 640 --batch 8 --data data/VisDrone.yaml \
  --weights "runs/ablation_swin/E0_swin_small_baseline/weights/best.pt" \
  --verbose --save-json --task val \
  --name E0_perclass_numeric --project runs/ablation_swin_val
```

Repeat for all four `runs/ablation_swin/*/weights/best.pt`, **and** for
`runs/train/yolov5_s_swin/weights/best.pt` — the checkpoint the paper's figures actually
came from.

Then add held-out test results with `--task test`. The labelled split is on disk:
`/mnt/mydrive/ajith/data_set/VisDrone/VisDrone2019-DET-test-dev` (1610 images, `labels/`
present). Closes R1.8 and R3.3 at near-zero cost.

### Step 3 — Manuscript corrections

In `/home/ajith/work/indiscon/INDISCON_2026/conference_INDISCON_PTP.tex`:

| Finding | Fix |
|---|---|
| F1 | Rebuild Table II from the 300-epoch runs. Headline becomes **E3, +1.5%**, not E4 +4.8%. Update abstract, contributions bullet, conclusion. Add a paragraph reporting that the 50-epoch gain does not persist — stating it yourself is far stronger than having it found again. |
| F2 | Table I caption and Section IV-A: 100 → **300 epochs**; fix "batch size 16 throughout" using per-run `opt.yaml` values. |
| F3 | Section III-A: delete "Patch Merging followed by"; state the stage is **resolution-preserving**, and note this deliberately avoids downsampling at the deepest scale — a design strength for small objects, and a direct rebuttal to R1.5. |
| F4 | Correct the Double-variant description: stages sit at **P2/4 and P5/32**, not consecutively. Restore `models/yolov5n_swin2.yaml` or drop the YOLOv5n+DoubleSwin row as unreproducible. |
| F5 | Re-caption Figs. 3–4 as **YOLOv5s+SingleSwin** (cheapest, and 0.347 then matches Table I), or regenerate them from `ablation_swin/E0_swin_small_baseline` to match Table III. |
| F6 | Add the effective-multiplier table to Section III-C and report the Step 1 control, converting a hidden confound into a deliberate ablation. |

Also still open from the language pass: the "+2%" contradiction (R1.2), `19.1` vs `0.191`,
and the mixed percentage-point / relative-% Δ columns in Tables II and IV.

### Step 4 — Cheap additions

- **Params / FLOPs / model size** — YOLOv5 prints these at model build; no dataset needed.
  Covers R2 (computational complexity) and R3.2.
- **FPS / latency** — measure on the test box; state the hardware explicitly.
- **Statistical significance** — run 3 seeds for E0/E3, or remove the significance language.

**Deliberately deferred:** WIoU/SIoU comparison rows (R1.7), UAVDT/AI-TOD generalization (R3.3),
and the Related Work comparison table (R2). Each needs substantial new training and belongs in a
revision cycle after Steps 1–3 settle what the actual contribution is.

---

## 5. Status log

| Date | Step | Action | Result |
|---|---|---|---|
| 2026-08-25 | — | Repo audit; F1–F6 established | This document |
| 2026-08-25 | 3 | Language + reference pass on the `.tex` and `.bib` | Applied; needs a BibTeX rebuild to take effect |
| | 1 | E5 box-gain control run | _pending_ |
| | 2 | Per-class `val.py` regeneration | _pending_ |
| | 2 | test-dev evaluation | _pending_ |
| | 3 | Manuscript corrections F1–F6 | _pending_ |
| | 4 | FLOPs / FPS / seeds | _pending_ |

---

## Framing note

Steps 1–3 will make the paper's claimed contribution substantially smaller: roughly **+1.5%
relative mAP at convergence from E3**, plus a resolution-preserving Swin stage worth **~+1.6%
relative on YOLOv5s**. That is a modest but defensible result, and it is what the data supports.
The current draft's stronger claims rest on a 50-epoch comparison that the 300-epoch runs in
this repo contradict — and Reviewer #1 is already looking directly at that seam.
