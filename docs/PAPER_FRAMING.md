# How to reframe paper 2887

Written 2026-08-27, after the Step 1 control runs. Reads
[REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) (the audit) and
[STEP1_MEASUREMENTS.md](STEP1_MEASUREMENTS.md) (the local evidence) as given.

---

## 1. What survives, and what does not

| Claim in the submitted paper | Status |
|---|---|
| Custom loss gives +4.8% mAP@0.5 | **Dead.** Does not reproduce at 300 epochs, and is negative even at 50. The effect is dominated by an unreported ~4.4× box-loss inflation which is itself harmful. |
| SwinStage improves YOLOv5s | **Alive but smaller than stated.** +2.04% rel mAP@0.5, +1.27% rel mAP@0.5:0.95. |
| SwinStage described as Patch Merging + W-MSA + SW-MSA | **Wrong, and the correction helps.** No Patch Merging exists; the stage is resolution-preserving. |
| "DoubleSwin offers no benefit" | **Wrong in one direction.** It is the best nano variant (+4.30%) and the worst small variant (−3.30%). |
| Swin variants' GFLOPs | **Overstated ~2.2×** by YOLOv5's printed figure. |

The paper currently leads with the one claim that is dead and undersells the two
findings that are real.

## 2. The reframe in one sentence

> A benchmarking study of attention placement in YOLOv5 for UAV small-object
> detection, which finds a modest but consistent gain from a **resolution-preserving**
> Swin stage, and shows that a plausible scale-aware loss modification's apparent
> benefit is an **artifact of box-loss magnitude** rather than scale awareness.

Two contributions, both defensible, and the second is more interesting than the
result it replaces. Loss-modification papers in this area (WIoU, CFIoU, SIOU,
Scale-Feedback) routinely change the effective magnitude of the box term without
controlling for it. A clean demonstration that the confound can fully account for an
apparent gain is a genuine methodological contribution, and it directly answers
R1.6, R1.3 and R3.1 instead of deflecting them.

## 3. Concrete edits

**Title.** Drop "Optimized" and "Enhancement" — both now oversell. Something closer
to *"Attention Placement and Loss Weighting for Small-Object Detection in UAV
Imagery: A Controlled Benchmarking Study on VisDrone2019."*

**Abstract.** Remove "+4.8%". State the Swin gain in one consistent form
(recommend absolute percentage points *and* relative once, e.g. "+0.7 pp / +2.0%
relative mAP@0.5"), then state the negative loss finding as a contribution. This
also closes R1.2, which is currently a contradiction across three sections.

**Contributions bullets.** Three, in this order:
1. A 300-epoch benchmark of Swin stage placement (P5-only vs P2+P5) across two
   model scales, with corrected FLOPs.
2. Evidence that the scale/resolution-aware box loss yields no gain at convergence,
   with a magnitude-matched control isolating the cause.
3. Held-out test-dev results, not validation-only.

**Section III-A (architecture).** Delete "Patch Merging followed by". State that the
stage is resolution-preserving and that this is deliberate: at stride 32 the P5 map
is ~20×20, and halving it would discard exactly the spatial detail small objects
depend on. This converts R1.5 from an unanswered objection into a design rationale.

**Section III-C + IV (loss).** Rewrite as a controlled ablation with the C0/C1/C2
table. Report the ~4.4× inflation as a measured property of the formulation. Replace
the "~50% → ~79%" P3-share figures with the measured 33.1% → 68.1%.

**Table I.** 100 → **300 epochs**. Fix per-run batch sizes from each `opt.yaml`
rather than the blanket "16 throughout". Either restore `models/yolov5n_swin2.yaml`
or drop the YOLOv5n+DoubleSwin row — as it stands the single best architecture
result in the paper rests on a config file that is not in the repo or its history.

**Figures 3–4.** Re-caption as **YOLOv5s+SingleSwin**, which also makes the 0.347 in
the caption agree with Table I. Regenerate per-class numbers with `val.py
--save-json` so Section IV-E is reproducible.

**References.** The `[?]` failure is a build problem, not a content problem. Confirm
`ref.bib` is beside the `.tex` and that BibTeX actually runs before resubmission.
Three reviewers raised this; it is the cheapest fix and the most fatal if missed.

## 4. What still needs running

| Item | Cost | Answers |
|---|---|---|
| 3-seed replication of C0/C1/C2 | ~36 h (running) | R2 significance |
| test-dev evaluation | minutes, needs checkpoints | R1.8, R3.3 |
| Per-class `val.py --save-json` | minutes, needs checkpoints | R1.4, Table IV |
| FPS / latency | ~1 h | R2, R3.2 |

Deliberately deferred: WIoU/SIoU comparison rows, UAVDT/AI-TOD, related-work table.

## 5. The honest risk

The paper as submitted contains a headline claim the authors now know does not
reproduce. That cannot be defended in a rebuttal, and it should not be quietly
softened either — Reviewer #1 already identified the exact seam and will read the
revision looking at it.

If INDISCON's revision stage permits a change of this size, the reframe above is
strictly stronger than the submission: it answers R1.3, R1.6, R3.1 and R1.5 with
evidence rather than argument. If the venue only allows minor revision, the
contribution being claimed no longer matches the paper, and withdrawal followed by
resubmission of the reframed study is the more defensible route.

Either way, the substantive work is done — what remains is mostly rewriting, plus
evaluations that cost minutes once the checkpoints are recovered.
