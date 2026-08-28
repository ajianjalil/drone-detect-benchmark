# Paper 2887 — the narrative after the experiments

2026-08-28. Supersedes the framing in [PAPER_FRAMING.md](PAPER_FRAMING.md) §2, which
was written before the seed replication. Evidence:
[STEP1_MEASUREMENTS.md](STEP1_MEASUREMENTS.md).

---

## The finding that reorganises the paper

Three seeds of the yolov5n baseline give a **relative spread of 1.4%**
(0.2612 / 0.2641 / 0.2605; range 0.0036 on a mean of 0.2619). That is the noise floor
for a single VisDrone run under this protocol.

Almost every headline number in the submitted paper sits at or below it.

| Claim | Reported | vs noise floor | Status |
|---|---|---|---|
| Loss E4 | +4.8% mAP@0.5 | 3.4× | **Does not reproduce** (n=3: −2.3%) |
| Loss E2 res-only | +2.2% | 1.6× | Marginal even as reported |
| Loss E1 scale-only | +0.5% | 0.4× | Was never separable from noise |
| **YOLOv5s+SingleSwin** | **~2% mAP@0.5:0.95** | **0.9×** | **Below the floor on its own metric** |
| YOLOv5n+SingleSwin | +1.17% mAP@0.5 | 0.9× | Within noise |
| YOLOv5n+DoubleSwin | +4.30% | 3.1× | Separable — but config file missing |
| YOLOv5s+DoubleSwin | −3.30% | 2.4× | Separable, and negative |

The architecture ablation is n=1 per row, so these are point estimates against a floor
measured on a different model size. The direction is unmistakable regardless: **the
paper's two positive contributions are both within run-to-run variance, and the only
statistically solid architecture results are one that is unreproducible and one that is
negative.**

This is uncomfortable, and it is also the paper's way out. The submitted draft is a
weak "we propose X and it works" paper. What the data supports is a much better
paper about *why* results like X keep appearing.

## The narrative

> Two families of modification dominate recent small-object UAV detection work:
> attention blocks inserted into the backbone, and scale-aware reweighting of the box
> regression loss. Both are typically validated with single runs, short schedules, and
> effect sizes of one to five percent. We benchmark both under one controlled protocol
> on VisDrone2019 — 300 epochs, three seeds, fixed hardware — and find that the
> reported effect sizes in this regime are not reliably separable from seed variance.
> We then take the loss modification apart and show the mechanism behind one such
> apparent gain: the reweighting silently multiplies total box loss by 4.4×, and a
> magnitude-matched control with no reweighting at all reproduces most of its effect.
> The scale awareness the method is credited for contributes little; what it mostly
> does is change the box/objectness balance.

Three claims, all defensible:

1. **A measured noise floor** (1.4%) for VisDrone single-run comparisons, and the
   observation that much of the published improvement in this class falls inside it.
2. **A mechanism** for one apparent gain — the 4.4× box-loss inflation — established
   by a magnitude-matched control, not argued.
3. **A corrected efficiency picture** — YOLOv5's printed GFLOPs overstates every Swin
   variant by ~2.2×, because the 32×32 extrapolation does not hold for windowed
   attention.

## What each reviewer objection becomes

| Objection | Under the new narrative |
|---|---|
| R1.3 epochs inconsistent | Becomes the paper's subject, not a defect. |
| R1.6 hyperparameters not ablated | Answered by the control: α/β sensitivity is moot when the effect is magnitude. |
| R3.1 net-zero gain weakens the claim | The net-zero gain *is* the finding. |
| R2 statistical significance | The noise floor is now the paper's opening move. |
| R1.5 Patch Merging halves resolution | Refuted — the stage is resolution-preserving. |
| R1.7 no comparison vs WIoU/SIoU | Reframed: those methods share the uncontrolled-magnitude problem. Naming it is a contribution; benchmarking them is future work. |
| R1.8 validation-only | Test-dev is on disk, 1610 labelled images. |

## Revision, 2026-08-28: the per-class result changes the loss story

The section above was written from aggregate mAP alone and is too harsh on the loss.
Per-class evaluation ([PERCLASS_FINDING.md](PERCLASS_FINDING.md)) shows the loss has a
large, seed-stable, mechanistically explicable effect that aggregate mAP averages away:

- **pedestrian +9.6%** across 3 seeds with *disjoint* distributions (C0 0.324/0.324/0.328
  vs C1 0.359/0.356/0.355) — the single most solid result in the study, and 3× what the
  paper claims for it
- people +4.8%, motor +4.5%, car +1.6%
- rare classes lose 6–20%
- the change correlates with **log instance count (r=+0.688)**, not box area (r=−0.365)

So the loss is a **frequency reallocation**, not a small-object prior, and it is a real
contribution — just not the one the paper claims. The narrative's three claims stand;
this becomes a fourth, and the strongest of them empirically.

## What must not be claimed

- Not "our loss improves detection **overall**." Aggregate mAP is unchanged at best.
- Not "E4 is worse than baseline." Three seeds show C1 below C0 on mAP@0.5 at every
  seed, but the mean deficit is only ~1.7× the floor and narrows across seeds; on
  mAP@0.5:0.95 one seed is positive. The supported claim is **no aggregate benefit**,
  not harm.
- Not a **small-object** mechanism. The data says frequency. `car` is the largest
  frequent class and gains; `bicycle` is small and loses 19.5%.
- Not "SwinStage improves YOLOv5s by ~2% mAP@0.5:0.95." At n=1 and 0.9× the floor,
  that is not established.
- Not the YOLOv5n+DoubleSwin row at all, until `models/yolov5n_swin2.yaml` is restored.

## What *should* be claimed about the loss

> The proposed weighting produces a consistent, seed-stable redistribution of detection
> performance across the class-frequency distribution — +9.6% AP on pedestrian, the most
> frequent small class, at the cost of 6–20% on rare classes — leaving aggregate mAP
> unchanged. Roughly half the effect is attributable to an unintended 4.4× inflation of
> total box loss; the remainder to the weighting itself.

That is defensible, novel, and directly answers R3.1: the net-zero aggregate is the
average of a large positive and several large negatives, and reporting only the average
is what concealed the finding.

## What is still needed

| Work | Cost | Why |
|---|---|---|
| 3 seeds × {YOLOv5s, YOLOv5s+SingleSwin} | ~60 h | The only way to keep an architecture claim. Currently n=1 at 1.5× floor. |
| Restore `yolov5n_swin2.yaml` | minutes, if recoverable | Otherwise drop the row |
| test-dev eval | minutes, needs checkpoints | R1.8, R3.3 |
| Per-class `val.py --save-json` | minutes, needs checkpoints | R1.4, Table IV |
| `ref.bib` beside the `.tex` | minutes | R1.1, R5, R7.6 — fatal if missed |

## The honest assessment

The reframed paper is more useful than the submitted one and answers the reviewers with
evidence rather than argument. But it is a different paper: different title, different
abstract, different contribution. If INDISCON's revision stage does not permit that,
withdrawing and resubmitting the controlled study is the more defensible route — and
the study is stronger for having a negative result at its centre, provided the
architecture arm gets the seeds it needs before any positive claim is made.
