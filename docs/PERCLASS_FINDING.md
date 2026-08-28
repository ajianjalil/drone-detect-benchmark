# The per-class result: the loss works, but not for the reason claimed

2026-08-28. Per-class AP@0.5 from `val.py --verbose` on the C0/C1/C2 checkpoints
(3 seeds for C0 and C1, 2 for C2), VisDrone2019 val, evaluated identically.

---

## Small classes do not suffer — frequent classes benefit

| class | instances | mean box area (×10⁻⁴) | C0 AP | C1 AP | C1 rel |
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

Sorted by instance count, the split is exact: **the four most frequent classes all
improve; the six rarest all decline.**

Correlation of the per-class change with:

| predictor | r |
|---|---|
| log instance count | **+0.688** |
| log mean box area | −0.365 |

**Frequency predicts the effect. Size does not** — and the size correlation is weak and
in the wrong direction for the paper's stated mechanism. Two classes settle it on their
own: `car` has the *largest* area of any frequent class and improves; `bicycle` is
small (area 7.08, comparable to `motor` at 7.41) and loses 19.5%.

So the scale-aware weight `α(2 − w̃h̃)` does not function as a small-object prior. Small
boxes in VisDrone are overwhelmingly pedestrian, people and motor — which are also the
most common classes — so upweighting small boxes concentrates gradient on categories
that already dominate the loss. It is a **frequency reallocation wearing the costume of
a scale reallocation**, and the rare classes pay for it.

## The pedestrian result is real, and the paper undersells it 3×

Per-seed pedestrian AP@0.5:

| | seed 42 | seed 43 | seed 44 | spread |
|---|---|---|---|---|
| C0 | 0.324 | 0.324 | 0.328 | 0.004 |
| C1 | 0.359 | 0.356 | 0.355 | 0.004 |

The distributions are **disjoint** — every C1 seed beats every C0 seed, and the gap
(0.032) is eight times the within-group spread. This is the most solidly established
result in the entire study, far more so than any aggregate mAP number.

The paper claims **+3.1–3.2%** pedestrian AP. The measured effect is **+9.6%**.

## How much of it is the weighting, and how much is magnitude?

C2 (box gain matched to E4's 4.4× inflation, no weighting at all) also lifts pedestrian:

| | pedestrian AP | rel |
|---|---|---|
| C0 | 0.325 | — |
| C2 magnitude only | 0.341 | +4.8% |
| C1 full E4 loss | 0.357 | **+9.6%** |

Roughly **half** the pedestrian gain comes from box-loss magnitude alone; the other
half is attributable to the scale/resolution weighting itself. Unlike the aggregate
mAP result — where the weighting merely offsets the harm the magnitude causes — here
the weighting contributes a genuine, separable effect.

## Why aggregate mAP hides all of this

mAP averages over classes without weighting by frequency, so a +9.6% gain on one class
and −20% on three rare ones nets out near zero. The submitted paper reports the
aggregate as its headline and the per-class effect as a footnote. That is backwards:
the aggregate is the noisy, null result, and the per-class effect is the reproducible
one.

## What this means for the paper

The loss modification **is** a contribution — just not the one claimed. It is a
precision/recall reallocation across the class-frequency distribution, and it should be
presented and evaluated as such:

- **Claim:** consistent, seed-stable gains on frequent classes, led by pedestrian at
  +9.6%, paid for by 6–20% losses on rare classes, with aggregate mAP unchanged.
- **Do not claim:** an overall mAP improvement (there is none), or a small-object prior
  (the mechanism is frequency, not scale).
- **The interesting question this opens:** whether the same reallocation can be had
  without the rare-class cost — which is a real avenue for future work rather than a
  hedge.

This also reframes R3.1 ("the net-zero gain weakens the combined model claim"). The
net-zero is not a weakness to explain away; it is the average of a large positive and
several large negatives, and reporting only the average was what hid the finding.
