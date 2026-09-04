# Handoff — drone-detect-benchmark / INDISCON paper 2887

**Written 2026-09-04.** Everything here is verifiable from this repo. Read
[SUMMARY.md](SUMMARY.md) first — it is the consolidated findings reference.
This file covers state, environment and traps that SUMMARY.md does not.

---

## 1. What this is

YOLOv5 fork benchmarked on VisDrone2019-DET. A paper (INDISCON 2026, #2887) was
submitted claiming two contributions; peer review found serious problems
([REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) transcribes all five reviews). Since
2026-08-25 the work has been re-measuring every claim under controlled conditions.

**Both original contributions turned out to be absent.** What replaced them is a
methodological result plus one large new positive finding (a P2 detection head).

---

## 2. Environment — read before running anything

| | |
|---|---|
| conda env | `yolov5` at `/home/avcom/miniconda3/envs/yolov5` |
| python | `/home/avcom/miniconda3/envs/yolov5/bin/python` |
| torch | 2.5.1+cu124 |
| GPU | NVIDIA RTX 2000 Ada, **16 GB** (not the 5090 some plans assume) |
| dataset yaml | `data/VisDrone_local.yaml` |
| dataset root | `/home/avcom/Documents/ajith/visdrone_yolov5` (symlink tree) |

**Traps, each of which has already cost time:**

1. **Do not use the `dygrid` env.** It has setuptools 82, which removed
   `pkg_resources`; `utils/general.py:32` imports it, so `train.py` will not even
   import. The `yolov5` env is a clone of `dygrid` with `setuptools<81` pinned.
   Never "fix" `dygrid` — another project (`phd_research/dygrid_edgeyolo`) uses it.
2. **The dataset root is a symlink tree on purpose.** The real VisDrone lives at
   `phd_research/dygrid_edgeyolo/data/VisDrone2019/`. YOLOv5 writes
   `<split>/labels.cache`, which would overwrite the ultralytics caches belonging to
   that other project. The symlink tree keeps the caches separate. Verified by md5
   before/after. **Do not point `path:` at the phd_research copy directly.**
3. **`/tmp` is wiped on reboot.** Per-class `val.py` outputs were lost that way once.
   Write anything that matters into the repo, not the scratchpad.
4. **`.pyc` files were tracked in git** until commit `ec4a742f`. If `git status`
   shows bytecode churn again, something re-added them.
5. **Batch-size ceilings on this GPU** (measured, peak allocated at 640px):
   yolov5s b32 = 5.4 GB · +P2 head b32 = 7.6 GB · +SwinP3 b32 = 11.9 GB ·
   +P2+Swin b32 = **14.1 GB (OOMs in practice)** · +P2+Swin b16 = 7.1–10.4 GB.
   **Use batch 16 for anything with attention + a P2 head.**

---

## 3. Findings established so far

Full detail with tables in [SUMMARY.md](SUMMARY.md) §2. Condensed:

| id | finding |
|---|---|
| **F-A** | **Noise floor 1.4%** (3 seeds, identical config). Confirmed twice on different models/batch sizes. **Epoch-dependent: 4.3% at ep25, 2.8% at ep50, 1.4% at ep300.** |
| **F-B** | The paper's +4.8% loss gain does not reproduce. n=3 at 300 ep: −2.3% mAP@0.5. |
| **F-C** | The custom loss silently inflates total box loss **×4.36** (measured, not estimated). No renormalisation in `utils/loss.py`. |
| **F-D** | The loss reallocates by **class frequency** (r=+0.688 with log instance count), **not object scale** (r=−0.365). |
| **F-E** | **Pedestrian AP +9.6%**, disjoint distributions across 3 seeds — the most solid result in the study. Paper claims +3.1–3.2%. |
| **F-F** | YOLOv5's printed GFLOPs overstates every Swin variant ~2.2× (32×32 extrapolation fails for windowed attention). |
| **F-G** | Table I is **300 epochs**, not the 100 the caption claims. Also: P3 loss share is 33.1%→68.1%, not 50%→79%; `SwinStage` contains **no PatchMerging**. |
| **F-H** | The SwinStage gain does not reproduce: **−0.96% ± 1.58** at n=3/300ep. Straddles zero, inside the noise floor. |

**Why the original +4.8% appeared** (this is the answer to the obvious question):
at 50 epochs the noise floor is 2.8%, so a single-run A/B carries σ ≈ 2.4%. +4.8% is
≈2σ — **and it was selected as the best of four configurations** tested against one
baseline. Noise plus selection is sufficient; no error is required.

### The new positive result

**A P2 detection head (stride 4) is worth ~+12–15% mAP@0.5 for +145K params (+2%).**

Rationale: at stride 32 a VisDrone pedestrian occupies **0.17 × 0.53 feature cells** —
sub-pixel. The paper's SwinStage sits there, so it cannot see small objects at all.
At stride 4 the same pedestrian is 1.37 × 4.27 cells.

| config | 50 ep (b16, n=1) | vs base |
|---|---|---|
| S0 yolov5s | 0.3115 | — |
| **S1 + P2 head** | **0.3479** | **+11.7%** |
| S2 + SwinP3 (attention at stride 8) | 0.3021 | −3.0% |
| S3 + P2 head + SwinP3 | 0.3435 | +10.3% |

A partial 300-epoch run of S1 reached **0.3935 by epoch 168** vs the n=3 yolov5s
baseline of 0.3407 — **+15.5%**. It was stopped by user request; resume with
`--resume runs/p2screen/S1_p2head_300/weights/last.pt` (17 periodic checkpoints exist).

**Attention has not helped at any placement tested**: P5 (−0.96%), P3 (−3.0%), or on
top of a P2 head (−0.7%). Resolution is the lever; attention is not.

---

## 4. What is running right now

> **See [CURRENT_RUN_STATE.md](CURRENT_RUN_STATE.md) for the live detail** — exact
> commands, PIDs, monitoring one-liners, recovery procedures if a run dies or the
> machine reboots, and what to do when the queue finishes. That file is a dated
> snapshot; re-check progress before trusting its numbers.

`run_indiscon_abcd.sh` → `runs/indiscon/`, driver log `logs/indiscon_driver.log`.
Executing the user's `INDISCON_revision_plan.md` §1.2 matrix. 100 epochs, batch 16,
seed 42, `--save-period 10`, all four arms on one epoch budget.

| ID | model cfg | loss | params | GFLOPs | ETA (from 06:55 Sep 4) |
|---|---|---|---|---|---|
| B | `models/yolov5s_p2.yaml` | CIoU | 7,192,244 | 18.7 | ~11:30 Sep 4 |
| C | `models/yolov5s_p2_swinP2.yaml` | CIoU | 7,395,780 | 30.4 | ~20:00 Sep 4 |
| A | `models/yolov5s.yaml` | CIoU | 7,046,599 | 15.8 | ~23:10 Sep 4 |
| D | `models/yolov5s_p2_swinP2.yaml` | α=1.5, β=[3,2,1,0.4] | 7,395,780 | 30.4 | ~07:30 Sep 5 |

Check progress:
```bash
cat logs/indiscon_driver.log
tr '\r' '\n' < logs/indiscon_B_p2head.log | grep -E "^ *[0-9]+/99" | tail -1
```

**The decision rule (plan §1.6) turns on C − B**, both of which finish before A and D.

**Deviations from the plan, and why:**
- **A is run, not reused.** The plan says 100 epochs matches Table I. It does not —
  Table I is 300 epochs (F-G). Reusing it would reintroduce the mixed-budget flaw.
- **The plan lists C as "in progress." It was not.** The run that was in progress and
  then stopped was B (`yolov5s_p2.yaml`, `scale_aware_loss: false`).
- **One β variant, not three.** β-mid `[3.0, 2.0, 1.0, 0.4]`. Add mild/strong only if
  D shows something; each is ~8.4 h.

---

## 5. When the runs finish

1. Fill plan §1.5 table. Per-class AP for B and C:
   ```bash
   python val.py --img 640 --batch 8 --data data/VisDrone_local.yaml \
     --weights runs/indiscon/B_p2head/weights/best.pt --task val --verbose
   ```
   Persist output into `runs/indiscon/perclass_val/` — **not** the scratchpad.
2. Apply plan §1.6 decision rule to C − B.
3. FPS: take it from `val.py`'s `Speed:` line, and state the hardware.
4. Commit run evidence with `git add -f` (`runs/` is gitignored):
   `results.csv`, `opt.yaml`, `hyp.yaml` per run, plus per-class outputs.
5. Update [SUMMARY.md](SUMMARY.md) — add a finding, and keep the
   finding→artifact table in §6 truthful.

### Still outstanding, independent of these runs

| task | cost | why |
|---|---|---|
| **`ref.bib` beside the `.tex`** | minutes | Three reviewers flagged `[?]`. **Fatal if missed.** |
| test-dev evaluation | minutes | 1610 labelled images already on disk (`--task test`) |
| Isolate anchors from architecture | ~3 h | B changes anchors (12 vs 9) *and* adds a head. Run yolov5s with `anchors: 3` and no P2 head. |
| 3 seeds × {A, B} at 300 ep | ~24 h | the P2 result is currently n=1 |
| Recover `models/yolov5n_swin2.yaml` | — | referenced by `runs/train/yolov5_n_swin2/opt.yaml`, never in git history; that Table I row is unreproducible |

---

## 6. Unresolved fork you must not paper over

`models/yolov5s_swin.yaml` is **RECONSTRUCTED, not recovered** (see its header). The
paper's YOLOv5s+SingleSwin run was a `resume: true` from a container checkpoint at
`/app/yolov5` that no longer exists, and no s-scale single-Swin config has ever been in
this repo's git history. The reconstruction matches Table I's size column (26.7 vs
"26 MB") but a parameter count cannot uniquely identify an architecture.

So **F-H reads either as "the Swin gain does not reproduce" or "the reconstruction is
the wrong architecture."** The runs cannot separate these. The A0 arm reproduces the
reference baseline exactly (0.3407 vs 0.3415), which isolates the discrepancy to the
Swin arm. Only a surviving `/app/yolov5` log with a model summary line would close it.

---

## 7. Document map

| file | contents |
|---|---|
| [SUMMARY.md](SUMMARY.md) | **start here** — all findings, claim language, exposures, repro commands |
| [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) | the five reviews verbatim + original repo audit (F1–F6) |
| [STEP1_MEASUREMENTS.md](STEP1_MEASUREMENTS.md) | the box-gain control experiment in detail |
| [PERCLASS_FINDING.md](PERCLASS_FINDING.md) | frequency-vs-scale analysis |
| [PAPER_NARRATIVE.md](PAPER_NARRATIVE.md) | how to reframe the paper (has a dated revision note) |
| [CURRENT_RUN_STATE.md](CURRENT_RUN_STATE.md) | live state of the executing A/B/C/D queue — monitoring, recovery, next steps |
| `tools/measure_*.py` | the three checkpoint-free measurements (F-C, F-F, F-G) |
| `runs/*/perclass_val/` | committed `val.py --verbose` output; checkpoints are gitignored |

**Working style that has served this project well:** measure rather than assume;
state exposures in the record before a reviewer finds them; never report n=1 as a
finding — two seeds said the Swin effect was −1.86%, three said −0.96% ± 1.58.
