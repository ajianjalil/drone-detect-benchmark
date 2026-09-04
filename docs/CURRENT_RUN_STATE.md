# Live run state — INDISCON A/B/C/D matrix

> **SNAPSHOT taken 2026-09-04 10:10 IDT.** Progress numbers and PIDs below go stale
> immediately. **Re-check with the commands in §5 before trusting anything here.**
> The queue definition, artifact paths and recovery procedures do not go stale.

Companion to [HANDOFF.md](HANDOFF.md) (project state) and [SUMMARY.md](SUMMARY.md)
(findings). This file is only about the jobs executing right now.

---

## 1. What is executing

A shell driver running four trainings **sequentially** — never in parallel, the GPU
has 16 GB and arm C alone peaks at 10.4 GB.

| | |
|---|---|
| driver script | `run_indiscon_abcd.sh` (committed) |
| driver PID | **498468** (`bash run_indiscon_abcd.sh`), started 06:55:14 Sep 4 |
| driver log | `logs/indiscon_driver.log` — one line per arm start/finish |
| per-run logs | `logs/indiscon_<NAME>.log` |
| output root | `runs/indiscon/<NAME>/` |
| launched with | `nohup ... &` — survives terminal close, **not** a reboot |

The driver has `set -e`: if a training exits non-zero the whole queue stops. It also
skips any arm whose output directory already exists, so it is safe to re-run after a
partial failure — it resumes at the next unfinished arm (but does **not** resume a
half-finished arm; see §6).

### Currently in flight

```
python train.py --img 640 --batch 16 --epochs 100 --save-period 10 \
  --data data/VisDrone_local.yaml --cfg models/yolov5s_p2.yaml \
  --hyp data/hyps/hyp.scratch-low.yaml --seed 42 --device 0 --workers 8 \
  --loss-log-interval 200 --name B_p2head --project runs/indiscon
```

PID **498472** (child of the driver) plus **24 dataloader workers**.

---

## 2. The queue

Order is deliberate: **B and C first**, because the revision plan's §1.6 decision rule
turns on **C − B**. A and D are not needed for that call.

| # | NAME | cfg | loss flags | status @snapshot |
|---|---|---|---|---|
| 1 | `B_p2head` | `models/yolov5s_p2.yaml` | none (CIoU) | **running, ep 70/100** |
| 2 | `C_swin_p2` | `models/yolov5s_p2_swinP2.yaml` | none (CIoU) | queued |
| 3 | `A_baseline` | `models/yolov5s.yaml` | none (CIoU) | queued |
| 4 | `D_swin_p2_loss` | `models/yolov5s_p2_swinP2.yaml` | `--scale-aware-loss --resolution-weighting --scale-alpha 1.5 --resolution-beta 3.0 2.0 1.0 0.4` | queued |

All four: **100 epochs, batch 16, seed 42, img 640, `--save-period 10`**, same hyp file
(`hyp.scratch-low.yaml`, `box: 0.05`). One epoch budget across all arms is the point —
mixing budgets is the flaw the revision plan exists to fix.

**B and C differ by exactly one line**: a `SwinStage` on the P2 neck branch.
**D and C differ by exactly the loss flags.** So each contrast is single-variable.

### Estimated completion (from 06:55 Sep 4)

| arm | est duration | est finish |
|---|---|---|
| B | ~4.6 h | ~11:30 Sep 4 |
| C | ~8.4 h | ~20:00 Sep 4 |
| A | ~3.2 h | ~23:10 Sep 4 |
| D | ~8.4 h | **~07:30 Sep 5** |

C and D are extrapolations from `yolov5s_p2_swinP3` (4.21 h / 50 ep), which has near
identical GFLOPs but runs attention over 6,400 tokens vs Swin@P2's 25,600. **Revise
once C has run an hour.** B's estimate was confirmed against its observed 2.68 it/s.

---

## 3. Progress at snapshot

`B_p2head`, epoch 70/100, best **mAP@0.5 = 0.3668**, mAP@0.5:0.95 = 0.1988 (ep 68).

| epoch | 10 | 25 | 50 | 70 |
|---|---|---|---|---|
| mAP@0.5 | 0.2377 | 0.3197 | 0.3594 | 0.3642 |

Reference points it will be compared against:

| | mAP@0.5 |
|---|---|
| yolov5s baseline, 300 ep, batch 32, n=3 | 0.3407 |
| yolov5s baseline, 50 ep, batch 16, n=1 (`S0_base`) | 0.3115 |
| + P2 head, 50 ep, batch 16, n=1 (`S1_p2head`) | 0.3479 |
| + P2 head, 168 ep, batch 4, n=1 (`S1_p2head_300`, stopped) | 0.3935 |

**Arm A is the only in-matrix baseline.** Do not compare B/C/D against the 300-epoch
or batch-32 numbers above — different budget and batch. That is the whole point of A.

---

## 4. Resources at snapshot

| | |
|---|---|
| GPU | 8,700 / 16,380 MiB, 100% util, 83 °C |
| disk | 58 G free (87% used); `runs/` is 8.4 G |

**Disk is the one thing that could bite.** `--save-period 10` writes 10 extra
checkpoints per arm (~14 MB each for B/A, ~27 MB for C/D) — roughly 0.7 G more across
the remaining three arms. Fine now, but if free space drops under ~10 G, delete
`runs/*/weights/epoch*.pt` from **completed, already-analysed** runs only.

---

## 5. Monitoring — copy-paste

```bash
cd /home/avcom/Documents/ajith/drone-detect-benchmark

# which arms have started/finished
cat logs/indiscon_driver.log

# is anything actually running, and what
ps -eo pid,etime,args | grep "[t]rain\.py" | cut -c1-150

# current epoch of the in-flight arm (swap the log name)
tr '\r' '\n' < logs/indiscon_B_p2head.log | grep -E "^ *[0-9]+/99" | tail -1

# best-so-far for every arm that has results
for d in runs/indiscon/*/; do
  python - "$d" <<'PY'
import csv,sys,os
f=os.path.join(sys.argv[1],'results.csv')
if os.path.exists(f):
    r=list(csv.DictReader(open(f))); k={c.strip():c for c in r[0]}
    b=max(float(x[k['metrics/mAP_0.5']]) for x in r)
    print(f"{os.path.basename(os.path.dirname(f)):18s}{b:.4f}  ({len(r)} ep)")
PY
done

# GPU + disk
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
df -h /home/avcom | tail -1
```

Failure signatures worth grepping for: `Traceback`, `CUDA out of memory`, `Killed`.

---

## 6. If something goes wrong

**A single arm crashed / was killed.** The driver's `set -e` stops the queue. Each arm
writes `last.pt` every epoch, so resume that arm, then re-run the driver — it skips
completed directories:
```bash
python train.py --resume runs/indiscon/<NAME>/weights/last.pt
bash run_indiscon_abcd.sh          # picks up the remaining arms
```
Note: a resumed run is **not** bit-identical to an uninterrupted one (the dataloader
reseeds). Record it if it happens — this already applies to `C1_e4_loss` in
`runs/control/`.

**Machine rebooted.** Nothing auto-restarts. Check `runs/indiscon/*/results.csv` row
counts to see how far each arm got, then resume as above. Checkpoints survive reboots;
`/tmp` does not.

**Need to stop everything.**
```bash
kill -TERM $(pgrep -f run_indiscon_abcd.sh)   # kill driver FIRST or it starts the next arm
kill -TERM $(pgrep -f "train.py --img 640")
```
Kill the driver first. VRAM is only released when the process exits — `SIGSTOP` pauses
compute but holds all ~9 GB.

**Need the GPU temporarily.** Same as above; every arm is resumable from `last.pt`.

---

## 7. What to do when the queue finishes

1. **Fill the plan's §1.5 table** — one command:
   ```bash
   python tools/make_results_table.py runs/indiscon \
     --order B_p2head C_swin_p2 A_baseline D_swin_p2_loss
   ```
   Emits markdown with P, R, mAP@0.5, mAP@0.5:0.95 at the **best-fitness epoch**
   (0.1·mAP@0.5 + 0.9·mAP@0.5:0.95, matching how YOLOv5 picks `best.pt`), plus
   params / GFLOPs / FP16 size rebuilt from each run's own recorded `cfg`. The FPS
   column populates once step 2 has written the per-class captures.
2. **Per-class AP@0.5 for B and C** (the plan asks for this specifically):
   ```bash
   python val.py --img 640 --batch 8 --data data/VisDrone_local.yaml \
     --weights runs/indiscon/B_p2head/weights/best.pt --task val --verbose
   ```
   Persist into `runs/indiscon/perclass_val/` — **not** `/tmp`, which is wiped on reboot.
3. **Apply the §1.6 decision rule** to C − B:
   - C − B ≳ 1 pp → attention-placement finding, Swin@P2 is contribution #1
   - C ≈ B (< ~0.5 pp) → "resolution, not attention, drives the gain"
   - C < B → attention at stride 4 disrupts features; report it
   All three are publishable in a benchmarking paper.
4. **Interpret against the noise floor**: 1.4% at 300 ep but **2.8% at 50 ep**, and
   these are n=1 at 100 ep. A 1 pp difference on a ~0.35 baseline is ~2.9% relative —
   right at the floor. **Do not call a sub-1-pp C−B difference a finding without seeds.**
5. **Commit the evidence** (`runs/` is gitignored, so force-add):
   ```bash
   for d in runs/indiscon/*/; do git add -f $d/results.csv $d/opt.yaml $d/hyp.yaml; done
   git add -f runs/indiscon/perclass_val/
   ```
6. **Update [SUMMARY.md](SUMMARY.md)** — add the finding and keep its §6
   finding→artifact table truthful.

---

## 8. Known caveats attached to these specific runs

- **n=1 per arm.** The plan did not ask for seeds. Given the measured noise floor,
  differences under ~1 pp are not interpretable. Seeds are the obvious follow-up.
- **B and C change anchors as well as architecture** — AutoAnchor evolved **12**
  anchors (confirmed: `Running kmeans for 12 anchors` in the B log) versus 9 for A.
  So A→B mixes "extra head" with "better anchors". Isolating that needs one ~3 h run of
  `yolov5s.yaml` with `anchors: 3` and no P2 head.
- **Arm C's SwinStage placement**: the *neck* P2 branch, after the P2 fusion C3 and
  before the P2 Detect head. It does **not** touch the backbone P2 map. Stated in the
  config header, and the paper must state it too (the plan's pre-flight asks).
- **AutoAnchor reports 29,644 of 343,201 labels are < 3 px.** ~8.6% of the dataset is
  invisible even at stride 4 — a natural ceiling argument for the paper.
