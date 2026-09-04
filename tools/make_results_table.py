#!/usr/bin/env python3
"""Generate the INDISCON revision plan §1.5 results table from a runs directory.

Emits P, R, mAP@0.5, mAP@0.5:0.95 (at the best-fitness epoch, matching how YOLOv5
selects best.pt), plus params / GFLOPs / FP16 size computed from each run's own
recorded cfg. FPS is read from a per-class val.py capture if one exists.

Usage:
    python tools/make_results_table.py runs/indiscon
    python tools/make_results_table.py runs/indiscon --order B_p2head C_swin_p2 A_baseline D_swin_p2_loss
"""
import argparse, contextlib, csv, io, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch, yaml
from torch.utils.flop_counter import FlopCounterMode
from models.yolo import Model


def best_row(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    k = {c.strip(): c for c in rows[0]}
    # YOLOv5 fitness = 0.1*mAP@0.5 + 0.9*mAP@0.5:0.95
    fit = lambda r: 0.1 * float(r[k["metrics/mAP_0.5"]]) + 0.9 * float(r[k["metrics/mAP_0.5:0.95"]])
    b = max(rows, key=fit)
    return {
        "P": float(b[k["metrics/precision"]]), "R": float(b[k["metrics/recall"]]),
        "m50": float(b[k["metrics/mAP_0.5"]]), "m5095": float(b[k["metrics/mAP_0.5:0.95"]]),
        "epoch": int(b[k["epoch"]].strip()), "n": len(rows),
    }


def complexity(cfg, nc, imgsz=640):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m = Model(cfg, ch=3, nc=nc); m.eval()
        p = sum(x.numel() for x in m.parameters())
        fc = FlopCounterMode(display=False)
        with fc, torch.no_grad():
            m(torch.zeros(1, 3, imgsz, imgsz))
        g = fc.get_total_flops() / 1e9
    return p, g, p * 2 / 1e6   # params, GFLOPs, FP16 MB (YOLOv5 saves half precision)


def fps_from(run_dir):
    """Read the Speed: line from a persisted val.py capture, if present."""
    for cand in (Path(run_dir).parent / "perclass_val" / f"{Path(run_dir).name}.txt",):
        if cand.exists():
            for ln in open(cand):
                mt = re.search(r"([\d.]+)ms inference", ln)
                if mt:
                    return 1000.0 / float(mt.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--order", nargs="*", default=None, help="run names, in table order")
    ap.add_argument("--nc", type=int, default=10)
    opt = ap.parse_args()

    dirs = [d for d in sorted(Path(opt.runs_dir).iterdir())
            if (d / "results.csv").exists() and (d / "opt.yaml").exists()]
    if opt.order:
        byname = {d.name: d for d in dirs}
        dirs = [byname[n] for n in opt.order if n in byname]

    print(f"\n| ID | Config | Ep | P | R | mAP@.5 | mAP@.5:.95 | Params | GFLOPs | Size (MB) | FPS |")
    print(f"|---|---|---|---|---|---|---|---|---|---|---|")
    for d in dirs:
        o = yaml.safe_load(open(d / "opt.yaml"))
        cfg = o.get("cfg") or ""
        r = best_row(d / "results.csv")
        if cfg and Path(cfg).exists():
            p, g, mb = complexity(cfg, opt.nc)
            pstr, gstr, mbstr = f"{p:,}", f"{g:.1f}", f"{mb:.1f}"
        else:
            pstr = gstr = mbstr = "?"   # resumed run: cfg not recorded
        f = fps_from(d)
        print(f"| {d.name} | `{cfg or '(resumed — cfg not recorded)'}` | {r['n']} | {r['P']:.3f} | {r['R']:.3f} | "
              f"{r['m50']:.4f} | {r['m5095']:.4f} | {pstr} | {gstr} | {mbstr} | "
              f"{f'{f:.0f}' if f else '—'} |")
    print("\nBest-fitness epoch used (0.1*mAP@0.5 + 0.9*mAP@0.5:0.95), matching best.pt selection.")
    print("FPS = 1000/inference-ms from runs/<dir>/perclass_val/<name>.txt; run val.py to populate.")


if __name__ == "__main__":
    main()
