#!/usr/bin/env python3
"""Measure the total-box-loss inflation caused by the custom loss (SUMMARY.md F-C).

docs/REVIEW_RESPONSE.md F6 estimated x2.80 (E3) and x5.64 (E4) from an ASSUMED
P3/P4/P5 box-loss split of 50/30/20. This measures the factor on real VisDrone
batches through the actual ComputeLoss path, which is what the C2 control run is
calibrated against (box: 0.05 -> 0.2182 = 0.05 x 4.364).

Usage (from the repo root):
    python tools/measure_box_inflation.py [--batches 40] [--batch-size 16]

Expected output (yolov5n at initialisation, seed 42):
    E3_both     x2.317      E4_strong   x4.364
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from models.yolo import Model
from utils.dataloaders import create_dataloader
from utils.loss import ComputeLoss

CONFIGS = {
    "E0_baseline": dict(use_scale_aware_loss=False, use_resolution_weighting=False,
                        scale_alpha=1.0, resolution_beta=[1, 1, 1]),
    "E1_scale":    dict(use_scale_aware_loss=True,  use_resolution_weighting=False,
                        scale_alpha=1.0, resolution_beta=[1, 1, 1]),
    "E2_res":      dict(use_scale_aware_loss=False, use_resolution_weighting=True,
                        scale_alpha=1.0, resolution_beta=[2.0, 1.0, 0.5]),
    "E3_both":     dict(use_scale_aware_loss=True,  use_resolution_weighting=True,
                        scale_alpha=1.0, resolution_beta=[2.0, 1.0, 0.5]),
    "E4_strong":   dict(use_scale_aware_loss=True,  use_resolution_weighting=True,
                        scale_alpha=1.5, resolution_beta=[3.0, 1.0, 0.4]),
}
# the two factors F6 estimated, for side-by-side comparison
F6_ESTIMATE = {"E0_baseline": 1.00, "E3_both": 2.80, "E4_strong": 5.64}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/VisDrone_local.yaml")
    ap.add_argument("--cfg", default="models/yolov5n.yaml")
    ap.add_argument("--hyp", default="data/hyps/hyp.scratch-low.yaml")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    opt = ap.parse_args()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    hyp = yaml.safe_load(open(opt.hyp))
    data = yaml.safe_load(open(opt.data))
    train_path = str(Path(data["path"]) / data["train"])

    model = Model(opt.cfg, ch=3, nc=len(data["names"]), anchors=hyp.get("anchors")).to(opt.device)
    model.hyp, model.nc = hyp, len(data["names"])
    # Detect() only returns the raw per-layer outputs ComputeLoss expects in train mode
    model.train()

    loader, _ = create_dataloader(train_path, opt.imgsz, opt.batch_size, int(model.stride.max()),
                                  hyp=hyp, augment=True, rect=False, workers=4,
                                  shuffle=True, seed=opt.seed)

    crits = {k: ComputeLoss(model, log_interval=0, **v) for k, v in CONFIGS.items()}
    totals = {k: 0.0 for k in CONFIGS}
    seen = 0
    with torch.no_grad():
        for i, (imgs, targets, _, _) in enumerate(loader):
            if i >= opt.batches:
                break
            imgs = imgs.to(opt.device, non_blocking=True).float() / 255
            targets = targets.to(opt.device)
            pred = model(imgs)
            for k, crit in crits.items():
                totals[k] += crit(pred, targets)[1][0].item()  # lbox component
            seen += 1

    base = totals["E0_baseline"]
    print(f"\n{seen * opt.batch_size} augmented images, {Path(opt.cfg).stem} at initialisation\n")
    print(f"{'config':14s}{'mean lbox':>12s}{'x baseline':>12s}{'F6 estimate':>13s}")
    for k in CONFIGS:
        est = F6_ESTIMATE.get(k)
        print(f"{k:14s}{totals[k] / seen:12.5f}{totals[k] / base:12.3f}"
              f"{(f'{est:.2f}' if est else '—'):>13s}")
    print(f"\nmagnitude-matched box gain (baseline box={hyp['box']}):")
    for k in ("E3_both", "E4_strong"):
        print(f"  {k:12s} -> box: {hyp['box'] * totals[k] / base:.4f}")


if __name__ == "__main__":
    main()
