#!/usr/bin/env python3
"""Measure the per-layer (P3/P4/P5) box-loss share (SUMMARY.md F-G).

The paper states the custom loss moves the P3 share from ~50% to ~79%. Both figures
are wrong: the baseline split is near-uniform, so the real shift is ~33% -> ~68%.

Each layer's share is isolated by running the real ComputeLoss with a one-hot
resolution-beta, so the numbers come from the shipped code path rather than a
reimplementation.

Usage (from the repo root):
    python tools/measure_layer_split.py

Expected output (yolov5n at initialisation, seed 42):
    baseline   33.1 / 32.8 / 34.2      E4 loss   68.1 / 22.5 / 9.4
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

E4_BETA = [3.0, 1.0, 0.4]
E4_ALPHA = 1.5


def one_hot(i):
    b = [0.0, 0.0, 0.0]
    b[i] = 1.0
    return b


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
    model.train()

    loader, _ = create_dataloader(train_path, opt.imgsz, opt.batch_size, int(model.stride.max()),
                                  hyp=hyp, augment=True, rect=False, workers=4,
                                  shuffle=True, seed=opt.seed)

    crits = {}
    for i in range(3):
        crits[f"base_P{3 + i}"] = ComputeLoss(
            model, log_interval=0, use_scale_aware_loss=False, use_resolution_weighting=True,
            scale_alpha=1.0, resolution_beta=one_hot(i))
        crits[f"e4_P{3 + i}"] = ComputeLoss(
            model, log_interval=0, use_scale_aware_loss=True, use_resolution_weighting=True,
            scale_alpha=E4_ALPHA,
            resolution_beta=[m * b for m, b in zip(one_hot(i), E4_BETA)])

    totals = {k: 0.0 for k in crits}
    seen = 0
    with torch.no_grad():
        for i, (imgs, targets, _, _) in enumerate(loader):
            if i >= opt.batches:
                break
            imgs = imgs.to(opt.device, non_blocking=True).float() / 255
            targets = targets.to(opt.device)
            pred = model(imgs)
            for k, crit in crits.items():
                totals[k] += crit(pred, targets)[1][0].item()
            seen += 1

    base = [totals[f"base_P{3 + i}"] for i in range(3)]
    e4 = [totals[f"e4_P{3 + i}"] for i in range(3)]
    print(f"\nPer-layer box-loss share, {seen * opt.batch_size} augmented images, "
          f"{Path(opt.cfg).stem} at initialisation\n")
    print(f"{'':10s}{'P3':>10s}{'P4':>10s}{'P5':>10s}")
    print(f"{'baseline':10s}" + "".join(f"{v / sum(base) * 100:9.1f}%" for v in base))
    print(f"{'E4 loss':10s}" + "".join(f"{v / sum(e4) * 100:9.1f}%" for v in e4))
    print(f"\npaper claims : ~50% -> ~79% P3 share")
    print(f"measured     : {base[0] / sum(base) * 100:.1f}% -> {e4[0] / sum(e4) * 100:.1f}%")


if __name__ == "__main__":
    main()
