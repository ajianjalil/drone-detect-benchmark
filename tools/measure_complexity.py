#!/usr/bin/env python3
"""Params / GFLOPs / size for every model config, three ways (SUMMARY.md F-F).

YOLOv5 prints GFLOPs by profiling at 32x32 and scaling by (imgsz/32)^2. That
extrapolation assumes every operation scales with H*W, which windowed attention does
not, so the printed figure overstates every Swin variant by ~2.2x.

  yolov5_builtin : what train.py prints
  thop_at_640    : thop profiled directly at full resolution
  aten_at_640    : torch.utils.flop_counter, real ATEN ops incl. attention matmuls

The three agree within 1% on every pure-CNN model, which is what validates them.
Quote aten_at_640.

Needs no dataset and no checkpoints — runs on CPU in a few seconds.

Usage (from the repo root):
    python tools/measure_complexity.py
"""
import argparse
import contextlib
import copy
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import thop
import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from models.yolo import Model

CFGS = [
    ("YOLOv5n", "models/yolov5n.yaml"),
    ("YOLOv5s", "models/yolov5s.yaml"),
    ("YOLOv5m", "models/yolov5m.yaml"),
    ("YOLOv5n + SingleSwin", "models/yolov5_swin.yaml"),
    ("YOLOv5s + DoubleSwin", "models/yolov5s_swin2.yaml"),
    ("YOLOv5m + DoubleSwin", "models/yolov5m_swin.yaml"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", default="data/hyps/hyp.scratch-low.yaml")
    ap.add_argument("--nc", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    opt = ap.parse_args()

    hyp = yaml.safe_load(open(opt.hyp))
    print(f"\n{'Model':22s}{'Params':>12s}{'FP16 MB':>9s}"
          f"{'builtin':>10s}{'thop@' + str(opt.imgsz):>10s}{'aten@' + str(opt.imgsz):>11s}")
    for name, cfg in CFGS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model = Model(cfg, ch=3, nc=opt.nc, anchors=hyp.get("anchors"))
            model.eval()
            n_p = sum(x.numel() for x in model.parameters())

            stride = max(int(model.stride.max()), 32)
            small = thop.profile(copy.deepcopy(model),
                                 inputs=(torch.zeros(1, 3, stride, stride),), verbose=False)[0] / 1e9 * 2
            builtin = small * opt.imgsz / stride * opt.imgsz / stride

            full = torch.zeros(1, 3, opt.imgsz, opt.imgsz)
            thop640 = thop.profile(copy.deepcopy(model), inputs=(full,), verbose=False)[0] / 1e9 * 2

            counter = FlopCounterMode(display=False)
            with counter, torch.no_grad():
                model(full)
            aten = counter.get_total_flops() / 1e9

        # YOLOv5 checkpoints are saved half-precision, which is what Table I's MB column reports
        print(f"{name:22s}{n_p:12,d}{n_p * 2 / 1e6:9.1f}{builtin:10.1f}{thop640:10.1f}{aten:11.1f}")


if __name__ == "__main__":
    main()
