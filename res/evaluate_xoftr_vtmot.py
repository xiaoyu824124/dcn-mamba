"""Probe official pretrained XoFTR on the existing VTMOT evaluation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .vtmot import VTMOTSingleFrameDataset
from .xoftr_probe import build_xoftr, evaluate_xoftr


def main() -> None:
    parser = argparse.ArgumentParser(description="Official XoFTR pretrained probe on VTMOT")
    parser.add_argument("--xoftr-root", type=Path, required=True,
                        help="separately extracted official XoFTR repository")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="official weights_xoftr_640.ckpt")
    parser.add_argument("--data-root", default="data/VTMOT_misaligned")
    parser.add_argument("--split-file", default="data_split/IVF/VTMOT/split.json")
    parser.add_argument("--split", choices=("eval", "test"), default="eval")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--ransac-iterations", type=int, default=1000)
    parser.add_argument("--ransac-threshold-px", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.ransac_iterations < 1 or args.ransac_threshold_px <= 0:
        parser.error("RANSAC iterations and threshold must be positive")
    device = torch.device(args.device)
    dataset = VTMOTSingleFrameDataset(
        args.data_root, split=args.split, split_file=args.split_file,
        frame_stride=args.frame_stride, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    model = build_xoftr(args.xoftr_root, args.checkpoint, device)
    print(f"XoFTR probe split={args.split} samples={len(dataset)} "
          f"fov={dataset.target_hw} device={device}", flush=True)
    report = evaluate_xoftr(model, loader, device,
                            ransac_iterations=args.ransac_iterations,
                            ransac_threshold_px=args.ransac_threshold_px)
    report.update({"model": "official_XoFTR_pretrained", "checkpoint": str(args.checkpoint),
                   "split": args.split, "frame_stride": args.frame_stride,
                   "field_of_view_hw": list(dataset.target_hw),
                   "ransac_threshold_px": args.ransac_threshold_px})
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
