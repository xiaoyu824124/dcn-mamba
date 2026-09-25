"""Evaluate the official CRFT model on the existing VTMOT split and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .crft_baseline import (DEFAULT_MODEL_HW, build_crft, evaluate_crft,
                            load_crft_weights, validate_model_hw)
from .vtmot import VTMOTSingleFrameDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Official CRFT baseline on VTMOT")
    parser.add_argument("--crft-root", type=Path, required=True,
                        help="separately extracted official CRFT repository")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="official RoadScene .ckpt or a train_crft_vtmot .pt")
    parser.add_argument("--data-root", default="data/VTMOT_misaligned")
    parser.add_argument("--split-file", default="data_split/IVF/VTMOT/split.json")
    parser.add_argument("--split", choices=("eval", "test"), default="eval")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--model-hw", type=int, nargs=2, default=None,
                        metavar=("HEIGHT", "WIDTH"),
                        help="defaults to checkpoint model size or 96x96 for official weights")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    dataset = VTMOTSingleFrameDataset(
        args.data_root, split=args.split, split_file=args.split_file,
        frame_stride=args.frame_stride, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    model = build_crft(args.crft_root, device)
    saved_hw = load_crft_weights(model, args.checkpoint)
    if args.model_hw is not None and saved_hw is not None and tuple(args.model_hw) != saved_hw:
        raise ValueError("--model-hw differs from the training checkpoint")
    model_hw = validate_model_hw(
        tuple(args.model_hw) if args.model_hw is not None else saved_hw or DEFAULT_MODEL_HW)
    report = evaluate_crft(model, loader, device, model_hw)
    report.update({"model": "official_CRFT", "checkpoint": str(args.checkpoint),
                   "split": args.split, "frame_stride": args.frame_stride,
                   "field_of_view_hw": [480, 640]})
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
