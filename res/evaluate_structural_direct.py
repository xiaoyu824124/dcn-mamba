"""Evaluate fixed structural-prior features with no trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader
import torch
from omegaconf import OmegaConf

from .evaluate_vtmot import _registration_status, evaluate
from .model_factory import build_global_registration
from .vtmot import VTMOTSingleFrameDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/VTMOT_misaligned")
    parser.add_argument("--split-file", default="data_split/IVF/VTMOT/split.json")
    parser.add_argument("--split", choices=("train", "eval", "test"), default="eval")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overlay", type=Path, action="append", default=[],
                        help="parameter-free matcher overlay, for a same-prior comparison")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    config = OmegaConf.merge(
        OmegaConf.load("res/configs/registration.yaml"),
        OmegaConf.load("res/configs/stage0_structural_prior.yaml"),
        OmegaConf.load("res/configs/ab_structural_direct.yaml"),
        *(OmegaConf.load(path) for path in args.overlay))
    model = build_global_registration(config).to(device).eval()
    model.matcher.return_correlation = True
    dataset = VTMOTSingleFrameDataset(
        args.data_root, split=args.split, split_file=args.split_file,
        frame_stride=args.frame_stride, max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"direct structural-prior split={args.split} samples={len(dataset)} "
          f"stride={args.frame_stride} device={device}")
    report = evaluate(model, loader, device)
    report["split"] = args.split
    report["frame_stride"] = args.frame_stride
    report["field_of_view_hw"] = [480, 640]
    if args.overlay:
        report["evaluation_overlays"] = [str(path) for path in args.overlay]
    beats_zero, fusion_ready, status = _registration_status(report, "ir")
    report["beats_zero_flow"] = beats_zero
    report["fusion_ready"] = fusion_ready
    print(json.dumps(report, indent=2))
    print(status)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
