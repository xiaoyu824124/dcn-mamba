"""Evaluate a standalone registration checkpoint on the VTMOT held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .metrics import endpoint_error
from .train_registration import build_model
from .vtmot import VTMOTSingleFrameDataset
from .warp import warp


def _threshold_hits(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    error = torch.linalg.vector_norm(predicted - target, dim=1, keepdim=True)
    denominator = valid.sum().clamp_min(1)
    return {f"pck_{threshold}px": float(((error <= threshold) * valid).sum() / denominator)
            for threshold in (1, 3, 5)}


@torch.no_grad()
def check_gt_direction(loader: DataLoader, device: torch.device, preview_dir: Path | None = None) -> Dict[str, float]:
    """Prove stored flow maps aligned visible RGB to misaligned visible RGB."""
    aligned_error = unwarped_error = 0.0
    valid_pixels = 0.0
    for batch in loader:
        rgb_gt, visible = batch["rgb_gt"].to(device), batch["vi"].to(device)
        flow, valid = batch["gt_flow"].to(device), batch["valid_mask"].to(device)
        aligned = warp(rgb_gt, flow)
        weights = valid.expand_as(aligned)
        aligned_error += float(((aligned - visible).abs() * weights).sum())
        unwarped_error += float(((rgb_gt - visible).abs() * weights).sum())
        valid_pixels += float(weights.sum())
    report = {"gt_warp_mae": aligned_error / max(valid_pixels, 1.0),
              "unwarped_mae": unwarped_error / max(valid_pixels, 1.0)}
    report["improvement_ratio"] = report["gt_warp_mae"] / max(report["unwarped_mae"], 1e-8)
    return report


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    total_epe = total_baseline = total_valid = 0.0
    hit_sums = {f"pck_{threshold}px": 0.0 for threshold in (1, 3, 5)}
    model.eval()
    for batch in loader:
        ir, vi = batch["ir"].to(device), batch["vi"].to(device)
        target, valid = batch["gt_flow"].to(device), batch["valid_mask"].to(device)
        output = model(ir, vi)
        if isinstance(output, tuple):
            coarse, refined = output
            predicted = refined.final_flow if refined.final_flow is not None else coarse.coarse_flow
        else:
            predicted = output.coarse_flow
        epe = endpoint_error(predicted, target, valid)
        baseline = endpoint_error(torch.zeros_like(target), target, valid)
        count = float(valid.sum())
        total_epe += float(epe) * count
        total_baseline += float(baseline) * count
        total_valid += count
        hits = _threshold_hits(predicted, target, valid)
        for name, value in hits.items():
            hit_sums[name] += value * count
    report = {"epe_px": total_epe / max(total_valid, 1.0),
              "zero_flow_epe_px": total_baseline / max(total_valid, 1.0),
              "valid_pixels": total_valid}
    report["relative_epe"] = report["epe_px"] / max(report["zero_flow_epe_px"], 1e-8)
    report.update({name: value / max(total_valid, 1.0) for name, value in hit_sums.items()})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone VTMOT single-frame registration evaluation")
    parser.add_argument("--data-root", default="data/VTMOT_misaligned")
    parser.add_argument("--split-file", default="data_split/IVF/VTMOT/split.json")
    parser.add_argument("--split", choices=("train", "eval", "test"), default="eval",
                        help="use eval while tuning; reserve test for one final report")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--crop-hw", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=None,
                        help="optional centre crop after 480x640 normalisation; must match training FOV")
    parser.add_argument("--check-gt", action="store_true",
                        help="check gt_h direction using visible_gt; no model required")
    parser.add_argument("--use-dcn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = VTMOTSingleFrameDataset(args.data_root, split=args.split, split_file=args.split_file,
                                      frame_stride=args.frame_stride, include_rgb_gt=args.check_gt,
                                      crop_hw=tuple(args.crop_hw) if args.crop_hw is not None else None,
                                      max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    fov = tuple(args.crop_hw) if args.crop_hw is not None else (480, 640)
    print(f"split={args.split} samples={len(dataset)} stride={args.frame_stride} fov={fov} device={device}")
    if args.check_gt:
        report = check_gt_direction(loader, device)
        print(json.dumps(report, indent=2))
        print("PASS" if report["gt_warp_mae"] < report["unwarped_mae"] else "FAIL: GT direction is inconsistent")
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --check-gt is used")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        config = OmegaConf.create(checkpoint["config"])
        model = build_model(config, use_dcn=args.use_dcn).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        report = evaluate(model, loader, device)
        report["field_of_view_hw"] = list(fov)
        print(json.dumps(report, indent=2))
        print("PASS: relative_epe < 1 means the model beats zero flow." if report["relative_epe"] < 1
              else "NOT READY: relative_epe >= 1, so do not connect this model to fusion.")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
