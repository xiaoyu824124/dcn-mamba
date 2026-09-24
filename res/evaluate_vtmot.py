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
from .model_factory import build_global_registration
from .affine import affine_corner_errors
from .matching import matching_diagnostics, windowed_diagnostics
from .local_matcher import local_matching_diagnostics
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
    total_epe = total_coarse_epe = total_baseline = total_valid = total_samples = 0.0
    total_corner_epe = total_cycle_epe = 0.0
    hit_sums = {f"pck_{threshold}px": 0.0 for threshold in (1, 3, 5)}
    diagnostic_sums: Dict[str, float] = {}
    model.eval()
    for batch in loader:
        ir, vi = batch["ir"].to(device), batch["vi"].to(device)
        target, valid = batch["gt_flow"].to(device), batch["valid_mask"].to(device)
        output = model(ir, vi)
        predicted = output.final_flow if output.final_flow is not None else output.coarse_flow
        epe = endpoint_error(predicted, target, valid)
        coarse_epe = endpoint_error(output.coarse_flow, target, valid)
        baseline = endpoint_error(torch.zeros_like(target), target, valid)
        count = float(valid.sum())
        total_epe += float(epe) * count
        total_coarse_epe += float(coarse_epe) * count
        total_baseline += float(baseline) * count
        total_valid += count
        corner_epe, cycle_epe = affine_corner_errors(
            output.affine_yx, batch["gt_h"].to(device), predicted.shape[-2:],
            output.confidence_1_8.shape[-2:])
        total_corner_epe += float(corner_epe.sum())
        total_cycle_epe += float(cycle_epe.sum())
        total_samples += float(predicted.shape[0])
        if output.match is not None:
            diagnostics = matching_diagnostics(
                output.match.matching_probability,
                tuple(output.match.coarse_flow.shape[-2:]), target, valid,
                appearance_scores=output.match.correlation,
                affine_confidence_power=model.matcher.affine_confidence_power,
                affine_border_margin=model.matcher.affine_border_margin)
            for radius in (2, 4):
                diagnostics.update(windowed_diagnostics(
                    output.match.matching_probability,
                    tuple(output.match.coarse_flow.shape[-2:]),
                    output.match.coarse_flow, target, valid, radius=radius,
                    appearance_scores=output.match.correlation))
            for name, value in diagnostics.items():
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + value * float(predicted.shape[0])
        if output.local_match is not None:
            for name, value in local_matching_diagnostics(output.local_match, target, valid).items():
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + value * float(predicted.shape[0])
        hits = _threshold_hits(predicted, target, valid)
        for name, value in hits.items():
            hit_sums[name] += value * count
    report = {"epe_px": total_epe / max(total_valid, 1.0),
              "coarse_epe_px": total_coarse_epe / max(total_valid, 1.0),
              "zero_flow_epe_px": total_baseline / max(total_valid, 1.0),
              "valid_pixels": total_valid}
    report["relative_epe"] = report["epe_px"] / max(report["zero_flow_epe_px"], 1e-8)
    report["affine_corner_epe_px"] = total_corner_epe / max(total_samples, 1.0)
    report["affine_gt_inverse_cycle_px"] = total_cycle_epe / max(total_samples, 1.0)
    if diagnostic_sums:
        report.update({name: value / max(total_samples, 1.0)
                       for name, value in diagnostic_sums.items()})
    report.update({name: value / max(total_valid, 1.0) for name, value in hit_sums.items()})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone VTMOT single-frame registration evaluation")
    parser.add_argument("--data-root", default="data/VTMOT_misaligned")
    parser.add_argument("--split-file", default="data_split/IVF/VTMOT/split.json")
    parser.add_argument("--split", choices=("train", "eval", "test"), default="eval",
                        help="use eval while tuning; reserve test for one final report")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--overlay", type=Path, action="append", default=[],
                        help="parameter-free config overlay for a same-checkpoint ablation")
    parser.add_argument("--diagnose-appearance", action="store_true",
                        help="also rank raw cosine matches, before dual softmax or spatial prior")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--crop-hw", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=None,
                        help="optional centre crop after 480x640 normalisation; must match training FOV")
    parser.add_argument("--check-gt", action="store_true",
                        help="check gt_h direction using visible_gt; no model required")
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
        config = OmegaConf.merge(OmegaConf.create(checkpoint["config"]),
                                 *(OmegaConf.load(path) for path in args.overlay))
        model = build_global_registration(config).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        if args.diagnose_appearance:
            model.matcher.return_correlation = True
        report = evaluate(model, loader, device)
        if args.overlay:
            report["evaluation_overlays"] = [str(path) for path in args.overlay]
        report["field_of_view_hw"] = list(fov)
        # Beating zero flow only means "not worse than doing nothing".  A field
        # this coarse still ghosts under fusion, so report readiness explicitly
        # instead of letting a ratio just below 1 look like success.
        report["beats_zero_flow"] = bool(report["relative_epe"] < 1.0)
        report["fusion_ready"] = bool(report["epe_px"] <= 2.0 and report["pck_3px"] >= 0.90)
        print(json.dumps(report, indent=2))
        if report["fusion_ready"]:
            print("FUSION READY: sub-2px EPE and at least 90% of pixels within 3px.")
        elif report["beats_zero_flow"]:
            print(f"NEEDS REFINEMENT: beats zero flow (ratio={report['relative_epe']:.3f}) but "
                  f"EPE={report['epe_px']:.2f}px and pck@3px={report['pck_3px']:.3f}. "
                  "Continue single-frame training before connecting to fusion.")
        else:
            print("NOT READY: relative_epe >= 1, so do not connect this model to fusion.")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
