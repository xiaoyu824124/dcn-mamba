"""Evaluate a standalone registration checkpoint on the VTMOT held-out split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .metrics import endpoint_error
from .model_factory import build_global_registration
from .affine import affine_corner_errors
from .matching import matching_diagnostics, windowed_diagnostics
from .local_matcher import local_matching_diagnostics
from .iterative_refinement import refinement_losses
from .checkpoint import load_registration_state
from .mind import paired_mind
from .motion_diagnostics import (frame_motion_diagnostics,
                                 summarize_motion_frames,
                                 translate_moving_for_stress)
from .vtmot import VTMOTSingleFrameDataset, registration_moving_image
from .warp import upsample_feature_flow, warp


def _threshold_hits(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    error = torch.linalg.vector_norm(predicted - target, dim=1, keepdim=True)
    denominator = valid.sum().clamp_min(1)
    return {f"pck_{threshold}px": float(((error <= threshold) * valid).sum() / denominator)
            for threshold in (1, 3, 5)}


def _registration_status(report: Dict[str, float], moving_source: str
                         ) -> tuple[bool, bool, str]:
    """Keep same-modal warmup accuracy separate from IR--VI fusion readiness."""
    beats_zero = bool(report["relative_epe"] < 1.0)
    accuracy_gate = bool(report["epe_px"] <= 2.0 and report["pck_3px"] >= 0.90)
    fusion_ready = moving_source == "ir" and beats_zero and accuracy_gate
    if moving_source == "visible_gt":
        message = (f"SAME-MODAL WARMUP: EPE={report['epe_px']:.2f}px and "
                   f"pck@3px={report['pck_3px']:.3f}; evaluate IR-VI separately.")
    elif fusion_ready:
        message = "FUSION READY: sub-2px EPE and at least 90% of pixels within 3px."
    elif beats_zero:
        message = (f"NEEDS REFINEMENT: beats zero flow (ratio={report['relative_epe']:.3f}) but "
                   f"EPE={report['epe_px']:.2f}px and pck@3px={report['pck_3px']:.3f}. "
                   "Continue single-frame training before connecting to fusion.")
    else:
        message = "NOT READY: relative_epe >= 1, so do not connect this model to fusion."
    return beats_zero, fusion_ready, message


def _top_confidence_gate(probability: torch.Tensor, valid_candidates: torch.Tensor,
                         image_hw: tuple[int, int], fraction: float) -> torch.Tensor:
    """Select the most confident sampled 1/4 queries, then lift to image size.

    Selection uses only model outputs; ground-truth flow and its validity mask
    are never consulted. The returned mask is [B,1,H,W].
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    batch, _, height, width = probability.shape
    if valid_candidates.shape != probability.shape:
        raise ValueError("valid_candidates must match probability shape")
    low_valid = valid_candidates.any(dim=1)
    confidence = probability.amax(dim=1)
    chosen = torch.zeros_like(low_valid)
    for batch_index in range(batch):
        indices = low_valid[batch_index].flatten().nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        count = max(1, math.ceil(indices.numel() * fraction))
        ranking = confidence[batch_index].flatten()[indices].topk(count).indices
        chosen[batch_index].flatten()[indices[ranking]] = True
    return F.interpolate(chosen[:, None].float(), size=image_hw, mode="nearest")


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
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             diagnose_local_mind: bool = False,
             diagnose_confidence_gate: bool = False,
             diagnose_local_centre: bool = False,
             diagnose_motion: bool = False,
             stress_translation: tuple[float, float] | None = None,
             moving_source: str = "ir") -> Dict[str, float]:
    total_epe = total_coarse_epe = total_global_epe = total_baseline = total_valid = total_samples = 0.0
    total_corner_epe = total_cycle_epe = 0.0
    hit_sums = {f"pck_{threshold}px": 0.0 for threshold in (1, 3, 5)}
    diagnostic_sums: Dict[str, float] = {}
    gate_sums: Dict[str, float] = {}
    motion_rows: list[dict] = []
    coarse_grid_hw = None
    if stress_translation is not None and moving_source != "ir":
        raise ValueError("stress translation requires moving_source=ir")
    model.eval()
    for batch in loader:
        ir = registration_moving_image(batch, moving_source).to(device)
        vi = batch["vi"].to(device)
        target, valid = batch["gt_flow"].to(device), batch["valid_mask"].to(device)
        gt_h = batch["gt_h"].to(device)
        if stress_translation is not None:
            ir, target, valid, gt_h = translate_moving_for_stress(
                ir, target, valid, gt_h, stress_translation)
        output = model(ir, vi)
        if output.match is not None:
            coarse_grid_hw = tuple(output.match.coarse_flow.shape[-2:])
        predicted = output.final_flow if output.final_flow is not None else output.coarse_flow
        epe = endpoint_error(predicted, target, valid)
        coarse_epe = endpoint_error(output.coarse_flow, target, valid)
        baseline = endpoint_error(torch.zeros_like(target), target, valid)
        count = float(valid.sum())
        total_epe += float(epe) * count
        total_coarse_epe += float(coarse_epe) * count
        if output.global_flow is not None:
            total_global_epe += float(endpoint_error(output.global_flow, target, valid)) * count
        total_baseline += float(baseline) * count
        total_valid += count
        corner_epe, cycle_epe = affine_corner_errors(
            output.affine_yx, gt_h, predicted.shape[-2:],
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
                window_diagnostics = windowed_diagnostics(
                    output.match.matching_probability,
                    tuple(output.match.coarse_flow.shape[-2:]),
                    output.match.coarse_flow, target, valid, radius=radius,
                    appearance_scores=output.match.correlation)
                if output.global_flow is not None:
                    diagnostics.update({"global_" + name: value
                                        for name, value in window_diagnostics.items()})
                else:
                    diagnostics.update(window_diagnostics)
            if not model.matcher.affine_projection:
                # GLU-CRFT fits affine to a learned cost-volume decoder, not
                # to probability-peak weighted soft matches.  A model may
                # expose global_flow while still using WLS (the 1/8
                # spatial-frequency ablation does), so that field alone must
                # not suppress its actual WLS diagnostics.
                diagnostics = {name: value for name, value in diagnostics.items()
                               if not name.startswith("affine_weight")}
            for name, value in diagnostics.items():
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + value * float(predicted.shape[0])
        if output.coarse_local_match is not None:
            for name, value in local_matching_diagnostics(
                    output.coarse_local_match, target, valid).items():
                diagnostic_sums["coarse_" + name] = (
                    diagnostic_sums.get("coarse_" + name, 0.0)
                    + value * float(predicted.shape[0]))
        if output.refinement is not None:
            # Per-round numbers, because the exit condition for this stage is
            # that the third round beats the first, and an average over rounds
            # cannot show that.  ``epe_px`` is comparable with coarse_epe_px.
            with torch.no_grad():
                reports = refinement_losses(output.refinement, target, valid)
            stride = target.shape[-2] / output.refinement.flows[0].shape[-2]
            for index, (epe, match, target_rate) in enumerate(
                    zip(reports["epe"], reports["match"],
                        reports["confidence_target"]), start=1):
                for name, value in ((f"refine_round{index}_epe_px", float(epe) * stride),
                                    (f"refine_round{index}_match", float(match)),
                                    (f"refine_round{index}_improved_fraction",
                                     float(target_rate))):
                    diagnostic_sums[name] = (diagnostic_sums.get(name, 0.0)
                                             + value * float(predicted.shape[0]))
        if output.local_match is not None:
            for name, value in local_matching_diagnostics(output.local_match, target, valid).items():
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + value * float(predicted.shape[0])
        if diagnose_local_centre and output.local_match is not None:
            # Re-run only the local stage with the *truth* as the window centre.
            # This removes the coarse-field error from the measurement: if the
            # truth-centred search still picks the wrong candidate, the 1/4
            # features are not discriminative enough and no coarse improvement
            # will help; if it picks correctly, the window content is fine and
            # the coarse field (or the window size) is what to fix.
            if not getattr(model, "supports_local_centre", False):
                raise ValueError(
                    "--diagnose-local-centre requires a registration model that accepts "
                    "local_centre; currently only StructuralPriorRegistration does")
            with torch.no_grad():
                truth_centred = model(ir, vi, local_centre=target)
            for name, value in local_matching_diagnostics(
                    truth_centred.local_match, target, valid).items():
                diagnostic_sums["truthcentre_" + name] = (
                    diagnostic_sums.get("truthcentre_" + name, 0.0)
                    + value * float(predicted.shape[0]))
        if diagnose_local_centre and output.local_match is None \
                and output.refinement is not None:
            # The same idea for the iterative loop: start it from the truth and
            # the per-round numbers lose the coarse-field error, so what remains
            # is the loop's own ability to walk to the correspondence.
            if not getattr(model, "supports_local_centre", False):
                raise ValueError(
                    "--diagnose-local-centre requires a 1/4 stage that accepts an "
                    "init_flow override")
            with torch.no_grad():
                truth_centred = model(ir, vi, init_flow=target)
                reports = refinement_losses(truth_centred.refinement, target, valid)
            stride = target.shape[-2] / truth_centred.refinement.flows[0].shape[-2]
            for index, (epe, match) in enumerate(zip(reports["epe"], reports["match"]),
                                                 start=1):
                for name, value in ((f"truthcentre_refine_round{index}_epe_px",
                                     float(epe) * stride),
                                    (f"truthcentre_refine_round{index}_match",
                                     float(match))):
                    diagnostic_sums[name] = (diagnostic_sums.get(name, 0.0)
                                             + value * float(predicted.shape[0]))
            if diagnose_confidence_gate:
                local = output.local_match
                feature_hw = local.coarse_flow.shape[-2:]
                stride = (target.shape[-2] / feature_hw[0], target.shape[-1] / feature_hw[1])
                soft_flow = upsample_feature_flow(local.coarse_flow + local.matched_residual,
                                                  target.shape[-2:], stride)
                for percent in (10, 25, 50):
                    gate = _top_confidence_gate(local.probability, local.valid_candidates,
                                                target.shape[-2:], percent / 100)
                    for kind, candidate in (("head", predicted), ("soft", soft_flow)):
                        gated = output.coarse_flow + gate * (candidate - output.coarse_flow)
                        name = f"local_gate_top{percent}_{kind}_epe_px"
                        gate_sums[name] = gate_sums.get(name, 0.0) + float(
                            endpoint_error(gated, target, valid)) * count
                    name = f"local_gate_top{percent}_valid_fraction"
                    gate_sums[name] = gate_sums.get(name, 0.0) + float(
                        (gate * valid).sum())
            if diagnose_local_mind:
                mind_ir, mind_vi = paired_mind(ir, vi, model.mind)
                mind_ir_4 = F.avg_pool2d(mind_ir.float(), kernel_size=4, stride=4)
                mind_vi_4 = F.avg_pool2d(mind_vi.float(), kernel_size=4, stride=4)
                mind_match = model.local_matcher(
                    mind_ir_4, mind_vi_4, output.local_match.coarse_flow)
                mind_diagnostics = local_matching_diagnostics(mind_match, target, valid)
                for name in ("local_argmax_epe_px", "local_soft_epe_px"):
                    diagnostic_sums["mind_" + name] = (
                        diagnostic_sums.get("mind_" + name, 0.0)
                        + mind_diagnostics[name] * float(predicted.shape[0]))
        hits = _threshold_hits(predicted, target, valid)
        for name, value in hits.items():
            hit_sums[name] += value * count
        if diagnose_motion:
            if predicted.shape[0] != 1:
                raise ValueError("--diagnose-motion requires --batch-size 1")
            match = output.match
            motion_rows.append(frame_motion_diagnostics(
                predicted, output.coarse_flow, target, valid,
                match.matching_probability if match is not None else None,
                tuple(match.coarse_flow.shape[-2:]) if match is not None else None,
                affine_power=model.matcher.affine_confidence_power,
                affine_border_margin=model.matcher.affine_border_margin,
                sequence=str(batch["sequence"][0]) if "sequence" in batch else "",
                stem=str(batch["stem"][0]) if "stem" in batch else ""))
    report = {"epe_px": total_epe / max(total_valid, 1.0),
              "coarse_epe_px": total_coarse_epe / max(total_valid, 1.0),
              "zero_flow_epe_px": total_baseline / max(total_valid, 1.0),
              "valid_pixels": total_valid}
    if coarse_grid_hw is not None:
        report["coarse_match_grid_hw"] = list(coarse_grid_hw)
        report["coarse_match_candidates"] = coarse_grid_hw[0] * coarse_grid_hw[1]
    if getattr(model, "coarse_decoder", None) is not None:
        report["global_epe_px"] = total_global_epe / max(total_valid, 1.0)
        report["architecture"] = "glu_crft"
    if getattr(model, "fusion", None) is not None:
        report["architecture"] = "spatial_frequency"
        report["representation"] = model.fusion.mode
    if getattr(model, "encoder", None) is not None and hasattr(model.encoder, "ir_shallow"):
        report["architecture"] = "structural_prior"
    if hasattr(model, "prior_downsample"):
        report["architecture"] = "structural_prior_direct"
    report["relative_epe"] = report["epe_px"] / max(report["zero_flow_epe_px"], 1e-8)
    report["affine_corner_epe_px"] = total_corner_epe / max(total_samples, 1.0)
    report["affine_gt_inverse_cycle_px"] = total_cycle_epe / max(total_samples, 1.0)
    if diagnostic_sums:
        report.update({name: value / max(total_samples, 1.0)
                       for name, value in diagnostic_sums.items()})
    if gate_sums:
        report.update({name: value / max(total_valid, 1.0)
                       for name, value in gate_sums.items()})
    report.update({name: value / max(total_valid, 1.0) for name, value in hit_sums.items()})
    if diagnose_motion:
        report.update(summarize_motion_frames(motion_rows))
    if stress_translation is not None:
        report["stress_translation_dy_dx"] = list(stress_translation)
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
    parser.add_argument("--diagnose-local-mind", action="store_true",
                        help="compare 1/4 Encoder features with pooled raw MIND descriptors at the same coarse field")
    parser.add_argument("--diagnose-confidence-gate", action="store_true",
                        help="evaluate top-confidence local corrections without changing the checkpoint")
    parser.add_argument("--diagnose-local-centre", action="store_true",
                        help="re-run the 1/4 search with the ground truth as the window centre, "
                             "separating 1/4 feature discriminability from coarse-field error "
                             "(requires structural_prior with an enabled local matcher)")
    parser.add_argument("--diagnose-motion", action="store_true",
                        help="report per-frame and per-pixel GT displacement bins plus coarse-match spatial distribution")
    parser.add_argument("--stress-translation", type=float, nargs=2,
                        metavar=("DY", "DX"), default=None,
                        help="add known [dy,dx] displacement to IR GT by translating only the moving raster")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--moving-source", choices=("ir", "visible_gt"), default="ir",
                        help="evaluate infrared (default) or aligned visible grayscale")
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
    if args.diagnose_motion and args.batch_size != 1:
        raise ValueError("--diagnose-motion requires --batch-size 1")
    if args.diagnose_motion and args.output is None:
        raise ValueError("--diagnose-motion requires --output to save per-frame records")
    if args.stress_translation is not None and args.moving_source != "ir":
        raise ValueError("--stress-translation requires --moving-source ir")
    if args.stress_translation is not None and args.check_gt:
        raise ValueError("--stress-translation cannot be combined with --check-gt")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = VTMOTSingleFrameDataset(args.data_root, split=args.split, split_file=args.split_file,
                                      frame_stride=args.frame_stride,
                                      include_rgb_gt=args.check_gt or args.moving_source == "visible_gt",
                                      crop_hw=tuple(args.crop_hw) if args.crop_hw is not None else None,
                                      max_samples=args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    fov = tuple(args.crop_hw) if args.crop_hw is not None else (480, 640)
    print(f"split={args.split} samples={len(dataset)} stride={args.frame_stride} "
          f"fov={fov} device={device} moving_source={args.moving_source}")
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
        # Tolerant of submodules the checkpoint predates (a 1/4 local stage added
        # after the coarse field was frozen), strict about everything else.
        load_report = load_registration_state(model, checkpoint["model"])
        if load_report["untrained"]:
            print(f"checkpoint={args.checkpoint} kept={load_report['kept']} "
                  f"newly built (untrained) modules: "
                  f"{sorted({name.split('.')[0] for name in load_report['untrained']})}")
        if args.diagnose_local_mind and model.local_matcher is None:
            raise ValueError("--diagnose-local-mind requires an enabled local matcher")
        if args.diagnose_confidence_gate and model.local_matcher is None:
            raise ValueError("--diagnose-confidence-gate requires an enabled local matcher")
        if args.diagnose_local_centre and not getattr(model, "supports_local_centre",
                                                      False):
            # The centre override is what the truth-centred diagnostic needs; a
            # model may expose it through init_flow without owning a single-shot
            # local matcher, which is the case for the iterative loop.
            raise ValueError(
                "--diagnose-local-centre requires a 1/4 stage that accepts a centre "
                "override (init_flow / local_centre)")
        if args.diagnose_appearance:
            model.matcher.return_correlation = True
        report = evaluate(model, loader, device,
                          diagnose_local_mind=args.diagnose_local_mind,
                          diagnose_confidence_gate=args.diagnose_confidence_gate,
                          diagnose_local_centre=args.diagnose_local_centre,
                          diagnose_motion=args.diagnose_motion,
                          stress_translation=(tuple(args.stress_translation)
                                              if args.stress_translation is not None else None),
                          moving_source=args.moving_source)
        report["moving_source"] = args.moving_source
        if args.overlay:
            report["evaluation_overlays"] = [str(path) for path in args.overlay]
        report["field_of_view_hw"] = list(fov)
        # Beating zero flow only means "not worse than doing nothing".  A field
        # this coarse still ghosts under fusion, so report readiness explicitly
        # instead of letting a ratio just below 1 look like success.
        beats_zero, fusion_ready, status = _registration_status(report, args.moving_source)
        report["beats_zero_flow"] = beats_zero
        report["fusion_ready"] = fusion_ready
        # Keep the console readable; the requested JSON file retains all
        # per-frame records for paired same-frame comparisons.
        console_report = {name: value for name, value in report.items()
                          if name != "motion_frames"}
        print(json.dumps(console_report, indent=2))
        print(status)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
