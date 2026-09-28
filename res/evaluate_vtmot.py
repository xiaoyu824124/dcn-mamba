"""Evaluate a standalone registration checkpoint on the VTMOT held-out split."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict

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


def _pool(store: Dict[str, list], name: str, value: float, count: float) -> None:
    """Accumulate a masked mean together with its weight.

    Pooling by pixel rather than by frame matters here: the refinement rounds
    cover different fractions of a frame, and a per-frame mean of per-frame
    means would weight a 2%-covered round the same as a fully covered one.
    """
    if count <= 0:
        return
    total = store.setdefault(name, [0.0, 0.0])
    total[0] += float(value) * count
    total[1] += count


def _pooled(store: Dict[str, list]) -> Dict[str, float]:
    return {name: total / max(count, 1.0) for name, (total, count) in store.items()}


def _ranking_auc(scores: torch.Tensor, positive: torch.Tensor) -> float:
    """P(a positive pixel scores above a negative one), ties counted at chance.

    A mean confidence says nothing about whether the confidence can tell a
    correction that helps from one that hurts, which is what a gate needs; the
    rank statistic is the acceptance test for that.
    """
    scores = scores.reshape(-1).to(torch.float64)
    positive = positive.reshape(-1)
    if scores.numel() != positive.numel():
        raise ValueError("scores and positive must have the same element count")
    n_pos = int(positive.sum())
    n_neg = int(scores.numel()) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5                       # one class missing: no evidence either way
    order = torch.argsort(scores)
    _, inverse, counts = torch.unique(scores[order], return_inverse=True,
                                      return_counts=True)
    starts = torch.cumsum(counts, dim=0) - counts
    midrank = starts.to(torch.float64) + (counts.to(torch.float64) + 1.0) / 2.0
    ranks = torch.empty(scores.numel(), dtype=torch.float64)
    ranks[order] = midrank[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _round_regions(reports: Dict[str, list], index: int, valid: torch.Tensor,
                   size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-resolution covered and valid-query masks for one refinement round.

    The masks live at feature resolution; nearest upsampling keeps the region
    exactly the set of pixels whose own cell was in the window, so the numbers
    stay comparable with the full-resolution ``epe_px``.
    """
    def upsample(mask: torch.Tensor) -> torch.Tensor:
        nearest = F.interpolate(mask.unsqueeze(1).to(torch.float32), size=size,
                                mode="nearest")
        return (nearest > 0.5).to(valid.dtype)
    covered = upsample(reports["covered"][index]) * valid
    return covered, upsample(reports["query"][index]) * valid


def refinement_gate_specs() -> tuple[tuple[str, Callable[[torch.Tensor], torch.Tensor]], ...]:
    """The confidence gates the diagnostic compares, raw included.

    ``raw`` must reproduce the ungated run exactly, so it doubles as a check
    that the hook is inert when no gate is asked for.
    """
    def threshold(value: float) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda confidence: confidence * (confidence >= value)
    return (("raw", lambda confidence: confidence),
            ("zero", lambda confidence: torch.zeros_like(confidence)),
            ("scale0.25", lambda confidence: confidence * 0.25),
            ("scale0.5", lambda confidence: confidence * 0.5),
            ("scale0.75", lambda confidence: confidence * 0.75),
            ("thr0.3", threshold(0.3)),
            ("thr0.5", threshold(0.5)),
            ("thr0.7", threshold(0.7)))


def _select_gate_specs(names: str | None):
    """Restrict the gate sweep to the named specs, in their canonical order.

    The full sweep re-runs the whole loop eight times, so the selector exists to
    keep an iteration short; the names are validated rather than ignored.
    """
    if not names:
        return None
    known = {name: spec for name, spec in refinement_gate_specs()}
    requested = [part.strip() for part in names.split(",") if part.strip()]
    unknown = [part for part in requested if part not in known]
    if unknown:
        raise ValueError(f"unknown gate specs {unknown}; known: {sorted(known)}")
    return tuple((name, known[name]) for name in requested)


def _batch_inputs(batch: Dict, moving_source: str,
                  stress_translation: tuple[float, float] | None,
                  device: torch.device):
    ir = registration_moving_image(batch, moving_source).to(device)
    vi = batch["vi"].to(device)
    target, valid = batch["gt_flow"].to(device), batch["valid_mask"].to(device)
    gt_h = batch["gt_h"].to(device)
    if stress_translation is not None:
        ir, target, valid, gt_h = translate_moving_for_stress(
            ir, target, valid, gt_h, stress_translation)
    return ir, vi, target, valid, gt_h


def evaluate_refinement_gates(model, loader, device: torch.device, *,
                              moving_source: str = "ir",
                              stress_translation: tuple[float, float] | None = None,
                              specs=None) -> Dict[str, float]:
    """Re-run the whole loop once per confidence gate.

    The gate has to act inside the loop: gating round one changes the field
    round two starts from, so filtering the cached corrections afterwards is an
    approximation, not the same experiment.
    """
    specs = refinement_gate_specs() if specs is None else specs
    totals: Dict[str, list] = {}
    model.eval()
    for name, gate in specs:
        print(f"  gate {name}", flush=True)
        for step, batch in enumerate(loader, start=1):
            ir, vi, target, valid, _ = _batch_inputs(
                batch, moving_source, stress_translation, device)
            with torch.no_grad():
                output = model(ir, vi, refinement_gate=gate)
                predicted = (output.final_flow if output.final_flow is not None
                             else output.coarse_flow)
                count = float(valid.sum())
                _pool(totals, f"gate_{name}_epe_px",
                      float(endpoint_error(predicted, target, valid)), count)
                for key, value in _threshold_hits(predicted, target, valid).items():
                    _pool(totals, f"gate_{name}_{key}", value, count)
                reports = refinement_losses(output.refinement, target, valid)
                feature_hw = output.refinement.flows[0].shape[-2:]
                stride_hw = (target.shape[-2] / feature_hw[0],
                             target.shape[-1] / feature_hw[1])
                for index, flow in enumerate(output.refinement.flows, start=1):
                    field = upsample_feature_flow(flow, target.shape[-2:], stride_hw)
                    _pool(totals, f"gate_{name}_round{index}_epe_px",
                          float(endpoint_error(field, target, valid)), count)
                    covered, _ = _round_regions(reports, index - 1, valid,
                                                target.shape[-2:])
                    _pool(totals, f"gate_{name}_round{index}_covered_fraction",
                          float(covered.sum()) / max(count, 1.0), count)
            if step % 20 == 0:
                print(f"    {step} frames", flush=True)
    return _pooled(totals)


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
             moving_source: str = "ir",
             gate_specs=None) -> Dict[str, float]:
    total_epe = total_coarse_epe = total_global_epe = total_baseline = total_valid = total_samples = 0.0
    total_corner_epe = total_cycle_epe = 0.0
    hit_sums = {f"pck_{threshold}px": 0.0 for threshold in (1, 3, 5)}
    diagnostic_sums: Dict[str, float] = {}
    gate_sums: Dict[str, float] = {}
    region_sums: Dict[str, list] = {}
    region_counts: Dict[str, float] = {"refine_rounds": 0.0}
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
            for index, (epe, match, target_rate, mean_norm, applied_std,
                        required_norm, alignment) in enumerate(
                    zip(reports["epe"], reports["match"],
                        reports["confidence_target"], reports["applied_mean_norm"],
                        reports["applied_std"], reports["required_mean_norm"],
                        reports["applied_mean_alignment"]), start=1):
                # The structure terms are already a per-frame norm, a per-frame
                # spatial std or a dimensionless alignment, so pooling them over
                # a batch cancels nothing (a signed per-frame mean would).
                for name, value in ((f"refine_round{index}_epe_px", float(epe) * stride),
                                    (f"refine_round{index}_match", float(match)),
                                    (f"refine_round{index}_improved_fraction",
                                     float(target_rate)),
                                    (f"refine_round{index}_applied_mean_norm_px",
                                     float(mean_norm.mean()) * stride),
                                    (f"refine_round{index}_applied_std_px",
                                     float(applied_std.mean()) * stride),
                                    (f"refine_round{index}_required_mean_norm_px",
                                     float(required_norm.mean()) * stride),
                                    (f"refine_round{index}_applied_mean_alignment",
                                     float(alignment.mean()))):
                    diagnostic_sums[name] = (diagnostic_sums.get(name, 0.0)
                                             + value * float(predicted.shape[0]))
            # Regions, so a round can be read without also having to ask which
            # pixels it covered.  The window is centred on the pre-update field
            # and moves every round, and after a large stress translation it can
            # hold a small, easy subset: ``refine_round*_epe_px`` above is over
            # that subset only and is not comparable with coarse_epe_px.
            feature_hw = output.refinement.flows[0].shape[-2:]
            stride_hw = (target.shape[-2] / feature_hw[0],
                         target.shape[-1] / feature_hw[1])
            frame_covered = []
            for index, (flow, applied, covered_cell, benefit, confidence_pixel) in enumerate(
                    zip(output.refinement.flows, output.refinement.applied,
                        reports["covered"], reports["benefit"],
                        reports["confidence_pixel"]), start=1):
                covered, _ = _round_regions(reports, index - 1, valid, target.shape[-2:])
                outside = valid * (1.0 - covered)
                frame_covered.append(covered)
                field = upsample_feature_flow(flow, target.shape[-2:], stride_hw)
                incoming = upsample_feature_flow(flow - applied, target.shape[-2:],
                                                 stride_hw)
                for name, value, mask in (
                        (f"refine_round{index}_epe_allvalid_px", field, valid),
                        (f"refine_round{index}_epe_noupdate_px", incoming, valid),
                        (f"refine_round{index}_epe_covered_px", field, covered),
                        (f"refine_round{index}_epe_outside_px", field, outside)):
                    _pool(region_sums, name, float(endpoint_error(value, target, mask)),
                          float(mask.sum()))
                for key, mask in ((f"refine_round{index}_covered_pixels", covered),
                                  (f"refine_round{index}_outside_pixels", outside)):
                    region_counts[key] = region_counts.get(key, 0.0) + float(mask.sum())
                region_counts["refine_rounds"] = max(region_counts["refine_rounds"], index)
                # Does the confidence rank the pixels this round helped above the
                # ones it hurt?  A change in the mean confidence says nothing
                # about that, so the ranking is what the gate has to pass.
                aucs, above_half = [], []
                for frame in range(covered_cell.shape[0]):
                    mask = covered_cell[frame]
                    if not bool(mask.any()):
                        continue
                    scores = confidence_pixel[frame][mask].to(torch.float64)
                    gain = benefit[frame][mask] * stride_hw[0]
                    helpful = gain > 0
                    auc = _ranking_auc(scores, helpful)
                    aucs.append(auc)
                    above_half.append(float(auc > 0.5))
                    _pool(region_sums, f"refine_round{index}_gain_all_px",
                          float(gain.mean()), 1.0)
                    if bool(helpful.any()):
                        _pool(region_sums, f"refine_round{index}_confidence_beneficial",
                              float(scores[helpful].mean()), 1.0)
                        _pool(region_sums, f"refine_round{index}_gain_beneficial_px",
                              float(gain[helpful].mean()), 1.0)
                    if bool((~helpful).any()):
                        _pool(region_sums, f"refine_round{index}_confidence_harmful",
                              float(scores[~helpful].mean()), 1.0)
                        _pool(region_sums, f"refine_round{index}_gain_harmful_px",
                              float(gain[~helpful].mean()), 1.0)
                if aucs:
                    frames = float(len(aucs))
                    for name, value in (
                            (f"refine_round{index}_confidence_auc",
                             sum(aucs) / frames),
                            (f"refine_round{index}_confidence_auc_above_half",
                             sum(above_half) / frames)):
                        diagnostic_sums[name] = (diagnostic_sums.get(name, 0.0)
                                                 + value * frames)
            # One fixed region for every round: the cells that stayed inside the
            # window at every round, so round-to-round differences cannot be a
            # change of sample set.
            common = torch.ones_like(valid)
            for covered in frame_covered:
                common = common * covered
            region_counts["refine_common_pixels"] = (
                region_counts.get("refine_common_pixels", 0.0) + float(common.sum()))
            for index in range(1, len(frame_covered) + 1):
                field = upsample_feature_flow(output.refinement.flows[index - 1],
                                              target.shape[-2:], stride_hw)
                _pool(region_sums, f"refine_round{index}_epe_common_px",
                      float(endpoint_error(field, target, common)),
                      float(common.sum()))
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
            for index, (epe, match, mean_norm, applied_std, required_norm) in enumerate(
                    zip(reports["epe"], reports["match"],
                        reports["applied_mean_norm"], reports["applied_std"],
                        reports["required_mean_norm"]), start=1):
                # Started from the truth, so ``required`` is ~0 and any applied
                # mean is pure drift: its norm is the cleanest "bias" number.
                for name, value in ((f"truthcentre_refine_round{index}_epe_px",
                                     float(epe) * stride),
                                    (f"truthcentre_refine_round{index}_match",
                                     float(match)),
                                    (f"truthcentre_refine_round{index}_applied_mean_norm_px",
                                     float(mean_norm.mean()) * stride),
                                    (f"truthcentre_refine_round{index}_applied_std_px",
                                     float(applied_std.mean()) * stride),
                                    (f"truthcentre_refine_round{index}_required_mean_norm_px",
                                     float(required_norm.mean()) * stride)):
                    diagnostic_sums[name] = (diagnostic_sums.get(name, 0.0)
                                             + value * float(predicted.shape[0]))
            if diagnose_confidence_gate and output.local_match is not None:
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
    # Regions: pooled by pixel, and every round's coverage next to its own
    # error, so a round that only saw an easy 5% of the frame cannot be read as
    # an improvement over the coarse field.
    report.update(_pooled(region_sums))
    rounds = int(region_counts.get("refine_rounds", 0.0))
    if rounds:
        report["refine_common_pixels"] = region_counts.get("refine_common_pixels", 0.0)
        report["refine_common_coverage"] = (
            report["refine_common_pixels"] / max(total_valid, 1.0))
        for index in range(1, rounds + 1):
            covered = region_counts[f"refine_round{index}_covered_pixels"]
            outside = region_counts[f"refine_round{index}_outside_pixels"]
            report[f"refine_round{index}_covered_pixels"] = covered
            report[f"refine_round{index}_outside_pixels"] = outside
            report[f"refine_round{index}_coverage"] = covered / max(total_valid, 1.0)
            report[f"refine_round{index}_outside_coverage"] = (
                outside / max(total_valid, 1.0))
    if diagnose_confidence_gate and getattr(model, "iterative_refinement", None) is not None:
        # A real re-run per gate: gating round one changes what round two starts
        # from, so this cannot be done by filtering the cached corrections.
        gates = evaluate_refinement_gates(
            model, loader, device, moving_source=moving_source,
            stress_translation=stress_translation, specs=gate_specs)
        for name, _ in (refinement_gate_specs() if gate_specs is None else gate_specs):
            key = f"gate_{name}_epe_px"
            if key in gates:
                gates[f"gate_{name}_relative_epe"] = (
                    gates[key] / max(report["zero_flow_epe_px"], 1e-8))
        report.update(gates)
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
                        help="compare confidence gates; for the iterative loop this "
                             "re-runs the whole loop once per gate (slow)")
    parser.add_argument("--gate-specs", default=None, metavar="NAMES",
                        help="comma-separated subset of raw,zero,scale0.25,scale0.5,"
                             "scale0.75,thr0.3,thr0.5,thr0.7 for the gate sweep")
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
        # The gate diagnostic applies to whichever 1/4 stage owns a confidence:
        # the single-shot local matcher, or the iterative loop, which is re-run
        # once per gate rather than filtered after the fact.
        if args.diagnose_confidence_gate and model.local_matcher is None \
                and getattr(model, "iterative_refinement", None) is None:
            raise ValueError(
                "--diagnose-confidence-gate requires an enabled local matcher or "
                "an enabled iterative refinement loop")
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
                          moving_source=args.moving_source,
                          gate_specs=_select_gate_specs(args.gate_specs))
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
