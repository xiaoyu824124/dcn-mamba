"""Optional displacement and coarse-match diagnostics for VTMOT evaluation."""

from __future__ import annotations

import math

import torch

from .matching import correspondence_targets, grid_coordinates
from .warp import warp


PIXEL_MOTION_EDGES = (0.0, 4.0, 8.0, 16.0, 32.0, math.inf)
FRAME_MOTION_EDGES = (0.0, 4.0, 8.0, 12.0, 16.0, math.inf)
MATCH_MOTION_EDGES = (0.0, 8.0, 16.0, 32.0, math.inf)


def translate_moving_for_stress(
        moving: torch.Tensor, gt_flow: torch.Tensor, valid: torch.Tensor,
        gt_h: torch.Tensor, extra_dy_dx: tuple[float, float]
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add a known fixed-to-moving translation with zero-padded borders.

    The new moving image is ``moving'(q)=moving(q-extra)``. Its GT map is
    therefore ``flow'(p)=flow(p)+extra``. The returned validity mask also
    rejects fixed pixels whose new moving-image coordinate falls outside.
    """
    if moving.ndim != 4 or gt_flow.shape != (moving.shape[0], 2, *moving.shape[-2:]):
        raise ValueError("moving and gt_flow shapes disagree")
    batch, _, height, width = moving.shape
    if valid.shape != (batch, 1, height, width) or gt_h.shape != (batch, 3, 3):
        raise ValueError("valid or gt_h shape disagrees with moving image")
    dy, dx = map(float, extra_dy_dx)
    if not math.isfinite(dy) or not math.isfinite(dx):
        raise ValueError("stress translation must be finite")
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError("stress translation must be smaller than the image")
    delta = gt_flow.new_tensor((dy, dx)).view(1, 2, 1, 1).expand_as(gt_flow)
    shifted_moving = warp(moving, -delta)
    stressed_flow = gt_flow + delta
    y = torch.arange(height, device=gt_flow.device, dtype=gt_flow.dtype).view(1, height, 1)
    x = torch.arange(width, device=gt_flow.device, dtype=gt_flow.dtype).view(1, 1, width)
    mapped_y = y + stressed_flow[:, 0]
    mapped_x = x + stressed_flow[:, 1]
    inside = ((mapped_y >= 0) & (mapped_y <= height - 1)
              & (mapped_x >= 0) & (mapped_x <= width - 1))
    stressed_valid = valid * inside.unsqueeze(1).to(valid.dtype)
    translation = torch.eye(3, device=gt_h.device, dtype=gt_h.dtype).expand(
        batch, -1, -1).clone()
    translation[:, 0, 2] = dx
    translation[:, 1, 2] = dy
    stressed_h = translation @ gt_h
    return shifted_moving, stressed_flow, stressed_valid, stressed_h


def _bin_label(low: float, high: float) -> str:
    return f">={int(low)}px" if math.isinf(high) else f"{int(low)}-{int(high)}px"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pixel_weighted_mean(rows: list[dict], key: str) -> float | None:
    count = sum(row["valid_pixels"] for row in rows)
    return (sum(row[key] * row["valid_pixels"] for row in rows) / count
            if count else None)


@torch.no_grad()
def frame_motion_diagnostics(predicted: torch.Tensor, coarse: torch.Tensor,
                             target: torch.Tensor, valid: torch.Tensor,
                             probability: torch.Tensor | None,
                             feature_hw: tuple[int, int] | None = None,
                             affine_power: float = 4.0,
                             affine_border_margin: int = 0,
                             *, sequence: str = "", stem: str = "") -> dict:
    """Describe one frame; GT is used only to score, never to select matches."""
    if predicted.shape != target.shape or coarse.shape != target.shape:
        raise ValueError("predicted, coarse and target flows must have equal shapes")
    if target.ndim != 4 or target.shape[0] != 1 or target.shape[1] != 2:
        raise ValueError("motion diagnostics require one [1,2,H,W] frame")
    if valid.shape != (1, 1, *target.shape[-2:]):
        raise ValueError("valid mask must be [1,1,H,W]")

    gt_motion = torch.linalg.vector_norm(target.float(), dim=1)
    error = torch.linalg.vector_norm((predicted - target).float(), dim=1)
    coarse_error = torch.linalg.vector_norm((coarse - target).float(), dim=1)
    usable = valid[:, 0] > 0.5
    count = int(usable.sum())
    if count == 0:
        raise ValueError("motion diagnostics require valid GT pixels")
    row = {
        "sequence": sequence, "stem": stem, "valid_pixels": count,
        "gt_motion_mean_px": float(gt_motion[usable].mean()),
        "gt_motion_p90_px": float(torch.quantile(gt_motion[usable], 0.9)),
        "epe_px": float(error[usable].mean()),
        "coarse_epe_px": float(coarse_error[usable].mean()),
        "zero_flow_epe_px": float(gt_motion[usable].mean()),
    }
    pixel_bins = {}
    for low, high in zip(PIXEL_MOTION_EDGES[:-1], PIXEL_MOTION_EDGES[1:]):
        selected = usable & (gt_motion >= low) & (gt_motion < high)
        n = int(selected.sum())
        pixel_bins[_bin_label(low, high)] = {
            "pixels": n,
            "epe_sum": float(error[selected].sum()) if n else 0.0,
            "coarse_epe_sum": float(coarse_error[selected].sum()) if n else 0.0,
            "zero_flow_epe_sum": float(gt_motion[selected].sum()) if n else 0.0,
        }
    row["pixel_bins"] = pixel_bins

    if probability is None:
        return row
    if probability.ndim != 3 or probability.shape[0] != 1:
        raise ValueError("probability must be [1,N,N]")
    if feature_hw is None:
        raise ValueError("feature_hw is required with matching probabilities")
    feature_h, feature_w = map(int, feature_hw)
    tokens = probability.shape[1]
    if feature_h * feature_w != tokens or probability.shape[2] != tokens:
        raise ValueError("coarse matching probability must match feature_hw")
    coordinates = grid_coordinates(feature_h, feature_w, probability)
    confidence, index = probability[0].float().max(dim=-1)
    weights = confidence.clamp_min(1e-8).pow(affine_power)
    if affine_border_margin:
        inner = ((coordinates[:, 0] >= affine_border_margin)
                 & (coordinates[:, 0] < feature_h - affine_border_margin)
                 & (coordinates[:, 1] >= affine_border_margin)
                 & (coordinates[:, 1] < feature_w - affine_border_margin))
        weights = weights * inner.float()
    total_weight = weights.sum().clamp_min(1e-20)
    top_count = max(1, math.ceil(tokens * 0.1))
    top = confidence.topk(top_count).indices
    top_mask = torch.zeros(tokens, dtype=torch.bool, device=probability.device)
    top_mask[top] = True
    upper = coordinates[:, 0] < feature_h / 2
    left = coordinates[:, 1] < feature_w / 2
    quadrants = {
        "upper_left": upper & left,
        "upper_right": upper & ~left,
        "lower_left": ~upper & left,
        "lower_right": ~upper & ~left,
    }
    matched_coordinates = coordinates[index]
    key_upper = matched_coordinates[:, 0] < feature_h / 2
    key_left = matched_coordinates[:, 1] < feature_w / 2
    key_quadrants = {
        "upper_left": key_upper & key_left,
        "upper_right": key_upper & ~key_left,
        "lower_left": ~key_upper & key_left,
        "lower_right": ~key_upper & ~key_left,
    }
    row["match_distribution"] = {
        "wls_weight_quadrants": {
            name: float(weights[mask].sum() / total_weight)
            for name, mask in quadrants.items()},
        "top10_conf_query_quadrants": {
            name: float((top_mask & mask).sum() / top_count)
            for name, mask in quadrants.items()},
        "top10_conf_key_quadrants": {
            name: float((top_mask & mask).sum() / top_count)
            for name, mask in key_quadrants.items()},
        "wls_weight_key_quadrants": {
            name: float(weights[mask].sum() / total_weight)
            for name, mask in key_quadrants.items()},
        "top10_conf_wls_weight_fraction": float(weights[top_mask].sum() / total_weight),
        "wls_effective_queries_ratio": float(total_weight.square() /
            (tokens * weights.square().sum().clamp_min(1e-20))),
    }
    target_cells, query_valid = correspondence_targets(target, feature_hw, valid)
    query_valid = query_valid[0]
    pixel_stride = target.new_tensor((target.shape[-2] / feature_h,
                                      target.shape[-1] / feature_w))
    match_error = torch.linalg.vector_norm(
        (matched_coordinates - target_cells[0]) * pixel_stride, dim=-1)
    selected_valid = top_mask & query_valid
    top_predicted_motion = torch.linalg.vector_norm(
        (matched_coordinates - coordinates) * pixel_stride, dim=-1)
    top_gt_motion = torch.linalg.vector_norm(
        (target_cells[0] - coordinates) * pixel_stride, dim=-1)
    motion_bins = {}
    for low, high in zip(MATCH_MOTION_EDGES[:-1], MATCH_MOTION_EDGES[1:]):
        name = _bin_label(low, high)
        motion_bins[name] = {
            "predicted_fraction": float(((top_predicted_motion >= low)
                                         & (top_predicted_motion < high)
                                         & top_mask).sum() / top_count),
            "gt_valid_fraction": (float(((top_gt_motion >= low)
                                         & (top_gt_motion < high)
                                         & selected_valid).sum() / selected_valid.sum())
                                  if bool(selected_valid.any()) else None),
        }
    row["match_distribution"].update({
        "top10_conf_gt_valid_fraction": float(selected_valid.sum() / top_count),
        "top10_conf_predicted_motion_mean_px": float(top_predicted_motion[top_mask].mean()),
        "top10_conf_gt_motion_mean_px": (
            float(top_gt_motion[selected_valid].mean()) if bool(selected_valid.any()) else None),
        "top10_conf_argmax_epe_px": (
            float(match_error[selected_valid].mean()) if bool(selected_valid.any()) else None),
        "top10_conf_match_pck_8px": (
            float((match_error[selected_valid] <= 8).float().mean())
            if bool(selected_valid.any()) else None),
        "top10_conf_motion_bins": motion_bins,
    })
    return row


def summarize_motion_frames(rows: list[dict]) -> dict:
    """Aggregate per-frame and per-pixel motion bins without hiding empty bins."""
    if not rows:
        return {"motion_frame_bins": {}, "motion_pixel_bins": {}, "motion_frames": []}
    frame_bins = {}
    for low, high in zip(FRAME_MOTION_EDGES[:-1], FRAME_MOTION_EDGES[1:]):
        selected = [row for row in rows if low <= row["gt_motion_mean_px"] < high]
        frame_bins[_bin_label(low, high)] = {
            "frames": len(selected),
            "valid_pixels": sum(row["valid_pixels"] for row in selected),
            "epe_px": _pixel_weighted_mean(selected, "epe_px"),
            "coarse_epe_px": _pixel_weighted_mean(selected, "coarse_epe_px"),
            "zero_flow_epe_px": _pixel_weighted_mean(selected, "zero_flow_epe_px"),
            "match_top10_argmax_epe_px": _mean([
                row["match_distribution"]["top10_conf_argmax_epe_px"]
                for row in selected if "match_distribution" in row
                and row["match_distribution"]["top10_conf_argmax_epe_px"] is not None]),
            "match_top10_pck_8px": _mean([
                row["match_distribution"]["top10_conf_match_pck_8px"]
                for row in selected if "match_distribution" in row
                and row["match_distribution"]["top10_conf_match_pck_8px"] is not None]),
            "match_top10_predicted_motion_mean_px": _mean([
                row["match_distribution"]["top10_conf_predicted_motion_mean_px"]
                for row in selected if "match_distribution" in row]),
            "match_top10_gt_motion_mean_px": _mean([
                row["match_distribution"]["top10_conf_gt_motion_mean_px"]
                for row in selected if "match_distribution" in row
                and row["match_distribution"]["top10_conf_gt_motion_mean_px"] is not None]),
            "wls_effective_queries_ratio": _mean([
                row["match_distribution"]["wls_effective_queries_ratio"]
                for row in selected if "match_distribution" in row]),
        }
    pixel_bins = {}
    for low, high in zip(PIXEL_MOTION_EDGES[:-1], PIXEL_MOTION_EDGES[1:]):
        name = _bin_label(low, high)
        count = sum(row["pixel_bins"][name]["pixels"] for row in rows)
        pixel_bins[name] = {
            "pixels": count,
            "epe_px": (sum(row["pixel_bins"][name]["epe_sum"] for row in rows) / count
                       if count else None),
            "coarse_epe_px": (sum(row["pixel_bins"][name]["coarse_epe_sum"] for row in rows) / count
                              if count else None),
            "zero_flow_epe_px": (sum(row["pixel_bins"][name]["zero_flow_epe_sum"] for row in rows) / count
                                 if count else None),
        }
    ordered = sorted(rows, key=lambda row: row["gt_motion_mean_px"], reverse=True)
    quartile_count = max(1, math.ceil(len(ordered) * 0.25))
    matched = [row["match_distribution"] for row in rows
               if "match_distribution" in row]
    match_summary = None
    if matched:
        match_summary = {
            "wls_weight_quadrants": {
                name: _mean([item["wls_weight_quadrants"][name] for item in matched])
                for name in matched[0]["wls_weight_quadrants"]},
            "top10_conf_query_quadrants": {
                name: _mean([item["top10_conf_query_quadrants"][name] for item in matched])
                for name in matched[0]["top10_conf_query_quadrants"]},
            "top10_conf_key_quadrants": {
                name: _mean([item["top10_conf_key_quadrants"][name] for item in matched])
                for name in matched[0]["top10_conf_key_quadrants"]},
            "wls_weight_key_quadrants": {
                name: _mean([item["wls_weight_key_quadrants"][name] for item in matched])
                for name in matched[0]["wls_weight_key_quadrants"]},
            "wls_effective_queries_ratio": _mean([
                item["wls_effective_queries_ratio"] for item in matched]),
            "top10_conf_wls_weight_fraction": _mean([
                item["top10_conf_wls_weight_fraction"] for item in matched]),
            "top10_conf_gt_valid_fraction": _mean([
                item["top10_conf_gt_valid_fraction"] for item in matched]),
            "top10_conf_predicted_motion_mean_px": _mean([
                item["top10_conf_predicted_motion_mean_px"] for item in matched]),
            "top10_conf_gt_motion_mean_px": _mean([
                item["top10_conf_gt_motion_mean_px"] for item in matched
                if item["top10_conf_gt_motion_mean_px"] is not None]),
            "top10_conf_argmax_epe_px": _mean([
                item["top10_conf_argmax_epe_px"] for item in matched
                if item["top10_conf_argmax_epe_px"] is not None]),
            "top10_conf_match_pck_8px": _mean([
                item["top10_conf_match_pck_8px"] for item in matched
                if item["top10_conf_match_pck_8px"] is not None]),
            "top10_conf_motion_bins": {
                name: {
                    "predicted_fraction": _mean([
                        item["top10_conf_motion_bins"][name]["predicted_fraction"]
                        for item in matched]),
                    "gt_valid_fraction": _mean([
                        item["top10_conf_motion_bins"][name]["gt_valid_fraction"]
                        for item in matched
                        if item["top10_conf_motion_bins"][name]["gt_valid_fraction"]
                        is not None]),
                }
                for name in matched[0]["top10_conf_motion_bins"]},
        }
    return {
        "motion_frame_bins": frame_bins,
        "motion_pixel_bins": pixel_bins,
        "highest_motion_quartile": {
            "frames": quartile_count,
            "epe_px": _pixel_weighted_mean(ordered[:quartile_count], "epe_px"),
            "zero_flow_epe_px": _pixel_weighted_mean(
                ordered[:quartile_count], "zero_flow_epe_px"),
        },
        "lowest_motion_quartile": {
            "frames": quartile_count,
            "epe_px": _pixel_weighted_mean(ordered[-quartile_count:], "epe_px"),
            "zero_flow_epe_px": _pixel_weighted_mean(
                ordered[-quartile_count:], "zero_flow_epe_px"),
        },
        "match_distribution_summary": match_summary,
        "motion_frames": ordered,
    }
