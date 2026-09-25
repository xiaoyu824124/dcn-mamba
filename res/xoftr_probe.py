"""Evaluate official XoFTR sparse VI->IR matches against VTMOT geometry.

The upstream repository and pretrained weights remain separate from this repo.
Coordinates are [x, y]; VTMOT flow channels are [dy, dx]. No ground truth is
used to estimate the affine transform.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .mind import rgb_to_gray


def build_xoftr(source_root: str | Path, checkpoint_path: str | Path,
                device: torch.device) -> torch.nn.Module:
    """Construct XoFTR from its official source and load a strict checkpoint."""
    root = Path(source_root).resolve()
    if not (root / "src" / "xoftr" / "xoftr.py").is_file() or not (
            root / "src" / "config" / "default.py").is_file():
        raise FileNotFoundError(f"official XoFTR source tree not found at {root}")
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"XoFTR checkpoint not found at {checkpoint_file}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    upstream_config = importlib.import_module("src.config.default")
    if not Path(upstream_config.__file__).resolve().is_relative_to(root):
        raise RuntimeError("a different 'src' package is imported; run in a fresh process")
    config = upstream_config.get_cfg_defaults(inference=True)
    config_dict = {str(key).lower(): _lower_config(value)
                   for key, value in config.items()}
    model = importlib.import_module("src.xoftr.xoftr").XoFTR(config_dict["xoftr"])
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("XoFTR checkpoint must contain a tensor state dictionary")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("XoFTR checkpoint has no tensor state dictionary")
    # Official Lightning checkpoints prefix the matching network with matcher.
    state = {name.removeprefix("matcher."): value for name, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _lower_config(value):
    if hasattr(value, "items"):
        return {str(key).lower(): _lower_config(item) for key, item in value.items()}
    return value


@torch.no_grad()
def run_xoftr(model: torch.nn.Module, ir: torch.Tensor,
              vi: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return official keypoints in full-image VI and IR pixel coordinates."""
    if ir.ndim != 4 or ir.shape[1] != 1 or vi.ndim != 4 or vi.shape[1] != 3:
        raise ValueError("expected IR [1,1,H,W] and visible [1,3,H,W]")
    if ir.shape[0] != 1 or vi.shape[0] != 1 or ir.shape[-2:] != vi.shape[-2:]:
        raise ValueError("XoFTR probe uses one equally sized image pair")
    height, width = ir.shape[-2:]
    if height % 8 or width % 8:
        raise ValueError("XoFTR input height and width must be divisible by 8")
    data = {"image0": rgb_to_gray(vi), "image1": ir}
    model(data)
    points0 = data["mkpts0_f"].detach().float().cpu().numpy()
    points1 = data["mkpts1_f"].detach().float().cpu().numpy()
    confidence = data["mconf_f"].detach().float().cpu().numpy()
    if (points0.ndim != 2 or points0.shape[1] != 2 or
            points1.shape != points0.shape or
            confidence.shape != (len(points0),)):
        raise ValueError("upstream XoFTR returned malformed matches")
    if not (np.isfinite(points0).all() and np.isfinite(points1).all()
            and np.isfinite(confidence).all()):
        raise ValueError("upstream XoFTR returned non-finite matches")
    if "m_bids" in data and not bool((data["m_bids"] == 0).all()):
        raise ValueError("upstream XoFTR returned matches for another batch item")
    inside = ((points0[:, 0] >= 0) & (points0[:, 0] <= width - 1)
              & (points0[:, 1] >= 0) & (points0[:, 1] <= height - 1)
              & (points1[:, 0] >= 0) & (points1[:, 0] <= width - 1)
              & (points1[:, 1] >= 0) & (points1[:, 1] <= height - 1))
    return points0[inside], points1[inside], confidence[inside]


def fit_affine_ransac(points0: np.ndarray, points1: np.ndarray, *,
                      seed: int, iterations: int = 1000,
                      threshold_px: float = 5.0) -> tuple[np.ndarray | None, int]:
    """Fit VI->IR affine using only predicted correspondences.

    A broad non-collinear support is required; otherwise a plausible local fit
    can extrapolate arbitrarily across the 480x640 field of view.
    """
    source = np.asarray(points0, dtype=np.float64)
    target = np.asarray(points1, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape:
        raise ValueError("correspondences must have matching [N,2] shapes")
    if len(source) < 6:
        return None, 0
    design = np.column_stack((source, np.ones(len(source))))
    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(len(source), dtype=bool)
    best_error = np.inf
    for _ in range(iterations):
        subset = rng.choice(len(source), size=3, replace=False)
        three = design[subset]
        if abs(np.linalg.det(three)) < 1000.0:
            continue
        candidate = np.linalg.solve(three, target[subset])
        error = np.linalg.norm(design @ candidate - target, axis=1)
        inliers = error <= threshold_px
        count = int(inliers.sum())
        mean_error = float(error[inliers].mean()) if count else np.inf
        if count > best_inliers.sum() or (count == best_inliers.sum()
                                          and mean_error < best_error):
            best_inliers, best_error = inliers, mean_error
            if count == len(source):
                break
    if best_inliers.sum() < 6:
        return None, int(best_inliers.sum())
    # Require support over a meaningful part of the image, not a tiny patch.
    support = source[best_inliers]
    if np.ptp(support[:, 0]) < 40 or np.ptp(support[:, 1]) < 40:
        return None, int(best_inliers.sum())
    affine, *_ = np.linalg.lstsq(design[best_inliers], target[best_inliers], rcond=None)
    if np.linalg.matrix_rank(design[best_inliers]) < 3 or not np.isfinite(affine).all():
        return None, int(best_inliers.sum())
    final_inliers = np.linalg.norm(design @ affine - target, axis=1) <= threshold_px
    return affine.T, int(final_inliers.sum())  # [2,3], maps VI [x,y,1] to IR [x,y]


def affine_to_flow(affine: np.ndarray, height: int, width: int,
                   device: torch.device) -> torch.Tensor:
    """Return a dense [1,2,H,W] VTMOT flow in [dy,dx] order."""
    matrix = torch.as_tensor(affine, dtype=torch.float32, device=device)
    if matrix.shape != (2, 3):
        raise ValueError("affine transform must be [2,3]")
    yy, xx = torch.meshgrid(torch.arange(height, device=device),
                            torch.arange(width, device=device), indexing="ij")
    dx = (matrix[0, 0] - 1) * xx + matrix[0, 1] * yy + matrix[0, 2]
    dy = matrix[1, 0] * xx + (matrix[1, 1] - 1) * yy + matrix[1, 2]
    return torch.stack((dy, dx), dim=0)[None]


def match_gt_error(points0: np.ndarray, points1: np.ndarray,
                   gt_flow: torch.Tensor,
                   valid_mask: torch.Tensor, *,
                   keep_invalid: bool = False) -> np.ndarray:
    """Return match errors; optionally retain invalid GT locations as NaN."""
    if len(points0) == 0:
        return np.empty(0, dtype=np.float32)
    grid, coords, sampled_flow = _sample_flow_xy(points0, gt_flow)
    sampled_valid = F.grid_sample(valid_mask, grid, mode="bilinear",
                                  align_corners=True).flatten()
    predicted = torch.as_tensor(points1.copy(), device=gt_flow.device,
                                dtype=torch.float32)
    errors = torch.linalg.vector_norm(predicted - coords - sampled_flow, dim=-1)
    errors = errors.cpu().numpy()
    supported = (sampled_valid >= 0.999).cpu().numpy()
    if keep_invalid:
        errors[~supported] = np.nan
        return errors
    return errors[supported]


def _sample_flow_xy(points0: np.ndarray, flow_yx: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a dense [dy,dx] field at VI points, returning [dx,dy]."""
    height, width = flow_yx.shape[-2:]
    coords = torch.as_tensor(points0.copy(), device=flow_yx.device,
                             dtype=torch.float32)
    grid = torch.stack((coords[:, 0] * (2 / (width - 1)) - 1,
                        coords[:, 1] * (2 / (height - 1)) - 1), dim=-1)
    grid = grid.view(1, 1, -1, 2)
    sampled_flow = F.grid_sample(flow_yx, grid, mode="bilinear",
                                 align_corners=True).reshape(2, -1).T[:, [1, 0]]
    return grid, coords, sampled_flow


def match_flow_disagreement(points0: np.ndarray, points1: np.ndarray,
                            flow_yx: torch.Tensor) -> np.ndarray:
    """Compare XoFTR displacement with a reference prediction, without GT."""
    if len(points0) == 0:
        return np.empty(0, dtype=np.float32)
    _, coords, sampled_flow = _sample_flow_xy(points0, flow_yx)
    predicted = torch.as_tensor(points1.copy(), device=flow_yx.device,
                                dtype=torch.float32)
    return torch.linalg.vector_norm(predicted - coords - sampled_flow,
                                    dim=-1).cpu().numpy()


@torch.no_grad()
def evaluate_xoftr(model: torch.nn.Module, loader, device: torch.device,
                   *, ransac_iterations: int = 1000,
                   ransac_threshold_px: float = 5.0,
                   reference_model: torch.nn.Module | None = None) -> dict[str, object]:
    """Report raw match quality and full-frame affine EPE with failure coverage."""
    model.eval()
    total_valid = total_epe = total_zero = 0.0
    fit_valid = fit_epe = 0.0
    pixel_hits = {threshold: 0.0 for threshold in (1, 3, 5)}
    raw_errors: list[np.ndarray] = []
    top_conf_errors: list[np.ndarray] = []
    remaining_conf_errors: list[np.ndarray] = []
    inlier_errors: list[np.ndarray] = []
    outlier_errors: list[np.ndarray] = []
    agreement_errors = {(stage, threshold): [] for stage in ("coarse", "final")
                        for threshold in (3, 5, 10)}
    agreement_counts = {(stage, threshold): 0 for stage in ("coarse", "final")
                        for threshold in (3, 5, 10)}
    combined_errors = {(stage, gate): [] for stage in ("coarse", "final")
                       for gate in ("top10pct_conf", "ransac_inlier")}
    combined_counts = {(stage, gate): 0 for stage in ("coarse", "final")
                       for gate in ("top10pct_conf", "ransac_inlier")}
    samples = matches = inliers = successes = 0
    if reference_model is not None:
        reference_model.eval()
    for batch in loader:
        ir, vi = batch["ir"].to(device), batch["vi"].to(device)
        target, valid = batch["gt_flow"].to(device), batch["valid_mask"].to(device)
        points0, points1, confidence = run_xoftr(model, ir, vi)
        errors = match_gt_error(points0, points1, target, valid,
                                keep_invalid=True)
        supported = np.isfinite(errors)
        raw_errors.append(errors[supported])
        top_mask = np.zeros(len(confidence), dtype=bool)
        if len(confidence):
            top_count = max(1, int(np.ceil(0.1 * len(confidence))))
            top_indices = np.argsort(-confidence, kind="stable")[:top_count]
            top_mask[top_indices] = True
            top_conf_errors.append(errors[top_mask & supported])
            remaining_conf_errors.append(errors[~top_mask & supported])
        matches += len(points0)
        affine, count = fit_affine_ransac(points0, points1, seed=samples,
                                         iterations=ransac_iterations,
                                         threshold_px=ransac_threshold_px)
        inlier_mask = np.zeros(len(points0), dtype=bool)
        if affine is not None:
            inliers += count
            design = np.column_stack((points0, np.ones(len(points0))))
            residual = np.linalg.norm(design @ affine.T - points1, axis=1)
            inlier_mask = residual <= ransac_threshold_px
            inlier_errors.append(errors[inlier_mask & supported])
            outlier_errors.append(errors[~inlier_mask & supported])
        if reference_model is not None:
            reference = reference_model(ir, vi)
            reference_fields = {"coarse": reference.coarse_flow,
                                "final": (reference.final_flow if reference.final_flow is not None
                                          else reference.coarse_flow)}
            for stage, field in reference_fields.items():
                disagreement = match_flow_disagreement(points0, points1, field)
                for threshold in (3, 5, 10):
                    selected = disagreement <= threshold
                    agreement_counts[(stage, threshold)] += int(selected.sum())
                    agreement_errors[(stage, threshold)].append(errors[selected & supported])
                    if threshold == 5:
                        for gate, mask in (("top10pct_conf", top_mask),
                                           ("ransac_inlier", inlier_mask)):
                            combined = selected & mask
                            combined_counts[(stage, gate)] += int(combined.sum())
                            combined_errors[(stage, gate)].append(
                                errors[combined & supported])
            del reference
        height, width = target.shape[-2:]
        prediction = (affine_to_flow(affine, height, width, device)
                      if affine is not None else torch.zeros_like(target))
        mask_count = float(valid.sum())
        pixel_error = torch.linalg.vector_norm(prediction - target, dim=1, keepdim=True)
        zero_error = torch.linalg.vector_norm(target, dim=1, keepdim=True)
        frame_epe = float((pixel_error * valid).sum())
        total_valid += mask_count
        total_epe += frame_epe
        total_zero += float((zero_error * valid).sum())
        for threshold in pixel_hits:
            pixel_hits[threshold] += float(((pixel_error <= threshold) * valid).sum())
        if affine is not None:
            successes += 1
            fit_valid += mask_count
            fit_epe += frame_epe
        samples += 1
    if total_valid <= 0:
        raise ValueError("VTMOT evaluation has no valid pixels")
    all_errors = np.concatenate(raw_errors) if raw_errors else np.empty(0)
    top_errors = np.concatenate(top_conf_errors) if top_conf_errors else np.empty(0)
    remaining_errors = (np.concatenate(remaining_conf_errors)
                        if remaining_conf_errors else np.empty(0))
    fit_inlier_errors = np.concatenate(inlier_errors) if inlier_errors else np.empty(0)
    fit_outlier_errors = np.concatenate(outlier_errors) if outlier_errors else np.empty(0)
    report: dict[str, object] = {
        "epe_px": total_epe / total_valid,
        "zero_flow_epe_px": total_zero / total_valid,
        "relative_epe": total_epe / total_zero,
        "valid_pixels": total_valid,
        "samples": samples,
        "fit_success_frames": successes,
        "fit_success_fraction": successes / samples,
        "fit_only_epe_px": fit_epe / fit_valid if fit_valid else None,
        "fit_failure_fallback": "zero_flow",
        "matches_total": matches,
        "matches_gt_valid": int(len(all_errors)),
        "ransac_inlier_matches": inliers,
        "match_epe_px": float(all_errors.mean()) if len(all_errors) else None,
        "match_median_epe_px": float(np.median(all_errors)) if len(all_errors) else None,
        "match_top10pct_conf_valid": int(len(top_errors)),
        "match_top10pct_conf_pck_3px": float((top_errors <= 3).mean())
        if len(top_errors) else None,
        "match_remaining90pct_conf_pck_3px": float((remaining_errors <= 3).mean())
        if len(remaining_errors) else None,
        "match_ransac_inlier_gt_valid": int(len(fit_inlier_errors)),
        "match_ransac_inlier_pck_3px": float((fit_inlier_errors <= 3).mean())
        if len(fit_inlier_errors) else None,
        "match_ransac_outlier_pck_3px": float((fit_outlier_errors <= 3).mean())
        if len(fit_outlier_errors) else None,
    }
    report.update({f"match_pck_{threshold}px": float((all_errors <= threshold).mean())
                   if len(all_errors) else None for threshold in (1, 3, 5)})
    report.update({f"pck_{threshold}px": count / total_valid
                   for threshold, count in pixel_hits.items()})
    report["beats_zero_flow"] = report["relative_epe"] < 1.0
    report["fusion_ready"] = (successes == samples and report["epe_px"] <= 2.0
                              and report["pck_3px"] >= 0.90)
    if reference_model is not None:
        for (stage, threshold), pieces in agreement_errors.items():
            selected_errors = np.concatenate(pieces) if pieces else np.empty(0)
            name = f"agreement_{stage}_{threshold}px"
            report[f"{name}_matches"] = agreement_counts[(stage, threshold)]
            report[f"{name}_gt_valid"] = int(len(selected_errors))
            report[f"{name}_match_pck_3px"] = (
                float((selected_errors <= 3).mean()) if len(selected_errors) else None)
        for (stage, gate), pieces in combined_errors.items():
            selected_errors = np.concatenate(pieces) if pieces else np.empty(0)
            name = f"agreement_{stage}_5px_{gate}"
            report[f"{name}_matches"] = combined_counts[(stage, gate)]
            report[f"{name}_gt_valid"] = int(len(selected_errors))
            report[f"{name}_match_pck_3px"] = (
                float((selected_errors <= 3).mean()) if len(selected_errors) else None)
    return report
