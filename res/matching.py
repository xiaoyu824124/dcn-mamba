"""Coarse correspondence supervision and localisation diagnostics.

The registration branch is only as good as its correspondence distribution: if
the ground-truth match is not near the mode of the softmax over keys, no
downstream affine fit or flow refinement can recover the alignment.  The dense
flow loss alone cannot see that failure -- a constant field already reaches the
mean-displacement error -- so this module provides

* :func:`coarse_matching_loss`, a dual-softmax negative log-likelihood applied
  directly to the matcher's probability matrix at the ground-truth cell, which
  is the only term that rewards putting mass on the correct key; and
* :func:`matching_diagnostics`, which reports how the ground-truth match ranks
  against the competing keys.

Both consume tensors the matcher already produces, so neither adds another
:math:`N \\times N` correlation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def grid_coordinates(height: int, width: int,
                     reference: torch.Tensor) -> torch.Tensor:
    """Feature-grid coordinates ``[N,2]`` in ``[y,x]`` order.

    The ordering matches ``GlobalMatcher._coordinates`` so diagnostics and the
    matcher always describe the same query/key index space.
    """
    y = torch.arange(height, device=reference.device, dtype=reference.dtype)
    x = torch.arange(width, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx), dim=-1).reshape(-1, 2)


def correspondence_targets(flow: torch.Tensor, feature_hw: Tuple[int, int],
                           valid_mask: torch.Tensor | None = None
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map image-grid ``[dy,dx]`` ground-truth flow to feature-grid key cells.

    Args:
        flow: ``[B,2,H,W]`` ground-truth flow defined on the reference grid.
        feature_hw: Matcher grid ``(Hf, Wf)``; ``H``/``W`` must be multiples.
        valid_mask: Optional ``[B,1,H,W]`` mask of usable reference pixels.

    Returns:
        ``(target_yx, query_mask)`` with ``target_yx`` in **feature-grid
        pixels** (``[y,x]``) and ``query_mask`` a boolean ``[B,N]`` tensor
        selecting queries whose target is inside the key grid.

    The flow is resampled with the same ``align_corners=False`` interpolation
    used by :func:`res.warp.upsample_feature_flow`, so the training target and
    the lifted inference flow stay on one convention.
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
    feature_h, feature_w = int(feature_hw[0]), int(feature_hw[1])
    if feature_h < 1 or feature_w < 1:
        raise ValueError(f"invalid feature grid {feature_hw}")
    batch, _, height, width = flow.shape
    stride_y, stride_x = height / feature_h, width / feature_w

    subsampled = F.interpolate(flow, size=(feature_h, feature_w),
                               mode="bilinear", align_corners=False)
    stride = flow.new_tensor((stride_y, stride_x)).view(1, 2, 1, 1)
    subsampled = subsampled / stride

    coordinates = grid_coordinates(feature_h, feature_w, flow)
    reference = coordinates.view(1, -1, 2)
    target = reference + subsampled.flatten(2).transpose(1, 2)

    if valid_mask is None:
        query_mask = torch.ones(batch, feature_h * feature_w,
                                dtype=torch.bool, device=flow.device)
    else:
        if valid_mask.shape != (batch, 1, height, width):
            raise ValueError("valid_mask must be [B,1,H,W] on the flow grid")
        resampled = F.interpolate(valid_mask.float(), size=(feature_h, feature_w),
                                  mode="nearest")
        query_mask = resampled.flatten(1).gt(0.5)

    inside = ((target[..., 0] >= 0) & (target[..., 0] <= feature_h - 1)
              & (target[..., 1] >= 0) & (target[..., 1] <= feature_w - 1))
    return target, query_mask & inside


def bilinear_target_cells(target_yx: torch.Tensor, feature_hw: Tuple[int, int]
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split sub-pixel targets into ``[B,N,4]`` flat cell indices and weights.

    Targets outside the grid are clamped, so the four weights always sum to one
    and duplicate indices simply accumulate on the border cell.
    """
    feature_h, feature_w = int(feature_hw[0]), int(feature_hw[1])
    y = target_yx[..., 0].clamp(0.0, feature_h - 1)
    x = target_yx[..., 1].clamp(0.0, feature_w - 1)
    y0, x0 = torch.floor(y), torch.floor(x)
    y1 = (y0 + 1).clamp(max=feature_h - 1)
    x1 = (x0 + 1).clamp(max=feature_w - 1)
    wy, wx = y - y0, x - x0
    indices = torch.stack((y0 * feature_w + x0, y0 * feature_w + x1,
                           y1 * feature_w + x0, y1 * feature_w + x1), dim=-1)
    weights = torch.stack(((1 - wy) * (1 - wx), (1 - wy) * wx,
                           wy * (1 - wx), wy * wx), dim=-1)
    return indices.long(), weights


def coarse_matching_loss(probability: torch.Tensor, feature_hw: Tuple[int, int],
                         flow: torch.Tensor, valid_mask: torch.Tensor | None = None,
                         focal_gamma: float = 0.0) -> torch.Tensor:
    """Dual-softmax NLL of the ground-truth correspondence.

    Args:
        probability: Matcher softmax ``[B,N,N]`` (already temperature scaled).
        feature_hw: Matcher grid ``(Hf, Wf)``.
        flow: Ground-truth ``[B,2,H,W]`` flow on the reference grid.
        valid_mask: Optional ``[B,1,H,W]`` mask of usable reference pixels.
        focal_gamma: ``0`` gives a plain NLL; ``>0`` down-weights queries whose
            target probability is already high.

    Returns:
        Scalar loss, averaged over queries with an in-bounds target.
    """
    if probability.ndim != 3 or probability.shape[-1] != probability.shape[-2]:
        raise ValueError("probability must be a square [B,N,N] tensor")
    target, query_mask = correspondence_targets(flow, feature_hw, valid_mask)
    if target.shape[1] != probability.shape[-1]:
        raise ValueError(
            f"probability has N={probability.shape[-1]} keys but the grid "
            f"{tuple(feature_hw)} has {target.shape[1]} cells")
    indices, weights = bilinear_target_cells(target, feature_hw)
    dense = probability if probability.dtype == torch.float32 else probability.float()
    cells = dense.gather(2, indices)
    log_probability = cells.clamp_min(1e-12).log()
    if focal_gamma > 0:
        modulator = (1.0 - cells).clamp_min(0.0) ** float(focal_gamma)
        per_query = -(modulator * weights * log_probability).sum(dim=-1)
    else:
        per_query = -(weights * log_probability).sum(dim=-1)
    mask = query_mask.to(per_query.dtype)
    return (per_query * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def windowed_diagnostics(probability: torch.Tensor, feature_hw: Tuple[int, int],
                         predicted_feature_flow: torch.Tensor, flow: torch.Tensor,
                         valid_mask: torch.Tensor | None = None,
                         radius: int = 2) -> Dict[str, float]:
    """Would a local window around the *predicted* key recover the true match?

    ``predicted_feature_flow`` is the matcher's own feature-grid ``[dy,dx]``
    field (``GlobalMatchOutput.coarse_flow``).  A ``(2*radius+1)^2`` cell window
    is centred on each predicted key and the ground-truth key is ranked inside
    it -- the quantity a sequential local/fine matcher would have to solve.  The
    global all-pairs matcher competes against every key, so its argmax can be
    hopeless while the answer inside a small window is easy.
    """
    if radius < 1:
        raise ValueError("radius must be at least one")
    if tuple(predicted_feature_flow.shape[-2:]) != (int(feature_hw[0]), int(feature_hw[1])):
        raise ValueError("predicted_feature_flow must live on the feature grid")

    target, query_mask = correspondence_targets(flow, feature_hw, valid_mask)
    predicted, _ = correspondence_targets(predicted_feature_flow, feature_hw, None)
    dense = probability.detach().float()
    height, width = int(feature_hw[0]), int(feature_hw[1])
    batch = dense.shape[0]
    channels = (2 * radius + 1) ** 2

    offsets = torch.arange(-radius, radius + 1, device=dense.device, dtype=dense.dtype)
    grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
    offsets = torch.stack((grid_y.reshape(-1), grid_x.reshape(-1)), dim=-1)
    window = predicted.unsqueeze(2) + offsets.view(1, 1, -1, 2)
    inside = ((window[..., 0] >= 0) & (window[..., 0] <= height - 1)
              & (window[..., 1] >= 0) & (window[..., 1] <= width - 1))
    flat = (window[..., 0].clamp(0, height - 1) * width
            + window[..., 1].clamp(0, width - 1)).long()
    window_probability = dense.gather(2, flat)
    window_probability = window_probability.masked_fill(~inside, -1.0)

    indices, weights = bilinear_target_cells(target, feature_hw)
    gt_probability = (dense.gather(2, indices) * weights).sum(dim=-1)

    delta = (target - predicted).abs()
    covered = (delta[..., 0] <= radius + 0.5) & (delta[..., 1] <= radius + 0.5)
    active = covered & query_mask
    mask = active.to(dense.dtype)
    denominator = mask.sum().clamp_min(1.0)
    valid_count = query_mask.to(dense.dtype).sum().clamp_min(1.0)

    height_px, width_px = flow.shape[-2:]
    stride = flow.new_tensor((height_px / height, width_px / width)).to(dense.device)
    error_px = torch.linalg.vector_norm(
        (target - predicted) * stride, dim=-1)

    def masked(value: torch.Tensor) -> float:
        return float((value * mask).sum() / denominator)

    beating = (window_probability > gt_probability.unsqueeze(-1)).sum(dim=-1)
    return {
        # Coverage is over every valid query.  The other window statistics are
        # conditional on the truth being inside the window.
        f"window{radius}_coverage": float(mask.sum() / valid_count),
        f"window{radius}_frac_cells_beating_gt": masked(beating.to(dense.dtype) / channels),
        f"window{radius}_argmax_correct": masked((beating == 0).to(dense.dtype)),
        "coarse_error_median_px": float(
            torch.quantile(error_px[query_mask].flatten().float(), 0.5)) if bool(query_mask.any()) else float("nan"),
        "coarse_error_p90_px": float(
            torch.quantile(error_px[query_mask].flatten().float(), 0.9)) if bool(query_mask.any()) else float("nan"),
    }


@torch.no_grad()
def matching_diagnostics(probability: torch.Tensor, feature_hw: Tuple[int, int],
                         flow: torch.Tensor,
                         valid_mask: torch.Tensor | None = None,
                         appearance_scores: torch.Tensor | None = None) -> Dict[str, float]:
    """How the ground-truth match ranks against every competing key.

    ``match_frac_keys_beating_gt`` is a tie-aware rank fraction: ``0`` means
    the target leads, while a uniform distribution reports about ``0.5``.
    """
    target, query_mask = correspondence_targets(flow, feature_hw, valid_mask)
    dense = probability.detach().float()
    indices, weights = bilinear_target_cells(target, feature_hw)
    target_probability = (dense.gather(2, indices) * weights).sum(dim=-1)

    coordinates = grid_coordinates(int(feature_hw[0]), int(feature_hw[1]), dense)
    reference = coordinates.view(1, -1, 2)
    gt_flow = target - reference
    argmax = dense.argmax(dim=-1)
    argmax_flow = coordinates[argmax] - reference
    expected_flow = torch.einsum("bnk,kc->bnc", dense, coordinates) - reference

    height, width = flow.shape[-2:]
    stride = flow.new_tensor((height / feature_hw[0], width / feature_hw[1]))
    stride = stride.to(dense.device)

    top2 = dense.topk(2, dim=-1).values
    entropy = -(dense.clamp_min(1e-12).log() * dense).sum(dim=-1)
    mask = query_mask.to(dense.dtype)
    denominator = mask.sum().clamp_min(1.0)

    def masked(value: torch.Tensor) -> float:
        return float((value * mask).sum() / denominator)

    # A strict greater-than comparison calls a completely uniform matcher
    # perfect (zero keys beat the GT).  Assign half credit to ties, excluding
    # the target's own cell when its score is an exact cell probability.
    higher = (dense > target_probability.unsqueeze(-1)).to(dense.dtype).sum(dim=-1)
    tied = (dense == target_probability.unsqueeze(-1)).to(dense.dtype).sum(dim=-1)
    rank_fraction = ((higher + 0.5 * (tied - 1).clamp_min(0)) / dense.shape[-1]).clamp_max(1)

    report = {
        "match_frac_keys_beating_gt": masked(rank_fraction),
        "match_epe_argmax_px": masked(
            torch.linalg.vector_norm((argmax_flow - gt_flow) * stride, dim=-1)),
        "match_epe_softargmax_px": masked(
            torch.linalg.vector_norm((expected_flow - gt_flow) * stride, dim=-1)),
        "match_p_max": masked(top2[..., 0]),
        "match_top1_top2_logit_gap": masked(
            top2[..., 0].clamp_min(1e-12).log() - top2[..., 1].clamp_min(1e-12).log()),
        "match_effective_keys_ratio": masked(entropy.exp() / dense.shape[-1]),
    }
    if appearance_scores is not None:
        if appearance_scores.shape != probability.shape:
            raise ValueError("appearance_scores must match probability shape")
        appearance = appearance_scores.detach().float()
        gt_score = (appearance.gather(2, indices) * weights).sum(dim=-1)
        higher = (appearance > gt_score.unsqueeze(-1)).to(dense.dtype).sum(dim=-1)
        tied = (appearance == gt_score.unsqueeze(-1)).to(dense.dtype).sum(dim=-1)
        appearance_rank = ((higher + 0.5 * (tied - 1).clamp_min(0))
                           / appearance.shape[-1]).clamp_max(1)
        appearance_argmax = coordinates[appearance.argmax(dim=-1)] - reference
        report["appearance_frac_keys_beating_gt"] = masked(appearance_rank)
        report["appearance_epe_argmax_px"] = masked(
            torch.linalg.vector_norm((appearance_argmax - gt_flow) * stride, dim=-1))
    return report
