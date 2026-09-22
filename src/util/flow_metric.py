# -*- coding: utf-8 -*-
"""Single definition of the registration error, shared by every entry point.

Before this module existed, three places measured "EPE" three different ways:

  * ``(pred - gt).abs().mean()``  -- a COMPONENT-WISE L1, not the standard
    end-point error, which is the mean per-pixel L2 norm ``mean ||dv||_2``.
    The two differ by up to sqrt(2) when the error is anisotropic.
  * once over the whole batch tensor, once per sample and then averaged;
  * over a random 208 crop (training) but over the full 480x640 frame
    (evaluation).  A ratio measured on a crop is NOT comparable with the same
    ratio measured on the full frame, which caused a wrong conclusion earlier
    in this project when a training ratio (~1.1) was compared against an
    evaluation ratio (0.89).

Everything here therefore uses one definition and reports the field of view
explicitly:

    epe_px      mean over pixels of ||pred - gt||_2, in pixels
    baseline_px mean over pixels of ||gt||_2, i.e. the zero-flow error
    epe_ratio   epe_px / baseline_px; 1.0 means "no better than predicting
                no motion", < 1 means the flow actually helps

Flow convention everywhere is ``[dy, dx]`` with
``warped(p) = source(p + flow(p))``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

__all__ = ["flow_epe", "flow_epe_dict", "centre_crop_flow"]

_EPS = 1e-6


def centre_crop_flow(flow: torch.Tensor, crop: int) -> torch.Tensor:
    """Centre-crop a ``[B, T, 2, H, W]`` flow field to ``crop x crop``."""
    if not crop or crop <= 0:
        return flow
    height, width = flow.shape[-2:]
    crop_h, crop_w = min(crop, height), min(crop, width)
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return flow[..., top:top + crop_h, left:left + crop_w]


def flow_epe(pred: torch.Tensor, gt: torch.Tensor, crop: int = 0,
             per_sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard end-point error and the zero-flow baseline, in pixels.

    ``pred``/``gt`` are ``[B, T, 2, H, W]`` (or ``[B, 2, H, W]``).  Returns
    ``(epe_px, baseline_px)`` either per sample (``[B]``) or as scalars.
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"flow_epe expects matching shapes, got {tuple(pred.shape)} and "
            f"{tuple(gt.shape)}")
    pred = pred.float()
    gt = gt.float()
    if pred.dim() not in (4, 5):
        raise ValueError(
            "expected a 4D [B,2,H,W] or 5D [B,T,2,H,W] flow field, got "
            f"{pred.dim()}D")
    if crop:
        pred = centre_crop_flow(pred, crop)
        gt = centre_crop_flow(gt, crop)

    # The flow-component axis is -3 in BOTH layouts: [B, T, 2, H, W] and
    # [B, 2, H, W].  Using dim=1 (as an earlier version did) silently norms over
    # the TIME axis for 5D input, which is a completely different quantity.
    error = (pred - gt).norm(dim=-3)            # 5D -> [B,T,H,W], 4D -> [B,H,W]
    baseline = gt.norm(dim=-3)
    # Derived from the rank rather than hard-coded: the previous version assumed
    # the 5D rank and crashed on 4D input (mean over dim 3 of a 3D tensor).
    spatial_dims = tuple(range(1, error.dim()))
    if per_sample:
        return error.mean(dim=spatial_dims), baseline.mean(dim=spatial_dims)
    return error.mean(), baseline.mean()


def flow_epe_dict(pred: torch.Tensor, gt: torch.Tensor,
                  crop: int = 0) -> Dict[str, float]:
    """``epe_px`` / ``baseline_px`` / ``epe_ratio`` averaged over the batch."""
    epe, baseline = flow_epe(pred, gt, crop=crop, per_sample=True)
    ratio = epe / baseline.clamp_min(_EPS)
    return {
        "epe_px": float(epe.mean()),
        "baseline_px": float(baseline.mean()),
        "epe_ratio": float(ratio.mean()),
    }
