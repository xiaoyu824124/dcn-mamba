"""Losses for the standalone single-frame IR--visible registration branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mind import MINDDescriptor, rgb_to_gray
from .affine import homography_to_normalised_affine_yx
from .global_matcher import GlobalMatchOutput
from .matching import coarse_matching_loss


@dataclass
class RegistrationLossOutput:
    """Named scalar losses returned by :class:`RegistrationLoss`."""

    total: torch.Tensor
    flow: torch.Tensor
    mind: torch.Tensor
    edge: torch.Tensor
    smooth: torch.Tensor
    affine: torch.Tensor
    match: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {"loss": self.total, "loss_flow": self.flow, "loss_mind": self.mind,
                "loss_edge": self.edge, "loss_smooth": self.smooth,
                "loss_affine": self.affine, "loss_match": self.match}


class RegistrationLoss(nn.Module):
    """Supervised flow plus modality-robust structural registration losses.

    Args:
        weights: Mapping with ``flow``, ``mind``, ``edge``, ``smooth`` and
            ``affine`` keys.
            Values are intentionally supplied by YAML rather than hard-coded.
        mind_descriptor: The shared MIND implementation used for structural loss.
        charbonnier_eps: Robust L1 epsilon for supervised flow.

    ``forward`` expects an aligned IR image, a fixed visible RGB image, the
    coarse flow, optional final flow and optional GT flow.  When a DCN run uses
    feature reconstruction only, it has no final explicit field: flow loss then
    supervises the coarse global field while structural losses supervise the
    refined reconstruction branch.

    ``match`` carries the matcher output so the coarse correspondence can be
    supervised directly.  Without it the dense flow term has a degenerate
    optimum (a constant field already reaches the mean-displacement error) and
    the encoder is never asked to make the ground-truth key the argmax.
    """

    REQUIRED_WEIGHTS = ("flow", "match", "mind", "edge", "smooth", "affine")

    def __init__(self, weights: Dict[str, float],
                 mind_descriptor: Optional[MINDDescriptor] = None,
                 charbonnier_eps: float = 1e-3,
                 match_focal_gamma: float = 0.0) -> None:
        super().__init__()
        missing = set(self.REQUIRED_WEIGHTS).difference(weights)
        if missing:
            raise ValueError(f"registration loss weights missing: {sorted(missing)}")
        self.weights = {key: float(weights[key]) for key in self.REQUIRED_WEIGHTS}
        if float(match_focal_gamma) < 0:
            raise ValueError("match_focal_gamma must be non-negative")
        self.match_focal_gamma = float(match_focal_gamma)
        self.mind_descriptor = (mind_descriptor if mind_descriptor is not None
                                else MINDDescriptor())
        self.charbonnier_eps = float(charbonnier_eps)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_x.t().reshape(1, 1, 3, 3))

    @staticmethod
    def charbonnier(value: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sqrt(value.square() + eps * eps).mean()

    def edge_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        """Contrast-reversal-invariant gradient magnitude of a gray image."""
        gray = rgb_to_gray(image)
        gx = F.conv2d(gray, self.sobel_x.to(dtype=gray.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(dtype=gray.dtype), padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)

    def edge_aware_smoothness(self, flow: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        """First-order flow regularity, relaxed across visible-image boundaries."""
        gray = rgb_to_gray(visible)
        edge_x = (gray[..., 1:] - gray[..., :-1]).abs()
        edge_y = (gray[..., 1:, :] - gray[..., :-1, :]).abs()
        flow_x = (flow[..., 1:] - flow[..., :-1]).abs()
        flow_y = (flow[..., 1:, :] - flow[..., :-1, :]).abs()
        return ((flow_x * torch.exp(-edge_x)).mean()
                + (flow_y * torch.exp(-edge_y)).mean())

    def forward(self, *, aligned_ir: torch.Tensor, visible: torch.Tensor,
                coarse_flow: torch.Tensor, final_flow: Optional[torch.Tensor] = None,
                gt_flow: Optional[torch.Tensor] = None,
                valid_mask: Optional[torch.Tensor] = None,
                predicted_affine_yx: Optional[torch.Tensor] = None,
                gt_h: Optional[torch.Tensor] = None,
                match: Optional[GlobalMatchOutput] = None) -> RegistrationLossOutput:
        if aligned_ir.ndim != 4 or aligned_ir.shape[1] != 1:
            raise ValueError("aligned_ir must be [B,1,H,W]")
        if visible.ndim != 4 or visible.shape[1] != 3:
            raise ValueError("visible must be [B,3,H,W]")
        if aligned_ir.shape[0] != visible.shape[0] or aligned_ir.shape[-2:] != visible.shape[-2:]:
            raise ValueError("aligned_ir and visible must share B,H,W")
        active_flow = final_flow if final_flow is not None else coarse_flow
        if active_flow.shape != (aligned_ir.shape[0], 2, *aligned_ir.shape[-2:]):
            raise ValueError("active flow must be [B,2,H,W] on the aligned image grid")

        zero = aligned_ir.new_zeros(())
        if valid_mask is not None:
            if valid_mask.shape != (aligned_ir.shape[0], 1, *aligned_ir.shape[-2:]):
                raise ValueError("valid_mask must be [B,1,H,W] on the aligned image grid")
            valid_mask = valid_mask.to(dtype=active_flow.dtype).clamp(0, 1)

        if gt_flow is None:
            loss_flow = zero
        else:
            if gt_flow.shape != active_flow.shape:
                raise ValueError("gt_flow must match active flow shape")
            flow_error = torch.sqrt((active_flow - gt_flow.to(active_flow)).square()
                                    + self.charbonnier_eps * self.charbonnier_eps)
            if valid_mask is None:
                loss_flow = flow_error.mean()
            else:
                loss_flow = (flow_error * valid_mask).sum() / (
                    valid_mask.sum().clamp_min(1) * active_flow.shape[1])

        if predicted_affine_yx is None and gt_h is None:
            loss_affine = zero
        elif predicted_affine_yx is None or gt_h is None:
            raise ValueError("predicted_affine_yx and gt_h must be supplied together")
        else:
            feature_hw = (active_flow.shape[-2] // 8, active_flow.shape[-1] // 8)
            if predicted_affine_yx.shape != (active_flow.shape[0], 3, 2):
                raise ValueError("predicted_affine_yx must be [B,3,2]")
            target_affine_yx = homography_to_normalised_affine_yx(
                gt_h.float(), active_flow.shape[-2:], feature_hw)
            # The 3x3 WLS solve is deliberately FP32; retain that precision for
            # its direct supervision even when the surrounding forward runs AMP.
            loss_affine = self.charbonnier(predicted_affine_yx.float() - target_affine_yx,
                                            self.charbonnier_eps)

        # Direct correspondence supervision.  ``match.coarse_flow`` is the
        # matcher's own feature grid, which is the only place the target cells
        # can be resolved from; the image-grid ``active_flow`` would not.
        if match is None or gt_flow is None:
            loss_match = zero
        else:
            match_hw = tuple(match.coarse_flow.shape[-2:])
            loss_match = coarse_matching_loss(
                match.matching_probability, match_hw, gt_flow, valid_mask,
                focal_gamma=self.match_focal_gamma)

        # MIND compares self-similarity patterns, not raw IR/RGB intensities.
        loss_mind = F.l1_loss(self.mind_descriptor(aligned_ir),
                              self.mind_descriptor(rgb_to_gray(visible)))
        loss_edge = F.l1_loss(self.edge_magnitude(aligned_ir),
                              self.edge_magnitude(visible))
        loss_smooth = self.edge_aware_smoothness(active_flow, visible)
        total = (self.weights["flow"] * loss_flow
                 + self.weights["match"] * loss_match
                 + self.weights["mind"] * loss_mind
                 + self.weights["edge"] * loss_edge
                 + self.weights["smooth"] * loss_smooth
                 + self.weights["affine"] * loss_affine)
        return RegistrationLossOutput(total, loss_flow, loss_mind, loss_edge, loss_smooth,
                                      loss_affine, loss_match)
