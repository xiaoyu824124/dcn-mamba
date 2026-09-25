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
from .matching import appearance_window_loss, coarse_matching_loss
from .local_matcher import LocalMatchOutput, local_matching_loss


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
    local: torch.Tensor
    appearance: torch.Tensor
    coarse_flow: torch.Tensor | None = None
    global_flow: torch.Tensor | None = None
    coarse_local: torch.Tensor | None = None

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {"loss": self.total, "loss_flow": self.flow, "loss_mind": self.mind,
                "loss_edge": self.edge, "loss_smooth": self.smooth,
                "loss_affine": self.affine, "loss_match": self.match,
                "loss_local": self.local, "loss_appearance": self.appearance,
                "loss_coarse_flow": self.coarse_flow,
                "loss_global_flow": self.global_flow,
                "loss_coarse_local": self.coarse_local}


class RegistrationLoss(nn.Module):
    """Supervised flow plus modality-robust structural registration losses.

    Args:
        weights: Mapping with ``flow``, ``match``, ``mind``, ``edge``,
            ``smooth`` and ``affine`` keys. ``local`` is optional for old
            coarse-only checkpoints.
            Values are intentionally supplied by YAML rather than hard-coded.
        mind_descriptor: The shared MIND implementation used for structural loss.
        charbonnier_eps: Robust L1 epsilon for supervised flow.

    ``forward`` expects an aligned IR image, a fixed visible RGB image, the
    coarse flow, optional final flow and optional GT flow. The final field is
    supervised when available; otherwise the coarse field is supervised.

    ``match`` carries the matcher output so the coarse correspondence can be
    supervised directly.  Without it the dense flow term has a degenerate
    optimum (a constant field already reaches the mean-displacement error) and
    the encoder is never asked to make the ground-truth key the argmax.
    ``local_match`` similarly supervises the 1/4 candidate distribution.
    """

    REQUIRED_WEIGHTS = ("flow", "match", "mind", "edge", "smooth", "affine")

    def __init__(self, weights: Dict[str, float],
                 mind_descriptor: Optional[MINDDescriptor] = None,
                 charbonnier_eps: float = 1e-3,
                 match_focal_gamma: float = 0.0,
                 appearance_window_radius: int = 4,
                 appearance_temperature: float = 0.07) -> None:
        super().__init__()
        missing = set(self.REQUIRED_WEIGHTS).difference(weights)
        if missing:
            raise ValueError(f"registration loss weights missing: {sorted(missing)}")
        self.weights = {key: float(weights[key]) for key in self.REQUIRED_WEIGHTS}
        self.weights["local"] = float(weights.get("local", 0.0))
        self.weights["appearance"] = float(weights.get("appearance", 0.0))
        for name in ("coarse_flow", "global_flow", "coarse_local"):
            self.weights[name] = float(weights.get(name, 0.0))
        if self.weights["appearance"] < 0:
            raise ValueError("appearance weight must be non-negative")
        if float(match_focal_gamma) < 0:
            raise ValueError("match_focal_gamma must be non-negative")
        self.match_focal_gamma = float(match_focal_gamma)
        if appearance_window_radius < 1 or appearance_temperature <= 0:
            raise ValueError("appearance window radius and temperature must be positive")
        self.appearance_window_radius = int(appearance_window_radius)
        self.appearance_temperature = float(appearance_temperature)
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
                global_flow: Optional[torch.Tensor] = None,
                gt_flow: Optional[torch.Tensor] = None,
                valid_mask: Optional[torch.Tensor] = None,
                predicted_affine_yx: Optional[torch.Tensor] = None,
                affine_feature_hw: Optional[tuple[int, int]] = None,
                gt_h: Optional[torch.Tensor] = None,
                match: Optional[GlobalMatchOutput] = None,
                local_match: Optional[LocalMatchOutput] = None,
                coarse_local_match: Optional[LocalMatchOutput] = None) -> RegistrationLossOutput:
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

        def flow_loss(predicted: torch.Tensor) -> torch.Tensor:
            if gt_flow is None:
                return zero
            if gt_flow.shape != predicted.shape:
                raise ValueError("gt_flow must match predicted flow shape")
            flow_error = torch.sqrt((predicted - gt_flow.to(predicted)).square()
                                    + self.charbonnier_eps * self.charbonnier_eps)
            if valid_mask is None:
                return flow_error.mean()
            return (flow_error * valid_mask).sum() / (
                valid_mask.sum().clamp_min(1) * predicted.shape[1])

        loss_flow = flow_loss(active_flow)
        loss_coarse_flow = (flow_loss(coarse_flow)
                            if self.weights["coarse_flow"] > 0 else zero)
        loss_global_flow = (flow_loss(global_flow)
                            if self.weights["global_flow"] > 0 and global_flow is not None
                            else zero)

        if predicted_affine_yx is None and gt_h is None:
            loss_affine = zero
        elif predicted_affine_yx is None or gt_h is None:
            raise ValueError("predicted_affine_yx and gt_h must be supplied together")
        else:
            feature_hw = (affine_feature_hw if affine_feature_hw is not None else
                          (active_flow.shape[-2] // 8, active_flow.shape[-1] // 8))
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
        loss_local = (local_matching_loss(local_match, gt_flow, valid_mask)
                      if local_match is not None and gt_flow is not None else zero)
        loss_coarse_local = (
            local_matching_loss(coarse_local_match, gt_flow, valid_mask)
            if (self.weights["coarse_local"] > 0 and coarse_local_match is not None
                and gt_flow is not None) else zero)
        loss_appearance = zero
        if self.weights["appearance"] > 0 and match is not None and gt_flow is not None:
            if match.correlation is None:
                raise ValueError("appearance loss requires global_matcher.return_correlation=true")
            loss_appearance = appearance_window_loss(
                match.correlation, tuple(match.coarse_flow.shape[-2:]), gt_flow,
                valid_mask, radius=self.appearance_window_radius,
                temperature=self.appearance_temperature)

        # MIND compares self-similarity patterns, not raw IR/RGB intensities.
        loss_mind = (F.l1_loss(self.mind_descriptor(aligned_ir),
                               self.mind_descriptor(rgb_to_gray(visible)))
                     if self.weights["mind"] > 0 else zero)
        loss_edge = F.l1_loss(self.edge_magnitude(aligned_ir),
                              self.edge_magnitude(visible))
        loss_smooth = self.edge_aware_smoothness(active_flow, visible)
        total = (self.weights["flow"] * loss_flow
                 + self.weights["coarse_flow"] * loss_coarse_flow
                 + self.weights["global_flow"] * loss_global_flow
                 + self.weights["match"] * loss_match
                 + self.weights["appearance"] * loss_appearance
                 + self.weights["local"] * loss_local
                 + self.weights["coarse_local"] * loss_coarse_local
                 + self.weights["mind"] * loss_mind
                 + self.weights["edge"] * loss_edge
                 + self.weights["smooth"] * loss_smooth
                 + self.weights["affine"] * loss_affine)
        return RegistrationLossOutput(total, loss_flow, loss_mind, loss_edge, loss_smooth,
                                      loss_affine, loss_match, loss_local, loss_appearance,
                                      loss_coarse_flow, loss_global_flow,
                                      loss_coarse_local)
