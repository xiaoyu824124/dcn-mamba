"""Prediction-centred 1/4-scale matching after the global coarse stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matching import correspondence_targets


@dataclass
class LocalMatchOutput:
    """All flows are [dy, dx] in 1/4-feature pixels."""

    coarse_flow: torch.Tensor
    matched_residual: torch.Tensor
    residual_flow: torch.Tensor
    refined_flow: torch.Tensor
    probability: torch.Tensor  # [B,K,H4,W4]
    log_probability: torch.Tensor
    valid_candidates: torch.Tensor
    offsets_yx: torch.Tensor  # [K,2]


class LocalMatcher(nn.Module):
    """Search IR features around each coarse VI-to-IR correspondence.

    Each candidate samples the moving feature at ``p + coarse(p) + offset``.
    Sampling at that coordinate, instead of shifting an already warped feature,
    keeps the search centred correctly when the coarse field varies spatially.
    Candidates are processed in chunks to limit the full-frame memory peak.
    """

    def __init__(self, radius: int = 4, temperature: float = 0.07,
                 candidate_chunk: int = 9, refinement_channels: int = 32) -> None:
        super().__init__()
        if radius < 1 or temperature <= 0 or candidate_chunk < 1 or refinement_channels < 1:
            raise ValueError("radius, temperature, candidate_chunk and refinement_channels must be positive")
        self.radius = int(radius)
        self.temperature = float(temperature)
        self.candidate_chunk = int(candidate_chunk)
        steps = torch.arange(-radius, radius + 1)
        yy, xx = torch.meshgrid(steps, steps, indexing="ij")
        self.register_buffer("offsets_yx", torch.stack((yy, xx), dim=-1).reshape(-1, 2),
                             persistent=False)
        candidates = (2 * radius + 1) ** 2
        self.refinement = nn.Sequential(
            nn.Conv2d(candidates + 4, refinement_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(refinement_channels, refinement_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(refinement_channels, 2, 3, padding=1),
        )
        # Begin at the coarse field. The head must earn every correction from
        # supervised flow rather than letting uncertain soft-argmax harm it.
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)

    def forward(self, feature_ir: torch.Tensor, feature_vi: torch.Tensor,
                coarse_flow: torch.Tensor) -> LocalMatchOutput:
        if feature_ir.ndim != 4 or feature_ir.shape != feature_vi.shape:
            raise ValueError("IR and VI 1/4 features must have equal [B,C,H,W] shapes")
        batch, _, height, width = feature_ir.shape
        if coarse_flow.shape != (batch, 2, height, width):
            raise ValueError("coarse_flow must be [B,2,H,W] in 1/4-feature pixels")
        # FP32 similarity and softmax remain stable inside mixed-precision training.
        moving = F.normalize(feature_ir.float(), dim=1)
        reference = F.normalize(feature_vi.float(), dim=1)
        y = torch.arange(height, device=feature_ir.device, dtype=torch.float32)
        x = torch.arange(width, device=feature_ir.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        centre = torch.stack((yy, xx), dim=-1).unsqueeze(0)
        centre = centre + coarse_flow.float().permute(0, 2, 3, 1)
        offsets = self.offsets_yx.to(device=feature_ir.device, dtype=torch.float32)
        scores, valid_parts = [], []
        for start in range(0, len(offsets), self.candidate_chunk):
            part = offsets[start:start + self.candidate_chunk]
            locations = centre.unsqueeze(3) + part.view(1, 1, 1, -1, 2)
            valid = ((locations[..., 0] >= 0) & (locations[..., 0] <= height - 1)
                     & (locations[..., 1] >= 0) & (locations[..., 1] <= width - 1))
            grid = torch.stack((2 * locations[..., 1] / max(width - 1, 1) - 1,
                                2 * locations[..., 0] / max(height - 1, 1) - 1), dim=-1)
            count = len(part)
            samples = F.grid_sample(moving, grid.reshape(batch, height, width * count, 2),
                                    mode="bilinear", padding_mode="zeros", align_corners=True)
            samples = samples.reshape(batch, moving.shape[1], height, width, count)
            score = (reference.unsqueeze(-1) * samples).sum(dim=1)
            scores.append(score)
            valid_parts.append(valid)
        logits = torch.cat(scores, dim=-1).permute(0, 3, 1, 2) / self.temperature
        valid_candidates = torch.cat(valid_parts, dim=-1).permute(0, 3, 1, 2)
        # If all candidates leave the image, a uniform distribution has zero
        # mean residual because the symmetric offset set sums to zero.
        logits = logits.masked_fill(~valid_candidates, -1.0e4)
        log_probability = F.log_softmax(logits, dim=1)
        probability = log_probability.exp()
        matched_residual = torch.einsum("bkhw,kc->bchw", probability, offsets)
        head_input = torch.cat((probability, coarse_flow.float(), matched_residual), dim=1)
        residual = self.radius * torch.tanh(self.refinement(head_input) / self.radius)
        refined = coarse_flow.float() + residual
        return LocalMatchOutput(coarse_flow.float(), matched_residual, residual, refined, probability,
                                log_probability, valid_candidates, offsets)


def local_matching_loss(match: LocalMatchOutput, gt_flow: torch.Tensor,
                        valid_mask: torch.Tensor | None = None,
                        sigma: float = 0.75) -> torch.Tensor:
    """Soft GT correspondence NLL for queries covered by the local window."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    height, width = match.coarse_flow.shape[-2:]
    target, query_mask = correspondence_targets(gt_flow, (height, width), valid_mask)
    target = target.transpose(1, 2).reshape(gt_flow.shape[0], 2, height, width)
    y = torch.arange(height, device=gt_flow.device, dtype=torch.float32)
    x = torch.arange(width, device=gt_flow.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    base = torch.stack((yy, xx), dim=0).unsqueeze(0)
    target_residual = target.float() - base - match.coarse_flow.float()
    radius = match.offsets_yx.abs().max()
    covered = (target_residual.abs() <= radius + 0.5).all(dim=1)
    distance = (target_residual.unsqueeze(1)
                - match.offsets_yx.to(target_residual).view(1, -1, 2, 1, 1)).square().sum(dim=2)
    weights = torch.exp(-distance / (2 * sigma * sigma)) * match.valid_candidates
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    per_query = -(weights * match.log_probability).sum(dim=1)
    mask = query_mask.reshape(gt_flow.shape[0], height, width) & covered
    return (per_query * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def local_matching_diagnostics(match: LocalMatchOutput, gt_flow: torch.Tensor,
                               valid_mask: torch.Tensor | None = None) -> dict[str, float]:
    """Separate local search capacity, feature matching, and learned correction."""
    height, width = match.coarse_flow.shape[-2:]
    target, query_mask = correspondence_targets(gt_flow, (height, width), valid_mask)
    target = target.transpose(1, 2).reshape(gt_flow.shape[0], 2, height, width)
    y = torch.arange(height, device=gt_flow.device, dtype=torch.float32)
    x = torch.arange(width, device=gt_flow.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    centre = torch.stack((yy, xx), dim=0).unsqueeze(0) + match.coarse_flow.float()
    radius = match.offsets_yx.abs().max()
    covered = ((target - centre).abs() <= radius + 0.5).all(dim=1)
    mask = query_mask.reshape(gt_flow.shape[0], height, width)
    winning_offset = match.offsets_yx[match.probability.argmax(dim=1)]
    prediction = centre + winning_offset.permute(0, 3, 1, 2)
    stride = gt_flow.new_tensor((gt_flow.shape[-2] / height,
                                gt_flow.shape[-1] / width)).view(1, 2, 1, 1)
    error = torch.linalg.vector_norm((prediction - target) * stride, dim=1)
    coarse_error = torch.linalg.vector_norm((centre - target) * stride, dim=1)
    soft_prediction = centre + match.matched_residual.float()
    soft_error = torch.linalg.vector_norm((soft_prediction - target) * stride, dim=1)
    refined_prediction = centre + match.residual_flow.float()
    refined_error = torch.linalg.vector_norm((refined_prediction - target) * stride, dim=1)
    candidate_positions = (centre.unsqueeze(1)
                           + match.offsets_yx.to(centre).view(1, -1, 2, 1, 1))
    candidate_error = torch.linalg.vector_norm(
        (candidate_positions - target.unsqueeze(1)) * stride.unsqueeze(1), dim=2)
    oracle_error = candidate_error.masked_fill(~match.valid_candidates, float("inf")).amin(dim=1)
    oracle_mask = mask & covered & match.valid_candidates.any(dim=1)
    valid_count = mask.sum().clamp_min(1)
    return {
        "local_window_coverage": float((covered & mask).sum() / valid_count),
        "local_argmax_epe_px": float((error * mask).sum() / valid_count),
        "local_oracle_epe_px": float(torch.where(oracle_mask, oracle_error, 0).sum()
                                      / oracle_mask.sum().clamp_min(1)),
        "local_soft_epe_px": float((soft_error * mask).sum() / valid_count),
        "local_gt_residual_px": float((coarse_error * mask).sum() / valid_count),
        "local_pred_residual_px": float((torch.linalg.vector_norm(
            match.residual_flow.float() * stride, dim=1) * mask).sum() / valid_count),
        "local_refinement_improved_fraction": float(
            ((refined_error < coarse_error) & mask).sum() / valid_count),
    }
