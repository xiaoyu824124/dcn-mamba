"""Iterative discrepancy-guided 1/4 refinement with supervised confidence.

One round does exactly this, and every round repeats it from the *original*
feature maps:

    u_t -> re-sample the moving IR feature at the current correspondence
        -> feature discrepancy + local correlation evidence + current flow
        -> shared update network -> residual delta
        -> confidence gate -> u_(t+1) = u_t + confidence * delta

Design decisions that matter, all of them deliberate:

* Sampling always reads ``feature_ir`` at ``p + u_t + offset``.  Warping a
  feature map once and interpolating it again every round would compound
  interpolation error and silently freeze the search to the first estimate.
* The search range and the update network's input are separate things.  Round one
  searches a radius-8 window (17x17 candidates) for *evidence*, but the update
  network sees a reduced form of it: a learned 1x1 reduction of the correlation
  vector plus the soft-argmax offset, the top-1 offset, the peak contrast and the
  entropy.  Feeding 289 channels directly would be wasteful; feeding only the
  correlation peak would confuse "this candidate is best" with "this correction
  is trustworthy", which is what the confidence head is for.
* The residual is dense and is added to the flow directly.  It is never
  re-projected onto a global affine, which would throw away the local correction
  the loop exists to produce.  The affine field stays available as a control.
* The confidence head is supervised, not read off the softmax: the loss marks a
  round's correction as reliable when applying it actually reduced the error
  against the ground truth.  At evaluation only the predicted confidence is used.
* The update network is shared across rounds, so extra rounds add computation
  rather than parameters, and rounds can be compared directly.
* The last update layer is zero-initialised, so rounds return the incoming flow
  unchanged at warm start and the coarse field stays the reference point.  Unlike
  a separate multiplicative gain, ``tanh`` has unit slope at zero, so the
  projections still receive gradient immediately.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matching import correspondence_targets


@dataclass
class IterativeRefinementOutput:
    """Per-round results, all flow tensors ``[B,2,H4,W4]`` in 1/4-feature pixels.

    ``flows[r]`` is the absolute field after round ``r+1``; the incoming coarse
    field is the round-zero reference and is not repeated here.
    """

    flows: list[torch.Tensor]
    residuals: list[torch.Tensor]
    applied: list[torch.Tensor]
    confidence: list[torch.Tensor]
    log_probability: list[torch.Tensor]
    valid_candidates: torch.Tensor
    offsets_yx: torch.Tensor


class DiscrepancyGuidedRefinement(nn.Module):
    """Three shared-weight rounds of local search and residual prediction."""

    def __init__(self, channels: int, radius: int = 8, iterations: int = 3,
                 evidence_channels: int = 16, hidden_channels: int = 32,
                 candidate_chunk: int = 9, temperature: float = 0.07,
                 max_step_cells: float = 4.0, confidence_channels: int = 16) -> None:
        super().__init__()
        if channels < 1 or hidden_channels < 1 or evidence_channels < 1:
            raise ValueError("channel counts must be positive")
        if radius < 1 or iterations < 1 or candidate_chunk < 1:
            raise ValueError("radius, iterations and candidate_chunk must be positive")
        if temperature <= 0 or max_step_cells <= 0:
            raise ValueError("temperature and max_step_cells must be positive")
        if confidence_channels < 1:
            raise ValueError("confidence_channels must be positive")
        self.channels = int(channels)
        self.radius = int(radius)
        self.iterations = int(iterations)
        self.evidence_channels = int(evidence_channels)
        self.candidate_chunk = int(candidate_chunk)
        self.temperature = float(temperature)
        self.max_step_cells = float(max_step_cells)
        steps = torch.arange(-radius, radius + 1)
        yy, xx = torch.meshgrid(steps, steps, indexing="ij")
        self.register_buffer("offsets_yx", torch.stack((yy, xx), dim=-1).reshape(-1, 2),
                             persistent=False)
        candidates = (2 * radius + 1) ** 2
        # Shared across modalities and across rounds.
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.correlation_reduce = nn.Sequential(
            nn.Conv2d(candidates, evidence_channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(evidence_channels, evidence_channels, 1),
        )
        # discrepancy + evidence + soft offset + top-1 offset + contrast
        # + entropy + current flow + previous confidence.
        update_inputs = hidden_channels + evidence_channels + 9
        self.update = nn.Sequential(
            nn.Conv2d(update_inputs, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        self.confidence = nn.Sequential(
            nn.Conv2d(update_inputs, confidence_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(confidence_channels, 1, 3, padding=1),
        )
        # Zero residual at warm start: the loop must earn every correction, and
        # the coarse field stays the exact round-zero reference.
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)
        nn.init.zeros_(self.confidence[-1].weight)
        nn.init.zeros_(self.confidence[-1].bias)

    def _sample(self, feature: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        """Bilinear sample of ``feature`` at ``[B,H,W,K,2]`` (y,x) feature coords."""
        batch, channels, height, width = feature.shape
        count = locations.shape[-2]
        grid = torch.stack((2 * locations[..., 1] / max(width - 1, 1) - 1,
                            2 * locations[..., 0] / max(height - 1, 1) - 1), dim=-1)
        samples = F.grid_sample(feature, grid.reshape(batch, height, width * count, 2),
                                mode="bilinear", padding_mode="zeros", align_corners=True)
        return samples.reshape(batch, channels, height, width, count)

    def forward(self, feature_ir: torch.Tensor, feature_vi: torch.Tensor,
                coarse_flow: torch.Tensor) -> IterativeRefinementOutput:
        if feature_ir.ndim != 4 or feature_ir.shape != feature_vi.shape:
            raise ValueError("IR and VI 1/4 features must have equal [B,C,H,W] shapes")
        batch, _, height, width = feature_ir.shape
        if coarse_flow.shape != (batch, 2, height, width):
            raise ValueError("coarse_flow must be [B,2,H,W] in 1/4-feature pixels")
        moving = F.normalize(feature_ir.float(), dim=1)
        reference = F.normalize(feature_vi.float(), dim=1)
        y = torch.arange(height, device=feature_ir.device, dtype=torch.float32)
        x = torch.arange(width, device=feature_ir.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack((yy, xx), dim=-1).unsqueeze(0)
        offsets = self.offsets_yx.to(device=feature_ir.device, dtype=torch.float32)

        flow = coarse_flow.float()
        confidence_previous = torch.zeros(batch, 1, height, width,
                                          device=flow.device, dtype=flow.dtype)
        flows, residuals, applied_parts, confidences, log_probabilities = [], [], [], [], []
        valid_candidates = None
        for _ in range(self.iterations):
            centre = base + flow.permute(0, 2, 3, 1)
            centre_sample = self._sample(moving, centre.unsqueeze(3)).squeeze(-1)
            scores, valid_parts = [], []
            for start in range(0, len(offsets), self.candidate_chunk):
                part = offsets[start:start + self.candidate_chunk]
                locations = centre.unsqueeze(3) + part.view(1, 1, 1, -1, 2)
                valid = ((locations[..., 0] >= 0) & (locations[..., 0] <= height - 1)
                         & (locations[..., 1] >= 0) & (locations[..., 1] <= width - 1))
                samples = self._sample(moving, locations)
                scores.append((reference.unsqueeze(-1) * samples).sum(dim=1))
                valid_parts.append(valid)
            correlation = torch.cat(scores, dim=-1).permute(0, 3, 1, 2)
            valid_candidates = torch.cat(valid_parts, dim=-1).permute(0, 3, 1, 2)
            logits = correlation / self.temperature
            logits = logits.masked_fill(~valid_candidates, -1.0e4)
            log_probability = F.log_softmax(logits, dim=1)
            probability = log_probability.exp()
            soft_offset = torch.einsum("bkhw,kc->bchw", probability, offsets)
            winning = probability.argmax(dim=1)
            top1_offset = offsets[winning].permute(0, 3, 1, 2)
            peak = correlation.amax(dim=1, keepdim=True)
            mean = correlation.mean(dim=1, keepdim=True)
            entropy = -(probability * log_probability).sum(dim=1, keepdim=True)
            entropy = entropy / torch.log(torch.tensor(float(len(offsets)),
                                                       device=entropy.device))
            discrepancy = (self.local_conv(reference)
                           - self.local_conv(centre_sample))
            evidence = self.correlation_reduce(correlation)
            update_input = torch.cat((discrepancy, evidence, soft_offset, top1_offset,
                                      peak - mean, entropy, flow,
                                      confidence_previous), dim=1)
            # A flow-magnitude loss has a degenerate optimum: a constant
            # correction already reaches the mean displacement error, and the
            # first probe of this loop found exactly that -- started from the
            # ground truth, one round moved 2.02 px away and a second 3.58 px,
            # while the aggregate error barely improved.  Removing the spatial
            # mean of the raw update makes that solution unreachable: a constant
            # output becomes exactly zero, whereas removing the mean *after* the
            # tanh would let a round exceed its step bound.
            raw_update = self.update(update_input)
            raw_update = raw_update - raw_update.mean(dim=(-2, -1), keepdim=True)
            delta = self.max_step_cells * torch.tanh(
                raw_update / self.max_step_cells)
            confidence = torch.sigmoid(self.confidence(update_input))
            applied = confidence * delta
            flow = flow + applied
            flows.append(flow)
            residuals.append(delta)
            applied_parts.append(applied)
            confidences.append(confidence)
            log_probabilities.append(log_probability)
            confidence_previous = confidence
        return IterativeRefinementOutput(
            flows=flows, residuals=residuals, applied=applied_parts,
            confidence=confidences, log_probability=log_probabilities,
            valid_candidates=valid_candidates, offsets_yx=offsets)


def refinement_losses(output: IterativeRefinementOutput, gt_flow: torch.Tensor,
                      valid_mask: torch.Tensor | None = None, sigma: float = 0.75,
                      smoothness_weight: float = 0.0, charbonnier_eps: float = 1e-3
                      ) -> dict[str, list[torch.Tensor]]:
    """Per-round flow error, correspondence NLL and confidence supervision.

    The confidence target comes from the ground truth: a round's correction is
    marked reliable where applying it actually reduced the distance to the true
    correspondence.  That is what stops the head from learning to echo the
    softmax peak, which the round-one evidence already contains.

    Round ``r`` searched with the window centred on the flow *before* its own
    update, so its correspondence target is the residual against ``before`` and
    not against the updated field.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    height, width = output.flows[0].shape[-2:]
    target, query_mask = correspondence_targets(gt_flow, (height, width), valid_mask)
    target = target.transpose(1, 2).reshape(gt_flow.shape[0], 2, height, width)
    y = torch.arange(height, device=gt_flow.device, dtype=torch.float32)
    x = torch.arange(width, device=gt_flow.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    base = torch.stack((yy, xx), dim=0).unsqueeze(0)
    offsets = output.offsets_yx.to(output.flows[0].dtype)
    radius = output.offsets_yx.abs().max()
    query = query_mask.reshape(gt_flow.shape[0], height, width)

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    reports: dict[str, list[torch.Tensor]] = {
        "epe": [], "flow": [], "match": [], "confidence": [],
        "confidence_target": [], "smooth": [], "improved_fraction": [],
    }
    for index, flow in enumerate(output.flows):
        before = (flow - output.applied[index]).detach()
        error_after = torch.linalg.vector_norm(target - base - flow, dim=1)
        error_before = torch.linalg.vector_norm(target - base - before, dim=1)
        # The window was centred on ``before``; only those queries were searched.
        search_residual = target - base - before
        covered = (search_residual.abs() <= radius + 0.5).all(dim=1) & query
        active = covered.to(flow.dtype)
        distance = (search_residual.detach().unsqueeze(1)
                    - offsets.view(1, -1, 2, 1, 1)).square().sum(dim=2)
        weights = torch.exp(-distance / (2 * sigma * sigma)) * output.valid_candidates
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        reports["match"].append(masked_mean(
            -(weights * output.log_probability[index]).sum(dim=1), active))
        reports["epe"].append(masked_mean(error_after, active))
        reports["flow"].append(masked_mean(
            torch.sqrt(error_after.square() + charbonnier_eps ** 2), active))
        improved = (error_after < error_before).to(flow.dtype).detach()
        confidence = output.confidence[index][:, 0]
        bce = -(improved * confidence.clamp_min(1e-6).log()
                + (1 - improved) * (1 - confidence).clamp_min(1e-6).log())
        reports["confidence"].append(masked_mean(bce, active))
        reports["confidence_target"].append(masked_mean(improved, active))
        reports["improved_fraction"].append(masked_mean(
            ((error_after < error_before) & covered).to(flow.dtype), active))
        if smoothness_weight > 0:
            applied = output.applied[index]
            reports["smooth"].append(
                (applied[..., 1:] - applied[..., :-1]).abs().mean()
                + (applied[..., 1:, :] - applied[..., :-1, :]).abs().mean())
    return reports
