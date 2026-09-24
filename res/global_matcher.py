"""Differentiable all-pairs global correspondence at one pyramid scale.

This module estimates a *coarse* pixel-displacement field.  It intentionally
does not use RAFT-style recurrent local lookup: every visible feature position
queries every infrared feature position once, then soft-argmax produces a
globally valid initial correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GlobalMatchOutput:
    """Outputs of :class:`GlobalMatcher` on the 1/8 feature grid.

    ``coarse_flow`` is in **feature-grid pixels**, ordered ``[dy, dx]``, with
    shape ``[B,2,H8,W8]``.  It follows the project's backward-warp convention:

    ``aligned_ir(p_vi) = ir(p_vi + coarse_flow(p_vi))``.

    Hence every visible/reference query coordinate ``p_vi`` receives the
    expected infrared/moving coordinate ``q_ir`` and stores ``q_ir - p_vi``.
    A future upsample to image resolution must scale dy by the height ratio and
    dx by the width ratio.
    """

    coarse_flow: torch.Tensor
    raw_flow: torch.Tensor
    confidence: torch.Tensor
    matching_probability: torch.Tensor
    correlation: torch.Tensor | None
    affine_yx: torch.Tensor | None


class GlobalMatcher(nn.Module):
    """All-pairs cosine matcher with soft correspondence expectation.

    Args:
        temperature: Positive softmax temperature.  Smaller values yield a
            sharper correspondence distribution.
        learnable_temperature: If true, optimise the positive temperature in
            log-space.  It is otherwise kept as a non-trainable buffer.
        return_correlation: Return the full ``[B,N,N]`` cosine matrix.  Disable
            it in memory-sensitive training once correspondence visualisation is
            unnecessary; the probability matrix is still returned as requested.
        max_tokens: Optional safety limit for ``N=H*W`` at this scale.  Zero
            disables the guard.  Global matching must remain at 1/8 or lower;
            never run it on the input-resolution grid.
        spatial_prior_sigma: Optional Gaussian displacement prior in feature
            cells, applied after appearance matching. Zero disables it.
        affine_confidence_power: Exponent on detached match confidence used by
            affine WLS. Four reproduces the established weighting exactly;
            zero gives an unweighted fit for a same-checkpoint ablation.
        affine_border_margin: Number of outer 1/8 feature cells excluded from
            affine WLS. Zero preserves the established full-frame fit.

    Inputs:
        feature_ir: moving feature tensor ``[B,C,H8,W8]``.
        feature_vi: fixed/reference feature tensor ``[B,C,H8,W8]``.
    """

    def __init__(self, temperature: float = 0.07,
                 learnable_temperature: bool = False,
                 return_correlation: bool = True,
                 max_tokens: int = 0, affine_projection: bool = True,
                 affine_ridge: float = 1e-3, dual_softmax: bool = False,
                 key_log_scale: bool = False,
                 spatial_prior_sigma: float = 0.0,
                 affine_confidence_power: float = 4.0,
                 affine_border_margin: int = 0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if affine_ridge < 0:
            raise ValueError("affine_ridge must be non-negative")
        if spatial_prior_sigma < 0:
            raise ValueError("spatial_prior_sigma must be non-negative")
        if affine_confidence_power < 0:
            raise ValueError("affine_confidence_power must be non-negative")
        if affine_border_margin < 0:
            raise ValueError("affine_border_margin must be non-negative")
        self.learnable_temperature = bool(learnable_temperature)
        self.return_correlation = bool(return_correlation)
        self.max_tokens = int(max_tokens)
        self.affine_projection = bool(affine_projection)
        self.affine_ridge = float(affine_ridge)
        self.affine_confidence_power = float(affine_confidence_power)
        self.affine_border_margin = int(affine_border_margin)
        self.dual_softmax = bool(dual_softmax)
        # Standard deviation in 1/8 feature cells. Zero preserves all prior
        # checkpoints and the unrestricted all-pairs baseline exactly.
        self.spatial_prior_sigma = float(spatial_prior_sigma)
        # LoFTR-style per-key learnable scale.  Only created when requested, so
        # disabling it leaves the state dict (and older checkpoints) unchanged;
        # warm-starting from such a checkpoint leaves exp(0)=1, i.e. neutral.
        self.key_log_scale = bool(key_log_scale)
        if self.key_log_scale:
            self.log_scale = nn.Parameter(torch.zeros(()))
        log_temperature = torch.tensor(float(temperature)).log()
        if self.learnable_temperature:
            self.log_temperature = nn.Parameter(log_temperature)
        else:
            self.register_buffer("log_temperature", log_temperature)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp_min(1e-4)

    @staticmethod
    def _coordinates(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        """Feature-grid coordinates ``[N,2]`` in ``[y,x]`` order."""
        y = torch.arange(height, device=reference.device, dtype=reference.dtype)
        x = torch.arange(width, device=reference.device, dtype=reference.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=-1).reshape(-1, 2)

    def forward(self, feature_ir: torch.Tensor,
                feature_vi: torch.Tensor) -> GlobalMatchOutput:
        if feature_ir.ndim != 4 or feature_vi.ndim != 4:
            raise ValueError("global matcher expects [B,C,H,W] feature tensors")
        if feature_ir.shape != feature_vi.shape:
            raise ValueError(
                "IR and VI features must have equal shapes, got "
                f"{tuple(feature_ir.shape)} and {tuple(feature_vi.shape)}")
        batch, channels, height, width = feature_ir.shape
        tokens = height * width
        if self.max_tokens and tokens > self.max_tokens:
            raise ValueError(
                f"1/8 global matcher received N={tokens}, above max_tokens="
                f"{self.max_tokens}; crop the input or explicitly raise the limit")

        # Query is fixed VI p; key is moving IR q.  Both normalisations are
        # explicit so callers cannot accidentally turn this into raw dot product.
        query_vi = F.normalize(feature_vi.float().flatten(2).transpose(1, 2),
                               p=2, dim=-1)
        key_ir = F.normalize(feature_ir.float().flatten(2), p=2, dim=1)
        if self.key_log_scale:
            key_ir = key_ir * self.log_scale.exp()
        correlation = torch.bmm(query_vi, key_ir)  # [B, N_vi, N_ir]
        scaled = correlation / self.temperature.float()
        if self.spatial_prior_sigma > 0:
            # A soft displacement prior suppresses distant false matches while
            # leaving all keys available. Apply it *after* appearance-only
            # dual softmax: including it in the column normaliser would give
            # border keys an artificial advantage over interior keys.
            logits = torch.log_softmax(scaled, dim=-1)
            if self.dual_softmax:
                logits = logits + torch.log_softmax(scaled, dim=-2)
            y = torch.arange(height, device=logits.device, dtype=logits.dtype)
            x = torch.arange(width, device=logits.device, dtype=logits.dtype)
            denominator = 2.0 * self.spatial_prior_sigma ** 2
            y_bias = -(y[:, None] - y[None, :]).square() / denominator
            x_bias = -(x[:, None] - x[None, :]).square() / denominator
            logits_grid = logits.view(batch, height, width, height, width)
            logits_grid.add_(y_bias.view(1, height, 1, height, 1))
            logits_grid.add_(x_bias.view(1, 1, width, 1, width))
            probability = torch.softmax(logits, dim=-1)
        elif self.dual_softmax:
            # Normalising both directions suppresses "hub" keys that are similar
            # to every query -- the dominant failure mode of a single softmax
            # over thousands of cross-modal keys.  Renormalising rows keeps the
            # soft-argmax expectation a convex combination.
            probability = torch.softmax(scaled, dim=-1) * torch.softmax(scaled, dim=-2)
            probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            probability = torch.softmax(scaled, dim=-1)

        coordinates = self._coordinates(height, width, feature_ir)
        # q_hat(p) = sum_q P(p,q) q.  This soft-argmax is differentiable.
        expected_ir_coordinate = torch.matmul(probability, coordinates)
        reference_coordinate = coordinates.unsqueeze(0)
        raw_flow_yx = expected_ir_coordinate - reference_coordinate
        raw_flow = raw_flow_yx.transpose(1, 2).reshape(batch, 2, height, width)
        confidence = probability.amax(dim=-1).reshape(batch, 1, height, width)
        if self.affine_projection:
            if self.affine_border_margin * 2 >= min(height, width):
                raise ValueError("affine_border_margin leaves no interior queries")
            # VTMOT's injected misalignment is affine.  Fitting the entire
            # all-pairs correspondence field to six parameters makes that
            # dataset prior explicit, rejects local soft-argmax noise and is
            # fully differentiable back to the MIND encoder and correlation.
            # Scale weights to mean one: this leaves WLS unchanged but keeps a
            # fixed ridge numerically meaningful at diffuse initial matching.
            # The default fourth power suppresses unmatched locations (whose softmax is
            # almost uniform) while retaining uniform weights when all matches
            # are initially diffuse. This prevents border-only unmatched tokens
            # from biasing an otherwise exact global translation fit.
            # Confidence chooses robust correspondences but does not receive
            # gradients through the matrix inverse. Differentiating those
            # confidence weights made an almost rank-one normal matrix blow
            # up under AMP after a few hundred real VTMOT updates.
            weights = confidence.detach().flatten(1).clamp_min(1e-8).pow(
                self.affine_confidence_power)
            if self.affine_border_margin:
                margin = self.affine_border_margin
                interior = ((coordinates[:, 0] >= margin)
                            & (coordinates[:, 0] < height - margin)
                            & (coordinates[:, 1] >= margin)
                            & (coordinates[:, 1] < width - margin))
                weights = weights * interior.to(weights.dtype).unsqueeze(0)
            # The interior ablation can remove the strongest border queries;
            # keep the nonzero weights at the same mean scale as the baseline
            # so the fixed ridge does not become a second changing variable.
            floor = 1e-20 if self.affine_border_margin else 1e-8
            weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(floor)
            # Solve in a [-1,1] coordinate system. Pixel coordinates make the
            # affine normal matrix unnecessarily ill-conditioned on large grids.
            center = coordinates.new_tensor(((height - 1) / 2, (width - 1) / 2))
            scale = coordinates.new_tensor((max((height - 1) / 2, 1.0),
                                            max((width - 1) / 2, 1.0)))
            normalized_reference = (coordinates - center) / scale
            normalized_expected = (expected_ir_coordinate - center) / scale
            design = torch.cat((normalized_reference,
                                torch.ones_like(normalized_reference[:, :1])), dim=1)
            design = design.unsqueeze(0).expand(batch, -1, -1)  # [B,N,3], [y,x,1]
            normal = torch.bmm(design.transpose(1, 2) * weights.unsqueeze(1), design)
            rhs = torch.bmm(design.transpose(1, 2) * weights.unsqueeze(1),
                            normalized_expected)
            eye = torch.eye(3, dtype=normal.dtype, device=normal.device).unsqueeze(0)
            # CUDA LU is not implemented for AMP FP16.  This is only a 3x3
            # system per image, so solve in FP32 and let autograd cast the
            # result back to the surrounding feature precision afterwards.
            affine_yx = torch.linalg.solve(
                normal.float() + self.affine_ridge * eye.float(), rhs.float())
            fitted_coordinate = torch.bmm(design, affine_yx) * scale + center
            # A degenerate correspondence set must not generate arbitrary
            # out-of-image samples that poison subsequent feature warping.
            lower = coordinates.new_zeros(2)
            upper = coordinates.new_tensor((height - 1, width - 1))
            fitted_coordinate = torch.maximum(torch.minimum(fitted_coordinate, upper), lower)
            flow_yx = fitted_coordinate - reference_coordinate
            coarse_flow = flow_yx.transpose(1, 2).reshape(batch, 2, height, width)
        else:
            affine_yx = None
            coarse_flow = raw_flow

        return GlobalMatchOutput(
            coarse_flow=coarse_flow.to(dtype=feature_ir.dtype),
            raw_flow=raw_flow.to(dtype=feature_ir.dtype),
            confidence=confidence.to(dtype=feature_ir.dtype),
            matching_probability=probability.to(dtype=feature_ir.dtype),
            correlation=(correlation.to(dtype=feature_ir.dtype)
                         if self.return_correlation else None),
            affine_yx=(affine_yx.to(dtype=feature_ir.dtype)
                       if affine_yx is not None else None),
        )
