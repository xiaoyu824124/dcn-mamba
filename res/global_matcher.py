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

    Inputs:
        feature_ir: moving feature tensor ``[B,C,H8,W8]``.
        feature_vi: fixed/reference feature tensor ``[B,C,H8,W8]``.
    """

    def __init__(self, temperature: float = 0.07,
                 learnable_temperature: bool = False,
                 return_correlation: bool = True,
                 max_tokens: int = 0, affine_projection: bool = True,
                 affine_ridge: float = 1e-3) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if affine_ridge < 0:
            raise ValueError("affine_ridge must be non-negative")
        self.learnable_temperature = bool(learnable_temperature)
        self.return_correlation = bool(return_correlation)
        self.max_tokens = int(max_tokens)
        self.affine_projection = bool(affine_projection)
        self.affine_ridge = float(affine_ridge)
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
        correlation = torch.bmm(query_vi, key_ir)  # [B, N_vi, N_ir]
        probability = torch.softmax(correlation / self.temperature.float(), dim=-1)

        coordinates = self._coordinates(height, width, feature_ir)
        # q_hat(p) = sum_q P(p,q) q.  This soft-argmax is differentiable.
        expected_ir_coordinate = torch.matmul(probability, coordinates)
        reference_coordinate = coordinates.unsqueeze(0)
        raw_flow_yx = expected_ir_coordinate - reference_coordinate
        raw_flow = raw_flow_yx.transpose(1, 2).reshape(batch, 2, height, width)
        confidence = probability.amax(dim=-1).reshape(batch, 1, height, width)
        if self.affine_projection:
            # VTMOT's injected misalignment is affine.  Fitting the entire
            # all-pairs correspondence field to six parameters makes that
            # dataset prior explicit, rejects local soft-argmax noise and is
            # fully differentiable back to the MIND encoder and correlation.
            # Scale weights to mean one: this leaves WLS unchanged but keeps a
            # fixed ridge numerically meaningful at diffuse initial matching.
            # A fourth power suppresses unmatched locations (whose softmax is
            # almost uniform) while retaining uniform weights when all matches
            # are initially diffuse. This prevents border-only unmatched tokens
            # from biasing an otherwise exact global translation fit.
            # Confidence chooses robust correspondences but does not receive
            # gradients through the matrix inverse. Differentiating those
            # fourth-power weights made an almost rank-one normal matrix blow
            # up under AMP after a few hundred real VTMOT updates.
            weights = confidence.detach().flatten(1).clamp_min(1e-8).pow(4)
            weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-8)
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
