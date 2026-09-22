"""Multi-scale deformable local refinement after global coarse registration.

DCN is deliberately *not* used to estimate large displacement.  Every level
first backward-warps the IR feature with the global matcher's coarse flow.  DCN
then samples only a 3x3 local neighbourhood around that already aligned feature
and produces a refined representation for reconstruction or an optional
explicit residual-flow head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d
except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "MultiScaleDCNRefiner requires torchvision.ops.DeformConv2d. Install a "
        "CUDA-compatible torch/torchvision pair; this module never silently "
        "falls back to ordinary convolution.") from exc

from .warp import upsample_feature_flow, warp


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _DCNFeatureBlock(nn.Module):
    """One local DCN refinement block at a fixed feature resolution."""

    def __init__(self, channels: int, has_prior: bool, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("DCN kernel_size must be a positive odd integer")
        self.channels = int(channels)
        self.has_prior = bool(has_prior)
        self.kernel_size = int(kernel_size)
        taps = kernel_size * kernel_size
        input_channels = 2 * channels + 1 + (channels if has_prior else 0)
        hidden = max(16, channels)
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # First 2K values: DCN sampling offsets; final K: modulation mask.
            nn.Conv2d(hidden, 3 * taps, 3, padding=1),
        )
        self.deform = DeformConv2d(channels, channels, kernel_size,
                                   padding=kernel_size // 2, bias=False)
        # Fuse DCN output + warped IR + VI (+ coarser prior when present).
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * channels + (channels if has_prior else 0), channels,
                      3, padding=1, bias=False),
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Offsets start at zero; DCN therefore starts as a standard local
        # convolution around the globally aligned feature, never a global search.
        nn.init.zeros_(self.offset_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].bias)

    def forward(self, warped_ir: torch.Tensor, vi: torch.Tensor,
                confidence: torch.Tensor,
                prior: Optional[torch.Tensor] = None):
        if self.has_prior != (prior is not None):
            raise ValueError("DCN prior presence does not match block configuration")
        if warped_ir.shape != vi.shape:
            raise ValueError("warped IR and VI features must share shape")
        if confidence.shape[:2] != (warped_ir.shape[0], 1):
            raise ValueError("confidence must have shape [B,1,H,W]")
        if confidence.shape[-2:] != warped_ir.shape[-2:]:
            raise ValueError("confidence and feature resolution must match")
        values = [warped_ir, vi, confidence]
        fuse_values = [warped_ir, vi]
        if prior is not None:
            if prior.shape != warped_ir.shape:
                raise ValueError("projected prior must match feature shape")
            values.append(prior)
            fuse_values.append(prior)
        prediction = self.offset_predictor(torch.cat(values, dim=1))
        taps = self.kernel_size * self.kernel_size
        offsets = prediction[:, :2 * taps]
        mask = torch.sigmoid(prediction[:, 2 * taps:])
        deformed = self.deform(warped_ir, offsets, mask)
        refined = self.fuse(torch.cat([deformed, *fuse_values], dim=1))
        return refined, offsets, mask


class ResidualFlowHead(nn.Module):
    """Optional explicit local flow head, separate from DCN sampling offsets.

    Its output is a *single* `[dy,dx]` residual field.  This is intentionally a
    distinct head: DCN's ``K`` sampling offsets are not an optical-flow field.
    """

    def __init__(self, channels: int, max_residual_feature_px: float = 2.0):
        super().__init__()
        self.max_residual_feature_px = float(max_residual_feature_px)
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(feature)) * self.max_residual_feature_px


@dataclass
class DCNRefinementOutput:
    """Feature-level DCN results and optional explicit final flow."""

    refined_aligned_ir: torch.Tensor
    refined_features: Dict[str, torch.Tensor]
    dcn_offsets: Dict[str, torch.Tensor]
    dcn_masks: Dict[str, torch.Tensor]
    residual_flow: Optional[torch.Tensor]
    final_flow: Optional[torch.Tensor]


class MultiScaleDCNRefiner(nn.Module):
    """Local 1/8 -> 1/4 -> 1/2 DCN feature refinement.

    Inputs use the output of ``MINDGlobalRegistration``.  The default mode has
    no explicit residual flow: it reconstructs the refined aligned IR from DCN
    features while retaining the coarse physical warp.  Set
    ``use_residual_flow_head=True`` only for an ablation requiring a final dense
    displacement field.
    """

    def __init__(self, channels_2: int, channels_4: int, channels_8: int,
                 kernel_size: int = 3, use_residual_flow_head: bool = False,
                 max_residual_feature_px: float = 2.0):
        super().__init__()
        self.block_8 = _DCNFeatureBlock(channels_8, has_prior=False,
                                        kernel_size=kernel_size)
        self.block_4 = _DCNFeatureBlock(channels_4, has_prior=True,
                                        kernel_size=kernel_size)
        self.block_2 = _DCNFeatureBlock(channels_2, has_prior=True,
                                        kernel_size=kernel_size)
        self.prior_8_to_4 = nn.Conv2d(channels_8, channels_4, 1, bias=False)
        self.prior_4_to_2 = nn.Conv2d(channels_4, channels_2, 1, bias=False)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels_2, channels_2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels_2, 1, 3, padding=1),
        )
        # At step zero the non-flow branch returns coarse-aligned IR exactly;
        # training must prove that DCN reconstruction improves it.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.residual_flow_head = (
            ResidualFlowHead(channels_2, max_residual_feature_px)
            if use_residual_flow_head else None)

    @staticmethod
    def _flow_to_finer(flow: torch.Tensor, size) -> torch.Tensor:
        """Map a flow to the next feature level (stride halves in both axes)."""
        in_height, in_width = flow.shape[-2:]
        out_height, out_width = size
        result = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
        return result * flow.new_tensor((out_height / in_height, out_width / in_width)).view(
            1, 2, 1, 1)

    def forward(self, *, ir: torch.Tensor, coarse_aligned_ir: torch.Tensor,
                coarse_flow: torch.Tensor, coarse_flow_1_8: torch.Tensor,
                confidence_1_8: torch.Tensor, features_ir: Dict[str, torch.Tensor],
                features_vi: Dict[str, torch.Tensor]) -> DCNRefinementOutput:
        for scale in ("1/2", "1/4", "1/8"):
            if scale not in features_ir or scale not in features_vi:
                raise ValueError(f"missing {scale} feature for DCN refinement")
        ir_8, vi_8 = features_ir["1/8"], features_vi["1/8"]
        ir_4, vi_4 = features_ir["1/4"], features_vi["1/4"]
        ir_2, vi_2 = features_ir["1/2"], features_vi["1/2"]
        if coarse_flow_1_8.shape[-2:] != ir_8.shape[-2:]:
            raise ValueError("coarse 1/8 flow and 1/8 feature shapes must match")

        warped_8 = warp(ir_8, coarse_flow_1_8)
        refined_8, offsets_8, mask_8 = self.block_8(
            warped_8, vi_8, confidence_1_8)

        flow_4 = self._flow_to_finer(coarse_flow_1_8, ir_4.shape[-2:])
        confidence_4 = F.interpolate(confidence_1_8, size=ir_4.shape[-2:],
                                     mode="bilinear", align_corners=False)
        prior_4 = self.prior_8_to_4(F.interpolate(
            refined_8, size=ir_4.shape[-2:], mode="bilinear", align_corners=False))
        warped_4 = warp(ir_4, flow_4)
        refined_4, offsets_4, mask_4 = self.block_4(
            warped_4, vi_4, confidence_4, prior_4)

        flow_2 = self._flow_to_finer(flow_4, ir_2.shape[-2:])
        confidence_2 = F.interpolate(confidence_4, size=ir_2.shape[-2:],
                                     mode="bilinear", align_corners=False)
        prior_2 = self.prior_4_to_2(F.interpolate(
            refined_4, size=ir_2.shape[-2:], mode="bilinear", align_corners=False))
        warped_2 = warp(ir_2, flow_2)
        refined_2, offsets_2, mask_2 = self.block_2(
            warped_2, vi_2, confidence_2, prior_2)

        if self.residual_flow_head is None:
            delta_image = F.interpolate(refined_2, size=ir.shape[-2:],
                                        mode="bilinear", align_corners=False)
            refined_ir = coarse_aligned_ir + self.decoder(delta_image)
            residual_flow = final_flow = None
        else:
            residual_2 = self.residual_flow_head(refined_2)
            residual_flow = upsample_feature_flow(
                residual_2, ir.shape[-2:], stride_yx=(2.0, 2.0))
            final_flow = coarse_flow + residual_flow
            refined_ir = warp(ir, final_flow)

        return DCNRefinementOutput(
            refined_aligned_ir=refined_ir,
            refined_features={"1/8": refined_8, "1/4": refined_4, "1/2": refined_2},
            dcn_offsets={"1/8": offsets_8, "1/4": offsets_4, "1/2": offsets_2},
            dcn_masks={"1/8": mask_8, "1/4": mask_4, "1/2": mask_2},
            residual_flow=residual_flow,
            final_flow=final_flow,
        )
