"""Single-frame MIND + global-matching coarse IR--visible registration.

This is the first complete spatial registration chain in ``res``.  It has no
temporal propagation, keyframes, RAFT or DCN.  The next stage will take its
coarse-aligned features and add only small local residual refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .encoder import MINDFeatureEncoder
from .dcn_refiner import DCNRefinementOutput, MultiScaleDCNRefiner
from .global_matcher import GlobalMatchOutput, GlobalMatcher
from .mind import MINDDescriptor, paired_mind
from .warp import upsample_feature_flow, warp


@dataclass
class CoarseRegistrationOutput:
    """Intermediate outputs of the MIND/global-matching registration chain.

    ``coarse_flow_1_8`` is feature-grid displacement in `[dy,dx]`; `coarse_flow`
    is its image-grid conversion in pixel units and is the field used to warp IR.
    """

    coarse_aligned_ir: torch.Tensor
    coarse_aligned_ir_feature: torch.Tensor
    coarse_flow: torch.Tensor
    raw_flow_1_8: torch.Tensor
    coarse_flow_1_8: torch.Tensor
    confidence_1_8: torch.Tensor
    mind_ir: torch.Tensor
    mind_vi: torch.Tensor
    features_ir: Dict[str, torch.Tensor]
    features_vi: Dict[str, torch.Tensor]
    matching_probability: torch.Tensor
    correlation: torch.Tensor | None


class MINDGlobalRegistration(nn.Module):
    """IR/VIS coarse registration through MIND and global soft matching.

    Inputs:
        ir: moving infrared frame ``[B,1,H,W]``.
        vi: fixed visible RGB frame ``[B,3,H,W]``.

    The input size must be divisible by eight.  The encoder's 1/8 grid has a
    physical stride of exactly eight input pixels, therefore 1/8 `[dy,dx]` flow
    is lifted by ``(8,8)`` before image-grid backward warping.
    """

    def __init__(self, mind: MINDDescriptor | None = None,
                 encoder: MINDFeatureEncoder | None = None,
                 matcher: GlobalMatcher | None = None) -> None:
        super().__init__()
        self.mind = mind if mind is not None else MINDDescriptor()
        self.encoder = (encoder if encoder is not None else
                        MINDFeatureEncoder(in_channels=self.mind.channels))
        self.matcher = matcher if matcher is not None else GlobalMatcher()
        if self.encoder.in_channels != self.mind.channels:
            raise ValueError(
                "encoder input channels must equal MIND channels: "
                f"{self.encoder.in_channels} != {self.mind.channels}")

    @staticmethod
    def _validate_inputs(ir: torch.Tensor, vi: torch.Tensor) -> None:
        if ir.ndim != 4 or ir.shape[1] != 1:
            raise ValueError(f"IR must be [B,1,H,W], got {tuple(ir.shape)}")
        if vi.ndim != 4 or vi.shape[1] != 3:
            raise ValueError(f"VI must be [B,3,H,W], got {tuple(vi.shape)}")
        if ir.shape[0] != vi.shape[0] or ir.shape[-2:] != vi.shape[-2:]:
            raise ValueError("IR and VI must share B,H,W")
        height, width = ir.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError(
                f"registration input must be divisible by 8, got {height}x{width}")

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> CoarseRegistrationOutput:
        self._validate_inputs(ir, vi)
        mind_ir, mind_vi = paired_mind(ir, vi, self.mind)
        features_ir, features_vi = self.encoder.encode_pair(mind_ir, mind_vi)
        match: GlobalMatchOutput = self.matcher(
            features_ir["1/8"], features_vi["1/8"])

        height, width = ir.shape[-2:]
        feature_height, feature_width = match.coarse_flow.shape[-2:]
        # The encoder was defined as three stride-two stages.  Keep this explicit
        # rather than silently multiplying by a magic constant during resizing.
        stride_y = height / feature_height
        stride_x = width / feature_width
        coarse_flow = upsample_feature_flow(
            match.coarse_flow, (height, width), (stride_y, stride_x))
        coarse_aligned_ir = warp(ir, coarse_flow)
        coarse_aligned_feature = warp(features_ir["1/8"], match.coarse_flow)

        return CoarseRegistrationOutput(
            coarse_aligned_ir=coarse_aligned_ir,
            coarse_aligned_ir_feature=coarse_aligned_feature,
            coarse_flow=coarse_flow,
            raw_flow_1_8=match.raw_flow,
            coarse_flow_1_8=match.coarse_flow,
            confidence_1_8=match.confidence,
            mind_ir=mind_ir,
            mind_vi=mind_vi,
            features_ir=features_ir,
            features_vi=features_vi,
            matching_probability=match.matching_probability,
            correlation=match.correlation,
        )


class MINDDCNRegistration(nn.Module):
    """Complete single-frame MIND -> global matching -> local DCN model."""

    def __init__(self, coarse_registration: MINDGlobalRegistration,
                 use_residual_flow_head: bool = False) -> None:
        super().__init__()
        self.coarse_registration = coarse_registration
        encoder = coarse_registration.encoder
        self.refiner = MultiScaleDCNRefiner(
            channels_2=encoder.base_channels,
            channels_4=encoder.base_channels * 2,
            channels_8=encoder.out_channels,
            use_residual_flow_head=use_residual_flow_head,
        )

    def forward(self, ir: torch.Tensor, vi: torch.Tensor):
        coarse = self.coarse_registration(ir, vi)
        refined: DCNRefinementOutput = self.refiner(
            ir=ir,
            coarse_aligned_ir=coarse.coarse_aligned_ir,
            coarse_flow=coarse.coarse_flow,
            coarse_flow_1_8=coarse.coarse_flow_1_8,
            confidence_1_8=coarse.confidence_1_8,
            features_ir=coarse.features_ir,
            features_vi=coarse.features_vi,
        )
        return coarse, refined
