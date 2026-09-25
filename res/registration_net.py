"""MIND, global coarse matching and local fine IR--visible registration."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MINDFeatureEncoder
from .coarse_transformer import CoarseSACATransformer
from .fine_interaction import FineScaleInteraction
from .fine_cross_attention import FineCrossModalAttention
from .ir_feature_adapter import IRFeatureAdapter
from .global_matcher import GlobalMatchOutput, GlobalMatcher
from .local_matcher import LocalMatchOutput, LocalMatcher
from .mind import MINDDescriptor, paired_mind
from .warp import upsample_feature_flow, warp


@dataclass
class CoarseRegistrationOutput:
    """Intermediate and final outputs of the single-frame registration chain.

    ``coarse_flow`` is the image-grid `[dy,dx]` displacement used to warp IR.
    ``match`` exposes the raw matcher output so callers can supervise the
    correspondence distribution (see :mod:`res.matching`) without recomputing
    the ``N x N`` correlation.
    """

    coarse_aligned_ir: torch.Tensor
    coarse_flow: torch.Tensor
    confidence_1_8: torch.Tensor
    affine_yx: torch.Tensor
    match: GlobalMatchOutput | None = None
    final_aligned_ir: torch.Tensor | None = None
    final_flow: torch.Tensor | None = None
    local_match: LocalMatchOutput | None = None


class MINDGlobalRegistration(nn.Module):
    """IR/VIS registration with 1/8 global and 1/4 local matching.

    Inputs:
        ir: moving infrared frame ``[B,1,H,W]``.
        vi: fixed visible RGB frame ``[B,3,H,W]``.

    The input size must be divisible by eight. The encoder emits a 1/8 grid;
    optionally pooling only the global-matching branch reduces its candidate
    count. Flow lifting uses the actual match-grid dimensions in either case.
    """

    def __init__(self, mind: MINDDescriptor | None = None,
                 encoder: MINDFeatureEncoder | None = None,
                 matcher: GlobalMatcher | None = None,
                 local_matcher: LocalMatcher | None = None,
                 coarse_transformer: CoarseSACATransformer | None = None,
                 fine_interaction: FineScaleInteraction | None = None,
                 fine_cross_attention: FineCrossModalAttention | None = None,
                 ir_feature_adapter: IRFeatureAdapter | None = None,
                 coarse_match_max_tokens: int = 0) -> None:
        super().__init__()
        self.mind = mind if mind is not None else MINDDescriptor()
        self.encoder = (encoder if encoder is not None else
                        MINDFeatureEncoder(in_channels=self.mind.channels))
        self.matcher = matcher if matcher is not None else GlobalMatcher()
        self.local_matcher = local_matcher
        self.coarse_transformer = coarse_transformer
        self.fine_interaction = fine_interaction
        self.fine_cross_attention = fine_cross_attention
        self.ir_feature_adapter = ir_feature_adapter
        if coarse_match_max_tokens < 0:
            raise ValueError("coarse_match_max_tokens must be non-negative")
        self.coarse_match_max_tokens = int(coarse_match_max_tokens)
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
        if self.ir_feature_adapter is not None:
            features_ir = self.ir_feature_adapter(features_ir)
        coarse_ir, coarse_vi = features_ir["1/8"], features_vi["1/8"]
        match_ir, match_vi = coarse_ir, coarse_vi
        coarse_height, coarse_width = coarse_ir.shape[-2:]
        if (self.coarse_match_max_tokens
                and coarse_height * coarse_width > self.coarse_match_max_tokens):
            scale = math.sqrt(self.coarse_match_max_tokens / (coarse_height * coarse_width))
            match_hw = (max(1, int(coarse_height * scale)),
                        max(1, int(coarse_width * scale)))
            if min(match_hw) < 3 and self.matcher.affine_projection:
                raise ValueError("coarse_match_max_tokens leaves fewer than 3 cells per axis for WLS")
            match_ir = F.adaptive_avg_pool2d(coarse_ir, match_hw)
            match_vi = F.adaptive_avg_pool2d(coarse_vi, match_hw)
        if self.coarse_transformer is not None:
            match_ir, match_vi = self.coarse_transformer(match_ir, match_vi)
        # The original 1/4 module expects full-size 1/8 context. In the
        # unpooled path, preserve its established post-SA-CA input exactly.
        if match_ir.shape[-2:] == coarse_ir.shape[-2:]:
            fine_context_ir, fine_context_vi = match_ir, match_vi
        else:
            fine_context_ir, fine_context_vi = coarse_ir, coarse_vi
        match: GlobalMatchOutput = self.matcher(match_ir, match_vi)

        height, width = ir.shape[-2:]
        feature_height, feature_width = match.coarse_flow.shape[-2:]
        # The encoder was defined as three stride-two stages.  Keep this explicit
        # rather than silently multiplying by a magic constant during resizing.
        stride_y = height / feature_height
        stride_x = width / feature_width
        coarse_flow = upsample_feature_flow(
            match.coarse_flow, (height, width), (stride_y, stride_x))
        coarse_aligned_ir = warp(ir, coarse_flow)
        local_match = None
        final_flow = None
        final_aligned_ir = None
        if self.local_matcher is not None:
            fine_ir, fine_vi = features_ir["1/4"], features_vi["1/4"]
            if self.fine_interaction is not None:
                fine_ir, fine_vi = self.fine_interaction(
                    fine_ir, fine_vi, fine_context_ir, fine_context_vi)
            feature_hw = fine_ir.shape[-2:]
            stride_4 = coarse_flow.new_tensor((height / feature_hw[0],
                                               width / feature_hw[1])).view(1, 2, 1, 1)
            coarse_flow_4 = F.interpolate(coarse_flow, size=feature_hw,
                                          mode="bilinear", align_corners=False) / stride_4
            if self.fine_cross_attention is not None:
                fine_ir, fine_vi = self.fine_cross_attention(fine_ir, fine_vi,
                                                              coarse_flow_4)
            local_match = self.local_matcher(fine_ir, fine_vi, coarse_flow_4)
            final_flow = upsample_feature_flow(local_match.refined_flow, (height, width),
                                               (height / feature_hw[0], width / feature_hw[1]))
            final_aligned_ir = warp(ir, final_flow)
        return CoarseRegistrationOutput(
            coarse_aligned_ir=coarse_aligned_ir,
            coarse_flow=coarse_flow,
            confidence_1_8=match.confidence,
            affine_yx=match.affine_yx,
            match=match,
            final_aligned_ir=final_aligned_ir,
            final_flow=final_flow,
            local_match=local_match,
        )
