"""Production single-frame MIND + global-affine IR--visible registration."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from .encoder import MINDFeatureEncoder
from .global_matcher import GlobalMatchOutput, GlobalMatcher
from .mind import MINDDescriptor, paired_mind
from .warp import upsample_feature_flow, warp


@dataclass
class CoarseRegistrationOutput:
    """Intermediate outputs of the MIND/global-matching registration chain.

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
        return CoarseRegistrationOutput(
            coarse_aligned_ir=coarse_aligned_ir,
            coarse_flow=coarse_flow,
            confidence_1_8=match.confidence,
            affine_yx=match.affine_yx,
            match=match,
        )
