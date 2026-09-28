"""MIND, global coarse matching and local fine IR--visible registration."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MINDFeatureEncoder, SharedPyramidEncoder
from .coarse_transformer import CoarseSACATransformer
from .fine_interaction import FineScaleInteraction
from .fine_cross_attention import FineCrossModalAttention
from .ir_feature_adapter import IRFeatureAdapter
from .global_matcher import GlobalMatchOutput, GlobalMatcher
from .local_matcher import LocalMatchOutput, LocalMatcher
from .mind import MINDDescriptor, paired_mind, rgb_to_gray
from .glu_crft_coarse import GlobalCostDecoder, fit_affine_flow, standardize_image
from .iterative_refinement import DiscrepancyGuidedRefinement, IterativeRefinementOutput
from .spatial_frequency import SpatialFrequencyFusion
from .structural_prior import StructuralPriorEncoder, structural_prior_input
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
    global_flow: torch.Tensor | None = None
    coarse_local_match: LocalMatchOutput | None = None
    refinement: IterativeRefinementOutput | None = None


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
            # Reduce the candidate count with an *integer* stride.  Adaptive
            # pooling to an arbitrary HxW splits e.g. 60 rows into alternating
            # 5/4 bins, displacing cell centres by up to half a cell (four image
            # pixels at 1/8) from the uniform stride that upsample_feature_flow
            # assumes -- a periodic bias comparable to the misalignment itself.
            # An integer factor keeps both the grid and the stride exact.
            factor = max(1, int(math.sqrt(coarse_height * coarse_width
                                          / self.coarse_match_max_tokens)))
            while ((coarse_height // factor) * (coarse_width // factor)
                   > self.coarse_match_max_tokens):
                factor += 1
            match_hw = (max(1, coarse_height // factor),
                        max(1, coarse_width // factor))
            if min(match_hw) < 3 and self.matcher.affine_projection:
                raise ValueError("coarse_match_max_tokens leaves fewer than 3 cells per axis for WLS")
            if factor > 1:
                match_ir = F.avg_pool2d(coarse_ir, factor, factor)
                match_vi = F.avg_pool2d(coarse_vi, factor, factor)
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


class GLUCRFTRegistration(nn.Module):
    """Z-scored shared features, small global cost volume, then local updates.

    The 16x16 global stage follows GLU-Net's fixed candidate count. SA/CA
    adapts cross-modal features before correlation as in CRFT. Decoded global
    flow, rather than diffuse soft-argmax matches, is projected to 6-DoF.
    An optional 1/8 local stage updates that field before 1/4 fine matching.
    """

    def __init__(self, *, encoder: SharedPyramidEncoder, matcher: GlobalMatcher,
                 coarse_decoder: GlobalCostDecoder,
                 coarse_transformer: CoarseSACATransformer | None = None,
                 coarse_refiner: LocalMatcher | None = None,
                 local_matcher: LocalMatcher | None = None,
                 fine_interaction: FineScaleInteraction | None = None,
                 fine_cross_attention: FineCrossModalAttention | None = None,
                 mind: MINDDescriptor | None = None) -> None:
        super().__init__()
        if encoder.in_channels != 1 or not encoder.extra_coarse_scale:
            raise ValueError("GLUCRFTRegistration needs a 1-channel 1/16 encoder")
        if matcher.affine_projection:
            raise ValueError("the global matcher must leave affine fitting to decoded flow")
        self.encoder = encoder
        self.matcher = matcher
        self.coarse_decoder = coarse_decoder
        self.coarse_transformer = coarse_transformer
        self.coarse_refiner = coarse_refiner
        self.local_matcher = local_matcher
        self.fine_interaction = fine_interaction
        self.fine_cross_attention = fine_cross_attention
        # This descriptor is only used by optional diagnostics, never by the
        # forward input path of the new architecture.
        self.mind = mind if mind is not None else MINDDescriptor()
        self.ir_feature_adapter = None

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> CoarseRegistrationOutput:
        MINDGlobalRegistration._validate_inputs(ir, vi)
        height, width = ir.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError("GLU-CRFT input must be divisible by 16")
        features_ir, features_vi = self.encoder.encode_pair(
            standardize_image(ir), standardize_image(rgb_to_gray(vi)))
        match_ir = F.adaptive_avg_pool2d(features_ir["1/16"], self.coarse_decoder.grid_hw)
        match_vi = F.adaptive_avg_pool2d(features_vi["1/16"], self.coarse_decoder.grid_hw)
        if self.coarse_transformer is not None:
            match_ir, match_vi = self.coarse_transformer(match_ir, match_vi)
        match = self.matcher(match_ir, match_vi)
        decoded = self.coarse_decoder(match.matching_probability, match_vi,
                                      match.raw_flow)
        global_feature_flow, global_affine = fit_affine_flow(decoded)
        match.coarse_flow = global_feature_flow
        match.affine_yx = global_affine
        grid_h, grid_w = self.coarse_decoder.grid_hw
        global_flow = upsample_feature_flow(global_feature_flow, (height, width),
                                            (height / grid_h, width / grid_w))
        coarse_flow = global_flow
        affine_yx = global_affine
        confidence = match.confidence
        coarse_local_match = None
        if self.coarse_refiner is not None:
            feature_8 = features_ir["1/8"].shape[-2:]
            stride_8 = global_flow.new_tensor((height / feature_8[0],
                                               width / feature_8[1])).view(1, 2, 1, 1)
            flow_8 = F.interpolate(global_flow, size=feature_8,
                                    mode="bilinear", align_corners=False) / stride_8
            coarse_local_match = self.coarse_refiner(
                features_ir["1/8"], features_vi["1/8"], flow_8)
            fitted_8, affine_yx = fit_affine_flow(coarse_local_match.refined_flow)
            coarse_flow = upsample_feature_flow(fitted_8, (height, width),
                                                 (height / feature_8[0],
                                                  width / feature_8[1]))
            confidence = coarse_local_match.probability.amax(dim=1, keepdim=True)

        coarse_aligned_ir = warp(ir, coarse_flow)
        local_match = None
        final_flow = final_aligned_ir = None
        if self.local_matcher is not None:
            fine_ir, fine_vi = features_ir["1/4"], features_vi["1/4"]
            if self.fine_interaction is not None:
                fine_ir, fine_vi = self.fine_interaction(
                    fine_ir, fine_vi, features_ir["1/8"], features_vi["1/8"])
            feature_4 = fine_ir.shape[-2:]
            stride_4 = coarse_flow.new_tensor((height / feature_4[0],
                                               width / feature_4[1])).view(1, 2, 1, 1)
            flow_4 = F.interpolate(coarse_flow, size=feature_4,
                                    mode="bilinear", align_corners=False) / stride_4
            if self.fine_cross_attention is not None:
                fine_ir, fine_vi = self.fine_cross_attention(fine_ir, fine_vi, flow_4)
            local_match = self.local_matcher(fine_ir, fine_vi, flow_4)
            final_flow = upsample_feature_flow(local_match.refined_flow, (height, width),
                                               (height / feature_4[0], width / feature_4[1]))
            final_aligned_ir = warp(ir, final_flow)
        return CoarseRegistrationOutput(
            coarse_aligned_ir=coarse_aligned_ir, coarse_flow=coarse_flow,
            confidence_1_8=confidence, affine_yx=affine_yx, match=match,
            final_aligned_ir=final_aligned_ir, final_flow=final_flow,
            local_match=local_match, global_flow=global_flow,
            coarse_local_match=coarse_local_match)


class SpatialFrequencyRegistration(nn.Module):
    """Isolated 1/8 correspondence experiment with four input feature modes.

    This stage uses the existing global matcher and 6-DoF projection, so the
    only experimental variable is what reaches the matcher.  Fine refinement
    and DCN deliberately remain outside this coarse-only experiment.
    """

    def __init__(self, *, encoder: SharedPyramidEncoder,
                 fusion: SpatialFrequencyFusion, matcher: GlobalMatcher,
                 coarse_transformer: CoarseSACATransformer | None = None) -> None:
        super().__init__()
        if encoder.in_channels != 1 or encoder.extra_coarse_scale:
            raise ValueError("spatial-frequency registration needs a 1-channel 1/8 encoder")
        if not matcher.affine_projection:
            raise ValueError("spatial-frequency matcher must produce a 6-DoF coarse field")
        self.encoder = encoder
        self.fusion = fusion
        self.matcher = matcher
        self.coarse_transformer = coarse_transformer
        self.local_matcher = None
        self.ir_feature_adapter = None
        self.mind = MINDDescriptor()

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> CoarseRegistrationOutput:
        MINDGlobalRegistration._validate_inputs(ir, vi)
        gray_ir = standardize_image(ir)
        gray_vi = standardize_image(rgb_to_gray(vi))
        features_ir, features_vi = self.encoder.encode_pair(gray_ir, gray_vi)
        match_ir = self.fusion(gray_ir, features_ir["1/8"])
        match_vi = self.fusion(gray_vi, features_vi["1/8"])
        if self.coarse_transformer is not None:
            match_ir, match_vi = self.coarse_transformer(match_ir, match_vi)
        match = self.matcher(match_ir, match_vi)
        height, width = ir.shape[-2:]
        grid_h, grid_w = match.coarse_flow.shape[-2:]
        coarse_flow = upsample_feature_flow(
            match.coarse_flow, (height, width),
            (height / grid_h, width / grid_w))
        return CoarseRegistrationOutput(
            coarse_aligned_ir=warp(ir, coarse_flow), coarse_flow=coarse_flow,
            confidence_1_8=match.confidence, affine_yx=match.affine_yx,
            match=match, global_flow=coarse_flow)


class StructuralPriorRegistration(nn.Module):
    """Modality-specific shallow encoders, shared pyramid, coarse then 1/4 local.

    The 1/8 route (SA-CA, global matcher, 6-DoF WLS projection) is unchanged and
    runs first, so wiring the optional 1/4 stage cannot alter the coarse field.
    ``local_matcher`` searches the shared encoder's 1/4 features in a window
    centred on that coarse field; ``local_centre`` overrides the centre for
    diagnostics, which is what separates "the 1/4 features cannot pick the right
    point" from "the coarse field put the window in the wrong place".
    """

    supports_local_centre = True

    def __init__(self, *, encoder: StructuralPriorEncoder,
                 matcher: GlobalMatcher,
                 coarse_transformer: CoarseSACATransformer | None = None,
                 local_matcher: LocalMatcher | None = None,
                 fine_interaction: FineScaleInteraction | None = None,
                 fine_cross_attention: FineCrossModalAttention | None = None,
                 iterative_refinement: DiscrepancyGuidedRefinement | None = None) -> None:
        super().__init__()
        if not matcher.affine_projection:
            raise ValueError("structural-prior matcher must fit the coarse affine")
        if iterative_refinement is not None and local_matcher is not None:
            raise ValueError(
                "iterative_refinement replaces the single-shot local matcher; enable "
                "only one so the two can be compared directly")
        has_fine_stage = local_matcher is not None or iterative_refinement is not None
        if fine_interaction is not None and not has_fine_stage:
            raise ValueError("fine_interaction requires a 1/4 stage")
        if fine_cross_attention is not None and not has_fine_stage:
            raise ValueError("fine_cross_attention requires a 1/4 stage")
        self.encoder = encoder
        self.matcher = matcher
        self.coarse_transformer = coarse_transformer
        self.local_matcher = local_matcher
        self.fine_interaction = fine_interaction
        self.fine_cross_attention = fine_cross_attention
        self.iterative_refinement = iterative_refinement
        self.ir_feature_adapter = None
        self.mind = MINDDescriptor()

    def forward(self, ir: torch.Tensor, vi: torch.Tensor, *,
                local_centre: torch.Tensor | None = None
                ) -> CoarseRegistrationOutput:
        MINDGlobalRegistration._validate_inputs(ir, vi)
        gray_ir = standardize_image(ir)
        gray_vi = standardize_image(rgb_to_gray(vi))
        features_ir, features_vi = self.encoder.encode_scales(gray_ir, gray_vi)
        feature_ir, feature_vi = features_ir["1/8"], features_vi["1/8"]
        if self.coarse_transformer is not None:
            feature_ir, feature_vi = self.coarse_transformer(feature_ir, feature_vi)
        match = self.matcher(feature_ir, feature_vi)
        height, width = ir.shape[-2:]
        grid_h, grid_w = match.coarse_flow.shape[-2:]
        coarse_flow = upsample_feature_flow(
            match.coarse_flow, (height, width),
            (height / grid_h, width / grid_w))
        coarse_aligned_ir = warp(ir, coarse_flow)

        local_match = None
        refinement = None
        final_flow = None
        final_aligned_ir = None
        if self.local_matcher is not None or self.iterative_refinement is not None:
            fine_ir, fine_vi = features_ir["1/4"], features_vi["1/4"]
            if self.fine_interaction is not None:
                fine_ir, fine_vi = self.fine_interaction(fine_ir, fine_vi,
                                                         feature_ir, feature_vi)
            feature_hw = fine_ir.shape[-2:]
            stride_4 = coarse_flow.new_tensor((height / feature_hw[0],
                                               width / feature_hw[1])).view(1, 2, 1, 1)
            centre = coarse_flow if local_centre is None else local_centre
            if centre.shape != coarse_flow.shape:
                raise ValueError(
                    "local_centre must be an image-grid [B,2,H,W] flow like coarse_flow, "
                    f"got {tuple(centre.shape)}")
            centre_4 = F.interpolate(centre, size=feature_hw, mode="bilinear",
                                     align_corners=False) / stride_4
            if self.fine_cross_attention is not None:
                fine_ir, fine_vi = self.fine_cross_attention(fine_ir, fine_vi, centre_4)
            if self.local_matcher is not None:
                local_match = self.local_matcher(fine_ir, fine_vi, centre_4)
                refined_4 = local_match.refined_flow
            else:
                # The loop is an alternative to the single-shot matcher, not an
                # addition: both start from the same 1/4 features and the same
                # coarse centre, so their round-by-round results are comparable.
                refinement = self.iterative_refinement(fine_ir, fine_vi, centre_4)
                refined_4 = refinement.flows[-1]
            final_flow = upsample_feature_flow(refined_4, (height, width),
                                               (height / feature_hw[0],
                                                width / feature_hw[1]))
            final_aligned_ir = warp(ir, final_flow)
        return CoarseRegistrationOutput(
            coarse_aligned_ir=coarse_aligned_ir, coarse_flow=coarse_flow,
            confidence_1_8=match.confidence, affine_yx=match.affine_yx,
            match=match, final_aligned_ir=final_aligned_ir, final_flow=final_flow,
            local_match=local_match, global_flow=coarse_flow,
            refinement=refinement)


class DirectStructuralPriorRegistration(nn.Module):
    """Fixed prior directly matched at 1/8, with no trainable feature learner."""

    def __init__(self, *, matcher: GlobalMatcher, prior_downsample: int = 4,
                 orientations: int = 4,
                 wavelengths: tuple[int, ...] = (3, 6, 12)) -> None:
        super().__init__()
        if not matcher.affine_projection:
            raise ValueError("direct matcher must fit the coarse affine")
        self.matcher = matcher
        self.prior_downsample = int(prior_downsample)
        self.orientations = int(orientations)
        self.wavelengths = tuple(map(int, wavelengths))
        self.local_matcher = None
        self.ir_feature_adapter = None
        self.mind = MINDDescriptor()

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> CoarseRegistrationOutput:
        MINDGlobalRegistration._validate_inputs(ir, vi)
        gray_ir = standardize_image(ir)
        gray_vi = standardize_image(rgb_to_gray(vi))
        settings = dict(downsample=self.prior_downsample,
                        orientations=self.orientations,
                        wavelengths=self.wavelengths)
        feature_ir = F.avg_pool2d(structural_prior_input(gray_ir, **settings), 8, 8)
        feature_vi = F.avg_pool2d(structural_prior_input(gray_vi, **settings), 8, 8)
        match = self.matcher(feature_ir, feature_vi)
        height, width = ir.shape[-2:]
        grid_h, grid_w = match.coarse_flow.shape[-2:]
        coarse_flow = upsample_feature_flow(
            match.coarse_flow, (height, width),
            (height / grid_h, width / grid_w))
        return CoarseRegistrationOutput(
            coarse_aligned_ir=warp(ir, coarse_flow), coarse_flow=coarse_flow,
            confidence_1_8=match.confidence, affine_yx=match.affine_yx,
            match=match, global_flow=coarse_flow)
