"""Construct the configured single-frame registration experiment."""

from __future__ import annotations

from omegaconf import OmegaConf

from .encoder import MINDFeatureEncoder, SharedPyramidEncoder
from .coarse_transformer import CoarseSACATransformer
from .fine_interaction import FineScaleInteraction
from .fine_cross_attention import FineCrossModalAttention
from .ir_feature_adapter import IRFeatureAdapter
from .global_matcher import GlobalMatcher
from .local_matcher import LocalMatcher
from .mind import MINDDescriptor
from .registration_net import MINDGlobalRegistration
from .registration_net import (GLUCRFTRegistration, SpatialFrequencyRegistration,
                               StructuralPriorRegistration,
                               DirectStructuralPriorRegistration)
from .glu_crft_coarse import GlobalCostDecoder
from .iterative_refinement import DiscrepancyGuidedRefinement
from .spatial_frequency import SpatialFrequencyFusion
from .structural_prior import StructuralPriorEncoder


def build_global_registration(config) -> (MINDGlobalRegistration |
                                          GLUCRFTRegistration |
                                          SpatialFrequencyRegistration |
                                          StructuralPriorRegistration |
                                          DirectStructuralPriorRegistration):
    if str(config.get("architecture", "mind_global")) == "structural_prior_direct":
        prior = OmegaConf.to_container(config.structural_prior, resolve=True)
        for name in ("base_channels", "out_channels", "blocks_per_scale"):
            prior.pop(name, None)
        matcher_settings = OmegaConf.to_container(config.global_matcher, resolve=True)
        matcher_settings["affine_projection"] = True
        matcher_settings["learnable_temperature"] = False
        return DirectStructuralPriorRegistration(
            matcher=GlobalMatcher(**matcher_settings), **prior)
    if str(config.get("architecture", "mind_global")) == "structural_prior":
        encoder = StructuralPriorEncoder(
            **OmegaConf.to_container(config.structural_prior, resolve=True))
        matcher_settings = OmegaConf.to_container(config.global_matcher, resolve=True)
        matcher_settings["affine_projection"] = True
        matcher = GlobalMatcher(**matcher_settings)
        transformer_config = config.get("coarse_transformer")
        transformer = None
        if transformer_config is not None and bool(transformer_config.get("enabled", False)):
            transformer = CoarseSACATransformer(
                encoder.shared.out_channels,
                **{key: value for key, value in
                   OmegaConf.to_container(transformer_config, resolve=True).items()
                   if key != "enabled"})
        # The 1/4 local stage is optional so that a coarse-only checkpoint still
        # describes the same model: enabling it adds parameters, and
        # res.checkpoint.load_registration_state is what lets an older file load.
        local_config = config.get("local_matcher")
        local_matcher = None
        if local_config is not None and bool(local_config.get("enabled", False)):
            local_matcher = LocalMatcher(**{
                key: value for key, value in
                OmegaConf.to_container(local_config, resolve=True).items()
                if key != "enabled"})
        fine_config = config.get("fine_interaction")
        fine_interaction = None
        if fine_config is not None and bool(fine_config.get("enabled", False)):
            if local_matcher is None and not bool(
                    config.get("iterative_refinement", {}).get("enabled", False)):
                raise ValueError("fine_interaction requires a 1/4 stage")
            fine_interaction = FineScaleInteraction(
                encoder.shared.base_channels * 2, encoder.shared.out_channels,
                **{key: value for key, value in
                   OmegaConf.to_container(fine_config, resolve=True).items()
                   if key != "enabled"})
        cross_config = config.get("fine_cross_attention")
        fine_cross_attention = None
        if cross_config is not None and bool(cross_config.get("enabled", False)):
            if local_matcher is None and not bool(
                    config.get("iterative_refinement", {}).get("enabled", False)):
                raise ValueError("fine_cross_attention requires a 1/4 stage")
            fine_cross_attention = FineCrossModalAttention(
                encoder.shared.base_channels * 2,
                **{key: value for key, value in
                   OmegaConf.to_container(cross_config, resolve=True).items()
                   if key != "enabled"})
        iterative_config = config.get("iterative_refinement")
        iterative_refinement = None
        if iterative_config is not None and bool(iterative_config.get("enabled", False)):
            iterative_refinement = DiscrepancyGuidedRefinement(
                encoder.shared.base_channels * 2,
                **{key: value for key, value in
                   OmegaConf.to_container(iterative_config, resolve=True).items()
                   if key != "enabled"})
        return StructuralPriorRegistration(
            encoder=encoder, matcher=matcher, coarse_transformer=transformer,
            local_matcher=local_matcher, fine_interaction=fine_interaction,
            fine_cross_attention=fine_cross_attention,
            iterative_refinement=iterative_refinement)
    if str(config.get("architecture", "mind_global")) == "spatial_frequency":
        settings = OmegaConf.to_container(config.encoder, resolve=True)
        settings["in_channels"] = 1
        settings["extra_coarse_scale"] = False
        encoder = SharedPyramidEncoder(**settings)
        matcher_settings = OmegaConf.to_container(config.global_matcher, resolve=True)
        matcher_settings["affine_projection"] = True
        matcher = GlobalMatcher(**matcher_settings)
        transformer_config = config.get("coarse_transformer")
        transformer = None
        if transformer_config is not None and bool(transformer_config.get("enabled", False)):
            transformer = CoarseSACATransformer(
                encoder.out_channels,
                **{key: value for key, value in
                   OmegaConf.to_container(transformer_config, resolve=True).items()
                   if key != "enabled"})
        # Construct common spatial/SA-CA weights before mode-specific branches
        # so all four fixed-seed ablations start from identical common weights.
        fusion = SpatialFrequencyFusion(
            encoder.out_channels,
            **OmegaConf.to_container(config.spatial_frequency, resolve=True))
        return SpatialFrequencyRegistration(
            encoder=encoder, fusion=fusion, matcher=matcher,
            coarse_transformer=transformer)
    if str(config.get("architecture", "mind_global")) == "glu_crft":
        settings = OmegaConf.to_container(config.encoder, resolve=True)
        settings["in_channels"] = 1
        settings["extra_coarse_scale"] = True
        encoder = SharedPyramidEncoder(**settings)
        matcher_settings = OmegaConf.to_container(config.global_matcher, resolve=True)
        matcher_settings["affine_projection"] = False
        matcher = GlobalMatcher(**matcher_settings)
        decoder = GlobalCostDecoder(
            feature_channels=encoder.out_channels,
            **OmegaConf.to_container(config.coarse_decoder, resolve=True))
        transformer_config = config.get("coarse_transformer")
        transformer = None
        if transformer_config is not None and bool(transformer_config.get("enabled", False)):
            transformer = CoarseSACATransformer(
                encoder.out_channels,
                **{key: value for key, value in
                   OmegaConf.to_container(transformer_config, resolve=True).items()
                   if key != "enabled"})
        coarse_config = config.get("coarse_refiner")
        coarse_refiner = None
        if coarse_config is not None and bool(coarse_config.get("enabled", False)):
            coarse_refiner = LocalMatcher(**{
                key: value for key, value in
                OmegaConf.to_container(coarse_config, resolve=True).items()
                if key != "enabled"})
        local_config = config.get("local_matcher")
        local_matcher = None
        if local_config is not None and bool(local_config.get("enabled", False)):
            local_matcher = LocalMatcher(**{
                key: value for key, value in
                OmegaConf.to_container(local_config, resolve=True).items()
                if key != "enabled"})
        fine_config = config.get("fine_interaction")
        fine_interaction = None
        if fine_config is not None and bool(fine_config.get("enabled", False)):
            if local_matcher is None:
                raise ValueError("fine_interaction requires local_matcher.enabled=true")
            fine_interaction = FineScaleInteraction(
                encoder.base_channels * 2, encoder.out_channels,
                **{key: value for key, value in
                   OmegaConf.to_container(fine_config, resolve=True).items()
                   if key != "enabled"})
        cross_config = config.get("fine_cross_attention")
        fine_cross_attention = None
        if cross_config is not None and bool(cross_config.get("enabled", False)):
            if local_matcher is None:
                raise ValueError("fine_cross_attention requires local_matcher.enabled=true")
            fine_cross_attention = FineCrossModalAttention(
                encoder.base_channels * 2,
                **{key: value for key, value in
                   OmegaConf.to_container(cross_config, resolve=True).items()
                   if key != "enabled"})
        return GLUCRFTRegistration(
            encoder=encoder, matcher=matcher, coarse_decoder=decoder,
            coarse_transformer=transformer, coarse_refiner=coarse_refiner,
            local_matcher=local_matcher, fine_interaction=fine_interaction,
            fine_cross_attention=fine_cross_attention)
    if str(config.get("architecture", "mind_global")) != "mind_global":
        raise ValueError(f"unknown registration architecture: {config.architecture}")
    mind = MINDDescriptor(**OmegaConf.to_container(config.mind, resolve=True))
    encoder = MINDFeatureEncoder(
        in_channels=mind.channels,
        **OmegaConf.to_container(config.encoder, resolve=True),
    )
    matcher = GlobalMatcher(**OmegaConf.to_container(config.global_matcher, resolve=True))
    adapter_config = config.get("ir_feature_adapter")
    ir_feature_adapter = None
    if adapter_config is not None and bool(adapter_config.get("enabled", False)):
        settings = {key: value for key, value in
                    OmegaConf.to_container(adapter_config, resolve=True).items()
                    if key != "enabled"}
        ir_feature_adapter = IRFeatureAdapter(encoder.base_channels * 2,
                                              encoder.out_channels, **settings)
    transformer_config = config.get("coarse_transformer")
    coarse_transformer = None
    if transformer_config is not None and bool(transformer_config.get("enabled", False)):
        settings = {key: value for key, value in
                    OmegaConf.to_container(transformer_config, resolve=True).items()
                    if key != "enabled"}
        coarse_transformer = CoarseSACATransformer(encoder.out_channels, **settings)
    local_config = config.get("local_matcher")
    local_matcher = None
    if local_config is not None and bool(local_config.get("enabled", False)):
        settings = {key: value for key, value in
                    OmegaConf.to_container(local_config, resolve=True).items()
                    if key != "enabled"}
        local_matcher = LocalMatcher(**settings)
    fine_config = config.get("fine_interaction")
    fine_interaction = None
    if fine_config is not None and bool(fine_config.get("enabled", False)):
        if local_matcher is None:
            raise ValueError("fine_interaction requires local_matcher.enabled=true")
        settings = {key: value for key, value in
                    OmegaConf.to_container(fine_config, resolve=True).items()
                    if key != "enabled"}
        fine_interaction = FineScaleInteraction(encoder.base_channels * 2,
                                                encoder.out_channels, **settings)
    cross_config = config.get("fine_cross_attention")
    fine_cross_attention = None
    if cross_config is not None and bool(cross_config.get("enabled", False)):
        if local_matcher is None:
            raise ValueError("fine_cross_attention requires local_matcher.enabled=true")
        settings = {key: value for key, value in
                    OmegaConf.to_container(cross_config, resolve=True).items()
                    if key != "enabled"}
        fine_cross_attention = FineCrossModalAttention(encoder.base_channels * 2,
                                                       **settings)
    return MINDGlobalRegistration(mind=mind, encoder=encoder, matcher=matcher,
                                  local_matcher=local_matcher,
                                  coarse_transformer=coarse_transformer,
                                  fine_interaction=fine_interaction,
                                  fine_cross_attention=fine_cross_attention,
                                  ir_feature_adapter=ir_feature_adapter,
                                  coarse_match_max_tokens=int(config.get(
                                      "coarse_match_max_tokens", 0)))
