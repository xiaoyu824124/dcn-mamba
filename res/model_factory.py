"""Construct the MIND -> global coarse -> local fine registration model."""

from __future__ import annotations

from omegaconf import OmegaConf

from .encoder import MINDFeatureEncoder
from .coarse_transformer import CoarseSACATransformer
from .fine_interaction import FineScaleInteraction
from .global_matcher import GlobalMatcher
from .local_matcher import LocalMatcher
from .mind import MINDDescriptor
from .registration_net import MINDGlobalRegistration


def build_global_registration(config) -> MINDGlobalRegistration:
    mind = MINDDescriptor(**OmegaConf.to_container(config.mind, resolve=True))
    encoder = MINDFeatureEncoder(
        in_channels=mind.channels,
        **OmegaConf.to_container(config.encoder, resolve=True),
    )
    matcher = GlobalMatcher(**OmegaConf.to_container(config.global_matcher, resolve=True))
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
    return MINDGlobalRegistration(mind=mind, encoder=encoder, matcher=matcher,
                                  local_matcher=local_matcher,
                                  coarse_transformer=coarse_transformer,
                                  fine_interaction=fine_interaction,
                                  coarse_match_max_tokens=int(config.get(
                                      "coarse_match_max_tokens", 0)))
