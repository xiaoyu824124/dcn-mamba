"""Construct the MIND -> global coarse -> local fine registration model."""

from __future__ import annotations

from omegaconf import OmegaConf

from .encoder import MINDFeatureEncoder
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
    local_config = config.get("local_matcher")
    local_matcher = None
    if local_config is not None and bool(local_config.get("enabled", False)):
        settings = {key: value for key, value in
                    OmegaConf.to_container(local_config, resolve=True).items()
                    if key != "enabled"}
        local_matcher = LocalMatcher(**settings)
    return MINDGlobalRegistration(mind=mind, encoder=encoder, matcher=matcher,
                                  local_matcher=local_matcher)
