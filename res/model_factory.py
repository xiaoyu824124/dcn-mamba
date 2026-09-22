"""Construct the production MIND-to-global-affine registration model."""

from __future__ import annotations

from omegaconf import OmegaConf

from .encoder import MINDFeatureEncoder
from .global_matcher import GlobalMatcher
from .mind import MINDDescriptor
from .registration_net import MINDGlobalRegistration


def build_global_registration(config) -> MINDGlobalRegistration:
    mind = MINDDescriptor(**OmegaConf.to_container(config.mind, resolve=True))
    encoder = MINDFeatureEncoder(
        in_channels=mind.channels,
        **OmegaConf.to_container(config.encoder, resolve=True),
    )
    matcher = GlobalMatcher(**OmegaConf.to_container(config.global_matcher, resolve=True))
    return MINDGlobalRegistration(mind=mind, encoder=encoder, matcher=matcher)
