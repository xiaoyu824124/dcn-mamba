"""Standalone IR--visible registration research package.

The package intentionally has no dependency on the project's fusion or legacy
registration code. Build and validate the single-frame registration chain here
before it is connected to video fusion.
"""

from .encoder import MINDFeatureEncoder
from .global_matcher import GlobalMatcher
from .local_matcher import LocalMatcher
from .matching import (coarse_matching_loss, matching_diagnostics,
                       windowed_diagnostics)
from .mind import MINDDescriptor, rgb_to_gray
from .losses import RegistrationLoss
from .metrics import endpoint_error
from .registration_net import MINDGlobalRegistration
from .vtmot import VTMOTSingleFrameDataset
from .warp import resize_flow, warp

__all__ = ["MINDDescriptor", "MINDFeatureEncoder", "GlobalMatcher", "LocalMatcher", "warp",
           "resize_flow", "MINDGlobalRegistration", "RegistrationLoss",
           "VTMOTSingleFrameDataset", "endpoint_error", "rgb_to_gray",
           "coarse_matching_loss", "matching_diagnostics",
           "windowed_diagnostics"]
