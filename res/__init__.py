"""Standalone IR--visible registration research package.

The package intentionally has no dependency on the project's fusion or legacy
registration code. Build and validate the single-frame registration chain here
before it is connected to video fusion.
"""

from .encoder import MINDFeatureEncoder
from .dcn_refiner import MultiScaleDCNRefiner
from .global_matcher import GlobalMatcher
from .mind import MINDDescriptor, rgb_to_gray
from .losses import RegistrationLoss
from .metrics import endpoint_error
from .registration_net import MINDDCNRegistration, MINDGlobalRegistration
from .synthetic import SyntheticRegistrationDataset
from .vtmot import VTMOTSingleFrameDataset
from .warp import resize_flow, warp

__all__ = ["MINDDescriptor", "MINDFeatureEncoder", "GlobalMatcher", "warp",
           "resize_flow", "MINDGlobalRegistration", "MINDDCNRegistration",
           "MultiScaleDCNRefiner", "RegistrationLoss", "SyntheticRegistrationDataset",
           "VTMOTSingleFrameDataset", "endpoint_error", "rgb_to_gray"]
