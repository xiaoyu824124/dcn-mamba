"""Shared multi-scale encoder for MIND descriptors.

The same :class:`MINDFeatureEncoder` instance encodes IR and visible MIND
descriptors. Weight sharing is intentional: after MIND removes much of the
intensity-domain gap, both modalities should occupy one structural feature
space before global cosine matching.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    """Choose a GroupNorm group count that always divides ``channels``."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """3x3 convolution followed by batch-independent normalisation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1,
                      bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )


class ResidualBlock(nn.Module):
    """Small residual block used independently at each pyramid scale."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.body(x))


class MINDFeatureEncoder(nn.Module):
    """One shared CNN encoder returning L2-normalised 1/2, 1/4 and 1/8 maps.

    Args:
        in_channels: Number of MIND descriptor channels (eight for radius one).
        base_channels: Width at the 1/2 scale.
        out_channels: Feature channels at the 1/8 scale used by global matching.
        blocks_per_scale: Number of residual blocks after each downsampling step.

    Input:
        descriptor: ``[B, C_mind, H, W]``.
    Output:
        A dictionary with feature tensors:

        - ``"1/2"``: ``[B, base_channels, H/2, W/2]``;
        - ``"1/4"``: ``[B, 2*base_channels, H/4, W/4]``;
        - ``"1/8"``: ``[B, out_channels, H/8, W/8]``.

    Each feature is L2-normalised on the channel axis. Inputs must have height
    and width divisible by eight so every feature coordinate maps exactly back to
    an input-pixel region in the future flow/warp modules.
    """

    def __init__(self, in_channels: int = 8, base_channels: int = 24,
                 out_channels: int = 64, blocks_per_scale: int = 2,
                 eps: float = 1e-6):
        super().__init__()
        if in_channels < 1 or base_channels < 1 or out_channels < 1:
            raise ValueError("channel counts must be positive")
        if blocks_per_scale < 1:
            raise ValueError("blocks_per_scale must be at least one")
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.out_channels = int(out_channels)
        self.eps = float(eps)

        channels_4 = base_channels * 2
        self.stage_2 = self._stage(in_channels, base_channels, blocks_per_scale)
        self.stage_4 = self._stage(base_channels, channels_4, blocks_per_scale)
        self.stage_8 = self._stage(channels_4, out_channels, blocks_per_scale)

    @staticmethod
    def _stage(in_channels: int, out_channels: int,
               blocks: int) -> nn.Sequential:
        layers = [ConvNormAct(in_channels, out_channels, stride=2)]
        layers.extend(ResidualBlock(out_channels) for _ in range(blocks))
        return nn.Sequential(*layers)

    def _normalise(self, feature: torch.Tensor) -> torch.Tensor:
        return F.normalize(feature, p=2, dim=1, eps=self.eps)

    def forward(self, descriptor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if descriptor.ndim != 4 or descriptor.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [B,{self.in_channels},H,W], got {tuple(descriptor.shape)}")
        height, width = descriptor.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError(
                "MINDFeatureEncoder requires H and W divisible by 8, got "
                f"{height}x{width}")
        feature_2 = self._normalise(self.stage_2(descriptor))
        feature_4 = self._normalise(self.stage_4(feature_2))
        feature_8 = self._normalise(self.stage_8(feature_4))
        return {"1/2": feature_2, "1/4": feature_4, "1/8": feature_8}

    def encode_pair(self, descriptor_ir: torch.Tensor,
                    descriptor_vi: torch.Tensor) -> Tuple[Dict[str, torch.Tensor],
                                                          Dict[str, torch.Tensor]]:
        """Encode IR and VI with this same module and validate their geometry."""
        if descriptor_ir.shape != descriptor_vi.shape:
            raise ValueError(
                "shared encoder needs equal IR/VI descriptor shapes, got "
                f"{tuple(descriptor_ir.shape)} and {tuple(descriptor_vi.shape)}")
        return self(descriptor_ir), self(descriptor_vi)
