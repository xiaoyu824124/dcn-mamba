"""Small infrared-only feature calibration after the shared MIND encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _ResidualAdapter(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(hidden_channels, 8), hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
        )
        nn.init.zeros_(self.network[-1].weight)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.network(feature)


class IRFeatureAdapter(nn.Module):
    """Calibrate IR 1/4 and 1/8 features while leaving VI features untouched.

    Zero-initialized final convolutions make an existing cross-modal checkpoint
    produce identical predictions at step zero. The shared encoder and matcher
    may then be frozen to isolate the effect of this IR-specific calibration.
    """

    def __init__(self, channels_4: int, channels_8: int,
                 hidden_channels: int = 32) -> None:
        super().__init__()
        if min(channels_4, channels_8, hidden_channels) < 1:
            raise ValueError("feature and hidden channel counts must be positive")
        self.channels_4 = int(channels_4)
        self.channels_8 = int(channels_8)
        self.adapter_4 = _ResidualAdapter(channels_4, hidden_channels)
        self.adapter_8 = _ResidualAdapter(channels_8, hidden_channels)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if ("1/4" not in features or "1/8" not in features
                or features["1/4"].ndim != 4 or features["1/8"].ndim != 4
                or features["1/4"].shape[1] != self.channels_4
                or features["1/8"].shape[1] != self.channels_8):
            raise ValueError("IR adapter needs 1/4 and 1/8 encoder features")
        return {**features,
                "1/4": self.adapter_4(features["1/4"]),
                "1/8": self.adapter_8(features["1/8"])}
