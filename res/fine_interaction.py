"""Shared 1/4 feature projection with top-down 1/8 context before local matching."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FineScaleInteraction(nn.Module):
    """Fuse each modality's 1/4 features with its own 1/8 context.

    The same projections and fusion weights process IR and VI. A zero-initialized
    residual gain preserves an existing checkpoint's local matcher predictions
    exactly at warm start; the module learns only when fine-stage training helps.
    """

    def __init__(self, channels_4: int, channels_8: int,
                 hidden_channels: int | None = None) -> None:
        super().__init__()
        if channels_4 < 1 or channels_8 < 1:
            raise ValueError("feature channels must be positive")
        hidden = channels_4 if hidden_channels is None else int(hidden_channels)
        if hidden < 1:
            raise ValueError("hidden_channels must be positive")
        self.channels_4 = int(channels_4)
        self.channels_8 = int(channels_8)
        self.fine_projection = nn.Conv2d(channels_4, hidden, 1, bias=False)
        self.coarse_projection = nn.Conv2d(channels_8, hidden, 1, bias=False)
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(hidden, 8), hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels_4, 1),
        )
        self.gain = nn.Parameter(torch.zeros(()))

    def _adapt(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        coarse_up = F.interpolate(coarse, size=fine.shape[-2:],
                                  mode="bilinear", align_corners=False)
        fused = self.fusion(torch.cat((self.fine_projection(fine),
                                       self.coarse_projection(coarse_up)), dim=1))
        return fine + self.gain * fused

    def forward(self, fine_ir: torch.Tensor, fine_vi: torch.Tensor,
                coarse_ir: torch.Tensor, coarse_vi: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if (fine_ir.ndim != 4 or coarse_ir.ndim != 4
                or fine_ir.shape != fine_vi.shape
                or coarse_ir.shape != coarse_vi.shape
                or fine_ir.shape[0] != coarse_ir.shape[0]
                or fine_ir.shape[1] != self.channels_4
                or coarse_ir.shape[1] != self.channels_8):
            raise ValueError("fine interaction needs paired 1/4 and 1/8 feature maps")
        return (self._adapt(fine_ir, coarse_ir),
                self._adapt(fine_vi, coarse_vi))
