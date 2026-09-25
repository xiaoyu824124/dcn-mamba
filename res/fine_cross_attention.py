"""Coarse-aligned local IR-to-VI interaction before 1/4 correspondence search."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warp import warp


class FineCrossModalAttention(nn.Module):
    """Condition VI queries on a small window of coarse-aligned IR features.

    IR features stay on their original grid so LocalMatcher can continue to
    sample them at ``p + coarse(p) + offset``. The zero residual gain makes
    warm-start predictions identical to the old model at step zero.
    """

    def __init__(self, channels: int, hidden_channels: int = 16,
                 window_size: int = 5, temperature: float = 0.2) -> None:
        super().__init__()
        if channels < 1 or hidden_channels < 1 or window_size < 1 or window_size % 2 != 1:
            raise ValueError("channels must be positive and window_size must be odd")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.window_size = int(window_size)
        self.temperature = float(temperature)
        self.query = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.key = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.value = nn.Conv2d(channels, hidden_channels, 1, bias=False)
        self.output = nn.Conv2d(hidden_channels, channels, 1, bias=False)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, fine_ir: torch.Tensor, fine_vi: torch.Tensor,
                coarse_flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (fine_ir.ndim != 4 or fine_ir.shape != fine_vi.shape
                or fine_ir.shape[1] != self.channels
                or coarse_flow.shape != (fine_ir.shape[0], 2, *fine_ir.shape[-2:])):
            raise ValueError("fine cross attention needs paired 1/4 features and a 1/4 coarse flow")
        batch, _, height, width = fine_ir.shape
        aligned_ir = warp(fine_ir.float(), coarse_flow.float())
        aligned_valid = warp(torch.ones_like(fine_ir[:, :1], dtype=torch.float32),
                             coarse_flow.float()) > 0.999
        query = F.normalize(self.query(fine_vi).float(), dim=1)
        key = self.key(aligned_ir).float()
        value = self.value(aligned_ir).float()
        count = self.window_size ** 2
        padding = self.window_size // 2
        key_patches = F.unfold(key, self.window_size, padding=padding).reshape(
            batch, self.hidden_channels, count, height, width)
        value_patches = F.unfold(value, self.window_size, padding=padding).reshape(
            batch, self.hidden_channels, count, height, width)
        valid_patches = F.unfold(aligned_valid.float(), self.window_size,
                                 padding=padding).reshape(batch, count, height, width) > 0.5
        key_patches = F.normalize(key_patches, dim=1)
        scores = (query.unsqueeze(2) * key_patches).sum(dim=1) / self.temperature
        scores = scores.masked_fill(~valid_patches, -1.0e4)
        weights = scores.softmax(dim=1) * valid_patches
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        message = (weights.unsqueeze(1) * value_patches).sum(dim=2)
        correction = self.output(message)
        return fine_ir, fine_vi + self.gain * correction
