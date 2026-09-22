# -*- coding: utf-8 -*-
"""Active HDO cross-modal video registration pipeline.

SEA-RAFT is used for high-accuracy keyframe re-estimation. Non-keyframes
transport the previous cross-modal field with same-modality Farneback motion,
then use a small residual head and optional bounded DCN refinement. WST
smooths keyframe transitions before the final backward warp.

Flow convention: [dy, dx], with
    warped(y, x) = moving(y + flow_y, x + flow_x)
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


def _conv(in_ch, out_ch, k, bias=False, stride=1):
    return nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=bias, stride=stride)


def _to_gray(image):
    if image.shape[1] == 1:
        return image
    return (0.299 * image[:, 0:1]
            + 0.587 * image[:, 1:2]
            + 0.114 * image[:, 2:3])


def _as_rgb(image):
    if image.shape[1] == 1:
        return image.repeat(1, 3, 1, 1)
    if image.shape[1] == 2:
        return torch.cat([image, image[:, :1]], dim=1)
    if image.shape[1] >= 3:
        return image[:, :3]
    raise ValueError("RAFT input must have at least one channel")


class ResBlock(nn.Module):
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.body = nn.Sequential(
            _conv(channels, channels, kernel),
            nn.LeakyReLU(0.1, inplace=True),
            _conv(channels, channels, kernel),
        )

    def forward(self, x):
        return x + self.body(x)


class SpatialTransformer(nn.Module):
    """Backward warp using a [dy, dx] sampling field."""

    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self._grid = None
        self._shape = None

    def _get_grid(self, shape, device, dtype):
        if (self._grid is None or self._shape != tuple(shape)
                or self._grid.device != device or self._grid.dtype != dtype):
            rows = torch.arange(shape[0], device=device, dtype=dtype)
            cols = torch.arange(shape[1], device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
            self._grid = torch.stack([grid_y, grid_x]).unsqueeze(0)
            self._shape = tuple(shape)
        return self._grid

    def forward(self, source, flow):
        height, width = flow.shape[-2:]
        locations = self._get_grid(
            (height, width), flow.device, flow.dtype) + flow
        locations[:, 0] = 2.0 * (
            locations[:, 0] / max(height - 1, 1) - 0.5)
        locations[:, 1] = 2.0 * (
            locations[:, 1] / max(width - 1, 1) - 0.5)
        locations = locations.permute(0, 2, 3, 1)[..., [1, 0]]
        warped = F.grid_sample(
            source, locations, mode=self.mode, padding_mode="zeros",
            align_corners=True)
        return warped, locations
