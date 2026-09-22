"""Lightweight non-keyframe residual and confidence refinement."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SpatialTransformer, _to_gray

class LiteResidualRefinement(nn.Module):
    """Quarter-resolution residual/confidence head for non-keyframes."""

    def __init__(self, channels=16, radius=4, max_residual=8.0):
        super().__init__()
        self.radius = int(radius)
        self.max_residual = float(max_residual)
        self.stn = SpatialTransformer()
        self.feature = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.GroupNorm(max(1, channels // 4), channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        corr_channels = (2 * self.radius + 1) ** 2
        self.body = nn.Sequential(
            nn.Conv2d(corr_channels + 3, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.head = nn.Conv2d(32, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[2] = 1.0

    def _correlation(self, query, key):
        height, width = query.shape[-2:]
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        padded = F.pad(
            key, (self.radius, self.radius, self.radius, self.radius),
            mode="replicate")
        values = []
        for dy in range(2 * self.radius + 1):
            for dx in range(2 * self.radius + 1):
                values.append((
                    query * padded[..., dy:dy + height, dx:dx + width]
                ).sum(dim=1, keepdim=True))
        return torch.cat(values, dim=1)

    def forward(self, moving, fixed, propagated_flow):
        moving_gray = _to_gray(moving)
        fixed_gray = _to_gray(fixed)
        warped = self.stn(moving_gray, propagated_flow)[0]
        target_size = (
            max(8, moving.shape[-2] // 4),
            max(8, moving.shape[-1] // 4))
        warped = F.interpolate(
            warped, size=target_size, mode="bilinear", align_corners=True)
        fixed_small = F.interpolate(
            fixed_gray, size=target_size, mode="bilinear", align_corners=True)
        flow_small = F.interpolate(
            propagated_flow, size=target_size,
            mode="bilinear", align_corners=True)
        flow_small = flow_small * flow_small.new_tensor([
            target_size[0] / max(propagated_flow.shape[-2], 1),
            target_size[1] / max(propagated_flow.shape[-1], 1),
        ]).view(1, 2, 1, 1)
        moving_feature = self.feature(warped)
        fixed_feature = self.feature(fixed_small)
        corr = self._correlation(fixed_feature, moving_feature)
        scale = flow_small.new_tensor([
            max(target_size[0] - 1, 1),
            max(target_size[1] - 1, 1),
        ]).view(1, 2, 1, 1)
        inputs = torch.cat(
            [corr, warped - fixed_small, flow_small / scale], dim=1)
        prediction = self.head(self.body(inputs))
        delta = torch.tanh(prediction[:, :2]) * self.max_residual
        confidence = torch.sigmoid(prediction[:, 2:3])
        delta = F.interpolate(
            delta, size=moving.shape[-2:], mode="bilinear",
            align_corners=True)
        delta = delta * delta.new_tensor([
            moving.shape[-2] / max(target_size[0], 1),
            moving.shape[-1] / max(target_size[1], 1),
        ]).view(1, 2, 1, 1)
        confidence = F.interpolate(
            confidence, size=moving.shape[-2:],
            mode="bilinear", align_corners=True)
        return delta, confidence
