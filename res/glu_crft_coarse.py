"""Small global cost volume followed by affine and prediction-centred refinement.

GLU-Net motivates decoding a 16x16 global cost volume before local search;
CRFT motivates shared cross-modal features before that volume.  The flow
convention throughout this module is fixed-VI to moving-IR ``[dy, dx]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def standardize_image(image: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Per-image, per-channel spatial z-score, as used before CRFT's encoder."""
    mean = image.mean(dim=(-2, -1), keepdim=True)
    scale = image.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
    return (image - mean) / scale


def fit_affine_flow(flow_yx: torch.Tensor, ridge: float = 1e-3
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a learned feature-grid flow onto a 6-DoF affine field.

    Unlike the old coarse path, this fits the *decoded* field rather than
    diffuse all-pairs soft-argmax correspondences. Coordinates remain affine
    outside the image; no boundary clipping is applied.
    """
    if flow_yx.ndim != 4 or flow_yx.shape[1] != 2:
        raise ValueError("flow_yx must be [B,2,H,W]")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    batch, _, height, width = flow_yx.shape
    if min(height, width) < 3:
        raise ValueError("affine fit needs at least three cells per axis")
    y = torch.arange(height, device=flow_yx.device, dtype=torch.float32)
    x = torch.arange(width, device=flow_yx.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coordinates = torch.stack((yy, xx), dim=-1).reshape(1, -1, 2)
    center = coordinates.new_tensor(((height - 1) / 2, (width - 1) / 2))
    scale = coordinates.new_tensor((max((height - 1) / 2, 1.0),
                                    max((width - 1) / 2, 1.0)))
    normalized = (coordinates - center) / scale
    design = torch.cat((normalized, torch.ones_like(normalized[..., :1])), dim=-1)
    design = design.expand(batch, -1, -1)
    targets = (coordinates + flow_yx.float().flatten(2).transpose(1, 2) - center) / scale
    normal = design.transpose(1, 2) @ design
    rhs = design.transpose(1, 2) @ targets
    eye = torch.eye(3, device=flow_yx.device, dtype=torch.float32).unsqueeze(0)
    identity = torch.tensor([[1., 0.], [0., 1.], [0., 0.]],
                            device=flow_yx.device).unsqueeze(0)
    affine_yx = torch.linalg.solve(normal + ridge * eye, rhs + ridge * identity)
    fitted = (design @ affine_yx) * scale + center - coordinates
    return fitted.transpose(1, 2).reshape(batch, 2, height, width).to(flow_yx.dtype), affine_yx


class GlobalCostDecoder(nn.Module):
    """Learn a low-resolution displacement from mutual global correlation.

    The grid is fixed so key positions become stable cost-volume channels.
    A zero-initialized output starts at identity while the matching NLL trains
    the descriptors to distinguish true correspondences.
    """

    def __init__(self, grid_hw: tuple[int, int] = (16, 16),
                 feature_channels: int = 64, hidden_channels: int = 96,
                 max_displacement_cells: float = 6.0) -> None:
        super().__init__()
        height, width = map(int, grid_hw)
        if min(height, width, feature_channels, hidden_channels) < 1:
            raise ValueError("decoder dimensions must be positive")
        if max_displacement_cells <= 0:
            raise ValueError("max_displacement_cells must be positive")
        self.grid_hw = (height, width)
        self.max_displacement_cells = float(max_displacement_cells)
        channels = height * width + feature_channels + 4
        self.network = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, probability: torch.Tensor, reference_feature: torch.Tensor,
                soft_flow: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = reference_feature.shape
        if (height, width) != self.grid_hw or probability.shape != (
                batch, height * width, height * width) or soft_flow.shape != (
                batch, 2, height, width):
            raise ValueError("cost decoder needs a square all-pairs matrix on grid_hw")
        y = torch.linspace(-1, 1, height, device=reference_feature.device)
        x = torch.linspace(-1, 1, width, device=reference_feature.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        position = torch.stack((yy, xx), dim=0).expand(batch, -1, -1, -1)
        cost = probability.float().transpose(1, 2).reshape(batch, height * width,
                                                             height, width) * (height * width)
        prior = soft_flow.float() / soft_flow.new_tensor((height, width)).view(1, 2, 1, 1)
        inputs = torch.cat((cost, reference_feature.float(), position, prior), dim=1)
        return (self.max_displacement_cells * torch.tanh(self.network(inputs)
                / self.max_displacement_cells)).to(reference_feature.dtype)
