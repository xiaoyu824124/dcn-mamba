"""Deterministic synthetic IR--visible pairs for registration smoke training.

The synthetic set is intentionally a geometry test bed, not a replacement for
VTMOT.  It creates one latent thermal structure, renders a differently toned
visible frame, and applies a known fixed-to-moving affine transform to make the
IR input.  The returned ``gt_flow`` follows the package-wide backward-warp
convention, so ``warp(ir, gt_flow)`` recovers ``fixed_ir`` on valid pixels.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .warp import warp


def _yx_grid(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    y = torch.arange(height, dtype=reference.dtype, device=reference.device)
    x = torch.arange(width, dtype=reference.dtype, device=reference.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx), dim=-1)


def affine_flow_yx(fixed_to_moving: torch.Tensor, height: int, width: int,
                   reference: torch.Tensor) -> torch.Tensor:
    """Create ``[dy,dx]`` fixed-grid flow for a 2x3 ``[y,x]`` affine map."""
    if fixed_to_moving.shape != (2, 3):
        raise ValueError("fixed_to_moving must have shape [2,3] in [y,x] order")
    points = _yx_grid(height, width, reference)
    mapped = torch.einsum("ij,hwj->hwi", fixed_to_moving[:, :2], points)
    mapped = mapped + fixed_to_moving[:, 2].view(1, 1, 2)
    return (mapped - points).permute(2, 0, 1).unsqueeze(0)


def invert_affine_yx(matrix: torch.Tensor) -> torch.Tensor:
    """Invert a 2x3 affine map expressed in ``[y,x]`` coordinates."""
    if matrix.shape != (2, 3):
        raise ValueError("matrix must have shape [2,3]")
    linear_inv = torch.linalg.inv(matrix[:, :2])
    translation = -linear_inv @ matrix[:, 2:]
    return torch.cat((linear_inv, translation), dim=1)


def valid_affine_mask(fixed_to_moving: torch.Tensor, height: int, width: int,
                      reference: torch.Tensor) -> torch.Tensor:
    """Mask pixels whose forward and inverse affine samples remain in bounds."""
    points = _yx_grid(height, width, reference)
    forward = torch.einsum("ij,hwj->hwi", fixed_to_moving[:, :2], points)
    forward = forward + fixed_to_moving[:, 2].view(1, 1, 2)
    inverse = invert_affine_yx(fixed_to_moving)
    source = torch.einsum("ij,hwj->hwi", inverse[:, :2], forward)
    source = source + inverse[:, 2].view(1, 1, 2)
    inside_forward = ((forward[..., 0] >= 0) & (forward[..., 0] <= height - 1)
                      & (forward[..., 1] >= 0) & (forward[..., 1] <= width - 1))
    inside_source = ((source[..., 0] >= 0) & (source[..., 0] <= height - 1)
                     & (source[..., 1] >= 0) & (source[..., 1] <= width - 1))
    return (inside_forward & inside_source).to(reference.dtype).unsqueeze(0)


def _latent_structure(height: int, width: int, generator: torch.Generator) -> torch.Tensor:
    """Create a textured but smooth thermal-like image in [0,1]."""
    low_height, low_width = max(2, height // 8), max(2, width // 8)
    coarse = torch.rand((1, 1, low_height, low_width), generator=generator)
    coarse = F.interpolate(coarse, size=(height, width), mode="bicubic", align_corners=True)
    fine = torch.rand((1, 1, height, width), generator=generator)
    fine = F.avg_pool2d(fine, kernel_size=5, stride=1, padding=2)
    image = 0.8 * coarse + 0.2 * fine
    return (image - image.amin()) / (image.amax() - image.amin()).clamp_min(1e-6)


class SyntheticRegistrationDataset(Dataset):
    """Known-affine cross-modal registration samples with reproducible indexing.

    ``translation_px`` controls the deliberately large search displacement.
    Rotation and scale are kept modest because this first smoke-training stage
    is meant to validate global correspondence before harder VTMOT content.
    """

    def __init__(self, *, length: int = 4096, height: int = 128, width: int = 160,
                 translation_px: float = 24.0, rotation_deg: float = 8.0,
                 scale_jitter: float = 0.08, seed: int = 1234) -> None:
        if length < 1 or height < 16 or width < 16:
            raise ValueError("length must be positive and image size must be at least 16")
        if height % 8 or width % 8:
            raise ValueError("synthetic image height and width must be divisible by 8")
        self.length, self.height, self.width = int(length), int(height), int(width)
        self.translation_px = float(translation_px)
        self.rotation_deg, self.scale_jitter, self.seed = float(rotation_deg), float(scale_jitter), int(seed)

    def __len__(self) -> int:
        return self.length

    def _generator(self, index: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(index))
        return generator

    def _sample_affine(self, generator: torch.Generator) -> torch.Tensor:
        angle = (torch.rand((), generator=generator).item() * 2 - 1) * self.rotation_deg
        angle = angle * math.pi / 180.0
        scale = 1.0 + (torch.rand((), generator=generator).item() * 2 - 1) * self.scale_jitter
        translation = (torch.rand(2, generator=generator) * 2 - 1) * self.translation_px
        cosine, sine = math.cos(angle), math.sin(angle)
        # Mapping is [y,x] -> [y,x], centred to avoid a transform around (0,0).
        linear = torch.tensor([[scale * cosine, -scale * sine],
                               [scale * sine, scale * cosine]], dtype=torch.float32)
        center = torch.tensor([(self.height - 1) / 2, (self.width - 1) / 2])
        bias = center + translation - linear @ center
        return torch.cat((linear, bias[:, None]), dim=1)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        generator = self._generator(index)
        fixed_ir = _latent_structure(self.height, self.width, generator)
        fixed_to_moving = self._sample_affine(generator)
        gt_flow = affine_flow_yx(fixed_to_moving, self.height, self.width, fixed_ir)
        inverse_flow = affine_flow_yx(invert_affine_yx(fixed_to_moving),
                                      self.height, self.width, fixed_ir)
        # ``moving_ir(q) = fixed_ir(A^-1 q)``. Therefore warp(moving_ir, A p-p)
        # reconstructs the fixed thermal image exactly away from image borders.
        ir = warp(fixed_ir, inverse_flow)
        visible_base = fixed_ir.clamp(0, 1)
        visible = torch.cat((visible_base.pow(0.7),
                             (0.15 + 0.85 * visible_base).sqrt(),
                             0.25 + 0.75 * visible_base.pow(1.4)), dim=1).clamp(0, 1)
        visible = (visible + 0.01 * torch.randn(visible.shape, generator=generator)).clamp(0, 1)
        valid_mask = valid_affine_mask(fixed_to_moving, self.height, self.width, fixed_ir)
        return {"ir": ir.squeeze(0), "vi": visible.squeeze(0),
                "gt_flow": gt_flow.squeeze(0), "valid_mask": valid_mask,
                "fixed_ir": fixed_ir.squeeze(0), "affine_yx": fixed_to_moving}
