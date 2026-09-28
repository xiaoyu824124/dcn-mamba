"""Local spectral cues on the 1/8 image lattice for cross-modal matching.

A whole-image FFT has no per-pixel correspondence coordinate.  Here each
1/8 location owns a small, windowed neighbourhood, so its amplitude and
phase remain attached to that location.  The fixed descriptor is followed by
shared, trainable projections; IR and VI use the same weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvNormAct


def local_spectrum(image: torch.Tensor, window_size: int = 5,
                   eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log amplitude and (sin phase, cos phase) at each 1/8 location.

    The DC bin is zeroed after patch mean subtraction.  A Hann window limits
    edge leakage.  FFT is evaluated in float32 because CUDA half precision FFT
    does not support arbitrary window lengths under automatic mixed precision.
    """
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("local_spectrum expects [B,1,H,W]")
    if window_size < 3 or window_size % 2 != 1:
        raise ValueError("window_size must be odd and at least 3")
    if eps <= 0:
        raise ValueError("eps must be positive")
    height, width = image.shape[-2:]
    if height % 8 or width % 8:
        raise ValueError("local_spectrum needs image dimensions divisible by 8")
    # A spectrum at every full-resolution pixel would be very expensive and
    # would then have to be pooled, destroying its spatial interpretation.
    reduced = F.avg_pool2d(image.float(), kernel_size=8, stride=8)
    grid_h, grid_w = reduced.shape[-2:]
    patches = F.unfold(reduced, kernel_size=window_size,
                       padding=window_size // 2)
    patches = patches.transpose(1, 2).reshape(-1, window_size, window_size)
    patches = patches - patches.mean(dim=(-2, -1), keepdim=True)
    hann = torch.hann_window(window_size, periodic=False,
                             device=image.device, dtype=torch.float32)
    spectrum = torch.fft.rfft2(patches * hann[:, None] * hann[None, :],
                               norm="ortho")
    magnitude = spectrum.abs()
    amplitude = torch.log1p(magnitude)
    unit = spectrum / magnitude.clamp_min(eps)
    bins = window_size * (window_size // 2 + 1)
    batch = image.shape[0]

    def as_map(values: torch.Tensor) -> torch.Tensor:
        return values.reshape(batch, grid_h * grid_w, bins).transpose(1, 2).reshape(
            batch, bins, grid_h, grid_w)

    # Phase of a zero-energy bin is undefined.  Dividing by clamped magnitude
    # maps it to zero, instead of claiming cos(phase)=1 at such locations.
    return as_map(amplitude), torch.cat((as_map(unit.imag), as_map(unit.real)), dim=1)


class SpatialFrequencyFusion(nn.Module):
    """Ablate spatial, local amplitude, local phase, or their learned fusion."""

    MODES = ("spatial", "amplitude", "phase", "fused")

    def __init__(self, channels: int, mode: str = "fused",
                 window_size: int = 5) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown spatial-frequency mode: {mode}")
        if window_size < 3 or window_size % 2 != 1:
            raise ValueError("window_size must be odd and at least 3")
        self.mode = mode
        self.window_size = int(window_size)
        bins = window_size * (window_size // 2 + 1)
        if mode in ("amplitude", "fused"):
            self.amplitude_encoder = nn.Sequential(
                ConvNormAct(bins, channels), ConvNormAct(channels, channels))
        if mode in ("phase", "fused"):
            self.phase_encoder = nn.Sequential(
                ConvNormAct(2 * bins, channels), ConvNormAct(channels, channels))
        if mode == "fused":
            self.cross_domain_fusion = nn.Sequential(
                ConvNormAct(3 * channels, channels),
                nn.Conv2d(channels, channels, 1))
            # The spatial branch can be warm-started from the existing
            # checkpoint; the new random branch begins with a modest effect.
            self.fusion_gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, image: torch.Tensor,
                spatial: torch.Tensor) -> torch.Tensor:
        if image.shape[0] != spatial.shape[0] or (
                image.shape[-2] // 8, image.shape[-1] // 8) != spatial.shape[-2:]:
            raise ValueError("spatial feature must be on the image's 1/8 lattice")
        if self.mode == "spatial":
            return spatial
        amplitude, phase = local_spectrum(image, self.window_size)
        if self.mode == "amplitude":
            return self.amplitude_encoder(amplitude.to(spatial.dtype))
        if self.mode == "phase":
            return self.phase_encoder(phase.to(spatial.dtype))
        amp_feature = self.amplitude_encoder(amplitude.to(spatial.dtype))
        phase_feature = self.phase_encoder(phase.to(spatial.dtype))
        joined = torch.cat((spatial, amp_feature, phase_feature), dim=1)
        return spatial + self.fusion_gain * self.cross_domain_fusion(joined)
