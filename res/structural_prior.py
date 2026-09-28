"""Phase-congruency priors and modality-specific shallow feature adapters.

The fixed prior has a position in the image: oriented, multi-scale analytic
filter responses are combined at each location.  This differs from treating
the magnitude or phase of a whole-image FFT bin as a pixel descriptor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ConvNormAct, ResidualBlock, SharedPyramidEncoder


def phase_congruency_prior(image: torch.Tensor, *, downsample: int = 4,
                           orientations: int = 4,
                           wavelengths: tuple[int, ...] = (3, 6, 12),
                           eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate local phase congruency and normalized log-band energy.

    A symmetric log-Gabor radial filter and orientation envelope select each
    band.  Keeping only one frequency half-plane yields a complex analytic
    response.  The magnitude of the sum of responses divided by the sum of
    magnitudes is the phase-congruency measure, in [0,1].  Sign-inverted image
    contrast leaves both outputs unchanged.
    """
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("phase_congruency_prior expects [B,1,H,W]")
    if downsample < 1 or orientations < 1 or not wavelengths or eps <= 0:
        raise ValueError("invalid phase-congruency parameters")
    if any(wavelength < 2 for wavelength in wavelengths):
        raise ValueError("wavelengths must be at least two pixels")
    height, width = image.shape[-2:]
    if height % downsample or width % downsample:
        raise ValueError("image dimensions must divide phase-congruency downsample")
    reduced = F.avg_pool2d(image.float(), downsample, downsample)
    grid_h, grid_w = reduced.shape[-2:]
    fy = torch.fft.fftfreq(grid_h, device=image.device)
    fx = torch.fft.fftfreq(grid_w, device=image.device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square()).clamp_min(eps)
    spectrum = torch.fft.fft2(reduced.squeeze(1), norm="ortho")
    congruencies = []
    energies = []
    for direction in range(orientations):
        theta = math.pi * direction / orientations
        along = xx * math.cos(theta) + yy * math.sin(theta)
        across = -xx * math.sin(theta) + yy * math.cos(theta)
        angle = torch.atan2(across.abs(), along.abs().clamp_min(eps))
        angular = torch.exp(-0.5 * (angle / (math.pi / orientations)) ** 2)
        # A one-sided spectrum gives even and odd quadrature responses in the
        # real and imaginary parts of one inverse FFT.
        analytic = 1.0 + along.sign()
        sum_response = torch.zeros_like(reduced.squeeze(1), dtype=torch.complex64)
        sum_amplitude = torch.zeros_like(reduced.squeeze(1))
        for wavelength in wavelengths:
            center = 1.0 / wavelength
            radial = torch.exp(-0.5 * (
                torch.log(radius / center) / math.log(0.65)) ** 2)
            radial = radial.masked_fill((xx == 0) & (yy == 0), 0.0)
            response = torch.fft.ifft2(
                spectrum * (radial * angular * analytic), norm="ortho")
            sum_response = sum_response + response
            sum_amplitude = sum_amplitude + response.abs()
        congruencies.append(sum_response.abs() / sum_amplitude.clamp_min(eps))
        energies.append(sum_amplitude)
    congruency = torch.stack(congruencies, dim=0).mean(dim=0).unsqueeze(1)
    amplitude = torch.log1p(torch.stack(energies, dim=0).mean(dim=0)).unsqueeze(1)
    # Normalize the energy map per image so its dynamic range does not simply
    # reveal the infrared/visible sensor gain.
    mean = amplitude.mean(dim=(-2, -1), keepdim=True)
    std = amplitude.std(dim=(-2, -1), keepdim=True, unbiased=False)
    amplitude = torch.sigmoid((amplitude - mean) / std.clamp_min(eps))
    if downsample != 1:
        congruency = F.interpolate(congruency, size=(height, width),
                                   mode="bilinear", align_corners=False)
        amplitude = F.interpolate(amplitude, size=(height, width),
                                  mode="bilinear", align_corners=False)
    return congruency.clamp(0, 1), amplitude


def structural_prior_input(image: torch.Tensor, *, downsample: int = 4,
                           orientations: int = 4,
                           wavelengths: tuple[int, ...] = (3, 6, 12)) -> torch.Tensor:
    """Fixed three-channel [standardized gray, congruency, amplitude] input."""
    with torch.no_grad():
        congruency, amplitude = phase_congruency_prior(
            image, downsample=downsample, orientations=orientations,
            wavelengths=wavelengths)
    return torch.cat((image, congruency.to(image.dtype),
                      amplitude.to(image.dtype)), dim=1)


class StructuralPriorEncoder(nn.Module):
    """IR/VI-specific shallow encoders followed by one shared CNN pyramid."""

    def __init__(self, base_channels: int = 24, out_channels: int = 64,
                 blocks_per_scale: int = 2, prior_downsample: int = 4,
                 orientations: int = 4,
                 wavelengths: tuple[int, ...] = (3, 6, 12)) -> None:
        super().__init__()
        if base_channels < 1 or out_channels < 1:
            raise ValueError("encoder channel counts must be positive")
        self.prior_downsample = int(prior_downsample)
        self.orientations = int(orientations)
        self.wavelengths = tuple(map(int, wavelengths))
        self.ir_shallow = nn.Sequential(
            ConvNormAct(3, base_channels), ResidualBlock(base_channels))
        self.vi_shallow = nn.Sequential(
            ConvNormAct(3, base_channels), ResidualBlock(base_channels))
        self.shared = SharedPyramidEncoder(
            in_channels=base_channels, base_channels=base_channels,
            out_channels=out_channels, blocks_per_scale=blocks_per_scale,
            normalise=False)

    def _input(self, image: torch.Tensor) -> torch.Tensor:
        return structural_prior_input(
            image, downsample=self.prior_downsample,
            orientations=self.orientations, wavelengths=self.wavelengths)

    def encode_scales(self, ir: torch.Tensor,
                      vi: torch.Tensor) -> tuple[dict, dict]:
        """Every shared-pyramid scale for both modalities (1/2, 1/4 and 1/8).

        The coarse route only ever needed 1/8, so it was the only scale returned;
        a 1/4 local stage has to reach the same pyramid without re-running the
        shallow adapters, which would put the two stages in different feature
        spaces.
        """
        if ir.shape != vi.shape or ir.ndim != 4 or ir.shape[1] != 1:
            raise ValueError("encoder expects equal grayscale [B,1,H,W] images")
        ir_shallow = self.ir_shallow(self._input(ir))
        vi_shallow = self.vi_shallow(self._input(vi))
        return self.shared.encode_pair(ir_shallow, vi_shallow)

    def forward(self, ir: torch.Tensor,
                vi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Coarse 1/8 features only, kept for callers that need just that scale."""
        ir_features, vi_features = self.encode_scales(ir, vi)
        return ir_features["1/8"], vi_features["1/8"]
