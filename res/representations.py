# -*- coding: utf-8 -*-
"""Candidate structural representations for cross-modal IR--visible matching.

Every representation maps a one-channel image ``[B, 1, H, W]`` (float, any
range) to a feature map ``[B, C, H, W]``.  Downsampling, channel normalisation
and matching are deliberately NOT part of these functions:
``representation_bench`` applies one identical protocol to every candidate so
the comparison stays controlled.

Why a *discriminability* benchmark and not an *invariance* benchmark
-------------------------------------------------------------------
Modality invariance alone does not make a representation usable for matching.
A descriptor can be perfectly invariant to the IR/VIS intensity gap and still be
useless, if the invariance was bought by collapsing every pixel onto nearly the
same vector: then all candidates tie and the argmax is arbitrary.

Measured on real VTMOT data, the stock MIND descriptor has an affine-intensity
invariance of 1.0000 and a contrast-reversal invariance of 1.0000 -- yet the
cosine of the best correspondence and of the second-best differ by only 0.0005,
because the whole score matrix lives in ``[0.94, 0.999]``.  Feeding that into a
1200- or 4800-way softmax produces a flat distribution, and the soft-argmax
collapses onto the image centre (~150 px, peak 0.004 against a 0.0008 uniform).

The quantity that decides whether a 1/8 all-pairs softmax can form a peak is
therefore the temperature-free margin ``cos(top1) - cos(top2)``, which is what
this module's features are benchmarked on.

Channel-count caveat
--------------------
Per-pixel L2 normalisation across channels (``F.normalize(dim=1)``) is only
meaningful for ``C > 1``.  For a one-channel map it reduces to the sign of the
value, which destroys all magnitude information.  The registry therefore marks
the one-channel candidates as ``l2=False`` and the benchmark skips the channel
normalisation for them.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "gaussian_blur", "central_difference", "standardise",
    "feat_raw", "feat_grad", "feat_orient", "feat_orient3", "feat_mind",
    "feat_mind_fixed", "feat_lss", "feat_census", "feat_rank",
    "feat_monogenic", "feat_pc", "patchify", "REPRESENTATIONS",
]


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with replicate padding.  ``[B,C,H,W] -> same``."""
    if sigma <= 0:
        return x
    channels = x.shape[1]
    kernel = _gaussian_kernel1d(sigma, x.device, x.dtype)
    radius = kernel.numel() // 2
    kx = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"),
                 kx, groups=channels)
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"),
                 ky, groups=channels)
    return x


def central_difference(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pixel differences ``(d/dx, d/dy)`` with replicate borders.

    ``dx`` differentiates along W and ``dy`` along H, so stacking the pair as
    ``(dy, dx)`` matches the project-wide flow convention.
    """
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")
    return dx, dy


def standardise(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image zero-mean, unit-variance normalisation over ``(C, H, W)``."""
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """``x(p + [dy, dx])`` with replicate borders."""
    pad = max(abs(dy), abs(dx))
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    h, w = x.shape[-2:]
    return padded[..., pad + dy: pad + dy + h, pad + dx: pad + dx + w]


def _offset_list(radius: int) -> List[Tuple[int, int]]:
    return [(dy, dx) for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1) if (dy, dx) != (0, 0)]


def _ring_offsets(radii) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    for r in radii:
        offsets += [(r, 0), (-r, 0), (0, r), (0, -r),
                    (r, r), (r, -r), (-r, r), (-r, -r)]
    return offsets


def _avg_pool(x: torch.Tensor, kernel: int) -> torch.Tensor:
    return F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2,
                        count_include_pad=False)


# --------------------------------------------------------------------------- #
# 1. raw grayscale
# --------------------------------------------------------------------------- #
def feat_raw(x: torch.Tensor) -> torch.Tensor:
    """Standardised grayscale.  ``[B,1,H,W]``.  The do-nothing baseline."""
    return standardise(x)


# --------------------------------------------------------------------------- #
# 2. gradient magnitude
# --------------------------------------------------------------------------- #
def feat_grad(x: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude, standardised.  ``[B,1,H,W]``."""
    dx, dy = central_difference(x)
    return standardise(torch.sqrt(dx * dx + dy * dy + 1e-8))


# --------------------------------------------------------------------------- #
# 3. gradient orientation (doubled angle, anisotropy weighted)
# --------------------------------------------------------------------------- #
def _structure_tensor(x: torch.Tensor, sigma: float):
    dx, dy = central_difference(x)
    jxx = gaussian_blur(dx * dx, sigma)
    jyy = gaussian_blur(dy * dy, sigma)
    jxy = gaussian_blur(dx * dy, sigma)
    trace = (jxx + jyy).clamp_min(1e-8)
    # (jxx-jyy)/trace = anisotropy * cos(2 theta);  2*jxy/trace = anisotropy * sin(2 theta)
    return (jxx - jyy) / trace, (2.0 * jxy) / trace


def feat_orient(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Orientation as the doubled-angle vector.  ``[B,2,H,W]``.

    Doubling the angle removes the ``theta`` / ``theta + pi`` ambiguity, so the
    encoding is unchanged by a contrast reversal: a polarity flip rotates every
    gradient by ``pi`` and ``(cos 2theta, sin 2theta)`` is invariant to that.
    The vector vanishes where no orientation is defined (flat or isotropic
    regions), which is the honest answer and doubles as a confidence signal.
    """
    p, q = _structure_tensor(x, sigma)
    return torch.cat([p, q], dim=1)


def feat_orient3(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Doubled-angle vector plus the anisotropy magnitude.  ``[B,3,H,W]``."""
    p, q = _structure_tensor(x, sigma)
    anisotropy = torch.sqrt(p * p + q * q + 1e-12)
    return torch.cat([p, q, anisotropy], dim=1)


# --------------------------------------------------------------------------- #
# 4. MIND -- the project implementation, loaded by path
# --------------------------------------------------------------------------- #
_MIND_CACHE: Dict[Tuple, nn.Module] = {}


def _load_mind_class():
    """Import ``res/mind.py`` by file path.

    Loading by path keeps this benchmark independent of whether the rest of the
    ``res`` package imports cleanly, so a work-in-progress module elsewhere
    cannot break the comparison.
    """
    path = pathlib.Path(__file__).with_name("mind.py")
    spec = importlib.util.spec_from_file_location("_bench_mind", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load MIND descriptor from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MINDDescriptor


def feat_mind(x: torch.Tensor, patch_size: int = 3,
              radius: int = 1) -> torch.Tensor:
    """Stock MIND descriptor.  ``[B,8,H,W]`` for the default radius."""
    key = (patch_size, radius, str(x.device), str(x.dtype))
    module = _MIND_CACHE.get(key)
    if module is None:
        module = _load_mind_class()(patch_size=patch_size, radius=radius)
        module = module.to(device=x.device, dtype=torch.float32).eval()
        _MIND_CACHE[key] = module
    return module(x.float())


# --------------------------------------------------------------------------- #
# 5. MIND with local-variance normalisation and a flat-region gate
# --------------------------------------------------------------------------- #
def feat_mind_fixed(x: torch.Tensor, patch_size: int = 3, radius: int = 1,
                    tau: float = 0.05) -> torch.Tensor:
    """MIND variant that normalises by local patch variance instead of the mean
    of the eight SSDs, and that gates genuinely flat regions.

    The stock implementation divides by ``mean_c(SSD_c)``, which forces the eight
    distances to average exactly 1 at every pixel.  In a flat region all eight
    SSDs are noise of similar magnitude, so the ratio stays ``O(1)`` and the
    descriptor becomes a *stable but meaningless* unit vector -- measured
    ``cos(flat, flat') = 0.997`` between two independent noise realisations.
    Dividing by the local patch variance of the image keeps the absolute scale
    instead, and the sigmoid gate drives truly flat pixels towards zero
    magnitude so the matcher can recognise them as uninformative.

    The returned map is deliberately NOT L2 normalised here: the benchmark's
    protocol applies one identical channel normalisation to every candidate.
    """
    window = 2 * radius + 3
    ssd = torch.cat([
        _avg_pool((x - _shift(x, dy, dx)).square(), patch_size)
        for dy, dx in _offset_list(radius)
    ], dim=1)
    mean = _avg_pool(x, window)
    variance = (_avg_pool(x * x, window) - mean * mean).clamp_min(0.0)
    floor = float(tau) ** 2
    descriptor = torch.exp(-ssd / variance.clamp_min(floor))
    gate = torch.sigmoid(8.0 * (variance / floor - 1.0))
    return descriptor * gate


# --------------------------------------------------------------------------- #
# 6. Local Self-Similarity (LSS)
# --------------------------------------------------------------------------- #
def feat_lss(x: torch.Tensor, patch_size: int = 3,
             radii=(2, 4)) -> torch.Tensor:
    """Local Self-Similarity, ``[B,8*len(radii),H,W]``.

    This is the direct ancestor of MIND: patch self-similarity against a ring of
    larger offsets, normalised by the maximum SSD in the window (the LSS
    convention) rather than by the mean (the MIND convention).
    """
    ssd = torch.cat([
        _avg_pool((x - _shift(x, dy, dx)).square(), patch_size)
        for dy, dx in _ring_offsets(radii)
    ], dim=1)
    denominator = ssd.max(dim=1, keepdim=True).values.clamp_min(1e-6)
    return torch.exp(-ssd / denominator)


# --------------------------------------------------------------------------- #
# 7. Census and Rank
# --------------------------------------------------------------------------- #
def feat_census(x: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Census transform: centre vs each neighbour.  ``[B,8,H,W]`` in ``{0,1}``.

    Invariant to any strictly monotonic intensity mapping, which is a stronger
    guarantee than MIND's SSD (measured ``cos = 0.9952`` under ``I**0.5``).  The
    price is quantisation: the pattern is a bit string, so per-pixel similarity
    is coarse and flat regions produce noise.
    """
    bits = [(x > _shift(x, dy, dx)).to(x.dtype)
            for dy, dx in _offset_list(radius)]
    return torch.cat(bits, dim=1)


def feat_rank(x: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Rank transform: fraction of neighbours below the centre.  ``[B,1,H,W]``."""
    offsets = _offset_list(radius)
    rank = sum((x > _shift(x, dy, dx)).to(x.dtype) for dy, dx in offsets)
    return rank / float(len(offsets))


# --------------------------------------------------------------------------- #
# 8. Monogenic signal / phase congruency
# --------------------------------------------------------------------------- #
def _log_gabor(h: int, w: int, wavelength: float, sigma_onf: float,
               device, dtype) -> torch.Tensor:
    fy = torch.fft.fftfreq(h, device=device, dtype=dtype).view(-1, 1)
    fx = torch.fft.fftfreq(w, device=device, dtype=dtype).view(1, -1)
    radius = torch.sqrt(fx * fx + fy * fy)
    radius = radius.clone()
    radius[0, 0] = 1.0                       # placeholder; DC is removed below
    f0 = 1.0 / float(wavelength)
    log_sigma = math.log(float(sigma_onf))
    transfer = torch.exp(-(torch.log(radius / f0) ** 2) / (2.0 * log_sigma ** 2))
    transfer[0, 0] = 0.0
    return transfer


def _monogenic_components(x: torch.Tensor, wavelength: float,
                          sigma_onf: float = 0.65, eps: float = 1e-6):
    """Bandpass (even) and the two Riesz components (odd1, odd2) of the
    monogenic signal, plus the local amplitude."""
    h, w = x.shape[-2:]
    spectrum = torch.fft.fft2(x.float(), dim=(-2, -1))
    transfer = _log_gabor(h, w, wavelength, sigma_onf, x.device, torch.float32)
    band = spectrum * transfer
    fy = torch.fft.fftfreq(h, device=x.device, dtype=torch.float32).view(-1, 1)
    fx = torch.fft.fftfreq(w, device=x.device, dtype=torch.float32).view(1, -1)
    radius = torch.sqrt(fx * fx + fy * fy).clamp_min(1e-12)
    h1 = -1j * fx / radius
    h2 = -1j * fy / radius
    h1[0, 0] = 0.0
    h2[0, 0] = 0.0
    even = torch.fft.ifft2(band, dim=(-2, -1)).real
    odd1 = torch.fft.ifft2(band * h1, dim=(-2, -1)).real
    odd2 = torch.fft.ifft2(band * h2, dim=(-2, -1)).real
    amplitude = torch.sqrt(even * even + odd1 * odd1 + odd2 * odd2 + eps)
    return even, odd1, odd2, amplitude


def feat_monogenic(x: torch.Tensor,
                   wavelengths=(4.0, 12.0)) -> torch.Tensor:
    """Unit monogenic phase vector, concatenated over scales.  ``[B,3S,H,W]``.

    ``(even, odd1, odd2) / amplitude`` is a unit vector that depends on the local
    phase rather than on the local contrast, which is exactly the
    "contrast-invariant but spatially sharp" combination that makes phase
    congruency attractive for IR/VIS.  This is the cheap route to it: two Riesz
    multiplications in the Fourier domain instead of a multi-orientation
    log-Gabor filter bank, so the cost is close to one ordinary convolution.
    """
    parts = []
    for wavelength in wavelengths:
        even, odd1, odd2, amplitude = _monogenic_components(x, wavelength)
        parts += [even / amplitude, odd1 / amplitude, odd2 / amplitude]
    return torch.cat(parts, dim=1)


def feat_pc(x: torch.Tensor, wavelength: float = 4.0) -> torch.Tensor:
    """Rectified local phase cosine ``max(0, even/amplitude)``.  ``[B,1,H,W]``."""
    even, _, _, amplitude = _monogenic_components(x, wavelength)
    return F.relu(even / amplitude)


# --------------------------------------------------------------------------- #
# 9. patch descriptor wrapper
# --------------------------------------------------------------------------- #
def patchify(x: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Stack shifted copies of a map into a local patch descriptor.

    ``[B,C,H,W] -> [B, C*(2r+1)^2, H, W]``.

    This wrapper is what makes the one-channel candidates comparable at all.  An
    all-pairs cosine needs a descriptor *vector* per pixel; for a one-channel map
    the "correlation" degenerates into ``a[p] * b[q]``, an outer product of two
    scalars with no notion of similarity.  Patchifying restores a real vector.
    """
    return torch.cat([_shift(x, dy, dx)
                      for dy in range(-radius, radius + 1)
                      for dx in range(-radius, radius + 1)], dim=1)


def _patch(base_fn, radius: int = 1):
    def wrapped(x: torch.Tensor) -> torch.Tensor:
        return patchify(base_fn(x), radius)
    wrapped.__name__ = f"patch_{getattr(base_fn, '__name__', 'map')}"
    return wrapped


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
# ``l2`` records whether per-pixel channel normalisation is meaningful.  It is
# False for every one-channel candidate, where F.normalize would reduce the map
# to its sign.
#
# ``cosine_ok`` records whether the map can be used with an all-pairs cosine
# matcher at all.  A one-channel map cannot; use its ``*_patch`` variant, which
# patches it into a genuine descriptor vector.
REPRESENTATIONS: Dict[str, Dict] = {
    "raw_patch":   {"fn": _patch(feat_raw),   "l2": True,  "family": "intensity"},
    "grad_patch":  {"fn": _patch(feat_grad),  "l2": True,  "family": "gradient"},
    "orient":      {"fn": feat_orient,        "l2": True,  "family": "orientation"},
    "orient3":     {"fn": feat_orient3,       "l2": False, "family": "orientation"},
    "mind":        {"fn": feat_mind,          "l2": True,  "family": "self-similarity"},
    "mind_fixed":  {"fn": feat_mind_fixed,    "l2": True,  "family": "self-similarity"},
    "mind_nol2":   {"fn": feat_mind,          "l2": False, "family": "self-similarity"},
    "lss":         {"fn": feat_lss,           "l2": True,  "family": "self-similarity"},
    "census":      {"fn": feat_census,        "l2": True,  "family": "ordinal"},
    "rank_patch":  {"fn": _patch(feat_rank),  "l2": True,  "family": "ordinal"},
    "monogenic":   {"fn": feat_monogenic,     "l2": True,  "family": "phase"},
    "pc_patch":    {"fn": _patch(feat_pc),    "l2": True,  "family": "phase"},
    # One-channel maps: reported by the invariance probes, not usable with the
    # all-pairs cosine matcher.
    "raw":         {"fn": feat_raw,    "l2": False, "family": "intensity", "cosine_ok": False},
    "grad":        {"fn": feat_grad,   "l2": False, "family": "gradient",  "cosine_ok": False},
    "rank":        {"fn": feat_rank,   "l2": False, "family": "ordinal",   "cosine_ok": False},
    "pc":          {"fn": feat_pc,     "l2": False, "family": "phase",     "cosine_ok": False},
}

