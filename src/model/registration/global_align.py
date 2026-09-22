# -*- coding: utf-8 -*-
"""Cross-modal GLOBAL alignment from a low-resolution global correlation.

Why this module exists
----------------------
Measured on VTMOT (numbers recorded at the bottom of ``sea_raft.py``): the
frozen, mono-modally pretrained SEA-RAFT used zero-shot on the IR/VIS pair has
an EPE of 63 px against an 11.6 px zero-flow baseline -- 5.4x WORSE than simply
predicting no motion -- while on a same-modality pair the very same network
recovers the injected affine to 0.045 px.  The failure is structural, not a
tuning problem.

RAFT-family networks (RAFT, SEA-RAFT) build a full 4-D correlation volume but
only ever LOOK IT UP in a local window of radius ``r`` around the current
estimate, iteratively.  They are refinement networks: they need a good
initialisation and they have no global search stage.  A zero-initialised,
cross-modal pair gives them nothing to refine.

GLU-Net -- the dense-flow network used by SmoothFusion -- is different in exactly
one structural respect that matters here: its first stage correlates features
over the WHOLE image at 1/16 resolution (``ratio = 16 / w_256``), applies mutual
matching and a soft-argmax, and only THEN refines locally with the same +-4 px
constrained correlation that RAFT uses.  That global stage is the ingredient
this pipeline was missing.

This module adds it, and goes one step further.  On VTMOT the target IS a global
affine (the generator used ``persp_amp: 0.0``), so the matched correspondences
are reduced to a 6-DoF affine by a closed-form weighted least-squares fit.  That
keeps the learnable part small (a modality-invariant structure encoder) and the
output dimension minimal (6 numbers instead of ``2*H*W``), which is what makes
it able to converge in dozens of steps rather than not at all.

Conventions match the rest of the package: flow is ``[dy, dx]``, and
``warped(p) = source(p + flow(p))``.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _to_gray


def structure_map(image: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """A modality-tolerant representation: local gradient energy, normalised.

    Per-pixel intensity is the worst possible cross-modal cue (IR and VIS have
    unrelated absolute levels), but the *location of structure* -- edges,
    corners, texture -- is largely shared.  This is the same first-order cue the
    registration loss already relies on.
    """
    gray = _to_gray(image)
    gx = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
    gy = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
    energy = (gx.square() + gy.square() + eps).sqrt()
    mean = energy.mean(dim=(2, 3), keepdim=True)
    std = energy.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (energy - mean) / std


class GlobalCorrelationAlignment(nn.Module):
    """6-DoF (or 2-DoF) global alignment supervised by the dense GT flow.

    ``forward(moving, fixed)`` returns a dict with

      ``flow``         ``[B, 2, H, W]`` affine-induced flow at full resolution
      ``matrix``       ``[B, 3, 3]`` the fitted transform (normalised coords)
      ``confidence``   ``[B, 1, H, W]`` per-pixel match confidence (0..1)
      ``match_flow``   ``[B, 2, H, W]`` dense soft-argmax flow (diagnostic only)
      ``match_peak``   ``[B]`` mean mutual-matching peak (diagnostic)
    """

    def __init__(self, channels: int = 32, mode: str = "affine",
                 temperature: float = 0.07, mutual: bool = True,
                 ridge: float = 1e-3, learned_encoder: bool = True,
                 identity_prior: float = 100.0):
        super().__init__()
        if mode not in {"translation", "affine"}:
            raise ValueError(f"unsupported mode {mode!r}")
        self.mode = mode
        self.mutual = bool(mutual)
        self.ridge = float(ridge)
        # MAP prior centred on the IDENTITY transform, i.e. a convergence
        # regulariser, NOT a fix for the untrained fit.
        # Measured (18x18 token grid): XtWX has diag [12.0, 16.9, 54.5] and
        # condition number 7.6, so the design matrix is well conditioned; sweeping
        # lambda from 0 to 1e4 leaves the untrained singular values at roughly
        # [0.6-0.7, 0.15-0.36] and the initial epe_ratio at ~3.5-4.0 either way.
        # The genuinely large untrained flows (158-379 px at crop 288 / full frame
        # against a 5-7 px target) are therefore a STARTUP TRANSIENT that training
        # removes within ~50 steps, not something a prior can prevent.
        # What the prior does buy is faster early convergence: after 60 overfit
        # steps the epe_ratio was 0.0654 with lambda=0 versus 0.0360 with
        # lambda=100.  100 is a moderate value relative to the ~12-55 diagonal.
        self.identity_prior = float(identity_prior)
        # Both of these MUST be learnable.  A softmax over ~1200 moving tokens is
        # inherently diffuse, so the mutual-matching peak lands far below 1 and a
        # fixed temperature cannot know how sharp the correlation should be for
        # this data.  Log-parameterised so the temperature stays positive.
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(max(float(temperature), 1e-3))))
        self.confidence_bias = nn.Parameter(torch.tensor(3.0))

        # Two stride-4 convs give an exact 1/16 grid for any input size that is a
        # multiple of 16 (208 -> 13, 640x480 -> 40x30), with no rounding games.
        if learned_encoder:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, channels, 4, stride=4, padding=1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 4, stride=4, padding=1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            )
        else:
            self.encoder = None
        self.channels = channels

    # ------------------------------------------------------------------ features
    def _features(self, image: torch.Tensor) -> torch.Tensor:
        structure = structure_map(image)
        if self.encoder is not None:
            features = self.encoder(structure)
        else:
            features = structure
        # L2 normalisation along channels -> the correlation becomes a cosine
        # similarity in [-1, 1], which keeps the softmax temperature meaningful.
        return F.normalize(features, dim=1, eps=1e-4)

    @staticmethod
    def _normalised_grid(height: int, width: int, device, dtype):
        """Token coordinates in [-1, 1], matching grid_sample(align_corners=True)."""
        rows = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        cols = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_v, grid_u = torch.meshgrid(rows, cols, indexing="ij")
        return grid_u, grid_v

    # ------------------------------------------------------------------- forward
    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> Dict[str, torch.Tensor]:
        if moving.shape != fixed.shape:
            raise ValueError(
                f"global alignment needs matching shapes, got {tuple(moving.shape)} "
                f"and {tuple(fixed.shape)}")
        batch, _, height, width = moving.shape
        moving_feat = self._features(moving)          # [B, C, hc, wc]
        fixed_feat = self._features(fixed)
        _, _, hc, wc = fixed_feat.shape
        tokens_f = hc * wc
        tokens_m = moving_feat.shape[2] * moving_feat.shape[3]

        # ---- global all-pairs correlation -------------------------------------
        # This is the part RAFT/SEA-RAFT does not have: every fixed token against
        # every moving token, so the first estimate is unbounded in displacement.
        flat_f = fixed_feat.flatten(2)                 # [B, C, Nf]
        flat_m = moving_feat.flatten(2)                # [B, C, Nm]
        temperature = self.log_temperature.exp().clamp_min(1e-3)
        corr = torch.bmm(flat_f.transpose(1, 2), flat_m) / temperature
        corr = corr.view(batch, hc, wc, tokens_m)

        # ---- mutual matching (GLU-Net / NC-Net style) -------------------------
        row_soft = F.softmax(corr, dim=-1)
        if self.mutual:
            col_soft = F.softmax(corr, dim=1)           # over the fixed tokens
            match = row_soft * col_soft
        else:
            match = row_soft
        match = match.view(batch, tokens_f, tokens_m)
        # Rows normalise to one so the soft-argmax below is a proper expectation.
        weights = match / match.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # Per-correspondence reliability, used BOTH as the least-squares weight
        # (sharp matches should dominate the global fit) and as the blend
        # confidence.  After row normalisation the row *sum* is always 1, so the
        # peak is the only meaningful scalar here.
        peak = weights.max(dim=-1).values               # [B, Nf]

        # ---- soft-argmax -> dense correspondence ------------------------------
        u_m, v_m = self._normalised_grid(
            moving_feat.shape[2], moving_feat.shape[3], corr.device, corr.dtype)
        u_m = u_m.reshape(-1)
        v_m = v_m.reshape(-1)
        expected_u = (weights * u_m).sum(dim=-1)        # [B, Nf]
        expected_v = (weights * v_m).sum(dim=-1)
        u_f, v_f = self._normalised_grid(hc, wc, corr.device, corr.dtype)
        u_f = u_f.reshape(-1)
        v_f = v_f.reshape(-1)

        # ---- closed-form weighted least-squares fit ---------------------------
        # The target on VTMOT is a global affine, so reducing thousands of noisy
        # correspondences to 6 numbers is both the correct model complexity and
        # by far the easiest thing to train.  Differentiable, no extra params.
        if self.mode == "translation":
            matrix = self._fit_translation(expected_u, expected_v, u_f, v_f, peak)
        else:
            matrix = self._fit_affine(expected_u, expected_v, u_f, v_f, peak)

        # ---- dense flow induced by the fitted transform ----------------------
        flow, grid = self._dense_flow(matrix, height, width)
        match_flow = self._match_flow(expected_u, expected_v, u_f, v_f,
                                      hc, wc, height, width)

        # Calibrated so that a diffuse peak still maps into a usable blending
        # weight: logit keeps the ordering, the learnable bias sets the level and
        # can be pushed to ~1 once the matcher is confident.
        peak_clamped = peak.clamp(1e-6, 1.0 - 1e-6)
        calibrated = torch.sigmoid(
            torch.logit(peak_clamped) + self.confidence_bias)
        confidence = calibrated.view(batch, 1, hc, wc)
        confidence = F.interpolate(confidence, size=(height, width),
                                   mode="bilinear", align_corners=True).clamp(0.0, 1.0)
        return {
            "flow": flow,
            "matrix": matrix,
            "confidence": confidence,
            "match_flow": match_flow,
            "match_peak": peak.mean(dim=-1),
            "grid": grid,
        }

    # ------------------------------------------------------------------- fitting
    def _design(self, u_f, v_f):
        return torch.stack([u_f, v_f, torch.ones_like(u_f)], dim=-1)      # [Nf, 3]

    def _solve(self, design, target, weights):
        """Weighted ridge MAP fit: A = (XᵀWX + λI)⁻¹ (XᵀWY + λ·A_prior).

        ``A_prior`` is the identity affine (``u_m = u_f``, ``v_m = v_f``), so with
        a degenerate correlation the solution collapses to "do nothing" instead of
        to a rank-deficient contraction.
        """
        xtwx = torch.einsum("bni,bn,bnj->bij", design, weights, design)
        xtwy = torch.einsum("bni,bn,bnj->bij", design, weights, target)
        # identity prior: column 0 (u_f) predicts expected_u, column 1 (v_f)
        # predicts expected_v, the constant column predicts nothing.
        prior = torch.zeros_like(xtwy)
        prior[:, 0, 0] = 1.0
        prior[:, 1, 1] = 1.0
        lam = self.identity_prior
        eye = torch.eye(design.shape[-1], device=design.device,
                        dtype=design.dtype).unsqueeze(0)
        solution = torch.linalg.solve(
            xtwx + (self.ridge + lam) * eye, xtwy + lam * prior)
        return solution                                                   # [B, 3, 2]

    def _fit_affine(self, expected_u, expected_v, u_f, v_f, weights):
        batch = expected_u.shape[0]
        design = self._design(u_f, v_f).unsqueeze(0).expand(batch, -1, -1)
        target = torch.stack([expected_u, expected_v], dim=-1)            # [B, Nf, 2]
        solution = self._solve(design, target, weights)                   # [B, 3, 2]
        matrix = torch.zeros(batch, 3, 3, device=u_f.device, dtype=u_f.dtype)
        matrix[:, :2, :] = solution.transpose(1, 2)
        matrix[:, 2, 2] = 1.0
        return matrix

    def _fit_translation(self, expected_u, expected_v, u_f, v_f, weights):
        batch = expected_u.shape[0]
        # translation only: the design is a constant 1, so the fit is a weighted mean
        total = weights.sum(dim=-1).clamp_min(1e-8)
        du = ((expected_u - u_f) * weights).sum(dim=-1) / total
        dv = ((expected_v - v_f) * weights).sum(dim=-1) / total
        matrix = torch.zeros(batch, 3, 3, device=u_f.device, dtype=u_f.dtype)
        matrix[:, 0, 0] = 1.0
        matrix[:, 1, 1] = 1.0
        matrix[:, 0, 2] = du
        matrix[:, 1, 2] = dv
        matrix[:, 2, 2] = 1.0
        return matrix

    # --------------------------------------------------------------------- dense
    def flow_from_matrix(self, matrix: torch.Tensor, height: int,
                         width: int) -> torch.Tensor:
        """Public wrapper: turn a ``[B, 3, 3]`` normalised-coordinate transform
        into a ``[B, 2, H, W]`` ``[dy, dx]`` field.

        Exposed because the temporal smoothing (WST) is applied to the MATRIX,
        not to the pixel field: smoothing six numbers is what keeps a global
        transform temporally coherent without blurring a field that no longer
        needs blurring.
        """
        return self._dense_flow(matrix, height, width)[0]

    def _dense_flow(self, matrix, height, width):
        device, dtype = matrix.device, matrix.dtype
        rows = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        cols = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_v, grid_u = torch.meshgrid(rows, cols, indexing="ij")
        ones = torch.ones_like(grid_u)
        homogeneous = torch.stack([grid_u, grid_v, ones], dim=0).reshape(3, -1)
        mapped = torch.einsum("bij,jn->bin", matrix, homogeneous)          # [B,3,HW]
        mapped_u = mapped[:, 0].reshape(-1, height, width)
        mapped_v = mapped[:, 1].reshape(-1, height, width)
        # normalised units [-1,1] -> pixels
        scale_x = (width - 1) / 2.0
        scale_y = (height - 1) / 2.0
        dx = (mapped_u - grid_u) * scale_x
        dy = (mapped_v - grid_v) * scale_y
        flow = torch.stack([dy, dx], dim=1)                                # [dy, dx]
        return flow, (grid_u, grid_v)

    def _match_flow(self, expected_u, expected_v, u_f, v_f, hc, wc,
                    height, width):
        """Diagnostic: the raw dense matching flow, before the global fit."""
        device, dtype = expected_u.device, expected_u.dtype
        du = (expected_u - u_f).reshape(-1, 1, hc, wc)
        dv = (expected_v - v_f).reshape(-1, 1, hc, wc)
        coarse = torch.cat([dv, du], dim=1)                                # [B,2,hc,wc]
        coarse = F.interpolate(coarse, size=(height, width), mode="bilinear",
                               align_corners=True)
        coarse[:, 0] *= (height - 1) / 2.0
        coarse[:, 1] *= (width - 1) / 2.0
        return coarse
