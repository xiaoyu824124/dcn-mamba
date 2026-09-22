"""Reliability-aware cross-modal motion memory.

The registration field maps a visible-frame pixel to its infrared sampling
location.  A field stored at frame ``s`` therefore cannot be propagated with
infrared motion alone: its *domain* first has to move with visible motion and
its sampled infrared location then has to move with infrared motion.  This
module implements that composition and keeps only a small, reliability-gated
set of historical fields.

All flows use the project convention ``[dy, dx]`` and are backward sampling
fields.  Memory state is deliberately represented by ordinary dictionaries,
instead of module buffers, so one registration call never leaks state into a
different video.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SpatialTransformer, _to_gray


MemoryEntry = Dict[str, torch.Tensor]


def dual_modal_transport(
        transformer: SpatialTransformer,
        field: torch.Tensor,
        visible_backward: torch.Tensor,
        infrared_forward: torch.Tensor) -> torch.Tensor:
    """Move a visible-to-infrared field from the previous frame to current.

    ``visible_backward`` maps a current visible coordinate to the previous
    visible coordinate. ``infrared_forward`` maps a previous infrared
    coordinate to the current infrared coordinate.  The resulting field is
    the discrete counterpart of ``T_ir(s->t) o Phi_s o T_vis(t->s)``.
    """
    previous_field_at_current = transformer(field, visible_backward)[0]
    previous_ir_offset = visible_backward + previous_field_at_current
    infrared_increment = transformer(
        infrared_forward, previous_ir_offset)[0]
    return previous_ir_offset + infrared_increment


class TemporalReliabilityCalibrator(nn.Module):
    """Convert geometry and cross-modal alignment cues into trust maps."""

    def __init__(self, channels: int = 16):
        super().__init__()
        hidden = max(8, int(channels))
        # Directional gradient agreement is much less brittle across IR/VIS
        # than raw intensity difference.  The other maps expose temporal
        # disagreement and memory age to the learned calibrator.
        self.body = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        # Start from the inherited confidence, so enabling memory does not
        # destabilise a checkpoint before the calibrator has been trained.
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_x.t().reshape(1, 1, 3, 3))

    def _gradient_error(self, moving, fixed, flow, transformer):
        moving = _to_gray(moving)
        fixed = _to_gray(fixed)
        warped = transformer(moving, flow)[0]
        gx_m = F.conv2d(warped, self.sobel_x, padding=1)
        gy_m = F.conv2d(warped, self.sobel_y, padding=1)
        gx_f = F.conv2d(fixed, self.sobel_x, padding=1)
        gy_f = F.conv2d(fixed, self.sobel_y, padding=1)
        inner = gx_m * gx_f + gy_m * gy_f
        norm_m = gx_m.square() + gy_m.square() + 1e-4
        norm_f = gx_f.square() + gy_f.square() + 1e-4
        return 1.0 - inner.square() / (norm_m * norm_f)

    def forward(self, moving, fixed, flow, inherited, disagreement, age,
                transformer, history_feature):
        inherited = inherited.clamp(0.01, 0.99)
        if age.ndim == 1:
            age = age[:, None, None, None]
        age = age.to(dtype=flow.dtype, device=flow.device).expand_as(inherited)
        features = torch.cat([
            self._gradient_error(moving, fixed, flow, transformer),
            (history_feature - _to_gray(fixed)).abs() /
            (history_feature.abs() + _to_gray(fixed).abs() + 0.1),
            inherited,
            disagreement,
            age / 16.0,
            flow.norm(dim=1, keepdim=True).tanh(),
        ], dim=1)
        return torch.sigmoid(torch.logit(inherited) + self.body(features))


class TrustedMotionMemory(nn.Module):
    """Bounded, reliability-gated geometric memory for one video sequence."""

    def __init__(self, capacity=3, age_decay=0.98, channels=16):
        super().__init__()
        self.capacity = max(1, int(capacity))
        self.age_decay = float(age_decay)
        self.transformer = SpatialTransformer()
        self.calibrator = TemporalReliabilityCalibrator(channels=channels)

    @staticmethod
    def _entry(flow, reliability, valid=None, age=None, anchor=False,
               feature=None):
        if valid is None:
            valid = torch.ones_like(reliability)
        if age is None:
            age = torch.zeros(
                flow.shape[0], dtype=torch.long, device=flow.device)
        return {
            "flow": flow,
            "reliability": reliability,
            "valid": valid,
            "age": age,
            "anchor": torch.full(
                (flow.shape[0],), bool(anchor), dtype=torch.bool,
                device=flow.device),
            "feature": feature,
        }

    @staticmethod
    def appearance_feature(image):
        """A compact, modality-local keyframe descriptor in VIS coordinates."""
        gray = _to_gray(image)
        local_mean = F.avg_pool2d(gray, 5, stride=1, padding=2)
        return (gray - local_mean).abs()

    def initialize(self, flow, reliability, feature):
        """Create the first, immediately trusted, keyframe entry."""
        return [self._entry(
            flow.detach(), reliability.detach(), anchor=True,
            feature=feature.detach())]

    def advance(self, entries: Sequence[MemoryEntry], visible_backward,
                infrared_forward) -> List[MemoryEntry]:
        """Transport all stored fields to the current frame and age them."""
        advanced = []
        for entry in entries:
            flow = dual_modal_transport(
                self.transformer, entry["flow"], visible_backward,
                infrared_forward).detach()
            reliability = self.transformer(
                entry["reliability"], visible_backward)[0].detach()
            valid = self.transformer(entry["valid"], visible_backward)[0].detach()
            feature = self.transformer(
                entry["feature"], visible_backward)[0].detach()
            advanced.append({
                "flow": flow,
                "reliability": (reliability * self.age_decay).clamp(0.0, 1.0),
                "valid": valid.clamp(0.0, 1.0),
                "age": entry["age"] + 1,
                "anchor": entry["anchor"],
                "feature": feature,
            })
        return advanced

    @staticmethod
    def _disagreement(flows, valid):
        """Per-candidate deviation from the valid weighted consensus."""
        weights = valid.clamp_min(1e-4)
        consensus = (flows * weights).sum(dim=1)
        consensus = consensus / weights.sum(dim=1).clamp_min(1e-4)
        return (flows - consensus.unsqueeze(1)).norm(dim=2, keepdim=True).tanh()

    def _score(self, moving, fixed, entries):
        flows = torch.stack([entry["flow"] for entry in entries], dim=1)
        inherited = torch.stack(
            [entry["reliability"] for entry in entries], dim=1)
        valid = torch.stack([entry["valid"] for entry in entries], dim=1)
        ages = torch.stack([entry["age"] for entry in entries], dim=1)
        disagreement = self._disagreement(flows, valid)
        reliability = []
        for index, entry in enumerate(entries):
            reliability.append(self.calibrator(
                moving, fixed, flows[:, index], inherited[:, index],
                disagreement[:, index], ages[:, index], self.transformer,
                entry["feature"])
                * valid[:, index])
        return flows, torch.stack(reliability, dim=1), valid, disagreement

    def fuse(self, moving, fixed, entries: Sequence[MemoryEntry],
             fallback_flow, fallback_reliability):
        """Read memory with a direct previous-frame candidate as a safe floor."""
        fallback = self._entry(
            fallback_flow, fallback_reliability,
            age=torch.ones(
                fallback_flow.shape[0], dtype=torch.long,
                device=fallback_flow.device),
            feature=self.appearance_feature(fixed).detach())
        candidates = list(entries) + [fallback]
        flows, reliability, valid, disagreement = self._score(
            moving, fixed, candidates)
        weights = (reliability * valid).clamp_min(1e-5)
        fused = (flows * weights).sum(dim=1)
        fused = fused / weights.sum(dim=1).clamp_min(1e-5)
        # Confidence represents the best supported geometric hypothesis, not
        # the number of slots. Summing would make a bad memory look certain
        # merely because it contains several mutually inconsistent entries.
        confidence = weights.max(dim=1).values
        diagnostics = {
            "disagreement": disagreement[:, :-1].mean(dim=1),
            "slots": torch.full(
                (fused.shape[0],), len(entries), dtype=torch.long,
                device=fused.device),
        }
        return fused, confidence, diagnostics

    def assess(self, moving, fixed, entry: MemoryEntry):
        """Re-evaluate a quarantined keyframe before admitting it."""
        _, reliability, valid, _ = self._score(moving, fixed, [entry])
        reliability = reliability[:, 0]
        score = (reliability * valid[:, 0]).sum(dim=(1, 2, 3))
        score = score / valid[:, 0].sum(dim=(1, 2, 3)).clamp_min(1e-5)
        updated = dict(entry)
        updated["reliability"] = reliability.detach()
        return updated, score

    def admit(self, entries: Sequence[MemoryEntry], entry: MemoryEntry):
        """Write one confirmed entry, replacing only the weakest non-anchor."""
        detached = {
            key: (value.detach() if torch.is_tensor(value) else value)
            for key, value in entry.items()
        }
        updated = list(entries)
        if len(updated) < self.capacity:
            return updated + [detached]
        replaceable = [
            index for index, item in enumerate(updated)
            if not bool(item["anchor"].all())]
        if not replaceable:
            # Capacity one intentionally retains the first reliable anchor.
            return updated
        scores = [
            (updated[index]["reliability"] * updated[index]["valid"])
            .mean().item() for index in replaceable]
        updated[replaceable[int(torch.tensor(scores).argmin())]] = detached
        return updated
