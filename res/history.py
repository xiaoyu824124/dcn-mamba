"""Reliable state carried from a keyframe to later frames.

Propagating a keyframe result to the current frame is a **composition**, not a
copy and not an average of neighbouring flows:

    p  (VI_t grid)
      -> q = p + flow_vi_t_to_vi_k(p)      (VI_k)   same-modality frame motion
      -> r = q + keyframe_flow(q)          (IR_k)   the keyframe registration
      -> s = r + flow_ir_k_to_ir_t(r)      (IR_t)   same-modality frame motion
    init_flow(p) = s - p

Each step samples the next field at the coordinate the previous step produced, so
camera or object motion between the frames is accounted for instead of assuming
the misalignment is static.  All fields are image-grid ``[B,2,H,W]`` ``[dy,dx]``
pixel displacements following the project convention

    aligned(p) = moving(p + flow(p)),

and every stage carries a validity mask so occluded or out-of-frame points never
enter the history cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .warp import warp


def _base_grid(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    y = torch.arange(height, device=reference.device, dtype=reference.dtype)
    x = torch.arange(width, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy, xx), dim=-1).unsqueeze(0)


def _in_bounds(coordinates: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return ((coordinates[..., 0] >= 0) & (coordinates[..., 0] <= height - 1)
            & (coordinates[..., 1] >= 0) & (coordinates[..., 1] <= width - 1))


def sample_field(field: torch.Tensor, coordinates_yx: torch.Tensor,
                 padding_mode: str = "border") -> torch.Tensor:
    """Sample a ``[B,2,H,W]`` field at ``[B,H,W,2]`` pixel coordinates."""
    if field.ndim != 4 or field.shape[1] != 2:
        raise ValueError("field must be [B,2,H,W] in [dy,dx] order")
    base = _base_grid(field.shape[-2], field.shape[-1], field)
    return warp(field, coordinates_yx.permute(0, 3, 1, 2) - base.permute(0, 3, 1, 2),
                padding_mode=padding_mode)


@dataclass
class RegistrationHistory:
    """What a keyframe hands to the frames that follow it.

    ``keyframe_flow`` is that frame's own registration; ``ir_features`` /
    ``vi_features`` hold the multi-scale maps the encoder already produced, so the
    next frame can align them by motion instead of re-encoding.  All fields are
    optional: an empty history means "no reliable past", and the single-frame
    pipeline runs unchanged.
    """

    keyframe_index: int = -1
    keyframe_flow: torch.Tensor | None = None
    ir_features: dict[str, torch.Tensor] = field(default_factory=dict)
    vi_features: dict[str, torch.Tensor] = field(default_factory=dict)
    reliability: torch.Tensor | None = None
    valid: torch.Tensor | None = None

    def is_empty(self) -> bool:
        return self.keyframe_flow is None


def history_from_output(output, frame_index: int) -> RegistrationHistory:
    """Capture a registration result as the state a later frame can reuse."""
    return RegistrationHistory(
        keyframe_index=int(frame_index),
        keyframe_flow=output.final_flow if output.final_flow is not None
        else output.coarse_flow,
        ir_features=dict(output.ir_features or {}),
        vi_features=dict(output.vi_features or {}),
        reliability=output.reliability,
        valid=output.reliable_mask,
    )


def propagate_flow(current_to_keyframe: torch.Tensor,
                   keyframe_flow: torch.Tensor,
                   keyframe_ir_to_current_ir: torch.Tensor,
                   padding_mode: str = "border"
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose the three motions into an initial field on the current frame.

    Returns the field and a ``[B,1,H,W]`` mask that is zero wherever any leg of
    the chain left the image, so an occluded or out-of-frame point cannot seed the
    current frame's local stage.
    """
    for name, tensor in (("current_to_keyframe", current_to_keyframe),
                         ("keyframe_flow", keyframe_flow),
                         ("keyframe_ir_to_current_ir", keyframe_ir_to_current_ir)):
        if tensor.ndim != 4 or tensor.shape[1] != 2:
            raise ValueError(f"{name} must be [B,2,H,W], got {tuple(tensor.shape)}")
        if tensor.shape != current_to_keyframe.shape:
            raise ValueError(f"{name} must share the current frame's shape, got "
                             f"{tuple(tensor.shape)} vs {tuple(current_to_keyframe.shape)}")
    height, width = current_to_keyframe.shape[-2:]
    base = _base_grid(height, width, current_to_keyframe)
    to_keyframe = base + current_to_keyframe.permute(0, 2, 3, 1)
    keyframe_coordinates = to_keyframe + sample_field(
        keyframe_flow, to_keyframe, padding_mode).permute(0, 2, 3, 1)
    current_ir_coordinates = keyframe_coordinates + sample_field(
        keyframe_ir_to_current_ir, keyframe_coordinates,
        padding_mode).permute(0, 2, 3, 1)
    valid = (_in_bounds(to_keyframe, height, width)
             & _in_bounds(keyframe_coordinates, height, width)
             & _in_bounds(current_ir_coordinates, height, width))
    initial = current_ir_coordinates - base
    return initial.permute(0, 3, 1, 2).contiguous(), valid.unsqueeze(1).to(
        current_to_keyframe.dtype)


def align_history_features(history: RegistrationHistory,
                           keyframe_ir_to_current_ir: torch.Tensor,
                           scales: tuple[str, ...] = ("1/8", "1/4", "1/2"),
                           padding_mode: str = "border"
                           ) -> dict[str, torch.Tensor]:
    """Warp cached keyframe IR features into the current IR frame.

    The features must be aligned before they reach an attention module; feeding
    them unaligned would ask the network to undo an unknown displacement it was
    never given.
    """
    if history.is_empty():
        return {}
    aligned: dict[str, torch.Tensor] = {}
    for scale in scales:
        feature = history.ir_features.get(scale)
        if feature is None:
            continue
        stride = keyframe_ir_to_current_ir.shape[-2] / feature.shape[-2]
        downsampled = torch.nn.functional.interpolate(
            keyframe_ir_to_current_ir, size=feature.shape[-2:], mode="bilinear",
            align_corners=False) / stride
        aligned[scale] = warp(feature, downsampled, padding_mode=padding_mode)
    return aligned
