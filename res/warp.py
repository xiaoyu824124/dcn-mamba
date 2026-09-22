"""Pixel-coordinate backward warping and explicit flow resizing.

Convention used by every module in ``res``:

* flow layout is ``[B, 2, H, W]`` in ``[dy, dx]`` order;
* flow values are **pixel displacements**, never normalised coordinates;
* ``warp(source, flow)[..., y, x] = source[..., y + dy, x + dx]``;
* `align_corners=True` is fixed everywhere so a zero flow is exact identity.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _base_grid(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    """Return pixel-coordinate ``[1,H,W,2]`` grid in grid_sample's ``[x,y]`` order."""
    y = torch.arange(height, device=reference.device, dtype=reference.dtype)
    x = torch.arange(width, device=reference.device, dtype=reference.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


def flow_to_normalized_grid(flow: torch.Tensor) -> torch.Tensor:
    """Convert pixel ``[dy,dx]`` flow to a ``grid_sample`` grid in ``[-1,1]``.

    Output shape is ``[B,H,W,2]`` in grid_sample's ``[x,y]`` order.  This helper
    is public so visualisation and tests share precisely the same convention.
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be [B,2,H,W] in [dy,dx] order, got {tuple(flow.shape)}")
    _, _, height, width = flow.shape
    if not flow.is_floating_point():
        raise TypeError("flow must be floating point pixel displacements")
    base = _base_grid(height, width, flow)
    displacement_xy = flow[:, [1, 0]].permute(0, 2, 3, 1)
    locations = base + displacement_xy
    x = 2.0 * locations[..., 0] / max(width - 1, 1) - 1.0
    y = 2.0 * locations[..., 1] / max(height - 1, 1) - 1.0
    return torch.stack((x, y), dim=-1)


def warp(source: torch.Tensor, flow: torch.Tensor,
         mode: str = "bilinear", padding_mode: str = "zeros") -> torch.Tensor:
    """Backward-warp ``source`` with a pixel displacement field.

    Args:
        source: Moving image/feature map ``[B,C,H,W]``.
        flow: Reference-grid pixel flow ``[B,2,H,W]`` in ``[dy,dx]`` order.
        mode: `grid_sample` interpolation mode.
        padding_mode: `grid_sample` border behaviour.
    """
    if source.ndim != 4:
        raise ValueError(f"source must be [B,C,H,W], got {tuple(source.shape)}")
    if source.shape[0] != flow.shape[0] or source.shape[-2:] != flow.shape[-2:]:
        raise ValueError(
            "source and flow must share B,H,W; got "
            f"{tuple(source.shape)} and {tuple(flow.shape)}")
    grid = flow_to_normalized_grid(flow)
    return F.grid_sample(source, grid, mode=mode, padding_mode=padding_mode,
                         align_corners=True)


def resize_flow(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize a pixel flow field and scale displacement magnitudes correctly.

    With ``align_corners=True``, the exact pixel-coordinate scale from an input
    grid of length ``n`` to an output grid of length ``m`` is ``(m-1)/(n-1)``.
    Scaling by ``m/n`` would introduce a systematic geometric error.
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
    out_height, out_width = (int(size[0]), int(size[1]))
    if out_height < 1 or out_width < 1:
        raise ValueError(f"invalid output size {size}")
    in_height, in_width = flow.shape[-2:]
    resized = F.interpolate(flow, size=(out_height, out_width), mode="bilinear",
                            align_corners=True)
    scale_y = ((out_height - 1) / max(in_height - 1, 1))
    scale_x = ((out_width - 1) / max(in_width - 1, 1))
    scale = flow.new_tensor((scale_y, scale_x)).view(1, 2, 1, 1)
    return resized * scale


def upsample_feature_flow(flow: torch.Tensor, size: Tuple[int, int],
                          stride_yx: Tuple[float, float]) -> torch.Tensor:
    """Lift a feature-grid flow to an image grid with known feature strides.

    This differs intentionally from :func:`resize_flow`: a 1/8 encoder feature
    coordinate represents an eight-pixel step in the input image, rather than a
    reparameterisation of the two grids' endpoints.  The flow is therefore
    interpolated on the feature lattice and then its ``[dy,dx]`` values are
    multiplied by the explicit physical strides ``(stride_y, stride_x)``.
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must be [B,2,H,W], got {tuple(flow.shape)}")
    stride_y, stride_x = float(stride_yx[0]), float(stride_yx[1])
    if stride_y <= 0 or stride_x <= 0:
        raise ValueError(f"strides must be positive, got {stride_yx}")
    resized = F.interpolate(flow, size=tuple(map(int, size)), mode="bilinear",
                            align_corners=False)
    return resized * flow.new_tensor((stride_y, stride_x)).view(1, 2, 1, 1)
