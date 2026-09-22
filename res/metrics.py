"""Small, convention-safe registration metrics."""

from __future__ import annotations

import torch


def endpoint_error(predicted_flow: torch.Tensor, target_flow: torch.Tensor,
                   valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean flow endpoint error for ``[dy,dx]`` fields, optionally masked."""
    if predicted_flow.shape != target_flow.shape or predicted_flow.ndim != 4 or predicted_flow.shape[1] != 2:
        raise ValueError("predicted_flow and target_flow must both be [B,2,H,W]")
    error = torch.linalg.vector_norm(predicted_flow - target_flow, ord=2, dim=1, keepdim=True)
    if valid_mask is None:
        return error.mean()
    if valid_mask.shape != error.shape:
        raise ValueError("valid_mask must have shape [B,1,H,W]")
    weights = valid_mask.to(dtype=error.dtype).clamp(0, 1)
    return (error * weights).sum() / weights.sum().clamp_min(1)
