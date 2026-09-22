"""Compact file visualisation for synthetic registration experiments."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def save_registration_preview(path: str | Path, *, moving_ir: torch.Tensor,
                              visible: torch.Tensor, aligned_ir: torch.Tensor,
                              predicted_flow: torch.Tensor,
                              gt_flow: torch.Tensor | None = None) -> None:
    """Save one four-panel registration preview without affecting training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not initialise matplotlib during training import.  In particular this
    # avoids writing to a home-directory cache on a locked-down A4000 server.
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    moving = moving_ir.detach().float().cpu()[0, 0]
    vi = visible.detach().float().cpu()[0].permute(1, 2, 0)
    aligned = aligned_ir.detach().float().cpu()[0, 0]
    flow = predicted_flow.detach().float().cpu()[0]
    magnitude = torch.linalg.vector_norm(flow, dim=0)
    panels = [(moving, "moving IR", "gray"), (vi, "fixed visible", None),
              (aligned, "aligned IR", "gray"), (magnitude, "predicted flow magnitude", "magma")]
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    if gt_flow is not None:
        gt = gt_flow.detach().float().cpu()[0]
        epe = torch.linalg.vector_norm(flow - gt, dim=0).mean().item()
        # This display has no validity mask: border padding is intentionally
        # visible as a harder, unmasked diagnostic. Training EPE is masked.
        figure.suptitle(f"flow EPE (unmasked): {epe:.3f} px")
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
