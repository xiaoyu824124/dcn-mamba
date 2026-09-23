"""Standalone descriptor acceptance probe: is the true match a peak?

A falling training loss does not prove that a descriptor makes the correct
location easier to match.  This module answers that question directly, on one
aligned or misaligned IR/VI pair, without any network head in the way:

* ``cos(GT)``               similarity at the true correspondence,
* ``contrast``              ``cos(GT)`` minus the mean similarity over a random
                            key sample (a cheap estimate of the field mean),
* ``global_frac_beat``      share of the random key sample that outranks the
                            truth -- ``0`` is perfect, ``~0.5`` is useless,
* ``local_win``            share of queries where the truth is the argmax
                            inside a small window centred on it,
* ``argmax_err_*_px``       how far that window argmax lands from the exact
                            (sub-pixel) truth,
* ``peak_frac@R``           share of queries where the truth beats every cell
                            beyond ``R`` pixels -- i.e. a genuine local peak.

Everything is measured per query, so a *local* peak and a hopeless *global*
ranking can be reported separately, which is exactly the distinction a coarse
plus fine pipeline depends on.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def descriptor_grid_hw(descriptor: torch.Tensor) -> Tuple[int, int]:
    if descriptor.ndim != 4:
        raise ValueError(f"descriptor must be [B,C,H,W], got {tuple(descriptor.shape)}")
    return int(descriptor.shape[-2]), int(descriptor.shape[-1])


@torch.no_grad()
def peak_report(ir_descriptor: torch.Tensor, vi_descriptor: torch.Tensor,
                flow: Optional[torch.Tensor] = None,
                stride_yx: Tuple[float, float] = (1.0, 1.0),
                n_queries: int = 256, window_px: float = 24.0,
                global_samples: int = 2048,
                query_mask: Optional[torch.Tensor] = None,
                radii_px: Sequence[float] = (4.0, 8.0, 16.0),
                argmax_radii_px: Sequence[float] = (4.0, 16.0),
                peak_margin: float = 1e-5,
                generator: Optional[torch.Generator] = None) -> Dict[str, float]:
    """Measure whether the true correspondence is a detectable local peak.

    Args:
        ir_descriptor: moving/IR descriptors ``[1,C,H,W]``.
        vi_descriptor: fixed/VI descriptors ``[1,C,H,W]``; queries live here.
        flow: optional ``[1,2,H,W]`` GT ``[dy,dx]`` **in image pixels** on the
            VI grid.  ``None`` means the pair is already aligned (``q* = p``).
        stride_yx: image pixels per descriptor cell on both grids.
        n_queries: number of randomly sampled query positions.
        window_px: half-width of the local window, in image pixels.
        global_samples: size of the random key sample used for the global rank.
        radii_px: radii for the ``peak_frac@R`` statistic.
        argmax_radii_px: window half-widths for the local argmax error.
        query_mask: optional ``[1,1,H,W]`` eligibility mask; queries are drawn
            only from positive cells.  Use it to restrict the probe to textured
            locations, because a descriptor cannot be expected to discriminate
            inside flat regions.
        generator: optional RNG for reproducible query/key sampling.

    Returns:
        A flat dict of floats; all pixel quantities are in image pixels.
    """
    if ir_descriptor.shape[0] != 1 or vi_descriptor.shape[0] != 1:
        raise ValueError("peak_report works on a single pair")
    if ir_descriptor.shape[1] != vi_descriptor.shape[1]:
        raise ValueError("IR and VI descriptors must share the channel count")
    height, width = descriptor_grid_hw(vi_descriptor)
    if descriptor_grid_hw(ir_descriptor) != (height, width):
        raise ValueError("IR and VI descriptors must share the grid size")
    if n_queries < 1 or global_samples < 1:
        raise ValueError("n_queries and global_samples must be positive")
    stride_y, stride_x = float(stride_yx[0]), float(stride_yx[1])
    if stride_y <= 0 or stride_x <= 0:
        raise ValueError("strides must be positive")
    device = vi_descriptor.device
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)

    query = F.normalize(vi_descriptor[0].float(), p=2, dim=0, eps=1e-6)
    key = F.normalize(ir_descriptor[0].float(), p=2, dim=0, eps=1e-6)
    channels = key.shape[0]
    query = query.reshape(channels, -1)                           # [C, N]
    key_flat = key.reshape(channels, -1)                          # [C, N]
    cells = key_flat.shape[1]

    total = height * width
    if cells != total:
        raise ValueError("descriptor channel/spatial layout is inconsistent")
    if query_mask is None:
        eligible = torch.arange(total)
    else:
        if tuple(query_mask.shape[-2:]) != (height, width):
            raise ValueError("query_mask must be [1,1,H,W] on the descriptor grid")
        eligible = (query_mask[0, 0].reshape(-1) > 0).nonzero().flatten().cpu()
        if eligible.numel() == 0:
            raise ValueError("query_mask excludes every cell")
    query_index = eligible[torch.randperm(eligible.numel(), generator=generator)[
        :min(n_queries, eligible.numel())]]
    if flow is None:
        target = torch.stack((query_index // width, query_index % width), dim=-1).float()
    else:
        if tuple(flow.shape[-2:]) != (height, width):
            raise ValueError("flow must be [1,2,H,W] on the VI descriptor grid")
        flat_flow = flow[0].float().reshape(2, -1)[:, query_index].transpose(0, 1)
        base = torch.stack((query_index // width, query_index % width), dim=-1).float()
        target = base + flat_flow.cpu() / torch.tensor((stride_y, stride_x))

    key_sample = torch.randperm(cells, generator=generator)[:min(global_samples, cells)]
    key_sample = key_sample.to(device)
    sampled_keys = key_flat[:, key_sample].to(device)

    window_cells_y = int(window_px / stride_y)
    window_cells_x = int(window_px / stride_x)
    offsets_y = torch.arange(-window_cells_y, window_cells_y + 1)
    offsets_x = torch.arange(-window_cells_x, window_cells_x + 1)
    off_y, off_x = torch.meshgrid(offsets_y, offsets_x, indexing="ij")
    off_y = off_y.reshape(-1).to(device)
    off_x = off_x.reshape(-1).to(device)
    offset_px = torch.sqrt((off_y.float() * stride_y) ** 2
                           + (off_x.float() * stride_x) ** 2).to(device)

    records = []
    for position in range(target.shape[0]):
        ty, tx = float(target[position, 0]), float(target[position, 1])
        if not (0 <= ty <= height - 1 and 0 <= tx <= width - 1):
            continue
        q = query[:, query_index[position]].to(device)
        # The query lives on the fixed grid, while the true key is at the
        # (possibly fractional) moving-grid target.  Comparing at the query
        # index silently measures the wrong point for every misaligned pair.
        y0, x0 = int(ty), int(tx)
        y1, x1 = min(y0 + 1, height - 1), min(x0 + 1, width - 1)
        wy, wx = ty - y0, tx - x0
        gt_key = ((1 - wy) * (1 - wx) * key_flat[:, y0 * width + x0]
                  + (1 - wy) * wx * key_flat[:, y0 * width + x1]
                  + wy * (1 - wx) * key_flat[:, y1 * width + x0]
                  + wy * wx * key_flat[:, y1 * width + x1]).to(device)
        cos_gt = float(torch.dot(q, gt_key))
        cos_sample = sampled_keys.t().mv(q)                        # [S]
        global_beat = float((cos_sample > cos_gt).float().mean())

        center_y, center_x = int(round(ty)), int(round(tx))
        raw_y = center_y + off_y
        raw_x = center_x + off_x
        inside = ((raw_y >= 0) & (raw_y <= height - 1)
                  & (raw_x >= 0) & (raw_x <= width - 1))
        cell_y = raw_y.clamp(0, height - 1)
        cell_x = raw_x.clamp(0, width - 1)
        local_keys = key_flat[:, (cell_y * width + cell_x)].to(device)   # [C, K]
        local_cos = local_keys.t().mv(q)                                # [K]
        inside = inside.to(device)
        local_cos = local_cos.masked_fill(~inside, -1.0)

        record = {"cos_gt": cos_gt,
                  "cos_mean": float(cos_sample.mean()),
                  "contrast": cos_gt - float(cos_sample.mean()),
                  "global_frac_beat": global_beat,
                  "local_frac_beat": float(
                      (local_cos[inside] > cos_gt).float().mean())}

        # Local argmax accuracy measured only over cells inside the grid.
        for radius in argmax_radii_px:
            within = inside & (offset_px <= float(radius) + 1e-6)
            masked = local_cos.masked_fill(~within, -1.0)
            best = int(masked.argmax())
            best_y = float(cell_y[best]) - ty
            best_x = float(cell_x[best]) - tx
            error_px = float(torch.sqrt(torch.tensor((best_y * stride_y) ** 2
                                                     + (best_x * stride_x) ** 2)))
            record[f"argmax_err_{int(radius)}px"] = error_px
            record[f"local_win_{int(radius)}px"] = float(error_px <= 0.5 * max(stride_y, stride_x))

        # Genuine local peak: the truth strictly beats every in-grid cell
        # beyond R pixels by at least ``peak_margin``.  The margin keeps float
        # noise from turning a flat field into peaks; without anything to
        # compare against, claim no peak.
        for radius in radii_px:
            beyond = inside & (offset_px > float(radius))
            record[f"peak_{int(radius)}px"] = (
                float(cos_gt > float(local_cos[beyond].max()) + peak_margin)
                if bool(beyond.any()) else 0.0)
        records.append(record)

    if not records:
        raise RuntimeError("no query landed inside the descriptor grid")

    keys = sorted(records[0])
    report: Dict[str, float] = {}
    for name in keys:
        values = torch.tensor([record[name] for record in records], dtype=torch.float64)
        report[name] = float(values.mean())
        if name.startswith("argmax_err_"):
            report[f"{name}_median"] = float(values.median())
    report["queries"] = float(len(records))
    report["stride_yx"] = float(max(stride_y, stride_x))
    return report


def format_report(name: str, report: Dict[str, float]) -> str:
    """One-line summary used by the CLI table."""
    return ("{name:30s} cosGT={cos_gt:6.3f} contrast={contrast:7.4f} "
            "gBeat={global_frac_beat:5.3f} lBeat={local_frac_beat:5.3f} "
            "win4={local_win_4px:5.3f} err4={argmax_err_4px:5.2f}px "
            "win16={local_win_16px:5.3f} err16={argmax_err_16px:5.2f}px "
            "peak4={peak_4px:4.2f} peak16={peak_16px:4.2f}").format(
        name=name,
        **{key: report[key] for key in
           ("cos_gt", "contrast", "global_frac_beat", "local_frac_beat",
            "local_win_4px", "argmax_err_4px", "local_win_16px",
            "argmax_err_16px", "peak_4px", "peak_16px")})
