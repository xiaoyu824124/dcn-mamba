# -*- coding: utf-8 -*-
"""Rank structural representations for IR--visible global matching.

Run from the repository root::

    python -B -m res.representation_bench --pairs 4 --scale 8

Protocol
--------
For every (representation, transform) pair the benchmark measures how well an
all-pairs cosine matcher on coarse features recovers a *known* misalignment.
No ground-truth flow is needed from the dataset, which is what makes this
runnable on HDO: the GT is derived exactly from the transform that generated
the misalignment.

    base pair   HDO ``aligned/ir`` and ``aligned/vi``, already registered
                (measured residual ~1-2 px, cross-modal NCC 0.43 against 0.21
                for a mismatched partner), so ``identity`` is the control row
    downsample  both images once, to the matching scale, with area pooling
    moving      the coarse IR sampling-warped by a KNOWN affine/elastic map
    fixed       the coarse VIS
    exact GT    ``F(p) = W^{-1}(p) - p`` in coarse pixels   (derived exactly)

**The transform is applied at the matching scale, not at full resolution.**
This matters more than it looks: area pooling is only shift-covariant for
displacements that are integer multiples of the stride, so warping at full
resolution and then pooling makes an exact 5-cell shift match perfectly while a
2.5-cell shift is destroyed by the pooling grid -- measuring the resampler, not
the representation.  Applying the transform to the already-coarse images removes
that confound and leaves a genuine sub-cell displacement to recover, which is
the real task.

Metrics -- all temperature-free except ``entropy``
--------------------------------------------------
    cover       fraction of fixed tokens that are in-bounds and informative
    margin      mean over fixed tokens of ``cos(top1) - cos(top2)``
    top1_acc    hard-argmax correspondence within one cell of the true cell
    vote_err    error of the Hough/consensus displacement: quantise every
                argmax into a displacement vote, take the modal bin.  This is
                the "can global matching still find the coarse correspondence
                when individual pixels are ambiguous" question, and it is the
                metric the parametric-fit design actually depends on.
    epe_affine  EPE of a margin-weighted closed-form affine fit, full-res px
    entropy     row-softmax entropy at the project-default tau, normalised

All reported EPE values are in **full-resolution pixels**; internally the
matcher works in coarse pixels.  Every flow is ``[dy, dx]`` with
``warped(p) = source(p + flow(p))``, matching the project convention.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

try:                                                    # module execution
    from .representations import REPRESENTATIONS
except ImportError:                                     # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from representations import REPRESENTATIONS

DEFAULT_HDO_ROOT = Path(r"E:\HDO")
DEFAULT_SCALE = 8
DEFAULT_TAU = 0.07          # the project's default softmax temperature
CANONICAL_HW = (480, 640)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def pixel_grid(h: int, w: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype), indexing="ij")
    return xx.unsqueeze(0), yy.unsqueeze(0)


def affine_maps(h: int, w: int, angle_deg: float, scale: float,
                disp: Sequence[float], device):
    """``y = A p + b`` (the exact GT map) and ``A^{-1}`` for sampling.

    ``p`` and ``y`` are ``(x, y)`` pixel coordinates at the matching scale.
    ``disp`` *is* the ground-truth displacement at the image centre.
    """
    centre = torch.tensor([(w - 1) / 2.0, (h - 1) / 2.0], dtype=torch.float64)
    theta = math.radians(float(angle_deg))
    A = torch.tensor([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]],
                     dtype=torch.float64) * float(scale)
    t = torch.tensor([float(disp[0]), float(disp[1])], dtype=torch.float64)
    b = centre + t - A @ centre
    return A.to(device), b.to(device), torch.linalg.inv(A).to(device)


def sample_affine(x: torch.Tensor, A_inv: torch.Tensor,
                  b: torch.Tensor) -> torch.Tensor:
    """``moving(p) = x(A^{-1} (p - b))`` -- backward sampling, align_corners=True."""
    batch, _, h, w = x.shape
    xx, yy = pixel_grid(h, w, x.device, torch.float64)
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    mapped = (pts - b) @ A_inv.transpose(0, 1)
    gx = 2.0 * mapped[:, 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * mapped[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=1).reshape(1, h, w, 2).to(x.dtype)
    return F.grid_sample(x, grid.expand(batch, -1, -1, -1), mode="bilinear",
                         padding_mode="zeros", align_corners=True)


def gt_affine_flow(A: torch.Tensor, b: torch.Tensor, h: int, w: int,
                   device) -> torch.Tensor:
    """Exact GT flow ``[1,2,h,w]`` as ``[dy, dx]``: ``F(p) = A p + b - p``."""
    xx, yy = pixel_grid(h, w, device, torch.float64)
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    mapped = pts @ A.transpose(0, 1) + b
    dx = (mapped[:, 0] - pts[:, 0]).reshape(1, 1, h, w)
    dy = (mapped[:, 1] - pts[:, 1]).reshape(1, 1, h, w)
    return torch.cat([dy, dx], dim=1).float()


def elastic_field(h: int, w: int, amplitude: float, grid: int, device,
                  seed: int) -> torch.Tensor:
    """A smooth random displacement field ``[1,2,h,w]`` as ``[dy, dx]``."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    small = torch.randn(1, 2, grid, grid, generator=generator)
    field = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=True)
    kernel_radius = max(1, int(round(3.0 * max(h, w) / (6.0 * grid))))
    x = torch.arange(-kernel_radius, kernel_radius + 1, dtype=torch.float32)
    sigma = max(h, w) / (6.0 * grid)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, 1, -1).expand(2, 1, 1, -1)
    field = F.conv2d(F.pad(field, (kernel_radius,) * 2 + (0, 0), mode="replicate"),
                     kernel, groups=2)
    kernel = kernel.transpose(2, 3)
    field = F.conv2d(F.pad(field, (0, 0) + (kernel_radius,) * 2, mode="replicate"),
                     kernel, groups=2)
    peak = field.abs().amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (field * (float(amplitude) / peak)).to(device)


def sample_by_field(x: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """``moving(p) = x(p + field(p))``."""
    batch, _, h, w = x.shape
    xx, yy = pixel_grid(h, w, x.device, field.dtype)
    qx = xx + field[:, 1]
    qy = yy + field[:, 0]
    grid = torch.stack([2.0 * qx / max(w - 1, 1) - 1.0,
                        2.0 * qy / max(h - 1, 1) - 1.0], dim=-1)
    return F.grid_sample(x, grid.expand(batch, -1, -1, -1), mode="bilinear",
                         padding_mode="zeros", align_corners=True)


def elastic_gt_flow(field: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Exact GT flow for ``W(p) = p + field(p)`` by fixed-point inversion.

    The flow must satisfy ``W(p + F(p)) = p``, i.e. ``F = -field(p + F)``.
    """
    batch, _, h, w = field.shape
    xx, yy = pixel_grid(h, w, field.device, field.dtype)
    flow = torch.zeros_like(field)
    for _ in range(int(iterations)):
        qx = xx + flow[:, 1]
        qy = yy + flow[:, 0]
        grid = torch.stack([2.0 * qx / max(w - 1, 1) - 1.0,
                            2.0 * qy / max(h - 1, 1) - 1.0], dim=-1)
        sampled = F.grid_sample(field, grid.expand(batch, -1, -1, -1),
                                mode="bilinear", padding_mode="border",
                                align_corners=True)
        flow = -sampled
    return flow


# --------------------------------------------------------------------------- #
# transforms -- displacements are given in FULL-RESOLUTION pixels and are
# divided by the matching scale before being applied to the coarse images
# --------------------------------------------------------------------------- #
TRANSFORMS: List[Tuple[str, Dict]] = [
    ("identity",   dict(kind="affine", angle=0.0,  scale=1.00, disp=(0.0, 0.0))),
    ("gt5",        dict(kind="affine", angle=0.0,  scale=1.00, disp=(5.0, 0.0))),
    ("gt20",       dict(kind="affine", angle=0.0,  scale=1.00, disp=(20.0, 0.0))),
    ("gt40",       dict(kind="affine", angle=0.0,  scale=1.00, disp=(40.0, 0.0))),
    ("gt80",       dict(kind="affine", angle=0.0,  scale=1.00, disp=(80.0, 0.0))),
    ("gt20diag",   dict(kind="affine", angle=0.0,  scale=1.00, disp=(14.14, 14.14))),
    ("rot5",       dict(kind="affine", angle=5.0,  scale=1.00, disp=(0.0, 0.0))),
    ("rot10",      dict(kind="affine", angle=10.0, scale=1.00, disp=(0.0, 0.0))),
    ("scale105",   dict(kind="affine", angle=0.0,  scale=1.05, disp=(0.0, 0.0))),
    ("scale095",   dict(kind="affine", angle=0.0,  scale=0.95, disp=(0.0, 0.0))),
    ("affine_mix", dict(kind="affine", angle=5.0,  scale=1.05, disp=(20.0, 10.0))),
    ("elastic8",   dict(kind="elastic", amplitude=8.0, grid=6)),
]


def build_case(spec: Dict, coarse: torch.Tensor, match_scale: int, seed: int
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(warped_coarse_image, exact_gt_flow_in_coarse_pixels)``."""
    _, _, h, w = coarse.shape
    device = coarse.device
    if spec["kind"] == "affine":
        disp = (spec["disp"][0] / match_scale, spec["disp"][1] / match_scale)
        A, b, A_inv = affine_maps(h, w, spec["angle"], spec["scale"], disp, device)
        return (sample_affine(coarse, A_inv, b),
                gt_affine_flow(A, b, h, w, device))
    if spec["kind"] == "elastic":
        field = elastic_field(h, w, spec["amplitude"] / match_scale,
                              spec["grid"], device, seed)
        return sample_by_field(coarse, field), elastic_gt_flow(field)
    raise ValueError(f"unknown transform kind {spec['kind']!r}")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_gray(path: Path, size_hw=CANONICAL_HW) -> torch.Tensor:
    """Load as ``[1,1,H,W]`` float in ``[0,1]``, centre-cropped to the canonical
    aspect ratio so a 4:3 IR and a 16:9 VIS can share one grid."""
    from PIL import Image
    import numpy as np

    image = Image.open(path).convert("L")
    width, height = image.size
    target = size_hw[1] / size_hw[0]
    if abs(width / height - target) > 1e-3:
        if width / height > target:
            new_w = int(round(height * target))
            left = (width - new_w) // 2
            image = image.crop((left, 0, left + new_w, height))
        else:
            new_h = int(round(width / target))
            top = (height - new_h) // 2
            image = image.crop((0, top, width, top + new_h))
    image = image.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    array = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(array)[None, None]


def select_pairs(root: Path, count: int) -> List[Tuple[Path, Path]]:
    """Pick ``count`` IR/VIS pairs spread across distinct HDO sequences."""
    ir_dir, vi_dir = root / "aligned" / "ir", root / "aligned" / "vi"
    if not ir_dir.is_dir() or not vi_dir.is_dir():
        raise FileNotFoundError(f"expected {ir_dir} and {vi_dir}")
    common = sorted(set(os.listdir(ir_dir)) & set(os.listdir(vi_dir)))
    groups: Dict[str, List[str]] = {}
    for name in common:
        match = re.match(r"^([a-z]+)(\d+)", name)
        if match:
            groups.setdefault(match.group(1), []).append(name)
    keys = sorted(groups)
    if not keys:
        raise RuntimeError(f"no HDO aligned pairs found under {root}")
    count = max(1, min(count, len(keys)))
    chosen = ([len(keys) // 2] if count == 1 else
              [int(round(i * (len(keys) - 1) / (count - 1))) for i in range(count)])
    pairs = []
    for index in chosen:
        names = sorted(groups[keys[index]])
        name = names[len(names) // 2]
        pairs.append((ir_dir / name, vi_dir / name))
    return pairs


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def normalise(feature: torch.Tensor, l2: bool) -> torch.Tensor:
    return F.normalize(feature, p=2, dim=1, eps=1e-4) if l2 else feature


def fit_affine(src: torch.Tensor, dst: torch.Tensor, weight: torch.Tensor,
               ridge: float = 1e-6) -> torch.Tensor:
    """Weighted closed-form least squares ``[x,y,1] -> [x',y']``."""
    n = src.shape[0]
    design = torch.cat([src, torch.ones(n, 1, dtype=src.dtype, device=src.device)], 1)
    eye = torch.eye(3, dtype=src.dtype, device=src.device)
    xtwx = torch.einsum("ni,n,nj->ij", design, weight, design) + ridge * eye
    xtwy = torch.einsum("ni,n,nj->ij", design, weight, dst)
    return torch.linalg.solve(xtwx, xtwy)


def evaluate(moving: torch.Tensor, fixed: torch.Tensor, gt_flow: torch.Tensor,
             name: str, match_scale: int, tau: float) -> Dict[str, float]:
    """Score one representation on one already-warped coarse pair."""
    spec = REPRESENTATIONS[name]
    l2 = bool(spec["l2"])
    moving_raw = spec["fn"](moving.float())
    fixed_raw = spec["fn"](fixed.float())
    moving_feat = normalise(moving_raw, l2)
    fixed_feat = normalise(fixed_raw, l2)

    _, _, hf, wf = fixed_feat.shape
    _, _, hm, wm = moving_feat.shape
    if (hf, wf) != (hm, wm):
        raise ValueError(f"{name}: coarse grids differ "
                         f"{tuple(fixed_feat.shape)} vs {tuple(moving_feat.shape)}")
    if fixed_feat.shape[1] < 2:
        raise ValueError(f"{name}: one-channel features have no channel axis for "
                         f"a cosine matcher")

    correlation = torch.bmm(fixed_feat.flatten(2).transpose(1, 2),
                            moving_feat.flatten(2))                     # [1,Nf,Nm]
    if not torch.isfinite(correlation).all():
        raise FloatingPointError(f"{name}: non-finite correlation")

    top2 = correlation.topk(2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1]).reshape(-1)
    argmax = correlation.argmax(dim=-1).reshape(-1)
    row = (argmax // wm).float()
    col = (argmax % wm).float()

    jj = torch.arange(wf, device=moving.device, dtype=torch.float32)
    jj = jj.view(1, -1).expand(hf, wf).reshape(-1)
    ii = torch.arange(hf, device=moving.device, dtype=torch.float32)
    ii = ii.view(-1, 1).expand(hf, wf).reshape(-1)

    gt_dy = gt_flow[:, 0].reshape(-1).float()
    gt_dx = gt_flow[:, 1].reshape(-1).float()

    # ---- validity: fixed token away from the border, target inside ---------- #
    pad = 2.0
    valid = ((jj > pad) & (jj < wf - 1 - pad) & (ii > pad) & (ii < hf - 1 - pad))
    valid &= ((jj + gt_dx > 0) & (jj + gt_dx < wf - 1)
              & (ii + gt_dy > 0) & (ii + gt_dy < hf - 1))
    if l2:
        # A zero-magnitude descriptor carries no information; without this the
        # flat-region problem is hidden rather than measured.
        norm = fixed_raw.norm(dim=1, keepdim=True)
        valid &= norm.reshape(-1) > 1e-3
    if int(valid.sum()) < 16:
        return {key: float("nan") for key in METRICS}

    # ---- top-1 correspondence accuracy (coarse cells -> full-res px) -------- #
    true_col = (jj + gt_dx).round()
    true_row = (ii + gt_dy).round()
    hit = (((col - true_col).abs() <= 1.0) & ((row - true_row).abs() <= 1.0))

    # ---- Hough / consensus displacement ------------------------------------ #
    cell_dx = (col - jj).round().long()
    cell_dy = (row - ii).round().long()
    bins = ((cell_dy + hf) * (2 * wf + 1) + (cell_dx + wf))
    counts = torch.bincount(bins[valid], minlength=(2 * hf + 1) * (2 * wf + 1))
    best = int(counts.argmax())
    vote_dy = (best // (2 * wf + 1) - hf) * match_scale
    vote_dx = (best % (2 * wf + 1) - wf) * match_scale
    mean_gt_dy = float(gt_dy[valid].mean()) * match_scale
    mean_gt_dx = float(gt_dx[valid].mean()) * match_scale
    vote_err = math.hypot(vote_dx - mean_gt_dx, vote_dy - mean_gt_dy)

    # ---- margin-weighted closed-form affine fit ---------------------------- #
    weight = margin.clamp_min(0.0)
    weight = weight / weight.sum().clamp_min(1e-8)
    src = torch.stack([jj / wf, ii / hf], dim=1)[valid]
    dst = torch.stack([col / wf, row / hf], dim=1)[valid]
    solution = fit_affine(src, dst, weight[valid])
    design = torch.stack([jj / wf, ii / hf, torch.ones_like(jj)], dim=1)
    mapped = design @ solution
    fit_dx = (mapped[:, 0] * wf - jj) * match_scale
    fit_dy = (mapped[:, 1] * hf - ii) * match_scale
    gt_dx_px, gt_dy_px = gt_dx * match_scale, gt_dy * match_scale
    epe_fit = torch.sqrt((fit_dx - gt_dx_px) ** 2 + (fit_dy - gt_dy_px) ** 2)

    # ---- one-pass RANSAC: fit only the consensus inliers ------------------- #
    # The margin-weighted fit above uses every token, so when per-pixel
    # correspondences are near chance it is driven by noise.  Restricting the
    # fit to the tokens that voted for the modal displacement is the cheap
    # two-stage scheme (vote, then fit) and is what the parametric design needs.
    inlier = valid & (bins == best)
    if int(inlier.sum()) >= 16:
        src_i = torch.stack([jj / wf, ii / hf], dim=1)[inlier]
        dst_i = torch.stack([col / wf, row / hf], dim=1)[inlier]
        uniform = torch.ones(int(inlier.sum()), device=moving.device,
                             dtype=torch.float32)
        solution_i = fit_affine(src_i, dst_i, uniform)
        mapped_i = design @ solution_i
        vote_fit_dx = (mapped_i[:, 0] * wf - jj) * match_scale
        vote_fit_dy = (mapped_i[:, 1] * hf - ii) * match_scale
        epe_votefit = torch.sqrt((vote_fit_dx - gt_dx_px) ** 2
                                 + (vote_fit_dy - gt_dy_px) ** 2)
        epe_votefit = float(epe_votefit[valid].mean())
    else:
        epe_votefit = float("nan")

    # ---- entropy at the project default temperature ------------------------- #
    soft = F.softmax(correlation.reshape(-1, correlation.shape[-1]) / tau, dim=-1)
    entropy = -(soft * (soft + 1e-12).log()).sum(dim=-1)
    entropy = entropy / math.log(max(correlation.shape[-1], 2))

    return {
        "cover": float(valid.float().mean()),
        "margin": float(margin[valid].mean()),
        "top1_acc": float(hit[valid].float().mean()),
        "vote_err": float(vote_err),
        "epe_affine": float(epe_fit[valid].mean()),
        "epe_votefit": epe_votefit,
        "entropy": float(entropy.reshape(-1)[valid].mean()),
    }


METRICS = ["margin", "top1_acc", "vote_err", "epe_affine", "epe_votefit",
           "entropy", "cover"]


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def measure_cost(name: str, shape=(1, 1) + CANONICAL_HW, device="cuda",
                 warmup: int = 5, repeats: int = 20) -> float:
    """Milliseconds per call, at full resolution and at the matching scale."""
    spec = REPRESENTATIONS[name]
    timings = []
    for size in (shape, (1, 1, CANONICAL_HW[0] // DEFAULT_SCALE,
                         CANONICAL_HW[1] // DEFAULT_SCALE)):
        image = torch.rand(size, device=device)
        for _ in range(warmup):
            spec["fn"](image)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            spec["fn"](image)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0 / repeats)
    return timings[0], timings[1]


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_matrix(title: str, results: Dict, names: Sequence[str],
                 transforms: Sequence[str], metric: str) -> None:
    header = f"{'representation':<14}" + "".join(f"{t:>10}" for t in transforms)
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for name in names:
        row = f"{name:<14}"
        for transform in transforms:
            value = results.get((name, transform), {}).get(metric, float("nan"))
            row += f"{value:>10.3f}" if math.isfinite(value) else f"{'--':>10}"
        print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hdo-root", type=Path, default=DEFAULT_HDO_ROOT)
    parser.add_argument("--pairs", type=int, default=4,
                        help="number of HDO aligned pairs (distinct sequences)")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE,
                        help="matching resolution denominator (default 1/8)")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--reps", type=str, default="",
                        help="comma separated subset of representations")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "bench_out")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = ([n.strip() for n in args.reps.split(",") if n.strip()]
             or [n for n, s in REPRESENTATIONS.items() if s.get("cosine_ok", True)])
    for name in names:
        if name not in REPRESENTATIONS:
            raise SystemExit(f"unknown representation {name!r}")
        if not REPRESENTATIONS[name].get("cosine_ok", True):
            raise SystemExit(
                f"{name!r} is a one-channel map: an all-pairs cosine over channels "
                f"degenerates into an outer product of scalars. Use its *_patch "
                f"variant instead.")
    transforms = [t for t, _ in TRANSFORMS]

    pairs = select_pairs(args.hdo_root, args.pairs)
    print(f"device={device}  match scale=1/{args.scale}  tau={args.tau}  "
          f"pairs={len(pairs)}  representations={len(names)}")
    for ir_path, _ in pairs:
        print(f"  pair {ir_path.name}")

    pairs_full = [(load_gray(p).to(device), load_gray(v).to(device))
                  for p, v in pairs]
    pairs_coarse = [
        (F.interpolate(ir, scale_factor=1.0 / args.scale, mode="area"),
         F.interpolate(vi, scale_factor=1.0 / args.scale, mode="area"))
        for ir, vi in pairs_full]

    results: Dict[Tuple[str, str], Dict[str, float]] = {}
    for name in names:
        for transform, spec in TRANSFORMS:
            rows = []
            for index, (ir, vi) in enumerate(pairs_coarse):
                moving, gt_flow = build_case(spec, ir, args.scale, 1234 + 97 * index)
                rows.append(evaluate(moving, vi, gt_flow, name, args.scale, args.tau))
            entry = {}
            for metric in METRICS:
                values = [r[metric] for r in rows if math.isfinite(r[metric])]
                entry[metric] = sum(values) / len(values) if values else float("nan")
            results[(name, transform)] = entry
        print(f"  evaluated {name}")

    costs = {name: measure_cost(name, device=device) for name in names}

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"representation_bench_scale{args.scale}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["representation", "family", "transform",
                         "cost_full_ms", "cost_coarse_ms"] + METRICS)
        for name in names:
            for transform in transforms:
                entry = results[(name, transform)]
                writer.writerow([name, REPRESENTATIONS[name]["family"], transform,
                                 f"{costs[name][0]:.3f}", f"{costs[name][1]:.3f}"]
                                + [f"{entry[m]:.6f}" for m in METRICS])

    for metric, title in (
            ("margin", "MARGIN  cos(top1)-cos(top2)   [higher better]"),
            ("top1_acc", "TOP-1 ACCURACY within one cell   [higher better]"),
            ("vote_err", "CONSENSUS (Hough) DISPLACEMENT ERROR, px   [lower better]"),
            ("epe_affine", "EPE after margin-weighted affine fit, px   [lower better]"),
            ("epe_votefit", "EPE after consensus-inlier affine fit, px   [lower better]"),
            ("entropy", "NORMALISED SOFTMAX ENTROPY   [lower better]")):
        print_matrix(title, results, names, transforms, metric)

    print("\n=== SUMMARY (mean over transforms) ===")
    print(f"{'representation':<14}{'family':<17}{'margin':>9}{'top1':>8}"
          f"{'vote':>8}{'epe_aff':>9}{'cover':>8}{'ms_1/1':>9}{'ms_coarse':>11}")
    print("-" * 91)
    ranking = []
    for name in names:
        mean = {m: sum(results[(name, t)][m] for t in transforms
                       if math.isfinite(results[(name, t)][m]))
                   / max(1, sum(1 for t in transforms
                                if math.isfinite(results[(name, t)][m])))
                for m in METRICS}
        ranking.append((mean["top1_acc"], name, mean))
    for _, name, mean in sorted(ranking, key=lambda t: -t[0]):
        print(f"{name:<14}{REPRESENTATIONS[name]['family']:<17}"
              f"{mean['margin']:>9.3f}{mean['top1_acc']:>8.3f}{mean['vote_err']:>8.1f}"
              f"{mean['epe_affine']:>9.2f}{mean['cover']:>8.3f}"
              f"{costs[name][0]:>9.2f}{costs[name][1]:>11.2f}")
    print(f"\nCSV written to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
