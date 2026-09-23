"""Standalone MIND acceptance run on VTMOT: is the true match a peak?

    python -B -m res.validate_mind --split eval --frame-stride 50

Measures, for one aligned pair (``ir`` vs ``visible_gt``) and one misaligned
pair (``ir`` vs ``visible_mis`` with the stored GT flow):

* raw MIND at full, 1/2, 1/4 and 1/8 resolution,
* the shared encoder's 1/4 and 1/8 features,

and reports the peak/rank statistics from :mod:`res.mind_probe`.  No network
head, no training and no checkpoint are involved -- this is a pure data
acceptance check.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .encoder import MINDFeatureEncoder
from .mind import MINDDescriptor, rgb_to_gray
from .mind_probe import format_report, peak_report
from .vtmot import VTMOTSingleFrameDataset


def _pool(descriptor: torch.Tensor, factor: int) -> torch.Tensor:
    """Cell-average a descriptor map and re-normalise (stride ``factor``)."""
    if factor == 1:
        return descriptor
    pooled = F.avg_pool2d(descriptor.float(), factor, factor)
    return F.normalize(pooled, p=2, dim=1, eps=1e-6)


def _texture_mask(gray: torch.Tensor, percentile: float) -> torch.Tensor:
    """Cells whose local gradient magnitude is above ``percentile``."""
    sobel = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0],
                          [-1.0, 0.0, 1.0]], device=gray.device)
    kernel_x = sobel.view(1, 1, 3, 3)
    kernel_y = sobel.t().reshape(1, 1, 3, 3)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-12)
    threshold = torch.quantile(magnitude.flatten().float(), percentile / 100.0)
    return (magnitude >= threshold).float()


def _run_pair(name: str, ir: torch.Tensor, vi: torch.Tensor, flow, mind: MINDDescriptor,
              encoder: MINDFeatureEncoder, args,
              encoder_tag: str = "") -> List[Tuple[str, Dict[str, float]]]:
    results: List[Tuple[str, Dict[str, float]]] = []
    # Queries live on the fixed VI grid.  A moving-IR texture mask selects
    # unrelated locations when the pair is misaligned.
    gray_fixed = rgb_to_gray(vi)
    masks = {"all": None,
             f"tex{100 - int(args.texture_percentile)}": _texture_mask(
                 gray_fixed, args.texture_percentile)}
    ir_mind = mind(ir)
    vi_mind = mind(rgb_to_gray(vi))
    representations = [(f"MIND 1/{factor}", _pool(ir_mind, factor), _pool(vi_mind, factor),
                        float(factor)) for factor in (1, 2, 4, 8)]
    with torch.no_grad():
        ir_features, vi_features = encoder.encode_pair(ir_mind, vi_mind)
    representations.append((f"enc{encoder_tag} 1/4", ir_features["1/4"], vi_features["1/4"], 4.0))
    representations.append((f"enc{encoder_tag} 1/8", ir_features["1/8"], vi_features["1/8"], 8.0))
    for label, ir_description, vi_description, stride in representations:
        for mask_label, mask in masks.items():
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            downsampled_mask = (None if mask is None
                                else F.avg_pool2d(mask, int(stride), int(stride)))
            report = peak_report(
                ir_description, vi_description,
                None if flow is None else F.avg_pool2d(flow, int(stride), int(stride)),
                stride_yx=(stride, stride), n_queries=args.queries,
                window_px=args.window_px, global_samples=args.global_samples,
                query_mask=downsampled_mask, generator=generator)
            results.append((f"{name} {label} [{mask_label}]", report))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone MIND peak acceptance run")
    parser.add_argument("--data-root", default="data/VTMOT_misaligned")
    parser.add_argument("--split-file", default="data_split/IVF/VTMOT/split.json")
    parser.add_argument("--split", choices=("train", "eval", "test"), default="eval")
    parser.add_argument("--target-hw", type=int, nargs=2, default=(480, 640))
    parser.add_argument("--frame-stride", type=int, default=50)
    parser.add_argument("--samples", type=int, default=4, help="pairs to average over")
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--texture-percentile", type=float, default=75.0,
                        help="queries in the textured subset are above this IR gradient percentile")
    parser.add_argument("--window-px", type=float, default=24.0)
    parser.add_argument("--global-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None,
                        help="optional registration checkpoint; its encoder weights replace the "
                             "random initialisation so the enc rows show the LEARNED representation")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = VTMOTSingleFrameDataset(
        args.data_root, split=args.split, split_file=args.split_file,
        target_hw=tuple(args.target_hw), frame_stride=args.frame_stride,
        include_rgb_gt=True, max_samples=args.samples)
    mind = MINDDescriptor().to(device)
    encoder = MINDFeatureEncoder().to(device).eval()
    encoder_tag = "(random)"
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
        encoder_config = dict(payload.get("config", {}).get("encoder", {}))
        if encoder_config:
            encoder = MINDFeatureEncoder(in_channels=mind.channels,
                                         **encoder_config).to(device).eval()
        prefix = "encoder."
        weights = {name[len(prefix):]: value for name, value in payload["model"].items()
                   if name.startswith(prefix)}
        report = encoder.load_state_dict(weights, strict=False)
        encoder_tag = "(trained)"
        print(f"encoder loaded from {args.checkpoint}: {len(weights)} tensors, "
              f"missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)}")
    print(f"split={args.split} pairs={len(dataset)} queries/pair={args.queries} "
          f"fov={tuple(args.target_hw)} device={device}")

    totals: Dict[str, List[Dict[str, float]]] = {}
    order: List[str] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        ir = sample["ir"][None].to(device)
        visible_gt = sample["rgb_gt"][None].to(device)
        visible_mis = sample["vi"][None].to(device)
        flow = sample["gt_flow"][None].to(device)
        for label, moving, fixed, pair_flow in (
                ("aligned ir|visible_gt", ir, visible_gt, None),
                ("misaligned ir|visible_mis", ir, visible_mis, flow),
                # Positive control: same modality, so the probe itself must show
                # a peak.  If it does not, the probe or the GT is wrong rather
                # than the cross-modal descriptor.
                ("same-vis vis_gt|vis_mis", rgb_to_gray(visible_gt), visible_mis, flow)):
            for key, report in _run_pair(label, moving, fixed, pair_flow, mind, encoder, args,
                                         encoder_tag):
                if key not in totals:
                    totals[key] = []
                    order.append(key)
                totals[key].append(report)

    print(f"\n== averages over {len(dataset)} pairs "
          f"(window centred on the TRUTH, image pixels) ==")
    for key in order:
        reports = totals[key]
        averaged = {name: sum(r[name] for r in reports) / len(reports)
                    for name in reports[0]}
        print(format_report(key, averaged))
    print("\nlegend: cosGT = similarity at the true match; contrast = cosGT - mean(key sample);"
          "\n  gBeat = share of a random key sample outranking the truth (0 best, 0.5 useless);"
          "\n  lBeat = share of the local window outranking the truth;"
          "\n  win4/win16 = share where the truth wins a +/-4 / +/-16 px window, err = its error;"
          "\n  peak4/peak16 = share where the truth beats everything beyond 4 / 16 px.")


if __name__ == "__main__":
    main()
