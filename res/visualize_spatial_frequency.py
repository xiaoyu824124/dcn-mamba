"""Save coarse correspondence, flow and confidence images for a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .model_factory import build_global_registration
from .vtmot import VTMOTSingleFrameDataset
from .warp import warp


def _uint8(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.detach().float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def _correspondence_image(ir: torch.Tensor, vi: torch.Tensor,
                          probability: torch.Tensor, gt_flow: torch.Tensor,
                          grid_hw: tuple[int, int]) -> Image.Image:
    height, width = ir.shape[-2:]
    grid_h, grid_w = grid_hw
    left = Image.fromarray(_uint8(vi.permute(1, 2, 0)), "RGB")
    right = Image.fromarray(_uint8(ir[0]), "L").convert("RGB")
    canvas = Image.new("RGB", (2 * width, height))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (width, 0))
    draw = ImageDraw.Draw(canvas)
    winners = probability.argmax(dim=-1).reshape(grid_h, grid_w).cpu()
    stride_y, stride_x = height / grid_h, width / grid_w
    # A sparse fixed sample avoids turning thousands of links into a solid block.
    sample_y = torch.linspace(2, grid_h - 3, steps=min(6, grid_h - 4)).round().long()
    sample_x = torch.linspace(2, grid_w - 3, steps=min(8, grid_w - 4)).round().long()
    for row in sample_y:
        for column in sample_x:
            q = int(winners[row, column])
            key_y, key_x = divmod(q, grid_w)
            origin_y = (int(row) + 0.5) * stride_y
            origin_x = (int(column) + 0.5) * stride_x
            match_y = (key_y + 0.5) * stride_y
            match_x = (key_x + 0.5) * stride_x
            gt = gt_flow[:, int(origin_y), int(origin_x)].cpu()
            error = ((match_y - origin_y - float(gt[0])) ** 2 +
                     (match_x - origin_x - float(gt[1])) ** 2) ** 0.5
            colour = (40, 220, 70) if error < 5 else (
                (255, 205, 40) if error < 15 else (240, 50, 50))
            draw.line((origin_x, origin_y, width + match_x, match_y),
                      fill=colour, width=1)
            draw.ellipse((origin_x - 2, origin_y - 2,
                          origin_x + 2, origin_y + 2), fill=colour)
    return canvas


def _flow_image(flow: torch.Tensor) -> Image.Image:
    field = flow.detach().float().cpu().numpy()
    # Fixed display scale across experiments: 24 image pixels per colour span.
    red = np.clip(127.5 + field[0] * 127.5 / 24, 0, 255)
    blue = np.clip(127.5 + field[1] * 127.5 / 24, 0, 255)
    green = np.clip(np.linalg.norm(field, axis=0) * 255 / 24, 0, 255)
    return Image.fromarray(np.stack((red, green, blue), axis=-1).astype(np.uint8),
                           "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", default="data/VTMOT_misaligned")
    parser.add_argument("--split-file", default="data_split/IVF/VTMOT/split.json")
    parser.add_argument("--split", choices=("train", "eval", "test"), default="eval")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.max_samples < 1:
        raise ValueError("max-samples must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = OmegaConf.create(checkpoint["config"])
    if str(config.get("architecture")) != "spatial_frequency":
        raise ValueError("visualizer requires a spatial_frequency checkpoint")
    model = build_global_registration(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = VTMOTSingleFrameDataset(
        args.data_root, split=args.split, split_file=args.split_file,
        frame_stride=args.frame_stride, max_samples=args.max_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            ir = sample["ir"].unsqueeze(0).to(device)
            vi = sample["vi"].unsqueeze(0).to(device)
            output = model(ir, vi)
            grid_hw = tuple(output.match.coarse_flow.shape[-2:])
            prefix = args.output_dir / f"{index:03d}_{sample['sequence']}_{sample['stem']}"
            _correspondence_image(sample["ir"], sample["vi"],
                                  output.match.matching_probability[0],
                                  sample["gt_flow"], grid_hw).save(
                                      f"{prefix}_matches.png")
            _flow_image(output.coarse_flow[0]).save(f"{prefix}_flow.png")
            confidence = F.interpolate(
                output.match.confidence.float(), size=ir.shape[-2:],
                mode="nearest")[0, 0]
            # Same absolute scale for every arm; values >=0.1 appear white.
            Image.fromarray(_uint8(confidence / 0.1), "L").save(
                f"{prefix}_confidence.png")
            aligned = warp(ir, output.coarse_flow)
            Image.fromarray(_uint8(aligned[0, 0]), "L").save(
                f"{prefix}_aligned_ir.png")
            print(f"saved {prefix}_*.png")


if __name__ == "__main__":
    main()
