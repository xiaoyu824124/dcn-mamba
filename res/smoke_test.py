"""Smoke-test the iterative pipeline frame by frame at the configured resolution.

A silent death inside the training entrypoint's initial validation says nothing
about *where* it happened.  This builds the same model from the same config stack
with no checkpoint, pushes synthetic frames through the same evaluation path and
prints a line after every frame, so a crash identifies its own frame number.

Usage:
    python -B -m res.smoke_test --frames 4
    python -B -m res.smoke_test --frames 1 --backward
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .evaluate_vtmot import evaluate
from .losses import RegistrationLoss
from .model_factory import build_global_registration


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative pipeline smoke test")
    parser.add_argument("--config", default="res/configs/registration.yaml")
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--fov", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"),
                        default=(480, 640))
    parser.add_argument("--backward", action="store_true",
                        help="also run the training loss and one backward step")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    overlays = args.overlay or [
        Path("res/configs/stage0_structural_prior.yaml"),
        Path("res/configs/ab_spatial_prior32.yaml"),
        Path("res/configs/stage3_structural_iterative.yaml")]
    config = OmegaConf.merge(OmegaConf.load(args.config),
                             *(OmegaConf.load(path) for path in overlays))
    print(f"device={device} config={args.config} overlays={[str(p) for p in overlays]}")
    print(f"fov={tuple(args.fov)} frames={args.frames} backward={args.backward}")

    torch.manual_seed(0)
    model = build_global_registration(config).to(device)
    height, width = int(args.fov[0]), int(args.fov[1])
    for index in range(args.frames):
        batch = {"ir": torch.rand(1, 1, height, width, device=device),
                 "vi": torch.rand(1, 3, height, width, device=device),
                 "gt_flow": torch.zeros(1, 2, height, width, device=device),
                 "valid_mask": torch.ones(1, 1, height, width, device=device),
                 "gt_h": torch.eye(3, device=device).unsqueeze(0)}
        if device.type == "cuda":
            # Per-frame peak, so a growth across frames is visible as growth
            # rather than hidden inside a monotone high-water mark.
            torch.cuda.reset_peak_memory_stats()
        report = evaluate(model.eval(), [batch], device)
        if device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated() / 2**20
            reserved = torch.cuda.memory_reserved() / 2**20
            live = torch.cuda.memory_allocated() / 2**20
        else:
            allocated = reserved = live = 0.0
        print(f"  frame {index}: coarse_epe={report['coarse_epe_px']:.4f} "
              f"epe={report['epe_px']:.4f} rounds="
              f"{[round(report[f'refine_round{step}_epe_px'], 4) for step in (1, 2, 3)]}"
              f"  peak={allocated:.0f}MiB live={live:.0f}MiB reserved={reserved:.0f}MiB",
              flush=True)
        if args.backward:
            model.train()
            output = model(batch["ir"], batch["vi"], return_features=True)
            losses = RegistrationLoss(config.loss.weights).to(device)(
                aligned_ir=(output.final_aligned_ir if output.final_aligned_ir is not None
                            else output.coarse_aligned_ir),
                visible=batch["vi"], coarse_flow=output.coarse_flow,
                final_flow=output.final_flow, global_flow=output.global_flow,
                gt_flow=batch["gt_flow"], valid_mask=batch["valid_mask"],
                predicted_affine_yx=output.affine_yx, gt_h=batch["gt_h"],
                affine_feature_hw=tuple(output.confidence_1_8.shape[-2:]),
                match=output.match, local_match=output.local_match,
                coarse_local_match=output.coarse_local_match,
                refinement=output.refinement)
            losses.total.backward()
            print(f"    backward ok: total={float(losses.total.detach()):.4f} "
                  f"refine={float(losses.refine.detach()):.4f} "
                  f"refine_match={float(losses.refine_match.detach()):.4f}", flush=True)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
