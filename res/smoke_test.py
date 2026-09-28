"""Smoke-test the iterative pipeline frame by frame at the configured resolution.

A silent death inside the training entrypoint's initial validation says nothing
about *where* it happened, and reproducing it there costs a multi-minute
validation first.  This builds the same model from the same config stack and runs
the same evaluation call, printing one line per frame with the per-frame peak and
live allocation, so a crash identifies its own frame number and an allocation
that grows across frames is visible instead of hidden inside a cumulative
high-water mark.

Three modes, in increasing fidelity:

    python -B -m res.smoke_test --frames 4
        synthetic frames, random weights
    python -B -m res.smoke_test --frames 4 --backward
        also runs the real training loss and one backward step
    python -B -m res.smoke_test --frames 8 --real-data \
        --checkpoint res_runs/structural_iterative_r3/best.pt
        the real dataset and a real checkpoint, one evaluate() call per frame

The last form reproduces the training entrypoint's initial validation exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from .checkpoint import load_registration_state
from .evaluate_vtmot import evaluate
from .losses import RegistrationLoss
from .model_factory import build_global_registration
from .vtmot import VTMOTSingleFrameDataset


def describe(prefix: str, index: int, report: dict, device: torch.device) -> None:
    rounds = [round(report[f"refine_round{step}_epe_px"], 4) for step in (1, 2, 3)
              if f"refine_round{step}_epe_px" in report]
    if device.type == "cuda":
        memory = (f"peak={torch.cuda.max_memory_allocated() / 2**20:.0f}MiB "
                  f"live={torch.cuda.memory_allocated() / 2**20:.0f}MiB "
                  f"reserved={torch.cuda.memory_reserved() / 2**20:.0f}MiB")
    else:
        memory = "cpu"
    print(f"  {prefix} {index}: coarse_epe={report['coarse_epe_px']:.4f} "
          f"epe={report['epe_px']:.4f} rounds={rounds}  {memory}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Iterative pipeline smoke test")
    parser.add_argument("--config", default="res/configs/registration.yaml")
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--fov", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"),
                        default=(480, 640))
    parser.add_argument("--backward", action="store_true",
                        help="also run the training loss and one backward step")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--real-data", action="store_true",
                        help="iterate the real VTMOT split instead of synthetic frames")
    parser.add_argument("--data-root", default="data/VTMOT_misaligned")
    parser.add_argument("--split-file", default="data_split/IVF/VTMOT/split.json")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--frame-stride", type=int, default=10)
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
    print(f"fov={tuple(args.fov)} frames={args.frames} backward={args.backward} "
          f"real_data={args.real_data} checkpoint={args.checkpoint}", flush=True)

    torch.manual_seed(0)
    model = build_global_registration(config).to(device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        loaded = load_registration_state(model, checkpoint["model"])
        print(f"checkpoint kept={loaded['kept']} newly built={len(loaded['untrained'])}",
              flush=True)

    if args.real_data:
        dataset = VTMOTSingleFrameDataset(
            args.data_root, split=args.split, split_file=args.split_file,
            target_hw=tuple(config.vtmot_data.target_hw),
            crop_hw=tuple(config.vtmot_data.crop_hw),
            frame_stride=args.frame_stride)
        print(f"real data: {len(dataset)} samples at stride {args.frame_stride}",
              flush=True)
        for index in range(min(args.frames, len(dataset))):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            # dataset[index] is unbatched; collate turns it into the batch the
            # loader would have produced, which is what evaluate expects.
            batch = default_collate([dataset[index]])
            print(f"  loaded sample {index} ({batch['sequence']}/{batch['stem']})",
                  flush=True)
            report = evaluate(model.eval(), [batch], device,
                              moving_source=str(config.vtmot_train.moving_source))
            describe("real", index, report, device)
        print("SMOKE OK", flush=True)
        return 0

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
        describe("frame", index, report, device)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
