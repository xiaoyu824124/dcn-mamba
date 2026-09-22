"""Fine-tune the standalone coarse registration model on real VTMOT frames.

This is intentionally separate from ``train.py``: it trains only MIND, the
shared feature encoder and the global matcher, with VTMOT dense GT flow.  It
does not load the fusion model, RAFT, temporal memory or DCN reconstruction.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .evaluate_vtmot import evaluate
from .losses import RegistrationLoss
from .metrics import endpoint_error
from .model_factory import build_global_registration
from .vtmot import VTMOTSingleFrameDataset


def _save(path: Path, step: int, model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, config) -> None:
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": OmegaConf.to_container(config, resolve=True)}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real VTMOT single-frame global-registration fine-tuning")
    parser.add_argument("--config", default="res/configs/registration.yaml")
    parser.add_argument("--data-root", default="data/VTMOT_misaligned")
    parser.add_argument("--split-file", default="data_split/IVF/VTMOT/split.json")
    parser.add_argument("--output-dir", default="res_runs/vtmot_global")
    parser.add_argument("--run", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--crop-hw", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=None,
                        help="override the configured common crop; default is the 3090 full-frame setting")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override batch size; full-frame all-pairs matching normally uses 1")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="override data-loader workers; use 0 for Windows debugging")
    parser.add_argument("--init", type=Path, default=None,
                        help="optional compatible registration checkpoint; does not resume optimiser state")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume a checkpoint produced by this entrypoint")
    args = parser.parse_args()
    if args.init is not None and args.resume is not None:
        raise ValueError("choose at most one of --init and --resume")
    config = OmegaConf.load(args.config)
    data_config, train_config = config.vtmot_data, config.vtmot_train
    steps = int(args.steps if args.steps is not None else
                (train_config.pilot_steps if args.run == "pilot" else train_config.full_steps))
    if steps < 1:
        raise ValueError("steps must be positive")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    random.seed(int(train_config.seed))
    torch.manual_seed(int(train_config.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(train_config.seed))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    target_hw = tuple(data_config.target_hw)
    crop_hw = tuple(args.crop_hw) if args.crop_hw is not None else tuple(data_config.crop_hw)
    batch_size = int(args.batch_size if args.batch_size is not None else train_config.batch_size)
    num_workers = int(args.num_workers if args.num_workers is not None else train_config.num_workers)
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch size must be positive and num-workers must be non-negative")
    train_dataset = VTMOTSingleFrameDataset(
        args.data_root, split="train", split_file=args.split_file, target_hw=target_hw,
        crop_hw=crop_hw, random_crop=True, frame_stride=int(data_config.train_frame_stride))
    eval_dataset = VTMOTSingleFrameDataset(
        args.data_root, split="eval", split_file=args.split_file, target_hw=target_hw,
        crop_hw=crop_hw, random_crop=False, frame_stride=int(data_config.eval_frame_stride))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              pin_memory=device.type == "cuda")
    eval_loader = DataLoader(eval_dataset, batch_size=int(train_config.eval_batch_size), shuffle=False,
                             num_workers=0, pin_memory=device.type == "cuda")
    model = build_global_registration(config).to(device)
    loss_fn = RegistrationLoss(config.loss.weights, charbonnier_eps=float(config.loss.charbonnier_eps)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config.lr),
                                  weight_decay=float(train_config.weight_decay))
    amp = bool(train_config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
    elif args.init is not None:
        checkpoint = torch.load(args.init, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")
    metrics_file = output_dir / "metrics.jsonl"
    print(f"device={device} run={args.run} steps={steps} train={len(train_dataset)} eval={len(eval_dataset)} crop={crop_hw} batch={batch_size} workers={num_workers}")
    iterator, best_ratio = iter(train_loader), float("inf")
    model.train()
    for step in range(start_step + 1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        ir, vi = batch["ir"].to(device, non_blocking=True), batch["vi"].to(device, non_blocking=True)
        target, valid = batch["gt_flow"].to(device, non_blocking=True), batch["valid_mask"].to(device, non_blocking=True)
        gt_h = batch["gt_h"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(ir, vi)
            losses = loss_fn(aligned_ir=output.coarse_aligned_ir, visible=vi,
                             coarse_flow=output.coarse_flow, gt_flow=target, valid_mask=valid,
                             predicted_affine_yx=output.affine_yx, gt_h=gt_h)
        scaler.scale(losses.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            train_epe = endpoint_error(output.coarse_flow, target, valid)
        record = {"step": step, "train_loss": float(losses.total.detach()),
                  "train_epe": float(train_epe), "flow": float(losses.flow.detach()),
                  "affine": float(losses.affine.detach())}
        if step == 1 or step % int(train_config.log_every) == 0:
            print("step={step:5d} loss={train_loss:.5f} train_epe={train_epe:.3f} affine={affine:.4f}".format(**record))
        if step % int(train_config.validation_every) == 0 or step == steps:
            validation = evaluate(model, eval_loader, device)
            record.update({f"val_{name}": value for name, value in validation.items()})
            print("  eval epe={epe_px:.3f} zero={zero_flow_epe_px:.3f} ratio={relative_epe:.3f} "
                  "corners={affine_corner_epe_px:.2f}px cycle={affine_gt_inverse_cycle_px:.2f}px".format(**validation))
            if validation["relative_epe"] < best_ratio:
                best_ratio = validation["relative_epe"]
                _save(output_dir / "best.pt", step, model, optimizer, config)
            model.train()
        with metrics_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        if step % int(train_config.checkpoint_every) == 0 or step == steps:
            _save(output_dir / "last.pt", step, model, optimizer, config)
    print(f"completed: best relative EPE={best_ratio:.4f}; output={output_dir.resolve()}")


if __name__ == "__main__":
    main()
