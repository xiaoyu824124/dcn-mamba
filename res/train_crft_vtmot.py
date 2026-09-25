"""Fine-tune the official CRFT backbone on VTMOT dense flow supervision.

This is an adapted VTMOT baseline, not a reproduction of CRFT's RoadScene
training schedule or of its paper metrics.  The official source stays in a
separate directory and is imported without edits.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .crft_baseline import (DEFAULT_MODEL_HW, build_crft, evaluate_crft,
                            load_crft_weights, masked_flow_loss, run_crft,
                            validate_model_hw)
from .vtmot import VTMOTSingleFrameDataset


def _save(path: Path, step: int, model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, model_hw: tuple[int, int],
          best_epe: float) -> None:
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "model_hw": model_hw,
                "best_epe": best_epe}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapted CRFT baseline training on VTMOT")
    parser.add_argument("--crft-root", type=Path, required=True)
    parser.add_argument("--data-root", default="data/VTMOT_misaligned")
    parser.add_argument("--split-file", default="data_split/IVF/VTMOT/split.json")
    parser.add_argument("--output-dir", type=Path, default=Path("res_runs/crft_vtmot_pilot"))
    parser.add_argument("--model-hw", type=int, nargs=2, default=DEFAULT_MODEL_HW,
                        metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-frame-stride", type=int, default=50)
    parser.add_argument("--eval-max-samples", type=int, default=0,
                        help="limit validation frames for a quick smoke test; 0 uses all")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--init", type=Path, default=None,
                        help="optional official RoadScene checkpoint")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume a checkpoint produced by this entrypoint")
    args = parser.parse_args()
    if args.init is not None and args.resume is not None:
        raise ValueError("choose at most one of --init and --resume")
    if args.steps < 1 or args.eval_every < 1 or args.eval_frame_stride < 1:
        raise ValueError("steps and evaluation intervals must be positive")
    if args.lr <= 0 or args.num_workers < 0 or args.eval_max_samples < 0:
        raise ValueError("learning rate must be positive and counts nonnegative")
    model_hw = validate_model_hw(tuple(args.model_hw))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    train_data = VTMOTSingleFrameDataset(
        args.data_root, split="train", split_file=args.split_file)
    eval_data = VTMOTSingleFrameDataset(
        args.data_root, split="eval", split_file=args.split_file,
        frame_stride=args.eval_frame_stride, max_samples=args.eval_max_samples)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, generator=generator,
                              pin_memory=device.type == "cuda")
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=device.type == "cuda")
    model = build_crft(args.crft_root, device)
    if args.init is not None:
        saved_hw = load_crft_weights(model, args.init)
        if saved_hw is not None and saved_hw != model_hw:
            print(f"init: transferring weights from model_hw={saved_hw} to {model_hw}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_step = 0
    best_epe = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if tuple(checkpoint["model_hw"]) != model_hw:
            raise ValueError("resume checkpoint model-hw does not match")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        best_epe = float(checkpoint["best_epe"])
        if args.output_dir.resolve() != args.resume.parent.resolve():
            raise ValueError("--resume requires --output-dir to be the checkpoint directory")
    elif any((args.output_dir / name).exists() for name in ("best.pt", "last.pt", "metrics.jsonl")):
        raise ValueError(f"output directory already contains a run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    print(f"CRFT VTMOT train={len(train_data)} eval={len(eval_data)} "
          f"model_hw={model_hw} device={device} start={start_step} steps={args.steps}")
    if args.resume is None:
        initial = evaluate_crft(model, eval_loader, device, model_hw)
        best_epe = initial["epe_px"]
        _save(args.output_dir / "best.pt", 0, model, optimizer, model_hw, best_epe)
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"step": 0, **{f"val_{k}": v for k, v in initial.items()}}) + "\n")
        print(f"init eval epe={best_epe:.3f} zero={initial['zero_flow_epe_px']:.3f}")
    # Keep data shuffling reproducible regardless of model initialization.
    torch.manual_seed(args.seed)
    iterator = iter(train_loader)
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        ir, vi = batch["ir"].to(device), batch["vi"].to(device)
        target, valid = batch["gt_flow"].to(device), batch["valid_mask"].to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted, coarse = run_crft(model, ir, vi, model_hw)
        fine_loss = masked_flow_loss(predicted, target, valid)
        coarse_loss = masked_flow_loss(coarse, target, valid)
        loss = fine_loss + 0.25 * coarse_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite CRFT loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        record = {"step": step, "loss": float(loss.detach()),
                  "fine_epe": float(fine_loss.detach()),
                  "coarse_epe": float(coarse_loss.detach()),
                  "peak_mem_mib": (torch.cuda.max_memory_allocated() / 2**20
                                   if device.type == "cuda" else 0.0)}
        if step % args.eval_every == 0 or step == args.steps:
            report = evaluate_crft(model, eval_loader, device, model_hw)
            record.update({f"val_{key}": value for key, value in report.items()})
            if report["epe_px"] < best_epe:
                best_epe = report["epe_px"]
                _save(args.output_dir / "best.pt", step, model, optimizer, model_hw, best_epe)
            print(f"step={step:5d} loss={record['loss']:.3f} "
                  f"eval_epe={report['epe_px']:.3f} coarse={report['coarse_epe_px']:.3f} "
                  f"zero={report['zero_flow_epe_px']:.3f}")
        elif step == 1 or step % 25 == 0:
            print(f"step={step:5d} loss={record['loss']:.3f} "
                  f"fine={record['fine_epe']:.3f} coarse={record['coarse_epe']:.3f}")
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
    _save(args.output_dir / "last.pt", args.steps, model, optimizer, model_hw, best_epe)
    print(f"completed: best eval EPE={best_epe:.3f}; output={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
