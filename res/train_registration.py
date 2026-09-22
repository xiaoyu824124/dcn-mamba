"""Train the standalone MIND/global-matching branch on synthetic geometry.

This entrypoint is deliberately independent from the legacy VTMOT fusion
trainer.  Use it as a quick proof that the data convention, all-pairs matcher,
losses and gradients run end-to-end before plugging real VTMOT paths into a
separate dataset adapter.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .encoder import MINDFeatureEncoder
from .global_matcher import GlobalMatcher
from .losses import RegistrationLoss
from .metrics import endpoint_error
from .mind import MINDDescriptor
from .registration_net import MINDDCNRegistration, MINDGlobalRegistration
from .synthetic import SyntheticRegistrationDataset
from .visualize import save_registration_preview


def _as_plain_dict(config) -> Dict:
    return OmegaConf.to_container(config, resolve=True)


def build_model(config, use_dcn: bool) -> torch.nn.Module:
    mind = MINDDescriptor(**_as_plain_dict(config.mind))
    encoder = MINDFeatureEncoder(in_channels=mind.channels, **_as_plain_dict(config.encoder))
    matcher = GlobalMatcher(**_as_plain_dict(config.global_matcher))
    coarse = MINDGlobalRegistration(mind=mind, encoder=encoder, matcher=matcher)
    return (MINDDCNRegistration(coarse, use_residual_flow_head=bool(config.dcn.use_residual_flow_head))
            if use_dcn else coarse)


def _save_checkpoint(path: Path, *, step: int, model: torch.nn.Module,
                     optimizer: torch.optim.Optimizer, config) -> None:
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": _as_plain_dict(config)}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic smoke training for standalone IR--visible registration")
    parser.add_argument("--config", default="res/configs/registration.yaml")
    parser.add_argument("--output-dir", default="res_runs/synthetic")
    parser.add_argument("--steps", type=int, default=None, help="override synthetic training steps")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--use-dcn", action=argparse.BooleanOptionalAction, default=None,
                        help="enable the DCN reconstruction refinement during smoke training")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    train_config = config.synthetic_train
    steps = int(args.steps if args.steps is not None else train_config.steps)
    if steps < 1:
        raise ValueError("--steps must be positive")
    use_dcn = bool(config.ablation.use_dcn if args.use_dcn is None else args.use_dcn)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    seed = int(train_config.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dataset = SyntheticRegistrationDataset(**_as_plain_dict(config.synthetic_data))
    loader = DataLoader(dataset, batch_size=int(train_config.batch_size), shuffle=True,
                        num_workers=int(train_config.num_workers), pin_memory=device.type == "cuda",
                        drop_last=True)
    model = build_model(config, use_dcn=use_dcn).to(device)
    loss_fn = RegistrationLoss(config.loss.weights, charbonnier_eps=float(config.loss.charbonnier_eps)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config.lr),
                                  weight_decay=float(train_config.weight_decay))
    amp = bool(train_config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")
    print(f"device={device}  dcn={use_dcn}  samples={len(dataset)}  steps={steps}")
    iterator = iter(loader)
    metrics_path = output_dir / "metrics.jsonl"
    model.train()
    for step in range(start_step + 1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(batch["ir"], batch["vi"])
            if use_dcn:
                coarse, refined = output
                aligned_ir, final_flow = refined.refined_aligned_ir, refined.final_flow
                coarse_flow = coarse.coarse_flow
            else:
                aligned_ir, coarse_flow, final_flow = output.coarse_aligned_ir, output.coarse_flow, None
            losses = loss_fn(aligned_ir=aligned_ir, visible=batch["vi"], coarse_flow=coarse_flow,
                             final_flow=final_flow, gt_flow=batch["gt_flow"],
                             valid_mask=batch["valid_mask"])
        scaler.scale(losses.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        active_flow = final_flow if final_flow is not None else coarse_flow
        with torch.no_grad():
            epe = endpoint_error(active_flow, batch["gt_flow"], batch["valid_mask"])
        record = {"step": step, "loss": float(losses.total.detach()), "epe": float(epe),
                  "flow": float(losses.flow.detach()), "mind": float(losses.mind.detach()),
                  "edge": float(losses.edge.detach()), "smooth": float(losses.smooth.detach())}
        if step == 1 or step % int(train_config.log_every) == 0 or step == steps:
            print("step={step:5d} loss={loss:.5f} epe={epe:.3f} flow={flow:.5f} mind={mind:.5f}".format(**record))
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
        if step % int(train_config.preview_every) == 0 or step == steps:
            save_registration_preview(output_dir / f"preview_{step:06d}.png", moving_ir=batch["ir"],
                                      visible=batch["vi"], aligned_ir=aligned_ir, predicted_flow=active_flow,
                                      gt_flow=batch["gt_flow"])
        if step % int(train_config.checkpoint_every) == 0 or step == steps:
            _save_checkpoint(output_dir / "last.pt", step=step, model=model,
                             optimizer=optimizer, config=config)
    print(f"completed: {output_dir.resolve()}\\last.pt")


if __name__ == "__main__":
    main()
