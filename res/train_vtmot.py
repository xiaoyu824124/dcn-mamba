"""Fine-tune single-frame coarse-to-fine registration on real VTMOT frames.

This is separate from ``train.py`` and does not load the fusion model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .evaluate_vtmot import evaluate
from .losses import RegistrationLoss
from .matching import matching_diagnostics
from .metrics import endpoint_error
from .model_factory import build_global_registration
from .motion_diagnostics import translate_moving_for_stress
from .vtmot import VTMOTSingleFrameDataset, registration_moving_image


def freeze_coarse_parameters(model: torch.nn.Module) -> None:
    """Hold the 1/8 field fixed while comparing alternative 1/4 refiners."""
    if model.local_matcher is None and getattr(model, "iterative_refinement",
                                              None) is None:
        raise ValueError("freeze_coarse requires a 1/4 stage (local matcher or "
                         "iterative refinement)")
    model.encoder.requires_grad_(False)
    model.matcher.requires_grad_(False)
    if model.coarse_transformer is not None:
        model.coarse_transformer.requires_grad_(False)
    for name in ("coarse_decoder", "coarse_refiner"):
        module = getattr(model, name, None)
        if module is not None:
            module.requires_grad_(False)


def freeze_local_refinement_parameters(model: torch.nn.Module) -> None:
    """Train local correspondence features without adapting the residual head."""
    if model.local_matcher is None:
        raise ValueError("freeze_local_refinement requires an enabled local matcher")
    model.local_matcher.refinement.requires_grad_(False)


def freeze_fine_interaction_parameters(model: torch.nn.Module) -> None:
    """Train the residual head against a fixed learned local descriptor."""
    if model.fine_interaction is None:
        raise ValueError("freeze_fine_interaction requires an enabled fine interaction")
    model.fine_interaction.requires_grad_(False)


def freeze_except_ir_adapter_parameters(model: torch.nn.Module) -> None:
    """Hold all established registration weights fixed for adapter ablation."""
    if model.ir_feature_adapter is None:
        raise ValueError("ir_adapter_only requires an enabled IR feature adapter")
    model.requires_grad_(False)
    model.ir_feature_adapter.requires_grad_(True)


def _save(path: Path, step: int, model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, config, best_ratio: float,
          provenance: dict | None = None) -> None:
    """Write a checkpoint, recording how it was produced as well.

    A warm start is not recoverable from the weights: ``--init`` was previously
    written nowhere, so a run could not be reproduced from its own checkpoint.
    The command line and the two starting paths go in alongside the config, and
    are ignored by every existing reader.
    """
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_ratio": best_ratio,
                "config": OmegaConf.to_container(config, resolve=True),
                "provenance": dict(provenance or {})}, path)


def sample_large_translation(settings, *, seed: int, step: int
                             ) -> tuple[int, int] | None:
    """Draw a reproducible one-axis displacement independent of loader RNG."""
    probability = float(settings.get("probability", 0.0))
    minimum = int(settings.get("min_abs_px", 16))
    maximum = int(settings.get("max_abs_px", 64))
    padding_mode = str(settings.get("padding_mode", "reflection"))
    if not 0 <= probability <= 1:
        raise ValueError("large_translation probability must be in [0,1]")
    if minimum < 1 or maximum < minimum:
        raise ValueError("large_translation requires 1 <= min_abs_px <= max_abs_px")
    if padding_mode not in ("zeros", "border", "reflection"):
        raise ValueError("large_translation padding_mode must be zeros, border or reflection")
    if step < 0:
        raise ValueError("step must be non-negative")
    rng = random.Random((int(seed) + 1) * 1000003 + int(step))
    if rng.random() >= probability:
        return None
    magnitude = rng.randint(minimum, maximum) * (1 if rng.randrange(2) else -1)
    return (magnitude, 0) if rng.randrange(2) else (0, magnitude)


def sample_init_flow(settings, gt_flow: torch.Tensor, *, seed: int, step: int
                     ) -> tuple[torch.Tensor | None, str]:
    """Mixed starting points for the 1/4 loop, used as residual supervision.

    Training the loop only from the model's own coarse field cannot teach it what
    to do when the input is already right, and evaluating it only from the ground
    truth would be passed by a head that never moves.  So a fraction of steps start
    from a *known* perturbation of the truth, and the per-round flow loss then
    demands the matching correction back:

    ``truth``        start on the truth, so the correct correction is zero;
    ``translation``  truth plus one constant offset, so the correct correction is
                     its negation everywhere -- exactly the case a frozen coarse
                     field still needs when it is globally off;
    ``local``        truth plus a smooth bounded field, so the correction varies
                     spatially.

    Returns ``(None, "coarse")`` for the remaining steps, which is what inference
    always uses.  The choice is a deterministic function of ``seed`` and ``step``,
    so a run stays reproducible.
    """
    if settings is None:
        return None, "coarse"
    probability = float(settings.get("probability", 0.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("init_flow_mix probability must be in [0, 1]")
    magnitudes = [float(value) for value in
                  settings.get("translation_px", (4.0, 8.0, 16.0))]
    if not magnitudes or min(magnitudes) <= 0.0:
        raise ValueError("init_flow_mix translation_px must be positive magnitudes")
    truth_weight = float(settings.get("truth_weight", 0.4))
    if not 0.0 <= truth_weight <= 1.0:
        raise ValueError("init_flow_mix truth_weight must be in [0, 1]")
    local_px = float(settings.get("local_px", 8.0))
    rng = random.Random(seed * 1_000_003 + step)
    if rng.random() >= probability:
        return None, "coarse"
    roll = rng.random()
    if roll < truth_weight:
        return gt_flow, "truth"
    if roll < truth_weight + (1.0 - truth_weight) / 2.0 or local_px <= 0.0:
        # Eight unit directions, not just the diagonals: a correction the head
        # only ever sees along one axis is a correction it cannot generalise,
        # and the magnitude is the Euclidean residual in every direction.
        magnitude = rng.choice(magnitudes)
        direction = rng.choice(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                                (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)))
        norm = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        offset = torch.tensor([direction[0] / norm * magnitude,
                               direction[1] / norm * magnitude],
                              device=gt_flow.device, dtype=gt_flow.dtype)
        return gt_flow + offset.view(1, 2, 1, 1), "translation"
    # Seeded from the same rng, so the whole choice stays a function of
    # (seed, step) and a resumed run replays it exactly.
    generator = torch.Generator(device=gt_flow.device).manual_seed(rng.randrange(2 ** 31))
    blocks = torch.randn(1, 2, max(gt_flow.shape[-2] // 16, 1),
                         max(gt_flow.shape[-1] // 16, 1),
                         device=gt_flow.device, dtype=gt_flow.dtype, generator=generator)
    smooth = blocks.repeat_interleave(16, dim=-2).repeat_interleave(16, dim=-1)
    smooth = smooth[..., :gt_flow.shape[-2], :gt_flow.shape[-1]]
    smooth = torch.nn.AvgPool2d(9, stride=1, padding=4)(smooth)
    smooth = smooth / smooth.abs().amax().clamp_min(1e-6) * local_px
    return gt_flow + smooth, "local"


def main() -> None:
    parser = argparse.ArgumentParser(description="Real VTMOT single-frame coarse-to-fine registration")
    parser.add_argument("--config", default="res/configs/registration.yaml")
    parser.add_argument("--overlay", type=Path, action="append", default=[],
                        help="optional YAML overlay; repeat to add stage-specific settings")
    parser.add_argument("--data-root", default="data/VTMOT_misaligned")
    parser.add_argument("--split-file", default="data_split/IVF/VTMOT/split.json")
    parser.add_argument("--output-dir", default="res_runs/vtmot_global")
    parser.add_argument("--run", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--crop-hw", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=None,
                        help="override the configured common crop; default is the A4000 full-frame setting")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override batch size; full-frame all-pairs matching normally uses 1")
    parser.add_argument("--lr", type=float, default=None,
                        help="override the configured learning rate")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="override data-loader workers; use 0 for Windows debugging")
    parser.add_argument("--moving-source", choices=("ir", "visible_gt"), default=None,
                        help="moving image: infrared (default) or aligned visible grayscale for warmup")
    parser.add_argument("--init", type=Path, default=None,
                        help="optional compatible registration checkpoint; does not resume optimiser state")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume a checkpoint produced by this entrypoint")
    args = parser.parse_args()
    if args.init is not None and args.resume is not None:
        raise ValueError("choose at most one of --init and --resume")
    # Written into every checkpoint: the weights alone cannot say which warm
    # start they came from, which is exactly what a controlled comparison needs.
    provenance = {"argv": sys.argv[1:],
                  "init": str(args.init) if args.init is not None else None,
                  "resume": str(args.resume) if args.resume is not None else None}
    config = OmegaConf.merge(OmegaConf.load(args.config),
                             *(OmegaConf.load(path) for path in args.overlay))
    data_config, train_config = config.vtmot_data, config.vtmot_train
    moving_source = args.moving_source or str(train_config.get("moving_source", "ir"))
    if moving_source not in ("ir", "visible_gt"):
        raise ValueError(f"unknown moving source: {moving_source}")
    # Persist the actual source so a checkpoint cannot silently mislabel its run.
    train_config.moving_source = moving_source
    steps = int(args.steps if args.steps is not None else
                (train_config.pilot_steps if args.run == "pilot" else train_config.full_steps))
    if steps < 1:
        raise ValueError("steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; check the A4000 environment")
    device = torch.device(args.device)
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
    large_translation = train_config.get("large_translation")
    if large_translation is not None:
        if moving_source != "ir" or batch_size != 1:
            raise ValueError("large_translation requires moving_source=ir and batch-size=1")
        sample_large_translation(large_translation, seed=int(train_config.seed), step=0)
        if int(large_translation.get("max_abs_px", 64)) >= min(crop_hw):
            raise ValueError("large_translation max_abs_px must be smaller than the crop")
    init_flow_mix = train_config.get("init_flow_mix")
    if init_flow_mix is not None:
        if batch_size != 1:
            raise ValueError("init_flow_mix requires batch-size=1")
        # Validate the settings once here, so a typo fails at startup instead of
        # on the first training step.
        sample_init_flow(init_flow_mix, torch.zeros(1, 2, *crop_hw),
                         seed=int(train_config.seed), step=0)
        largest = max(float(value) for value in
                      init_flow_mix.get("translation_px", (4.0, 8.0, 16.0)))
        rounds = int(config.iterative_refinement.get("iterations", 3))
        # The loop runs at 1/4 resolution, so one cell of its step bound is four
        # input pixels.  A start further out than rounds * step cannot be reached
        # within one run, which is worth saying out loud rather than assuming.
        stride = crop_hw[0] / max(crop_hw[0] // 4, 1)
        step_px = float(config.iterative_refinement.get("max_step_cells", 4.0)) * stride
        if largest > step_px * rounds + 1e-6:
            print(f"note: init_flow_mix samples up to {largest:.0f}px while "
                  f"{rounds} rounds of {step_px:.0f}px cover {step_px * rounds:.0f}px; "
                  f"the remainder is beyond one run's reach", flush=True)
    train_dataset = VTMOTSingleFrameDataset(
        args.data_root, split="train", split_file=args.split_file, target_hw=target_hw,
        crop_hw=crop_hw, random_crop=True, frame_stride=int(data_config.train_frame_stride),
        include_rgb_gt=moving_source == "visible_gt")
    eval_dataset = VTMOTSingleFrameDataset(
        args.data_root, split="eval", split_file=args.split_file, target_hw=target_hw,
        crop_hw=crop_hw, random_crop=False, frame_stride=int(data_config.eval_frame_stride),
        include_rgb_gt=moving_source == "visible_gt")
    # Keep sample order independent of the model's parameter count, so the
    # coarse-only and SA-CA runs see the same images in a controlled ablation.
    train_generator = torch.Generator().manual_seed(int(train_config.seed))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              pin_memory=device.type == "cuda", generator=train_generator)
    eval_loader = DataLoader(eval_dataset, batch_size=int(train_config.eval_batch_size), shuffle=False,
                             num_workers=0, pin_memory=device.type == "cuda")
    model = build_global_registration(config).to(device)
    freeze_coarse = bool(train_config.get("freeze_coarse", False))
    if freeze_coarse:
        freeze_coarse_parameters(model)
    freeze_local_refinement = bool(train_config.get("freeze_local_refinement", False))
    if freeze_local_refinement:
        freeze_local_refinement_parameters(model)
    freeze_fine_interaction = bool(train_config.get("freeze_fine_interaction", False))
    if freeze_fine_interaction:
        freeze_fine_interaction_parameters(model)
    ir_adapter_only = bool(train_config.get("ir_adapter_only", False))
    if ir_adapter_only:
        if moving_source != "ir":
            raise ValueError("ir_adapter_only requires moving_source=ir")
        freeze_except_ir_adapter_parameters(model)
    loss_fn = RegistrationLoss(config.loss.weights,
                               charbonnier_eps=float(config.loss.charbonnier_eps),
                               match_focal_gamma=float(config.loss.get("match_focal_gamma", 0.0)),
                               appearance_window_radius=int(config.loss.get("appearance_window_radius", 4)),
                               appearance_temperature=float(config.loss.get("appearance_temperature", 0.07))
                               ).to(device)
    learning_rate = float(args.lr if args.lr is not None else train_config.lr)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad), lr=learning_rate,
                                  weight_decay=float(train_config.weight_decay))
    amp = bool(train_config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        saved_source = str(checkpoint.get("config", {}).get("vtmot_train", {}).get(
            "moving_source", "ir"))
        if saved_source != moving_source:
            raise ValueError(f"resume moving source is {saved_source}, but this run uses "
                             f"{moving_source}; pass the original overlay or use --init")
        saved_translation = checkpoint.get("config", {}).get("vtmot_train", {}).get(
            "large_translation")
        current_translation = (OmegaConf.to_container(large_translation, resolve=True)
                               if large_translation is not None else None)
        if saved_translation != current_translation:
            raise ValueError("resume large_translation settings differ from checkpoint")
        # The starting-point mix is part of the experiment, so a resume must not
        # silently switch it either.
        saved_mix = checkpoint.get("config", {}).get("vtmot_train", {}).get("init_flow_mix")
        current_mix = (OmegaConf.to_container(init_flow_mix, resolve=True)
                       if init_flow_mix is not None else None)
        if saved_mix != current_mix:
            raise ValueError("resume init_flow_mix settings differ from checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
    elif args.init is not None:
        checkpoint = torch.load(args.init, map_location=device, weights_only=True)
        saved_source = str(checkpoint.get("config", {}).get("vtmot_train", {}).get(
            "moving_source", "ir"))
        if saved_source != moving_source:
            print(f"init: moving source {saved_source} -> {moving_source}")
        # Warm start: architecture knobs (width, key scale, ...) may add or
        # resize tensors, so keep every tensor whose shape still matches and
        # report the rest instead of refusing to start.  ``--resume`` stays
        # strict because it must match exactly.
        current = model.state_dict()
        compatible = {name: value for name, value in checkpoint["model"].items()
                      if name in current and current[name].shape == value.shape}
        skipped = sorted(name for name in checkpoint["model"] if name not in compatible)
        report = model.load_state_dict(compatible, strict=False)
        fresh = sorted(report.missing_keys) + sorted(report.unexpected_keys)
        if skipped or fresh:
            print(f"init: warm start from {args.init} | kept={len(compatible)} "
                  f"skipped={len(skipped)} fresh={len(fresh)}")
            for name in skipped[:8]:
                print(f"  skipped {name}")
            for name in fresh[:8]:
                print(f"  fresh   {name}")

    output_dir = Path(args.output_dir)
    if args.resume is not None and output_dir.resolve() != args.resume.parent.resolve():
        raise ValueError("--resume requires --output-dir to be the checkpoint directory; "
                         "use --init for a new run")
    if args.resume is None and any((output_dir / name).exists()
                                   for name in ("metrics.jsonl", "best.pt", "last.pt")):
        raise ValueError(f"output directory already contains a run: {output_dir}; "
                         "choose a new directory or use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "config.yaml")
    metrics_file = output_dir / "metrics.jsonl"
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    # Which validation metric decides ``best.pt``.  The default keeps the
    # historical behaviour.  A stage whose final field is frozen -- the 1/4 local
    # stage holds the residual head at zero, so relative EPE is constant there --
    # must select on the diagnostic it is actually training instead, otherwise
    # best.pt never leaves step 0.
    selection_metric = str(train_config.get("selection_metric", "relative_epe"))
    if not selection_metric:
        raise ValueError("vtmot_train.selection_metric must name a validation key")
    print(f"device={device} ({device_name}) run={args.run} steps={steps} train={len(train_dataset)} "
          f"eval={len(eval_dataset)} crop={crop_hw} batch={batch_size} workers={num_workers} "
          f"lr={learning_rate:g} moving_source={moving_source} freeze_coarse={freeze_coarse} "
          f"freeze_local_refinement={freeze_local_refinement} "
          f"freeze_fine_interaction={freeze_fine_interaction} "
          f"ir_adapter_only={ir_adapter_only} selection_metric={selection_metric} "
          f"large_translation={OmegaConf.to_container(large_translation, resolve=True) if large_translation is not None else None}")
    print(f"init_flow_mix={OmegaConf.to_container(init_flow_mix, resolve=True) if init_flow_mix is not None else None} "
          f"refine_proposal_weight={float(config.loss.weights.get('refine_proposal', 0.0))}")
    best_ratio = float("inf")
    if args.resume is not None:
        best_ratio = float(checkpoint.get("best_ratio", float("inf")))
        # Old checkpoints lack best_ratio.  Recover it from the existing log so
        # a worse first validation after resume cannot replace best.pt.
        history_path = args.resume.parent / "metrics.jsonl"
        history_key = f"val_{selection_metric}"
        if best_ratio == float("inf") and history_path.is_file():
            for line in history_path.read_text(encoding="utf-8").splitlines():
                try:
                    previous = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (int(previous.get("step", 0)) <= start_step
                        and history_key in previous):
                    best_ratio = min(best_ratio, float(previous[history_key]))
        if best_ratio == float("inf"):
            best_ratio = float(evaluate(model, eval_loader, device,
                                        moving_source=moving_source)[selection_metric])
        print(f"resume: step={start_step} prior best {selection_metric}={best_ratio:.4f}")
    elif args.init is not None:
        # A fine-tuning run must not discard a better warm-start checkpoint
        # merely because every subsequent validation becomes worse.
        initial_validation = evaluate(model, eval_loader, device,
                                      moving_source=moving_source)
        best_ratio = float(initial_validation[selection_metric])
        _save(output_dir / "best.pt", 0, model, optimizer, config, best_ratio,
              provenance)
        with metrics_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"step": 0, **{f"val_{name}": value
                                                for name, value in initial_validation.items()}}) + "\n")
        print(f"init eval epe={initial_validation['epe_px']:.3f} "
              f"coarse={initial_validation['coarse_epe_px']:.3f} "
              f"{selection_metric}={best_ratio:.4f}; saved step-0 best.pt")
    # num_workers=0 draws random crops on the main process. Restore its RNG
    # after model construction, which otherwise differs across architectures.
    torch.manual_seed(int(train_config.seed))
    iterator = iter(train_loader)
    model.train()
    for step in range(start_step + 1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        ir = registration_moving_image(batch, moving_source).to(device, non_blocking=True)
        vi = batch["vi"].to(device, non_blocking=True)
        target, valid = batch["gt_flow"].to(device, non_blocking=True), batch["valid_mask"].to(device, non_blocking=True)
        gt_h = batch["gt_h"].to(device, non_blocking=True)
        extra_translation = None
        if large_translation is not None:
            extra_translation = sample_large_translation(
                large_translation, seed=int(train_config.seed), step=step)
            if extra_translation is not None:
                ir, target, valid, gt_h = translate_moving_for_stress(
                    ir, target, valid, gt_h, extra_translation,
                    padding_mode=str(large_translation.get("padding_mode", "reflection")))
        # A known, synthetic starting point for the 1/4 loop on some fraction of
        # steps: only a start whose residual is known teaches the loop what to do
        # about it, and only a start on the truth teaches it to stay put.
        initial_flow, start_mode = sample_init_flow(init_flow_mix, target,
                                                   seed=int(train_config.seed), step=step)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(ir, vi, init_flow=initial_flow)
            losses = loss_fn(aligned_ir=(output.final_aligned_ir if output.final_aligned_ir is not None
                                         else output.coarse_aligned_ir), visible=vi,
                             coarse_flow=output.coarse_flow, final_flow=output.final_flow,
                             global_flow=output.global_flow,
                             gt_flow=target, valid_mask=valid,
                             predicted_affine_yx=output.affine_yx, gt_h=gt_h,
                             affine_feature_hw=tuple(output.confidence_1_8.shape[-2:]),
                             match=output.match, local_match=output.local_match,
                             coarse_local_match=output.coarse_local_match,
                             refinement=output.refinement)
        scaler.scale(losses.total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            train_epe = endpoint_error(output.final_flow if output.final_flow is not None
                                       else output.coarse_flow, target, valid)
        record = {"step": step, "train_loss": float(losses.total.detach()),
                  "train_epe": float(train_epe), "flow": float(losses.flow.detach()),
                  "match": float(losses.match.detach()),
                  "appearance": float(losses.appearance.detach()),
                  "local": float(losses.local.detach()),
                  "coarse_local": float(losses.coarse_local.detach()),
                  "global_flow": float(losses.global_flow.detach()),
                  "temperature": float(model.matcher.temperature.detach()),
                  "peak_mem_mib": (torch.cuda.max_memory_allocated() / 2 ** 20
                                   if device.type == "cuda" else 0.0),
                  "affine": float(losses.affine.detach()),
                  "refine": float(losses.refine.detach()) if losses.refine is not None else 0.0,
                  "refine_proposal": (float(losses.refine_proposal.detach())
                                      if losses.refine_proposal is not None else 0.0)}
        if init_flow_mix is not None:
            record["init_flow_mode"] = start_mode
            if initial_flow is not None:
                # How far the synthetic start sits from the truth, in pixels: the
                # number the per-round losses have to walk back.
                start_error = (initial_flow - target).abs().mean()
                record["init_flow_error_px"] = float(start_error)
        if large_translation is not None:
            record["extra_translation_dy_dx"] = list(extra_translation or (0, 0))
            record["augmented_valid_fraction"] = float(valid.float().mean())
        if step == 1 or step % int(train_config.log_every) == 0:
            print("step={step:5d} loss={train_loss:.5f} train_epe={train_epe:.3f} "
                  "match={match:.4f} appearance={appearance:.4f} "
                  "local={local:.4f} coarse_local={coarse_local:.4f} "
                  "global_flow={global_flow:.4f} affine={affine:.4f} "
                  "refine={refine:.4f} refine_proposal={refine_proposal:.4f}"
                  .format(**record))
            if large_translation is not None:
                print(f"  train shift={record['extra_translation_dy_dx']} "
                      f"valid={record['augmented_valid_fraction']:.3f}")
            if init_flow_mix is not None:
                print(f"  init_flow={record['init_flow_mode']} "
                      f"start_error_px={record.get('init_flow_error_px', 0.0):.2f}")
            # Localisation view of the same batch: does the correct key rank first?
            record.update(matching_diagnostics(output.match.matching_probability,
                                               tuple(output.match.coarse_flow.shape[-2:]),
                                               target, valid))
        if step % int(train_config.validation_every) == 0 or step == steps:
            validation = evaluate(model, eval_loader, device,
                                  moving_source=moving_source)
            record.update({f"val_{name}": value for name, value in validation.items()})
            print("  eval epe={epe_px:.3f} coarse={coarse_epe_px:.3f} "
                  "zero={zero_flow_epe_px:.3f} ratio={relative_epe:.3f} "
                  "gt_rank_frac={match_frac_keys_beating_gt:.3f} "
                  "argmax_epe={match_epe_argmax_px:.1f}px".format(**validation))
            if "global_epe_px" in validation:
                print(f"  global epe={validation['global_epe_px']:.3f}px "
                      f"candidates={validation['coarse_match_candidates']}")
            if "local_soft_epe_px" in validation:
                print("  local soft_epe={local_soft_epe_px:.3f}px "
                      "argmax_epe={local_argmax_epe_px:.1f}px "
                      "oracle_epe={local_oracle_epe_px:.3f}px".format(**validation))
            if selection_metric not in validation:
                raise ValueError(
                    f"selection_metric {selection_metric!r} is missing from the "
                    f"validation report; available keys: {sorted(validation)}")
            if validation[selection_metric] < best_ratio:
                best_ratio = validation[selection_metric]
                _save(output_dir / "best.pt", step, model, optimizer, config,
                      best_ratio, provenance)
            model.train()
        with metrics_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        if step % int(train_config.checkpoint_every) == 0 or step == steps:
            _save(output_dir / "last.pt", step, model, optimizer, config,
                  best_ratio, provenance)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
    print(f"completed: best {selection_metric}={best_ratio:.4f}; output={output_dir.resolve()}")


if __name__ == "__main__":
    main()
