# Last modified: 2025-10-19

import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from typing import List, Union

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.model.net import IRVisibleFusion
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger
from src.util.lr_scheduler import IterExponential
from src.util.loss import get_loss
from src.util.metric import MetricTracker
from src.util.seeding import generate_seed_sequence


class IRVisibleTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: IRVisibleFusion,
        train_dataloader: DataLoader,
        accelerator: Accelerator,
        out_dir_ckpt,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: IRVisibleFusion = model

        self.accelerator = accelerator
        self.device = accelerator.device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders or []
        self.accumulation_steps: int = accumulation_steps

        # RTX 3090/PyTorch 2.x can use the fused CUDA Adam kernel.
        lr = self.cfg.lr
        optimizer_kwargs = {"lr": lr}
        if self.cfg.trainer.get("fused_optimizer", False):
            optimizer_kwargs["fused"] = True

        # SEA-RAFT gets its own learning rate.  Its weights are pretrained for
        # MONO-modal optical flow and have to be adapted to the IR/VIS domain,
        # while the heads that were initialised from scratch (lite refinement,
        # DCN) are the ones a 1e-4 is actually sized for.  Sharing one lr made
        # the backbone effectively frozen: the measured initial held-out
        # epe_ratio was ~0.89 and training only degraded it.
        #   model.registration.raft.lr_scale: 1.0 reproduces the old behaviour.
        self.raft_lr_scale = float(self.cfg.model.get("registration", {}).get(
            "raft", {}).get("lr_scale", 1.0))
        self.raft_param_names: List[str] = []
        param_groups = self._build_param_groups(lr)

        try:
            self.optimizer = Adam(param_groups, **optimizer_kwargs)
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            self.optimizer = Adam(param_groups, **optimizer_kwargs)
        self.use_fused_optimizer = bool(
            getattr(self.optimizer, "defaults", {}).get("fused", False))
        if self.raft_lr_scale != 1.0 and len(self.optimizer.param_groups) > 1:
            logging.info(
                "Optimiser: %d SEA-RAFT parameter tensors at lr %.2e (scale %.2f), "
                "everything else at lr %.2e",
                len(self.raft_param_names), lr * self.raft_lr_scale,
                self.raft_lr_scale, lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        self.loss_keys = [
            "loss",
            "loss_fusion",
            "loss_registration",
            "loss_int",
            "loss_grad",
            "loss_ssim",
            "loss_temp",
            "loss_reg",
            "loss_flow_smooth",
            "loss_reg_temp",
            "loss_propagation",
            "loss_fb",
            "loss_memory",
            "loss_flow_gt",
            # Normalised EPE: gt-flow end-point error over the sample's own
            # zero-flow baseline. < 1 means the registration is better than
            # predicting no motion. This is the metric that says whether the
            # supervised registration is actually learning.
            "flow_epe_ratio",
            # Diagnostics only, never part of the objective.
            "local_flow_mag",
            "alignment_score_mean",
            "keyframe_ratio",
            "adaptive_interval_mean",
            "confidence_mean",
            "fb_error_mean",
        ]
        self.train_metrics = MetricTracker(*self.loss_keys)

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.validation_period = int(self.cfg.trainer.get("validation_period", 0))
        self.max_grad_norm = float(self.cfg.trainer.get("max_grad_norm", 0.0))
        stage_cfg = self.cfg.trainer.get("stages", {})
        self.registration_steps = int(stage_cfg.get("registration_steps", 0))
        self.fusion_steps = int(stage_cfg.get("fusion_steps", 0))

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming
        self.best_val_loss = float("inf")

        # Model selection.  "Best" must be chosen with the metric the run is
        # actually about: the total validation loss mixes fusion and
        # registration, so in the joint stage it can pick a checkpoint whose
        # registration is worse.  Prefer the normalised registration EPE
        # (flow_epe_ratio, < 1 means better than predicting no motion) whenever a
        # dense GT flow exists.
        # IMPORTANT: without a GT (HDO), `flow_epe_ratio` is identically 0 for
        # every batch, so `0 < best` would be false forever and NO checkpoint
        # would ever be saved.  Hence the explicit fallback.
        self.val_has_registration_gt = any(
            getattr(getattr(loader, "dataset", None), "registration_gt_dir", None)
            is not None for loader in self.val_loaders)
        self.selection_metric_name = (
            "flow_epe_ratio" if self.val_has_registration_gt
            else "loss_registration")
        self.best_selection_metric = float("inf")
        logging.info(
            "Model selection metric: %s (val GT flow present: %s)",
            self.selection_metric_name, self.val_has_registration_gt)

        # Hardware guard.  Defaults are conservative for a laptop (this machine
        # has a history of hard black-screen shutdowns under sustained load) and
        # harmless on a desktop, which will never reach 84 C.
        #   trainer.gpu_temp_check_every: 0 disables the whole guard
        #   trainer.step_sleep: idle seconds after each optimiser step, the
        #     cheapest way to keep a laptop out of its power limit
        self.gpu_temp_check_every = int(
            self.cfg.trainer.get("gpu_temp_check_every", 50))
        self.gpu_temp_limit = float(self.cfg.trainer.get("gpu_temp_limit", 84.0))
        self.gpu_temp_resume = float(self.cfg.trainer.get("gpu_temp_resume", 76.0))
        self.gpu_temp_abort = float(self.cfg.trainer.get("gpu_temp_abort", 90.0))
        self.step_sleep = float(self.cfg.trainer.get("step_sleep", 0.0))
        self.peak_gpu_temp = 0.0
        self.current_stage = None

    def _training_stage(self):
        if self.effective_iter < self.registration_steps:
            return "registration"
        if self.effective_iter < self.registration_steps + self.fusion_steps:
            return "fusion"
        return "joint"

    def _set_training_stage(self, stage):
        model = self.accelerator.unwrap_model(self.model)
        model.set_training_stage(stage)
        if stage != self.current_stage:
            self.current_stage = stage
            if self.accelerator.is_main_process:
                logging.info("Training stage changed to: %s", stage)

    def _build_param_groups(self, lr):
        """Split parameters into a base group and a scaled SEA-RAFT group.

        With ``raft.lr_scale == 1`` (the default) or no trainable SEA-RAFT this
        returns a single group containing every parameter, which is exactly the
        previous behaviour.  ``self.raft_param_names`` is filled in for logging.
        """
        registration = getattr(self.model, "registration", None)
        raft = getattr(registration, "keyframe_raft", None)
        raft_ids = set()
        self.raft_param_names = []
        if (raft is not None and self.raft_lr_scale != 1.0
                and getattr(raft, "trainable", False)):
            for name, param in raft.named_parameters():
                if param.requires_grad:
                    raft_ids.add(id(param))
                    self.raft_param_names.append(
                        f"registration.keyframe_raft.{name}")
        base, scaled = [], []
        for param in self.model.parameters():
            (scaled if id(param) in raft_ids else base).append(param)
        groups = [{"params": base, "lr": lr}]
        if scaled:
            groups.append({"params": scaled, "lr": lr * self.raft_lr_scale})
        return groups

    def _thermal_gate(self):
        """Pause (or stop) when the GPU gets too hot.

        A sustained 100% GPU+CPU load is what trips laptop power-delivery or
        thermal protection, and the failure mode is a hard black screen rather
        than an exception -- this machine has unexpected-shutdown events in its
        System log from before this project started.  Cheap insurance: one
        nvidia-smi poll every `trainer.gpu_temp_check_every` steps.

        Returns True when training should stop.
        """
        if not self.gpu_temp_check_every or not self.gpu_temp_limit:
            return False
        if self.effective_iter % self.gpu_temp_check_every:
            return False
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            temp = float(out.stdout.strip().splitlines()[0])
        except Exception:
            return False
        self.peak_gpu_temp = max(self.peak_gpu_temp, temp)
        if temp >= self.gpu_temp_abort:
            logging.error(
                "GPU at %.0f C >= abort threshold %.0f C -- stopping training.",
                temp, self.gpu_temp_abort)
            return True
        if temp >= self.gpu_temp_limit:
            logging.warning(
                "GPU at %.0f C >= %.0f C -- pausing until it drops below %.0f C.",
                temp, self.gpu_temp_limit, self.gpu_temp_resume)
            while temp >= self.gpu_temp_resume:
                time.sleep(10)
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=temperature.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5)
                    temp = float(out.stdout.strip().splitlines()[0])
                except Exception:
                    break
            logging.info("Resuming at %.0f C", temp)
        return False

    def _normalize_grad_layout(self):
        """Give every gradient the exact memory layout of its parameter.

        The fused CUDA Adam kernel compares parameter and gradient strides
        element-wise, and rejects equivalent-but-differently-strided layouts.
        cuDNN returns exactly such a gradient for the VSS blocks, whose
        convolutions consume tensors that went through ``permute().reshape()``:
        on this build 30 weights are affected, e.g.
        ``ssm_blocks_1.*.in_proj.weight`` has parameter stride (48, 1, 1, 1)
        but gradient stride (48, 1, 48, 48), and ``ssm_blocks_1.*.dwconv.weight``
        has (9, 9, 3, 1) against (9, 1, 3, 1).  Both report
        ``is_contiguous() == True``, so calling ``.contiguous()`` does NOT fix
        it, and the failure message ("same dtype, device, and layout") is
        misleading because dtype/device/layout really are identical.

        ``torch.empty_like(param).copy_(grad)`` keeps the values bit for bit and
        restores the parameter's stride pattern, so the fused kernel becomes
        usable with identical numerics (verified in
        ``tools/diag_fused_adam.py``).

        Returns the number of gradients that needed normalising.
        """
        fixed = 0
        for param in self.model.parameters():
            grad = param.grad
            if grad is None or grad.stride() == param.stride():
                continue
            param.grad = torch.empty_like(param).copy_(grad)
            fixed += 1
        return fixed

    def _prepare_accelerator(self):
        # Keep master weights in FP32. Accelerator autocast/GradScaler handles
        # FP16 safely; casting BatchNorm and optimizer weights to FP16 here is unstable.
        self.model = self.model.to(self.accelerator.device)
        self.loss = self.loss.to(self.accelerator.device)

        logging.debug(
            f"{self.accelerator.device}: GPU memory allocated before accelerator.prepare(): {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )
        logging.debug(
            f"{self.accelerator.device}: GPU memory reserved before accelerator.prepare(): {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )

        # Call prepare()
        (
            self.model,
            self.optimizer,
            self.train_loader,
        ) = self.accelerator.prepare(self.model, self.optimizer, self.train_loader)
        self.val_loaders = [
            self.accelerator.prepare(loader) for loader in self.val_loaders
        ]
        logging.debug(
            f"{self.accelerator.device}: GPU memory allocated after accelerator.prepare(): {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )
        logging.debug(
            f"{self.accelerator.device}: GPU memory reserved after accelerator.prepare(): {torch.cuda.memory_reserved() / 1e9:.2f} GB"
        )

    def train(self, t_end=None):
        if self.accelerator.is_main_process:
            logging.info("Start training")

        device = self.device
        self._prepare_accelerator()
        self.train_metrics.reset()
        accumulated_step = 0
        iter_start_time = time.time()

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            if self.accelerator.is_main_process:
                logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.train()
                stage = self._training_stage()
                self._set_training_stage(stage)

                # Training step
                ir_img = batch["ir"].to(self.device, non_blocking=True)
                rgb_img = batch["rgb"].to(self.device, non_blocking=True)
                gt_flow = batch.get("gt_flow")
                if gt_flow is not None:
                    gt_flow = gt_flow.to(self.device, non_blocking=True)

                # Predict the noise residual
                fusion_pred, registration = self.model(
                    ir_img, rgb_img, stage=stage)
                if fusion_pred is not None and torch.isnan(fusion_pred).any():
                    logging.warning(
                        f"device: {self.accelerator.device} model_pred contains NaN."
                    )

                # Loss
                loss_output = self.loss(
                    fusion_pred,  # B, 3, C, H, W
                    ir_img,  # B, 5, C, H, W
                    rgb_img,  # B, 5, C, H, W
                    registration,
                    compute_fusion=stage != "registration",
                    compute_registration=stage != "fusion",
                    gt_flow=gt_flow,
                )

                for key in self.loss_keys:
                    self.train_metrics.update(key, loss_output[key].item())
                loss = loss_output["loss"] / self.gradient_accumulation_steps

                self.accelerator.backward(loss)
                accumulated_step += 1

                self.n_batch_in_epoch += 1

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    if self.use_fused_optimizer:
                        self._normalize_grad_layout()
                    if self.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated_step = 0
                    if self.step_sleep > 0:
                        time.sleep(self.step_sleep)
                    if self._thermal_gate():
                        self.save_checkpoint(ckpt_name="latest",
                                             save_train_state=True)
                        logging.error(
                            "Stopped by the GPU temperature guard at iter %d; "
                            "peak temperature %.0f C.", self.effective_iter,
                            self.peak_gpu_temp)
                        return

                    self.effective_iter += 1

                    # Calculate ETA
                    elapsed = time.time() - iter_start_time
                    avg_time_per_iter = elapsed / self.effective_iter
                    if self.max_iter > 0:
                        remaining_iter = self.max_iter - self.effective_iter
                        eta_seconds = int(avg_time_per_iter * remaining_iter)
                        eta_str = str(timedelta(seconds=eta_seconds))
                    else:
                        eta_str = "N/A"

                    train_loss_results = self.train_metrics.result()
                    for k, v in train_loss_results.items():
                        v_tensor = torch.tensor(v, device=device)
                        gathered = self.accelerator.gather(v_tensor)
                        gathered = torch.mean(gathered).item()

                        if self.accelerator.is_main_process:
                            tb_logger.writer.add_scalar(
                                f"train/{k}", gathered, global_step=self.effective_iter
                            )
                            train_loss_results[k] = gathered

                    if self.accelerator.is_main_process:
                        lr = self.lr_scheduler.get_last_lr()[0]
                        tb_logger.writer.add_scalar(
                            "lr", lr, global_step=self.effective_iter
                        )
                        tb_logger.writer.add_scalar(
                            "n_batch_in_epoch",
                            self.n_batch_in_epoch,
                            global_step=self.effective_iter,
                        )

                        # loss logging
                        loss_str = ", ".join(
                            f"{k}={v:.5f}" for k, v in train_loss_results.items()
                        )
                        logging.info(
                            f"iter {self.effective_iter:5d} (epoch {epoch:2d}, stage={stage}): {loss_str}, lr={lr:.3e}, ETA={eta_str}"
                        )

                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.save_checkpoint(
                                ckpt_name=self._get_backup_ckpt_name(),
                                save_train_state=False,
                            )
                            logging.info("Training ended.")

                        self.accelerator.wait_for_everyone()
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.save_checkpoint(
                                ckpt_name="latest", save_train_state=True
                            )
                            logging.info("Time is up, training paused.")

                        self.accelerator.wait_for_everyone()
                        return

            # Epoch end
            self.n_batch_in_epoch = 0

    def _train_step_callback(self):
        """Executed after every iteration"""
        self.accelerator.wait_for_everyone()

        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            if self.accelerator.is_main_process:
                self.save_checkpoint(
                    ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                )

        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and self.accelerator.is_main_process
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        if (
            self.validation_period > 0
            and self.val_loaders
            and self.effective_iter % self.validation_period == 0
        ):
            val_results = self.validate()
            val_loss = val_results["loss"]
            # Lower is better for both candidates: flow_epe_ratio is a ratio to
            # the zero-flow baseline, loss_registration is a positive penalty.
            metric = float(val_results.get(self.selection_metric_name, val_loss))
            if self.accelerator.is_main_process and metric < self.best_selection_metric:
                self.best_selection_metric = metric
                # kept in sync for checkpoint/resume compatibility
                self.best_val_loss = val_loss
                self.save_checkpoint(ckpt_name="best", save_train_state=False)
                logging.info("New best validation %s: %.6f (loss %.6f)",
                             self.selection_metric_name, metric, val_loss)

        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def validate(self):
        """Run a small, scene-disjoint loss validation pass."""
        self.model.eval()
        local_sums = torch.zeros(len(self.loss_keys), device=self.device)
        local_count = torch.zeros(1, device=self.device)

        for loader in self.val_loaders:
            for batch in loader:
                ir_img = batch["ir"].to(self.device, non_blocking=True)
                rgb_img = batch["rgb"].to(self.device, non_blocking=True)
                gt_flow = batch.get("gt_flow")
                if gt_flow is not None:
                    gt_flow = gt_flow.to(self.device, non_blocking=True)
                fusion_pred, registration = self.model(
                    ir_img, rgb_img, stage="joint")
                loss_output = self.loss(
                    fusion_pred, ir_img, rgb_img, registration,
                    compute_fusion=True,
                    compute_registration=True,
                    gt_flow=gt_flow,
                )
                batch_size = ir_img.shape[0]
                local_sums += torch.stack([
                    loss_output[key].detach().float() for key in self.loss_keys
                ]) * batch_size
                local_count += batch_size

        packed = torch.cat((local_sums, local_count))
        gathered = self.accelerator.gather(packed)
        gathered = gathered.reshape(-1, len(self.loss_keys) + 1)
        totals = gathered[:, :-1].sum(dim=0)
        count = gathered[:, -1].sum().clamp_min(1.0)
        values = (totals / count).cpu().tolist()
        results = dict(zip(self.loss_keys, values))

        if self.accelerator.is_main_process:
            for key, value in results.items():
                tb_logger.writer.add_scalar(
                    f"val/{key}", value, global_step=self.effective_iter
                )
            logging.info(
                "validation iter %d: %s",
                self.effective_iter,
                ", ".join(f"{key}={value:.5f}"
                          for key, value in results.items()),
            )

        self.model.train()
        return results

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            if self.accelerator.is_main_process:
                logging.info(
                    f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
                )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        os.makedirs(ckpt_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(ckpt_dir, "model.pth")
        model_unwrap = self.accelerator.unwrap_model(self.model)
        torch.save(model_unwrap.state_dict(), model_path)
        logging.info(f"Model is saved to: {model_path}")

        if save_train_state:
            optimizer = self.optimizer
            state = {
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "global_seed_sequence": self.global_seed_sequence,
                "best_val_loss": self.best_val_loss,
                "best_selection_metric": self.best_selection_metric,
                "selection_metric_name": self.selection_metric_name,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            with open(os.path.join(ckpt_dir, "iter.txt"), "w+") as f:
                f.write(self._get_backup_ckpt_name())

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")

        # Load model
        model_path = os.path.join(ckpt_path, "model.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logging.info(f"Model parameters are loaded from {model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(
                os.path.join(ckpt_path, "trainer.ckpt"), weights_only=False
            )
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.best_selection_metric = checkpoint.get(
                "best_selection_metric", float("inf"))

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
