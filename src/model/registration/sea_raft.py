"""SEA-RAFT checkpoint adapter."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _as_rgb

class SeaRAFT(nn.Module):
    """SEA-RAFT-S with the VF-Bench backward-flow adapter."""

    def __init__(self, config_path="config/module/spring-S.json",
                 weights_path=None, num_flow_updates=None, trainable=True,
                 max_flow=None, allow_missing_weights=False):
        super().__init__()
        from ..raft import RAFT
        from ..utils import load_args_from_json

        root = Path(__file__).resolve().parents[3]
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = root / config_file
        if not config_file.is_file():
            raise FileNotFoundError(f"SEA-RAFT config not found: {config_file}")

        args = load_args_from_json(str(config_file))
        self.net = RAFT(args)
        configured = weights_path or getattr(args, "path", None)
        checkpoint_file = None
        if configured:
            checkpoint_file = Path(configured)
            if not checkpoint_file.is_absolute():
                checkpoint_file = root / checkpoint_file
            if not checkpoint_file.is_file() and checkpoint_file.suffix == ".pth":
                alternative = checkpoint_file.with_suffix(".safetensors")
                if alternative.is_file():
                    checkpoint_file = alternative

        self.checkpoint_path = checkpoint_file
        self.pretrained = False
        if checkpoint_file is not None and checkpoint_file.is_file():
            if checkpoint_file.suffix == ".safetensors":
                try:
                    from safetensors.torch import load_file
                except ImportError as exc:
                    raise ImportError(
                        "Install safetensors to load the SEA-RAFT checkpoint."
                    ) from exc
                checkpoint = load_file(str(checkpoint_file), device="cpu")
            else:
                checkpoint = torch.load(str(checkpoint_file), map_location="cpu")
            if isinstance(checkpoint, dict):
                checkpoint = checkpoint.get(
                    "state_dict", checkpoint.get("model", checkpoint))
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Unsupported SEA-RAFT checkpoint: {checkpoint_file}")
            checkpoint = {
                key[7:] if key.startswith("module.") else key: value
                for key, value in checkpoint.items()
            }
            if not set(self.net.state_dict()).intersection(checkpoint):
                raise RuntimeError(
                    f"SEA-RAFT checkpoint does not match the model: {checkpoint_file}")
            self.net.load_state_dict(checkpoint, strict=False)
            self.pretrained = True
        elif not allow_missing_weights:
            raise FileNotFoundError(
                "SEA-RAFT weights are missing. Expected checkpoint at "
                f"{checkpoint_file}")

        updates = getattr(args, "iters", 4)
        self.num_flow_updates = max(
            1, int(updates if num_flow_updates is None else num_flow_updates))
        self.trainable = bool(trainable)
        self.max_flow = None if max_flow is None else float(max_flow)
        self.net.requires_grad_(self.trainable)
        if not self.trainable:
            self.net.eval()

    def train(self, mode=True):
        super().train(mode)
        if not self.trainable:
            self.net.eval()
        return self

    def forward(self, moving, fixed):
        """Return the ``[dy, dx]`` field that samples ``moving`` onto ``fixed``.

        VERIFIED CONVENTION (measured with synthetic translations, see the note
        at the bottom of this file): the output ``F`` satisfies
        ``warp(moving, F) == fixed``, i.e. ``moving(p + F(p)) == fixed(p)``.
        A translation of ``moving`` by ``+d`` yields ``F == -d`` to ~0.02 px.

        Note the argument order is deliberately confusing: ``self.net`` is called
        with ``(fixed, moving)``.  Do not "fix" it without re-running that test --
        the composition is correct as written even though the parameter names
        suggest the opposite.
        """
        moving = _as_rgb(moving.float()).clamp(0.0, 1.0)
        fixed = _as_rgb(fixed.float()).clamp(0.0, 1.0)
        height, width = moving.shape[-2:]
        padded_height = max(height, 128)
        padded_width = max(width, 128)
        padded_height += (8 - padded_height % 8) % 8
        padded_width += (8 - padded_width % 8) % 8
        pad_bottom = padded_height - height
        pad_right = padded_width - width
        if pad_bottom or pad_right:
            moving = F.pad(moving, (0, pad_right, 0, pad_bottom),
                           mode="replicate")
            fixed = F.pad(fixed, (0, pad_right, 0, pad_bottom),
                          mode="replicate")

        # The bundled SEA-RAFT implementation expects images in [0, 255] and
        # performs its own normalization. It returns a dictionary whose
        # ``flow`` entry contains the iterative predictions.
        predictions = self.net(
            fixed.mul(255.0),
            moving.mul(255.0),
            iters=self.num_flow_updates)
        if isinstance(predictions, dict):
            flow_predictions = predictions.get("flow")
            flow = (flow_predictions[-1] if flow_predictions else
                    predictions["final"])
        else:
            flow = predictions[-1]
        flow = flow[..., :height, :width][:, [1, 0]]
        if self.max_flow is not None:
            flow = flow.clamp(-self.max_flow, self.max_flow)
        return flow


# ---------------------------------------------------------------------------
# Verified 2026-09-21 on CPU by translating a real VTMOT frame by a known `d`
# and calling SeaRAFT(moving, base): the returned field was -d to ~0.02 px
# (d=(3,0) -> (-3.01, +0.01); d=(4,3) -> (-4.01, -2.99); d=(-6,2) -> (+5.98, -1.97)).
# So the field samples `moving` onto `fixed`, which is what
# `SpatialTransformer(moving, flow)` needs.
#
# The same measurement exposed a much bigger problem, so read this before
# trusting the coarse field: on a same-modality pair (visible_gt -> visible_mis)
# SEA-RAFT recovers the injected affine to 0.045 px EPE, but on the CROSS-MODAL
# pair (infrared -> visible_mis) its zero-shot EPE is 63 px against an 11.6 px
# zero-flow baseline -- a ratio of 5.4, i.e. five times worse than predicting no
# motion.  The error is systematic (per-pixel p50 = 52 px, |F| up to 148 px), not
# outlier noise, and clamping to +-32 only improves it to 40 px (ratio 3.5).
# Since the learnable residual heads are bounded (+-8 px lite, +-3 px DCN) they
# cannot correct a 50-150 px coarse error, which is why the supervised runs
# plateau instead of converging.
# ---------------------------------------------------------------------------
