"""Use the separately installed official CRFT on the VTMOT evaluation protocol.

CRFT's image0 is the fixed visible image, image1 is the moving infrared
image, and its flow channels are [dx, dy].  VTMOT stores [dy, dx] at the
480x640 evaluation resolution.  Keep those conversions in one place.
"""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_MODEL_HW = (96, 96)


def validate_model_hw(model_hw: tuple[int, int]) -> tuple[int, int]:
    height, width = map(int, model_hw)
    if height < 16 or width < 16 or height % 8 or width % 8:
        raise ValueError("CRFT input height and width must be multiples of 8 and at least 16")
    if height != width:
        raise ValueError("this CRFT release requires a square input for fine position encoding")
    return height, width


def _lower_config(value):
    """Convert a YACS subtree to the lowercase dict accepted by CRFT."""
    if hasattr(value, "items"):
        return {str(key).lower(): _lower_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_lower_config(item) for item in value)
    return value


def build_crft(crft_root: str | Path, device: torch.device) -> torch.nn.Module:
    """Load upstream code without copying or modifying its source tree."""
    root = Path(crft_root).resolve()
    config_file = root / "configs" / "crft" / "outdoor" / "visible_thermal.py"
    model_file = root / "src" / "crft" / "crft.py"
    if not config_file.is_file() or not model_file.is_file():
        raise FileNotFoundError(f"official CRFT source tree not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    upstream_config = importlib.import_module("src.config.default")
    if not Path(upstream_config.__file__).resolve().is_relative_to(root):
        raise RuntimeError("a different 'src' package is already imported; run in a fresh process")
    # The official .py config mutates the default YACS tree.  It sets the
    # visible/thermal matching options used by the RoadScene checkpoint.
    runpy.run_path(str(config_file))
    config = upstream_config.get_cfg_defaults(inference=True)
    model_class = importlib.import_module("src.crft.crft").CRFT
    return model_class(_lower_config(config.CRFT)).to(device)


def load_crft_weights(model: torch.nn.Module, checkpoint_path: str | Path
                      ) -> tuple[int, int] | None:
    """Accept upstream Lightning or a checkpoint written by our train entrypoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("CRFT checkpoint has no tensor state dictionary")
    official = {name.removeprefix("matcher."): tensor
                for name, tensor in state.items() if name.startswith("matcher.")}
    if official:
        state = official
    model.load_state_dict(state, strict=True)
    saved_hw = checkpoint.get("model_hw")
    return validate_model_hw(tuple(saved_hw)) if saved_hw is not None else None


def flow_xy_to_yx_at(flow_xy: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Upsample CRFT [dx,dy] flow and express displacement in target pixels."""
    if flow_xy.ndim != 4 or flow_xy.shape[1] != 2:
        raise ValueError("CRFT flow must be [B,2,H,W]")
    height, width = target_hw
    source_height, source_width = flow_xy.shape[-2:]
    resized = F.interpolate(flow_xy.float(), size=(height, width),
                            mode="bilinear", align_corners=True)
    scale = resized.new_tensor((height / source_height,
                                width / source_width)).view(1, 2, 1, 1)
    return resized[:, [1, 0]] * scale


def letterbox_geometry(image_hw: tuple[int, int], model_hw: tuple[int, int]
                       ) -> tuple[int, int, int, int]:
    """Return resized H,W and top,left for a centered aspect-preserving pad."""
    height, width = image_hw
    model_height, model_width = validate_model_hw(model_hw)
    scale = min(model_height / height, model_width / width)
    resized_height = min(model_height, max(1, round(height * scale)))
    resized_width = min(model_width, max(1, round(width * scale)))
    return (resized_height, resized_width,
            (model_height - resized_height) // 2,
            (model_width - resized_width) // 2)


def _letterbox(image: torch.Tensor, model_hw: tuple[int, int],
               geometry: tuple[int, int, int, int]) -> torch.Tensor:
    resized_height, resized_width, top, left = geometry
    resized = F.interpolate(image, size=(resized_height, resized_width),
                            mode="bilinear", align_corners=False)
    return F.pad(resized, (left, model_hw[1] - resized_width - left,
                           top, model_hw[0] - resized_height - top))


def flow_from_letterbox(flow_xy: torch.Tensor, image_hw: tuple[int, int],
                        model_hw: tuple[int, int]) -> torch.Tensor:
    """Map a square CRFT [dx,dy] field into full-resolution VTMOT [dy,dx]."""
    resized_height, resized_width, top, left = letterbox_geometry(image_hw, model_hw)
    if flow_xy.ndim != 4 or flow_xy.shape[1] != 2:
        raise ValueError("CRFT flow must be [B,2,H,W]")
    if tuple(flow_xy.shape[-2:]) == model_hw:
        model_flow = flow_xy
    else:
        model_flow = F.interpolate(flow_xy, size=model_hw,
                                   mode="bilinear", align_corners=True)
        model_flow = model_flow * model_flow.new_tensor(
            (model_hw[1] / flow_xy.shape[-1],
             model_hw[0] / flow_xy.shape[-2])).view(1, 2, 1, 1)
    region = model_flow[:, :, top:top + resized_height, left:left + resized_width]
    full = F.interpolate(region.float(), size=image_hw,
                         mode="bilinear", align_corners=False)
    return full[:, [1, 0]] * full.new_tensor(
        (image_hw[0] / resized_height,
         image_hw[1] / resized_width)).view(1, 2, 1, 1)


def run_crft(model: torch.nn.Module, ir: torch.Tensor, vi: torch.Tensor,
             model_hw: tuple[int, int] = DEFAULT_MODEL_HW) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final and coarse VTMOT-style [dy,dx] fields at image resolution."""
    model_hw = validate_model_hw(model_hw)
    if ir.ndim != 4 or ir.shape[1] != 1 or vi.ndim != 4 or vi.shape[1] != 3:
        raise ValueError("expected IR [B,1,H,W] and visible [B,3,H,W]")
    if ir.shape[0] != vi.shape[0] or ir.shape[-2:] != vi.shape[-2:]:
        raise ValueError("IR and visible must share batch and spatial dimensions")
    # Upstream RoadScene reads both images as RGB.  Replicating the infrared
    # channel keeps its pretrained three-channel stem on the same input path.
    target_hw = tuple(ir.shape[-2:])
    geometry = letterbox_geometry(target_hw, model_hw)
    resized_height, resized_width, top, left = geometry
    full_mask = ir.new_zeros((1, 1, *model_hw))
    full_mask[:, :, top:top + resized_height, left:left + resized_width] = 1
    coarse_mask = F.interpolate(full_mask, size=(model_hw[0] // 8, model_hw[1] // 8),
                                mode="nearest")[:, 0].bool().expand(ir.shape[0], -1, -1)
    inputs = {
        "image0": _letterbox(vi, model_hw, geometry),
        "image1": _letterbox(ir.repeat(1, 3, 1, 1), model_hw, geometry),
        "mask0": coarse_mask,
        "mask1": coarse_mask,
    }
    model(inputs)  # upstream CRFT writes flow_f_full and flow_c into the dict
    if "flow_f_full" not in inputs or "flow_c" not in inputs:
        raise RuntimeError("upstream CRFT did not produce flow_f_full and flow_c")
    return (flow_from_letterbox(inputs["flow_f_full"], target_hw, model_hw),
            flow_from_letterbox(inputs["flow_c"], target_hw, model_hw))


def masked_flow_loss(flow_yx: torch.Tensor, target_yx: torch.Tensor,
                     valid: torch.Tensor) -> torch.Tensor:
    """Supervise a field in full-image pixel units with the VTMOT mask."""
    model_height, model_width = flow_yx.shape[-2:]
    target_small = F.interpolate(target_yx, size=(model_height, model_width),
                                 mode="bilinear", align_corners=True)
    valid_small = F.interpolate(valid, size=(model_height, model_width), mode="nearest")
    diff = torch.linalg.vector_norm(flow_yx - target_small, dim=1, keepdim=True)
    return (diff * valid_small).sum() / valid_small.sum().clamp_min(1)


@torch.no_grad()
def evaluate_crft(model: torch.nn.Module, loader, device: torch.device,
                  model_hw: tuple[int, int] = DEFAULT_MODEL_HW) -> dict[str, object]:
    """Compute the same valid-pixel EPE and PCK as evaluate_vtmot."""
    from .metrics import endpoint_error

    model.eval()
    total_valid = total_epe = total_coarse = total_zero = 0.0
    hits = {threshold: 0.0 for threshold in (1, 3, 5)}
    samples = 0
    for batch in loader:
        ir, vi = batch["ir"].to(device), batch["vi"].to(device)
        target, valid = batch["gt_flow"].to(device), batch["valid_mask"].to(device)
        predicted, coarse = run_crft(model, ir, vi, model_hw)
        count = float(valid.sum())
        total_valid += count
        samples += ir.shape[0]
        total_epe += float(endpoint_error(predicted, target, valid)) * count
        total_coarse += float(endpoint_error(coarse, target, valid)) * count
        total_zero += float(endpoint_error(torch.zeros_like(target), target, valid)) * count
        error = torch.linalg.vector_norm(predicted - target, dim=1, keepdim=True)
        for threshold in hits:
            hits[threshold] += float(((error <= threshold) * valid).sum())
    if total_valid <= 0:
        raise ValueError("VTMOT evaluation has no valid pixels")
    report = {
        "epe_px": total_epe / total_valid,
        "coarse_epe_px": total_coarse / total_valid,
        "zero_flow_epe_px": total_zero / total_valid,
        "valid_pixels": total_valid,
        "samples": samples,
        "model_input_hw": list(model_hw),
    }
    report["relative_epe"] = report["epe_px"] / report["zero_flow_epe_px"]
    report.update({f"pck_{threshold}px": count / total_valid
                   for threshold, count in hits.items()})
    report["beats_zero_flow"] = report["relative_epe"] < 1.0
    report["fusion_ready"] = report["epe_px"] <= 2.0 and report["pck_3px"] >= 0.90
    return report
