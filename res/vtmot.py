"""Read-only single-frame VTMOT adapter for the standalone registration branch.

Unlike the legacy nine-frame fusion loader, this module reads exactly one
``infrared`` / ``visible_mis`` pair and materialises the stored ``gt_h`` matrix
as dense backward flow.  It never writes to VTMOT and does not reuse the
fusion dataset classes, so geometry validation is independent of that stack.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def aspect_resize_affine(source_hw: Tuple[int, int], target_hw: Tuple[int, int]) -> np.ndarray:
    """Original ``[x,y,1]`` pixels -> centred-crop/resized pixels.

    The half-pixel term matches PIL bilinear resize and the existing VTMOT
    loader, so ``A @ gt_h @ inv(A)`` remains a geometrically valid target.
    """
    source_h, source_w = map(int, source_hw)
    target_h, target_w = map(int, target_hw)
    if min(source_h, source_w, target_h, target_w) < 1:
        raise ValueError("image dimensions must be positive")
    target_aspect, source_aspect = target_w / target_h, source_w / source_h
    if source_aspect > target_aspect:
        crop_h, crop_w, top, left = source_h, round(source_h * target_aspect), 0, 0
        left = (source_w - crop_w) // 2
    elif source_aspect < target_aspect:
        crop_h, crop_w, top, left = round(source_w / target_aspect), source_w, 0, 0
        top = (source_h - crop_h) // 2
    else:
        crop_h, crop_w, top, left = source_h, source_w, 0, 0
    sx, sy = target_w / crop_w, target_h / crop_h
    crop = np.array([[1.0, 0.0, -left], [0.0, 1.0, -top], [0.0, 0.0, 1.0]])
    resize = np.array([[sx, 0.0, 0.5 * sx - 0.5],
                       [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]])
    return resize @ crop


def homography_to_flow(homography: np.ndarray, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialise fixed->moving ``H`` as `[dy,dx]` and an in-bounds mask."""
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"homography must be [3,3], got {matrix.shape}")
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float64),
                         np.arange(width, dtype=np.float64), indexing="ij")
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=0).reshape(3, -1)
    mapped = matrix @ pixels
    mapped_x = (mapped[0] / mapped[2]).reshape(height, width)
    mapped_y = (mapped[1] / mapped[2]).reshape(height, width)
    flow = np.stack((mapped_y - yy, mapped_x - xx), axis=0).astype(np.float32)
    valid = ((mapped_x >= 0) & (mapped_x <= width - 1)
             & (mapped_y >= 0) & (mapped_y <= height - 1)).astype(np.float32)
    return torch.from_numpy(flow), torch.from_numpy(valid[None])


def _resample(image: Image.Image, target_hw: Tuple[int, int]) -> Image.Image:
    target_h, target_w = target_hw
    source_w, source_h = image.size
    target_aspect, source_aspect = target_w / target_h, source_w / source_h
    if source_aspect > target_aspect:
        crop_w = round(source_h * target_aspect)
        left = (source_w - crop_w) // 2
        image = image.crop((left, 0, left + crop_w, source_h))
    elif source_aspect < target_aspect:
        crop_h = round(source_w / target_aspect)
        top = (source_h - crop_h) // 2
        image = image.crop((0, top, source_w, top + crop_h))
    return image.resize((target_w, target_h), Image.Resampling.BILINEAR)


def _image_tensor(path: Path, mode: str, target_hw: Tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as opened:
        image = _resample(opened.convert(mode), target_hw)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        return torch.from_numpy(array[None])
    return torch.from_numpy(np.transpose(array, (2, 0, 1)).copy())


class VTMOTSingleFrameDataset(Dataset):
    """Single-frame ``ir + visible_mis + gt_flow`` VTMOT split.

    Args:
        root: ``VTMOT_misaligned`` directory itself.
        split: ``train``, ``eval`` or held-out ``test`` from ``split.json``.
        split_file: repository split file; paired frames come from its per-sequence CSVs.
        target_hw: normalised spatial resolution, divisible by eight by default.
        frame_stride: retain every Nth frame in each sequence for fast checks.
        include_rgb_gt: load aligned visible RGB only for GT-direction checking.
    """

    def __init__(self, root: str | Path, *, split: str,
                 split_file: str | Path = "data_split/IVF/VTMOT/split.json",
                 target_hw: Tuple[int, int] = (480, 640), frame_stride: int = 1,
                 crop_hw: Tuple[int, int] | None = None, random_crop: bool = False,
                 include_rgb_gt: bool = False, max_samples: int = 0) -> None:
        self.root = Path(root)
        self.split_file = Path(split_file)
        self.target_hw = tuple(map(int, target_hw))
        self.crop_hw = tuple(map(int, crop_hw)) if crop_hw is not None else None
        self.random_crop = bool(random_crop)
        self.include_rgb_gt = bool(include_rgb_gt)
        if not self.root.is_dir():
            raise FileNotFoundError(f"VTMOT root not found: {self.root}")
        if not self.split_file.is_file():
            raise FileNotFoundError(f"VTMOT split file not found: {self.split_file}")
        if self.target_hw[0] % 8 or self.target_hw[1] % 8:
            raise ValueError("target_hw must be divisible by 8 for MINDGlobalRegistration")
        if self.crop_hw is not None:
            if self.crop_hw[0] % 8 or self.crop_hw[1] % 8:
                raise ValueError("crop_hw must be divisible by 8 for MINDGlobalRegistration")
            if (self.crop_hw[0] > self.target_hw[0] or self.crop_hw[1] > self.target_hw[1]
                    or min(self.crop_hw) < 16):
                raise ValueError("crop_hw must be at least 16 and no larger than target_hw")
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        splits = json.loads(self.split_file.read_text(encoding="utf-8"))
        if split not in splits:
            raise ValueError(f"unknown VTMOT split {split!r}; available: {sorted(splits)}")
        self.split = split
        self.samples: List[Tuple[str, str]] = []
        for sequence in splits[split]:
            infrared_dir, visible_dir = self.root / sequence / "infrared", self.root / sequence / "visible_mis"
            h_dir = self.root / sequence / "gt_h"
            if not infrared_dir.is_dir() or not visible_dir.is_dir() or not h_dir.is_dir():
                raise FileNotFoundError(f"incomplete VTMOT sequence: {sequence}")
            # The dataset directory may also contain raw frames with hashed
            # suffixes. Only the curated CSV rows have matching visible/GT files.
            manifest = self.split_file.parent / f"{sequence}.csv"
            if not manifest.is_file():
                raise FileNotFoundError(f"VTMOT sequence manifest not found: {manifest}")
            with manifest.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or not {"ir", "rgb"}.issubset(reader.fieldnames):
                    raise ValueError(f"VTMOT sequence manifest needs ir,rgb columns: {manifest}")
                stems = []
                for row in reader:
                    ir = Path(row["ir"])
                    rgb = Path(row["rgb"])
                    if (ir.parent != Path("infrared") or ir.suffix.lower() != ".jpg"
                            or rgb != Path("visible_mis") / f"{ir.stem}.png"):
                        raise ValueError(f"unsupported VTMOT pair in {manifest}: {row}")
                    stems.append(ir.stem)
            for stem in stems[::frame_stride]:
                required = (infrared_dir / f"{stem}.jpg", visible_dir / f"{stem}.png",
                            h_dir / f"{stem}.npy")
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"VTMOT manifest {manifest} references missing files "
                                            f"for {sequence}/{stem}: {', '.join(missing)}")
                if self.include_rgb_gt and not (self.root / sequence / "visible_gt" / f"{stem}.png").is_file():
                    raise FileNotFoundError(f"missing visible_gt for {sequence}/{stem}")
                self.samples.append((sequence, stem))
                if max_samples and len(self.samples) >= max_samples:
                    break
            if max_samples and len(self.samples) >= max_samples:
                break
        if not self.samples:
            raise RuntimeError(f"VTMOT split {split!r} has no samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sequence, stem = self.samples[index]
        base = self.root / sequence
        infrared_path = base / "infrared" / f"{stem}.jpg"
        visible_path = base / "visible_mis" / f"{stem}.png"
        with Image.open(infrared_path) as opened:
            source_hw = (opened.height, opened.width)
        ir = _image_tensor(infrared_path, "L", self.target_hw)
        visible = _image_tensor(visible_path, "RGB", self.target_hw)
        if tuple(visible.shape[-2:]) != self.target_hw or tuple(ir.shape[-2:]) != self.target_hw:
            raise RuntimeError("resized VTMOT rasters do not match requested target_hw")
        h_original = np.load(base / "gt_h" / f"{stem}.npy").astype(np.float64)
        transform = aspect_resize_affine(source_hw, self.target_hw)
        h_final = transform @ h_original @ np.linalg.inv(transform)
        gt_flow, valid_mask = homography_to_flow(h_final, *self.target_hw)
        h_output = h_final
        if self.crop_hw is not None:
            crop_h, crop_w = self.crop_hw
            max_top, max_left = self.target_hw[0] - crop_h, self.target_hw[1] - crop_w
            if self.random_crop:
                top = int(torch.randint(max_top + 1, ()).item())
                left = int(torch.randint(max_left + 1, ()).item())
            else:
                top, left = max_top // 2, max_left // 2
            ir = ir[:, top:top + crop_h, left:left + crop_w]
            visible = visible[:, top:top + crop_h, left:left + crop_w]
            gt_flow = gt_flow[:, top:top + crop_h, left:left + crop_w]
            # The flow values remain physical pixels after a common crop, but
            # locations mapping outside the *cropped moving image* cannot be
            # supervised or structurally compared.
            y = torch.arange(crop_h, dtype=gt_flow.dtype).view(crop_h, 1)
            x = torch.arange(crop_w, dtype=gt_flow.dtype).view(1, crop_w)
            crop_valid = ((y + gt_flow[0] >= 0) & (y + gt_flow[0] <= crop_h - 1)
                          & (x + gt_flow[1] >= 0) & (x + gt_flow[1] <= crop_w - 1))
            valid_mask = valid_mask[:, top:top + crop_h, left:left + crop_w] * crop_valid[None]
            # Return a homography expressed on the cropped coordinate system as
            # well; otherwise callers could accidentally pair cropped tensors
            # with a full-frame matrix.
            crop_transform = np.array([[1.0, 0.0, -left], [0.0, 1.0, -top],
                                       [0.0, 0.0, 1.0]])
            h_output = crop_transform @ h_final @ np.linalg.inv(crop_transform)
        sample: Dict[str, torch.Tensor | str] = {
            "ir": ir, "vi": visible, "gt_flow": gt_flow, "valid_mask": valid_mask,
            "gt_h": torch.from_numpy(h_output.astype(np.float32)),
            "sequence": sequence, "stem": stem,
        }
        if self.include_rgb_gt:
            rgb_gt = _image_tensor(base / "visible_gt" / f"{stem}.png", "RGB", self.target_hw)
            if self.crop_hw is not None:
                rgb_gt = rgb_gt[:, top:top + crop_h, left:left + crop_w]
            sample["rgb_gt"] = rgb_gt
        return sample
