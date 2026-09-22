# Author: Zixiang Zhao
# Last modified: 2025-10-20

import csv
import io
import json
import logging
import pickle
import random
import re
import zipfile
from enum import Enum
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image
from torch.utils.data import Dataset

class DatasetMode(Enum):
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"


def read_csv(filename, delimiter=","):
    with open(filename, "r", newline="") as f:
        csv_reader = csv.reader(f, delimiter=delimiter)
        header = next(csv_reader)
        content = [row for row in csv_reader if row]
    return header, content


def to_unit_range(image):
    """``[0, 255] -> [0, 1]``, the default ``rgb_transform``.

    This MUST stay a module-level function and never go back to being a lambda
    in the signature.  With ``num_workers > 0`` the DataLoader pickles the whole
    dataset to send it to the worker processes; a lambda defined inside a
    signature cannot be pickled, so every Windows run (spawn start method)
    failed with::

        _pickle.PicklingError: Can't pickle <function
        BaseTwoModalDataset.<lambda> ...>

    Linux happened to survive it because fork does not serialise the dataset,
    which is why the bug could stay hidden.
    """
    return image / 255.0


class BaseTwoModalDataset(Dataset):
    SEQ_IDX_KEY = "seq_idx"
    FRAME_IDX_LS_KEY = "frame_idx_ls"
    DIR_SEP_SYMBOL = "^"

    def __init__(
        self,
        filename_col_names: Dict,
        mode: DatasetMode,
        csv_dir: str,
        dataset_dir: str,
        disp_name: str,
        # Moving window
        num_frames: int,
        frame_gap_ls: List[int],
        stride: int,
        init_seed: int = None,
        frame_padding: bool = False,
        # Preprocessing
        augmentation_args: dict = None,
        rgb_transform=to_unit_range,  # must be picklable, see to_unit_range()
        resize_hw: List[int] = None,
        center_crop_to_aspect: bool = False,
        # Other
        split_filename="split.json",
        registration_gt_dir: str = None,
        registration_gt_flow: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.filename_col_names = filename_col_names[self.mode]

        # Optional supervised-registration ground truth: a per-frame homography
        # stored as `<seq>/<registration_gt_dir>/<frame-stem>.npy`.  The files
        # live in the *original* image grid and are re-projected into the final
        # (aspect-cropped / resized / flipped / random-cropped) grid by
        # `_registration_gt`.
        self.registration_gt_dir = registration_gt_dir
        self.registration_gt_flow = bool(registration_gt_flow)
        # Per-sample transform bookkeeping, reset at the start of __getitem__.
        # `_source_hw` is keyed by column because the two modalities may
        # legitimately have different native sizes (HDO: IR 1280x1024 vs
        # VIS 1920x1080, unified by `center_crop_to_aspect`).
        self._source_hw = {}
        self._aug_affine = None

        # Dataset info
        self.dataset_dir = Path(dataset_dir)
        self.csv_dir = Path(csv_dir) if csv_dir is not None else None
        del dataset_dir, csv_dir
        assert self.dataset_dir.exists(), (
            f"Dataset does not exist at: {self.dataset_dir}"
        )
        self.disp_name = disp_name

        # For multi-thread random generator
        self.is_process_init = False
        self.worker_info = None
        self.worker_id = None
        self.init_seed = init_seed

        self.rgn: torch.Generator = None

        # Moving window
        self.n_frames_per_sample: int = num_frames
        self.frame_gap_ls = frame_gap_ls
        self.stride = stride
        self.frame_padding = frame_padding
        # Random sampling setting
        self.random_sample_cfg = kwargs.pop("random_sample", None)
        flat_sequence_ids = kwargs.pop("flat_sequence_ids", None)
        flat_modal_dirs = kwargs.pop("flat_modal_dirs", None)
        self.use_random_sample = self.random_sample_cfg is not None

        if len(kwargs) > 0:
            logging.warning(f"Unexpected kwargs: {kwargs.keys()}")

        logging.debug(
            f"Dataset info: `{self.disp_name = }`: {self.n_frames_per_sample = }, {self.frame_gap_ls = }, {self.stride = }"
        )

        self.rgb_transform = rgb_transform
        # Fail here with an actionable message rather than inside a worker, where
        # the traceback is a bare PicklingError from the spawn machinery.
        try:
            pickle.dumps(rgb_transform)
        except Exception as exc:
            raise ValueError(
                "rgb_transform must be picklable: with num_workers > 0 the "
                "DataLoader serialises the dataset to its worker processes. "
                "Use a module-level function instead of a lambda or a closure."
            ) from exc
        self.resize_hw = tuple(resize_hw) if resize_hw is not None else None
        self.center_crop_to_aspect = center_crop_to_aspect

        # Training augmentation settings
        self.augm_args = augmentation_args
        logging.debug(f"{self.augm_args = }")

        # Handler of zip dataset
        self.zip_ref = None
        self.is_zip = (
            True
            if self.dataset_dir.is_file() and zipfile.is_zipfile(self.dataset_dir)
            else False
        )

        self.csv_header = None
        self.seq_rel_dir_ls_w_nframe = []  # [['sequence', '#frames'], [], ...]
        self.seq_csv_body_ls = []
        if self.registration_gt_dir is not None and (
                self.csv_dir is None or flat_sequence_ids is not None):
            raise ValueError(
                "registration_gt_dir requires CSV mode (csv_dir + split.json): "
                "the flat HDO layout has no per-frame ground-truth files, and "
                "the GT stem is taken from the first raster column's filename."
            )
        if flat_sequence_ids is not None:
            self._load_flat_sequences(flat_sequence_ids, flat_modal_dirs)
        else:
            # Get scene list for the standard CSV-based datasets.
            with open((self.csv_dir / split_filename)) as f:
                split_json = json.load(f)
            self.scene_name_ls = split_json[self.mode.value]
            for scene_name in self.scene_name_ls:
                csv_name = f"{scene_name}.csv"
                header, csv_body = read_csv((self.csv_dir / csv_name))
                if self.csv_header is None:
                    self.csv_header = header
                assert header == self.csv_header, "csv headers don't match"
                assert set(self.filename_col_names).issubset(set(header)), (
                    f"csv header doesn't contain required columns {self.filename_col_names}"
                )
                n_frame = len(csv_body)
                seq_name = Path(csv_name).stem
                seq_rel_dir = seq_name.replace(self.DIR_SEP_SYMBOL, "/")
                self.seq_rel_dir_ls_w_nframe.append([seq_rel_dir, n_frame])
                self.seq_csv_body_ls.append(csv_body)
        assert len(self.seq_rel_dir_ls_w_nframe) == len(self.seq_csv_body_ls)

        # Generate sub-sequence id
        if not self.use_random_sample:
            self.sub_seq_dict_ls = self._generate_subsequence_ids() 

    def _load_flat_sequences(self, split_ids, modal_dirs):
        """Load HDO-style flat files such as ``22 (17).jpg`` without CSVs."""
        if modal_dirs is None:
            raise ValueError("flat_modal_dirs is required with flat_sequence_ids")
        sequence_ids = split_ids[self.mode.value]
        pattern = re.compile(r"^(\d+) \((\d+)\)\.[^.]+$", re.IGNORECASE)
        modality_maps = {}
        for column in self.filename_col_names:
            directory = self.dataset_dir / modal_dirs[column]
            if not directory.is_dir():
                raise FileNotFoundError(f"HDO modality directory not found: {directory}")
            indexed = {}
            for path in directory.iterdir():
                match = pattern.match(path.name)
                if match:
                    indexed[(int(match.group(1)), int(match.group(2)))] = path.name
            modality_maps[column] = indexed

        self.csv_header = list(self.filename_col_names)
        self.scene_name_ls = []
        for sequence_id in sequence_ids:
            common_frames = None
            for column in self.filename_col_names:
                frames = {frame for seq, frame in modality_maps[column]
                          if seq == int(sequence_id)}
                common_frames = frames if common_frames is None else common_frames & frames
            common_frames = sorted(common_frames or [])
            if not common_frames:
                raise ValueError(f"No paired HDO frames found for sequence {sequence_id}")
            runs, start = [], 0
            for index in range(1, len(common_frames) + 1):
                at_end = index == len(common_frames)
                if at_end or common_frames[index] != common_frames[index - 1] + 1:
                    run = common_frames[start:index]
                    if len(run) >= self.n_frames_per_sample:
                        runs.append(run)
                    start = index
            for run_index, run in enumerate(runs):
                rows = []
                for frame in run:
                    rows.append([
                        str(Path(modal_dirs[column])
                            / modality_maps[column][(int(sequence_id), frame)])
                        for column in self.filename_col_names
                    ])
                name = f"{sequence_id}-run{run_index + 1}"
                self.scene_name_ls.append(name)
                self.seq_rel_dir_ls_w_nframe.append(["", len(rows)])
                self.seq_csv_body_ls.append(rows)
        if not self.seq_csv_body_ls:
            raise ValueError("No contiguous HDO run is long enough for one sample")

    def _generate_subsequence_ids(self):
        # Check window settings
        if self.n_frames_per_sample < 0:
            logging.info(
                "num_frames < 0. Will use all frames in the sequence. Ignoring `frame_gap_ls` and `stride`."
            )
            self.frame_gap_ls = [0]
            self.stride = 0
        else:
            assert min(self.frame_gap_ls) >= 0, (
                f"frame gap has to be >=0, found {min(self.frame_gap_ls) = }"
            )
            assert self.stride > 0, f"stride has to be > 0, found {self.stride}."

        sub_seq_dict_ls = []  # [{"seq_idx": int, "frame_idx_ls": [...]}, ...]
        for frame_gap in self.frame_gap_ls:
            frame_interval = frame_gap + 1
            for seq_id, line in enumerate(self.seq_rel_dir_ls_w_nframe):
                seq_rel_dir, n_frame = line[:2]
                n_frame = int(n_frame)

                if self.n_frames_per_sample > 0:
                    # Number of frames is specified
                    n_min_frame = frame_interval * (self.n_frames_per_sample - 1) + 1
                    # Check if frames are enough
                    if n_min_frame > n_frame:
                        logging.warning(
                            f"Not enough frames ({n_frame}) in sequence {seq_rel_dir}, min. {n_min_frame} required."
                        )
                        continue
                    # Generate sub-sequence ids
                    for i in range(0, n_frame - n_min_frame + 1, self.stride):
                        frame_id_ls = list(range(i, i + n_min_frame, frame_interval))
                        sub_seq_dict_ls.append(
                            {
                                self.SEQ_IDX_KEY: seq_id,
                                self.FRAME_IDX_LS_KEY: frame_id_ls,
                            }
                        )
                        assert max(frame_id_ls) < n_frame
                    # Repeat first and last frame
                    if self.frame_padding:
                        assert 1 == self.stride
                        assert 0 == (self.n_frames_per_sample + 1) % 2, (
                            "number of frames per sample should be odd when `frame_padding`"
                        )
                        assert self.n_frames_per_sample > 1, (
                            f"too few frames for `frame_padding` {self.n_frames_per_sample = }"
                        )
                        for n_repeat in range(
                            1, int((self.n_frames_per_sample + 1) / 2)
                        ):
                            # first frame
                            frame_id_ls = [0] * n_repeat
                            frame_id_ls.extend(
                                range(0, self.n_frames_per_sample - n_repeat)
                            )
                            sub_seq_dict_ls.insert(
                                0,
                                {
                                    self.SEQ_IDX_KEY: seq_id,
                                    self.FRAME_IDX_LS_KEY: frame_id_ls,
                                },
                            )
                            # last frame
                            frame_id_ls = list(
                                range(
                                    n_frame - (self.n_frames_per_sample - n_repeat),
                                    n_frame,
                                )
                            )
                            frame_id_ls.extend([n_frame - 1] * n_repeat)
                            sub_seq_dict_ls.append(
                                {
                                    self.SEQ_IDX_KEY: seq_id,
                                    self.FRAME_IDX_LS_KEY: frame_id_ls,
                                }
                            )
                else:
                    # Use all frames
                    sub_seq_dict_ls.append(
                        {
                            self.SEQ_IDX_KEY: seq_id,
                            self.FRAME_IDX_LS_KEY: list(range(0, n_frame)),
                        }
                    )

        return sub_seq_dict_ls

    def __len__(self):
        if self.use_random_sample:
            return self.random_sample_cfg.random_length
        else:
            return len(self.sub_seq_dict_ls)

    def _process_init_(self):
        # get worker info
        self.worker_info = torch.utils.data.get_worker_info()
        if self.worker_info is None:
            self.worker_id = 0
        else:
            self.worker_id = self.worker_info.id
        logging.debug(
            f"dataloader process initialized at {self.worker_id = }. worker_info: {self.worker_info}"
        )
        self.is_process_init = True

        # Set random seed for rgn
        if self.init_seed is not None:
            seed = (
                self.init_seed
                + 222000 * self.worker_id
                + 1100 * self.n_frames_per_sample
            )  # type: ignore
            self.rgn = torch.Generator().manual_seed(seed)
            logging.debug(
                f"Generator of '{self.disp_name}' is seeded at {self.worker_id = } with {seed = }"
            )

    def __getitem__(self, index):
        if not self.is_process_init:
            self._process_init_()

        self._source_hw = {}
        self._aug_affine = None
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)

        # merge
        outputs = rasters
        outputs.update(other)
        if self.registration_gt_dir is not None:
            outputs.update(self._registration_gt(index, outputs))
        return outputs

    def _get_data_item(self, index):
        data_path_ls_dict = self._get_data_path(index=index)

        rasters: Dict[str, List] = {}  # list of rastersW

        for col_name in self.filename_col_names:
            rasters[col_name] = self._load_rgb_data(
                data_path_ls_dict[col_name], col_name=col_name)

        other = {
            "index": index,
            "data_path_ls_dict": data_path_ls_dict,
            "dataset": self.disp_name,
        }

        return rasters, other

    def _get_data_path(self, index):
        sub_seq_dict = self.sub_seq_dict_ls[index]
        seq_idx = sub_seq_dict[self.SEQ_IDX_KEY]
        frame_idx_ls = sub_seq_dict[self.FRAME_IDX_LS_KEY]

        seq_filenames = self.seq_csv_body_ls[seq_idx]
        seq_rel_dir = self.seq_rel_dir_ls_w_nframe[seq_idx][0]
        result_dict = {}
        for i_col in range(len(self.filename_col_names)):
            col_name = self.filename_col_names[i_col]
            result_dict[col_name] = [
                str(Path(seq_rel_dir) / seq_filenames[idx][i_col])
                for idx in frame_idx_ls
            ]
        return result_dict

    def _load_rgb_data(self, rgb_rel_path_ls, col_name=None):
        # Read RGB data
        rgb_int_ls = []
        rgb_norm_ls = []
        for rgb_rel_path in rgb_rel_path_ls:
            rgb = self._read_single_image(rgb_rel_path, col_name=col_name)
            rgb_norm = self.rgb_transform(rgb)
            rgb_int_ls.append(torch.from_numpy(rgb).int().unsqueeze(0))
            rgb_norm_ls.append(torch.from_numpy(rgb_norm).float().unsqueeze(0))

        # rgb_int = torch.concat(rgb_int_ls, dim=0)  # [N, rgb, H, W]
        rgb_norm = torch.concat(rgb_norm_ls, dim=0)  # [N, rgb, H, W]

        return rgb_norm

    def _aspect_crop_box(self, source_w, source_h):
        """Centre-crop box used to reach the target aspect ratio.

        Returns ``(crop_h, crop_w, top, left)``.  Shared by
        :meth:`_read_single_image` and :meth:`_preprocess_affine` so the pixels
        that are read and the coordinates used for the homography can never
        drift apart.
        """
        if self.resize_hw is None or not self.center_crop_to_aspect:
            return source_h, source_w, 0, 0
        target_h, target_w = self.resize_hw
        target_aspect = target_w / target_h
        source_aspect = source_w / source_h
        if source_aspect > target_aspect:
            crop_w = round(source_h * target_aspect)
            return source_h, crop_w, 0, (source_w - crop_w) // 2
        if source_aspect < target_aspect:
            crop_h = round(source_w / target_aspect)
            return crop_h, source_w, (source_h - crop_h) // 2, 0
        return source_h, source_w, 0, 0

    def _read_single_image(self, img_rel_path, col_name=None) -> np.ndarray:
        if self.is_zip:
            image_to_read = self._read_from_zip(img_rel_path)
        else:
            image_to_read = self.dataset_dir / img_rel_path
        image = Image.open(image_to_read).convert("RGB")
        source_w, source_h = image.size
        # All frames of one column must share a geometry, otherwise the shared
        # homography target would be meaningless.  Different columns may differ
        # (HDO), which is exactly why the bookkeeping is per column.
        if col_name is not None:
            previous = self._source_hw.get(col_name)
            if previous is None:
                self._source_hw[col_name] = (source_w, source_h)
            elif previous != (source_w, source_h):
                raise ValueError(
                    f"frames of one sample differ in size for column "
                    f"'{col_name}': {previous} vs {(source_w, source_h)} at "
                    f"{img_rel_path}"
                )
        if self.resize_hw is not None:
            target_h, target_w = self.resize_hw
            crop_h, crop_w, top, left = self._aspect_crop_box(source_w, source_h)
            if crop_w < source_w or crop_h < source_h:
                image = image.crop((left, top, left + crop_w, top + crop_h))
            image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
        image = np.asarray(image)

        # Check if the image is grayscale (i.e., has only one channel)
        if image.ndim == 2:  # Grayscale image
            image = np.stack(
                [image] * 3, axis=0
            )  # Convert to 3 channels by stacking the same image three times
        else:
            image = np.transpose(image, (2, 0, 1))  # [rgb, H, W]
        image = image.astype(int)  # Convert the image to integer type
        return image

    def _read_npy(self, npy_rel_path) -> np.ndarray:
        if self.is_zip:
            npy_to_read = self._read_from_zip(npy_rel_path)
        else:
            npy_to_read = self.dataset_dir, npy_rel_path
        image = np.load(npy_to_read)
        return image

    def _read_from_zip(self, rel_path: str) -> io.BytesIO:
        if self.zip_ref is None:
            self.zip_ref = zipfile.ZipFile(self.dataset_dir)
        file_data = self.zip_ref.read(rel_path)
        file_data = io.BytesIO(file_data)
        return file_data

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        return rasters

    def _augment_data(self, rasters_dict):
        # Affine mapping post-resize pixel coords -> final pixel coords, built in
        # the same order as the operations below (flip, then random crop).
        affine = np.eye(3, dtype=np.float64)

        # left-right flipping
        lr_flip_p = self.augm_args.lr_flip_p
        if random.random() < lr_flip_p:
            rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}
            width = rasters_dict[list(rasters_dict.keys())[0]].shape[-1]
            flip = np.array([[-1.0, 0.0, width - 1.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
            affine = flip @ affine

        # Add a smooth, time-varying geometric offset to IR only. This expands
        # the registration motion range while preserving real HDO content.
        misalignment = self.augm_args.get("synthetic_misalignment")
        if misalignment and random.random() < misalignment.get("p", 0.0):
            if self.registration_gt_dir is not None:
                raise RuntimeError(
                    "synthetic_misalignment perturbs the IR geometry without "
                    "recording the transform, which would silently corrupt the "
                    "supervised registration target. Set "
                    "augmentation.synthetic_misalignment.p = 0 for supervised "
                    "datasets such as VTMOT."
                )
            infrared = rasters_dict.get("ir")
            if infrared is not None:
                max_translation = float(misalignment.get("max_translation", 0.0))
                max_rotation = float(misalignment.get("max_rotation", 0.0))
                max_drift = float(misalignment.get("max_drift", 0.0))
                base_x = random.uniform(-max_translation, max_translation)
                base_y = random.uniform(-max_translation, max_translation)
                base_angle = random.uniform(-max_rotation, max_rotation)
                drift_x = random.uniform(-max_drift, max_drift)
                drift_y = random.uniform(-max_drift, max_drift)
                drift_angle = random.uniform(-0.25 * max_rotation,
                                             0.25 * max_rotation)
                transformed = []
                denominator = max(infrared.shape[0] - 1, 1)
                for index, frame in enumerate(infrared):
                    phase = index / denominator - 0.5
                    transformed.append(F.affine(
                        frame,
                        angle=base_angle + phase * drift_angle,
                        translate=[round(base_x + phase * drift_x),
                                   round(base_y + phase * drift_y)],
                        scale=1.0,
                        shear=[0.0, 0.0],
                        interpolation=InterpolationMode.BILINEAR,
                    ))
                rasters_dict["ir"] = torch.stack(transformed)

        # random crop
        crop_size = self.augm_args.random_crop_hw
        if crop_size is not None:
            _, _, h, w = rasters_dict[list(rasters_dict.keys())[0]].shape
            top = torch.randint(0, h - crop_size[0] + 1, (1,)).item()
            left = torch.randint(0, w - crop_size[1] + 1, (1,)).item()
            rasters_dict = {
                k: F.crop(img, top, left, crop_size[0], crop_size[1])
                for k, img in rasters_dict.items()
            }
            translate = np.array([[1.0, 0.0, -float(left)],
                                  [0.0, 1.0, -float(top)],
                                  [0.0, 0.0, 1.0]], dtype=np.float64)
            affine = translate @ affine

        self._aug_affine = affine
        return rasters_dict

    def _preprocess_affine(self):
        """Original-image pixel coords -> post-(aspect-crop + resize) coords.

        The half-pixel intercept ``0.5 * s - 0.5`` is not a guess: it was
        measured on this PIL build with single-pixel impulses and fits
        ``x_dst = s * (x_src + 0.5) - 0.5`` to better than 0.005 px (see
        ``tools/test_gt_transform.py``).
        """
        affine = np.eye(3, dtype=np.float64)
        if self.resize_hw is None:
            return affine
        # The GT is defined on the grid of the column the frame stems come from.
        source_hw = self._source_hw.get(self.filename_col_names[0])
        if source_hw is None:
            return affine
        target_h, target_w = self.resize_hw
        source_w, source_h = source_hw
        crop_h, crop_w, top, left = self._aspect_crop_box(source_w, source_h)
        crop = np.array([[1.0, 0.0, -float(left)],
                         [0.0, 1.0, -float(top)],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
        sx, sy = target_w / crop_w, target_h / crop_h
        scale = np.array([[sx, 0.0, 0.5 * sx - 0.5],
                          [0.0, sy, 0.5 * sy - 0.5],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
        return scale @ crop

    def _registration_gt(self, index, outputs):
        """Supervised-registration target in the *final* (augmented) image grid.

        ``gt_h[n]`` maps fixed-image pixel coords to moving-image pixel coords::

            rgb_final(p) == moving_final(gt_h @ p)

        where ``rgb`` is the misaligned reference and ``moving`` is either ``ir``
        or the aligned ``rgb_gt`` frame (both share the same target, which is why
        the mono-modal warm-up and the cross-modal main task need no separate GT).
        ``gt_flow`` is the matching ``[dy, dx]`` field,
        ``gt_flow(p) = gt_h @ p - p``, so ``warp(moving, gt_flow) == rgb``.

        The stored homography is defined on the original image grid; a change of
        coordinates by ``A`` conjugates it, ``H_final = A @ H_orig @ inv(A)``,
        which handles the aspect crop, the resize, the horizontal flip and the
        random crop uniformly.
        """
        sub_seq = self.sub_seq_dict_ls[index]
        seq_idx = sub_seq[self.SEQ_IDX_KEY]
        frame_idx_ls = sub_seq[self.FRAME_IDX_LS_KEY]
        seq_rel_dir = self.seq_rel_dir_ls_w_nframe[seq_idx][0]
        seq_filenames = self.seq_csv_body_ls[seq_idx]

        h_ls = []
        for frame_idx in frame_idx_ls:
            stem = Path(seq_filenames[frame_idx][0]).stem
            path = (self.dataset_dir / seq_rel_dir / self.registration_gt_dir
                    / f"{stem}.npy")
            if not path.is_file():
                raise FileNotFoundError(f"registration GT missing: {path}")
            h = np.load(path).astype(np.float64)
            if h.shape != (3, 3):
                raise ValueError(
                    f"{path}: expected a 3x3 homography, got {h.shape}")
            h_ls.append(h)
        h_orig = np.stack(h_ls)  # [N, 3, 3] on the original image grid

        # A single 3x3 can only relate the two modalities if they share one
        # native grid, which is what makes `gt_flow` valid for both the
        # cross-modal and the mono-modal moving column.
        native = {col: self._source_hw.get(col) for col in self.filename_col_names}
        if len(set(native.values())) != 1:
            raise ValueError(
                "registration GT requires all columns to share one native "
                f"resolution, got {native}"
            )

        affine = self._preprocess_affine()
        if self._aug_affine is not None:
            affine = self._aug_affine @ affine
        affine_inv = np.linalg.inv(affine)
        h_final = np.ascontiguousarray(affine @ h_orig @ affine_inv,
                                       dtype=np.float32)  # [N, 3, 3]

        result = {"gt_h": torch.from_numpy(h_final)}
        if self.registration_gt_flow:
            height, width = outputs[self.filename_col_names[0]].shape[-2:]
            yy, xx = np.meshgrid(np.arange(height, dtype=np.float64),
                                 np.arange(width, dtype=np.float64),
                                 indexing="ij")
            pixels = np.stack([xx, yy, np.ones_like(xx)], axis=0).reshape(3, -1)
            mapped = np.einsum("nij,jk->nik", h_final.astype(np.float64), pixels)
            mapped = mapped[:, :2] / mapped[:, 2:]
            dx = mapped[:, 0].reshape(-1, height, width) - xx
            dy = mapped[:, 1].reshape(-1, height, width) - yy
            # Flow convention matches SpatialTransformer: [dy, dx] and
            # warped(y, x) = source(y + dy, x + dx).
            flow = np.stack([dy, dx], axis=1).astype(np.float32)  # [N, 2, H, W]
            result["gt_flow"] = torch.from_numpy(flow)
        return result

    def _close_zip(self):
        if hasattr(self, "zip_ref") and self.zip_ref is not None:
            self.zip_ref.close()
            self.zip_ref = None

    def __del__(self):
        self._close_zip()
