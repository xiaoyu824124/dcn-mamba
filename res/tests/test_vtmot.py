"""VTMOT geometry and curated split loading tests."""

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from res.evaluate_vtmot import evaluate
from res.mind import rgb_to_gray
from res.model_factory import build_global_registration
from res.vtmot import (VTMOTSingleFrameDataset, aspect_resize_affine,
                       homography_to_flow, registration_moving_image)
from res.warp import warp


class VTMOTGeometryTest(unittest.TestCase):
    def test_translation_is_dy_dx_and_has_correct_valid_mask(self):
        # H maps fixed [x,y] -> moving [x+4,y-3].
        h = np.array([[1., 0., 4.], [0., 1., -3.], [0., 0., 1.]])
        flow, valid = homography_to_flow(h, height=12, width=16)
        self.assertTrue(torch.allclose(flow[0], torch.full((12, 16), -3.)))
        self.assertTrue(torch.allclose(flow[1], torch.full((12, 16), 4.)))
        self.assertEqual(float(valid[:, 3:, :12].mean()), 1.0)
        self.assertEqual(float(valid[:, :3].sum()), 0.0)

    def test_flow_matches_backward_warp_convention(self):
        source = torch.arange(12 * 16, dtype=torch.float32).reshape(1, 1, 12, 16)
        h = np.array([[1., 0., 2.], [0., 1., 1.], [0., 0., 1.]])
        flow, valid = homography_to_flow(h, height=12, width=16)
        warped = warp(source, flow.unsqueeze(0), mode="nearest")
        # fixed p reads moving (x+2,y+1), which verifies x/y -> [dy,dx].
        self.assertEqual(float(warped[0, 0, 4, 5]), float(source[0, 0, 5, 7]))
        self.assertEqual(float(valid[0, 4, 5]), 1.0)

    def test_aspect_resize_is_identity_for_equal_resolution(self):
        matrix = aspect_resize_affine((480, 640), (480, 640))
        self.assertTrue(np.allclose(matrix, np.eye(3)))


class VTMOTSplitTest(unittest.TestCase):
    def test_same_modal_evaluation_uses_visible_gt_without_infrared(self):
        config = OmegaConf.merge(OmegaConf.load("res/configs/registration.yaml"),
                                 OmegaConf.load("res/configs/ab_visible_warmup.yaml"))
        config.encoder.base_channels = 8
        config.encoder.out_channels = 16
        config.encoder.blocks_per_scale = 1
        config.local_matcher.radius = 2
        model = build_global_registration(config)
        batch = {"vi": torch.rand(1, 3, 32, 40),
                 "rgb_gt": torch.rand(1, 3, 32, 40),
                 "gt_flow": torch.zeros(1, 2, 32, 40),
                 "valid_mask": torch.ones(1, 1, 32, 40),
                 "gt_h": torch.eye(3).unsqueeze(0)}
        report = evaluate(model, [batch], torch.device("cpu"),
                          moving_source="visible_gt")
        self.assertTrue(np.isfinite(report["epe_px"]))
        self.assertEqual(report["zero_flow_epe_px"], 0.0)

    def test_same_modal_warmup_uses_aligned_visible_as_moving_image(self):
        infrared = torch.full((2, 1, 16, 24), 0.2)
        aligned_visible = torch.rand(2, 3, 16, 24)
        batch = {"ir": infrared, "rgb_gt": aligned_visible}
        self.assertIs(registration_moving_image(batch, "ir"), infrared)
        self.assertTrue(torch.allclose(registration_moving_image(batch, "visible_gt"),
                                       rgb_to_gray(aligned_visible)))
        with self.assertRaisesRegex(ValueError, "include_rgb_gt"):
            registration_moving_image({"ir": infrared}, "visible_gt")
        with self.assertRaisesRegex(ValueError, "unknown moving source"):
            registration_moving_image(batch, "visible_mis")

    def test_ignores_unlisted_raw_frames_but_checks_listed_pairs(self):
        split_file = Path("lists/split.json")
        manifest = "ir,rgb,rgb_gt\ninfrared/000002.jpg,visible_mis/000002.png,visible_gt/000002.png\n"
        missing = set()
        with (patch.object(Path, "is_dir", return_value=True),
              patch.object(Path, "is_file", autospec=True,
                           side_effect=lambda path: str(path) not in missing),
              patch.object(Path, "read_text", return_value=json.dumps({"train": ["photo-0310-28"]})),
              patch.object(Path, "open", side_effect=lambda *args, **kwargs: io.StringIO(manifest)),
              patch.object(Path, "iterdir", side_effect=AssertionError("must use the CSV manifest"))):
            dataset = VTMOTSingleFrameDataset("data/VTMOT_misaligned", split="train",
                                               split_file=split_file)
            self.assertEqual(dataset.samples, [("photo-0310-28", "000002")])

            missing.add(str(Path("data/VTMOT_misaligned/photo-0310-28/gt_h/000002.npy")))
            with self.assertRaisesRegex(FileNotFoundError, r"gt_h.*000002\.npy"):
                VTMOTSingleFrameDataset("data/VTMOT_misaligned", split="train",
                                         split_file=split_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
