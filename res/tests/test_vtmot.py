"""VTMOT geometry and curated split loading tests."""

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from res.vtmot import VTMOTSingleFrameDataset, aspect_resize_affine, homography_to_flow
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
