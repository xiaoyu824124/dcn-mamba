"""Pure geometry tests for VTMOT homography-to-flow conversion."""

import unittest

import numpy as np
import torch

from res.vtmot import aspect_resize_affine, homography_to_flow
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
