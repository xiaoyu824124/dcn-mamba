"""Coordinate tests for VTMOT-to-matcher affine supervision."""

import unittest

import torch

from res.affine import (homography_to_normalised_affine_yx,
                        normalised_affine_yx_to_homography, affine_corner_errors)


class AffineConversionTest(unittest.TestCase):
    def test_round_trip_preserves_xy_affine(self):
        gt_h = torch.tensor([[[1.02, -0.04, 12.0],
                              [0.03, 0.97, -7.0],
                              [0.0, 0.0, 1.0]]])
        affine = homography_to_normalised_affine_yx(gt_h, (480, 640), (60, 80))
        restored = normalised_affine_yx_to_homography(affine, (480, 640), (60, 80))
        self.assertTrue(torch.allclose(restored, gt_h, atol=2e-5, rtol=1e-5))

    def test_projective_gt_is_rejected(self):
        gt_h = torch.eye(3).unsqueeze(0)
        gt_h[:, 2, 0] = 1e-3
        with self.assertRaisesRegex(ValueError, "affine"):
            homography_to_normalised_affine_yx(gt_h, (64, 80), (8, 10))

    def test_matching_affine_has_zero_corner_and_cycle_error(self):
        gt_h = torch.tensor([[[1.0, 0.0, 5.0],
                              [0.0, 1.0, -3.0],
                              [0.0, 0.0, 1.0]]])
        affine = homography_to_normalised_affine_yx(gt_h, (64, 80), (8, 10))
        corner, cycle = affine_corner_errors(affine, gt_h, (64, 80), (8, 10))
        self.assertLess(float(corner.max()), 1e-4)
        self.assertLess(float(cycle.max()), 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
