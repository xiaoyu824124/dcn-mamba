"""Tests for the standalone synthetic registration geometry."""

import unittest

import torch

from res.metrics import endpoint_error
from res.synthetic import SyntheticRegistrationDataset
from res.warp import warp


class SyntheticRegistrationDatasetTest(unittest.TestCase):
    def test_identity_pair_is_exact_and_has_zero_flow(self):
        dataset = SyntheticRegistrationDataset(length=2, height=32, width=40,
                                               translation_px=0, rotation_deg=0,
                                               scale_jitter=0, seed=7)
        sample = dataset[0]
        self.assertEqual(tuple(sample["ir"].shape), (1, 32, 40))
        self.assertEqual(tuple(sample["vi"].shape), (3, 32, 40))
        self.assertEqual(tuple(sample["gt_flow"].shape), (2, 32, 40))
        self.assertTrue(torch.equal(sample["valid_mask"], torch.ones_like(sample["valid_mask"])))
        self.assertTrue(torch.allclose(sample["gt_flow"], torch.zeros_like(sample["gt_flow"])))
        self.assertTrue(torch.allclose(sample["ir"], sample["fixed_ir"], atol=1e-6))

    def test_known_geometry_recovers_fixed_thermal_structure_on_valid_pixels(self):
        dataset = SyntheticRegistrationDataset(length=2, height=64, width=80,
                                               translation_px=12, rotation_deg=4,
                                               scale_jitter=0.03, seed=17)
        sample = dataset[1]
        recovered = warp(sample["ir"].unsqueeze(0), sample["gt_flow"].unsqueeze(0))
        error = (recovered[0] - sample["fixed_ir"]).abs() * sample["valid_mask"]
        self.assertGreater(float(sample["valid_mask"].mean()), 0.4)
        self.assertLess(float(error.sum() / sample["valid_mask"].sum()), 0.08)

    def test_indexing_is_deterministic(self):
        dataset = SyntheticRegistrationDataset(length=8, height=32, width=32, seed=91)
        first, repeated = dataset[5], dataset[5]
        for name in ("ir", "vi", "gt_flow", "valid_mask", "affine_yx"):
            self.assertTrue(torch.equal(first[name], repeated[name]))

    def test_masked_endpoint_error_ignores_invalid_locations(self):
        predicted = torch.zeros(1, 2, 4, 4)
        target = torch.zeros_like(predicted)
        predicted[..., 0, 0] = 10
        mask = torch.ones(1, 1, 4, 4)
        mask[..., 0, 0] = 0
        self.assertEqual(float(endpoint_error(predicted, target, mask)), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
