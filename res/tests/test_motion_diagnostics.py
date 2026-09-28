"""Motion-bin and match-distribution checks for optional evaluation output."""

import unittest

import torch

from res.motion_diagnostics import (frame_motion_diagnostics,
                                    summarize_motion_frames,
                                    translate_moving_for_stress)
from res.warp import warp


class MotionDiagnosticsTest(unittest.TestCase):
    def test_stress_translation_preserves_alignment_and_updates_homography(self):
        source = torch.arange(12 * 16, dtype=torch.float32).reshape(1, 1, 12, 16)
        flow = torch.zeros(1, 2, 12, 16)
        flow[:, 0] = 1
        flow[:, 1] = 2
        valid = torch.ones(1, 1, 12, 16)
        valid[:, :, 11:, :] = 0
        valid[:, :, :, 14:] = 0
        h = torch.tensor([[[1., 0., 2.], [0., 1., 1.], [0., 0., 1.]]])
        shifted, new_flow, new_valid, new_h = translate_moving_for_stress(
            source, flow, valid, h, (2, -1))
        self.assertTrue(torch.allclose(new_flow[:, 0], torch.full((1, 12, 16), 3.)))
        self.assertTrue(torch.allclose(new_flow[:, 1], torch.full((1, 12, 16), 1.)))
        self.assertEqual(float(new_h[0, 0, 2]), 1.0)
        self.assertEqual(float(new_h[0, 1, 2]), 3.0)
        self.assertEqual(int(new_valid.sum()), 9 * 14)
        original_aligned = warp(source, flow)
        stressed_aligned = warp(shifted, new_flow)
        self.assertTrue(torch.allclose(original_aligned * new_valid,
                                       stressed_aligned * new_valid, atol=1e-5))

    def test_pixel_bins_and_match_distribution(self):
        target = torch.zeros(1, 2, 16, 16)
        target[:, 1, :, 8:] = 8
        predicted = target.clone()
        predicted[:, 1] += 2
        coarse = torch.zeros_like(target)
        valid = torch.ones(1, 1, 16, 16)
        valid[:, :, 0, 0] = 0
        probability = torch.eye(4).unsqueeze(0) * 0.9 + 0.025
        row = frame_motion_diagnostics(
            predicted, coarse, target, valid, probability, (2, 2),
            sequence="sample", stem="frame")
        self.assertEqual(row["valid_pixels"], 255)
        self.assertEqual(row["pixel_bins"]["0-4px"]["pixels"], 127)
        self.assertEqual(row["pixel_bins"]["8-16px"]["pixels"], 128)
        self.assertAlmostEqual(row["epe_px"], 2.0)
        self.assertAlmostEqual(sum(row["match_distribution"]["wls_weight_quadrants"].values()),
                               1.0, places=5)
        self.assertAlmostEqual(sum(
            item["predicted_fraction"] for item in
            row["match_distribution"]["top10_conf_motion_bins"].values()),
            1.0, places=5)
        report = summarize_motion_frames([row])
        self.assertEqual(report["motion_frame_bins"]["4-8px"]["frames"], 1)
        self.assertEqual(report["motion_pixel_bins"]["0-4px"]["pixels"], 127)
        self.assertAlmostEqual(report["motion_pixel_bins"]["8-16px"]["epe_px"], 2.0)
        self.assertEqual(report["motion_frames"][0]["sequence"], "sample")

    def test_empty_motion_bins_are_explicit(self):
        target = torch.zeros(1, 2, 16, 16)
        valid = torch.ones(1, 1, 16, 16)
        row = frame_motion_diagnostics(target, target, target, valid, None)
        report = summarize_motion_frames([row])
        self.assertIsNone(report["motion_pixel_bins"][">=32px"]["epe_px"])
        self.assertEqual(report["motion_frame_bins"][">=16px"]["frames"], 0)
        self.assertIsNone(report["match_distribution_summary"])


if __name__ == "__main__":
    unittest.main()
