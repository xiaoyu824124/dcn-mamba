"""Motion-bin and match-distribution checks for optional evaluation output."""

import unittest

import torch

from res.motion_diagnostics import frame_motion_diagnostics, summarize_motion_frames


class MotionDiagnosticsTest(unittest.TestCase):
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
