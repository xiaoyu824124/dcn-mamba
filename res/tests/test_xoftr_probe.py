"""Check the external sparse matcher boundary and affine evaluation geometry."""

import unittest

import numpy as np
import torch

from res.xoftr_probe import (affine_to_flow, evaluate_xoftr, fit_affine_ransac,
                             match_gt_error, run_xoftr)


class FakeXoFTR(torch.nn.Module):
    def __init__(self, points0, points1, confidence=None):
        super().__init__()
        self.points0 = torch.tensor(points0, dtype=torch.float32)
        self.points1 = torch.tensor(points1, dtype=torch.float32)
        self.confidence = (torch.tensor(confidence, dtype=torch.float32)
                           if confidence is not None else torch.ones(len(points0)))

    def forward(self, data):
        assert data["image0"].shape[1] == 1  # fixed visible grayscale
        assert data["image1"].shape[1] == 1  # moving infrared
        assert torch.allclose(data["image0"], torch.full_like(data["image0"], 0.299))
        assert torch.allclose(data["image1"], torch.full_like(data["image1"], 0.25))
        data["mkpts0_f"] = self.points0
        data["mkpts1_f"] = self.points1
        data["mconf_f"] = self.confidence
        data["m_bids"] = torch.zeros(len(self.points0), dtype=torch.long)


def one_frame():
    ir = torch.full((1, 1, 96, 128), 0.25)
    vi = torch.zeros(1, 3, 96, 128)
    vi[:, 0] = 1.0
    target = torch.empty(1, 2, 96, 128)
    target[:, 0] = -2.0
    target[:, 1] = 4.0
    valid = torch.ones(1, 1, 96, 128)
    return {"ir": ir, "vi": vi, "gt_flow": target, "valid_mask": valid}


class XoFTRProbeTest(unittest.TestCase):
    def test_affine_ransac_rejects_outliers_without_gt(self):
        rng = np.random.default_rng(7)
        source = rng.uniform((8, 8), (120, 88), size=(30, 2))
        affine = np.array([[1.01, 0.02, 4.0], [-0.01, 1.02, -2.0]])
        target = np.column_stack((source, np.ones(len(source)))) @ affine.T
        source = np.concatenate((source, rng.uniform((8, 8), (120, 88), size=(20, 2))))
        target = np.concatenate((target, rng.uniform((8, 8), (120, 88), size=(20, 2))))
        estimated, inliers = fit_affine_ransac(source, target, seed=0, iterations=500,
                                                threshold_px=1.0)
        self.assertIsNotNone(estimated)
        self.assertGreaterEqual(inliers, 30)
        np.testing.assert_allclose(estimated, affine, atol=1e-4)

    def test_flow_order_and_match_gt_sampling(self):
        affine = np.array([[1, 0, 4], [0, 1, -2]], dtype=np.float32)
        flow = affine_to_flow(affine, 96, 128, torch.device("cpu"))
        self.assertTrue(torch.allclose(flow[:, 0], torch.full((1, 96, 128), -2.0)))
        self.assertTrue(torch.allclose(flow[:, 1], torch.full((1, 96, 128), 4.0)))
        source = np.array([[20.25, 30.5], [50.0, 45.0]], dtype=np.float32)
        target = source + np.array([4.0, -2.0], dtype=np.float32)
        valid = torch.ones(1, 1, 96, 128)
        valid[:, :, 44:47, 49:52] = 0
        errors = match_gt_error(source, target, flow, valid)
        self.assertEqual(len(errors), 1)
        self.assertAlmostEqual(float(errors[0]), 0.0, places=4)
        all_errors = match_gt_error(source, target, flow, valid, keep_invalid=True)
        self.assertEqual(len(all_errors), 2)
        self.assertTrue(np.isnan(all_errors[1]))

    def test_full_report_includes_failed_frames_with_zero_fallback(self):
        points0 = np.array([[12, 12], [60, 12], [110, 12],
                            [12, 80], [60, 80], [110, 80],
                            [30, 30], [90, 60]], dtype=np.float32)
        points1 = points0 + np.array([4, -2], dtype=np.float32)
        batch = one_frame()
        model = FakeXoFTR(points0, points1)
        pts0, pts1, confidence = run_xoftr(model, batch["ir"], batch["vi"])
        self.assertEqual(len(pts0), 8)
        self.assertEqual(len(pts1), len(confidence))
        report = evaluate_xoftr(model, [batch], torch.device("cpu"),
                                ransac_iterations=50)
        self.assertAlmostEqual(report["epe_px"], 0.0, places=4)
        self.assertAlmostEqual(report["match_pck_3px"], 1.0)
        self.assertEqual(report["fit_success_frames"], 1)
        self.assertEqual(report["fit_success_fraction"], 1.0)

        failed = evaluate_xoftr(FakeXoFTR(points0[:2], points1[:2]),
                                 [batch], torch.device("cpu"))
        self.assertEqual(failed["fit_success_frames"], 0)
        self.assertIsNone(failed["fit_only_epe_px"])
        self.assertAlmostEqual(failed["epe_px"], failed["zero_flow_epe_px"])
        self.assertEqual(failed["match_pck_3px"], 1.0)

    def test_confidence_and_ransac_inlier_diagnostics_use_gt_only_for_scoring(self):
        rng = np.random.default_rng(21)
        points0 = rng.uniform((10, 10), (95, 75), size=(20, 2)).astype(np.float32)
        points1 = points0 + np.array([4, -2], dtype=np.float32)
        points1[14:] = points0[14:] + np.array([18, 12], dtype=np.float32)
        confidence = np.r_[np.full(14, 0.9), np.full(6, 0.1)]
        report = evaluate_xoftr(FakeXoFTR(points0, points1, confidence),
                                [one_frame()], torch.device("cpu"),
                                ransac_iterations=300)
        self.assertEqual(report["matches_total"], 20)
        self.assertAlmostEqual(report["match_pck_3px"], 0.7)
        self.assertAlmostEqual(report["match_top10pct_conf_pck_3px"], 1.0)
        self.assertLess(report["match_remaining90pct_conf_pck_3px"], 1.0)
        self.assertAlmostEqual(report["match_ransac_inlier_pck_3px"], 1.0)
        self.assertAlmostEqual(report["match_ransac_outlier_pck_3px"], 0.0)


if __name__ == "__main__":
    unittest.main()
