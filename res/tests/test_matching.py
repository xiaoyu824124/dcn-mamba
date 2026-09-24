"""Tests for coarse correspondence supervision and localisation diagnostics.

Run from the repository root:
    python -B -m unittest res.tests.test_matching -v
"""

import unittest

import torch

from res.global_matcher import GlobalMatcher
from res.losses import RegistrationLoss
from res.matching import (bilinear_target_cells, coarse_matching_loss,
                          correspondence_targets, matching_diagnostics,
                          windowed_diagnostics)
from res.registration_net import CoarseRegistrationOutput


def constant_flow(batch: int, height: int, width: int, dy: float, dx: float) -> torch.Tensor:
    flow = torch.zeros(batch, 2, height, width)
    flow[:, 0] = dy
    flow[:, 1] = dx
    return flow


class CorrespondenceTargetTest(unittest.TestCase):
    def test_constant_translation_maps_to_the_expected_cells(self):
        # 64x80 image on an 8x10 grid: the stride is 8px in both axes, so
        # +16px / +24px are exactly +2 / +3 cells and the interpolation of a
        # constant field stays exact.
        flow = constant_flow(1, 64, 80, 16.0, 24.0)
        target, mask = correspondence_targets(flow, (8, 10))
        self.assertEqual(tuple(target.shape), (1, 80, 2))
        picked = target[0][[11, 14, 51, 54]]                       # (1,1) (1,4) (5,1) (5,4)
        reference = target.new_tensor([[1.0, 1.0], [1.0, 4.0], [5.0, 1.0], [5.0, 4.0]])
        self.assertTrue(torch.allclose(picked, reference + target.new_tensor([2.0, 3.0])))
        self.assertTrue(bool(mask[0][[11, 14, 51, 54]].all()))
        # Border queries whose target leaves the key grid are excluded.
        self.assertFalse(bool(mask[0, 79]))

    def test_out_of_grid_targets_are_masked_out(self):
        flow = constant_flow(1, 64, 80, 32.0, 0.0)  # +4 cells on an 8-row grid
        _, mask = correspondence_targets(flow, (8, 10))
        self.assertTrue(bool(mask[0, :40].all()))     # rows 0-3 stay inside
        self.assertFalse(bool(mask[0, 40:].any()))    # rows 4-7 leave the key grid

    def test_invalid_mask_queries_are_dropped(self):
        flow = constant_flow(1, 64, 80, 0.0, 0.0)
        valid = torch.ones(1, 1, 64, 80)
        valid[:, :, :32, :] = 0.0                     # drop the top four feature rows
        _, mask = correspondence_targets(flow, (8, 10), valid)
        self.assertEqual(int(mask.sum()), 40)

    def test_bilinear_weights_sum_to_one_and_reproduce_pixel_centres(self):
        target = torch.tensor([[[1.25, 2.75]]])
        indices, weights = bilinear_target_cells(target, (8, 10))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(1, 1)))
        self.assertEqual(indices[0, 0, 0].item(), 1 * 10 + 2)
        self.assertEqual(indices[0, 0, 3].item(), 2 * 10 + 3)


class CoarseMatchingLossTest(unittest.TestCase):
    def test_target_cell_peak_has_zero_loss(self):
        # Keep one query only, so the average is not diluted by the 79 queries
        # whose target cell holds no mass.
        valid = torch.zeros(1, 1, 64, 80)
        valid[:, :, 0:8, 0:8] = 1.0
        probability = torch.zeros(1, 80, 80)
        probability[0, 0, 0] = 1.0                     # exactly the (0,0) target cell
        flow = constant_flow(1, 64, 80, 0.0, 0.0)
        self.assertAlmostEqual(float(coarse_matching_loss(probability, (8, 10), flow, valid)),
                               0.0, places=5)

    def test_flat_distribution_matches_log_n(self):
        probability = torch.full((1, 80, 80), 1.0 / 80.0)
        flow = constant_flow(1, 64, 80, 0.0, 0.0)
        loss = coarse_matching_loss(probability, (8, 10), flow)
        self.assertAlmostEqual(float(loss), torch.tensor(80.0).log().item(), places=4)

    def test_four_cell_target_spreads_the_loss(self):
        # Keep a single query (feature cell 1,1) and give its four sub-pixel
        # target cells equal mass: the loss floor is then exactly log(4).
        valid = torch.zeros(1, 1, 64, 80)
        valid[:, :, 8:16, 8:16] = 1.0
        flow = constant_flow(1, 64, 80, 4.0, 4.0)      # +0.5 / +0.4 cells
        probability = torch.zeros(1, 80, 80)
        for flat in (11, 12, 21, 22):
            probability[0, 11, flat] = 0.25
        loss = coarse_matching_loss(probability, (8, 10), flow, valid)
        self.assertAlmostEqual(float(loss), torch.tensor(4.0).log().item(), places=4)

    def test_gradient_reaches_the_probability(self):
        probability = torch.full((1, 80, 80), 1.0 / 80.0, requires_grad=True)
        flow = constant_flow(1, 64, 80, 8.0, 8.0)
        coarse_matching_loss(probability, (8, 10), flow).backward()
        self.assertIsNotNone(probability.grad)
        self.assertGreater(float(probability.grad.abs().sum()), 0.0)

    def test_focal_variant_is_finite(self):
        probability = torch.full((1, 80, 80), 1.0 / 80.0)
        flow = constant_flow(1, 64, 80, 8.0, 8.0)
        value = coarse_matching_loss(probability, (8, 10), flow, focal_gamma=2.0)
        self.assertTrue(torch.isfinite(value))


class MatchingDiagnosticsTest(unittest.TestCase):
    def test_identity_distribution_localises_every_query(self):
        probability = torch.zeros(1, 80, 80)
        probability[0] = torch.eye(80)                 # every query peaks on itself
        flow = constant_flow(1, 64, 80, 0.0, 0.0)
        report = matching_diagnostics(probability, (8, 10), flow)
        self.assertLess(report["match_frac_keys_beating_gt"], 1e-4)
        self.assertAlmostEqual(report["match_p_max"], 1.0, places=4)
        self.assertLess(report["match_epe_argmax_px"], 1e-3)
        self.assertLess(report["match_effective_keys_ratio"], 0.02)

    def test_flat_distribution_is_uninformative(self):
        probability = torch.full((1, 80, 80), 1.0 / 80.0)
        flow = constant_flow(1, 64, 80, 0.0, 0.0)
        report = matching_diagnostics(probability, (8, 10), flow)
        self.assertAlmostEqual(report["match_frac_keys_beating_gt"],
                               0.5 * (79 / 80), places=4)
        self.assertAlmostEqual(report["match_effective_keys_ratio"], 1.0, places=4)
        self.assertAlmostEqual(report["match_top1_top2_logit_gap"], 0.0, places=4)

    def test_appearance_rank_separates_feature_quality_from_probability_prior(self):
        probability = torch.eye(80).unsqueeze(0)
        appearance = torch.zeros(1, 80, 80)
        flow = constant_flow(1, 64, 80, 0.0, 0.0)
        report = matching_diagnostics(probability, (8, 10), flow,
                                      appearance_scores=appearance)
        self.assertLess(report["match_frac_keys_beating_gt"], 1e-4)
        self.assertAlmostEqual(report["appearance_frac_keys_beating_gt"],
                               0.5 * (79 / 80), places=4)
        self.assertGreater(report["appearance_epe_argmax_px"], 0.0)

    def test_metrics_are_finite_under_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        probability = torch.full((1, 80, 80), 1.0 / 80.0, device="cuda")
        flow = constant_flow(1, 64, 80, 4.0, -4.0).cuda()
        report = matching_diagnostics(probability, (8, 10), flow)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in report.values()))


class WindowedDiagnosticsTest(unittest.TestCase):
    def test_perfect_prediction_is_covered_and_correct(self):
        probability = torch.zeros(1, 80, 80)
        probability[0] = torch.eye(80)
        flow = constant_flow(1, 64, 80, 0.0, 0.0)
        report = windowed_diagnostics(probability, (8, 10), torch.zeros(1, 2, 8, 10), flow, radius=2)
        self.assertAlmostEqual(report["window2_coverage"], 1.0, places=5)
        self.assertAlmostEqual(report["window2_argmax_correct"], 1.0, places=5)
        self.assertAlmostEqual(report["window2_frac_cells_beating_gt"], 0.0, places=5)
        self.assertAlmostEqual(report["coarse_error_median_px"], 0.0, places=4)

    def test_far_prediction_is_uncovered(self):
        probability = torch.full((1, 80, 80), 1.0 / 80.0)
        flow = constant_flow(1, 64, 80, 0.0, 0.0)
        # predicted key 5 cells (40 px) away from the truth
        predicted = constant_flow(1, 8, 10, 5.0, 0.0)
        report = windowed_diagnostics(probability, (8, 10), predicted, flow, radius=2)
        self.assertAlmostEqual(report["window2_coverage"], 0.0, places=5)
        self.assertGreater(report["coarse_error_median_px"], 39.0)

    def test_bigger_window_recovers_coverage(self):
        probability = torch.zeros(1, 80, 80)
        probability[0] = torch.eye(80)
        flow = constant_flow(1, 64, 80, 0.0, 0.0)
        predicted = constant_flow(1, 8, 10, 3.0, 0.0)
        small = windowed_diagnostics(probability, (8, 10), predicted, flow, radius=2)
        large = windowed_diagnostics(probability, (8, 10), predicted, flow, radius=4)
        self.assertLess(small["window2_coverage"], large["window4_coverage"])
        self.assertAlmostEqual(large["window4_coverage"], 1.0, places=5)

    def test_coverage_counts_uncovered_queries_in_denominator(self):
        probability = torch.full((1, 80, 80), 1.0 / 80.0)
        flow = constant_flow(1, 64, 80, 0.0, 0.0)
        predicted = constant_flow(1, 8, 10, 0.0, 0.0)
        predicted[:, 0, 4:] = 5.0
        report = windowed_diagnostics(probability, (8, 10), predicted, flow, radius=2)
        self.assertGreater(report["window2_coverage"], 0.0)
        self.assertLess(report["window2_coverage"], 1.0)

    def test_competitor_inside_the_window_breaks_argmax(self):
        valid = torch.zeros(1, 1, 64, 80)
        valid[:, :, 0:8, 0:8] = 1.0                    # keep query (0,0) only
        probability = torch.zeros(1, 80, 80)
        probability[0, 0, 0] = 0.4                     # the true key
        probability[0, 0, 10] = 0.6                    # a stronger neighbour one cell down
        flow = constant_flow(1, 64, 80, 0.0, 0.0)
        report = windowed_diagnostics(probability, (8, 10), torch.zeros(1, 2, 8, 10),
                                      flow, valid, radius=2)
        self.assertAlmostEqual(report["window2_argmax_correct"], 0.0, places=5)
        self.assertAlmostEqual(report["window2_frac_cells_beating_gt"], 1.0 / 25.0, places=5)


class RegistrationLossIntegrationTest(unittest.TestCase):
    def test_match_term_backpropagates_through_the_matcher(self):
        matcher = GlobalMatcher(temperature=0.1, return_correlation=True)
        query = torch.randn(1, 8, 8, 8, requires_grad=True)
        key = torch.randn(1, 8, 8, 8, requires_grad=True)
        match = matcher(query, key)
        loss_fn = RegistrationLoss({"flow": 1.0, "match": 1.0, "mind": 0.5,
                                    "edge": 0.25, "smooth": 0.05, "affine": 1.0})
        flow = constant_flow(1, 64, 80, 4.0, 4.0)
        output = CoarseRegistrationOutput(
            coarse_aligned_ir=torch.rand(1, 1, 64, 80), coarse_flow=flow,
            confidence_1_8=match.confidence, affine_yx=match.affine_yx, match=match)
        losses = loss_fn(aligned_ir=output.coarse_aligned_ir, visible=torch.rand(1, 3, 64, 80),
                         coarse_flow=flow, gt_flow=flow, match=output.match)
        self.assertGreater(float(losses.match), 0.0)
        losses.match.backward()
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.isfinite(query.grad).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
