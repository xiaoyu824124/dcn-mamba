"""Geometry and training checks for prediction-centred local matching."""

import unittest
from types import SimpleNamespace

import torch

from res.fine_interaction import FineScaleInteraction
from res.local_matcher import LocalMatcher, local_matching_diagnostics, local_matching_loss
from res.train_vtmot import (freeze_fine_interaction_parameters,
                             freeze_local_refinement_parameters)


class LocalMatcherTest(unittest.TestCase):
    def test_known_shift_and_local_supervision(self):
        torch.manual_seed(8)
        fixed = torch.randn(1, 32, 16, 20, requires_grad=True)
        moving = torch.roll(fixed.detach(), shifts=(2, -1), dims=(-2, -1)).requires_grad_()
        coarse = torch.zeros(1, 2, 16, 20)
        matcher = LocalMatcher(radius=3, temperature=0.02, candidate_chunk=5)
        output = matcher(moving, fixed, coarse)
        central = output.matched_residual[0, :, 4:-4, 4:-4]
        self.assertLess(float((central[0] - 2).abs().mean()), 0.1)
        self.assertLess(float((central[1] + 1).abs().mean()), 0.1)
        self.assertTrue(torch.allclose(output.refined_flow, coarse))
        gt = torch.zeros(1, 2, 16 * 4, 20 * 4)
        gt[:, 0] = 8
        gt[:, 1] = -4
        diagnostics = local_matching_diagnostics(output, gt)
        self.assertAlmostEqual(diagnostics["local_window_coverage"], 1.0, places=5)
        self.assertLess(diagnostics["local_oracle_epe_px"], 1e-4)
        self.assertAlmostEqual(diagnostics["local_pred_residual_px"], 0.0, places=5)
        self.assertAlmostEqual(diagnostics["local_refinement_improved_fraction"],
                               0.0, places=5)
        self.assertLess(diagnostics["local_soft_epe_px"],
                        diagnostics["local_gt_residual_px"])
        loss = local_matching_loss(output, gt)
        self.assertTrue(torch.isfinite(loss))
        (loss + (output.refined_flow - gt[:, :, ::4, ::4] / 4).square().mean()).backward()
        self.assertTrue(torch.isfinite(fixed.grad).all())
        self.assertTrue(torch.isfinite(moving.grad).all())
        self.assertTrue(torch.isfinite(matcher.refinement[-1].weight.grad).all())

    def test_invalid_inputs(self):
        matcher = LocalMatcher(radius=2)
        with self.assertRaisesRegex(ValueError, "coarse_flow"):
            matcher(torch.rand(1, 4, 8, 8), torch.rand(1, 4, 8, 8),
                    torch.zeros(1, 2, 4, 4))

    def test_local_only_loss_trains_feature_adapter_without_updating_head(self):
        torch.manual_seed(13)
        adapter = FineScaleInteraction(8, 8, hidden_channels=8)
        matcher = LocalMatcher(radius=2, temperature=0.1)
        fine_ir, fine_vi = torch.randn(1, 8, 8, 10), torch.randn(1, 8, 8, 10)
        coarse_ir, coarse_vi = torch.randn(1, 8, 4, 5), torch.randn(1, 8, 4, 5)
        adapted_ir, adapted_vi = adapter(fine_ir, fine_vi, coarse_ir, coarse_vi)
        match = matcher(adapted_ir, adapted_vi, torch.zeros(1, 2, 8, 10))
        local_matching_loss(match, torch.zeros(1, 2, 32, 40)).backward()
        self.assertIsNotNone(adapter.gain.grad)
        self.assertGreater(float(adapter.gain.grad.abs()), 0.0)
        self.assertIsNone(matcher.refinement[-1].weight.grad)
        freeze_local_refinement_parameters(SimpleNamespace(local_matcher=matcher))
        self.assertTrue(all(not parameter.requires_grad
                            for parameter in matcher.refinement.parameters()))
        with self.assertRaisesRegex(ValueError, "freeze_local_refinement"):
            freeze_local_refinement_parameters(SimpleNamespace(local_matcher=None))

    def test_head_only_loss_keeps_descriptor_fixed(self):
        torch.manual_seed(14)
        adapter = FineScaleInteraction(8, 8, hidden_channels=8)
        matcher = LocalMatcher(radius=2, temperature=0.1)
        freeze_fine_interaction_parameters(SimpleNamespace(fine_interaction=adapter))
        fine_ir, fine_vi = torch.randn(1, 8, 8, 10), torch.randn(1, 8, 8, 10)
        coarse_ir, coarse_vi = torch.randn(1, 8, 4, 5), torch.randn(1, 8, 4, 5)
        adapted_ir, adapted_vi = adapter(fine_ir, fine_vi, coarse_ir, coarse_vi)
        output = matcher(adapted_ir, adapted_vi, torch.zeros(1, 2, 8, 10))
        target = torch.ones_like(output.refined_flow)
        (output.refined_flow - target).square().mean().backward()
        self.assertIsNone(adapter.gain.grad)
        self.assertIsNotNone(matcher.refinement[-1].weight.grad)
        self.assertGreater(float(matcher.refinement[-1].weight.grad.abs().sum()), 0.0)
        with self.assertRaisesRegex(ValueError, "freeze_fine_interaction"):
            freeze_fine_interaction_parameters(SimpleNamespace(fine_interaction=None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
