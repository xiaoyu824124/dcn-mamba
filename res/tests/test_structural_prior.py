"""Checks for phase-congruency priors and cross-modal shallow encoders."""

import unittest

import torch
from omegaconf import OmegaConf

from res.evaluate_vtmot import evaluate
from res.model_factory import build_global_registration
from res.structural_prior import phase_congruency_prior


class StructuralPriorTest(unittest.TestCase):
    def test_phase_congruency_is_finite_and_contrast_sign_invariant(self):
        torch.manual_seed(8)
        image = torch.rand(1, 1, 64, 80) * 2 - 1
        congruency, amplitude = phase_congruency_prior(image)
        flipped_congruency, flipped_amplitude = phase_congruency_prior(-image)
        self.assertEqual(tuple(congruency.shape), (1, 1, 64, 80))
        self.assertTrue(torch.isfinite(congruency).all())
        self.assertTrue(torch.isfinite(amplitude).all())
        self.assertTrue(torch.allclose(congruency, flipped_congruency, atol=1e-5))
        self.assertTrue(torch.allclose(amplitude, flipped_amplitude, atol=1e-5))
        flat_congruency, flat_amplitude = phase_congruency_prior(
            torch.zeros_like(image))
        self.assertTrue(torch.isfinite(flat_congruency).all())
        self.assertTrue(torch.isfinite(flat_amplitude).all())
        self.assertEqual(float(flat_congruency.max()), 0.0)

    def test_modality_shallow_and_shared_encoder_receive_gradient(self):
        config = OmegaConf.merge(
            OmegaConf.load("res/configs/registration.yaml"),
            OmegaConf.load("res/configs/stage0_structural_prior.yaml"))
        config.structural_prior.base_channels = 8
        config.structural_prior.out_channels = 16
        config.structural_prior.blocks_per_scale = 1
        config.global_matcher.max_tokens = 80
        config.coarse_transformer.num_layers = 1
        model = build_global_registration(config)
        torch.manual_seed(9)
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        output = model(ir, vi)
        self.assertEqual(output.match.matching_probability.shape, (1, 80, 80))
        self.assertEqual(output.coarse_flow.shape, (1, 2, 64, 80))
        self.assertTrue(torch.isfinite(output.coarse_flow).all())
        loss = (output.coarse_flow - 1).square().mean()
        loss = loss + output.match.matching_probability[:, 0, 1].mean()
        loss.backward()
        for parameter in (
                model.encoder.ir_shallow[0][0].weight,
                model.encoder.vi_shallow[0][0].weight,
                model.encoder.shared.stage_8[0][0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        batch = {"ir": ir, "vi": vi,
                 "gt_flow": torch.zeros(1, 2, 64, 80),
                 "valid_mask": torch.ones(1, 1, 64, 80),
                 "gt_h": torch.eye(3).unsqueeze(0)}
        report = evaluate(model.eval(), [batch], torch.device("cpu"))
        self.assertEqual(report["architecture"], "structural_prior")
        self.assertIn("affine_weight_effective_queries_ratio", report)

    def test_direct_baseline_has_no_feature_parameters(self):
        config = OmegaConf.merge(
            OmegaConf.load("res/configs/registration.yaml"),
            OmegaConf.load("res/configs/stage0_structural_prior.yaml"),
            OmegaConf.load("res/configs/ab_structural_direct.yaml"))
        config.global_matcher.max_tokens = 80
        model = build_global_registration(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 0)
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        output = model(ir, vi)
        self.assertEqual(output.match.matching_probability.shape, (1, 80, 80))
        self.assertTrue(torch.isfinite(output.coarse_flow).all())
        batch = {"ir": ir, "vi": vi,
                 "gt_flow": torch.zeros(1, 2, 64, 80),
                 "valid_mask": torch.ones(1, 1, 64, 80),
                 "gt_h": torch.eye(3).unsqueeze(0)}
        report = evaluate(model.eval(), [batch], torch.device("cpu"))
        self.assertEqual(report["architecture"], "structural_prior_direct")


if __name__ == "__main__":
    unittest.main()
