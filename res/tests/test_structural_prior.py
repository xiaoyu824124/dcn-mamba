"""Checks for phase-congruency priors and cross-modal shallow encoders."""

import unittest

import torch
from omegaconf import OmegaConf

from res.evaluate_vtmot import evaluate
from res.losses import RegistrationLoss
from res.model_factory import build_global_registration
from res.motion_diagnostics import translate_moving_for_stress
from res.structural_prior import phase_congruency_prior


class StructuralPriorTest(unittest.TestCase):
    def test_large_shift_overlay_keeps_supervised_backward_finite(self):
        config = OmegaConf.merge(
            OmegaConf.load("res/configs/registration.yaml"),
            OmegaConf.load("res/configs/stage0_structural_prior.yaml"),
            OmegaConf.load("res/configs/ab_spatial_prior32.yaml"),
            OmegaConf.load("res/configs/ab_large_translation_train.yaml"))
        config.structural_prior.base_channels = 8
        config.structural_prior.out_channels = 16
        config.structural_prior.blocks_per_scale = 1
        config.coarse_transformer.num_layers = 1
        config.global_matcher.max_tokens = 80
        model = build_global_registration(config)
        ir = torch.rand(1, 1, 64, 80)
        vi = torch.rand(1, 3, 64, 80)
        gt = torch.zeros(1, 2, 64, 80)
        valid = torch.ones(1, 1, 64, 80)
        gt_h = torch.eye(3).unsqueeze(0)
        ir, gt, valid, gt_h = translate_moving_for_stress(
            ir, gt, valid, gt_h, (0, 24), padding_mode="reflection")
        output = model(ir, vi)
        loss = RegistrationLoss(config.loss.weights)(
            aligned_ir=output.coarse_aligned_ir, visible=vi,
            coarse_flow=output.coarse_flow, gt_flow=gt, valid_mask=valid,
            predicted_affine_yx=output.affine_yx, gt_h=gt_h,
            affine_feature_hw=tuple(output.confidence_1_8.shape[-2:]),
            match=output.match)
        self.assertTrue(torch.isfinite(loss.total))
        loss.total.backward()
        self.assertGreater(float(model.encoder.ir_shallow[0][0].weight.grad.abs().sum()),
                           0.0)

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
        batch["sequence"] = ["synthetic"]
        batch["stem"] = ["one"]
        motion_report = evaluate(model.eval(), [batch], torch.device("cpu"),
                                 diagnose_motion=True)
        self.assertEqual(motion_report["motion_frames"][0]["sequence"], "synthetic")
        self.assertEqual(motion_report["motion_pixel_bins"]["0-4px"]["pixels"],
                         64 * 80)
        self.assertIsNotNone(motion_report["match_distribution_summary"])
        stress_report = evaluate(model.eval(), [batch], torch.device("cpu"),
                                 diagnose_motion=True, stress_translation=(0, 8))
        self.assertEqual(stress_report["stress_translation_dy_dx"], [0, 8])
        self.assertEqual(stress_report["motion_pixel_bins"]["8-16px"]["pixels"],
                         64 * 72)

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
        prior_config = OmegaConf.merge(
            config, OmegaConf.load("res/configs/ab_spatial_prior32.yaml"))
        prior_model = build_global_registration(prior_config)
        self.assertEqual(prior_model.matcher.spatial_prior_sigma, 4.0)


if __name__ == "__main__":
    unittest.main()
