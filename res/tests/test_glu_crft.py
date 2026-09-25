"""Geometry and integration checks for the new coarse registration path."""

import unittest

import torch
from omegaconf import OmegaConf

from res.affine import affine_corner_errors
from res.glu_crft_coarse import fit_affine_flow, standardize_image
from res.losses import RegistrationLoss
from res.model_factory import build_global_registration
from res.evaluate_vtmot import evaluate


def tiny_config(with_refiner: bool = False):
    files = ["res/configs/registration.yaml", "res/configs/stage0_glu_crft_coarse.yaml"]
    if with_refiner:
        files.append("res/configs/stage1_glu_crft_coarse_local.yaml")
    config = OmegaConf.merge(*(OmegaConf.load(path) for path in files))
    config.encoder.base_channels = 8
    config.encoder.out_channels = 16
    config.encoder.blocks_per_scale = 1
    config.coarse_decoder.grid_hw = [4, 5]
    config.coarse_decoder.hidden_channels = 16
    config.global_matcher.max_tokens = 20
    config.coarse_transformer.num_layers = 1
    return config


class GLUCRFTCoarseTest(unittest.TestCase):
    def test_affine_projection_preserves_translation_and_shear_without_clipping(self):
        y, x = torch.meshgrid(torch.arange(8), torch.arange(10), indexing="ij")
        flow = torch.stack((2 + 0.1 * y + 0.2 * x,
                            -3 + 0.05 * y - 0.1 * x), dim=0).unsqueeze(0)
        fitted, parameters = fit_affine_flow(flow)
        self.assertTrue(torch.allclose(fitted, flow, atol=2e-3))
        self.assertEqual(tuple(parameters.shape), (1, 3, 2))
        self.assertLess(float(fitted[0, 1, 0, 0]), -2.9)

    def test_affine_projection_matches_image_grid_direction(self):
        flow = torch.zeros(1, 2, 4, 5)
        flow[:, 0] = -2 / 16
        flow[:, 1] = 3 / 16
        _, parameters = fit_affine_flow(flow)
        gt_h = torch.tensor([[[1., 0., 3.], [0., 1., -2.], [0., 0., 1.]]])
        corner_error, _ = affine_corner_errors(parameters, gt_h, (64, 80), (4, 5))
        self.assertLess(float(corner_error), 1e-2)

    def test_image_standardization_handles_flat_modality(self):
        flat = torch.full((2, 1, 8, 10), 0.5)
        self.assertTrue(torch.isfinite(standardize_image(flat)).all())
        self.assertTrue(torch.equal(standardize_image(flat), torch.zeros_like(flat)))

    def test_stage0_cost_decoder_receives_gradient(self):
        torch.manual_seed(12)
        config = tiny_config()
        model = build_global_registration(config)
        ir = torch.rand(1, 1, 64, 80)
        vi = torch.rand(1, 3, 64, 80)
        result = model(ir, vi)
        self.assertEqual(tuple(result.match.matching_probability.shape), (1, 20, 20))
        self.assertEqual(tuple(result.coarse_flow.shape), (1, 2, 64, 80))
        self.assertIsNone(result.coarse_local_match)
        self.assertEqual(tuple(result.confidence_1_8.shape), (1, 1, 4, 5))
        criterion = RegistrationLoss(config.loss.weights)
        target = torch.ones_like(result.coarse_flow)
        loss = criterion(aligned_ir=result.coarse_aligned_ir, visible=vi,
                         coarse_flow=result.coarse_flow, gt_flow=target,
                         valid_mask=torch.ones(1, 1, 64, 80), match=result.match,
                         predicted_affine_yx=result.affine_yx,
                         affine_feature_hw=result.confidence_1_8.shape[-2:],
                         gt_h=torch.eye(3).unsqueeze(0))
        loss.total.backward()
        self.assertTrue(torch.isfinite(loss.total))
        self.assertIsNotNone(model.coarse_decoder.network[-1].weight.grad)
        self.assertGreater(float(model.coarse_decoder.network[-1].weight.grad.abs().sum()), 0)
        self.assertIsNotNone(model.encoder.stage_16[0][0].weight.grad)

    def test_stage1_refines_on_one_eighth_grid(self):
        torch.manual_seed(13)
        config = tiny_config(with_refiner=True)
        model = build_global_registration(config)
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        result = model(ir, vi)
        self.assertEqual(tuple(result.coarse_local_match.probability.shape),
                         (1, 81, 8, 10))
        self.assertEqual(tuple(result.confidence_1_8.shape), (1, 1, 8, 10))
        self.assertEqual(tuple(result.global_flow.shape), (1, 2, 64, 80))
        self.assertEqual(tuple(result.coarse_flow.shape), (1, 2, 64, 80))
        criterion = RegistrationLoss(config.loss.weights)
        loss = criterion(aligned_ir=result.coarse_aligned_ir, visible=vi,
                         coarse_flow=result.coarse_flow, global_flow=result.global_flow,
                         gt_flow=torch.ones_like(result.coarse_flow),
                         valid_mask=torch.ones(1, 1, 64, 80),
                         predicted_affine_yx=result.affine_yx,
                         affine_feature_hw=result.confidence_1_8.shape[-2:],
                         gt_h=torch.eye(3).unsqueeze(0), match=result.match,
                         coarse_local_match=result.coarse_local_match)
        loss.total.backward()
        self.assertIsNotNone(model.coarse_refiner.refinement[-1].weight.grad)
        self.assertGreater(float(model.coarse_refiner.refinement[-1].weight.grad.abs().sum()), 0)
        self.assertIsNotNone(model.coarse_decoder.network[-1].weight.grad)

    def test_stage2_and_shared_evaluator(self):
        torch.manual_seed(14)
        config = tiny_config(with_refiner=True)
        config = OmegaConf.merge(config,
                                 OmegaConf.load("res/configs/stage2_glu_crft_fine.yaml"))
        model = build_global_registration(config)
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        result = model(ir, vi)
        self.assertEqual(tuple(result.local_match.probability.shape), (1, 169, 16, 20))
        criterion = RegistrationLoss(config.loss.weights)
        batch = {"ir": ir, "vi": vi, "gt_flow": torch.zeros(1, 2, 64, 80),
                 "valid_mask": torch.ones(1, 1, 64, 80),
                 "gt_h": torch.eye(3).unsqueeze(0)}
        loss = criterion(aligned_ir=result.final_aligned_ir, visible=vi,
                         coarse_flow=result.coarse_flow, final_flow=result.final_flow,
                         global_flow=result.global_flow, gt_flow=batch["gt_flow"],
                         valid_mask=batch["valid_mask"],
                         predicted_affine_yx=result.affine_yx,
                         affine_feature_hw=result.confidence_1_8.shape[-2:],
                         gt_h=batch["gt_h"], match=result.match,
                         local_match=result.local_match,
                         coarse_local_match=result.coarse_local_match)
        self.assertTrue(torch.isfinite(loss.total))
        self.assertGreater(float(loss.coarse_local), 0)
        self.assertGreater(float(loss.global_flow), 0)
        report = evaluate(model, [batch], torch.device("cpu"))
        self.assertEqual(report["coarse_match_candidates"], 20)
        self.assertIn("global_epe_px", report)
        self.assertIn("coarse_local_window_coverage", report)


if __name__ == "__main__":
    unittest.main()
