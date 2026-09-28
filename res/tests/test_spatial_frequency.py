"""Local spectrum geometry and coarse-only ablation integration."""

import unittest

import torch
from omegaconf import OmegaConf

from res.model_factory import build_global_registration
from res.evaluate_vtmot import evaluate
from res.spatial_frequency import local_spectrum
from res.visualize_spatial_frequency import _correspondence_image


class SpatialFrequencyTest(unittest.TestCase):
    def test_local_spectrum_stays_on_the_one_eighth_lattice(self):
        torch.manual_seed(4)
        image = torch.rand(1, 1, 64, 80)
        amplitude, phase = local_spectrum(image, window_size=5)
        self.assertEqual(tuple(amplitude.shape), (1, 15, 8, 10))
        self.assertEqual(tuple(phase.shape), (1, 30, 8, 10))
        self.assertTrue(torch.isfinite(amplitude).all())
        self.assertTrue(torch.isfinite(phase).all())
        shifted = torch.roll(image, shifts=8, dims=-1)
        shifted_amp, shifted_phase = local_spectrum(shifted, window_size=5)
        # One 8px image shift becomes one feature-cell shift in the interior.
        self.assertTrue(torch.allclose(amplitude[..., 2:-2, 2:-3],
                                       shifted_amp[..., 2:-2, 3:-2], atol=1e-5))
        self.assertTrue(torch.allclose(phase[..., 2:-2, 2:-3],
                                       shifted_phase[..., 2:-2, 3:-2], atol=1e-5))
        zero_amp, zero_phase = local_spectrum(torch.zeros_like(image))
        self.assertEqual(float(zero_amp.abs().max()), 0.0)
        self.assertEqual(float(zero_phase.abs().max()), 0.0)

    def test_four_modes_share_coarse_geometry_and_train(self):
        config = OmegaConf.merge(
            OmegaConf.load("res/configs/registration.yaml"),
            OmegaConf.load("res/configs/stage0_spatial_frequency.yaml"))
        config.encoder.base_channels = 8
        config.encoder.out_channels = 16
        config.encoder.blocks_per_scale = 1
        config.coarse_transformer.num_layers = 1
        config.global_matcher.max_tokens = 80
        torch.manual_seed(7)
        ir = torch.rand(1, 1, 64, 80)
        vi = torch.rand(1, 3, 64, 80)
        for mode in ("spatial", "amplitude", "phase", "fused"):
            with self.subTest(mode=mode):
                config.spatial_frequency.mode = mode
                model = build_global_registration(config)
                result = model(ir, vi)
                self.assertEqual(result.match.matching_probability.shape,
                                 (1, 80, 80))
                self.assertEqual(result.coarse_flow.shape, (1, 2, 64, 80))
                self.assertIsNone(result.local_match)
                self.assertIsNotNone(result.affine_yx)
                self.assertTrue(torch.isfinite(result.coarse_flow).all())
                loss = (result.coarse_flow - 1).square().mean()
                loss = loss + result.match.matching_probability[:, 0, 1].mean()
                loss.backward()
                if mode in ("spatial", "fused"):
                    self.assertIsNotNone(model.encoder.stage_8[0][0].weight.grad)
                if mode == "amplitude":
                    self.assertIsNotNone(model.fusion.amplitude_encoder[0][0].weight.grad)
                if mode == "phase":
                    self.assertIsNotNone(model.fusion.phase_encoder[0][0].weight.grad)
                if mode == "fused":
                    self.assertIsNotNone(model.fusion.fusion_gain.grad)
                    self.assertIsNotNone(model.fusion.amplitude_encoder[0][0].weight.grad)
                    self.assertIsNotNone(model.fusion.phase_encoder[0][0].weight.grad)
                picture = _correspondence_image(
                    ir[0], vi[0],
                    result.match.matching_probability[0].detach(),
                    torch.zeros(2, 64, 80), (8, 10))
                self.assertEqual(picture.size, (160, 64))
                if mode == "fused":
                    batch = {"ir": ir, "vi": vi,
                             "gt_flow": torch.zeros(1, 2, 64, 80),
                             "valid_mask": torch.ones(1, 1, 64, 80),
                             "gt_h": torch.eye(3).unsqueeze(0)}
                    report = evaluate(model.eval(), [batch], torch.device("cpu"))
                    self.assertEqual(report["coarse_match_candidates"], 80)
                    self.assertEqual(report["representation"], "fused")
                    self.assertIn("affine_weight_effective_queries_ratio", report)
                    self.assertIn("affine_weighted_raw_epe_px", report)

    def test_common_branches_start_from_the_same_weights(self):
        config = OmegaConf.merge(
            OmegaConf.load("res/configs/registration.yaml"),
            OmegaConf.load("res/configs/stage0_spatial_frequency.yaml"))
        torch.manual_seed(21)
        config.spatial_frequency.mode = "spatial"
        spatial = build_global_registration(config)
        torch.manual_seed(21)
        config.spatial_frequency.mode = "fused"
        fused = build_global_registration(config)
        for name, value in spatial.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, fused.encoder.state_dict()[name]))
        for name, value in spatial.coarse_transformer.state_dict().items():
            self.assertTrue(torch.equal(
                value, fused.coarse_transformer.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
