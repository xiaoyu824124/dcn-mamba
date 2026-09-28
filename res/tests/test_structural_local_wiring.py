"""Wiring checks for the structural-prior 1/4 local stage.

These encode the step-1 acceptance conditions directly: exposing the shared
encoder's 1/4 scale and attaching a local matcher must not change the coarse
field, must not change the final flow while the refinement head is still
zero-initialised, and the diagnostic centre override must touch only the local
result.
"""

import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from res.checkpoint import load_registration_state
from res.evaluate_vtmot import evaluate
from res.local_matcher import local_matching_diagnostics
from res.model_factory import build_global_registration
from res.warp import upsample_feature_flow


def structural_config(*, local_enabled: bool, radius: int = 2):
    config = OmegaConf.merge(
        OmegaConf.load("res/configs/registration.yaml"),
        OmegaConf.load("res/configs/stage0_structural_prior.yaml"))
    config.structural_prior.base_channels = 8
    config.structural_prior.out_channels = 16
    config.structural_prior.blocks_per_scale = 1
    config.coarse_transformer.num_layers = 1
    config.global_matcher.max_tokens = 80
    config.loss.weights.local = 0.0
    if local_enabled:
        config = OmegaConf.merge(
            config, OmegaConf.load("res/configs/stage1_structural_local.yaml"))
        config.local_matcher.radius = radius
    return config


def shared_state(source: torch.nn.Module, target: torch.nn.Module) -> dict:
    """The tensors both models define, so only the coarse path is compared."""
    return {name: value for name, value in source.state_dict().items()
            if name in target.state_dict()}


class StructuralLocalWiringTest(unittest.TestCase):
    def test_stage_config_searches_32_pixels(self):
        overlay = OmegaConf.load("res/configs/stage1_structural_local.yaml")
        self.assertTrue(bool(overlay.local_matcher.enabled))
        # radius 8 cells at 1/4 resolution is +/-32 image pixels.
        self.assertEqual(int(overlay.local_matcher.radius) * 4, 32)
        self.assertEqual(float(overlay.loss.weights.local), 0.0)

    def test_encoder_returns_every_scale_and_keeps_the_coarse_pair(self):
        model = build_global_registration(structural_config(local_enabled=False))
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        scales_ir, scales_vi = model.encoder.encode_scales(
            torch.rand(1, 1, 64, 80), torch.rand(1, 1, 64, 80))
        self.assertEqual(set(scales_ir), {"1/2", "1/4", "1/8"})
        self.assertEqual(tuple(scales_ir["1/4"].shape[-2:]), (16, 20))
        self.assertEqual(tuple(scales_ir["1/8"].shape[-2:]), (8, 10))
        coarse_ir, coarse_vi = model.encoder(torch.rand(1, 1, 64, 80),
                                             torch.rand(1, 1, 64, 80))
        self.assertEqual(tuple(coarse_ir.shape[-2:]), (8, 10))
        self.assertEqual(coarse_ir.shape, coarse_vi.shape)
        model(ir, vi)

    def test_local_stage_leaves_the_coarse_field_byte_identical(self):
        torch.manual_seed(4)
        without = build_global_registration(structural_config(local_enabled=False))
        with_local = build_global_registration(structural_config(local_enabled=True))
        without.load_state_dict(shared_state(with_local, without), strict=True)
        self.assertIsNone(without.local_matcher)
        self.assertIsNotNone(with_local.local_matcher)
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        plain, refined = without(ir, vi), with_local(ir, vi)
        self.assertTrue(torch.equal(plain.coarse_flow, refined.coarse_flow))
        self.assertTrue(torch.equal(plain.coarse_aligned_ir, refined.coarse_aligned_ir))
        self.assertTrue(torch.equal(plain.match.matching_probability,
                                    refined.match.matching_probability))
        self.assertTrue(torch.equal(plain.affine_yx, refined.affine_yx))

    def test_feature_flow_round_trip_is_exact_for_an_affine_field(self):
        """The 1/4 centre is a resampling of the coarse field, so the round trip
        must reproduce a planar field exactly or every local window is offset."""
        height, width = 64, 80
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                              torch.arange(width, dtype=torch.float32), indexing="ij")
        planar = torch.stack((0.5 * y + 0.1 * x + 3.0, -0.2 * y + 0.3 * x - 2.0))
        planar = planar.unsqueeze(0)
        quarter = F.interpolate(planar, size=(16, 20), mode="bilinear",
                                align_corners=False) / 4.0
        restored = upsample_feature_flow(quarter, (height, width), (4.0, 4.0))
        error = (restored - planar).abs()
        # Bilinear resampling reproduces a planar field exactly, and the down and
        # up conventions agree, so everything except the outermost two rows and
        # columns round-trips to float precision.  There the second interpolation
        # asks for index -0.375 and align_corners=False clamps it to 0 instead of
        # extrapolating, which costs |grad flow| * 0.5 pixels.
        self.assertLess(float(error[..., 2:-2, 2:-2].max()), 1e-4)
        self.assertLess(float(error[..., 0, :].max()), 1.0)

    def test_untrained_head_reproduces_the_coarse_field(self):
        torch.manual_seed(5)
        model = build_global_registration(structural_config(local_enabled=True))
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        output = model(ir, vi)
        self.assertIsNotNone(output.local_match)
        self.assertEqual(tuple(output.final_flow.shape), (1, 2, 64, 80))
        # A zero-initialised last layer means the head starts by predicting no
        # correction, so the fine stage cannot silently move the frozen field.
        self.assertTrue(torch.equal(output.local_match.refined_flow,
                                    output.local_match.coarse_flow))
        self.assertTrue(torch.equal(
            output.local_match.residual_flow,
            torch.zeros_like(output.local_match.residual_flow)))
        # A random 6-DoF fit saturates at the border, and a saturated field is
        # piecewise planar, so the 1/4 round trip cannot reproduce it exactly.
        # Bound the drift against the field's own magnitude: the head's reach is
        # radius * 4 = 8 pixels, so this still catches a non-zeroed head.
        drift = float((output.final_flow - output.coarse_flow).abs().max())
        scale = max(1.0, float(output.coarse_flow.abs().max()))
        self.assertLessEqual(drift / scale, 0.05)
        self.assertTrue(torch.isfinite(output.final_flow).all())

    def test_local_centre_override_moves_only_the_local_window(self):
        torch.manual_seed(6)
        model = build_global_registration(structural_config(local_enabled=True))
        model.eval()
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        truth = torch.zeros(1, 2, 64, 80)
        with torch.no_grad():
            plain = model(ir, vi)
            centred = model(ir, vi, local_centre=truth)
        self.assertTrue(torch.equal(plain.coarse_flow, centred.coarse_flow))
        mask = torch.ones(1, 1, 64, 80)
        plain_report = local_matching_diagnostics(plain.local_match, truth, mask)
        report = local_matching_diagnostics(centred.local_match, truth, mask)
        # A truth-centred window always contains the truth, by construction, and
        # can only cover more than the coarse-centred one.
        self.assertEqual(report["local_window_coverage"], 1.0)
        self.assertGreaterEqual(report["local_window_coverage"],
                                plain_report["local_window_coverage"])
        with self.assertRaisesRegex(ValueError, "local_centre must be"):
            model(ir, vi, local_centre=torch.zeros(1, 2, 32, 40))

    def test_legacy_checkpoint_loads_and_unknown_keys_are_rejected(self):
        torch.manual_seed(7)
        model = build_global_registration(structural_config(local_enabled=True))
        legacy = {name: value for name, value in model.state_dict().items()
                  if not name.startswith("local_matcher.")}
        report = load_registration_state(model, legacy)
        self.assertEqual(report["kept"], len(legacy))
        self.assertTrue(all(name.startswith("local_matcher.")
                            for name in report["untrained"]))
        self.assertTrue(report["untrained"])
        with self.assertRaisesRegex(ValueError, "does not define these checkpoint keys"):
            load_registration_state(model, {**legacy, "bogus.weight": torch.zeros(1)})
        resized = dict(legacy)
        first = sorted(resized)[0]
        resized[first] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "different shape"):
            load_registration_state(model, resized)

    def test_evaluator_reports_truth_centred_local_diagnostics(self):
        torch.manual_seed(8)
        model = build_global_registration(structural_config(local_enabled=True))
        batch = {"ir": torch.rand(1, 1, 64, 80), "vi": torch.rand(1, 3, 64, 80),
                 "gt_flow": torch.zeros(1, 2, 64, 80),
                 "valid_mask": torch.ones(1, 1, 64, 80),
                 "gt_h": torch.eye(3).unsqueeze(0)}
        report = evaluate(model.eval(), [batch], torch.device("cpu"),
                          diagnose_local_centre=True)
        for name in ("local_window_coverage", "local_argmax_epe_px",
                     "local_oracle_epe_px"):
            self.assertIn(name, report)
            self.assertIn("truthcentre_" + name, report)
        self.assertEqual(report["truthcentre_local_window_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
