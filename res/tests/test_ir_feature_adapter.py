"""IR-only calibration preserves an existing model and receives supervision."""

import unittest

import torch
from omegaconf import OmegaConf

from res.ir_feature_adapter import IRFeatureAdapter
from res.local_matcher import local_matching_loss
from res.matching import coarse_matching_loss
from res.model_factory import build_global_registration
from res.train_vtmot import freeze_except_ir_adapter_parameters


class IRFeatureAdapterTest(unittest.TestCase):
    @staticmethod
    def _models():
        config = OmegaConf.merge(OmegaConf.load("res/configs/registration.yaml"),
                                 OmegaConf.load("res/configs/stage2_fine_interaction.yaml"))
        original = build_global_registration(config).eval()
        extended = build_global_registration(OmegaConf.merge(
            config, OmegaConf.load("res/configs/ab_ir_feature_adapter.yaml"))).eval()
        report = extended.load_state_dict(original.state_dict(), strict=False)
        assert not report.unexpected_keys
        assert all(name.startswith("ir_feature_adapter.") for name in report.missing_keys)
        return original, extended

    def test_adapter_preserves_both_scales_at_initialisation(self):
        adapter = IRFeatureAdapter(12, 16, hidden_channels=8)
        features = {"1/2": torch.randn(1, 6, 16, 20),
                    "1/4": torch.randn(1, 12, 8, 10),
                    "1/8": torch.randn(1, 16, 4, 5)}
        output = adapter(features)
        for scale in features:
            self.assertTrue(torch.equal(output[scale], features[scale]))

    def test_old_checkpoint_predictions_are_identical_at_step_zero(self):
        torch.manual_seed(31)
        original, extended = self._models()
        ir, vi = torch.rand(1, 1, 32, 40), torch.rand(1, 3, 32, 40)
        with torch.no_grad():
            before, after = original(ir, vi), extended(ir, vi)
        self.assertTrue(torch.equal(before.coarse_flow, after.coarse_flow))
        self.assertTrue(torch.equal(before.final_flow, after.final_flow))
        self.assertTrue(torch.equal(before.local_match.probability,
                                    after.local_match.probability))

    def test_only_ir_adapter_receives_global_and_local_match_gradients(self):
        torch.manual_seed(32)
        _, model = self._models()
        freeze_except_ir_adapter_parameters(model)
        trainable = {name for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("ir_feature_adapter.")
                            for name in trainable))
        output = model(torch.rand(1, 1, 32, 40), torch.rand(1, 3, 32, 40))
        gt_flow = torch.zeros_like(output.coarse_flow)
        global_loss = coarse_matching_loss(
            output.match.matching_probability,
            tuple(output.match.coarse_flow.shape[-2:]), gt_flow)
        local_loss = local_matching_loss(output.local_match, gt_flow)
        (global_loss + local_loss).backward()
        for scale in ("adapter_4", "adapter_8"):
            gradient = getattr(model.ir_feature_adapter, scale).network[-1].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.encoder.parameters()))

    def test_freeze_requires_enabled_adapter(self):
        model = build_global_registration(OmegaConf.load("res/configs/registration.yaml"))
        with self.assertRaisesRegex(ValueError, "IR feature adapter"):
            freeze_except_ir_adapter_parameters(model)


if __name__ == "__main__":
    unittest.main()
