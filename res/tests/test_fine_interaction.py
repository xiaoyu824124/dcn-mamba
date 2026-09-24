"""Shared cross-scale 1/4 feature adapter checks."""

import unittest

import torch
from omegaconf import OmegaConf

from res.fine_interaction import FineScaleInteraction
from res.model_factory import build_global_registration


class FineScaleInteractionTest(unittest.TestCase):
    def test_identity_warm_start_and_trainable_gain(self):
        torch.manual_seed(11)
        adapter = FineScaleInteraction(16, 24)
        fine_ir = torch.randn(1, 16, 8, 10, requires_grad=True)
        fine_vi = torch.randn(1, 16, 8, 10, requires_grad=True)
        coarse_ir = torch.randn(1, 24, 4, 5, requires_grad=True)
        coarse_vi = torch.randn(1, 24, 4, 5, requires_grad=True)
        adapted_ir, adapted_vi = adapter(fine_ir, fine_vi, coarse_ir, coarse_vi)
        self.assertTrue(torch.equal(adapted_ir, fine_ir))
        self.assertTrue(torch.equal(adapted_vi, fine_vi))
        (adapted_ir.square().mean() + adapted_vi.square().mean()).backward()
        self.assertIsNotNone(adapter.gain.grad)
        self.assertTrue(torch.isfinite(adapter.gain.grad))
        with torch.no_grad():
            adapter.gain.fill_(0.2)
        shifted_ir, shifted_vi = adapter(fine_ir.detach(), fine_vi.detach(),
                                        coarse_ir.detach(), coarse_vi.detach())
        self.assertGreater(float((shifted_ir - fine_ir.detach()).abs().mean()), 0.0)
        self.assertGreater(float((shifted_vi - fine_vi.detach()).abs().mean()), 0.0)

    def test_factory_inserts_adapter_before_unchanged_local_matcher(self):
        config = OmegaConf.merge(OmegaConf.load("res/configs/registration.yaml"),
                                 OmegaConf.load("res/configs/stage2_fine_interaction.yaml"))
        model = build_global_registration(config)
        self.assertIsNotNone(model.fine_interaction)
        self.assertIsNotNone(model.local_matcher)
        output = model(torch.rand(1, 1, 32, 40), torch.rand(1, 3, 32, 40))
        self.assertEqual(tuple(output.final_flow.shape), (1, 2, 32, 40))
        self.assertTrue(torch.isfinite(output.final_flow).all())

    def test_adapter_requires_local_matcher(self):
        config = OmegaConf.merge(OmegaConf.load("res/configs/registration.yaml"),
                                 OmegaConf.load("res/configs/stage0_coarse.yaml"),
                                 OmegaConf.load("res/configs/stage2_fine_interaction.yaml"))
        with self.assertRaisesRegex(ValueError, "local_matcher"):
            build_global_registration(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
