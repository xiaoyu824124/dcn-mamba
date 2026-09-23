"""Small checks for the CRFT-style 1/8 linear SA-CA stage."""

import unittest

import torch
from omegaconf import OmegaConf

from res.coarse_transformer import CoarseSACATransformer
from res.model_factory import build_global_registration


class CoarseTransformerTest(unittest.TestCase):
    def test_identity_warm_start_and_gradients(self):
        torch.manual_seed(4)
        block = CoarseSACATransformer(channels=16, num_layers=2, num_heads=4)
        ir = torch.randn(2, 16, 8, 10, requires_grad=True)
        vi = torch.randn(2, 16, 8, 10, requires_grad=True)
        out_ir, out_vi = block(ir, vi)
        self.assertTrue(torch.equal(out_ir, ir))
        self.assertTrue(torch.equal(out_vi, vi))
        (out_ir.square().mean() + out_vi.square().mean()).backward()
        self.assertTrue(torch.isfinite(ir.grad).all())
        self.assertTrue(torch.isfinite(vi.grad).all())
        self.assertIsNotNone(block.blocks[0].self_gain.grad)
        self.assertTrue(torch.isfinite(block.blocks[0].cross_gain.grad))
        with torch.no_grad():
            block.blocks[0].cross_gain.fill_(0.2)
        shifted_ir, shifted_vi = block(ir.detach(), vi.detach())
        self.assertGreater(float((shifted_ir - ir.detach()).abs().mean()), 0)
        self.assertGreater(float((shifted_vi - vi.detach()).abs().mean()), 0)

    def test_factory_overlay_keeps_global_and_disables_local(self):
        config = OmegaConf.merge(
            OmegaConf.load("res/configs/registration.yaml"),
            OmegaConf.load("res/configs/stage1_saca.yaml"))
        model = build_global_registration(config)
        self.assertIsNotNone(model.coarse_transformer)
        self.assertIsNone(model.local_matcher)
        output = model(torch.rand(1, 1, 32, 40), torch.rand(1, 3, 32, 40))
        self.assertEqual(tuple(output.coarse_flow.shape), (1, 2, 32, 40))
        self.assertIsNone(output.final_flow)
        self.assertTrue(torch.isfinite(output.coarse_flow).all())

    def test_stage0_is_coarse_only(self):
        config = OmegaConf.merge(
            OmegaConf.load("res/configs/registration.yaml"),
            OmegaConf.load("res/configs/stage0_coarse.yaml"))
        model = build_global_registration(config)
        self.assertIsNone(model.coarse_transformer)
        self.assertIsNone(model.local_matcher)
        self.assertEqual(float(config.loss.weights.local), 0.0)

    def test_invalid_channels(self):
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            CoarseSACATransformer(channels=10, num_heads=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
