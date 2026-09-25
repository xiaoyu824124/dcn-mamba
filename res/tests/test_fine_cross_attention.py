"""Geometry, warm-start and gradient checks for local cross-modal context."""

import unittest

import torch
from omegaconf import OmegaConf

from res.fine_cross_attention import FineCrossModalAttention
from res.local_matcher import LocalMatcher, local_matching_loss
from res.model_factory import build_global_registration


class FineCrossModalAttentionTest(unittest.TestCase):
    def test_zero_gain_preserves_old_checkpoint_predictions(self):
        torch.manual_seed(29)
        config = OmegaConf.merge(OmegaConf.load("res/configs/registration.yaml"),
                                 OmegaConf.load("res/configs/stage2_fine_interaction.yaml"))
        original = build_global_registration(config).eval()
        extended = build_global_registration(OmegaConf.merge(
            config, OmegaConf.load("res/configs/ab_fine_cross_attention.yaml"))).eval()
        extended.load_state_dict(original.state_dict(), strict=False)
        ir = torch.rand(1, 1, 32, 40)
        vi = torch.rand(1, 3, 32, 40)
        with torch.no_grad():
            before = original(ir, vi)
            after = extended(ir, vi)
        self.assertTrue(torch.equal(before.coarse_flow, after.coarse_flow))
        self.assertTrue(torch.equal(before.final_flow, after.final_flow))
        self.assertTrue(torch.equal(before.local_match.probability,
                                    after.local_match.probability))

    def test_local_cross_modal_parameters_receive_gradient(self):
        torch.manual_seed(30)
        module = FineCrossModalAttention(8, hidden_channels=4, window_size=3)
        module.gain.data.fill_(0.1)
        moving = torch.randn(1, 8, 8, 10)
        reference = torch.randn(1, 8, 8, 10)
        flow = torch.zeros(1, 2, 8, 10)
        flow[:, 0] = 1
        conditioned_ir, conditioned_vi = module(moving, reference, flow)
        self.assertTrue(torch.equal(conditioned_ir, moving))
        self.assertEqual(conditioned_vi.shape, reference.shape)
        self.assertTrue(torch.isfinite(conditioned_vi).all())
        match = LocalMatcher(radius=2, temperature=0.1)(conditioned_ir,
                                                          conditioned_vi, flow)
        local_matching_loss(match, torch.zeros(1, 2, 32, 40)).backward()
        self.assertGreater(float(module.gain.grad.abs()), 0.0)
        self.assertGreater(float(module.query.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.key.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.value.weight.grad.abs().sum()), 0.0)

    def test_out_of_image_coarse_flow_adds_no_cross_modal_message(self):
        module = FineCrossModalAttention(8, hidden_channels=4, window_size=3)
        module.gain.data.fill_(0.1)
        moving = torch.randn(1, 8, 8, 10)
        reference = torch.randn(1, 8, 8, 10)
        flow = torch.full((1, 2, 8, 10), 100.0)
        _, conditioned_vi = module(moving, reference, flow)
        self.assertTrue(torch.equal(conditioned_vi, reference))


if __name__ == "__main__":
    unittest.main()
