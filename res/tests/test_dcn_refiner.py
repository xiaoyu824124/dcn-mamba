"""Tests for local multi-scale DCN refinement.

Run from the repository root:
    python -B -m unittest res.tests.test_dcn_refiner -v
"""

import unittest

import torch

from res.encoder import MINDFeatureEncoder
from res.global_matcher import GlobalMatcher
from res.mind import MINDDescriptor
from res.registration_net import MINDDCNRegistration, MINDGlobalRegistration


def model(with_flow_head=False):
    coarse = MINDGlobalRegistration(
        mind=MINDDescriptor(),
        encoder=MINDFeatureEncoder(8, base_channels=8, out_channels=12,
                                   blocks_per_scale=1),
        matcher=GlobalMatcher(temperature=0.1),
    )
    return MINDDCNRegistration(coarse, use_residual_flow_head=with_flow_head)


class MultiScaleDCNRefinerTest(unittest.TestCase):
    def test_feature_refinement_shapes_offsets_and_gradients(self):
        torch.manual_seed(23)
        net = model()
        ir = torch.rand(1, 1, 64, 80, requires_grad=True)
        vi = torch.rand(1, 3, 64, 80, requires_grad=True)
        coarse, refined = net(ir, vi)
        self.assertEqual(tuple(refined.refined_aligned_ir.shape), (1, 1, 64, 80))
        self.assertIsNone(refined.residual_flow)
        self.assertIsNone(refined.final_flow)
        self.assertEqual(tuple(refined.refined_features["1/8"].shape), (1, 12, 8, 10))
        self.assertEqual(tuple(refined.refined_features["1/4"].shape), (1, 16, 16, 20))
        self.assertEqual(tuple(refined.refined_features["1/2"].shape), (1, 8, 32, 40))
        for scale, size in (("1/8", (8, 10)), ("1/4", (16, 20)), ("1/2", (32, 40))):
            self.assertEqual(tuple(refined.dcn_offsets[scale].shape), (1, 18, *size))
            self.assertEqual(tuple(refined.dcn_masks[scale].shape), (1, 9, *size))
            self.assertTrue(torch.isfinite(refined.dcn_offsets[scale]).all())
        # Decoder starts at zero, so DCN cannot perturb the physical coarse IR
        # before supervised training explicitly learns to do so.
        self.assertTrue(torch.allclose(refined.refined_aligned_ir,
                                       coarse.coarse_aligned_ir, atol=2e-6))
        refined.refined_aligned_ir.mean().backward()
        self.assertTrue(torch.isfinite(ir.grad).all())
        self.assertTrue(torch.isfinite(vi.grad).all())

    def test_optional_residual_flow_is_explicit_and_zero_initially(self):
        net = model(with_flow_head=True)
        ir, vi = torch.rand(1, 1, 64, 64), torch.rand(1, 3, 64, 64)
        coarse, refined = net(ir, vi)
        self.assertEqual(tuple(refined.residual_flow.shape), (1, 2, 64, 64))
        self.assertEqual(tuple(refined.final_flow.shape), (1, 2, 64, 64))
        self.assertTrue(torch.allclose(refined.residual_flow,
                                       torch.zeros_like(refined.residual_flow)))
        self.assertTrue(torch.allclose(refined.final_flow, coarse.coarse_flow))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda(self):
        net = model().cuda()
        _, refined = net(torch.rand(1, 1, 64, 64, device="cuda"),
                         torch.rand(1, 3, 64, 64, device="cuda"))
        self.assertTrue(refined.refined_aligned_ir.is_cuda)
        self.assertTrue(torch.isfinite(refined.refined_features["1/2"]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
