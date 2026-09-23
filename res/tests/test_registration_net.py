"""Integration tests for the MIND -> encoder -> global matcher -> warp chain.

Run from the repository root:
    python -B -m unittest res.tests.test_registration_net -v
"""

import unittest

import torch

from res.encoder import MINDFeatureEncoder
from res.global_matcher import GlobalMatcher
from res.mind import MINDDescriptor
from res.local_matcher import LocalMatcher
from res.registration_net import MINDGlobalRegistration


def small_model() -> MINDGlobalRegistration:
    return MINDGlobalRegistration(
        mind=MINDDescriptor(),
        encoder=MINDFeatureEncoder(
            in_channels=8, base_channels=8, out_channels=12, blocks_per_scale=1),
        matcher=GlobalMatcher(temperature=0.1),
        local_matcher=LocalMatcher(radius=2, candidate_chunk=5),
    )


class MINDGlobalRegistrationTest(unittest.TestCase):
    def test_output_shapes_finiteness_and_gradient(self):
        torch.manual_seed(17)
        model = small_model()
        ir = torch.rand(2, 1, 64, 80, requires_grad=True)
        vi = torch.rand(2, 3, 64, 80, requires_grad=True)
        output = model(ir, vi)
        self.assertEqual(tuple(output.coarse_aligned_ir.shape), (2, 1, 64, 80))
        self.assertEqual(tuple(output.coarse_flow.shape), (2, 2, 64, 80))
        self.assertEqual(tuple(output.final_flow.shape), (2, 2, 64, 80))
        self.assertEqual(tuple(output.final_aligned_ir.shape), (2, 1, 64, 80))
        self.assertEqual(tuple(output.local_match.probability.shape), (2, 25, 16, 20))
        self.assertEqual(tuple(output.confidence_1_8.shape), (2, 1, 8, 10))
        self.assertEqual(tuple(output.affine_yx.shape), (2, 3, 2))
        for tensor in (output.coarse_aligned_ir, output.coarse_flow,
                       output.confidence_1_8, output.affine_yx):
            self.assertTrue(torch.isfinite(tensor).all())
        output.final_aligned_ir.mean().backward()
        self.assertIsNotNone(ir.grad)
        self.assertIsNotNone(vi.grad)
        self.assertTrue(torch.isfinite(ir.grad).all())
        self.assertTrue(torch.isfinite(vi.grad).all())

    def test_non_multiple_of_eight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            small_model()(torch.rand(1, 1, 65, 64), torch.rand(1, 3, 65, 64))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda(self):
        model = small_model().cuda()
        output = model(torch.rand(1, 1, 64, 64, device="cuda"),
                       torch.rand(1, 3, 64, 64, device="cuda"))
        self.assertTrue(output.coarse_aligned_ir.is_cuda)
        self.assertTrue(torch.isfinite(output.coarse_flow).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
