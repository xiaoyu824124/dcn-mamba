"""Minimal CPU/CUDA tests for :mod:`res.mind`.

Run from the repository root:
    python -B -m unittest res.tests.test_mind -v
"""

import unittest

import torch

from res.mind import MINDDescriptor, paired_mind, rgb_to_gray


class MINDDescriptorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.mind = MINDDescriptor(patch_size=3, radius=1)

    def test_shape_finite_and_l2_normalised(self):
        image = torch.rand(2, 1, 33, 49, requires_grad=True)
        descriptor = self.mind(image)
        self.assertEqual(descriptor.shape, (2, 8, 33, 49))
        self.assertTrue(torch.isfinite(descriptor).all())
        self.assertTrue(torch.allclose(
            descriptor.norm(dim=1), torch.ones(2, 33, 49), atol=2e-5))
        descriptor.mean().backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_visible_is_grayscale_then_same_shape_as_ir(self):
        ir = torch.rand(3, 1, 32, 48)
        vi = torch.rand(3, 3, 32, 48)
        d_ir, d_vi = paired_mind(ir, vi, self.mind)
        self.assertEqual(d_ir.shape, d_vi.shape)
        self.assertEqual(rgb_to_gray(vi).shape, (3, 1, 32, 48))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_batch_execution(self):
        mind = MINDDescriptor().cuda()
        image = torch.rand(2, 1, 32, 32, device="cuda")
        descriptor = mind(image)
        self.assertTrue(descriptor.is_cuda)
        self.assertTrue(torch.isfinite(descriptor).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
