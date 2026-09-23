"""Tests for global correspondence and its ``[dy, dx]`` flow convention.

Run from the repository root:
    python -B -m unittest res.tests.test_global_matcher -v
"""

import unittest

import torch

from res.global_matcher import GlobalMatcher


def translated_unique_features(height: int, width: int, dy: int, dx: int):
    """Make exact one-hot IR/VI matches with known reference->moving flow.

    For every valid fixed/VI coordinate p, its unique feature is put at the
    moving/IR coordinate q=p+[dy,dx].  Thus the expected backward warp flow is
    exactly ``[dy,dx]`` on the valid interior region.
    """
    tokens = height * width
    vi = torch.zeros(1, tokens, height, width)
    ir = torch.zeros_like(vi)
    valid = torch.zeros(height, width, dtype=torch.bool)
    for y in range(height):
        for x in range(width):
            qy, qx = y + dy, x + dx
            if 0 <= qy < height and 0 <= qx < width:
                index = y * width + x
                vi[0, index, y, x] = 1.0
                ir[0, index, qy, qx] = 1.0
                valid[y, x] = True
    return ir, vi, valid


class GlobalMatcherTest(unittest.TestCase):
    def test_known_large_translation_and_flow_direction(self):
        # At a 1/8 grid, (dy,dx)=(5,-4) represents a 40 px / -32 px input shift.
        dy, dx = 5, -4
        ir, vi, valid = translated_unique_features(16, 17, dy, dx)
        matcher = GlobalMatcher(temperature=0.01)
        output = matcher(ir, vi)
        predicted = output.coarse_flow[0, :, valid]
        expected = torch.tensor([[dy], [dx]], dtype=predicted.dtype)
        self.assertTrue(torch.allclose(predicted, expected, atol=1e-3))
        self.assertGreater(float(output.confidence[0, 0, valid].mean()), 0.99)

    def test_shapes_probability_and_gradient(self):
        ir = torch.randn(2, 12, 8, 10, requires_grad=True)
        vi = torch.randn(2, 12, 8, 10, requires_grad=True)
        output = GlobalMatcher(temperature=0.1, learnable_temperature=True)(ir, vi)
        self.assertEqual(tuple(output.coarse_flow.shape), (2, 2, 8, 10))
        self.assertEqual(tuple(output.confidence.shape), (2, 1, 8, 10))
        self.assertEqual(tuple(output.matching_probability.shape), (2, 80, 80))
        self.assertTrue(torch.allclose(
            output.matching_probability.sum(dim=-1), torch.ones(2, 80), atol=2e-5))
        output.coarse_flow.square().mean().backward()
        self.assertIsNotNone(ir.grad)
        self.assertIsNotNone(vi.grad)
        self.assertTrue(torch.isfinite(ir.grad).all())
        self.assertTrue(torch.isfinite(vi.grad).all())

    def test_dual_softmax_keeps_rows_normalised_and_gradients_finite(self):
        ir = torch.randn(2, 12, 8, 10, requires_grad=True)
        vi = torch.randn(2, 12, 8, 10, requires_grad=True)
        output = GlobalMatcher(temperature=0.1, dual_softmax=True)(ir, vi)
        self.assertTrue(torch.allclose(
            output.matching_probability.sum(dim=-1), torch.ones(2, 80), atol=2e-5))
        output.coarse_flow.square().mean().backward()
        self.assertTrue(torch.isfinite(ir.grad).all())
        self.assertTrue(torch.isfinite(vi.grad).all())

    def test_key_log_scale_is_neutral_at_zero_and_learnable(self):
        ir = torch.randn(1, 12, 8, 10)
        vi = torch.randn(1, 12, 8, 10)
        plain = GlobalMatcher(temperature=0.1)
        scaled = GlobalMatcher(temperature=0.1, key_log_scale=True)
        self.assertNotIn("log_scale", dict(plain.named_parameters()))
        self.assertIn("log_scale", dict(scaled.named_parameters()))
        # exp(0) = 1, so a fresh key scale must reproduce the plain matcher.
        self.assertTrue(torch.allclose(plain(ir, vi).matching_probability,
                                       scaled(ir, vi).matching_probability, atol=1e-6))

    def test_token_guard(self):
        matcher = GlobalMatcher(max_tokens=15)
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            matcher(torch.rand(1, 4, 4, 4), torch.rand(1, 4, 4, 4))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda(self):
        matcher = GlobalMatcher().cuda()
        output = matcher(torch.rand(1, 8, 8, 8, device="cuda"),
                         torch.rand(1, 8, 8, 8, device="cuda"))
        self.assertTrue(output.coarse_flow.is_cuda)
        self.assertTrue(torch.isfinite(output.coarse_flow).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
