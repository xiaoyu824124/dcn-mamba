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

    def test_soft_spatial_prior_suppresses_distant_ties_without_new_weights(self):
        # Every cosine score is identical. The optional prior should select
        # the same location while retaining a normalised all-pairs matrix.
        ir = torch.ones(1, 4, 8, 10)
        vi = ir.clone()
        baseline = GlobalMatcher(dual_softmax=True, affine_projection=False)
        prior = GlobalMatcher(dual_softmax=True, affine_projection=False,
                              spatial_prior_sigma=4.0)
        self.assertEqual(set(baseline.state_dict()), set(prior.state_dict()))
        output = prior(ir, vi)
        indices = output.matching_probability.argmax(dim=-1)
        self.assertTrue(torch.equal(indices[0], torch.arange(80)))
        self.assertTrue(torch.allclose(output.matching_probability.sum(-1),
                                       torch.ones(1, 80), atol=1e-6))
        self.assertGreater(float(output.matching_probability[0, 11, 11]),
                           float(output.matching_probability[0, 11, 79]))

    def test_spatial_prior_keeps_a_distinct_match_five_cells_away(self):
        ir, vi, valid = translated_unique_features(16, 17, 5, -4)
        matcher = GlobalMatcher(temperature=0.01, affine_projection=False,
                                spatial_prior_sigma=4.0)
        output = matcher(ir, vi)
        picked = output.matching_probability.argmax(dim=-1).reshape(16, 17)[valid]
        yy, xx = torch.meshgrid(torch.arange(16), torch.arange(17), indexing="ij")
        expected = ((yy + 5) * 17 + xx - 4)[valid]
        self.assertTrue(torch.equal(picked, expected))

    def test_spatial_prior_gradients_are_finite(self):
        ir = torch.randn(1, 8, 8, 10, requires_grad=True)
        vi = torch.randn(1, 8, 8, 10, requires_grad=True)
        output = GlobalMatcher(dual_softmax=True, spatial_prior_sigma=4.0)(ir, vi)
        output.coarse_flow.square().mean().backward()
        self.assertTrue(torch.isfinite(ir.grad).all())
        self.assertTrue(torch.isfinite(vi.grad).all())

    def test_wls_confidence_power_changes_only_affine_projection(self):
        torch.manual_seed(8)
        ir = torch.randn(1, 12, 8, 10)
        vi = torch.randn_like(ir)
        uniform = GlobalMatcher(temperature=0.12, dual_softmax=True,
                                affine_confidence_power=0.0)
        default = GlobalMatcher(temperature=0.12, dual_softmax=True)
        explicit = GlobalMatcher(temperature=0.12, dual_softmax=True,
                                 affine_confidence_power=4.0)
        self.assertEqual(set(uniform.state_dict()), set(default.state_dict()))
        unweighted, weighted, same = (matcher(ir, vi)
                                      for matcher in (uniform, default, explicit))
        self.assertTrue(torch.equal(weighted.coarse_flow, same.coarse_flow))
        self.assertTrue(torch.equal(unweighted.matching_probability,
                                    weighted.matching_probability))
        self.assertTrue(torch.equal(unweighted.raw_flow, weighted.raw_flow))
        self.assertGreater(float((unweighted.coarse_flow - weighted.coarse_flow).abs().max()),
                           0.1)
        with self.assertRaisesRegex(ValueError, "affine_confidence_power"):
            GlobalMatcher(affine_confidence_power=-1.0)

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
