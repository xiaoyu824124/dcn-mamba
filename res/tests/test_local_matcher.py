"""Geometry and training checks for prediction-centred local matching."""

import unittest

import torch

from res.local_matcher import LocalMatcher, local_matching_loss


class LocalMatcherTest(unittest.TestCase):
    def test_known_shift_and_local_supervision(self):
        torch.manual_seed(8)
        fixed = torch.randn(1, 32, 16, 20, requires_grad=True)
        moving = torch.roll(fixed.detach(), shifts=(2, -1), dims=(-2, -1)).requires_grad_()
        coarse = torch.zeros(1, 2, 16, 20)
        matcher = LocalMatcher(radius=3, temperature=0.02, candidate_chunk=5)
        output = matcher(moving, fixed, coarse)
        central = output.matched_residual[0, :, 4:-4, 4:-4]
        self.assertLess(float((central[0] - 2).abs().mean()), 0.1)
        self.assertLess(float((central[1] + 1).abs().mean()), 0.1)
        self.assertTrue(torch.allclose(output.refined_flow, coarse))
        gt = torch.zeros(1, 2, 16 * 4, 20 * 4)
        gt[:, 0] = 8
        gt[:, 1] = -4
        loss = local_matching_loss(output, gt)
        self.assertTrue(torch.isfinite(loss))
        (loss + (output.refined_flow - gt[:, :, ::4, ::4] / 4).square().mean()).backward()
        self.assertTrue(torch.isfinite(fixed.grad).all())
        self.assertTrue(torch.isfinite(moving.grad).all())
        self.assertTrue(torch.isfinite(matcher.refinement[-1].weight.grad).all())

    def test_invalid_inputs(self):
        matcher = LocalMatcher(radius=2)
        with self.assertRaisesRegex(ValueError, "coarse_flow"):
            matcher(torch.rand(1, 4, 8, 8), torch.rand(1, 4, 8, 8),
                    torch.zeros(1, 2, 4, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
