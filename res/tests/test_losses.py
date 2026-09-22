"""Tests for supervised and structural registration losses.

Run from the repository root:
    python -B -m unittest res.tests.test_losses -v
"""

import unittest

import torch

from res.losses import RegistrationLoss


class RegistrationLossTest(unittest.TestCase):
    def setUp(self):
        self.loss = RegistrationLoss({"flow": 1.0, "mind": 0.5,
                                      "edge": 0.25, "smooth": 0.05, "affine": 1.0})

    def test_all_terms_are_finite_and_differentiable(self):
        aligned = torch.rand(2, 1, 32, 40, requires_grad=True)
        visible = torch.rand(2, 3, 32, 40, requires_grad=True)
        flow = torch.randn(2, 2, 32, 40, requires_grad=True)
        gt = torch.randn(2, 2, 32, 40)
        output = self.loss(aligned_ir=aligned, visible=visible,
                           coarse_flow=flow, gt_flow=gt)
        for value in output.as_dict().values():
            self.assertTrue(torch.isfinite(value))
        output.total.backward()
        self.assertTrue(torch.isfinite(aligned.grad).all())
        self.assertTrue(torch.isfinite(visible.grad).all())
        self.assertTrue(torch.isfinite(flow.grad).all())

    def test_gt_matching_flow_has_only_charbonnier_floor(self):
        aligned, visible = torch.rand(1, 1, 24, 24), torch.rand(1, 3, 24, 24)
        flow = torch.randn(1, 2, 24, 24)
        output = self.loss(aligned_ir=aligned, visible=visible,
                           coarse_flow=flow, gt_flow=flow.clone())
        self.assertLess(float(output.flow), 1.1e-3)

    def test_no_gt_disables_only_flow_term(self):
        output = self.loss(aligned_ir=torch.rand(1, 1, 24, 24),
                           visible=torch.rand(1, 3, 24, 24),
                           coarse_flow=torch.zeros(1, 2, 24, 24))
        self.assertEqual(float(output.flow), 0.0)
        self.assertGreater(float(output.mind + output.edge), 0.0)

    def test_affine_gt_has_small_loss_for_matching_parameters(self):
        height, width = 32, 40
        gt_h = torch.tensor([[[1.0, 0.0, 3.0],
                              [0.0, 1.0, -2.0],
                              [0.0, 0.0, 1.0]]])
        from res.affine import homography_to_normalised_affine_yx
        affine = homography_to_normalised_affine_yx(gt_h, (height, width), (4, 5))
        output = self.loss(aligned_ir=torch.rand(1, 1, height, width),
                           visible=torch.rand(1, 3, height, width),
                           coarse_flow=torch.zeros(1, 2, height, width),
                           predicted_affine_yx=affine, gt_h=gt_h)
        self.assertLess(float(output.affine), 1.1e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda(self):
        loss = self.loss.cuda()
        output = loss(aligned_ir=torch.rand(1, 1, 24, 24, device="cuda"),
                      visible=torch.rand(1, 3, 24, 24, device="cuda"),
                      coarse_flow=torch.zeros(1, 2, 24, 24, device="cuda"))
        self.assertTrue(output.total.is_cuda)
        self.assertTrue(torch.isfinite(output.total))


if __name__ == "__main__":
    unittest.main(verbosity=2)
