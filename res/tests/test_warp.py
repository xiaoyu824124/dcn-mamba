"""Direction and resize tests for the shared pixel-flow warp utility.

Run from the repository root:
    python -B -m unittest res.tests.test_warp -v
"""

import unittest

import torch

from res.warp import resize_flow, warp


class WarpTest(unittest.TestCase):
    def test_identity_flow_is_exact_identity(self):
        image = torch.randn(2, 3, 15, 19)
        result = warp(image, torch.zeros(2, 2, 15, 19))
        # A zero flow is exact in float64 (~1e-13).  In float32 the
        # pixel -> [-1,1] -> pixel round trip inside grid_sample leaves about
        # 1e-6 here, 5e-5 at 160x160 and 2e-4 at 480x640 -- four orders of
        # magnitude below one pixel, so it cannot bias a pixel-unit EPE.
        # The tolerance sits above that noise floor instead of exactly on it.
        self.assertTrue(torch.allclose(result, image, atol=1e-5))

    def test_dy_dx_direction_on_inner_region(self):
        height, width = 13, 17
        # A coordinate-coded source makes a wrong x/y order immediately visible.
        y = torch.arange(height).view(1, 1, height, 1).float()
        x = torch.arange(width).view(1, 1, 1, width).float()
        source = 100.0 * y + x
        dy, dx = 2.0, -3.0
        flow = torch.empty(1, 2, height, width)
        flow[:, 0] = dy
        flow[:, 1] = dx
        result = warp(source, flow)
        # Exclude locations whose source coordinate falls outside the image.
        expected = source[..., int(dy):, :width + int(dx)]
        actual = result[..., :height - int(dy), -int(dx):]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_resize_scales_pixel_displacement(self):
        flow = torch.zeros(1, 2, 5, 7)
        flow[:, 0] = 2.0
        flow[:, 1] = -3.0
        resized = resize_flow(flow, (9, 13))
        # align_corners=True: y scale=(9-1)/(5-1)=2, x scale=(13-1)/(7-1)=2.
        self.assertTrue(torch.allclose(resized[:, 0], torch.full((1, 9, 13), 4.0)))
        self.assertTrue(torch.allclose(resized[:, 1], torch.full((1, 9, 13), -6.0)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda(self):
        source = torch.rand(2, 1, 16, 16, device="cuda")
        result = warp(source, torch.zeros(2, 2, 16, 16, device="cuda"))
        self.assertTrue(result.is_cuda)
        # CUDA grid_sample may differ from a direct copy by a few fp32 ulps.
        self.assertTrue(torch.allclose(result, source, atol=2e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
