"""Tests for the descriptor peak-acceptance probe.

Run from the repository root:
    python -B -m unittest res.tests.test_mind_probe -v
"""

import unittest

import torch

from res.mind_probe import peak_report


def impulse_descriptors(height: int, width: int, shift_y: int = 0, shift_x: int = 0):
    """One-hot descriptors with a globally unique code at every pixel.

    The channel count equals the cell count, so each descriptor is unique and
    the true correspondence is an unambiguous peak -- the positive control.
    """
    channels = height * width
    ir = torch.zeros(1, channels, height, width)
    vi = torch.zeros(1, channels, height, width)
    for y in range(height):
        for x in range(width):
            code = y * width + x
            vi[0, code, y, x] = 1.0
            ty, tx = y + shift_y, x + shift_x
            if 0 <= ty < height and 0 <= tx < width:
                ir[0, code, ty, tx] = 1.0
    return ir, vi


class PeakReportTest(unittest.TestCase):
    def test_unique_codes_give_a_perfect_local_peak(self):
        ir, vi = impulse_descriptors(24, 24)
        report = peak_report(ir, vi, n_queries=32, global_samples=64,
                             window_px=8.0, radii_px=(2.0, 4.0),
                             argmax_radii_px=(2.0, 4.0))
        self.assertAlmostEqual(report["local_win_4px"], 1.0, places=5)
        self.assertAlmostEqual(report["argmax_err_4px"], 0.0, places=5)
        self.assertAlmostEqual(report["peak_4px"], 1.0, places=5)
        self.assertAlmostEqual(report["contrast"], 1.0, places=2)
        self.assertAlmostEqual(report["local_frac_beat"], 0.0, places=5)

    def test_constant_field_has_no_peak(self):
        ir = torch.ones(1, 576, 24, 24)
        vi = torch.ones(1, 576, 24, 24)
        report = peak_report(ir, vi, n_queries=32, global_samples=64,
                             window_px=8.0, radii_px=(2.0, 4.0),
                             argmax_radii_px=(2.0, 4.0))
        self.assertAlmostEqual(report["contrast"], 0.0, places=5)
        self.assertAlmostEqual(report["peak_4px"], 0.0, places=5)
        self.assertAlmostEqual(report["global_frac_beat"], 0.0, places=5)

    def test_known_shift_is_measured_in_image_pixels(self):
        shift = 3
        ir, vi = impulse_descriptors(12, 12, shift_x=shift)
        flow = torch.zeros(1, 2, 12, 12)
        flow[:, 1] = float(shift)
        report = peak_report(ir, vi, flow=flow, n_queries=32, global_samples=64)
        self.assertLess(report["argmax_err_4px"], 1e-4)
        self.assertAlmostEqual(report["local_win_4px"], 1.0, places=5)
        self.assertAlmostEqual(report["cos_gt"], 1.0, places=5)
        self.assertAlmostEqual(report["peak_4px"], 1.0, places=5)

    def test_stride_scales_the_reported_pixel_error(self):
        ir, vi = impulse_descriptors(8, 10, shift_x=1)
        flow = torch.zeros(1, 2, 8, 10)
        flow[:, 1] = 4.0                       # one cell at stride four
        report = peak_report(ir, vi, flow=flow, stride_yx=(4.0, 4.0),
                             n_queries=24, global_samples=64)
        self.assertLess(report["argmax_err_4px"], 1e-4)
        self.assertAlmostEqual(report["stride_yx"], 4.0, places=5)

    def test_missing_flow_uses_zero_correspondence(self):
        ir, vi = impulse_descriptors(8, 10)
        report = peak_report(ir, vi, flow=None, n_queries=24, global_samples=64)
        self.assertAlmostEqual(report["cos_gt"], 1.0, places=5)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "share the grid size"):
            peak_report(torch.rand(1, 8, 16, 20), torch.rand(1, 8, 16, 24))
        with self.assertRaisesRegex(ValueError, "single pair"):
            peak_report(torch.rand(2, 8, 16, 20), torch.rand(2, 8, 16, 20))


if __name__ == "__main__":
    unittest.main(verbosity=2)
