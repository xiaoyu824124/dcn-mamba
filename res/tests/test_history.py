"""Temporal interfaces: initial flow, feature cache, reliability, propagation."""

import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from res.history import (RegistrationHistory, align_history_features,
                         history_from_output, propagate_flow)
from res.model_factory import build_global_registration


def iterative_config():
    config = OmegaConf.merge(
        OmegaConf.load("res/configs/registration.yaml"),
        OmegaConf.load("res/configs/stage0_structural_prior.yaml"),
        OmegaConf.load("res/configs/ab_spatial_prior32.yaml"),
        OmegaConf.load("res/configs/stage3_structural_iterative.yaml"))
    config.structural_prior.base_channels = 8
    config.structural_prior.out_channels = 16
    config.structural_prior.blocks_per_scale = 1
    config.coarse_transformer.num_layers = 1
    config.global_matcher.max_tokens = 80
    config.iterative_refinement.radius = 2
    config.iterative_refinement.iterations = 2
    return config


class HistoryTest(unittest.TestCase):
    def test_propagation_resamples_instead_of_copying(self):
        """The keyframe flow must be evaluated at the motion-warped coordinate.

        With zero frame motion the composition degenerates to ``f_k(p) + t``,
        which a copy would also produce.  A non-zero frame motion makes the
        difference visible: the correct field is ``f_k(p + m) + t``, and the
        copied field is wrong by ``slope * m``.
        """
        height, width = 24, 32
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                              torch.arange(width, dtype=torch.float32), indexing="ij")
        slope = 0.2
        motion_x = 5.0
        keyframe_flow = torch.stack((torch.zeros_like(y), slope * x)).unsqueeze(0)
        motion = torch.stack((torch.zeros_like(y),
                              torch.full_like(x, motion_x))).unsqueeze(0)
        ir_motion = torch.stack((torch.full_like(y, 3.0),
                                 torch.full_like(x, -2.0))).unsqueeze(0)
        initial, valid = propagate_flow(motion, keyframe_flow, ir_motion)
        # q = (y, x + m); r = q + f_k(q) = (y, 1.2x + m + slope*(x + m));
        # s = r + t, so init_dx = slope*(x + m) + m - 2.
        expected_dx = slope * (x + motion_x) + motion_x - 2.0
        inside = valid[0, 0] > 0.5
        self.assertGreater(float(inside.float().mean()), 0.5)
        self.assertTrue(torch.allclose(initial[0, 0][inside],
                                       torch.full_like(initial[0, 0][inside], 3.0),
                                       atol=1e-4))
        self.assertTrue(torch.allclose(initial[0, 1][inside], expected_dx[inside],
                                       atol=1e-3))
        # Copying the keyframe flow and adding the motion would give this, which
        # is off by slope * motion_x.
        copied = (slope * x - 2.0 + motion_x)[inside]
        self.assertTrue(torch.allclose((expected_dx[inside] - copied).abs(),
                                       torch.full_like(copied, slope * motion_x),
                                       atol=1e-3))

    def test_propagation_masks_out_of_frame_legs(self):
        height, width = 16, 20
        motion = torch.zeros(1, 2, height, width)
        keyframe_flow = torch.zeros(1, 2, height, width)
        keyframe_flow[0, 0] = -20.0                      # every r leaves the frame
        _, valid = propagate_flow(motion, keyframe_flow,
                                  torch.zeros(1, 2, height, width))
        self.assertEqual(float(valid.max()), 0.0)

    def test_aligned_history_features_are_identity_for_zero_motion(self):
        history = RegistrationHistory(
            keyframe_index=0, keyframe_flow=torch.zeros(1, 2, 32, 40),
            ir_features={"1/8": torch.randn(1, 5, 4, 5),
                         "1/4": torch.randn(1, 6, 8, 10)})
        aligned = align_history_features(history, torch.zeros(1, 2, 32, 40))
        self.assertEqual(set(aligned), {"1/8", "1/4"})
        self.assertEqual(tuple(aligned["1/4"].shape), (1, 6, 8, 10))
        self.assertTrue(torch.allclose(aligned["1/4"], history.ir_features["1/4"],
                                       atol=1e-5))
        self.assertEqual(align_history_features(RegistrationHistory(), None or
                                                torch.zeros(1, 2, 32, 40)), {})

    def test_single_frame_model_exposes_the_temporal_interfaces(self):
        torch.manual_seed(13)
        model = build_global_registration(iterative_config()).eval()
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        with torch.no_grad():
            plain = model(ir, vi)
            # init_flow must move only the 1/4 stage's starting point.  A large
            # shift is used so part of the frame leaves the moving image and the
            # reliability mask is genuinely exercised.
            given = torch.full_like(plain.coarse_flow, 40.0)
            started = model(ir, vi, init_flow=given)
            featured = model(ir, vi, init_flow=given, return_features=True)
        self.assertTrue(torch.equal(plain.coarse_flow, started.coarse_flow))
        self.assertFalse(torch.equal(plain.final_flow, started.final_flow))
        # The zero-initialised rounds return exactly the field they were given,
        # so the loop's starting point is observable.  The loop works on the 1/4
        # lattice, so the image-grid field is converted with its stride.
        quarter = F.interpolate(given, size=(16, 20), mode="bilinear",
                                align_corners=False) / 4.0
        self.assertTrue(torch.allclose(started.refinement.flows[0], quarter, atol=1e-5))
        self.assertTrue(torch.allclose(started.refinement.flows[-1], quarter, atol=1e-5))
        self.assertIsNone(plain.ir_features)
        self.assertEqual(set(featured.ir_features), {"1/2", "1/4", "1/8"})
        self.assertEqual(set(featured.vi_features), {"1/2", "1/4", "1/8"})
        for value in (featured.reliability, featured.reliable_mask):
            self.assertEqual(tuple(value.shape), (1, 1, 64, 80))
        self.assertGreaterEqual(float(featured.reliability.min()), 0.0)
        self.assertLessEqual(float(featured.reliability.max()), 1.0)
        # Reliability is zero wherever the region is unusable.
        outside = featured.reliable_mask < 0.5
        self.assertTrue(bool(outside.any()))
        self.assertEqual(float(featured.reliability[outside].abs().max()), 0.0)
        history = history_from_output(featured, frame_index=7)
        self.assertFalse(history.is_empty())
        self.assertEqual(history.keyframe_index, 7)
        self.assertEqual(set(history.ir_features), {"1/2", "1/4", "1/8"})
        with self.assertRaisesRegex(ValueError, "either init_flow or local_centre"):
            model(ir, vi, init_flow=given, local_centre=given)


if __name__ == "__main__":
    unittest.main()
