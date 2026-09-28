"""Mixed starting points for the 1/4 loop's residual supervision.

The loop's failure mode is a starting point it cannot leave: from the truth it
drifts and never returns.  These checks pin the sampler that is meant to teach
it otherwise, and the fact that the overlay changes that one thing.
"""

import unittest

import torch
from omegaconf import OmegaConf

from res.train_vtmot import sample_init_flow

OVERLAYS = ("res/configs/stage0_structural_prior.yaml",
            "res/configs/ab_spatial_prior32.yaml",
            "res/configs/stage3_structural_iterative_r2.yaml",
            "res/configs/stage4_structural_mixed_start.yaml")


def _settings(**overrides):
    base = {"probability": 1.0, "truth_weight": 0.4,
            "translation_px": [3.0, 6.0, 10.0], "local_px": 8.0}
    return OmegaConf.create({**base, **overrides})


class InitFlowMixTest(unittest.TestCase):
    def test_overlay_is_a_single_variable_change_over_r2(self):
        r2 = OmegaConf.merge(
            OmegaConf.load("res/configs/registration.yaml"),
            *(OmegaConf.load(path) for path in OVERLAYS[:3]))
        mixed = OmegaConf.merge(
            OmegaConf.load("res/configs/registration.yaml"),
            *(OmegaConf.load(path) for path in OVERLAYS))
        # Only the two intended knobs may differ.
        self.assertIsNone(r2.vtmot_train.get("init_flow_mix"))
        self.assertEqual(mixed.vtmot_train.init_flow_mix.probability, 0.75)
        # Sized from the natural residual (median 6.857 px, p90 10.224 px).
        self.assertEqual(list(mixed.vtmot_train.init_flow_mix.translation_px),
                         [3.0, 6.0, 10.0])
        self.assertEqual(float(r2.loss.weights.get("refine_proposal", 0.0)), 0.0)
        # The proposal term is off in this run: mixed starts are the only change,
        # so the r2 loss stands as the control.
        self.assertEqual(float(mixed.loss.weights.refine_proposal), 0.0)
        for key in ("iterations", "max_step_cells", "radius", "remove_mean"):
            self.assertEqual(getattr(mixed.iterative_refinement, key),
                             getattr(r2.iterative_refinement, key), key)
        for key in ("refine", "refine_match", "refine_confidence", "refine_smooth"):
            self.assertEqual(float(mixed.loss.weights[key]),
                             float(r2.loss.weights[key]), key)

    def test_modes_reproducible_and_truth_start_is_exact(self):
        gt = torch.randn(1, 2, 64, 80) * 3.0
        first = [sample_init_flow(_settings(), gt, seed=7, step=step)
                 for step in range(1, 121)]
        second = [sample_init_flow(_settings(), gt, seed=7, step=step)
                  for step in range(1, 121)]
        for (field, mode), (again, other) in zip(first, second):
            self.assertEqual(mode, other)
            if field is None:
                self.assertIsNone(again)
            else:
                self.assertTrue(torch.equal(field, again))
        modes = {mode for _, mode in first}
        self.assertEqual(modes, {"truth", "translation", "local"})
        for field, mode in first:
            if mode == "truth":
                self.assertTrue(torch.equal(field, gt))

    def test_translation_offsets_are_symmetric_and_cover_both_axes(self):
        gt = torch.zeros(1, 2, 64, 80)
        offsets = []
        for step in range(1, 601):
            field, mode = sample_init_flow(_settings(truth_weight=0.0), gt,
                                           seed=11, step=step)
            if mode == "translation":
                offsets.append((float(field[0, 0, 0, 0]), float(field[0, 1, 0, 0])))
        self.assertTrue(offsets)
        # Constant to the last bit over the field: the correction is its
        # negation everywhere.  (std() in float32 can show ~1e-6 of rounding on
        # a constant tensor, so compare the extremes instead.)
        field, mode = sample_init_flow(_settings(truth_weight=0.0), gt, seed=11, step=1)
        self.assertEqual(mode, "translation")
        self.assertEqual(float(field.max() - field.min()), 0.0)
        magnitudes = {round((dy ** 2 + dx ** 2) ** 0.5, 3) for dy, dx in offsets}
        self.assertEqual(magnitudes, {3.0, 6.0, 10.0})
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
            self.assertTrue(any(
                (offset[axis] > 0) == (sign > 0) and abs(offset[1 - axis]) < 1e-6
                for offset in offsets), f"axis={axis} sign={sign} missing")

    def test_local_mode_is_bounded_and_not_constant(self):
        gt = torch.zeros(1, 2, 64, 80)
        seen = 0
        for step in range(1, 601):
            field, mode = sample_init_flow(
                _settings(truth_weight=0.0, translation_px=[4.0], local_px=8.0),
                gt, seed=5, step=step)
            if mode != "local":
                continue
            seen += 1
            self.assertLessEqual(float(field.abs().max()), 8.0 + 1e-4)
            self.assertGreater(float(field.std()), 0.5)
            break
        self.assertEqual(seen, 1)

    def test_probability_and_validation(self):
        gt = torch.zeros(1, 2, 64, 80)
        self.assertEqual(sample_init_flow(None, gt, seed=1, step=1), (None, "coarse"))
        self.assertEqual(sample_init_flow(_settings(probability=0.0), gt, seed=1, step=1),
                         (None, "coarse"))
        with self.assertRaisesRegex(ValueError, "probability"):
            sample_init_flow(_settings(probability=1.5), gt, seed=1, step=1)
        with self.assertRaisesRegex(ValueError, "translation_px"):
            sample_init_flow(_settings(translation_px=[0.0]), gt, seed=1, step=1)
        with self.assertRaisesRegex(ValueError, "truth_weight"):
            sample_init_flow(_settings(truth_weight=2.0), gt, seed=1, step=1)


if __name__ == "__main__":
    unittest.main()
