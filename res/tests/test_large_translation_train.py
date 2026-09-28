"""Controlled large-displacement sampling for the coarse-stage training run."""

import unittest

from omegaconf import OmegaConf

from res.train_vtmot import sample_large_translation


class LargeTranslationTrainingTest(unittest.TestCase):
    def test_overlay_enables_only_the_experimental_training_change(self):
        default = OmegaConf.load("res/configs/registration.yaml")
        config = OmegaConf.merge(
            default, OmegaConf.load("res/configs/stage0_structural_prior.yaml"),
            OmegaConf.load("res/configs/ab_spatial_prior32.yaml"),
            OmegaConf.load("res/configs/ab_large_translation_train.yaml"))
        self.assertIsNone(default.vtmot_train.get("large_translation"))
        self.assertFalse(config.local_matcher.enabled)
        self.assertEqual(config.global_matcher.spatial_prior_sigma, 4.0)
        self.assertEqual(config.loss.weights.edge, 0.0)
        self.assertEqual(config.vtmot_train.large_translation.max_abs_px, 64)

    def test_samples_are_reproducible_one_axis_and_cover_both_signs(self):
        settings = OmegaConf.create({"probability": 1.0, "min_abs_px": 16,
                                     "max_abs_px": 64, "padding_mode": "reflection"})
        first = [sample_large_translation(settings, seed=2025, step=step)
                 for step in range(1, 101)]
        second = [sample_large_translation(settings, seed=2025, step=step)
                  for step in range(1, 101)]
        self.assertEqual(first, second)
        self.assertTrue(any(dy > 0 for dy, _ in first))
        self.assertTrue(any(dy < 0 for dy, _ in first))
        self.assertTrue(any(dx > 0 for _, dx in first))
        self.assertTrue(any(dx < 0 for _, dx in first))
        for dy, dx in first:
            self.assertEqual((dy == 0) + (dx == 0), 1)
            self.assertLessEqual(max(abs(dy), abs(dx)), 64)
            self.assertGreaterEqual(max(abs(dy), abs(dx)), 16)

    def test_probability_and_range_validation(self):
        settings = {"probability": 0, "min_abs_px": 16, "max_abs_px": 64}
        self.assertIsNone(sample_large_translation(settings, seed=1, step=1))
        with self.assertRaisesRegex(ValueError, "probability"):
            sample_large_translation({**settings, "probability": 1.1}, seed=1, step=1)
        with self.assertRaisesRegex(ValueError, "min_abs_px"):
            sample_large_translation({**settings, "min_abs_px": 65}, seed=1, step=1)
        with self.assertRaisesRegex(ValueError, "padding_mode"):
            sample_large_translation({**settings, "padding_mode": "invalid"}, seed=1, step=1)


if __name__ == "__main__":
    unittest.main()
