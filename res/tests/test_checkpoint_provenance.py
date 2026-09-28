"""Identifying a checkpoint's warm start from the artefacts alone."""

import unittest

import torch

from res.checkpoint_provenance import (describe, find_init, frozen_digest,
                                       frozen_keys)


def _model(encoder_value: float = 1.0, loop_value: float = 0.0) -> dict:
    return {"encoder.weight": torch.full((2, 2), encoder_value),
            "matcher.temperature": torch.tensor(1.0),
            "fine_interaction.gain": torch.full((2,), loop_value),
            "iterative_refinement.update.weight": torch.full((2,), loop_value)}


class CheckpointProvenanceTest(unittest.TestCase):
    def test_digest_ignores_the_trainable_14_stage_only(self):
        base = _model()
        # A later run: the coarse stage is untouched, the 1/4 stage moved.
        trained = _model(loop_value=1.0)
        self.assertEqual(frozen_digest(base), frozen_digest(trained))
        # A different warm start: the coarse stage itself differs.
        other = _model(encoder_value=2.0, loop_value=1.0)
        self.assertNotEqual(frozen_digest(base), frozen_digest(other))

    def test_frozen_keys_exclude_every_trainable_module(self):
        keys = frozen_keys(_model())
        self.assertIn("encoder.weight", keys)
        self.assertIn("matcher.temperature", keys)
        self.assertNotIn("fine_interaction.gain", keys)
        self.assertNotIn("iterative_refinement.update.weight", keys)

    def test_find_init_prefers_a_parent_over_a_later_run(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, model, steps, loop, val in (
                    ("parent", _model(), 3000, False, 6.111),
                    ("child", _model(loop_value=1.0), 3000, True, 5.9),
                    ("unrelated", _model(encoder_value=5.0), 3000, False, 7.0)):
                (root / name).mkdir()
                torch.save({"step": steps, "model": model,
                            "config": {"iterative_refinement": {"enabled": loop}}},
                           root / name / "last.pt")
                # The child's step-0 validation is its parent's final one: the
                # same weights scored on the same frames.
                first = 6.111 if loop else val
                (root / name / "metrics.jsonl").write_text(
                    json.dumps({"step": 0, "val_epe_px": first}) + "\n"
                    + json.dumps({"step": steps, "val_epe_px": val}) + "\n",
                    encoding="utf-8")
            target, matches = find_init(root, root / "child" / "last.pt")
            self.assertEqual(target["step"], 3000)
            self.assertAlmostEqual(target["warm_start_epe"], 6.111, places=6)
            digests = {row["path"] for row in matches}
            self.assertIn(str(root / "parent" / "last.pt"), digests)
            self.assertNotIn(str(root / "unrelated" / "last.pt"), digests)
            # The digest alone matches a whole chain, so the metrics decide:
            # exactly one candidate reproduces the step-0 validation.
            flagged = [row["path"] for row in matches if row["is_warm_start"]]
            self.assertEqual(flagged, [str(root / "parent" / "last.pt")])

    def test_describe_reports_recorded_provenance(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            torch.save({"step": 7, "model": _model(),
                        "config": {"iterative_refinement": {"enabled": True}},
                        "provenance": {"init": "res_runs/parent/last.pt"}}, path)
            row = describe(path)
            self.assertEqual(row["step"], 7)
            self.assertTrue(row["loop"])
            self.assertEqual(row["init"], "res_runs/parent/last.pt")


if __name__ == "__main__":
    unittest.main()
