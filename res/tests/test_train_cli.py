"""Rehearse the training CLI itself: one step, on a stubbed dataset.

The training entry point owns argument handling, the output-directory guard,
the checkpoint and its provenance, the mixed-start sampler and one full
optimiser step.  A wiring mistake there costs a server run, so this drives
``res.train_vtmot.main()`` end to end on synthetic frames, and pins the guard
that refuses to write into a directory that already holds a run.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

import res.train_vtmot as tv

STACK = ("res/configs/registration.yaml",
         "res/configs/stage0_structural_prior.yaml",
         "res/configs/ab_spatial_prior32.yaml",
         "res/configs/stage3_structural_iterative_r2.yaml",
         "res/configs/stage4_structural_mixed_start.yaml")


class _StubDataset:
    """One synthetic frame in place of VTMOTSingleFrameDataset."""

    def __init__(self, *args, **kwargs):
        torch.manual_seed(0)
        self.pair = {"ir": torch.rand(1, 64, 80), "vi": torch.rand(3, 64, 80),
                     "gt_flow": torch.randn(2, 64, 80) * 4.0,
                     "valid_mask": torch.ones(1, 64, 80), "gt_h": torch.eye(3),
                     "sequence": "synthetic", "stem": "0001"}

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return self.pair


def _shrink(path: Path) -> Path:
    """A generated overlay that makes the run small enough for a CPU test."""
    overlay = path / "tiny.yaml"
    OmegaConf.save(OmegaConf.create({
        "structural_prior": {"base_channels": 8, "out_channels": 16,
                             "blocks_per_scale": 1},
        "coarse_transformer": {"num_layers": 1},
        "global_matcher": {"max_tokens": 80},
        "vtmot_data": {"target_hw": [64, 80], "crop_hw": [64, 80],
                       "train_frame_stride": 1, "eval_frame_stride": 1},
        "vtmot_train": {"batch_size": 1, "num_workers": 0, "steps": 1,
                        "eval_every": 1, "checkpoint_every": 1, "log_every": 1,
                        "amp": False, "seed": 3}}), overlay)
    return overlay


class TrainCliTest(unittest.TestCase):
    def _argv(self, root: Path, output: Path, overlay: Path):
        argv = ["train_vtmot.py", "--device", "cpu", "--steps", "1",
                "--num-workers", "0", "--data-root", str(root),
                "--split-file", str(root / "split.json"),
                "--output-dir", str(output)]
        for path in STACK[:-1]:
            argv += ["--overlay", path]
        argv += ["--overlay", str(overlay), "--overlay", STACK[-1]]
        return argv

    def test_one_step_run_writes_a_checkpoint_that_records_its_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overlay = _shrink(root)
            output = root / "run"
            with patch.object(tv, "VTMOTSingleFrameDataset", _StubDataset), \
                    patch.object(sys, "argv", self._argv(root, output, overlay)):
                tv.main()
            self.assertTrue((output / "last.pt").is_file())
            self.assertTrue((output / "metrics.jsonl").is_file())
            record = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
            # The mixed-start sampler must actually run and report its draw.
            self.assertIn(record["init_flow_mode"], ("truth", "translation", "local",
                                                     "coarse"))
            checkpoint = torch.load(output / "last.pt", map_location="cpu",
                                    weights_only=True)
            self.assertEqual(checkpoint["config"]["vtmot_train"]
                             ["init_flow_mix"]["probability"], 0.75)
            self.assertIn("--overlay", checkpoint["provenance"]["argv"])
            self.assertIsNone(checkpoint["provenance"]["init"])

    def test_an_occupied_output_directory_is_refused(self):
        """The guard that stops a real run being overwritten by accident."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overlay = _shrink(root)
            output = root / "run"
            output.mkdir()
            (output / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
            with patch.object(tv, "VTMOTSingleFrameDataset", _StubDataset), \
                    patch.object(sys, "argv", self._argv(root, output, overlay)):
                with self.assertRaisesRegex(ValueError, "already contains a run"):
                    tv.main()


if __name__ == "__main__":
    unittest.main()
