"""Rehearse the evaluation CLI itself, on a synthetic checkpoint.

Argument validation, the config merge, the checkpoint load, the per-round
diagnostics, the gate sweep and the JSON write all live in ``main()`` and none
of them are covered by calling ``evaluate()`` directly.  A stale guard there cost
a server round trip once, so this drives the real entry point end to end with a
stubbed dataset; only the image loading is substituted.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

import res.evaluate_vtmot as ev
from res.model_factory import build_global_registration


class _StubDataset:
    """Two synthetic frames in place of VTMOTSingleFrameDataset."""

    def __init__(self, *args, **kwargs):
        torch.manual_seed(0)
        self.batches = [
            {"ir": torch.rand(1, 32, 40), "vi": torch.rand(3, 32, 40),
             "gt_flow": torch.randn(2, 32, 40) * 4.0,
             "valid_mask": torch.ones(1, 32, 40), "gt_h": torch.eye(3)}
            for _ in range(2)]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]


def _small_config():
    """The iterative overlay stack, shrunk.  Passed as the checkpoint config, so
    no overlay may be given on the command line or the widths would come back."""
    config = OmegaConf.merge(
        OmegaConf.load("res/configs/registration.yaml"),
        OmegaConf.load("res/configs/stage0_structural_prior.yaml"),
        OmegaConf.load("res/configs/ab_spatial_prior32.yaml"),
        OmegaConf.load("res/configs/stage3_structural_iterative_r2.yaml"))
    config.structural_prior.base_channels = 8
    config.structural_prior.out_channels = 16
    config.structural_prior.blocks_per_scale = 1
    config.coarse_transformer.num_layers = 1
    config.global_matcher.max_tokens = 80
    config.iterative_refinement.iterations = 2
    return config


class EvaluateCliTest(unittest.TestCase):
    def _run(self, extra_argv=()):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _small_config()
            model = build_global_registration(config)
            with torch.no_grad():               # a real correction, not a zero one
                for head in (model.iterative_refinement.update,
                             model.iterative_refinement.confidence):
                    head[-1].weight.normal_(0, 0.2)
                    head[-1].bias.normal_(0, 0.5)
            checkpoint = root / "last.pt"
            torch.save({"config": OmegaConf.to_container(config, resolve=True),
                        "model": model.state_dict()}, checkpoint)
            output = root / "report.json"
            argv = ["evaluate_vtmot.py", "--checkpoint", str(checkpoint),
                    "--data-root", str(root), "--split-file", str(root / "split.json"),
                    "--device", "cpu", "--batch-size", "1", "--output", str(output)]
            argv += list(extra_argv)
            with patch.object(ev, "VTMOTSingleFrameDataset", _StubDataset), \
                    patch.object(sys, "argv", argv):
                ev.main()
            return json.loads(output.read_text(encoding="utf-8"))

    def test_plain_run_reports_regions_that_partition_the_valid_area(self):
        report = self._run()
        for index in (1, 2):
            self.assertAlmostEqual(
                report[f"refine_round{index}_coverage"]
                + report[f"refine_round{index}_outside_coverage"], 1.0, places=5)
            self.assertIn(f"refine_round{index}_epe_allvalid_px", report)
            self.assertIn(f"refine_round{index}_epe_noupdate_px", report)
            self.assertIn(f"refine_round{index}_confidence_auc", report)
        self.assertLessEqual(report["refine_common_coverage"],
                             report["refine_round1_coverage"] + 1e-6)
        self.assertIn("pck_3px", report)

    def test_gate_sweep_and_truth_centred_run_accept_the_loop(self):
        """The guard must accept an iterative model with no local matcher, the
        raw gate must reproduce the ungated run, and a re-run per gate must
        produce its own per-round numbers."""
        report = self._run(("--diagnose-local-centre", "--diagnose-confidence-gate",
                            "--gate-specs", "raw,zero,scale0.5"))
        self.assertAlmostEqual(report["gate_raw_epe_px"], report["epe_px"], places=5)
        self.assertAlmostEqual(report["gate_raw_round1_epe_px"],
                               report["refine_round1_epe_allvalid_px"], places=5)
        for name in ("zero", "scale0.5"):
            self.assertIn(f"gate_{name}_epe_px", report)
            self.assertIn(f"gate_{name}_relative_epe", report)
            self.assertIn(f"gate_{name}_round2_epe_px", report)
            self.assertIn(f"gate_{name}_pck_3px", report)
        # No update at all must leave every round on the field the loop received.
        self.assertAlmostEqual(report["gate_zero_round1_epe_px"],
                               report["gate_zero_round2_epe_px"], places=5)
        for index in (1, 2):
            self.assertIn(f"truthcentre_refine_round{index}_epe_px", report)
            self.assertIn(f"truthcentre_refine_round{index}_applied_mean_norm_px",
                          report)

    def test_unknown_gate_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            self._run(("--diagnose-confidence-gate", "--gate-specs", "zero,bogus"))


if __name__ == "__main__":
    unittest.main()
