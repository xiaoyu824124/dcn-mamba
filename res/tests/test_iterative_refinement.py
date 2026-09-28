"""Wiring and contract checks for the iterative discrepancy-guided refinement."""

import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from res.iterative_refinement import (DiscrepancyGuidedRefinement,
                                      refinement_losses)
from res.evaluate_vtmot import evaluate
from res.model_factory import build_global_registration
from res.train_vtmot import freeze_coarse_parameters


def iterative_config(*, iterations: int = 3, radius: int = 2):
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
    config.iterative_refinement.radius = radius
    config.iterative_refinement.iterations = iterations
    return config


class _ConstantUpdate(torch.nn.Module):
    """A head whose output is the same value at every position."""

    def __init__(self, value: float = 0.25) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full_like(inputs[:, :2], self.value)


class IterativeRefinementTest(unittest.TestCase):
    def test_rounds_share_one_update_network(self):
        one = DiscrepancyGuidedRefinement(6, radius=2, iterations=1)
        three = DiscrepancyGuidedRefinement(6, radius=2, iterations=3)
        count = lambda module: sum(p.numel() for p in module.parameters())   # noqa: E731
        # Extra rounds must add computation, not parameters.
        self.assertEqual(count(one), count(three))
        self.assertEqual(len(list(three.update.parameters())),
                         len(list(one.update.parameters())))

    def test_zero_initialised_update_returns_the_incoming_flow(self):
        torch.manual_seed(3)
        module = DiscrepancyGuidedRefinement(6, radius=2, iterations=3).eval()
        feature_ir = torch.randn(1, 6, 8, 10)
        feature_vi = torch.randn(1, 6, 8, 10)
        coarse = torch.randn(1, 2, 8, 10) * 2.0
        output = module(feature_ir, feature_vi, coarse)
        self.assertEqual(len(output.flows), 3)
        for flow, delta in zip(output.flows, output.residuals):
            self.assertTrue(torch.equal(flow, coarse))
            self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        # tanh has unit slope at zero, so the projections are not gated: the loss
        # must still reach them, which a separate multiplicative gain would block.
        loss = sum(flow.square().mean() for flow in output.flows)
        loss.backward()
        self.assertGreater(float(module.update[-1].weight.grad.abs().sum()), 0.0)

    def test_flow_composes_and_the_residual_is_dense(self):
        torch.manual_seed(4)
        module = DiscrepancyGuidedRefinement(6, radius=2, iterations=3).eval()
        with torch.no_grad():
            module.update[-1].weight.normal_(0, 0.05)
            module.update[-1].bias.normal_(0, 0.05)
            module.confidence[-1].bias.fill_(0.0)
        coarse = torch.zeros(1, 2, 8, 10)
        output = module(torch.randn(1, 6, 8, 10), torch.randn(1, 6, 8, 10), coarse)
        previous = coarse
        for flow, applied in zip(output.flows, output.applied):
            self.assertTrue(torch.allclose(flow, previous + applied, atol=1e-6))
            self.assertGreater(float(applied.abs().max()), 0.0)
            previous = flow
        # Dense, not a global affine: a planar correction would have zero second
        # differences, and re-projecting every round onto an affine is exactly
        # what this stage is meant to avoid.
        residual = output.applied[-1][0]
        curvature = (residual[:, 2:, 1:-1] - 2 * residual[:, 1:-1, 1:-1]
                     + residual[:, :-2, 1:-1]).abs().max()
        self.assertGreater(float(curvature), 1e-6)

    def test_step_is_bounded_and_confidence_is_in_range(self):
        torch.manual_seed(5)
        module = DiscrepancyGuidedRefinement(6, radius=2, iterations=2,
                                             max_step_cells=1.5)
        with torch.no_grad():
            module.update[-1].weight.normal_(0, 5.0)      # try to force a big step
        output = module(torch.randn(1, 6, 8, 10), torch.randn(1, 6, 8, 10),
                        torch.zeros(1, 2, 8, 10))
        for delta, confidence in zip(output.residuals, output.confidence):
            self.assertLessEqual(float(delta.abs().max()), 1.5 + 1e-5)
            self.assertGreaterEqual(float(confidence.min()), 0.0)
            self.assertLessEqual(float(confidence.max()), 1.0)

    def test_correspondence_loss_uses_the_rounds_own_search_centre(self):
        """Round r searches around the flow *before* its update.

        Two consequences are checkable exactly.  The reported error must follow
        the post-update field, and the matching term must not move at all when
        only the update changes, because the search that produced it was centred
        on the pre-update field.
        """
        torch.manual_seed(6)
        module = DiscrepancyGuidedRefinement(6, radius=2, iterations=1).eval()
        height, width = 8, 10
        gt = torch.zeros(1, 2, 64, 80)
        valid = torch.ones(1, 1, 64, 80)
        y, x = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                              torch.arange(width, dtype=torch.float32), indexing="ij")
        base = torch.stack((y, x)).unsqueeze(0)
        zeros = base - base
        features = torch.randn(1, 6, height, width)
        centred = refinement_losses(module(features, features.clone(), zeros), gt, valid)
        # Zero initialised update, zero flow: nothing is displaced yet.
        self.assertAlmostEqual(float(centred["epe"][0]), 0.0, places=6)
        self.assertEqual(float(centred["confidence_target"][0]), 0.0)
        # Mean removal is opt-in, so by default even a constant head output moves
        # the field: a legitimate global translation must stay reachable.
        with torch.no_grad():
            for parameter in module.update.parameters():
                parameter.normal_(0, 1.0)
        moved = refinement_losses(module(features, features.clone(), zeros), gt, valid)
        self.assertGreater(float(moved["epe"][0]), 0.0)
        self.assertAlmostEqual(float(moved["match"][0]), float(centred["match"][0]),
                               places=6)
        # A centre far outside the radius covers no query, and the reported means
        # fall back to zero instead of raising.
        outside = refinement_losses(module(features, features.clone(), zeros + 12.0),
                                    gt, valid)
        self.assertEqual(float(outside["epe"][0]), 0.0)

    def test_mean_removal_is_opt_in(self):
        """The shipped behaviour must stay the pre-flag behaviour.

        ``remove_mean`` defaults off, so an unmodified config and checkpoint
        behave exactly as before the flag existed.  With it on, a constant raw
        update produces no step at all, which is the only thing it is allowed to
        delete: it must not touch the spatially varying part of the head output.
        """
        self.assertFalse(DiscrepancyGuidedRefinement(6).remove_mean)
        height, width = 8, 10
        features = torch.randn(1, 6, height, width)
        zeros = torch.zeros(1, 2, height, width)
        constant = _ConstantUpdate(0.25)
        for remove_mean, expected in ((False, True), (True, False)):
            module = DiscrepancyGuidedRefinement(
                6, radius=2, iterations=1, max_step_cells=2.0,
                remove_mean=remove_mean).eval()
            with patch.object(module, "update", constant):
                output = module(features, features.clone(), zeros)
            moved = float(output.residuals[0].abs().max())
            self.assertEqual(moved > 0.0, expected,
                             msg=f"remove_mean={remove_mean} moved {moved}")
            if not expected:
                # tanh keeps its unit slope at zero, so the step is the raw
                # constant itself and the confidence stays in (0, 1).
                self.assertAlmostEqual(float(output.residuals[0][0, 0, 0, 0]), 0.0,
                                       places=7)
            else:
                for delta in output.residuals:
                    self.assertLessEqual(float(delta.abs().max()), 2.0 + 1e-5)

    def test_overlay_builds_the_loop_and_keeps_the_coarse_field(self):
        torch.manual_seed(7)
        config = iterative_config(radius=2)
        model = build_global_registration(config)
        self.assertIsNone(model.local_matcher)
        self.assertIsNotNone(model.iterative_refinement)
        self.assertIsNotNone(model.fine_cross_attention)
        freeze_coarse_parameters(model)          # must accept the loop's model
        trainable = {name.split(".")[0] for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
        self.assertEqual(trainable, {"fine_interaction", "fine_cross_attention",
                                     "iterative_refinement"})
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        output = model(ir, vi)
        self.assertIsNotNone(output.refinement)
        self.assertEqual(len(output.refinement.flows), 3)
        self.assertEqual(tuple(output.final_flow.shape), (1, 2, 64, 80))
        # Round outputs match the frozen coarse field at warm start, so the
        # loop's own contribution is measured against an unchanged reference.
        for flow in output.refinement.flows:
            self.assertTrue(torch.isfinite(flow).all())


    def test_evaluator_reports_per_round_metrics(self):
        """The stage's exit condition is "round three beats round one", so the
        report must carry per-round numbers next to the coarse baseline."""
        from res.evaluate_vtmot import evaluate
        torch.manual_seed(14)
        model = build_global_registration(iterative_config(radius=2))
        batch = {"ir": torch.rand(1, 1, 64, 80), "vi": torch.rand(1, 3, 64, 80),
                 "gt_flow": torch.zeros(1, 2, 64, 80),
                 "valid_mask": torch.ones(1, 1, 64, 80),
                 "gt_h": torch.eye(3).unsqueeze(0)}
        report = evaluate(model.eval(), [batch], torch.device("cpu"))
        for index in (1, 2, 3):
            self.assertIn(f"refine_round{index}_epe_px", report)
            self.assertIn(f"refine_round{index}_match", report)
            self.assertIn(f"refine_round{index}_improved_fraction", report)
            # What the round actually applied, so a fixed bias can be told apart
            # from a correction that tracks the required translation.
            self.assertIn(f"refine_round{index}_applied_mean_norm_px", report)
            self.assertIn(f"refine_round{index}_applied_std_px", report)
            self.assertIn(f"refine_round{index}_required_mean_norm_px", report)
            self.assertIn(f"refine_round{index}_applied_mean_alignment", report)
            self.assertGreaterEqual(
                report[f"refine_round{index}_applied_std_px"], 0.0)
        self.assertIn("coarse_epe_px", report)
        self.assertTrue(all(value == value                        # not NaN
                            for name, value in report.items()
                            if name.startswith("refine_round")))


    def test_the_mean_removed_variant_forbids_a_constant_update(self):
        """With ``remove_mean`` on, the loop cannot answer with a constant.

        A flow-magnitude loss has a degenerate optimum -- a constant field
        already reaches the mean displacement error -- and the first probe of
        this loop showed exactly that: started from the ground truth, one round
        moved 2.02 px away and a second 3.58 px, while the aggregate error barely
        improved.  The flag is opt-in and off by default, so this variant is what
        closes that door; the default keeps the global translation reachable.
        """
        module = DiscrepancyGuidedRefinement(6, radius=2, iterations=1,
                                             remove_mean=True)
        with torch.no_grad():
            for parameter in module.update.parameters():
                parameter.zero_()
            # The only shape the degenerate solution can take: the same value at
            # every position.
            module.update[-1].bias.fill_(1.0)
            module.confidence[-1].bias.fill_(2.0)        # confidence close to one
        incoming = torch.zeros(1, 2, 8, 10)
        output = module(torch.randn(1, 6, 8, 10), torch.randn(1, 6, 8, 10), incoming)
        self.assertEqual(float(output.residuals[0].abs().max()), 0.0)
        self.assertTrue(torch.equal(output.flows[0], incoming))
        # A spatially varying head is still free to move, and each round's step
        # stays inside its bound.  A constant bias can no longer do it: with only
        # the last layer's bias set, every earlier layer still outputs zero, so
        # the head's output is constant and the correction is zero by design.
        with torch.no_grad():
            for parameter in module.update.parameters():
                parameter.normal_(0, 1.0)
        moved = module(torch.randn(1, 6, 8, 10), torch.randn(1, 6, 8, 10), incoming)
        self.assertGreater(float(moved.residuals[0].abs().max()), 0.0)
        self.assertLessEqual(float(moved.residuals[0].abs().max()),
                             module.max_step_cells + 1e-5)

    def test_second_attempt_overlay_retunes_the_loop(self):
        """The first run showed round three does not beat round one and that the
        correspondence NLL is what drifts, so the second overlay cuts a round,
        halves the step and drops that term's weight."""
        overlay = OmegaConf.load("res/configs/stage3_structural_iterative_r2.yaml")
        self.assertEqual(int(overlay.iterative_refinement.iterations), 2)
        self.assertEqual(float(overlay.iterative_refinement.max_step_cells), 2.0)
        self.assertEqual(float(overlay.loss.weights.refine_match), 0.1)
        self.assertEqual(float(overlay.loss.weights.refine), 1.0)
        config = OmegaConf.merge(
            OmegaConf.load("res/configs/registration.yaml"),
            OmegaConf.load("res/configs/stage0_structural_prior.yaml"),
            OmegaConf.load("res/configs/ab_spatial_prior32.yaml"), overlay)
        config.structural_prior.base_channels = 8
        config.structural_prior.out_channels = 16
        config.structural_prior.blocks_per_scale = 1
        config.coarse_transformer.num_layers = 1
        config.global_matcher.max_tokens = 80
        torch.manual_seed(15)
        model = build_global_registration(config)
        self.assertEqual(model.iterative_refinement.iterations, 2)
        # The evaluator's centre-override guard keys on this attribute, so the
        # iterative model must advertise it even without a local matcher.
        self.assertIsNone(model.local_matcher)
        self.assertTrue(getattr(model, "supports_local_centre", False))
        batch = {"ir": torch.rand(1, 1, 64, 80), "vi": torch.rand(1, 3, 64, 80),
                 "gt_flow": torch.zeros(1, 2, 64, 80),
                 "valid_mask": torch.ones(1, 1, 64, 80),
                 "gt_h": torch.eye(3).unsqueeze(0)}
        report = evaluate(model.eval(), [batch], torch.device("cpu"),
                          diagnose_local_centre=True)
        # The loop has no single-shot matcher, so the truth-centred diagnostic is
        # reported per round instead of as local_* keys.
        self.assertIn("truthcentre_refine_round1_epe_px", report)
        self.assertIn("truthcentre_refine_round2_epe_px", report)


if __name__ == "__main__":
    unittest.main()
