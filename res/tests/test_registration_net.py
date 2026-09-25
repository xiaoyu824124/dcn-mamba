"""Integration tests for the MIND -> encoder -> global matcher -> warp chain.

Run from the repository root:
    python -B -m unittest res.tests.test_registration_net -v
"""

import unittest

import torch

from res.encoder import MINDFeatureEncoder
from res.fine_interaction import FineScaleInteraction
from res.global_matcher import GlobalMatcher
from res.mind import MINDDescriptor
from res.local_matcher import LocalMatcher
from res.registration_net import MINDGlobalRegistration
from res.train_vtmot import freeze_coarse_parameters


def small_model() -> MINDGlobalRegistration:
    return MINDGlobalRegistration(
        mind=MINDDescriptor(),
        encoder=MINDFeatureEncoder(
            in_channels=8, base_channels=8, out_channels=12, blocks_per_scale=1),
        matcher=GlobalMatcher(temperature=0.1),
        local_matcher=LocalMatcher(radius=2, candidate_chunk=5),
    )


class MINDGlobalRegistrationTest(unittest.TestCase):
    def test_freeze_coarse_leaves_local_refiner_trainable(self):
        model = small_model()
        model.matcher = GlobalMatcher(temperature=0.1, learnable_temperature=True)
        model.fine_interaction = FineScaleInteraction(16, 12)
        freeze_coarse_parameters(model)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.matcher.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.local_matcher.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.fine_interaction.parameters()))
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        before = model(ir, vi).coarse_flow.detach().clone()
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                       if parameter.requires_grad), lr=1e-3)
        model(ir, vi).final_flow.square().mean().backward()
        optimizer.step()
        after = model(ir, vi).coarse_flow.detach()
        self.assertTrue(torch.equal(before, after))

    def test_output_shapes_finiteness_and_gradient(self):
        torch.manual_seed(17)
        model = small_model()
        ir = torch.rand(2, 1, 64, 80, requires_grad=True)
        vi = torch.rand(2, 3, 64, 80, requires_grad=True)
        output = model(ir, vi)
        self.assertEqual(tuple(output.coarse_aligned_ir.shape), (2, 1, 64, 80))
        self.assertEqual(tuple(output.coarse_flow.shape), (2, 2, 64, 80))
        self.assertEqual(tuple(output.final_flow.shape), (2, 2, 64, 80))
        self.assertEqual(tuple(output.final_aligned_ir.shape), (2, 1, 64, 80))
        self.assertEqual(tuple(output.local_match.probability.shape), (2, 25, 16, 20))
        self.assertEqual(tuple(output.confidence_1_8.shape), (2, 1, 8, 10))
        self.assertEqual(tuple(output.affine_yx.shape), (2, 3, 2))
        for tensor in (output.coarse_aligned_ir, output.coarse_flow,
                       output.confidence_1_8, output.affine_yx):
            self.assertTrue(torch.isfinite(tensor).all())
        output.final_aligned_ir.mean().backward()
        self.assertIsNotNone(ir.grad)
        self.assertIsNotNone(vi.grad)
        self.assertTrue(torch.isfinite(ir.grad).all())
        self.assertTrue(torch.isfinite(vi.grad).all())

    def test_non_multiple_of_eight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            small_model()(torch.rand(1, 1, 65, 64), torch.rand(1, 3, 65, 64))

    def test_compact_coarse_matching_preserves_full_resolution_fine_stage(self):
        class RecordingTransformer(torch.nn.Module):
            def forward(self, ir, vi):
                self.input_hw = tuple(ir.shape[-2:])
                return ir, vi

        class RecordingFineInteraction(FineScaleInteraction):
            def forward(self, fine_ir, fine_vi, coarse_ir, coarse_vi):
                self.coarse_hw = tuple(coarse_ir.shape[-2:])
                return super().forward(fine_ir, fine_vi, coarse_ir, coarse_vi)

        torch.manual_seed(23)
        model = small_model()
        model.coarse_transformer = RecordingTransformer()
        model.fine_interaction = RecordingFineInteraction(16, 12)
        model.eval()
        ir, vi = torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80)
        full = model(ir, vi)
        self.assertEqual(model.coarse_transformer.input_hw, (8, 10))
        model.coarse_match_max_tokens = 100
        unchanged = model(ir, vi)
        self.assertTrue(torch.equal(full.coarse_flow, unchanged.coarse_flow))
        model.coarse_match_max_tokens = 20
        compact = model(ir, vi)
        self.assertEqual(model.coarse_transformer.input_hw, (4, 5))
        self.assertEqual(model.fine_interaction.coarse_hw, (8, 10))
        self.assertEqual(tuple(compact.match.coarse_flow.shape[-2:]), (4, 5))
        self.assertEqual(tuple(compact.match.matching_probability.shape), (1, 20, 20))
        self.assertEqual(tuple(compact.coarse_flow.shape), (1, 2, 64, 80))
        self.assertEqual(tuple(compact.local_match.probability.shape[-2:]), (16, 20))
        self.assertTrue(torch.isfinite(compact.final_flow).all())
        compact.final_flow.square().mean().backward()
        self.assertIsNotNone(model.encoder.stage_8[0][0].weight.grad)

    def test_compact_coarse_matching_rejects_degenerate_wls_grid(self):
        model = small_model()
        model.coarse_match_max_tokens = 4
        with self.assertRaisesRegex(ValueError, "fewer than 3 cells"):
            model(torch.rand(1, 1, 64, 80), torch.rand(1, 3, 64, 80))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda(self):
        model = small_model().cuda()
        output = model(torch.rand(1, 1, 64, 64, device="cuda"),
                       torch.rand(1, 3, 64, 64, device="cuda"))
        self.assertTrue(output.coarse_aligned_ir.is_cuda)
        self.assertTrue(torch.isfinite(output.coarse_flow).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
