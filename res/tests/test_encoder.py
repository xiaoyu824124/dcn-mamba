"""Unit tests for the shared MIND multi-scale feature encoder.

Run from the repository root:
    python -B -m unittest res.tests.test_encoder -v
"""

import unittest

import torch

from res.encoder import MINDFeatureEncoder


class MINDFeatureEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.encoder = MINDFeatureEncoder(
            in_channels=8, base_channels=12, out_channels=20, blocks_per_scale=1)

    def test_pyramid_shapes_normalisation_and_gradient(self):
        descriptor = torch.rand(2, 8, 64, 80, requires_grad=True)
        features = self.encoder(descriptor)
        self.assertEqual(tuple(features["1/2"].shape), (2, 12, 32, 40))
        self.assertEqual(tuple(features["1/4"].shape), (2, 24, 16, 20))
        self.assertEqual(tuple(features["1/8"].shape), (2, 20, 8, 10))
        for feature in features.values():
            self.assertTrue(torch.isfinite(feature).all())
            self.assertTrue(torch.allclose(
                feature.norm(dim=1), torch.ones_like(feature[:, 0]), atol=2e-5))
        features["1/8"].square().mean().backward()
        self.assertIsNotNone(descriptor.grad)
        self.assertTrue(torch.isfinite(descriptor.grad).all())

    def test_pair_uses_one_shared_parameter_set(self):
        descriptor = torch.rand(1, 8, 64, 64)
        left, right = self.encoder.encode_pair(descriptor, descriptor.clone())
        for scale in ("1/2", "1/4", "1/8"):
            self.assertTrue(torch.allclose(left[scale], right[scale]))
        parameter_ids = {id(parameter) for parameter in self.encoder.parameters()}
        self.assertEqual(len(parameter_ids), len(list(self.encoder.parameters())))

    def test_invalid_spatial_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            self.encoder(torch.rand(1, 8, 63, 64))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_pair(self):
        encoder = self.encoder.cuda()
        descriptor = torch.rand(2, 8, 64, 64, device="cuda")
        ir, vi = encoder.encode_pair(descriptor, descriptor)
        self.assertTrue(ir["1/8"].is_cuda and vi["1/8"].is_cuda)


if __name__ == "__main__":
    unittest.main(verbosity=2)
