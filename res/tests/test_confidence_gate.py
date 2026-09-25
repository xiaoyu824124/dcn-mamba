"""The evaluation gate should use confidence and improve only trusted cells."""

import unittest

import torch

from res.evaluate_vtmot import _top_confidence_gate
from res.metrics import endpoint_error


class ConfidenceGateTest(unittest.TestCase):
    def test_gating_keeps_helpful_correction_and_rejects_harmful_one(self):
        coarse = torch.zeros(1, 2, 4, 4)
        target = torch.zeros_like(coarse)
        target[:, 0, :2, :2] = 1
        local = torch.ones_like(coarse)
        local[:, 1] = 0
        probability = torch.tensor([[[[0.95, 0.8], [0.7, 0.6]],
                                     [[0.05, 0.2], [0.3, 0.4]]]])
        candidates = torch.ones_like(probability, dtype=torch.bool)
        valid = torch.ones(1, 1, 4, 4)

        gate = _top_confidence_gate(probability, candidates, (4, 4), 0.25)
        gated = coarse + gate * (local - coarse)

        self.assertEqual(float(gate.sum()), 4.0)
        self.assertTrue(torch.equal(gate[0, 0, :2, :2], torch.ones(2, 2)))
        self.assertEqual(float(endpoint_error(gated, target, valid)), 0.0)
        self.assertGreater(float(endpoint_error(coarse, target, valid)), 0.0)
        self.assertGreater(float(endpoint_error(local, target, valid)), 0.0)

    def test_invalid_queries_cannot_be_selected(self):
        probability = torch.tensor([[[[0.95, 0.8], [0.7, 0.6]],
                                     [[0.05, 0.2], [0.3, 0.4]]]])
        candidates = torch.ones_like(probability, dtype=torch.bool)
        candidates[:, :, 0, 0] = False
        gate = _top_confidence_gate(probability, candidates, (4, 4), 0.25)
        self.assertEqual(float(gate[:, :, :2, :2].sum()), 0.0)
        self.assertEqual(float(gate.sum()), 4.0)


if __name__ == "__main__":
    unittest.main()
