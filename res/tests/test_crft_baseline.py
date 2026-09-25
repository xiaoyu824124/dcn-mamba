"""Check the flow and image conventions at the external CRFT boundary."""

import unittest

import torch

from res.crft_baseline import (evaluate_crft, flow_xy_to_yx_at, run_crft,
                               validate_model_hw)


class FakeCRFT(torch.nn.Module):
    def forward(self, data):
        self.assert_inputs(data)
        batch = data["image0"].shape[0]
        device = data["image0"].device
        data["flow_f_full"] = torch.tensor([2.0, -1.0], device=device).view(1, 2, 1, 1).expand(
            batch, 2, 24, 24)
        data["flow_c"] = torch.tensor([0.25, -0.125], device=device).view(1, 2, 1, 1).expand(
            batch, 2, 3, 3)

    @staticmethod
    def assert_inputs(data):
        assert data["image0"].shape[1:] == (3, 24, 24)
        assert data["image1"].shape[1:] == (3, 24, 24)
        assert torch.allclose(data["image0"][:, 0, 4:20],
                              torch.full((1, 16, 24), 0.5))
        assert torch.allclose(data["image1"][:, 0, 4:20],
                              torch.full((1, 16, 24), 0.25))
        assert torch.count_nonzero(data["image0"][:, :, :4]) == 0
        assert torch.allclose(data["image1"][:, 0], data["image1"][:, 1])
        assert data["mask0"].shape == (1, 3, 3)
        assert torch.equal(data["mask0"], data["mask1"])


class CRFTBaselineTest(unittest.TestCase):
    def test_xy_to_yx_and_physical_scaling(self):
        flow = torch.tensor([2.0, -1.0]).view(1, 2, 1, 1).expand(1, 2, 16, 24)
        converted = flow_xy_to_yx_at(flow, (32, 48))
        self.assertEqual(tuple(converted.shape), (1, 2, 32, 48))
        self.assertTrue(torch.allclose(converted[:, 0], torch.full((1, 32, 48), -2.0)))
        self.assertTrue(torch.allclose(converted[:, 1], torch.full((1, 32, 48), 4.0)))

    def test_external_input_order_and_masked_metrics(self):
        visible = torch.zeros(1, 3, 32, 48)
        visible[:, 0] = 0.5
        infrared = torch.ones(1, 1, 32, 48) * 0.25
        target = torch.empty(1, 2, 32, 48)
        target[:, 0] = -2.0
        target[:, 1] = 4.0
        valid = torch.ones(1, 1, 32, 48)
        valid[:, :, :4] = 0
        sample = {"ir": infrared[0], "vi": visible[0],
                  "gt_flow": target[0], "valid_mask": valid[0]}
        report = evaluate_crft(FakeCRFT(), [
            {name: value[None] for name, value in sample.items()}],
            torch.device("cpu"), (24, 24))
        self.assertAlmostEqual(report["epe_px"], 0.0, places=5)
        self.assertAlmostEqual(report["coarse_epe_px"], 0.0, places=5)
        self.assertAlmostEqual(report["pck_3px"], 1.0, places=5)
        self.assertGreater(report["zero_flow_epe_px"], 4.0)

    def test_rejects_unsupported_shape(self):
        with self.assertRaises(ValueError):
            validate_model_hw((17, 24))
        with self.assertRaises(ValueError):
            validate_model_hw((96, 128))
        with self.assertRaises(ValueError):
            run_crft(FakeCRFT(), torch.zeros(1, 3, 32, 48),
                     torch.zeros(1, 3, 32, 48), (24, 24))


if __name__ == "__main__":
    unittest.main()
