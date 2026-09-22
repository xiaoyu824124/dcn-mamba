"""Explicit bounded DCN residual-flow refinement."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from .common import ResBlock, SpatialTransformer, _conv, _to_gray

class DCNLocalRefinement(nn.Module):
    """Bounded explicit local residual-flow refinement."""

    def __init__(self, channels=16, coarse_kernel=3, fine_kernel=1,
                 max_flow=3.0, fine_max_flow=1.0):
        super().__init__()
        self.max_flow = float(max_flow)
        self.fine_max_flow = float(fine_max_flow)
        self.coarse_kernel = int(coarse_kernel)
        self.fine_kernel = int(fine_kernel)
        coarse_taps = self.coarse_kernel ** 2
        fine_taps = self.fine_kernel ** 2

        self.coarse = nn.Sequential(
            _conv(6, channels, 3),
            nn.LeakyReLU(0.1, inplace=True),
            _conv(channels, channels, 3, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            _conv(channels, channels, 3, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(channels),
            ResBlock(channels),
        )
        self.coarse_head = nn.Conv2d(
            channels, 3 * coarse_taps, 3, padding=1)
        self.fine = nn.Sequential(
            _conv(7, channels, 3),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(channels),
        )
        self.fine_head = nn.Conv2d(channels, 3 * fine_taps, 3, padding=1)
        nn.init.zeros_(self.coarse_head.weight)
        nn.init.zeros_(self.coarse_head.bias)
        nn.init.zeros_(self.fine_head.weight)
        nn.init.zeros_(self.fine_head.bias)

        coarse_weight = torch.zeros(
            2, 2, self.coarse_kernel, self.coarse_kernel)
        for channel in range(2):
            coarse_weight[channel, channel] = 1.0 / coarse_taps
        self.coarse_weight = nn.Parameter(coarse_weight)
        self.register_buffer(
            "fine_weight", torch.eye(2).view(2, 2, 1, 1).clone())
        self.transformer = SpatialTransformer()

    @staticmethod
    def _normalized_flow(flow, height, width):
        scale = flow.new_tensor(
            [max(height - 1, 1), max(width - 1, 1)]).view(1, 2, 1, 1)
        return flow / scale

    @staticmethod
    def _coordinate_base(height, width, device):
        rows = torch.arange(height, device=device, dtype=torch.float32)
        cols = torch.arange(width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        return torch.stack([
            grid_y - (height - 1) / 2.0,
            grid_x - (width - 1) / 2.0,
        ]).unsqueeze(0)

    @staticmethod
    def _split_prediction(prediction, kernel_size):
        taps = kernel_size ** 2
        offsets = prediction[:, :2 * taps]
        mask = torch.sigmoid(prediction[:, 2 * taps:] + 4.0)
        return offsets, mask

    @staticmethod
    def _sample_base(base, offsets, weight, mask, kernel_size):
        if base.shape[0] != offsets.shape[0]:
            base = base.expand(offsets.shape[0], -1, -1, -1)
        padding = kernel_size // 2
        padded = (F.pad(base, (padding, padding, padding, padding),
                         mode="replicate") if padding else base)
        sampled = deform_conv2d(
            padded, offsets, weight, padding=0, mask=mask)
        baseline = deform_conv2d(
            padded, torch.zeros_like(offsets), weight,
            padding=0, mask=mask)
        return sampled - baseline

    def _residual(self, base, prediction, kernel_size, weight, bound, dtype):
        offsets, mask = self._split_prediction(prediction, kernel_size)
        raw = self._sample_base(
            base, offsets, weight.float(), mask, kernel_size)
        return (bound * torch.tanh(raw / bound)).to(dtype)

    def forward(self, moving, fixed, coarse_flow):
        moving_gray = _to_gray(moving)
        fixed_gray = _to_gray(fixed)
        height, width = moving_gray.shape[-2:]

        aligned = self.transformer(moving_gray, coarse_flow)[0]
        flow_norm = self._normalized_flow(coarse_flow, height, width)
        coarse_input = torch.cat([
            aligned, fixed_gray, aligned - fixed_gray,
            flow_norm, flow_norm.abs().sum(dim=1, keepdim=True),
        ], dim=1)
        coarse_features = self.coarse(coarse_input)
        prediction = self.coarse_head(coarse_features).float()
        level_h, level_w = coarse_features.shape[-2:]
        base = self._coordinate_base(
            level_h, level_w, coarse_features.device)
        offsets, mask = self._split_prediction(
            prediction, self.coarse_kernel)
        raw = self._sample_base(
            base, offsets, self.coarse_weight.float(),
            mask, self.coarse_kernel)
        raw = F.interpolate(
            raw, size=(height, width), mode="bilinear",
            align_corners=True) * 4.0
        coarse_residual = (
            self.max_flow * torch.tanh(raw / self.max_flow)
        ).to(coarse_flow.dtype)
        refined_coarse = coarse_flow + coarse_residual

        aligned_fine = self.transformer(moving_gray, refined_coarse)[0]
        fine_norm = self._normalized_flow(refined_coarse, height, width)
        fine_input = torch.cat([
            aligned_fine, fixed_gray, aligned_fine - fixed_gray,
            fine_norm, fine_norm.abs().sum(dim=1, keepdim=True),
            (coarse_residual / self.max_flow).abs().sum(dim=1, keepdim=True),
        ], dim=1)
        fine_features = self.fine(fine_input)
        fine_prediction = self.fine_head(fine_features).float()
        base = self._coordinate_base(
            height, width, fine_features.device)
        fine_residual = self._residual(
            base, fine_prediction, self.fine_kernel, self.fine_weight,
            self.fine_max_flow, refined_coarse.dtype)
        refined = refined_coarse + fine_residual
        return refined, {
            "flow_refined": refined,
            "flow_coarse": coarse_flow,
            "residual_coarse": coarse_residual,
            "residual_fine": fine_residual,
            "residual": refined - coarse_flow,
        }
