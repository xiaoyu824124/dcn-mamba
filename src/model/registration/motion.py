"""Same-modality temporal motion estimation and fallback."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SpatialTransformer, _to_gray

class MonoModalFlow(nn.Module):
    """Small Lucas-Kanade fallback used only when OpenCV is unavailable."""

    def __init__(self, levels=3, window=7, iterations=2, max_flow=32.0):
        super().__init__()
        self.levels = max(1, int(levels))
        self.window = int(window)
        self.iterations = max(1, int(iterations))
        self.max_flow = float(max_flow)
        self.transformer = SpatialTransformer()

    @staticmethod
    def _gradients(gray):
        gx = torch.zeros_like(gray)
        gy = torch.zeros_like(gray)
        gx[..., 1:-1] = (gray[..., 2:] - gray[..., :-2]) * 0.5
        gy[..., 1:-1, :] = (gray[..., 2:, :] - gray[..., :-2, :]) * 0.5
        return gx, gy

    def _increment(self, previous, current, flow):
        warped = self.transformer(previous, flow)[0]
        gx, gy = self._gradients(current)
        diff = current - warped
        pad = self.window // 2
        kernel = torch.ones(
            1, 1, self.window, self.window,
            device=current.device, dtype=current.dtype)

        def box(value):
            return F.conv2d(
                F.pad(value, (pad, pad, pad, pad), mode="replicate"),
                kernel, padding=0)

        ixx = box(gx * gx)
        ixy = box(gx * gy)
        iyy = box(gy * gy)
        ixt = box(gx * diff)
        iyt = box(gy * diff)
        determinant = ixx * iyy - ixy * ixy
        valid = determinant > (1e-3 * (ixx + iyy) + 1e-8)
        safe = torch.where(valid, determinant, torch.ones_like(determinant))
        dx = torch.where(
            valid, (iyy * ixt - ixy * iyt) / safe, torch.zeros_like(safe))
        dy = torch.where(
            valid, (ixx * iyt - ixy * ixt) / safe, torch.zeros_like(safe))
        return torch.cat([dy, dx], dim=1)

    def forward(self, previous, current):
        previous = _to_gray(previous)
        current = _to_gray(current)
        pyramid_previous = [previous]
        pyramid_current = [current]
        for _ in range(self.levels - 1):
            pyramid_previous.append(F.interpolate(
                pyramid_previous[-1], scale_factor=0.5, mode="area"))
            pyramid_current.append(F.interpolate(
                pyramid_current[-1], scale_factor=0.5, mode="area"))

        flow = None
        for level in range(self.levels - 1, -1, -1):
            level_previous = pyramid_previous[level]
            level_current = pyramid_current[level]
            if flow is None:
                flow = level_current.new_zeros(
                    level_current.shape[0], 2,
                    level_current.shape[-2], level_current.shape[-1])
            else:
                flow = F.interpolate(
                    flow, size=level_current.shape[-2:],
                    mode="bilinear", align_corners=True) * 2.0
            for _ in range(self.iterations):
                flow = flow + self._increment(
                    level_previous, level_current, flow)
        return flow.clamp(-self.max_flow, self.max_flow)


class FarnebackFlow(nn.Module):
    """Same-modality temporal motion for transporting the previous field.

    Gradient note (important for any claim about what the network learns):
    with OpenCV present this is ``cv2.calcOpticalFlowFarneback`` on numpy
    arrays, so the transport *sampling geometry* can never be differentiated --
    only the amount of trust placed in the transported field, via the
    ``grid_sample`` derivative with respect to the grid.  Because roughly
    ``1 - keyframe_ratio`` of frames take their flow from this path, that is a
    real architectural limitation and should be stated as such.

    ``grad_through`` only affects the pure-torch Lucas-Kanade fallback used when
    OpenCV is missing: setting it to True lets that fallback carry gradients.
    The default False reproduces the previous behaviour exactly.
    """

    def __init__(self, levels=3, window=7, iterations=2, max_flow=32.0,
                 pyr_scale=0.5, grad_through=False):
        super().__init__()
        self.max_flow = float(max_flow)
        self.pyr_scale = float(pyr_scale)
        self.grad_through = bool(grad_through)
        self.fallback = MonoModalFlow(
            levels=levels, window=window, iterations=iterations,
            max_flow=max_flow)

    def forward(self, previous, current):
        try:
            import cv2
            import numpy as np
        except ImportError:
            if self.grad_through:
                return self.fallback(previous, current)
            with torch.no_grad():
                return self.fallback(previous, current)
        with torch.no_grad():
            return self._cv2_farneback(previous, current, cv2, np)

    def _cv2_farneback(self, previous, current, cv2, np):
        previous_gray = _to_gray(previous)
        current_gray = _to_gray(current)
        fields = []
        for index in range(previous_gray.shape[0]):
            previous_np = (
                previous_gray[index:index + 1].cpu().numpy()[0, 0] * 255.0
            ).clip(0, 255).astype("uint8")
            current_np = (
                current_gray[index:index + 1].cpu().numpy()[0, 0] * 255.0
            ).clip(0, 255).astype("uint8")
            field = cv2.calcOpticalFlowFarneback(
                current_np, previous_np, None, self.pyr_scale,
                3, 15, 3, 5, 1.2, 0)
            fields.append(torch.from_numpy(
                field[..., [1, 0]].copy()).permute(2, 0, 1))
        return torch.stack(fields).to(
            device=previous.device, dtype=previous.dtype).clamp(
            -self.max_flow, self.max_flow)
