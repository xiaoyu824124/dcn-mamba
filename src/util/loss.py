import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import SSIMLoss

from .flow_metric import flow_epe


def rgb2ycrcb(rgb_tensor):
    r, g, b = rgb_tensor[:, 0], rgb_tensor[:, 1], rgb_tensor[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5
    cb = (b - y) * 0.564 + 0.5
    return torch.stack([y, cr, cb], dim=1).clamp(0.0, 1.0)


def get_loss(loss_name, **kwargs):
    if loss_name == "IRVisibleFusionLoss":
        return IRVisibleFusionLoss(**kwargs)
    raise ValueError(f"Unknown loss function: {loss_name}")


class IRVisibleFusionLoss(nn.Module):
    """Fusion loss plus registration- and output-temporal constraints."""

    def __init__(self, coef, registration_coef=None):
        super().__init__()
        self.coef = tuple(coef)
        self.registration_coef = registration_coef or {}
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.ssim = SSIMLoss(11, reduction="none")

    @staticmethod
    def _gray(tensor):
        if tensor.shape[1] == 1:
            return tensor
        return (0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2]
                + 0.114 * tensor[:, 2:3])

    def _gradient_xy(self, tensor):
        tensor = self._gray(tensor)
        return (F.conv2d(tensor, self.sobel_x, padding=1),
                F.conv2d(tensor, self.sobel_y, padding=1))

    def sobel_filter(self, tensor):
        grad_x, grad_y = self._gradient_xy(tensor)
        return grad_x.abs() + grad_y.abs()

    @staticmethod
    def charbonnier(tensor, eps=1e-3):
        return torch.sqrt(tensor.square() + eps * eps).mean()

    def saliency_weights(self, infrared, visible, temperature=0.5):
        """Build fixed modality weights from local contrast and edges."""
        infrared = self._gray(infrared)
        visible = self._gray(visible)
        ir_local = (infrared - F.avg_pool2d(
            infrared, 9, stride=1, padding=4)).abs()
        vi_local = (visible - F.avg_pool2d(
            visible, 9, stride=1, padding=4)).abs()
        ir_score = ir_local + 0.5 * self.sobel_filter(infrared) + 0.25 * infrared
        vi_score = vi_local + 0.5 * self.sobel_filter(visible)
        ir_score = ir_score / (ir_score.mean(dim=(2, 3), keepdim=True) + 1e-6)
        vi_score = vi_score / (vi_score.mean(dim=(2, 3), keepdim=True) + 1e-6)
        weights = torch.softmax(
            torch.cat((ir_score, vi_score), dim=1) / temperature, dim=1)
        return weights[:, :1].detach(), weights[:, 1:].detach()

    def compute_single_loss(self, fused, infrared, visible):
        fused = self._gray(fused)
        infrared = self._gray(infrared)
        visible = self._gray(visible)
        grad_f = self.sobel_filter(fused)
        grad_ir = self.sobel_filter(infrared)
        grad_vi = self.sobel_filter(visible)
        weight_ir, weight_vi = self.saliency_weights(infrared, visible)
        intensity_target = weight_ir * infrared + weight_vi * visible
        loss_int = F.l1_loss(fused, intensity_target)
        loss_grad = F.l1_loss(grad_f, torch.maximum(grad_ir, grad_vi))
        loss_ssim = (
            weight_ir * self.ssim(fused, infrared)
            + weight_vi * self.ssim(fused, visible)
        ).mean()
        return loss_int, loss_grad, loss_ssim

    def spatial_loss(self, fused, infrared, visible):
        losses = [self.compute_single_loss(fused[:, i], infrared[:, i], visible[:, i])
                  for i in range(fused.shape[1])]
        return tuple(torch.stack([item[index] for item in losses]).mean()
                     for index in range(3))

    def temporal_loss(self, fused, infrared, visible):
        batch, time, _, height, width = fused.shape
        fused_y = self._gray(fused.reshape(-1, 3, height, width)).reshape(
            batch, time, 1, height, width)
        infrared_y = self._gray(infrared.reshape(-1, 3, height, width)).reshape(
            batch, time, 1, height, width)
        visible_y = self._gray(visible.reshape(-1, 3, height, width)).reshape(
            batch, time, 1, height, width)
        flat_ir = infrared_y.reshape(batch * time, 1, height, width)
        flat_vi = visible_y.reshape(batch * time, 1, height, width)
        weight_ir, weight_vi = self.saliency_weights(flat_ir, flat_vi)
        weight_ir = weight_ir.reshape(batch, time, 1, height, width)
        weight_vi = weight_vi.reshape(batch, time, 1, height, width)
        pair_weight_ir = 0.5 * (weight_ir[:, 1:] + weight_ir[:, :-1])
        pair_weight_vi = 0.5 * (weight_vi[:, 1:] + weight_vi[:, :-1])

        diff_f = fused_y[:, 1:] - fused_y[:, :-1]
        diff_ir = infrared_y[:, 1:] - infrared_y[:, :-1]
        diff_vi = visible_y[:, 1:] - visible_y[:, :-1]
        target_change = pair_weight_ir * diff_ir + pair_weight_vi * diff_vi
        return self.charbonnier(diff_f - target_change)

    def normalized_gradient_loss(self, moving, fixed, eps=1e-3):
        """Modality-insensitive edge alignment (gradient sign is ignored)."""
        gx_m, gy_m = self._gradient_xy(moving)
        gx_f, gy_f = self._gradient_xy(fixed)
        inner = gx_m * gx_f + gy_m * gy_f
        norm_m = gx_m.square() + gy_m.square() + eps
        norm_f = gx_f.square() + gy_f.square() + eps
        return (1.0 - inner.square() / (norm_m * norm_f)).mean()

    @staticmethod
    def flow_smoothness(flow):
        dx = (flow[..., 1:] - flow[..., :-1]).abs().mean()
        dy = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
        return dx + dy

    def registration_loss(self, registration, visible, gt_flow=None):
        zero = visible.new_zeros(())
        if not registration or "flows" not in registration:
            return {
                "loss_reg": zero, "loss_flow_smooth": zero,
                "loss_reg_temp": zero,
                "loss_propagation": zero, "loss_fb": zero,
                "loss_memory": zero, "loss_flow_gt": zero,
                "flow_epe_ratio": zero,
                "weighted": zero,
            }

        aligned = registration["aligned"]
        batch, time, channels, height, width = aligned.shape
        # `aligned` is warped with the TOTAL field, so this term sees every
        # explicit alignment stage at once.  For supervised VTMOT training its
        # weight is 0 (see the config): measured on held-out windows the metric
        # scores the ground-truth alignment WORSE than identity, so it must not
        # be optimised while a dense GT flow exists.  It is kept computed as a
        # diagnostic.
        loss_reg = self.normalized_gradient_loss(
            aligned.reshape(batch * time, channels, height, width),
            visible.reshape(batch * time, visible.shape[2], height, width),
        )

        # Smoothness acts on the LEARNED LOCAL RESIDUAL only (`local_flows` is
        # the lite head's contribution where it survives into the flow, plus the
        # DCN delta everywhere).  This pipeline has no analytic affine part; the
        # point is to avoid penalising the legitimate smooth coarse motion that
        # transport / SEA-RAFT produce.
        local_flows = registration.get("local_flows")
        smooth_source = local_flows if local_flows is not None else registration["flows"]
        loss_flow_smooth = self.flow_smoothness(
            smooth_source.reshape(batch * time, 2, height, width))

        # Temporal continuity of the physical field (global + local composed).
        flows = registration["flows"]
        if time >= 3:
            loss_reg_temp = (flows[:, 2:] - 2 * flows[:, 1:-1]
                             + flows[:, :-2]).abs().mean()
        else:
            loss_reg_temp = zero

        residuals = registration.get("flow_residuals")
        key_mask = registration.get("keyframe_mask")
        if residuals is not None and key_mask is not None:
            non_key = (~key_mask).to(residuals.dtype).view(batch, time, 1, 1, 1)
            loss_propagation = ((residuals.abs() * non_key).sum()
                                / (non_key.sum() * residuals.shape[2]
                                   * height * width + 1e-6))
        else:
            loss_propagation = zero

        fb_error = registration.get("fb_error")
        key_mask = registration.get("keyframe_mask")
        if fb_error is not None and key_mask is not None:
            key_weight = key_mask.to(fb_error.dtype)
            loss_fb = (fb_error * key_weight).sum() / (key_weight.sum() + 1e-6)
        else:
            loss_fb = zero

        # The memory reader should be confident only when its candidates
        # agree.  The target is detached: it regularises calibration without
        # encouraging the flow field itself to collapse to a trivial value.
        memory_confidence = registration.get("memory_confidence")
        memory_disagreement = registration.get("memory_disagreement")
        if memory_confidence is not None and memory_disagreement is not None:
            agreement_target = torch.exp(-4.0 * memory_disagreement.detach())
            loss_memory = F.smooth_l1_loss(
                memory_confidence, agreement_target)
        else:
            loss_memory = zero

        # VTMOT supplies a dense, augmentation-aware target in exactly the
        # [dy, dx] backward-warp convention used by registration["flows"].
        # HDO has no target, so this branch remains identically zero there.
        if gt_flow is not None:
            if gt_flow.shape != flows.shape:
                raise ValueError(
                    "gt_flow must match registration flows: "
                    f"got {tuple(gt_flow.shape)}, expected {tuple(flows.shape)}")
            error = flows - gt_flow.to(device=flows.device, dtype=flows.dtype)
            loss_flow_gt = self.charbonnier(error)
            # Diagnostic, NOT part of the objective: the standard end-point error
            # (mean per-pixel L2 norm, not a component-wise L1) divided by the
            # SAME sample's zero-flow baseline. 1.0 means "no better than
            # predicting no motion".  This uses the shared definition in
            # src/util/flow_metric.py so the probe, the long-run script and this
            # loss can never report three different quantities again.
            # `loss_flow_gt` deliberately stays a robust Charbonnier objective,
            # so the two numbers are different quantities by design.
            epe_px, baseline_px = flow_epe(
                flows, gt_flow.to(device=flows.device, dtype=flows.dtype),
                per_sample=True)
            flow_epe_ratio = (epe_px / baseline_px.clamp_min(1e-6)).mean()
        else:
            loss_flow_gt = zero
            flow_epe_ratio = zero

        weights = self.registration_coef
        weighted = (
            weights.get("alignment", 0.0) * loss_reg
            + weights.get("smooth", 0.0) * loss_flow_smooth
            + weights.get("temporal", 0.0) * loss_reg_temp
            + weights.get("propagation", 0.0) * loss_propagation
            + weights.get("fb_consistency", 0.0) * loss_fb
            + weights.get("memory_reliability", 0.0) * loss_memory
            + weights.get("flow_supervision", 0.0) * loss_flow_gt
        )
        return {
            "loss_reg": loss_reg,
            "loss_flow_smooth": loss_flow_smooth,
            "loss_reg_temp": loss_reg_temp,
            "loss_propagation": loss_propagation,
            "loss_fb": loss_fb,
            "loss_memory": loss_memory,
            "loss_flow_gt": loss_flow_gt,
            "flow_epe_ratio": flow_epe_ratio,
            "weighted": weighted,
        }

    def forward(self, fused, infrared, visible, registration=None,
                compute_fusion=True, compute_registration=True, gt_flow=None):
        zero = visible.new_zeros(())
        if compute_registration:
            registration_losses = self.registration_loss(
                registration, visible, gt_flow=gt_flow)
        else:
            registration_losses = {
                "loss_reg": zero, "loss_flow_smooth": zero,
                "loss_reg_temp": zero,
                "loss_propagation": zero, "loss_fb": zero,
                "loss_memory": zero, "loss_flow_gt": zero,
                "flow_epe_ratio": zero,
                "weighted": zero,
            }

        # Diagnostics: not part of the objective. `alignment_score_mean` is what
        # the keyframe `alignment_threshold` should be calibrated on.
        local_flows = registration.get("local_flows") if registration else None
        local_flow_mag = (local_flows.abs().mean() if local_flows is not None
                          else zero)
        scores = registration.get("alignment_scores") if registration else None
        alignment_score_mean = scores.mean() if scores is not None else zero
        key_mask = registration.get("keyframe_mask") if registration else None
        keyframe_ratio = (key_mask.to(visible.dtype).mean() if key_mask is not None
                          else zero)
        intervals = registration.get("adaptive_intervals") if registration else None
        adaptive_interval_mean = (intervals.to(visible.dtype).mean()
                                  if intervals is not None else zero)
        confidence = registration.get("confidence") if registration else None
        confidence_mean = confidence.mean() if confidence is not None else zero
        fb_error = registration.get("fb_error") if registration else None
        fb_error_mean = fb_error.mean() if fb_error is not None else zero

        # Fusion is supervised with the registered IR sequence, not the raw one.
        fusion_ir = (registration["aligned"]
                     if registration and "aligned" in registration else infrared)
        fusion_ir = fusion_ir.detach()
        if compute_fusion:
            time = fused.shape[1]
            if fusion_ir.shape[1] > time:
                start = (fusion_ir.shape[1] - time) // 2
                fusion_ir = fusion_ir[:, start:start + time]
                fusion_visible = visible[:, start:start + time]
            else:
                fusion_visible = visible

            loss_int, loss_grad, loss_ssim = self.spatial_loss(
                fused, fusion_ir, fusion_visible)
            loss_temp = (self.temporal_loss(fused, fusion_ir, fusion_visible)
                         if self.coef[3] > 0 else zero)
            loss_fusion = (
                self.coef[0] * loss_int
                + self.coef[1] * loss_grad
                + self.coef[2] * loss_ssim
                + self.coef[3] * loss_temp
            )
        else:
            loss_int = loss_grad = loss_ssim = loss_temp = zero
            loss_fusion = zero

        loss_registration = registration_losses["weighted"]
        total = loss_fusion + loss_registration
        return {
            "loss": total,
            "loss_fusion": loss_fusion,
            "loss_registration": loss_registration,
            "loss_int": loss_int,
            "loss_grad": loss_grad,
            "loss_ssim": loss_ssim,
            "loss_temp": loss_temp,
            "local_flow_mag": local_flow_mag.detach(),
            "alignment_score_mean": alignment_score_mean.detach(),
            "keyframe_ratio": keyframe_ratio.detach(),
            "adaptive_interval_mean": adaptive_interval_mean.detach(),
            "confidence_mean": confidence_mean.detach(),
            "fb_error_mean": fb_error_mean.detach(),
            **{key: value for key, value in registration_losses.items()
               if key != "weighted"},
        }
