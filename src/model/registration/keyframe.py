"""Keyframe scheduling, field propagation, and WST."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SpatialTransformer, _to_gray
from .dcn_refinement import DCNLocalRefinement
from .global_align import GlobalCorrelationAlignment
from .lite_refinement import LiteResidualRefinement
from .motion import FarnebackFlow
from .sea_raft import SeaRAFT
from .trust_memory import TrustedMotionMemory, dual_modal_transport

class KeyframeRegistration(nn.Module):
    """Keyframe scheduling with a global (SEA-RAFT or correlation) coarse field."""

    def __init__(self, channels=16, alignment_threshold=0.35,
                 wst_enabled=True, motion=None, raft=None,
                 lite_refinement=None, keyframe=None,
                 local_refinement=None, memory=None, global_align=None):
        super().__init__()
        self.wst_enabled = bool(wst_enabled)

        # ---- coarse keyframe estimator -------------------------------------
        # Two mutually exclusive options; the global correlation matcher is the
        # default because the frozen SEA-RAFT is 5.4x worse than predicting no
        # motion on this cross-modal pair (see sea_raft.py).  Set
        # raft.enabled: true to restore the previous behaviour for an A/B.
        align_cfg = dict(global_align or {})
        self.global_align = None
        self.global_align_per_frame = bool(align_cfg.get("per_frame", True))
        # Parameter-space WST: smooth the 3x3 transform instead of the pixel
        # field.  Field-space blending degenerates once the per-frame coarse
        # field is accurate (`candidate` == `propagated` because both come from
        # the same deterministic matcher), so the alpha loses its meaning.  Six
        # numbers, on the other hand, can be smoothed meaningfully, and smoothing
        # a global transform is exactly what the temporal metrics (ITF/T-SSIM)
        # reward.  Requires global_align, so it defaults to off otherwise.
        self.wst_in_matrix_space = (
            self.global_align_per_frame
            and bool(align_cfg.get("wst_in_matrix_space", True)))
        if align_cfg.get("enabled", False):
            self.global_align = GlobalCorrelationAlignment(
                channels=align_cfg.get("channels", 32),
                mode=align_cfg.get("mode", "affine"),
                temperature=align_cfg.get("temperature", 0.07),
                mutual=align_cfg.get("mutual", True),
                ridge=align_cfg.get("ridge", 1e-3),
                learned_encoder=align_cfg.get("learned_encoder", True),
            )

        raft_cfg = dict(raft or {})
        self.keyframe_raft = None
        if raft_cfg.get("enabled", True):
            backend = str(raft_cfg.get("backend", "sea_raft")).lower()
            if backend not in {"sea_raft", "sea-raft", "vfbench"}:
                raise ValueError(
                    "The active registration pipeline requires "
                    "raft.backend=sea_raft")
            self.keyframe_raft = SeaRAFT(
                config_path=raft_cfg.get(
                    "config_path", "config/module/spring-S.json"),
                weights_path=raft_cfg.get("weights_path"),
                num_flow_updates=raft_cfg.get("num_flow_updates"),
                trainable=raft_cfg.get("trainable", True),
                max_flow=raft_cfg.get("max_flow"),
                allow_missing_weights=raft_cfg.get(
                    "allow_missing_weights", False),
            )
        if self.global_align is None and self.keyframe_raft is None:
            raise ValueError(
                "keyframe registration needs at least one coarse estimator: "
                "enable model.registration.global_align or "
                "model.registration.raft")
        lite_cfg = dict(lite_refinement or {})
        self.lite_refinement = LiteResidualRefinement(
            channels=lite_cfg.get("channels", 16),
            radius=lite_cfg.get("radius", 4),
            max_residual=lite_cfg.get("max_residual", 8.0),
        )
        motion_cfg = dict(motion or {})
        self.farneback = FarnebackFlow(
            levels=motion_cfg.get("levels", 3),
            window=motion_cfg.get("window", 7),
            iterations=motion_cfg.get("iterations", 2),
            max_flow=motion_cfg.get("max_flow", 32.0),
            grad_through=motion_cfg.get("grad_through", False),
        )
        self.stn = SpatialTransformer()
        self.score_stn = SpatialTransformer()
        # Gain on the memory's residual correction.  Zero-initialised, like the
        # lite and DCN heads, so that at initialisation the field is exactly the
        # global matcher's output and the memory cannot inject transport noise
        # before the data has asked for it.  Measured without this: adding the
        # memory correction from step 0 degraded the fixed-sample epe_ratio from
        # 0.0715 to 0.0915, i.e. the prior was a net negative until trained --
        # this makes "does the prior help?" a learned decision instead of a
        # hard-wired one.
        self.memory_gain = nn.Parameter(torch.zeros(1))
        memory_cfg = dict(memory or {})
        self.memory_enabled = bool(memory_cfg.get("enabled", True))
        self.memory_promotion_threshold = float(
            memory_cfg.get("promotion_threshold", 0.55))
        self.motion_memory = TrustedMotionMemory(
            capacity=memory_cfg.get("capacity", 3),
            age_decay=memory_cfg.get("age_decay", 0.98),
            channels=memory_cfg.get("channels", channels),
        )

        key_cfg = dict(keyframe or {})
        self.min_interval = max(1, int(key_cfg.get("min_interval", 3)))
        self.max_interval = max(
            self.min_interval, int(key_cfg.get("max_interval", 10)))
        self.quality_threshold = float(
            key_cfg.get("quality_threshold", alignment_threshold))
        self.confidence_threshold = float(
            key_cfg.get("confidence_threshold", 0.35))
        self.fb_threshold = float(key_cfg.get("fb_threshold", 2.5))
        self.quality_patience = max(
            1, int(key_cfg.get("quality_patience", 2)))
        self.fb_tau = float(key_cfg.get("fb_tau", 1.5))
        self.wst_alpha = tuple(
            key_cfg.get("wst_alpha", [0.259, 0.741, 1.0]))
        if len(self.wst_alpha) != 3:
            raise ValueError("keyframe.wst_alpha must contain three values")

        sigma = float(key_cfg.get("gaussian_sigma", 0.8))
        kernel_size = int(key_cfg.get("gaussian_kernel", 5))
        if kernel_size % 2 == 0:
            raise ValueError("keyframe.gaussian_kernel must be odd")
        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
        self.register_buffer(
            "gaussian_kernel", (kernel / kernel.sum()).view(
                1, 1, kernel_size, kernel_size))

        self.local_refinement = None
        local_cfg = dict(local_refinement or {})
        if local_cfg.get("enabled", False):
            self.local_refinement = DCNLocalRefinement(
                channels=local_cfg.get("channels", channels),
                coarse_kernel=local_cfg.get("coarse_kernel", 3),
                fine_kernel=local_cfg.get("fine_kernel", 1),
                max_flow=local_cfg.get("max_flow", 3.0),
                fine_max_flow=local_cfg.get("fine_max_flow", 1.0),
            )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_x.t().reshape(1, 1, 3, 3))

    @property
    def keyframe_net(self):
        return self.keyframe_raft

    @torch.no_grad()
    def _alignment_score(self, moving, fixed, flow):
        moving = _to_gray(moving).float()
        fixed = _to_gray(fixed).float()
        small = (
            max(8, moving.shape[-2] // 2),
            max(8, moving.shape[-1] // 2))
        moving = F.interpolate(moving, size=small, mode="area")
        fixed = F.interpolate(fixed, size=small, mode="area")
        flow = F.interpolate(
            flow.float(), size=small, mode="bilinear", align_corners=True)
        flow = flow * flow.new_tensor([
            small[0] / max(flow.shape[-2], 1),
            small[1] / max(flow.shape[-1], 1),
        ]).view(1, 2, 1, 1)
        warped = self.score_stn(moving, flow)[0]
        gx_m = F.conv2d(warped, self.sobel_x, padding=1)
        gy_m = F.conv2d(warped, self.sobel_y, padding=1)
        gx_f = F.conv2d(fixed, self.sobel_x, padding=1)
        gy_f = F.conv2d(fixed, self.sobel_y, padding=1)
        inner = gx_m * gx_f + gy_m * gy_f
        norm_m = gx_m.square() + gy_m.square() + 1e-3
        norm_f = gx_f.square() + gy_f.square() + 1e-3
        return (1.0 - inner.square() / (norm_m * norm_f)).mean(
            dim=(1, 2, 3))

    @torch.no_grad()
    def _temporal_fb_error(self, previous, current, backward_flow):
        forward_flow = self.farneback(current, previous)
        reverse_at_current = self.stn(forward_flow, backward_flow)[0]
        return (backward_flow + reverse_at_current).norm(
            dim=1).mean(dim=(1, 2))

    def _smooth_residual(self, residual):
        channels = residual.shape[1]
        kernel = self.gaussian_kernel.to(
            dtype=residual.dtype).expand(channels, 1, -1, -1)
        radius = kernel.shape[-1] // 2
        residual = F.pad(
            residual, (radius, radius, radius, radius), mode="replicate")
        return F.conv2d(residual, kernel, groups=channels)

    def _blend_transition(self, old_flow, new_flow, age):
        alpha = old_flow.new_tensor(self.wst_alpha)
        index = age.clamp(1, 3) - 1
        alpha = alpha[index].view(-1, 1, 1, 1)
        return (1.0 - alpha) * old_flow + alpha * new_flow

    def _blend_matrix(self, old_matrix, new_matrix, age):
        """Same WST schedule, applied to a ``[B, 3, 3]`` transform."""
        alpha = old_matrix.new_tensor(self.wst_alpha)
        alpha = alpha[age.clamp(1, 3) - 1].view(-1, 1, 1)
        return (1.0 - alpha) * old_matrix + alpha * new_matrix

    def _keyframe_candidate(self, moving, fixed, propagated):
        """Coarse keyframe field, plus a per-pixel confidence for blending it in.

        With ``global_align`` enabled the field comes from a 1/16-resolution
        global correlation reduced to a 6-DoF affine: no initialisation is
        needed, which is exactly what the RAFT-family estimator cannot offer
        cross-modally.  Confidence is the mutual-matching peak (how unambiguous
        the correspondence was), and the "fb error" slot reported to the loss is
        its complement, so all the existing loss/plumbing keeps working.
        """
        if self.global_align is not None:
            out = self.global_align(moving, fixed)
            fresh = out["flow"]
            confidence = out["confidence"]
            batch, _, height, width = fresh.shape
            fb_error = (1.0 - out["match_peak"]).view(batch, 1, 1, 1).expand(
                batch, 1, height, width).contiguous()
            candidate = propagated + confidence * (fresh - propagated)
            return candidate, confidence, fb_error

        forward = self.keyframe_raft(moving, fixed)
        backward = self.keyframe_raft(fixed, moving)
        backward_at_fixed = self.stn(backward, forward)[0]
        fb_error = (forward + backward_at_fixed).norm(
            dim=1, keepdim=True)
        confidence = torch.exp(
            -fb_error / max(self.fb_tau, 1e-3)).clamp(0.0, 1.0)
        fresh = self._smooth_residual(forward)
        candidate = propagated + confidence * (fresh - propagated)
        return candidate, confidence, fb_error

    def forward(self, moving_sequence, fixed_sequence):
        if moving_sequence.ndim != 5 or fixed_sequence.ndim != 5:
            raise ValueError("registration expects [B,T,C,H,W] inputs")
        if moving_sequence.shape[:2] != fixed_sequence.shape[:2]:
            raise ValueError("moving and fixed sequences must share B and T")

        batch, frames = moving_sequence.shape[:2]
        static_flow = moving_sequence.new_zeros(
            batch, 2, moving_sequence.shape[-2], moving_sequence.shape[-1])
        aligned, flows, residuals = [], [], []
        local_parts = []
        keyframe_masks, scores = [], []
        confidences, fb_errors, intervals = [], [], []
        temporal_fbs = []
        memory_confidences, memory_disagreements = [], []
        memory_slots, memory_promotions = [], []
        previous_flow = None
        previous_confidence = None
        trusted_memory = []
        pending_memory = None
        transition_old = None
        transition_new = None
        transition_age = torch.zeros(
            batch, dtype=torch.long, device=moving_sequence.device)
        last_key = torch.zeros(
            batch, dtype=torch.long, device=moving_sequence.device)
        quality_bad = torch.zeros(
            batch, dtype=torch.long, device=moving_sequence.device)
        # Running transform for the parameter-space WST.
        previous_matrix = None

        for index in range(frames):
            moving = moving_sequence[:, index]
            fixed = fixed_sequence[:, index]
            quality = moving.new_zeros(batch)
            if previous_flow is None:
                propagated = static_flow
                temporal_conf = moving.new_ones(
                    batch, 1, moving.shape[-2], moving.shape[-1])
                temporal_fb = moving.new_zeros(batch)
                memory_disagreement = moving.new_zeros(
                    batch, 1, moving.shape[-2], moving.shape[-1])
                memory_slot = torch.ones(
                    batch, dtype=torch.long, device=moving.device)
                memory_promotion = torch.zeros(
                    batch, dtype=torch.bool, device=moving.device)
                key_mask = torch.ones(
                    batch, dtype=torch.bool, device=moving.device)
                emergency = torch.zeros_like(key_mask)
                # Frame 0 has no transport and no memory history, so the memory
                # "verification" weight is 1 (trust the matcher) and its
                # reliability signal is that same 1 -- there is nothing yet for
                # the memory to disagree with.
                memory_conf = temporal_conf
                if self.global_align is not None and self.global_align_per_frame:
                    coarse = self.global_align(moving, fixed)
                    propagated = coarse["flow"]
                    temporal_conf = coarse["confidence"]
                    memory_conf = temporal_conf
                    previous_matrix = coarse["matrix"].detach()
                lite_delta, lite_conf = self.lite_refinement(
                    moving, fixed, propagated)
                lite_part = lite_conf * lite_delta
                provisional = propagated + lite_part
                temporal_conf = temporal_conf * lite_conf
            else:
                # A cross-modal field must be transported in both coordinate
                # systems.  IR-only transport, used in the old pipeline,
                # shifts the sampled IR location but leaves its VIS domain at
                # the wrong time step and therefore accumulates drift.
                infrared_backward = self.farneback(
                    moving_sequence[:, index - 1], moving)
                visible_backward = self.farneback(
                    fixed_sequence[:, index - 1], fixed)
                infrared_forward = self.farneback(
                    moving, moving_sequence[:, index - 1])

                trusted_memory = self.motion_memory.advance(
                    trusted_memory, visible_backward, infrared_forward)
                if pending_memory is not None:
                    pending_memory = self.motion_memory.advance(
                        [pending_memory], visible_backward, infrared_forward)[0]
                    pending_memory, promotion_score = self.motion_memory.assess(
                        moving, fixed, pending_memory)
                    memory_promotion = (
                        promotion_score >= self.memory_promotion_threshold)
                    if bool(memory_promotion.any()):
                        # Admission is masked per sample.  Failed candidates
                        # occupy no useful weight and are replaced by the next
                        # verified keyframe rather than contaminating reads.
                        pending_memory["valid"] = pending_memory["valid"] * (
                            memory_promotion[:, None, None, None])
                        trusted_memory = self.motion_memory.admit(
                            trusted_memory, pending_memory)
                    pending_memory = None
                else:
                    memory_promotion = torch.zeros(
                        batch, dtype=torch.bool, device=moving.device)

                direct_flow = dual_modal_transport(
                    self.stn, previous_flow, visible_backward, infrared_forward)
                direct_confidence = self.stn(
                    previous_confidence, visible_backward)[0]
                if self.memory_enabled:
                    propagated, temporal_conf, memory_details = self.motion_memory.fuse(
                        moving, fixed, trusted_memory, direct_flow,
                        direct_confidence)
                    memory_disagreement = memory_details["disagreement"]
                    memory_slot = memory_details["slots"]
                else:
                    propagated = direct_flow
                    temporal_conf = direct_confidence
                    memory_disagreement = moving.new_zeros(
                        batch, 1, moving.shape[-2], moving.shape[-1])
                    memory_slot = torch.zeros(
                        batch, dtype=torch.long, device=moving.device)
                # Reliability of the memory consensus, kept separate from the
                # confidence of the field that ends up in the output: `loss_memory`
                # must keep supervising the memory's own calibrator.
                memory_conf = temporal_conf
                # ---- per-frame global coarse field ---------------------------
                # Measured: with the matcher only at keyframes, frames 0-2 reach
                # epe_ratio 0.03-0.20 (50x better than zero flow) while the next
                # keyframe only reaches 0.91 and the frame after it 4.98.  The
                # cause was WST letting only 26% of the (now accurate) estimate
                # in and keeping 74% of the drifting transported field.  Running
                # the matcher on every frame removes that at the root: the coarse
                # field no longer depends on transport at all.  Cost is
                # negligible -- 26K parameters and a 1/16 correlation.
                if self.global_align is not None and self.global_align_per_frame:
                    coarse = self.global_align(moving, fixed)
                    memory_flow, memory_conf = propagated, temporal_conf
                    if self.wst_in_matrix_space:
                        # WST in PARAMETER space.  Blending pixel fields
                        # degenerated because `candidate` == `propagated` (both
                        # come from the same deterministic matcher), so alpha had
                        # no effect.  Six numbers can still be smoothed
                        # meaningfully, and smoothing a global transform is what
                        # the temporal metrics reward.
                        if previous_matrix is None:
                            matrix_used = coarse["matrix"]
                        else:
                            # `transition_age == 0` means no transition is
                            # running, and the flow-space code then uses the fresh
                            # field directly.  `_blend_matrix` would clamp age to 1
                            # and over-smooth, so the inactive case is explicit.
                            blending = (transition_age > 0).view(-1, 1, 1)
                            matrix_used = torch.where(
                                blending,
                                self._blend_matrix(
                                    previous_matrix, coarse["matrix"], transition_age),
                                coarse["matrix"])
                        previous_matrix = matrix_used.detach()
                        coarse_flow = self.global_align.flow_from_matrix(
                            matrix_used, *coarse["flow"].shape[-2:])
                        # The trusted memory becomes a VERIFIER: it contributes a
                        # reliability-weighted residual correction on top of the
                        # global transform.  Where transport drifted the memory
                        # disagrees with the matcher, its calibrator lowers the
                        # weight, and the correction vanishes.  This is what puts
                        # the memory back in the gradient path -- and it is now
                        # supervised by the GT flow through `flows`, instead of
                        # only regularising itself through `loss_memory`.
                        propagated = coarse_flow + torch.tanh(self.memory_gain) * (
                            memory_conf * (memory_flow - coarse_flow))
                        temporal_conf = coarse["confidence"]
                    else:
                        propagated = coarse["flow"]
                        temporal_conf = coarse["confidence"]
                        memory_conf = temporal_conf
                lite_delta, lite_conf = self.lite_refinement(
                    moving, fixed, propagated)
                lite_part = lite_conf * lite_delta
                provisional = propagated + lite_part
                temporal_fb = self._temporal_fb_error(
                    moving_sequence[:, index - 1], moving, infrared_backward)
                temporal_conf = temporal_conf * lite_conf
                quality = self._alignment_score(
                    moving, fixed, provisional)
                quality_bad = torch.where(
                    quality > self.quality_threshold,
                    quality_bad + 1, torch.zeros_like(quality_bad))
                elapsed = index - last_key
                emergency = (
                    (temporal_conf.mean(dim=(1, 2, 3))
                     < self.confidence_threshold)
                    | (temporal_fb > self.fb_threshold)
                    | (quality > self.quality_threshold * 1.75))
                regular = elapsed >= self.max_interval
                early = (
                    (elapsed >= self.min_interval)
                    & (quality_bad >= self.quality_patience))
                key_mask = emergency | regular | early

            # Realized keyframe age: frames elapsed since the last keyframe, per
            # frame.  Averaged over a window this measures the actual keyframe
            # cadence: 0 means every frame is a keyframe (degenerate scheduling,
            # the transport/trusted-memory path never runs), while (k-1)/2 means
            # a keyframe roughly every k frames.  It is a real measurement, not
            # the constant `max_interval` it used to report.
            intervals.append(index - last_key)
            fresh_conf = temporal_conf
            fresh_fb = temporal_fb
            if self.wst_in_matrix_space:
                # The coarse estimator is the SAME matcher on every frame, so a
                # keyframe re-estimation would return exactly `propagated` and cost
                # a second forward pass plus a second least-squares fit.  Skip it.
                candidate = provisional
            elif bool(key_mask.any()):
                candidate, key_conf, key_fb = self._keyframe_candidate(
                    moving, fixed, propagated)
                selector = key_mask.view(batch, 1, 1, 1)
                candidate = torch.where(selector, candidate, provisional)
                fresh_conf = torch.where(
                    selector, key_conf, temporal_conf)
                fresh_fb = torch.where(
                    key_mask, key_fb.mean(dim=(1, 2, 3)), temporal_fb)
            else:
                candidate = provisional

            active = transition_age > 0
            if transition_old is None:
                transitioned = propagated
            else:
                old_now = dual_modal_transport(
                    self.stn, transition_old, visible_backward, infrared_forward)
                new_now = dual_modal_transport(
                    self.stn, transition_new, visible_backward, infrared_forward)
                transitioned = self._blend_transition(
                    old_now, new_now, transition_age)
                transitioned = torch.where(
                    active.view(batch, 1, 1, 1),
                    transitioned, propagated)

            # ---- coarse / local decomposition ------------------------------
            # `local_part` accumulates ONLY what the learned local residual
            # heads add on top of the coarse field: the lite head's output on the
            # frames where it actually survives into the flow, plus the DCN delta
            # everywhere.  `flows` stays the TOTAL field, because that is what the
            # dense GT flow supervises and what `aligned` is warped with;
            # `local_flows` exists so the smoothness penalty and the local_flow_mag
            # diagnostic act on the learned residual instead of on the legitimate
            # smooth global motion.
            # The coarse field is now `global_align`'s affine (per-frame) rather
            # than transport / SEA-RAFT; see the block above.
            local_part = moving.new_zeros(batch, 2, *moving.shape[-2:])
            if self.wst_in_matrix_space:
                # Temporal smoothing already happened in parameter space and the
                # coarse field is computed per frame, so the keyframe / transition
                # machinery has nothing left to contribute to the field: every
                # frame takes the full coarse + residual path.  This also removes
                # the previous behaviour of DROPPING the lite residual on keyframe
                # and transition frames -- arbitrary, since the keyframe boundary
                # no longer means anything for the field, and measurably harmful
                # (the worst frame in the overfit test was the one right after a
                # transition).
                local_part = lite_part
                current_flow = provisional
                if index == 0:
                    transition_age.zero_()
            elif index == 0:
                current_flow = candidate
                transition_age.zero_()
            else:
                alpha_first = self.wst_alpha[0]
                key_target = (
                    candidate if not self.wst_enabled else
                    (1.0 - alpha_first) * propagated
                    + alpha_first * candidate)
                key_target = torch.where(
                    emergency.view(batch, 1, 1, 1),
                    candidate, key_target)
                current_flow = torch.where(
                    key_mask.view(batch, 1, 1, 1),
                    key_target, transitioned)
                no_transition = (~key_mask) & (~active)
                provisional_used = no_transition.view(batch, 1, 1, 1)
                current_flow = torch.where(
                    provisional_used, provisional, current_flow)
                local_part = torch.where(
                    provisional_used, lite_part, local_part)

            if self.local_refinement is not None:
                pre_dcn = current_flow
                current_flow, _ = self.local_refinement(
                    moving, fixed, current_flow)
                local_part = local_part + (current_flow - pre_dcn)
            warped = self.stn(moving, current_flow)[0]

            flow_residual = (
                torch.zeros_like(current_flow)
                if previous_flow is None else current_flow - previous_flow)
            previous_flow = current_flow
            previous_confidence = fresh_conf.detach()

            # The first keyframe seeds the memory.  Later keyframes are first
            # quarantined for one temporal step, then admitted only after the
            # reliability calibrator sees whether their geometry stays valid.
            if index == 0:
                trusted_memory = self.motion_memory.initialize(
                    current_flow, fresh_conf,
                    self.motion_memory.appearance_feature(fixed))
            elif bool(key_mask.any()) and self.memory_enabled:
                # Store the FINAL refined field, not the pre-refinement
                # `candidate`.  The frame-0 anchor already stored `current_flow`
                # (i.e. after DCN refinement), so using `candidate` here made
                # every later keyframe contribute a strictly lower-quality field
                # to the memory consensus, and the reliability calibrator was
                # scoring that weaker field when deciding whether to admit it.
                # Note the semantic choice: later keyframes store the WST-blended
                # field while frame 0 stores the pure candidate; consistency of
                # refinement quality is the property that matters for the memory.
                pending_memory = self.motion_memory._entry(
                    current_flow.detach(), fresh_conf.detach(),
                    valid=(key_mask[:, None, None, None].to(
                        dtype=fresh_conf.dtype) * torch.ones_like(fresh_conf)),
                    feature=self.motion_memory.appearance_feature(fixed).detach())

            if transition_old is None:
                transition_old = propagated.detach()
                transition_new = candidate.detach()
            else:
                selector = key_mask.view(batch, 1, 1, 1)
                transition_old = torch.where(
                    selector, propagated.detach(), transition_old)
                transition_new = torch.where(
                    selector, candidate.detach(), transition_new)
            if index == 0:
                transition_age.zero_()
            else:
                transition_age = torch.where(
                    key_mask & (~emergency),
                    torch.full_like(transition_age, 2),
                    transition_age)
                previous_age = transition_age.clone()
                transition_age = torch.where(
                    (~key_mask) & (previous_age > 0),
                    (previous_age + 1).clamp(max=3),
                    previous_age)
                transition_age = torch.where(
                    (~key_mask) & (previous_age >= 3),
                    torch.zeros_like(transition_age),
                    transition_age)
            last_key = torch.where(
                key_mask, torch.full_like(last_key, index), last_key)

            aligned.append(warped)
            flows.append(current_flow)
            local_parts.append(local_part)
            residuals.append(flow_residual)
            keyframe_masks.append(key_mask)
            scores.append(quality)
            confidences.append(fresh_conf.mean(dim=(1, 2, 3)))
            fb_errors.append(fresh_fb)
            # Farneback temporal consistency, kept separate from `fresh_fb`:
            # `fresh_fb` mixes this with the SEA-RAFT keyframe error, and the two
            # live on completely different scales (~1 px vs ~20 px), so a
            # threshold calibrated on the mixture is meaningless. `emergency`
            # compares `fb_threshold` against THIS quantity.
            temporal_fbs.append(temporal_fb)
            memory_confidences.append(memory_conf.mean(dim=(1, 2, 3)))
            memory_disagreements.append(
                memory_disagreement.mean(dim=(1, 2, 3)))
            memory_slots.append(memory_slot)
            memory_promotions.append(memory_promotion)

        return {
            "aligned": torch.stack(aligned, dim=1),
            "flows": torch.stack(flows, dim=1),
            "local_flows": torch.stack(local_parts, dim=1),
            "flow_residuals": torch.stack(residuals, dim=1),
            "keyframe_mask": torch.stack(keyframe_masks, dim=1),
            "alignment_scores": torch.stack(scores, dim=1),
            "adaptive_intervals": torch.stack(intervals, dim=1),
            "static_flow": static_flow,
            "confidence": torch.stack(confidences, dim=1),
            "fb_error": torch.stack(fb_errors, dim=1),
            "temporal_fb": torch.stack(temporal_fbs, dim=1),
            "memory_confidence": torch.stack(memory_confidences, dim=1),
            "memory_disagreement": torch.stack(memory_disagreements, dim=1),
            "memory_slots": torch.stack(memory_slots, dim=1),
            "memory_promotions": torch.stack(memory_promotions, dim=1),
        }
