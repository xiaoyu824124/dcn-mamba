import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import Fusion_Net
from .registration import KeyframeRegistration


def rgb_to_ycrcb(rgb):
    """Convert an RGB tensor in [0, 1] to Y/Cb/Cr."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return torch.cat((y, cb, cr), dim=1)


def ycrcb_to_rgb(ycrcb):
    """Convert Y/Cb/Cr to RGB and keep the image in its valid range."""
    y, cb, cr = ycrcb[:, 0:1], ycrcb[:, 1:2], ycrcb[:, 2:3]
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    return torch.cat((r, g, b), dim=1).clamp(0.0, 1.0)


class IRVisibleFusion(nn.Module):
    """Registration-guided infrared-visible video fusion.

    Pipeline:
      1. SEA-RAFT high-precision re-estimation on keyframes;
      2. reliability-gated dual-domain motion-memory transport plus a
         lightweight residual head on non-keyframes;
      3. optional bounded DCN residual-flow refinement;
      4. temporal Mamba/SSM fusion and RGB chroma restoration.

    Everything that geometrically aligns the two modalities lives inside the
    registration branch, so it is optimized only by registration losses. The
    fusion branch never receives gradients that could move its own target.
    """

    def __init__(self, model_config):
        super().__init__()
        mcfg = model_config["model"]
        channels = mcfg["dim"]
        reg_cfg = mcfg.get("registration", {})
        self.registration_enabled = reg_cfg.get("enabled", False)

        if self.registration_enabled:
            # 配准链：关键帧粗配准（全局相关或 SEA-RAFT）、双域可信记忆输运 +
            # 轻量残差，以及可选显式 DCN 局部细化。
            self.registration = KeyframeRegistration(
                channels=reg_cfg.get("channels", 16),
                alignment_threshold=reg_cfg.get("alignment_threshold", 0.35),
                wst_enabled=reg_cfg.get("wst", True),
                motion=reg_cfg.get("motion"),
                raft=reg_cfg.get("raft"),
                global_align=reg_cfg.get("global_align"),
                lite_refinement=reg_cfg.get("lite_refinement"),
                keyframe=reg_cfg.get("keyframe"),
                local_refinement=reg_cfg.get("local_refinement"),
                memory=reg_cfg.get("memory"),
            )
        else:
            self.registration = None

        self.fusion_net = Fusion_Net(
            dim=channels,
            num_blocks=mcfg["num_blocks"],
            bias=mcfg["bias"],
            downscale=mcfg.get("downscale", 2),
            enc_blocks=mcfg.get("enc_blocks", 1),
            dec_blocks=mcfg.get("dec_blocks", 2),
            ssm_d_state=mcfg.get("ssm_d_state", 4),
        )

    def extract_features(self, frames, index):
        batch, time, channels, height, width = frames.shape
        flat = frames.reshape(batch * time, channels, height, width)
        encoder = self.fusion_net.encoder_1 if index == 1 else self.fusion_net.encoder_2
        features = encoder(flat)
        _, out_channels, out_height, out_width = features.shape
        return features.reshape(batch, time, out_channels, out_height, out_width)

    def set_training_stage(self, stage):
        """Select which branch is trainable for staged optimization."""
        if stage not in {"registration", "fusion", "joint"}:
            raise ValueError(f"Unknown training stage: {stage}")
        registration_trainable = stage in {"registration", "joint"}
        fusion_trainable = stage in {"fusion", "joint"}

        if self.registration is not None:
            # 全局运动、局部残差 flow 细化都在配准分支内，因此只由配准损失训练。
            self.registration.requires_grad_(registration_trainable)
            keyframe_raft = getattr(self.registration, "keyframe_raft", None)
            if (keyframe_raft is not None
                    and not getattr(keyframe_raft, "trainable", True)):
                keyframe_raft.requires_grad_(False)
            if stage == "fusion":
                self.registration.eval()
            elif self.training:
                self.registration.train()
        self.fusion_net.requires_grad_(fusion_trainable)

    @staticmethod
    def prepare_window_features(features):
        """Turn five frame features into the three overlapping triplets."""
        return [torch.cat([features[:, i], features[:, i + 1], features[:, i + 2]],
                          dim=1) for i in range(features.shape[1] - 2)]

    def _register_and_encode(self, sources_1, sources_2, stage="joint"):
        if self.registration_enabled:
            if stage == "fusion":
                with torch.no_grad():
                    registration = self.registration(sources_1, sources_2)
            else:
                registration = self.registration(sources_1, sources_2)

            # The fusion objective must not move its own registration target.
            # Registration is optimized only by explicit registration losses.
            aligned_sources_1 = registration["aligned"].detach()
        else:
            registration = {}
            aligned_sources_1 = sources_1

        feat1 = self.extract_features(aligned_sources_1, index=1)
        feat2 = self.extract_features(sources_2, index=2)

        return (self.prepare_window_features(feat1),
                self.prepare_window_features(feat2), registration)

    @staticmethod
    def _upsample_to(image, reference):
        if image.shape[-2:] != reference.shape[-2:]:
            image = F.interpolate(image, size=reference.shape[-2:], mode="bilinear",
                                  align_corners=False)
        return image

    def _fuse_window(self, feat1_window, feat2_window, index, visible_sources):
        fused_y = self.fusion_net(feat1_window[index], feat2_window[index])
        visible = visible_sources[:, index + 1]
        visible_ycrcb = rgb_to_ycrcb(visible)
        visible_y = visible_ycrcb[:, :1]

        # Suppress unstable chroma in dark regions; luminance is produced by fusion.
        color_weight = ((visible_y - 0.05) * 2.5).clamp(0.0, 1.0)
        clean_chroma = (visible_ycrcb[:, 1:] - 0.5) * color_weight + 0.5
        fused_y = self._upsample_to(fused_y, visible_y)
        return ycrcb_to_rgb(torch.cat((fused_y, clean_chroma), dim=1))

    def forward(self, sources_1, sources_2, stage="joint"):
        if stage == "registration":
            if not self.registration_enabled:
                raise RuntimeError("registration stage requires registration.enabled=true")
            return None, self.registration(sources_1, sources_2)

        feat1_window, feat2_window, registration = self._register_and_encode(
            sources_1, sources_2, stage=stage)
        outputs = [self._fuse_window(feat1_window, feat2_window, i, sources_2)
                   for i in range(len(feat1_window))]
        return torch.stack(outputs, dim=1), registration

    def forward_stream(self, sources_1, sources_2):
        """Fuse the center frame of the latest five-frame window."""
        feat1_window, feat2_window, _ = self._register_and_encode(sources_1, sources_2)
        center_window = len(feat1_window) // 2
        return self._fuse_window(feat1_window, feat2_window, center_window, sources_2)
