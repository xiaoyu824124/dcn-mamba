# -*- coding: utf-8 -*-
"""
MambaVF-style fusion module (replaces the Restormer self-attention fusion).

Design (based on arXiv:2602.06017 "MambaVF"):
  * per-source lightweight feature embedding (conv stem with stride -> the
    "patch / tubelet embedding" idea of MambaVF). Working at 1/4 resolution
    is what makes 640x480 real-time inference possible on an RTX 3060.
  * dual-stream temporal SSM fusion: each modality's DCN-aligned triplet
    (prev, cur, next) is scanned by VSS blocks along the temporal axis with a
    Mamba-style selective state-space model (linear complexity, no quadratic
    self-attention, no custom CUDA kernel -> Windows/3060 friendly).
  * cross-modal channel-wise concatenation (as in MambaVF Eq. 3) followed by
    a decoder made of 2D residual conv blocks (the paper keeps 2D ResBlocks).

DCN (deformable-conv) alignment is kept OUTSIDE this module in `net.py`.
The external interface is unchanged, so `net.py` still works:
    encoder_1 / encoder_2 : per-frame feature extractor  [B,3,H,W] -> [B,C,H/4,W/4]
    forward(features1, features2) : [B,3C,H',W'] x2 -> [B,out_ch,H',W']
"""
import torch
import torch.nn as nn

from .mamba_block import VSSBlock, ConvResBlock


class Fusion_Net(nn.Module):
    def __init__(
        self,
        dim=32,
        num_blocks=3,          # number of temporal VSS (SSM) fusion blocks per stream
        head=8,                # kept for config compatibility (unused)
        ffn_expansion_factor=2,  # kept for config compatibility (unused)
        bias=False,
        LayerNorm_type="WithBias",  # kept for config compatibility (unused)
        output_mask=False,
        downscale=2,           # number of stride-2 convs in the encoder stem (2 -> 1/4)
        enc_blocks=1,          # extra residual conv blocks after downsampling
        dec_blocks=2,          # 2D residual blocks in the decoder
        ssm_d_state=4,         # hidden-state dim of the temporal selective scan
    ):
        super(Fusion_Net, self).__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.output_mask = output_mask
        self.out_ch = 1 if output_mask else 3

        # ---- per-source feature embedding (stride-2 stem + light ResBlocks) ----
        self.encoder_1 = self._build_encoder(dim, bias, downscale, enc_blocks)
        self.encoder_2 = self._build_encoder(dim, bias, downscale, enc_blocks)

        # ---- dual-stream temporal SSM fusion (the Mamba part) ----
        self.ssm_blocks_1 = nn.ModuleList([
            VSSBlock(dim, d_state=ssm_d_state, bias=bias) for _ in range(num_blocks)
        ])
        self.ssm_blocks_2 = nn.ModuleList([
            VSSBlock(dim, d_state=ssm_d_state, bias=bias) for _ in range(num_blocks)
        ])

        # ---- cross-modal decoder (concat + 2D residual refinement) ----
        self.decoder = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            *[ConvResBlock(dim, bias=bias) for _ in range(dec_blocks)],
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, self.out_ch, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Sigmoid(),
        )

    @staticmethod
    def _build_encoder(dim, bias, downscale, enc_blocks):
        layers = []
        in_c = 3
        for i in range(downscale):
            layers += [
                nn.Conv2d(in_c, dim, kernel_size=3, stride=2, padding=1, bias=bias),
                nn.GELU(),
            ]
            in_c = dim
        for _ in range(enc_blocks):
            layers.append(ConvResBlock(dim, bias=bias))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # per-frame feature extraction (called by net.py's extract_features)
    # ------------------------------------------------------------------
    def extract(self, frames, index):
        if index == 1:
            return self.encoder_1(frames)
        return self.encoder_2(frames)

    # ------------------------------------------------------------------
    # fusion of one aligned triplet per modality
    # ------------------------------------------------------------------
    def forward(self, features1, features2):
        """
        features1 / features2: [B, 3C, H', W']  (prev_aligned, cur, next_aligned)
        Returns:              [B, out_ch, H', W']
        """
        B, C3, H, W = features1.shape
        C = C3 // 3
        assert C == self.dim, (f"expect 3*dim channels, got {C3} (dim={self.dim})")

        # channel order in features is (prev, cur, next) -> [B, C, T=3, H, W]
        x1 = features1.view(B, 3, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        x2 = features2.view(B, 3, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # temporal SSM fusion inside each source
        for blk in self.ssm_blocks_1:
            x1 = blk(x1)
        for blk in self.ssm_blocks_2:
            x2 = blk(x2)

        # cross-modal fusion: channel-wise concatenation of the central frames
        m1 = x1[:, :, 1]          # [B, C, H, W]
        m2 = x2[:, :, 1]
        fused = torch.cat([m1, m2], dim=1)   # [B, 2C, H, W]

        out = self.decoder(fused)            # [B, out_ch, H, W]
        return out


def unit_test():
    import numpy as np
    torch.manual_seed(0)
    x1 = torch.tensor(np.random.rand(2, 3 * 16, 64, 64).astype(np.float32))
    x2 = torch.tensor(np.random.rand(2, 3 * 16, 64, 64).astype(np.float32))
    model = Fusion_Net(dim=16, num_blocks=2)
    y = model(x1, x2)
    print('forward output shape:', tuple(y.shape), ' (expected (2, 3, 64, 64))')

    frames = torch.rand(2 * 5, 3, 256, 256)
    f = model.extract(frames, 1)
    print('encoder_1 output shape:', tuple(f.shape), ' (expected (10, 16, 64, 64))')


if __name__ == '__main__':
    unit_test()
