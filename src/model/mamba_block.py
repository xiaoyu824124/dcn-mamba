# -*- coding: utf-8 -*-
"""
MambaVF-inspired lightweight building blocks (pure PyTorch, NO custom CUDA kernel).

Reference: "MambaVF: State Space Model for Efficient Video Fusion" (arXiv:2602.06017).

Core idea transplanted from the paper:
  * video fusion is treated as a sequential *state update* over frames
      h_t = A_bar * h_{t-1} + B_bar * x_t ,   y_t = C^T h_t + D x_t      (Eq. 2)
  * instead of heavy Restormer self-attention, use small Vision-State-Space
    (VSS) blocks whose selective scan runs along the *temporal* axis of a
    short aligned window (T = 3). Because T is tiny, the scan is a plain
    PyTorch loop with O(T) linear complexity -> real-time friendly on an
    RTX 3060 / Windows (no mamba-ssm compilation needed).

Blocks implemented here:
  * ConvResBlock            2D residual conv block used by the decoder
                             (paper keeps 2D ResBlocks, not 3D, in decoder).
  * SelectiveScanTemporal   input-dependent ("selective") SSM along the
                             temporal axis of a [B, C, T, H, W] tensor.
  * VSSBlock                LayerNorm -> 1x1 expand(2x) -> SiLU -> 3x3
                             depthwise conv -> temporal selective scan
                             (forward + backward) -> gate -> 1x1 + residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvResBlock(nn.Module):
    """2D residual conv block (MambaVF decoder refinement block)."""

    def __init__(self, dim, bias=False):
        super(ConvResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class SelectiveScanTemporal(nn.Module):
    """
    Mamba-style selective state-space scan along the temporal axis.

    Input x: [B, C, T, H, W]  (T = number of aligned frames in a window)

    For every timestep t and every token (channel c, pixel p) it performs
        A_bar_t = exp(dt_t * A_c)             # A = -exp(A_log) < 0, so decay in (0,1]
        h_t     = A_bar_t * h_{t-1} + (dt_t * B_t) * x_t
        y_t     = C_t^T h_t + D_c * x_t
    where dt / B / C are *input-dependent* (computed by cheap 1x1 convs),
    which is the "selective" property of Mamba. The scan runs both forward
    and backward over T (bidirectional, like the paper's STB mechanism),
    so the center frame receives context from both neighbours.
    """

    def __init__(self, dim, d_state=4, bias=False):
        super(SelectiveScanTemporal, self).__init__()
        self.dim = dim
        self.d_state = d_state
        # input-dependent discretization step dt (>=0 via softplus)
        self.dt_proj = nn.Conv2d(dim, 1, kernel_size=1, bias=True)
        # input-dependent B / C (per token, shared over channels as in Mamba)
        self.B_proj = nn.Conv2d(dim, d_state, kernel_size=1, bias=False)
        self.C_proj = nn.Conv2d(dim, d_state, kernel_size=1, bias=False)
        # per-channel log decay A = -exp(A_log)  -> A_log in R
        self.A_log = nn.Parameter(torch.randn(dim, d_state) * 0.2 - 1.0)
        # per-channel skip connection D (as in Mamba)
        self.D = nn.Parameter(torch.ones(dim))

    def _scan_dir(self, x):
        """
        x: [B, C, T, H, W]  ->  y: [B, C, T, H, W]
        One directional scan (t = 0 -> T-1).
        """
        B, C, T, H, W = x.shape
        N = B * H * W  # number of spatial tokens

        # flatten spatial tokens: [N, C, T]
        x_tok = x.permute(0, 3, 4, 1, 2).reshape(N, C, T)

        # input-dependent params, computed per (b, t, h, w) with 1x1 convs
        x4 = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
        dt = F.softplus(self.dt_proj(x4)).reshape(B, T, H, W)  # [B,T,H,W]
        Bp = self.B_proj(x4).reshape(B, T, H, W, self.d_state)  # [B,T,H,W,d]
        Cp = self.C_proj(x4).reshape(B, T, H, W, self.d_state)  # [B,T,H,W,d]

        A = -torch.exp(self.A_log)  # [C, d]  (always negative -> stable decay)

        h = x.new_zeros(N, C, self.d_state)
        ys = []
        for t in range(T):
            dt_t = dt[:, t].reshape(N)                          # [N]
            B_t = Bp[:, t].reshape(N, self.d_state)             # [N,d]
            C_t = Cp[:, t].reshape(N, self.d_state)             # [N,d]
            x_t = x_tok[:, :, t]                                # [N,C]

            A_bar = torch.exp(dt_t[:, None, None] * A.unsqueeze(0))  # [N,C,d]
            dB_x = dt_t[:, None] * B_t                          # [N,d]
            h = A_bar * h + dB_x[:, None, :] * x_t[:, :, None]  # [N,C,d]
            y_t = torch.einsum('nd,ncd->nc', C_t, h) + self.D.unsqueeze(0) * x_t  # [N,C]

            ys.append(y_t)

        y = torch.stack(ys, dim=-1)                            # [N,C,T]
        return y.view(B, H, W, C, T).permute(0, 3, 4, 1, 2)    # [B,C,T,H,W]

    def forward(self, x):
        fwd = self._scan_dir(x)
        bwd = self._scan_dir(torch.flip(x, dims=[2]))
        bwd = torch.flip(bwd, dims=[2])
        return fwd + bwd


class VSSBlock(nn.Module):
    """
    Lightweight Vision-State-Space block for a [B, C, T, H, W] tensor.

    Structure (adapted from VMamba / MambaVF VSS blocks):
        x + out_proj( gate * SelectiveScanTemporal( dwconv( silu( x ) ) ) )
    with a 1x1 expansion -> SiLU -> 3x3 depthwise conv before the scan and a
    channel LayerNorm at the entrance.
    """

    def __init__(self, dim, d_state=4, bias=False):
        super(VSSBlock, self).__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)                       # per-pixel channel LN
        self.in_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                padding=1, groups=dim, bias=bias)
        self.ssm = SelectiveScanTemporal(dim, d_state=d_state, bias=bias)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.act = nn.SiLU()   # out-of-place: x1/x2 are views of in_proj output

    def _norm_c(self, x):
        # x: [B, C, T, H, W] -> LayerNorm over the channel dim per pixel
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, C)
        x = self.norm(x)
        return x.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)

    def forward(self, x):
        skip = x
        B, C, T, H, W = x.shape
        x = self._norm_c(x)

        x4 = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)   # [B*T, C, H, W]
        x1, x2 = self.in_proj(x4).chunk(2, dim=1)               # [B*T, C, H, W]
        x1 = self.act(x1)
        x1 = self.dwconv(x1)
        x1 = self.act(x1)
        # back to 5D for the temporal scan
        x1 = x1.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)      # [B, C, T, H, W]
        x1 = self.ssm(x1)
        x2 = x2.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)      # [B, C, T, H, W]
        x = x1 * self.act(x2)
        # back to 4D for the output projection
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.out_proj(x)
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return x + skip
