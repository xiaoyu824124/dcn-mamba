"""Linear self/cross attention for 1/8 cross-modal features.

The SA-CA ordering and ELU+1 attention follow CRFT's coarse feature
transformation. Residual gains start at zero so a coarse checkpoint can be
warm-started without changing its predictions before training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sine_position(height: int, width: int, channels: int,
                  reference: torch.Tensor) -> torch.Tensor:
    """Deterministic 2D position encoding with shape [1,H*W,C]."""
    if channels % 4:
        raise ValueError("coarse attention channels must be divisible by four")
    quarter = channels // 4
    y = torch.linspace(-math.pi, math.pi, height, device=reference.device)
    x = torch.linspace(-math.pi, math.pi, width, device=reference.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    frequency = torch.exp(-math.log(10000.0) *
                          torch.arange(quarter, device=reference.device) / max(quarter - 1, 1))
    phase_y = yy.reshape(-1, 1) * frequency
    phase_x = xx.reshape(-1, 1) * frequency
    position = torch.cat((phase_y.sin(), phase_y.cos(),
                          phase_x.sin(), phase_x.cos()), dim=-1)
    return position.unsqueeze(0).to(dtype=reference.dtype)


class LinearAttention(nn.Module):
    """ELU+1 kernel attention with O(N*C^2) memory and computation."""

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels < 1 or num_heads < 1 or channels % num_heads:
            raise ValueError("channels must be a positive multiple of num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = channels // num_heads
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if target.ndim != 3 or source.ndim != 3 or target.shape[0] != source.shape[0]:
            raise ValueError("attention expects [B,N,C] sequences with equal batch size")
        batch, target_length, channels = target.shape
        source_length = source.shape[1]
        if source.shape[2] != channels:
            raise ValueError("attention sequences must have equal channel counts")
        shape_q = (batch, target_length, self.num_heads, self.head_dim)
        shape_k = (batch, source_length, self.num_heads, self.head_dim)
        query = F.elu(self.query(target).reshape(shape_q).float()) + 1.0
        key = F.elu(self.key(source).reshape(shape_k).float()) + 1.0
        value = self.value(source).reshape(shape_k).float()
        # Scaling values before reduction keeps FP16 training safe when N is
        # large; all reductions themselves run in FP32.
        normalizer = max(source_length, 1)
        summary = torch.einsum("bshd,bshe->bhde", key, value / normalizer)
        denominator = torch.einsum("bnhd,bhd->bnh", query, key.sum(dim=1)).clamp_min(1e-6)
        message = torch.einsum("bnhd,bhde->bnhe", query, summary)
        message = message * (normalizer / denominator).unsqueeze(-1)
        return self.output(message.reshape(batch, target_length, channels).to(target.dtype))


class _Message(nn.Module):
    def __init__(self, channels: int, num_heads: int, ffn_ratio: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.source_norm = nn.LayerNorm(channels)
        self.attention = LinearAttention(channels, num_heads)
        self.fuse = nn.Sequential(nn.Linear(2 * channels, ffn_ratio * channels),
                                  nn.GELU(), nn.Linear(ffn_ratio * channels, channels),
                                  nn.LayerNorm(channels))

    def forward(self, target: torch.Tensor, source: torch.Tensor,
                position: torch.Tensor) -> torch.Tensor:
        message = self.attention(self.query_norm(target + position),
                                 self.source_norm(source + position))
        return self.fuse(torch.cat((target, message), dim=-1))


class _SACABlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, ffn_ratio: int) -> None:
        super().__init__()
        self.self_message = _Message(channels, num_heads, ffn_ratio)
        self.cross_message = _Message(channels, num_heads, ffn_ratio)
        self.self_gain = nn.Parameter(torch.zeros(()))
        self.cross_gain = nn.Parameter(torch.zeros(()))

    def forward(self, ir: torch.Tensor, vi: torch.Tensor,
                position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ir_self = self.self_message(ir, ir, position)
        vi_self = self.self_message(vi, vi, position)
        ir = ir + self.self_gain * ir_self
        vi = vi + self.self_gain * vi_self
        # Both updates use the same pre-cross pair to avoid an order-dependent
        # IR/VI asymmetry in the matching features.
        ir_cross = self.cross_message(ir, vi, position)
        vi_cross = self.cross_message(vi, ir, position)
        return ir + self.cross_gain * ir_cross, vi + self.cross_gain * vi_cross


class CoarseSACATransformer(nn.Module):
    """Apply alternating linear SA and CA to paired 1/8 feature maps."""

    def __init__(self, channels: int, num_layers: int = 2,
                 num_heads: int = 4, ffn_ratio: int = 2) -> None:
        super().__init__()
        if channels % 4 or num_layers < 1 or ffn_ratio < 1:
            raise ValueError("channels must be divisible by four; layers and ratio must be positive")
        self.blocks = nn.ModuleList(_SACABlock(channels, num_heads, ffn_ratio)
                                    for _ in range(num_layers))

    def forward(self, feature_ir: torch.Tensor,
                feature_vi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_ir.ndim != 4 or feature_ir.shape != feature_vi.shape:
            raise ValueError("SA-CA expects equal [B,C,H,W] IR/VI features")
        batch, channels, height, width = feature_ir.shape
        position = sine_position(height, width, channels, feature_ir)
        ir = feature_ir.flatten(2).transpose(1, 2)
        vi = feature_vi.flatten(2).transpose(1, 2)
        for block in self.blocks:
            ir, vi = block(ir, vi, position)
        return (ir.transpose(1, 2).reshape(batch, channels, height, width),
                vi.transpose(1, 2).reshape(batch, channels, height, width))
