# -*- coding: utf-8 -*-
"""
Cross-modal registration module (adapted from IMF / MPDRNet)

来源：IMF, "Improving Misaligned Multi-modality Image Fusion with One-stage
      Progressive Dense Registration", IEEE TCSVT 2024
      代码：third_party/IMF/models/mpdrnet.py

本文件相对原版的改动：
  1. 去掉「伪红外」依赖 —— 原版要把 VIS 风格迁移成伪红外再配准；这里直接用 VIS
  2. 用真值形变场做 EPE 监督 —— 我们有 gt_h/ 真值，不需要靠相似度间接监督
  3. 去硬编码 —— 原版写死 shape=[256,256] 且 SpatialTransformer 里 .cuda()，
     这里改成任意分辨率 + 任意 device
  4. 支持多通道输入（IR 1通道 / VIS 3通道，内部转灰度）
  5. 预留时序接口（temporal 参数），当前版本先做单帧

核心结构（与原版一致）：
    双流编码器 → 瓶颈 → 逐尺度：预测 flow → DFF 融合多尺度 flow
                                    → warp 特征 → PFF 精修 → 预测更细的 flow
                          最后 DFF 融合全部尺度 → 最终形变场

约定：
    flow 是把 mov(IR) 对齐到 fix(VIS) 所需的稠密位移场，形状 [B,2,H,W]
    采样约定：warped(y,x) = mov(y + flow_y, x + flow_x)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 基础模块（与 IMF 保持一致）
# ----------------------------------------------------------------------
def conv(in_ch, out_ch, k, bias=False, stride=1):
    return nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=bias, stride=stride)


def predict_flow(in_ch, dim=2, k=3, bias=False, stride=1):
    return nn.Conv2d(in_ch, dim, k, padding=k // 2, bias=bias, stride=stride)


class ConvBnLeakyRelu2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1, stride=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1)


class TransConvBnLeakyRelu2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, stride=2, padding=1, dilation=1, groups=1):
        super().__init__()
        self.transconv = nn.ConvTranspose2d(in_ch, out_ch, k, padding=padding,
                                            stride=stride, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.transconv(x)), negative_slope=0.1)


class ConvSigmoid(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, padding=1, stride=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, n_feat, k, bias=True, bn=False, act=None):
        super().__init__()
        act = act if act is not None else nn.ReLU(True)
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, k, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x) + x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=bias))
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=bias))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.avg_fc(self.avg_pool(x)) + self.max_fc(self.max_pool(x)))


# ----------------------------------------------------------------------
# STN（去掉硬编码 .cuda()，支持任意 device 与分辨率）
# ----------------------------------------------------------------------
class SpatialTransformer(nn.Module):
    """用 grid_sample 做形变。grid 按需缓存，避免每次 forward 重建。"""

    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode
        self._grid = None
        self._shape = None

    def _get_grid(self, shape, device, dtype):
        if self._shape != tuple(shape) or self._grid is None or self._grid.device != device:
            vectors = [torch.arange(0, s, device=device, dtype=dtype) for s in shape]
            grids = torch.meshgrid(vectors, indexing="ij")   # 新版 torch 需要 indexing
            g = torch.stack(grids)                            # [2,H,W]
            g = g.unsqueeze(0)                                # [1,2,H,W]
            self._grid = g
            self._shape = tuple(shape)
        return self._grid

    def forward(self, src, flow):
        shape = flow.shape[2:]
        grid = self._get_grid(shape, flow.device, flow.dtype)
        new_locs = grid + flow

        # 归一化到 [-1,1]
        for i in range(len(shape)):
            new_locs[:, i] = 2.0 * (new_locs[:, i] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)      # [B,H,W,2]
        new_locs = new_locs[..., [1, 0]]             # (x,y) -> (x, y) 顺序调整
        out = F.grid_sample(src, new_locs, mode=self.mode,
                            padding_mode="zeros", align_corners=True)
        return out, new_locs


class VecInt(nn.Module):
    """向量积分（scaling and squaring），保证形变场平滑可逆。"""

    def __init__(self, nsteps=7):
        super().__init__()
        assert nsteps >= 0
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** nsteps)
        self.transformer = SpatialTransformer()

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)[0]
        return vec

# ----------------------------------------------------------------------
# 编码器（4 级金字塔，逐级 stride=2）
# ----------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=8, k=3):
        super().__init__()
        self.level3 = nn.Sequential(
            ConvBnLeakyRelu2d(in_ch, out_ch, k, stride=2))
        self.level2 = nn.Sequential(
            ConvBnLeakyRelu2d(out_ch, 2 * out_ch, k, stride=2),
            ResBlock(2 * out_ch, k, bias=True, act=nn.LeakyReLU(0.1)))
        self.level1 = nn.Sequential(
            ConvBnLeakyRelu2d(2 * out_ch, 4 * out_ch, k, stride=2),
            ResBlock(4 * out_ch, k, bias=True, act=nn.LeakyReLU(0.1)))
        self.level0 = nn.Sequential(
            ConvBnLeakyRelu2d(4 * out_ch, 8 * out_ch, k, stride=2),
            ResBlock(8 * out_ch, k, bias=True, act=nn.LeakyReLU(0.1)))

    def forward(self, x):
        e3 = self.level3(x)    # H/2
        e2 = self.level2(e3)   # H/4
        e1 = self.level1(e2)   # H/8
        e0 = self.level0(e1)   # H/16
        return [e3, e2, e1, e0]


# ----------------------------------------------------------------------
# PFF：渐进特征精修（三路特征 + softmax 权重 + 通道注意力）
# ----------------------------------------------------------------------
class PFF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        c = in_channels
        self.channels = c
        self.downconv_1 = nn.Sequential(conv(3 * c, c, 1), nn.ReLU(inplace=True))
        self.plainconv = nn.Sequential(conv(c, c, 3), nn.ReLU(inplace=True))
        self.downconv_2 = nn.Sequential(conv(c, 3, 3), nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=-1)
        self.ca_layer = ChannelAttention(3 * c)

    def forward(self, dec, mov_warp, fix_enc):
        """三路：解码特征 / warp 后的移动图特征 / 固定图特征"""
        x = torch.cat([dec, mov_warp, fix_enc], dim=1)
        f = self.plainconv(self.downconv_1(x))
        w_in = self.downconv_2(f)                                    # [B,3,H,W]
        w = self.softmax(w_in.view(w_in.shape[0], 3, -1))            # 空间维 softmax
        w = w.view_as(w_in)
        fms = torch.cat([dec * w[:, :1], mov_warp * w[:, 1:2], fix_enc * w[:, 2:]], dim=1)
        return self.ca_layer(fms) * fms


# ----------------------------------------------------------------------
# DFF：形变场融合（对多个尺度的粗形变场做重加权 + 向量积分）
#   注意：这里融合的是「形变场」，不是图像
# ----------------------------------------------------------------------
class DFF(nn.Module):
    def __init__(self, in_channels=2, list_num=4):
        super().__init__()
        self.channels = in_channels
        self.num = list_num
        self.exp = 16
        self.step = 7
        self.conv_1 = nn.Sequential(
            conv(self.num * self.channels, self.exp * self.channels, 3),
            nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(
            conv(self.exp * self.channels, self.exp * self.channels, 3),
            nn.ReLU(inplace=True))
        self.convsig = nn.Sequential(
            *[ConvSigmoid(self.exp * self.channels, self.channels, 3) for _ in range(self.num)])
        self.integrate = VecInt(nsteps=self.step)

    def forward(self, predict_flows):
        """
        predict_flows: list of [B,2,h_i,w_i]，从最粗尺度开始
        返回：融合后的形变场，分辨率 = flows[0] 的 2^num 倍
        """
        cache = []
        for i, flow in enumerate(predict_flows):
            s = 2 ** (self.num - i)
            cache.append(F.interpolate(flow, scale_factor=(s, s),
                                       mode="bilinear", align_corners=True))
        cat = torch.cat(cache, dim=1)
        w_cat = self.conv_2(self.conv_1(cat))

        field = None
        for i, flow in enumerate(cache):
            w = self.convsig[i](w_cat)
            term = flow * w
            field = term if field is None else field + term
        return self.integrate(field)

# ----------------------------------------------------------------------
# 主模块
# ----------------------------------------------------------------------
class CrossModalRegistration(nn.Module):
    """
    跨模态配准模块（MPDRNet 改造版）

    输入:
        mov : [B,C,H,W]  移动图（待对齐，例如红外）
        fix : [B,C,H,W]  固定图（参考，例如可见光）
    输出:
        warped_mov : [B,C,H,W]  对齐后的移动图
        flow       : [B,2,H,W]  形变场（把 mov 对齐到 fix）
        aux        : dict       中间量，用于多尺度监督与调试

    说明:
      · 内部按灰度处理（多通道会先转灰度），最后 warped_mov 按原通道数返回
      · H、W 需为 16 的倍数，不足会自动 pad 后再裁回
      · 用真值形变场做 EPE 监督时，直接用返回的 flow 即可
    """

    def __init__(self, channels=16, nsteps=7):
        super().__init__()
        c = channels
        self.channels = c

        # 双流编码器
        self.fix_encoder = Encoder(1, c, 3)
        self.mov_encoder = Encoder(1, c, 3)

        # 瓶颈
        self.bottle_1 = ConvBnLeakyRelu2d(16 * c, 8 * c, 3)
        self.bottle_2 = ConvBnLeakyRelu2d(8 * c, 8 * c, 3)

        # 解码器上采样
        self.upsample0 = TransConvBnLeakyRelu2d(8 * c, 4 * c, 4, 2, 1)
        self.upsample1 = TransConvBnLeakyRelu2d(12 * c, 2 * c, 4, 2, 1)
        self.upsample2 = TransConvBnLeakyRelu2d(6 * c, c, 4, 2, 1)

        # 逐尺度形变场预测
        self.predict_flow0 = predict_flow(8 * c, 2)
        self.predict_flow1 = predict_flow(12 * c, 2)
        self.predict_flow2 = predict_flow(6 * c, 2)
        self.predict_flow3 = predict_flow(3 * c, 2)

        # DFF（融合的形变场个数逐级增加）
        self.DFF_1 = DFF(list_num=1)
        self.DFF_2 = DFF(list_num=2)
        self.DFF_3 = DFF(list_num=3)
        self.DFF_4 = DFF(list_num=4)

        # PFF（通道数随分辨率升高而减少）
        self.PFF_1 = PFF(4 * c)
        self.PFF_2 = PFF(2 * c)
        self.PFF_3 = PFF(c)

        self.stn = SpatialTransformer()

    @staticmethod
    def _to_gray(x):
        if x.shape[1] == 1:
            return x
        return x.mean(dim=1, keepdim=True)

    @staticmethod
    def _pad_to(x, mult=16):
        _, _, h, w = x.shape
        ph = (mult - h % mult) % mult
        pw = (mult - w % mult) % mult
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return x, ph, pw

    def forward(self, mov, fix, aux_out=False):
        # 记录原始尺寸，必要时 padding 到 16 的倍数
        _, _, H0, W0 = mov.shape
        mov_p, ph, pw = self._pad_to(mov)
        fix_p, _, _ = self._pad_to(fix)
        H, W = mov_p.shape[2], mov_p.shape[3]

        mov_g = self._to_gray(mov_p)
        fix_g = self._to_gray(fix_p)

        # ---- 双流编码 ----
        mov_e3, mov_e2, mov_e1, mov_e0 = self.mov_encoder(mov_g)
        fix_e3, fix_e2, fix_e1, fix_e0 = self.fix_encoder(fix_g)

        # ---- 瓶颈 ----
        dec0 = self.bottle_2(self.bottle_1(torch.cat([mov_e0, fix_e0], dim=1)))

        flows = []
        # ---- 尺度 0（最粗，H/16）----
        flow0 = self.predict_flow0(dec0)
        flows.append(flow0)

        # ---- 尺度 1（H/8）----
        up0 = self.upsample0(dec0)
        phi1 = self.DFF_1(flows)
        warped1, _ = self.stn(mov_e1, phi1)
        dec1 = self.PFF_1(up0, warped1, fix_e1)
        flow1 = self.predict_flow1(dec1)
        flows.append(flow1)

        # ---- 尺度 2（H/4）----
        up1 = self.upsample1(dec1)
        phi2 = self.DFF_2(flows)
        warped2, _ = self.stn(mov_e2, phi2)
        dec2 = self.PFF_2(up1, warped2, fix_e2)
        flow2 = self.predict_flow2(dec2)
        flows.append(flow2)

        # ---- 尺度 3（H/2）----
        up2 = self.upsample2(dec2)
        phi3 = self.DFF_3(flows)
        warped3, _ = self.stn(mov_e3, phi3)
        dec3 = self.PFF_3(up2, warped3, fix_e3)
        flow3 = self.predict_flow3(dec3)
        flows.append(flow3)

        # ---- 最终融合（全分辨率）----
        flow = self.DFF_4(flows)

        # 裁回原始尺寸
        if ph or pw:
            flow = flow[:, :, :H0, :W0]
        warped_mov, _ = self.stn(mov, flow)

        if aux_out:
            aux = dict(flow0=flow0, flow1=flow1, flow2=flow2, flow3=flow3,
                       phi1=phi1, phi2=phi2, phi3=phi3)
            return warped_mov, flow, aux
        return warped_mov, flow


# ----------------------------------------------------------------------
# 损失
# ----------------------------------------------------------------------
def epe_loss(flow_pred, flow_gt, mask=None):
    """端点误差：预测形变场与真值形变场的 L1 距离（平均到像素）。"""
    d = (flow_pred - flow_gt).abs()
    if d.shape[1] == 2:
        d = d.sum(dim=1)               # dx + dy
    else:
        d = d.mean(dim=1)
    if mask is not None:
        d = d * mask
        return d.sum() / (mask.sum() + 1e-6)
    return d.mean()


def smooth_loss(flow, order=1):
    """形变场平滑正则（一阶差分），抑制高频抖动。"""
    if order == 1:
        dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
        dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
        return dx + dy
    dx2 = (flow[:, :, :, 2:] - 2 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]).abs().mean()
    dy2 = (flow[:, :, 2:, :] - 2 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :]).abs().mean()
    return dx2 + dy2


def temporal_smooth_loss(flow_t, flow_prev, order=1):
    """时序平滑：抑制逐帧形变场抖动。flow_t / flow_prev 均为 [B,2,H,W]。"""
    if order == 1:
        return (flow_t - flow_prev).abs().mean()
    return (flow_t - 2 * flow_prev + flow_prev).abs().mean()   # 占位，二阶需三帧