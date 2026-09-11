import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from .fusion import Fusion_Net
from .utils import load_args_from_json

# ==========================================
# 核心工具：RGB 与 YCrCb 空间互转
# 用于彻底找回丢失的“红色”
# ==========================================
def rgb_to_ycrcb(rgb):
    # 输入 [B, 3, H, W], 范围 [0, 1]
    r = rgb[:, 0:1, :, :]
    g = rgb[:, 1:2, :, :]
    b = rgb[:, 2:3, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return torch.cat((y, cb, cr), dim=1)

def ycrcb_to_rgb(ycrcb):
    # 输入 [B, 3, H, W], Y在[0,1], Cb/Cr在[0,1]
    y = ycrcb[:, 0:1, :, :]
    cb = ycrcb[:, 1:2, :, :]
    cr = ycrcb[:, 2:3, :, :]
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    
    rgb = torch.cat((r, g, b), dim=1)
    
    # 🚀 核心修复：强制截断！大于 1.0 的全部按 1.0 算，绝不翻转！
    return torch.clamp(rgb, 0.0, 1.0)

# ==========================================
# DCN 特征级软对齐模块
# ==========================================
class DCNAlignment(nn.Module):
    def __init__(self, channels):
        super(DCNAlignment, self).__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 27, kernel_size=3, padding=1)
        )
        self.weight = nn.Parameter(torch.Tensor(channels, channels, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)

    def forward(self, feat_current, feat_previous):
        concat_feat = torch.cat([feat_current, feat_previous], dim=1)
        out = self.offset_conv(concat_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offsets = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        aligned_feat = deform_conv2d(input=feat_previous, offset=offsets, weight=self.weight, mask=mask, padding=1)
        return aligned_feat

# ==========================================
# 融合网络主干 (色彩解耦版)
# ==========================================
class VideoFusion(nn.Module):
    def __init__(self, model_config):
        super(VideoFusion, self).__init__()
        channels = model_config['model']['dim']
        self.dcn_align = DCNAlignment(channels=channels)
        self.output_mask = model_config['model']['output_mask']
        self.fusion_net = Fusion_Net(
            dim=model_config['model']['dim'],
            num_blocks=model_config['model']['num_blocks'],
            head=model_config['model']['head'],
            ffn_expansion_factor=model_config['model']['ffn_expansion_factor'],
            bias=model_config['model']['bias'],
            LayerNorm_type=model_config['model']['LayerNorm_type'],
            output_mask=model_config['model']['output_mask']
        )

    def extract_features(self, frames, index):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
        if index == 1:
            frames = self.fusion_net.encoder_1(frames)
        else:
            frames = self.fusion_net.encoder_2(frames)
        return frames.view(B, T, -1, H, W)

    def prepare_window_features(self, feats):
        B, T, C, H, W = feats.shape
        fused_features = []
        for i in range(3):
            f_prev, f_cur, f_nxt = feats[:, i], feats[:, i+1], feats[:, i+2]
            f_prev_a = self.dcn_align(f_cur, f_prev)
            f_nxt_a = self.dcn_align(f_cur, f_nxt)
            fused_features.append(torch.cat([f_prev_a, f_cur, f_nxt_a], dim=1))
        return fused_features

    def forward(self, sources_1, sources_2):
        # sources_1: 红外(IR), sources_2: 可见光(RGB)
        feat1 = self.extract_features(sources_1, index=1)
        feat2 = self.extract_features(sources_2, index=2)

        feat1_win = self.prepare_window_features(feat1)
        feat2_win = self.prepare_window_features(feat2)

        outputs = []
        for i in range(3):
            # 1. 得到模型生成的融合结果 
            fused_out = self.fusion_net(feat1_win[i], feat2_win[i])
            fused_y = fused_out[:, 0:1, :, :] # 只取第 0 个通道作为亮度 Y

            # 2. 拿到当前时刻的原始可见光图 (RGB)
            vi_rgb = sources_2[:, i + 1, :, :, :]

            # 3. 空间转换：RGB -> YCrCb
            vi_ycrcb = rgb_to_ycrcb(vi_rgb)

            # ====================================================
            # 🚀 终极净化：色度平滑 (Chroma Smoothing) + 柔和门控
            # ====================================================
            vi_y = vi_ycrcb[:, 0:1, :, :]
            vi_cb = vi_ycrcb[:, 1:2, :, :]
            vi_cr = vi_ycrcb[:, 2:3, :, :]
            
            # 【杀招 1】：物理抹平高频噪点。
            # 用 3x3 的均值滤波把孤立的蓝绿噪点“糊”成灰色，但不影响大面积的红色！
           #cb_smooth = F.avg_pool2d(vi_cb, kernel_size=3, stride=1, padding=1)
            #cr_smooth = F.avg_pool2d(vi_cr, kernel_size=3, stride=1, padding=1)
            
            # 【杀招 2】：更严苛的门控。
            # 剔除极暗处的微弱反光。只有可见光亮度超过 0.05 的地方，才允许出现颜色。
            color_weight = torch.clamp((vi_y - 0.05) * 2.5, 0.0, 1.0)
            
            # 清洗色彩通道
            clean_cb = (vi_cb - 0.5) * color_weight + 0.5
            clean_cr = (vi_cr - 0.5) * color_weight + 0.5

            # 4. 严丝合缝的拼接
            final_ycrcb = torch.cat((fused_y, clean_cb, clean_cr), dim=1)

            # 5. 转回 RGB 输出
            final_rgb = ycrcb_to_rgb(final_ycrcb)
            outputs.append(final_rgb.unsqueeze(1))

        return torch.cat(outputs, dim=1), None