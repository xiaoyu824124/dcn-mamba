# -*- coding: utf-8 -*-
# Final Optimized Version: 2026-03-17 (Added Light-weight Temporal Gradient Loss)

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import SSIMLoss
from math import exp

# ==========================================
# 工具函数：RGB 转换到 YCrCb
# ==========================================
def rgb2ycrcb(rgb_tensor):
    r = rgb_tensor[:, 0, :, :]
    g = rgb_tensor[:, 1, :, :]
    b = rgb_tensor[:, 2, :, :]
    
    # Conversion formula (ITU-R BT.601 standard)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 0.5  
    cb = (b - y) * 0.564 + 0.5  
    
    ycrcb_tensor = torch.stack([y, cr, cb], dim=1)
    ycrcb_tensor = torch.clamp(ycrcb_tensor, 0.0, 1.0)
    
    return ycrcb_tensor

# ==========================================
# Loss 工厂函数
# ==========================================
def get_loss(loss_name, **kwargs):
    if loss_name == "IVF_MVF_Loss":
        return IVF_MVF_Loss(**kwargs)
    elif loss_name == "MEF_Loss":
        return MEF_Loss(**kwargs)
    elif loss_name == "MFF_Loss":
        return MFF_Loss(**kwargs)   
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

# ==========================================
# 核心：红外与可见光融合 Loss (纯亮度 + 时序稳定版)
# ==========================================
class IVF_MVF_Loss(nn.Module):
    def __init__(self, coef, use_occlusion, occ_threshold, max_offset_mask):
        super().__init__()
        self.coef = tuple(coef)
        self.use_occlusion = use_occlusion
        self.occ_threshold = occ_threshold
        self.max_offset_mask = max_offset_mask
        # 兼容旧版参数，但这里已经不再依赖光流 warp
        from src.model.utils import flow_warp
        self.flow_warp = flow_warp

    def sobel_filter(self, tensor):
        # 强制在单通道上计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_x = sobel_x.to(tensor.device)
        sobel_y = sobel_y.to(tensor.device)
        
        if tensor.shape[1] > 1:
            tensor = tensor[:, :1, :, :] # 提取 Y 通道
            
        grad_x = F.conv2d(tensor, sobel_x, padding=1)
        grad_y = F.conv2d(tensor, sobel_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def compute_single_loss(self, f, a, b):
        # 确保输入是单通道 [B, 1, H, W]
        if f.dim() == 3: f = f.unsqueeze(1)
        if a.dim() == 3: a = a.unsqueeze(1)
        if b.dim() == 3: b = b.unsqueeze(1)
        
        grad_f = self.sobel_filter(f)
        grad_a = self.sobel_filter(a)
        grad_b = self.sobel_filter(b)
        
        # 亮度损失和梯度损失 (只算Y通道)
        loss_int = F.l1_loss(f, torch.max(a, b))
        loss_grad = F.l1_loss(grad_f, torch.max(grad_a, grad_b))
        
        # 彻底关闭颜色损失
        loss_color = 0.0
        
        # SSIM 损失 (只算Y通道)
        loss_ssim = 0.5 * SSIMLoss(11, reduction='mean')(f, a) + 0.5 * SSIMLoss(11, reduction='mean')(f, b)
        
        return loss_int, loss_grad, loss_color, loss_ssim

    def spatial_loss(self, f, a, b):
        B, T, C, H, W = f.shape
        
        # 切片操作会导致内存不连续，改用 reshape 完美解决！
        f_ycrcb = rgb2ycrcb(f.reshape(-1, 3, H, W)).reshape(B, T, 3, H, W)
        a_ycrcb = rgb2ycrcb(a.reshape(-1, 3, H, W)).reshape(B, T, 3, H, W)
        b_ycrcb = rgb2ycrcb(b.reshape(-1, 3, H, W)).reshape(B, T, 3, H, W)
        
        total_loss_int, total_loss_grad, total_loss_color, total_loss_ssim = 0.0, 0.0, 0.0, 0.0
        
        for i in range(T):
            # [:, i=frame, 0=Y通道, :, :]，只提取 Y 通道传入
            fi, ai, bi = f_ycrcb[:, i, 0, :, :], a_ycrcb[:, i, 0, :, :], b_ycrcb[:, i, 0, :, :]
            loss_int, loss_grad, loss_color, loss_ssim = self.compute_single_loss(fi, ai, bi)
            
            total_loss_int += loss_int
            total_loss_grad += loss_grad
            total_loss_color += loss_color 
            total_loss_ssim += loss_ssim
            
        return total_loss_int, total_loss_grad, total_loss_color, total_loss_ssim

    def temporal_loss(self, f, a, b):
        # 🚀 全新登场：【无光流·时序梯度损失】
        # 极低显存消耗，精准打击画面闪烁 (Flickering)
        B, T, C, H, W = f.shape
        
        # 1. 剥离色彩，只提取 3 帧的 Y 通道 (亮度)
        f_y = rgb2ycrcb(f.reshape(-1, 3, H, W)).reshape(B, T, 3, H, W)[:, :, 0:1, :, :]
        a_y = rgb2ycrcb(a.reshape(-1, 3, H, W)).reshape(B, T, 3, H, W)[:, :, 0:1, :, :]
        b_y = rgb2ycrcb(b.reshape(-1, 3, H, W)).reshape(B, T, 3, H, W)[:, :, 0:1, :, :]
        
        # 2. 计算相邻帧的亮度绝对差值 (当前帧 - 上一帧)
        diff_f = torch.abs(f_y[:, 1:] - f_y[:, :-1])
        diff_a = torch.abs(a_y[:, 1:] - a_y[:, :-1])
        diff_b = torch.abs(b_y[:, 1:] - b_y[:, :-1])
        
        # 3. 核心惩罚逻辑：向“原图的最大时序变化”看齐！
        loss_temp = F.l1_loss(diff_f, torch.max(diff_a, diff_b))
        
        return loss_temp

    def forward(self, f, a, b, flow_net=None):
        B, T, C, H, W = f.shape
        
        # 🚀 统一在这里进行切片对齐 (将 5 帧的原图砍成中间 3 帧，与预测图 f 完美对齐)
        if a.shape[1] > T:
            diff = (a.shape[1] - T) // 2
            a = a[:, diff : diff+T, :, :, :]
            b = b[:, diff : diff+T, :, :, :]

        # 1. 计算空间损失
        loss_int, loss_grad, loss_color, loss_ssim = self.spatial_loss(f, a, b)
        
        # 2. 计算时序损失 (直接调用我们新写的无光流版本，彻底摆脱 flow_net 依赖)
        if self.coef[3] > 0:
            loss_temp = self.temporal_loss(f, a, b)
        else:
            loss_temp = torch.tensor(0.0, device=f.device)

        # 3. 汇总总 Loss (依然保持色彩 loss_color = 0 的解耦特性)
        total_loss = self.coef[0] * (loss_int + loss_color) + self.coef[1] * loss_grad + self.coef[2] * loss_ssim + self.coef[3] * loss_temp
        
        return {"loss": total_loss, "loss_int": loss_int, "loss_grad": loss_grad, "loss_ssim": loss_ssim, "loss_temp": loss_temp}

# ==========================================
# 占位符：MEF / MFF 任务 (当前暂未使用)
# ==========================================
class MEF_Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, *args, **kwargs):
        return {"loss": torch.tensor(0.0, requires_grad=True).to(args[0].device)}

class MFF_Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, *args, **kwargs):
        return {"loss": torch.tensor(0.0, requires_grad=True).to(args[0].device)}