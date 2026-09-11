# -*- coding: utf-8 -*-
# Final Master Version: 2026-03-17
# Features: 5D Tensor Support, Symmetric Padding (No Green Border), Correct Folder Split

import os
import sys
import argparse
import logging
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

# 自动把当前工作目录加入环境变量，彻底解决 ModuleNotFoundError
sys.path.append(os.getcwd())

from src.dataset.base_two_modal_dataset import DatasetMode
from src.dataset import get_multi_frame_dataset
from src.model.net import VideoFusion
from src.util import metric 
from src.util.metric import MetricTracker, compute_metrics
from src.util.io import pred_2_8bit, save_image
from src.util.logging_util import eval_dic_to_text, setup_logging

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Model evaluation script")
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="latest")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--base_data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_save_vis", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    exp_dir = os.path.join("output", args.exp_path)
    ckpt_dir = os.path.join(exp_dir, "checkpoint")
    model_path = os.path.join(ckpt_dir, args.ckpt_path if args.ckpt_path else "latest")
    eval_dir = os.path.join(exp_dir, "test_results", args.ckpt_path if args.ckpt_path else "latest", args.dataset_name)
    os.makedirs(eval_dir, exist_ok=True)
    
    setup_logging(os.path.join(eval_dir, "test.log")) 
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # 开启 4090 硬件加速
    
    cfg = OmegaConf.load(os.path.join(exp_dir, "config.yaml"))

    model = VideoFusion(model_config={'model': cfg.model}).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=device))
    model.eval()

    # 数据集加载
    dataset_cfg_path = f"config/dataset/{args.task_name}/{args.dataset_name}/{args.dataset_name.lower()}_5-frame.yaml"
    dataset = get_multi_frame_dataset(OmegaConf.load(dataset_cfg_path), args.base_data_dir, mode=DatasetMode.TEST)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    eval_metrics_names = [str(m) for m in cfg.eval.eval_metrics]
    
    try:
        metric_tracker = MetricTracker(*eval_metrics_names)
    except TypeError:
        metric_tracker = MetricTracker(eval_metrics_names)

    metric_funcs = [getattr(metric, m) for m in eval_metrics_names]
    data_dict = {"UniVF": {}}

    print(f"🚀 Master Test Started: {args.dataset_name} | Batch Size: {args.batch_size}")

    for i, batch in enumerate(tqdm(dataloader, desc=f"Testing {args.task_name}-{args.dataset_name}")):
        with torch.no_grad():
            I_1 = batch["ir"].to(device)  # 原图 [B, 5, C, H, W]
            I_2 = batch["rgb"].to(device)
            
            b, f_in, c, h, w = I_1.shape 
            pad_factor = 64
            
            # === 🚀 修复绿边：计算对称 Padding ===
            pad_h_total = (pad_factor - h % pad_factor) % pad_factor
            pad_w_total = (pad_factor - w % pad_factor) % pad_factor
            
            pad_top = pad_h_total // 2
            pad_bottom = pad_h_total - pad_top
            pad_left = pad_w_total // 2
            pad_right = pad_w_total - pad_left

            if pad_h_total > 0 or pad_w_total > 0:
                # 使用 replicate 复制边缘像素，彻底杜绝 DCN 越界产生的黑边/绿边
                I_1_in = F.pad(I_1.view(-1, c, h, w), (pad_left, pad_right, pad_top, pad_bottom), mode='replicate').view(b, f_in, c, h+pad_h_total, w+pad_w_total)
                I_2_in = F.pad(I_2.view(-1, c, h, w), (pad_left, pad_right, pad_top, pad_bottom), mode='replicate').view(b, f_in, c, h+pad_h_total, w+pad_w_total)
            else:
                I_1_in, I_2_in = I_1, I_2

            # === 模型推理 ===
            fusion_pred, _ = model(I_1_in, I_2_in)
            
            # === 🚀 对称裁剪：切回原图尺寸 ===
            if pad_h_total > 0 or pad_w_total > 0:
                fusion_pred = fusion_pred[..., pad_top : pad_top + h, pad_left : pad_left + w]

            # === 循环拆解 Batch，同时保持 5D 骗过底层函数 ===
            for b_idx in range(b):
                # 保持 [1, T, C, H, W] 形状
                f_pred_5d = fusion_pred[b_idx:b_idx+1, ...] 
                f_i1_5d = I_1[b_idx:b_idx+1, ...]
                f_i2_5d = I_2[b_idx:b_idx+1, ...]

                # 安全获取文件路径
                try:
                    # 尝试取中心帧路径
                    img_path = batch["data_path_ls_dict"]["ir"][2][b_idx]
                except Exception:
                    img_path = batch["data_path_ls_dict"]["ir"][0][0]
                
                if isinstance(img_path, (list, tuple)):
                    img_path = img_path[0]

                # 🚀 修复文件夹全变 infrared 的 Bug (-3 拿序列名)
                dir_name = img_path.split('/')[-3]
                file_name = img_path.split('/')[-1]

                # 保存图片 (传入 5D)
                if not args.no_save_vis:
                    save_path = os.path.join(eval_dir, "eval_visual", dir_name, file_name)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    # f_i1_5d[..., :h, :w] 确保尺寸一致
                    img_8bit = pred_2_8bit(f_pred_5d, f_i1_5d[..., :h, :w], f_i2_5d[..., :h, :w])
                    save_image(img_8bit, save_path)

                # 计算指标 (原封不动传入 5D，让 compute_metrics 内部自己去切)
                res = compute_metrics(
                    metric_funcs, 
                    f_pred_5d, 
                    f_i1_5d[..., :h, :w], 
                    f_i2_5d[..., :h, :w]
                )

                if dir_name not in data_dict["UniVF"]:
                    data_dict["UniVF"][dir_name] = {k: [] for k in res.keys()}

                # 更新指标 (防撑爆内存)
                for k, v in res.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    metric_tracker.update(k, val, n=1)
                    data_dict["UniVF"][dir_name][k].append(val)

    # --- 最终打印并保存成绩单 ---
    final_res = metric_tracker.result()
    logging.info(f"Final: {final_res}")
    print("\n" + "="*40)
    print(f"📊 Final Results for {args.dataset_name}:")
    for k, v in final_res.items():
        print(f"  - {k}: {v:.4f}")
    print("="*40 + "\n")

    with open(os.path.join(eval_dir, f"eval-{args.dataset_name}.txt"), "w") as file_txt:
        file_txt.write(eval_dic_to_text(final_res, f"Dataset: {args.dataset_name}"))
    
    with open(os.path.join(eval_dir, f"eval-{args.dataset_name}.csv"), "w", newline="") as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(final_res.keys())
        writer.writerow(final_res.values())

if __name__ == "__main__":
    main()