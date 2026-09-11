# -*- coding: UTF-8 -*-
# Final Defensive Demo Version: 2026-03-17 
# Features: Batch Acceleration, Symmetric Padding, 5D Tensor Support
# **NEW**: Defensive Edge Cropping to Eliminate Source Artifacts

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import logging
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

# 自动把当前工作目录加入环境变量，彻底解决 ModuleNotFoundError
sys.path.append(os.getcwd())

from src.dataset.base_two_modal_dataset import DatasetMode
from src.dataset import get_multi_frame_dataset
from src.model.net import VideoFusion
from src.util.io import pred_2_8bit, save_image
from src.util.logging_util import setup_logging
from src.util.video_util import generate_video_from_image_paths

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Model testing script")
    parser.add_argument("--exp_path", type=str, default='UniVF-MEF', help="Experiment directory")
    parser.add_argument("--task_name", type=str, default="MEF", help="Task name (IVF, MEF, MFF, MVF)")
    parser.add_argument("--dataset_name", type=str, default="YouTube-demo", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--fps", type=int, default=24, help="Frame rate for videos")
    parser.add_argument("--bitrate", type=int, default=50000)
    # 🚀 增加防御性裁剪参数，默认裁掉四周的 1 像素脏数据
    parser.add_argument("--edge_crop", type=int, default=1, help="Pixels to crop from edges")
    return parser.parse_args()

def main():
    args = parse_args()

    setup_logging(
        os.path.join("output_demo", args.task_name, args.dataset_name),
        file_log_level="INFO",
        console_log_level=args.log_level.upper(),
    )
    logging.info(f"Testing script started with args: {args}")

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True # 开启硬件加速
    
    if args.task_name == "IVF":
        args.config = "config/train/ivf-train.yaml"
        if args.dataset_name == "VTMOT":
            args.data_cfg = "config/dataset/IVF/VTMOT/vtmot_5-frame.yaml"
        elif "demo" in args.dataset_name:
            args.data_cfg = "config/dataset/demo/vtmot_demo_5-frame.yaml"
        else:
            raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")
    elif args.task_name == "MEF":
        args.config = "config/train/mef-train.yaml"
        args.data_cfg = f"config/dataset/MEF/YouTubeHDR/{args.dataset_name.lower().replace('-', '_')}_5-frame.yaml"
    elif args.task_name == "MFF":
        args.config = "config/train/mff-train.yaml"
        args.data_cfg = f"config/dataset/demo/davis_demo_5-frame.yaml" if "demo" in args.dataset_name else "config/dataset/MFF/DAVIS/davis_5-frame.yaml"
    elif args.task_name == "MVF":
        args.config = "config/train/mvf-train.yaml"
        args.data_cfg = f"config/dataset/demo/harvard_demo_5-frame.yaml" if "demo" in args.dataset_name else "config/dataset/MVF/Harvard/harvard_5-frame.yaml"
    
    cfg = OmegaConf.load(args.config)
    model_path = os.path.join("output", args.exp_path, "checkpoint", "latest", "model.pth")

    logging.info("Initializing model...")
    model = VideoFusion(model_config={'model': cfg.model}).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))

    logging.info(f"Loading test dataset: {args.data_cfg}")
    dataset = get_multi_frame_dataset(OmegaConf.load(args.data_cfg), base_data_dir="data", mode=DatasetMode.TEST)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    vis_dir = os.path.join("output_demo", args.task_name, args.dataset_name)
    os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Testing {args.dataset_name}")):
            if args.task_name == "IVF":
                I_1 = batch["ir"].to(device)
                I_2 = batch["rgb"].to(device)
            elif args.task_name == "MEF":
                I_1 = batch["under"].to(device)
                I_2 = batch["over"].to(device)
            elif args.task_name == "MFF":
                I_1 = batch["far"].to(device)
                I_2 = batch["near"].to(device)
            elif args.task_name == "MVF":
                I_1 = batch["mri"].to(device)
                I_2 = batch["other"].to(device)

            # === 对称 Padding (处理网络计算越界) ===
            pad_factor = 64
            b, f, c, h, w = I_1.shape 
            
            pad_h_total = (pad_factor - h % pad_factor) % pad_factor
            pad_w_total = (pad_factor - w % pad_factor) % pad_factor
            
            pad_top = pad_h_total // 2
            pad_bottom = pad_h_total - pad_top
            pad_left = pad_w_total // 2
            pad_right = pad_w_total - pad_left

            if pad_h_total > 0 or pad_w_total > 0:
                I_1_in = F.pad(I_1.view(-1, c, h, w), (pad_left, pad_right, pad_top, pad_bottom), mode='replicate').view(b, f, c, h+pad_h_total, w+pad_w_total)
                I_2_in = F.pad(I_2.view(-1, c, h, w), (pad_left, pad_right, pad_top, pad_bottom), mode='replicate').view(b, f, c, h+pad_h_total, w+pad_w_total)
            else:
                I_1_in, I_2_in = I_1, I_2

            fusion_pred, _ = model(I_1_in, I_2_in)
            
            # === 拆卸护甲：恢复原尺寸 ===
            if pad_h_total > 0 or pad_w_total > 0:
                fusion_pred = fusion_pred[..., pad_top : pad_top + h, pad_left : pad_left + w]

            # === 🚀 NEW: 防御性边缘裁剪 (处理原始数据脏边) 🚀 ===
            if args.edge_crop > 0:
                c_crop = args.edge_crop
                # 从画面四周齐刷刷裁掉 c_crop 个像素
                fusion_pred = fusion_pred[..., c_crop : h - c_crop, c_crop : w - c_crop]
                I_1 = I_1[..., c_crop : h - c_crop, c_crop : w - c_crop]
                I_2 = I_2[..., c_crop : h - c_crop, c_crop : w - c_crop]

            # === 循环拆解 Batch ===
            for b_idx in range(b):
                f_pred_5d = fusion_pred[b_idx:b_idx+1, ...]
                f_i1_5d = I_1[b_idx:b_idx+1, ...]
                f_i2_5d = I_2[b_idx:b_idx+1, ...]

                middle_idx = f_i1_5d.shape[1] // 2
                
                try:
                    if args.task_name == "IVF":
                        ir_path = batch["data_path_ls_dict"]["ir"][middle_idx][b_idx]
                    elif args.task_name == "MEF":
                        ir_path = batch["data_path_ls_dict"]["under"][middle_idx][b_idx]
                    elif args.task_name == "MFF":
                        ir_path = batch["data_path_ls_dict"]["far"][middle_idx][b_idx]
                    elif args.task_name == "MVF":
                        ir_path = batch["data_path_ls_dict"]["mri"][middle_idx][b_idx]
                except Exception:
                    ir_path = batch["data_path_ls_dict"]["ir"][0][0]

                if isinstance(ir_path, (list, tuple)):
                    ir_path = ir_path[0]

                # 统一将 Windows 反斜杠替换为标准正斜杠，兼容 Windows/Linux
                norm_path = ir_path.replace("\\", "/")
                dir_name = norm_path.split("/")[-3]
                file_name = os.path.splitext(norm_path.split("/")[-1])[0] + ".png"

                save_path = os.path.join(vis_dir, dir_name)
                os.makedirs(save_path, exist_ok=True)
                
                # 存图，传进去的已经是去除了脏边的干净矩阵了
                vis_8bit = pred_2_8bit(f_pred_5d, f_i1_5d, f_i2_5d)
                save_image(vis_8bit, os.path.join(save_path, file_name))

    # --- 视频合成阶段 ---
    for dir_name in os.listdir(vis_dir):
        dir_path = os.path.join(vis_dir, dir_name)
        if os.path.isdir(dir_path):
            logging.info(f"Merging images into video for: {dir_name}")
            filenames = sorted(glob.glob(dir_path + "/*.png", recursive=False))
            if len(filenames) > 0:
                logging.info(f"Found {len(filenames)} images for {dir_name}")
                output_video_path = os.path.join(vis_dir, f"{dir_name}.mp4")
                generate_video_from_image_paths(
                    image_paths=filenames,
                    output_path=output_video_path,
                    fps=args.fps,
                    bitrate=args.bitrate,
                    verbose=True,
                )

if __name__ == "__main__":
    main()