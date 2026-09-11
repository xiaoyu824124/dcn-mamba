# -*- coding: utf-8 -*-
"""
benchmark.py - 在目标 GPU (RTX 3060 等) 上测量 DCN + MambaVF 风格融合模型的
参数 / 显存 / 延迟 / FPS。

两种实时口径：
  * full   : 一次前向输入 5 帧窗口 -> 输出 3 帧融合结果 (离线滑动窗口, stride=2)
  * stream : forward_stream, 输入最新 5 帧窗口 -> 只输出正中间 1 帧
             (在线逐帧出图, 衡量"每来一帧新图出一帧结果"的实时延迟)

用法示例:
  python benchmark.py --H 480 --W 640 --batch 1 --iters 50
  python benchmark.py --H 480 --W 640 --fp16
"""
import argparse
import time
import warnings

import torch
from omegaconf import OmegaConf

from src.model.net import VideoFusion

warnings.filterwarnings("ignore")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/train/ivf-train.yaml")
    p.add_argument("--H", type=int, default=480)
    p.add_argument("--W", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--fp16", action="store_true", help="use torch.cuda.amp autocast")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    model = VideoFusion(model_config={"model": cfg.model}).to(device).eval()
    n_params = sum(pp.numel() for pp in model.parameters())

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Params: {n_params/1e6:.3f} M")
    print(f"Input : [{args.batch}, 5, 3, {args.H}, {args.W}] x2 (IR + RGB)")

    H, W = args.H, args.W
    # 让 H/W 对齐编码器 1/4 分辨率 (stride-2 x2)
    H4 = (H + 1) // 4 * 4
    W4 = (W + 1) // 4 * 4
    if (H, W) != (H4, W4):
        print(f"NOTE: H/W 自动对齐到 4 的倍数: ({H},{W}) -> ({H4},{W4})")
        H, W = H4, W4

    x1 = torch.rand(args.batch, 5, 3, H, W, device=device)
    x2 = torch.rand(args.batch, 5, 3, H, W, device=device)

    def timeit(fn, n):
        for _ in range(args.warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if device.type == "cuda":
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(n):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / n  # ms
        t0 = time.time()
        for _ in range(n):
            fn()
        return (time.time() - t0) / n * 1000.0

    use_amp = args.fp16 and device.type == "cuda"

    def run_full():
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            model(x1, x2)

    def run_stream():
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            model.forward_stream(x1, x2)

    ms_full = timeit(run_full, args.iters)
    ms_stream = timeit(run_stream, args.iters)

    print(f"\n[full  ] {ms_full:7.2f} ms / 5帧窗口(出3帧)   -> {3.0/ms_full*1000:6.1f} 融合帧/s (离线,stride=2)")
    print(f"[stream] {ms_stream:6.2f} ms / 5帧窗口(出1帧)   -> {1.0/ms_stream*1000:6.1f} 融合帧/s (在线逐帧)")
    print(f"stream 帧延迟 = {ms_stream:.1f} ms (理论 30fps 预算 33.3ms, 25fps 预算 40ms)")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n峰值显存 (1次 full 前向 + warmup): {peak:.2f} GB")


if __name__ == "__main__":
    main()
