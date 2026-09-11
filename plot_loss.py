import argparse
import re
import matplotlib.pyplot as plt
import os

def plot_active_losses(log_path):
    if not os.path.exists(log_path):
        print(f"❌ 错误: 找不到指定的日志文件 -> {log_path}")
        return

    print(f"✅ 正在解析日志文件: {log_path}")
    iters, losses, loss_int, loss_grad, loss_ssim, loss_temp = [], [], [], [], [], []

    # 🚀 升级正则表达式：精准提取 5 个核心指标 (加入了 loss_temp)
    pattern = re.compile(r"iter\s+(\d+).*?loss=([\d.]+),\s*loss_int=([\d.]+),\s*loss_grad=([\d.]+),\s*loss_ssim=([\d.]+),\s*loss_temp=([\d.]+)")

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                iters.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                loss_int.append(float(match.group(3)))
                loss_grad.append(float(match.group(4)))
                loss_ssim.append(float(match.group(5)))
                loss_temp.append(float(match.group(6)))

    if not iters:
        print("❌ 提取失败: 日志文件中没有找到有效的 Loss 数据。请检查正则表达式或日志内容。")
        return

    # 开始绘图 (改成了 3 个子图，让 loss_temp 单独显示，因为它的量级比其他的要小得多)
    plt.figure(figsize=(14, 12))
    plt.suptitle(f'Training Loss Curves\n(Log: {os.path.basename(os.path.dirname(log_path))})', fontsize=16, fontweight='bold')

    # 【上】：Total Loss
    plt.subplot(3, 1, 1)
    plt.plot(iters, losses, label='Total Loss', color='red', linewidth=2)
    plt.title('Total Loss', fontsize=12)
    plt.ylabel('Loss Value')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 【中】：空间子 Loss (Int, Grad, SSIM)
    plt.subplot(3, 1, 2)
    plt.plot(iters, loss_int, label='Loss Int (Intensity)', color='blue', alpha=0.8)
    plt.plot(iters, loss_grad, label='Loss Grad (Gradient)', color='green', alpha=0.8)
    plt.plot(iters, loss_ssim, label='Loss SSIM (Structure)', color='purple', alpha=0.8)
    plt.title('Spatial Sub-Losses', fontsize=12)
    plt.ylabel('Loss Value')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 🚀 【下】：时序子 Loss (Temp)
    plt.subplot(3, 1, 3)
    plt.plot(iters, loss_temp, label='Loss Temp (Temporal Stability)', color='darkorange', linewidth=2)
    plt.title('Temporal Sub-Loss (Anti-Flickering)', fontsize=12)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss Value')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 自动根据实验名生成图片名字
    exp_name = os.path.basename(os.path.dirname(log_path))
    save_name = f'loss_curves_with_temp_{exp_name}.png'
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_name, dpi=300)
    print(f"🎉 绘图成功！图片已保存为: {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='绘制指定日志文件的 Loss 曲线 (包含 loss_temp)')
    parser.add_argument('--log', type=str, required=True, help='train.log 文件的完整路径')
    args = parser.parse_args()
    
    plot_active_losses(args.log)