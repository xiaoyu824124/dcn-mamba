# DCN + MambaVF 风格融合模块（IVF 实时化改造）

> 参考论文：**MambaVF: State Space Model for Efficient Video Fusion** (arXiv:2602.06017)
> 本仓库原有融合模块是 Restormer 风格自注意力（`TransformerBlock`），参数量虽小，但
> 在 **全分辨率** 上对 5 帧 × 2 模态逐帧做多次 MDTA/GDFN，FLOPs 高、无法在 RTX 3060 上实时。
> 本次改造：**保留 DCN 特征级软对齐**，把融合模块替换为 MambaVF 思想的轻量 SSM 融合模块。

---

## 1. 对应论文的核心思想

| MambaVF 论文做法 | 本仓库实现 |
| --- | --- |
| 视频融合建模成“沿帧的隐状态更新”（Eq.2 选择性状态空间） | `SelectiveScanTemporal`：沿对齐后 3 帧做双向选择性扫描（纯 PyTorch，无 mamba-ssm / CUDA 扩展） |
| patch/tubelet embedding，避免全分辨率自注意力 | 编码器用 stride-2 卷积把主干降到 1/4 分辨率（`downscale: 2`），Y 最后上采样回原分辨率 |
| 双流编码器 + 通道拼接（Eq.3）+ decoder 2D 残差块 | `encoder_1/2` + `torch.cat([m1,m2])` + `ConvResBlock` decoder |
| 每源 3 个 VSS 块、dim=32 | `num_blocks: 3`、`dim: 32`（可调） |
| 论文消融 Exp.II：1D temporal-only scan 能捕捉运动 | 我们的时间轴 SSM 正属于这一类；空间错位由保留的 **DCN** 负责，两者互补 |

与论文唯一有意保留的差异：论文完全去掉了显式对齐（flow-free），而按你的要求我们**保留 DCN**
（deformable-conv 软对齐），用于在 1/4 分辨率特征上把邻帧特征对齐到当前帧，再由时间 SSM 做跨帧状态融合。

## 2. 改动文件

- `src/model/mamba_block.py`（新）：`ConvResBlock` / `SelectiveScanTemporal` / `VSSBlock`
- `src/model/fusion.py`（重写）：`Fusion_Net`，对外接口不变
  - `encoder_1/encoder_2`：每帧特征提取（带下采样）
  - `forward(features1, features2)`：`[B,3C,H',W'] ×2 -> [B,out_ch,H',W']`
- `src/model/net.py`：**DCNAlignment 与 YCrCb 逻辑原样保留**；新增
  - 低分辨率 Y 上采样回原分辨率（`_upsample_to`）
  - `forward_stream()`：实时逐帧模式，只输出 5 帧窗口正中间 1 帧
- `config/train/ivf-train.yaml`：模型配置 + 3060 训练建议注释
- `benchmark.py`（新）：测参数量 / 显存 / 延迟 / FPS
- `_backup_20260906/`：改造前的 `net.py / fusion.py / mamba_block.py / ivf-train.yaml` 备份

> ⚠️ 旧 checkpoint 与 Restormer 版权重**不兼容**，改造后需要重新训练。

## 3. 模型配置（config/train/ivf-train.yaml）

```yaml
model:
  align_mode: dcn        # 保留 DCN 软对齐
  dim: 32
  num_blocks: 3          # 每条模态时间 VSS(SSM) 块数
  downscale: 2           # 编码器 stride-2 次数 => 主干 1/4 分辨率（实时性关键）
  enc_blocks: 1          # 下采样后每帧轻量残差块
  dec_blocks: 2          # decoder 2D 残差块
  ssm_d_state: 4         # 时间选择性扫描状态维数
  output_mask: false
```

约 **0.17M** 参数（原 Restormer 版约 0.31M）。

## 4. 实时性 & 使用

```bash
# 1) 训练（在 3060 上，参考 config 注释可把 random_crop_hw 提到 192/256）
python train.py --task_name IVF --base_data_dir <数据根目录>

# 2) 测速（推荐先 warmup，再多次取平均；--fp16 用 AMP 推理）
python benchmark.py --H 480 --W 640 --batch 1 --iters 100 --fp16

# 3) 离在线 demo（兼容原 test_demo.py，输出仍为 [B,3,3,H,W] 全分辨率）
python test_demo.py --task_name IVF --dataset_name VTMOT-demo --exp_path <新训练exp> ...
```

实时口径（`benchmark.py` 会同时打印）：
- `full`：一次前向 = 5 帧窗口出 3 帧 → 用于离线滑动窗口，吞吐 = 3 / latency_full
- `stream`：`forward_stream()` = 最新 5 帧窗口出 1 帧 → 在线逐帧延迟

再进一步提速的方向（代码已预留、可按需加）：
- 逐帧滑动窗口时缓存上一窗口已算好的 4/5 帧特征与 DCN 对齐结果（只算新增 1 帧），可再省 ~60-80%；
- 3060 上开 `torch.backends.cudnn.benchmark=True` + AMP(fp16)；
- 对 decoder 输出 Y 后接一层 1/2 分辨率的轻量 refine 卷积（当前直接用 bilinear 上采样，最快）。

## 5. 说明 / 局限

- 时间选择性扫描目前只在 **时间轴 (T=3)** 上做双向 SSM（论文 Eq.2 + STB 的时间维部分）；
  2D/8-way 空间扫描需要 mamba-ssm 类 CUDA 算子，Windows 上不便编译，故未引入——
  空间建模由 dw-conv + DCN + decoder 残差块承担，换来的是“任何显卡都能直接跑”。
- 训练/推理对输入分辨率无硬性要求（编码器 stride 只要求 H/W 是偶数倍数即可，
  demo 脚本已按 64 对齐 + replicate padding）。
