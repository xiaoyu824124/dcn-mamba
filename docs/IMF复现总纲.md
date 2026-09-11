# IMF 复现总纲

> 论文：Improving Misaligned Multi-modality Image Fusion with One-stage Progressive
> Dense Registration（**IEEE TCSVT 2024, 一区**，arXiv 2308.11165）
> 代码：https://github.com/wdhudiekou/IMF（已克隆到 `third_party\IMF\`）

---

## 一、总体路线图

```
┌─ 阶段 0：环境 + 资源（约 1 周）───────────────────────┐
│  · 建环境（3090）                                     │
│  · 下载：伪红外图像 / 预训练权重 / 数据集              │
│  · 跑通官方 test_reg.py                               │
└──────────────────────────────────────────────────────┘
                      ↓
┌─ 阶段 1：复现 IMF（约 1-2 周）───────────────────────┐
│  · 复现官方配准结果（MEE / 形变场可视化）              │
│  · 复现官方融合结果                                   │
│  · 整理成你自己的评测格式                             │
└──────────────────────────────────────────────────────┘
                      ↓
┌─ 阶段 2：迁移到你的数据（约 1 周）───────────────────┐
│  · 用 data\VTMOT_misaligned 跑 IMF 的 MPDR            │
│  · 接上你已有的 eval_registration.py 测 MEE           │
│  → 得到「图像级 IMF 在你的数据上的基线」              │
└──────────────────────────────────────────────────────┘
                      ↓
┌─ 阶段 3：加时序 —— 你的核心创新（约 2-3 周）─────────┐
│  · DFF → TFF：融合「多尺度 × 多帧」形变场            │
│  · PFF → 时序特征精修（复用你的 SelectiveScanTemporal）│
│  · 加时序平滑损失                                     │
│  · 消融：只 TFF / 只时序 PFF / 都加                   │
└──────────────────────────────────────────────────────┘
                      ↓
┌─ 阶段 4：完整实验（约 4-6 周）───────────────────────┐
│  · 多数据集（VTMOT + RoadScene + TNO）                │
│  · 多基线（SIFT / identity / static / C2RF / MINIMA） │
│  · 接你自己的 Mamba 融合，做端到端                    │
│  · 下游任务（检测或分割）                             │
└──────────────────────────────────────────────────────┘
```

---

## 二、资源清单（复现前必须先确认能下到）

| 资源 | 用途 | 位置 | 必需 |
|------|------|------|:---:|
| **伪红外图像（CPSTN 生成）** | **训练 C-MPDR 的前置** | 百度网盘 code `qqyj` | ★★★ |
| **MPDR 预训练权重** | 配准（RoadScene/TNO/M3FD/MSIFT）| 百度网盘 + Google Drive | ★★★ |
| TCF 预训练权重 | 融合（RoadScene/TNO/M3FD）| 百度网盘 + Google Drive | ★★ |
| RoadScene / TNO / M3FD | 官方复现数据集 | 官方链接（GitHub / figshare） | ★★ |
| **你自己的 VTMOT 数据** | 最终目标 | ✅ 已有 | ★★★ |

**⚠️ 最大的前置风险：伪红外图像**

IMF 的配准训练依赖 **UMF 生成的伪红外图像**（把 VIS 风格迁移成 IR）。
- 下载：百度网盘 `https://pan.baidu.com/s/1M79RuHVe6udKhcJIA7yXgA` code `qqyj`
- 关联项目：https://github.com/wdhudiekou/UMF-CMGR（IJCAI 2022）

**如果这个下不到，C-MPDR 就没法按原样训练。** 必须先确认。

---

## 三、环境配置（在 3090 上）

### 3.1 关于原版依赖

README 要求：
```
CUDA 10.1 / Python 3.6 / PyTorch 1.6.0 / Torchvision 0.7.0 / OpenCV 3.4 / Kornia 0.5.11
```

**⚠️ 这套环境在 3090 上跑不了。** 原因：

> PyTorch 1.6 编译时支持到 `sm_75`，**不含 `sm_86`**（3090 的算力代号）。
> 所以必须升级 PyTorch。

### 3.2 推荐环境（复制即用）

```bash
conda create -n imf python=3.8 -y
conda activate imf

# PyTorch：支持 sm_86（3090）
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117

# 关键：kornia 必须 < 0.7（见下方说明）
pip install "kornia==0.6.12"

# 其余依赖
pip install opencv-python numpy scipy matplotlib tqdm scikit-image
```

### 3.3 为什么 kornia 必须 < 0.7

IMF 的代码里用了：

```python
kornia.augmentation.RandomAffine(..., return_transform=True)
```

**`return_transform` 这个参数在 kornia 0.7.0 被移除了。** 用 0.7+ 会直接报错。

所以锁定 `kornia==0.6.12`（0.6.x 最后一个版本）。

### 3.4 验证环境

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); a=torch.randn(512,512,device='cuda'); print((a@a).sum().item())"
```

能打印出数字 = 环境 OK。

---

## 四、显卡选择

| 卡 | 算力代号 | 能否跑原版 torch 1.6 | 推荐 |
|---|:---:|:---:|:---:|
| **RTX 3090** | sm_86 | ❌ 需升级 torch | ★ **推荐** |
| RTX 4090 | sm_89 | ❌ | 可用（更快）|
| RTX 3060 12G | sm_86 | ❌ | 可用（慢，但显存够）|
| RTX 5060 8G | sm_120 | ❌ | **不推荐**（显存吃紧）|

**结论：用 3090。**
- 24GB 显存：够跑 IMF 的联合训练（`train_reg_co_fusion_sa.py`）
- sm_86：PyTorch 1.13 / 2.x 都完整支持，零折腾
- 速度：约为 4090 的一半，但足够

**为什么不用 5060**：8GB 显存跑 IMF 的多尺度配准 + 融合联合训练会很紧张。

---

## 五、复现步骤

### 步骤 1：数据准备

```bash
cd third_party/IMF/data
python generate_affine_deform_data.py     # 生成错位数据
python get_svs_map_softmax.py             # 生成显著性图（融合训练用）
```

### 步骤 2：跑通官方配准

```bash
cd third_party/IMF/Test
python test_reg.py
```

对照 README 里的"Experimental Results → Registration"下载官方结果，比对是否一致。

### 步骤 3：三种训练模式（按需）

| 模式 | 命令 | 说明 |
|------|------|------|
| 只训配准 | `Trainer/train_reg.py` | **你最先需要这个** |
| 分别训 | `train_reg.py` + `train_co_fuse.py` | 配准和融合解耦 |
| 联合训 | `train_reg_co_fusion_sa.py` | 端到端 |

### 步骤 4：迁移到你的数据

```
你的数据：E:\download\源码\VF-Bench-main\data\VTMOT_misaligned\
  红外 + 错位可见光 + 真值 gt_h/
      ↓
  喂给 IMF 的 MPDR 做配准
      ↓
  输出形变场 → 转成单应矩阵 → 喂给 eval_registration.py
      ↓
  MEE / PSNR / SSIM 对比表
```

---

## 六、验收标准

| 阶段 | 验收标准 |
|------|---------|
| 阶段 0 | `test_reg.py` 能跑出结果 |
| 阶段 1 | 复现的数字与官方论文接近（±10% 以内）|
| 阶段 2 | 在你的 VTMOT 数据上跑出 MEE，**优于 identity(8.71) 和 static(12.71)** |
| 阶段 3 | 加时序后 MEE 继续下降，且**帧间抖动明显减小** |
| 阶段 4 | 完整对比表 + 消融 + 下游任务 |

---

## 七、风险清单

| 风险 | 影响 | 应对 |
|------|------|------|
| **伪红外图像下不到** | 无法训练 C-MPDR | 提前确认；备选：自己用 UMF-CMGR 生成 |
| kornia 版本冲突 | 代码报错 | 锁 `kornia==0.6.12` |
| PyTorch 升级后其他 API 变化 | 零星报错 | 逐个修，主要是 `.cuda()` 和 `torch.FloatTensor` 之类的老写法 |
| 显存不足（若用 5060/3060）| 训练中断 | 降 batch / 用 3090 |
| IMF 用固定分辨率（256×256）| 你的数据是 480×640 | 需要适配或 resize |

---

## 八、和最终目标的关系

```
IMF（图像级、配准+融合一体）
      ↓ 复现
你的基座
      ↓ 加时序（DFF→TFF、PFF→时序、时序损失）
你的方法（视频级、配准+融合一体）  ← 论文的核心贡献
      ↓ 接你的 Mamba 融合
端到端系统
```