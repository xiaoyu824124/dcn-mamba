# IR–VI 单帧配准：先粗后细

## Linux A4000 服务器准备

在 SSH 连接的 VS Code Linux 终端、仓库根目录执行。建议 Python 3.10；
本地验证环境为 Python 3.10、PyTorch 2.4.0 CUDA 12.1。先用
`nvidia-smi` 检查显卡与驱动。

```bash
git pull origin main
conda create -n res-reg python=3.10 -y
conda activate res-reg
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r res/requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -B -m unittest discover -s res/tests -t .
```

`res` 单帧训练只用 `torch`、NumPy、Pillow 和 OmegaConf；不需要安装 CRFT
仓库，也无需安装根目录完整 `requirements.txt`。如已有可用的 PyTorch CUDA
环境，先做上面的 CUDA 检查和测试即可，不必重复安装。

Git 包含 `res/` 与 `data_split/IVF/VTMOT/split.json`。数据集
`data/VTMOT_misaligned/` 和权重目录 `res_runs/` 被 Git 忽略，必须在服务器上
单独放置。数据目录中每个序列至少有 `infrared/*.jpg`、
`visible_mis/*.png` 和 `gt_h/*.npy`；运行 `--check-gt` 时还需
`visible_gt/*.png`。数据不在默认位置时给训练和评估命令加
`--data-root /你的路径/VTMOT_misaligned`。旧粗场权重仅是可选的
`--init` 热启动。本地 `res_runs/vtmot_affine_stable_3060/best.pt` 来自早期
160×160 裁剪实验，checkpoint 记录为第 400 步；它没有上传到 Git，也不是
服务器从零训练的前置条件。要公平比较 SA-CA，从零分别训练
`stage0_coarse.yaml` 和 `stage1_saca.yaml` 即可。

## CRFT 分步改造：阶段 1

已加入可选的 1/8 线性 SA-CA，插在 Encoder 与 GlobalMatcher 之间。
它沿用 CRFT 的 self/cross 顺序及 ELU+1 线性注意力思路；残差增益从零开始，
因此旧粗场权重加载后，改造前后的首个预测相同。此阶段保留本项目的
dual-softmax、软 argmax 和 6-DoF WLS；独立配置关闭 1/4 局部头，避免
两个结构同时变化。参考 [CRFT 论文](https://arxiv.org/html/2604.05689v1)
及[官方实现](https://github.com/NEU-Liuxuecong/CRFT)。

在 A4000 上用相同数据、步数与随机种子分别运行粗场基线和阶段 1：

```bash
python -B -m res.train_vtmot --device cuda --steps 1 --num-workers 0 --overlay res/configs/stage1_saca.yaml --output-dir res_runs/saca_smoke
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --overlay res/configs/stage0_coarse.yaml --output-dir res_runs/coarse_pilot
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --overlay res/configs/stage1_saca.yaml --output-dir res_runs/saca_pilot
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/coarse_pilot/best.pt --output res_runs/coarse_pilot/eval_stride10.json
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/saca_pilot/best.pt --output res_runs/saca_pilot/eval_stride10.json
```

请保留两组 `metrics.jsonl`、`eval_stride10.json` 和一步试跑的显存峰值。
先比较 `coarse_epe_px`、
`match_frac_keys_beating_gt`、`match_epe_argmax_px` 和峰值显存；若注意力
改善匹配但 WLS 粗场仍不改善，再检查置信度权重和仿射拟合。

后续阶段依次为：1/4 的粗场中心局部窗口与轻量 FSFT、1/2 小范围残差修正，
最后才评估 DCN 亚像素修正。CRFT 的 `fine_process` 使用窗口展开后的注意力，
在 480×640 上需改为逐窗口或分块计算；不能直接复制全序列实现。

当前 `res` 分支实现如下：

```text
IR、VI
  → MIND 自相似描述子（固定计算，无可训练参数）
  → 共享 Encoder（学习跨模态结构特征）
  → 1/8 全局 all-pairs 匹配，置信度加权仿射拟合，得到粗场
  → 用粗场预对齐 IR 图像，并定位每个 1/4 特征的搜索中心
  → 1/4 局部相关匹配，轻量修正头预测残差，得到最终流场与配准图
```

流场在 VI 网格上定义，通道顺序为 `[dy, dx]`。`aligned_ir(y,x) =
ir(y + dy, x + dx)`。局部匹配直接在 IR 原特征中采样
`p + coarse(p) + offset`，因此搜索窗口始终围绕当前像素的粗预测。
默认半径 6 个 1/4 特征格，对应图像上的约 ±24 像素；候选分块计算，
避免一次保留所有采样特征。修正头的末层零初始化，训练开始时最终流场等于
粗场；它需要根据局部相关分布和流场监督学会修正。最终图像只用最终流场做一次
backward warp。

训练使用 VTMOT 的仿射 GT：1/8 对应分布、仿射参数、最终流场、
1/4 局部对应分布，以及 MIND、边缘和光滑项共同监督。局部对应损失只对
GT 落在搜索窗口内的像素计算。`res/configs/registration.yaml` 是默认配置。
`--init` 从旧粗配准权重加载兼容张量；`--resume` 严格恢复同一次训练。

## A4000 上逐步验证

数据位置：`data/VTMOT_misaligned/`，划分文件：
`data_split/IVF/VTMOT/split.json`。在仓库根目录执行：

1. 在服务器执行 `git pull origin main` 拉取本次提交，然后在服务器的 Python 环境运行
   `python -B -m unittest discover -s res/tests -t .`。应全部通过。
2. 核对 GT 方向：
   `python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 50 --check-gt`。
   `gt_warp_mae` 应小于 `unwarped_mae`。
3. 做一步完整分辨率试跑，检查数据、梯度和显存：
   `python -B -m res.train_vtmot --device cuda --steps 1 --num-workers 0 --output-dir res_runs/a4000_local_smoke`。
   应出现有限的 `loss`、`train_epe`、`local` 和 `peak_mem_mib`。
4. 用新的目录训练 3000 步：
   `python -B -m res.train_vtmot --device cuda --run full --num-workers 0 --output-dir res_runs/a4000_local_full`。
   每 100 步看 `val_epe_px`、`val_coarse_epe_px`、`val_local_window_coverage`、
   `val_pck_3px`。局部细化应使最终 EPE 低于粗场 EPE；若持续变差，先保留
   `best.pt`，再调整局部温度、半径或局部损失权重。
5. 在较密的开发集上确认 `best.pt`：
   `python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/a4000_local_full/best.pt --output res_runs/a4000_local_full/eval_stride10.json`。
   报告同时列出最终 `epe_px`、`coarse_epe_px`、零场 EPE、PCK 和局部覆盖率。
   达到 `EPE ≤ 2 px` 且 `PCK@3px ≥ 0.90` 后，再在 `test` 划分做一次最终评估。

输出目录已有 `metrics.jsonl`、`best.pt` 或 `last.pt` 时，新训练会拒绝覆盖。
续训用同一目录和 `--resume .../last.pt --steps <新的总步数>`。
这些是验证门槛，并非当前已经达到的结果；A4000 训练结果需以实际日志为准。

## 后续关键帧与时序

当前 VTMOT 入口按独立帧训练和评估；它没有连续帧、关键帧选择或时序 GT。
先验证单帧最终流场稳定，再接入序列数据：保存可靠关键帧的 VI/IR 特征与
最终流场，在后续帧做运动传播，以当前帧的局部匹配修正，并按置信度更新
关键帧。验证时分开报告关键帧与非关键帧的 EPE/PCK、相邻帧流场一致性和
遮挡区域表现。不要把单帧结果当作时序结果。
