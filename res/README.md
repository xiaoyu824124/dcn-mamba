# IR–VI 单帧配准：先粗后细

## CRFT 独立基线（VTMOT 同协议）

官方 [CRFT 代码](https://github.com/NEU-Liuxuecong/CRFT) 单独解压，不复制进本仓库。
本仓库的 `res.evaluate_crft_vtmot` 和 `res.train_crft_vtmot` 直接调用其模型，
沿用本仓库的 VTMOT 清单、480×640 预处理、有效像素掩码、EPE 和逐像素 PCK。
CRFT 输入顺序为可见光 `image0`、红外 `image1`，输出流场为 `[dx,dy]`；
适配器转成 VTMOT 的 `[dy,dx]` 后才计算指标。

官方细尺度位置编码需要方形输入，因此适配器默认把 480×640 等比缩至
72×96，居中补边到 96×96，预测后裁去补边，再把位移按比例升回
480×640 评估；JSON 会记录 `model_input_hw`。这一设置是 VTMOT 适配实验，
不是复现论文的 RoadScene 2.37 px。论文的 CMR@3px 是按样本 EPE
是否低于阈值计数，也不能直接与这里逐像素 `pck_3px` 比较。

在 A4000 服务器安装本仓库依赖和适配器所需的 `yacs`、`einops`，
并将官方 CRFT 仓库单独解压，例如 `G:\cxj\CRFT-main`。若已下载
官方 RoadScene 权重，可先评估零样本迁移：

```bat
python -m pip install yacs einops -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -B -m res.evaluate_crft_vtmot --crft-root G:\cxj\CRFT-main --checkpoint G:\cxj\CRFT-main\checkpoints\CRFT_RoadScene.ckpt --device cuda --split eval --frame-stride 10 --output res_runs\crft_roadscene\eval_stride10.json
```

若没有官方权重，可从零训练；有权重则在命令中加入
`--init G:\cxj\CRFT-main\checkpoints\CRFT_RoadScene.ckpt`，
以相同 VTMOT 数据进行微调。先做一步烟雾测试，再跑 300 步试验：

```bat
python -B -m res.train_crft_vtmot --crft-root G:\cxj\CRFT-main --device cuda --steps 1 --eval-every 1 --eval-max-samples 2 --num-workers 0 --output-dir res_runs\crft_smoke
python -B -m res.train_crft_vtmot --crft-root G:\cxj\CRFT-main --device cuda --steps 300 --num-workers 0 --output-dir res_runs\crft_vtmot_pilot
python -B -m res.evaluate_crft_vtmot --crft-root G:\cxj\CRFT-main --checkpoint res_runs\crft_vtmot_pilot\best.pt --device cuda --split eval --frame-stride 10 --output res_runs\crft_vtmot_pilot\eval_stride10.json
```

训练入口用相同的 VTMOT 密集真值监督 CRFT 最终场，并给粗场 0.25
权重；它不声称复现官方训练目标。先比较同一 80 帧的 `epe_px`、
`coarse_epe_px`、`pck_3px`、`zero_flow_epe_px` 与显存峰值。
完整数据和 CRFT 官方权重均不随 Git 提交，服务器须分别放置。

96×96 模型从零训练 300 步后，可以先用同一个权重在相同的 80 帧上
做输入分辨率诊断；`--model-hw` 只改变评估时的方形输入大小，不改权重。
JSON 同时记录训练和评估输入大小。先运行 96×96，再尝试 128×128；
若显存不足，不再增大输入。不同分辨率的推理结果是诊断，正式基线
应使用相应分辨率继续训练并重新评估。

```bat
python -B -m res.evaluate_crft_vtmot --crft-root G:\cxj\CRFT-main --checkpoint res_runs\crft_vtmot_pilot\best.pt --device cuda --split eval --frame-stride 10 --model-hw 96 96 --output res_runs\crft_vtmot_pilot\eval_96_stride10.json
python -B -m res.evaluate_crft_vtmot --crft-root G:\cxj\CRFT-main --checkpoint res_runs\crft_vtmot_pilot\best.pt --device cuda --split eval --frame-stride 10 --model-hw 128 128 --output res_runs\crft_vtmot_pilot\eval_128_stride10.json
```

## 本模型：限制粗匹配候选数

480×640 输入的 1/8 网格有 60×80＝4800 个候选。可选配置
`ab_compact_global300.yaml` 只把送入全局匹配器的 1/8 特征平均池化到
15×20，保留全图视场，但把每个查询的候选数降为 300。若启用 SA–CA，
它也在池化后的小网格上运行；原始 1/8 特征仍送入 1/4 跨尺度模块。
局部匹配器、WLS 和全分辨率流场监督不变。
训练与推理必须使用同一配置；不开启时旧 checkpoint 的行为不变。

先对既有 1/4 描述子 checkpoint 做同权重 80 帧诊断：

```bat
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/local_head_from_descriptor_pilot/last.pt --overlay res/configs/ab_compact_global300.yaml --output res_runs/local_head_from_descriptor_pilot/eval_compact300_stride10.json
```

与该 checkpoint 原始全局网格的结果比较 `coarse_match_candidates`、
`coarse_epe_px`、`match_epe_argmax_px`、`local_window_coverage` 和
`epe_px`。这是复用旧权重的诊断：池化同时改变了特征和候选数，结果
不能单独证明候选数的因果作用，也不能代替在小网格上重新训练。

如果继续检验小网格上的学习，从同一 1/4 描述子／局部头权重做两组
300 步训练。两组都显式启用 `stage2_fine_interaction.yaml`，因为
`--init` 只加载权重，不恢复 checkpoint 的结构配置。第二组仅增加
SA–CA；其残差增益从零开始，因此两组的第 0 步输出相同。

```bat
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --lr 0.0001 --init res_runs/local_head_from_descriptor_pilot/last.pt --overlay res/configs/stage2_fine_interaction.yaml --overlay res/configs/ab_compact_global300.yaml --output-dir res_runs/compact300_control_pilot
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --lr 0.0001 --init res_runs/local_head_from_descriptor_pilot/last.pt --overlay res/configs/stage2_fine_interaction.yaml --overlay res/configs/ab_compact_global300.yaml --overlay res/configs/ab_saca_full.yaml --output-dir res_runs/compact300_saca_pilot
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/compact300_control_pilot/last.pt --output res_runs/compact300_control_pilot/eval_last_stride10.json
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/compact300_saca_pilot/last.pt --output res_runs/compact300_saca_pilot/eval_last_stride10.json
```

比较 0、100、200、300 步和两组 `last.pt` 的 80 帧指标；`best.pt`
按 16 帧 EPE 选取，可能保留第 0 步，不能用来判断 SA–CA 是否学会。

## A4000 服务器准备（当前为 Windows 环境）

你贴出的 `G:\cxj\VF-Bench-main` 和 `C:\Users\cxj\.conda` 表明当前 SSH 终端连接的是
Windows 服务器。在服务器的 VS Code 终端、仓库根目录执行。建议 Python 3.10；
本地验证环境为 Python 3.10、PyTorch 2.4.0 CUDA 12.1。先用
`nvidia-smi` 检查显卡与驱动。

```bat
git pull origin main
conda create -n res-reg python=3.10 -y
conda activate res-reg
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r res/requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -B -m unittest discover -s res/tests -t .
```

普通 Python 依赖使用[清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)；
PyTorch CUDA 12.1 包仍从[官方 cu121 源](https://docs.pytorch.org/get-started/previous-versions/)安装，
以固定 CUDA 包版本。如果 `vfbench-a4000` 环境已有可用的 PyTorch CUDA 和上述依赖，
可直接复用，不必新建环境。

`res` 单帧训练只用 `torch`、NumPy、Pillow 和 OmegaConf；不需要安装 CRFT
仓库，也无需安装根目录完整 `requirements.txt`。

Git 包含 `res/` 与 `data_split/IVF/VTMOT/` 下的划分文件及序列 CSV。数据集
`data/VTMOT_misaligned/` 和权重目录 `res_runs/` 被 Git 忽略，必须在服务器上
单独放置。数据目录中每个序列至少有 `infrared/*.jpg`、
`visible_mis/*.png` 和 `gt_h/*.npy`；运行 `--check-gt` 时还需
`visible_gt/*.png`。读取器以 `data_split/IVF/VTMOT/<序列>.csv` 中列出的帧为准，
忽略 `infrared` 中未列出的带哈希后缀原始文件。如果清单中的帧缺失，错误会指出
具体路径；此时需把本地已处理的 `data/VTMOT_misaligned/` 同步到服务器，或用
`--data-root G:/你的路径/VTMOT_misaligned` 指向已有的完整数据。旧粗场权重仅是可选的
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

## 阶段 2：检验 1/8 远距离错配

A4000 的 1500 步局部实验在 80 帧验证集上得到粗场 EPE 7.18 px，
但全局匹配 argmax EPE 仍为 160.5 px。新增可选软位置先验，
`spatial_prior_sigma=4` 表示 1/8 特征格上的标准差为 4 格，即图像上的
32 px；所有位置仍参与匹配。本地抽样的验证集 GT 角点位移 99% 不超过
约 21 px，因此先从较宽的 32 px 先验试起。这个开关不增加权重，
可在相同 checkpoint 上直接比较打开前后，无需先训练：

```bat
git pull origin main
python -B -m unittest discover -s res/tests -t .
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/local_from_coarse_pilot/best.pt --overlay res/configs/ab_spatial_prior32.yaml --diagnose-appearance --output res_runs/local_from_coarse_pilot/eval_prior32_stride10.json
```

将新报告与已保存的 `eval_stride10_step1500.json` 对照，优先看
`coarse_epe_px`、`match_epe_argmax_px`、`match_frac_keys_beating_gt`、
`pck_3px` 和 `epe_px`。新增的 `appearance_frac_keys_beating_gt` 与
`appearance_epe_argmax_px` 只看 Encoder 的原始余弦分数，不包含位置先验，
用于判断特征是否真的找到 GT。若粗场 EPE 未改善，先分析特征和 GT 对应点的相似度，
不要仅凭概率分布变集中就增加训练步数。`--overlay` 评估时仍严格加载原权重，
适用于这类不改变参数形状的开关；`--diagnose-appearance` 会暂时多保留一张
4800×4800 的分数矩阵，适合评估，不要用于训练。

A4000 的 80 帧对照中，32 px 先验把匹配 argmax EPE 从约 160.5 px 降到
13.7 px，但粗场 EPE 从 7.18 px 升到 7.23 px，最终 EPE 从 7.26 px 升到
7.37 px。原始外观分数的 argmax EPE 仍为 160.8 px；峰值位置的改善主要来自
先验，不能据此认定 Encoder 已学到可靠的全局对应。局部头在该权重上也没有
补偿粗场误差。先用同一权重试更宽的 64 px 先验，并查看新增的
`appearance_window2_*` 和 `appearance_window4_*`：这些指标仅比较粗场附近的
原始余弦分数，排除位置先验。评估命令：

```bat
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/local_from_coarse_pilot/best.pt --overlay res/configs/ab_spatial_prior64.yaml --diagnose-appearance --output res_runs/local_from_coarse_pilot/eval_prior64_stride10.json
```

若 64 px 先验仍未降低 `coarse_epe_px`，先保持当前最好的无先验权重，
针对 1/8 特征的局部判别力调整训练目标，不直接增加训练步数或接入融合。

后续阶段依次为：1/4 的粗场中心局部窗口与轻量 FSFT、1/2 小范围残差修正，
最后才评估 DCN 亚像素修正。CRFT 的 `fine_process` 使用窗口展开后的注意力，
在 480×640 上需改为逐窗口或分块计算；不能直接复制全序列实现。

## 下一轮结构对照：SA-CA 与 1/4 跨尺度特征

64 px 先验的 80 帧对照粗场 EPE 为 7.25 px、最终 EPE 为 7.37 px，
仍未优于无先验的 7.18/7.26 px。原始外观分数在粗场附近的 ±2 格窗口内，
真实对应点仅约 36.8% 的查询排第一。因此先停止扩大先验，检查特征学习。

`ab_saca_full.yaml` 在已训练的 Encoder 与 1/8 全局匹配器之间插入 SA-CA，
保留已训练的局部头。SA-CA 的残差增益从零初始化，所以热启动第一步仍与
原权重输出一致。此前从零训练 300 步的 SA-CA 对照未见收益，这次验证的是
已训练权重上继续学习的效果。`stage2_fine_interaction.yaml` 在 1/4 局部匹配前添加共享的
IR/VI 投影及各自的 1/8→1/4 上采样上下文融合；其残差增益也从零开始。
全局匹配器、WLS、局部匹配器及其坐标约定不变。当前 `res` 单帧入口尚未接
DCN；原有 `src/model/registration/dcn_refinement.py` 保留，待单帧粗细场达到
精度门槛后再评估接入，避免 DCN 掩盖上游误差。

先用相同权重、数据顺序、学习率和步数做有无 SA-CA 对照：

```bat
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --lr 0.0001 --init res_runs/local_from_coarse_pilot/best.pt --output-dir res_runs/control_warm_pilot
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --lr 0.0001 --init res_runs/local_from_coarse_pilot/best.pt --overlay res/configs/ab_saca_full.yaml --output-dir res_runs/saca_warm_pilot
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/control_warm_pilot/best.pt --diagnose-appearance --output res_runs/control_warm_pilot/eval_stride10.json
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/saca_warm_pilot/best.pt --diagnose-appearance --output res_runs/saca_warm_pilot/eval_stride10.json
```

比较 `coarse_epe_px`、`epe_px`、`appearance_window2_argmax_correct` 和
`appearance_window4_argmax_correct`。只有 SA-CA 至少改善外观判别力且不损害
粗场 EPE，才以其 `best.pt` 为起点测试 1/4 跨尺度模块。独立的
`ab_appearance_window4.yaml` 可为两组训练同时增加不含位置先验的局部外观
监督；不要只给其中一组开启。该损失会额外保留全局相关矩阵，先做一步显存
试跑，再决定是否用于完整训练。

若 SA-CA 对照通过，再从同一 SA-CA 权重测试 1/4 投影与跨尺度融合；两组
都显式加载 SA-CA 配置，因为 `--init` 只导入权重，不导入模型结构配置：

```bat
python -B -m res.train_vtmot --device cuda --steps 1 --num-workers 0 --lr 0.0001 --init res_runs/saca_warm_pilot/best.pt --overlay res/configs/ab_saca_full.yaml --overlay res/configs/stage2_fine_interaction.yaml --output-dir res_runs/fine_interaction_smoke
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --lr 0.0001 --init res_runs/saca_warm_pilot/best.pt --overlay res/configs/ab_saca_full.yaml --output-dir res_runs/local_saca_control_pilot
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --lr 0.0001 --init res_runs/saca_warm_pilot/best.pt --overlay res/configs/ab_saca_full.yaml --overlay res/configs/stage2_fine_interaction.yaml --output-dir res_runs/fine_interaction_pilot
```

先确认一步试跑的损失、梯度和显存正常，再比较两组 80 帧评估中的
`coarse_epe_px`、`epe_px`、`local_argmax_epe_px` 与 `pck_3px`。

第一次从 `control_warm_pilot/best.pt` 继续训练 1/4 模块时，两组 300 步在
16 帧上的最佳最终 EPE 分别为 7.641 px（原局部头）和 7.645 px（跨尺度），
几乎相同，而且均高于热启动时的 6.946 px。两组的粗场也发生明显漂移，
因此这轮不能单独判断 1/4 模块的效果。下一轮用 `ab_freeze_coarse.yaml`
冻结共享 Encoder、SA-CA（若启用）及全局匹配器，只更新原局部头和可选
跨尺度模块。`--init` 现在会先在 16 帧验证集评估热启动权重，并将第 0 步
保存为候选 `best.pt`；后续训练只有真正超过它才替换。

```bat
python -B -m res.train_vtmot --device cuda --steps 1 --num-workers 0 --lr 0.0001 --init res_runs/control_warm_pilot/best.pt --overlay res/configs/ab_freeze_coarse.yaml --overlay res/configs/stage2_fine_interaction.yaml --output-dir res_runs/fine_frozen_smoke
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --lr 0.0001 --init res_runs/control_warm_pilot/best.pt --overlay res/configs/ab_freeze_coarse.yaml --output-dir res_runs/local_frozen_control_pilot
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --lr 0.0001 --init res_runs/control_warm_pilot/best.pt --overlay res/configs/ab_freeze_coarse.yaml --overlay res/configs/stage2_fine_interaction.yaml --output-dir res_runs/fine_frozen_pilot
```

两组在 16 帧上的 `coarse_epe_px` 应固定在相同的热启动值（约 6.867 px）。
若它漂移，先停下检查冻结开关。之后分别对两组 `best.pt` 做相同的 80 帧
评估，只有跨尺度组的最终 EPE 和 PCK 均优于冻结控制组，才保留该模块。

## 检查匹配概率到 WLS 粗场的转换

冻结粗场后的 80 帧结果：跨尺度模块将局部 argmax EPE 从 19.33 px 降到
17.35 px，但最终 EPE 仅从 6.839 px 到 6.826 px，几乎等于原权重的
6.827 px。随后加入 1/8 原始外观窗口损失继续训练 300 步，80 帧粗场
EPE 为 7.347 px；同学习率控制组为 7.215 px，均差于原权重的 6.843 px。
外观窗口的排名改善也很小，因此暂不叠加更多特征模块或延长训练。

全局匹配器当前用每个查询的最大匹配概率的四次方作为仿射 WLS 权重。
新增 `affine_confidence_power` 只改变这一步的权重，不改变 Encoder、
匹配概率或 checkpoint 参数。用同一 `control_warm_pilot/best.pt` 分别
评估四次方（原设置）、二次方和均匀权重：

```bat
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/control_warm_pilot/best.pt --output res_runs/control_warm_pilot/eval_wls_power4.json
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/control_warm_pilot/best.pt --overlay res/configs/ab_wls_power2.yaml --output res_runs/control_warm_pilot/eval_wls_power2.json
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/control_warm_pilot/best.pt --overlay res/configs/ab_wls_uniform.yaml --output res_runs/control_warm_pilot/eval_wls_uniform.json
```

先看 `affine_weight_effective_queries_ratio`：数值接近零说明拟合主要依赖
少数点。再比较 `coarse_epe_px`、`affine_corner_epe_px` 和最终 `epe_px`。
只有粗场与最终 EPE 都改善，才考虑更改默认权重；否则继续检查原始
1/8 特征与 GT 对应关系。

80 帧结果确认四次方仍最好：有效查询比例仅 0.00428（4800 格中约 21 格），
粗场 EPE 6.843 px；二次方有效比例 0.01839，粗场 EPE 10.583 px；均匀
权重粗场 EPE 97.405 px。匹配概率与原始外观完全相同，因此问题是低置信
查询的软坐标严重偏离 GT，而不是 WLS 需要更均匀的权重。保持四次方默认值。

下一步同一权重再报告三项 WLS 支撑诊断：
`affine_weight_valid_fraction` 是 WLS 权重落在有 GT 且对应点在图内查询的
比例；`affine_weighted_raw_epe_px` 是这些点的加权软匹配误差；
`affine_weight_spread_ratio` 是高权重查询在较窄空间方向上的分布宽度，
均匀覆盖为 1，接近 0 表示高度聚集。

```bat
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/control_warm_pilot/best.pt --output res_runs/control_warm_pilot/eval_wls_support.json
```

若加权软匹配误差仍大，优先改进高置信点本身的跨模态特征；若它较小但
空间分布高度聚集，先检查仿射拟合的几何覆盖与稳定性。若有效区域权重
比例低，则先修正 WLS 对无效查询的处理。

实测四次方 WLS 的加权软匹配误差为 11.65 px，经过仿射拟合后粗场为
6.84 px；空间分布比例为 1.60，没有集中在一条窄带。但只有 47.7% 的
WLS 权重对应 GT 在图内的查询。`valid_mask` 在当前 VTMOT 读取器中仅由
GT 映射坐标是否落在图内决定；生成 `visible_mis` 时图像边界使用复制填充。
这提示边界复制纹理可能产生高置信伪匹配，尚需同权重实验验证。

`affine_border_margin` 只在仿射 WLS 拟合时排除 VI 查询网格的外圈，不改
MIND、Encoder、全局匹配概率或局部头，也不需重训。默认 0 保持原行为。
对 60×80 的 1/8 网格，下面两个配置分别排除约 16 px 和 32 px：

```bat
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/control_warm_pilot/best.pt --overlay res/configs/ab_wls_border16.yaml --output res_runs/control_warm_pilot/eval_wls_border16.json
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/control_warm_pilot/best.pt --overlay res/configs/ab_wls_border32.yaml --output res_runs/control_warm_pilot/eval_wls_border32.json
```

比较 `affine_weight_valid_fraction`、`affine_weight_effective_queries_ratio`、
`affine_weight_spread_ratio`、粗场 EPE 与最终 EPE。只有有效权重比例和
配准误差都改善，才保留边界屏蔽；若比例升高但 EPE 变差，说明被排除
的点中仍有关键的可靠匹配。

80 帧实测排除 16 px 后有效权重比例从 0.477 升至 0.996，但粗场 EPE
从 6.843 升至 16.838 px，最终 EPE 从 6.827 升至 14.005 px；排除
32 px 后粗场与最终 EPE 继续恶化至 24.353 和 19.937 px。对应的加权
软匹配误差从 11.653 升至 16.156、19.633 px，空间分布比例从 1.597
降至 0.666、0.472。边缘高权重点虽然常被 GT 有效掩码排除，却对当前
仿射估计有关键作用。保留默认完整 WLS，不按 GT 有效比例单独筛点。

下一步先诊断 1/4 局部头的增益上限。同一 checkpoint 重新评估即可，
不需重训：

```bat
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/control_warm_pilot/best.pt --output res_runs/control_warm_pilot/eval_local_diagnostics.json
```

`local_oracle_epe_px` 是窗口内最佳合法候选的误差，只在 GT 被窗口覆盖的
查询上统计；`local_soft_epe_px` 是候选概率加权平均位移的误差；
`local_gt_residual_px` 是 1/4 网格上粗场剩余误差；
`local_pred_residual_px` 是局部头实际修正幅度；
`local_refinement_improved_fraction` 是局部头降低误差的查询比例。
若 oracle 很好而 soft 很差，应优先改 1/4 特征或匹配目标；若 soft
已经好而最终场仍无增益，应检查残差头训练与输出方式。

80 帧结果：窗口覆盖率 1.000，oracle EPE 1.506 px；当前 1/4 匹配的
argmax EPE 19.333 px，概率加权位移 EPE 7.260 px，比粗场在 1/4 网格
上的 6.857 px 还高。局部头平均只修正 0.592 px，只有 35.0% 的查询
改善。首先需要改善窗口内候选的区分能力，而不是扩大窗口或叠加 DCN。

同一 checkpoint 可比较 1/4 Encoder 特征与原始 MIND 描述子。评估时将
MIND 平均池化到 1/4 网格，使用完全相同的粗场、候选窗口和局部温度；
仅比较候选 argmax 和概率加权位移，原模型最终 EPE 不变：

```bat
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/control_warm_pilot/best.pt --diagnose-local-mind --output res_runs/control_warm_pilot/eval_local_mind.json
```

若 `mind_local_argmax_epe_px` 和 `mind_local_soft_epe_px` 均显著低于
Encoder 对应指标，再考虑把 MIND 直通特征送入专门的 1/4 局部分支；
若 MIND 也差，则优先改局部描述子的监督和采样方式。

实测原始 MIND 的局部 argmax／soft EPE 为 20.433／7.432 px，均差于
Encoder 的 19.333／7.260 px。因此继续使用 Encoder 的 1/4 特征，
不接入原始 MIND 细匹配。下一轮只训练已有的共享 1/4 特征交互模块，
用局部候选的真值 NLL 判断它能否学到更好的跨模态描述子。
`ab_local_descriptor_only.yaml` 固定 Encoder、全局匹配器、WLS 和局部
残差头，其他损失权重置零；只有局部特征交互模块更新。

```bat
python -B -m res.train_vtmot --device cuda --steps 1 --num-workers 0 --lr 0.0001 --init res_runs/control_warm_pilot/best.pt --overlay res/configs/ab_local_descriptor_only.yaml --output-dir res_runs/local_descriptor_only_smoke
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --lr 0.0001 --init res_runs/control_warm_pilot/best.pt --overlay res/configs/ab_local_descriptor_only.yaml --output-dir res_runs/local_descriptor_only_pilot
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/local_descriptor_only_pilot/last.pt --output res_runs/local_descriptor_only_pilot/eval_stride10.json
```

这一轮按 `last.pt` 比较局部指标：`best.pt` 仍按最终 EPE 选取，可能保留
第 0 步而错过描述子是否学会匹配。粗场 EPE 应保持约 6.843 px；
局部 soft EPE 基线为 7.260 px，粗场在 1/4 网格上的误差为 6.857 px。
只有局部 soft EPE 低于粗场且 argmax EPE 也下降，才继续训练残差头。

300 步局部描述子实验保持 80 帧粗场 EPE 6.843 px 不变；局部 argmax EPE
从 19.333 降至 17.225 px，soft EPE 从 7.260 降至 7.002 px；最终 EPE
仅从 6.827 降至 6.820 px。训练有信号，但局部概率加权位移仍差于
6.857 px 的粗场。下一轮从相同的热启动权重训练 900 步，使前 300 步
与上一轮可比，再观察后 600 步是否持续下降。训练输出现在会在每次验证
时打印局部 soft、argmax 与 oracle EPE。

```bat
python -B -m res.train_vtmot --device cuda --steps 900 --num-workers 0 --lr 0.0001 --init res_runs/control_warm_pilot/best.pt --overlay res/configs/ab_local_descriptor_only.yaml --output-dir res_runs/local_descriptor_only_900
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/local_descriptor_only_900/last.pt --output res_runs/local_descriptor_only_900/eval_stride10.json
```

若局部 soft EPE 不再下降，停止延长这一结构；若稳定低于粗场且
argmax 继续改善，再单独解冻残差头。此处继续看 `last.pt` 的特征质量，
不以最终 EPE 选出的 `best.pt` 代替它。

900 步结果显示候选排序继续改善：80 帧局部 argmax EPE 从原来的
19.333 降至 15.939 px；但 soft EPE 仅从 7.260 降至 7.051 px，
仍高于粗场的 6.857 px。16 帧验证集上的局部 soft EPE 在约 400 步
达到 7.068 px 后没有持续下降，900 步为 7.134 px。80 帧最终 EPE
只从 6.827 降至 6.810 px。停止延长描述子单独训练。

下一步保留 900 步学到的描述子，固定 Encoder、全局匹配器、WLS 和
1/4 特征交互模块，只用 GT 流场训练已有局部残差头。这验证较好的候选
排序能否转化成最终配准收益：平均 soft EPE 虽然高于粗场，残差头仍可能
只选择可靠区域来修正。不改变搜索半径或增加 DCN。

```bat
python -B -m res.train_vtmot --device cuda --steps 1 --num-workers 0 --lr 0.0001 --init res_runs/local_descriptor_only_900/last.pt --overlay res/configs/ab_local_head_from_descriptor.yaml --output-dir res_runs/local_head_from_descriptor_smoke
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --lr 0.0001 --init res_runs/local_descriptor_only_900/last.pt --overlay res/configs/ab_local_head_from_descriptor.yaml --output-dir res_runs/local_head_from_descriptor_pilot
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/local_head_from_descriptor_pilot/last.pt --output res_runs/local_head_from_descriptor_pilot/eval_stride10.json
```

80 帧粗场 EPE 应继续固定在 6.843 px，局部 argmax 和 soft EPE 应
保持约 15.939 和 7.051 px；变化只应出现在残差修正与最终流场。
若最终 EPE 仍只改善几百分之一像素，就不再延长这一路局部头训练。

实测局部头 300 步后，80 帧粗场 EPE 和局部 soft／argmax EPE 完全不变；
最终 EPE 从 6.810 降至 6.760 px，改善查询比例从 37.1% 升至 47.5%，
但 PCK@3px 只从 0.16865 到 0.16874。停止延长这一局部头训练。

同一 checkpoint 增加局部匹配可靠性诊断，不需重训：

```bat
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/local_head_from_descriptor_pilot/last.pt --output res_runs/local_head_from_descriptor_pilot/eval_local_confidence.json
```

`local_soft_improved_fraction` 表示直接采用局部概率加权位移时改善粗场
的查询比例。`local_top10pct_*` 只看最大候选概率最高的 10% 有效查询，
分别比较这些位置的粗场、概率加权位移和 argmax 误差，以及概率加权位移
实际改善的比例。若高置信区域也无法优于粗场，后续应重做 1/4 特征；
若这部分明显更准，再试按置信度选择性修正。

近年工作的可借鉴点与当前试验顺序：

- [XoFTR（2024）](https://arxiv.org/abs/2404.09692) 在可见光–热红外匹配中使用跨模态预训练、伪热红外增强和细尺度重匹配；
  [MINIMA（2025）](https://arxiv.org/abs/2412.19412) 更强调先用充足的配对数据学习匹配先验，再做跨模态微调。两者说明仅延长现有小规模训练不一定能解决表征问题。
- [RoMa（2024）](https://arxiv.org/abs/2305.15404) 区分粗级多峰匹配分布与细级局部回归，并显式估计匹配概率；
  [Efficient LoFTR（2024）](https://arxiv.org/abs/2403.04765) 使用两阶段细尺度相关来提高亚像素定位。
  对本模型最便宜的第一步是验证局部概率能否识别可靠修正。
- [XoFTR++（2026）](https://www.nature.com/articles/s41598-026-68975-9) 在细级使用双向窗口跨模态交互。
  当前 `fine_interaction.py` 只有共享投影和跨尺度融合，还没有 1/4 局部 IR–VI 交互；若置信度筛选仍无效，这一模块比继续扩大局部头更值得单独消融。

同一权重运行置信度门控诊断，不改训练和默认推理。报告中的
`local_gate_top{10,25,50}_{head,soft}_epe_px` 是只在置信度最高的相应比例
1/4 查询处采用局部头或 soft 位移后，在**完整有效像素**上计算的 EPE；
`local_gate_top*_valid_fraction` 是这些查询覆盖的有效像素比例。比较它们
与 `coarse_epe_px` 和 `epe_px`，即可判断是否值得把选择性细化变成正式推理选项。
查询的选择只看预测概率和候选采样是否落在图像内，不看 GT 流场或 GT
有效区掩码；GT 仅用于最终 EPE。

```bat
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/local_head_from_descriptor_pilot/last.pt --diagnose-confidence-gate --output res_runs/local_head_from_descriptor_pilot/eval_confidence_gate_stride10.json
```

80 帧实测：完整局部头 EPE 6.760 px，粗场 6.843 px；只采用最高置信度
10%／25%／50% 的局部头，EPE 为 6.833／6.815／6.789 px，均不及完整
局部头。相应 soft 位移为 6.864／6.863／6.873 px；最高置信 10% 查询的
局部 soft EPE 7.283 px，比同位置粗场 7.158 px 更差。因此最大候选概率
目前不能识别可靠修正，不把门控接入推理。

下一步单独检验 [XoFTR++](https://www.nature.com/articles/s41598-026-68975-9)
启发的细尺度跨模态交互：`fine_cross_attention.py` 在 1/4 上先按粗场
预对齐 IR 特征，再用 5×5 局部注意力给 VI 查询补充 IR 上下文。
IR 原特征仍留在原坐标供原局部匹配器搜索；全局 WLS、搜索窗口与
残差头均不变。模块残差增益从零开始，加载旧权重时第 0 步输出相同。
这是对论文思想的轻量消融，不声称复现其完整双向模块。

先固定粗场、已有跨尺度模块和局部残差头，只以局部匹配 NLL 训练
新增模块。训练不需要重新计算基线；旧 checkpoint 就是同初始化的
零模块对照。

```bat
python -B -m res.train_vtmot --device cuda --steps 1 --num-workers 0 --lr 0.0001 --init res_runs/local_head_from_descriptor_pilot/last.pt --overlay res/configs/ab_fine_cross_attention.yaml --output-dir res_runs/fine_cross_smoke
python -B -m res.train_vtmot --device cuda --run pilot --num-workers 0 --lr 0.0001 --init res_runs/local_head_from_descriptor_pilot/last.pt --overlay res/configs/ab_fine_cross_attention.yaml --output-dir res_runs/fine_cross_pilot
python -B -m res.evaluate_vtmot --device cuda --split eval --frame-stride 10 --checkpoint res_runs/fine_cross_pilot/last.pt --output res_runs/fine_cross_pilot/eval_last_stride10.json
```

核对第 0 步评估与旧权重一致，`coarse_epe_px` 保持约 6.843 px；
重点比较 `local_argmax_epe_px`（旧 15.939）、`local_soft_epe_px`
（旧 7.051）和最终 `epe_px`（旧 6.760）。`best.pt` 可能因 16 帧验证
保留第 0 步，所以用 `last.pt` 看模块是否学到东西。若局部排名和
soft 位移都没有改善，就停止这一模块；若明显改善，再解冻局部残差头
做第二阶段训练。

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
