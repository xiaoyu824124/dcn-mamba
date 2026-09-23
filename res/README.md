# IR–VI 单帧配准：先粗后细

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
   `python -B -m res.train_vtmot --device cuda --steps 1 --num-workers 0 --init res_runs/vtmot_affine_stable_3060/best.pt --output-dir res_runs/a4000_local_smoke`。
   应出现有限的 `loss`、`train_epe`、`local` 和 `peak_mem_mib`。`res_runs/` 不随 Git 同步；
   若服务器没有该旧权重，去掉 `--init`，或者先单独把权重复制到服务器。
4. 用新的目录训练 3000 步：
   `python -B -m res.train_vtmot --device cuda --run full --num-workers 0 --init res_runs/vtmot_affine_stable_3060/best.pt --output-dir res_runs/a4000_local_full`。
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
