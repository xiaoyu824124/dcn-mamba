# OS-RFS / GLoIR 拆解与移植方案

> 来源论文：**Robust One-stop Multi-modality Image Registration-Fusion-Segmentation
> Framework Against Misalignments and Adversarial Attacks**
> （IEEE TMM 2025，DOI 10.1109/TMM.2025.3535291）
> 代码：https://github.com/wdhudiekou/OS-RFS
> 本地 PDF：`E:\论文\配准+融合\25一区Robust_One-stop_...pdf`

本文件只拆解其中的 **GLoIR（Global-Local Incremental Registration）配准模块**，
因为它是我们新配准模块的骨架来源。

---

## 1. GLoIR 要解决什么

论文把 IR 与 VIS 之间的错位拆成两部分（Eq.1）：

```
I_ir^lo = W(I_ir ; [Δx̃, Δỹ], η)      # 局部形变：高斯滤波（标准差 σ）平滑的随机场
Ĩ_ir    = W(I_ir^lo ; M_g)             # 全局位移：大范围像素平移
```

**注意：OS-RFS 也是在线合成错位来造训练数据的**，和我们用 `make_misaligned.py`
做的思路一致 —— 所以它有真值形变场 D_gt，可以监督。

含义：错位 = **大范围全局位移** + **细微局部形变**。
单个全局变换解决不了后者，所以必须两阶段。

---

## 2. 网络结构

### 2.1 整体（Fig.3）

```
(Ĩ_ir, I_vis)
      │
      ▼
┌─────────────────────────────────────────┐
│  GSR：全局位移配准                        │
│   双流 ResNet-50 → 最大尺度特征 f_ir^K,f_vis^K │
│   → Global Correlation Volume (GCV)      │
│   → CBR 卷积 + FC → 全局矩阵 M_g (2×3)    │
│   → 映射到网格 → D_g (H×W×2)             │
│   → STN warp → Î_ir                      │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  LDR：局部形变配准（增量式）               │
│   双尺度特征 + CSE（互补通道注意力）        │
│   → 级联 2 个 LoDE 模块 → 形变场 D_l       │
│   → STN warp → I_ir^reg                  │
└─────────────────────────────────────────┘
```

### 2.2 GSR 细节

**Global Correlation Volume**（Eq.5）—— 全位置两两相关：

```
C_GSR(x, y) = f_ir^K(x) × f_vis^K(y)
```
（x、y 是位置坐标。这比直接 concat 强，因为把"找对应"显式化了。）

**全局矩阵估计**（Eq.6）：

```
M_g = FC( CBRs( [f_ir^K ; f_vis^K ; C_GSR] ) )
```

**网格化 + 变换**：

```
D_g = M_g 映射到 H×W×2 的规则网格
Î_ir = Ĩ_ir ∘ D_g          # ∘ 表示 STN（Spatial Transformer Network）
```

### 2.3 LDR 细节

**CSE（互补挤压激励）**（Eq.7）—— 用通道注意力融合两模态：

```
f̂_ir^0  = f̂_ir^0  + Sig( FC( [P(f̂_ir^0) ; P(f̂_vis^0)] ) ) ⊗ (f̂_ir^0 + f̂_vis^0)
f̂_vis^0 = f̂_vis^0 + Sig( C3( [C1(f̂_ir^0) ; C1(f̂_vis^0)] ) ) ⊗ (f̂_ir^0 + f̂_vis^0)
```
- `P(·)` = 全局平均池化
- `C_n(·)` = n×n 卷积
- `Sig` = Sigmoid，`⊗` = 逐元素乘

**级联 LoDE（局部形变估计）**（Eq.8）：

```
d_c^s = M^s( [f̂_ir^1 ; f̂_vis^1] )
d_f^s = M^s( d_c^s ) ⊕ d_c^s          # ⊕ 逐元素加（残差）
```
- s = 0, 1 两级级联
- s = 1 时输入替换为 `[f̂_ir^0 ; d_f^{s-1}]`（**增量式**：第二级在第一级结果上继续修）
- 最终 `D_l = d_f^s`

**最终对齐**：

```
I_ir^reg = Î_ir ∘ D_l
```

### 2.4 损失（Eq.9–12）

| 损失 | 公式 | 作用 |
|------|------|------|
| **EPE** | `L_EPE = (1/N)Σ( ‖D_g − D_g^gt‖₁ + ‖D_l − D_l^gt‖₁ )` | 直接监督形变场（**需要真值**）|
| **SIM** | `L_SIM = ‖I^reg − I_ir‖₁ + Σ_j ‖ψ_j(I^reg) − ψ_j(I_ir)‖₁` | 像素级 + 感知级（ψ = VGG-19 第 j 层）|
| **Smooth** | `L_Smooth = ‖∇D_g‖₁ + ‖∇D_l‖₁` | 形变场平滑正则 |

**总损失**（Eq.12）：

```
L_Reg = λ·L_EPE + β·L_SIM + γ·L_Smooth        λ=1.0, β=10.0, γ=2.0
```

---

## 3. 三个关键设计（值得直接借鉴）

1. **用相关体（Correlation Volume）而不是 concat 来找对应**
   → 这正是跨模态配准该用的做法（我之前给你提过，RAFT/LoFTR 也是这条路）

2. **全局 → 局部的增量式结构**
   GSR 先修大位移，LDR 再在这个基础上修细微形变（注意 s=1 的输入是上一级结果）
   → 对应你要做的"全局粗配准 + 局部精配准"

3. **有真值形变场监督（EPE loss）**
   → 因为它是合成错位，所以能做到。你也一样（`gt_h/` 就是真值）

---

## 4. 移植到我们工程的位置

### 4.1 现在的数据流

```
IR 5帧  ──encoder_1──> feat1 ──DCN(时序)──> [prev',cur,next'] ─┐
                                                               ├─ concat ─> decoder ─> 融合Y
RGB 5帧 ──encoder_2──> feat2 ──DCN(时序)──> [prev',cur,next'] ─┘
```

（`src/model/net.py` 的 `VideoFusion`）

### 4.2 目标数据流

```
IR 5帧  ─┐
         ├─> [GLoIR 风格配准] ─> 对齐后的 IR', VIS'（5帧）
VIS 5帧 ─┘         ↑
              （这里要加时序 —— 我们的创新）
                   │
                   ▼
   编码 → DCN(帧间时序对齐) → Mamba SSM 时序融合 → decoder → 融合视频
```

### 4.3 接口设计

配准模块要和现有代码解耦，建议接口：

```python
class CrossModalRegistration(nn.Module):
    """
    输入:
        ir : [B, T, 1, H, W]   红外（参考帧系）
        vis: [B, T, 3, H, W]   可见光（待对齐）
    输出:
        vis_aligned : [B, T, 3, H, W]   对齐后的可见光
        fields      : dict              D_g, D_l 等中间量（用于监督/可视化）
    """
```

**注意**：输出既要给下游用（`vis_aligned`），又要保留形变场（`fields`）用于 EPE 损失。

---

## 5. 我们要改的地方（增量创新点）

| 项 | OS-RFS（图像级）| 我们（视频级）|
|----|----------------|--------------|
| 输入 | 单对图 (Ĩ_ir, I_vis) | **窗口/序列** |
| GSR | 单帧估全局矩阵 | **时序约束下的全局矩阵序列** |
| LDR | 单帧形变场 | **形变场带时序传播/平滑** |
| 损失 | EPE + SIM + Smooth | 上述 + **时序平滑项** |
| 下游 | 融合→分割 | 融合（我们的 Mamba）|

**最直接的增量**：给 GSR 和 LDR 的输出加时序约束 ——
让 warp(t) 不要逐帧独立抖动，而是平滑演变。

---

## 6. 复现步骤

| 步骤 | 做什么 | 产出 |
|:---:|--------|------|
| 1 | clone OS-RFS，读 README，**确认权重/数据能下** | 环境可用 |
| 2 | 跑通官方 demo / test | 官方结果 |
| 3 | 读 `GLoIR` 对应的源码，对照本文件第 2 节 | 代码级理解 |
| 4 | 在**我们的数据**上跑 OS-RFS 的配准部分 | MEE 数字（对比基线）|
| 5 | 把它的配准模块移植成独立模块，接到我们的 pipeline | 可运行骨架 |
| 6 | 加时序改造 | **我们的方法** |

---

## 7. 待确认事项

- [ ] OS-RFS 的预训练权重在 Google Drive 还是其他位置？能否下载？
- [ ] 代码里 GLoIR 是独立文件还是嵌在整体训练框架里？
- [ ] 它用的训练数据是什么（是否也是合成错位）？
- [ ] ResNet-50 双流特征提取对我们的实时性目标是否太重？（可能要换成轻量 backbone）