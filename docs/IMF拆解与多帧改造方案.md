# IMF (C-MPDR) 拆解 与 多帧改造方案

> 来源：**Improving Misaligned Multi-Modality Image Fusion with One-stage
> Progressive Dense Registration**（Di Wang, Jinyuan Liu, Long Ma, Risheng Liu, Xin Fan）
> **正式出处**：IEEE TCSVT 2024, Vol.34, No.11, pp.10944-10958（中科院一区）
> arXiv: 2308.11165
> **代码（已开源，完整）**：https://github.com/wdhudiekou/IMF
> 本地代码：`E:\download\源码\VF-Bench-main\third_party\IMF\`
> 本地 PDF：`E:\论文\配准、融合一体化论文\Improving Misaligned ... .pdf`

**代码对应关系**：`models/mpdrnet.py` 里有 `PFF`(L189) / `DFF`(L225) / `MPDRNet`(L290)；`data/generate_affine_deform_data.py` 是官方合成错位脚本。

**为什么拆这篇**：它提出的"单阶段渐进"直接可以时序化——把**多尺度**换成**多帧**。

---

## 1. 它反对什么

> 现有做法要么用**分离的粗配准 + 精配准两阶段**，要么**直接预测全尺寸形变场**。
> 前者不够紧凑，后者不够准。

它的方案：**在一个网络里、用一次优化完成 coarse-to-fine**。

动机（很关键）：
```
深层特征（低分辨率、大感受野）→ 表征全局结构 → 适合修大位移 → 粗配准
浅层特征（高分辨率、小感受野）→ 表征局部细节 → 适合修小位移 → 精配准
```

---

## 2. 整体结构（Fig.2）

```
IR(失真) + VIS
     │
     ▼
  CPST（跨模态风格迁移，冻结）        ← 来自 UMF，把 VIS 翻译成"伪红外" Î_ir
     │
     ▼
  MPDRN（多尺度渐进密集配准子网络）    ← 把 I_ir 配准到 Î_ir（近似单模态配准）
     │   ├─ 可学习双流金字塔特征提取器
     │   ├─ 多尺度解码器
     │   ├─ DFF × 多级   ← 核心1
     │   └─ PFF × 多级   ← 核心2
     ▼
  I_reg（对齐后的红外）
     │
     ▼
  TCF（Transformer-Conv 融合子网络，含 DAU 双注意力）
     ▼
  融合图
```

K = 4 个尺度。训练时 C-MPDR 和 TCF **联合训练**，CPST **冻结**。

---

## 3. DFF（Deformable Field Fusion）—— 最核心

**要解决的问题**：直接预测全尺寸形变场不准 → 改成**逐尺度累积、重加权融合**

**第 1 步：把之前各尺度的粗形变场插值到当前尺度**（Eq.4）

```
φ_i↑ = P_{2^{i-k}}(φ_i)        i ∈ [K, K−1, ..., k]
```
`P` 是双线性插值，把 φ_i 放大到 φ_k 的尺寸。

**第 2 步：算权重向量**（Eq.5）

```
V_w = M_CRB( [φ_K↑, φ_{K−1}↑, ..., φ_k↑] )
```
把所有插值后的形变场**拼接**，过 **CRB（Conv-ReLU Blocks）** 得到权重向量。

**第 3 步：拆成每个场对应的标量权重**（Eq.6）

```
ω_t = M_CSB^[t]( V_w )          M_CSB = Conv-Sigmoid Block
```

**第 4 步：加权求和 + 向量积分，得到当前尺度精修后的场**（Eq.7）

```
φ_{k−1} = V( Σ_i Σ_t  ω_t ⊗ φ_i↑ )
```
- `⊗` 逐元素乘
- `V(·)` = **vector integration**（向量积分），实现**递归 warp**

**一句话理解**：
> 不是相加、不是拼接，而是让网络**自己学出每个尺度形变场该占多少权重**，
> 然后加权融合 + 递归累积。越靠后的尺度越精细。

---

## 4. PFF（Progressive Feature Fine）

**要解决的问题**：DFF 只复用了形变场，但**准不准还取决于特征质量**。

**第 1 步：三路特征拼接**（Eq.3，以第 K−1 尺度为例）

```
c^{K−1}_refine = PFF( [ c^{K−1}_warp , d^K↑2 , c̃^{K−1} ] )
```
| 特征 | 含义 |
|------|------|
| `c^{K−1}_warp` | 用 φ^{K−1} **warp 之后**的源（失真红外）特征 |
| `d^K↑2` | 上一层**解码特征**上采样 2 倍 |
| `c̃^{K−1}` | **伪红外**（参考）特征 |

**第 2 步：算三路权重**（Eq.8）

```
s1, s2, s3 = Softmax( M_CRB( c^{K−1}_mix ) )
```

**第 3 步：加权合并 + 通道注意力精修**（Eq.9）

```
c^{K−1}_w      = [ s1·c^{K−1}_warp , s2·d^K↑2 , s3·c̃^{K−1} ]
c^{K−1}_refine = c^{K−1}_w ⊗ A_channel( c^{K−1}_w )
```

**"渐进"的含义**：
> 早期尺度已经把固有偏移修得差不多了，**当前 PFF 只需要关注剩下的那点偏移**。

---

## 5. 损失函数（⚠️ 不需要形变场真值）

**双向相似度损失**（Eq.10）：

```
L_sim = ‖ψ_j(I^reg) − ψ_j(Î_ir)‖₁  +  λ_rev · ‖ψ_j(φ∘Î_ir) − ψ_j(I_ir)‖₁
                    ↑ 正向                                ↑ 反向（λ_rev = 0.2）
```
- 正向：配准后的红外 靠近 伪红外
- 反向：把伪红外用反向形变场拉回去，靠近 源失真红外

**平滑损失**（Eq.11）：`L_smooth = ‖∇φ‖₁`

**总损失**（Eq.12）：`L_reg = L_sim + λ_sm · L_smooth`，`λ_sm = 10`

> 💡 **重要**：IMF 靠"风格迁移 + 双向相似度"，**不需要形变场真值**。
> 而 OS-RFS 需要 EPE 真值。**你的数据两者都能满足** —— 你有真值，所以两条路都走得通。

---

## 6. ⭐ 多帧改造方案（你的创新点）

### 核心转换

```
图像级（IMF）:   同一帧的多个尺度    → 重加权融合
视频级（你的）:  同一帧的多个尺度  +  历史帧  → 重加权融合
                        ↑
                  在这一步插入时序
```

### 方案 A：DFF → TFF（Temporal Field Fusion）

**原版**（只看当前帧的多尺度）：
```
φ_K(t), φ_{K−1}(t), ..., φ_k(t)   →  DFF  →  φ_{k−1}(t)
```

**改造后**（加入历史帧）：
```
当前帧: φ_K(t),   φ_{K−1}(t),   ..., φ_k(t)
历史帧: φ_K(t−1), φ_{K−1}(t−1), ..., φ_k(t−1)
                  ↓
              全部拼接 → CRB → 权重向量
                  ↓
        加权求和 → φ_{k−1}(t)
```

**关键改动**：权重计算时把历史帧的形变场一起放进去。

```python
# 伪代码
fields = []
for scale_i in range(K, k-1, -1):
    fields.append(interp(phi[scale_i][t], target_size))     # 当前帧
    if use_temporal:
        fields.append(interp(phi[scale_i][t-1], target_size))  # 历史帧 ← 新增
V_w = CRB(cat(fields))
weights = [ConvSigmoid(V_w) for _ in fields]
phi_k_minus_1 = vector_integration(sum(w * f for w, f in zip(weights, fields)))
```

**好处**：
- 历史帧的形变场会**约束**当前帧不要突然跳变 → 天然的时序平滑
- 不需要额外加一个平滑损失模块
- 和 IMF 的思路一脉相承（都是"重加权融合"）

### 方案 B：PFF → 时序特征精修

**原版**：
```
c^{K−1}_refine(t) = PFF([ c_warp(t), d↑2(t), c̃(t) ])
```

**改造后**（多一路历史特征）：
```
c^{K−1}_refine(t) = PFF([ c_warp(t), d↑2(t), c̃(t), c_refine(t−1) ])
                                                      ↑
                                          前一帧的精修特征
```

**更好**：不要简单拼接，而是用你已有的 `SelectiveScanTemporal` 沿时间聚合：

```python
# 把 T 帧的特征堆成 [B, C, T, H, W]，用 SSM 沿时间轴扫描
c_hist = ssm_block( torch.stack([c_refine(t-2), c_refine(t-1)], dim=2) )
c_refine(t) = PFF([ c_warp(t), d↑2(t), c̃(t), c_hist(t) ])
```

→ **复用你现有的 mamba_block.py，不用新写时序模块。**

### 方案 C：损失加时序项

```
L_time = ‖φ(t) − φ(t−1)‖₁          （一阶，抑制抖动）
       + ‖φ(t) − 2φ(t−1) + φ(t−2)‖₁  （二阶，抑制加速度）
```
占位在 IMF 的 `L_reg = L_sim + λ_sm·L_smooth + λ_t·L_time`

---

## 7. 和你现有代码的对接

### 现状
```
IR 5帧 + VIS 5帧
  → encoder_1/encoder_2 → feat
  → DCN(帧间时序对齐)
  → Mamba SSM(T=3 时序)
  → decoder → 融合
```

### 目标
```
IR 5帧 + VIS 5帧
  → 【新增】TFF + 时序 PFF 配准模块       ← 借 IMF 的 DFF/PFF
  → 对齐后的 IR', VIS'
  → encoder → DCN → Mamba SSM → decoder → 融合
```

### 建议的模块接口

```python
class TemporalMultiScaleRegistration(nn.Module):
    """
    输入:
        ir  : [B, T, 1, H, W]
        vis : [B, T, 3, H, W]
    输出:
        ir_aligned : [B, T, 1, H, W]
        fields     : list of 多尺度形变场（用于损失）
    """
    def forward(self, ir, vis):
        # 1. 多尺度特征金字塔（双流）
        # 2. 逐尺度：TFF 融合多尺度+多帧形变场
        # 3. 逐尺度：PFF 精修特征（含历史帧，用 SelectiveScanTemporal）
        # 4. vector integration → 最终形变场
        # 5. STN warp
        ...
```

---

## 8. 待办

- [ ] 用 `E:\download\源码\VF-Bench-main\data\VTMOT_misaligned` 验证配准模块
- [ ] 先实现**无时序版**（纯 IMF 复现），拿到 MEE 基线
- [ ] 再加时序（TFF / 时序 PFF），对比 MEE 与抖动
- [ ] 消融：只加 TFF、只加时序 PFF、两者都加
- [ ] 报告时序一致性指标（帧间 warp 差分的标准差）

---

## 9. 关键提醒

1. **IMF 不需要形变场真值** —— 但你有真值，可以考虑**额外加 EPE 损失**加速收敛
2. **不要在"图像级"上和 IMF 硬拼** —— 它已经做得很好了
3. **你的差异化只在"视频/时序"** —— 所以 TFF/PFF 的时序改造是论文的核心
