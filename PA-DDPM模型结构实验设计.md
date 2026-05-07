# 基于模型结构改进的 DDPM InSAR 相位解缠实验设计

> 研究主题：在不提高数据分辨率、不增加训练数据数量的前提下，从 DDPM 去噪网络结构入手，提高 InSAR 相位解缠质量。

---

## 1. 实验设计原则

根据已有实验记录，当前最佳 DDPM 配置为：

- 输入尺寸：256×256
- 条件通道：3 通道，即 `sin(wrapped phase)`、`cos(wrapped phase)`、`coherence`
- 噪声步数：200
- 噪声调度：线性调度，β=1e-4→0.02
- 推理方式：DDIM-50
- Batch size：4
- AMP：开启
- 当前结果：Masked MAE = 7.55，Gradient MAE = 0.61，WinRate = 96%

已有实验说明：

1. 残差预测几乎无效。
2. 条件通道从 3ch 增加到 7ch 几乎无效。
3. 余弦噪声调度无明显提升。
4. 掩膜感知损失无明显提升。
5. 128→256 分辨率提升有效，但本阶段明确禁止继续依赖提高分辨率。

因此，后续实验必须遵循以下约束：

| 项目 | 固定要求 |
|---|---|
| 输入分辨率 | 固定 256×256，不再提高 |
| 训练数据数量 | 固定当前训练样本数量，不再增加 |
| 条件输入 | 固定 3ch：sin(wrapped)、cos(wrapped)、coherence |
| 噪声步数 | 固定 200 |
| 噪声调度 | 固定线性调度 |
| 推理方式 | 固定 DDIM-50 |
| 训练轮数 | 固定当前实验轮数，例如 80 epochs |
| Batch size | 固定 4，除非显存不足时才用梯度累积保持等效 batch |
| AMP | 固定开启 |

本阶段实验只允许改变：

1. DDPM 去噪 UNet 的主干结构；
2. 条件信息注入方式；
3. 注意力模块结构；
4. ResBlock / 图像恢复模块；
5. 相位任务专用结构；
6. 输出头结构。

---

## 2. 总体实验目标

| 指标 | 当前 DDPM-256 | 阶段目标 |
|---|---:|---:|
| Masked MAE | 7.55 | ≤ 7.20 |
| Gradient MAE | 0.61 | ≤ 0.57 |
| DDPM vs UNet 差距 | 0.76 rad | ≤ 0.40 rad |
| WinRate vs LS | 96% | ≥ 96% |
| Wrapped MAE | 未明确 | 不明显恶化 |

核心目标不是证明“更大数据、更高分辨率会更好”，而是证明：

> 通过面向相位解缠任务的 DDPM 去噪网络结构设计，可以在固定数据和固定分辨率下提升模型质量。

---

## 3. 基线模型

后续所有实验都以当前最佳模型为基线。

### 3.1 Baseline-DDPM 配置

| 模块 | 配置 |
|---|---|
| 图像尺寸 | 256×256 |
| 条件输入 | 3ch：sin(wrapped)、cos(wrapped)、coherence |
| DDPM timesteps | 200 |
| 采样 | DDIM-50 |
| UNet base channels | 64 |
| channel multipliers | [1, 2, 4] |
| ResBlocks | 2 |
| Attention | @32 |
| Batch size | 4 |
| AMP | 开启 |

### 3.2 基线结果

| 模型 | Masked MAE | Masked RMSE | Wrapped MAE | Grad MAE | WinRate |
|---|---:|---:|---:|---:|---:|
| DDPM-256 Baseline | 7.55 | 待补充 | 待补充 | 0.61 | 96% |

---

## 4. 第一组实验：UNet 主干容量结构消融

### 4.1 实验目的

验证当前 DDPM 质量受限是否来自 UNet 去噪网络容量不足。

DDPM 原始工作使用 UNet 作为核心去噪网络，通过输入 noisy sample 和 timestep 来预测噪声。因此，在固定数据和分辨率的情况下，UNet 的宽度、深度和感受野会直接影响去噪质量。

### 4.2 实验矩阵

| 实验编号 | 模型名称 | base channels | ResBlocks | channel multipliers | Attention | 目的 |
|---|---|---:|---:|---|---|---|
| A0 | Baseline-DDPM | 64 | 2 | [1,2,4] | @32 | 当前基线 |
| A1 | Wide-DDPM | 96 | 2 | [1,2,4] | @32 | 验证通道容量是否不足 |
| A2 | Deep-DDPM | 64 | 3 | [1,2,4] | @32 | 验证局部非线性建模是否不足 |
| A3 | MoreScale-DDPM | 64 | 2 | [1,2,4,4] | @32/@16 | 验证全局感受野是否不足 |
| A4 | WideDeep-DDPM | 96 | 3 | [1,2,4] | @32 | 综合增强容量 |
| A5 | Light-DDPM | 48 | 2 | [1,2,4] | @32 | 验证当前模型是否存在冗余或过拟合 |

### 4.3 预期分析

| 结果现象 | 解释 |
|---|---|
| A1 明显优于 A0 | 当前模型通道容量不足 |
| A2 明显优于 A0 | 当前模型局部非线性表达不足 |
| A3 明显优于 A0 | 当前模型全局相位趋势建模不足 |
| A4 明显优于 A1/A2 | 宽度和深度需要同时增强 |
| A5 接近 A0 | 当前模型可能不是容量瓶颈，而是结构瓶颈 |

### 4.4 推荐执行顺序

```text
A0 → A1 → A2 → A3 → A4 → A5
```

---

## 5. 第二组实验：条件融合结构改造

### 5.1 实验目的

已有实验表明，简单增加条件通道从 3ch 到 7ch 几乎没有提升。这说明问题不一定是条件信息不足，而可能是条件融合方式太浅。

当前方法大概率是：

```text
x_t 与 condition 在输入层 concat → UNet → 预测噪声
```

这种做法的问题是：

1. 条件信息主要在浅层进入网络；
2. 下采样后深层特征可能弱化条件约束；
3. coherence 与 sin/cos 相位通道被混在一起，模型未必能正确区分其物理含义；
4. 条件信息没有显式参与每个 ResBlock 的调制。

FiLM 的核心思想是用条件信息生成 feature-wise scale 和 shift，对中间特征进行调制。ControlNet 则说明在扩散模型中，空间条件可以通过独立条件分支和 zero convolution 稳定注入主网络。

### 5.2 实验矩阵

| 实验编号 | 模型名称 | 条件融合方式 | 结构说明 | 目的 |
|---|---|---|---|---|
| B0 | InputConcat-DDPM | 输入层直接 concat | 当前方法 | 基线 |
| B1 | MultiScaleCond-DDPM | 多尺度条件编码器 | 条件图单独编码成多尺度特征，注入 encoder/decoder | 防止条件信息只停留在浅层 |
| B2 | FiLMCond-DDPM | FiLM / AdaGN 调制 | 条件特征生成 γ、β，调制每个 ResBlock | 强化条件控制 |
| B3 | DualBranchCond-DDPM | 双分支条件编码 | sin/cos phase 分支与 coherence 分支分别编码后融合 | 避免 coherence 被相位通道淹没 |
| B4 | ControlBranch-DDPM | 条件控制分支 | 类似 ControlNet 的轻量条件分支，通过 zero conv 注入主干 | 稳定引入空间条件 |

### 5.3 推荐结构：FiLM/AdaGN 条件 DDPM

推荐重点实验 B2。

结构示意：

```text
x_t ───────────────────────► DDPM UNet ResBlocks ─────► noise prediction
                                  ▲
                                  │ scale, shift
condition = [sin, cos, coh] ─► Condition Encoder ──────► γ, β
                                  ▲
                                  │
time embedding ──────────────────┘
```

每个 ResBlock 的调制方式：

```text
h = GroupNorm(h)
h = h * (1 + gamma_cond_time) + beta_cond_time
h = SiLU(h)
h = Conv(h)
```

### 5.4 预期分析

| 结果现象 | 解释 |
|---|---|
| B1 提升 | 条件信息需要多尺度注入 |
| B2 提升 | 条件信息需要参与中间特征调制 |
| B3 提升 | coherence 与 phase 的物理属性应分开编码 |
| B4 提升 | 条件分支比输入拼接更适合扩散去噪 |

### 5.5 推荐执行顺序

```text
B0 → B2 → B1 → B3 → B4
```

---

## 6. 第三组实验：注意力结构消融

### 6.1 实验目的

InSAR 相位场具有局部连续性，也具有大范围坡度趋势。注意力模块可以帮助建模长距离依赖，但如果使用不当，也可能引入高频伪影或增加训练不稳定性。

因此，本组实验不应盲目增加 attention，而应系统比较不同注意力位置和类型。

### 6.2 实验矩阵

| 实验编号 | 模型名称 | 注意力设置 | 说明 | 目的 |
|---|---|---|---|---|
| C0 | Attn32-DDPM | attention @32 | 当前设置 | 基线 |
| C1 | NoAttn-DDPM | 无 attention | 纯卷积 UNet | 判断 attention 是否有副作用 |
| C2 | Attn16-DDPM | attention @16 | 只在更低分辨率层使用 attention | 强化全局趋势建模 |
| C3 | Attn32-16-DDPM | attention @32 + @16 | 两个尺度使用 attention | 同时建模中程和全局关系 |
| C4 | LinearAttn-DDPM | Linear Attention | 替代普通 self-attention | 降低显存和复杂度 |
| C5 | AxialAttn-DDPM | Axial Attention | 横向和纵向分解注意力 | 适合条纹方向结构 |
| C6 | WindowAttn-DDPM | Window Attention | 局部窗口注意力 | 强化局部结构恢复 |

### 6.3 预期分析

| 结果现象 | 解释 |
|---|---|
| C1 优于 C0 | 当前 attention 引入噪声或伪影，任务更适合卷积 |
| C2/C3 优于 C0 | 当前全局相位趋势建模不足 |
| C5 优于 C0 | 相位条纹具有方向性，轴向注意力更适合 |
| C6 优于 C0 | 局部窗口关系比全局关系更重要 |

### 6.4 推荐执行顺序

```text
C0 → C1 → C2 → C3 → C5 → C6
```

---

## 7. 第四组实验：图像恢复型 Block 替换

### 7.1 实验目的

InSAR 相位解缠本质上更接近图像恢复 / 图像到图像回归任务，而不是自然图像无条件生成任务。因此，可以把普通 DDPM ResBlock 替换为更适合图像恢复的模块。

NAFNet 等图像恢复网络表明，简洁的恢复型模块可以在去噪、去模糊等任务中取得较强效果，并且计算效率较高。

### 7.2 实验矩阵

| 实验编号 | 模型名称 | Block 类型 | 结构说明 | 目的 |
|---|---|---|---|---|
| D0 | ResBlock-DDPM | 原始 DDPM ResBlock | GroupNorm + SiLU + Conv | 基线 |
| D1 | DenseBlock-DDPM | Residual Dense Block | 增加密集连接 | 提高细节恢复能力 |
| D2 | GatedResBlock-DDPM | 门控残差块 | 使用 coherence 或条件特征生成 gate | 抑制低相干区域错误 |
| D3 | NAFBlock-DDPM | NAFBlock | 使用图像恢复型 block 替换 ResBlock | 提高恢复质量 |
| D4 | ConvNeXtBlock-DDPM | ConvNeXt Block | Depthwise Conv + MLP | 增强局部建模能力 |
| D5 | HybridBlock-DDPM | ResBlock + NAFBlock | 混合结构 | 平衡稳定性和恢复能力 |

### 7.3 推荐结构：Coherence-Gated ResBlock

结构示意：

```text
feature ───────────────► Conv ───────────────► main feature
condition/coherence ───► Gate Encoder ───────► sigmoid gate
main feature × gate ───► output
```

其中 gate 可以来自 coherence feature：

```text
gate = sigmoid(Conv(coherence_feature))
out = residual + main_feature * gate
```

### 7.4 预期分析

| 结果现象 | 解释 |
|---|---|
| D2 提升 | 相干性信息确实适合控制特征传播 |
| D3 提升 | 图像恢复型模块比生成式 ResBlock 更适合本任务 |
| D4 提升 | 局部结构建模能力不足是瓶颈 |
| D5 提升 | 混合结构可以兼顾稳定性与恢复能力 |

### 7.5 推荐执行顺序

```text
D0 → D2 → D3 → D4 → D5
```

---

## 8. 第五组实验：相位任务专用结构

### 8.1 实验目的

这是最适合作为论文创新点的一组实验。相比单纯修改 UNet 宽度或注意力模块，面向 InSAR 相位解缠的结构设计更有针对性。

建议构建：

```text
PA-DDPM: Phase-Aware Conditional DDPM
```

核心模块包括：

1. Phase-Aware ResBlock；
2. Coherence-Gated Skip Connection；
3. Phase Gradient Branch；
4. Dual-Head ε/x0 Prediction。

---

### 8.2 结构 1：Phase-Aware ResBlock

普通 ResBlock：

```text
Conv → Norm → SiLU → Conv
```

Phase-Aware ResBlock：

```text
x_t feature
+ timestep embedding
+ wrapped phase embedding
+ coherence embedding
→ phase-aware feature
```

具体可以把条件编码器输出加入每个 ResBlock：

```text
cond_embed = CondEncoder([sin_wrapped, cos_wrapped, coherence])
time_embed = TimeEmbedding(t)
mod_embed = MLP(time_embed + cond_embed)
ResBlock(feature, mod_embed)
```

预期作用：

- 让模型在每一层都知道当前 wrapped phase 和 coherence；
- 避免条件信息只在输入层生效；
- 强化 DDPM 对相位结构的感知能力。

---

### 8.3 结构 2：Coherence-Gated Skip Connection

普通 UNet skip connection：

```text
decoder_feature = concat(decoder_feature, encoder_skip)
```

改进后：

```text
gate = sigmoid(Conv(coherence_feature))
gated_skip = encoder_skip × gate
decoder_feature = concat(decoder_feature, gated_skip)
```

预期作用：

- 高相干区域保留更多 encoder 细节；
- 低相干区域抑制不可靠纹理；
- 减少低相干区域引入的错误解缠结构。

---

### 8.4 结构 3：Phase Gradient Branch

增加一个专门提取 wrapped phase 梯度的轻量分支：

```text
sin(wrapped), cos(wrapped)
        │
        ▼
learnable gradient conv / Sobel-like conv
        │
        ▼
gradient feature
        │
        ▼
inject into decoder
```

推荐输入：

```text
phase_input = [sin_wrapped, cos_wrapped]
```

推荐输出：

```text
grad_feature_x, grad_feature_y
```

预期作用：

- 强化模型对相位跳变边界的识别；
- 改善 Gradient MAE；
- 帮助模型区分真实相位坡度和 wrap 造成的跳变。

---

### 8.5 结构 4：Dual-Head ε/x0 Prediction

当前 DDPM 通常只预测噪声：

```text
UNet(x_t, t, cond) → ε_pred
```

改进为双头输出：

```text
UNet shared feature → ε_pred head
                    → x0_pred head
```

其中：

- `ε_pred head` 保持 DDPM 原始训练目标；
- `x0_pred head` 直接预测干净 unwrapped phase；
- 训练时使用辅助损失约束 x0 head。

建议损失：

```text
L_total = L_epsilon + λ_x0 * L_x0
```

推荐初始权重：

```text
λ_x0 = 0.1
```

预期作用：

- 保留 DDPM 生成式去噪训练；
- 同时让网络中间特征更接近最终相位解缠目标；
- 缓解“噪声预测目标与最终 MAE 指标不完全一致”的问题。

---

## 9. 第五组完整实验矩阵：PA-DDPM 消融

| 实验编号 | 模型名称 | Phase-Aware ResBlock | Coherence-Gated Skip | Phase Gradient Branch | Dual-Head ε/x0 | 目的 |
|---|---|---|---|---|---|---|
| P0 | Baseline-DDPM | 否 | 否 | 否 | 否 | 当前基线 |
| P1 | PARes-DDPM | 是 | 否 | 否 | 否 | 验证相位感知 block |
| P2 | CGSkip-DDPM | 否 | 是 | 否 | 否 | 验证相干性门控 skip |
| P3 | GradBranch-DDPM | 否 | 否 | 是 | 否 | 验证相位梯度分支 |
| P4 | DualHead-DDPM | 否 | 否 | 否 | 是 | 验证双头输出 |
| P5 | PA-CG-DDPM | 是 | 是 | 否 | 否 | 验证 phase-aware + coherence gate |
| P6 | PA-CG-Grad-DDPM | 是 | 是 | 是 | 否 | 验证三结构组合 |
| P7 | Full PA-DDPM | 是 | 是 | 是 | 是 | 最终模型 |

---

## 10. 总推荐执行路线

为了控制实验数量，建议不要一次性做所有结构，而是按如下路线执行。

### 阶段 1：确认普通 UNet 是否容量不足

```text
A0 Baseline-DDPM
A1 Wide-DDPM
A2 Deep-DDPM
A3 MoreScale-DDPM
```

如果 A1/A2/A3 都提升不明显，说明不是简单容量问题，应进入条件融合和相位专用结构。

---

### 阶段 2：验证条件融合是否是瓶颈

```text
B0 InputConcat-DDPM
B2 FiLMCond-DDPM
B3 DualBranchCond-DDPM
B4 ControlBranch-DDPM
```

重点看 B2 是否优于 B0。若 B2 有明显提升，说明后续所有相位专用结构都应基于 FiLM/AdaGN 条件注入。

---

### 阶段 3：验证任务专用结构

```text
P0 Baseline-DDPM
P1 PARes-DDPM
P2 CGSkip-DDPM
P3 GradBranch-DDPM
P4 DualHead-DDPM
P7 Full PA-DDPM
```

这一阶段最适合作为论文主要创新实验。

---

### 阶段 4：可选强化模块

```text
D2 GatedResBlock-DDPM
D3 NAFBlock-DDPM
C5 AxialAttn-DDPM
C6 WindowAttn-DDPM
```

这部分可以作为补充实验，不建议放在主创新前面。

---

## 11. 最终建议主模型

最终模型建议命名为：

```text
PA-DDPM: Phase-Aware Conditional Denoising Diffusion Probabilistic Model
```

中文名称：

```text
相位感知条件去噪扩散概率模型
```

完整结构：

```text
Input:
    x_t noisy unwrapped phase
    condition = [sin(wrapped), cos(wrapped), coherence]
    timestep t

Backbone:
    Phase-Aware Conditional UNet

Modules:
    1. Multi-Scale Condition Encoder
    2. FiLM/AdaGN Conditional ResBlocks
    3. Coherence-Gated Skip Connections
    4. Phase Gradient Branch
    5. Dual-Head ε/x0 Prediction

Output:
    ε_pred: predicted noise
    x0_pred: auxiliary clean unwrapped phase
```

---

## 12. 推荐主实验表格模板

| 模型 | 结构改动 | Masked MAE | Masked RMSE | Wrapped MAE | Grad MAE | WinRate | 参数量 | 推理时间 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline-DDPM | 原始 UNet | 7.55 | 待补充 | 待补充 | 0.61 | 96% | 待补充 | 待补充 |
| Wide-DDPM | base 64→96 |  |  |  |  |  |  |  |
| Deep-DDPM | resblocks 2→3 |  |  |  |  |  |  |  |
| FiLMCond-DDPM | 条件调制 ResBlock |  |  |  |  |  |  |  |
| CGSkip-DDPM | 相干性门控 skip |  |  |  |  |  |  |  |
| GradBranch-DDPM | 相位梯度分支 |  |  |  |  |  |  |  |
| DualHead-DDPM | ε/x0 双头输出 |  |  |  |  |  |  |  |
| Full PA-DDPM | 完整相位感知结构 |  |  |  |  |  |  |  |

---

## 13. 推荐消融表格模板

| 模型 | FiLM 条件调制 | Coherence Gate | Gradient Branch | Dual Head | Masked MAE | Grad MAE | 结论 |
|---|---|---|---|---|---:|---:|---|
| P0 Baseline | × | × | × | × | 7.55 | 0.61 | 基线 |
| P1 | √ | × | × | × |  |  | 验证条件调制 |
| P2 | × | √ | × | × |  |  | 验证相干性门控 |
| P3 | × | × | √ | × |  |  | 验证梯度分支 |
| P4 | × | × | × | √ |  |  | 验证双头结构 |
| P7 Full | √ | √ | √ | √ |  |  | 完整模型 |

---

## 14. 论文中可使用的实验设计表述

可以在论文中这样描述：

> 为避免模型性能提升来源于输入分辨率扩大或训练数据规模增加，本文在后续实验中固定训练样本数量、输入尺寸、训练轮数、扩散步数、噪声调度和采样策略，仅从 DDPM 去噪网络结构入手开展消融研究。首先分析 UNet backbone 的宽度、深度和注意力位置对相位解缠精度的影响；其次，将传统输入拼接式条件注入替换为多尺度条件编码和 FiLM/AdaGN 特征调制，以增强 wrapped phase 与 coherence 信息在不同尺度下的约束作用；进一步地，本文设计相干性门控跳跃连接和相位梯度分支，以提高模型对低相干区域和相位跳变区域的建模能力；最后，引入 ε/x0 双头预测结构，使网络在保持扩散噪声预测目标的同时，直接学习干净解缠相位表示，从而提升 DDPM 在 InSAR 相位解缠任务中的结构一致性和恢复精度。

---

## 15. 推荐论文创新点写法

本文模型结构创新可总结为：

1. **相位感知条件调制机制**  
   将 wrapped phase 的 sin/cos 表示和 coherence 图编码为多尺度条件特征，并通过 FiLM/AdaGN 方式注入 DDPM 的每个 ResBlock，而不是仅在输入层拼接条件通道。

2. **相干性门控跳跃连接**  
   利用 coherence 图生成空间门控权重，对 UNet encoder-decoder skip connection 进行调制，降低低相干区域不可靠纹理对解缠结果的干扰。

3. **相位梯度分支**  
   通过独立分支提取 wrapped phase 的梯度结构，增强模型对相位跳变、条纹密集区和边界区域的识别能力。

4. **ε/x0 双头预测结构**  
   在保留 DDPM 噪声预测主任务的同时，引入干净相位辅助输出头，使网络特征更加贴合最终解缠相位恢复目标。

---

## 16. 实验风险与应对策略

| 风险 | 可能原因 | 应对策略 |
|---|---|---|
| Wide-DDPM 显存增加明显 | base channels 增大 | 使用梯度累积保持等效 batch，不改变数据和分辨率 |
| Deep-DDPM 收敛变慢 | ResBlocks 增加 | 保持 epoch 不变先比较，必要时只报告同 epoch 性能 |
| FiLMCond-DDPM 训练不稳定 | 条件调制过强 | 将 γ 初始化为接近 0，采用 residual modulation |
| Coherence Gate 导致细节损失 | gate 过度抑制 skip | 使用 `skip * (1 + sigmoid(gate))` 或限制 gate 范围 |
| Gradient Branch 无提升 | 梯度特征注入位置不合适 | 改为 decoder 中后层注入，而不是最浅层注入 |
| Dual Head 影响 ε 预测 | x0 辅助损失过大 | λ_x0 从 0.05、0.1、0.2 小范围扫描 |

---

## 17. 最终推荐最小实验集

如果时间有限，建议只做以下 8 个实验：

| 顺序 | 实验 | 必要性 |
|---:|---|---|
| 1 | Baseline-DDPM | 必须，作为对照 |
| 2 | Wide-DDPM | 验证容量 |
| 3 | Deep-DDPM | 验证深度 |
| 4 | FiLMCond-DDPM | 验证条件调制 |
| 5 | CGSkip-DDPM | 验证相干性门控 |
| 6 | GradBranch-DDPM | 验证相位梯度分支 |
| 7 | DualHead-DDPM | 验证结构化输出 |
| 8 | Full PA-DDPM | 最终模型 |

最核心的实验链路是：

```text
Baseline-DDPM
→ FiLMCond-DDPM
→ CGSkip-DDPM
→ GradBranch-DDPM
→ DualHead-DDPM
→ Full PA-DDPM
```

这条链路最能体现“从模型结构提升 DDPM 相位解缠质量”的研究主线。

---

## 18. 参考依据

1. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.  
   https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

2. Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, 2017/2018.  
   https://arxiv.org/abs/1709.07871

3. Zhang et al., *Adding Conditional Control to Text-to-Image Diffusion Models*, ICCV 2023.  
   https://arxiv.org/abs/2302.05543

4. Chen et al., *Simple Baselines for Image Restoration / NAFNet*, ECCV 2022.  
   https://arxiv.org/abs/2204.04676

5. 用户已有实验记录：`实验记录(1).md`。

