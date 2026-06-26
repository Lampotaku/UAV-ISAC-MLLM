# Phase 1 CTL-only Warmup — 当前状态与分析

**2026-06-26**

## 背景

UAV-ISAC-MLLM：用 Gemma 3 12B (LoRA + 约束投影头) 为 SCA-FP 数值优化器提供智能 warmstart。

**Step-200 训练 checkpoint 评估结果**：

| 指标 | 值 | 期望 |
|------|-----|------|
| 文本生成 | 模式坍塌，输出乱码 | 有效 JSON |
| Control sensitivity (±10m) | 0.0000 | > 0.1 |
| SCA-FP 加速比 | 1.0×（无加速） | ≥ 1.5× |

## 根因分析

### 梯度密度失衡

```
3456 个文本 token → CE loss → dense token-level gradients
   8 个控制 token → 控制 loss → sparse token-level gradients
                                                  ↓
                           梯度密度比 = 3456:8 = 432:1
                                                  ↓
                    CE 完全主导训练，控制信号被淹没
```

### Mean Pooling 信息破坏

`ControlReadout.mean(dim=1)` 将 8 个 control token 的 hidden states 平均为一个向量。这把 token 专门化（position / layout / power）的空间结构完全抹平。

### 结论

控制信号根本没有有效进入 representation。模型只学会了"UAV 存在"，没学会 UAV 在哪、怎么优化 UAV 位置。

## 修复：Layer 1 架构改动

### 1. Attention Pooling

```
旧: mean_pool([h1, h2, ..., h8]) → δ̃     (平均抹平)
新: attn(query, [h1..h8]) → δ̃              (可学习 query 动态聚焦)
```

允许不同 control token 专门化不同子任务。初始化 `query ~ N(0, 0.02)` 使初始注意力接近均匀。

### 2. Phase 1 CTL-only Warmup

完全关闭 CE loss，只训练控制 loss：`L = λ_ctl × L_ctl`。强制 LoRA 学会将环境信息编码到 control token hidden states。

**切换条件**：跨环境 sensitivity > 0.1（不是 loss_ctl 值）。

```
Phase 1 (CTL-only, max 400 steps)
  loss = 1.0 × loss_ctl
  每 50 步: check cross-env sensitivity
    ├── sens > 0.1 → 自动切 Phase 2
    └── sens ≤ 0.1 → 继续 Phase 1
```

### 3. Cross-Env Sensitivity 测试

```
旧: 同一个 prompt，只改 q_current tensor (±10m)
    → 投影头裁剪是恒等映射（UAV 不靠近边界）
    → sens 永远 = 0

新: 两个独立环境 (seed=42, 43)
    → 各自建 prompt，各自跑 forward
    → sens = ||delta_q(env_b) - delta_q(env_a)|| / ||delta_q(env_a)||
    → 真正测模型是否编码了环境特定信息
```

## 实验观察

### Run 1: LoRA LR = 2e-4（默认），5000 样本

| Step | loss_ctl | sens |
|------|----------|------|
| 51   | 24.77    | 0.0000 |
| 79   | 24.69    | 0.0056 |
| 109  | 37.80    | 0.0102 |

观察：
- sens 从 0 → 0.01：control pathway **不是完全死的**，这是关键信号
- 但增长速度极慢，线性外推需 ~2000 步才到 0.1
- loss_ctl 不降反升（24 → 37），不是过拟合，是 representation 重构期的 temporary worsening

### 诊断结论

问题已从"架构完全错误"缩小到 **optimization dynamics 不匹配**：

- Phase 1 只有 regression loss，无 CE 噪声，无 catastrophic forgetting 风险
- 本质上在做 conditioned feature learning，不是 stable language finetune
- 2e-4 的 LoRA LR（为 CE 任务调优）对这个 regime 太保守

### Run 2: LoRA LR = 5e-4（当前进行中）

改动：
```yaml
phase1:
  lr_lora: 5.0e-4    # 2.5× 提升，保守起步防 shortcut
  # (Phase 2 恢复 2.0e-4)
```

| 参数组 | Phase 1 LR | Phase 2 LR |
|--------|-----------|-----------|
| Projection Head (随机初始化) | 1e-3 | 1e-3 |
| LoRA (pretrained manifold) | 5e-4 | 2e-4 |

设计考量：
- 5e-4 而非 1e-3：LoRA 在 12B pretrained manifold 上做 perturbation，太高 LR 可能学会 shortcut（如只 encode UAV density 而非真正连续几何结构）
- 观察 sens 斜率，不是 loss_ctl 绝对值。如果 5e-4 仍不够（sens 卡在 < 0.02），再升到 1e-3

## Sens 值的含义

| sens | 含义 |
|------|------|
| 0.0000 | 模型对所有环境输出相同 delta_q — 没学会 |
| 0.01-0.05 | 开始感知环境差异，但极微弱 — 在渗透 |
| > 0.1 | 两个环境输出差异 ≥ 10% — 控制表示成形 |
| > 0.2 | 高度环境敏感 — 可以安全进入 Phase 2 |

## 后续

1. 等待 Run 2 的 step 50/100/150 sens 曲线
2. 如果 sens 斜率明显改善 → 验证 architecture 路线正确
3. 如果 sens 仍卡在 < 0.02 → 尝试 lr_lora = 1e-3，或考虑第二层改动（8 → 32 control tokens）
4. Phase 2（Joint SFT+CTL）在 sens > 0.1 后自动启动
5. 第一个 Phase 2 checkpoint (step 200) 重新跑 eval_generation.py Part 3 检查 SCA-FP 加速比
