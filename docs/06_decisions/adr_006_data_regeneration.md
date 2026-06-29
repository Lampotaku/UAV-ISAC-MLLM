---
type: decision
status: Accepted
stage: all
date: 2026-06-29
commits: [f34129c]
related: [adr_001_unsloth_removal, adr_002_dpo_independent_ref, data_degeneracy, status]
---

# ADR 006: 数据重生 + Direct Preference Optimization 路线

## Context

SFT 训练完成后发现完全模态坍塌——所有 checkpoint 对所有输入输出相同的预测。EDA 揭示根因在于数据生成阶段：SCA-FP 求解器缺乏高度 trade-off 建模，导致 5000 个环境的"最优解"几乎完全同质（97.4% 向下、84.7% 满速、84.3% 满功率）。

同时，旧数据的 DPO preference pairs（186,896 对）同样基于退化分布——chosen 全是"砸地板"变体，rejected 也是"砸地板"变体。在此数据上训练 DPO 只会将极端策略夯得更死。

此外，SFT（Teacher Forcing + CE Loss）本身对连续物理量（3D 坐标）存在硬天花板——CE 将预测 5.1 和 5.2 视为跟预测"猫"和"狗"一样不同的错误。即使数据分布修复，SFT 仍面临根本局限。

## Decision

### 1. 立即停止旧数据 DPO 训练
旧数据完全废弃。DPO 进程应立即终止。

### 2. 求解器级修复：加入地面杂波
在 `SCAFPConfig` 新增 `ground_clutter_db: 12.0` 参数，创造高度维度的真实 trade-off。详见 [data_degeneracy.md](../02_training_log/data_degeneracy.md)。

### 3. 全量数据重生
用修复后的求解器重新生成 5000 环境 + DPO pairs。新增 EDA Diversity Check 为数据验收红线。

### 4. 跳过 SFT，直接 DPO
理由：
- step_150 的 SFT checkpoint 已具备 JSON 生成能力（文本 loss 正常）
- SFT（Teacher Forcing）对连续物理量有硬天花板
- DPO 是对比学习——比较 chosen vs rejected，天然适合学习"哪种位移更优"的相对排序
- 新数据的 preference pairs 来自 ground-clutter solver 的 Best-of-N，chosen 不再是"砸地板"
- DPO 不需要完美 ground truth——只需要相对偏好信号，对连续坐标的误差更宽容

### 5. DPO 配置沿用
```yaml
per_device_train_batch_size: 1     # 双模型 (policy + reference)
gradient_accumulation_steps: 16
dpo_beta: 0.1
dpo_mu: 0.05                       # SFT anchor — 防止 reward hacking
lambda_sep: 0.1
stage1_ckpt: step_150              # JSON 能力最好的 checkpoint
```

## Consequences

### 正面
- **数据质量从根上修复**：每个环境的解不再是同一个极端点
- **DPO 的对比学习比 SFT 更适合连续物理量**：不需要 CE 的离散化惩罚
- **step_150 在退化数据上仍有 JSON 能力**：迁移到新数据后不会从零开始

### 负面
- **数据重生需 ~3.5h × 2**（SFT + DPO pairs 重新生成）
- **step_150 的控制表征在新数据上不完全适配**：旧数据的方向分布与新数据不同，可能需要 DPO 训练更久
- **DPO 在新数据上首次尝试**：经验不足，可能需要调 β/μ

### 风险
- **DPO 在新数据上的 reward margin 不保证显著**：如果 ground-clutter 创造的分布差异不够大，chosen vs rejected 的偏好信号可能弱
- **DPO 的 SFT anchor (μ=0.05) 可能限制探索**：如果新数据的分布与旧 SFT 所学差异大，μ 需调低

## Backup Plans (若 DPO 也不达标)

按工程 ROI 排序：

| 优先级 | 方案 | 改动量 | 原理 |
|--------|------|--------|------|
| **P0** | Chain-of-Thought 注入 | 改 prompt 模板 | 让 LLM 在输出坐标前先做语义推理（"用户密集在东南角→应该向西北移"），利用 LLM 的 reasoning 能力缓冲数值预测 |
| **P1** | Regression Head (MLP) | ~200 行 | 从最后一层 hidden state 接 MLP (MSE) 直接回归连续坐标，完全绕过 CE 离散化。LLM 仍负责 A/P 矩阵（离散/半连续决策） |
| **P2** | Online RL with PPO | 大改动 | 用 SCA-FP 收敛后的 Sum-Rate 做 reward，让模型直接优化加速比。绕过所有模仿学习的局限，但计算代价高（每次 rollout 需运行 SCA-FP） |

### P0: CoT 注入详解

当前 prompt: `Current position: [...] Users: [...] → Output: {json}`
CoT prompt: `Current position: [...] Users: [...] → Think: analyze geometry → Output: {json}`

数据生成时要求 LLM 先用 Best-of-N 结果构造推理链，训练时学推理 + 输出联合分布。

**为什么优先 CoT**: 改动最小（只改 prompt 模板），利用了 LLM 的语义推理能力，不需要模型架构改动。

### P1: Regression Head 详解

```
Gemma 3 → last_hidden_state → MLP_head (MSE) → δ_q (continuous)
         → control tokens → ProjectionHead → δ_a, δ_p (discrete-ish)
```

LLM 仍负责需要"理解"的离散决策（关联矩阵 A、功率分配 P），但连续 3D 坐标由 MSE 回归头学习。MSE 对连续量有天然的 "距离度量"——预测偏差 0.1m 和 10m 的惩罚不同。

### P2: Online RL with PPO 详解

```
for each training step:
    1. Model generates warmstart (δ_q, δ_a, δ_p)
    2. SCA-FP runs from warmstart → converges or not
    3. Reward = f(iterations_saved, final_sum_rate)
    4. PPO update step
```

**优势**: 直接优化最终目标（加速比）；**风险**: 训练时 SCA-FP 运行开销大，reward 信号稀疏。

## 决策时间线

```
2026-06-26  SFT 训练完成（旧数据）
           ↓
2026-06-28  评估发现模态坍塌
           ↓
2026-06-29  EDA 确诊数据退化 → 求解器修复 → 此 ADR
           ↓
NOW        ⏳ 快速验证求解器修复 (50 envs)
           ↓
           全量数据重生 (~3.5h)
           ↓
           EDA 验收 (红线检查)
           ↓
           DPO 训练 (~5-10h)
           ↓
           评估 → 若不达标 → P0→P1→P2 逐级出牌
```
