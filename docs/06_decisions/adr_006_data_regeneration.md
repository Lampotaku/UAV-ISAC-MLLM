---
type: decision
status: Accepted
stage: all
date: 2026-06-29
last_updated: 2026-06-29
commits: [f34129c, 7cedb02]
related: [adr_001_unsloth_removal, adr_002_dpo_independent_ref, data_degeneracy, status, CONTEXT.md]
grilling_rounds: 5
---

# ADR 006: 数据重生 + DPO 路线 (5 轮 Grilling 火烤后终稿)

## Context

SFT 训练完成后发现完全模态坍塌——所有 checkpoint 对所有输入输出相同的预测。EDA 揭示根因在于数据生成阶段：SCA-FP 求解器缺乏高度 trade-off 建模，导致 5000 个环境的"最优解"几乎完全同质。

经过 **5 轮领域模型火烤 (Grilling)**，以下架构决策已被精确定义并消除歧义。

## Decision: 数据重生全流程

### Phase 0: ε 标定 (5 min)

在正式生成前，必须运行 `scripts/calibrate_epsilon.py`：

```
对 50 个随机环境测试 ε ∈ {0.5, 1.0, 2.0, 4.0, 8.0}m
选择使回弹步数出现最大区分度（方差）的 ε
预期最佳值：1.0-2.5m
```

**原理**：ε 太小 → 所有候选 1-2 步滑回谷底（无区分度）。ε 太大 → 跳出原 basin 进入未知惩罚区（无区分度）。要找的是"盆地边缘"的特征尺度。

### Phase 1: 暴力数据生成 (20,000 环境, ~2-3h)

```bash
python scripts/generate_data.py \
    --num-envs 20000 --num-restarts 10 \
    --output-dir /root/autodl-tmp/data/full20000_v2 \
    --num-workers 70
```

每个环境的内部逻辑：

```
FOR each environment:
    1. 10 次 Random Restart SCA-FP → 10 个局部最优解
    2. Pareto 过滤：
       - 丢弃 Sum-Rate 不如 [0,0,0] 不动方案的解
       - 丢弃低于全局最高 Sum-Rate 5% 以上的劣质坑
    3. 提取 Top-3 候选 q*₁, q*₂, q*₃
    4. 微扰回弹测试：对每个候选 +ε 随机扰动 → 重跑 SCA-FP
       步数最少者 → Chosen
    5. 构造 Rejected 池 (混合策略)：
       - Design B (~70%)：次优局部解的 δ_q (同一 Best-of-N)
       - Design A' (~30%)：启发式物理陷阱：
         a. 短视直线 (Greedy Line) → clip_to_physics_bounds
         b. 原地不动 [0,0,0] → clip_to_physics_bounds
         c. 旧世界残影 [0,0,±15] → clip_to_physics_bounds
       ⚠️ 所有 Rejected 必须经过 Constraint Projections 洗礼
    6. 写入 .jsonl — Chosen + Rejected 打包
```

**计算预算**：20,000 × 13 SCA-FP 调用 = 260,000 次。70 workers 下预估 25 min (理想) 到 2-3h (含杂波非凸收敛困难)。

### Phase 2: 质量闸门 → Top-5000 精选

```
按 Chosen-Rejected Composite Score Gap 排序
取前 5,000 名作为最终 DPO 训练集
```

- **不设绝对阈值**（如 >5% gap），避免在困难环境中全军覆没
- **Yield Rate 预估 25-35%** → 20,000 生成量确保选出 5,000 条"黄金数据"
- Gap 太小（<1%）的环境缺乏偏好信号强度，对 DPO 是噪声

### Phase 3: Masked DPO 训练

**关键架构发现**（Grilling Q12）：DPO loss 操作在**文本 token log-probabilities**，而非投影头输出空间。DPO 比较的是"生成包含 good δ_q 的 JSON 文本"vs"生成包含 bad δ_q 的 JSON 文本"的概率。

**Masked DPO 实现**：在 `dataset.py` tokenization 阶段：
1. 正则匹配找到 JSON 中 `"delta_a"` 和 `"delta_p"` 对应的字符区间
2. 映射到 token indices
3. 将对应 label 设为 `-100`（ignore index）
4. DPO 的 `_compute_logprob` 自动跳过这些 token

**效果**：梯度只在 δ_q 相关的 token 上产生偏好拉扯。δ_a 和 δ_p 变成"背景音"——它们在多次 SCA-FP Restart 间几乎无差异，全维度 DPO 会稀释偏好信号。

### DPO 配置

```yaml
per_device_train_batch_size: 1     # 双模型 (policy + reference)
gradient_accumulation_steps: 16
dpo_beta: 0.1
dpo_mu: 0.05                       # SFT anchor
lambda_sep: 0.1
stage1_ckpt: step_150              # JSON 能力最好的 checkpoint
```

## 核心概念澄清 (Grilling 成果)

| 概念 | 旧理解 | 精确定义 |
|------|--------|----------|
| **Warmstart** | 模型预测最优解 | q_current + δ_q̂ 作为 SCA-FP 初始点。MSE 代理损失——假设最优解邻域 = 好起点 |
| **Solver 角色** | 同一个类调用 | Oracle/Simulator/Evaluator 必须严格同一 SCAFPConfig 实例 |
| **Chosen 选择** | Sum-Rate 最高 | **微扰回弹步数最少**（盆地宽度 > 谷底深度） |
| **Rejected 构造** | 模型坍塌输出 | 70% SCA-FP 次优解 + 30% 启发式物理陷阱 (经约束投影) |
| **DPO 操作空间** | 投影头输出 | **文本 token log-probabilities** — 偏好信号通过 JSON 文本传递 |
| **δ_a/δ_p 在 DPO 中** | 参与全维度 DPO | **Masked DPO** — token 级 ignore index 遮蔽 |

## 拍死的错误路线

| 路线 | 问题 | 替代方案 |
|------|------|----------|
| 用旧模型输出做 Rejected | 时序悖论——旧数据分布已废弃 | 启发式物理陷阱 (Design A') |
| Sum-Rate − λ×Iterations | λ 无法标定 | 微扰回弹测试 (Snap-back) |
| 全维度 DPO | δ_a/δ_p 无区分度，稀释信号 | Masked DPO (token-level) |
| 硬阈值 >5% gap | 困难环境可能全军覆没 | Top-K 排序截断 |
| 5000 环境直接生成 | Yield rate 25-35% → 仅剩 1500 | 生成 20,000 → 精选 5,000 |

## Consequences

### 正面
- **微扰回弹测试消除运气因子**：统一初始误差后，迭代步数 = 盆地曲率的纯粹代理
- **Masked DPO 集中偏好信号**：只在 δ_q 上产生梯度拉扯，不被 δ_a/δ_p 稀释
- **Top-K 精选保证数据纯度**：每一条 DPO 对都是物理区分度最强的"黄金数据"
- **ε 标定消除超参数不确定性**：数据生成前就确定了最优扰动尺度

### 负面
- **计算量翻 4 倍**：20,000 环境 × 13 SCA-FP vs 旧方案 5,000 × 10
- **生成时间增加**：预估 2-3h（vs 旧方案 ~3.5h，因环境数 ×4 但 worker 效率提升）
- **DPO 实现变复杂**：Dataset 需 token-span 正则匹配 + label masking

### 风险
- **微扰回弹不一定选出全局最佳 Warmstart**：盆地宽度代理对 ε 尺度敏感
- **Masked DPO 可能遗漏 δ_a/δ_p 中隐含的偏好信号**：如果新分布下关联/功率确实有差异
- **Yield rate 可能低于 25%**：如果 ground-clutter 的效果不足以创造足够的局部最优多样性

## 多 UAV 空间排斥力 (补充决策)

独立于数据重生路线，但解决同一类问题：目标函数缺乏空间 repellent 导致 UAV 扎堆。

在 SCA-FP 目标函数中添加反比惩罚项：
```
Penalty_repel = λ_repel × Σ_i Σ_{j>i} 1 / max(||q_i - q_j||², ε_min)
```

- 作用：给每架 UAV 装上"同极磁铁"，物理上杜绝重合
- λ_repel 建议从 0.01 起步，gradually increase 直到 UAV 间最小距离 > 安全阈值
- 这个修改与 ground_clutter_db 正交——两者互补

## Backup Plans (若 DPO 也不达标)

| 优先级 | 方案 | 改动量 | 原理 |
|--------|------|--------|------|
| **P0** | Chain-of-Thought 注入 | 改 prompt 模板 | 让 LLM 在输出坐标前先做几何推理 |
| **P1** | Regression Head (MLP) | ~200 行 | 连续坐标用 MSE 回归，完全绕过 CE |
| **P2** | Online RL with PPO | 大改动 | 直接优化 SCA-FP 加速比 |

## 决策时间线

```
2026-06-29  09:00  EDA 确诊数据退化 → solver 修复 (ground_clutter_db)
2026-06-29  10:00  5 轮 Grilling 火烤 → 消除所有语义模糊
2026-06-29  11:00  ADR 006 终稿 → 决策锁死
2026-06-29  14:00  代码落地 (commit 7cedb02) — 6 文件, +752/-84 行
           ├── sca_fp.py (max_iters + lambda_repel)
           ├── oracle_generator.py (snap-back + Rejected 混合 + clip 投影)
           ├── dataset.py (Masked DPO token-span -100)
           ├── generate_data.py (ground_clutter_db + snapback CLI)
           ├── calibrate_epsilon.py (NEW 294 行)
           └── quick_validate_fix.py (修复 solve 签名)
2026-06-29  14:30  自检审查 → 3 个 bug 现场斩杀
           ↓
NOW        ⏳ 服务器: ε-Calibration (50 envs, 5 min)
           ↓
           服务器: 全量数据生成 20,000 envs (~2-3h)
           ↓
           服务器: 质量闸门 → Top-5000 精选
           ↓
           服务器: EDA 验收 (红线检查)
           ↓
           服务器: Masked DPO 训练 (~5-10h)
           ↓
           服务器: 评估 → 若不达标 → P0→P1→P2 逐级出牌
```
