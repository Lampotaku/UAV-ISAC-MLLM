---
type: training_log
status: decision_point
last_updated: 2026-07-01
related: [status.md, gemini, oom_incidents.md, data_degeneracy.md]
---

# DPO 困境：Masked vs Unmasked 的终极诊断

**日期**: 2026-07-01 | **阶段**: 🟡 Stage II DPO — 烟雾测试完成，待决策

## 一、当前进展

### 已完成

| 里程碑 | 状态 | 关键指标 |
|--------|------|----------|
| 20K 数据重生 | ✅ | 19,925 SFT + 19,925 DPO, 99.6% yield |
| Top-5000 质量闸门 | ✅ | cutoff utility gap 43.9, median 66.9 |
| Stage I SFT 训练 | ✅ | step 150 最佳 checkpoint |
| Unsloth → 纯 PyTorch | ✅ | SDPA + bf16 全精度 |
| OOM 1-6 全部修复 | ✅ | bs=2 终极配置稳定运行 |
| 5 Bug 狩猎闭合 | ✅ | 负数阈值 / baseline 误杀 / DPO 退化 / snapback 浪费 |
| DPO 烟雾测试 × 3 | ✅ | 详见下文 |

### 数据就绪

```
/root/autodl-tmp/data/cache/
├── sft_dataset.jsonl      19,925 条
├── dpo_dataset.jsonl      19,925 条 (全量)
├── dpo_top5000.jsonl       5,000 条 (质量精选)
└── dpo_smoke.jsonl           50 条 (烟雾测试)
```

---

## 二、核心发现：量纲冲突（Dimensional Conflict）

### 问题的本质

UAV-ISAC 的 warmstart 输出包含三个物理量纲完全不同的向量：

| 输出 | 物理含义 | 值域 | 量纲 |
|------|----------|------|------|
| δ_q | UAV 位移向量 | 连续 ℝ³ (米) | 连续坐标 |
| δ_a | 用户关联矩阵 | {0,1} 二值 | 离散分配 |
| δ_p | 功率分配 | [0, P_max] 归一化 | 连续比例 |

在 DPO 训练中，这三个量纲的 log-probability 梯度被直接求和。由于量纲不同，它们的梯度大小天然相差数个数量级——这就是**量纲冲突**。

### 实验证据：三次烟雾测试

#### 测试 1：Masked DPO（初始配置）
- 配置：mask δ_a/δ_p, mu=0.05, beta=0.1, lambda_ctl=0.5
- 结果：梯度稳定，但模型输出常数 — δ_q=[0,0,0], δ_a=0.25, δ_p=0.05
- L2 ratio vs shifted = 0.0000, vs zero = 6.13
- **诊断**：疑似模态坍塌，但 eval 脚本有 prompt 重建 bug

#### 测试 2：Unmasked DPO（拆掉 Mask）
- 配置：全量 DPO, mu=0.01, beta=0.2, lambda_ctl=0.02
- 结果：**梯度爆炸** — Step 10 grad_norm=5347, loss_ctl=74984
- Step 35/37 余震不断，grad_norm 多次破千
- **结论**：Unmasked DPO 在数学上不可行 — δ_q/δ_a/δ_p 的梯度量级差异导致 log-prob 求和失控

#### 测试 3：Masked DPO + 修复 Eval
- 配置：恢复 mask, mu=0.05, beta=0.1, lambda_ctl=0.02
- 梯度：grad_norm 100~200，健康稳定 ✅
- 生成质量：
  - Sample 1: δ_q=[0,0,0]（全零）, δ_a 第一行有变化但均值=0.25, δ_p 几乎全是 0.05
  - Sample 2: δ_q=[0,0,5]（非零！）, δ_a 几乎全是 0, δ_p 几乎全是 0/0.1
- Control Sensitivity（修复后的 eval）：
  - δ_q L2 ratio vs shifted = 0.0000（两个样本）
  - δ_q L2 ratio vs zero = 6.3（两个样本）
  - δ_a L2 ratio vs shifted = 0.0000, vs zero = 0.0000
  - δ_p L2 ratio vs shifted = 0.0000, vs zero = 0.0000
- SCA-FP 加速比：**平均 1.30x**（5 样本，范围 1.0-1.5x）

---

## 三、当下困境：Masked DPO 的结构性矛盾

### 困境描述

Masked DPO 是通过将 δ_a 和 δ_p 的 token label 设为 -100，使 DPO loss 的梯度只流向 δ_q。这是一个"两害相权取其轻"的选择：

```
         Unmasked DPO                  Masked DPO
         ────────────                  ──────────
    ✅ 三个量纲都学偏好           ✅ 梯度稳定，不会爆炸
    ❌ 梯度爆炸，数学不可行        ❌ δ_a/δ_p 退化为常数
```

### 为什么 δ_a/δ_p 会退化为常数？

这是架构层面的结构性矛盾，不是超参调优能解决的：

1. **DPO 梯度只流向 δ_q**：mask 机制使 δ_a/δ_p token 的 label 为 -100，log-prob 对这些 token 的梯度为零
2. **Control token 是共享的**：8 个 `<ctrl_i>` embedding 同时服务三个输出。δ_q 独占了 DPO 的偏好信号，挤占了 control token 的表征空间
3. **δ_a/δ_p 只能靠 λ_ctl 的 MSE**：projection head 在没有 DPO 梯度引导的情况下，面对 δ_q 主导的 control token 表征，学到的最安全策略就是输出常数（初始化默认值）
4. **MSE 信号的困境**：MSE 确实给了 δ_a/δ_p 监督信号，但这个信号远弱于 DPO 梯度对 control token 的塑造力。control token 被 δ_q 的偏好拉扯变形后，projection head 看到的输入已经不再包含 δ_a/δ_p 所需的环境信息

### 火焰图

```
                     Control Tokens (8 个共享 embedding)
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         δ_q 表征          δ_a 表征        δ_p 表征
         (DPO 梯度)       (仅 MSE)        (仅 MSE)
              │               │               │
         ~~~~強い~~~~      ~~~~弱い~~~~    ~~~~弱い~~~~
              │               │               │
         [0,0,5]           0.25 常数       0.05 常数
```

---

## 四、对测试 3 结果的诚实解读

### gemini 的乐观解读 vs 实际情况

| 论点 | gemini 的判断 | 实际情况 |
|------|---------------|----------|
| δ_a/δ_p 不再坍塌 | "169 个丰富多样的浮点数" | δ_a mean 恒为 0.25, δ_p mean 恒为 0.05 — **就是常数** |
| L2 ratio=0 是鲁棒性 | "拓扑鲁棒性，宏观战略不应变" | **换了一个完全不同环境**（Sample 1 vs 2），δ_a 结构仍然高度雷同 |
| 1.30x 加速比 | "黎明已至，直接全量总攻" | Warmstart 2.0 vs Cold-start 2.6 iter，差距仅 0.6 次迭代，5 样本统计意义弱 |

### 真实状态

- 🟢 **梯度稳定**：Masked DPO 确实封印了量纲冲突，grad_norm 健康
- 🟢 **δ_q 有微弱学习信号**：归零测试 L2=6.3 证明模型不是完全瞎猜；Sample 2 输出 [0,0,5] 也说明不同环境有不同输出
- 🟡 **δ_a/δ_p 仍是常数**：没有证据表明它们学到了环境相关的策略
- 🟡 **1.30x 加速比**：50 步婴儿模型的正信号，但太弱太小，不足以判断全量后的效果
- 🔴 **δ_q 对微小扰动不敏感**：shifted L2=0 说明模型学到的 δ_q 变化非常粗粒度

---

## 五、当前的抉择

### 选项 A：直接全量 DPO（Masked）🔥 推荐

**理由**：50 步太少了。梯度稳定 + δ_q 有信号 = 值得赌一把全量。即使 δ_a/δ_p 保持常数，如果 δ_q 的质量能从 5000 条偏好数据中大幅提升，SCA-FP 加速比可能突破 1.5x。

**风险**：δ_a/δ_p 常数可能成为硬天花板，加速比卡在 1.3-1.5x。

**行动**：
```bash
# 恢复生产配置
# dpo_file: dpo_dataset.jsonl (或 dpo_top5000.jsonl)
# grad_accum: 16
# save_steps: 200
# logging_steps: 10
tmux new -s dpo_train
python src/training/train_dpo.py --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/checkpoints/stage1_step_150 \
    --data_dir /root/autodl-tmp/data/cache
```

### 选项 B：拆分 Control Token

**理由**：当前 8 个 control token 被 δ_q 独占。拆成 3 组独立 token（δ_q: 4, δ_a: 2, δ_p: 2），让每个输出有自己的专属表征通道。

**代价**：需要修改 projection head 和数据集代码，约 2-3 小时开发 + 重新跑 SFT Phase 1 和 2。

**风险**：架构改动引入了新的不确定性，且可能不解决根本问题（δ_a/δ_p 的监督信号弱不是因为表征通道不够，而是因为没有偏好梯度）。

### 选项 C：CoT 注入（P0 后备方案）

**理由**：让模型在生成数值前先做语义推理（"UAV 1 距离用户群 A 最近，应服务用户 3/5/7..."），利用 LLM 的推理能力显式化 δ_a/δ_p 的决策过程。

**代价**：修改 prompt 模板 + 重新生成数据（或改写现有数据）。

**适用场景**：选项 A 失败后。

---

## 六、建议

**先跑 A，同时准备 B 的代码草图。**

理由：A 的成本最低（10 小时 GPU 时间），失败信息量最大。如果全量 Masked DPO 跑完，SCA-FP 加速比 ≥ 1.5x——问题解决，直接写论文。如果加速比卡在 1.3x 以下——我们知道 δ_a/δ_p 常数是瓶颈，B 或 C 就是下一步。

不要在烟雾测试阶段做架构改动。50 步的数据不足以支撑任何架构决策。

---

## 七、关键经验

1. **量纲冲突是真实的**：不同物理量纲的 log-prob 不能直接求和。Unmasked DPO 的梯度爆炸不是偶然，是数学必然。
2. **Masked DPO 是必要但不充分的**：它解决了梯度爆炸，但没有解决 δ_a/δ_p 的学习问题。
3. **Eval 的 prompt 重建是隐形杀手**：LLM 自回归生成完全依赖 prompt 文本。不重建 prompt 就改 q_current，等于没改。
4. **不要用 50 步的数据做最终判断**：烟雾测试只能排除爆炸/坍塌，不能评估最终性能。
5. **SCA-FP 加速比是唯一不会骗人的数字**：loss、accuracy、L2 ratio 都是代理指标。只有加速比是真金。
