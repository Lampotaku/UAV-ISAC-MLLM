---
type: status
status: current
stage: phase1_evaluation
last_updated: 2026-06-28
related: [phase1_status_2026-06-26, handoff_2026-06-26, talk, oom_incidents]
---

# 项目当前状态

**最后更新**: 2026-06-28 | **阶段**: 代码就绪，等待服务器重跑 → SCA-FP 批量评估 → 决策

> 📋 **接手此项目的工程师请先阅读 [handoff.md](handoff.md)** — 一次性包含所有你需要的上下文。

## 概述

UAV-ISAC-MLLM：用 Gemma 3 12B (LoRA + 约束投影头) 为 SCA-FP 数值优化器提供智能 warmstart。

## 🔴 核心问题 — 控制表示学习失败

**Step-200 初始 checkpoint 评估**：

| 指标 | 值 | 期望 |
|------|-----|------|
| 文本生成 | 模式坍塌，输出乱码 | 有效 JSON |
| Control sensitivity | **0.0000** | > 0.1 |
| SCA-FP 加速比 | **1.0×**（无加速） | ≥ 1.5× |

## 🔬 根因链（逐层确诊）

### 第一层：梯度密度失衡

```
3456 个文本 token → CE loss → 密集梯度
   8 个控制 token → 控制 loss → 稀疏梯度
                              ↓
               梯度密度比 = 3456:8 = 432:1
                              ↓
            CE 完全主导训练，控制信号被淹没
```

**同时**：`ControlReadout.mean(dim=1)` 将 8 个 control token 的 hidden states 平均为一个向量，抹平 token 间空间结构。

### 第二层：单 Query 出口瓶颈（核心致命伤）

注意力池化只有一个可学习 query 向量：
- Softmax 强制注意力总和为 1 → 关注 UAV1 就必须少关注 UAV2/3/4
- 一个 3840-dim 向量无法同时编码 4 架无人机的独立 3D 位置、关联矩阵和功率分配

### 第三层：MSE 收缩效应

MSE loss 天然厌恶风险。训练后期，模型发现"对每个环境输出条件均值"比"做环境特定预测"更安全（平方惩罚小），导致环境区分度（sens）在 Loss 下降的同时反而萎缩。

## 🧪 修复历程

### 修复 1：Attention Pooling + Phase 1 CTL-only Warmup

- Mean pooling → 可学习 query attention pooling
- Phase 1 完全关闭 CE loss，只优化 `L_ctl`
- 切换条件：跨环境 sensitivity > 0.1
- **结果**：sens 从 0.0000 → 0.0134（step 50），控制通路不再是死的

### 失败尝试 1：32 Control Token（信息带宽扩容）

- **假设**：8 token 容量不足
- **结果**：sens 暴跌 0.0134 → 0.0021 → 0.0008
- **诊断**：入口再大，出口仍是单 query。Softmax 将梯度稀释 32×

### 失败尝试 2：LR 1e-3（优化动力学激进）

- **假设**：纯 regression 任务需要更高 LR
- **结果**：sens 从 0.0134 暴跌至 0.0061（step 54）
- **诊断**：梯度震荡/Thrashing —— LR 不是瓶颈

### 修复 2：Multi-Query Attention Pooling ✅ (commit `0546493`)

单 query → **M=4 个独立 query**（每 UAV 一个）：
- 每个 query 独立计算 softmax + 池化
- 每个 UAV 有自己的专属"视角"读 control token
- 共享 readout MLP（权重在所有 UAV 间复用）
- `num_tokens` 回到 8（多 query 不需要多 token）

## 📊 实验数据

### 完整历史对比

| Run | 配置 | Step 50 sens | 峰值 sens | 结论 |
|-----|------|-------------|-----------|------|
| Run 1 | 单 query, 8 token, LR 2e-4 | 0.0000 | 0.0102 @109 | 控制通路濒死 |
| Run 2 | 单 query, 8 token, LR 5e-4 | 0.0134 | 0.0240 @152 | 慢速线性，天花板低 |
| Run 3 | 单 query, 8 token, LR 1e-3 | 0.0061 | — | 梯度震荡，反跌 |
| Run 4 | 单 query, 32 token, LR 5e-4 | 0.0021 | — | 注意力稀释，脑死亡 |
| **Run 5** | **4 query, 8 token, LR 5e-4** | **0.0140** | **0.0901 @150** | ✅ 架构验证成功 |

### Run 5 详细时序

| Step | loss_ctl | sens | 阶段 |
|------|----------|------|------|
| 50 | 35.51 | 0.0140 | 对称性破缺 — 4 query 在"抢地盘" |
| 100 | 17.88 | 0.0084 | 低垂果实 — 学会预测全局均值 |
| 150 | 15.01 | **0.0901** | 🏔️ **峰值** — 环境区分度最高 |
| 200 | 17.47 | 0.0705 | 开始震荡回调 |
| 250 | **10.84** | 0.0391 | MSE 最优但 sens 腰斩 — 均值收缩 |

**sens 定义**：`sens = ||δ_q(env_B) − δ_q(env_A)|| / ||δ_q(env_A)||`（seed 42 vs 43 跨环境）

**sens 下降的数学原因**：
1. **分子萎缩**（主因）：MSE 驱使模型向条件均值收缩 → 环境间绝对差异变小
2. **分母膨胀**（次因）：模型预测的位移尺度变大 → 分母增大

## 🏗️ 架构终态

```
Control Token (8 个, <ctrl_0>..<ctrl_7>)
       ↓ Gemma 3 12B (LoRA, rank=16)
Control Hidden States [B, 8, 3840]
       ↓ Multi-Query Attention Pooling
  query_0 → attn → pooled_0 [B, 3840]  (UAV 0 专属)
  query_1 → attn → pooled_1 [B, 3840]  (UAV 1 专属)
  query_2 → attn → pooled_2 [B, 3840]  (UAV 2 专属)
  query_3 → attn → pooled_3 [B, 3840]  (UAV 3 专属)
       ↓ 共享 Readout MLP (3840→1920→960→44 per UAV)
  out_0..out_3 [B, 44 each] → concat [B, 176]
       ↓ ResidualMLP → Unflatten → Constraint Projections
  δ_q [B,4,3] + δ_a [B,4,20] + δ_p [B,4,21]
```

## ✅ 已完成

| 项 | 状态 |
|----|------|
| 7 轮代码审查 + 一审修复 | ✅ |
| 5000 环境数据生成 (SFT+DPO) | ✅ |
| OOM #1-5 修复 (省 ~54 GB) | ✅ |
| Plan A：纯 PyTorch CE + SDPA | ✅ |
| 根因 1：梯度密度失衡 → Attention Pooling | ✅ |
| 根因 2：单 Query 出口瓶颈 → Multi-Query | ✅ |
| OOM #6 (Phase 2 切换) | ⚠️ 修复已推送 (7f8bc54) — KeyError 权重重绑 + gc 硬加固，**待服务器验证** |
| Phase 1 checkpoint 保存 | ✅ (commit `910d967`) |
| MSE 收缩效应确诊 | ✅ |

## ⏭️ 行动方案

### 总决策树

```
服务器重跑 Phase 1
       ↓
批量 SCA-FP 评估 (step 150/200/250/300/400)
       ↓
 SCA-FP 加速比 ≥ 1.5×?
  ├── YES → 最佳 checkpoint → Phase 2 (不改一行代码)
  └── NO  → 深度消融诊断 → P0→P3 逐级出牌
```

**核心原则**: 当前 Multi-Query 架构已验证 sound（sens 峰值 0.0901）。问题大概率只是 MSE 早停时机——step 150 的 sens 和 loss_ctl 可能已经"够好"，不需要任何架构改动。

### 步骤 1：服务器重跑 Phase 1

```bash
ssh 服务器
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm
python src/training/train_sft.py --config configs/default.yaml --data_dir /root/autodl-tmp/data/full5000
```

新 checkpoint 保存策略：
- 每 200 步：`phase1_step_200`、`phase1_step_400`
- Phase 1 退出时（sens 触发或 max_steps）：`phase1_step_N`
- **额外**：手动监控 step 150 附近，sens 预计在 0.08-0.10 区间达峰

### 步骤 2：SCA-FP 批量评估（三级验收标准）

对候选 checkpoint 跑完整评估，观察三个硬指标：

| 指标 | 测量方法 | 期望 | 权重 |
|------|----------|------|------|
| **SCA-FP 迭代收敛次数** | MLLM warmstart vs 随机初始值的迭代比 | ≥ 1.5× | 🔴 最核心 |
| **目标函数最终值** | MLLM 初始值 vs 传统算法最终 Sum-rate / EE | 更高（避开局部洼地） | 🟡 重要 |
| **零样本插值响应** | 构造训练集外极端场景，看输出是否平滑微调 | 非全局均值 | 🟢 辅助 |

```bash
# 批量评估
for STEP in 150 200 250 300 400; do
    python scripts/eval_generation.py \
        --config configs/default.yaml \
        --checkpoint /root/autodl-tmp/checkpoints/phase1_step_${STEP} \
        --n_samples 3 --n_scafp 100
done
```

### 步骤 3：按结果分叉

**分支 A — 任一 checkpoint 加速比 ≥ 1.5×**：
- 选加速比最高的 checkpoint
- Phase 1 用 `phase1.enabled: false` 跳过
- 直接启动 Phase 2 Joint SFT+CTL
- **不改一行模型代码**

**分支 B — 加速比均 < 1.5×**：
先跑深度消融诊断（见下方），确认问题性质，再按 P0→P3 顺序出牌。

---

## 🔬 深度消融测试（分支 B 触发）

### A. 零样本插值 (Zero-Shot Interpolation)

构造训练集未出现的边界场景（如拉远某两个用户距离），检验模型是背均值还是学了几何映射。

- **健康信号**: 输出随输入坐标平滑微调，delta_q 方向与几何直觉一致
- **坍塌信号**: 对所有场景输出近乎相同的 delta_q（全局均值）

### B. 单 UAV 约束消融 (Single UAV Ablation)

固定 1 架 UAV（不提供其真实最优位移），让模型预测其余 3 架的 δ_q 和 δ_p。

- **健康信号**: 模型根据被固定 UAV 的状态动态调整其余 3 架的策略（功率互补、关联重分配）
- **坍塌信号**: 其余 3 架输出与被固定 UAV 无关的固定模式

---

## 🃏 备用改进路线（按工程 ROI 排序）

若所有 checkpoint 加速比均不达标，按以下顺序逐级干预：

| 优先级 | 方案 | 改动量 | 原理 |
|--------|------|--------|------|
| **P0** | 调低早停阈值 `sensitivity_threshold: 0.08` | 1 行 YAML | 不跟 MSE 硬刚，sens 达峰即切 Phase 2 |
| **P1** | Margin Loss 保底 `L += λ * max(0, 0.08 - sens)` | ~5 行 loss 代码 | 硬约束环境区分度底线，sens 不敢掉 |
| **P2** | 启发式锚点残差 — 提供便宜 `q_init`，只学 `q_true - q_init` | 改数据 pipeline | 残差均值为 0，根治均值收缩 + 分母膨胀 |
| **P3** | Huber Loss 替代 MSE | 1 行 loss | 对大误差用 L1（线性惩罚），减小平方压力 |

**不在短期路线上的方案**：
- MDN（高斯混合密度网络）：176 维输出做混合密度 → 参数量爆炸 + 模式坍塌风险 + SCA-FP 只需一个点
- 反坍塌 CosSim 正则：需要 batch 内凑环境对，不如 P1 的直接 margin 干净

## 📋 快速命令参考

```bash
# 服务器登录后
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm

# 环境变量 (Blackwell 必须)
export TORCHINDUCTOR_FLEX_ATTENTION=0

# SFT 训练 (Phase 1 → Phase 2 自动)
python src/training/train_sft.py --config configs/default.yaml --data_dir /root/autodl-tmp/data/full5000

# 批量 SCA-FP 评估 (Phase 1 完成后)
for STEP in 150 200 250 300 400; do
    python scripts/eval_generation.py \
        --config configs/default.yaml \
        --checkpoint /root/autodl-tmp/checkpoints/phase1_step_${STEP} \
        --n_samples 3 --n_scafp 100
done

# 跳过 Phase 1 直接跑 Phase 2 (用最佳 checkpoint)
python src/training/train_sft.py --config configs/default.yaml --data_dir /root/autodl-tmp/data/full5000
# 或改 config: phase1.enabled: false, 从最佳 checkpoint 恢复

# 过拟合测试
python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000

# 数据验证
python scripts/validate_data.py --data-dir /root/autodl-tmp/data/full5000
```

## 🔑 关键经验

1. **MSE 代理指标陷阱**：loss_ctl 和 sens 在训练后期背离——loss 越低不代表表征越好。只有 SCA-FP 加速比是真实判据。
2. **单 Attention Query 是回归读出的隐形杀手**：softmax 互斥性使一个 query 无法同时关注多个独立目标。Multi-Query 是正确的回归读出范式。
3. **增加 token 数不解决出口瓶颈**：在 softmax 下，更多 token 只会稀释梯度，不会增加信息吞吐量。
4. **Phase 1 存在早停点**：控制表征的环境区分度在中间步骤达峰后，会在 MSE 压力下向均值收缩。
5. **Cross-Environment Sensitivity 比 Perturbation 测试有效**：原 ±10m 扰动测试因 DeploymentProjection 裁剪恒等映射而永远返回 0。
6. **奥卡姆剃刀优先**：在能解决问题的前提下不增加复杂度。P0（调阈值）→ P1（margin loss）→ P2（锚点残差）→ P3（Huber），逐级出牌。
7. **残差预测已在架构中**：模型输出 δ_q/δ_a/δ_p（扰动量），不是绝对坐标。P2 的改进是显式提供启发式锚点。
