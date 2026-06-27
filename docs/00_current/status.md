---
type: status
status: current
stage: phase1_debugging
last_updated: 2026-06-27
related: [phase1_status_2026-06-26, handoff_2026-06-26, talk, oom_incidents]
---

# 项目当前状态

**最后更新**: 2026-06-27 | **阶段**: Phase 1 架构调试完毕，等待重跑 + SCA-FP 早停搜索

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
| OOM #6 (Phase 1 Phase 2 切换双模型) | ⚠️ 未验证 |
| Phase 1 checkpoint 保存 | ✅ (commit `910d967`, 待 push) |
| MSE 收缩效应确诊 | ✅ |

## ⏭️ 下一步

### 1. 立即（网络恢复后）

```bash
git push  # 推送 910d967 (Phase 1 checkpoint 保存)
```

### 2. 服务器重跑 Phase 1

```bash
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm
python src/training/train_sft.py --config configs/default.yaml --data_dir /root/autodl-tmp/data/full5000
```

新代码会在以下时机保存 checkpoint：
- 每 200 步：`phase1_step_200`、`phase1_step_400`
- Phase 1 退出时（sens 触发或 max_steps）：`phase1_step_N`

同时手动监控并记录 step 150 附近状态（sens 预测峰值区间）。

### 3. 批量 SCA-FP 早停搜索

对候选 checkpoint（预估 150/200/250/300/400）分别跑评估：

```bash
for STEP in 200 250 300 400; do
    python src/eval/eval_generation.py \
        --config configs/default.yaml \
        --checkpoint /root/autodl-tmp/checkpoints/phase1_step_${STEP} \
        --output /root/autodl-tmp/eval/step_${STEP}/
done
```

**核心判据**：SCA-FP 加速比（不是 sens，不是 loss_ctl）。

### 4. 最佳 Checkpoint → Phase 2

- 选加速比最高的 Phase 1 checkpoint
- 手动加载该 checkpoint 启动 Phase 2 (Joint SFT+CTL)
- 或直接设 `phase1.enabled: false` 跳过 Phase 1

### 5. 长期

- 如果最优在 step ~150 → Phase 1 存在特征早停点，需实现自动早停逻辑
- 如果最优在 step 400 → sens 下降不影响下游，当前 max_steps 合理

## 📋 快速命令参考

```bash
# 服务器登录后
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm

# 环境变量 (Blackwell 必须)
export TORCHINDUCTOR_FLEX_ATTENTION=0

# SFT 训练 (Phase 1 → Phase 2 自动)
python src/training/train_sft.py --config configs/default.yaml --data_dir /root/autodl-tmp/data/full5000

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
