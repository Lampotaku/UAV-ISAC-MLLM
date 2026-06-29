---
type: status
status: current
stage: data_regeneration
last_updated: 2026-06-29
related: [data_degeneracy, oom_incidents, adr_006_data_regeneration, CONTEXT.md]
---

# 项目当前状态

**最后更新**: 2026-06-29 | **阶段**: ⚠️ 数据层大修 — SFT 诊断出根因 → 求解器修复 → 等待数据重生

## 🔴 核心发现：数据退化 — 项目最致命的阿喀琉斯之踵

SFT 训练在 Phase 2 进行到 step 200+ 后暴露出**完全模态坍塌（Modality Collapse）**：模型对所有输入输出相同的 `[0, 0, -5]` 预测，SCA-FP 加速比 1.0×（零加速）。经过 4 层诊断（eval → 消融 → EDA → 求解器源码分析），根因被锁定：

**不是模型架构问题，不是训练方法问题，是数据本身的分布退化。**

### EDA 三组致命数字

| 指标 | 旧值 | 问题 |
|------|------|------|
| 位移方向 | **97.4% 向下，0% 向上** | 求解器目标函数对高度单调递减 |
| 位移速度 | **84.7% = 15m/s（v_max）** | "低飞=好信号"的单调激励 |
| 功率分配 | **84.3% = 1.0W（P_max）** | 满功率无代价 |

**物理直觉**：旧 SCA-FP 求解器的 UAV 像在真空中飞行——没有地面杂波、没有障碍物、低飞只有好处没有代价。"往下砸地板、油门踩到底"是数学上的全局最优，所有 5000 个环境都收敛到同一个极端解。CE Loss 在这个数据集上学到的唯一策略就是：**不管输入是什么，输出最大速度砸地板**。

### 根因与修复

**根因**：`SCAFPConfig` 缺乏地面杂波（Ground Clutter）建模。真实物理中，低空飞行时地面建筑物/树木产生额外多径衰减，随高度升高指数衰减。没有这个 trade-off，"低飞+满速+满功率"是唯一的最优策略。

**修复**（commit `f34129c`）：在 `SCAFPConfig` 新增 `ground_clutter_db: 12.0`：

```
clutter_db = 12.0 × (1 - h_norm)    # H_min → +12dB, H_max → 0dB
```

这创造了一个非凸 trade-off：
- **飞太低**：距离近但地面杂波吞噬信号
- **飞太高**：杂波消失但自由空间衰减放大
- **最优解**：根据用户/目标分布动态变化的"甜点"

详见 [data_degeneracy.md](../02_training_log/data_degeneracy.md)。

## 📊 修复历程（2026-06-26 → 2026-06-29）

### Phase 1: 控制表示学习（已完成 ✅）

| Run | 配置 | 峰值 sens | 结论 |
|-----|------|-----------|------|
| Run 1 | 单 query, 8 token, LR 2e-4 | 0.0102 | 控制通路濒死 |
| Run 2 | 单 query, 8 token, LR 5e-4 | 0.0240 | 慢速线性，天花板低 |
| Run 3 | 单 query, 8 token, LR 1e-3 | 0.0061 | 梯度震荡 |
| Run 4 | 单 query, 32 token, LR 5e-4 | 0.0021 | 注意力稀释 |
| **Run 5** | **4 query, 8 token, LR 5e-4** | **0.0901** | ✅ 架构验证成功 |

### Phase 2: SFT 联合训练 → 核心故障发现

| Step | 事件 |
|------|------|
| 50 | Golden Cross 达成（loss_ctl 首次低于纯 MSE baseline） |
| 100 | loss_ctl 持续下降，文本质量正常 |
| 150 | loss_ctl 0.015，JSON 生成正常 → **最佳 checkpoint** |
| 200 | **Grad Diagnostic 崩溃**（retain_graph 冲突）→ 修复 |
| 200+ | 模型开始退化，输出向全局均值收缩 |
| 评估 | **完全模态坍塌** — 所有 checkpoint (50/100/150/200) 输出相同预测 |

### Step 200 崩溃：Grad Diagnostic retain_graph 冲突

**症状**：`RuntimeError: Trying to backward through the graph a second time` 恰好在 step 200 触发。

**根因链**：
1. `grad_diag_interval=200` → 诊断只在 `step % 200 == 0` 时运行
2. 诊断代码：`torch.autograd.grad(_scaled_ctl, lora_params, retain_graph=False)` 释放了 `delta_hat → projection_head → LoRA` 的计算图
3. 紧接的 `accelerator.backward(total_loss)` 需要同一路径 → 崩溃

**修复**（commit `458d4c1`）：将第 576 行 `retain_graph=False` → `retain_graph=True`。

**教训**：Grad diagnostic 的 `torch.autograd.grad()` 与 `loss.backward()` 共享计算图时，所有 `autograd.grad` 调用必须 `retain_graph=True`。

### OOM #6: Phase 2 切换三重内存泄漏

Phase 2 启动时（CE loss 与 CTL loss 联合训练）触发 OOM：
1. `lm_head` 权重重绑导致额外 ~8GB 显存
2. Python GC 未及时回收 Phase 1 遗留的中间张量
3. CE logits 未显式释放

修复（commits `0532186`, `7f8bc54`）：
- `lm_head` 解绑 → 省 ~8GB
- `gc.collect()` + `torch.cuda.empty_cache()` 硬加固
- logits 显式 `del` + `empty_cache()`

详见 [oom_incidents.md](../02_training_log/oom_incidents.md)。

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

**求解器修改**：
```
SCAFPConfig.ground_clutter_db: 0.0 → 12.0  (commit f34129c)
  → 通信路径损耗: pl_db += clutter_db
  → 感知路径损耗: pl_db += clutter_db
  → clutter_db = 12.0 * (1 - h_norm),  h_norm ∈ [0, 1]
```

## ✅ 已完成

| 项 | 状态 |
|----|------|
| 7 轮代码审查 + 一审修复 | ✅ |
| 5000 环境旧版数据生成 | ✅ (已废弃 — 数据退化) |
| OOM #1-5 修复 (省 ~54 GB) | ✅ |
| OOM #6 修复 (Phase 2 切换) | ✅ |
| Plan A：纯 PyTorch CE + SDPA | ✅ |
| 根因 1：梯度密度失衡 → Attention Pooling | ✅ |
| 根因 2：单 Query 出口瓶颈 → Multi-Query | ✅ |
| 根因 3：数据退化 → 地面杂波修复 | ✅ |
| Step 200 Grad Crash 修复 | ✅ |
| Phase 1 checkpoint 保存 (step 150 最佳) | ✅ |
| MSE 收缩效应确诊 | ✅ |
| DPO 旧数据训练已停止 | ✅ |
| 5 轮领域模型火烤 (Grilling) | ✅ — 10 个术语精确定义, 3 条错误路线拍死, CONTEXT.md 建立 |

## ⏭️ 下一步行动 (2026-06-29 Grilling 终稿)

### 步骤 0：ε 标定 — Pilot Sweep (5 min)

在全量数据前，先跑微扰步长标定。ε 决定了微扰回弹测试测的是"盆地内壁"还是"跨山脊"。

```bash
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm
python scripts/calibrate_epsilon.py
```

对 50 个环境测试 ε ∈ {0.5, 1.0, 2.0, 4.0, 8.0}m，选回弹步数区分度最大的值。预期最佳：1.0-2.5m。

### 步骤 1：快速验证求解器修复 (2 min)

```bash
python scripts/quick_validate_fix.py
```

验收标准：满速 < 40%，微调 > 10%，**上升 > 15%（红线）**。

### 步骤 2：全量数据暴力生成 (~2-3h, 70 workers)

**注意：生成 20,000 环境（非 5,000），后续 Top-K 精选 5,000。**

每个环境执行：10 次 Random Restart → Pareto 过滤 → Top-3 候选 → 微扰回弹测试 → Chosen/Rejected 构造。所有 Rejected 经 `clip_to_physics_bounds` 约束投影。

```bash
python scripts/generate_data.py \
    --num-envs 20000 --num-restarts 10 \
    --output-dir /root/autodl-tmp/data/full20000_v2 \
    --num-workers 70
```

### 步骤 3：质量闸门 → Top-5000 精选

按 Chosen-Rejected Composite Score Gap 排序，取前 5,000 名。

```bash
python scripts/eda_data.py --data-dir /root/autodl-tmp/data/full20000_v2
```

EDA Section 3 三条红线全部通过后才进入训练。

### 步骤 4：Masked DPO 训练 (~5-10h)

在 `dataset.py` 中实施 token-span-level masking：JSON 中 δ_a 和 δ_p 对应 token 的 label 设为 `-100`，DPO 自动跳过。梯度只集中在 δ_q 的偏好拉扯上。

```bash
python src/training/train_dpo.py \
    --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/checkpoints/stage1_step_150 \
    --data_dir /root/autodl-tmp/data/full20000_v2
```

预期 VRAM: ~65-75 GB / 96 GB (bs=1, 双模型)。

### 步骤 5：评估

```bash
python scripts/eval_generation.py \
    --config configs/default.yaml \
    --checkpoint <dpo_checkpoint> \
    --n_samples 3 --n_scafp 100
```

## 🃏 DPO 若也失败：三级后备方案

| 优先级 | 方案 | 改动量 | 原理 |
|--------|------|--------|------|
| **P0** | CoT 注入 — 让模型先推理再输出 | 改 prompt 模板 | LLM 在生成数值前先做语义分析 |
| **P1** | Regression Head — 用 MLP (MSE) 替代控制 token | ~200 行 | 连续坐标天然适合回归，跳过 CE 离散化 |
| **P2** | Online RL with PPO — 用 SCA-FP 函数值做 reward | 大改动 | 直接优化加速比，绕开 Teacher Forcing 困境 |

详见 [adr_006_data_regeneration.md](../06_decisions/adr_006_data_regeneration.md)。

## 🔑 关键经验

1. **数据分布才是真正的天花板**：架构再好，算法再妙，训练数据如果是退化的，模型必然会坍塌。先看数据，再看模型。
2. **CE Loss 对连续物理量是毒药**：CE 没有距离度量——预测 5.1 和 5.2 跟预测"猫"和"狗"一样错误。这是 SFT 在物理回归任务上的硬天花板。
3. **Teacher Forcing 不会教 error recovery**：SFT 训练时每步都有 ground truth，模型从不学习从自身错误中恢复，一旦自回归生成就滚雪球式偏离。
4. **MSE 代理指标陷阱**：loss_ctl 和 sens 在训练后期背离——loss 越低不代表表征越好。只有 SCA-FP 加速比是真实判据。
5. **单 Attention Query 是回归读出的隐形杀手**：softmax 互斥性使一个 query 无法同时关注多个独立目标。Multi-Query 是正确的范式。
6. **增加 token 数不解决出口瓶颈**：在 softmax 下，更多 token 只会稀释梯度，不会增加信息吞吐量。
7. **Phase 1 存在早停点**：控制表征在中间步骤达峰后会在 MSE 压力下向均值收缩。
8. **Grad diagnostic 的 autograd.grad() 与 backward() 共享图时必须全部 retain_graph=True**。
9. **Unsloth 不存在"局部借用"**：即使是 `import` 在函数体内、仅用于独立 kernel，Unsloth 仍然全局 monkey-patch。与 Gemma 3 + SDPA + grad checkpoint 不可共存。
10. **奥卡姆剃刀优先**：在能解决问题的前提下不增加复杂度。先修数据 → 再换训练方法 → 最后改架构。

## 📋 快速命令参考

```bash
# 服务器登录后
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm
export TORCHINDUCTOR_FLEX_ATTENTION=0

# 第 1 步：快速验证求解器修复
python scripts/quick_validate_fix.py

# 第 2 步：全量数据重生
python scripts/generate_data.py \
    --num-envs 5000 --num-restarts 10 \
    --output-dir /root/autodl-tmp/data/full5000_v2 \
    --num-workers 70

# 第 3 步：EDA 验收
python scripts/eda_data.py --data-dir /root/autodl-tmp/data/full5000_v2

# 第 4 步：DPO 重训
python src/training/train_dpo.py \
    --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/checkpoints/stage1_step_150 \
    --data_dir /root/autodl-tmp/data/full5000_v2

# 第 5 步：评估
python scripts/eval_generation.py \
    --config configs/default.yaml \
    --checkpoint <dpo_checkpoint> \
    --n_samples 3 --n_scafp 100
```
