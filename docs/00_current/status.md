---
type: status
status: current
stage: dpo_ready
last_updated: 2026-07-01
related: [data_degeneracy, oom_incidents, adr_006_data_regeneration, CONTEXT.md, implementation_2026-06-29, session_2026-07-01]
---

# 项目当前状态

**最后更新**: 2026-07-01 | **阶段**: 🟢 20K 数据生成完成 + Top-5000 精选完毕 → DPO 训练即将点火

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
| **Grilling 终稿代码落地 (commit `7cedb02`)** | ✅ — 6 文件, +752/-84 行 |

## 🏗️ 代码落地详情 (2026-06-29, commit `7cedb02`)

### 改造文件

| 文件 | 变更 | 关键内容 |
|------|------|----------|
| `src/solver/sca_fp.py` | +27 | `max_iters=100` 安全帽, `lambda_repel=0.01` 空间互斥力, `epsilon_min_repel` |
| `src/data/oracle_generator.py` | +374/-84 | 核心重写: snap-back 测试, Pareto 过滤, Rejected 混合构造 (70%次优+30%陷阱), `clip_to_physics_bounds` 约束投影, baseline 缓存优化 |
| `src/data/dataset.py` | +92 | **Masked DPO**: `_find_field_spans_in_json` + token-span 级 label `-100` 遮蔽 δ_a/δ_p |
| `scripts/generate_data.py` | +20 | `ground_clutter_db=12.0`, `lambda_repel=0.01`, `--snapback-epsilon` 等新 CLI 参数 |
| `scripts/calibrate_epsilon.py` | **新文件** 294 行 | ε ∈ {0.5,1,2,4,8}m sweep → 方差分析 → 推荐最佳 ε |
| `scripts/quick_validate_fix.py` | +29 | 修复 `solve()` 调用签名 + 改用 `SCAFPConfig(lambda_repel=0.01)` |

### 审查中斩杀 3 个 Bug

| # | Bug | 位置 | 修复 |
|---|-----|------|------|
| 12a | `converged` 仍引用旧 `max_outer_iters` (应引用新 `max_iters`) | sca_fp.py:187 | → 局部变量 `max_iters` |
| 12b | `_compute_utility_of_delta_q` 每次多跑一次 SCA-FP (×20,000 额外求解) | oracle_generator.py | → 用已有信息估算 gap, 方法删除 |
| 12c | `calibrate_epsilon._pareto_filter` baseline 用随机重启而非 `[0,0,0]` | calibrate_epsilon.py | → zero warm_start |

详见 [implementation_2026-06-29.md](../02_training_log/implementation_2026-06-29.md)

## 📦 2026-07-01 Session — 数据重生执行全纪录

### 执行摘要

| 阶段 | 状态 | 耗时 | 关键指标 |
|------|------|------|----------|
| ε 标定 | ✅ (诊断完成) | ~5 min | 发现 15m 墙 — 所有 ε variance=0 |
| Bug 狩猎 | ✅ 5 个修复 | — | 负数阈值 / 变量缺失 / baseline 误杀 / DPO 退化 / snapback 浪费 |
| 算力优化 | ✅ 14→3 SCA-FP/env | — | 节省 79% 算力，质量零损失 |
| 200-env Dry Run | ✅ | ~2 min | 100% yield, 0 issues |
| 20K 全量生成 | ✅ | 3.98h | 19,925 SFT + 19,925 DPO (99.6% yield) |
| Top-5000 质量闸门 | ✅ | ~5s | cutoff gap 43.9, median 66.9 |
| **DPO 训练** | **🟢 待点火** | ~5-10h | — |

详见 [session_2026-07-01.md](../02_training_log/session_2026-07-01.md)。

### 关键发现：15m 速度墙

在 1000×1000m 区域、v_max=15m/s 约束下，SCA-FP 的最优解在所有 restart 中收敛到同一个硬约束边界点：
- **Snap-back 测试失效**: 所有 ε 的方差 = 0 → 从 `_process_one_environment` 移除
- **Baseline 检查误杀**: [0,0,0] 与 best-of-N 收敛到同一点 → 80% envs 被丢弃 → 从 `_pareto_filter` 移除
- **DPO 对退化**: worst ≈ best (allclose atol=0.5) → 启发式物理陷阱替代
- **Restart 冗余**: 10→3 次，70% 算力节省，数据质量不变

### 5 个 Bug 修复 (commits `f9c968b` → `4fc398d`)

| # | Bug | 症状 | 修复 |
|---|-----|------|------|
| 13 | 负数 utility Pareto 灾难 | 39/50 envs 被误杀 | `max - abs(max) × (1-ratio)` |
| 14 | `delta_q_perturbed` 未定义 | NameError (已预防) | 补全赋值 |
| 15 | Baseline 检查误杀 | 80% envs 丢弃 | 从 `_pareto_filter` 移除 |
| 16 | DPO Chosen≈Rejected | 偏好信号为零 | 退化检测 → 启发式陷阱 |
| 17 | Snapback 无区分度 | 0 variance, 浪费 3 calls/env | 从主流程移除 |

### 当前数据

```
/root/autodl-tmp/data/cache/
├── sft_dataset.jsonl      19,925 条 (SFT)
├── dpo_dataset.jsonl      19,925 条 (全量 DPO — 建议备份)
├── dpo_top5000.jsonl       5,000 条 (质量闸门精选)
├── dpo_top5000.jsonl.report  统计摘要
└── checkpoint.txt          进度记录
```

## ⏭️ 下一步行动 (2026-07-01)

### 步骤 1：替换 DPO 训练文件

```bash
cd /root/UAV-ISAC-MLLM && git pull
cd /root/autodl-tmp/data/cache
# 方案 A: 直接 mv (简单粗暴)
cp dpo_dataset.jsonl dpo_dataset_full.jsonl   # 留底
mv dpo_top5000.jsonl dpo_dataset.jsonl
# 方案 B: 改 config 一行 (推荐)
sed -i 's|dpo_file: "dpo_dataset.jsonl"|dpo_file: "dpo_top5000.jsonl"|' /root/UAV-ISAC-MLLM/configs/default.yaml
```

### 步骤 2：DPO 训练点火 (~5-10h)

```bash
tmux new -s dpo_train
python src/training/train_dpo.py \
    --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/checkpoints/stage1_step_150 \
    --data_dir /root/autodl-tmp/data/cache
```

**监控要点**:
| 指标 | 健康 | 危险 |
|------|------|------|
| `loss_ctl / loss_dpo` | 2~3× | >10× → `lambda_ctl` 砍半到 0.25 |
| `loss_dpo` | 缓慢下降 | 迅速→0 (CTL 在绞杀 DPO) |
| VRAM | ~65-75 GB | >90 GB |

### 步骤 3：评估 (~30 min)

```bash
python scripts/eval_generation.py \
    --config configs/default.yaml \
    --checkpoint /root/autodl-tmp/checkpoints/stage2_dpo_step_XXX \
    --n_samples 3 --n_scafp 100
```

**验收标准**: SCA-FP 加速比 > 1.5×（论文核心贡献立住）。

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

## 📋 快速命令参考 (2026-07-01 更新)

```bash
# 服务器登录后
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm
export TORCHINDUCTOR_FLEX_ATTENTION=0

# 第 0 步：质量闸门 (已完成 ✅)
# python scripts/select_top5000.py \
#     --input /root/autodl-tmp/data/cache/dpo_dataset.jsonl \
#     --output /root/autodl-tmp/data/cache/dpo_top5000.jsonl --top 5000

# 第 1 步：替换 DPO 文件
sed -i 's|dpo_file: "dpo_dataset.jsonl"|dpo_file: "dpo_top5000.jsonl"|' configs/default.yaml

# 第 2 步：DPO 训练 (5-10h)
tmux new -s dpo_train
python src/training/train_dpo.py \
    --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/checkpoints/stage1_step_150 \
    --data_dir /root/autodl-tmp/data/cache

# 第 3 步：评估 — 拿到加速比
python scripts/eval_generation.py \
    --config configs/default.yaml \
    --checkpoint <dpo_checkpoint_path> \
    --n_samples 3 --n_scafp 100
```
