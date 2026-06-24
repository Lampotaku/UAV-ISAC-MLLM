# 交接文档 #4 — 数据生成后：验证结果、SFT 准备、下一步

> 状态: 2026-06-25 | 位置: 5000 环境数据生成完成 → 准备 SFT 训练
> 下一步: 在 AutoDL 服务器上运行过拟合测试 → 启动 Stage I SFT

---

## 目录

1. [当前状态总览](#当前状态总览)
2. [最终 5000 数据验证结果](#最终-5000-数据验证结果)
3. [merge: feature/multiprocessing → master](#merge-featuremultiprocessing--master)
4. [SFT 过拟合测试 — 训练正确性的黄金标准](#sft-过拟合测试--训练正确性的黄金标准)
5. [下一步操作指南](#下一步操作指南)
6. [完整管线流程](#完整管线流程)
7. [关键文件索引](#关键文件索引)

---

## 当前状态总览

```
项目进度: ████████████░░░░░░░░ ~60%

✅ 源码开发 (18 files, ~4200 lines)
✅ 7 轮代码审查 (25+ issues closed)
✅ 服务器环境 (AutoDL RTX 5090, CUDA 12.8, Unsloth 4-bit QLoRA)
✅ 烟雾测试 (5 envs) — P0 物理约束修复
✅ P0 双 Bug 修复 (环境多样性 + token 截断)
✅ P0-1 Token 溢出修复 (BPE 碎片化, 3 轮迭代)
✅ 多进程 Code Review (10 bugs fixed)
✅ 5000 环境数据生成成功 (SFT: 5000, DPO: 186,896)
✅ 数据质量验证 (0 issues)
✅ feature/multiprocessing → master 合并
✅ SFT 过拟合测试脚本就绪

⏳ 待执行: 服务器运行过拟合测试 (5 min)
⏳ 待执行: Stage I SFT 训练 (3 epochs, ~3-8h)
⏳ 待执行: Stage II DPO 训练 (2 epochs, ~5-10h)
⏳ 待执行: 评估 (200 test envs, 9 baselines)
```

---

## 最终 5000 数据验证结果

### 数据规模

| 数据集 | 样本数 | 文件 |
|--------|-------|------|
| SFT | 5,000 | `/root/autodl-tmp/data/full5000/sft_dataset.jsonl` |
| DPO | 186,896 | `/root/autodl-tmp/data/full5000/dpo_dataset.jsonl` |

### 物理约束

```
δ_q 3D位移 (‖Δq‖₂): mean=15.0m [14.2, 15.0]  (上限=15m) ✅
```

位移在 15m 边界饱和 — 这是预期行为：最优 UAV 移动通常是"尽可能远地移动"以改善目标函数，约束有效限制了范围。

### DPO 偏好对质量

```
Utility chosen:  mean=924.78  [256.69, 4729.64]
Utility rejected: mean=891.88  [224.82, 4718.47]
Utility Δ:        mean=32.90   [0.09, 2016.23]
```

平均效用差 ~33，正样本始终优于负样本。最小 gap 0.09 (边界情况，仍有效)。

### Token 长度

```
Total tokens (prompt+ctrl+resp): mean=1696  [1649, 1717]  ← budget 4096 ✅
Prompt tokens:                    mean=1344  P99=1361      ← budget 3072 ✅
Response tokens:                  mean=344   [340, 348]    ← budget 1024 ✅
```

所有 token 计数安全落在预算内。无截断。

### EDA Section 2: 物理可视化

3 个随机环境的 3D 场景验证：
- UAV 当前位置 → 建议位置 (位移 |Δ|₂ = 15.0m)
- 每 UAV 功率: communication + sensing = P_max (1W)
- ASCII 俯视图: UAV 起点/终点、用户、目标全部可见
- 高度剖面: 全部在 [50, 300]m 范围内

### 最终判定

```
✅ 数据质量正常 — 可以继续训练
✅ Section 1 PASS — no truncation, no format issues
✅ Section 2 PASS — 3D displacement, power budget, altitude all within constraints
✅ Section 3 PASS — diversity, utility gap distribution
✅ Section 4 PASS — δ_q direction diversity (360° uniform)
```

详见: [docs/04_data_results/final_result.md](docs/04_data_results/final_result.md)

---

## merge: feature/multiprocessing → master

### 合并状态

```
Commit: 6ea0de9
Message: merge: feature/multiprocessing → master
         (P0-P2 fixes, Q1-Q5 cleanup, EDA, overfit test)
```

master 现已包含全部修复和功能：
- P0 物理约束 (Box→Sphere)
- P0 环境多样性 (per-sample RNG)
- P0 Token 截断 (budget 512→1024)
- P0-1 BPE 碎片化 (3 轮精度修复)
- P0 多进程原子写入
- P1 Ctrl+C 信号处理
- Q1-Q5 代码清理
- SFT 过拟合测试脚本

合并时的唯一冲突 (`dataset.py`) 已用 feature 分支版本解决 — `_tokenize_pair()` 共享函数 + `<eos>` 修复。

### 分支状态

```
master                   ← 主要开发分支 (当前)
feature/multiprocessing  ← 代码与 master 一致，可删除
```

---

## SFT 过拟合测试 — 训练正确性的黄金标准

### 原理

一个正确的训练管线在极小数据（5 样本）上一定能过拟合。如果 loss 降不到接近 0，说明代码有 bug。

### 5 项检查

| # | 检查项 | 标准 | 验证的组件 |
|---|--------|------|-----------|
| 1 | `loss_total` 下降 | >50% reduction | 梯度流、前向/反向传播 |
| 2 | `loss_sft` 下降 | <0.5 | label_mask 对齐、token prediction |
| 3 | `loss_ctl` 下降 | <0.01 | 投影头梯度流、δ 目标 |
| 4 | 单调性 | 最后 50 步持续下降 | 优化器、学习率 |
| 5 | 数值稳定性 | 无 NaN/Inf | 梯度裁剪、loss 计算 |

### 在服务器上运行

```bash
cd /root/UAV-ISAC-MLLM
git pull origin master
conda activate uavmllm
python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000
```

**预期**: 3-5 分钟, 全部 5 项检查通过。

### 通过后 → 直接启动 SFT

如果过拟合测试通过，SFT 训练代码被证明是正确的，可以直接启动全量训练。

---

## 下一步操作指南

### Step 1: Git Pull (服务器)

```bash
cd /root/UAV-ISAC-MLLM
git pull origin master
```

获取最新代码（含过拟合测试脚本、所有修复）。

### Step 2: 过拟合测试 (5 min)

```bash
conda activate uavmllm
python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000
```

验证训练代码正确性。预期输出:

```
✓ ALL CHECKS PASSED
  The SFT training pipeline is correctly wired:
    • Tokenization + control token injection
    • Gemma3 forward pass (4-bit QLoRA)
    • Control token hidden state extraction
    • Projection head (readout → MLP → constraints)
    • Combined loss (L_SFT + λ_ctl * L_ctl)
    • Gradient flow through LoRA + projection head
    • Optimizer updates

  → Safe to proceed with full 5000-sample SFT training.
```

### Step 3: Stage I SFT 训练 (~3-8h)

```bash
python src/training/train_sft.py --config configs/default.yaml
```

**配置**:
| 参数 | 值 |
|------|-----|
| 数据集 | 5000 SFT samples |
| Epochs | 3 |
| Learning rate | 2e-4 |
| LoRA rank/alpha | r=16, α=32 |
| Batch size | 1 per device × grad_accum=16 → effective batch=16 |
| Max sequence length | 4096 |
| 量化 | 4-bit QLoRA (Unsloth) |
| 优化器 | AdamW + cosine scheduler + warmup |
| 损失 | L_SFT + λ_ctl·L_ctl |

**输出**: `outputs/stage1_sft_final/` (LoRA weights + projection_head.pt + tokenizer)

### Step 4: Stage II DPO 训练 (~5-10h)

```bash
python src/training/train_dpo.py --config configs/default.yaml
```

使用 Stage I 的 LoRA weights 作为初始化和 reference model (独立加载，不 deepcopy)。

### Step 5: 评估

```bash
python src/eval/evaluate.py --config configs/default.yaml
```

200 测试环境，9 基线对比 (6 指标)。

---

## 完整管线流程

```
┌─ Step 0: 服务器首次设置 ──────────────────────────────────── 已完成 ✅
│   git clone → autodl_setup.sh → huggingface-cli login          │
├─ Step 1: 数据生成 ────────────────────────────────────────── 已完成 ✅
│   generate_data.py --num-env 5000 --num-restarts 10            │
│   产出: SFT 5000 + DPO 186,896                                 │
├─ Step 2: 数据验证 ────────────────────────────────────────── 已完成 ✅
│   validate_data.py → 0 issues                                  │
│   eda_data.py → 4 sections all PASS                            │
├─ Step 3: 过拟合测试 ──────────────────────────────────────── ⏳ 待执行
│   test_sft_overfit.py → 5 checks, ~5 min                       │
├─ Step 4: Stage I SFT ─────────────────────────────────────── ⏳ 待执行
│   train_sft.py → 3 epochs, ~3-8h                               │
│   产出: LoRA weights + projection_head.pt                      │
├─ Step 5: Stage II DPO ────────────────────────────────────── ⏳ 待执行
│   train_dpo.py → 2 epochs, ~5-10h                              │
│   产出: 最终模型                                               │
└─ Step 6: 评估 ────────────────────────────────────────────── ⏳ 待执行
    evaluate.py → 200 envs × 9 baselines × 6 metrics             │
```

---

## 关键文件索引

### 数据文件 (服务器)

| 文件 | 路径 |
|------|------|
| SFT 数据集 | `/root/autodl-tmp/data/full5000/sft_dataset.jsonl` |
| DPO 数据集 | `/root/autodl-tmp/data/full5000/dpo_dataset.jsonl` |
| 工作目录 | `/root/autodl-tmp/data/full5000/working/` |

### 代码文件

| 文件 | 用途 |
|------|------|
| [train_sft.py](src/training/train_sft.py) | Stage I SFT 训练主循环 |
| [train_dpo.py](src/training/train_dpo.py) | Stage II DPO 训练主循环 |
| [gemma_isac.py](src/model/gemma_isac.py) | Gemma3ISAC 模型 (Unsloth 4-bit + LoRA + 投影头) |
| [projection_head.py](src/model/projection_head.py) | ConstraintProjectionHead (Proj_Q/A/P) |
| [losses.py](src/model/losses.py) | UAVISACLosses (SFT + DPO + 约束) |
| [dataset.py](src/data/dataset.py) | SFTDataset + DPODataset (共享 _tokenize_pair) |
| [test_sft_overfit.py](scripts/test_sft_overfit.py) | SFT 过拟合测试 |

### 配置文件

| 文件 | 用途 |
|------|------|
| [default.yaml](configs/default.yaml) | 全局超参数 (硬件/模型/训练/仿真/数据) |

### 交接文档

| # | 文档 | 内容 |
|---|------|------|
| 1 | [16_handoff_01_project_direction](docs/05_handoff/16_handoff_01_project_direction.md) | 论文总体方向 |
| 2 | [17_handoff_02_pre_datagen](docs/05_handoff/17_handoff_02_pre_datagen.md) | 数据生成前的准备 |
| 3 | [18_handoff_03_datagen_problems](docs/05_handoff/18_handoff_03_datagen_problems.md) | 数据生成中的问题与修复 |
| 4 | [19_handoff_04_post_datagen](docs/05_handoff/19_handoff_04_post_datagen.md) | 本文档 — 当前状态与下一步 |

### 事后分析文档

| # | 文档 | 内容 |
|---|------|------|
| 10 | [10_physical_constraint_bug_postmortem](docs/03_bug_postmortems/10_physical_constraint_bug_postmortem.md) | P0 物理约束穿透 |
| 11 | [11_pre_training_data_eda_postmortem](docs/03_bug_postmortems/11_pre_training_data_eda_postmortem.md) | P0 双 Bug (多样性+截断) |
| 12 | [12_remaining_verification_gaps](docs/03_bug_postmortems/12_remaining_verification_gaps.md) | 20 个未验证维度审计 |
| 13 | [13_response_token_bug_postmortem](docs/03_bug_postmortems/13_response_token_bug_postmortem.md) | P0-1 BPE 浮点碎片化 |
| 14 | [14_first_review_post_datagen](docs/02_code_reviews/14_first_review_post_datagen.md) | 多进程 Code Review 发现 |
| 15 | [15_first_review_fix_report](docs/02_code_reviews/15_first_review_fix_report.md) | Code Review 修复报告 |

### 结果文件

| 文件 | 运行 | 内容 |
|------|------|------|
| [result.md](docs/04_data_results/result.md) | Run 1: Smoke 5 | P0 物理约束发现 |
| [result2.md](docs/04_data_results/result2.md) | Run 2: Smoke 70 | P0 双 Bug 发现 |
| [result3.md](docs/04_data_results/result3.md) | Run 3: Smoke 20 | P0-1 Token 溢出发现 |
| [final_result.md](docs/04_data_results/final_result.md) | **Run 4: Full 5000** | **✅ 0 issues, all clean** |
