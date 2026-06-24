# 交接文档 #2 — 数据生成前：架构搭建与代码审查

> 时间段: 2026-06-22 → 2026-06-23
> 本阶段目标: 完成全部源码、7 轮审查闭合、服务器环境就绪、烟雾测试通过

---

## 目录

1. [阶段概述](#阶段概述)
2. [源码架构](#源码架构)
3. [七轮代码审查](#七轮代码审查)
4. [服务器环境搭建](#服务器环境搭建)
5. [烟雾测试 (5 envs)](#烟雾测试-5-envs)
6. [烟雾测试中发现的 P0 Bug](#烟雾测试中发现的-p0-bug)
7. [本阶段结束状态](#本阶段结束状态)

---

## 阶段概述

在开始 5000 环境的大规模数据生成之前，需要完成：

1. **全部源码实现** — 18 个 Python 文件, ~4200 行
2. **多轮代码审查** — 7 轮, 25+ 问题闭合
3. **AutoDL 服务器环境** — RTX 5090, CUDA 12.8, Unsloth 4-bit QLoRA
4. **烟雾测试** — 5 环境小规模验证管线

---

## 源码架构

```
src/
├── env/               ← 仿真环境层
│   ├── uav_network.py      UAV/用户/目标拓扑
│   ├── uav_channel.py      物理信道 (LoS/NLoS, 3GPP UMa)
│   └── isac_scenario.py    场景生成 + BEV 网格
│
├── solver/            ← 数值优化器
│   └── sca_fp.py           SCA-FP 交替优化 (30 轮外循环)
│
├── data/              ← 数据层
│   ├── prompt_builder.py   Prompt 构造 (自然语言 + BEV 网格)
│   ├── oracle_generator.py Oracle 生成 (Best-of-N 知识蒸馏)
│   └── dataset.py          SFTDataset + DPODataset (共享 _tokenize_pair)
│
├── model/             ← 模型层
│   ├── gemma_isac.py       Gemma3ISAC (Unsloth 4-bit QLoRA + LoRA)
│   ├── projection_head.py  ConstraintProjectionHead (Proj_Q/A/P)
│   └── losses.py           UAVISACLosses (SFT + DPO + 约束损失)
│
├── training/          ← 训练层
│   ├── train_sft.py        Stage I SFT (Accelerate + 4-bit QLoRA)
│   └── train_dpo.py        Stage II DPO (reference model 独立加载)
│
└── eval/              ← 评估层
    └── evaluate.py         6 指标 × 9 基线对比

scripts/
├── generate_data.py    ← 数据生成入口 (支持 multiprocessing)
├── validate_data.py    ← 数据质量验证
├── eda_data.py         ← 探索性数据分析 (EDA)
├── test_sft_overfit.py ← SFT 过拟合测试 (证明训练代码正确性)
└── autodl_setup.sh     ← AutoDL 服务器一键环境搭建

configs/
└── default.yaml        ← 全局超参数 (硬件/模型/训练/仿真/数据)
```

### 关键设计决策

| 决策 | 原因 |
|------|------|
| Unsloth 4-bit QLoRA 而非 bitsandbytes | Blackwell sm_120 无 bitsandbytes 支持 |
| LoRA rank=16, α=32 | 平衡适配能力与显存 (~10GB 模型) |
| Control Token × 8 | 足够编码 176 个优化变量 |
| DPO reference model 独立加载 | 不 deepcopy (会 OOM 在 32GB 卡上) |
| 所有数据路径用 `/root/autodl-tmp/` | 系统盘仅 30GB |

---

## 七轮代码审查

| 轮次 | 文档 | 发现/修复 |
|------|------|----------|
| **#1** | [02_first_review_codex.md](docs/02_code_reviews/02_first_review_codex.md) | 初始代码质量审查 |
| **#2** | [03_second_review.md](docs/02_code_reviews/03_second_review.md) | 第二轮深入审查 |
| **#3** | [04_third_review_gemini.md](docs/02_code_reviews/04_third_review_gemini.md) | Gemini 辅助审查 |
| **#4** | [05_fourth_review_report.md](docs/02_code_reviews/05_fourth_review_report.md) | P0 路径损耗不一致 + evaluate noise_power + NaN guard |
| **#5** | [06_fifth_review_final.md](docs/02_code_reviews/06_fifth_review_final.md) | 第五轮终审 |
| **#6** | [07_sixth_review_final.md](docs/02_code_reviews/07_sixth_review_final.md) | 第六轮终审 |
| **#7** | [08_pre_launch_technical_report.md](docs/01_project_setup/08_pre_launch_technical_report.md) | 上线前完整技术报告 |

### 审查中修复的关键问题

| 问题 | 严重级别 | 修复 commit |
|------|---------|-------------|
| 路径损耗不一致 (通信 vs 感知) | P0 | `9db55d5` |
| DPO deepcopy OOM | P0 | `e5e5025` |
| channel_gain RNG 确定性 | P1 | `e5e5025` |
| evaluate.py 缺失 solver 参数 | P1 | `2b75aa1` |
| AutoDL 路径适配 | P0 | `bfb8d9b` |
| 物理一致性 (动态波长, 统一路径损耗, NaN guard) | P0/P1 | `9db55d5` |

---

## 服务器环境搭建

### 硬件

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090 |
| VRAM | 31.4 GB |
| Compute Capability | 12.0 (Blackwell, `sm_120`) |
| 系统盘 | 30 GB |
| 数据盘 | 50 GB 配额 (`/root/autodl-tmp/`) |

### 软件栈

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.11 | conda env `uavmllm` |
| PyTorch | 2.10.0+cu128 | 深度学习框架 |
| CUDA | 12.8 | GPU 计算 |
| transformers | 4.57.2 | HuggingFace 模型加载 |
| Unsloth | 2025.11.1 | 4-bit QLoRA (替代 bitsandbytes) |
| peft | 0.19.1 | LoRA checkpoint 加载 |
| accelerate | 1.14.0 | 分布式训练 |
| trl | 0.23.0 | DPO 训练 |
| scipy | 1.17.1 | SCA-FP 优化 (L-BFGS-B) |

### 一键搭建

```bash
git clone git@github.com:Lampotaku/UAV-ISAC-MLLM.git /root/UAV-ISAC-MLLM
cd /root/UAV-ISAC-MLLM
bash scripts/autodl_setup.sh
```

---

## 烟雾测试 (5 envs)

在 AutoDL 服务器上运行 5 环境烟雾测试验证管线：

```bash
python scripts/generate_data.py --num-env 5 --num-restarts 10 --save-every 1 \
    --output-dir /root/autodl-tmp/data/smoke_test
```

输出:
```
Done in 125.0s (0.03h)
  SFT: 5  |  DPO: 187
```

### 验证触发告警

运行 `validate_data.py` 后暴露出大规模异常：

```
100 issues found:
  ✗ delta_q 水平位移 max=701.5m > 2*v_max*Δt=30.0m
  ✗ delta_q 垂直位移 max=168.0m > 50m

  δ_q 水平位移: mean=382.4m [88.3, 864.8]
```

物理约束 `v_max·Δt = 15m`，实际 `delta_q` 均值 **382m**，最大值 **864m** — 超过约束 **57 倍**。

---

## 烟雾测试中发现的 P0 Bug

详见 [10_physical_constraint_bug_postmortem](docs/03_bug_postmortems/10_physical_constraint_bug_postmortem.md)。

### Bug: SCA-FP 随机初始化穿透物理约束

**根因**: `_random_init()` 在 Best-of-N 随机重启时将 UAV 随机抛掷在整个 1000×1000m 区域内，完全无视当前物理位置 `q_current`。

```
修复前: Q_random ~ Uniform(0, 1000) for x,y    → delta_q 可达 800m+
修复后: Q_random = q_current + v_max·Δt · random_direction_on_sphere()  → delta_q ≤ 15m
```

**修复**: Box 约束 → Sphere (3D Euclidean) 约束
- `1caa482`: 在 SCA-FP 求解器中强制 `v_max·Δt` 位移约束
- `2b75aa1`: 统一 3D 移动约束 + 添加缺失的 evaluate.py solver 参数
- `14afd9a`: Box→Sphere — 3D Euclidean 惩罚 + 球面采样 + 3D 验证

---

## 本阶段结束状态

| 项目 | 状态 |
|------|------|
| 全部源代码 (18 文件, ~4200 行) | ✅ |
| 7 轮代码审查 | ✅ 25+ 问题闭合 |
| GitHub 私有仓库 | ✅ `Lampotaku/UAV-ISAC-MLLM` |
| AutoDL 服务器环境 | ✅ 一键搭建脚本就绪 |
| 烟雾测试 (5 envs) | ✅ P0 物理约束修复后通过 |
| 数据验证脚本 | ✅ `validate_data.py` |
| EDA 脚本 | ✅ `eda_data.py` |

**下一阶段**: 全量 5000 环境数据生成 → 见 [交接文档 #3](docs/05_handoff/18_handoff_03_datagen_problems.md)

---

## 相关文档

- [[16_handoff_01_project_direction](docs/05_handoff/16_handoff_01_project_direction.md)] — 论文总体方向
- [[18_handoff_03_datagen_problems](docs/05_handoff/18_handoff_03_datagen_problems.md)] — 数据生成中的问题与修复
- [[19_handoff_04_post_datagen](docs/05_handoff/19_handoff_04_post_datagen.md)] — 数据生成后的验证与下一步
- [[10_physical_constraint_bug_postmortem](docs/03_bug_postmortems/10_physical_constraint_bug_postmortem.md)] — P0 物理约束 Bug 详细分析
