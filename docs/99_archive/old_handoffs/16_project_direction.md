# 交接文档 #1 — 论文总体方向与项目概要

> 写给接手此项目的新成员：读完本文档即可理解"我们在做什么、为什么这么做"。
> 最后更新: 2026-06-25 | 状态: 数据生成完成，准备 SFT 训练

---

## 目录

1. [一句话概括](#一句话概括)
2. [论文背景与问题定义](#论文背景与问题定义)
3. [核心贡献](#核心贡献)
4. [方法总览](#方法总览)
5. [系统参数](#系统参数)
6. [数学框架（简化版）](#数学框架简化版)
7. [项目仓库与硬件](#项目仓库与硬件)

---

## 一句话概括

**用 Gemma 3 12B 大模型（LoRA + 约束投影头）为无人机通信感知一体化（UAV-ISAC）的数值优化器（SCA-FP）提供智能热启动，大幅减少优化迭代次数。**

论文: *Constraint-Aware MLLM Adaptation for UAV-Enabled ISAC in Low-Altitude IoT Networks*

---

## 论文背景与问题定义

### 场景

低空物联网网络中，无人机（UAV）需要同时执行两项任务：

| 任务 | 描述 |
|------|------|
| **通信服务 (Communication)** | 为地面 20 个 IoT 用户提供下行链路，分配波束成形与功率 |
| **目标感知 (Sensing)** | 对地面 6 个目标发射感知波束，检测/定位 |

这种"通信+感知一体化"被称为 **ISAC (Integrated Sensing and Communication)**。

### 核心挑战

传统数值优化方法 **SCA-FP**（Successive Convex Approximation / Fractional Programming）精度高，但每个时间槽需要在线求解一个非凸优化问题：

- **外循环**: 最多 30 轮迭代才收敛
- **每轮**: 交替优化 UAV 位置 Q、用户关联 A、通信波束 W_c、感知波束 W_s
- **计算代价**: 每次求解需要秒级时间 → 无法满足实时性

### 论文方案：MLLM 智能热启动

核心洞察——SCA-FP 的收敛速度严重依赖于初始猜测的质量。如果用一个训练过的 MLLM 先输出一个"接近最优解"的猜测 δ̂，喂给 SCA-FP 作为起点，迭代次数可以大幅减少。

```
传统方法:  随机初始化  →  SCA-FP 迭代 ~30 轮  →  收敛
本论文:    MLLM 推理  →  warm-start δ̂  →  SCA-FP 迭代 5-10 轮  →  收敛
                                        ↑ 加速 2-5×
```

MLLM 不是替代优化器——它是**学习预测"从历史求解经验中总结出的最优解模式"**。

---

## 核心贡献

| 贡献 | 技术实现 | 论文位置 |
|------|---------|---------|
| **Control Token 机制** | Gemma 3 词表中插入 8 个 `<ctrl_i>` token，其 hidden states 编码网络状态 → 优化变量 | Section 4.1 |
| **可微约束投影头** | 三层投影 Proj_Q/A/P，将 LLM 原始输出强制映射到物理可行域（高度/功率预算/单用户关联） | Section 4.1 |
| **Best-of-N 知识蒸馏** | SCA-FP 在 5000 环境上各求解 10 次 → 最优解为 SFT 标签，效用差 > ε 的 pair 为 DPO 偏好 | Section 4.2 |
| **两阶段训练** | Stage I: SFT 模仿最优解；Stage II: DPO 学习区分好/差解 + 保持约束可行性 | Section 4.2-4.3 |

---

## 方法总览

```
┌─ Offline / Training ───────────────────────────────────────────┐
│                                                                  │
│  ISACScenarioGenerator → SCA-FP (N=10 restarts) → Best-of-N 排序 │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    ▼                   ▼                         │
│              SFT Dataset          DPO Dataset                   │
│            (δ_best → label)    (δ_win, δ_lose pairs)            │
│                    │                   │                         │
│                    ▼                   ▼                         │
│            Stage I SFT           Stage II DPO                   │
│        L = L_SFT + λL_ctl   L = L_DPO + μL_SFT + λL_ctl + λL_sep│
│                                                                  │
│  产出: outputs/stage2_dpo_final/                                 │
└──────────────────────────────────────────────────────────────────┘

┌─ Online / Inference ────────────────────────────────────────────┐
│                                                                  │
│  Π(t) → Gemma 3 (LoRA + Ctrl Token) → h_Φ(Z_c) → δ̂              │
│                                                                  │
│  δ̂ → SCA-FP (warm start) → Ω* → UAV 控制指令                    │
│                                                                  │
│  指标: SCA-FP 收敛加速 2-5×, 通信+感知 QoS 保持                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 系统参数

| 符号 | 含义 | 值 |
|------|------|----|
| `M` | UAV 数量 | 4 |
| `K` | 地面 IoT 用户数 | 20 |
| `T` | 感知目标数 | 6 |
| `N_t, N_r` | 发射/接收天线数 | 8, 8 |
| `f_c` | 载波频率 | 5.8 GHz |
| `B` | 带宽 | 20 MHz |
| `P_max` | 最大发射功率 | 30 dBm (1W) |
| `H_min, H_max` | UAV 高度范围 | 50–300 m |
| `v_max` | 最大飞行速度 | 15 m/s |
| `Δt` | 时间槽 | 1.0 s |
| `Area` | 区域 | 1000×1000 m² |
| `v_max·Δt` | 每槽最大位移 | 15 m |

### 优化变量

```
Ω = {Q, A, W_c, W_s}
```

- **Q** ∈ ℝ^{M×3} — UAV 3D 位置 (x, y, h)
- **A** ∈ {0,1}^{M×K} — 用户关联矩阵
- **W_c** ∈ ℂ^{M×K×N_t} — 通信波束成形向量
- **W_s** ∈ ℂ^{M×N_t} — 感知波束成形向量

### MLLM Prior (δ̂)

```
δ_q = Q* - Q(t)        — UAV 位移建议 (4×3, 每行 [dx, dy, dh], 受 v_max·Δt=15m 约束)
δ_a = A*               — 用户关联矩阵 (4×20, 每行 sum ≤ K_max=5, Sinkhorn 投影)
δ_p = {P_c, P_s}       — 功率分配 (4×21, 20 用户 + 1 感知, softmax → sum ≤ P_max=1W)
```

总共: 4×3 + 4×20 + 4×21 = **176 个浮点数/整数** 作为 LLM 输出

---

## 数学框架（简化版）

### 联合效用函数

```
f(Ω) = Σ_{m,k} A_{m,k}·ω_k·log₂(1 + γ_{m,k})     ← 通信速率
     + λ_s · Σ_{m,ℓ} SINR^s_{m,ℓ}                 ← 感知质量
     - λ_f · Σ_m I[|K_m| = 0]                      ← 空闲 UAV 惩罚
```

### SCA-FP 交替优化（下游求解器）

```
外循环 (max 30 iters, tol=1e-4):
  1. 固定 Q, A → 优化 W_c, W_s (闭式功率注水)
  2. 固定 W_c, W_s, A → SCA 优化 Q (L-BFGS-B)
  3. 固定 Q, W_c, W_s → 优化 A (Hungarian 算法)
  4. 收敛检查: |utility - prev_utility| < tol
```

### 训练损失

**Stage I SFT**:
```
L_I = L_SFT + λ_ctl × L_ctl

L_SFT = causal LM cross-entropy (masked to response tokens only)
L_ctl = λ_q·MSE(δ_q) + λ_a·BCE(δ_a) + λ_p·MSE(δ_p)
```

**Stage II DPO**:
```
L_II = L_DPO + μ·L_SFT + λ_ctl·L_ctl + λ_sep·L_sep

L_DPO = -log σ(β(log π_θ(y_w|x) - log π_θ(y_l|x))
                 - β(log π_ref(y_w|x) - log π_ref(y_l|x)))
L_sep = MSE(q_current + δ_q, q_current + δ_q_ref)  ← 防止 DPO 偏离约束
```

---

## 项目仓库与硬件

| 项 | 值 |
|----|-----|
| GitHub | `Lampotaku/UAV-ISAC-MLLM` (private) |
| 本地开发 | Windows, `h:\Projects\UAV` |
| 训练服务器 | AutoDL, RTX 5090 32GB |
| 服务器路径 | `/root/UAV-ISAC-MLLM` |
| 数据盘 | `/root/autodl-tmp/` (系统盘仅 30GB) |
| GPU | Blackwell sm_120, CUDA 12.8 |
| 量化方案 | Unsloth 4-bit QLoRA (bitsandbytes 不支持 Blackwell) |
| Python | 3.11, conda env: `uavmllm` |
| Backbone | `google/gemma-3-12b-it` |

---

## 相关文档

- [[17_handoff_02_pre_datagen](docs/05_handoff/17_handoff_02_pre_datagen.md)] — 数据生成前的准备工作
- [[18_handoff_03_datagen_problems](docs/05_handoff/18_handoff_03_datagen_problems.md)] — 数据生成中的问题与修复
- [[19_handoff_04_post_datagen](docs/05_handoff/19_handoff_04_post_datagen.md)] — 数据生成后：验证结果与下一步
- [[09_handoff_document](docs/01_project_setup/09_handoff_document.md)] — 原始技术交接文档
- [[08_pre_launch_technical_report](docs/01_project_setup/08_pre_launch_technical_report.md)] — 完整技术报告
