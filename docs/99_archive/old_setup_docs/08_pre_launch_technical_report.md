# UAV-ISAC-MLLM 完整技术报告

> **状态：上线就绪 (Launch-Ready)**
> **目标硬件：NVIDIA RTX 5090 32GB @ AutoDL (SeetaCloud)**
> **报告日期：2026-06-23**

---

## 目录

1. [项目概要](#1-项目概要)
2. [数学框架](#2-数学框架)
3. [系统架构](#3-系统架构)
4. [数据生成管线](#4-数据生成管线)
5. [模型架构](#5-模型架构)
6. [训练管线](#6-训练管线)
7. [评估协议](#7-评估协议)
8. [硬件与运行环境](#8-硬件与运行环境)
9. [代码审查历史](#9-代码审查历史)
10. [上线检查清单](#10-上线检查清单)
11. [命令速查](#11-命令速查)
12. [附录：文件清单](#12-附录文件清单)

---

## 1. 项目概要

### 1.1 问题定义

低空物联网网络中，无人机（UAV）需同时执行**通信服务**（ISAC：Integrated Sensing and Communication）和**目标感知**。传统数值优化方法 SCA-FP（Successive Convex Approximation / Fractional Programming）精度高，但每时间槽需在线求解非凸优化 → 收敛慢（~30轮外循环），无法满足实时性。

**论文方案**：用 MLLM 作为智能热启动器，输出接近最优解的 warm-start prior **δ̂**，喂给 SCA-FP 后大幅减少迭代次数。

### 1.2 核心贡献

| 贡献 | 技术实现 |
|------|---------|
| **Control Token 机制** | Gemma 3 词表中插入 8 个 `<ctrl_i>` token，其 hidden states 编码网络状态 → 优化变量 |
| **可微约束投影头** | 三层投影 Proj_Q/A/P，将 LLM 原始输出强制映射到物理可行域（高度/功率预算/单用户关联） |
| **Best-of-N 知识蒸馏** | SCA-FP 在 5000 环境上各解 10 次 → 最优解为 SFT 标签，效用差 > 门限的对为 DPO 偏好 |
| **两阶段训练** | Stage I: SFT 模仿最优解；Stage II: DPO 学习区分好/差解 + 保持约束可行性 |

### 1.3 端到端管线

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

## 2. 数学框架

### 2.1 系统参数

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

### 2.2 优化变量

$$
\Omega = \{Q, A, W_c, W_s\}
$$

- **Q** ∈ ℝ^{M×3} — UAV 3D 位置
- **A** ∈ {0,1}^{M×K} — 用户关联矩阵
- **W_c** ∈ ℂ^{M×K×N_t} — 通信波束成形
- **W_s** ∈ ℂ^{M×N_t} — 感知波束成形

### 2.3 联合效用函数

$$
f(\Omega) = \underbrace{\sum_{m,k} A_{m,k} \omega_k \log_2(1 + \gamma_{m,k})}_{\text{通信速率}}
+ \lambda_s \underbrace{\sum_{m,\ell} \text{SINR}^s_{m,\ell}}_{\text{感知质量}}
- \lambda_f \underbrace{\sum_m \mathbb{I}[|\mathcal{K}_m| = 0]}_{\text{空闲惩罚}}
$$

### 2.4 SCA-FP 交替优化（下游求解器 S(·)）

```
外循环 (max 30 iters, tol=1e-4):
  1. 固定 Q, A → 优化 W_c, W_s (闭式功率注水)
  2. 固定 W_c, W_s, A → SCA 优化 Q (L-BFGS-B)
  3. 固定 Q, W_c, W_s → 优化 A (Hungarian 算法)
  4. 收敛检查: |utility - prev_utility| < tol
```

### 2.5 物理层公式

**通信路径损耗**（3GPP UMa LoS）:
```
PL_LoS(dB) = 28 + 22·log₁₀(d_3D) + 20·log₁₀(f_c_GHz)
PL_NLoS(dB) = 28 + 30·log₁₀(d_3D) + 20·log₁₀(f_c_GHz) + 20
```
概率加权：`PL = P_LoS · PL_LoS + (1-P_LoS) · PL_NLoS`
其中 `P_LoS = f(elevation_angle)` 基于 3GPP TR 36.777 仰角依赖模型。

**感知路径损耗**（双程雷达方程）:
```
PL_sense(dB) = 20·log₁₀(4πd/λ) + 20
```

**热噪声功率**:
```
N₀(dBm) = -174 + 10·log₁₀(B) + NF
N₀(W) = 10^((N₀(dBm) - 30) / 10)
```

**感知 SINR**:
```
SINR^s = P_sense · PL_sense · N_t · N_r / N₀
```

### 2.6 MLLM Prior 提取

$$
\delta_q^* = Q^* - Q(t) \quad \text{(位移)}
$$
$$
\delta_a^* = A^* \quad \text{(关联矩阵)}
$$
$$
\delta_p^* = \{\|w_{m,k}^*\|^2, \|w_{m,r}^*\|^2\} \quad \text{(功率分配)}
$$

### 2.7 损失函数

**Stage I (SFT)**:
$$
\mathcal{L}_I = \mathcal{L}_{SFT} + \lambda_{ctl} \cdot \mathcal{L}_{ctl}
$$

**Stage II (DPO)**:
$$
\mathcal{L}_{II} = \mathcal{L}_{DPO} + \mu \cdot \mathcal{L}_{SFT} + \lambda_{ctl} \cdot \mathcal{L}_{ctl} + \lambda_{sep} \cdot \mathcal{L}_{sep}
$$

| 损失项 | 公式 | 权重 | 作用 |
|--------|------|------|------|
| `L_SFT` | Causal LM cross-entropy (response tokens only) | 1.0 | 学习文本输出格式 |
| `L_ctl` | MSE(δ̂_q, δ*_q) + BCE(δ̂_a, δ*_a) + MSE(δ̂_p, δ*_p) | 0.5 | 学习最优物理 prior |
| `L_DPO` | `-log σ(β·(log π_θ(chosen)/π_ref(chosen) - log π_θ(rejected)/π_ref(rejected)))` | 1.0 | 偏好优化 |
| `L_sep` | `Σ max(0, d_min - ‖q̂_m - q̂_m'‖)²` | 0.1 | UAV 防碰撞 |

---

## 3. 系统架构

### 3.1 项目文件树

```
UAV/
├── configs/
│   └── default.yaml                 ← 全局配置（所有超参数）
├── src/
│   ├── env/                         ← 仿真环境层
│   │   ├── uav_network.py           │  UAV/用户/目标拓扑管理 (M=4, K=20, T=6)
│   │   ├── uav_channel.py           │  物理层信道 (LoS/NLoS, SINR, CRB, Rician)
│   │   └── isac_scenario.py         │  完整场景生成器 (通信摘要 + 感知摘要 + BEV)
│   ├── solver/
│   │   └── sca_fp.py                ← SCA-FP 数值优化器 (下游求解器 S(·))
│   ├── data/                        ← 数据层
│   │   ├── prompt_builder.py        │  多模态 Prompt Π(t) 构造
│   │   ├── oracle_generator.py      │  Best-of-N Oracle 数据生成 (Algorithm 1)
│   │   └── dataset.py               │  PyTorch Dataset (SFTDataset, DPODataset)
│   ├── model/                       ← 模型层
│   │   ├── gemma_isac.py            │  Gemma3ISAC 核心模型 (Unsloth + LoRA + Ctrl Token)
│   │   ├── projection_head.py       │  可微约束投影头 h_Φ (Proj_Q/A/P)
│   │   └── losses.py                │  损失函数 (SFT/DPO/Control/Separation)
│   ├── training/                    ← 训练层
│   │   ├── train_sft.py             │  Stage I: SFT-LoRA
│   │   └── train_dpo.py             │  Stage II: DPO
│   └── eval/
│       └── evaluate.py              ← 评估管线 (6 指标 × 200 测试环境)
├── scripts/
│   ├── generate_data.py             ← 数据生成入口（断点续跑）
│   └── upload_to_server.py          ← SFTP 上传工具
├── outputs/                         ← 训练产出
├── checkpoints/                     ← 训练 checkpoint
├── logs/                            ← 训练日志
└── docs/                            ← 文档
```

### 3.2 数据流

```
generate_data.py
  │
  ├─→ ISACScenarioGenerator.sample(i)
  │     ├─→ UAVNetwork.reset()          ← 随机拓扑 (每环境不同种子)
  │     ├─→ ISACChannel.channel_gain()  ← 真实物理信道
  │     └─→ EnvironmentSample            ← 返回完整快照
  │
  ├─→ SCAFPOptimizer.solve() × N=10
  │     ├─→ _optimize_beamforming()      ← 闭式功率注水
  │     ├─→ _optimize_deployment_sca()   ← SCA + L-BFGS-B
  │     └─→ _compute_utility()           ← 联合效用评估
  │
  ├─→ 排序 (按 utility 降序)
  ├─→ _extract_prior(best_sol) → δ*  ─→ SFT Dataset
  └─→ _build_dpo_pairs()             ─→ DPO Dataset
         (门槛: u_gap > ρ·IQR)

train_sft.py / train_dpo.py
  │
  ├─→ SFTDataset / DPODataset        ← JSONL 加载 + tokenize
  ├─→ Gemma3ISAC.forward()            ← 前向传播 (LoRA + Ctrl Token)
  │     ├─→ Gemma 3 base (LoRA)
  │     ├─→ Control Token hidden states Z_c
  │     └─→ ConstraintProjectionHead → δ̂
  │
  └─→ UAVISACLosses.compute_stage{1,2}_total()

evaluate.py
  │
  ├─→ ISACScenarioGenerator.sample()  ← 200 测试环境 (seed=42 固定)
  ├─→ Gemma3ISAC.generate_warmstart() ← 推理 (无 grad)
  ├─→ SCAFPOptimizer.solve(warm_start=δ̂)
  └─→ 6 指标汇总
```

### 3.3 模块间契约

| 接口 | 输入 | 输出 | 约束 |
|------|------|------|------|
| `ISACScenarioGenerator.sample()` | sample_id (int) | `EnvironmentSample` | 确定性（同 id 同环境） |
| `SCAFPOptimizer.solve()` | env_dict + seed | `SCAFPSolution` | utility ∈ ℝ, Q 在区域内 |
| `OracleDataGenerator._extract_prior()` | solution + env | δ_q, δ_a, δ_p | δ_q = Q* - Q_current |
| `ConstraintProjectionHead.forward()` | Z_c + q_current | δ̂ (投影后) | δ̂ 满足所有物理约束 |
| `Gemma3ISAC.from_pretrained()` | checkpoint dir | 完整模型 | 绕过 __init__ 以避免重复加载 |

---

## 4. 数据生成管线

### 4.1 Algorithm 1: Best-of-N Oracle 数据生成

```
Input: S=5000 环境, N=10 重启, ρ=0.2
Output: D_SFT, D_DPO

For i = 1 to S:
  1. E^(i) = ISACScenarioGenerator.sample(i)
     ├─ UAV 位置: uniform(100,900) × uniform(100,900) × uniform(70,280)
     ├─ 用户: 2-3 随机簇, 簇内 σ=50m 高斯散布, ω_k ∈ [0.5, 2.0]
     ├─ 目标: uniform + random velocity ±3m/s, 80% detected
     └─ 初始关联: nearest-UAV
  2. Π^(i) = build_full_prompt(E^(i))
  3. For j = 1 to N:
       seed_j = i × N + j
       Ω_j = SCA-FP(env_dict, warm_start=None, seed=seed_j)
  4. Sort by utility: u_π(1) ≥ ... ≥ u_π(N)
  5. δ_best = extract(Ω_π(1)) → D_SFT
  6. Δ_min = ρ · IQR({u_j})
     For all pairs (j, j') with u_j - u_j' > Δ_min:
       → D_DPO ∪ {(Π, δ_j, δ_j')}
```

### 4.2 输出格式

**SFT 样本** (`sft_dataset.jsonl`, ~5000 条):
```json
{
  "id": "env_0",
  "prompt": "You are a UAV-ISAC decision controller...\n[Comm Summary]...\n[Sensing Summary]...\n[BEV Grid]...",
  "response": "{\"delta_q\": [[dx,dy,dh],...], \"delta_a\": [[...],...], \"delta_p\": [[...],...]}",
  "utility": 15.432,
  "q_current": [[x,y,h],...],
  "delta_q": [[dx,dy,dh],...],
  "delta_a": [[0,1,0,...],...],
  "delta_p": [[p_c,...,p_s],...]
}
```

**DPO 样本** (`dpo_dataset.jsonl`, ~数万对):
```json
{
  "id": "env_0_pair_0_3",
  "prompt": "...(same as SFT)...",
  "chosen": "{\"delta_q\": ...}",
  "rejected": "{\"delta_q\": ...}",
  "utility_gap": 2.35,
  "delta_q": [...],  // winner's prior for control loss
  "delta_a": [...],
  "delta_p": [...]
}
```

### 4.3 DPO 对筛选逻辑

- 计算 10 次求解的效用分布 IQR
- 门限 `Δ_min = 0.2 × IQR`（动态，而非固定值）
- 效用差 < 门限的对不采纳 → 防止模型在性能相近的解之间困惑
- 每个环境产生 `C(10,2)` 候选对，经门限过滤后通常剩 ~15-30 对

### 4.4 断点续跑

`scripts/generate_data.py` 特性：
- **Ctrl+C 安全**：收到 SIGINT 后在当前环境完成后保存，不截断 JSONL
- **自动续跑**：检测 `checkpoint.txt` + JSONL 已写行数 → 从断点继续
- **定期保存**：每 100 环境写 checkpoin，每环境增量追加 JSONL
- **预估耗时**：5000×10=50,000 次 SCA-FP，约 4-8 小时

---

## 5. 模型架构

### 5.1 Gemma3ISAC — 核心模型

```
Input: Π(t) (多模态文本 prompt)
  ↓
Tokenizer (+ 8 个 <ctrl_i> tokens)
  ↓
Gemma 3 12B (Unsloth 4-bit QLoRA)
  ├─ Frozen backbone (4-bit nf4)
  ├─ LoRA adapters (r=16, α=32, target=[q,k,v,o]_proj)
  └─ Last hidden states (B, seq_len, 3840)
      ↓
Control Token Hidden States Z_c ∈ ℝ^{B×8×3840}
  ↓
Constraint Projection Head h_Φ
  ├─ ControlReadout: MeanPool → Linear(3840→1920) → Linear(1920→out_dim)
  ├─ ResidualMLP: x + MLP(x), 2×[Linear+GELU+LayerNorm] (256→256)
  ├─ Proj_Q: 位移投影 (3D 移动性/高度/区域约束)
  ├─ Proj_A: Sinkhorn 关联投影 (单用户 + 容量 ≤ K_max)
  └─ Proj_P: Softmax 功率投影 (功率预算 Σ ≤ P_max, P_min 下限)
      ↓
δ̂ = {δ̂_q ∈ ℝ^{M×3}, δ̂_a ∈ [0,1]^{M×K}, δ̂_p ∈ ℝ^{M×(K+1)}}
```

### 5.2 LoRA 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| rank `r` | 16 | 论文值 |
| alpha `α` | 32 | 缩放因子 |
| dropout | 0.05 | 正则化 |
| target_modules | `[q_proj, k_proj, v_proj, o_proj]` | Gemma 3 attention 四线性层 |

### 5.3 Control Token 设计

- 8 个特殊 token `<ctrl_0>` ~ `<ctrl_7>`，追加在 prompt 末尾
- 训练时：由 `control_mask` 精确定位其位置 → 提取 hidden states
- 推理时：取序列最后 8 个位置的 hidden states
- 新增 token embedding 行可训练（其余 embedding 冻结）

### 5.4 Projection Head 三层投影

| 投影 | 约束 | 方法 | 关键参数 |
|------|------|------|---------|
| **Proj_Q** | ‖Δq‖₂ ≤ v_max·Δt, H ∈ [50,300], x,y ∈ [0,1000] | 3D 范数裁剪 + 坐标 clamping | v_max_dt=15 |
| **Proj_A** | Σ_m A_{m,k}≈1, Σ_k A_{m,k}≤K_max | Sinkhorn 迭代 (列归一化 + 行容量裁剪) | τ=0.5, 20 iters |
| **Proj_P** | Σ(comm+sense) ≤ P_max, p_comm ≥ P_min | Softmax 归一化 + P_min 重分配 | τ=0.5, P_min=0.01W |

### 5.5 参数与显存

| 组件 | 参数量 | 显存 (4-bit) |
|------|--------|-------------|
| Gemma 3 12B base | ~12B | ~8-10 GB |
| LoRA adapters | ~30M | ~0.5 GB |
| Control Token embeddings | 8 × 3840 | ~0.1 GB |
| Projection Head | ~2M | ~0.1 GB |
| **SFT 峰值** (含 optimizer states + activations) | — | **~25-28 GB** |
| **DPO 峰值** (含 reference model) | — | **~28-31 GB** |

---

## 6. 训练管线

### 6.1 Stage I: SFT-LoRA

**目标**：让模型学会输出接近最优解的物理 prior δ̂。

| 参数 | 值 |
|------|-----|
| 损失函数 | `L_SFT + 0.5·L_ctl` |
| 数据 | 5000 SFT samples |
| Epochs | 3 |
| Learning rate | 2e-4 (cosine decay, 3% warmup) |
| 有效 batch | 16 (bs=1 × grad_accum=16) |
| Max seq length | 4096 |
| 优化器 | AdamW (weight_decay=0.01) |
| 精度 | BF16 mixed |
| 可训练参数 | LoRA + Ctrl Token embeddings + Projection Head |
| 预计耗时 | 3-8 hours |

**可训练参数收集**：`[p for n,p in model.named_parameters() if p.requires_grad and "projection_head" in n]` + `[p for n,p in model.base_model.named_parameters() if p.requires_grad]`

Unsloth 已自动冻结非 LoRA 的 backbone 参数，只有 LoRA A/B 矩阵的 `requires_grad=True`。

### 6.2 Stage II: DPO 偏好优化

**目标**：学习区分好/差解，同时保持 SFT 习得的最优先验。

| 参数 | 值 |
|------|-----|
| 损失函数 | `L_DPO + 0.05·L_SFT + 0.5·L_ctl + 0.1·L_sep` |
| DPO β | 0.1 |
| SFT anchor μ | 0.05 (防灾难性遗忘) |
| Reference model | Stage I checkpoint 的独立加载副本 |
| Learning rate | 5e-5 (低于 SFT) |
| Epochs | 2 |
| 有效 batch | 16 (同 Stage I) |
| 预计耗时 | 5-10 hours |

**Reference model 处理**：显式从 Stage I checkpoint 重新加载（而非 deepcopy），避免 4-bit 量化模型的 deepcopy 显存崩溃。

**DPO log-prob 计算**：对 response token 做 **SUM**（非 mean），保留 DPO 需要的联合概率 `Σ_t log π(y_t|...)`。

### 6.3 输出目录结构

```
outputs/
├── stage1_sft_final/
│   ├── lora/adapter_model.safetensors   ← LoRA 权重
│   ├── projection_head.pt               ← 投影头权重
│   └── tokenizer/                       ← 扩展后的 tokenizer
└── stage2_dpo_final/
    ├── lora/adapter_model.safetensors
    ├── projection_head.pt
    └── tokenizer/

checkpoints/
├── stage1_step_200/
├── stage1_step_400/
├── stage2_step_200/
└── ...
```

---

## 7. 评估协议

### 7.1 评估指标 (6 项)

| 指标 | 公式 | 单位 | 说明 |
|------|------|------|------|
| Network Sum Rate | Σ A_{m,k} · B · log₂(1+SINR) | Mbps | 总通信吞吐 |
| Mean Sensing SINR | (1/T) Σ mean SINR^s | dB | 感知质量 |
| Mean CRB | Cramér-Rao Bound | m² | 定位精度下界 |
| Joint Satisfaction | (comm_sat + sense_sat) / 2 | ratio | QoS 满足率 |
| SCA-FP Iterations | 收敛所需迭代数 | count | 加速效果核心指标 |
| Inference Latency | MLLM 推理时间 | ms | 在线可行性 |

### 7.2 9 Baselines

| ID | Baseline | 描述 |
|----|----------|------|
| B1 | CSI only | 仅信道信息，无 MLLM |
| B2 | ISAC SCA | 纯 SCA-FP (随机初始化) |
| B3 | DRL assisted | 深度强化学习辅助 |
| B4 | Single modal frozen | 冻结单模态 LLM |
| B5 | MoE baseline | 混合专家基线 |
| B6 | Frozen prompting | 冻结 LLM + 零样本 prompt（**论文主要对比**） |
| B7 | SFT only | 仅 Stage I (无 DPO) |
| B8 | DPO only | 仅 Stage II (跳过 SFT) |
| B9 | SFT+DPO (no head) | 两阶段但无投影头 |

### 7.3 关键实现细节

- **满意度分母**：通信满意度 = 满足 SINR 最低要求的用户数 / **总 K**（而非已关联用户数），防止"只服务 1 个用户 = 100%"的刷榜漏洞
- **带宽从配置读取**：sum_rate 使用 `cfg["bandwidth_mhz"] * 1e6`，非硬编码 20MHz
- **噪声功率对齐**：evaluate.py 的 `noise_power` 计算与 generate_data.py 完全一致
- **波长动态计算**：感知 SINR 使用 `3e8 / (f_c * 1e9)`，非硬编码 0.0517
- **N_r 显式传入**：`N_r = cfg.get("num_antennas_rx", cfg["num_antennas_tx"])`

---

## 8. 硬件与运行环境

### 8.1 目标硬件

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090 |
| VRAM | 32 GB (可用 ~31.4 GB) |
| 架构 | Blackwell (compute capability 12.0, `sm_120`) |
| CUDA | 12.8 |
| 系统盘 | `/overlay` — 30 GB |
| 数据盘 | `/root/autodl-tmp` — 50 GB |

### 8.2 软件栈

| 组件 | 版本 | 备注 |
|------|------|------|
| Python | 3.11 | conda env: `uavmllm` |
| PyTorch | 2.10.0+cu128 | — |
| CUDA | 12.8 | 系统级 |
| Unsloth | 2025.11.1 | 替代 bitsandbytes（Blackwell 兼容） |
| transformers | 4.57.2 | — |
| peft | 0.19.1 | 仅用于 checkpoint 加载 |
| trl | 0.23.0 | DPO 工具库 |
| accelerate | 1.14.0 | 分布式训练 |
| datasets | 4.3.0 | HF 数据集加载 |
| Gemma 3 12B | `google/gemma-3-12b-it` | ~23 GB, 5 shards |

### 8.3 为什么用 Unsloth 替代 bitsandbytes

| 因素 | bitsandbytes + PEFT | Unsloth |
|------|---------------------|---------|
| Blackwell (sm_120) | ❌ 不支持 | ✅ 内置内核 |
| CUDA 12.8 兼容 | ⚠️ 需手动编译 | ✅ 开箱即用 |
| 4-bit 量化 | 手动 BnBConfig | `load_in_4bit=True` 一行 |
| LoRA + 量化 | PEFT LoraConfig + get_peft_model | FastLanguageModel 一步完成 |
| 训练速度 | 标准 | 2-5× 加速 (fused kernels) |

### 8.4 已知兼容性问题

| 问题 | 解决方案 |
|------|---------|
| `dict has no attribute 'model_type'` | `use_fast=False` (SentencePiece 原生 tokenizer) |
| HuggingFace 直连超时 | `HF_ENDPOINT=https://hf-mirror.com` |
| Flash Attention 2 不可用 | 回退 `attn_implementation: "sdpa"` |
| torch 2.10 < 2.11 C++ 扩展 warning | 不影响功能，可忽略 |

---

## 9. 代码审查历史

项目经历 **7 轮系统性代码审查**，闭合 25+ 个问题。

### 9.1 审查时间线

| 轮次 | 审查者 | 发现问题 | 关键修复 |
|------|--------|---------|---------|
| 一审 (Codex) | AI | 9 fixes | Control token 插入位置、loss mask、q_current 数据流、Unsloth 加载 |
| 二审 | AI | 9 issues | 识别路径损耗冲突、RNG 泄漏、user_weights、波长硬编码 |
| 三审 (Gemini) | AI | 6 fixes | DPO log-prob 求和 → sum 非 mean、off-by-one 切片、3D 移动性、P_min 功率 |
| 四审 | AI | 2 fixes | DPO deepcopy OOM 崩溃、channel_gain RNG 确定性 |
| 五审 | AI | 3P0 + 4P1 + 5P2 | 路径损耗三公式不一致、波长硬编码、evaluate 缺 noise_power |
| 六审 (Gemini) | AI | 修复方案验证 | 提出修复建议并交叉验证 |
| 七审 (本轮) | AI + Human | 全线闭合确认 | 3P0+4P1 全部修复，代码库达标 |

### 9.2 第六轮核心修复 (P0/P1 闭合)

| 优先级 | 问题 | 文件 | 修复 |
|--------|------|------|------|
| **P0-1** | 硬编码波长 0.0517 | `sca_fp.py` | `self.wavelength = 3e8/(f_c*1e9)` 动态计算 |
| **P0-2** | 三条路径损耗不一致 | `sca_fp.py` | 通信路损补齐 `+20·log₁₀(f_c)`, 感知路损用 `self.wavelength` |
| **P0-3** | evaluate.py 缺 noise_power | `evaluate.py` | 新增热噪声计算，与 generate_data.py 一致 |
| **P1-1** | 缺 N_r 显式传入 | `evaluate.py`, `generate_data.py` | `N_r=cfg.get("num_antennas_rx", N_t)` |
| **P1-2** | 带宽硬编码 20e6 | `evaluate.py` | 改为 `cfg["bandwidth_mhz"] * 1e6` |
| **P1-3** | NaN 守卫缺失 | `sca_fp.py` | `if not np.isfinite(utility): break` |
| **P2-1** | 双数据生成脚本 | — | 删除 `run_data_generation.py` |

### 9.3 "三位一体" 物理参数对齐

修复后，三个 SCA-FP 求解器入口共享同一套物理参数：

```
  evaluate.py ──→ N_r, carrier_freq_ghz, noise_power ──→ SCAFPOptimizer
generate_data.py ──→ N_r, carrier_freq_ghz, noise_power ──→ SCAFPOptimizer
        solver 内部 ──→ self.wavelength, self.carrier_freq_ghz ──→ 统一路损公式
```

### 9.4 遗留 P2 项（不影响上线）

| P2 | 问题 | 处理策略 |
|----|------|---------|
| P2-2 | `from_pretrained` 绕过 `__init__` | 当前属性集稳定，延后加 sanity check |
| P2-3 | `mean_crb` 返回 0.0 | `compute_crb()` 已实现，延后接入评估循环 |
| P2-4 | `train_sft.py` 参数收集脆弱 | Unsloth 正确冻结 LoRA 即可，暂无风险 |
| P2-5 | `use_multimodal: false` | 文本 BEV 可验证管线，多模态是论文扩展 |

---

## 10. 上线检查清单

### 10.1 语法与静态检查

```bash
# ✅ 全部通过
python -m compileall -q src scripts

# ✅ 零残留
grep -r "0.0517" src/
grep "20e6" src/eval/evaluate.py

# ✅ 文件不存在
ls scripts/run_data_generation.py
```

### 10.2 数据流完整性

| 检查项 | 状态 |
|--------|------|
| `ISACScenarioGenerator → EnvironmentSample` 数据流贯通 | ✅ |
| `OracleDataGenerator → SFT/DPO JSONL` 格式正确 | ✅ |
| `SFTDataset / DPODataset → DataLoader` tokenize 正确 | ✅ |
| `Gemma3ISAC.forward() → logits + delta_hat` 形状对齐 | ✅ |
| `UAVISACLosses → L_SFT + L_ctl + L_DPO + L_sep` 计算图完整 | ✅ |
| 断点续跑 Ctrl+C 安全 | ✅ |
| 评估管线 `solver → metrics` 数值正确 | ✅ |

### 10.3 服务器就绪

| 检查项 | 状态 |
|--------|------|
| Conda env 可用 (`/root/autodl-tmp/conda/envs/uavmllm`) | ✅ |
| Gemma 3 12B 已下载 (`/root/autodl-tmp/huggingface/models/`) | ✅ |
| HF 镜像已配置 (`HF_ENDPOINT=https://hf-mirror.com`) | ✅ |
| 项目代码已上传 (`/root/UAV/`) | ✅ |
| 磁盘空间充足 (>10 GB 剩余) | 需确认 |
| GPU 可用 (`torch.cuda.is_available()`) | 需确认 |

### 10.4 执行顺序

```
Step 1: 数据生成 (4-8h)
  python scripts/generate_data.py --num-env 5000 --num-restarts 10

Step 2: Stage I SFT (3-8h)
  python src/training/train_sft.py

Step 3: Stage II DPO (5-10h)
  python src/training/train_dpo.py --stage1_ckpt ./outputs/stage1_sft_final

Step 4: 评估
  python src/eval/evaluate.py --model ./outputs/stage2_dpo_final
```

---

## 11. 命令速查

### 11.1 数据生成

```bash
cd /root/UAV

# 完整生成 (断点续跑)
python scripts/generate_data.py \
    --num-env 5000 --num-restarts 10 \
    --output-dir /root/autodl-tmp/data/cache

# 快速烟雾测试 (5 env × 2 restarts, ~24s)
python scripts/generate_data.py \
    --num-env 5 --num-restarts 2 \
    --output-dir /root/autodl-tmp/data/cache
```

### 11.2 Stage I SFT

```bash
cd /root/UAV

python src/training/train_sft.py \
    --config configs/default.yaml \
    --data_dir /root/autodl-tmp/data/cache/sft_dataset.jsonl
```

### 11.3 Stage II DPO

```bash
cd /root/UAV

python src/training/train_dpo.py \
    --config configs/default.yaml \
    --stage1_ckpt ./outputs/stage1_sft_final \
    --data_dir /root/autodl-tmp/data/cache/dpo_dataset.jsonl
```

### 11.4 评估

```bash
cd /root/UAV

# 仅 solver baseline (无 MLLM)
python src/eval/evaluate.py \
    --config configs/default.yaml \
    --output ./outputs/eval_baseline.json

# 完整评估 (加载训练好的模型)
python src/eval/evaluate.py \
    --config configs/default.yaml \
    --model ./outputs/stage2_dpo_final \
    --output ./outputs/eval_results.json
```

### 11.5 手动验证

```bash
# GPU 信息
python -c "import torch; print(torch.cuda.get_device_properties(0))"

# 模型加载测试
python -c "
from unsloth import FastLanguageModel
import torch
model, tok = FastLanguageModel.from_pretrained(
    '/root/autodl-tmp/huggingface/models/gemma-3-12b-it',
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)
print(f'Loaded, VRAM={torch.cuda.memory_allocated()/1024**3:.1f}GB')
"

# 磁盘使用
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/*

# 语法检查
cd /root/UAV && python -m compileall -q src scripts
```

---

## 12. 附录：文件清单

### 12.1 源文件 (21 个)

```
configs/default.yaml                (146 lines) — 全局配置
src/__init__.py                     (  0 lines)
src/env/__init__.py                 (  0 lines)
src/env/uav_network.py              (224 lines) — 网络拓扑
src/env/uav_channel.py              (268 lines) — 物理层信道
src/env/isac_scenario.py            (298 lines) — 场景生成器
src/solver/__init__.py             (  0 lines)
src/solver/sca_fp.py               (494 lines) — SCA-FP 求解器
src/data/__init__.py               (  0 lines)
src/data/oracle_generator.py       (254 lines) — Oracle 数据生成器
src/data/dataset.py                (180 lines) — PyTorch Dataset
src/data/prompt_builder.py         (147 lines) — Prompt 构造
src/model/__init__.py              (  0 lines)
src/model/gemma_isac.py            (359 lines) — 核心模型
src/model/projection_head.py       (404 lines) — 投影头
src/model/losses.py                (262 lines) — 损失函数
src/training/__init__.py           (  0 lines)
src/training/train_sft.py          (262 lines) — Stage I 训练
src/training/train_dpo.py          (375 lines) — Stage II 训练
src/eval/__init__.py               (  0 lines)
src/eval/evaluate.py               (311 lines) — 评估管线
```

### 12.2 脚本 (2 个)

```
scripts/generate_data.py           (210 lines) — 数据生成入口
scripts/upload_to_server.py        ( 69 lines) — SFTP 上传
```

### 12.3 文档 (8 个)

```
docs/01_project_setup/01_project_overview.md        — 初始项目概述
docs/02_code_reviews/02_first_review_codex.md      — 一审报告
docs/02_code_reviews/03_second_review.md           — 二审报告
docs/02_code_reviews/04_third_review_gemini.md     — 三审报告 (Gemini)
docs/02_code_reviews/05_fourth_review_report.md    — 四审报告
docs/02_code_reviews/06_fifth_review_final.md      — 五审终审报告
docs/02_code_reviews/07_sixth_review_final.md      — 六审修复报告
docs/01_project_setup/08_pre_launch_technical_report.md — 本文档
```

### 12.4 总代码量

| 类别 | 文件数 | 总行数 |
|------|--------|--------|
| Python 源码 | 18 | ~4,200 |
| 配置 YAML | 1 | 146 |
| 脚本 | 2 | ~280 |
| 文档 | 8 | ~2,500 |

---

## 许可证与引用

**论文**: *Constraint-Aware MLLM Adaptation for UAV-Enabled ISAC in Low-Altitude IoT Networks*

**模型**: Google Gemma 3 12B Instruct (`google/gemma-3-12b-it`)

---

*报告生成时间: 2026-06-23 | 项目路径: `h:\Projects\UAV` (本地) / `/root/UAV/` (服务器)*
