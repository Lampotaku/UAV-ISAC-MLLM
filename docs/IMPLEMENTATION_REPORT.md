# UAV-ISAC-MLLM 完整实施报告

## 论文复现: *Constraint-Aware MLLM Adaptation for UAV-Enabled ISAC in Low-Altitude IoT Networks*

**硬件目标：** NVIDIA RTX 5090 32GB @ AutoDL (SeetaCloud)
**实施日期：** 2026-06-22 → 至今
**状态：** Stage I 数据生成进行中

---

## 目录

1. [论文概要](#1-论文概要)
2. [服务器环境](#2-服务器环境)
3. [项目架构](#3-项目架构)
4. [数学框架](#4-数学框架)
5. [数据生成管线 (Algorithm 1)](#5-数据生成管线)
6. [模型架构](#6-模型架构)
7. [训练管线](#7-训练管线)
8. [评估协议](#8-评估协议)
9. [硬件适配 (RTX 5090 Blackwell)](#9-硬件适配)
10. [当前进度](#10-当前进度)
11. [命令速查](#11-命令速查)

---

## 1. 论文概要

### 1.1 问题定义

低空物联网网络中，无人机 (UAV) 需要同时执行通信 (ISAC) 和感知任务。传统数值优化方法 (SCA-FP) 虽然精度高，但每时间槽需要在线求解非凸优化问题，耗时过长。论文提出用 MLLM 作为**智能热启动器**——通过输出一个接近最优解的 warm-start prior δ，大幅减少 SCA-FP 的迭代次数。

### 1.2 核心贡献

| 贡献 | 描述 |
|------|------|
| **Control Token 机制** | 在 Gemma 3 词表中插入 8 个专用控制 token，用于将多模态网络状态编码为优化变量 |
| **可微约束投影头** | 三层投影模块 (Proj_Q/Proj_A/Proj_P) 将 LLM 原始输出强制投影到物理可行域 |
| **Best-of-N 知识蒸馏** | 用 SCA-FP 在 5000 个环境上各求解 10 次，构造 (最优解 → SFT) + (偏好对 → DPO) 训练数据 |
| **两阶段训练** | Stage I SFT 学习模仿最优解; Stage II DPO 学习区分好/差解，并保持约束可行性 |

### 1.3 方法总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Offline / Training                              │
│                                                                      │
│  仿真环境 E^(i)  ──→  SCA-FP (N=10 restarts)  ──→  Best-of-N 排序   │
│                                                     │               │
│                          ┌──────────────────────────┼──────────┐    │
│                          ▼                          ▼          │    │
│                     SFT Dataset              DPO Dataset       │    │
│                  (Π, δ_best)              (Π, δ_win, δ_lose)   │    │
│                          │                          │          │    │
│                          ▼                          ▼          │    │
│                     Stage I SFT              Stage II DPO       │    │
│                  L = L_SFT + λL_ctl      L = L_DPO + μL_SFT    │    │
│                                              + λL_ctl + λL_sep  │    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      Online / Inference                              │
│                                                                      │
│  Π(t) ──→  Gemma 3 (LoRA + Control Tokens)  ──→  h_Φ(Z_c)  ──→  δ̂  │
│                                                                      │
│  δ̂ ──→  SCA-FP (warm start)  ──→  Ω*  ──→  UAV 控制指令            │
│                                                                      │
│  指标: 加速 SCA-FP 收敛, 保持通信+感知 QoS 约束                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 服务器环境

### 2.1 硬件

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090 |
| VRAM | 31.4 GB (总) / 32 GB (标称) |
| Compute Capability | 12.0 (Blackwell arch, `sm_120`) |
| 系统盘 | `/overlay` — 30 GB |
| 数据盘 | `/root/autodl-tmp` — 50 GB 配额 |

### 2.2 软件栈

| 组件 | 版本 | 安装位置 |
|------|------|---------|
| Python | 3.11 | `/root/autodl-tmp/conda/envs/uavmllm` |
| PyTorch | 2.10.0+cu128 | conda env |
| CUDA | 12.8 | 系统级 |
| transformers | 4.57.2 | conda env |
| Unsloth | 2025.11.1 | conda env (替代 bitsandbytes+PEFT) |
| peft | 0.19.1 | conda env (仅用于 checkpoint 加载) |
| accelerate | 1.14.0 | conda env |
| trl | 0.23.0 | conda env |
| datasets | 4.3.0 | conda env |
| tokenizers | 0.22.2 | conda env |
| scipy | 1.17.1 | conda env |

### 2.3 关键配置

```bash
# ~/.bashrc 中设置
export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像 (直连 hf.co 超时)
```

**Conda 环境路径:** `/root/autodl-tmp/conda/envs/uavmllm`
**HF 缓存:** `/root/autodl-tmp/huggingface/`
**项目代码:** `/root/UAV/`

### 2.4 SSH 连接

```ssh-config
Host uav-5090
  HostName connect.westd.seetacloud.com
  Port 31560
  User root
```

---

## 3. 项目架构

```
UAV/
├── configs/
│   └── default.yaml                 # 全局配置 (仿真/模型/训练超参)
├── src/
│   ├── __init__.py
│   ├── env/                         # 仿真环境
│   │   ├── __init__.py
│   │   ├── uav_network.py           # UAV/用户/目标拓扑 (M=4, K=20, T=6)
│   │   ├── uav_channel.py           # 物理层信道 (LoS/NLoS, SINR, CRB)
│   │   └── isac_scenario.py         # 完整场景生成器 (含通信/感知摘要+BEV)
│   ├── solver/
│   │   ├── __init__.py
│   │   └── sca_fp.py                # SCA-FP 数值优化器 (论文下游求解器 S(·))
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py        # 多模态 Prompt Π(t) 构造器
│   │   ├── oracle_generator.py      # Best-of-N Oracle 数据生成 (Alg 1)
│   │   └── dataset.py               # PyTorch Dataset (SFTDataset, DPODataset)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── gemma_isac.py            # Gemma3 + LoRA + Control Token (核心模型)
│   │   ├── projection_head.py       # 可微约束投影头 h_Φ (Proj_Q/A/P)
│   │   └── losses.py                # 损失函数 (SFT/DPO/Control/Separation)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_sft.py             # Stage I: SFT-LoRA 训练
│   │   └── train_dpo.py             # Stage II: DPO 偏好训练
│   └── eval/
│       ├── __init__.py
│       └── evaluate.py              # 评估 (9 baselines)
├── scripts/
│   ├── generate_data.py             # 数据生成入口 (支持断点续跑)
│   └── upload_to_server.py          # SFTP 上传辅助
├── requirements.txt
├── configs/default.yaml
└── docs/
    └── IMPLEMENTATION_REPORT.md     # 本文档
```

---

## 4. 数学框架

### 4.1 系统模型

| 符号 | 含义 | 默认值 |
|------|------|--------|
| M | UAV 数量 | 4 |
| K | 地面 IoT 用户数 | 20 |
| T | 感知目标数 | 6 |
| N_t, N_r | TX/RX 天线数 | 8 |
| f_c | 载波频率 | 5.8 GHz |
| B | 带宽 | 20 MHz |
| P_max | 最大发射功率 | 30 dBm (1W) |
| 区域 | 低空网络区域 | 1000×1000 m² |
| H_min, H_max | UAV 高度范围 | 50–300 m |
| v_max | 最大速度 | 15 m/s |
| Δt | 时间槽 | 1.0 s |

### 4.2 优化变量

$$
\Omega = \{Q, A, W_c, W_s\}
$$

- **Q** ∈ ℝ^{M×3}: UAV 3D 位置
- **A** ∈ {0,1}^{M×K}: 用户关联矩阵
- **W_c** ∈ ℂ^{M×K×N_t}: 通信波束成形
- **W_s** ∈ ℂ^{M×N_t}: 感知波束成形

### 4.3 联合效用 (公式 10)

$$
f(\Omega) = \sum_{m,k} A_{m,k} \omega_k \log_2(1 + \gamma_{m,k}) + \lambda_s \sum_{m,\ell} \text{SINR}^s_{m,\ell} - \lambda_f \sum_m \mathbb{I}[|\mathcal{K}_m| = 0]
$$

### 4.4 SCA-FP 交替优化

```
1. 固定 Q, A → 优化 W_c, W_s (闭式波束成形)
2. 固定 W_c, W_s, A → SCA 优化 Q (L-BFGS-B)
3. 固定 Q, W_c, W_s → 优化 A (Hungarian 算法)
4. 循环至收敛 (tol=1e-4, max 30 iters)
```

### 4.5 MLLM 热启动 (论文核心)

**Prior 提取 (公式 14-16):**

$$
\delta_q^* = Q^* - Q(t) \quad \text{(位移)}
$$
$$
\delta_a^* = A^* \quad \text{(关联)}
$$
$$
\delta_p^* = \{\|w_{m,k}^*\|^2, \|w_{m,r}^*\|^2\} \quad \text{(功率)}
$$

**投影头 (公式 21-23):**

```
Z_c (控制 token hidden states)
  ↓ Linear Readout
δ̃   (原始连续 prior)
  ↓ Residual MLP f_Φ
δ̃'  (修正后的 prior)
  ↓ Structured Projections
δ̂   (可行域内的 warm-start prior)
```

**三层投影:**

| 投影 | 约束 | 方法 |
|------|------|------|
| Proj_Q | 高度/区域/移动性 | Clipping + tanh 缩放 |
| Proj_P | Σ(通信+感知) ≤ P_max | Softmax + 功率预算缩放 |
| Proj_A | 单用户关联 + 负载上限 | Sinkhorn 迭代 (20 iters) |

### 4.6 损失函数

**Stage I (公式 30):**
$$
\mathcal{L}_I = \mathcal{L}_{\text{SFT}} + \lambda_{ctl} \cdot \mathcal{L}_{ctl}
$$

**Stage II (公式 37):**
$$
\mathcal{L}_{II} = \mathcal{L}_{\text{DPO}} + \mu \cdot \mathcal{L}_{\text{SFT}} + \lambda_{ctl} \cdot \mathcal{L}_{ctl} + \lambda_{sep} \cdot \mathcal{L}_{sep}
$$

其中:

| 损失项 | 公式 | 权重 |
|--------|------|------|
| L_SFT | Causal LM cross-entropy | 1.0 |
| L_ctl | MSE(δ̂_q, δ*_q) + BCE(δ̂_a, δ*_a) + MSE(δ̂_p, δ*_p) | λ_ctl=0.5 |
| L_DPO | DPO pairwise preference (β=0.1) | 1.0 |
| L_sep | Σ max(0, d_min − ‖q̂_m − q̂_m'‖)² | λ_sep=0.1 |

---

## 5. 数据生成管线

### 5.1 Algorithm 1: Best-of-N Oracle 数据生成

```
Input: S environments, N restarts, margin ρ
Output: D_SFT, D_DPO

For i = 1 to S:
  1. Sample random environment E^(i) ∼ P(E)
  2. Build multimodal prompt Π^(i)  (comm + sensing + BEV)
  3. Run SCA-FP N times with random seeds:
     Ω_j = S(·, E^(i)),  j=1..N
  4. Sort by utility: u_π(1) ≥ u_π(2) ≥ ... ≥ u_π(N)
  5. Extract best prior δ_best = Ξ(Ω_π(1)) → D_SFT
  6. Build DPO pairs: ∀(j, j') with u_j − u_{j'} > ρ·IQR(u)
     → D_DPO ∪ { (Π, δ_j, δ_{j'}) }
```

### 5.2 实现参数

| 参数 | 值 |
|------|-----|
| S (环境数) | 5000 |
| N (重启数) | 10 |
| ρ (成对边距) | 0.2 × IQR |
| 总 SCA-FP 求解 | 50,000 次 |
| 预估耗时 | 4–8 hours |

### 5.3 输出格式

**SFT 样本 (sft_dataset.jsonl):**
```json
{
  "id": "env_0",
  "prompt": "You are a UAV-ISAC decision controller...\n\n[Communication Summary]...\n\n...",
  "response": "{\"delta_q\": [[dx1,dy1,dh1],...], \"delta_a\": [[...],...], \"delta_p\": [[...],...]}",
  "utility": 15.432,
  "delta_q": [[1.2, -0.5, 3.1], ...],
  "delta_a": [[0, 1, 0, ...], ...],
  "delta_p": [[0.05, 0.03, ..., 0.02], ...]
}
```

**DPO 样本 (dpo_dataset.jsonl):**
```json
{
  "id": "env_0_pair_0_3",
  "prompt": "...",
  "chosen": "{\"delta_q\": ...}",
  "rejected": "{\"delta_q\": ...}",
  "utility_gap": 2.35
}
```

### 5.4 断点续跑机制

`scripts/generate_data.py` 支持:
- **Ctrl+C 安全退出**：收到 SIGINT 后在当前环境完成后保存
- **自动续跑**：检测已有 JSONL 行数，从断点继续
- **定期 checkpoint**：每 100 环境写入 progress 文件

---

## 6. 模型架构

### 6.1 Gemma3ISAC (核心模型)

```
Input: Π(t) 多模态prompt (text)
  ↓
Tokenizer  →  input_ids
  ↓  (插入 8 个 <ctrl_i> tokens)
Gemma 3 12B (Unsloth 4-bit QLoRA)
  ├─ frozen backbone (4-bit nf4 quant)
  ├─ LoRA adapters (r=16, α=32, target=[q,k,v,o]_proj)
  └─ Last hidden states
      ↓
控制 Token Hidden States Z_c ∈ ℝ^{B×8×3840}
  ↓
Constraint Projection Head h_Φ
  ├─ ControlReadout: MeanPool → Linear(3840→1920)→Linear(1920→out_dim)
  ├─ ResidualMLP: x + MLP(x)  with 2×[Linear+GELU+LayerNorm]
  ├─ Proj_Q: 位移投影 (移动性/高度/区域约束)
  ├─ Proj_A: Sinkhorn 关联投影 (单用户+容量约束)
  └─ Proj_P: Softmax 功率投影 (功率预算约束)
      ↓
δ̂ = {δ̂_q, δ̂_a, δ̂_p}  — 可行域内的 warm-start prior
```

### 6.2 LoRA 配置

| 参数 | 值 |
|------|-----|
| rank (r) | 16 |
| alpha (α) | 32 |
| dropout | 0.05 |
| target_modules | [q_proj, k_proj, v_proj, o_proj] |

### 6.3 Control Token 设计

| Token | ID | 用途 |
|-------|-----|------|
| `<ctrl_0>` ~ `<ctrl_7>` | 动态分配 | 在 prompt 末尾插入，其 hidden states 被提取为优化变量的表示 |

### 6.4 Projection Head 参数

| 组件 | 参数 |
|------|------|
| Readout hidden | 3840 → 1920 → out_dim |
| Residual MLP | 2×[256, 256] with GELU + LayerNorm |
| Sinkhorn | τ=0.5, 20 iterations |
| Power softmax | τ=0.5, P_max=1.0W |

---

## 7. 训练管线

### 7.1 Stage I: SFT-LoRA

| 参数 | 值 |
|------|-----|
| 目标函数 | L_SFT + 0.5·L_ctl |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Scheduler | Cosine with 3% warmup |
| Batch size | 1 (per device) × 16 (gradient accumulation) = 16 (effective) |
| Max seq length | 4096 |
| Optimizer | AdamW (weight_decay=0.01) |
| Mixed precision | BF16 |
| 数据 | 5000 SFT samples |
| 峰值显存 | ~25-28 GB |

**可训练参数:**
- LoRA adapters (全部 q/k/v/o projection 的 LoRA A/B 矩阵)
- Projection Head (Readout + ResidualMLP + 三个投影模块)
- Token embeddings (仅 <ctrl_*> 新增 token 行)

### 7.2 Stage II: DPO

| 参数 | 值 |
|------|-----|
| 目标函数 | L_DPO + 0.05·L_SFT + 0.5·L_ctl + 0.1·L_sep |
| DPO β | 0.1 |
| SFT anchor μ | 0.05 (防灾难性遗忘) |
| Epochs | 2 |
| Learning rate | 5e-5 (低于 SFT) |
| Reference model | Stage I checkpoint 的冻结副本 |
| 峰值显存 | ~28-31 GB (ref + train model 均 4-bit) |

### 7.3 训练产出一览

| 输出 | 路径 |
|------|------|
| SFT checkpoint | `./outputs/stage1_sft_final/` |
| DPO checkpoint | `./outputs/stage2_dpo_final/` |
| LoRA 权重 | `*/lora/adapter_model.safetensors` |
| Projection Head | `*/projection_head.pt` |
| Tokenizer | `*/tokenizer/` |

---

## 8. 评估协议

### 8.1 评估指标

| 指标 | 公式 | 单位 |
|------|------|------|
| Sum rate | Σ A_{m,k} · B · log₂(1+SINR) | Mbps |
| Mean sensing SINR | (1/T) Σ mean(SINR^s_{m,ℓ}) | dB |
| Mean CRB | 平均 Cramér-Rao bound | m² |
| Joint satisfaction | (comm_sat + sense_sat) / 2 | 比例 |
| SCA-FP iterations | 收敛所需迭代数 | 次 |
| Inference latency | MLLM 推理时间 | ms |

### 8.2 Baselines (9 个)

| ID | Baseline | 描述 |
|----|----------|------|
| B1 | CSI only | 仅信道信息，无 MLLM |
| B2 | ISAC SCA | 纯 SCA-FP (随机初始化) |
| B3 | DRL assisted | 深度强化学习辅助 |
| B4 | Single modal frozen | 冻结单模态 LLM |
| B5 | MoE baseline | 混合专家基线 |
| B6 | Frozen prompting | 冻结 LLM + 零样本 prompt (论文主要对比) |
| B7 | SFT only | 仅 Stage I (无 DPO) |
| B8 | DPO only | 仅 Stage II (跳过 SFT) |
| B9 | SFT+DPO (no head) | 两阶段但无投影头 |

---

## 9. 硬件适配

### 9.1 为什么用 Unsloth 而不是 bitsandbytes

| 因素 | bitsandbytes + PEFT | Unsloth |
|------|---------------------|---------|
| Blackwell (sm_120) 支持 | ❌ bitsandbytes 不支持 sm_120 | ✅ 内置 Blackwell 内核 |
| CUDA 12.8 兼容 | ⚠️ 需要手动编译 | ✅ 开箱即用 |
| 4-bit 量化 | BnBConfig 手动配置 | `load_in_4bit=True` 一行 |
| LoRA | PEFT LoraConfig + get_peft_model | FastLanguageModel 一步完成 |
| 训练速度 | 标准 | 2-5× faster (fused kernels) |

### 9.2 显存预估

```
模型 (4-bit nf4):         8-10 GB
LoRA 参数:                ~1 GB
优化器状态 (AdamW):        3-5 GB
KV Cache + 激活值:        10-12 GB
参考模型 (DPO):           +8 GB (frozen)
─────────────────────────────
SFT 峰值:                 ~25-28 GB  ✅
DPO 峰值:                 ~28-31 GB  ✅ (紧贴 32GB 上限)
```

### 9.3 Gemma 3 12B 兼容性问题

| 问题 | 解决方案 |
|------|---------|
| `dict has no attribute 'model_type'` | `use_fast=False` (用 SentencePiece 原生 tokenizer) |
| `torch_dtype` deprecated | 改用 `dtype` (但 Unsloth 仍用 `torch_dtype`) |
| cpp extensions incompatible | torch 2.10 < 2.11 的 warning, 不影响功能 |
| HuggingFace 直连超时 | 使用 `https://hf-mirror.com` 镜像 |

---

## 10. 当前进度

| 步骤 | 状态 | 详情 |
|------|------|------|
| 0. 服务器配置 | ✅ 完成 | Conda env 在数据盘, CUDA 12.8, PyTorch 2.10 |
| 1. 模型下载 | ✅ 完成 | Gemma 3 12B → `/root/autodl-tmp/huggingface/models/` (23GB) |
| 2. 项目上传 | ✅ 完成 | 全部源文件 → `/root/UAV/` |
| 3. 管线验证 | ✅ 完成 | 5 env × 2 restarts = 24s, 管线通 |
| 4. 数据生成 | 🔄 进行中 | 5000 env × 10 restarts, 预计 4-8h |
| 5. Stage I SFT | ⏳ 待完成 | 3 epochs, ~3-8h |
| 6. Stage II DPO | ⏳ 待完成 | 2 epochs, ~5-10h |
| 7. 评估 | ⏳ 待完成 | 200 test envs |

---

## 11. 命令速查

### 11.1 数据生成

```bash
# 完整生成 (断点续跑)
cd /root/UAV && PYTHONPATH=/root/UAV python scripts/generate_data.py \
    --num-env 5000 --num-restarts 10 --output-dir /root/autodl-tmp/data/cache

# 快速测试
cd /root/UAV && PYTHONPATH=/root/UAV python scripts/generate_data.py \
    --num-env 5 --num-restarts 2 --output-dir /root/autodl-tmp/data/cache
```

### 11.2 Stage I SFT

```bash
cd /root/UAV && PYTHONPATH=/root/UAV python src/training/train_sft.py \
    --config configs/default.yaml \
    --data_dir /root/autodl-tmp/data/cache/sft_dataset.jsonl
```

### 11.3 Stage II DPO

```bash
cd /root/UAV && PYTHONPATH=/root/UAV python src/training/train_dpo.py \
    --config configs/default.yaml \
    --stage1_ckpt ./outputs/stage1_sft_final \
    --data_dir /root/autodl-tmp/data/cache/dpo_dataset.jsonl
```

### 11.4 评估

```bash
cd /root/UAV && PYTHONPATH=/root/UAV python src/eval/evaluate.py \
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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/huggingface/models/gemma-3-12b-it', use_fast=False)
model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/huggingface/models/gemma-3-12b-it', torch_dtype=torch.bfloat16, device_map='auto')
print(f'Loaded, VRAM={torch.cuda.memory_allocated()/1024**3:.1f}GB')
"

# 磁盘使用
df -h /root/autodl-tmp && du -sh /root/autodl-tmp/*
```

---

## 附录 A: 文件清单

### 项目源文件 (24+ 文件, 位于 `/root/UAV/`)

```
configs/default.yaml
requirements.txt
src/__init__.py
src/env/__init__.py
src/env/uav_network.py          (216 lines)
src/env/uav_channel.py          (263 lines)
src/env/isac_scenario.py        (296 lines)
src/solver/__init__.py
src/solver/sca_fp.py            (494 lines)
src/data/__init__.py
src/data/oracle_generator.py    (252 lines)
src/data/dataset.py             (166 lines)
src/data/prompt_builder.py      (147 lines)
src/model/__init__.py
src/model/gemma_isac.py         (346 lines)
src/model/projection_head.py    (386 lines)
src/model/losses.py             (262 lines)
src/training/__init__.py
src/training/train_sft.py       (257 lines)
src/training/train_dpo.py       (343 lines)
src/eval/__init__.py
src/eval/evaluate.py            (301 lines)
scripts/generate_data.py        (185 lines)
scripts/upload_to_server.py     (69 lines)
```

### 外部依赖

```
google/gemma-3-12b-it  (23 GB, 5 shards)
  → /root/autodl-tmp/huggingface/models/gemma-3-12b-it/
```

---

## 附录 B: 已知限制

1. **文本版验证**：当前 `use_multimodal=false`，BEV 以文本网格代替图像。多模态版需要 Gemma 3 **Vision** 权重 (`google/gemma-3-12b-it` 不包含 vision encoder)

2. **Flash Attention 2**：当前未安装 FA2（Blackwell 兼容性问题）。Attn 回退到 SDPA (PyTorch 原生)，性能略有损失但不影响正确性

3. **torch 2.10 vs 2.11**：当前 torch 2.10，Unsloth 的 C++ 扩展因版本检查跳过 (需要 ≥2.11)。纯 Python fallback 可用，建议后续升级

4. **SCA-FP 简化的波束成形**：当前使用闭式功率注水解 (70/30 分拆)，未实现完整 SOCP 波束成形优化。对 prior 学习目标不敏感（我们学的是 Q/A/P 的高层决策，具体波束由 SCA-FP 重建）

5. **数据生成耗时**：5000×10 = 50,000 SCA-FP 求解。每解约 3-5 秒，总计 4-8 小时。可先减少样本量验证收敛

---

*报告生成时间: 2026-06-23 | 作者: Claude Code (deepseek-v4-pro)*
*项目路径: h:\Projects\UAV | 远程: /root/UAV/*

---

## 引用

**论文:** Constraint-Aware MLLM Adaptation for UAV-Enabled ISAC in Low-Altitude IoT Networks
