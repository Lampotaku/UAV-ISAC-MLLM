# UAV-ISAC-MLLM 完整交接文档

> **给新对话的完整上下文 —— 读完本文档即可了解一切**
> 最后更新: 2026-06-23 | 状态: 上线就绪，待跑数据生成

---

## 目录

1. [一句话概括](#一句话概括)
2. [当前状态：做了什么，接下来要做什么](#当前状态做了什么接下来要做什么)
3. [项目架构总览](#项目架构总览)
4. [完整数据流](#完整数据流)
5. [服务器环境](#服务器环境)
6. [逐步操作指南](#逐步操作指南)
7. [关键配置参数](#关键配置参数)
8. [所有文件清单与职责](#所有文件清单与职责)
9. [已知问题与注意事项](#已知问题与注意事项)
10. [命令速查表](#命令速查表)

---

## 一句话概括

**用 Gemma 3 12B 大模型（LoRA + 约束投影头）为无人机通信感知一体化（UAV-ISAC）的数值优化器（SCA-FP）提供智能热启动，大幅减少优化迭代次数。**

论文: *Constraint-Aware MLLM Adaptation for UAV-Enabled ISAC in Low-Altitude IoT Networks*

---

## 当前状态：做了什么，接下来要做什么

### ✅ 已完成

| 项目 | 状态 |
|------|------|
| 全部源代码 (18 个 Python 文件, ~4200 行) | ✅ 完成 |
| 全局配置文件 `configs/default.yaml` | ✅ 完成，路径已适配 AutoDL |
| 7 轮代码审查 | ✅ 25+ 个问题全部闭合 |
| GitHub 仓库设为私有 | ✅ `Lampotaku/UAV-ISAC-MLLM` (private) |
| SSH Key 配置 | ✅ `autoDL-5090-UAV` 已添加到 GitHub |
| 服务器环境搭建脚本 `scripts/autodl_setup.sh` | ✅ 完成 |
| 数据质量验证脚本 `scripts/validate_data.py` | ✅ 完成，支持 `--watch` 模式 |
| 所有 AutoDL 路径适配 | ✅ `/root/autodl-tmp/outputs`, `/root/autodl-tmp/data/cache`, etc. |

### ⏳ 待执行（按顺序）

1. **服务器首次设置**: `git clone` → `bash scripts/autodl_setup.sh` → `huggingface-cli login`
2. **烟雾测试**: 生成 5 个环境验证管线 → `python scripts/validate_data.py` 检查数据质量
3. **全量数据生成**: 5000 环境 × 10 重启 = 50,000 次 SCA-FP 求解 (预计 4-8 小时)
4. **Stage I SFT 训练**: 3 epochs (预计 3-8 小时)
5. **Stage II DPO 训练**: 2 epochs (预计 5-10 小时)
6. **评估**: 200 测试环境，9 基线对比

---

## 项目架构总览

### 核心思路

```
传统方法:  随机初始化 → SCA-FP 迭代 30 轮 → 收敛
本论文:    MLLM 推理 → warm-start prior → SCA-FP 迭代 5-10 轮 → 收敛
                                         ↑ 加速 2-5×
```

MLLM 不是替代优化器，而是**学习输出接近最优解的初始猜测**。训练数据来自 SCA-FP 的大量历史求解（Best-of-N Oracle）。

### 系统分层

```
┌─────────────────────────────────────────────────────┐
│  scripts/generate_data.py     ← 数据生成入口        │
│  scripts/validate_data.py     ← 数据质量监控        │
├─────────────────────────────────────────────────────┤
│  src/env/                     ← 仿真环境层          │
│    uav_network.py             UAV/用户/目标拓扑     │
│    uav_channel.py             物理信道 (LoS/NLoS)   │
│    isac_scenario.py           场景生成 + BEV 网格   │
├─────────────────────────────────────────────────────┤
│  src/solver/sca_fp.py         ← SCA-FP 数值优化器   │
│    交替优化: 波束成形 → 部署 → 关联                 │
├─────────────────────────────────────────────────────┤
│  src/data/                    ← 数据层              │
│    prompt_builder.py          多模态 Prompt 构造    │
│    oracle_generator.py        Best-of-N 数据生成    │
│    dataset.py                 PyTorch Dataset       │
├─────────────────────────────────────────────────────┤
│  src/model/                   ← 模型层              │
│    gemma_isac.py              Gemma 3 + LoRA + Ctrl │
│    projection_head.py         约束投影头 (Q/A/P)    │
│    losses.py                  损失函数 (SFT/DPO)    │
├─────────────────────────────────────────────────────┤
│  src/training/                ← 训练层              │
│    train_sft.py               Stage I: SFT-LoRA    │
│    train_dpo.py               Stage II: DPO        │
├─────────────────────────────────────────────────────┤
│  src/eval/evaluate.py         ← 评估层              │
│    6 指标 × 200 环境 × 9 基线                       │
└─────────────────────────────────────────────────────┘
```

### 优化变量

每次决策 MLLM 输出三个量：

| 变量 | 形状 | 含义 | 物理约束 |
|------|------|------|---------|
| `delta_q` | (M, 3) | UAV 位移增量 | ‖Δq‖₂ ≤ v_max·Δt, H ∈ [50,300] |
| `delta_a` | (M, K) | 用户-UAV 关联 | 每用户一列和≈1, 每 UAV 最多 K_max 用户 |
| `delta_p` | (M, K+1) | 通信+感知功率 | 总和 ≤ P_max, 通信功率 ≥ P_min |

### 训练两阶段

**Stage I (SFT)**: 模仿学习 — 让模型学会输出最优 prior
```
L_I = L_SFT(文本生成) + 0.5 × L_ctl(控制变量 MSE/BCE)
```

**Stage II (DPO)**: 偏好学习 — 让模型区分好解和差解
```
L_II = L_DPO(偏好优化) + 0.05 × L_SFT(防遗忘) + 0.5 × L_ctl + 0.1 × L_sep(防碰撞)
```

### 关键技术点

1. **Control Token**: 8 个 `<ctrl_0>`~`<ctrl_7>` 特殊 token 追加在 prompt 末尾，从它们的 hidden states 解码出优化变量
2. **Constraint Projection Head**: 三层可微投影将 LLM 原始输出强制映射到物理可行域
   - Proj_Q: 3D 范数裁剪 + 坐标 clamping
   - Proj_A: Sinkhorn 迭代 (20 轮)
   - Proj_P: Softmax + P_min 功率重分配
3. **Unsloth 替代 bitsandbytes**: Blackwell (sm_120) 不支持 bitsandbytes，用 Unsloth 的 4-bit QLoRA
4. **Reference Model 独立加载**: DPO 训练不 deepcopy（4-bit 模型 deepcopy 会 OOM），而是重新 `from_pretrained` 加载一份

---

## 完整数据流

### 1. 数据生成 (离线)

```
generate_data.py
  │
  for i in 0..4999:    # 5000 个环境
  │
  ├─ ISACScenarioGenerator.sample(i)
  │   ├─ UAVNetwork.reset()          随机拓扑 (UAV/用户/目标)
  │   ├─ ISACChannel.channel_gain()  计算所有 M×K 链路增益
  │   └─ → EnvironmentSample          含通信摘要/感知摘要/BEV网格
  │
  ├─ build_full_prompt(env_sample)   → Prompt Π(i)
  │
  for j in 0..9:       # 每个环境跑 10 次 SCA-FP
  │
  ├─ SCAFPOptimizer.solve(env, seed=i*10+j)
  │   ├─ _optimize_beamforming()     闭式功率注水
  │   ├─ _optimize_deployment_sca()  L-BFGS-B 优化 UAV 位置
  │   ├─ _optimize_association()     Hungarian 算法
  │   └─ → SCAFPSolution (Q, A, P, utility)
  │
  ├─ 按 utility 排序 10 个解
  ├─ 最优解 → extract prior (δ_q*, δ_a*, δ_p*) → SFT 样本
  └─ 效用差 > 0.2×IQR 的对 → DPO 偏好对
      │
      └─ → sft_dataset.jsonl (~5000 条)
          dpo_dataset.jsonl (~数万对)
```

### 2. 训练 (离线)

```
train_sft.py
  SFTDataset(jsonl) → DataLoader
    → Gemma3ISAC.forward(prompt, q_current)
      → Control Token hidden states → ProjectionHead → δ̂
    → compute_stage1_total(δ̂, δ*, logits, labels)
    → backward → step

train_dpo.py
  DPODataset(jsonl) → DataLoader
    → Gemma3ISAC.forward(chosen) → δ̂_chosen, logp_chosen
    → Gemma3ISAC.forward(rejected) → δ̂_rejected, logp_rejected
    → RefModel.forward(chosen) → logp_ref_chosen  (no_grad)
    → RefModel.forward(rejected) → logp_ref_rejected (no_grad)
    → compute_stage2_total(...)
    → backward → step
```

### 3. 推理/评估 (在线)

```
evaluate.py
  for i in 0..199:    # 200 个测试环境
  │
  ├─ ISACScenarioGenerator.sample(i)
  ├─ build_full_prompt(env_sample)
  ├─ Gemma3ISAC.generate_warmstart(prompt, q_current)
  │   → δ̂ = {delta_q, delta_a, delta_p}
  │
  ├─ SCAFPOptimizer.solve(env, warm_start=δ̂)
  │   → 对比: 无 warm-start 的纯 SCA-FP
  │
  └─ 6 指标: sum_rate, sensing_sinr, joint_satisfaction,
              sca_fp_iterations, inference_latency
```

---

## 服务器环境

### AutoDL (SeetaCloud) 配置

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 5090 32GB |
| 架构 | Blackwell (sm_120) |
| CUDA | 12.8 |
| Python | 3.11 (conda env: `uavmllm`) |
| PyTorch | 2.7+ with CUDA 12.8 |
| Unsloth | 替代 bitsandbytes (Blackwell 兼容) |

### 关键路径

| 用途 | 路径 | 说明 |
|------|------|------|
| **项目代码** | `/root/UAV-ISAC-MLLM/` | git clone 到这里 |
| **数据盘根** | `/root/autodl-tmp/` | ⚠️ 系统盘只有 30GB，所有写入必须到这里 |
| **训练数据** | `/root/autodl-tmp/data/cache/` | 5000 SFT + 数万 DPO JSONL |
| **训练输出** | `/root/autodl-tmp/outputs/` | stage1_sft_final, stage2_dpo_final |
| **Checkpoints** | `/root/autodl-tmp/checkpoints/` | 训练中间保存 |
| **日志** | `/root/autodl-tmp/logs/` | TensorBoard 日志 |
| **HF 模型** | `/root/autodl-tmp/huggingface/models/` | Gemma 3 12B 缓存 |

### Git 工作流

```
本地 Windows (h:\Projects\UAV)  ← 代码编辑
        │ git push
        ▼
GitHub (Lampotaku/UAV-ISAC-MLLM, private)
        │ git clone / git pull
        ▼
AutoDL 服务器 (/root/UAV-ISAC-MLLM)  ← 只执行，不改代码
```

**没有客户端-服务器通信。** 纯 git 工作流：本地改代码 → push → 服务器 pull。

---

## 逐步操作指南

### Step 0: 服务器首次设置

```bash
# SSH 登录 AutoDL
ssh root@<你的IP> -p <端口>

# Clone 项目
cd /root
git clone git@github.com:Lampotaku/UAV-ISAC-MLLM.git
cd UAV-ISAC-MLLM

# 一键环境搭建
bash scripts/autodl_setup.sh

# 登录 HuggingFace (获取 Gemma 3 权重)
huggingface-cli login
# 输入 HF token
```

### Step 1: 烟雾测试 (验证管线)

```bash
conda activate uavmllm
cd /root/UAV-ISAC-MLLM

# 生成 5 个环境 (约 2 分钟)
python scripts/generate_data.py \
    --num-env 5 --num-restarts 10 \
    --save-every 1

# 验证数据质量
python scripts/validate_data.py \
    --data-dir /root/autodl-tmp/data/cache
```

**期望输出**: `✅ 数据质量正常 — 可以继续训练`

如果发现问题，检查详细报告中的具体条目。

### Step 2: 全量数据生成 (4-8 小时)

```bash
conda activate uavmllm
cd /root/UAV-ISAC-MLLM

# 全量 5000 环境，支持 Ctrl+C 断点续跑
python scripts/generate_data.py \
    --num-env 5000 --num-restarts 10
```

**断点续跑**: 如果中断，重新运行相同命令即可。脚本会自动检测已有的 JSONL 行数并从断点继续。

**监控**: 另开一个终端，周期性检查：
```bash
conda activate uavmllm
cd /root/UAV-ISAC-MLLM
python scripts/validate_data.py \
    --data-dir /root/autodl-tmp/data/cache \
    --watch 120   # 每 2 分钟检查一次
```

### Step 3: Stage I — SFT 训练 (3-8 小时)

```bash
conda activate uavmllm
cd /root/UAV-ISAC-MLLM

python src/training/train_sft.py \
    --config configs/default.yaml \
    --data_dir /root/autodl-tmp/data/cache/sft_dataset.jsonl
```

输出到 `/root/autodl-tmp/outputs/stage1_sft_final/`
- `lora/adapter_model.safetensors` — LoRA 权重
- `projection_head.pt` — 投影头权重
- `tokenizer/` — 扩展后的 tokenizer (含 8 个 ctrl token)

预计显存: 25-28 GB / 32 GB

### Step 4: Stage II — DPO 训练 (5-10 小时)

```bash
conda activate uavmllm
cd /root/UAV-ISAC-MLLM

python src/training/train_dpo.py \
    --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/outputs/stage1_sft_final \
    --data_dir /root/autodl-tmp/data/cache/dpo_dataset.jsonl
```

输出到 `/root/autodl-tmp/outputs/stage2_dpo_final/`

预计显存: 28-31 GB / 32 GB (含 reference model)

### Step 5: 评估

```bash
conda activate uavmllm
cd /root/UAV-ISAC-MLLM

# 仅 solver baseline (无 MLLM) — 建立对比基线
python src/eval/evaluate.py \
    --config configs/default.yaml \
    --output /root/autodl-tmp/outputs/eval_baseline.json

# 完整评估 (加载训练好的 MLLM)
python src/eval/evaluate.py \
    --config configs/default.yaml \
    --model /root/autodl-tmp/outputs/stage2_dpo_final \
    --output /root/autodl-tmp/outputs/eval_results.json
```

---

## 关键配置参数

全部在 [configs/default.yaml](configs/default.yaml) 中。

### 仿真参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `M` (num_uavs) | 4 | UAV 数量 |
| `K` (num_users) | 20 | 地面用户数 |
| `T` (num_targets) | 6 | 感知目标数 |
| `N_t`, `N_r` | 8, 8 | 天线数 |
| `f_c` | 5.8 GHz | 载频 |
| `B` | 20 MHz | 带宽 |
| `P_max` | 30 dBm (1W) | 最大发射功率 |
| `v_max` | 15 m/s | UAV 最大速度 |
| `H` | 50-300 m | UAV 高度范围 |
| Area | 1000×1000 m² | 区域大小 |

### 训练参数

| 参数 | SFT | DPO |
|------|-----|-----|
| Epochs | 3 | 2 |
| LR | 2e-4 | 5e-5 |
| Batch size | 1 | 1 |
| Grad accum | 16 | 16 |
| Max seq len | 4096 | 4096 |
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |

### 数据参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `S` | 5000 | 环境数 |
| `N` | 10 | 每环境 SCA-FP 重启次数 |
| `ρ` | 0.2 | DPO 对筛选门限系数 |

---

## 所有文件清单与职责

### 核心源码 (src/)

```
src/env/uav_network.py          UAVNetwork, UAVState, UserState, TargetState
                                → 管理 4 UAV + 20 用户 + 6 目标的拓扑
                                → 随机初始化, step(), 最近邻关联

src/env/uav_channel.py          ISACChannel
                                → 3GPP LoS/NLoS 路径损耗 (仰角依赖)
                                → Rician 小尺度衰落
                                → 通信 SINR / 感知 SINR / CRB 计算
                                → 动态波长 = c / f_c

src/env/isac_scenario.py        ISACScenarioGenerator, EnvironmentSample
                                → 生成完整环境快照
                                → 通信摘要 (per-user SINR, load, rate pressure)
                                → 感知摘要 (per-target SINR, difficulty)
                                → 10×10 BEV 文本网格

src/solver/sca_fp.py            SCAFPOptimizer, SCAFPConfig, SCAFPSolution
                                → 交替优化: 波束成形 → 部署 → 关联
                                → 闭式功率注水 + L-BFGS-B + Hungarian
                                → NaN 守卫, 动态波长, 统一路损公式

src/data/prompt_builder.py      build_full_prompt(), format_oracle_response()
                                → 构造包含系统指令+通信摘要+感知摘要+BEV 的 prompt
                                → JSON 格式输出规范

src/data/oracle_generator.py    OracleDataGenerator
                                → Best-of-N 数据生成核心
                                → 排序 10 次求解 → 最优 prior → SFT 标签
                                → IQR 门限筛选 → DPO 偏好对

src/data/dataset.py             SFTDataset, DPODataset
                                → JSONL 加载 + tokenize
                                → Control token 插入 (prompt 和 response 之间)
                                → 返回 padded tensors + control_mask + label_mask

src/model/gemma_isac.py         Gemma3ISAC (nn.Module)
                                → Unsloth 4-bit 加载 Gemma 3 12B
                                → 扩展 8 个 control token
                                → forward: 提取 control hidden states → 投影头
                                → generate_warmstart: 推理接口
                                → save_pretrained / from_pretrained

src/model/projection_head.py    ConstraintProjectionHead
                                → ControlReadout: MeanPool → Linear
                                → ResidualMLP: x + MLP(x)
                                → Proj_Q: 3D 范数裁剪 + 坐标 clamping
                                → Proj_A: Sinkhorn 迭代 (20 轮)
                                → Proj_P: Softmax + P_min 重分配

src/model/losses.py             UAVISACLosses
                                → compute_control_loss (MSE + BCE)
                                → compute_separation_penalty (防碰撞)
                                → compute_dpo_loss (含 label_smoothing)
                                → compute_sft_loss (response-only cross-entropy)
                                → Stage I / Stage II 总损失

src/training/train_sft.py       train_stage1()
                                → Accelerator + BF16 混合精度
                                → AdamW + cosine scheduler + warmup
                                → 训练循环 + checkpoint 保存

src/training/train_dpo.py       train_stage2()
                                → 加载 policy model + 独立 reference model
                                → 双前向传播 (chosen/rejected)
                                → DPO loss + SFT anchor + control loss
```

### 脚本 (scripts/)

```
scripts/generate_data.py        数据生成入口
                                → 断点续跑 (checkpoint.txt + JSONL 行数检测)
                                → Ctrl+C 优雅中断
                                → 增量追加 JSONL

scripts/validate_data.py        数据质量验证
                                → JSONL 格式检查
                                → SFT/DPO 字段完整性
                                → utility 单调性 (chosen > rejected)
                                → prior 物理合理性 (位移/功率/位置范围)
                                → --watch N 周期性监控模式

scripts/autodl_setup.sh         一键环境搭建
                                → conda env + PyTorch CUDA 12.8 + Unsloth
                                → transformers, peft, accelerate, trl, datasets
                                → flash-attn (含 Blackwell 回退)
                                → GPU 验证

scripts/upload_to_server.py     SFTP 上传 (备用，日常用 git)
```

### 配置与文档

```
configs/default.yaml            全局配置 (146 行)
docs/01_project_overview.md    初始项目概述
docs/02-07_*review*.md         6 轮代码审查报告
docs/08_pre_launch_technical_report.md  完整技术报告 (最详细)
docs/09_handoff_document.md    本文档 — 新对话快速上手
```

---

## 已知问题与注意事项

### ⚠️ 重要注意事项

1. **系统盘空间小**: AutoDL 系统盘只有 30 GB，所有输出/checkpoint/数据必须写到 `/root/autodl-tmp/`
2. **Gemma 3 权重是 gated model**: 需要先在 huggingface.co 申请授权，然后在服务器上 `huggingface-cli login`
3. **Flash Attention 可能安装失败**: 脚本有回退逻辑，失败后自动用 `sdpa`
4. **HuggingFace 连接超时**: 国内服务器可能需要 `HF_ENDPOINT=https://hf-mirror.com`
5. **tokenizer 需要 `use_fast=False`**: Gemma 3 用 SentencePiece 原生 tokenizer，fast tokenizer 会报错

### 🔧 已知技术细节

1. **DPO reference model 不能 deepcopy**: 4-bit 量化模型 deepcopy 会 OOM，现在从 checkpoint 独立加载一份
2. **DPO log-prob 用 SUM 不是 mean**: 保持联合概率 `Σ_t log π(y_t|...)` 的正确性
3. **Control token off-by-one 已修复**: 推理时取 `hidden_states[b, seq_len-num_ctrl+1 : seq_len+1]`
4. **三个入口的参数对齐**: `generate_data.py`, `train_sft.py`, `evaluate.py` 的 noise_power 计算方式一致
5. **波长为动态计算**: `wavelength = 3e8 / (f_c * 1e9)`，不硬编码 0.0517
6. **evaluate.py CRB 指标**: 当前返回 0.0 (占位)，`compute_crb()` 已实现但未接入评估循环

### ⚠️ 当前限制

1. **仅文本模式**: `use_multimodal: false`，BEV 用 10×10 文本网格替代图像
2. **单 GPU**: 代码针对单 RTX 5090 优化，未测试多 GPU
3. **`from_pretrained` 绕过 `__init__`**: 用 `__new__` + 手动赋值方式加载，新增模型属性时需同步更新

---

## 命令速查表

### 服务器初始化

```bash
ssh root@<IP> -p <端口>
cd /root && git clone git@github.com:Lampotaku/UAV-ISAC-MLLM.git && cd UAV-ISAC-MLLM
bash scripts/autodl_setup.sh
huggingface-cli login
```

### 数据生成

```bash
conda activate uavmllm && cd /root/UAV-ISAC-MLLM

# 烟雾测试 (2 分钟)
python scripts/generate_data.py --num-env 5 --num-restarts 10 --save-every 1

# 全量生成 (4-8 小时)
python scripts/generate_data.py --num-env 5000 --num-restarts 10
```

### 数据验证

```bash
# 单次检查
python scripts/validate_data.py --data-dir /root/autodl-tmp/data/cache

# 持续监控 (每 2 分钟)
python scripts/validate_data.py --data-dir /root/autodl-tmp/data/cache --watch 120
```

### 训练

```bash
# Stage I SFT
python src/training/train_sft.py --config configs/default.yaml \
    --data_dir /root/autodl-tmp/data/cache/sft_dataset.jsonl

# Stage II DPO
python src/training/train_dpo.py --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/outputs/stage1_sft_final \
    --data_dir /root/autodl-tmp/data/cache/dpo_dataset.jsonl
```

### 评估

```bash
# Baseline (纯 SCA-FP, 无 MLLM)
python src/eval/evaluate.py --config configs/default.yaml \
    --output /root/autodl-tmp/outputs/eval_baseline.json

# 完整评估 (加载 MLLM)
python src/eval/evaluate.py --config configs/default.yaml \
    --model /root/autodl-tmp/outputs/stage2_dpo_final \
    --output /root/autodl-tmp/outputs/eval_results.json
```

### 诊断

```bash
# GPU 信息
python -c "import torch; print(torch.cuda.get_device_properties(0))"

# 磁盘使用
df -h /root/autodl-tmp

# 语法检查
cd /root/UAV-ISAC-MLLM && python -m compileall -q src scripts

# 更新代码
cd /root/UAV-ISAC-MLLM && git pull origin master && pip install -e . --quiet 2>/dev/null || true
```

---

## 附录：完整执行时间线

```
Day 1:
  [SSH 登录] → git clone → autodl_setup.sh → huggingface-cli login
  [烟雾测试] 5 env × 10 restarts (~2min)
  [验证数据] validate_data.py
  [启动全量] nohup python scripts/generate_data.py --num-env 5000 --num-restarts 10 &
  [过夜] 5000 环境生成 (4-8h)

Day 2:
  [检查数据] validate_data.py
  [启动 SFT] nohup python src/training/train_sft.py ... &
  [过夜/白天] Stage I 训练 (3-8h)

Day 3:
  [检查 SFT] ls /root/autodl-tmp/outputs/stage1_sft_final/
  [启动 DPO] nohup python src/training/train_dpo.py ... &
  [过夜/白天] Stage II 训练 (5-10h)

Day 4:
  [评估] python src/eval/evaluate.py --model .../stage2_dpo_final
  [分析结果] 对比 9 基线，确认 SCA-FP 迭代数减少
```

---

*文档生成: 2026-06-23 | 项目仓库: github.com/Lampotaku/UAV-ISAC-MLLM (private)*
