---
type: reference
status: current
stage: all
last_updated: 2026-06-26
related: [problem_formulation, training_pipeline, hardware_adaptation]
---

# System Architecture — Module Topology & Data Flow

## 源码结构

```
src/
├── env/                    # 仿真环境
│   ├── uav_network.py      # UAVNetwork: 拓扑 + 物理参数
│   ├── channel.py          # ISACChannel: 路径损耗 + Rician 衰落
│   └── scenario.py         # ISACScenarioGenerator: 场景采样
│
├── solver/                 # SCA-FP 数值优化器
│   └── sca_fp.py           # SCAFP: 交替优化 (部署/关联/波束成形)
│
├── data/                   # 数据层
│   ├── prompt_builder.py   # 环境状态 → 文本 prompt
│   ├── oracle_generator.py # Oracle 生成 (Best-of-N + SCA-FP)
│   └── dataset.py          # SFTDataset / DPODataset (tokenization)
│
├── model/                  # MLLM 模型
│   ├── gemma_isac.py       # Gemma3ISAC: HF Gemma3 + 投影头
│   ├── projection_head.py  # Proj_Q, Proj_A, Proj_P
│   └── losses.py           # L_SFT, L_DPO, L_ctl, L_sep
│
├── training/               # 训练脚本
│   ├── train_sft.py        # Stage I: SFT
│   └── train_dpo.py        # Stage II: DPO
│
└── eval/                   # 评估
    └── evaluate.py         # 6 指标 × 9 基线

scripts/
├── generate_data.py        # 批量数据生成 (多进程)
├── validate_data.py        # 数据质量验证
├── eda_data.py             # 探索性数据分析
├── test_sft_overfit.py     # SFT 过拟合测试
└── autodl_setup.sh         # 服务器环境自动安装

configs/
└── default.yaml            # 全部超参数
```

## 数据流

### 训练数据生成

```
┌─────────────────────┐
│ ISACScenarioGenerator│  seed=base_seed*100000+sample_id
│ (env/scenario.py)    │  确定性 RNG (每个 sample 独立)
└────────┬────────────┘
         │ 环境状态: q_current, user_pos, target_pos, channel_gains
         v
┌─────────────────────┐
│ Prompt Builder       │  环境 → 文本描述 + 数值表格
│ (data/prompt_builder)│  176 个浮点数的 JSON 模板
└────────┬────────────┘
         │
         v
┌─────────────────────┐
│ SCA-FP Solver        │  Best-of-N (N=10) 重启
│ (solver/sca_fp.py)   │  → best solution (max utility)
└────────┬────────────┘
         │ q_opt, a_opt, p_opt, utility
         v
┌─────────────────────┐
│ Oracle Generator     │  提取 prior: Δq = q_opt - q_current
│ (data/oracle)        │  构造 SFT label + DPO pairs (utility gap filter)
└────────┬────────────┘
         │
         v
┌─────────────────────┐
│ Dataset              │  Tokenization (Gemma 3 SentencePiece)
│ (data/dataset.py)    │  + control token insertion
└────────┬────────────┘
         │
         v
   SFTDataset / DPODataset → DataLoader
```

### 推理/训练 Forward

```
┌──────────────┐
│ Prompt text   │  e.g. "UAV 0 is at (450, 320, 85)..."
└──────┬───────┘
       │ tokenize
       v
┌──────────────────┐
│ Gemma 3 12B       │  HF AutoModel (SDPA attention)
│ (base model)      │  LoRA adapters (r=16, α=32)
│                   │  4-bit QLoRA
└──────┬───────────┘
       │ hidden_states at <ctrl_i> positions
       v
┌──────────────────┐
│ Control States    │  control_states: (bs, 8, d_model)
│ Extraction        │  (from output_hidden_states)
└──────┬───────────┘
       │
       v
┌──────────────────┐
│ Projection Head   │  Proj_Q → ΔQ̂ (12 floats)
│ (f32, 3 modules) │  Proj_A → Â (80 ints, Sinkhorn)
│                   │  Proj_P → P̂ (84 floats, Softmax)
└──────┬───────────┘
       │
       v
  warmstart = {delta_q, association, power}
  → SCA-FP(warmstart) → optimized solution
```

## 关键接口契约

| 模块 | 输入 | 输出 |
|------|------|------|
| `ISACScenarioGenerator.sample()` | `seed: int` | `q_current, user_pos, target_pos, channel_gains` |
| `SCAFP.solve(q_current, ...)` | `q_current (4,3), user_pos (20,2), ...` | `q_opt (4,3), a_opt (4,20), p_opt (4,21), utility` |
| `OracleGenerator.generate()` | env state + solver output | `{prompt, response_json, utility, q_current}` |
| `SFTDataset.__getitem__()` | `idx` | `{input_ids, labels, attention_mask, token_type_ids, q_current}` |
| `Gemma3ISAC.forward()` | `input_ids, attention_mask, token_type_ids, q_current` | `{loss, logits, warmstart_dict}` |
| `ProjectionHead.forward()` | `control_states (bs, 8, d_model)` | `{delta_q, association, power}` |

## 模块依赖图

```
env/ ──→ data/ ──→ training/ ──→ eval/
  │        │          │
  └──→ solver/ ───→ model/
```

- `env/` 无内部依赖
- `solver/` 依赖 `env/` (信道参数)
- `data/` 依赖 `env/` + `solver/`
- `model/` 依赖 `data/` (tokenizer)
- `training/` 依赖 `model/` + `data/`
- `eval/` 依赖 `solver/` + `model/`

## 三个 SCA-FP 入口点

SCA-FP 求解器被三个位置调用，物理参数必须一致：

| 调用位置 | 用途 | 参数来源 |
|----------|------|----------|
| `generate_data.py` | Oracle 生成 | Config → solver |
| `evaluate.py` | 评估推理 | Config → solver |
| `sca_fp.py` (内部) | 求解器自身 | `__init__` |

**Trinity Alignment** (第七轮审查闭合): 三个入口点现在共享相同的 `N_r`, `carrier_freq_ghz`, `noise_power`, `wavelength`。无硬编码。

## 项目规模

- 18+ Python 源文件, ~4200 行
- 3 脚本 (~600 行) + 1 Shell 脚本
- 1 配置文件 (default.yaml)
- 已生成数据: 5000 SFT + 186,896 DPO samples
