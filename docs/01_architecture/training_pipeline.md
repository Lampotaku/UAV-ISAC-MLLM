---
type: reference
status: current
stage: all
last_updated: 2026-06-26
related: [system_design, problem_formulation, sft_live, oom_incidents]
---

# Training Pipeline — Stage I SFT + Stage II DPO

## 两阶段训练

```
Stage I: SFT (Supervised Fine-Tuning)
  └── 从 Best-of-N SCA-FP solutions 学习 → 预测近优热启动
       ↓
Stage II: DPO (Direct Preference Optimization)
  └── 从 utility comparison pairs 学习 → 偏好更优解
```

## Stage I: SFT

### 目标函数

```
L_SFT = L_MSE(delta_q̂, delta_q*) + L_CE(â, a*) + L_MSE(p̂, p*)
       + λ_ctl · L_ctl  (control token auxiliary loss)
```

- **L_MSE**: 连续变量 (位移、功率) 的均方误差
- **L_CE**: 离散变量 (关联) 的交叉熵
- **L_ctl**: 控制 token hidden state 的辅助预测损失 (λ_ctl = 0.5)

### 数据

- 5000 个环境，每个 Best-of-N (N=10) SCA-FP 求解
- 取 utility 最高的解作为 SFT label
- 数据格式: `{prompt_text, response_json, q_current}`

### 超参数

| 参数 | 值 |
|------|-----|
| Epochs | 3 |
| Learning rate | 2e-4 |
| LR schedule | Cosine warmup (10%) |
| Batch size (effective) | 16 (2 × 8 grad_accum) |
| Max sequence length | 3456 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Dropout | 0.0 |
| Optimizer | AdamW 8-bit |
| Precision | bf16 |
| Attention | SDPA |

### Layer-wise Learning Rate

```python
proj_params:  lr = 1e-3   # Projection head (random init, needs faster convergence)
lora_params:  lr = 2e-4   # LoRA adapters (pretrained, fine-tune gently)
```

### 梯度累积正确性 (关键!)

所有同步操作必须在 `accelerator.sync_gradients` 条件内：

```python
if accelerator.sync_gradients:
    optimizer.step()       # ← 只在真 step 时更新
    scheduler.step()       # ← 避免 cosine warmup 过早完成
    optimizer.zero_grad()  # ← 不提前清零累积梯度
    global_step += 1       # ← 避免 checkpoint 过于频繁
```

## Stage II: DPO

### 目标函数

```
L_DPO = -E[ log σ(β · (log π_θ(y_w|x) / π_ref(y_w|x) - log π_θ(y_l|x) / π_ref(y_l|x))) ]
       + λ_sft · L_SFT   (SFT anchor — 防遗忘)
       + λ_ctl · L_ctl   (Control token preservation)
       + λ_sep · L_sep   (Separation penalty)
```

### DPO Pair 构造

- 对每个环境取 Best-of-N 中 utility 最高 (chosen) 和最低 (rejected) 的解
- 过滤条件: `Δ_utility > threshold` (threshold = 0.2 × IQR, 动态)
- 186,896 pairs generated (5000 envs × ~37 high-utility pairs each)

### DPO Log-Prob 计算

**使用 SUM (非 mean)**: DPO 公式要求对 response tokens 的 log-prob 求和，保留联合概率。用 mean 会破坏 KL 散度约束。

### Reference Model

**独立加载 (不 deepcopy)**: 在 4-bit QLoRA 模型上 `deepcopy` 行为未定义且 OOM。
```python
ref_model = Gemma3ISAC.from_pretrained(checkpoint_path, ...)
```

### 超参数

| 参数 | 值 |
|------|-----|
| Epochs | 2 |
| Learning rate | 5e-5 |
| DPO beta | 0.1 |
| SFT anchor μ | 0.05 |
| Separation penalty λ_sep | 0.1 |
| Batch size (effective) | 16 (1 × 16 grad_accum) |

## 过拟合测试

在正式训练前必须通过的 5 项检查:

| # | 检查 | 标准 |
|---|------|------|
| 1 | Loss 下降 | `loss_total` 减少 >50% |
| 2 | SFT loss | `loss_sft` < 0.5 |
| 3 | Control loss | `loss_ctl` < 0.01 |
| 4 | 单调性 | 最后 50 步 loss 单调下降 |
| 5 | 数值稳定 | 无 NaN / Inf |

运行时间: ~3-5 分钟。命令:
```bash
python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000
```

## 评估协议

### 6 项指标

| 指标 | 描述 |
|------|------|
| Communication Rate | 用户和速率 (bps/Hz) |
| Sensing MI | 感知互信息 |
| Weighted Utility | λ·R_comm + (1-λ)·R_sens |
| Satisfaction Rate | 满足 QoS 的用户比例 (分母=K total) |
| SCA-FP Iterations | 收敛所需迭代次数 (衡量热启动质量) |
| CRB | Cramér-Rao Bound (感知精度) — 当前为占位符 |

### 9 条基线

B1-B9: 随机初始点、启发式方法、消融实验 (无投影头、无 control token、无 DPO 等)

### 测试集

200 个环境 (seed=42 固定)，与训练集无重叠。

## DPO 数据集验证

`validate_data.py` 的双重验证路径:
1. 主路径: 检查 `utility_chosen` + `utility_rejected` (新格式)
2. 向后兼容: 回退到 `utility_gap` (旧格式)

两种路径都验证: utility_gap > 0, chosen > rejected, JSON 格式正确。
