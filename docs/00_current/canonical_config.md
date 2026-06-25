---
type: reference
status: current
stage: sft
last_updated: 2026-06-26
related: [status, sft_live, adr_001_unsloth_removal, adr_003_sdpa_canonical]
---

# Canonical Configuration

**这是项目当前的 blessed 配置。所有训练必须使用此配置。**

## `configs/default.yaml` — SFT Phase

```yaml
# ── Model ──
model_name: "google/gemma-3-12b-pt"
use_4bit: false            # bf16 全精度 (96GB 显存无需量化)
lora_r: 16
lora_alpha: 32
lora_dropout: 0.0
use_gradient_checkpointing: true

# ── Training (Stage I SFT) ──
per_device_train_batch_size: 2    # ★ bs=2 (历史: bs=4→1→2)
gradient_accumulation_steps: 8    # ★ 有效 batch = 2×8 = 16
max_seq_length: 3456              # ★ 对齐 128, 安全包裹所有样本 (max 3329)
learning_rate: 2.0e-4
lr_scheduler_type: "cosine"
warmup_ratio: 0.1
num_train_epochs: 3
max_steps: -1                     # 由 epochs 决定 (1250×3=3750)
save_steps: 200
logging_steps: 10
weight_decay: 0.01
optim: "adamw_8bit"

# ── Attention ──
attn_implementation: "sdpa"       # ★ SDPA (非 eager, 非 FA2)

# ── Precision ──
bf16: true
tf32: true

# ── Loss weights ──
lambda_sft: 1.0
lambda_ctl: 0.5                  # Control token auxiliary loss
label_smoothing: 0.0

# ── Dataloader ──
dataloader_num_workers: 2
```

## DPO Phase Configuration

```yaml
# ── Training (Stage II DPO) ──
per_device_train_batch_size: 1    # DPO 双模型加载，bs 必须为 1
gradient_accumulation_steps: 16   # 有效 batch = 1×16 = 16
max_seq_length: 3456
learning_rate: 5.0e-5
num_train_epochs: 2

# ── DPO specific ──
dpo_beta: 0.1
dpo_mu: 0.05                     # SFT anchor (防遗忘)
dpo_loss_type: "sigmoid"

# ── Loss weights ──
lambda_dpo: 1.0
lambda_sft: 0.05                 # SFT anchor 强度
lambda_ctl: 0.5
lambda_sep: 0.1                  # Separation penalty
```

## 环境变量 (Blackwell RTX PRO 6000 必须)

```bash
# ★ 必须 — 否则 FlexAttention backward 会 OOM
export TORCHINDUCTOR_FLEX_ATTENTION=0

# ★ 建议 — 防止 DataLoader CPU 过载
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## 服务器执行命令

```bash
# 1. 拉取最新代码
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm

# 2. 验证数据路径
ls /root/autodl-tmp/data/full5000/sft_dataset.jsonl
ls /root/autodl-tmp/data/full5000/dpo_dataset.jsonl

# 3. 过拟合测试 (5 min, 验证训练代码)
python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000

# 4. Stage I SFT (~8.7h)
python src/training/train_sft.py --config configs/default.yaml

# 5. Stage II DPO (~5-10h, 使用 SFT 的 LoRA 权重初始化)
python src/training/train_dpo.py --config configs/default.yaml

# 6. 评估
python src/eval/evaluate.py --config configs/default.yaml
```

## VRAM 预算 (bs=2, seq=3456, SDPA, bf16 全精度)

| 组件 | 显存 |
|------|------|
| Gemma 3 12B (bf16) | ~24 GB |
| LoRA adapters (bf16) | ~1 GB |
| Activations (grad ckpt) | ~15 GB |
| CE fp32 中间张量 | ~7 GB |
| Optimizer states (8-bit AdamW) | ~8 GB |
| CUDA context + 碎片 | ~18 GB |
| 其他开销 | ~3 GB |
| **峰值总计** | **~76 GB / 96 GB** |

余量: ~20 GB (安全)

## 配置演进历史

| 版本 | bs | grad_accum | attention | Unsloth? | 速度 | 原因 |
|------|----|------------|-----------|----------|------|------|
| v0 (原始) | 4 | 4 | eager | ✅ | 21s/step | Unsloth 强制 eager |
| v1 (Plan B) | 4 | 4 | eager | ✅ (CE only) | ~18s/step | Unsloth chunked CE → Bug #5 |
| v2 (Plan A) | 1 | 16 | SDPA | ❌ | 2.5s/step | 彻底清除 Unsloth |
| **v3 (current)** | **2** | **8** | **SDPA** | **❌** | **4.1s/step** | 提升 epoch 吞吐 18% |

当前 v3 为 canonical 配置。演进详情见:
- [02_training_log/speed_optimization.md](../02_training_log/speed_optimization.md)
- [02_training_log/oom_incidents.md](../02_training_log/oom_incidents.md)
- [06_decisions/adr_001_unsloth_removal.md](../06_decisions/adr_001_unsloth_removal.md)
