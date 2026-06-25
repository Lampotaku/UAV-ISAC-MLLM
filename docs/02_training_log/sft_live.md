---
type: reference
status: current
stage: sft
last_updated: 2026-06-26
related: [status, canonical_config, oom_incidents, speed_optimization]
---

# Stage I SFT — 训练进行中

**来源**: 交接文档 #26 (handoff_07_sft_training_live.md) | **最后更新**: 2026-06-26

## 当前状态: ~80% 完成

| 指标 | 值 |
|------|-----|
| **进度** | ~1000 / 1250 steps × 3 epochs |
| **速度** | ~4.1s/micro-batch (bs=2, seq=3456) |
| **每 epoch 时间** | ~2.9h |
| **总预估时间** | ~8.7h |
| **VRAM** | 76.3GB / 95.6GB (20GB 余量) |
| **GPU 利用率** | 100% |
| **状态** | ✅ 无 OOM, 无 CheckpointError |

## 终极配置

```yaml
per_device_train_batch_size: 2      # bs=2 (从 1 升级)
gradient_accumulation_steps: 8      # 有效 batch = 16
max_seq_length: 3456                # 128 对齐, 安全包含所有样本
attn_implementation: "sdpa"         # 纯 PyTorch SDPA
bf16: true
learning_rate: 2.0e-4
lora_r: 16, lora_alpha: 32
dropout: 0.0
use_4bit: true                      # QLoRA (仅加载，非训练时 Unsloth)
gradient_checkpointing: true
num_train_epochs: 3
warmup_ratio: 0.1
```

## bs=1 → bs=2 升级分析

| 指标 | bs=1, ga=16 | bs=2, ga=8 | 变化 |
|------|-------------|-------------|------|
| micro-batch 时间 | 2.54s | 4.14s | +63% |
| 每 epoch steps | 5000 | 2500 | -50% |
| 每 epoch 时间 | 3.5h | 2.9h | **-18%** |
| CE fp32 中间张量 | ~3.5GB | ~7GB | +100% |
| 峰值 VRAM | ~48GB | ~76GB | +58% |

**结论**: bs=2 提升 epoch 吞吐 18%，VRAM 仍在安全范围内 (76/96 GB)。

## Seq Length 分析

对全部 5000 个 SFT 样本的 token 计数:
- 最短: ~3137 tokens
- 最长: ~3329 tokens
- 选择 3456 (3329 + 余量 + 128 对齐)

原以为降低 seq_len 会显著提速，但 SDPA 对 seq_len 不敏感 — 瓶颈在 CE fp32 中间张量的内存带宽。

## 四大核心优化

| # | 优化 | 贡献 |
|---|------|------|
| 1 | 移除 Unsloth → 纯 PyTorch SDPA | 21s → 2.5s/step (~8x) |
| 2 | SDPA attention | O(n²) eager → 优化的 memory-efficient attention |
| 3 | max_seq_length 3456 | ~10% 显存节省 vs 4096 |
| 4 | bs=2 + grad_accum=8 | +18% epoch 吞吐 vs bs=1 |

## DPO 配置就绪

```yaml
per_device_train_batch_size: 1      # 双模型加载 (policy + reference)
gradient_accumulation_steps: 16
dpo_beta: 0.1
dpo_mu: 0.05                        # SFT anchor
lambda_sep: 0.1
```

预估 DPO 显存: ~75 GB / 96 GB (边界安全)

## 时间线

```
2026-06-25  数据生成完成 (5k SFT + 187k DPO)
           ↓
2026-06-25  OOM #1-5 诊断与修复
           │  ├── Bug #1: HF CausalLM 隐藏状态 + fp32 logits → bypass wrapper
           │  ├── Bug #2: logits .contiguous() 拷贝 → 接受为必要代价
           │  ├── Bug #3: GQA log_softmax fp32 存储 → gradient checkpointing
           │  ├── Bug #4: F.cross_entropy fp32 梯度 → Plan B (Unsloth chunked CE)
           │  └── Bug #5: CheckpointError (Unsloth monkey-patch) → Plan A
           ↓
2026-06-26  Plan A 验证通过 (纯 PyTorch + SDPA, bs=1)
           ↓
2026-06-26  Seq length 分析 → 3456
           ↓
2026-06-26  bs 升级 → 2
           ↓
2026-06-26  SFT 训练启动 → 进行中
```

## 待观察

- [ ] 3 个 epoch 后 loss 是否收敛
- [ ] Checkpoint 保存是否正常 (每 200 步)
- [ ] DPO 阶段是否需要调整 β 或 μ
- [ ] `--resume_from_checkpoint` 功能尚未实现 (TODO)

## 服务器参考卡

```bash
# 路径
项目: /root/UAV-ISAC-MLLM
数据: /root/autodl-tmp/data/full5000
输出: /root/autodl-tmp/outputs/stage1_sft_final

# 环境
conda activate uavmllm
export TORCHINDUCTOR_FLEX_ATTENTION=0

# 监控
nvidia-smi -l 1
tensorboard --logdir outputs/stage1_sft_final/logs --port 6006

# 训练
python src/training/train_sft.py --config configs/default.yaml
```
