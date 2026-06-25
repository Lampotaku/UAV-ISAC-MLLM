---
type: decision
status: accepted
stage: sft
last_updated: 2026-06-26
related: [adr_001_unsloth_removal, adr_004_4bit_qlora_blackwell, speed_optimization]
---

# ADR-003: SDPA as Canonical Attention

## Status

**Accepted** (2026-06-26). 当前 canonical attention 实现。

## Context

Gemma 3 使用 5:1 sliding window + global attention 交错的 attention pattern。在 RTX PRO 6000 (Blackwell sm_120) 上，有三种 attention 实现可选:

| 实现 | 速度 | VRAM | 可行性 |
|------|------|------|--------|
| Eager | 16-21s/step | ~80 GB | ✅ 但太慢 |
| SDPA | 2.5-4.1s/step | ~76 GB | ✅ 可用 |
| FA2 | ~2s/step (预估) | ~70 GB | ❌ 无 sm_120 wheel |

## Decision

使用 PyTorch 原生 `SDPA` (`torch.nn.functional.scaled_dot_product_attention`)，通过 `attn_implementation="sdpa"` 启用。

**选择理由**:
- 8.4x faster than eager
- 内置 PyTorch，无外部依赖
- Memory-efficient (不使用 O(n²) 中间矩阵)
- 在 Gemma 3 的 5:1 sliding window pattern 上正确工作

**不选择 FA2 的理由**:
- Flash Attention 2 在 2026-06 没有预编译的 Blackwell sm_120 wheel
- 从源码编译 FA2 需要 CUDA toolchain + CUTLASS，复杂度高且不稳定

**不选择 eager 的理由**:
- 16-21s/step 太慢 (~67h for 3 epochs SFT)

## Consequences

### Positive
- 训练速度合理 (2.5-4.1s/step)
- 零外部依赖 — 标准 PyTorch 安装自带
- 跨 GPU 架构可移植 (非 Blackwell 也可用)
- 与 gradient checkpointing 兼容

### Negative
- 仍比 FA2 慢 ~2x (预估)
- 对 seq_len 不敏感 (瓶颈在 CE fp32 张量而非 attention)
- 如果未来 FA2 的 Blackwell wheel 发布，SDPA 将降级为 fallback

## Migration Path

如果 FA2 在 Blackwell 上可用:
1. 测试 `attn_implementation="flash_attention_2"`
2. 在烟雾测试中验证数值稳定性
3. 对比 SDPA 和 FA2 的速度和显存
4. 如果 FA2 稳定且快 ≥20%，更新此 ADR 状态为 Superseded

## References
- [02_training_log/speed_optimization.md](../02_training_log/speed_optimization.md)
- [01_architecture/hardware_adaptation.md](../01_architecture/hardware_adaptation.md)
