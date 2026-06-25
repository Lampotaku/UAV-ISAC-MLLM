---
type: decision
status: accepted
stage: sft
last_updated: 2026-06-26
related: [adr_003_sdpa_canonical, adr_004_4bit_qlora_blackwell, oom_incidents]
---

# ADR-001: Unsloth Removal — Pure PyTorch Pipeline

## Status

**Accepted** (2026-06-26). 已部署到 `master` 分支。

## Context

项目最初使用 Unsloth 进行 4-bit QLoRA 加载和训练加速。但在 Stage I SFT 训练中，出现了严重问题:

1. Unsloth 强制将 Gemma 3 的 attention 覆盖为 `eager` (16-21s/step vs 预期的 2-3s/step)
2. Unsloth 的 `fast_cross_entropy_loss` (Plan B) 在 forward 时产生 CheckpointError，因为其全局 monkey-patch 使 backward recompute 产生不同数量的张量
3. 即使只 `import unsloth.kernels` 用于单个 CE kernel，monkey-patch 仍然是全局且不可逆的

**关键发现**: Unsloth 不存在"局部借用"。一经 import 就全盘劫持 transformers 底层，与 Gemma 3 的 SDPA + gradient checkpointing 不可共存。

## Decision

**彻底移除训练路径中的所有 Unsloth 引用。** 仅保留 Unsloth 的 `FastLanguageModel.from_pretrained()` 用于 4-bit 量化加载（只调用一次，不参与 forward/backward）。

训练路径使用:
- HF `AutoModel` + `peft` (`LoraConfig`, `get_peft_model`)
- 纯 PyTorch `F.cross_entropy`（无 Triton kernel）
- `attn_implementation="sdpa"`（PyTorch 原生 memory-efficient attention）
- bs 从 4 降至 1（补偿 CE fp32 梯度张量），后升至 2

## Consequences

### Positive
- 训练速度 8.4x 提升: 21s/step → 2.5s/step (bs=1) or 4.1s/step (bs=2)
- 消除 CheckpointError (Bug #5)
- 消除 Unsloth 的隐蔽 monkey-patch 风险
- 代码路径简化: 无需处理混用 Unsloth+PyTorch 的边界情况
- 向后兼容: 纯 PyTorch + PEFT 路径在任何 GPU 上都能工作

### Negative
- CE fp32 梯度张量仍然较大 (bs=2 时 ~7GB)，但通过 bs 降低控制在安全范围内
- 失去 Unsloth 的潜在未来速度优化（当它支持 Gemma 3 SDPA 时）
- 4-bit 加载时仍需要 Unsloth（无法完全消除该依赖）

## Alternatives Considered

### Plan B: Unsloth Chunked CE Only
- 尝试仅在 loss 计算中借用 Unsloth 的 Triton CE kernel
- **拒绝原因**: Unsloth import 的全局 monkey-patch 导致 CheckpointError

### Wait for Unsloth Update
- 等待 Unsloth 支持 Gemma 3 的 SDPA attention
- **拒绝原因**: 时间不确定，项目不能等待

## References
- [02_training_log/oom_incidents.md](../02_training_log/oom_incidents.md)
- [02_training_log/speed_optimization.md](../02_training_log/speed_optimization.md)
- [99_archive/deprecated_experiments/planB_unchunked_ce.md](../99_archive/deprecated_experiments/planB_unchunked_ce.md)
