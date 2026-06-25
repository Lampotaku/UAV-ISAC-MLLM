---
type: decision
status: accepted
stage: dpo
last_updated: 2026-06-23
related: [training_pipeline, adr_001_unsloth_removal]
---

# ADR-002: DPO Reference Model Independent Loading

## Status

**Accepted** (2026-06-23). 已实现。

## Context

DPO 训练需要同时持有 policy model 和 reference model。两个 12B bf16 模型的内存管理至关重要 (reference model 占用 ~24GB)。

常见方案是 `ref_model = copy.deepcopy(policy_model)`，但在 LoRA 模型上 `deepcopy` 行为未定义且容易 OOM。

## Decision

**独立加载 reference model**，不 deepcopy:

```python
ref_model = Gemma3ISAC.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    is_reference=True,  # freeze + no LoRA
)
```

Reference model:
- 独立加载 (单独的内存分配)
- 立即 freeze (`requires_grad=False`)
- 不添加 LoRA adapters (仅推理)
- 使用相同的 bf16 权重

## Consequences

### Positive
- 避免 `deepcopy` 的 OOM 风险
- Reference 和 policy 模型的数据独立，无共享状态
- Reference 模型可以更激进地优化内存 (无 optimizer states, 无 LoRA)

### Negative
- VRAM 开销: reference 模型占用 ~24GB (bf16) 独立空间
- DPO 训练峰值 VRAM (~75GB / 96GB, 安全余量 ~21GB)
- 加载时间加倍 (两个独立的 `from_pretrained` 调用)

## References
- Doc 08 (pre-launch technical report) — DPO 内存预算
- [01_architecture/training_pipeline.md](../01_architecture/training_pipeline.md) — DPO 配置
