---
type: decision
status: accepted
stage: all
last_updated: 2026-06-23
related: [adr_001_unsloth_removal, hardware_adaptation]
---

# ADR-004: bf16 Full Precision on RTX PRO 6000 (No Quantization)

## Status

**Accepted** (2026-06-23, updated 2026-06-26). 使用 bf16 全精度训练，无需量化。

## Context

RTX PRO 6000 (Blackwell sm_120, 96GB) 上，标准量化工具 `bitsandbytes` 在 2026-06 不支持 sm_120。但 96GB VRAM 足以容纳 Gemma 3 12B bf16 (~24GB) + LoRA (~1GB) + activations (~22GB) + optimizer (~8GB)，峰值约 76GB，无需量化。

备选方案:
1. **bitsandbytes 4-bit**: ❌ 不支持 Blackwell sm_120
2. **Unsloth 4-bit QLoRA**: ⚠️ 支持但引入全局 monkey-patch → 与 SDPA 冲突 (见 ADR-001)
3. **GPTQ/AWQ**: ❌ 需要离线量化 + 特定 kernel，Blackwell 支持有限
4. **bf16 全精度**: ✅ 96GB 显存足够，无需量化，训练质量最优

## Decision

使用 **bf16 全精度** (`torch.bfloat16`)，不进行任何量化。

```python
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

model = AutoModel.from_pretrained(
    "google/gemma-3-12b-pt",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model = get_peft_model(model, LoraConfig(...))
```

**理由**: 96GB VRAM 消除了量化的必要性。bf16 全精度提供最佳训练质量，且完全避免 Unsloth 或 bitsandbytes 的兼容性问题。

## Consequences

### Positive
- 零量化依赖 — 纯 PyTorch + HF + PEFT
- bf16 精度高于 4-bit，训练质量更好
- 完全避免了 Unsloth 的 monkey-patch 风险
- 代码更简洁，调试更容易

### Negative
- VRAM 使用更高 (~24GB vs ~8GB for 4-bit 模型权重)
- 在 32GB 卡上不可行 (96GB 是必要条件)
- 如果未来扩展模型规模 (如 Gemma 27B)，可能需要重新引入量化

### VRAM Budget (bf16, bs=2, seq=3456)
```
模型权重 (bf16):      ~24 GB
LoRA adapters:         ~1 GB
Activations (ckpt):   ~15 GB
CE fp32 中间张量:      ~7 GB
Optimizer (8-bit):     ~8 GB
CUDA context + 其他:  ~21 GB
─────────────────────────
峰值总计:              ~76 GB / 96 GB (20 GB 余量)
```

## Migration Path

NA — bf16 全精度是本项目的 canonical 精度方案。仅在以下情况考虑重新引入量化:
1. 模型规模显著增大 (如 Gemma 27B)
2. bitsandbytes 全面支持 Blackwell sm_120

## References
- [01_architecture/hardware_adaptation.md](../01_architecture/hardware_adaptation.md)
- [06_decisions/adr_001_unsloth_removal.md](adr_001_unsloth_removal.md)
- [00_current/canonical_config.md](../00_current/canonical_config.md)
