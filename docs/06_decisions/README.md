---
type: reference
status: current
stage: all
last_updated: 2026-06-26
---

# Architecture Decision Records (ADR)

## 索引

| ADR | 标题 | 状态 | 日期 |
|-----|------|------|------|
| [001](adr_001_unsloth_removal.md) | Unsloth Removal — Pure PyTorch Pipeline | Accepted | 2026-06-26 |
| [002](adr_002_dpo_independent_ref.md) | DPO Reference Model Independent Loading | Accepted | 2026-06-23 |
| [003](adr_003_sdpa_canonical.md) | SDPA as Canonical Attention | Accepted | 2026-06-26 |
| [004](adr_004_4bit_qlora_blackwell.md) | bf16 Full Precision on RTX PRO 6000 (No Quantization) | Accepted | 2026-06-23 |
| [005](adr_005_control_token_mechanism.md) | Control Token Design | Accepted | 2026-06-23 |
| [006](adr_006_data_regeneration.md) | Data Regeneration + DPO Strategy | Accepted | 2026-06-29 |

## 格式

所有 ADR 使用标准格式:
- **Title**: 决策简述
- **Status**: Proposed / Accepted / Deprecated / Superseded
- **Context**: 为什么需要这个决策
- **Decision**: 我们选择了什么
- **Consequences**: 正面和负面的后果

## 如何提出新 ADR

1. 复制 `adr_NNN_template.md` 模式，分配下一个序号
2. 填写所有章节
3. 更新本 README 的索引表
4. 在 PR 中引用 ADR 编号

决策被取代时: 将状态改为 `Superseded by ADR-NNN`，保留原文件。
