---
type: reference
status: deprecated
stage: all
last_updated: 2026-06-26
---

# Archive

本目录包含已废弃、被取代或仅作历史参考的文档。

**⚠️ 此处信息可能已过时。** 当前信息请参考:
- [../00_current/](../00_current/) — 项目状态
- [../01_architecture/](../01_architecture/) — 技术参考
- [../03_bugs/](../03_bugs/) — Bug registry
- [../06_decisions/](../06_decisions/) — 架构决策

## 子目录

| 目录 | 内容 | 归档原因 |
|------|------|----------|
| `deprecated_experiments/` | Plan B (Unsloth chunked CE) | 被 Plan A 取代 |
| `old_results/` | 数据验证 v1-v3 | 被 v4 (final) 取代 |
| `old_handoffs/` | 文档 16-21 | 有效信息已提取到 canonical 文档 |
| `old_setup_docs/` | 文档 01/08/09 | 被 `01_architecture/` 取代 |

## 重构说明

2026-06-26: 整个 docs 目录按照 `docs/refactor.md` 要求进行了重构。归档前的原目录结构:

```
docs/
├── 01_project_setup/
├── 02_code_reviews/
├── 03_bug_postmortems/
├── 04_data_results/
└── 05_handoff/
```

所有原始文件已通过 `git mv` 保留历史。
