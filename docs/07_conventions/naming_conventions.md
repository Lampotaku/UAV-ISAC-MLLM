---
type: reference
status: current
stage: all
last_updated: 2026-06-26
---

# Documentation Naming Conventions

## 文件命名

| 文档类型 | 命名规则 | 示例 |
|----------|----------|------|
| **状态文档** | 稳定名称，原地更新 | `status.md`, `canonical_config.md` |
| **架构文档** | `snake_case.md` 描述性名称 | `problem_formulation.md`, `system_design.md` |
| **Bug 文件** | `short_kebab_description.md` | `physical_constraint.md`, `rng_diversity_collapse.md` |
| **ADR** | `adr_NNN_short_kebab.md` 顺序编号 | `adr_001_unsloth_removal.md` |
| **交接文档** | 不再使用编号交接文档 | 改用 `status.md` + `training_log/` |
| **审查报告** | `round_NN_descriptor.md` | `round_01_codex.md` |
| **归档** | 保留原始文件名 | `16_project_direction.md` |

## 目录命名

- 数字前缀表示阅读顺序: `00_current/`, `01_architecture/`
- `99_archive/` 始终最后
- 子目录用 `snake_case`: `pre_launch/`, `multiprocessing_branch/`

## 禁止的做法

- ❌ 不要创建新的编号交接文档 (`NN_handoff_XX_description.md`)
- ❌ 不要使用中文文件名
- ❌ 不要使用空格或特殊字符
- ❌ 不要将不同性质的文档混在同一目录

## Metadata Header

所有非归档文档必须包含 YAML frontmatter:

```yaml
---
type: postmortem | handoff | review | decision | result | status | reference
status: resolved | open | in_progress | deprecated | current
severity: P0 | P1 | P2 | P3 | N/A
stage: setup | datagen | sft | dpo | eval
commits: [abc1234, def5678]
related: [other-file-slug, adr-001]
last_updated: YYYY-MM-DD
---
```
