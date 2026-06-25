---
type: reference
status: current
stage: all
last_updated: 2026-06-26
---

# Bug Postmortem 模板

## 何时使用

创建 bug postmortem 当:
- P0/P1 bug 被修复 (必须)
- P2 bug 被修复且包含重要教训 (建议)
- P3 不需要 postmortem

## 模板

```markdown
---
type: postmortem
status: resolved
severity: P0 | P1 | P2
stage: datagen | sft | dpo | eval
commits: [abc1234, def5678]
last_updated: YYYY-MM-DD
related: [other-file-slug]
---

# Bug: [简短标题]

## 症状

[用户/系统观察到的现象]

## 根因

[技术根因 — 不要只描述症状，要深入到根本机制]

## 修复

[具体的代码修改或配置变更]

## 教训

[可泛化的工程教训]

## 影响

[如果不修复会导致什么]
```

## 要求

1. **一个文件一个 bug** (或紧密关联的 bug chain)
2. **必须包含根因** — 不要只写"修复了 X"
3. **必须包含教训** — 泛化成工程原则
4. **交叉引用** — 链接到相关的 ADR、审查、或其他 bug
5. **更新 bug registry** — 在 `03_bugs/README.md` 中添加条目
6. **更新 status.md** — 如果该 bug 是当前 blocker

## 示例

见已解决的 bug 文件:
- [physical_constraint.md](../03_bugs/resolved/physical_constraint.md) — 简洁 postmortem
- [response_token_overflow.md](../03_bugs/resolved/response_token_overflow.md) — 包含迭代修复历程
- [oom_1_through_5.md](../03_bugs/resolved/oom_1_through_5.md) — Bug chain 简洁版 (完整叙事在 training_log)
