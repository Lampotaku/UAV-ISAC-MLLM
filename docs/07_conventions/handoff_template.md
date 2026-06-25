---
type: reference
status: current
stage: all
last_updated: 2026-06-26
---

# Handoff 规范

## 核心原则

**不再使用编号交接文档** (`NN_handoff_XX_description.md`)。

交接信息现在分布在三个位置:
1. **`docs/00_current/status.md`**: 当前状态、blocker、下一步 ← **唯一活跃的交接点**
2. **`docs/02_training_log/`**: 训练运行指标和事件
3. **`docs/06_decisions/`**: 架构决策

## 交接场景

### 场景 A: 日常状态更新

直接修改 `docs/00_current/status.md`:
- 更新进度百分比、VRAM 指标
- 添加新的 blocker
- 更新下一步计划
- **不要创建新文件**

### 场景 B: 重大事件

在 `docs/02_training_log/` 中创建事件文件:
- OOM 事件、训练崩溃、配置变更
- 命名: `{event_description}.md`
- 在 `status.md` 中添加链接

### 场景 C: 架构决策

使用 ADR 流程 (`docs/06_decisions/`):
- 复制上一个 ADR 的格式
- 分配下一个序号
- 在 `status.md` 中引用 ADR 编号

### 场景 D: 新人接手

引导他们阅读 `docs/README.md` 的快速导航 (5 个文档, 30 分钟)。

## 旧的交接文档

`docs/05_handoff/` 中的所有编号文档已迁移到 `docs/99_archive/old_handoffs/`。它们的有效信息已被提取到新的 canonical 文档中。归档版本仅作为历史参考。

## 反模式

- ❌ 创建 `27_handoff_08_xxx.md` — 使用 status.md 更新
- ❌ 在交接文档中嵌入 bug postmortem — 使用 bug registry
- ❌ 在多个交接文档中重复配置信息 — 更新 `canonical_config.md`
- ❌ 交接文档中写长篇架构讨论 — 使用 ADR
