````markdown
# Documentation Refactor Task

你现在是这个项目的 Documentation Architect / Technical Knowledge Manager。

你的任务不是简单整理 markdown 文件，而是：

> 重新建立整个项目的知识结构（Project Knowledge System）

---

# 项目背景

当前项目已经进入中期阶段。

当前阶段：

- SFT 炼丹
- datagen
- verification
- bug fixing
- pipeline stabilization

项目总体背景位于：

```text
docs/01_project_setup/
````

最新交接文档：

```text
26_handoff_07_sft_training_live.md
```

---

# 当前 docs 存在的问题

当前 docs 已经出现明显的信息结构失控：

* handoff 文档和 bug 文档混杂
* review 文档和阶段成果混杂
* result 文档缺乏上下文
* 命名规范混乱
* 时间线不连续
* 当前状态不明确
* 已废弃方案和当前方案混在一起
* 很多文档已经过时
* 文档之间存在重复信息
* 无法快速定位：

  * 当前 blocker
  * 当前推荐 pipeline
  * 当前训练状态
  * 当前 datagen 状态
  * 当前 verification 状态
  * 已解决问题
  * 未解决技术债

目前的 docs 更像：

> “发生过什么事件”

而不是：

> “项目当前是什么状态”

---

# 你的核心任务

你需要：

## 1. 重构整个 docs 目录结构

不是简单重命名。

而是：

* 重新设计目录结构
* 建立长期可维护的 documentation system
* 建立项目知识层次
* 建立状态型文档
* 建立技术决策追踪

---

## 2. 重新分类所有 markdown

请按：

* 文档真实用途
* 当前有效性
* 技术作用
* 项目阶段

进行重新分类。

---

## 3. 不要按文件名分类

尤其注意：

很多：

```text
handoff_xxx.md
```

实际上是：

* bug postmortem
* architecture decision
* debugging log
* failed experiment analysis

请按“真实内容”重新归类。

---

# 允许进行的大规模修改

你可以：

* 修改文件内容
* 重写文档
* 合并文档
* 拆分文档
* 新建文档
* 删除重复内容
* 重命名文件
* 重构目录
* 建立索引
* 建立时间线
* 建立 cross reference

不要保守修改。

目标是：

> 提升整个项目知识系统的可维护性。

---

# Documentation 重构目标

最终目标：

让一个新接手项目的工程师能够：

* 30 分钟理解整个项目
* 10 分钟定位当前 blocker
* 快速知道：

  * 当前推荐方案
  * 当前训练状态
  * 当前 pipeline
  * 当前有效 config
  * 当前 datagen 状态
  * 当前 verification 状态
* 快速区分：

  * 历史方案
  * 已废弃方案
  * 当前 canonical 方案
  * 已解决 bug
  * 未解决技术债

---

# 重点要求

## 不要机械保留旧结构

请从：

> “项目知识管理”

而不是：

> “文件归档”

的角度重构。

优先保证：

* 信息可检索
* 当前状态明确
* 时间线清晰
* 技术决策可追踪
* 知识不重复
* 结构长期可维护

---

# 推荐的新 docs 组织方向（可调整）

你不需要严格遵循，但请参考：

```text
docs/

00_overview/
  README.md
  current_status.md
  roadmap.md
  timeline.md

01_architecture/
  training_pipeline.md
  datagen_pipeline.md
  verification_pipeline.md
  model_architecture.md

02_training/
  sft_progress.md
  datasets.md
  configs.md
  experiments.md

03_bug_registry/
  oom_issues.md
  tokenization_issues.md
  constraint_failures.md
  resolved_bugs.md

04_reviews/
  code_review_history.md
  architecture_reviews.md

05_handoffs/
  active_handoff.md
  archived_handoffs/

06_results/
  benchmarks.md
  eval_results.md
  failed_experiments.md

99_archive/
  deprecated_designs/
  old_reviews/
```

---

# Metadata 要求（重要）

请考虑为文档增加 metadata header：

例如：

```md
---
type: postmortem
status: resolved
stage: sft
priority: high
related:
  - oom
  - datagen
  - verification
---

# OOM issue in SPD pipeline
```

用于：

* 后续检索
* 状态过滤
* 技术债管理
* timeline 构建
* future RAG / indexing

---

# 关于过时内容

如果发现某些文档：

* 已经过时
* 已失效
* 已被新方案替代

请不要机械保留。

而是：

* 标记 deprecated
* 提炼仍然有效的信息
* 合并到新的 canonical 文档
* 将废弃方案移入 archive

---

# 你最终需要输出

请最终给出：

## A. 新的 docs 目录结构

包括：

* 新目录
* 新文件
* 分类逻辑

---

## B. 文件迁移方案

列出：

* 原文件
* 新位置
* 是否重命名
* 是否拆分
* 是否合并
* 是否废弃

---

## C. 新增的重要文档

例如：

* current_status.md
* roadmap.md
* training_progress.md
* bug_registry.md
* architecture_overview.md
* active_pipeline.md

---

## D. 当前项目状态总结

请基于文档内容总结：

* 当前 SFT 状态
* 当前 datagen 状态
* 当前 blocker
* 当前主要 bug
* 当前技术债
* 当前推荐 pipeline

---

## E. Documentation 维护规范

请建立后续规范：

* 文档命名规范
* handoff 规范
* bug postmortem 规范
* experiment log 规范
* status 更新规范
* archive 规则

---

# 最后要求

开始前：

请先完整扫描整个 docs 目录。

理解：

* 文档之间关系
* 项目时间线
* 当前阶段
* 已废弃路线
* 当前 canonical 方案

之后再开始重构。

不要直接机械整理目录。
