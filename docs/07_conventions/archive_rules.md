---
type: reference
status: current
stage: all
last_updated: 2026-06-26
---

# Archive Rules

## 何时归档

| 条件 | 操作 |
|------|------|
| 实验被放弃 (如 Plan B) | 移至 `deprecated_experiments/` |
| 文档被新版本取代 | 移至对应子目录 |
| 旧的数据结果 | 移至 `old_results/` |
| 旧的交接文档 | 移至 `old_handoffs/` |
| 旧的项目文档 | 移至 `old_setup_docs/` |

## 归档规则

1. **绝不删除** — 始终使用 git mv 移至 `99_archive/`
2. **保留原始文件名** — 便于追溯
3. **添加 README.md** — 每个 archive 子目录必须解释内容和归档原因
4. **交叉引用** — canonical 文档中引用归档版本: `见 [99_archive/old_results/result_v1_failed.md]`

## Archive 目录结构

```
99_archive/
├── README.md                        # 归档总览
├── deprecated_experiments/          # 失败的实验方案
│   └── planB_unchunked_ce.md       # 例: Plan B
├── old_results/                     # 被取代的数据结果
│   ├── result_v1_failed.md
│   ├── result_v2_trial.md
│   └── result_v3_trial.md
├── old_handoffs/                    # 旧版交接文档
│   └── *.md
└── old_setup_docs/                  # 旧版项目文档
    └── *.md
```

## 时效性

归档文档的信息可能已经过时。始终参考 canonical 文档作为真源:
- 当前状态: `docs/00_current/`
- 架构: `docs/01_architecture/`
- Bug: `docs/03_bugs/`
- 决策: `docs/06_decisions/`
