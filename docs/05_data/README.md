---
type: reference
status: current
stage: datagen
last_updated: 2026-06-25
related: [final_validation, rng_diversity_collapse, response_token_overflow]
---

# Data Generation Results

## 结果时间线

| Run | 文档 | 环境数 | 状态 | 关键发现 |
|-----|------|--------|------|----------|
| 1 (首次) | [99_archive/old_results/result_v1_failed.md](../99_archive/old_results/result_v1_failed.md) | 5000 SFT | ❌ 失败 | P0: 环境多样性崩溃 + 512 token 截断 |
| 2 (试运行) | [99_archive/old_results/result_v2_trial.md](../99_archive/old_results/result_v2_trial.md) | 70 SFT | ✅ 通过 | 修复后小规模验证 |
| 3 (试运行) | [99_archive/old_results/result_v3_trial.md](../99_archive/old_results/result_v3_trial.md) | 5 SFT + 196 DPO | ✅ 通过 | 通过，新增数据质量报告 |
| **4 (最终)** | [**final_validation.md**](final_validation.md) | **5000 SFT + 186,896 DPO** | **✅ 通过** | **当前数据集** |

## 当前数据 (Run 4)

| 指标 | 值 |
|------|-----|
| **SFT 样本** | 5,000 |
| **DPO 样本** | 186,896 |
| **质量** | 0 issues — all clean |
| **生成时间** | ~3.5h (70 workers) |
| **位置** | `/root/autodl-tmp/data/full5000/` |

详细验证结果: [final_validation.md](final_validation.md)

## 数据文件

```
/root/autodl-tmp/data/full5000/
├── sft_dataset.jsonl       # 5000 samples
├── dpo_dataset.jsonl       # 186,896 samples
├── sft_dataset.jsonl.lock  # (生成中)
└── dpo_dataset.jsonl.lock
```

## 验证工具

- `scripts/validate_data.py`: 物理正确性 + 格式完整性
- `scripts/eda_data.py`: 统计多样性 + 分布检查 (3-section EDA)
