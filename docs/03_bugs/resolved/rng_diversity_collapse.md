---
type: postmortem
status: resolved
severity: P0
stage: datagen
commits: [8daddac]
last_updated: 2026-06-25
related: [physical_constraint, response_token_overflow, final_validation]
---

# Twin P0 Bugs: RNG Diversity Collapse & Response Truncation

**来源**: Docs 11 (EDA postmortem) + 18 (datagen problems)

## Bug #1: 环境多样性崩溃 (RNG Pickle)

### 症状
EDA 第 3 部分显示所有 5000 个环境中 UAV 初始位置**完全相同**:
```
UAV0: (464, 766, 85) — 全部 5000 个环境
UAV1: (339, 270, 105)
UAV2: (379, 899, 176)
UAV3: (478, 786, 274)
```
方向分布: 43.6% 东北方向 (严重偏斜)。

### 根因
`ProcessPoolExecutor` 在 `submit()` 时 pickle 参数。主进程提交所有 5000 个任务而不改变 generator 的 RNG 状态，因此每个 worker 接收到 `RandomState(42)` 的相同 pickle 副本 → 所有环境从相同的 UAV 位置开始。

### 修复
将随机性从共享可变 RNG 改为由 `sample_id` 驱动的确定性独立 RNG:
```python
rng = np.random.RandomState(base_seed * 100000 + sample_id)
```
每个 sample 从不同的 seed → 不同的初始位置 → 完全的统计多样性。

## Bug #2: 响应 JSON 截断

### 症状
EDA 第 1 部分显示 5000 个 SFT 响应 tokens 为 886-897，而预算为 **512**。所有响应在训练期间被截断。

### 根因
`dataset.py` 硬编码了 512 个 token 的响应预算，但包含 176 个浮点数的 JSON 响应需要 ~890 tokens。

### 修复 (两阶段)
1. 响应预算: 512 → 1024 tokens
2. 提示预算: 4096 → 3072 (prompt) + 1024 (response) = 4096 total

后经 Doc 13 进一步修复了 BPE 浮点数碎片化问题 (1024 → 最终 824 tokens)。

## 共同主题

**"验证通过" ≠ "数据可用"**: `validate_data.py` 检查物理正确性和格式完整性，但不检查统计多样性或训练管线兼容性。EDA (探索性数据分析) 是发现这类 bug 的必要补充。

## 事后结果
所有 5000 SFT + 186,896 DPO 重新生成后通过全部验证。
