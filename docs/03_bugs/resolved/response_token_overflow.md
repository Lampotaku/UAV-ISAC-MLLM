---
type: postmortem
status: resolved
severity: P0
stage: datagen
commits: [8b1a77c, 4f4e4e8, 223aace, c7a0685, 6c2bfac]
last_updated: 2026-06-25
related: [rng_diversity_collapse, server_runtime_errors]
---

# Bug: Response Token Overflow — BPE Float Fragmentation

**来源**: Docs 13 (response token postmortem) + Doc 12 (P0-1 verification gap)

## 症状

使用 Gemma 3 的 SentencePiece BPE tokenizer 实际计数后，176 个浮点数的 JSON 响应膨胀到 **1678 tokens**，超出 1024 预算 64%。

## 根因: 德国式三层叠加

### 第 1 层: float32 高精度存储
`np.float32(0.191)` 在 IEEE 754 中存储为 `0.19099999964237213` (17 位有效数字)。`json.dumps` 通过 Python 的 `__repr__` 忠实输出这个 17 位字符串。

### 第 2 层: BPE tokenizer 碎片化
Gemma 3 的 SentencePiece BPE tokenizer 的训练数据中没有高精度浮点数，因此 `0.1910400390625` 被拆分为 5-8 个子词 token。

### 第 3 层: JSON 缩进空白符
`json.dumps(indent=2)` 为 176 个浮点数产生 ~1400 个空白字符 (空格 + 换行)，BPE 为每个空白符消耗一个 token。

## 修复历程 (3 轮迭代)

### 第 1 轮: 精度截断 — 失败
```python
np.round(x, 4)  # → json.dumps 仍输出 float64 的 __repr__
```
`np.round` 改变的是 float32 的显示，但 `.tolist()` 转换为 Python float (float64) 后 `json.dumps` 使用 `float.__repr__`，不经过 `np.round` 的修饰。

**结果**: 1698 tokens (无改善)。

### 第 2 轮: Python round() 后处理 — 部分成功
```python
round(x, 4)  # 在 .tolist() 转换为 float64 后
```
**结果**: 1698 → 1257 tokens (-26%)，仍超出 1024 预算 23%。

### 第 3 轮: Compact JSON — 最终成功
```python
json.dumps(data, indent=None, separators=(",", ":"))
```
**结果**: 1257 → 824 tokens (-34%)。在 1024 预算内，利用率 80%。
**累计降幅**: 1678 → 824 = -51%。

## 教训

1. **`np.round` 不是精度控制** — 它修饰的是 numpy 的 `__str__`，不是序列化后的值
2. **BPE 对浮点数极不友好** — 176 个 float32 在 BPE 下可膨胀到 1400+ tokens
3. **JSON 空白符不是免费的** — `indent=2` 为 176 个浮点数增加 ~200 tokens
4. **最终配方**: `round(x, 4)` after `.tolist()` + `indent=None` + `separators=(",", ":")`

## 验证

使用 Gemma 3 tokenizer 实测:
```
Original (indent=2, full precision): 1678 tokens
Round 1  (indent=2, np.round(4)):    1698 tokens (worse!)
Round 2  (indent=2, Python round):   1257 tokens
Round 3  (compact, Python round):     824 tokens ✅
```
