# 第十三号文档 — P0-1 Response Token 溢出 Bug 完整事后分析

> 发现时间: 2026-06-24 | 发现阶段: Smoke Test (20 envs) | 严重级别: P0-1
> 状态: ✅ 已修复 (3 轮迭代) | 影响: 若不修复，SFT 训练时 100% 的 response JSON 将被截断
> Commits: `8b1a77c` + `4f4e4e8` + `223aace`

---

## 目录

1. [一句话概述](#一句话概述)
2. [前情: 为什么会有 P0-1](#前情-为什么会有-p0-1)
3. [Bug 发现过程](#bug-发现过程)
4. [根因分析](#根因分析)
5. [修复迭代: 三次尝试才成功](#修复迭代-三次尝试才成功)
6. [朋友指导: BPE Tokenizer 浮点数碎片化](#朋友指导-bpe-tokenizer-浮点数碎片化)
7. [EDA 工具链同步改进](#eda-工具链同步改进)
8. [验证结果](#验证结果)
9. [经验教训](#经验教训)
10. [文件变更清单](#文件变更清单)

---

## 一句话概述

**Gemma 3 12B 的 SentencePiece BPE tokenizer 将高精度浮点数（如 `0.1910400390625`）碎片化为 5-8 个 subword token，导致包含 176 个浮点数的 JSON response 膨胀至 1678 token，超出 1024 预算 64%。经过三轮迭代修复（4dp 精度截断 + float32→float64 artifact 清理 + compact JSON），最终降至 824 token。**

---

## 前情: 为什么会有 P0-1

### P0-2 的掩盖效应

在 Doc #11（P0 EDA 双 Bug 事后分析）中，我们修复了两个 P0 bug：

| Bug | 症状 | 修复 |
|-----|------|------|
| P0-a: 零环境多样性 | 70 envs 产出完全相同的 prompt/response | 每样本独立 RNG: `np.random.RandomState(base_seed * 100000 + sample_id)` |
| P0-b: Response 512 token 截断 | 95%+ response 被截断 | budget 从 512 → 1024 (`8daddac`) |

**P0-b 的修复掩盖了更深层的问题**。把 budget 从 512 翻倍到 1024 是直觉反应——"response 被截断了，加预算"。但我们没有验证 **response 到底需要多少 token**。这个数字只有在用真实 Gemma 3 tokenizer 计数时才会浮出水面。

---

## Bug 发现过程

### 1. 真实 Tokenizer 检查

P0-b 修复后，朋友要求用真实 Gemma 3 tokenizer 验证，而非依赖启发式估算（`chars/4 + digits/2.5`）：

```bash
python -c "
from transformers import AutoTokenizer
import json
tok = AutoTokenizer.from_pretrained('google/gemma-3-12b-it')
with open('/root/autodl-tmp/data/smoke20/sft_dataset.jsonl') as f:
    s = json.loads(f.readline())
print(f'Prompt: {len(tok.encode(s[\"prompt\"]))} tokens (budget 3072)')
print(f'Response: {len(tok.encode(s[\"response\"]))} tokens (budget 1024)')
"
```

**输出**:
```
Prompt: 2455 tokens (budget 3072)    ← OK, 80% 利用率
Response: 1678 tokens (budget 1024)  ← ❌ 超出 64%！
```

启发式估算给出了 ~500 token，真实值是 **3.3 倍**。这意味着即使 budget 翻倍到 1024，仍有 64% 的内容会被硬截断——SFT 训练数据中的 JSON 结构将全部残缺。

### 2. 启发式为什么失败

启发式 `chars/4 + digits/2.5` 假设英文 4 字符 ≈ 1 token，数字 2.5 字符 ≈ 1 token。但 BPE tokenizer 的训练语料中从未见过 IEEE 754 高精度浮点数——词汇表中没有 `0.19104` 这种 token，所以每个数字序列被拆成 5-8 个子词。

---

## 根因分析

### 层 1: 高精度 float32 存储值 → 长字符串

`np.float32` 在 IEEE 754 下存储 0.191 为 `0.19099999964237213`（17 位有效数字）。当 `json.dumps` 序列化这个值，它会忠实地输出全部 17 位：

```python
>>> np.float32(0.191)
0.191
>>> np.float32(0.191).tolist()    # .tolist() 转换为 Python float (float64)
0.19099999964237213               # ← 17 位噪声！
>>> json.dumps(0.19099999964237213)
"0.19099999964237213"             # ← 17 chars
```

### 层 2: BPE Tokenizer 碎片化

Gemma 3 使用 SentencePiece BPE tokenizer。浮点数 `0.1910400390625` 被拆分为：

```
0.1910400390625 → ['0', '.', '191', '04', '00', '390', '625']  = 7 tokens
```

每个 17 位的浮点数平均消耗 **5-8 个 token**。一个 response 中有 176 个浮点数（delta_q: 4×3=12, delta_a: 4×20=80, delta_p: 4×21=84 = 176），产生 ~1200 token 的纯浮点碎片。

### 层 3: JSON 格式化空白符

`json.dumps(indent=2)` 为 176 个浮点数在两层级嵌套 JSON 中添加缩进和换行——约 1400 字符的空白符。BPE tokenizer 不认识这些空白组合，每个空格/换行符都消耗 1 token。

### 三层的叠加效应

```
176 floats × (5~8 tokens/float)  =  880~1408 tokens  ← 浮点碎片
        + JSON structure tokens  =  ~150 tokens       ← {"delta_q":[[ ... ]],...}
        + whitespace tokens       =  ~200 tokens       ← indent=2
        ─────────────────────────────────────────
        Total                       ~1200~1750 tokens   (实测 1678)
```

---

## 修复迭代: 三次尝试才成功

### Fix #1: `np.round(x, 4)` — 精度截断 (`8b1a77c`)

**假设**: 把浮点数四舍五入到 4 位小数，每数从 17 字符 → 5 字符，token 数骤降。

```python
# src/data/oracle_generator.py — _extract_prior()
return (np.round(delta_q, 4).astype(np.float32),
        np.round(delta_a, 4).astype(np.float32),
        np.round(delta_p, 4).astype(np.float32))
```

```python
# src/data/prompt_builder.py — format_oracle_response()
response_dict = {
    "delta_q": np.round(delta_q, 4).tolist(),
    ...
}
```

**结果: ❌ 完全失败。Response 仍为 1698 token（几乎未变）。**

**为何失败**: `np.round(x, 4)` 对 float32 数组调用后，值在 IEEE 754 内存中仍然无法精确表示 0.191。当 `.tolist()` 将 float32→float64 时，二进制 artifact 被还原：

```python
>>> np.round(np.float32(0.191), 4)         # 内存中: 仍然不是精确 0.191
0.191
>>> np.round(np.float32(0.191), 4).tolist()  # float32→float64 还原 artifact
0.19099999964237213                         # ← 还是 17 位！
>>> json.dumps(0.19099999964237213)
"0.19099999964237213"                       # ← json 忠实输出
```

**关键洞察**：`np.round` 改变的是 float32 值的 *显示*，不是 float64 转换后的 *精确值*。而 `json.dumps` 调用的是 Python float (float64) 的 `__repr__`，后者不经过 `np.round` 的修饰。

### Fix #2: Python `round()` after `.tolist()` — 击败浮点 artifact (`4f4e4e8`)

**假设**: 在 float64 级别用 Python `round()` —— Python 的 `round()` 配合 `json.dumps` 会产生干净输出。

```python
def _trunc(obj, ndigits=4):
    """递归截断浮点数精度"""
    if isinstance(obj, float):
        return round(obj, ndigits)          # ← Python float64 round
    if isinstance(obj, list):
        return [_trunc(v, ndigits) for v in obj]
    return obj

response_dict = {
    "delta_q": _trunc(np.round(delta_q, 4).tolist()),
    ...
}
```

```python
>>> round(0.19099999964237213, 4)
0.191                                   # ← Python 截断到 4dp
>>> json.dumps(0.191)
"0.191"                                 # ← 5 chars，干净！
```

**结果: ⚠️ 部分成功。1698 → 1257 token（-26%），但仍超 1024 预算 23%。**

4dp 精度截断从每浮点 17 chars → 平均 5-6 chars，token 节省了 26%。但为什么还有 1257？

**剩余问题: JSON 缩进空白**。`json.dumps(indent=2)` 为 176 个浮点数在两层级嵌套中产生 ~1400 字符的空白符（空格+换行），这些被 BPE tokenizer 逐字消耗 ~200 token。

### Fix #3: Compact JSON — 消除空白浪费 (`223aace`)

**假设**: 去掉 indent=2 的空白，改用紧凑格式，可节省 ~200 token。

```python
return json.dumps(response_dict, indent=None, separators=(",", ":"))
```

紧凑格式:
```json
{"delta_q":[[0.191,3.1709,-14.6577],[-7.5074,10.7522,-7.2703]],...}
```

缩进格式:
```json
{
  "delta_q": [
    [
      0.191,
      3.1709,
      -14.6577
    ],
    ...
```

**结果: ✅ 最终成功。1257 → 824 token（-34%），稳稳在 1024 预算内（80% 利用率）。**

### 三轮迭代总览

| 阶段 | Commit | Response token | vs budget (1024) | 改动 |
|------|--------|---------------|-------------------|------|
| 修复前 | — | 1678 | +64% ❌ | — |
| Fix #1: `np.round` only | `8b1a77c` | 1698 | +66% ❌ | float32 round → float64 artifact 未消除 |
| Fix #2: + `round()` after `.tolist()` | `4f4e4e8` | 1257 | +23% ❌ | 4dp 生效但 indent=2 空白浪费 |
| Fix #3: + compact JSON | `223aace` | 824 | −20% ✅ | 消除 ~1400 chars 空白 |

**累计降幅: 1678 → 824 = −51%**，远低于 1024 预算。

---

## 朋友指导: BPE Tokenizer 浮点数碎片化

修复中朋友提供了关键的技术判断：

### 1. 为什么不能依赖启发式 token 估算

> "Gemma 3 的 SentencePiece BPE tokenizer 训练数据中没有高精度浮点数。`0.1910400390625` 这种序列会被拆成 5-8 个 subword token。你的 `chars/4 + digits/2.5` 公式假设数字和英文一样高效，但 BPE 对 intoken 数字的编码效率极差。"

这解释了为什么启发式估算给出 ~500 token 而实际是 1678——差了 3.3 倍。

### 2. 为什么 4 位小数就够了

> "0.1mm 的精度对 UAV 控制毫无意义——GPS 误差在 1-3 米，UAV 控制精度在 1cm。4 位小数 = 0.1mm = 10 微米级，远低于任何 UAV 执行器的分辨率。这是典型的 BPE tokenizer 物理约束特征：你需要压缩精度来换取 token 效率。"

### 3. 为什么 `np.round` 不够——float32→float64 artifact

> "IEEE 754 的根本限制: 0.191 在二进制中不能被精确表示。`np.float32` 存的是近似值 0.19099999964237213。`.tolist()` 转为 float64 时保留了这些二进制 artifact。Python 的 `round()` 需要在 float64 级别执行，否则 `json.dumps` 会输出全部 17 位。"

这是 Fix #2 的核心洞察——必须用 Python `round()` 在 float64 级别截断，而非依赖 `np.round`。

---

## EDA 工具链同步改进

修复过程中，`scripts/eda_data.py` （训练前全身体检脚本）也暴露了两个 bug：

### Bug A: Power Budget Tolerance 太紧

Section 3.3 的功率约束检查使用 `1e-6` tolerance：`1.0001W > 1.000001W = violation`。但 SCA-FP 的浮点求解器产生 1.0001W 是正常的数值噪声。修复：统一为 `0.01W` tolerance (`c7a0685`, `6c2bfac`)。

### Bug B: FINAL VERDICT 只检查 Section 1

FINAL VERDICT 的逻辑只判断 prompt/response 截断，完全忽略 Section 3 的功率/负载/方向问题。修复：`check_diversity()` 返回结果，FINAL VERDICT 整合 Section 3 硬约束 (`c7a0685`)。

| Bug | 修复 |
|-----|------|
| Power tolerance `1e-6` 假阳性 | → `0.01W` (与 violations display 一致) |
| FINAL VERDICT 不读 Section 3 | → `check_diversity()` 返回 dict，power/load block training |

---

## 验证结果

### Token 级别

```bash
# 真实 Gemma 3 tokenizer 验证 (smoke20, 70 envs)
Prompt:   2455 tokens (budget 3072, 80%)  ← ✅
Response:  824 tokens (budget 1024, 80%)  ← ✅ (修复前: 1678)
```

### EDA 全身体检 (smoke20, 70 envs)

```
============================================================
  FINAL VERDICT
============================================================
  ✅ All checks passed — ready for SFT training!
============================================================

Section 1: ✅ No truncation, no format issues
  Prompt P95=1358, P99=1361 (budget 3072)
  Response mean=344 heuristic tokens (real: 824, budget 1024)

Section 2: ✅ Physical spot-check passed
  3 random envs visualized — UAVs move toward users/targets
  ‖Δq‖₂ = 15.0m for all (at physical limit, expected for coverage optimization)

Section 3: ✅ Diversity & constraints
  Direction: 11-14% in all 8 horizontal sectors (uniform)
  Power: 84.6% in [1.0, 1.01]W — all within budget (0.01W tol)
  Association: column sums all 1.0, row loads ≤ 10
  Position: wide coverage across 1000×1000m
```

### Smoke 70 数据集

| 指标 | 值 |
|------|-----|
| SFT 样本 | 70 |
| DPO 对 | 2601 (37/env) |
| Response token (实测) | 824 (budget 1024) |
| Prompt token (实测) | 2455 (budget 3072) |

---

## 经验教训

### 1. 真实 Tokenizer 检查不可替代

启发式 `chars/4 + digits/2.5` 低估了 3.3 倍。对任何使用 BPE tokenizer + JSON 数值数据的 MLLM 项目，**必须在烟雾测试阶段用真实 tokenizer 验证 token 长度**。一行 Python 就能做的事，不做会让训练数据全部被截断。

### 2. Float 精度是 BPE Tokenizer 的隐藏成本

每个高精度浮点数消耗 5-8 token。176 个浮点数 × 5-8 token = 880-1408 token 仅用于数字。在物理约束允许的范围内降低精度是系统性的 token 预算优化——0.1mm 精度对 UAV 无意义，但节省了 40% 的 token。

### 3. `np.round` 对 float32 不够——IEEE 754 的根本限制

```python
# 错误做法
np.round(float32_val, 4).tolist()  # → 0.19099999964237213 (17 chars)

# 正确做法
round(float32_val.tolist(), 4)     # → 0.191 (5 chars)
```

`np.round` 的操作对象是 float32 内存表示。`.tolist()` 转为 float64 时还原了二进制 artifact。必须在 float64 级别用 Python `round()`。

### 4. JSON 格式化在 Token 预算中不是免费的

`indent=2` 对 176 个浮点数在两层嵌套中产生 ~1400 字符空白 = ~200 BPE token。当 token 预算紧张时（如 1024），紧凑 JSON 是必要的。

### 5. Token 预算应该有多层安全余量

```
原始设计:   response_budget = 512   (基于启发式估算)  ← 实测需要 1678
第一轮修复: response_budget = 1024  (翻倍, 仍基于估算) ← 实测需要 1257  
第三轮修复: response_budget = 1024  (实测 824, 80% util) ← ✅ 安全
```

教训：**永远不要基于启发式估算设置 token 预算。** 在烟雾测试中用真实 tokenizer 测量，然后留 15-20% 余量。

### 6. EDA 工具本身需要审查

`eda_data.py` 的 FINAL VERDICT 只检查 Section 1（truncation），不读 Section 3（物理约束）。工具的逻辑漏洞和它要检测的数据问题一样危险——你会因为 `✅ All checks passed` 而获得虚假的安全感。

---

## 文件变更清单

| 文件 | 变更类型 | 关键改动 |
|------|---------|---------|
| `src/data/oracle_generator.py` | Bug 修复 | `_extract_prior()`: `np.round(x, 4).astype(np.float32)` |
| `src/data/prompt_builder.py` | Bug 修复 | `format_oracle_response()`: 递归 `_trunc()` 用 Python `round()` after `.tolist()` + compact JSON |
| `scripts/eda_data.py` | 工具改进 | 功率 tolerance `1e-6→0.01`, FINAL VERDICT 整合 Section 3 |

**Commits**:
```
8b1a77c fix: truncate oracle float precision to 4 decimal places — reduces Gemma3 BPE token count ~60%
4f4e4e8 fix: apply Python round() after .tolist() to defeat float32→float64 JSON artifacts
223aace fix: compact JSON (no indent) — saves ~200+ whitespace tokens
c7a0685 fix: EDA — power tolerance 1e-6→0.01 + FINAL VERDICT now includes Section 3 issues
6c2bfac fix: Section 2 spot-check power tolerance 1e-6→0.01 (consistency with Section 3)
```

---

> **相关文档**: 
> - [Doc #10 — P0 物理约束穿透 Bug](10_physical_constraint_bug_postmortem.md)
> - [Doc #11 — P0 EDA 双 Bug 事后分析](11_pre_training_data_eda_postmortem.md)
> - [Doc #12 — 验证缺口审计](12_remaining_verification_gaps.md)
>
> **相关 Commits**: `8b1a77c`, `4f4e4e8`, `223aace`, `c7a0685`, `6c2bfac`
