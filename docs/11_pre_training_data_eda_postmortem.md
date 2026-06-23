# 第十一号文档 — 训练前数据 EDA 双 P0 事后分析

> 发现时间: 2026-06-24 | 发现阶段: 训练前数据 EDA | 严重级别: P0 × 2
> 状态: ✅ 已修复，待重新生成 | 影响: 不修复直接训练 = 3.5h 数据 + ~10h GPU 训练全部作废
> Commit: `8daddac`

---

## 目录

1. [一句话概述](#一句话概述)
2. [背景](#背景)
3. [Bug #1: 环境多样性崩溃](#bug-1-环境多样性崩溃)
4. [Bug #2: Response JSON 截断](#bug-2-response-json-截断)
5. [EDA 脚本设计](#eda-脚本设计)
6. [文件变更清单](#文件变更清单)
7. [经验教训](#经验教训)
8. [附录: 验证结果对比](#附录-验证结果对比)

---

## 一句话概述

**在 5000 环境、3.54 小时的数据生成完成后，EDA（探索性数据分析）检查发现两个 P0 级问题：(1) 多进程 pickle 导致所有 5000 个环境的 UAV 初始位置完全一致——环境多样性为零；(2) 所有 5000 条 SFT response JSON (~890 tokens) 全部超出 512-token 预算，训练时将被截断。两个 Bug 都源于"数据生成管线已验证通过"与"训练管线可消费"之间的 gap。**

---

## 背景

### 数据生成完成后的信心

5000 环境数据生成顺利完成：

```
Done in 12736.8s (3.54h)
  SFT: 5000  |  DPO: 193831
  Files: /root/autodl-tmp/data/full5000/sft_dataset.jsonl, /root/autodl-tmp/data/full5000/dpo_dataset.jsonl
```

`validate_data.py` 验证通过 — **0 issues, all clean**：

```
SFT Samples: 5000
  δ_q 3D位移 (‖Δq‖₂):  mean=15.0m  [14.7, 15.0]  (上限=15m)
✅ 数据质量正常 — 可以继续训练
```

从 `validate_data.py` 的视角，数据完美。物理约束满足，格式完整，零解析错误。

### 朋友的专业提醒

> "在把宝贵的 GPU 算力烧在 SFT 训练之前，做一次彻底的数据探伤（Data EDA）是极其必要的。虽然你之前用 validate_data.py 验证了 3D 位移没有越界，但这只是物理合法性的检查。在大模型炼丹前，我们还需要进行逻辑与格式的合法性验证。"

这个提醒直接触发了 EDA 脚本的编写，并随后发现了 `validate_data.py` 盲区中的两个灾难性缺陷。

---

## Bug #1: 环境多样性崩溃

### 现象

EDA Section 3.5 报告了令人震惊的结果：

```
── 3.5 UAV Initial Position Distribution ──
UAV0: x∈[464,464] y∈[766,766] h∈[85,85]m
UAV1: x∈[339,339] y∈[270,270] h∈[105,105]m
UAV2: x∈[379,379] y∈[899,899] h∈[176,176]m
UAV3: x∈[478,478] y∈[786,786] h∈[274,274]m
```

**5000 个环境的 UAV 初始位置完全相同**。min == max，零方差。

这不仅影响了初始位置 — `UAVNetwork` 使用相同 seed 也产生完全相同的用户位置、目标位置和初始关联矩阵。唯一的差异来自信道小尺度衰落的随机性（产生 612.97 vs 611.25 vs 612.90 的效用差异），但这只是噪声，不是结构多样性。

### 影响

| 维度 | 后果 |
|------|------|
| 训练 | 模型只学到从一种初始状态出发的 prior，无法泛化到任何其他配置 |
| 推理 | UAV 从不同位置出发时，模型输出无意义的位移预测 |
| δ_q 方向 | 43.6% 偏向 NE，因为固定的相对几何（目标集中在该方向） |
| 评估 | 9 个基线比较全部失准，因为 SFT/DPO 模型 == "单环境过拟合" |

### 根因分析

Bug 位于 `src/env/isac_scenario.py` 第 83-98 行。

**原始代码**:

```python
def __init__(self, ..., seed=None):
    ...
    self.rng = np.random.RandomState(seed)   # line 83

def sample(self, sample_id: int) -> EnvironmentSample:
    network = UAVNetwork(
        ...
        seed=self.rng.randint(0, 2**31 - 1),   # line 98
    )
    ...
    channel_gains[m, k] = self.channel.channel_gain(
        uav_pos, user_pos, rng=self.rng          # line 109
    )
```

**问题链条**:

```
1. 主进程: scenario_gen = ISACScenarioGenerator(seed=42)
   → self.rng = RandomState(42)

2. generate_data.py 循环:
   for batch_start in range(0, 5000, 100):
       futures = executor.submit(generator._process_one_environment, i)
       # ↑ ALL submissions happen in main thread
       # ↑ generator.scenario_gen.rng state NEVER mutates in main thread

3. ProcessPoolExecutor pickles generator 为每个 task:
   → ALL tasks get THE SAME pickled RNG state (RandomState(42), never advanced)

4. 每个 worker 的第一个 sample() 调用:
   → self.rng.randint(0, 2**31-1) → 返回相同的值 (state 0 的第一次高级调用)
   → UAVNetwork(seed=same_value) → 完全相同的网络拓扑
   → channel_gain(rng=self.rng) → 相同的信道采样序列
```

**核心机制**: Python `ProcessPoolExecutor` 在 `submit()` 时立即 pickle 参数。主进程在循环中连续 submit 100 个 task，**从未对 `generator` 做任何变异**。因此所有 task 都收到状态完全相同的 `generator` 副本。

即使 worker 内部的 RNG 状态会随 `sample()` 调用推进，但**每个 worker 的初始状态互为精确克隆**，因此：
- Worker 0 的 env 0 == Worker 1 的 env 0 == Worker N 的 env 0
- Worker 0 的 env 1 == Worker 1 的 env 1 (推进一步后仍同步)

5000 环境 = 70 workers × ~72 envs/worker = 实际上只有 ~72 种不同的环境状态（每个 worker 内部的序列）。但由于 `sample_id` 范围 [0, 4999] 和 RNG `randint(0, 2³¹-1)` 碰撞概率极低，更精确的结论是：**所有 worker 的第一批环境全等，之后每个 worker 内部产生独立序列，但跨 worker 存在大量重复**。

更关键的是：EDA 显示的零方差意味着甚至是同一个 worker 内部的不同 env 也可能因 RNG 序列短而重复。验证这个假设需要检查 `UAVNetwork.__init__` 的 `seed` → `RandomState(seed)` 是否有碰撞。

**实际确认**: 所有 5000 环境位置 min==max，说明整个生成过程产生了完全相同的位置。这与 "worker 内部有 ~72 种不同状态" 矛盾，除非 `randint()` 调用返回了相同值。最可能的解释是：pickle 的 RNG state 在主进程从未被触及，所有 worker 得到的是 **创建后从未调用过的** `RandomState(42)`，第一条 `randint()` 总是返回相同值。

### 修复

将环境随机性从"共享可变 RNG"改为"`sample_id` 驱动的确定性独立 RNG"：

```python
def __init__(self, ..., seed=None):
    ...
    self.base_seed = seed if seed is not None else 0    # 保存基种
    self.rng = np.random.RandomState(seed)              # 保留（向后兼容）

def sample(self, sample_id: int) -> EnvironmentSample:
    # 每个 sample_id → 唯一的确定性的独立 RNG
    sample_rng = np.random.RandomState(
        self.base_seed * 100000 + sample_id
    )

    network = UAVNetwork(
        ...
        seed=int(sample_rng.randint(0, 2**31 - 1)),
    )
    ...
    channel_gains[m, k] = self.channel.channel_gain(
        uav_pos, user_pos, rng=sample_rng
    )
```

**设计原则**:
1. **确定性**: `sample_id=1559` 无论在哪个 worker、何时执行，环境完全相同 — 可复现
2. **多样性**: `sample_id=0` ≠ `sample_id=1` ≠ ... ≠ `sample_id=4999` — 5000 种独立拓扑
3. **隔离性**: 不依赖任何共享可变状态，不受 multiprocessing pickle/fork 语义影响
4. **向后兼容**: `self.rng` 保留但不用于环境采样（用于 `_build_*` 等非关键路径的兼容性）

### 为什么 `validate_data.py` 没拦住

`validate_data.py` 检查的是**物理合法性** — "δ_q 是否 ≤ 15m?" — 而非**统计多样性** — "5000 个环境是否真的不同？"。

它的设计假设环境生成器是正确的，只验证优化器输出不违反约束。检测环境崩溃需要统计视角（min/max/分布），这恰好是 EDA 的职责。

---

## Bug #2: Response JSON 截断

### 现象

EDA Section 1 报告：

```
── Token Length Summary ──
Prompt tokens:
  mean=1345  min=1345  max=1345     budget=3584  ← OK
Response tokens:
  mean=891   min=878   max=897      budget=512   ← ALL EXCEED BUDGET
SECTION 1 ISSUES: 5000 responses will be truncated
```

### 影响

`dataset.py` 中的 tokenization 逻辑：

```python
# 原始代码
prompt_enc = self.tokenizer(prompt, truncation=True,
                             max_length=self.max_length - 512)
resp_enc = self.tokenizer(response, truncation=True, max_length=512)
```

Response 被硬截断至 512 tokens。SFT 训练的 labels 只计算 response 部分 — 意味着 **模型永远看不到 δ_a (80 个值) 和 δ_p (84 个值) 的后半部分**。

具体来说，512 tokens 大致覆盖：
- `{"delta_q": [[...4行×3列...]],` 
- `"delta_a": [[...前 6-8 行...` 
- 然后在矩阵中间截断

模型学到的输出格式：
```
{"delta_q": [...], "delta_a": [[0,1,0,1,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, ← TRUNCATED
```

> ⚠️ **注**: 准确地说是 ~890 估算 token，实际取决于 Gemma 3 SentencePiece tokenizer 对浮点数的分词粒度。在最坏情况下（每数字 1-2 token），实际 token 数可能更高。即使在最好情况下（每行 1 token），176 个浮点数 + JSON 结构也远超 512。截断是确定性的。

### 根因分析

`dataset.py` 中硬编码了 512 作为 response budget：

```python
# SFTDataset.__getitem__ (line 43-44)
prompt_enc = self.tokenizer(prompt, truncation=True,
                             max_length=self.max_length - 512)  # ← 4096 - 512 = 3584
resp_enc = self.tokenizer(response, truncation=True, max_length=512)

# DPODataset._encode_pair (line 113-115)  
prompt_enc = self.tokenizer(prompt, truncation=True,
                             max_length=self.max_length - 512)
resp_enc = self.tokenizer(response, truncation=True, max_length=512)
```

512 是典型的短回答预算（适合对话系统的一两句话），但对于包含 176 个浮点数 + JSON 结构的结构化输出，严重不足。

Response JSON 的结构：
```
delta_q:  M×3 = 4×3   = 12 个浮点数
delta_a:  M×K = 4×20  = 80 个浮点数
delta_p:  M×(K+1) = 4×21 = 84 个浮点数
──────────────────────────────────
           合计          = 176 个浮点数 + JSON 语法字符
```

### 修复

将 response budget 从 512 提升至 1024，相应缩小 prompt budget：

```python
# 修复后
prompt_enc = self.tokenizer(prompt, truncation=True,
                             max_length=self.max_length - 1024)  # 4096 - 1024 = 3072
resp_enc = self.tokenizer(response, truncation=True, max_length=1024)
```

**空间核算**:

| 组件 | 修复前 | 修复后 | 实际用量 |
|------|--------|--------|---------|
| Prompt | 3584 | 3072 | ~1345 ✓ |
| Control tokens | 8 | 8 | 8 |
| Response | 512 ✗ | 1024 | ~890 ✓ |
| **总计** | 4096 | 4096 | ~2243 |

Prompt 实际 ~1345 tokens，在 3072 的预算内余量充足。Response ~890 tokens 在 1024 预算内有 ~15% 的安全余量。

### 为什么 `validate_data.py` 没拦住

`validate_data.py` 检查数据文件的**内容正确性**（JSON 可解析、字段完整、数值合法），不检查**训练管线的消费正确性**（tokenization 是否截断）。这是格式验证与管线验证之间的 gap — 数据"对了"不代表训练"能看到"。

---

## EDA 脚本设计

EDA 脚本 (`scripts/eda_data.py`) 设计为三个独立 section，互补 `validate_data.py`：

| Section | 检查维度 | `validate_data.py` 盲区 |
|---------|---------|----------------------|
| **1. 格式 & Token 长度** | 打印完整样本、估算 token 数、检查截断 | 不知道 tokenizer budget |
| **2. 物理常识 3D** | 随机采样场景、ASCII 俯视图、高度剖面、功率预算 | 只做聚合统计，不做个案可视化 |
| **3. 多样性 & 崩溃** | 位移方向分布（风玫瑰图）、功率分布直方图、关联矩阵约束、位置覆盖 | 不检查跨环境的统计分布 |

**关键设计选择**:

1. **纯 CPU**: 不加载模型或 tokenizer。Token 估算用启发式 `chars/4 + digits/2.5`，在数据级探测截断问题
2. **快速**: Section 1 只 JSON-parse 5000 条 SFT + spot-check 5 条 DPO（之前卡住就是因为 parse 了 19 万条 DPO）
3. **人类可读**: ASCII 俯视图 + 风玫瑰图 + 直方图，肉眼抓逻辑 bug 比代码快

---

## 文件变更清单

| 文件 | 变更类型 | 关键改动 |
|------|---------|---------|
| `src/env/isac_scenario.py` | Bug 修复 | `__init__`: 保存 `base_seed`；`sample()`: 创建 per-sample `sample_rng = RandomState(base_seed*100000+sample_id)`，network 和 channel gain 均使用 `sample_rng` |
| `src/data/dataset.py` | Bug 修复 | `SFTDataset.__getitem__`: response budget 512→1024, prompt budget `max_length-512`→`max_length-1024`；`DPODataset._encode_pair`: 同上 |
| `scripts/eda_data.py` | 新增工具 | 三 section 训练前数据体检：格式/长度/多样性/可视化 |

**Commit**:
```
8daddac fix: P0 bugs — (1) per-sample RNG for env diversity (2) response budget 512→1024 to prevent JSON truncation
```

---

## 经验教训

### 1. "验证通过" ≠ "数据可用"

`validate_data.py` 验证的是**物理正确性**和**格式完整性**。EDA 检查的是**统计多样性**和**管线兼容性**。两者互补，缺一不可。

```
validate_data.py:  "这个数据合法吗？"    ✅ → δ_q ≤ 15m, JSON 可解析
EDA:              "这个数据能训练吗？"    ✗ → 5000 环境全等, response 全截断
```

### 2. 多进程 pickle 是确定性的，RNG 共享不是

`ProcessPoolExecutor` 在 `submit()` 时 pickle 参数。如果参数中包含可变 RNG 且主进程不推进它，所有 worker 拿到相同状态。这不是 bug — 是 pickle 语义的固有特性。

**原则**: 在并行数据生成中，**不要依赖跨进程的 RNG 状态推进**。使用 `sample_id` 或 `(base_seed, sample_id)` 元组作为确定性 seed。

### 3. Token budget 必须在数据生成前验证，而非训练时才发现

理想流程：
```
生成 1 个环境 → tokenize → 确认在 budget 内 → 生成全部 5000 环境
```

当前的 token 估算在 EDA 阶段才做，但此时数据已生成完毕。应该在 `oracle_generator.py` 的 `format_oracle_response()` 或 smoke test 阶段就验证 response 长度。

**后续改进**: 在 `generate_data.py` 的 smoke test 阶段添加 token 长度检查，或在数据输出时携带 token 计数元数据。

### 4. EDA 的"防呆"价值

3.54 小时的 CPU 生成 + 10 分钟的 EDA 检查 vs. 3.54h 生成 + ~10h GPU 训练 + 发现模型不收敛。EDA 用 **0.5% 的追加成本** 防止了 **100% 的资源浪费**。

### 5. 人类直觉在数据审查中的不可替代性

EDA Section 2 的 ASCII 俯视图和高度剖面设计，是为了让研究者在 30 秒内用肉眼判断 "UAV 是不是朝用户/目标移动了"。这种模式识别能力是纯数值统计无法替代的 — 就像 Section 3.5 的 `min==max` 一样，一眼就能发现异常。

---

## 附录: 验证结果对比

### 修复前（当前 full5000 数据 — 不可用于训练）

```
Section 1: 5000 responses truncated ← P0
Section 3: Zero position diversity    ← P0
  UAV0: x∈[464,464] y∈[766,766] h∈[85,85]m
  Direction bias: NE 43.6%
  Power: all OK (1.0W, within budget)
FINAL VERDICT: ✗ DO NOT TRAIN
```

### 修复后预期（需重新生成后验证）

```
Section 1: 0 responses truncated     ← FIXED (1024 budget > ~890 actual)
Section 3: Diverse positions          ← FIXED (per-sample RNG)
  UAV0: x∈[100,900] y∈[100,900] h∈[70,280]m   (均匀分布)
  Direction: ~12.5% per octant (均匀分布)
  Power: all OK (1.0W, within budget)
FINAL VERDICT: ✅ Ready for training
```

---

> **相关文档**: 
> - [10_physical_constraint_bug_postmortem.md](10_physical_constraint_bug_postmortem.md) — 上一个 P0（物理约束穿透）
> - [09_handoff_document.md](09_handoff_document.md) — 完整项目交接
>
> **相关 Commits**: `8daddac`
