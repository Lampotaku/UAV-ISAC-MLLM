# 第十四号文档 — feature/multiprocessing 分支一审完整事后分析

> 审查时间: 2026-06-24 | 审查阶段: 数据生成前 Code Review (高力度 8 角度) | 严重级别: P0–P2
> 状态: 🔴 待修复 | 影响: 若不修复，多进程数据生成将出现静默数据丢失 + 中断无响应 + DPO 质量未验证
> 审查范围: `feature/multiprocessing` vs `origin/master` — 14 commits, 12 files, +2672/-58

---

## 目录

1. [一句话概述](#一句话概述)
2. [审查背景与方法](#审查背景与方法)
3. [Bug 发现: 5 个正确性缺陷](#bug-发现-5-个正确性缺陷)
4. [质量发现: 5 个架构/效率问题](#质量发现-5-个架构效率问题)
5. [驳回: 2 个候选人被证伪](#驳回-2-个候选人被证伪)
6. [修复方案](#修复方案)
7. [经验教训](#经验教训)
8. [文件变更清单](#文件变更清单)

---

## 一句话概述

**对 `feature/multiprocessing` 分支进行 8 角度高力度 Code Review，发现 6 个确认/可信的正确性缺陷（含 1 个 P0 多进程续跑数据丢失、1 个 P1 中断无响应），5 个架构/效率问题（含重复 tokenization 逻辑复刻、EDA 工具自欺性 token 估算器），以及 1 个预存 Bug（DPO 质量验证被绕过）。2 个候选人被证伪。**

---

## 审查背景与方法

### 为什么需要一审

`feature/multiprocessing` 分支在 `master` 之上累积了 14 个 commit，涵盖：

| 域 | 变更 |
|---|------|
| 数据生成 | 多进程并发 + 进度报告 + 中断续跑 |
| 数据层 | `<bos>`/`<eos>` tokenization 修复 + 浮点精度截断 + compact JSON |
| 环境 | 每样本独立 RNG (修复零多样性 P0-a) |
| 求解器 | Box→Sphere 3D 移动约束 (修复物理穿透 P0) |
| 工具链 | eda_data.py (574 行新文件) + validate_data.py 改进 |

分支已准备好推送到服务器执行 5000 环境 × 10 重启 × 70 worker 的全量数据生成。但**多进程并发引入了全新的故障模式**（部分写入、中断传播、RNG fork 安全），这些在原顺序模式下不存在。一审的目标是在跑 1.5 小时的全量生成之前，用系统性 Code Review 拦截这些问题。

### 审查方法

使用 8 角度并行 Finder + 1 票验证器 (recall-biased) 的审查流水线：

| 角度 | 职责 | 发现数 |
|------|------|--------|
| A — 逐行扫描 | 正确性 Bug | 6 |
| B — 删除行为审计 | 被移除的不变量 | 3 |
| C — 跨文件追踪 | 调用链断裂 | 6 |
| D — 复用 | 重复实现 | 6 |
| E — 简化 | 不必要复杂度 | 6 |
| F — 效率 | 浪费的工作 | 6 |
| G — 深度 | 治标 vs 治本 | 6 |
| H — 约定 | CLAUDE.md 违规 | 0 |

共 39 个候选 → 去重 → 11 个进入验证 → **6 个确认/可信 + 5 个质量发现 + 2 个驳回**。

---

## Bug 发现: 6 个正确性缺陷

### Bug 1 (P0 — CONFIRMED): 多进程续跑导致静默数据丢失

- **文件**: `scripts/generate_data.py:122-123, 238-239`
- **严重级别**: P0 — 影响数据完整性
- **类型**: 并发安全 + 恢复逻辑错误

**根因**: 续跑位置 `start_env` 通过 `_count_existing(sft_path)` (统计 JSONL 行数) 推导。在多进程模式下，checkpoint 仅在**整批完成**后写入 (line 238-239)，但每个 worker 完成时立即通过 `_incremental_append` 写入 JSONL 行。若进程在批次中间被 kill：

```
批次 [0, 100) 提交 100 个 future → 60 个完成并写入 JSONL → 进程被 OOM kill
→ checkpoint.txt 未写入（仍指向 0）
→ sft_dataset.jsonl 有 60 行
→ 续跑: start_env = 60 → 从 env 60 开始，env 0-59 中未完成的 ~40 个永久丢失
```

**更糟的是**: `as_completed()` 返回顺序不定——完成写入的 60 个环境在 [0, 100) 区间内任意分布。续跑时不仅丢失了未完成的 40 个，而且无法知道哪些 env ID 丢失了。数据集将出现不可检测的缺口。

**修复方向**:
- 方案 A (最小): 将批次内的结果写入临时文件，整批完成后原子重命名
- 方案 B (更健壮): 写入 per-env `.done` 标记文件，续跑时用标记文件而非行数推导
- 方案 C (折中): 顺序模式下用行数续跑 (安全)，多进程模式下用 checkpoint.txt (仅批次边界)

---

### Bug 2 (P1 — CONFIRMED): Ctrl+C 在多进程批次执行期间被忽略

- **文件**: `scripts/generate_data.py:222-235`
- **严重级别**: P1 — 用户体验 + 潜在数据丢失
- **类型**: 信号处理 + 并发控制

**根因**: `_stop_requested` 标志仅在批次循环顶部检查 (line 205)。`as_completed()` 循环 (lines 222-235) 没有标志检查、没有超时、没有取消机制：

```python
for future in as_completed(future_to_id):   # ← 无 _stop_requested 检查
    i = future_to_id[future]
    try:
        sft_sample, dpo_samples = future.result()  # ← 阻塞等待
        ...
```

用户按 Ctrl+C → 信号处理器打印 "Stopping after current batch..." 并设置标志 → `as_completed()` 无视标志，继续处理所有 100 个 future → **用户被强制等待 10+ 分钟**。

在 Linux/fork 模式下，子进程继承信号处理器但 `_process_one_environment` 不读取标志，继续运行。在 Windows/spawn 模式下，子进程没有信号处理器，`KeyboardInterrupt` (BaseException) 未捕获，导致 `BrokenProcessPool`。

**修复方向**: 在 `as_completed()` 循环中添加 `if _stop_requested: executor.shutdown(wait=False, cancel_futures=True); break`。同时为 `future.result()` 添加 `timeout` 参数。

---

### Bug 3 (P2 — CONFIRMED): Ctrl+C 在第一个循环迭代前触发 NameError

- **文件**: `scripts/generate_data.py:285-288`
- **严重级别**: P2 — 干净退出失败
- **类型**: 变量作用域

**根因**: 中断清理代码引用了仅在循环体内赋值的变量：

```python
# 循环体内:
batch_end = min(batch_start + batch_size, num_envs)  # line 208
i = batch_start  # line 254 (仅 sequential 分支)

# 循环后:
if _stop_requested:
    last_ckpt = f"~{batch_end}" if n_workers > 0 else f"~{i+1}"  # line 286
```

如果在进入循环前 `_stop_requested` 已为 True，循环体从未执行，`batch_end` 和 `i` 都未定义 → `NameError`。

**触发窗口**: 信号注册 (line 193) 和首次循环迭代 (line 204) 之间的任意 Ctrl+C。窄窗口但真实存在——例如在 "Initializing components..." 期间按 Ctrl+C。

**修复方向**: 在循环前初始化 `batch_end = start_env` 和 `i = start_env - 1`。

---

### Bug 4 (P1 — CONFIRMED, 预存): DPO 效用验证被完全绕过

- **文件**: `scripts/validate_data.py:95-96, 180-181, 299-301`
- **严重级别**: P1 — 质量保证失效
- **类型**: 字段名不匹配 (预存 Bug, 非本分支引入)

**根因**: `validate_data.py` 期望 DPO 记录有字段 `utility_chosen` 和 `utility_rejected`，但 `OracleDataGenerator._build_dpo_pairs()` (oracle_generator.py line 186) 只写 `utility_gap`。三处检查路径全部静默失效：

```python
# 路径 1: validate_dpo_sample()
u_chosen = item.get("utility_chosen", None)   # → None
u_rejected = item.get("utility_rejected", None) # → None
if u_chosen is not None:  # 永远 False
    # 单调性检查 (chosen > rejected) 从未执行
    ...

# 路径 2: compute_stats()
chosen = [r["utility_chosen"] for r in dpo_records if "utility_chosen" in r]  # → []
if len(u_chosen) > 0:  # 永远 False
    # 统计计算从未执行
    ...

# 路径 3: 汇总段相同模式
```

**影响**: 运行 `validate_data.py` 后报告 "0 issues"，即使存在 chosen 效用低于 rejected 的无效 DPO 对。DPO 训练可能学到倒退的偏好信号，而没有任何工具能检测到。

**修复方向**: 
- 方案 A: 将 `_build_dpo_pairs()` 改为写入 `utility_chosen` + `utility_rejected` + `utility_gap`
- 方案 B: 将 `validate_data.py` 改为从 `utility_gap` 验证 (gap > 0 即为有效)

---

### Bug 5 (P2 — PLAUSIBLE): `ISACScenarioGenerator(seed=None)` 行为契约变更

- **文件**: `src/env/isac_scenario.py:83, 97`
- **严重级别**: P2 — 无当前调用方受影响
- **类型**: API 语义变更

**根因**: 新增 `self.base_seed = seed if seed is not None else 0`。当 `seed=None` 时 `base_seed=0`，导致 `sample_rng = RandomState(0 * 100000 + sample_id) = RandomState(sample_id)`——完全确定性。而旧代码 `RandomState(None)` 使用系统熵。

```python
# 旧行为: ISACScenarioGenerator(seed=None).sample(0) → 每次运行不同
# 新行为: ISACScenarioGenerator(seed=None).sample(0) → 每次运行相同 (seed=0 的确定性派生)
```

当前所有调用方 (`generate_data.py --seed 42`, `evaluate.py seed=42`) 传递显式整数，未触发此路径。但 `self.rng` 属性成为死代码 (创建后从未读取)，且 API 合约为隐式变更。

**修复方向**: 用 `seed = seed if seed is not None else np.random.randint(0, 2**31-1)` 保留 `None` = 随机的语义，或移除 `seed` 参数的默认值并强制调用方显式传入。

---

### Bug 6 (P2 — CONFIRMED): `eda_data.py` 在空 SFT 文件上崩溃

- **文件**: `scripts/eda_data.py:350-356`
- **严重级别**: P2 — 诊断工具自崩溃
- **类型**: 边界条件

**根因**: `check_diversity()` 假设 SFT 文件至少有一条有效 JSON：

```python
dq = np.array(all_dq)   # 如果 all_dq 为空 → shape = (0,) — 1-D 数组
N, M = dq.shape[0], dq.shape[1]  # IndexError: tuple index out of range
```

如果数据生成失败产生只有空行或畸形 JSON 的文件，EDA 工具自身崩溃，而非打印 "0 valid records"。

**修复方向**: `if len(all_dq) == 0: print("No valid records"); return`

---

## 质量发现: 5 个架构/效率问题

### Q1 — 重复 tokenization 逻辑 (SFTDataset vs DPODataset)

- **文件**: `src/data/dataset.py:41-65, 118-155`
- **严重级别**: 维护性 — 每次 budget/BOS/EOS 变更需双倍工作

`SFTDataset.__getitem__` 和 `DPODataset._encode_pair` 包含完全相同的 tokenization 代码 (prompt 截断、response 截断 + `add_special_tokens=False`、eos 追加、label/control/label-mask 构造、padding)。30 行逐字复刻。未来每处改动 (如 budget 1024→1536) 必须在两处同步，遗漏一处即导致训练数据分布不一致。

**建议**: 抽取共享 `_tokenize(prompt, response)` 方法，SFT 在其基础上追加 auxiliary tensors，DPO 在其基础上返回 base dict。

---

### Q2 — `eda_data.py` 使用启发式 token 估算器 (自欺性)

- **文件**: `scripts/eda_data.py:49-53`
- **严重级别**: 功能缺陷 — Section 1 截断检查不可靠

```python
def estimate_tokens(text: str) -> int:
    alpha_chars = sum(1 for c in text if c.isalpha() or c == ' ')
    other_chars = len(text) - alpha_chars
    return int(alpha_chars / 4 + other_chars / 2.5)
```

项目自身的 Doc #13 事后分析证明了此启发式低估真实 SentencePiece token 数 **3.3 倍**（估算 ~500 vs 实测 1678）。EDA 的 Section 1（截断检查）因此形同虚设：启发式报告 "900 tokens < 1024 budget ✅"，但真实 tokenizer 计数可能是 3000。

**建议**: 加载真实 `AutoTokenizer.from_pretrained('google/gemma-3-12b-it')` 进行计数（一次性成本 ~2 秒 + 几百 MB），或至少将启发式乘以 3.3x 安全因子并在输出中标注 "approximate"。

---

### Q3 — 硬编码 `CFG` 字典与 `configs/default.yaml` 脱节

- **文件**: `scripts/eda_data.py:34-40`
- **严重级别**: 维护性 — 配置漂移导致误报/漏报

`eda_data.py` 硬编码了 `M`, `K`, `v_max`, `p_max_W`, `prompt_budget` 等关键参数。`generate_data.py` 从 `default.yaml` 读取同一批参数。当配置变更（如 `p_max_W` 从 1.0 改为 2.0，或 `v_max` 从 15 改为 30），EDA 静默使用旧阈值，产生误报（"功率超限" 实则新限制更宽）或漏报（真正的超限在新阈值以下）。

**建议**: 使用 `yaml.safe_load()` 读取 `configs/default.yaml`，与 `generate_data.py` 保持一致。

---

### Q4 — BLAS 线程抑制仅设置在 `generate_data.py`

- **文件**: `scripts/generate_data.py:27-31`
- **严重级别**: 效率 — 训练脚本可能遭遇同样的 CPU 死锁

```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```

此修复防止 Intel MKL/OpenBLAS 在每个 multiprocessing worker 内展开全部核心的线程池。但 `src/training/train_sft.py` (DataLoader `num_workers=4`) 和 `src/training/train_dpo.py` (`num_workers=2`) 也使用多进程数据加载，且每个 worker 导入 torch (→ numpy → MKL)。它们未设置这些环境变量。

**建议**: 将 BLAS 线程抑制迁移到项目入口 (`src/__init__.py` 或顶层 `__init__.py`)，或在训练脚本开头添加相同设置。

---

### Q5 — 冗余 `np.round()` + 递归 `_trunc()` 双重舍入

- **文件**: `src/data/prompt_builder.py:146-155` + `src/data/oracle_generator.py:227-229`
- **严重级别**: 效率 — 每次响应生成浪费 ~176 次 Python 函数调用

数据管道对每份响应执行三次浮点舍入：
1. `oracle_generator._extract_prior()` → `np.round(delta_q, 4).astype(np.float32)` — float32 下的 np.round
2. `prompt_builder.format_oracle_response()` → `np.round(delta_q, 4).tolist()` — 再次 np.round (数据已经 4dp)
3. 同上 → `_trunc(np.round(...).tolist())` — 递归 Python `round()` 遍历 176 个元素

第一次 `np.round` 是冗余的——`_trunc` 已经对每个元素执行 `round(obj, 4)`。对于 5000 envs × 91 次 `format_oracle_response` 调用 × 176 个浮点数，这是 ~8000 万次冗余 `round()` 调用。

**建议**: 移除 `prompt_builder.py` 中的 `np.round()`，表达式改为 `_trunc(delta_q.tolist())`。`oracle_generator.py` 中的 `np.round` 保留（用于 float32 存储精度）。

---

## 驳回: 2 个候选人被证伪

### Refuted 1: `"disp": False` 从 scipy L-BFGS-B 中移除导致 stdout 洪水

**候选人声称**: 移除 `options={"maxiter": 20, "disp": False}` 中的 `"disp": False` 会导致 scipy 在多进程下打印收敛诊断到 stdout。

**驳回原因**: scipy 的 L-BFGS-B 在 `disp` 不在 options 中时默认 `iprint=-1`（无输出），与 `disp=False` 行为完全相同。这是零影响的变更。

---

### Refuted 2: `tokenizer.eos_token_id` 可能为 None 导致训练崩溃

**候选人声称**: `self.tokenizer.eos_token_id` 未做 None 检查，追加到 `resp_ids` 可能产生 `[..., None]` 引发 PyTorch 崩溃。

**驳回原因**: Gemma 3 使用 SentencePiece tokenizer，EOS token (id=1) 直接硬编码在二进制 `.model` protobuf 文件中——它不是仅在 JSON config 中声明的可丢失字段。对于此类 tokenizer，`eos_token_id` 永远不可能为 None。即便 `tokenizer_config.json` 损坏，ID 仍可从 SentencePiece 模型文件中恢复。

---

## 修复方案

### 立即修复 (跑全量数据前)

| Bug | 优先级 | 修复 |
|-----|--------|------|
| Bug 1: 续跑数据丢失 | P0 | 批次结果写入临时文件，批次完成时原子 rename；或在循环中用 checkpoint 文件而非行数续跑 |
| Bug 2: Ctrl+C 被忽略 | P1 | `as_completed()` 内添加 `_stop_requested` 检查 + `executor.shutdown(cancel_futures=True)` |
| Bug 3: NameError 早期间断 | P2 | 循环前初始化 `batch_end = start_env`, `i = start_env - 1` |
| Bug 4: DPO 验证绕过 | P1 | `_build_dpo_pairs` 写入 `utility_chosen` + `utility_rejected`，或 `validate_data.py` 改用 `utility_gap` 检查 |
| Q2: EDA token 启发式 | P1 | 加载真实 tokenizer 或对启发式结果应用 3.3× 安全因子 |

### 后续优化 (训练阶段前)

| 问题 | 建议 |
|------|------|
| Q1: 重复 tokenization | 抽取 `_tokenize()` 共享方法 |
| Q3: EDA 硬编码 CFG | 改为读取 `configs/default.yaml` |
| Q4: BLAS 抑制缺失 | 迁移至项目入口或训练脚本 |
| Q5: 双重舍入 | 移除 `prompt_builder` 中的冗余 `np.round` |
| Bug 5: seed=None 契约 | 保留 `None` = 随机的语义 |
| Bug 6: EDA 空文件崩溃 | 添加空数据保护 |

---

## 经验教训

### 1. 多进程 ≠ 顺序循环 + 速度

多进程引入了原顺序模式中不存在的故障模式：
- **部分写入**: 批次内的成功/失败是独立的——不能假设批次完成 = 所有 env 完成
- **顺序不确定性**: `as_completed()` 的完成顺序不可预测——不能依赖 env ID 连续性
- **信号传播**: fork vs spawn 的信号处理器继承行为完全不同——`KeyboardInterrupt` 在 spawn 下是 `BaseException` 而非 `Exception`

### 2. 基于统计的恢复是不安全的

"统计已有行数 → 推导完成位置" 是顺序模式的合理启发式，但在多进程下不成立。恢复必须基于显式的完成跟踪（标记文件、checkpoint 文件），而非对副作用的统计推断。

### 3. 中断响应是正确性需求，非体验需求

在顺序模式下，每个环境处理后检查中断标志 → 最多 1 个环境的延迟。在多进程模式下，批次提交后无法被中断 → 可能 100 个环境的延迟。中断响应不是 "优化体验"——它是数据完整性保障。用户因无法中断而 `kill -9` → checkpoint 未写入 → 数据丢失。

### 4. 验证工具本身需要验证

Bug 4 (DPO 验证静默绕过) 是预存 Bug——它可能在数据生成工具首次编写时就存在了。Doc #10 和 #11 的经验反复出现：EDA/验证工具的逻辑漏洞和被验证的数据 Bug 同样危险。工具报告 "0 issues" 可能意味着 "没有问题" 或 "检查从未执行"。每个验证函数的输出都应该包含 "N records checked" 计数，以便区分 "检查通过" 和 "未检查"。

### 5. 8 角度并行审查的性价比

| 指标 | 值 |
|------|-----|
| Finder 角度 | 8 |
| 候选发现 | 39 |
| 去重后 | ~25 |
| 验证后确认/可信 | 6 + 5 质量 = 11 |
| 驳回 | 2 |
| 发现率 | 11/39 = 28% |
| 关键发现 | 1 P0 + 2 P1 |

高力度 (high effort) 审查产生的候选中有 72% 被验证驳回或归入低优先级。但 1 个 P0 (多进程数据丢失) 如果能拦截在数据生成之前，节省的是 ~1.5 小时的服务器时间 + 5000 条不可恢复的训练数据。28% 的准确率在这个场景下是合理的——宁可多报不可漏报。

---

## 文件变更清单

| 文件 | 变更类型 | 关键改动 |
|------|---------|---------|
| `scripts/generate_data.py` | Bug 修复 | 批次原子写入、`as_completed()` 中断检查、变量初始化 |
| `scripts/validate_data.py` | Bug 修复 | DPO utility 字段改用 `utility_gap` |
| `scripts/eda_data.py` | 改进 | 真实 tokenizer 计数 / 安全因子、硬编码 CFG 改为 yaml 读取、空数据保护 |
| `src/data/oracle_generator.py` | Bug 修复 | `_build_dpo_pairs` 添加 `utility_chosen` + `utility_rejected` |
| `src/data/dataset.py` | 重构 | 抽取共享 `_tokenize()` 方法 |
| `src/data/prompt_builder.py` | 优化 | 移除冗余 `np.round()`, `_trunc()` 改为非递归 |
| `src/env/isac_scenario.py` | Bug 修复 | `seed=None` 保留随机语义 |
| `src/training/train_sft.py` | 改进 | 添加 BLAS 线程抑制 |
| `src/training/train_dpo.py` | 改进 | 添加 BLAS 线程抑制 |

**待提交**: 修复完成后在 `feature/multiprocessing` 分支上新开 commit。

---

> **相关文档**:
> - [Doc #10 — P0 物理约束穿透 Bug](10_physical_constraint_bug_postmortem.md)
> - [Doc #11 — P0 EDA 双 Bug 事后分析](11_pre_training_data_eda_postmortem.md)
> - [Doc #12 — 验证缺口审计](12_remaining_verification_gaps.md)
> - [Doc #13 — P0-1 Response Token 溢出 Bug](13_response_token_bug_postmortem.md)
>
> **审查 Commit**: `619abcc` (feature/multiprocessing HEAD) | **审查范围**: 14 commits vs `origin/master`
