# 交接文档 #3 — 数据生成：四次运行与遇到的问题

> 时间段: 2026-06-23 → 2026-06-25
> 本阶段目标: 生成 5000 环境 × 10 重启 = 50,000 次 SCA-FP 求解的高质量训练数据

---

## 目录

1. [四次数据生成运行总览](#四次数据生成运行总览)
2. [Run 1: 烟雾测试 (5 envs) — P0 物理约束穿透](#run-1-烟雾测试-5-envs)
3. [Run 2: 首次全量尝试 (70 envs) — P0 双 Bug 发现](#run-2-首次全量尝试-70-envs)
4. [Run 3: Smoke20 验证 — P0-1 Token 溢出](#run-3-smoke20-验证)
5. [Run 4: 最终 5000 envs 成功](#run-4-最终-5000-envs)
6. [多进程 Code Review 与 Q1-Q5 清理](#多进程-code-review-与-q1-q5-清理)
7. [所有 Bug 清单与修复汇总](#所有-bug-清单与修复汇总)

---

## 四次数据生成运行总览

| 运行 | 规模 | 结果文件 | 状态 | 关键发现 |
|------|------|---------|------|---------|
| **Run 1** | 5 envs | [result.md](docs/04_data_results/result.md) | ❌ P0 发现 | 物理约束穿透 — δ_q=800m (57× 上限) |
| **Run 2** | 70 envs | [result2.md](docs/04_data_results/result2.md) | ❌ P0×2 发现 | 零环境多样性 + Response 512 token 截断 |
| **Run 3** | 20 envs | [result3.md](docs/04_data_results/result3.md) | ⚠️ P0-1 发现 | BPE 浮点碎片化 → 1678 tokens (预算 1024) |
| **Run 4** | 5000 envs | [final_result.md](docs/04_data_results/final_result.md) | ✅ 成功 | 0 issues, all clean, 准备 SFT |

---

## Run 1: 烟雾测试 (5 envs)

### 运行参数

```bash
python scripts/generate_data.py --num-env 5 --num-restarts 10 --save-every 1
```

- **时间**: 125s
- **产出**: SFT 5 | DPO 187
- **结果文件**: [docs/04_data_results/result.md](docs/04_data_results/result.md)

### 发现: P0 物理约束穿透

`validate_data.py` 检查发现：
- δ_q 水平位移: mean=382.4m, max=864.8m
- 约束上限: 15m (v_max·Δt = 15 m/s × 1.0s)
- **超出约束 57 倍**

### 根因

SCA-FP 的 `_random_init()` 将 UAV 随机抛掷在整个 1000×1000m 区域，无视当前位置：

```python
# 修复前
Q = np.random.uniform(0, area_w, (M, 3))  # 任意位置

# 修复后
Q = q_current + v_max_dt * random_on_sphere(M)  # 球面约束 15m 内
```

### 修复

| Commit | 内容 |
|--------|------|
| `1caa482` | SCA-FP 求解器强制 v_max·Δt 约束 |
| `2b75aa1` | 统一 3D 移动约束 + evaluate.py 参数 |
| `14afd9a` | Box→Sphere 3D Euclidean 惩罚 + 球面采样 |
| `1296736` | Doc #10: P0 物理约束 Bug 事后分析 |

详见: [docs/03_bug_postmortems/10_physical_constraint_bug_postmortem.md](docs/03_bug_postmortems/10_physical_constraint_bug_postmortem.md)

---

## Run 2: 首次全量尝试 (70 envs)

### 运行参数

修复物理约束后，尝试中等规模生成：

- **环境数**: 70 (小规模验证后再跑全量)
- **重启数**: 10 per env
- **时间**: ~3.5h
- **结果文件**: [docs/04_data_results/result2.md](docs/04_data_results/result2.md)

### 验证通过 → EDA 却发现 2 个 P0

`validate_data.py` 报告: **0 issues, all clean**。但 EDA（探索性数据分析）揭示了两个灾难性缺陷：

#### Bug P0-a: 环境多样性崩溃

```
EDA Section 3.5 — UAV Initial Position Distribution:
UAV0: x∈[464,464] y∈[766,766] h∈[85,85]m    ← min == max == 零方差!
UAV1: x∈[339,339] y∈[270,270] h∈[105,105]m
UAV2: x∈[379,379] y∈[899,899] h∈[176,176]m
UAV3: x∈[478,478] y∈[786,786] h∈[274,274]m
```

**5000 个环境的 UAV 初始位置完全相同**。根因：多进程 pickle 序列化导致全局 RNG 状态在所有 worker 间共享——每个进程 fork 时继承相同的 NumPy RNG 种子。

**修复**: 每样本独立 RNG — `np.random.RandomState(base_seed * 100000 + sample_id)`

#### Bug P0-b: Response JSON 截断

```
⚠ SFT L1: response ~886 tokens > budget 512 — will be TRUNCATED
⚠ SFT L2: response ~889 tokens > budget 512 — will be TRUNCATED
```

100% 的 SFT response 超出 512-token 预算。训练时 JSON 结构被硬截断——模型永远学不会生成完整的 output。

**修复**: budget 从 512 → 1024 (`8daddac`)

### 修复 Commit

| Commit | 内容 |
|--------|------|
| `8daddac` | P0-a: per-sample RNG + P0-b: budget 512→1024 |
| `6cf0d13` | Doc #11: P0 EDA 双 Bug 事后分析 |

详见: [docs/03_bug_postmortems/11_pre_training_data_eda_postmortem.md](docs/03_bug_postmortems/11_pre_training_data_eda_postmortem.md)

---

## Run 3: Smoke20 验证

### 运行参数

P0-a/b 修复后，生成 20 环境进行真实 tokenizer 验证：

- **环境数**: 20
- **结果文件**: [docs/04_data_results/result3.md](docs/04_data_results/result3.md)

### 发现: P0-1 BPE 浮点数碎片化

用真实 Gemma 3 tokenizer 计数后发现：

```
Prompt: 2455 tokens (budget 3072)    ← OK
Response: 1678 tokens (budget 1024)  ← ❌ 超出 64%！
```

**根本原因**: Gemma 3 的 SentencePiece BPE tokenizer 将高精度浮点数（如 `0.1910400390625`）碎片化为 5-8 个 subword token。176 个浮点数 × 每数 3-8 token = 实际 900-1400 token。启发式估算 `chars/4 + digits/2.5` 完全失效。

### 三轮迭代修复

| 轮次 | 修复 | Commit | 效果 |
|------|------|--------|------|
| **Fix 1** | 4dp 精度截断 (`round(x, 4)`) | `8b1a77c` | 1678 → ~1100 tokens |
| **Fix 2** | Python `round()` 在 `.tolist()` 之后 (消除 float32→float64 artifact) | `4f4e4e8` | ~1100 → ~950 tokens |
| **Fix 3** | Compact JSON (无缩进, 无空格) | `223aace` | ~950 → **824 tokens** ✅ |

**最终结果**: Response 从 1678 tokens 降至 824 tokens，安全落在 1024 budget 内。

### 同步修复

| Commit | 内容 |
|--------|------|
| `8b1a77c` | 4dp 截断 |
| `4f4e4e8` | float32→float64 artifact 清理 |
| `223aace` | Compact JSON |
| `560023d` | Doc #13: P0-1 Response Token 溢出事后分析 |
| `c7a0685` | EDA power tolerance 修复 (`1e-6 → 0.01`) |
| `d0d51b1` | Doc: EDA smoke20 完整报告 |

详见: [docs/03_bug_postmortems/13_response_token_bug_postmortem.md](docs/03_bug_postmortems/13_response_token_bug_postmortem.md)

### 额外修复

| Commit | 内容 |
|--------|------|
| `feb4f50` | 防止重复 `<bos>` + 添加 `<eos>` + budget 512→1024 |
| `b12ec65` | 移除无用 torch imports + UTF-8 编码 |
| `4532695` | 每 10 env 进度报告 (修复小 batch 静默) |
| `808b271` | EDA budget 常量同步 (512→1024) |

---

## Run 4: 最终 5000 envs 成功

### 运行参数

所有修复应用后，执行全量 5000 环境生成：

```bash
python scripts/generate_data.py \
    --num-env 5000 \
    --num-restarts 10 \
    --workers 70 \
    --save-every 500 \
    --output-dir /root/autodl-tmp/data/full5000
```

- **时间**: ~3.5h (使用 70 进程并行)
- **产出**: SFT 5000 | DPO 186,896

### 验证结果

```
SFT Samples: 5000
  δ_q 3D位移 (‖Δq‖₂): mean=15.0m [14.2, 15.0]  (上限=15m)

DPO Samples: 186896
  Utility chosen: mean=924.78 [256.69, 4729.64]
  Utility rejected: mean=891.88 [224.82, 4718.47]
  Utility Δ: mean=32.90 [0.09, 2016.23]

Issues: 0 — all clean
```

### EDA 确认

- Token: mean=1696, safe in 4096 budget
- Response: mean=344, 安全落在 1024 budget
- δ_q 位移全部 ≤ 15.0m (物理边界饱和 — 最优解通常在边界)
- Power 全部在 P_max=1W 预算内
- 方向分布均匀 (360° 覆盖)
- 0 NaN, 0 Inf, 0 解析错误

详见: [docs/04_data_results/final_result.md](docs/04_data_results/final_result.md)

---

## 多进程 Code Review 与 Q1-Q5 清理

在最终 5000 环境运行前，对 `feature/multiprocessing` 分支进行了 8 角度高力度 Code Review。

### 发现的 5 个 Bug

| # | 严重级别 | Bug | 修复 |
|---|---------|-----|------|
| 1 | **P0** | 多进程续跑静默数据丢失 | 批次原子写入 (临时文件+合并) |
| 2 | **P1** | Ctrl+C 在 `as_completed()` 中被吞 | 显式信号处理 + `future.cancel()` |
| 3 | **P2** | 早期 SIGINT 触发 `NameError` | 变量声明提前 |
| 4 | **P1** | DPO 效用验证完全绕过 | 修复 oracle 返回字段 |
| 5 | **P2** | EDA 空文件崩溃 + dead code | 空文件 guard + 死代码移除 |

### Q1-Q5 清理

| # | 内容 | 效果 |
|---|------|------|
| Q1 | 提取共享 `_tokenize_pair()` | 消除 ~30 行重复代码 |
| Q2 | (已在前述修复中覆盖) | — |
| Q3 | EDA config 加载修复 | 不再硬编码路径 |
| Q4 | BLAS 线程抑制 (在 numpy 前) | 防止 DataLoader workers CPU 100% |
| Q5 | 移除 float 双精度舍入 | 精度截断链简化 |

### 相关 Commits

| Commit | 内容 |
|--------|------|
| `a27bc04` | P0-P2 一审修复 |
| `ee6352d` | Q1/Q3/Q4/Q5 清理 |
| `11269cc` | Doc #15: 一审修复报告 |
| `f775593` | Doc #15 更新 (标记 Q 项已修复) |

详见: [docs/02_code_reviews/14_first_review_post_datagen.md](docs/02_code_reviews/14_first_review_post_datagen.md) 和 [docs/02_code_reviews/15_first_review_fix_report.md](docs/02_code_reviews/15_first_review_fix_report.md)

---

## 所有 Bug 清单与修复汇总

| # | 发现阶段 | 严重级别 | Bug | 影响 | Commit |
|---|---------|---------|-----|------|--------|
| 1 | Smoke Test | **P0** | 物理约束穿透 (δ_q=800m) | 5000 条训练数据全部作废 | `14afd9a` |
| 2 | EDA Run 2 | **P0** | 环境多样性崩溃 (相同种子) | 模型只学到一种状态 | `8daddac` |
| 3 | EDA Run 2 | **P0** | Response 512 token 截断 | 100% SFT 样本 JSON 不完整 | `8daddac` |
| 4 | Token 检查 | **P0-1** | BPE 浮点碎片化 (1678 tokens) | Response 超出 1024 budget 64% | `8b1a77c`+`4f4e4e8`+`223aace` |
| 5 | Code Review | **P0** | 多进程续跑数据丢失 | Mid-batch 崩溃后 env 永久缺失 | `a27bc04` |
| 6 | Code Review | **P1** | Ctrl+C 被 `as_completed()` 吞掉 | 中断后进程继续运行数分钟 | `a27bc04` |
| 7 | Code Review | **P1** | DPO 效用验证绕过 | 偏好对质量未验证 | `a27bc04` |
| 8 | Code Review | **P2** | 早期 SIGINT 触发 NameError | 用户中断时崩溃 | `a27bc04` |
| 9 | Code Review | **P2** | EDA 空文件崩溃 | 工具链鲁棒性 | `a27bc04` |
| 10 | Dataset | **P0** | 重复 `<bos>` + 缺失 `<eos>` | 模型无法学会停止生成 | `feb4f50` |

**总共: 4 个 P0 级 Bug + 1 个 P0-1 + 2 个 P1 + 2 个 P2 = 10 个缺陷全部修复**

---

## 相关文档

- [[16_handoff_01_project_direction](docs/05_handoff/16_handoff_01_project_direction.md)] — 论文总体方向
- [[17_handoff_02_pre_datagen](docs/05_handoff/17_handoff_02_pre_datagen.md)] — 数据生成前的准备
- [[19_handoff_04_post_datagen](docs/05_handoff/19_handoff_04_post_datagen.md)] — 数据生成后的验证与下一步
- [[10_physical_constraint_bug_postmortem](docs/03_bug_postmortems/10_physical_constraint_bug_postmortem.md)] — P0 物理约束 Bug
- [[11_pre_training_data_eda_postmortem](docs/03_bug_postmortems/11_pre_training_data_eda_postmortem.md)] — P0 EDA 双 Bug
- [[13_response_token_bug_postmortem](docs/03_bug_postmortems/13_response_token_bug_postmortem.md)] — P0-1 Token 溢出
- [[14_first_review_post_datagen](docs/02_code_reviews/14_first_review_post_datagen.md)] — 一审发现
- [[15_first_review_fix_report](docs/02_code_reviews/15_first_review_fix_report.md)] — 一审修复报告
