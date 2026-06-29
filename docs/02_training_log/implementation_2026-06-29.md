---
type: postmortem
status: complete
severity: N/A
stage: code
commits: [7cedb02]
last_updated: 2026-06-29
related: [status, data_degeneracy, adr_006_data_regeneration, CONTEXT.md, oom_incidents]
---

# Grilling 终稿代码落地 — 全纪录

**Commit**: [`7cedb02`](https://github.com/Lampotaku/UAV-ISAC-MLLM/commit/7cedb02)
**变更**: 6 文件, +752/-84 行 | **审查**: 3 bug 当场斩杀

## 改了什么

### 1. `src/solver/sca_fp.py` — SCA-FP 求解器 (+27 行)

**新增 SCAFPConfig 字段**:

```python
max_iters: int = 100              # 硬上限安全帽, 覆盖 max_outer_iters
lambda_repel: float = 0.01        # 多 UAV 空间互斥力权重
epsilon_min_repel: float = 1e-6   # 互斥力分母数值地板
```

**solve() 循环**: 改用 `max_iters` (默认 100) 而非 `max_outer_iters` (30)。
- 正常数据生成: SCA-FP 在 30 步内收敛, 100 仅作为 safety net
- Snap-back 重跑: 从扰动点出发可能需要更多步, 100 提供充分预算
- 逻辑: `max_iters = self.cfg.max_iters if self.cfg.max_iters > 0 else self.cfg.max_outer_iters`

**空间互斥力** (`_optimize_deployment_sca` + `_compute_utility`):

```
Penalty_repel = λ_repel × Σ_i Σ_{j>i} 1 / max(||q_i - q_j||², ε_min)
```

- 部署优化: 每步 L-BFGS-B 的 objective 函数中对 UAV m 施加对其他 UAV 的反比惩罚
- 效用计算: 全局 utility 减去互斥力惩罚 (与 objective 函数一致)
- λ_repel = 0.01 起步, 1/d² 自动衰减远距离 UAV 对的惩罚

### 2. `src/data/oracle_generator.py` — Oracle 数据生成器 (+374/-84 行)

**核心重写** — `_process_one_environment()` 全流程:

```
1. 采样环境 + 构造 prompt
2. N=10 次 Random Restart SCA-FP → 10 个局部最优解
3. Pareto 过滤:
   - 计算 [0,0,0] 不动方案 baseline utility
   - 丢弃 utility < baseline 的解
   - 丢弃 utility < max_utility × 0.95 的劣质坑
   - 存活数 < 2 → 丢弃该环境 (缺乏偏好信号)
4. 微扰回弹测试 (Snap-back):
   - 取 Top-K 候选 (默认 K=3)
   - 每个候选施加随机方向 + 固定幅度 ε 的 3D 扰动
   - 以扰动点为 warm_start 重跑 SCA-FP
   - 迭代步数最少者当选 Chosen (盆地最宽)
5. Rejected δ_q 构造 (混合策略):
   - 70%: SCA-FP 次优解 — 取 utility 最低的有效解
   - 30%: 启发式物理陷阱 — 短视直线 / 原地不动 / 旧世界残影
   - 所有 Rejected 必经 clip_to_physics_bounds 约束投影
6. 返回 SFT 样本 (Chosen prior) + DPO 对 (Chosen vs Rejected)
```

**新增方法**:

| 方法 | 作用 |
|------|------|
| `_pareto_filter(solutions, baseline_util)` | 双闸门过滤 (baseline + 95% utility ratio) |
| `_compute_baseline_utility(env_dict)` | [0,0,0] 不动方案 utility |
| `_run_snapback_test(env_dict, candidate, epsilon, seed)` | 微扰回弹 → 返回迭代步数 |
| `_construct_rejected(env_dict, solutions, q_current, sample_id, baseline_util)` | 混合 Rejected 构造 → (delta_q, utility_estimate) |
| `_construct_heuristic_rejected(env_dict, q_current, rng)` | 三选一物理陷阱 |
| `_clip_to_physics_bounds(delta_q, q_current)` | 3D 移动性 + 区域 + 高度三约束投影 |
| `_format_rejected_response(sample_id, delta_q, delta_a, delta_p)` | Rejected JSON (δ_q=陷阱, δ_a/δ_p=Chosen) |

**性能优化**:
- baseline_util 缓存在 `_process_one_environment` 中计算一次, 传递给 `_pareto_filter` 和 `_construct_rejected`
- Gap 估算不再跑额外 SCA-FP (审查修复: `_compute_utility_of_delta_q` 已删除)
- 从 15 次 SCA-FP/env → 13 次 (10 restarts + 1 baseline + K snap-back)

**新增配置参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `snapback_epsilon` | 1.5 | 微扰幅度 (m) |
| `snapback_top_k` | 3 | 参与回弹测试的候选数 |
| `pareto_utility_ratio` | 0.95 | 低于全局最高 ×ratio 丢弃 |
| `heuristic_reject_ratio` | 0.3 | 启发式陷阱占比 |

### 3. `src/data/dataset.py` — Masked DPO (+92 行)

**核心机制**: DPO 操作在文本 token log-probabilities 上。通过在 tokenization 阶段将 JSON 中 δ_a/δ_p 对应 token 的 label 设为 `-100` (ignore index), DPO 的 `_compute_logprob` 自动跳过这些 token, 梯度集中在 δ_q 的偏好拉扯上。

**实现**:

1. `_find_field_spans_in_json(response, fields)`:
   - 正则匹配 `"delta_a":` 和 `"delta_p":` 在紧凑 JSON 中的字符位置
   - 返回 `[(field_name, char_start, char_end), ...]` 区间列表
   - 区间边界: field_start = key 首字符, field_end = 下一个字段首字符 (或字符串末尾)

2. `_tokenize_pair(..., mask_fields=None)`:
   - 新增 `mask_fields` 参数 (默认 None, SFTDataset 不受影响)
   - 使用 `return_offsets_mapping=True` 获取每个 token 的字符区间
   - 对落于 mask 区间的 token: `labels[i] = -100` + `label_mask[i] = 0`
   - `<eos>` token (手动追加, 无字符偏移) 永远不被 mask

3. `DPODataset._encode_pair()`:
   - 传入 `mask_fields=["delta_a", "delta_p"]`

**为什么 mask label_mask 也必须置 0**:

`_compute_logprob` 中使用 `safe_labels = shift_labels.masked_fill(shift_labels < 0, 0)` 把 -100 替换为 token 0 后再 gather。如果 label_mask 仍为 1, 该位置的 log-prob 会被计入 sum, 产生来自随机 token (id=0) 的噪声。同时置 0 确保该位置对总 log-prob 的贡献为零。

### 4. `scripts/generate_data.py` — 数据生成入口 (+20 行)

**SCAFPConfig 显式参数**:

```python
SCAFPConfig(
    max_iters=100,
    ground_clutter_db=12.0,   # ★ 地面杂波
    lambda_repel=0.01,         # ★ 空间互斥力
)
```

**新 CLI 参数**:

```bash
--snapback-epsilon 1.5       # 微扰幅度 (m)
--snapback-top-k 3           # snap-back 候选数
--pareto-utility-ratio 0.95  # Pareto 过滤阈值
--heuristic-reject-ratio 0.3 # 启发式 Rejected 占比
```

### 5. `scripts/calibrate_epsilon.py` — ε 标定脚本 (新文件, 294 行)

**流程**:
1. 生成 50 个随机环境
2. 每环境: Best-of-N (N=10) → Pareto 过滤 → Top-K 候选
3. 对每个 ε ∈ {0.5, 1.0, 2.0, 4.0, 8.0}m:
   - 对每个候选施加随机扰动 → 重跑 SCA-FP → 记录迭代步数
4. 按 ε 汇总, 计算 variance of iterations
5. 推荐 variance 最大的 ε

**输出示例**:

```
ε (m)      Mean Iters   Std Iters    Variance     CV         Recommend
0.5        3.2          1.1          1.2          0.344
1.0        5.8          3.4          11.6         0.586
2.0        12.3         8.7          75.7         0.707      ★ BEST
4.0        45.2         28.1         789.6        0.622
8.0        87.4         14.2         201.6        0.162
```

**诊断逻辑**:
- Variance < 1.0 → 所有 ε 缺乏区分度 → 可能 ground_clutter 太弱或过滤太激进
- < 5 valid envs → sweep 不可靠, 建议增加 --num-envs

### 6. `scripts/quick_validate_fix.py` — 快速验证脚本 (+29 行)

**Bug 修复**: 原脚本调用 `solver.solve(q_current=env['q'], users=..., targets=...)` — keyword 参数不匹配 `solve(environment, warm_start, seed)` 签名。

修复为:
```python
env_dict = {
    "q_current": env_sample.q_current.copy(),
    "user_positions": env_sample.u_positions.copy(),
    ...
}
sol = solver.solve(env_dict, warm_start=None, seed=i)
dq = sol.Q - env_dict["q_current"]  # SCAFPSolution.Q, 非 dict key
```

同步更新: `SCAFPConfig(lambda_repel=0.01)`。

## 🔍 审查中现场斩杀的 3 个 Bug

### Bug 12a — `converged` 引用旧 `max_outer_iters`

**位置**: `sca_fp.py` 第 187 行

**问题**:
```python
# 修复前:
converged=(outer_iter + 1 < self.cfg.max_outer_iters)  # max_outer_iters=30
```
但循环现在使用 `max_iters=100`。如果求解器在 50 步收敛, `converged` 会是 `False` (50 < 30 为假) — 明明收敛了却被标记为未收敛。

**修复**: `self.cfg.max_outer_iters` → `max_iters` (局部变量)

### Bug 12b — 额外 SCA-FP 调用浪费

**位置**: `oracle_generator.py` `_compute_utility_of_delta_q`

**问题**: 对每个 environment 额外跑一次 `solver.solve()` 来计算 gap。20,000 环境 × 1 次 ≈ 20,000 次不必要求解, 占总预算 260,000 次的 7.7%。

**修复**: 方法已删除。Gap 从已有信息估算:
- SCA-FP 次优解: `gap = chosen.utility - rejected_sol.utility` (已知)
- 启发式陷阱: `gap = chosen.utility - baseline_utility` (已知)

### Bug 12c — ε 标定 baseline 用随机重启

**位置**: `calibrate_epsilon.py` `_pareto_filter`

**问题**:
```python
# 修复前:
zero_sol = solver.solve(env_dict, warm_start=None, seed=999999)
```
`warm_start=None` → 随机初始化, 非 `[0,0,0]` 不动方案。baseline 在不同环境间不可比。

**修复**:
```python
zero_warm = {"delta_q": np.zeros_like(q_cur), ...}
zero_sol = solver.solve(env_dict, warm_start=zero_warm, seed=999999)
```

## 设计验证清单

| 验证项 | 状态 |
|--------|------|
| SCA-FP max_iters 安全帽在所有路径生效 | ✅ |
| 空间互斥力在 objective + utility 中一致 | ✅ |
| Snap-back 测试使用与模型相同的约束投影逻辑 | ✅ |
| Rejected 必经 `_clip_to_physics_bounds` (Deterministic Forward Projection) | ✅ |
| Masked DPO 正则匹配 δ_a/δ_p 且排除 δ_q | ✅ (已验证 3 种测试用例) |
| baseline_util 缓存避免重复 SCA-FP 调用 | ✅ |
| `converged` 字段正确引用 `max_iters` 局部变量 | ✅ (审查修复) |
| `label_mask` 与 `labels` 同步 mask (防止 _compute_logprob 噪声) | ✅ |
| ε 标定与 oracle_generator 共享相同的 baseline 计算逻辑 | ✅ |
| quick_validate_fix 的 `solve()` 调用签名修复 | ✅ |
| 所有文件 AST 语法检查通过 | ✅ |
| 向后兼容: SFTDataset 不受 Masked DPO 影响 | ✅ (mask_fields=None) |

## 服务器执行顺序

```bash
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm
export TORCHINDUCTOR_FLEX_ATTENTION=0

# Step 0: ε 标定 (5 min)
python scripts/calibrate_epsilon.py

# Step 1: 快速验证 (2 min)
python scripts/quick_validate_fix.py

# Step 2: 全量生成 (2-3h) — 用 Step 0 的 ε
python scripts/generate_data.py \
    --num-envs 20000 --num-restarts 10 \
    --snapback-epsilon <EPSILON> --workers 70 \
    --output-dir /root/autodl-tmp/data/full20000_v2

# Step 3: EDA + Top-5000
python scripts/eda_data.py --data-dir /root/autodl-tmp/data/full20000_v2

# Step 4: Masked DPO (5-10h)
python src/training/train_dpo.py \
    --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/checkpoints/stage1_step_150 \
    --data_dir /root/autodl-tmp/data/full20000_v2
```

## 教训

1. **代码审查应在 push 前跑一遍 AST + 数据流追踪** — Bug 12a (converged 字段错引) 如果不修, 每个 snap-back 解的 `converged=False` 会在评估阶段产生混淆
2. **每个 SCA-FP 调用都要算账** — 20,000 envs × 14 vs 13 次求解 = 多浪费 ~1h。删除 `_compute_utility_of_delta_q` 的优化看似微小, 但工程上值得
3. **Masked DPO 的 label_mask 同步是必须的** — 只设 labels=-100 但不设 label_mask=0 会导致 `_compute_logprob` 从 token 0 收集随机 log-prob 并计入 sum
4. **baseline 计算必须使用 warm_start=[0,0,0] 而非随机重启** — 否则不同环境的 baseline 语义不一致, Pareto 过滤失去意义
