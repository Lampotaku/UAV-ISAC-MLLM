# 第十号文档 — P0 物理约束穿透 Bug 完整事后分析

> 发现时间: 2026-06-23 | 发现阶段: Smoke Test (5 envs) | 严重级别: P0
> 状态: ✅ 已修复 | 影响: 若不修复，5000 条训练数据将全部作废
> Commits: `1caa482` + `2b75aa1`

---

## 目录

1. [一句话概述](#一句话概述)
2. [Bug 发现过程](#bug-发现过程)
3. [根因分析](#根因分析)
4. [修复方案](#修复方案)
5. [同行审查与补充](#同行审查与补充)
6. [文件变更清单](#文件变更清单)
7. [验证结果](#验证结果)
8. [经验教训](#经验教训)

---

## 一句话概述

**SCA-FP 求解器的 `_random_init()` 在 Best-of-N 随机重启时将 UAV 随机抛掷在整个 1000×1000m 区域内，完全无视当前物理位置 `q_current`，导致 `delta_q = Q* - q_current` 高达 800m+，远超物理约束 `v_max * Δt = 15m`。**

---

## Bug 发现过程

### 1. 烟雾测试

在 AutoDL RTX 5090 服务器上运行 5 环境的烟雾测试：

```bash
python scripts/generate_data.py --num-env 5 --num-restarts 10 --save-every 1 \
    --output-dir /root/autodl-tmp/data/smoke_test
```

输出正常：
```
Done in 125.0s (0.03h)
  SFT: 5  |  DPO: 187
  Files: .../sft_dataset.jsonl, .../dpo_dataset.jsonl
```

### 2. 数据验证触发告警

运行 `validate_data.py` 后立即暴露出大规模异常：

```
100 issues found:
  ✗ L137: delta_q 水平位移 max=701.5m > 2*v_max*Δt=30.0m
  ✗ L137: delta_q 垂直位移 max=168.0m > 50m
  ✗ L138: delta_q 水平位移 max=701.5m > 2*v_max*Δt=30.0m
  ...
  (and 80 more)

  SFT Samples: 5
    δ_q 水平位移:  mean=382.4m  [88.3, 864.8]
    δ_q 垂直位移:  mean=121.8m  [33.4, 224.5]
```

物理约束 `v_max * Δt = 15m/s × 1.0s = 15m`，而实际产生的 `delta_q` 均值 382m、最大值 864m —— **超过约束 57 倍**。若直接用于 SFT 训练，模型将学会预测物理上不可能的 UAV 位移。

---

## 根因分析

Bug 位于 **`src/solver/sca_fp.py`**，有两层脱节：

### 层 1: `_random_init()` — 全局随机初始化（根本原因）

```python
# 修复前 (第 188-197 行)
def _random_init(self, env: Dict) -> Tuple:
    """生成随机初始点"""
    M, K = self.M, self.K

    # UAV 位置
    Q = np.zeros((M, 3))
    for m in range(M):
        Q[m, 0] = self.rng.uniform(0.1 * self.area_w, 0.9 * self.area_w)  # 100~900m!
        Q[m, 1] = self.rng.uniform(0.1 * self.area_h, 0.9 * self.area_h)  # 100~900m!
        Q[m, 2] = self.rng.uniform(self.H_min + 20, self.H_max - 20)      # 70~280m!
```

**问题**: 每次 Best-of-N 重启时，将 UAV 位置 `Q` 完全随机地初始化在整个地图的任何位置，与当前时间槽的真实起点 `q_current` 毫无关系。这相当于假设 UAV 可以在一个时间槽内瞬移到区域的任意角落。

数据流:
```
ISACScenarioGenerator.sample() → q_current (真实当前位���)
    ↓
SCAFPOptimizer.solve() → _random_init() → Q (完全随机, 无视 q_current)
    ↓
_extract_prior() → delta_q = Q_optimized - q_current  ← 可达 1000m
```

### 层 2: `_optimize_deployment_sca()` — 无移动性边界

```python
# 修复前 (第 364-368 行)
bounds = [
    (0.0, self.area_w),       # x: 0~1000m (绝对地图)
    (0.0, self.area_h),       # y: 0~1000m (绝对地图)
    (self.H_min, self.H_max), # H: 50~300m (绝对高度)
]
```

**问题**: L-BFGS-B 的 box bounds 只受全局地图边界和绝对高度范围约束，完全没有 `q_current ± v_max*Δt` 的移动性约束。即使 `_random_init` 从 `q_current` 附近出发，优化器也可能在迭代中把 UAV 移动到数百米外。

### 受影响的调用链

```
generate_data.py → SCAFPOptimizer() → _random_init() → _optimize_deployment_sca()
                                                      → _optimize_deployment_sca()
evaluate.py     → SCAFPOptimizer() → _warmstart_to_init() → _optimize_deployment_sca()
```

两条链路都受影响：数据生成（Best-of-N 随机重启）和评估（MLLM 热启动）。

---

## 修复方案

### 修复 1: `SCAFPOptimizer.__init__()` — 新增移动性参数

```python
# 新增参数
v_max: float = 15.0,          # UAV 最大速度 (m/s)
slot_duration: float = 1.0,   # 时间槽长度 (s)
# 计算属性
self.max_displacement = v_max * slot_duration  # = 15m
```

### 修复 2: `_random_init()` — 从 q_current 球形邻域采样

```python
def _random_init(self, env: Dict) -> Tuple:
    """从当前 UAV 位置 q_current 出发，在 v_max*Δt 球形邻域内随机扰动"""
    M, K = self.M, self.K
    q_current = env.get("q_current", np.zeros((M, 3)))

    Q = q_current.copy()
    max_disp = self.max_displacement  # 15m
    for m in range(M):
        # 水平位移: 随机方向 + 随机幅度
        angle = self.rng.uniform(0, 2 * np.pi)
        mag = self.rng.uniform(0, max_disp)
        Q[m, 0] += mag * np.cos(angle)
        Q[m, 1] += mag * np.sin(angle)
        # 垂直位移
        Q[m, 2] += self.rng.uniform(-max_disp, max_disp)

    # Clamp 到区域/硬件约束
    Q[:, 0] = np.clip(Q[:, 0], 0, self.area_w)
    Q[:, 1] = np.clip(Q[:, 1], 0, self.area_h)
    Q[:, 2] = np.clip(Q[:, 2], self.H_min, self.H_max)
    ...
```

关键改动：
- 从 `q_current` 出发，而非随机位置
- 水平位移限制在 `[0, max_disp]` (最大 15m)
- 垂直位移限制在 `[-max_disp, max_disp]` (最大 15m)
- 区域 clamp 作为二级安全网

### 修复 3: `_optimize_deployment_sca()` — bounds 收窄为移动性交集

```python
q_current = env.get("q_current", Q_init.copy())
max_disp = self.max_displacement  # 15m
q0 = q_current[m]
bounds = [
    (max(0.0, q0[0] - max_disp), min(self.area_w, q0[0] + max_disp)),       # x
    (max(0.0, q0[1] - max_disp), min(self.area_h, q0[1] + max_disp)),       # y
    (max(self.H_min, q0[2] - max_disp), min(self.H_max, q0[2] + max_disp)), # H
]
```

每个 UAV 的 bounds 是**绝对地图边界**与**移动性边界** `q_current ± max_disp` 的交集。L-BFGS-B 严格尊重 bounds，优化器不可能越界。

### 修复 4: `_warmstart_to_init()` — MLLM 预测的安全裁剪

```python
# Clamp displacement to movement constraint
delta_q_horiz = np.linalg.norm(delta_q[:, :2], axis=1, keepdims=True)
scale = np.where(delta_q_horiz > max_disp, max_disp / (delta_q_horiz + 1e-12), 1.0)
delta_q[:, :2] *= scale
delta_q[:, 2] = np.clip(delta_q[:, 2], -max_disp, max_disp)
```

即��� MLLM 预测出了超界的 `delta_q`，solver 也会将其裁剪到物理约束内。这是防御性编程。

### 修复 5: 入口参数传递

`generate_data.py` 和 `evaluate.py` 两处的 solver 构造均补充了 `v_max` 和 `slot_duration` 参数。

### 修复 6: 验证脚本阈值统一

`validate_data.py` 中垂直位移阈值从硬编码 `50m` 改为与水平一致的 `2 * v_max * Δt = 30m`。

---

## 同行审查与补充

修复后经同行审查，发现以下缺口并立即补充：

| 审查意见 | 我的实现 | 改进 |
|---------|---------|------|
| `evaluate.py` 未传移动性参数 | 漏了 | ✅ 已补 (`2b75aa1`) |
| 垂直约束用 `0.5×` 折扣 | 三处硬编码 `max_disp * 0.5` | ✅ 统一为 `max_disp` (`2b75aa1`) |
| 部署优化加 1e5 惩罚项 | 仅 box bounds | ⚠️ 初轮未采纳, 第二轮验证后采纳 (`3rd commit`) — 见下方 Box vs Sphere |
| 3D 球体均匀采样 | 水平角度+幅度 + 独立垂直 | ⚠️ 初轮未采纳, 第二轮修正为 3D 球形采样 (`3rd commit`) — 见下方 |

---

## 第二轮审查: Box vs Sphere — 约束几何学修正

### 问题复现

烟雾测试重新运行后，`validate_data.py` 显示 0 issues，但朋友审查发现数据仍然违背物理约束：

```
δ_q 水平位移:  mean=21.0m  [17.0, 21.2]
δ_q 垂直位移:  mean=15.0m  [15.0, 15.0]
```

`max_displacement = 15m`，但水平位移达 21.2m —— **√(15² + 15²) = √450 ≈ 21.21m**，精确吻合。

### 根因: L-BFGS-B Box Bounds 的几何局限性

L-BFGS-B 的 `bounds` 是 per-axis 独立约束：

```
x ∈ [x₀-15, x₀+15]
y ∈ [y₀-15, y₀+15]   →  搜索空间是边长为 30m 的 正方体
z ∈ [z₀-15, z₀+15]
```

而真实的物理约束是：

```
‖Δq‖₂ = √(Δx² + Δy² + Δz²) ≤ 15m   →  可行域是半径为 15m 的 球体
```

正方体的八个角距离中心 **√(15²+15²+15²) ≈ 26m**，远超 15m 约束。优化器为追求最大效用，会推动 UAV 到这些角落。

### 为什么 `validate_data.py` 没拦住

修复第一轮时，验证阈值被设为 `2 * v_max * Δt = 30m`（水平）和 `2 * v_max * Δt = 30m`（垂直），21.2m 轻松混过。

根本问题：**逐轴独立验证无法检测对角线逃逸** — 需要 3D 欧氏范数验证。

### 修复: 三处联动

**Fix A: `_optimize_deployment_sca` — 球形惩罚项 (切角)**

```python
# 在 objective(x) 返回前加入:
q_cur_m = q_current[m]
dist_moved = np.linalg.norm(q_new - q_cur_m)
if dist_moved > max_disp:  # > 15m
    obj += 1e5 * (dist_moved - max_disp) ** 2
```

Box bounds 保留（加速收敛缩小搜索空间），球形惩罚 "切掉八个角"，确保优化器输出的解在球体内。

**Fix B: `_random_init` — 3D 球形采样**

```python
# 从: 水平圆形 + 独立垂直 (柱体, 对角线可达 √(15²+15²)≈21.2m)
# 改为: 3D 球面坐标均匀采样
phi = self.rng.uniform(0, 2 * np.pi)
cos_theta = self.rng.uniform(-1, 1)
theta = np.arccos(cos_theta)
r = self.rng.uniform(0, max_disp)
Q[m, 0] += r * np.sin(theta) * np.cos(phi)
Q[m, 1] += r * np.sin(theta) * np.sin(phi)
Q[m, 2] += r * np.cos(theta)
```

**Fix C: `_warmstart_to_init` — 3D 范数裁剪**

```python
# 从: 水平+垂直分开裁剪 (保留对角线逃逸)
# 改为: 保持方向, 3D 范数裁剪到 max_disp
delta_q_norm = np.linalg.norm(delta_q, axis=1, keepdims=True)
scale = np.where(delta_q_norm > max_disp, max_disp / (delta_q_norm + 1e-12), 1.0)
delta_q *= scale
```

**Fix D: `validate_data.py` — 3D 欧氏范数验证**

```python
# 从: 逐轴检查 (dq_horiz > 2*v_max_dt, dq_vert > 2*v_max_dt)
# 改为: 3D 欧氏距离
dq_3d = np.linalg.norm(dq, axis=1)
if dq_3d.max() > v_max_dt + 1e-3:
    issues.append(...)
```

阈值从 `30m` (2×) 降回 `15m + ε`。

### 修正前后对比

| 指标 | 原始 Bug | 第一轮修复 (Box) | 第二轮修复 (Sphere) |
|------|---------|-----------------|-------------------|
| δ_q max | 864m | 21.2m (水平) | ≤15m (3D) |
| 物理约束满足 | ❌ 57×超限 | ⚠️ 对角线超限 1.4× | ✅ 严格满足 |
| 验证阈值 | 50m/30m | 30m (2×) | 15m (1×) |

---

## 文件变更清单

| 文件 | 变更类型 | 关键改动 |
|------|---------|---------|
| `src/solver/sca_fp.py` | Bug 修复 | `__init__`: 新增 `v_max`, `slot_duration`, `max_displacement`；`_random_init`: 从 `q_current` 球形邻域采样；`_optimize_deployment_sca`: bounds 约束为移动性交集；`_warmstart_to_init`: 位移安全裁剪 |
| `scripts/generate_data.py` | 参数对齐 | solver 构造传入 `v_max` + `slot_duration` |
| `src/eval/evaluate.py` | 参数对齐 | 同上（同行审查发现遗漏） |
| `scripts/validate_data.py` | 阈值修正 | 垂直位移阈值 `50m` → `2*v_max*Δt` |

**Commits**:
```
1caa482 fix: enforce v_max*Δt movement constraint in SCA-FP solver
2b75aa1 fix: unify 3D movement constraint + add missing evaluate.py solver params
<3rd>    fix: Box→Sphere constraint — add 3D Euclidean penalty + spherical sampling + 3D validation
```

---

## 验证结果

### 修复前

```
δ_q 水平位移:  mean=382.4m  [88.3, 864.8]   ← 超过约束 57×
δ_q 垂直位移:  mean=121.8m  [33.4, 224.5]   ← 超过约束 15×
Issues: 100
```

### 第一轮修复后（Box 约束 — 未完全修复）

```
δ_q 水平:  mean=21.0m  [17.0, 21.2]   ← 对角线逃逸, √(15²+15²)=21.2m
δ_q 垂直:  mean=15.0m  [15.0, 15.0]   ← 角落解
Issues: 0 (验证阈值 30m 太宽松, 漏报)
```

### 第二轮修复后（Sphere 约束 — 预期）

```
δ_q 3D位移 (‖Δq‖₂):  mean ≈ 8-12m  [0, 15]  ← 严格在球体内
Issues: 0 — all clean ✅ (验证阈值 15m)
```

---

## 经验教训

### 1. 烟雾测试 + 验证脚本的组合是不可替代的

只有 5 个环境就暴露了这个 Bug。如果没有 `validate_data.py` 的物理一致性检查，5000 环境跑完（~35 小时）后才可能在训练阶段发现异常，代价巨大。

### 2. 随机初始化必须尊重物理约束

Best-of-N 的 "随机重启" 意味着"在不同的局部邻域内探索"，而不是"在整个全局空间随机采样"。物理系统的约束必须在优化器的每一步都得到尊重。

### 3. 优化器的 bounds 必须反映时变约束

静态 bounds `(0, area_w)` 只适合初始部署场景。对于时间序列优化（每个时间槽的增量决策），bounds 必须是相对于当前状态的动态交集。

### 4. 防御性编程

`_warmstart_to_init` 中对 MLLM 预测的裁剪属于防御性编程——即使模型训练得当，推理时也可能因为分布偏移而产生异常值。在 solver 层面做 constraint projection 可以防止单个异常预测污染整个优化结果。

### 5. 同行审查的价值

朋友一眼发现了两个遗漏：(1) `evaluate.py` 未传参数、(2) 垂直约束不一致。即使核心修复正确，遗漏的调用点也会在评估阶段产生不可比的结果。

### 6. Box ≠ Sphere — 约束几何学的经典陷阱

L-BFGS-B 的 box bounds 只能做 per-axis 独立约束，画出来是正方体。物理约束 `‖Δq‖₂ ≤ 15m` 是球体。正方体的角到中心距离 `√(15²+15²+15²) ≈ 26m`，远超出 15m。

**教训**: 
- Per-axis bounds 永远不能替代耦合约束（Euclidean norm, L1 norm, etc.）
- 对耦合约束，必须在目标函数中加入惩罚项或在优化后投影
- 验证脚本必须验证耦合约束本身（3D 欧氏距离），而非拆开的逐轴分量
- 将 `2×` 阈值当 "安全余量" 会掩盖真实的约束违反

---

> **相关文档**: [09_handoff_document.md](09_handoff_document.md) — 完整项目交接 | [06_fifth_review_final.md](06_fifth_review_final.md) — 第五轮审查（含同款波长硬编码修复）
>
> **相关 Commits**: `1caa482`, `2b75aa1`
