---
type: postmortem
status: resolved
severity: P0
stage: eval
commits: [f62f783]
last_updated: 2026-06-28
related: [phase1_status_2026-06-26, training_code_bugs]
---

# Bug: Eval Pipeline 审查 — 7 处缺陷闭合

**来源**: 系统性代码审查 eval 全链路 (`eval_generation.py` + `evaluate.py` + `sca_fp.py`) | **发现者**: Claude code review

## 审查范围

```
scripts/eval_generation.py   — 轻量 checkpoint 质量检查 (Part 1-3)
src/eval/evaluate.py         — 完整评估管线 (200 samples, 6 metrics)
src/solver/sca_fp.py         — SCA-FP 求解器 (warmstart / cold-start)
docs/00_current/status.md    — 当前状态文档 (评估命令)
```

---

## B1 (P0): `eval_generation.py` 缺少 `tqdm` import — 必崩

### 症状

Part 3 批量 SCA-FP 循环在执行到 `tqdm(range(n_scafp))` 时抛出 `NameError`，整个评估脚本崩溃。

### 根因

文件顶部 import 块缺少 `from tqdm import tqdm`。对比 `evaluate.py` 第 24 行有该 import，但 `eval_generation.py` 漏掉了。

### 修复

```python
# 添加
from tqdm import tqdm
```

### 教训

**新增依赖必须在所有使用该依赖的文件中独立 import**。不能假设"另一个文件 import 了就行"——Python 没有跨文件 import 共享。

---

## B2 (P1): `evaluate.py` 模型从未移到 CUDA — CPU 推理

### 症状

`from_pretrained` 加载模型后，没有调用 `.to("cuda")`。`device = next(model.parameters()).device` 返回 CPU，所有推理在 CPU 上执行。200 个样本 × 每次推理 ~3s(CPU) ≈ 10 分钟+，而 GPU 只需 ~30 秒。

### 根因

`eval_generation.py` 有显式 `model = model.to("cuda")` (第 89 行)，但 `evaluate.py` 漏掉了这行。两个文件的模型加载路径不一致。

### 修复

在 `evaluate.py` 第 124 行 `model.eval()` 前添加：

```python
model = model.to("cuda")
```

### 教训

**模型加载后永远显式指定设备**。依赖 `from_pretrained` 的默认行为 (CPU) 是危险的——它在本地测试时不会报错，只会慢得不可接受。

---

## B3 (P1): `evaluate.py` 无 cold-start 基线 — 算不出加速比

### 症状

`_evaluate_one_sample` 只运行 `solver.solve(warm_start=warm_start_dict)`，记录 `sca_fp_iterations`。但 SCA-FP 加速比需要 warmstart vs cold-start 对比才有意义——仅有 warmstart 迭代次数无法判断模型是否真正加速了求解器。

### 根因

最初设计时 `evaluate.py` 只关注 6 个顶层指标（sum rate、SINR 等），未将加速比列为独立指标。Phase 1 评估的核心判据恰好是加速比，暴露了这个设计缺口。

### 修复

- 添加 `sol_cold = solver.solve(env_dict, warm_start=None, seed=sample_id)`
- `results` 字典拆分为 `sca_fp_iterations_warm`、`sca_fp_iterations_cold`、`sca_fp_speedup` 三个指标
- `speedup = sol_cold.iterations / max(sol_warm.iterations, 1)`

### 教训

**评估指标必须与实验目标对齐**。Phase 1 的目标是"warmstart 加速 SCA-FP"，但评估管线没测加速比——这是典型的指标-目标失配。

---

## B4 (P2): `eval_generation.py` 异常捕获范围过大

### 症状

Part 3 的整个 try/except 块捕获所有异常 (`except Exception as e`)，消息写死 "solver import failed"。如果 SCA-FP 求解器在运行时抛出数值错误（NaN、shape mismatch），会被静默吞掉并打印误导信息。

### 根因

原始代码用一个大 try/except 包裹 Part 3 全部逻辑（import → solver setup → loop → print），意图是优雅处理 solver 不可用的情况。但实际上 `ImportError` 和其他 `Exception` 应该区别对待。

### 修复

```python
# 拆分: ImportError → 早返回; 运行时异常 → 让 Python 自然抛出完整 traceback
try:
    from src.solver import SCAFPOptimizer, SCAFPConfig
except ImportError as e:
    print(f"  Skipped (solver import failed): {e}")
    return
# Part 3 主逻辑不再被 try/except 包裹
```

### 教训

**try/except 的作用域要尽可能小，异常类型要尽可能具体**。宽泛的 `except Exception` + 硬编码消息 = 静默吞错 + 误导调试方向。

---

## B5 (P2): `sca_fp.py` NaN 退出时 `converged` 标志错误

### 症状

SCA-FP 求解器在 utility 变为 NaN 时通过 `if not np.isfinite(utility): break` 退出迭代循环。但 `converged = (outer_iter + 1 < max_outer_iters)` 在 NaN 退出时返回 `True`（因为 `outer_iter < max_iters - 1`），错误地将发散标记为"收敛"。

### 根因

`converged` 只检查了"是否早于 max_iters 退出"，没有检查"退出原因是否为收敛"。NaN 退出和 tol 收敛退出共用同一个 `break` 后的代码路径。

### 修复

```python
converged=(outer_iter + 1 < self.cfg.max_outer_iters) and np.isfinite(utility),
utility=utility if np.isfinite(utility) else -np.inf,
```

### 教训

**收敛标志必须显式区分退出原因**。任何 `break` 语句都应该在后续代码中留下可追溯的状态——布尔标志或 sentinel 值。

---

## B6 (P3): `eval_generation.py` 冗余 `.detach()` 调用

### 症状

```python
"delta_q": ws["delta_q"].detach().numpy(),
```

`generate_warmstart` 整个方法体在 `torch.no_grad()` 上下文中执行，返回的 tensor 已无梯度图。`.detach()` 无害但误导——暗示此处可能有梯度泄露风险。

### 修复

直接 `.numpy()`。

---

## B7 (P3): `status.md` 评估命令路径错误

### 症状

`status.md` 两处批量评估命令引用 `src/eval/eval_generation.py`，但文件实际位于 `scripts/eval_generation.py`。命令直接复制粘贴会报 `No such file or directory`。

### 修复

路径改为 `scripts/eval_generation.py`，同时去掉不存在的 `--output` 参数。

---

## 影响

如果不修复:
- **B1**: `eval_generation.py` Part 3 在第一个 `tqdm` 调用处崩溃，无法获取 SCA-FP 加速比数据
- **B2**: `evaluate.py` 200 样本评估在 CPU 上耗时 ~10 分钟而非 ~30 秒
- **B3**: `evaluate.py` 无法回答"模型是否加速了 SCA-FP"这一核心问题
- **B4**: 运行时错误被静默吞掉，错误消息指向完全错误的方向
- **B5**: 如果后续代码依赖 `converged` 字段过滤发散解，NaN 解会被错误纳入统计

## 教训总结

1. **新增依赖必须在每个使用文件中独立 import** (B1)
2. **模型加载后永远显式指定设备** (B2)
3. **评估指标必须与实验目标对齐** (B3)
4. **try/except 作用域最小化，异常类型具体化** (B4)
5. **收敛标志显式区分退出原因** (B5)
6. **文档中的命令路径必须在真实环境中验证过** (B7)
