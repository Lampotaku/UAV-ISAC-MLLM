# 第五轮终审报告 — 上线前全面代码审查

> 审查范围：全项目 18 个源文件 + 1 个配置文件
> 审查方法：逐文件阅读，公式对照论文，数据流追踪
> 状态：已完成 4 轮修复，本轮发现 3 个 P0 + 4 个 P1 + 5 个 P2

---

## 已确认修复（前四轮累积）

| 轮次 | 问题数 | 关键修复 |
|------|--------|----------|
| Codex 一审 | 9 fixes | control token插入、loss mask、q_current数据流、Unsloth加载 |
| 二审 | 9 issues | 识别了路径损耗冲突、RNG泄漏、user_weights、波长硬编码等 |
| Gemini 三审 | 6 fixes | DPO log-prob求和、off-by-one切片、3D移动性、P_min功率、满意度分母、N_t·N_r |
| 四审 | 2 fixes | DPO deepcopy OOM、channel_gain RNG确定性 |

**当前代码库状态**：核心训练管线已闭合，18个源文件语法检查通过，数据流从 `ISACScenarioGenerator → OracleDataGenerator → Dataset → train_sft/dpo` 完整贯通。

---

## P0 — 会导致数值结果错误或评估指标不可信

### P0-1: `sca_fp.py` 两处硬编码波长 `0.0517`

**文件**: [src/solver/sca_fp.py](src/solver/sca_fp.py)

**位置 1** — `_optimize_deployment_sca()` 第 350 行（感知路径损耗）:
```python
pl_db = 20 * np.log10((4 * np.pi * dist_3d) / 0.0517) + 20
```

**位置 2** — `_compute_utility()` 第 471 行（感知效用计算）:
```python
pl_db = 20 * np.log10((4 * np.pi * dist_3d) / 0.0517) + 20
```

**问题**: `0.0517 = 3e8 / 5.8e9` 是当 `carrier_freq_ghz = 5.8` 时的波长。如果修改配置中的载波频率（如换成 2.4GHz 或 28GHz），SCA-FP 求解器会静默使用错误的路径损耗，导致优化方向偏差。`evaluate.py` 已经修复了同款问题（第 238 行动态计算），但 `sca_fp.py` 内部仍然硬编码。

**影响**: 当前默认配置 `carrier_freq_ghz: 5.8` 下恰好正确，但切换频率后 SCA-FP 的感知项全部算错。训练数据 `generate_data.py` 依赖 SCA-FP 生成 oracle，所以换频率后整个数据集的感知质量不可信。

**修复建议**: 在 `SCAFPOptimizer.__init__` 中计算 `self.wavelength`，两处替换为 `self.wavelength`。`ISACChannel` 已经做了这件事（第 54 行），solver 应保持一致。

---

### P0-2: SCA-FP 内部三条路径损耗公式互不一致

**文件**: [src/solver/sca_fp.py](src/solver/sca_fp.py)

SCA-FP 的三个子模块使用了三种不同的路径损耗模型：

| 子模块 | 行号 | 公式 | 用途 |
|--------|------|------|------|
| `_optimize_beamforming` | 288 | 直接用 env 传入的 `channel_gains`（来自 `ISACChannel.channel_gain()`，含概率 LoS/NLoS + Rician 衰落） | 波束功率分配 |
| `_optimize_deployment_sca` | 339 | `pl_db = 28 + 22*log10(d_3D)` — 简化版，无频率项 | UAV 位置优化 |
| `_compute_utility` | 471 | `pl_db = 20*log10(4πd/0.0517) + 20` — 双程雷达方程 | 效用评估 |

**问题**: 部署优化子问题（`_optimize_deployment_sca`）使用简化路径损耗寻找 UAV 位置，但效用评估（`_compute_utility`）用不同的公式评判结果，波束成形又用第三种。这意味着 **SCA-FP 在优化一个目标函数，但收敛判定和 DPO 排序用的是另一个目标函数**。

具体来说：
- `_optimize_deployment_sca` 第 339 行：`pl_db = 28 + 22*np.log10(max(dist_3d, 1.0))` — 没有 `20*log10(f_c)` 项。在 5.8GHz 下，`20*log10(5.8) ≈ 15.3dB` 的差异被忽略。
- `_compute_utility` 第 471 行：用了双程雷达方程，但常数项 `+20` 与 `ISACChannel.compute_sensing_sinr` 第 199 行的 `+20` 一致——但这与 `_optimize_deployment_sca` 中的感知目标函数不一致。

**影响**: 外循环中交替优化依赖一致的效用评价来决定收敛。公式不一致意味着：
1. 位置优化朝着错误的方向移动 UAV（因为梯度来自错误的路径损耗）
2. 收敛判据 `abs(utility - prev_utility) < tol` 在比较不同公式算出的值
3. Best-of-N 的排序依据（`_compute_utility`）与优化过程（`_optimize_deployment_sca`）脱节

**修复建议**: 统一使用 `ISACChannel` 的路径损耗模型。最佳方案是让 `SCAFPOptimizer` 持有一个 `ISACChannel` 实例（或至少复用其 `path_loss_db()` 和 `compute_sensing_sinr()` 方法），而不是在 solver 内部重新实现简化版。

---

### P0-3: `evaluate.py` 未向 solver 传入 `noise_power`，导致评估指标系统偏低

**文件**: [src/eval/evaluate.py](src/eval/evaluate.py) 第 76-86 行

```python
solver = SCAFPOptimizer(
    config=solver_cfg,
    M=sim_cfg["num_uavs"],
    K=sim_cfg["num_users"],
    T=sim_cfg["num_targets"],
    N_t=sim_cfg["num_antennas_tx"],
    # ← 缺少 N_r
    # ← 缺少 noise_power
    ...
)
```

**对比**: `generate_data.py` 第 147 行正确计算并传入：
```python
noise_power=10 ** ((-174 + 10 * np.log10(sc["bandwidth_mhz"] * 1e6) + sc["noise_figure_db"] - 30) / 10),
```

**问题**: `SCAFPOptimizer.__init__` 的默认 `noise_power = 1e-12` W（约 -90 dBm）。正确的热噪声（20MHz + 9dB NF）约为 `6.3e-13` W（约 -92 dBm）。虽然差距只有 ~2 dB，但：
- 评估脚本的 sum-rate、sensing SINR、satisfaction 全部基于 solver 优化后的解计算
- Solver 用错误的 noise floor 做波束功率分配和关联优化 → 产生次优解
- 论文中的数值结果将不可复现

**影响**: 如果用户用 `evaluate.py` 跑 baselines 对比，所有方法的绝对值都偏高（噪声被低估），但相对排名可能变化（不同方法对噪声敏感度不同）。

---

## P1 — 边缘情况下行为异常或配置灵活性问题

### P1-1: `evaluate.py` 和 `generate_data.py` 均未传 `N_r` 给 solver

**文件**: 
- [src/eval/evaluate.py:76-86](src/eval/evaluate.py#L76-L86)
- [scripts/generate_data.py:138-149](scripts/generate_data.py#L138-L149)

两个脚本都传了 `N_t=sim_cfg["num_antennas_tx"]` 但没传 `N_r=sim_cfg["num_antennas_rx"]`。`SCAFPOptimizer.__init__` 的默认是 `N_r = N_t`，恰好当前配置 `num_antennas_tx=8, num_antennas_rx=8`，所以结果正确。

**修复**: 显式传入 `N_r=sim_cfg["num_antennas_rx"]`。防御性编程 —— 如果后续实验需要非对称阵列（如 N_t=8, N_r=4），现在的代码会静默错误。

---

### P1-2: `evaluate.py` 硬编码带宽 `20e6`

**文件**: [src/eval/evaluate.py:230](src/eval/evaluate.py#L230)

```python
sum_rate += 20e6 * np.log2(1 + sinr)  # B=20MHz
```

应改为：
```python
sum_rate += cfg["simulation"]["bandwidth_mhz"] * 1e6 * np.log2(1 + sinr)
```

与 P0-1 同款问题 —— 带宽已存在于配置中但被硬编码覆盖。

---

### P1-3: SCA-FP 收敛检查缺少 NaN 守卫

**文件**: [src/solver/sca_fp.py:161-163](src/solver/sca_fp.py#L161-L163)

```python
if abs(utility - prev_utility) < self.cfg.tol:
    break
prev_utility = utility
```

如果因数值问题 `utility` 变为 `NaN`，`abs(NaN - prev_utility)` 也是 `NaN`，`NaN < tol` 为 `False`，循环继续。更糟的是，`prev_utility = NaN` 后下一轮继续产生 NaN，30 轮外循环全部跑完才退出 —— 浪费计算且返回垃圾解。

**修复**:
```python
if not np.isfinite(utility):
    break
if abs(utility - prev_utility) < self.cfg.tol:
    break
```

---

### P1-4: `_compute_control_loss` 中 BCE 的数值稳定性

**文件**: [src/model/losses.py:73](src/model/losses.py#L73)

```python
loss_a = F.binary_cross_entropy(delta_hat["delta_a"], delta_target["delta_a"])
```

`F.binary_cross_entropy` 内部会 clamp `input` 到 `[1e-7, 1-1e-7]` 防止 log(0)。Sinkhorn 输出理论上在 [0,1] 内，但经过 MLP 残差修正后 `delta_a_raw` 可能超出此范围。`AssociationProjection.forward` 通过 `torch.exp(a_tilde / self.tau)` 保证了非负，Sinkhorn 迭代保证了行/列归一化，所以输出确实在 [0,1] 内。当前实现安全，但建议在 `_unflatten` 后对 `da`（未经投影的原始关联得分）在使用前做显式检查。

---

## P2 — 代码质量 / 可维护性 / 次要问题

### P2-1: 两个数据生成脚本并存，功能重叠

| 脚本 | 断点续跑 | 增量保存 | Ctrl+C安全 | 进度显示 |
|------|----------|----------|------------|----------|
| `scripts/generate_data.py` | ✅ | ✅ | ✅ | 定期打印 |
| `scripts/run_data_generation.py` | ❌ | ❌ (批量) | ❌ | tqdm |

文档（`01_project_overview.md`）和论文指明使用 `generate_data.py`。`run_data_generation.py` 是早期版本，中途崩溃会丢失所有进度。建议：
- 删除 `run_data_generation.py` 或添加 deprecation warning
- 避免用户误用导致浪费算力

---

### P2-2: `Gemma3ISAC.from_pretrained` 绕过 `__init__`

**文件**: [src/model/gemma_isac.py:349-357](src/model/gemma_isac.py#L349-L357)

```python
instance = cls.__new__(cls)
nn.Module.__init__(instance)
instance.base_model = base_model
instance.tokenizer = tokenizer
...
```

这个模式使 `from_pretrained` 与 `__init__` 完全解耦。如果未来有人在 `__init__` 中添加新属性（如 `self.use_4bit`），`from_pretrained` 加载的模型会缺少该属性，导致 `AttributeError`。

**修复建议**: 在 `from_pretrained` 末尾加一个 sanity check，确保所有 `__init__` 设置的属性都在 instance 上存在。或者重构为让 `__init__` 支持一个 `skip_load` 参数。

---

### P2-3: `mean_crb` 仍返回占位值

**文件**: [src/eval/evaluate.py:286](src/eval/evaluate.py#L286)

```python
"mean_crb": 0.0,  # 需要 channel.compute_crb
```

CRB（Cramér-Rao Bound）是论文中的核心感知指标之一。当前评估结果中 `mean_crb` 始终为 0。`ISACChannel.compute_crb()` 已经实现，只需要在 `_evaluate_one_sample` 中调用。

---

### P2-4: `train_sft.py` 可训练参数收集存在潜在重复

**文件**: [src/training/train_sft.py:127-135](src/training/train_sft.py#L127-L135)

```python
trainable_params = [
    p for n, p in model.named_parameters()
    if p.requires_grad and "projection_head" in n
]
trainable_params += [
    p for n, p in model.base_model.named_parameters()
    if p.requires_grad
]
```

两段列表推导可能捕获到重叠的参数（如果某个 base_model 参数名字恰好包含 "projection_head"）——实际不会，但逻辑脆弱。建议统一为 `[p for p in model.parameters() if p.requires_grad]`，依赖 Unsloth 已经正确冻结了非 LoRA 参数。

---

### P2-5: `use_multimodal: false` — 仍是文本版

**文件**: [configs/default.yaml:51](configs/default.yaml#L51)

当前 `use_multimodal: false`，BEV 使用 10×10 文本网格而非图像。从工程角度看这不影响训练管线验证，但从论文复现角度看，这是核心差异。不算 bug，但应明确列为 TODO。

---

## 汇总

### 按优先级

| 级别 | 数量 | 关键项 |
|------|------|--------|
| P0 | 3 | 路径损耗不一致（3条公式 + 硬编码波长）、evaluate.py 缺 noise_power |
| P1 | 4 | 缺 N_r 传入、带宽硬编码、NaN 守卫、BCE 边界 |
| P2 | 5 | 双脚本、from_pretrained 脆弱、CRB 占位、参数收集、多模态未实现 |

### 按文件

| 文件 | P0 | P1 | P2 |
|------|-----|-----|-----|
| [src/solver/sca_fp.py](src/solver/sca_fp.py) | 2 | 1 | — |
| [src/eval/evaluate.py](src/eval/evaluate.py) | 1 | 2 | 1 |
| [src/model/losses.py](src/model/losses.py) | — | 1 | — |
| [src/model/gemma_isac.py](src/model/gemma_isac.py) | — | — | 1 |
| [src/training/train_sft.py](src/training/train_sft.py) | — | — | 1 |
| [scripts/run_data_generation.py](scripts/run_data_generation.py) | — | — | 1 |
| [configs/default.yaml](configs/default.yaml) | — | — | 1 |

### 与前一报告（04_third_review_gemini.md）的关系

Gemini 的报告覆盖了 DPO log-prob、control token 切片、3D 移动性、P_min、满意度分母、N_t·N_r。本轮关注的是 **SCA-FP 求解器内部一致性**和**评估管线数值准确性**，属于更深层的审查。

---

## 上线前建议修复顺序

1. **P0-1 + P0-2**：重构 `sca_fp.py` 统一路径损耗（使用 `ISACChannel` 或至少统一公式）
2. **P0-3**：`evaluate.py` 传入正确的 `noise_power`
3. **P1-1 + P1-2**：`evaluate.py` 传入 `N_r`，带宽从配置读取
4. **P1-3**：加 NaN 守卫
5. **P2-1**：删除或 deprecated `run_data_generation.py`
6. 以上修完后重新生成 smoke test 数据验证

**总评**：四轮修复后代码库已大幅改善。剩余 3 个 P0 集中在 SCA-FP 求解器的物理层建模一致性上 —— 这些问题在默认配置下不会暴露（5.8GHz、N_t=N_r=8），但一旦调整实验参数就会导致不可信的数值结果。建议在上 5000 环境数据生成前修复。
