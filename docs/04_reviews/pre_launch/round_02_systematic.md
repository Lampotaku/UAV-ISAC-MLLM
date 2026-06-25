# 第二轮审查报告 (Systematic Debugging)

审查方法：按 systematic-debugging 的 Phase 1 方法论，逐模块追踪数据流（Environment → Oracle → Solver → Dataset → Model → Training → Evaluation），在每个模块边界验证数据形状、语义正确性、数值一致性、随机性控制。

审查范围：`src/` 下全部 12 个源文件 + `scripts/generate_data.py` + `scripts/run_data_generation.py` + `configs/default.yaml`

---

## P0 — 会导致训练数据/评估结果静默错误

### 1. SCA-FP 求解器内部使用了 3 套不同的路径损耗公式

**文件**: `src/solver/sca_fp.py`

**根因追踪**:

| 位置 | 用途 | 公式 | 问题 |
|------|------|------|------|
| `_optimize_deployment_sca:337` | 通信目标函数 | `pl_db = 28 + 22*log10(d_3D)` | 无频率项，与信道模型不一致 |
| `_optimize_deployment_sca:348` | 感知目标函数 | `pl_db = 20*log10((4π·d)/0.0517) + 20` | 硬编码 0.0517 (5.8GHz 波长) |
| `_compute_utility:469` | 效用评估 | `pl_db = 20*log10((4π·d)/0.0517) + 20` | 同上，但与部署优化使用不同通信公式 |
| `uav_channel.py:100-105` | 实际信道 | LoS/NLoS 概率加权 + 频率相关自由空间参考 | 与其他两套完全不同 |

**后果**: SCA-FP 优化器对着一个错误的目标函数做优化，然后用另一个公式评估效用，再用第三个公式生成"ground truth"信道增益。最终选出的"最优解"可能并非真正最优。Best-of-N 排序的可靠性受质疑。

**修复建议**: 将 SCA-FP 内部所有路径损耗计算统一为 `ISACChannel` 的方法，或至少统一为同一套简化公式。不要在求解器里硬编码波长。

---

### 2. `channel_gain()` 绕过 seeded RNG，破坏可复现性

**文件**: `src/env/uav_channel.py:139-141`

**根因追踪**:
```python
# uav_channel.py:139-141
small_scale = np.abs(
    los_component + nlos_component * (
        np.random.randn() + 1j * np.random.randn()  # ← 全局 RNG!
    ) / np.sqrt(2)
) ** 2
```

`ISACScenarioGenerator` 使用 `self.rng = np.random.RandomState(seed)` 保证确定性采样，但 `ISACChannel.channel_gain()` 使用 `np.random.randn()` (全局 numpy RNG) 生成小尺度衰落。

**后果**:
- 同一个 `sample_id` 两次调用产生不同信道增益
- 断点续跑后，之前未保存的环境在 resume 时产生不同数据
- 无法精确复现训练数据集
- 任何调用 `np.random` 的代码（日志、数据增强、错误处理）都会扰动信道序列

**修复建议**: 将 `channel_gain()` 改为接受一个 `rng: np.random.RandomState` 参数，由 `ISACScenarioGenerator` 传入其 `self.rng`。

---

### 3. Oracle 数据生成时丢弃了用户权重，优化器对错误目标优化

**文件**: `src/data/oracle_generator.py:226-235`

**根因追踪**:
```python
# oracle_generator.py:233
"user_weights": np.ones(self.solver.K, dtype=np.float32),
```

`UAVNetwork` 在初始化用户时生成异构权重（`uav_network.py:129`: `weight = self.rng.uniform(0.5, 2.0)`），这些权重被保存在 `EnvironmentSample` 的 `comm_summary` 中，但**没有**传给 solver。

`_env_sample_to_dict()` 硬编码为全 1。而 SCA-FP 的效用函数和匈牙利关联都使用 `user_weights` 来决定优化方向。

**后果**: Oracle 对"所有用户等权重"的目标做优化，但论文设定是异构权重（模拟 IoT 设备的差异化 QoS 需求）。生成的 Best-of-N 排序和选出的"最优"prior 与论文设定不符。

**修复建议**: 将 `env_sample.user_weights` 传入 solver，或从 `UAVNetwork.get_state_dict()` 导出并在 `_env_sample_to_dict()` 中使用。

---

### 4. `evaluate.py` 有两处路径损耗计算，Codex 只修了一处

**文件**: `src/eval/evaluate.py:236-240`

**根因追踪**:

Codex 修了第 269 行（`joint_satisfaction` 部分），但第 240 行（`sum_rate` 部分的 sensing SINR）仍然硬编码：
```python
# evaluate.py:237-240 — 未修复
pl_db = 20 * np.log10((4 * np.pi * dist_3d) / 0.0517) + 20
```

对比已修复的第 269 行：
```python
# evaluate.py:269 — 已修复
wavelength = 3e8 / (cfg["simulation"]["carrier_freq_ghz"] * 1e9)
pl_db = 20 * np.log10((4 * np.pi * dist_3d) / wavelength) + 20
```

**后果**: 评估报告的 `mean_sensing_sinr_db` 指标使用硬编码波长，当配置中 `carrier_freq_ghz` 不是 5.8 时结果错误。

**修复建议**: 统一提取 `wavelength` 到函数顶部，两处共用。

---

## P1 — 训练中可能崩溃或行为异常

### 5. `copy.deepcopy(model)` 对 Unsloth 4-bit 模型的兼容风险

**文件**: `src/training/train_dpo.py:136`

**根因追踪**:
```python
ref_model = copy.deepcopy(model)
```

Unsloth 使用自定义 4-bit 量化内核（`bitsandbytes` 替代品，针对 Blackwell sm_120 优化）。`copy.deepcopy` 对量化张量的行为取决于底层实现：
- 如果 Unsloth 的 4-bit 张量正确实现了 `__deepcopy__`，会复制量化数据 → 显存翻倍
- 如果没有正确实现，可能复制底层未量化的 float 数据 → 显存爆炸
- 也可能在 deepcopy 过程中触发 CUDA 同步问题

**证据**: Unsloth 文档推荐的 reference model 创建方式是用 `FastLanguageModel.from_pretrained` 重新加载，而非 `deepcopy`。

**后果**: DPO 训练第一个 batch 可能 OOM（已在 Codex 报告的 "DPO 仍需单独小规模验证" 中提及但归因不准确——问题不在 reference model 本身占用，而在 deepcopy 对 4-bit 张量的行为不确定）。

**修复建议**: 用以下方式创建 reference model：
```python
# 方案 A: 重新加载 (安全但慢)
ref_model = Gemma3ISAC.from_pretrained(stage1_ckpt, base_model_name=...)
ref_model.eval()

# 方案 B: 只对 LoRA 权重做 deepcopy，base model 共享
# （需要更复杂的实现）
```

---

### 6. SCA-FP 收敛检测在 NaN 输入下无限循环

**文件**: `src/solver/sca_fp.py:133,159`

**根因追踪**:
```python
prev_utility = -np.inf          # line 133
# ...
if abs(utility - prev_utility) < self.cfg.tol:  # line 159
    break
prev_utility = utility
```

当 `utility` 为 `NaN`（例如由除零或 log(negative) 触发）：
- `abs(NaN - (-inf))` = `NaN`
- `NaN < tol` = `False` → 不退出
- `prev_utility = NaN`
- 下一轮: `abs(anything - NaN)` = `NaN` → 永远不退出
- 循环直到 `max_outer_iters`（30 次 × 50 次内循环 × 4 UAV = 6000 次 L-BFGS-B 调用）

**后果**: 遇到病态环境样本时，单次 SCA-FP 求解耗时从 ~0.5s 暴增到数十秒，5 个病态样本就可能导致数据生成卡住。

**修复建议**: 在 `_compute_utility` 返回后检查 `np.isfinite(utility)`。

---

### 7. DPO Dataset — `q_current` 和 `delta_*_target` 的返回条件不一致

**文件**: `src/data/dataset.py:172-178`

**根因追踪**:
```python
# DPODataset.__getitem__:172-178
if "delta_q" in item:
    result["q_current"] = torch.tensor(item.get("q_current", []), ...)
    result["delta_q_target"] = torch.tensor(item["delta_q"], ...)
    ...
```

如果 JSONL 中有 `delta_q` 但没有 `q_current`（旧格式数据），`item.get("q_current", [])` 返回空列表，`torch.tensor([], dtype=torch.float32)` 创建形状 `(0,)` 的张量。

训练代码 (`train_dpo.py:286`) 检查：
```python
if batch.get("q_current") is not None and batch["q_current"].numel() > 0:
```

这个条件对空张量返回 False，所以不会崩溃，但**静默跳过了 separation penalty**。问题在于没有任何 warning——训练者不知道数据有问题。

**修复建议**: 在 dataset 加载时验证 `q_current` 字段完整性，缺失时打印 warning 或直接报错。

---

## P2 — 代码质量 / 可维护性

### 8. 硬编码常量散布在多个文件中

| 常量 | 出现位置 | 应有来源 |
|------|---------|---------|
| `0.0517` (波长) | `sca_fp.py:348,469`, `evaluate.py:237` | `ISACChannel.wavelength` |
| `28 + 22*log10(d)` (路径损耗) | `sca_fp.py:337` | `ISACChannel.path_loss_db()` |
| `+ 20` (额外 dB) | `sca_fp.py:348,469`, `evaluate.py:237,269` | 应定义常量或参数化 |
| `0.7` / `0.3` (通信/感知功率比) | `sca_fp.py:292-295` | 应来自 config |
| `20 * np.log10(...)` | 多处 | 应封装为 `free_space_path_loss(d, wavelength)` |

**后果**: 修改配置中的 `carrier_freq_ghz` 不会传递到求解器内部，导致 solver 行为与配置不一致。这是一个"改动一处、另一处静默失效"的典型维护陷阱。

---

### 9. `generate_data.py` 的双击 Ctrl+C 提示是虚假的

**文件**: `scripts/generate_data.py:40`

```python
print("\n[INTERRUPT] Stopping after current environment... (Ctrl+C again to force quit)")
```

第二次 Ctrl+C 仍然调用同一个 handler（`_stop_requested = True` 已经是 True），不会强制退出。用户只能 `kill -9`。

---

## 与 Codex 报告的交叉验证

| Codex 发现 | 第二轮确认 | 备注 |
|-----------|-----------|------|
| #1 `Tuple` 导入 | ✅ 已修复 | |
| #2 control token IDs | ✅ 已修复 | |
| #3 labels mask | ✅ 已修复 | |
| #4 `q_current` 字段 | ✅ 已修复 | 但 P1-#7 指出 DPO dataset 的 guard 太宽松 |
| #5 训练传 `q_current` | ✅ 已修复 | |
| #6 DPO log-prob | ✅ 已修复 | |
| #7 Unsloth 两步加载 | ✅ 已修复 | 但 P1-#5 指出 `deepcopy` 仍有风险 |
| #8 eval 变量作用域 | ⚠️ 部分修复 | 修了 line 269，漏了 line 237-240 |
| #9 噪声功率公式 | ✅ 已修复 | |

---

## 优先级建议

| 优先级 | 问题 | 理由 |
|--------|------|------|
| **立即修复** | #2 信道随机性 | 破坏数据复现性，影响断点续跑正确性 |
| **立即修复** | #3 用户权重 | Oracle 优化目标与论文不一致，所有 SFT/DPO 数据受影响 |
| **训练前修复** | #1 路径损耗不一致 | SCA-FP 内部不一致，Best-of-N 排序可能不可靠 |
| **训练前修复** | #4 eval 第二处硬编码 | 评估结果错误 |
| **训练前修复** | #5 deepcopy Unsloth | DPO 训练可能第一天就 OOM |
| **择机修复** | #6 NaN 收敛 | 低频但影响大（卡住数据生成） |
| **择机修复** | #7 数据验证 | 静默问题难以发现 |
| **择机修复** | #8, #9 代码质量 | 不影响当前训练但降低维护性 |

---

## 总结

Codex 的 9 个修复主要集中在"代码能跑起来"层面（导入错误、张量形状、变量作用域）。本轮审查聚焦"跑出来的结果对不对"，发现了 4 个会导致训练数据和评估结果与论文设定不一致的问题：

1. **SCA-FP 内部路径损耗公式碎片化** — 优化器在不同步骤使用不同公式
2. **信道随机性不受控** — 数据集不可精确复现
3. **用户权重丢失** — Oracle 在错误的目标函数上做 Best-of-N
4. **评估脚本残留硬编码** — Codex 修了一处漏了一处

其中 #2 和 #3 意味着：**如果要重新生成 5000 环境数据，建议先把这两个问题修好再跑**，否则生成的数据集存在语义偏差。
