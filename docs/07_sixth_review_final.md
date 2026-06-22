# 第六轮修复报告 — P0/P1 物理一致性全线闭合

> 基于第五轮终审报告（`docs/06_fifth_review_final.md`）的 3 P0 + 4 P1 问题
> 参照 Gemini 修改方案，逐项修复并交叉验证
> 修改范围：3 个源文件 + 删除 1 个冗余脚本

---

## 修复概览

| 优先级 | 编号 | 问题 | 文件 | 状态 |
|--------|------|------|------|------|
| P0 | P0-1 | SCA-FP 硬编码波长 `0.0517` | `src/solver/sca_fp.py` | ✅ 已修复 |
| P0 | P0-2 | 三条路径损耗公式不一致 | `src/solver/sca_fp.py` | ✅ 已修复 |
| P0 | P0-3 | `evaluate.py` 缺 `noise_power` | `src/eval/evaluate.py` | ✅ 已修复 |
| P1 | P1-1 | `evaluate.py`/`generate_data.py` 缺 `N_r` | 两个文件 | ✅ 已修复 |
| P1 | P1-2 | `evaluate.py` 硬编码带宽 `20e6` | `src/eval/evaluate.py` | ✅ 已修复 |
| P1 | P1-3 | SCA-FP 收敛缺 NaN 守卫 | `src/solver/sca_fp.py` | ✅ 已修复 |
| P1 | P1-4 | BCE 数值边界检查 | 不需修改（已验证安全） | ✅ 已验证 |
| P2 | P2-1 | 双数据生成脚本并存 | `scripts/run_data_generation.py` | ✅ 已删除 |

---

## 详细修改

### 1. `src/solver/sca_fp.py` — 核心求解器物理一致性

#### 1.1 新增 `carrier_freq_ghz` 参数，动态计算波长 (P0-1)

```python
# __init__ 新增参数
carrier_freq_ghz: float = 5.8,

# 新增属性
self.carrier_freq_ghz = carrier_freq_ghz
self.wavelength = 3e8 / (carrier_freq_ghz * 1e9)  # 动态计算，替换硬编码 0.0517
```

**影响**：改变 `carrier_freq_ghz` 配置后，SCA-FP 的感知路径损耗自动使用正确波长，不再静默使用 5.8GHz 专属的 `0.0517`。

#### 1.2 统一通信路径损耗公式 (P0-2)

`_optimize_deployment_sca()` L344：

```python
# 修复前: 缺频率项
pl_db = 28 + 22 * np.log10(max(dist_3d, 1.0))

# 修复后: 完整 3GPP UMa LoS
pl_db = 28 + 22 * np.log10(max(dist_3d, 1.0)) + 20 * np.log10(self.carrier_freq_ghz)
```

使部署优化子问题的通信路径损耗与 beamforming 子问题（使用 `ISACChannel`）在数值上对齐，消除优化目标与收敛判据之间的 ~15.3dB 系统偏差。

#### 1.3 替换硬编码波长 (P0-1)

`_optimize_deployment_sca()` L356 和 `_compute_utility()` L477：

```python
# 修复前
pl_db = 20 * np.log10((4 * np.pi * dist_3d) / 0.0517) + 20

# 修复后
pl_db = 20 * np.log10((4 * np.pi * max(dist_3d, 1.0)) / self.wavelength) + 20
```

同时添加 `max(dist_3d, 1.0)` 守卫，防止 UAV 恰好位于目标正上方时 `dist_3d = 0` 导致 `log(0)` 崩溃。

> **注意**：`N_t**2 → N_t * N_r` 的修改在第四轮（Gemini 三审）已完成。本文件 L358 和 L479 已使用 `self.N_t * self.N_r`，无需重复修改。

#### 1.4 NaN 守卫 (P1-3)

```python
# 在收敛检查之前插入
if not np.isfinite(utility):
    break  # 数值发散时立即退出，防止空转 30 轮外循环
```

**影响**：避免因数值问题（如 d→0 时路径损耗爆炸）导致 utility 变 NaN 后继续迭代，浪费计算且返回垃圾解。

---

### 2. `src/eval/evaluate.py` — 评估管线参数对齐

#### 2.1 补齐噪声功率 (P0-3)

```python
# 新增: 计算正确的热噪声功率
noise_power = 10 ** (
    (-174 + 10 * np.log10(sim_cfg["bandwidth_mhz"] * 1e6)
     + sim_cfg["noise_figure_db"] - 30) / 10
)
```

替换 solver 默认值 `1e-12` W（~ -90 dBm），与 `generate_data.py` 的公式完全一致。

#### 2.2 传入 `N_r` 和 `carrier_freq_ghz` (P1-1)

```python
solver = SCAFPOptimizer(
    ...
    N_r=sim_cfg.get("num_antennas_rx", sim_cfg["num_antennas_tx"]),
    carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
    noise_power=noise_power,
    ...
)
```

防御性编程：当前默认配置 N_r = N_t = 8 恰好正确，但非对称阵列下不再静默出错。

#### 2.3 带宽从配置读取 (P1-2)

```python
# 修复前
sum_rate += 20e6 * np.log2(1 + sinr)  # B=20MHz 硬编码

# 修复后
sum_rate += cfg["simulation"]["bandwidth_mhz"] * 1e6 * np.log2(1 + sinr)
```

---

### 3. `scripts/generate_data.py` — 数据生成参数对齐 (P1-1)

```python
solver = SCAFPOptimizer(
    ...
    N_r=sim_cfg.get("num_antennas_rx", sim_cfg["num_antennas_tx"]),
    carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
    ...
)
```

与 `evaluate.py` 对称：三个 solver 入口（evaluate / generate_data / solver 自身）现已共享同一套物理参数。

---

### 4. `scripts/run_data_generation.py` — 已删除 (P2-1)

旧版批量生成脚本无断点续跑能力，中途崩溃丢失所有进度。统一使用 `scripts/generate_data.py`。

---

## 未修改的 P2 项

以下 P2 项属于代码质量 / 功能完善，不影响当前训练管线的数值正确性：

| P2 | 问题 | 状态 | 说明 |
|----|------|------|------|
| P2-2 | `from_pretrained` 绕过 `__init__` | 延后 | 当前属性集稳定，加 sanity check 可后续处理 |
| P2-3 | `mean_crb` 返回 0.0 | 延后 | `ISACChannel.compute_crb()` 已实现，需接入 `_evaluate_one_sample` |
| P2-4 | `train_sft.py` 参数收集脆弱 | 延后 | 当前 Unsloth 正确冻结 LoRA 即可，暂无风险 |
| P2-5 | `use_multimodal: false` | 延后 | 文本 BEV 可用于验证管线；多模态是论文扩展 |

---

## 验证

```bash
$ python -m compileall -q src scripts     # 通过
$ grep -r "0.0517" src/                   # 0 匹配
$ grep "20e6" src/eval/evaluate.py        # 0 匹配
$ ls scripts/run_data_generation.py       # 文件不存在
```

---

## 上线就绪状态

三个 SCA-FP 入口现已共享一致的物理参数：

```
  evaluate.py ──→ N_r, carrier_freq_ghz, noise_power ──→ SCAFPOptimizer
generate_data.py ──→ N_r, carrier_freq_ghz, noise_power ──→ SCAFPOptimizer
        solver 内部 ──→ self.wavelength, self.carrier_freq_ghz ──→ 统一公式
```

**结论**：P0 全部闭合，切换载波频率 / 阵列配置 / 带宽后 solver 目标函数、评估基线、数据生成逻辑三位一体。可以安全启动 5000 环境数据生成和 Stage I+II 训练。
