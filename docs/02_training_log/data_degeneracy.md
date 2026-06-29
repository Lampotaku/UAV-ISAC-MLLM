---
type: postmortem
status: resolved
severity: P0
stage: datagen
commits: [f34129c]
last_updated: 2026-06-29
related: [status, oom_incidents, adr_006_data_regeneration, sft_live, phase1_status_2026-06-26]
---

# 数据退化 — 项目最致命的阿喀琉斯之踵

**这是一次大师级的 Debug**。SFT 模态坍塌的根因不在模型，不在训练，而在数据生成阶段——SCA-FP 求解器产出的"最优解"在 5000 个环境间几乎完全同质。

## 发现过程：四层诊断

```
SFT 评估 → 完全模态坍塌 (所有 checkpoint [0,0,-5])
         ↓
消融测试 → 模型不是没学到，是学到了唯一解
         ↓
EDA 分析 → 数据分布极端退化 (3 组致命数字)
         ↓
求解器分析 → 物理模型缺乏高度 trade-off
```

### Layer 1: SFT 评估 — 完全模态坍塌

对 step 100/150/200 的 checkpoint 跑完整评估：

| Checkpoint | Text Quality | Control Sensitivity | SCA-FP Speedup |
|-----------|-------------|---------------------|----------------|
| step_100 | JSON 乱码 | 0.0000 | 1.0× |
| step_150 | JSON 正常 | 0.0000 | 1.0× |
| step_200 | JSON 退化 | 0.0000 | 1.0× |

**关键发现**：Control sensitivity = 0.0000 对所有 checkpoint。±10m 扰动输入，模型输出完全不变。模型不是没训练好——它训练得很好，好到把所有输入都映射到同一个"最优"输出。

### Layer 2: 消融 — 模型学到了什么？

对 step_150（JSON 能力最好的 checkpoint）做深度消融：
- 替换 q_current → 输出不变
- 替换 user_locs → 输出不变
- 替换 target_locs → 输出不变
- **结论**：模型完全忽略了输入文本，学会了输出一个固定向量 `[0, 0, -5]`

### Layer 3: EDA — 数据长什么样？

`scripts/eda_data.py` 对全量 5000 SFT 样本的分析：

#### 致命缺陷一："全员砸地板"

```
Vertical direction distribution:
  ↓steep down:  33.2%
  ↓down:        48.7%
  ↘slight↓:     15.5%
  →flat:         2.4%
  ↗slight↑:      0.3%
  ↑up:           0.0%    ← 完全不存在向上飞
  ↑steep up:     0.0%
```

**97.4% 的样本是向下位移，0% 向上。** SCA-FP 求解器认定的最优策略是"飞越低信号越好"——这是真空中的物理模型，没有任何对抗力。

#### 致命缺陷二："全员满油门"

```
Displacement magnitude distribution:
  at exactly 15.0m: 56.8%
  in [14.5, 15.0]:  84.7%    ← v_max 边界聚集
  in [10.0, 14.5]:  15.3%
  < 10m:             0.0%    ← 没有一个精细微调
```

**84.7% 的 UAV 以最大速度 15m/s 飞行，0% 需要小于 10m 的微调。** 模型从未见过"轻轻挪一下"的场景——它只学会了"往死里推油门"。

#### 致命缺陷三："全员满功率"

```
Total power distribution:
  0.99-1.0W: 15.7%
  1.0-1.01W: 84.3%    ← P_max 边界聚集
  < 0.99W:    0.0%    ← 没有一个低功率调度
```

**84.3% 的 UAV 满功率运行。** 没有功率 trade-off，没有节能考量，没有资源博弈。

### Layer 4: 求解器源码 — 为什么？

原 `SCAFPConfig` 的目标函数：

```
Utility = SumRate(comm) + w × SensingMI(sens)
         - λ × IdlePenalty
```

- **通信效用 SumRate**：路径损耗 ∝ log(dist)，低飞 = 距离近 = 信号强。单调递减于高度。
- **感知效用 SensingMI**：同样取决于距离。单调递减于高度。
- **Idle Penalty**：只惩罚"不动"，不限制方向。

**结论**：目标函数对高度是严格单调递减的。在物理约束允许的范围内（H_min=50m），"往死里砸地板"就是全局最优。这不是 bug——数学上求解器是正确的。问题在于物理模型不完整。

## 🔧 修复：地面杂波（Ground Clutter）建模

### 物理原理

真实低空环境中，电磁波遇到地面建筑物、树木、地形起伏产生**多径反射和散射**。这种"地面杂波"（Ground Clutter）在小擦地角（低高度）时尤为严重，随高度升高呈指数衰减。

**杂波衰减模型**：

```
clutter_db(h) = C₀ × (1 - h_norm)
  where h_norm = (h - H_min) / (H_max - H_min) ∈ [0, 1]
        C₀ = ground_clutter_db = 12.0 dB
```

- 在 H_min (50m): clutter = 12.0 dB（最大衰减）
- 在 H_max (300m): clutter = 0 dB（无杂波）

### 代码变更（`src/solver/sca_fp.py`）

```python
# SCAFPConfig 新增参数 (line 43)
ground_clutter_db: float = 12.0   # H_min 处额外损耗, H_max 处为 0

# 目标函数中应用 (lines 367-396)
h_norm = max(0.0, min(1.0, (q_new[2] - self.H_min) / (self.H_max - self.H_min)))
clutter_db = self.cfg.ground_clutter_db * (1.0 - h_norm)

# 通信路径损耗
pl_db = 28 + 22 * np.log10(max(dist_3d, 1.0)) + 20 * np.log10(self.carrier_freq_ghz)
pl_db += clutter_db  # NEW

# 感知路径损耗
pl_db = 20 * np.log10((4 * np.pi * max(dist_3d, 1.0)) / self.wavelength) + 20
pl_db += clutter_db  # NEW
```

### 创造的非凸曲面

修复后，高度决策有了真实的 trade-off：

```
高度 (m)   通信路径损耗     杂波损耗      总损耗
─────────────────────────────────────────────────
  50       ~100 dB        +12 dB       ~112 dB   ← 太低了
 150       ~105 dB         +6 dB       ~111 dB
 250       ~110 dB         +1 dB       ~111 dB
 300       ~112 dB          0 dB       ~112 dB   ← 太高了
       ↑ Sweet spot 大约在 150-250m，取决于水平距离
```

- **飞太低（~50m）**: 水平距离近(路径损耗小)，但地面杂波严重(+12dB) → 总损耗大
- **飞太高（~300m）**: 杂波消失，但水平距离远(路径损耗大) → 总损耗大
- **最优高度**: 在两者之间，且依赖具体的用户/目标水平分布——每个环境都有不同的"甜点"

**这才是我们需要大模型去学习的"物理直觉"！**

## 📊 期待的新分布

用 `ground_clutter_db=12.0` 重新生成数据后，期待看到：

| 指标 | 旧值 (退化) | 期待新值 |
|------|-----------|---------|
| 满速飞行 (≥14.5m) | 84.7% | < 40% |
| 精细微调 (<5m) | 0.0% | > 10% |
| 向下位移 | 97.4% | 40-60% |
| 向上位移 | 0.0% | 15-30% |
| 满功率 (≈1.0W) | 84.3% | < 50% |

## 🧪 快速验证

```bash
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm
python scripts/quick_validate_fix.py
```

验收标准：向上飞比例破冰（>15%），满速比例跌破 50%，精细微调出现（>10%）。

## 🔬 后续：DPO 数据生成策略 (Grilling 终稿)

数据重生不仅仅是更换 solver 参数。经过 5 轮领域模型火烤，完整的 DPO 数据生成流程已锁死。详见 [ADR 006](../06_decisions/adr_006_data_regeneration.md)。

关键要点：

1. **微扰回弹测试选 Chosen**：不迷信 Sum-Rate 最高的局部最优。统一施加 ε 扰动后重跑 SCA-FP，步数最少者当选——确保选的是"宽盆地"而非"窄深谷"。

2. **Masked DPO**：DPO 操作在文本 token log-probabilities 上。在 `dataset.py` 中将 JSON 里 δ_a/δ_p 对应 token 的 label 设为 `-100`，梯度集中在 δ_q 的偏好拉扯上。

3. **Top-K 精选**：生成 20,000 环境（非 5,000），按 Chosen-Rejected Gap 排序取前 5,000 名。预估存活率 25-35%。

4. **所有 Rejected 必经 Constraint Projections**：启发式陷阱（短视直线、原地不动、旧世界残影）必须通过 `clip_to_physics_bounds` 投影，确保是模型输出空间中的合法点。
| 满功率 (≈1.0W) | 84.3% | < 50% |

## 🧪 快速验证

```bash
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm
python scripts/quick_validate_fix.py
```

验收标准：向上飞比例破冰（>15%），满速比例跌破 50%，精细微调出现（>10%）。

## 为什么 SFT 在这个退化数据上注定失败？

### 数学分析

SFT 使用 Cross-Entropy Loss 训练。当数据中 97.4% 的样本都有 "向下" 的 q₃，CE 学到的条件概率分布是：

```
P(q₃ = "down" | any input) ≈ 0.974
```

在自回归生成时，一旦模型开始采样，它总是选择"向下"的 token。即使有 2.6% 的概率进入其他路径，模型从未见过向上飞的 token 序列——训练数据里根本就没有这种序列。所以它不可能"凭空发明"一个向上的位移。

### 为什么 DPO 在旧数据上也是浪费时间？

DPO 的本质是最大化 chosen 相对于 rejected 的概率。但旧数据的 chosen 全是"全速砸地板"——DPO 只会把这套极端策略夯得更死。数据底座退化，DPO 训一万个 epoch 也是在炼丹炉里烧垃圾。

**旧数据已全部废弃。DPO 训练已立即停止。**

## 教训

1. **"All checks passed" 可能是最大的谎言。** EDA 的 `FINAL VERDICT` 说数据没问题，但实际上分布极度病态。语义检查（"无 NaN"、"约束满足"）不等于分布健康检查。
2. **物理建模的完备性决定数据质量的上限。** 如果仿真环境缺乏关键的对抗力（trade-off），求解器会找到一条让所有场景坍缩的"捷径最优解"。
3. **先看数据，再看模型。** 我们花了大量时间调试架构（Multi-Query、Attention Pooling、MSE vs CE），但根因在数据层。如果一开始就跑 EDA 的 Section 3，能省下 3-4 天的调试时间。
4. **EDA 的 Diversity Check 应该放在数据生成后的第一个验收步骤。** 不是可选项目，是必须红线。Section 3 的三个指标（方向、速度、功率）不通过，不能进入训练。
5. **CE Loss 对连续物理量有硬天花板。** 即使修好了数据分布，SFT + CE 仍然面临根本挑战——CE 认为预测 5.1 和 5.2 跟预测"猫"和"狗"一样错误。对连续坐标的回归任务，需要 DPO 的对比学习或回归头。
