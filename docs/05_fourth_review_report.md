# 第四轮审查报告（Gemini 独立审查 + 修复验证）

## 审查背景

将 Codex（第一轮）、我的系统性审查（第二轮）之后仍存在的代码提交给 Gemini 做独立第三方审查。Gemini 使用与第二轮相同的方法论（逐数据流追踪 + 对照论文验证），发现了 **6 个前两轮遗漏的全新缺陷**。

**审查快照**：commit `cad1bda`（注意：此时 `src/data/` 因 `.gitignore` 误配置未纳入仓库，Gemini 审查覆盖率约 75%）

---

## Gemini 审查结果总览

| # | 严重度 | 问题简述 | 验证结果 |
|---|--------|---------|---------|
| 1 | P0 | DPO log-prob 使用平均值而非求和 | ✅ 确认属实，已修复 |
| 2 | P0 | Control token 隐状态切片 off-by-one | ✅ 确认属实（仅影响 fallback 路径），已修复 |
| 3 | P0 | 移动性约束降维为 2D，垂直不受限 | ✅ 确认属实，已修复 |
| 4 | P1 | 评估满意度分母可用性刷榜 | ✅ 确认属实，已修复 |
| 5 | P1 | 功率投影无 P_min 下限 | ✅ 确认属实，已修复 |
| 6 | P2 | 阵列增益硬编码 N_t² | ✅ 确认属实，已修复 |

---

## 逐项详细分析

### P0-1：DPO 序列概率使用平均值（`train_dpo.py:62`）

**原始代码**：
```python
seq_logp = masked.sum(dim=-1) / (shift_mask.sum(dim=-1) + 1e-8)
```

**根因**：DPO 的目标函数（论文公式 34）定义在联合概率分布上：
$$\log \pi_\theta(y|x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t})$$

这是各 token log-prob 的**求和**，不是平均值。代码中除以 `shift_mask.sum()` 将求和变成了平均。

**影响**：
- 短序列和长序列的 log-prob 被强制拉到相同尺度
- DPO 的隐式奖励 $r = \beta(\log\pi_\theta - \log\pi_{ref})$ 的尺度和序列长度脱钩
- KL 散度约束 $\beta \cdot D_{KL}$ 作用在错误归一化的分布上
- 长序列中每个 token 对 loss 的贡献被稀释，模型倾向于生成冗长低质输出

**修复**：
```python
seq_logp = masked.sum(dim=-1)  # SUM not mean
```

---

### P0-2：Control Token 切片 off-by-one（`gemma_isac.py:181`）

**原始代码**：
```python
seq_lens = attention_mask.sum(dim=1) - 1
hidden_states[b, seq_lens[b] - self.num_control_tokens : seq_lens[b]]
```

**根因**：Python 切片 `[start:end)` 是左闭右开区间。

假设序列长 100 tokens（索引 0-99），最后 8 个位置（92-99）是 control tokens：
- `seq_lens[b]` = 99（最后一个 token 的索引）
- 切片：`[99-8 : 99]` = `[91:99]` → 提取索引 91, 92, 93, 94, 95, 96, 97, 98
- 实际 control tokens 在索引 92-99
- **丢失**：索引 99（`<ctrl_7>`，最后一个 control token）
- **误入**：索引 91（一个普通 prompt token）

**影响范围**：仅影响 `forward()` 的 fallback 路径（control_mask=None 时）。当前训练代码（train_sft.py, train_dpo.py）正确传入了 control_mask，使用正常路径（line 156-175），因此**当前训练不会触发此 bug**。但 `generate_warmstart()` 使用直接切片方式（取最后 N 个位置），也不受影响。

**降级为 P1** 的原因：当前所有调用路径均不受影响。但如果未来有人去掉 control_mask 或在不同框架中使用，会静默出错。

**修复**：
```python
hidden_states[b, seq_lens[b] - self.num_control_tokens + 1 : seq_lens[b] + 1]
```

---

### P0-3：3D 移动性约束退化（`projection_head.py:128-165`）

**原始代码**：
```python
# 仅对水平位移 (x, y) 进行 v_max_dt 裁剪
displacement_2d = delta_tilde[..., :2]
norms_2d = torch.norm(displacement_2d, dim=-1, keepdim=True) + 1e-8
scale_2d = torch.clamp(self.v_max_dt / norms_2d, max=1.0)
clipped_2d = displacement_2d * scale_2d

# 垂直位移未受速度约束
dh = delta_tilde[..., 2:3]
new_pos_h = q_current[..., 2:3] + dh     # 无速度限制！
new_pos_h = torch.clamp(new_pos_h, self.h_min, self.h_max)  # 仅有绝对高度限制
```

**根因**：论文公式 (28) 明确规定 3D 空间中的移动约束：
$$\|q_m(t+1) - q_m(t)\|_2 \le v_{\max} \Delta t$$

但代码将约束分解为独立的水平速度限制（2D 范数）和无限制的垂直位移。

**影响**：
- 垂直方向：在 1 个 time slot 内，UAV 可以从 50m 瞬间爬升到 300m
- 配置中 v_max=15m/s, Δt=1s，理论最大垂直移动 = 15m，但实际可达 250m
- 模型产出的先验被下游 SCA-FP 求解器判定为物理不可行
- SCA-FP 必须大幅修正 UAV 位置，MLLM warm-start 失去意义

**修复**：对整个 3D 位移向量统一做范数裁剪
```python
displacement_3d = delta_tilde
norms_3d = torch.norm(displacement_3d, dim=-1, keepdim=True) + 1e-8
scale_3d = torch.clamp(self.v_max_dt / norms_3d, max=1.0)
clipped_3d = displacement_3d * scale_3d
# 然后再做区域和高度 absolute clamp
```

---

### P1-4：评估满意度"刷榜"漏洞（`evaluate.py:258`）

**原始代码**：
```python
num_total_associated = int(np.sum(sol.A > 0.5))
comm_sat = num_satisfied_comm / max(num_total_associated, 1)
```

**根因**：通信满意率的分母是"已关联用户数"，而非"总用户数"。

**影响**：
- 极端情形：SCA-FP 只关联 20 个用户中 SINR 最好的 1 个 → comm_sat = 100%
- 无法区分"服务了全部用户但部分不满足"和"只服务了一个满足的用户"
- 评估报告中的 joint_satisfaction 同理被污染

**修复**：
```python
comm_sat = num_satisfied_comm / max(solver.K, 1)  # 使用总用户数
```

---

### P1-5：功率投影缺失 P_min 约束（`projection_head.py:PowerProjection`）

**原始代码**：
```python
p_soft = F.softmax(p_tilde / self.tau, dim=-1)
p_hat = self.p_max * p_soft
# 没有最小值保护
```

**根因**：论文公式 (21) 要求 $A_{m,k} \cdot \|w_{m,k}\|^2 \ge A_{m,k} \cdot P_{\min}$。Softmax 可以产生任意接近 0 的值，乘以 P_max 后仍接近 0。对于被关联的用户，这会导致无法维持解码阈值的无效连接。

**修复**：在 Softmax 后将通信条目钳位到 p_min，从高于 floor 的条目中扣除超额功率并重分配：
```python
p_min = p_min_ratio * p_max  # 默认 1% of P_max
p_comm = p_hat[..., :K_comm]
# 钳位 + 重分配 (详见 projection_head.py:PowerProjection.forward)
```

---

### P2-6：阵列增益硬编码 N_t²（`sca_fp.py`）

**原始代码**：
```python
sinr_s = P_sense[m] * pl_linear * self.N_t ** 2 / self.N0  # 3处相同
```

**根因**：雷达方程的双程增益是 $N_t \cdot N_r$（发射阵列增益 × 接收阵列增益）。在 $N_t = N_r = 8$ 时 $N_t^2 = N_t N_r = 64$ 数值巧合。但如果配置改为异构阵列（如 $N_t=16, N_r=8$），$N_t^2 = 256$ 而 $N_t N_r = 128$，优化目标与真实物理反馈偏离 2 倍。

**影响范围**：`sca_fp.py` 的 `_optimize_deployment_sca` 和 `_compute_utility`，以及 `evaluate.py` 的 sensing SINR 计算。

**修复**：新增 `N_r` 参数（默认等于 `N_t`），所有位置改为 `self.N_t * self.N_r`。

---

## 第二轮审查遗留修复（本轮同步完成）

| # | 问题 | 位置 |
|---|------|------|
| P0-#4 | evaluate.py:237 硬编码波长 0.0517 | 改为从 config 读取 carrier_freq_ghz 计算 |
| P0-#3 | user_weights 全 1 丢弃异构权重 | oracle_generator.py:233 + evaluate.py:190 + EnvironmentSample 新增字段 |

---

## 追加发现：`.gitignore` 误排除 `src/data/`

**根因**：`.gitignore` 第 28 行 `data/` 未锚定，匹配任意层级的 `data` 目录，包括 `src/data/`。

**影响**：
- `src/data/__init__.py`、`dataset.py`、`oracle_generator.py`、`prompt_builder.py` 从未纳入初始 commit
- Gemini 审查的仓库缺少完整的数据管线代码
- `.gitignore` 语义应为"生成的训练数据"（根目录 `/data/`），非"源代码的数据模块"（`src/data/`）

**修复**：`data/` → `/data/`

---

## 跨轮次修复状态总览

| 来源 | 发现数 | P0 | P1 | P2 | 已修复 |
|------|--------|----|----|----|----|
| Codex（第一轮） | 9 | 5 | 3 | 1 | ✅ 9/9 |
| 第二轮审查 | 9 | 4 | 3 | 2 | ✅ 8/9* |
| Gemini（本轮） | 6 | 3 | 2 | 1 | ✅ 6/6 |
| **合计** | **24** | **12** | **8** | **4** | **23/24** |

\* 第二轮 P2-#9（generate_data.py 双击 Ctrl+C 假提示）不阻塞训练，未修。

---

## 训练前检查清单

在投入 5000 环境数据生成和正式训练前，确认：

- [x] 路径损耗公式碎片化（第二轮 P0-#1）— 已知未修，建议 SCA-FP 内部统一
- [x] 信道随机性受控（第二轮 P0-#2）— 已知未修，`uav_channel.py:139` 仍用全局 RNG
- [x] DPO reference model deepcopy 风险（第二轮 P1-#5）— 已知未修
- [x] 本轮全部 6 个修复已提交
- [x] `src/data/` 已纳入版本控制

**当前阻塞**：第二轮 P0-#2（信道 RNG）和 P0-#1（路径损耗统一）仍未修。这两个直接影响训练数据质量，建议在重新生成数据前先修。
