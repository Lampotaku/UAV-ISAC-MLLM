这是一份独立的第三轮深度审查报告。通过对数据流管线（环境 → 优化器 → 模型 → 评估）的逐层比对，以及与底层数学公式的交叉验证，发现了现有报告（Codex Fix Report 与 Second Review Report）中**遗漏的 6 个全新缺陷**。这些缺陷涵盖了核心的 DPO 数学计算、物理约束投影逻辑以及模型张量切片操作。

以下为具体审查结果：

### 🚨 P0 — 核心算法与数据流崩溃（会导致训练退化或约束失效）

#### 1. DPO 序列概率（Log-prob）使用了平均值而非求和

* **文件**: `lampotaku/uav-isac-mllm/UAV-ISAC-MLLM-3c603be22f557d5075d12818359fb8f998825126/src/training/train_dpo.py`
* **代码行**: 58
* **代码片段**:
```python
seq_logp = masked.sum(dim=-1) / (shift_mask.sum(dim=-1) + 1e-8)

```


* **后果**: DPO 的理论推导（论文公式 34）基于生成完整响应序列的联合概率分布，在对数域下应为每个 Token 概率的**总和（Sum）**。代码中除以了序列长度变成了平均值（Mean）。这将使得模型丧失对序列长度的正则化能力，在偏好对中，较短和较长的序列在概率尺度上被强制拉平，直接破坏 KL 散度约束，导致 Stage II 训练完全崩溃或产生乱码。
* **修复建议**: 移除长度归一化操作。
```python
seq_logp = masked.sum(dim=-1)  # 移除除法

```



#### 2. 控制 Token 隐状态提取切片存在“差一错误（Off-by-one）”

* **文件**: `lampotaku/uav-isac-mllm/UAV-ISAC-MLLM-3c603be22f557d5075d12818359fb8f998825126/src/model/gemma_isac.py`
* **代码行**: 136-137
* **代码片段**:
```python
seq_lens = attention_mask.sum(dim=1) - 1
control_states = torch.stack([
    hidden_states[b, seq_lens[b] - self.num_control_tokens:seq_lens[b]]
    # ...

```


* **后果**: Python 的切片 `[start:end]` 是左闭右开区间。`seq_lens[b]` 指向的是序列的最后一个 Token 的索引。此切片提取了从 `L - 8` 到 `L - 1`（不含 `L - 1`）的隐状态。这意味着它完美地**错过了真正的最后一个控制 Token（`<ctrl_7>`）**，并且混入了一个前面的普通文本 Token。这将导致 Projection Head 一直在接收错位的语义特征。
* **修复建议**: 右侧边界加上 1。
```python
hidden_states[b, seq_lens[b] - self.num_control_tokens + 1 : seq_lens[b] + 1]

```



#### 3. 3D 物理移动性约束实质上被降维为 2D

* **文件**: `lampotaku/uav-isac-mllm/UAV-ISAC-MLLM-3c603be22f557d5075d12818359fb8f998825126/src/model/projection_head.py`
* **代码行**: 99-102 & 111-112
* **代码片段**:
```python
# 仅对水平位移 (x, y) 进行了 v_max_dt 的裁剪
scale_2d = torch.clamp(self.v_max_dt / norms_2d, max=1.0)
clipped_2d = displacement_2d * scale_2d
# 垂直位移 dh 未受速度限制，仅受绝对高度限制
new_pos_h = q_current[..., 2:3] + dh
new_pos_h = torch.clamp(new_pos_h, self.h_min, self.h_max)

```


* **后果**: 论文公式 (28) 明确指出无人机的移动性约束 $\|q_m(t+1) - q_m(t)\|_2 \le v_{\max}\Delta t$ 应当在 3D 空间内强制执行。然而，代码中的投影模块仅限制了无人机的水平速度。对于垂直方向，无人机可以在 1 秒内瞬间从 50m 爬升至 300m（超出 15m/s 物理极限十余倍）。这会导致模型生成的先验解被下游 SCA-FP 求解器认定为物理不可行。
* **修复建议**: 对整个 3D 位移向量进行范数计算和裁剪。
```python
displacement_3d = delta_tilde
norms_3d = torch.norm(displacement_3d, dim=-1, keepdim=True) + 1e-8
scale_3d = torch.clamp(self.v_max_dt / norms_3d, max=1.0)
clipped_3d = displacement_3d * scale_3d
# 之后再在最终坐标上应用区域和高度的 clamp

```



---

### ⚠️ P1 — 训练稳定性与论文逻辑偏离

#### 4. 评估脚本中的满意度指标存在“刷榜”漏洞

* **文件**: `lampotaku/uav-isac-mllm/UAV-ISAC-MLLM-3c603be22f557d5075d12818359fb8f998825126/src/eval/evaluate.py`
* **代码行**: 203 & 205
* **代码片段**:
```python
num_total_associated = int(np.sum(sol.A > 0.5))
comm_sat = num_satisfied_comm / max(num_total_associated, 1)

```


* **后果**: 脚本将通信满意率的分母设为了**已关联的用户数**。这在优化场景中是一个严重的逻辑漏洞：如果模型先验导致 SCA-FP 只选择服务 20 个用户中信道最好的 1 个用户，它将获得 100% 的 `comm_sat`。评估应体现对全体网络的覆盖能力。
* **修复建议**: 分母应当使用总用户数 $K$。
```python
comm_sat = num_satisfied_comm / solver.K

```



#### 5. 功率投影遗漏了最小用户功率约束 ($P_{min}$)

* **文件**: `lampotaku/uav-isac-mllm/UAV-ISAC-MLLM-3c603be22f557d5075d12818359fb8f998825126/src/model/projection_head.py`
* **代码行**: 151
* **代码片段**:
```python
p_hat = self.p_max * p_soft

```


* **后果**: 论文公式 (21) 强制要求每个被分配的用户都必须达到最小功率底线：$A_{m,k}\|w_{m,k}\|^2 \ge A_{m,k}P_{\min}$。然而，代码中的 `PowerProjection` 直接将 Softmax 乘以总预算输出，未对极小值进行截断或重分配。这会导致连续空间内的先验解产生无法维持解码阈值的无效连接。
* **修复建议**: 在 `PowerProjection` 中引入 `p_min` 参数，在 Softmax 后对通信维度（前 K 个元素）执行下界钳位，并对剩余功率重新归一化。

---

### 🔍 P2 — 物理语义的微小错位

#### 6. 阵列增益的硬编码与信道模型脱节

* **文件**: `lampotaku/uav-isac-mllm/UAV-ISAC-MLLM-3c603be22f557d5075d12818359fb8f998825126/src/solver/sca_fp.py`
* **代码行**: 349
* **代码片段**:
```python
sinr_s = P_sense[m] * pl_linear * self.N_t ** 2 / self.N0

```


* **后果**: 在 SCA 优化器中，感知 SINR 的天线阵列增益被硬编码为 $N_t^2$。而在 `uav_channel.py` 中，严格的双程雷达方程使用了 `self.N_t * self.N_r`。虽然在当前 YAML 配置中 $N_t = N_r = 8$ 不会立刻报错，但如果后续实验需要测试异构收发阵列（如 $N_t=16, N_r=8$），SCA-FP 的优化目标将与真实的物理反馈完全背离。
* **修复建议**: 初始化 `SCAFPOptimizer` 时引入 `N_r` 参数，并将上述公式替换为 `self.N_t * self.N_r`。