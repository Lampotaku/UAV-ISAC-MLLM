# UAV-ISAC-MLLM — Domain Glossary

> 本文档是项目的 **领域词汇表（Glossary）**，不含任何实现细节。
> 当术语在讨论中被精确定义后，即时更新此文件。

## Core Concepts

### Warmstart
模型预测的 `q_current + δ_q̂`，作为 SCA-FP 数值优化器的迭代初始点。

**关键设计妥协**：训练时用 MSE 让模型逼近 Oracle 的全局最优解 δ_q*（Surrogate Loss），而非直接优化"SCA-FP 从该点出发的迭代次数"。前提假设：**最优解的邻域通常也是极佳的 Warmstart 点**。直接优化迭代次数需要将 SCA-FP 可微化（Differentiable Optimization），工程代价极高。

### SCA-FP Solver
交替优化算法（SCA-FP），在 UAV 通信/感知联合优化问题中求解 UAV 部署位置、用户关联和波束成形的数值解。

**三个角色，一个实例**：
- **Oracle/Teacher**（数据生成）：Best-of-N (N restarts) → 取 Sum-Rate 最高的解作为 Chosen
- **Physics Simulator**：目标函数编码物理规律（含 ground clutter），是"物理世界"的数学表示
- **Evaluator**（评估）：Restart=0，只接受模型预测的单一 Warmstart → 测量迭代次数和最终 Sum-Rate

**铁律**：三阶段必须使用完全相同的 `SCAFPConfig`（包括 `ground_clutter_db` 等所有物理参数）。任何差异都会导致系统性评估偏差。

### δ_q (Displacement Vector)
UAV 位置的变化量，shape `[N_uav, 3]` = [Δx, Δy, Δh]。三种上下文：

| 上下文 | 来源 | 语义 |
|--------|------|------|
| SFT Oracle δ_q* | SCA-FP Best-of-N 最优解 | "正确答案" |
| DPO Chosen δ_q⁺ | SCA-FP Best-of-N 最高 Sum-Rate 解 | "偏好解" |
| DPO Rejected δ_q⁻ | 次优局部解 或 模型坍塌输出 | "应避免的解" |
| Model output δ_q̂ | Projection Head | "模型预测" |

### Ground Clutter
低空飞行时地面建筑物、树木和地形起伏对电磁波产生的额外多径衰减。数学上建模为高度归一化后的线性衰减：
```
clutter_db(h) = C₀ × (1 - h_norm)
```
其中 `h_norm ∈ [0,1]`（H_min→0, H_max→1），`C₀ = 12.0 dB`。

**作用**：在高度维度创造一个非凸 trade-off——太低则杂波大，太高则距离远。打破了旧求解器中"飞越低信号越好"的单调性，使每个环境的最优高度依赖于具体的用户/目标分布。

### DPO Preference Pair
- **Chosen**：SCA-FP Best-of-N 中 Sum-Rate 最高的解，必须通过两道质量闸门（见下方）
- **Rejected**（混合策略，最终方案）：
  - **Design A' — 启发式边界构造法**（~30%）：不使用模型历史预测，而是主动构造物理上"看似合理但致命"的陷阱：
    1. **短视直线（Greedy Line）**：忽略杂波，以最大速度飞向目标
    2. **原地不动（Zero-Movement）**：`[0,0,0]` —— 惩罚惰性
    3. **旧世界残影（Old Ghost）**：`[0,0,±15]` —— 明确标记旧分布的坍塌模式为异端
  - **Design B — SCA-FP 次优局部解**（~70%）：同一个 Best-of-N 运行中 Sum-Rate 最低的局部最优解

### Preference Pair Quality Gates
DPO 偏好对的质量闸门（数据生成时执行，宁缺毋滥）：

1. **方差过滤（Variance Gating）**：N 次 Restart 中，Chosen 与 Rejected 的 Sum-Rate 差距必须 >5%。差距 <5% 的样本缺乏足够的偏好信号强度，**直接丢弃该环境**。
2. **基线校验（Baseline Verification）**：Chosen 的 Sum-Rate 必须严格大于 `[0,0,0]` 不动方案的 Sum-Rate。否则该 Chosen 是垃圾，整个环境废弃。

**设计原则**：不迷信 N=10 的"伪全局最优"。用显著的相对差距保证偏好信号强度，用绝对物理基线保证 Chosen 的下限。

### Perturbation Snap-back Test
替代 λ 加权复合分数的 Chosen 选择方法。对 SCA-FP N 次 Restart 中表现最好的 Top-K 个候选局部最优解，施加统一的高斯小扰动 ε，以扰动后的点作为 SCA-FP 初始值重跑一次。收敛所需迭代步数最少的候选，即为真正的"宽盆地"——因为所有候选的起跑线偏差被强行统一了（都偏离谷底 ε），迭代次数成为盆地曲率的纯粹反向代理。

**流程**：
1. N 次随机 Restart → 记录 N 个局部最优解
2. Pareto 过滤：丢弃 Sum-Rate 不如 `[0,0,0]` 的解，丢弃低于全局最高 Sum-Rate 5% 以上的劣质坑
3. 对剩余的 Top-K 候选施加 ±2m 随机扰动
4. 以扰动点为初始值重跑 SCA-FP，步数最少者当选 **Chosen**

**优势**：彻底消除了"随机初始值的运气"对迭代步数的混淆，无需标定 λ 超参数。

### Masked DPO
DPO 训练时的分项解耦策略。在 `dataset.py` 的 tokenization 阶段，通过正则匹配找到 JSON 中 δ_a 和 δ_p 对应的字符区间，映射到 token indices 后将 label 强制设为 `-100`（ignore index）。DPO 的 `_compute_logprob` 自动跳过这些 token，梯度只在 δ_q 相关的 token 上产生偏好拉扯。

**注意**：Masked DPO 操作在**文本空间**（JSON token 的 log-probabilities），而非投影头输出空间。DPO 比较的是"生成包含 good δ_q 的 JSON 文本"vs"生成包含 bad δ_q 的 JSON 文本"的概率。控制头（ProjectionHead）的训练仍由 CTL/MSE loss 独立驱动。

### ε-Calibration (Pilot Sweep)
在正式数据生成前运行的微扰步长标定脚本。对 50 个随机环境测试 ε ∈ {0.5, 1.0, 2.0, 4.0, 8.0}m，观察回弹步数的方差（区分度）。选择能使步数出现明显阶梯差异（而非全部 1-2 步或全部超时）的 ε。该值通常落在 1.0-2.5m 区间，对应物理模型中"盆地边缘"的特征尺度。

### Spatial Repulsion (多 UAV 互斥力)
SCA-FP 目标函数中的反比惩罚项，防止多 UAV 扎堆到同一位置：
```
Penalty_repel = λ_repel × Σ_i Σ_{j>i} 1 / max(||q_i - q_j||², ε_min)
```
- λ_repel 建议 0.01 起步，逐步增加至 UAV 间最小距离 > 安全阈值
- 与 ground_clutter_db 正交——杂波解决"高度单调性"，互斥力解决"空间拥挤"
- 对应症状：多 UAV 全部重合在同一"避风港"，AI 因惩罚过重而选择保守策略

### Control Token
插入到文本序列中的特殊 token（`<ctrl_0>`..`<ctrl_7>`），其 hidden states 被 Multi-Query Attention Pooling 读取，作为投影头的输入来预测连续物理量（δ_q, δ_a, δ_p）。与文本 token 共享同一个 Transformer backbone 但通过独立的 attention query 读出。

### Deterministic Forward Projection
所有启发式构造的 Rejected 样本在写入数据前，必须经过与模型输出相同的约束投影层（`DeploymentProjection` 等）。确保 Rejected 是**合法但愚蠢**的物理点——模型在数学上有能力映射到它（因为约束投影划定的凸包包含了该点），但选择它会导致低 Sum-Rate。不经投影的裸向量会导致 DPO 梯度指向模型输出空间的非法区域。

### Composite Score (Chosen Selection Criterion)
替代纯 Sum-Rate 排名的选择标准：
```
Score = Sum-Rate − λ × Iterations
```
其中 `Iterations` 是该次 SCA-FP Restart 从随机初始值收敛所需的迭代步数。

**设计动机**：模型预测必有误差。对 Warmstart 而言，一个对误差宽容的"宽盆地"（Sum-Rate 略低但收敛快）优于要求绝对精度的"深峡谷"（Sum-Rate 极高但一偏差就坠崖）。λ 将迭代次数折算为 Sum-Rate 的等效惩罚。

### Yield Rate & Top-K Selection
- **Yield Rate**：生成的环境经过方差过滤和基线校验后的存活比例，预估 25-35%
- **Top-K 策略**：不设绝对阈值，而是生成过量环境（如 20,000），按 Chosen-Rejected 的 Composite Score Gap 排序，取前 K=5,000 名。保证满编容量的同时确保每一条都是物理区分度最强的"黄金数据"
