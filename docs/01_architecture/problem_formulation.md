---
type: reference
status: current
stage: all
last_updated: 2026-06-26
related: [system_design, training_pipeline, canonical_config]
---

# Problem Formulation — UAV-ISAC with MLLM Warm-Start

## 场景

**UAV-ISAC (Integrated Sensing and Communication)**: 无人机集群在通信和感知之间共享频谱资源，同时服务地面 IoT 用户并感知目标。

| 参数 | 符号 | 值 |
|------|------|-----|
| UAV 数量 | M | 4 |
| IoT 用户数量 | K | 20 |
| 感知目标数量 | T | 6 |
| 天线数量 | N_t = N_r | 8 |
| 载波频率 | f_c | 5.8 GHz |
| 带宽 | B | 20 MHz |
| 最大发射功率 | P_max | 30 dBm (1W) |
| UAV 高度范围 | H | 50-300 m |
| 最大速度 | v_max | 15 m/s |
| 区域 | - | 1000×1000 m² |
| 时隙长度 | Δt | 1s |

## 优化问题

目标是在每个时隙联合优化 UAV 三维位置 **Q**、用户关联矩阵 **A** 和波束成形向量 **W_c, W_s**，最大化加权系统效用：

```
maximize  U(Q, A, W_c, W_s) = λ·R_comm + (1-λ)·R_sens
subject to:
  - UAV 移动性约束: ||q_m^{(t)} - q_m^{(t-1)}||₂ ≤ v_max · Δt = 15m
  - 功率约束:      Σ_k ||w_{c,mk}||² + Σ_t ||w_{s,mt}||² ≤ P_max, ∀m
  - 关联约束:      a_{mk} ∈ {0,1}, Σ_m a_{mk} ≤ 1, ∀k
  - QoS 约束:      SINR_k ≥ SINR_min
```

其中 R_comm 为通信和速率，R_sens 为感知互信息。

## SCA-FP 数值优化器

**SCA-FP** (Successive Convex Approximation — Fractional Programming) 交替优化三个子问题：

1. **部署优化 (_optimize_deployment_sca)**: 固定 A, W, 通过 L-BFGS-B 优化 Q
2. **关联优化 (_optimize_association)**: 固定 Q, W, 通过匈牙利算法优化 A
3. **波束成形优化 (_optimize_beamforming)**: 固定 Q, A, 通过闭式注水解优化 W

### 收敛特性

- 每次 Best-of-N 重启 (N=10) 从不同初始点运行
- 收敛速度高度依赖初始猜测质量
- **核心洞察**: MLLM 可以从历史模式预测近优解，大幅减少 SCA-FP 迭代次数

## MLLM 方案

### 为什么用 MLLM？

传统上 SCA-FP 从随机或启发式初始点启动。MLLM 学习从环境状态到优化变量之间的映射，提供**智能热启动**：

1. 编码环境状态 (UAV 位置、用户分布、信道条件) 为文本 prompt
2. Gemma 3 12B 输出优化变量预测
3. 约束投影头将原始输出投影到可行域
4. 投影后的解作为 SCA-FP 的初始点

### 输出维度 (176 个值)

| 变量 | 维度 | 类型 | 范围 |
|------|------|------|------|
| ΔQ (位移) | 4 UAV × 3D = 12 | float | [-15, 15] m |
| A (关联) | 4 UAV × 20 users = 80 | int (0/1) | {0, 1} |
| P_c (通信功率) | 4 UAV × 20 users = 80 | float | [0, P_max] |
| P_s (感知功率) | 4 UAV × 1 = 4 | float | [0, P_max] |

共计 12 + 80 + 80 + 4 = 176 个值，编码为 JSON 格式。

### 三个约束投影头

| 投影头 | 输出 | 约束 | 方法 |
|--------|------|------|------|
| **Proj_Q** | ΔQ | 3D 球约束: ‖Δq_m‖₂ ≤ 15m | Tanh 缩放 + 裁剪 |
| **Proj_A** | A | 行随机: Σ_m a_{mk} ≤ 1, a_{mk} ∈ {0,1} | Sinkhorn 迭代 (20 iters) |
| **Proj_P** | P_c, P_s | 功率预算: Σ P ≤ P_max | Softmax + 预算缩放 |

### Control Token 机制

8 个特殊 token `<ctrl_0>` 到 `<ctrl_7>` 插入在 prompt 末尾。Gemma 3 对这些 token 产生的 hidden states 被提取并送入投影头，将环境上下文直接编码为优化变量的参数化。

## 物理层模型

- **信道**: 3GPP UMa LoS 路径损耗 + Rician 小尺度衰落
- **LoS 概率**: 3GPP TR 36.777 仰角依赖模型
- **感知**: 双雷达方程 (双基地雷达模型)
- **波长**: λ = c / f_c = 3e8 / (f_c × 1e9) (动态计算，不再硬编码)
- **噪声功率**: N₀ = k_B × T × B = 1.38e-23 × 290 × B

## 参考文献

- 论文: UAV-ISAC with MLLM Warm-Start (SCA-FP + Gemma 3)
- SCA-FP: 基于连续凸逼近的无人机通信感知一体化资源分配
- DPO: Direct Preference Optimization (Rafailov et al., 2023)
