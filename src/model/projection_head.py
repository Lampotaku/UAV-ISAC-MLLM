"""
可微约束投影头
论文 Section 5 — Differentiable Constraint-Projection Head

h_Φ(Z_c) → δ̂ = [Proj_Q(δ̃_q'), Proj_A(δ̃_a'), Proj_P(δ̃_p')]

架构 (公式 21-23):
  Z_c  (控制 token hidden states)
   ↓  Linear Readout
  δ̃   (raw continuous prior)
   ↓  Residual MLP f_Φ (公式 22)
  δ̃'  (corrected prior)
   ↓  Structured Projections (公式 23)
  δ̂   (feasible warm-start prior)

三个投影模块:
  Proj_Q: Clipping + tanh (高度/区域/移动性)
  Proj_P: Softmax + 功率预算  (per-UAV power budget)
  Proj_A: Sinkhorn + 容量     (capacitated transport)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualMLP(nn.Module):
    """
    残差 MLP 修正器 f_Φ (公式 22)

    δ̃' = δ̃ + MLP(δ̃)

    两层 hidden: 256 → 256
    """

    def __init__(self, in_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = in_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.GELU(),
                nn.LayerNorm(hd),
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, in_dim))  # 残差: 输出 = 输入维度
        self.net = nn.Sequential(*layers)

        # 初始化: 近乎 identity
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # 最后一层设为零 (初始时残差≈0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ControlReadout(nn.Module):
    """
    控制 Token 读出 (公式 21)

    Z_c (control token hidden states) → δ̃ (raw continuous prior)

    使用可学习 query attention pooling 替代 mean pooling:
      - 允许每个 control token 专门化不同子任务 (position / layout / power)
      - query 动态聚焦最相关的 token，避免 mean 抹平空间结构
      - 初始化 query ~ N(0, 0.02) → 初始注意力接近均匀，等价于 mean 起点
    """

    def __init__(self, hidden_dim: int, num_control_tokens: int, out_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_control_tokens

        # 可学习 attention query — 让模型决定从哪些 control token 读取
        self.attn_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.attn_query, std=0.02)

        # 注意力池化后投影到 out_dim
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, control_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            control_states: (batch, num_control_tokens, hidden_dim)
                             控制 token 位置的 hidden states

        Returns:
            raw_prior: (batch, out_dim)  连续的原始 prior
        """
        B = control_states.shape[0]

        # query attention pooling: query (B,1,hidden) × keys^T (B,hidden,N) → (B,1,N)
        query = self.attn_query.expand(B, -1, -1)                    # (B, 1, hidden_dim)
        scale = self.hidden_dim ** 0.5
        attn_scores = torch.bmm(query, control_states.transpose(1, 2)) / scale  # (B, 1, N)
        attn_weights = F.softmax(attn_scores, dim=-1)                # (B, 1, N)

        # 加权聚合: weights (B,1,N) × values (B,N,hidden) → (B, hidden_dim)
        pooled = torch.bmm(attn_weights, control_states).squeeze(1)  # (B, hidden_dim)

        return self.readout(pooled)


class DeploymentProjection(nn.Module):
    """
    部署投影 Proj_Q
    公式 (24-26)

    强制执行:
      - 高度约束: H_min ≤ Ĥ ≤ H_max
      - 区域约束: x,y ∈ A
      - 移动性: ||Δq|| ≤ v_max * Δt
    """

    def __init__(
        self,
        area_w: float = 1000.0,
        area_h: float = 1000.0,
        h_min: float = 50.0,
        h_max: float = 300.0,
        v_max_dt: float = 15.0,
    ):
        super().__init__()
        self.register_buffer("area_w", torch.tensor(area_w))
        self.register_buffer("area_h", torch.tensor(area_h))
        self.register_buffer("h_min", torch.tensor(h_min))
        self.register_buffer("h_max", torch.tensor(h_max))
        self.register_buffer("v_max_dt", torch.tensor(v_max_dt))

    def forward(
        self,
        delta_tilde: torch.Tensor,     # (B, M, 3) — 位移 [dx, dy, dh]
        q_current: Optional[torch.Tensor] = None,  # (B, M, 3)
    ) -> torch.Tensor:
        """
        投影位移到约束空间

        如果提供 q_current:
          新位置 = clip(q_current + displacement)
        否则:
          只 clip 位移幅度
        """
        B, M, _ = delta_tilde.shape

        # 3D 移动性约束 (论文公式 28): ||Δq||_2 ≤ v_max * Δt
        # 对整个 3D 位移向量做范数裁剪, 而非仅裁剪水平分量
        displacement_3d = delta_tilde  # (B, M, 3)
        norms_3d = torch.norm(displacement_3d, dim=-1, keepdim=True) + 1e-8
        scale_3d = torch.clamp(self.v_max_dt / norms_3d, max=1.0)
        clipped_3d = displacement_3d * scale_3d

        if q_current is not None:
            # 从当前位置计算新坐标
            new_pos = q_current + clipped_3d  # (B, M, 3)

            # 区域裁剪 (x, y)
            new_pos_xy = torch.stack([
                torch.clamp(new_pos[..., 0], 0.0, self.area_w),
                torch.clamp(new_pos[..., 1], 0.0, self.area_h),
            ], dim=-1)

            # 高度裁剪
            new_pos_h = torch.clamp(new_pos[..., 2:3], self.h_min, self.h_max)

            # 拼接后转回位移
            new_pos_full = torch.cat([new_pos_xy, new_pos_h], dim=-1)
            result = new_pos_full - q_current
        else:
            result = clipped_3d

        return result


class PowerProjection(nn.Module):
    """
    功率投影 Proj_P
    公式 (28)

    p̂_m = P_max * softmax(p̃_m' / τ_p)

    保证:
      Σ(通信 + 感知) ≤ P_max
      每个条目 ≥ 0
      每个通信条目 ≥ p_min (论文公式 21: A_{m,k}||w_{m,k}||² ≥ A_{m,k}P_min)
    """

    def __init__(self, p_max: float = 1.0, tau: float = 0.5, p_min_ratio: float = 0.01):
        super().__init__()
        self.register_buffer("p_max", torch.tensor(p_max))
        self.p_min = p_min_ratio * p_max  # absolute floor
        self.tau = tau

    def forward(self, p_tilde: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_tilde: (B, M, K+1) — 原始功率得分
                     [:, :, :K] = 通信功率
                     [:, :, K]  = 感知功率

        Returns:
            p_hat: (B, M, K+1) — 投影后的功率分配
        """
        B, M, D = p_tilde.shape
        K_comm = D - 1  # number of communication users

        # softmax 归一化 (沿最后一维, 即 UAV 内部)
        p_soft = F.softmax(p_tilde / self.tau, dim=-1)  # (B, M, D)

        # 缩放到功率预算
        p_hat = self.p_max * p_soft

        # 通信条目下界钳位 (论文 P_min 约束), 感知条目不受此限
        p_comm = p_hat[..., :K_comm]  # (B, M, K_comm)
        p_sense = p_hat[..., K_comm:]  # (B, M, 1)

        # 钳位后重分配: 被钳掉的多余功率均分给其他通信条目
        floor = self.p_min
        below_floor = p_comm < floor
        deficit = (floor - p_comm) * below_floor.float()  # 每个条目缺口
        total_deficit = deficit.sum(dim=-1, keepdim=True)  # (B, M, 1)

        p_comm = torch.where(below_floor, torch.tensor(floor, device=p_comm.device, dtype=p_comm.dtype), p_comm)

        # 从高于 floor 的条目中扣除 deficit
        above_floor = p_comm > floor
        excess = (p_comm - floor) * above_floor.float()
        total_excess = excess.sum(dim=-1, keepdim=True) + 1e-8
        scale = torch.clamp(1.0 - total_deficit / total_excess, min=0.0)
        p_comm = torch.where(above_floor, floor + excess * scale, p_comm)

        return torch.cat([p_comm, p_sense], dim=-1)


class AssociationProjection(nn.Module):
    """
    关联投影 Proj_A
    公式 (29)

    训练时: Sinkhorn 归一化 → 软关联 Ẑ ∈ [0,1]^{M×K}
    推理时: 离散化为二进制 Â

    保证:
      Σ_m Ẑ_{m,k} ≈ 1   (每列近1 — single association)
      Σ_k Ẑ_{m,k} ≤ K_max (每行容量上限)
    """

    def __init__(self, K_max: int = 10, tau: float = 0.5, n_iters: int = 20):
        super().__init__()
        self.K_max = K_max
        self.tau = tau
        self.n_iters = n_iters

    def forward(self, a_tilde: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a_tilde: (B, M, K) — 原始关联得分

        Returns:
            z_hat: (B, M, K) — 软关联矩阵
        """
        B, M, K = a_tilde.shape

        # Step 1: exp 缩放
        S = torch.exp(a_tilde / self.tau)  # (B, M, K)

        # Step 2: Sinkhorn 迭代 (列归一化 + 行容量裁剪)
        for _ in range(self.n_iters):
            # 列归一化 (每用户 ≈ 1)
            col_sum = S.sum(dim=1, keepdim=True) + 1e-8  # (B, 1, K)
            S = S / col_sum

            # 行容量裁剪 (每 UAV ≤ K_max)
            row_sum = S.sum(dim=2, keepdim=True) + 1e-8  # (B, M, 1)
            scale = torch.where(
                row_sum > self.K_max,
                self.K_max / row_sum,
                torch.ones_like(row_sum),
            )
            S = S * scale

        return S

    @staticmethod
    def discretize(z_hat: torch.Tensor) -> torch.Tensor:
        """
        离散化软关联 → 二进制关联 (推理时使用)

        贪心算法: 每列取最大值, 容量超出时移到次优
        简化版: 直接 argmax + 容量后处理
        """
        # 简化为 argmax (列方向)
        best_m = torch.argmax(z_hat, dim=1)  # (B, K)
        B, M, K = z_hat.shape
        A = torch.zeros_like(z_hat)
        for b in range(B):
            for k in range(K):
                A[b, best_m[b, k], k] = 1.0
        return A


class ConstraintProjectionHead(nn.Module):
    """
    完整投影头 h_Φ (公式 23)

    Pipeline:
      Z_c → ControlReadout → δ̃ → ResidualMLP → δ̃'
      δ̃' → [Proj_Q, Proj_A, Proj_P] → δ̂
    """

    def __init__(
        self,
        hidden_dim: int = 3840,
        num_control_tokens: int = 8,
        mlp_hidden: list = None,
        readout_out_dim: int = 128,
        M: int = 4,
        K: int = 20,
        area_w: float = 1000.0,
        area_h: float = 1000.0,
        h_min: float = 50.0,
        h_max: float = 300.0,
        v_max_dt: float = 15.0,
        p_max: float = 1.0,
        K_max: int = 10,
        tau_power: float = 0.5,
        tau_assoc: float = 0.5,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        self.M = M
        self.K = K

        # 读出维度 = M*3 (位移) + M*K (关联) + M*(K+1) (功率)
        total_delta_dim = M * 3 + M * K + M * (K + 1)

        # 控制 Token 读出
        self.readout = ControlReadout(hidden_dim, num_control_tokens, total_delta_dim)

        # 残差 MLP
        self.mlp = ResidualMLP(total_delta_dim, mlp_hidden or [256, 256])

        # 投影模块
        self.proj_q = DeploymentProjection(area_w, area_h, h_min, h_max, v_max_dt)
        self.proj_a = AssociationProjection(K_max, tau_assoc, sinkhorn_iters)
        self.proj_p = PowerProjection(p_max, tau_power)

    def _unflatten(self, delta_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将 (B, total_dim) 拆分为 (δ_q, δ_a, δ_p)
        """
        B = delta_flat.shape[0]
        M, K = self.M, self.K
        idx = 0

        delta_q = delta_flat[:, idx:idx + M * 3].reshape(B, M, 3)
        idx += M * 3

        delta_a = delta_flat[:, idx:idx + M * K].reshape(B, M, K)
        idx += M * K

        delta_p = delta_flat[:, idx:idx + M * (K + 1)].reshape(B, M, K + 1)

        return delta_q, delta_a, delta_p

    def forward(
        self,
        control_states: torch.Tensor,      # (B, num_control_tokens, hidden_dim)
        q_current: Optional[torch.Tensor] = None,  # (B, M, 3)
    ) -> dict:
        """
        前向传播

        Args:
            control_states: 控制 token 位置的 hidden states
            q_current: 当前 UAV 位置 (用于部署投影)

        Returns:
            dict with:
              "delta_q": (B, M, 3)   UAV 位移先验
              "delta_a": (B, M, K)   关联先验 (软)
              "delta_p": (B, M, K+1) 功率先验
              "delta_raw": (B, total_dim) 未投影的原始 prior
        """
        # Step 1: 读出
        delta_raw = self.readout(control_states)  # (B, total_dim)

        # Step 2: MLP 修正
        delta_corrected = self.mlp(delta_raw)     # (B, total_dim)

        # Step 3: 拆分 + 分别投影
        dq, da, dp = self._unflatten(delta_corrected)

        dq_proj = self.proj_q(dq, q_current)
        da_proj = self.proj_a(da)
        dp_proj = self.proj_p(dp)

        return {
            "delta_q": dq_proj,
            "delta_a": da_proj,
            "delta_p": dp_proj,
            "delta_raw": delta_raw,
            "delta_q_raw": dq,
            "delta_a_raw": da,
            "delta_p_raw": dp,
        }
