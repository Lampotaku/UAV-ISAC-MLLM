"""
SCA-FP 优化器
论文 Section 3 — 下游求解器 S(·)

核心算法: 逐次凸近似 (SCA) + 分数规划 (FP)
优化变量 Ω = {Q, A, W_c, W_s}

联合效用 (公式 10):
  f(Ω) = Σ A_{m,k} ω_k log₂(1+γ_{m,k})
         + λ_s Σ SINR^s_{m,ℓ}
         - λ_f Σ I[|K_m| = 0]

约束:
  (15)-(17)  通信 QoS
  (18)-(19)  感知 QoS
  (20)-(24)  功率/关联
  (25)-(28)  部署/移动性

求解策略: 交替优化 (Alternating Optimization)
  1. 固定 Q, 优化 W_c, W_s (波束成形 → 凸子问题)
  2. 固定 W_c, W_s, 优化 Q  (部署 → SCA子问题)
  3. 固定 Q, W_c, W_s, 优化 A (关联 → 指派问题)
  4. 循环至收敛
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import time
from scipy.optimize import minimize, linear_sum_assignment


@dataclass
class SCAFPConfig:
    """SCA-FP 求解器配置"""
    max_outer_iters: int = 30         # 最大外循环迭代
    max_inner_iters: int = 50         # SCA 内循环 (部署子问题)
    tol: float = 1e-4                 # 收敛容差
    lambda_sensing: float = 0.5       # λ_s — 感知权重
    lambda_idle_penalty: float = 5.0  # λ_f — 闲置 UAV 惩罚
    sinr_c_min: float = 1.0           # Γ_c^min (线性, 0dB)
    sinr_s_min: float = 10.0          # Γ_s^min (线性, 10dB)
    ground_clutter_db: float = 12.0   # 地面杂波 (dB) — H_min 处额外损耗, H_max 处为 0
    verbose: bool = False


@dataclass
class SCAFPSolution:
    """SCA-FP 求解结果"""
    Q: np.ndarray           # (M, 3) — UAV 位置
    A: np.ndarray           # (M, K) — 关联矩阵
    W_c_power: np.ndarray   # (M, K) — 通信波束功率 ||w_{m,k}||²
    W_s_power: np.ndarray   # (M,) — 感知波束功率 ||w_{m,r}||²
    utility: float          # 最终效用 f(Ω)
    iterations: int
    converged: bool
    solve_time: float


class SCAFPOptimizer:
    """
    SCA-FP 级联优化器

    对应论文中的 S(·) 函数 (公式 13):
      Ω*(t) = S(Γ(δ(t)), E(t))

    支持:
      - 随机初始化 (用于 Best-of-N 数据生成)
      - 热启动初始化 (从 MLLM prior 启动)
    """

    def __init__(
        self,
        config: SCAFPConfig,
        M: int = 4,
        K: int = 20,
        T: int = 6,
        N_t: int = 8,
        N_r: int = None,
        carrier_freq_ghz: float = 5.8,
        area_size: Tuple[float, float] = (1000.0, 1000.0),
        altitude_range: Tuple[float, float] = (50.0, 300.0),
        p_max: float = 1.0,  # Watts (30dBm)
        noise_power: float = 1e-12,
        load_cap: int = 10,
        v_max: float = 15.0,
        slot_duration: float = 1.0,
    ):
        self.cfg = config
        self.M = M
        self.K = K
        self.T = T
        self.N_t = N_t
        self.N_r = N_r if N_r is not None else N_t  # default: symmetric array
        self.carrier_freq_ghz = carrier_freq_ghz
        self.wavelength = 3e8 / (carrier_freq_ghz * 1e9)  # dynamic wavelength from carrier freq
        self.area_w, self.area_h = area_size
        self.H_min, self.H_max = altitude_range
        self.P_max = p_max
        self.N0 = noise_power
        self.K_max = load_cap
        self.v_max = v_max
        self.slot_duration = slot_duration
        self.max_displacement = v_max * slot_duration  # 15m per slot

        self.rng = np.random.RandomState()

    def solve(
        self,
        environment: Dict,
        warm_start: Optional[Dict] = None,
        seed: Optional[int] = None,
    ) -> SCAFPSolution:
        """
        主导求解入口

        Args:
            environment: E(t) 环境样本 (包含信道增益、位置等)
            warm_start: 可选热启动 δ = {delta_q, delta_a, delta_p}
            seed: 随机种子 (用于 Best-of-N restarts)

        Returns:
            SCAFPSolution
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        t0 = time.time()

        # 提取环境数据
        M, K = self.M, self.K
        gains_comm = environment.get("channel_gains", np.ones((M, K)))
        target_positions = environment.get("target_positions", np.zeros((self.T, 2)))
        user_weights = environment.get("user_weights", np.ones(K))

        # ---- 初始化 ----
        if warm_start is not None:
            Q, A, P_comm, P_sense = self._warmstart_to_init(
                warm_start, environment
            )
        else:
            Q, A, P_comm, P_sense = self._random_init(environment)

        prev_utility = -np.inf

        for outer_iter in range(self.cfg.max_outer_iters):
            # Step 1: 固定 Q, A → 优化波束功率
            P_comm, P_sense = self._optimize_beamforming(
                Q, A, gains_comm, target_positions
            )

            # Step 2: 固定 A, P → SCA 优化 UAV 位置
            Q = self._optimize_deployment_sca(
                Q, A, P_comm, P_sense, gains_comm, target_positions, environment
            )

            # Step 3: 固定 Q, P → 优化关联
            A = self._optimize_association(
                Q, gains_comm, P_comm, P_sense, user_weights
            )

            # 计算效用
            utility = self._compute_utility(
                Q, A, P_comm, P_sense, gains_comm, target_positions, user_weights
            )

            if self.cfg.verbose:
                print(f"  SCA-FP iter {outer_iter}: utility = {utility:.4f}")

            # NaN guard: break early if utility diverges (numerical instability)
            if not np.isfinite(utility):
                break
            if abs(utility - prev_utility) < self.cfg.tol:
                break
            prev_utility = utility

        elapsed = time.time() - t0

        return SCAFPSolution(
            Q=Q,
            A=A,
            W_c_power=P_comm,
            W_s_power=P_sense,
            utility=utility if np.isfinite(utility) else -np.inf,
            iterations=outer_iter + 1,
            converged=(outer_iter + 1 < self.cfg.max_outer_iters) and np.isfinite(utility),
            solve_time=elapsed,
        )

    # ================================================================
    # 随机初始化 (用于 Best-of-N 重启)
    # ================================================================

    def _random_init(self, env: Dict) -> Tuple:
        """
        生成随机初始点

        从当前 UAV 位置 q_current 出发，在移动约束 v_max * Δt 内随机扰动。
        物理上 UAV 不能瞬移 — 每个时间槽最多移动 v_max * Δt 米。
        """
        M, K = self.M, self.K
        q_current = env.get("q_current", np.zeros((M, 3)))

        # UAV 位置: q_current + 有界随机扰动
        Q = q_current.copy()
        max_disp = self.max_displacement  # v_max * Δt (15m)
        for m in range(M):
            # 3D 球形均匀采样: 方向均匀 + 半径均匀, 保证 ‖Δq‖₂ ≤ max_disp
            # Box 采样 (per-axis 独立) 会产生对角线超出 (√3·15≈26m), 违反物理约束
            phi = self.rng.uniform(0, 2 * np.pi)
            cos_theta = self.rng.uniform(-1, 1)
            theta = np.arccos(cos_theta)
            r = self.rng.uniform(0, max_disp)
            Q[m, 0] += r * np.sin(theta) * np.cos(phi)
            Q[m, 1] += r * np.sin(theta) * np.sin(phi)
            Q[m, 2] += r * np.cos(theta)

        # Clamp 到区域/硬件约束
        Q[:, 0] = np.clip(Q[:, 0], 0, self.area_w)
        Q[:, 1] = np.clip(Q[:, 1], 0, self.area_h)
        Q[:, 2] = np.clip(Q[:, 2], self.H_min, self.H_max)

        # 关联: 最近 UAV
        A = np.zeros((M, K), dtype=np.float32)
        user_positions = env.get("user_positions", np.zeros((K, 2)))
        for k in range(K):
            distances = [np.linalg.norm(Q[m, :2] - user_positions[k]) for m in range(M)]
            best_m = int(np.argmin(distances))
            A[best_m, k] = 1.0

        # 功率: 均分
        P_comm = np.zeros((M, K))
        P_sense = np.zeros(M)
        for m in range(M):
            load = max(int(A[m].sum()), 1)
            p_per = self.P_max / (load + 1)
            for k in range(K):
                if A[m, k] > 0.5:
                    P_comm[m, k] = p_per
            P_sense[m] = p_per

        return Q, A, P_comm, P_sense

    def _warmstart_to_init(self, warm_start: Dict, env: Dict) -> Tuple:
        """
        将 MLLM 热启动 δ 转换为初始点

        δ = {delta_q, delta_a, delta_p}
          delta_q  → 位移 (相对当前 Q)
          delta_a  → 软关联矩阵
          delta_p  → 通信/感知功率分配
        """
        current_Q = env.get("q_current", np.zeros((self.M, 3)))

        # 位移累加
        delta_q = np.array(warm_start.get("delta_q", np.zeros((self.M, 3))))
        # Clamp displacement to movement constraint v_max * Δt (3D Euclidean sphere)
        # 保持方向不变, 将 3D 范数裁剪到 max_disp 以内
        max_disp = self.max_displacement
        delta_q_norm = np.linalg.norm(delta_q, axis=1, keepdims=True)
        scale = np.where(delta_q_norm > max_disp, max_disp / (delta_q_norm + 1e-12), 1.0)
        delta_q *= scale
        Q = current_Q + delta_q

        # Clamp 区域/硬件约束
        Q[:, 2] = np.clip(Q[:, 2], self.H_min, self.H_max)
        Q[:, 0] = np.clip(Q[:, 0], 0, self.area_w)
        Q[:, 1] = np.clip(Q[:, 1], 0, self.area_h)

        # 关联离散化 (取每列最大值)
        delta_a = np.array(warm_start.get("delta_a", np.zeros((self.M, self.K))))
        A = np.zeros_like(delta_a)
        best_m_per_k = np.argmax(delta_a, axis=0)
        for k in range(self.K):
            A[best_m_per_k[k], k] = 1.0

        # 功率
        delta_p = np.array(warm_start.get("delta_p", np.zeros((self.M, self.K + 1))))
        P_comm = delta_p[:, :self.K]
        P_sense = delta_p[:, self.K]
        # 确保功率预算
        for m in range(self.M):
            total = P_comm[m].sum() + P_sense[m]
            if total > self.P_max:
                scale = self.P_max / total
                P_comm[m] *= scale
                P_sense[m] *= scale

        return Q, A, P_comm, P_sense

    # ================================================================
    # 子问题求解器
    # ================================================================

    def _optimize_beamforming(
        self,
        Q: np.ndarray,
        A: np.ndarray,
        channel_gains: np.ndarray,
        target_positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        波束成形优化 (固定 Q, A)

        使用闭式解: 功率注水 + ZF 预编码功率分配
        通信: p_{m,k} ∝ 1/|h_{m,k}|² (信道反转, 在关联用户间)
        感知: 剩余功率按目标重要性分配

        Returns:
            P_comm: (M, K) 通信波束功率
            P_sense: (M,) 感知波束功率
        """
        M, K = self.M, self.K
        P_comm = np.zeros((M, K))
        P_sense = np.zeros(M)

        for m in range(M):
            active_users = np.where(A[m] > 0.5)[0]
            if len(active_users) == 0:
                P_sense[m] = self.P_max
                continue

            # 通信功率分配 (注水简化版)
            gains_active = channel_gains[m, active_users]
            # 信道反转: 更多功率给弱用户
            inv_gains = 1.0 / (gains_active + 1e-12)
            weights = inv_gains / inv_gains.sum()

            # 70% 功率给通信, 30% 给感知 (初始分拆)
            p_comm_total = self.P_max * 0.7
            for i, k in enumerate(active_users):
                P_comm[m, k] = p_comm_total * weights[i]
            P_sense[m] = self.P_max * 0.3

        return P_comm, P_sense

    def _optimize_deployment_sca(
        self,
        Q_init: np.ndarray,
        A: np.ndarray,
        P_comm: np.ndarray,
        P_sense: np.ndarray,
        channel_gains: np.ndarray,
        target_positions: np.ndarray,
        env: Dict,
    ) -> np.ndarray:
        """
        SCA 部署优化 (固定波束和关联)

        逐次凸近似 UAV 位置:
          - 通信速率对 UAV 位置的非凸依赖由一阶泰勒近似
          - 感知 SINR 类似处理
          - 每步求解一个凸 SOCP
          - 约束: 高度, 区域, 间距, 移动性

        简化版: 使用 L-BFGS-B 直接优化 (约束通过 penalty + bounds)
        """
        Q = Q_init.copy()
        user_positions = env.get("user_positions", np.zeros((self.K, 2)))
        q_current = env.get("q_current", Q_init.copy())  # initial positions for movement constraint

        for _ in range(self.cfg.max_inner_iters):
            # 对每架 UAV 单独优化 (解耦, 简化)
            for m in range(self.M):
                def objective(x):
                    """对 UAV m 的局部目标 (负效用)"""
                    q_new = np.array([x[0], x[1], x[2]])
                    # 通信项: -Σ_k A_{m,k} log₂(1 + γ_{m,k})
                    obj_comm = 0.0
                    # 地面杂波: 低飞时额外的障碍物损耗 (建筑物/树木)
                    h_norm = max(0.0, min(1.0, (q_new[2] - self.H_min) / (self.H_max - self.H_min)))
                    clutter_db = self.cfg.ground_clutter_db * (1.0 - h_norm)
                    for k in range(self.K):
                        if A[m, k] < 0.5:
                            continue
                        dist_2d = np.linalg.norm(q_new[:2] - user_positions[k])
                        dist_3d = np.sqrt(dist_2d ** 2 + q_new[2] ** 2)
                        # 3GPP UMa LoS: PL = 28 + 22*log10(d_3D) + 20*log10(f_c)
                        pl_db = 28 + 22 * np.log10(max(dist_3d, 1.0)) + 20 * np.log10(self.carrier_freq_ghz)
                        pl_db += clutter_db  # 地面杂波附加损耗
                        pl_linear = 10 ** (-pl_db / 10)
                        sinr = P_comm[m, k] * pl_linear / self.N0
                        obj_comm -= np.log2(1 + sinr)

                    # 感知项: -Σ_ℓ SINR^s
                    obj_sense = 0.0
                    for t in range(self.T):
                        t_pos = target_positions[t]
                        dist_2d = np.linalg.norm(q_new[:2] - t_pos)
                        dist_3d = np.sqrt(dist_2d ** 2 + q_new[2] ** 2)
                        pl_db = 20 * np.log10((4 * np.pi * max(dist_3d, 1.0)) / self.wavelength) + 20
                        pl_db += clutter_db  # 地面杂波附加损耗
                        pl_linear = 10 ** (-pl_db / 10)
                        sinr_s = P_sense[m] * pl_linear * self.N_t * self.N_r / self.N0
                        obj_sense -= sinr_s

                    obj = obj_comm + self.cfg.lambda_sensing * obj_sense

                    # 3D 球形移动约束惩罚 (切掉 Box bounds 的八个角)
                    # L-BFGS-B bounds 只能做 per-axis 独立约束 → 搜索空间是正方体
                    # 物理约束是 ‖Δq‖₂ ≤ v_max*Δt → 可行域是球体
                    # 正方体的角 (Δx=±15, Δy=±15, Δz=±15) 的 3D 距离达 √675≈26m
                    # 惩罚项确保优化器不会逃逸到球外
                    q_cur_m = q_current[m]
                    dist_moved = np.linalg.norm(q_new - q_cur_m)
                    if dist_moved > max_disp:
                        obj += 1e5 * (dist_moved - max_disp) ** 2

                    return obj

                # L-BFGS-B 优化 m-th UAV
                # 约束: 区域边界 与 移动性约束 (v_max * Δt) 的交集
                max_disp = self.max_displacement  # 15m
                q0 = q_current[m]
                bounds = [
                    (max(0.0, q0[0] - max_disp), min(self.area_w, q0[0] + max_disp)),       # x
                    (max(0.0, q0[1] - max_disp), min(self.area_h, q0[1] + max_disp)),       # y
                    (max(self.H_min, q0[2] - max_disp), min(self.H_max, q0[2] + max_disp)),  # H
                ]
                res = minimize(
                    objective,
                    Q[m],
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 20},
                )
                Q[m] = res.x

        return Q

    def _optimize_association(
        self,
        Q: np.ndarray,
        channel_gains: np.ndarray,
        P_comm: np.ndarray,
        P_sense: np.ndarray,
        user_weights: np.ndarray,
    ) -> np.ndarray:
        """
        关联优化 (固定 Q, W)

        min Σ_{m,k} A_{m,k} * cost(m,k)
        s.t.  Σ_m A_{m,k} = 1       (每用户一个 UAV)
              Σ_k A_{m,k} ≤ K_max   (每 UAV 负载上限)

        使用: Hungarian 算法 (无容量) + 后处理 (容量)
        完整版: 最小费用流
        """
        M, K = self.M, self.K

        # 代价矩阵: 负速率 (我们想最大化速率的加权和)
        cost = np.zeros((M, K))
        for m in range(M):
            for k in range(K):
                # 可达速率
                sinr = channel_gains[m, k] * P_comm[m, k] / self.N0
                rate = np.log2(1 + sinr + 1e-12)
                cost[m, k] = -user_weights[k] * rate

        # Hungarian 算法 (处理单用户约束)
        if M >= K:
            row_ind, col_ind = linear_sum_assignment(cost)
            A = np.zeros((M, K), dtype=np.float32)
            for r, c in zip(row_ind, col_ind):
                A[r, c] = 1.0
        else:
            # M < K: 需要复制 UAV 行并处理容量
            # 简化: 每用户分配最佳 UAV, 然后裁切溢出
            A = np.zeros((M, K), dtype=np.float32)
            best_m = np.argmin(cost, axis=0)  # 每列最小代价
            for k in range(K):
                A[best_m[k], k] = 1.0

            # 容量裁切
            for m in range(M):
                if A[m].sum() > self.K_max:
                    # 移出最差的用户
                    users_of_m = np.where(A[m] > 0.5)[0]
                    excess = int(A[m].sum() - self.K_max)
                    # 按代价排序, 移出代价最高的
                    sorted_users = users_of_m[np.argsort(cost[m, users_of_m])[::-1]]
                    for uk in sorted_users[:excess]:
                        A[m, uk] = 0.0
                        # 重分配给次优 UAV
                        other_ms = [om for om in range(M) if om != m]
                        if other_ms:
                            best_other = other_ms[np.argmin([cost[om, uk] for om in other_ms])]
                            A[best_other, uk] = 1.0

        return A

    # ================================================================
    # 效用计算
    # ================================================================

    def _compute_utility(
        self,
        Q: np.ndarray,
        A: np.ndarray,
        P_comm: np.ndarray,
        P_sense: np.ndarray,
        channel_gains: np.ndarray,
        target_positions: np.ndarray,
        user_weights: np.ndarray,
    ) -> float:
        """
        计算联合效用 f(Ω) — 公式 (10)

        f(Ω) = Σ A_{m,k} ω_k log₂(1+γ_{m,k})
               + λ_s Σ SINR^s_{m,ℓ}
               - λ_f Σ I[|K_m| = 0]
        """
        utility = 0.0

        # 通信项
        for m in range(self.M):
            for k in range(self.K):
                if A[m, k] > 0.5:
                    sinr = channel_gains[m, k] * P_comm[m, k] / self.N0
                    utility += user_weights[k] * np.log2(1 + sinr + 1e-12)

        # 感知项
        for t in range(self.T):
            t_pos = target_positions[t]
            for m in range(self.M):
                dist_2d = np.linalg.norm(Q[m, :2] - t_pos)
                dist_3d = np.sqrt(dist_2d ** 2 + Q[m, 2] ** 2)
                pl_db = 20 * np.log10((4 * np.pi * max(dist_3d, 1.0)) / self.wavelength) + 20
                pl_linear = 10 ** (-pl_db / 10)
                sinr_s = P_sense[m] * pl_linear * self.N_t * self.N_r / self.N0
                utility += self.cfg.lambda_sensing * sinr_s

        # 闲置惩罚
        for m in range(self.M):
            if A[m].sum() < 0.5:
                utility -= self.cfg.lambda_idle_penalty

        return float(utility)

    def compute_utility_from_solution(
        self, sol: SCAFPSolution, env: Dict
    ) -> float:
        """对已有解重新计算效用 (用于 DPO 排序)"""
        return self._compute_utility(
            sol.Q,
            sol.A,
            sol.W_c_power,
            sol.W_s_power,
            env.get("channel_gains", np.ones((self.M, self.K))),
            env.get("target_positions", np.zeros((self.T, 2))),
            env.get("user_weights", np.ones(self.K)),
        )
