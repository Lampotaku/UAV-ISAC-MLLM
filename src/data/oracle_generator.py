"""
Oracle 数据生成器
论文 Section 4.1 / Algorithm 1 — Best-of-N 数据生成

核心流程:
  1. 采样环境 E^(i)
  2. 构造 prompt Π^(i)
  3. 运行 N 次 SCA-FP (随机初始点)
  4. 按效用排序 u_π(1) ≥ u_π(2) ≥ ... ≥ u_π(N)
  5. 最优解 → D_SFT (监督信号)
  6. 偏好对 (满足 u_diff > Δ_min) → D_DPO
  7. 提取 prior: Ξ(Ω*) → δ = (δ_q, δ_a, δ_p)

公式参考:
  - Prior 提取 (14-16): δ_q* = Q* - Q(t); δ_a* = A*; δ_p* = {||w*||²}
  - 对选择边距 (18): Δ_min = ρ · IQR({u_j})
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import time

from ..env import ISACScenarioGenerator, EnvironmentSample
from ..solver.sca_fp import SCAFPOptimizer, SCAFPConfig, SCAFPSolution
from .prompt_builder import build_full_prompt, format_oracle_response


class OracleDataGenerator:
    """
    Best-of-N 数据生成器 (Algorithm 1)

    生成:
      - SFT 数据集: (Π, δ_best)
      - DPO 数据集: (Π, δ_winner, δ_loser)
    """

    def __init__(
        self,
        scenario_gen: ISACScenarioGenerator,
        solver: SCAFPOptimizer,
        config: dict,
        sim_config: dict = None,
    ):
        self.scenario_gen = scenario_gen
        self.solver = solver
        self.cfg = config

        self.num_restarts = config.get("num_restarts", 10)
        self.num_environments = config.get("num_environments", 20000)
        self.pair_margin_rho = config.get("pair_margin_rho", 0.2)
        self.min_pairs = config.get("dpo_min_pairs_per_sample", 2)

        # ── 微扰回弹测试参数 ──
        self.snapback_epsilon = config.get("snapback_epsilon", 1.5)        # 微扰幅度 (m)
        self.snapback_top_k = config.get("snapback_top_k", 3)              # 参与回弹测试的候选数
        self.pareto_utility_ratio = config.get("pareto_utility_ratio", 0.85)  # 低于全局最高 ×ratio 丢弃

        # ── Rejected 构造参数 ──
        self.heuristic_reject_ratio = config.get("heuristic_reject_ratio", 0.3)  # 启发式陷阱占比

        # sim_config: 仿真参数 (num_uavs, num_users, etc.)
        # 回退: 从 scenario_gen 提取
        self.sim_cfg = sim_config if sim_config is not None else config
        self.output_dir = Path(config.get("output_dir", "./data/cache"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── 物理约束缓存 (用于 clip_to_physics_bounds) ──
        self._max_disp = self.solver.max_displacement
        self._area_w = self.solver.area_w
        self._area_h = self.solver.area_h
        self._H_min = self.solver.H_min
        self._H_max = self.solver.H_max

    def generate_all(self) -> Tuple[List[Dict], List[Dict]]:
        """
        运行完整数据生成管线

        Returns:
            sft_data: List of {"prompt": str, "response": str, "utility": float}
            dpo_data: List of {"prompt": str, "chosen": str, "rejected": str}
        """
        sft_data = []
        dpo_data = []

        pbar = tqdm(range(self.num_environments), desc="Generating oracle data")
        for i in pbar:
            try:
                sft_sample, dpo_samples = self._process_one_environment(i)
                if sft_sample is not None:
                    sft_data.append(sft_sample)
                    dpo_data.extend(dpo_samples)

                if len(sft_data) > 0:
                    pbar.set_postfix({
                        "SFT": len(sft_data),
                        "DPO": len(dpo_data),
                    })
            except Exception as e:
                print(f"\n[WARN] Sample {i} failed: {e}")
                continue

        # 保存
        self._save_dataset(sft_data, "sft_dataset.jsonl")
        self._save_dataset(dpo_data, "dpo_dataset.jsonl")

        print(f"\nGenerated: {len(sft_data)} SFT samples, {len(dpo_data)} DPO pairs")
        return sft_data, dpo_data

    def _process_one_environment(self, sample_id: int) -> Tuple[Optional[Dict], List[Dict]]:
        """
        处理单个环境样本 — Grilling 终稿流程

        1. 采样环境 & 构造 prompt
        2. N 次 Random Restart SCA-FP
        3. Pareto 过滤 (baseline + utility ratio)
        4. 微扰回弹测试 → 选 Chosen
        5. 构造 Rejected (70% SCA-FP 次优解 + 30% 启发式陷阱)
        6. 返回 (sft_sample, [dpo_sample])

        Returns:
            (sft_sample, list_of_dpo_samples) — 单个 DPO 对
        """
        # Step 1: 采样环境 & 构造 prompt
        env_sample: EnvironmentSample = self.scenario_gen.sample(sample_id)
        prompt = build_full_prompt(env_sample, self.sim_cfg)
        env_dict = self._env_sample_to_dict(env_sample)
        q_current = env_dict["q_current"]

        # Step 2: N 次 SCA-FP 重启
        solutions: List[SCAFPSolution] = []
        for j in range(self.num_restarts):
            seed = sample_id * self.num_restarts + j
            sol = self.solver.solve(env_dict, warm_start=None, seed=seed)
            solutions.append(sol)

        # 按效用排序
        solutions.sort(key=lambda s: s.utility, reverse=True)

        # Step 3: Pareto 过滤 (仅 utility ratio 门 — 15m 墙下 baseline 检查会误杀)
        candidates = self._pareto_filter(solutions)
        if len(candidates) < 1:
            return None, []

        # Step 4: 选 Chosen (15m 墙下 snapback 无区分度 → 直取效用最高)
        chosen_sol = candidates[0]

        # Step 5: 构造 Rejected
        rejected_delta_q, rejected_util = self._construct_rejected(
            env_dict, solutions, q_current, sample_id,
        )

        # Step 6: 构造输出
        # SFT 样本 — Chosen 的 prior
        delta_q_chosen, delta_a_chosen, delta_p_chosen = self._extract_prior(
            chosen_sol, env_sample,
        )
        response_chosen = format_oracle_response(
            sample_id, delta_q_chosen, delta_a_chosen, delta_p_chosen,
        )

        sft_sample = {
            "id": f"env_{sample_id}",
            "prompt": prompt,
            "response": response_chosen,
            "utility": float(chosen_sol.utility),
            "q_current": q_current.tolist(),
            "delta_q": delta_q_chosen.tolist(),
            "delta_a": delta_a_chosen.tolist(),
            "delta_p": delta_p_chosen.tolist(),
        }

        # DPO 样本 — 单个对: Chosen vs Rejected
        # Rejected 只有 δ_q 为陷阱; δ_a/δ_p 复用 Chosen 的 — 被 Masked DPO 忽略
        rejected_response = self._format_rejected_response(
            sample_id, rejected_delta_q, delta_a_chosen, delta_p_chosen,
        )

        # Gap: 启发式陷阱 → 保守估计 5% gap; SCA-FP 次优解 → 实际 gap
        if rejected_util is not None:
            gap = float(chosen_sol.utility) - rejected_util
        else:
            gap = abs(float(chosen_sol.utility)) * 0.05

        # 如果 Rejected δ_q 在物理上退化为 Chosen → 丢弃
        if np.allclose(rejected_delta_q, delta_q_chosen, atol=1e-3):
            dpo_sample = None
        elif gap <= 0:  # Rejected 居然比 Chosen 好 → 退化
            dpo_sample = None
        else:
            dpo_sample = {
                "id": f"env_{sample_id}_dpo",
                "prompt": prompt,
                "chosen": response_chosen,
                "rejected": rejected_response,
                "utility_chosen": float(chosen_sol.utility),
                "utility_gap": gap,
                "q_current": q_current.tolist(),
                "delta_q": delta_q_chosen.tolist(),
                "delta_a": delta_a_chosen.tolist(),
                "delta_p": delta_p_chosen.tolist(),
            }

        dpo_samples = [dpo_sample] if dpo_sample is not None else []
        return sft_sample, dpo_samples

    # ═══════════════════════════════════════════════════════════════
    # Pareto 过滤
    # ═══════════════════════════════════════════════════════════════

    def _pareto_filter(
        self, solutions: List[SCAFPSolution],
    ) -> List[SCAFPSolution]:
        """
        Pareto 过滤:
          丢弃 utility < max_utility × pareto_utility_ratio 的劣质坑
          (15m 墙下 baseline [0,0,0] 与最优解收敛到同一点 → 不设 baseline 门)
        """
        if not solutions:
            return []

        max_util = solutions[0].utility
        # 用绝对值计算阈值, 完美兼容正负数 utility
        # max_util<0 时 max_util*ratio 会比 max_util 还高, 导致最优解被自己踢掉
        threshold = max_util - abs(max_util) * (1.0 - self.pareto_utility_ratio)

        filtered = [
            s for s in solutions
            if s.utility >= threshold
        ]
        return filtered

    def _compute_baseline_utility(self, env_dict: Dict) -> float:
        """计算 [0,0,0] 不动方案的 utility"""
        q_cur = env_dict["q_current"]
        zero_warm = {
            "delta_q": np.zeros_like(q_cur),
            "delta_a": np.zeros((self.solver.M, self.solver.K)),
            "delta_p": np.zeros((self.solver.M, self.solver.K + 1)),
        }
        zero_sol = self.solver.solve(
            env_dict, warm_start=zero_warm, seed=999999,
        )
        return zero_sol.utility

    # ═══════════════════════════════════════════════════════════════
    # 微扰回弹测试
    # ═══════════════════════════════════════════════════════════════

    def _run_snapback_test(
        self, env_dict: Dict, candidate: SCAFPSolution,
        epsilon: float, seed_offset: int,
    ) -> int:
        """
        对候选解施加 ε 扰动，重跑 SCA-FP，返回迭代步数。

        扰动: 每个 UAV 随机 3D 方向 + 固定幅度 ε
        """
        M = self.solver.M
        q_opt = candidate.Q.copy()
        q_current = env_dict["q_current"]

        rng = np.random.RandomState(seed_offset)
        perturbed_q = q_opt.copy()
        for m in range(M):
            phi = rng.uniform(0, 2 * np.pi)
            cos_theta = rng.uniform(-1, 1)
            theta = np.arccos(cos_theta)
            direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            perturbed_q[m] += epsilon * direction

        # Clamp 到物理边界
        perturbed_q[:, 0] = np.clip(perturbed_q[:, 0], 0, self._area_w)
        perturbed_q[:, 1] = np.clip(perturbed_q[:, 1], 0, self._area_h)
        perturbed_q[:, 2] = np.clip(perturbed_q[:, 2], self._H_min, self._H_max)

        delta_q_perturbed = perturbed_q - q_current

        # 不传 perfect A/P — 否则 SCA-FP 的 Q 子问题退化为凸优化, 2 步内收敛
        # 传零初始化 → 迫使求解器在联合空间重新寻优 → 真正测试盆地宽度
        warm_start = {
            "delta_q": delta_q_perturbed,
            "delta_a": np.zeros((self.solver.M, self.solver.K)),
            "delta_p": np.zeros((self.solver.M, self.solver.K + 1)),
        }

        rerun_sol = self.solver.solve(
            env_dict, warm_start=warm_start,
            seed=seed_offset + 10000,
        )
        return rerun_sol.iterations

    # ═══════════════════════════════════════════════════════════════
    # Rejected δ_q 构造 (混合策略)
    # ═══════════════════════════════════════════════════════════════

    def _construct_rejected(
        self, env_dict: Dict,
        solutions: List[SCAFPSolution],
        q_current: np.ndarray,
        sample_id: int,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Rejected δ_q 混合构造:
          - ~70%: SCA-FP 次优解的 δ_q (效用最低的有效解)
          - ~30%: 启发式物理陷阱 (随机选一种)
        所有 Rejected 必须通过 clip_to_physics_bounds 投影。

        Returns:
            (delta_q, utility_or_none): utility=None 表示启发式陷阱 (保守估计)
        """
        rng = np.random.RandomState(sample_id * 777 + 13)

        if rng.random() < self.heuristic_reject_ratio:
            delta_q = self._construct_heuristic_rejected(
                env_dict, q_current, rng,
            )
            return delta_q, None  # 启发式陷阱: 效用无解析值
        else:
            # SCA-FP 次优解 — 取效用最低的有效解
            if len(solutions) >= 2:
                worst_valid = solutions[-1]
                # 确保 worst 不是 best (退化检查)
                for s in reversed(solutions):
                    if s.utility < solutions[0].utility - 0.01:
                        worst_valid = s
                        break
                # 检测 15m 墙退化: 若所有 restart 撞同一面约束墙,
                # worst ≈ best → DPO pair 无偏好信号 → 回退到启发式陷阱
                if np.allclose(worst_valid.Q, solutions[0].Q, atol=0.5):
                    delta_q = self._construct_heuristic_rejected(
                        env_dict, q_current, rng,
                    )
                    return delta_q, None
                delta_q = self._clip_to_physics_bounds(
                    worst_valid.Q - q_current, q_current,
                )
                return delta_q, float(worst_valid.utility)
            else:
                # 回退: 用启发式陷阱
                delta_q = self._construct_heuristic_rejected(
                    env_dict, q_current, rng,
                )
                return delta_q, None

    def _construct_heuristic_rejected(
        self, env_dict: Dict, q_current: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        启发式物理陷阱 — 三选一:
          a. 短视直线 (Greedy Line): 以最大速度飞向用户/目标中心
          b. 原地不动 (Zero-Movement): [0, 0, 0]
          c. 旧世界残影 (Old Ghost): [0, 0, ±15]
        """
        M = self.solver.M
        choice = rng.randint(0, 3)

        if choice == 0:
            # 短视直线 — 忽略杂波, 全速冲向用户中心
            user_centroid = np.mean(env_dict["user_positions"], axis=0)
            delta_q = np.zeros((M, 3))
            for m in range(M):
                direction_2d = user_centroid - q_current[m, :2]
                dist_2d = np.linalg.norm(direction_2d)
                if dist_2d > 0.01:
                    direction_2d /= dist_2d
                else:
                    direction_2d = np.array([1.0, 0.0])
                # 满速冲向用户中心, 高度不变
                delta_q[m, 0] = direction_2d[0] * self._max_disp
                delta_q[m, 1] = direction_2d[1] * self._max_disp
                delta_q[m, 2] = 0.0  # 保持高度 — 忽略杂波

        elif choice == 1:
            # 原地不动 — 惩罚惰性
            delta_q = np.zeros((M, 3))

        else:
            # 旧世界残影 — 明确标记旧分布的坍塌模式
            delta_q = np.zeros((M, 3))
            sign = 1.0 if rng.random() > 0.5 else -1.0
            for m in range(M):
                delta_q[m, 2] = sign * self._max_disp  # 全速向上或向下

        # ⚠️ 所有 Rejected 必经 Constraint Projections
        return self._clip_to_physics_bounds(delta_q, q_current)

    # ═══════════════════════════════════════════════════════════════
    # 物理约束投影 (Deterministic Forward Projection)
    # ═══════════════════════════════════════════════════════════════

    def _clip_to_physics_bounds(
        self, delta_q: np.ndarray, q_current: np.ndarray,
    ) -> np.ndarray:
        """
        将 δ_q 投影到可行物理空间 — 镜像 DeploymentProjection.forward()

        三大约束:
          1. 3D 移动性: ||Δq[m]||₂ ≤ v_max × Δt
          2. 区域边界: x ∈ [0, area_w], y ∈ [0, area_h]
          3. 高度边界: h ∈ [H_min, H_max]
        """
        delta_q = np.asarray(delta_q, dtype=np.float64).copy()

        # 约束 1: 3D 移动性 — 范数裁剪
        norms = np.linalg.norm(delta_q, axis=1, keepdims=True)
        scale = np.where(
            norms > self._max_disp,
            self._max_disp / (norms + 1e-12),
            1.0,
        )
        delta_q *= scale

        # 约束 2 & 3: 区域 + 高度 — 计算新位置后裁剪
        new_pos = q_current + delta_q
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0.0, self._area_w)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0.0, self._area_h)
        new_pos[:, 2] = np.clip(new_pos[:, 2], self._H_min, self._H_max)

        return new_pos - q_current

    # ═══════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════

    def _format_rejected_response(
        self, sample_id: int,
        delta_q: np.ndarray, delta_a: np.ndarray, delta_p: np.ndarray,
    ) -> str:
        """格式化 Rejected 响应 JSON — δ_q 是陷阱, δ_a/δ_p 是 Chosen 的"""
        delta_q_disp = delta_q  # 已经在 _construct_rejected 中处理
        return format_oracle_response(sample_id, delta_q_disp, delta_a, delta_p)

    def _extract_prior(
        self,
        solution: SCAFPSolution,
        env_sample: EnvironmentSample,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prior 提取函数 Ξ(Ω*) → δ

        公式:
          δ_q* = Q* - Q(t)        (14) — UAV 位移
          δ_a* = A*               (15) — 关联矩阵
          δ_p* = {||w*_{m,k}||², ||w*_{m,r}||²}  (16) — 功率

        Beamformer 方向不保留, 由 SCA-FP 重建
        """
        M, K = self.solver.M, self.solver.K

        # δ_q: 位移 = 最优位置 - 当前位置
        delta_q = solution.Q - env_sample.q_current

        # δ_a: 关联矩阵
        delta_a = solution.A.copy()

        # δ_p: 通信功率 (M×K) + 感知功率 (M×1)
        delta_p = np.zeros((M, K + 1), dtype=np.float32)
        delta_p[:, :K] = solution.W_c_power
        delta_p[:, K] = solution.W_s_power

        # Round to 4 decimal places (0.1mm) — drastically reduces token count
        # for BPE tokenizers that fragment high-precision floats like 0.1910400390625
        # into 5-8 subword tokens each. 4 decimals is ~10μm, well below UAV control limits.
        # .astype(np.float32) removed: np.round already produces clean 4-decimal values;
        # downstream JSON serialization strips the dtype anyway.
        return (np.round(delta_q, 4),
                np.round(delta_a, 4),
                np.round(delta_p, 4))

    def _env_sample_to_dict(self, env_sample: EnvironmentSample) -> Dict:
        """将 EnvironmentSample 转换为 solver 期望的 dict 格式"""
        return {
            "q_current": env_sample.q_current.copy(),
            "user_positions": env_sample.u_positions.copy(),
            "target_positions": env_sample.s_positions.copy(),
            "channel_gains": env_sample.channel_gains_users.copy(),
            "user_weights": env_sample.user_weights.copy().astype(np.float32),
            "association": env_sample.association.copy(),
        }

    def _save_dataset(self, data: List[Dict], filename: str):
        """保存数据集为 JSONL 格式"""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {len(data)} records to {filepath}")

    def load_dataset(self, filename: str) -> List[Dict]:
        """加载 JSONL 数据集"""
        filepath = self.output_dir / filename
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
