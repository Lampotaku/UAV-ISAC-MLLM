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
        self.num_environments = config.get("num_environments", 5000)
        self.pair_margin_rho = config.get("pair_margin_rho", 0.2)
        self.min_pairs = config.get("dpo_min_pairs_per_sample", 2)

        # sim_config: 仿真参数 (num_uavs, num_users, etc.)
        # 回退: 从 scenario_gen 提取
        self.sim_cfg = sim_config if sim_config is not None else config
        self.output_dir = Path(config.get("output_dir", "./data/cache"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        处理单个环境样本 (Algorithm 1 内循环)

        Returns:
            (sft_sample, list_of_dpo_samples)
        """
        # Step 1: 采样环境 & 构造 prompt
        env_sample: EnvironmentSample = self.scenario_gen.sample(sample_id)
        prompt = build_full_prompt(env_sample, self.sim_cfg)

        # Step 2: N 次 SCA-FP 重启
        solutions: List[SCAFPSolution] = []
        env_dict = self._env_sample_to_dict(env_sample)

        for j in range(self.num_restarts):
            seed = sample_id * self.num_restarts + j
            sol = self.solver.solve(env_dict, warm_start=None, seed=seed)
            solutions.append(sol)

        # Step 3: 按效用排序
        solutions.sort(key=lambda s: s.utility, reverse=True)
        utilities = np.array([s.utility for s in solutions])

        best_sol = solutions[0]

        # Step 4: 构造 SFT 样本 (最优 prior)
        delta_q, delta_a, delta_p = self._extract_prior(best_sol, env_sample)
        response = format_oracle_response(sample_id, delta_q, delta_a, delta_p)

        sft_sample = {
            "id": f"env_{sample_id}",
            "prompt": prompt,
            "response": response,
            "utility": float(best_sol.utility),
            "q_current": env_sample.q_current.tolist(),
            "delta_q": delta_q.tolist(),
            "delta_a": delta_a.tolist(),
            "delta_p": delta_p.tolist(),
        }

        # Step 5: 构造 DPO 对
        dpo_samples = self._build_dpo_pairs(
            sample_id, prompt, solutions, utilities, env_sample
        )

        return sft_sample, dpo_samples

    def _build_dpo_pairs(
        self,
        sample_id: int,
        prompt: str,
        solutions: List[SCAFPSolution],
        utilities: np.ndarray,
        env_sample: EnvironmentSample,
    ) -> List[Dict]:
        """
        构建 DPO 偏好对 (Algorithm 1 行 9-11)

        pair if: u_π(j) - u_π(j') > Δ_min
        其中 Δ_min = ρ · IQR({u_j})
        """
        iqr = np.subtract(*np.percentile(utilities, [75, 25]))
        delta_min = self.pair_margin_rho * max(iqr, 1e-6)

        dpo_pairs = []

        # winner-loser pairs
        for j in range(len(solutions)):
            for jj in range(j + 1, len(solutions)):
                gap = utilities[j] - utilities[jj]
                if gap <= delta_min:
                    continue

                winner_prior = self._extract_prior(solutions[j], env_sample)
                loser_prior = self._extract_prior(solutions[jj], env_sample)

                chosen = format_oracle_response(
                    sample_id, *winner_prior
                )
                rejected = format_oracle_response(
                    sample_id, *loser_prior
                )

                dpo_pairs.append({
                    "id": f"env_{sample_id}_pair_{j}_{jj}",
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "utility_chosen": float(utilities[j]),
                    "utility_rejected": float(utilities[jj]),
                    "utility_gap": float(gap),
                    # Oracle targets (from winner) for control loss
                    "q_current": env_sample.q_current.tolist(),
                    "delta_q": winner_prior[0].tolist(),
                    "delta_a": winner_prior[1].tolist(),
                    "delta_p": winner_prior[2].tolist(),
                })

        return dpo_pairs

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
        return (np.round(delta_q, 4).astype(np.float32),
                np.round(delta_a, 4).astype(np.float32),
                np.round(delta_p, 4).astype(np.float32))

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
