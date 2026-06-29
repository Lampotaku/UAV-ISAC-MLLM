#!/usr/bin/env python3
"""
ε-Calibration: 微扰步长标定脚本

对 50 个随机环境测试 ε ∈ {0.5, 1.0, 2.0, 4.0, 8.0}m，
选择使 snap-back 回弹步数区分度（方差）最大的 ε。

原理:
  - ε 太小 → 所有候选 1-2 步滑回谷底（无区分度）
  - ε 太大 → 跳出原 basin 进入未知惩罚区（无区分度）
  - 要找的是"盆地边缘"的特征尺度

用法:
    python scripts/calibrate_epsilon.py [--num-envs 50] [--num-restarts 10]

预期运行时间: ~5 分钟 (50 envs × 10 restarts × 5 epsilons × ~1s/solve)
"""

import sys
import os
import time
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.solver.sca_fp import SCAFPOptimizer, SCAFPConfig
from src.env import ISACScenarioGenerator


def _env_to_dict(env_sample, solver):
    """将 EnvironmentSample 转为 solver 期望的 dict 格式"""
    return {
        "q_current": env_sample.q_current.copy(),
        "user_positions": env_sample.u_positions.copy(),
        "target_positions": env_sample.s_positions.copy(),
        "channel_gains": env_sample.channel_gains_users.copy(),
        "user_weights": env_sample.user_weights.copy().astype(np.float32),
        "association": env_sample.association.copy(),
    }


def _compute_utility_from_solution(solver, sol, env_dict):
    """重新计算解的效用（使用 solver 内部的 _compute_utility）"""
    return solver._compute_utility(
        sol.Q, sol.A, sol.W_c_power, sol.W_s_power,
        env_dict["channel_gains"], env_dict["target_positions"],
        env_dict["user_weights"],
    )


def _run_best_of_n(solver, env_dict, n_restarts, base_seed):
    """N 次随机重启 SCA-FP，返回按 utility 排序的解列表"""
    solutions = []
    for j in range(n_restarts):
        seed = base_seed * n_restarts + j
        sol = solver.solve(env_dict, warm_start=None, seed=seed)
        solutions.append(sol)
    solutions.sort(key=lambda s: s.utility, reverse=True)
    return solutions


def _compute_baseline_utility(solver, env_dict):
    """计算 [0,0,0] 不动方案的 utility"""
    q_cur = env_dict["q_current"]
    zero_warm = {
        "delta_q": np.zeros_like(q_cur),
        "delta_a": np.zeros((solver.M, solver.K)),
        "delta_p": np.zeros((solver.M, solver.K + 1)),
    }
    zero_sol = solver.solve(env_dict, warm_start=zero_warm, seed=999999)
    return zero_sol.utility


def _pareto_filter(solutions, baseline_utility, utility_ratio=0.95):
    """
    Pareto 过滤:
      1. 丢弃 utility < baseline 的解
      2. 丢弃低于全局最高 utility × utility_ratio 的劣质坑
    返回通过过滤的解列表
    """
    if not solutions:
        return []

    max_utility = solutions[0].utility
    threshold = max_utility * utility_ratio

    filtered = [
        s for s in solutions
        if s.utility > baseline_utility and s.utility >= threshold
    ]
    return filtered


def _snapback_test(solver, env_dict, candidate_solution, epsilon, seed_offset):
    """
    微扰回弹测试:
      1. 对候选解的 Q 施加随机方向、固定幅度 ε 的扰动
      2. 以扰动点为初始值重跑 SCA-FP
      3. 返回收敛所需迭代步数
    """
    M = solver.M
    q_opt = candidate_solution.Q.copy()

    # 生成随机 3D 方向 (每个 UAV 独立)
    rng = np.random.RandomState(seed_offset)
    perturbed_q = q_opt.copy()
    for m in range(M):
        # 单位球面均匀采样
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
    perturbed_q[:, 0] = np.clip(perturbed_q[:, 0], 0, solver.area_w)
    perturbed_q[:, 1] = np.clip(perturbed_q[:, 1], 0, solver.area_h)
    perturbed_q[:, 2] = np.clip(perturbed_q[:, 2], solver.H_min, solver.H_max)

    # 构造 warm_start: δ_q = perturbed_q - q_current
    q_current = env_dict["q_current"]
    delta_q_perturbed = perturbed_q - q_current

    warm_start = {
        "delta_q": delta_q_perturbed,
        "delta_a": candidate_solution.A.copy(),
        "delta_p": np.concatenate([
            candidate_solution.W_c_power,
            candidate_solution.W_s_power.reshape(-1, 1),
        ], axis=1),
    }

    # 重跑 SCA-FP (max_iters=100 安全帽)
    rerun_sol = solver.solve(env_dict, warm_start=warm_start, seed=seed_offset + 10000)
    return rerun_sol.iterations


def main():
    parser = argparse.ArgumentParser(description="ε-Calibration: snap-back perturbation sweep")
    parser.add_argument("--num-envs", type=int, default=50,
                        help="Number of random environments (default: 50)")
    parser.add_argument("--num-restarts", type=int, default=10,
                        help="SCA-FP restarts per environment (default: 10)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-K candidates for snap-back test (default: 3)")
    args = parser.parse_args()

    epsilons = [0.5, 1.0, 2.0, 4.0, 8.0]
    n_envs = args.num_envs
    n_restarts = args.num_restarts
    top_k = args.top_k

    print("=" * 60)
    print("ε-Calibration: Snap-back Perturbation Sweep")
    print("=" * 60)
    print(f"  Environments:  {n_envs}")
    print(f"  Restarts/env:  {n_restarts}")
    print(f"  Top-K:         {top_k}")
    print(f"  Epsilons (m):  {epsilons}")
    print()

    # ── 初始化 solver（ground_clutter_db=12.0）──
    solver_cfg = SCAFPConfig(
        ground_clutter_db=12.0,
        max_iters=100,
        max_outer_iters=30,
        max_inner_iters=50,
        lambda_repel=0.01,
    )
    solver = SCAFPOptimizer(
        solver_cfg, M=4, K=20, T=6, N_t=8,
        carrier_freq_ghz=5.8,
        area_size=(1000, 1000),
        altitude_range=(50, 300),
        p_max=1.0,
        noise_power=1e-13,
        load_cap=10,
    )

    # ── 初始化场景生成器 ──
    scenario_gen = ISACScenarioGenerator(
        num_uavs=4, num_users=20, num_targets=6,
        area_size=(1000, 1000), carrier_freq_ghz=5.8,
        bandwidth_mhz=20, num_antennas=8, p_max_dbm=30, seed=42,
    )

    # ── 收集 snap-back 迭代数据 ──
    # iterations_by_epsilon[ε_idx] = list of iteration values (flat across all envs)
    iterations_by_epsilon = {eps: [] for eps in epsilons}
    n_skipped = 0
    n_valid = 0

    print(f"Running Best-of-N + Snap-back on {n_envs} environments...\n")
    t_start = time.time()

    for env_idx in range(n_envs):
        env_sample = scenario_gen.sample(env_idx)
        env_dict = _env_to_dict(env_sample, solver)

        # Step 1: Best-of-N
        solutions = _run_best_of_n(solver, env_dict, n_restarts, env_idx)

        # Step 2: Pareto filter
        baseline_util = _compute_baseline_utility(solver, env_dict)
        candidates = _pareto_filter(solutions, baseline_util, utility_ratio=0.95)
        if len(candidates) < 2:
            n_skipped += 1
            continue  # 不足 2 个候选无法计算方差

        n_valid += 1
        # Take top-K
        candidates = candidates[:top_k]

        # Step 3: Snap-back test for each epsilon
        for eps in epsilons:
            for cand_idx, cand in enumerate(candidates):
                seed_offset = env_idx * 1000 + cand_idx * 10
                try:
                    iters = _snapback_test(
                        solver, env_dict, cand, eps, seed_offset,
                    )
                    iterations_by_epsilon[eps].append(iters)
                except Exception as e:
                    # 微扰点可能崩溃 → 记录 max_iters (100)
                    iterations_by_epsilon[eps].append(100)

        if (env_idx + 1) % 10 == 0 or env_idx == 0:
            elapsed = time.time() - t_start
            rate = elapsed / (env_idx + 1)
            eta = rate * (n_envs - env_idx - 1)
            print(f"  [{env_idx+1}/{n_envs}] {n_valid} valid, {n_skipped} skipped | "
                  f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s. {n_valid} valid envs, {n_skipped} skipped.\n")

    # ── 分析结果 ──
    print("=" * 60)
    print("ε-CALIBRATION REPORT")
    print("=" * 60)
    print(f"{'ε (m)':<10} {'Mean Iters':<12} {'Std Iters':<12} {'Variance':<12} {'CV':<10} {'Recommend':<12}")
    print("-" * 70)

    best_eps = None
    best_variance = -1

    for eps in epsilons:
        vals = iterations_by_epsilon[eps]
        if len(vals) < 5:
            continue
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        var_v = np.var(vals)
        cv_v = std_v / mean_v if mean_v > 0 else 0

        if var_v > best_variance:
            best_variance = var_v
            best_eps = eps

        marker = " ★ BEST" if eps == best_eps else ""
        print(f"{eps:<10.1f} {mean_v:<12.1f} {std_v:<12.1f} {var_v:<12.1f} {cv_v:<10.3f}{marker}")

    print("-" * 70)
    print()

    # ── 诊断：检查区分度 ──
    if n_valid < 5:
        print("⚠️  WARNING: < 5 valid environments — sweep is unreliable.")
        print("   Consider increasing --num-envs or checking solver correctness.")
    elif best_variance < 1.0:
        print("⚠️  WARNING: Best variance < 1.0 — all epsilons give similar iteration counts.")
        print("   Possible causes:")
        print("   1. Ground clutter effect too weak (try increasing ground_clutter_db)")
        print("   2. Solver converges too quickly from all perturbed starts")
        print("   3. Pareto filter too aggressive — all candidates identical")
    else:
        print(f"✅ Recommended ε = {best_eps} m  (variance = {best_variance:.1f})")
        print(f"   This epsilon produces the clearest distinction between wide and narrow basins.")
        print(f"   Use this value in generate_data.py: --snapback-epsilon {best_eps}")

    print()
    print("Next step: python scripts/quick_validate_fix.py  # verify solver fix")
    print("Then:      python scripts/generate_data.py --num-envs 20000 ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
