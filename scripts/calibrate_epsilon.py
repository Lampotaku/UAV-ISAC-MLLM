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
    python scripts/calibrate_epsilon.py [--num-envs 50] [--num-restarts 10] [--workers 16]

预期运行时间: ~5 分钟 (单进程) / ~30 秒 (32 workers)
"""

import sys
import os

# ⚠️ 必须在 import numpy 之前！
# 防止 Intel MKL / OpenBLAS 与 Python multiprocessing 线程打架
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.solver.sca_fp import SCAFPOptimizer, SCAFPConfig
from src.env import ISACScenarioGenerator


# ── 全局配置 (模块级, 供 worker 函数读取) ──
_WORKER_SOLVER_CFG = None
_WORKER_SCENARIO_SEED = None
_WORKER_N_RESTARTS = None
_WORKER_TOP_K = None
_WORKER_EPSILONS = None


def _init_worker():
    """每个 worker 进程的初始化: 创建自己的 solver + scenario_gen 实例"""
    global _worker_solver, _worker_scenario_gen

    cfg = _WORKER_SOLVER_CFG
    _worker_solver = SCAFPOptimizer(
        cfg, M=4, K=20, T=6, N_t=8,
        carrier_freq_ghz=5.8,
        area_size=(1000, 1000),
        altitude_range=(50, 300),
        p_max=1.0,
        noise_power=1e-13,
        load_cap=10,
    )

    _worker_scenario_gen = ISACScenarioGenerator(
        num_uavs=4, num_users=20, num_targets=6,
        area_size=(1000, 1000), carrier_freq_ghz=5.8,
        bandwidth_mhz=20, num_antennas=8, p_max_dbm=30,
        seed=_WORKER_SCENARIO_SEED,
    )


_worker_solver = None
_worker_scenario_gen = None


def _env_to_dict(env_sample):
    """将 EnvironmentSample 转为 solver 期望的 dict 格式"""
    return {
        "q_current": env_sample.q_current.copy(),
        "user_positions": env_sample.u_positions.copy(),
        "target_positions": env_sample.s_positions.copy(),
        "channel_gains": env_sample.channel_gains_users.copy(),
        "user_weights": env_sample.user_weights.copy().astype(np.float32),
        "association": env_sample.association.copy(),
    }


def _run_best_of_n(env_dict, base_seed):
    """N 次随机重启 SCA-FP，返回按 utility 排序的解列表"""
    solutions = []
    for j in range(_WORKER_N_RESTARTS):
        seed = base_seed * _WORKER_N_RESTARTS + j
        sol = _worker_solver.solve(env_dict, warm_start=None, seed=seed)
        solutions.append(sol)
    solutions.sort(key=lambda s: s.utility, reverse=True)
    return solutions


def _compute_baseline_utility(env_dict):
    """计算 [0,0,0] 不动方案的 utility"""
    q_cur = env_dict["q_current"]
    zero_warm = {
        "delta_q": np.zeros_like(q_cur),
        "delta_a": np.zeros((_worker_solver.M, _worker_solver.K)),
        "delta_p": np.zeros((_worker_solver.M, _worker_solver.K + 1)),
    }
    zero_sol = _worker_solver.solve(env_dict, warm_start=zero_warm, seed=999999)
    return zero_sol.utility


def _pareto_filter(solutions, baseline_utility, utility_ratio=0.85):
    """Pareto 过滤: 丢弃 utility < baseline 或低于全局最高 × utility_ratio 的解"""
    if not solutions:
        return []
    max_utility = solutions[0].utility
    threshold = max_utility * utility_ratio
    return [s for s in solutions if s.utility > baseline_utility and s.utility >= threshold]


def _snapback_test(env_dict, candidate_solution, epsilon, seed_offset):
    """
    微扰回弹测试:
      1. 对候选解的 Q 施加随机方向、固定幅度 ε 的扰动
      2. 以扰动点为初始值重跑 SCA-FP
      3. 返回收敛所需迭代步数
    """
    M = _worker_solver.M
    q_opt = candidate_solution.Q.copy()

    rng = np.random.RandomState(seed_offset)
    perturbed_q = q_opt.copy()
    for m in range(M):
        # 单位球面均匀采样 (cos_θ 均匀 → 面积微元均匀 → 无极点聚集)
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
    perturbed_q[:, 0] = np.clip(perturbed_q[:, 0], 0, _worker_solver.area_w)
    perturbed_q[:, 1] = np.clip(perturbed_q[:, 1], 0, _worker_solver.area_h)
    perturbed_q[:, 2] = np.clip(perturbed_q[:, 2], _worker_solver.H_min, _worker_solver.H_max)

    # 构造 warm_start
    q_current = env_dict["q_current"]
    delta_q_perturbed = perturbed_q - q_current
    # 不传 perfect A/P — 否则 SCA-FP 的 Q 子问题退化为凸优化, 2 步内收敛
    # 传零初始化 → 迫使求解器在联合空间重新寻优 → 真正测试盆地宽度
    warm_start = {
        "delta_q": delta_q_perturbed,
        "delta_a": np.zeros((_worker_solver.M, _worker_solver.K)),
        "delta_p": np.zeros((_worker_solver.M, _worker_solver.K + 1)),
    }

    rerun_sol = _worker_solver.solve(env_dict, warm_start=warm_start, seed=seed_offset + 10000)
    return rerun_sol.iterations


def _process_one_env(env_idx):
    """
    处理单个环境 (worker 进程入口).
    返回 (env_idx, valid, iterations_dict, skipped_reason)
      - valid=True  → iterations_dict = {eps: [iter1, iter2, ...]}
      - valid=False → iterations_dict 为空, skipped_reason 说明原因
    """
    env_sample = _worker_scenario_gen.sample(env_idx)
    env_dict = _env_to_dict(env_sample)

    # Step 1: Best-of-N
    solutions = _run_best_of_n(env_dict, env_idx)

    # Step 2: Pareto filter
    baseline_util = _compute_baseline_utility(env_dict)
    candidates = _pareto_filter(solutions, baseline_util, utility_ratio=0.95)
    if len(candidates) < 2:
        return (env_idx, False, {}, f"only {len(candidates)} candidate(s) after Pareto filter")

    candidates = candidates[:_WORKER_TOP_K]

    # Step 3: Snap-back test for each epsilon
    iterations_by_eps = {}
    for eps in _WORKER_EPSILONS:
        for cand_idx, cand in enumerate(candidates):
            seed_offset = env_idx * 1000 + cand_idx * 10
            try:
                iters = _snapback_test(env_dict, cand, eps, seed_offset)
            except Exception:
                iters = 100  # 微扰点崩溃 → 记录 max_iters
            iterations_by_eps.setdefault(eps, []).append(iters)

    return (env_idx, True, iterations_by_eps, "")


def main():
    global _WORKER_SOLVER_CFG, _WORKER_SCENARIO_SEED, _WORKER_N_RESTARTS, _WORKER_TOP_K, _WORKER_EPSILONS

    parser = argparse.ArgumentParser(description="ε-Calibration: snap-back perturbation sweep")
    parser.add_argument("--num-envs", type=int, default=50,
                        help="Number of random environments (default: 50)")
    parser.add_argument("--num-restarts", type=int, default=10,
                        help="SCA-FP restarts per environment (default: 10)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-K candidates for snap-back test (default: 3)")
    parser.add_argument("--workers", type=int, default=-1,
                        help="Worker processes: -1=auto (cpu-2), 0=sequential, N=explicit (default: -1)")
    args = parser.parse_args()

    epsilons = [0.5, 1.0, 2.0, 4.0, 8.0]
    n_envs = args.num_envs
    n_restarts = args.num_restarts
    top_k = args.top_k

    if args.workers == -1:
        n_workers = max(1, os.cpu_count() - 2) if os.cpu_count() else 1
    elif args.workers == 0:
        n_workers = 0
    else:
        n_workers = args.workers

    print("=" * 60)
    print("ε-Calibration: Snap-back Perturbation Sweep")
    print("=" * 60)
    print(f"  Environments:  {n_envs}")
    print(f"  Restarts/env:  {n_restarts}")
    print(f"  Top-K:         {top_k}")
    print(f"  Epsilons (m):  {epsilons}")
    if n_workers > 0:
        print(f"  Workers:       {n_workers} (multiprocessing)")
    else:
        print(f"  Mode:          sequential")
    print()

    # ── 初始化 solver 配置 (用于 worker 进程) ──
    _WORKER_SOLVER_CFG = SCAFPConfig(
        ground_clutter_db=6.0,
        max_iters=100,
        max_outer_iters=30,
        max_inner_iters=50,
        lambda_repel=0.01,
    )
    _WORKER_SCENARIO_SEED = 42
    _WORKER_N_RESTARTS = n_restarts
    _WORKER_TOP_K = top_k
    _WORKER_EPSILONS = epsilons

    print(f"Running Best-of-N + Snap-back on {n_envs} environments...\n")
    t_start = time.time()

    # ── 收集结果 ──
    iterations_by_epsilon = {eps: [] for eps in epsilons}
    n_skipped = 0
    n_valid = 0

    if n_workers > 0:
        # ── 多进程模式 ──
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
        ) as executor:
            future_to_idx = {
                executor.submit(_process_one_env, i): i
                for i in range(n_envs)
            }

            for future in as_completed(future_to_idx):
                env_idx, is_valid, iters_dict, reason = future.result()
                if is_valid:
                    n_valid += 1
                    for eps, vals in iters_dict.items():
                        iterations_by_epsilon[eps].extend(vals)
                else:
                    n_skipped += 1

                done = n_valid + n_skipped
                if done % 10 == 0 or done == 1:
                    elapsed = time.time() - t_start
                    rate = elapsed / done
                    eta = rate * (n_envs - done)
                    print(f"  [{done}/{n_envs}] {n_valid} valid, {n_skipped} skipped | "
                          f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
    else:
        # ── 顺序模式 ──
        _init_worker()
        for env_idx in range(n_envs):
            _, is_valid, iters_dict, _ = _process_one_env(env_idx)
            if is_valid:
                n_valid += 1
                for eps, vals in iters_dict.items():
                    iterations_by_epsilon[eps].extend(vals)
            else:
                n_skipped += 1

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
    print(f"{'ε (m)':<10} {'Mean Iters':<12} {'Std Iters':<12} {'Variance':<12} {'CV':<10} {'Samples':<10} {'Recommend':<12}")
    print("-" * 80)

    best_eps = None
    best_variance = -1

    for eps in epsilons:
        vals = iterations_by_epsilon[eps]
        if len(vals) < 5:
            print(f"{eps:<10.1f} {'—':<12} {'—':<12} {'—':<12} {'—':<10} {len(vals):<10}  (too few samples)")
            continue
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        var_v = np.var(vals)
        cv_v = std_v / mean_v if mean_v > 0 else 0

        if var_v > best_variance:
            best_variance = var_v
            best_eps = eps

        marker = " ★ BEST" if eps == best_eps else ""
        print(f"{eps:<10.1f} {mean_v:<12.1f} {std_v:<12.1f} {var_v:<12.1f} {cv_v:<10.3f} {len(vals):<10}{marker}")

    print("-" * 80)
    print()

    # ── 诊断 ──
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
        print(f"   Use in generate_data.py: --snapback-epsilon {best_eps}")

    print()
    print("Next: python scripts/quick_validate_fix.py")
    print("Then: python scripts/generate_data.py --num-envs 20000 --workers 70 --snapback-epsilon <EPSILON>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
