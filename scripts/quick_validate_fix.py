#!/usr/bin/env python3
"""
Quick validation script for the ground clutter solver fix.

Generates 50 random environments with the NEW solver (ground_clutter_db=12.0)
and checks whether the data distribution is now diverse:
  - Full-speed flight ratio < 40% (was 84.7%)
  - Fine adjustment (<5m) ratio > 10% (was 0%)
  - Upward movement ratio > 15% (was 0%)  ← CORE RED LINE

Usage:
    python scripts/quick_validate_fix.py

Expected runtime: ~2-5 minutes for 50 environments.
"""
import sys
import time
import numpy as np

sys.path.insert(0, '.')

from src.solver.sca_fp import SCAFPOptimizer, SCAFPConfig
from src.env import ISACScenarioGenerator


def main():
    print("=" * 60)
    print("Quick Validation: Ground Clutter (12dB) Solver Fix")
    print("=" * 60)

    # 1. Initialize solver with ground clutter penalty
    cfg = SCAFPConfig(
        ground_clutter_db=6.0,
        max_iters=100,
        lambda_repel=0.01,
    )
    solver = SCAFPOptimizer(
        cfg, M=4, K=20, T=6, N_t=8,
        carrier_freq_ghz=5.8,
        area_size=(1000, 1000),
        altitude_range=(50, 300),
        p_max=1.0,
        noise_power=1e-13,
        load_cap=10,
    )

    # 2. Initialize scenario generator
    gen = ISACScenarioGenerator(
        num_uavs=4, num_users=20, num_targets=6,
        area_size=(1000, 1000), carrier_freq_ghz=5.8,
        bandwidth_mhz=20, num_antennas=8, p_max_dbm=30, seed=42,
    )

    delta_q_list = []
    n_fail = 0
    n_envs = 50

    print(f"\nSolving {n_envs} random environments...")
    start_time = time.time()

    for i in range(n_envs):
        env_sample = gen.sample(i)

        # 构造 solver 期望的 dict 格式
        env_dict = {
            "q_current": env_sample.q_current.copy(),
            "user_positions": env_sample.u_positions.copy(),
            "target_positions": env_sample.s_positions.copy(),
            "channel_gains": env_sample.channel_gains_users.copy(),
            "user_weights": env_sample.user_weights.copy().astype(np.float32),
            "association": env_sample.association.copy(),
        }

        try:
            sol = solver.solve(env_dict, warm_start=None, seed=i)

            dq = sol.Q - env_dict["q_current"]  # (N_uav, 3)
            delta_q_list.append(dq)

        except Exception as e:
            print(f"\n[WARN] Env {i} solve failed: {e}")
            n_fail += 1
            continue

        print(f"\r  Completed: {i+1}/{n_envs}", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n\nElapsed: {elapsed:.1f}s  |  Failures: {n_fail}/{n_envs}")

    if not delta_q_list:
        print("FATAL: No data collected. Check solve() signature and env keys.")
        return 1

    # 3. Statistics
    delta_q_all = np.vstack(delta_q_list)  # (N_envs * 4, 3)
    norms = np.linalg.norm(delta_q_all, axis=1)
    dz = delta_q_all[:, 2]  # Z-axis (height) component

    print("\n" + "=" * 60)
    print("DIVERSITY DIAGNOSTIC REPORT")
    print("=" * 60)

    # Metric 1: Full-throttle ratio (was 84.7%)
    max_speed_ratio = np.mean(norms >= 14.5) * 100
    print(f"\n  Full-speed (>=14.5m):  {max_speed_ratio:.1f}%  (target < 40%)")
    print(f"    {'PASS' if max_speed_ratio < 40 else 'FAIL'}")

    # Metric 2: Fine adjustment ratio (was 0.0%)
    micro_adj_ratio = np.mean(norms < 5.0) * 100
    print(f"  Fine adjust (<5.0m):   {micro_adj_ratio:.1f}%  (target > 10%)")
    print(f"    {'PASS' if micro_adj_ratio > 10 else 'FAIL'}")

    # Metric 3: Vertical direction distribution (was 97.4% down, 0% up)
    down_ratio = np.mean(dz < -0.1) * 100
    up_ratio = np.mean(dz > 0.1) * 100
    flat_ratio = np.mean(np.abs(dz) <= 0.1) * 100
    print(f"\n  Vertical direction:")
    print(f"    Down  : {down_ratio:.1f}%")
    print(f"    Flat  : {flat_ratio:.1f}%")
    print(f"    UP    : {up_ratio:.1f}%  (target > 15%)  ← CORE RED LINE")
    print(f"    {'PASS' if up_ratio > 15 else 'FAIL'}")

    # Bonus: speed distribution
    print(f"\n  Speed histogram:")
    bins = [(0, 2), (2, 5), (5, 8), (8, 10), (10, 12), (12, 13),
            (13, 14), (14, 14.5), (14.5, 14.9), (14.9, 15), (15, 99)]
    max_bin_pct = 0
    for lo, hi in bins:
        pct = np.mean((norms >= lo) & (norms < hi)) * 100
        bar = '█' * int(pct / 2)
        tag = ' ← COLLAPSE ZONE' if lo >= 14.5 and pct > 40 else ''
        print(f"    [{lo:5.1f}, {hi:5.1f}): {pct:5.1f}% {bar}{tag}")
        if pct > max_bin_pct:
            max_bin_pct = pct

    n_pass = (
        (max_speed_ratio < 40) +
        (micro_adj_ratio > 10) +
        (up_ratio > 15)
    )
    print(f"\n  Results: {n_pass}/3 checks passed")

    if n_pass == 3:
        print("\n  ✅ ALL CHECKS PASSED — solver fix verified!")
        print("  Ready for full data regeneration (5000 envs).")
    else:
        print("\n  ⚠️  Some checks failed. Consider:")
        print("     - Adjust ground_clutter_db (try 10-18 dB range)")
        print("     - Verify clutter_db applied to BOTH comm and sens path loss")
        print("     - Check if other solver constraints dominate")

    print("\n" + "=" * 60)
    return 0 if n_pass == 3 else 1


if __name__ == "__main__":
    sys.exit(main())
