#!/usr/bin/env python
"""
UAV-ISAC-MLLM 训练数据生成入口
论文 Algorithm 1 — Best-of-N (S=5000, N=10)

支持断点续跑: Ctrl+C 或中途崩溃后, 重新运行相同命令即可从上次中断处继续

用法:
  python scripts/generate_data.py --num-env 5000 --num-restarts 10

输出:
  - {output_dir}/sft_dataset.jsonl  (增量追加)
  - {output_dir}/dpo_dataset.jsonl  (增量追加)
  - {output_dir}/checkpoint.txt     (当前进度)
"""
import sys
import os
import argparse
import yaml
import time
import json
import signal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.env import ISACScenarioGenerator
from src.solver.sca_fp import SCAFPOptimizer, SCAFPConfig
from src.data.oracle_generator import OracleDataGenerator


# 全局变量用于优雅中断
_stop_requested = False


def _on_interrupt(sig, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[INTERRUPT] Stopping after current environment... (Ctrl+C again to force quit)")


def _incremental_append(filepath, record):
    """追加单条 JSONL 记录"""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _count_existing(filepath):
    """统计已有行数"""
    if not os.path.exists(filepath):
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate UAV-ISAC training data (resumable)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num-env", type=int, default=None)
    parser.add_argument("--num-restarts", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N environments")
    args = parser.parse_args()

    # 加载配置
    with open(os.path.join(PROJECT_ROOT, args.config), "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["simulation"]
    data_cfg = cfg["data"]

    if args.num_env is not None:
        data_cfg["num_environments"] = args.num_env
    if args.num_restarts is not None:
        data_cfg["num_restarts"] = args.num_restarts
    if args.output_dir is not None:
        data_cfg["output_dir"] = args.output_dir

    num_envs = data_cfg["num_environments"]
    num_restarts = data_cfg["num_restarts"]
    output_dir = data_cfg["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    sft_path = os.path.join(output_dir, "sft_dataset.jsonl")
    dpo_path = os.path.join(output_dir, "dpo_dataset.jsonl")
    ckpt_path = os.path.join(output_dir, "checkpoint.txt")

    # ---- 断点续跑 ----
    existing_sft = _count_existing(sft_path)
    start_env = existing_sft  # SFT 每环境一条, 所以已有条数=已完成环境数
    if start_env > 0:
        print(f"[RESUME] Found {existing_sft} existing SFT samples, resuming from env {start_env}")
    if start_env >= num_envs:
        print(f"All {num_envs} environments already done! Exiting.")
        return

    # ---- 初始化 ----
    print("=" * 60)
    print("UAV-ISAC-MLLM: Best-of-N Oracle Data Generator")
    print("=" * 60)
    print(f"  Environments:  S = {num_envs}  ({start_env} already done)")
    print(f"  Restarts/env:  N = {num_restarts}")
    print(f"  Output:        {output_dir}")
    print(f"  Save every:    {args.save_every} envs")
    print()

    print("Initializing components...")
    scenario_gen = ISACScenarioGenerator(
        num_uavs=sim_cfg["num_uavs"],
        num_users=sim_cfg["num_users"],
        num_targets=sim_cfg["num_targets"],
        area_size=tuple(sim_cfg["area_size"]),
        carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
        bandwidth_mhz=sim_cfg["bandwidth_mhz"],
        num_antennas=sim_cfg["num_antennas_tx"],
        p_max_dbm=sim_cfg["p_max_dbm"],
        seed=args.seed,
    )

    solver_config = SCAFPConfig(
        max_outer_iters=30,
        max_inner_iters=50,
        tol=1e-4,
        lambda_sensing=0.5,
        lambda_idle_penalty=5.0,
        sinr_c_min=10 ** (sim_cfg["sinr_c_min_db"] / 10),
        sinr_s_min=10 ** (sim_cfg["sinr_s_min_db"] / 10),
        verbose=False,
    )

    solver = SCAFPOptimizer(
        config=solver_config,
        M=sim_cfg["num_uavs"],
        K=sim_cfg["num_users"],
        T=sim_cfg["num_targets"],
        N_t=sim_cfg["num_antennas_tx"],
        N_r=sim_cfg.get("num_antennas_rx", sim_cfg["num_antennas_tx"]),
        carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
        area_size=tuple(sim_cfg["area_size"]),
        altitude_range=(sim_cfg["altitude_min_m"], sim_cfg["altitude_max_m"]),
        p_max=10 ** ((sim_cfg["p_max_dbm"] - 30) / 10),
        noise_power=10 ** ((-174 + 10 * np.log10(sim_cfg["bandwidth_mhz"] * 1e6) + sim_cfg["noise_figure_db"] - 30) / 10),
        load_cap=sim_cfg["load_cap_per_uav"],
    )

    generator = OracleDataGenerator(
        scenario_gen=scenario_gen,
        solver=solver,
        config={**data_cfg, "output_dir": output_dir},
        sim_config=sim_cfg,
    )

    # ---- 运行主循环 (增量保存) ----
    signal.signal(signal.SIGINT, _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    print(f"\nGenerating envs {start_env}..{num_envs-1}...\n")

    t_start = time.time()
    n_sft, n_dpo = start_env, _count_existing(dpo_path)

    for i in range(start_env, num_envs):
        if _stop_requested:
            print(f"\nStopped at env {i}. {n_sft} SFT, {n_dpo} DPO saved.")
            print(f"Resume with the same command.")
            break

        try:
            sft_sample, dpo_samples = generator._process_one_environment(i)
            if sft_sample is not None:
                _incremental_append(sft_path, sft_sample)
                n_sft += 1
                for d in dpo_samples:
                    _incremental_append(dpo_path, d)
                    n_dpo += 1
        except Exception as e:
            print(f"\n[ERROR] env {i}: {e}")
            continue

        # 进度输出 + 定期 checkpoint
        if (i - start_env + 1) % args.save_every == 0 or i == start_env:
            elapsed = time.time() - t_start
            done = i - start_env + 1
            rate = elapsed / done
            remaining = (num_envs - i - 1) * rate
            print(f"  [{i+1}/{num_envs}] {n_sft} SFT, {n_dpo} DPO | "
                  f"{elapsed:.0f}s elapsed, ~{remaining/3600:.1f}h remaining | "
                  f"{rate:.1f}s/env", flush=True)
            # 写 checkpoint
            with open(ckpt_path, "w") as f:
                f.write(f"{i+1}\n")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"  SFT: {n_sft}  |  DPO: {n_dpo}")
    print(f"  Files: {sft_path}, {dpo_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
