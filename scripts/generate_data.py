#!/usr/bin/env python
"""
UAV-ISAC-MLLM 训练数据生成入口
论文 Algorithm 1 — Best-of-N (S=5000, N=10)

支持断点续跑: Ctrl+C 或中途崩溃后, 重新运行相同命令即可从上次中断处继续
支持多进程并发 (--workers): 按 save_every 分批, 每批内并行执行

用法:
  # 单进程 (兼容原有行为)
  python scripts/generate_data.py --num-env 2000 --num-restarts 10

  # 多进程 (Intel Xeon / EPYC 加速)
  python scripts/generate_data.py --num-env 2000 --num-restarts 10 --workers 70

输出:
  - {output_dir}/sft_dataset.jsonl  (增量追加)
  - {output_dir}/dpo_dataset.jsonl  (增量追加)
  - {output_dir}/checkpoint.txt     (当前进度)
"""
import sys
import os

# ⚠️ 必须在 import numpy 之前！
# 防止 Intel MKL / OpenBLAS 与 Python multiprocessing 线程打架
# 每个 MKL worker 都试图开满全部核心 → CPU 100% 但进度卡死
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import yaml
import time
import json
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    print("\n[INTERRUPT] Stopping after current batch... (Ctrl+C again to force quit)")


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


def _read_checkpoint(ckpt_path):
    """读取 checkpoint 文件, 返回已完成的环境数 (批次边界)"""
    if not os.path.exists(ckpt_path):
        return 0
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        return 0


def _atomic_merge_batch(tmp_path, main_path):
    """将临时批次文件追加到主文件 (容错: 若主文件不存在则创建)"""
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        return
    with open(tmp_path, "r", encoding="utf-8") as src:
        with open(main_path, "a", encoding="utf-8") as dst:
            for line in src:
                if line.strip():
                    dst.write(line)
    os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Generate UAV-ISAC training data (resumable)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num-env", type=int, default=None)
    parser.add_argument("--num-restarts", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N environments (also batch size in multiprocessing)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Worker processes: 0=sequential, -1=auto (cpu-2), N=explicit")
    parser.add_argument("--snapback-epsilon", type=float, default=1.5,
                        help="Perturbation magnitude for snap-back test (meters, default: 1.5)")
    parser.add_argument("--snapback-top-k", type=int, default=3,
                        help="Top-K candidates for snap-back test (default: 3)")
    parser.add_argument("--pareto-utility-ratio", type=float, default=0.85,
                        help="Discard solutions below this ratio of max utility (default: 0.85)")
    parser.add_argument("--heuristic-reject-ratio", type=float, default=0.3,
                        help="Fraction of heuristic trap rejected samples (default: 0.3)")
    args = parser.parse_args()

    # ── 确定并发数 ──
    if args.workers == -1:
        n_workers = max(1, os.cpu_count() - 2)
    elif args.workers > 1:
        n_workers = args.workers
    else:
        n_workers = 0  # 顺序模式

    # 加载配置
    with open(os.path.join(PROJECT_ROOT, args.config), "r", encoding="utf-8") as f:
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
    # 多进程模式: 用 checkpoint 文件 (批次边界, 安全)
    # 顺序模式:   用 SFT 行数 (逐环境顺序写入, 可靠)
    if n_workers > 0:
        start_env = _read_checkpoint(ckpt_path)
    else:
        start_env = _count_existing(sft_path)
    if start_env > 0:
        print(f"[RESUME] Resuming from env {start_env}")
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
    if n_workers > 0:
        print(f"  Workers:       {n_workers} (multiprocessing, batch={args.save_every})")
    else:
        print(f"  Mode:          sequential")
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
        max_iters=100,                     # 安全帽: snap-back 重跑最多 100 步
        max_outer_iters=30,
        max_inner_iters=50,
        tol=1e-4,
        lambda_sensing=0.5,
        lambda_idle_penalty=5.0,
        sinr_c_min=10 ** (sim_cfg["sinr_c_min_db"] / 10),
        sinr_s_min=10 ** (sim_cfg["sinr_s_min_db"] / 10),
        ground_clutter_db=12.0,            # ★ 地面杂波 — 数据多样性核心
        lambda_repel=0.01,                 # ★ 空间互斥力 — 防止 UAV 扎堆
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
        v_max=sim_cfg.get("uav_max_speed_ms", 15),
        slot_duration=sim_cfg.get("slot_duration_s", 1.0),
    )

    generator = OracleDataGenerator(
        scenario_gen=scenario_gen,
        solver=solver,
        config={
            **data_cfg,
            "output_dir": output_dir,
            "snapback_epsilon": args.snapback_epsilon,
            "snapback_top_k": args.snapback_top_k,
            "pareto_utility_ratio": args.pareto_utility_ratio,
            "heuristic_reject_ratio": args.heuristic_reject_ratio,
        },
        sim_config=sim_cfg,
    )

    # ---- 运行主循环 ----
    signal.signal(signal.SIGINT, _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    print(f"\nGenerating envs {start_env}..{num_envs-1}...\n")

    t_start = time.time()
    n_sft, n_dpo = _count_existing(sft_path), _count_existing(dpo_path)

    # NameError 防护: 在循环前初始化 (Bug 3)
    batch_end = start_env
    i = start_env - 1

    # 批次大小: 多进程用 save_every, 顺序用 1
    batch_size = args.save_every if n_workers > 0 else 1

    for batch_start in range(start_env, num_envs, batch_size):
        if _stop_requested:
            break

        batch_end = min(batch_start + batch_size, num_envs)
        batch_ids = list(range(batch_start, batch_end))

        if n_workers > 0:
            # ── 多进程并发模式 ──
            # 写入临时文件, 批次完成后原子合并 → 防止 mid-batch 崩溃导致
            # 部分 JSONL 行被 _count_existing 误读为 "已完成"
            tmp_sft = sft_path + f".batch_{batch_start}_{batch_end}.tmp"
            tmp_dpo = dpo_path + f".batch_{batch_start}_{batch_end}.tmp"

            batch_t0 = time.time()
            batch_sft, batch_dpo = 0, 0

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_id = {
                    executor.submit(generator._process_one_environment, i): i
                    for i in batch_ids
                }

                for future in as_completed(future_to_id):
                    # Ctrl+C / SIGTERM 检查 — 不再无视中断 (Bug 2)
                    if _stop_requested:
                        # 取消尚未开始的所有 future, 不等待运行中的
                        for f in future_to_id:
                            f.cancel()
                        break

                    i = future_to_id[future]
                    try:
                        sft_sample, dpo_samples = future.result()
                        if sft_sample is not None:
                            _incremental_append(tmp_sft, sft_sample)
                            n_sft += 1
                            batch_sft += 1
                            for d in dpo_samples:
                                _incremental_append(tmp_dpo, d)
                                n_dpo += 1
                                batch_dpo += 1
                    except Exception as e:
                        print(f"\n  [ERROR] env {i}: {e}")

            # 若被中断则跳过合并 + checkpoint (下次从本批次起点续跑)
            if _stop_requested:
                # 清理临时文件: 未完成的批次数据丢弃 (下次重跑)
                for tmp in [tmp_sft, tmp_dpo]:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                break

            # 批次完成 → 原子合并 + checkpoint
            _atomic_merge_batch(tmp_sft, sft_path)
            _atomic_merge_batch(tmp_dpo, dpo_path)
            with open(ckpt_path, "w") as f:
                f.write(f"{batch_end}\n")

            elapsed = time.time() - t_start
            batch_elapsed = time.time() - batch_t0
            done = batch_end - start_env
            rate = elapsed / done if done > 0 else 0
            remaining = (num_envs - batch_end) * rate
            batch_count = len(batch_ids)
            print(f"  [{batch_end}/{num_envs}] +{batch_sft} SFT, +{batch_dpo} DPO | "
                  f"batch {batch_elapsed:.0f}s ({batch_elapsed/batch_count:.1f}s/env) | "
                  f"{elapsed:.0f}s elapsed, ~{remaining/3600:.1f}h remaining",
                  flush=True)

        else:
            # ── 顺序模式 (完全兼容原有行为) ──
            i = batch_start
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

            # 进度输出
            # 小批量 (≤10): 每个 env 都打印，避免 smoke test 静默恐慌
            per_env = num_envs - start_env <= 10
            if per_env or (i - start_env + 1) % args.save_every == 0 or i == start_env:
                elapsed = time.time() - t_start
                done = i - start_env + 1
                rate = elapsed / done
                remaining = (num_envs - i - 1) * rate
                print(f"  [{i+1}/{num_envs}] {n_sft} SFT, {n_dpo} DPO | "
                      f"{elapsed:.0f}s elapsed, ~{remaining/3600:.1f}h remaining | "
                      f"{rate:.1f}s/env", flush=True)
                with open(ckpt_path, "w") as f:
                    f.write(f"{i+1}\n")
            elif (i - start_env + 1) % 10 == 0:
                elapsed = time.time() - t_start
                done = i - start_env + 1
                rate = elapsed / done
                remaining = (num_envs - i - 1) * rate
                print(f"  [{i+1}/{num_envs}] {n_sft} SFT, {n_dpo} DPO | "
                      f"{elapsed:.0f}s elapsed, ~{remaining/3600:.1f}h remaining", flush=True)

    if _stop_requested:
        last_ckpt = f"~{batch_end}" if n_workers > 0 else f"~{i+1}"
        print(f"\nStopped at env {last_ckpt}. {n_sft} SFT, {n_dpo} DPO saved.")
        print(f"Resume with the same command.")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"  SFT: {n_sft}  |  DPO: {n_dpo}")
    print(f"  Files: {sft_path}, {dpo_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
