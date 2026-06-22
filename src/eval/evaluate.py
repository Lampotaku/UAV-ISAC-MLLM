"""
推理与评估脚本
论文 Section 6 — Evaluation Protocol

评估指标:
  1. Network sum rate (通信总速率)
  2. Mean sensing SINR
  3. Mean CRB
  4. Joint satisfaction rate
  5. SCA-FP convergence iterations
  6. Inference latency per slot
"""

import os
import sys
import yaml
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import Gemma3ISAC
from src.solver import SCAFPOptimizer, SCAFPConfig
from src.env import ISACScenarioGenerator
from src.data.prompt_builder import build_full_prompt


def run_evaluation(
    config_path: str,
    model_path: str,
    output_path: str = "./outputs/eval_results.json",
):
    """完整评估管线"""

    # ---- 加载配置 ----
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["simulation"]
    eval_cfg = cfg["eval"]
    model_cfg = cfg["model"]

    num_test = eval_cfg.get("num_test_environments", 200)

    # ---- 初始化仿真环境 ----
    scenario_gen = ISACScenarioGenerator(
        num_uavs=sim_cfg["num_uavs"],
        num_users=sim_cfg["num_users"],
        num_targets=sim_cfg["num_targets"],
        area_size=tuple(sim_cfg["area_size"]),
        carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
        bandwidth_mhz=sim_cfg["bandwidth_mhz"],
        num_antennas=sim_cfg["num_antennas_tx"],
        p_max_dbm=sim_cfg["p_max_dbm"],
        seed=42,
    )

    # ---- 初始化 SCA-FP solver ----
    solver_cfg = SCAFPConfig(
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
        config=solver_cfg,
        M=sim_cfg["num_uavs"],
        K=sim_cfg["num_users"],
        T=sim_cfg["num_targets"],
        N_t=sim_cfg["num_antennas_tx"],
        area_size=tuple(sim_cfg["area_size"]),
        altitude_range=(sim_cfg["altitude_min_m"], sim_cfg["altitude_max_m"]),
        p_max=10 ** ((sim_cfg["p_max_dbm"] - 30) / 10),
        load_cap=sim_cfg["load_cap_per_uav"],
    )

    # ---- 加载模型 (如果提供) ----
    model = None
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = Gemma3ISAC.from_pretrained(
            load_dir=model_path,
            base_model_name=model_cfg["backbone"],
            use_4bit=cfg["hardware"]["use_4bit"],
            lora_rank=model_cfg["lora"]["rank"],
            lora_alpha=model_cfg["lora"]["alpha"],
            num_control_tokens=model_cfg["control_token"]["num_tokens"],
            proj_head_config={
                "hidden_dim": model_cfg["control_token"]["hidden_dim"],
                "num_control_tokens": model_cfg["control_token"]["num_tokens"],
                "M": sim_cfg["num_uavs"],
                "K": sim_cfg["num_users"],
                "area_w": sim_cfg["area_size"][0],
                "area_h": sim_cfg["area_size"][1],
                "h_min": sim_cfg["altitude_min_m"],
                "h_max": sim_cfg["altitude_max_m"],
                "v_max_dt": sim_cfg["uav_max_speed_ms"] * sim_cfg["slot_duration_s"],
                "p_max": 10 ** ((sim_cfg["p_max_dbm"] - 30) / 10),
                "K_max": sim_cfg["load_cap_per_uav"],
            },
        )
        model.eval()
        device = next(model.parameters()).device
    else:
        device = torch.device("cpu")

    # ---- 评估循环 ----
    results = {
        "sum_rate": [],
        "mean_sensing_sinr_db": [],
        "mean_crb": [],
        "joint_satisfaction": [],
        "sca_fp_iterations": [],
        "inference_latency_ms": [],
    }

    for i in tqdm(range(num_test), desc="Evaluating"):
        try:
            metrics = _evaluate_one_sample(
                i, scenario_gen, solver, model, cfg, device
            )
            for k, v in metrics.items():
                results[k].append(v)
        except Exception as e:
            print(f"\n  Sample {i} failed: {e}")
            continue

    # ---- 汇总统计 ----
    summary = {}
    for k, vals in results.items():
        if vals:
            arr = np.array(vals)
            summary[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

    summary["num_samples"] = len(results["sum_rate"])

    # ---- 保存 ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- 打印 ----
    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    for metric, stats in summary.items():
        if metric == "num_samples":
            print(f"\n  Total valid samples: {stats}")
        elif isinstance(stats, dict):
            print(f"\n  {metric}:")
            print(f"    mean = {stats['mean']:.4f}  ±  {stats['std']:.4f}")

    print(f"\nResults saved to {output_path}")
    return summary


def _evaluate_one_sample(
    sample_id: int,
    scenario_gen: ISACScenarioGenerator,
    solver: SCAFPOptimizer,
    model: Gemma3ISAC,
    cfg: dict,
    device: torch.device,
) -> Dict[str, float]:
    """评估一个测试样本"""

    # 采样环境
    env_sample = scenario_gen.sample(sample_id)
    env_dict = {
        "q_current": env_sample.q_current,
        "user_positions": env_sample.u_positions,
        "target_positions": env_sample.s_positions,
        "channel_gains": env_sample.channel_gains_users,
        "user_weights": env_sample.user_weights.copy(),  # use actual heterogeneous weights from env
        "association": env_sample.association,
    }

    # ---- MLLM 热启动 ----
    t0 = time.time()
    if model is not None:
        prompt = build_full_prompt(env_sample, cfg["simulation"])
        q_current_t = torch.tensor(
            env_sample.q_current, dtype=torch.float32, device=device
        ).unsqueeze(0)

        warm_start = model.generate_warmstart(
            prompt,
            q_current=q_current_t,
        )
        warm_start_dict = {
            "delta_q": warm_start["delta_q"].numpy(),
            "delta_a": warm_start["delta_a"].numpy(),
            "delta_p": warm_start["delta_p"].numpy(),
        }
    else:
        warm_start_dict = None

    inference_time_ms = (time.time() - t0) * 1000

    # ---- SCA-FP 优化 ----
    sol = solver.solve(env_dict, warm_start=warm_start_dict, seed=sample_id)

    # ---- 计算指标 ----
    # Sum rate
    sum_rate = 0.0
    for m in range(solver.M):
        for k in range(solver.K):
            if sol.A[m, k] > 0.5:
                sinr = (
                    env_sample.channel_gains_users[m, k]
                    * sol.W_c_power[m, k]
                    / (solver.N0 + 1e-12)
                )
                sum_rate += 20e6 * np.log2(1 + sinr)  # B=20MHz

    # Sensing SINR
    sensing_sinrs = []
    for t in range(solver.T):
        for m in range(solver.M):
            dist_2d = np.linalg.norm(sol.Q[m, :2] - env_sample.s_positions[t])
            dist_3d = np.sqrt(dist_2d ** 2 + sol.Q[m, 2] ** 2)
            wavelength = 3e8 / (cfg["simulation"]["carrier_freq_ghz"] * 1e9)
            pl_db = 20 * np.log10((4 * np.pi * dist_3d) / wavelength) + 20
            pl = 10 ** (-pl_db / 10)
            sinr_s = sol.W_s_power[m] * pl * solver.N_t * solver.N_r / solver.N0
            sensing_sinrs.append(10 * np.log10(sinr_s + 1e-12))

    mean_sinr_db = float(np.mean(sensing_sinrs)) if sensing_sinrs else 0.0

    # Joint satisfaction
    num_satisfied_comm = 0
    for m in range(solver.M):
        for k in range(solver.K):
            if sol.A[m, k] > 0.5:
                sinr = (
                    env_sample.channel_gains_users[m, k]
                    * sol.W_c_power[m, k]
                    / solver.N0
                )
                if 10 * np.log10(sinr + 1e-12) >= cfg["simulation"]["sinr_c_min_db"]:
                    num_satisfied_comm += 1

    num_total_associated = int(np.sum(sol.A > 0.5))
    comm_sat = num_satisfied_comm / max(solver.K, 1)  # 分母=总用户数, 避免"只服务1个用户=100%"的刷榜漏洞

    num_satisfied_sense = 0
    num_targets = solver.T  # s_positions is always shape (T, 2); use solver.T directly
    for t in range(num_targets):
        # Compute sensing SINR from optimised positions (same as sum-rate section)
        best_sinr_db = -np.inf
        for m in range(solver.M):
            dist_2d = np.linalg.norm(sol.Q[m, :2] - env_sample.s_positions[t])
            dist_3d = np.sqrt(dist_2d ** 2 + sol.Q[m, 2] ** 2)
            wavelength = 3e8 / (cfg["simulation"]["carrier_freq_ghz"] * 1e9)
            pl_db = 20 * np.log10((4 * np.pi * dist_3d) / wavelength) + 20
            pl = 10 ** (-pl_db / 10)
            sinr_s = sol.W_s_power[m] * pl * solver.N_t * solver.N_r / solver.N0
            sinr_s_db = 10 * np.log10(sinr_s + 1e-12)
            if sinr_s_db > best_sinr_db:
                best_sinr_db = sinr_s_db
        if best_sinr_db >= cfg["simulation"]["sinr_s_min_db"]:
            num_satisfied_sense += 1

    sense_sat = num_satisfied_sense / max(num_targets, 1)
    joint_sat = (comm_sat + sense_sat) / 2

    return {
        "sum_rate": float(sum_rate / 1e6),  # Mbps
        "mean_sensing_sinr_db": float(mean_sinr_db),
        "mean_crb": 0.0,  # 需要 channel.compute_crb
        "joint_satisfaction": float(joint_sat),
        "sca_fp_iterations": float(sol.iterations),
        "inference_latency_ms": float(inference_time_ms),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="./outputs/eval_results.json")
    args = parser.parse_args()

    run_evaluation(args.config, args.model, args.output)
