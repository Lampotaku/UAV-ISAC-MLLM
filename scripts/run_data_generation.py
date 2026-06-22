#!/usr/bin/env python
"""
数据生成入口脚本
论文 Algorithm 1 — Best-of-N SCA-FP Restarts

用法:
  python scripts/run_data_generation.py --config configs/default.yaml
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np

from src.env import ISACScenarioGenerator
from src.solver import SCAFPOptimizer, SCAFPConfig
from src.data import OracleDataGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate Best-of-N oracle data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_env", type=int, default=None,
                        help="Override number of environments")
    parser.add_argument("--num_restarts", type=int, default=None,
                        help="Override number of SCA-FP restarts")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sc = cfg["simulation"]
    dc = cfg["data"]

    if args.num_env is not None:
        dc["num_environments"] = args.num_env
    if args.num_restarts is not None:
        dc["num_restarts"] = args.num_restarts
    if args.output_dir is not None:
        dc["output_dir"] = args.output_dir

    print("=" * 60)
    print("Best-of-N Data Generation")
    print("=" * 60)
    print(f"  Environments:  S = {dc['num_environments']}")
    print(f"  Restarts/env:  N = {dc['num_restarts']}")
    print(f"  Total SCA-FP:  {dc['num_environments'] * dc['num_restarts']}")
    print(f"  Output:        {dc['output_dir']}")
    print()

    # 初始化仿真
    scenario_gen = ISACScenarioGenerator(
        num_uavs=sc["num_uavs"],
        num_users=sc["num_users"],
        num_targets=sc["num_targets"],
        area_size=tuple(sc["area_size"]),
        carrier_freq_ghz=sc["carrier_freq_ghz"],
        bandwidth_mhz=sc["bandwidth_mhz"],
        num_antennas=sc["num_antennas_tx"],
        p_max_dbm=sc["p_max_dbm"],
        seed=cfg["training"]["seed"],
    )

    # 初始化 SCA-FP solver
    solver_cfg = SCAFPConfig(
        max_outer_iters=30,
        max_inner_iters=50,
        tol=1e-4,
        lambda_sensing=0.5,
        lambda_idle_penalty=5.0,
        sinr_c_min=10 ** (sc["sinr_c_min_db"] / 10),
        sinr_s_min=10 ** (sc["sinr_s_min_db"] / 10),
        verbose=False,
    )

    solver = SCAFPOptimizer(
        config=solver_cfg,
        M=sc["num_uavs"],
        K=sc["num_users"],
        T=sc["num_targets"],
        N_t=sc["num_antennas_tx"],
        area_size=tuple(sc["area_size"]),
        altitude_range=(sc["altitude_min_m"], sc["altitude_max_m"]),
        p_max=10 ** ((sc["p_max_dbm"] - 30) / 10),
        noise_power=10 ** ((-174 + 10 * np.log10(sc["bandwidth_mhz"] * 1e6) + sc["noise_figure_db"] - 30) / 10),
        load_cap=sc["load_cap_per_uav"],
    )

    # 生成数据
    gen = OracleDataGenerator(scenario_gen, solver, dc, sim_config=sc)
    sft_data, dpo_data = gen.generate_all()

    print(f"\nDone! Generated {len(sft_data)} SFT samples "
          f"and {len(dpo_data)} DPO pairs.")


if __name__ == "__main__":
    main()
