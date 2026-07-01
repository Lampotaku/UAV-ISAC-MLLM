"""
轻量级 Generation Eval — 不花钱的 checkpoint 质量检查

评估三个维度:
  1. 文本生成质量: 模型能不能输出合法 JSON? 有没有重复/乱码/模板化?
  2. Control-conditioned: 改 q_current 或 prompt, delta 预测跟着变吗?
  3. 批量 SCA-FP 加速比: warmstart vs cold-start, N 个样本的均值 ± 方差

用法 (服务器上):
  cd /root/UAV-ISAC-MLLM
  # 快速抽查 (Parts 1-2 only, 5 samples)
  python scripts/eval_generation.py \
    --checkpoint /root/autodl-tmp/checkpoints/phase1_step_150 \
    --config configs/default.yaml \
    --n_samples 5 --n_scafp 0

  # 完整评估 (Part 3 用 100 samples 测加速比)
  python scripts/eval_generation.py \
    --checkpoint /root/autodl-tmp/checkpoints/phase1_step_150 \
    --config configs/default.yaml \
    --n_samples 3 --n_scafp 100
"""

# ⚠️ 必须在 import numpy / torch 之前！
# 防止 Intel MKL / OpenBLAS 与 PyTorch DataLoader 多进程打架
import os as _os
_os.environ["OMP_NUM_THREADS"] = "1"
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["MKL_NUM_THREADS"] = "1"
_os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
_os.environ["NUMEXPR_NUM_THREADS"] = "1"
_os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
del _os

import os
import sys
import json
import copy
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import Gemma3ISAC
from src.env import ISACScenarioGenerator
from src.data.prompt_builder import build_full_prompt

SEPARATOR = "=" * 72


def load_model(checkpoint_path: str, config_path: str) -> Gemma3ISAC:
    """加载 checkpoint"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["simulation"]
    model_cfg = cfg["model"]

    print(f"Loading checkpoint: {checkpoint_path}")
    model = Gemma3ISAC.from_pretrained(
        load_dir=checkpoint_path,
        base_model_name=model_cfg["backbone"],
        use_4bit=cfg["hardware"]["use_4bit"],
        lora_rank=model_cfg["lora"]["rank"],
        lora_alpha=model_cfg["lora"]["alpha"],
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        torch_dtype=torch.bfloat16,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
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
            "tau_power": model_cfg["projection_head"]["tau_power"],
            "tau_assoc": model_cfg["projection_head"]["tau_assoc"],
            "sinkhorn_iters": model_cfg["projection_head"]["sinkhorn_iters"],
        },
    )
    # from_pretrained 默认在 CPU, 显式移到 GPU
    model = model.to("cuda")
    model.eval()
    return model, cfg


def generate_text(model: Gemma3ISAC, prompt: str, max_new_tokens: int = 256) -> str:
    """自回归文本生成 — 检查语言质量"""
    device = next(model.base_model.parameters()).device

    inputs = model.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=4096 - max_new_tokens,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print(f"  Prompt tokens: {input_ids.shape[1]}, device: {device}")

    # 追加 control tokens (训练时学到的模式: prompt + <ctrl_0>...<ctrl_7> + JSON)
    ctrl_ids = torch.tensor([model.control_token_ids], device=device)
    input_ids = torch.cat([input_ids, ctrl_ids], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(ctrl_ids)], dim=1)

    # Gemma tokenizer 默认没有 pad_token — 用 eos_token 兜底
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    eos_id = model.tokenizer.eos_token_id

    print(f"  Generating (max {max_new_tokens} tokens)...", end=" ", flush=True)
    with torch.no_grad():
        output_ids = model.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
    print(f"done ({output_ids.shape[1]} total tokens)")

    # 只取新生成的部分
    generated_ids = output_ids[0][input_ids.shape[1]:]
    return model.tokenizer.decode(generated_ids, skip_special_tokens=True)


def check_json_validity(text: str) -> dict:
    """尝试解析生成的 JSON, 返回诊断信息

    处理: markdown 代码块, 截断的 JSON (缺 }]), 空输出
    """
    result = {"valid_json": False, "has_delta_q": False, "has_delta_a": False,
              "has_delta_p": False, "num_floats": 0, "truncated": False, "error": None}

    # 1. 剥离 markdown 代码块 (```json ... ```)
    stripped = text.strip()
    # 移除开头的 ```json 或 ```
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
    # 移除末尾的 ```
    if stripped.rstrip().endswith("```"):
        last_fence = stripped.rstrip().rfind("```")
        stripped = stripped.rstrip()[:last_fence]

    # 2. 查找 JSON 边界
    brace_start = stripped.find("{")
    brace_end = stripped.rfind("}")

    if brace_start == -1:
        result["error"] = "No JSON braces found"
        return result

    json_str = stripped[brace_start:brace_end + 1 if brace_end != -1 else len(stripped)]

    # 3. 尝试直接解析
    try:
        parsed = json.loads(json_str)
        result["valid_json"] = True
        for key in ["delta_q", "delta_a", "delta_p"]:
            if key in parsed:
                result[f"has_{key}"] = True
                val = parsed[key]
                if isinstance(val, list):
                    def count_floats(x):
                        if isinstance(x, (int, float)):
                            return 1
                        if isinstance(x, list):
                            return sum(count_floats(v) for v in x)
                        return 0
                    result["num_floats"] += count_floats(val)
        return result
    except json.JSONDecodeError:
        pass

    # 4. 截断恢复: 从后往前找到第一个未闭合的 [, 补全缺失的 ] → }
    # 典型截断: "...  [0.0, 0.0, 0.0],\n    [0.0, 1.0, " (缺 ] ] } } )
    if brace_end == -1:
        result["truncated"] = True
        # 计算未闭合的括号
        depth_sq = 0   # [
        depth_br = 0   # {
        for ch in json_str:
            if ch == '[':
                depth_sq += 1
            elif ch == ']':
                depth_sq -= 1
            elif ch == '{':
                depth_br += 1
            elif ch == '}':
                depth_br -= 1

        repair = json_str.rstrip().rstrip(",")  # 去掉末尾逗号
        repair += "]" * max(0, depth_sq)
        repair += "}" * max(0, depth_br)
        try:
            parsed = json.loads(repair)
            result["valid_json"] = True
            for key in ["delta_q", "delta_a", "delta_p"]:
                if key in parsed:
                    result[f"has_{key}"] = True
                    val = parsed[key]
                    if isinstance(val, list):
                        def count_floats(x):
                            if isinstance(x, (int, float)):
                                return 1
                            if isinstance(x, list):
                                return sum(count_floats(v) for v in x)
                            return 0
                        result["num_floats"] += count_floats(val)
        except json.JSONDecodeError as e:
            result["error"] = f"Repair failed: {e}"
        return result

    return result


def run_generation_eval(model, cfg, n_samples: int = 5, n_scafp: int = 100):
    """主评估: 文本生成 + control 预测 + 批量 SCA-FP 加速比"""
    sim_cfg = cfg["simulation"]

    print(f"\n{'='*72}")
    print(f"PART 1: Text Generation Quality ({n_samples} samples)")
    print(f"{'='*72}")

    for i in range(n_samples):
        # 用固定 seed 生成可复现的测试环境
        scenario_gen = ISACScenarioGenerator(
            num_uavs=sim_cfg["num_uavs"],
            num_users=sim_cfg["num_users"],
            num_targets=sim_cfg["num_targets"],
            area_size=tuple(sim_cfg["area_size"]),
            carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
            bandwidth_mhz=sim_cfg["bandwidth_mhz"],
            num_antennas=sim_cfg["num_antennas_tx"],
            p_max_dbm=sim_cfg["p_max_dbm"],
            seed=100 + i,
        )
        env = scenario_gen.sample(100 + i)
        prompt = build_full_prompt(env, sim_cfg)

        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt length: {len(prompt)} chars")

        # 生成文本
        generated = generate_text(model, prompt, max_new_tokens=1024)
        print(f"\n[Generated Text] ({len(generated)} chars):")
        print(generated[:1200])
        if len(generated) > 1200:
            print(f"... (truncated, {len(generated)} total)")

        # JSON 检查
        check = check_json_validity(generated)
        print(f"\n[JSON Check] valid={check['valid_json']}, "
              f"delta_q={'✓' if check.get('has_delta_q') else '✗'}, "
              f"floats={check.get('num_floats', 0)}")
        if check.get("error"):
            print(f"  Error: {check['error']}")

    print(f"\n{'='*72}")
    print(f"PART 2: Control-Conditioned Check")
    print(f"{'='*72}")

    for i in range(min(n_samples, 3)):
        scenario_gen = ISACScenarioGenerator(
            num_uavs=sim_cfg["num_uavs"],
            num_users=sim_cfg["num_users"],
            num_targets=sim_cfg["num_targets"],
            area_size=tuple(sim_cfg["area_size"]),
            carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
            bandwidth_mhz=sim_cfg["bandwidth_mhz"],
            num_antennas=sim_cfg["num_antennas_tx"],
            p_max_dbm=sim_cfg["p_max_dbm"],
            seed=200 + i,
        )
        env = scenario_gen.sample(200 + i)
        prompt = build_full_prompt(env, sim_cfg)

        q_current = torch.tensor(env.q_current, dtype=torch.float32)

        # Baseline: 真实 q_current
        ws1 = model.generate_warmstart(prompt, q_current=q_current.clone())

        # 干扰 1: q_current + 10m 随机偏移 → 必须重建 prompt!
        q_shifted = q_current.clone()
        q_shifted[:, :2] += torch.randn_like(q_shifted[:, :2]) * 10.0
        env_shifted = copy.deepcopy(env)
        env_shifted.q_current = q_shifted.numpy()
        prompt_shifted = build_full_prompt(env_shifted, sim_cfg)
        ws2 = model.generate_warmstart(prompt_shifted, q_current=q_shifted)

        # 干扰 2: 所有 UAV 位置归零 → 必须重建 prompt!
        q_zero = torch.zeros_like(q_current)
        env_zero = copy.deepcopy(env)
        env_zero.q_current = q_zero.numpy()
        prompt_zero = build_full_prompt(env_zero, sim_cfg)
        ws3 = model.generate_warmstart(prompt_zero, q_current=q_zero)

        print(f"\n--- Sample {i+1} (Control Sensitivity) ---")
        print(f"q_current norm: {q_current.norm().item():.1f}")
        print(f"q_shifted norm: {q_shifted.norm().item():.1f}")
        print(f"q_zero norm:    {q_zero.norm().item():.1f}")

        for key in ["delta_q", "delta_a", "delta_p"]:
            d1 = ws1[key]
            d2 = ws2[key]
            d3 = ws3[key]

            print(f"\n  {key}:")
            print(f"    baseline:  mean={d1.mean().item():+.4f}, std={d1.std().item():.4f}, "
                  f"range=[{d1.min().item():.4f}, {d1.max().item():.4f}]")
            # 只有当 key 是 delta_q (与 q_current 物理耦合) 时, 才应明显改变;
            # delta_a/delta_p 也可能变化 (q_current 改变会通过投影头传播)
            rel_change_shift = (d2 - d1).norm().item() / (d1.norm().item() + 1e-8)
            rel_change_zero = (d3 - d1).norm().item() / (d1.norm().item() + 1e-8)
            print(f"    Δ vs shifted:    L2 ratio = {rel_change_shift:.4f}")
            print(f"    Δ vs zero:       L2 ratio = {rel_change_zero:.4f}")

    print(f"\n{'='*72}")
    print(f"PART 3: Batch SCA-FP Acceleration Ratio ({n_scafp} samples)")
    print(f"{'='*72}")

    try:
        from src.solver import SCAFPOptimizer, SCAFPConfig
    except ImportError as e:
        print(f"  Skipped (solver import failed): {e}")
        print(f"\n{'='*72}")
        print("Eval complete (Parts 1-2 only).")
        print(f"{'='*72}")
        return

    noise_power = 10 ** (
        (-174 + 10 * np.log10(sim_cfg["bandwidth_mhz"] * 1e6)
         + sim_cfg["noise_figure_db"] - 30) / 10
    )

    solver_cfg = SCAFPConfig(
        max_outer_iters=30, max_inner_iters=50, tol=1e-4,
        lambda_sensing=0.5, lambda_idle_penalty=5.0,
        sinr_c_min=10 ** (sim_cfg["sinr_c_min_db"] / 10),
        sinr_s_min=10 ** (sim_cfg["sinr_s_min_db"] / 10),
        verbose=False,
    )

    solver = SCAFPOptimizer(
        config=solver_cfg,
        M=sim_cfg["num_uavs"], K=sim_cfg["num_users"], T=sim_cfg["num_targets"],
        N_t=sim_cfg["num_antennas_tx"],
        N_r=sim_cfg.get("num_antennas_rx", sim_cfg["num_antennas_tx"]),
        carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
        area_size=tuple(sim_cfg["area_size"]),
        altitude_range=(sim_cfg["altitude_min_m"], sim_cfg["altitude_max_m"]),
        p_max=10 ** ((sim_cfg["p_max_dbm"] - 30) / 10),
        noise_power=noise_power,
        load_cap=sim_cfg["load_cap_per_uav"],
    )

    scenario_gen = ISACScenarioGenerator(
        num_uavs=sim_cfg["num_uavs"],
        num_users=sim_cfg["num_users"],
        num_targets=sim_cfg["num_targets"],
        area_size=tuple(sim_cfg["area_size"]),
        carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
        bandwidth_mhz=sim_cfg["bandwidth_mhz"],
        num_antennas=sim_cfg["num_antennas_tx"],
        p_max_dbm=sim_cfg["p_max_dbm"],
        seed=999,
    )

    speedups = []
    warm_iters_list = []
    cold_iters_list = []

    for idx in tqdm(range(n_scafp), desc="SCA-FP eval"):
        seed = 900 + idx
        env = scenario_gen.sample(seed)
        prompt = build_full_prompt(env, sim_cfg)
        q = torch.tensor(env.q_current, dtype=torch.float32)

        ws = model.generate_warmstart(prompt, q_current=q.clone())

        env_dict = {
            "q_current": env.q_current,
            "user_positions": env.u_positions,
            "target_positions": env.s_positions,
            "channel_gains": env.channel_gains_users,
            "user_weights": env.user_weights.copy(),
            "association": env.association,
        }

        warm_start_dict = {
            "delta_q": ws["delta_q"].detach().numpy(),
            "delta_a": ws["delta_a"].detach().numpy(),
            "delta_p": ws["delta_p"].detach().numpy(),
        }

        sol_warm = solver.solve(env_dict, warm_start=warm_start_dict, seed=seed)
        sol_cold = solver.solve(env_dict, warm_start=None, seed=seed)

        speedup = sol_cold.iterations / max(sol_warm.iterations, 1)
        speedups.append(speedup)
        warm_iters_list.append(sol_warm.iterations)
        cold_iters_list.append(sol_cold.iterations)

    speedups_arr = np.array(speedups)
    warm_arr = np.array(warm_iters_list)
    cold_arr = np.array(cold_iters_list)

    print(f"\n  Warmstart iterations:   {warm_arr.mean():.1f} ± {warm_arr.std():.1f}")
    print(f"  Cold-start iterations:  {cold_arr.mean():.1f} ± {cold_arr.std():.1f}")
    print(f"  SCA-FP Speedup:         {speedups_arr.mean():.3f}x ± {speedups_arr.std():.3f}")
    print(f"  Min/Max speedup:        {speedups_arr.min():.3f}x / {speedups_arr.max():.3f}x")
    print(f"  Speedup ≥ 1.5×:         {(speedups_arr >= 1.5).sum()}/{n_scafp} samples ({(speedups_arr >= 1.5).mean()*100:.1f}%)")
    print(f"  Warmstart ≤ 1 iter (possible collapse): {(warm_arr <= 1).sum()}/{n_scafp} samples")

    print(f"\n{'='*72}")
    print("Eval complete. Key questions to answer:")
    print("  1. Is generated text valid JSON? [Part 1]")
    print("  2. Are delta values in reasonable range (not NaN, not all-zero, not all-same)? [Part 2]")
    print("  3. Does perturbing q_current change the predictions? [Part 2]")
    print(f"  4. Mean SCA-FP speedup ≥ 1.5×?  ← THE ONLY METRIC THAT MATTERS [Part 3]")
    print(f"{'='*72}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint dir (e.g. /root/autodl-tmp/checkpoints/stage1_step_200)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of test samples for Parts 1 & 2 (default 5)")
    parser.add_argument("--n_scafp", type=int, default=100,
                        help="Number of SCA-FP samples for Part 3 acceleration ratio (default 100)")
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.config)
    run_generation_eval(model, cfg, args.n_samples, args.n_scafp)
