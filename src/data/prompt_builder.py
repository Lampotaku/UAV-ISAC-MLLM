"""
Prompt 构造器
论文 Section 3 — 构造多模态 prompt Π(t)

模态:
  - 通信摘要 c(t): 每用户 SINR, UAV 负载, 速率压力
  - 感知摘要 r(t): 每目标 SINR, 定位难度, 未覆盖目标
  - BEV 地图 V(t): 文本网格或图像
  - 系统指令: 优化目标 + 约束说明 + 输出格式
"""
import numpy as np


SYSTEM_INSTRUCTION = """You are a UAV-ISAC decision controller for low-altitude IoT networks.

Your task: Given the current network state (communication summary, sensing summary, and bird's-eye-view map), propose a warm-start decision prior δ = [δ_q, δ_a, δ_p] that will be used to initialize a numerical optimizer.

## Decision Variables
- δ_q: UAV displacement suggestions (shape {M}×3). Each row is [dx, dy, dh] in meters. dx,dy are horizontal moves; dh is altitude change. Clamp dx,dy to [-v_max*Δt, +v_max*Δt] ≈ [-15, 15]m per slot. Clamp altitude to [{H_min}, {H_max}]m.
- δ_a: User association proposals (shape {M}×{K}). A soft assignment matrix where each column sums to 1. Higher values mean stronger recommendation for that UAV to serve that user.
- δ_p: Power allocation hints (shape {M}×({K}+1)). First {K} entries per UAV are communication power for each user; last entry is sensing probe power. Each UAV row must satisfy sum(δ_p[m]) ≤ P_max = {P_max_W:.4f}W.

## Constraints (must respect)
1. Per-UAV power budget: sum of communication + sensing power ≤ P_max
2. Altitude: {H_min}m ≤ H_m ≤ {H_max}m
3. Single association: each user served by exactly one UAV
4. Per-UAV load cap: at most {K_max} users per UAV
5. UAV minimum separation: ≥ {d_min}m
6. Communication SINR ≥ {sinr_c_min_dB}dB for associated users
7. Sensing SINR ≥ {sinr_s_min_dB}dB for detected targets

## Output Format
Return ONLY a valid JSON object with the following structure:
```json
{{
  "delta_q": [[dx1, dy1, dh1], [dx2, dy2, dh2], ...],
  "delta_a": [[a_11, a_12, ...], [a_21, a_22, ...], ...],
  "delta_p": [[p_c11, p_c12, ..., p_s1], [p_c21, ..., p_s2], ...]
}}
```

Propose a warm start that maximizes: weighted sum-rate + λ_s × sensing SINR − λ_f × idle UAV penalty."""


def build_system_prompt(config: dict) -> str:
    """
    构造系统指令 (填入具体参数值)

    Args:
        config: 仿真配置 dict

    Returns:
        参数化的系统指令字符串
    """
    params = {
        "M": config.get("num_uavs", 4),
        "K": config.get("num_users", 20),
        "H_min": config.get("altitude_min_m", 50),
        "H_max": config.get("altitude_max_m", 300),
        "v_max": config.get("uav_max_speed_ms", 15),
        "K_max": config.get("load_cap_per_uav", 10),
        "d_min": config.get("uav_min_separation_m", 10),
        "sinr_c_min_dB": config.get("sinr_c_min_db", 0),
        "sinr_s_min_dB": config.get("sinr_s_min_db", 10),
        "P_max_W": 10 ** ((config.get("p_max_dbm", 30) - 30) / 10),
    }
    return SYSTEM_INSTRUCTION.format(**params)


def build_communication_summary_str(summary: dict) -> str:
    """格式化通信摘要 c(t) 为文本"""
    lines = ["[Communication Summary c(t)]"]
    lines.append(f"  Per-user SINR (dB): {summary['per_user_sinr_db']}")
    lines.append(f"  Per-UAV load (#users): {summary['per_uav_load']}")
    lines.append(f"  Rate pressure (req/achievable): {summary['rate_pressure']}")
    return "\n".join(lines)


def build_sensing_summary_str(summary: dict) -> str:
    """格式化感知摘要 r(t) 为文本"""
    lines = ["[Sensing Summary r(t)]"]
    lines.append(f"  Per-target sensing SINR (dB): {summary['per_target_sinr_db']}")
    lines.append(f"  Localization difficulty (CRB/ε_max): {summary['localization_difficulty']}")
    lines.append(f"  Uncovered targets (< Γ_s^min): {summary['uncovered_targets']}")
    lines.append(f"  Best UAV per target: {summary['best_uav_per_target']}")
    return "\n".join(lines)


def build_full_prompt(
    env_sample,
    config: dict,
) -> str:
    """
    构造完整的多模态 prompt Π(t)

    格式: System Instruction + Communication Summary + Sensing Summary + BEV Grid

    Args:
        env_sample: EnvironmentSample 对象
        config: 仿真配置 dict

    Returns:
        完整 prompt 字符串
    """
    parts = []

    # 1. 系统指令
    parts.append(build_system_prompt(config))

    # 2. 通信摘要
    parts.append(build_communication_summary_str(env_sample.comm_summary))

    # 3. 感知摘要
    parts.append(build_sensing_summary_str(env_sample.sensing_summary))

    # 4. BEV 文本网格
    parts.append(env_sample.bev_grid_text)

    # 5. 最终指令
    parts.append("\nNow propose the warm-start decision prior δ in JSON format.")

    return "\n\n".join(parts)


def format_oracle_response(sample_id: int, delta_q, delta_a, delta_p) -> str:
    """
    将 Oracle prior 序列化为 JSON 响应字符串

    对应论文中的 Ξ(Ω*) → δ (公式 14-16)

    Args:
        delta_q: (M, 3) UAV 位移
        delta_a: (M, K) 关联矩阵
        delta_p: (M, K+1) 功率分配

    Returns:
        JSON 格式的响应字符串
    """
    import json

    response_dict = {
        "delta_q": np.round(delta_q, 4).tolist(),
        "delta_a": np.round(delta_a, 4).tolist(),
        "delta_p": np.round(delta_p, 4).tolist(),
    }

    return json.dumps(response_dict, indent=2)
