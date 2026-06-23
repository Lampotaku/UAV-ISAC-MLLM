#!/usr/bin/env python
"""
数据 EDA (Exploratory Data Analysis) — 训练前全身体检
检查维度:
  1. 格式 & Token 长度  — 防截断、防乱码
  2. 物理常识可视化      — 随机场景 ASCII 3D 视图
  3. 多样性 & 模式崩溃   — 方向分布、功率约束、关联矩阵
"""

import json
import os
import sys
import argparse
import numpy as np
from collections import defaultdict

# ── Colour ──────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"
def ok(s):    return f"{GREEN}{s}{RESET}"
def fail(s):  return f"{RED}{s}{RESET}"
def warn(s):  return f"{YELLOW}{s}{RESET}"
def hdr(s):   return f"{BOLD}{s}{RESET}"
def info(s):  return f"{CYAN}{s}{RESET}"


# ── Config (mirrors default.yaml) ──────────────────
CFG = {
    "M": 4, "K": 20, "T": 6,
    "area_size": [1000, 1000],
    "h_range": [50, 300],
    "v_max": 15,
    "slot_duration": 1.0,
    "p_max_W": 1.0,          # 30 dBm = 1W
    "K_max": 10,
    "max_seq_length": 4096,
    "control_tokens": 8,
    "prompt_budget": 4096 - 512,   # prompt truncated to max_seq_length - 512
    "response_budget": 512,
}


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数: 英文 ~4 chars/token, 数字/JSON ~3 chars/token"""
    alpha_chars = sum(1 for c in text if c.isalpha() or c == ' ')
    other_chars = len(text) - alpha_chars
    return int(alpha_chars / 4 + other_chars / 2.5)


# ====================================================================
# SECTION 1: 格式 & Token 长度检查
# ====================================================================

def check_format_and_length(sft_path, dpo_path):
    print(hdr("\n" + "=" * 70))
    print(hdr("  SECTION 1: Format & Token Length Check"))
    print(hdr("=" * 70))

    sft_lengths = []
    sft_prompt_lens = []
    sft_resp_lens = []
    truncated_prompts = 0
    truncated_responses = 0
    empty_fields = 0
    malformed_json = 0

    # ---- SFT ----
    with open(sft_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                malformed_json += 1
                continue

            # 必需字段
            for field in ["prompt", "response", "delta_q", "delta_a", "delta_p"]:
                if field not in data or not data[field]:
                    empty_fields += 1
                    if empty_fields <= 3:
                        print(f"  {fail('✗')} SFT L{i}: missing '{field}'")

            prompt = data.get("prompt", "")
            response = data.get("response", "")

            p_tok = estimate_tokens(prompt)
            r_tok = estimate_tokens(response)
            total_tok = p_tok + CFG["control_tokens"] + r_tok

            sft_prompt_lens.append(p_tok)
            sft_resp_lens.append(r_tok)
            sft_lengths.append(total_tok)

            if p_tok > CFG["prompt_budget"]:
                truncated_prompts += 1
                if truncated_prompts <= 3:
                    print(f"  {warn('⚠')} SFT L{i}: prompt ~{p_tok} tokens > budget {CFG['prompt_budget']} — will be TRUNCATED")
            if r_tok > CFG["response_budget"]:
                truncated_responses += 1
                if truncated_responses <= 3:
                    print(f"  {warn('⚠')} SFT L{i}: response ~{r_tok} tokens > budget {CFG['response_budget']} — will be TRUNCATED")

            # 打印前 2 条完整样本
            if i <= 2:
                print(hdr(f"\n  ── SFT Sample {i} ──"))
                print(f"  {info('Prompt')} ({len(prompt)} chars, ~{p_tok} tokens):")
                print(f"    {prompt[:500]}...")
                if len(prompt) > 500:
                    print(f"    ... (truncated, {len(prompt)} total chars)")
                print(f"\n  {info('Response')} ({len(response)} chars, ~{r_tok} tokens):")
                print(f"    {response[:600]}...")
                if len(response) > 600:
                    print(f"    ... (truncated, {len(response)} total chars)")

                # Parse response JSON
                try:
                    resp_json = json.loads(response)
                    dq = np.array(resp_json["delta_q"])
                    da = np.array(resp_json["delta_a"])
                    dp = np.array(resp_json["delta_p"])
                    print(f"\n  {info('Parsed shapes:')} δ_q {dq.shape}, δ_a {da.shape}, δ_p {dp.shape}")
                except Exception as e:
                    print(f"  {fail(f'Response not valid JSON: {e}')}")

    # ---- DPO (spot check) ----
    dpo_count = 0
    dpo_truncated = 0
    with open(dpo_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            dpo_count += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                malformed_json += 1
                continue

            if i <= 2:
                prompt = data.get("prompt", "")
                chosen = data.get("chosen", "")
                rejected = data.get("rejected", "")
                p_tok = estimate_tokens(prompt)
                c_tok = estimate_tokens(chosen)
                r_tok = estimate_tokens(rejected)
                print(hdr(f"\n  ── DPO Sample {i} ──"))
                print(f"  Prompt: ~{p_tok} tokens | Chosen: ~{c_tok} tokens | Rejected: ~{r_tok} tokens")
                print(f"  Utility gap: {data.get('utility_gap', 'N/A')}")

    # ---- Summary ----
    sft_lengths = np.array(sft_lengths)
    sft_prompt_lens = np.array(sft_prompt_lens)
    sft_resp_lens = np.array(sft_resp_lens)

    print(hdr(f"\n  ── Token Length Summary ──"))
    print(f"  SFT samples: {len(sft_lengths)}")
    print(f"  Total tokens (prompt+ctrl+resp):")
    print(f"    mean={sft_lengths.mean():.0f}  min={sft_lengths.min():.0f}  max={sft_lengths.max():.0f}")
    print(f"  Prompt tokens:")
    print(f"    mean={sft_prompt_lens.mean():.0f}  min={sft_prompt_lens.min():.0f}  max={sft_prompt_lens.max():.0f}")
    percentile_95 = np.percentile(sft_prompt_lens, 95)
    percentile_99 = np.percentile(sft_prompt_lens, 99)
    print(f"    P95={percentile_95:.0f}  P99={percentile_99:.0f}  budget={CFG['prompt_budget']}")
    print(f"  Response tokens:")
    print(f"    mean={sft_resp_lens.mean():.0f}  min={sft_resp_lens.min():.0f}  max={sft_resp_lens.max():.0f}")
    print(f"    budget={CFG['response_budget']}")

    # Token 分布直方图 (ASCII)
    print(f"\n  {hdr('Prompt token distribution:')}")
    bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3584, 99999]
    labels = ["0-500", "500-1k", "1k-1.5k", "1.5k-2k", "2k-2.5k", "2.5k-3k", "3k-3.5k", ">3.5k(TRUNC)"]
    hist, _ = np.histogram(sft_prompt_lens, bins=bins)
    max_bar = 50
    for lbl, cnt in zip(labels, hist):
        bar = "█" * min(int(cnt / max(hist) * max_bar), max_bar) if max(hist) > 0 else ""
        marker = fail(f"  ← {cnt} TRUNCATED!") if "TRUNC" in lbl and cnt > 0 else ""
        print(f"    {lbl:>16s}: {bar} {cnt}{marker}")

    issues_found = []
    if truncated_prompts > 0:
        issues_found.append(f"{truncated_prompts} prompts will be truncated")
    if truncated_responses > 0:
        issues_found.append(f"{truncated_responses} responses will be truncated")
    if empty_fields > 0:
        issues_found.append(f"{empty_fields} samples have empty required fields")
    if malformed_json > 0:
        issues_found.append(f"{malformed_json} malformed JSON lines")

    if issues_found:
        print(f"\n  {fail(f'SECTION 1 ISSUES:')} {', '.join(issues_found)}")
    else:
        print(f"\n  {ok('✅ Section 1 PASS')} — no truncation, no format issues")

    return {
        "sft_prompt_lens": sft_prompt_lens,
        "sft_resp_lens": sft_resp_lens,
        "sft_lengths": sft_lengths,
        "truncated_prompts": truncated_prompts,
        "truncated_responses": truncated_responses,
        "dpo_count": dpo_count,
    }


# ====================================================================
# SECTION 2: 物理常识 & 3D 场景可视化
# ====================================================================

def ascii_topdown(q_current, u_pos, s_pos, delta_q, area_w=1000, area_h=1000):
    """打印 40×40 的 ASCII 俯视图"""
    grid_w, grid_h = 40, 40
    canvas = [["·" for _ in range(grid_w)] for _ in range(grid_h)]

    def to_grid(x, y):
        gx = int(np.clip(x / area_w * grid_w, 0, grid_w - 1))
        gy = int(np.clip(y / area_h * grid_h, 0, grid_h - 1))
        return gx, grid_h - 1 - gy  # flip Y for display

    # Users: 'U'
    for ux, uy in u_pos[:, :2]:
        gx, gy = to_grid(ux, uy)
        canvas[gy][gx] = info("U")

    # Targets: 'T'
    for sx, sy in s_pos[:, :2]:
        gx, gy = to_grid(sx, sy)
        if canvas[gy][gx] == "·":
            canvas[gy][gx] = warn("T")
        else:
            canvas[gy][gx] = "?"  # overlap

    # UAV starts: '0','1','2','3'
    for m in range(len(q_current)):
        gx, gy = to_grid(q_current[m, 0], q_current[m, 1])
        canvas[gy][gx] = str(m)

    # UAV destinations: '◈' (diamond)
    for m in range(len(delta_q)):
        dx, dy = q_current[m, 0] + delta_q[m, 0], q_current[m, 1] + delta_q[m, 1]
        gx, gy = to_grid(dx, dy)
        if canvas[gy][gx] in "·UT?" or canvas[gy][gx] == str(m):
            canvas[gy][gx] = ok("◈")

    return "\n".join("".join(row) for row in canvas)


def check_physical_spotcheck(sft_path):
    print(hdr("\n" + "=" * 70))
    print(hdr("  SECTION 2: Physical Spot-Check (3D Scene Visualization)"))
    print(hdr("=" * 70))

    # Load all data for random sampling
    all_data = []
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_data), size=min(3, len(all_data)), replace=False)

    for idx in indices:
        data = all_data[idx]
        q_current = np.array(data["q_current"])       # (M, 3)
        delta_q = np.array(data["delta_q"])           # (M, 3)
        delta_p = np.array(data["delta_p"])           # (M, K+1)

        print(hdr(f"\n  ── Environment {data['id']} ──"))
        print(f"  Utility (best of 10): {data.get('utility', 'N/A')}")

        # UAV positions
        print(f"\n  {info('UAV Positions (current → proposed):')}")
        for m in range(len(q_current)):
            dq_3d = np.linalg.norm(delta_q[m])
            print(f"    UAV{m}: ({q_current[m,0]:6.1f}, {q_current[m,1]:6.1f}, {q_current[m,2]:6.1f})m "
                  f"→ Δ=({delta_q[m,0]:+5.1f}, {delta_q[m,1]:+5.1f}, {delta_q[m,2]:+5.1f})m  "
                  f"‖Δ‖₂={dq_3d:.1f}m")

        # Power budget check per UAV
        print(f"\n  {info('Per-UAV Power Budget:')}")
        for m in range(len(delta_p)):
            comm_power = delta_p[m, :CFG["K"]].sum()
            sens_power = delta_p[m, CFG["K"]]
            total = comm_power + sens_power
            status = ok("OK") if total <= CFG["p_max_W"] + 1e-6 else fail(f"OVER! {total:.4f}>{CFG['p_max_W']}")
            print(f"    UAV{m}: comm={comm_power:.4f}W  sens={sens_power:.4f}W  total={total:.4f}W  {status}")

        # ASCII top-down view
        print(f"\n  {info('Top-Down View (U=user, T=target, 0-3=UAV start, ◈=UAV dest):')}")
        print(f"    " + ascii_topdown(q_current, u_pos=None, s_pos=None, delta_q=delta_q).replace("\n", "\n    "))

        # Altitude view
        print(f"\n  {info('Altitude Profile:')}")
        for m in range(len(q_current)):
            bar_start = "█" * int(q_current[m, 2] / 5)
            bar_end = "█" * int((q_current[m, 2] + delta_q[m, 2]) / 5)
            arrow = "↗" if delta_q[m, 2] > 0 else ("↘" if delta_q[m, 2] < 0 else "→")
            print(f"    UAV{m}: {q_current[m,2]:5.0f}m {bar_start} {arrow} {bar_end} {(q_current[m,2]+delta_q[m,2]):5.0f}m")

    print(f"\n  {hdr('Spot-check verdict:')} visually inspect the above for:")
    print(f"    1. UAVs moving toward users/targets (not away into empty space)")
    print(f"    2. Reasonable altitude changes (not all max up/max down)")
    print(f"    3. Power budgets not violated")


# ====================================================================
# SECTION 3: 多样性 & 模式崩溃检查
# ====================================================================

def check_diversity(sft_path):
    print(hdr("\n" + "=" * 70))
    print(hdr("  SECTION 3: Diversity & Mode Collapse Check"))
    print(hdr("=" * 70))

    all_dq = []    # (N, M, 3)
    all_dp = []    # (N, M, K+1)
    all_da = []    # (N, M, K)
    all_qc = []    # (N, M, 3)

    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            all_dq.append(np.array(data["delta_q"]))
            all_dp.append(np.array(data["delta_p"]))
            all_da.append(np.array(data["delta_a"]))
            all_qc.append(np.array(data["q_current"]))

    dq = np.array(all_dq)   # (N, M, 3)
    dp = np.array(all_dp)   # (N, M, K+1)
    da = np.array(all_da)   # (N, M, K)
    qc = np.array(all_qc)   # (N, M, 3)

    N, M = dq.shape[0], dq.shape[1]
    K = CFG["K"]

    # ── 3.1 δ_q 位移幅度分布 ──
    print(hdr(f"\n  ── 3.1 δ_q Displacement Magnitude ──"))
    dq_3d = np.linalg.norm(dq, axis=2)  # (N, M)
    dq_flat = dq_3d.ravel()

    bins = [0, 2, 5, 8, 10, 12, 13, 14, 14.5, 14.9, 15.0, 15.1]
    hist, edges = np.histogram(dq_flat, bins=bins)
    max_bar = 40
    print(f"    Mean={dq_flat.mean():.2f}m  Std={dq_flat.std():.2f}m")
    print(f"    Min={dq_flat.min():.2f}m  Max={dq_flat.max():.2f}m")
    print(f"    % at exactly 15.0m: {100*(dq_flat >= 14.99).mean():.1f}%")
    print(f"    % in [14.5, 15.0]: {100*((dq_flat >= 14.5) & (dq_flat <= 15.0)).mean():.1f}%")
    print(f"    % < 10m: {100*(dq_flat < 10).mean():.1f}%")

    # ── 3.2 δ_q 方向分布 ──
    print(hdr(f"\n  ── 3.2 δ_q Direction Distribution ──"))
    # Horizontal azimuth (dx, dy)
    azimuths = np.degrees(np.arctan2(dq[:, :, 1], dq[:, :, 0])).ravel()  # (-180, 180)
    # Elevation angle from horizontal
    dxdy_norm = np.linalg.norm(dq[:, :, :2], axis=2)  # (N, M)
    elevations = np.degrees(np.arctan2(dq[:, :, 2], dxdy_norm)).ravel()

    # 8-direction wind rose
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    az_bins = np.linspace(-180, 180, 9)  # -180, -135, -90, -45, 0, 45, 90, 135, 180
    az_hist, _ = np.histogram(azimuths, bins=az_bins)
    total = az_hist.sum()
    print(f"    Horizontal direction (wind rose):")
    for i, (dname, cnt) in enumerate(zip(dirs, az_hist)):
        pct = 100 * cnt / total
        bar = "█" * int(pct / 2) if pct > 0 else ""
        flag = fail(" ← BIAS?") if pct > 30 else ""
        print(f"      {dname:>3s}: {pct:5.1f}% {bar}{flag}")

    # Elevation distribution
    el_bins = [-90, -60, -30, -10, 0, 10, 30, 60, 90]
    el_labels = ["↓steep down", "↓down", "↘slight↓", "→flat-", "→flat+", "↗slight↑", "↑up", "↑steep up"]
    el_hist, _ = np.histogram(elevations, bins=el_bins)
    print(f"    Vertical direction:")
    for lbl, cnt in zip(el_labels, el_hist):
        pct = 100 * cnt / total
        bar = "█" * int(pct / 2) if pct > 0 else ""
        print(f"      {lbl:>12s}: {pct:5.1f}% {bar}")

    # ── 3.3 δ_p 功率约束 ──
    print(hdr(f"\n  ── 3.3 δ_p Power Constraint Check ──"))
    comm_power = dp[:, :, :K].sum(axis=2)   # (N, M) — communication power per UAV
    sens_power = dp[:, :, K]                # (N, M) — sensing power per UAV
    total_power = comm_power + sens_power   # (N, M)

    over_budget = (total_power > CFG["p_max_W"] + 1e-6).sum()
    negative_power = (dp < -1e-6).sum()
    zero_power = (total_power < 1e-8).sum()

    print(f"    Per-UAV power budget: P_max = {CFG['p_max_W']}W")
    print(f"    Mean total power: {total_power.mean():.4f}W  (range [{total_power.min():.4f}, {total_power.max():.4f}])")
    print(f"    Mean comm power:  {comm_power.mean():.4f}W")
    print(f"    Mean sens power:  {sens_power.mean():.4f}W")
    print(f"    Sens/Total ratio: {100*(sens_power/total_power.clip(min=1e-8)).mean():.1f}%")

    if over_budget > 0:
        print(f"    {fail(f'✗ {over_budget}/{N*M} UAV-slots EXCEED power budget!')}")
        # Show worst offenders
        violations = np.where(total_power > CFG["p_max_W"] + 0.01)
        for i in range(min(3, len(violations[0]))):
            ni, mi = violations[0][i], violations[1][i]
            print(f"      env={ni} UAV={mi}: {total_power[ni,mi]:.4f}W > {CFG['p_max_W']}W")
    else:
        print(f"    {ok('✓ All UAVs within power budget')}")

    if negative_power > 0:
        print(f"    {fail(f'✗ {negative_power} negative power entries!')}")
    else:
        print(f"    {ok('✓ No negative power values')}")

    if zero_power > 0:
        pct_zero = 100 * zero_power / (N * M)
        print(f"    {warn(f'⚠ {zero_power} UAV-slots have ZERO total power ({pct_zero:.1f}%)')}")
    else:
        print(f"    {ok('✓ All UAVs have non-zero power')}")

    # Power distribution histogram
    print(f"\n    Total power distribution:")
    p_bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0, 1.01, 1.5]
    p_labels = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-0.99", "0.99-1.0", "1.0-1.01", ">1.01(OVER)"]
    p_hist, _ = np.histogram(total_power.ravel(), bins=p_bins)
    for lbl, cnt in zip(p_labels, p_hist):
        pct = 100 * cnt / (N * M)
        bar = "█" * int(pct * 2) if pct > 0 else ""
        marker = fail("  ← OVER BUDGET!") if "OVER" in lbl and cnt > 0 else ""
        print(f"      {lbl:>15s}: {pct:5.1f}% {bar}{marker}")

    # ── 3.4 δ_a 关联矩阵 ──
    print(hdr(f"\n  ── 3.4 δ_a Association Matrix Check ──"))
    da_abs = np.abs(da)
    col_sums = da_abs.sum(axis=1)          # (N, K) — each column (user) sum across UAVs
    row_sums = da_abs.sum(axis=2)          # (N, M) — each row (UAV) sum across users

    col_violations = np.abs(col_sums - 1.0) > 0.2
    n_col_viol = col_violations.sum()

    print(f"    Column sums (per-user soft assignment):")
    print(f"      mean={col_sums.mean():.3f}  std={col_sums.std():.3f}")
    print(f"      range [{col_sums.min():.4f}, {col_sums.max():.4f}]")
    if n_col_viol > 0:
        print(f"    {fail(f'✗ {n_col_viol}/{N*K} entries deviate >0.2 from 1.0')}")
    else:
        print(f"    {ok('✓ All column sums ≈ 1.0 (within ±0.2)')}")

    print(f"    Row sums (per-UAV load):")
    print(f"      mean={row_sums.mean():.2f}  range [{row_sums.min():.2f}, {row_sums.max():.2f}]")
    print(f"      Load cap K_max={CFG['K_max']}")
    overloaded = (row_sums > CFG["K_max"] + 0.5).sum()
    if overloaded > 0:
        print(f"    {fail(f'✗ {overloaded}/{N*M} UAV-slots exceed load cap!')}")
    else:
        print(f"    {ok('✓ All UAV loads within K_max')}")

    # ── 3.5 q_current 初始位置分布 ──
    print(hdr(f"\n  ── 3.5 UAV Initial Position Distribution ──"))
    for m in range(M):
        qm = qc[:, m, :]  # (N, 3)
        print(f"    UAV{m}: x∈[{qm[:,0].min():.0f},{qm[:,0].max():.0f}] "
              f"y∈[{qm[:,1].min():.0f},{qm[:,1].max():.0f}] "
              f"h∈[{qm[:,2].min():.0f},{qm[:,2].max():.0f}]m")

    # ── Section 3 Verdict ──
    issues = []
    if over_budget > 0:
        issues.append(f"{over_budget} power budget violations")
    if negative_power > 0:
        issues.append(f"{negative_power} negative power entries")
    if n_col_viol > 100:
        issues.append(f"{n_col_viol} association column deviations")
    if overloaded > 0:
        issues.append(f"{overloaded} load cap violations")
    # Direction bias: any sector > 35%?
    if any(100 * cnt / total > 35 for cnt in az_hist):
        dominant = dirs[np.argmax(az_hist)]
        issues.append(f"Direction bias toward {dominant} ({100*az_hist.max()/total:.1f}%)")

    print()
    if issues:
        print(f"  {fail(f'SECTION 3 ISSUES:')} {', '.join(issues)}")
    else:
        print(f"  {ok('✅ Section 3 PASS')} — good diversity, no constraint violations")


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Data EDA before training")
    parser.add_argument("--data-dir", type=str, default="/root/autodl-tmp/data/full5000",
                        help="Path to data directory")
    args = parser.parse_args()

    sft_path = os.path.join(args.data_dir, "sft_dataset.jsonl")
    dpo_path = os.path.join(args.data_dir, "dpo_dataset.jsonl")

    for p, name in [(sft_path, "SFT"), (dpo_path, "DPO")]:
        if not os.path.exists(p):
            print(f"{fail('ERROR')}: {name} file not found: {p}")
            sys.exit(1)
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"{name}: {p} ({size_mb:.1f} MB)")

    stats = check_format_and_length(sft_path, dpo_path)
    check_physical_spotcheck(sft_path)
    check_diversity(sft_path)

    # ── Final Verdict ──
    print(hdr("\n" + "=" * 70))
    print(hdr("  FINAL VERDICT"))
    print(hdr("=" * 70))

    all_ok = True
    if stats["truncated_prompts"] > 0:
        print(f"  {warn('⚠')} Prompt truncation: {stats['truncated_prompts']} samples — may lose task context")
        all_ok = False
    if stats["truncated_responses"] > 0:
        print(f"  {fail('✗')} Response truncation: {stats['truncated_responses']} samples — JSON output CUT OFF!")
        all_ok = False

    if all_ok:
        print(f"  {ok('✅ All checks passed — ready for SFT training!')}")
    else:
        print(f"  {fail('⚠️  Address the issues above before training.')}")

    print(hdr("=" * 70))
    print()


if __name__ == "__main__":
    main()
