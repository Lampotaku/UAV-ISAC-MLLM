#!/usr/bin/env python
"""
数据质量验证脚本
在数据生成过程中或完成后运行, 快速检查数据管线是否正常

用法:
  python scripts/validate_data.py --data-dir /root/autodl-tmp/data/cache
  python scripts/validate_data.py --data-dir /root/autodl-tmp/data/cache --watch 60   # 每60秒检查一次

检查项:
  1. 文件完整性 — JSONL 行数, 格式合法性
  2. SFT 样本 — prompt/response 非空, q_current 维度正确
  3. DPO 样本 — chosen/rejected 非空, utility 单调性 (chosen > rejected)
  4. Prior 合理性 — delta_q 量级, delta_a 概率范围, delta_p 功率范围
  5. 物理一致性 — channel_gain 非 NaN, 位置在区域内
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from collections import defaultdict


# ── Colour helpers ────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(s):    return f"{GREEN}{s}{RESET}"
def fail(s):  return f"{RED}{s}{RESET}"
def warn(s):  return f"{YELLOW}{s}{RESET}"
def info(s):  return f"{CYAN}{s}{RESET}"
def hdr(s):   return f"{BOLD}{s}{RESET}"


# ── Validation functions ──────────────────────────────────────────

def validate_jsonl(path):
    """检查 JSONL 文件: 逐行可解析, 返回 (records, errors)"""
    records = []
    errors = []
    if not os.path.exists(path):
        return [], [f"文件不存在: {path}"]
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"L{i}: JSON 解析失败 — {e}")
    return records, errors


def validate_sft_sample(item, idx, cfg):
    """验证单条 SFT 样本"""
    issues = []

    # 必需字段
    for field in ["prompt", "response"]:
        if field not in item or not item[field]:
            issues.append(f"L{idx}: 缺字段 '{field}'")

    # q_current 维度
    q = item.get("q_current", [])
    M = cfg["M"]
    if len(q) != M or any(len(qi) != 3 for qi in q):
        issues.append(f"L{idx}: q_current 维度错误, 期望 ({M},3), 实际 {np.array(q).shape if len(q)>0 else 'empty'}")

    # delta_q 维度
    dq = item.get("delta_q", [])
    if len(dq) != M or any(len(dqi) != 3 for dqi in dq):
        issues.append(f"L{idx}: delta_q 维度错误, 期望 ({M},3), 实际 {np.array(dq).shape if len(dq)>0 else 'empty'}")

    return issues


def validate_dpo_sample(item, idx, cfg):
    """验证单条 DPO 样本"""
    issues = []
    M, K = cfg["M"], cfg["K"]

    for role in ["chosen", "rejected"]:
        if role not in item or not item[role]:
            issues.append(f"L{idx}: 缺 '{role}' response")

    # Utility 单调性
    u_chosen = item.get("utility_chosen", None)
    u_rejected = item.get("utility_rejected", None)
    if u_chosen is not None and u_rejected is not None:
        if u_chosen <= u_rejected:
            issues.append(f"L{idx}: utility 不单调 — chosen={u_chosen:.4f} <= rejected={u_rejected:.4f}")
        elif u_chosen - u_rejected < 1e-8:
            issues.append(f"L{idx}: utility 差距过小 — Δ={u_chosen-u_rejected:.2e}")

    return issues


def validate_prior(item, idx, cfg):
    """验证 prior 物理合理性"""
    issues = []
    M, K = cfg["M"], cfg["K"]
    area_w, area_h = cfg["area_size"]
    h_min, h_max = cfg["h_range"]
    p_max = cfg["p_max"]
    v_max_dt = cfg["v_max"] * cfg["slot_duration"]

    # --- delta_q ---
    dq = np.array(item.get("delta_q", []))
    if dq.size > 0:
        if dq.shape != (M, 3):
            issues.append(f"L{idx}: delta_q shape={dq.shape}, 期望 ({M},3)")
        elif not np.all(np.isfinite(dq)):
            issues.append(f"L{idx}: delta_q 含 NaN/Inf")
        else:
            dq_3d = np.linalg.norm(dq, axis=1)  # (M,) 3D Euclidean displacement
            # 物理约束: ‖Δq‖₂ ≤ v_max * Δt (球体, 非正方体 — Box bounds 的角可超 √3·15≈26m)
            if dq_3d.max() > v_max_dt + 1e-3:  # 1e-3 容忍浮点误差
                issues.append(
                    f"L{idx}: delta_q 3D位移 max={dq_3d.max():.1f}m "
                    f"超出物理约束 v_max*Δt={v_max_dt}m"
                )

    # --- delta_a ---
    da = np.array(item.get("delta_a", []))
    if da.size > 0 and da.shape == (M, K):
        if not np.all(np.isfinite(da)):
            issues.append(f"L{idx}: delta_a 含 NaN/Inf")
        elif np.any(da < 0):
            issues.append(f"L{idx}: delta_a 含负值 (range [{da.min():.4f}, {da.max():.4f}])")

    # --- delta_p ---
    dp = np.array(item.get("delta_p", []))
    if dp.size > 0:
        if not np.all(np.isfinite(dp)):
            issues.append(f"L{idx}: delta_p 含 NaN/Inf")
        elif np.any(dp < 0):
            issues.append(f"L{idx}: delta_p 含负值")
        elif dp.max() > p_max * 2:
            issues.append(f"L{idx}: delta_p max={dp.max():.4f}W > 2*P_max={2*p_max:.2f}W")

    # --- q_current ---
    qc = np.array(item.get("q_current", []))
    if qc.size > 0 and qc.shape == (M, 3):
        if qc[:, 0].min() < 0 or qc[:, 0].max() > area_w:
            issues.append(f"L{idx}: q_current x 超出区域 [{qc[:,0].min():.0f}, {qc[:,0].max():.0f}]")
        if qc[:, 1].min() < 0 or qc[:, 1].max() > area_h:
            issues.append(f"L{idx}: q_current y 超出区域 [{qc[:,1].min():.0f}, {qc[:,1].max():.0f}]")
        if qc[:, 2].min() < h_min or qc[:, 2].max() > h_max:
            issues.append(f"L{idx}: q_current h 超出 [{h_min},{h_max}]")

    return issues


# ── Statistics helpers ────────────────────────────────────────────

def compute_stats(sft_records, dpo_records, cfg):
    """计算汇总统计"""
    stats = {}
    M, K = cfg["M"], cfg["K"]

    # SFT
    if sft_records:
        dq_all = np.array([r["delta_q"] for r in sft_records])  # (N, M, 3)
        dq_3d = np.linalg.norm(dq_all, axis=2)                   # (N, M) 3D Euclidean
        stats["sft"] = {
            "count": len(sft_records),
            "delta_q_3d": {"min": float(dq_3d.min()), "mean": float(dq_3d.mean()), "max": float(dq_3d.max())},
        }

    # DPO
    if dpo_records:
        u_chosen  = np.array([r["utility_chosen"]  for r in dpo_records if "utility_chosen"  in r])
        u_rejected = np.array([r["utility_rejected"] for r in dpo_records if "utility_rejected" in r])
        if len(u_chosen) > 0 and len(u_rejected) > 0:
            deltas = u_chosen - u_rejected
            stats["dpo"] = {
                "count": len(dpo_records),
                "utility_chosen":  {"min": u_chosen.min(),  "mean": u_chosen.mean(),  "max": u_chosen.max()},
                "utility_rejected": {"min": u_rejected.min(), "mean": u_rejected.mean(), "max": u_rejected.max()},
                "utility_delta":    {"min": deltas.min(),     "mean": deltas.mean(),     "max": deltas.max()},
            }

    return stats


def print_summary(stats, issues_total, max_displacement):
    """打印人类可读的统计摘要"""
    print()
    print(hdr("─" * 60))
    print(hdr("  Data Quality Report"))
    print(hdr("─" * 60))

    if "sft" in stats:
        s = stats["sft"]
        print(f"\n  {hdr('SFT Samples')}: {s['count']}")
        print(f"    δ_q 3D位移 (‖Δq‖₂):  mean={s['delta_q_3d']['mean']:.1f}m  "
              f"[{s['delta_q_3d']['min']:.1f}, {s['delta_q_3d']['max']:.1f}]  "
              f"(上限={max_displacement:.0f}m)")

    if "dpo" in stats:
        d = stats["dpo"]
        print(f"\n  {hdr('DPO Samples')}: {d['count']}")
        print(f"    Utility chosen:   mean={d['utility_chosen']['mean']:.4f}  "
              f"[{d['utility_chosen']['min']:.4f}, {d['utility_chosen']['max']:.4f}]")
        print(f"    Utility rejected: mean={d['utility_rejected']['mean']:.4f}  "
              f"[{d['utility_rejected']['min']:.4f}, {d['utility_rejected']['max']:.4f}]")
        print(f"    Utility Δ:        mean={d['utility_delta']['mean']:.4f}  "
              f"[{d['utility_delta']['min']:.4f}, {d['utility_delta']['max']:.4f}]")
        neg_delta = int(np.sum(np.array([
            r.get("utility_chosen", 0) <= r.get("utility_rejected", 0)
            for r in (dpo_records_cache if 'dpo_records_cache' in dir() else [])
        ])))
        # We'll count negative deltas from the issues list instead

    print(f"\n  {hdr('Issues')}: {issues_total if issues_total > 0 else ok('0 — all clean')}")
    if issues_total > 0:
        print(f"  (see details above for per-sample issues)")

    # Verdict
    print()
    if issues_total == 0:
        print(f"  {ok('✅ 数据质量正常 — 可以继续训练')}")
    else:
        print(f"  {warn('⚠️  存在异常 — 建议检查上面列出的具体问题')}")
    print(hdr("─" * 60))
    print()


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate UAV-ISAC generated training data")
    parser.add_argument("--data-dir", type=str, default="/root/autodl-tmp/data/cache",
                        help="Path to data directory containing sft_dataset.jsonl and dpo_dataset.jsonl")
    parser.add_argument("--watch", type=int, default=0,
                        help="Watch mode: re-check every N seconds (0 = once)")
    parser.add_argument("--max-issues", type=int, default=20,
                        help="Maximum issue lines to print (avoid flooding)")
    args = parser.parse_args()

    # Configuration (mirrors default.yaml simulation section)
    cfg = {
        "M": 4, "K": 20, "T": 6,
        "area_size": [1000, 1000],
        "h_range": [50, 300],
        "v_max": 15,
        "slot_duration": 1.0,
        "p_max": 1.0,  # 30 dBm = 1W
    }

    sft_path = os.path.join(args.data_dir, "sft_dataset.jsonl")
    dpo_path = os.path.join(args.data_dir, "dpo_dataset.jsonl")

    while True:
        print(hdr(f"\n{'='*60}"))
        print(hdr(f"  UAV-ISAC Data Validation"))
        print(hdr(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"))
        print(hdr(f"{'='*60}"))

        # ── 1. Load ──
        print(f"\n  Loading SFT: {sft_path}")
        sft_records, sft_parse_errs = validate_jsonl(sft_path)
        print(f"  Loading DPO: {dpo_path}")
        dpo_records, dpo_parse_errs = validate_jsonl(dpo_path)

        parse_issues = sft_parse_errs + dpo_parse_errs
        if parse_issues:
            for e in parse_issues[:args.max_issues]:
                print(f"  {fail(e)}")

        # ── 2. Validate SFT ──
        print(f"\n  Checking {len(sft_records)} SFT samples...")
        sft_issues = []
        for i, item in enumerate(sft_records[-20:], len(sft_records) - min(20, len(sft_records)) + 1):  # last 20
            sft_issues.extend(validate_sft_sample(item, i, cfg))

        # ── 3. Validate DPO ──
        print(f"  Checking {len(dpo_records)} DPO samples...")
        dpo_issues = []
        sample_size = min(50, len(dpo_records))
        for item in dpo_records[-sample_size:]:
            idx = dpo_records.index(item)
            dpo_issues.extend(validate_dpo_sample(item, idx, cfg))
            dpo_issues.extend(validate_prior(item, idx, cfg))

        # ── 4. Aggregate ──
        all_issues = parse_issues + sft_issues + dpo_issues
        # Count utility violations from DPO records
        if dpo_records:
            dpo_arr = np.array([
                (r.get("utility_chosen", 0), r.get("utility_rejected", 0))
                for r in dpo_records
                if "utility_chosen" in r and "utility_rejected" in r
            ])
            if len(dpo_arr) > 0:
                violations = int(np.sum(dpo_arr[:, 0] <= dpo_arr[:, 1]))
                if violations > 0:
                    all_issues.append(f"Total utility violations (chosen<=rejected): {violations}/{len(dpo_arr)}")

        # ── 5. Stats ──
        stats = compute_stats(sft_records, dpo_records, cfg)

        # Print issues (limited)
        if all_issues:
            print(f"\n  {fail(f'{len(all_issues)} issues found')}:")
            for e in all_issues[:args.max_issues]:
                print(f"    {fail('✗')} {e}")
            if len(all_issues) > args.max_issues:
                print(f"    ... and {len(all_issues) - args.max_issues} more")

        print_summary(stats, len(all_issues), cfg['v_max'] * cfg['slot_duration'])

        if args.watch <= 0:
            break

        print(f"\n  Next check in {args.watch}s... (Ctrl+C to stop)")
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
