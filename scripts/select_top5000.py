#!/usr/bin/env python3
"""
Phase 2: 质量闸门 — Top-5000 精选

按 utility_gap 降序排列 DPO 对，取前 5000 条。
不设绝对阈值（ADR 006: 避免困难环境中全军覆没）。

用法:
    python scripts/select_top5000.py \
        --input /root/autodl-tmp/data/cache/dpo_dataset.jsonl \
        --output /root/autodl-tmp/data/cache/dpo_top5000.jsonl \
        --top 5000

输出:
    - {output}           — Top-K DPO 对
    - {output}.report    — 统计摘要
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Top-K quality gate for DPO pairs")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to full dpo_dataset.jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="Path for top-K output JSONL")
    parser.add_argument("--top", type=int, default=5000,
                        help="Number of top pairs to keep (default: 5000)")
    args = parser.parse_args()

    # ── 加载全部 DPO 对 ──
    print(f"Loading: {args.input}")
    pairs = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))

    n_total = len(pairs)
    print(f"  Loaded {n_total} pairs")

    if n_total == 0:
        print("ERROR: No DPO pairs found.")
        return 1

    # ── 按 utility_gap 降序 ──
    pairs.sort(key=lambda d: d.get("utility_gap", 0), reverse=True)

    # ── 取 Top-K ──
    top_k = min(args.top, n_total)
    selected = pairs[:top_k]

    with open(args.output, "w", encoding="utf-8") as f:
        for d in selected:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ── 统计 ──
    gaps = [d["utility_gap"] for d in selected]
    all_gaps = [d["utility_gap"] for d in pairs]

    report_lines = [
        "=" * 60,
        "Top-5000 Quality Gate Report",
        "=" * 60,
        f"  Input:         {n_total} pairs",
        f"  Selected:      {top_k} pairs  (top {top_k/n_total*100:.1f}%)",
        "",
        f"  Full gap range:    [{all_gaps[-1]:.6f}, {all_gaps[0]:.6f}]",
        f"  Top-K gap range:   [{gaps[-1]:.6f}, {gaps[0]:.6f}]",
        f"  Top-K gap median:  {gaps[len(gaps)//2]:.6f}",
        f"  Cutoff gap:        {gaps[-1]:.6f}  (min gap in selected)",
        f"  Discarded:         {n_total - top_k} pairs  (max gap: {pairs[top_k]['utility_gap']:.6f} if available)",
        "",
        f"  Output: {args.output}",
        "=" * 60,
    ]

    report = "\n".join(report_lines)
    print(report)

    report_path = args.output + ".report"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\nReport saved: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
