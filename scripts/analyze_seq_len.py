#!/usr/bin/env python
"""
分析 SFT 数据集中 Prompt+Response 的真实 Token 长度分布

如果大部分样本 << 4096, 砍短 max_seq_length 可大幅提速
(注意力计算与序列长度呈超线性增长)

用法:
  python scripts/analyze_seq_len.py --data-dir /root/autodl-tmp/data/full5000
"""

import os
import sys
import json
import argparse

# 前置环境变量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TORCHINDUCTOR_FLEX_ATTENTION"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import numpy as np
from pathlib import Path
from transformers import AutoTokenizer


def analyze(data_path: str, model_name: str = "/root/autodl-tmp/huggingface/models/gemma-3-12b-it"):
    # ---- 加载 tokenizer ----
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ---- 加载数据 ----
    print(f"Loading dataset from {data_path}...")
    samples = []
    with open(data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Total samples: {len(samples)}")

    # ---- 统计 ----
    lengths = []
    prompt_lengths = []
    response_lengths = []

    for s in samples:
        # prompt: instruction + control part
        prompt_text = s.get("prompt", "")
        # response: expected output (labels)
        response_text = s.get("response", s.get("completion", ""))

        full_text = prompt_text + response_text

        full_len = len(tokenizer.encode(full_text, add_special_tokens=True))
        prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=True))
        resp_len = full_len - prompt_len

        lengths.append(full_len)
        prompt_lengths.append(prompt_len)
        response_lengths.append(resp_len)

    lengths = np.array(lengths)
    prompt_lengths = np.array(prompt_lengths)
    response_lengths = np.array(response_lengths)

    # ---- 报告 ----
    print("\n" + "=" * 60)
    print("  Token 长度分布 (Prompt + Response)")
    print("=" * 60)
    print(f"  Samples:     {len(lengths)}")
    print(f"  Min:         {lengths.min()}")
    print(f"  Max:         {lengths.max()}")
    print(f"  Mean:        {lengths.mean():.0f}")
    print(f"  Median:      {np.median(lengths):.0f}")
    print(f"  Std:         {lengths.std():.0f}")
    print(f"  90th %ile:   {np.percentile(lengths, 90):.0f}")
    print(f"  95th %ile:   {np.percentile(lengths, 95):.0f}")
    print(f"  99th %ile:   {np.percentile(lengths, 99):.0f}")
    print(f"  99.9th %ile: {np.percentile(lengths, 99.9):.0f}")

    # ---- 分桶 ----
    buckets = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    print(f"\n{'Bucket':>8}  {'Count':>6}  {'Cum%':>8}")
    print("-" * 30)
    for b in buckets:
        count = (lengths <= b).sum()
        pct = count / len(lengths) * 100
        print(f"  ≤{b:5d}  {count:6d}  {pct:7.1f}%")

    # ---- 推荐 ----
    print("\n" + "=" * 60)
    print("  推荐 max_seq_length")
    print("=" * 60)

    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)

    for percentile, val in [(95, p95), (99, p99)]:
        # 向上取整到 128 的倍数
        rounded = int(np.ceil(val / 128) * 128)
        # 裁剪到合理范围
        rounded = max(512, min(4096, rounded))
        truncated = (lengths > rounded).sum()
        trunc_pct = truncated / len(lengths) * 100
        print(f"  {percentile}th %ile → {rounded:4d} tokens "
              f"(截断 {truncated}/{len(lengths)} = {trunc_pct:.2f}% 样本)")

    # ---- 加速预估 ----
    current = 4096
    print(f"\n  当前 max_seq_length = {current}")
    for candidate in [2048, 2560, 3072]:
        if candidate < p95:
            continue
        speedup = (current ** 2) / (candidate ** 2)  # 近似: 注意力 O(n²)
        print(f"  若改为 {candidate} → 注意力加速约 {speedup:.1f}×"
              f" (截断 ~{(lengths > candidate).sum()} 样本)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to data directory containing sft_dataset.jsonl")
    parser.add_argument("--model", type=str,
                        default="/root/autodl-tmp/huggingface/models/gemma-3-12b-it")
    args = parser.parse_args()

    data_path = os.path.join(args.data_dir, "sft_dataset.jsonl")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found!")
        sys.exit(1)

    analyze(data_path, args.model)
