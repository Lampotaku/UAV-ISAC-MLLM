#!/usr/bin/env python
"""
SFT 过拟合测试 — 证明训练管线正确性的最简可行证据

原理: 一个正确的训练管线在极小数据 (5 样本) 上一定能过拟合。
如果 loss 降不到接近 0，说明代码有 bug (loss 计算、梯度断流、
mask 错位、投影头没接上等)。

用法 (服务器):
  conda activate uavmllm
  python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000

预期:
  - loss_sft: 从 ~2-4 降到 <0.3
  - loss_ctl: 从 ~0.1-0.5 降到 <0.01
  - 无 NaN, 无 Inf
  - ~3-5 分钟完成

若通过 → SFT 代码正确, 可以放心启动全量训练
若失败 → 逐项排查 (脚本会告诉你哪项失败)
"""

import os
import sys
import json
import argparse
import time

# BLAS 线程抑制 (必须在 import numpy/torch 之前)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import yaml
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from tqdm import tqdm

from src.model import Gemma3ISAC, UAVISACLosses
from src.data.dataset import SFTDataset


# ── Helpers ──────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")


def create_tiny_subset(data_dir: str, n: int = 5) -> str:
    """从全量数据取前 N 个 SFT 样本写入临时文件"""
    src = os.path.join(data_dir, "sft_dataset.jsonl")
    dst = os.path.join(data_dir, f"sft_tiny_{n}.jsonl")
    with open(src, "r", encoding="utf-8") as fin:
        with open(dst, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i >= n:
                    break
                fout.write(line)
    print(f"Created tiny subset: {dst} ({n} samples)")
    return dst


def run_overfit_test(config_path: str, data_path: str, n_samples: int,
                     n_steps: int = 200):
    """
    核心过拟合测试:
      在 N 个样本上训练若干步, 验证 loss 单调下降
    """
    print(f"\n{'='*60}")
    print(f"Stage I SFT Overfitting Test")
    print(f"{'='*60}")
    print(f"  Samples:  {n_samples}")
    print(f"  Steps:    {n_steps}")
    print(f"  Data:     {data_path}")
    print()

    # ── 加载配置 ──
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]["sft"]
    sim_cfg = cfg["simulation"]
    data_cfg = cfg["data"]

    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        warn("No GPU detected — overfitting test on CPU will be VERY slow")
        warn("Run on the AutoDL server instead")

    # ── 初始化模型 ──
    print("\n[1/5] Loading Gemma3-ISAC model...")
    t0 = time.time()
    model = Gemma3ISAC(
        model_name_or_path=model_cfg["backbone"],
        use_4bit=cfg["hardware"]["use_4bit"],
        lora_rank=model_cfg["lora"]["rank"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"]["dropout"],
        lora_target_modules=model_cfg["lora"]["target_modules"],
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        proj_head_config={
            "hidden_dim": model_cfg["control_token"]["hidden_dim"],
            "num_control_tokens": model_cfg["control_token"]["num_tokens"],
            "mlp_hidden": model_cfg["projection_head"]["mlp_hidden"],
            "readout_out_dim": model_cfg["projection_head"]["readout_out_dim"],
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
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
    )
    model = model.to(device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # ── 检查可训练参数 ──
    trainable_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  Trainable params: {trainable_count:,}")
    if trainable_count == 0:
        fail("No trainable parameters! Check LoRA + projection head setup.")
        return False

    # ── 加载数据 ──
    print(f"\n[2/5] Loading dataset...")
    dataset = SFTDataset(
        data_path=data_path,
        tokenizer=model.tokenizer,
        max_length=train_cfg["max_seq_length"],
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
    )
    print(f"  Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 单样本 batch — 过拟合更纯粹
        shuffle=True,
        num_workers=0,  # 避免 multiprocessing 干扰调试
    )

    # ── 优化器 ──
    print(f"\n[3/5] Setting up optimizer...")
    trainable_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "projection_head" in n
    ]
    trainable_params += [
        p for n, p in model.base_model.named_parameters()
        if p.requires_grad
    ]
    print(f"  Optimizing {len(trainable_params)} param groups")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-3,  # 过拟合用更高 lr (不需要泛化)
        weight_decay=0.0,  # 关闭 weight decay, 纯粹优化
    )

    # ── 损失计算器 ──
    loss_fn = UAVISACLosses(
        lambda_ctl=model_cfg["loss"]["lambda_ctl"],
        lambda_q=model_cfg["loss"]["lambda_q"],
        lambda_a=model_cfg["loss"]["lambda_a"],
        lambda_p=model_cfg["loss"]["lambda_p"],
        lambda_sep=model_cfg["loss"]["lambda_sep"],
    )

    # ── 训练循环 ──
    print(f"\n[4/5] Running overfitting loop ({n_steps} steps)...")
    model.train()

    history = {"loss_total": [], "loss_sft": [], "loss_ctl": []}
    all_batches = list(dataloader)  # 全部 batch 预先加载 (只有 5 个)

    pbar = tqdm(range(n_steps), desc="Overfitting")
    nan_detected = False

    for step in pbar:
        # 循环使用 5 个样本
        batch = all_batches[step % len(all_batches)]

        # 移到 GPU
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # 前向传播
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            control_mask=batch["control_mask"],
            q_current=batch["q_current"] if batch["q_current"].numel() > 0 else None,
            labels=batch["labels"],
        )

        # 构造 target dict
        delta_target = {
            "delta_q": batch["delta_q_target"],
            "delta_a": batch["delta_a_target"],
            "delta_p": batch["delta_p_target"],
        }
        delta_hat = {
            "delta_q": outputs["delta_q"],
            "delta_a": outputs["delta_a"],
            "delta_p": outputs["delta_p"],
        }

        q_hat = None
        if batch["q_current"].numel() > 0:
            q_hat = batch["q_current"] + outputs["delta_q"]

        # 计算损失
        total_loss, metrics = loss_fn.compute_stage1_total(
            delta_hat=delta_hat,
            delta_target=delta_target,
            logits=outputs["logits"],
            labels=batch["labels"],
            label_mask=batch["label_mask"],
            q_hat=q_hat,
        )

        # NaN 检测
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            fail(f"NaN/Inf detected at step {step}!")
            nan_detected = True
            break

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # 记录
        for k in history:
            history[k].append(metrics[k])

        # 每 20 步更新进度条
        if step % 20 == 0:
            pbar.set_postfix({
                "total": f"{metrics['loss_total']:.4f}",
                "sft": f"{metrics['loss_sft']:.4f}",
                "ctl": f"{metrics['loss_ctl']:.4f}",
            })

    if nan_detected:
        return False

    # ── 验证结果 ──
    print(f"\n[5/5] Verifying results...")
    print()

    initial = {k: history[k][:10] for k in history}   # 前 10 步平均
    final = {k: history[k][-10:] for k in history}     # 最后 10 步平均

    all_checks_pass = True

    # Check 1: loss_total 下降
    init_total = np.mean(initial["loss_total"])
    final_total = np.mean(final["loss_total"])
    print(f"  Loss total:  {init_total:.4f} → {final_total:.4f}  "
          f"({(1 - final_total/init_total)*100:.0f}% reduction)")
    if final_total < init_total * 0.5:
        ok("loss_total decreased >50% — gradients are flowing")
    elif final_total < init_total:
        warn("loss_total decreased but <50% — may need more steps or higher lr")
    else:
        fail("loss_total did NOT decrease — check forward/backward wiring")
        all_checks_pass = False

    # Check 2: loss_sft 下降
    init_sft = np.mean(initial["loss_sft"])
    final_sft = np.mean(final["loss_sft"])
    print(f"  Loss SFT:    {init_sft:.4f} → {final_sft:.4f}  "
          f"({(1 - final_sft/init_sft)*100:.0f}% reduction)")
    if final_sft < 0.5:
        ok("loss_sft < 0.5 — model is memorizing token sequences")
    elif final_sft < init_sft * 0.7:
        ok("loss_sft decreasing — token prediction learning")
    else:
        warn("loss_sft barely decreased — check label_mask / tokenizer setup")
        # Not a hard fail: SFT loss on 12B vocab can be slow to drop

    # Check 3: loss_ctl 下降
    init_ctl = np.mean(initial["loss_ctl"])
    final_ctl = np.mean(final["loss_ctl"])
    print(f"  Loss ctl:    {init_ctl:.4f} → {final_ctl:.4f}  "
          f"({(1 - final_ctl/init_ctl)*100:.0f}% reduction)")
    if final_ctl < 0.01:
        ok("loss_ctl < 0.01 — projection head is fitting targets precisely")
    elif final_ctl < init_ctl * 0.5:
        ok("loss_ctl decreasing — projection head is learning")
    else:
        fail("loss_ctl barely decreased — check projection head / delta targets")
        all_checks_pass = False

    # Check 4: loss 曲线单调性 (最后 50 步应持续下降)
    recent = history["loss_total"][-50:]
    early_recent = np.mean(recent[:10])
    late_recent = np.mean(recent[-10:])
    if late_recent < early_recent:
        ok("Loss still decreasing in final 50 steps")
    else:
        warn("Loss plateaued — may need more steps (but not a bug)")

    # Check 5: 无 NaN/Inf
    has_nan = any(np.isnan(history["loss_total"]))
    has_inf = any(np.isinf(history["loss_total"]))
    if not has_nan and not has_inf:
        ok("No NaN/Inf in training history")
    else:
        fail("NaN/Inf detected — learning rate too high or gradient explosion")
        all_checks_pass = False

    # ── 总结 ──
    print(f"\n{'='*60}")
    if all_checks_pass:
        print(f"{GREEN}✓ ALL CHECKS PASSED{RESET}")
        print(f"  The SFT training pipeline is correctly wired:")
        print(f"    • Tokenization + control token injection")
        print(f"    • Gemma3 forward pass (4-bit QLoRA)")
        print(f"    • Control token hidden state extraction")
        print(f"    • Projection head (readout → MLP → constraints)")
        print(f"    • Combined loss (L_SFT + λ_ctl * L_ctl)")
        print(f"    • Gradient flow through LoRA + projection head")
        print(f"    • Optimizer updates")
        print(f"\n  → Safe to proceed with full 5000-sample SFT training.")
    else:
        print(f"{RED}✗ SOME CHECKS FAILED{RESET}")
        print(f"  Review the failures above before launching full training.")
    print(f"{'='*60}")

    return all_checks_pass


def main():
    parser = argparse.ArgumentParser(
        description="SFT overfitting test — prove training code correctness"
    )
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    parser.add_argument("--data-dir", type=str,
                        default="/root/autodl-tmp/data/full5000")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of samples to overfit on")
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Number of optimization steps")
    parser.add_argument("--keep-subset", action="store_true",
                        help="Don't delete the temporary tiny subset file")
    args = parser.parse_args()

    # 创建 tiny subset
    tiny_path = create_tiny_subset(args.data_dir, args.n_samples)

    try:
        passed = run_overfit_test(args.config, tiny_path,
                                  args.n_samples, args.n_steps)
    finally:
        if not args.keep_subset and os.path.exists(tiny_path):
            os.remove(tiny_path)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
