"""
Stage I: SFT-LoRA 训练
论文 Section 4.2

L_I = L_SFT + λ_ctl * L_ctl

训练配置 (论文值):
  - LoRA rank r=16, α=32
  - S=5000 环境样本
  - 3 epochs
  - lr=2e-4, cosine scheduler
  - 有效 batch ≈ 16 (bs=1 × grad_accum=16)

硬件: RTX 5090 32GB AutoDL
  - 4-bit QLoRA: 模型占用 ~8-10GB
  - 训练峰值显存: ~25-30GB
"""

import os
import sys

# ⚠️ 必须在 import numpy / torch 之前！
# 防止 Intel MKL / OpenBLAS 与 PyTorch DataLoader 多进程打架
# 每个 worker 都试图开满全部核心 → CPU 100% 但进度卡死
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ── 【防爆盾 1】核弹级环境变量 ──
# Blackwell sm_120: 禁止 Inductor 使用 FlexAttention (共享内存 101KB < 需 114KB)
os.environ["TORCHINDUCTOR_FLEX_ATTENTION"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ── 【防爆盾 2】Unsloth 强插队 ──
# 必须在 torch / transformers 之前导入, 确保底层 Triton 补丁 100% 打上!
import unsloth

import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── 【防爆盾 3】代码级物理超度 FlexAttention ──
import torch._inductor.config as inductor_config
if hasattr(inductor_config, "flex_attention"):
    inductor_config.flex_attention = False
if hasattr(inductor_config, "use_flex_attention"):
    inductor_config.use_flex_attention = False

from transformers import (
    get_cosine_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator
from tqdm import tqdm
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import Gemma3ISAC, UAVISACLosses
from src.data.dataset import SFTDataset


# ================================================================
# Training Loop
# ================================================================

def train_stage1(config_path: str, data_dir: Optional[str] = None):
    """
    Stage I SFT-LoRA 主训练函数
    """

    # ---- 加载配置 ----
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]["sft"]
    sim_cfg = cfg["simulation"]
    data_cfg = cfg["data"]
    output_cfg = cfg

    set_seed(cfg["training"]["seed"])

    # ---- Accelerator ----
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        mixed_precision="bf16",
        log_with="tensorboard",
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ---- 初始化模型 ----
    logger.info("Loading Gemma3-ISAC model...")
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

    # ---- 加载数据集 ----
    sft_file = data_dir or os.path.join(data_cfg["output_dir"], data_cfg["sft_file"])
    logger.info(f"Loading SFT dataset from {sft_file}...")
    dataset = SFTDataset(
        data_path=sft_file,
        tokenizer=model.tokenizer,
        max_length=train_cfg["max_seq_length"],
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ---- 优化器 (分层学习率) ----
    # 投影头从零训练 → 需要较大 LR；LoRA 微调预训练权重 → 用小 LR
    proj_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "projection_head" in n
    ]
    lora_params = [
        p for n, p in model.base_model.named_parameters()
        if p.requires_grad
    ]

    base_lr = train_cfg["learning_rate"]
    optimizer = torch.optim.AdamW(
        [
            {"params": proj_params, "lr": 1e-3},
            {"params": lora_params, "lr": base_lr},
        ],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # ---- 学习率调度器 ----
    total_steps = (
        len(dataloader)
        * train_cfg["epochs"]
        // train_cfg["gradient_accumulation_steps"]
    )
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- Accelerator 准备 ----
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # ---- 损失计算器 ----
    loss_fn = UAVISACLosses(
        lambda_ctl=model_cfg["loss"]["lambda_ctl"],
        lambda_q=model_cfg["loss"]["lambda_q"],
        lambda_a=model_cfg["loss"]["lambda_a"],
        lambda_p=model_cfg["loss"]["lambda_p"],
        lambda_sep=model_cfg["loss"]["lambda_sep"],
    )

    # ---- 训练循环 ----
    output_dir = output_cfg.get("output_dir", "./outputs")
    checkpoint_dir = output_cfg.get("checkpoint_dir", "./checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    global_step = 0
    model.train()

    for epoch in range(train_cfg["epochs"]):
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")

        for batch in progress:
            with accelerator.accumulate(model):
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

                # 构造 hat dict
                delta_hat = {
                    "delta_q": outputs["delta_q"],
                    "delta_a": outputs["delta_a"],
                    "delta_p": outputs["delta_p"],
                }

                # 计算损失
                q_hat = None
                if batch["q_current"].numel() > 0:
                    q_hat = batch["q_current"] + outputs["delta_q"]

                total_loss, metrics = loss_fn.compute_stage1_total(
                    delta_hat=delta_hat,
                    delta_target=delta_target,
                    logits=outputs["logits"],
                    labels=batch["labels"],
                    label_mask=batch["label_mask"],
                    q_hat=q_hat,
                )

                # 反向传播
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        cfg["hardware"]["max_grad_norm"],
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # 仅在真正执行梯度同步 (optimizer step) 后才推进 global_step / scheduler / zero_grad
            # 防止 grad_accum=16 时每个 micro-batch:
            #   - scheduler.step() 被调 16 次 → LR 衰减 16 倍过快
            #   - zero_grad() 清空累积梯度 → 有效 batch=1 (非 16)
            #   - global_step 被 +16 次 → 疯狂写 checkpoint 撑爆硬盘
            if accelerator.sync_gradients:
                global_step += 1

                # 日志
                if global_step % train_cfg["logging_steps"] == 0:
                    progress.set_postfix(metrics)
                    accelerator.log(metrics, step=global_step)

                # 保存 checkpoint
                if global_step % train_cfg["save_steps"] == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"stage1_step_{global_step}")
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(ckpt_path)
                    logger.info(f"Checkpoint saved to {ckpt_path}")

    # 最终保存
    final_path = os.path.join(output_dir, "stage1_sft_final")
    accelerator.unwrap_model(model).save_pretrained(final_path)
    logger.info(f"Stage I complete! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    train_stage1(args.config, args.data_dir)
