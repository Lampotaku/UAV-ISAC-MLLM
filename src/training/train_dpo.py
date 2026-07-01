"""
Stage II: DPO 偏好优化
论文 Section 4.3

L_II = L_DPO + μ * L_SFT + λ_ctl * L_ctl + λ_sep * L_sep

训练配置 (论文):
  - DPO β = 0.1
  - SFT anchor μ = 0.05 (防遗忘)
  - lr = 5e-5 (低于 SFT)
  - 从 Stage I checkpoint 热启动

硬件: RTX PRO 6000 96GB AutoDL
  - 需要同时加载 reference model (冻结)
  - bf16 双模型 (~48GB) + 4×logits bf16 (~4GB bs=1) + log_softmax fp32 (~4GB, _grad_ckpt 避免 forward 存储) + 激活 ≈ 65-75GB (bs=1, 96GB 安全)
  - 注意: Unsloth fast_cross_entropy_loss 只对 SFT 的 scalar CE 有分块内核, DPO 需要 per-token log-prob 故仍用 _grad_ckpt
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

# PyTorch CUDA 内存分配器: 允许动态释放缓存段, 减少碎片化 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── 【防爆盾 0】网络/遥测静默 ──
# 0a: 禁止 Unsloth 连接 HuggingFace 上报统计 (国内超时 120s)
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
# 0b: HuggingFace 镜像 (国内加速, 不影响本地模型加载)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ── 【防爆盾 1】核弹级环境变量 ──
# Blackwell sm_120: 禁止 Inductor 使用 FlexAttention (共享内存 101KB < 需 114KB)
os.environ["TORCHINDUCTOR_FLEX_ATTENTION"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ── 【防爆盾 2】项目已彻底肃清 Unsloth ──
# Unsloth 即使局部导入也会触发全局 monkey-patch → CheckpointError.
# DPO 用纯 PyTorch _grad_ckpt, 完全不依赖 Unsloth.

import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.checkpoint import checkpoint as _grad_ckpt
from torch.utils.data import Dataset, DataLoader

# ── 【防爆盾 3】代码级物理超度 FlexAttention ──
import torch._inductor.config as inductor_config
if hasattr(inductor_config, "flex_attention"):
    inductor_config.flex_attention = False
if hasattr(inductor_config, "use_flex_attention"):
    inductor_config.use_flex_attention = False

from transformers import get_cosine_schedule_with_warmup, set_seed
from accelerate import Accelerator
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import Gemma3ISAC, UAVISACLosses
from src.data.dataset import DPODataset


# ================================================================
# Helper: compute per-sample log-probability from logits
# ================================================================

def _logp_gather(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    log_softmax + gather at target indices — thin wrapper for _grad_ckpt.

    必须定义在模块级别, _grad_ckpt(use_reentrant=False) 需要 pickle-able 的函数引用.
    避免 F.log_softmax 在 256K 类上内部 upcast 到 fp32 后保存输出 (DPO bs=1 → ~4 GB;
    SFT bs=4 → ~16 GB, 但 SFT 走 Unsloth Chunked CE, 不经过此函数).

    注意: Unsloth fast_cross_entropy_loss 返回标量 CE, 无法用于 per-token log-prob.
    因此 DPO 仍用 _grad_ckpt, 4 GB 峰值对 96 GB GPU 安全.
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    return log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)


def _compute_logprob(
    logits: torch.Tensor,   # (B, seq_len, vocab_size)
    labels: torch.Tensor,   # (B, seq_len)  — prompt 部分填 -100
    label_mask: torch.Tensor,  # (B, seq_len)  — 1.0 on response tokens
) -> torch.Tensor:
    """
    计算每条样本在 response token 上的平均 log-probability

    返回 (B,) tensor, 每个元素是 log π(response | prompt)

    内存优化 (两层):
      1. transpose(1,2) 替代 .contiguous() — 避免 logits 二次拷贝 (~2 GB per tensor)
      2. _grad_ckpt 包装 log_softmax — 避免内部 fp32 输出存储 (DPO bs=1 → ~4 GB;
         Unsloth fast_cross_entropy_loss 不可用 → 返回标量 CE 而非 per-token log-prob)
    """
    # 右移: predict next token (全部是 view, 不拷贝 ~2 GB per tensor)
    shift_logits = logits[:, :-1, :].transpose(1, 2)   # (B, V, S-1)
    shift_labels = labels[:, 1:]                         # (B, S-1)
    shift_mask = label_mask[:, 1:]                       # (B, S-1)
    safe_labels = shift_labels.masked_fill(shift_labels < 0, 0)

    # Gradient-checkpoint the log_softmax: 内部 fp32 输出不在 forward 时存储,
    # 改为 backward 时重算 — 峰值显存下降 ~4-16 GB (取决于 bs).
    per_token_logp = _grad_ckpt(
        _logp_gather, shift_logits, safe_labels,
        use_reentrant=False,
    )  # → (B, S-1)

    masked = per_token_logp * shift_mask
    seq_logp = masked.sum(dim=-1)  # SUM not mean — DPO needs joint log-prob Σ_t log π(y_t|...)
    return seq_logp


# ================================================================
# DPO Training Loop
# ================================================================

def train_stage2(
    config_path: str,
    stage1_ckpt: str,
    data_dir: Optional[str] = None,
):
    """
    Stage II DPO 主训练函数

    需要:
      1. Stage I checkpoint (作为初始模型)
      2. 冻结 reference model (从 Stage I 或 base model)
    """

    # ---- 加载配置 ----
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]["dpo"]
    sim_cfg = cfg["simulation"]
    data_cfg = cfg["data"]
    output_cfg = cfg

    set_seed(cfg["training"]["seed"])

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=cfg.get("log_dir", "/root/autodl-tmp/logs"),
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ---- 加载 Stage I 模型 (可训练) ----
    logger.info(f"Loading Stage I checkpoint from {stage1_ckpt}...")
    model = Gemma3ISAC.from_pretrained(
        load_dir=stage1_ckpt,
        base_model_name=model_cfg["backbone"],
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
        },
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
    )

    # ---- Reference Model (冻结, 不更新) ----
    # 显式从 Stage I 重新加载，避免 4-bit 量化模型的 deepcopy 显存崩溃风险
    logger.info("Creating reference model (frozen) by reloading...")
    ref_model = Gemma3ISAC.from_pretrained(
        load_dir=stage1_ckpt,
        base_model_name=model_cfg["backbone"],
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
        },
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # ---- 加载 DPO 数据集 ----
    dpo_file = os.path.join(data_dir, data_cfg["dpo_file"]) if data_dir else os.path.join(data_cfg["output_dir"], data_cfg["dpo_file"])
    sft_file = os.path.join(data_dir, data_cfg["sft_file"]) if data_dir else os.path.join(data_cfg["output_dir"], data_cfg["sft_file"])

    logger.info(f"Loading DPO dataset from {dpo_file}...")
    dpo_dataset = DPODataset(
        data_path=dpo_file,
        tokenizer=model.tokenizer,
        max_length=train_cfg["max_seq_length"],
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
    )

    dpo_dataloader = DataLoader(
        dpo_dataset,
        batch_size=train_cfg["per_device_batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # ---- 优化器 (分层学习率) ----
    # Stage II: 投影头已从 Stage I 预训练 → 用较小 LR 微调
    # LoRA 继续用 DPO LR 微调语言偏好
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
            {"params": proj_params, "lr": base_lr},
            {"params": lora_params, "lr": base_lr},
        ],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # ---- 学习率调度 ----
    total_steps = (
        len(dpo_dataloader)
        * train_cfg["epochs"]
        // train_cfg["gradient_accumulation_steps"]
    )
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- Accelerator ----
    model, ref_model, optimizer, dpo_dataloader, scheduler = accelerator.prepare(
        model, ref_model, optimizer, dpo_dataloader, scheduler
    )

    # ---- Loss ----
    loss_fn = UAVISACLosses(
        lambda_ctl=model_cfg["loss"]["lambda_ctl"],
        lambda_q=model_cfg["loss"]["lambda_q"],
        lambda_a=model_cfg["loss"]["lambda_a"],
        lambda_p=model_cfg["loss"]["lambda_p"],
        lambda_sep=model_cfg["loss"]["lambda_sep"],
        dpo_beta=train_cfg["beta"],
        sft_anchor_mu=train_cfg.get("mu", 0.05),
    )

    # ---- 训练 ----
    output_dir = output_cfg.get("output_dir", "./outputs")
    checkpoint_dir = output_cfg.get("checkpoint_dir", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 激活 TensorBoard 写入器 — 没有这行 accelerator.log() 全部静默丢弃！
    accelerator.init_trackers("stage2_dpo")

    global_step = 0
    model.train()

    # ---- LoRA 梯度诊断 (仅观测, 不影响训练) ----
    lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and ('lora_A' in n or 'lora_B' in n)
    ]
    if lora_params:
        logger.info(f"Grad norm diag: {len(lora_params)} LoRA tensors")

    for epoch in range(train_cfg["epochs"]):
        total_batches = len(dpo_dataloader)
        logger.info(f"=== DPO Epoch {epoch+1}/{train_cfg['epochs']}  ({total_batches} batches) ===")

        progress = tqdm(dpo_dataloader, desc=f"DPO E{epoch+1}", unit="batch", ncols=100)
        for batch_idx, batch in enumerate(progress):
            with accelerator.accumulate(model):
                # === Chosen 前向传播 ===
                outputs_chosen = model(
                    input_ids=batch["input_ids_chosen"],
                    attention_mask=batch["attention_mask_chosen"],
                    control_mask=batch["control_mask_chosen"],
                    q_current=batch["q_current"] if batch.get("q_current") is not None and batch["q_current"].numel() > 0 else None,
                    labels=batch["labels_chosen"],
                )

                # === Rejected 前向传播 ===
                outputs_rejected = model(
                    input_ids=batch["input_ids_rejected"],
                    attention_mask=batch["attention_mask_rejected"],
                    control_mask=batch["control_mask_rejected"],
                    q_current=batch["q_current"] if batch.get("q_current") is not None and batch["q_current"].numel() > 0 else None,
                    labels=batch["labels_rejected"],
                )

                # === Reference model log-probs (frozen, no grad) ===
                with torch.no_grad():
                    ref_out_chosen = ref_model(
                        input_ids=batch["input_ids_chosen"],
                        attention_mask=batch["attention_mask_chosen"],
                        control_mask=batch["control_mask_chosen"],
                        q_current=batch["q_current"] if batch.get("q_current") is not None and batch["q_current"].numel() > 0 else None,
                        labels=batch["labels_chosen"],
                    )
                    ref_out_rejected = ref_model(
                        input_ids=batch["input_ids_rejected"],
                        attention_mask=batch["attention_mask_rejected"],
                        control_mask=batch["control_mask_rejected"],
                        q_current=batch["q_current"] if batch.get("q_current") is not None and batch["q_current"].numel() > 0 else None,
                        labels=batch["labels_rejected"],
                    )

                # === Compute per-sample log-probabilities (response tokens only) ===
                logp_chosen = _compute_logprob(
                    outputs_chosen["logits"],
                    batch["labels_chosen"],
                    batch["label_mask_chosen"],
                )
                logp_rejected = _compute_logprob(
                    outputs_rejected["logits"],
                    batch["labels_rejected"],
                    batch["label_mask_rejected"],
                )
                logp_ref_chosen = _compute_logprob(
                    ref_out_chosen["logits"],
                    batch["labels_chosen"],
                    batch["label_mask_chosen"],
                )
                logp_ref_rejected = _compute_logprob(
                    ref_out_rejected["logits"],
                    batch["labels_rejected"],
                    batch["label_mask_rejected"],
                )

                # === Control loss targets (from winner oracle) ===
                delta_target = {
                    "delta_q": batch.get("delta_q_target"),
                    "delta_a": batch.get("delta_a_target"),
                    "delta_p": batch.get("delta_p_target"),
                }
                delta_hat = {
                    "delta_q": outputs_chosen["delta_q"],
                    "delta_a": outputs_chosen["delta_a"],
                    "delta_p": outputs_chosen["delta_p"],
                }

                # === Total Stage II loss ===
                # L = L_DPO + μ*L_SFT + λ_ctl*L_ctl + λ_sep*L_sep
                if delta_target["delta_q"] is not None:
                    q_hat = None
                    if batch.get("q_current") is not None and batch["q_current"].numel() > 0:
                        q_hat = batch["q_current"] + outputs_chosen["delta_q"]
                    total_loss, metrics = loss_fn.compute_stage2_total(
                        delta_hat=delta_hat,
                        delta_target=delta_target,
                        logp_chosen=logp_chosen,
                        logp_rejected=logp_rejected,
                        logp_ref_chosen=logp_ref_chosen,
                        logp_ref_rejected=logp_ref_rejected,
                        logits=outputs_chosen["logits"],
                        labels=batch["labels_chosen"],
                        label_mask=batch["label_mask_chosen"],
                        q_hat=q_hat,
                    )
                else:
                    # No oracle targets in data — skip control loss
                    total_loss, metrics = loss_fn.compute_stage2_total(
                        delta_hat=delta_hat,
                        delta_target=delta_hat,  # self-target (L_ctl=0)
                        logp_chosen=logp_chosen,
                        logp_rejected=logp_rejected,
                        logp_ref_chosen=logp_ref_chosen,
                        logp_ref_rejected=logp_ref_rejected,
                    )

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    # ---- LoRA 总梯度 norm (免费: 梯度已累积完毕) ----
                    if lora_params:
                        _gn_total = sum(
                            p.grad.data.norm().item() ** 2
                            for p in lora_params if p.grad is not None
                        ) ** 0.5
                        metrics["grad_norm_lora_total"] = _gn_total

                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        cfg["hardware"]["max_grad_norm"],
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # 仅在真正执行梯度同步后才推进 scheduler / global_step / log / save
            # 防止 grad_accum=16 时每个 micro-batch:
            #   - scheduler.step() 被调 16 次 → LR 衰减 16 倍过快
            #   - zero_grad() 清空累积梯度 → 有效 batch=1 (非 16)
            #   - global_step 被 +16 次 → 疯狂写 checkpoint 撑爆硬盘
            if accelerator.sync_gradients:
                global_step += 1

                # 每个 step 逐行打印 metrics (可复制) + tqdm 动态进度条保留
                parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
                tqdm.write(f"[Step {global_step:5d}]  {', '.join(parts)}")
                accelerator.log(metrics, step=global_step)
                # 同步更新进度条 postfix
                short = {k: f"{v:.3f}" if isinstance(v, float) else v
                         for k, v in metrics.items()
                         if k in ("loss_dpo", "loss_ctl")}
                progress.set_postfix(short)

                if global_step % train_cfg["save_steps"] == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"stage2_step_{global_step}")
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(ckpt_path)
                    logger.info(f"Checkpoint saved to {ckpt_path}")

    # 最终保存
    final_path = os.path.join(output_dir, "stage2_dpo_final")
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(final_path)
    logger.info(f"Stage II complete! Model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stage1_ckpt", type=str, required=True,
                        help="Path to Stage I checkpoint")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    train_stage2(args.config, args.stage1_ckpt, args.data_dir)
