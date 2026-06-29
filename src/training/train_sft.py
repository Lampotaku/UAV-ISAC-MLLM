"""
Stage I: SFT-LoRA 训练
论文 Section 4.2

L_I = L_SFT + λ_ctl * L_ctl

训练配置 (论文值):
  - LoRA rank r=16, α=32
  - S=5000 环境样本
  - 3 epochs
  - lr=2e-4, cosine scheduler
  - 有效 batch = 16 (bs=2 × grad_accum=8)
  - 5000/bs=2 = 2500 micro-batches/epoch, ~313 optimizer steps/epoch

硬件: RTX PRO 6000 96GB AutoDL
  - bf16 全精度 LoRA: 模型占用 ~24GB
  - modules_to_save (embed_tokens): AdamW 状态 ~8GB
  - 前向: logits bf16 ~3.5GB (bs=2×3456×256K) + last_hidden_state ~128MB
  - 反向: grad_logits ~3.5GB (bf16) + grad_embed ~2GB + 激活梯度 ~5GB
  - CE 损失: 纯 PyTorch F.cross_entropy, bs=2 时 fp32 中间约 ~7GB
  - 实测峰值: ~78GB (bs=2, seq=3456 or 4096, grad_accum=8, ~20GB 余量)

  实测速度 (server RTX PRO 6000):
  - bs=2/seq=4096: ~4.1s/micro-batch, 2500 steps/epoch → ~2.9h/epoch, ~8.7h total
  - bs=2/seq=3456 (预期): ~3.3s/micro-batch, ~2.3h/epoch, ~7h total
  - bs=1/seq=4096 (旧): ~2.5s/micro-batch, 5000 steps/epoch → ~3.5h/epoch
  - bs=2 每步更慢但步数减半 → epoch 吞吐 +18%, 全训练省 ~2h
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
# Unsloth 即使局部导入 (from unsloth.kernels...) 也会触发全局 monkey-patch,
# 导致 grad checkpoint 在 forward/recompute 间张量数不一致 (68≠65) → CheckpointError.
# 现在全项目 0 处 unsloth 引用 — 纯 PyTorch F.cross_entropy + bs=1/grad_accum=16.
# 模型加载走原生 HF + SDPA → 2-3s/step.

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
# Phase 1 Sensitivity Check
# ================================================================

def _check_cross_env_sensitivity(
    model,
    prompt_a: str, q_a: torch.Tensor,
    prompt_b: str, q_b: torch.Tensor,
) -> float:
    """
    跨环境 sensitivity: 两个不同 UAV 位置的环境，模型输出是否不同。

    返回 L2 ratio: ||Δ_q(env_b) - Δ_q(env_a)|| / ||Δ_q(env_a)||
    > 0.1 → 模型学到了环境特定的控制表示，可切换 Phase 2。
    ≈ 0  → 模型对所有环境输出相同 delta，尚未学到几何编码。

    两个 env 使用固定的确定性 seed (42 和 43)，保证每次检查可复现。
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        ws_a = model.generate_warmstart(prompt_a, q_current=q_a)
        ws_b = model.generate_warmstart(prompt_b, q_current=q_b)

        d_a = ws_a["delta_q"]
        d_b = ws_b["delta_q"]
        ratio = (d_b - d_a).norm().item() / (d_a.norm().item() + 1e-8)

    if was_training:
        model.train()

    return ratio


# ================================================================
# Training Loop
# ================================================================

def train_stage1(config_path: str, data_dir: Optional[str] = None, resume_from: Optional[str] = None):
    """
    Stage I SFT-LoRA 主训练函数

    Args:
        config_path: yaml 配置文件路径
        data_dir: 数据目录 (覆盖 config 中的路径)
        resume_from: Phase 1 checkpoint 路径 → 跳过 Phase 1, 直接从 Phase 2 开始
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
        project_dir=cfg.get("log_dir", "/root/autodl-tmp/logs"),
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ---- Proj head config (shared by init / resume paths) ----
    _proj_head_cfg = {
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
    }

    # ---- 初始化模型 (fresh / resume) ----
    if resume_from:
        logger.info(f"Resuming from Phase 1 checkpoint: {resume_from}")
        logger.info("  → Skipping Phase 1, starting Phase 2 directly")
        model = Gemma3ISAC.from_pretrained(
            load_dir=resume_from,
            base_model_name=model_cfg["backbone"],
            use_4bit=cfg["hardware"]["use_4bit"],
            lora_rank=model_cfg["lora"]["rank"],
            lora_alpha=model_cfg["lora"]["alpha"],
            lora_dropout=model_cfg["lora"]["dropout"],
            lora_target_modules=model_cfg["lora"]["target_modules"],
            num_control_tokens=model_cfg["control_token"]["num_tokens"],
            torch_dtype=torch.bfloat16,
            attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
            proj_head_config=_proj_head_cfg,
        )
        model = model.to("cuda")
        model.train()
        # ══ OOM #6 防御: gc + lm_head 冻结 ══
        model.base_model.gradient_checkpointing_enable()
        try:
            model.base_model.model.model.gradient_checkpointing_enable()
        except Exception:
            pass
        # gc 验证 + 硬加固 (直接设底层 Gemma3Model 属性)
        transformer = model.base_model.model.model  # Gemma3Model
        gc_ok = getattr(transformer, 'gradient_checkpointing', False)
        if not gc_ok:
            logger.warning(
                "GC NOT enabled on Gemma3Model after all enable() calls — "
                "forcing via direct attr. Without GC activations consume ~60 GB → OOM."
            )
            transformer.gradient_checkpointing = True
            if not hasattr(transformer, '_gradient_checkpointing_func') or \
               transformer._gradient_checkpointing_func is None:
                transformer._gradient_checkpointing_func = \
                    torch.utils.checkpoint.checkpoint
            gc_ok = getattr(transformer, 'gradient_checkpointing', False)
        if not gc_ok:
            logger.critical(
                "FATAL: Cannot enable gradient checkpointing on Gemma3Model. "
                "Activations ~60 GB → OOM at ~94 GB. "
                "Fallback: reduce per_device_batch_size to 1, set grad_accum to 16."
            )
        # lm_head 冻结: 检查权重绑定状态, 若仍绑定则 clone 解绑
        causal_lm = model.base_model.model  # PeftModel → Gemma3ForCausalLM
        lm_head = causal_lm.lm_head
        embed = causal_lm.get_input_embeddings()
        if lm_head.weight.data_ptr() == embed.weight.data_ptr():
            # 权重仍绑定 → clone 解绑 + 冻结
            lm_head._parameters['weight'] = torch.nn.Parameter(
                lm_head.weight.data.clone(), requires_grad=False
            )
        else:
            lm_head.weight.requires_grad = False
        tied_ok = (
            causal_lm.lm_head.weight.data_ptr()
            == causal_lm.get_input_embeddings().weight.data_ptr()
        )
        logger.info(
            f"OOM6 guards: gc={'✓' if gc_ok else '✗ FAILED'}, "
            f"tied={'✓' if tied_ok else '✗ (untied to protect embed_tokens)'}, "
            f"lm_head_grad={causal_lm.lm_head.weight.requires_grad}"
        )
        # Force-skip Phase 1: 模型已完成 CTL-only 预训练
        train_cfg["phase1"] = {"enabled": False}
    else:
        logger.info("Loading Gemma3-ISAC model...")
        model = Gemma3ISAC(
            model_name_or_path=model_cfg["backbone"],
            use_4bit=cfg["hardware"]["use_4bit"],
            lora_rank=model_cfg["lora"]["rank"],
            lora_alpha=model_cfg["lora"]["alpha"],
            lora_dropout=model_cfg["lora"]["dropout"],
            lora_target_modules=model_cfg["lora"]["target_modules"],
            num_control_tokens=model_cfg["control_token"]["num_tokens"],
            proj_head_config=_proj_head_cfg,
            attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        )

    # ---- 加载数据集 ----
    sft_file = os.path.join(data_dir, data_cfg["sft_file"]) if data_dir else os.path.join(data_cfg["output_dir"], data_cfg["sft_file"])
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

    # 激活 TensorBoard 写入器 — 没有这行 accelerator.log() 全部静默丢弃！
    accelerator.init_trackers("stage1_sft")

    # ---- Phase 1: CTL-only warmup (控制表示预训练) ----
    # 完全关闭 CE loss，强制 LoRA 学会将环境信息编码到 control token hidden states。
    # 每隔 N 步检查 perturbation sensitivity — 达标后自动切换到 Phase 2。
    phase1_cfg = train_cfg.get("phase1", {})
    phase1_enabled = phase1_cfg.get("enabled", False)
    phase1_sensitivity = None

    if phase1_enabled:
        phase1_max_steps = phase1_cfg.get("max_steps", 400)
        phase1_check_interval = phase1_cfg.get("sensitivity_check_steps", 50)
        phase1_threshold = phase1_cfg.get("sensitivity_threshold", 0.1)
        phase1_lambda_ctl = phase1_cfg.get("lambda_ctl", 1.0)

        logger.info("=" * 60)
        logger.info(
            f"Phase 1: CTL-only warmup — max {phase1_max_steps} steps, "
            f"sensitivity check every {phase1_check_interval} steps, "
            f"threshold = {phase1_threshold}"
        )
        logger.info("=" * 60)

        # 生成两个固定的 sensitivity 测试环境 (seed=42/43, 可复现)
        # 用不同 seed 得到不同 UAV 位置，检查模型是否对不同环境输出不同 delta
        from src.env import ISACScenarioGenerator
        from src.data.prompt_builder import build_full_prompt

        _sens_gen = ISACScenarioGenerator(
            num_uavs=sim_cfg["num_uavs"],
            num_users=sim_cfg["num_users"],
            num_targets=sim_cfg["num_targets"],
            area_size=tuple(sim_cfg["area_size"]),
            carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
            bandwidth_mhz=sim_cfg["bandwidth_mhz"],
            num_antennas=sim_cfg["num_antennas_tx"],
            p_max_dbm=sim_cfg["p_max_dbm"],
            seed=42,
        )
        _sens_env_a = _sens_gen.sample(42)
        _sens_prompt_a = build_full_prompt(_sens_env_a, sim_cfg)
        _sens_q_a = torch.tensor(_sens_env_a.q_current, dtype=torch.float32, device="cuda")

        _sens_env_b = _sens_gen.sample(43)
        _sens_prompt_b = build_full_prompt(_sens_env_b, sim_cfg)
        _sens_q_b = torch.tensor(_sens_env_b.q_current, dtype=torch.float32, device="cuda")

        # Phase 1 使用更高 LoRA LR (纯 regression, 无 CE 噪声, 允许激进更新)
        phase1_lr_lora = phase1_cfg.get("lr_lora", 1e-3)
        optimizer.param_groups[1]["lr"] = phase1_lr_lora
        logger.info(f"Phase 1 LoRA LR: {phase1_lr_lora} (Phase 2 will use {base_lr})")

        phase1_step = 0
        model.train()
        phase1_pbar = tqdm(total=phase1_max_steps, desc="Phase 1 (CTL-only)")
        phase1_iter = iter(dataloader)

        while phase1_step < phase1_max_steps:
            # 数据耗尽时重新 shuffle
            try:
                batch = next(phase1_iter)
            except StopIteration:
                phase1_iter = iter(dataloader)
                batch = next(phase1_iter)

            with accelerator.accumulate(model):
                # 前向传播 (logits 也在计算，但 Phase 1 不使用)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    control_mask=batch["control_mask"],
                    q_current=batch["q_current"] if batch["q_current"].numel() > 0 else None,
                    labels=batch["labels"],
                )

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

                total_loss, metrics = loss_fn.compute_phase1_total(
                    delta_hat=delta_hat,
                    delta_target=delta_target,
                    phase1_lambda_ctl=phase1_lambda_ctl,
                )

                accelerator.backward(total_loss)

            if accelerator.sync_gradients:
                phase1_step += 1
                phase1_pbar.update(1)

                accelerator.clip_grad_norm_(
                    model.parameters(),
                    cfg["hardware"]["max_grad_norm"],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 定期检查 perturbation sensitivity
                if phase1_step % phase1_check_interval == 0:
                    _raw_model = accelerator.unwrap_model(model)
                    phase1_sensitivity = _check_cross_env_sensitivity(
                        _raw_model, _sens_prompt_a, _sens_q_a, _sens_prompt_b, _sens_q_b
                    )
                    phase1_pbar.write(
                        f"step {phase1_step}: loss_ctl={metrics['loss_ctl']:.2f}, "
                        f"sens={phase1_sensitivity:.4f}"
                    )
                    accelerator.log({
                        "phase1/loss_ctl": metrics["loss_ctl"],
                        "phase1/sensitivity": phase1_sensitivity,
                    }, step=phase1_step)

                    if phase1_sensitivity > phase1_threshold:
                        logger.info(
                            f"Phase 1 complete at step {phase1_step}: "
                            f"sensitivity {phase1_sensitivity:.4f} > threshold {phase1_threshold}"
                        )
                        break

                # 每 50 步记录一次 (即使未到 check_interval)
                if phase1_step % 50 == 0 and phase1_step % phase1_check_interval != 0:
                    accelerator.log({"phase1/loss_ctl": metrics["loss_ctl"]}, step=phase1_step)

                # 保存 Phase 1 checkpoint (同 Phase 2 save_steps 节奏)
                if phase1_step % train_cfg["save_steps"] == 0 or phase1_step == phase1_max_steps:
                    ckpt_path = os.path.join(checkpoint_dir, f"phase1_step_{phase1_step}")
                    accelerator.unwrap_model(model).save_pretrained(ckpt_path)
                    logger.info(f"Phase 1 checkpoint saved to {ckpt_path} (sens={phase1_sensitivity:.4f})")

        phase1_pbar.close()

        # Phase 1 结束时保存最终 checkpoint (如果最后一步还没存过)
        if phase1_step % train_cfg["save_steps"] != 0 and phase1_step < phase1_max_steps:
            ckpt_path = os.path.join(checkpoint_dir, f"phase1_step_{phase1_step}")
            accelerator.unwrap_model(model).save_pretrained(ckpt_path)
            logger.info(f"Phase 1 final checkpoint saved to {ckpt_path} (sens={phase1_sensitivity:.4f})")

        if phase1_step >= phase1_max_steps:
            logger.warning(
                f"Phase 1 reached max_steps {phase1_max_steps} without hitting "
                f"sensitivity threshold {phase1_threshold}. "
                f"Final sensitivity: {phase1_sensitivity}. Proceeding to Phase 2 anyway."
            )
        else:
            logger.info(
                f"Phase 1 auto-switch to Phase 2 at step {phase1_step}, "
                f"sensitivity={phase1_sensitivity:.4f}"
            )

    # ---- Phase 2 / Main: Joint SFT + CTL ----
    # 恢复 Phase 2 LoRA LR
    optimizer.param_groups[1]["lr"] = base_lr
    logger.info(f"Phase 2 LoRA LR restored to {base_lr}")

    global_step = 0
    model.train()

    # ---- LoRA 梯度诊断 (仅观测, 不影响训练) ----
    # 收集 requires_grad=True 的 LoRA A/B 矩阵, 用于 per-component grad norm
    lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and ('lora_A' in n or 'lora_B' in n)
    ]
    grad_diag_interval = train_cfg.get("grad_diag_steps", 200)
    _diag_pending = False
    if lora_params:
        logger.info(f"Grad norm diag: {len(lora_params)} LoRA tensors, every {grad_diag_interval} steps")

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

                # ---- 分量梯度诊断 (每 grad_diag_interval 步, 仅第一 micro-batch) ----
                # retain_graph=True 会暂留前向图 → 额外 10-15GB, 仅在 sync 间隙运行
                if _diag_pending and lora_params:
                    _loss_sft = loss_fn.compute_sft_loss(
                        outputs["logits"], batch["labels"], batch["label_mask"]
                    )
                    _loss_ctl = loss_fn.compute_control_loss(delta_hat, delta_target)
                    _scaled_ctl = model_cfg["loss"]["lambda_ctl"] * _loss_ctl
                    try:
                        _grads_sft = torch.autograd.grad(
                            _loss_sft, lora_params, retain_graph=True, allow_unused=True
                        )
                        _gn_sft = sum(
                            g.detach().norm().item() ** 2
                            for g in _grads_sft if g is not None
                        ) ** 0.5
                        _grads_ctl = torch.autograd.grad(
                            _scaled_ctl, lora_params, retain_graph=False, allow_unused=True
                        )
                        _gn_ctl = sum(
                            g.detach().norm().item() ** 2
                            for g in _grads_ctl if g is not None
                        ) ** 0.5
                        # 清零 autograd.grad 写入的 .grad (正常 backward 会重算)
                        for p in lora_params:
                            if p.grad is not None:
                                p.grad.zero_()
                        metrics["grad_norm_sft"] = _gn_sft
                        metrics["grad_norm_ctl"] = _gn_ctl
                        metrics["grad_ratio_ctl_sft"] = _gn_ctl / (_gn_sft + 1e-8)
                    except RuntimeError as e:
                        logger.warning(f"Grad diag OOM, disabling: {e}")
                        lora_params = []  # 永久禁用本 session
                    _diag_pending = False

                # 释放 logits (3.5 GB bf16) — CE loss 已计算完毕, 省给 backward 峰值
                del outputs["logits"]

                # 反向传播
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    # ---- LoRA 总梯度 norm (免费: 梯度已累积完毕, 直接读取 .grad) ----
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

                    # 下一 step 是否需要分量梯度诊断
                    if (global_step + 1) % grad_diag_interval == 0:
                        _diag_pending = True

            # 仅在真正执行梯度同步 (optimizer step) 后才推进 global_step / scheduler / zero_grad
            # 防止 grad_accum=16 时每个 micro-batch:
            #   - scheduler.step() 被调 16 次 → LR 衰减 16 倍过快
            #   - zero_grad() 清空累积梯度 → 有效 batch=1 (非 16)
            #   - global_step 被 +16 次 → 疯狂写 checkpoint 撑爆硬盘
            if accelerator.sync_gradients:
                global_step += 1

                # 日志
                if global_step % train_cfg["logging_steps"] == 0:
                    _metrics_str = "  ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in metrics.items()
                    )
                    _d = progress.format_dict
                    _elapsed = _d.get("elapsed", 0)
                    _rate = _d.get("rate") or 1e-8
                    _remaining = (_d.get("total", 1) - _d.get("n", 0)) / _rate
                    progress.write(
                        f"{global_step}/{_d.get('total', '?')} "
                        f"[{_elapsed:.0f}s<{_remaining:.0f}s, "
                        f"{_rate:.2f}it/s, {_metrics_str}]"
                    )
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
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from Phase 1 checkpoint → skip Phase 1, start Phase 2 directly")
    args = parser.parse_args()

    train_stage1(args.config, args.data_dir, resume_from=args.resume_from)
