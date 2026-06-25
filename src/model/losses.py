"""
损失函数
论文 Section 4 & 5

损失汇总:
  Stage I:
    L_I = L_SFT + λ_ctl * L_ctl              (公式 30)

  Stage II:
    L_II = L_DPO + μ * L_SFT + λ_ctl * L_ctl (公式 37)

  Total:
    L = L_II + λ_sep * L_sep                  (公式 39)

其中:
  L_SFT: causal LM cross-entropy (公式 27)
  L_ctl: continuous warm-start regression (公式 28)
  L_DPO: direct preference optimization (公式 34)
  L_sep: UAV separation penalty (公式 27)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class UAVISACLosses:
    """
    UAV-ISAC 训练损失计算器

    封装所有论文中的损失函数, 解耦 token-level 和 continuous losses
    """

    def __init__(
        self,
        lambda_ctl: float = 0.5,
        lambda_q: float = 1.0,
        lambda_a: float = 0.5,
        lambda_p: float = 0.3,
        lambda_sep: float = 0.1,
        dpo_beta: float = 0.1,
        sft_anchor_mu: float = 0.05,
    ):
        self.lambda_ctl = lambda_ctl
        self.lambda_q = lambda_q
        self.lambda_a = lambda_a
        self.lambda_p = lambda_p
        self.lambda_sep = lambda_sep
        self.dpo_beta = dpo_beta
        self.sft_anchor_mu = sft_anchor_mu

    def compute_control_loss(
        self,
        delta_hat: Dict[str, torch.Tensor],
        delta_target: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        连续控制损失 L_ctl (公式 28)

        L_ctl = λ_q ||δ̂_q - δ*_q||² + λ_a BCE(δ̂_a, δ*_a) + λ_p ||δ̂_p - δ*_p||²

        其中:
          - δ̂_q, δ̂_p → MSE (连续回归)
          - δ̂_a → BCE (软关联 vs 二值 oracle)
        """
        # Auto-align dtypes (projection head may be f32, model bf16, etc.)
        common_dtype = torch.float32
        dq_hat = delta_hat["delta_q"].to(dtype=common_dtype)
        da_hat = delta_hat["delta_a"].to(dtype=common_dtype)
        dp_hat = delta_hat["delta_p"].to(dtype=common_dtype)
        dq_tgt = delta_target["delta_q"].to(dtype=common_dtype)
        da_tgt = delta_target["delta_a"].to(dtype=common_dtype)
        dp_tgt = delta_target["delta_p"].to(dtype=common_dtype)

        # 位移 loss (MSE)
        loss_q = F.mse_loss(dq_hat, dq_tgt)

        # 关联 loss (BCE: 软关联 vs 二值 oracle)
        loss_a = F.binary_cross_entropy(da_hat, da_tgt)

        # 功率 loss (MSE)
        loss_p = F.mse_loss(dp_hat, dp_tgt)

        return self.lambda_q * loss_q + self.lambda_a * loss_a + self.lambda_p * loss_p

    def compute_separation_penalty(
        self,
        q_hat: torch.Tensor,         # (B, M, 3) — 投影后的 UAV 位置
        d_min: float = 10.0,
    ) -> torch.Tensor:
        """
        UAV 分离惩罚 L_sep (公式 27)

        L_sep = Σ_{m<m'} [max(0, d_min - ||q̂_m - q̂_m'||_2)]²

        非凸约束转化为可微惩罚项
        """
        B, M, _ = q_hat.shape
        if M < 2:
            return torch.tensor(0.0, device=q_hat.device)

        total_penalty = 0.0
        for m in range(M):
            for mp in range(m + 1, M):
                diff = q_hat[:, m, :2] - q_hat[:, mp, :2]  # (B, 2) — 仅水平
                dist = torch.norm(diff, dim=-1)             # (B,)
                penalty = F.relu(d_min - dist) ** 2         # (B,)
                total_penalty += penalty.mean()

        return self.lambda_sep * total_penalty

    def compute_dpo_loss(
        self,
        logp_chosen: torch.Tensor,       # (B,)
        logp_rejected: torch.Tensor,     # (B,)
        logp_ref_chosen: torch.Tensor,   # (B,) — 冻结参考模型
        logp_ref_rejected: torch.Tensor, # (B,)
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """
        DPO 损失 L_DPO (公式 34)

        L_DPO = -E[ log σ( β * log(π_θ(chosen)/π_ref(chosen))
                            - β * log(π_θ(rejected)/π_ref(rejected)) ) ]

        Args:
            logp_chosen: log π_θ(chosen|Π)
            logp_rejected: log π_θ(rejected|Π)
            logp_ref_chosen: log π_0(chosen|Π)
            logp_ref_rejected: log π_0(rejected|Π)
        """
        # 对数比 (相对于参考)
        chosen_ratio = logp_chosen - logp_ref_chosen       # (B,)
        rejected_ratio = logp_rejected - logp_ref_rejected # (B,)

        # DPO 目标
        logits = self.dpo_beta * (chosen_ratio - rejected_ratio)

        # label_smoothing (可选)
        if label_smoothing > 0:
            targets = 1.0 - label_smoothing
        else:
            targets = 1.0

        loss = -F.logsigmoid(logits)
        loss = loss.mean()

        # 准确率监控
        with torch.no_grad():
            accuracy = (logits > 0).float().mean()

        return loss, accuracy

    def compute_sft_loss(
        self,
        logits: torch.Tensor,        # (B, seq_len, vocab_size)
        labels: torch.Tensor,        # (B, seq_len)
        label_mask: Optional[torch.Tensor] = None,  # (B, seq_len)
    ) -> torch.Tensor:
        """
        SFT 损失 L_SFT (公式 27)

        标准 causal LM cross-entropy
        可用 label_mask 只计算 response 部分的 token

        纯 PyTorch 原生实现:
          直接用 F.cross_entropy 展平计算, 不依赖 Unsloth (会产生
          CheckpointError — 局部 import 仍触发全局 monkey-patch, 导致
          forward/recompute 张量数不一致 68≠65) 也不依赖梯度检查点.
          bs=1 时单步 CE 的 fp32 梯度约 4GB, 通过 grad_accum=16
          保持有效 batch=16 且显存安全.
        """
        # 右移: predict next token
        shift_logits = logits[:, :-1, :].contiguous()    # (B, S-1, V)
        shift_labels = labels[:, 1:].clone()              # (B, S-1)

        if label_mask is not None:
            shift_mask = label_mask[:, 1:]                # (B, S-1)
            shift_labels[shift_mask == 0] = -100

        # 展平为 2D (N, V) 匹配 F.cross_entropy 标准输入
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def compute_stage1_total(
        self,
        delta_hat: Dict[str, torch.Tensor],
        delta_target: Dict[str, torch.Tensor],
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None,
        q_hat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage I 总损失: L_I = L_SFT + λ_ctl * L_ctl  (+ λ_sep * L_sep)

        Returns:
            total_loss, metrics_dict
        """
        loss_sft = self.compute_sft_loss(logits, labels, label_mask)
        loss_ctl = self.compute_control_loss(delta_hat, delta_target)

        total = loss_sft + self.lambda_ctl * loss_ctl

        metrics = {
            "loss_sft": loss_sft.item(),
            "loss_ctl": loss_ctl.item(),
        }

        if q_hat is not None:
            loss_sep = self.compute_separation_penalty(q_hat)
            total = total + loss_sep
            metrics["loss_sep"] = loss_sep.item()

        metrics["loss_total"] = total.item()
        return total, metrics

    def compute_stage2_total(
        self,
        delta_hat: Dict[str, torch.Tensor],
        delta_target: Dict[str, torch.Tensor],
        logp_chosen: torch.Tensor,
        logp_rejected: torch.Tensor,
        logp_ref_chosen: torch.Tensor,
        logp_ref_rejected: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        q_hat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage II 总损失: L = L_DPO + μ*L_SFT + λ_ctl*L_ctl + λ_sep*L_sep

        Returns:
            total_loss, metrics_dict
        """
        loss_dpo, dpo_acc = self.compute_dpo_loss(
            logp_chosen, logp_rejected, logp_ref_chosen, logp_ref_rejected
        )

        loss_ctl = self.compute_control_loss(delta_hat, delta_target)

        total = loss_dpo + self.lambda_ctl * loss_ctl

        metrics = {
            "loss_dpo": loss_dpo.item(),
            "dpo_accuracy": dpo_acc.item(),
            "loss_ctl": loss_ctl.item(),
        }

        # SFT anchor (防遗忘)
        if self.sft_anchor_mu > 0 and logits is not None and labels is not None:
            loss_sft = self.compute_sft_loss(logits, labels, label_mask)
            total = total + self.sft_anchor_mu * loss_sft
            metrics["loss_sft_anchor"] = loss_sft.item()

        # 分离惩罚
        if q_hat is not None:
            loss_sep = self.compute_separation_penalty(q_hat)
            total = total + loss_sep
            metrics["loss_sep"] = loss_sep.item()

        metrics["loss_total"] = total.item()
        return total, metrics
