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
from torch.utils.checkpoint import checkpoint as _grad_ckpt
from typing import Dict, Optional, Tuple


def _ce_none(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy with reduction='none' — thin wrapper for _grad_ckpt.

    必须定义在模块级别 (而非 lambda), _grad_ckpt(use_reentrant=False) 需要
    pickle-able 的函数引用.
    """
    return F.cross_entropy(logits, labels, reduction="none")


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

        内存优化 (方案 B — Unsloth Chunked CE):
          Unsloth fast_cross_entropy_loss 是 Triton 分块内核, 在 256K 词表上
          逐 chunk 计算 CE + backward, 永不成完整的 fp32 梯度张量 ∂L/∂logits
          (~16 GB for bs=4 × 4096 × 256K). 省显存同时保持全速 (2-3s/step),
          因为只借用了 Unsloth 的 loss 内核, 不动模型加载路径 (SDPA 不受影响).
          Fallback: _grad_ckpt 包装 F.cross_entropy (纯 PyTorch, backward 重算).
        """
        # 右移: predict next token
        # 必须 .contiguous(): [:, :-1, :] 对中间维切片产生非连续 stride,
        # Unsloth 内核的 .view(batch*seq_len, d) 要求连续内存.
        # bf16 拷贝 ~8 GB, 砍掉 fp32 梯度 16 GB 后绰绰有余.
        shift_logits = logits[:, :-1, :].contiguous()    # (B, S-1, V)
        shift_labels = labels[:, 1:]         # (B, S-1)

        if label_mask is not None:
            shift_mask = label_mask[:, 1:]   # (B, S-1)

        try:
            # 🚀 Unsloth Chunked Cross-Entropy (Triton 内核)
            # 逐 chunk 计算 → backward 时永不成完整 16 GB fp32 梯度张量
            # 对 bs=4 × 4096 × 256K: 峰值 ~70 GB (vs _grad_ckpt ~90 GB)
            from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

            if label_mask is not None:
                # fast_cross_entropy_loss 遵循 ignore_index=-100 约定
                _labels = shift_labels.clone()
                _labels[shift_mask == 0] = -100
                loss = fast_cross_entropy_loss(shift_logits, _labels)
            else:
                loss = fast_cross_entropy_loss(shift_logits, shift_labels)

        except ImportError:
            # 回退路径: 梯度检查点 + 原生 CE
            # _ce_none 用 reduction='none' 不支持 ignore_index, 所以用显式 mask
            shift_logits_t = shift_logits.transpose(1, 2)   # (B, V, S-1)
            loss = _grad_ckpt(
                _ce_none, shift_logits_t, shift_labels,
                use_reentrant=False,
            )  # → (B, S-1)

            if label_mask is not None:
                loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
            else:
                loss = loss.mean()

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
