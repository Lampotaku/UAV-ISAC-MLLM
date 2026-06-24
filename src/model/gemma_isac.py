"""
Gemma3-ISAC 模型
论文核心模型 — 冻结 Gemma3 + LoRA + Control Token + Projection Head

架构:
  Input (Π) → Gemma3 (LoRA) → Control Token Hidden States (Z_c)
                                        ↓
                              ConstraintProjectionHead
                                        ↓
                              δ̂ = [δ̂_q, δ̂_a, δ̂_p]

两种使用模式:
  1. 训练: forward() 返回 logits + control_states
  2. 推理: generate_warmstart() 返回投影后的 δ̂

Blackwell (RTX 5090) 兼容:
  使用 Unsloth 替代 bitsandbytes + raw PEFT
  Unsloth 内置 Blackwell sm_120 优化内核, 也兼容 CUDA 12.8+
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Dict, Optional, List
import os

from .projection_head import ConstraintProjectionHead
from .losses import UAVISACLosses


class Gemma3ISAC(nn.Module):
    """
    Gemma3 + LoRA + Control Token + Projection Head

    论文参数:
      backbone: Gemma 3 12B (google/gemma-3-12b-it)
      LoRA rank: 16, α=32
      Control tokens: 8 个特殊 token
      Projection head: 2-layer MLP [256, 256]

    Blackwell 适配:
      使用 Unsloth FastLanguageModel 加载 4-bit + LoRA
      (替代 bitsandbytes + PEFT, Unsloth 内置 sm_120 内核)
    """

    def __init__(
        self,
        model_name_or_path: str = "google/gemma-3-12b-it",
        use_4bit: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        num_control_tokens: int = 8,
        proj_head_config: Optional[Dict] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        max_seq_length: int = 4096,
        **kwargs,  # 兼容旧参数 (bnb_4bit_compute_dtype, bnb_4bit_quant_type 等)
    ):
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # ---- Unsloth: 4-bit 加载 + LoRA 一步完成 ----
        # 替代 bitsandbytes BnBConfig + PEFT LoraConfig + get_peft_model
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is required for RTX 5090 Blackwell support. "
                "Install: pip install unsloth"
            )

        self.base_model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=max_seq_length,
            load_in_4bit=use_4bit,
            dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Unwrap actual tokenizer from Gemma3Processor (Gemma 3 wraps tokenizer)
        if hasattr(tokenizer_or_processor, 'tokenizer'):
            self.tokenizer = tokenizer_or_processor.tokenizer
        else:
            self.tokenizer = tokenizer_or_processor

        # Gemma 专用: 确保 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 获取 hidden_dim (兼容 Gemma 3 嵌套 config 结构)
        config = self.base_model.config
        if hasattr(config, "hidden_size"):
            hidden_dim = config.hidden_size
        elif hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            hidden_dim = config.text_config.hidden_size
        else:
            raise AttributeError(
                f"Cannot find hidden_size in model config. "
                f"Available: {[k for k in dir(config) if not k.startswith('_')]}"
            )

        # ---- 控制 Token 配置 ----
        self.num_control_tokens = num_control_tokens
        self.hidden_dim = hidden_dim

        # 扩展 tokenizer vocabulary (添加控制 tokens)
        control_tokens = [f"<ctrl_{i}>" for i in range(num_control_tokens)]
        num_added = self.tokenizer.add_tokens(control_tokens, special_tokens=True)
        if num_added > 0:
            self.base_model.resize_token_embeddings(len(self.tokenizer))

        self.control_token_ids = self.tokenizer.convert_tokens_to_ids(control_tokens)

        self.base_model = FastLanguageModel.get_peft_model(
            self.base_model,
            r=lora_rank,
            target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
        )

        # ---- Projection Head ----
        if proj_head_config is None:
            proj_head_config = {}
        proj_head_config.setdefault("hidden_dim", hidden_dim)
        proj_head_config.setdefault("num_control_tokens", num_control_tokens)

        self.projection_head = ConstraintProjectionHead(**proj_head_config)

        # 将投影头转换为与 base model 一致的 dtype (bfloat16)
        base_dtype = next(self.base_model.parameters()).dtype
        self.projection_head = self.projection_head.to(dtype=base_dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        control_mask: Optional[torch.Tensor] = None,
        q_current: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向传播

        Args:
            input_ids: (B, seq_len) — tokenized prompt + response
            attention_mask: (B, seq_len)
            control_mask: (B, seq_len) — 标记 control token 位置 (bool)
            q_current: (B, M, 3) — 当前 UAV 位置 (用于部署投影)
            labels: (B, seq_len) — 语言模型标签

        Returns:
            dict with logits, control_states, projected_prior, etc.
        """
        # Gemma 3 text-only: token_type_ids 全部设为 1 (text), 0 已由 attention_mask 处理
        token_type_ids = torch.ones_like(input_ids)

        # Gemma3 前向传播
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True,  # 需要 hidden states 提取 Z_c
            return_dict=True,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # 最后一层 (B, seq_len, hidden_dim)

        # 提取控制 token 的 hidden states
        if control_mask is not None:
            # control_mask: (B, seq_len) bool — True at control token positions
            # 对每个样本提取
            batch_size = hidden_states.shape[0]
            control_states_list = []
            for b in range(batch_size):
                ctrl_positions = control_mask[b].nonzero(as_tuple=True)[0]
                ctrl_hidden = hidden_states[b, ctrl_positions]  # (num_ctrl, hidden_dim)
                # Pad to num_control_tokens if needed
                if ctrl_hidden.shape[0] < self.num_control_tokens:
                    pad = torch.zeros(
                        self.num_control_tokens - ctrl_hidden.shape[0],
                        self.hidden_dim,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    ctrl_hidden = torch.cat([ctrl_hidden, pad], dim=0)
                elif ctrl_hidden.shape[0] > self.num_control_tokens:
                    ctrl_hidden = ctrl_hidden[:self.num_control_tokens]
                control_states_list.append(ctrl_hidden)
            control_states = torch.stack(control_states_list, dim=0)  # (B, num_ctrl, hidden_dim)
        else:
            # Fallback: 使用序列末尾的 hidden states (近似)
            # +1 to include the last control token (Python slice is [start:end) right-exclusive)
            seq_lens = attention_mask.sum(dim=1) - 1
            control_states = torch.stack([
                hidden_states[b, seq_lens[b] - self.num_control_tokens + 1 : seq_lens[b] + 1]
                for b in range(hidden_states.shape[0])
            ], dim=0)

        # Projection Head
        prior_hat = self.projection_head(control_states, q_current)

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "control_states": control_states,
            **prior_hat,  # delta_q, delta_a, delta_p, delta_raw, ...
        }

    def generate_warmstart(
        self,
        prompt: str,
        q_current: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        推理: 生成 warm-start prior

        Args:
            prompt: 完整的多模态 prompt Π(t)
            q_current: (M, 3) 或 (1, M, 3) — 当前 UAV 位置
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            dict with delta_q, delta_a, delta_p
        """
        self.eval()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096 - max_new_tokens,
        )
        inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}

        # 添加控制 tokens 到 prompt 末尾
        ctrl_input_ids = torch.tensor(
            [self.control_token_ids] * inputs["input_ids"].shape[0],
            device=self.base_model.device,
        )
        input_ids = torch.cat([inputs["input_ids"], ctrl_input_ids], dim=1)
        attention_mask = torch.cat([
            inputs["attention_mask"],
            torch.ones_like(ctrl_input_ids),
        ], dim=1)

        # 前向传播
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=torch.ones_like(input_ids),
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]
        # 取最后 num_control_tokens 个位置的 hidden states
        control_states = hidden_states[:, -self.num_control_tokens:]  # (1, num_ctrl, hidden_dim)

        # Projection Head
        if q_current is not None:
            if q_current.ndim == 2:
                q_current = q_current.unsqueeze(0)  # (1, M, 3)

        prior_hat = self.projection_head(control_states, q_current)

        return {
            "delta_q": prior_hat["delta_q"].squeeze(0).cpu(),      # (M, 3)
            "delta_a": prior_hat["delta_a"].squeeze(0).cpu(),      # (M, K)
            "delta_p": prior_hat["delta_p"].squeeze(0).cpu(),      # (M, K+1)
        }

    def save_pretrained(self, save_dir: str):
        """保存 LoRA 权重 + Projection Head"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存 LoRA (Unsloth model 兼容 PEFT save)
        self.base_model.save_pretrained(os.path.join(save_dir, "lora"))

        # 保存 Projection Head
        torch.save(
            self.projection_head.state_dict(),
            os.path.join(save_dir, "projection_head.pt"),
        )

        # 保存 tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

    @classmethod
    def from_pretrained(cls, load_dir: str, base_model_name: str,
                        torch_dtype: torch.dtype = torch.bfloat16,
                        attn_implementation: str = "flash_attention_2",
                        **kwargs):
        """
        加载完整模型 (LoRA + Projection Head + Tokenizer)

        Blackwell 优化: 用 Unsloth 加载 base model (仅一次 4-bit 加载),
        然后挂载已训练的 LoRA 权重, 不再重复加载完整模型.
        """
        from unsloth import FastLanguageModel
        from peft import PeftModel

        # 提取构造参数 (kwargs 里的旧 BnB 参数被忽略)
        use_4bit = kwargs.pop("use_4bit", True)
        lora_rank = kwargs.pop("lora_rank", 16)
        lora_alpha = kwargs.pop("lora_alpha", 32)
        lora_dropout = kwargs.pop("lora_dropout", 0.05)
        lora_target_modules = kwargs.pop("lora_target_modules",
                                          ["q_proj", "k_proj", "v_proj", "o_proj"])
        num_control_tokens = kwargs.pop("num_control_tokens", 8)
        proj_head_config = kwargs.pop("proj_head_config", {})
        max_seq_length = kwargs.pop("max_seq_length", 4096)

        # ---- 用 Unsloth 加载 base model (4-bit + fresh LoRA) ----
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=use_4bit,
            dtype=torch_dtype,
            trust_remote_code=True,
        )

        # ---- 控制 Token 扩展 ----
        hidden_dim = base_model.config.hidden_size
        control_tokens = [f"<ctrl_{i}>" for i in range(num_control_tokens)]
        num_added = tokenizer.add_tokens(control_tokens, special_tokens=True)
        if num_added > 0:
            base_model.resize_token_embeddings(len(tokenizer))
        control_token_ids = tokenizer.convert_tokens_to_ids(control_tokens)

        # ---- 加载已训练的 LoRA 权重 ----
        lora_path = os.path.join(load_dir, "lora")
        if os.path.exists(lora_path):
            base_model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)
        else:
            base_model = FastLanguageModel.get_peft_model(
                base_model,
                r=lora_rank,
                target_modules=lora_target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing=True,
                random_state=42,
            )

        # ---- 加载 Projection Head ----
        if proj_head_config is None:
            proj_head_config = {}
        proj_head_config.setdefault("hidden_dim", hidden_dim)
        proj_head_config.setdefault("num_control_tokens", num_control_tokens)
        projection_head = ConstraintProjectionHead(**proj_head_config)

        proj_path = os.path.join(load_dir, "projection_head.pt")
        if os.path.exists(proj_path):
            proj_state = torch.load(proj_path, map_location="cpu")
            projection_head.load_state_dict(proj_state)

        # ---- 构造实例 (绕过 __init__, 避免重复加载) ----
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.base_model = base_model
        instance.tokenizer = tokenizer
        instance.num_control_tokens = num_control_tokens
        instance.hidden_dim = hidden_dim
        instance.control_token_ids = control_token_ids
        instance.projection_head = projection_head

        return instance
