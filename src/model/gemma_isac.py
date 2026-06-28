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

模型加载双轨制:
  轨道 A (use_4bit=True):  Unsloth FastLanguageModel → eager attention (16-21s/step)
  轨道 B (use_4bit=False): Native HF + PEFT + SDPA → cuDNN Fused Attention (~2-3s/step)
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

        # ---- 模型加载: 双轨制 (Unsloth 4-bit / Native bf16 SDPA) ----
        if use_4bit:
            # 轨道 A: Unsloth 4-bit QLoRA (省显存, 但 Gemma 3 只能 eager attention)
            try:
                from unsloth import FastLanguageModel
            except ImportError:
                raise ImportError(
                    "Unsloth is required for 4-bit QLoRA. "
                    "Install: pip install unsloth"
                )

            self.base_model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
                model_name=model_name_or_path,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )

            # Unwrap actual tokenizer from Gemma3Processor
            if hasattr(tokenizer_or_processor, 'tokenizer'):
                self.tokenizer = tokenizer_or_processor.tokenizer
            else:
                self.tokenizer = tokenizer_or_processor
        else:
            # 轨道 B: Native HuggingFace + PEFT bf16 — SDPA 真正生效, ~2-3s/step
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )

            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
            # 不用 device_map="auto" — Accelerate 负责设备放置

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

        # ---- LoRA 注入 ----
        if use_4bit:
            # Unsloth 路径: FastLanguageModel.get_peft_model
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
            # 新增 token embedding 默认冻结, 需手动开启 (否则控制 token 保持随机初始化)
            if num_added > 0:
                embed = self.base_model.get_input_embeddings()
                if hasattr(embed, 'weight'):
                    embed.weight.requires_grad = True
        else:
            # Native PEFT 路径: LoraConfig + get_peft_model
            # modules_to_save 确保新增的控制 token embedding 参与训练
            # 注意: modules_to_save 仅含 embed_tokens (不含 lm_head)
            # lm_head 冻结: CE 梯度通过 frozen lm_head 回传到 hidden_states → LoRA
            # 含 lm_head 会打断 Gemma 3 的 tie_word_embeddings, 浪费 ~12 GB
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                modules_to_save=["embed_tokens"],
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            self.base_model.gradient_checkpointing_enable()
            # 确保 lm_head 冻结 + 权重绑定 (防御)
            causal_lm = self.base_model.model
            causal_lm.lm_head.weight.requires_grad = False
            if (hasattr(causal_lm.config, 'tie_word_embeddings')
                    and causal_lm.config.tie_word_embeddings):
                causal_lm.lm_head.weight = causal_lm.get_input_embeddings().weight

        # ---- Projection Head ----
        if proj_head_config is None:
            proj_head_config = {}
        proj_head_config.setdefault("hidden_dim", hidden_dim)
        proj_head_config.setdefault("num_control_tokens", num_control_tokens)

        self.projection_head = ConstraintProjectionHead(**proj_head_config)
        # 注意: projection_head 保持 float32，不转为 bf16
        # 原因: 训练目标 (delta_q/a/p 标签) 是 float32，loss 计算需要同 dtype
        # 在 forward() 中 control_states 会从 bf16 cast 到 f32 再送入投影头

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

        # ── 高效前向: 绕过 output_hidden_states + 内部 fp32 logits 膨胀 ──
        # 旧代码两个内存陷阱 (合计 ~22 GB 峰值):
        #   a) output_hidden_states=True → 存 48 层 hidden states (~6 GB)
        #   b) 传 labels 给 CausalLM → HF 内部 cross-entropy 把 bf16 logits (8 GB)
        #       cast 到 fp32 (16 GB) → 峰值 24 GB 仅 logits, 触发 OOM
        # 修复: 直接调底层 transformer, 拿 last_hidden_state + 手动 lm_head
        # 结构: base_model (PeftModel) → .model (Gemma3ForCausalLM) → .model (Transformer) + .lm_head
        causal_lm = self.base_model.model       # Gemma3ForCausalLM
        transformer = causal_lm.model           # Gemma3Model (48 layers, norm)
        lm_head = causal_lm.lm_head             # vocab projection

        transformer_out = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,          # 只要 last_hidden_state, 省 ~6 GB
            use_cache=False,                     # 训练不需要 KV cache
            return_dict=True,
        )
        hidden_states = transformer_out.last_hidden_state  # (B, seq_len, hidden_dim)
        logits = lm_head(hidden_states)          # 保持 bf16, 不产生 fp32 副本

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
            # ⚠️ 脆弱: 假设 control token 在序列绝对末尾, right-padding 下可能切到 pad token.
            # 训练主路径应始终传入 control_mask.
            import logging
            _logger = logging.getLogger(__name__)
            _logger.warning(
                "control_mask is None — using fragile fallback extraction from sequence tail. "
                "This assumes control tokens are at the absolute end and may fail with "
                "right-padded batches. Pass control_mask for reliable extraction."
            )
            # +1 to include the last control token (Python slice is [start:end) right-exclusive)
            seq_lens = attention_mask.sum(dim=1) - 1
            control_states = torch.stack([
                hidden_states[b, seq_lens[b] - self.num_control_tokens + 1 : seq_lens[b] + 1]
                for b in range(hidden_states.shape[0])
            ], dim=0)

        # Projection Head (投影头是 float32，需要将 bf16 hidden states 转为 f32)
        prior_hat = self.projection_head(control_states.float(), q_current)

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

        # 前向传播 (高效路径: 跳过 CausalLM wrapper, 直接拿 last_hidden_state)
        with torch.no_grad():
            causal_lm = self.base_model.model
            transformer = causal_lm.model
            transformer_out = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=torch.ones_like(input_ids),
                output_hidden_states=False,     # 只要 last_hidden_state
                use_cache=False,
                return_dict=True,
            )
        hidden_states = transformer_out.last_hidden_state
        # 取最后 num_control_tokens 个位置的 hidden states
        control_states = hidden_states[:, -self.num_control_tokens:]  # (1, num_ctrl, hidden_dim)

        # Projection Head (推理时: bf16 control_states → f32 输入投影头)
        if q_current is not None:
            if q_current.ndim == 2:
                q_current = q_current.unsqueeze(0)  # (1, M, 3)
            q_current = q_current.to(self.base_model.device)  # 对齐设备 (外部输入默认 CPU)

        prior_hat = self.projection_head(control_states.float(), q_current)

        # .detach() 防御: no_grad() 下 projection_head 参数仍带 requires_grad,
        # .cpu() 不切断 grad 属性 → 外部 .numpy() 会崩 (RuntimeError: Can't call numpy() on Tensor that requires grad)
        return {
            "delta_q": prior_hat["delta_q"].squeeze(0).detach().cpu().float(),      # (M, 3)
            "delta_a": prior_hat["delta_a"].squeeze(0).detach().cpu().float(),      # (M, K)
            "delta_p": prior_hat["delta_p"].squeeze(0).detach().cpu().float(),      # (M, K+1)
        }

    def save_pretrained(self, save_dir: str):
        """
        保存 LoRA adapter + Control Token Embedding + Projection Head + Tokenizer

        ⚠️  关键优化: 不保存 modules_to_save (embed_tokens + lm_head) 的完整 256K 词表
        权重 (→4GB), 只保存 8 个 control token 对应的行 (→60KB)。
        4GB → ~100MB per checkpoint。
        """
        import json
        from safetensors.torch import save_file as safe_save_file
        from peft.utils.save_and_load import get_peft_model_state_dict

        os.makedirs(save_dir, exist_ok=True)
        lora_dir = os.path.join(save_dir, "lora")
        os.makedirs(lora_dir, exist_ok=True)

        # ── Step 1: 分离 LoRA 权重 vs modules_to_save 权重 ──
        full_state = get_peft_model_state_dict(self.base_model)
        lora_state = {}
        for key, tensor in full_state.items():
            if "lora_" in key:
                lora_state[key] = tensor
            # modules_to_save 的 key (无 "lora_") 跳过 — 单独保存 control token 行

        # ── Step 2: 保存精简版 LoRA safetensors ──
        safe_save_file(lora_state, os.path.join(lora_dir, "adapter_model.safetensors"))

        # ── Step 3: 保存 adapter_config.json (不含 modules_to_save) ──
        peft_cfg = self.base_model.peft_config["default"]
        cfg_dict = peft_cfg.to_dict()
        cfg_dict.pop("modules_to_save", None)
        with open(os.path.join(lora_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)

        # ── Step 4: 只保存 control token 的 embedding 行 (~60KB vs 4GB) ──
        embed = self.base_model.get_input_embeddings()
        ctrl_embed = embed.weight.data[self.control_token_ids].clone().cpu()
        torch.save(ctrl_embed, os.path.join(save_dir, "ctrl_embed.pt"))

        # lm_head: 仅当与 embed 不共享权重时保存 (Gemma 3 默认 weight tying)
        output_embeds = self.base_model.get_output_embeddings()
        if output_embeds is not None and output_embeds.weight.data_ptr() != embed.weight.data_ptr():
            ctrl_lm = output_embeds.weight.data[self.control_token_ids].clone().cpu()
            torch.save(ctrl_lm, os.path.join(save_dir, "ctrl_lm_head.pt"))

        # ── Step 5: 保存 Projection Head ──
        torch.save(
            self.projection_head.state_dict(),
            os.path.join(save_dir, "projection_head.pt"),
        )

        # ── Step 6: 保存 tokenizer ──
        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))

    @classmethod
    def from_pretrained(cls, load_dir: str, base_model_name: str,
                        torch_dtype: torch.dtype = torch.bfloat16,
                        attn_implementation: str = "flash_attention_2",
                        **kwargs):
        """
        加载完整模型 (LoRA + Projection Head + Tokenizer)

        双轨制: use_4bit=True → Unsloth, use_4bit=False → Native HF + PEFT SDPA.
        """
        from peft import PeftModel, LoraConfig, get_peft_model

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

        # ---- 模型加载: 双轨制 ----
        if use_4bit:
            from unsloth import FastLanguageModel

            base_model, tokenizer_or_processor = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=True,
                dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )

            # Unwrap actual tokenizer from Gemma3Processor
            if hasattr(tokenizer_or_processor, 'tokenizer'):
                tokenizer = tokenizer_or_processor.tokenizer
            else:
                tokenizer = tokenizer_or_processor
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )

        # 确保 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ---- 控制 Token 扩展 ----
        # 兼容 Gemma 3 嵌套 config 结构
        config = base_model.config
        if hasattr(config, "hidden_size"):
            hidden_dim = config.hidden_size
        elif hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            hidden_dim = config.text_config.hidden_size
        else:
            raise AttributeError(
                f"Cannot find hidden_size in model config. "
                f"Available: {[k for k in dir(config) if not k.startswith('_')]}"
            )
        control_tokens = [f"<ctrl_{i}>" for i in range(num_control_tokens)]
        num_added = tokenizer.add_tokens(control_tokens, special_tokens=True)
        if num_added > 0:
            base_model.resize_token_embeddings(len(tokenizer))
        control_token_ids = tokenizer.convert_tokens_to_ids(control_tokens)

        # ---- 加载 LoRA 权重 (双轨) ----
        lora_path = os.path.join(load_dir, "lora")
        if os.path.exists(lora_path):
            # 已有训练好的 LoRA → 直接加载
            base_model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=True)

            # ══ OOM #6 修复: 三管齐下 ══
            # 1) Gradient checkpointing (必须: from_pretrained 不恢复 gc 设置)
            base_model.gradient_checkpointing_enable()
            # 防御: 直接对底层 transformer 开 gc (PeftModel 的委托可能不完整)
            try:
                base_model.model.model.gradient_checkpointing_enable()
            except Exception:
                pass

            # 2) 冻结 lm_head (省 ~12 GB: 权重 2GB + Adam m/v 8GB)
            #    CE 梯度通过 frozen lm_head 回传到 hidden_states → LoRA, 无需训练 lm_head
            causal_lm = base_model.model  # PeftModel → Gemma3ForCausalLM
            causal_lm.lm_head.weight.requires_grad = False

            # 3) 重绑 embed_tokens ↔ lm_head (PEFT modules_to_save 可能打断 tie_word_embeddings)
            if (hasattr(causal_lm.config, 'tie_word_embeddings')
                    and causal_lm.config.tie_word_embeddings):
                # 强制重绑: lm_head.weight 指向 embed_tokens.weight 同一张量
                embed = causal_lm.get_input_embeddings()
                causal_lm.lm_head.weight = embed.weight
        elif use_4bit:
            # 4-bit 路径: 用 Unsloth 创建 fresh LoRA
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
            # 新增 token embedding 需手动开启训练
            if num_added > 0:
                embed = base_model.get_input_embeddings()
                if hasattr(embed, 'weight'):
                    embed.weight.requires_grad = True
        else:
            # Native PEFT 路径: LoraConfig + get_peft_model
            # (不含 lm_head: 冻结 lm_head 省 ~12 GB, 梯度通过 frozen lm_head 回传)
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                modules_to_save=["embed_tokens"],
            )
            base_model = get_peft_model(base_model, peft_config)
            base_model.gradient_checkpointing_enable()
            # 确保 lm_head 冻结 + 权重绑定
            causal_lm = base_model.model
            causal_lm.lm_head.weight.requires_grad = False
            if (hasattr(causal_lm.config, 'tie_word_embeddings')
                    and causal_lm.config.tie_word_embeddings):
                causal_lm.lm_head.weight = causal_lm.get_input_embeddings().weight

        # ---- 恢复 Control Token Embedding (新格式: 单独保存的 8 行) ----
        # 新 save_pretrained 只保存 LoRA + ctrl_embed.pt, 不含完整 embed_tokens.
        # 此处将保存的 8 行 patch 回 base_model embedding (覆盖随机初始化).
        ctrl_embed_path = os.path.join(load_dir, "ctrl_embed.pt")
        if os.path.exists(ctrl_embed_path):
            ctrl_embed = torch.load(ctrl_embed_path, map_location="cpu")
            embed = base_model.get_input_embeddings()
            embed.weight.data[control_token_ids] = ctrl_embed.to(
                dtype=embed.weight.dtype, device=embed.weight.device
            )
            # lm_head: 仅当独立文件存在时 patch (共享权重时无需)
            ctrl_lm_path = os.path.join(load_dir, "ctrl_lm_head.pt")
            if os.path.exists(ctrl_lm_path):
                ctrl_lm = torch.load(ctrl_lm_path, map_location="cpu")
                output_embeds = base_model.get_output_embeddings()
                if output_embeds is not None:
                    output_embeds.weight.data[control_token_ids] = ctrl_lm.to(
                        dtype=output_embeds.weight.dtype,
                        device=output_embeds.weight.device,
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
        # 确保投影头与 base_model 在同一设备 (state_dict 从 CPU 加载)
        instance.projection_head = projection_head.to(base_model.device)

        return instance
