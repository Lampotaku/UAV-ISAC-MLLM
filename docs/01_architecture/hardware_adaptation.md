---
type: reference
status: current
stage: all
last_updated: 2026-06-26
related: [canonical_config, adr_001_unsloth_removal, adr_003_sdpa_canonical, adr_004_4bit_qlora_blackwell]
---

# Hardware Adaptation — Blackwell RTX PRO 6000

## 硬件规格

| 规格 | 值 |
|------|-----|
| GPU | NVIDIA RTX PRO 6000 96GB |
| 架构 | Blackwell (sm_120) |
| CUDA | 13.0 |
| Python | 3.12 |
| PyTorch | 2.12.1 |
| 系统盘 | 30 GB (不可用于数据) |
| 数据盘 | `/root/autodl-tmp/` (充足空间) |

## Blackwell 生态现状 (2026-06)

Blackwell sm_120 是新架构，生态系统仍不成熟。以下是已知的兼容性问题和解决方案：

| 组件 | 状态 | 替代方案 |
|------|------|----------|
| **bitsandbytes** | ❌ 不支持 Blackwell | bf16 全精度 (96GB 无需量化) |
| **Flash Attention 2** | ❌ 无预编译 sm_120 wheel | PyTorch SDPA (native) |
| **Triton FlexAttention** | ⚠️ Shared memory 不足 | 禁用: `TORCHINDUCTOR_FLEX_ATTENTION=0` |
| **Triton kernels** | ⚠️ 未针对 sm_120 调优 | 接受性能损失 |
| **Unsloth** | ⚠️ 全局劫持 + 强制 eager | **已彻底移除** (Plan A) |

## Unsloth → 移除 (Plan A)

### 为什么移除？

Unsloth 存在三个无法共存的问题：

1. **全局 monkey-patch**: 一旦 `import unsloth`，立即替换 transformers 底层的 attention 实现
2. **强制 eager attention**: Gemma 3 的 5:1 sliding window + global attention 交错模式没有对应的 Triton kernel，Unsloth 强制 `attn_implementation="eager"`
3. **与 grad checkpoint 冲突**: forward 用纯净 HF 路径，backward 时 Unsloth 的替换导致激活张量数量不匹配 → `CheckpointError`

### 替代: 纯 PyTorch + PEFT

```python
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

# No Unsloth import anywhere
model = AutoModel.from_pretrained(
    "google/gemma-3-12b-pt",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",       # ★ Native PyTorch SDPA
)
model = get_peft_model(model, lora_config)
```

结果: 2.5s/step (vs Unsloth eager 的 16-21s/step)，~8x 提速。

详见 [06_decisions/adr_001_unsloth_removal.md](../06_decisions/adr_001_unsloth_removal.md)

## SDPA vs Eager vs FA2

| 方案 | 速度 | VRAM | 可行性 |
|------|------|------|--------|
| **SDPA (当前)** | 2.5-4.1s/step | ~76 GB | ✅ 原生 PyTorch |
| Eager | 16-21s/step | ~80 GB | ✅ 但太慢 |
| Flash Attention 2 | ~2s/step (预估) | ~70 GB | ❌ 无 sm_120 wheel |

## Triton FlexAttention 禁用

**必须**在所有训练脚本中设置，**在 `import torch` 之前**:

```bash
export TORCHINDUCTOR_FLEX_ATTENTION=0
```

原因: FlexAttention backward kernel 需要 114KB shared memory/sm，RTX PRO 6000 (Blackwell sm_120) 每个 SM 只有 101KB。

适用的文件:
- `scripts/test_sft_overfit.py`
- `src/training/train_sft.py`
- `src/training/train_dpo.py`

## bf16 全精度 (96GB 无需量化)

RTX PRO 6000 拥有 96GB VRAM，Gemma 3 12B 在 bf16 下仅需 ~24GB，无需 4-bit 量化即可完整训练:

```python
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

model = AutoModel.from_pretrained(
    "google/gemma-3-12b-pt",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",       # ★ Native PyTorch SDPA
)
model = get_peft_model(model, lora_config)
```

优势:
- 无需 Unsloth 或 bitsandbytes 量化加载
- bf16 权重精度高于 4-bit，训练质量更好
- 96GB 显存足以容纳 bf16 模型 (~24GB) + LoRA (~1GB) + activations (~22GB) + optimizer (~8GB)，峰值 ~76GB，仍有 ~20GB 余量

## VRAM 剖析 (bs=2, seq=3456, SDPA, bf16 全精度)

```
CUDA context + 碎片:         ~18 GB  (96GB 卡的固定开销)
Gemma 3 12B (bf16):          ~24 GB  (全精度权重)
LoRA adapters (bf16):        ~1 GB   (r=16, α=32)
Activations (grad ckpt):     ~15 GB  (GQA, 48 layers × hidden 3840)
CE fp32 中间张量:             ~7 GB   (bs=2 × seq=3456 × vocab=256K × 4B)
Optimizer states (8-bit):    ~8 GB   (AdamW, 仅 LoRA 参数)
其他开销:                     ~3 GB
─────────────────────────────────
峰值总计:                     ~76 GB / 96 GB  (20 GB 余量)
```

## DPO VRAM (bs=1, seq=3456, SDPA, bf16 全精度)

DPO 需要同时加载两个模型 (policy + reference):

```
Policy model (bf16):         ~28 GB  (12B bf16 + LoRA)
Reference model (bf16):      ~24 GB  (12B bf16, 无 LoRA, 仅推理)
CE + activations (bs=1):     ~12 GB
Optimizer + context:         ~11 GB
─────────────────────────────────
峰值总计:                      ~75 GB / 96 GB  (边界安全)
```

## Grad Checkpoint 策略

```python
model.gradient_checkpointing_enable()
```

- 节省 ~16 GB (GQA 的 attention 中间张量特别大)
- 仅对 transformer layers 生效
- Projection head 和 lm_head 不在 checkpoint 范围内

## 已知陷阱和解决方案

### 1. Gemma 3 multimodal config

`Gemma3Config` 嵌套 multimodal 配置，`hidden_size` 在 `config.text_config` 而非顶层:

```python
hidden_size = getattr(config, 'hidden_size', None) or config.text_config.hidden_size
```

### 2. `token_type_ids` 要求

Gemma 3 在训练模式下需要 `token_type_ids`:

```python
token_type_ids = torch.ones_like(input_ids)  # text-only mode
```

### 3. `from_pretrained` 设备不一致

HF 的 `from_pretrained` 将 projection head 加载到 CPU，即使 base model 在 GPU:

```python
model.projection_head = model.projection_head.to(base_model.device)
```

### 4. BFloat16 与 Float32 协调

Projection head 保持 f32 (随机初始化参数少，精度更重要):
```python
control_states = hidden_states.float()  # bf16 → f32 before projection head
```

### 5. BCE Loss dtype

确保 prediction 和 target 的 dtype 一致:
```python
loss = F.binary_cross_entropy_with_logits(
    pred.to(target.dtype), target
)
```

### 6. BLAS 线程抑制

在所有训练脚本中，**在 `import torch` 之前**:
```python
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
```

防止 DataLoader workers 争抢 CPU 资源。

## 服务器路径约定

| 用途 | 路径 |
|------|------|
| 项目根 | `/root/UAV-ISAC-MLLM/` |
| 数据 (读写) | `/root/autodl-tmp/data/` |
| 模型输出 | `/root/autodl-tmp/outputs/` |
| 系统盘 (只读) | `/root/` (30GB, 不存放数据) |

**规则**: 所有数据、checkpoint、日志必须写入 `/root/autodl-tmp/`。系统盘仅 30GB。
