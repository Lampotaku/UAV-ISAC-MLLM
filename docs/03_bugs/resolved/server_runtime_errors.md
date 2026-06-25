---
type: postmortem
status: resolved
severity: P0
stage: sft
commits: [13c402d, ..., 975931f]  # 9 fix commits
last_updated: 2026-06-26
related: [training_code_bugs, oom_1_through_5, hardware_adaptation]
---

# Bug: Server Runtime Errors — Blackwell 8 连击

**来源**: Doc 20 (server debug log) | **发生**: 首次在 RTX PRO 6000 服务器上运行 SFT 过拟合测试时

## 8 个运行时错误

| # | 错误 | 症状 | 根因 | 修复 |
|---|------|------|------|------|
| 1 | Unsloth 版本过旧 | ImportError / 不支持 Gemma 3 | Unsloth 旧版不认识 Gemma 3 | `pip install --upgrade unsloth` |
| 2 | `Gemma3Config.hidden_size` 缺失 | AttributeError | Multimodal 嵌套 config，`hidden_size` 在 `text_config` 中 | Fallback: `config.text_config.hidden_size` |
| 3 | Unsloth 返回 Processor 非 Tokenizer | TypeError | Gemma 3 multimodal 模型需要 Processor | 解包: `processor.tokenizer` |
| 4 | `token_type_ids` 缺失 | RuntimeError | Gemma 3 训练模式需要此字段 | `torch.ones_like(input_ids)` |
| 5 | Dtype 不匹配 (bf16 vs f32) | 数值错误 | Hidden states 为 bf16，projection head 为 f32 | `control_states.float()` (保持 head 在 f32) |
| 6 | BCE dtype 不匹配 | RuntimeError | bf16 prediction vs f32 target | 显式 dtype 对齐 |
| 7 | `from_pretrained` 设备不一致 | 设备错误 | Projection head 在 CPU，base model 在 GPU | `.to(base_model.device)` |
| 8 | **Triton FlexAttention OOM** | CUDA OOM (shared memory) | Backward kernel 需要 114KB/sm，RTX PRO 6000 仅 101KB | `TORCHINDUCTOR_FLEX_ATTENTION=0` |

## Error #8 详解 (最终 Boss)

### 症状
```
RuntimeError: Triton Error: shared memory required (114 KB) exceeds 
device shared memory per SM (101 KB)
```

### 根因
PyTorch Inductor 编译器自动检测到 Gemma 3 的 attention pattern 适合 FlexAttention，生成 Triton kernel。但 FlexAttention backward kernel 需要 114KB shared memory per SM：

| GPU | Shared Memory / SM |
|-----|--------------------|
| H100 | 228 KB |
| RTX 4090 | 100 KB |
| **RTX PRO 6000 (Blackwell)** | **101 KB** |

### 修复
```bash
export TORCHINDUCTOR_FLEX_ATTENTION=0
```
必须在 `import torch` **之前**设置。`attn_implementation="eager"` 单独不够 (Inductor 仍然可能自动替换)。

已添加到 3 个文件: `test_sft_overfit.py`, `train_sft.py`, `train_dpo.py`

## 系统性观察

Blackwell sm_120 在 2026-06 的生态系统成熟度:
- bitsandbytes: 不支持
- Flash Attention 2: 无预编译 sm_120 wheel
- FlexAttention: Shared memory 太小
- Triton: 未针对 sm_120 调优

## 5 条 Blackwell PyTorch 建议

1. 不要用 bitsandbytes → bf16 全精度 (96GB 显存无需量化)
2. 在 `import torch` 前设置 `TORCHINDUCTOR_FLEX_ATTENTION=0`
3. 使用 `attn_implementation="eager"` 或 `"sdpa"`，绝不用 flash_attention_2
4. Projection head 保持 f32，将 hidden states cast 为 f32
5. 对 multimodal 模型 config 使用 `hasattr` / fallback

## 影响

所有 8 个错误已修复。过拟合测试在修复后成功通过。
