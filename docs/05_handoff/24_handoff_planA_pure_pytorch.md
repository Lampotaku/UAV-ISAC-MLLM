# Handoff #24 — Plan A: 彻底肃清 Unsloth, 纯 PyTorch CE

**日期**: 2026-06-26  
**上接**: [#23 OOM#4 Unsloth Chunked CE](23_handoff_oom4_unchunked_ce.md)  
**下接**: 待服务器验证

## 问题: CheckpointError — Unsloth "局部借用"不可能

### 现象

```
torch.utils.checkpoint.CheckpointError: A different number of tensors was saved
during the original forward and recomputation.
Number of tensors saved during forward: 68
Number of tensors saved during recomputation: 65.
```

### 根因

即使只是**局部导入** (`from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss`，在 `losses.py` 函数体内)，Unsloth 一旦被 `import` 就会全局 monkey-patch transformers 底层实现。

时间线:
1. **Forward**: 模型以纯净 HF 原生状态跑完 → grad checkpoint 记录 68 个激活张量
2. **Compute loss**: `losses.py` 中 `from unsloth.kernels...` 触发 Unsloth 初始化 → 强换 Gemma3 注意力层
3. **Backward recompute**: grad checkpoint 用被替换的层重算 forward → 只产出 65 个张量
4. **Crash**: 68 ≠ 65 → `CheckpointError`

**结论**: Unsloth 不存在"局部借用"。只要你 `import` 它，就全盘劫持。与 Gemma 3 + SDPA 不可共存。

## 方案 A — 纯 PyTorch + 降 bs 提 grad_accum

### 核心思路

- 完全删除项目中所有 Unsloth 引用
- 用纯 `F.cross_entropy` 计算 SFT loss
- 通过 `bs=1, grad_accum=16` 控制单步显存 (fp32 CE 梯度从 16GB → 4GB)
- 数学上 `bs=1×16` ≡ `bs=4×4` (梯度更新完全一致)

### 改动清单

| 文件 | 改动 |
|------|------|
| `src/model/losses.py` | 删除 `_grad_ckpt` import + `_ce_none` 函数 + Unsloth try/except; `compute_sft_loss` 改为纯 `F.cross_entropy(view(-1,V), view(-1), ignore_index=-100)` |
| `configs/default.yaml` | `per_device_batch_size: 4→1`, `gradient_accumulation_steps: 4→16` |
| `src/training/train_sft.py` | 更新 docstring 显存估算 + 防爆盾注释 |
| `src/training/train_dpo.py` | 更新防爆盾注释 |
| `scripts/test_sft_overfit.py` | 更新防爆盾注释 |

### 不动的代码

- `src/model/gemma_isac.py` — `use_4bit=True` 分支保留 `from unsloth import FastLanguageModel`，仅在 `use_4bit=True` 时执行 (当前 `false`，永不触发)
- DPO `_grad_ckpt` (train_dpo.py) — PyTorch 原生，无 Unsloth 依赖，安全

### 显存估算

| 组件 | bs=1 |
|------|------|
| bf16 LoRA 模型 | ~24 GB |
| AdamW 状态 (embed_tokens) | ~8 GB |
| logits bf16 (1×4096×256K) | ~2.1 GB |
| CE fp32 log_softmax 中间 | ~4.2 GB |
| 激活梯度 | ~5 GB |
| 其他 (last_hidden_state, grad_embed, ...) | ~5 GB |
| **峰值** | **~48 GB** (96GB 内 ~48GB 余量) |

## 服务器更新步骤

```bash
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm

# 过拟合测试 (验证代码正确)
python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000

# 全量 SFT
python src/training/train_sft.py --config configs/default.yaml --data_dir /root/autodl-tmp/data/full5000
```

## 预期

- 日志中 **不再**出现 `🦥 Unsloth: Will patch your computer`
- 不再有 `CheckpointError`
- 速度: **~2-3s/step** (SDPA 全速)
- bs=1 × grad_accum=16 → 有效 batch=16
