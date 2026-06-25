---
type: postmortem
status: resolved
stage: sft
severity: P0
commits: [68c2567, 8c3b2a8, fe0c34a, 666e23f]
last_updated: 2026-06-26
related: [speed_optimization, adr_001_unsloth_removal, sft_live]
---

# OOM 五连杀 — 从 16GB×5 到 2.5s/step

**来源**: Docs 23 (Plan B), 24 (Plan A), 25 (postmortem) | **最终状态**: 全部解决

## Bug Chain 全景

```
OOM #1 → OOM #2 → OOM #3 → OOM #4 → Bug #5
  ↓         ↓         ↓         ↓         ↓
HF wrapper  contiguous  GQA fp32   CE grad  CheckpointError
(~14 GB)    (~8 GB)    (~16 GB)  (~16 GB)  (Plan B → Plan A)
```

## Bug #1: HF CausalLM 隐藏状态 + fp32 logits (~14 GB)

**症状**: `Gemma3ForCausalLM` (wrapper) 存储所有层 hidden states (fp32) + 最终 logits (fp32)

**根因**: HF 的 CausalLM wrapper 为 generation 设计，保留了不必要的中间张量

**修复**: 
```python
# 之前: model = Gemma3ForCausalLM.from_pretrained(...)
# 之后: 绕过 wrapper, 直接调用 base model
outputs = model.base_model.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=True,
)
hidden_states = outputs.last_hidden_state
logits = model.lm_head(hidden_states)  # 手动 lm_head
```

**节省**: ~14 GB (不再缓存 fp32 logits + hidden states)

**Commit**: `68c2567`

## Bug #2: logits .contiguous() 拷贝 (~8 GB)

**症状**: CE loss 计算前的 `.contiguous()` 调用产生额外 8.4 GB 拷贝

**根因**: 3D tensor 的中间维切片 (`logits[:, control_positions, :]`) 必然非连续，`.contiguous()` 是 PyTorch CE 的要求

**修复**: 无法消除 — 这是必要的拷贝。CE 要求连续的输入张量。

**节省**: 不可消除，接受为必要代价

**Commit**: `8c3b2a8` (性能优化，非消除)

## Bug #3: GQA log_softmax fp32 存储 (~16 GB)

**症状**: Grouped Query Attention (GQA) 的 `log_softmax` 产生大量 fp32 中间张量

**根因**: Gemma 3 使用 GQA (5:1 sliding window + global attention 交错)，attention 中间张量比 MHA 大得多。`log_softmax` 在 fp32 中存储完整 attention 矩阵 (~16 GB for 4096 seq)。

**修复**: 
```python
model.gradient_checkpointing_enable()
```

**节省**: ~16 GB (activations 不存储, backward 时重计算)

**Commit**: `fe0c34a`

## Bug #4: F.cross_entropy fp32 梯度 (~16 GB)

**症状**: `F.cross_entropy(logits, labels)` 产生 16 GB fp32 梯度张量

**根因**: PyTorch 在 CE 内部将 logits upcast 为 fp32:
```
4 (bs) × 4096 (seq) × 256,000 (vocab) × 4 (fp32 bytes) = 16.38 GiB
```
Gemma 3 的 256K 词表 (vs 7B/8B 的 128K) 是特殊挑战。

**Plan B (失败)**: 使用 Unsloth 的 `fast_cross_entropy_loss` (Triton 分块 CE)
- 节省 16 GB ✓
- 但引入 **Bug #5**: Unsloth import 的全局 monkey-patch 导致 grad checkpoint 失败

**最终修复 (Plan A)**: 降低 batch size
```
bs: 4 → 1, grad_accum: 4 → 16
fp32 梯度: 16.38 GB → 4.10 GB (线性缩放)
```

**Commit**: `666e23f` (Plan A)

## Bug #5: CheckpointError — Unsloth Monkey-Patch

**症状**: 
```
RuntimeError: torch.utils.checkpoint: Recomputed one more tensor 
than originally saved (68 vs 65)
```

**根因 — 时空悖论**:
1. Forward: 用纯净 HF 原生路径, grad checkpoint 记录 68 个激活张量
2. Loss 计算: `from unsloth.kernels import fast_cross_entropy_loss` → Unsloth 全局替换 attention 层
3. Backward: recompute 走被替换的层, 产出 65 个张量
4. 68 ≠ 65 → CheckpointError

**关键教训**: **Unsloth 不存在"局部借用"**。即使 `import` 在函数体内、仅用于独立 kernel，Unsloth 仍然全局 monkey-patch transformers 底层。与 Gemma 3 + SDPA + grad checkpoint 不可共存。

**最终修复**: Plan A — 彻底删除所有 Unsloth 引用，使用纯 PyTorch CE + bs=1

## 最终架构

```
纯 PyTorch (0 Unsloth)
  ├── HF AutoModel (Gemma 3 12B) + PEFT LoRA (r=16, α=32)
  ├── SDPA attention (2-3s/step)
  ├── F.cross_entropy (纯 PyTorch, 无 Triton kernel)
  ├── bs=1, grad_accum=16 → effective bs=16
  ├── gradient_checkpointing=True
  └── 最终结果: 2.5s/step, ~48GB/96GB
```

bs 后续升级至 2 (见 [sft_live.md](sft_live.md))，峰值 ~76GB。

## 反模式清单

| # | 反模式 | 替代 |
|---|--------|------|
| 1 | 假设 HF wrapper 零开销 | 直接调用 `base_model.model` |
| 2 | 忽略 `.contiguous()` 对 stride 内存的影响 | 接受必要代价，或重排算子 order |
| 3 | 不开 gradient checkpointing | 大模型训练必须开 |
| 4 | 未检查 vocab_size 对 CE 显存的影响 | 256K vocab → 4× vs 128K |
| 5 | 依赖 Unsloth 的"局部功能" | Unsloth 不存在局部 — 全有或全无 |
| 6 | 在同一个 process 中混用 Unsloth 和 PyTorch 原生路径 | 选择一条路径并坚持 |
| 7 | 在 monkey-patch 框架上依赖 grad checkpoint | 先验证 recompute 一致性 |

## 演进总结

| 阶段 | bs | attention | CE 方案 | Unsloth? | 速度 | 峰值 VRAM |
|------|----|-----------|---------|----------|------|-----------|
| OOM #1-3 | 4 | eager | 原生 | ✅ | 21s | OOM |
| Plan B (#4) | 4 | eager | Unsloth chunked | ✅ (CE only) | 18s | OOM + Bug #5 |
| Plan A | 1 | SDPA | 原生 | ❌ | 2.5s | ~48 GB |
| **Current** | **2** | **SDPA** | **原生** | **❌** | **4.1s** | **~76 GB** |
