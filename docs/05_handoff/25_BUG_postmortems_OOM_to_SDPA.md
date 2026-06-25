# Bug Postmortems — OOM 五连杀到 SDPA 极速

**项目**: UAV-ISAC-MLLM (Gemma 3 12B + LoRA SFT on RTX PRO 6000 96GB)  
**时间跨度**: 2026-06  
**最终状态**: ✅ 2.54s/step, ~48GB 峰值, 纯 PyTorch + SDPA

---

## 总览

```
Bug #1: OOM — HF CausalLM 隐藏状态 + fp32 logits           → 省 30 GB
Bug #2: OOM — logits .contiguous() 拷贝                      → 省 8 GB
Bug #3: OOM — GQA log_softmax fp32 存储                      → 省 16 GB
Bug #4: OOM — F.cross_entropy fp32 梯度张量 (16 GB)          → Plan B (Unsloth CE)
Bug #5: CheckpointError — Unsloth 局部导入全局劫持           → Plan A (纯 PyTorch)
```

**核心经验**: Unsloth 与 Gemma 3 + SDPA 不可共存。任何形式的 `import unsloth`（全局或局部）都会劫持模型加载管线，强制降级到 eager attention。最终方案是**纯 PyTorch + bs=1/grad_accum=16**。

---

## Bug #1 — HF CausalLM 输出隐藏状态 + fp32 logits

**症状**: 训练启动即 OOM，GPU 分配失败。

**根因**: HuggingFace `AutoModelForCausalLM` 默认 `output_hidden_states=False` 不生效（Gemma3 实现 bug），仍产出所有 48 层 hidden states（~6 GB）。同时 HF wrapper 将 bf16 logits cast 到 fp32（~8 GB），完全无必要。

**修复** (`src/model/gemma_isac.py`):
```python
# 绕过 HF CausalLM wrapper, 直接调用 base_model
outputs = self.base_model.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=False,  # 明确关闭
    use_cache=False,
)
# 手动计算 lm_head (保持 bf16)
logits = self.base_model.lm_head(hidden_states)
```

**省**: ~14 GB (hidden_states 6GB + fp32 upcast 8GB)

**教训**: HuggingFace wrapper 在非标准模型上有多余开销。直接用 `base_model.model(...)` 绕过。

---

## Bug #2 — logits `.contiguous()` 拷贝

**症状**: 前向传播 OOM，大块 allocation 失败。

**根因**: `logits[:, :-1, :]` 对中间维切片产生非连续 stride。Unsloth 的 `fast_cross_entropy_loss` 在内部调用 `.view(batch*seq_len, d)`，要求 tensor 连续。`.contiguous()` 创建完整拷贝：bs=4 × 4096 × 256K × 2 bytes (bf16) ≈ 8.4 GB。

**修复**: 当时的方案是接受这个拷贝（砍掉 fp32 梯度 16 GB 后仍然有余量）。最终方案（Plan A）改用 `view(-1, V)` 直接传入 `F.cross_entropy`，同样需要 contiguous → `.contiguous()` 保留。

**省**: 无（必要拷贝），但后续优化省出空间让它不再是问题。

**教训**: 3D tensor 的中间维切片必然非连续。`.contiguous()` 是必须的代价。

---

## Bug #3 — GQA log_softmax fp32 存储

**症状**: Backward OOM，~16 GB allocation。

**根因**: Gemma 3 使用 Grouped Query Attention (GQA)，`F.log_softmax` 对 attention weights 上采样到 fp32 并存储整个 (B, heads, seq, seq) 矩阵用于 backward。bs=4 时 ~16 GB。

**修复** (`src/model/gemma_isac.py`):
```python
# 启用 gradient checkpointing
self.base_model.gradient_checkpointing_enable()
```

**省**: ~16 GB（不再存储中间激活）

**教训**: Large model training 必须开 gradient checkpointing。GQA 的 attention 中间张量比 MHA 更大（key-value 扩展）。

---

## Bug #4 — F.cross_entropy fp32 梯度张量 (Plan B 尝试)

**症状**: Backward 时 OOM，`F.cross_entropy` 内部分配 16 GB。

**根因**: `F.cross_entropy(logits, labels)` 对 256K 词表：
1. Forward: 计算 `log_softmax` → 存储 fp32 输出 (bs=4 × 4096 × 256K × 4 = 16 GB)
2. Backward: 分配 fp32 梯度张量 ∂L/∂logits (又 16 GB)

总计峰值 ~32 GB 仅 CE 相关。

**尝试修复 (Plan B)**:
```python
from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss
loss = fast_cross_entropy_loss(shift_logits, shift_labels)
```
Unsloth 的 Triton 内核逐 chunk 计算 CE，永远不生成完整 fp32 张量。

**结果**: ✅ 省了 16 GB，但引入了 Bug #5。

---

## Bug #5 — CheckpointError: Unsloth 时空悖论 ⭐ 最关键教训

**症状**:
```
torch.utils.checkpoint.CheckpointError:
Number of tensors saved during forward: 68
Number of tensors saved during recomputation: 65.
```

**根因时间线**:
```
1. Forward → 模型以纯净 HF 原生状态跑完
             grad checkpoint 记录: "保存了 68 个激活张量"

2. Compute Loss → losses.py 触发 from unsloth.kernels...
                  Unsloth 初始化 → 全�的 monkey-patch transformers 底层
                  包括 Gemma3 的 attention 实现被替换

3. Backward recompute → grad checkpoint 用已被替换的层重算 forward
                        产出: 65 个张量

4. Crash → 68 ≠ 65 → CheckpointError
```

**关键认知**: Unsloth 的 monkey-patch 是**全局且不可逆的**。不存在"局部借用" — 只要 Python 进程碰到 `import unsloth`（即使是函数体内的 `from unsloth.kernels...`），它就立即替换 `transformers` 的所有底层实现。gradient checkpointing 在 forward 和 recompute 之间看到了不同的模型代码。

**修复 (Plan A — 最终方案)**:
1. 彻底删除项目中所有 Unsloth 引用
2. `compute_sft_loss` 改为纯 PyTorch:
```python
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].clone()
if label_mask is not None:
    shift_labels[label_mask[:, 1:] == 0] = -100

loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
```
3. `bs: 4 → 1`, `grad_accum: 4 → 16`（数学等价，CE fp32 梯度 16GB → 4GB）

**结果**: ✅ 2.54s/step, ~48GB 峰值, 无 Unsloth / 无 CheckpointError / 无 OOM

**教训**: 
- **永远不要在训练循环的任何地方导入 Unsloth**。即使是在函数体内的局部导入，也会在 backward 时触发 gradient checkpointing 的不一致。
- Unsloth 的设计假设"全盘控制"：模型加载 + 训练 loop 都由它管理。Gemma 3 + SDPA 不在它的支持矩阵内。
- 对于不支持的模型架构，纯 PyTorch + 调 bs/grad_accum 是最可靠的路径。

---

## 最终架构

```
Model Loading:  Native HF AutoModel + PEFT LoRA + SDPA (2-3s/step)
Loss:           F.cross_entropy (纯 PyTorch, 无第三方依赖)
Memory:         bs=1, grad_accum=16, gradient_checkpointing=True
Peak VRAM:      ~48 GB / 96 GB
Speed:          2.54 s/step (1250 steps × 3 epochs ≈ 2.6 hours)
```

## 反模式清单

| 反模式 | 为什么是陷阱 | 替代方案 |
|--------|------------|---------|
| `import unsloth` 在训练脚本顶部 | 全局 monkey-patch, 强降 eager attention | 不用 Unsloth |
| `from unsloth.kernels...` 在函数体内 | 仍然触发全局 monkey-patch, 导致 CheckpointError | 不用 Unsloth |
| 不用 gradient checkpointing | GQA 中间激活 ~16 GB | `model.gradient_checkpointing_enable()` |
| 用 HF CausalLM wrapper 获取 logits | 多余 hidden_states + fp32 upcast | `base_model.model(...)` + 手动 lm_head |
| bs=4 直接 CE | fp32 梯度 16 GB | bs=1 + grad_accum=16 |
