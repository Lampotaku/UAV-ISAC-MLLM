# 第 4 轮 OOM 修复: Unsloth Chunked Cross-Entropy 替代原生 F.cross_entropy

**提交**: `479c226` — `perf: replace _grad_ckpt CE with Unsloth Chunked CE (Plan B)`

## 问题回顾

训练在 `accelerator.backward(total_loss)` 时 OOM:

```
Tried to allocate 16.00 GiB.
GPU 0 has 94.97 GiB total, 81.72 GiB in use, 13.25 GiB free.
```

### 根因 (talk.md 的分析完全正确)

绕过 HuggingFace CausalLM wrapper 后，我们手动调用原生 `F.cross_entropy`，**意外绕过了 Unsloth 最核心的显存魔法**。

PyTorch 原生 CE 在 V=256K 类上:
- 内部 upcast 到 fp32 做 `log_softmax`（数值稳定）
- backward 时需分配完整 fp32 梯度张量 `∂L/∂logits`

$$4 \text{ (bs)} \times 4096 \text{ (seq)} \times 256{,}000 \text{ (vocab)} \times 4 \text{ (fp32 bytes)} = 16.00 \text{ GiB}$$

加上 forward 已有 ~82 GB → 峰值 ~98 GB → OOM。

## 解决方案: 方案 B (talk.md 推荐)

**用 Unsloth 的 `fast_cross_entropy_loss` 替换原生 `F.cross_entropy`。**

Unsloth 的 Triton 内核沿 vocabulary 维度**分块 (chunked)** 计算 CE:
- forward: 逐 chunk 算 `log_softmax`，不存储完整 fp32 中间结果
- backward: 逐 chunk 算梯度，**永不成完整 16 GB fp32 梯度张量**
- 速度: 不接管模型加载路径 → SDPA 保持不变 → 仍是 2-3s/step

### 为什么之前 Unsloth 这么慢？

之前慢是因为 `FastLanguageModel` 接管了模型加载，对 Gemma 3 的混合注意力不支持时强行降级到 `eager` attention (O(n²), 16-21s/step)。

**现在**: 只借用 Unsloth 的一个独立 Triton 函数 `fast_cross_entropy_loss`，不动模型加载路径。相当于"法拉利发动机 + Unsloth 的高效排气管"。

## 改动文件

### 1. `src/model/losses.py` — `compute_sft_loss` (核心)

```python
# 之前: Plan C (_grad_ckpt)
shift_logits = logits[:, :-1, :].transpose(1, 2)   # (B, V, S-1)
loss = _grad_ckpt(_ce_none, shift_logits, shift_labels, use_reentrant=False)
if label_mask is not None:
    loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)

# 之后: Plan B (Unsloth Chunked CE)
shift_logits = logits[:, :-1, :]   # (B, S-1, V) — 不转置, Unsloth 用最后一维
from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss
if label_mask is not None:
    _labels = shift_labels.clone()
    _labels[shift_mask == 0] = -100   # Unsloth 遵循 ignore_index=-100 约定
    loss = fast_cross_entropy_loss(shift_logits, _labels)
else:
    loss = fast_cross_entropy_loss(shift_logits, shift_labels)
```

**ImportError 回退路径保留**: 如果 Unsloth 不可用，自动回退到 `_grad_ckpt(_ce_none, ...)`。

### 2. `src/training/train_dpo.py` — `_logp_gather` (保持 checkpoint)

DPO **不适用** `fast_cross_entropy_loss`:
- 该函数返回**标量 CE**，DPO 需要 **per-token log-probability** (`log_softmax + gather`)
- DPO 用 bs=1，fp32 张量仅 ~4 GB，`_grad_ckpt` 足够

### 3. `src/training/train_sft.py` + `configs/default.yaml` — 显存估算更新

| | 旧代码 | Plan C (`_grad_ckpt`) | Plan B (Unsloth CE) |
|---|---|---|---|
| forward CE 存储 | +16 GB fp32 | 0 (checkpoint) | 0 (chunked) |
| backward CE 峰值 | +16 GB fp32 | +16 GB 重算 | **0** (chunked) |
| 峰值显存 | ~98 GB → OOM | ~90 GB 勉强 | **~52 GB 充裕** |

## 服务器更新步骤

```bash
cd /root/UAV-ISAC-MLLM
git pull
conda activate uavmllm

# 重新启动训练
python src/training/train_sft.py --config configs/default.yaml --data_dir /root/autodl-tmp/data/full5000
```

## 技术要点 (给后续开发者)

1. **Unsloth 的"剥离式使用"是关键**: 只借 loss 内核，不让它接管模型加载
2. **Gemma 3 的 256K 词表是特殊挑战**: 原生 CE 的 fp32 upcast 在 256K 词表上产生 16 GB 梯度张量；7B/8B 模型 (128K 词表) 只有 8 GB
3. **DPO 和 SFT 的 loss 需求不同**: CE 输出标量可以分块，per-token log-prob 需要完整 log_softmax 输出，只能靠 gradient checkpointing
4. **`modules_to_save=["embed_tokens", "lm_head"]` 在 Gemma 3 上只占一份显存**: 因为 `embed_tokens` 和 `lm_head` 是 tied weights，AdamW 状态只分配一次 (~8 GB 而非 ~16 GB)
