---
type: postmortem
status: resolved
stage: sft
severity: P0
last_updated: 2026-06-26
related: [oom_incidents, adr_001_unsloth_removal, adr_003_sdpa_canonical]
---

# SFT 速度战 — 从 21s/step 到 2.5s/step

**来源**: 交接文档 #22 (handoff_06_sft_speed_war.md) | **最终状态**: 已解决 (Plan A)

## 症状

SFT 训练以 16-21s/step 运行，预期为 ~2-3s/step。约 8x 差距。

## 根因诊断

### 发现: Unsloth 强制 eager attention

关键日志行:
```
Unsloth: Gemma3 does not support SDPA - switching to fast eager.
```

**根因**: Gemma 3 使用 5:1 sliding window + global attention 交错的 attention pattern。Unsloth 没有对应的 Triton kernel，强制将 `attn_implementation` 覆盖为 `"eager"`。

### 影响量化

- **Unsloth "fast eager"**: 比原生 eager 快 ~25%，但仍为 O(n²) 复杂度
- **SDPA**: O(n²) 但 memory-efficient, >8x faster
- **Dropout=0.05 额外减速**: 从 21s → 16s (设为零后)，因为 eager attention 中 dropout 增加 kernel launch overhead

### Attention 计算量

```
单个 attention: ~275 GFLOPS
48 层 forward:  ~16.8 TFLOPS
backward:       ~50 TFLOPS
```

在 RTX PRO 6000 (~100 TFLOPS fp16 理论值) 上，SDPA 的 memory-efficient 实现可以在 2-3s 内完成，而 eager 需要 16-21s。

## 尝试的解决方案

### 路径 A: 接受 16s/step
- 3 epochs × 5000 samples = 15000 steps × 16s = ~67h
- **不可接受**

### 路径 B: 绕过 Unsloth，原生 PyTorch + PEFT SDPA
- 风险: 当时不确定原生 PEFT 在 Blackwell 上的兼容性
- 推荐度: 2/5 (当时)

### 路径 C: 降低 seq_len
- 风险: 响应截断，训练数据质量受损
- 推荐度: 1/5

### 路径 D: 等待 Unsloth 更新
- 风险: 时间不确定
- 推荐度: 1/5

## 最终解决

路径 B 被采纳并验证成功。通过 Plan A (彻底移除 Unsloth)，实现了:

```
21s/step (Unsloth eager) → 2.5s/step (纯 SDPA) = 8.4x 提速
```

关键验证:
1. Native HF `AutoModel` + `peft` 的 `get_peft_model` 在 Blackwell 上工作正常
2. SDPA 在 Gemma 3 的 5:1 sliding window attention 上正确运行
3. 4-bit QLoRA 加载仍通过 Unsloth 完成（仅 `from_pretrained`），但训练路径 100% 纯净 PyTorch

后续 bs=1→2 升级进一步将 epoch 吞吐提升 18% (代价是 48GB→76GB VRAM)。

详见:
- [oom_incidents.md](oom_incidents.md) — OOM 修复全链
- [06_decisions/adr_001_unsloth_removal.md](../06_decisions/adr_001_unsloth_removal.md) — Plan A 决策
- [06_decisions/adr_003_sdpa_canonical.md](../06_decisions/adr_003_sdpa_canonical.md) — SDPA 决策
