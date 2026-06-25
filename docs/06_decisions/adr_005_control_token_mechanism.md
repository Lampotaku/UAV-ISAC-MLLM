---
type: decision
status: accepted
stage: all
last_updated: 2026-06-23
related: [problem_formulation, training_pipeline]
---

# ADR-005: Control Token Mechanism

## Status

**Accepted** (2026-06-23). 已实现为模型架构的核心组件。

## Context

MLLM 需要输出 176 个优化变量 (ΔQ, A, P_c, P_s)。如果直接通过文本输出这些浮点数，需要 ~890 tokens 且受 tokenizer 精度限制。

需要一种方法将 Gemma 3 的 hidden states 直接映射到连续优化变量，绕过文本解码器。

## Decision

使用 **Control Token** 机制: 8 个特殊 token `<ctrl_0>` 到 `<ctrl_7>` 插入在 prompt 末尾。这些 token 通过 Gemma 3 处理后在特定位置产生 hidden states，通过 f32 projection head 映射到优化变量。

```
prompt_text + [<ctrl_0>, ..., <ctrl_7>]
         ↓ Gemma 3 forward
hidden_states at control positions (bs, 8, d_model)
         ↓ Projection Head (f32)
Proj_Q → ΔQ̂ (12 floats)     [clipping + tanh]
Proj_A → Â (80 ints)         [Sinkhorn, 20 iters]
Proj_P → P̂ (84 floats)       [Softmax + budget scaling]
```

**设计理由**:
1. **绕过 tokenizer**: 浮点数不需要通过文本 → 避免 BPE 碎片化 (见 response_token_overflow bug)
2. **连续表示**: Hidden states 可以编码任意精度的连续值
3. **端到端可微**: Projection head 的梯度通过 hidden states 回流到 transformer
4. **约束投影**: 每个 head 内置约束满足 (Sinkhorn for association, softmax for power, tanh for displacement)

## Token 数量分析

选择 8 个 control tokens（而非 176 个 = 每变量一个）:
- 8 个 tokens × 3840 hidden dim = 30720 维 → projection head 压缩到 176 维
- 足够的容量编码 176 个变量
- 极小的 overhead (8 tokens in 3456 seq = 0.23%)

## Token 嵌入初始化

Control tokens 的嵌入从 Gemma 3 的 `embed_tokens` 权重中以正态分布初始化，而非零初始化，确保初始 forward 有合理的梯度信号。

## Consequences

### Positive
- 绕过 text decoder 的精度和 token 数量限制
- Projection head 内置物理约束，确保输出始终可行
- 端到端可微，梯度流通畅
- 低 overhead (8/3456 tokens)

### Negative
- 引入额外的 f32 参数 (~11.8M for projection head)
- 需要从 hidden states 中精确提取 control token 位置 (off-by-one 风险)
- 训练需要额外的 L_ctl 辅助 loss 确保 control tokens 被有效利用
- 不能用于纯文本生成 — 仅用于结构化预测

## Alternatives Considered

### Text-based Regression
- MLLM 以文本输出浮点数 → tokenizer 解析
- **拒绝原因**: BPE 碎片化使 176 个 float 膨胀到 ~890-1678 tokens，精度损失

### Per-Variable Token
- 176 个 control tokens = 每变量一个
- **拒绝原因**: Overhead 太大 (176/3456 = 5%)，信息冗余

## References
- [01_architecture/problem_formulation.md](../01_architecture/problem_formulation.md)
- [03_bugs/resolved/response_token_overflow.md](../03_bugs/resolved/response_token_overflow.md)
- Doc 08 (pre-launch technical report) — Control token 规范
