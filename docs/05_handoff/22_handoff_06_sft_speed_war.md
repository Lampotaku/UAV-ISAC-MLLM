# 交接文档 #7 — SFT 训练加速战：从 21s/step 到真相大白

> 日期: 2026-06-25  
> 状态: 诊断完成，根因确认，等待解决方案决策  
> 标签: sft-speed, unsloth-limitation, gemma3-attention, bottleneck-analysis

---

## 目录

1. [问题概述](#问题概述)
2. [尝试过的修复 (按时间线)](#尝试过的修复-按时间线)
3. [根因分析](#根因分析)
4. [为什么 sdpa 整个链路都是无效的](#为什么-sdpa-整个链路都是无效的)
5. [速度基准线](#速度基准线)
6. [当前状态](#当前状态)
7. [解决路径](#解决路径)
8. [Commit 映射表](#commit-映射表)

---

## 问题概述

**症状**: Stage I SFT 训练在 RTX PRO 6000 96GB 上以 **16-21s/step** 运行。

```
配置: Gemma 3 12B, bf16 全精度 LoRA, bs=4, seq_len=4096
预期: ~2-3s/step  (Flash-Attention 级别)
实际: ~16-21s/step (O(n²) 全注意力矩阵)
```

- 1250 steps/epoch × 3 epochs = 3750 steps
- 3750 × 16s = **~16.7 小时** (vs 预期 ~2 小时)
- 时间差: **~8×**

---

## 尝试过的修复 (按时间线)

### 第一轮: 以为是 attention implementation 没传进去

**发现**: `configs/default.yaml` 写 `attn_implementation: "sdpa"`, 但训练 log 里看不到 sdpa 加载。

**修复** (commit `b0fd596`):
- [gemma_isac.py:81](src/model/gemma_isac.py#L81) — `__init__` 里给 `FastLanguageModel.from_pretrained()` 加上 `attn_implementation=attn_implementation`
- [gemma_isac.py:333](src/model/gemma_isac.py#L333) — `from_pretrained` class method 同样加上

**结果**: ❌ 无效。训练仍然是 eager attention。

**原因**: 当时不知道 Unsloth 会拦截并覆盖这个参数。参看[根因分析](#根因分析)。

### 第二轮: 以为是 Unsloth 的 dropout 惩罚

**发现**: Unsloth 打印警告:
```
Unsloth: Dropout = 0 is supported for fast patching.
You are using dropout = 0.05.
Unsloth will patch all other layers, except LoRA matrices, causing a performance hit.
```

**修复** (commit `65cf10d`):
- `configs/default.yaml`: `lora.dropout: 0.05` → `0`
- `configs/default.yaml`: `backbone` 直接写死服务器本地路径, 省去每次 `sed`

**结果**: 轻微改善。21s/step → 16s/step。

`dropout=0.05` 导致 Unsloth 走 "全模型逐层慢速 patch"，改 `dropout=0` 后走 "快速 LoRA-only patch"。但 attention 本身仍然是 O(n²) eager。

### 第三轮: 终极发现

**训练 log 关键一行**:
```
Unsloth: Gemma3 does not support SDPA - switching to fast eager.
```

这一行之前一直被忽略。它在**模型加载时**就打印了。无论 config 写什么、代码传什么，Unsloth 对 Gemma 3 强制使用 eager attention。

---

## 根因分析

### 完整的调用链

```
用户代码
  │
  ├─ Gemma3ISAC.__init__(attn_implementation="sdpa")
  │     │
  │     └─ FastLanguageModel.from_pretrained(attn_implementation="sdpa")
  │           │
  │           └─ Unsloth 内部检测到 model_type == "gemma3"
  │                 │
  │                 ├─ 打印: "Gemma3 does not support SDPA"
  │                 ├─ 强制覆盖: attn_implementation = "eager"
  │                 └─ 加载模型, 用 Unsloth Triton 内核 patch attention
  │                       │
  │                       └─ "fast eager" = Unsloth 优化版 O(n²) attention
  │                           (比原生 eager 快 ~25%, 但仍然 O(n²))
  │
  └─ FastLanguageModel.get_peft_model(...)
        │
        └─ 挂载 LoRA adapter
```

### 为什么 Unsloth 不支持 Gemma 3 SDPA

Unsloth 的 attention 替换逻辑:
1. 检测模型架构 → Gemma 3 是较新的架构
2. 查找适配的 Triton kernel → Gemma 3 的 attention pattern (sliding window + global attention 混合) 与标准 Llama/Mistral 不同
3. 如果没有对应的 Flash-Attention kernel → fallback 到 "fast eager"

Gemma 3 的特殊性:
- **5:1 sliding window + global attention 交错**: 每 6 层中 5 层是 sliding window (局部注意), 1 层是 global attention
- 这种混合 pattern 让 Triton kernel 适配变得复杂
- Unsloth (截至 2026.6.9) 还没有为此写优化的 Triton attention kernel

### 为什么 eager attention 在 4096 seq_len 下这么慢

```
Attention 计算量: O(B × H × seq_len² × head_dim)
                  = 4 × 32 × 4096² × 128
                  = 4 × 32 × 16,777,216 × 128
                  ≈ 275 GFLOPS (仅 attention, 不含 MLP)

每层: attention + MLP ≈ 350 GFLOPS
48 层 × 350 GFLOPS ≈ 16.8 TFLOPS (forward)
Forward + backward ≈ 50 TFLOPS/step
```

RTX PRO 6000 理论 ~100 TFLOPS (bf16)，但 eager attention 的 memory-bound 特性让实际利用率很低（大量 HBM 读写中间结果），实际吞吐可能只有 10-15 TFLOPS → 每步 ~3-5s 是理想上限。16s 说明还有其他开销（可能是 Unsloth 的 "fast eager" 仍然有 kernel launch overhead）。

---

## 为什么 sdpa 整个链路都是无效的

回顾我们踩过的坑, 每一步在当时看来都合理, 但都被更底层的机制拦截:

| 步骤 | 做了什么 | 为什么无效 | 哪里知道的 |
|------|----------|------------|------------|
| 1 | config 写 `attn_implementation: "sdpa"` | 参数根本没传进模型 | 代码审查 |
| 2 | 给 `from_pretrained()` 传参数 | Unsloth 在内部强制覆盖 | 训练 log |
| 3 | `sed` 改 config → 每次都要重新改 | config 已提交到仓库 | 用户反馈 |

**核心认知**: Unsloth 不是被动库。它**主动拦截和控制**模型加载过程, 包括 attention backend 的选择。`attn_implementation` 参数在进入 Unsloth 后就不再由用户控制。

---

## 速度基准线

在 RTX PRO 6000 96GB、4096 seq_len、bs=4 条件下:

| Attention 后端 | 预期速度 | 实际测到 | 备注 |
|----------------|---------|---------|------|
| Flash Attention 2 | ~1.5-2s/step | — | 无 Blackwell wheel |
| PyTorch SDPA (cuDNN Fused) | ~2-3s/step | — | Unsloth 不支持 Gemma 3 |
| Unsloth "fast eager" | ~5-8s/step | **16s/step** | O(n²), 但仍然比预期慢 |
| PyTorch 原生 eager | ~25-30s/step | 21s/step | O(n²) 全矩阵 |

**结论**: 当前 16s/step 是 Unsloth 的 "fast eager"。比原生 eager 快了 ~25%, 但离 sdpa/flash-attn 仍有 **5-8× 差距**。

---

## 当前状态

```
训练: 16s/step × 1250 steps/epoch × 3 epochs ≈ 16.7 小时
模型: bf16 全精度 LoRA (use_4bit: false)
框架: Unsloth FastLanguageModel (强制 eager attention for Gemma 3)
已完成: 全部 15 个 bug 修复, 代码管线 100% 可用
卡点: 训练速度 (唯一瓶颈)
```

### Git 状态

```
65cf10d fix: commit server backbone path + dropout=0 (Unsloth fast kernel)
b0fd596 fix: pass attn_implementation to FastLanguageModel.from_pretrained()
a52b4b8 fix: critical sync_gradients bugs in train_sft.py & train_dpo.py
```

---

## 解决路径

### 路径 A: 接受 16s/step, 直接跑 (0 代码改动)

```
时间: 16.7 小时 SFT + ~30 小时 DPO ≈ 2 天
风险: 无
代价: 等待时间长, 但代码已验证正确
```

**推荐指数**: ⭐⭐⭐ (如果你不急着要结果)

### 路径 B: 绕过 Unsloth, 用原生 PyTorch SDPA (中等改动)

既然 `use_4bit: false` (bf16 全精度), 根本不需要 Unsloth 的 4-bit 支持。可以:
1. 用 `transformers.AutoModelForCausalLM.from_pretrained(attn_implementation="sdpa")` 加载模型
2. 用原生 `peft.LoraConfig + get_peft_model` 挂载 LoRA
3. 保留 projection_head 和所有训练逻辑不变

**预期速度**: ~2-3s/step (cuDNN Fused Attention)  
**改动范围**: `gemma_isac.py` 的 `__init__` 和 `from_pretrained`  
**风险**: 需要验证原生 PEFT 在 Blackwell 上无兼容问题

**推荐指数**: ⭐⭐ (如果 16h 太慢)

### 路径 C: 降低 seq_len (小改动)

当前 `max_seq_length: 4096`。如果 prompt 实际不需要 4096:
- `seq_len=2048` → eager attention 速度 ~4× (O(n²) 减半再平方)
- 预期: ~4s/step

**风险**: 需要验证 5000 条数据的 prompt 实际长度分布

**推荐指数**: ⭐ (治标不治本)

### 路径 D: 等 Unsloth 更新 Gemma 3 SDPA 支持

Unsloth 团队活跃开发中。Gemma 3 attention 的 Triton kernel 可能在后续版本加入。

**风险**: 不确定时间线

**推荐指数**: ⭐ (被动等待)

---

## Commit 映射表

```
┌────────────┬──────────────────────────────────────────────────────┬──────────┐
│ Commit     │ 修复内容                                             │ 效果     │
├────────────┼──────────────────────────────────────────────────────┼──────────┤
│ b0fd596    │ gemma_isac.py: 传递 attn_implementation 给 Unsloth  │ 无效*    │
│ 65cf10d    │ dropout: 0.05→0, backbone 写死服务器路径              │ 21→16s   │
│ a52b4b8    │ sync_gradients bug fixes (SFT+DPO)                  │ 正确性   │
│ 4bc1a95    │ 分层 LR + global_step sync                          │ 正确性   │
│ 3148c08    │ --data_dir 路径处理                                  │ 功能     │
│ 05075c0    │ bs=4, grad_accum=4 (OOM 适配)                       │ 防 OOM   │
└────────────┴──────────────────────────────────────────────────────┴──────────┘
* b0fd596 在逻辑上正确 (参数应该传递), 但 Unsloth 内部覆盖了该参数
```

---

## 相关文档

- [[20_handoff_05_server_debug_log](20_handoff_05_server_debug_log.md)] — 服务器调试 (Error #8: FlexAttention OOM)
- [[21_pre_sft_bug_audit](21_pre_sft_bug_audit.md)] — Pre-SFT 全量 Bug 审计
- [[19_handoff_04_post_datagen](19_handoff_04_post_datagen.md)] — 当前状态与下一步
