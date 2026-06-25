# 交接文档 #5 — 服务器调试日志：从 0 到过拟合测试通过

> 时间段: 2026-06-25  
> 状态: 过拟合测试通过 ✅ → 准备 Stage I SFT 训练  
> 标签: 服务器调试, Blackwell RTX 5090, 8 个 Runtime Error 复盘

---

## 目录

1. [背景](#背景)
2. [错误全景图](#错误全景图)
3. [逐错误详解](#逐错误详解)
4. [系统性问题回顾](#系统性问题回顾)
5. [当前状态与下一步](#当前状态与下一步)

---

## 背景

我们在做 UAV-ISAC-MLLM 项目：用 Gemma 3 12B 大模型 + LoRA 微调，给无人机通信/感知协同优化 (SCA-FP 数值求解器) 提供智能热启动。

之前已经完成了：
- 全部源码 (18 files, ~4200 lines Python)
- 5000 环境训练数据生成 (SFT: 5000, DPO: 186,896)
- 7 轮代码审查闭合
- feature/multiprocessing → master 合并

当前卡点：在 AutoDL RTX 5090 服务器上运行**过拟合测试**（5 样本 × 200 步，验证训练管线是否正确），结果连续爆了 8 个 runtime error。

硬件环境：
| 项 | 值 |
|----|-----|
| GPU | RTX 5090 32GB (Blackwell sm_120) |
| CUDA | 12.8 |
| Python | 3.11 |
| 量化 | Unsloth 4-bit QLoRA (bitsandbytes 不支持 Blackwell) |
| 模型 | Gemma 3 12B Instruct |

---

## 错误全景图

```
┌──────────────────────────────────────────────────────────────┐
│ #1  模型加载        Unsloth 版本太旧，不支持 Gemma 3          │
│ #2  模型 init       Gemma3Config 无 hidden_size 属性          │
│ #3  Tokenizer       Gemma3Processor 包裹了 tokenizer          │
│ #4  Forward         token_type_ids 缺失 (Gemma 3 必需)        │
│ #5  Forward         bf16/f32 dtype 不匹配 (投影头 × 隐藏态)   │
│ #6  Loss            BCE dtype 不匹配 (bf16 预测 × f32 标签)  │
│ #7  from_pretrained 投影头加载在 CPU，base_model 在 GPU       │
│ #8  Backward        Triton FlexAttention 共享内存溢出         │
│     (循环 2 次)     114KB 需要 > 101KB 硬件限制               │
└──────────────────────────────────────────────────────────────┘
```

每个错误都发生在上一个修复之后——典型的 "打地鼠" 式调试。根本原因：**Blackwell (sm_120) 太新，整个软件栈 (Unsloth/PyTorch/Triton/HuggingFace) 都有兼容性暗坑**。

---

## 逐错误详解

### Error #1: Unsloth 版本太旧

```
NotImplementedError: Unsloth: google/gemma-3-12b-it is not supported
```

**根因**: 服务器上的 Unsloth 版本不支持 Gemma 3 架构。

**修复**: 从 GitHub 源码升级 Unsloth：
```bash
pip install --upgrade unsloth unsloth-zoo
```

**耗时**: ~5 分钟 (下载 + 编译)

---

### Error #2: hidden_size 属性缺失

```
AttributeError: 'Gemma3Config' object has no attribute 'hidden_size'
```

**根因**: Gemma 3 的多模态 config 结构是嵌套的——`hidden_size` 不在 `config` 顶层，而在 `config.text_config.hidden_size`（因为多模态模型有独立的 text/vision config）。

**修复** (`gemma_isac.py`):
```python
# 修复前
hidden_dim = config.hidden_size

# 修复后
if hasattr(config, "hidden_size"):
    hidden_dim = config.hidden_size
elif hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
    hidden_dim = config.text_config.hidden_size
```

**教训**: 不同模型的 config 结构不同。Gemma 3 是多模态架构，即使只用 text 部分，config 结构也是嵌套的。

---

### Error #3: Processor 包裹 Tokenizer

```
AttributeError: 'Gemma3Processor' object has no attribute 'add_tokens'
```

**根因**: `FastLanguageModel.from_pretrained()` 对于 Gemma 3 返回的不是裸 tokenizer，而是 `Gemma3Processor`（一个包装器，包含 tokenizer + image processor）。Processor 没有 `add_tokens` 方法。

**修复** (`gemma_isac.py`):
```python
# 解包真正的 tokenizer
if hasattr(tokenizer_or_processor, 'tokenizer'):
    self.tokenizer = tokenizer_or_processor.tokenizer
else:
    self.tokenizer = tokenizer_or_processor
```

**教训**: Unsloth 对多模态模型的返回类型和单模态模型不同。解包是必需的。

---

### Error #4: token_type_ids 缺失

```
ValueError: token_type_ids is required as a model input when training
```

**根因**: Gemma 3 在训练模式下强制要求 `token_type_ids`（用于区分 text vs image token，以及构建 causal mask）。纯文本场景下，所有 token 都是 type=1 (text)。

**修复** (`gemma_isac.py:forward()`):
```python
# Gemma 3 text-only: 所有位置都是 type 1 (text)
token_type_ids = torch.ones_like(input_ids)
```

**教训**: 多模态模型即使在纯文本模式下也有额外的输入要求。`attention_mask=0` 的位置默认是 type 0 (padding)，`attention_mask=1` 的位置需要指定 type 1 (text)。

---

### Error #5: dtype 不匹配 — bf16 × f32

```
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

**根因**: 这是一个**架构决策失误**：
- Base model 隐藏态是 **bf16**（模型 dtype）
- 投影头参数是 **f32**（为了数值精度）
- `bf16 @ f32.weight` → PyTorch 报错

**错误修复** (已撤销): 把投影头也转成 bf16 → 引发 Error #6

**正确修复** (`gemma_isac.py:forward()`):
```python
# 投影头保持 f32，隐藏态向上 cast
prior_hat = self.projection_head(control_states.float(), q_current)
```

**原则**: **训练目标 (标签) 是 f32 → 投影头应该保持 f32 → 输入向上 cast**。不要为了匹配 dtype 而降低投影头的精度。

---

### Error #6: BCE dtype 不匹配

```
RuntimeError: Found dtype Float but expected BFloat16
```

**根因**: Error #5 的错误修复把投影头变成了 bf16 → 输出 bf16 预测，但目标是 f32 → `F.binary_cross_entropy(bf16, f32)` 报错。

**修复** (`losses.py`): 在 loss 计算时显式统一 dtype：
```python
common_dtype = torch.float32
dq_hat = delta_hat["delta_q"].to(dtype=common_dtype)
da_hat = delta_hat["delta_a"].to(dtype=common_dtype)
# ...
```

**教训**: **不要修 symptom，修 root cause**。Error #5 的正确方案是 cast 输入而不是 cast 参数。

---

### Error #7: from_pretrained 设备不匹配

```
(如果走到 DPO 阶段会触发)
RuntimeError: Expected all tensors to be on the same device, but found cuda:0 and cpu
```

**根因**: `Gemma3ISAC.from_pretrained()` 手动构造实例：
1. `projection_head` 从 `torch.load(..., map_location="cpu")` 加载 → 在 CPU
2. `base_model` 由 Unsloth 加载 → 在 GPU
3. Forward 时 `control_states` (GPU) 送进 `projection_head` (CPU) → 报错

**修复** (`gemma_isac.py:from_pretrained()`):
```python
instance.projection_head = projection_head.to(base_model.device)
```

**教训**: `cls.__new__(cls)` + 手动 `nn.Module.__init__()` 绕过了 `__init__` 的设备管理逻辑。`model.to(device)` 只对已注册的 module 生效，手动 setattr 的不算。

---

### Error #8 (最终 Boss): Triton FlexAttention 共享内存溢出

```
torch._inductor.exc.InductorError:
RuntimeError: No valid triton configs.
OutOfMemoryError: out of resource: triton_tem_fused_flex_attention_backward_mul_1
Required: 114688  Hardware limit: 101376
Reducing block sizes or `num_stages` may help.
```

**根因**: 这是整个调试过程中最深的一个坑，需要理解整个软件栈：

```
用户代码
  ↓ attn_implementation="flash_attention_2" (或 "eager")
HuggingFace / Unsloth
  ↓ 加载模型, 可能应用 torch.compile
PyTorch Inductor
  ↓ 自动检测 attention 模式 → 替换为 FlexAttention
Triton
  ↓ 编译 FlexAttention backward kernel
  ↓ 需要 114KB 共享内存
RTX 5090 Blackwell (sm_120)
  ↓ 只有 101KB 共享内存
  ✗ BOOM — OOM
```

关键认知：
- **Hopper (H100)**: 228KB shared memory per SM → FlexAttention OK
- **Ada (RTX 4090)**: 100KB shared memory per SM → FlexAttention 勉强
- **Blackwell (RTX 5090)**: 101KB shared memory per SM → **不够！**

Blackwell 的共享内存比 Hopper **小了 56%**，但 FlexAttention backward kernel 的 shared memory 需求是固定的 (~114KB)。这是 PyTorch/Triton 对 Blackwell 的适配不完善。

**第一次尝试** (失败): 改 `attn_implementation: "flash_attention_2"` → `"eager"`

以为 eager attention 就不会触发 FlexAttention。结果 Inductor 仍然自动检测 attention pattern 并替换。

**第二次尝试** (成功): 在 `import torch` **之前**设置环境变量：
```python
os.environ["TORCHINDUCTOR_FLEX_ATTENTION"] = "0"
```

这从根本上禁止 PyTorch Inductor 使用 FlexAttention。

同步应用到 3 个文件：
- `scripts/test_sft_overfit.py` (过拟合测试)
- `src/training/train_sft.py` (Stage I SFT)
- `src/training/train_dpo.py` (Stage II DPO)

**教训**: 在新硬件上，环境变量级的手术比配置级的手术更可靠。`TORCHINDUCTOR_FLEX_ATTENTION=0` 是一个深藏于 PyTorch 内部的环境变量，但它是解决 Blackwell attention 问题的关键。

---

## 系统性问题回顾

### 问题 1: "打地鼠" 调试

8 个错误是一个接一个暴露的。每修一个，下一个才露出来。为什么审查没预判到？

- Error #2–#4 是 **API 兼容性**问题 — 需要实际运行才能暴露
- Error #5–#6 是 **dtype 推理**问题 — 静态审查很难追踪混合精度
- Error #7 是 **代码路径**问题 — `from_pretrained` 在当时还没被调用过
- Error #8 是 **硬件特定**问题 — 只在 Blackwell 上出现

### 问题 2: Blackwell 生态不成熟

| 组件 | 问题 |
|------|------|
| bitsandbytes | 不支持 sm_120，必须用 Unsloth |
| flash-attn | 没有 Blackwell 预编译 wheel |
| PyTorch FlexAttention | backward kernel 共享内存超限 |
| Triton | 对 sm_120 的自动调优还没覆盖 |

**结论**: RTX 5090 在 2026 年 6 月仍然是一个 "early adopter" 硬件。做深度学习训练需要做好踩坑准备。

### 问题 3: 4-bit 量化的边界效应

LoRA + 4-bit 量化 + gradient checkpointing 的组合使得每个组件都在边界条件下运行：
- 量化模型不支持某些 dtype 操作
- Gradient checkpointing 在 backward 时重新触发 compile
- 混合精度 (bf16 base + f32 head) 增加了 dtype 协调的复杂度

---

## 当前状态与下一步

```
项目进度: ██████████████░░░░░░ ~65%

✅ 源码开发 (18 files, ~4200 lines)
✅ 7 轮代码审查 (25+ issues closed)
✅ 服务器环境 (AutoDL RTX 5090, CUDA 12.8, Unsloth 4-bit QLoRA)
✅ 5000 环境数据生成成功 (SFT: 5000, DPO: 186,896)
✅ 数据质量验证 (0 issues)
✅ 8 个服务器 Runtime Error 全部修复
✅ 过拟合测试通过 ← 当前阶段

⏳ 待执行: Stage I SFT 训练 (3 epochs, ~3-8h)
⏳ 待执行: Stage II DPO 训练 (2 epochs, ~5-10h)
⏳ 待执行: 评估 (200 test envs, 9 baselines)
```

### 修复汇总

| Commit | 修复 |
|--------|------|
| `13c402d` | Gemma3 hidden_dim 兼容 (text_config.hidden_size) |
| `62911f7` | Tokenizer 解包 (Gemma3Processor → tokenizer) |
| `fe56d2e` | token_type_ids 自动生成 |
| `73be561` | (已撤销) 错误的 bf16 cast |
| `b9bb4e4` | 投影头保持 f32 + control_states cast |
| `b9207eb` | BCE loss dtype 自动对齐 |
| `0a44528` | from_pretrained 设备对齐 |
| `87c9c1f` | attn_implementation → eager |
| `975931f` | TORCHINDUCTOR_FLEX_ATTENTION=0 |

### 给下一个踩坑的人

如果你在 Blackwell (RTX 5090) 上跑 PyTorch 训练：

1. **别用 bitsandbytes** — Unsloth 是唯一选择
2. **在 import torch 之前设 `TORCHINDUCTOR_FLEX_ATTENTION=0`** — 省你 2 小时
3. **用 `attn_implementation="eager"`** — 等 PyTorch/Triton 修好 Blackwell 再切回来
4. **投影头保持 f32** — 不要为了匹配 dtype 降精度
5. **多用 `hasattr`** — 多模态模型的 config 结构各有不同

---

## 相关文档

- [[16_handoff_01_project_direction](16_handoff_01_project_direction.md)] — 论文方向
- [[17_handoff_02_pre_datagen](17_handoff_02_pre_datagen.md)] — 数据生成前准备
- [[18_handoff_03_datagen_problems](18_handoff_03_datagen_problems.md)] — 数据生成问题与修复
- [[19_handoff_04_post_datagen](19_handoff_04_post_datagen.md)] — 当前状态与下一步
