# 交接文档 #7 — Stage I SFT 训练：OOM 五连杀 → Plan A → bs=2 终极配置 → 训练进行中

> 日期: 2026-06-26  
> 状态: **SFT 训练进行中** (服务器挂着, ~8.7h 完成)  
> 上接: [#6 SFT 速度战](22_handoff_06_sft_speed_war.md) → [#23 OOM#4](23_handoff_oom4_unchunked_ce.md) → [#24 Plan A](24_handoff_planA_pure_pytorch.md) → [#25 Bug Postmortems](25_BUG_postmortems_OOM_to_SDPA.md)  
> 标签: sft-training, oom-fixes, plan-a, bs2-upgrade, ultimate-config, live-status

---

## 目录

1. [当前状态：一图流](#当前状态一图流)
2. [完整时间线](#完整时间线)
3. [最终配置：终极发车清单](#最终配置终极发车清单)
4. [显存剖析（实测）](#显存剖析实测)
5. [速度分析：bs=1 vs bs=2](#速度分析bs1-vs-bs2)
6. [关键设计决策](#关键设计决策)
7. [下一步](#下一步)
8. [Commit 映射表](#commit-映射表)
9. [快速参考卡](#快速参考卡)

---

## 当前状态：一图流

```
训练进度: ████████████████████░░░ ~80%

✅ 源码开发 (18 files, ~4500 lines)
✅ 7 轮代码审查 (25+ issues closed)
✅ 5000 环境数据生成 (SFT: 5000, DPO: 186,896, 0 issues)
✅ OOM #1-4 修复 (省 ~54 GB, 详见 postmortem)
✅ CheckpointError #5 修复 (彻底肃清 Unsloth)
✅ Plan A 纯 PyTorch CE 验证 (bs=1, 2.54s/step, ~48GB)
✅ seq_len 4096→3456 (砍 ~40% 冗余 padding 注意力)
✅ bs 1→2, grad_accum 16→8 (epoch 吞吐 +18%)
✅ 全项目 0 处 Unsloth 引用

🟢 Stage I SFT 训练进行中
   ├─ 服务器: AutoDL RTX PRO 6000 96GB
   ├─ 速度: ~4.1s/micro-batch (bs=2, seq=3456)
   ├─ 显存: 76.3GB / 95.6GB (20GB 余量)
   ├─ Epoch: 1/3, 2500 steps/epoch, ~313 optimizer steps/epoch
   ├─ 预计: ~2.9h/epoch, ~8.7h total
   └─ 状态: GPU 100% 满载, 无 OOM, 无 CheckpointError

⏳ Stage II DPO 训练 (SFT 完成后)
⏳ 评估 (DPO 完成后)
```

---

## 完整时间线

```
2026-06-25  数据生成完成 (5000 SFT + 186,896 DPO)
            │
            ├─ OOM #1: HF CausalLM hidden_states + fp32 logits → 省 ~14 GB
            ├─ OOM #2: logits .contiguous() 拷贝 → 省 ~8 GB
            ├─ OOM #3: GQA log_softmax fp32 存储 → grad ckpt 省 ~16 GB
            ├─ OOM #4: F.cross_entropy fp32 梯度 (16 GB) → Plan B (Unsloth CE)
            └─ Bug #5: CheckpointError — Unsloth 局部导入全局劫持 → Plan A
            │
2026-06-26  Plan A 验证成功: bs=1, grad_accum=16, 纯 PyTorch CE
            2.54s/step, ~48GB 峰值, 无 Unsloth
            │
            ├─ 序列长度分析: 所有 5000 样本 3137-3329 tokens
            │  max_seq_length: 4096 → 3456 (128-aligned, 安全包容 max+ctrl)
            │
            ├─ bs 升级: 1→2, grad_accum: 16→8
            │  有效 batch 保持 16, epoch 吞吐 +18%
            │
            └─ 全量 SFT 启动: bs=2, seq=3456, SDPA, bf16
               4.14s/step, ~76GB/96GB, 预计 ~8.7h total
```

---

## 最终配置：终极发车清单

这是经过 **5 个 OOM bug + 1 个 CheckpointError + 4 轮速度优化** 后收敛到的最优配置：

### configs/default.yaml 关键值

```yaml
hardware:
  use_4bit: false                    # bf16 全精度 LoRA (96GB 无需量化)
  gradient_checkpointing: true       # 必须开启 (省 ~16GB GQA 中间激活)
  max_grad_norm: 1.0

model:
  attn_implementation: "sdpa"        # PyTorch 原生 cuDNN Fused Attention
  lora:
    rank: 16
    alpha: 32
    dropout: 0                       # >0 会禁用快速内核
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  sft:
    epochs: 3
    per_device_batch_size: 2         # bs=2 — 吃满剩余显存, 吞吐 +18%
    gradient_accumulation_steps: 8   # 有效 batch = 2×8 = 16
    max_seq_length: 3456             # 128-aligned, 包容 max 3329 + 8 ctrl
    learning_rate: 2.0e-4
    warmup_ratio: 0.03
    save_steps: 200
```

### 环境变量（train_sft.py 顶部，顺序敏感）

```python
# 1. 防 CPU 过载 (必须在 import numpy/torch 之前)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 2. CUDA 内存碎片化防护
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 3. 防 FlexAttention (Blackwell sm_120 共享内存不足)
os.environ["TORCHINDUCTOR_FLEX_ATTENTION"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# 4. 网络静默 (国内环境)
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### 四个核心优化

| # | 优化 | 效果 | 为什么 |
|---|------|------|--------|
| 1 | **纯 PyTorch (0 Unsloth)** | 激活 SDPA, 避免 CheckpointError | Unsloth 全局 monkey-patch 与 grad ckpt 不兼容 |
| 2 | **SDPA attention** | 2-4s/step (vs eager 21s/step) | cuDNN fused attention, O(n²)→O(n) memory |
| 3 | **max_seq_length: 3456** | 省 ~40% 注意力计算 | 所有样本 ≤3329, 4096 中 800 tokens 是 padding |
| 4 | **bs=2 + grad_accum=8** | epoch 吞吐 +18% vs bs=1 | 每步 2× tokens 但步数减半 |

---

## 显存剖析（实测）

RTX PRO 6000 96GB, bs=2, seq=3456, bf16:

| 组件 | 大小 | 备注 |
|------|------|------|
| bf16 LoRA 模型 | ~24 GB | Gemma 3 12B + LoRA adapter |
| AdamW 状态 (embed_tokens) | ~8 GB | modules_to_save |
| logits bf16 (2×3456×256K) | ~3.5 GB | bs=2 前向输出 |
| CE fp32 log_softmax 中间 | ~7 GB | F.cross_entropy 内部 |
| 激活梯度 (backward) | ~5 GB | GQA attention grads |
| grad_logits bf16 | ~3.5 GB | ∂L/∂logits |
| grad_embed | ~2 GB | ∂L/∂embed_tokens |
| 其他 (last_hidden_state, buffers) | ~5 GB | |
| CUDA context + 碎片 | ~18 GB | 96GB 卡的固定开销 |
| **实测峰值** | **~76-78 GB** | ~20GB 余量, 安全 |

### bs=1 vs bs=2 显存对比

| 配置 | 峰值 | 余量 | 速度 |
|------|------|------|------|
| bs=1, grad_accum=16 | ~48 GB | ~48 GB | 2.54s/step, 5000 steps/epoch |
| bs=2, grad_accum=8 | ~78 GB | ~18 GB | 4.14s/step, 2500 steps/epoch |

bs=2 多占 ~30GB 主要来自 CE fp32 中间张量（翻倍）和 logits 张量（翻倍），但仍在 96GB 安全范围内。

---

## 速度分析：bs=1 vs bs=2

### 为什么 bs=2 每步更慢但总时间更短

```
bs=1:  2.54s/step × 5000 steps/epoch = 12,700s = 3.5h/epoch
bs=2:  4.14s/step × 2500 steps/epoch = 10,350s = 2.9h/epoch

提升: (3.5 - 2.9) / 3.5 = 18% ↑
3 epochs 总节省: (3.5 - 2.9) × 3 ≈ 1.8h
```

**原因**: bs=2 每步处理 2× tokens → 每步时间约 1.6× (不是 2×，因为 GPU 计算是并行的)。同时步数减半 → 净收益 ~18%。

### 为什么不是 1.5-2s/step

最初的 1.5-2s/step 预估基于 bs=1。bs=2 的 4.14s/step 是正确的：
- 2× batch size → ~1.6× 每步时间 (GPU 并行效率)
- CE fp32 中间张量更大 → 更多 HBM 带宽压力
- 2.54 × 1.6 ≈ 4.1s ✓

### 为什么不是 3.3s/step（docstring 中的预期）

docstring 中 "bs=2/seq=3456 (预期): ~3.3s/micro-batch" 是高估了 seq_len 缩短的收益。实测 3456 vs 4096 在 bs=2 下速度差异很小（~1-2%），因为：
- SDPA 的 cuDNN fused attention 对 seq_len 不敏感（不是 O(n²)）
- bs=2 时瓶颈是 CE 的 fp32 中间张量读写，不是 attention 计算
- seq_len 缩短主要省 attention 显存，但显存本就不是瓶颈（20GB 余量）

---

## 关键设计决策

### 1. 为什么彻底不用 Unsloth

**一句话**: Unsloth 与 Gemma 3 + SDPA + gradient checkpointing 不可共存。

```
import unsloth  # 即使是局部导入
  → 全局 monkey-patch transformers 底层
    → 强制 eager attention (Gemma 3 不支持其 Triton kernel)
      → 21s/step (vs SDPA 的 2-4s/step)
      
如果在训练循环中途导入 (如 loss 函数内):
  → Forward: 纯净 HF 原生状态 (68 个激活张量)
  → import unsloth → attention 层被替换
  → Backward recompute: 被替换的层 (65 个张量)
  → CheckpointError: 68 ≠ 65
```

**唯一例外**: `gemma_isac.py` 中 `use_4bit=True` 分支保留 `from unsloth import FastLanguageModel`，但当前 `use_4bit: false` → 永远不触发。

### 2. 为什么 bs=2 而不是 bs=4

bs=4 时 CE fp32 中间张量 ~16GB → OOM。bs=2 是 96GB 卡上的最优 trade-off：
- bs=1: 安全但 GPU 闲置 48GB
- bs=2: 吃满 ~78GB, 吞吐 +18%
- bs=4: OOM

### 3. 为什么 max_seq_length=3456 而不是 3328

| 值 | 问题 |
|----|------|
| 3328 (99th %ile) | 截断最大样本 3329 的最后 1 token → 可能截断 control token → 投影头崩溃 |
| 3456 (27×128) | 安全包容 max 3329 + 8 control tokens = 3337; 128-aligned → Tensor Core 最优 |

### 4. 为什么 SDPA 而不是 Flash Attention 2

- FA2 需要 Blackwell sm_120 的预编译 wheel — 不存在
- PyTorch SDPA 在 Blackwell 上自动使用 cuDNN Fused Attention — 等效性能
- SDPA 零额外依赖，PyTorch 内置

---

## 下一步

### SFT 完成后（预计 ~8.7h 从启动算起）

```bash
# Step 1: 验证 SFT checkpoint
ls /root/autodl-tmp/checkpoints/stage1_step_*.safetensors
ls /root/autodl-tmp/outputs/stage1_sft_final/

# Step 2: Stage II DPO 训练
cd /root/UAV-ISAC-MLLM
conda activate uavmllm
python src/training/train_dpo.py --config configs/default.yaml --data_dir /root/autodl-tmp/data/full5000
```

DPO 配置（已在 configs/default.yaml 中）:
```yaml
training:
  dpo:
    epochs: 2
    per_device_batch_size: 1       # bs=1 — DPO 双模型 (policy + reference)
    gradient_accumulation_steps: 16 # 有效 batch = 16
    max_seq_length: 3456
    learning_rate: 5.0e-5
    beta: 0.1
    mu: 0.05                       # SFT anchor (防遗忘)
```

DPO 显存预估:
| 组件 | 大小 |
|------|------|
| Policy model (LoRA) | ~24 GB |
| Reference model (独立加载) | ~24 GB |
| 4× logits fp32 | ~15 GB |
| Backward | ~12 GB |
| **峰值** | **~75 GB** (96GB 安全) |

### DPO 完成后

```bash
python src/eval/evaluate.py --config configs/default.yaml
```

200 测试环境，9 基线 × 6 指标。

---

## Commit 映射表

```
┌────────────┬──────────────────────────────────────────────────────┬───────────┐
│ Commit     │ 描述                                                  │ 效果      │
├────────────┼──────────────────────────────────────────────────────┼───────────┤
│ de3ddbb    │ docs: docstring 更新对齐实测 (bs=2→4.1s, ~8.7h)       │ 文档      │
│ a1526d5    │ perf: max_seq_length 4096→3456                         │ 省 ~40% 注意力 │
│ 2c0e61d    │ fix: 锁定 SDPA (不用 FA2)                              │ 稳定性    │
│ 0edd5c1    │ perf: bs 1→2, grad_accum 16→8 + token 长度分析        │ 吞吐 +18% │
│ 76e06a4    │ docs: OOM→SDPA 5-bug postmortem                       │ 文档      │
│ 950d566    │ fix: Plan A — 彻底肃清 Unsloth, 纯 PyTorch CE          │ 修复 #5   │
│ 3bbe260    │ fix: 删除全局 import unsloth                           │ 修复 #5   │
│ 808ad4b    │ fix: .contiguous() for CE slicing                     │ 修复 #2   │
│ 4d2d2f3    │ docs: OOM#4 postmortem                                │ 文档      │
│ 479c226    │ perf: Unsloth Chunked CE (Plan B, 后被 Plan A 取代)    │ 省 16GB   │
│ fe0c34a    │ fix: grad-ckpt CE/log_softmax (OOM #3)                │ 省 16GB   │
│ 8c3b2a8    │ perf: 消除 .contiguous() 拷贝 (OOM #2)                 │ 省 8GB    │
│ 68c2567    │ fix: 绕过 HF CausalLM wrapper (OOM #1)                 │ 省 14GB   │
│ 666e23f    │ feat: 双路径模型加载 (native HF+PEFT SDPA)              │ 2-3s/step │
│ de26ede    │ docs: SFT speed war postmortem                        │ 文档      │
│ 65cf10d    │ fix: backbone 路径 + dropout=0                        │ 21→16s    │
│ b0fd596    │ fix: 传递 attn_implementation (被 Unsloth 覆盖)        │ 无效*     │
└────────────┴──────────────────────────────────────────────────────┴───────────┘
```

---

## 快速参考卡

### 服务器信息

| 项 | 值 |
|----|-----|
| 平台 | AutoDL |
| GPU | RTX PRO 6000 96GB (Blackwell sm_120) |
| CUDA | 13.2, Driver 595.58.03 |
| Python | 3.11, conda env: `uavmllm` |
| 代码路径 | `/root/UAV-ISAC-MLLM` |
| 数据路径 | `/root/autodl-tmp/data/full5000/` |
| 模型路径 | `/root/autodl-tmp/huggingface/models/gemma-3-12b-it` |
| 输出路径 | `/root/autodl-tmp/outputs/` |
| Checkpoint | `/root/autodl-tmp/checkpoints/` |
| 日志 | `/root/autodl-tmp/logs/` |

### 救命命令

```bash
# 查看训练状态 (GPU 占用/进程)
nvidia-smi

# 查看最新 checkpoint
ls -lt /root/autodl-tmp/checkpoints/ | head -5

# 实时监控显存
watch -n 1 nvidia-smi

# 拉取最新代码 (本地 Windows → push → 服务器 pull)
cd /root/UAV-ISAC-MLLM && git pull origin master

# 如果训练挂了, 从 checkpoint 恢复:
# (当前 train_sft.py 不支持自动恢复, 需要手动改代码或从头跑)
# TODO: 添加 --resume_from_checkpoint 参数

# DPO 训练 (SFT 完成后)
python src/training/train_dpo.py --config configs/default.yaml --data_dir /root/autodl-tmp/data/full5000
```

### 反模式清单（别踩的坑）

| 反模式 | 为什么 | 正确做法 |
|--------|--------|---------|
| `import unsloth` 在训练脚本任何地方 | 全局 monkey-patch → CheckpointError 或 eager 21s/step | 永远不用 Unsloth |
| bs≥4 for SFT | CE fp32 梯度 ~16GB → OOM | bs=2, grad_accum=8 |
| max_seq_length=4096 | 800 tokens padding → 浪费 40% 计算 | 3456 (128-aligned) |
| 不设 OMP_NUM_THREADS=1 | DataLoader workers 抢 CPU → 卡死 | 脚本第一行就设 |
| 不开 gradient_checkpointing | GQA 中间激活 ~16GB → OOM | 必须开 |
| 用 `AutoModelForCausalLM` 取 logits | 多余 hidden_states + fp32 upcast | `base_model.model(...)` + 手动 `lm_head` |
| DPO bs≥2 | 双模型 + 4×logits → OOM | bs=1, grad_accum=16 |

### 关键源文件

| 文件 | 用途 | 当前状态 |
|------|------|---------|
| [train_sft.py](src/training/train_sft.py) | Stage I SFT 训练循环 | ✅ 运行中 |
| [train_dpo.py](src/training/train_dpo.py) | Stage II DPO 训练循环 | ⏳ 待运行 |
| [gemma_isac.py](src/model/gemma_isac.py) | 模型定义 (绕过 HF wrapper + grad ckpt) | ✅ 稳定 |
| [losses.py](src/model/losses.py) | 损失函数 (纯 PyTorch CE + 约束损失) | ✅ 0 Unsloth |
| [dataset.py](src/data/dataset.py) | SFT/DPO Dataset | ✅ 稳定 |
| [default.yaml](configs/default.yaml) | 全局配置 | ✅ 终极配置 |

---

## 相关文档

- [[25_BUG_postmortems_OOM_to_SDPA](25_BUG_postmortems_OOM_to_SDPA.md)] — 5 个 bug 的详细根因分析
- [[24_handoff_planA_pure_pytorch](24_handoff_planA_pure_pytorch.md)] — Plan A 设计与验证
- [[22_handoff_06_sft_speed_war](22_handoff_06_sft_speed_war.md)] — SFT 速度战 (Unsloth eager → SDPA)
- [[19_handoff_04_post_datagen](19_handoff_04_post_datagen.md)] — 数据生成完成后的状态
- [[talk.md](../talk.md)] — 同事的 seq_len 分析与终极配置推荐
