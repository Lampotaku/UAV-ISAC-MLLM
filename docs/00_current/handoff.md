---
type: handoff
status: current
stage: sft
last_updated: 2026-06-28
related: [status, oom_incidents, quickstart]
---

# 项目交接文档 — UAV-ISAC-MLLM

**交接日期**: 2026-06-28 | **上一任**: Lampota | **下一任**: TBD

## 1. 项目一句话

用 **Gemma 3 12B + LoRA + 约束投影头** 为无人机通信感知一体化 (UAV-ISAC) 的 SCA-FP 数值优化器提供**智能热启动**——即用神经网络预测一个接近最优的初始解，让传统优化器从该点开始迭代，从而减少迭代次数、提高解的质量。

**类比**: 传统优化器是从随机点爬山找山顶；我们的模型看一张"地形图"后直接指一个离山顶很近的位置。

---

## 2. 当前状态（2026-06-28，真实情况）

### 已完成 ✅

| 项 | 状态 |
|----|------|
| 全部源码 | ✅ 完成，7 轮审查闭合 |
| GitHub 仓库 | ✅ `Lampotaku/UAV-ISAC-MLLM` (private) |
| 5000 环境数据生成 | ✅ SFT: 5000, DPO: 186,896 |
| OOM #1-#5 | ✅ 已修复 (省 ~54 GB) |
| Plan A (纯 PyTorch, 0 Unsloth) | ✅ 落地 |
| Phase 1 CTL-only 训练 | ✅ 完成，step 150 checkpoint 保存 |
| 控制表示学习验证 | ✅ Multi-Query Attention Pooling 有效 (sens 峰值 0.0901) |
| 代码在服务器上可运行 | ✅ |

### 当前阻塞 🔴

**Phase 2 训练启动时 OOM** (OOM #6)：

- **症状**: `--resume_from phase1_step_150` 启动 Phase 2 (CTL + CE joint training) 时，首次 forward 就 OOM @ 93.65 GB / 94.97 GB
- **根因**: 三层叠加——
  1. PEFT `modules_to_save=["lm_head"]` 打断了 Gemma 3 的 `tie_word_embeddings`（权重绑定），`lm_head.weight` 变成独立 984M 参数张量 → +2 GB 权重 + ~12 GB Adam 状态
  2. `PeftModel.gradient_checkpointing_enable()` 的委托链可能未完整传递到 `Gemma3Model` → activations ~60 GB（而非 gc 下的 ~3 GB）
  3. `outputs["logits"]` (3.5 GB bf16) 存活到 backward
- **最近修复** (commits `0532186` + `7f8bc54`):
  - `lm_head` 从 `modules_to_save` 移除 → 冻结 `lm_head.weight`，省 ~12 GB
  - `_parameters` OrderedDict 直接操作 clone 解绑（绕过 `nn.Module.__setattr__` 的 KeyError）
  - GC 硬加固：直接设 `transformer.gradient_checkpointing = True` + 补 `_gradient_checkpointing_func`
  - `del outputs["logits"]` 在 CE loss 计算后、backward 前
- **预期修复后显存**: ~75-80 GB（匹配 Phase 1 的 ~76 GB）
- **⚠️ 此修复尚未在服务器上验证！** 上次尝试遇到了 KeyError（`lm_head.weight` 赋值冲突），已修但未重试。

### 待办 ⏳

| 项 | 优先级 |
|----|--------|
| 在服务器上验证 OOM #6 修复 | 🔴 P0 |
| Phase 2 joint SFT+CTL 训练 (3 epochs, ~8.7h) | 🔴 P0 |
| 批量 SCA-FP 评估 (step 150/200/250/300/400) | 🟡 P1 |
| Stage II DPO 训练 (2 epochs) | 🟡 P1 |
| 最终评估 (200 test envs, 9 baselines) | 🟢 P2 |

---

## 3. 立即要做的事（接手第一步）

### Step 1: 登录服务器

```
服务器: AutoDL
GPU: RTX PRO 6000 96GB (Blackwell sm_120)
实例: 需要从 Lampota 获取 SSH 连接信息
```

### Step 2: 拉最新代码

```bash
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm
```

### Step 3: 验证环境

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
python -c "from transformers import AutoModel; print('HF OK')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

### Step 4: 启动 Phase 2 训练（核心任务）

```bash
# 清上次 OOM 残留
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"

# 环境变量 (Blackwell 必须)
export TORCHINDUCTOR_FLEX_ATTENTION=0

# 启动 Phase 2 (从 Phase 1 checkpoint 恢复)
python src/training/train_sft.py \
    --config configs/default.yaml \
    --data_dir /root/autodl-tmp/data/full5000 \
    --resume_from /root/autodl-tmp/checkpoints/phase1_step_150
```

### Step 5: 看日志

启动后**立即**看这行：

```
OOM6 guards: gc=✓, tied=... , lm_head_grad=False
```

| 日志 | 含义 | 行动 |
|------|------|------|
| `gc=✓` | gradient checkpointing 生效 ✅ | 继续 |
| `gc=✗ FAILED` | gc 彻底失败 🔴 | 必须降 `bs=1` (见下文 Fallback) |
| `tied=✗ (untied to protect embed_tokens)` | 正常 — embed_tokens 独立训练 | 继续 |
| `tied=✓` | 权重仍绑定 — embed_tokens 和 lm_head 共享张量 | 检查 `lm_head_grad` 必须为 `False` |
| `lm_head_grad=False` | lm_head 冻结成功 ✅ | 继续 |

### Fallback: 如果还是 OOM

```bash
# 方案 A: 减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 方案 B: 降 batch size (最可靠)
# 修改 configs/default.yaml:
#   training.sft.per_device_batch_size: 1
#   training.sft.gradient_accumulation_steps: 16
# 有效 batch 保持 16，显存降 ~7 GB
```

### 如果一切正常

训练会跑 ~8.7 小时（3 epochs × 2500 steps, ~4.1s/step）。建议在 tmux 中跑：

```bash
tmux new -s phase2
conda activate uavmllm
export TORCHINDUCTOR_FLEX_ATTENTION=0
python src/training/train_sft.py \
    --config configs/default.yaml \
    --data_dir /root/autodl-tmp/data/full5000 \
    --resume_from /root/autodl-tmp/checkpoints/phase1_step_150
# Ctrl+B D 断开
# tmux attach -t phase2 重连
```

---

## 4. 服务器环境

| 项 | 值 |
|----|-----|
| 平台 | AutoDL |
| GPU | RTX PRO 6000 96GB (Blackwell sm_120) |
| CUDA | 13.0 |
| Driver | 595.58.03 |
| 系统盘 | 30GB (`/root/`) — **不要写大文件到这里！** |
| 数据盘 | `/root/autodl-tmp/` — 所有数据、checkpoint、输出放这里 |
| Python | 3.12, conda env: `uavmllm` |
| 本地开发 | Windows, `h:\Projects\UAV` → git push → 服务器 git pull |

### 关键路径

```
/root/UAV-ISAC-MLLM/                     # 代码
/root/autodl-tmp/huggingface/models/     # Gemma 3 12B 权重 (~24 GB)
/root/autodl-tmp/data/full5000/          # 训练数据
/root/autodl-tmp/checkpoints/            # 模型 checkpoint
/root/autodl-tmp/outputs/                # 训练输出 / 日志
```

---

## 5. 架构速览

```
                      ┌─────────────────────────┐
                      │   Input Prompt (文本)      │
                      │   "Environment: ... UAVs: ... │
                      │    <ctrl_0>...<ctrl_7>"    │
                      └───────────┬─────────────┘
                                  │
                      ┌───────────▼─────────────┐
                      │   Gemma 3 12B (LoRA)     │
                      │   - 48 layers, 3840 dim  │
                      │   - LoRA rank=16, α=32   │
                      │   - 仅训练 q/k/v/o_proj  │
                      │   - gradient ckpt        │
                      └───────────┬─────────────┘
                                  │
                      ┌───────────▼─────────────┐
                      │   Control Token States    │
                      │   [B, 8, 3840]           │
                      └───────────┬─────────────┘
                                  │
                      ┌───────────▼─────────────┐
                      │ Multi-Query Attn Pooling │
                      │  4 queries → 4 UAV pools │
                      │  [B, 4, 3840]           │
                      └───────────┬─────────────┘
                                  │
                      ┌───────────▼─────────────┐
                      │   Projection Head (MLP)  │
                      │   3840→1920→960→44×4     │
                      └───────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
  ┌───────▼──────┐       ┌───────▼──────┐       ┌───────▼──────┐
  │ δ_q [B,4,3]  │       │ δ_a [B,4,20] │       │ δ_p [B,4,21] │
  │ 位移调整量     │       │ 用户关联矩阵   │       │ 功率调整量     │
  └───────────────┘       └───────────────┘       └───────────────┘

同时: lm_head(hidden_states) → logits → CE loss
```

### 源码目录

```
src/
├── env/          # UAV 仿真 (拓扑、信道、场景生成)
├── solver/       # SCA-FP 数值优化器 (Oracle)
├── data/         # Prompt 构造、Oracle 标注、Dataset
├── model/        # ★ 核心: gemma_isac.py, projection_head.py, losses.py
├── training/     # train_sft.py (Stage I), train_dpo.py (Stage II)
└── eval/         # evaluate.py, eval_generation.py
scripts/          # 辅助脚本 (数据生成、验证、过拟合测试)
configs/          # default.yaml (所有超参数)
```

### 损失函数

**Phase 1 (CTL-only)**: 仅优化控制损失，验证控制 token 学到了环境信息
```
L = λ_ctl × L_ctl   (L_ctl = λ_q·MSE(δ_q) + λ_a·BCE(δ_a) + λ_p·MSE(δ_p))
```

**Phase 2 (Joint SFT+CTL)**: 联合优化语言建模 + 控制预测
```
L = L_CE + λ_ctl × L_ctl  (+ λ_sep × L_sep)
```

---

## 6. OOM 完整历史（重要！每个都是血的教训）

| Bug | 根因 | 占用 | 修复 | 状态 |
|-----|------|------|------|------|
| #1 | HF CausalLM wrapper 存所有 hidden states + fp32 logits | ~14 GB | 直接调 `transformer()` + 手动 `lm_head` | ✅ |
| #2 | `.contiguous()` 拷贝 CE 输入 | ~8 GB | 必要代价，不可消除 | ✅ accepted |
| #3 | GQA attention fp32 中间张量 | ~16 GB | `gradient_checkpointing_enable()` | ✅ |
| #4 | `F.cross_entropy` 内部 fp32 拷贝 (256K vocab × 3456 seq × 4 bytes) | ~16 GB | bs 从 4 降到 1 (后升到 2) | ✅ |
| #5 | Unsloth 全局 monkey-patch 与 grad checkpoint 冲突 | N/A (CheckpointError) | Plan A: 彻底移除 Unsloth | ✅ |
| **#6** | **Phase 2: lm_head 解绑 + gc 未生效 + logits 存活** | **~18 GB** | **lm_head 冻结 + gc 硬加固 + del logits** | **⚠️ 待验证** |

### 显存预算（Phase 2, bs=2, seq=3456, bf16）

| 组件 | 预期占用 | 备注 |
|------|----------|------|
| Gemma 3 12B 权重 (bf16) | ~24 GB | |
| Embedding 权重 + Adam | ~15 GB | embed_tokens trainable, lm_head frozen |
| Activations (with gc) | ~3-5 GB | 无 gc 时 ~60 GB! |
| LoRA 参数 + Adam | ~2 GB | rank=16, 仅 q/k/v/o_proj |
| logits (bf16) | ~3.5 GB | `lm_head(hidden_states)` |
| shift_logits (bf16) | ~3.5 GB | CE 的 `.contiguous()` 拷贝 |
| CE 内部 fp32 | ~7 GB | PyTorch 内部 upcast |
| CUDA context + 其他 | ~3 GB | |
| **总计 (预期)** | **~75-80 GB** | 96GB 内安全 |
| **总计 (gc 失效时)** | **~94 GB** | OOM! |

---

## 7. 关键设计决策

### ADR 1: 彻底移除 Unsloth ([adr_001](docs/06_decisions/adr_001_unsloth_removal.md))

**决定**: 整个项目 `import unsloth` 次数 = 0。

**原因**: Unsloth 是全局 monkey-patch 框架。即使 `from unsloth.kernels import ...` 写在函数体里、只用于 loss 计算，Unsloth 仍然在 import 时全局替换 transformers 的 attention 层。这导致 forward (纯净 HF) 和 backward (被替换的层) 张量数不一致 → CheckpointError。

**替代**: 纯 PyTorch `F.cross_entropy` + SDPA attention。

### ADR 2: DPO Reference Model 独立加载

**决定**: DPO 训练时 reference model 独立从磁盘加载，不与 policy model 共享权重。

**原因**: `copy.deepcopy(policy_model)` 会 OOM (两个 24GB 模型)。独立加载 + 各自 `.to("cuda")` 避免双份显存峰值。

### Multi-Query Attention Pooling（核心架构创新）

**问题**: Phase 1 初期，控制表示的 sensitivity 为 0.0000（模型输出不随输入环境变化）。

**根因链**:
```
单 attention query (3840-dim)
  → softmax 强制注意力总和为 1
    → 关注 UAV1 就必须少关注 UAV2/3/4
      → 无法同时编码 4 架无人机的独立信息
```

**修复**: 单 query → 每 UAV 一个独立 query (M=4)，共享 readout MLP。

**效果**: sens 从 0.0000 → 峰值 0.0901 (step 150)。

### Phase 1 → Phase 2 分阶段训练

**Phase 1 (CTL-only)**: 完全关闭 CE loss，只训练控制损失。目的：让 control token 先学会编码环境信息，不被海量文本 token 的梯度淹没（梯度密度比 = 3456:8 = 432:1）。

**Phase 2 (Joint)**: 联合 CE + CTL 训练。Phase 1 的 checkpoint 作为起点。

---

## 8. 常见问题 & 排查

### Q: 训练 OOM 了，怎么办？

1. 先看日志里 `OOM6 guards` 行：`gc` 和 `tied` 的状态
2. 如果 `gc=✗`：gradient checkpointing 未生效，尝试手动设 `transformer.gradient_checkpointing = True`
3. 终极方案：降 `per_device_batch_size: 1` + 升 `gradient_accumulation_steps: 16`
4. 确认 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Q: 我想在本地 Windows 跑代码？

**不行。** Gemma 3 12B 需要 ~24 GB 仅加载权重，加上训练状态需要 ~76 GB。本地 Windows 没有这种 GPU。代码在本地修改，git push，服务器 pull 并运行。

### Q: 怎么在服务器上 debug？

```bash
# 快速显存检查
python -c "
import torch
from transformers import AutoModel
model = AutoModel.from_pretrained(
    '/root/autodl-tmp/huggingface/models/gemma-3-12b-it',
    torch_dtype=torch.bfloat16,
    attn_implementation='sdpa',
)
model = model.to('cuda')
print(f'Base model: {torch.cuda.memory_allocated()/1e9:.1f} GB')
"

# 看 GPU 使用
nvidia-smi
watch -n 1 nvidia-smi

# 看训练日志
tail -f /root/autodl-tmp/outputs/stage1_sft_final/logs/*.log
```

### Q: checkpoint 在哪里？怎么恢复训练？

```bash
# 保存位置
/root/autodl-tmp/checkpoints/
├── phase1_step_50/
├── phase1_step_100/
├── phase1_step_150/    ← Phase 1 最佳
├── phase1_step_200/
└── phase1_step_250/

# 恢复训练
python src/training/train_sft.py \
    --config configs/default.yaml \
    --resume_from /root/autodl-tmp/checkpoints/phase1_step_150
```

### Q: 怎么知道 Phase 1 学得怎么样？

两个指标：
- **`loss_ctl`**: 控制损失，越低越好
- **`sensitivity`**: 跨环境区分度。对两个不同随机种子生成的环境，`||δ_q(env_B) - δ_q(env_A)|| / ||δ_q(env_A)||`。> 0.05 = 有效，> 0.08 = 良好

**关键经验**: `loss_ctl` 和 sensitivity 在训练后期背离——loss 持续下降但 sens 也会下降（MSE 收缩效应）。选 sens 最高的 checkpoint，不选 loss 最低的。

---

## 9. 日常操作清单

### 代码修改 → 服务器运行

```bash
# 本地 Windows (h:\Projects\UAV)
# ... 改代码 ...
git add -A && git commit -m "描述你的改动" && git push

# 服务器
cd /root/UAV-ISAC-MLLM && git pull
# 运行训练 / 测试
```

### 数据相关

```bash
# 验证数据完整性
python scripts/validate_data.py --data-dir /root/autodl-tmp/data/full5000

# 重新生成数据 (⚠️ ~3.5 小时)
python scripts/generate_data.py --num-envs 5000 --output-dir /root/autodl-tmp/data/full5000 --num-workers 70
```

### 过拟合测试（证明训练管道没问题）

```bash
# 在小数据上验证代码正确性 (~5 min)
export TORCHINDUCTOR_FLEX_ATTENTION=0
python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000
```

### 评估

```bash
# SCA-FP 批量评估
python scripts/eval_generation.py \
    --config configs/default.yaml \
    --checkpoint /root/autodl-tmp/checkpoints/phase1_step_150 \
    --n_samples 3 --n_scafp 100

# 完整评估 (200 test envs, 9 baselines)
python src/eval/evaluate.py --config configs/default.yaml
```

---

## 10. 关键文件速查

| 你想知道... | 看这个 |
|-------------|--------|
| 项目当前状态 | [status.md](status.md) |
| 怎么在服务器跑起来 | [quickstart.md](quickstart.md) |
| 所有配置参数 | [configs/default.yaml](../configs/default.yaml) |
| 模型定义 | [src/model/gemma_isac.py](../src/model/gemma_isac.py) |
| 训练循环 | [src/training/train_sft.py](../src/training/train_sft.py) |
| OOM 历史 | [oom_incidents.md](../02_training_log/oom_incidents.md) |
| Phase 1 调试全纪录 | [phase1_status_2026-06-26.md](../02_training_log/phase1_status_2026-06-26.md) |
| 为什么不用 Unsloth | [adr_001](../06_decisions/adr_001_unsloth_removal.md) |
| 问题数学定义 | [problem_formulation.md](../01_architecture/problem_formulation.md) |
| 模块拓扑 | [system_design.md](../01_architecture/system_design.md) |

---

## 11. 禁忌清单

1. ❌ **不要在项目里 `import unsloth`** — 全局 monkey-patch 破坏一切
2. ❌ **不要在系统盘 (`/root/`) 写大文件** — 只有 30GB，写满服务器会挂
3. ❌ **不要 `copy.deepcopy(model)`** — 双份 24GB 直接 OOM
4. ❌ **不要改 `modules_to_save` 加回 `lm_head`** — 会解绑权重 + ~12 GB Adam 状态
5. ❌ **不要关闭 gradient checkpointing** — activations ~60 GB，必然 OOM
6. ❌ **不要用 `output_hidden_states=True`** — 存 48 层 hidden states ~6 GB
7. ❌ **不要在 CausalLM wrapper 上传 `labels`** — HF 内部 CE 产生 fp32 logits ~16 GB
8. ❌ **不要用 Flash Attention 2** — Blackwell sm_120 没有预编译 wheel，用 SDPA

---

## 12. 联系人

| 角色 | 人 | 联系 |
|------|----|------|
| 上一任 | Lampota | GitHub: `Lampotaku` |
| GitHub | `Lampotaku/UAV-ISAC-MLLM` | private repo |

如果遇到 Block 级问题（训练不收敛、OOM 无法解决、架构需要大改），建议先：
1. 查 [docs/](.) 里有没有相关 postmortem
2. 读 [status.md](status.md) 的决策树和备用路线 (P0-P3)
3. 联系 Lampota 讨论
