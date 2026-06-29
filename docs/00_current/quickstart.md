---
type: reference
status: current
stage: data_regeneration
last_updated: 2026-06-29
related: [canonical_config, status, hardware_adaptation, data_degeneracy]
---

# Quickstart — 从零到训练

**目标受众**: 刚接手项目的工程师，需要在服务器上跑起来。

> 📋 **第一次接触项目？先读 [status.md](status.md)** — 包含当前状态、blocker、已知问题。

**预计时间**: 20 分钟 (不含数据生成和训练时间)

## ⚠️ 重要：数据层已修复

旧版数据（full5000）存在**分布退化**——SCA-FP 求解器缺乏地面杂波建模，导致所有样本偏向"全速砸地板"。旧数据已完全废弃。

**新版数据必须用修复后的求解器生成**（`ground_clutter_db=12.0`）。详见 [data_degeneracy.md](../02_training_log/data_degeneracy.md)。

## 前置条件

- AutoDL RTX PRO 6000 96GB 服务器 (或同等 Blackwell GPU)
- GitHub 访问权限 (repo: `Lampotaku/UAV-ISAC-MLLM`, private)
- HuggingFace 认证 (Gemma 3 是 gated model)

## Step 1: 环境搭建

```bash
# 克隆仓库
cd /root
git clone git@github.com:Lampotaku/UAV-ISAC-MLLM.git
cd UAV-ISAC-MLLM

# 运行自动安装脚本
bash scripts/autodl_setup.sh

# 或手动安装 (纯 PyTorch — 不使用 Unsloth)
conda create -n uavmllm python=3.12 -y
conda activate uavmllm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers trl datasets accelerate peft
pip install scipy numpy matplotlib pyyaml wandb
```

> ⚠️ **不再安装 Unsloth**。Plan A 使用纯 PyTorch CE + SDPA。Unsloth 的全局 monkey-patch 与 Gemma 3 + SDPA + grad checkpoint 不兼容。

## Step 2: HuggingFace 认证

```bash
huggingface-cli login
# 输入你的 HF token (需要 Gemma 3 access)
```

## Step 3: 验证环境

```bash
conda activate uavmllm
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModel; print('HF OK')"

# 语法检查
python -m compileall -q src scripts
```

## Step 4: 数据准备

### 第 1 步：快速验证求解器修复 (5 min)

**在生成全量数据前必做** — 确认地面杂波修复产生了多样化分布：

```bash
python scripts/quick_validate_fix.py
```

**验收标准**：
- 满速飞行比例 < 40%（原 84.7%）
- 精细微调 (<5m) 比例 > 10%（原 0%）
- **上升比例 > 15%（原 0%）** — 核心红线

不通过 → 检查 `ground_clutter_db` 值（建议 10-15dB），或者 solver 的 `clutter_db` 是否正确应用到两个路径损耗计算中。

### 第 2 步：全量数据生成 (~3.5h, 70 workers)

```bash
python scripts/generate_data.py \
    --num-envs 5000 \
    --num-restarts 10 \
    --output-dir /root/autodl-tmp/data/full5000_v2 \
    --num-workers 70
```

### 第 3 步：数据验证 + EDA 验收

```bash
# 语义验证 (无 NaN, 约束满足)
python scripts/validate_data.py --data-dir /root/autodl-tmp/data/full5000_v2

# 多样性 EDA (红线检查: 方向/速度/功率)
python scripts/eda_data.py --data-dir /root/autodl-tmp/data/full5000_v2
```

**EDA Section 3 三条红线全部通过才能进入训练。**

## Step 5: DPO 训练 (~5-10h)

**跳过 SFT，直接 DPO**。理由是：
- Step 150 的 SFT checkpoint 已具备 JSON 生成能力
- SFT (Teacher Forcing + CE) 对连续物理量有硬天花板
- DPO 对比学习更适合学习"哪种位移更优"的相对排序

```bash
# 在 tmux/screen 中运行
tmux new -s dpo_train

export TORCHINDUCTOR_FLEX_ATTENTION=0
conda activate uavmllm

python src/training/train_dpo.py \
    --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/checkpoints/stage1_step_150 \
    --data_dir /root/autodl-tmp/data/full5000_v2

# Ctrl+B D 离开 tmux
```

**预期 VRAM**: ~65-75 GB / 96 GB (bs=1, 双模型)

监控:
```bash
watch -n 1 nvidia-smi
# 关注: loss_dpo, reward_margin
```

## Step 6: 评估

```bash
python scripts/eval_generation.py \
    --config configs/default.yaml \
    --checkpoint <dpo_checkpoint> \
    --n_samples 3 --n_scafp 100
```

**验收标准**：
- SCA-FP 加速比 ≥ 1.5×（核心判据）
- Control sensitivity > 0.1
- 文本生成 JSON 格式正常

---

## ⚠️ Blackwell RTX PRO 6000 陷阱

| 陷阱 | 症状 | 解决 |
|------|------|------|
| FlexAttention OOM | "shared memory" error in backward | `export TORCHINDUCTOR_FLEX_ATTENTION=0` |
| bitsandbytes 不支持 | ImportError / CUDA error | bf16 全精度 (96GB 无需量化) |
| Flash Attention 2 不可用 | 无预编译 sm_120 wheel | 用 SDPA (`attn_implementation="sdpa"`) |
| Triton 未调优 sm_120 | 性能下降 | 接受 — 等待上游更新 |
| Gemma 3 需要 `token_type_ids` | 训练崩溃 | 已处理: `torch.ones_like(input_ids)` |
| **Unsloth 全局劫持** | SDPA 被覆盖为 eager | **彻底移除 Unsloth (Plan A)** |
| `from_pretrained` 加载不一致 | projection head 在 CPU | 已处理: `.to(base_model.device)` |
| Grad diagnostic 崩溃 | step 200 突然 backward error | 已修复: `retain_graph=True` |

详见 [01_architecture/hardware_adaptation.md](../01_architecture/hardware_adaptation.md)

## 工作流速查

```
git clone → autodl_setup.sh → HF login → verify env
  → quick_validate_fix.py (5 min)
  → generate_data.py full5000_v2 (~3.5h) → validate + EDA
  → DPO training (~5-10h) → evaluate
```

## 本地开发流程

```
Windows (h:\Projects\UAV) → git push → AutoDL (git pull) → 执行训练
```

- 代码修改在本地 Windows
- 推送后在服务器上 pull 并运行
- `configs/default.yaml` 通过 git 同步

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| OOM 在 Phase 2 切换 | `gc.collect()` + `torch.cuda.empty_cache()` — 已修复 (OOM #6) |
| Step 200 crash | `retain_graph=True` — 已修复 (OOM #7) |
| 数据分布退化 | 用新版 solver (ground_clutter_db=12.0) 重新生成 |
| DPO OOM | 降 bs=1, 确认 reference model 用独立 load (非 deepcopy) |
