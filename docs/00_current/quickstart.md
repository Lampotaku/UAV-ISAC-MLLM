---
type: reference
status: current
stage: sft
last_updated: 2026-06-26
related: [canonical_config, status, hardware_adaptation]
---

# Quickstart — 从零到训练

**目标受众**: 刚接手项目的工程师，需要在服务器上跑起来。

**预计时间**: 15 分钟 (不含数据生成和训练时间)

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

# 或手动安装
conda create -n uavmllm python=3.12 -y
conda activate uavmllm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install unsloth transformers trl datasets accelerate peft
pip install scipy numpy matplotlib pyyaml wandb
```

## Step 2: HuggingFace 认证

```bash
huggingface-cli login
# 输入你的 HF token (需要 Gemma 3 access)
```

## Step 3: 验证环境

```bash
conda activate uavmllm
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import unsloth; print(f'Unsloth: {unsloth.__version__}')"
python -c "from transformers import AutoModel; print('HF OK')"

# 语法检查
python -m compileall -q src scripts
```

## Step 4: 数据准备

**如果已有生成好的数据** (推荐):
```bash
ls /root/autodl-tmp/data/full5000/sft_dataset.jsonl  # 应存在
ls /root/autodl-tmp/data/full5000/dpo_dataset.jsonl   # 应存在
```

**如果需要重新生成** (~3.5h, 70 workers):
```bash
python scripts/generate_data.py \
    --num-envs 5000 \
    --num-restarts 10 \
    --output-dir /root/autodl-tmp/data/full5000 \
    --num-workers 70
```

**验证数据质量**:
```bash
python scripts/validate_data.py --data-dir /root/autodl-tmp/data/full5000
python scripts/eda_data.py --data-dir /root/autodl-tmp/data/full5000
```

## Step 5: 过拟合测试 (5 min)

**在正式训练前必须运行** — 证明训练代码正确:

```bash
export TORCHINDUCTOR_FLEX_ATTENTION=0
python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000
```

5 项检查全部通过标准:
- `loss_total` 下降 >50%
- `loss_sft` < 0.5
- `loss_ctl` < 0.01
- 最后 50 步单调下降
- 无 NaN/Inf

## Step 6: 启动 SFT 训练 (~8.7h)

```bash
# 在 tmux/screen 中运行 (防止断连)
tmux new -s sft_train

export TORCHINDUCTOR_FLEX_ATTENTION=0
conda activate uavmllm
python src/training/train_sft.py --config configs/default.yaml

# Ctrl+B D 离开 tmux
# tmux attach -t sft_train 重新连接
```

监控:
```bash
# TensorBoard
tensorboard --logdir outputs/stage1_sft_final/logs --port 6006

# 显存
watch -n 1 nvidia-smi
```

## Step 7: DPO 训练 (~5-10h)

SFT 完成后:
```bash
tmux new -s dpo_train

export TORCHINDUCTOR_FLEX_ATTENTION=0
conda activate uavmllm
python src/training/train_dpo.py --config configs/default.yaml
```

## Step 8: 评估

```bash
python src/eval/evaluate.py --config configs/default.yaml
```

---

## ⚠️ Blackwell RTX PRO 6000 陷阱

| 陷阱 | 症状 | 解决 |
|------|------|------|
| FlexAttention OOM | "shared memory" error in backward | `export TORCHINDUCTOR_FLEX_ATTENTION=0` |
| bitsandbytes 不支持 | ImportError / CUDA error | 用 Unsloth 替代 |
| Flash Attention 2 不可用 | 无预编译 sm_120 wheel | 用 SDPA (`attn_implementation="sdpa"`) |
| Triton 未调优 sm_120 | 性能下降 | 接受 — 等待上游更新 |
| Gemma 3 需要 `token_type_ids` | 训练崩溃 | 已处理: `torch.ones_like(input_ids)` |
| Unsloth 全局劫持 | SDPA 被覆盖为 eager | 彻底移除 Unsloth (Plan A) |
| `from_pretrained` 加载不一致 | projection head 在 CPU | 已处理: `.to(base_model.device)` |

详见 [01_architecture/hardware_adaptation.md](../01_architecture/hardware_adaptation.md)

## 工作流速查

```
git clone → autodl_setup.sh → HF login → verify env
  → (generate data if needed) → validate data → EDA
  → test_sft_overfit.py (5 min) → SFT (~8.7h) → DPO (~5-10h) → evaluate
```

## 本地开发流程

```
Windows (h:\Projects\UAV) → git push → AutoDL (git pull) → 执行训练
```

- 代码修改在本地 Windows
- 推送后在服务器上 pull 并运行
- `configs/default.yaml` 通过 git 同步
