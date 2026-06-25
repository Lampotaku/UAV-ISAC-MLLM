---
type: status
status: current
stage: sft
last_updated: 2026-06-26
related: [canonical_config, sft_live, oom_incidents, verification_gaps]
---

# 项目当前状态

**最后更新**: 2026-06-26 | **阶段**: Stage I SFT 训练进行中

## 🟢 SFT 训练状态

| 指标 | 值 |
|------|-----|
| **进度** | ~80% (~7h / ~8.7h estimated) |
| **服务器** | AutoDL RTX PRO 6000 96GB (Blackwell sm_120) |
| **配置** | bs=2, grad_accum=8, seq=3456, SDPA, bf16, lr=2e-4 |
| **速度** | ~4.1s/micro-batch, ~2.9h/epoch |
| **VRAM** | ~76GB peak / 96GB available (20GB headroom) |
| **状态** | ✅ 无 OOM, 无 CheckpointError, GPU 100% 满载 |
| **Epochs** | 3 epochs, 1250 steps/epoch |

详细指标见 [02_training_log/sft_live.md](../02_training_log/sft_live.md)

## ✅ 数据生成状态

| 指标 | 值 |
|------|-----|
| **SFT 样本** | 5,000 (0 issues) |
| **DPO 样本** | 186,896 (0 issues) |
| **验证** | 全部 4 项 EDA 检查通过 |
| **位置** | `/root/autodl-tmp/data/full5000/` |

详细验证见 [05_data/final_validation.md](../05_data/final_validation.md)

## 🚫 Blocker

**当前无 blocker。** SFT 训练正常运行中。

## ⏭️ 下一步

1. **SFT 完成** (预计剩余 ~1.7h) → 检查 loss 曲线和 checkpoint
2. **Stage II DPO** — 配置就绪:
   - bs=1, grad_accum=16, β=0.1, μ=0.05 (SFT anchor)
   - DPO reference model 独立加载
   - 预估 VRAM ~75GB / 96GB
   - 命令: `python src/training/train_dpo.py --config configs/default.yaml`
3. **评估** — 6 指标 × 9 基线

## 🟡 已知待解决问题

| 问题 | 严重度 | 状态 | 详情 |
|------|--------|------|------|
| CRB metric 占位符 (evaluate.py) | P2 | 未开始 | 依赖 `channel.compute_crb()` 未实现 |
| Multimodal 未实现 | P2 | 未开始 | 当前为 text-only BEV |
| `from_pretrained` 绕过 `__init__` | P2 | 未开始 | 新属性需手动同步 |
| 20 项验证缺口 | P0-P2 | 审计完成，未修复 | 见 [03_bugs/open/verification_gaps.md](../03_bugs/open/verification_gaps.md) |

## 📊 已解决 Bug 总览

| # | Bug | 严重度 | 阶段 | Commit |
|---|-----|--------|------|--------|
| 1 | 物理约束违反 (SCA-FP 随机初始化) | P0 | datagen | `1caa482`, `2b75aa1` |
| 2 | 环境多样性崩溃 (RNG pickle) | P0 | datagen | `8daddac` |
| 3 | 响应 JSON 截断 (512→1024→824 tokens) | P0 | datagen | `8daddac`, `223aace` |
| 4-8 | OOM 1-5 (HF wrapper, contiguous, GQA, CE, CheckpointError) | P0 | sft | 见 [oom_incidents.md](../02_training_log/oom_incidents.md) |
| 9-11 | 训练代码 bug (scheduler, zero_grad, LR) | P0 | sft | `4bc1a95`, `a52b4b8` |
| 12-19 | 服务器运行时错误 (Blackwell 8 连击) | P0 | sft | 见 [server_runtime_errors.md](../03_bugs/resolved/server_runtime_errors.md) |

**全部 P0 bug 已闭合。** 详见 [03_bugs/resolved/](../03_bugs/resolved/)

## 🔑 关键经验

1. **Unsloth 不可局部借用** — 一旦 import 就全局 monkey-patch transformers，与 Gemma 3 SDPA 冲突 → 彻底移除
2. **Blackwell sm_120 生态不成熟** (2026-06): bitsandbytes 不支持, FA2 无预编译 wheel, FlexAttention shared memory 不足
3. **BPE tokenizer 对浮点数碎片化严重** — 176 个 float32 可膨胀到 1678 tokens → compact JSON + 精度截断
4. **框约束 ≠ 球约束** — L-BFGS-B 的逐轴边界构建立方体，物理约束是球体

## 📋 快速命令参考

```bash
# 服务器登录后
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm

# 环境变量 (Blackwell 必须)
export TORCHINDUCTOR_FLEX_ATTENTION=0

# SFT 训练
python src/training/train_sft.py --config configs/default.yaml

# 过拟合测试 (5 min)
python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000

# 数据验证
python scripts/validate_data.py --data-dir /root/autodl-tmp/data/full5000
```
