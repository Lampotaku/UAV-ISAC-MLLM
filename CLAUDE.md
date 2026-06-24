# UAV-ISAC-MLLM — Constraint-Aware MLLM for UAV-ISAC

**目标**: 用 Gemma 3 12B (LoRA + 约束投影头) 为 UAV-ISAC 的 SCA-FP 数值优化器提供智能热启动。

**交接文档 (新成员必读)**:
- [docs/05_handoff/16_handoff_01_project_direction.md](docs/05_handoff/16_handoff_01_project_direction.md) — 论文方向
- [docs/05_handoff/17_handoff_02_pre_datagen.md](docs/05_handoff/17_handoff_02_pre_datagen.md) — 数据生成前准备
- [docs/05_handoff/18_handoff_03_datagen_problems.md](docs/05_handoff/18_handoff_03_datagen_problems.md) — 数据生成问题与修复
- [docs/05_handoff/19_handoff_04_post_datagen.md](docs/05_handoff/19_handoff_04_post_datagen.md) — 当前状态与下一步

## 当前状态

- ✅ 全部源码完成，7 轮审查闭合 + 一审修复闭合
- ✅ GitHub 私有仓库: `Lampotaku/UAV-ISAC-MLLM`
- ✅ 5000 环境数据生成完成 (SFT: 5000, DPO: 186,896, 0 issues)
- ✅ feature/multiprocessing → master 已合并
- ⏳ 待执行: 过拟合测试 → SFT 训练 → DPO 训练 → 评估

## 关键环境信息

| 项 | 值 |
|----|-----|
| 本地 | Windows, `h:\Projects\UAV` |
| 服务器 | AutoDL RTX 5090 32GB, `/root/UAV-ISAC-MLLM` |
| 数据盘 | `/root/autodl-tmp/` (系统盘仅 30GB) |
| GPU | Blackwell sm_120, CUDA 12.8 |
| 量化 | Unsloth 4-bit QLoRA (bitsandbytes 不支持 Blackwell) |
| Python | 3.11, conda env: `uavmllm` |

## 架构速览

```
src/env/        → 仿真环境 (UAV 拓扑 + 物理信道 + 场景生成)
src/solver/     → SCA-FP 数值优化器 (交替优化)
src/data/       → 数据层 (Prompt 构造 + Oracle 生成 + Dataset)
src/model/      → Gemma3ISAC + ProjectionHead + Losses
src/training/   → Stage I SFT + Stage II DPO
src/eval/       → 评估 (6 指标 × 9 基线)
scripts/        → generate_data.py, validate_data.py, eda_data.py, test_sft_overfit.py, autodl_setup.sh
configs/        → default.yaml (全部超参数)
```

## 工作流

```
✅ git clone → autodl_setup.sh → smoke test (5 envs)
✅ validate → full generation (5000 envs) → EDA
⏳ overfitting test (5 min) → Stage I SFT → Stage II DPO → evaluate
```

### 下一步 (服务器上执行)

```bash
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm
# Step 1: 过拟合测试 (5 min, 证明训练代码正确)
python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000
# Step 2: 通过后启动 SFT
python src/training/train_sft.py --config configs/default.yaml
```

## 关键约定

- 所有路径用 `/root/autodl-tmp/`，不写系统盘
- 代码修改在本地 Windows，git push/pull 同步
- DPO reference model 独立加载（不 deepcopy，会 OOM）
- 数据生成支持 Ctrl+C 断点续跑
