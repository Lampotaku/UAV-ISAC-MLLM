# UAV-ISAC-MLLM — Constraint-Aware MLLM for UAV-ISAC

**目标**: 用 Gemma 3 12B (LoRA + 约束投影头) 为 UAV-ISAC 的 SCA-FP 数值优化器提供智能热启动。

**完整文档**: [docs/09_handoff_document.md](docs/09_handoff_document.md) — 新对话必读

## 当前状态

- ✅ 全部源码完成，7 轮审查闭合
- ✅ GitHub 私有仓库: `Lampotaku/UAV-ISAC-MLLM`
- ⏳ 待执行: 服务器数据生成 → SFT 训练 → DPO 训练 → 评估

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
scripts/        → generate_data.py, validate_data.py, autodl_setup.sh
configs/        → default.yaml (全部超参数)
```

## 工作流

```
git clone → autodl_setup.sh → smoke test (5 envs)
→ validate → full generation (5000 envs)
→ Stage I SFT → Stage II DPO → evaluate
```

## 关键约定

- 所有路径用 `/root/autodl-tmp/`，不写系统盘
- 代码修改在本地 Windows，git push/pull 同步
- DPO reference model 独立加载（不 deepcopy，会 OOM）
- 数据生成支持 Ctrl+C 断点续跑
