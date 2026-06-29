---
type: reference
status: current
stage: data_regeneration
last_updated: 2026-06-29
---

# UAV-ISAC-MLLM — Documentation

**目标**: 用 Gemma 3 12B (LoRA + 约束投影头) 为 UAV-ISAC 的 SCA-FP 数值优化器提供智能热启动。

## 快速导航

### 🚀 新成员？从这里开始

| 顺序 | 文档 | 时间 | 内容 |
|------|------|------|------|
| **1** | [00_current/status.md](00_current/status.md) | **3 min** | 项目当前状态：数据层大修，求解器修复，等待数据重生 |
| **2** | [00_current/quickstart.md](00_current/quickstart.md) | **10 min** | 从零开始在服务器上跑起来（含验证步骤） |
| **3** | [00_current/canonical_config.md](00_current/canonical_config.md) | **5 min** | 当前 blessed 配置和 pipeline 命令 |
| **4** | [01_architecture/problem_formulation.md](01_architecture/problem_formulation.md) | **10 min** | 我们到底在解决什么问题？数学框架 |
| **5** | [01_architecture/system_design.md](01_architecture/system_design.md) | **10 min** | 模块拓扑、数据流、接口契约 |

**总计: ~30 分钟**理解整个项目。

### 📂 目录地图

```
docs/
├── README.md                          ← 你在这里
│
├── 00_current/                        ★ 项目脉搏 — 第一站
│   ├── status.md                      SFT 状态、blocker、下一步
│   ├── canonical_config.md            Blessed 配置 + server 命令
│   └── quickstart.md                  Zero-to-running
│
├── 01_architecture/                   稳定技术参考
│   ├── problem_formulation.md         UAV-ISAC 数学、系统模型
│   ├── system_design.md               模块拓扑、数据流
│   ├── training_pipeline.md           Stage I SFT + II DPO 设计
│   └── hardware_adaptation.md         Blackwell RTX PRO 6000 特定方案
│
├── 02_training_log/                   训练纪实
│   ├── sft_live.md                    SFT 训练指标和配置 (旧数据，已废弃)
│   ├── oom_incidents.md               OOM 1-7 诊断全链 ⭐
│   ├── data_degeneracy.md             数据退化根因分析 + 求解器修复 ⭐
│   ├── speed_optimization.md          21s→2.5s/step 提速战
│   └── phase1_status_2026-06-26.md   Phase 1 控制表示学习调试全纪录
│
├── 03_bugs/                           Bug 注册中心
│   ├── README.md                      严重度定义、如何登记
│   ├── resolved/                      已修复: 物理约束、RNG 崩溃、Token 溢出、OOM、训练代码
│   └── open/                          开放: 验证缺口审计
│
├── 04_reviews/                        代码审查历史
│   ├── README.md                      7 轮审查总结 + 累计修复表
│   ├── pre_launch/                    Rounds 1-6
│   └── multiprocessing_branch/        Round 7
│
├── 05_data/                           数据生成结果
│   ├── README.md                      结果时间线
│   └── final_validation.md            最终 5k SFT + 187k DPO 验证
│
├── 06_decisions/                      架构决策记录 (ADR)
│   ├── README.md                      ADR 索引
│   ├── adr_001_unsloth_removal.md     ★ 最重要的决策
│   ├── adr_002_dpo_independent_ref.md
│   ├── adr_003_sdpa_canonical.md
│   ├── adr_004_4bit_qlora_blackwell.md
│   ├── adr_005_control_token_mechanism.md
│   └── adr_006_data_regeneration.md   ★ 数据重生 + DPO 路线
│
├── 07_conventions/                    文档维护规范
│   ├── naming_conventions.md
│   ├── handoff_template.md
│   ├── bug_postmortem_template.md
│   └── archive_rules.md
│
└── 99_archive/                        已废弃 / 历史参考
    ├── README.md
    ├── deprecated_experiments/        失败方案 (Plan B 等)
    ├── old_results/                   早期数据验证结果
    ├── old_handoffs/                  历史交接文档 (handoff 2026-06-26, docs 13-26)
    └── old_setup_docs/                旧版项目文档 (docs 01/08/09)
```

### 🔗 关键外部资源

| 资源 | 路径/URL |
|------|----------|
| GitHub | `Lampotaku/UAV-ISAC-MLLM` (private) |
| 服务器 | AutoDL RTX PRO 6000 96GB, `/root/UAV-ISAC-MLLM` |
| 数据盘 | `/root/autodl-tmp/` (系统盘仅 30GB) |
| 配置文件 | `configs/default.yaml` |
| Conda env | `uavmllm` (Python 3.12) |

### 📖 阅读建议

- **接手的工程师**: 按快速导航 1→2→3→4→5 顺序
- **排查 bug**: 先去 [03_bugs/resolved/](03_bugs/resolved/) 看有没有一样的
- **理解架构决策**: 去 [06_decisions/](06_decisions/) 看 ADR
- **查训练状态**: 看 [00_current/status.md](00_current/status.md)
- **找废弃方案**: 去 [99_archive/](99_archive/)
