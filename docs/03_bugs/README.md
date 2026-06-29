---
type: reference
status: current
stage: all
last_updated: 2026-06-29
---

# Bug Registry

## 严重度定义

| 级别 | 定义 | 示例 |
|------|------|------|
| **P0** | 训练/数据无效，或结果不可用 | 所有 SFT 响应截断、物理约束违反 |
| **P1** | 训练运行但严重降级 | DPO validation 被绕过 |
| **P2** | 次优但可运行 | 硬编码常数的消除 |
| **P3** | 装饰性 / 未来增强 | 文档改进 |

## 已解决 Bugs

| # | Bug | 严重度 | 阶段 | 文件 |
|---|-----|--------|------|------|
| 1 | 物理约束违反 (SCA-FP 随机初始化) | P0 | datagen | [physical_constraint.md](resolved/physical_constraint.md) |
| 2 | 环境多样性崩溃 (RNG pickle) | P0 | datagen | [rng_diversity_collapse.md](resolved/rng_diversity_collapse.md) |
| 3 | 响应 JSON 截断 (BPE 碎片化) | P0 | datagen | [response_token_overflow.md](resolved/response_token_overflow.md) |
| 4 | 服务器运行时错误 (Blackwell 8 连击) | P0 | sft | [server_runtime_errors.md](resolved/server_runtime_errors.md) |
| 5 | 训练代码 Bug (scheduler/zero_grad/LR) | P0 | sft | [training_code_bugs.md](resolved/training_code_bugs.md) |
| 6 | OOM 1-5 (HF wrapper→CE→CheckpointError) | P0 | sft | [oom_1_through_5.md](resolved/oom_1_through_5.md) |
| 7 | TensorBoard 日志静默丢失 (缺失 init_trackers) | P1 | sft | [tensorboard_init_trackers.md](resolved/tensorboard_init_trackers.md) |
| 8 | Eval Pipeline 审查 — 7 处缺陷闭合 (tqdm/CPU/加速比) | P0 | eval | [eval_pipeline_7_bugs.md](resolved/eval_pipeline_7_bugs.md) |
| 9 | Checkpoint 4GB→100MB (modules_to_save 完整权重保存) | P0 | sft/eval | [checkpoint_modules_to_save_4gb.md](resolved/checkpoint_modules_to_save_4gb.md) |
| 10 | OOM #6-7 (Phase 2 切换泄漏 + Grad diag retain_graph) | P0 | sft | 详见 [oom_incidents.md](../02_training_log/oom_incidents.md) |
| 11 | SFT 模态坍塌 — 数据退化 (求解器无高度 trade-off) | P0 | datagen | [data_degeneracy.md](../02_training_log/data_degeneracy.md) |
| 12 | 代码审查 3 连杀 (converged/max_iters 错引, 额外 SCA-FP 调用, calibrate baseline 错误) | P1 | sft/datagen | [implementation_2026-06-29.md](../02_training_log/implementation_2026-06-29.md) |

**全部 P0/P1 bug 已解决。** 详见各阶段 postmortem。

## 开放 Issues

| # | Issue | 严重度 | 阶段 | 文件 |
|---|-------|--------|------|------|
| 1 | 验证缺口审计 (20 项) | P0-P2 | all | [verification_gaps.md](open/verification_gaps.md) |
| 2 | DPO 在新数据上的表现待验证 | P0 | dpo | [adr_006](../06_decisions/adr_006_data_regeneration.md) |

## 如何登记新 Bug

1. 在 `resolved/` 或 `open/` 中创建新文件
2. 使用 [bug postmortem 模板](../07_conventions/bug_postmortem_template.md)
3. 添加 metadata header (YAML frontmatter)
4. 更新本文件 (README) 的对应表格
5. 更新 [00_current/status.md](../00_current/status.md) 如果该 bug 是新的 blocker

## Bug 文件命名

```
{short_kebab_description}.md
```
例: `physical_constraint.md`, `rng_diversity_collapse.md`
