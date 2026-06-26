---
type: postmortem
status: resolved
severity: P1
stage: sft
commits: [8b5b8f1]
last_updated: 2026-06-26
related: [training_code_bugs]
---

# Bug: TensorBoard 日志静默丢失 — 缺失 `accelerator.init_trackers()`

**来源**: SFT 全量训练 live run | **发现者**: Friend (TensorBoard 端口活跃但无曲线)

## 症状

1. 训练正常进行，tqdm 进度条显示 loss 在下降
2. TensorBoard 进程正常启动，盯住了正确的 `/root/autodl-tmp/logs` 目录
3. **但 TensorBoard 页面无任何曲线** — SCALARS 面板为空，刷新无效
4. 检查 `/root/autodl-tmp/logs` 目录：**一个 event file 都没有**

## 根因

HuggingFace Accelerate 的 `accelerator.log()` **不是** 开箱即用的日志函数。它只是一个"请求"接口 — 将 `(tag, value, step)` 元组发送给后台 tracker。

**真正的磁盘 I/O 由 `accelerator.init_trackers()` 触发**。没有这行调用：

- `accelerator.log()` 照常执行，不报错（静音丢弃模式）
- TensorBoard event writer 从未创建
- 硬盘上 0 个 `events.out.tfevents.*` 文件
- TensorBoard 网页盯着空目录 → 空白面板

```python
# ❌ 错误: Accelerator 配置里声明了 log_with="tensorboard"，但没激活
accelerator = Accelerator(log_with="tensorboard", project_dir="/root/autodl-tmp/logs")
# ... 
accelerator.log(metrics, step=global_step)  # 全部进黑洞！

# ✅ 正确: 训练循环开始前必须显式初始化 tracker
accelerator.init_trackers("stage1_sft")  # 这一步才真正创建 SummaryWriter
```

这是 Accelerate 库的经典设计陷阱：`log_with` 参数只是"能力声明"，`init_trackers` 才是"物资下发"。无数用户被 `log_with` 的名字误导，以为配了就生效。

## 修复

**`src/training/train_sft.py`** — 在训练循环入口添加一行：

```python
# 约 line 230 (makedirs 之后, global_step=0 之前)
accelerator.init_trackers("stage1_sft")
```

**`src/training/train_dpo.py`** — 同样的问题，同样修复：

```python
# 约 line 306
accelerator.init_trackers("stage2_dpo")
```

**`configs/default.yaml`** — 缩短日志间隔，更快在 TensorBoard 看到曲线：

```yaml
# training.sft.logging_steps: 10 → 5
# 每 5 个 global step 打一个点 (~2-3 分钟可见首条曲线)
```

## 教训

1. **`log_with` ≠ 已启用** — Accelerate 的 `log_with` 只是注册 intent，`init_trackers()` 才是实例化后端的唯一入口。类似 `torch.no_grad()` 需要显式 enter，不是传个 flag 就行。
2. **静默失败是最坏的失败** — `accelerator.log()` 在 tracker 未初始化时不抛异常、不打印 warning，直接把数据丢弃。如果 API 能返回一个布尔值或发出 `logging.WARNING`，能省掉大量排查时间。
3. **最小化假设验证** — 训练脚本写完第一件事应该检查 `ls -la <log_dir>` 有没有 event file，而不是等到 TensorBoard 网页打开再看。

## 影响

- SFT 全量训练（~8.7h）前几分钟无任何日志记录，loss 曲线完全缺失
- 无法实时监控训练健康状况（loss 是否发散、NaN 是否出现）
- DPO 训练同样受影响（已一并修复）
- 已打断训练、加代码、重新启动 — 沉没成本仅几分钟 GPU 时间
