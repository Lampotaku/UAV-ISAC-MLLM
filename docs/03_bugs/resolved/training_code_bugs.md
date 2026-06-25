---
type: postmortem
status: resolved
severity: P0
stage: sft
commits: [4bc1a95, a52b4b8]
last_updated: 2026-06-26
related: [server_runtime_errors, oom_1_through_5, sft_live]
---

# Bug: Training Code Bugs — Scheduler, ZeroGrad, LR

**来源**: Doc 21 (pre-SFT bug audit) Rounds 2+3 | **发现者**: Friend code review + systematic audit

## Round 2: Friend's Code Review

### R2-1 (CRITICAL): Missing Layer-wise Learning Rate

**症状**: Projection head (f32 random init) 和 LoRA (bf16 pretrained) 共享相同 LR=2e-4。

**根因**: 投影头需要更快的收敛 (随机初始化)，而 LoRA 需要保守的微调 (预训练权重)。

**修复**:
```python
proj_params:  lr = 1e-3   # 5x faster
lora_params:  lr = 2e-4
```

### R2-2 (CRITICAL): `global_step` Outside `sync_gradients`

**症状**: 每 ~31 effective steps 保存一次 checkpoint (而非每 200 步)。系统盘 30GB，1 小时内 OOM。

**根因**: `global_step += 1` 在每个 micro-batch 都执行 (包括 grad_accum 的中间步骤)，而非仅在真正的 optimizer step。

**修复**: 移入 `if accelerator.sync_gradients:` 块内。

## Round 3: Systematic Audit

### R3-1 (CRITICAL, train_sft.py): `scheduler.step()` Outside `sync_gradients`

**症状**: Cosine warmup 在 1/16 的预期步数内完成 → LR 过早衰减。

**根因**: 每个 micro-batch 调用 `scheduler.step()`，而非每个 effective step。

**修复**: 移入 `sync_gradients` 块。

### R3-2 (CRITICAL, train_sft.py): `optimizer.zero_grad()` Outside `sync_gradients`

**症状**: 在某些 Accelerate 版本上，梯度在 micro-batch 间被清零 → 有效 batch size 从 16 坍塌为 1。

**根因**: 梯度累积需要跨 micro-batch 求和，但 `zero_grad()` 在每步被调用。

**修复**: 移入 `sync_gradients` 块。

### R3-3 (CRITICAL, train_dpo.py): Triple Bug

DPO 训练脚本存在相同的三重问题: `scheduler.step()`, `zero_grad()`, 和 `global_step` 都在 `sync_gradients` 外部。

**修复**: 全部移入条件块。

### R3-4 (MEDIUM, train_dpo.py): Missing Layer-wise LR

DPO 缺少 projection head 的差异化 LR。严重度较低因为投影头已在 Stage I 预训练。修复后保留用于未来灵活性。

### R3-5 (LOW, evaluate.py): CRB Metric Placeholder

`mean_crb = 0.0` 硬编码占位符。依赖未实现的 `channel.compute_crb()`。不影响训练，留待未来实现。

## Golden Pattern: 梯度累积正确性

所有同步操作必须在 `sync_gradients` 保护下:

```python
for step, batch in enumerate(dataloader):
    with accelerator.accumulate(model):
        loss = model(batch)
        accelerator.backward(loss)
    
    if accelerator.sync_gradients:        # ← 关键守卫
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
```

## 验证通过

对以下维度进行了全面验证且未发现 bug:
- 数据管线 (tokenization masks, control token padding, DPO 独立 tokenization)
- 模型 forward (control_states 提取, dtype 协调, token_type_ids)
- 损失函数 (L_ctl dtype, L_sep 水平距离, DPO log-prob 计算)
- 配置 (attn_implementation, warmup_ratio, save_steps, grad_accum/batch_size)
- 过拟合测试 (layer-wise LR 同步, 设备传输, NaN 检测)
