---
type: postmortem
status: resolved
severity: P0
stage: sft
commits: [f49a5f7]
last_updated: 2026-06-28
related: [oom_1_through_5]
---

# Bug #9: Checkpoint 4GB — modules_to_save 保存完整词表权重

**来源**: 服务器磁盘告警 (97%) 诊断 | **发现者**: Claude disk usage analysis

## 症状

- 每个 Phase 1 LoRA checkpoint 大小为 **3.9GB**（正常应 ~100MB）
- 4 个 checkpoint 占满 50GB 数据盘（16GB/50GB）
- Step 250 最终保存因 `No space left on device` 崩溃

## 根因

`LoraConfig(modules_to_save=["embed_tokens", "lm_head"])` 告诉 PEFT：**将这两个模块的完整权重标记为可训练参数，并在 save_pretrained 时原封不动保存。**

- Gemma 3 12B 词表大小 = 256,128 tokens × 3,840 dims = **~2GB per module (bf16)**
- `embed_tokens` + `lm_head` = **~4GB per checkpoint**
- 实际只需 8 个 control token 的 embedding 行：8 × 3,840 × 2 bytes = **60KB**

`modules_to_save` 的作用是让新添加的 `<ctrl_0>..<ctrl_7>` token embedding 可训练，但副作用是把全部 256K 词表的权重都保存了。

## 修复 (commit f49a5f7)

### save_pretrained: 手动分离 LoRA vs modules_to_save

```python
# Before: 完整 PEFT save (LoRA + embed_tokens + lm_head → 4GB)
self.base_model.save_pretrained(os.path.join(save_dir, "lora"))

# After: 只保存 LoRA 权重 + 8 行 ctrl embedding
full_state = get_peft_model_state_dict(self.base_model)
lora_state = {k: v for k, v in full_state.items() if "lora_" in k}
safe_save_file(lora_state, "adapter_model.safetensors")
# 另存 8 行 ctrl embedding → ctrl_embed.pt (~60KB)
```

### from_pretrained: 加载后 patch control token 行

```python
# 检测新格式 (ctrl_embed.pt 存在) → 手动 patch
ctrl_embed = torch.load("ctrl_embed.pt", map_location="cpu")
embed.weight.data[control_token_ids] = ctrl_embed
```

### 训练路径不变

`__init__` 保留 `modules_to_save` — 训练时仍需 embed_tokens 可训练来学习 control token 的语义。

## 后向兼容

| checkpoint 格式 | 检测条件 | 加载路径 |
|----------------|----------|---------|
| 旧 (4GB, 含完整 embed) | `ctrl_embed.pt` 不存在 | PEFT 原生 load (完整 safetensors) |
| 新 (~100MB, LoRA only) | `ctrl_embed.pt` 存在 | PEFT load + 手动 patch ctrl rows |

旧 checkpoint 不受影响。

## 影响

- **修复前**: 50GB 数据盘存 4 个 checkpoint 即满，Step 250 保存崩溃
- **修复后**: 每个 checkpoint ~100MB，50GB 可存 500+ checkpoint
- Phase 2 DPO 训练不会再因 checkpoint 保存撑爆磁盘

## 教训

1. **`modules_to_save` 是双刃剑**：它把指定模块完整纳入训练和保存。对于大词表模型 (256K vocab)，embed_tokens 的完整保存代价极高。
2. **新增 special token 的 embedding 训练有更省的方案**：只保存/加载新 token 对应行，而非整个词表。
3. **磁盘空间应作为训练监控指标**：`shutil.disk_usage()` 在 save 前预检查可提前预警。
4. **每个 checkpoint 保存 tokenizer 也是浪费** (~20MB × N)：tokenizer 只在第一次保存即可。
