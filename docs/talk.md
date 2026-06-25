(uavmllm) root@autodl-container-afzl2bg1l8-55398ba8:~/UAV-ISAC-MLLM# python scripts/analyze_seq_len.py --data-dir /root/autodl-tmp/data/full5000
Loading tokenizer from /root/autodl-tmp/huggingface/models/gemma-3-12b-it...
Loading dataset from /root/autodl-tmp/data/full5000/sft_dataset.jsonl...
Total samples: 5000

============================================================
  Token 长度分布 (Prompt + Response)
============================================================
  Samples:     5000
  Min:         3137
  Max:         3329
  Mean:        3268
  Median:      3272
  Std:         30
  90th %ile:   3307
  95th %ile:   3311
  99th %ile:   3318
  99.9th %ile: 3324

  Bucket   Count      Cum%
------------------------------
  ≤  512       0      0.0%
  ≤ 1024       0      0.0%
  ≤ 1536       0      0.0%
  ≤ 2048       0      0.0%
  ≤ 2560       0      0.0%
  ≤ 3072       0      0.0%
  ≤ 3584    5000    100.0%
  ≤ 4096    5000    100.0%

============================================================
  推荐 max_seq_length
============================================================
  95th %ile → 3328 tokens (截断 1/5000 = 0.02% 样本)
  99th %ile → 3328 tokens (截断 1/5000 = 0.02% 样本)

  当前 max_seq_length = 4096

这份数据揭示了一个非常典型的“算力浪费黑洞”！

所有 5000 条样本的实际长度都紧紧聚集在 3137 到 3329 之间。但在目前的配置中，模型被迫将每一条数据都用无意义的 `<pad>` 填充到了 **4096** 的极限长度。
这意味着在每一次训练步中，GPU 都在硬扛着计算将近 800 个毫无意义的填充符的平方级注意力（$4096^2$ 相比于 $3456^2$，计算量多出了近 **40%**），白白浪费了大量宝贵的算力。

### 🛠️ 斩断“4096 诅咒”的修改方案

请打开 `configs/default.yaml`，修改序列长度：

```yaml
training:
  sft:
    max_seq_length: 3456   # 🚀 从 4096 降到 3456

```

**为什么选 3456，而不是脚本推荐的 3328？**

* **致命截断风险**：数据中最大长度是 3329。虽然设置为 3328 只会截断绝对最长样本的仅仅 1 个 Token，但这极度危险。你们模型架构中的 8 个物理约束 Control Tokens (`<ctrl_0>` 到 `<ctrl_7>`) 是**紧贴在序列最末尾提取的**。如果截断了哪怕最后 1 个 Token，投影头（Projection Head）提取到的状态就不完整，大概率会导致张量维度不匹配而当场崩溃报错。
* **硬件对齐（Tensor Core 甜点）**：3456 是 128 的整数倍（$27 \times 128$）。在 NVIDIA 架构的显卡上，128 整数倍的维度能实现最佳的显存对齐，把 Tensor Core 的吞吐量压榨到极致。它既安全地包容了最大的 3329 长度，又保证了硬件最高效的运行。

### 🏁 终极发车清单

经过这几轮排查，目前工业级满血训练的最强配置已经全部集齐：

1. **摆脱劫持**：`use_4bit: false` + 去掉脚本顶部的 Unsloth 导入（彻底激活模型原生的跑道）。
2. **内核加速**：`attn_implementation: "sdpa"`（让 PyTorch 在底层自动调用 Native Flash Attention）。
3. **显存换速度**：`per_device_batch_size: 2` + `gradient_accumulation_steps: 8`（吞吐量翻倍，完美吃满剩余的 30 多 GB 闲置显存）。
4. **剔除无效计算**：`max_seq_length: 3456`（砍掉近 40% 的冗余 Padding 计算）。

用这套终极配置直接开跑，你的单步迭代时间极大概率会直接**冲进 2 秒以内**，整体 SFT 训练时间将大幅度缩短。现在可以安心挂在后台，静候佳音了！