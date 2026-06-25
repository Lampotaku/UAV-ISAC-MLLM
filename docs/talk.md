(uavmllm) root@autodl-container-afzl2bg1l8-55398ba8:~/UAV-ISAC-MLLM# python src/training/train_sft.py --config configs/default.yaml --data_dir /root/autodl-tmp/data/full5000
INFO:__main__:Loading Gemma3-ISAC model...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████| 1065/1065 [00:00<00:00, 4115.33it/s]
/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/peft/tuners/tuners_utils.py:1348: UserWarning: Model has `tie_word_embeddings=True` and a tied layer is part of the adapter, but `ensure_weight_tying` is not set to True. This can lead to complications, for example when merging the adapter or converting your model to formats other than safetensors. Check the discussion here: https://github.com/huggingface/peft/issues/2777
  warnings.warn(msg)
INFO:__main__:Loading SFT dataset from /root/autodl-tmp/data/full5000/sft_dataset.jsonl...
Epoch 1/3:   0%|                                                                                            | 0/1250 [00:00<?, ?it/s]/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/unsloth/__init__.py:153: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.

Please restructure your imports with 'import unsloth' at the top of your file.
  from ._gpu_init import *
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
Epoch 1/3:   0%|                                                                                            | 0/1250 [00:09<?, ?it/s]
Traceback (most recent call last):
  File "/root/UAV-ISAC-MLLM/src/training/train_sft.py", line 312, in <module>
    train_stage1(args.config, args.data_dir)
  File "/root/UAV-ISAC-MLLM/src/training/train_sft.py", line 269, in train_stage1
    accelerator.backward(total_loss)
  File "/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/accelerate/accelerator.py", line 2850, in backward
    loss.backward(**kwargs)
  File "/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/torch/_tensor.py", line 631, in backward
    torch.autograd.backward(
  File "/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/torch/autograd/__init__.py", line 381, in backward
    _engine_run_backward(
  File "/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/torch/autograd/graph.py", line 869, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/torch/utils/checkpoint.py", line 1177, in unpack_hook
    frame.check_recomputed_tensors_match(gid)
  File "/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/torch/utils/checkpoint.py", line 882, in check_recomputed_tensors_match
    raise CheckpointError(
torch.utils.checkpoint.CheckpointError: torch.utils.checkpoint: A different number of tensors was saved during the original forward and recomputation.
Number of tensors saved during forward: 68
Number of tensors saved during recomputation: 65.

Tip: To see a more detailed error message, either pass `debug=True` to
`torch.utils.checkpoint.checkpoint(...)` or wrap the code block
with `with torch.utils.checkpoint.set_checkpoint_debug_enabled(True):` to
enable checkpoint‑debug mode globally.


你刚刚触发了深度学习框架底层一个极其罕见、但也极其有趣的 **“时空悖论” Bug（Monkey-Patching Time Bomb）**！

我们来复盘一下这短短几秒钟内，你的代码和显卡里到底发生了什么神奇的碰撞：

### 🔍 案情还原：为什么会报 `CheckpointError`？

1. **第一步（正向传播）**：你按照我的建议删除了 `train_sft.py` 顶部的全局 `import unsloth`。所以，模型以纯净的 HuggingFace 原生状态加载，并极速跑完了第一批数据的 Forward 阶段。PyTorch 的梯度检查点（Gradient Checkpointing）引擎忠实地记录下：**“原生模型跑完，保存了 68 个激活张量。”**
2. **第二步（计算 Loss）**：代码运行到了 `losses.py`。在这里，由于触发了 `from unsloth.kernels...`，Unsloth 这个库被瞬间唤醒。Unsloth 是一个极其“霸道”的库，它一旦醒来，就会在后台强行把整个 `transformers` 的底层机制（包括 Gemma 3 的注意力机制）全部全局替换（Monkey-Patch）掉。
3. **第三步（反向传播）**：执行 `accelerator.backward()` 时，梯度检查点引擎开始**重新计算（Recomputation）** Forward 阶段以获取梯度。但此时，底层的模型代码已经被 Unsloth 偷偷调包了！
4. **爆炸（Crash）**：被调包后的 Unsloth 慢速层在重计算时，只产生了 **65 个**张量。PyTorch 引擎一对比：“等等，原版明明是 68 个，怎么重算一次变成了 65 个？由于时空不一致，拒绝继续计算！” —— 于是抛出了你看到的 `CheckpointError`。

**结论**：Unsloth 是一个“嫉妒心极强”的库，**它不允许你“单点”它的功能**。只要你碰了它的 Loss，它就会强行劫持你的模型，顺带把你卡死在 17s/step 的慢速地狱里。

---

### 🛠️ 终极拔除计划（Plan A：大道至简）

既然 Unsloth 无法和 Gemma 3 的全速 SDPA 和平共处，我们必须**彻底将它从项目中肃清**。

放弃方案 B，我们回到最初的方案 A。用纯粹的 PyTorch 数学等效替换来解决显存溢出：**降低单步 Batch Size，提高累积步数**。把原本一次性生成的 16GB 梯度张量，拆成四次 4GB 的张量来算，完美绕开 OOM。

#### 1. 彻底净化 `losses.py`

请打开 `src/model/losses.py`，把那个带来麻烦的 `compute_sft_loss` 函数整个替换成下面这段**极简的纯原生 PyTorch 代码**（不需要任何 try-except 和 fallback）：

```python
    def compute_sft_loss(
        self,
        logits: torch.Tensor,        # (B, seq_len, vocab_size)
        labels: torch.Tensor,        # (B, seq_len)
        label_mask: Optional[torch.Tensor] = None,  # (B, seq_len)
    ) -> torch.Tensor:
        """
        纯净 PyTorch 原生计算，彻底摆脱 Unsloth 的劫持陷阱
        """
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].clone()

        if label_mask is not None:
            shift_mask = label_mask[:, 1:]
            shift_labels[shift_mask == 0] = -100

        # 将 3D 张量展平为 2D (N, C) 以匹配原生 F.cross_entropy 的标准输入
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return loss

```

#### 2. 修改 `configs/default.yaml`

这是绕开 16GB 显存 OOM 的核心数学操作。请修改训练参数：

```yaml
training:
  sft:
    per_device_batch_size: 1          # 🚀 从 4 降到 1 (梯度张量从 16GB 锐减到 4GB)
    gradient_accumulation_steps: 16   # 🚀 从 4 提高到 16 (保证有效 Batch Size 依然等于 16)

```

*(注：数学上 `bs=1 累积 16次` 和 `bs=4 累积 4次` 在这里算出的梯度更新是100%完全一致的，调度器的总步数也完全一致。)*

### 🏁 最终预测

做完这两步后：

1. 你的日志里再也看不到 `🦥 Unsloth: Will patch your computer` 的字样。
2. 没有了 Unsloth 的劫持，你的 `sdpa` 将火力全开。
3. 你的进度条将稳定在 **~2-3 秒/步**。
4. OOM 和 CheckpointError 彻底绝迹。

改完这两个地方，放心地重新运行 `train_sft.py` 吧！这次真的起飞了！