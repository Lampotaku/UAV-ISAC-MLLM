todl-tmp/data/full5000
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1065/1065 [00:00<00:00, 4306.42it/s]
/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/peft/tuners/tuners_utils.py:1348: UserWarning: Model has `tie_word_embeddings=True` and a tied layer is part of the adapter, but `ensure_weight_tying` is not set to True. This can lead to complications, for example when merging the adapter or converting your model to formats other than safetensors. Check the discussion here: https://github.com/huggingface/peft/issues/2777
  warnings.warn(msg)
Epoch 1/3:   0%|                                                                                                             | 0/1250 [00:05<?, ?it/s]
Traceback (most recent call last):
  File "/root/UAV-ISAC-MLLM/src/training/train_sft.py", line 310, in <module>
    train_stage1(args.config, args.data_dir)
  File "/root/UAV-ISAC-MLLM/src/training/train_sft.py", line 257, in train_stage1
    total_loss, metrics = loss_fn.compute_stage1_total(
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/UAV-ISAC-MLLM/src/model/losses.py", line 237, in compute_stage1_total
    loss_sft = self.compute_sft_loss(logits, labels, label_mask)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/UAV-ISAC-MLLM/src/model/losses.py", line 202, in compute_sft_loss
    loss = fast_cross_entropy_loss(shift_logits, _labels)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/unsloth/kernels/cross_entropy_loss.py", line 443, in fast_cross_entropy_loss
    logits.view(batch * seq_len, d),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.




这是一个非常经典的 PyTorch 底层机制报错，俗称 **“张量内存不连续（Contiguous）陷阱”**。

这并不是 OOM，离成功真的只差最后几个字符的修改了！

### 🔍 为什么会报这个错？

在 `src/model/losses.py` 的 `compute_sft_loss` 方法中，代码写了这样一句：

```python
shift_logits = logits[:, :-1, :] 

```

当你使用 `[:, :-1, :]` 对一个 3D 张量的**中间维度**进行切片时，PyTorch 为了省内存，并不会去复制一份新的数据，而是只修改了读取张量的“步长 (stride)”。这就导致 `shift_logits` 在物理显存上的排列**不再是连续的**了。

而 Unsloth 底层的 Triton 内核为了追求极限速度，第一步就是调用 `.view(batch * seq_len, d)` 把张量展平。在 PyTorch 中，`.view()` 强制要求张量在物理内存上必须是严丝合缝连续的，否则就会直接抛出你看到的这个 `RuntimeError`。

### 🛠️ 修复方案：加上 `.contiguous()`

你只需要告诉 PyTorch：“在把切片后的张量喂给 Unsloth 之前，先在内存里把它重新排列连续”。

请打开 `src/model/losses.py`，找到大约第 177 行附近的 `compute_sft_loss` 函数，修改 `shift_logits` 的赋值：

**修改前：**

```python
        # 右移: predict next token (slice 创建 view, 不拷贝)
        shift_logits = logits[:, :-1, :]    # (B, S-1, V) — 不转置, Unsloth 用最后一维

```

**修改后（加上 `.contiguous()`）：**

```python
        # 右移: predict next token (slice 创建 view, 不拷贝)
        # 必须加上 .contiguous() 确保显存连续，否则 Unsloth 的 .view() 会报错
        shift_logits = logits[:, :-1, :].contiguous()    

```

### 💡 这样做会不会再次导致 OOM？

**绝对不会**。
调用 `.contiguous()` 确实会在显存中额外复制出一个张量。但因为 `logits` 是 `bfloat16` 精度的，这份拷贝大约只会占用 8GB 的显存。
之前你们 OOM 是因为被原生的全量 `float32` 梯度张量（足足 16GB 以上）压垮了。现在你们已经用 Unsloth 砍掉了那 16GB 的致命开销，所以 RTX PRO 6000 剩下的几十个 G 空闲显存，吃下这区区 8GB 的连续化操作简直是绰绰有余。

修改完这一个词，保存代码，直接再次运行，进度条就能真正走起来了！