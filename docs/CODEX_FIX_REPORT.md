# Codex 修复说明

本文档说明本次对 UAV-ISAC-MLLM 项目做过的代码审阅和修复，供后续上服务器训练前交接使用。

## 背景

用户准备在服务器上正式训练，因此本次检查重点不是代码风格，而是会导致训练直接崩溃、训练语义偏离论文、或评估结果不可用的问题。

检查对象包括：

- `论文.txt`
- `docs/IMPLEMENTATION_REPORT.md`
- `configs/default.yaml`
- `src/model/`
- `src/data/`
- `src/training/`
- `src/eval/`
- `scripts/`

## 已修复问题

### 1. 修复 `losses.py` 导入错误

文件：`src/model/losses.py`

问题：

`compute_stage1_total()` 和 `compute_stage2_total()` 的返回类型注解使用了 `Tuple`，但文件中没有从 `typing` 导入 `Tuple`。在 Python 3.11 下导入模块时可能直接报 `NameError`。

修复：

```python
from typing import Dict, Optional, Tuple
```

影响：

避免训练脚本在导入 `src.model` 阶段直接失败。

### 2. 修复 control token 实际未插入的问题

文件：`src/data/dataset.py`

问题：

原实现中，SFT/DPO 数据集把论文里的 8 个 control token 位置写成了：

```python
[self.tokenizer.pad_token_id] * self.num_control_tokens
```

这意味着模型读出的 control hidden states 实际来自 padding token，而不是 `<ctrl_0>` 到 `<ctrl_7>`。这样即使训练成功，也不符合论文中的 control-token readout 机制。

修复：

- 数据集初始化时读取真实 control token id：

```python
self.control_token_ids = tokenizer.convert_tokens_to_ids(
    [f"<ctrl_{i}>" for i in range(num_control_tokens)]
)
```

- 输入序列中插入真实 control token：

```python
input_ids = prompt_ids + self.control_token_ids + response_ids
```

- 如果 tokenizer 中没有这些 token，会直接抛错，避免静默训练错误模型。

影响：

让 `control_mask` 标记的位置真正对应论文中的控制 token hidden states。

### 3. 修复 token-level loss 错误覆盖 control token 的问题

文件：`src/data/dataset.py`

问题：

原实现中 `labels` 和 `label_mask` 从 prompt 之后开始计算，这会把 control token 也当成 response token 训练。

修复：

现在 prompt 和 control token 都被 mask 掉，只对 oracle JSON response 计算语言模型 loss：

```python
labels = [-100] * (prompt_len + control_len) + response_ids
label_mask = [0] * (prompt_len + control_len) + [1] * len(response_ids)
```

影响：

符合论文中“token-level reasoning/formatting loss”和“control readout loss”分离的设计。

### 4. 为训练样本补充 `q_current`

文件：`src/data/oracle_generator.py`

问题：

论文中的 deployment projection 需要当前 UAV 位置 `q_current`，因为投影逻辑是：

```text
new_position = q_current + delta_q
```

但原数据集中没有保存 `q_current`，训练时 projection head 只能在没有当前坐标的情况下处理 displacement，和论文定义不一致。

修复：

SFT 和 DPO 样本都新增字段：

```json
"q_current": [...]
```

影响：

训练时可以把当前 UAV 位置传入 projection head，保证高度、区域和分离惩罚基于真实 UAV 绝对位置。

注意：

旧数据集不含 `q_current`，建议重新生成训练数据。至少 smoke test 数据必须重新生成。

### 5. 训练时传入 `q_current`

文件：

- `src/training/train_sft.py`
- `src/training/train_dpo.py`

问题：

原训练代码调用模型时没有传 `q_current`：

```python
outputs = model(..., labels=batch["labels"])
```

导致 projection head 无法按论文中的当前 UAV 位置进行投影。

修复：

训练时传入：

```python
q_current=batch["q_current"]
```

同时 separation penalty 现在使用绝对位置：

```python
q_hat = batch["q_current"] + outputs["delta_q"]
```

影响：

让 `L_sep` 更接近论文定义，而不是错误地在 displacement 空间中计算 UAV 间距。

### 6. 修复 DPO log-prob 计算会因 `-100` 崩溃的问题

文件：`src/training/train_dpo.py`

问题：

原 `_compute_logprob()` 中直接用 `labels` 做 `gather()`，但 prompt 部分 labels 是 `-100`：

```python
per_token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1))
```

这会导致 index out of bounds。

修复：

先把无效 label 替换为安全 token id，再用 `label_mask` 屏蔽：

```python
safe_labels = shift_labels.masked_fill(shift_labels < 0, 0)
per_token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
masked = per_token_logp * shift_mask
```

影响：

DPO 第一批 batch 不会因为 prompt mask 直接崩溃。

### 7. 调整 Unsloth LoRA 加载方式

文件：`src/model/gemma_isac.py`

问题：

原代码把 LoRA 参数直接传给：

```python
FastLanguageModel.from_pretrained(...)
```

这不是常见的 Unsloth 用法，存在版本兼容风险。

修复：

改成更标准的两步式：

```python
base_model, tokenizer = FastLanguageModel.from_pretrained(...)
base_model = FastLanguageModel.get_peft_model(...)
```

加载 checkpoint 时：

- 先扩展 tokenizer / embedding，加入 `<ctrl_*>`
- 再加载 LoRA adapter

影响：

降低服务器上因 Unsloth API 参数不兼容导致模型加载失败的风险。

### 8. 修复评估脚本变量作用域错误

文件：`src/eval/evaluate.py`

问题：

`_evaluate_one_sample()` 内部使用了未定义变量 `sim_cfg`：

```python
wavelength = 3e8 / (sim_cfg["carrier_freq_ghz"] * 1e9)
```

这会导致评估阶段报 `NameError`。

修复：

改为：

```python
wavelength = 3e8 / (cfg["simulation"]["carrier_freq_ghz"] * 1e9)
```

影响：

评估脚本不会在 sensing satisfaction 计算阶段直接崩溃。

### 9. 修复备用数据生成脚本噪声功率公式

文件：`scripts/run_data_generation.py`

问题：

备用脚本中噪声功率公式写成了近似错误形式，没有使用 `log10(B)`。

修复：

```python
noise_power = 10 ** (
    (-174 + 10 * np.log10(sc["bandwidth_mhz"] * 1e6)
     + sc["noise_figure_db"] - 30) / 10
)
```

并补充：

```python
import numpy as np
```

影响：

备用脚本和主脚本 `scripts/generate_data.py` 的噪声功率计算保持一致。

## 已做验证

在本地执行了语法编译检查：

```bash
python -m compileall -q src scripts
```

结果：通过。

说明：

本地环境没有安装 `torch`，因此没有在本地执行完整模型加载和训练 batch。需要在服务器环境中做 smoke test。

## 上服务器前必须注意

### 1. 旧训练数据建议废弃

因为旧数据集没有 `q_current` 字段，且 control token 的输入语义已经修复，建议重新生成数据。

至少先重新生成小规模 smoke test 数据：

```bash
PYTHONPATH=/root/UAV python scripts/generate_data.py \
  --num-env 2 \
  --num-restarts 2 \
  --output-dir /root/autodl-tmp/data/smoke
```

### 2. 先跑 smoke test，不要直接跑 5000 环境

建议先跑 1 到 2 个 batch 的 SFT：

```bash
PYTHONPATH=/root/UAV python src/training/train_sft.py \
  --config configs/default.yaml \
  --data_dir /root/autodl-tmp/data/smoke/sft_dataset.jsonl
```

确认以下事项：

- 模型能成功加载
- control token 能成功加入 tokenizer
- projection head 输出 shape 正确
- loss 正常下降或至少不是 NaN
- 显存没有立即 OOM

### 3. DPO 仍需单独小规模验证

DPO 同时涉及 train model 和 reference model，32GB 显存较紧。建议在正式 DPO 前先用极小数据跑通。

重点关注：

- `copy.deepcopy(model)` 是否能在服务器环境中正常工作
- reference model 是否导致 OOM
- 每个 batch 的显存峰值

如果 DPO OOM，可以考虑后续改成单独加载 reference model、CPU/offload reference，或使用 TRL/PEFT 推荐的 reference 策略。

## 仍未解决或需要确认的问题

### 1. 当前仍是文本版，不是真正多模态 MLLM

`configs/default.yaml` 中：

```yaml
use_multimodal: false
```

当前 BEV 使用文本表格，不是真实图像输入。严格来说，这和论文中的 MLLM / BEV image pipeline 仍有差距。

### 2. `mean_crb` 评估仍是占位

`src/eval/evaluate.py` 当前仍返回：

```python
"mean_crb": 0.0
```

如果要复现实验指标，需要后续接入真实 CRB 计算。

### 3. SCA-FP 仍是简化求解器

实现报告中也提到当前 beamforming / SCA-FP 是简化版本。它可以用于 pipeline 验证，但如果目标是严格复现论文数值结果，还需要进一步完善物理层优化器。

## 建议正式训练顺序

1. 重新生成 2 环境 smoke 数据。
2. 跑 SFT smoke test，确认不会崩。
3. 生成 20 到 50 环境小数据。
4. 跑短 SFT，检查 loss、输出 shape、显存。
5. 跑短 DPO，确认 reference model 不 OOM。
6. 再开始 5000 环境数据生成和正式训练。

## 总结

本次修复主要解决了以下高风险问题：

- 导入即崩
- control token 未真正生效
- SFT/DPO loss mask 错误
- DPO log-prob 索引崩溃
- projection head 缺少 `q_current`
- separation penalty 坐标语义错误
- Unsloth LoRA 加载兼容风险
- 评估脚本变量作用域错误
- 备用数据生成脚本噪声功率公式错误

修复后，代码更接近论文中“control-token readout + projection head + SFT/DPO”的设计，但仍建议先在服务器做小规模 smoke test，再投入正式训练费用。
