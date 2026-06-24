# 第十二号文档 — 剩余验证缺口全面审计

> 审计时间: 2026-06-24 | 触发: "还有什么方向是没验证到的？"
> 状态: 逐项列出，标注优先级 | 范围: 整个数据→训练→评估管线

---

## 目录

1. [一句话概述](#一句话概述)
2. [已验证 vs 未验证全景图](#已验证-vs-未验证全景图)
3. [P0 — 会阻塞或静默毁掉训练](#p0-会阻塞或静默毁掉训练)
4. [P1 — 会显著降低训练质量](#p1-会显著降低训练质量)
5. [P2 — 值得监控但非阻塞](#p2-值得监控但非阻塞)
6. [建议的验证顺序](#建议的验证顺序)

---

## 一句话概述

**当前验证体系覆盖了物理合法性（validate_data.py）和部分统计多样性（eda_data.py Section 1-3），但管线中还留有 20 个未验证维度，其中 3 个 P0 级（实际 tokenizer 计数、Unsloth Blackwell 兼容性、Gemma3 权重可访问性）、7 个 P1 级（求解器行为、DPO 质量、数据泄漏）、10 个 P2 级。**

---

## 已验证 vs 未验证全景图

```
数据生成管线 ──────────────────────────────────────────────── 训练管线 ──────────────────
                                                                   
[环境采样] ──→ [SCA-FP 求解] ──→ [Prior提取] ──→ [Prompt构造] ──→ [Tokenizer] ──→ [训练]
    │               │               │               │               │              │
    │               │               │               │               │              │
    ✅ 位置多样性    ❓ 收敛率       ✅ 维度正确      ✅ 格式完整      ❓ 实际token计数  ❓ Unsloth加载
    ❓ 用户位置多样  ❓ 解多样性     ❓ NaN/Inf       ❓ 摘要数值多样  ❓ 浮点数分词     ❓ 权重下载
    ❓ 目标位置多样  ❓ 运行时分布   ❓ SFT-DPO泄漏   ❓ BEV网格多样   ❓ 特殊字符分词   ❓ VRAM fit
    ❓ 信道增益合理  ❓ 边界饱和原因 ✅ JSON可解析    ✅ 必需字段      ❓ 截断确认       ❓ LoRA模块名
    ✅ δ_q 方向分布  ❓ NaN出现频率  ✅ 字段完整      ❓ Prompt长度    ❓ Control token  ❓ DPO ref model
    ✅ δ_p 功率约束  ❓ 10次重启全   ✅ 功率≤P_max                     ❓ Gemma3 tokenizer ❓ Sinkhorn收敛
    ✅ δ_a 关联约束     收敛?                                                                 
    ❓ 信道增益范围
```

图例: ✅ = 已验证 | ❓ = 未验证 | ❗ = 有风险

---

## P0 — 会阻塞或静默毁掉训练

### P0-1: 实际 Tokenizer 计数 vs 启发式估算 🔴

**现状**: `eda_data.py` 使用 `chars/4 + digits/2.5` 估算 token 数。这是启发式，不是真实计数。

**风险**: Gemma 3 使用 SentencePiece (Unigram) tokenizer。浮点数（如 `-10.450347900390625`）的分词行为不可预测：
- `-` 可能是独立 token
- `10` 可能是 1 个 token
- `.450347900390625` 可能被拆成多个 token
- 每个 17-位浮点数可能占用 3-8 个实际 token

176 个浮点数 × 每数 3-8 token = **实际 response 可能达 900-1400 tokens**，超出 1024 budget。

**验证方法**:
```python
# 在服务器上跑 1 条真实数据
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
# 读取一条 sft_dataset.jsonl 的第一行
import json
with open("/root/autodl-tmp/data/smoke20/sft_dataset.jsonl") as f:
    sample = json.loads(f.readline())
resp_tokens = len(tokenizer.encode(sample["response"]))
prompt_tokens = len(tokenizer.encode(sample["prompt"]))
print(f"Actual prompt tokens: {prompt_tokens}  (budget: 3072)")
print(f"Actual response tokens: {resp_tokens}  (budget: 1024)")
```

**阈值**: 如果 `resp_tokens > 1000`（留 24 token 安全余量），需要进一步缩小 response 或增大 budget。

---

### P0-2: Unsloth 4-bit 在 RTX 5090 Blackwell 上的实际兼容性 🔴

**现状**: `gemma_isac.py:69` 尝试 `from unsloth import FastLanguageModel`。CLAUDE.md 注明 "bitsandbytes 不支持 Blackwell"。

**风险**: 
- Unsloth 对 Gemma 3 的支持状态未知（Gemma 3 是 2025 年的新模型）
- Unsloth 对 Blackwell sm_120 的原生内核支持可能不完整
- 如果 Unsloth 不支持，训练需要 fallback 方案（如加载 FP16 到 CPU offload、或切换为 8-bit via `torchao`）
- 最坏情况：模型加载失败，整个 GPU 训练计划需要重新设计

**验证方法**:
```python
# 烟雾测试 — 在 RTX 5090 上尝试加载
from unsloth import FastLanguageModel
import torch
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-12b-it",
    max_seq_length=4096,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
# 跑一条推理
inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
_ = model.generate(**inputs, max_new_tokens=5)
print("✅ Inference OK")
```

---

### P0-3: Gemma 3 12B 权重是否可下载 🔴

**现状**: `configs/default.yaml:50` 配置 `backbone: "google/gemma-3-12b-it"`。Gemma 模型是 gated — 需要 HuggingFace 认证。

**风险**: 
- AutoDL 服务器可能没有 HF token 配置
- 如果没有提前申请 Gemma 3 访问权限，`from_pretrained()` 会直接报错
- 24GB 权重下载需要 ~30 分钟 + 100GB 磁盘空间

**验证方法**:
```bash
huggingface-cli whoami                    # 检查是否登录
huggingface-cli login --token hf_xxx      # 如未登录
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/gemma-3-12b-it'); print('✅ Access OK')"
```

---

## P1 — 会显著降低训练质量

### P1-1: SCA-FP 求解器收敛行为未审计

**现状**: 每个环境跑 10 次随机重启，按效用排序。但没有任何指标记录：
- **收敛率**: 10 次中有几次真正收敛（`converged=True`）？
- **解多样性**: 10 次重启产生的解是否不同？如果求解器总是收敛到同一局部最优，DPO 的 "chosen vs rejected" 只是数值噪声
- **NaN 出现频率**: `sca_fp.py:170` 有 NaN guard（`break if not np.isfinite(utility)`），但不记录频率。如果某些环境频繁产生 NaN，说明数值稳定性有问题

**验证方法**: 在 EDA 中新增 "Section 4: Solver Behavior"
- 对 smoke20 数据，分析 10 次重启的效用方差
- 如果 10 次重启的效用 max-min < 1.0（噪声级），则 DPO 对无意义
- 对比最优解和次优解的 δ_q/δ_a/δ_p 差异

**代码位置**: `src/solver/sca_fp.py:145-177`（`solve()` 主循环）、`src/data/oracle_generator.py:112-115`（10 次重启调用）

---

### P1-2: DPO 对质量分布未审计

**现状**: EDA 只 spot-check 5 条 DPO。193,831 对的质量分布未知。

**关键问题**:
1. **效用间距分布**: `delta_min = 0.2 * IQR` 作为过滤阈值。实际 IQR 是多少？过滤后还剩多少对？
2. **每环境产出对数**: 10 次重启最多 C(10,2)=45 对。实际每环境产出几对？有没有环境产出 0 对？
3. **Chosen/Rejected 内容差异**: 如果 chosen 和 rejected 的 δ_q 方向几乎相同（差异 < 1m），DPO 只是在学 "哪种浮点格式更好看"

**验证方法**: EDA 新增 DPO 分布统计
- 效用间距直方图
- 每环境产出对数分布
- 随机采样 10 对，对比 chosen vs rejected 的 δ_q 欧氏距离

---

### P1-3: 用户位置 & 目标位置多样性未检查

**现状**: `eda_data.py` Section 3.5 只检查 UAV 位置。由于 Bug #1 的根因是 `UAVNetwork(seed=...)` 的 seed 相同导致整个拓扑相同，修复后 UAV/用户/目标位置应该都恢复多样性。但**没有显式验证**。

**验证方法**: EDA Section 3.5 扩展
```python
# 也打印 user_positions 和 target_positions 的 min/max
```

---

### P1-4: Prompt 内部数值多样性未验证（始终 4048 chars）

**现状**: EDA Section 1 显示 **所有 5000 条 prompt 恰好 4048 chars**（min==max==4048）。这是红色信号 — 在真正多样的环境中，不同 SINR 值、速率压力值的字符串长度不可能完全相同。

**可能原因**:
- 所有 prompt 被截断到 4048 chars（模板固定大小）
- BEV 网格文本始终 61 字符宽 × 13 行，确实是固定长度
- `comm_summary` 和 `sensing_summary` 中的浮点数被 format 到固定宽度
- 但这仍然可疑 — 如果有足够多的有效数字，`-999.0` 和 `-12.345` 长度相同吗？

**验证方法**: 检查 `build_communication_summary_str()` 和 `build_sensing_summary_str()` — 它们直接打印 Python 列表的 `str()` 表示，不做固定宽度格式化。在多样数据下 prompt 长度应该略有变化。如果没有变化，说明摘要列表中的数值实际上没有足够多样性。

**代码位置**: `src/data/prompt_builder.py:69-85`（摘要格式化）

---

### P1-5: SFT Response 与 DPO Chosen 可能数据泄漏

**现状**: SFT 的 response 是 10 次重启中**最优**的解。DPO 的 chosen 可能是**次优**或**任意优于 rejected**的解。但如果 `format_oracle_response()` 的输出格式完全相同（仅浮点数值不同），SFT response 可能出现在 DPO chosen 集合中。

**风险**: 如果 SFT 最优解作为 DPO chosen 出现，模型在 DPO 阶段已经在 SFT 阶段见过最优解 — DPO 的 "偏好学习" 变成了 "复习已知答案"。

**验证方法**: 检查 DPO pair 中是否存在 `chosen.utility == SFT.utility` 的对。

**代码位置**: `src/data/oracle_generator.py:139`（DPO pair 构建）、`src/data/oracle_generator.py:165-193`（`_build_dpo_pairs()`）

---

### P1-6: δ_a 关联矩阵的二元性 — Oracle 硬 vs 训练软

**现状**: Oracle 产生的是**硬**关联（Hungarian 算法 → {0,1} 矩阵）。但训练目标是让 MLLM 预测**软**关联（投影头输出 Sinkhorn 归一化的连续值）。

**风险**: 
- 损失函数用 BCE（`losses.py:73`） — 对软预测 vs 硬标签是合理的
- 但如果所有最优解的关联矩阵都相同（相同的 Hungarian 输出），DPO 的 chosen/rejected 在 δ_a 维度没有差异 → 模型学不到有意义的关联先验
- EDA Section 3.4 报告 `col_sums≈1.0` 和 `row_sums≤10` — 但没报告跨环境的 δ_a 多样性

**验证方法**: EDA 新增 δ_a 模式多样性 — 计算所有环境最优解 δ_a 的 pairwise Hamming 距离分布。

---

### P1-7: 信道增益物理合理性未检查

**现状**: 信道增益 `|h|²` 应在 `~1e-10 到 ~1e-6` 范围（对应 UMa LoS, 100m-1km 距离）。如果大量信道增益接近 0 或异常大，说明路径损耗模型有问题。

**验证方法**: EDA 新增信道增益分布直方图。

**代码位置**: `src/env/uav_channel.py:108-150`（`channel_gain()`）

---

## P2 — 值得监控但非阻塞

### P2-1: Checkpoint 断点续跑完整性

**风险**: `generate_data.py` 用行数判断进度（`_count_existing(sft_path)`）。如果某次写入中途崩溃导致最后一行不完整，重启后行数偏大 1。恢复后的数据第一行可能是半截 JSON。

**验证**: 重启后验证 SFT 和 DPO 文件最后一行可解析。

---

### P2-2: JSON 浮点精度丢失

**风险**: `json.dumps()` 默认精度可能导致小功率值（如 `1e-12W` 噪声功率级）丢失。Python 的 `json.dumps` 对 `float32` 值使用 `repr()`，通常保留 17 位有效数字 — 对 `float32`（7 位精度）足够。但 `oracle_generator.py` 中的 `delta_q.astype(np.float32)` 和 `format_oracle_response()` 使用 `json.dumps` — 需要确认往返精度。

**验证**: JSON 序列化后反序列化，计算 `max(|original - reconstructed|)`。

---

### P2-3: Control Token 初始化行为

**风险**: `gemma_isac.py:97` 用 `tokenizer.add_tokens()` 扩展词表。新 token 的 embedding 是随机初始化的。如果 control token embedding 的初始分布与其他 token 差异过大，投影头可能从噪声开始学习。

**验证**: 检查新 token 的 embedding 初始值（`model.base_model.get_input_embeddings().weight[ctrl_token_ids]`）。

---

### P2-4: LoRA Target Module 名称对 Gemma 3 是否正确

**风险**: `configs/default.yaml:59-63` 配置 `target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]`。这是 Gemma 2 的命名。Gemma 3 可能有不同的 attention 实现（如 GQA 或 MLA），模块名称可能不同。

**验证**: 加载模型后，`print(model.base_model)` 检查 attention 层的实际名称。

---

### P2-5: Sinkhorn 收敛性 — 20 次迭代是否足够

**风险**: `projection_head.py:265` 的 Sinkhorn 循环固定 20 次迭代。对于 `softmax(a/tau)` 的极端值（τ=0.5，某些 entry 远大于其他），20 次迭代可能不够 — 列和可能偏离 1.0。

**验证**: 在训练时监控 `delta_a.sum(dim=1)` 的 mean/std — 应接近 1.0。

---

### P2-6: DPO Reference Model 内存 — 32GB 够吗

**风险**: DPO 需要同时加载：
- 训练模型（4-bit LoRA）= ~8-10GB
- Reference 模型（4-bit frozen）= ~8-10GB
- 优化器状态 + 梯度 + 激活 = ~10-15GB
- **总计: 26-35GB** — 逼近或超出 32GB 上限

CLAUDE.md 提到 "DPO reference model 独立加载（不 deepcopy，会 OOM）"，说明这是已知风险点。需要实际测试 VRAM 峰值。

**代码位置**: `src/training/train_dpo.py`

---

### P2-7: 训练步数是否足够收敛

**现状**: 5000 SFT 样本，bs=1×grad_accum=16，3 epochs = 938 optimizer steps。对 LoRA rank=16 来说，约 1000 步可能偏少。

**验证**: 训练后检查 loss curve 是否 plateau。

---

### P2-8: BEV 文本网格的物理正确性

**风险**: `isac_scenario.py:283-288` 的 BEV 网格只统计 "在每个格子内的用户/目标/UAV 数量"。但如果修复前所有环境位置相同，BEV 网格也完全相同。修复后应该多样化。但 EDA 没有直接检查 BEV 文本内容。

**验证**: 随机 5 个环境，对比 BEV 文本是否不同。

---

### P2-9: SCA-FP 求解器运行时间异常值

**风险**: 某些环境可能导致求解器在 SCA 内循环中振荡（`sca_fp.py:361-421`），运行时间显著增长。如果数据生成时未监控单环境耗时，可能导致某些环境 timeout。

**验证**: `generate_data.py` 记录每环境耗时分布（batch 级别已有，单环境级别未记录）。

---

### P2-10: `compute_sensing_sinr()` 不接收 RNG — 确定性是否有问题

**风险**: `isac_scenario.py:132` 调用 `self.channel.compute_sensing_sinr()` 但**不传入 `sample_rng`**。`compute_sensing_sinr()` 是确定性的（纯公式计算），不需要 RNG。但对比 `channel_gain()` 接收 `rng` 参数用于小尺度衰落采样 — 这里的不一致可能导致未来维护时的困惑。

**不需要修复**，但值得注意：感知 SINR 计算是确定性的，不包含小尺度衰落。这在论文中可能是合理的简化（感知信道的相干处理时间更长）。

---

## 建议的验证顺序

### 在服务器重新生成数据之前（现在）

| 序号 | 验证项 | 优先级 | 耗时 | 方法 |
|------|--------|--------|------|------|
| 1 | P0-1: 真实 Tokenizer 计数 | 🔴 | 2 min | 调用一次 Gemma3 tokenizer |
| 2 | P0-3: Gemma3 权重可下载 | 🔴 | 5 min | `huggingface-cli whoami` |
| 3 | P2-4: LoRA 模块名称 | 🟡 | 1 min | 检查模型结构 |

### Smoke 20 环境生成后

| 序号 | 验证项 | 优先级 | 耗时 | 方法 |
|------|--------|--------|------|------|
| 4 | P1-3: 用户/目标位置多样性 | 🟠 | 0 min | EDA 扩展 |
| 5 | P1-4: Prompt 数值多样性 | 🟠 | 0 min | EDA 扩展 |
| 6 | P1-5: SFT-DPO 泄漏检查 | 🟠 | 1 min | 小脚本 |
| 7 | P1-7: 信道增益分布 | 🟠 | 1 min | EDA 新增 |

### 全量 5000 环境后、训练前

| 序号 | 验证项 | 优先级 | 耗时 | 方法 |
|------|--------|--------|------|------|
| 8 | P1-1: 求解器收敛统计 | 🟠 | 2 min | EDA 新增 Section 4 |
| 9 | P1-2: DPO 对质量分布 | 🟠 | 5 min | EDA 新增 |
| 10 | P1-6: δ_a 模式多样性 | 🟠 | 2 min | EDA 新增 |
| 11 | P0-2: Unsloth 烟雾测试 | 🔴 | 10 min | 加载模型 + 一条推理 |

### 训练开始后

| 序号 | 验证项 | 优先级 | 耗时 | 方法 |
|------|--------|--------|------|------|
| 12 | P2-6: VRAM 峰值监控 | 🟡 | 5 min | `nvidia-smi` 或 `torch.cuda` |
| 13 | P2-5: Sinkhorn 收敛监控 | 🟡 | 训练中 | Tensorboard |
| 14 | P2-7: Loss curve 平台 | 🟡 | 训练中 | Tensorboard |

---

## 总结

```
P0 (阻塞级):  3 项 — 真实 tokenizer 计数、Unsloth 兼容性、Gemma3 权重下载
P1 (质量级):  7 项 — 求解器行为、DPO 质量、提示词多样性、数据泄漏、关联多样性、信道合理性
P2 (监控级): 10 项 — 断点续跑、精度、初始化、模块名、Sinkhorn、内存、步数等
─────────────────────
总计:        20 项未验证维度
```

**最重要的 3 件事**（与之前两个 P0 Bug 同级重要的下一批）：
1. **用真实 tokenizer 确认 response < 1024 tokens** — 启发式估算可能报假绿灯
2. **在 RTX 5090 上跑一次 Unsloth 加载** — 如果 Unsloth 不支持 Gemma3 或 Blackwell，整个训练计划需要重新设计
3. **确认 HuggingFace 可下载 Gemma3 权重** — gated model，没有权限就全卡住

---

> **相关文档**:
> - [11_pre_training_data_eda_postmortem.md](11_pre_training_data_eda_postmortem.md) — 双 P0 Bug 事后分析
> - [10_physical_constraint_bug_postmortem.md](10_physical_constraint_bug_postmortem.md) — 物理约束穿透 Bug
> - [09_handoff_document.md](09_handoff_document.md) — 完整项目交接
