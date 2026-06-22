# Constraint-Aware MLLM Adaptation for UAV-Enabled ISAC

复现论文: *Constraint-Aware MLLM Adaptation for UAV-Enabled ISAC in Low-Altitude IoT Networks*

目标硬件: **RTX 5090 32GB @ AutoDL**

---

## 🗂️ 项目结构

```
UAV/
├── configs/
│   └── default.yaml              # 主配置 (仿真/模型/训练参数)
├── src/
│   ├── env/                      # 仿真环境
│   │   ├── uav_network.py        #   UAV/用户/目标拓扑
│   │   ├── uav_channel.py        #   信道模型 (LoS/NLoS, SINR, CRB)
│   │   └── isac_scenario.py      #   ISAC场景生成器
│   ├── solver/
│   │   └── sca_fp.py             # SCA-FP 优化器 (下游求解器)
│   ├── data/
│   │   ├── prompt_builder.py     #   Prompt 构造器 (Π(t) 生成)
│   │   └── oracle_generator.py   #   Best-of-N 数据生成 (Algorithm 1)
│   ├── model/
│   │   ├── gemma_isac.py         #   Gemma3 + LoRA + Control Token
│   │   ├── projection_head.py    #   可微约束投影头 (Section 5)
│   │   └── losses.py             #   损失函数 (SFT/DPO/Control/Sep)
│   ├── training/
│   │   ├── train_sft.py          #   Stage I: SFT-LoRA 训练
│   │   └── train_dpo.py          #   Stage II: DPO 训练
│   └── eval/
│       └── evaluate.py           #   评估脚本 (9 baselines)
├── scripts/
│   └── autodl_setup.sh           # AutoDL 一键环境搭建
├── requirements.txt
└── README.md
```

---

## 🚀 AutoDL 快速开始 (四步)

### Step 0: 上传项目到 AutoDL

在本地 Windows 上:
```bash
# 打包
cd h:\Projects
tar -czf uav.tar.gz UAV/

# 上传到 AutoDL (通过 JupyterLab 上传 或 scp)
scp uav.tar.gz root@<your-autodl-ip>:/root/autodl-tmp/
```

在 AutoDL Terminal:
```bash
cd /root/autodl-tmp/
tar -xzf uav.tar.gz
cd UAV/
```

### Step 1: 环境配置

```bash
# 一键安装
bash scripts/autodl_setup.sh

# 或手动
conda activate uavmllm
```

### Step 2: 生成训练数据 (耗时最长!)

```bash
conda activate uavmllm

python -c "
import yaml
from src.env import ISACScenarioGenerator
from src.solver import SCAFPOptimizer, SCAFPConfig
from src.data import OracleDataGenerator

# 加载配置
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)

sc = cfg['simulation']

# 初始化组件
scenario_gen = ISACScenarioGenerator(
    num_uavs=sc['num_uavs'],
    num_users=sc['num_users'],
    num_targets=sc['num_targets'],
    area_size=tuple(sc['area_size']),
    carrier_freq_ghz=sc['carrier_freq_ghz'],
    bandwidth_mhz=sc['bandwidth_mhz'],
    num_antennas=sc['num_antennas_tx'],
    p_max_dbm=sc['p_max_dbm'],
    seed=42,
)

solver_cfg = SCAFPConfig(
    max_outer_iters=30, max_inner_iters=50,
    tol=1e-4, lambda_sensing=0.5,
    lambda_idle_penalty=5.0,
    verbose=False,
)

solver = SCAFPOptimizer(
    config=solver_cfg,
    M=sc['num_uavs'], K=sc['num_users'],
    T=sc['num_targets'], N_t=sc['num_antennas_tx'],
    area_size=tuple(sc['area_size']),
    altitude_range=(sc['altitude_min_m'], sc['altitude_max_m']),
    p_max=10**((sc['p_max_dbm']-30)/10),
    load_cap=sc['load_cap_per_uav'],
)

# 生成数据 (S=5000, N=10 → 50,000 SCA-FP 求解)
gen = OracleDataGenerator(scenario_gen, solver, cfg['data'])
sft_data, dpo_data = gen.generate_all()

print(f'SFT samples: {len(sft_data)}')
print(f'DPO pairs: {len(dpo_data)}')
"
```

**预计时间**: 几小时到半天 (取决于 SCA-FP 实现效率)

### Step 3: Stage I — SFT-LoRA 训练

```bash
conda activate uavmllm

# 单卡 5090, 4-bit QLoRA, bs=1, grad_accum=16
python src/training/train_sft.py \
    --config configs/default.yaml \
    --data_dir ./data/cache/sft_dataset.jsonl
```

**预计显存**: 25-28 GB / 32 GB ✅\
**预计时间**: 3-8 小时

### Step 4: Stage II — DPO 训练

```bash
conda activate uavmllm

python src/training/train_dpo.py \
    --config configs/default.yaml \
    --stage1_ckpt ./outputs/stage1_sft_final \
    --data_dir ./data/cache/dpo_dataset.jsonl
```

**预计显存**: 22-25 GB / 32 GB ✅ (reference model 也 4-bit)\
**预计时间**: 5-10 小时

---

## 📊 评估

```bash
python src/eval/evaluate.py \
    --config configs/default.yaml \
    --model ./outputs/stage2_dpo_final \
    --output ./outputs/eval_results.json
```

评估输出包含 9 基线对比所需的全部指标:
- Sum rate (Mbps)
- Mean sensing SINR (dB)
- Joint satisfaction rate
- SCA-FP convergence iterations
- Per-slot inference latency (ms)

---

## ⚙️ RTX 5090 最佳配置参考

```
| 参数                        | 推荐值          | 说明                            |
|----------------------------|----------------|--------------------------------|
| 精度                        | BF16 + 4-bit   | QLoRA nf4 quant                |
| LoRA rank                  | 16             | 论文值, 5090 完全够               |
| LoRA alpha                 | 32             | 论文值                           |
| Per-device batch size      | 1              | 5090 单卡最大                    |
| Gradient accumulation      | 16             | 有效 batch ≈ 16                  |
| Max sequence length        | 4096           | Gemma3 原生上下文                  |
| Flash Attention 2          | 开启            | 5090 完美支持                    |
| Gradient checkpointing     | 开启            | 省显存                           |

预估显存分配:
  模型 (4-bit):      8-10 GB
  LoRA 参数:         ~1 GB
  优化器状态:         3-5 GB
  KV Cache + 激活:   10-12 GB
  参考模型 (DPO):    +8 GB
  ─────────────────────────
  SFT 峰值:          ~25-28 GB
  DPO 峰值:          ~28-31 GB  (刚好在 32GB 边缘)
```

---

## 📝 Gemma 3 权重获取

Gemma 3 12B 是 Google 的 gated model:

1. 访问 [huggingface.co/google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)
2. 点击 "Access repository" 申请授权 (通常几分钟通过)
3. 在 AutoDL 上登录 HuggingFace:
```bash
huggingface-cli login
# 输入你的 HF token
```

文本版验证阶段可以用较小的 Gemma 2 2B/9B 先跑通管线:
```yaml
# configs/default.yaml
model:
  backbone: "google/gemma-2-9b-it"  # 先用 9B 验证
```

---

## 🔧 开发建议

### 第 1 周: 跑通管线
1. 用 Gemma 2 9B (文本版) 验证 SFT 训练能收敛
2. 确认 SCA-FP 数据生成管线正确
3. 验证 Projection Head 约束投影正确

### 第 2 周: 论文复现
1. 用 Gemma 3 12B 训练完整 SFT
2. DPO 微调
3. 与 B6 (frozen prompting) 对比

### 第 3 周: 实验完善
1. 消融实验 (B7-B9)
2. 鲁棒性测试 (用户密度/目标移动/CSI 不确定性)
3. 推理延迟分析

---

## 引用

论文: Constraint-Aware MLLM Adaptation for UAV-Enabled ISAC in Low-Altitude IoT Networks
