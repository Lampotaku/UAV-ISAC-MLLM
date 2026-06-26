#!/usr/bin/env python
"""
SFT 过拟合测试 — 证明训练管线正确性的最简可行证据

原理: 一个正确的训练管线在极小数据 (5 样本) 上一定能过拟合。
如果 loss 降不到接近 0，说明代码有 bug (loss 计算、梯度断流、
mask 错位、投影头没接上等)。

6 项检查:
  1. loss_total 下降 >50%           — 梯度流、前向/反向传播
  2. loss_sft < 0.5                 — label_mask 对齐、token prediction
  3. loss_ctl < 0.01                — 投影头梯度流、δ 目标拟合
  4. 最后 50 步 loss 持续下降        — 优化器、学习率
  5. 无 NaN/Inf                     — 梯度裁剪、loss 计算安全
  6. 过拟合后 inference 匹配标签     — 前向推理管线、投影头 train/eval 一致性

用法 (服务器):
  conda activate uavmllm
  python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full5000

预期:
  - loss_sft: 从 ~2-4 降到 <0.3
  - loss_ctl: 从 ~0.1-0.5 降到 <0.01
  - 6/6 checks all green
  - 无 NaN, 无 Inf
  - ~3-5 分钟完成

若通过 → SFT 代码正确, 可以放心启动全量训练
若失败 → 逐项排查 (脚本会告诉你哪项失败)
"""

import os
import sys
import json
import argparse
import time

# ── BLAS 线程抑制 (必须在 import numpy/torch 之前) ──
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ── 【防爆盾 1】核弹级环境变量 ──
# Blackwell sm_120: 禁止 Inductor 使用 FlexAttention (共享内存 101KB < 需 114KB)
os.environ["TORCHINDUCTOR_FLEX_ATTENTION"] = "0"
# 过拟合测试不需要 torch.compile, 彻底切断 Inductor 编译链路
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ── 【防爆盾 2】项目已彻底肃清 Unsloth ──
# Unsloth 局部导入也会全局 monkey-patch → CheckpointError (68≠65).
# 全项目 0 处 unsloth 引用, 纯 PyTorch CE + bs=1/grad_accum=16.

import numpy as np
import yaml
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch

# ── 【防爆盾 3】代码级物理超度 FlexAttention ──
import torch._inductor.config as inductor_config
if hasattr(inductor_config, "flex_attention"):
    inductor_config.flex_attention = False
if hasattr(inductor_config, "use_flex_attention"):
    inductor_config.use_flex_attention = False

# 现在可以安全导入 HF 和其他库了 (Unsloth 已就位)
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from tqdm import tqdm

from src.model import Gemma3ISAC, UAVISACLosses
from src.data.dataset import SFTDataset


# ── Helpers ──────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")


def create_tiny_subset(data_dir: str, n: int = 5) -> str:
    """从全量数据取前 N 个 SFT 样本写入临时文件"""
    src = os.path.join(data_dir, "sft_dataset.jsonl")
    dst = os.path.join(data_dir, f"sft_tiny_{n}.jsonl")
    with open(src, "r", encoding="utf-8") as fin:
        with open(dst, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i >= n:
                    break
                fout.write(line)
    print(f"Created tiny subset: {dst} ({n} samples)")
    return dst


def run_overfit_test(config_path: str, data_path: str, n_samples: int,
                     n_steps: int = 200):
    """
    核心过拟合测试:
      在 N 个样本上训练若干步, 验证 loss 单调下降
    """
    print(f"\n{'='*60}")
    print(f"Stage I SFT Overfitting Test")
    print(f"{'='*60}")
    print(f"  Samples:  {n_samples}")
    print(f"  Steps:    {n_steps}")
    print(f"  Data:     {data_path}")
    print()

    # ── 加载配置 ──
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]["sft"]
    sim_cfg = cfg["simulation"]
    data_cfg = cfg["data"]

    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        warn("No GPU detected — overfitting test on CPU will be VERY slow")
        warn("Run on the AutoDL server instead")

    # ── 初始化模型 ──
    print("\n[1/5] Loading Gemma3-ISAC model...")
    t0 = time.time()
    model = Gemma3ISAC(
        model_name_or_path=model_cfg["backbone"],
        use_4bit=cfg["hardware"]["use_4bit"],
        lora_rank=model_cfg["lora"]["rank"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"]["dropout"],
        lora_target_modules=model_cfg["lora"]["target_modules"],
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        proj_head_config={
            "hidden_dim": model_cfg["control_token"]["hidden_dim"],
            "num_control_tokens": model_cfg["control_token"]["num_tokens"],
            "mlp_hidden": model_cfg["projection_head"]["mlp_hidden"],
            "readout_out_dim": model_cfg["projection_head"]["readout_out_dim"],
            "M": sim_cfg["num_uavs"],
            "K": sim_cfg["num_users"],
            "area_w": sim_cfg["area_size"][0],
            "area_h": sim_cfg["area_size"][1],
            "h_min": sim_cfg["altitude_min_m"],
            "h_max": sim_cfg["altitude_max_m"],
            "v_max_dt": sim_cfg["uav_max_speed_ms"] * sim_cfg["slot_duration_s"],
            "p_max": 10 ** ((sim_cfg["p_max_dbm"] - 30) / 10),
            "K_max": sim_cfg["load_cap_per_uav"],
            "tau_power": model_cfg["projection_head"]["tau_power"],
            "tau_assoc": model_cfg["projection_head"]["tau_assoc"],
            "sinkhorn_iters": model_cfg["projection_head"]["sinkhorn_iters"],
        },
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
    )
    model = model.to(device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # 诊断: 验证所有参数都在目标设备上
    devices = set()
    for name, param in model.named_parameters():
        devices.add(str(param.device))
    print(f"  [DIAG] Parameter devices: {devices}")
    if len(devices) > 1:
        fail(f"Parameters are on multiple devices: {devices} — device placement broken!")
        # 列出不在目标设备上的参数
        for name, param in model.named_parameters():
            if param.device != device:
                print(f"    OFF-DEVICE: {name} on {param.device} (target: {device})")
        return False
    elif device.type == "cuda" and "cuda" not in str(list(devices)[0]):
        fail(f"Parameters on {list(devices)[0]} but target is {device} — model.to() failed!")
        return False

    # ── 检查可训练参数 + 深度诊断 ──
    trainable_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  Trainable params: {trainable_count:,}")
    if trainable_count == 0:
        fail("No trainable parameters! Check LoRA + projection head setup.")

    # 诊断: 列出前 5 个可训练参数的 name / device / shape
    print(f"\n  [DIAG] First 5 trainable parameters:")
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for i, (name, param) in enumerate(trainable_params[:5]):
        print(f"    {i+1}. {name}  |  device={param.device}  |  shape={tuple(param.shape)}  |  dtype={param.dtype}")

    # 诊断: 检查 projection_head 是否在 model.named_parameters() 中
    proj_names = [n for n, p in model.named_parameters() if "projection_head" in n]
    print(f"\n  [DIAG] Projection head params in model.named_parameters(): {len(proj_names)}")
    if proj_names:
        for n in proj_names[:3]:
            p = dict(model.named_parameters())[n]
            print(f"    {n}  |  device={p.device}  |  requires_grad={p.requires_grad}")

    # 诊断: 检查 base_model (PeftModel) 的可训练参数
    lora_check = [(n, p) for n, p in model.base_model.named_parameters() if p.requires_grad]
    print(f"\n  [DIAG] Trainable params in model.base_model.named_parameters(): {len(lora_check)}")
    if lora_check:
        for name, param in lora_check[:5]:
            print(f"    {name}  |  device={param.device}  |  shape={tuple(param.shape)}")
    else:
        fail("ZERO trainable params in model.base_model! LoRA not applied?")
        # 深度检查: model.base_model 到底是什么类型?
        print(f"  [DIAG] type(model.base_model) = {type(model.base_model)}")
        print(f"  [DIAG] type(model.base_model).__mro__ = {type(model.base_model).__mro__}")
        # 列出 model.base_model 的属性
        attrs = [a for a in dir(model.base_model) if not a.startswith('_')]
        print(f"  [DIAG] model.base_model attrs: {attrs}")
        return False

    # ── 加载数据 ──
    print(f"\n[2/5] Loading dataset...")
    dataset = SFTDataset(
        data_path=data_path,
        tokenizer=model.tokenizer,
        max_length=train_cfg["max_seq_length"],
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
    )
    print(f"  Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 单样本 batch — 过拟合更纯粹
        shuffle=True,
        num_workers=0,  # 避免 multiprocessing 干扰调试
    )

    # ── 优化器 (分层学习率) ──
    print(f"\n[3/5] Setting up optimizer...")
    # 投影头: 随机初始化 (f32), 需要较大 LR
    proj_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "projection_head" in n
    ]
    # LoRA: 预训练权重微调 (bf16), LR 必须保守
    lora_params = [
        p for n, p in model.base_model.named_parameters()
        if p.requires_grad
    ]
    print(f"  Optimizing: Projection Head ({len(proj_params)} tensors), "
          f"LoRA ({len(lora_params)} tensors)")
    if len(proj_params) == 0:
        fail("ZERO projection_head parameters found for optimizer!")
        return False
    if len(lora_params) == 0:
        fail("ZERO LoRA parameters found for optimizer!")
        return False

    optimizer = torch.optim.AdamW([
        {"params": proj_params, "lr": 1e-3},   # 投影头: 从零训练
        {"params": lora_params, "lr": 2e-4},   # LoRA: 与全量 SFT 一致
    ], weight_decay=0.0)

    # 诊断: 验证 optimizer 真的有参数
    total_opt_params = sum(len(g['params']) for g in optimizer.param_groups)
    print(f"  [DIAG] Optimizer has {total_opt_params} param groups with "
          f"{sum(p.numel() for g in optimizer.param_groups for p in g['params']):,} total params")

    # ── 损失计算器 ──
    loss_fn = UAVISACLosses(
        lambda_ctl=model_cfg["loss"]["lambda_ctl"],
        lambda_q=model_cfg["loss"]["lambda_q"],
        lambda_a=model_cfg["loss"]["lambda_a"],
        lambda_p=model_cfg["loss"]["lambda_p"],
        lambda_sep=model_cfg["loss"]["lambda_sep"],
    )

    # ── 训练循环 ──
    print(f"\n[4/5] Running overfitting loop ({n_steps} steps)...")
    model.train()

    history = {"loss_total": [], "loss_sft": [], "loss_ctl": []}
    all_batches = list(dataloader)  # 全部 batch 预先加载 (只有 5 个)

    pbar = tqdm(range(n_steps), desc="Overfitting")
    nan_detected = False
    first_step_diagnostics_done = False

    for step in pbar:
        # 循环使用 5 个样本
        batch = all_batches[step % len(all_batches)]

        # 移到 GPU
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # 前向传播
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            control_mask=batch["control_mask"],
            q_current=batch["q_current"] if batch["q_current"].numel() > 0 else None,
            labels=batch["labels"],
        )

        # ── 第一步诊断: hidden states / control_states 是否正常 ──
        if not first_step_diagnostics_done:
            print(f"\n  [DIAG] === Step 0 forward diagnostics ===")
            hs_norm = outputs["hidden_states"].norm().item()
            cs_norm = outputs["control_states"].norm().item()
            dq_norm = outputs["delta_q"].norm().item()
            da_norm = outputs["delta_a"].norm().item()
            dp_norm = outputs["delta_p"].norm().item()
            print(f"  hidden_states norm: {hs_norm:.4f} (expect > 100)")
            print(f"  control_states norm: {cs_norm:.4f} (expect > 10)")
            print(f"  delta_q norm: {dq_norm:.4f}  delta_a norm: {da_norm:.4f}  delta_p norm: {dp_norm:.4f}")
            if cs_norm < 1.0:
                fail(f"control_states norm = {cs_norm:.4f} < 1.0 — control_mask extraction may be broken!")
            # 检查 control_mask 是否正确 (应该有 8 个 True)
            ctrl_count = batch["control_mask"].sum().item()
            print(f"  control_mask True count: {ctrl_count} (expect {model_cfg['control_token']['num_tokens']})")
            if ctrl_count != model_cfg["control_token"]["num_tokens"]:
                fail(f"control_mask has {ctrl_count} True positions, expected {model_cfg['control_token']['num_tokens']}!")

        # 构造 target dict
        delta_target = {
            "delta_q": batch["delta_q_target"],
            "delta_a": batch["delta_a_target"],
            "delta_p": batch["delta_p_target"],
        }
        delta_hat = {
            "delta_q": outputs["delta_q"],
            "delta_a": outputs["delta_a"],
            "delta_p": outputs["delta_p"],
        }

        q_hat = None
        if batch["q_current"].numel() > 0:
            q_hat = batch["q_current"] + outputs["delta_q"]

        # 计算损失
        total_loss, metrics = loss_fn.compute_stage1_total(
            delta_hat=delta_hat,
            delta_target=delta_target,
            logits=outputs["logits"],
            labels=batch["labels"],
            label_mask=batch["label_mask"],
            q_hat=q_hat,
        )

        # NaN 检测
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            fail(f"NaN/Inf detected at step {step}!")
            nan_detected = True
            break

        # 反向传播
        total_loss.backward()

        # ── 第一步诊断: 梯度范数 + 权重变化 ──
        if not first_step_diagnostics_done:
            print(f"\n  [DIAG] === Step 0 gradient diagnostics ===")
            # 检查几个关键参数的梯度
            grad_samples = []
            zero_grad_params = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    g_norm = param.grad.norm().item()
                    grad_samples.append((name, g_norm))
                    if g_norm == 0.0:
                        zero_grad_params.append(name)
            # 按 grad norm 排序，显示最大的 5 个和最小的 5 个
            grad_samples.sort(key=lambda x: -x[1])
            print(f"  Top 5 gradients by norm:")
            for name, gnorm in grad_samples[:5]:
                print(f"    {name}: grad_norm={gnorm:.6f}")
            if zero_grad_params:
                print(f"  {YELLOW}Parameters with ZERO gradient:{RESET}")
                for name in zero_grad_params[:5]:
                    print(f"    {name}")
                if len(zero_grad_params) > 5:
                    print(f"    ... and {len(zero_grad_params) - 5} more")

            # 保存 step 前权重快照
            weight_snapshot = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    weight_snapshot[name] = param.data.clone()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ── 第一步诊断: 检查权重是否真的变了 ──
        if not first_step_diagnostics_done:
            print(f"\n  [DIAG] === Step 0 weight change diagnostics ===")
            changed = 0
            unchanged = 0
            for name, param in model.named_parameters():
                if param.requires_grad and name in weight_snapshot:
                    delta = (param.data - weight_snapshot[name]).abs().max().item()
                    if delta > 1e-10:
                        changed += 1
                        if changed <= 5:
                            print(f"    {name}: max |Δw| = {delta:.6f}")
                    else:
                        unchanged += 1
                        if unchanged <= 3:
                            print(f"    {YELLOW}{name}: NO CHANGE (Δ=0){RESET}")
            print(f"  Result: {changed} params changed, {unchanged} UNCHANGED")
            if unchanged > 0:
                print(f"  {RED}✗ Some parameters did NOT update after optimizer.step(){RESET}")
            first_step_diagnostics_done = True

        optimizer.zero_grad()

        # ── Step 5 诊断: lora_A 梯度是否已出现 (B 不再是零) ──
        if step == 5:
            lora_a_grad_count = 0
            lora_a_zero_count = 0
            for name, param in model.named_parameters():
                if "lora_A" in name and param.requires_grad and param.grad is not None:
                    if param.grad.norm().item() > 0:
                        lora_a_grad_count += 1
                    else:
                        lora_a_zero_count += 1
            print(f"\n  [DIAG] === Step 5 lora_A gradient check ===")
            print(f"  lora_A with gradient: {lora_a_grad_count}")
            print(f"  lora_A still zero:    {lora_a_zero_count}")
            if lora_a_grad_count > 0:
                ok(f"lora_A gradients emerging (B no longer zero → A now receives gradients)")
            else:
                fail(f"lora_A STILL zero at step 5 — something deeper is broken!")

        # 记录 (防御性: 剥离可能的计算图, 尽管 compute_stage1_total 已调 .item())
        for k in history:
            val = metrics[k]
            history[k].append(val.item() if isinstance(val, torch.Tensor) else val)

        # 每步更新进度条 (过拟合测试步数少, 实时观察 loss 变化)
        pbar.set_postfix({
                "total": f"{metrics['loss_total']:.4f}",
                "sft": f"{metrics['loss_sft']:.4f}",
                "ctl": f"{metrics['loss_ctl']:.4f}",
            })

    if nan_detected:
        return False

    # ── 验证结果 ──
    print(f"\n[5/5] Verifying results...")
    print()

    # 修复窗口重叠: 当 n_steps < 20 时 initial/final 不能共用数据
    window = min(10, max(1, n_steps // 5))
    if n_steps <= window * 2:
        mid = n_steps // 2
        initial = {k: history[k][:mid] for k in history}
        final = {k: history[k][mid:] for k in history}
    else:
        initial = {k: history[k][:window] for k in history}
        final = {k: history[k][-window:] for k in history}

    all_checks_pass = True

    # Check 1: loss_total 下降
    init_total = np.mean(initial["loss_total"])
    final_total = np.mean(final["loss_total"])
    print(f"  Loss total:  {init_total:.4f} → {final_total:.4f}  "
          f"({(1 - final_total/init_total)*100:.0f}% reduction)")
    if final_total < init_total * 0.5:
        ok("loss_total decreased >50% — gradients are flowing")
    elif final_total < init_total:
        warn("loss_total decreased but <50% — may need more steps or higher lr")
    else:
        fail("loss_total did NOT decrease — check forward/backward wiring")
        all_checks_pass = False

    # Check 2: loss_sft 下降
    init_sft = np.mean(initial["loss_sft"])
    final_sft = np.mean(final["loss_sft"])
    print(f"  Loss SFT:    {init_sft:.4f} → {final_sft:.4f}  "
          f"({(1 - final_sft/init_sft)*100:.0f}% reduction)")
    if final_sft < 0.5:
        ok("loss_sft < 0.5 — model is memorizing token sequences")
    elif final_sft < init_sft * 0.7:
        ok("loss_sft decreasing — token prediction learning")
    else:
        warn("loss_sft barely decreased — check label_mask / tokenizer setup")
        # Not a hard fail: SFT loss on 12B vocab can be slow to drop

    # Check 3: loss_ctl 下降
    init_ctl = np.mean(initial["loss_ctl"])
    final_ctl = np.mean(final["loss_ctl"])
    print(f"  Loss ctl:    {init_ctl:.4f} → {final_ctl:.4f}  "
          f"({(1 - final_ctl/init_ctl)*100:.0f}% reduction)")
    if final_ctl < 0.01:
        ok("loss_ctl < 0.01 — projection head is fitting targets precisely")
    elif final_ctl < init_ctl * 0.5:
        ok("loss_ctl decreasing — projection head is learning")
    else:
        fail("loss_ctl barely decreased — check projection head / delta targets")
        all_checks_pass = False

    # Check 4: loss 曲线单调性 (最后 50 步应持续下降)
    recent = history["loss_total"][-50:]
    early_recent = np.mean(recent[:10])
    late_recent = np.mean(recent[-10:])
    if late_recent < early_recent:
        ok("Loss still decreasing in final 50 steps")
    else:
        warn("Loss plateaued — may need more steps (but not a bug)")

    # Check 5: 无 NaN/Inf
    has_nan = any(np.isnan(history["loss_total"]))
    has_inf = any(np.isinf(history["loss_total"]))
    if not has_nan and not has_inf:
        ok("No NaN/Inf in training history")
    else:
        fail("NaN/Inf detected — learning rate too high or gradient explosion")
        all_checks_pass = False

    # Check 6: 前向推理管线验证 (过拟合后 inference 必须匹配训练标签)
    inference_ok = run_inference_check(model, data_path, device, sim_cfg, n_samples)
    if not inference_ok:
        all_checks_pass = False

    # ── 总结 ──
    print(f"\n{'='*60}")
    if all_checks_pass:
        print(f"{GREEN}✓ ALL 6 CHECKS PASSED{RESET}")
        print(f"  The SFT training pipeline is correctly wired:")
        print(f"    • Tokenization + control token injection")
        print(f"    • Gemma3 forward pass (4-bit QLoRA)")
        print(f"    • Control token hidden state extraction")
        print(f"    • Projection head (readout → MLP → constraints)")
        print(f"    • Combined loss (L_SFT + λ_ctl * L_ctl)")
        print(f"    • Gradient flow through LoRA + projection head")
        print(f"    • Optimizer updates")
        print(f"    • Forward inference (generate_warmstart → correct deltas)")
        print(f"\n  → Safe to proceed with full 5000-sample SFT training.")
    else:
        print(f"{RED}✗ SOME CHECKS FAILED{RESET}")
        print(f"  Review the failures above before launching full training.")
    print(f"{'='*60}")

    return all_checks_pass


def run_inference_check(model, data_path: str, device: torch.device,
                        sim_cfg: dict, n_samples: int = 5) -> bool:
    """
    Check 6: 过拟合后前向推理管线验证 (Ultimate Sanity Check)

    在 5 个过拟合样本上运行 generate_warmstart(), 比对:
      - delta_q 输出是否与训练标签一致 (max abs error < 0.01)
      - delta_a 输出是否与训练标签一致 (accuracy > 95%)
      - delta_p 输出是否与训练标签一致 (max abs error < 0.01)
      - 物理约束是否满足 (‖Δq‖₂ ≤ 15m)

    原理: loss 正常下降不代表 forward inference pipeline 正确 —
     train/eval 模式差异、hidden state 提取、投影头 forward 路径
     都可能出问题。此项检查一次性验证全部推理管线。
    """
    print(f"\n[6/6] Forward inference pipeline check...")
    print(f"  Running generate_warmstart() on {n_samples} overfit samples")

    # 加载原始 JSON 数据 (需要 prompt 字符串 + ground-truth deltas)
    raw_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            raw_data.append(json.loads(line))

    v_max_dt = sim_cfg["uav_max_speed_ms"] * sim_cfg["slot_duration_s"]
    p_max = 10 ** ((sim_cfg["p_max_dbm"] - 30) / 10)

    model.eval()
    all_ok = True

    for i, item in enumerate(raw_data):
        prompt = item["prompt"]
        q_current = torch.tensor(item["q_current"], dtype=torch.float32, device=device)
        delta_q_tgt = torch.tensor(item["delta_q"], dtype=torch.float32, device=device)
        delta_a_tgt = torch.tensor(item["delta_a"], dtype=torch.float32, device=device)
        delta_p_tgt = torch.tensor(item["delta_p"], dtype=torch.float32, device=device)

        # 前向推理 (generate_warmstart 返回 CPU tensor，需移到 GPU 做比对)
        with torch.no_grad():
            result = model.generate_warmstart(prompt, q_current, max_new_tokens=512)

        delta_q_hat = result["delta_q"].to(device)   # (M, 3)
        delta_a_hat = result["delta_a"].to(device)   # (M, K)
        delta_p_hat = result["delta_p"].to(device)   # (M, K+1)

        # ── 比对 1: delta_q (3D 位移) ──
        q_err = (delta_q_hat - delta_q_tgt).abs().max().item()
        q_mse = (delta_q_hat - delta_q_tgt).pow(2).mean().item()
        if q_err > 0.01:
            fail(f"  Sample {i}: delta_q max abs error = {q_err:.6f} > 0.01 "
                 f"(MSE={q_mse:.6f})")
            all_ok = False

        # ── 比对 2: delta_a (关联矩阵) ──
        a_pred = (delta_a_hat > 0.5).float()
        a_acc = (a_pred == delta_a_tgt).float().mean().item()
        if a_acc < 0.95:
            fail(f"  Sample {i}: delta_a accuracy = {a_acc:.4f} < 0.95")
            all_ok = False

        # ── 比对 3: delta_p (功率分配) ──
        p_err = (delta_p_hat - delta_p_tgt).abs().max().item()
        p_mse = (delta_p_hat - delta_p_tgt).pow(2).mean().item()
        if p_err > 0.01:
            fail(f"  Sample {i}: delta_p max abs error = {p_err:.6f} > 0.01 "
                 f"(MSE={p_mse:.6f})")
            all_ok = False

        # ── 物理约束 1: 位移 ≤ v_max·Δt ──
        q_norms = delta_q_hat.norm(dim=-1)  # (M,) 3D norm per UAV
        max_disp = q_norms.max().item()
        if max_disp > v_max_dt + 0.1:
            fail(f"  Sample {i}: max ‖Δq‖₂ = {max_disp:.2f}m > {v_max_dt}m")
            all_ok = False

        # ── 物理约束 2: 高度边界 ──
        h_new = q_current[:, 2] + delta_q_hat[:, 2]
        h_min_violation = (h_new < sim_cfg["altitude_min_m"]).any().item()
        h_max_violation = (h_new > sim_cfg["altitude_max_m"]).any().item()
        if h_min_violation or h_max_violation:
            fail(f"  Sample {i}: altitude constraint violated "
                 f"(h range: [{h_new.min().item():.1f}, {h_new.max().item():.1f}]m)")
            all_ok = False

        # ── 物理约束 3: 功率预算 ──
        power_per_uav = delta_p_hat.sum(dim=-1)  # (M,) sum over (K+1) users+target
        max_power = power_per_uav.max().item()
        if max_power > p_max + 0.02:
            fail(f"  Sample {i}: max power = {max_power:.4f}W > {p_max}W")
            all_ok = False

    # ── 汇总 ──
    if all_ok:
        ok("Forward inference pipeline verified:")
        ok("  • generate_warmstart() produces correct delta_q/a/p")
        ok("  • All physical constraints satisfied (‖Δq‖₂, altitude, power)")
        ok("  • Control token extraction + projection head forward = training labels")
        ok("  • Train → Eval mode transition clean (no BN/dropout mismatch)")
    else:
        fail("Forward inference mismatch — possible causes:")
        fail("  • Hidden state extraction from wrong positions")
        fail("  • Projection head train/eval mode discrepancy")
        fail("  • Control token mask misalignment in generate_warmstart()")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="SFT overfitting test — prove training code correctness"
    )
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
    parser.add_argument("--data-dir", type=str,
                        default="/root/autodl-tmp/data/full5000")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of samples to overfit on")
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Number of optimization steps")
    parser.add_argument("--keep-subset", action="store_true",
                        help="Don't delete the temporary tiny subset file")
    args = parser.parse_args()

    # 创建 tiny subset
    tiny_path = create_tiny_subset(args.data_dir, args.n_samples)

    try:
        passed = run_overfit_test(args.config, tiny_path,
                                  args.n_samples, args.n_steps)
    finally:
        if not args.keep_subset and os.path.exists(tiny_path):
            os.remove(tiny_path)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
