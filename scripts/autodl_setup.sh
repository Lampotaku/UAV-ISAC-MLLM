#!/bin/bash
# ==========================================================================
# AutoDL RTX 5090 (Blackwell) 一键环境搭建脚本
# 在 AutoDL 的 JupyterLab Terminal 中运行:
#   bash scripts/autodl_setup.sh
#
# 要求:
#   - AutoDL 基础镜像: Miniconda + CUDA 12.8 (或更高)
#   - RTX 5090 32GB (Blackwell sm_120)
#   - NVIDIA Driver ≥ 570
# ==========================================================================

set -e

echo "============================================"
echo " UAV-ISAC-MLLM AutoDL Setup"
echo " Target: RTX 5090 32GB (Blackwell sm_120)"
echo " CUDA: 12.8+"
echo "============================================"

# ---- 1. Conda 环境 ----
echo "[1/6] Creating conda environment..."
conda create -n uavmllm python=3.11 -y
source activate uavmllm

# ---- 2. PyTorch (CUDA 12.8 for Blackwell RTX 5090) ----
echo "[2/6] Installing PyTorch with CUDA 12.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ---- 3. Unsloth (替代 bitsandbytes, 内置 Blackwell sm_120 优化) ----
echo "[3/6] Installing Unsloth + core dependencies..."
pip install unsloth
pip install transformers==4.49.0
pip install peft==0.14.0
pip install accelerate==1.3.0
pip install trl==0.15.0
pip install datasets==3.2.0

# Flash Attention — 先试官方版，失败则用社区 Blackwell 版
echo "[3b/6] Installing flash-attention..."
pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo "  Official flash-attn failed, trying community Blackwell build..."
    pip install https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120/resolve/main/flash_attn-2.7.0%2Bcu128-cp311-cp311-linux_x86_64.whl 2>/dev/null || {
        echo "  WARNING: flash-attn not available. Falling back to scaled_dot_product_attention."
        echo "  Set model.attn_implementation='sdpa' in configs/default.yaml"
    }
}

# ---- 4. 科学计算 ----
echo "[4/6] Installing scientific packages..."
pip install numpy scipy matplotlib pyyaml tqdm

# ---- 5. 验证 GPU ----
echo "[5/6] Verifying GPU..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
"

# ---- 6. 验证关键库 (含 Unsloth) ----
echo "[6/6] Verifying key libraries..."
python -c "
import transformers; print(f'transformers: {transformers.__version__}')
import peft; print(f'peft: {peft.__version__}')
from accelerate import Accelerator; print('accelerate: OK')
try:
    from unsloth import FastLanguageModel
    print('unsloth: OK (Blackwell ready)')
except ImportError:
    print('unsloth: NOT FOUND — check install')
print('All dependencies ready!')
"

echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Login to HuggingFace:  huggingface-cli login"
echo "  2. Generate training data:"
echo "     python scripts/run_data_generation.py --config configs/default.yaml"
echo "  3. Train Stage I (SFT):"
echo "     python src/training/train_sft.py --config configs/default.yaml"
echo "  4. Train Stage II (DPO):"
echo "     python src/training/train_dpo.py --config configs/default.yaml --stage1_ckpt ./outputs/stage1_sft_final"
echo "  5. Evaluate:"
echo "     python src/eval/evaluate.py --config configs/default.yaml --model ./outputs/stage2_dpo_final"
