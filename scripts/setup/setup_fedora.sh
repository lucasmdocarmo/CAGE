#!/bin/bash
set -e

echo "============================================="
echo "   CAGE: Fedora VM Environment Setup Script  "
echo "============================================="

# 1. System Dependencies
echo "[1] Installing OS-level Dependencies..."
sudo dnf install -y python3.12 python3.12-devel python3.12-pip gcc gcc-c++ make cmake
sudo dnf install -y htop wget git patch

# 2. Virtual Environment
echo "[2] Creating isolated Python environment 'cage-env'..."
python3.12 -m venv cage-env
source cage-env/bin/activate
pip install --upgrade pip setuptools wheel

# 3. Core PyTorch (CPU-only to save space on Parallels)
echo "[3] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. vLLM Engine
echo "[4] Installing vLLM and locking dependencies..."
# If pip finds the wheel for your architecture it will download it.
# Otherwise, it builds the CPU backend natively via CMake.
export VLLM_TARGET_DEVICE="cpu"
pip install vllm==0.8.3
pip install transformers==4.46.1

# 5. CAGE Project Requirements
echo "[5] Installing CAGE dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found in current directory."
fi

echo "============================================="
echo " Setup Complete. To start your cluster:      "
echo " source cage-env/bin/activate                "
echo " python scripts/manage_vllm_cluster.py start --model Qwen/Qwen2.5-Coder-0.5B-Instruct --replicas 1 "
echo " python -m pytest tests/                     "
echo "============================================="
