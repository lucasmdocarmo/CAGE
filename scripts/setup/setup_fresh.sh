#!/bin/bash
# LEGACY: local CPU-only env that BUILDS vLLM 0.8.3 from source (slow).
# Only needed on platforms without a working vLLM wheel (e.g. macOS/ARM CPU).
# For cloud/GPU use the official wheel/image — see cloud_docs/RUNBOOK.md.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
rm -rf cage-env
# Now that python3.12-venv is installed, this will work properly
python3.12 -m venv cage-env
source cage-env/bin/activate

echo "Installing pip packages..."
pip install --upgrade pip setuptools wheel
echo "Installing torch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Building vLLM from source..."
rm -rf /tmp/vllm_build
mkdir -p /tmp/vllm_build
cd /tmp/vllm_build
# Download without target device to avoid pip metadata bug
export VLLM_TARGET_DEVICE=""
pip download vllm==0.8.3 --no-deps --no-binary :all:
tar -xzf vllm-0.8.3.tar.gz
cd vllm-0.8.3
# Set target device for CMake compilation
export VLLM_TARGET_DEVICE="cpu"
pip install .

cd "$PROJECT_DIR"
echo "Installing transformers and requirements..."
pip install transformers==4.46.1
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
echo "Environment setup complete!"
