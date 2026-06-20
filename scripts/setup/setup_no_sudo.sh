#!/bin/bash
# LEGACY: local CPU-only env without sudo (pinned, older vLLM 0.8.3).
# For the supported setup (local + cloud/GPU) see cloud_docs/RUNBOOK.md.
# Assumes an already-activated Python env on PATH.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Installing pip packages..."
pip install --upgrade pip setuptools wheel
echo "Installing torch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
export VLLM_TARGET_DEVICE="cpu"
echo "Installing vLLM..."
pip install vllm==0.8.3 transformers==4.46.1
echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
echo "Setup script finished successfully."
