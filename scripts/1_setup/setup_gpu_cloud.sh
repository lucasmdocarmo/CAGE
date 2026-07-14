#!/bin/bash
# =============================================================================
# CAGE GPU cloud bootstrap  (Phase 2 single-GPU driver / Phase 3 router driver)
# =============================================================================
# Run ONCE on a fresh GCP GPU VM (Deep Learning VM image, CUDA already present)
# to make the box ready to run scripts/3_run/cloud_run.sh. This sets up the full CAGE
# Python environment so that EVERYTHING the dissertation describes runs on GCP:
# the orchestrator, vLLM serving, the nine baselines, cage-stats serving
# telemetry, GPU memory-pressure telemetry, and the analytical components.
#
# Unlike scripts/deprecated/setup_ubuntu.sh / setup_fresh.sh (CPU-only, build vLLM
# from source), this installs the official pinned vLLM GPU wheel.
#
# Usage (on the GPU VM, from the repo root):
#   bash scripts/1_setup/setup_gpu_cloud.sh
# Then:
#   source cage-env/bin/activate
#   nohup bash scripts/3_run/cloud_run.sh Qwen/Qwen3-8B 500 3 > run.log 2>&1 &
#
# See Cloud/PHASE2_CHECKLIST.md for the full ordered procedure.
# =============================================================================
set -euo pipefail

# Keep in sync with Cloud/VLLM_COMPATIBILITY.md (the single pinned version).
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

echo "[cage] ============================================================"
echo "[cage]  GPU cloud bootstrap (vLLM ${VLLM_VERSION})"
echo "[cage] ============================================================"

# 0. Sanity: a working NVIDIA GPU must be visible (this is the whole point of Phase 2).
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "[cage] ERROR: no working NVIDIA GPU (nvidia-smi failed)." >&2
  echo "[cage]        This bootstrap is for GPU VMs. On a DLVM, wait for the driver install to finish." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# 0b. System packages the DLVM's MINIMAL system python lacks. Without these the run
#     fails in non-obvious ways: no python3.10-venv -> can't create the venv;
#     no python3.10-dev/build-essential -> vLLM's Triton/torch.compile gcc step fails
#     ("InductorError: cuda_utils.c"); no redis-server -> redis/hybrid baselines fail.
echo "[cage] [0b] installing system packages (python3.10-venv/-dev, build-essential, redis)..."
sudo apt-get update -qq || true
sudo apt-get install -y python3.10-venv python3.10-dev build-essential redis-server || true
sudo systemctl enable --now redis-server 2>/dev/null || redis-server --daemonize yes 2>/dev/null || true

# 1. Isolated virtual environment.
echo "[cage] [1/5] creating venv cage-env..."
python3 -m venv cage-env
# shellcheck disable=SC1091
source cage-env/bin/activate
pip install --upgrade pip setuptools wheel

# 2. Official pinned vLLM GPU wheel (provides `vllm serve`, used by manage_vllm_server.sh).
echo "[cage] [2/5] installing vLLM ${VLLM_VERSION} (GPU wheel)..."
pip install "vllm==${VLLM_VERSION}"

# 3. CAGE requirements: brings cage-stats (git), pynvml (GPU telemetry), datasets,
#    transformers, FAISS, the metric stack, etc.
echo "[cage] [3/5] installing CAGE requirements..."
pip install -r requirements.txt

# 3b. vLLM 0.11.0 needs openai>=2 (it imports ResponsePrompt), but lettucedetect pins
#     openai==1.66.3, so the requirements install leaves the old one and vLLM then
#     CRASHES on startup. Force-upgrade (safe: CAGE talks to vLLM over raw HTTP, and
#     lettucedetect's core ModernBERT grounding detector works fine with openai 2.x).
echo "[cage] [3b] reconciling openai for vLLM 0.11.0..."
pip install -U "openai>=2.0"

# 4. Stage the Phase-2 datasets so they are not lazy-downloaded mid-run.
echo "[cage] [4/5] staging datasets (squad_v2, natural_questions, musique)..."
python scripts/1_setup/download_datasets.py --dataset squad_v2 || true
python scripts/1_setup/download_datasets.py --dataset natural_questions || true
python scripts/1_setup/download_datasets.py --dataset musique || true

# 4b. Prefetch model weights ROBUSTLY so a stalled Hugging Face connection cannot hang the timed
#     vLLM server start mid-sweep. Observed 2026-07-13: a plain snapshot_download hung ~57 min at
#     12/15 GB on a dead socket with no timeout, wasting GPU time. HF_HUB_DOWNLOAD_TIMEOUT makes a
#     stalled read RAISE (then hf_hub resumes); the retry loop covers a shard that dies mid-transfer.
#     Non-fatal by design: the server start is the backstop. Override PREFETCH_MODELS, or set
#     SKIP_MODEL_PREFETCH=1 to bypass (e.g. a single-model run).
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"
PREFETCH_MODELS="${PREFETCH_MODELS:-Qwen/Qwen3-8B XiaomiMiMo/MiMo-7B-RL AngelSlim/Qwen3-8B_eagle3}"
if [ "${SKIP_MODEL_PREFETCH:-0}" != "1" ]; then
  echo "[cage] [4b] prefetching model weights (HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}s): ${PREFETCH_MODELS}"
  for _m in ${PREFETCH_MODELS}; do
    _ok=0
    for _a in 1 2 3 4 5 6; do
      if python - "$_m" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], max_workers=8)
PY
      then _ok=1; break; fi
      echo "[cage]   ${_m}: download attempt ${_a} stalled/failed; resuming in 5s..."; sleep 5
    done
    if [ "$_ok" = "1" ]; then
      echo "[cage]   ${_m}: cached"
    else
      echo "[cage]   WARNING: ${_m} not fully prefetched after retries; the vLLM server start will retry (bounded by HF_HUB_DOWNLOAD_TIMEOUT)."
    fi
  done
else
  echo "[cage] [4b] model prefetch SKIPPED (SKIP_MODEL_PREFETCH=1)"
fi

# 5. Verify the telemetry stack the dissertation depends on.
echo "[cage] [5/5] verifying telemetry stack..."
python - <<'PY'
try:
    import pynvml
    pynvml.nvmlInit()
    print("[cage]   pynvml OK -> GPU memory-pressure telemetry WILL be captured")
except Exception as e:
    print(f"[cage]   WARNING: pynvml not working -> GPU metrics will be null: {e}")
try:
    # Import the API path CAGE actually uses (pulls in httpx + prometheus_client), NOT just
    # the bare package, so a missing telemetry dep is caught HERE at setup rather than
    # silently zeroing speculative-acceptance / KV telemetry during the real run.
    from cage_stats.api import snapshot_dict  # noqa: F401
    print("[cage]   cage_stats.api import OK -> serving telemetry available")
except Exception as e:
    print(f"[cage]   NOTE: cage_stats.api not importable ({e}); set CAGE_STATS_HOME / "
          "pip install httpx prometheus-client, or telemetry is skipped")
PY

echo
echo "[cage] ============================================================"
echo "[cage]  Bootstrap complete. Next:"
echo "[cage]    source cage-env/bin/activate"
echo "[cage]    nohup bash scripts/3_run/cloud_run.sh Qwen/Qwen3-8B 500 3 > run.log 2>&1 &"
echo "[cage]  Launch-time levers (run from their own scripts, they restart the server):"
echo "[cage]    bash scripts/3_run/run_compression.sh Qwen/Qwen3-8B   # FP8 2x2 (gates FP8 x prefix-cache)"
echo "[cage]    bash scripts/3_run/run_speculative_matrix.sh Qwen/Qwen3-8B   # speculative 2x2 (repeat for MiMo-7B-RL)"
echo "[cage]  Full procedure + definition of done: Cloud/PHASE2_CHECKLIST.md"
echo "[cage] ============================================================"
