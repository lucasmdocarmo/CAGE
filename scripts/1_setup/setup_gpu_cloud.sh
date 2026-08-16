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
# See Cloud/RUNBOOK.md for the full ordered procedure.
#
# PROVISIONING (2026-08-02 charter): the GCP campaign path now PROVISIONS via
# terraform/ (sessions/*.tfvars; `terraform apply` gated by explicit user
# approval). This script does NOT provision -- it bootstraps an ALREADY-CREATED
# box, and remains the SSH-config + neocloud-manual path.
# =============================================================================
set -euo pipefail

# Keep in sync with Cloud/VLLM_COMPATIBILITY.md (the single pinned version).
# 0.19.1 is the Phase-3 pin (2026-07-26). Phase-2 numbers were measured under 0.11.0
# and are NOT comparable across pins (0.19.0 turned the async scheduler on by default).
# Export VLLM_VERSION=0.11.0 to reproduce the Phase-2 environment.
VLLM_VERSION="${VLLM_VERSION:-0.19.1}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

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
#     fails in non-obvious ways: no <py>-venv -> can't create the venv;
#     no <py>-dev/build-essential -> vLLM's Triton/torch.compile gcc step fails
#     ("InductorError: cuda_utils.c"); no redis-server -> redis/hybrid baselines fail.
echo "[cage] [0b] installing system packages (build-essential, redis) + canonical python..."
# Loud (not silent) fallbacks for build/redis: a failed apt step is announced and the
# downstream checks catch anything that actually mattered; the bootstrap continues.
sudo apt-get update -qq || warn "apt-get update failed; continuing with stale package lists"
sudo apt-get install -y build-essential redis-server \
  || warn "apt-get install failed (build-essential, redis-server); vLLM compile or redis baselines may fail below"
sudo systemctl enable --now redis-server 2>/dev/null || redis-server --daemonize yes 2>/dev/null \
  || warn "could not start redis-server (systemctl and --daemonize both failed); redis/hybrid baselines will fail until it is started"

# 0c. Canonical interpreter (code assertion 2026-08-07, finding B1): the Tier-1
#     exact pins in requirements.txt were frozen on CPython ${CAGE_CANONICAL_PYTHON};
#     numpy/pandas at those pins do not even resolve on older interpreters. FAIL
#     CLOSED if it cannot be provisioned — do NOT fall back to bare `python3`
#     (whatever the image ships), which is exactly how untested-interpreter
#     drift happens. deadsnakes is attempted on Ubuntu images only.
PYBIN="python${CAGE_CANONICAL_PYTHON}"
if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "[cage] [0c] ${PYBIN} not on PATH; attempting apt install..."
  sudo apt-get install -y "${PYBIN}-venv" "${PYBIN}-dev" 2>/dev/null || {
    if command -v add-apt-repository >/dev/null 2>&1; then
      echo "[cage] [0c] not in default archives; trying deadsnakes PPA (Ubuntu)..."
      sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
      sudo apt-get update -qq 2>/dev/null || true
      sudo apt-get install -y "${PYBIN}-venv" "${PYBIN}-dev" 2>/dev/null || true
    fi
  }
fi
command -v "$PYBIN" >/dev/null 2>&1 \
  || die "canonical interpreter ${PYBIN} unavailable on this image (finding B1: refusing to fall back to bare python3). Use an image that provides ${PYBIN}, or provision it, then re-run."
"$PYBIN" -m venv --help >/dev/null 2>&1 \
  || die "${PYBIN} exists but its venv module is missing (install ${PYBIN}-venv), refusing to continue"

# 1. Isolated virtual environment (canonical interpreter, never bare python3).
echo "[cage] [1/5] creating venv cage-env with ${PYBIN}..."
"$PYBIN" -m venv cage-env
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

# 3b. vLLM (>=0.11) needs openai>=2 (it imports ResponsePrompt), but lettucedetect pins
#     openai==1.66.3, so the requirements install leaves the old one and vLLM then
#     CRASHES on startup. Force-upgrade (safe: CAGE talks to vLLM over raw HTTP, and
#     lettucedetect's core ModernBERT grounding detector works fine with openai 2.x).
#     MIGRATION NOTE (0.19.1): re-verify this reconcile plus the lmcache<->vLLM pairing
#     and the transformers<5 pin at the next preflight; all three were 0.11-era fixes.
echo "[cage] [3b] reconciling openai for vLLM ${VLLM_VERSION}..."
pip install -U "openai>=2.0"

# 4. Stage the Phase-2 datasets so they are not lazy-downloaded mid-run.
echo "[cage] [4/5] staging datasets (squad_v2, natural_questions, musique)..."
# Loud fallbacks: a failed stage is announced (the run would lazy-download mid-sweep).
python scripts/1_setup/download_datasets.py --dataset squad_v2 || warn "dataset stage failed: squad_v2 (will lazy-download mid-run)"
python scripts/1_setup/download_datasets.py --dataset natural_questions || warn "dataset stage failed: natural_questions (will lazy-download mid-run)"
python scripts/1_setup/download_datasets.py --dataset musique || warn "dataset stage failed: musique (will lazy-download mid-run)"

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
      warn "${_m} not fully prefetched after retries; the vLLM server start will retry (bounded by HF_HUB_DOWNLOAD_TIMEOUT)."
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
echo "[cage]  (speculative 2x2 RETIRED per charter §7.5 -> scripts/deprecated/)"
echo "[cage]  Full procedure + definition of done: Cloud/RUNBOOK.md"
echo "[cage] ============================================================"
