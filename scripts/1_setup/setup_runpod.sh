#!/bin/bash
# =============================================================================
# CAGE RunPod bootstrap — PRIMARY provider setup (task #137, finding J7)
# =============================================================================
# RunPod is the PRIMARY campaign provider (owner decision 2026-08-16, FINAL
# SCOPE v2 in MyDocs/COST_NEBIUS_RUNPOD_2026-08-16.md: two runs, RunPod secure,
# A100 pods + one L40S S0 gate). setup_gpu_cloud.sh is RETAINED as the GCP
# portability backend.
#
# CONTAINER-SHAPED (J7: the GCP script is DLVM-shaped — sudo/systemctl/PPA are
# dead inside RunPod containers): this script assumes ROOT inside a CUDA
# container (RunPod official PyTorch/CUDA images) —
#   - no sudo ceremony (root already), no systemd (redis via --daemonize),
#   - no deadsnakes PPA: the canonical CPython (CAGE_CANONICAL_PYTHON, finding
#     B1 — fail-closed, never bare python3) comes from apt when the image
#     archive carries it, else from `uv python install` (standalone CPython
#     builds; no PPA, no systemd),
#   - HF_HUB_DOWNLOAD_TIMEOUT exported BEFORE dataset staging AND model
#     prefetch (J7: the GCP script exported it only AFTER the dataset step, so
#     the 2026-07-13 stalled-socket hang was still live during staging),
#   - stages the FULL charter dataset roster (D5), not the 3-dataset pilot set,
#   - prefetches the FINAL-SCOPE model roster, not the pilot-era one.
#
# Usage (inside the pod, from the repo root):
#   bash scripts/1_setup/setup_runpod.sh
# Then:
#   source cage-env/bin/activate
#   export CAGE_BACKUP_TARGET=s3://<network-volume>[/prefix]   # or ssh://...
#   nohup bash scripts/3_run/cloud_run.sh <model> <N> <T> > run.log 2>&1 &
#
# Env:
#   VLLM_VERSION          vLLM pin override (default: the campaign pin below)
#   CHARTER_DATASETS      override the staged dataset roster (space-separated keys)
#   PREFETCH_MODELS       override the model prefetch roster (space-separated HF ids)
#   SKIP_MODEL_PREFETCH=1 bypass model prefetch (e.g. a single-model pod)
#   HF_HUB_DOWNLOAD_TIMEOUT  stalled-read timeout seconds (default 30)
# =============================================================================
set -euo pipefail

# Keep in sync with Cloud/VLLM_COMPATIBILITY.md (the single pinned version).
VLLM_VERSION="${VLLM_VERSION:-0.19.1}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

echo "[cage] ============================================================"
echo "[cage]  RunPod bootstrap (PRIMARY provider; vLLM ${VLLM_VERSION})"
echo "[cage] ============================================================"

# 0. Sanity: a working NVIDIA GPU must be visible.
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "[cage] ERROR: no working NVIDIA GPU (nvidia-smi failed)." >&2
  echo "[cage]        This bootstrap is for RunPod GPU pods (CUDA container images)." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# 0b. Container packages. ROOT inside the container — no sudo, no systemctl.
#     build-essential: vLLM's Triton/torch.compile gcc step; redis-server: the
#     redis/hybrid baselines. Loud (never silent) fallbacks: a failed apt step
#     is announced and the downstream checks catch anything that mattered.
echo "[cage] [0b] installing container packages (build-essential, redis)..."
if [ "$(id -u)" != "0" ]; then
  warn "not running as root — RunPod containers normally run as root; apt installs may fail below"
fi
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq || warn "apt-get update failed; continuing with stale package lists"
  DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential redis-server curl rsync \
    || warn "apt-get install failed (build-essential, redis-server, curl, rsync); vLLM compile / redis baselines / ssh transport may fail below"
else
  warn "apt-get not found (non-Debian container image?); install build-essential + redis-server manually"
fi
# No systemd in containers: daemonize redis directly (idempotent).
if ! redis-cli ping >/dev/null 2>&1; then
  redis-server --daemonize yes 2>/dev/null \
    || warn "could not start redis-server (--daemonize failed); redis/hybrid baselines will fail until it is started"
fi

# 0c. Canonical interpreter (finding B1): the Tier-1 exact pins in
#     requirements.txt were frozen on CPython ${CAGE_CANONICAL_PYTHON}. FAIL
#     CLOSED — never fall back to bare python3 (untested-interpreter drift).
#     Container path (no PPA, no systemd): default archive apt first, then
#     `uv python install` (standalone CPython builds).
PYBIN="python${CAGE_CANONICAL_PYTHON}"
if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "[cage] [0c] ${PYBIN} not on PATH; attempting apt install (default archives only)..."
  if command -v apt-get >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${PYBIN}-venv" "${PYBIN}-dev" 2>/dev/null \
      || echo "[cage] [0c] ${PYBIN} not in the image's default archives"
  fi
fi
if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "[cage] [0c] provisioning ${PYBIN} via uv (standalone CPython; no PPA)..."
  if ! command -v uv >/dev/null 2>&1; then
    # pip-install uv into the system interpreter (bootstrap-only usage; the
    # CAGE venv below is still created from the CANONICAL interpreter).
    python3 -m pip install --quiet uv 2>/dev/null || pip install --quiet uv 2>/dev/null \
      || warn "could not pip-install uv"
  fi
  if command -v uv >/dev/null 2>&1; then
    uv python install "${CAGE_CANONICAL_PYTHON}" || warn "uv python install ${CAGE_CANONICAL_PYTHON} failed"
    # Expose the uv-managed interpreter as python<ver> on PATH for the venv step.
    _uv_py="$(uv python find "${CAGE_CANONICAL_PYTHON}" 2>/dev/null || true)"
    if [ -n "${_uv_py}" ] && [ -x "${_uv_py}" ]; then
      ln -sf "${_uv_py}" "/usr/local/bin/${PYBIN}" 2>/dev/null \
        || PYBIN="${_uv_py}"   # no /usr/local/bin write access: use the absolute path
    fi
  fi
fi
command -v "$PYBIN" >/dev/null 2>&1 || [ -x "$PYBIN" ] \
  || die "canonical interpreter python${CAGE_CANONICAL_PYTHON} unavailable in this container (finding B1: refusing to fall back to bare python3). Use an image that provides it, or install uv, then re-run."
"$PYBIN" -m venv --help >/dev/null 2>&1 \
  || die "python${CAGE_CANONICAL_PYTHON} exists but its venv module is missing, refusing to continue"

# 1. Isolated virtual environment (canonical interpreter, never bare python3).
echo "[cage] [1/5] creating venv cage-env with ${PYBIN}..."
"$PYBIN" -m venv cage-env
# shellcheck disable=SC1091
source cage-env/bin/activate
pip install --upgrade pip setuptools wheel

# 2. Official pinned vLLM GPU wheel (provides `vllm serve`).
echo "[cage] [2/5] installing vLLM ${VLLM_VERSION} (GPU wheel)..."
pip install "vllm==${VLLM_VERSION}"

# 3. CAGE requirements (the repo's pinned manifest: cage-stats, pynvml,
#    datasets, transformers, FAISS, the metric stack, ...).
echo "[cage] [3/5] installing CAGE requirements..."
pip install -r requirements.txt

# 3b. vLLM (>=0.11) needs openai>=2, but lettucedetect pins openai==1.66.3 —
#     same reconcile as the GCP port (see setup_gpu_cloud.sh [3b] for history).
echo "[cage] [3b] reconciling openai for vLLM ${VLLM_VERSION}..."
pip install -U "openai>=2.0"

# 4. HF download robustness FIRST (finding J7: this export must precede BOTH
#    the dataset staging and the model prefetch — the GCP pilot script exported
#    it only after the dataset step, leaving staging exposed to the observed
#    2026-07-13 dead-socket hang: ~57 min stalled at 12/15 GB with no timeout).
#    HF_HUB_DOWNLOAD_TIMEOUT makes a stalled read RAISE (then hf_hub resumes).
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"

# 4a. Stage the FULL charter dataset roster (D5, MyDocs/PUBLICATION.md — J7:
#     the pilot script staged only squad_v2/natural_questions/musique):
#       F1 locality        : squad_v2 (high-sharing pole + abstention),
#                            hotpotqa (partial-overlap middle),
#                            musique (private-evidence pole)
#       F2 pressure        : qasper (THE quality-instrumented pressure workload;
#                            loader validation is a launch blocker)
#       external/load      : scbench (charter 2-subset slice: kv + qa_eng),
#                            sharegpt (load-shape donor ONLY, never quality-scored)
#     NOT staged, per charter: RULER is a GENERATED instrument (src/data/ruler.py,
#     never downloaded) and CRAG is CITE-ONLY (D5#8: no loader work).
#     trivia_qa / natural_questions are pilot-era extras outside the charter
#     roster (still available via download_datasets.py individually).
echo "[cage] [4a/5] staging charter datasets (HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}s)..."
CHARTER_DATASETS="${CHARTER_DATASETS:-squad_v2 hotpotqa musique qasper scbench sharegpt}"
_failed_datasets=""
for _d in ${CHARTER_DATASETS}; do
  python scripts/1_setup/download_datasets.py --dataset "$_d" \
    || { warn "dataset stage FAILED: $_d (the run would lazy-download mid-sweep)"; _failed_datasets="${_failed_datasets} $_d"; }
done
if [ -n "$_failed_datasets" ]; then
  warn "datasets NOT fully staged:${_failed_datasets} — fix these BEFORE launching a timed run (qasper is a launch blocker)"
else
  echo "[cage]   all charter datasets staged: ${CHARTER_DATASETS}"
fi

# 4b. Prefetch model weights ROBUSTLY (bounded by HF_HUB_DOWNLOAD_TIMEOUT above;
#     the retry loop covers a shard that dies mid-transfer; the vLLM server
#     start is the backstop, so this is non-fatal by design).
#     FINAL-SCOPE roster (FINAL SCOPE v2, owner 2026-08-16 — J7: the pilot
#     roster was Qwen3-8B/MiMo/EAGLE): Session A + the PD overlay run the
#     anchor Qwen3-14B; Session B scale runs Llama-3.3-70B. Qwen3-Next +
#     DeepSeek-V3 are [Extension] and deliberately NOT prefetched. Override per
#     pod role — a 1xA100 Session-A pod needs only the anchor:
#       PREFETCH_MODELS="Qwen/Qwen3-14B" bash scripts/1_setup/setup_runpod.sh
PREFETCH_MODELS="${PREFETCH_MODELS:-Qwen/Qwen3-14B meta-llama/Llama-3.3-70B-Instruct}"
if [ "${SKIP_MODEL_PREFETCH:-0}" != "1" ]; then
  echo "[cage] [4b/5] prefetching model weights (HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}s): ${PREFETCH_MODELS}"
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
  echo "[cage] [4b/5] model prefetch SKIPPED (SKIP_MODEL_PREFETCH=1)"
fi

# 5. Verify the telemetry stack the campaign depends on (standard verify step,
#    identical to the GCP port).
echo "[cage] [5/5] verifying telemetry stack..."
python - <<'PY'
try:
    import pynvml
    pynvml.nvmlInit()
    print("[cage]   pynvml OK -> GPU memory-pressure telemetry WILL be captured")
except Exception as e:
    print(f"[cage]   WARNING: pynvml not working -> GPU metrics will be null: {e}")
try:
    # Import the API path CAGE actually uses (pulls in httpx + prometheus_client),
    # NOT just the bare package, so a missing telemetry dep is caught HERE at
    # setup rather than silently zeroing KV/prefix telemetry during the run.
    from cage_stats.api import snapshot_dict  # noqa: F401
    print("[cage]   cage_stats.api import OK -> serving telemetry available")
except Exception as e:
    print(f"[cage]   NOTE: cage_stats.api not importable ({e}); set CAGE_STATS_HOME / "
          "pip install httpx prometheus-client, or telemetry is skipped")
PY

echo
echo "[cage] ============================================================"
echo "[cage]  RunPod bootstrap complete. Next:"
echo "[cage]    source cage-env/bin/activate"
echo "[cage]    export CAGE_BACKUP_TARGET=s3://<network-volume>[/prefix]   # or ssh://[user@]host/path"
echo "[cage]    #   (s3 backend: also export CAGE_S3_ENDPOINT + AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"
echo "[cage]    #    from the RunPod network-volume S3 API credentials)"
echo "[cage]    nohup bash scripts/3_run/cloud_run.sh <model> <N> <T> > run.log 2>&1 &"
echo "[cage]  A run with NO backup target REFUSES to start (J4); teardown goes"
echo "[cage]  through scripts/6_teardown/teardown_pod.sh (ledger-gated pull first)."
echo "[cage] ============================================================"
