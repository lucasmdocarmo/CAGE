#!/bin/bash
# Order:     provisioning bracket — on the fresh pod, before/as stage 1 (it stages datasets + prefetches models itself)
# Objective: Container-shaped RunPod bootstrap (root, no sudo/systemd/PPA): canonical-CPython venv, pinned vLLM, SGLang and LMDeploy in their own venvs, charter datasets, model prefetch
# Cloud:     runpod
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
#   bash scripts/runpod/setup_runpod.sh
# Then (the docs/RUNBOOK.md lifecycle — the preflight gate is NOT optional):
#   source cage-env/bin/activate
#   export CAGE_BACKUP_TARGET=s3://<network-volume-id>[/prefix]   # or ssh://... (J4 gate)
#   export CAGE_S3_ENDPOINT=https://s3api-<dc>.runpod.io AWS_DEFAULT_REGION=<DC>   # s3: region REQUIRED
#   <start the serving engine: scripts/2_serving/manage_vllm_server.sh>
#   bash scripts/checks/preflight_check.sh <MODEL> <API_BASE>   # gates (a)-(p); red = do NOT launch
#   nohup bash scripts/3_run/run_full_sweep.sh <model> <N> <T> > sweep.log 2>&1 &
#
# Env:
#   VLLM_VERSION          vLLM pin override (default: the campaign pin below)
#   CHARTER_DATASETS      override the staged dataset roster (space-separated keys)
#   PREFETCH_MODELS       override the model prefetch roster (space-separated HF ids)
#   SKIP_MODEL_PREFETCH=1 bypass model prefetch (e.g. a single-model pod)
#   HF_HUB_DOWNLOAD_TIMEOUT  stalled-read timeout seconds (default 30)
#   SGLANG_VERSION        SGLang pin override (default: the section 7 pin below; own venv sglang-env)
#   SGLANG_PYTHON_VERSION CPython for sglang-env (default 3.12; ADR-0120: SGLang 0.5.10.post1 pulls
#                         outlines_core 0.1.26, which ships no cp313 wheel, and the image has no Rust)
#   LMDEPLOY_VERSION      LMDeploy pin override (default: the section 7 pin below; own venv lmdeploy-env)
#   LMDEPLOY_TORCH_VERSION  torch pinned inside lmdeploy-env (default 2.10.0: vLLM 0.19.1's CUDA 12.8 line)
#   SKIP_ENGINE_INSTALL=1 bypass the SGLang/LMDeploy venvs (a vLLM-only pod)
#   INSTALL_LMCACHE=1     install the LMCache KV-connector package into cage-env for the
#                         retr-store arm (B8; S0F-57, ADR-0158): opt-in per profile, FATAL
#                         when it fails; LMCACHE_VERSION pins it (default: the newest wheel)
#   CAGE_VENV_ROOT        where the three venvs REALLY live (default /root/cage-venvs, the
#                         container disk; ADR-0125, S0F-6). The repo-root names cage-env,
#                         sglang-env and lmdeploy-env are symlinks to them, so every consumer
#                         keeps its path. On a volume-backed pod the repo is on the MooseFS
#                         network volume and a venv there costs 20 s per `import vllm`,
#                         7.5 min per engine start and 79 min per bootstrap (S0, 2026-09-30).
# =============================================================================
set -euo pipefail

# Keep in sync with docs/VLLM_COMPATIBILITY.md (the single pinned version).
VLLM_VERSION="${VLLM_VERSION:-0.19.1}"
# ADR-0128 (S0F-13, S0 2026-09-30): the NIXL transfer library for the prefill/decode
# pair, installed in the SAME pip call as vLLM. Nothing installed it at S0 (layer 1), and
# the nixl 1.x wheels ship nixl_ep built for torch 2.11+, which vLLM imports on sight
# (layer 2). 0.9.0 is the newest release inside vLLM 0.19.1's own declared range
# (requirements/kv_connectors.txt: >=0.7.1,<0.10.0) and its wheels contain no nixl_ep.
# nixl-cu12, not nixl-cu13: the pod runs CUDA 12.8 torch and the 0.9.0 dispatcher tries
# cu13 first when it is present. Keep in sync with section 7 and manage_vllm_pd.sh NIXL_PIN.
NIXL_VERSION="${NIXL_VERSION:-0.9.0}"
# Charter engines #2 and #3 (docs/VLLM_COMPATIBILITY.md section 7; pre-GO item 10,
# 2026-09-26). Each gets its OWN venv beside cage-env: SGLang pins transformers 5.x
# against the repo's transformers<5, and every SGLang release since 2026-05 pins a
# CUDA 13 torch, so the pins below stay on the CUDA 12.8 runtime line the image
# (cu1281) and vLLM 0.19.1 (torch 2.10.0) run on: SGLang 0.5.10.post1 pins torch
# 2.9.1 (cu12.8); LMDeploy 0.17.0 accepts torch 2.0 to 2.12.1, so its venv pins
# torch explicitly or pip would resolve a CUDA 13 build. Keep in sync with section 7.
SGLANG_VERSION="${SGLANG_VERSION:-0.5.10.post1}"
# ADR-0120 (backlog W30, live L40S 2026-09-27): sglang-env is the ONE venv off the
# canonical interpreter. SGLang 0.5.10.post1 -> outlines 0.1.11 -> outlines_core 0.1.26,
# which ships no cp313 wheel (PyPI read 2026-09-27; the cp313 wheels start in the 0.2.x
# line vLLM uses), so on 3.13 pip builds it from source and the image has no Rust. A
# rustup toolchain plus libssl-dev built it on the pod (about 4 min), the heavier fix;
# the cp312 wheel exists and Ubuntu 24.04's system interpreter is 3.12.
SGLANG_PYTHON_VERSION="${SGLANG_PYTHON_VERSION:-3.12}"
LMDEPLOY_VERSION="${LMDEPLOY_VERSION:-0.17.0}"
LMDEPLOY_TORCH_VERSION="${LMDEPLOY_TORCH_VERSION:-2.10.0}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

# ADR-0125 (S0F-6, live H100 2026-09-30): the venvs live on the container disk, never
# on the network volume the repo sits on. Measured on S0 against the two local boxes
# on the same code: bootstrap 79 min vs 7 to 13 min, vLLM launch to API up 246 to 446 s
# vs 62 to 86 s, `import vllm` 21.9 s; 1.8 to 2.4 h of pod time per S0-sized day. The
# venvs are created at their REAL paths under CAGE_VENV_ROOT and the repo-root names are
# links to them (CPython refuses to create a venv through a link and fails on a dangling
# one, venv/__init__.py, so the real path comes first and the link after). logs/ and
# results/ stay on the volume: collect_logs.sh finds logs through no link, and logs must
# outlive a seatbelt kill.
CAGE_VENV_ROOT="${CAGE_VENV_ROOT:-/root/cage-venvs}"
link_venv() {
  # $1 venv name: links $PROJECT_DIR/$1 -> $CAGE_VENV_ROOT/$1. A dangling link (the
  # container disk was wiped by a pod stop or restart) is re-pointed; a REAL directory
  # at the repo root (a pre-ADR-0125 venv on the volume) is refused and never deleted.
  # Returns 1 on refusal (the caller decides whether that is fatal; review 2026-09-30
  # LOW 4: the engine step stays loud and non-fatal).
  local name="$1" real="$CAGE_VENV_ROOT/$1" link="$PROJECT_DIR/$1"
  if [ -e "$link" ] && [ ! -L "$link" ]; then
    warn "${name}: ${link} is a real directory at the repo root (a venv created on the volume before ADR-0125); move it aside by hand (e.g. mv ${link} ${link}.volume), then rerun this bootstrap. Nothing was deleted."
    return 1
  fi
  if ! ln -sfn "$real" "$link"; then
    warn "${name}: could not link ${link} -> ${real}"
    return 1
  fi
  echo "[cage]   ${name}: ${link} -> ${real}"
}
mkdir -p "$CAGE_VENV_ROOT" || die "CAGE_VENV_ROOT=${CAGE_VENV_ROOT} is not writable (ADR-0125: the venvs live there, on the container disk)"
# Name every pre-ADR-0125 real venv directory at once, before any work (review LOW 4).
_real_venv_dirs=""
for _n in cage-env sglang-env lmdeploy-env; do
  if [ -e "$PROJECT_DIR/$_n" ] && [ ! -L "$PROJECT_DIR/$_n" ]; then _real_venv_dirs="${_real_venv_dirs} ${_n}"; fi
done
[ -z "$_real_venv_dirs" ] || die "real venv directories at the repo root (created on the volume before ADR-0125):${_real_venv_dirs}; move each aside by hand (mv <name> <name>.volume), then rerun. Nothing was deleted."

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

# 0d. AWS CLI v2 for the s3 transport (the RunPod network-volume S3 API; scripts/lib/
#     transport.sh refuses s3:// targets without `aws`). Ubuntu 24.04 (the pod image)
#     carries NO `awscli` apt package (removed from noble), so the official installer
#     is used; it lives under /usr/local, OUTSIDE cage-env, so the S0-16 lockfile stays
#     clean. Guarded and loud, never silent: the s3 backend fails closed later anyway.
if command -v aws >/dev/null 2>&1; then
  echo "[cage] [0d] aws already present: $(aws --version 2>&1 | head -1)"
else
  echo "[cage] [0d] installing AWS CLI v2 (official installer; no awscli apt package in Ubuntu 24.04)..."
  _awsdir="$(mktemp -d)"
  if { command -v unzip >/dev/null 2>&1 || DEBIAN_FRONTEND=noninteractive apt-get install -y unzip; } \
     && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o "$_awsdir/awscliv2.zip" \
     && unzip -q -o "$_awsdir/awscliv2.zip" -d "$_awsdir" \
     && "$_awsdir/aws/install" >/dev/null; then
    echo "[cage]   aws: $(aws --version 2>&1 | head -1)"
  else
    warn "AWS CLI v2 install failed; the s3 transport refuses s3:// targets without it (transport.sh)"
  fi
  rm -rf "$_awsdir"
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

# 1. Isolated virtual environment (canonical interpreter, never bare python3), at its
#    REAL container-disk path, then linked at the repo root (ADR-0125).
echo "[cage] [1/5] creating venv cage-env with ${PYBIN} under ${CAGE_VENV_ROOT}..."
"$PYBIN" -m venv "$CAGE_VENV_ROOT/cage-env"
link_venv cage-env || die "cage-env could not be linked at the repo root (see the warning above)"
# shellcheck disable=SC1091
source cage-env/bin/activate
pip install --upgrade pip setuptools wheel

# 2. Official pinned vLLM GPU wheel (provides `vllm serve`) plus the NIXL pair for the
#    prefill/decode launcher (ADR-0128), resolved together so vLLM's own range check
#    sees them.
echo "[cage] [2/5] installing vLLM ${VLLM_VERSION} (GPU wheel) + nixl ${NIXL_VERSION} (cu12)..."
pip install "vllm==${VLLM_VERSION}" "nixl==${NIXL_VERSION}" "nixl-cu12==${NIXL_VERSION}"

# 3. CAGE requirements (the repo's pinned manifest: cage-stats, pynvml,
#    datasets, transformers, FAISS, the metric stack, ...).
echo "[cage] [3/5] installing CAGE requirements..."
pip install -r requirements.txt

# 3b. vLLM (>=0.11) needs openai>=2, but lettucedetect pins openai==1.66.3 —
#     same reconcile as the GCP port (see setup_gpu_cloud.sh [3b] for history).
echo "[cage] [3b] reconciling openai for vLLM ${VLLM_VERSION}..."
pip install -U "openai>=2.0"

# 3d. LMCache (S0F-57, ADR-0158): the KV-connector package of the retr-store arm
#     (B8; the relaunch sets VLLM_KV_TRANSFER_CONFIG LMCacheConnectorV1). Opt-in per
#     profile: INSTALL_LMCACHE=1 when the plan has executable vLLM B8 cells (the
#     master's stage 0 reports NEEDS_LMCACHE from the registered plan and its stage 4
#     proves the import on the pod). The recipe is run_kv_store.sh's: lmcache and the
#     repo's transformers<5 pin in the SAME pip call (lmcache alone pulls transformers
#     5.x). The pairing with vLLM 0.19.1 is NOT live-validated [?] (run_kv_store.sh
#     header; no release note read on 2026-10-09 names 0.19); the import proof below
#     and the stage 4 probe are the gates, and the S0 rehearsal is the live test. The
#     2026-10-08 landing lost its B8 relaunch to ModuleNotFoundError after hours of
#     billing, so a failed install here is FATAL when opted in (never a warning).
if [ "${INSTALL_LMCACHE:-0}" = "1" ]; then
  echo "[cage] [3d] installing lmcache into cage-env (opt-in, INSTALL_LMCACHE=1${LMCACHE_VERSION:+, pin ${LMCACHE_VERSION}})..."
  if pip install --no-cache-dir "lmcache${LMCACHE_VERSION:+==${LMCACHE_VERSION}}" "transformers>=4.36,<5" \
     && python -c 'import vllm, lmcache; print("[cage]   lmcache:", getattr(lmcache, "__version__", "?"), "installed beside vllm", vllm.__version__)'; then
    :
  else
    die "lmcache install or import FAILED (S0F-57, ADR-0158): the retr-store (B8) relaunches cannot start; fix the install (LMCACHE_VERSION) or deregister B8 for this session"
  fi
else
  echo "[cage] [3d] lmcache SKIPPED (INSTALL_LMCACHE=${INSTALL_LMCACHE:-0}; the plan's vLLM B8 cells need it)"
fi

# 3c. Charter engines #2 and #3 in their OWN venvs (pre-GO item 10). lmdeploy-env is
#     created from the canonical interpreter like cage-env; sglang-env from CPython
#     SGLANG_PYTHON_VERSION (ADR-0120). Both are installed through each venv's own
#     pip and without a pip cache (the 120 GB container disk holds the three venvs
#     and the model prefetch, ADR-0125; cage-env stays the active environment for
#     the steps below). Loud and
#     non-fatal like the dataset step: the launchers resolve these venvs by
#     default (manage_sglang_server.sh CAGE_SGLANG_PYTHON, manage_lmdeploy_server.sh
#     CAGE_LMDEPLOY_BIN) and fail closed at start when one is missing. Idempotent
#     without deleting anything: an existing venv is reused and pip completes a
#     killed install on the rerun.
install_engine_venv() {
  # $1 venv NAME (created at $CAGE_VENV_ROOT/<name> and linked at the repo root,
  # ADR-0125) or an absolute venv path (used as is, no link); $2 label, $3.. pip
  # install arguments. The interpreter is the canonical PYBIN unless the caller sets
  # ENGINE_PYBIN for the call (ADR-0120: sglang-env only).
  local name="$1" label="$2" venv
  case "$1" in /*) venv="$1" ;; *) venv="$CAGE_VENV_ROOT/$1" ;; esac
  local pybin="${ENGINE_PYBIN:-$PYBIN}"
  shift 2
  # Rerun guard (review 2026-09-27, MEDIUM 1): `venv` without --clear keeps an existing
  # venv's bin/python links and only rewrites pyvenv.cfg, so re-creating a venv with a
  # DIFFERENT CPython would leave a mixed tree (the 3.13 sglang-env of an earlier
  # bootstrap, for instance). Refuse LOUDLY and name the operator action; never delete.
  if [ -f "$venv/pyvenv.cfg" ] && [ -x "$venv/bin/python3" ]; then
    local have want
    have="$("$venv/bin/python3" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown)"
    want="$("$pybin" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown)"
    if [ "$have" != "$want" ]; then
      warn "${label}: existing ${venv} runs CPython ${have} but ${pybin} is CPython ${want}; refusing to re-create it in place (a rerun would mix the two). Remove ${venv} by hand, then rerun this bootstrap"
      return 1
    fi
  fi
  echo "[cage] [3c] installing ${label} into ${venv} (${pybin})..."
  if "$pybin" -m venv "$venv"; then
    # The link lands right after the venv exists, before pip: a rerun after a killed
    # install then resumes through the same name, and a missing package still fails
    # closed at the launcher's start gate.
    case "$name" in /*) : ;; *) link_venv "$name" || return 1 ;; esac
    if "$venv/bin/pip" install --quiet --no-cache-dir --upgrade pip setuptools wheel \
       && "$venv/bin/pip" install --no-cache-dir "$@"; then
      return 0
    fi
  fi
  warn "${label} install FAILED (${venv}); its launcher refuses to start until this is fixed"
  return 1
}
# An ambient ENGINE_PYBIN must never reach the calls below (review 2026-09-27, LOW 3):
# only the sglang-env call sets it, for that call alone.
unset ENGINE_PYBIN
if [ "${SKIP_ENGINE_INSTALL:-0}" != "1" ]; then
  # sglang-env interpreter (ADR-0120): python${SGLANG_PYTHON_VERSION} with a venv module,
  # resolved like step 0c (default archives, then uv). A miss skips SGLang LOUDLY and
  # leaves cage-env and lmdeploy-env on the canonical interpreter.
  SGLANG_PYBIN="python${SGLANG_PYTHON_VERSION}"
  if ! "$SGLANG_PYBIN" -m venv --help >/dev/null 2>&1; then
    echo "[cage] [3c] ${SGLANG_PYBIN} (or its venv module) not available; attempting apt install (default archives only)..."
    if command -v apt-get >/dev/null 2>&1; then
      DEBIAN_FRONTEND=noninteractive apt-get install -y "${SGLANG_PYBIN}-venv" "${SGLANG_PYBIN}-dev" 2>/dev/null \
        || echo "[cage] [3c] ${SGLANG_PYBIN} not in the image's default archives"
    fi
  fi
  if ! "$SGLANG_PYBIN" -m venv --help >/dev/null 2>&1; then
    # uv goes into the SYSTEM interpreter, never into the activated cage-env (the S0-16
    # lockfile is cage-env's pip freeze; review 2026-09-27, LOW 2): drop the activation
    # inside a subshell first.
    command -v uv >/dev/null 2>&1 || ( deactivate >/dev/null 2>&1; python3 -m pip install --quiet uv ) 2>/dev/null || true
    if command -v uv >/dev/null 2>&1; then
      uv python install "${SGLANG_PYTHON_VERSION}" || warn "uv python install ${SGLANG_PYTHON_VERSION} failed"
      _uv_sg="$(uv python find "${SGLANG_PYTHON_VERSION}" 2>/dev/null || true)"
      if [ -n "${_uv_sg}" ] && [ -x "${_uv_sg}" ]; then SGLANG_PYBIN="${_uv_sg}"; fi
    fi
  fi
  if "$SGLANG_PYBIN" -m venv --help >/dev/null 2>&1; then
    if ENGINE_PYBIN="$SGLANG_PYBIN" install_engine_venv sglang-env "SGLang ${SGLANG_VERSION}" "sglang==${SGLANG_VERSION}"; then
      echo "[cage]   sglang: $(sglang-env/bin/python3 -c 'import sglang, sys; print(sglang.__version__, "on CPython", sys.version.split()[0])' 2>&1 | tail -1)"
    fi
  else
    warn "SGLang ${SGLANG_VERSION} SKIPPED: no CPython ${SGLANG_PYTHON_VERSION} with a venv module in this container (ADR-0120); its launcher refuses to start until sglang-env exists"
  fi
  if install_engine_venv lmdeploy-env "LMDeploy ${LMDEPLOY_VERSION} + torch ${LMDEPLOY_TORCH_VERSION}" \
       "torch==${LMDEPLOY_TORCH_VERSION}" "lmdeploy==${LMDEPLOY_VERSION}"; then
    echo "[cage]   lmdeploy: $(lmdeploy-env/bin/python3 -c 'import lmdeploy; print(lmdeploy.__version__)' 2>&1 | tail -1)"
  fi
else
  echo "[cage] [3c] SGLang/LMDeploy venvs SKIPPED (SKIP_ENGINE_INSTALL=1)"
fi

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

# 4a-live. ADR-0127 (S0F-5): prove the qasper route on THIS pod. datasets 4.x refuses
#     allenai/qasper's loading script, so the loader and the stage above read the Hub's
#     parquet export at one pinned commit; the real library must load it here and the
#     rebuilt 50x3 manifest must hash to the tracked digest. The fake-module unit tests
#     prove only the call shape; this is the live proof. Loud and non-fatal like the
#     stage above: gate (p) refuses an unstaged cache and the loader fails closed.
echo "[cage] [4a-live] qasper: rebuilding the 50x3 manifest through the pinned Hub route..."
if CAGE_HF_LIVE=1 python -m pytest tests/test_qasper_revision_s0f5.py -m integration -q -p no:cacheprovider; then
  echo "[cage]   qasper: the pinned Hub route reproduces data/manifests/qasper_50x3_seed42.json"
else
  warn "qasper live digest check FAILED (tests/test_qasper_revision_s0f5.py): do NOT launch a qasper cell until the pinned route reproduces the manifest"
fi

# 4b. Prefetch model weights ROBUSTLY (bounded by HF_HUB_DOWNLOAD_TIMEOUT above;
#     the retry loop covers a shard that dies mid-transfer; the vLLM server
#     start is the backstop, so this is non-fatal by design).
#     FINAL-SCOPE roster (FINAL SCOPE v2, owner 2026-08-16 — J7: the pilot
#     roster was Qwen3-8B/MiMo/EAGLE): Session A + the PD overlay run the
#     anchor Qwen3-14B; Session B scale runs Llama-3.3-70B. Qwen3-Next +
#     DeepSeek-V3 are [Extension] and deliberately NOT prefetched. Override per
#     pod role — a 1xA100 Session-A pod needs only the anchor:
#       PREFETCH_MODELS="Qwen/Qwen3-14B" bash scripts/runpod/setup_runpod.sh
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
echo "[cage]  RunPod bootstrap complete. Next (docs/RUNBOOK.md lifecycle):"
echo "[cage]    # venvs live under ${CAGE_VENV_ROOT} (container disk, ADR-0125); the repo-root"
echo "[cage]    # names cage-env / sglang-env / lmdeploy-env are links to them"
echo "[cage]    source cage-env/bin/activate"
echo "[cage]    export CAGE_BACKUP_TARGET=s3://<network-volume-id>[/prefix]   # or ssh://[user@]host/path"
echo "[cage]    #   (s3 backend: also export CAGE_S3_ENDPOINT=https://s3api-<dc>.runpod.io,"
echo "[cage]    #    AWS_DEFAULT_REGION=<DC> (REQUIRED: the endpoint rejects any other signing region),"
echo "[cage]    #    and the account-level S3 API key from the console (AWS_ACCESS_KEY_ID = its"
echo "[cage]    #    access key, AWS_SECRET_ACCESS_KEY = its secret; separate from RUNPOD_API_KEY);"
echo "[cage]    #    the bucket name is the network volume id. The AWS CLI v2 was installed above by"
echo "[cage]    #    the official installer (Ubuntu 24.04 has no awscli apt package).)"
echo "[cage]    # 1. start the serving engine (scripts/2_serving/manage_vllm_server.sh;"
echo "[cage]    #    manage_sglang_server.sh / manage_lmdeploy_server.sh use sglang-env /"
echo "[cage]    #    lmdeploy-env from step 3c by default, no activation needed)"
echo "[cage]    # 2. GATE the launch -- a red gate means do NOT launch:"
echo "[cage]    bash scripts/checks/preflight_check.sh <MODEL> <API_BASE>"
echo "[cage]    # 3. run (one run-id for the whole matrix; resume via CAGE_RUN_ID):"
echo "[cage]    nohup bash scripts/3_run/run_full_sweep.sh <model> <N> <T> > sweep.log 2>&1 &"
echo "[cage]  A run with NO backup target REFUSES to start (J4). Teardown goes"
echo "[cage]  through scripts/runpod/teardown_pod.sh (ledger-gated pull first;"
echo "[cage]  NOTE: harness trees carry no ledger.json until the campaign driver"
echo "[cage]  lands -- see docs/RUNBOOK.md section 5 for the teardown contract)."
echo "[cage] ============================================================"
