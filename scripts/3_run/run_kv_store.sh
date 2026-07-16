#!/bin/bash
# KV-store family (task #83, per Documentation/RELATED_WORK_KVCACHE_STORES.md verdict:
# "add exactly ONE real KV-block store: LMCache (with CacheBlend)").
#
#   lmcache_rag   RAG serving with prefill KV served from LMCache via vLLM's
#                 KV-connector (--kv-transfer-config LMCacheConnectorV1). This is the
#                 position-independent-caching (PIC) comparison arm: CacheBlend
#                 (EuroSys'25, arXiv 2405.16444) reuses retrieved-chunk KV regardless
#                 of position -- the literature's answer to the fact that per-query
#                 context breaks vanilla prefix caching (CAGE's measured -3.3%).
#
# STATUS: EXPERIMENTAL until live-validated. The LMCache<->vLLM 0.11.0 pairing is
# verified by the gates below at run time (import + server health with the connector),
# NOT assumed. First live validation: the 100x3 validation-run preflight.
# Install on the VM first:  pip install lmcache "transformers>=4.36,<5"
#   (re-assert the transformers pin IN THE SAME CALL: lmcache alone pulls transformers 5.x,
#   which breaks vLLM 0.11.0's tokenizer path. Not in requirements.txt on purpose --
# optional heavy dep; setup_gpu_cloud.sh stays lean).
#
# Outputs: $CAGE_RUN_ROOT/kv_store/lmcache_rag/trial_*/  (tree "kv_store" in _results_loader)
# Resume: complete cell skipped; CAGE_FORCE_RERUN=1 wipes and re-runs.
# Usage: [NUM_QUERIES=100 NUM_TRIALS=3] bash scripts/3_run/run_kv_store.sh [model]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
source scripts/lib/_serving_config.sh

MODEL=${1:-"Qwen/Qwen3-8B"}
DATASET="${DATASET:-squad_v2}"
NUM_QUERIES=${NUM_QUERIES:-500}
NUM_TRIALS=${NUM_TRIALS:-3}
SEED=${SEED:-42}
OUTPUT_DIR="${CAGE_RUN_ROOT:-results/phase2/local}/kv_store"
LABEL="lmcache_rag"
mkdir -p "$OUTPUT_DIR"

TELEMETRY_FLAG=""
if [ "${VLLM_TELEMETRY:-0}" != "0" ]; then TELEMETRY_FLAG="--vllm-telemetry"; fi

cleanup() { ./scripts/2_serving/manage_vllm_server.sh stop >/dev/null 2>&1 || true; }
trap cleanup EXIT

fail_cell() {  # <reason>
    mkdir -p "$OUTPUT_DIR/$LABEL"
    echo "STATUS=failed reason=$1 model=$MODEL $(date)" > "$OUTPUT_DIR/$LABEL/STATUS"
    echo "KV_STORE_FAILED: $LABEL ($1)"
    exit 1
}

cell_complete() {
    local t
    for ((t = 1; t <= NUM_TRIALS; t++)); do
        [ -f "$OUTPUT_DIR/$LABEL/trial_${t}/metrics.json" ] || return 1
    done
    return 0
}

if cell_complete && [ "${CAGE_FORCE_RERUN:-0}" != "1" ]; then
    echo "SKIP (complete): $LABEL"
    echo "KV_STORE_DONE"
    exit 0
fi
[ -d "$OUTPUT_DIR/$LABEL" ] && { echo "wiping partial $LABEL"; rm -rf "$OUTPUT_DIR/$LABEL"; }

# Gate 1: connector package importable in the serving interpreter (version pairing is
# exactly what this catches -- fail loud, never serve a silently-connectorless arm).
if ! python3 - <<'PY'
import sys
try:
    import lmcache  # noqa: F401
    print(f"lmcache {getattr(lmcache, '__version__', '?')} importable")
except Exception as exc:
    print(f"GATE FAIL: lmcache not importable: {exc}", file=sys.stderr)
    sys.exit(1)
PY
then
    fail_cell lmcache_missing
fi

# Gate 2: server actually starts WITH the connector and answers /health.
echo "=== restarting vLLM with LMCache connector ==="
if ! VLLM_KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    ./scripts/2_serving/manage_vllm_server.sh restart "$MODEL"; then
    fail_cell server_with_connector
fi
sleep 5

# The arm: standard RAG serving; the KV reuse happens server-side in the connector.
# Compare against the plain `rag` cell -- same retrieval, same prompts; the connector
# is the only serving delta. cached_prompt_tokens + TTFT tell the reuse story.
# --reset-cache-between-trials: audit 2026-07-16 M1 -- lmcache_rag ran without it, so
# trials 2/3 started warm while the `rag` comparator started cold (trial independence
# broken). All arms now reset between trials.
if ! python3 scripts/3_run/run_experiment.py \
    --baseline rag \
    --baseline-label "$LABEL" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num-queries "$NUM_QUERIES" \
    --num-trials "$NUM_TRIALS" \
    --seed "$SEED" \
    --reset-cache-between-trials \
    --output-dir "$OUTPUT_DIR/$LABEL" \
    $TELEMETRY_FLAG; then
    fail_cell run_experiment
fi

echo "KV_STORE_DONE ($LABEL complete)"
