#!/bin/bash
# =============================================================================
# Gate: does the model-native speculative method ACTUALLY engage on THIS vLLM,
# or does it silently no-op?
# =============================================================================
# run_speculative_matrix.sh's native-draft cell (MiMo "mimo_mtp", Qwen "eagle3") can be
# SOFT-ACCEPTED by vLLM (the server starts healthy) yet never actually speculate -- producing
# a COMPLETED cell whose spec_decode_acceptance_rate is null. That looks valid but measures
# nothing, and the existing server-fail sentinel only catches a HARD reject. This gate closes
# the soft hole: it launches the server with the speculative config, generates real tokens,
# and asserts vllm:spec_decode_num_draft_tokens_total > 0 (the SAME counter the telemetry
# scraper reads, so a PASS here means the downstream acceptance number will be non-null).
#
#   PASS (exit 0)         = speculation engages; the native-draft cell is valid.
#   FAIL (exit 1)         = server rejected the config OR no draft tokens (silent no-op).
#   INCONCLUSIVE (exit 2) = probe/parse error.
#
# GPU-only. Mirrors check_fp8_prefix_cache.sh.
# Usage: bash scripts/checks/check_mtp_spec_decode.sh <MODEL> '<spec_json>'
#    or: VLLM_SPECULATIVE_CONFIG='<spec_json>' bash scripts/checks/check_mtp_spec_decode.sh <MODEL>
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-Qwen/Qwen3-8B}"
SPEC="${2:-${VLLM_SPECULATIVE_CONFIG:-}}"
PORT="${VLLM_PORT:-8000}"

if [ -z "$SPEC" ]; then
    echo ">>> [spec-gate] no speculative config given (arg2 or VLLM_SPECULATIVE_CONFIG)."
    exit 2
fi

echo ">>> [spec-gate] launching vLLM with speculative config: $SPEC"
if ! VLLM_SPECULATIVE_CONFIG="$SPEC" "$SCRIPT_DIR/../2_serving/manage_vllm_server.sh" restart "$MODEL"; then
    echo ">>> [spec-gate] FAIL - vLLM rejected the speculative config (server did not start)."
    "$SCRIPT_DIR/../2_serving/manage_vllm_server.sh" stop >/dev/null 2>&1 || true
    exit 1
fi
sleep 10

RESULT=$(MODEL="$MODEL" PORT="$PORT" python3 - <<'PY'
import json, os, urllib.request
model, port = os.environ["MODEL"], os.environ["PORT"]

def gen():
    body = json.dumps({"model": model, "prompt": "Summarize briefly: the quick brown fox jumps over the lazy dog.",
                       "max_tokens": 32, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        json.load(r)

def metrics():
    with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=30) as r:
        return r.read().decode()

def counter(text, name):
    # Sum all label-series of a Prometheus counter; None if the series is absent entirely.
    total = 0.0
    seen = False
    for ln in text.splitlines():
        if ln.startswith("#"):
            continue
        if ln.startswith(name):
            try:
                total += float(ln.rsplit(" ", 1)[1])
                seen = True
            except Exception:
                pass
    return total if seen else None

try:
    for _ in range(3):
        gen()  # generate real tokens so the draft counters accumulate
    draft = counter(metrics(), "vllm:spec_decode_num_draft_tokens_total")
    if draft is None:
        print("ABSENT")   # counter not exposed -> speculation not engaged / wrong vLLM build
    else:
        print(int(draft))
except Exception as e:
    print(f"ERR {e}")
PY
)

"$SCRIPT_DIR/../2_serving/manage_vllm_server.sh" stop >/dev/null 2>&1 || true

case "$RESULT" in
    ERR*)         echo ">>> [spec-gate] probe failed: $RESULT"; echo "INCONCLUSIVE."; exit 2 ;;
    ABSENT)       echo ">>> [spec-gate] FAIL - vllm:spec_decode_num_draft_tokens_total is absent: speculation did NOT engage (silent no-op)."; exit 1 ;;
    ''|*[!0-9]*)  echo ">>> [spec-gate] unexpected output: '$RESULT'"; exit 2 ;;
esac
if [ "$RESULT" -gt 0 ]; then
    echo ">>> [spec-gate] PASS - speculation engaged (draft_tokens=$RESULT). The native-draft cell is valid."
    exit 0
else
    echo ">>> [spec-gate] FAIL - draft_tokens=0: config accepted but no speculation occurred (silent no-op)."
    echo "          Fix the method string (MIMO_MTP_CONFIG) or pin a vLLM that supports it."
    exit 1
fi
