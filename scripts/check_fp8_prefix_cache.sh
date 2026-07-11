#!/bin/bash
# =============================================================================
# Gate: does FP8 KV cache coexist with prefix caching on the CURRENT vLLM?
# =============================================================================
# compressed_cag launches with --kv-cache-dtype fp8 AND relies on prefix reuse. Historically
# FP8 KV disabled prefix caching in vLLM; if it still does on the pulled version, compressed_cag
# becomes "no-reuse + compression" and CONFOUNDS the compression axis (RQ5/H4).
#
# This launches vLLM with fp8 + prefix caching, sends a long repeated prefix twice, and checks
# that usage.prompt_tokens_details.cached_tokens > 0 on the second request.
#   PASS (exit 0) = safe to run compressed_cag.   FAIL (exit 1) = do NOT trust compressed_cag.
# GPU-only (FP8 KV needs a CUDA device). See Cloud/VLLM_COMPATIBILITY.md sec 4.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:-Qwen/Qwen3-4B}"
PORT="${VLLM_PORT:-8000}"

echo ">>> [gate] launching vLLM: --kv-cache-dtype fp8 --enable-prefix-caching"
VLLM_KV_CACHE_DTYPE=fp8 "$SCRIPT_DIR/manage_vllm_server.sh" restart "$MODEL"
sleep 12

RESULT=$(MODEL="$MODEL" PORT="$PORT" python3 - <<'PY'
import json, os, urllib.request
model, port = os.environ["MODEL"], os.environ["PORT"]
prompt = "Shared system prefix. " + ("context tokens " * 300)  # spans more than one KV block
def ask(p):
    body = json.dumps({"model": model, "prompt": p, "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)
try:
    ask(prompt)                       # warm the prefix
    d = ask(prompt)                   # should hit the cache
    cached = (d.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    print(cached or 0)
except Exception as e:
    print(f"ERR {e}")
PY
)

"$SCRIPT_DIR/manage_vllm_server.sh" stop >/dev/null 2>&1 || true

case "$RESULT" in
    ERR*)         echo ">>> [gate] request failed: $RESULT"; echo "INCONCLUSIVE (server/flag error)."; exit 2 ;;
    ''|*[!0-9]*)  echo ">>> [gate] unexpected output: '$RESULT'"; exit 2 ;;
esac
if [ "$RESULT" -gt 0 ]; then
    echo ">>> [gate] PASS — FP8 + prefix caching coexist (cached_tokens=$RESULT). compressed_cag is valid."
    exit 0
else
    echo ">>> [gate] FAIL — cached_tokens=0 under FP8: prefix caching is OFF, so compressed_cag would be confounded."
    echo "          Pin a vLLM where they coexist (Cloud/VLLM_COMPATIBILITY.md sec 4)."
    exit 1
fi
