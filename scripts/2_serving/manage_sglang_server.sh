#!/bin/bash
# =============================================================================
# SGLang Server Management Script   (charter D2 engine #2 -- RadixAttention)
# =============================================================================
# Manages the SGLang inference server for CAGE experiments, mirroring
# manage_vllm_server.sh's daemon discipline (pidfile, health-wait, per-start
# serving-config capture) so every engine launches under the SAME uniform
# serving regime (scripts/lib/_serving_config.sh).
#
# Closes finding D1 (MyDocs/CODE_ASSERTION_2026-08.md Topic 4): the client
# adapter existed (src/inference/sglang_adapter.py -> http://localhost:30000)
# but nothing in the repo STARTED an SGLang server, and the charter §6.5
# iso-BYTES budget had no launch-level mapping onto SGLang's native dial.
#
# ISO-BYTES BUDGET (§6.5): the uniform operating point
# VLLM_GPU_MEMORY_UTILIZATION maps ONE-TO-ONE TO FIRST ORDER onto SGLang's
# --mem-fraction-static -- NOT identically [VERIFY-LIVE at S0]: vLLM's F also
# covers its profiled activation workspace while SGLang budgets activations
# from 1-F (so the same F overshoots SGLang's KV bytes by ~that workspace),
# and SGLang sizes against memory available at init (= device total only on
# an empty GPU; co-resident metric models shrink it). See
# cage_sglang_mem_fraction() in _serving_config.sh for the full rationale.
# The mapping sets the dial; the preflight iso-bytes gate -- run WITH the
# co-resident stack loaded -- asserts the REALIZED KV-pool bytes across
# engines (never assumed from the dial).
#
# [VERIFY-LIVE at S0]: every SGLang CLI flag below follows SGLang's documented
# server CLI, but none has been exercised by this codebase yet (SGLang is not
# installed locally; its exact pin is minted at S0 -- VLLM_COMPATIBILITY.md
# §7). S0 shakedown item 2 proves this launcher end-to-end.
#
# Usage:
#   ./scripts/2_serving/manage_sglang_server.sh start <model> [--no-prefix-cache]
#   ./scripts/2_serving/manage_sglang_server.sh stop
#   ./scripts/2_serving/manage_sglang_server.sh restart <model> [--no-prefix-cache]
#   ./scripts/2_serving/manage_sglang_server.sh status
# =============================================================================

set -euo pipefail

# Anchor paths to the repo root so logs ALWAYS land in <repo>/logs/sglang/,
# regardless of the caller's working directory (same rule as the vLLM launcher).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"
# Serving-uniformity source of truth (Option A) + §6.5 budget-mapping helpers.
# shellcheck source=scripts/lib/_serving_config.sh
source "$PROJECT_DIR/scripts/lib/_serving_config.sh"

PORT="${SGLANG_PORT:-30000}"   # SGLangAdapter's default api_base port
LOG_DIR="$PROJECT_DIR/logs/sglang"
# Daemon discipline: the launched server's PID is recorded here at start and
# cleared at stop, so status/stop have an authoritative handle.
PID_FILE="$LOG_DIR/sglang_server.pid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

mkdir -p "$LOG_DIR"

get_sglang_pid() {
    # Prefer the pidfile written at start; validate the PID is alive AND still
    # an SGLang process (PIDs get recycled) before trusting it.
    local fpid
    if [ -f "$PID_FILE" ]; then
        fpid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [ -n "$fpid" ] && ps -p "$fpid" -o command= 2>/dev/null | grep -q "sglang"; then
            echo "$fpid"
            return 0
        fi
    fi
    # Fallback (stale pidfile): pgrep. head -n1 because -f can match the
    # launcher plus scheduler/detokenizer workers.
    pgrep -f "sglang.launch_server" | head -n1 || true
}

get_loaded_model() {
    curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | \
        python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data.get('data') else '')" 2>/dev/null || echo ""
}

get_server_radix_mode() {
    # RadixAttention (SGLang's prefix reuse) is DEFAULT-ON: absence of
    # --disable-radix-cache on the live cmdline means enabled.
    local pid cmd
    pid=$(get_sglang_pid)
    if [ -z "$pid" ]; then
        echo "unknown"
        return 1
    fi
    cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
    if [[ "$cmd" == *"--disable-radix-cache"* ]]; then
        echo "disabled"
    else
        echo "enabled"
    fi
    return 0
}

start_server() {
    local model="$1"
    local want_prefix_cache=true
    if [ "${2:-}" = "--no-prefix-cache" ]; then
        want_prefix_cache=false
    fi

    echo -e "${YELLOW}Starting SGLang server with model: $model${NC}"

    # §6.5 mapping: first-order identity onto --mem-fraction-static (semantics
    # differ -- see the header + cage_sglang_mem_fraction; realized-bytes gate
    # is the equalizer). Computed BEFORE the reuse check so reuse can require
    # dial parity on the live cmdline (adversarial review 2026-08-12: a
    # pressure-sweep iteration invoked via `start` must never reuse the
    # previous budget's server while the driver labels data with the new one).
    local mem_fraction
    mem_fraction=$(cage_sglang_mem_fraction) \
        || die "cage_sglang_mem_fraction failed (is scripts/lib/_serving_config.sh intact?)"

    # Check if already running
    local pid
    pid=$(get_sglang_pid)
    if [ -n "$pid" ]; then
        local loaded_model radix_mode has_prefix_cache live_cmd dials_match
        loaded_model=$(get_loaded_model)
        # `|| radix_mode=unknown`: the probe returns non-zero when the pid
        # vanished between checks; a bare assignment would abort the whole
        # script under set -e instead of falling through to the restart path.
        radix_mode=$(get_server_radix_mode) || radix_mode="unknown"
        has_prefix_cache=true
        [ "$radix_mode" = "disabled" ] && has_prefix_cache=false
        [ "$radix_mode" = "unknown" ] && has_prefix_cache="unknown"

        # Reuse ONLY when no launch lever is requested AND the live cmdline
        # matches what this environment would launch: exact budget dial +
        # uniform context length (adversarial review 2026-08-12). The live
        # --kv-cache-dtype cannot be read back over the API, so if it is set
        # we force a restart rather than risk mislabeling the arm's data.
        live_cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
        dials_match=true
        [[ "$live_cmd" == *"--mem-fraction-static $mem_fraction"* ]] || dials_match=false
        [[ "$live_cmd" == *"--context-length ${VLLM_MAX_MODEL_LEN}"* ]] || dials_match=false

        if [ "$loaded_model" = "$model" ] && [ "$has_prefix_cache" = "$want_prefix_cache" ] \
           && [ "$dials_match" = "true" ] \
           && [ -z "${SGLANG_KV_CACHE_DTYPE:-}" ]; then
            echo -e "${GREEN}✓ Server already running with correct model, cache mode, and dials ($model)${NC}"
            return 0
        else
            echo -e "${RED}✗ Server state does not match requested model/cache mode/dials${NC}"
            echo -e "${YELLOW}  Loaded model: $loaded_model | radix cache: $has_prefix_cache | dials match: $dials_match${NC}"
            echo -e "${YELLOW}  Requested model: $model | radix cache: $want_prefix_cache${NC}"
            echo -e "${YELLOW}  Stopping and restarting...${NC}"
            stop_server
            sleep 2
        fi
    fi

    local timestamp log_file
    timestamp=$(date +%Y%m%d_%H%M%S)
    log_file="$LOG_DIR/sglang_${model//\//_}_${timestamp}.log"

    # Argv as an ARRAY so values are never word-split (vLLM-launcher rule).
    local -a sglang_args=( --model-path "$model" --port "$PORT" )
    # Parity with the vLLM launcher: repos shipping custom modeling code
    # (MiMo-class) fail model validation without this; benchmark box only.
    sglang_args+=( --trust-remote-code )
    sglang_args+=( --mem-fraction-static "$mem_fraction" )
    # Uniform context length (vLLM --max-model-len analogue).
    sglang_args+=( --context-length "${VLLM_MAX_MODEL_LEN}" )

    # RadixAttention is default-ON; the cache-off arm disables it explicitly.
    if [ "$want_prefix_cache" = "false" ]; then
        sglang_args+=( --disable-radix-cache )
    fi

    # Uniform eager lever: SGLang's CUDA-graph toggle (vLLM --enforce-eager
    # analogue). Same recorded-deviation semantics as the vLLM launcher.
    if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
        sglang_args+=( --disable-cuda-graph )
        echo "Eager mode ON: --disable-cuda-graph"
    fi

    # Optional server-side KV-cache compression (compressed_cag analogue), e.g.
    #   SGLANG_KV_CACHE_DTYPE=fp8_e5m2 ./scripts/2_serving/manage_sglang_server.sh restart <model>
    if [ -n "${SGLANG_KV_CACHE_DTYPE:-}" ]; then
        sglang_args+=( --kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE}" )
        echo "KV-cache compression enabled: --kv-cache-dtype ${SGLANG_KV_CACHE_DTYPE}"
    fi

    # Engine-version provenance (the SGLang pin is minted at S0; record what
    # actually served every start).
    local engine_version
    engine_version=$(python3 -c "import sglang; print(getattr(sglang, '__version__', 'unknown'))" 2>/dev/null || echo "unavailable")

    echo "Server args: python3 -m sglang.launch_server ${sglang_args[*]}  (sglang=$engine_version)"

    # Per-(re)start serving-config capture (same contract as the vLLM launcher:
    # run_manifest.json is built once, so per-tree restarts must self-record).
    # Skipped silently when CAGE_RUN_ROOT is unset; never fatal to startup.
    if [ -n "${CAGE_RUN_ROOT:-}" ]; then
        local cfg_dir="$CAGE_RUN_ROOT/observability/serving_configs"
        local model_slug cfg_file
        model_slug=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]' | sed -E 's|.*/||; s|[^a-z0-9]+|-|g; s|^-+||; s|-+$||')
        cfg_file="$cfg_dir/$(date -u +%Y%m%dT%H%M%SZ)_sglang_${model_slug}.json"
        mkdir -p "$cfg_dir" 2>/dev/null || true
        SC_ENGINE="sglang" \
        SC_VERSION="$engine_version" \
        SC_MODEL="$model" \
        SC_PORT="$PORT" \
        SC_PREFIX="$want_prefix_cache" \
        SC_MAX_LEN="${VLLM_MAX_MODEL_LEN}" \
        SC_BUDGET_F="${VLLM_GPU_MEMORY_UTILIZATION}" \
        SC_DIAL_FLAG="--mem-fraction-static" \
        SC_DIAL_VALUE="$mem_fraction" \
        SC_KV_DTYPE="${SGLANG_KV_CACHE_DTYPE:-auto}" \
        SC_EAGER="${VLLM_ENFORCE_EAGER:-0}" \
        SC_ARGS="python3 -m sglang.launch_server ${sglang_args[*]}" \
        SC_FILE="$cfg_file" \
        python3 - <<'PYEOF' || echo "  (serving-config capture failed; non-fatal)"
import datetime
import json
import os

cfg = {
    "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "engine": os.environ["SC_ENGINE"],
    "engine_version": os.environ.get("SC_VERSION") or None,
    "model": os.environ["SC_MODEL"],
    "port": int(os.environ["SC_PORT"]),
    "enable_prefix_caching": os.environ.get("SC_PREFIX") == "true",
    "max_model_len": int(os.environ["SC_MAX_LEN"]),
    "uniform_budget_fraction": float(os.environ["SC_BUDGET_F"]),
    "native_budget_dial": {
        "flag": os.environ["SC_DIAL_FLAG"],
        "value": float(os.environ["SC_DIAL_VALUE"]),
    },
    "kv_cache_dtype": os.environ.get("SC_KV_DTYPE") or "auto",
    "enforce_eager": os.environ.get("SC_EAGER") == "1",
    "args": os.environ["SC_ARGS"],
}
with open(os.environ["SC_FILE"], "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")
PYEOF
        echo "  Serving config captured: $cfg_file"
    fi

    # Bound Hugging Face downloads so a dead socket RAISES instead of hanging
    # the start window (same backstop as the vLLM launcher).
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"

    echo "Starting SGLang server (logging to $log_file)..."
    nohup python3 -m sglang.launch_server "${sglang_args[@]}" > "$log_file" 2>&1 &

    local server_pid=$!
    printf '%s\n' "$server_pid" > "$PID_FILE"
    echo "Server PID: $server_pid (pidfile: $PID_FILE)"

    # Wait for readiness by polling the OpenAI surface the adapter actually
    # uses (/v1/models): it answers correctly only once the model is served,
    # which makes it a stricter probe than /health. CUDA-graph capture can
    # take minutes on smaller GPUs; override with SGLANG_START_TIMEOUT.
    echo "Waiting for server to start..."
    local max_wait="${SGLANG_START_TIMEOUT:-300}"
    local waited=0
    local loaded
    while [ "$waited" -lt "$max_wait" ]; do
        loaded=$(get_loaded_model)
        if [ "$loaded" = "$model" ]; then
            echo -e "${GREEN}✓ Server ready with model: $model${NC}"
            echo "  View logs: tail -f $log_file"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done

    echo -e "\n${RED}✗ Server failed to start within ${max_wait}s${NC}"
    echo "Check logs: $log_file"
    return 1
}

stop_server() {
    echo -e "${YELLOW}Stopping SGLang server...${NC}"

    # SGLang runs scheduler/detokenizer WORKER processes alongside the launch
    # process; kill the whole family or a worker keeps the GPU (the exact
    # orphaned-EngineCore failure mode the vLLM launcher fixed).
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang::"             2>/dev/null || true
    sleep 2
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "sglang::"             2>/dev/null || true

    # Belt-and-suspenders: kill any remaining SGLang process still holding the
    # GPU, but do NOT kill co-resident GPU users (metric models / cage-stats).
    local held
    held=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    for p in $held; do
        local cmd
        cmd=$(ps -p "$p" -o args= 2>/dev/null || true)
        case "$cmd" in
            *sglang*) kill -9 "$p" 2>/dev/null || true ;;
            *) [ -n "$cmd" ] && echo "  (left non-SGLang GPU process $p alive: ${cmd:0:60})" ;;
        esac
    done
    sleep 2

    # The daemon is down: clear its pidfile so a stale PID can never be trusted.
    rm -f "$PID_FILE"

    local gpu_mem
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true)
    echo -e "${GREEN}✓ Server stopped${NC} (GPU mem used: ${gpu_mem:-n/a})"
}

status_server() {
    local pid loaded_model
    pid=$(get_sglang_pid)

    if [ -z "$pid" ]; then
        echo -e "${RED}✗ SGLang server is NOT running${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ SGLang server is running${NC}"
    echo "  PID: $pid"

    loaded_model=$(get_loaded_model)
    if [ -n "$loaded_model" ]; then
        echo "  Model: $loaded_model"
        echo "  Port: $PORT"
        echo "  Radix cache: $(get_server_radix_mode)"
    else
        echo -e "${YELLOW}  Warning: Unable to query loaded model${NC}"
    fi
}

case "${1:-}" in
    start)
        if [ -z "${2:-}" ]; then
            echo "Usage: $0 start <model> [--no-prefix-cache]"
            echo "Example: $0 start Qwen/Qwen3-4B"
            exit 1
        fi
        start_server "$2" "${3:-}"
        ;;
    stop)
        stop_server
        ;;
    restart)
        if [ -z "${2:-}" ]; then
            echo "Usage: $0 restart <model> [--no-prefix-cache]"
            exit 1
        fi
        stop_server
        sleep 2
        start_server "$2" "${3:-}"
        ;;
    status)
        status_server
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status} [model]"
        echo ""
        echo "Commands:"
        echo "  start <model>   - Start SGLang server with specified model"
        echo "  stop            - Stop SGLang server"
        echo "  restart <model> - Restart SGLang server with specified model"
        echo "  status          - Check SGLang server status"
        exit 1
        ;;
esac
