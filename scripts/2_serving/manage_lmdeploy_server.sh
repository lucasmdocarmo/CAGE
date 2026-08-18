#!/bin/bash
# =============================================================================
# LMDeploy Server Management Script  (charter D2 engine #3 -- TurboMind)
# =============================================================================
# Manages the LMDeploy api_server for CAGE experiments, mirroring
# manage_vllm_server.sh's daemon discipline (pidfile, health-wait, per-start
# serving-config capture) so every engine launches under the SAME uniform
# serving regime (scripts/lib/_serving_config.sh).
#
# Closes finding D1 (MyDocs/CODE_ASSERTION_2026-08.md Topic 4): the client
# adapter existed (src/inference/lmdeploy_adapter.py -> http://localhost:23333)
# but nothing in the repo STARTED an LMDeploy server, and the charter §6.5
# iso-BYTES budget had no launch-level mapping onto LMDeploy's native dial.
#
# TURBOMIND POLICY (P7, user-confirmed 2026-07-27): TurboMind-pinned-where-
# supported, ABSENT elsewhere, NEVER silently served by the PyTorch fallback
# engine. --backend turbomind is passed explicitly, and after health-up this
# launcher asserts TurboMind from the launch log (assert_turbomind_selected)
# -- the adapter is transport-only and cannot observe the backend, so the
# launch side owns the first line of defense; the preflight gate re-verifies.
#
# ISO-BYTES BUDGET (§6.5): LMDeploy's cache_max_entry_count is a fraction of
# the FREE memory left AFTER weights load (>=0.4 semantics) -- a DIFFERENT
# denominator than vLLM's total-memory fraction. cage_lmdeploy_cache_fraction()
# in _serving_config.sh converts: frac_free = (F*T - W)/(T - W). It needs the
# weight footprint W: set CAGE_MODEL_WEIGHTS_GIB (fp16/bf16 ~ 2 x params-in-B,
# e.g. ~28 for Qwen3-14B), or set LMDEPLOY_CACHE_MAX_ENTRY_COUNT explicitly as
# a RECORDED deviation. No mapping -> refuse to launch (fail-closed): an
# unmapped dial silently breaks §6.5 budget parity. Preflight gate (j) -- the
# CAGE-ISO-BYTES-GATE in scripts/checks/preflight_check.sh -- parses this
# launcher's startup log ([BlockManager] block_size/max_block_count) and
# asserts the REALIZED KV-pool bytes; the mapping only sets the dial.
#
# [VERIFY-LIVE at S0]: every LMDeploy CLI flag below follows LMDeploy's
# documented api_server CLI, but none has been exercised by this codebase yet
# (LMDeploy is not installed locally; its exact pin is minted at S0 --
# VLLM_COMPATIBILITY.md §7). The TurboMind log-marker patterns are likewise
# VERIFY-LIVE. S0 shakedown item 2 proves this launcher end-to-end.
#
# Usage:
#   ./scripts/2_serving/manage_lmdeploy_server.sh start <model> [--no-prefix-cache]
#   ./scripts/2_serving/manage_lmdeploy_server.sh stop
#   ./scripts/2_serving/manage_lmdeploy_server.sh restart <model> [--no-prefix-cache]
#   ./scripts/2_serving/manage_lmdeploy_server.sh status
# =============================================================================

set -euo pipefail

# Anchor paths to the repo root so logs ALWAYS land in <repo>/logs/lmdeploy/,
# regardless of the caller's working directory (same rule as the vLLM launcher).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"
# Serving-uniformity source of truth (Option A) + §6.5 budget-mapping helpers.
# shellcheck source=scripts/lib/_serving_config.sh
source "$PROJECT_DIR/scripts/lib/_serving_config.sh"

PORT="${LMDEPLOY_PORT:-23333}"   # LMDeployAdapter's default api_base port
LOG_DIR="$PROJECT_DIR/logs/lmdeploy"
# Daemon discipline: the launched server's PID is recorded here at start and
# cleared at stop, so status/stop have an authoritative handle.
PID_FILE="$LOG_DIR/lmdeploy_server.pid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

mkdir -p "$LOG_DIR"

get_lmdeploy_pid() {
    # Prefer the pidfile written at start; validate the PID is alive AND still
    # an LMDeploy process (PIDs get recycled) before trusting it.
    local fpid
    if [ -f "$PID_FILE" ]; then
        fpid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [ -n "$fpid" ] && ps -p "$fpid" -o command= 2>/dev/null | grep -q "lmdeploy"; then
            echo "$fpid"
            return 0
        fi
    fi
    # Fallback (stale pidfile): pgrep.
    pgrep -f "lmdeploy serve api_server" | head -n1 || true
}

get_loaded_model() {
    curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | \
        python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data.get('data') else '')" 2>/dev/null || echo ""
}

get_server_prefix_cache_mode() {
    # LMDeploy prefix caching is DEFAULT-OFF: presence of --enable-prefix-caching
    # on the live cmdline means enabled (inverse of SGLang's default-on radix).
    local pid cmd
    pid=$(get_lmdeploy_pid)
    if [ -z "$pid" ]; then
        echo "unknown"
        return 1
    fi
    cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
    if [[ "$cmd" == *"--enable-prefix-caching"* ]]; then
        echo "enabled"
    else
        echo "disabled"
    fi
    return 0
}

# Set by assert_turbomind_selected; recorded machine-readably post-assert.
TURBOMIND_CHECK_RESULT="unchecked"

assert_turbomind_selected() {
    # P7: --backend turbomind is requested above, but LMDeploy applies AUTO
    # backend selection server-side even then (autoget_backend can fall back
    # to the PyTorch engine for unsupported models), and the HTTP adapter
    # cannot observe the choice -- verify from the launch log.
    #
    # PRECEDENCE MATTERS (adversarial review 2026-08-12, BLOCKER fix): the
    # fallback warning itself ("Fallback to pytorch engine because ... not
    # supported by turbomind engine") CONTAINS the word 'turbomind', so the
    # fallback check runs FIRST and UNCONDITIONALLY -- co-occurrence of both
    # markers is a fallback, never a confirmation. The positive check then
    # requires a selection-anchored marker (TurboMind's [TM] logger prefix or
    # an engine/backend context line), not bare substring presence. Depends on
    # --log-level INFO below (api_server's ERROR default suppresses these
    # lines entirely). Marker patterns are [VERIFY-LIVE at S0]; only drop to
    # CAGE_LMDEPLOY_BACKEND_CHECK=warn as a RECORDED deviation.
    #
    # Fail-closed teardown: a refusal STOPS the just-launched server -- die
    # alone would leave a healthy, policy-refused (possibly PyTorch) server on
    # $PORT for any later "is a server up?" check to adopt.
    local log_file="$1"
    TURBOMIND_CHECK_RESULT="unconfirmed"
    if grep -qiE "fallback to pytorch|pytorch.{0,20}(engine|backend)|(engine|backend).{0,20}pytorch" "$log_file"; then
        TURBOMIND_CHECK_RESULT="pytorch-fallback"
        stop_server
        die "LMDeploy selected/fell back to the PyTorch engine (log: $log_file). P7 forbids mixing mechanism families -- this model is ABSENT-by-policy on LMDeploy, not served by the fallback. Refused server torn down."
    fi
    if grep -qiE "\[TM\]|turbomind.{0,30}(engine|backend|model|start)|(engine|backend).{0,30}turbomind" "$log_file"; then
        TURBOMIND_CHECK_RESULT="confirmed"
        echo -e "${GREEN}✓ TurboMind backend confirmed in launch log${NC}"
        return 0
    fi
    if [ "${CAGE_LMDEPLOY_BACKEND_CHECK:-strict}" = "warn" ]; then
        TURBOMIND_CHECK_RESULT="warn-bypassed"
        warn "no TurboMind marker found in $log_file (pattern is VERIFY-LIVE); proceeding because CAGE_LMDEPLOY_BACKEND_CHECK=warn -- recorded in the backend-check sidecar, re-verify at preflight"
        return 0
    fi
    stop_server
    die "cannot confirm the TurboMind backend from $log_file (no marker; pattern is VERIFY-LIVE at S0). Refusing under P7 never-mixed; refused server torn down. If inspection shows TurboMind IS running, update the marker pattern here; CAGE_LMDEPLOY_BACKEND_CHECK=warn overrides as a recorded deviation."
}

start_server() {
    local model="$1"
    local want_prefix_cache=true
    if [ "${2:-}" = "--no-prefix-cache" ]; then
        want_prefix_cache=false
    fi

    echo -e "${YELLOW}Starting LMDeploy (TurboMind) server with model: $model${NC}"

    # §6.5 mapping: derive cache_max_entry_count (fraction of POST-WEIGHTS free
    # memory) from the uniform total-memory budget. Fail-closed: no mapping
    # inputs -> no launch. Computed BEFORE the reuse check so reuse can require
    # dial parity on the live cmdline (adversarial review 2026-08-12: a
    # pressure-sweep iteration invoked via `start` must never reuse the
    # previous budget's server while the driver labels data with the new one).
    local cache_fraction mapping_inputs=""
    if [ -n "${LMDEPLOY_CACHE_MAX_ENTRY_COUNT:-}" ]; then
        # Explicit operator override: a RECORDED deviation from the computed
        # iso-bytes mapping (captured in the serving-config JSON below). Like
        # LMDEPLOY_QUANT_POLICY, an explicit override always forces a fresh
        # start below -- never a silent reuse.
        cache_fraction="${LMDEPLOY_CACHE_MAX_ENTRY_COUNT}"
        warn "using explicit LMDEPLOY_CACHE_MAX_ENTRY_COUNT=$cache_fraction (recorded deviation from the computed §6.5 mapping)"
    else
        [ -n "${CAGE_MODEL_WEIGHTS_GIB:-}" ] || die "LMDeploy iso-BYTES mapping needs the model's weight footprint: set CAGE_MODEL_WEIGHTS_GIB=<GiB> (fp16/bf16 ~ 2 x params-in-B, e.g. 28 for Qwen3-14B) or set LMDEPLOY_CACHE_MAX_ENTRY_COUNT explicitly (recorded deviation). Refusing to guess: cache_max_entry_count meters POST-WEIGHTS free memory, so an unmapped dial breaks §6.5 budget parity."
        cache_fraction=$(cage_lmdeploy_cache_fraction "${CAGE_MODEL_WEIGHTS_GIB}") \
            || die "iso-BYTES mapping failed: malformed CAGE_MODEL_WEIGHTS_GIB/F, budget F=${VLLM_GPU_MEMORY_UTILIZATION} x total VRAM leaves no KV headroom after ${CAGE_MODEL_WEIGHTS_GIB} GiB of weights, or nvidia-smi is unavailable"
        mapping_inputs=$(printf '{"uniform_fraction": %s, "weights_gib": %s, "gpu_total_mib": %s}' \
            "${VLLM_GPU_MEMORY_UTILIZATION}" "${CAGE_MODEL_WEIGHTS_GIB}" "$(cage_gpu_total_mib)")
        echo "iso-BYTES mapping: F=${VLLM_GPU_MEMORY_UTILIZATION} (total) -> cache_max_entry_count=$cache_fraction (post-weights free), W=${CAGE_MODEL_WEIGHTS_GIB} GiB"
    fi

    # Check if already running
    local pid
    pid=$(get_lmdeploy_pid)
    if [ -n "$pid" ]; then
        local loaded_model cache_mode has_prefix_cache live_cmd dials_match
        loaded_model=$(get_loaded_model)
        cache_mode=$(get_server_prefix_cache_mode) || cache_mode="unknown"
        has_prefix_cache=false
        [ "$cache_mode" = "enabled" ] && has_prefix_cache=true

        # Reuse ONLY when no launch lever is requested AND the live cmdline
        # matches what this environment would launch (adversarial review
        # 2026-08-12): require the requested-backend flag (--backend turbomind;
        # actual selection was asserted at the original fresh start and is
        # re-verified at preflight -- a manual PyTorch-flag server forces a
        # restart here), the exact budget dial, and the uniform session-len.
        # Any mismatch, including a manually-started server, restarts.
        live_cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
        dials_match=true
        [[ "$live_cmd" == *"--backend turbomind"* ]] || dials_match=false
        [[ "$live_cmd" == *"--cache-max-entry-count $cache_fraction"* ]] || dials_match=false
        [[ "$live_cmd" == *"--session-len ${VLLM_MAX_MODEL_LEN}"* ]] || dials_match=false

        if [ "$loaded_model" = "$model" ] && [ "$has_prefix_cache" = "$want_prefix_cache" ] \
           && [ "$dials_match" = "true" ] \
           && [ -z "${LMDEPLOY_QUANT_POLICY:-}" ] && [ -z "${LMDEPLOY_CACHE_MAX_ENTRY_COUNT:-}" ]; then
            echo -e "${GREEN}✓ Server already running with correct model, cache mode, backend flag, and dials ($model)${NC}"
            return 0
        else
            echo -e "${RED}✗ Server state does not match requested model/cache/backend/dials${NC}"
            echo -e "${YELLOW}  Loaded model: $loaded_model | prefix cache: $has_prefix_cache | dials match: $dials_match${NC}"
            echo -e "${YELLOW}  Requested model: $model | prefix cache: $want_prefix_cache${NC}"
            echo -e "${YELLOW}  Stopping and restarting...${NC}"
            stop_server
            sleep 2
        fi
    fi

    local timestamp log_file
    timestamp=$(date +%Y%m%d_%H%M%S)
    log_file="$LOG_DIR/lmdeploy_${model//\//_}_${timestamp}.log"

    # Argv as an ARRAY so values are never word-split (vLLM-launcher rule).
    # --model-name pins the served id to the requested string so the
    # /v1/models readiness probe and the adapters compare like-for-like.
    local -a lmdeploy_args=( serve api_server "$model" )
    lmdeploy_args+=( --backend turbomind )
    lmdeploy_args+=( --server-port "$PORT" )
    lmdeploy_args+=( --model-name "$model" )
    # REQUIRED for the P7 backend gate (adversarial review 2026-08-12):
    # api_server defaults to --log-level ERROR, which suppresses both the
    # autoget_backend PyTorch-fallback warning and the engine-config echoes --
    # the exact lines assert_turbomind_selected greps. Without INFO the gate
    # is blind: strict mode false-dies on healthy TurboMind, warn mode misses
    # real fallbacks.
    lmdeploy_args+=( --log-level INFO )
    # Uniform context length (vLLM --max-model-len analogue).
    lmdeploy_args+=( --session-len "${VLLM_MAX_MODEL_LEN}" )
    lmdeploy_args+=( --cache-max-entry-count "$cache_fraction" )

    # LMDeploy prefix caching is default-OFF; the cache-on arm enables it.
    if [ "$want_prefix_cache" = "true" ]; then
        lmdeploy_args+=( --enable-prefix-caching )
    fi

    # Uniform eager lever: TurboMind documents no CUDA-graph/eager toggle --
    # the lever is N/A on this engine. Not fatal (serving stays uniform per
    # engine); recorded in the serving-config capture as eager_supported=false.
    if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
        warn "VLLM_ENFORCE_EAGER=1 requested but TurboMind has no eager/CUDA-graph toggle; lever is N/A on this engine (recorded deviation in serving-config capture)"
    fi

    # Optional server-side KV-cache quantization (TurboMind's compressed_cag
    # analogue; a DIFFERENT mechanism than vLLM/SGLang fp8 kv-dtype -- int KV
    # via quant policy), e.g.
    #   LMDEPLOY_QUANT_POLICY=8 ./scripts/2_serving/manage_lmdeploy_server.sh restart <model>
    if [ -n "${LMDEPLOY_QUANT_POLICY:-}" ]; then
        lmdeploy_args+=( --quant-policy "${LMDEPLOY_QUANT_POLICY}" )
        echo "KV quantization enabled: --quant-policy ${LMDEPLOY_QUANT_POLICY}"
    fi

    # Engine-version provenance (the LMDeploy pin is minted at S0; record what
    # actually served every start).
    local engine_version
    engine_version=$(python3 -c "import lmdeploy; print(getattr(lmdeploy, '__version__', 'unknown'))" 2>/dev/null || echo "unavailable")

    echo "Server args: lmdeploy ${lmdeploy_args[*]}  (lmdeploy=$engine_version)"

    # Per-(re)start serving-config capture (same contract as the vLLM launcher:
    # run_manifest.json is built once, so per-tree restarts must self-record).
    # Skipped silently when CAGE_RUN_ROOT is unset; never fatal to startup.
    if [ -n "${CAGE_RUN_ROOT:-}" ]; then
        local cfg_dir="$CAGE_RUN_ROOT/observability/serving_configs"
        local model_slug cfg_file
        model_slug=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]' | sed -E 's|.*/||; s|[^a-z0-9]+|-|g; s|^-+||; s|-+$||')
        cfg_file="$cfg_dir/$(date -u +%Y%m%dT%H%M%SZ)_lmdeploy_${model_slug}.json"
        mkdir -p "$cfg_dir" 2>/dev/null || true
        SC_ENGINE="lmdeploy-turbomind" \
        SC_VERSION="$engine_version" \
        SC_MODEL="$model" \
        SC_PORT="$PORT" \
        SC_PREFIX="$want_prefix_cache" \
        SC_MAX_LEN="${VLLM_MAX_MODEL_LEN}" \
        SC_BUDGET_F="${VLLM_GPU_MEMORY_UTILIZATION}" \
        SC_DIAL_FLAG="--cache-max-entry-count" \
        SC_DIAL_VALUE="$cache_fraction" \
        SC_MAPPING_INPUTS="$mapping_inputs" \
        SC_QUANT_POLICY="${LMDEPLOY_QUANT_POLICY:-}" \
        SC_EAGER="${VLLM_ENFORCE_EAGER:-0}" \
        SC_ARGS="lmdeploy ${lmdeploy_args[*]}" \
        SC_FILE="$cfg_file" \
        python3 - <<'PYEOF' || echo "  (serving-config capture failed; non-fatal)"
import datetime
import json
import os


def _maybe_json(raw):
    """Mapping inputs arrive as a JSON string; embed parsed, keep raw on error."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw  # keep the raw string rather than dropping provenance


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
    "budget_mapping_inputs": _maybe_json(os.environ.get("SC_MAPPING_INPUTS", "")),
    "quant_policy": os.environ.get("SC_QUANT_POLICY") or None,
    "enforce_eager_requested": os.environ.get("SC_EAGER") == "1",
    "eager_supported": False,  # TurboMind has no eager/CUDA-graph toggle
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

    echo "Starting LMDeploy server (logging to $log_file)..."
    nohup lmdeploy "${lmdeploy_args[@]}" > "$log_file" 2>&1 &

    local server_pid=$!
    printf '%s\n' "$server_pid" > "$PID_FILE"
    echo "Server PID: $server_pid (pidfile: $PID_FILE)"

    # Wait for readiness by polling the OpenAI surface the adapter actually
    # uses (/v1/models). TurboMind converts weights to its own format on first
    # load of a model, which can take well past 5 min on large models --
    # default 600s; override with LMDEPLOY_START_TIMEOUT.
    echo "Waiting for server to start..."
    local max_wait="${LMDEPLOY_START_TIMEOUT:-600}"
    local waited=0
    local loaded
    while [ "$waited" -lt "$max_wait" ]; do
        loaded=$(get_loaded_model)
        if [ "$loaded" = "$model" ]; then
            echo -e "${GREEN}✓ Server ready with model: $model${NC}"
            echo "  View logs: tail -f $log_file"
            # P7: refuse to hand over a server whose backend cannot be
            # confirmed as TurboMind (see assert_turbomind_selected).
            assert_turbomind_selected "$log_file"
            # Durable machine-readable record of the check outcome next to the
            # serving-config capture -- warn-mode calls itself a RECORDED
            # deviation, so record it (the capture above is written pre-launch
            # and cannot carry a post-launch result).
            if [ -n "${CAGE_RUN_ROOT:-}" ] && [ -n "${cfg_file:-}" ]; then
                printf '{"utc_timestamp": "%s", "backend_check_mode": "%s", "backend_check_result": "%s"}\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${CAGE_LMDEPLOY_BACKEND_CHECK:-strict}" "$TURBOMIND_CHECK_RESULT" \
                    > "${cfg_file%.json}_backend_check.json" 2>/dev/null \
                    || echo "  (backend-check record failed; non-fatal)"
            fi
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
    echo -e "${YELLOW}Stopping LMDeploy server...${NC}"

    pkill -f "lmdeploy serve api_server" 2>/dev/null || true
    sleep 2
    pkill -9 -f "lmdeploy serve api_server" 2>/dev/null || true

    # Belt-and-suspenders: kill any remaining LMDeploy/TurboMind process still
    # holding the GPU, but do NOT kill co-resident GPU users (metric models /
    # cage-stats). Same discipline as the vLLM launcher's orphan cleanup.
    local held
    held=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    for p in $held; do
        local cmd
        cmd=$(ps -p "$p" -o args= 2>/dev/null || true)
        case "$cmd" in
            *lmdeploy*|*turbomind*) kill -9 "$p" 2>/dev/null || true ;;
            *) [ -n "$cmd" ] && echo "  (left non-LMDeploy GPU process $p alive: ${cmd:0:60})" ;;
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
    pid=$(get_lmdeploy_pid)

    if [ -z "$pid" ]; then
        echo -e "${RED}✗ LMDeploy server is NOT running${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ LMDeploy server is running${NC}"
    echo "  PID: $pid"

    loaded_model=$(get_loaded_model)
    if [ -n "$loaded_model" ]; then
        echo "  Model: $loaded_model"
        echo "  Port: $PORT"
        echo "  Prefix cache: $(get_server_prefix_cache_mode)"
    else
        echo -e "${YELLOW}  Warning: Unable to query loaded model${NC}"
    fi
}

case "${1:-}" in
    start)
        if [ -z "${2:-}" ]; then
            echo "Usage: $0 start <model> [--no-prefix-cache]"
            echo "Example: CAGE_MODEL_WEIGHTS_GIB=28 $0 start Qwen/Qwen3-14B"
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
        echo "  start <model>   - Start LMDeploy (TurboMind) server with specified model"
        echo "  stop            - Stop LMDeploy server"
        echo "  restart <model> - Restart LMDeploy server with specified model"
        echo "  status          - Check LMDeploy server status"
        echo ""
        echo "Required for start (fail-closed §6.5 mapping):"
        echo "  CAGE_MODEL_WEIGHTS_GIB=<GiB>            weight footprint for the budget mapping"
        echo "  or LMDEPLOY_CACHE_MAX_ENTRY_COUNT=<f>   explicit dial (recorded deviation)"
        exit 1
        ;;
esac
