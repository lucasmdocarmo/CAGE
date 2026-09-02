#!/bin/bash
# Order:     stage 2 — after 1_setup, before the pd-topology 3_run cells (DIST overlay); re-run per pd relaunch
# Objective: Start/stop/status the intra-node vLLM prefill/decode disaggregation stack (2 role instances + pd_proxy.py) under the uniform serving regime
# Cloud:     both
# =============================================================================
# vLLM Prefill/Decode (P/D) Disaggregation Launcher  (Wave-3 T3.2)
# =============================================================================
# Starts TWO vLLM servers on distinct ports — a PREFILL instance (KV producer
# role) and a DECODE instance (KV consumer role) — wired via
# --kv-transfer-config (NixlConnector), plus the stdlib pd_proxy.py front-end
# that realizes the 1-prefill/1-decode request pattern. The pilot "cluster"
# path (manage_vllm_cluster.py) is router-over-REPLICAS, NOT disaggregation;
# this launcher is the first real PD topology in the repo.
#
# Usage:
#   ./scripts/2_serving/manage_vllm_pd.sh start <model> [--no-prefix-cache]
#   ./scripts/2_serving/manage_vllm_pd.sh stop
#   ./scripts/2_serving/manage_vllm_pd.sh status
#   (no restart verb: `start` is self-cleaning — it tears down any prior pd
#    stack tolerantly before launching, so a relaunch boundary is one `start`.)
#
# Env contract (validated BEFORE any process is stopped or started; `stop` is
# NEVER gated — teardown discipline, Wave-1 style):
#   CAGE_KV_BUDGET_BYTES_PREFILL  REQUIRED positive int — the prefill pool's
#                                 byte budget (--kv-cache-memory-bytes on the
#                                 prefill instance). Charter §6.5: the P/D
#                                 split is an EXPLICIT recorded input; a
#                                 missing role budget is a refusal, never a
#                                 default.
#   CAGE_KV_BUDGET_BYTES_DECODE   REQUIRED positive int — decode pool budget.
#   CAGE_VLLM_TENSOR_PARALLEL     T3.1 knob, applied PER INSTANCE (both roles
#                                 get --tensor-parallel-size N when set >= 2;
#                                 value 1/unset omits the flag entirely).
#   CAGE_PD_PREFILL_PORT / CAGE_PD_DECODE_PORT / CAGE_PD_PROXY_PORT
#                                 defaults 8100 / 8200 / 8000 (mirrored by
#                                 run_campaign.py's telemetry-endpoint
#                                 emission — override BOTH sides together).
# REFUSED when set at start (ambiguity is a refusal, not a precedence rule):
#   CAGE_KV_BUDGET_BYTES / CAGE_VLLM_GPU_BLOCKS_OVERRIDE  (single-instance
#     budget knobs: which pool would they cap? — unset them for pd runs)
#   VLLM_KV_TRANSFER_CONFIG  (this launcher OWNS the per-role connector JSON)
#
# [VERIFY-LIVE at Run-C-prime preflight] — engine-facing names on this path
# that the pinned offline docs (docs/VLLM_COMPATIBILITY.md §0/§7/§8,
# docs/RUNBOOK.md §1.1 cd-act1) cannot fully confirm:
#   - the NixlConnector kv_role tokens "kv_producer"/"kv_consumer": the docs
#     record NixlConnector + "kv_role per instance", but the only kv_role
#     value recorded verbatim in-repo is LMCache's "kv_both" — the exact
#     per-role spellings are unproven until the pd preflight smoke;
#   - --kv-cache-memory-bytes per-role pool semantics under a connector
#     (§6.5 pool-sum realization; gate (j) closes on the startup logs);
#   - the whole NIXL data path (transfer, kv_transfer_params provenance):
#     until it passes live, run_experiment's campaign PD gate REFUSING
#     source-less transfer params is the CORRECT end-to-end outcome.
# These are surfaced in the start banner below, not just in comments.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"
# Uniform serving regime + the shared positive-int validator (Option A source
# of truth) — sourced HERE like every engine launcher (D1 doctrine).
# shellcheck source=scripts/lib/_serving_config.sh
source "$PROJECT_DIR/scripts/lib/_serving_config.sh"

PREFILL_PORT="${CAGE_PD_PREFILL_PORT:-8100}"
DECODE_PORT="${CAGE_PD_DECODE_PORT:-8200}"
PROXY_PORT="${CAGE_PD_PROXY_PORT:-8000}"

LOG_DIR="$PROJECT_DIR/logs/vllm"
PREFILL_PID_FILE="$LOG_DIR/vllm_pd_prefill.pid"
DECODE_PID_FILE="$LOG_DIR/vllm_pd_decode.pid"
PROXY_PID_FILE="$LOG_DIR/vllm_pd_proxy.pid"
mkdir -p "$LOG_DIR"

# Per-role connector JSON. Connector name NixlConnector per the pinned docs
# (VLLM_COMPATIBILITY.md §0 migration gate + §8); the kv_role tokens are
# [VERIFY-LIVE at Run-C-prime preflight] — see the header block.
PREFILL_KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
DECODE_KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --- validation (start only; stop/status never gated) ------------------------

cage_validate_pd_env() {
    # Both role budgets REQUIRED (§6.5: the split is explicit, never derived
    # here) — missing either refuses BEFORE any process is touched.
    local missing=""
    [ -n "${CAGE_KV_BUDGET_BYTES_PREFILL:-}" ] || missing="$missing CAGE_KV_BUDGET_BYTES_PREFILL"
    [ -n "${CAGE_KV_BUDGET_BYTES_DECODE:-}" ]  || missing="$missing CAGE_KV_BUDGET_BYTES_DECODE"
    if [ -n "$missing" ]; then
        printf '[cage] REFUSING pd launch: missing REQUIRED per-role byte budget(s):%s (charter §6.5: the P/D split is an explicit recorded input, never defaulted)\n' \
            "$missing" >&2
        return 1
    fi
    cage_require_positive_int CAGE_KV_BUDGET_BYTES_PREFILL "${CAGE_KV_BUDGET_BYTES_PREFILL}" || return 1
    cage_require_positive_int CAGE_KV_BUDGET_BYTES_DECODE  "${CAGE_KV_BUDGET_BYTES_DECODE}"  || return 1
    # Single-instance budget knobs alongside per-role budgets: two caps for
    # the same pools in different scopes — a refusal, never a precedence rule.
    if [ -n "${CAGE_KV_BUDGET_BYTES:-}" ] || [ -n "${CAGE_VLLM_GPU_BLOCKS_OVERRIDE:-}" ]; then
        printf '[cage] REFUSING pd launch: single-instance budget env (CAGE_KV_BUDGET_BYTES / CAGE_VLLM_GPU_BLOCKS_OVERRIDE) is set alongside the per-role pd budgets -- which pool would it cap? unset it for pd runs\n' >&2
        return 1
    fi
    # This launcher OWNS the per-role connector config; an ambient override
    # would silently rewire the roles.
    if [ -n "${VLLM_KV_TRANSFER_CONFIG:-}" ]; then
        printf '[cage] REFUSING pd launch: VLLM_KV_TRANSFER_CONFIG is set -- the pd launcher composes the per-role NixlConnector config itself; unset it\n' >&2
        return 1
    fi
    if [ "$PREFILL_PORT" = "$DECODE_PORT" ] || [ "$PREFILL_PORT" = "$PROXY_PORT" ] || [ "$DECODE_PORT" = "$PROXY_PORT" ]; then
        printf '[cage] REFUSING pd launch: ports collide (prefill=%s decode=%s proxy=%s) -- the pd stack needs three distinct ports\n' \
            "$PREFILL_PORT" "$DECODE_PORT" "$PROXY_PORT" >&2
        return 1
    fi
    return 0
}

case "${1:-}" in
    start)
        cage_validate_pd_env \
            || die "invalid P/D environment (see refusal above) -- not touching any server"
        # T3.1 TP gate, same before-any-teardown discipline; applied per role.
        cage_validate_vllm_tp_env \
            || die "invalid tensor-parallel environment (see refusal above) -- not touching any server"
        ;;
esac

# --- helpers -----------------------------------------------------------------

get_role_pid() {
    # $1 = pidfile. Alive AND still a vLLM process (PIDs get recycled).
    local fpid
    if [ -f "$1" ]; then
        fpid="$(cat "$1" 2>/dev/null || true)"
        if [ -n "$fpid" ] && ps -p "$fpid" -o command= 2>/dev/null | grep -q "vllm"; then
            echo "$fpid"
            return 0
        fi
    fi
    return 1
}

get_proxy_pid() {
    local fpid
    if [ -f "$PROXY_PID_FILE" ]; then
        fpid="$(cat "$PROXY_PID_FILE" 2>/dev/null || true)"
        if [ -n "$fpid" ] && ps -p "$fpid" -o command= 2>/dev/null | grep -q "pd_proxy"; then
            echo "$fpid"
            return 0
        fi
    fi
    return 1
}

wait_for_health() {
    # $1 = label, $2 = port, $3 = timeout seconds. Reuses the single-launcher
    # readiness idiom (curl /health poll); TIMEOUT=0 fails fast after the argv
    # echo, which is what the offline suite drives.
    local label="$1" port="$2" max_wait="$3" waited=0
    echo "Waiting for $label on port $port..."
    while [ "$waited" -lt "$max_wait" ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $label ready (port $port)${NC}"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    echo -e "\n${RED}✗ $label failed to become ready within ${max_wait}s${NC}"
    return 1
}

# --- start -------------------------------------------------------------------

compose_role_args() {
    # $1 = role (prefill|decode), $2 = port, $3 = kv-transfer JSON, $4 = byte
    # budget, $5 = cache flag. Emits the argv into the global array ROLE_ARGS.
    # Mirrors manage_vllm_server.sh's composition (uniform regime, array form
    # so the connector JSON is never word-split).
    local role="$1" port="$2" kv_cfg="$3" budget="$4" cache_flag="$5"
    ROLE_ARGS=( --port "$port" "$cache_flag" --trust-remote-code )
    if [ "${VLLM_DISABLE_LOG_REQUESTS:-1}" != "0" ]; then
        ROLE_ARGS+=( --disable-log-requests )
    fi
    # Per-role connector config — the disaggregation wiring itself.
    ROLE_ARGS+=( --kv-transfer-config "$kv_cfg" )
    ROLE_ARGS+=( --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" )
    ROLE_ARGS+=( --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" )
    # §6.5 per-role pool budget: the BINDING byte cap for this role's pool.
    # Per-role semantics under a connector are unproven offline
    # [VERIFY-LIVE at Run-C-prime preflight]; gate (j) closes on the logs.
    ROLE_ARGS+=( --kv-cache-memory-bytes "$budget" )
    # T3.1 TP knob per instance (value 1/unset omits the flag entirely —
    # byte-identical single-GPU argv discipline carried over).
    if [ -n "${CAGE_VLLM_TENSOR_PARALLEL:-}" ] && [ "${CAGE_VLLM_TENSOR_PARALLEL}" != "1" ]; then
        ROLE_ARGS+=( --tensor-parallel-size "${CAGE_VLLM_TENSOR_PARALLEL}" )
    fi
    if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
        ROLE_ARGS+=( --enforce-eager )
    fi
    ROLE_ARGS+=( --enable-prompt-tokens-details )
}

capture_role_config() {
    # Per-(re)start serving-config capture, one file PER ROLE (audit M7/COMP-5
    # discipline carried from manage_vllm_server.sh). Skipped silently when
    # CAGE_RUN_ROOT is unset; never fatal to startup.
    local role="$1" model="$2" port="$3" kv_cfg="$4" budget="$5" args_line="$6" prefix="$7"
    [ -n "${CAGE_RUN_ROOT:-}" ] || return 0
    local cfg_dir="$CAGE_RUN_ROOT/observability/serving_configs"
    local model_slug cfg_file
    model_slug=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]' | sed -E 's|.*/||; s|[^a-z0-9]+|-|g; s|^-+||; s|-+$||')
    cfg_file="$cfg_dir/$(date -u +%Y%m%dT%H%M%SZ)_${model_slug}_pd-${role}.json"
    mkdir -p "$cfg_dir" 2>/dev/null || true
    SC_ROLE="$role" \
    SC_MODEL="$model" \
    SC_PORT="$port" \
    SC_KV_TRANSFER="$kv_cfg" \
    SC_KV_BUDGET_BYTES="$budget" \
    SC_TENSOR_PARALLEL="${CAGE_VLLM_TENSOR_PARALLEL:-}" \
    SC_PREFIX="$prefix" \
    SC_MAX_LEN="${VLLM_MAX_MODEL_LEN:-4096}" \
    SC_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
    SC_EAGER="${VLLM_ENFORCE_EAGER:-0}" \
    SC_ARGS="$args_line" \
    SC_FILE="$cfg_file" \
    python3 - <<'PYEOF' || echo "  (pd serving-config capture failed; non-fatal)"
import datetime
import json
import os


def _maybe_json(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw  # keep the raw string rather than dropping provenance


cfg = {
    "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "topology": "pd",
    "role": os.environ["SC_ROLE"],
    "model": os.environ["SC_MODEL"],
    "port": int(os.environ["SC_PORT"]),
    "kv_transfer_config": _maybe_json(os.environ.get("SC_KV_TRANSFER", "")),
    # §6.5 per-role budget: REQUIRED at launch, so always an int here.
    "kv_budget_bytes": int(os.environ["SC_KV_BUDGET_BYTES"]),
    # T3.1 semantics: null = knob not requested; explicit =1 records 1 while
    # the args line shows the omitted flag (requested vs passed).
    "tensor_parallel": (
        int(os.environ["SC_TENSOR_PARALLEL"])
        if os.environ.get("SC_TENSOR_PARALLEL") else None
    ),
    "enable_prefix_caching": os.environ.get("SC_PREFIX") == "true",
    "max_model_len": int(os.environ.get("SC_MAX_LEN", "4096")),
    "gpu_memory_utilization": float(os.environ.get("SC_MEM_UTIL", "0.90")),
    "enforce_eager": os.environ.get("SC_EAGER") == "1",
    "args": os.environ["SC_ARGS"],
}
with open(os.environ["SC_FILE"], "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")
PYEOF
    echo "  Serving config captured: $cfg_file"
}

start_stack() {
    local model="$1"
    local cache_flag="--enable-prefix-caching"
    local want_prefix_cache=true
    if [ "${2:-}" = "--no-prefix-cache" ]; then
        cache_flag="--no-enable-prefix-caching"
        want_prefix_cache=false
    fi

    echo -e "${YELLOW}Starting vLLM P/D disaggregation stack with model: $model${NC}"
    echo "[cage] PD start banner:"
    echo "[cage]   connector=NixlConnector  roles: prefill=kv_producer decode=kv_consumer"
    echo "[cage]   [VERIFY-LIVE at Run-C-prime preflight] exact kv_role tokens, per-role"
    echo "[cage]   --kv-cache-memory-bytes pool semantics under the connector, and the"
    echo "[cage]   whole NIXL data path are UNPROVEN offline; until the pd preflight"
    echo "[cage]   smoke passes, the campaign PD provenance gate refusing source-less"
    echo "[cage]   kv_transfer_params is the CORRECT outcome, not a bug."

    # start is self-cleaning: a stale pd stack (or a lone single-instance
    # server on these ports) must never be reused under new dials — the
    # per-role budget IS a dial. Tolerant teardown, then a fresh launch.
    stop_stack

    local timestamp prefill_log decode_log proxy_log
    timestamp=$(date +%Y%m%d_%H%M%S)
    prefill_log="$LOG_DIR/vllm_pd_prefill_${model//\//_}_${timestamp}.log"
    decode_log="$LOG_DIR/vllm_pd_decode_${model//\//_}_${timestamp}.log"
    proxy_log="$LOG_DIR/pd_proxy_${timestamp}.log"

    # Gate-(j) pin form (preflight_check.sh accepts engine[:role]=path):
    # copy-paste this into the preflight env to pin BOTH startup logs.
    echo "[cage] gate-(j) log pin: CAGE_ISO_BYTES_LOGS=\"vllm:prefill=${prefill_log},vllm:decode=${decode_log}\""

    export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"

    local -a ROLE_ARGS

    compose_role_args prefill "$PREFILL_PORT" "$PREFILL_KV_TRANSFER_CONFIG" \
        "$CAGE_KV_BUDGET_BYTES_PREFILL" "$cache_flag"
    local -a prefill_args=( "${ROLE_ARGS[@]}" )
    echo "Server args [prefill]: vllm serve $model ${prefill_args[*]}"
    capture_role_config prefill "$model" "$PREFILL_PORT" "$PREFILL_KV_TRANSFER_CONFIG" \
        "$CAGE_KV_BUDGET_BYTES_PREFILL" "vllm serve $model ${prefill_args[*]}" "$want_prefix_cache"
    echo "Starting prefill instance (logging to $prefill_log)..."
    nohup vllm serve "$model" "${prefill_args[@]}" > "$prefill_log" 2>&1 &
    printf '%s\n' "$!" > "$PREFILL_PID_FILE"
    echo "Prefill PID: $(cat "$PREFILL_PID_FILE") (pidfile: $PREFILL_PID_FILE)"

    compose_role_args decode "$DECODE_PORT" "$DECODE_KV_TRANSFER_CONFIG" \
        "$CAGE_KV_BUDGET_BYTES_DECODE" "$cache_flag"
    local -a decode_args=( "${ROLE_ARGS[@]}" )
    echo "Server args [decode]: vllm serve $model ${decode_args[*]}"
    capture_role_config decode "$model" "$DECODE_PORT" "$DECODE_KV_TRANSFER_CONFIG" \
        "$CAGE_KV_BUDGET_BYTES_DECODE" "vllm serve $model ${decode_args[*]}" "$want_prefix_cache"
    echo "Starting decode instance (logging to $decode_log)..."
    nohup vllm serve "$model" "${decode_args[@]}" > "$decode_log" 2>&1 &
    printf '%s\n' "$!" > "$DECODE_PID_FILE"
    echo "Decode PID: $(cat "$DECODE_PID_FILE") (pidfile: $DECODE_PID_FILE)"

    # Readiness: BOTH instances, then the proxy (whose /health requires both
    # upstreams — a proxy over a half-up pair must never report ready).
    local max_wait="${VLLM_START_TIMEOUT:-300}"
    wait_for_health "prefill instance" "$PREFILL_PORT" "$max_wait" || return 1
    wait_for_health "decode instance" "$DECODE_PORT" "$max_wait" || return 1

    echo "Starting pd_proxy on port $PROXY_PORT (logging to $proxy_log)..."
    nohup python3 "$SCRIPT_DIR/pd_proxy.py" \
        --port "$PROXY_PORT" \
        --prefill-url "http://localhost:${PREFILL_PORT}" \
        --decode-url "http://localhost:${DECODE_PORT}" \
        > "$proxy_log" 2>&1 &
    printf '%s\n' "$!" > "$PROXY_PID_FILE"
    echo "Proxy PID: $(cat "$PROXY_PID_FILE") (pidfile: $PROXY_PID_FILE)"
    wait_for_health "pd proxy" "$PROXY_PORT" "${PD_PROXY_START_TIMEOUT:-60}" || return 1

    echo -e "${GREEN}✓ P/D stack up${NC}: proxy :$PROXY_PORT -> prefill :$PREFILL_PORT + decode :$DECODE_PORT"
    echo "  Logs: prefill=$prefill_log decode=$decode_log proxy=$proxy_log"
    return 0
}

# --- stop (tolerant; NEVER blocked by validation) ----------------------------

stop_stack() {
    echo -e "${YELLOW}Stopping vLLM P/D stack (proxy + both instances)...${NC}"

    # Proxy first: no client may reach a half-dismantled pair.
    local ppid
    ppid=$(get_proxy_pid || true)
    [ -n "$ppid" ] && kill "$ppid" 2>/dev/null || true
    pkill -f "pd_proxy.py" 2>/dev/null || true
    rm -f "$PROXY_PID_FILE"

    # Role instances by pidfile, then the same belt-and-suspenders sweep as
    # manage_vllm_server.sh (vLLM v1 spawns EngineCore workers "vllm serve"
    # does not match; orphaning one keeps the GPU held).
    local rpid
    for pf in "$PREFILL_PID_FILE" "$DECODE_PID_FILE"; do
        rpid=$(get_role_pid "$pf" || true)
        [ -n "$rpid" ] && kill "$rpid" 2>/dev/null || true
        rm -f "$pf"
    done
    pkill -f "vllm serve"          2>/dev/null || true
    pkill -f "VLLM::EngineCore"    2>/dev/null || true
    pkill -f "vllm.v1.engine.core" 2>/dev/null || true
    sleep 2
    pkill -9 -f "vllm serve"          2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore"    2>/dev/null || true
    pkill -9 -f "vllm.v1.engine.core" 2>/dev/null || true

    # Match-before-kill GPU sweep (leave co-resident metric models alive).
    local held
    held=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)
    for p in $held; do
        local cmd
        cmd=$(ps -p "$p" -o args= 2>/dev/null || true)
        case "$cmd" in
            *vllm*|*VLLM*|*EngineCore*) kill -9 "$p" 2>/dev/null || true ;;
            *) [ -n "$cmd" ] && echo "  (left non-vLLM GPU process $p alive: ${cmd:0:60})" ;;
        esac
    done

    echo -e "${GREEN}✓ P/D stack stopped${NC}"
}

# --- status ------------------------------------------------------------------

status_stack() {
    local ok=0 pid
    for spec in "prefill:$PREFILL_PID_FILE:$PREFILL_PORT" "decode:$DECODE_PID_FILE:$DECODE_PORT"; do
        local role="${spec%%:*}" rest="${spec#*:}"
        local pf="${rest%%:*}" port="${rest##*:}"
        pid=$(get_role_pid "$pf" || true)
        if [ -n "$pid" ]; then
            echo -e "${GREEN}✓ $role instance running${NC} (PID $pid, port $port)"
        else
            echo -e "${RED}✗ $role instance NOT running${NC} (port $port)"
            ok=1
        fi
    done
    pid=$(get_proxy_pid || true)
    if [ -n "$pid" ]; then
        echo -e "${GREEN}✓ pd proxy running${NC} (PID $pid, port $PROXY_PORT)"
    else
        echo -e "${RED}✗ pd proxy NOT running${NC} (port $PROXY_PORT)"
        ok=1
    fi
    # Honest pending state, never a PASS: process liveness above says nothing
    # about the transfer path.
    echo "PD data path: PENDING [VERIFY-LIVE at Run-C-prime preflight] — NIXL transfer + kv_transfer_params provenance unproven offline"
    return "$ok"
}

case "${1:-}" in
    start)
        if [ -z "${2:-}" ]; then
            echo "Usage: $0 start <model> [--no-prefix-cache]"
            exit 1
        fi
        start_stack "$2" "${3:-}"
        ;;
    stop)
        stop_stack
        ;;
    status)
        status_stack
        ;;
    *)
        echo "Usage: $0 {start|stop|status} [model]"
        echo ""
        echo "Commands:"
        echo "  start <model>   - Start prefill+decode vLLM instances + pd proxy (self-cleaning)"
        echo "  stop            - Stop proxy + both instances (tolerant, never gated)"
        echo "  status          - Report per-component state (data path stays PENDING until live-verified)"
        exit 1
        ;;
esac
