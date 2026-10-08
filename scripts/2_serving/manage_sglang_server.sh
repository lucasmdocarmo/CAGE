#!/bin/bash
# Order:     stage 2 — after 1_setup, before any 3_run driver; charter engine #2 launcher
# Objective: Start/stop/status the SGLang server under the SAME uniform regime as vLLM (iso-bytes dial mapping)
# Cloud:     both
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
# The mapping sets the dial; preflight gate (j) -- the CAGE-ISO-BYTES-GATE in
# scripts/checks/preflight_check.sh, run WITH the co-resident stack loaded --
# parses this launcher's startup log and asserts the REALIZED KV-pool bytes
# across engines (never assumed from the dial).
#
# [VERIFY-LIVE at S0]: every SGLang CLI flag below follows SGLang's documented
# server CLI, but none has been exercised by this codebase yet: SGLang is not
# installed on the dev box; the pod gets the pinned 0.5.10.post1
# (VLLM_COMPATIBILITY.md section 7, pinned 2026-09-26) through setup_runpod.sh
# step 3c. S0-3 proves this launcher end-to-end.
#
# Usage:
#   ./scripts/2_serving/manage_sglang_server.sh start <model> [--no-prefix-cache]
#   ./scripts/2_serving/manage_sglang_server.sh stop
#   ./scripts/2_serving/manage_sglang_server.sh restart <model> [--no-prefix-cache]
#   ./scripts/2_serving/manage_sglang_server.sh status
#
# Budget env contract (T2.1 — CacheBudgetPlanner wiring; charter P2 iso-bytes):
#   CAGE_SGLANG_MAX_TOTAL_TOKENS  positive integer; when set, the launch adds
#                                 `--max-total-tokens <N>` — SGLang's
#                                 token-capped budget dial, derived by
#                                 src/orchestration/cache_budget.py FROM the
#                                 byte budget (bytes / kv_bytes_per_token).
# The value is validated BEFORE any server is stopped or launched
# (cage_validate_sglang_budget_env in scripts/lib/_serving_config.sh);
# preflight gate (j) verifies the REALIZED pool bytes from the startup log.
#
# Tensor-parallel env contract (T3.1 — Wave-3 distributed serving; the
# 2026-08-27 audit verified NO launcher passed any TP flag):
#   CAGE_SGLANG_TP                positive integer; when set >= 2, the launch
#                                 adds `--tp-size <N>` — SGLang's documented
#                                 TP flag spelling (--tp is its alias), chosen
#                                 to match this launcher's long-form flag
#                                 convention; unproven against the pinned
#                                 SGLang 0.5.10.post1 (section 7)
#                                 [VERIFY-LIVE at Run-C-prime preflight].
#                                 Value 1 = flag OMITTED ENTIRELY (single-GPU
#                                 argv stays byte-identical to pre-T3.1).
# Validated BEFORE any server is stopped or launched
# (cage_validate_sglang_tp_env in scripts/lib/_serving_config.sh).
#
# Piecewise CUDA-graph lever (S0F-35, live H100 2026-10-07): SGLang 0.5.10.post1
# died in its piecewise CUDA-graph capture of Qwen3-14B ("FusedAddRMSNorm failed
# ... an illegal memory access"; its own log names the workaround):
#   CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH=1  adds `--disable-piecewise-cuda-graph`
#                                 (the piecewise capture only; full CUDA graphs
#                                 stay on, unlike VLLM_ENFORCE_EAGER). A reuse
#                                 dial and a serving-config field like the rest.
#
# Readiness (S0F-54, live H100 2026-10-08): /v1/models answers two seconds
# after uvicorn comes up, while SGLang's startup warm-up generation is still
# running; the cold-start flush posted in that window was refused (400
# "pending requests ... #queue-req: 0, #running-req: 0"). "Server ready"
# therefore also waits for SGLang's own ready line in the start log
# (SGLANG_READY_LINE below), the line it logs after the warm-up.
#
# Interpreter contract (pre-GO item 10, 2026-09-26): SGLang lives in its OWN
# venv (setup_runpod.sh step 3c, <repo>/sglang-env; its transformers and torch
# pins conflict with cage-env), so this launcher never depends on the caller's
# PATH: CAGE_SGLANG_PYTHON when set (an absolute path, or a bare command name
# resolved through command -v), else <repo>/sglang-env/bin/python3 when it
# exists, else the PATH python3 (a hand-activated environment). Whatever is
# resolved must be an executable FILE that imports sglang, checked in the
# start|restart gate BEFORE any server is touched (review 2026-09-26): a
# missing or half-installed venv fails closed here, never after the readiness
# wait. The resolved interpreter is printed at every start and recorded in the
# serving-config args.
# =============================================================================

set -euo pipefail

# Anchor paths to the repo root so logs ALWAYS land in <repo>/logs/sglang/,
# regardless of the caller's working directory (same rule as the vLLM launcher).
# CAGE_LOG_ROOT redirects the root (S0F-23, test hygiene); unset on a pod.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"
# Serving-uniformity source of truth (Option A) + §6.5 budget-mapping helpers.
# shellcheck source=scripts/lib/_serving_config.sh
source "$PROJECT_DIR/scripts/lib/_serving_config.sh"

# Interpreter resolution (see the header): explicit env, else the item-10 venv,
# else the PATH python3; a bare name resolves through command -v (an
# unresolvable one keeps the bare name and fails the gate below). Never a
# silent guess: the choice is printed at start.
SGLANG_PYTHON="${CAGE_SGLANG_PYTHON:-}"
if [ -z "$SGLANG_PYTHON" ]; then
    if [ -x "$PROJECT_DIR/sglang-env/bin/python3" ]; then
        SGLANG_PYTHON="$PROJECT_DIR/sglang-env/bin/python3"
    else
        SGLANG_PYTHON="python3"
    fi
fi
case "$SGLANG_PYTHON" in
    */*) ;;
    *) SGLANG_PYTHON="$(command -v "$SGLANG_PYTHON" 2>/dev/null || printf '%s' "$SGLANG_PYTHON")" ;;
esac

# Budget-knob refusal gate (T2.1): a malformed budget env must be refused
# BEFORE any server is touched — on `restart` it must not even tear down the
# healthy server it would fail to replace. Gated to launch commands only:
# `stop` must never be blocked by a bad budget (teardown discipline).
case "${1:-}" in
    start|restart)
        cage_validate_sglang_budget_env \
            || die "invalid KV-budget environment (see refusal above) -- not touching any server"
        # Tensor-parallel refusal gate (T3.1): same before-any-teardown
        # discipline — a malformed TP degree on `restart` must not tear down
        # the healthy server it would fail to replace.
        cage_validate_sglang_tp_env \
            || die "invalid tensor-parallel environment (see refusal above) -- not touching any server"
        # Interpreter gate (item 10; review 2026-09-26 MEDIUM 1): the resolved
        # interpreter must be an executable FILE (a directory passes -x alone)
        # that imports sglang; a missing or half-installed venv, or a PATH
        # python3 without the package, refuses here, before any teardown,
        # never after a 300 s readiness wait on a launch that cannot succeed.
        { [ -f "$SGLANG_PYTHON" ] && [ -x "$SGLANG_PYTHON" ]; } \
            || die "SGLang interpreter '$SGLANG_PYTHON' is not an executable file (CAGE_SGLANG_PYTHON, <repo>/sglang-env/bin/python3, or python3 on PATH) -- not touching any server (setup_runpod.sh step 3c creates sglang-env)"
        "$SGLANG_PYTHON" -c 'import sglang' >/dev/null 2>&1 \
            || die "SGLang is not importable from '$SGLANG_PYTHON' -- not touching any server (setup_runpod.sh step 3c installs it into sglang-env; CAGE_SGLANG_PYTHON selects another interpreter)"
        ;;
esac

PORT="${SGLANG_PORT:-30000}"   # SGLangAdapter's default api_base port
# SGLang's own ready line, logged after its startup warm-up generation
# (0.5.10.post1, sglang/srt/entrypoints/http_server.py); the start waits for
# it besides /v1/models (S0F-54). A version that drops the line times out
# loudly at SGLANG_START_TIMEOUT, never ready early.
SGLANG_READY_LINE="The server is fired up and ready to roll!"
LOG_DIR="${CAGE_LOG_ROOT:-$PROJECT_DIR/logs}/sglang"
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
        # uniform context length (adversarial review 2026-08-12) + /metrics
        # exposure (S0-23, checked below). The live
        # --kv-cache-dtype cannot be read back over the API, so if it is set
        # we force a restart rather than risk mislabeling the arm's data.
        live_cmd=$(ps -p "$pid" -o command= 2>/dev/null || true)
        dials_match=true
        [[ "$live_cmd" == *"--mem-fraction-static $mem_fraction"* ]] || dials_match=false
        [[ "$live_cmd" == *"--context-length ${VLLM_MAX_MODEL_LEN}"* ]] || dials_match=false
        # The token-cap budget knob (T2.1) is a dial too: reusing a server
        # launched under a DIFFERENT budget (or none) labels data with a pool
        # it never had. Requested => exact value must be live; absent => flag
        # must be absent from the live cmdline.
        if [ -n "${CAGE_SGLANG_MAX_TOTAL_TOKENS:-}" ]; then
            # Space-anchored: a token cap that is a decimal prefix of the live
            # value must not false-match (decade sweeps; verifier minor).
            [[ " $live_cmd " == *" --max-total-tokens ${CAGE_SGLANG_MAX_TOTAL_TOKENS} "* ]] || dials_match=false
        else
            [[ "$live_cmd" != *"--max-total-tokens"* ]] || dials_match=false
        fi
        # Tensor parallelism (T3.1) is a dial too: a TP=2 server must never
        # be reused for a TP=4 sweep point. Requested >= 2 => exact
        # space-anchored value must be live (2 must not prefix-match 24);
        # unset OR =1 => the flag must be absent from the live cmdline.
        if [ -n "${CAGE_SGLANG_TP:-}" ] && [ "${CAGE_SGLANG_TP}" != "1" ]; then
            [[ " $live_cmd " == *" --tp-size ${CAGE_SGLANG_TP} "* ]] || dials_match=false
        else
            [[ "$live_cmd" != *"--tp-size"* ]] || dials_match=false
        fi
        # The piecewise CUDA-graph lever (S0F-35) is a dial too: requested =>
        # the flag must be live; absent => it must be absent.
        if [ "${CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-0}" = "1" ]; then
            [[ " $live_cmd " == *" --disable-piecewise-cuda-graph "* ]] || dials_match=false
        else
            [[ "$live_cmd" != *"--disable-piecewise-cuda-graph"* ]] || dials_match=false
        fi
        # /metrics exposure (S0-23, ADR-0102) is a reuse requirement too: a
        # server started without --enable-metrics (pre-A14, or by hand) has
        # no running-requests gauge, so every window served from it would
        # record cold_start.verified == False. Restart rather than reuse.
        [[ " $live_cmd " == *" --enable-metrics "* ]] || dials_match=false

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
    # Prometheus exposure (S0-23 smoke item A14; ADR-0102 cold start per
    # window). The campaign driver's strict reset reads the running-requests
    # gauge sglang:num_running_reqs from GET /metrics before every
    # /flush_cache and persists the verdict as metrics.json['cold_start']
    # .verified (scripts/3_run/run_experiment.py COLD_START_RUNNING_GAUGE).
    # SGLang serves /metrics ONLY under --enable-metrics; without it every
    # SGLang window records verified: False with a WARNING, which fails
    # S0-23. Unconditional on purpose: never an env knob, never optional.
    # [VERIFY-LIVE at S0]: flag spelling per SGLang's documented server CLI.
    sglang_args+=( --enable-metrics )

    # Token-capped KV budget (T2.1; CacheBudgetPlanner SGLang knob, derived
    # FROM the byte budget). Passed in addition to the fraction dial — the
    # token cap is the binding budget; gate (j) verifies realized pool bytes.
    # Value was validated positive-integer at the top-of-script gate.
    if [ -n "${CAGE_SGLANG_MAX_TOTAL_TOKENS:-}" ]; then
        sglang_args+=( --max-total-tokens "${CAGE_SGLANG_MAX_TOTAL_TOKENS}" )
        echo "KV token budget enabled: --max-total-tokens ${CAGE_SGLANG_MAX_TOTAL_TOKENS}"
    fi

    # Tensor parallelism (T3.1; Wave-3 distributed stack). `--tp-size` is
    # SGLang's documented long-form TP flag (--tp is its alias; long form
    # matches this launcher's convention); the exact spelling is unproven
    # against the pinned 0.5.10.post1 [VERIFY-LIVE at Run-C-prime
    # preflight]. Value 1 OMITS the flag entirely so the single-GPU argv
    # stays byte-identical to pre-T3.1; validated positive-integer at the
    # top-of-script gate.
    if [ -n "${CAGE_SGLANG_TP:-}" ] && [ "${CAGE_SGLANG_TP}" != "1" ]; then
        sglang_args+=( --tp-size "${CAGE_SGLANG_TP}" )
        echo "Tensor parallelism enabled: --tp-size ${CAGE_SGLANG_TP}"
    fi

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

    # S0F-35 (live H100, 2026-10-07): the piecewise capture only (see the header).
    if [ "${CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-0}" = "1" ]; then
        sglang_args+=( --disable-piecewise-cuda-graph )
        echo "Piecewise CUDA graph OFF: --disable-piecewise-cuda-graph (S0F-35)"
    fi

    # Optional server-side KV-cache compression (compressed_cag analogue), e.g.
    #   SGLANG_KV_CACHE_DTYPE=fp8_e5m2 ./scripts/2_serving/manage_sglang_server.sh restart <model>
    if [ -n "${SGLANG_KV_CACHE_DTYPE:-}" ]; then
        sglang_args+=( --kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE}" )
        echo "KV-cache compression enabled: --kv-cache-dtype ${SGLANG_KV_CACHE_DTYPE}"
    fi

    # Engine-version provenance (the pin is section 7's 0.5.10.post1; record
    # what actually served every start).
    local engine_version
    engine_version=$("$SGLANG_PYTHON" -c "import sglang; print(getattr(sglang, '__version__', 'unknown'))" 2>/dev/null || echo "unavailable")

    echo "Interpreter: $SGLANG_PYTHON"
    echo "Server args: $SGLANG_PYTHON -m sglang.launch_server ${sglang_args[*]}  (sglang=$engine_version)"

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
        SC_MAX_TOTAL_TOKENS="${CAGE_SGLANG_MAX_TOTAL_TOKENS:-}" \
        SC_TENSOR_PARALLEL="${CAGE_SGLANG_TP:-}" \
        SC_KV_DTYPE="${SGLANG_KV_CACHE_DTYPE:-auto}" \
        SC_EAGER="${VLLM_ENFORCE_EAGER:-0}" \
        SC_PIECEWISE_OFF="${CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH:-0}" \
        SC_ARGS="$SGLANG_PYTHON -m sglang.launch_server ${sglang_args[*]}" \
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
    # T2.1 token-cap budget knob; null = knob not requested (fraction-only
    # launch), honest absence rather than a fabricated 0.
    "max_total_tokens": (
        int(os.environ["SC_MAX_TOTAL_TOKENS"])
        if os.environ.get("SC_MAX_TOTAL_TOKENS") else None
    ),
    # T3.1 tensor-parallel degree; null = knob not requested (engine-default
    # single-GPU launch), honest absence rather than a fabricated 1. Note an
    # EXPLICIT =1 request is recorded as 1 even though the flag is omitted:
    # the field captures what was requested, the args line what was passed.
    "tensor_parallel": (
        int(os.environ["SC_TENSOR_PARALLEL"])
        if os.environ.get("SC_TENSOR_PARALLEL") else None
    ),
    "kv_cache_dtype": os.environ.get("SC_KV_DTYPE") or "auto",
    "enforce_eager": os.environ.get("SC_EAGER") == "1",
    # S0F-35: true when the piecewise CUDA-graph capture was switched off.
    "disable_piecewise_cuda_graph": os.environ.get("SC_PIECEWISE_OFF") == "1",
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

    # S0F-24 (ADR-0142): a capture record never outlives the start it
    # described (removed before the spawn; see manage_vllm_server.sh).
    rm -f "$LOG_DIR/CURRENT.kvpool.json"

    echo "Starting SGLang server (logging to $log_file)..."
    nohup "$SGLANG_PYTHON" -m sglang.launch_server "${sglang_args[@]}" > "$log_file" 2>&1 &

    local server_pid=$!
    printf '%s\n' "$server_pid" > "$PID_FILE"
    echo "Server PID: $server_pid (pidfile: $PID_FILE)"

    # Wait for readiness: /v1/models must name the model (the surface the
    # adapter uses; it answers only once the model is served) AND the start
    # log must carry SGLang's own ready line. Live 2026-10-08 (S0F-54):
    # /v1/models answered two seconds after uvicorn came up while the startup
    # warm-up generation was still running, and the flush posted in that
    # window was refused ("pending requests ... #running-req: 0": the
    # in-flight forward sits in the scheduler's overlap result queue, which
    # no gauge shows). CUDA-graph capture can take minutes on smaller GPUs;
    # override with SGLANG_START_TIMEOUT.
    echo "Waiting for server to start..."
    local max_wait="${SGLANG_START_TIMEOUT:-300}"
    local waited=0
    local loaded=""
    while [ "$waited" -lt "$max_wait" ]; do
        loaded=$(get_loaded_model)
        if [ "$loaded" = "$model" ] && grep -qF -- "$SGLANG_READY_LINE" "$log_file" 2>/dev/null; then
            echo -e "${GREEN}✓ Server ready with model: $model${NC}"
            echo "  View logs: tail -f $log_file"
            # S0F-24 (ADR-0142): record the realized KV pool for gate (j);
            # the serving-config JSON (CAGE_RUN_ROOT only) gets it too.
            cage_kv_pool_capture --engine sglang --log "$log_file" --out "$LOG_DIR/CURRENT.kvpool.json" ${cfg_file:+--merge-into "$cfg_file"}
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done

    echo -e "\n${RED}✗ Server failed to start within ${max_wait}s${NC}"
    if [ "$loaded" = "$model" ]; then
        echo "  /v1/models named the model, but the start log never showed SGLang's ready line: $SGLANG_READY_LINE (S0F-54)"
    fi
    echo "Check logs: $log_file"
    # Fail closed (review 2026-10-08, LOW 1): a start that timed out must not
    # leave a half-ready server behind, or a later `start` would reuse it on
    # /v1/models and the dials alone, without the ready-line gate above.
    stop_server
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
    # S0F-24: no running engine, no current KV pool record.
    rm -f "$LOG_DIR/CURRENT.kvpool.json"

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
