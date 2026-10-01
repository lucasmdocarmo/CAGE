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
#   CAGE_PD_NIXL_PORT_PREFILL / CAGE_PD_NIXL_PORT_DECODE
#                                 NIXL handshake side-channel port per role,
#                                 defaults 5600 / 5601, passed to each role as
#                                 VLLM_NIXL_SIDE_CHANNEL_PORT in its child env
#                                 (ADR-0128, S0F-13 layer 3: both roles on one
#                                 host defaulted to 5600 at S0 and the prefill's
#                                 listener died EADDRINUSE). Equal values, a
#                                 value equal to an HTTP port, or an ambient
#                                 VLLM_NIXL_SIDE_CHANNEL_PORT are refused.
#   CAGE_PD_PYTHON                the interpreter the nixl import gate probes
#                                 (default python3: the activated cage-env that
#                                 runs `vllm serve`); the offline suite points it
#                                 at a stub. The gate refuses, before the
#                                 self-cleaning stop, when nixl._api /
#                                 nixl._bindings do not import (layer 1) or a
#                                 top-level nixl_ep is findable but broken
#                                 (layer 2: a nixl 1.x wheel built for another
#                                 torch).
#   CAGE_PD_PREFILL_GPUS / CAGE_PD_DECODE_GPUS
#                                 OPTIONAL per-role CUDA_VISIBLE_DEVICES pins
#                                 (comma-separated GPU indices, e.g. "0" and
#                                 "1", or "0,1" and "2,3" under TP). Set BOTH
#                                 to pairwise-disjoint sets or NEITHER; one
#                                 pinned role or an overlap is a refusal.
#                                 Unset = both roles share ONE GPU, which
#                                 drives the per-instance memory dial:
#   VLLM_GPU_MEMORY_UTILIZATION   per-instance --gpu-memory-utilization
#                                 OVERRIDE. Default 0.45 on a shared GPU
#                                 (SHARED_GPU_MEM_UTIL, backlog A1 / S0-9 /
#                                 S0-20; vLLM's startup check refuses the
#                                 second instance at 0.90) and 0.90 with
#                                 distinct pins; an explicit value above 0.50
#                                 on a shared GPU is REFUSED. The decision is
#                                 printed at start and captured per role.
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
# Capture the CALLER's memory-utilization request BEFORE the serving config
# is sourced: that lib exports VLLM_GPU_MEMORY_UTILIZATION=0.90 whenever it is
# unset, and the shared-GPU rule below must not mistake the lib's default for
# an explicit 0.90 (which it would have to refuse on a shared GPU).
PD_MEM_UTIL_REQUESTED="${VLLM_GPU_MEMORY_UTILIZATION:-}"
# Uniform serving regime + the shared positive-int validator (Option A source
# of truth) — sourced HERE like every engine launcher (D1 doctrine).
# shellcheck source=scripts/lib/_serving_config.sh
source "$PROJECT_DIR/scripts/lib/_serving_config.sh"

PREFILL_PORT="${CAGE_PD_PREFILL_PORT:-8100}"
DECODE_PORT="${CAGE_PD_DECODE_PORT:-8200}"
PROXY_PORT="${CAGE_PD_PROXY_PORT:-8000}"
# ADR-0128 (S0F-13 layer 3): one NIXL side-channel port PER ROLE. vLLM's
# NixlConnector binds tcp://<host>:<VLLM_NIXL_SIDE_CHANNEL_PORT> for its KV
# handshake; the default is 5600 for every worker, so two roles on one host
# collide (S0: the prefill's listener died EADDRINUSE and never served
# /health). vLLM's own 1P1D harness uses 5600 and 5601.
NIXL_PORT_PREFILL="${CAGE_PD_NIXL_PORT_PREFILL:-5600}"
NIXL_PORT_DECODE="${CAGE_PD_NIXL_PORT_DECODE:-5601}"
# The interpreter that runs `vllm serve` (the activated cage-env); the nixl
# import gate probes it. Override only for the offline suite.
PD_PYTHON="${CAGE_PD_PYTHON:-python3}"
# Must equal setup_runpod.sh NIXL_VERSION (the refusal messages name the fix).
NIXL_PIN="0.9.0"

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

# --- shared-GPU memory-utilization rule (backlog Tier A item A1) -------------
# vLLM 0.19.1's startup check requests --gpu-memory-utilization of the device
# unconditionally, so two role instances on ONE GPU at the 0.90 uniform
# operating point cannot both start: the second one is refused and the S0
# proofs S0-9 (cluster lifecycle) and S0-20 (pd preflight smoke) cannot begin.
# Rule (mirrors manage_vllm_cluster.py, backlog A1, S0 rows S0-9 and S0-20):
#   shared   = no distinct per-role CUDA_VISIBLE_DEVICES pins -> per-instance
#              default SHARED_GPU_MEM_UTIL; an explicit value above
#              SHARED_GPU_MEM_UTIL_CEILING is REFUSED (the fix is named).
#   distinct = both roles pinned to disjoint GPU sets -> DISTINCT_GPU_MEM_UTIL
#              (the Option-A operating point) stands; the override is honored.
# VLLM_GPU_MEMORY_UTILIZATION remains the override on both paths.
SHARED_GPU_MEM_UTIL=0.45
SHARED_GPU_MEM_UTIL_CEILING=0.50
DISTINCT_GPU_MEM_UTIL=0.90

cage_pd_gpu_list_ok() {
    # $1 = env name, $2 = value: a comma-separated list of GPU indices with no
    # empty entries and no leading zeros ("00" and "0" would name one device
    # under two spellings and defeat the disjointness check).
    local name="$1" value="$2" item
    case "$value" in
        ''|*[!0-9,]*|,*|*,|*,,*)
            printf '[cage] REFUSING pd launch: %s=%s is not a comma-separated list of GPU indices (set CAGE_PD_PREFILL_GPUS and CAGE_PD_DECODE_GPUS to disjoint sets, or neither)\n' \
                "$name" "${2:-<empty>}" >&2
            return 1
            ;;
    esac
    local IFS=','
    for item in $value; do
        case "$item" in
            0?*)
                printf '[cage] REFUSING pd launch: %s=%s has a leading-zero GPU index (%s); write the plain index (CAGE_PD_PREFILL_GPUS / CAGE_PD_DECODE_GPUS)\n' \
                    "$name" "$value" "$item" >&2
                return 1
                ;;
        esac
    done
    return 0
}

cage_pd_gpu_lists_overlap() {
    # $1, $2 = comma-separated GPU index lists. 0 iff they share an index.
    local a b
    local IFS=','
    for a in $1; do
        for b in $2; do
            [ "$a" = "$b" ] && return 0
        done
    done
    return 1
}

cage_resolve_pd_gpu_share() {
    # Resolves the rule into globals consumed by compose/capture/launch:
    #   PD_GPU_SHARE        shared | distinct
    #   PD_MEM_UTIL         the per-instance --gpu-memory-utilization value
    #   PD_MEM_UTIL_SOURCE  default | explicit
    #   PD_PREFILL_CUDA / PD_DECODE_CUDA  per-role CUDA_VISIBLE_DEVICES ('' =
    #                       unpinned; the pin rides the child env only)
    # Returns 1 (after a REFUSING line) on any ambiguity: one pinned role,
    # overlapping or malformed pins, or a shared explicit value the vLLM
    # startup check cannot honor.
    PD_PREFILL_CUDA="${CAGE_PD_PREFILL_GPUS:-}"
    PD_DECODE_CUDA="${CAGE_PD_DECODE_GPUS:-}"
    if [ -n "$PD_PREFILL_CUDA" ] || [ -n "$PD_DECODE_CUDA" ]; then
        if [ -z "$PD_PREFILL_CUDA" ] || [ -z "$PD_DECODE_CUDA" ]; then
            printf '[cage] REFUSING pd launch: only one role is pinned (CAGE_PD_PREFILL_GPUS=%s CAGE_PD_DECODE_GPUS=%s); pin BOTH roles to disjoint GPU sets or NEITHER (shared GPU)\n' \
                "${PD_PREFILL_CUDA:-<unset>}" "${PD_DECODE_CUDA:-<unset>}" >&2
            return 1
        fi
        cage_pd_gpu_list_ok CAGE_PD_PREFILL_GPUS "$PD_PREFILL_CUDA" || return 1
        cage_pd_gpu_list_ok CAGE_PD_DECODE_GPUS "$PD_DECODE_CUDA" || return 1
        if cage_pd_gpu_lists_overlap "$PD_PREFILL_CUDA" "$PD_DECODE_CUDA"; then
            printf '[cage] REFUSING pd launch: CAGE_PD_PREFILL_GPUS=%s and CAGE_PD_DECODE_GPUS=%s share a GPU index; distinct pins must be pairwise disjoint (or unset both for the shared-GPU regime)\n' \
                "$PD_PREFILL_CUDA" "$PD_DECODE_CUDA" >&2
            return 1
        fi
        PD_GPU_SHARE=distinct
    else
        PD_GPU_SHARE=shared
    fi

    local requested="${PD_MEM_UTIL_REQUESTED:-}"
    if [ -z "$requested" ]; then
        PD_MEM_UTIL_SOURCE=default
        if [ "$PD_GPU_SHARE" = shared ]; then
            PD_MEM_UTIL="$SHARED_GPU_MEM_UTIL"
        else
            PD_MEM_UTIL="$DISTINCT_GPU_MEM_UTIL"
        fi
        return 0
    fi
    case "$requested" in
        *[!0-9.]*|.|*.*.*)
            printf '[cage] REFUSING pd launch: VLLM_GPU_MEMORY_UTILIZATION=%s is not a decimal fraction\n' \
                "$requested" >&2
            return 1
            ;;
    esac
    # LC_ALL=C: a comma-radix locale would misread the fraction.
    if ! LC_ALL=C awk -v v="$requested" 'BEGIN { exit !(v > 0 && v <= 1) }'; then
        printf '[cage] REFUSING pd launch: VLLM_GPU_MEMORY_UTILIZATION=%s must be a fraction in (0, 1]\n' \
            "$requested" >&2
        return 1
    fi
    if [ "$PD_GPU_SHARE" = shared ] \
        && LC_ALL=C awk -v v="$requested" -v c="$SHARED_GPU_MEM_UTIL_CEILING" 'BEGIN { exit !(v > c) }'; then
        printf '[cage] REFUSING pd launch: prefill and decode share one GPU and VLLM_GPU_MEMORY_UTILIZATION=%s asks each instance for --gpu-memory-utilization %s of the device; the vLLM startup check requests that fraction unconditionally, so the second instance cannot start. Fix: lower VLLM_GPU_MEMORY_UTILIZATION to at most %s (default %s), or pin the roles to distinct GPUs with CAGE_PD_PREFILL_GPUS / CAGE_PD_DECODE_GPUS (e.g. 0 and 1)\n' \
            "$requested" "$requested" "$SHARED_GPU_MEM_UTIL_CEILING" "$SHARED_GPU_MEM_UTIL" >&2
        return 1
    fi
    PD_MEM_UTIL="$requested"
    PD_MEM_UTIL_SOURCE=explicit
    return 0
}

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
    # ADR-0128 (S0F-13 layer 3): the NIXL side-channel ports, one per role,
    # distinct from each other and from the three HTTP ports; the launcher
    # owns VLLM_NIXL_SIDE_CHANNEL_PORT like the connector JSON.
    cage_require_positive_int CAGE_PD_NIXL_PORT_PREFILL "$NIXL_PORT_PREFILL" || return 1
    cage_require_positive_int CAGE_PD_NIXL_PORT_DECODE  "$NIXL_PORT_DECODE"  || return 1
    if [ "$NIXL_PORT_PREFILL" = "$NIXL_PORT_DECODE" ]; then
        printf '[cage] REFUSING pd launch: NIXL side-channel ports collide (prefill=%s decode=%s): both roles on one host would bind the same handshake listener and the second dies EADDRINUSE (S0F-13 layer 3, S0 2026-09-30); set CAGE_PD_NIXL_PORT_PREFILL / CAGE_PD_NIXL_PORT_DECODE to distinct values\n' \
            "$NIXL_PORT_PREFILL" "$NIXL_PORT_DECODE" >&2
        return 1
    fi
    local _p
    for _p in "$NIXL_PORT_PREFILL" "$NIXL_PORT_DECODE"; do
        if [ "$_p" = "$PREFILL_PORT" ] || [ "$_p" = "$DECODE_PORT" ] || [ "$_p" = "$PROXY_PORT" ]; then
            printf '[cage] REFUSING pd launch: NIXL side-channel port %s equals an HTTP port (prefill=%s decode=%s proxy=%s); pick a free port for CAGE_PD_NIXL_PORT_PREFILL / CAGE_PD_NIXL_PORT_DECODE\n' \
                "$_p" "$PREFILL_PORT" "$DECODE_PORT" "$PROXY_PORT" >&2
            return 1
        fi
    done
    if [ -n "${VLLM_NIXL_SIDE_CHANNEL_PORT:-}" ]; then
        printf '[cage] REFUSING pd launch: VLLM_NIXL_SIDE_CHANNEL_PORT is set in the shell -- the pd launcher assigns one side-channel port per role (CAGE_PD_NIXL_PORT_PREFILL / CAGE_PD_NIXL_PORT_DECODE); unset it\n' >&2
        return 1
    fi
    # Backlog A1 (S0-9 / S0-20): shared-vs-distinct GPU decision, resolved
    # here so a refusal fires BEFORE the self-cleaning teardown.
    cage_resolve_pd_gpu_share || return 1
    return 0
}

cage_pd_nixl_import_gate() {
    # ADR-0128 (S0F-13 layers 1 and 2), start only, BEFORE the self-cleaning
    # stop: the interpreter that runs `vllm serve` must import what the
    # NixlConnector imports, and a top-level nixl_ep that is findable but
    # broken (the nixl 1.x wheels build it for torch 2.11+; vLLM imports it on
    # sight) is the exact S0 layer-2 crash. Probe exit codes: 0 ok, 2 nixl
    # missing, 3 nixl_ep broken; the probe's own message is shown.
    local probe out rc
    # (a heredoc inside $(...) does not parse on bash 3.2, the macOS test host)
    probe='import importlib.util
import sys
try:
    import nixl._api, nixl._bindings  # noqa: F401  (what the vLLM NixlConnector imports)
except Exception as exc:
    print(f"nixl: {type(exc).__name__}: {exc}")
    sys.exit(2)
if importlib.util.find_spec("nixl_ep") is not None:
    try:
        import nixl_ep  # noqa: F401
    except Exception as exc:
        print(f"nixl_ep: {type(exc).__name__}: {exc}")
        sys.exit(3)
print("ok")
'
    out="$(printf '%s' "$probe" | "$PD_PYTHON" - 2>&1)" && rc=0 || rc=$?
    case "$rc" in
        0)
            echo "[cage] nixl import gate: ok ($PD_PYTHON imports nixl._api, nixl._bindings)"
            return 0
            ;;
        2)
            printf '[cage] REFUSING pd launch: %s cannot import nixl._api, nixl._bindings (%s) -- the NixlConnector cannot load; install nixl==%s nixl-cu12==%s beside vLLM (setup_runpod.sh step 2, S0F-13 layer 1)\n' \
                "$PD_PYTHON" "$out" "$NIXL_PIN" "$NIXL_PIN" >&2
            return 1
            ;;
        3)
            printf '[cage] REFUSING pd launch: a top-level nixl_ep is installed but fails to import (%s) -- a nixl 1.x wheel built for another torch, the S0F-13 layer-2 crash; reinstall nixl==%s nixl-cu12==%s, whose wheels ship no nixl_ep\n' \
                "$out" "$NIXL_PIN" "$NIXL_PIN" >&2
            return 1
            ;;
        *)
            printf '[cage] REFUSING pd launch: nixl import probe failed under %s (exit %s): %s\n' \
                "$PD_PYTHON" "$rc" "$out" >&2
            return 1
            ;;
    esac
}

case "${1:-}" in
    start)
        cage_validate_pd_env \
            || die "invalid P/D environment (see refusal above) -- not touching any server"
        # T3.1 TP gate, same before-any-teardown discipline; applied per role.
        cage_validate_vllm_tp_env \
            || die "invalid tensor-parallel environment (see refusal above) -- not touching any server"
        # ADR-0128: the connector's imports, before any process is touched.
        cage_pd_nixl_import_gate \
            || die "nixl import gate failed (see refusal above) -- not touching any server"
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

role_exited() {
    # $1 = role label, $2 = pidfile, $3 = log. 0 iff the role's API-server pid
    # is gone (bash reaps a dead background child, so kill -0 fails); prints the
    # log tail so the S0 failure class is read at once, not after the budget.
    local pid
    pid="$(cat "$2" 2>/dev/null || true)"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo -e "\n${RED}✗ $1 instance exited before serving /health (pid ${pid:-none}); last 20 log lines ($3):${NC}"
        tail -n 20 "$3" 2>/dev/null || true
        return 0
    fi
    return 1
}

wait_for_roles() {
    # $1 = timeout seconds, $2 = prefill log, $3 = decode log. BOTH roles in
    # ONE loop (ADR-0128 part 4): they become ready in either order, and a role
    # whose process is gone fails the start at once. At S0, attempts 1 and 2
    # each waited the full 11 min on role processes that had already died.
    local max_wait="$1" prefill_log="$2" decode_log="$3" waited=0
    local prefill_ready=0 decode_ready=0
    echo "Waiting for prefill (port $PREFILL_PORT) and decode (port $DECODE_PORT) instances..."
    while [ "$waited" -lt "$max_wait" ]; do
        if [ "$prefill_ready" = 0 ]; then
            if curl -s "http://localhost:${PREFILL_PORT}/health" > /dev/null 2>&1; then
                prefill_ready=1
                echo -e "${GREEN}✓ prefill instance ready (port $PREFILL_PORT) after ${waited}s${NC}"
            elif role_exited prefill "$PREFILL_PID_FILE" "$prefill_log"; then
                return 1
            fi
        fi
        if [ "$decode_ready" = 0 ]; then
            if curl -s "http://localhost:${DECODE_PORT}/health" > /dev/null 2>&1; then
                decode_ready=1
                echo -e "${GREEN}✓ decode instance ready (port $DECODE_PORT) after ${waited}s${NC}"
            elif role_exited decode "$DECODE_PID_FILE" "$decode_log"; then
                return 1
            fi
        fi
        if [ "$prefill_ready" = 1 ] && [ "$decode_ready" = 1 ]; then
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    echo -e "\n${RED}✗ pd roles not ready within ${max_wait}s (prefill ready=$prefill_ready decode ready=$decode_ready)${NC}"
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
    # W29 (live L40S 2026-09-27): vLLM 0.19.1 has no --disable-log-requests; request
    # logging is off unless --enable-log-requests (mirrors manage_vllm_server.sh:
    # VLLM_DISABLE_LOG_REQUESTS=0 keeps the logs, anything else passes no flag).
    if [ "${VLLM_DISABLE_LOG_REQUESTS:-1}" = "0" ]; then
        ROLE_ARGS+=( --enable-log-requests )
    fi
    # Per-role connector config — the disaggregation wiring itself.
    ROLE_ARGS+=( --kv-transfer-config "$kv_cfg" )
    ROLE_ARGS+=( --max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" )
    # Per-instance dial from the shared-GPU rule (backlog A1), never an
    # inline fallback: the value depends on whether the roles share a GPU.
    ROLE_ARGS+=( --gpu-memory-utilization "$PD_MEM_UTIL" )
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
    local role="$1" model="$2" port="$3" kv_cfg="$4" budget="$5" args_line="$6" prefix="$7" nixl_port="$8"
    [ -n "${CAGE_RUN_ROOT:-}" ] || return 0
    local cuda_pin=""
    if [ "$role" = prefill ]; then cuda_pin="$PD_PREFILL_CUDA"; else cuda_pin="$PD_DECODE_CUDA"; fi
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
    SC_MEM_UTIL="$PD_MEM_UTIL" \
    SC_GPU_SHARE="$PD_GPU_SHARE" \
    SC_MEM_UTIL_SOURCE="$PD_MEM_UTIL_SOURCE" \
    SC_CUDA="$cuda_pin" \
    SC_EAGER="${VLLM_ENFORCE_EAGER:-0}" \
    SC_NIXL_PORT="$nixl_port" \
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
    # Backlog A1 (S0-9 / S0-20): the realized per-instance dial, the
    # shared-vs-distinct decision behind it, and this role's GPU pin.
    "gpu_memory_utilization": float(os.environ["SC_MEM_UTIL"]),
    "gpu_share": os.environ["SC_GPU_SHARE"],
    "gpu_memory_utilization_source": os.environ["SC_MEM_UTIL_SOURCE"],
    "cuda_visible_devices": os.environ.get("SC_CUDA") or None,
    "enforce_eager": os.environ.get("SC_EAGER") == "1",
    # ADR-0128 (S0F-13): this role's NIXL handshake listener port, and the
    # operator's UCX_* values as found (RECORDED, never set by the launcher:
    # part 5 of the ADR is record-only; the RDMA rung has its own UCX_TLS rule,
    # docs/VLLM_COMPATIBILITY.md section 8.3).
    "nixl_side_channel_port": int(os.environ["SC_NIXL_PORT"]),
    "ucx_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("UCX_")},
    "args": os.environ["SC_ARGS"],
}
with open(os.environ["SC_FILE"], "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")
PYEOF
    echo "  Serving config captured: $cfg_file"
}

launch_role_instance() {
    # $1 = CUDA_VISIBLE_DEVICES pin ('' = unpinned, ambient visibility), $2 =
    # this role's NIXL side-channel port, $3 = log file, $4 = pidfile, rest =
    # vllm serve argv. The pin and the port ride the child env ONLY (backlog
    # A1, ADR-0128): nothing is exported into this shell.
    local cuda_pin="$1" nixl_port="$2" log_file="$3" pid_file="$4"
    shift 4
    if [ -n "$cuda_pin" ]; then
        VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl_port" CUDA_VISIBLE_DEVICES="$cuda_pin" \
            nohup vllm serve "$@" > "$log_file" 2>&1 &
    else
        VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl_port" \
            nohup vllm serve "$@" > "$log_file" 2>&1 &
    fi
    printf '%s\n' "$!" > "$pid_file"
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
    local share_rule="DISTINCT_GPU_MEM_UTIL"
    [ "$PD_GPU_SHARE" = shared ] && share_rule="SHARED_GPU_MEM_UTIL"
    echo "[cage] gpu-share decision: $PD_GPU_SHARE (prefill=${PD_PREFILL_CUDA:-unpinned} decode=${PD_DECODE_CUDA:-unpinned}) -> --gpu-memory-utilization $PD_MEM_UTIL per instance [$PD_MEM_UTIL_SOURCE; rule $share_rule, backlog A1 / S0-9 / S0-20]"
    echo "[cage] nixl side channel: prefill=tcp://localhost:${NIXL_PORT_PREFILL} decode=tcp://localhost:${NIXL_PORT_DECODE} (VLLM_NIXL_SIDE_CHANNEL_PORT per role, ADR-0128 / S0F-13)"

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
        "$CAGE_KV_BUDGET_BYTES_PREFILL" "vllm serve $model ${prefill_args[*]}" "$want_prefix_cache" "$NIXL_PORT_PREFILL"
    echo "Starting prefill instance (logging to $prefill_log)..."
    launch_role_instance "$PD_PREFILL_CUDA" "$NIXL_PORT_PREFILL" "$prefill_log" "$PREFILL_PID_FILE" "$model" "${prefill_args[@]}"
    echo "Prefill PID: $(cat "$PREFILL_PID_FILE") (pidfile: $PREFILL_PID_FILE)"

    compose_role_args decode "$DECODE_PORT" "$DECODE_KV_TRANSFER_CONFIG" \
        "$CAGE_KV_BUDGET_BYTES_DECODE" "$cache_flag"
    local -a decode_args=( "${ROLE_ARGS[@]}" )
    echo "Server args [decode]: vllm serve $model ${decode_args[*]}"
    capture_role_config decode "$model" "$DECODE_PORT" "$DECODE_KV_TRANSFER_CONFIG" \
        "$CAGE_KV_BUDGET_BYTES_DECODE" "vllm serve $model ${decode_args[*]}" "$want_prefix_cache" "$NIXL_PORT_DECODE"
    echo "Starting decode instance (logging to $decode_log)..."
    launch_role_instance "$PD_DECODE_CUDA" "$NIXL_PORT_DECODE" "$decode_log" "$DECODE_PID_FILE" "$model" "${decode_args[@]}"
    echo "Decode PID: $(cat "$DECODE_PID_FILE") (pidfile: $DECODE_PID_FILE)"

    # Readiness: BOTH instances in one loop that fails at once on a dead role
    # (ADR-0128 part 4), then the proxy (whose /health requires both upstreams;
    # a proxy over a half-up pair must never report ready).
    local max_wait="${VLLM_START_TIMEOUT:-300}"
    wait_for_roles "$max_wait" "$prefill_log" "$decode_log" || return 1

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

# Sourced (bash-level unit tests of the functions above): define only, never
# dispatch. Executed: fall through to the verb dispatch.
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return 0
fi

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
