#!/bin/bash
# =============================================================================
# Compression axis (the 2x2) — ratio-matched at ~2x
# =============================================================================
#   context source {CAG, RAG}  x  compression {full, compressed}
#     cag_full        = prefix_cache              (CAG, full precision)
#     rag_full        = rag                       (RAG, full text)
#     compressed_rag  = rag + LLMLingua-2         (~2x fewer prompt tokens; CLIENT-side)
#     compressed_cag  = prefix_cache + FP8 KV     (~2x smaller KV; server LAUNCH-time lever)
#
# Read the 2x2 DOWN (CAG vs RAG) or ACROSS (full vs compressed), never on the diagonal.
# FP8 KV is GPU-meaningful. A pre-flight gate verifies FP8 does NOT disable prefix caching
# (else compressed_cag is confounded — see Cloud/VLLM_COMPATIBILITY.md sec 4).
# =============================================================================
# No -e: one failed cell must NOT abort the 2x2. Cells write a STATUS sentinel on failure,
# are recorded in FAILED, and the script exits nonzero at the END (after attempting all).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL="${1:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-squad_v2}"
NUM_QUERIES="${NUM_QUERIES:-500}"
NUM_TRIALS="${NUM_TRIALS:-3}"
SEED="${SEED:-42}"
SKIP_GATE="${SKIP_GATE:-0}"

# Outputs land under the shared run root minted by cloud_run.sh (CAGE_RUN_ROOT/compression) so the
# core + compression + speculative trees for ONE sweep cell aggregate together; a standalone run
# self-mints the SAME date_HHMM_model_NxT run-id under results/<phase>/<run-id>/.
if [ -n "${CAGE_RUN_ROOT:-}" ]; then
    RUN_ROOT="$CAGE_RUN_ROOT"
else
    _phase="${CAGE_PHASE:-phase2}"
    _slug="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | sed -E 's|.*/||; s|[^a-z0-9]+|-|g; s|^-+||; s|-+$||')"
    _rid="${CAGE_RUN_ID:-$(date +%Y-%m-%d_%H%M)_${_slug}_${NUM_QUERIES}x${NUM_TRIALS}}"
    RUN_ROOT="$PROJECT_DIR/results/$_phase/$_rid"
fi
OUTPUT_DIR="$RUN_ROOT/compression"
# Point the sourced log guard's GCS mirror at this run root (relative to PROJECT_DIR), not analysis/.
export CAGE_SYNC_DIR="${CAGE_SYNC_DIR:-${RUN_ROOT#"$PROJECT_DIR/"}}"
# Continuous log+results mirror to GCS + full collect on exit (this script has no sync loop).
source "$SCRIPT_DIR/../lib/_log_guard.sh"

# Pre-flight: refuse to run the compression axis with the compression escape hatches set.
# CAGE_DISABLE_COMPRESSION=1 makes the compressor a pass-through, and CAGE_ALLOW_NO_COMPRESSION=1
# disables strict mode; either one left in the VM environment would silently turn compressed_rag /
# compressed_cag into a NON-compressed arm (ratio 1.0) mislabeled as compressed. Fail loud.
for _cvar in CAGE_DISABLE_COMPRESSION CAGE_ALLOW_NO_COMPRESSION; do
    _cval="${!_cvar:-}"
    if [ -n "$_cval" ] && [ "$_cval" != "0" ]; then
        echo "ABORT: $_cvar is set (${_cvar}=${_cval}); the compression arm would be invalid."
        echo "  unset $_cvar before running the compression sweep."
        exit 1
    fi
done

# Uniform serving config across ALL trees (Option A): non-eager / max_len 4096 / mem-util 0.90,
# so the compression 2x2 is served under the SAME regime as the core suite and the speculative
# tree, and cross-mechanism comparisons are fair. Non-eager now pays a ~2-3 min CUDA-graph
# capture per per-cell restart (accepted for comparability); if a cell OOMs non-eager on the
# 24GB L4, fall back for THIS tree only via VLLM_ENFORCE_EAGER=1 -- a recorded deviation the run
# manifest captures. mem-util (0.90) is the swept memory-pressure axis.
source "$SCRIPT_DIR/../lib/_serving_config.sh"

cd "$PROJECT_DIR"
# Activate the project venv if the caller has not already (cage-env on the VM, .venv locally),
# so a standalone `nohup bash scripts/3_run/run_compression.sh` does not fall back to system python.
if [ -z "${VIRTUAL_ENV:-}" ]; then
  for _v in cage-env .venv ../cage-env; do
    [ -f "$_v/bin/activate" ] && { echo "Activating venv: $_v"; source "$_v/bin/activate"; break; }
  done
fi
mkdir -p "$OUTPUT_DIR"

# Model tag so a second model (MiMo) through the same 2x2 never collides with Qwen's dirs.
case "$MODEL" in *MiMo*|*mimo*) MTAG="_mimo7b" ;; *) MTAG="" ;; esac

# ---------------------------------------------------------------------------
# Fault tolerance + resume (mirrors run_speculative_matrix.sh's sentinel pattern):
# complete cells (all trial_1..NUM_TRIALS metrics.json) are skipped unless CAGE_FORCE_RERUN=1;
# a failed cell/server writes STATUS=failed and the 2x2 continues; exit nonzero at the end.
# ---------------------------------------------------------------------------
FAILED=()

cell_complete() {  # <cell_dir> -> 0 iff trial_1..NUM_TRIALS all have metrics.json
    local dir="$1" t
    for ((t = 1; t <= NUM_TRIALS; t++)); do
        [ -f "$dir/trial_${t}/metrics.json" ] || return 1
    done
    return 0
}

prepare_cell() {  # <full label> -> 0 = run it (stale dir wiped), 1 = skip (already complete)
    local label="$1" dir="$OUTPUT_DIR/$1"
    if [ "${CAGE_FORCE_RERUN:-0}" = "1" ]; then
        [ -d "$dir" ] && echo "    FORCE RERUN (CAGE_FORCE_RERUN=1): wiping $label"
        rm -rf "$dir"
        return 0
    fi
    if cell_complete "$dir"; then
        echo "SKIP (complete): $label"
        return 1
    fi
    if [ -d "$dir" ]; then
        echo "    PARTIAL: wiping incomplete $label and re-running"
        rm -rf "$dir"
    fi
    return 0
}

group_complete() {  # <bare-label...> -> 0 iff every cell is complete (server start unnecessary)
    [ "${CAGE_FORCE_RERUN:-0}" = "1" ] && return 1
    local lbl
    for lbl in "$@"; do
        cell_complete "$OUTPUT_DIR/${lbl}${MTAG}" || return 1
    done
    return 0
}

skip_group() {  # <bare-label...> announce each already-complete cell
    local lbl
    for lbl in "$@"; do echo "SKIP (complete): ${lbl}${MTAG}"; done
}

mark_cells_failed() {  # <reason> <bare-label...> sentinel the cells a dead server orphaned
    local reason="$1" lbl full; shift
    for lbl in "$@"; do
        full="${lbl}${MTAG}"
        cell_complete "$OUTPUT_DIR/$full" && continue  # keep a previously-complete cell
        mkdir -p "$OUTPUT_DIR/$full"
        echo "STATUS=failed reason=$reason model=$MODEL $(date)" > "$OUTPUT_DIR/$full/STATUS"
        FAILED+=("$full($reason)")
    done
}

run_baseline() {  # <baseline> <label> [extra args...]
    local baseline=$1 label="$2${MTAG}"; shift 2
    echo ""; echo ">>> $label ($baseline)  $(date)"
    prepare_cell "$label" || return 0
    # All 4 compression cells run on a prefix-caching-ON server, so cold-start each trial
    # (vLLM /reset_prefix_cache) for independent trials, consistent with the core suite.
    if ! python3 scripts/3_run/run_experiment.py \
        --baseline "$baseline" --baseline-label "$label" \
        --model "$MODEL" --dataset "$DATASET" \
        --num-queries "$NUM_QUERIES" --num-trials "$NUM_TRIALS" --seed "$SEED" \
        --reset-cache-between-trials \
        --vllm-telemetry --output-dir "$OUTPUT_DIR/$label" "$@"; then
        echo "    CELL $label RUN-FAIL"
        mkdir -p "$OUTPUT_DIR/$label"
        echo "STATUS=failed reason=run model=$MODEL baseline=$baseline $(date)" > "$OUTPUT_DIR/$label/STATUS"
        FAILED+=("$label(run)")
        return 0
    fi
}

echo "=============================================="
echo "CAGE Compression Axis (2x2, ratio-matched ~2x)"
echo "Model: $MODEL  Dataset: $DATASET  Q:$NUM_QUERIES  Trials:$NUM_TRIALS"
echo "=============================================="

# Pre-flight: FP8 must NOT disable prefix caching, or compressed_cag is confounded (RQ5/H4).
if [ "$SKIP_GATE" != "1" ]; then
    echo ">>> Pre-flight: FP8 x prefix-caching gate"
    if ! bash "$SCRIPT_DIR/../checks/check_fp8_prefix_cache.sh" "$MODEL"; then
        echo "GATE FAILED -> compressed_cag would be 'no-reuse + compression'."
        echo "Pin a compatible vLLM (Cloud/VLLM_COMPATIBILITY.md sec 4) or SKIP_GATE=1 to override."
        exit 1
    fi
fi

# Pre-flight: compressed_rag needs LLMLingua-2 or it silently no-ops (Phase-2 bug:
# compression_applied=False, ratio=1.0 for all rows -> the arm measured plain RAG).
if ! python3 -c "import llmlingua" 2>/dev/null; then
    echo "GATE FAILED -> 'llmlingua' not importable; compressed_rag would NO-OP (ratio 1.0)."
    echo "Run: pip install llmlingua   (or SKIP_GATE=1 to override and accept an invalid arm)."
    [ "$SKIP_GATE" = "1" ] || exit 1
fi

# --- Full row + compressed_rag (full-precision server, prefix caching ON) ---
if group_complete cag_full rag_full compressed_rag; then
    echo ">>> all full-precision cells complete -- skipping server start"
    skip_group cag_full rag_full compressed_rag
elif { echo ">>> Server: full precision, prefix caching ON"; ./scripts/2_serving/manage_vllm_server.sh restart "$MODEL"; }; then
    sleep 10
    run_baseline prefix_cache   cag_full
    run_baseline rag            rag_full
    export CAGE_REQUIRE_COMPRESSION=1   # raise (not silent no-op) if LLMLingua can't compress
    # --context-source retrieved is REQUIRED: compressed_rag's family is not in the
    # retrieval set {rag,redis,hybrid}, so without it the arm compresses GOLD context
    # (CAG+compression) instead of RETRIEVED context (RAG+compression), breaking the
    # 2x2 ACROSS read (rag_full vs compressed_rag). This is the Phase-2 confound.
    run_baseline compressed_rag compressed_rag --context-source retrieved
    unset CAGE_REQUIRE_COMPRESSION
else
    echo ">>> SERVER-FAIL (full precision) -> marking dependent cells failed and continuing"
    mark_cells_failed server cag_full rag_full compressed_rag
fi

# --- compressed_cag (FP8 KV — the same launch-lever speculative uses) ---
if group_complete compressed_cag; then
    echo ">>> compressed_cag complete -- skipping FP8 server start"
    skip_group compressed_cag
elif { echo ">>> Server: FP8 KV cache ON (compressed_cag)"; VLLM_KV_CACHE_DTYPE=fp8 ./scripts/2_serving/manage_vllm_server.sh restart "$MODEL"; }; then
    sleep 10
    run_baseline compressed_cag compressed_cag
else
    echo ">>> SERVER-FAIL (FP8 KV) -> marking compressed_cag failed"
    mark_cells_failed server compressed_cag
fi

./scripts/2_serving/manage_vllm_server.sh stop || true
echo ""; echo "=============================================="
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Compression 2x2 INCOMPLETE: ${#FAILED[@]} cell(s) failed: ${FAILED[*]} -> $OUTPUT_DIR"
    echo "(failed cells carry a STATUS sentinel; re-run with the same CAGE_RUN_ID to resume)"
    echo "=============================================="
    exit 1
fi
echo "Compression 2x2 complete -> $OUTPUT_DIR"
echo "Read DOWN (CAG vs RAG) or ACROSS (full vs compressed), not diagonally."
echo "=============================================="
