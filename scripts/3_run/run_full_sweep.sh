#!/bin/bash
# =============================================================================
# CAGE full-sweep orchestrator: core suite (+plots) -> compression 2x2 ->
# speculative 2x2 -> consolidated stats, ALL under ONE run-id
# (results/<phase>/<run-id>/{baselines,compression,speculative,stats,plots,observability}).
#
# RESUME SEMANTICS
#   The run-id is taken from an exported CAGE_RUN_ID if present (resume), else minted here
#   with cloud_run.sh's exact convention: <YYYY-MM-DD_HHMM>_<model-slug>_<Q>x<T>.
#   Every tree skips cells that are already COMPLETE (all trial_1..NUM_TRIALS/metrics.json
#   present) and continues past failed cells (STATUS=failed sentinels). After a crash,
#   preemption, or partial failure:
#       export CAGE_RUN_ID=<the-id-printed-at-launch>
#       bash scripts/3_run/run_full_sweep.sh [MODEL] [NUM_QUERIES] [NUM_TRIALS]
#   re-runs ONLY the missing/failed cells into the SAME run tree. CAGE_FORCE_RERUN=1 wipes
#   and re-runs completed cells too. Trees run in sequence; a failed tree does NOT stop the
#   sweep -- per-tree exit codes are collected, a final matrix summary is printed, and the
#   sweep exits nonzero if ANY tree failed.
#
# Usage (survive SSH drops):
#   nohup bash scripts/3_run/run_full_sweep.sh [MODEL] [NUM_QUERIES] [NUM_TRIALS] > sweep.log 2>&1 &
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR" || exit 1

MODEL="${1:-Qwen/Qwen3-8B}"
export NUM_QUERIES="${2:-${NUM_QUERIES:-500}}"
export NUM_TRIALS="${3:-${NUM_TRIALS:-3}}"

# Uniform serving config (Option A) first, so every tree inherits the identical serving env
# and the manifest records the real enforce_eager / max_model_len / gpu_memory_utilization.
source "$SCRIPT_DIR/../lib/_serving_config.sh"

# DECOUPLED SCORING (default ON, 2026-07-15): the serving loops skip inline model-based
# quality metrics (the ~90%-of-wall-clock CPU sink that idles the GPU); model quality is
# scored AFTER all serving trees, on the freed GPU, from qa_evidence.jsonl (scoring tree
# below). F1/EM/abstention are still computed inline (model-free). Set CAGE_SKIP_QUALITY=0
# to restore inline scoring.
export CAGE_SKIP_QUALITY="${CAGE_SKIP_QUALITY:-1}"

# ONE run-id for the whole matrix. Reuses cloud_run.sh's minting convention (cloud_run.sh:47)
# and EXPORTS it before delegating, so cloud_run.sh + both lever trees + stats all resolve
# the SAME results/<phase>/<run-id>/ root instead of each minting a fresh one.
PHASE="${PHASE:-phase2}"
_model_slug="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | sed -E 's|.*/||; s|[^a-z0-9]+|-|g; s|^-+||; s|-+$||')"
export CAGE_PHASE="$PHASE"
export CAGE_RUN_ID="${CAGE_RUN_ID:-$(date +%Y-%m-%d_%H%M)_${_model_slug}_${NUM_QUERIES}x${NUM_TRIALS}}"
export CAGE_RUN_ROOT="$PROJECT_DIR/results/${PHASE}/${CAGE_RUN_ID}"
mkdir -p "$CAGE_RUN_ROOT"

echo "=============================================="
echo "CAGE FULL SWEEP  model=$MODEL  Q=$NUM_QUERIES  trials=$NUM_TRIALS"
echo "run-id:   $CAGE_RUN_ID"
echo "run root: $CAGE_RUN_ROOT"
if [ -n "${CAGE_QUERY_MANIFEST:-}" ]; then
  echo "manifest: $CAGE_QUERY_MANIFEST  (uniform yardstick: every cell measures its query set)"
else
  echo "manifest: NONE -- per-script seeded sampling (build one with scripts/1_setup/build_query_manifest.py"
  echo "          for the uniform-N fairness contract; required for cag_true full-N pairing)"
fi
echo "RESUME: if this sweep dies, re-run with:"
echo "    export CAGE_RUN_ID=$CAGE_RUN_ID"
echo "    bash scripts/3_run/run_full_sweep.sh $MODEL $NUM_QUERIES $NUM_TRIALS"
echo "  -> completed cells are skipped; only missing/failed cells re-run."
echo "=============================================="

TREE_NAMES=()
TREE_RCS=()

run_tree() {  # <name> <cmd...> -- run one tree, record its exit code, never abort the sweep
    local name="$1" rc; shift
    echo ""
    echo "############## TREE $name START  $(date) ##############"
    "$@"
    rc=$?
    TREE_NAMES+=("$name")
    TREE_RCS+=("$rc")
    if [ "$rc" -eq 0 ]; then
        echo "############## TREE $name OK  $(date) ##############"
    else
        echo "############## TREE $name FAILED (exit $rc) -- continuing  $(date) ##############"
    fi
}

# 1. Core 6 baselines + plots (cloud_run.sh also runs GCS mirroring + the observability sidecar).
run_tree core bash scripts/3_run/cloud_run.sh "$MODEL" "$NUM_QUERIES" "$NUM_TRIALS"

# 2. Compression 2x2 (FP8-x-prefix-cache and LLMLingua gates run inside).
run_tree compression bash scripts/3_run/run_compression.sh "$MODEL"

# 3. Speculative 2x2 (native-draft engagement gate runs inside).
run_tree speculative bash scripts/3_run/run_speculative_matrix.sh "$MODEL"

# 4. Prefix-cache workload envelope + true-CAG cells (cag_true_off/on, grouped,
#    multiturn, repeat) -- the cells that let the prefix/CAG mechanism show itself.
run_tree envelope bash scripts/3_run/run_prefix_envelope.sh "$MODEL"

# 4b. OPT-IN: LMCache/CacheBlend kv_store arm (EXPERIMENTAL until its live gates pass;
#     needs `pip install lmcache` on the VM). Enable with CAGE_ENABLE_LMCACHE=1.
if [ "${CAGE_ENABLE_LMCACHE:-0}" = "1" ]; then
    run_tree kv_store bash scripts/3_run/run_kv_store.sh "$MODEL"
fi

# 5. Post-serving quality scoring on the freed GPU (decoupled mode): re-scores every
#    tree's qa_evidence.jsonl with the full metric stack and merges the quality columns
#    back into each trial's results.csv (one-time .pre_rescore backups). Runs before
#    stats so the Wilcoxon tables see the scored values.
if [ "${CAGE_SKIP_QUALITY}" = "1" ]; then
    run_tree scoring python3 scripts/4_analysis/rescore_quality.py \
        --run-root "$CAGE_RUN_ROOT" --full --device cuda --apply
fi

# 6. Consolidated per-query stats over the whole run root (also reads CAGE_RUN_ROOT from env);
#    regenerates plots over ALL cells at the end (fixes the 6-of-14 stale-plots failure mode).
run_tree stats bash scripts/4_analysis/run_phase2_stats.sh "$CAGE_RUN_ROOT"

echo ""
echo "=============================================="
echo "FULL SWEEP SUMMARY  (run-id: $CAGE_RUN_ID)"
ANY_FAILED=0
for i in "${!TREE_NAMES[@]}"; do
    if [ "${TREE_RCS[$i]}" -eq 0 ]; then
        echo "  ${TREE_NAMES[$i]} -> OK"
    else
        echo "  ${TREE_NAMES[$i]} -> FAILED (exit ${TREE_RCS[$i]})"
        ANY_FAILED=1
    fi
done
if [ "$ANY_FAILED" -ne 0 ]; then
    echo "SWEEP INCOMPLETE -- resume with:"
    echo "    export CAGE_RUN_ID=$CAGE_RUN_ID && bash scripts/3_run/run_full_sweep.sh $MODEL $NUM_QUERIES $NUM_TRIALS"
    echo "=============================================="
    exit 1
fi
echo "SWEEP COMPLETE -- results in $CAGE_RUN_ROOT"
echo "=============================================="
