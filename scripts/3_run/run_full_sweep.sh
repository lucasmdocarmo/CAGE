#!/bin/bash
# Order:     stage 3 — THE top-level sweep entry; drives cloud_run.sh + lever trees + run_phase2_stats.sh under ONE run-id
# Objective: Pilot-harness full-sweep orchestrator (core suite -> compression 2x2 -> opt-in trees -> consolidated stats) with skip-completed resume
# Cloud:     both
# =============================================================================
# PILOT HARNESS — drives the retired 9-name taxonomy via the alias map; the
# campaign harness (CellSpec-native, D6 open-loop) lands at tranche P1; use for
# pilot re-scoring only.
# =============================================================================
# CAGE full-sweep orchestrator: core suite (+plots) -> compression 2x2 ->
# [speculative 2x2: RETIRED, opt-in only] -> consolidated stats, ALL under ONE run-id
# (results/<phase>/<run-id>/{baselines,compression,speculative,stats,plots,observability}).
#
# RESUME SEMANTICS
#   The run-id is taken from an exported CAGE_RUN_ID if present (resume), else minted here
#   with cloud_run.sh's exact convention (mint_run_id in scripts/lib/_common.sh):
#   <YYYY-MM-DD_HHMMSS>_<model-slug>_<Q>x<T>_<4-hex>_<dataset>.
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
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"

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

# ONE run-id for the whole matrix. Reuses cloud_run.sh's minting convention (_common.sh
# mint_run_id: seconds + random suffix + dataset, finding J3 -- minute-granular ids could
# fragment/converge runs) and EXPORTS it before delegating, so cloud_run.sh + both lever
# trees + stats all resolve the SAME results/<phase>/<run-id>/ root instead of each
# minting a fresh one.
PHASE="${PHASE:-phase2}"
_model_slug="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | sed -E 's|.*/||; s|[^a-z0-9]+|-|g; s|^-+||; s|-+$||')"
export CAGE_PHASE="$PHASE"
# Campaign v2 mode (task #116): CAGE_CAMPAIGN (+ CAGE_SESSION), or a preset
# CAGE_CAMPAIGN_ROOT, routes the whole sweep into the RESULTS_LAYOUT tree
# results/<campaign>/<session>/<run_id>/ — the core + compression trees write v2
# cells through run_experiment's campaign mode, pilot-only trees are skipped, and
# the run is SEALED (§5) + verified at the end. Pilot mode (default) is untouched.
CAMPAIGN_MODE=0
if [ -n "${CAGE_CAMPAIGN_ROOT:-}" ] || [ -n "${CAGE_CAMPAIGN:-}" ]; then
    CAMPAIGN_MODE=1
    if [ -z "${CAGE_CAMPAIGN_ROOT:-}" ]; then
        [ -n "${CAGE_SESSION:-}" ] || die "campaign mode needs CAGE_SESSION (a|b|cd-act1|cd-act2) alongside CAGE_CAMPAIGN"
        export CAGE_RUN_ID="${CAGE_RUN_ID:-$(mint_campaign_run_id "$CAGE_SESSION" "$_model_slug")}"
        export CAGE_CAMPAIGN_ROOT="$PROJECT_DIR/results/${CAGE_CAMPAIGN}/${CAGE_SESSION}/${CAGE_RUN_ID}"
    else
        export CAGE_RUN_ID="${CAGE_RUN_ID:-$(basename "$CAGE_CAMPAIGN_ROOT")}"
    fi
    export CAGE_RUN_ROOT="$CAGE_CAMPAIGN_ROOT"
    log "CAMPAIGN MODE (task #116): v2 run root $CAGE_CAMPAIGN_ROOT"
else
    export CAGE_RUN_ID="${CAGE_RUN_ID:-$(mint_run_id "$_model_slug" "$NUM_QUERIES" "$NUM_TRIALS" "${DATASET:-squad_v2}")}"
    export CAGE_RUN_ROOT="$PROJECT_DIR/results/${PHASE}/${CAGE_RUN_ID}"
fi
mkdir -p "$CAGE_RUN_ROOT"

# One orchestrator per run root (finding J3): a second resume instance on the same root
# could rm -rf a cell a live tree is writing. The child runners re-enter via
# CAGE_RUN_LOCK_HELD instead of re-acquiring. MUST happen BEFORE the backup daemon
# start below: detached daemons are launched with the lock fd closed (200>&-), and a
# daemon that inherited the fd would hold the lock after this orchestrator died.
acquire_run_lock "$CAGE_RUN_ROOT"

# Redundant cloud backup of the WHOLE results/<phase>/ tree (every run-id + every tree,
# including the lever trees / scoring / stats that cloud_run.sh's core-only syncer misses).
# Runs for the entire sweep; final sync + stop on exit. LOUD no-op if CAGE_RESULTS_BUCKET
# is unset (set it to this run's bucket). Concurrent with cloud_run.sh's syncer is safe
# (no --delete anywhere).
_BACKUP_TREE="results/${PHASE}"
if [ "$CAMPAIGN_MODE" = "1" ]; then
    _BACKUP_TREE="${CAGE_RUN_ROOT#"$PROJECT_DIR/"}"   # the campaign run tree itself
fi
bash "$SCRIPT_DIR/../5_observability/gcs_backup_daemon.sh" start "$_BACKUP_TREE" || true
trap 'bash "$SCRIPT_DIR/../5_observability/gcs_backup_daemon.sh" stop "'"${_BACKUP_TREE}"'" >/dev/null 2>&1 || true' EXIT

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

# 3. Speculative 2x2 -- RETIRED (charter §7.5, MyDocs/PUBLICATION.md): the speculative
#    arms are out of the campaign design; the harness moved to scripts/deprecated/.
#    Default = SKIP (loudly). Pilot re-scoring may opt back in explicitly.
if [ "${CAGE_RUN_RETIRED_SPECULATIVE:-0}" = "1" ]; then
    warn "running the RETIRED speculative 2x2 (CAGE_RUN_RETIRED_SPECULATIVE=1; charter §7.5) -- pilot forensics only"
    run_tree speculative bash scripts/deprecated/run_speculative_matrix.sh "$MODEL"
else
    log "TREE speculative SKIPPED -- retired per charter §7.5 (set CAGE_RUN_RETIRED_SPECULATIVE=1 to run the deprecated pilot harness)"
fi

# 4. Prefix-cache workload envelope + true-CAG cells (cag_true_off/on, grouped,
#    multiturn, repeat) -- the cells that let the prefix/CAG mechanism show itself.
#    Campaign mode: NOT campaign-wired (pilot-only tree) -- skipped loudly.
if [ "$CAMPAIGN_MODE" = "1" ]; then
    log "TREE envelope SKIPPED -- pilot-only tree (not campaign-wired; task #116 wires core+compression)"
else
    run_tree envelope bash scripts/3_run/run_prefix_envelope.sh "$MODEL"
fi

# 4b. OPT-IN: LMCache/CacheBlend kv_store arm (EXPERIMENTAL until its live gates pass;
#     needs `pip install lmcache "transformers>=4.36,<5"` on the VM -- the pin re-assert
#     stops lmcache dragging in transformers 5.x, which broke the pilot-pin vLLM
#     0.11.0 tokenizer path; the campaign-pin 0.19.1 pairing is NOT live-validated --
#     see the STATUS note in run_kv_store.sh). Enable with CAGE_ENABLE_LMCACHE=1.
if [ "${CAGE_ENABLE_LMCACHE:-0}" = "1" ]; then
    if [ "$CAMPAIGN_MODE" = "1" ]; then
        log "TREE kv_store SKIPPED -- pilot-only tree (not campaign-wired)"
    else
        run_tree kv_store bash scripts/3_run/run_kv_store.sh "$MODEL"
    fi
fi

if [ "$CAMPAIGN_MODE" = "1" ]; then
    # 5c. Campaign run end (task #116): write-time-hash cross-check + the §5 seal
    #     (ledger.json, ONCE, before any analysis), then the v2 verification gate
    #     ON THE NODE (S0-15). Scoring stays offline/decoupled (§6:
    #     rescore_quality.py --scoring-run-id into scoring/<id>/, never cells/),
    #     and the pilot stats tree does not read v2 trees -- the analysis chain is
    #     verify_results.py -> organize_results.py -> run_campaign_analysis.py.
    run_tree seal python3 scripts/3_run/seal_campaign_run.py "$CAGE_RUN_ROOT"
    run_tree verify python3 scripts/4_analysis/verify_results.py "$CAGE_RUN_ROOT" \
        --out "${CAGE_RUN_ROOT}_verification"
    log "TREE scoring SKIPPED in-sweep -- campaign scoring is decoupled (§6): rescore_quality.py --scoring-run-id <id> post-run"
    log "TREE stats SKIPPED -- pilot stats do not read v2 trees; use organize_results.py + run_campaign_analysis.py on the pulled run"
else
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
fi

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
