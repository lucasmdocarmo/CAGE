#!/bin/bash
# Consolidate all Phase-2 baselines (core + compression 2x2 + speculative 2x2) into one
# dir and run the per-query statistical layer (Wilcoxon + Holm + bootstrap) vs no_cache.
# Emits a JSON summary + a paper-ready LaTeX table, then syncs to GCS.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
if [ -z "${VIRTUAL_ENV:-}" ]; then
  for _v in cage-env .venv ../cage-env; do
    [ -f "$_v/bin/activate" ] && { source "$_v/bin/activate"; break; }
  done
fi

# Aggregate an EXISTING run root (results/<phase>/<run-id>/). Prefer CAGE_RUN_ROOT (exported by
# the sweep), else the run root passed as $1, else the newest run under results/<phase>/.
RUN_ROOT="${CAGE_RUN_ROOT:-${1:-}}"
if [ -z "$RUN_ROOT" ]; then
  _phase="${CAGE_PHASE:-phase2}"
  RUN_ROOT="$(ls -dt "$PROJECT_DIR/results/$_phase"/*/ 2>/dev/null | head -1)"; RUN_ROOT="${RUN_ROOT%/}"
fi
if [ -z "$RUN_ROOT" ] || [ ! -d "$RUN_ROOT" ]; then
  echo "ERROR: no run root to aggregate. Set CAGE_RUN_ID/CAGE_RUN_ROOT (same as the sweep)," >&2
  echo "       or pass it explicitly: bash scripts/4_analysis/run_phase2_stats.sh results/<phase>/<run-id>" >&2
  exit 1
fi
export CAGE_SYNC_DIR="${CAGE_SYNC_DIR:-${RUN_ROOT#"$PROJECT_DIR/"}}"
# Continuous log+results mirror to GCS + full collect on exit (no built-in sync loop).
source scripts/lib/_log_guard.sh
echo "[stats] aggregating run root: $RUN_ROOT"

ALL_Q="$RUN_ROOT/stats/all_results"        # Qwen (primary)
ALL_M="$RUN_ROOT/stats/all_results_mimo"   # MiMo (within-model)
rm -rf "$ALL_Q" "$ALL_M"; mkdir -p "$ALL_Q" "$ALL_M"

# Route each arm to its own model bucket by label tag so EVERY comparison is WITHIN-MODEL.
# MiMo arms (labels contain 'mimo') go to ALL_M; everything else (Qwen, the primary) to ALL_Q.
# Speculative decoding is output-lossless, so a MiMo spec arm's quality equals MiMo's -- it must
# be tested against the MiMo no_cache, never the Qwen one (that would report a model artifact).
# envelope + kv_store added 2026-07-16 (audit fix M4): they were silently excluded, so the
# run's cag_true and lmcache deltas carried no significance testing at all.
for d in "$RUN_ROOT"/baselines/*/ "$RUN_ROOT"/compression/*/ "$RUN_ROOT"/speculative/*/ "$RUN_ROOT"/envelope/*/ "$RUN_ROOT"/kv_store/*/; do
  [ -d "$d" ] || continue
  _b="$(basename "$d")"
  case "$_b" in
    *mimo*) ln -sfn "$(cd "$d" && pwd)" "$ALL_M/$_b" ;;
    *)      ln -sfn "$(cd "$d" && pwd)" "$ALL_Q/$_b" ;;
  esac
done
echo "Qwen baselines:"; ls "$ALL_Q" 2>/dev/null | tr '\n' ' '; echo
echo "MiMo baselines:"; ls "$ALL_M" 2>/dev/null | tr '\n' ' '; echo

# WARNING (rewritten 2026-07-16, audit S8 -- the old text described a cross-tree
# eager/max-len split that no longer exists): the serving config is UNIFORM across ALL
# trees as of 2026-07-15 (Option A, single source of truth scripts/lib/_serving_config.sh:
# non-eager, max_model_len=4096, gpu_memory_utilization=0.90; verified in the 100x3 run by
# identical kv_capacity_tokens across baselines and compression cells). Per-arm launch
# levers (fp8 KV, speculative config, connector) are captured per (re)start under
# observability/serving_configs/. Remaining caveats when reading deltas vs no_cache:
#  (1) Greedy T=0 is near-lossless, NOT identical: FP non-associativity (prefix-cache
#      reuse, CUDA-graph capture, fp8 KV) can flip near-tie argmaxes -- the measured
#      same-config token-divergence floor is ~11%. Quality deltas near that floor are not
#      attributable to the mechanism; read them beside token_divergence.json.
#  (2) Retrieval-context differences: rag/compressed/multiturn/corpus-prefix arms feed
#      DIFFERENT prompts than no_cache, so their quality deltas mix input effects with the
#      mechanism (see the input_effect flag in token_divergence.json).
echo "[stats] NOTE: serving config is uniform across trees (lib/_serving_config.sh, 2026-07-15). Remaining caveats vs no_cache: greedy near-losslessness (~11% same-config divergence floor) + retrieval-context (input-effect) differences -- see token_divergence.json."

# f1_answerable/exact_match_answerable/no_answer_correct are the SQuAD v2 no-answer
# decomposition (fix #4): answerable-only extraction quality + abstention accuracy. They are
# None on inapplicable rows, so statistical_tests.py subsets them automatically (and simply
# skips them for datasets without unanswerable items, e.g. NQ/MuSiQue).
# completeness_bertscore/rouge_l added 2026-07-15: the most-plotted quality metric was never
# significance-tested. abstention_precision pairs with no_answer_correct (recall) so the
# abstention behaviour is testable as a classifier, not just an accuracy.
METRICS="grounding_score faithfulness context_relevance ttft_ms latency_ms tpot_ms f1_score exact_match f1_answerable exact_match_answerable no_answer_correct abstention_precision hallucinated_span_ratio completeness_bertscore completeness_rouge_l"

# --- Qwen pass (reference no_cache) ---
if [ ! -d "$ALL_Q/no_cache" ]; then
  echo "ERROR: reference baseline 'no_cache' is missing from $ALL_Q." >&2
  echo "       Run the core suite (cloud_run.sh -> run_baselines.sh) before phase-2 stats." >&2
  exit 1
fi
python3 scripts/4_analysis/statistical_tests.py --results-dir "$ALL_Q" --reference no_cache \
    --metrics $METRICS \
    --output "$ALL_Q/phase2_stats.json" --latex-out "$ALL_Q/phase2_stats.tex" 2>&1 | tail -50 \
    || echo "STATS_FAILED (qwen)"

# --- cag_true paired mechanism pass (audit fix M4): on vs off, the run's cleanest pair,
# previously asserted "identical quality" with zero significance testing. Reference = off.
if [ -d "$RUN_ROOT/envelope/cag_true_on" ] && [ -d "$RUN_ROOT/envelope/cag_true_off" ]; then
  ALL_CT="$RUN_ROOT/stats/all_results_cagtrue"
  rm -rf "$ALL_CT"; mkdir -p "$ALL_CT"
  ln -sfn "$(cd "$RUN_ROOT/envelope/cag_true_on" && pwd)"  "$ALL_CT/cag_true_on"
  ln -sfn "$(cd "$RUN_ROOT/envelope/cag_true_off" && pwd)" "$ALL_CT/cag_true_off"
  python3 scripts/4_analysis/statistical_tests.py --results-dir "$ALL_CT" --reference cag_true_off \
      --metrics $METRICS --latex-label tab:significance-cagtrue \
      --output "$ALL_CT/cagtrue_stats.json" --latex-out "$ALL_CT/cagtrue_stats.tex" 2>&1 | tail -25 \
      || echo "STATS_FAILED (cag_true pair)"
fi

# --- MiMo pass (WITHIN-model reference no_cache_mimo7b) ---
# Only runs if MiMo was taken through the core suite (so a MiMo no_cache exists). If MiMo was
# speculative-only, there is no valid within-model reference: skip loudly rather than mis-compare.
if [ -d "$ALL_M/no_cache_mimo7b" ]; then
  python3 scripts/4_analysis/statistical_tests.py --results-dir "$ALL_M" --reference no_cache_mimo7b \
      --metrics $METRICS --latex-label tab:significance-mimo \
      --output "$ALL_M/phase2_stats.json" --latex-out "$ALL_M/phase2_stats.tex" 2>&1 | tail -50 \
      || echo "STATS_FAILED (mimo)"
elif [ -n "$(ls -A "$ALL_M" 2>/dev/null)" ]; then
  echo "[stats] MiMo arms present but no no_cache_mimo7b reference (MiMo was speculative-only)."
  echo "[stats]   -> skipping the within-MiMo Wilcoxon pass; MiMo acceptance is still in the spec summary below."
fi

# Aggregate speculative-decode acceptance + TPOT across the spec matrix cells so the paper's
# per-method serving comparison is produced automatically, not hand-copied from per-cell JSON.
export CAGE_SPEC_ROOT="$RUN_ROOT/speculative"
export CAGE_SPEC_OUT="$ALL_Q/spec_acceptance_summary.csv"
python3 - <<'PY'
import glob, json, os, csv, re, statistics
root = os.environ["CAGE_SPEC_ROOT"]
rows = []


def _trial_key(path):
    # NUMERIC trial sort (audit 2026-07-16 S10): the old lexicographic path sort breaks
    # at trial_10+ (trial_10 < trial_2), silently picking the wrong "last" trial.
    m = re.search(r"trial_(\d+)", path)
    return (int(m.group(1)) if m else 0, path)


for cell in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.isdir(cell):
        continue
    name = os.path.basename(cell)
    accs, tpots, method = [], [], None
    # Numeric trial order (trial_1 < trial_2 < ... < trial_10): acceptance counters are
    # CUMULATIVE-since-server-start (one server serves warmup + all trials of an arm),
    # so the LAST trial's value is the whole-arm acceptance (2026-07-15 review, B2).
    # CAVEAT (audit 2026-07-16 S10): cumulative-since-server-start means the figure
    # INCLUDES warmup-traffic drafts (the post-restart warmup request), not only the
    # measured queries; snapshot-and-subtract before trial_1 would remove it.
    for tj in sorted(glob.glob(os.path.join(cell, "**", "vllm_telemetry.json"), recursive=True), key=_trial_key):
        try:
            j = json.load(open(tj))
        except Exception:
            continue
        a = j.get("spec_decode_acceptance_rate")
        if a is None:
            a = (j.get("spec_decode") or {}).get("spec_decode_acceptance_rate")
        if isinstance(a, (int, float)):
            accs.append(float(a))
    for mj in glob.glob(os.path.join(cell, "**", "metrics.json"), recursive=True):
        try:
            m = json.load(open(mj))
        except Exception:
            continue
        t = (m.get("performance") or {}).get("avg_tpot_ms")
        if isinstance(t, (int, float)):
            tpots.append(float(t))
        method = method or (m.get("baseline_config") or {}).get("speculative_method")
    status = "ok"
    sfile = os.path.join(cell, "STATUS")
    if os.path.exists(sfile):
        status = open(sfile).read().strip()
    rows.append({
        "cell": name,
        "method": method or "",
        "n_with_acceptance": len(accs),
        # LAST trial = cumulative over the whole arm. Averaging the three nested
        # cumulative ratios (the old acceptance_rate_mean) double-counted early
        # traffic and was statistically meaningless -- the per-trial values are not
        # independent (2026-07-15 review, B2). tpot means stay: those are client-side
        # per-trial measurements, legitimately independent.
        "acceptance_rate_cumulative": round(accs[-1], 4) if accs else None,
        "avg_tpot_ms_mean": round(statistics.fmean(tpots), 3) if tpots else None,
        "status": status,
    })
if rows:
    out = os.environ["CAGE_SPEC_OUT"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(out.replace(".csv", ".json"), "w"), indent=2)
    print(f"[spec] acceptance/TPOT summary -> {out}")
    for r in rows:
        print("   ", r)
else:
    print("[spec] no speculative_matrix cells found; skipping acceptance summary.")
PY

# Token-divergence (near-lossless quantification, fix #6): how often each arm's greedy output
# differs from the within-model no_cache reference. Backs the manuscript's "near-lossless" claim
# with a measured number and bounds how much of a cross-config quality delta is token divergence
# vs the mechanism. Non-fatal (a missing reference just skips).
python3 scripts/4_analysis/token_divergence.py --results-dir "$ALL_Q" --reference no_cache \
    --output "$ALL_Q/token_divergence.json" || echo "DIVERGENCE_FAILED (qwen)"
if [ -d "$ALL_M/no_cache_mimo7b" ]; then
  python3 scripts/4_analysis/token_divergence.py --results-dir "$ALL_M" --reference no_cache_mimo7b \
      --output "$ALL_M/token_divergence.json" || echo "DIVERGENCE_FAILED (mimo)"
fi

# Regenerate the run's figures from the SAME canonical loader/estimand as the stats
# (pooled per-example medians + trial-level throughput), so plots and tables cannot
# disagree in sign. Also renders delta_vs_no_cache_forest.png from phase2_stats.json.
python3 scripts/4_analysis/generate_plots.py --results-dir "$RUN_ROOT" --plots-dir "$RUN_ROOT/plots" || echo "PLOTS_FAILED"

bash scripts/5_observability/sync_results_to_gcs.sh "$CAGE_SYNC_DIR" || true
echo "STATS_DONE"
