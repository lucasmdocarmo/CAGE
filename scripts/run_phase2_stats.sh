#!/bin/bash
# Consolidate all Phase-2 baselines (core + compression 2x2 + speculative 2x2) into one
# dir and run the per-query statistical layer (Wilcoxon + Holm + bootstrap) vs no_cache.
# Emits a JSON summary + a paper-ready LaTeX table, then syncs to GCS.
set -uo pipefail
cd "$HOME/CAGE"
source cage-env/bin/activate
# Continuous log+results mirror to GCS + full collect on exit (no built-in sync loop).
source scripts/_log_guard.sh
ALL="$HOME/CAGE/analysis/all_results"
rm -rf "$ALL"; mkdir -p "$ALL"

for d in analysis/phase1/results/*/ analysis/compression/results/*/ analysis/speculative_matrix/*/; do
  [ -d "$d" ] || continue
  ln -sfn "$(cd "$d" && pwd)" "$ALL/$(basename "$d")"
done
echo "consolidated baselines:"; ls "$ALL" | tr '\n' ' '; echo

python3 scripts/statistical_tests.py --results-dir "$ALL" --reference no_cache \
    --metrics grounding_score faithfulness context_relevance ttft_ms latency_ms f1_score \
    --output "$ALL/phase2_stats.json" --latex-out "$ALL/phase2_stats.tex" 2>&1 | tail -50 \
    || echo "STATS_FAILED"

bash scripts/sync_results_to_gcs.sh analysis || true
echo "STATS_DONE"
