#!/bin/bash
# Consolidate all Phase-2 baselines (core + compression 2x2 + speculative 2x2) into one
# dir and run the per-query statistical layer (Wilcoxon + Holm + bootstrap) vs no_cache.
# Emits a JSON summary + a paper-ready LaTeX table, then syncs to GCS.
set -uo pipefail
cd "$HOME/CAGE"
source cage-env/bin/activate
# Continuous log+results mirror to GCS + full collect on exit (no built-in sync loop).
source scripts/_log_guard.sh
ALL_Q="$HOME/CAGE/analysis/all_results"        # Qwen (primary)
ALL_M="$HOME/CAGE/analysis/all_results_mimo"   # MiMo (within-model)
rm -rf "$ALL_Q" "$ALL_M"; mkdir -p "$ALL_Q" "$ALL_M"

# Route each arm to its own model bucket by label tag so EVERY comparison is WITHIN-MODEL.
# MiMo arms (labels contain 'mimo') go to ALL_M; everything else (Qwen, the primary) to ALL_Q.
# Speculative decoding is output-lossless, so a MiMo spec arm's quality equals MiMo's -- it must
# be tested against the MiMo no_cache, never the Qwen one (that would report a model artifact).
for d in analysis/phase1/results/*/ analysis/compression/results/*/ analysis/speculative_matrix/*/; do
  [ -d "$d" ] || continue
  _b="$(basename "$d")"
  case "$_b" in
    *mimo*) ln -sfn "$(cd "$d" && pwd)" "$ALL_M/$_b" ;;
    *)      ln -sfn "$(cd "$d" && pwd)" "$ALL_Q/$_b" ;;
  esac
done
echo "Qwen baselines:"; ls "$ALL_Q" 2>/dev/null | tr '\n' ' '; echo
echo "MiMo baselines:"; ls "$ALL_M" 2>/dev/null | tr '\n' ' '; echo

# WARNING on serving metrics (ttft_ms, latency_ms, tpot_ms): the core suite runs non-eager /
# max-len 8192 while the compression and speculative trees run --enforce-eager / max-len 4096.
# Per PHASE2_PLAN_OF_RECORD.md, serving numbers are only comparable WITHIN a tree, so those rows
# for compression/speculative arms vs the core no_cache mix the mechanism with the eager/context
# serving difference -- interpret them within-tree only. Quality metrics (greedy T=0 -> identical
# tokens regardless of serving config) ARE valid cross-tree.
echo "[stats] NOTE: serving metrics (ttft/latency/tpot) are within-tree comparable only (core=non-eager/8192 vs compression&speculative=eager/4096)."

METRICS="grounding_score faithfulness context_relevance ttft_ms latency_ms tpot_ms f1_score exact_match hallucinated_span_ratio"

# --- Qwen pass (reference no_cache) ---
if [ ! -d "$ALL_Q/no_cache" ]; then
  echo "ERROR: reference baseline 'no_cache' is missing from $ALL_Q." >&2
  echo "       Run the core suite (cloud_run.sh -> run_phase1.sh) before phase-2 stats." >&2
  exit 1
fi
python3 scripts/statistical_tests.py --results-dir "$ALL_Q" --reference no_cache \
    --metrics $METRICS \
    --output "$ALL_Q/phase2_stats.json" --latex-out "$ALL_Q/phase2_stats.tex" 2>&1 | tail -50 \
    || echo "STATS_FAILED (qwen)"

# --- MiMo pass (WITHIN-model reference no_cache_mimo7b) ---
# Only runs if MiMo was taken through the core suite (so a MiMo no_cache exists). If MiMo was
# speculative-only, there is no valid within-model reference: skip loudly rather than mis-compare.
if [ -d "$ALL_M/no_cache_mimo7b" ]; then
  python3 scripts/statistical_tests.py --results-dir "$ALL_M" --reference no_cache_mimo7b \
      --metrics $METRICS \
      --output "$ALL_M/phase2_stats.json" --latex-out "$ALL_M/phase2_stats.tex" 2>&1 | tail -50 \
      || echo "STATS_FAILED (mimo)"
elif [ -n "$(ls -A "$ALL_M" 2>/dev/null)" ]; then
  echo "[stats] MiMo arms present but no no_cache_mimo7b reference (MiMo was speculative-only)."
  echo "[stats]   -> skipping the within-MiMo Wilcoxon pass; MiMo acceptance is still in the spec summary below."
fi

# Aggregate speculative-decode acceptance + TPOT across the spec matrix cells so the paper's
# per-method serving comparison is produced automatically, not hand-copied from per-cell JSON.
python3 - <<'PY'
import glob, json, os, csv, statistics
root = "analysis/speculative_matrix"
rows = []
for cell in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.isdir(cell):
        continue
    name = os.path.basename(cell)
    accs, tpots, method = [], [], None
    for tj in glob.glob(os.path.join(cell, "**", "vllm_telemetry.json"), recursive=True):
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
        "acceptance_rate_mean": round(statistics.fmean(accs), 4) if accs else None,
        "avg_tpot_ms_mean": round(statistics.fmean(tpots), 3) if tpots else None,
        "status": status,
    })
if rows:
    out = "analysis/all_results/spec_acceptance_summary.csv"
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

bash scripts/sync_results_to_gcs.sh analysis || true
echo "STATS_DONE"
