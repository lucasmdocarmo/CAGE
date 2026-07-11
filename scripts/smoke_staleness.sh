#!/bin/bash
# =============================================================================
# 5-query staleness SMOKE: validate the staleness serving path end-to-end BEFORE
# committing the full sweep (validate-before-run).
# =============================================================================
# The staleness axis is a manual env-var sweep (CAGE_STALE_FRACTION), absent from
# run_phase2.sh, and had no smoke artifact -- only a code comment. This script runs 5
# queries at a non-trivial stale fraction and asserts that (a) the summary carries a
# non-null staleness block (stale_hit_rate / unsafe_served_rate), and (b) at least one
# query was actually served the STALE (v0) evidence version. A pass means the injection
# fired and the metrics are computable; a fail means an env/context problem to fix before
# spending the sweep. GPU-only (needs a served vLLM). See
# Documentation/STALENESS_BASELINE_DESIGN.md.
#
# Usage: bash scripts/smoke_staleness.sh [MODEL]
#   CAGE_STALE_FRACTION overrides the stale fraction (default 0.5).
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL="${1:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-squad_v2}"
FRACTION="${CAGE_STALE_FRACTION:-0.5}"
OUT="${OUT:-$PROJECT_DIR/analysis/smoke/staleness}"

rm -rf "$OUT"; mkdir -p "$OUT"
echo "[smoke] staleness: model=$MODEL frac=$FRACTION Q=5 trials=1"

# Staleness serves from cache -> prefix caching ON. Restart to a clean server.
"$SCRIPT_DIR/manage_vllm_server.sh" restart "$MODEL"; sleep 10

if ! CAGE_STALE_FRACTION="$FRACTION" python3 "$PROJECT_DIR/scripts/run_experiment.py" \
        --baseline staleness --baseline-label staleness_smoke \
        --model "$MODEL" --dataset "$DATASET" \
        --num-queries 5 --num-trials 1 --seed 42 \
        --vllm-telemetry --output-dir "$OUT/staleness_smoke"; then
    echo "[smoke] FAIL: run errored."
    "$SCRIPT_DIR/manage_vllm_server.sh" stop >/dev/null 2>&1 || true
    exit 1
fi

"$SCRIPT_DIR/manage_vllm_server.sh" stop >/dev/null 2>&1 || true

python3 - "$OUT/staleness_smoke" <<'PY'
import csv, glob, json, os, sys
d = sys.argv[1]

# (a) a non-null staleness block in any metrics json produced by the run
ok_block = False
for mf in sorted(set(glob.glob(os.path.join(d, "**", "*metrics*.json"), recursive=True))):
    try:
        st = (json.load(open(mf)) or {}).get("staleness")
    except Exception:
        continue
    if st and (st.get("stale_hit_rate") is not None or st.get("unsafe_served_rate") is not None):
        print(f"[smoke] staleness block in {os.path.basename(mf)}: {st}")
        ok_block = True
        break

# (b) at least one served STALE (v0) row across the results csvs
v0 = 0
for cf in glob.glob(os.path.join(d, "**", "results*.csv"), recursive=True):
    try:
        for row in csv.DictReader(open(cf)):
            if row.get("evidence_version") == "v0":
                v0 += 1
    except Exception:
        pass
print(f"[smoke] served v0 (stale) rows: {v0}")

if not ok_block:
    print("[smoke] FAIL: no non-null staleness block found -- injection or aggregation did not run.")
    sys.exit(1)
if v0 < 1:
    print("[smoke] FAIL: no v0 (stale) rows -- staleness injection did not fire (check CAGE_STALE_FRACTION).")
    sys.exit(1)
print("[smoke] PASS: staleness path active (non-null metrics + >=1 stale row).")
PY
