#!/bin/bash
# =============================================================================
# Gate 2: live infra preflight — run BEFORE every GPU sweep (validate-before-run).
# =============================================================================
# Codifies the user-mandated component check so a broken dependency fails LOUDLY in
# ~1 minute instead of hours into a paid sweep. Checks:
#   (a) vLLM /health 200 + the target model listed at /v1/models
#   (b) the quality layer loads and scores a REAL pair (LettuceDetect grounding + NLI),
#       grounding_score is a real number (not None -> model-load failure)
#   (c) cage-stats importable (rich telemetry, not spec-decode-only)
#   (d) FAISS + the retrieval embedding model load (RAG/redis/hybrid retrieval path)
#   (e) no mock / no disable escape-hatch env var is set
#
# Exit 0 = all green, safe to launch. Non-zero = at least one gate failed (do NOT launch).
# Usage: bash scripts/checks/preflight_check.sh [MODEL] [API_BASE]
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
MODEL="${1:-Qwen/Qwen3-8B}"
API_BASE="${2:-http://localhost:8000}"
FAILED=0
pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1"; FAILED=1; }

echo "=== Gate 2 preflight: model=$MODEL api=$API_BASE ==="

# (a) vLLM health + model listed
echo "(a) vLLM serving"
if curl -fsS "$API_BASE/health" >/dev/null 2>&1; then
    pass "/health 200"
else
    fail "/health not reachable at $API_BASE (start the server first)"
fi
if curl -fsS "$API_BASE/v1/models" 2>/dev/null | grep -q "$MODEL"; then
    pass "/v1/models lists $MODEL"
else
    fail "/v1/models does not list $MODEL (wrong model served?)"
fi
# Cold-start-per-trial depends on vLLM's dev endpoint POST /reset_prefix_cache (gated by
# VLLM_SERVER_DEV_MODE=1). Assert it actually returns 200 here, so the reset is not a silent
# no-op that would leave trials 2-3 measuring a warm cache.
if curl -fsS -X POST "$API_BASE/reset_prefix_cache" >/dev/null 2>&1; then
    pass "POST /reset_prefix_cache 200 (dev mode ON -> cold-start-per-trial will work)"
else
    fail "POST /reset_prefix_cache failed -> serve with VLLM_SERVER_DEV_MODE=1, else cold-start-per-trial silently no-ops"
fi

# (g) vllm serve ENTRYPOINT importable (2026-07-16 live finding): the sweep RESTARTS the
# server per tree, so gate (a) -- which probes the ALREADY-RUNNING server -- passes even
# when the venv is broken, and then every tree boot dies. Concretely: any lettucedetect
# (re)install pins openai==1.66.3, which lacks openai.types.responses.ResponsePrompt, and
# `vllm` cannot even import. This is a venv-level import check of the exact CLI path the
# restarts use.
echo "(g) vllm CLI entrypoint importable (venv-level, catches pip resolver drift)"
if python3 -c "from vllm.entrypoints.cli.main import main" 2>/dev/null; then
    pass "vllm CLI import OK"
else
    fail "vllm CLI import FAILED -- pip resolver drift (check openai/transformers: reinstalling lettucedetect downgrades openai below what vllm needs)"
fi

# (e) no mock / disable escape hatches (checked in-shell so it is loud even if python is skipped)
echo "(e) no mock / no disable escape hatches"
for _v in CAGE_TELEMETRY_MOCK CAGE_DISABLE_LETTUCEDETECT CAGE_DISABLE_COMPRESSION CAGE_ALLOW_NO_COMPRESSION; do
    _val="$(printf '%s' "${!_v:-}")"
    if [ -n "$_val" ] && [ "$_val" != "0" ]; then
        fail "$_v is set ($_v=$_val) -- would mock/disable a real component"
    else
        pass "$_v unset"
    fi
done

# (b)(c)(d) component loads via python
echo "(b/c/d) metric models + cage-stats + FAISS"
python3 - "$MODEL" <<'PY'
import sys
ok = True
def pw(m): print(f"  [PASS] {m}")
def pf(m):
    global ok; ok = False; print(f"  [FAIL] {m}")

# (c) cage-stats importable
try:
    import cage_stats.api  # noqa: F401
    pw("cage_stats.api importable")
except Exception as e:
    pf(f"cage_stats.api NOT importable: {e}")

# (b) quality layer: DISCRIMINATION gate (2026-07-15, task #59). Non-None alone would
# pass a scorer stuck at a constant (always 1.0), silently nulling the PRIMARY metric
# for a multi-day run. A grounded and a deliberately UNGROUNDED answer on the same
# context must separate by >= 0.3 on grounding AND faithfulness.
try:
    from src.evaluation.quality import QualityEvaluator
    qe = QualityEvaluator(device="cpu")
    # PROBE MUST USE A PARAGRAPH-LENGTH CONTEXT (2026-07-16 live finding): LettuceDetect is
    # RAGTruth-trained; against a one-line toy context it over-flags EVERYTHING (both probes
    # -> 0.000, false "constant scorer" gate failure). A realistic SQuAD-style paragraph
    # discriminates 0.00 vs 1.00 on the same stack.
    q = "In what country is Normandy located?"
    ctx = ["The Normans (Norman: Nourmands; French: Normands) were the people who in the 10th "
           "and 11th centuries gave their name to Normandy, a region in France. They were "
           "descended from Norse ('Norman' comes from 'Norseman') raiders and pirates from "
           "Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty "
           "to King Charles III of West Francia. Through generations of assimilation and "
           "mixing with the native Frankish and Roman-Gaulish populations, their descendants "
           "would gradually merge with the Carolingian-based cultures of West Francia. The "
           "distinct cultural and ethnic identity of the Normans emerged initially in the "
           "first half of the 10th century, and it continued to evolve over the succeeding "
           "centuries."]
    good = qe.evaluate(question=q, context=ctx,
                       generated_text="Normandy is located in France.",
                       reference_answer="France").to_dict()
    # NOT an abstention phrase (abstentions short-circuit grounding to None by design).
    bad = qe.evaluate(question=q, context=ctx,
                      generated_text="Normandy is located in Portugal and was founded by "
                                     "the Romans in 3 BC.",
                      reference_answer="France").to_dict()
    g_good, g_bad = good.get("grounding_score"), bad.get("grounding_score")
    if g_good is None or g_bad is None:
        pf("grounding_score is None -- LettuceDetect did not load (PRIMARY metric would be null all run)")
    elif (g_good - g_bad) < 0.3:
        pf(f"grounding does NOT discriminate: grounded={g_good:.3f} ungrounded={g_bad:.3f} "
           f"(separation < 0.3 -- constant/broken scorer)")
    else:
        pw(f"grounding discriminates: grounded={g_good:.3f} vs ungrounded={g_bad:.3f}")
    f_good, f_bad = good.get("faithfulness"), bad.get("faithfulness")
    if f_good is None or f_bad is None:
        pf("faithfulness is None -- NLI model did not load")
    elif (f_good - f_bad) < 0.3:
        pf(f"faithfulness does NOT discriminate: {f_good:.3f} vs {f_bad:.3f} (separation < 0.3)")
    else:
        pw(f"NLI discriminates: faithful={f_good:.3f} vs unfaithful={f_bad:.3f}")
except Exception as e:
    pf(f"quality layer error: {e}")

# (d) FAISS + retrieval embedding model
try:
    import faiss  # noqa: F401
    pw("faiss importable")
    from src.orchestration.baselines import get_baseline_config
    emb = get_baseline_config("rag").embedding_model
    from sentence_transformers import SentenceTransformer
    # Load on CPU to mirror the real run: run_experiment.py builds the retriever with
    # device="cpu" because vLLM reserves ~92% of the GPU (Qwen3-8B leaves only MiBs free),
    # so a default-CUDA load here would OOM on a config the sweep never actually uses.
    SentenceTransformer(emb, device="cpu")
    pw(f"retrieval embedding model loads on cpu: {emb}")
except Exception as e:
    pf(f"FAISS/retrieval error: {e}")

sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && FAILED=1

# (f) boot-disk free space: a multi-day sweep writes vLLM logs, observability snapshots and
# per-trial results continuously; a full boot disk kills the run hours in. Gate on the
# filesystem under $HOME (the boot disk on the GPU VM), falling back to /.
echo "(f) boot-disk free space"
MIN_FREE_GB="${CAGE_MIN_FREE_GB:-20}"
_free_kb="$(df -Pk "${HOME:-/}" 2>/dev/null | awk 'NR==2 {print $4}')"
[ -z "$_free_kb" ] && _free_kb="$(df -Pk / 2>/dev/null | awk 'NR==2 {print $4}')"
if [ -z "$_free_kb" ]; then
    fail "could not determine free disk space (df failed)"
else
    _free_gb=$(( _free_kb / 1024 / 1024 ))
    if [ "$_free_gb" -lt "$MIN_FREE_GB" ]; then
        fail "free disk ${_free_gb}GB < ${MIN_FREE_GB}GB (CAGE_MIN_FREE_GB) -- a multi-day sweep would fill the disk"
    else
        pass "free disk ${_free_gb}GB >= ${MIN_FREE_GB}GB (threshold: CAGE_MIN_FREE_GB)"
    fi
fi

echo "=============================================="
if [ "$FAILED" -eq 0 ]; then
    echo "PREFLIGHT PASS -- all Gate-2 components green. Safe to launch the sweep."
    exit 0
else
    echo "PREFLIGHT FAIL -- fix the [FAIL] items above before launching (do NOT spend GPU time)."
    exit 1
fi
