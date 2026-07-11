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
# Usage: bash scripts/preflight_check.sh [MODEL] [API_BASE]
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
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

# (b) quality layer: LettuceDetect grounding + NLI faithfulness score a real pair
try:
    from src.evaluation.quality import QualityEvaluator
    qe = QualityEvaluator(device="cpu")
    q = "What color is the sky on a clear day?"
    ctx = ["On a clear day the sky appears blue because of Rayleigh scattering."]
    m = qe.evaluate(question=q, context=ctx, generated_text="The sky is blue.", reference_answer="blue")
    d = m.to_dict()
    if d.get("grounding_score") is None:
        pf("grounding_score is None -- LettuceDetect did not load (PRIMARY metric would be null all run)")
    else:
        pw(f"LettuceDetect grounding_score={d['grounding_score']:.3f}")
    if d.get("faithfulness") is None:
        pf("faithfulness is None -- NLI model did not load")
    else:
        pw(f"NLI faithfulness={d['faithfulness']:.3f}")
except Exception as e:
    pf(f"quality layer error: {e}")

# (d) FAISS + retrieval embedding model
try:
    import faiss  # noqa: F401
    pw("faiss importable")
    from src.orchestration.baselines import get_baseline_config
    emb = get_baseline_config("rag").embedding_model
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(emb)
    pw(f"retrieval embedding model loads: {emb}")
except Exception as e:
    pf(f"FAISS/retrieval error: {e}")

sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && FAILED=1

echo "=============================================="
if [ "$FAILED" -eq 0 ]; then
    echo "PREFLIGHT PASS -- all Gate-2 components green. Safe to launch the sweep."
    exit 0
else
    echo "PREFLIGHT FAIL -- fix the [FAIL] items above before launching (do NOT spend GPU time)."
    exit 1
fi
