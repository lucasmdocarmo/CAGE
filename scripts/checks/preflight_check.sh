#!/bin/bash
# Order:     gate — after 2_serving is up, before EVERY 3_run launch (Gate 2; re-run after every engine relaunch)
# Objective: Live infra preflight gates (a)-(s): serving health, quality stack, telemetry, retrieval, env poison, iso-bytes parity, layout round-trip, blindness matrix, engine-model gate matrix
# Cloud:     both
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
#   (e) no mock / no disable escape-hatch / no unrecorded-deviation env var is set
#       (+ loud CAGE_QUALITY_STRICT / CAGE_CLAIM_CHECKER state flags -- J6, #120)
#   (f) boot-disk free space   (g) vllm CLI importable   (h) D2 telemetry parity
#   (i) environment-vs-registration (interpreter + pins + pip check)
#   (j) charter §6.5 realized-KV iso-BYTES parity across engine startup logs
#       (J5 fix, task #138 -- the gate the launchers defer to; live at S0)
#   (k) per-backend endpoint liveness (final-scope engines, serial serving)
#   (l) campaign-layout round-trip (v2 producer -> organizer)      [#118 item]
#   (m) open-loop schedule + measured-replay guard smoke           [#118 item]
#   (n) calibration artifact (cal-v1) presence/shape               [#118 item]
#   (o) regime-inputs bridge on live telemetry (live-only; skips)  [#118 item]
#   (p) dataset staleness refusal (requested charter datasets staged on disk)
#   (q) cage-stats pin parity (requirements.txt pinned SHA == installed commit;
#       task #143, finding L-A -- a lagging install fabricates rho_KV readings)
#   (r) engine blindness matrix (task T8.1, 2026-08-27 audit item S4): per-
#       backend /metrics presence/granularity over the telemetry fields
#       cage-stats consumes; absence is DATA, offline is skip-3 (live-only)
#   (s) engine x model gate-matrix runner (task T8.3; VLLM_COMPATIBILITY.md
#       section 7): deterministic-mode / TurboMind-selected / fp8xprefix /
#       cached-token checks -- PASS/FAIL/PENDING-VERIFY-LIVE, never a
#       fabricated PASS
#
# Exit 0 = all green, safe to launch. Non-zero = at least one gate failed (do NOT launch).
# Usage: bash scripts/checks/preflight_check.sh [MODEL] [API_BASE]
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/lib/_common.sh
source "$PROJECT_DIR/scripts/lib/_common.sh"
# No -e in this script (accumulates FAILED): guard the cd so the python probes below
# can never import from the wrong working directory.
cd "$PROJECT_DIR" || die "cannot cd to $PROJECT_DIR"
MODEL="${1:-Qwen/Qwen3-8B}"
API_BASE="${2:-http://localhost:8000}"
FAILED=0
pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1"; FAILED=1; }
# Maps a python sub-gate's exit code onto the accumulator: 0 = pass (the gate
# printed its own [PASS] lines), 3 = explicit skip-with-reason (the gate
# printed a [SKIP] line naming WHY it cannot run yet -- not a failure),
# anything else = gate failed. Skips are only legal where the gate itself
# declares a precondition (live server, staged calibration artifact).
gate_rc() {
    case "$1" in
        0|3) : ;;
        *) FAILED=1 ;;
    esac
}

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
# CAGE-POISON-ENV-GATE-BEGIN (J11/J6 fix, task #138; extracted and
# behavior-tested by tests/test_preflight_gates.py -- keep the BEGIN/END
# markers intact). Poison semantics: a confirmatory run REFUSES to launch
# while any of these is set, unless the deviation is explicitly expected AND
# recorded for the run (unset it here, re-set it inside the recorded arm).
echo "(e) no mock / no disable escape hatches / no unrecorded-deviation levers"
for _v in CAGE_TELEMETRY_MOCK CAGE_DISABLE_LETTUCEDETECT CAGE_DISABLE_COMPRESSION \
          CAGE_ALLOW_NO_COMPRESSION CAGE_ALLOW_REPLAY CAGE_ALLOW_NO_BACKUP \
          LMDEPLOY_CACHE_MAX_ENTRY_COUNT LMDEPLOY_QUANT_POLICY; do
    _val="$(printf '%s' "${!_v:-}")"
    if [ -n "$_val" ] && [ "$_val" != "0" ]; then
        fail "$_v is set ($_v=$_val) -- would mock/disable/bypass a real component or hand an engine an unrecorded launch lever"
    else
        pass "$_v unset"
    fi
done
# CAGE_QUALITY_STRICT: src/evaluation/quality.py treats unset as strict=ON
# (default "1"); the poison is an explicit falsy value ("0"/"false"/"no" --
# quality.py's exact falsy set), which downgrades instrument load/score
# failures from raise to score=None for the WHOLE run (J6 context, #120).
_qs_raw="${CAGE_QUALITY_STRICT:-}"
# strip whitespace to mirror quality.py's .strip().lower() exactly (' 0 ' is poison there too)
case "$(printf '%s' "$_qs_raw" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')" in
    0|false|no)
        fail "CAGE_QUALITY_STRICT=$_qs_raw DISABLES the fail-closed quality layer (instrument failures would silently record None instead of raising; src/evaluation/quality.py) -- forbidden for confirmatory runs"
        ;;
    *)
        pass "CAGE_QUALITY_STRICT strict (value: ${_qs_raw:-unset -> default strict})"
        ;;
esac
# CAGE_LMDEPLOY_BACKEND_CHECK=warn bypasses the P7 TurboMind assertion in
# scripts/2_serving/manage_lmdeploy_server.sh (a PyTorch-engine fallback could
# serve silently). Only unset/'strict' is confirmatory-safe.
_bc="${CAGE_LMDEPLOY_BACKEND_CHECK:-}"
case "$_bc" in
    ""|strict)
        pass "CAGE_LMDEPLOY_BACKEND_CHECK strict (value: ${_bc:-unset})"
        ;;
    *)
        fail "CAGE_LMDEPLOY_BACKEND_CHECK=$_bc bypasses the P7 TurboMind backend assertion (never-mixed policy) -- only 'strict' (or unset) is confirmatory-safe"
        ;;
esac
# J6 (#120 pending): the claim-checker default is IN-CODE ('alignscore',
# src/evaluation/quality.py) and unloadable in the project venv -> the quality
# layer fails closed at evaluator construction. FLAG the effective state
# loudly on every preflight so the operator sees which checker a run would
# demand BEFORE spending GPU time. Informational: never a pass/fail by itself.
if [ -n "${CAGE_CLAIM_CHECKER:-}" ]; then
    echo "  [flag] CAGE_CLAIM_CHECKER=${CAGE_CLAIM_CHECKER} (explicit selection; must match the D8 registered instrument)"
else
    echo "  [flag] CAGE_CLAIM_CHECKER unset -> in-code default 'nli' (src/evaluation/quality.py; in-process-safe per owner decision #120/F8 2026-08-19; alignscore = Instrument B, requested explicitly by scripts/4_analysis/score_instrument_b.py)"
fi
# CAGE-POISON-ENV-GATE-END

# (b)(c)(d) component loads via python
echo "(b/c/d) metric models + cage-stats + FAISS"
if ! python3 - "$MODEL" <<'PY'
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
then
    FAILED=1
fi

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

# (h) charter-D2 telemetry-parity gate: every measured backend must DECLARE
# streamed first-token TTFT in its adapter capabilities() (ADR-0007 data feed;
# the adapters encode unverified entries as 'verify-live', never fabricated
# True). Backends checked: CAGE_PREFLIGHT_BACKENDS (comma-separated adapter
# tokens, default "vllm,sglang,lmdeploy" -- ALL final-scope engines, J11 fix
# task #138; scope down explicitly ONLY for a single-engine pilot tree, as a
# recorded deviation -- the scope is echoed into the run log by gates h/j/k).
# Each configured adapter's full capabilities() dict is printed for the run
# log; a backend lacking streamed_ttft FAILS the gate (its TTFT would be a
# full-response proxy, silently mixing methodologies against the streamed
# single-stream baseline -- charter D2/D6 sec. 6.3).
echo "(h) telemetry parity: adapter capabilities() must declare streamed_ttft (charter D2)"
PREFLIGHT_BACKENDS="${CAGE_PREFLIGHT_BACKENDS:-vllm,sglang,lmdeploy}"
if ! python3 - "$PREFLIGHT_BACKENDS" <<'PY'
# CAGE-D2-TELEMETRY-PARITY-GATE (extracted and exercised by tests/test_integration_wiring.py)
import json
import sys

backends = [
    b.strip()
    for b in (sys.argv[1] if len(sys.argv) > 1 else "vllm").split(",")
    if b.strip()
]
ok = True
def pw(m): print(f"  [PASS] {m}")
def pf(m):
    global ok; ok = False; print(f"  [FAIL] {m}")

try:
    from src.inference.vllm_adapter import VLLMAdapter
    from src.inference.sglang_adapter import SGLangAdapter
    from src.inference.lmdeploy_adapter import LMDeployAdapter
except Exception as e:
    pf(f"could not import adapter classes: {e}")
    sys.exit(1)

# HTTP adapter tokens whose constructors are offline (no request at init).
ADAPTERS = {
    "vllm": VLLMAdapter,
    "sglang": SGLangAdapter,
    "lmdeploy": LMDeployAdapter,
    "lmdeploy-turbomind": LMDeployAdapter,
}

for backend in backends:
    cls = ADAPTERS.get(backend)
    if cls is None:
        pf(f"backend '{backend}': no serving-grade adapter with a verifiable "
           f"streamed_ttft capability (hf-oracle is the reference engine, "
           f"gemini/ollama are legacy) -- charter D2 telemetry parity cannot "
           f"hold for a measured arm on this backend")
        continue
    try:
        caps = cls(
            model_name="preflight-probe", api_base="http://localhost:1"
        ).capabilities()
    except Exception as e:
        pf(f"backend '{backend}': constructing the adapter to read "
           f"capabilities() failed: {e}")
        continue
    print(f"  [caps] {backend}: {json.dumps(caps, sort_keys=True, default=str)}")
    if caps.get("streamed_ttft") is True:
        pw(f"backend '{backend}' declares streamed_ttft=True")
    else:
        pf(f"backend '{backend}' lacks streamed_ttft "
           f"(got {caps.get('streamed_ttft')!r}) -- charter D2 telemetry "
           f"parity requires real streamed first-token TTFT on measured arms")

sys.exit(0 if ok else 1)
PY
then
    FAILED=1
fi

# (i) environment matches the registration (code assertion 2026-08-07, findings
#     B1/B4): the interpreter must be the canonical one the Tier-1 pins were
#     frozen on, `pip check` must be clean, every Tier-1 exact pin must be
#     installed at exactly its pinned version, and — once B2's lockfile exists
#     (requirements.lock.gpu.txt) — the full locked set must match. Runs with
#     the SAME python3 every other gate probes (the activated cage-env).
echo "[gate i] environment-vs-registration (interpreter + Tier-1 pins + pip check)..."
if ! python3 - "$CAGE_CANONICAL_PYTHON" "$CAGE_ROOT" <<'PY'
import importlib.metadata as md
import re
import subprocess
import sys
from pathlib import Path

canonical, root = sys.argv[1], Path(sys.argv[2])
ok = True


def pf(msg: str) -> None:
    global ok
    ok = False
    print(f"  [FAIL] {msg}")


got = f"{sys.version_info.major}.{sys.version_info.minor}"
if got != canonical:
    pf(f"interpreter is CPython {got}, canonical is {canonical} "
       f"(Tier-1 pins in requirements.txt were frozen on {canonical}; "
       f"finding B1 -- this venv was created with the wrong python)")
else:
    print(f"  [ok] interpreter CPython {got} == canonical {canonical}")

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?==([^,;\s]+)$")


def check_pins(path: Path, label: str) -> None:
    n_checked = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-") or "@" in line:
            continue
        m = _PIN.match(line)
        if not m:
            continue
        name, want = m.groups()
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            pf(f"{label}: {name}=={want} declared but NOT installed")
            continue
        n_checked += 1
        if have != want:
            pf(f"{label}: {name} installed {have} != pinned {want}")
    print(f"  [ok] {label}: {n_checked} exact pins verified against the venv")


check_pins(root / "requirements.txt", "requirements.txt Tier-1")
lock = root / "requirements.lock.gpu.txt"
if lock.is_file():
    check_pins(lock, "requirements.lock.gpu.txt")
else:
    print("  [note] requirements.lock.gpu.txt absent (B2 pending; minted at S0)")

res = subprocess.run(
    [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
)
if res.returncode != 0:
    pf(f"pip check reports broken requirements:\n{res.stdout.strip()}")
else:
    print("  [ok] pip check clean")

sys.exit(0 if ok else 1)
PY
then
    FAILED=1
fi

# ---------------------------------------------------------------------------
# (j) charter §6.5 realized-KV iso-BYTES parity gate (J5 fix, task #138).
# The launch-side mapping (scripts/lib/_serving_config.sh) only SETS each
# engine's native dial; THIS gate proves the bytes. It parses each scoped
# engine's newest startup log (written by scripts/2_serving/manage_*_server.sh
# with the co-resident stack loaded) for the REALIZED KV pool size and asserts
# pairwise parity within CAGE_ISO_BYTES_TOL (default 0.05 relative). Scope
# follows CAGE_PREFLIGHT_BACKENDS; pin exact logs (e.g. one budget point of a
# pressure sweep) with CAGE_ISO_BYTES_LOGS="vllm=/path/a.log,sglang=/path/b.log".
# Role-split (prefill/decode disaggregation) logs pin as 'engine:role=path'
# (roles are free-form tokens, e.g. vllm:prefill=a.log,vllm:decode=b.log):
# each role log is parsed alone and the engine enters cross-engine parity with
# its role pools SUMMED (charter §6.5: budget B = pools SUMMED). Optional
# CAGE_ISO_POOL_SUM_BYTES=<int> (feed it gate_j.expected_bytes_total from
# src/orchestration/cache_budget.py) additionally asserts each role-split
# engine's summed realized bytes against that budget within CAGE_ISO_BYTES_TOL;
# setting it with NO role-split logs pinned is operator error and FAILS.
# Duplicate engine[:role] keys REFUSE loudly -- never silent last-wins.
# NOTE: the RUNBOOK env-contract table does not yet carry the PD-grammar row
# (engine[:role]= / CAGE_ISO_POOL_SUM_BYTES); this comment is the contract
# until that row lands.
# Parsers are fixture-tested locally (tests/test_preflight_gates.py); the LIVE
# execution against real engine logs happens at S0.
# ---------------------------------------------------------------------------
echo "(j) charter 6.5 realized-KV iso-bytes parity (scope: $PREFLIGHT_BACKENDS; tol: ${CAGE_ISO_BYTES_TOL:-0.05})"
python3 - "$PREFLIGHT_BACKENDS" <<'PY'
# CAGE-ISO-BYTES-GATE (charter §6.5; J5 fix, task #138). Extracted and
# fixture-tested by tests/test_preflight_gates.py: the per-engine parsers and
# the parity math are unit-tested against realistic startup-log variants.
# Engine log-line formats are [VERIFY-LIVE at S0]: a line that carries a
# recognized anchor but does not parse is CORRUPTED and fails LOUD, and a log
# with no recognizable KV-pool line fails LOUD -- never a silent pass.
import os
import re
import sys
from pathlib import Path

GIB = 1024 ** 3
MIB = 1024 ** 2
#: vLLM V0's default KV block size in tokens (legacy "# GPU blocks" channel
#: only; the modern channels report tokens/bytes directly). [VERIFY-LIVE at S0]
VLLM_BLOCK_TOKENS = 16


class IsoBytesError(ValueError):
    """Loud parse/parity failure -- the §6.5 gate must never guess."""


_NUM = r"(-?[0-9][0-9_,]*(?:\.[0-9]+)?)"


def _num(text: str) -> float:
    value = float(text.replace(",", "").replace("_", ""))
    if value < 0:
        raise IsoBytesError(f"negative KV-pool quantity {text!r}")
    return value


# Per engine: (anchor substring, full regex, channel, scale-to-channel-unit).
# channel 'bytes' | 'tokens' are direct; the composite channels are folded in
# parse_engine_log. An anchor hit whose full regex does NOT match raises
# (corrupted line), so silent engine log-format drift is impossible.
_RULES = {
    "vllm": [
        # V1 (>=0.8) kv_cache_utils.py: realized pool size in tokens.
        ("GPU KV cache size:",
         re.compile(rf"GPU KV cache size:\s*{_NUM}\s*tokens"), "tokens", 1.0),
        # V1 gpu_worker.py: realized pool memory.
        ("Available KV cache memory:",
         re.compile(rf"Available KV cache memory:\s*{_NUM}\s*GiB"), "bytes", float(GIB)),
        # V0 0.6.x memory-profile summary ("...the rest of the memory reserved
        # for KV Cache is 5.33GiB").
        ("memory reserved for KV Cache is",
         re.compile(rf"memory reserved for KV Cache is\s*{_NUM}\s*GiB"), "bytes", float(GIB)),
        # Legacy block report; VLLM_BLOCK_TOKENS tokens per block.
        ("# GPU blocks:",
         re.compile(rf"# GPU blocks:\s*{_NUM}"), "tokens", float(VLLM_BLOCK_TOKENS)),
    ],
    "sglang": [
        # srt.mem_cache.memory_pool allocation report (K + V summed; SGLang
        # prints "GB" but computes 1024**3 -- binary units, like vLLM's GiB).
        ("KV Cache is allocated",
         re.compile(
             rf"KV Cache is allocated\.?\s*#tokens:\s*{_NUM},\s*"
             rf"K size:\s*{_NUM}\s*GB,\s*V size:\s*{_NUM}\s*GB"),
         "sglang-alloc", None),
        # Scheduler config echo (tokens-only fallback channel).
        ("max_total_num_tokens",
         re.compile(rf"max_total_num_tokens\s*[=:]\s*{_NUM}"), "tokens", 1.0),
    ],
    "lmdeploy": [
        # TurboMind BlockManager pair: bytes = block_size(MB=2^20) x count.
        ("[BlockManager] block_size",
         re.compile(rf"\[BlockManager\]\s*block_size\s*=\s*{_NUM}\s*MB"),
         "lmdeploy-block-mb", None),
        ("[BlockManager] max_block_count",
         re.compile(rf"\[BlockManager\]\s*max_block_count\s*=\s*{_NUM}"),
         "lmdeploy-block-count", None),
    ],
}


def parse_engine_log(engine, text):
    """Parse one engine startup log -> {'engine', 'bytes', 'tokens', 'evidence'}.

    'bytes'/'tokens' are ints or None (channel not present in this log); the
    LAST occurrence of a channel wins (a log holds exactly one server start,
    but a re-profiled pool must supersede its predecessor). 'evidence' keeps
    every matched line verbatim for the run log."""
    rules = _RULES.get(engine)
    if rules is None:
        raise IsoBytesError(
            f"engine {engine!r} has no KV-pool parser -- add its startup-log "
            f"rules before scoping it into the iso-bytes gate")
    found = {}
    evidence = []
    for line in text.splitlines():
        for anchor, pattern, channel, scale in rules:
            if anchor in line:
                m = pattern.search(line)
                if not m:
                    raise IsoBytesError(
                        f"{engine}: corrupted KV-pool line (anchor {anchor!r} "
                        f"present but the value did not parse): {line.strip()!r}")
                if channel == "sglang-alloc":
                    found["tokens"] = _num(m.group(1))
                    found["bytes"] = (_num(m.group(2)) + _num(m.group(3))) * GIB
                else:
                    value = _num(m.group(1))
                    # scale None = composite channel folded below (lmdeploy).
                    found[channel] = value if scale is None else value * scale
                evidence.append(line.strip())
    if engine == "lmdeploy":
        mb = found.pop("lmdeploy-block-mb", None)
        count = found.pop("lmdeploy-block-count", None)
        if (mb is None) != (count is None):
            raise IsoBytesError(
                "lmdeploy: found only one of the [BlockManager] block_size / "
                "max_block_count pair -- cannot compute the realized pool from "
                "half a product")
        if mb is not None:
            found["bytes"] = mb * MIB * count
    if not found:
        raise IsoBytesError(
            f"{engine}: NO recognizable KV-pool line in the startup log -- "
            f"§6.5 realized-bytes parity cannot be certified (engine log "
            f"format drift? update the parser rules; never skip this gate)")
    return {
        "engine": engine,
        "bytes": int(found["bytes"]) if "bytes" in found else None,
        "tokens": int(found["tokens"]) if "tokens" in found else None,
        "evidence": evidence,
    }


def relative_gap(a, b):
    top = max(float(a), float(b))
    if top <= 0:
        raise IsoBytesError(
            "KV pool of size 0 -- an engine realized NO cache; parity over "
            "zero is meaningless")
    return abs(float(a) - float(b)) / top


def compare_pair(ra, rb, tol):
    """-> (basis, gap, within). bytes when both sides report bytes; tokens as
    an explicit PROXY basis (valid only at uniform KV dtype) when bytes are
    unavailable on both sides; no common basis raises (incomparable)."""
    if ra["bytes"] is not None and rb["bytes"] is not None:
        basis, gap = "bytes", relative_gap(ra["bytes"], rb["bytes"])
    elif ra["tokens"] is not None and rb["tokens"] is not None:
        basis, gap = "tokens", relative_gap(ra["tokens"], rb["tokens"])
    else:
        raise IsoBytesError(
            f"{ra['engine']} vs {rb['engine']}: no common basis (one log "
            f"reports only bytes, the other only tokens) -- cannot certify "
            f"§6.5 parity; capture a log variant carrying the missing channel")
    return basis, gap, gap <= tol


def newest_log(root, engine):
    logs = sorted((root / engine).glob("*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def main(argv):
    raw = argv[1] if len(argv) > 1 else "vllm,sglang,lmdeploy"
    engines = []
    for token in raw.split(","):
        token = token.strip()
        token = "lmdeploy" if token == "lmdeploy-turbomind" else token
        if token and token not in engines:
            engines.append(token)
    tol_raw = os.environ.get("CAGE_ISO_BYTES_TOL", "0.05")
    try:
        tol = float(tol_raw)
    except ValueError:
        print(f"  [FAIL] CAGE_ISO_BYTES_TOL={tol_raw!r} is not a float")
        return 1
    if not (0.0 < tol < 1.0):
        print(f"  [FAIL] CAGE_ISO_BYTES_TOL={tol} outside (0, 1) -- refusing a "
              f"vacuous or impossible tolerance")
        return 1

    # pins: engine -> {role_or_None: Path}. role None = the legacy bare
    # 'engine=path' form (one whole-pool log); role tokens are free-form
    # ('engine:role=path', split on the FIRST colon). Duplicate engine[:role]
    # keys and a bare+role mix for one engine both REFUSE: either would make
    # which log counts a silent coin-flip.
    pins = {}
    seen_keys = set()
    for entry in os.environ.get("CAGE_ISO_BYTES_LOGS", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            print(f"  [FAIL] CAGE_ISO_BYTES_LOGS entry {entry!r} is not engine[:role]=path")
            return 1
        key, _, path = entry.partition("=")
        eng, colon, role = key.strip().partition(":")
        eng = eng.strip()
        eng = "lmdeploy" if eng == "lmdeploy-turbomind" else eng
        role = role.strip() if colon else None
        if not eng or (colon and not role):
            print(f"  [FAIL] CAGE_ISO_BYTES_LOGS key {key.strip()!r} is malformed "
                  f"(want engine or engine:role)")
            return 1
        norm_key = eng if role is None else f"{eng}:{role}"
        if norm_key in seen_keys:
            print(f"  [FAIL] CAGE_ISO_BYTES_LOGS duplicate key {norm_key!r} -- each "
                  f"engine[:role] may be pinned exactly once (silent last-wins "
                  f"would hide a log from the §6.5 gate)")
            return 1
        seen_keys.add(norm_key)
        pins.setdefault(eng, {})[role] = Path(path.strip())
    for eng, eng_pins in pins.items():
        if None in eng_pins and len(eng_pins) > 1:
            print(f"  [FAIL] CAGE_ISO_BYTES_LOGS mixes bare {eng!r} with role-split "
                  f"{eng}:<role> entries -- ambiguous whether the bare log is the "
                  f"whole pool or another role; pin one form only")
            return 1

    pool_sum_raw = os.environ.get("CAGE_ISO_POOL_SUM_BYTES")
    pool_sum = None
    if pool_sum_raw is not None:
        try:
            pool_sum = int(pool_sum_raw.strip())
        except ValueError:
            print(f"  [FAIL] CAGE_ISO_POOL_SUM_BYTES={pool_sum_raw!r} is not an integer")
            return 1
        if pool_sum <= 0:
            print(f"  [FAIL] CAGE_ISO_POOL_SUM_BYTES={pool_sum} is not a positive "
                  f"byte count")
            return 1
        if not any(role is not None
                   for eng in engines for role in pins.get(eng, {})):
            print("  [FAIL] CAGE_ISO_POOL_SUM_BYTES is set but CAGE_ISO_BYTES_LOGS "
                  "pins no role-split (engine:role) log in scope -- a declared "
                  "pool sum with nothing to sum is operator error; drop the env "
                  "or pin the role logs")
            return 1

    log_root = Path(os.environ.get("CAGE_ISO_BYTES_LOG_ROOT", "logs"))
    readings, ok = [], True
    for engine in engines:
        engine_pins = pins.get(engine, {})
        role_pins = {r: p for r, p in engine_pins.items() if r is not None}
        if not role_pins:
            path = engine_pins.get(None) or newest_log(log_root, engine)
            if path is None or not path.is_file():
                print(f"  [FAIL] {engine}: no startup log under {log_root / engine}/ "
                      f"-- launch the engine via scripts/2_serving/ first or pin "
                      f"CAGE_ISO_BYTES_LOGS; scope down CAGE_PREFLIGHT_BACKENDS "
                      f"ONLY as a recorded deviation")
                ok = False
                continue
            try:
                reading = parse_engine_log(
                    engine, path.read_text(encoding="utf-8", errors="replace"))
            except IsoBytesError as exc:
                print(f"  [FAIL] {engine}: {exc} (log: {path})")
                ok = False
                continue
            reading["log"] = str(path)
            readings.append(reading)
            size = "n/a" if reading["bytes"] is None else f"{reading['bytes'] / GIB:.3f} GiB"
            toks = "n/a" if reading["tokens"] is None else str(reading["tokens"])
            # mtime printed so a STALE log (older budget point) is visible in the run log.
            print(f"  [pool] {engine}: bytes={size} tokens={toks} "
                  f"log={path} mtime={path.stat().st_mtime:.0f}")
            continue
        # Role-split (P/D disaggregation): parse each role log alone, then the
        # engine enters cross-engine parity as the SUM of its role pools
        # (charter §6.5: budget B = pools SUMMED). Role logs are always
        # explicit pins -- newest-log discovery cannot tell roles apart.
        parts, engine_ok = [], True
        for role, path in role_pins.items():
            label = f"{engine}:{role}"
            if not path.is_file():
                print(f"  [FAIL] {label}: pinned role log {path} does not exist -- "
                      f"every engine:role entry in CAGE_ISO_BYTES_LOGS must point "
                      f"at a real startup log")
                engine_ok = False
                continue
            try:
                part = parse_engine_log(
                    engine, path.read_text(encoding="utf-8", errors="replace"))
            except IsoBytesError as exc:
                print(f"  [FAIL] {label}: {exc} (log: {path})")
                engine_ok = False
                continue
            part["log"] = str(path)
            parts.append(part)
            size = "n/a" if part["bytes"] is None else f"{part['bytes'] / GIB:.3f} GiB"
            toks = "n/a" if part["tokens"] is None else str(part["tokens"])
            print(f"  [pool] {label}: bytes={size} tokens={toks} "
                  f"log={path} mtime={path.stat().st_mtime:.0f}")
        if not engine_ok:
            ok = False
            continue
        # A channel sums only when EVERY role log carries it -- summing a
        # present value with an absent one would fabricate a pool size.
        agg_bytes = (None if any(p["bytes"] is None for p in parts)
                     else sum(p["bytes"] for p in parts))
        agg_tokens = (None if any(p["tokens"] is None for p in parts)
                      else sum(p["tokens"] for p in parts))
        if agg_bytes is None and agg_tokens is None:
            print(f"  [FAIL] {engine}: role logs share no common channel (one "
                  f"reports only bytes, another only tokens) -- the §6.5 pool "
                  f"SUM cannot be formed; capture log variants carrying the "
                  f"missing channel")
            ok = False
            continue
        readings.append({
            "engine": engine,
            "bytes": agg_bytes,
            "tokens": agg_tokens,
            "evidence": [line for p in parts for line in p["evidence"]],
            "log": ",".join(p["log"] for p in parts),
            "pd_roles": len(parts),
        })
        size = "n/a" if agg_bytes is None else f"{agg_bytes / GIB:.3f} GiB"
        toks = "n/a" if agg_tokens is None else str(agg_tokens)
        print(f"  [pool] {engine}: SUM of {len(parts)} role pools -> "
              f"bytes={size} tokens={toks}")
    if pool_sum is not None:
        # Charter §6.5 budget assertion: each role-split engine's realized
        # pools must SUM to the planned budget B (cache_budget.py emits it as
        # gate_j.expected_bytes_total). Bytes-denominated by definition, so
        # token-only role logs cannot certify it and FAIL.
        for reading in readings:
            if "pd_roles" not in reading:
                continue
            engine = reading["engine"]
            if reading["bytes"] is None:
                print(f"  [FAIL] {engine}: CAGE_ISO_POOL_SUM_BYTES declared but the "
                      f"role logs carry no bytes channel -- a BYTES pool sum "
                      f"cannot be certified from token-only logs")
                ok = False
                continue
            gap = relative_gap(reading["bytes"], pool_sum)
            if gap <= tol:
                print(f"  [PASS] {engine}: pool SUM {reading['bytes']} vs declared "
                      f"CAGE_ISO_POOL_SUM_BYTES={pool_sum}: gap {gap:.4f} <= tol {tol}")
            else:
                print(f"  [FAIL] {engine}: pool SUM {reading['bytes']} vs declared "
                      f"CAGE_ISO_POOL_SUM_BYTES={pool_sum}: gap {gap:.4f} > tol {tol} "
                      f"-- realized P/D pools do NOT sum to the planned budget B "
                      f"(charter §6.5); fix the pool split before spending GPU time")
                ok = False
    if not ok:
        return 1
    if len(readings) < 2:
        print(f"  [note] single-engine scope ({raw!r}): pairwise parity is "
              f"vacuous; realized pool recorded above. The FULL §6.5 gate "
              f"needs all final-scope engines in CAGE_PREFLIGHT_BACKENDS.")
        print("  [PASS] iso-bytes gate (vacuous parity; realized pool recorded)")
        return 0
    for i in range(len(readings)):
        for j in range(i + 1, len(readings)):
            ra, rb = readings[i], readings[j]
            try:
                basis, gap, within = compare_pair(ra, rb, tol)
            except IsoBytesError as exc:
                print(f"  [FAIL] {exc}")
                ok = False
                continue
            proxy = (" (PROXY basis: tokens certify §6.5 only at uniform KV "
                     "dtype)" if basis == "tokens" else "")
            if within:
                print(f"  [PASS] {ra['engine']} vs {rb['engine']}: {basis} gap "
                      f"{gap:.4f} <= tol {tol}{proxy}")
            else:
                print(f"  [FAIL] {ra['engine']} vs {rb['engine']}: {basis} gap "
                      f"{gap:.4f} > tol {tol} -- realized KV pools are NOT "
                      f"iso-bytes; fix the dial mapping in "
                      f"scripts/lib/_serving_config.sh before spending GPU "
                      f"time{proxy}")
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
gate_rc $?

# ---------------------------------------------------------------------------
# (k) per-backend endpoint liveness (J11 fix, task #138).
# CAGE-BACKEND-ENDPOINTS-GATE: every scoped backend's OpenAI surface is
# probed. Engines serve SERIALLY on one GPU, so a down endpoint is a loud
# note (its launcher + gate (j) cover it at its own start), but ZERO live
# endpoints means no engine is up at all -> FAIL.
# ---------------------------------------------------------------------------
echo "(k) per-backend endpoint liveness (scope: $PREFLIGHT_BACKENDS)"
_live_count=0
for _b in $(printf '%s' "$PREFLIGHT_BACKENDS" | tr ',' ' '); do
    case "$_b" in
        vllm) _url="$API_BASE" ;;
        sglang) _url="http://localhost:${SGLANG_PORT:-30000}" ;;
        lmdeploy|lmdeploy-turbomind) _url="http://localhost:${LMDEPLOY_PORT:-23333}" ;;
        *) fail "backend '$_b' has no known endpoint mapping (add it here alongside its adapter)"; continue ;;
    esac
    if curl -fsS "$_url/v1/models" >/dev/null 2>&1; then
        pass "backend '$_b' live at $_url"
        _live_count=$((_live_count + 1))
    else
        echo "  [note] backend '$_b' not serving at $_url (engines serve serially; covered by its launcher + gate (j) at its start)"
    fi
done
if [ "$_live_count" -eq 0 ]; then
    fail "no scoped backend endpoint is live (backends=$PREFLIGHT_BACKENDS) -- start the engine under test before preflight"
fi

# ---------------------------------------------------------------------------
# (l) campaign-layout round-trip (task #138 gate c1; merges #118's S0 item).
# ---------------------------------------------------------------------------
echo "(l) campaign-layout round-trip (v2 producer -> seal -> organizer)"
python3 - <<'PY'
# CAGE-CAMPAIGN-LAYOUT-GATE (task #138 gate c1; merges #118). Synthesizes ONE
# minimal-but-complete §1 cell via the production library
# (src/orchestration/campaign_layout.py) in a throwaway tempdir, seals it, and
# asserts scripts/4_analysis/organize_results.py ACCEPTS it -- producer and
# reader must compose BEFORE GPU time is spent writing a tree the organizer
# would reject. Pure-python: also executed for real by
# tests/test_preflight_gates.py.
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts" / "4_analysis"))

try:
    import organize_results as org
    from src.analysis.cellspec import CellSpec
    from src.orchestration import campaign_layout as cl

    _ZOH = [
        {"ts_s": 2.0, "kv_cache_usage": 0.5, "preemptions_total": 5},
        {"ts_s": 6.0, "kv_cache_usage": 1.0, "preemptions_total": 5},
        {"ts_s": 8.0, "kv_cache_usage": 0.8, "preemptions_total": 9},
    ]
    with tempfile.TemporaryDirectory(prefix="cage_preflight_layout_") as td:
        run_id = "20260818-0000-a-qwen3-14b"
        run_root = Path(td) / "results" / "preflight" / "a" / run_id
        run = cl.CampaignRun.create(
            run_root,
            campaign="preflight",
            session="a",
            run_id=run_id,
            model="qwen3-14b",
            engine="vllm",
            engine_version="preflight-probe",
            seed=1,
            provider="preflight",
            hardware="preflight-gate",
            dataset_manifests_sha256="0" * 64,
            cellspec_schema_version=1,
            # Synthetic provenance: this gate probes the LIBRARY round-trip,
            # not the checkout's git state.
            git_provenance=lambda _repo: ("0" * 40, False),
        )
        cell = run.cell(CellSpec.from_baseline("B1", model="qwen3-14b"))
        handle = cell.add_window(
            "squad_v2",
            seed=1,
            rep=1,
            t_start=0.0,
            t_end=10.0,
            requests=[{"example_id": "sq-0", "ttft_ms": 100.0}],
            cage_stats=list(_ZOH),
            engine_metrics={"snapshot": "preflight"},
            qa_evidence=[{"example_id": "sq-0", "generated_answer": "x"}],
        )
        cl.write_window_regime(handle.window_dir, t_start=0.0, t_end=10.0)
        run.seal()
        csv_path, md_path = org.organize_run(run.run_root)
        if not Path(csv_path).is_file() or not Path(md_path).is_file():
            raise RuntimeError(
                f"organizer reported {csv_path} / {md_path} but wrote no file")
except Exception as exc:  # loud: ANY break in the produce->seal->organize chain
    print(f"  [FAIL] campaign-layout round-trip broke: {type(exc).__name__}: {exc}")
    sys.exit(1)
print("  [PASS] campaign_layout produced a sealed cell 1 and organize_results indexed it")
sys.exit(0)
PY
gate_rc $?

# ---------------------------------------------------------------------------
# (m) open-loop schedule + measured-replay guard (task #138 gate c2; #118).
# ---------------------------------------------------------------------------
echo "(m) open-loop schedule + measured-replay guard smoke (D6 6.1)"
python3 - <<'PY'
# CAGE-OPENLOOP-GATE (task #138 gate c2; merges #118). Proves the D6 §6.1
# load generator END-TO-END offline: a seeded schedule draws
# deterministically, its manifest is recordable, and the E4 measured-replay
# guard ENGAGES on a wrapping schedule. Pure-python; also executed by
# tests/test_preflight_gates.py.
import sys

try:
    from src.orchestration.load_generator import (
        LoadGeneratorError,
        ensure_no_measured_replay,
        generate_arrival_schedule,
    )

    a = generate_arrival_schedule(4.0, seed=20260818, duration_s=5.0)
    b = generate_arrival_schedule(4.0, seed=20260818, duration_s=5.0)
    if a.offsets_s != b.offsets_s:
        raise RuntimeError("same-seed schedules diverged (non-deterministic draw)")
    manifest = a.to_manifest()
    for key in ("rate_qps", "seed", "distribution", "n_arrivals", "bit_generator"):
        if key not in manifest:
            raise RuntimeError(f"schedule manifest lost provenance key {key!r}")
    fixed = generate_arrival_schedule(4.0, seed=1, n_requests=8)
    if ensure_no_measured_replay(fixed, 8) is not False:
        raise RuntimeError("replay guard flagged a schedule that FITS its request set")
    try:
        ensure_no_measured_replay(fixed, 4)
    except LoadGeneratorError:
        pass  # the guard ENGAGED -- exactly the E4 fail-closed contract
    else:
        raise RuntimeError("replay guard did NOT engage on a wrapping schedule (E4 pin broken)")
except Exception as exc:
    print(f"  [FAIL] open-loop smoke broke: {type(exc).__name__}: {exc}")
    sys.exit(1)
print(f"  [PASS] open-loop schedule deterministic ({a.n_arrivals} arrivals in 5s @ 4 qps) and the replay guard engages")
sys.exit(0)
PY
gate_rc $?

# ---------------------------------------------------------------------------
# (n) calibration artifact presence/shape (task #138 gate c3; #118).
# ---------------------------------------------------------------------------
echo "(n) calibration artifact (cal-v1) presence/shape"
python3 - <<'PY'
# CAGE-CALIBRATION-ARTIFACT-GATE (task #138 gate c3; merges #118). Once
# scripts/3_run/calibrate_cell.py has minted cal-v1 artifacts, every declared
# manifest must exist, parse, and carry the CellCalibration.to_manifest shape
# (src/orchestration/calibration.py). Pre-calibration the gate SKIPS with an
# explicit reason -- declared-but-missing is a FAILURE, never a skip.
import glob
import json
import math
import os
import sys

raw = os.environ.get("CAGE_CALIBRATION_MANIFESTS", "").strip()
if not raw:
    print("  [SKIP] pre-calibration: no cal-v1 manifest declared (set "
          "CAGE_CALIBRATION_MANIFESTS=<path-or-glob[,...]> once "
          "scripts/3_run/calibrate_cell.py has run at S0)")
    sys.exit(3)
paths = []
for entry in raw.split(","):
    entry = entry.strip()
    if entry:
        paths.extend(sorted(glob.glob(entry)))
if not paths:
    print(f"  [FAIL] CAGE_CALIBRATION_MANIFESTS={raw!r} matched NO files -- a "
          f"declared calibration artifact must exist (unset the variable if "
          f"calibration has not run yet)")
    sys.exit(1)
REQUIRED = ("procedure_version", "model", "engine", "budget_fraction",
            "procedure", "confirmatory", "floor", "lambda_star")
ok = True
for p in paths:
    try:
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
    except Exception as exc:
        print(f"  [FAIL] {p}: unparseable calibration JSON ({exc})")
        ok = False
        continue
    missing = [k for k in REQUIRED if k not in doc]
    if missing:
        print(f"  [FAIL] {p}: missing required cal-v1 keys {missing}")
        ok = False
        continue
    problems = []
    if not str(doc["procedure_version"]).startswith("cal-v1"):
        problems.append(
            f"procedure_version={doc['procedure_version']!r} is not cal-v1")
    if doc["confirmatory"] is not False:
        problems.append("confirmatory must be False (calibration data NEVER "
                        "enters confirmatory analysis)")
    bf = doc["budget_fraction"]
    if (not isinstance(bf, (int, float)) or isinstance(bf, bool)
            or not math.isfinite(bf) or bf <= 0):
        problems.append(f"budget_fraction={bf!r} is not positive finite")
    for section in ("floor", "lambda_star"):
        if not isinstance(doc[section], dict) or not doc[section]:
            problems.append(f"{section} is not a non-empty object")
    if problems:
        print(f"  [FAIL] {p}: " + "; ".join(problems))
        ok = False
    else:
        print(f"  [PASS] {p}: cal-v1 shape OK (model={doc['model']} "
              f"engine={doc['engine']} budget_fraction={bf})")
sys.exit(0 if ok else 1)
PY
gate_rc $?

# ---------------------------------------------------------------------------
# (o) regime-inputs bridge on live telemetry (task #138 gate c4; #118).
# ---------------------------------------------------------------------------
echo "(o) regime-inputs bridge on live telemetry (live-only)"
python3 - "$API_BASE" <<'PY'
# CAGE-REGIME-BRIDGE-GATE (task #138 gate c4; merges #118). LIVE-ONLY: samples
# the serving engine's Prometheus /metrics a few times and pushes the series
# through src/analysis/regime_inputs.compute_window_regime_inputs -- the exact
# bridge the campaign uses to certify rho_KV/scarcity per window. No server ->
# SKIP with the reason (executed for real at S0); a live server whose
# telemetry the bridge REFUSES is a FAILURE.
import os
import re
import sys
import time
import urllib.request

api = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
n = int(os.environ.get("CAGE_REGIME_GATE_SAMPLES", "5"))
interval = float(os.environ.get("CAGE_REGIME_GATE_INTERVAL", "1.0"))
kv_metric = os.environ.get("CAGE_REGIME_KV_METRIC", "vllm:gpu_cache_usage_perc")
pre_metric = os.environ.get("CAGE_REGIME_PREEMPT_METRIC", "vllm:num_preemptions_total")
if n < 2 or interval <= 0:
    print(f"  [FAIL] CAGE_REGIME_GATE_SAMPLES={n} / CAGE_REGIME_GATE_INTERVAL="
          f"{interval}: need >= 2 samples at a positive interval")
    sys.exit(1)


def scrape():
    with urllib.request.urlopen(f"{api}/metrics", timeout=5) as resp:
        return resp.read().decode("utf-8", "replace")


def metric_sum(text, name):
    """Sum every sample of one metric family (label sets summed); absent -> KeyError."""
    total = None
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^\n]*\}})?\s+(-?[0-9.eE+]+|NaN)\s*$")
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            total = (total or 0.0) + float(m.group(1))
    if total is None:
        raise KeyError(name)
    return total


try:
    text = scrape()
except Exception as exc:
    print(f"  [SKIP] live-only gate: no server at {api}/metrics "
          f"({type(exc).__name__}: {exc}) -- executed for real at S0 with an "
          f"engine up")
    sys.exit(3)

rows = []
try:
    for i in range(n):
        if i:
            time.sleep(interval)
            text = scrape()
        rows.append({
            "ts_s": time.monotonic(),
            "kv_cache_usage": metric_sum(text, kv_metric),
            "preemptions_total": metric_sum(text, pre_metric),
        })
except KeyError as exc:
    print(f"  [FAIL] live /metrics exposes no {exc.args[0]!r} -- the regime "
          f"bridge cannot certify rho_KV/scarcity from this engine (override "
          f"names via CAGE_REGIME_KV_METRIC / CAGE_REGIME_PREEMPT_METRIC)")
    sys.exit(1)
except Exception as exc:
    print(f"  [FAIL] telemetry sampling broke mid-gate: {type(exc).__name__}: {exc}")
    sys.exit(1)

import pandas as pd
from src.analysis.regime_inputs import RegimeInputError, compute_window_regime_inputs

try:
    out = compute_window_regime_inputs(
        pd.DataFrame(rows), rows[0]["ts_s"], rows[-1]["ts_s"] + interval)
except RegimeInputError as exc:
    print(f"  [FAIL] regime bridge REFUSED the live sample window: {exc}")
    sys.exit(1)
print(f"  [PASS] regime bridge certified live telemetry: {out.to_flat_dict()}")
sys.exit(0)
PY
gate_rc $?

# ---------------------------------------------------------------------------
# (p) dataset staleness refusal (task #138 gate e).
# ---------------------------------------------------------------------------
echo "(p) dataset staleness refusal (requested charter datasets staged on disk)"
python3 - <<'PY'
# CAGE-DATASET-STALENESS-GATE (task #138 gate e; K-led4 fix, task #141). A run
# request naming a charter dataset that is NOT staged on disk must REFUSE here
# -- never launch and silently serve a subset. Requested set: CAGE_DATASETS
# (comma-separated roster keys) falling back to the runner's exported DATASET.
# K-led4 (Topic 12): nothing in the launch path exports either variable, so the
# old exit-3 skip default meant an operator got NO staleness protection
# silently. With neither set, this gate now evaluates the FULL default
# campaign roster (download_datasets.py CAMPAIGN_KEYS -- the charter D5 set
# that `--dataset all` stages) and says so loudly; it never silently skips.
# Staging ground truth: the HF datasets cache filled by
# scripts/1_setup/download_datasets.py (HF_DATASETS_CACHE > HF_HOME/datasets >
# ~/.cache/huggingface/datasets); the cache-directory layout convention is
# [VERIFY-LIVE at S0].
import importlib.util
import os
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "cage_download_datasets",
    Path.cwd() / "scripts" / "1_setup" / "download_datasets.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as exc:
    print(f"  [FAIL] cannot load the staging roster "
          f"(scripts/1_setup/download_datasets.py): {exc}")
    sys.exit(1)

raw = os.environ.get("CAGE_DATASETS") or os.environ.get("DATASET") or ""
requested = [t.strip() for t in raw.split(",") if t.strip()]
if not requested:
    requested = list(mod.CAMPAIGN_KEYS)
    print("  [INFO] no explicit dataset request (CAGE_DATASETS/DATASET unset) "
          f"-- evaluating the DEFAULT campaign roster {requested} instead of "
          "skipping (K-led4: this gate never silently skips)")

known = set(mod.CAMPAIGN_KEYS) | set(mod.CALIBRATION_KEYS) | {"ruler"}
unknown = [k for k in requested if k not in known]
if unknown:
    print(f"  [FAIL] requested dataset key(s) {unknown} are not in the charter "
          f"roster {sorted(known)} -- refusing (a typo here would silently "
          f"drop a dataset)")
    sys.exit(1)

cache_env = os.environ.get("HF_DATASETS_CACHE")
if cache_env:
    cache = Path(cache_env)
else:
    hf_home = os.environ.get("HF_HOME")
    cache = (Path(hf_home) if hf_home
             else Path.home() / ".cache" / "huggingface") / "datasets"


def staged(hf_path):
    d = cache / hf_path.replace("/", "___")
    return d.is_dir() and any(d.iterdir())


specs = mod.dataset_specs()
missing = []
for key in requested:
    if key == "ruler":
        # Synthetic instrument (src/data/ruler.py): generated at run time,
        # never staged -- charter D5 item 5.
        print("  [PASS] dataset 'ruler' is synthetic (generated, never staged)")
        continue
    absent = [p + (f":{c}" if c else "") for p, c in specs[key] if not staged(p)]
    if absent:
        missing.append(f"{key} (HF: {', '.join(absent)})")
    else:
        print(f"  [PASS] dataset '{key}' staged in {cache}")
if missing:
    print(f"  [FAIL] REFUSING: requested charter dataset(s) NOT staged on disk "
          f"-- {'; '.join(missing)}. No silent subset: stage with "
          f"scripts/1_setup/download_datasets.py --dataset <key> (cache: {cache})")
    sys.exit(1)
sys.exit(0)
PY
gate_rc $?

# ---------------------------------------------------------------------------
# (q) cage-stats pin parity (task #143, finding L-A CRITICAL).
# The E2b lesson: an installed cage-stats one commit behind the requirements
# pin fabricated rho_KV=0.0 from an ABSENT occupancy gauge -- run-validity
# critical and invisible to the test suite (which runs on fixtures /
# CAGE_STATS_HOME, never the installed dist). This gate proves the venv's
# INSTALLED cage-stats commit equals the requirements.txt pinned SHA.
# Fail-closed: missing install, unparseable pin, or no recorded install
# commit all FAIL (cage-stats is required infra). Never exits 3.
# ---------------------------------------------------------------------------
echo "(q) cage-stats pin parity (requirements.txt pin == installed commit)"
python3 - "$PROJECT_DIR/requirements.txt" <<'PY'
# CAGE-STATS-PIN-PARITY-GATE (task #143, finding L-A). Extracted and exercised
# for real by tests/test_preflight_gates.py (parse fixtures + end-to-end runs
# against the live venv and synthetic requirements files).
#
# Installed-commit evidence channels, all recorded AT INSTALL TIME (this gate
# refuses to guess -- no channel, or disagreeing channels, is a FAILURE):
#   1. PEP 610 direct_url.json vcs_info.commit_id -- pip's own record for a
#      `git+...@sha` install (the S0/GPU-box channel).
#   2. `pip freeze` fallback: the `cage-stats @ git+...@sha` line (derived
#      from the same PEP 610 record; kept as a belt-and-braces channel).
#   3. PEP 440 local version `+g<full sha>` -- minted from `git rev-parse` by
#      the offline local-wheel build (no-network dev boxes, task #143).
import importlib.metadata as md
import json
import re
import subprocess
import sys
from pathlib import Path

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_PIN = re.compile(r"^cage-stats\s*@\s*git\+\S+@([0-9A-Fa-f]+)\s*$")
_FREEZE = re.compile(r"^cage-stats\s*@\s*git\+\S+@([0-9A-Fa-f]{40})")
_LOCAL_G = re.compile(r"\+g([0-9a-f]{40})$")


class PinParityError(ValueError):
    """Loud parse failure -- the pin-parity gate must never guess."""


def pinned_sha(requirements_text):
    """The full 40-char SHA pinned for cage-stats in requirements.txt.

    Exactly one non-comment `cage-stats @ git+...@<sha>` line must exist and
    its SHA must be full-length: a missing, duplicated, or short/branch pin is
    unparseable and REFUSED (a short pin cannot be compared to an installed
    commit without guessing)."""
    hits = []
    for raw in requirements_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("cage-stats"):
            hits.append(line)
    if not hits:
        raise PinParityError(
            "requirements.txt has NO cage-stats pin line -- cage-stats is "
            "required infra (rho_KV telemetry); restore the "
            "`cage-stats @ git+...@<full-sha>` pin")
    if len(hits) > 1:
        raise PinParityError(
            f"requirements.txt has {len(hits)} cage-stats lines -- exactly one "
            f"pin must exist (found: {hits!r})")
    m = _PIN.match(hits[0])
    if not m:
        raise PinParityError(
            f"cage-stats pin is not a parseable `cage-stats @ git+...@<sha>` "
            f"line: {hits[0]!r}")
    sha = m.group(1).lower()
    if not _SHA40.match(sha):
        raise PinParityError(
            f"cage-stats pin ref {m.group(1)!r} is not a FULL 40-char commit "
            f"SHA -- short/branch refs are ambiguous; pin the full SHA")
    return sha


def commit_from_direct_url(doc):
    """PEP 610 vcs_info.commit_id (lowered), or None when not a vcs install."""
    if not isinstance(doc, dict):
        return None
    commit = doc.get("vcs_info", {}).get("commit_id")
    return commit.lower() if isinstance(commit, str) and commit else None


def commit_from_version(version):
    """Full SHA from a `+g<40-hex>` PEP 440 local segment, or None."""
    m = _LOCAL_G.search(version)
    return m.group(1) if m else None


def commit_from_freeze_line(line):
    """Full SHA from a `cage-stats @ git+...@<sha>` pip-freeze line, or None."""
    m = _FREEZE.match(line.strip())
    return m.group(1).lower() if m else None


def installed_commits(dist, freeze_lines=None):
    """{channel_name: sha} for every evidence channel present on the dist.

    `freeze_lines` (an iterable of pip-freeze output lines) is only consulted
    when the direct_url channel yields nothing -- pip freeze derives its git
    line from the same PEP 610 record."""
    channels = {}
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        raw = None
    if raw:
        try:
            doc = json.loads(raw)
        except ValueError as exc:
            raise PinParityError(f"installed direct_url.json is corrupt: {exc}")
        commit = commit_from_direct_url(doc)
        if commit:
            channels["direct_url.vcs_info"] = commit
    local = commit_from_version(dist.version)
    if local:
        channels["version.+g"] = local
    if "direct_url.vcs_info" not in channels and freeze_lines is not None:
        for line in freeze_lines:
            commit = commit_from_freeze_line(line)
            if commit:
                channels["pip.freeze"] = commit
                break
    return channels


def main(argv):
    req_path = Path(argv[1]) if len(argv) > 1 else Path("requirements.txt")
    try:
        pin = pinned_sha(req_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"  [FAIL] cannot read {req_path}: {exc}")
        return 1
    except PinParityError as exc:
        print(f"  [FAIL] {exc}")
        return 1

    try:
        dist = md.distribution("cage-stats")
    except md.PackageNotFoundError:
        print(f"  [FAIL] cage-stats is NOT installed in this venv but "
              f"requirements.txt pins {pin} -- required infra (the rho_KV "
              f"telemetry source); install it at the pinned commit")
        return 1

    def freeze_lines():
        # Generator: the pip subprocess only runs if the fallback is consulted.
        res = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True)
        yield from (res.stdout.splitlines() if res.returncode == 0 else [])

    try:
        channels = installed_commits(dist, freeze_lines=freeze_lines())
    except PinParityError as exc:
        print(f"  [FAIL] {exc}")
        return 1
    if not channels:
        print(f"  [FAIL] installed cage-stats {dist.version} records NO "
              f"install commit (no direct_url vcs_info, no git+ pip-freeze "
              f"line, no +g<sha> local version) -- cannot certify it matches "
              f"the pinned {pin}; reinstall from the pinned commit")
        return 1
    if len(set(channels.values())) > 1:
        print(f"  [FAIL] installed cage-stats commit channels DISAGREE: "
              f"{channels} -- corrupted install metadata; reinstall from the "
              f"pinned {pin}")
        return 1
    channel, installed = next(iter(channels.items()))
    if installed != pin:
        print(f"  [FAIL] cage-stats pin parity: requirements.txt pins {pin} "
              f"but the venv has {installed} (channel: {channel}) -- a "
              f"lagging install fabricates telemetry (E2b: absent KV gauge "
              f"-> rho_KV=0.0); reinstall at the pinned commit before launch")
        return 1
    print(f"  [PASS] cage-stats installed commit == requirements.txt pin "
          f"({pin}; channel: {channel})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
gate_rc $?

# ---------------------------------------------------------------------------
# (r) engine blindness matrix (task T8.1; 2026-08-27 audit item S4).
# The S4 finding: the engine-x-telemetry-field "blindness matrix" existed only
# as a doc table -- no code produced it. This gate scrapes every scoped
# backend's live /metrics (same env-derived endpoints as gate (k)) and records
# presence/granularity for the telemetry fields cage-stats consumes, plus the
# adapter capabilities probe for the per-request cached-token channel.
# ABSENCE of a field is DATA (recorded "absent"), never a failure; the gate
# fails ONLY on an unreachable scoped backend (while the fleet is live) or an
# unwritable artifact. Fully offline -> exit 3 (live-only gate). Artifact:
# CAGE_BLINDNESS_OUT, default <CAGE_RUN_ROOT|results/preflight>/observability/
# blindness_matrix.json.
# ---------------------------------------------------------------------------
echo "(r) engine blindness matrix (scope: $PREFLIGHT_BACKENDS; out: ${CAGE_BLINDNESS_OUT:-<run observability dir>})"
python3 - "$PREFLIGHT_BACKENDS" "$API_BASE" <<'PY'
# CAGE-BLINDNESS-MATRIX-GATE (task T8.1; 2026-08-27 audit item S4). Extracted
# and executed by tests/test_preflight_gates.py against a loopback Prometheus
# stub. Engine blindness is a REPORTED ecosystem finding (charter: absence
# reported, never a silent hole), so "absent" cells are data; only an
# unreachable backend or an unwritable artifact fails the gate, and a fully
# offline preflight skips with reason (exit 3 -- this is a live-only gate).
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

#: Telemetry-field table -- the MECHANISM-CRITICAL subset of what cage-stats
#: consumes (cage_stats/metrics/engine.py + sglang_dialect.py at the pinned
#: commit) plus the per-request cached-token channel the vLLM adapter reads.
#: Deliberately NOT rows here (consumed, but not blindness-gated: every
#: engine exposes some spelling and their absence is not a mechanism hole):
#: running/waiting queue gauges, generation/prompt token counters,
#: ttft/tpot/e2e latency histograms, iteration_tokens_total,
#: cache_config_info.
#: Entry: (field_key, probe, candidates); a /metrics candidate is
#: (sample_name, kind, granularity) where kind:
#:   exact     -> the sample name itself appears in the scrape
#:   histogram -> <name>_bucket/_sum/_count appears (Prometheus histogram)
#:   prefix    -> any sample name starting with <name> appears
#: Candidates are ordered richest-granularity-first; the FIRST match names
#: the recorded granularity (every match is kept as evidence).
FIELDS = (
    # KV occupancy gauge: cage-stats engine.py kv_usage reads
    # vllm:kv_cache_usage_perc (multi-label-set = refusal there);
    # vllm:gpu_cache_usage_perc is the pre-rename spelling (gate (o)'s
    # default), sglang:token_usage the pre-translation SGLang dialect
    # (sglang_dialect.SGLANG_TO_VLLM).
    ("occupancy_gauge", "metrics", (
        ("vllm:kv_cache_usage_perc", "exact", "gauge (0..1 of the KV pool)"),
        ("vllm:gpu_cache_usage_perc", "exact", "gauge (0..1 of the KV pool)"),
        ("sglang:token_usage", "exact", "gauge (0..1 of the KV pool)"),
    )),
    # Preemption/retraction counter: engine.py preemptions_total; the SGLang
    # retraction spellings (sglang_dialect.RETRACTION_CANDIDATES, preference
    # order preserved) map onto vllm:num_preemptions_total.
    ("preemption_counter", "metrics", (
        ("vllm:num_preemptions_total", "exact", "cumulative counter"),
        ("sglang:num_retracted_reqs_total", "exact", "cumulative counter"),
        ("sglang:num_retracted_reqs", "exact", "cumulative counter"),
        ("sglang:retracted_reqs_total", "exact", "cumulative counter"),
        ("sglang:num_preempted_reqs", "exact", "cumulative counter"),
    )),
    # Raw prefix-cache QUERY counter (engine.py prefix_cache_queries_total):
    # needed for exact windowed hit ratios; SGLang exposes no raw query
    # counter that cage-stats models -- its absence here IS the blindness.
    ("prefix_query_counter", "metrics", (
        ("vllm:prefix_cache_queries_total", "exact", "cumulative counter"),
    )),
    # Raw prefix-cache HIT counter; sglang:cache_hit_rate is the explicitly
    # coarser alternate (a rate gauge -- no raw counts to diff per window).
    ("prefix_hit_counter", "metrics", (
        ("vllm:prefix_cache_hits_total", "exact", "cumulative counter"),
        ("sglang:cache_hit_rate", "exact", "rate gauge (coarser: no raw counts)"),
    )),
    # KV-transfer counters: engine.py external-KV attribution counters plus
    # the connector families _transfer_counters() captures. The prefix rows
    # MIRROR cage-stats engine.py exactly -- _TRANSFER_FAMILY_RE is
    # ^vllm:(kv_transfer|nixl|kv_connector) and _TRANSFER_DOC_PREFIXES is
    # ("nixl_",) -- so a backend exposing connector telemetry under ANY of
    # the four spellings cage-stats consumes is recorded present, never
    # false-absent in the audit artifact (VLLM_COMPATIBILITY.md sec. 8.4:
    # without these the transfer-cost instrumentation is blind).
    ("transfer_counters", "metrics", (
        ("vllm:external_prefix_cache_queries_total", "exact", "cumulative counter"),
        ("vllm:prompt_tokens_by_source_total", "exact", "cumulative counter by source"),
        ("vllm:kv_transfer", "prefix", "kv-transfer connector family"),
        ("vllm:kv_connector", "prefix", "kv-connector family"),
        ("vllm:nixl", "prefix", "nixl transfer family"),
        ("nixl_", "prefix", "nixl transfer family (bare doc spelling, sec. 8.4)"),
    )),
    # Per-phase request-time histograms (engine.py _PHASE_HIST): prefill-time
    # growth under prefix-cache eviction is the mechanism readout.
    ("phase_histograms", "metrics", (
        ("vllm:request_prefill_time_seconds", "histogram", "histogram (_sum/_count/_bucket)"),
        ("vllm:request_decode_time_seconds", "histogram", "histogram (_sum/_count/_bucket)"),
        ("vllm:request_inference_time_seconds", "histogram", "histogram (_sum/_count/_bucket)"),
        ("vllm:request_queue_time_seconds", "histogram", "histogram (_sum/_count/_bucket)"),
    )),
    # Aggregate cached-token counter (engine.py cached_tokens_total; a
    # missing series must not fabricate a zero -- same rule here: absent
    # stays absent).
    ("cached_token_aggregate", "metrics", (
        ("vllm:prompt_tokens_cached_total", "exact", "cumulative counter (aggregate)"),
    )),
    # Per-request cached-token field (usage.prompt_tokens_details.
    # cached_tokens): NOT scrapeable from /metrics. The capabilities probe
    # that exists is the adapter's capabilities()["cached_token_telemetry"]
    # declaration: True = verified in this codebase, "verify-live" =
    # documented upstream only (recorded as such, NEVER coerced to present),
    # anything else = absent.
    ("cached_token_per_request", "capabilities", ()),
)

_NAME = re.compile(r"([A-Za-z_:][A-Za-z0-9_:]*)")


def sample_names(text):
    """The set of Prometheus sample names in one /metrics exposition."""
    names = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _NAME.match(line)
        if m:
            names.add(m.group(1))
    return names


def match_field(names, candidates):
    """-> {'status', 'granularity', 'matched', 'source'} for one metrics field.

    status 'present' when at least one candidate matched; 'absent' is DATA
    (the scrape ran and the field is not exposed), never a failure."""
    matched = []
    for cand, kind, gran in candidates:
        if kind == "exact":
            hit = cand in names
            label = cand
        elif kind == "histogram":
            hit = any(f"{cand}{sfx}" in names
                      for sfx in ("_bucket", "_sum", "_count"))
            label = cand
        elif kind == "prefix":
            hit = any(n.startswith(cand) for n in names)
            label = f"{cand}*"
        else:  # a typo in the FIELDS table is a code bug -- loud, never skipped
            raise ValueError(f"unknown candidate kind {kind!r}")
        if hit:
            matched.append({"name": label, "granularity": gran})
    return {
        "status": "present" if matched else "absent",
        "granularity": matched[0]["granularity"] if matched else None,
        "matched": matched,
        "source": "metrics-scrape",
    }


def capabilities_field(backend, adapters):
    """Per-request cached-token row from the adapter capabilities() probe."""
    caps = adapters[backend](
        model_name="preflight-probe", api_base="http://localhost:1"
    ).capabilities()
    declared = caps.get("cached_token_telemetry")
    if declared is True:
        status = "present"
    elif declared == "verify-live":
        # [VERIFY-LIVE at Run-C-prime preflight]: documented upstream,
        # unverified in this codebase -- surfaced as its own status.
        status = "verify-live"
    else:
        status = "absent"
    return {
        "status": status,
        "granularity": "per-request usage field (adapter-declared)",
        "matched": [],
        "source": "capabilities-probe",
        "declared": repr(declared),
    }


def endpoint_for(backend, api_base):
    """Same env-derived endpoint mapping as gate (k); None = unknown token."""
    if backend == "vllm":
        return api_base
    if backend == "sglang":
        return f"http://localhost:{os.environ.get('SGLANG_PORT', '30000')}"
    if backend == "lmdeploy":
        return f"http://localhost:{os.environ.get('LMDEPLOY_PORT', '23333')}"
    return None


def out_path():
    """CAGE_BLINDNESS_OUT wins; default = the run's observability dir."""
    explicit = os.environ.get("CAGE_BLINDNESS_OUT")
    if explicit:
        return Path(explicit)
    root = os.environ.get("CAGE_RUN_ROOT")
    base = Path(root) if root else Path("results/preflight")
    return base / "observability" / "blindness_matrix.json"


def cell_text(entry):
    if entry is None:
        return "unreachable"
    if entry["status"] == "present":
        return f"present ({entry['granularity']}): " + ",".join(
            m["name"] for m in entry["matched"])
    if entry["status"] == "verify-live":
        return "verify-live (adapter-declared, unverified)"
    return "absent"


def main(argv):
    raw = argv[1] if len(argv) > 1 else "vllm,sglang,lmdeploy"
    api_base = (argv[2] if len(argv) > 2 else "http://localhost:8000").rstrip("/")
    backends = []
    for token in raw.split(","):
        token = token.strip()
        token = "lmdeploy" if token == "lmdeploy-turbomind" else token
        if token and token not in backends:
            backends.append(token)

    # Adapter import is venv infrastructure (gate (h) proves the same path):
    # a broken import is a real failure even offline, never a silent skip.
    try:
        from src.inference.lmdeploy_adapter import LMDeployAdapter
        from src.inference.sglang_adapter import SGLangAdapter
        from src.inference.vllm_adapter import VLLMAdapter
    except Exception as exc:
        print(f"  [FAIL] could not import adapter classes for the "
              f"capabilities probe: {exc}")
        return 1
    adapters = {"vllm": VLLMAdapter, "sglang": SGLangAdapter,
                "lmdeploy": LMDeployAdapter}

    ok = True
    matrix = {}
    unreachable = []
    for backend in backends:
        url = endpoint_for(backend, api_base)
        if url is None or backend not in adapters:
            print(f"  [FAIL] backend '{backend}' has no known endpoint/adapter "
                  f"mapping (add it here alongside gate (k))")
            ok = False
            continue
        try:
            with urllib.request.urlopen(f"{url}/metrics", timeout=5) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception as exc:
            matrix[backend] = {"endpoint": url, "reachable": False,
                               "fields": {}}
            unreachable.append((backend, url, f"{type(exc).__name__}: {exc}"))
            continue
        names = sample_names(text)
        fields = {}
        for key, probe, candidates in FIELDS:
            if probe == "metrics":
                fields[key] = match_field(names, candidates)
            else:
                fields[key] = capabilities_field(backend, adapters)
        matrix[backend] = {"endpoint": url, "reachable": True,
                           "fields": fields}

    reachable = [b for b in matrix if matrix[b]["reachable"]]
    if not reachable and ok:
        # No fabricated matrix from thin air: with NO live backend there is
        # nothing to record -- this is a live-only gate.
        print(f"  [SKIP] live-only gate: no scoped backend reachable "
              f"(scope: {raw!r}) -- the blindness matrix records what a LIVE "
              f"/metrics exposes; executed for real at Run-C-prime preflight")
        return 3

    doc = {
        "schema": "cage-blindness-matrix-v1",
        "generated_at_unix": time.time(),
        "scope": backends,
        "fields": [key for key, _, _ in FIELDS],
        "matrix": matrix,
    }
    out = out_path()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    except OSError as exc:
        print(f"  [FAIL] blindness-matrix artifact cannot be written to "
              f"{out}: {exc} (override via CAGE_BLINDNESS_OUT)")
        return 1

    # Rendered markdown table (rows = fields, columns = scoped backends).
    cols = [b for b in backends if b in matrix]
    print("  [matrix] engine x telemetry-field blindness matrix "
          "(absent cells are DATA, not failures):")
    print(f"| field | {' | '.join(cols)} |")
    print(f"|---|{'---|' * len(cols)}")
    for key, _, _ in FIELDS:
        cells = [cell_text(matrix[b]["fields"].get(key)
                           if matrix[b]["reachable"] else None)
                 for b in cols]
        print(f"| {key} | {' | '.join(cells)} |")
    print(f"  [artifact] {out}")

    for backend, url, why in unreachable:
        print(f"  [FAIL] backend '{backend}' unreachable at {url}/metrics "
              f"({why}) while the fleet is live -- scope down "
              f"CAGE_PREFLIGHT_BACKENDS to the serving engine ONLY as a "
              f"recorded deviation")
        ok = False
    if ok:
        print(f"  [PASS] blindness matrix recorded for "
              f"{', '.join(reachable)} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
gate_rc $?

# ---------------------------------------------------------------------------
# (s) engine x model gate-matrix runner (task T8.3).
# The 2026-08-27 audit: the VLLM_COMPATIBILITY.md section-7 4x4 VERIFY-LIVE
# cell gates existed only as a doc table -- no code ran them. This gate holds
# the declarative engine x check table and runs every applicable cell:
# PASS (proven here) / FAIL (proven broken) / PENDING-VERIFY-LIVE (offline,
# backend not serving, or upstream-documented-only -- NEVER upgraded to a
# fabricated PASS). rc: 1 only on an actual FAIL, 3 when every applicable
# check is PENDING (reason printed), 0 when at least one check ran and none
# failed. The fp8xprefix cell delegates to
# scripts/checks/check_fp8_prefix_cache.sh (opt-in: it RESTARTS vLLM).
# ---------------------------------------------------------------------------
echo "(s) engine x model gate matrix (scope: $PREFLIGHT_BACKENDS; docs/VLLM_COMPATIBILITY.md section 7)"
python3 - "$PREFLIGHT_BACKENDS" "$API_BASE" "$MODEL" <<'PY'
# CAGE-ENGINE-MODEL-GATE-MATRIX (task T8.3; docs/VLLM_COMPATIBILITY.md
# section 7). Extracted and executed by tests/test_preflight_gates.py.
# Discipline: a check that cannot run HERE reports PENDING-VERIFY-LIVE with
# its reason -- pending is surfaced in behavior, never silently defaulted to
# PASS (fail-closed doctrine; every section-7 cell is VERIFY-LIVE until its
# gate has run on the provisioned node at the pinned version).
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
PENDING = "PENDING-VERIFY-LIVE"

CHECKS = ("deterministic_mode", "turbomind_selected", "fp8_prefix_coexist",
          "cached_token_telemetry")

#: Declarative engine x check table, sourced from VLLM_COMPATIBILITY.md
#: section 7 (True = the check gates that engine's campaign cells, False =
#: not applicable -- an N/A cell is reported, never silently omitted):
#:   deterministic_mode -- sec. 7 SGLang pin row: "deterministic-mode
#:     availability (T=0 reproducible sampling) per model"; the vLLM Group-C
#:     cell carries the sec. 5.2 "mandatory T=0 token-identity smoke", whose
#:     precondition is T=0 repeatability, so vLLM probes too. LMDeploy has no
#:     sec.-7 deterministic row -> N/A.
#:   turbomind_selected -- sec. 7 LMDeploy pin row: "TurboMind backend only
#:     ... TurboMind is actually selected (not the silent PyTorch-engine
#:     fallback)" (charter P7). vLLM/SGLang -> N/A.
#:   fp8_prefix_coexist -- sec. 2 flag matrix + sec. 4: the compressed_cag
#:     lever (--kv-cache-dtype fp8 x --enable-prefix-caching) is vLLM-only;
#:     sec. 7 vLLM column D additionally flags fp8-KV-on-MLA [VERIFY-LIVE].
#:     Delegates to scripts/checks/check_fp8_prefix_cache.sh.
#:   cached_token_telemetry -- sec. 7 row A gates: vLLM "prefix telemetry
#:     (cached_tokens)"; SGLang "radix-cache telemetry (per-request
#:     granularity [VERIFY-LIVE])"; LMDeploy "blocked-KV pressure counters
#:     (weakest documented telemetry -- a failed cache-telemetry gate ->
#:     serving-only or excluded cells)".
TABLE = {
    "vllm": {"deterministic_mode": True, "turbomind_selected": False,
             "fp8_prefix_coexist": True, "cached_token_telemetry": True},
    "sglang": {"deterministic_mode": True, "turbomind_selected": False,
               "fp8_prefix_coexist": False, "cached_token_telemetry": True},
    "lmdeploy": {"deterministic_mode": False, "turbomind_selected": True,
                 "fp8_prefix_coexist": False, "cached_token_telemetry": True},
}

# TurboMind detection patterns MIRROR scripts/2_serving/
# manage_lmdeploy_server.sh assert_turbomind_selected (the P7 enforcement
# point; the launcher dies at start in strict mode -- this runner reports).
# The marker patterns themselves are [VERIFY-LIVE at Run-C-prime preflight].
_TM_FALLBACK = re.compile(
    r"fallback to pytorch|pytorch.{0,20}(engine|backend)|"
    r"(engine|backend).{0,20}pytorch", re.I)
_TM_POSITIVE = re.compile(
    r"\[TM\]|turbomind.{0,30}(engine|backend|model|start)|"
    r"(engine|backend).{0,30}turbomind", re.I)


def endpoint_for(backend, api_base):
    """Same env-derived endpoint mapping as gates (k)/(r)."""
    if backend == "vllm":
        return api_base
    if backend == "sglang":
        return f"http://localhost:{os.environ.get('SGLANG_PORT', '30000')}"
    if backend == "lmdeploy":
        return f"http://localhost:{os.environ.get('LMDEPLOY_PORT', '23333')}"
    return None


def check_deterministic(url, model):
    """Two identical T=0 seed-pinned completions must be byte-identical.

    Live-only: any transport/HTTP error is PENDING (with the reason), a real
    divergence between two identical requests is a FAIL."""
    def ask():
        body = json.dumps({"model": model,
                           "prompt": "Deterministic-mode probe: 2+2=",
                           "max_tokens": 8, "temperature": 0,
                           "seed": 20260902}).encode()
        req = urllib.request.Request(f"{url}/v1/completions", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)["choices"][0]["text"]

    try:
        a, b = ask(), ask()
    except Exception as exc:
        return (PENDING, f"live-only probe could not run at {url} "
                         f"({type(exc).__name__}: {exc}) "
                         f"[VERIFY-LIVE at Run-C-prime preflight]")
    if a == b:
        return (PASS, f"two T=0 seed-pinned completions byte-identical ({a!r})")
    return (FAIL, f"T=0 completions DIVERGED ({a!r} vs {b!r}) -- "
                  f"deterministic mode is NOT holding (sec. 7 pin row)")


def check_turbomind(log_root):
    """TurboMind-actually-selected from the newest LMDeploy launch log."""
    logs = sorted((log_root / "lmdeploy").glob("*.log"),
                  key=lambda p: p.stat().st_mtime)
    if not logs:
        return (PENDING, f"no LMDeploy launch log under {log_root / 'lmdeploy'}/ "
                         f"-- the launcher's assert_turbomind_selected runs at "
                         f"server start [VERIFY-LIVE at Run-C-prime preflight]")
    log = logs[-1]
    text = log.read_text(encoding="utf-8", errors="replace")
    if _TM_FALLBACK.search(text):
        return (FAIL, f"PyTorch-engine fallback marker in {log} -- the P7 "
                      f"never-mixed policy is violated (sec. 7 LMDeploy pin row)")
    if _TM_POSITIVE.search(text):
        return (PASS, f"TurboMind selection marker found in {log}")
    return (PENDING, f"no backend marker in {log} (the marker pattern itself "
                     f"is VERIFY-LIVE; the strict launcher is the enforcement "
                     f"point) [VERIFY-LIVE at Run-C-prime preflight]")


def check_fp8_prefix(model):
    """Delegate to check_fp8_prefix_cache.sh -- opt-in, it RESTARTS vLLM.

    Delegate rc mapping: 0 -> PASS, 1 -> FAIL (confound proven: fp8 KV turned
    prefix caching off), 2 -> PENDING (the script's own INCONCLUSIVE code),
    anything else -> FAIL (a crashed instrument is never a silent skip)."""
    script = Path(os.environ.get("CAGE_FP8_GATE_SCRIPT",
                                 "scripts/checks/check_fp8_prefix_cache.sh"))
    if os.environ.get("CAGE_RUN_FP8_GATE", "") != "1":
        return (PENDING, f"delegate {script} not run -- set CAGE_RUN_FP8_GATE=1 "
                         f"to launch it (GPU-only; it RESTARTS vLLM with fp8 "
                         f"KV) [VERIFY-LIVE at Run-C-prime preflight]")
    if not script.is_file():
        return (FAIL, f"CAGE_RUN_FP8_GATE=1 but delegate script {script} does "
                      f"not exist")
    proc = subprocess.run(["bash", str(script), model],
                          capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        print(f"    [fp8-delegate] {line}")
    if proc.returncode == 0:
        return (PASS, f"delegate {script.name} exit 0 (fp8 KV + prefix "
                      f"caching coexist)")
    if proc.returncode == 1:
        return (FAIL, f"delegate {script.name} exit 1 -- fp8 KV disables "
                      f"prefix caching: compressed_cag would be confounded "
                      f"(sec. 4)")
    if proc.returncode == 2:
        return (PENDING, f"delegate {script.name} exit 2 (INCONCLUSIVE: "
                         f"server/flag error) [VERIFY-LIVE at Run-C-prime "
                         f"preflight]")
    return (FAIL, f"delegate {script.name} crashed (exit {proc.returncode}): "
                  f"{proc.stderr.strip()[:200]}")


def check_cached_token(backend, adapters):
    """Cached-token telemetry from the adapter capabilities() declaration.

    True = verified in this codebase -> PASS; "verify-live" = documented
    upstream only -> PENDING (never coerced); anything else = the adapter
    declares NO cached-token channel -> FAIL (sec. 7: those cells go
    serving-only or excluded -- loud, so the operator scopes them)."""
    caps = adapters[backend](
        model_name="preflight-probe", api_base="http://localhost:1"
    ).capabilities()
    declared = caps.get("cached_token_telemetry")
    if declared is True:
        return (PASS, "adapter declares cached_token_telemetry=True "
                      "(verified in-codebase)")
    if declared == "verify-live":
        return (PENDING, "adapter declares 'verify-live' (documented "
                         "upstream, unverified here) [VERIFY-LIVE at "
                         "Run-C-prime preflight]")
    return (FAIL, f"adapter declares cached_token_telemetry={declared!r} -- "
                  f"no cached-token channel; scope those cells serving-only "
                  f"(sec. 7)")


def main(argv):
    raw = argv[1] if len(argv) > 1 else "vllm,sglang,lmdeploy"
    api_base = (argv[2] if len(argv) > 2 else "http://localhost:8000").rstrip("/")
    model = argv[3] if len(argv) > 3 else "Qwen/Qwen3-8B"
    backends = []
    for token in raw.split(","):
        token = token.strip()
        token = "lmdeploy" if token == "lmdeploy-turbomind" else token
        if token and token not in backends:
            backends.append(token)

    try:
        from src.inference.lmdeploy_adapter import LMDeployAdapter
        from src.inference.sglang_adapter import SGLangAdapter
        from src.inference.vllm_adapter import VLLMAdapter
    except Exception as exc:
        print(f"  [FAIL] could not import adapter classes: {exc}")
        return 1
    adapters = {"vllm": VLLMAdapter, "sglang": SGLangAdapter,
                "lmdeploy": LMDeployAdapter}

    # Same logs tree gate (j) discovers from (one env var, already part of
    # the gate-env contract).
    log_root = Path(os.environ.get("CAGE_ISO_BYTES_LOG_ROOT", "logs"))

    ok = True
    results = {}
    for backend in backends:
        if backend not in TABLE:
            print(f"  [FAIL] backend '{backend}' has no row in the sec.-7 "
                  f"engine x check table -- add it before scoping it in")
            ok = False
            continue
        row = {}
        for check in CHECKS:
            if not TABLE[backend][check]:
                row[check] = ("N/A", "not applicable (sec. 7)")
                continue
            if check == "deterministic_mode":
                row[check] = check_deterministic(
                    endpoint_for(backend, api_base), model)
            elif check == "turbomind_selected":
                row[check] = check_turbomind(log_root)
            elif check == "fp8_prefix_coexist":
                row[check] = check_fp8_prefix(model)
            else:
                row[check] = check_cached_token(backend, adapters)
        results[backend] = row

    print("  [matrix] engine x check gate matrix (sec. 7; PENDING is a "
          "pending check, never a pass):")
    print(f"| engine | {' | '.join(CHECKS)} |")
    print(f"|---|{'---|' * len(CHECKS)}")
    for backend, row in results.items():
        print(f"| {backend} | "
              + " | ".join(row[c][0] for c in CHECKS) + " |")
    for backend, row in results.items():
        for check in CHECKS:
            status, why = row[check]
            print(f"  [cell] {backend} x {check}: {status} -- {why}")

    statuses = [row[c][0] for row in results.values() for c in CHECKS]
    if any(s == FAIL for s in statuses):
        ok = False
    if not ok:
        return 1
    if any(s == PASS for s in statuses):
        print("  [PASS] gate matrix: at least one check ran and none failed "
              "(PENDING cells above re-run live)")
        return 0
    print(f"  [SKIP] every applicable sec.-7 check is {PENDING} (offline "
          f"preflight or engines not serving) -- re-run with the fleet up; "
          f"this gate never fabricates a PASS")
    return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
PY
gate_rc $?

echo "=============================================="
if [ "$FAILED" -eq 0 ]; then
    echo "PREFLIGHT PASS -- all Gate-2 components green. Safe to launch the sweep."
    exit 0
else
    echo "PREFLIGHT FAIL -- fix the [FAIL] items above before launching (do NOT spend GPU time)."
    exit 1
fi
