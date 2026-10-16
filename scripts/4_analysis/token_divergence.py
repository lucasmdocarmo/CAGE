#!/usr/bin/env python3
"""Order:     stage 4 — §8.9 divergence pass; pilot lane driven by run_phase2_stats.sh, campaign lane standalone
Objective: Quantify T=0 output divergence (agreement rate, first-divergence token position, answer-changing split)
           — pilot: per arm vs the reference arm on one engine; campaign (T6.4): per model x ENGINE-PAIR
Cloud:     local

Token-divergence metric: how often does an arm's greedy output differ from no_cache?

Greedy (T=0) decoding is NEAR-lossless, not identical, across serving configs: floating-point
non-associativity (prefix-cache reuse, eager-vs-compiled kernels, context-length changes) can
flip a near-tie argmax. This tool QUANTIFIES that -- for each baseline arm it compares the
generated answer to the reference arm's answer for the same (example_id, trial, repeat_index)
and reports the fraction that differ. That number is what lets the write-up say "prefix caching
is near-lossless (diverged on X% of queries)" instead of an unquantified "lossless", and it
bounds how much of any cross-config quality delta is token divergence rather than the mechanism.

Charter D8 sec. 8.9 statistics (AMENDED 2026-08-01; implemented 2026-08-04):
  1. exact-token AGREEMENT RATE per arm (``agreement_rate`` = 1 - raw divergence rate, plus
     ``token_agreement_rate`` under the configured tokenizer);
  2. FIRST-DIVERGENCE TOKEN POSITION between paired T=0 outputs (0-based index of the first
     differing token == length of the common token prefix), summarized per arm;
  3. ANSWER-CHANGING vs ANSWER-PRESERVING divergence -- the decisive one: did the divergence
     flip EM/F1/abstention, or only re-word? Extracted final answers are compared under the
     SAME normalization quality.py uses (``sanitize_answer`` + the official SQuAD v2
     ``QualityEvaluator.evaluate_f1_score`` normalization -- imported, never reimplemented).
     When the results CSV carries ``reference_answer`` (gold), both outputs are scored against
     gold and the classification compares EM/F1/abstention; without gold the two outputs are
     compared pairwise (abstention flags + official-normalization exact match).
  4. Per-cell REPRODUCIBILITY-VIOLATION RATE: across the >=2 (charter: >=3) repetitions of the
     same (example_id, trial) within one arm, continuous batching is not batch-invariant, so
     the same query at the same load can yield different answers; a group violates
     reproducibility when its repeats' answers are not all identical.

Input-effect labeling (audit 2026-07-16 S5): arms whose PROMPTS differ from the reference's
(e.g. rag's retrieved 3-doc context vs no_cache's gold paragraph) are flagged
``"input_effect": true`` -- their divergence measures different *inputs*, not engine
nondeterminism, and must not be read as a losslessness number. The flag is derived from the
loaded rows: same_prompt iff the arm's median prompt_tokens is within 5% of the reference's.

Interface mirrors statistical_tests.py:
    python scripts/4_analysis/token_divergence.py --results-dir results/<phase>/<run-id>/stats/all_results --reference no_cache \
        --output results/<phase>/<run-id>/stats/all_results/token_divergence.json

CAMPAIGN mode (T6.4 — the charter sec. 8.9 TARGET: per model x ENGINE-PAIR; the
pilot lane above compares ARMS on one engine):
    python scripts/4_analysis/token_divergence.py --campaign-root <root> \
        --out <dir> [--force] [--tokenizer ...]
walks a RESULTS_LAYOUT-v2 campaign tree (every run root holding
cells/<row_key>/window_<dataset>-NN/), groups windows by
(model, dataset, arm, grid-point, replicate), and computes the SAME sec. 8.9
statistics for EVERY engine pair serving a group. Pairs containing the HF
reference are the registered ORACLE comparison; engine-engine pairs without HF
are DeepSeek-V3's registered D4 substitute — labeled as a substitute, weaker
than an oracle, never silently promoted. Output: one JSON report per
(model, dataset) under --out (existing artifacts REFUSE without --force,
mirroring build_floor_table) plus a stdout summary table. Fail-closed lanes:
zero eligible groups is a loud error naming what was searched; mismatched
query sets across a group's engines and windows without generations are
LABELED skips (never a silent intersection, never a fabricated row).

CROSS-ARM lane (Spec Block A2, 2026-10-16; MyDocs/RunPod/S1_PARTS_AB_SPECS_2026-10-15.md):
    python scripts/4_analysis/token_divergence.py --campaign-root <root> --out <dir> \
        --cross-arm corpus-comp:corpus-reuse [--cross-arm CELL:REF ...]
compares two ARMS on the SAME engine for every (model, dataset, engine, grid-point,
replicate) both arms serve: the fp8 KV-cache arm (B10 corpus-comp) against its
bf16 twin (B3 corpus-reuse) is the registered pair. The statistics are the same
sec. 8.9 numbers (``_pair_stats``, cell arm on the a-side, reference arm on the
b-side). Each cross-arm record carries the input-effect gate from the pilot lane
(median prompt_tokens of the two sides within 5 percent, read from the windows'
requests.jsonl): ``input_effect`` False labels the pair "same inputs", True labels
it "different inputs" (the divergence then measures a prompt change, never the
cache effect), None labels it "unknown" (no prompt_tokens on the rows). A
registered pair that matches no group refuses loudly. The campaign lane also
writes ``token_divergence_summary.md`` beside the JSON reports.

Group key (both lanes): the corpus budget rung ``corpus_budget_tokens`` (ADR-0106,
the ``cb<N>`` row-key coordinate of B12) is part of the grid point since 2026-10-16;
before that two B12 budgets of one engine collided as an "ambiguous replicate".

RAW divergence = exact string mismatch after strip() (most sensitive).
NORMALIZED divergence = mismatch after lowercase + punctuation/article strip (whether the
    difference survives QA-style normalization, i.e. is a *meaningfully* different answer).

Tokenizer for first-divergence positions: ``--tokenizer whitespace`` (default: deterministic
``str.split()``, no dependency) or an HF tokenizer name (e.g. ``Qwen/Qwen3-8B``) to measure
positions in the serving model's real token ids. A requested HF tokenizer that cannot load
raises ``TokenizerUnavailableError`` (fail-closed, mirroring quality.py's
``InstrumentUnavailableError``) -- never a silent fallback to whitespace.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import statistics
import string
import sys
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, NamedTuple, Optional, Sequence, Tuple

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))  # generated answers can be long

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


class TokenizerUnavailableError(RuntimeError):
    """A requested tokenizer failed to load (fail-closed, no silent fallback).

    Mirrors ``src.evaluation.quality.InstrumentUnavailableError``: substituting the
    whitespace tokenizer for a requested model tokenizer would silently change what
    "first-divergence token position" means mid-analysis.
    """

    def __init__(self, tokenizer: str, cause: str) -> None:
        self.tokenizer = tokenizer
        self.cause = cause
        super().__init__(f"tokenizer '{tokenizer}' unavailable: {cause}")


def _normalize(text: str) -> str:
    t = text.lower().translate(_PUNCT)
    t = _ARTICLES.sub(" ", t)
    return " ".join(t.split())


def _is_error(row: Dict[str, str]) -> bool:
    # Shared predicate from the canonical loader (2026-07-15): one error-semantics
    # definition across all analysis tools. Divergence deliberately keeps its
    # error-only skip (an empty generation is a legitimate divergence outcome).
    from _results_loader import is_error
    return is_error(row.get("error"))


def _load_quality() -> Any:
    """Import src.evaluation.quality (fail-closed): the sec. 8.9 answer-changing
    classification MUST use the same sanitizer/normalization/EM-F1 protocol as
    the quality module -- imported, never reimplemented here."""
    try:
        from src.evaluation import quality  # type: ignore
        return quality
    except ImportError:
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from src.evaluation import quality  # type: ignore
            return quality
        except ImportError as exc:  # fail-closed: no local reimplementation fallback
            raise RuntimeError(
                "src.evaluation.quality is required for the answer-changing vs "
                f"answer-preserving classification and could not be imported: {exc}"
            ) from exc


def _make_tokenizer(spec: str) -> Tuple[str, Callable[[str], List[Any]]]:
    """Build the token stream used for first-divergence positions.

    'whitespace' -> deterministic ``str.split()`` (no dependency, position is a
    word index). Any other spec is an HF tokenizer name; positions are then real
    model token-id indices. Load failure raises TokenizerUnavailableError.
    """
    if spec == "whitespace":
        return "whitespace", lambda t: t.split()
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(spec)
    except Exception as exc:  # ImportError, OSError, HTTP errors, ...
        raise TokenizerUnavailableError(spec, str(exc)) from exc

    def _encode(text: str) -> List[Any]:
        return tok.encode(text, add_special_tokens=False)

    return f"hf:{spec}", _encode


def _first_divergence_position(tokens_a: Sequence[Any], tokens_b: Sequence[Any]) -> Optional[int]:
    """0-based index of the first differing token (== common-prefix length).

    None when the token sequences are identical. When one sequence is a strict
    prefix of the other, the position is the shorter length (the first token one
    side emitted and the other did not).
    """
    n = min(len(tokens_a), len(tokens_b))
    for i in range(n):
        if tokens_a[i] != tokens_b[i]:
            return i
    if len(tokens_a) != len(tokens_b):
        return n
    return None


class _Row(NamedTuple):
    answer: str
    prompt_tokens: Optional[float]
    # None = column absent from the CSV (gold unknown); "" = present-but-empty
    # (a legitimate SQuAD v2 unanswerable gold -- scored, not skipped).
    reference_answer: Optional[str]


def _load_answers(baseline_dir: Path) -> Dict[Tuple[str, str, str], _Row]:
    """Map (example_id, trial, repeat_index) -> _Row across all trial CSVs (skip errors).

    Keying includes the TRIAL (audit 2026-07-16 S5): the previous (example_id, repeat_index)
    key with first-occurrence-wins was correct only because the 100x3 manifest draws DISJOINT
    per-trial query blocks (0 example_id overlap verified); under a manifest that repeats
    queries across trials it would silently discard 2 of 3 trials. prompt_tokens is carried
    for the per-arm input-effect flag; reference_answer for the sec. 8.9 classification.
    """
    out: Dict[Tuple[str, str, str], _Row] = {}
    csv_files = sorted(baseline_dir.glob("trial_*/results.csv")) or sorted(baseline_dir.glob("results.csv"))
    for csv_path in csv_files:
        parent = csv_path.parent.name
        trial = parent if parent.startswith("trial_") else "trial_1"
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ex = (row.get("example_id") or "").strip()
                if not ex or _is_error(row):
                    continue
                rep = (row.get("repeat_index") or "0").strip() or "0"
                pt_raw = (row.get("prompt_tokens") or "").strip()
                try:
                    pt: Optional[float] = float(pt_raw) if pt_raw else None
                except ValueError:
                    pt = None
                gold: Optional[str] = row["reference_answer"] if "reference_answer" in row else None
                # First non-error occurrence wins (stable if a trial was re-run).
                out.setdefault(
                    (ex, trial, rep),
                    _Row(answer=row.get("generated_answer") or "", prompt_tokens=pt, reference_answer=gold),
                )
    return out


def _classify_pair(
    arm_text: str,
    ref_text: str,
    gold: Optional[str],
    quality: Any,
) -> Tuple[bool, str]:
    """Charter sec. 8.9 classification for ONE divergent pair.

    Returns (answer_changing, basis). Answers are extracted with
    ``quality.sanitize_answer`` and compared under the official SQuAD v2
    normalization via ``QualityEvaluator.evaluate_f1_score`` (invoked unbound,
    the documented instance-state-free protocol) -- imported, not reimplemented.

    basis='gold' (reference_answer present in the CSV): answer-changing iff the
    divergence flips EM, F1, or the abstention flag against the SAME gold.
    basis='pairwise' (no gold column): answer-changing iff the abstention flags
    differ, or the two extracted answers are not exact-match-equal under the
    official normalization.
    """
    f1_fn = quality.QualityEvaluator.evaluate_f1_score
    sanitized_arm = quality.sanitize_answer(arm_text)
    sanitized_ref = quality.sanitize_answer(ref_text)

    if gold is not None:
        scores_arm = f1_fn(None, sanitized_arm, gold)
        scores_ref = f1_fn(None, sanitized_ref, gold)
        changing = (
            scores_arm["exact_match"] != scores_ref["exact_match"]
            or abs(scores_arm["f1"] - scores_ref["f1"]) > 1e-9
            or scores_arm["predicted_no_answer"] != scores_ref["predicted_no_answer"]
        )
        return changing, "gold"

    abstain_arm = quality.is_no_answer_prediction(sanitized_arm)
    abstain_ref = quality.is_no_answer_prediction(sanitized_ref)
    if abstain_arm != abstain_ref:
        return True, "pairwise"
    if abstain_arm and abstain_ref:
        return False, "pairwise"  # both abstained: re-worded abstention, metrics identical
    # Neither abstained: exact match of the two extracted answers under the official
    # normalization (evaluate_f1_score's own normalize_text, reused via EM).
    em = f1_fn(None, sanitized_arm, sanitized_ref)["exact_match"]
    return em == 0.0, "pairwise"


def _reproducibility_for_arm(answers: Dict[Tuple[str, str, str], _Row]) -> Dict[str, object]:
    """Per-cell reproducibility across repeats (charter sec. 8.9 within-cell companion).

    Groups rows by (example_id, trial); every group with >=2 non-error repeats is
    checked: a VIOLATION is a group whose repeat answers are not all identical
    (raw, after strip); the normalized variant applies QA normalization first.
    Computed from data the chassis already collects -- zero extra runs.
    """
    groups: Dict[Tuple[str, str], List[str]] = {}
    for (ex, trial, _rep), row in answers.items():
        groups.setdefault((ex, trial), []).append(row.answer)

    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    n_groups = len(multi)
    n_violations = sum(1 for v in multi.values() if len({a.strip() for a in v}) > 1)
    n_norm_violations = sum(1 for v in multi.values() if len({_normalize(a) for a in v}) > 1)
    sizes = sorted(len(v) for v in multi.values())
    return {
        "n_groups": n_groups,
        "n_violations": n_violations,
        "violation_rate": round(n_violations / n_groups, 4) if n_groups else None,
        "n_normalized_violations": n_norm_violations,
        "normalized_violation_rate": round(n_norm_violations / n_groups, 4) if n_groups else None,
        "repeats_min": sizes[0] if sizes else None,
        "repeats_max": sizes[-1] if sizes else None,
    }


def _pair_stats(
    a: Dict[Tuple[Any, ...], _Row],
    b: Dict[Tuple[Any, ...], _Row],
    keys: Collection[Tuple[Any, ...]],
    tokenize: Callable[[str], List[Any]],
    tok_label: str,
    quality: Any,
) -> Dict[str, object]:
    """Charter sec. 8.9 statistics for ONE paired comparison over matched ``keys``.

    ``b`` is the REFERENCE side (the pilot's reference arm; the HF oracle in a
    campaign engine-pair). Every emitted number is orientation-symmetric — raw/
    normalized divergence, token agreement, first-divergence position, and the
    answer-changing classification (gold-based EM/F1/abstention flips compare
    both outputs against the SAME gold; the pairwise fallback's abstention-flip
    and official-normalization EM are symmetric) — so pair ordering is a
    LABELING choice, never a statistics choice. Extracted UNCHANGED from the
    pilot-proven compute_divergence loop (T6.4): both the pilot arm-vs-reference
    lane and the campaign engine-pair lane MUST emit byte-identical statistics.
    ``keys`` must be non-empty and present in both maps (caller-enforced).
    """
    n = len(keys)
    raw_div = 0
    norm_div = 0
    token_agree = 0
    positions: List[int] = []
    answer_changing = 0
    answer_preserving = 0
    bases: set[str] = set()
    for k in keys:
        a_text, b_text = a[k].answer, b[k].answer
        raw_divergent = a_text.strip() != b_text.strip()
        if _normalize(a_text) != _normalize(b_text):
            norm_div += 1
        if not raw_divergent:
            token_agree += 1  # identical strings are token-identical under any tokenizer
            continue
        raw_div += 1
        # Charter sec. 8.9 stat 2: first-divergence token position.
        pos = _first_divergence_position(tokenize(a_text), tokenize(b_text))
        if pos is None:
            token_agree += 1  # raw-divergent but token-identical (e.g. whitespace-only)
        else:
            positions.append(pos)
        # Charter sec. 8.9 stat 3 (the decisive one): answer-changing vs -preserving.
        gold = b[k].reference_answer if b[k].reference_answer is not None else a[k].reference_answer
        changing, basis = _classify_pair(a_text, b_text, gold, quality)
        bases.add(basis)
        if changing:
            answer_changing += 1
        else:
            answer_preserving += 1
    return {
        "n_compared": n,
        "raw_divergent": raw_div,
        "raw_divergence_rate": round(raw_div / n, 4),
        "normalized_divergent": norm_div,
        "normalized_divergence_rate": round(norm_div / n, 4),
        # Charter sec. 8.9 stat 1: exact agreement rates (string + token stream).
        "agreement_rate": round(1.0 - raw_div / n, 4),
        "token_agreement_rate": round(token_agree / n, 4),
        # Charter sec. 8.9 stat 2: first-divergence token position summary
        # (0-based common-prefix length over token-divergent pairs).
        "first_divergence": {
            "tokenizer": tok_label,
            "n_raw_divergent": raw_div,
            "n_token_divergent": len(positions),
            "n_token_identical_divergent": raw_div - len(positions),
            "mean_position": round(statistics.mean(positions), 2) if positions else None,
            "median_position": statistics.median(positions) if positions else None,
            "min_position": min(positions) if positions else None,
            "max_position": max(positions) if positions else None,
        },
        # Charter sec. 8.9 stat 3: did the divergence flip EM/F1/abstention?
        "answer_divergence": {
            "n_classified": raw_div,
            "answer_changing": answer_changing,
            "answer_preserving": answer_preserving,
            "answer_changing_rate": round(answer_changing / n, 4),
            "answer_changing_share_of_divergent": (
                round(answer_changing / raw_div, 4) if raw_div else None
            ),
            "classification_basis": sorted(bases),
        },
    }


def compute_divergence(
    results_dir: str,
    reference: str,
    *,
    tokenizer: str = "whitespace",
) -> Dict[str, object]:
    root = Path(results_dir)
    ref_dir = root / reference
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"reference arm '{reference}' not found under {results_dir}")
    ref = _load_answers(ref_dir)
    if not ref:
        raise ValueError(f"reference arm '{reference}' has no non-error answers")

    quality = _load_quality()
    tok_label, tokenize = _make_tokenizer(tokenizer)

    rows: List[Dict[str, object]] = []
    reproducibility: List[Dict[str, object]] = [
        {"arm": reference, "is_reference": True, **_reproducibility_for_arm(ref)}
    ]
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        arm = arm_dir.name
        if arm == reference:
            continue
        ans = _load_answers(arm_dir)
        if ans:
            reproducibility.append({"arm": arm, "is_reference": False, **_reproducibility_for_arm(ans)})
        keys = set(ans) & set(ref)  # compare only matched (example_id, trial, repeat_index)
        if not keys:
            continue

        # Sec. 8.9 stats 1-3 via the shared pair engine (arm = a-side, reference = b-side).
        stats = _pair_stats(ans, ref, keys, tokenize, tok_label, quality)

        # Input-effect flag (audit 2026-07-16 S5): same_prompt iff the arm's median
        # prompt_tokens is within 5% of the reference's over the matched keys. Arms that
        # fail it (rag/compressed/multiturn/corpus-prefix families) feed DIFFERENT prompts
        # to the model, so their divergence measures input change, not engine
        # nondeterminism -- flag them so the JSON cannot be misread as a losslessness row.
        arm_pts = [ans[k].prompt_tokens for k in keys if ans[k].prompt_tokens is not None]
        ref_pts = [ref[k].prompt_tokens for k in keys if ref[k].prompt_tokens is not None]
        arm_med = statistics.median(arm_pts) if arm_pts else None
        ref_med = statistics.median(ref_pts) if ref_pts else None
        input_effect: Optional[bool] = None
        if arm_med is not None and ref_med is not None and ref_med > 0:
            input_effect = abs(arm_med - ref_med) > 0.05 * ref_med
        entry: Dict[str, object] = {
            "arm": arm,
            **stats,
            "median_prompt_tokens": arm_med,
            "reference_median_prompt_tokens": ref_med,
            "input_effect": input_effect,
        }
        if input_effect:
            entry["note"] = (
                "input-effect arm: median prompt_tokens differs from the reference by >5%; "
                "divergence reflects different prompt contexts, not engine nondeterminism"
            )
        rows.append(entry)
    return {
        "reference": reference,
        "results_dir": str(root),
        "tokenizer": tok_label,
        "arms": rows,
        # Charter sec. 8.9 within-cell companion: per-cell reproducibility across repeats.
        "reproducibility": reproducibility,
    }


# ---------------------------------------------------------------------------
# CAMPAIGN mode (T6.4): the sec. 8.9 instrument aimed at its charter target —
# per model x ENGINE-PAIR over a RESULTS_LAYOUT-v2 campaign tree. The pilot
# lane above compared ARMS on one engine; this lane compares ENGINES on one
# (model, dataset, arm, grid-point, replicate) group.
# ---------------------------------------------------------------------------

#: MUST equal organize_results.WINDOW_DIR_RE == campaign_layout.WINDOW_DIR_RE
#: (§1: k = <dataset>-<ordinal>); pinned by test_token_divergence_campaign.py
#: the way campaign_layout pins its copy — importing campaign_layout here would
#: drag pandas into a stdlib-only CLI.
_CAMPAIGN_WINDOW_DIR_RE = re.compile(r"^window_([a-z0-9_]+)-(\d+)$")

#: MUST equal campaign_layout.QA_EVIDENCE_EXEMPT_DATASETS (§1: ShareGPT is the
#: load donor — its windows carry serving streams only, no QA generations).
_QA_EVIDENCE_EXEMPT_DATASETS = frozenset({"sharegpt"})

#: Registered pair labels (charter D4 + sec. 8.9). A pair containing the HF
#: reference implementation is THE oracle comparison; an engine-engine pair
#: without HF is DeepSeek-V3's recorded oracle exemption substitute — the label
#: says SUBSTITUTE and the strength says weaker-than-oracle, so the report can
#: never be read as an oracle number.
PAIR_LABEL_ORACLE = "oracle"
PAIR_LABEL_SUBSTITUTE = "cross-engine agreement — oracle-exempt substitute (D4)"
PAIR_STRENGTH_ORACLE = "oracle (HF reference implementation on the b-side)"
PAIR_STRENGTH_SUBSTITUTE = (
    "substitute — weaker than oracle: cross-engine agreement bounds consistency, "
    "not correctness (D4 recorded oracle exemption, e.g. DeepSeek-V3 at 671B)"
)

#: T=0 provenance is a PRODUCER contract, not a per-row field: the campaign
#: runner pins greedy decoding (temperature=0.0 hard-coded at every dispatch
#: site in scripts/3_run/run_experiment.py) and the §1 window artifacts carry
#: no per-row sampling params to re-verify. Stated in every report so the
#: assumption travels with the numbers.
T0_CONTRACT = (
    "generations are T=0 by producer contract (run_experiment.py pins "
    "temperature=0.0); window artifacts carry no per-row sampling params"
)

#: 2 (2026-10-16): group records carry ``corpus_budget_tokens``; reports carry
#: ``cross_arm`` when the lane is requested.
_CAMPAIGN_SCHEMA_VERSION = 2


class CampaignDivergenceError(RuntimeError):
    """Campaign-mode refusal (fail-closed): no eligible engine-pair groups, or
    a tree that violates the RESULTS_LAYOUT contract. Never a silent default."""


def _load_cellspec() -> Any:
    """Import src.analysis.cellspec (fail-closed, mirrors _load_quality): row
    keys are parsed by round-tripping cell.json's ``cellspec`` mapping through
    the PUBLIC surface (``CellSpec.from_flat_dict(...).to_row_key()`` vs the
    dirname, exactly organize_results' ``_read_cell_meta`` contract) — never by
    string-splitting directory names."""
    try:
        from src.analysis import cellspec  # type: ignore
        return cellspec
    except ImportError:
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            from src.analysis import cellspec  # type: ignore
            return cellspec
        except ImportError as exc:  # fail-closed: no dirname-splitting fallback
            raise RuntimeError(
                f"src.analysis.cellspec is required to parse campaign row keys: {exc}"
            ) from exc


def _find_run_roots(root: Path) -> List[Path]:
    """Every RESULTS_LAYOUT-v2 run root at-or-under ``root`` (a dir owning
    ``cells/``). One run = one engine x one model (§3), so >=2-engine groups
    necessarily span multiple run roots under one campaign root."""
    if (root / "cells").is_dir():
        return [root]
    return sorted({p.parent for p in root.rglob("cells") if p.is_dir()})


def _fmt_key(key: Tuple[Any, Any, Any]) -> str:
    return f"(example_id={key[0]!r}, repeat_index={key[1]!r}, record_index={key[2]!r})"


class _WindowLoad(NamedTuple):
    #: (example_id, repeat_index, record_index) -> _Row; None when the window
    #: contributes no T=0 generations (skip_reason then says why).
    answers: Optional[Dict[Tuple[Any, Any, Any], _Row]]
    skip_reason: Optional[str]
    n_error_rows: int


def _load_prompt_tokens(window_dir: Path) -> Dict[Tuple[Any, Any, Any], float]:
    """(example_id, repeat_index, record_index) -> prompt_tokens from the window's
    requests.jsonl (the serving rows; qa_evidence.jsonl carries no prompt length).
    Provenance for the cross-arm input-effect gate only: a missing file, an
    unparseable line or a non-numeric value leaves the key out, and the gate then
    reads None ("unknown"), never a fabricated length."""
    path = window_dir / "requests.jsonl"
    out: Dict[Tuple[Any, Any, Any], float] = {}
    if not path.is_file():
        return out
    seen: set = set()
    duplicated: set = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        key = (row.get("example_id"), row.get("repeat_index"), row.get("record_index"))
        if key in seen:
            duplicated.add(key)  # an ambiguous length is no length: the gate reads None
            continue
        seen.add(key)
        pt = row.get("prompt_tokens")
        if isinstance(pt, bool) or not isinstance(pt, (int, float)):
            continue
        out[key] = float(pt)
    for key in duplicated:
        out.pop(key, None)
    return out


def _load_window_generations(window_dir: Path, dataset: str) -> _WindowLoad:
    """Load a window's T=0 generations from qa_evidence.jsonl — the ONE §1
    generation-carrying window artifact (requests.jsonl carries serving rows,
    cage_stats.jsonl telemetry, engine_metrics.json backend metadata).

    Keyed on the #127 identity triple ``(example_id, repeat_index, record_index)``
    VERBATIM — no type coercion, so producer drift across engines surfaces as a
    mismatched-query-set skip instead of being papered over. Error rows are
    dropped (the pilot lane's error-only skip: an EMPTY generation is a
    legitimate divergence outcome and is kept). Schema violations — a non-error
    row without ``generated_answer``, a missing example_id, a duplicate triple —
    make the whole window a labeled skip: fabricating a value or guessing a join
    is worse than not comparing."""
    ev_path = window_dir / "qa_evidence.jsonl"
    if not ev_path.is_file():
        if dataset in _QA_EVIDENCE_EXEMPT_DATASETS:
            return _WindowLoad(
                None,
                f"load-donor dataset {dataset!r} carries serving streams only — "
                "no T=0 generations by RESULTS_LAYOUT §1 design",
                0,
            )
        return _WindowLoad(
            None, "no qa_evidence.jsonl — window carries no T=0 generations", 0
        )
    answers: Dict[Tuple[Any, Any, Any], _Row] = {}
    prompt_tokens_by_key = _load_prompt_tokens(window_dir)
    n_error = 0
    for lineno, line in enumerate(
        ev_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            return _WindowLoad(
                None, f"qa_evidence.jsonl:{lineno}: invalid JSON ({exc})", n_error
            )
        if not isinstance(row, dict):
            return _WindowLoad(
                None,
                f"qa_evidence.jsonl:{lineno}: record is not an object",
                n_error,
            )
        if _is_error(row):
            n_error += 1  # error rows never generated; dropping them is the pilot precedent
            continue
        ex = row.get("example_id")
        if not isinstance(ex, str) or not ex:
            return _WindowLoad(
                None,
                f"qa_evidence.jsonl:{lineno}: non-error row without example_id "
                "(the §8 join key) — unjoinable rows are unaccountable rows",
                n_error,
            )
        text = row.get("generated_answer")
        if not isinstance(text, str):
            return _WindowLoad(
                None,
                f"qa_evidence.jsonl:{lineno}: non-error row without a string "
                "generated_answer — refusing to fabricate an empty generation",
                n_error,
            )
        key = (ex, row.get("repeat_index"), row.get("record_index"))
        if key in answers:
            return _WindowLoad(
                None,
                f"qa_evidence.jsonl:{lineno}: duplicate identity triple "
                f"{_fmt_key(key)} — the engine-pair join would be ambiguous",
                n_error,
            )
        gold = row.get("reference_answer")
        answers[key] = _Row(
            answer=text,
            prompt_tokens=prompt_tokens_by_key.get(key),  # from requests.jsonl; None when absent
            reference_answer=gold if isinstance(gold, str) else None,
        )
    if not answers:
        return _WindowLoad(
            None,
            f"qa_evidence.jsonl carries no non-error generations "
            f"({n_error} error row(s) dropped)",
            n_error,
        )
    return _WindowLoad(answers, None, n_error)


class _GroupMember(NamedTuple):
    engine: str
    run_root: Path
    window_rel: str  # cells/<row_key>/window_<k>, relative to its run root
    seed: Any
    answers: Dict[Tuple[Any, Any, Any], _Row]


def compute_campaign_divergence(
    campaign_root: str,
    *,
    tokenizer: str = "whitespace",
    cross_arm_pairs: Sequence[Tuple[str, str]] = (),
) -> Dict[str, object]:
    """Walk a RESULTS_LAYOUT-v2 campaign tree and compute the sec. 8.9
    statistics for every engine pair of every eligible group.

    A GROUP is one (model, dataset, arm, grid-point, replicate) where
    grid-point = the non-engine cell axes (retriever, policy, topology, family,
    budget_r, rate_frac, corpus_budget_tokens) and replicate = the §1 windows[] ``rep``. ELIGIBLE =
    served by >=2 distinct engines with T=0 generations. Zero eligible groups
    raises CampaignDivergenceError naming everything searched. Layout-contract
    violations (unparseable cell.json, row-key round-trip mismatch, stray
    non-window entries, windows[] contradicting a dirname) refuse loudly with
    EVERY problem listed. Mismatched query sets across a group's engines and
    windows without generations are LABELED skips carried into the reports.
    """
    root = Path(campaign_root)
    if not root.is_dir():
        raise CampaignDivergenceError(f"--campaign-root {root} is not a directory")
    cellspec_mod = _load_cellspec()
    quality = _load_quality()
    tok_label, tokenize = _make_tokenizer(tokenizer)

    run_roots = _find_run_roots(root)
    if not run_roots:
        raise CampaignDivergenceError(
            f"no RESULTS_LAYOUT-v2 run tree found under {root} — searched "
            "recursively for cells/ directories and found none; nothing to compare"
        )

    problems: List[str] = []
    skips: List[Dict[str, object]] = []
    groups: Dict[Tuple[Any, ...], List[_GroupMember]] = {}
    n_cells = 0
    n_windows = 0
    n_windows_nogen = 0

    for run_root in run_roots:
        run_rel = "." if run_root == root else str(run_root.relative_to(root))
        for cell_dir in sorted((run_root / "cells").iterdir()):
            if cell_dir.name.startswith("."):
                continue
            cell_rel = f"{run_rel}/cells/{cell_dir.name}"
            if not cell_dir.is_dir():
                problems.append(f"{cell_rel}: not a directory — not a §1 cell")
                continue
            meta_path = cell_dir / "cell.json"
            if not meta_path.is_file():
                problems.append(f"{cell_rel}: cell.json missing (§2 cell identity)")
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                problems.append(f"{cell_rel}/cell.json: invalid JSON: {exc}")
                continue
            cellspec_map = meta.get("cellspec") if isinstance(meta, dict) else None
            if not isinstance(cellspec_map, dict):
                problems.append(f"{cell_rel}/cell.json: no 'cellspec' mapping (§2)")
                continue
            try:
                spec = cellspec_mod.CellSpec.from_flat_dict(cellspec_map)
            except cellspec_mod.CellSpecError as exc:
                problems.append(f"{cell_rel}/cell.json: cellspec rejected: {exc}")
                continue
            if spec.to_row_key() != cell_dir.name:
                problems.append(
                    f"{cell_rel}: cell.json cellspec round-trips to row key "
                    f"{spec.to_row_key()!r} != dirname (§2 identity drift)"
                )
                continue
            n_cells += 1
            windows_meta = meta.get("windows")
            windows_meta = windows_meta if isinstance(windows_meta, dict) else {}
            for window_dir in sorted(cell_dir.iterdir()):
                if window_dir.name.startswith(".") or window_dir.name == "cell.json":
                    continue
                match = _CAMPAIGN_WINDOW_DIR_RE.match(window_dir.name)
                if not window_dir.is_dir() or match is None:
                    problems.append(
                        f"{cell_rel}/{window_dir.name}: not a "
                        "window_<dataset>-<NN> directory (§1)"
                    )
                    continue
                n_windows += 1
                dataset = match.group(1)
                window_key = f"{dataset}-{match.group(2)}"  # reader-verbatim digits
                window_rel = f"cells/{cell_dir.name}/{window_dir.name}"
                where = f"{run_rel}/{window_rel}"
                entry = windows_meta.get(window_key)
                if not isinstance(entry, dict) or "rep" not in entry:
                    # Incomplete emission (resume residue): windows[] is the §1
                    # authority for the replicate id — guessing rep from the
                    # ordinal would fabricate a join coordinate.
                    skips.append({
                        "model": spec.model,
                        "dataset": dataset,
                        "where": where,
                        "reason": (
                            "window dir not declared in cell.json windows[] "
                            "(incomplete emission) — replicate unknown, refusing to guess"
                        ),
                    })
                    continue
                if entry.get("dataset") != dataset:
                    problems.append(
                        f"{cell_rel}/{window_dir.name}: windows[{window_key!r}].dataset="
                        f"{entry.get('dataset')!r} contradicts the directory name"
                    )
                    continue
                load = _load_window_generations(window_dir, dataset)
                if load.answers is None:
                    n_windows_nogen += 1
                    skips.append({
                        "model": spec.model,
                        "dataset": dataset,
                        "where": where,
                        "reason": f"window without T=0 generations: {load.skip_reason}",
                    })
                    continue
                group_key = (
                    spec.model,
                    dataset,
                    spec.arm,
                    spec.retriever,
                    spec.policy,
                    spec.topology,
                    spec.family,
                    spec.budget_r,
                    spec.rate_frac,
                    spec.corpus_budget_tokens,
                    entry["rep"],
                )
                groups.setdefault(group_key, []).append(
                    _GroupMember(
                        engine=spec.engine,
                        run_root=run_root,
                        window_rel=window_rel,
                        seed=entry.get("seed"),
                        answers=load.answers,
                    )
                )
    if problems:
        lines = "\n".join(f"  [{i + 1}] {p}" for i, p in enumerate(problems))
        raise CampaignDivergenceError(
            f"refusing campaign walk — {len(problems)} RESULTS_LAYOUT contract "
            f"violation(s) under {root}:\n{lines}"
        )

    n_single = 0
    n_multi = 0
    n_group_skips = 0
    group_records: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for group_key, members in sorted(
        groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        model, dataset, arm, retriever, policy, topology, family, budget_r, rate_frac, cb, rep = group_key
        engines = sorted({m.engine for m in members})
        group_desc = (
            f"model={model} dataset={dataset} arm={arm} retriever={retriever} "
            f"policy={policy} topology={topology} family={family} "
            f"budget_r={budget_r} rate_frac={rate_frac} corpus_budget_tokens={cb} rep={rep}"
        )
        if len(engines) < 2:
            n_single += 1
            continue
        n_multi += 1
        by_engine: Dict[str, List[_GroupMember]] = {}
        for m in members:
            by_engine.setdefault(m.engine, []).append(m)
        ambiguous = {e: ms for e, ms in by_engine.items() if len(ms) > 1}
        if ambiguous:
            detail = "; ".join(
                f"{e}: {len(ms)} windows ({', '.join(sorted(m.window_rel for m in ms))})"
                for e, ms in sorted(ambiguous.items())
            )
            skips.append({
                "model": model,
                "dataset": dataset,
                "where": group_desc,
                "reason": (
                    f"ambiguous replicate — engine(s) contribute multiple windows "
                    f"for one (model, dataset, arm, grid-point, replicate); "
                    f"refusing to choose: {detail}"
                ),
            })
            n_group_skips += 1
            continue
        key_sets = {e: set(by_engine[e][0].answers) for e in engines}
        union = set().union(*key_sets.values())
        if any(key_sets[e] != union for e in engines):
            parts = []
            for e in engines:
                missing = union - key_sets[e]
                part = f"{e}: {len(key_sets[e])} key(s)"
                if missing:
                    sample = ", ".join(
                        _fmt_key(k) for k in sorted(missing, key=repr)[:3]
                    )
                    part += f", missing {len(missing)} e.g. {sample}"
                parts.append(part)
            skips.append({
                "model": model,
                "dataset": dataset,
                "where": group_desc,
                "reason": (
                    "mismatched query sets across engines — refusing to "
                    "silently intersect (fail-closed): " + "; ".join(parts)
                ),
            })
            n_group_skips += 1
            continue

        record: Dict[str, object] = {
            "arm": arm,
            "retriever": retriever,
            "policy": policy,
            "topology": topology,
            "family": family,
            "budget_r": budget_r,
            "rate_frac": rate_frac,
            "corpus_budget_tokens": cb,
            "rep": rep,
            "n_queries": len(union),
            "engines": {
                e: {
                    "run_root": (
                        "." if by_engine[e][0].run_root == root
                        else str(by_engine[e][0].run_root.relative_to(root))
                    ),
                    "window": by_engine[e][0].window_rel,
                    "seed": by_engine[e][0].seed,
                    "n_answers": len(by_engine[e][0].answers),
                }
                for e in engines
            },
            "pairs": [],
        }
        for engine_x, engine_y in itertools.combinations(engines, 2):
            if "hf" in (engine_x, engine_y):
                # Oracle orientation: HF is always the b-side (the reference
                # of the answer-changing classification's gold fallback).
                engine_a = engine_x if engine_y == "hf" else engine_y
                engine_b = "hf"
                label, strength = PAIR_LABEL_ORACLE, PAIR_STRENGTH_ORACLE
            else:
                engine_a, engine_b = engine_x, engine_y  # sorted already
                label, strength = PAIR_LABEL_SUBSTITUTE, PAIR_STRENGTH_SUBSTITUTE
            stats = _pair_stats(
                by_engine[engine_a][0].answers,
                by_engine[engine_b][0].answers,
                union,
                tokenize,
                tok_label,
                quality,
            )
            record["pairs"].append({
                "engine_a": engine_a,
                "engine_b": engine_b,
                "label": label,
                "strength": strength,
                **stats,
            })
        group_records.setdefault((model, dataset), []).append(record)

    if n_multi == 0 and not cross_arm_pairs:
        raise CampaignDivergenceError(
            f"no eligible engine-pair groups under {root}: searched "
            f"{len(run_roots)} run tree(s), {n_cells} cell(s), {n_windows} "
            f"window(s) ({n_windows_nogen} without T=0 generations); "
            f"{n_single} group(s) had a single engine only. An eligible group "
            "is one (model, dataset, arm, grid-point, replicate) served by "
            ">=2 engines with qa_evidence generations."
        )

    cross_records, cross_skips = _cross_arm_records(
        groups, cross_arm_pairs, root, tokenize, tok_label, quality
    )
    if n_multi == 0 and not cross_records:
        raise CampaignDivergenceError(
            f"no eligible engine-pair groups and no compared cross-arm group under {root}: searched "
            f"{len(run_roots)} run tree(s), {n_cells} cell(s), {n_windows} window(s); "
            f"{n_single} single-engine group(s); cross-arm skips: {len(cross_skips)}"
        )

    # One report per (model, dataset) — every pairing touched (compared or
    # skipped) gets a report so labeled skips are carried into the artifact.
    touched = set(group_records) | {
        (str(s["model"]), str(s["dataset"])) for s in skips
    } | set(cross_records) | {(str(s["model"]), str(s["dataset"])) for s in cross_skips}
    reports: List[Dict[str, object]] = []
    for model, dataset in sorted(touched):
        md_groups = group_records.get((model, dataset), [])
        md_skips = [s for s in skips if (str(s["model"]), str(s["dataset"])) == (model, dataset)]
        report: Dict[str, object] = {
            "schema_version": _CAMPAIGN_SCHEMA_VERSION,
            "mode": "campaign-engine-pair",
            "campaign_root": str(root),
            "model": model,
            "dataset": dataset,
            "tokenizer": tok_label,
            "t0_contract": T0_CONTRACT,
            "counts": {
                "groups_compared": len(md_groups),
                "pairs": sum(len(g["pairs"]) for g in md_groups),
                "skips": len(md_skips),
            },
            "groups": md_groups,
            "skips": md_skips,
        }
        if cross_arm_pairs:
            md_cross = cross_records.get((model, dataset), [])
            md_cross_skips = [
                s for s in cross_skips if (str(s["model"]), str(s["dataset"])) == (model, dataset)
            ]
            report["cross_arm"] = {
                "pairs_registered": [f"{a}:{b}" for a, b in cross_arm_pairs],
                "input_effect_rule": CROSS_ARM_INPUT_RULE,
                "counts": {"groups_compared": len(md_cross), "skips": len(md_cross_skips)},
                "groups": md_cross,
                "skips": md_cross_skips,
            }
        reports.append(report)

    return {
        "mode": "campaign-engine-pair",
        "campaign_root": str(root),
        "tokenizer": tok_label,
        "t0_contract": T0_CONTRACT,
        "counts": {
            "run_roots": len(run_roots),
            "cells": n_cells,
            "windows_scanned": n_windows,
            "windows_without_generations": n_windows_nogen,
            "groups_total": len(groups),
            "groups_single_engine": n_single,
            "groups_multi_engine": n_multi,
            "groups_compared": sum(len(g) for g in group_records.values()),
            "groups_skipped": n_group_skips,
            "cross_arm_groups_compared": sum(len(g) for g in cross_records.values()),
            "cross_arm_groups_skipped": len(cross_skips),
        },
        "reports": reports,
        "skips": skips,
        "cross_arm_skips": cross_skips,
    }


#: The pilot lane's input-effect gate, reused verbatim for the cross-arm lane.
CROSS_ARM_INPUT_EFFECT_TOLERANCE = 0.05
CROSS_ARM_INPUT_RULE = (
    "input_effect = |median prompt_tokens(cell) - median prompt_tokens(reference)| "
    f"> {CROSS_ARM_INPUT_EFFECT_TOLERANCE:g} x reference median, from the windows' requests.jsonl; "
    "None when either side carries no prompt_tokens"
)
CROSS_ARM_LABEL_SAME = "same inputs: divergence is the serving-config effect"
CROSS_ARM_LABEL_DIFFERENT = (
    "different inputs: median prompt_tokens differ by more than the tolerance; "
    "divergence measures a prompt change, not the cache effect"
)
CROSS_ARM_LABEL_UNKNOWN = (
    "input effect unknown: no prompt_tokens on one side's rows; not readable as a cache effect"
)


def _answer_change_breakdown(
    a: Dict[Tuple[Any, ...], _Row],
    b: Dict[Tuple[Any, ...], _Row],
    keys: Collection[Tuple[Any, ...]],
    quality: Any,
) -> Dict[str, Any]:
    """Spec Block A2 OUTPUTS: over the raw-divergent pairs, how many flip EM, change F1
    or flip the abstention flag, each counted on its own (a pair can count in several).
    Same sanitizer and scorer as ``_classify_pair``; with gold the flips are against the
    same gold, without gold EM and abstention are compared pairwise and F1 change is
    not defined (None)."""
    f1_fn = quality.QualityEvaluator.evaluate_f1_score
    em_flip = f1_change = abst_flip = 0
    n_divergent = 0
    gold_basis = True
    for k in keys:
        a_text, b_text = a[k].answer, b[k].answer
        if a_text.strip() == b_text.strip():
            continue
        n_divergent += 1
        sa, sb = quality.sanitize_answer(a_text), quality.sanitize_answer(b_text)
        gold = b[k].reference_answer if b[k].reference_answer is not None else a[k].reference_answer
        if gold is not None:
            ra, rb = f1_fn(None, sa, gold), f1_fn(None, sb, gold)
            em_flip += int(ra["exact_match"] != rb["exact_match"])
            f1_change += int(abs(ra["f1"] - rb["f1"]) > 1e-9)
            abst_flip += int(ra["predicted_no_answer"] != rb["predicted_no_answer"])
        else:
            gold_basis = False
            abst_a, abst_b = quality.is_no_answer_prediction(sa), quality.is_no_answer_prediction(sb)
            abst_flip += int(abst_a != abst_b)
            if not abst_a and not abst_b:
                em_flip += int(f1_fn(None, sa, sb)["exact_match"] == 0.0)
    return {
        "n_divergent": n_divergent,
        "em_flip": em_flip,
        "f1_change": f1_change if gold_basis else None,
        "abstention_flip": abst_flip,
        "basis": "gold" if gold_basis else "pairwise",
    }


def _cross_arm_records(
    groups: Dict[Tuple[Any, ...], List[_GroupMember]],
    cross_arm_pairs: Sequence[Tuple[str, str]],
    root: Path,
    tokenize: Callable[[str], List[Any]],
    tok_label: str,
    quality: Any,
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, object]]], List[Dict[str, object]]]:
    """Spec Block A2: for every registered (cell_arm, reference_arm) pair, compare
    the two arms on the SAME engine for every (model, dataset, engine, grid-point,
    replicate) both arms serve. Same fail-closed lanes as the engine-pair lane
    (ambiguous replicate and mismatched query sets are labeled skips); a
    registered pair that matches no group at all refuses loudly."""
    records: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    skips: List[Dict[str, object]] = []
    if not cross_arm_pairs:
        return records, skips
    # (model, dataset, retriever, policy, topology, family, budget_r, rate_frac, cb, rep, engine)
    #   -> arm -> members
    by_slot: Dict[Tuple[Any, ...], Dict[str, List[_GroupMember]]] = {}
    for group_key, members in groups.items():
        model, dataset, arm, retriever, policy, topology, family, budget_r, rate_frac, cb, rep = group_key
        for m in members:
            slot = (model, dataset, retriever, policy, topology, family, budget_r, rate_frac, cb, rep, m.engine)
            by_slot.setdefault(slot, {}).setdefault(arm, []).append(m)
    for cell_arm, ref_arm in cross_arm_pairs:
        n_matched = 0
        for slot in sorted(by_slot, key=lambda k: tuple(str(x) for x in k)):
            arms = by_slot[slot]
            model, dataset, retriever, policy, topology, family, budget_r, rate_frac, cb, rep, engine = slot
            desc = (
                f"cross-arm {cell_arm}:{ref_arm} model={model} dataset={dataset} engine={engine} "
                f"retriever={retriever} policy={policy} topology={topology} family={family} "
                f"budget_r={budget_r} rate_frac={rate_frac} corpus_budget_tokens={cb} rep={rep}"
            )
            if ref_arm in arms and cell_arm not in arms:
                # The informative direction (review 2026-10-16 MEDIUM 2): the reference
                # was served here but the cell arm was not, so the pair has no record.
                skips.append({"model": model, "dataset": dataset, "where": desc, "reason": (
                    f"cell arm {cell_arm!r} absent beside reference arm {ref_arm!r} on this engine slot"
                )})
                continue
            if cell_arm not in arms or ref_arm not in arms:
                continue
            n_matched += 1
            ambiguous = {a: ms for a, ms in arms.items() if a in (cell_arm, ref_arm) and len(ms) > 1}
            if ambiguous:
                detail = "; ".join(
                    f"{a}: {len(ms)} windows ({', '.join(sorted(m.window_rel for m in ms))})"
                    for a, ms in sorted(ambiguous.items())
                )
                skips.append({"model": model, "dataset": dataset, "where": desc, "reason": (
                    "ambiguous replicate: an arm contributes multiple windows for one "
                    f"(model, dataset, engine, grid-point, replicate); refusing to choose: {detail}"
                )})
                continue
            cell = arms[cell_arm][0]
            ref = arms[ref_arm][0]
            cell_keys, ref_keys = set(cell.answers), set(ref.answers)
            if cell_keys != ref_keys:
                parts = []
                for name, keys, other in ((cell_arm, cell_keys, ref_keys), (ref_arm, ref_keys, cell_keys)):
                    missing = other - keys
                    part = f"{name}: {len(keys)} key(s)"
                    if missing:
                        sample = ", ".join(_fmt_key(k) for k in sorted(missing, key=repr)[:3])
                        part += f", missing {len(missing)} e.g. {sample}"
                    parts.append(part)
                skips.append({"model": model, "dataset": dataset, "where": desc, "reason": (
                    "mismatched query sets across arms, refusing to silently intersect "
                    "(fail-closed): " + "; ".join(parts)
                )})
                continue
            stats = _pair_stats(cell.answers, ref.answers, cell_keys, tokenize, tok_label, quality)
            breakdown = _answer_change_breakdown(cell.answers, ref.answers, cell_keys, quality)
            cell_pts = [r.prompt_tokens for r in cell.answers.values() if r.prompt_tokens is not None]
            ref_pts = [r.prompt_tokens for r in ref.answers.values() if r.prompt_tokens is not None]
            cell_med = statistics.median(cell_pts) if cell_pts else None
            ref_med = statistics.median(ref_pts) if ref_pts else None
            input_effect: Optional[bool] = None
            if cell_med is not None and ref_med is not None and ref_med > 0:
                input_effect = abs(cell_med - ref_med) > CROSS_ARM_INPUT_EFFECT_TOLERANCE * ref_med
            label = (
                CROSS_ARM_LABEL_UNKNOWN if input_effect is None
                else CROSS_ARM_LABEL_DIFFERENT if input_effect else CROSS_ARM_LABEL_SAME
            )
            records.setdefault((model, dataset), []).append({
                "cell_arm": cell_arm,
                "reference_arm": ref_arm,
                "engine": engine,
                "retriever": retriever,
                "policy": policy,
                "topology": topology,
                "family": family,
                "budget_r": budget_r,
                "rate_frac": rate_frac,
                "corpus_budget_tokens": cb,
                "rep": rep,
                "n_queries": len(cell_keys),
                "windows": {
                    cell_arm: {"run_root": "." if cell.run_root == root else str(cell.run_root.relative_to(root)),
                               "window": cell.window_rel, "seed": cell.seed},
                    ref_arm: {"run_root": "." if ref.run_root == root else str(ref.run_root.relative_to(root)),
                              "window": ref.window_rel, "seed": ref.seed},
                },
                "median_prompt_tokens": cell_med,
                "reference_median_prompt_tokens": ref_med,
                "input_effect": input_effect,
                "label": label,
                "answer_change_breakdown": breakdown,
                **stats,
            })
        if n_matched == 0:
            raise CampaignDivergenceError(
                f"cross-arm pair {cell_arm}:{ref_arm} matched no group under {root}: no "
                "(model, dataset, engine, grid-point, replicate) is served by both arms "
                f"(searched {len(by_slot)} engine slot(s)); a registered pair without data is a refusal, not a silence"
            )
    return records, skips


def _pooled_engine_pairs(report: Dict[str, object]) -> List[Tuple[str, str, str, str, Dict[str, int]]]:
    """(model, dataset, pair, kind, agg) pooled across groups. Pooled rates are
    display-only recomputations from the emitted counts (never new measurements);
    per-group numbers live in the JSON artifacts."""
    pooled: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    for group in report["groups"]:
        for pair in group["pairs"]:
            key = (pair["engine_a"], pair["engine_b"], pair["label"])
            agg = pooled.setdefault(key, {"groups": 0, "n": 0, "raw_div": 0, "changing": 0})
            agg["groups"] += 1
            agg["n"] += pair["n_compared"]
            agg["raw_div"] += pair["raw_divergent"]
            agg["changing"] += pair["answer_divergence"]["answer_changing"]
    out = []
    for (engine_a, engine_b, label), agg in sorted(pooled.items()):
        kind = "oracle" if label == PAIR_LABEL_ORACLE else "substitute"
        out.append((str(report["model"]), str(report["dataset"]), engine_a + "<->" + engine_b, kind, agg))
    return out


def _pooled_cross_arm(report: Dict[str, object]) -> List[Tuple[str, str, str, str, str, Dict[str, int]]]:
    """(model, dataset, cell:ref, engine, input flag, agg) pooled across groups."""
    cross = report.get("cross_arm")
    if not isinstance(cross, dict):
        return []
    pooled: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    for g in cross["groups"]:
        flag = "unknown" if g["input_effect"] is None else ("different" if g["input_effect"] else "same")
        key = (f"{g['cell_arm']}:{g['reference_arm']}", g["engine"], flag)
        agg = pooled.setdefault(key, {"groups": 0, "n": 0, "raw_div": 0, "changing": 0})
        agg["groups"] += 1
        agg["n"] += g["n_compared"]
        agg["raw_div"] += g["raw_divergent"]
        agg["changing"] += g["answer_divergence"]["answer_changing"]
    return [
        (str(report["model"]), str(report["dataset"]), pair, engine, flag, agg)
        for (pair, engine, flag), agg in sorted(pooled.items())
    ]


def _print_campaign_summary(
    result: Dict[str, object], written: Dict[Tuple[str, str], Path]
) -> None:
    """Stdout summary: per (model, dataset, pair) pooled across groups; a second
    table for the cross-arm lane when it ran."""
    counts = result["counts"]
    print(f"\n[divergence] campaign engine-pair T=0 comparison — {result['campaign_root']}")
    print(
        f"[divergence] searched: {counts['run_roots']} run tree(s), "
        f"{counts['cells']} cell(s), {counts['windows_scanned']} window(s); "
        f"groups: {counts['groups_compared']} compared, "
        f"{counts['groups_skipped']} skipped, "
        f"{counts['groups_single_engine']} single-engine"
    )
    print(
        f"{'model':<18}{'dataset':<12}{'pair':<22}{'kind':<12}"
        f"{'groups':>7}{'n':>8}{'agree %':>9}{'ans-chg %':>11}"
    )
    for report in result["reports"]:
        for model, dataset, pair, kind, agg in _pooled_engine_pairs(report):
            print(
                f"{model:<18}{dataset:<12}{pair:<22}{kind:<12}"
                f"{agg['groups']:>7}{agg['n']:>8}"
                f"{100 * (1 - agg['raw_div'] / agg['n']):>8.2f}%"
                f"{100 * agg['changing'] / agg['n']:>10.2f}%"
            )
    if any("cross_arm" in r for r in result["reports"]):
        print(
            f"[divergence] cross-arm lane (Spec Block A2): "
            f"{counts['cross_arm_groups_compared']} group(s) compared, "
            f"{counts['cross_arm_groups_skipped']} skipped"
        )
        print(
            f"{'model':<18}{'dataset':<12}{'cell:reference':<26}{'engine':<9}{'inputs':<10}"
            f"{'groups':>7}{'n':>8}{'agree %':>9}{'ans-chg %':>11}"
        )
        for report in result["reports"]:
            for model, dataset, pair, engine, flag, agg in _pooled_cross_arm(report):
                print(
                    f"{model:<18}{dataset:<12}{pair:<26}{engine:<9}{flag:<10}"
                    f"{agg['groups']:>7}{agg['n']:>8}"
                    f"{100 * (1 - agg['raw_div'] / agg['n']):>8.2f}%"
                    f"{100 * agg['changing'] / agg['n']:>10.2f}%"
                )
    all_skips = list(result["skips"]) + list(result.get("cross_arm_skips", []))
    if all_skips:
        print(f"[divergence] {len(all_skips)} labeled skip(s):")
        for s in all_skips:
            print(f"  - [{s['model']}/{s['dataset']}] {s['where']}: {s['reason']}")
    for (model, dataset), path in sorted(written.items()):
        print(f"[divergence] -> {path}")


def _campaign_summary_markdown(result: Dict[str, object]) -> str:
    """The Markdown twin of the stdout summary (Spec Block A2 output), with the
    per-group cross-arm rows that the pooled table hides."""
    counts = result["counts"]
    lines = [
        "# Charter section 8.9 T=0 divergence (campaign lane)",
        "",
        f"- campaign root: `{result['campaign_root']}`",
        f"- tokenizer: {result['tokenizer']}",
        f"- {result['t0_contract']}",
        f"- searched {counts['run_roots']} run tree(s), {counts['cells']} cell(s), {counts['windows_scanned']} window(s); "
        f"engine-pair groups: {counts['groups_compared']} compared, {counts['groups_skipped']} skipped, "
        f"{counts['groups_single_engine']} single-engine",
        "",
        "## Engine pairs, pooled across groups",
        "",
        "| model | dataset | pair | kind | groups | n | agreement | answer-changing |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for report in result["reports"]:
        for model, dataset, pair, kind, agg in _pooled_engine_pairs(report):
            lines.append(
                f"| {model} | {dataset} | {pair} | {kind} | {agg['groups']} | {agg['n']} | "
                f"{100 * (1 - agg['raw_div'] / agg['n']):.2f}% | {100 * agg['changing'] / agg['n']:.2f}% |"
            )
    if any("cross_arm" in r for r in result["reports"]):
        lines += [
            "",
            "## Cross-arm pairs on one engine (Spec Block A2), per group",
            "",
            f"- {CROSS_ARM_INPUT_RULE}",
            f"- {counts['cross_arm_groups_compared']} group(s) compared, {counts['cross_arm_groups_skipped']} skipped",
            "",
            "| model | dataset | cell:reference | engine | family | rung | inputs | n | agreement | token agreement | first divergence (median) | answer-changing | answer-preserving | median prompt tokens (cell / reference) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for report in result["reports"]:
            cross = report.get("cross_arm")
            if not isinstance(cross, dict):
                continue
            for g in cross["groups"]:
                flag = "unknown" if g["input_effect"] is None else ("different" if g["input_effect"] else "same")
                rung = "" if g["budget_r"] is None else f"r{g['budget_r']:g}"
                med = g["first_divergence"]["median_position"]
                lines.append(
                    f"| {report['model']} | {report['dataset']} | {g['cell_arm']}:{g['reference_arm']} | {g['engine']} | "
                    f"{g['family']} | {rung} | {flag} | {g['n_compared']} | {100 * g['agreement_rate']:.2f}% | "
                    f"{100 * g['token_agreement_rate']:.2f}% | {'' if med is None else med} | "
                    f"{g['answer_divergence']['answer_changing']} | {g['answer_divergence']['answer_preserving']} | "
                    f"{g['median_prompt_tokens']} / {g['reference_median_prompt_tokens']} |"
                )
    all_skips = list(result["skips"]) + list(result.get("cross_arm_skips", []))
    if all_skips:
        lines += ["", f"## Labeled skips ({len(all_skips)})", ""]
        for s in all_skips:
            lines.append(f"- [{s['model']}/{s['dataset']}] {s['where']}: {s['reason']}")
    return "\n".join(lines) + "\n"


def _parse_cross_arm(specs: Sequence[str]) -> List[Tuple[str, str]]:
    """'CELL:REF' -> (cell_arm, reference_arm); malformed or self-paired is a usage error (ValueError)."""
    pairs: List[Tuple[str, str]] = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 2 or not parts[0] or not parts[1] or parts[0] == parts[1]:
            raise ValueError(
                f"--cross-arm {spec!r}: expected CELL_ARM:REFERENCE_ARM with two distinct, non-empty arm names"
            )
        if (parts[0], parts[1]) in pairs:
            raise ValueError(f"--cross-arm {spec!r} given twice")
        pairs.append((parts[0], parts[1]))
    return pairs


_CAMPAIGN_SUMMARY_MD = "token_divergence_summary.md"


def _run_campaign_cli(args: argparse.Namespace) -> int:
    try:
        result = compute_campaign_divergence(
            args.campaign_root, tokenizer=args.tokenizer, cross_arm_pairs=tuple(args.cross_arm_pairs)
        )
    except RuntimeError as exc:
        # CampaignDivergenceError, TokenizerUnavailableError, and the fail-closed
        # import lanes — all loud (exit 2), unlike the pilot lane's benign SKIP.
        print(f"[divergence] REFUSED: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    targets = {
        (str(r["model"]), str(r["dataset"])): out_dir
        / f"token_divergence__{r['model']}__{r['dataset']}.json"
        for r in result["reports"]
    }
    md_path = out_dir / _CAMPAIGN_SUMMARY_MD
    existing = sorted(str(p) for p in list(targets.values()) + [md_path] if p.exists())
    if existing and not args.force:
        # Mirror of build_floor_table: a silently rewritten report cannot be
        # audited against the run that minted it.
        print(
            "[divergence] REFUSED: output artifact(s) already exist — "
            "refusing to silently rewrite a divergence report: "
            + ", ".join(existing)
            + " ; pass --force to overwrite deliberately.",
            file=sys.stderr,
        )
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    for report in result["reports"]:
        path = targets[(str(report["model"]), str(report["dataset"]))]
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    md_path.write_text(_campaign_summary_markdown(result), encoding="utf-8")
    _print_campaign_summary(result, targets)
    print(f"[divergence] -> {md_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Charter sec. 8.9 T=0 divergence: pilot arm-vs-reference "
        "(--results-dir) or campaign per-model engine-pair (--campaign-root)"
    )
    ap.add_argument(
        "--results-dir",
        default=None,
        help="PILOT lane: dir of baseline subdirs (like statistical_tests).",
    )
    ap.add_argument("--reference", default="no_cache", help="Pilot lane: reference arm dir name (default: no_cache).")
    ap.add_argument("--output", default=None, help="Pilot lane: path to write the JSON summary.")
    ap.add_argument(
        "--campaign-root",
        default=None,
        help="CAMPAIGN lane (T6.4): a RESULTS_LAYOUT-v2 tree; computes the "
        "sec. 8.9 stats per model x engine-pair. Requires --out.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Campaign lane: directory for one JSON report per (model, dataset).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Campaign lane: overwrite existing report artifacts (refused "
        "otherwise, mirroring build_floor_table).",
    )
    ap.add_argument(
        "--cross-arm",
        action="append",
        default=[],
        metavar="CELL_ARM:REFERENCE_ARM",
        help="Campaign lane (Spec Block A2): compare two arms on the same engine for every "
        "group both serve, e.g. corpus-comp:corpus-reuse (the fp8 KV cache against its bf16 twin). Repeatable.",
    )
    ap.add_argument(
        "--tokenizer",
        default="whitespace",
        help="Tokenizer for first-divergence positions: 'whitespace' (default) or an HF "
             "tokenizer name (e.g. the arm's serving model). HF load failure is fail-closed.",
    )
    args = ap.parse_args(argv)

    if bool(args.campaign_root) == bool(args.results_dir):
        ap.error(
            "exactly one of --results-dir (pilot layout) or --campaign-root "
            "(RESULTS_LAYOUT-v2 campaign tree) is required"
        )
    if args.campaign_root:
        if args.output:
            ap.error("--output belongs to the pilot lane; the campaign lane writes per-(model, dataset) reports under --out")
        if not args.out:
            ap.error("--campaign-root requires --out (directory for the per-(model, dataset) reports)")
        try:
            args.cross_arm_pairs = _parse_cross_arm(args.cross_arm)
        except ValueError as exc:
            ap.error(str(exc))
        return _run_campaign_cli(args)
    if args.out or args.force or args.cross_arm:
        ap.error("--out/--force/--cross-arm belong to the campaign lane (--campaign-root)")

    try:
        summary = compute_divergence(args.results_dir, args.reference, tokenizer=args.tokenizer)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[divergence] SKIP: {exc}", file=sys.stderr)
        return 0  # non-fatal: absent reference is a skip, not a run failure

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\n[divergence] greedy output vs '{args.reference}' (near-lossless quantification)")
    print(f"{'arm':<28}{'n':>7}{'raw %':>9}{'norm %':>9}{'ans-chg %':>11}{'med 1st-div':>13}")
    for r in summary["arms"]:
        ans_div = r["answer_divergence"]
        first_div = r["first_divergence"]
        med = first_div["median_position"]
        print(f"{r['arm']:<28}{r['n_compared']:>7}"
              f"{100 * r['raw_divergence_rate']:>8.2f}%{100 * r['normalized_divergence_rate']:>8.2f}%"
              f"{100 * ans_div['answer_changing_rate']:>10.2f}%"
              f"{(str(med) if med is not None else '-'):>13}")
    repro = [r for r in summary["reproducibility"] if r["n_groups"]]
    if repro:
        print(f"\n[divergence] per-cell reproducibility violations across repeats (sec. 8.9)")
        print(f"{'arm':<28}{'groups':>8}{'violations':>12}{'rate %':>9}")
        for r in repro:
            rate = r["violation_rate"]
            print(f"{r['arm']:<28}{r['n_groups']:>8}{r['n_violations']:>12}"
                  f"{(100 * rate if rate is not None else 0):>8.2f}%")
    if args.output:
        print(f"[divergence] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
