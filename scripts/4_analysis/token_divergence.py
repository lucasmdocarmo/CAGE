#!/usr/bin/env python3
"""Token-divergence metric: how often does an arm's greedy output differ from no_cache?

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
import json
import re
import statistics
import string
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

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

        n = len(keys)
        raw_div = 0
        norm_div = 0
        token_agree = 0
        positions: List[int] = []
        answer_changing = 0
        answer_preserving = 0
        bases: set[str] = set()
        for k in keys:
            arm_text, ref_text = ans[k].answer, ref[k].answer
            raw_divergent = arm_text.strip() != ref_text.strip()
            if _normalize(arm_text) != _normalize(ref_text):
                norm_div += 1
            if not raw_divergent:
                token_agree += 1  # identical strings are token-identical under any tokenizer
                continue
            raw_div += 1
            # Charter sec. 8.9 stat 2: first-divergence token position.
            pos = _first_divergence_position(tokenize(arm_text), tokenize(ref_text))
            if pos is None:
                token_agree += 1  # raw-divergent but token-identical (e.g. whitespace-only)
            else:
                positions.append(pos)
            # Charter sec. 8.9 stat 3 (the decisive one): answer-changing vs -preserving.
            gold = ref[k].reference_answer if ref[k].reference_answer is not None else ans[k].reference_answer
            changing, basis = _classify_pair(arm_text, ref_text, gold, quality)
            bases.add(basis)
            if changing:
                answer_changing += 1
            else:
                answer_preserving += 1

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


def main() -> int:
    ap = argparse.ArgumentParser(description="Token-divergence vs a reference arm")
    ap.add_argument("--results-dir", required=True, help="Dir of baseline subdirs (like statistical_tests).")
    ap.add_argument("--reference", default="no_cache", help="Reference arm dir name (default: no_cache).")
    ap.add_argument("--output", default=None, help="Path to write the JSON summary.")
    ap.add_argument(
        "--tokenizer",
        default="whitespace",
        help="Tokenizer for first-divergence positions: 'whitespace' (default) or an HF "
             "tokenizer name (e.g. the arm's serving model). HF load failure is fail-closed.",
    )
    args = ap.parse_args()

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
