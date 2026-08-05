"""Wiring tests for the 2026-08-04 rescore_quality handoffs.

Covers the two construction-program handoffs closed in
``scripts/4_analysis/rescore_quality.py`` (CAGE_TECHNICAL_REVIEW_2026-08-04.md,
Construction Addendum):

1. ``--batch-size`` wiring (review §4.6 L3, charter D8 §8.1): re-scoring routes
   through ``QualityEvaluator.batch_evaluate``; default (no flag) preserves the
   historical sequential row-by-row behavior via ``batched=False``, an integer
   enables the cross-row batched path forwarding the value as
   ``nli_batch_size``. Output equivalence batched-vs-sequential is proven on a
   synthetic evidence file, at the function level AND end-to-end through
   ``main()``.

2. Scoring-manifest instrument provenance (review §4.8, charter D8 §8.1 drift
   audit): every ``--scoring-run-id`` pass embeds the canonical
   ``src.observability.provenance.instrument_versions()`` package-version map
   (fast mode included -- absent packages record None, never vanish), the
   instrument model ids in use, and the calibration id, replacing (not
   duplicating) the old full-mode-only ``instruments`` field.

No GPU, no network: all scoring runs the fast (model-free) path; constructors
never lazy-load a model stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rescore_quality as rq  # noqa: E402
from src.analysis.stats.ledger import (  # noqa: E402
    hash_artifacts,
    verify_ledger,
    write_ledger,
)
from src.evaluation.quality import QualityEvaluator  # noqa: E402
from src.observability.provenance import SCORING_STACK_PACKAGES  # noqa: E402

RUN_ID = "20260804-1200-a-qwen3-14b"

#: Synthetic evidence exercising every parser/scorer branch the rescorer owns:
#: plain answerable row, B4 abstention (with a stale grounding score the
#: re-score must expose as old_grounding_score), M5 all-answers max-over-golds
#: with stringified contexts (older-run tolerance), and a B3d dual-scoring row
#: carrying pre-compression original_contexts.
EVIDENCE_ROWS: list[dict[str, Any]] = [
    {
        "example_id": "e0",
        "question": "What color is the sky?",
        "used_contexts": ["The sky is blue."],
        "generated_answer": "blue",
        "reference_answer": "blue",
        "baseline": "B1",
        "repeat_index": 0,
    },
    {
        "example_id": "e1",
        "question": "Who wrote the lost manuscript?",
        "used_contexts": ["An unrelated passage."],
        "generated_answer": "I don't know.",
        "reference_answer": "",
        "baseline": "B1",
        "repeat_index": 0,
        "grounding_score": 0.9,
    },
    {
        "example_id": "e2",
        "question": "What is the capital of France?",
        "used_contexts": json.dumps(["Paris is the capital of France."]),
        "generated_answer": "Paris",
        "reference_answer": "the capital city",
        "all_answers": ["the capital city", "Paris"],
        "baseline": "B3",
        "repeat_index": 1,
    },
    {
        "example_id": "e3",
        "question": "What does CAG stand for?",
        "used_contexts": ["compressed ctx"],
        "generated_answer": "cache-augmented generation",
        "reference_answer": "cache-augmented generation",
        "original_contexts": ["Cache-augmented generation preloads the KV cache."],
        "baseline": "B6",
        "repeat_index": 0,
    },
]


def _write_evidence(path: Path, rows: list[dict[str, Any]] = EVIDENCE_ROWS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in rows]
    lines.insert(1, "")  # blank line: the parser must tolerate it
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fast_evaluator() -> QualityEvaluator:
    """Model-free evaluator: exactly what _build_evaluator makes without --full."""
    return QualityEvaluator(
        use_nli=False,
        use_embeddings=False,
        use_bertscore=False,
        use_rouge=False,
        use_lettucedetect=False,
        device="cpu",
    )


def _args(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(full=False, device="cpu", apply=False, batch_size=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def _make_v2_tree(root: Path) -> Path:
    """Minimal RESULTS_LAYOUT v2 run root: manifest + cells evidence + seal."""
    run = root / RUN_ID
    (run / "cells").mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"run_id": RUN_ID}), encoding="utf-8"
    )
    for cell in ("B1", "B3"):
        _write_evidence(run / "cells" / cell / "window_squad_v2-01" / "qa_evidence.jsonl")
    sealed = [p for p in sorted(run.rglob("*")) if p.is_file()]
    write_ledger(hash_artifacts(sealed, base_dir=run), run / "ledger.json")
    return run


# ---------------------------------------------------------------------------
# Handoff 1: --batch-size wiring
# ---------------------------------------------------------------------------


def test_batch_size_flag_default_none_and_parses_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["rescore_quality.py", "--run-root", "x"])
    assert rq.parse_args().batch_size is None  # default: sequential behavior
    monkeypatch.setattr(
        sys, "argv", ["rescore_quality.py", "--run-root", "x", "--batch-size", "8"]
    )
    assert rq.parse_args().batch_size == 8


@pytest.mark.parametrize("bad", ["0", "-4", "nope"])
def test_batch_size_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    """Fail-closed: a non-positive or non-integer batch size never starts a run."""
    monkeypatch.setattr(
        sys, "argv", ["rescore_quality.py", "--run-root", "x", "--batch-size", bad]
    )
    with pytest.raises(SystemExit) as exc:
        rq.parse_args()
    assert exc.value.code == 2


class _SpyEvaluator:
    """Delegating wrapper recording batch_evaluate keyword arguments."""

    def __init__(self, inner: QualityEvaluator) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def batch_evaluate(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self._inner.batch_evaluate(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_scoring_routes_through_batch_evaluate(tmp_path: Path) -> None:
    """Default -> batched=False (the historical loop); --batch-size N ->
    batched=True with N forwarded as nli_batch_size."""
    ev = _write_evidence(tmp_path / "trial_0" / "qa_evidence.jsonl")
    spy = _SpyEvaluator(_fast_evaluator())

    rq._score_evidence_file(ev, spy)
    assert spy.calls[-1]["batched"] is False

    rq._score_evidence_file(ev, spy, batch_size=7)
    assert spy.calls[-1]["batched"] is True
    assert spy.calls[-1]["nli_batch_size"] == 7


def test_batched_rows_equal_sequential(tmp_path: Path) -> None:
    """Output equivalence, the core --batch-size guarantee: identical row dicts."""
    ev = _write_evidence(tmp_path / "trial_0" / "qa_evidence.jsonl")
    rows_seq = rq._score_evidence_file(ev, _fast_evaluator())
    rows_bat = rq._score_evidence_file(ev, _fast_evaluator(), batch_size=2)

    assert rows_seq == rows_bat
    assert len(rows_seq) == len(EVIDENCE_ROWS)

    by_id = {r["example_id"]: r for r in rows_seq}
    # B4 abstention short-circuit + stale-score exposure survived the rewiring.
    assert by_id["e1"]["abstained"] is True
    assert by_id["e1"]["old_grounding_score"] == 0.9
    assert by_id["e1"]["grounding_score"] is None
    # Plain answerable row still scores.
    assert by_id["e0"]["f1_score"] == 1.0
    # M5 max-over-golds: "Paris" matches via all_answers, not the reference.
    assert by_id["e2"]["exact_match"] == 1.0
    # B3d columns still emitted (None in fast mode: models off).
    assert "faithfulness_source" in by_id["e3"]
    assert by_id["e3"]["faithfulness_source"] is None


def test_main_legacy_batched_csv_identical_to_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through main(): the legacy results_rescored.csv is
    byte-identical with and without --batch-size."""
    seq_root = tmp_path / "seq"
    bat_root = tmp_path / "bat"
    _write_evidence(seq_root / "trial_0" / "qa_evidence.jsonl")
    _write_evidence(bat_root / "trial_0" / "qa_evidence.jsonl")

    monkeypatch.setattr(
        sys, "argv", ["rescore_quality.py", "--run-root", str(seq_root)]
    )
    assert rq.main() == 0
    monkeypatch.setattr(
        sys,
        "argv",
        ["rescore_quality.py", "--run-root", str(bat_root), "--batch-size", "3"],
    )
    assert rq.main() == 0

    seq_csv = (seq_root / "trial_0" / "results_rescored.csv").read_bytes()
    bat_csv = (bat_root / "trial_0" / "results_rescored.csv").read_bytes()
    assert seq_csv == bat_csv
    assert seq_csv  # non-empty: something was actually scored


# ---------------------------------------------------------------------------
# Handoff 2: scoring-manifest instrument provenance (D8 drift audit)
# ---------------------------------------------------------------------------


def test_scoring_manifest_records_instrument_provenance(tmp_path: Path) -> None:
    run = _make_v2_tree(tmp_path)
    assert rq.run_scoring_tree(run, "s01-seq", _args()) == 0

    sdir = run / "scoring" / "s01-seq"
    man = json.loads((sdir / "scoring_manifest.json").read_text(encoding="utf-8"))

    # D8 §8.1: EVERY pass (fast included) records the full canonical scoring
    # stack -- one entry per package, version string or explicit None.
    assert set(man["instrument_versions"]) == set(SCORING_STACK_PACKAGES)
    for pkg, ver in man["instrument_versions"].items():
        assert ver is None or isinstance(ver, str), (pkg, ver)

    # Fast mode consults no model stack: honest empty mapping, key present.
    assert man["instrument_models"] == {}
    assert "calibration_id" in man
    assert man["batch_size"] is None
    # Renamed, not duplicated: the old full-mode-only field is gone.
    assert "instruments" not in man

    # The manifest is still sealed by the pass's own ledger.
    assert verify_ledger(sdir / "ledger.json", sdir) == []


def test_scoring_tree_batched_pass_equivalent_and_recorded(tmp_path: Path) -> None:
    """A batched scoring pass writes byte-identical qa_scores.jsonl and records
    its batch size in the manifest."""
    run = _make_v2_tree(tmp_path)
    assert rq.run_scoring_tree(run, "s01-seq", _args()) == 0
    assert rq.run_scoring_tree(run, "s02-bat", _args(batch_size=2)) == 0

    seq_dir = run / "scoring" / "s01-seq"
    bat_dir = run / "scoring" / "s02-bat"
    seq_scores = sorted(seq_dir.glob("cells/*/window_*/qa_scores.jsonl"))
    assert seq_scores
    for sp in seq_scores:
        bp = bat_dir / sp.relative_to(seq_dir)
        assert bp.read_bytes() == sp.read_bytes()

    man = json.loads(
        (bat_dir / "scoring_manifest.json").read_text(encoding="utf-8")
    )
    assert man["batch_size"] == 2


def test_instrument_models_full_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """With every instrument enabled, _instrument_models records the resolved
    constructor-time id per instrument (no model is lazy-loaded)."""
    for var in ("CAGE_DISABLE_LETTUCEDETECT", "CAGE_CLAIM_CHECKER"):
        monkeypatch.delenv(var, raising=False)
    ev = QualityEvaluator(
        use_nli=True,
        use_embeddings=True,
        use_bertscore=True,
        use_rouge=True,
        use_lettucedetect=True,
        device="cpu",
    )
    assert rq._instrument_models(ev) == {
        "nli": ev.nli_model_name,
        "claim_checker": ev.claim_checker_name,
        "embedding": ev.embedding_model_name,
        "bertscore": ev.bertscore_model_name,
        "lettucedetect": ev.lettucedetect_model_name,
    }


def test_scoring_tree_tolerates_pre_batchsize_namespace(tmp_path: Path) -> None:
    """Back-compat: a hand-built Namespace without batch_size (older callers,
    existing tests) still runs, defaulting to the sequential path."""
    run = _make_v2_tree(tmp_path)
    ns = argparse.Namespace(full=False, device="cpu", apply=False)
    assert rq.run_scoring_tree(run, "s03-compat", ns) == 0
    man = json.loads(
        (run / "scoring" / "s03-compat" / "scoring_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert man["batch_size"] is None
