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
    base: dict[str, Any] = dict(
        full=False, device="cpu", apply=False, batch_size=None,
        allow_duplicates=False,
    )
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
    rows_seq, n_dup_seq = rq._score_evidence_file(ev, _fast_evaluator())
    rows_bat, n_dup_bat = rq._score_evidence_file(ev, _fast_evaluator(), batch_size=2)
    assert n_dup_seq == n_dup_bat == 0

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


# ---------------------------------------------------------------------------
# Task #127: duplicate guard (audit H3) + per-metric denominators (audit H11)
# ---------------------------------------------------------------------------

#: Two rows sharing the FULL (example_id, repeat_index, record_index) key --
#: a true duplicate -- plus a replayed pair disambiguated by record_index only.
DUP_EVIDENCE_ROWS: list[dict[str, Any]] = [
    {
        "example_id": "d0",
        "question": "Q?",
        "used_contexts": ["ctx"],
        "generated_answer": "first",
        "reference_answer": "second",
        "baseline": "B1",
        "repeat_index": 0,
    },
    {
        "example_id": "d0",
        "question": "Q?",
        "used_contexts": ["ctx"],
        "generated_answer": "second",
        "reference_answer": "second",
        "baseline": "B1",
        "repeat_index": 0,
    },
]

REPLAY_EVIDENCE_ROWS: list[dict[str, Any]] = [
    {
        "example_id": "r0",
        "question": "Q?",
        "used_contexts": ["ctx"],
        "generated_answer": "a",
        "reference_answer": "a",
        "baseline": "B1",
        "repeat_index": 0,
        "record_index": 3,
        "arrival_s": 0.5,
    },
    {
        "example_id": "r0",
        "question": "Q?",
        "used_contexts": ["ctx"],
        "generated_answer": "a",
        "reference_answer": "a",
        "baseline": "B1",
        "repeat_index": 0,
        "record_index": 9,
        "arrival_s": 2.5,
    },
]


def test_duplicate_rows_refused_by_default(tmp_path: Path) -> None:
    """Identical (example_id, repeat_index, record_index) triples raise, with
    counts and the offending keys in the message (aligned with instrument-B)."""
    ev = _write_evidence(tmp_path / "trial_0" / "qa_evidence.jsonl", DUP_EVIDENCE_ROWS)
    with pytest.raises(rq.DuplicateEvidenceError) as exc:
        rq._score_evidence_file(ev, _fast_evaluator())
    msg = str(exc.value)
    assert "d0" in msg
    assert "1 duplicate" in msg
    assert "--allow-duplicates" in msg


def test_replayed_rows_with_record_index_are_not_duplicates(tmp_path: Path) -> None:
    """Open-loop replay disambiguation: same example_id, different record_index
    -> two distinct rows, no refusal, record_index carried into the score rows."""
    ev = _write_evidence(
        tmp_path / "trial_0" / "qa_evidence.jsonl", REPLAY_EVIDENCE_ROWS
    )
    rows, n_dup = rq._score_evidence_file(ev, _fast_evaluator())
    assert n_dup == 0
    assert [r["record_index"] for r in rows] == [3, 9]


def test_allow_duplicates_keeps_last_and_counts(tmp_path: Path) -> None:
    ev = _write_evidence(tmp_path / "trial_0" / "qa_evidence.jsonl", DUP_EVIDENCE_ROWS)
    rows, n_dup = rq._score_evidence_file(
        ev, _fast_evaluator(), allow_duplicates=True
    )
    assert n_dup == 1
    assert len(rows) == 1
    # keep-LAST: the second occurrence (the exact-match answer) survived.
    assert rows[0]["generated_answer"] == "second"
    assert rows[0]["exact_match"] == 1.0


def test_legacy_main_refuses_duplicates_and_sidecar_on_allow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end legacy mode: refusal is exit 2; --allow-duplicates persists
    n_duplicates_dropped in the accounting sidecar, not just stdout (§9.10)."""
    root = tmp_path / "run"
    _write_evidence(root / "trial_0" / "qa_evidence.jsonl", DUP_EVIDENCE_ROWS)

    monkeypatch.setattr(sys, "argv", ["rescore_quality.py", "--run-root", str(root)])
    assert rq.main() == 2
    assert not (root / "trial_0" / "results_rescored.csv").exists()

    monkeypatch.setattr(
        sys, "argv",
        ["rescore_quality.py", "--run-root", str(root), "--allow-duplicates"],
    )
    assert rq.main() == 0
    accounting = json.loads(
        (root / "trial_0" / "results_rescored.csv.accounting.json").read_text(
            encoding="utf-8"
        )
    )
    assert accounting["n_duplicates_dropped"] == 1
    assert accounting["n_rows_scored"] == 1
    assert (root / "trial_0" / "results_rescored.csv").is_file()


def test_scoring_tree_refuses_duplicates_before_creating_the_tree(
    tmp_path: Path,
) -> None:
    """The duplicate refusal must not burn the append-only scoring_run_id:
    no scoring/<id>/ directory may exist after the refusal."""
    run = _make_v2_tree(tmp_path)
    _write_evidence(
        run / "cells" / "B1" / "window_squad_v2-01" / "qa_evidence.jsonl",
        DUP_EVIDENCE_ROWS,
    )
    # re-seal: _make_v2_tree sealed the original evidence; overwrite the ledger
    # with a fresh one so the precondition checks still pass.
    (run / "ledger.json").unlink()
    sealed = [p for p in sorted(run.rglob("*")) if p.is_file()]
    write_ledger(hash_artifacts(sealed, base_dir=run), run / "ledger.json")

    assert rq.run_scoring_tree(run, "s04-dup", _args()) == 2
    assert not (run / "scoring" / "s04-dup").exists()


def test_scoring_tree_allow_duplicates_persists_counts(tmp_path: Path) -> None:
    """Tree mode with --allow-duplicates: n_duplicates_dropped lands in BOTH the
    per-window quality.json and the scoring manifest total."""
    run = _make_v2_tree(tmp_path)
    _write_evidence(
        run / "cells" / "B1" / "window_squad_v2-01" / "qa_evidence.jsonl",
        DUP_EVIDENCE_ROWS,
    )
    (run / "ledger.json").unlink()
    sealed = [p for p in sorted(run.rglob("*")) if p.is_file()]
    write_ledger(hash_artifacts(sealed, base_dir=run), run / "ledger.json")

    assert rq.run_scoring_tree(run, "s05-dup-ok", _args(allow_duplicates=True)) == 0
    sdir = run / "scoring" / "s05-dup-ok"
    q_b1 = json.loads(
        (sdir / "cells" / "B1" / "window_squad_v2-01" / "quality.json").read_text(
            encoding="utf-8"
        )
    )
    assert q_b1["n_duplicates_dropped"] == 1
    q_b3 = json.loads(
        (sdir / "cells" / "B3" / "window_squad_v2-01" / "quality.json").read_text(
            encoding="utf-8"
        )
    )
    assert q_b3["n_duplicates_dropped"] == 0
    man = json.loads((sdir / "scoring_manifest.json").read_text(encoding="utf-8"))
    assert man["n_duplicates_dropped"] == 1
    assert man["allow_duplicates"] is True


def test_quality_aggregate_emits_per_metric_denominators(tmp_path: Path) -> None:
    """Audit H11: every aggregated key carries n and n_none alongside mean, so
    a metric scored on 1/4 rows is distinguishable from full coverage; mean is
    None (never 0) when nothing scored; provenance numerics are NOT folded in."""
    ev = _write_evidence(tmp_path / "trial_0" / "qa_evidence.jsonl")
    rows, _ = rq._score_evidence_file(ev, _fast_evaluator())
    agg = rq._quality_aggregate(rows, n_duplicates_dropped=2)

    assert agg["rows"] == len(EVIDENCE_ROWS)
    assert agg["n_duplicates_dropped"] == 2
    # model-free metric: scored on every row.
    f1 = agg["metrics"]["f1_score"]
    assert f1["n"] == len(EVIDENCE_ROWS)
    assert f1["n_none"] == 0
    assert isinstance(f1["mean"], float)
    # model metric in fast mode: nothing scored -> mean None, full n_none.
    g = agg["metrics"]["grounding_score"]
    assert g == {"mean": None, "n": 0, "n_none": len(EVIDENCE_ROWS)}
    # partial coverage stays visible: abstention_precision only on abstained rows.
    ap = agg["metrics"]["abstention_precision"]
    assert ap["n"] == 1
    assert ap["n_none"] == len(EVIDENCE_ROWS) - 1
    # provenance numerics (e.g. the stale old_grounding_score on e1) are NOT
    # aggregation keys -- the H11 fold-in defect stays fixed in BOTH views.
    assert "old_grounding_score" not in agg["metrics"]
    assert "record_index" not in agg["metrics"]
    assert "old_grounding_score" not in agg["means"]
    assert "record_index" not in agg["means"]
    # the back-compat means view carries scored keys only and agrees with the
    # authoritative denominator-bearing metrics mapping.
    assert set(agg["means"]) == {
        k for k, m in agg["metrics"].items() if m["n"] > 0
    }
    assert agg["means"]["f1_score"] == agg["metrics"]["f1_score"]["mean"]
    assert "grounding_score" not in agg["means"]  # n=0 in fast mode


def test_tree_quality_json_carries_denominators(tmp_path: Path) -> None:
    run = _make_v2_tree(tmp_path)
    assert rq.run_scoring_tree(run, "s06-denoms", _args()) == 0
    quality = json.loads(
        (
            run / "scoring" / "s06-denoms" / "cells" / "B1"
            / "window_squad_v2-01" / "quality.json"
        ).read_text(encoding="utf-8")
    )
    for entry in quality["metrics"].values():
        assert set(entry) == {"mean", "n", "n_none"}
        assert entry["n"] + entry["n_none"] == quality["rows"]


# ---------------------------------------------------------------------------
# Task #130 decision (b): sidecar-only pilot rescoring — --apply refuses the
# read-only pilot archive (RESULTS_LAYOUT §7)
# ---------------------------------------------------------------------------


def _results_csv_for(evidence_rows: list[dict[str, Any]]) -> str:
    header = "example_id,repeat_index,error\n"
    return header + "".join(
        f"{r['example_id']},{r.get('repeat_index', 0)},\n" for r in evidence_rows
    )


def test_apply_refused_inside_pilot_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--apply on an archive tree: exit 2, doctrine cited, NOTHING written —
    no rescored CSV, no backup, results.csv byte-identical."""
    archive = tmp_path / "archive"
    run = archive / "phase2" / "2026-07-16_0143_qwen3-8b_100x3"
    _write_evidence(run / "trial_0" / "qa_evidence.jsonl")
    results_csv = run / "trial_0" / "results.csv"
    results_csv.write_text(_results_csv_for(EVIDENCE_ROWS), encoding="utf-8")
    before = results_csv.read_bytes()

    monkeypatch.setenv(rq.PILOT_ARCHIVE_ENV, str(archive))
    monkeypatch.setattr(
        sys, "argv", ["rescore_quality.py", "--run-root", str(run), "--apply"]
    )
    assert rq.main() == 2
    err = capsys.readouterr().err
    assert "read-only pilot archive" in err
    assert "§7" in err
    assert "sidecar" in err
    assert results_csv.read_bytes() == before
    assert not (run / "trial_0" / "results_rescored.csv").exists()
    assert not (run / "trial_0" / "results.csv.pre_rescore").exists()


def test_sidecar_rescore_still_allowed_inside_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WITHOUT --apply the archive stays rescorable: the sidecar CSV is the
    sanctioned product; results.csv is untouched."""
    archive = tmp_path / "archive"
    run = archive / "phase2" / "runx"
    _write_evidence(run / "trial_0" / "qa_evidence.jsonl")
    results_csv = run / "trial_0" / "results.csv"
    results_csv.write_text(_results_csv_for(EVIDENCE_ROWS), encoding="utf-8")
    before = results_csv.read_bytes()

    monkeypatch.setenv(rq.PILOT_ARCHIVE_ENV, str(archive))
    monkeypatch.setattr(sys, "argv", ["rescore_quality.py", "--run-root", str(run)])
    assert rq.main() == 0
    assert (run / "trial_0" / "results_rescored.csv").is_file()
    assert results_csv.read_bytes() == before


def test_apply_keeps_working_outside_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(rq.PILOT_ARCHIVE_ENV, str(tmp_path / "elsewhere"))
    root = tmp_path / "scratch"
    _write_evidence(root / "trial_0" / "qa_evidence.jsonl")
    results_csv = root / "trial_0" / "results.csv"
    results_csv.write_text(_results_csv_for(EVIDENCE_ROWS), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv", ["rescore_quality.py", "--run-root", str(root), "--apply"]
    )
    assert rq.main() == 0
    assert (root / "trial_0" / "results.csv.pre_rescore").is_file()
    applied = results_csv.read_text(encoding="utf-8")
    assert "f1_score" in applied.splitlines()[0]  # quality columns merged in


def test_pilot_archive_default_is_repo_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Archive-root detection is EXPLICIT: repo results/ unless overridden."""
    monkeypatch.delenv(rq.PILOT_ARCHIVE_ENV, raising=False)
    assert rq._pilot_archive_root() == (rq.REPO_ROOT / "results").resolve()
    monkeypatch.setenv(rq.PILOT_ARCHIVE_ENV, "/tmp/somewhere-else")
    assert rq._pilot_archive_root() == Path("/tmp/somewhere-else").resolve()


# ---------------------------------------------------------------------------
# Task #130 decision (d): --abandon (audit H10 — free a crashed pass's id;
# completed passes are audit record)
# ---------------------------------------------------------------------------


def _crashed_pass(run: Path, sid: str) -> Path:
    """A pass that died mid-flight: partial cells, manifest, NO ledger."""
    pass_dir = run / "scoring" / sid
    wdir = pass_dir / "cells" / "B1" / "window_squad_v2-01"
    wdir.mkdir(parents=True)
    (wdir / "qa_scores.jsonl").write_text("{}\n", encoding="utf-8")
    (pass_dir / "scoring_manifest.json").write_text("{}", encoding="utf-8")
    return pass_dir


def test_abandon_crashed_pass_frees_the_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _make_v2_tree(tmp_path)
    crashed = _crashed_pass(run, "s07-crash")

    monkeypatch.setattr(sys, "argv", [
        "rescore_quality.py", "--run-root", str(run),
        "--abandon", "s07-crash", "--reason", "worker OOM mid-pass",
    ])
    assert rq.main() == 0
    assert not crashed.exists()

    abandoned = list((run / "scoring").glob("s07-crash.abandoned-*"))
    assert len(abandoned) == 1
    tomb = json.loads(
        (abandoned[0] / rq.ABANDONED_TOMBSTONE_NAME).read_text(encoding="utf-8")
    )
    assert tomb["scoring_run_id"] == "s07-crash"
    assert tomb["reason"] == "worker OOM mid-pass"
    assert tomb["forced"] is False
    assert tomb["ledger_state"] == "absent"
    assert tomb["present"]["manifest"] is True
    assert tomb["present"]["ledger"] is False
    assert tomb["present"]["n_files"] == 2
    assert tomb["present"]["n_cell_files"] == 1

    # The id is FREE: a clean retry under the same id succeeds.
    assert rq.run_scoring_tree(run, "s07-crash", _args()) == 0


def test_abandon_verified_pass_refused_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _make_v2_tree(tmp_path)
    assert rq.run_scoring_tree(run, "s08-done", _args()) == 0

    with pytest.raises(rq.ScoringAbandonError, match="VERIFIED complete ledger"):
        rq.abandon_scoring_pass(run, "s08-done", reason="oops")
    assert (run / "scoring" / "s08-done").is_dir()  # untouched

    monkeypatch.setattr(sys, "argv", [
        "rescore_quality.py", "--run-root", str(run),
        "--abandon", "s08-done", "--reason", "oops",
    ])
    assert rq.main() == 2
    assert "VERIFIED" in capsys.readouterr().err

    # --force-abandon overrides, and the tombstone records the force.
    target = rq.abandon_scoring_pass(
        run, "s08-done", reason="superseded by s09", force=True
    )
    tomb = json.loads(
        (target / rq.ABANDONED_TOMBSTONE_NAME).read_text(encoding="utf-8")
    )
    assert tomb["forced"] is True
    assert tomb["ledger_state"] == "verified"
    assert tomb["present"]["ledger"] is True


def test_abandon_argument_refusals(tmp_path: Path) -> None:
    run = _make_v2_tree(tmp_path)
    _crashed_pass(run, "s07-crash")
    # empty reason
    with pytest.raises(rq.ScoringAbandonError, match="--reason"):
        rq.abandon_scoring_pass(run, "s07-crash", reason="   ")
    # grammar violation (also blocks path escapes)
    with pytest.raises(rq.ScoringAbandonError, match="grammar"):
        rq.abandon_scoring_pass(run, "../evil", reason="r")
    # missing pass
    with pytest.raises(rq.ScoringAbandonError, match="no scoring pass"):
        rq.abandon_scoring_pass(run, "s99-none", reason="r")


def test_abandon_cli_is_exclusive_with_scoring_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = _make_v2_tree(tmp_path)
    _crashed_pass(run, "s07-crash")
    monkeypatch.setattr(sys, "argv", [
        "rescore_quality.py", "--run-root", str(run),
        "--abandon", "s07-crash", "--reason", "r",
        "--scoring-run-id", "s10-new",
    ])
    assert rq.main() == 2
    assert "exclusive" in capsys.readouterr().err
    assert (run / "scoring" / "s07-crash").is_dir()  # nothing happened
