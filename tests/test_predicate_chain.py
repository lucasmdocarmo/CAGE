"""Task #119 — the §8.5 veridicality predicate + decoupled-scoring join chain.

Covers (Topic-6 F1-F4, MyDocs/registration/CODE_ASSERTION_2026-08.md):

- src/analysis/predicate.py: the §8.5 component-logic table (span-QA
  correctness branch / Qasper groundedness branch / None-propagation), the
  #127-keyed join (duplicates, unmatched rows BOTH directions, missing ok
  stamp), the explicit-config doctrine (qasper τ = #120, null-fraction bound
  required), and the counted accounting (not-ok nulling, contradiction /
  neutral reported separately);
- scripts/4_analysis/build_predicate_table.py: seal preconditions (raw +
  scoring pass + cross-match), window reconciliation both directions, the
  produced tree shape + manifest + own ledger, qasper-without-τ refusal;
- organize_results.validate_predicate_trees: schema guard + mirror-only rule
  + contamination sweep (predicate.jsonl inside cells/ refuses);
- run_campaign_analysis: the predicate joins the per-query loader (the
  registered #4 predicate leg computes McNemar END-TO-END on real fixture
  artifacts), and the #14 truth-tax executor computes the §9.2 estimand
  (in-regime population, G − Y per window via goodput.evaluate_window,
  ms→s at the registered seam, cross-engine batch-means legs vs the vLLM
  anchor) — plus the loud-refusal pins (missing table, ambiguous tables,
  missing slo_floors / regime artifacts).

The #119 stub-message pin (the "#119 has not landed" claim is GONE while the
missing-column refusal REMAINS) lives in tests/test_campaign_analysis.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_predicate_table as bpt  # noqa: E402
import organize_results as org  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
from src.analysis.cellspec import CellSpec  # noqa: E402
from src.analysis.predicate import (  # noqa: E402
    PredicateConfig,
    PredicateError,
    compute_window_predicate,
    join_window_rows,
    required_verdict_column,
)
from src.analysis.stats.ledger import hash_artifacts, write_ledger  # noqa: E402

RUN_ID = "20260817-0900-a-qwen3-14b"
CAMPAIGN = "camp1"
SESSION = "a"
MODEL = "qwen3-14b"
DATASET = "squad_v2"
SCORING_ID = "s01-fast"
N_EXAMPLES = 16

#: Per-baseline count of predicate-FALSE examples in every F1 window (kept
#: identical across a cell's windows so the loader's cross-window average
#: stays strictly 0/1 for McNemar).
F1_FALSE_COUNT = {"B3": 3, "B6": 6}
#: Per-(engine, ordinal) count of predicate-FALSE examples in F2 windows —
#: sglang pays a higher truth tax than the vllm anchor, with across-window
#: variance so the Welch contrast is defined.
F2_FALSE_COUNT = {("vllm", 1): 1, ("vllm", 2): 2, ("sglang", 1): 6, ("sglang", 2): 8}
SLO_FLOORS = {
    "vllm": {"ttft_s": 0.1, "tpot_s": 0.05},
    "sglang": {"ttft_s": 0.1, "tpot_s": 0.05},
}


# ---------------------------------------------------------------------------
# Row factories (synthetic #127-stamped evidence + sidecar score rows)
# ---------------------------------------------------------------------------


def _evidence_row(i: int, *, ok: bool = True) -> dict[str, Any]:
    return {
        "example_id": f"e{i:03d}",
        "baseline": "blind-token",
        "repeat_index": 0,
        "record_index": None,
        "generated_answer": "" if not ok else f"answer {i}",
        # #127 integrity stamp (evidence_integrity_fields semantics).
        "ok": ok,
        "error": None if ok else "HTTP 500",
        "empty_generation": False,
    }


def _score_row(i: int, *, exact_match: float | None = 1.0, **extra: Any) -> dict[str, Any]:
    return {
        "example_id": f"e{i:03d}",
        "repeat_index": "0",
        "record_index": None,
        "exact_match": exact_match,
        "f1_score": exact_match,
        "grounding_score": None,
        **extra,
    }


def _config(**kw: Any) -> PredicateConfig:
    kw.setdefault("max_null_fraction", 0.5)
    return PredicateConfig(**kw)


# ---------------------------------------------------------------------------
# §8.5 component-logic table (span-QA / qasper / None cases)
# ---------------------------------------------------------------------------


def test_span_qa_branch_component_table() -> None:
    # The abstention-aware EM column carries BOTH registered clauses
    # (normalized EM on answerable items; correct-abstention credit on
    # unanswerables) — the predicate binarizes it at 1.0.
    joined = join_window_rows(
        [_evidence_row(0), _evidence_row(1), _evidence_row(2, ok=False),
         _evidence_row(3)],
        [_score_row(0, exact_match=1.0), _score_row(1, exact_match=0.0),
         _score_row(2, exact_match=1.0), _score_row(3, exact_match=None)],
        window="w",
    )
    rows, summary = compute_window_predicate(joined, DATASET, _config(), window="w")
    by_id = {r["example_id"]: r for r in rows}
    assert by_id["e000"]["predicate"] is True
    assert by_id["e001"]["predicate"] is False
    # not-ok: H2 consumer-side nulling — the scored verdict is DISCARDED.
    assert by_id["e002"]["predicate"] is None
    assert by_id["e002"]["verdict"] is None
    assert by_id["e002"]["predicate_null_reason"] == "not_ok"
    # missing verdict on an ok row: None, counted — never fabricated.
    assert by_id["e003"]["predicate"] is None
    assert by_id["e003"]["predicate_null_reason"] == "missing_verdict"
    assert summary["n_true"] == 1 and summary["n_false"] == 1
    assert summary["n_null"] == 2
    assert summary["n_not_ok_nulled"] == 1
    assert summary["n_missing_verdict"] == 1
    assert summary["predicate_rule"] == "span_qa_em_abstention_aware"


def test_span_qa_refuses_non_binary_exact_match() -> None:
    joined = join_window_rows(
        [_evidence_row(0)], [_score_row(0, exact_match=0.5)], window="w"
    )
    with pytest.raises(PredicateError, match="outside"):
        compute_window_predicate(joined, DATASET, _config(), window="w")


def test_qasper_branch_thresholds_at_explicit_tau() -> None:
    joined = join_window_rows(
        [_evidence_row(0), _evidence_row(1), _evidence_row(2)],
        [
            _score_row(0, grounding_score=0.9),
            _score_row(1, grounding_score=0.5),
            _score_row(2, grounding_score=None),
        ],
        window="w",
    )
    rows, summary = compute_window_predicate(
        joined, "qasper", _config(qasper_tau=0.8), window="w"
    )
    by_id = {r["example_id"]: r for r in rows}
    assert by_id["e000"]["predicate"] is True   # 0.9 >= τ
    assert by_id["e001"]["predicate"] is False  # 0.5 < τ
    assert by_id["e002"]["predicate"] is None   # unscored -> None, counted
    assert summary["predicate_rule"] == "qasper_grounding_at_tau"
    assert summary["verdict_column"] == "grounding_score"


def test_qasper_without_tau_refuses_naming_120() -> None:
    # #120 owns the τ pairing: no silent default, ever.
    joined = join_window_rows(
        [_evidence_row(0)], [_score_row(0, grounding_score=0.9)], window="w"
    )
    with pytest.raises(PredicateError, match="#120"):
        compute_window_predicate(joined, "qasper", _config(), window="w")


def test_non_predicate_dataset_refuses() -> None:
    with pytest.raises(PredicateError, match="predicate universe"):
        required_verdict_column("scbench")


def test_missing_verdict_column_entirely_refuses() -> None:
    # Column absent from EVERY row (not merely None) = the instrument never
    # ran — a different failure from per-row None-propagation.
    joined = join_window_rows(
        [_evidence_row(0)],
        [{"example_id": "e000", "repeat_index": "0", "record_index": None}],
        window="w",
    )
    with pytest.raises(PredicateError, match="absent from"):
        compute_window_predicate(joined, DATASET, _config(), window="w")


def test_null_fraction_bound_refuses_the_window() -> None:
    joined = join_window_rows(
        [_evidence_row(0, ok=False), _evidence_row(1, ok=False), _evidence_row(2)],
        [_score_row(0), _score_row(1), _score_row(2)],
        window="w",
    )
    with pytest.raises(PredicateError, match="max_null_fraction"):
        compute_window_predicate(
            joined, DATASET, _config(max_null_fraction=0.5), window="w"
        )
    # The same rows pass under a deliberately stated looser bound.
    rows, summary = compute_window_predicate(
        joined, DATASET, _config(max_null_fraction=0.7), window="w"
    )
    assert summary["n_null"] == 2 and len(rows) == 3


def test_contradiction_and_neutral_reported_separately() -> None:
    # §8.5: "contradiction and neutral reported separately" — they ride the
    # rows + summary and NEVER flip the predicate.
    joined = join_window_rows(
        [_evidence_row(0)],
        [_score_row(0, exact_match=1.0,
                    faithfulness_contradiction=0.9, faithfulness_neutral=0.1)],
        window="w",
    )
    rows, summary = compute_window_predicate(joined, DATASET, _config(), window="w")
    assert rows[0]["predicate"] is True  # contradiction never enters
    assert rows[0]["faithfulness_contradiction"] == 0.9
    sep = summary["reported_separately"]
    assert sep["faithfulness_contradiction"] == {"n": 1, "mean": 0.9}
    assert sep["faithfulness_neutral"] == {"n": 1, "mean": 0.1}


# ---------------------------------------------------------------------------
# Join-chain refusals (F2-F4; #127 keys)
# ---------------------------------------------------------------------------


def test_join_refuses_unmatched_rows_both_directions() -> None:
    with pytest.raises(PredicateError) as exc:
        join_window_rows(
            [_evidence_row(0), _evidence_row(1)],
            [_score_row(1), _score_row(2)],
            window="w",
        )
    msg = str(exc.value)
    assert "1 evidence row(s) WITHOUT a score row" in msg
    assert "1 score row(s) WITHOUT a serving parent" in msg


def test_join_refuses_duplicate_keys() -> None:
    with pytest.raises(PredicateError, match="duplicate"):
        join_window_rows(
            [_evidence_row(0), _evidence_row(0)], [_score_row(0)], window="w"
        )
    with pytest.raises(PredicateError, match="duplicate"):
        join_window_rows(
            [_evidence_row(0)], [_score_row(0), _score_row(0)], window="w"
        )


def test_join_distinguishes_replay_rows_by_record_index() -> None:
    # H3: open-loop replays duplicate example_id BY DESIGN — the
    # record_index disambiguator keeps them distinct joinable rows.
    ev = [dict(_evidence_row(0), record_index=0), dict(_evidence_row(0), record_index=1)]
    sc = [dict(_score_row(0), record_index=0), dict(_score_row(0), record_index=1)]
    joined = join_window_rows(ev, sc, window="w")
    assert len(joined) == 2


def test_join_refuses_pre_127_evidence_without_ok_stamp() -> None:
    ev = _evidence_row(0)
    del ev["ok"]
    with pytest.raises(PredicateError, match="'ok' integrity stamp"):
        join_window_rows([ev], [_score_row(0)], window="w")


def test_config_is_explicit_no_silent_defaults() -> None:
    with pytest.raises(TypeError):
        PredicateConfig()  # type: ignore[call-arg]  # the bound is REQUIRED
    with pytest.raises(PredicateError):
        PredicateConfig(max_null_fraction=1.5)
    with pytest.raises(PredicateError):
        PredicateConfig(max_null_fraction=0.1, qasper_tau=2.0)


# ---------------------------------------------------------------------------
# Fixture tree: sealed raw run + sealed scoring pass
# ---------------------------------------------------------------------------


def _f1_specs() -> list[CellSpec]:
    return [CellSpec.from_baseline(b, model=MODEL) for b in ("B3", "B6")]  # type: ignore[arg-type]


def _f2_specs() -> list[CellSpec]:
    return [
        CellSpec(
            arm="gold-fresh", retriever="none", policy="none", topology="single",
            engine=engine, model=MODEL, family="F2",  # type: ignore[arg-type]
            budget_r=0.5, rate_frac=0.9,
        )
        for engine in ("vllm", "sglang")
    ]


def _n_false_for(spec: CellSpec, baseline: str, ordinal: int) -> int:
    if spec.family == "F2":
        return F2_FALSE_COUNT[(spec.engine, ordinal)]
    return F1_FALSE_COUNT[baseline]


def _build_sealed_run(tmp_path: Path) -> Path:
    """RESULTS_LAYOUT §1 tree: F1 B3/B6 + F2 vllm/sglang, 2 windows each."""
    run_dir = tmp_path / "results" / CAMPAIGN / SESSION / RUN_ID
    run_dir.mkdir(parents=True)
    manifest = {
        "campaign": CAMPAIGN, "session": SESSION, "run_id": RUN_ID,
        "model": MODEL, "git_sha": "deadbeef", "git_dirty": False,
        "engine": "vllm", "engine_version": "0.0-test", "seed": 1,
        "provider": "test", "hardware": "test-gpu",
        "dataset_manifests_sha256": "0" * 64, "cellspec_schema_version": 1,
        "created_utc": "2026-08-17T09:00:00Z",
        # E3 floor artifact the #14 executor consumes (§6.1 SLO pair).
        "slo_floors": SLO_FLOORS,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    sealed: list[Path] = []
    for spec in _f1_specs() + _f2_specs():
        baseline = org.BASELINE_OF_CELL.get((spec.arm, spec.retriever), "")
        cell_dir = run_dir / "cells" / spec.to_row_key()
        cell_dir.mkdir(parents=True)
        windows: dict[str, dict[str, Any]] = {}
        for ordinal in (1, 2):
            k = f"{DATASET}-{ordinal:02d}"
            windows[k] = {
                "dataset": DATASET, "seed": 1, "rep": ordinal,
                "budget_r": spec.budget_r, "rate_frac": spec.rate_frac,
                "t_start": 0.0, "t_end": 60.0,
            }
            wdir = cell_dir / f"window_{k}"
            wdir.mkdir()
            n_false = _n_false_for(spec, baseline, ordinal)
            requests_lines = []
            evidence_lines = []
            for i in range(N_EXAMPLES):
                ok = i != N_EXAMPLES - 1  # one serving failure per window
                requests_lines.append(json.dumps({
                    "example_id": f"e{i:03d}",
                    "ok": ok,
                    "ttft_ms": 200.0 + i if ok else None,
                    "tpot_ms": 20.0 if ok else None,
                    "latency_ms": 250.0 + i,
                }))
                evidence_lines.append(json.dumps(_evidence_row(i, ok=ok)))
            payloads = {
                "requests.jsonl": "\n".join(requests_lines) + "\n",
                "qa_evidence.jsonl": "\n".join(evidence_lines) + "\n",
                "engine_metrics.json": json.dumps({"snapshot": "x"}),
                "cage_stats.jsonl": json.dumps({"ts_s": 0.0, "kv_cache_usage": 0.95}) + "\n",
                # §6.1 regime referee (#126): every fixture window is
                # certified IN_REGIME so it enters the §9.2 population.
                "regime.json": json.dumps({
                    "schema_version": 1, "t_start": 0.0, "t_end": 60.0,
                    "telemetry_ok": True, "label": "IN_REGIME",
                    "refusal_reason": None,
                }),
                # noted below: n_false examples score EM=0 in this window
                "_n_false": None,
            }
            del payloads["_n_false"]
            for name, text in payloads.items():
                path = wdir / name
                path.write_text(text, encoding="utf-8")
                sealed.append(path)
        cell_json = cell_dir / "cell.json"
        cell_json.write_text(
            json.dumps({
                "cellspec": spec.to_flat_dict(),
                "baseline": baseline,
                "windows": windows,
            }),
            encoding="utf-8",
        )
        sealed.append(cell_json)
    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")
    return run_dir


def _write_scoring_pass(run_dir: Path, scoring_id: str = SCORING_ID) -> Path:
    """A sealed RESULTS_LAYOUT §6 scoring pass mirroring every window."""
    raw_sha = json.loads((run_dir / "ledger.json").read_text())["entries_sha256"]
    scoring_dir = run_dir / "scoring" / scoring_id
    written: list[Path] = []
    for ev_path in sorted(run_dir.glob("cells/*/window_*/qa_evidence.jsonl")):
        rel_window = ev_path.parent.relative_to(run_dir)
        row_key = rel_window.parts[1]
        spec = CellSpec.from_flat_dict(
            json.loads((run_dir / "cells" / row_key / "cell.json").read_text())["cellspec"]
        )
        baseline = org.BASELINE_OF_CELL.get((spec.arm, spec.retriever), "")
        ordinal = int(rel_window.name.rsplit("-", 1)[1])
        n_false = _n_false_for(spec, baseline, ordinal)
        out_window = scoring_dir / rel_window
        out_window.mkdir(parents=True)
        score_lines = []
        for i in range(N_EXAMPLES):
            # predicate-FALSE examples occupy the low indices; the serving
            # failure (last example) is scored anyway — the H2 hazard the
            # consumer-side nulling neutralizes.
            em = 0.0 if i < n_false else 1.0
            score_lines.append(json.dumps(_score_row(i, exact_match=em)))
        scores = out_window / "qa_scores.jsonl"
        scores.write_text("\n".join(score_lines) + "\n", encoding="utf-8")
        quality = out_window / "quality.json"
        quality.write_text(json.dumps({"rows": N_EXAMPLES}), encoding="utf-8")
        written.extend([scores, quality])
    manifest = scoring_dir / "scoring_manifest.json"
    manifest.write_text(
        json.dumps({
            "scoring_run_id": scoring_id,
            "created_utc": "2026-08-17T10:00:00Z",
            "raw_run_ledger_entries_sha256": raw_sha,
            "mode": "fast",
        }),
        encoding="utf-8",
    )
    written.append(manifest)
    write_ledger(hash_artifacts(written, base_dir=scoring_dir), scoring_dir / "ledger.json")
    return scoring_dir


@pytest.fixture()
def sealed_run(tmp_path: Path) -> Path:
    run_dir = _build_sealed_run(tmp_path)
    _write_scoring_pass(run_dir)
    return run_dir


@pytest.fixture()
def predicate_run(sealed_run: Path) -> Path:
    rc = bpt.main([
        str(sealed_run), "--scoring-run-id", SCORING_ID,
        "--max-null-fraction", "0.5",
    ])
    assert rc == 0
    assert (
        org.main([str(sealed_run)]) == 0
    ), "organize_results must accept the predicate tree"
    return sealed_run


# ---------------------------------------------------------------------------
# Producer CLI (build_predicate_table.py)
# ---------------------------------------------------------------------------


def test_build_produces_sealed_mirrored_table(predicate_run: Path) -> None:
    pred_dir = predicate_run / "predicate" / SCORING_ID
    manifest = json.loads((pred_dir / "predicate_manifest.json").read_text())
    raw_sha = json.loads((predicate_run / "ledger.json").read_text())["entries_sha256"]
    assert manifest["raw_run_ledger_entries_sha256"] == raw_sha
    assert manifest["scoring_run_id"] == SCORING_ID
    assert manifest["config"]["max_null_fraction"] == 0.5
    assert manifest["config"]["qasper_tau"] is None
    assert (pred_dir / "ledger.json").is_file()
    # mirrored per-window rows: 4 cells x 2 windows
    rows_files = sorted(pred_dir.glob("cells/*/window_*/predicate.jsonl"))
    assert len(rows_files) == 8
    assert manifest["counts"]["n_windows"] == 8
    assert manifest["counts"]["n_rows"] == 8 * N_EXAMPLES
    # every window carries exactly one not-ok nulled row (counted, §9.10)
    assert manifest["counts"]["n_not_ok_nulled"] == 8
    rows = [json.loads(l) for l in rows_files[0].read_text().splitlines()]
    assert {r["predicate"] for r in rows} <= {True, False, None}
    assert all(r["predicate_rule"] == "span_qa_em_abstention_aware" for r in rows)


def test_build_refuses_unsealed_raw_tree(tmp_path: Path) -> None:
    run_dir = _build_sealed_run(tmp_path)
    _write_scoring_pass(run_dir)
    (run_dir / "ledger.json").unlink()
    rc = bpt.main([
        str(run_dir), "--scoring-run-id", SCORING_ID,
        "--max-null-fraction", "0.5",
    ])
    assert rc == 1


def test_build_refuses_scoring_pass_of_a_different_seal(tmp_path: Path) -> None:
    run_dir = _build_sealed_run(tmp_path)
    scoring_dir = _write_scoring_pass(run_dir)
    manifest_path = scoring_dir / "scoring_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_run_ledger_entries_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # reseal the pass so its OWN ledger stays valid — the cross-match must
    # still refuse.
    (scoring_dir / "ledger.json").unlink()
    files = [p for p in scoring_dir.rglob("*") if p.is_file()]
    write_ledger(hash_artifacts(files, base_dir=scoring_dir), scoring_dir / "ledger.json")
    with pytest.raises(bpt.BuildPredicateError, match="DIFFERENT raw seal"):
        bpt.build_predicate_table(
            run_dir, SCORING_ID, PredicateConfig(max_null_fraction=0.5)
        )


def test_build_refuses_partial_scoring_coverage(tmp_path: Path) -> None:
    run_dir = _build_sealed_run(tmp_path)
    scoring_dir = _write_scoring_pass(run_dir)
    # drop one scored window, reseal -> reconciliation must name it
    victim = sorted(scoring_dir.glob("cells/*/window_*/qa_scores.jsonl"))[0]
    victim.unlink()
    (victim.parent / "quality.json").unlink()
    victim.parent.rmdir()
    (scoring_dir / "ledger.json").unlink()
    files = [p for p in scoring_dir.rglob("*") if p.is_file()]
    write_ledger(hash_artifacts(files, base_dir=scoring_dir), scoring_dir / "ledger.json")
    with pytest.raises(bpt.BuildPredicateError, match="never scored"):
        bpt.build_predicate_table(
            run_dir, SCORING_ID, PredicateConfig(max_null_fraction=0.5)
        )


def test_build_refuses_existing_table_without_force(predicate_run: Path) -> None:
    with pytest.raises(bpt.BuildPredicateError, match="--force"):
        bpt.build_predicate_table(
            predicate_run, SCORING_ID, PredicateConfig(max_null_fraction=0.5)
        )
    # --force rebuilds cleanly
    out = bpt.build_predicate_table(
        predicate_run, SCORING_ID,
        PredicateConfig(max_null_fraction=0.5), force=True,
    )
    assert (out / "predicate_manifest.json").is_file()


def test_build_max_null_fraction_flag_is_required(sealed_run: Path) -> None:
    with pytest.raises(SystemExit):
        bpt.main([str(sealed_run), "--scoring-run-id", SCORING_ID])


# ---------------------------------------------------------------------------
# organize_results / verify_results wiring
# ---------------------------------------------------------------------------


def test_organize_reports_predicate_table(predicate_run: Path) -> None:
    report = (predicate_run / "index" / "coverage_report.md").read_text()
    assert "Predicate tables" in report
    assert SCORING_ID in report


def test_organize_refuses_tampered_predicate_rows(predicate_run: Path) -> None:
    rows_file = sorted(
        (predicate_run / "predicate" / SCORING_ID).glob(
            "cells/*/window_*/predicate.jsonl"
        )
    )[0]
    row = json.loads(rows_file.read_text().splitlines()[0])
    row["predicate"] = 0.7  # tri-state violation
    rows_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(org.LayoutError) as exc:
        org.validate_predicate_trees(predicate_run)
    problems = "\n".join(exc.value.problems)
    assert "outside" in problems  # tri-state guard
    assert "ledger mismatch" in problems  # the seal caught the edit too


def test_contamination_sweep_catches_predicate_inside_cells(
    predicate_run: Path,
) -> None:
    window = sorted((predicate_run / "cells").glob("*/window_*"))[0]
    (window / "predicate.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(org.LayoutError, match="sealed raw tree"):
        org.validate_scoring_tree(predicate_run)


def test_verify_results_gates_on_predicate_tree(predicate_run: Path, tmp_path: Path) -> None:
    import verify_results as vr

    report = vr.verify_run(predicate_run)
    assert report["ok"] is True
    assert any(SCORING_ID in line for line in report["predicate_tables"])
    # tamper -> FAIL finding under the predicate check
    manifest_path = (
        predicate_run / "predicate" / SCORING_ID / "predicate_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_run_ledger_entries_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = vr.verify_run(predicate_run)
    assert report["ok"] is False
    assert any(f["check"] == "predicate" for f in report["findings"])


# ---------------------------------------------------------------------------
# Consumer: run_campaign_analysis (per-query #4 predicate leg + #14 estimand)
# ---------------------------------------------------------------------------


def test_end_to_end_predicate_reaches_mcnemar(predicate_run: Path) -> None:
    # F1 CLOSED: evidence rows + scoring sidecar -> join -> predicate ->
    # the registered #4 predicate leg computes McNemar on real artifacts.
    rc = rca.main([
        str(predicate_run), "--contrasts", "4",
        "--metrics", "ttft_ms", "predicate",
    ])
    assert rc == 0
    analysis_dir = sorted((predicate_run / "analysis").iterdir())[-1]
    stats = json.loads((analysis_dir / "stats.json").read_text())
    note = stats["loader_notes"]["predicate_table"]
    assert note["path"] == f"predicate/{SCORING_ID}"
    entry = next(
        e for e in stats["contrasts"]
        if e["contrast_id"] == 4 and e["metric"] == "predicate"
    )
    assert entry["test"] == "mcnemar_binary"
    row = entry["per_dataset"][0]
    assert row["dataset"] == DATASET
    # the not-ok example is predicate=None on BOTH sides -> dropped, counted
    assert row["n_dropped_nan"] == 1
    assert row["n_pairs"] == N_EXAMPLES - 1
    # B6 (cell) has MORE predicate-false examples than B3 -> every
    # discordant pair is cell-fails/reference-passes
    assert row["n_10"] == 0
    assert row["n_01"] == F1_FALSE_COUNT["B6"] - F1_FALSE_COUNT["B3"]
    assert row["n_discordant"] == F1_FALSE_COUNT["B6"] - F1_FALSE_COUNT["B3"]


def test_end_to_end_truth_tax_executes_registered_row_shape(
    predicate_run: Path,
) -> None:
    # F1+F2+F3 CLOSED for #14: in-regime windows -> evaluate_window (ms→s
    # seam) -> per-window G−Y -> cross-engine batch means vs the vllm anchor
    # -> the contrast-14 chain endpoint outcome keyed <dataset>|truth_tax.
    rc = rca.main([
        str(predicate_run), "--contrasts", "4", "14",
        "--metrics", "ttft_ms", "predicate",
    ])
    assert rc == 0
    analysis_dir = sorted((predicate_run / "analysis").iterdir())[-1]
    stats = json.loads((analysis_dir / "stats.json").read_text())
    section = stats["truth_tax"]
    assert section["estimand"] == "truth_tax"
    assert section["higher_is_better"] is False  # ESTIMAND registry untouched
    legs = section["legs"]
    assert len(legs) == 1
    leg = legs[0]
    assert leg["engine"] == "sglang" and leg["anchor_engine"] == "vllm"
    assert leg["n_windows_cell"] == 2 and leg["n_windows_anchor"] == 2
    # sglang pays MORE truth tax than the anchor (per the fixture counts;
    # denominators = issued requests, §6.1 completed-only goodput).
    expected_cell = (6 / N_EXAMPLES + 8 / N_EXAMPLES) / 2
    expected_anchor = (1 / N_EXAMPLES + 2 / N_EXAMPLES) / 2
    assert leg["mean_truth_tax_cell"] == pytest.approx(expected_cell)
    assert leg["mean_truth_tax_anchor"] == pytest.approx(expected_anchor)
    assert leg["mean_diff"] == pytest.approx(expected_cell - expected_anchor)
    assert 0.0 <= leg["p_value"] <= 1.0
    assert leg["executed_alternative"] == "two-sided"
    assert "one-sided" in section["registered_sidedness"]
    iu = section["per_dataset_intersection"][0]
    assert iu["dataset"] == DATASET
    # the endpoint outcome entered the §9.3 chain under contrast-14
    gate = stats["gatekeeping"]
    assert any(
        p["endpoint"] == "contrast-14"
        and p["dataset_metric"] == f"{DATASET}|truth_tax"
        for p in gate["primaries"]
    )
    # summary.md renders the section
    summary_md = (analysis_dir / "summary.md").read_text()
    assert "Truth-tax estimand (#14" in summary_md


def test_truth_tax_refuses_without_slo_floors(predicate_run: Path) -> None:
    manifest_path = predicate_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["slo_floors"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(rca.AnalysisError, match="slo_floors"):
        rca.run_analysis(
            predicate_run, contrast_ids=[14], metrics=["ttft_ms"],
            mode="design-input",
        )


def test_truth_tax_refuses_without_regime_artifact(tmp_path: Path) -> None:
    # Build a tree WITHOUT regime.json: the §9.2 population is undecidable.
    run_dir = _build_sealed_run(tmp_path)
    for regime in run_dir.glob("cells/*/window_*/regime.json"):
        regime.unlink()
    # reseal (the fixture edits pre-date the predicate build)
    (run_dir / "ledger.json").unlink()
    sealed = [p for p in (run_dir / "cells").rglob("*") if p.is_file()]
    sealed.append(run_dir / "manifest.json")
    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")
    _write_scoring_pass(run_dir)
    assert bpt.main([
        str(run_dir), "--scoring-run-id", SCORING_ID,
        "--max-null-fraction", "0.5",
    ]) == 0
    assert org.main([str(run_dir)]) == 0
    with pytest.raises(rca.AnalysisError, match="regime.json"):
        rca.run_analysis(
            run_dir, contrast_ids=[14], metrics=["ttft_ms"],
            mode="design-input",
        )


def test_out_of_regime_windows_are_excluded_and_counted(tmp_path: Path) -> None:
    run_dir = _build_sealed_run(tmp_path)
    # push ONE sglang window out of regime -> < 2 in-regime windows on the
    # cell side -> the leg becomes a labeled skip and #14 refuses loudly
    # (no computable leg), never a silent number.
    victim = sorted(run_dir.glob("cells/*sglang*/window_*/regime.json"))[0]
    doc = json.loads(victim.read_text())
    doc["label"] = "PAST_CLIFF"
    victim.write_text(json.dumps(doc), encoding="utf-8")
    (run_dir / "ledger.json").unlink()
    sealed = [p for p in (run_dir / "cells").rglob("*") if p.is_file()]
    sealed.append(run_dir / "manifest.json")
    write_ledger(hash_artifacts(sealed, base_dir=run_dir), run_dir / "ledger.json")
    _write_scoring_pass(run_dir)
    assert bpt.main([
        str(run_dir), "--scoring-run-id", SCORING_ID,
        "--max-null-fraction", "0.5",
    ]) == 0
    assert org.main([str(run_dir)]) == 0
    with pytest.raises(rca.AnalysisError, match="no computable cross-engine leg"):
        rca.run_analysis(
            run_dir, contrast_ids=[14], metrics=["ttft_ms"],
            mode="design-input",
        )


def test_multiple_predicate_tables_refuse_without_flag(predicate_run: Path) -> None:
    second = _write_scoring_pass(predicate_run, "s02-fast")
    assert second.is_dir()
    assert bpt.main([
        str(predicate_run), "--scoring-run-id", "s02-fast",
        "--max-null-fraction", "0.5",
    ]) == 0
    with pytest.raises(rca.AnalysisError, match="--predicate-run-id"):
        rca.run_analysis(
            predicate_run, contrast_ids=[4], metrics=["ttft_ms"],
            mode="design-input",
        )
    # naming one resolves the ambiguity
    result = rca.run_analysis(
        predicate_run, contrast_ids=[4], metrics=["ttft_ms", "predicate"],
        mode="design-input", predicate_run_id=SCORING_ID,
    )
    stats = json.loads(result.stats_path.read_text())
    assert stats["loader_notes"]["predicate_table"]["path"] == f"predicate/{SCORING_ID}"


def test_tampered_predicate_table_refuses_at_analysis(predicate_run: Path) -> None:
    rows_file = sorted(
        (predicate_run / "predicate" / SCORING_ID).glob(
            "cells/*/window_*/predicate.jsonl"
        )
    )[0]
    lines = rows_file.read_text().splitlines()
    row = json.loads(lines[0])
    row["predicate"] = True
    rows_file.write_text(
        "\n".join([json.dumps(row)] + lines[1:]) + "\n", encoding="utf-8"
    )
    with pytest.raises(rca.AnalysisError, match="seal"):
        rca.run_analysis(
            predicate_run, contrast_ids=[4], metrics=["ttft_ms", "predicate"],
            mode="design-input",
        )
