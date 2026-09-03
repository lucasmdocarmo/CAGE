"""Tests for src/analysis/degradation.py (W4.10, PUBLICATION.md §8.11) and the
driver threading in scripts/4_analysis/run_campaign_analysis.py.

Covered:

1. Each §8.11 classifier's positive AND negative case on synthetic stored
   rows (the run_experiment.py producer schema), plus None-honesty: every
   unjudgeable row yields ``value=None`` with a NAMED reason — never a
   guessed label.
2. Reuse pins: abstention detection IS quality.py's detector (identity, not
   a duplicate); the grounding fallback applies the producer's τ.
3. The OWNER-GATED S2 policy-event join stub refuses with the one named
   reason (shared verbatim with the §8.12 conditioned-curve stub).
4. Driver threading (W4.10 wiring): ``load_per_query`` classifies each
   merged trial row and threads judged labels as ``deg_*`` mean columns;
   unjudgeable labels stay ABSENT with counted reasons in
   ``frame.attrs['degradation_reasons']``; a producer row carrying a
   reserved ``deg_*`` field refuses; ``_metric_direction`` knows the label
   columns (lower is better); ``_absent_instrument_detail`` surfaces what
   remains genuinely missing.
5. End-to-end: the #13 fingerprint executor consumes a degradation label as
   its instrument (``--equivalence-metric deg_fabrication``) on a synthetic
   organized run — the §9.3 decomposition legs run on label rates.

All fixtures are tiny, deterministic, and clearly test-only. No GPU, no
network, no models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import organize_results as org  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
import test_campaign_analysis as tca  # noqa: E402
from src.analysis import degradation as deg  # noqa: E402
from src.analysis.cellspec import CellSpec  # noqa: E402
from src.evaluation import quality  # noqa: E402


def _row(**fields: Any) -> dict[str, Any]:
    """A minimal judgeable stored row; keyword overrides mirror the producer."""
    base: dict[str, Any] = {
        "example_id": "e001",
        "finish_reason": "stop",
        "error": None,
        "generated_answer": "Paris is the capital of France.",
        "reference_answer": "Paris",
        "predicted_no_answer": 0.0,
        "is_answerable": 1.0,
        "grounded": True,
        "grounding_score": 0.9,
        "gold_position_in_prompt": 0,
    }
    base.update(fields)
    return base


# ---------------------------------------------------------------------------
# 1. Classifiers — positive / negative / None-honesty
# ---------------------------------------------------------------------------


class TestTruncatedGeneration:
    def test_length_is_truncated(self) -> None:
        result = deg.classify_truncated_generation(_row(finish_reason="length"))
        assert result.value is True and result.reason is None

    def test_stop_is_not_truncated(self) -> None:
        result = deg.classify_truncated_generation(_row(finish_reason="stop"))
        assert result.value is False and result.reason is None

    def test_missing_finish_reason_is_named_none(self) -> None:
        row = _row()
        del row["finish_reason"]
        result = deg.classify_truncated_generation(row)
        assert result.value is None
        assert "finish_reason absent" in result.reason

    def test_error_row_is_named_none(self) -> None:
        result = deg.classify_truncated_generation(
            _row(error="HTTP 500", finish_reason="error")
        )
        assert result.value is None
        assert "error row" in result.reason

    def test_unknown_finish_reason_is_named_none(self) -> None:
        result = deg.classify_truncated_generation(_row(finish_reason="abort"))
        assert result.value is None
        assert "'abort'" in result.reason

    def test_non_string_finish_reason_is_named_none(self) -> None:
        result = deg.classify_truncated_generation(_row(finish_reason=5))
        assert result.value is None
        assert "not a string" in result.reason


class TestRepetition:
    def test_pinned_parameters(self) -> None:
        # The detector is a registered instrument: its parameters are pinned
        # module constants, not tunables.
        assert deg.REPETITION_NGRAM_N == 3
        assert deg.REPETITION_MIN_TOKENS == 20
        assert deg.REPETITION_MAX_DISTINCT_RATIO == 0.5

    def test_looped_generation_is_repetition(self) -> None:
        looped = "the cat sat on the mat " * 10  # 60 tokens, 6 distinct 3-grams
        result = deg.classify_repetition(_row(generated_answer=looped))
        assert result.value is True and result.reason is None

    def test_distinct_long_generation_is_not_repetition(self) -> None:
        distinct = " ".join(f"token{i}" for i in range(40))
        result = deg.classify_repetition(_row(generated_answer=distinct))
        assert result.value is False and result.reason is None

    def test_short_generation_is_negative_not_none(self) -> None:
        # A short answer IS classifiable — too short to certify a loop.
        result = deg.classify_repetition(_row(generated_answer="Paris."))
        assert result.value is False and result.reason is None

    def test_missing_text_is_named_none(self) -> None:
        result = deg.classify_repetition(_row(generated_answer=None))
        assert result.value is None
        assert "generated_answer absent" in result.reason

    def test_error_row_is_named_none(self) -> None:
        result = deg.classify_repetition(_row(error="boom"))
        assert result.value is None
        assert "error row" in result.reason


class TestAbstentionShift:
    def test_abstained_on_answerable_is_shift(self) -> None:
        result = deg.classify_abstention_shift(
            _row(predicted_no_answer=1.0, is_answerable=1.0)
        )
        assert result.value is True

    def test_confident_answer_is_not_shift(self) -> None:
        result = deg.classify_abstention_shift(_row(predicted_no_answer=0.0))
        assert result.value is False

    def test_correct_abstention_on_unanswerable_is_not_shift(self) -> None:
        result = deg.classify_abstention_shift(
            _row(predicted_no_answer=1.0, is_answerable=0.0)
        )
        assert result.value is False

    def test_fallback_reuses_quality_detector(self) -> None:
        # No scored columns: the SAME quality.py detector runs on the
        # sanitized generation — identity pin, never a second regex.
        assert deg.is_no_answer_prediction is quality.is_no_answer_prediction
        assert deg.sanitize_answer is quality.sanitize_answer
        row = _row(generated_answer="I don't know.", reference_answer="Paris")
        del row["predicted_no_answer"]
        del row["is_answerable"]
        result = deg.classify_abstention_shift(row)
        assert result.value is True

    def test_all_answers_empty_means_unanswerable(self) -> None:
        row = _row(predicted_no_answer=1.0, all_answers=[])
        del row["is_answerable"]
        del row["reference_answer"]
        result = deg.classify_abstention_shift(row)
        assert result.value is False  # correct abstention, not a shift

    def test_no_answerability_signal_is_named_none(self) -> None:
        row = _row(predicted_no_answer=1.0)
        for key in ("is_answerable", "reference_answer"):
            del row[key]
        result = deg.classify_abstention_shift(row)
        assert result.value is None
        assert "no answerability signal" in result.reason

    def test_no_abstention_signal_is_named_none(self) -> None:
        row = _row(generated_answer=None)
        del row["predicted_no_answer"]
        result = deg.classify_abstention_shift(row)
        assert result.value is None
        assert "predicted_no_answer" in result.reason


class TestFabrication:
    def test_confident_ungrounded_is_fabrication(self) -> None:
        result = deg.classify_fabrication(
            _row(predicted_no_answer=0.0, grounded=False)
        )
        assert result.value is True

    def test_grounded_answer_is_not_fabrication(self) -> None:
        result = deg.classify_fabrication(
            _row(predicted_no_answer=0.0, grounded=True)
        )
        assert result.value is False

    def test_abstention_is_not_fabrication(self) -> None:
        result = deg.classify_fabrication(
            _row(predicted_no_answer=1.0, grounded=False)
        )
        assert result.value is False

    def test_grounding_score_fallback_at_producer_tau(self) -> None:
        low = _row(predicted_no_answer=0.0, grounded=None, grounding_score=0.3)
        high = _row(predicted_no_answer=0.0, grounded=None, grounding_score=0.9)
        assert deg.classify_fabrication(low).value is True
        assert deg.classify_fabrication(high).value is False
        assert deg.GROUNDED_TAU == 0.5  # the producer's grounded threshold

    def test_unscored_grounding_is_named_none(self) -> None:
        result = deg.classify_fabrication(
            _row(predicted_no_answer=0.0, grounded=None, grounding_score=None)
        )
        assert result.value is None
        assert "grounding columns absent/unscored" in result.reason


class TestWrongContext:
    def test_grounded_with_gold_absent_is_wrong_context(self) -> None:
        result = deg.classify_wrong_context(
            _row(grounded=True, gold_position_in_prompt=-1)
        )
        assert result.value is True

    def test_grounded_with_gold_present_is_not(self) -> None:
        result = deg.classify_wrong_context(
            _row(grounded=True, gold_position_in_prompt=2)
        )
        assert result.value is False

    def test_ungrounded_answer_is_not_wrong_context(self) -> None:
        result = deg.classify_wrong_context(
            _row(grounded=False, gold_position_in_prompt=-1)
        )
        assert result.value is False

    def test_missing_containment_column_is_named_none(self) -> None:
        result = deg.classify_wrong_context(
            _row(grounded=True, gold_position_in_prompt=None)
        )
        assert result.value is None
        assert "gold_position_in_prompt absent" in result.reason

    def test_unscored_grounding_is_named_none(self) -> None:
        result = deg.classify_wrong_context(
            _row(grounded=None, grounding_score=None)
        )
        assert result.value is None
        assert "grounding columns absent/unscored" in result.reason

    def test_malformed_position_is_named_none(self) -> None:
        result = deg.classify_wrong_context(
            _row(grounded=True, gold_position_in_prompt="x")
        )
        assert result.value is None
        assert "not an integer" in result.reason


class TestClassifyRequest:
    def test_all_labels_in_charter_order(self) -> None:
        results = deg.classify_request(_row())
        assert tuple(results) == deg.DEGRADATION_LABELS

    def test_deterministic(self) -> None:
        row = _row(finish_reason="length", grounded=False)
        assert deg.classify_request(row) == deg.classify_request(row)

    def test_non_mapping_refuses(self) -> None:
        with pytest.raises(deg.DegradationError, match="mapping"):
            deg.classify_request(["not", "a", "row"])  # type: ignore[arg-type]

    def test_label_result_invariant(self) -> None:
        with pytest.raises(deg.DegradationError, match="exactly one"):
            deg.LabelResult("fabrication", None, None)
        with pytest.raises(deg.DegradationError, match="exactly one"):
            deg.LabelResult("fabrication", True, "and also a reason")

    def test_column_of_refuses_unknown_label(self) -> None:
        with pytest.raises(deg.DegradationError, match="unknown degradation"):
            deg.column_of("not_a_label")


# ---------------------------------------------------------------------------
# 3. The OWNER-GATED S2 stub
# ---------------------------------------------------------------------------


class TestS2Stub:
    def test_join_policy_events_refuses_named(self) -> None:
        with pytest.raises(deg.DegradationError) as excinfo:
            deg.join_policy_events()
        message = str(excinfo.value)
        assert "S2" in message
        assert "OWNER-GATED" in message
        assert message == deg.S2_POLICY_EVENT_JOIN_UNAVAILABLE

    def test_conditioned_curve_stub_shares_the_reason(self) -> None:
        from src.analysis import conditioned_curves as cc

        with pytest.raises(deg.DegradationError) as excinfo:
            cc.policy_event_curve()
        assert str(excinfo.value) == deg.S2_POLICY_EVENT_JOIN_UNAVAILABLE


# ---------------------------------------------------------------------------
# 4. Driver threading (load_per_query + direction + refusal enrichment)
# ---------------------------------------------------------------------------


def _mini_window(tmp_path: Path, requests: list[dict], evidence: list[dict]) -> tuple[Path, pd.DataFrame]:
    """One window on disk + the minimal index frame load_per_query needs."""
    window_dir = tmp_path / "cells" / "cellA" / "window_squad_v2-01"
    window_dir.mkdir(parents=True)
    (window_dir / "requests.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in requests), encoding="utf-8"
    )
    (window_dir / "qa_evidence.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in evidence), encoding="utf-8"
    )
    index = pd.DataFrame(
        [
            {
                "row_key": "cellA",
                "dataset": "squad_v2",
                "window_key": "squad_v2-01",
                "window_dir": "cells/cellA/window_squad_v2-01",
            }
        ]
    )
    return tmp_path, index


def test_loader_threads_label_columns_with_none_honesty(tmp_path: Path) -> None:
    requests = [
        # e1: fabricates (confident + ungrounded), finished by stop.
        {"example_id": "e001", "finish_reason": "stop", "ttft_ms": 100.0,
         "predicted_no_answer": False, "grounded": False},
        # e2: clean grounded answer; finish_reason ABSENT -> the
        # truncated_generation label must stay absent with a counted reason.
        {"example_id": "e002", "ttft_ms": 90.0,
         "predicted_no_answer": False, "grounded": True},
    ]
    evidence = [
        {"example_id": "e001", "generated_answer": "made-up claim",
         "gold_position_in_prompt": 0},
        {"example_id": "e002", "generated_answer": "Paris.",
         "gold_position_in_prompt": 1},
    ]
    run_dir, index = _mini_window(tmp_path, requests, evidence)
    frame = rca.load_per_query(run_dir, index, {"cellA"})
    assert "deg_fabrication" in frame.columns
    by_id = frame.set_index("example_id")
    assert by_id.loc["e001", "deg_fabrication"] == 1.0
    assert by_id.loc["e002", "deg_fabrication"] == 0.0
    # e1 judged (stop -> False); e2 unjudgeable -> NaN, reason counted.
    assert by_id.loc["e001", "deg_truncated_generation"] == 0.0
    assert pd.isna(by_id.loc["e002", "deg_truncated_generation"])
    reasons = frame.attrs["degradation_reasons"]
    truncation_reasons = reasons["deg_truncated_generation"]
    assert sum(truncation_reasons.values()) == 1
    assert any("finish_reason absent" in r for r in truncation_reasons)


def test_loader_refuses_producer_shadowing_label_namespace(tmp_path: Path) -> None:
    requests = [{"example_id": "e001", "deg_fabrication": 1.0}]
    evidence = [{"example_id": "e001"}]
    run_dir, index = _mini_window(tmp_path, requests, evidence)
    with pytest.raises(rca.AnalysisError, match="reserved degradation-label"):
        rca.load_per_query(run_dir, index, {"cellA"})


def test_metric_direction_knows_label_columns() -> None:
    for column in deg.LABEL_COLUMNS:
        assert rca._metric_direction(column) is False  # failure rate: lower better
    # And they must NOT leak into the registered caller roster (G7 pin).
    from src.analysis.stats.families import REGISTERED_METRICS

    assert not set(deg.LABEL_COLUMNS) & REGISTERED_METRICS


def test_absent_instrument_detail_names_what_remains_missing() -> None:
    frame = pd.DataFrame({"example_id": ["e1"]})
    frame.attrs["degradation_reasons"] = {
        "deg_truncated_generation": {"finish_reason absent from the stored row — x": 3}
    }
    detail = rca._absent_instrument_detail(frame, "deg_truncated_generation")
    assert "finish_reason absent" in detail
    assert "(x3)" in detail
    assert "threaded by the loader" in detail
    # Non-label metrics get no enrichment.
    assert rca._absent_instrument_detail(frame, "grounding_score") == ""
    # A label column absent WITHOUT recorded reasons still names the state.
    assert "no judged trial" in rca._absent_instrument_detail(
        frame, "deg_fabrication"
    )


# ---------------------------------------------------------------------------
# 5. End-to-end: #13 fingerprint legs consume a degradation label instrument
# ---------------------------------------------------------------------------


def _degradation_fingerprint_specs() -> list[tuple[CellSpec, Any]]:
    """evict/compress-fp8 cells whose rows FABRICATE vs a clean policy=none
    reference — same §7.6.1 geometry as tca._fingerprint_specs, but the
    instrument is the threaded deg_fabrication label."""
    base = dict(
        arm="gold-fresh", retriever="none", topology="single", engine="vllm",
        model=tca.MODEL, family="F2", budget_r=0.5, rate_frac=0.9,
    )

    def fabricating(i: int) -> dict[str, Any]:
        return {"predicted_no_answer": False, "grounded": i % 8 != 0}

    def clean(i: int) -> dict[str, Any]:
        return {"predicted_no_answer": False, "grounded": True}

    return [
        (CellSpec(policy="evict", **base), fabricating),  # type: ignore[arg-type]
        (CellSpec(policy="compress-fp8", **base), fabricating),  # type: ignore[arg-type]
        (CellSpec(policy="none", **base), clean),  # type: ignore[arg-type]
    ]


def test_fingerprint_legs_run_on_degradation_label(tmp_path: Path) -> None:
    run_dir = tca._build_run_tree(
        tmp_path, special_specs=_degradation_fingerprint_specs()
    )
    org.organize_run(run_dir)
    rc = rca.main([str(run_dir), "--equivalence-metric", "deg_fabrication"])
    assert rc == 0
    _, stats = tca._load_stats(run_dir)
    fp_section = stats["fingerprint"]
    legs = fp_section["legs"]
    assert {leg["leg"] for leg in legs} == {"evict", "compress"}
    for leg in legs:
        assert leg["metric"] == "deg_fabrication"
        # Label = failure rate (lower better) -> the registered harm tail is
        # "greater" (the coping policy RAISES the failure rate).
        assert leg["executed_alternative"] == "greater"
        assert leg["p_value"] < 0.05
