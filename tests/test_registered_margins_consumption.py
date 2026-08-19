"""K-COV2 (task #140): EXECUTE the registered-margin consumption (G1d/G9).

The Topic-12 finding: ``resolve_registered_margin`` — the §9.5 consumer of the
registered-margins artifact (``REGISTERED_MARGINS_PATH``, resolved from
run_campaign_analysis.py itself) — never executed its success arm or its
malformed-artifact refusals in the suite, i.e. the exact path that fires at
the real one look. This file drives the consumer directly against artifact
files on disk:

- one VALID artifact: the registered margin is CONSUMED (returned + recorded,
  with and without a matching CLI cross-check);
- malformed variants each produce a TYPED refusal (AnalysisError), never a
  crash or a silent default: invalid JSON, non-object JSON, missing/unknown
  metric key (with a CLI margin to cross-check), wrong-type value
  (string/bool), out-of-range value (NaN/inf/<= 0);
- the honest pre-freeze skips stay skips: no artifact / unregistered metric
  WITHOUT a CLI margin return (None, record), a labeled skip — not a minted
  margin;
- ordering pin: the confirmatory driver resolves margins BEFORE the §9.11
  one-look lock is acquired, so a margin refusal never burns the look.

The wrong-type and NaN/<= 0 arms pin the task-#140 production fix: the old
``float(margins[metric])`` crashed (ValueError) on a string and silently
consumed NaN.

Pure local: no GPU, no network; the artifact path is monkeypatched to tmp.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_campaign_analysis as rca  # noqa: E402

METRIC = "grounding_score"


@pytest.fixture()
def margins_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the consumer at a tmp artifact bearing the real frozen name."""
    path = tmp_path / rca.REGISTERED_MARGINS_PATH.name
    monkeypatch.setattr(rca, "REGISTERED_MARGINS_PATH", path)
    return path


def _write(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# Success arm: the registered margin is CONSUMED
# ---------------------------------------------------------------------------


def test_valid_artifact_margin_consumed(margins_path: Path) -> None:
    _write(margins_path, json.dumps({METRIC: 0.05, "faithfulness_mean": 0.03}))
    margin, record = rca.resolve_registered_margin(None, METRIC)
    assert margin == 0.05
    assert record["margin_used"] == 0.05
    assert record["margin_source"] == margins_path.name
    assert record["artifact_present"] is True
    assert record["equivalence_metric"] == METRIC


def test_valid_artifact_matching_cli_margin_consumed(margins_path: Path) -> None:
    _write(margins_path, json.dumps({METRIC: 0.05}))
    margin, record = rca.resolve_registered_margin(0.05, METRIC)
    assert margin == 0.05
    assert record["cli_margin"] == 0.05
    assert record["margin_used"] == 0.05


def test_mismatching_cli_margin_refuses(margins_path: Path) -> None:
    _write(margins_path, json.dumps({METRIC: 0.05}))
    with pytest.raises(rca.AnalysisError, match="REGISTERED margin"):
        rca.resolve_registered_margin(0.1, METRIC)


# ---------------------------------------------------------------------------
# Malformed artifacts: typed refusals, never crashes or silent defaults
# ---------------------------------------------------------------------------


def test_invalid_json_refuses_typed(margins_path: Path) -> None:
    _write(margins_path, '{"grounding_score": 0.0')  # truncated
    with pytest.raises(rca.AnalysisError, match="not valid JSON"):
        rca.resolve_registered_margin(None, METRIC)


def test_non_object_artifact_refuses_typed(margins_path: Path) -> None:
    _write(margins_path, json.dumps([METRIC, 0.05]))
    with pytest.raises(rca.AnalysisError, match="JSON object"):
        rca.resolve_registered_margin(None, METRIC)


def test_missing_key_with_cli_margin_refuses_typed(margins_path: Path) -> None:
    _write(margins_path, json.dumps({"faithfulness_mean": 0.03}))
    with pytest.raises(rca.AnalysisError, match="no registered"):
        rca.resolve_registered_margin(0.05, METRIC)


@pytest.mark.parametrize("bad_value", ["0.05", "wide", True, None, [0.05], {"m": 0.05}])
def test_wrong_type_value_refuses_typed(margins_path: Path, bad_value) -> None:
    """A string/bool/null/list value is not a margin: AnalysisError, never the
    pre-#140 float() ValueError crash (or bool -> 1.0 silent coercion)."""
    _write(margins_path, json.dumps({METRIC: bad_value}))
    with pytest.raises(rca.AnalysisError, match="not a number"):
        rca.resolve_registered_margin(None, METRIC)


@pytest.mark.parametrize("bad_number", [float("nan"), float("inf"), 0.0, -0.05])
def test_out_of_range_value_refuses_typed(margins_path: Path, bad_number: float) -> None:
    """NaN/inf/<= 0 is not a usable TOST margin: refusal, never the pre-#140
    silent consumption (a NaN margin poisons every §9.5 leg downstream)."""
    payload = f'{{"{METRIC}": {"NaN" if math.isnan(bad_number) else ("Infinity" if math.isinf(bad_number) else bad_number)}}}'
    _write(margins_path, payload)
    with pytest.raises(rca.AnalysisError, match="finite and > 0"):
        rca.resolve_registered_margin(None, METRIC)


# ---------------------------------------------------------------------------
# Honest labeled skips (the registered pre-freeze semantics) stay skips
# ---------------------------------------------------------------------------


def test_unregistered_metric_without_cli_is_labeled_skip(margins_path: Path) -> None:
    _write(margins_path, json.dumps({"faithfulness_mean": 0.03}))
    margin, record = rca.resolve_registered_margin(None, METRIC)
    assert margin is None
    assert record["margin_used"] is None
    assert record["artifact_present"] is True  # the record says WHY: key absent


def test_no_artifact_without_cli_is_labeled_skip(margins_path: Path) -> None:
    assert not margins_path.exists()
    margin, record = rca.resolve_registered_margin(None, METRIC)
    assert margin is None
    assert record["artifact_present"] is False


def test_no_artifact_with_cli_margin_refuses_typed(margins_path: Path) -> None:
    assert not margins_path.exists()
    with pytest.raises(rca.AnalysisError, match="registration content"):
        rca.resolve_registered_margin(0.05, METRIC)


def test_cli_margin_without_metric_refuses_typed(margins_path: Path) -> None:
    _write(margins_path, json.dumps({METRIC: 0.05}))
    with pytest.raises(rca.AnalysisError, match="cross-checked"):
        rca.resolve_registered_margin(0.05, None)


# ---------------------------------------------------------------------------
# Wiring: margin resolution precedes the one-look lock
# ---------------------------------------------------------------------------


def test_margin_resolution_precedes_the_one_look_lock() -> None:
    """A malformed-artifact refusal must fire BEFORE §9.11 spends the lock —
    pin the confirmatory ordering in the driver source."""
    src = (_SCRIPTS_DIR / "run_campaign_analysis.py").read_text(encoding="utf-8")
    # CALL sites, not the defs: the margin resolution assignment vs the
    # indented lock-acquire statement.
    call = src.index("= resolve_registered_margin(")
    lock = src.index("        _acquire_confirmatory_lock(run_dir,")
    assert call < lock, (
        "resolve_registered_margin no longer runs before "
        "_acquire_confirmatory_lock — a margin refusal would burn the one look"
    )
