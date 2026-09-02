"""Tests for T2.3 — PD (summed-pool) regime certification, charter §6.5/§6.1.

WHAT: pins ``compute_pd_window_regime_inputs`` / ``PDWindowRegimeInputs``
(src/analysis/regime_inputs.py) and the ``role_budgets`` routing of
``campaign_layout.write_window_regime``: hand-computed byte-weighted pooling
(budgets 3:1, so the weighting is visibly NOT an average), scarcity summing,
MIN-of-coverages, the single-role differential pin against
``compute_window_regime_inputs`` (field for field, EXACT), the full
fail-closed refusal matrix, and all three write_window_regime paths
(single-instance byte-identical, multi WITHOUT budgets = T4.1 refusal
verbatim, multi WITH budgets = §6.5 PD lane).

WHY: §6.5 — under prefill/decode disaggregation the budget B is the TOTAL
bytes of sequence-state cache, pools SUMMED, split recorded. Per-role
occupancy FRACTIONS have different byte denominators, so a naive sum or mean
of fractions is meaningless; the honest pooled occupancy is
``sum_r(rho_r * B_r) / sum_r(B_r)``. Every absence/mismatch must refuse
loudly (E2b: absence is not zero) — a softened refusal here would let a PD
campaign certify fabricated pressure.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.goodput import UNPRESSURED  # noqa: E402
from src.analysis.regime_inputs import (  # noqa: E402
    PDWindowRegimeInputs,
    REGIME_UNKNOWN,
    RegimeInputError,
    WindowRegimeInputs,
    compute_pd_window_regime_inputs,
    compute_window_regime_inputs,
)
from src.orchestration import campaign_layout as cl  # noqa: E402

# ---------------------------------------------------------------------------
# Hand-computed fixtures over the window [0, 10)
# ---------------------------------------------------------------------------


def _series(ts: list, kv: list, pre: list) -> pd.DataFrame:
    return pd.DataFrame(
        {"ts_s": ts, "kv_cache_usage": kv, "preemptions_total": pre}
    )


def _prefill() -> pd.DataFrame:
    """The pinned single-series ZOH case (tests/test_regime_inputs.py):
    covered 8 (first sample at 2), integral 0.5*4 + 1.0*2 + 0.8*2 = 5.6 ->
    rho 0.7; counter 5 -> 9 => 4 events; n 3; coverage 0.8."""
    return _series([2.0, 6.0, 8.0], [0.5, 1.0, 0.8], [5, 5, 9])


def _decode() -> pd.DataFrame:
    """Covered 8, integral 0.1*4 + 0.3*4 = 1.6 -> rho 0.2; counter 1 -> 3
    => 2 events; n 2; coverage 0.8."""
    return _series([2.0, 6.0], [0.1, 0.3], [1, 3])


def _decode_full_coverage() -> pd.DataFrame:
    """First sample at 0 -> coverage 1.0; rho 0.2; 0 events; n 2."""
    return _series([0.0, 5.0], [0.2, 0.2], [0, 0])


_BUDGETS_3_TO_1 = {"prefill": 3, "decode": 1}

#: Byte-weighted pool with the 3:1 split: 0.7*(3/4) + 0.2*(1/4) = 0.575.
#: The naive per-role AVERAGE would be 0.45 and the naive SUM 0.9 — the 3:1
#: budgets make the weighting visibly neither.
_POOLED_RHO_3_TO_1 = 0.575


def _pd_inputs(**overrides: Any) -> PDWindowRegimeInputs:
    kwargs: dict[str, Any] = {
        "series_by_role": {"prefill": _prefill(), "decode": _decode()},
        "window_start_s": 0.0,
        "window_end_s": 10.0,
        "budgets_by_role": dict(_BUDGETS_3_TO_1),
    }
    kwargs.update(overrides)
    series_by_role = kwargs.pop("series_by_role")
    start = kwargs.pop("window_start_s")
    end = kwargs.pop("window_end_s")
    return compute_pd_window_regime_inputs(series_by_role, start, end, **kwargs)


# ---------------------------------------------------------------------------
# 1. Pooled math: byte-weighted rho, summed scarcity/n, MIN coverage
# ---------------------------------------------------------------------------


class TestPooledMath:
    def test_pinned_byte_weighted_pool_3_to_1(self) -> None:
        pooled = _pd_inputs()
        assert pooled.rho_kv_time_avg == pytest.approx(_POOLED_RHO_3_TO_1)
        # §6.5: NOT an average of fractions (0.45) and NOT a sum (0.9).
        assert pooled.rho_kv_time_avg != pytest.approx(0.45)
        assert pooled.rho_kv_time_avg != pytest.approx(0.9)
        assert pooled.scarcity_events == 4 + 2
        assert pooled.n_samples == 3 + 2
        assert pooled.coverage == pytest.approx(0.8)
        assert pooled.window_start_s == 0.0
        assert pooled.window_end_s == 10.0

    def test_budget_weights_flip_with_the_split(self) -> None:
        # Same gauges, split recorded the other way round: the pool moves —
        # 0.7*(1/4) + 0.2*(3/4) = 0.325. Occupancy follows the BYTES.
        pooled = _pd_inputs(budgets_by_role={"prefill": 1, "decode": 3})
        assert pooled.rho_kv_time_avg == pytest.approx(0.325)

    def test_scarcity_events_sum_and_stay_int(self) -> None:
        pooled = _pd_inputs()
        assert pooled.scarcity_events == 6
        assert isinstance(pooled.scarcity_events, int)

    def test_coverage_is_min_of_roles_conservative(self) -> None:
        # prefill coverage 0.8, decode coverage 1.0: the pool is only
        # certified where EVERY role is — MIN (0.8), never the mean (0.9)
        # and never the best role (1.0).
        pooled = _pd_inputs(series_by_role={
            "prefill": _prefill(), "decode": _decode_full_coverage()
        })
        assert pooled.coverage == pytest.approx(0.8)
        assert pooled.coverage != pytest.approx(0.9)
        assert pooled.coverage != pytest.approx(1.0)

    def test_per_role_breakdown_is_the_single_series_certification(self) -> None:
        pooled = _pd_inputs()
        assert set(pooled.per_role) == {"prefill", "decode"}
        assert pooled.per_role["prefill"] == compute_window_regime_inputs(
            _prefill(), 0.0, 10.0
        )
        assert pooled.per_role["decode"] == compute_window_regime_inputs(
            _decode(), 0.0, 10.0
        )
        # §6.5: the split is recorded, not just consumed.
        assert pooled.budgets_by_role == _BUDGETS_3_TO_1

    def test_flat_dict_keys_match_the_single_instance_schema(self) -> None:
        # The §6.1 referee and the CSV joins consume PD windows under the
        # SAME field names as single-instance ones (T2.3 contract).
        pooled_flat = _pd_inputs().to_flat_dict()
        single_flat = compute_window_regime_inputs(_prefill(), 0.0, 10.0).to_flat_dict()
        assert set(pooled_flat) == set(single_flat)
        assert set(pooled_flat) == {
            f.name for f in WindowRegimeInputs.__dataclass_fields__.values()
        }


# ---------------------------------------------------------------------------
# 2. Differential pin: one role through the PD function == the single-series
#    function, field for field (EXACT — weight B_r/B is exactly 1.0)
# ---------------------------------------------------------------------------


class TestSingleRoleDifferential:
    def test_single_role_equals_single_series_field_for_field(self) -> None:
        single = compute_window_regime_inputs(_prefill(), 0.0, 10.0)
        pooled = compute_pd_window_regime_inputs(
            {"prefill": _prefill()}, 0.0, 10.0, budgets_by_role={"prefill": 7}
        )
        # EXACT equality, not approx: with one role the byte weight is
        # B_r/B == 1.0 and rho * 1.0 == rho bit-for-bit.
        assert pooled.rho_kv_time_avg == single.rho_kv_time_avg
        assert pooled.scarcity_events == single.scarcity_events
        assert pooled.n_samples == single.n_samples
        assert pooled.coverage == single.coverage
        assert pooled.window_start_s == single.window_start_s
        assert pooled.window_end_s == single.window_end_s
        assert pooled.to_flat_dict() == single.to_flat_dict()

    def test_single_role_budget_magnitude_is_irrelevant(self) -> None:
        # One role owns the whole pool whatever B_r is: same result for any
        # positive budget (the weight is identically 1.0).
        a = compute_pd_window_regime_inputs(
            {"prefill": _prefill()}, 0.0, 10.0, budgets_by_role={"prefill": 1}
        )
        b = compute_pd_window_regime_inputs(
            {"prefill": _prefill()}, 0.0, 10.0,
            budgets_by_role={"prefill": 10**12},
        )
        assert a.to_flat_dict() == b.to_flat_dict()


# ---------------------------------------------------------------------------
# 3. Refusal matrix (fail-closed, every path loud)
# ---------------------------------------------------------------------------


class TestRefusalMatrix:
    def test_empty_series_mapping_refuses(self) -> None:
        with pytest.raises(RegimeInputError, match="empty"):
            compute_pd_window_regime_inputs({}, 0.0, 10.0, budgets_by_role={})

    def test_role_without_budget_refuses_naming_it(self) -> None:
        with pytest.raises(RegimeInputError, match="decode"):
            _pd_inputs(budgets_by_role={"prefill": 3})

    def test_budget_without_series_refuses_naming_it(self) -> None:
        # Exact key match in BOTH directions: a budget with no telemetry
        # certifies nothing.
        with pytest.raises(RegimeInputError, match="ghost"):
            _pd_inputs(budgets_by_role={**_BUDGETS_3_TO_1, "ghost": 5})

    @pytest.mark.parametrize(
        "bad",
        [0, -3, 2.5, 3.0, True, "3", None, math.nan],
        ids=["zero", "negative", "float", "float-integral", "bool", "str", "none", "nan"],
    )
    def test_bad_budget_refuses(self, bad: object) -> None:
        with pytest.raises(RegimeInputError, match="budgets_by_role"):
            _pd_inputs(budgets_by_role={"prefill": 3, "decode": bad})

    def test_per_series_refusal_propagates_naming_the_role(self) -> None:
        # decode's gauge has a hole: the existing per-series refusal must
        # PROPAGATE (same type, message intact) with the role named — never
        # softened into a partial pool.
        broken = _series([2.0, 6.0], [0.1, None], [1, 3])
        with pytest.raises(
            RegimeInputError, match=r"role 'decode'.*absence is not zero"
        ):
            _pd_inputs(series_by_role={"prefill": _prefill(), "decode": broken})

    def test_per_series_too_few_samples_propagates(self) -> None:
        with pytest.raises(RegimeInputError, match=r"role 'decode'.*need >= 2"):
            _pd_inputs(
                series_by_role={
                    "prefill": _prefill(),
                    "decode": _series([2.0], [0.1], [1]),
                }
            )

    def test_bad_window_bounds_refuse(self) -> None:
        with pytest.raises(RegimeInputError, match="window_end_s"):
            _pd_inputs(window_start_s=10.0, window_end_s=0.0)

    def test_overlapping_role_spellings_refuse(self) -> None:
        # "prefill" and "Prefill" are one role under two spellings: budgeting
        # or certifying one pool twice is refused, never silently merged.
        series = {"prefill": _prefill(), "Prefill": _decode()}
        budgets = {"prefill": 3, "Prefill": 1}
        with pytest.raises(RegimeInputError, match="duplicate/overlapping"):
            compute_pd_window_regime_inputs(
                series, 0.0, 10.0, budgets_by_role=budgets
            )

    def test_whitespace_alias_role_keys_refuse(self) -> None:
        series = {"prefill": _prefill(), " prefill": _decode()}
        budgets = {"prefill": 3, " prefill": 1}
        with pytest.raises(RegimeInputError, match="duplicate/overlapping"):
            compute_pd_window_regime_inputs(
                series, 0.0, 10.0, budgets_by_role=budgets
            )

    @pytest.mark.parametrize("bad_key", [1, "", "   ", None], ids=["int", "empty", "blank", "none"])
    def test_non_string_or_empty_role_key_refuses(self, bad_key: object) -> None:
        with pytest.raises(RegimeInputError, match="non-empty string"):
            compute_pd_window_regime_inputs(
                {bad_key: _prefill()}, 0.0, 10.0, budgets_by_role={bad_key: 1}
            )

    def test_non_dataframe_series_refuses(self) -> None:
        with pytest.raises(RegimeInputError, match="DataFrame"):
            _pd_inputs(
                series_by_role={"prefill": _prefill(), "decode": [1, 2, 3]}
            )

    def test_non_mapping_budgets_refuse(self) -> None:
        with pytest.raises(RegimeInputError, match="mapping"):
            compute_pd_window_regime_inputs(
                {"prefill": _prefill()}, 0.0, 10.0,
                budgets_by_role=[("prefill", 3)],  # type: ignore[arg-type]
            )

    def test_threshold_kwargs_reach_the_per_series_machinery(self) -> None:
        # min_coverage forwards: decode coverage 0.8 < 0.9 must refuse.
        with pytest.raises(RegimeInputError, match=r"role 'decode'.*coverage"):
            _pd_inputs(
                series_by_role={
                    "prefill": _series([0.0, 6.0], [0.5, 1.0], [5, 9]),
                    "decode": _decode(),
                },
                min_coverage=0.9,
            )


# ---------------------------------------------------------------------------
# 4. write_window_regime routing (all three paths)
# ---------------------------------------------------------------------------

_PD_RECORDS = [
    # prefill = the pinned ZOH case; decode = the 0.2-rho case; interleaved
    # by timestamp exactly as a merged PD stream arrives (T4.1 save order).
    {"ts_s": 2.0, "kv_cache_usage": 0.5, "preemptions_total": 5, "instance": "prefill"},
    {"ts_s": 2.0, "kv_cache_usage": 0.1, "preemptions_total": 1, "instance": "decode"},
    {"ts_s": 6.0, "kv_cache_usage": 1.0, "preemptions_total": 5, "instance": "prefill"},
    {"ts_s": 6.0, "kv_cache_usage": 0.3, "preemptions_total": 3, "instance": "decode"},
    {"ts_s": 8.0, "kv_cache_usage": 0.8, "preemptions_total": 9, "instance": "prefill"},
]

_LEGACY_RECORDS = [
    {"ts_s": 2.0, "kv_cache_usage": 0.5, "preemptions_total": 5},
    {"ts_s": 6.0, "kv_cache_usage": 1.0, "preemptions_total": 5},
    {"ts_s": 8.0, "kv_cache_usage": 0.8, "preemptions_total": 9},
]


def _write_series(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return path


def _run_bridge(
    tmp_path: Path,
    records: list[dict[str, Any]],
    name: str,
    **kwargs: Any,
) -> dict[str, Any]:
    series = _write_series(tmp_path / f"{name}.jsonl", records)
    window = tmp_path / f"window_{name}"
    window.mkdir()
    path = cl.write_window_regime(
        window, t_start=0.0, t_end=10.0, telemetry_path=series, **kwargs
    )
    return json.loads(path.read_text(encoding="utf-8"))


class TestWriteWindowRegimeRouting:
    def test_single_instance_path_unchanged_without_budgets(self, tmp_path: Path) -> None:
        # Path 1: no budgets, untagged series — the pre-T2.3 document,
        # byte-identical: same key set (no "pd"), same pinned numbers.
        doc = _run_bridge(tmp_path, _LEGACY_RECORDS, "legacy")
        assert set(doc) == {
            "schema_version", "t_start", "t_end", "telemetry_source",
            "attainment", "telemetry_ok", "inputs", "label", "refusal_reason",
        }
        assert "pd" not in doc
        assert doc["telemetry_ok"] is True
        assert doc["inputs"]["rho_kv_time_avg"] == pytest.approx(0.7)
        assert doc["inputs"]["scarcity_events"] == 4

    def test_multi_without_budgets_keeps_t41_refusal_verbatim(self, tmp_path: Path) -> None:
        # Path 2: the T4.1 load-bearing refusal, message VERBATIM (same match
        # as tests/test_multi_instance_telemetry.py pins).
        with pytest.raises(
            cl.CampaignLayoutError,
            match="pooled PD regime math requires per-role budgets",
        ) as exc:
            _run_bridge(tmp_path, _PD_RECORDS, "pd-nobudget")
        assert "compute_pd_window_regime_inputs" in str(exc.value)
        assert "T2.3" in str(exc.value)

    def test_multi_with_budgets_certifies_the_summed_pool(self, tmp_path: Path) -> None:
        # Path 3: budgets route to the §6.5 PD lane — pooled inputs under the
        # single-instance field names + the recorded split and breakdown.
        doc = _run_bridge(
            tmp_path, _PD_RECORDS, "pd", role_budgets=dict(_BUDGETS_3_TO_1)
        )
        assert doc["telemetry_ok"] is True
        assert doc["refusal_reason"] is None
        assert doc["inputs"]["rho_kv_time_avg"] == pytest.approx(_POOLED_RHO_3_TO_1)
        assert doc["inputs"]["scarcity_events"] == 6
        assert doc["inputs"]["n_samples"] == 5
        assert doc["inputs"]["coverage"] == pytest.approx(0.8)
        # No attainment yet -> §6.1 labeling deferred, never fabricated.
        assert doc["label"] is None
        assert doc["pd"]["budgets_by_role"] == {"prefill": 3, "decode": 1}
        per_role = doc["pd"]["per_role"]
        assert per_role["prefill"]["rho_kv_time_avg"] == pytest.approx(0.7)
        assert per_role["prefill"]["scarcity_events"] == 4
        assert per_role["decode"]["rho_kv_time_avg"] == pytest.approx(0.2)
        assert per_role["decode"]["scarcity_events"] == 2

    def test_pd_lane_labels_with_attainment(self, tmp_path: Path) -> None:
        # Pooled rho 0.575 < 0.9 with attainment 0.95 -> UNPRESSURED, from
        # the ONE §6.1 threshold source (goodput.classify_regime).
        doc = _run_bridge(
            tmp_path, _PD_RECORDS, "pd-att",
            role_budgets=dict(_BUDGETS_3_TO_1), attainment=0.95,
        )
        assert doc["label"] == UNPRESSURED
        assert doc["attainment"] == 0.95

    def test_pd_telemetry_refusal_is_recorded_not_raised(self, tmp_path: Path) -> None:
        # A genuine telemetry hole inside one role: recorded as
        # UNKNOWN_TELEMETRY (absence stays absence), split still recorded,
        # per-role breakdown None — a refused pool carries no partial numbers.
        records = [
            dict(r) for r in _PD_RECORDS
        ]
        records[1] = {**records[1], "kv_cache_usage": None}
        doc = _run_bridge(
            tmp_path, records, "pd-hole", role_budgets=dict(_BUDGETS_3_TO_1)
        )
        assert doc["telemetry_ok"] is False
        assert doc["label"] == REGIME_UNKNOWN
        assert doc["inputs"] is None
        assert "role 'decode'" in doc["refusal_reason"]
        assert "absence is not zero" in doc["refusal_reason"]
        assert doc["pd"]["budgets_by_role"] == {"prefill": 3, "decode": 1}
        assert doc["pd"]["per_role"] is None

    def test_budgets_on_untagged_series_raise_caller_bug(self, tmp_path: Path) -> None:
        series = _write_series(tmp_path / "legacy.jsonl", _LEGACY_RECORDS)
        window = tmp_path / "window_untagged"
        window.mkdir()
        with pytest.raises(cl.CampaignLayoutError, match="no instance role tags"):
            cl.write_window_regime(
                window, t_start=0.0, t_end=10.0, telemetry_path=series,
                role_budgets=dict(_BUDGETS_3_TO_1),
            )
        # Caller-bug lane: nothing written, no quiet regime.json to limp on.
        assert not (window / "regime.json").exists()

    def test_budget_role_mismatch_raises_caller_bug(self, tmp_path: Path) -> None:
        series = _write_series(tmp_path / "pd.jsonl", _PD_RECORDS)
        window = tmp_path / "window_mismatch"
        window.mkdir()
        with pytest.raises(cl.CampaignLayoutError, match="decode"):
            cl.write_window_regime(
                window, t_start=0.0, t_end=10.0, telemetry_path=series,
                role_budgets={"prefill": 3},
            )
        with pytest.raises(cl.CampaignLayoutError, match="ghost"):
            cl.write_window_regime(
                window, t_start=0.0, t_end=10.0, telemetry_path=series,
                role_budgets={**_BUDGETS_3_TO_1, "ghost": 1},
            )
        assert not (window / "regime.json").exists()

    def test_untagged_records_mixed_with_tags_raise_under_budgets(self, tmp_path: Path) -> None:
        records = _PD_RECORDS[:4] + [_LEGACY_RECORDS[2]]  # last record untagged
        series = _write_series(tmp_path / "mixed.jsonl", records)
        window = tmp_path / "window_mixed"
        window.mkdir()
        with pytest.raises(cl.CampaignLayoutError, match="untagged"):
            cl.write_window_regime(
                window, t_start=0.0, t_end=10.0, telemetry_path=series,
                role_budgets=dict(_BUDGETS_3_TO_1),
            )

    @pytest.mark.parametrize("bad", [0, -1, 3.0, True, "3"], ids=["zero", "neg", "float", "bool", "str"])
    def test_bad_budget_value_raises_caller_bug(self, tmp_path: Path, bad: object) -> None:
        series = _write_series(tmp_path / "pd.jsonl", _PD_RECORDS)
        window = tmp_path / "window_badbudget"
        window.mkdir()
        with pytest.raises(cl.CampaignLayoutError, match="positive int"):
            cl.write_window_regime(
                window, t_start=0.0, t_end=10.0, telemetry_path=series,
                role_budgets={"prefill": bad, "decode": 1},
            )

    def test_empty_role_budgets_mapping_raises(self, tmp_path: Path) -> None:
        series = _write_series(tmp_path / "pd.jsonl", _PD_RECORDS)
        window = tmp_path / "window_emptybudgets"
        window.mkdir()
        # {} is not None: the caller ASKED for the PD lane with no recorded
        # split — refuse loudly, never fall back to the single path.
        with pytest.raises(cl.CampaignLayoutError, match="empty"):
            cl.write_window_regime(
                window, t_start=0.0, t_end=10.0, telemetry_path=series,
                role_budgets={},
            )

    def test_single_tagged_role_with_budget_routes_through_pd(self, tmp_path: Path) -> None:
        # One tagged role + its budget: the PD lane accepts (weight 1.0) and
        # matches the single-instance numbers — the routing differential.
        records = [{**r, "instance": "single"} for r in _LEGACY_RECORDS]
        doc = _run_bridge(
            tmp_path, records, "one-role", role_budgets={"single": 42}
        )
        assert doc["telemetry_ok"] is True
        assert doc["inputs"]["rho_kv_time_avg"] == pytest.approx(0.7)
        assert doc["inputs"]["scarcity_events"] == 4
        assert doc["pd"]["budgets_by_role"] == {"single": 42}
