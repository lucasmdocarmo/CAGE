"""Tests for the CAGE -> cage-stats telemetry bridge (src/monitoring/vllm_telemetry.py).

Covers the graceful-skip path (cage-stats absent / server unreachable -> None, never
raises) and resolution of the in-process cage-stats bridge when the sibling repo is
present. There is NO mock/synthetic path: CAGE only ever records LIVE telemetry, so an
unavailable server yields None rather than fabricated numbers.
"""

import os

import pytest

from src.monitoring import vllm_telemetry as t

# CAGE/tests -> /Users/lucasmariano -> cage-stats
_SIBLING_CAGE_STATS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "cage-stats")
)
_HAVE_CAGE_STATS = os.path.isdir(os.path.join(_SIBLING_CAGE_STATS, "cage_stats"))


def test_graceful_skip_when_unavailable(monkeypatch):
    """No cage-stats, no CLI, no reachable /metrics -> everything degrades to None."""
    monkeypatch.setattr(t, "_try_import_api", lambda: None)
    monkeypatch.setattr(t.shutil, "which", lambda *_a, **_k: None)
    # Also neutralize the stdlib /metrics fallback so the test needs no live socket.
    monkeypatch.setattr(t, "scrape_spec_decode", lambda *_a, **_k: None)
    assert t.available() is False
    assert t.capture_snapshot("http://localhost:8000") is None
    assert t.dashboard_text("http://localhost:8000") is None
    assert t.capture("http://localhost:8000") == (None, None)


@pytest.mark.skipif(not _HAVE_CAGE_STATS, reason="cage-stats sibling repo not present")
def test_bridge_resolves_via_cage_stats_home(monkeypatch):
    """With cage-stats resolvable (CAGE_STATS_HOME), the in-process bridge is available.

    We deliberately do NOT assert a captured snapshot: without a live vLLM server there is
    no telemetry to read (and no mock path any more), so availability is the meaningful
    signal that the bridge wires up.
    """
    monkeypatch.setenv("CAGE_STATS_HOME", _SIBLING_CAGE_STATS)
    assert t.available() is True


# ---------------------------------------------------------------------------
# S2 block F row F-4: the engine-side prefix-hit share bounded to the window.
# S1 recorded the engine's prefix-cache counters only as lifetime finals, so a
# window's cache share was a lifetime quotient that included the warm-up pool
# and every earlier window of the process. The helper below differences the
# raw cumulative counters between the first and last sampler tick INSIDE
# [t_start, t_end), per instance role, and never raises at write time: every
# case it cannot compute is a null share with a named reason.
# ---------------------------------------------------------------------------

QUERIES = "prefix_cache_queries_total"
HITS = "prefix_cache_hits_total"


def _tick(ts: float, queries, hits, role: str = "single") -> dict:
    return {"ts_s": ts, "instance": role, QUERIES: queries, HITS: hits}


def test_window_prefix_hit_delta_uses_the_first_and_last_tick_inside_the_window() -> None:
    records = [
        _tick(0.0, 100, 50),   # before the window: excluded
        _tick(1.0, 200, 120),
        _tick(2.5, 300, 180),
        _tick(3.0, 400, 300),  # t_end is exclusive: excluded
    ]
    out = t.window_prefix_hit_delta(records, 1.0, 3.0)
    assert set(out) == {"single"}
    single = out["single"]
    assert single["share"] == pytest.approx(60 / 100)
    assert (single["delta_hits"], single["delta_queries"]) == (60, 100)
    assert (single["n_ticks_in_window"], single["n_ticks_carrying"]) == (2, 2)
    assert (single["first_ts_s"], single["last_ts_s"]) == (1.0, 2.5)
    assert single["counters"] == [QUERIES, HITS]
    assert single["reason"] is None
    # the pod-side names are the analysis-side names (dist_contrasts reads them)
    from src.analysis import dist_contrasts as dc

    assert (t.PREFIX_QUERIES_FIELD, t.PREFIX_HITS_FIELD) == (dc.QUERIES_FIELD, dc.HITS_FIELD) == (QUERIES, HITS)
    # endpoints are chosen by ts_s, not by list position
    shuffled = [records[2], records[1], records[0], records[3]]
    assert t.window_prefix_hit_delta(shuffled, 1.0, 3.0) == out
    # a non-carrying tick ahead of the first carrying one is counted but never an endpoint
    lead = [_tick(1.0, None, None), _tick(2.0, 5, 2), _tick(3.0, 9, 4)]
    led = t.window_prefix_hit_delta(lead, 0.0, 4.0)["single"]
    assert led["share"] == pytest.approx(0.5)
    assert (led["n_ticks_in_window"], led["n_ticks_carrying"]) == (3, 2)
    assert (led["first_ts_s"], led["last_ts_s"]) == (2.0, 3.0)


def test_window_prefix_hit_delta_names_every_reason_instead_of_raising() -> None:
    def reason(records, t0=0.0, t1=4.0, role="single"):
        out = t.window_prefix_hit_delta(records, t0, t1)
        assert out[role]["share"] is None
        return out[role]["reason"]

    assert reason([]) == "no telemetry series"
    assert "inside" in reason([_tick(10.0, 1, 1)])  # no tick inside the window
    # SGLang's dialect maps no prefix-cache counter family: None on every tick
    assert "dialect" in reason([_tick(1.0, None, None), _tick(2.0, None, None)])
    assert "dialect" in reason([{"ts_s": 1.0, "instance": "single"}])  # fields absent
    assert "single carrying tick" in reason([_tick(1.0, 5, 2), _tick(2.0, None, None)])
    assert "zero query delta" in reason([_tick(1.0, 5, 2), _tick(2.0, 5, 2)])
    assert "backwards" in reason([_tick(1.0, 9, 2), _tick(2.0, 5, 2)])
    assert "backwards" in reason([_tick(1.0, 5, 4), _tick(2.0, 9, 2)])
    assert "exceed" in reason([_tick(1.0, 5, 0), _tick(2.0, 10, 9)])
    # a NaN or non-numeric counter is a tick that carries nothing
    assert "single carrying tick" in reason([_tick(1.0, 5, 2), _tick(2.0, float("nan"), 3)])
    assert "single carrying tick" in reason([_tick(1.0, 5, 2), _tick(2.0, "7", 3)])
    # review LOW 1: an integer beyond the float range is not a number (math.isfinite raises on it)
    assert t._is_number(10 ** 400) is False
    assert "single carrying tick" in reason([_tick(1.0, 5, 2), _tick(2.0, 10 ** 400, 3)])
    # a record without a timestamp cannot be placed in the window
    assert "inside" in reason([{"instance": "single", QUERIES: 1, HITS: 1}])


def test_window_prefix_hit_delta_reports_each_instance_role_separately() -> None:
    records = [
        _tick(1.0, 10, 2, "prefill"), _tick(2.0, 30, 12, "prefill"),
        _tick(1.0, 100, 90, "decode"), _tick(2.0, 100, 90, "decode"),
        {"ts_s": 1.5, QUERIES: 3, HITS: 1},  # legacy record without a role: single
        {"ts_s": 2.5, "instance": 7, QUERIES: 7, HITS: 4},  # a non-string role: single too
    ]
    out = t.window_prefix_hit_delta(records, 0.0, 4.0)
    assert set(out) == {"prefill", "decode", "single"}
    assert out["prefill"]["share"] == pytest.approx(0.5)
    assert out["decode"]["share"] is None and "zero query delta" in out["decode"]["reason"]
    assert out["single"]["share"] == pytest.approx(0.75)
