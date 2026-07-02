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
