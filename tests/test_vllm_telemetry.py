"""Tests for the CAGE -> cage-stats telemetry bridge (src/monitoring/vllm_telemetry.py).

Covers the graceful-skip path (cage-stats absent) and the mock-data capture path
(when the cage-stats repo is present as a sibling), so the integration stays green
without needing a live vLLM server.
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
    """No cage-stats import and no CLI -> everything degrades to None, never raises."""
    monkeypatch.setattr(t, "_try_import_api", lambda: None)
    monkeypatch.setattr(t.shutil, "which", lambda *_a, **_k: None)
    assert t.available() is False
    assert t.capture_snapshot("http://localhost:8000", mock=True) is None
    assert t.dashboard_text("http://localhost:8000", mock=True) is None
    assert t.capture("http://localhost:8000", mock=True) == (None, None)


@pytest.mark.skipif(not _HAVE_CAGE_STATS, reason="cage-stats sibling repo not present")
def test_mock_capture_via_cage_stats_home(monkeypatch):
    """With cage-stats resolvable (CAGE_STATS_HOME), capture mock telemetry headlessly."""
    monkeypatch.setenv("CAGE_STATS_HOME", _SIBLING_CAGE_STATS)
    assert t.available() is True
    snap, dash = t.capture("http://router:9000", mock=True)
    assert snap is not None
    assert "kv" in snap and snap.get("spec_acceptance") is not None
    assert dash and "CONCURRENCY" in dash
