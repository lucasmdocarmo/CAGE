"""Regression pins for the cage-stats E2b fix in the INSTALLED package (task #143).

Finding L-A (CODE_ASSERTION_2026-08.md, Topic 13, CRITICAL): requirements.txt
pinned cage-stats at 1cb902e2 -- one commit BEHIND df0eab46, the E2b fix. At
the old commit ``kv_usage = first_value(...) or 0.0`` FABRICATED an
"unpressured" reading whenever the ``vllm:kv_cache_usage_perc`` gauge was
absent from a scrape, so the regime gate's E2b refusal never fired and section
6.1(a) windows silently mislabeled. Invisible to the suite because every other
test runs on fixtures or a CAGE_STATS_HOME checkout -- never the INSTALLED
distribution the GPU box actually uses.

These tests import the installed ``cage_stats`` and exercise the exact seam
the df0eab46 diff touched (metrics/engine.py, metrics/kv.py, metrics/state.py;
mirrors that commit's own tests/test_kv_usage_absence.py):
absent gauge -> None, genuine 0.0 -> 0.0, and used-tokens never fabricates 0.
cage-stats is REQUIRED infra: a missing install is a loud import error here,
never a skip. Preflight gate (q) (scripts/checks/preflight_check.sh,
CAGE-STATS-PIN-PARITY-GATE) asserts the same pin==installed parity at launch.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import re
import time
from pathlib import Path

import cage_stats
from cage_stats.metrics import engine as cage_stats_engine
from cage_stats.metrics.engine import MetricsEngine
from cage_stats.metrics.parse import parse_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"

#: The commit that FABRICATED kv_usage=0.0 on an absent gauge (pre-E2b).
BROKEN_COMMIT = "1cb902e2881a22fe2580ed264bd8a8ce16eb43fb"

# Minimal scrape carrying SOME vllm telemetry but NO occupancy gauge (mirrors
# the upstream df0eab46 test fixture).
BASE = 'vllm:num_requests_running{model_name="m",engine="0"} 0.0\n'


def _derive(text: str):
    return MetricsEngine().derive(parse_metrics(text), now=time.time())


def _pinned_sha() -> str:
    lines = [
        line.split("#", 1)[0].strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    ]
    hits = [line for line in lines if line.startswith("cage-stats")]
    assert len(hits) == 1, f"expected exactly one cage-stats pin, got {hits!r}"
    m = re.match(r"^cage-stats\s*@\s*git\+\S+@([0-9a-f]{40})$", hits[0])
    assert m, f"cage-stats pin is not a full-SHA git pin: {hits[0]!r}"
    return m.group(1)


# ---------------------------------------------------------------------------
# Pin + install identity
# ---------------------------------------------------------------------------


def test_requirements_pin_is_a_full_sha_past_the_broken_commit() -> None:
    sha = _pinned_sha()
    assert sha != BROKEN_COMMIT, (
        "requirements.txt regressed to the pre-E2b cage-stats commit that "
        "fabricates rho_KV=0.0 from an absent KV-usage gauge (finding L-A)")


def test_cage_stats_imports_from_site_packages_not_a_checkout() -> None:
    """These pins are about the INSTALLED distribution (what the GPU box
    runs). If cage_stats resolves from a repo checkout (CAGE_STATS_HOME on
    PYTHONPATH), the parity claim is vacuous -- fail loudly."""
    assert "site-packages" in Path(cage_stats.__file__).parts, (
        f"cage_stats imported from {cage_stats.__file__} -- not the installed "
        f"distribution; unset PYTHONPATH/CAGE_STATS_HOME shadowing")


def test_installed_commit_matches_requirements_pin() -> None:
    """Python-level mirror of preflight gate (q): the installed dist records
    the pinned commit via PEP 610 vcs_info or the +g<sha> local version."""
    dist = md.distribution("cage-stats")
    commits = set()
    raw = dist.read_text("direct_url.json")
    if raw:
        commit = json.loads(raw).get("vcs_info", {}).get("commit_id")
        if commit:
            commits.add(commit.lower())
    m = re.search(r"\+g([0-9a-f]{40})$", dist.version)
    if m:
        commits.add(m.group(1))
    assert commits, (
        f"installed cage-stats {dist.version} records NO install commit "
        f"(no direct_url vcs_info, no +g<sha> local version) -- reinstall "
        f"from the pinned commit so parity is verifiable")
    assert commits == {_pinned_sha()}, (
        f"installed cage-stats commit {commits} != requirements.txt pin "
        f"{_pinned_sha()} -- the venv does not match the registration")


# ---------------------------------------------------------------------------
# E2b behavior on the installed package (the seam the df0eab46 diff touched)
# ---------------------------------------------------------------------------


def test_kv_usage_is_none_when_gauge_absent() -> None:
    """THE L-A regression: no occupancy gauge in the scrape -> unknown (None),
    never a fabricated "unpressured" 0.0."""
    assert _derive(BASE).kv_usage is None


def test_kv_usage_genuine_zero_survives_as_zero() -> None:
    """A REAL 0.0 gauge reading (empty cache) must stay 0.0, not None: the
    old ``or 0.0`` collapsed both cases (0.0 is falsy)."""
    text = BASE + 'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.0\n'
    snap = _derive(text)
    assert snap.kv_usage == 0.0
    assert snap.kv_usage is not None


def test_kv_usage_passes_through_nonzero_value() -> None:
    text = BASE + 'vllm:kv_cache_usage_perc{model_name="m",engine="0"} 0.42\n'
    assert _derive(text).kv_usage == 0.42


def test_kv_used_tokens_none_when_usage_absent() -> None:
    """Capacity is knowable from cache_config_info, but without the occupancy
    gauge the used-token count is not -> None, never a fabricated 0."""
    cc = (
        'vllm:cache_config_info{block_size="16",cache_dtype="auto",'
        'num_gpu_blocks="100",engine="0"} 1.0\n'
    )
    snap = _derive(BASE + cc)
    assert snap.kv_capacity_tokens == 1600
    assert snap.kv_used_tokens is None


def test_installed_engine_source_has_no_or_zero_fabrication() -> None:
    """Belt-and-braces source pin on the INSTALLED engine.py: the exact
    pre-E2b fabrication expression must be gone."""
    source = Path(cage_stats_engine.__file__).read_text(encoding="utf-8")
    assert 'first_value(fam, "vllm:kv_cache_usage_perc") or 0.0' not in source, (
        "installed cage_stats/metrics/engine.py still fabricates kv_usage=0.0 "
        "for an absent gauge (pre-E2b code) -- the install lags the pin")
