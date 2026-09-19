"""Pilot-tree campaign-env guards + pilot byte-identity pin (#116 verifier findings).

Verifier findings on the #116 seam (2026-08-21):
- Finding 1 (BLOCKER, fixed): run_experiment.py inserted ``measured_window``
  into every PILOT metrics.json unconditionally, falsifying the "pilot path
  byte-identical" contract. The insertion is now gated on
  ``campaign_session is not None``.
- Finding 6 (minor, fixed): the pilot-only variant runners
  (run_prefix_envelope.sh / run_memory_sweep.sh / run_kv_store.sh) had no
  standalone campaign guard — with CAGE_CAMPAIGN_ROOT exported, their
  run_experiment invocations would enter campaign mode and mint variant
  windows as ordinary campaign cells. Each now refuses fast under
  CAGE_CAMPAIGN_ROOT / CAGE_CAMPAIGN.

These tests pin both fixes. All offline, seconds.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"

PILOT_ONLY_RUNNERS = [
    "scripts/3_run/run_prefix_envelope.sh",
    "scripts/3_run/run_memory_sweep.sh",
    "scripts/3_run/run_kv_store.sh",
]

GUARD_MSG = "pilot-only runner: refusing under CAGE_CAMPAIGN_ROOT/CAGE_CAMPAIGN"


def _clean_env() -> dict:
    env = os.environ.copy()
    for k in list(env):
        if k.startswith("CAGE_CAMPAIGN"):
            env.pop(k)
    return env


@pytest.mark.parametrize("runner", PILOT_ONLY_RUNNERS)
def test_pilot_runner_refuses_under_campaign_root(runner: str) -> None:
    """Executing the real script with CAGE_CAMPAIGN_ROOT set must die fast."""
    env = _clean_env()
    env["CAGE_CAMPAIGN_ROOT"] = "/tmp/nonexistent-campaign-root"
    proc = subprocess.run(
        ["bash", runner],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0, f"{runner} must refuse under CAGE_CAMPAIGN_ROOT"
    combined = proc.stdout + proc.stderr
    assert GUARD_MSG in combined, (
        f"{runner} refusal must carry the guard message; got:\n{combined[-500:]}"
    )


@pytest.mark.parametrize("runner", PILOT_ONLY_RUNNERS)
def test_pilot_runner_refuses_under_campaign_flag_env(runner: str) -> None:
    """CAGE_CAMPAIGN (without a root) must refuse identically."""
    env = _clean_env()
    env["CAGE_CAMPAIGN"] = "s0"
    proc = subprocess.run(
        ["bash", runner],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert GUARD_MSG in proc.stdout + proc.stderr


@pytest.mark.parametrize("runner", PILOT_ONLY_RUNNERS)
def test_guard_sits_before_any_work(runner: str) -> None:
    """Source pin: the guard must appear before the first mutating command.

    The refusal fires right after the _common.sh source — before argument
    handling, directory creation, or any run_experiment invocation.
    """
    lines = (REPO_ROOT / runner).read_text(encoding="utf-8").splitlines()
    source_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip() == "source scripts/lib/_common.sh"),
        None,
    )
    assert source_idx is not None, f"{runner}: _common.sh source line not found"
    guard_idx = next((i for i, ln in enumerate(lines) if GUARD_MSG in ln), None)
    assert guard_idx is not None, f"{runner} lost its campaign guard"
    # "Immediately after sourcing": nothing but the guard block (comments/blank
    # lines allowed) may sit between the source line and the refusal — so the
    # guard fires before argument handling, mkdir, or any run_experiment call.
    assert source_idx < guard_idx <= source_idx + 10, (
        f"{runner}: guard at line {guard_idx + 1} must sit within 10 lines "
        f"after the _common.sh source (line {source_idx + 1})"
    )


#: Campaign-only metrics.json keys: ``measured_window`` (#116) and
#: ``quality_scoring`` (Batch 2 W1, ADR-0055). Both share the byte-identity
#: contract, so both are pinned by the same source guard.
CAMPAIGN_ONLY_SUMMARY_KEYS = ["measured_window", "quality_scoring"]


@pytest.mark.parametrize("key", CAMPAIGN_ONLY_SUMMARY_KEYS)
def test_campaign_only_summary_key_is_gated_in_source(key: str) -> None:
    """Pilot byte-identity pin (verifier finding 1; extended to quality_scoring
    by the W1 review).

    A campaign-only key may enter experiment_summary ONLY inside the
    ``campaign_session is not None`` branch — never as an inline literal of
    the summary dict (which the pilot path writes verbatim).
    """
    lines = RUN_EXPERIMENT.read_text(encoding="utf-8").splitlines()
    literal_sites = [
        i for i, ln in enumerate(lines) if re.search(rf'^\s*"{key}"\s*:', ln)
    ]
    assert not literal_sites, (
        f"{key} must not be an inline experiment_summary literal "
        f"(pilot path would carry it); found at lines {[i + 1 for i in literal_sites]}"
    )
    assign_sites = [
        i
        for i, ln in enumerate(lines)
        if f'experiment_summary["{key}"]' in ln
    ]
    assert assign_sites, f"campaign path must still record {key}"
    for i in assign_sites:
        window = "\n".join(lines[max(0, i - 12) : i])
        assert "campaign_session is not None" in window, (
            f"{key} assignment at line {i + 1} is not guarded by "
            "'campaign_session is not None' within the preceding lines"
        )


def test_run_experiment_compiles() -> None:
    """The gated-insertion edit must leave the module byte-compilable."""
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(RUN_EXPERIMENT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
