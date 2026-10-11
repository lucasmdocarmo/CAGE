"""S0F-86 LOW batch (2026-10-10, ADR-0164): the self-contained regressions.

Each test fails on the code at d6a2066 and passes after the fix it names:
O1 the ledger EXTRA sweep skips dot entries the seal skips; S3 holm-any
per-dataset decisions use the Holm-adjusted p; S5 a logged UNBLIND event
refuses a second unblind; B7 the GPU tracker needs one handle per device;
B9a the first psutil cpu_percent() only primes; C3 a None gold position names
both causes. R1, O2, O4, O7, O8 live in tests/test_campaign_session.py, B1 and
B5 in tests/test_adapter_served_finish_s0f25.py, B4 in
tests/test_engine_adapters.py, C8 in tests/test_predicate_chain.py.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import conditioned_curves as cc  # noqa: E402
from src.analysis import degradation as deg  # noqa: E402
from src.analysis.stats import blinding  # noqa: E402
from src.analysis.stats.gatekeeping import (  # noqa: E402
    PrimaryOutcome,
    SecondaryOutcome,
    evaluate_chain,
)
from src.analysis.stats.ledger import hash_artifacts, verify_ledger, write_ledger  # noqa: E402
from src.evaluation import performance  # noqa: E402


def test_the_extra_sweep_skips_dot_entries_like_the_seal_o1(tmp_path: Path) -> None:
    """O1: Finder's .DS_Store and the driver's .STATUS-* files under cells/
    are outside the seal; the sweep reported them as EXTRA and organize and
    verify failed. A non-dot file added after the seal is still EXTRA."""
    cell = tmp_path / "cells" / "rk1"
    cell.mkdir(parents=True)
    (cell / "requests.jsonl").write_text("{}\n", encoding="utf-8")
    ledger = write_ledger(
        hash_artifacts([cell / "requests.jsonl"], base_dir=tmp_path),
        tmp_path / "ledger.json",
    )
    (cell / ".DS_Store").write_bytes(b"\x00")
    (cell / ".STATUS-qasper").write_text("running", encoding="utf-8")
    (tmp_path / "cells" / ".partial").mkdir()
    (tmp_path / "cells" / ".partial" / "x.json").write_text("{}", encoding="utf-8")
    assert verify_ledger(ledger, tmp_path, extra_roots=[tmp_path / "cells"]) == []
    (cell / "added.json").write_text("{}", encoding="utf-8")
    assert verify_ledger(ledger, tmp_path, extra_roots=[tmp_path / "cells"]) == [
        "EXTRA cells/rk1/added.json"
    ]


def test_holm_any_decisions_use_the_adjusted_p_s3() -> None:
    """S3: under holm-any the set failed (Holm 0.06 and 0.06) while each leg
    passed on its raw p and opened its secondary family confirmatorily."""
    primaries = [
        PrimaryOutcome("contrast-4", "d1", 0.03),
        PrimaryOutcome("contrast-4", "d2", 0.04),
    ]
    secondaries = [SecondaryOutcome("c13a", "f13", "contrast-4", "d1", 0.001)]
    trace = evaluate_chain(
        primaries, secondaries, intra_set_rule="holm-any",
        primary_order=["contrast-4"],
    )
    (set_decision,) = trace.set_decisions
    assert set_decision.passed is False
    assert set_decision.binding_p == pytest.approx(0.06)
    assert [d.passed for d in trace.primaries] == [False, False]
    assert [d.p_value for d in trace.primaries] == [0.03, 0.04]  # raw p reported
    (secondary,) = trace.secondaries
    assert secondary.status == "descriptive"
    # a passing set keeps its legs passing on the adjusted p
    ok = evaluate_chain(
        [PrimaryOutcome("contrast-4", "d1", 0.01), PrimaryOutcome("contrast-4", "d2", 0.04)],
        [], intra_set_rule="holm-any", primary_order=["contrast-4"],
    )
    assert [d.passed for d in ok.primaries] == [True, True]
    # review L2: plain bools, so stats.json can serialize the decisions
    assert all(type(d.passed) is bool for d in ok.primaries + trace.primaries)
    json.dumps([d.passed for d in ok.primaries])


def test_a_logged_unblind_refuses_a_reset_seal_s5(tmp_path: Path) -> None:
    """S5: map_sha256 does not cover unblinded_utc, so resetting the stamp to
    null re-armed a seal whose UNBLIND event the log records."""
    df = pd.DataFrame({"arm": ["B1", "B3", "B6", "B1"]})
    _, sealed = blinding.scramble_labels(df, "arm", 7, tmp_path / "sealed_arm_map.json")
    log = tmp_path / "unblind_log.jsonl"
    blinding.unblind(sealed, log)
    doc = json.loads(sealed.read_text(encoding="utf-8"))
    doc["unblinded_utc"] = None
    sealed.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(blinding.AlreadyUnblindedError, match="unblind log"):
        blinding.unblind(sealed, log)
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1


class _NVMLError(Exception):
    pass


def _fake_pynvml(failing_handle: int | None) -> types.ModuleType:
    mod = types.ModuleType("pynvml")
    mod.NVMLError = _NVMLError

    def handle(i: int) -> str:
        if i == failing_handle:
            raise _NVMLError(f"handle {i} unreadable")
        return f"h{i}"

    mod.nvmlInit = lambda: None
    mod.nvmlShutdown = lambda: None
    mod.nvmlDeviceGetCount = lambda: 2
    mod.nvmlDeviceGetHandleByIndex = handle
    mod.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(total=80 * 1024**3)
    mod.nvmlDeviceGetPowerManagementLimit = lambda h: 700_000
    return mod


def test_the_gpu_tracker_needs_a_handle_per_device_b7(monkeypatch: pytest.MonkeyPatch) -> None:
    """B7: one failing handle left the tracker available with no handles, and
    every tick counted a sample that read nothing."""
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(failing_handle=1))
    tracker = performance.GPUMetricsTracker()
    assert tracker.is_available() is False
    assert tracker.start_monitoring() is False
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(failing_handle=None))
    assert performance.GPUMetricsTracker().is_available() is True


def test_the_first_cpu_percent_only_primes_b9a() -> None:
    """B9a: psutil's first cpu_percent() on a Process returns a 0.0
    placeholder [V psutil 7.2.2 __init__.py:1080-1083]; it entered the
    average (0.0 and 50.0 read 25.0)."""
    readings = iter([0.0, 50.0])
    evaluator = performance.PerformanceEvaluator(monitor_resources=True)
    evaluator.process = types.SimpleNamespace(
        cpu_percent=lambda: next(readings),
        memory_info=lambda: types.SimpleNamespace(rss=100 * 1024 * 1024),
    )
    evaluator.start()
    evaluator.record_request("r1", ttft_ms=10.0, total_time_ms=20.0, num_tokens=3)
    evaluator.stop()
    metrics = evaluator.compute_metrics()
    assert metrics.avg_cpu_percent == pytest.approx(50.0)
    assert metrics.resource_sample_count == 2  # memory is read on both calls


def test_a_none_gold_position_names_both_causes_c3() -> None:
    """C3: after A2 a None position also means the example records no gold
    evidence; both reasons named only an absent column."""
    _, reason = cc.evidence_position_bin(None, 5)
    assert "gold_position_in_prompt absent" in reason
    assert "no gold evidence" in reason
    result = deg.classify_wrong_context(
        {"grounded": True, "grounding_score": 0.9, "gold_position_in_prompt": None}
    )
    assert result.value is None
    assert "no gold evidence" in (result.reason or "")


RUNNER_PATH = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"


def test_rerank_counters_report_the_measured_stage_only_c2() -> None:
    """C2: one reranker serves the warm-up stages too, and the summary wrote
    its cumulative counters (20 warm-up queries at pool 10 added 200 pairs
    beside 50 measured rows). The counters are now read after both warm-up
    stages and before the measured stage, and the summary reports the
    increase. A source-order pin plus the helper's behavior: the runner has
    no stub dense-plus-reranker harness (limitation, ADR-0164)."""
    import importlib.util

    src = RUNNER_PATH.read_text(encoding="utf-8")
    warm_pool = src.index('stage_name="WarmupPool"')
    warm = src.index('stage_name="Warmup")')
    snapshot = src.index("rerank_counts_before = (")
    measured = src.index("    stage_t_start = time.time()")
    assert warm_pool < warm < snapshot < measured
    assert 'getattr(reranker, "scored_pairs", None), rerank_counts_before[0]' in src
    assert 'getattr(reranker, "truncated_pairs", None), rerank_counts_before[1]' in src

    spec = importlib.util.spec_from_file_location("run_experiment_c2", RUNNER_PATH)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    assert runner._count_since(250, 200) == 50
    assert runner._count_since(None, 0) is None
    assert runner._count_since(3, None) is None


def test_a_logged_unblind_keeps_the_committed_lock_m1(tmp_path: Path) -> None:
    """M1 (review of ADR-0164): unblind logs its event, then stamps the seal.
    A failure between the two left the stamp null; the lock release dropped
    the OUTPUTS_COMMITTED lock while S5 refused every retry, so the look was
    blocked and never recorded as spent. The logged event now keeps it."""
    scripts = REPO_ROOT / "scripts" / "4_analysis"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_campaign_analysis as rca

    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis" / "20261010T000000Z"
    analysis_dir.mkdir(parents=True)
    df = pd.DataFrame({"arm": ["B1", "B3", "B6"]})
    _, sealed = blinding.scramble_labels(df, "arm", 7, rca.sealed_map_path(run_dir))
    log = run_dir / rca.BLINDING_DIRNAME / rca.UNBLIND_LOG_NAME
    sha = "a" * 40

    # Without a logged event an empty committed look is released (C4-1).
    rca._acquire_confirmatory_lock(run_dir, sha)
    rca._commit_confirmatory_lock(run_dir, sha, analysis_dir)
    rca._release_placeholder_lock(run_dir, analysis_dir)
    assert not (run_dir / rca.LOCK_NAME).exists()

    # The M1 state: the log line was written, the stamp was not.
    rca._acquire_confirmatory_lock(run_dir, sha)
    rca._commit_confirmatory_lock(run_dir, sha, analysis_dir)
    blinding.unblind(sealed, log)
    doc = json.loads(sealed.read_text(encoding="utf-8"))
    doc["unblinded_utc"] = None
    sealed.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    rca._release_placeholder_lock(run_dir, analysis_dir)
    lock = json.loads((run_dir / rca.LOCK_NAME).read_text(encoding="utf-8"))
    assert lock["phase"] == rca.LOCK_PHASE_COMMITTED  # the look stays spent
    with pytest.raises(blinding.AlreadyUnblindedError):
        blinding.unblind(sealed, log)
