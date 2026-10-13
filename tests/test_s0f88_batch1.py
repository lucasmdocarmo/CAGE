"""S0F-88 batch 1 (S0 attempt 4 bugs, 2026-10-12): self-contained regressions.

Each test fails on the code at cef4cde and passes after the fix it names:
B-12 the window error counts include errors returned inside a response;
B-4 a connect-phase failure refuses a window and a calibration probe; B-1
the runners raise the open-file soft limit, record it, and every pod job
raises it too; Q-6 BUILD_INFO answers before git; Q-7 the strict token
counter passes verbose=False; Q-9 a cell skipped whole by a failed dry
window is a SKIP, not a layout FAIL; Q-1 the ladder label names both
attainments when they disagree. P-2 lives in tests/test_experiment_master.py.
"""

from __future__ import annotations

import asyncio
import errno
import importlib.util
import inspect
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestration import load_generator as lg  # noqa: E402
from src.orchestration.campaign_session import (  # noqa: E402
    CampaignSessionError,
    refuse_connect_error_rows,
)

SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: The four connect-error texts S0 attempt 4 wrote (requests.jsonl ``error``).
REAL_CONNECT_ERRORS = (
    "Cannot connect to host localhost:8000 ssl:default [Too many open files]",
    "Cannot connect to host localhost:30000 ssl:default [Too many open files]",
    "Cannot connect to host localhost:30000 ssl:default [None]",
    "Cannot connect to host localhost:8000 ssl:default [None]",
)


# --------------------------------------------------------------------------- #
# B-12: errors returned inside a result count
# --------------------------------------------------------------------------- #


def _record(index: int, result: Any) -> lg.RequestRecord:
    rec = lg.RequestRecord(index=index, scheduled_offset_s=float(index), scheduled_ts=float(index))
    rec.actual_send_ts = float(index)
    rec.completion_ts = float(index) + 1.0
    rec.result = result
    return rec


def test_a_returned_error_is_not_completed_and_counts_as_an_error_b12() -> None:
    """B-12: one S0 window reported n_errors 0 and 6,610 completions beside
    1,557 error rows: the adapters RETURN their errors inside the response."""
    ok = types.SimpleNamespace(error=None)
    bad = types.SimpleNamespace(error=REAL_CONNECT_ERRORS[0])
    empty = types.SimpleNamespace(error="")  # empty text is absence
    records = [_record(0, ok), _record(1, bad), _record(2, empty)]
    assert [r.completed for r in records] == [True, False, True]
    assert records[1].result_error == REAL_CONNECT_ERRORS[0]
    report = lg.DispatchReport(
        schedule=lg.generate_arrival_schedule(1.0, seed=1, n_requests=3),
        run_start_ts=0.0,
        records=records,
    )
    assert report.n_completed == 2 and report.n_errors == 1
    assert report.to_manifest()["n_errors"] == 1


# --------------------------------------------------------------------------- #
# B-4: connect-phase failures are client faults
# --------------------------------------------------------------------------- #


def test_the_connect_predicate_matches_every_s0_text_and_aiohttp_b4() -> None:
    import aiohttp

    key = types.SimpleNamespace(host="localhost", port=8000, ssl=True, is_ssl=False)
    built = (
        str(aiohttp.ClientConnectorError(key, OSError(errno.EMFILE, os.strerror(errno.EMFILE)))),
        str(aiohttp.ClientConnectorError(key, OSError("mixed happy-eyeballs failure"))),
    )
    for text in REAL_CONNECT_ERRORS + built:
        assert lg.is_connect_error(text), text
    # the dispatcher's own label form ("<Type>: <message>")
    assert lg.is_connect_error(f"ClientConnectorError: {REAL_CONNECT_ERRORS[0]}")
    for text in (
        None,
        "",
        "400 Client Error: Bad Request for url: http://localhost:30000/v1/chat/completions",
        "TimeoutError",
        "engine_error: out of memory",
        "the model said Cannot connect to host",  # not at a text or label start
    ):
        assert not lg.is_connect_error(text), text


def test_a_window_with_one_connect_error_row_is_refused_b4() -> None:
    rows = [{"example_id": f"q{i}", "error": None} for i in range(5)]
    refuse_connect_error_rows(rows)  # clean rows pass
    rows[3]["error"] = REAL_CONNECT_ERRORS[2]
    with pytest.raises(CampaignSessionError, match="1 request row") as exc:
        refuse_connect_error_rows(rows)
    assert "row 3" in str(exc.value) and "B-1" in str(exc.value)
    # the dispatcher's own field is checked too
    with pytest.raises(CampaignSessionError):
        refuse_connect_error_rows([{"dispatch_error": f"ClientConnectorError: {REAL_CONNECT_ERRORS[0]}"}])


def test_emit_window_refuses_connect_errors_before_writing_b4() -> None:
    """The refusal runs first in emit_window, beside the other two summary
    refusals, so a refused window writes nothing."""
    from src.orchestration.campaign_session import CampaignCellSession

    src = inspect.getsource(CampaignCellSession.emit_window)
    i_refuse = src.index("refuse_connect_error_rows(results_rows)")
    assert i_refuse < src.index("self._ensure_run(") < src.index("cell.add_window(")


def test_a_calibration_probe_with_a_connect_error_raises_b4(monkeypatch: pytest.MonkeyPatch) -> None:
    cc = _load("calibrate_cell_s0f88", SCRIPTS / "3_run" / "calibrate_cell.py")
    from src.orchestration.calibration import CalibrationError

    monkeypatch.setattr(cc, "PROBE_WINDOW_S", 0.4)
    monkeypatch.setattr(cc, "PROBE_WARMUP_S", 0.0)

    class _Adapter:
        def __init__(self, error: Any) -> None:
            self.error = error

        async def async_stream_generate(self, request: Any, on_first_token: Any = None) -> Any:
            return types.SimpleNamespace(error=self.error)

    step = asyncio.run(cc.probe_rate(_Adapter(None), [object()], 40.0, seed=7))
    assert step.n_completed == step.n_scheduled > 0
    with pytest.raises(CalibrationError, match="connect"):
        asyncio.run(cc.probe_rate(_Adapter(REAL_CONNECT_ERRORS[0]), [object()], 40.0, seed=7))


# --------------------------------------------------------------------------- #
# B-1: the open-file soft limit is raised and recorded
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX resource limits")
def test_the_open_file_limit_is_raised_from_1024_b1() -> None:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard <= 1024:
        pytest.skip(f"hard limit {hard} leaves nothing to raise")
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))
    try:
        limits = lg.raise_open_file_limit()
        assert limits["soft_before"] == 1024
        after = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        assert after == resource.RLIM_INFINITY or after > 1024
        assert limits["soft_after"] in ("unlimited", after)
        # never lowers: a second call keeps the raised value
        lg.raise_open_file_limit()
        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == after
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_both_runners_raise_the_limit_first_and_every_pod_job_raises_it_b1() -> None:
    runner_src = (SCRIPTS / "3_run" / "run_experiment.py").read_text(encoding="utf-8")
    main_src = runner_src[runner_src.index("\ndef main():"):]
    assert main_src.index("raise_open_file_limit()") < main_src.index("argparse.ArgumentParser(")
    assert '"open_file_limit": dict(_OPEN_FILE_LIMITS) or None' in runner_src
    cal_src = (SCRIPTS / "3_run" / "calibrate_cell.py").read_text(encoding="utf-8")
    cal_main = cal_src[cal_src.index("\ndef main("):]
    assert cal_main.index("raise_open_file_limit()") < cal_main.index("build_requests(")
    pod_job = (SCRIPTS / "6_experiments" / "pod_job.sh").read_text(encoding="utf-8")
    assert "( ulimit -Sn \\$(ulimit -Hn) 2>/dev/null || true; PYTHONUNBUFFERED=1 bash" in pod_job


# --------------------------------------------------------------------------- #
# Q-6: BUILD_INFO answers before git
# --------------------------------------------------------------------------- #


def test_build_info_answers_without_a_failed_git_call_q6(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from src.observability import provenance

    sha = "c" * 40
    (tmp_path / "BUILD_INFO").write_text(f"sha={sha}\ndirty=0\npackaged_at=x\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="cage.observability.provenance"):
        assert provenance.git_sha(str(tmp_path)) == sha
        assert provenance.git_dirty(str(tmp_path)) is False
    assert not [r for r in caplog.records if "failed" in r.getMessage()]


def test_capture_git_metadata_falls_back_to_build_info_q6(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every S0 attempt 4 metrics.json recorded git_commit null: the pod has
    BUILD_INFO and no .git."""
    runner = _load("run_experiment_s0f88", SCRIPTS / "3_run" / "run_experiment.py")
    from src.observability import provenance

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=128, stdout="", stderr="fatal: not a git repository"),
    )
    monkeypatch.setattr(provenance, "_build_info", lambda repo: {"sha": "d" * 40, "dirty": "0"})
    meta = runner.capture_git_metadata()
    assert meta["git_commit"] == "d" * 40 and meta["git_dirty"] is False
    assert meta["git_repository_present"] is False and meta["git_source"] == "BUILD_INFO"


# --------------------------------------------------------------------------- #
# Q-7: the strict counter never asks for the length warning
# --------------------------------------------------------------------------- #


def test_the_strict_token_counter_passes_verbose_false_q7() -> None:
    from src.evaluation.quality import QualityEvaluator

    seen: dict[str, Any] = {}

    class _Tok:
        def __call__(self, texts: Any, **kw: Any) -> dict[str, Any]:
            seen.update(kw)
            return {"input_ids": [[1] * 40000 for _ in texts]}

    assert QualityEvaluator._count_tokens_strict(["x"], _Tok()) == [40000]
    assert seen == {"add_special_tokens": False, "verbose": False}


# --------------------------------------------------------------------------- #
# Q-9: a cell skipped whole by a failed dry window is a SKIP
# --------------------------------------------------------------------------- #


def _sentinel(cell: Path, name: str, reason: str | None) -> None:
    cell.mkdir(parents=True, exist_ok=True)
    body = "STATUS=failed dataset=qasper" + (f" reason={reason}" if reason else "") + " utc=2026-10-11T19:16:00Z\n"
    (cell / name).write_text(body, encoding="utf-8")


def test_only_dry_window_sentinels_make_a_skipped_cell_q9(tmp_path: Path) -> None:
    org = _load("organize_results_s0f88", SCRIPTS / "4_analysis" / "organize_results.py")

    dry = tmp_path / "dry"
    _sentinel(dry, ".STATUS-qasper", "dry-window-UNPRESSURED")
    _sentinel(dry, ".STATUS-squad_v2", "dry-window-PAST_CLIFF")
    assert org.dry_window_skip_reasons(dry) == [
        ".STATUS-qasper: dry-window-UNPRESSURED",
        ".STATUS-squad_v2: dry-window-PAST_CLIFF",
    ]
    runner_exit = tmp_path / "runner"
    _sentinel(runner_exit, ".STATUS-qasper", "runner-exit-1")
    assert org.dry_window_skip_reasons(runner_exit) is None
    mixed = tmp_path / "mixed"
    _sentinel(mixed, ".STATUS-qasper", "dry-window-UNPRESSURED")
    _sentinel(mixed, ".STATUS-squad_v2", "relaunch-failed")
    assert org.dry_window_skip_reasons(mixed) is None
    no_reason = tmp_path / "noreason"
    _sentinel(no_reason, ".STATUS-qasper", None)
    assert org.dry_window_skip_reasons(no_reason) is None
    beside = tmp_path / "beside"
    _sentinel(beside, ".STATUS-qasper", "dry-window-UNPRESSURED")
    (beside / "cell.json").write_text("{}", encoding="utf-8")
    assert org.dry_window_skip_reasons(beside) is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert org.dry_window_skip_reasons(empty) is None


def test_verify_results_reports_a_dry_window_skip_not_a_fail_q9(tmp_path: Path) -> None:
    vr = _load("verify_results_s0f88", SCRIPTS / "4_analysis" / "verify_results.py")
    cells = tmp_path / "cells"
    key = "corpus-reuse|none|none|single|sglang|qwen3-14b|F3|r0.5|lam0.95"
    _sentinel(cells / key, ".STATUS-qasper", "dry-window-UNPRESSURED")
    bad = "corpus-comp|none|none|single|vllm|qwen3-14b|F3|r0.5|lam0.95"
    _sentinel(cells / bad, ".STATUS-qasper", "runner-exit-1")
    findings: list[Any] = []
    vr._walk_cells(tmp_path, findings)
    by_where = {f.where: f for f in findings}
    assert by_where[f"cells/{key}"].severity == "SKIP"
    assert "dry-window-UNPRESSURED" in by_where[f"cells/{key}"].detail
    # review 2026-10-13 (decision 3): the runner failure is a FAIL named by its
    # reason at the cell, no longer a "missing cell.json" at cell.json
    assert by_where[f"cells/{bad}"].severity == "FAIL" and "runner-exit-1" in by_where[f"cells/{bad}"].detail


def test_a_cell_that_failed_before_any_window_is_excluded_and_named_fail(tmp_path: Path) -> None:
    """Q-9 (R15) and decision 3 (2026-10-13): a sentinel-only cell whose reason
    is a relaunch or runner failure is excluded from the index (the walk no
    longer raises on it) and verify_results names it FAIL by its reason."""
    import json

    import test_verify_results_v2 as tv

    org = tv.vr.org
    run_dir = tv._build_tree(tmp_path / "run")  # unsealed: the failed cell lands before the seal
    good = sorted(p.name for p in (run_dir / "cells").iterdir())
    model = good[0].split("|")[5]
    failed_key = f"corpus-reuse|none|none|single|sglang|{model}|F3|r0.5|lam0.95"
    assert failed_key not in good
    _sentinel(run_dir / "cells" / failed_key, ".STATUS-qasper", "relaunch-failed")
    assert org.sentinel_only_reasons(run_dir / "cells" / failed_key) == [".STATUS-qasper: relaunch-failed"]
    assert org.dry_window_skip_reasons(run_dir / "cells" / failed_key) is None
    tv._seal(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = org.walk_run_tree(run_dir, manifest)  # raised LayoutError("missing cell.json") before
    assert {r.row_key for r in records} == set(good)
    report = tv.vr.verify_run(run_dir)
    assert report["ok"] is False
    (fail,) = [f for f in report["findings"] if f["check"] == "layout" and f["severity"] == "FAIL"]
    assert fail["where"] == f"cells/{failed_key}" and "relaunch-failed" in fail["detail"] and "R15" in fail["detail"]
    assert not any("missing cell.json" in f["detail"] for f in report["findings"])


# --------------------------------------------------------------------------- #
# Q-1: the ladder label names both attainments when they disagree
# --------------------------------------------------------------------------- #


def test_the_ladder_label_names_the_trimmed_attainment_when_it_flips_q1(tmp_path: Path) -> None:
    """vLLM r=0.5 s05 of S0 attempt 4: regime.json read IN_REGIME on every
    row (warm-up included); the trimmed rows the step uses read 0.898."""
    rc = _load("run_campaign_s0f88", SCRIPTS / "3_run" / "run_campaign.py")
    row_key = "gold-fresh|none|none|single|vllm|qwen3-14b|F2|r0.5|lam0.85"
    window = tmp_path / "cells" / row_key / "window_qasper-01"
    window.mkdir(parents=True)
    # 20 warm-up rows (arrival < 10 s), all ok; 100 measured rows, 85 ok
    rows = [{"arrival_s": 1.0 + i * 0.4, "ok": True} for i in range(20)]
    rows += [{"arrival_s": 10.0 + i * 0.6, "ok": i >= 15} for i in range(100)]
    (window / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    regime = {
        "label": "IN_REGIME",
        "attainment": 105 / 120,
        "inputs": {"rho_kv_time_avg": 0.95, "queue_waiting_share": 0.99},
    }
    (window / "regime.json").write_text(json.dumps(regime), encoding="utf-8")
    step, label = rc._read_ladder_window(tmp_path, row_key, "qasper", 20.0, "ladder")
    assert step.attainment == pytest.approx(0.85)
    assert label == "IN_REGIME (trimmed attainment 0.850: PAST_CLIFF)"
    # agreeing labels stay a bare label
    regime["inputs"]["rho_kv_time_avg"] = 0.5
    regime["label"] = "PAST_CLIFF"
    (window / "regime.json").write_text(json.dumps(regime), encoding="utf-8")
    assert rc._read_ladder_window(tmp_path, row_key, "qasper", 20.0, "ladder")[1] == "PAST_CLIFF"
