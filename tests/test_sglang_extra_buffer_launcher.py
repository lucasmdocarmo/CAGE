"""ADR-0195: the SGLang launcher's Mamba-scheduler and overlap-schedule knobs.

The extra-buffer probe (one no-load B3 cell, owner decision 2 option 3,
2026-11-08) serves SGLang with `--mamba-scheduler-strategy extra_buffer
--disable-overlap-schedule`. Two env knobs add exactly those flags; unset
leaves the argv byte-identical; both are reuse dials and serving-config
fields; a strategy value outside {no_buffer, extra_buffer} is refused at the
start|restart gate before any server is touched.
Hermetic: the stubs and helpers of tests/test_tp_flags.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tp_flags import SGLANG_SH, _run_launcher, _server_args_line, stub_bin  # noqa: E402,F401

MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


def test_the_strategy_knob_adds_only_its_flag_and_value(stub_bin: Path) -> None:
    on = _run_launcher(SGLANG_SH, stub_bin, "start", MODEL, CAGE_SGLANG_MAMBA_SCHEDULER_STRATEGY="extra_buffer")
    line = _server_args_line(on.stdout)
    assert " --mamba-scheduler-strategy extra_buffer" in line
    assert "--disable-overlap-schedule" not in line
    assert "Mamba scheduler strategy: extra_buffer" in on.stdout
    off = _run_launcher(SGLANG_SH, stub_bin, "start", MODEL)
    off_line = _server_args_line(off.stdout)
    assert "--mamba-scheduler-strategy" not in off_line
    assert line.replace(" --mamba-scheduler-strategy extra_buffer", "") == off_line


def test_the_overlap_knob_adds_only_its_flag(stub_bin: Path) -> None:
    on = _run_launcher(SGLANG_SH, stub_bin, "start", MODEL, CAGE_SGLANG_DISABLE_OVERLAP_SCHEDULE="1")
    line = _server_args_line(on.stdout)
    assert " --disable-overlap-schedule" in line and "--mamba-scheduler-strategy" not in line
    assert "Overlap schedule OFF" in on.stdout
    off = _run_launcher(SGLANG_SH, stub_bin, "start", MODEL)
    assert line.replace(" --disable-overlap-schedule", "") == _server_args_line(off.stdout)


def test_both_knobs_together_compose_the_probe_argv(stub_bin: Path) -> None:
    proc = _run_launcher(
        SGLANG_SH, stub_bin, "start", MODEL,
        CAGE_SGLANG_MAMBA_SCHEDULER_STRATEGY="extra_buffer",
        CAGE_SGLANG_DISABLE_OVERLAP_SCHEDULE="1",
    )
    line = _server_args_line(proc.stdout)
    assert " --mamba-scheduler-strategy extra_buffer" in line and " --disable-overlap-schedule" in line


@pytest.mark.parametrize("value", ["0", ""])
def test_overlap_values_other_than_one_leave_the_argv_alone(stub_bin: Path, value: str) -> None:
    proc = _run_launcher(SGLANG_SH, stub_bin, "start", MODEL, CAGE_SGLANG_DISABLE_OVERLAP_SCHEDULE=value)
    assert "--disable-overlap-schedule" not in _server_args_line(proc.stdout)


def test_no_buffer_is_passed_through_as_the_explicit_default(stub_bin: Path) -> None:
    proc = _run_launcher(SGLANG_SH, stub_bin, "start", MODEL, CAGE_SGLANG_MAMBA_SCHEDULER_STRATEGY="no_buffer")
    assert " --mamba-scheduler-strategy no_buffer" in _server_args_line(proc.stdout)


@pytest.mark.parametrize("value", ["bogus", "EXTRA_BUFFER", "extra_buffer "])
def test_an_unregistered_strategy_is_refused_before_any_server_is_touched(stub_bin: Path, value: str) -> None:
    proc = _run_launcher(SGLANG_SH, stub_bin, "start", MODEL, CAGE_SGLANG_MAMBA_SCHEDULER_STRATEGY=value)
    assert proc.returncode != 0
    assert "Server args:" not in proc.stdout, "the launch was composed despite the refusal"
    assert "CAGE_SGLANG_MAMBA_SCHEDULER_STRATEGY" in proc.stdout + proc.stderr
    assert "not touching any server" in proc.stdout + proc.stderr


def test_the_reuse_check_and_the_serving_config_carry_both_knobs() -> None:
    text = SGLANG_SH.read_text(encoding="utf-8")
    # Space-anchored dial checks, both directions, like the TP and piecewise dials.
    assert '" --mamba-scheduler-strategy ${CAGE_SGLANG_MAMBA_SCHEDULER_STRATEGY} "' in text
    assert '"$live_cmd" != *"--mamba-scheduler-strategy"*' in text
    assert '" --disable-overlap-schedule "' in text
    assert '"$live_cmd" != *"--disable-overlap-schedule"*' in text
    # The fraction memo key carries both (a different scheduler is a different launch).
    memo = text[text.index("_sglang_memo_key() {"):]
    memo = memo[: memo.index("\n}")]
    assert "CAGE_SGLANG_MAMBA_SCHEDULER_STRATEGY" in memo and "CAGE_SGLANG_DISABLE_OVERLAP_SCHEDULE" in memo
    # Serving-config fields.
    assert '"mamba_scheduler_strategy": os.environ.get("SC_MAMBA_STRATEGY") or None' in text
    assert '"disable_overlap_schedule": os.environ.get("SC_OVERLAP_OFF") == "1"' in text
