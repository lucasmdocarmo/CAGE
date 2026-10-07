"""S0F-35: the SGLang launcher's piecewise CUDA-graph lever.

Live H100 (2026-10-07): SGLang 0.5.10.post1 died in its piecewise CUDA-graph
capture of Qwen3-14B and named `--disable-piecewise-cuda-graph` as the
workaround. CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH=1 adds exactly that flag;
unset leaves the argv byte-identical; the reuse check treats it as a dial.
Hermetic: the stubs and helpers of tests/test_tp_flags.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tp_flags import SGLANG_SH, _run_launcher, _server_args_line, stub_bin  # noqa: E402,F401


def test_the_lever_adds_only_the_piecewise_flag(stub_bin: Path) -> None:
    on = _run_launcher(SGLANG_SH, stub_bin, "start", "Qwen/Qwen3-14B", CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH="1")
    line = _server_args_line(on.stdout)
    assert "--disable-piecewise-cuda-graph" in line and "--disable-cuda-graph" not in line
    assert "Piecewise CUDA graph OFF" in on.stdout
    off = _run_launcher(SGLANG_SH, stub_bin, "start", "Qwen/Qwen3-14B")
    off_line = _server_args_line(off.stdout)
    assert "--disable-piecewise-cuda-graph" not in off_line
    assert line.replace(" --disable-piecewise-cuda-graph", "") == off_line


@pytest.mark.parametrize("value", ["0", ""])
def test_values_other_than_one_leave_the_argv_alone(stub_bin: Path, value: str) -> None:
    proc = _run_launcher(SGLANG_SH, stub_bin, "start", "Qwen/Qwen3-14B", CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH=value)
    assert "--disable-piecewise-cuda-graph" not in _server_args_line(proc.stdout)


def test_the_reuse_check_and_the_serving_config_carry_the_lever() -> None:
    text = SGLANG_SH.read_text(encoding="utf-8")
    assert '" --disable-piecewise-cuda-graph "' in text and '"$live_cmd" != *"--disable-piecewise-cuda-graph"*' in text
    assert '"disable_piecewise_cuda_graph": os.environ.get("SC_PIECEWISE_OFF") == "1"' in text
