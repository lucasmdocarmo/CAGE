"""ADR-0195: the identity gate's fourth phase, the SGLang extra-buffer probe.

Stage 4 starts SGLang a second time under the probe's launcher knobs and runs
the same prompts for two passes (phase ``sglang_xb``). The verdict records two
more comparisons, F (vLLM cache-off vs the probe's pass 1) and G (the probe
against itself across passes); neither decides. The phase is requested with
``--sglang-xb``; without it the verdict records it absent, honestly, and the
telemetry keys hold None. Fixtures: tests/test_identity_gate.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_identity_gate import MODEL, _phase_doc, _phases, _verdict, fakes, ig  # noqa: E402,F401

XB_VERSION = "0.5.10.post1"


def _xb_cached(k: int, ids: list) -> int:
    return 0 if k == 1 else len(ids) - 7


def _with_xb(tmp_path: Path, *, outputs: Optional[Callable] = None, cached: Callable = _xb_cached,
             **tweak: Any) -> tuple:
    d, pdoc = _phases(tmp_path, **tweak)
    kw: Dict[str, Any] = {"version": XB_VERSION, "cached": cached}
    if outputs is not None:
        kw["outputs"] = outputs
    (d / "sglang_xb.json").write_text(json.dumps(_phase_doc(pdoc, "sglang_xb", "sglang", 2, **kw)), encoding="utf-8")
    return d, pdoc


def test_the_phase_and_the_two_recorded_comparisons_are_registered() -> None:
    assert ig.PHASES["sglang_xb"] == ("sglang", 2)
    keys = [c[0] for c in ig.COMPARISONS]
    assert keys == ["A", "B", "C", "D", "E", "F", "G"]
    assert ("F", "vllm_off", 1, "sglang_xb", 1, False) in ig.COMPARISONS
    assert ("G", "sglang_xb", 1, "sglang_xb", 2, False) in ig.COMPARISONS
    assert ig.DECIDING == ["A", "B"]


def test_with_the_flag_the_probe_phase_is_read_and_f_and_g_are_recorded(tmp_path, fakes, capsys) -> None:
    d, pdoc = _with_xb(tmp_path)
    code, v = _verdict(tmp_path, d, "--sglang-xb")
    assert code == 0 and v["verdict"] == "PASS"
    assert v["phases"]["sglang_xb"] == {"engine": "sglang", "passes": 2, "engine_version": XB_VERSION}
    n = [len(p["token_ids"]) for p in pdoc["prompts"]]
    assert v["telemetry"]["sglang_xb_pass1_hits"] == 0
    assert v["telemetry"]["sglang_xb_pass2_cached_tokens"] == {"min": min(n) - 7, "median": (n[0] + n[1]) / 2 - 7, "max": max(n) - 7}
    assert v["comparisons"]["F"] == {"decides": False, "left": "vllm_off/1", "right": "sglang_xb/1", "compared": 2,
                                     "identical": 2, "first_divergence": None}
    assert v["comparisons"]["G"] == {"decides": False, "left": "sglang_xb/1", "right": "sglang_xb/2", "compared": 2,
                                     "identical": 2, "first_divergence": None}
    out = capsys.readouterr().out
    assert "F vllm_off/1 vs sglang_xb/1: 2/2 identical" in out and "G sglang_xb/1 vs sglang_xb/2: 2/2 identical" in out


def test_a_divergence_of_the_probe_alone_is_recorded_and_does_not_decide(tmp_path, fakes) -> None:
    d, _ = _with_xb(tmp_path, outputs=lambda i, ids, k: [1000 + i, 2000, 3001])
    code, v = _verdict(tmp_path, d, "--sglang-xb")
    assert code == 0 and v["verdict"] == "PASS"
    assert v["comparisons"]["F"]["identical"] == 0 and v["comparisons"]["F"]["first_divergence"]["position"] == 2
    assert v["comparisons"]["G"]["identical"] == 2, "the probe agrees with itself across passes"
    assert v["comparisons"]["C"]["identical"] == 2, "the default SGLang phase is untouched"


def test_without_the_flag_the_probe_is_absent_and_f_and_g_are_skipped_by_name(tmp_path, fakes) -> None:
    d, _ = _phases(tmp_path)
    code, v = _verdict(tmp_path, d)
    assert code == 0 and v["verdict"] == "PASS"
    assert v["phases"]["sglang_xb"] == "absent" and isinstance(v["phases"]["sglang"], dict)
    assert v["telemetry"]["sglang_xb_pass1_hits"] is None and v["telemetry"]["sglang_xb_pass2_cached_tokens"] is None
    for key in "FG":
        assert v["comparisons"][key] == {"decides": False, "left": None, "right": None, "compared": 0,
                                         "identical": None, "first_divergence": None, "skipped": "sglang_xb absent"}, key
    assert all(v["comparisons"][k]["compared"] == 2 for k in "CDE")


def test_no_sglang_and_the_probe_together_skip_only_the_default_phase(tmp_path, fakes) -> None:
    d, _ = _with_xb(tmp_path, sglang="absent")
    code, v = _verdict(tmp_path, d, "--no-sglang", "--sglang-xb")
    assert code == 0 and v["verdict"] == "PASS"
    assert v["phases"]["sglang"] == "absent" and isinstance(v["phases"]["sglang_xb"], dict)
    assert all(v["comparisons"][k]["skipped"] == "sglang absent" for k in "CDE")
    assert all(v["comparisons"][k]["compared"] == 2 for k in "FG")


def test_the_flag_without_the_phase_file_is_an_error(tmp_path, fakes, capsys) -> None:
    d, _ = _phases(tmp_path)
    code, v = _verdict(tmp_path, d, "--sglang-xb")
    assert code == 1 and v["verdict"] == "ERROR"
    assert "sglang_xb.json not found" in v["error"]
    assert capsys.readouterr().out.rstrip().splitlines()[-1] == "IDENTITY_GATE ERROR"


def test_a_probe_pass_1_hit_is_a_limitation_that_names_the_probe(tmp_path, fakes) -> None:
    d, _ = _with_xb(tmp_path, cached=lambda k, ids: 100 if k == 1 else len(ids) - 7)
    code, v = _verdict(tmp_path, d, "--sglang-xb")
    assert code == 0 and v["verdict"] == "PASS"
    assert v["telemetry"]["sglang_xb_pass1_hits"] == 2
    lim = [l for l in v["limitations"] if "sglang_xb" in l or "extra-buffer" in l]
    assert len(lim) == 1 and "2 of 2" in lim[0] and "F" in lim[0]
