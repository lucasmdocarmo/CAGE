"""Hardened pilot-era fences (task #135, findings I9 + I11-legacy).

Pins ACTUAL refusal/deprecation behavior — not banner strings — for all four
legacy analysis entry points against a synthetic RESULTS_LAYOUT-v2 campaign
tree, plus the artifact rename and disclosure fixes:

- statistical_tests.py: stderr deprecation banner on every invocation; the
  JSON summary stamped ``{"engine": "pilot-era statistical_tests.py — NOT the
  registered D9 artifact"}``; the registered ``stats.json`` output name
  REFUSED; a campaign root as ``--results-dir`` REFUSED (exit 2, no artifact);
- generate_plots.py: campaign root REFUSED before any rendering; the
  ``fillna(0)`` on cached_prompt_tokens replaced by drop-with-counted
  disclosure (absent telemetry must not render as 0% KV reuse); caption N
  derived from the loaded data, never the hardcoded pilot constants;
  pilot_stats.json is the stats artifact (legacy phase2_stats.json fallback
  for frozen archives);
- token_divergence.py: a campaign root yields a SKIP (exit 0) and mints NO
  artifact (the pilot-layout loader finds no reference arm);
- run_phase2_stats.sh: the new pilot-tree guard refuses campaign trees and
  non-pilot directories BEFORE any rm -rf/mkdir mutation, and passes a real
  pilot tree (guard-only test seam);
- _pub_tables.py: every emitted .tex carries the "PILOT DATA — design input
  only" header comment + a visible table note, and _tex_escape covers
  backslash, braces, $, ^, ~ in a single pass.

The old banner-string fences stay in tests/test_campaign_analysis.py; this
file carries the behavioral pins.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import statistical_tests as st  # noqa: E402

_STATS_PY = _SCRIPTS_DIR / "statistical_tests.py"
_PLOTS_PY = _SCRIPTS_DIR / "generate_plots.py"
_DIVERGE_PY = _SCRIPTS_DIR / "token_divergence.py"
_STATS_SH = _SCRIPTS_DIR / "run_phase2_stats.sh"


# ---------------------------------------------------------------------------
# Synthetic trees
# ---------------------------------------------------------------------------

def _make_campaign_tree(tmp_path: Path) -> Path:
    """Minimal RESULTS_LAYOUT v2 campaign run root: manifest.json + cells/."""
    run_dir = tmp_path / "campaign_run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "campaign": "cage-2026",
            "session": "group-a",
            "run_id": "r0001",
            "model": "qwen3-14b",
        }),
        encoding="utf-8",
    )
    window = (run_dir / "cells"
              / "gold-fresh|dense|none|single|vllm|qwen3-14b|F1"
              / "window_squad_v2-01")
    window.mkdir(parents=True)
    (window / "requests.jsonl").write_text(
        json.dumps({"example_id": "q1", "ttft_ms": 12.0}) + "\n",
        encoding="utf-8",
    )
    return run_dir


_PILOT_HEADER = ("example_id,error,empty_generation,repeat_index,ttft_ms,"
                 "latency_ms,prompt_tokens,cached_prompt_tokens,generated_answer")


def _pilot_rows(cell: str, trial: int, cached: dict[str, str]) -> str:
    lines = [_PILOT_HEADER]
    for i, ex in enumerate(("q1", "q2", "q3", "q4")):
        ttft = 100.0 + 10 * i + trial + (0.0 if cell == "no_cache" else -40.0)
        lat = ttft + 500.0 + 5 * i
        lines.append(f"{ex},,False,0,{ttft},{lat},1000,{cached[ex]},answer {i}")
    return "\n".join(lines) + "\n"


def _make_pilot_tree(tmp_path: Path) -> Path:
    """Pilot (Phase-2) run root: baselines/<cell>/trial_N/results.csv.

    no_cache: cached_prompt_tokens telemetry ABSENT on every row (the I11
    fillna(0) trap). prefix_cache: telemetry present, mixed 0 / nonzero.
    2 trials x 4 questions -> run_shape must derive 8/2/4, never 300/3/100.
    """
    root = tmp_path / "pilot_run"
    missing = {"q1": "", "q2": "", "q3": "", "q4": ""}
    present = {"q1": "800", "q2": "0", "q3": "800", "q4": "0"}
    for cell, cached in (("no_cache", missing), ("prefix_cache", present)):
        for trial in (1, 2):
            d = root / "baselines" / cell / f"trial_{trial}"
            d.mkdir(parents=True)
            (d / "results.csv").write_text(_pilot_rows(cell, trial, cached),
                                           encoding="utf-8")
    return root


def _run(cmd: list[str], **env_extra: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **env_extra}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                          cwd=str(REPO_ROOT), env=env)


# ---------------------------------------------------------------------------
# statistical_tests.py
# ---------------------------------------------------------------------------

class TestStatisticalTestsFences:
    def test_banner_stamp_and_pilot_stats_rename(self, tmp_path: Path) -> None:
        """Every invocation banners to stderr; pilot_stats.json carries the
        engine stamp as its FIRST key; the analysis itself still runs."""
        root = _make_pilot_tree(tmp_path)
        out = tmp_path / "out" / "pilot_stats.json"
        proc = _run([sys.executable, str(_STATS_PY),
                     "--results-dir", str(root / "baselines"),
                     "--metrics", "ttft_ms", "latency_ms",
                     "--bootstrap-iters", "50",
                     "--output", str(out)])
        assert proc.returncode == 0, proc.stderr
        assert "DEPRECATED" in proc.stderr
        assert "run_campaign_analysis.py" in proc.stderr
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["engine"] == st.PILOT_ENGINE_STAMP
        assert "NOT the registered D9 artifact" in payload["engine"]
        assert next(iter(payload)) == "engine"
        assert payload["comparisons"], "pilot analysis must still reproduce"

    def test_refuses_registered_stats_json_namesake(self, tmp_path: Path) -> None:
        root = _make_pilot_tree(tmp_path)
        out = tmp_path / "out" / "stats.json"
        proc = _run([sys.executable, str(_STATS_PY),
                     "--results-dir", str(root / "baselines"),
                     "--metrics", "ttft_ms",
                     "--output", str(out)])
        assert proc.returncode == 2
        assert "pilot_stats.json" in proc.stderr
        assert not out.exists(), "the registered namesake must never be minted"

    def test_refuses_campaign_tree(self, tmp_path: Path) -> None:
        run_dir = _make_campaign_tree(tmp_path)
        out = tmp_path / "out" / "pilot_stats.json"
        proc = _run([sys.executable, str(_STATS_PY),
                     "--results-dir", str(run_dir),
                     "--output", str(out)])
        assert proc.returncode == 2
        assert "CAMPAIGN" in proc.stderr
        assert "run_campaign_analysis.py" in proc.stderr
        assert not out.exists()

    def test_is_campaign_tree_heuristic(self, tmp_path: Path) -> None:
        assert st.is_campaign_tree(_make_campaign_tree(tmp_path))
        assert not st.is_campaign_tree(_make_pilot_tree(tmp_path))


# ---------------------------------------------------------------------------
# generate_plots.py
# ---------------------------------------------------------------------------

class TestGeneratePlotsFences:
    @pytest.fixture()
    def gp(self):
        pytest.importorskip("matplotlib")
        pytest.importorskip("seaborn")
        import generate_plots
        return generate_plots

    def test_refuses_campaign_tree(self, gp, tmp_path: Path) -> None:
        run_dir = _make_campaign_tree(tmp_path)
        plots = tmp_path / "plots"
        proc = _run([sys.executable, str(_PLOTS_PY),
                     "--results-dir", str(run_dir),
                     "--plots-dir", str(plots)])
        assert proc.returncode != 0
        assert "CAMPAIGN" in proc.stderr
        assert "figure_pipeline" in proc.stderr
        assert not plots.exists() or not any(plots.iterdir()), \
            "refusal must precede any rendering"
        # ... and the deprecation banner prints on EVERY invocation.
        assert "PILOT-ERA" in proc.stderr

    def test_build_summary_drops_and_counts_missing_cached(
            self, gp, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Absent cached_prompt_tokens telemetry: dropped AND counted, never
        fillna(0) -> a fabricated 0% KV reuse (finding I11)."""
        import math
        root = _make_pilot_tree(tmp_path)
        df, _long = gp.build_summary(root, bootstrap_iters=50)
        rows = {r["baseline"]: r for _, r in df.iterrows()}

        no_cache = rows["no_cache"]
        # ALL rows lack telemetry -> NO cached_pct/marginal_kv, not 0.
        assert "cached_pct" not in no_cache or math.isnan(float(no_cache["cached_pct"]))
        assert "marginal_kv" not in no_cache or math.isnan(float(no_cache["marginal_kv"]))
        assert int(no_cache["cached_missing_n"]) == 8
        # resident_kv only needs prompt_tokens -> still present.
        assert float(no_cache["resident_kv"]) == 1000.0

        prefix = rows["prefix_cache"]
        # Present telemetry (incl. genuine zeros) keeps its measured share:
        # (800+0+800+0)*2 / (1000*8) = 40%.
        assert float(prefix["cached_pct"]) == pytest.approx(40.0)
        assert int(prefix["cached_missing_n"]) == 0

        disclosed = capsys.readouterr().out
        assert "cached_prompt_tokens telemetry missing on 8/8 rows" in disclosed
        assert "0% KV reuse" in disclosed

    def test_caption_n_derived_from_data(self, gp, tmp_path: Path) -> None:
        from _results_loader import load_results_long
        root = _make_pilot_tree(tmp_path)
        shape = gp.run_shape(load_results_long(root))
        assert shape == {"rows": "8", "trials": "2", "questions": "4"}
        caveat = gp.caveat_line(shape)
        assert "N = 8 valid measurements per cell" in caveat
        assert "2 trials x 4 questions" in caveat
        assert "300" not in caveat and "100 questions" not in caveat

    def test_pilot_stats_path_rename_with_legacy_fallback(
            self, gp, tmp_path: Path) -> None:
        stats_dir = tmp_path / "all_results"
        stats_dir.mkdir()
        # Neither file: the renamed artifact is the target.
        assert gp._pilot_stats_path(stats_dir).name == "pilot_stats.json"
        # Frozen pre-rename archive: falls back to phase2_stats.json.
        (stats_dir / "phase2_stats.json").write_text("{}", encoding="utf-8")
        assert gp._pilot_stats_path(stats_dir).name == "phase2_stats.json"
        # Renamed artifact present: it wins.
        (stats_dir / "pilot_stats.json").write_text("{}", encoding="utf-8")
        assert gp._pilot_stats_path(stats_dir).name == "pilot_stats.json"


# ---------------------------------------------------------------------------
# token_divergence.py
# ---------------------------------------------------------------------------

class TestTokenDivergenceFence:
    def test_campaign_tree_yields_skip_and_no_artifact(self, tmp_path: Path) -> None:
        """The pilot-layout loader finds no reference arm in a campaign tree:
        the tool SKIPs (exit 0, non-fatal by design) and mints NOTHING."""
        run_dir = _make_campaign_tree(tmp_path)
        out = tmp_path / "token_divergence.json"
        proc = _run([sys.executable, str(_DIVERGE_PY),
                     "--results-dir", str(run_dir),
                     "--output", str(out)])
        assert proc.returncode == 0
        assert "SKIP" in proc.stderr
        assert not out.exists(), "no artifact may be minted from a campaign tree"


# ---------------------------------------------------------------------------
# run_phase2_stats.sh — the new pilot-tree guard
# ---------------------------------------------------------------------------

class TestRunPhase2StatsGuard:
    def _bash(self, run_root: Path, **env: str) -> subprocess.CompletedProcess[str]:
        return _run(["bash", str(_STATS_SH), str(run_root)],
                    CAGE_RUN_ROOT="", CAGE_LOG_GUARD="0", **env)

    def test_refuses_campaign_tree_before_any_mutation(self, tmp_path: Path) -> None:
        run_dir = _make_campaign_tree(tmp_path)
        # Pre-seed the exact paths the script would `rm -rf`: survival proves
        # the guard fires BEFORE the mutation (finding I11).
        marker = run_dir / "stats" / "all_results" / "marker.txt"
        marker.parent.mkdir(parents=True)
        marker.write_text("must survive", encoding="utf-8")
        proc = self._bash(run_dir)
        assert proc.returncode == 1
        assert "FATAL" in proc.stderr
        assert "CAMPAIGN" in proc.stderr
        assert "run_campaign_analysis.py" in proc.stderr
        assert marker.exists(), "guard must refuse BEFORE rm -rf"
        assert (run_dir / "manifest.json").exists()

    def test_refuses_directory_that_is_not_a_pilot_root(self, tmp_path: Path) -> None:
        stray = tmp_path / "not_a_run"
        stray.mkdir()
        proc = self._bash(stray)
        assert proc.returncode == 1
        assert "does not look like a pilot run root" in proc.stderr

    def test_accepts_pilot_tree(self, tmp_path: Path) -> None:
        root = _make_pilot_tree(tmp_path)
        proc = self._bash(root, CAGE_PILOT_GUARD_ONLY="1")
        assert proc.returncode == 0, proc.stderr
        assert "guard PASSED" in proc.stdout

    def test_driver_references_renamed_artifact(self) -> None:
        text = _STATS_SH.read_text(encoding="utf-8")
        assert "pilot_stats.json" in text
        assert "phase2_stats.json" not in text, \
            "the pilot driver must not mint the pre-rename artifact"


# ---------------------------------------------------------------------------
# _pub_tables.py — pilot stamps + _tex_escape hardening
# ---------------------------------------------------------------------------

class TestPubTablesPilotStamps:
    @pytest.fixture()
    def pt(self):
        pytest.importorskip("matplotlib")
        import _pub_tables
        return _pub_tables

    @staticmethod
    def _frame():
        import pandas as pd
        return pd.DataFrame([
            {"baseline": "no_cache", "tree": "baselines", "ttft_ms": 900.0,
             "latency_ms": 1400.0, "tpot_ms": 60.0, "cached_pct": 0.0,
             "grounding_mean": 0.98, "f1_answerable_mean": 0.55,
             "abstention_pct": 40.0, "resident_kv": 1000.0,
             "marginal_kv": 1000.0},
            {"baseline": "prefix_cache", "tree": "baselines", "ttft_ms": 120.0,
             "latency_ms": 600.0, "tpot_ms": 61.0, "cached_pct": 90.0,
             "grounding_mean": 0.97, "f1_answerable_mean": 0.54,
             "abstention_pct": 41.0, "resident_kv": 1000.0,
             "marginal_kv": 100.0},
        ])

    def test_every_emitted_tex_carries_pilot_stamp(self, pt, tmp_path: Path) -> None:
        written = pt.write_main_results_table(self._frame(), None, tmp_path)
        written += pt.write_trilemma_table(self._frame(), tmp_path)
        tex_files = [n for n in written if n.endswith(".tex")]
        assert sorted(tex_files) == ["main_results_table.tex", "trilemma_table.tex"]
        for name in tex_files:
            text = (tmp_path / name).read_text(encoding="utf-8")
            head = "\n".join(text.splitlines()[:3])
            assert "PILOT DATA -- design input only" in head, name
            # Visible note in the rendered table, not just a comment.
            assert "\\textbf{PILOT DATA}" in text, name
            assert text.count("{") == text.count("}"), f"{name}: unbalanced braces"
        for name in (n for n in written if n.endswith(".md")):
            text = (tmp_path / name).read_text(encoding="utf-8")
            assert "PILOT DATA" in text, name

    def test_speculative_md_carries_pilot_note(self, pt, tmp_path: Path) -> None:
        import pandas as pd
        df = self._frame()
        df = pd.concat([df, pd.DataFrame([
            {"baseline": "spec_qwen8b_eagle3", "tree": "speculative",
             "ttft_ms": 900.0, "latency_ms": 1300.0, "tpot_ms": 45.0}])],
            ignore_index=True)
        written = pt.write_speculative_table(df, None, None, tmp_path)
        assert written == ["speculative_summary_table.md"]
        text = (tmp_path / written[0]).read_text(encoding="utf-8")
        assert "PILOT DATA" in text

    def test_tex_escape_covers_the_full_special_set(self, pt) -> None:
        esc = pt._tex_escape
        assert esc("a_b&c%d#e") == r"a\_b\&c\%d\#e"
        assert esc("$5") == r"\$5"
        assert esc("{x}") == r"\{x\}"
        assert esc("a^b~c") == r"a\textasciicircum{}b\textasciitilde{}c"
        assert esc("\\") == r"\textbackslash{}"
        # Single pass: the braces inside \textbackslash{} are escape OUTPUT
        # and must not be re-escaped.
        assert esc("\\{") == r"\textbackslash{}\{"

    def test_table_note_n_derived_from_frame_never_300(self, pt, tmp_path: Path) -> None:
        """The 'N = ...' table note derives from the frame's n_rows column
        (I11: the hardcoded pilot 'N = 300 ... 100 questions x 3 trials'
        went stale on any other run shape); absent n_rows, the N clause is
        OMITTED, never fabricated."""
        # Frame WITHOUT n_rows: no N clause anywhere, and never the pilot 300.
        pt.write_main_results_table(self._frame(), None, tmp_path)
        for name in ("main_results_table.md", "main_results_table.tex"):
            text = (tmp_path / name).read_text(encoding="utf-8")
            assert "N = 300" not in text and "N = 300".replace(" ", "") not in text
            assert "100 questions" not in text
            assert "valid measurements per cell" not in text, name

        # Frame WITH n_rows: the derived count renders in md, tex note + caption.
        df = self._frame()
        df["n_rows"] = [8, 8]
        pt.write_main_results_table(df, None, tmp_path)
        md = (tmp_path / "main_results_table.md").read_text(encoding="utf-8")
        tex = (tmp_path / "main_results_table.tex").read_text(encoding="utf-8")
        assert "N = 8 valid measurements per cell" in md
        assert tex.count("$N = 8$ valid measurements per cell") == 2  # caption + note
        assert tex.count("{") == tex.count("}")

        # Unequal cells render as a lo-hi range (tex: en-dash outside math).
        df["n_rows"] = [6, 8]
        pt.write_main_results_table(df, None, tmp_path)
        md = (tmp_path / "main_results_table.md").read_text(encoding="utf-8")
        tex = (tmp_path / "main_results_table.tex").read_text(encoding="utf-8")
        assert "N = 6-8 valid measurements per cell" in md
        assert "$N$ = 6--8 valid measurements per cell" in tex

    def test_tex_stamp_and_escape_survive_hostile_labels(self, pt, tmp_path: Path) -> None:
        """A cell label carrying TeX specials must not unbalance the emitted
        table (the old escape chain let \\ { } $ ^ ~ through raw)."""
        df = self._frame()
        df.loc[df["baseline"] == "prefix_cache", "baseline"] = "arm_{v2}^~$"
        written = pt.write_main_results_table(df, None, tmp_path)
        text = (tmp_path / "main_results_table.tex").read_text(encoding="utf-8")
        assert "main_results_table.tex" in written
        assert text.count("{") == text.count("}")
        assert r"\textasciicircum{}" in text and r"\textasciitilde{}" in text
