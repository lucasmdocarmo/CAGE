"""S2 block F batch 3, row F-10 (2026-11-03): the regime gate for the 0.95 column.

The gate (scripts/6_experiments/regime_gate.py, stage 14) judges the loads
REGIME_GATE_LOADS names at the floor MIN_IN_REGIME_WINDOWS (one integer or the
per-load form "0.95:2 1.05:3"), keyed by (engine, arm, retriever, policy,
topology, dataset, rung, rate fraction) so B5 and B6 never merge, blocking for
the confirmatory roster CONFIRMATORY_ARMS and descriptive elsewhere. The
projection verb of run_campaign.py prints the registered windows per gated
group at $0 and refuses a floor the grid cannot meet. Both read the profile
strings through src/analysis/regime_gate_rules.py.
Spec: MyDocs/RunPod/S2_BLOCK_F_BATCH2-3_SPEC_2026-11-02.md section 5.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "6_experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import regime_gate  # noqa: E402
from src.analysis import regime_gate_rules as rules  # noqa: E402


def _load_rc():
    found = sys.modules.get("run_campaign")
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location("run_campaign", REPO_ROOT / "scripts" / "3_run" / "run_campaign.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rc = _load_rc()


# --------------------------------------------------------------------------- #
# fixtures: a run tree with cell records
# --------------------------------------------------------------------------- #


def _row_key(engine: str, *, arm: str = "retr-fresh", retriever: str = "rerank", policy: str = "none",
             family: str = "F2", r: Optional[float] = 0.25, lam: Optional[float] = 1.05) -> str:
    parts = [arm, retriever, policy, "single", engine, "llama-3.3-70b", family]
    if r is not None:
        parts.append(f"r{r:g}")
    if lam is not None:
        parts.append(f"lam{lam:g}")
    return "|".join(parts)


def _write_window(root: Path, row_key: str, dataset: str, k: int, label: str, stationarity: Optional[str]) -> None:
    d = root / "cells" / row_key / f"window_{dataset}-{k:02d}"
    d.mkdir(parents=True, exist_ok=True)
    doc: Dict[str, Any] = {"label": label}
    if stationarity is not None:
        doc["stationarity"] = {"label": stationarity}
    (d / "regime.json").write_text(json.dumps(doc), encoding="utf-8")


def _write_cell(root: Path, row_key: str, baseline: Optional[str]) -> None:
    d = root / "cells" / row_key
    d.mkdir(parents=True, exist_ok=True)
    (d / "cell.json").write_text(json.dumps({"baseline": baseline, "windows": {}}), encoding="utf-8")


def _fill(root: Path, key: str, baseline: Optional[str], labels: list, dataset: str = "qasper") -> None:
    if baseline is not None:
        _write_cell(root, key, baseline)
    for k, label in enumerate(labels, start=1):
        _write_window(root, key, dataset, k, label, "stationary")


K_B5 = "vllm|retr-fresh|dense|none|single|qasper|0.25|1.05"
K_B6 = "vllm|retr-fresh|rerank|none|single|qasper|0.25|1.05"
K_B6_095 = "vllm|retr-fresh|rerank|none|single|qasper|0.25|0.95"
LOADS_S2 = "0.95 1.05"
FLOOR_S2 = "0.95:2 1.05:3"


# --------------------------------------------------------------------------- #
# the shared grammar
# --------------------------------------------------------------------------- #


def test_the_loads_grammar_names_a_floor_or_an_exact_load() -> None:
    one = rules.parse_loads("1.0+")
    assert one == (rules.GatedLoad(1.0, True, "1.0+"),)
    two = rules.parse_loads(" 0.95  1.05 ")
    assert [(g.value, g.at_or_above, g.token) for g in two] == [(0.95, False, "0.95"), (1.05, False, "1.05")]
    assert rules.load_token_for(1.05, two) == "1.05" and rules.load_token_for(0.95, two) == "0.95"
    assert rules.load_token_for(1.0, two) is None and rules.load_token_for(None, two) is None
    assert rules.load_token_for(1.2, one) == "1.0+" and rules.load_token_for(0.9, one) is None
    assert rules.DEFAULT_LOADS == "1.0+"
    for bad in ("", "abc", "0.95 abc", "-1", "0", "1.05 1.05", "inf"):
        with pytest.raises(rules.RulesError):
            rules.parse_loads(bad)


def test_the_floor_grammar_is_one_integer_or_one_per_load() -> None:
    two = rules.parse_loads(LOADS_S2)
    assert rules.parse_floor("", two) is None and rules.parse_floor(None, two) is None
    assert rules.parse_floor("2", two) == {"0.95": 2, "1.05": 2}
    assert rules.parse_floor(3, two) == {"0.95": 3, "1.05": 3}
    assert rules.parse_floor(FLOOR_S2, two) == {"0.95": 2, "1.05": 3}
    with pytest.raises(rules.RulesError, match="not an integer"):
        rules.parse_floor("x", two)
    with pytest.raises(rules.RulesError, match=">= 1"):
        rules.parse_floor("0", two)
    with pytest.raises(rules.RulesError, match="1.05"):
        rules.parse_floor("0.95:2", two)          # a gated load without a floor
    with pytest.raises(rules.RulesError, match="0.7"):
        rules.parse_floor("0.95:2 1.05:3 0.7:1", two)   # a floor for a load not gated
    with pytest.raises(rules.RulesError, match="not an integer"):
        rules.parse_floor("0.95:two 1.05:3", two)


def test_the_roster_grammar_and_the_mode_rule() -> None:
    assert rules.parse_roster("") == frozenset()
    assert rules.parse_roster(" B1 B6  B12 ") == frozenset({"B1", "B6", "B12"})
    with pytest.raises(rules.RulesError, match="retr-fresh"):
        rules.parse_roster("B1 retr-fresh")
    empty, roster = frozenset(), frozenset({"B1", "B6"})
    assert rules.mode_for("B5", empty) == rules.BLOCKING          # no roster: every group blocks
    assert rules.mode_for("B6", roster) == rules.BLOCKING
    assert rules.mode_for("B5", roster) == rules.DESCRIPTIVE
    assert rules.mode_for(None, roster) == rules.BLOCKING          # unknown baseline: fail closed
    assert rules.GROUP_KEY_FIELDS == ("engine", "arm", "retriever", "policy", "topology", "dataset", "budget_r", "rate_frac")
    assert rules.PRESSURE_FAMILIES == ("F2", "F3")


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def test_b5_and_b6_are_separate_groups(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _fill(root, _row_key("vllm", retriever="dense"), "B5", ["IN_REGIME", "IN_REGIME", "UNPRESSURED"])
    _fill(root, _row_key("vllm", retriever="rerank"), "B6", ["IN_REGIME", "IN_REGIME", "IN_REGIME"])
    res = regime_gate.evaluate(root, 3, loads=LOADS_S2)
    assert set(res["groups"]) == {K_B5, K_B6}
    assert res["groups"][K_B5]["n_qualifying"] == 2 and res["groups"][K_B5]["baseline"] == "B5"
    assert res["groups"][K_B6]["n_qualifying"] == 3 and res["groups"][K_B6]["baseline"] == "B6"
    assert res["short"] == {K_B5: 2} and res["ok"] is False      # no roster: B5 blocks too
    assert res["groups"][K_B5]["need"] == 3 and res["groups"][K_B5]["load"] == "1.05"


def test_a_roster_arm_short_at_095_fails_by_name_and_passes_at_the_floor(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fill(root, _row_key("vllm", lam=0.95), "B6", ["IN_REGIME", "UNPRESSURED"])
    _fill(root, _row_key("vllm", lam=1.05), "B6", ["IN_REGIME", "IN_REGIME", "IN_REGIME"])
    res = regime_gate.evaluate(root, FLOOR_S2, loads=LOADS_S2, confirmatory_arms="B1 B6")
    assert res["groups"][K_B6_095]["mode"] == rules.BLOCKING and res["groups"][K_B6_095]["need"] == 2
    assert res["short"] == {K_B6_095: 1} and res["short_descriptive"] == {} and res["ok"] is False
    assert res["groups"][K_B6]["n_qualifying"] == 3 and K_B6 not in res["short"]
    code = regime_gate.main(["regime_gate.py", str(root), FLOOR_S2, "--loads", LOADS_S2, "--confirmatory-arms", "B1 B6"])
    out = capsys.readouterr().out
    assert code == 1
    assert f"[FAIL] MIN_IN_REGIME_WINDOWS={FLOOR_S2}: group(s) below it: {{'{K_B6_095}': 1}}" in out
    assert f"{K_B6_095}: qualifying=1/2 [BLOCKING, need 2]" in out
    # the second window reads the regime: at the floor the stage passes
    _write_window(root, _row_key("vllm", lam=0.95), "qasper", 2, "IN_REGIME", "unknown")
    assert regime_gate.main(["regime_gate.py", str(root), FLOOR_S2, "--loads", LOADS_S2, "--confirmatory-arms", "B1 B6"]) == 0
    out = capsys.readouterr().out
    assert f"[OK] MIN_IN_REGIME_WINDOWS={FLOOR_S2}: every blocking group meets the E12 criterion" in out


def test_a_non_roster_arm_short_prints_descriptive_and_exits_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fill(root, _row_key("vllm", retriever="dense"), "B5", ["IN_REGIME", "UNPRESSURED", "UNPRESSURED"])
    _fill(root, _row_key("vllm", retriever="rerank"), "B6", ["IN_REGIME", "IN_REGIME", "IN_REGIME"])
    # the tree carries the 1.05 column only, so that is the gated load (review H1 fails an unused one)
    res = regime_gate.evaluate(root, 3, loads="1.05", confirmatory_arms="B1 B6")
    assert res["groups"][K_B5]["mode"] == rules.DESCRIPTIVE and res["groups"][K_B6]["mode"] == rules.BLOCKING
    assert res["short"] == {} and res["short_descriptive"] == {K_B5: 1} and res["ok"] is True
    code = regime_gate.main(["regime_gate.py", str(root), "3", "--loads", "1.05", "--confirmatory-arms", "B1 B6"])
    out = capsys.readouterr().out
    assert code == 0
    assert f"{K_B5}: qualifying=1/3 [DESCRIPTIVE, need 3]" in out
    assert f"[NOTE] descriptive group(s) below the floor (no exit code): {{'{K_B5}': 1}}" in out
    assert "[OK] MIN_IN_REGIME_WINDOWS=3" in out


def test_a_cell_without_a_baseline_record_blocks_when_a_roster_is_set(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _fill(root, _row_key("vllm", retriever="dense"), None, ["UNPRESSURED", "UNPRESSURED", "UNPRESSURED"])
    res = regime_gate.evaluate(root, 3, loads=LOADS_S2, confirmatory_arms="B1 B6")
    assert res["groups"][K_B5]["baseline"] is None and res["groups"][K_B5]["mode"] == rules.BLOCKING
    assert res["short"] == {K_B5: 0} and res["ok"] is False
    # an unreadable record is the same unknown
    (root / "cells" / _row_key("vllm", retriever="dense") / "cell.json").write_text("{not json", encoding="utf-8")
    assert regime_gate.evaluate(root, 3, loads=LOADS_S2, confirmatory_arms="B1 B6")["groups"][K_B5]["baseline"] is None


def test_the_default_loads_keep_todays_rule_and_name_an_engine_without_a_gated_load(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _fill(root, _row_key("vllm", lam=0.95), "B6", ["IN_REGIME", "IN_REGIME"])
    _fill(root, _row_key("sglang", lam=1.0), "B6", ["IN_REGIME", "IN_REGIME", "IN_REGIME"])
    res = regime_gate.evaluate(root, 2)
    assert res["loads"] == "1.0+"
    assert set(res["groups"]) == {"sglang|retr-fresh|rerank|none|single|qasper|0.25|1.0"}
    assert res["engines_without_a_gated_load"] == ["vllm"] and res["ok"] is False
    assert regime_gate.evaluate(root, 2, loads=LOADS_S2)["engines_without_a_gated_load"] == ["sglang"]
    assert regime_gate.evaluate(root, None)["ok"] is True       # report only: no engine check either


def test_the_cli_refuses_a_bad_load_or_floor_form(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "run"
    _fill(root, _row_key("vllm"), "B6", ["IN_REGIME"])
    assert regime_gate.main(["regime_gate.py", str(root), "2", "--loads", "abc"]) == 2
    assert "REGIME_GATE_LOADS" in capsys.readouterr().err
    assert regime_gate.main(["regime_gate.py", str(root), "0.95:2", "--loads", LOADS_S2]) == 2
    assert "1.05" in capsys.readouterr().err
    assert regime_gate.main(["regime_gate.py", str(root), "2", "--confirmatory-arms", "B1 gold-fresh"]) == 2
    assert "gold-fresh" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# the projection verb
# --------------------------------------------------------------------------- #


def test_the_projection_counts_the_registered_windows_per_gated_group_on_both_landings() -> None:
    s2a = rc.regime_gate_projection("s2a", loads=LOADS_S2)
    sizes = {}
    for key, g in s2a["groups"].items():
        sizes.setdefault(g["load"], []).append(g["registered"])
    assert sorted(sizes) == ["0.95", "1.05"]
    assert len(sizes["0.95"]) == 30 and set(sizes["0.95"]) == {2}
    assert len(sizes["1.05"]) == 42 and set(sizes["1.05"]) == {3}
    for engine in ("vllm", "sglang"):
        assert s2a["groups"][K_B5.replace("vllm", engine)]["baseline"] == "B5"
        assert s2a["groups"][K_B6.replace("vllm", engine)]["baseline"] == "B6"
    assert s2a["unresolved_load_cells"] == 0 and s2a["need"] == "" and s2a["short_descriptive"] == {}
    assert all(g["mode"] == rules.BLOCKING and g["need"] is None for g in s2a["groups"].values())
    cd = rc.regime_gate_projection("cd-act1", loads=LOADS_S2, need=FLOOR_S2)
    by_load = {}
    for g in cd["groups"].values():
        by_load.setdefault(g["load"], []).append((g["registered"], g["need"]))
    assert by_load == {"0.95": [(2, 2)] * 33, "1.05": [(3, 3)] * 33}
    assert all(k.split("|")[4] == "single" for k in cd["groups"])    # the pd block is family DIST, outside
    # the S1 grid under the default: no gated load, and its frontier cells resolve only at plan time
    s1 = rc.regime_gate_projection("s1")
    assert s1["groups"] == {} and s1["unresolved_load_cells"] == 16


def test_the_projection_refuses_a_floor_above_a_blocking_groups_registered_windows() -> None:
    with pytest.raises(rc.PlanError, match=r"MIN_IN_REGIME_WINDOWS=6 exceeds the 2 registered windows of .*BLOCKING.*by construction"):
        rc.regime_gate_projection("s2a", loads=LOADS_S2, need="6")
    with pytest.raises(rc.PlanError, match="exceeds the 3 registered windows of"):
        rc.regime_gate_projection("s2a", need="6")            # the default 1.0+ gates the 1.05 column only
    assert rc.regime_gate_projection("s2a", loads=LOADS_S2, need=FLOOR_S2)["short_descriptive"] == {}
    # a floor only descriptive groups miss is reported, not refused: B9 has no 0.95 cell on s2a
    doc = rc.regime_gate_projection("s2a", loads=LOADS_S2, need="0.95:3 1.05:3", confirmatory_arms="B9")
    assert len(doc["short_descriptive"]) == 30 and set(doc["short_descriptive"].values()) == {2}
    assert sum(g["mode"] == rules.BLOCKING for g in doc["groups"].values()) == 2    # B9 on two engines


def test_the_projection_refuses_an_unknown_or_absent_roster_arm() -> None:
    with pytest.raises(rc.PlanError, match="B99"):
        rc.regime_gate_projection("s2a", loads=LOADS_S2, confirmatory_arms="B1 B99")
    with pytest.raises(rc.PlanError, match="B8"):
        rc.regime_gate_projection("s2a", loads=LOADS_S2, confirmatory_arms="B8")   # no store arm on the 70B
    with pytest.raises(rc.PlanError, match="abc"):
        rc.regime_gate_projection("s2a", loads="abc")


def test_the_projection_cli_prints_one_line_per_group(capsys: pytest.CaptureFixture[str]) -> None:
    code = rc.main(["regime-gate-projection", "--session", "cd-act1", "--loads", LOADS_S2, "--min", FLOOR_S2,
                    "--confirmatory-arms", "B1 B3 B4 B6 B7 B8"])
    out = capsys.readouterr().out
    assert code == 0
    head = out.splitlines()[0]
    assert head.startswith("regime-gate projection session=cd-act1 loads='0.95 1.05' floor='0.95:2 1.05:3' roster=B1 B3 B4 B6 B7 B8")
    assert "gated groups=66 blocking=66 descriptive=0 unresolved_load_cells=0" in head
    assert "  vllm|gold-fresh|none|none|single|musique|1.0|0.95: registered=2 need=2 BLOCKING" in out
    assert "  sglang|corpus-reuse|none|none|single|musique|0.25|1.05: registered=3 need=3 BLOCKING" in out
    assert out.count("BLOCKING") == 66     # one per group line; the header counts in lower case
    code = rc.main(["regime-gate-projection", "--session", "s2a", "--loads", LOADS_S2, "--min", "6"])
    assert code == 2 and "REFUSED: MIN_IN_REGIME_WINDOWS=6 exceeds the 2 registered windows of" in capsys.readouterr().err   # the planner refuses with 2


# --------------------------------------------------------------------------- #
# the independent review of 2026-11-03: H1, M1, L2 and the L8 gaps
# --------------------------------------------------------------------------- #


def test_a_gated_load_no_window_carries_fails_the_gate_when_a_floor_is_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Review H1: a mistyped token (0.9 for 0.95) gated no window, the group set
    had no shortfall, and the gate passed with that column never judged."""
    root = tmp_path / "run"
    _fill(root, _row_key("vllm", lam=0.95), "B6", ["IN_REGIME", "IN_REGIME"])
    _fill(root, _row_key("vllm", lam=1.05), "B6", ["IN_REGIME", "IN_REGIME", "IN_REGIME"])
    res = regime_gate.evaluate(root, "0.9:2 1.05:3", loads="0.9 1.05")
    assert res["loads_without_a_group"] == ["0.9"] and res["short"] == {} and res["ok"] is False
    assert regime_gate.evaluate(root, "2", loads="1.1")["loads_without_a_group"] == ["1.1"]
    assert regime_gate.evaluate(root, FLOOR_S2, loads=LOADS_S2)["loads_without_a_group"] == []
    assert regime_gate.evaluate(root, None, loads="1.1")["ok"] is True       # report only: nothing is judged
    code = regime_gate.main(["regime_gate.py", str(root), "0.9:2 1.05:3", "--loads", "0.9 1.05"])
    out = capsys.readouterr().out
    assert code == 1
    assert "[FAIL] gated load(s) with no window (0.9 1.05): ['0.9']" in out


def test_the_projection_refuses_a_gated_load_no_registered_cell_carries(capsys: pytest.CaptureFixture[str]) -> None:
    """Review H1, the plan-time side: with a floor set and every load resolved, a
    token no registered pressure cell carries refuses at stage 0 by name; without
    a floor, or while frontier loads resolve at plan time (S1), it is a NOTE."""
    with pytest.raises(rc.PlanError, match=r"REGIME_GATE_LOADS token\(s\) \['0.9'\] gate no registered pressure cell"):
        rc.regime_gate_projection("cd-act1", loads="0.9 1.05", need="0.9:2 1.05:3")
    with pytest.raises(rc.PlanError, match=r"\['1.1'\]"):
        rc.regime_gate_projection("cd-act1", loads="1.1", need="2")
    assert rc.regime_gate_projection("cd-act1", loads="0.9 1.05")["loads_without_a_group"] == ["0.9"]
    s1 = rc.regime_gate_projection("s1", need="6")
    assert s1["loads_without_a_group"] == ["1.0+"] and s1["unresolved_load_cells"] == 16
    assert rc.regime_gate_projection("s2a", loads=LOADS_S2, need=FLOOR_S2)["loads_without_a_group"] == []
    assert rc.main(["regime-gate-projection", "--session", "s1", "--min", "6"]) == 0
    assert "NOTE: gated load(s) with no registered cell: ['1.0+']" in capsys.readouterr().out


def test_a_cell_record_whose_baseline_is_not_an_id_blocks_under_a_roster(tmp_path: Path) -> None:
    """Review M1: "" or an arm name in cell.json's baseline read as a known id
    outside the roster, so the group went DESCRIPTIVE (fail open)."""
    root = tmp_path / "run"
    key = _row_key("vllm", retriever="dense")
    _fill(root, key, "B5", ["IN_REGIME"])
    for raw in ("", "retr-fresh", 5, None):
        (root / "cells" / key / "cell.json").write_text(json.dumps({"baseline": raw}), encoding="utf-8")
        g = regime_gate.evaluate(root, 2, loads=LOADS_S2, confirmatory_arms="B1 B6")["groups"][K_B5]
        assert g["baseline"] is None and g["mode"] == rules.BLOCKING, raw
    (root / "cells" / key / "cell.json").write_text(json.dumps({"baseline": "B5"}), encoding="utf-8")
    assert regime_gate.evaluate(root, 2, loads=LOADS_S2, confirmatory_arms="B1 B6")["groups"][K_B5]["mode"] == rules.DESCRIPTIVE


def test_the_loads_grammar_refuses_a_shadowed_or_equal_token() -> None:
    """Review L2: after "1.0+" the token "1.05" can never match (the floor token
    takes it first), and "1.0 1.00" names one rate twice; the order "1.05 1.0+" is read."""
    with pytest.raises(rules.RulesError, match="shadowed"):
        rules.parse_loads("1.0+ 1.05")
    with pytest.raises(rules.RulesError, match="shadowed"):
        rules.parse_loads("0.95+ 1.0+")
    with pytest.raises(rules.RulesError, match="twice"):
        rules.parse_loads("1.0 1.00")
    assert [g.token for g in rules.parse_loads("1.05 1.0+")] == ["1.05", "1.0+"]
    assert [g.token for g in rules.parse_loads("1.0+ 0.95")] == ["1.0+", "0.95"]


def test_the_projection_on_a_rehearsal_grid_counts_one_window_per_group() -> None:
    """Review L8: the rehearsal grid (REHEARSAL_N with the S0 manifests) registers
    one window per cell (the B12 rungs and the RULER tasks pool into 2 and 4), so
    a floor above 1 refuses at stage 0 naming a one-window group."""
    manifests = ("squad_v2=data/manifests/squad_v2_50x3_seed42.json", "qasper=data/manifests/qasper_100x3_seed42.json")
    doc = rc.regime_gate_projection("a", rehearsal_n=50, query_manifests=manifests)
    sizes = sorted(g["registered"] for g in doc["groups"].values())
    assert len(doc["groups"]) == 37 and sizes == [1] * 31 + [2] * 2 + [4] * 4      # measured 2026-11-03
    with pytest.raises(rc.PlanError, match="MIN_IN_REGIME_WINDOWS=2 exceeds the 1 registered windows of"):
        rc.regime_gate_projection("a", need="2", rehearsal_n=50, query_manifests=manifests)


def test_the_gate_and_the_projection_key_every_s2a_pressure_cell_the_same_way(tmp_path: Path) -> None:
    """Review L8: a landing built from the registered row keys of s2a (one window
    per cell) is grouped by the gate under the keys the projection printed."""
    root = tmp_path / "run"
    grid = rc.get_session_grid("s2a")
    cells = [c for c in rc.enumerate_cells(grid) if c.spec.family in rules.PRESSURE_FAMILIES and c.blocked_on is None]
    assert len(cells) == 72
    for c in cells:
        key = c.spec.to_row_key()
        _write_cell(root, key, c.baseline_id)
        _write_window(root, key, c.dataset, 1, "IN_REGIME", "stationary")
    gate = regime_gate.evaluate(root, None, loads=LOADS_S2)["groups"]
    proj = rc.regime_gate_projection("s2a", loads=LOADS_S2)["groups"]
    assert set(gate) == set(proj) and len(gate) == 72
    assert all(gate[k]["baseline"] == proj[k]["baseline"] and gate[k]["load"] == proj[k]["load"] for k in gate)
