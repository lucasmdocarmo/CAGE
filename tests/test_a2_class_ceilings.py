"""ADR-0177 (owner 2026-10-25, option A2): a measured fill ceiling for every
class the plan interpolates, without moving any rate.

- ``calibrate-rungs --ceilings-only`` ladders the classes between the two
  anchors at their pressure rungs <= KV_CEILING_MAX_R and writes a
  ``cage-class-ceilings-v1`` artifact;
- ``plan --class-ceilings`` pins those ceilings on the interpolated classes,
  every rate and argv unchanged, and ``ceiling_plan_diff`` refuses a re-plan
  that moved anything else;
- ``relabel-regime`` re-judges the windows already written, through the
  session's journaled write, keeping the writer's label under "registered";
- ``drift-window`` reruns one completed window into its own root, record only.

Pure of RunPod: the runner and the launchers are in-process fakes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_batch_b_pressure import FakeWorld, _calibrate, _write_doc  # noqa: E402
from test_run_campaign import _ANCHOR_SEQ_TOKENS, _tiny_grid, rc  # noqa: E402
from test_experiment_master import _master, world  # noqa: E402,F401  (fixture re-export)
from src.orchestration.campaign_session import read_write_time_journal  # noqa: E402

CEILING = 0.86


class QueuedWorld(FakeWorld):
    """FakeWorld whose ladder windows record the regime inputs a ceiling is
    read from: a window above the class capacity queues (share 0.9) at a
    queued occupancy of CEILING; one below it does not queue."""

    def __call__(self, argv, env) -> int:
        rc_ = super().__call__(argv, env)
        argv = list(argv)
        if "--campaign-root" in argv and "--rate" in argv and rc_ == 0:
            rate = float(argv[argv.index("--rate") + 1])
            cap = self.cap_by_arm.get(env.get("CAGE_CELL_ARM", ""), self.cap)
            root = Path(argv[argv.index("--campaign-root") + 1])
            for regime in root.rglob("regime.json"):
                regime.write_text(json.dumps({
                    "label": self.regime,
                    "attainment": min(1.0, cap / rate),
                    "kv_ceiling": None,
                    "inputs": {
                        "rho_kv_time_avg": 0.8, "queue_waiting_share": 0.9 if rate > cap else 0.1,
                        "rho_kv_queued_median": CEILING if rate > cap else None,
                    },
                }), encoding="utf-8")
        return rc_


def _mid_grid():
    return _tiny_grid(
        f1_baselines=("B1",), f2_baselines=("B1", "B6", "B11"), f2_budgets=(1.0, 0.5), f2_rates=(0.85,),
    )


def _mid(grid) -> tuple:
    cell = next(c for c in rc.enumerate_cells(grid) if c.baseline_id == "B6" and c.spec.family == "F2")
    return cell.spec.arm, rc.demand_seq_tokens(grid, cell.spec)


# ---------------------------------------------------------------------------
# 1. the ceiling-only ladder
# ---------------------------------------------------------------------------


def test_ceilings_only_ladders_only_the_interpolated_classes(tmp_path):
    grid = _mid_grid()
    arm, seq = _mid(grid)
    world_ = QueuedWorld(cap=6.0, cap_by_arm={"retr-trunc": 24.0, arm: 12.0})
    doc = _calibrate(tmp_path, world_, grid, ceilings_only=True)
    assert doc["schema"] == rc.CLASS_CEILINGS_SCHEMA and doc["complete"] is True
    assert list(doc["classes"]) == [str(seq)]
    cls = doc["classes"][str(seq)]
    assert cls["role"] == rc.CLASS_CEILING_ROLE and cls["arm"] == arm
    for key in ("1", "0.5"):
        assert cls["rungs"][key]["kv_ceiling"] == pytest.approx(CEILING)
        assert cls["rungs"][key]["kv_ceiling_windows"] >= 1
    # no anchor, smallest-class or RULER ladder ran: every window is the mid class's
    roots = [Path(a[a.index("--campaign-root") + 1]).name for a, _e in world_.runner_calls()]
    assert roots and all(name.startswith(f"cal-vllm-c{seq}-") for name in roots)
    assert all(e.get("CAGE_CELL_ARM") == arm for _a, e in world_.runner_calls())


def test_ceilings_only_refuses_the_other_ladder_modes(tmp_path):
    grid = _mid_grid()
    for kw in ({"all_classes": True}, {"anchor_only": True}):
        with pytest.raises(rc.PlanError, match="ceilings-only"):
            _calibrate(tmp_path / next(iter(kw)), QueuedWorld(cap=6.0), grid, ceilings_only=True, **kw)


def test_a_grid_with_no_interpolated_class_refuses(tmp_path):
    grid = _tiny_grid(f1_baselines=("B1",), f2_baselines=("B1", "B11"), f2_budgets=(1.0, 0.5), f2_rates=(0.85,))
    with pytest.raises(rc.PlanError, match="interpolates no class"):
        _calibrate(tmp_path, QueuedWorld(cap=6.0), grid, ceilings_only=True)


# ---------------------------------------------------------------------------
# 2. the loader, the registration and the pinned source
# ---------------------------------------------------------------------------


def _rung_and_ceilings(tmp_path: Path):
    grid = _mid_grid()
    arm, seq = _mid(grid)
    caps = {"retr-trunc": 24.0, arm: 12.0}
    rungs = _calibrate(tmp_path / "rungs", QueuedWorld(cap=6.0, cap_by_arm=caps), grid)
    ceil = _calibrate(tmp_path / "ceil", QueuedWorld(cap=6.0, cap_by_arm=caps), grid, ceilings_only=True)
    return grid, seq, _write_doc(tmp_path, rungs, "rungs_vllm.json"), _write_doc(tmp_path, ceil, "ceil_vllm.json")


def test_the_pinned_ceiling_moves_no_rate(tmp_path):
    grid, seq, rung_path, ceil_path = _rung_and_ceilings(tmp_path)
    plain = rc.load_rung_calibration(rung_path)
    table, _header = rc._register_class_ceilings(grid, {"vllm": plain}, {"vllm": ceil_path}, _ANCHOR_SEQ_TOKENS)
    pinned = table["vllm"]
    for r in (1.0, 0.5):
        lam0, basis0, src0 = rc.rung_lambda_for_class(plain, r, seq, _ANCHOR_SEQ_TOKENS)
        lam1, basis1, src1 = rc.rung_lambda_for_class(pinned, r, seq, _ANCHOR_SEQ_TOKENS)
        assert (lam1, basis1) == (lam0, basis0) and basis1 == rc.LAMBDA_BASIS_INTERPOLATED
        assert src0["kv_ceiling"] is None and src1["kv_ceiling"] == pytest.approx(CEILING)
        assert src1["kv_ceiling_source"]["sha256"] == hashlib.sha256(ceil_path.read_bytes()).hexdigest()
        # the anchor keeps its own measured ceiling, untouched
        assert rc.rung_lambda_for_class(pinned, r, _ANCHOR_SEQ_TOKENS, _ANCHOR_SEQ_TOKENS) == \
            rc.rung_lambda_for_class(plain, r, _ANCHOR_SEQ_TOKENS, _ANCHOR_SEQ_TOKENS)


def test_registration_refuses_a_laddered_class_and_a_foreign_engine(tmp_path):
    grid, seq, rung_path, ceil_path = _rung_and_ceilings(tmp_path)
    plain = rc.load_rung_calibration(rung_path)
    doc = json.loads(ceil_path.read_text(encoding="utf-8"))
    doc["classes"][str(_ANCHOR_SEQ_TOKENS)] = dict(doc["classes"][str(seq)], seq_tokens=_ANCHOR_SEQ_TOKENS)
    bad = _write_doc(tmp_path, doc, "ceil_anchor.json")
    with pytest.raises(rc.PlanError, match="laddered by the rung artifact"):
        rc._register_class_ceilings(grid, {"vllm": plain}, {"vllm": bad}, _ANCHOR_SEQ_TOKENS)
    other = dict(json.loads(ceil_path.read_text(encoding="utf-8")), engine="sglang")
    with pytest.raises(rc.PlanError, match="is for engine"):
        rc._register_class_ceilings(grid, {"vllm": plain}, {"vllm": _write_doc(tmp_path, other, "ceil_sgl.json")}, _ANCHOR_SEQ_TOKENS)


@pytest.mark.parametrize("edit, match", [
    ({"schema": "cage-rung-calibration-v2"}, "schema"),
    ({"complete": False}, "INCOMPLETE"),
    ({"confirmatory": True}, "confirmatory"),
])
def test_the_loader_fails_closed(tmp_path, edit, match):
    _grid, _seq, _rung_path, ceil_path = _rung_and_ceilings(tmp_path)
    doc = dict(json.loads(ceil_path.read_text(encoding="utf-8")), **edit)
    with pytest.raises(rc.PlanError, match=match):
        rc.load_class_ceilings(_write_doc(tmp_path, doc, "bad.json"))


# ---------------------------------------------------------------------------
# 3. the diff gate
# ---------------------------------------------------------------------------


def _step(i: int, pin: Any = None, **extra: Any) -> Dict[str, Any]:
    env = {"CAGE_CELL_ARM": "retr-comp"}
    source = {"artifact_sha256": "a", "kv_ceiling": None}
    if pin is not None:
        env[rc.KV_CEILING_ENV] = pin
        # getattr: the parametrize below is built at collection, so the falsification
        # run on the pre-ADR-0177 code collects and each test fails on its own
        source = {"artifact_sha256": "a", "kv_ceiling": float(pin),
                  "kv_ceiling_source": {"adr": getattr(rc, "CLASS_CEILINGS_ADR", "ADR-0177")}}
    return {"kind": "cell", "index": i, "row_key": f"k{i}", "argv": ["--rate", "1"], "env": env,
            "lambda_star_source": source, **extra}


def _plan(steps: List[Dict[str, Any]], **header: Any) -> Dict[str, Any]:
    return {"schema": rc.PLAN_SCHEMA, "generated_at_utc": "t0", "counts": {"cells": len(steps)}, "steps": steps, **header}


def test_the_diff_gate_passes_ceiling_fields_alone():
    old = _plan([_step(0), _step(1), _step(2, pin="0.93")])
    new = _plan([_step(0, pin="0.836"), _step(1), _step(2, pin="0.93")], generated_at_utc="t1",
                class_ceilings={"adr": rc.CLASS_CEILINGS_ADR})
    assert rc.ceiling_plan_diff(old, new) == ([], 1)


@pytest.mark.parametrize("new_steps, header, match", [
    ([_step(0, pin="0.8"), _step(1), _step(2, pin="0.91")], {}, "changes a measured ceiling"),
    ([_step(0), _step(1, argv=["--rate", "2"]), _step(2, pin="0.93")], {}, "differs outside the ceiling fields"),
    ([_step(0), _step(1)], {}, "steps"),
    ([_step(0), _step(1), _step(2, pin="0.93")], {"counts": {"cells": 99}}, "header key 'counts'"),
])
def test_the_diff_gate_refuses_anything_else(new_steps, header, match):
    old = _plan([_step(0), _step(1), _step(2, pin="0.93")])
    problems, _changed = rc.ceiling_plan_diff(old, _plan(new_steps, **header))
    assert any(match in p for p in problems), problems


# ---------------------------------------------------------------------------
# 4. the re-label
# ---------------------------------------------------------------------------

_INPUTS = {"rho_kv_time_avg": 0.83, "queue_waiting_share": 0.964, "rho_kv_queued_median": 0.851, "scarcity_events": 0}


def _regime(label: str = "UNPRESSURED", kv_ceiling: Any = None) -> Dict[str, Any]:
    return {"telemetry_ok": True, "inputs": dict(_INPUTS), "attainment": 0.97, "label": label,
            "kv_ceiling": kv_ceiling, "clause_a_rule": "rho_kv_queued_median >= RHO_KV_MIN", "refusal_reason": None}


def _journaled(root: Path, path: Path, doc: Dict[str, Any]) -> None:
    from src.orchestration.campaign_session import append_write_time_hashes

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    append_write_time_hashes(root, [path])


def _relabel_world(tmp_path: Path, pin: str):
    root = tmp_path / "results" / "camp1" / "s1" / "20261024-000946-s1-qwen3-14b"
    cell = {"kind": "cell", "family": "F2", "row_key": "retr-comp|cell", "dataset": "qasper", "windows": 1,
            "trial_index": 1, "window_ordinal_base": 0, "blocked_on": None, "env": {rc.KV_CEILING_ENV: pin}}
    dry = {"kind": "dry_window", "row_key": "retr-comp|dry", "dataset": "qasper", "blocked_on": None,
           "env": {rc.KV_CEILING_ENV: pin}, "serving": {"engine": "sglang", "prefix_mode": "OFF", "topology": "single"},
           "demand_class": {"seq_tokens": 702}, "offered_rate_rps": 21.0, "lambda_star_rps": 17.5, "rate_frac": 1.2,
           "duration_s": 75.0, "lambda_star_source": {"artifact_sha256": "a"}, "budget_plan": {"budget_bytes_total": 1}}
    plan = {"rung_calibration": {"artifacts": {}}, "steps": [cell, dry]}
    dry_root = rc.dry_window_root(root, plan)
    cw = root / "cells" / cell["row_key"] / "window_qasper-01" / "regime.json"
    dw = dry_root / "cells" / dry["row_key"] / "window_qasper-01" / "regime.json"
    _journaled(root, cw, _regime())
    _journaled(dry_root, dw, _regime())
    rc._write_dry_sidecar(dry_root, dry, 1, "UNPRESSURED", 0)
    return root, dry_root, plan, cw, dw, dry


def test_relabel_rejudges_journals_and_keeps_the_registered_label(tmp_path):
    root, dry_root, plan, cw, dw, dry = _relabel_world(tmp_path, "0.88")
    assert not rc._segment_step_complete(root, dry_root, dry)  # a failed dry window is rerun today
    report = rc.relabel_windows(plan, root, "sha")
    assert report["relabeled"] == 2 and len(report["flips"]) == 2
    for path, run_root in ((cw, root), (dw, dry_root)):
        doc = json.loads(path.read_text(encoding="utf-8"))
        # 0.851 >= 0.95 x 0.88 = 0.836: the class's own bar, where 0.90 refused it
        assert doc["label"] == "IN_REGIME" and doc["kv_ceiling"] == pytest.approx(0.88)
        assert doc["clause_a_rule"] == "rho_kv_queued_median >= RHO_KV_REL_MIN x kv_ceiling"
        assert doc["registered"] == {"label": "UNPRESSURED", "kv_ceiling": None,
                                     "clause_a_rule": "rho_kv_queued_median >= RHO_KV_MIN"}
        assert doc["inputs"] == _INPUTS and doc["relabel"]["adr"] == rc.CLASS_CEILINGS_ADR
        rel = path.relative_to(run_root).as_posix()
        assert read_write_time_journal(run_root)[rel] == hashlib.sha256(path.read_bytes()).hexdigest()
    side = rc._read_dry_sidecar(dry_root, dry, 1)
    assert (side["label"], side["verdict"], side["registered_label"]) == ("IN_REGIME", "ok", "UNPRESSURED")
    # the reuse rule now reuses the re-judged attempt: no fresh dry window
    assert rc._segment_step_complete(root, dry_root, dry)
    # idempotent: a second pass changes nothing, byte for byte
    before = cw.read_bytes()
    again = rc.relabel_windows(plan, root, "sha")
    assert (again["relabeled"], again["unchanged"]) == (0, 2) and cw.read_bytes() == before


def test_relabel_under_a_ceiling_that_still_refuses_keeps_the_skip(tmp_path):
    root, dry_root, plan, cw, _dw, dry = _relabel_world(tmp_path, "0.95")
    report = rc.relabel_windows(plan, root, "sha")
    # 0.851 < 0.95 x 0.95 = 0.9025: re-judged, still UNPRESSURED, no flip
    assert report["relabeled"] == 2 and report["flips"] == []
    assert json.loads(cw.read_text(encoding="utf-8"))["label"] == "UNPRESSURED"
    assert not rc._segment_step_complete(root, dry_root, dry)


def test_relabel_leaves_an_unjudgeable_window_as_written(tmp_path):
    root, _dry_root, plan, cw, _dw, _dry = _relabel_world(tmp_path, "0.88")
    unknown = {"telemetry_ok": False, "inputs": None, "label": "UNKNOWN_TELEMETRY", "attainment": 0.9, "kv_ceiling": None}
    _journaled(root, cw, unknown)
    before = cw.read_bytes()
    report = rc.relabel_windows(plan, root, "sha")
    assert report["not_judged"] == 1 and cw.read_bytes() == before


_S0_TREE = Path("/Users/lucasmariano/CAGE/experiments/S0/2026-10-14_0016/run")


@pytest.mark.skipif(not _S0_TREE.is_dir(), reason="the S0 attempt 5 landing is not on this machine")
def test_the_rejudgment_reproduces_the_writer_on_every_s0_window():
    """Falsification of the re-label rule on real windows: under the ceiling
    each window was written with, classify_regime on its recorded inputs and
    attainment returns the label the writer recorded, for every window."""
    from src.analysis.goodput import classify_regime

    checked = 0
    for path in sorted(_S0_TREE.rglob("regime.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        inputs = doc.get("inputs")
        if not doc.get("telemetry_ok") or not isinstance(inputs, dict) or doc.get("attainment") is None:
            continue
        label = classify_regime(
            rho_kv=inputs["rho_kv_time_avg"], attainment=float(doc["attainment"]),
            queue_waiting_share=inputs["queue_waiting_share"], scarcity_events=inputs.get("scarcity_events"),
            rho_kv_queued_median=inputs.get("rho_kv_queued_median"), kv_ceiling=doc.get("kv_ceiling"),
        )
        assert label == doc["label"], path
        checked += 1
    assert checked > 50


# ---------------------------------------------------------------------------
# 5. the drift window
# ---------------------------------------------------------------------------


def _drift_world(tmp_path: Path):
    root = tmp_path / "results" / "camp1" / "s1" / "20261024-000946-s1-qwen3-14b"
    relaunch = {"kind": "relaunch", "launcher_key": "sglang", "argv": ["launch", "restart", "m"],
                "stop_argv": ["launch", "stop"], "env": {}}
    cell = {"kind": "cell", "family": "F2", "row_key": "gold-fresh|cell", "dataset": "qasper", "windows": 1,
            "trial_index": 1, "window_ordinal_base": 0, "blocked_on": None, "demand_class": {"seq_tokens": 4779},
            "argv": ["runner", "--rate", "3"], "env": {}}
    plan = {"demand_classes": {"anchor_seq_tokens": 4779}, "steps": [relaunch, cell]}
    original = root / "cells" / cell["row_key"] / "window_qasper-01"
    original.mkdir(parents=True)
    (original / "metrics.json").write_text("{}", encoding="utf-8")
    (original / "regime.json").write_text(json.dumps(_regime("IN_REGIME", 0.93)), encoding="utf-8")
    (original / "requests.jsonl").write_text(json.dumps({"ok": True, "ttft_from_send_ms": 500.0}) + "\n", encoding="utf-8")
    return root, plan


def test_the_drift_window_reruns_one_window_into_its_own_root(tmp_path):
    root, plan = _drift_world(tmp_path)
    calls: List[List[str]] = []

    def fake(argv, env):
        argv = list(argv)
        calls.append(argv)
        if "--campaign-root" in argv:
            w = Path(argv[argv.index("--campaign-root") + 1]) / "cells" / "gold-fresh|cell" / "window_qasper-01"
            w.mkdir(parents=True)
            (w / "metrics.json").write_text("{}", encoding="utf-8")
            (w / "regime.json").write_text(json.dumps(_regime("IN_REGIME", 0.93)), encoding="utf-8")
        return 0

    report = rc.run_drift_window(plan, root, exec_fn=fake)
    drift = rc.drift_root(root)
    assert drift.name.endswith(rc.DRIFT_ROOT_SUFFIX) and drift.parent == root.parent
    # review 2026-10-25 L8: the plan's launchers stopped first (the run's clean room)
    assert [c[1] if c[0] == "launch" else "runner" for c in calls] == ["stop", "restart", "runner", "stop"]
    assert calls[2][-2:] == ["--campaign-root", str(drift)] and report["stop_exit"] == 0
    assert report["original"]["label"] == report["rerun"]["label"] == "IN_REGIME"
    assert report["original"]["ttft_from_send_ms_median"] == 500.0
    assert (drift / "drift_report.json").is_file()
    assert not (root / "cells" / "gold-fresh|cell" / "window_qasper-02").exists()  # the campaign tree is untouched
    calls.clear()
    rc.run_drift_window(plan, root, exec_fn=fake)
    assert calls == []  # a drift window already on disk is not run again


def test_no_completed_anchor_window_refuses_the_drift(tmp_path):
    root, plan = _drift_world(tmp_path)
    (root / "cells" / "gold-fresh|cell" / "window_qasper-01" / "metrics.json").unlink()
    with pytest.raises(rc.RunError, match="no completed anchor-class"):
        rc.drift_step(plan, root)


# ---------------------------------------------------------------------------
# 6. the master: the switch adds the ladders, the gate, the re-label and the drift
# ---------------------------------------------------------------------------


def test_the_master_plan_stage_carries_the_ceiling_steps_only_under_the_switch(world):  # noqa: F811
    off = _master(world, "--plan", "--only", "plan")
    assert off.returncode == 0, off.stdout + off.stderr
    for marker in ("--ceilings-only", "--class-ceilings", "plan-diff", "relabel-regime", "drift-window"):
        assert marker not in off.stdout, marker
    on_profile = world["tmp"] / "S1_ceilings.env"
    on_profile.write_text(world["profile"].read_text(encoding="utf-8") + "CLASS_CEILINGS=1\n", encoding="utf-8")
    world["profile"] = on_profile
    on = _master(world, "--plan", "--only", "plan")
    assert on.returncode == 0, on.stdout + on.stderr
    for marker in ("--ceilings-only", "--class-ceilings", "plan-diff", "relabel-regime", "drift-window"):
        assert marker in on.stdout, marker
    assert on.stdout.index("--ceilings-only") < on.stdout.index("--class-ceilings") < on.stdout.index("plan-diff")


# ---------------------------------------------------------------------------
# 7. the fixes of the independent review (2026-10-25)
# ---------------------------------------------------------------------------


def test_a_ceiling_ladder_with_a_failed_relaunch_fails_its_job_m5(tmp_path, monkeypatch, capsys):
    """M5: RELAUNCH_FAILED and PROBE_FAILED are infrastructure failures, never a
    missing ceiling: the CLI exits 1 (the stage fails), the artifact written."""
    doc = {"classes": {"702": {"arm": "retr-comp", "rungs": {"0.25": {"label": rc.RUNG_LABEL_RELAUNCH_FAILED}}}},
           "complete": True}
    monkeypatch.setattr(rc, "calibrate_rungs", lambda *a, **k: doc)
    monkeypatch.setattr(rc, "load_floor_table", lambda p: None)
    out = tmp_path / "ceil.json"
    rc_ = rc.main(["calibrate-rungs", "--ceilings-only", "--session", "s1", "--engine", "sglang",
                   "--floor-table", "ft", "--calibration", "cal", "--out", str(out)])
    assert rc_ == 1 and out.is_file() and "CEILING LADDER FAILED" in capsys.readouterr().err
    doc["classes"]["702"]["rungs"]["0.25"]["label"] = "LADDER_EXHAUSTED"  # a measurement: not a failure
    assert rc.main(["calibrate-rungs", "--ceilings-only", "--session", "s1", "--engine", "sglang",
                    "--floor-table", "ft", "--calibration", "cal", "--out", str(out)]) == 0


def test_a_ceiling_that_would_fail_a_passed_dry_window_refuses_m6(tmp_path):
    """M6: a pin that tightens the bar enough to turn a passed dry window into a
    failure refuses the whole re-label before any write; the owner decides."""
    root, dry_root, plan, cw, dw, dry = _relabel_world(tmp_path, "0.98")
    passed = dict(_regime("IN_REGIME"), inputs=dict(_INPUTS, rho_kv_queued_median=0.92))
    _journaled(dry_root, dw, passed)
    rc._write_dry_sidecar(dry_root, dry, 1, "IN_REGIME", 0)
    before = (cw.read_bytes(), dw.read_bytes())
    with pytest.raises(rc.RunError, match="would fail 1 dry window"):
        rc.relabel_windows(plan, root, "sha")  # 0.92 < 0.95 x 0.98 = 0.931
    assert (cw.read_bytes(), dw.read_bytes()) == before


def test_a_sealed_tree_is_never_relabeled_l9(tmp_path):
    root, _dry_root, plan, cw, _dw, _dry = _relabel_world(tmp_path, "0.88")
    (root / "ledger.json").write_text("{}", encoding="utf-8")
    before = cw.read_bytes()
    with pytest.raises(rc.RunError, match="sealed"):
        rc.relabel_windows(plan, root, "sha")
    assert cw.read_bytes() == before


def test_the_drift_cli_records_a_skip_when_no_anchor_window_exists_m3(tmp_path, monkeypatch, capsys):
    root, plan = _drift_world(tmp_path)
    (root / "cells" / "gold-fresh|cell" / "window_qasper-01" / "metrics.json").unlink()
    monkeypatch.setattr(rc, "load_plan", lambda p: plan)
    monkeypatch.setattr(rc, "run_drift_window", lambda *a, **k: pytest.fail("no window to rerun"))
    assert rc.main(["drift-window", "--plan", "p.json", "--campaign-root", str(root)]) == 0
    assert "DRIFT_WINDOW_SKIPPED" in capsys.readouterr().out


def test_the_replan_is_a_candidate_until_the_diff_gate_passes_m2(world):  # noqa: F811
    on_profile = world["tmp"] / "S1_ceilings_m2.env"
    on_profile.write_text(world["profile"].read_text(encoding="utf-8") + "CLASS_CEILINGS=1\n", encoding="utf-8")
    world["profile"] = on_profile
    out = _master(world, "--plan", "--only", "plan").stdout
    cand, diff, promote = out.index("_candidate.json"), out.index("plan-diff"), out.index("promote the re-plan")
    assert cand < diff < promote < out.index("relabel-regime") < out.index("drift-window")
