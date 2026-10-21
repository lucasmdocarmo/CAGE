"""ADR-0175 (Run 2, session s1) runner and run-side pieces: B1 per-trial
steps (the runner's --trial-index and the per-trial window range)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts" / "3_run")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_campaign as rc  # noqa: E402


def test_a_trial_step_counts_its_own_window_only_b1(tmp_path) -> None:
    cell = {"kind": "cell", "row_key": "gold-fresh|none|none|single|vllm|qwen3-14b|F2|r0.5|lam0.85",
            "dataset": "qasper", "window_ordinal_base": 0, "windows": 3}
    assert rc._window_range(cell) == (0, 3)
    t2 = rc._trial_step(cell | {"argv": [], "env": {}}, 2)
    assert rc._window_range(t2) == (1, 1) and t2["cell_windows"] == 3 and t2["argv"] == ["--trial-index", "2"]
    # window 02 complete: trial 2 is complete, trial 1 is not
    wdir = tmp_path / "cells" / cell["row_key"] / "window_qasper-02"
    wdir.mkdir(parents=True)
    (wdir / "metrics.json").write_text(json.dumps({"experiment": {}}), encoding="utf-8")
    dry = tmp_path / "dry"
    assert rc._segment_step_complete(tmp_path, dry, t2) is True
    assert rc._segment_step_complete(tmp_path, dry, rc._trial_step(cell | {"argv": [], "env": {}}, 1)) is False


def test_the_runner_serves_only_the_named_trial_b1(tmp_path, monkeypatch) -> None:
    import test_run_experiment_campaign_layout as layout

    seeds = []
    real = layout.runner.run_experiment

    def spy(**kwargs):
        seeds.append(kwargs.get("seed"))
        return real(**kwargs)

    monkeypatch.setattr(layout.runner, "run_experiment", spy)
    root = tmp_path / "camp" / "a" / layout.RUN_ID
    layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=100.0, num_trials=3,
                     extra_argv=("--trial-index", "2"))
    names = [d.name for d in layout._window_dirs(root, "no_cache", "squad_v2")]
    assert names == ["window_squad_v2-02"] and seeds == [7 + 1]  # --seed 7, trial 2


def test_the_runner_refuses_a_trial_outside_its_cell_b1(tmp_path, monkeypatch) -> None:
    import test_run_experiment_campaign_layout as layout

    root = tmp_path / "camp" / "a" / layout.RUN_ID
    with pytest.raises(SystemExit):
        layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=100.0, num_trials=3,
                         extra_argv=("--trial-index", "4"))


# --------------------------------------------------------------------------- #
# B4 option 3 (owner 2026-10-20, "One start, cut prompts"): per-class floors
# --------------------------------------------------------------------------- #


def _calibrate_cell():
    import importlib.util

    spec = importlib.util.spec_from_file_location("calibrate_cell_b4", REPO_ROOT / "scripts" / "3_run" / "calibrate_cell.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _WordTokenizer:
    """One token per word; the chat template adds 10 tokens."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=True):
        assert enable_thinking is False  # rendered as the adapters send it
        return ["t"] * (10 + sum(len(m["content"].split()) for m in messages))


class _FakeEngine:
    def __init__(self):
        self.flushes = 0
        self.seen = []

    def flush_cache(self):
        self.flushes += 1

    async def async_stream_generate(self, request):
        import types as _types

        self.seen.append(request)
        words = sum(len(m["content"].split()) for m in request.messages)
        tokens = 1 if len(self.seen) % 2 else 3  # every other answer stops after one token
        return _types.SimpleNamespace(error=None, num_tokens=tokens, ttft_ms=float(words), total_time_ms=float(words) + 20.0)


def test_class_floors_use_the_measured_shape_at_the_class_length_b4(tmp_path) -> None:
    import asyncio

    cc = _calibrate_cell()
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"blocks": [{"block_id": i, "text": f"w{i} " * 40} for i in range(30)]}), encoding="utf-8")
    engine = _FakeEngine()
    floors = asyncio.run(cc.measure_class_floors(engine, str(manifest), [200, 60], _WordTokenizer()))
    assert set(floors) == {"200", "60"}
    for seq, rec in ((200, floors["200"]), (60, floors["60"])):
        assert rec["n_requests"] == 30 and rec["n_tpot"] == 15 and rec["shape"].startswith("chat messages")
        assert abs(rec["prompt_tokens"] - seq) <= 1  # the rendered prompt is about the class length
    assert floors["200"]["ttft_s"] > floors["60"]["ttft_s"] and floors["60"]["tpot_s"] == 0.01
    assert engine.flushes == 60  # every request starts cold
    assert all(r.stop == ["\n"] and r.messages for r in engine.seen)
    assert len({r.messages[-1]["content"] for r in engine.seen[:30]}) == 30  # distinct prompts per class


# --------------------------------------------------------------------------- #
# ADR-0175 amendment: the K1 frontier rule on the anchor ladders
# --------------------------------------------------------------------------- #


def _ladder_window(tmp_path: Path, name: str, row_key: str, ttfts_ms) -> str:
    root = tmp_path / name
    wdir = root / "cells" / row_key / "window_qasper-01"
    wdir.mkdir(parents=True)
    rows = [{"arrival_s": 5.0, "ok": True, "ttft_from_scheduled_ms": 99999.0}]  # warm-up: never counted
    rows += [{"arrival_s": 10.0 + i, "ok": t is not None, "ttft_from_scheduled_ms": t} for i, t in enumerate(ttfts_ms)]
    (wdir / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(root)


def test_the_frontier_is_the_last_rate_before_the_share_drops(tmp_path) -> None:
    key = "gold-fresh|none|none|single|vllm|qwen3-14b|F2|r0.5|lam1"
    # target 1.0 s: shares 1.0, 0.9, 0.8 then a recovery to 1.0 that must not count
    steps = [
        {"rate_qps": 2.0, "root": _ladder_window(tmp_path, "s0", key, [100] * 10)},
        {"rate_qps": 2.6, "root": _ladder_window(tmp_path, "s1", key, [100] * 9 + [1500])},
        {"rate_qps": 3.4, "root": _ladder_window(tmp_path, "s2", key, [100] * 8 + [None, 2000])},
        {"rate_qps": 4.4, "root": _ladder_window(tmp_path, "s3", key, [100] * 10)},
    ]
    docs = {"vllm": {"rungs": {
        "0.5": {"label": "ESTIMATED", "lambda_star_qps": 4.0, "cell_row_key": key, "steps": steps},
        "0.25": {"label": "ESTIMATED", "lambda_star_qps": 3.0, "cell_row_key": key, "steps": steps[:1]},
    }}}
    out = rc.frontier_loads_from_ladders(docs, 1.0, "qasper", (0.5, 0.25))
    assert out["loads"]["vllm"] == {"0.5": 0.65, "0.25": float(f"{2.0 / 3.0:.4g}")}  # 2.6/4.0, 2.0/3.0
    assert out["windows"]["vllm"]["0.5"][2] == [3.4, 0.8] and out["share"] == 0.9
    path = tmp_path / "frontier.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    assert rc.load_frontier_loads(path) == {"vllm": {0.5: 0.65, 0.25: float(f"{2.0 / 3.0:.4g}")}}
    # a rung whose lowest window already misses the share has no frontier
    docs["vllm"]["rungs"]["0.25"]["steps"] = steps[2:3]
    with pytest.raises(rc.PlanError, match="owner decision at the review pause"):
        rc.frontier_loads_from_ladders(docs, 1.0, "qasper", (0.5, 0.25))


def test_a_trial_step_leaves_another_trials_partial_window_alone_b1_review_l2(tmp_path, monkeypatch) -> None:
    import test_run_experiment_campaign_layout as layout

    root = tmp_path / "camp" / "a" / layout.RUN_ID
    partial = root / "cells" / layout.ROW_KEY_OF["no_cache"] / "window_squad_v2-01"
    partial.mkdir(parents=True)
    (partial / "requests.jsonl").write_text("{}\n", encoding="utf-8")  # trial 1 died half-written
    layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=100.0, num_trials=3,
                     extra_argv=("--trial-index", "2"))
    assert (partial / "requests.jsonl").is_file()  # trial 1's evidence stays for its own step
    assert (partial.parent / "window_squad_v2-02" / "metrics.json").is_file()


def test_a_cell_the_budget_stop_left_unrun_is_a_skip(tmp_path) -> None:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "4_analysis"))
    import organize_results as org

    cell = tmp_path / "cells" / "gold-fresh|none|none|single|vllm|qwen3-14b|F2|r0.25|lam0.7"
    cell.mkdir(parents=True)
    rc._write_failed_sentinel(tmp_path, cell.name, "qasper", "budget-stop", 0)
    assert org.dry_window_skip_reasons(cell) is not None
    (cell / ".STATUS-qasper").write_text("reason=runner-exit-1\n", encoding="utf-8")
    assert org.dry_window_skip_reasons(cell) is None  # a real failure stays a FAIL


# --------------------------------------------------------------------------- #
# A6 (owner 2026-10-20): one-sided A/A rule and the McNemar A/A on repeats
# --------------------------------------------------------------------------- #


def test_the_aa_rule_is_one_sided_and_repeats_pair_each_question_a6(tmp_path) -> None:
    import importlib.util

    import numpy as np
    from src.analysis.stats.calibration import AAResult, aa_repeat_pairs

    assert AAResult(1000, 0.05, 0, 0.0, 0.0, 0.004).approximates_nominal  # conservative: passes
    assert not AAResult(1000, 0.05, 120, 0.12, 0.10, 0.14).approximates_nominal  # inflated: fails
    spec = importlib.util.spec_from_file_location("run_calibration_a6", REPO_ROOT / "scripts" / "4_analysis" / "run_calibration.py")
    rcal = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rcal  # dataclasses resolve their module at class creation
    spec.loader.exec_module(rcal)
    same = np.array([1.0, 0.0] * 40)
    res = aa_repeat_pairs(same, same.copy(), rcal.mcnemar_p, 200, 7)
    assert res.n_rejections == 0 and res.approximates_nominal  # identical repeats never reject
    # the S1 mode pairs a cell's first two windows by example_id
    for k, rows in ((1, [("q1", True), ("q2", False), ("q3", True), ("q4", True), ("q5", None)]),
                    (2, [("q2", False), ("q1", True), ("q3", True), ("q4", False), ("q5", True)])):
        w = tmp_path / "predicate" / "s" / "cells" / "gold-fresh|none|none|single|vllm|qwen3-14b|F1" / f"window_qasper-{k:02d}"
        w.mkdir(parents=True)
        (w / "predicate.jsonl").write_text("\n".join(json.dumps({"example_id": e, "predicate": v}) for e, v in rows), encoding="utf-8")
    pairs = rcal.campaign_repeat_pairs(tmp_path / "predicate" / "s")
    ((name, (a, b)),) = pairs.items()
    assert name.endswith("window_qasper") and a == [1.0, 0.0, 1.0, 1.0] and b == [1.0, 0.0, 1.0, 0.0]
    report = rcal.campaign_aa_report(tmp_path / "predicate" / "s", alpha=0.05, n_resamples=50, seed=1)
    assert report["arms"][name]["n_pairs"] == 4 and "pending" in report["injections"]
    with_inj = rcal.campaign_aa_report(tmp_path / "predicate" / "s", alpha=0.05, n_resamples=50, seed=1, injection_shift=0.25)
    assert with_inj["arms"][name]["injection"]["shift"] == 0.25 and with_inj["pass"] == report["pass"]  # reported, not gated


def _pool_step(engine, nbytes, tokens, *, topology="single", tp=1, flag=None, value=None, r=1.25):
    flag = flag or ("--kv-cache-memory-bytes" if engine == "vllm" else "--max-total-tokens")
    value = value or (str(nbytes) if engine == "vllm" else str(tokens))
    return {
        "engine": engine, "budget_r": r, "topology": topology, "max_model_len": 32768,
        "demand_class": {"seq_tokens": 4779},
        "budget_plan": {"tp": tp, "budget_bytes_total": nbytes, "budget_tokens_total": tokens,
                        "engine_args": [{"kind": "primary", "args": [flag, value]},
                                        {"kind": "fallback", "args": ["--num-gpu-blocks-override", "1"]}]},
    }


def test_the_largest_planned_pool_per_engine_a11_item4() -> None:
    steps = [
        _pool_step("vllm", 100, 10, r=0.5), _pool_step("vllm", 300, 30), _pool_step("vllm", 200, 20, r=1.0),
        _pool_step("vllm", 900, 90, topology="tp", tp=2),  # tp and pd are out of scope
        _pool_step("sglang", 250, 25),
        {"engine": "sglang", "topology": "single"},  # a budget-free relaunch
    ]
    got = rc.largest_planned_pools(steps)
    assert got["vllm"]["planned_bytes"] == 300 and got["vllm"]["planned_tokens"] == 30
    assert got["vllm"]["env"] == {"CAGE_KV_BUDGET_BYTES": "300", "VLLM_MAX_MODEL_LEN": "32768"}
    assert got["sglang"]["env"] == {"CAGE_SGLANG_MAX_TOTAL_TOKENS": "25", "VLLM_MAX_MODEL_LEN": "32768"}
    assert rc.largest_planned_pools([]) == {}


def test_the_s1_largest_pools_are_the_1_25_anchor_pool_a11_item4() -> None:
    # the plan's own arithmetic at c = 50: both engines' largest pool is the S0
    # anchor pool (298,687 tokens, 48,936,960,000 B; run_campaign.py FREE_BYTES note)
    grid = rc.get_session_grid("s1")
    report = rc.preflight_registration_check(
        "s1", floor_avg_seq_tokens=str(grid.demand_seq_tokens[rc.DEMAND_ANCHOR_ARM]),
        floor_concurrency=50, gpu_id="NVIDIA H100 80GB HBM3")
    assert report["problems"] == []
    for engine in ("vllm", "sglang"):
        pool = report["largest_pools"][engine]
        assert (pool["budget_r"], pool["planned_tokens"], pool["planned_bytes"]) == (1.25, 298687, 48936960000)
    assert report["largest_pools"]["sglang"]["env"]["CAGE_SGLANG_MAX_TOTAL_TOKENS"] == "298687"


def test_the_pool_check_command_applies_the_relaunch_rule_a11_item4(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CAGE_LOG_ROOT", str(tmp_path))
    record = tmp_path / "sglang" / rc.KV_POOL_RECORD_NAME
    record.parent.mkdir(parents=True)
    argv = ["pool-check", "--engine", "sglang", "--planned-tokens", "298687", "--planned-bytes", "48936960000",
            "--out", str(tmp_path / "check.json")]
    record.write_text(json.dumps({"tokens": 298687, "bytes": 48936960000}), encoding="utf-8")
    assert rc.main(argv) == 0 and "LARGEST_POOL_OK engine=sglang" in capsys.readouterr().out
    assert json.loads((tmp_path / "check.json").read_text(encoding="utf-8"))["ok"] is True
    record.write_text(json.dumps({"tokens": 282583, "bytes": 1}), encoding="utf-8")  # the S0 5.4% shortfall
    assert rc.main(argv) == 1 and "LARGEST_POOL_FAILED" in capsys.readouterr().out
    record.unlink()  # no record is never a pass
    assert rc.main(argv) == 1 and "no readable pool record" in capsys.readouterr().out


def test_the_power_injection_raises_the_accepted_share_by_the_registered_margin_a6() -> None:
    import importlib.util

    import numpy as np
    from src.analysis.stats.calibration import CalibrationError, recover_power_repeat_pairs

    spec = importlib.util.spec_from_file_location("run_calibration_a6i", REPO_ROOT / "scripts" / "4_analysis" / "run_calibration.py")
    rcal = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rcal
    spec.loader.exec_module(rcal)
    rng = np.random.default_rng(3)
    a = (rng.random(852) < 0.8).astype(float)  # the K1 Qasper size, about the S0 accepted share
    big, capped = recover_power_repeat_pairs(a, a.copy(), rcal.mcnemar_p, 0.05, 200, 11)
    # 43 flips are all discordant pairs in one direction: McNemar sees them at n = 852
    assert big.power > 0.95 and capped == 0 and big.target_power is None and big.effect_size == 0.05
    small, _ = recover_power_repeat_pairs(a[:50], a[:50].copy(), rcal.mcnemar_p, 0.05, 200, 11)
    assert small.power < big.power  # 3 flips at n = 50 are rarely significant
    ones = np.ones(20)
    _, capped = recover_power_repeat_pairs(ones, ones.copy(), rcal.mcnemar_p, 0.05, 20, 1)
    assert capped == 20  # nothing to flip: every resample is capped and reported
    with pytest.raises(CalibrationError, match="binary"):
        recover_power_repeat_pairs(a + 0.5, a, rcal.mcnemar_p, 0.05, 10, 1)
    with pytest.raises(CalibrationError, match="absolute share"):
        recover_power_repeat_pairs(a, a, rcal.mcnemar_p, 0.0, 10, 1)
    # the registered size is the freeze file's binary_predicate margin
    freeze = REPO_ROOT / "MyDocs" / "registration" / "freeze_resolutions.json"
    if freeze.is_file():
        assert rcal.registered_binary_margin(freeze) == 0.05


def test_the_setup_prefetch_points_refs_main_at_the_pin_review_2026_10_21_c1(tmp_path) -> None:
    """C1: a download at a commit SHA writes no refs/main, and under
    HF_HUB_OFFLINE=1 a load that names no revision resolves "main" through it
    (verified on a real cache 2026-10-21). The setup's own prefetch snippet,
    run against a stub hub, must write refs/main = the pin, and only for a pin."""
    import re
    import subprocess

    text = (REPO_ROOT / "scripts" / "runpod" / "setup_runpod.sh").read_text(encoding="utf-8")
    snippet = re.search(r'python - "\$_m" "\$\{PREFETCH_REVISIONS:-\}" <<\'PY\'\n(.*?)\nPY\n', text, re.S).group(1)
    stub = tmp_path / "stub" / "huggingface_hub"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(
        "import os\n"
        "def snapshot_download(repo, max_workers=8, revision=None):\n"
        "    root = os.path.join(os.environ['STUB_CACHE'], 'models--' + repo.replace('/', '--'))\n"
        "    snap = os.path.join(root, 'snapshots', revision or 'headsha')\n"
        "    os.makedirs(snap, exist_ok=True)\n"
        "    return snap\n", encoding="utf-8")
    env = {"PYTHONPATH": str(tmp_path / "stub"), "STUB_CACHE": str(tmp_path / "cache"), "PATH": "/usr/bin:/bin"}
    pinned = subprocess.run([sys.executable, "-c", snippet, "org/pinned", "org/pinned=abc123 other/x=def"],
                            env=env, capture_output=True, text=True)
    assert pinned.returncode == 0, pinned.stderr
    assert (tmp_path / "cache" / "models--org--pinned" / "refs" / "main").read_text(encoding="utf-8") == "abc123"
    free = subprocess.run([sys.executable, "-c", snippet, "org/free", "other/x=def"], env=env, capture_output=True, text=True)
    assert free.returncode == 0, free.stderr
    assert not (tmp_path / "cache" / "models--org--free" / "refs").exists()  # an unpinned model keeps the hub's own refs
