"""S0F-88 batch 2 (S0 attempt 4 bugs, 2026-10-12): regressions for the fixes
the owner ruled on (R1 to R19, "Approve all").

Each test fails on the code at cef4cde and passes after the fix it names.
Q-3 (R10, W19 option A) the HF oracle serves the chat template the engines
serve; Q-5 (R12) the dashboard reads the engine's dialect and open-loop
windows record a served rate; Q-8 (R14) the sampler's interval reaches
cage-stats' poll gap and both runner sites tick at 0.25 s; B-11 (R3, owner
choice "0.95 x ceiling") regime clause (a) reads the queued occupancy
against each laddered class's measured KV ceiling; B-7 (R7) stage 0 refuses
a KV pool larger than the memory the GPU has left after the weights; B-5 (R9)
the plan carries a measured run projection the master compares with HOURS.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.hf_oracle_adapter import HFOracleAdapter  # noqa: E402
from src.monitoring import vllm_telemetry as vt  # noqa: E402
from src.orchestration import load_generator as lg  # noqa: E402
from src.utils.prompting import format_qa_messages, messages_to_fallback_prompt  # noqa: E402

SCRIPTS = REPO_ROOT / "scripts"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Q-3: the oracle serves the chat template (W19 option A)
# --------------------------------------------------------------------------- #


class _RecordingTokenizer:
    """Records what the oracle tokenizes and how it renders."""

    pad_token_id = 0
    eos_token_id = 0

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.template_kwargs: Dict[str, Any] = {}

    def apply_chat_template(self, messages: Any, **kw: Any) -> str:
        self.template_kwargs = kw
        return "<|im_start|>user\n" + messages[-1]["content"] + "<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text: str, **kw: Any) -> Any:
        self.calls.append({"text": text, **kw})
        # KeyError is a programming error the adapter re-raises (B4), so the
        # test stops right after the tokenizer call it inspects
        raise KeyError("stop after tokenization")


def _bare_oracle(tokenizer: Any) -> HFOracleAdapter:
    oracle = HFOracleAdapter.__new__(HFOracleAdapter)
    oracle.tokenizer = tokenizer
    oracle.enforce_greedy = True
    oracle._torch = types.SimpleNamespace()
    oracle._corpus_cache = None
    oracle._corpus_prefix_text = None
    oracle.device = "cpu"
    oracle.device_map = None
    return oracle


def _request(messages: Any = None) -> Any:
    from src.inference.engine import InferenceRequest

    req = InferenceRequest(prompt="Context 1: X\n\nQuestion: Q?\nAnswer:", max_tokens=8,
                           temperature=0.0, top_p=1.0, request_id="r")
    if messages is not None:
        req.messages = messages
    return req


def test_a_chat_request_is_served_through_the_template_without_extra_specials_q3() -> None:
    tok = _RecordingTokenizer()
    oracle = _bare_oracle(tok)
    msgs = format_qa_messages("Who founded it?", ["Eric Isaacs founded it."])
    with pytest.raises(KeyError, match="stop after tokenization"):
        oracle.generate(_request(msgs))
    (call,) = tok.calls
    assert call["text"].startswith("<|im_start|>user\n") and call["add_special_tokens"] is False
    assert tok.template_kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert oracle.capabilities()["chat_template_thinking_pin"] is True
    # a raw request is served as given, with the default special tokens
    tok.calls.clear()
    with pytest.raises(KeyError):
        oracle.generate(_request())
    assert tok.calls[0]["text"].endswith("Answer:") and tok.calls[0]["add_special_tokens"] is True


def _qwen3_tokenizer() -> Any:
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", local_files_only=True)
    except Exception as exc:  # never download in a test
        pytest.skip(f"Qwen/Qwen3-14B tokenizer not in the local cache ({type(exc).__name__})")


def test_the_oracle_prompt_matches_the_engines_and_its_prefix_ids_q3() -> None:
    """S0 attempt 4: every oracle prompt was exactly 12 tokens shorter than
    the engines' and ended "Answer:"; 52 of 300 answers were one space."""
    tok = _qwen3_tokenizer()
    oracle = _bare_oracle(tok)
    runner = _load("run_experiment_s0f88_b2", SCRIPTS / "3_run" / "run_experiment.py")
    ctxs = ["Eric Isaacs founded the company in 1999.", "A second paragraph ends here 42"]
    msgs = format_qa_messages("Who founded the company?", ctxs)
    rendered = oracle.render_messages(msgs)
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    served = tok(rendered, add_special_tokens=False).input_ids
    engine_view = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    assert served == list(engine_view)
    assert len(served) - len(tok(messages_to_fallback_prompt(msgs)).input_ids) == 12
    # the preloader's chat prefix (rendered the same way) tokenizes as the
    # served prompt's own leading ids (F-01 under the template)
    prefix = runner.derive_corpus_prompt_prefix(
        lambda q: oracle.render_messages(format_qa_messages(q, ctxs))
    )
    assert rendered.startswith(prefix)
    head = tok(prefix, add_special_tokens=False).input_ids
    assert served[: len(head)] == head


def test_a_chat_preloader_refuses_an_oracle_without_the_render_seam_q3() -> None:
    runner = _load("run_experiment_s0f88_b2b", SCRIPTS / "3_run" / "run_experiment.py")

    class _NoRender:
        def capabilities(self) -> Dict[str, Any]:
            return {"corpus_prefix_reuse": True}

        def preload_corpus_prefix(self, text: str, **kw: Any) -> float:
            return 0.0

        def clear_corpus_prefix(self) -> None:
            pass

    runner.OracleCorpusPreloader(_NoRender(), "raw")  # raw mode needs no renderer
    with pytest.raises(RuntimeError, match="render_messages"):
        runner.OracleCorpusPreloader(_NoRender(), "chat")


# --------------------------------------------------------------------------- #
# Q-5: the dashboard reads the dialect; open-loop windows record a served rate
# --------------------------------------------------------------------------- #


def test_the_dashboard_forwards_the_dialect_and_skips_the_vllm_cli_q5(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Dict[str, Any] = {}

    def fake_dashboard(url: str, **kw: Any) -> str:
        seen.update(kw)
        raise RuntimeError("no server")

    monkeypatch.setattr(vt, "_try_import_api", lambda: types.SimpleNamespace(dashboard_text=fake_dashboard))
    monkeypatch.setattr(vt.shutil, "which", lambda name: "/usr/bin/cage-stats")
    ran: List[Any] = []
    monkeypatch.setattr(vt.subprocess, "run", lambda *a, **k: ran.append(a) or types.SimpleNamespace(returncode=1, stdout=""))
    assert vt.dashboard_text("http://localhost:30000", dialect="sglang") is None
    assert seen.get("dialect") == "sglang" and ran == []
    seen.clear()
    vt.dashboard_text("http://localhost:8000")
    assert "dialect" not in seen and ran  # vLLM keeps its CLI fallback


def test_the_runner_dashboard_uses_the_sampler_dialect_q5() -> None:
    src = (SCRIPTS / "3_run" / "run_experiment.py").read_text(encoding="utf-8")
    assert "dashboard_text(api_base, dialect=_telemetry_dialect)" in src
    assert "_dash = dashboard_text(api_base)\n" not in src


def test_an_open_loop_window_records_its_served_rate_q5() -> None:
    """S0 attempt 4: the window QPS read 0.54 (completions over summed
    latency) while 50 completions finished in 3.54 s (14.1 rps)."""
    recs = []
    for i in range(50):
        r = lg.RequestRecord(index=i, scheduled_offset_s=i * 0.05, scheduled_ts=i * 0.05)
        r.actual_send_ts = i * 0.05
        r.completion_ts = i * 0.05 + 1.09  # last completion at 3.54 s
        r.result = types.SimpleNamespace(error=None)
        recs.append(r)
    report = lg.DispatchReport(
        schedule=lg.generate_arrival_schedule(1.0, seed=1, n_requests=50), run_start_ts=0.0, records=recs
    )
    assert report.served_rps == pytest.approx(50 / 3.54)
    assert report.to_manifest()["served_rps"] == pytest.approx(50 / 3.54)
    empty = lg.DispatchReport(schedule=report.schedule, run_start_ts=0.0, records=[])
    assert empty.served_rps is None


# --------------------------------------------------------------------------- #
# Q-8: the sampler's interval reaches cage-stats and both sites tick at 0.25 s
# --------------------------------------------------------------------------- #


def test_the_sampler_forwards_its_interval_to_the_capture_q8(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler = vt.VllmTelemetrySampler("http://localhost:9", interval=0.25)
    seen: List[Any] = []

    def fake_capture(url: str, **kw: Any) -> Any:
        seen.append(kw.get("interval"))
        sampler._stop.set()
        return {"connected": True}

    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(vt, "capture_snapshot", fake_capture)
    sampler._run()
    assert seen == [0.25]


def test_both_runner_sampler_sites_tick_at_the_shared_interval_q8() -> None:
    src = (SCRIPTS / "3_run" / "run_experiment.py").read_text(encoding="utf-8")
    assert src.count("interval=TELEMETRY_SAMPLE_INTERVAL_S, dialect=_telemetry_dialect") == 2
    assert "interval=1.0, dialect=_telemetry_dialect" not in src
    runner = _load("run_experiment_s0f88_b2c", SCRIPTS / "3_run" / "run_experiment.py")
    campaign = _load("run_campaign_s0f88_b2", SCRIPTS / "3_run" / "run_campaign.py")
    assert runner.TELEMETRY_SAMPLE_INTERVAL_S == campaign.TELEMETRY_SAMPLE_INTERVAL_S == 0.25


# --------------------------------------------------------------------------- #
# Q-4: warm-ups fit the server's request cap (ADR-0102 amendment, R11)
# --------------------------------------------------------------------------- #


def _examples(n: int) -> List[Any]:
    from src.data.loader import CAGExample

    return [
        CAGExample(id=f"w{i}", question=f"q{i}?", context=[f"paragraph {i} " + "x " * (i * 10)], answer="a", metadata={})
        for i in range(n)
    ]


def test_a_capped_warmup_draw_keeps_only_requests_that_fit_q4() -> None:
    runner = _load("run_experiment_s0f88_b2d", SCRIPTS / "3_run" / "run_experiment.py")
    pool = _examples(40)
    long_ids = {f"w{i}" for i in range(0, 40, 2)}  # every even example is too long
    kw = dict(measured_ids=set(), measured_context_keys=set(), n=10, seed=42, trial=1)
    drawn = runner.draw_warmup_pool(pool, fits=lambda ex: ex.id not in long_ids, **kw)
    assert len(drawn) == 10 and not ({ex.id for ex in drawn} & long_ids)
    again = runner.draw_warmup_pool(pool, fits=lambda ex: ex.id not in long_ids, **kw)
    assert [ex.id for ex in again] == [ex.id for ex in drawn]  # deterministic
    with pytest.raises(runner.WarmupPoolError, match="request cap"):
        runner.draw_warmup_pool(pool, fits=lambda ex: ex.id in {"w1", "w3"}, **kw)
    # the uncapped draw is the pre-S0F-88 rng.sample path, unchanged
    import random

    expect = random.Random("cage-warmup-pool:42:1").sample(pool, 10)
    assert [ex.id for ex in runner.draw_warmup_pool(pool, **kw)] == [ex.id for ex in expect]


def test_the_served_prompt_counter_renders_like_the_engine_q4() -> None:
    runner = _load("run_experiment_s0f88_b2e", SCRIPTS / "3_run" / "run_experiment.py")
    seen: Dict[str, Any] = {}

    class _Tok:
        def apply_chat_template(self, messages: Any, **kw: Any) -> List[int]:
            seen.update(kw)
            return [0] * 77

        def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
            return [0] * len(text.split())

    assert runner.served_prompt_tokens("q?", ["ctx"], tokenizer=_Tok(), prompt_mode="chat") == 77
    assert seen.get("enable_thinking") is False and seen.get("add_generation_prompt") is True
    assert runner.served_prompt_tokens("q?", ["a b c"], tokenizer=_Tok(), prompt_mode="raw") > 0


def test_cells_carry_their_cap_and_a_refused_warmup_fails_the_window_q4() -> None:
    campaign_src = (SCRIPTS / "3_run" / "run_campaign.py").read_text(encoding="utf-8")
    assert 'argv += ["--request-cap-tokens", str(request_cap_tokens(grid, cap_class))]' in campaign_src
    runner_src = (SCRIPTS / "3_run" / "run_experiment.py").read_text(encoding="utf-8")
    assert runner_src.count('f"warm-up request {example.id} was refused for its length "') == 2
    assert "except WarmupPoolError:\n                        raise" in runner_src
    # S0F-88 audit L5: both draws take the raw-context filter except on a
    # retrieval arm, whose prepared requests are checked instead (review L5:
    # a retrieval arm served gold context keeps the filter)
    assert runner_src.count("fits=_draw_fits,") == 2 and '"length_cap": _warmup_cap_record' in runner_src
    assert "_draw_fits = None if _judged_on_prepared else _warmup_fits" in runner_src
    assert '_judged_on_prepared = _serves_retrieved and context_source != "gold"' in runner_src


# --------------------------------------------------------------------------- #
# B-6: the cross-encoder runs on the GPU (R8) and the device is recorded
# --------------------------------------------------------------------------- #


def test_rerank_cells_pin_cuda_and_record_the_device_b6() -> None:
    campaign = _load("run_campaign_s0f88_b2f", SCRIPTS / "3_run" / "run_campaign.py")
    argv = campaign.RETRIEVER_ARGV["rerank"]
    assert argv[argv.index("--reranker-device") + 1] == "cuda"
    assert "--reranker-device" not in campaign.RETRIEVER_ARGV["dense"]
    runner_src = (SCRIPTS / "3_run" / "run_experiment.py").read_text(encoding="utf-8")
    assert '"reranker_device": reranker_device if reranker is not None else None,' in runner_src


# --------------------------------------------------------------------------- #
# Q-10: context_relevance records whether it scored a cut context (R19)
# --------------------------------------------------------------------------- #


def test_context_relevance_flags_a_context_longer_than_the_embedder_limit_q10() -> None:
    import numpy as np

    from src.evaluation.quality import QualityEvaluator, QualityMetrics

    class _Model:
        max_seq_length = 8

        def tokenizer(self, text: str, **kw: Any) -> Dict[str, Any]:
            return {"input_ids": [0] * (len(text.split()) + 2)}

        def encode(self, x: Any, **kw: Any) -> Any:
            return np.ones(4, dtype="float32") if isinstance(x, str) else np.ones((len(x), 4), dtype="float32")

    ev = QualityEvaluator.__new__(QualityEvaluator)
    ev.use_embeddings = True
    ev._embedding_model = _Model()
    ev._instrument_unavailable = set()
    ev.embedding_model_name = "fake"
    assert ev.evaluate_relevance("q?", ["one two three"]) is not None
    assert ev.last_relevance_truncated is False
    ev.evaluate_relevance("q?", ["short", "a b c d e f g h i j k"])
    assert ev.last_relevance_truncated is True
    no_tok = _Model()
    no_tok.tokenizer = None  # type: ignore[assignment]
    ev._embedding_model = no_tok
    ev.evaluate_relevance("q?", ["a b c d e f g h i j k"])
    assert ev.last_relevance_truncated is None  # unknown, never assumed False
    m = QualityMetrics(faithfulness=None, relevance=0.5, completeness_bertscore=None,
                       completeness_rouge_l=None, context_relevance_truncated=True)
    assert m.to_dict()["context_relevance_truncated"] is True


# --------------------------------------------------------------------------- #
# B-11: clause (a) against the measured KV ceiling (R3, "0.95 x ceiling")
# --------------------------------------------------------------------------- #

sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_campaign_session import (  # noqa: E402
    MANIFEST_ENV,
    _emit_regime,
    _regime_doc,
    _regime_series,
    _session,
)
from test_run_campaign import (  # noqa: E402
    _cells,
    _dump,
    _floor_table_doc,
    _rungs_for,
    _write_calibrations,
    rc,
)

#: S0 attempt 4 ladder windows, (queue_waiting_share, rho_kv_queued_median),
#: recomputed from each window's cage_stats.jsonl by the production
#: compute_window_regime_inputs [V replay 2026-10-12, scratchpad b11/replay.py]
_SGLANG_SMALLEST_R075 = [(0.4308, 0.8619), (0.9873, 0.8729), (0.9863, 0.8578), (0.9882, 0.8635), (0.9872, 0.8551)]
_VLLM_ANCHOR_R125 = [
    (0.0149, 0.1271), (0.1212, 0.2133), (0.6232, 0.5549), (0.9759, 0.6812), (0.9717, 0.6888),
    (0.9778, 0.6979), (0.9885, 0.697), (0.9912, 0.7), (0.9959, 0.6978), (0.996, 0.7007), (0.9919, 0.7001),
]


def _records(pairs: List[Any]) -> List[Dict[str, Any]]:
    return [{"queue_waiting_share": a, "rho_kv_queued_median": b} for a, b in pairs]


def test_the_rung_ceiling_is_the_median_fill_of_its_queued_windows_b11() -> None:
    ceiling, n = rc.rung_kv_ceiling(_records(_SGLANG_SMALLEST_R075))
    assert n == 4 and ceiling == pytest.approx((0.8578 + 0.8635) / 2)  # the 0.43-share window is out
    ceiling, n = rc.rung_kv_ceiling(_records(_VLLM_ANCHOR_R125))
    assert n == 9 and ceiling == pytest.approx(0.6978)
    # absent, null, boolean or out-of-range inputs never count
    junk = [{}, {"queue_waiting_share": None, "rho_kv_queued_median": 0.9},
            {"queue_waiting_share": True, "rho_kv_queued_median": 0.9},
            {"queue_waiting_share": 0.9, "rho_kv_queued_median": 0.0}]
    assert rc.rung_kv_ceiling(junk) == (None, 0)
    assert rc.KV_CEILING_MAX_R == 1.0


def test_a_ladder_window_reports_its_queue_inputs_or_none_b11(tmp_path: Path) -> None:
    wdir = tmp_path / "cells" / "rk" / "window_qasper-01"
    wdir.mkdir(parents=True)
    assert rc._ladder_window_regime_fields(tmp_path, "rk", "qasper") == (None, None, None)
    (wdir / "regime.json").write_text(json.dumps({"inputs": None, "attainment": 0.9}), encoding="utf-8")
    assert rc._ladder_window_regime_fields(tmp_path, "rk", "qasper") == (None, None, 0.9)
    (wdir / "regime.json").write_text(json.dumps(
        {"inputs": {"queue_waiting_share": 0.98, "rho_kv_queued_median": 0.8729}, "attainment": 1.0}),
        encoding="utf-8")
    assert rc._ladder_window_regime_fields(tmp_path, "rk", "qasper") == (0.98, 0.8729, 1.0)


def test_the_artifact_parser_reads_ceilings_and_refuses_malformed_ones_b11() -> None:
    problems: List[str] = []
    table = {"0.5": {"kv_ceiling": 0.8651}, "0.75": {"kv_ceiling": None}, "1.25": {}}
    assert rc._parse_rung_ceilings(table, "rungs", problems) == {0.5: 0.8651} and problems == []
    for bad in (0.0, 1.2, True, "0.9", float("nan")):
        problems = []
        rc._parse_rung_ceilings({"0.5": {"kv_ceiling": bad}}, "rungs", problems)
        assert problems and "kv_ceiling" in problems[0]


def _plan_with_ceilings(tmp_path: Path, ceiling: float, small_ceiling: float) -> Dict[str, Any]:
    floor = tmp_path / "ft.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm", "sglang"))
    rungs = _rungs_for(cal, "a")
    for path in rungs.values():
        doc = json.loads(path.read_text(encoding="utf-8"))
        for rec in doc["rungs"].values():
            rec["kv_ceiling"] = ceiling
        for cls in doc["classes"].values():
            for rec in cls["rungs"].values():
                rec["kv_ceiling"] = ceiling if cls["role"] == "anchor" else small_ceiling
        path.write_text(json.dumps(doc), encoding="utf-8")
    return rc.build_plan(
        "a", rc.load_floor_table(floor), window_duration_s=300.0,
        calibrations=cal, rung_calibrations=rungs,
    )


def test_laddered_classes_pin_their_ceiling_at_r_up_to_1_and_nothing_else_does_b11(tmp_path: Path) -> None:
    plan = _plan_with_ceilings(tmp_path, ceiling=0.9081, small_ceiling=0.8607)
    pressure = [s for s in _cells(plan) if s["family"] in ("F2", "F3") and s["blocked_on"] is None]
    assert pressure
    seen = {"anchor": 0, "smallest": 0, "ruler": 0, "interpolated": 0, "loose": 0}
    for step in pressure:
        src = step["lambda_star_source"]
        r, pin = step["cellspec"]["budget_r"], step["env"].get(rc.KV_CEILING_ENV)
        if r > rc.KV_CEILING_MAX_R:
            assert src["kv_ceiling"] is None and pin is None
            seen["loose"] += 1
        elif step["rate_basis"] == rc.LAMBDA_BASIS_RUNG:
            assert src["kv_ceiling"] == 0.9081 and pin == "0.9081"
            seen["anchor"] += 1
        elif step["rate_basis"] == rc.LAMBDA_BASIS_SMALL:
            assert src["kv_ceiling"] == 0.8607 and pin == "0.8607"
            seen["smallest"] += 1
        elif step["rate_basis"] == rc.LAMBDA_BASIS_RULER:
            # ADR-0169: the RULER ladder measured its own ceiling (the fixture
            # gives every non-anchor class small_ceiling), pinned like any ladder
            assert step["dataset"] == "ruler" and src["kv_ceiling"] == 0.8607 and pin == "0.8607"
            seen["ruler"] += 1
        else:  # an interpolated class has no ladder, so no ceiling
            assert src["kv_ceiling"] is None and pin is None
            seen["interpolated"] += 1
    assert seen["anchor"] and seen["smallest"] and seen["ruler"] and seen["interpolated"]
    # a dry window replays its cell's env, the pin included
    for dry in (s for s in plan["steps"] if s["kind"] == "dry_window"):
        cell = next(s for s in plan["steps"][dry["index"] + 1:] if s["kind"] == "cell")
        assert dry["env"].get(rc.KV_CEILING_ENV) == cell["env"].get(rc.KV_CEILING_ENV)


def test_the_session_reads_the_pin_and_clause_a_uses_it_b11(tmp_path: Path) -> None:
    from src.analysis.goodput import IN_REGIME, UNPRESSURED
    from src.orchestration import campaign_session as cs

    pinned = _session(tmp_path / "p", env=MANIFEST_ENV, num_trials=2, kv_ceiling=0.8607)
    doc = _regime_doc(_emit_regime(pinned, tmp_path / "p", 1, series=_regime_series(0.87)))
    assert doc["label"] == IN_REGIME and doc["kv_ceiling"] == 0.8607  # 0.87 >= 0.95 x 0.8607
    plain = _session(tmp_path / "u", env=MANIFEST_ENV, num_trials=2)
    doc = _regime_doc(_emit_regime(plain, tmp_path / "u", 1, series=_regime_series(0.87)))
    assert doc["label"] == UNPRESSURED and doc["kv_ceiling"] is None  # 0.87 < RHO_KV_MIN 0.9
    for bad in (0.0, 1.5, float("nan"), True):
        with pytest.raises(cs.CampaignSessionError, match="CAGE_KV_CEILING"):
            _session(tmp_path / "b", kv_ceiling=bad)
    src = (REPO_ROOT / "src" / "orchestration" / "campaign_session.py").read_text(encoding="utf-8")
    assert "kv_ceiling = _env_float(env, KV_CEILING_ENV, ceiling_problems)" in src


def test_the_ladder_label_rederives_clause_a_as_the_runner_did_b11(tmp_path: Path) -> None:
    """The Q-1 re-derivation must judge clause (a) on the queued median the
    runner used, or a window whose attainments agree gets a false suffix."""
    row_key = "gold-fresh|none|none|single|sglang|qwen3-14b|F2|r0.75|lam0.85"
    window = tmp_path / "cells" / row_key / "window_qasper-01"
    window.mkdir(parents=True)
    rows = [{"arrival_s": 10.0 + i * 0.6, "ok": True} for i in range(100)]
    (window / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    regime = {  # time average below 0.9, queued median above it [S0 SGLang shape]
        "label": "IN_REGIME", "attainment": 1.0, "kv_ceiling": None,
        "inputs": {"rho_kv_time_avg": 0.8885, "queue_waiting_share": 0.98, "rho_kv_queued_median": 0.9093},
    }
    (window / "regime.json").write_text(json.dumps(regime), encoding="utf-8")
    assert rc._read_ladder_window(tmp_path, row_key, "qasper", 20.0, "ladder")[1] == "IN_REGIME"
    # a recorded ceiling is honored the same way
    regime["inputs"]["rho_kv_queued_median"], regime["kv_ceiling"] = 0.8754, 0.8607
    (window / "regime.json").write_text(json.dumps(regime), encoding="utf-8")
    assert rc._read_ladder_window(tmp_path, row_key, "qasper", 20.0, "ladder")[1] == "IN_REGIME"


def test_the_s0_sglang_queue_bound_windows_read_in_regime_under_the_ceiling_b11() -> None:
    """S0 attempt 4: three SGLang dry windows held the queue full (share 0.97
    to 1.0, attainment 1.0) and read UNPRESSURED on the time average. The
    mid classes had no ladder in attempt 4; the stand-in ceiling is the
    SGLang smallest class's at the same r (0.8651 at 0.5, 0.8607 at 0.75)
    [D, the next run ladders each class]."""
    from src.analysis.goodput import IN_REGIME, UNPRESSURED, classify_regime

    cases = [  # (time avg, share, queued median, stand-in ceiling) [V replay]
        (0.8951, 1.000, 0.8994, 0.8651),  # retr-reuse rerank F3 r=0.5 (class 1126)
        (0.8885, 0.980, 0.9093, 0.8607),  # retr-fresh dense F2 r=0.75 (class 1127)
        (0.8622, 0.970, 0.8754, 0.8607),  # retr-comp rerank F2 r=0.75 (class 702)
    ]
    for avg, share, qmed, ceiling in cases:
        before = classify_regime(rho_kv=avg, attainment=1.0, queue_waiting_share=share)
        after = classify_regime(rho_kv=avg, attainment=1.0, queue_waiting_share=share,
                                rho_kv_queued_median=qmed, kv_ceiling=ceiling)
        assert (before, after) == (UNPRESSURED, IN_REGIME)
    # vLLM at r=1.25 carries no pin: its 0.70 queued fill stays UNPRESSURED
    assert classify_regime(rho_kv=0.69, attainment=1.0, queue_waiting_share=0.99,
                           rho_kv_queued_median=0.6978) == UNPRESSURED


# --------------------------------------------------------------------------- #
# B-7: a pool above the GPU's memory after the weights is refused at stage 0
# --------------------------------------------------------------------------- #

_H100 = "NVIDIA H100 80GB HBM3"


def _relaunch(per_rank: int, *, r: float = 1.5, pools: Any = None) -> Dict[str, Any]:
    return {
        "engine": "vllm", "budget_r": r, "topology": "single", "demand_class": {"seq_tokens": 4779},
        "budget_plan": {"model": "qwen3-14b", "r": r, "per_rank_bytes": per_rank, "pools_bytes": pools},
    }


def test_a_pool_above_the_memory_left_after_the_weights_is_an_overrun_b7() -> None:
    free = rc.FREE_BYTES_AFTER_WEIGHTS[("qwen3-14b", _H100)]
    assert free == int(50.98 * 1024 ** 3)  # SGLang "avail mem=50.98 GB" (GiB) on S0 [V]
    assert rc.pool_ceiling_overruns([_relaunch(48_936_960_000, r=1.25)], _H100) == []  # started in S0
    (over,) = rc.pool_ceiling_overruns([_relaunch(58_724_352_000)], _H100)  # the r=1.5 anchor
    assert (over["pool_bytes"], over["free_bytes"], over["budget_r"]) == (58_724_352_000, free, 1.5)
    # a pd stack checks each role's pool; a step without a BudgetPlan is skipped
    pd = rc.pool_ceiling_overruns([_relaunch(0, pools=[free + 1, 1000]), {"engine": "vllm"}], _H100)
    assert [o["pool"] for o in pd] == ["prefill"]
    # an unmeasured (model, GPU) pair never passes blind
    blind = rc.pool_ceiling_overruns([_relaunch(1)], "NVIDIA H200")
    assert len(blind) == 1 and blind[0]["free_bytes"] is None
    assert "no measured value" in rc._pool_ceiling_text(blind, "where")


def test_stage_0_refuses_the_s1_anchor_at_r1p5_on_one_h100_and_passes_s0_b7() -> None:
    manifests = {  # S0's manifests (U3: qasper serves 100-arrival pressure windows)
        "squad_v2": str(REPO_ROOT / "data/manifests/squad_v2_50x3_seed42.json"),
        "qasper": str(REPO_ROOT / "data/manifests/qasper_100x3_seed42.json"),
    }
    kw = dict(floor_avg_seq_tokens="4779", query_manifests=manifests, floor_concurrency=50)
    s0 = rc.preflight_registration_check("a", rehearsal_n=50, gpu_id=_H100, **kw)
    assert s0["problems"] == [] and s0["pool_ceiling_overruns"] == []
    s1 = rc.preflight_registration_check("a", gpu_id=_H100, **kw)
    assert sorted((o["engine"], o["seq_tokens"], o["budget_r"]) for o in s1["pool_ceiling_overruns"]) == [
        ("sglang", 4779, 1.5), ("vllm", 4779, 1.5),
    ]
    assert any("exceed the memory the GPU has left after the weights" in p for p in s1["problems"])
    assert rc.preflight_registration_check("a", **kw)["pool_ceiling_overruns"] == []  # no GPU id, no check
    master = (SCRIPTS / "6_experiments" / "cage_experiment.sh").read_text(encoding="utf-8")
    assert '"$FLOOR_CONCURRENCY" "$GPU_ID" <<\'PY\'' in master and "gpu_id=gpu_id)" in master


# --------------------------------------------------------------------------- #
# B-9 wiring: S0 ladders every class (R5) through the master and the CLI
# --------------------------------------------------------------------------- #


def test_s0_ladders_every_class_through_the_master_and_the_cli_b9(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = (SCRIPTS / "6_experiments" / "profiles" / "S0.env").read_text(encoding="utf-8")
    assert "\nRUNG_ALL_CLASSES=1\n" in profile
    master = (SCRIPTS / "6_experiments" / "cage_experiment.sh").read_text(encoding="utf-8")
    assert '[ "${RUNG_ALL_CLASSES:-0}" = "1" ] && allc=" --all-classes"' in master
    # review 2026-10-13 item 3: the ladder job also carries $ffast (--fail-fast under LADDER_FAIL_FAST=1)
    assert "--rungs $rungs$qm2$reh2$allc$ffast$rstart " in master  # ADR-0169: $rstart = RULER_START_QPS
    seen: Dict[str, Any] = {}

    def fake_calibrate(*args: Any, **kw: Any) -> Dict[str, Any]:
        seen.update(kw)
        return {"classes": {}}

    monkeypatch.setattr(rc, "calibrate_rungs", fake_calibrate)
    monkeypatch.setattr(rc, "load_floor_table", lambda path: None)
    base = ["calibrate-rungs", "--session", "a", "--engine", "vllm", "--floor-table", "ft.json",
            "--calibration", "cal.json", "--out", str(tmp_path / "rungs.json"), "--out-root", str(tmp_path)]
    assert rc.main(base + ["--all-classes"]) == 0 and seen["all_classes"] is True
    assert rc.main(base) == 0 and seen["all_classes"] is False


# --------------------------------------------------------------------------- #
# B-5: the plan projects the run from measured step times (R9)
# --------------------------------------------------------------------------- #


def _step(engine: str, retriever: str, *, kind: str = "cell", n: int = 50, windows: int = 1,
          blocked: Any = None, key: str = "k") -> Dict[str, Any]:
    return {"kind": kind, "row_key": key, "num_queries": n, "windows": windows, "blocked_on": blocked,
            "cellspec": {"engine": engine, "retriever": retriever}}


def test_the_run_projection_sums_the_measured_table_and_lists_what_it_cannot_project_b5() -> None:
    steps = [
        {"kind": "relaunch", "engine": "sglang"}, {"kind": "relaunch", "engine": "vllm"},
        _step("vllm", "rerank"), _step("sglang", "none", windows=2),
        _step("vllm", "rerank", kind="dry_window"), _step("sglang", "dense", kind="dry_window"),
        _step("vllm", "none", blocked="no launcher"),  # never runs, costs nothing
    ]
    rp = rc.run_projection(steps)
    # G26 (2026-10-17): the prep is paid once per window (2 x 70.6 on the two-window step)
    expect = 254.8 + 55.0 + (378.1 + 208.4) + 2 * (70.6 + 49.2) + 735.5 + 189.6
    assert rp["total_s"] == pytest.approx(expect, abs=0.2) and rp["unmeasured_steps"] == []
    # another query count scales linearly (an upper bound); an unmeasured
    # (engine, retriever) or engine is listed, never guessed
    odd = rc.run_projection([
        _step("vllm", "none", n=200, key="big"), _step("hf", "dense", key="x"),
        {"kind": "relaunch", "engine": "lmdeploy"},
    ])
    assert odd["total_s"] == pytest.approx((21.8 + 65.8) * 4, abs=0.2) and odd["scaled_steps"] == 1
    assert len(odd["unmeasured_steps"]) == 2
    # the U3 pre-load adds its seconds per window
    pre = _step("vllm", "none")
    pre["argv"] = ["--open-loop-preload-s", "30"]
    assert rc.run_projection([pre])["total_s"] == pytest.approx(21.8 + 65.8 + 30.0, abs=0.2)


def test_the_s0_rehearsal_plan_projects_inside_its_hours_bound_b5(tmp_path: Path) -> None:
    """The rehearsal plan has attempt 4's shape; every step has a measured
    basis and the projection fits the profile's HOURS (S0 attempt 4: 35,639 s
    run with 31 cells not run; this plan runs them all)."""
    floor = tmp_path / "ft.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm", "sglang"))
    manifests = {  # S0's manifests (U3: qasper serves 100-arrival pressure windows)
        "squad_v2": REPO_ROOT / "data/manifests/squad_v2_50x3_seed42.json",
        "qasper": REPO_ROOT / "data/manifests/qasper_100x3_seed42.json",
    }
    plan = rc.build_plan("a", rc.load_floor_table(floor), window_duration_s=300.0, calibrations=cal,
                         rung_calibrations=_rungs_for(cal, "a"), query_manifests=manifests, rehearsal_n=50)
    c, rp = plan["counts"], plan["run_projection"]
    assert (c["cells"], c["relaunches"], c["dry_windows"], c["blocked"]) == (110, 38, 23, 3)
    assert rp["unmeasured_steps"] == []
    profile = (SCRIPTS / "6_experiments" / "profiles" / "S0.env").read_text(encoding="utf-8")
    hours = float(next(line for line in profile.splitlines() if line.startswith("HOURS=")).split("=")[1])
    # 100-arrival pressure windows plus the 30 s pre-load (S0F-88 U3)
    assert rp["scaled_steps"] == 51
    # B3 (ADR-0172, 2026-10-16): a window cell is charged its preparation scaled
    # by the pool plus, per window, the pre-load, the class duration and one
    # latency bound; before B3 the per-window cost scaled with the pool too and
    # this plan projected 15.67 h.
    assert rp["total_h"] == pytest.approx(14.23, abs=0.01) and rp["total_s"] < hours * 3600


# --------------------------------------------------------------------------- #
# Independent review (fresh Opus 5.5, 2026-10-12): C1, C2, H1, L1, L3, L6
# --------------------------------------------------------------------------- #


def test_a_capped_cell_draws_its_warmup_pool_without_a_name_error_review_c1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real run_experiment() with --request-cap-tokens: the warm-up
    length check rendered with the prompt mode before it was bound, so every
    capped cell raised NameError at the draw. Now the draw passes and the run
    stops at the engine (none runs here)."""
    import test_integration_wiring as tiw
    from src.data.loader import CAGExample

    runner = _load("run_experiment_s0f88_review_c1", SCRIPTS / "3_run" / "run_experiment.py")
    # S0 attempt 5 stage 4 (the pod runs every file in ONE process): a previous
    # test's main() exported its cell env (CAGE_CORPUS_RUNG=60 among them) into
    # os.environ, so this cell read "--corpus-rung 60 must equal
    # --corpus-prefix-budget 0"; clear the same list the layout harness clears.
    for name in (
        "CAGE_CAMPAIGN_ROOT", "CAGE_QUERY_MANIFEST", "CAGE_CELL_ARM", "CAGE_CELL_RETRIEVER",
        "CAGE_CELL_POLICY", "CAGE_CELL_FAMILY", "CAGE_CELL_TOPOLOGY", "CAGE_CELL_BUDGET_R",
        "CAGE_CELL_RATE_FRAC", "CAGE_CORPUS_PREFIX_BUDGET", "CAGE_CORPUS_RUNG",
        "CAGE_CELL_CORPUS_BUDGET", "CAGE_ORDER_BY_CONTEXT", "CAGE_MANIFEST_TRIAL",
        "VLLM_KV_CACHE_DTYPE", "CAGE_RULER_TASK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CAGE_LOG_ROOT", str(tmp_path / "logs"))

    class _Loader:
        def load(self, max_examples: Any = None) -> List[Any]:
            return [CAGExample(id=f"e{i}", question=f"question {i}?", context=[f"paragraph {i} " * 5],
                               answer="a", metadata={}) for i in range(max_examples or 200)]

    class _Tok:
        def apply_chat_template(self, messages: Any, **kw: Any) -> List[int]:
            return [0] * 50

        def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
            return [0] * len(text.split())

    monkeypatch.setattr(runner, "get_loader", lambda *a, **k: _Loader())
    monkeypatch.setattr(runner, "ruler_token_counter", lambda model: (lambda t: len(t.split()), "fake", _Tok()))
    # S0 attempt 5 stage 4: the pod runs the whole suite in ONE process, and an
    # earlier test's Prometheus server already holds the metrics port
    # ("Address already in use"); the layout harness stubs it, so does this test.
    monkeypatch.setattr(runner, "start_http_server", lambda *a, **k: None)
    kw = tiw._run_experiment_kwargs(workload_mode="single", output_dir=str(tmp_path / "run"),
                                    warmup_pool_queries=3, request_cap_tokens=4096, num_queries=2)
    with pytest.raises(RuntimeError, match="not ready"):
        runner.run_experiment(**kw)


def _plan_with_measured_classes(tmp_path: Path) -> Dict[str, Any]:
    floor = tmp_path / "ft.json"
    floor.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    cal = _write_calibrations(tmp_path / "cal", ("vllm", "sglang"))
    rungs = _rungs_for(cal, "a")
    grid = rc.get_session_grid("a")
    for engine, path in rungs.items():
        doc = json.loads(path.read_text(encoding="utf-8"))
        for template, class_rungs in rc.measured_pressure_classes(grid, engine, None):
            seq = rc.demand_seq_tokens(grid, template.spec)
            doc["classes"][str(seq)] = {
                "role": "measured", "seq_tokens": seq, "arm": template.spec.arm,
                "rungs": {f"{r:g}": {"r": r, "label": "ESTIMATED", "lambda_star_qps": 20.0 * r,
                                     "kv_ceiling": 0.86} for r in class_rungs},
            }
        path.write_text(json.dumps(doc), encoding="utf-8")
    return rc.build_plan("a", rc.load_floor_table(floor), window_duration_s=300.0,
                         calibrations=cal, rung_calibrations=rungs)


def test_a_plan_with_measured_classes_loads_and_its_kv_pins_are_rechecked_review_c2_l3(tmp_path: Path) -> None:
    plan = _plan_with_measured_classes(tmp_path)
    measured = [s for s in _cells(plan) if s.get("rate_basis") == rc.LAMBDA_BASIS_MEASURED]
    assert measured  # the B-9 basis reaches the plan
    rc.load_plan(_dump(tmp_path, plan, "measured.json"))  # S0 attempt 5's stage 7 path
    pinned = next(s for s in measured if s["env"].get(rc.KV_CEILING_ENV))
    pinned["env"][rc.KV_CEILING_ENV] = "0.5"  # a pin the source never recorded
    with pytest.raises(rc.RunError, match="CAGE_KV_CEILING"):
        rc.load_plan(_dump(tmp_path, plan, "tampered.json"))


def test_run_tests_runs_an_explicit_test_path_alone_review_h1(tmp_path: Path) -> None:
    """The stage 0 load gate passed its one file to run_tests.sh, which ran
    `pytest tests/ <file>`: the whole Mac suite."""
    import subprocess

    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python3").write_text('#!/bin/bash\necho "ARGS: $*"\n', encoding="utf-8")
    (fake / "python3").chmod(0o755)
    env = {**os.environ, "VIRTUAL_ENV": str(tmp_path), "PATH": f"{fake}:{os.environ['PATH']}"}
    script = SCRIPTS / "checks" / "run_tests.sh"
    one = subprocess.run(["bash", str(script), "-q", "tests/test_load_gate_p1.py"], env=env,
                         capture_output=True, text=True, check=True).stdout
    assert "ARGS: -m pytest -q tests/test_load_gate_p1.py" in one
    every = subprocess.run(["bash", str(script), "-q"], env=env, capture_output=True, text=True, check=True).stdout
    assert "ARGS: -m pytest tests/ -q" in every


def test_the_cold_floor_retries_a_refused_flush_then_gives_up_review_l1(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from src.inference.errors import EngineCapabilityUnavailableError

    cc = _load("calibrate_cell_s0f88_review_l1", SCRIPTS / "3_run" / "calibrate_cell.py")
    monkeypatch.setattr(cc, "FLOOR_FLUSH_POLL_S", 0.0)
    calls: List[int] = []

    def busy_twice() -> None:
        calls.append(1)
        if len(calls) <= 2:
            raise EngineCapabilityUnavailableError("sglang", "flush_endpoint", "pending requests")

    asyncio.run(cc._flush_cold(busy_twice))
    assert len(calls) == 3
    monkeypatch.setattr(cc, "FLOOR_FLUSH_WAIT_S", 0.0)

    def always() -> None:
        raise EngineCapabilityUnavailableError("sglang", "flush_endpoint", "no endpoint")

    with pytest.raises(EngineCapabilityUnavailableError):
        asyncio.run(cc._flush_cold(always))


def test_the_run_projection_lists_tp_and_pd_steps_as_unmeasured_review_l6() -> None:
    pd_cell = _step("vllm", "none", key="pd1")
    pd_cell["cellspec"]["topology"] = "pd"
    rp = rc.run_projection([pd_cell, {"kind": "relaunch", "engine": "vllm", "topology": "tp"}])
    assert rp["total_s"] == 0 and len(rp["unmeasured_steps"]) == 2


# --------------------------------------------------------------------------- #
# Owner decision 2026-10-12, option A: every calibration ladder serves with the
# prefix cache OFF; a prefix-ON-only class's cells keep prefix ON
# --------------------------------------------------------------------------- #


def test_a_prefix_on_only_class_is_laddered_with_the_cache_off_option_a(tmp_path: Path) -> None:
    from test_batch_b_pressure import FakeWorld, _calibrate
    from test_run_campaign import _tiny_grid

    # B3 corpus-reuse (F3, prefix ON) is the only cell of its class: no OFF twin
    grid = _tiny_grid(
        f1_baselines=("B1",), f2_baselines=("B1", "B11"), f2_budgets=(0.5,), f2_rates=(1.05,),
        f3_baselines=("B3",), f3_budgets=(0.5,), f3_rates=(1.05,),
    )
    on_only = [c for c in rc.enumerate_cells(grid) if c.baseline_id == "B3" and c.spec.family == "F3"]
    assert on_only and not rc._prefix_off(on_only[0].spec)
    seq = rc.demand_seq_tokens(grid, on_only[0].spec)
    doc = _calibrate(tmp_path, FakeWorld(cap=6.0, cap_by_arm={"retr-trunc": 24.0, "corpus-reuse": 12.0}),
                     grid, all_classes=True)
    rec = doc["classes"][str(seq)]
    assert (rec["role"], rec["arm"], rec["prefix_mode"], rec["ladder_prefix_mode"]) == (
        "measured", "corpus-reuse", "ON", "OFF",
    )
    for rung in rec["rungs"].values():
        assert "--no-prefix-cache" in rung["relaunch"]["argv"]
    # the class's own cell keeps the prefix cache ON
    assert rc._serving_config(on_only[0], grid)[1] is False and rc._ladder_config(on_only[0], grid)[1] is True
