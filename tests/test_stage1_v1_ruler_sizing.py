"""Stage 1 Batch S1-B, finding V1 of the Batch 1 code validation (CRITICAL):
RULER contexts were sized in whitespace words, not model tokens (owner:
"ok proceed with the best solution for this one", 2026-10-05).

Three parts, all pinned here:

1. The RULER haystack is counted with the TARGET MODEL's tokenizer. The
   loader already accepted a token counter (``RulerLoader(tokenizer=...)``);
   ``get_loader`` now forwards one, and the runner builds it from ``--model``
   (``ruler_token_counter``, a monkeypatch seam over
   ``src.data.ruler.model_token_counter``).
2. The RENDERED request fits the registered input shape. The driver passes a
   haystack target of SHAPE-32K minus a registered wrapper allowance plus the
   cap itself (``--ruler-rendered-input-cap 32512``); before any engine work
   the runner renders every measured RULER prompt exactly as the engine will
   (the chat template with the vLLM adapter's pinned kwargs) and refuses the
   cell when any rendered prompt exceeds the cap.
3. An engine "context length" error on the campaign path is a typed cell
   refusal (``ContextLengthError`` through the ADR-0116 record guard), never
   an error row with exit 0.

Measurement behind the allowance (2026-10-05, the real Qwen/Qwen3-14B
tokenizer, 30 items per task at a 2,048-token haystack, chat template with
``enable_thinking=False``): wrapper tokens niah_multikey 85 to 87,
niah_multiquery 99 to 106, variable_tracking 92 to 96, qa 75 to 91; the
system instruction plus the template alone cost 72. The allowance 128 covers
the maximum (106) with 22 to spare; the runner's render check covers the rest.
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

from src.data import ruler as ruler_mod  # noqa: E402
from src.data.loader import get_loader  # noqa: E402
from src.utils import prompting  # noqa: E402
from src.inference.vllm_adapter import VLLMAdapter  # noqa: E402

RUN_CAMPAIGN_PY = REPO_ROOT / "scripts" / "3_run" / "run_campaign.py"
RUN_EXPERIMENT_PY = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rc = _load("run_campaign_stage1_v1", RUN_CAMPAIGN_PY)
runner = _load("run_experiment_stage1_v1", RUN_EXPERIMENT_PY)

#: The 2026-10-05 measurements with the Qwen/Qwen3-14B tokenizer: the maximum
#: wrapper tokens per task over the 200 items of each campaign seed (42, 43,
#: 44) at the REAL haystack size 32,384 (independent review run; the earlier
#: 30-item run at 2,048 read 87/106/96/91). Restated here so the registered
#: allowance is checked against evidence, never asserted.
MEASURED_WRAPPER_MAX: Dict[str, int] = {
    "niah_multikey": 88,
    "niah_multiquery": 107,
    "variable_tracking": 97,
    "qa": 99,
}
#: The worst rendered request observed in that run (qa, seed 42): fits the cap.
MEASURED_RENDERED_MAX: int = 32_481


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    return len(text.split())


class _FakeTokenizer:
    """Counts whitespace words; the chat template adds a fixed wrapper of
    ``wrapper`` tokens and records the kwargs it was rendered with."""

    def __init__(self, wrapper: int = 10) -> None:
        self.wrapper = wrapper
        self.calls: List[Dict[str, Any]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [1] * _word_count(text)

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        self.calls.append({"add_generation_prompt": add_generation_prompt, "tokenize": tokenize, **kwargs})
        n = sum(_word_count(m["content"]) for m in messages) + self.wrapper
        return [1] * n if tokenize else " ".join(["x"] * n)


def _fake_transformers(monkeypatch, tokenizer: _FakeTokenizer, seen: List[str]) -> None:
    module = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any):
            seen.append(model_id)
            return tokenizer

    module.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)


# ===========================================================================
# Part 1: the model tokenizer reaches the loader
# ===========================================================================


def test_model_token_counter_loads_the_model_tokenizer(monkeypatch) -> None:
    tok = _FakeTokenizer()
    seen: List[str] = []
    _fake_transformers(monkeypatch, tok, seen)
    counter, name, tokenizer = ruler_mod.model_token_counter("Qwen/Qwen3-14B")
    assert seen == ["Qwen/Qwen3-14B"]
    assert name == "Qwen/Qwen3-14B"
    assert tokenizer is tok
    assert counter("one two three") == 3


def test_get_loader_forwards_the_counter_to_the_ruler_loader(monkeypatch) -> None:
    monkeypatch.setenv("CAGE_RULER_CONTEXT_TOKENS", "256")
    monkeypatch.setenv("CAGE_RULER_TASK", "niah_multikey")
    loader = get_loader("ruler", seed=3, token_counter=_word_count, token_counter_name="fake-words")
    assert type(loader).__name__ == "RulerLoader"
    assert loader.tokenizer_name == "fake-words"
    item = loader.load(max_examples=1)[0]
    assert item.metadata["tokenizer_name"] == "fake-words"
    assert item.metadata["actual_context_tokens"] == _word_count(item.context[0])
    assert item.metadata["actual_context_tokens"] <= 256


def test_get_loader_without_a_counter_keeps_the_labeled_proxy(monkeypatch) -> None:
    monkeypatch.setenv("CAGE_RULER_CONTEXT_TOKENS", "256")
    loader = get_loader("ruler", seed=3)
    assert loader.tokenizer_name == "whitespace-proxy"


def test_get_loader_refuses_a_counter_for_a_dataset_that_takes_none() -> None:
    with pytest.raises(ValueError, match="token counter"):
        get_loader("squad_v2", token_counter=_word_count)


# ===========================================================================
# Part 2: the rendered request fits the cap
# ===========================================================================


def test_render_helper_applies_the_template_with_the_pinned_kwargs() -> None:
    tok = _FakeTokenizer(wrapper=7)
    messages = prompting.format_qa_messages("Q?", ["a b c"])
    n = prompting.rendered_chat_prompt_tokens(tok, messages, chat_template_kwargs={"enable_thinking": False})
    content_words = sum(_word_count(m["content"]) for m in messages)
    assert n == content_words + 7
    assert tok.calls == [{"add_generation_prompt": True, "tokenize": True, "enable_thinking": False}]


def test_the_render_kwargs_are_the_vllm_adapter_pin() -> None:
    # The runner renders with exactly what the vLLM adapter sends; a drift
    # would count a different template than the engine applies.
    payload: Dict[str, Any] = {}
    VLLMAdapter._apply_engine_chat_extras(VLLMAdapter.__new__(VLLMAdapter), payload)
    assert payload["chat_template_kwargs"] == runner.RULER_RENDER_TEMPLATE_KWARGS


def _items(n: int, words: int) -> list:
    from src.data.loader import CAGExample

    return [
        CAGExample(id=f"ruler_x_{i:04d}", question="What is it?", context=[" ".join(["w"] * words)],
                   answer="a", metadata={"dataset": "ruler", "task": "qa"})
        for i in range(n)
    ]


def test_check_ruler_rendered_inputs_passes_and_records_the_sizing() -> None:
    tok = _FakeTokenizer(wrapper=10)
    items = _items(3, words=50)
    # system instruction words + user content words (the haystack plus
    # "Context 1:" and "Question: What is it?") + the template wrapper: 102
    # under the fake tokenizer, so a cap of 200 passes and the record is exact.
    sys_words = _word_count(prompting.CHAT_SYSTEM_INSTRUCTION)
    user_words = _word_count(prompting._qa_user_content("What is it?", [" ".join(["w"] * 50)]))
    record = runner.check_ruler_rendered_inputs(
        items, tokenizer=tok, prompt_mode="chat", cap=200, max_tokens=256,
    )
    assert record["rendered_max"] == record["rendered_min"] == sys_words + user_words + 10 == 102
    assert record["haystack_max"] == record["haystack_min"] == 50
    assert record["cap"] == 200 and record["n_items"] == 3 and record["prompt_mode"] == "chat"
    assert record["max_tokens"] == 256
    assert record["finding"] == "Batch 1 V1"


def test_check_ruler_rendered_inputs_refuses_an_over_cap_item_naming_it() -> None:
    tok = _FakeTokenizer(wrapper=10)
    items = _items(2, words=50) + _items(1, words=90)
    items[2].id = "ruler_x_long"
    with pytest.raises(runner.RulerSizingError) as exc:
        runner.check_ruler_rendered_inputs(items, tokenizer=tok, prompt_mode="chat", cap=100, max_tokens=256)
    msg = str(exc.value)
    assert "ruler_x_long" in msg and "100" in msg and "Batch 1 V1" in msg


def test_check_ruler_rendered_inputs_counts_the_raw_prompt_in_raw_mode() -> None:
    tok = _FakeTokenizer(wrapper=10)
    items = _items(1, words=20)
    record = runner.check_ruler_rendered_inputs(items, tokenizer=tok, prompt_mode="raw", cap=1000, max_tokens=256)
    expected = _word_count(prompting.format_qa_prompt("What is it?", items[0].context))
    assert record["rendered_max"] == expected
    assert tok.calls == [], "raw mode renders no chat template"


def test_check_ruler_rendered_inputs_refuses_an_empty_set() -> None:
    with pytest.raises(runner.RulerSizingError, match="no measured"):
        runner.check_ruler_rendered_inputs([], tokenizer=_FakeTokenizer(), prompt_mode="chat", cap=10, max_tokens=1)


# ===========================================================================
# Part 3: an engine context-length error fails the cell
# ===========================================================================


@pytest.mark.parametrize(
    "text",
    [
        # what the adapter records today on a rejected request: the status
        # line only (requests, then aiohttp on the open-loop path)
        "400 Client Error: Bad Request for url: http://localhost:8000/v1/chat/completions",
        "400, message='Bad Request', url='http://localhost:8000/v1/chat/completions'",
        "413 Client Error: Payload Too Large for url: http://localhost:8000/v1/completions",
        # the engines' own wording, when a body reaches the text
        "This model's maximum context length is 32768 tokens. However, you requested 32900 tokens",
        "ValueError: the input (33000 tokens) is longer than the model's context length (32768)",
        "Requested token count exceeds the model's maximum context length of 32768",
        "prompt is too long: 40000 tokens",
        "max_total_tokens exceeded",
    ],
)
def test_request_refusals_are_recognized(text: str) -> None:
    assert runner.is_context_length_error(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "", None, "HTTPConnectionPool: Read timed out", "no_response", "dropped_by_cap",
        "internal server error",
        "500 Server Error: Internal Server Error for url: http://localhost:8000/v1/chat/completions",
        "503, message='Service Unavailable', url='http://localhost:8000/v1/chat/completions'",
        "request exceeds the max in-flight cap",
        "LoRA adapter context length mismatch",
        "took 400 ms",
    ],
)
def test_other_errors_are_not_request_refusals(text: Any) -> None:
    assert runner.is_context_length_error(text) is False


def test_the_status_line_the_adapter_records_is_what_the_regex_sees() -> None:
    # Review F-1: the adapter never reads the body on an HTTPError, so the
    # text part 3 sees is requests' status line. Build one and check both.
    import requests

    response = requests.Response()
    response.status_code = 400
    response.reason = "Bad Request"
    response.url = "http://localhost:8000/v1/chat/completions"
    response._content = b'{"error": {"message": "This model\'s maximum context length is 32768 tokens"}}'
    with pytest.raises(requests.exceptions.HTTPError) as exc:
        response.raise_for_status()
    status_line = str(exc.value)
    assert status_line.startswith("400 Client Error")
    assert runner.is_context_length_error(status_line) is True


def test_context_length_error_names_the_row_and_the_finding() -> None:
    exc = runner.ContextLengthError("ruler_qa_32384_0007", "maximum context length is 32768")
    assert "ruler_qa_32384_0007" in str(exc) and "Batch 1 V1" in str(exc)
    assert isinstance(exc, RuntimeError)


def _body(src: str, start: str, end: str) -> str:
    return src[src.index(start):src.index(end)]


def test_wiring_the_refusal_sits_first_in_record_result_and_the_check_before_the_engine() -> None:
    src = RUN_EXPERIMENT_PY.read_text(encoding="utf-8")
    run = _body(src, "def run_experiment(", "class CacheResetError(")
    record = _body(run, "def record_result(", "def execute_work_units(")
    # part 3: the typed refusal is the first thing record_result does with an
    # error row, so no counter or histogram sees a row the cell then refuses
    assert record.index("is_context_length_error(") < record.index("performance_evaluator.record_request(")
    # part 2: the render check runs after the prompt mode is known and
    # before the engine is set up
    assert run.index("_prompt_mode = prompt_mode()") < run.index("check_ruler_rendered_inputs(")
    assert run.index("check_ruler_rendered_inputs(") < run.index("engine = setup_inference_engine(")
    # part 1: the loader gets the counter built from the model
    assert run.index("ruler_token_counter(") < run.index("loader = get_loader(")
    # the record lands in metrics.json
    assert 'experiment_summary["ruler_sizing"]' in run
    # review F-6: the warm-up pool items are checked with the measured set
    at = run.index("check_ruler_rendered_inputs(")
    assert "warmup_pool_examples" in run[at: at + 300]


def test_cli_threads_the_rendered_input_cap(monkeypatch) -> None:
    calls: List[Dict[str, Any]] = []

    def _recorder(**kwargs: Any) -> Dict[str, Any]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(runner, "run_experiment", _recorder)
    monkeypatch.setattr(
        sys, "argv",
        ["run_experiment.py", "--baseline", "no_cache", "--model", "m/x", "--dataset", "ruler",
         "--ruler-context-tokens", "512", "--ruler-task", "qa", "--ruler-rendered-input-cap", "640",
         "--num-queries", "2"],
    )
    runner.main()
    assert len(calls) == 1
    assert calls[0]["ruler_rendered_input_cap"] == 640


def test_cli_default_cap_is_none(monkeypatch) -> None:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(runner, "run_experiment", lambda **kw: calls.append(kw) or {})
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--baseline", "no_cache", "--model", "m/x", "--num-queries", "1"])
    runner.main()
    assert calls[0]["ruler_rendered_input_cap"] is None


# ===========================================================================
# F3 (review of S1-A): a prepare drop under --arrival-count is named
# ===========================================================================


def test_prepared_set_must_cover_the_arrival_count() -> None:
    runner.check_prepared_covers_arrivals(n_prepared=200, n_arrivals=200, dropped_ids=[])
    runner.check_prepared_covers_arrivals(n_prepared=200, n_arrivals=None, dropped_ids=["a"])  # duration mode
    with pytest.raises(runner.LoadGeneratorError) as exc:
        runner.check_prepared_covers_arrivals(n_prepared=199, n_arrivals=200, dropped_ids=["sq_17"])
    msg = str(exc.value)
    assert "sq_17" in msg and "199" in msg and "200" in msg and "CAGE_ALLOW_REPLAY" not in msg


def test_wiring_the_coverage_check_precedes_the_dispatch() -> None:
    src = RUN_EXPERIMENT_PY.read_text(encoding="utf-8")
    body = _body(src, "def execute_open_loop_measured(", "# ADR-0102 cold-start warm-up")
    assert body.index("check_prepared_covers_arrivals(") < body.index("dispatch_open_loop(")


# ===========================================================================
# the driver: registered haystack, allowance and cap
# ===========================================================================


def test_the_registered_shape_arithmetic() -> None:
    assert rc.RULER_SIZING_FINDING == "Batch 1 V1"
    assert rc.RULER_HAYSTACK_TOKENS == rc.RULER_CONTEXT_TOKENS - rc.RULER_WRAPPER_ALLOWANCE
    measured_max = max(MEASURED_WRAPPER_MAX.values())
    assert rc.RULER_WRAPPER_ALLOWANCE > measured_max, "the allowance must exceed the measured wrapper"
    # the rendered request fits the input shape, and the shape plus the output fits the cap
    assert rc.RULER_HAYSTACK_TOKENS + measured_max <= rc.RULER_CONTEXT_TOKENS
    assert MEASURED_RENDERED_MAX <= rc.RULER_CONTEXT_TOKENS
    assert rc.RULER_CONTEXT_TOKENS + rc.RULER_OUTPUT_TOKENS <= rc.DEFAULT_MAX_MODEL_LEN
    assert set(MEASURED_WRAPPER_MAX) == set(rc.RULER_F2_TASKS)


# -- a hermetic plan with one RULER coordinate ------------------------------

_DEMAND = 10_000_000_000


def _floor_table(tmp_path: Path) -> Path:
    rows = [
        {"r": r, "demand_bytes": _DEMAND, "budget_bytes": int(r * _DEMAND), "lambda_kv_rps": 2.0 * r,
         "lambda_compute_rps": None, "lambda_star_pred_rps": 2.0 * r, "lambda_star_basis": "test"}
        for r in (1.5, 1.0, 0.5, 0.25)
    ]
    doc = {"schema": "floor-table-v1",
           "generated_inputs": {"model": "qwen3-14b", "engine": "vllm", "kv_dtype": "bf16", "grid": "test",
                                # ADR-0155: the registered anchor arm's served tokens
                                "avg_seq_tokens": rc.DEMAND_SEQ_TOKENS_2026_10_08[rc.DEMAND_ANCHOR_ARM],
                                "concurrency_target": 32},
           "rows": rows}
    path = tmp_path / "floor_table.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _calibrations(tmp_path: Path) -> Dict[str, Path]:
    from src.orchestration.calibration import FLOOR_N_REQUESTS, FLOOR_STATISTIC, PROCEDURE_VERSION

    directory = tmp_path / "cal"
    directory.mkdir(exist_ok=True)
    doc = {"procedure_version": PROCEDURE_VERSION, "model": "Qwen/Qwen3-14B", "engine": "vllm",
           "budget_fraction": 1.5, "procedure": {}, "confirmatory": False,
           "floor": {"ttft_s": 0.1, "tpot_s": 0.01, "n_requests": FLOOR_N_REQUESTS, "statistic": FLOOR_STATISTIC},
           "lambda_star": {"label": "ESTIMATED", "lambda_star_qps": 2.0}}
    path = directory / "calibration_vllm.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return {"vllm": path}


def _rung_calibrations(tmp_path: Path, rungs=(1.0,)) -> Dict[str, Path]:
    """ADR-0154: the vLLM rung artifact (lambda* = 8 x r) for the ruler grid."""
    from src.orchestration.calibration import PROCEDURE_VERSION

    directory = tmp_path / "cal"
    directory.mkdir(exist_ok=True)
    doc = {"schema": rc.RUNG_CALIBRATION_SCHEMA, "procedure_version": PROCEDURE_VERSION,
           "confirmatory": False, "engine": "vllm", "model": "Qwen/Qwen3-14B", "session": "a",
           "workload": {}, "ladder": {},
           "rungs": {f"{r:g}": {"r": r, "label": "ESTIMATED", "lambda_star_qps": 8.0 * r,
                                "sustained_rate_qps": 8.0 * r, "first_unsustainable_qps": 10.4 * r,
                                "steps": []} for r in rungs}}
    path = directory / "rungs_vllm.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return {"vllm": path}


def _ruler_plan(tmp_path: Path) -> Dict[str, Any]:
    grid = rc.SessionGrid(
        session="a", group="A", model="qwen3-14b",
        f1_baselines=("B1",), f1_engines=("vllm",), f1_datasets=("squad_v2",), hf_oracle_cells=(),
        f2_baselines=("B1",), f2_engines=("vllm",), f2_budgets=(1.0,), f2_rates=(0.85,), f2_dataset="qasper",
        f2_ruler_baselines=("B1",), f2_ruler_tasks=("qa", "niah_multikey"),
        f3_baselines=(), f3_engines=("vllm",), f3_budgets=(), f3_rates=(), f3_dataset="qasper", dist_cells=(),
    )
    orig = rc.SESSION_GRIDS
    rc.SESSION_GRIDS = {"a": grid}
    try:
        return rc.build_plan(
            "a", rc.load_floor_table(_floor_table(tmp_path)), window_duration_s=60.0,
            runner_cmd=("stub",), launcher_cmds={"vllm": ("stub",), "sglang": ("stub",)},
            calibrations=_calibrations(tmp_path),
            rung_calibrations=_rung_calibrations(tmp_path),
        )
    finally:
        rc.SESSION_GRIDS = orig


def _argv_value(argv: List[str], flag: str):
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_ruler_cells_carry_the_haystack_and_the_cap(tmp_path) -> None:
    plan = _ruler_plan(tmp_path)
    ruler = [s for s in plan["steps"] if s["kind"] == "cell" and s["dataset"] == "ruler"]
    others = [s for s in plan["steps"] if s["kind"] == "cell" and s["dataset"] != "ruler"]
    assert len(ruler) == 2 and len(others) == 2
    for step in ruler:
        assert _argv_value(step["argv"], "--ruler-context-tokens") == str(rc.RULER_HAYSTACK_TOKENS)
        assert _argv_value(step["argv"], "--ruler-rendered-input-cap") == str(rc.RULER_CONTEXT_TOKENS)
        assert _argv_value(step["argv"], "--max-tokens") == str(rc.RULER_OUTPUT_TOKENS)
    for step in others:
        assert "--ruler-rendered-input-cap" not in step["argv"]
        assert "--ruler-context-tokens" not in step["argv"]


def test_the_header_records_the_sizing(tmp_path) -> None:
    plan = _ruler_plan(tmp_path)
    ruler_f2 = plan["ruler_f2"]
    assert ruler_f2["context_tokens"] == rc.RULER_CONTEXT_TOKENS
    assert ruler_f2["haystack_tokens"] == rc.RULER_HAYSTACK_TOKENS
    assert ruler_f2["wrapper_allowance_tokens"] == rc.RULER_WRAPPER_ALLOWANCE
    assert ruler_f2["rendered_input_cap"] == rc.RULER_CONTEXT_TOKENS
    assert ruler_f2["sizing_finding"] == rc.RULER_SIZING_FINDING


def _write(tmp_path: Path, plan: Dict[str, Any], name: str = "plan.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


class TestLoadPlanV1:
    def test_the_fresh_plan_loads(self, tmp_path) -> None:
        rc.load_plan(_write(tmp_path, _ruler_plan(tmp_path)))

    def test_a_ruler_cell_sized_at_the_old_32512_haystack_refuses(self, tmp_path) -> None:
        plan = _ruler_plan(tmp_path)
        step = next(s for s in plan["steps"] if s["kind"] == "cell" and s["dataset"] == "ruler")
        step["argv"][step["argv"].index("--ruler-context-tokens") + 1] = "32512"
        with pytest.raises(rc.RunError, match="ruler-context-tokens.*32512.*Batch 1 V1"):
            rc.load_plan(_write(tmp_path, plan))

    def test_a_ruler_cell_without_the_cap_refuses(self, tmp_path) -> None:
        plan = _ruler_plan(tmp_path)
        step = next(s for s in plan["steps"] if s["kind"] == "cell" and s["dataset"] == "ruler")
        at = step["argv"].index("--ruler-rendered-input-cap")
        del step["argv"][at:at + 2]
        with pytest.raises(rc.RunError, match="lacks --ruler-rendered-input-cap.*Batch 1 V1"):
            rc.load_plan(_write(tmp_path, plan))

    def test_a_non_ruler_cell_carrying_the_cap_refuses(self, tmp_path) -> None:
        plan = _ruler_plan(tmp_path)
        step = next(s for s in plan["steps"] if s["kind"] == "cell" and s["dataset"] != "ruler")
        step["argv"] += ["--ruler-rendered-input-cap", "32512"]
        with pytest.raises(rc.RunError, match="ruler-rendered-input-cap.*Batch 1 V1"):
            rc.load_plan(_write(tmp_path, plan))


# ===========================================================================
# the live measurement (opt-in: downloads or reads the Qwen3-14B tokenizer)
# ===========================================================================


@pytest.mark.skipif(os.getenv("CAGE_HF_LIVE") != "1", reason="set CAGE_HF_LIVE=1 to load the real Qwen3-14B tokenizer")
def test_live_the_allowance_covers_the_real_wrapper_on_every_task() -> None:
    counter, name, tok = ruler_mod.model_token_counter(rc.HF_ID_OF_SLUG["qwen3-14b"])
    worst_wrapper = 0
    worst_rendered = 0
    for task in rc.RULER_F2_TASKS:
        # the real haystack size, 20 items per task (seconds each)
        loader = ruler_mod.RulerLoader(context_length_tokens=rc.RULER_HAYSTACK_TOKENS, num_items=20,
                                       task=task, tokenizer=counter, tokenizer_name=name)
        for item in loader.load():
            rendered = prompting.rendered_chat_prompt_tokens(
                tok, prompting.format_qa_messages(item.question, item.context),
                chat_template_kwargs=runner.RULER_RENDER_TEMPLATE_KWARGS,
            )
            worst_wrapper = max(worst_wrapper, rendered - counter(item.context[0]))
            worst_rendered = max(worst_rendered, rendered)
    assert worst_wrapper <= rc.RULER_WRAPPER_ALLOWANCE
    assert worst_rendered <= rc.RULER_CONTEXT_TOKENS
