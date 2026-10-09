"""S0F-59 (ADR-0151): scripts/checks/probe_thinking_pin.py, the stage 4 gate
that proves live that an engine serves answers, not thinking scaffolding.

The judgment is pure (``judge``); the run goes through a fake engine factory
so no server is needed: the probe must send the chat messages, stop ["\\n"]
and T=0 the campaign sends, record what chat_template_kwargs the adapter
emits, and print the one line the master greps.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.engine import InferenceResponse  # noqa: E402

_PATH = REPO_ROOT / "scripts" / "checks" / "probe_thinking_pin.py"
_spec = importlib.util.spec_from_file_location("probe_thinking_pin", _PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def _response(text: str, *, error: str | None = None, num_tokens: int = 3, finish_reason: str = "stop") -> InferenceResponse:
    return InferenceResponse(
        request_id=probe.PROBE_REQUEST_ID, generated_text=text, ttft_ms=1.0, total_time_ms=2.0,
        num_tokens=num_tokens, model_name="m", finish_reason=finish_reason, error=error,
    )


class _FakeEngine:
    """Records the request and answers a canned response; exposes the chat
    payload builder the way the OpenAI-compatible adapters do."""

    def __init__(self, response: InferenceResponse, kwargs: dict | None) -> None:
        self.response = response
        self.kwargs = kwargs
        self.requests: list = []

    def _build_chat_payload(self, request, messages, *, stream):
        payload = {"model": "m", "messages": list(messages), "stream": stream}
        if self.kwargs is not None:
            payload["chat_template_kwargs"] = dict(self.kwargs)
        return payload

    def generate(self, request, *, stream=False):
        self.requests.append((request, stream))
        return self.response


@pytest.mark.parametrize(
    "text, error, want",
    [
        ("Paris", None, None),
        ("  Paris.  ", None, None),
        ("<think>", None, "carries '<think>'"),
        ("<think>\nThe user asks", None, "carries '<think>'"),
        ("", None, "served text is empty"),
        ("   ", None, "served text is empty"),
        ("Paris", "engine_error: 500", "request failed: engine_error: 500"),
        ("<think>", "timeout", "request failed: timeout"),  # the error wins
    ],
)
def test_judge(text: str, error: str | None, want: str | None) -> None:
    got = probe.judge(text, error, 2, "stop")
    if want is None:
        assert got is None
    else:
        assert got is not None and want in got


def test_run_probe_sends_the_campaign_request_shape_and_records_the_pin() -> None:
    engine = _FakeEngine(_response("Paris"), {"enable_thinking": False})
    record = probe.run_probe(
        "sglang", "http://localhost:30000", "Qwen/Qwen3-14B",
        engine_factory=lambda backend, api_base, model: engine,
    )
    assert record["ok"] is True and record["reason"] is None
    assert record["chat_template_kwargs"] == {"enable_thinking": False}
    assert record["generated_text"] == "Paris" and record["num_tokens"] == 3
    assert record["marker"] == "<think>"
    ((request, stream),) = engine.requests
    assert stream is True
    assert request.stop == ["\n"] and request.temperature == 0.0 and request.max_tokens == 64
    assert request.request_id == probe.PROBE_REQUEST_ID
    assert [m["role"] for m in request.messages] == ["system", "user"]
    assert probe.PROBE_QUESTION in request.messages[1]["content"]
    assert probe.PROBE_CONTEXT in request.messages[1]["content"]


def test_run_probe_fails_on_the_landing_shape_and_records_a_missing_pin() -> None:
    engine = _FakeEngine(_response("<think>", num_tokens=2), None)
    record = probe.run_probe(
        "sglang", "http://localhost:30000", "Qwen/Qwen3-14B",
        engine_factory=lambda *_a: engine,
    )
    assert record["ok"] is False
    assert "carries '<think>'" in record["reason"] and "2 tokens" in record["reason"]
    assert record["chat_template_kwargs"] is None


def test_main_prints_the_one_line_and_writes_the_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "probe" / "record.json"
    monkeypatch.setattr(
        probe, "_runner_engine",
        lambda backend, api_base, model: _FakeEngine(_response("Paris"), {"enable_thinking": False}),
    )
    code = probe.main(["--backend", "vllm", "--api-base", "http://localhost:8000", "--model", "m", "--out", str(out)])
    assert code == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("THINKING_PIN_OK backend=vllm num_tokens=3")
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["ok"] is True and record["backend"] == "vllm"

    monkeypatch.setattr(
        probe, "_runner_engine",
        lambda backend, api_base, model: _FakeEngine(_response("<think>", num_tokens=2), None),
    )
    code = probe.main(["--backend", "sglang", "--api-base", "http://localhost:30000", "--model", "m", "--out", str(out)])
    assert code == 1
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("THINKING_PIN_FAILED backend=sglang reason=served text carries '<think>'")
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is False


def test_main_turns_a_construction_error_into_the_failed_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    def _boom(backend, api_base, model):
        raise RuntimeError("Model mismatch: expected m, server has other")

    monkeypatch.setattr(probe, "_runner_engine", _boom)
    out = tmp_path / "record.json"
    code = probe.main(["--backend", "vllm", "--api-base", "http://localhost:8000", "--model", "m", "--out", str(out)])
    assert code == 1
    assert "THINKING_PIN_FAILED backend=vllm reason=RuntimeError: Model mismatch" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is False


def test_probe_uses_the_runner_engine_construction_by_default() -> None:
    import inspect

    assert inspect.signature(probe.run_probe).parameters["engine_factory"].default is None
    src = _PATH.read_text(encoding="utf-8")
    assert "(engine_factory or _runner_engine)(backend, api_base, model)" in src
    assert "from run_experiment import setup_inference_engine" in src
    assert "strict=True" in src
