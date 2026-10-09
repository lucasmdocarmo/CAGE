#!/usr/bin/env python3
"""Order:     stage 4 (validate), after the engine's own probes and before the launcher stop
Objective: Prove live that a chat request served by this engine carries no thinking scaffolding (S0F-59, ADR-0151)
Cloud:     both

One chat request, built the way the campaign runner builds every request
(``run_experiment.setup_inference_engine`` for the adapter, the Decision 1B
chat messages, stop ["\\n"], T=0), goes to the engine at --api-base. The probe
FAILS when the request errors, when the served text is empty, or when it
carries the thinking-open marker "<think>": on the 2026-10-08 landing SGLang
served 2,450 of 2,450 rows as "<think>" in 2 tokens because nothing pinned
enable_thinking=false on that engine, and no gate said a word. It prints ONE
line the master greps (THINKING_PIN_OK or THINKING_PIN_FAILED), writes the
evidence record to --out, and exits 0 or 1.

The verifier's check (n) (scripts/4_analysis/verify_results.py) reads the same
marker on every served row after the run; THINKING_MARKER is pinned equal by
tests/test_verify_results_v2.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "3_run")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: The chat template's thinking-open tag (Qwen3); = verify_results._THINKING_MARKER.
THINKING_MARKER = "<think>"
PROBE_QUESTION = "What is the capital of France?"
PROBE_CONTEXT = "France is a country in Western Europe. Its capital city is Paris."
PROBE_REQUEST_ID = "thinking-pin-probe"


def judge(text: str, error: Optional[str], num_tokens: Any, finish_reason: Any) -> Optional[str]:
    """None when the served text is an answer; otherwise the failure reason."""
    if error:
        return f"request failed: {error}"
    if THINKING_MARKER in text:
        return (
            f"served text carries {THINKING_MARKER!r} ({num_tokens} tokens, "
            f"finish_reason={finish_reason!r}): the chat template opened a thinking block "
            "and the stop sequence ended the request before any answer"
        )
    if not text.strip():
        return f"served text is empty ({num_tokens} tokens, finish_reason={finish_reason!r})"
    return None


def _runner_engine(backend: str, api_base: str, model: str) -> Any:
    """The adapter exactly as the campaign runner constructs it (its env
    overrides and its served-model check included)."""
    from run_experiment import setup_inference_engine  # noqa: E402 (heavy, pod venv)

    config = types.SimpleNamespace(api_base=api_base, enable_prefix_caching=True)
    return setup_inference_engine(model, config, backend=backend, strict=True)


def run_probe(
    backend: str,
    api_base: str,
    model: str,
    *,
    max_tokens: int = 64,
    engine_factory: Optional[Callable[[str, str, str], Any]] = None,
) -> Dict[str, Any]:
    """Send the one request and judge it; returns the evidence record.
    ``engine_factory`` defaults to the runner's own construction, resolved
    at call time (tests substitute a fake engine)."""
    from src.inference.engine import InferenceRequest
    from src.utils.prompting import format_qa_messages, messages_to_fallback_prompt

    engine = (engine_factory or _runner_engine)(backend, api_base, model)
    messages = format_qa_messages(PROBE_QUESTION, [PROBE_CONTEXT])
    request = InferenceRequest(
        prompt=messages_to_fallback_prompt(messages),
        max_tokens=max_tokens,
        temperature=0.0,
        stop=["\n"],
        request_id=PROBE_REQUEST_ID,
    )
    request.messages = messages  # the chat path, as the runner attaches it
    build = getattr(engine, "_build_chat_payload", None)
    sent_kwargs = (
        build(request, messages, stream=True).get("chat_template_kwargs") if callable(build) else None
    )
    response = engine.generate(request, stream=True)
    text = response.generated_text or ""
    reason = judge(text, response.error, response.num_tokens, response.finish_reason)
    return {
        "backend": backend,
        "api_base": api_base,
        "model": model,
        "engine_id": getattr(response, "engine_id", None),
        "chat_template_kwargs": sent_kwargs,
        "generated_text": text,
        "num_tokens": response.num_tokens,
        "finish_reason": response.finish_reason,
        "error": response.error,
        "marker": THINKING_MARKER,
        "ok": reason is None,
        "reason": reason,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    parser.add_argument("--backend", required=True, choices=["vllm", "sglang", "lmdeploy"])
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, default=None, help="evidence JSON record")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        record = run_probe(args.backend, args.api_base, args.model, max_tokens=args.max_tokens)
    except Exception as exc:  # noqa: BLE001 -- the one line the master greps must still print
        record = {
            "backend": args.backend, "api_base": args.api_base, "model": args.model,
            "ok": False, "reason": f"{type(exc).__name__}: {exc}", "marker": THINKING_MARKER,
        }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if record["ok"]:
        print(
            f"THINKING_PIN_OK backend={args.backend} num_tokens={record.get('num_tokens')} "
            f"finish_reason={record.get('finish_reason')} text={record.get('generated_text')!r}"
        )
        return 0
    print(f"THINKING_PIN_FAILED backend={args.backend} reason={record['reason']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
