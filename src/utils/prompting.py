"""Prompt formatting helpers.

Centralizes prompt templates so that baselines (CAG/RAG/Hybrid) build prompts consistently.

Notes on prefix-caching:
- vLLM prefix caching benefits when many requests share a long common prefix.
- Keeping a stable, shared system prefix can help make caching effects measurable.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_SYSTEM_PREFIX = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "Give a SHORT, direct answer with no explanation or step-by-step reasoning. "
    "If the context is insufficient, say you don't know.\n\n"
)

# Decision 2A (approved pre-run package, 2026-07-16): ONE explicit abstention
# instruction, byte-identical across ALL arms. The exact reply token
# "unanswerable" is caught by src.evaluation.quality.is_no_answer_prediction
# (via the _NO_ANSWER_RE word-boundary pattern on short answers).
ABSTAIN_INSTRUCTION = (
    "If the answer cannot be found in the context, reply exactly: unanswerable"
)

# Decision 1B: chat-template serving. The system message carries the task
# instruction (incl. the shared abstention instruction); the user message
# carries context first, question last, so prompts sharing a context share the
# longest possible rendered-token prefix (prefix-cache locality preserved).
CHAT_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "Give a SHORT, direct answer with no explanation or step-by-step reasoning. "
    + ABSTAIN_INSTRUCTION
)


def prompt_mode(env: Dict[str, str] | None = None) -> str:
    """Resolve the serving prompt mode (Decision 1B).

    CAGE_PROMPT_MODE = "chat" (default): serve via the model's chat template
    (/v1/chat/completions on vLLM; tokenizer.apply_chat_template on the
    reference engine). "raw" is the escape hatch that reproduces the legacy
    raw-completions path byte-for-byte.
    """
    environ = env if env is not None else os.environ
    mode = (environ.get("CAGE_PROMPT_MODE") or "chat").strip().lower()
    if mode not in {"chat", "raw"}:
        raise ValueError(f"CAGE_PROMPT_MODE must be 'chat' or 'raw', got {mode!r}")
    return mode


def format_context_blocks(contexts: Sequence[str]) -> str:
    """Format a list of context strings into numbered blocks."""
    blocks = []
    for i, c in enumerate(contexts):
        if not c:
            continue
        blocks.append(f"Context {i+1}: {c}")
    return "\n\n".join(blocks)


def format_qa_prompt(
    question: str,
    contexts: Sequence[str] | None = None,
    *,
    system_prefix: str | None = DEFAULT_SYSTEM_PREFIX,
) -> str:
    """Build a simple QA prompt with optional context blocks."""
    parts: list[str] = []

    if system_prefix:
        parts.append(system_prefix.rstrip() + "\n")

    if contexts:
        ctx = format_context_blocks(contexts)
        if ctx:
            parts.append(ctx)
            parts.append("\n\n")

    parts.append(f"Question: {question}\nAnswer:")
    return "".join(parts)


def format_multi_turn_prompt(
    question: str,
    contexts: Sequence[str] | None = None,
    *,
    history: Sequence[tuple[str, str]] | None = None,
    system_prefix: str | None = DEFAULT_SYSTEM_PREFIX,
) -> str:
    """Build a multi-turn conversational prompt with optional history."""
    parts: list[str] = []

    if system_prefix:
        parts.append(system_prefix.rstrip() + "\n")

    if contexts:
        ctx = format_context_blocks(contexts)
        if ctx:
            parts.append(ctx)
            parts.append("\n\n")

    for user_q, assistant_a in (history or []):
        parts.append(f"User: {user_q}\n")
        parts.append(f"Assistant: {assistant_a}\n")

    parts.append(f"User: {question}\nAssistant:")
    return "".join(parts)


def _qa_user_content(question: str, contexts: Sequence[str] | None) -> str:
    """User-message content: context blocks FIRST, question LAST.

    Mirrors the raw layout's slot order so the shared corpus/context prefix
    stays at the front of the rendered prompt (prefix sharing preserved).
    """
    parts: list[str] = []
    if contexts:
        ctx = format_context_blocks(contexts)
        if ctx:
            parts.append(ctx)
            parts.append("\n\n")
    parts.append(f"Question: {question}")
    return "".join(parts)


def format_qa_messages(
    question: str,
    contexts: Sequence[str] | None = None,
    *,
    system_instruction: str = CHAT_SYSTEM_INSTRUCTION,
) -> List[Dict[str, str]]:
    """Chat-template QA messages (Decision 1B).

    [system: task instruction + abstention instruction (Decision 2A, identical
    across arms), user: context blocks first + question last].
    """
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": _qa_user_content(question, contexts)},
    ]


def format_multi_turn_messages(
    question: str,
    contexts: Sequence[str] | None = None,
    *,
    history: Sequence[tuple[str, str]] | None = None,
    system_instruction: str = CHAT_SYSTEM_INSTRUCTION,
) -> List[Dict[str, str]]:
    """Chat-template multi-turn messages (Decision 1B).

    History becomes proper alternating user/assistant turns after the system
    message (system+history form the shared cacheable prefix); the CURRENT
    turn's context+question go in the final user message, context first,
    question last.
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_instruction}
    ]
    for user_q, assistant_a in (history or []):
        messages.append({"role": "user", "content": user_q})
        messages.append({"role": "assistant", "content": assistant_a or ""})
    messages.append({"role": "user", "content": _qa_user_content(question, contexts)})
    return messages


def messages_to_fallback_prompt(messages: Sequence[Dict[str, str]]) -> str:
    """Flatten chat messages to one raw text prompt.

    Used only as the InferenceRequest.prompt fallback for backends that do not
    understand messages (and for logging); the vLLM adapter serves the actual
    messages via /v1/chat/completions. Content (incl. the abstention
    instruction) is identical to the chat form; only the framing differs.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            parts.append(content)
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    return "\n\n".join(parts) + "\nAnswer:"


def select_distractor_texts(
    pool_examples: Sequence[Any],
    exclude_texts: Iterable[str],
    n: int,
) -> List[str]:
    """Deterministic distractor paragraphs for retrieval-corpus widening (Decision 3B).

    Walks the full-split pool in load order (seed-stable: the manifest path
    loads the split without shuffling), collects each example's context
    paragraphs deduplicated by content (sha1, matching
    src.orchestration.ir.stable_text_id), EXCLUDES the trial's gold paragraphs,
    and returns the first ``n`` survivors. Lives here (pure stdlib) so the lean
    analysis venv can unit-test it without the runner's serving dependencies;
    scripts/3_run/run_experiment.py wraps the texts into IRDocuments.
    """
    if n <= 0:
        return []

    def _text_id(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    excluded = {_text_id(t) for t in exclude_texts if t}
    seen: set[str] = set()
    out: List[str] = []
    for ex in pool_examples:
        for ctx in (getattr(ex, "context", None) or []):
            if not ctx:
                continue
            text = str(ctx)
            doc_id = _text_id(text)
            if doc_id in excluded or doc_id in seen:
                continue
            seen.add(doc_id)
            out.append(text)
            if len(out) >= n:
                return out
    return out


def extract_cacheable_prefix_text(prompt: str) -> str:
    """Return the shared prompt prefix that should drive routing/cache locality.

    The cacheable prefix is the portion of the prompt before the current user
    query/question text. This preserves locality for requests that share the
    same system/context/history prefix but ask different final questions.
    """
    markers = ("\nUser: ", "User: ", "\nQuestion: ", "Question: ")
    for marker in markers:
        idx = prompt.rfind(marker)
        if idx != -1:
            return prompt[: idx + len(marker)]
    return prompt
