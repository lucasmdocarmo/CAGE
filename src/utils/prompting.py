"""Prompt formatting helpers.

Centralizes prompt templates so that baselines (CAG/RAG/Hybrid) build prompts consistently.

Notes on prefix-caching:
- vLLM prefix caching benefits when many requests share a long common prefix.
- Keeping a stable, shared system prefix can help make caching effects measurable.
"""

from __future__ import annotations

from typing import Iterable, Sequence


DEFAULT_SYSTEM_PREFIX = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "Give a SHORT, direct answer with no explanation or step-by-step reasoning. "
    "If the context is insufficient, say you don't know.\n\n"
)


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
