"""Shared corpus builder for true-CAG (Cache-Augmented Generation) cells.

True CAG (Chan et al. 2024, arXiv 2412.15605, "Don't Do RAG") precomputes the
KV cache of ONE fixed corpus block and answers every query against it. Both
engines that implement the cag_true cell -- the HF reference runner
(scripts/3_run/run_cag_reference.py) and the vLLM serving arm -- must assemble
the corpus text identically, byte for byte, or their KV caches (and therefore
their idea-gain / engine-gain decomposition) are not comparable. This module is
the single shared builder; its public API (CorpusBlock / build_corpus_block) is
a cross-workstream CONTRACT. Do not change the signatures without updating both
call sites.

Deliberately dependency-free (no torch / transformers / datasets): callers
inject the token counter -- the real serving tokenizer in production, a cheap
fake in unit tests -- so the module imports anywhere, including the lean
analysis venv.

Semantics (deterministic, prefix-greedy):
- Examples are consumed strictly in the given order.
- An example's gold paragraph(s) are ``example.context`` (for SQuAD: exactly
  one paragraph).
- Paragraphs are deduplicated by exact string equality across examples: five
  questions over one paragraph add that paragraph ONCE.
- Each included paragraph is formatted as ``Document {i}:\\n{paragraph}`` with
  1-based numbering, joined (after the header) by blank lines.
- Budget check is on the FULL assembled block text (header + document
  formatting included) under the provided counter. The first example whose new
  paragraph(s) would push the block over ``token_budget`` stops all further
  paragraph additions (prefix semantics -- no skip-and-fill), so the block
  never exceeds the budget.
- ``example_ids`` lists ALL examples whose gold paragraph(s) made it into the
  block, including later duplicates-by-paragraph encountered after the budget
  stop. An example with multiple gold paragraphs is included only if EVERY one
  of its non-empty paragraphs is in the block (all-or-nothing; for SQuAD this
  reduces to the single-paragraph case).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

DEFAULT_HEADER = "You are given the following reference documents:"


def default_token_counter(text: str) -> int:
    """Cheap deterministic heuristic: ~4/3 tokens per whitespace-split word."""
    return len(text.split()) * 4 // 3


def _assemble(header: str, paragraphs: Sequence[str]) -> str:
    """Assemble the corpus block text: header, then 1-based Document blocks."""
    parts = [header]
    for i, paragraph in enumerate(paragraphs, start=1):
        parts.append(f"Document {i}:\n{paragraph}")
    return "\n\n".join(parts)


@dataclass
class CorpusBlock:
    """The assembled true-CAG corpus and its provenance."""

    text: str                 # the assembled corpus block
    example_ids: List[str]    # ids of examples whose gold paragraph is IN the block
    paragraphs: List[str]     # the included paragraph strings, in block order
    token_count: int          # count of ``text`` under the provided counter


def build_corpus_block(
    examples: Sequence[Any],
    token_budget: int,
    count_tokens: Optional[Callable[[str], int]] = None,
    header: str = DEFAULT_HEADER,
) -> CorpusBlock:
    """Build the fixed corpus block for a true-CAG cell.

    Args:
        examples: loader QAExamples (``src.data.loader.CAGExample``-shaped:
            ``.id``, ``.question``, ``.context`` List[str], ``.answer``,
            ``.metadata``). Only ``.id`` and ``.context`` are read here.
        token_budget: maximum size of the assembled block text under
            ``count_tokens`` (e.g. 2800). Never exceeded.
        count_tokens: token counter applied to the assembled text. Inject the
            real serving tokenizer's counter in production so the corpus is
            tokenized exactly as served; defaults to the 4/3-words heuristic.
        header: leading instruction line of the block.

    Returns:
        CorpusBlock. Deterministic for identical inputs.
    """
    counter = count_tokens if count_tokens is not None else default_token_counter

    included_paragraphs: List[str] = []
    included_set: set = set()
    example_ids: List[str] = []
    budget_exhausted = False

    for example in examples:
        gold_paragraphs = [p for p in (getattr(example, "context", None) or []) if p]
        if not gold_paragraphs:
            # No gold paragraph -> nothing of this example can be "in the block".
            continue

        new_paragraphs = []
        for p in gold_paragraphs:
            if p not in included_set and p not in new_paragraphs:
                new_paragraphs.append(p)

        if not new_paragraphs:
            # Pure duplicate-by-paragraph: its gold paragraph(s) are already in
            # the block, so the example counts -- even after the budget stop.
            example_ids.append(example.id)
            continue

        if budget_exhausted:
            # Prefix semantics: once one example's paragraphs failed to fit, no
            # further NEW paragraphs are added (no skip-and-fill).
            continue

        candidate = included_paragraphs + new_paragraphs
        if counter(_assemble(header, candidate)) > token_budget:
            budget_exhausted = True
            continue

        included_paragraphs = candidate
        included_set.update(new_paragraphs)
        example_ids.append(example.id)

    text = _assemble(header, included_paragraphs)
    return CorpusBlock(
        text=text,
        example_ids=example_ids,
        paragraphs=list(included_paragraphs),
        token_count=counter(text),
    )
