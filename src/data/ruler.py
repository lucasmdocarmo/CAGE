"""RULER-style synthetic length instrument for CAGE — charter D5 item 5.

Implements the needle-in-a-haystack (NIAH) retrieval task family from
Hsieh et al. (2024), "RULER: What's the Real Context Size of Your Long-Context
Language Models?", COLM 2024, arXiv:2404.06654 [hsieh2024ruler]: a synthetic
haystack of repeated noise sentences (RULER's canonical NIAH filler) with one
target needle sentence ("One of the special magic numbers for <key> is:
<value>.") inserted at a controlled depth, queried by a single retrieval
question. Task variants built here:

- ``niah_single``   — single needle, single query (the charter minimum).
- ``niah_multikey`` — the target needle plus ``num_distractors`` distractor
  needles with different keys/values (RULER's multi-key hardening); the query
  still targets exactly one key.

Charter conditions honored (D5#5 / §5.1):
- RULER is an INSTRUMENT, not a workload: payloads are GENERATED at exact
  controlled lengths (never downloaded), per-tokenizer regeneration is
  supported by injecting a token counter, and the emitted gold answer feeds
  RULER's native per-task string-match scoring downstream (never an
  aggregated mean).
- Hard 32,512-token INPUT cap (SHAPE-32K: 32,512 in + 256 out = 32,768
  total; the cap applies to ``context_length_tokens``, the input side);
  requesting more raises ``ValueError`` — fail-closed, never silent clipping.
- Fully deterministic: item ``i`` under seed ``s`` is identical across runs
  and independent of how many items are drawn (per-item child RNGs).

The loader is registered as dataset name ``"ruler"`` in
``src.data.loader.get_loader`` so the harness treats it like any dataset.
"""

from __future__ import annotations

import os
import random
from typing import Callable, List, Optional

from src.data.loader import CAGExample, DatasetLoader

# A callable counting tokens of a text under the model tokenizer of the arm
# being measured (charter: per-tokenizer regeneration). The default is an
# explicit whitespace-word proxy, recorded as such in every item's metadata.
TokenCounter = Callable[[str], int]

#: SHAPE-32K's reserved generation budget, exported as a per-item hint.
OUTPUT_TOKENS_HINT: int = 256
#: Charter total-shape pin (D5 §5.1 SHAPE-32K, PINNED 2026-08-02:
#: input 32,512 + output 256 = 32,768 total).
TOTAL_TOKENS_CAP: int = 32768
#: INPUT-side cap applied to ``context_length_tokens``: the pinned 32,768
#: total minus the reserved 256-token output budget. Capping the input at the
#: TOTAL would let a 32,768-token context + 256 output overshoot the pinned
#: shape by exactly the output budget at the headline top-pressure point.
MAX_CONTEXT_TOKENS: int = TOTAL_TOKENS_CAP - OUTPUT_TOKENS_HINT  # 32,512
#: Below this the haystack degenerates (needle + question no longer embedded
#: in meaningful noise), so the instrument refuses to generate.
MIN_CONTEXT_TOKENS: int = 64

#: RULER's canonical NIAH noise sentence (Hsieh et al. 2024, needle task
#: haystack "noise" variant).
NOISE_SENTENCE: str = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)

_KEY_WORDS: List[str] = [
    "amber", "basalt", "cobalt", "dune", "ember", "fjord", "garnet", "harbor",
    "iris", "juniper", "krypton", "lagoon", "meridian", "nimbus", "onyx",
    "prism", "quartz", "russet", "sierra", "topaz", "umber", "vertex",
    "willow", "xenon", "yarrow", "zephyr",
]

_TASKS = ("niah_single", "niah_multikey")


def _whitespace_token_counter(text: str) -> int:
    """Default token proxy: whitespace word count (recorded as such)."""
    return len(text.split())


class RulerLoader(DatasetLoader):
    """Deterministic seeded RULER-style NIAH generator exposed as a loader.

    Args:
        split: accepted for loader-interface parity; synthetic data has no
            splits (stored verbatim, recorded in metadata).
        seed: master seed; item ``i`` derives its own child RNG from
            ``(seed, i)`` so items are stable under any ``max_examples``.
        context_length_tokens: target context length under ``tokenizer``
            (default: env ``CAGE_RULER_CONTEXT_TOKENS``, else 4096). Must be
            in [MIN_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS]; the assembled
            haystack never exceeds the target.
        num_items: items generated when ``load(max_examples=None)``
            (default: env ``CAGE_RULER_NUM_ITEMS``, else 100).
        task: "niah_single" | "niah_multikey"
            (default: env ``CAGE_RULER_TASK``, else "niah_single").
        num_distractors: distractor needles for niah_multikey (default 3).
        tokenizer: token counter for the target model's tokenizer; defaults
            to the whitespace proxy (recorded in metadata as
            ``tokenizer_name="whitespace-proxy"``).
        tokenizer_name: label recorded in metadata when a real tokenizer is
            injected.
    """

    def __init__(
        self,
        split: str = "synthetic",
        seed: int = 42,
        context_length_tokens: Optional[int] = None,
        num_items: Optional[int] = None,
        task: Optional[str] = None,
        num_distractors: int = 3,
        tokenizer: Optional[TokenCounter] = None,
        tokenizer_name: Optional[str] = None,
    ):
        super().__init__("ruler", split, seed)

        if context_length_tokens is None:
            context_length_tokens = int(os.getenv("CAGE_RULER_CONTEXT_TOKENS", "4096"))
        if num_items is None:
            num_items = int(os.getenv("CAGE_RULER_NUM_ITEMS", "100"))
        if task is None:
            task = os.getenv("CAGE_RULER_TASK", "niah_single")

        if task not in _TASKS:
            raise ValueError(
                f"Unknown RULER task '{task}'. Supported: {list(_TASKS)} "
                f"(charter D5#5 subset; VT/QA tasks not yet built)."
            )
        if context_length_tokens > MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"context_length_tokens={context_length_tokens} exceeds the charter "
                f"INPUT cap of {MAX_CONTEXT_TOKENS} (D5 §5.1 SHAPE-32K: input "
                f"{MAX_CONTEXT_TOKENS} + output {OUTPUT_TOKENS_HINT} = "
                f"{TOTAL_TOKENS_CAP} total = 32768 tokens). Refusing to "
                f"generate — fail-closed, no silent clipping."
            )
        if context_length_tokens < MIN_CONTEXT_TOKENS:
            raise ValueError(
                f"context_length_tokens={context_length_tokens} is below the "
                f"minimum of {MIN_CONTEXT_TOKENS}; the haystack would degenerate."
            )
        if num_items < 1:
            raise ValueError(f"num_items must be >= 1, got {num_items}")
        if num_distractors < 1:
            raise ValueError(f"num_distractors must be >= 1, got {num_distractors}")

        self.context_length_tokens = context_length_tokens
        self.num_items = num_items
        self.task = task
        self.num_distractors = num_distractors
        if tokenizer is None:
            self._count = _whitespace_token_counter
            self.tokenizer_name = "whitespace-proxy"
        else:
            self._count = tokenizer
            self.tokenizer_name = tokenizer_name or "injected-tokenizer"

    # -- generation helpers -------------------------------------------------

    @staticmethod
    def _needle(key: str, value: int) -> str:
        # RULER NIAH needle template (Hsieh et al. 2024).
        return f"One of the special magic numbers for {key} is: {value}."

    @staticmethod
    def _question(key: str) -> str:
        # RULER NIAH retrieval query template (Hsieh et al. 2024).
        return (
            f"What is the special magic number for {key} mentioned in the "
            f"provided text?"
        )

    def _draw_key(self, rng: random.Random, taken: List[str]) -> str:
        """Draw a unique two-word key (RULER-style word keys)."""
        while True:
            key = f"{rng.choice(_KEY_WORDS)}-{rng.choice(_KEY_WORDS)}"
            if key not in taken:
                return key

    def _assemble(self, rng: random.Random, needles: List[str], depths: List[float]) -> str:
        """Assemble a haystack of noise sentences with needles at ``depths``.

        Greedy fill up to the token target, then trim noise (never needles)
        so the result NEVER exceeds ``context_length_tokens``.
        """
        per_noise = max(1, self._count(NOISE_SENTENCE + " "))
        needle_tokens = sum(self._count(n + " ") for n in needles)
        budget = self.context_length_tokens - needle_tokens
        n_noise = max(1, budget // per_noise)
        sentences = [NOISE_SENTENCE] * n_noise

        # Insert needles back-to-front so earlier indices stay valid.
        placements = sorted(
            ((d, needle) for d, needle in zip(depths, needles)),
            key=lambda p: p[0], reverse=True,
        )
        for depth, needle in placements:
            idx = round(depth * len(sentences))
            sentences.insert(min(idx, len(sentences)), needle)

        text = " ".join(sentences)
        # Trim trailing NOISE (never a needle) until within target.
        while self._count(text) > self.context_length_tokens and len(sentences) > len(needles):
            for i in range(len(sentences) - 1, -1, -1):
                if sentences[i] == NOISE_SENTENCE:
                    del sentences[i]
                    break
            else:  # pragma: no cover - only needles left
                break
            text = " ".join(sentences)
        return text

    def _generate_item(self, index: int) -> CAGExample:
        # Child RNG per (seed, index): items are identical across runs and
        # independent of how many items are drawn.
        rng = random.Random(self.seed * 1_000_003 + index)

        keys: List[str] = []
        target_key = self._draw_key(rng, keys)
        keys.append(target_key)
        target_value = rng.randint(1_000_000, 9_999_999)
        needles = [self._needle(target_key, target_value)]
        depths = [rng.random()]

        distractor_keys: List[str] = []
        if self.task == "niah_multikey":
            for _ in range(self.num_distractors):
                key = self._draw_key(rng, keys)
                keys.append(key)
                distractor_keys.append(key)
                needles.append(self._needle(key, rng.randint(1_000_000, 9_999_999)))
                depths.append(rng.random())

        haystack = self._assemble(rng, needles, depths)

        return CAGExample(
            id=f"ruler_{self.task}_{self.context_length_tokens}_{index:04d}",
            question=self._question(target_key),
            context=[haystack],
            answer=str(target_value),
            metadata={
                "dataset": "ruler",
                "task": self.task,
                "target_context_tokens": self.context_length_tokens,
                "actual_context_tokens": self._count(haystack),
                "needle_key": target_key,
                "needle_depth": depths[0],
                "distractor_keys": distractor_keys,
                "tokenizer_name": self.tokenizer_name,
                "seed": self.seed,
                "item_index": index,
                # SHAPE-32K reserves 256 output tokens (charter §5.1 item 1).
                "max_output_tokens_hint": OUTPUT_TOKENS_HINT,
                # Charter D5#5: score with RULER's native per-task string
                # match; never claim RULER accuracy = real answer quality.
                "native_metrics_only": True,
            },
        )

    # -- loader interface ---------------------------------------------------

    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Generate ``max_examples`` (default ``num_items``) deterministic items."""
        n = max_examples if max_examples else self.num_items
        return [self._generate_item(i) for i in range(n)]
