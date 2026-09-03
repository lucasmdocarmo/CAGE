"""RULER-style synthetic length instrument for CAGE — charter D5 item 5.

Implements the charter's 4-task RULER subset (D5#5: "NIAH-MK/MQ, VT, QA; no
aggregation") from Hsieh et al. (2024), "RULER: What's the Real Context Size
of Your Long-Context Language Models?", COLM 2024, arXiv:2404.06654
[hsieh2024ruler]. Retrieval tasks build a synthetic haystack of repeated
noise sentences (RULER's canonical NIAH filler) with target sentence(s)
inserted at controlled depths. Task variants built here:

- ``niah_single``       — single needle, single query (the original charter
  minimum; kept registered: it is the loader default, the ``--ruler-task``
  CLI default path, and the pinned subject of the existing unit tests).
- ``niah_multikey``     — the target needle plus ``num_distractors``
  distractor needles with different keys/values (RULER's multi-key
  hardening); the query still targets exactly one key. NOTE: the Wave-4
  fixed-slot assembly relocates needles on MOST pre-Wave-4 multikey items
  (haystack bytes changed; ids/questions/answers/RNG draws unchanged) —
  disclosed and quantified in ``_assemble``, pinned by regression test.
- ``niah_multiquery``   — ``num_queries`` needles, ONE query asking for all
  of their values at once (RULER paper multi-query semantics); scored with
  RULER's ``string_match_all`` (mean containment over the needle set).
- ``variable_tracking`` — a chain of variable assignments (``VAR A = 42.``,
  ``VAR B = VAR A.`` …) whose hops are scattered in-order through the
  filler; the query asks for the FINAL binding of the last chain variable
  (charter task directive: answer = final binding). ``num_chains`` adds
  decoy chains with different values (RULER's VT hardening knob).
- ``qa``                — RULER-QA: a REAL QA pair embedded in a haystack of
  real distractor paragraphs. Source: SQuAD v2 via the existing
  ``SquadV2Loader`` (already staged locally; never downloaded here), the
  same source RULER's own qa_1 task uses; filtered to ANSWERABLE items only
  (containment scoring is meaningless for the unanswerable half). A custom
  ``qa_source`` callable may be injected (tests use tiny deterministic
  fixtures; per-run substitution must go through a real loader).

Native scoring (charter D5#5: RULER's native per-task metric, never an
aggregated mean) lives here too — ``score_native`` / ``score_example``
implement RULER's containment match (case-insensitive, as in the reference
implementation): ``string_match_all`` = mean containment over ALL gold
strings (niah_* and variable_tracking), ``string_match_part`` = max
containment over gold ALIASES (qa, where golds are alternative phrasings of
one answer). Every item records ``metadata["task"]``, ``["native_metric"]``
and ``["gold_answers"]`` so analysis can score and split per task.

Charter conditions honored (D5#5 / §5.1):
- RULER is an INSTRUMENT, not a workload: payloads are GENERATED at exact
  controlled lengths (never downloaded), per-tokenizer regeneration is
  supported by injecting a token counter, and the emitted gold answers feed
  the per-task string-match scoring above (never an aggregated mean).
- Hard 32,512-token INPUT cap (SHAPE-32K: 32,512 in + 256 out = 32,768
  total; the cap applies to ``context_length_tokens``, the input side);
  requesting more raises ``ValueError`` — fail-closed, never silent clipping.
- Fully deterministic: item ``i`` under seed ``s`` is identical across runs
  and independent of how many items are drawn (per-item child RNGs). The
  ``qa`` pool is a seeded-shuffle draw from the source loader, so it too is
  pinned by the loader seed.

The loader is registered as dataset name ``"ruler"`` in
``src.data.loader.get_loader`` so the harness treats it like any dataset.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Dict, List, Optional, Tuple

from src.data.loader import CAGExample, DatasetLoader, SquadV2Loader

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

#: Charter D5#5 subset (NIAH-MK/MQ + VT + QA) plus ``niah_single``, which
#: stays registered as the loader/CLI default and the existing tests' pin.
_TASKS = (
    "niah_single",
    "niah_multikey",
    "niah_multiquery",
    "variable_tracking",
    "qa",
)

#: RULER's native per-task metric (reference implementation naming):
#: ``string_match_all`` averages containment over ALL gold strings (every
#: gold must appear for full credit); ``string_match_part`` takes the MAX
#: over golds (golds are aliases of one answer — any one suffices). Exported
#: so analysis reports per task and never a cross-task mean (charter D5#5).
NATIVE_METRICS: Dict[str, str] = {
    "niah_single": "string_match_all",
    "niah_multikey": "string_match_all",
    "niah_multiquery": "string_match_all",
    "variable_tracking": "string_match_all",
    "qa": "string_match_part",
}

#: Size of the seeded-shuffle draw from SQuAD v2 that backs the ``qa`` task's
#: gold+distractor pool. ~2000 answerable items yield >100k tokens of unique
#: distractor paragraphs — comfortably above the 32,512 SHAPE-32K input cap —
#: while keeping pool construction fast.
_QA_POOL_EXAMPLES: int = 2000


def _whitespace_token_counter(text: str) -> int:
    """Default token proxy: whitespace word count (recorded as such)."""
    return len(text.split())


class RulerLoader(DatasetLoader):
    """Deterministic seeded RULER-style generator exposed as a loader.

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
        task: one of ``_TASKS``
            (default: env ``CAGE_RULER_TASK``, else "niah_single").
        num_distractors: distractor needles for niah_multikey (default 3).
        num_queries: needles queried at once by niah_multiquery (default 4,
            the RULER paper setting; must be >= 2 — one query IS
            niah_single, refuse the degenerate alias).
        num_hops: assignment hops per variable_tracking chain (default 4,
            the RULER paper setting; chain length = num_hops + 1 variables).
        num_chains: variable_tracking chains (default 1 as in RULER; chains
            beyond the first are decoys with different values).
        qa_source: zero-arg callable returning the ``qa`` task's candidate
            pool as ``CAGExample``s (question/context/answer + the loader
            ``all_answers``/``is_impossible`` metadata keys). Default: a
            seeded-shuffle draw of ``_QA_POOL_EXAMPLES`` items from SQuAD v2
            validation via ``SquadV2Loader`` (already-local; RULER's own
            qa_1 source).
        qa_source_name: label recorded in ``qa`` metadata when a custom
            ``qa_source`` is injected (default source records
            ``"squad_v2:validation"``).
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
        num_queries: int = 4,
        num_hops: int = 4,
        num_chains: int = 1,
        qa_source: Optional[Callable[[], List[CAGExample]]] = None,
        qa_source_name: Optional[str] = None,
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
                f"(charter D5#5 subset: NIAH-MK/MQ + VT + QA, plus the "
                f"niah_single default)."
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
        if num_queries < 2:
            raise ValueError(
                f"num_queries must be >= 2 for niah_multiquery, got "
                f"{num_queries}; a one-needle query IS niah_single — use that "
                f"task instead of a degenerate multiquery."
            )
        if num_hops < 1:
            raise ValueError(f"num_hops must be >= 1, got {num_hops}")
        if num_chains < 1:
            raise ValueError(f"num_chains must be >= 1, got {num_chains}")

        self.context_length_tokens = context_length_tokens
        self.num_items = num_items
        self.task = task
        self.num_distractors = num_distractors
        self.num_queries = num_queries
        self.num_hops = num_hops
        self.num_chains = num_chains
        self._qa_source = qa_source
        if qa_source is None:
            self.qa_source_name = "squad_v2:validation"
        else:
            self.qa_source_name = qa_source_name or "injected-qa-source"
        # Lazily built once per loader (deterministic under seed); tuple of
        # (answerable pool, ordered-deduped distractor paragraphs).
        self._qa_pool_cache: Optional[Tuple[List[CAGExample], List[str]]] = None
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

    @staticmethod
    def _multiquery_question(keys: List[str]) -> str:
        # RULER multi-query NIAH: ONE query covering every needle key.
        listed = f"{', '.join(keys[:-1])}, and {keys[-1]}" if len(keys) > 2 else " and ".join(keys)
        return (
            f"What are the special magic numbers for {listed} mentioned in "
            f"the provided text?"
        )

    def _draw_key(self, rng: random.Random, taken: List[str]) -> str:
        """Draw a unique two-word key (RULER-style word keys)."""
        while True:
            key = f"{rng.choice(_KEY_WORDS)}-{rng.choice(_KEY_WORDS)}"
            if key not in taken:
                return key

    def _draw_variable(self, rng: random.Random, taken: List[str]) -> str:
        """Draw a unique uppercase variable name (RULER-style VT variables)."""
        while True:
            name = f"{rng.choice(_KEY_WORDS).upper()}{rng.choice(_KEY_WORDS).upper()}"
            if name not in taken:
                return name

    def _assemble(self, rng: random.Random, needles: List[str], depths: List[float]) -> str:
        """Assemble a haystack of noise sentences with needles at ``depths``.

        Greedy fill up to the token target, then trim noise (never needles)
        so the result NEVER exceeds ``context_length_tokens``. Needles whose
        depths are ordered come out in that order — GUARANTEED, at every
        context length: every slot is computed against the fixed noise count
        BEFORE any needle is placed (``round`` is monotone in depth; slot
        ties break by depth, then listing order), so two depths can never
        invert no matter how close they sit. ``variable_tracking``'s
        in-order chain invariant rides on this.

        DISCLOSED BEHAVIOR CHANGE (Wave 4): the pre-Wave-4 assembly inserted
        needles back-to-front into a list that GREW during insertion, so each
        slot depended on how many needles were already placed. The fixed-slot
        rewrite therefore moves needles by ~1 noise sentence on MOST
        pre-existing multi-needle items — measured on ``niah_multikey`` under
        seed 42: 57/60 haystacks differ at ctx=256, 54/60 at 1024, 53/60 at
        4096 — while ids, questions, answers, gold_answers and every RNG draw
        are unchanged. Single-needle assembly (``niah_single``) is
        byte-identical at every depth (one placement cannot see a grown
        list). No frozen artifact pins the old bytes; both halves of this
        split are regression-pinned in ``tests/test_ruler_tasks.py``.
        """
        per_noise = max(1, self._count(NOISE_SENTENCE + " "))
        needle_tokens = sum(self._count(n + " ") for n in needles)
        budget = self.context_length_tokens - needle_tokens
        n_noise = max(1, budget // per_noise)

        # Slot every needle against the FIXED n_noise up front. Recomputing
        # ``round(depth * len(sentences))`` against a list that grows during
        # insertion (the previous scheme) let depths closer than ~1/n_noise
        # land out of order: the shallower needle's index, scaled by the
        # already-grown list, could overshoot the deeper needle's slot.
        slotted = sorted(
            (min(round(depth * n_noise), n_noise), depth, j)
            for j, depth in enumerate(depths)
        )
        sentences: List[str] = []
        pos = 0
        for slot in range(n_noise + 1):
            while pos < len(slotted) and slotted[pos][0] == slot:
                sentences.append(needles[slotted[pos][2]])
                pos += 1
            if slot < n_noise:
                sentences.append(NOISE_SENTENCE)

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

    def _assemble_qa(
        self,
        rng: random.Random,
        gold_paragraph: str,
        depth: float,
        distractors: List[str],
    ) -> Tuple[str, int]:
        """Assemble a qa haystack: real distractor paragraphs + the gold one.

        Distractors are drawn without replacement (shuffled under the item
        RNG) up to the token target, the gold paragraph is inserted at
        ``depth``, and trailing DISTRACTORS (never the gold) are trimmed so
        the result NEVER exceeds ``context_length_tokens``. Fail-closed on a
        gold paragraph that cannot fit and on a haystack with zero
        distractors (no "long distractor haystack" would exist).

        Returns ``(haystack_text, num_distractor_paragraphs)``.
        """
        budget = self.context_length_tokens
        gold_cost = self._count(gold_paragraph)
        if gold_cost >= budget:
            raise ValueError(
                f"qa gold paragraph ({gold_cost} tokens under "
                f"'{self.tokenizer_name}') does not fit "
                f"context_length_tokens={budget}. Refusing to generate — "
                f"fail-closed, the gold evidence is never clipped; raise "
                f"context_length_tokens (cap {MAX_CONTEXT_TOKENS})."
            )

        pool = list(distractors)
        rng.shuffle(pool)
        chosen: List[str] = []
        used = gold_cost
        for paragraph in pool:
            cost = self._count(paragraph)
            if used + cost > budget:
                continue  # smaller paragraphs later in the pool may still fit
            chosen.append(paragraph)
            used += cost
            if used >= budget:
                break
        if not chosen:
            raise ValueError(
                f"qa haystack would contain ZERO distractor paragraphs at "
                f"context_length_tokens={budget}: no qa_source paragraph fits "
                f"beside the {gold_cost}-token gold. Refusing to generate — "
                f"raise context_length_tokens or supply a qa_source with "
                f"shorter paragraphs."
            )

        idx = min(round(depth * len(chosen)), len(chosen))
        chosen.insert(idx, gold_paragraph)
        text = "\n\n".join(chosen)
        # Separator/tokenizer slack: trim trailing DISTRACTORS (never gold).
        while self._count(text) > budget and len(chosen) > 1:
            for i in range(len(chosen) - 1, -1, -1):
                if chosen[i] != gold_paragraph:
                    del chosen[i]
                    break
            else:  # pragma: no cover - only the gold left
                break
            text = "\n\n".join(chosen)
        return text, len(chosen) - 1

    def _qa_pool(self) -> Tuple[List[CAGExample], List[str]]:
        """Build (once) the qa task's answerable pool + distractor paragraphs.

        Filters the source to ANSWERABLE items with a non-empty single-string
        context (RULER-QA scores gold-answer containment; the unanswerable
        half of SQuAD v2 has no containable gold). Fail-closed when the
        filtered pool cannot form gold + at least one distinct distractor.
        """
        if self._qa_pool_cache is None:
            if self._qa_source is None:
                # Seeded-shuffle draw (SquadV2Loader shuffles under our seed
                # before selecting) — deterministic under the loader seed,
                # already-local dataset, never downloaded here.
                raw = SquadV2Loader(split="validation", seed=self.seed).load(
                    max_examples=_QA_POOL_EXAMPLES
                )
            else:
                raw = self._qa_source()
            pool = [
                ex for ex in raw
                if not (ex.metadata or {}).get("is_impossible")
                and ex.answer
                and ex.context
                and ex.context[0].strip()
            ]
            # Ordered dedupe: SQuAD repeats one paragraph across many
            # questions; distractor draws need distinct text.
            paragraphs = list(dict.fromkeys(ex.context[0] for ex in pool))
            if len(pool) < 2 or len(paragraphs) < 2:
                raise ValueError(
                    f"qa_source '{self.qa_source_name}' yields "
                    f"{len(pool)} answerable example(s) over "
                    f"{len(paragraphs)} distinct paragraph(s); the qa task "
                    f"needs >= 2 of each (gold + at least one distractor). "
                    f"Refusing to generate — fix the source, do not thin the "
                    f"haystack silently."
                )
            self._qa_pool_cache = (pool, paragraphs)
        return self._qa_pool_cache

    # -- per-task item builders --------------------------------------------

    def _metadata(self, haystack: str, index: int, gold_answers: List[str], **task_fields) -> dict:
        """Metadata shared by every task + per-task reporting fields.

        ``task`` / ``native_metric`` / ``gold_answers`` are the per-task
        scoring contract consumed by ``score_example`` and by analysis'
        split-by-task reporting (charter D5#5: never an aggregated mean).
        """
        md = {
            "dataset": "ruler",
            "task": self.task,
            "target_context_tokens": self.context_length_tokens,
            "actual_context_tokens": self._count(haystack),
            "tokenizer_name": self.tokenizer_name,
            "seed": self.seed,
            "item_index": index,
            # SHAPE-32K reserves 256 output tokens (charter §5.1 item 1).
            "max_output_tokens_hint": OUTPUT_TOKENS_HINT,
            # Charter D5#5: score with RULER's native per-task string
            # match; never claim RULER accuracy = real answer quality.
            "native_metrics_only": True,
            "native_metric": NATIVE_METRICS[self.task],
            "gold_answers": gold_answers,
        }
        md.update(task_fields)
        return md

    def _item_niah(self, rng: random.Random, index: int) -> CAGExample:
        """niah_single / niah_multikey (RNG draw order preserved verbatim).

        Draw preservation keeps ids/questions/answers/gold_answers byte-
        stable vs pre-Wave-4; haystack BYTES are unchanged only for
        niah_single — multikey needle placement moved with the fixed-slot
        ``_assemble`` (disclosed there).
        """
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
            metadata=self._metadata(
                haystack, index, [str(target_value)],
                needle_key=target_key,
                needle_depth=depths[0],
                distractor_keys=distractor_keys,
            ),
        )

    def _item_multiquery(self, rng: random.Random, index: int) -> CAGExample:
        """niah_multiquery: ``num_queries`` needles, ONE query for all values."""
        keys: List[str] = []
        values: List[int] = []
        needles: List[str] = []
        depths: List[float] = []
        for _ in range(self.num_queries):
            key = self._draw_key(rng, keys)
            keys.append(key)
            value = rng.randint(1_000_000, 9_999_999)
            values.append(value)
            needles.append(self._needle(key, value))
            depths.append(rng.random())

        haystack = self._assemble(rng, needles, depths)
        gold_answers = [str(v) for v in values]

        return CAGExample(
            id=f"ruler_{self.task}_{self.context_length_tokens}_{index:04d}",
            question=self._multiquery_question(keys),
            context=[haystack],
            # Human-readable joined form; scoring uses metadata["gold_answers"]
            # (string_match_all: EVERY value must appear). Deliberately NOT
            # exported as metadata["all_answers"] — that key means
            # max-over-gold ALIASES downstream, which would grant full credit
            # for one value out of num_queries.
            answer=", ".join(gold_answers),
            metadata=self._metadata(
                haystack, index, gold_answers,
                needle_keys=keys,
                needle_depths=depths,
                num_queries=self.num_queries,
            ),
        )

    def _item_variable_tracking(self, rng: random.Random, index: int) -> CAGExample:
        """variable_tracking: in-order assignment hops; answer = final binding."""
        taken: List[str] = []
        target_value = rng.randint(1_000_000, 9_999_999)
        chain_variables: List[str] = []
        decoy_variables: List[str] = []
        statements: List[str] = []
        depths: List[float] = []

        for chain_idx in range(self.num_chains):
            if chain_idx == 0:
                value = target_value
            else:
                value = rng.randint(1_000_000, 9_999_999)
                while value == target_value:  # decoys must not alias the gold
                    value = rng.randint(1_000_000, 9_999_999)
            names = []
            for _ in range(self.num_hops + 1):
                name = self._draw_variable(rng, taken)
                taken.append(name)
                names.append(name)
            if chain_idx == 0:
                chain_variables = names
            else:
                decoy_variables.extend(names)
            chain_statements = [f"VAR {names[0]} = {value}."] + [
                f"VAR {names[i]} = VAR {names[i - 1]}." for i in range(1, len(names))
            ]
            # Ascending depths per chain: hops appear IN ORDER in the text so
            # the final binding is traceable left-to-right.
            chain_depths = sorted(rng.random() for _ in chain_statements)
            statements.extend(chain_statements)
            depths.extend(chain_depths)

        haystack = self._assemble(rng, statements, depths)
        query_variable = chain_variables[-1]

        return CAGExample(
            id=f"ruler_{self.task}_{self.context_length_tokens}_{index:04d}",
            question=(
                f"What is the final numeric value of variable {query_variable} "
                f"after following the chain of variable assignments in the "
                f"provided text?"
            ),
            context=[haystack],
            answer=str(target_value),
            metadata=self._metadata(
                haystack, index, [str(target_value)],
                chain_variables=chain_variables,
                query_variable=query_variable,
                num_hops=self.num_hops,
                num_chains=self.num_chains,
                decoy_variables=decoy_variables,
            ),
        )

    def _item_qa(self, rng: random.Random, index: int) -> CAGExample:
        """qa: real SQuAD-v2 pair inside a real-paragraph distractor haystack."""
        pool, paragraphs = self._qa_pool()
        source = pool[rng.randrange(len(pool))]
        gold_paragraph = source.context[0]
        depth = rng.random()
        distractors = [p for p in paragraphs if p != gold_paragraph]
        haystack, num_distractor_paragraphs = self._assemble_qa(
            rng, gold_paragraph, depth, distractors
        )

        # Gold ALIASES (string_match_part: any one suffices), primary first.
        aliases = (source.metadata or {}).get("all_answers") or []
        gold_answers = list(dict.fromkeys([source.answer] + [a for a in aliases if a]))

        return CAGExample(
            id=f"ruler_{self.task}_{self.context_length_tokens}_{index:04d}",
            question=source.question,
            context=[haystack],
            answer=source.answer,
            metadata=self._metadata(
                haystack, index, gold_answers,
                qa_source=self.qa_source_name,
                qa_source_id=source.id,
                gold_depth=depth,
                num_distractor_paragraphs=num_distractor_paragraphs,
                # Alias semantics DO apply here, so the downstream
                # max-over-golds key is exported (same key SQuAD v2 emits).
                all_answers=gold_answers,
            ),
        )

    def _generate_item(self, index: int) -> CAGExample:
        # Child RNG per (seed, index): items are identical across runs and
        # independent of how many items are drawn.
        rng = random.Random(self.seed * 1_000_003 + index)
        if self.task in ("niah_single", "niah_multikey"):
            return self._item_niah(rng, index)
        if self.task == "niah_multiquery":
            return self._item_multiquery(rng, index)
        if self.task == "variable_tracking":
            return self._item_variable_tracking(rng, index)
        if self.task == "qa":
            return self._item_qa(rng, index)
        raise ValueError(  # pragma: no cover - constructor already validates
            f"Unknown RULER task '{self.task}'. Supported: {list(_TASKS)}."
        )

    # -- loader interface ---------------------------------------------------

    def load(self, max_examples: Optional[int] = None) -> List[CAGExample]:
        """Generate ``max_examples`` (default ``num_items``) deterministic items."""
        n = max_examples if max_examples else self.num_items
        return [self._generate_item(i) for i in range(n)]


# -- native per-task scoring (charter D5#5: RULER's own metric, per task) ----


def score_native(task: str, gold_answers: List[str], prediction: str) -> float:
    """RULER's native containment match for one prediction, per task.

    Case-insensitive substring containment (the RULER reference convention):
    ``string_match_all`` (niah_*, variable_tracking) returns the MEAN
    containment over ``gold_answers`` — every gold must appear for 1.0;
    ``string_match_part`` (qa) returns the MAX — golds are aliases and any
    one suffices. Fail-closed on an unknown task, an empty gold list, or a
    non-string prediction (an ABSENT prediction must stay absent upstream —
    None is never scored as 0.0).
    """
    if task not in NATIVE_METRICS:
        raise ValueError(
            f"Unknown RULER task '{task}' for native scoring. Supported: "
            f"{list(NATIVE_METRICS)}."
        )
    if not gold_answers:
        raise ValueError(
            f"score_native(task='{task}') got an empty gold_answers list; a "
            f"gold-free item is a generator bug, not a 0.0 — refusing to score."
        )
    if not isinstance(prediction, str):
        raise ValueError(
            f"score_native(task='{task}') needs a str prediction, got "
            f"{type(prediction).__name__}; absence stays absence — score only "
            f"real model outputs."
        )
    pred = prediction.lower()
    hits = [1.0 if str(gold).lower() in pred else 0.0 for gold in gold_answers]
    if NATIVE_METRICS[task] == "string_match_part":
        return max(hits)
    return sum(hits) / len(hits)


def score_example(example: CAGExample, prediction: str) -> float:
    """Score a prediction against a RulerLoader item's metadata contract."""
    md = example.metadata or {}
    task = md.get("task")
    gold_answers = md.get("gold_answers")
    if task is None or gold_answers is None:
        raise ValueError(
            f"Example '{example.id}' lacks the RULER scoring contract "
            f"(metadata['task'] + metadata['gold_answers']); only items "
            f"emitted by RulerLoader are scorable here."
        )
    return score_native(task, gold_answers, prediction)
