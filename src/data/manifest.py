"""Uniform-yardstick query manifest (2026-07-15).

One seeded, auditable artifact per (dataset, N, T, seed) that pre-draws the measured
query set for EVERY cell, tree, engine, and model, so per-query pairing holds
universally and no script can drift to its own sample ("different numbers for
different things"). This is the QA-benchmark analogue of the serving literature's
workload-trace file, and the corpus-first construction is exactly Chan et al.'s CAG
evaluation design (fixed document tiers; the test questions ARE the in-corpus
questions; arXiv 2412.15605).

Construction (corpus-first):
  1. Order examples with same-paragraph questions adjacent, paragraph groups in
     seeded-random order.
  2. Iteratively pack paragraphs into corpus BLOCKS of <= block_budget tokens
     (reusing the tested ``build_corpus_block``); an example whose paragraph alone
     exceeds the budget is EXCLUDED and counted (the auditable exclusion rate).
  3. Stop when the in-corpus question pool reaches ``pool_target``
     (default max(3N, N*T)).
  4. Per trial t: seeded draw (seed + t - 1) of N question ids, WITHOUT replacement
     across trials when the pool allows (trial independence by construction).

The manifest stores block TEXTS verbatim so every engine serves byte-identical
corpus prompts with no tokenizer dependency.

Overlap (D5 F1/F3, charter: "overlap ENGINEERED and reported (never assumed)"):
every manifest carries a MEASURED ``overlap`` field (src/data/overlap.py) --
realized mean pairwise shared-paragraph fraction and shared-prefix-token
estimate per corpus block -- in natural mode too, so natural overlap is
reported, never assumed. Passing ``overlap_target`` switches step 1-2 to the
seeded engineered-overlap planner (one planned query group per block) and
gates fail-closed: a target the corpus cannot realize raises ``OverlapError``
naming realized vs target, never silent best-effort.

Corpus-truncation rungs (B12, charter §7.7(d), ADR-0106, owner decision
2026-09-16): ``trunc_budgets`` derives, from the SAME packed blocks (same seed,
same packing order, no repacking), one descending rung per budget b <
block_budget: each block keeps its paragraphs in packing order until b would
be exceeded, and every pool query is labeled in-corpus (its gold paragraph
survived in its own block) or out-of-corpus BY CONSTRUCTION. Out-of-corpus
queries are SERVED against the rung block (expected abstention), never
dropped. The full block_budget point of the ladder is B3's own cell; it is
never a rung. Written as ``manifest["trunc_rungs"]`` keyed by the rung budget.

Pure stdlib + src.data.corpus: importable (and unit-testable) without the
``datasets`` package, torch, or a GPU.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from src.data.corpus import DEFAULT_HEADER, _assemble, build_corpus_block, default_token_counter
from src.data.overlap import (  # re-exported: consumers import from either module
    OverlapError,
    enforce_overlap_target,
    gold_set,
    group_overlap_stats,
    overlap_report,
    plan_overlap_groups,
)

# v2 (2026-09-02): + top-level "overlap" (measured, both modes)
# v3 (2026-09-16, ADR-0106): + top-level "trunc_rungs" (B12 ladder; {} when none)
MANIFEST_VERSION = 3


class ManifestError(ValueError):
    """Raised when a manifest cannot be built or applied as specified."""


def _ctx_key(example: Any) -> tuple:
    return tuple(example.context or [])


def build_manifest(
    examples: Sequence[Any],
    num_queries: int,
    num_trials: int,
    seed: int,
    block_budget: int = 2800,
    pool_target: Optional[int] = None,
    dataset: str = "",
    split: str = "",
    context_selector: Optional[Callable[[Any], List[str]]] = None,
    overlap_target: Optional[float] = None,
    overlap_tolerance: float = 0.05,
    overlap_group_size: int = 4,
    trunc_budgets: Tuple[int, ...] = (),
) -> Dict[str, Any]:
    """Build the manifest dict from loader examples (.id/.question/.context/.answer).

    ``context_selector``, if given, maps each example to the paragraph list that is
    actually shared-corpus material (default: ``example.context`` unchanged -- the
    SQuAD-shaped "context IS the gold paragraph" case). Loaders that keep BOTH gold
    and distractor paragraphs in ``.context`` (HotpotQA, MuSiQue: gold recoverable via
    ``metadata["supporting_titles"]``) must pass a selector such as
    ``src.data.loader.gold_only`` here -- otherwise unique-per-question distractor
    text pollutes the shared block budget and same-paragraph-questions grouping keys
    off it too, degenerating every block to ~1 example and defeating the cross-
    question corpus reuse the manifest exists to measure. Applied FIRST, before
    grouping/packing, so every downstream step (grouping, packing, exclusion) is
    consistently gold-based.

    ``overlap_target`` (D5 F1/F3 engineered-overlap store), if given, replaces the
    natural shuffle-and-pack with the seeded planner (src/data/overlap.py): query
    groups are composed to a target mean pairwise shared-paragraph fraction, one
    planned group per corpus block, then the PACKED blocks are measured and gated
    fail-closed against ``overlap_tolerance`` (``OverlapError`` on an unreachable
    target). ``overlap_group_size`` caps questions per engineered group. In both
    modes the realized overlap is measured and written to ``manifest["overlap"]``.

    ``trunc_budgets`` (B12 ladder, ADR-0106): strictly descending integer rung
    budgets, each < ``block_budget``; validated fail-closed BEFORE packing. The
    rungs are derived from the packed blocks by ``derive_trunc_rungs`` and
    written to ``manifest["trunc_rungs"]`` ({} when no ladder is requested).
    """
    if num_queries < 1 or num_trials < 1:
        raise ManifestError("num_queries and num_trials must be >= 1")
    validate_trunc_budgets(trunc_budgets, block_budget)
    target = pool_target or max(3 * num_queries, num_queries * num_trials)

    if context_selector is not None:
        examples = [
            dataclasses.replace(ex, context=context_selector(ex)) for ex in examples
        ]

    blocks: List[Dict[str, Any]] = []
    block_paragraphs: List[List[str]] = []  # per block, for overlap measurement
    question_to_block: Dict[str, int] = {}
    excluded: List[str] = []

    if overlap_target is not None:
        # Engineered-overlap store: seeded planner composes the query groups
        # (validates the knobs fail-closed), then each planned group packs into
        # its OWN block -- the group IS the store unit contrast #17 pressures.
        planned, excluded = plan_overlap_groups(
            examples,
            target=overlap_target,
            group_size=overlap_group_size,
            block_budget=block_budget,
            seed=seed,
            pool_target=target,
            tolerance=overlap_tolerance,
        )
        for group in planned:
            block = build_corpus_block(group, token_budget=block_budget)
            # The planner budget-checked every addition with this same builder;
            # a partial pack here is a structural impossibility, not bad input.
            assert len(block.example_ids) == len(group), (
                "engineered-overlap group failed to pack whole despite the "
                "planner's per-candidate budget check (planner/packer drift?)"
            )
            block_id = len(blocks)
            blocks.append({
                "block_id": block_id,
                "text": block.text,
                "token_count": block.token_count,
                "n_paragraphs": len(block.paragraphs),
            })
            block_paragraphs.append(list(block.paragraphs))
            for ex_id in block.example_ids:
                question_to_block[ex_id] = block_id
    else:
        # Natural mode: same-paragraph questions adjacent; paragraph groups in
        # seeded-random order.
        groups: Dict[tuple, List[Any]] = {}
        order: List[tuple] = []
        for ex in examples:
            key = _ctx_key(ex)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(ex)
        rng = random.Random(seed)
        rng.shuffle(order)
        ordered: List[Any] = [ex for key in order for ex in groups[key]]

        remaining = ordered
        while remaining and len(question_to_block) < target:
            block = build_corpus_block(remaining, token_budget=block_budget)
            if not block.example_ids:
                # The first example's paragraph alone exceeds the budget: exclude
                # that whole paragraph group (counted) and continue with the rest.
                bad = _ctx_key(remaining[0])
                excluded.extend(ex.id for ex in remaining if _ctx_key(ex) == bad)
                remaining = [ex for ex in remaining if _ctx_key(ex) != bad]
                continue
            block_id = len(blocks)
            blocks.append({
                "block_id": block_id,
                "text": block.text,
                "token_count": block.token_count,
                "n_paragraphs": len(block.paragraphs),
            })
            block_paragraphs.append(list(block.paragraphs))
            for ex_id in block.example_ids:
                question_to_block[ex_id] = block_id
            packed = set(block.example_ids)
            remaining = [ex for ex in remaining if ex.id not in packed]

    # MEASURED overlap of the packed store, both modes (charter D5: overlap
    # "ENGINEERED and reported (never assumed)" -- natural overlap is a report,
    # engineered overlap is a report + a fail-closed contract).
    gold_by_id = {ex.id: gold_set(ex) for ex in examples}
    members_by_block: List[List[str]] = [[] for _ in blocks]
    for ex_id, b in question_to_block.items():  # insertion order = block pack order
        members_by_block[b].append(ex_id)
    per_group_stats = [
        group_overlap_stats(
            b, block_paragraphs[b], [gold_by_id[i] for i in members_by_block[b]]
        )
        for b in range(len(blocks))
    ]
    mode = "engineered" if overlap_target is not None else "natural"
    overlap = overlap_report(
        per_group_stats,
        mode=mode,
        target_shared_fraction=overlap_target,
        tolerance=overlap_tolerance if overlap_target is not None else None,
        group_size=overlap_group_size if overlap_target is not None else None,
    )
    if overlap_target is not None:
        enforce_overlap_target(overlap, dataset=dataset)

    pool_ids = list(question_to_block)  # insertion order = deterministic
    if len(pool_ids) < num_queries:
        raise ManifestError(
            f"in-corpus pool ({len(pool_ids)}) < num_queries ({num_queries}); "
            f"raise block_budget, load more examples, or lower N"
        )

    # Per-trial draws: without replacement ACROSS trials while the pool allows, so
    # trials stay independent; falls back to full-pool sampling (flagged) otherwise.
    trials: Dict[str, List[str]] = {}
    disjoint = len(pool_ids) >= num_queries * num_trials
    available = list(pool_ids)
    for t in range(1, num_trials + 1):
        rng_t = random.Random(seed + t - 1)
        source = available if disjoint else pool_ids
        picked = rng_t.sample(source, num_queries)
        trials[str(t)] = picked
        if disjoint:
            chosen = set(picked)
            available = [i for i in available if i not in chosen]

    trunc_rungs = derive_trunc_rungs(
        blocks, block_paragraphs, question_to_block, gold_by_id, trunc_budgets
    )

    n_loaded = len(examples)
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "num_queries": num_queries,
        "num_trials": num_trials,
        "block_budget": block_budget,
        "blocks": blocks,
        "question_to_block": question_to_block,
        "trials": trials,
        "overlap": overlap,
        "trunc_rungs": trunc_rungs,
        "stats": {
            "source_examples_loaded": n_loaded,
            "pool_size": len(pool_ids),
            "n_blocks": len(blocks),
            "examples_excluded": len(excluded),
            "exclusion_rate": (len(excluded) / n_loaded) if n_loaded else 0.0,
            "trials_disjoint": disjoint,
        },
    }


def validate_trunc_budgets(trunc_budgets: Sequence[int], block_budget: int) -> None:
    """Fail-closed ladder check: ints >= 1, strictly descending, every rung < block_budget.

    A rung equal to ``block_budget`` is refused explicitly: that point of the
    ladder is B3's own cell (charter §7.7(d)), never a duplicate B12 rung.
    """
    seen: List[int] = []
    for b in trunc_budgets:
        if isinstance(b, bool) or not isinstance(b, int):
            raise ManifestError(f"trunc_budgets entry {b!r} is not an int")
        if b < 1:
            raise ManifestError(f"trunc_budgets entry {b} must be >= 1")
        if b >= block_budget:
            raise ManifestError(
                f"trunc rung {b} must be < block_budget {block_budget}: the full "
                "budget is B3's own cell, and a rung above it is not a truncation"
            )
        if seen and b >= seen[-1]:
            raise ManifestError(
                f"trunc_budgets must be strictly descending (got {tuple(trunc_budgets)})"
            )
        seen.append(b)


def derive_trunc_rungs(
    blocks: Sequence[Dict[str, Any]],
    block_paragraphs: Sequence[Sequence[str]],
    question_to_block: Dict[str, int],
    gold_by_id: Dict[str, Any],
    trunc_budgets: Sequence[int],
    *,
    header: str = DEFAULT_HEADER,
    count_tokens: Optional[Callable[[str], int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """The B12 ladder from packed blocks (ADR-0106): no repacking, labels by construction.

    For each rung budget, each block keeps the longest PREFIX of its paragraphs
    (packing order) whose assembled text stays within the budget; the rung text
    is therefore a literal prefix of the full block text (same Document
    numbering). A query is in-corpus at a rung iff EVERY paragraph of its gold
    set survived in its own block's rung text. Keys are the rung budgets as
    strings (JSON-stable), in the registered descending order.
    """
    counter = count_tokens if count_tokens is not None else default_token_counter
    rungs: Dict[str, Dict[str, Any]] = {}
    for budget in trunc_budgets:
        rung_blocks: List[Dict[str, Any]] = []
        kept_sets: List[Set[str]] = []
        for block, paragraphs in zip(blocks, block_paragraphs):
            kept: List[str] = []
            for paragraph in paragraphs:
                candidate = kept + [paragraph]
                if counter(_assemble(header, candidate)) > budget:
                    break  # prefix semantics: stop at the first overflow
                kept = candidate
            text = _assemble(header, kept)
            rung_blocks.append({
                "block_id": block["block_id"],
                "text": text,
                "token_count": counter(text),
                "n_paragraphs": len(kept),
                "n_paragraphs_full": len(paragraphs),
            })
            kept_sets.append(set(kept))
        in_corpus_ids = [
            ex_id for ex_id, b in question_to_block.items()  # pack order
            if gold_by_id[ex_id] and gold_by_id[ex_id] <= kept_sets[b]
        ]
        rungs[str(budget)] = {
            "budget": budget,
            "blocks": rung_blocks,
            "in_corpus_ids": in_corpus_ids,
            "n_in_corpus": len(in_corpus_ids),
            "n_out_of_corpus": len(question_to_block) - len(in_corpus_ids),
        }
    return rungs


def trunc_rung_for(manifest: Dict[str, Any], budget: int) -> Dict[str, Any]:
    """The B12 rung record for ``budget`` (fail closed on a missing rung/ladder)."""
    rungs = manifest.get("trunc_rungs")
    if not rungs:
        raise ManifestError(
            f"manifest carries no truncation rungs (rung {budget} requested); "
            "rebuild it with build_query_manifest.py --trunc-budgets"
        )
    rung = rungs.get(str(budget))
    if rung is None:
        raise ManifestError(
            f"manifest has no truncation rung {budget} (registered rungs: "
            f"{sorted((int(k) for k in rungs), reverse=True)})"
        )
    return rung


def select_examples(manifest: Dict[str, Any], trial: int, examples: Sequence[Any]) -> List[Any]:
    """The trial's measured set, in manifest order. Raises on any missing id."""
    ids = manifest.get("trials", {}).get(str(trial))
    if not ids:
        raise ManifestError(f"manifest has no trial {trial}")
    id_map = {ex.id: ex for ex in examples}
    missing = [i for i in ids if i not in id_map]
    if missing:
        raise ManifestError(
            f"{len(missing)} manifest ids not found in the loaded dataset "
            f"(first: {missing[:3]}); dataset/split/seed mismatch?"
        )
    return [id_map[i] for i in ids]


def block_for(manifest: Dict[str, Any], example_id: str) -> Dict[str, Any]:
    """The corpus block record assigned to a question id."""
    q2b = manifest.get("question_to_block", {})
    if example_id not in q2b:
        raise ManifestError(f"{example_id} has no corpus block in this manifest")
    return manifest["blocks"][q2b[example_id]]
