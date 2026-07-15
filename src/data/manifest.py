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

Pure stdlib + src.data.corpus: importable (and unit-testable) without the
``datasets`` package, torch, or a GPU.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

from src.data.corpus import build_corpus_block

MANIFEST_VERSION = 1


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
) -> Dict[str, Any]:
    """Build the manifest dict from loader examples (.id/.question/.context/.answer)."""
    if num_queries < 1 or num_trials < 1:
        raise ManifestError("num_queries and num_trials must be >= 1")
    target = pool_target or max(3 * num_queries, num_queries * num_trials)

    # Same-paragraph questions adjacent; paragraph groups in seeded-random order.
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

    blocks: List[Dict[str, Any]] = []
    question_to_block: Dict[str, int] = {}
    excluded: List[str] = []
    remaining = ordered
    while remaining and len(question_to_block) < target:
        block = build_corpus_block(remaining, token_budget=block_budget)
        if not block.example_ids:
            # The first example's paragraph alone exceeds the budget: exclude that
            # whole paragraph group (counted) and continue with the rest.
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
        for ex_id in block.example_ids:
            question_to_block[ex_id] = block_id
        packed = set(block.example_ids)
        remaining = [ex for ex in remaining if ex.id not in packed]

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
        "stats": {
            "source_examples_loaded": n_loaded,
            "pool_size": len(pool_ids),
            "n_blocks": len(blocks),
            "examples_excluded": len(excluded),
            "exclusion_rate": (len(excluded) / n_loaded) if n_loaded else 0.0,
            "trials_disjoint": disjoint,
        },
    }


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
