"""Engineered-overlap store control + MEASURED overlap statistics (D5 F1/F3).

Charter (MyDocs/PUBLICATION.md D5 item 2 + F3; carrier of contrast #17): the
HotpotQA store's cross-question paragraph overlap must be "ENGINEERED and
reported (never assumed)". This module supplies both halves for the
uniform-yardstick manifest (src/data/manifest.py):

- ``plan_overlap_groups``: a deterministic (seeded) greedy planner that
  composes query groups whose MEAN pairwise shared-paragraph fraction targets
  an explicit value; ``build_manifest`` packs each planned group into its own
  corpus block, so the store's overlap is a controlled input, not an accident
  of dataset ordering.
- ``group_overlap_stats`` / ``overlap_report``: the MEASURED realized overlap
  of a manifest's blocks. ``build_manifest`` writes this report into the
  artifact in BOTH modes -- natural overlap is also reported, never assumed.
- ``enforce_overlap_target``: the fail-closed gate -- an engineered target the
  loaded corpus cannot realize raises ``OverlapError`` naming realized vs
  target and the concrete fixes. No silent best-effort.

Definitions (one group == one corpus block == one shared store):
- pairwise shared-paragraph fraction of two questions = Jaccard of their gold
  paragraph sets, |A∩B| / |A∪B| (exact-string identity -- the same identity
  the corpus builder dedupes on).
- pairwise shared-tokens estimate = token estimate of the intersection
  paragraphs (the KV-reusable material between the pair), under the same
  4/3-words heuristic the manifest's ``token_count`` fields use
  (``src.data.corpus.default_token_counter``) -- an ESTIMATE, labeled ``_est``.
- shared-prefix-tokens estimate of a group = token estimate of the leading run
  of block Documents present in EVERY member's gold set (scaffolding included,
  header excluded -- the header is constant across all groups), i.e. the
  prefix-cache-relevant sharing when the block is served front-to-back.

Absence stays absence: a group with <2 questions contributes no pairs; when a
manifest has no pairs at all the means are ``None`` with a named
``no_pairs_reason``, never fabricated zeros.

Pure stdlib + src.data.corpus, like the manifest module: importable and
unit-testable without ``datasets``, torch, or a GPU.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from src.data.corpus import build_corpus_block, default_token_counter

OVERLAP_STATS_VERSION = 1

#: Planner candidate-scan bounds (deterministic; bound the O(F) scan per step).
_SCAN_OVERLAPPING = 20   # families reachable via a shared paragraph
_SCAN_FRESH = 8          # next not-yet-used families in seeded order


class OverlapError(ValueError):
    """Raised when an engineered-overlap request cannot be honored as specified."""


def gold_set(example: Any) -> FrozenSet[str]:
    """The example's gold paragraph set (non-empty strings, exact identity)."""
    return frozenset(p for p in (getattr(example, "context", None) or []) if p)


def pairwise_shared_fraction(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Jaccard |A∩B| / |A∪B| of two gold paragraph sets (0.0 for two empties)."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def group_overlap_stats(
    block_id: int,
    block_paragraphs: Sequence[str],
    member_sets: Sequence[FrozenSet[str]],
    count_tokens: Optional[Callable[[str], int]] = None,
) -> Dict[str, Any]:
    """Realized overlap of ONE group (= one corpus block) of member questions.

    ``block_paragraphs`` is the block's paragraph list in served order
    (``CorpusBlock.paragraphs``); ``member_sets`` the gold paragraph set of each
    question assigned to the block (always subsets of the block, by the corpus
    builder's all-or-nothing rule). Means are ``None`` when the group has <2
    members (no pairs -- absence stays absence).
    """
    counter = count_tokens if count_tokens is not None else default_token_counter
    n = len(member_sets)
    fractions: List[float] = []
    shared_tokens: List[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = member_sets[i] & member_sets[j]
            fractions.append(pairwise_shared_fraction(member_sets[i], member_sets[j]))
            if inter:
                # Intersection paragraphs in served (block) order, scaffolded the
                # way the block scaffolds them, so the estimate tracks served text.
                shared = [p for p in block_paragraphs if p in inter]
                text = "\n\n".join(
                    f"Document {k}:\n{p}" for k, p in enumerate(shared, start=1)
                )
                shared_tokens.append(counter(text))
            else:
                shared_tokens.append(0)

    # Leading run of block Documents present in EVERY member's set: the prefix
    # all of the group's queries share when the block is served front-to-back.
    prefix: List[str] = []
    if n >= 2:
        for p in block_paragraphs:
            if all(p in s for s in member_sets):
                prefix.append(p)
            else:
                break
    prefix_tokens = (
        counter("\n\n".join(f"Document {k}:\n{p}" for k, p in enumerate(prefix, start=1)))
        if prefix else 0
    )

    n_pairs = len(fractions)
    return {
        "block_id": block_id,
        "n_questions": n,
        "n_pairs": n_pairs,
        "mean_pairwise_shared_fraction":
            (sum(fractions) / n_pairs) if n_pairs else None,
        "mean_pairwise_shared_tokens_est":
            (sum(shared_tokens) / n_pairs) if n_pairs else None,
        "shared_prefix_tokens_est": prefix_tokens if n >= 2 else None,
    }


def overlap_report(
    per_group: Sequence[Dict[str, Any]],
    mode: str,
    target_shared_fraction: Optional[float] = None,
    tolerance: Optional[float] = None,
    group_size: Optional[int] = None,
) -> Dict[str, Any]:
    """The manifest's ``overlap`` field: knob provenance + realized statistics.

    Overall pairwise means are PAIR-weighted (mean over all pairs across all
    groups); the prefix estimate is averaged over multi-question groups. Written
    in natural mode too (``target_shared_fraction=None``) so overlap is always
    measured, never assumed.
    """
    n_pairs = sum(g["n_pairs"] for g in per_group)
    multi = [g for g in per_group if g["n_questions"] >= 2]
    measured: Dict[str, Any] = {
        "n_groups": len(per_group),
        "n_multi_question_groups": len(multi),
        "n_pairs": n_pairs,
        "mean_pairwise_shared_fraction": None,
        "mean_pairwise_shared_tokens_est": None,
        "mean_shared_prefix_tokens_est": None,
        "per_group": list(per_group),
    }
    if n_pairs:
        measured["mean_pairwise_shared_fraction"] = (
            sum(g["mean_pairwise_shared_fraction"] * g["n_pairs"] for g in multi)
            / n_pairs
        )
        measured["mean_pairwise_shared_tokens_est"] = (
            sum(g["mean_pairwise_shared_tokens_est"] * g["n_pairs"] for g in multi)
            / n_pairs
        )
        measured["mean_shared_prefix_tokens_est"] = (
            sum(g["shared_prefix_tokens_est"] for g in multi) / len(multi)
        )
    else:
        measured["no_pairs_reason"] = (
            "every group holds a single question (no within-group pairs), so "
            "pairwise overlap is undefined -- reported as None, not fabricated"
        )
    return {
        "overlap_stats_version": OVERLAP_STATS_VERSION,
        "mode": mode,
        "target_shared_fraction": target_shared_fraction,
        "tolerance": tolerance,
        "group_size": group_size,
        "measured": measured,
    }


def enforce_overlap_target(report: Dict[str, Any], dataset: str = "") -> None:
    """Fail-closed gate for engineered mode: realized must be within tolerance.

    Raises ``OverlapError`` (NAMED: realized vs target vs tolerance + the
    concrete fixes) when the loaded corpus could not support the request.
    Natural mode never calls this -- measurement without a target is reporting,
    not a contract.
    """
    target = report["target_shared_fraction"]
    tolerance = report["tolerance"]
    measured = report["measured"]
    label = f" for dataset '{dataset}'" if dataset else ""
    realized = measured["mean_pairwise_shared_fraction"]
    if realized is None:
        raise OverlapError(
            f"engineered overlap UNVERIFIABLE{label}: target "
            f"{target:.3f} requested but the planned store has no within-group "
            f"question pairs ({measured['n_groups']} single-question groups) -- "
            "load more examples (more questions per paragraph set), raise "
            "block_budget, or drop the overlap target for natural mode"
        )
    if abs(realized - target) > tolerance:
        raise OverlapError(
            f"engineered overlap UNREACHABLE{label}: realized mean pairwise "
            f"shared-paragraph fraction {realized:.3f} vs target {target:.3f} "
            f"(tolerance ±{tolerance:.3f}) over {measured['n_pairs']} pairs in "
            f"{measured['n_multi_question_groups']} multi-question groups; the "
            "loaded corpus cannot support this target -- load more examples "
            "(more questions per paragraph set), adjust the overlap "
            "target/tolerance, or raise block_budget"
        )


def _validate_knobs(
    target: float, tolerance: float, group_size: int
) -> None:
    """NAMED refusals for malformed engineering knobs (fail-closed, upfront)."""
    if not (0.0 <= target <= 1.0):
        raise OverlapError(
            f"overlap target {target!r} is not a fraction in [0, 1]"
        )
    if tolerance <= 0:
        raise OverlapError(
            f"overlap tolerance {tolerance!r} must be > 0 (an exact-match "
            "contract can never certify on real corpora)"
        )
    if group_size < 2:
        raise OverlapError(
            f"overlap group_size {group_size!r} must be >= 2 (a single-question "
            "group has no pairs to engineer)"
        )


def plan_overlap_groups(
    examples: Sequence[Any],
    target: float,
    group_size: int,
    block_budget: int,
    seed: int,
    pool_target: int,
    tolerance: float = 0.05,
) -> Tuple[List[List[Any]], List[str]]:
    """Compose query groups targeting a mean pairwise shared-paragraph fraction.

    Deterministic under ``seed``. Greedy: families (questions sharing one exact
    gold paragraph set) are visited in seeded-random order; each group starts
    from the next family's question and repeatedly adds the candidate question
    -- same-family, paragraph-overlapping-family, or fresh-family -- whose
    addition moves the group's running mean closest to ``target``, stopping at
    ``group_size``, when the block budget refuses every candidate, or when the
    best addition would move the mean AWAY from target (target beats size).

    Every candidate is budget-checked with the SAME builder/counter that will
    pack the block (``build_corpus_block``), so planned groups always pack
    whole. Returns ``(groups, excluded_ids)`` where ``excluded_ids`` are
    questions whose gold paragraphs alone exceed ``block_budget`` (the
    auditable exclusion, as in natural mode). Examples with no gold paragraphs
    are dropped exactly as the corpus builder drops them.

    This function plans; it does not certify. The caller measures the PACKED
    blocks and gates with ``enforce_overlap_target`` -- planning quality never
    substitutes for measured overlap.
    """
    _validate_knobs(target, tolerance, group_size)

    # Families: exact gold-set key -> questions, insertion-ordered (determinism).
    fam_keys: List[FrozenSet[str]] = []
    fam_members: Dict[FrozenSet[str], List[Any]] = {}
    for ex in examples:
        key = gold_set(ex)
        if not key:
            continue  # no gold paragraph: never in any block (corpus.py rule)
        if key not in fam_members:
            fam_members[key] = []
            fam_keys.append(key)
        fam_members[key].append(ex)

    order = list(range(len(fam_keys)))
    random.Random(seed).shuffle(order)  # seeded family visit order
    rank = {fam_idx: pos for pos, fam_idx in enumerate(order)}

    # Paragraph -> family indices holding it (for overlapping-family candidates),
    # each list in seeded-rank order for deterministic bounded scans.
    para_index: Dict[str, List[int]] = {}
    for fam_idx, key in enumerate(fam_keys):
        for p in key:
            para_index.setdefault(p, []).append(fam_idx)
    for fams in para_index.values():
        fams.sort(key=lambda i: rank[i])

    queues: Dict[int, List[Any]] = {
        i: list(fam_members[fam_keys[i]]) for i in range(len(fam_keys))
    }

    def fits(group: List[Any], candidate: Any) -> bool:
        trial = group + [candidate]
        block = build_corpus_block(trial, token_budget=block_budget)
        return len(block.example_ids) == len(trial)

    groups: List[List[Any]] = []
    excluded: List[str] = []
    cursor = 0  # over `order`
    total = 0
    while total < pool_target:
        # Seed the next group from the next non-empty family in seeded order.
        while cursor < len(order) and not queues[order[cursor]]:
            cursor += 1
        if cursor >= len(order):
            break
        seed_fam = order[cursor]
        seed_ex = queues[seed_fam][0]
        if not fits([], seed_ex):
            # Gold paragraphs alone exceed the budget: exclude the whole family,
            # counted -- the natural path's auditable-exclusion rule.
            excluded.extend(ex.id for ex in queues[seed_fam])
            queues[seed_fam] = []
            continue
        queues[seed_fam].pop(0)
        group = [seed_ex]
        group_sets = [fam_keys[seed_fam]]
        used_order: List[int] = [seed_fam]  # first-use order (determinism)
        used = {seed_fam}
        pair_sum = 0.0

        while len(group) < group_size:
            # Bounded, deterministically-ordered candidate families:
            # (a) families already in the group (raise the mean toward 1.0),
            # (b) families sharing >=1 paragraph with the group (interior J),
            # (c) the next fresh families in seeded order (typically J=0).
            cand_fams: List[int] = [f for f in used_order if queues[f]]
            seen = set(cand_fams) | used
            overlapping = 0
            group_paras = sorted({p for s in group_sets for p in s})
            for p in group_paras:
                if overlapping >= _SCAN_OVERLAPPING:
                    break
                for f in para_index.get(p, []):
                    if overlapping >= _SCAN_OVERLAPPING:
                        break
                    if f not in seen and queues[f]:
                        cand_fams.append(f)
                        seen.add(f)
                        overlapping += 1
            fresh = 0
            for pos in range(cursor, len(order)):
                f = order[pos]
                if f in seen or not queues[f]:
                    continue
                cand_fams.append(f)
                seen.add(f)
                fresh += 1
                if fresh >= _SCAN_FRESH:
                    break

            n_now = len(group)
            pairs_now = n_now * (n_now - 1) // 2
            err_now = (
                abs(pair_sum / pairs_now - target) if pairs_now else float("inf")
            )
            best: Optional[Tuple[float, int, int, float]] = None
            for idx, f in enumerate(cand_fams):
                cand_key = fam_keys[f]
                delta = sum(pairwise_shared_fraction(cand_key, s) for s in group_sets)
                new_mean = (pair_sum + delta) / (pairs_now + n_now)
                err = abs(new_mean - target)
                if best is None or (err, idx) < (best[0], best[1]):
                    if fits(group, queues[f][0]):
                        best = (err, idx, f, delta)
            if best is None:
                break  # budget refuses every candidate: close the group
            if pairs_now and best[0] > err_now + 1e-12:
                break  # would drag the mean away from target: target beats size
            _, _, fam, delta = best
            group.append(queues[fam].pop(0))
            group_sets.append(fam_keys[fam])
            pair_sum += delta
            if fam not in used:
                used.add(fam)
                used_order.append(fam)

        groups.append(group)
        total += len(group)

    return groups, excluded
