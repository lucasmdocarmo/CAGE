"""Staleness/freshness metrics for the CAGE staleness baseline (SCAFFOLD).

These quantify the GROUNDING cost of serving stale-but-cheap cache hits. They are ported
from GroundedCache ("Grounded Cache Routing for RAG: When Is It Safe to Reuse an Answer?",
S. H. Shah, 2026, https://arxiv.org/abs/2605.27494) and adapted to CAGE's controlled
`stale_fraction` sweep. They sit NEXT TO the LettuceDetect grounding scorer, not instead of
it: the staleness arm reports CAGE's standard quality metrics PLUS these three.

Definitions (per GroundedCache):
    USR  Unsafe-Served Rate    = fraction of ALL queries served a wrong (ungrounded) cached
                                 answer.  USR = unsafe_served / total.
    aHR  Answer/cache Hit Rate = fraction of queries answered from cache.
    FH   False-Hit Rate        = error rate CONDITIONAL on a cache hit.
                                 FH = USR / aHR = Pr[wrong | served-from-cache].
    SHR  Stale-Hit Rate        = fraction of served STALE (v0) hits that produced an
                                 ungrounded answer -- CAGE's controlled analogue of
                                 GroundedCache's Stale-Hit (SH). The `stale_fraction` knob
                                 drives this directly.

A per-query record is a mapping with (at least):
    served_from_cache : bool   was this query answered from a cache hit?
    grounded          : bool   did the produced answer pass CAGE grounding? (e.g.
                               grounding_score >= threshold, or hallucinated_span_ratio == 0)
    evidence_version  : str    "v0" (stale) | "v1" (fresh) | None (no cache hit)

This module is dependency-free (stdlib only) so it is unit-testable without model loads.
Wiring into the run loop is pending; see cloud_docs/STALENESS_BASELINE_DESIGN.md.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional


def _served(rec: Mapping) -> bool:
    return bool(rec.get("served_from_cache"))


def _grounded(rec: Mapping) -> bool:
    return bool(rec.get("grounded"))


def _is_stale(rec: Mapping, *, stale_value: str = "v0", field: str = "evidence_version") -> bool:
    return rec.get(field) == stale_value


def unsafe_served_rate(records: Iterable[Mapping]) -> Optional[float]:
    """USR = fraction of ALL queries served a wrong (ungrounded) cached answer."""
    recs = list(records)
    if not recs:
        return None
    unsafe = sum(1 for r in recs if _served(r) and not _grounded(r))
    return unsafe / len(recs)


def answer_hit_rate(records: Iterable[Mapping]) -> Optional[float]:
    """aHR = fraction of queries answered from cache."""
    recs = list(records)
    if not recs:
        return None
    return sum(1 for r in recs if _served(r)) / len(recs)


def false_hit_rate(records: Iterable[Mapping]) -> Optional[float]:
    """FH = error rate conditional on a cache hit = USR / aHR = Pr[wrong | served]."""
    served = [r for r in records if _served(r)]
    if not served:
        return None
    wrong = sum(1 for r in served if not _grounded(r))
    return wrong / len(served)


def stale_hit_rate(records: Iterable[Mapping], *, stale_value: str = "v0",
                   field: str = "evidence_version") -> Optional[float]:
    """SHR = fraction of served STALE (v0) hits that produced an ungrounded answer."""
    stale_served = [
        r for r in records
        if _served(r) and _is_stale(r, stale_value=stale_value, field=field)
    ]
    if not stale_served:
        return None
    ungrounded = sum(1 for r in stale_served if not _grounded(r))
    return ungrounded / len(stale_served)


def staleness_metrics(records: Iterable[Mapping], *, stale_value: str = "v0",
                      field: str = "evidence_version") -> dict:
    """Compute {usr, ahr, fh, shr} together (each None when undefined)."""
    recs = list(records)
    return {
        "unsafe_served_rate": unsafe_served_rate(recs),
        "answer_hit_rate": answer_hit_rate(recs),
        "false_hit_rate": false_hit_rate(recs),
        "stale_hit_rate": stale_hit_rate(recs, stale_value=stale_value, field=field),
    }


# --- Serving-path helpers (deterministic; used by the staleness baseline) ---

def select_stale(example_id: str, stale_fraction: float, seed: int = 42) -> bool:
    """Deterministically decide whether THIS query is served a STALE (v0) entry.

    A uniform hash of (seed, example_id) < stale_fraction, so exactly ~stale_fraction of the
    query set is stale, reproducibly across trials and machines (no wall-clock, no RNG state).
    """
    if stale_fraction <= 0.0:
        return False
    if stale_fraction >= 1.0:
        return True
    import hashlib
    h = hashlib.sha1(f"{seed}:{example_id}".encode("utf-8")).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < stale_fraction


def make_stale_context(contexts, answer):
    """Return a STALE (v0) variant of the gold contexts.

    The answer span is redacted so the served evidence no longer supports the answer. If the
    answer is not literally present, drop the first (typically most relevant) passage so the
    evidence is still deterministically degraded. This is the controlled analogue of an
    outdated cached entry (GroundedCache G3 "Version Match" failure).
    """
    import re
    ans = (answer or "").strip()
    out = []
    for c in (contexts or []):
        if ans and ans.lower() in c.lower():
            c = re.sub(re.escape(ans), "[redacted]", c, flags=re.IGNORECASE)
        out.append(c)
    if ans and len(out) > 1 and not any("[redacted]" in c for c in out):
        out = out[1:]
    return out
