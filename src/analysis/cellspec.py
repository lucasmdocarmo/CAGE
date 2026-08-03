"""D7 cell identity — the CellSpec tuple (PUBLICATION.md §7.1-§7.6.1; audit gap P0-1).

Cell identity is a TUPLE, not a name: ``arm · retriever · policy · topology ·
engine · model · family`` plus the optional §6.1 pressure coordinates
(``budget_r`` = capacity ratio r = B/D, ``rate_frac`` = fraction of predicted
saturation rate λ*). The pilot's flat ~20 baseline names conflated these axes;
``from_legacy`` translates them (best-effort, documented per name) so pilot
archives can be re-keyed. Downstream stats/figure code — including every
serving-yield (Y) scored cell — keys rows on ``CellSpec.to_row_key()``.

Charter bindings enforced here:
- §7.1 arms (11 names; ``gold-trunc`` retired) and the B1-B12 numbered layer,
  including the reuse-bit classification (FRESH = B1/B5/B6/B9/B11,
  REUSE = B2/B3/B4/B7/B8/B10/B12; B12 reuse=ON resolved 2026-08-02).
- §7.3 policy axis: ``truncate`` is NOT a policy value — B11/B12 arms carry
  policy=none (no axis double-counting, declared 2026-08-02).
- §7.6.1 family × arm legality; hf = sub-pressure F1 oracle only; LMDeploy
  restricted to 14B/70B (P7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, get_args

Arm = Literal[
    "gold-fresh",
    "gold-reuse",
    "corpus-fresh",
    "corpus-reuse",
    "retr-fresh",
    "retr-reuse",
    "retr-store",
    "retr-comp",
    "corpus-comp",
    "retr-trunc",
    "corpus-trunc",
]
Retriever = Literal["dense", "rerank", "bm25", "rrf", "none"]
Policy = Literal["none", "recompute", "evict", "offload", "compress-fp8"]
Topology = Literal["single", "tp", "pd"]
Engine = Literal["vllm", "sglang", "lmdeploy", "hf"]
Model = Literal["qwen3-14b", "llama-3.3-70b", "qwen3-next-80b", "deepseek-v3"]
Family = Literal["F1", "F2", "F3", "DIST"]
Reuse = Literal["fresh", "reuse"]
BaselineID = Literal[
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12"
]


class CellSpecError(ValueError):
    """Base error for cell-identity violations."""


class InvalidCellSpecError(CellSpecError):
    """An axis value or axis combination is illegal under the charter."""


class UnknownBaselineError(CellSpecError):
    """A baseline / legacy name is not in the registered maps (fail closed)."""


_ALLOWED: dict[str, frozenset[str]] = {
    "arm": frozenset(get_args(Arm)),
    "retriever": frozenset(get_args(Retriever)),
    "policy": frozenset(get_args(Policy)),
    "topology": frozenset(get_args(Topology)),
    "engine": frozenset(get_args(Engine)),
    "model": frozenset(get_args(Model)),
    "family": frozenset(get_args(Family)),
}

_RETRIEVAL_ARMS: frozenset[str] = frozenset(
    {"retr-fresh", "retr-reuse", "retr-store", "retr-comp", "retr-trunc"}
)
# §7.6.1: F2 carries the FRESH-set arms (prefix OFF); F3 carries the REUSE-set
# arms (prefix ON × pressure — includes corpus-fresh, B4's reuse-bit is REUSE);
# DIST carries the transfer pair only.
_ARMS_BY_FAMILY: dict[str, frozenset[str]] = {
    "F1": _ALLOWED["arm"],
    "F2": frozenset({"gold-fresh", "retr-fresh", "retr-comp", "retr-trunc"}),
    "F3": frozenset(
        {
            "gold-reuse",
            "corpus-reuse",
            "corpus-fresh",
            "retr-reuse",
            "retr-store",
            "corpus-comp",
            "corpus-trunc",
        }
    ),
    "DIST": frozenset({"corpus-reuse", "gold-fresh"}),
}
_LMDEPLOY_MODELS: frozenset[str] = frozenset({"qwen3-14b", "llama-3.3-70b"})
_PRESSURE_FAMILIES: frozenset[str] = frozenset({"F2", "F3"})


@dataclass(frozen=True)
class CellSpec:
    """One cell of the campaign grid — the §7 tuple with §6.1 pressure coords."""

    arm: Arm
    retriever: Retriever
    policy: Policy
    topology: Topology
    engine: Engine
    model: Model
    family: Family
    budget_r: float | None = None
    rate_frac: float | None = None

    def __post_init__(self) -> None:
        for name, allowed in _ALLOWED.items():
            value = getattr(self, name)
            if value not in allowed:
                raise InvalidCellSpecError(
                    f"{name}={value!r} is not a charter value; allowed: {sorted(allowed)}"
                )
        if self.arm in _RETRIEVAL_ARMS and self.retriever == "none":
            raise InvalidCellSpecError(
                f"arm={self.arm!r} is a retrieval arm and requires a retriever (§7.2)"
            )
        if self.arm not in _RETRIEVAL_ARMS and self.retriever != "none":
            raise InvalidCellSpecError(
                f"arm={self.arm!r} has no retrieval stage; retriever must be 'none', "
                f"got {self.retriever!r}"
            )
        if self.arm not in _ARMS_BY_FAMILY[self.family]:
            raise InvalidCellSpecError(
                f"arm={self.arm!r} is not carried by family {self.family} "
                f"(§7.6.1: {sorted(_ARMS_BY_FAMILY[self.family])})"
            )
        if self.policy != "none" and self.family not in _PRESSURE_FAMILIES:
            raise InvalidCellSpecError(
                f"policy={self.policy!r} requires a pressure family (F2/F3); "
                f"family={self.family} (§7.7b: policies FIRE under pressure)"
            )
        if self.engine == "hf":
            if self.family != "F1":
                raise InvalidCellSpecError(
                    f"engine=hf is the sub-pressure F1 oracle only (§7.3/§7.4); "
                    f"family={self.family}"
                )
            if self.topology != "single":
                raise InvalidCellSpecError(
                    f"engine=hf runs batch-1 single-device only; topology={self.topology!r}"
                )
        if self.engine == "lmdeploy" and self.model not in _LMDEPLOY_MODELS:
            raise InvalidCellSpecError(
                f"engine=lmdeploy supports {sorted(_LMDEPLOY_MODELS)} only (§7.3 P7); "
                f"model={self.model!r}"
            )
        if self.topology == "pd" and self.family != "DIST":
            raise InvalidCellSpecError(
                f"topology=pd is the disaggregation overlay (family DIST); "
                f"family={self.family}"
            )
        if self.family == "DIST" and self.topology == "single":
            raise InvalidCellSpecError(
                "family=DIST is the tp/pd topology overlay; topology='single' is illegal"
            )
        for name in ("budget_r", "rate_frac"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidCellSpecError(f"{name}={value!r} must be a float or None")
            if not math.isfinite(value) or value <= 0.0:
                raise InvalidCellSpecError(f"{name}={value!r} must be finite and > 0")
            if self.family == "F1":
                raise InvalidCellSpecError(
                    f"{name} set but family=F1 is sub-pressure by definition (§7.6.1)"
                )
            object.__setattr__(self, name, float(value))

    def to_flat_dict(self) -> dict[str, str | float | None]:
        """Flat mapping suitable as CSV columns (optional coords stay None)."""
        return {
            "arm": self.arm,
            "retriever": self.retriever,
            "policy": self.policy,
            "topology": self.topology,
            "engine": self.engine,
            "model": self.model,
            "family": self.family,
            "budget_r": self.budget_r,
            "rate_frac": self.rate_frac,
        }

    @classmethod
    def from_flat_dict(cls, row: Mapping[str, Any]) -> CellSpec:
        """Rebuild from ``to_flat_dict`` output / a CSV row (extra keys ignored).

        Optional coords accept None, "", or NaN (pandas missing) as absent.
        """
        try:
            axes = {name: row[name] for name in _ALLOWED}
        except KeyError as exc:
            raise CellSpecError(f"flat row is missing required column {exc.args[0]!r}") from exc
        coords: dict[str, float | None] = {}
        for name in ("budget_r", "rate_frac"):
            value = row.get(name)
            if value is None or value == "":
                coords[name] = None
            elif isinstance(value, float) and math.isnan(value):
                coords[name] = None
            else:
                coords[name] = float(value)
        return cls(**axes, **coords)  # type: ignore[arg-type]

    def to_row_key(self) -> str:
        """Canonical results-row key: the 7 axes plus any pressure coords."""
        parts = [
            self.arm,
            self.retriever,
            self.policy,
            self.topology,
            self.engine,
            self.model,
            self.family,
        ]
        if self.budget_r is not None:
            parts.append(f"r{self.budget_r:g}")
        if self.rate_frac is not None:
            parts.append(f"lam{self.rate_frac:g}")
        return "|".join(parts)

    @classmethod
    def from_baseline(
        cls,
        baseline_id: str,
        *,
        engine: Engine = "vllm",
        model: Model = "qwen3-14b",
        family: Family = "F1",
        topology: Topology = "single",
        policy: Policy = "none",
        budget_r: float | None = None,
        rate_frac: float | None = None,
    ) -> CellSpec:
        """Instantiate a B1-B12 presentation-layer baseline as a concrete cell."""
        try:
            base = BASELINES[baseline_id]  # type: ignore[index]
        except KeyError:
            raise UnknownBaselineError(
                f"unknown baseline id {baseline_id!r}; known: {list(BASELINES)}"
            ) from None
        return cls(
            arm=base.arm,
            retriever=base.retriever,
            policy=policy,
            topology=topology,
            engine=engine,
            model=model,
            family=family,
            budget_r=budget_r,
            rate_frac=rate_frac,
        )


@dataclass(frozen=True)
class BaselineDef:
    """§7.1 numbered-layer entry: arm, retriever, and the reuse-bit classification."""

    arm: Arm
    retriever: Retriever
    reuse: Reuse


# §7.1 numbered presentation layer (v2, 12 total). Ranking rule: the reranker is
# ablated exactly once (B5 vs B6); B7-B12's RAG-side baselines inherit the RANKED
# pipeline. B4's reuse-bit is REUSE (it rides F3 with its pair B3, §7.6 Group B).
BASELINES: dict[BaselineID, BaselineDef] = {
    "B1": BaselineDef("gold-fresh", "none", "fresh"),
    "B2": BaselineDef("gold-reuse", "none", "reuse"),
    "B3": BaselineDef("corpus-reuse", "none", "reuse"),
    "B4": BaselineDef("corpus-fresh", "none", "reuse"),
    "B5": BaselineDef("retr-fresh", "dense", "fresh"),
    "B6": BaselineDef("retr-fresh", "rerank", "fresh"),
    "B7": BaselineDef("retr-reuse", "rerank", "reuse"),
    "B8": BaselineDef("retr-store", "rerank", "reuse"),
    "B9": BaselineDef("retr-comp", "rerank", "fresh"),
    "B10": BaselineDef("corpus-comp", "none", "reuse"),
    "B11": BaselineDef("retr-trunc", "rerank", "fresh"),
    "B12": BaselineDef("corpus-trunc", "none", "reuse"),
}

FRESH_SET: frozenset[BaselineID] = frozenset({"B1", "B5", "B6", "B9", "B11"})
REUSE_SET: frozenset[BaselineID] = frozenset(
    {"B2", "B3", "B4", "B7", "B8", "B10", "B12"}
)


def _legacy(
    arm: Arm,
    retriever: Retriever = "none",
    *,
    policy: Policy = "none",
    family: Family = "F1",
    topology: Topology = "single",
) -> CellSpec:
    # Best-effort pilot defaults: anchor model (the pilot's Qwen3-8B is off the D4
    # roster), vLLM (the only pilot engine), single GPU, sub-pressure F1 unless the
    # retired name implies a policy (compressed_cag) or the topology axis (distributed).
    return CellSpec(
        arm=arm,
        retriever=retriever,
        policy=policy,
        topology=topology,
        engine="vllm",
        model="qwen3-14b",
        family=family,
    )


# Pilot-era name -> tuple. Sources: BaselineType enum (orchestration/baselines.py),
# the §7.5 retirement list, §7.1/§7.7a merge notes, and the 100x3 archive's
# `baseline` column (RAG-side pilot cells ran the RANKED pipeline —
# retrieval_reranked=1.0 in the archive — except the spec-matrix rag cells).
LEGACY_ALIASES: dict[str, CellSpec] = {
    # 2x2 spine
    "no_cache": _legacy("gold-fresh"),
    "prefix_cache": _legacy("gold-reuse"),
    "cag_full": _legacy("gold-reuse"),  # §7.7a: pilot cag_full ≡ prefix_cache
    "cag_true_on": _legacy("corpus-reuse"),
    "cag_true_off": _legacy("corpus-fresh"),
    # prefix-envelope scenarios -> workload manifest inside F1, same cell tuple (§7.5)
    "prefix_cache_grouped": _legacy("gold-reuse"),
    "prefix_cache_multiturn": _legacy("gold-reuse"),
    "prefix_cache_repeat": _legacy("gold-reuse"),
    # retrieval side
    "rag": _legacy("retr-fresh", "rerank"),
    "rag_full": _legacy("retr-fresh", "rerank"),
    "redis": _legacy("retr-fresh", "rerank"),  # §7.5 retired: artifact cache ≈ rag
    "redis_retrieval_cache_cold": _legacy("retr-fresh", "rerank"),
    "redis_retrieval_cache_warm": _legacy("retr-fresh", "rerank"),
    "hybrid": _legacy("retr-reuse", "rerank"),  # cold/warm are runtime labels (§7.5)
    "hybrid_retrieval_cache_cold": _legacy("retr-reuse", "rerank"),
    "hybrid_retrieval_cache_warm": _legacy("retr-reuse", "rerank"),
    "lmcache_rag": _legacy("retr-store", "rerank"),
    # compression axis
    "compressed_rag": _legacy("retr-comp", "rerank"),
    # §7.5: fp8 KV dtype -> the compress POLICY on the CAG cell (policy => F3)
    "compressed_cag": _legacy("corpus-reuse", policy="compress-fp8", family="F3"),
    # retired axes, mapped to the underlying serving arm (documented approximations)
    "distributed": _legacy("gold-fresh", family="DIST", topology="pd"),  # §7.5 -> topology axis
    "staleness": _legacy("gold-reuse"),  # warmth-held engine-reuse serving shape
    "speculative": _legacy("gold-reuse"),  # spec decoding out of scope (§7.5)
    "spec_qwen8b_eagle3_cag": _legacy("gold-reuse"),
    "spec_qwen8b_ngram_cag": _legacy("gold-reuse"),
    "spec_qwen8b_eagle3_rag": _legacy("retr-fresh", "dense"),  # spec matrix ran unranked
    "spec_qwen8b_ngram_rag": _legacy("retr-fresh", "dense"),
}


def from_legacy(name: str) -> CellSpec:
    """Translate a pilot-era baseline name into its charter cell tuple.

    Fails closed: names outside ``LEGACY_ALIASES`` raise ``UnknownBaselineError``.
    Returned specs carry documented best-effort defaults (anchor model, vLLM,
    single topology, F1) — see ``LEGACY_ALIASES`` for the per-name exceptions.
    """
    try:
        spec = LEGACY_ALIASES[name]
    except KeyError:
        raise UnknownBaselineError(
            f"unknown legacy baseline {name!r}; known: {sorted(LEGACY_ALIASES)}"
        ) from None
    return replace(spec)
