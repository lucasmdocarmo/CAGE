"""Typed fail-closed errors for the inference-engine layer.

Mirrors the fail-closed doctrine of ``src/evaluation/quality.py``'s
``InstrumentUnavailableError`` (audit P0-5): a missing dependency or an
unsupported/unverified engine capability raises a TYPED error instead of
silently degrading, substituting, or fabricating a value. A silent fallback
here would let a cell serve (or record telemetry) under semantics different
from what its CellSpec claims, voiding the charter D2 telemetry-parity gate.
"""

from __future__ import annotations


class EngineDependencyUnavailableError(RuntimeError):
    """A dependency an inference engine needs failed to import or load.

    Raised INSTEAD of degrading to a different engine or a stub: an adapter
    that cannot construct its real engine must fail loudly at setup time,
    never serve rows under a different runtime (charter P7 anti-pattern:
    the silent LMDeploy PyTorch-backend fallback).
    """

    def __init__(self, engine: str, dependency: str, cause: str) -> None:
        self.engine = engine
        self.dependency = dependency
        self.cause = cause
        super().__init__(
            f"inference engine '{engine}' dependency '{dependency}' unavailable: {cause}"
        )


class EngineCapabilityUnavailableError(RuntimeError):
    """A requested capability is absent or unverified on this engine.

    Raised when the harness asks an adapter for something the engine does not
    (verifiably) support -- e.g. a cache-flush endpoint that does not exist,
    or a vLLM-only request parameter on a non-vLLM engine. Failing closed here
    keeps 'the request was honored' a fact, never an assumption (charter D2:
    None-with-provenance, never a fabricated number; ADR-0007 consequences).
    """

    def __init__(self, engine: str, capability: str, detail: str) -> None:
        self.engine = engine
        self.capability = capability
        self.detail = detail
        super().__init__(
            f"inference engine '{engine}' capability '{capability}' unavailable: {detail}"
        )
