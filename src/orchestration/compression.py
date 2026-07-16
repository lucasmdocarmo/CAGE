"""Text/context compression for the CAGE compression axis (`compressed_rag` baseline).

Wraps LLMLingua / LLMLingua-2 (Microsoft, MIT) to compress retrieved documents before
they are placed in the prompt — the standard RAG-side compression studied by RECOMP and
LongLLMLingua. KV-cache compression (the `compressed_cag` arm) is server-side and handled
via vLLM `--kv-cache-dtype fp8` / an MLA model, not here.

HONEST LABEL (B3c, 2026-07-16 audit): the default LLMLingua-2 path is QUESTION-BLIND.
`compress_prompt_llmlingua2` ignores the `question` kwarg entirely — LLMLingua-2 is
TASK-AGNOSTIC prompt compression by design (Pan et al. 2024, "LLMLingua-2: Data
Distillation for Efficient and Faithful Task-Agnostic Prompt Compression"). The
`question` parameter is kept only for API compatibility with the v1 LLMLingua path
(which is question-aware). Do not describe the compressed_rag arm as query-conditioned.

Operating point (B3a, 2026-07-16 audit): the LLMLingua-2 default
`use_context_level_filter=True` STACKS a coarse context-dropping stage on top of the
token-level filter — a configured 2x target compressed to ~3.3x in the audit. We pass
`use_context_level_filter=False` (token-level only) and pin the target in TOKENS of the
concatenated context (`target_token` from the actual tokenizer count), so the achieved
ratio tracks the configured knob.

Design:
- Lazy, optional dependency. If `llmlingua` is not installed (or `CAGE_DISABLE_COMPRESSION=1`),
  the compressor is a transparent pass-through that returns the originals with ratio 1.0 and a
  reason — so a run never hard-fails, it just records "no compression applied".
- Returns the compressed texts plus a stats dict (original/compressed token counts, ratio,
  and the wall-clock compression_latency_ms — compression cost is part of the arm's
  end-to-end latency story, not free).
- Rate knob: CAGE_COMPRESSION_RATE (fraction of tokens to KEEP, default 0.5) overrides the
  per-baseline `target_ratio` so multi-rate sweeps are an env/config knob, no code edits.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Default fraction of tokens to KEEP (0.5 = 2x compression) when neither the
# CAGE_COMPRESSION_RATE env var nor an explicit target_ratio is provided.
DEFAULT_COMPRESSION_RATE = 0.5


@dataclass
class CompressionStats:
    method: str
    target_ratio: float            # fraction of tokens we aimed to keep
    original_tokens: int
    compressed_tokens: int
    applied: bool
    note: str = ""
    # B3b: wall-clock cost of the compress() call itself (time.perf_counter). None when
    # compression was not attempted (pass-through/disabled). Compression latency is part
    # of the compressed_rag arm's end-to-end serving cost and must reach results.csv.
    compression_latency_ms: Optional[float] = None

    @property
    def compression_ratio(self) -> float:
        """compressed / original tokens (1.0 = no compression, 0.5 = 2x smaller)."""
        return (self.compressed_tokens / self.original_tokens) if self.original_tokens else 1.0

    def to_dict(self) -> dict:
        return {
            "compress_method": self.method,
            "compress_target_ratio": self.target_ratio,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "compression_applied": self.applied,
            "compression_note": self.note,
            "compression_latency_ms": self.compression_latency_ms,
        }


class ContextCompressor:
    """LLMLingua-backed compressor for retrieved context documents."""

    # method -> (llmlingua model_name, use_llmlingua2)
    _MODELS = {
        "llmlingua2": ("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", True),
        "llmlingua": ("NousResearch/Llama-2-7b-hf", False),  # heavier; needs a causal LM
    }

    def __init__(self, method: str = "llmlingua2", device: str = "cpu"):
        self.method = method
        self.device = device
        self._compressor = None
        self._disabled_reason: Optional[str] = None
        if os.getenv("CAGE_DISABLE_COMPRESSION", "").strip().lower() in {"1", "true", "yes"}:
            self._disabled_reason = "CAGE_DISABLE_COMPRESSION set"
        # Strict by DEFAULT: refuse to silently no-op. In Phase 2 a missing llmlingua package
        # made compressed_rag fall through to ratio 1.0 (compression_applied=False) for all
        # rows, silently invalidating the baseline. So a missing/failed compressor now RAISES
        # unless the operator explicitly opts out with CAGE_ALLOW_NO_COMPRESSION=1 (e.g. a
        # local dry-run). If compression was explicitly DISABLED, pass-through is intended, so
        # strictness is moot. (CAGE_REQUIRE_COMPRESSION is kept only for back-compat and is now
        # redundant with the strict default.)
        _allow_noop = os.getenv("CAGE_ALLOW_NO_COMPRESSION", "").strip().lower() in {"1", "true", "yes"}
        self._strict = (not _allow_noop) and (self._disabled_reason is None)

    @property
    def compressor(self):
        if self._compressor is None and self._disabled_reason is None:
            try:
                from llmlingua import PromptCompressor  # type: ignore
                model_name, use_v2 = self._MODELS.get(self.method, self._MODELS["llmlingua2"])
                self._compressor = PromptCompressor(
                    model_name=model_name,
                    use_llmlingua2=use_v2,
                    device_map=self.device,
                )
            except Exception as e:  # missing package or model
                self._disabled_reason = str(e)
                msg = (
                    f"LLMLingua unavailable ({e}); compressed_rag would fall back to NO "
                    f"compression (ratio 1.0). `pip install llmlingua` to enable."
                )
                if self._strict:
                    raise RuntimeError(f"CAGE_REQUIRE_COMPRESSION=1 but {msg}") from e
                print(f"Warning: {msg}")
        return self._compressor

    @staticmethod
    def _approx_tokens(texts: List[str]) -> int:
        # Whitespace token count is a stable, model-agnostic proxy for the ratio.
        return sum(len(t.split()) for t in texts if t)

    @staticmethod
    def _resolve_rate(target_ratio: Optional[float]) -> float:
        """Resolve the fraction of tokens to KEEP (B3a rate knob).

        Precedence: CAGE_COMPRESSION_RATE env var (multi-rate sweeps flip ONE env var,
        no per-baseline config edits) > explicit target_ratio argument > 0.5 default.
        """
        env_rate = os.getenv("CAGE_COMPRESSION_RATE", "").strip()
        if env_rate:
            try:
                rate = float(env_rate)
                if 0.0 < rate <= 1.0:
                    return rate
                print(f"Warning: CAGE_COMPRESSION_RATE={env_rate} outside (0, 1]; ignoring.")
            except ValueError:
                print(f"Warning: CAGE_COMPRESSION_RATE={env_rate!r} is not a float; ignoring.")
        if target_ratio is not None:
            return target_ratio
        return DEFAULT_COMPRESSION_RATE

    def _actual_token_count(self, comp, texts: List[str]) -> int:
        """Token count of the concatenated context in the COMPRESSOR's own tokenizer.

        B3a: `target_token` must be computed from the actual token count the compressor
        operates on, so the achieved ratio is anchored in TOKENS of the concatenated
        context, not the whitespace proxy. Falls back to the whitespace count when the
        backend exposes no usable tokenizer.
        """
        joined = "\n\n".join(texts)
        try:
            tokenizer = getattr(comp, "tokenizer", None)
            if tokenizer is not None:
                return len(tokenizer(joined, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
        return self._approx_tokens(texts)

    def compress(
        self,
        contexts: List[str],
        question: str = "",
        target_ratio: Optional[float] = None,
    ) -> Tuple[List[str], CompressionStats]:
        """Compress a list of context docs. Returns (compressed_docs, stats).

        NOTE (B3c): with the default LLMLingua-2 backend this is QUESTION-BLIND,
        task-agnostic compression (Pan et al. 2024) — the `question` argument is
        forwarded for the question-aware v1 path only and is IGNORED by
        `compress_prompt_llmlingua2`.

        target_ratio: fraction of tokens to KEEP; overridden by CAGE_COMPRESSION_RATE
        (see _resolve_rate). Default 0.5 (2x compression).
        """
        rate = self._resolve_rate(target_ratio)
        nonempty = [c for c in (contexts or []) if c and c.strip()]
        original_tokens = self._approx_tokens(nonempty)
        if not nonempty:
            return list(contexts or []), CompressionStats(
                self.method, rate, 0, 0, applied=False, note="empty context"
            )

        comp = self.compressor
        if comp is None:
            if self._strict:
                raise RuntimeError(
                    f"CAGE_REQUIRE_COMPRESSION=1 but compressor unavailable: "
                    f"{self._disabled_reason or 'unknown'}"
                )
            return list(contexts), CompressionStats(
                self.method, rate, original_tokens, original_tokens,
                applied=False, note=self._disabled_reason or "compressor unavailable",
            )

        try:
            # B3a operating point: pin the target in TOKENS of the concatenated context
            # (actual tokenizer count, not the whitespace proxy) and disable the
            # context-level filter — LLMLingua-2's default context filter STACKED on the
            # token filter and drove a configured 2x to ~3.3x in the 2026-07-16 audit.
            actual_tokens = self._actual_token_count(comp, nonempty)
            target_token = max(1, int(actual_tokens * rate))
            # B3b: compression latency is real serving-path cost; measure the call itself.
            _t0 = time.perf_counter()
            result = comp.compress_prompt(
                nonempty,
                # Question-BLIND on the LLMLingua-2 path (task-agnostic, Pan et al. 2024);
                # forwarded only for the question-aware v1 LLMLingua backend.
                question=question,
                rate=rate,                # fraction of tokens to keep
                target_token=target_token,  # anchors the ratio in actual tokens
                use_context_level_filter=False,  # token-level only (B3a)
                force_tokens=["\n", "?"],
            )
            latency_ms = (time.perf_counter() - _t0) * 1000.0
            compressed_text = result.get("compressed_prompt", "") if isinstance(result, dict) else str(result)
            compressed_docs = [compressed_text] if compressed_text else nonempty
            compressed_tokens = self._approx_tokens(compressed_docs)
            return compressed_docs, CompressionStats(
                self.method, rate, original_tokens, compressed_tokens, applied=True,
                compression_latency_ms=latency_ms,
            )
        except Exception as e:
            if self._strict:
                raise RuntimeError(f"CAGE_REQUIRE_COMPRESSION=1 but compression failed: {e}") from e
            print(f"Warning: compression failed ({e}); using original context.")
            return list(contexts), CompressionStats(
                self.method, rate, original_tokens, original_tokens,
                applied=False, note=f"compress error: {e}",
            )
