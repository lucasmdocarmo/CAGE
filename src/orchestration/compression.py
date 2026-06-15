"""Text/context compression for the CAGE compression axis (`compressed_rag` baseline).

Wraps LLMLingua / LLMLingua-2 (Microsoft, MIT) to compress retrieved documents before
they are placed in the prompt — the standard RAG-side compression studied by RECOMP and
LongLLMLingua. KV-cache compression (the `compressed_cag` arm) is server-side and handled
via vLLM `--kv-cache-dtype fp8` / an MLA model, not here.

Design:
- Lazy, optional dependency. If `llmlingua` is not installed (or `CAGE_DISABLE_COMPRESSION=1`),
  the compressor is a transparent pass-through that returns the originals with ratio 1.0 and a
  reason — so a run never hard-fails, it just records "no compression applied".
- Returns the compressed texts plus a stats dict (original/compressed token counts, ratio).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CompressionStats:
    method: str
    target_ratio: float            # fraction of tokens we aimed to keep
    original_tokens: int
    compressed_tokens: int
    applied: bool
    note: str = ""

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
                print(
                    f"Warning: LLMLingua unavailable ({e}); compressed_rag falls back to "
                    f"NO compression (ratio 1.0). `pip install llmlingua` to enable."
                )
        return self._compressor

    @staticmethod
    def _approx_tokens(texts: List[str]) -> int:
        # Whitespace token count is a stable, model-agnostic proxy for the ratio.
        return sum(len(t.split()) for t in texts if t)

    def compress(
        self,
        contexts: List[str],
        question: str = "",
        target_ratio: float = 0.5,
    ) -> Tuple[List[str], CompressionStats]:
        """Compress a list of context docs. Returns (compressed_docs, stats)."""
        nonempty = [c for c in (contexts or []) if c and c.strip()]
        original_tokens = self._approx_tokens(nonempty)
        if not nonempty:
            return list(contexts or []), CompressionStats(
                self.method, target_ratio, 0, 0, applied=False, note="empty context"
            )

        comp = self.compressor
        if comp is None:
            return list(contexts), CompressionStats(
                self.method, target_ratio, original_tokens, original_tokens,
                applied=False, note=self._disabled_reason or "compressor unavailable",
            )

        try:
            # LLMLingua compresses the concatenated context conditioned on the question.
            result = comp.compress_prompt(
                nonempty,
                question=question,
                rate=target_ratio,        # fraction of tokens to keep
                force_tokens=["\n", "?"],
            )
            compressed_text = result.get("compressed_prompt", "") if isinstance(result, dict) else str(result)
            compressed_docs = [compressed_text] if compressed_text else nonempty
            compressed_tokens = self._approx_tokens(compressed_docs)
            return compressed_docs, CompressionStats(
                self.method, target_ratio, original_tokens, compressed_tokens, applied=True,
            )
        except Exception as e:
            print(f"Warning: compression failed ({e}); using original context.")
            return list(contexts), CompressionStats(
                self.method, target_ratio, original_tokens, original_tokens,
                applied=False, note=f"compress error: {e}",
            )
