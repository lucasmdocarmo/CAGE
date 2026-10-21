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
- A14 (ADR-0174 amendment 2026-10-20): off the CPU the weights load in float32, the CPU's
  precision (llmlingua 0.2.2 prompt_compressor.py:139-144 loads "auto" on device "cuda");
  CAGE_COMPRESSOR_REVISION pins the HF commit; CAGE_COMPRESSION_CACHE_DIR keeps one memo
  file per distinct input, so each passage set is compressed once per run; release()
  drops the weights before a measured window and the next compress() reloads them.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default fraction of tokens to KEEP (0.5 = 2x compression) when neither the
# CAGE_COMPRESSION_RATE env var nor an explicit target_ratio is provided.
DEFAULT_COMPRESSION_RATE = 0.5

#: A14 item 2: the run-level memo folder (the driver sets it per run). Unset = no memo.
COMPRESSION_CACHE_ENV = "CAGE_COMPRESSION_CACHE_DIR"
#: A14 item 8: the HF commit the compressor loads at (freeze slot
#: INSTRUMENT_REVISIONS.compressor). Unset = the repo's current head.
COMPRESSOR_REVISION_ENV = "CAGE_COMPRESSOR_REVISION"
#: B3a operating point, shared by the call and the memo key.
FORCE_TOKENS = ["\n", "?"]


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
    # A14 item 2: True when the memo served this input (compression_latency_ms is then
    # the original computation's time), False when computed, None when no memo is set.
    cache_hit: Optional[bool] = None

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
            "compression_cache_hit": self.cache_hit,
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
        self.revision: Optional[str] = (os.getenv(COMPRESSOR_REVISION_ENV) or "").strip() or None
        _cache = (os.getenv(COMPRESSION_CACHE_ENV) or "").strip()
        self.cache_dir: Optional[Path] = Path(_cache) if _cache else None

    @property
    def compressor(self):
        if self._compressor is None and self._disabled_reason is None:
            try:
                from llmlingua import PromptCompressor  # type: ignore
                model_name, use_v2 = self._MODELS.get(self.method, self._MODELS["llmlingua2"])
                # A14 items 5 and 8: model_config reaches AutoConfig, AutoTokenizer and
                # from_pretrained (llmlingua 0.2.2 prompt_compressor.py:116-149); it is
                # passed only when it carries a key, so the CPU unpinned call is S0's.
                model_config: Dict[str, Any] = {}
                if self.revision:
                    model_config["revision"] = self.revision
                if self.device != "cpu":
                    import torch  # type: ignore

                    model_config["torch_dtype"] = torch.float32
                self._compressor = PromptCompressor(
                    model_name=model_name,
                    use_llmlingua2=use_v2,
                    device_map=self.device,
                    **({"model_config": model_config} if model_config else {}),
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

    def release(self) -> bool:
        """A14 item 6: drop the loaded weights (True when there were any); the next
        compress() reloads them on the same device."""
        released = self._compressor is not None
        self._compressor = None
        return released

    def _memo_key(self, docs: List[str], question: str, rate: float) -> str:
        """sha256 over every input the compressed text depends on (A14 item 2)."""
        model_name, use_v2 = self._MODELS.get(self.method, self._MODELS["llmlingua2"])
        try:
            from importlib.metadata import version

            library = version("llmlingua")
        except Exception:  # noqa: BLE001 -- an unversioned install keys as None
            library = None
        payload = {
            "method": self.method,
            "model": model_name,
            "revision": self.revision,
            "llmlingua": library,
            "device": self.device,
            "dtype": "float32",  # the CPU default and the forced precision elsewhere
            "rate": rate,
            "use_context_level_filter": False,
            "force_tokens": FORCE_TOKENS,
            # LLMLingua-2 ignores the question (module docstring), so it is no input there
            "question": None if use_v2 else question,
            "contexts": docs,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _memo_get(self, key: str) -> Optional[Dict[str, Any]]:
        """The stored record for ``key``, or None (absent, or unreadable and recomputed)."""
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            docs = record["compressed_docs"]
            if record.get("key") != key or not isinstance(docs, list) or not all(
                isinstance(d, str) for d in docs
            ):
                raise ValueError("malformed memo record")
            return record
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Warning: compression memo {path} unreadable ({exc}); recomputing.")
            return None

    def _memo_put(self, key: str, docs: List[str], latency_ms: float) -> None:
        """Atomic write (tmp file + os.replace); a failed write costs time, never data."""
        path = self.cache_dir / f"{key}.json"
        tmp = self.cache_dir / f".{key}.{os.getpid()}.tmp"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(
                    {"key": key, "compressed_docs": docs, "compression_latency_ms": latency_ms},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as exc:
            print(f"Warning: compression memo write to {path} failed ({exc}); serving the computed text.")

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
                # S0F-88 audit (U10): this call only counts tokens, and on S0
                # attempt 4 it was the source of transformers' "sequence length
                # is longer than the specified maximum" warning; the count is
                # what B3a needs, so the warning is dropped (verbose=False).
                return len(tokenizer(joined, add_special_tokens=False, verbose=False)["input_ids"])
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

        # a disabled compressor passes the originals through, memo or not
        memo_key = (
            self._memo_key(nonempty, question, rate)
            if self.cache_dir is not None and self._disabled_reason is None
            else None
        )
        if memo_key is not None:
            hit = self._memo_get(memo_key)
            if hit is not None:
                docs = list(hit["compressed_docs"])
                return docs, CompressionStats(
                    self.method, rate, original_tokens, self._approx_tokens(docs), applied=True,
                    compression_latency_ms=hit.get("compression_latency_ms"), cache_hit=True,
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
                force_tokens=FORCE_TOKENS,
            )
            latency_ms = (time.perf_counter() - _t0) * 1000.0
            compressed_text = result.get("compressed_prompt", "") if isinstance(result, dict) else str(result)
            compressed_docs = [compressed_text] if compressed_text else nonempty
            compressed_tokens = self._approx_tokens(compressed_docs)
            if memo_key is not None:
                self._memo_put(memo_key, compressed_docs, latency_ms)
            return compressed_docs, CompressionStats(
                self.method, rate, original_tokens, compressed_tokens, applied=True,
                compression_latency_ms=latency_ms,
                cache_hit=False if memo_key is not None else None,
            )
        except Exception as e:
            if self._strict:
                raise RuntimeError(f"CAGE_REQUIRE_COMPRESSION=1 but compression failed: {e}") from e
            print(f"Warning: compression failed ({e}); using original context.")
            return list(contexts), CompressionStats(
                self.method, rate, original_tokens, original_tokens,
                applied=False, note=f"compress error: {e}",
            )
