"""HF Transformers oracle adapter for the CAGE framework (ADR-0007 / charter D2).

HFOracleAdapter implements the InferenceEngine interface for the T=0 batch-1
HuggingFace reference path -- the charter's "idea-gain zero point": reuse
without management. Plain HF transformers has no PagedAttention, no continuous
batching, no scheduler, no CUDA graphs; KV reuse exists ONLY by manually
holding one contiguous ``DynamicCache`` and cropping it back after every
query. That manual recipe is Cache-Augmented Generation exactly as in
Chan et al. 2024 (arXiv:2412.15605, "Don't Do RAG"; reference impl
github.com/hhhuang/CAG) [chan2024cag], and the mechanics here mirror
scripts/3_run/run_cag_reference.py verbatim:

- ``preload_corpus_prefix(text)``: prefill ONE fixed corpus block's KV with a
  single forward pass into a ``DynamicCache`` (recording corpus_prefill_ms);
- ``generate(request)``: the request prompt must literally extend the cached
  prefix; only the suffix is tokenized (``add_special_tokens=False``) and
  decoded greedily against the cache with an attention mask covering
  cached-corpus + query tokens;
- after EVERY query the cache is cropped back to the corpus length
  (``cache.crop(base_len)``) so query B never attends to query A's tokens --
  NON-OPTIONAL per the Chan et al. recipe: skipping it silently corrupts
  every subsequent row.

Serving metrics are honestly labeled reference-engine: there is no streaming,
so ``ttft_ms == total_time_ms`` (the whole ``generate()`` call -- the CAG
"TTFT-equivalent" of run_cag_reference.py), every response is stamped
``engine_id="hf_reference"`` / ``reference_engine=True``, and these numbers
are NOT comparable to serving-engine arms -- compare hf_reference rows only
against other hf_reference rows (idea-gain / engine-gain attribution).

Fail-closed doctrine (mirrors InstrumentUnavailableError in
src/evaluation/quality.py): torch/transformers import lazily and a missing
stack raises the typed EngineDependencyUnavailableError -- never a silent
degradation to another backend. Protocol violations (sampling temperature on
the greedy oracle, stop sequences, a prompt that does not extend the loaded
corpus prefix) raise loudly instead of producing rows under wrong semantics.

Cached-token telemetry is self-instrumented (charter D2.1: "N/A -- we
instrument it ourselves (it's our code path)"): in corpus-reuse mode
``cached_prompt_tokens`` is exactly the resident corpus KV length; without a
loaded prefix it is exactly 0. Both are facts, not engine claims.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .engine import InferenceEngine, InferenceRequest, InferenceResponse
from .errors import EngineCapabilityUnavailableError, EngineDependencyUnavailableError

ENGINE_ID = "hf_reference"


def _import_ml_stack() -> Tuple[Any, Any, Any, Any]:
    """Lazily import (torch, AutoModelForCausalLM, AutoTokenizer, DynamicCache).

    Module-level seam so unit tests can monkeypatch it with fakes (no GPU or
    ML stack in tests). ImportError propagates untouched; the adapter converts
    it into the typed EngineDependencyUnavailableError (fail-closed doctrine,
    mirroring src/evaluation/quality.py's InstrumentUnavailableError).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    return torch, AutoModelForCausalLM, AutoTokenizer, DynamicCache


class HFOracleAdapter(InferenceEngine):
    """T=0 batch-1 HF Transformers reference engine (the idea-gain zero point)."""

    engine_id: str = ENGINE_ID

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        enforce_greedy: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load the HF model/tokenizer (fail-closed on a missing ML stack).

        Args:
            model_name: HF model id (or local path).
            device: "auto" (cuda if available else cpu), "cuda", or "cpu".
            dtype: "bfloat16" | "float16" | "float32". An unknown string
                raises (no silent default -- the S3 silent-dtype-fallback
                anti-pattern from the 2026-08-04 review).
            enforce_greedy: If True (default), a request whose temperature is
                not 0.0 raises: the oracle is T=0 greedy BY CONSTRUCTION
                (charter D2.1) and must never silently ignore a sampling
                request.
            **kwargs: Forwarded to InferenceEngine (stored in self.config).

        Raises:
            EngineDependencyUnavailableError: torch/transformers not importable.
            ValueError: unknown device/dtype string.
        """
        super().__init__(model_name, **kwargs)
        try:
            torch, AutoModelForCausalLM, AutoTokenizer, DynamicCache = _import_ml_stack()
        except ImportError as exc:
            raise EngineDependencyUnavailableError(
                ENGINE_ID, "torch+transformers", str(exc)
            ) from exc

        self._torch = torch
        self._DynamicCache = DynamicCache
        self.enforce_greedy = enforce_greedy

        if device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"unknown device '{device}' (expected auto|cuda|cpu)")
        self.device: str = (
            device if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            # Fail closed on an unrecognized dtype string rather than silently
            # defaulting (review finding S3 is exactly that anti-pattern).
            raise ValueError(
                f"unknown dtype '{dtype}' (expected one of {sorted(dtype_map)})"
            )
        self.dtype_name = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype_map[dtype]
        )
        self.model.to(self.device)
        self.model.eval()

        # Corpus-prefix reuse state (the manual CAG recipe, chan2024cag).
        self._corpus_cache: Optional[Any] = None
        self._corpus_prefix_text: Optional[str] = None
        self._corpus_base_len: int = 0
        self._corpus_prefill_ms: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _sync(self) -> None:
        """CUDA barrier so perf_counter spans measure completed device work."""
        if self.device == "cuda":
            self._torch.cuda.synchronize()

    def _pad_token_id(self) -> Optional[int]:
        """Tokenizer pad id, falling back to EOS (run_cag_reference.py convention)."""
        pad = self.tokenizer.pad_token_id
        return pad if pad is not None else self.tokenizer.eos_token_id

    def _validate_request(self, request: InferenceRequest) -> None:
        """Fail closed on requests the T=0 greedy oracle cannot honor."""
        if self.enforce_greedy and (request.temperature or 0.0) != 0.0:
            raise ValueError(
                f"HFOracleAdapter is the T=0 greedy reference engine (charter "
                f"D2.1); request '{request.request_id}' asked for temperature="
                f"{request.temperature}. Construct oracle requests with "
                f"temperature=0.0 (or pass enforce_greedy=False deliberately)."
            )
        if request.stop:
            raise EngineCapabilityUnavailableError(
                ENGINE_ID,
                "stop_sequences",
                "stop-string support is not implemented on the reference path "
                "(run_cag_reference.py never uses it); refusing to silently "
                "ignore the request's stop list",
            )
        if request.truncate_prompt_tokens is not None:
            raise EngineCapabilityUnavailableError(
                ENGINE_ID,
                "truncate_prompt_tokens",
                "vLLM extension parameter; the reference engine serves the "
                "prompt exactly as given",
            )

    # ------------------------------------------------------------------ #
    # Corpus-prefix reuse (the manual CAG recipe -- chan2024cag)
    # ------------------------------------------------------------------ #

    def preload_corpus_prefix(
        self, prefix_text: str, *, add_special_tokens: bool = True
    ) -> float:
        """Prefill the KV cache of ONE fixed corpus prefix (single forward pass).

        Mirrors run_cag_reference.py: the prefix is tokenized exactly as
        served (default special-token handling for the sequence start -- pass
        ``add_special_tokens=False`` for a chat-template-rendered prefix whose
        special tokens are already text), prefilled into a fresh DynamicCache,
        and kept resident; subsequent ``generate()`` calls append their query
        suffix after the cached KV and crop back after every query
        (Chan et al. 2024, arXiv:2412.15605).

        Returns:
            The corpus prefill wall-clock time in milliseconds.
        """
        torch = self._torch
        self.clear_corpus_prefix()

        enc = self.tokenizer(
            prefix_text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).to(self.device)

        cache = self._DynamicCache()
        self._sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            self.model(**enc, past_key_values=cache, use_cache=True)
        self._sync()
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        self._corpus_cache = cache
        self._corpus_prefix_text = prefix_text
        self._corpus_base_len = int(cache.get_seq_length())
        self._corpus_prefill_ms = prefill_ms
        return prefill_ms

    def clear_corpus_prefix(self) -> None:
        """Release the resident corpus KV cache (true CAG serves one corpus at
        a time -- run_cag_reference.py releases before the next block's prefill)."""
        if self._corpus_cache is not None:
            self._corpus_cache = None
            if self.device == "cuda":
                self._torch.cuda.empty_cache()
        self._corpus_prefix_text = None
        self._corpus_base_len = 0
        self._corpus_prefill_ms = None

    # ------------------------------------------------------------------ #
    # InferenceEngine interface
    # ------------------------------------------------------------------ #

    def generate(self, request: InferenceRequest, *, stream: bool = False) -> InferenceResponse:
        """Greedy T=0 batch-1 generation (reference engine).

        Note:
            ``stream`` is accepted for interface compatibility but ignored:
            the reference path has no streaming, so ``ttft_ms`` honestly
            reports the whole generate() call (the CAG "TTFT-equivalent" of
            run_cag_reference.py), never a fabricated first-token estimate.

        With a corpus prefix loaded (``preload_corpus_prefix``), the request
        prompt MUST literally extend the cached prefix text; only the suffix
        is tokenized (``add_special_tokens=False``) and served against the
        resident cache, which is cropped back to the corpus length after the
        query -- NON-OPTIONAL per Chan et al. 2024 (else the next query
        attends to this one's question and answer).

        Raises (fail-closed protocol violations, never error rows):
            ValueError: sampling temperature on the greedy oracle, or a prompt
                that does not extend the loaded corpus prefix.
            EngineCapabilityUnavailableError: stop sequences /
                truncate_prompt_tokens requested.
        """
        self._validate_request(request)
        torch = self._torch

        reuse = self._corpus_cache is not None
        if reuse and not request.prompt.startswith(self._corpus_prefix_text or ""):
            raise ValueError(
                "prompt does not extend the preloaded corpus prefix -- refusing "
                "to serve against a mismatched KV cache (the CAG recipe requires "
                "prompt == corpus_prefix + query_suffix; call "
                "clear_corpus_prefix() for prefix-free serving)"
            )

        answer = ""
        num_generated = 0
        prompt_tokens = 0
        error: Optional[str] = None

        self._sync()
        t0 = time.perf_counter()
        try:
            if reuse:
                assert self._corpus_prefix_text is not None
                suffix_text = request.prompt[len(self._corpus_prefix_text):]
                q_enc = self.tokenizer(
                    suffix_text, return_tensors="pt", add_special_tokens=False
                ).to(self.device)
                q_len = int(q_enc.input_ids.shape[1])
                prompt_tokens = q_len
                # Attention mask must cover cached corpus + new query tokens.
                attention_mask = torch.ones(
                    (1, self._corpus_base_len + q_len),
                    dtype=torch.long,
                    device=self.model.device,
                )
                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=q_enc.input_ids,
                        attention_mask=attention_mask,
                        past_key_values=self._corpus_cache,
                        max_new_tokens=request.max_tokens,
                        do_sample=False,
                        pad_token_id=self._pad_token_id(),
                    )
                answer = self.tokenizer.decode(out[0, q_len:], skip_special_tokens=True)
                num_generated = int(out.shape[1]) - q_len
            else:
                enc = self.tokenizer(request.prompt, return_tensors="pt").to(self.device)
                p_len = int(enc.input_ids.shape[1])
                prompt_tokens = p_len
                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        max_new_tokens=request.max_tokens,
                        do_sample=False,
                        pad_token_id=self._pad_token_id(),
                    )
                answer = self.tokenizer.decode(out[0, p_len:], skip_special_tokens=True)
                num_generated = int(out.shape[1]) - p_len
        except Exception as exc:  # noqa: BLE001 -- record, crop, continue (run_cag_reference.py)
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self._sync()
            total_time_ms = (time.perf_counter() - t0) * 1000.0
            if reuse and self._corpus_cache is not None:
                # NON-OPTIONAL (Chan et al. 2024 recipe): crop back to the
                # corpus length after EVERY query -- else the next query
                # attends to this one's question AND generated answer,
                # silently corrupting every subsequent row.
                self._corpus_cache.crop(self._corpus_base_len)

        if error is not None:
            response = InferenceResponse(
                request_id=request.request_id,
                generated_text="",
                ttft_ms=0.0,
                total_time_ms=total_time_ms,
                num_tokens=0,
                model_name=self.model_name,
                finish_reason="error",
                error=error,
            )
        else:
            response = InferenceResponse(
                request_id=request.request_id,
                generated_text=answer,
                # Reference engine, no streaming: TTFT is the whole generate()
                # call (honest "unobservable -> full response time" convention).
                ttft_ms=total_time_ms,
                total_time_ms=total_time_ms,
                num_tokens=num_generated,
                model_name=self.model_name,
                finish_reason=("length" if num_generated >= request.max_tokens else "stop"),
                prompt_tokens=prompt_tokens,
                # Self-instrumented cache telemetry (charter D2.1: "we
                # instrument it ourselves"): exact resident corpus KV length
                # under reuse; exactly 0 without a loaded prefix.
                cached_prompt_tokens=(self._corpus_base_len if reuse else 0),
            )

        # Honest reference-engine labeling (plain attributes via getattr; the
        # shared InferenceResponse schema stays untouched).
        response.engine_id = ENGINE_ID
        response.reference_engine = True
        response.corpus_prefill_ms = self._corpus_prefill_ms if reuse else None
        response.usage_telemetry_available = error is None
        response.cached_token_telemetry_available = error is None
        return response

    def batch_generate(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Sequential batch-1 generation.

        The reference engine IS the no-batching zero point (charter D2.1:
        scheduler "None (sequential)") -- requests are served strictly one at
        a time by construction, never concurrently.
        """
        return [self.generate(req) for req in requests]

    def is_ready(self) -> bool:
        """Ready once the model is loaded (construction is fail-closed)."""
        return self.model is not None

    def shutdown(self) -> None:
        """Release the corpus cache and the model."""
        self.clear_corpus_prefix()
        if getattr(self, "model", None) is not None:
            self.model = None
            if self.device == "cuda":
                self._torch.cuda.empty_cache()

    def capabilities(self) -> Dict[str, Any]:
        """Capability declaration driving the charter-D2 telemetry-parity gate."""
        return {
            "engine": self.engine_id,
            "serving_grade": False,  # reference engine: numbers NOT comparable to serving arms
            "in_process": True,
            "streamed_ttft": False,  # whole-generate-call latency, honestly labeled
            "cached_token_telemetry": True,  # self-instrumented, exact (D2.1)
            "cached_token_server_flag": None,
            "cached_token_absent_means_zero": False,
            "kv_usage_gauge": False,
            "flush_endpoint": None,  # in-process: clear_corpus_prefix()
            "kv_transfer_params": False,
            "chat_template_thinking_pin": False,  # caller renders prompts (see run_cag_reference.py chat seams)
            "logprobs": False,
            "truncate_prompt_tokens": False,
            "corpus_prefix_reuse": True,  # the manual CAG recipe (chan2024cag)
        }
