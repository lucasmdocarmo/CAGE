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
  prefix; the FULL prompt is tokenized once, exactly as the plain path
  tokenizes it, its first ``base_len`` ids must equal the preloaded corpus
  ids (else ``CorpusPrefixMismatchError``, a protocol violation, never an
  error row), and the full ids are handed to ``generate`` beside the cache
  (transformers 4.57 slices the cached part off itself and refused
  suffix-only ids with an IndexError on the 2026-10-08 landing, S0F-60).
  Review 2026-10-10 (F-01): tokenizing the suffix on its own and
  concatenating it produced different ids than the whole prompt, because
  Qwen3's BPE merges across the cut (``.`` + ``\\n\\n`` is one token in the
  whole prompt and two tokens when split), so the reuse path answered a
  different input than the plain path. The caller places the cut after the
  ``\\n\\n`` before ``Question:`` so the two tokenizations agree, and this
  adapter verifies it on every request;
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
the greedy oracle, a prompt that does not extend the loaded corpus prefix)
raise loudly instead of producing rows under wrong semantics.

Stop sequences (S0F-60): the request's stop list is honored through a
transformers ``StoppingCriteria`` that decodes the generated suffix after
every step and ends generation once a stop string appears; the returned
text is cut before the first stop string and ``finish_reason`` is "stop",
the campaign's ``stop=["\n"]`` convention served exactly as on the serving
engines. Before S0F-60 the oracle refused stop lists and the runner sent
none, so every oracle row ran to ``max_tokens`` (256 tokens of text past
the answer on the 2026-10-08 landing).

Cached-token telemetry is self-instrumented (charter D2.1: "N/A -- we
instrument it ourselves (it's our code path)"): in corpus-reuse mode
``cached_prompt_tokens`` is exactly the resident corpus KV length; without a
loaded prefix it is exactly 0. Both are facts, not engine claims.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .engine import (
    InferenceEngine,
    InferenceRequest,
    InferenceResponse,
    NUM_TOKENS_SOURCE_TOKEN_IDS,
)
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


def _import_stopping_criteria() -> Tuple[Any, Any]:
    """Lazily import (StoppingCriteria, StoppingCriteriaList) (S0F-60).

    A second module-level seam beside ``_import_ml_stack`` so the unit tests
    install fakes without touching the four-tuple that seam returns.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    return StoppingCriteria, StoppingCriteriaList


class CorpusPrefixMismatchError(ValueError):
    """The full prompt's leading ids differ from the preloaded corpus ids.

    The KV cache holds the corpus ids as prefilled; serving a prompt whose own
    tokenization starts differently (a BPE merge across the prefix cut, or a
    prefix cut at the wrong character) would attend a cache that does not
    match the input. Raised BEFORE generation and propagated as a protocol
    violation (review 2026-10-10, F-01), never recorded as an error row.
    """


def _cut_at_stop(text: str, stops: Sequence[str]) -> Tuple[str, bool]:
    """The text before the earliest stop string, and whether one was found."""
    cuts = [i for i in (text.find(s) for s in stops if s) if i >= 0]
    if not cuts:
        return text, False
    return text[: min(cuts)], True


class HFOracleAdapter(InferenceEngine):
    """T=0 batch-1 HF Transformers reference engine (the idea-gain zero point)."""

    engine_id: str = ENGINE_ID

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        enforce_greedy: bool = True,
        device_map: Optional[str] = None,
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
            device_map: None (default: load, then ``.to(device)``) or
                "auto": shard across the box's GPUs at load (C15,
                2026-10-10; session b's llama-3.3-70b oracle cells do not
                fit one GPU). Inputs then go to the first parameter's device.
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
        if device_map not in (None, "auto"):
            raise ValueError(f"unknown device_map {device_map!r} (expected None or 'auto')")
        self.device_map = device_map
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
        load_kwargs: Dict[str, Any] = {"torch_dtype": dtype_map[dtype]}
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device_map is None:
            self.model.to(self.device)
        else:
            # The loader placed the shards; inputs go where the first
            # parameter (the embedding) lives [V: transformers 4.57.6
            # PreTrainedModel.device -> modeling_utils.get_parameter_device].
            self.device = str(self.model.device)
        self.model.eval()

        # Corpus-prefix reuse state (the manual CAG recipe, chan2024cag).
        self._corpus_cache: Optional[Any] = None
        self._corpus_input_ids: Optional[Any] = None  # the prefilled ids (S0F-60)
        self._corpus_add_special_tokens: bool = True  # how the prefix was tokenized
        self._corpus_prefix_text: Optional[str] = None
        self._corpus_base_len: int = 0
        self._corpus_prefill_ms: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _sync(self) -> None:
        """CUDA barrier so perf_counter spans measure completed device work.

        Fresh review 2026-10-10 (C15-1): ``torch.cuda.synchronize()`` waits on
        the current device only; under ``device_map="auto"`` the layers span
        several GPUs, so every device is synchronized.
        """
        if not str(self.device).startswith("cuda"):
            return
        if self.device_map is None:
            self._torch.cuda.synchronize()
            return
        for index in range(int(self._torch.cuda.device_count())):
            self._torch.cuda.synchronize(index)

    def _pad_token_id(self) -> Optional[int]:
        """Tokenizer pad id, falling back to EOS (run_cag_reference.py convention)."""
        pad = self.tokenizer.pad_token_id
        return pad if pad is not None else self.tokenizer.eos_token_id

    def _validate_request(self, request: InferenceRequest) -> None:
        """Fail closed on requests the T=0 greedy oracle cannot honor
        (sampling temperature, truncate_prompt_tokens); stop lists are
        honored since S0F-60 (see ``_stop_criteria``)."""
        if self.enforce_greedy and (request.temperature or 0.0) != 0.0:
            raise ValueError(
                f"HFOracleAdapter is the T=0 greedy reference engine (charter "
                f"D2.1); request '{request.request_id}' asked for temperature="
                f"{request.temperature}. Construct oracle requests with "
                f"temperature=0.0 (or pass enforce_greedy=False deliberately)."
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
        self._corpus_input_ids = enc.input_ids
        self._corpus_add_special_tokens = bool(add_special_tokens)
        self._corpus_prefix_text = prefix_text
        self._corpus_base_len = int(cache.get_seq_length())
        self._corpus_prefill_ms = prefill_ms
        return prefill_ms

    def clear_corpus_prefix(self) -> None:
        """Release the resident corpus KV cache (true CAG serves one corpus at
        a time -- run_cag_reference.py releases before the next block's prefill)."""
        if self._corpus_cache is not None:
            self._corpus_cache = None
            if str(self.device).startswith("cuda"):
                self._torch.cuda.empty_cache()
        self._corpus_input_ids = None
        self._corpus_prefix_text = None
        self._corpus_base_len = 0
        self._corpus_prefill_ms = None

    def _stop_classes(self) -> Tuple[Any, Any]:
        """(StoppingCriteria, StoppingCriteriaList), or the typed fail-closed
        raise; called BEFORE the timed block so a missing import never becomes
        an error row (Fable review 2026-10-09, L1)."""
        try:
            return _import_stopping_criteria()
        except ImportError as exc:
            raise EngineDependencyUnavailableError(
                ENGINE_ID, "transformers.StoppingCriteria", str(exc)
            ) from exc

    def _stop_criteria(
        self, stops: Sequence[str], prompt_len: int, classes: Tuple[Any, Any]
    ) -> Tuple[Any, Any]:
        """A StoppingCriteriaList ending generation once the decoded generated
        text (the ids past ``prompt_len``) carries one of ``stops`` (S0F-60).

        Returns the list and the criterion (its ``matched`` flag records
        whether a stop string ended the generation). Decoding the generated
        suffix at every step is O(n) per step over at most max_tokens ids,
        far below one forward pass of the model.
        """
        StoppingCriteria, StoppingCriteriaList = classes
        tokenizer = self.tokenizer
        torch = self._torch

        class _StopOnStrings(StoppingCriteria):  # type: ignore[misc,valid-type]
            def __init__(self) -> None:
                super().__init__()
                self.matched = False

            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                text = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True)
                self.matched = any(s in text for s in stops)
                return torch.full(
                    (input_ids.shape[0],), self.matched, dtype=torch.bool, device=input_ids.device
                )

        criterion = _StopOnStrings()
        return StoppingCriteriaList([criterion]), criterion

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
        prompt MUST literally extend the cached prefix text; the WHOLE prompt
        is tokenized once (F-01), its leading ids must equal the cached corpus
        ids, the full ids are handed to ``generate`` with the resident cache
        (S0F-60), and the cache is cropped back to the corpus length after
        the query -- NON-OPTIONAL per Chan et al. 2024 (else the next query
        attends to this one's question and answer). A stop list ends the
        generation at the first stop string and the text is cut before it.

        Raises (fail-closed protocol violations, never error rows):
            ValueError: sampling temperature on the greedy oracle, or a prompt
                that does not extend the loaded corpus prefix.
            CorpusPrefixMismatchError (a ValueError): the full prompt's leading
                ids differ from the cached corpus ids, or no suffix remains.
            EngineCapabilityUnavailableError: truncate_prompt_tokens requested.
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
        stop_matched = False
        stops: List[str] = [s for s in (request.stop or []) if s]
        stop_classes = self._stop_classes() if stops else None  # fail closed, before the clock
        error: Optional[str] = None

        self._sync()
        t0 = time.perf_counter()
        try:
            gen_kwargs: Dict[str, Any] = dict(
                max_new_tokens=request.max_tokens,
                do_sample=False,
                pad_token_id=self._pad_token_id(),
            )
            if reuse:
                assert self._corpus_prefix_text is not None
                # Review 2026-10-10 (F-01): the FULL prompt is tokenized once,
                # the way the plain path and the preload tokenize, so a BPE
                # merge across the prefix cut can never make the reuse path
                # answer different ids than the plain path. The leading
                # base_len ids must equal the prefilled corpus ids, or the
                # cache does not match the input: refuse, never an error row.
                full = self.tokenizer(
                    request.prompt, return_tensors="pt",
                    add_special_tokens=self._corpus_add_special_tokens,
                ).to(self.device)
                base_len = self._corpus_base_len
                prompt_len = int(full.input_ids.shape[1])
                if prompt_len <= base_len:
                    raise CorpusPrefixMismatchError(
                        f"prompt tokenizes to {prompt_len} ids, not more than the "
                        f"{base_len} cached corpus ids: no query suffix to serve"
                    )
                head = full.input_ids[:, :base_len]
                if not torch.equal(head, self._corpus_input_ids):
                    raise CorpusPrefixMismatchError(
                        "the full prompt's first ids differ from the preloaded corpus "
                        "ids (a tokenizer merge across the prefix cut, or a prefix "
                        "derived at the wrong character); serving it would attend a "
                        "KV cache that does not match the input (review 2026-10-10, "
                        "F-01). Cut the cacheable prefix after the '\\n\\n' before "
                        "'Question:' (derive_corpus_prompt_prefix)."
                    )
                # ADR-0118 / review F-05: prompt_tokens is the WHOLE prompt the
                # row was served against (cached corpus + query), the same
                # quantity the serving engines report, so cached / prompt is
                # a ratio in [0, 1] (the suffix-only count read above 1).
                prompt_tokens = prompt_len
                # S0F-60: generate() takes the FULL ids beside the prefilled
                # cache (it slices the cached part off itself); the suffix
                # alone raised IndexError in transformers 4.57 on the landing.
                gen_kwargs["input_ids"] = full.input_ids
                gen_kwargs["attention_mask"] = full.attention_mask
                gen_kwargs["past_key_values"] = self._corpus_cache
            else:
                enc = self.tokenizer(request.prompt, return_tensors="pt").to(self.device)
                prompt_len = int(enc.input_ids.shape[1])
                prompt_tokens = prompt_len
                gen_kwargs["input_ids"] = enc.input_ids
                gen_kwargs["attention_mask"] = enc.attention_mask
            criterion = None
            if stop_classes is not None:
                gen_kwargs["stopping_criteria"], criterion = self._stop_criteria(
                    stops, prompt_len, stop_classes
                )
            with torch.no_grad():
                out = self.model.generate(**gen_kwargs)
            answer = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
            num_generated = int(out.shape[1]) - prompt_len
            answer, cut = _cut_at_stop(answer, stops)
            stop_matched = cut or bool(criterion is not None and criterion.matched)
        except CorpusPrefixMismatchError:
            raise  # a protocol violation, never an error row (the finally still crops)
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
                finish_reason=(
                    "stop"
                    if stop_matched or num_generated < request.max_tokens
                    else "length"
                ),
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
        # ADR-0118 (W5): the count is the generated id count, never a word count.
        response.num_tokens_source = NUM_TOKENS_SOURCE_TOKEN_IDS if error is None else None
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
            if str(self.device).startswith("cuda"):
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
