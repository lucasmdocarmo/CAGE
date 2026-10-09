"""S0F-60: the HF oracle against a real transformers stack (no download).

Two defects of the 2026-10-08 landing, exercised on the real stack (transformers
4.57, the pod's and the Mac's version) with a config-built two-layer Qwen2 and
a word-level tokenizer saved under tmp_path:

1. ``generate()`` with a prefilled DynamicCache needs the FULL ids (cached
   corpus + suffix); the suffix alone raised IndexError on every oracle B3
   request. The cached path must produce the SAME ids as the plain
   full-prompt path, and the cache must be cropped back after the query.
2. the request stop list is honored through a StoppingCriteria: generation
   ends at the first stop string, the text is cut before it, finish_reason
   reads "stop", and the id count includes the token that completed the stop.

The words carry no substring relation to one another, so a stop string can
never match inside another word.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from src.inference.engine import InferenceRequest  # noqa: E402
from src.inference.hf_oracle_adapter import HFOracleAdapter  # noqa: E402

WORDS = [
    "[PAD]", "[UNK]", "[EOS]",
    "apple", "bread", "cloud", "drum", "eagle", "frost", "grape",
    "house", "iron", "jade", "kite", "lemon", "mango",
]
PREFIX = "apple bread cloud drum eagle"
SUFFIX = " frost grape house"


def _save_tiny_model(root: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    raw = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    torch.manual_seed(1234)
    config = Qwen2Config(
        vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(config)
    target = root / "tiny-qwen2"
    model.save_pretrained(target)
    tok.save_pretrained(target)
    return target


@pytest.fixture()
def oracle(tmp_path: Path) -> HFOracleAdapter:
    return HFOracleAdapter(str(_save_tiny_model(tmp_path)), device="cpu", dtype="float32")


def _ids(oracle: HFOracleAdapter, text: str) -> list[int]:
    return oracle.tokenizer(text, add_special_tokens=False)["input_ids"]


def test_cached_corpus_path_matches_the_plain_full_prompt_path(oracle: HFOracleAdapter) -> None:
    plain = oracle.generate(
        InferenceRequest(prompt=PREFIX + SUFFIX, temperature=0.0, max_tokens=6, request_id="plain")
    )
    assert plain.error is None, plain.error
    assert plain.num_tokens >= 1

    prefill_ms = oracle.preload_corpus_prefix(PREFIX)
    assert prefill_ms > 0.0
    base_len = len(_ids(oracle, PREFIX))
    assert oracle._corpus_base_len == base_len == 5

    cached = oracle.generate(
        InferenceRequest(prompt=PREFIX + SUFFIX, temperature=0.0, max_tokens=6, request_id="cached")
    )
    assert cached.error is None, cached.error
    # the suffix-only call raised IndexError on the landing; the full-ids call
    # serves the same greedy continuation as the plain path
    assert cached.generated_text == plain.generated_text
    assert cached.num_tokens == plain.num_tokens
    assert cached.prompt_tokens == len(_ids(oracle, SUFFIX)) == 3
    assert cached.cached_prompt_tokens == base_len
    assert cached.corpus_prefill_ms == pytest.approx(prefill_ms)
    # cropped back to the corpus length after the query (the Chan et al. recipe)
    assert oracle._corpus_cache.get_seq_length() == base_len
    # and a second query against the same cache serves the same text again
    again = oracle.generate(
        InferenceRequest(prompt=PREFIX + SUFFIX, temperature=0.0, max_tokens=6, request_id="again")
    )
    assert again.generated_text == plain.generated_text
    assert oracle._corpus_cache.get_seq_length() == base_len


def test_stop_list_ends_generation_and_cuts_the_text(oracle: HFOracleAdapter) -> None:
    oracle.preload_corpus_prefix(PREFIX)
    free = oracle.generate(
        InferenceRequest(prompt=PREFIX + SUFFIX, temperature=0.0, max_tokens=8, request_id="free")
    )
    assert free.error is None, free.error
    words = free.generated_text.split()
    assert len(words) >= 2, free.generated_text
    # the stop string is a word the free run produced; cut at its first position
    stop_word = words[-1]
    k = words.index(stop_word)

    stopped = oracle.generate(
        InferenceRequest(
            prompt=PREFIX + SUFFIX, temperature=0.0, max_tokens=8, stop=[stop_word], request_id="stopped"
        )
    )
    assert stopped.error is None, stopped.error
    assert stopped.generated_text.split() == words[:k]
    assert stopped.finish_reason == "stop"
    # the id count includes the token that completed the stop string (ADR-0118)
    assert stopped.num_tokens == k + 1
    assert oracle._corpus_cache.get_seq_length() == oracle._corpus_base_len

    # a stop string the model never produces leaves the free run untouched
    untouched = oracle.generate(
        InferenceRequest(
            prompt=PREFIX + SUFFIX, temperature=0.0, max_tokens=8, stop=["[UNK]"], request_id="untouched"
        )
    )
    assert untouched.generated_text == free.generated_text
    assert untouched.num_tokens == free.num_tokens


def test_plain_path_honors_the_campaign_newline_stop_without_a_corpus(oracle: HFOracleAdapter) -> None:
    # the campaign sends stop=["\\n"] on every backend (run_experiment
    # _request_stop); a word-level tokenizer never emits a newline, so the
    # request runs to its bound and the oracle labels it "length"
    resp = oracle.generate(
        InferenceRequest(prompt=PREFIX, temperature=0.0, max_tokens=3, stop=["\n"], request_id="nl")
    )
    assert resp.error is None, resp.error
    assert resp.num_tokens == 3 and resp.finish_reason == "length"
    assert resp.cached_prompt_tokens == 0
