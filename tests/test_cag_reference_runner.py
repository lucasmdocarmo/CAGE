"""Pure-logic seam tests for scripts/3_run/run_cag_reference.py.

Torch-free by design: the runner imports torch/transformers/datasets lazily
inside main(), so its module and prompt/output-dir seams must stay importable
and testable in the lean analysis venv.
"""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "3_run" / "run_cag_reference.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_cag_reference", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: the dataclass decorator resolves cls.__module__ via
    # sys.modules while processing the runner's BlockWork dataclass.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


@dataclass
class FakeExample:
    id: str
    question: str
    context: List[str]
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def test_module_imports_without_torch():
    """Lazy-import contract: loading the module must not pull the ML stack.

    The module-level _load_runner() above already executed the runner in this
    torch-free venv; here we additionally assert nothing heavy leaked into the
    module namespace (i.e. the imports really are inside main()/functions).
    """
    for heavy in ("torch", "transformers", "datasets", "get_loader",
                  "AutoModelForCausalLM", "AutoTokenizer", "DynamicCache"):
        assert heavy not in vars(runner), f"{heavy} imported at module level"


def test_query_suffix_matches_cag_recipe():
    assert runner.build_query_suffix("Who?") == "\n\nQuestion: Who?\nAnswer:"


def test_corpus_prompt_layout_matches_vllm_raw_completion_path():
    from src.utils.prompting import DEFAULT_SYSTEM_PREFIX, format_qa_prompt

    block_text = "You are given the following reference documents:\n\nDocument 1:\npara"
    prompt = runner.build_corpus_prompt(block_text)
    assert prompt == DEFAULT_SYSTEM_PREFIX.rstrip() + "\n" + block_text
    # Full served prompt has the same slot layout as format_qa_prompt, with the
    # corpus block in the per-query-context slot.
    full = prompt + runner.build_query_suffix("Who?")
    expected = format_qa_prompt("Who?", ["CTX"]).replace("Context 1: CTX", block_text)
    assert full == expected


def test_result_columns_schema():
    assert runner.RESULT_COLUMNS == [
        "example_id", "baseline", "engine", "trial", "question",
        "reference_answer", "generated_answer", "corpus_prefill_ms",
        "query_latency_ms", "num_tokens", "prompt_tokens", "corpus_tokens",
        "corpus_block", "error", "empty_generation",
    ]
    assert runner.BASELINE == "cag_reference"
    assert runner.ENGINE == "hf_reference"


def test_resolve_output_dir_cli_wins():
    out = runner.resolve_output_dir("results/x", env={"CAGE_RUN_ROOT": "/run/root"})
    assert out == Path("results/x")


def test_resolve_output_dir_env_fallback():
    out = runner.resolve_output_dir(None, env={"CAGE_RUN_ROOT": "/run/root"})
    assert out == Path("/run/root/reference/cag_reference")


def test_resolve_output_dir_neither_returns_none():
    assert runner.resolve_output_dir(None, env={}) is None


def test_select_in_corpus_examples_filters_and_preserves_order():
    from src.data.corpus import build_corpus_block

    examples = [
        FakeExample("e1", "q1", ["para one " * 5], "a1"),
        FakeExample("e2", "q2", ["para two " * 50], "a2"),  # too big to fit
        FakeExample("e3", "q3", ["para one " * 5], "a3"),   # dupe of e1's paragraph
    ]
    wc = lambda t: len(t.split())  # noqa: E731
    budget = build_corpus_block(examples[:1], 10**9, count_tokens=wc).token_count
    block = build_corpus_block(examples, budget, count_tokens=wc)
    selected = runner.select_in_corpus_examples(examples, block)
    assert [ex.id for ex in selected] == ["e1", "e3"]


def test_default_dataset_split_convention():
    assert runner.default_dataset_split("squad_v2") == "validation"
    assert runner.default_dataset_split("humaneval") == "test"


def test_parse_args_defaults():
    args = runner.parse_args([])
    assert args.model == "Qwen/Qwen3-8B"
    assert args.dataset == "squad_v2"
    assert args.seed == 42
    assert args.token_budget == 2800
    assert args.max_new_tokens == 256
    assert args.device == "auto"
    assert args.dtype == "bfloat16"
    assert args.output_dir is None
    assert args.query_manifest is None


# --------------------------------------------------------------------------
# Manifest-path pure seams (uniform yardstick)
# --------------------------------------------------------------------------

def _manifest_fixture():
    return {
        "manifest_version": 1,
        "dataset": "squad_v2",
        "split": "validation",
        "seed": 42,
        "num_queries": 3,
        "num_trials": 2,
        "block_budget": 2800,
        "blocks": [
            {"block_id": 0, "text": "HDR\n\nDocument 1:\npara-a", "token_count": 11,
             "n_paragraphs": 1},
            {"block_id": 1, "text": "HDR\n\nDocument 1:\npara-b", "token_count": 22,
             "n_paragraphs": 1},
        ],
        "question_to_block": {"e1": 0, "e2": 1, "e3": 1},
        "trials": {"1": ["e2", "e1", "e3"], "2": ["e3", "e1", "e2"]},
        "stats": {"pool_size": 3, "n_blocks": 2},
    }


def _fx(ex_id: str) -> FakeExample:
    return FakeExample(id=ex_id, question=f"q-{ex_id}", context=["p"], answer=f"a-{ex_id}")


def test_resolve_manifest_path_precedence():
    assert runner.resolve_manifest_path("m.json", env={"CAGE_QUERY_MANIFEST": "env.json"}) == "m.json"
    assert runner.resolve_manifest_path(None, env={"CAGE_QUERY_MANIFEST": "env.json"}) == "env.json"
    assert runner.resolve_manifest_path(None, env={"CAGE_QUERY_MANIFEST": "  "}) is None
    assert runner.resolve_manifest_path(None, env={}) is None


def test_validate_manifest_dataset_mismatch_refused():
    import pytest

    manifest = _manifest_fixture()
    with pytest.raises(ValueError, match="mismatched yardstick"):
        runner.validate_manifest_dataset(manifest, "hotpotqa")
    # Matching dataset and legacy manifests without the field both pass.
    runner.validate_manifest_dataset(manifest, "squad_v2")
    runner.validate_manifest_dataset({"dataset": ""}, "hotpotqa")


def test_plan_blocks_groups_in_block_id_order():
    manifest = _manifest_fixture()
    # Trial 1 manifest order interleaves blocks: e2 (b1), e1 (b0), e3 (b1).
    trial_examples = [_fx("e2"), _fx("e1"), _fx("e3")]
    blocks = runner.plan_blocks(manifest, trial_examples)
    assert [b.block_id for b in blocks] == [0, 1]
    assert [ex.id for ex in blocks[0].examples] == ["e1"]
    assert [ex.id for ex in blocks[1].examples] == ["e2", "e3"]  # manifest order kept
    # Block TEXT verbatim from the manifest, token_count carried for corpus_tokens.
    assert blocks[0].text == "HDR\n\nDocument 1:\npara-a"
    assert blocks[1].text == "HDR\n\nDocument 1:\npara-b"
    assert blocks[0].manifest_token_count == 11
    assert blocks[1].manifest_token_count == 22


def test_plan_blocks_missing_id_raises():
    import pytest
    from src.data.manifest import ManifestError

    manifest = _manifest_fixture()
    with pytest.raises(ManifestError, match="no corpus block"):
        runner.plan_blocks(manifest, [_fx("e1"), _fx("ghost")])


def test_build_result_row_per_block_prefill_accounting():
    b0 = runner.BlockWork(block_id=0, text="T0", examples=[], manifest_token_count=11)
    b1 = runner.BlockWork(block_id=1, text="T1", examples=[], manifest_token_count=22)
    common = dict(trial=1, query_latency_ms=5.0, num_tokens=3, prompt_tokens=9,
                  base_len=999, answer="x", error=None)
    r0 = runner.build_result_row(example=_fx("e1"), block=b0, corpus_prefill_ms=100.0, **common)
    r1 = runner.build_result_row(example=_fx("e2"), block=b1, corpus_prefill_ms=200.0, **common)
    # Each row carries its OWN block's prefill, block id, and manifest token_count.
    assert (r0["corpus_prefill_ms"], r0["corpus_block"], r0["corpus_tokens"]) == (100.0, 0, 11)
    assert (r1["corpus_prefill_ms"], r1["corpus_block"], r1["corpus_tokens"]) == (200.0, 1, 22)
    assert list(r0.keys()) == runner.RESULT_COLUMNS


def test_build_result_row_self_built_falls_back_to_served_kv():
    blk = runner.BlockWork(block_id=0, text="T", examples=[])  # manifest_token_count=None
    row = runner.build_result_row(
        example=_fx("e1"), block=blk, trial=1, corpus_prefill_ms=1.0,
        query_latency_ms=2.0, num_tokens=0, prompt_tokens=9, base_len=777,
        answer="", error=None,
    )
    assert row["corpus_tokens"] == 777
    assert row["corpus_block"] == 0
    assert row["empty_generation"] is True


def test_plan_blocks_roundtrip_with_built_manifest():
    from src.data.manifest import build_manifest, select_examples

    examples = [
        FakeExample(id=f"e{i}", question=f"q{i}", context=[f"para {i // 2} " * 10],
                    answer=f"a{i}")
        for i in range(8)  # 4 distinct paragraphs, 2 questions each
    ]
    manifest = build_manifest(examples, num_queries=2, num_trials=2, seed=7,
                              block_budget=2800, dataset="squad_v2")
    for trial in (1, 2):
        selected = select_examples(manifest, trial, examples)
        blocks = runner.plan_blocks(manifest, selected)
        assert [b.block_id for b in blocks] == sorted(b.block_id for b in blocks)
        grouped_ids = [ex.id for b in blocks for ex in b.examples]
        assert sorted(grouped_ids) == sorted(ex.id for ex in selected)
        for b in blocks:
            assert b.text == manifest["blocks"][b.block_id]["text"]
            assert b.manifest_token_count == manifest["blocks"][b.block_id]["token_count"]
