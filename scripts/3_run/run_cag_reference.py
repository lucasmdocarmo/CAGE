#!/usr/bin/env python3
"""True-CAG reference-engine runner (HF transformers, greedy decoding).

Implements Cache-Augmented Generation exactly as in Chan et al. 2024
(arXiv 2412.15605, "Don't Do RAG"; reference impl github.com/hhhuang/CAG):
precompute the KV cache of ONE fixed corpus block with a single forward pass,
answer every query against that cache, and crop the cache back to the corpus
length after EVERY query so query B never attends to query A's tokens.

Role in CAGE: this is the REFERENCE engine for the idea-gain / engine-gain
decomposition. Every row carries engine="hf_reference" and its serving numbers
(corpus_prefill_ms, query_latency_ms) characterize plain HF transformers --
no PagedAttention, no continuous batching, no CUDA graphs. They are NOT
comparable to the vLLM serving arms and must never appear in vLLM latency
tables; compare them only against other hf_reference rows.

Corpus assembly is delegated to the shared contract builder
(src.data.corpus.build_corpus_block) so this runner and the vLLM cag_true
cells serve a byte-identical corpus block. The prompt layout mirrors the
repo's raw-completion vLLM path (src.utils.prompting.format_qa_prompt):

    DEFAULT_SYSTEM_PREFIX.rstrip() + "\\n" + <corpus block>      <- built ONCE,
                                                                    KV-cached
    "\\n\\nQuestion: {q}\\nAnswer:"                              <- per query

i.e. the corpus block sits exactly where the per-query context blocks go in
the vLLM arms, and each query appends after the cached prefix.

Prompt mode (Decision 1B, 2026-07-16): by DEFAULT the runner now serves via
the tokenizer's CHAT TEMPLATE (tokenizer.apply_chat_template with
enable_thinking=False -- Qwen3 thinking off), rendering the SAME message
layout as the vLLM chat arms (src.utils.prompting.format_qa_messages: system
= task + abstention instruction; user = corpus block first, question last).
The cacheable prefix is the rendered template up to the corpus/question
boundary; each query appends its templated suffix after the cached KV.
CAGE_PROMPT_MODE=raw is the escape hatch that reproduces the legacy raw
layout above byte-for-byte.

Uniform query manifest (--query-manifest / CAGE_QUERY_MANIFEST): when set, the
measured query set comes from ONE pre-drawn, auditable artifact
(src.data.manifest) shared by every cell/engine/model -- the fairness
yardstick. The runner then serves the manifest's corpus block TEXTS verbatim
(no self-built block), iterates blocks in block_id order so each block's KV
stays resident while its questions run (same ordering run_experiment.py uses
in corpus mode, keeping the reference/vLLM pairing clean), prefills one
DynamicCache per block (that block's corpus_prefill_ms), and crops after every
query as before. Without a manifest the original single self-built-block
behavior is unchanged.

Outputs (per trial dir trial_N/ under --output-dir):
- results.csv         one row per measured query (schema in RESULT_COLUMNS)
- qa_evidence.jsonl   appended incrementally; consumed by
                      scripts/4_analysis/rescore_quality.py for offline quality
                      scoring (grounding_score=None here -- scored later)
- corpus_blocks.json  per-block provenance (served KV length, prefill ms; the
                      self-built path also embeds the block text -- manifest
                      block texts live in the manifest artifact itself)

torch / transformers / datasets import lazily inside main() so the module's
pure-logic seams stay unit-testable in the lean analysis venv; if the ML deps
are missing the runner exits 2 with a clear message.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.corpus import CorpusBlock, build_corpus_block  # noqa: E402  (dependency-free)
from src.data.manifest import ManifestError, select_examples  # noqa: E402  (pure stdlib)
from src.utils.prompting import (  # noqa: E402  (pure)
    DEFAULT_SYSTEM_PREFIX,
    format_qa_messages,
    prompt_mode,
)

BASELINE = "cag_reference"
ENGINE = "hf_reference"

# results.csv schema -- fixed order, one row per measured query.
RESULT_COLUMNS = [
    "example_id",
    "baseline",
    "engine",
    "trial",
    "question",
    "reference_answer",
    "generated_answer",
    "corpus_prefill_ms",   # prefill of the row's OWN block (one prefill per block per trial)
    "query_latency_ms",    # the CAG "TTFT-equivalent": whole generate() call, no streaming
    "num_tokens",          # generated tokens
    "prompt_tokens",       # query-only tokens appended after the cached corpus
    "corpus_tokens",       # manifest mode: the block's manifest token_count (the shared
                           # yardstick number); self-built mode: base_len (served KV length)
    "corpus_block",        # block id (0 for the single self-built block)
    "error",
    "empty_generation",
]

QUERY_TEMPLATE = "\n\nQuestion: {question}\nAnswer:"


# --------------------------------------------------------------------------
# Pure-logic seams (unit-testable without torch)
# --------------------------------------------------------------------------

def build_query_suffix(question: str) -> str:
    """Per-query text appended after the cached corpus prefix.

    Tokenized with add_special_tokens=False so the concatenation
    corpus_prompt + suffix tokenizes exactly as one served prompt would.
    """
    return QUERY_TEMPLATE.format(question=question)


def build_corpus_prompt(block_text: str) -> str:
    """The shared cacheable prefix: system prefix + corpus block.

    Mirrors format_qa_prompt's layout (prefix.rstrip() + "\\n" + <context>),
    with the corpus block in the per-query-context slot, so answers are
    comparable with the vLLM raw-completion arms.
    """
    return DEFAULT_SYSTEM_PREFIX.rstrip() + "\n" + block_text


# --------------------------------------------------------------------------
# Chat-template seams (Decision 1B) -- pure logic, torch-free
# --------------------------------------------------------------------------

def chat_messages_for_block(block_text: str, question: str) -> List[Dict[str, str]]:
    """Chat messages for one corpus-block query.

    Delegates to src.utils.prompting.format_qa_messages with the corpus block
    as the single context doc, so the reference engine renders EXACTLY the
    same message content the vLLM chat arms serve (system = task + abstention
    instruction; user = "Context 1: <block>\\n\\nQuestion: <q>").
    """
    return format_qa_messages(question, [block_text])


def compute_chat_prefix(render_full) -> str:
    """The cacheable shared prefix of the chat-templated corpus prompt.

    ``render_full(question) -> str`` renders the FULL templated prompt for a
    question (tokenizer.apply_chat_template, add_generation_prompt=True).
    Probing with two sentinel questions and trimming the common prefix back to
    just before the "\\n\\nQuestion:" marker yields everything the template
    renders up to and including the corpus block -- the exact same
    corpus/question boundary the raw path caches at. Template-agnostic.
    """
    common = os.path.commonprefix([render_full("A?"), render_full("B?")])
    cut = common.rfind("\n\nQuestion:")
    if cut == -1:
        raise ValueError(
            "chat template did not preserve the corpus-block/question layout; "
            "cannot derive a cacheable corpus prefix (use CAGE_PROMPT_MODE=raw)"
        )
    return common[:cut]


def chat_query_suffix(full_prompt: str, corpus_prefix: str) -> str:
    """Per-query text appended after the cached chat-mode corpus prefix.

    Includes the question, the user-turn close, and the generation header the
    template emits (for Qwen3 with enable_thinking=False that header carries
    the empty <think> block -- prompt tokens, never generated ones).
    """
    if not full_prompt.startswith(corpus_prefix):
        raise ValueError(
            "templated prompt does not share the computed corpus prefix; "
            "chat-template rendering is not prefix-stable for this block"
        )
    return full_prompt[len(corpus_prefix):]


def default_dataset_split(dataset_name: str) -> str:
    """Same split convention as scripts/3_run/run_experiment.py."""
    if dataset_name in {"humaneval", "mbpp", "hpc_code"}:
        return "test"
    return "validation"


def resolve_output_dir(cli_value: Optional[str], env: Optional[dict] = None) -> Optional[Path]:
    """--output-dir wins; else derive from CAGE_RUN_ROOT (cloud_run.sh convention,
    like run_baselines.sh does with $CAGE_RUN_ROOT/baselines); else None."""
    if cli_value:
        return Path(cli_value)
    environ = env if env is not None else os.environ
    run_root = environ.get("CAGE_RUN_ROOT")
    if run_root:
        return Path(run_root) / "reference" / "cag_reference"
    return None


def select_in_corpus_examples(examples: Sequence[Any], block: CorpusBlock) -> List[Any]:
    """Examples whose gold paragraph is in the block, preserving load order."""
    in_corpus = set(block.example_ids)
    return [ex for ex in examples if ex.id in in_corpus]


@dataclass
class BlockWork:
    """One corpus block to serve in a trial, with its assigned queries."""

    block_id: int
    text: str
    examples: List[Any]
    # Manifest mode: the manifest block's token_count, reported as corpus_tokens
    # (the engine-independent yardstick number every cell shares). None in the
    # self-built path -> rows report the served KV length (base_len) instead.
    manifest_token_count: Optional[int] = None


def resolve_manifest_path(cli_value: Optional[str], env: Optional[dict] = None) -> Optional[str]:
    """--query-manifest wins; else CAGE_QUERY_MANIFEST (run_experiment.py convention)."""
    if cli_value:
        return cli_value
    environ = env if env is not None else os.environ
    path = (environ.get("CAGE_QUERY_MANIFEST") or "").strip()
    return path or None


def validate_manifest_dataset(manifest: Dict[str, Any], dataset: str) -> None:
    """Refuse a mismatched yardstick loudly (same contract as run_experiment.py)."""
    if manifest.get("dataset") and manifest["dataset"] != dataset:
        raise ValueError(
            f"manifest is for dataset '{manifest['dataset']}' but this run uses "
            f"'{dataset}' -- refusing to serve a mismatched yardstick"
        )


def plan_self_built_block(block: CorpusBlock, examples: Sequence[Any]) -> List[BlockWork]:
    """Legacy (non-manifest) path: the single self-built block, id 0."""
    return [BlockWork(block_id=0, text=block.text,
                      examples=select_in_corpus_examples(examples, block))]


def plan_blocks(manifest: Dict[str, Any], examples: Sequence[Any]) -> List[BlockWork]:
    """Group a trial's manifest-selected examples by their assigned corpus block.

    Blocks come out in block_id order -- each block's KV stays resident while
    its questions run, the same ordering run_experiment.py applies in corpus
    mode, so the reference/vLLM pairing stays clean. Within a block, manifest
    order is preserved. Block TEXTS are taken from the manifest verbatim.
    """
    q2b = manifest["question_to_block"]
    blocks = manifest["blocks"]
    missing = [ex.id for ex in examples if ex.id not in q2b]
    if missing:
        raise ManifestError(
            f"{len(missing)} trial ids have no corpus block in this manifest "
            f"(first: {missing[:3]})"
        )
    grouped: Dict[int, List[Any]] = {}
    for ex in examples:
        grouped.setdefault(int(q2b[ex.id]), []).append(ex)
    return [
        BlockWork(
            block_id=bid,
            text=blocks[bid]["text"],
            examples=grouped[bid],
            manifest_token_count=blocks[bid].get("token_count"),
        )
        for bid in sorted(grouped)
    ]


def build_result_row(
    *,
    example: Any,
    block: BlockWork,
    trial: int,
    corpus_prefill_ms: float,
    query_latency_ms: float,
    num_tokens: int,
    prompt_tokens: int,
    base_len: int,
    answer: str,
    error: Optional[str],
) -> Dict[str, Any]:
    """Pure results.csv row assembly (unit-tested without torch).

    corpus_prefill_ms is the prefill of the row's OWN block; corpus_tokens is
    the manifest block's token_count when available (yardstick number), else
    the served KV length.
    """
    return {
        "example_id": example.id,
        "baseline": BASELINE,
        "engine": ENGINE,
        "trial": trial,
        "question": example.question,
        "reference_answer": example.answer,
        "generated_answer": answer,
        "corpus_prefill_ms": corpus_prefill_ms,
        "query_latency_ms": query_latency_ms,
        "num_tokens": num_tokens,
        "prompt_tokens": prompt_tokens,
        "corpus_tokens": (block.manifest_token_count
                          if block.manifest_token_count is not None else base_len),
        "corpus_block": block.block_id,
        "error": error,
        "empty_generation": (error is None) and not (answer or "").strip(),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "True-CAG reference-engine runner (HF transformers, greedy). "
            "Precomputes the corpus KV cache once per trial, answers every "
            "in-corpus query against it, crops the cache after each query. "
            "All rows are engine=hf_reference: NOT comparable to vLLM arms."
        ),
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model id (default: Qwen/Qwen3-8B)")
    p.add_argument("--dataset", default="squad_v2",
                   help="Dataset name for src.data.loader.get_loader (default: squad_v2)")
    p.add_argument("--num-queries", type=int, default=500,
                   help="Examples to LOAD per trial (matches NUM_QUERIES convention); only the "
                        "in-corpus subset -- whose gold paragraph fit the token budget -- is answered.")
    p.add_argument("--num-trials", type=int, default=3, help="Trials; trial t uses seed+t-1.")
    p.add_argument("--seed", type=int, default=42, help="Base seed (default: 42)")
    p.add_argument("--token-budget", type=int, default=2800,
                   help="Corpus block token budget under the REAL model tokenizer (default: 2800)")
    p.add_argument("--max-new-tokens", type=int, default=256, help="Greedy decode cap (default: 256)")
    p.add_argument("--query-manifest", default=None,
                   help="Path to the uniform query manifest (src.data.manifest JSON). All "
                        "trials then measure the manifest's pre-drawn query ids over its "
                        "corpus block texts verbatim -- the fairness yardstick shared by "
                        "every cell/engine/model. Defaults to $CAGE_QUERY_MANIFEST when set. "
                        "Manifest mode ignores --num-queries/--token-budget (the manifest "
                        "fixes both).")
    p.add_argument("--output-dir", default=None,
                   help="Run output dir. Default: $CAGE_RUN_ROOT/reference/cag_reference when "
                        "CAGE_RUN_ROOT is set (cloud_run.sh convention); otherwise required.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="auto = cuda if available else cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="Model dtype (default: bfloat16)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Runner (torch/transformers imported lazily)
# --------------------------------------------------------------------------

def _print_banner(model: str, device: str, dtype: str) -> None:
    print("=" * 74)
    print(" REFERENCE ENGINE RUN  --  baseline=cag_reference  engine=hf_reference")
    print(f"   model={model}  device={device}  dtype={dtype}")
    print("   Serving metrics (corpus_prefill_ms / query_latency_ms) characterize")
    print("   plain HF transformers: NO PagedAttention, NO continuous batching,")
    print("   NO CUDA graphs. They are NOT comparable to the vLLM serving arms.")
    print("   Use only for idea-gain vs engine-gain attribution.")
    print("=" * 74)


def run_trial(
    *,
    torch,
    tok,
    model,
    DynamicCache,
    blocks: List[BlockWork],
    trial: int,
    trial_dir: Path,
    max_new_tokens: int,
    serving_prompt_mode: str = "raw",
) -> dict:
    """One trial: per corpus block, prefill its KV once, answer the block's
    queries (crop after each), release the cache, move to the next block.

    The self-built path passes exactly one BlockWork; manifest mode passes the
    trial's blocks in block_id order.
    """
    device_type = model.device.type

    def _sync() -> None:
        if device_type == "cuda":
            torch.cuda.synchronize()

    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    trial_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = trial_dir / "qa_evidence.jsonl"
    rows: List[dict] = []
    block_records: List[dict] = []

    for block in blocks:
        if serving_prompt_mode == "chat":
            # Decision 1B: SAME chat-template layout as the vLLM chat arms, rendered
            # through tokenizer.apply_chat_template with enable_thinking=False (the
            # template kwarg Qwen3 reads; other templates ignore the unused variable).
            # The cacheable prefix is everything up to the corpus/question boundary.
            def _render_full(question: str, _text: str = block.text) -> str:
                return tok.apply_chat_template(
                    chat_messages_for_block(_text, question),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

            corpus_prompt = compute_chat_prefix(_render_full)
            # The rendered template already CONTAINS the special tokens as text
            # (<|im_start|> etc.), so nothing extra may be prepended.
            enc = tok(corpus_prompt, return_tensors="pt",
                      add_special_tokens=False).to(model.device)
        else:
            corpus_prompt = build_corpus_prompt(block.text)
            # Corpus tokenized exactly as served: default special-token handling for
            # the sequence start; query suffixes are appended with add_special_tokens=False.
            enc = tok(corpus_prompt, return_tensors="pt").to(model.device)

        cache = DynamicCache()
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**enc, past_key_values=cache, use_cache=True)
        _sync()
        corpus_prefill_ms = (time.perf_counter() - t0) * 1000.0
        base_len = cache.get_seq_length()

        print(f"[trial {trial}] block {block.block_id}: KV cached {base_len} tokens "
              f"(prefill {corpus_prefill_ms:.1f} ms), {len(block.examples)} queries")

        for example in block.examples:
            if serving_prompt_mode == "chat":
                suffix_text = chat_query_suffix(_render_full(example.question), corpus_prompt)
            else:
                suffix_text = build_query_suffix(example.question)
            q_enc = tok(suffix_text, return_tensors="pt",
                        add_special_tokens=False).to(model.device)
            q_len = int(q_enc.input_ids.shape[1])
            # Attention mask must cover cached corpus + new query tokens.
            attention_mask = torch.ones((1, base_len + q_len), dtype=torch.long,
                                        device=model.device)

            answer = ""
            num_generated = 0
            error: Optional[str] = None
            _sync()
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    out = model.generate(
                        input_ids=q_enc.input_ids,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                    )
                answer = tok.decode(out[0, q_len:], skip_special_tokens=True)
                num_generated = int(out.shape[1]) - q_len
            except Exception as exc:  # noqa: BLE001 -- record, crop, continue
                error = f"{type(exc).__name__}: {exc}"
            finally:
                _sync()
                query_latency_ms = (time.perf_counter() - t0) * 1000.0
                # NON-OPTIONAL (Chan et al. recipe): crop back to the corpus
                # length after EVERY query -- else the next query attends to
                # this one's question AND generated answer, silently corrupting
                # every subsequent row of the trial.
                cache.crop(base_len)

            rows.append(build_result_row(
                example=example,
                block=block,
                trial=trial,
                corpus_prefill_ms=corpus_prefill_ms,
                query_latency_ms=query_latency_ms,
                num_tokens=num_generated,
                prompt_tokens=q_len,
                base_len=base_len,
                answer=answer,
                error=error,
            ))

            # Evidence appended INCREMENTALLY (one JSON line per query, same
            # convention as run_experiment.py) so a mid-trial crash preserves
            # completed rows for the offline re-scorer. used_contexts carries
            # the FULL corpus block this row was conditioned on;
            # grounding_score stays None -- rescore_quality.py --full scores it.
            evidence = {
                "example_id": example.id,
                "baseline": BASELINE,
                "engine": ENGINE,
                "trial": trial,
                "corpus_block": block.block_id,
                # group_id mirrors run_experiment.py's qa_evidence contract field
                # (the manifest block id); prompt_mode records Decision 1B provenance.
                "group_id": block.block_id,
                "prompt_mode": serving_prompt_mode,
                "question": example.question,
                "reference_answer": example.answer,
                "generated_answer": answer,
                "used_contexts": [block.text],
                "grounding_score": None,
            }
            with evidence_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(evidence, default=str) + "\n")

        record = {
            "block_id": block.block_id,
            "n_queries": len(block.examples),
            "reported_corpus_tokens": (block.manifest_token_count
                                       if block.manifest_token_count is not None
                                       else base_len),
            "served_kv_tokens": base_len,   # system prefix + block + special tokens
            "corpus_prefill_ms": corpus_prefill_ms,
        }
        if block.manifest_token_count is None:
            # Self-built block: its text exists nowhere else, so persist it here.
            # Manifest block texts live in the manifest artifact itself.
            record["text"] = block.text
        block_records.append(record)

        # Release this block's cache before prefill of the next one: true CAG
        # serves one resident corpus at a time; sequential blocks emulate that
        # within GPU capacity.
        del cache, enc
        if device_type == "cuda":
            torch.cuda.empty_cache()

    results_path = trial_dir / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    (trial_dir / "corpus_blocks.json").write_text(
        json.dumps({"trial": trial, "blocks": block_records}, indent=2),
        encoding="utf-8",
    )

    n_err = sum(1 for r in rows if r["error"])
    n_empty = sum(1 for r in rows if r["empty_generation"])
    lat = [r["query_latency_ms"] for r in rows if r["error"] is None]
    mean_lat = (sum(lat) / len(lat)) if lat else None
    summary = (f"[trial {trial}] done: {len(rows)} queries over {len(blocks)} blocks, "
               f"{n_err} errors, {n_empty} empty")
    if mean_lat is not None:
        summary += f", mean query_latency_ms={mean_lat:.1f}"
    print(summary)
    return {
        "trial": trial,
        "num_queries": len(rows),
        "n_blocks": len(blocks),
        "errors": n_err,
        "empty_generations": n_empty,
        "total_corpus_prefill_ms": sum(r["corpus_prefill_ms"] for r in block_records),
        "mean_query_latency_ms": mean_lat,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    output_dir = resolve_output_dir(args.output_dir)
    if output_dir is None:
        print("ERROR: --output-dir is required (or set CAGE_RUN_ROOT; outputs then go to "
              "$CAGE_RUN_ROOT/reference/cag_reference).", file=sys.stderr)
        return 2

    # Uniform-yardstick manifest: resolved and validated BEFORE the ML imports so
    # a mismatched yardstick fails fast even on a torch-free box.
    manifest_path = resolve_manifest_path(args.query_manifest)
    manifest = None
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        validate_manifest_dataset(manifest, args.dataset)

    # Graceful failure on the lean analysis venv: the ML stack is imported
    # lazily so the pure seams above stay importable without it.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    except ImportError as exc:
        print(f"ERROR: reference engine needs torch + transformers (missing: {exc}).\n"
              "Run on the GPU VM env (cage-env) or pip install torch transformers.",
              file=sys.stderr)
        return 2
    try:
        from src.data.loader import get_loader
    except ImportError as exc:
        print(f"ERROR: dataset loader needs the 'datasets' package (missing: {exc}).",
              file=sys.stderr)
        return 2

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    _print_banner(args.model, device, args.dtype)

    print(f"Loading tokenizer + model: {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # Decision 1B: chat-template serving by default, matching the vLLM chat arms
    # (system = task + abstention instruction, user = corpus block + question,
    # enable_thinking=False). CAGE_PROMPT_MODE=raw reproduces the legacy raw layout.
    serving_prompt_mode = prompt_mode()
    if serving_prompt_mode == "chat":
        if not getattr(tok, "chat_template", None):
            print(f"ERROR: CAGE_PROMPT_MODE=chat but {args.model} ships no chat "
                  "template. Re-run with CAGE_PROMPT_MODE=raw.", file=sys.stderr)
            return 2
        print("PROMPT MODE: chat -- tokenizer.apply_chat_template(enable_thinking="
              "False); the cacheable prefix is the rendered template up to the "
              "corpus/question boundary, every query appends its templated suffix "
              "after the cached KV. Same message layout as the vLLM chat arms.")
    else:
        print("PROMPT MODE: raw (escape hatch) -- chat template deliberately NOT "
              "used; legacy layout: the corpus prompt (system prefix + corpus block) "
              "is built EXACTLY once and KV-cached; every query appends "
              "'\\n\\nQuestion: ...\\nAnswer:' after it.")

    # Budget under the REAL served tokenizer: corpus tokenized exactly as served.
    def count_tokens(text: str) -> int:
        return len(tok(text, add_special_tokens=False).input_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    trial_summaries = []

    pool = None
    if manifest is not None:
        stats = manifest.get("stats", {})
        print(f"MANIFEST yardstick: {manifest_path} (dataset={manifest.get('dataset')}, "
              f"N={manifest.get('num_queries')}, T={manifest.get('num_trials')}, "
              f"pool={stats.get('pool_size')}, blocks={stats.get('n_blocks')}); "
              f"--num-queries/--token-budget ignored (fixed by the manifest).")
        # Full split, loaded ONCE: manifest selection is by id, so no per-trial
        # sampling and no dependence on the loader seed (no shuffle without
        # max_examples). Same convention as run_experiment.py's manifest path.
        loader = get_loader(args.dataset, split=default_dataset_split(args.dataset),
                            seed=args.seed)
        pool = loader.load(max_examples=None)

    for trial in range(1, args.num_trials + 1):
        trial_seed = args.seed + trial - 1  # run_experiment.py convention
        print(f"\n--- Trial {trial}/{args.num_trials} (seed={trial_seed}) ---")

        if manifest is not None:
            trial_examples = select_examples(manifest, trial, pool)
            blocks = plan_blocks(manifest, trial_examples)
            print(f"[trial {trial}] MANIFEST workload: {len(trial_examples)} queries "
                  f"over {len(blocks)} blocks, block-ordered")
        else:
            loader = get_loader(args.dataset, split=default_dataset_split(args.dataset),
                                seed=trial_seed)
            examples = loader.load(max_examples=args.num_queries)
            block = build_corpus_block(examples, args.token_budget, count_tokens=count_tokens)
            if not block.example_ids:
                print(f"[trial {trial}] WARNING: no example fit token_budget="
                      f"{args.token_budget}; skipping trial.")
                continue
            blocks = plan_self_built_block(block, examples)

        trial_summaries.append(run_trial(
            torch=torch,
            tok=tok,
            model=model,
            DynamicCache=DynamicCache,
            blocks=blocks,
            trial=trial,
            trial_dir=output_dir / f"trial_{trial}",
            max_new_tokens=args.max_new_tokens,
            serving_prompt_mode=serving_prompt_mode,
        ))

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "baseline": BASELINE,
                "engine": ENGINE,
                "model": args.model,
                "dataset": args.dataset,
                "dataset_split": default_dataset_split(args.dataset),
                "num_queries_loaded": args.num_queries,
                "num_trials": args.num_trials,
                "seed": args.seed,
                "token_budget": args.token_budget,
                "max_new_tokens": args.max_new_tokens,
                "device": device,
                "dtype": args.dtype,
                "prompt_mode": serving_prompt_mode,
                "query_manifest": manifest_path,
                "manifest_stats": (manifest or {}).get("stats"),
                "trials": trial_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nAll trials complete. Outputs under: {output_dir}")
    print("Reminder: engine=hf_reference rows are reference-engine numbers, "
          "NOT comparable to vLLM arms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
