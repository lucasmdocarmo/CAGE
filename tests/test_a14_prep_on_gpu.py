"""A14 / A9 / A8 (owner rulings 2026-10-20, ADR-0174 amendment): the
preparation helpers move to the pod GPU and leave it before the window.

- A14: the compressor loads in float32 off the CPU at its pinned revision,
  keeps one memo file per distinct input, and releases its weights; the dense
  encoder and the reranker release theirs too; each document set gets its own
  index folder, stamped with the device it was embedded on.
- A9: every row carries its preparation span and retrieval and rerank times;
  metrics.json records the index served (built or loaded).
- A8: both truncation configurations keep their untransformed documents, so
  served-versus-source scoring covers truncation as it covers compression.
- The S0 reference fixture and the device probe's pass rules.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestration.compression import (  # noqa: E402
    COMPRESSION_CACHE_ENV,
    COMPRESSOR_REVISION_ENV,
    ContextCompressor,
)
from src.orchestration.ir import (  # noqa: E402
    STALE_INDEX_OPT_IN_ENV,
    CrossEncoderReranker,
    FaissIRIndex,
    IRDocument,
    IRHit,
    corpus_doc_ids_sha1,
    default_index_dir,
    ensure_ir_index,
)
from test_ir import _DOCS, _E5_MODEL, _FakeCrossEncoder, _write_index_dir, stub_ir_deps  # noqa: E402,F401

_REV = "ebaba9b0e874dadd3003ffcff828e4397e568089"
_DOCS_TEXT = [
    "The quick brown fox jumps over the lazy dog near the river bank today.",
    "A second passage with enough words to keep half of them in the prompt.",
]


# --------------------------------------------------------------------------- #
# A14: the compressor
# --------------------------------------------------------------------------- #


class _FakePromptCompressor:
    """llmlingua.PromptCompressor stand-in: records each load and each call."""

    inits: List[Dict[str, Any]] = []
    calls = 0

    def __init__(self, model_name=None, use_llmlingua2=True, device_map="cpu", **kwargs):
        _FakePromptCompressor.inits.append({"device_map": device_map, **kwargs})
        self.tokenizer = lambda text, add_special_tokens=False, verbose=False: {
            "input_ids": [0] * len(text.split())
        }

    def compress_prompt(self, context, **kwargs):
        _FakePromptCompressor.calls += 1
        words = " ".join(context).split()
        return {"compressed_prompt": " ".join(words[: max(1, int(len(words) * kwargs["rate"]))])}


@pytest.fixture()
def lingua(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    mod = types.ModuleType("llmlingua")
    mod.PromptCompressor = _FakePromptCompressor
    monkeypatch.setitem(sys.modules, "llmlingua", mod)
    for var in ("CAGE_DISABLE_COMPRESSION", "CAGE_ALLOW_NO_COMPRESSION", "CAGE_COMPRESSION_RATE",
                COMPRESSION_CACHE_ENV, COMPRESSOR_REVISION_ENV):
        monkeypatch.delenv(var, raising=False)
    _FakePromptCompressor.inits = []
    _FakePromptCompressor.calls = 0
    return mod


def test_a_memo_hit_serves_the_stored_text_without_the_model_a14(lingua, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(COMPRESSION_CACHE_ENV, str(tmp_path / "memo"))
    first, s1 = ContextCompressor().compress(_DOCS_TEXT, target_ratio=0.5)
    assert s1.cache_hit is False and _FakePromptCompressor.calls == 1
    assert len(list((tmp_path / "memo").glob("*.json"))) == 1
    # a later window builds its own compressor: the memo serves it, no load, no call
    second, s2 = ContextCompressor().compress(_DOCS_TEXT, target_ratio=0.5)
    assert second == first and s2.cache_hit is True
    assert _FakePromptCompressor.calls == 1 and len(_FakePromptCompressor.inits) == 1
    assert s2.compression_latency_ms == s1.compression_latency_ms  # the original computation's time
    assert (s2.original_tokens, s2.compressed_tokens, s2.applied) == (s1.original_tokens, s1.compressed_tokens, True)
    assert s2.to_dict()["compression_cache_hit"] is True


def test_the_memo_key_tracks_every_input_of_the_text_a14(lingua, monkeypatch) -> None:
    c = ContextCompressor()
    base = c._memo_key(_DOCS_TEXT, "q", 0.5)
    assert ContextCompressor()._memo_key(list(_DOCS_TEXT), "q", 0.5) == base
    assert c._memo_key(_DOCS_TEXT, "another question", 0.5) == base  # LLMLingua-2 is question-blind
    assert c._memo_key(_DOCS_TEXT, "q", 0.4) != base
    assert c._memo_key(_DOCS_TEXT[:1], "q", 0.5) != base
    assert ContextCompressor(device="cuda")._memo_key(_DOCS_TEXT, "q", 0.5) != base
    monkeypatch.setenv(COMPRESSOR_REVISION_ENV, _REV)
    assert ContextCompressor()._memo_key(_DOCS_TEXT, "q", 0.5) != base
    v1 = ContextCompressor(method="llmlingua")
    assert v1._memo_key(_DOCS_TEXT, "q", 0.5) != v1._memo_key(_DOCS_TEXT, "other", 0.5)  # v1 reads it


def test_without_the_env_nothing_is_stored_and_the_flag_is_absent_a14(lingua, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    _, stats = ContextCompressor().compress(_DOCS_TEXT, target_ratio=0.5)
    assert stats.cache_hit is None and stats.to_dict()["compression_cache_hit"] is None
    assert list(tmp_path.iterdir()) == []


def test_a_corrupt_or_foreign_memo_file_is_recomputed_and_rewritten_a14(lingua, monkeypatch, tmp_path, capsys) -> None:
    memo = tmp_path / "memo"
    monkeypatch.setenv(COMPRESSION_CACHE_ENV, str(memo))
    docs, _ = ContextCompressor().compress(_DOCS_TEXT, target_ratio=0.5)
    (path,) = memo.glob("*.json")
    for bad in ("{not json", json.dumps({"key": "0" * 64, "compressed_docs": ["other"]})):
        path.write_text(bad, encoding="utf-8")
        again, stats = ContextCompressor().compress(_DOCS_TEXT, target_ratio=0.5)
        assert again == docs and stats.cache_hit is False
        assert "unreadable" in capsys.readouterr().out
        assert json.loads(path.read_text(encoding="utf-8"))["compressed_docs"] == docs
    assert _FakePromptCompressor.calls == 3


def test_a_disabled_compressor_passes_the_originals_through_even_with_a_memo_entry_a14(
    lingua, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(COMPRESSION_CACHE_ENV, str(tmp_path / "memo"))
    ContextCompressor().compress(_DOCS_TEXT, target_ratio=0.5)
    monkeypatch.setenv("CAGE_DISABLE_COMPRESSION", "1")
    docs, stats = ContextCompressor().compress(_DOCS_TEXT, target_ratio=0.5)
    assert docs == _DOCS_TEXT and stats.applied is False and stats.cache_hit is None


def test_a_failed_memo_write_still_serves_the_computed_text_a14(lingua, monkeypatch, tmp_path, capsys) -> None:
    blocker = tmp_path / "memo"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    monkeypatch.setenv(COMPRESSION_CACHE_ENV, str(blocker))
    docs, stats = ContextCompressor().compress(_DOCS_TEXT, target_ratio=0.5)
    assert docs and stats.applied and stats.cache_hit is False
    assert "failed" in capsys.readouterr().out


def test_off_the_cpu_the_weights_load_in_float32_at_the_pin_and_the_cpu_call_is_s0s_a14(lingua, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    assert ContextCompressor(device="cuda").compressor is not None
    assert _FakePromptCompressor.inits[-1] == {"device_map": "cuda", "model_config": {"torch_dtype": torch.float32}}
    monkeypatch.setenv(COMPRESSOR_REVISION_ENV, _REV)
    assert ContextCompressor(device="cuda").compressor is not None
    assert _FakePromptCompressor.inits[-1]["model_config"] == {"torch_dtype": torch.float32, "revision": _REV}
    assert ContextCompressor(device="cpu").compressor is not None
    assert _FakePromptCompressor.inits[-1] == {"device_map": "cpu", "model_config": {"revision": _REV}}
    monkeypatch.delenv(COMPRESSOR_REVISION_ENV)
    assert ContextCompressor(device="cpu").compressor is not None
    assert _FakePromptCompressor.inits[-1] == {"device_map": "cpu"}  # the call S0 made


def test_release_drops_the_weights_and_the_next_call_reloads_a14(lingua) -> None:
    c = ContextCompressor()
    assert c.release() is False
    c.compress(_DOCS_TEXT, target_ratio=0.5)
    assert c.release() is True and c._compressor is None
    c.compress(_DOCS_TEXT, target_ratio=0.5)
    assert len(_FakePromptCompressor.inits) == 2 and _FakePromptCompressor.calls == 2


# --------------------------------------------------------------------------- #
# A14 / A9: the index and the reranker
# --------------------------------------------------------------------------- #


def test_each_document_set_gets_its_own_index_folder_a14(tmp_path) -> None:
    sha = corpus_doc_ids_sha1(_DOCS)
    named = default_index_dir(base_dir=tmp_path, dataset_name="qasper",
                              embedding_model="intfloat/e5-large-v2", corpus_sha1=sha)
    assert named.name == f"ir_qasper_intfloat_e5-large-v2_{sha[:12]}"
    legacy = default_index_dir(base_dir=tmp_path, dataset_name="qasper", embedding_model="intfloat/e5-large-v2")
    assert legacy.name == "ir_qasper_intfloat_e5-large-v2"
    other = default_index_dir(base_dir=tmp_path, dataset_name="qasper", embedding_model="intfloat/e5-large-v2",
                              corpus_sha1=corpus_doc_ids_sha1(_DOCS[:1]))
    assert other != named


def test_the_index_rebuilds_on_another_query_device_and_records_its_build_a14_a9(
    tmp_path, monkeypatch, stub_ir_deps, capsys
) -> None:
    monkeypatch.delenv(STALE_INDEX_OPT_IN_ENV, raising=False)
    d = tmp_path / "ir"
    built = ensure_ir_index(index_dir=d, documents=_DOCS, embedding_model=_E5_MODEL)
    assert built.build_s is not None and built.build_s >= 0.0
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["embedding_device"] == "cpu"
    (d / "faiss.index").write_bytes(b"untouched")
    loaded = ensure_ir_index(index_dir=d, documents=_DOCS, embedding_model=_E5_MODEL)
    assert loaded.build_s is None and (d / "faiss.index").read_bytes() == b"untouched"
    moved = ensure_ir_index(index_dir=d, documents=_DOCS, embedding_model=_E5_MODEL, device="cuda")
    assert moved.build_s is not None and (d / "faiss.index").read_bytes() != b"untouched"
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["embedding_device"] == "cuda"
    assert "rebuilding so passages and queries share one device" in capsys.readouterr().out


def test_an_unstamped_index_reads_as_cpu_built_a14(tmp_path, monkeypatch, stub_ir_deps) -> None:
    """Every persisted index before the stamp was built by the runner on the CPU."""
    monkeypatch.delenv(STALE_INDEX_OPT_IN_ENV, raising=False)
    d = tmp_path / "ir_old"
    _write_index_dir(d, embedding_model=_E5_MODEL, uses_e5_prefixes=True)
    assert ensure_ir_index(index_dir=d, documents=_DOCS, embedding_model=_E5_MODEL).build_s is None
    assert (d / "faiss.index").read_bytes() == b"placeholder"
    assert ensure_ir_index(index_dir=d, documents=_DOCS, embedding_model=_E5_MODEL, device="cuda").build_s is not None


def test_the_encoder_releases_its_weights_and_reloads_on_its_device_a14(stub_ir_deps) -> None:
    idx = FaissIRIndex(embedding_model=_E5_MODEL, device="cuda")
    idx.build(_DOCS)
    assert idx.release() is True and idx._st_model is None and idx._index is not None
    assert idx.release() is False
    idx._ensure_deps()
    assert idx._st_model.device == "cuda"


def test_the_reranker_releases_its_weights_keeps_its_counters_and_reloads_a14(monkeypatch) -> None:
    loads: List[str] = []

    class _Counting(_FakeCrossEncoder):
        def __init__(self, model_name: str, device: str = "cpu") -> None:
            super().__init__(model_name, device=device)
            loads.append(device)

    st_mod = types.ModuleType("sentence_transformers")
    st_mod.CrossEncoder = _Counting
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)
    monkeypatch.delenv("CAGE_RERANKER_REVISION", raising=False)
    docs = {"a": IRDocument(doc_id="a", text="short", metadata={}),
            "b": IRDocument(doc_id="b", text="a longer passage", metadata={})}
    resolver = types.SimpleNamespace(resolve_hits=lambda hits: [docs[h.doc_id] for h in hits])
    hits = [IRHit(doc_id="a", score=0.0), IRHit(doc_id="b", score=0.0)]
    reranker = CrossEncoderReranker("fake/cross-encoder", device="cuda")
    first = reranker.rerank("q", hits, resolver)
    assert reranker.release() is True and reranker._model is None and reranker.release() is False
    second = reranker.rerank("q", hits, resolver)
    assert loads == ["cuda", "cuda"] and reranker.scored_pairs == 4
    assert [h.doc_id for h in second] == [h.doc_id for h in first] == ["b", "a"]


# --------------------------------------------------------------------------- #
# The runner: prep timing, truncation originals, release before the window
# --------------------------------------------------------------------------- #


class _FakeDenseIndex:
    """What ensure_ir_index returns, minus the encoder: ranks the corpus in order."""

    def __init__(self, documents, device: str, events: List[tuple]) -> None:
        self._docs = list(documents)
        self.device = device
        self.build_s = 0.25
        self.stale_index_opt_in = False
        self.events = events

    def search(self, query: str, *, top_k: int = 5) -> List[IRHit]:
        return [IRHit(doc_id=d.doc_id, score=1.0 - i / 100) for i, d in enumerate(self._docs[:top_k])]

    def resolve_hits(self, hits):
        by_id = {d.doc_id: d for d in self._docs}
        return [by_id[h.doc_id] for h in hits if h.doc_id in by_id]

    def release(self) -> bool:
        self.events.append(("release",))
        return True


def _dense_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str, open_loop: bool = False):
    import test_run_experiment_campaign_layout as layout

    events: List[tuple] = []
    made: List[Dict[str, Any]] = []

    def fake_ensure(*, index_dir, documents, embedding_model, rebuild=False, device="cpu",
                    embedding_revision=None):
        made.append({"dir": Path(index_dir), "device": device,
                     "index": _FakeDenseIndex(documents, device, events)})
        return made[-1]["index"]

    class _Recording(layout._AsyncCampaignEngine):
        def generate(self, request, stream=False):
            events.append(("send", str(request.request_id)))
            return super().generate(request, stream=stream)

        async def async_stream_generate(self, request, *, on_first_token=None):
            events.append(("send", str(request.request_id)))
            return await super().async_stream_generate(request, on_first_token=on_first_token)

    monkeypatch.setattr(layout.runner, "ensure_ir_index", fake_ensure)
    root = tmp_path / "camp" / "a" / layout.RUN_ID
    layout._run_cell(
        monkeypatch, root, "rag", "squad_v2", ttft_base=100.0, num_trials=1,
        engine_factory=lambda model: _Recording(model, "rag", 100.0),
        extra_argv=("--top-k", "3", "--ir-index-dir", str(tmp_path / "ir"), *extra,
                    *(layout.OPEN_LOOP_ARGV if open_loop else ())),
    )
    (wdir,) = sorted(root.glob("cells/*/window_squad_v2-*"))
    return wdir, made, events, layout


def test_a_capped_retrieval_cell_keeps_its_uncapped_documents_and_frees_the_gpu_first_a8_a14_a9(
    tmp_path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    freed: List[int] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: freed.append(1))
    wdir, made, events, layout = _dense_cell(
        tmp_path, monkeypatch, "--max-context-docs", "1", "--encoder-device", "cuda"
    )
    (built,) = made
    assert built["device"] == "cuda"
    assert built["dir"].name.startswith("ir_squad_v2_") and len(built["dir"].name.rsplit("_", 1)[1]) == 12
    evidence = layout._read_jsonl(wdir / "qa_evidence.jsonl")
    assert evidence
    for e in evidence:
        assert len(e["used_contexts"]) == 1 and len(e["original_contexts"]) == 3
        assert e["used_contexts"][0] == e["original_contexts"][0]  # the cap keeps the head
    rows = layout._read_jsonl(wdir / "requests.jsonl")
    assert rows
    for r in rows:
        assert r["retrieval_ms"] is not None and r["rerank_ms"] is None  # no reranker in this cell
        assert r["prep_start_ts"] <= r["prep_end_ts"] and r["prep_ms"] >= r["retrieval_ms"]
    exp = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))["experiment"]
    assert exp["encoder_device"] == "cuda" and exp["prep_helpers_released"] == ["encoder"]
    assert exp["ir_index"] == {
        "dir": str(built["dir"]), "corpus_sha1": exp["ir_index"]["corpus_sha1"],
        "num_documents": exp["ir_index"]["num_documents"], "built": True, "build_s": 0.25, "device": "cuda",
    }
    assert built["dir"].name.endswith(exp["ir_index"]["corpus_sha1"][:12])
    # released once, after the measured preparation and before the first measured send
    assert events.count(("release",)) == 1 and freed == [1]
    first_measured = next(i for i, e in enumerate(events) if e[0] == "send" and "warmup" not in e[1])
    assert events.index(("release",)) < first_measured


def test_an_open_loop_window_frees_the_gpu_after_its_preparation_and_before_its_first_send_a14(
    tmp_path, monkeypatch
) -> None:
    """S1's pressure windows are open loop: the measured set and the pre-load
    are prepared, the helpers leave the GPU, then the dispatcher sends."""
    torch = pytest.importorskip("torch")
    freed: List[int] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: freed.append(1))
    wdir, made, events, layout = _dense_cell(tmp_path, monkeypatch, "--encoder-device", "cuda", open_loop=True)
    assert events.count(("release",)) == 1 and freed == [1]
    first_measured = next(i for i, e in enumerate(events) if e[0] == "send" and "warmup" not in e[1])
    assert events.index(("release",)) < first_measured
    exp = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))["experiment"]
    assert exp["prep_helpers_released"] == ["encoder"]
    rows = layout._read_jsonl(wdir / "requests.jsonl")
    assert rows and all(r["retrieval_ms"] is not None and r["prep_ms"] is not None for r in rows)


def test_a_cpu_retrieval_cell_releases_nothing_and_keeps_no_originals_without_a_cap_a14_a8(
    tmp_path, monkeypatch
) -> None:
    wdir, made, events, layout = _dense_cell(tmp_path, monkeypatch)
    assert made[0]["device"] == "cpu" and ("release",) not in events
    exp = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))["experiment"]
    assert exp["prep_helpers_released"] == [] and exp["encoder_device"] == "cpu"
    assert exp["compressor_device"] is None and exp["compression_cache_dir"] is None
    assert all(e["original_contexts"] is None for e in layout._read_jsonl(wdir / "qa_evidence.jsonl"))


def test_a_gold_cell_rows_carry_the_prep_columns_and_no_index_a9(tmp_path, monkeypatch) -> None:
    import test_run_experiment_campaign_layout as layout

    root = tmp_path / "camp" / "a" / layout.RUN_ID
    layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=100.0, num_trials=1)
    (wdir,) = layout._window_dirs(root, "no_cache", "squad_v2")
    for r in layout._read_jsonl(wdir / "requests.jsonl"):
        assert r["prep_ms"] >= 0.0 and r["prep_start_ts"] <= r["prep_end_ts"]
        assert r["retrieval_ms"] is None and r["rerank_ms"] is None and r["compression_cache_hit"] is None
    exp = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))["experiment"]
    assert exp["ir_index"] is None and exp["encoder_device"] is None


def test_open_loop_stub_rows_carry_the_prep_columns_a9() -> None:
    """A dropped or failed arrival is a row too (D6 attainment): it carries the
    preparation of the request it would have sent."""
    src = (REPO_ROOT / "scripts" / "3_run" / "run_experiment.py").read_text(encoding="utf-8")
    stub = src[src.index("stub_row: Dict[str, Any] = {"):src.index("merge_open_loop_row(stub_row, record)")]
    assert "**_prep_timing(meta)," in stub


def test_a_corpus_rung_keeps_the_full_block_and_the_full_budget_keeps_none_a8(tmp_path, monkeypatch) -> None:
    import test_run_experiment_campaign_layout as layout

    mpath, manifest = layout._trunc_manifest(tmp_path)
    rung_root = tmp_path / "rung" / "a" / layout.RUN_ID
    layout._run_corpus_cell(
        monkeypatch, rung_root, env=layout.TRUNC_ENV,
        extra=("--query-manifest", str(mpath), "--corpus-prefix-budget", str(layout.TRUNC_RUNG),
               "--corpus-rung", str(layout.TRUNC_RUNG)),
    )
    row_key = f"corpus-trunc|none|none|single|vllm|qwen3-14b|F1|cb{layout.TRUNC_RUNG}"
    evidence = layout._read_jsonl(layout._window(rung_root, row_key, "squad_v2", 1) / "qa_evidence.jsonl")
    assert evidence
    for e in evidence:
        full = manifest["blocks"][manifest["question_to_block"][e["example_id"]]]["text"]
        assert e["original_contexts"] == [full] and e["used_contexts"] != [full]
    full_root = tmp_path / "full" / "a" / layout.RUN_ID
    layout._run_corpus_cell(
        monkeypatch, full_root, env=layout.B3_ENV,
        extra=("--query-manifest", str(mpath), "--corpus-prefix-budget", str(layout.TRUNC_FULL)),
    )
    (wdir,) = sorted(full_root.glob("cells/*/window_squad_v2-01"))
    assert all(e["original_contexts"] is None for e in layout._read_jsonl(wdir / "qa_evidence.jsonl"))


# --------------------------------------------------------------------------- #
# A14 item 7: the S0 reference and the device probe
# --------------------------------------------------------------------------- #


def _probe():
    path = REPO_ROOT / "scripts" / "checks" / "probe_compressor_device.py"
    spec = importlib.util.spec_from_file_location("probe_compressor_device_a14", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_shipped_reference_holds_s0s_167_distinct_inputs_a14() -> None:
    probe = _probe()
    with gzip.open(probe.DEFAULT_REFERENCE, "rt", encoding="utf-8") as fh:
        header = json.loads(fh.readline())["_meta"]
    assert header["distinct_inputs"] == 167 and header["keep_ratio"] == 0.5 and header["device"] == "cpu"
    items = probe.load_reference(probe.DEFAULT_REFERENCE)
    assert len(items) == 167 and len({i["key"] for i in items}) == 167
    for item in items:
        assert item["contexts"] and all(c.strip() for c in item["contexts"])
        assert isinstance(item["compressed"], list) and item["compressed"]
        blob = json.dumps(item["contexts"], ensure_ascii=False).encode("utf-8")
        assert hashlib.sha256(blob).hexdigest() == item["key"]


def test_the_probe_passes_identical_fitting_faster_output_and_names_each_failure_a14() -> None:
    probe = _probe()
    items = [{"key": f"{i:064x}", "contexts": [f"doc {i} words"], "compressed": [f"doc {i}"]} for i in range(4)]
    ticks = iter([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0])  # 1,000 ms per input
    good = probe.compare_outputs(items, lambda docs: [docs[0].rsplit(" ", 1)[0]], clock=lambda: next(ticks))
    assert (good["inputs"], good["identical"], good["median_ms"], good["p90_ms"]) == (4, 4, 1000.0, 1000.0)
    good["peak_reserved_gib"] = 3.5
    assert probe.failure_reasons(good, headroom_gib=8.0) == []
    ticks = iter([0.0, 1.0] * 4)
    bad = probe.compare_outputs(items, lambda docs: ["changed"], clock=lambda: next(ticks))
    bad.update(peak_reserved_gib=9.0)
    reasons = probe.failure_reasons(bad, headroom_gib=8.0)
    assert len(reasons) == 2 and "4 of 4 outputs differ" in reasons[0] and "9.0 GiB" in reasons[1]
    slow = dict(good, median_ms=probe.S0_CPU_MEDIAN_MS)
    assert "not below S0's CPU median" in probe.failure_reasons(slow, headroom_gib=8.0)[0]
    assert probe.failure_reasons(dict(good, peak_reserved_gib=None), headroom_gib=8.0)


def test_the_probe_prints_its_one_line_and_refuses_a_rate_override_a14(monkeypatch, tmp_path, capsys) -> None:
    probe = _probe()
    monkeypatch.setenv("CAGE_COMPRESSION_RATE", "0.3")
    out = tmp_path / "probe.json"
    assert probe.main(["--out", str(out)]) == 1
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("COMPRESS_DEVICE_FAILED reason=ValueError: CAGE_COMPRESSION_RATE is set")
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is False


def test_a_windows_metrics_carry_the_relaunch_realized_pool_a11(tmp_path, monkeypatch) -> None:
    import test_run_experiment_campaign_layout as layout

    pool = {"engine": "vllm", "planned_tokens": 298687, "realized_tokens": 298672, "ok": True, "gap": -5.0e-5}
    root = tmp_path / "camp" / "a" / layout.RUN_ID
    layout._run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=100.0, num_trials=1,
                     extra_env={"CAGE_REALIZED_POOL_JSON": json.dumps(pool)})
    (wdir,) = layout._window_dirs(root, "no_cache", "squad_v2")
    exp = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))["experiment"]
    assert exp["realized_pool"] == pool
    monkeypatch.delenv("CAGE_REALIZED_POOL_JSON")  # extra_env set it through this monkeypatch
    other = tmp_path / "camp2" / "a" / layout.RUN_ID
    layout._run_cell(monkeypatch, other, "no_cache", "squad_v2", ttft_base=100.0, num_trials=1)
    (wdir2,) = layout._window_dirs(other, "no_cache", "squad_v2")
    assert json.loads((wdir2 / "metrics.json").read_text(encoding="utf-8"))["experiment"]["realized_pool"] is None
