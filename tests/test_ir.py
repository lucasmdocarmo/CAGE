"""Tests for IR (Information Retrieval) utilities.

These are unit-level tests that avoid building large FAISS indexes.
"""

import pytest

np = pytest.importorskip("numpy")

from src.orchestration.ir import build_corpus_from_contexts, retrieval_hit_rate, stable_text_id
from src.data.loader import CAGExample
from src.utils.prompting import format_qa_prompt


def test_stable_text_id_deterministic():
    a = stable_text_id("hello")
    b = stable_text_id("hello")
    c = stable_text_id("hello!")
    assert a == b
    assert a != c


def test_build_corpus_from_contexts_deduplicates():
    ex1 = CAGExample(
        id="1",
        question="q1",
        context=["doc a", "doc b"],
        answer="a",
        metadata={},
    )
    ex2 = CAGExample(
        id="2",
        question="q2",
        context=["doc a", "doc c"],
        answer="b",
        metadata={},
    )

    docs = build_corpus_from_contexts([ex1, ex2], dataset_name="unit")
    texts = sorted([d.text for d in docs])

    assert texts == ["doc a", "doc b", "doc c"]


def test_retrieval_hit_rate():
    gold = ["a", "b"]
    assert retrieval_hit_rate(gold_doc_ids=gold, retrieved_doc_ids=["x", "y"]) == 0.0
    assert retrieval_hit_rate(gold_doc_ids=gold, retrieved_doc_ids=["x", "b"]) == 1.0


def test_retrieval_rank_of_gold():
    # Graded companion (fix #5-C): returns the 1-based rank of the first gold match, or
    # None on a miss / when gold is unknown. Powers MRR = mean(1/rank) downstream.
    from src.orchestration.ir import retrieval_rank_of_gold

    gold = ["a", "b"]
    assert retrieval_rank_of_gold(gold_doc_ids=gold, retrieved_doc_ids=["a", "x"]) == 1
    assert retrieval_rank_of_gold(gold_doc_ids=gold, retrieved_doc_ids=["x", "b", "a"]) == 2
    assert retrieval_rank_of_gold(gold_doc_ids=gold, retrieved_doc_ids=["x", "y"]) is None  # miss
    assert retrieval_rank_of_gold(gold_doc_ids=[], retrieved_doc_ids=["x"]) is None  # gold unknown
    # Text fallback preserves order when ids do not match.
    assert (
        retrieval_rank_of_gold(
            gold_doc_ids=["zzz"],
            retrieved_doc_ids=["p", "q"],
            gold_texts=["the sky is blue"],
            retrieved_texts=["grass is green", "the sky is blue"],
        )
        == 2
    )


def test_format_qa_prompt_contains_context_and_question():
    prompt = format_qa_prompt("What?", ["ctx1", "ctx2"], system_prefix="SYS\n")
    assert "SYS" in prompt
    assert "Context 1:" in prompt
    assert "ctx1" in prompt
    assert "Question: What?" in prompt
    assert prompt.rstrip().endswith("Answer:")


# --------------------------------------------------------------------------
# F6: a persisted index built BEFORE the e5/bge query:/passage: prefix fix is a
# typed refusal on load (StaleIndexError), never a printed WARNING that serves
# out-of-distribution retrieval silently. CAGE_ALLOW_STALE_INDEX=1 is the only
# escape hatch (pilot archives only) and it stamps provenance on the way through.
# --------------------------------------------------------------------------

import json
import sys
import types
from pathlib import Path

from src.orchestration.ir import (
    STALE_INDEX_OPT_IN_ENV,
    FaissIRIndex,
    IndexRevisionMismatchError,
    IRDocument,
    StaleIndexError,
    corpus_doc_ids_sha1,
    ensure_ir_index,
)

_E5_MODEL = "intfloat/e5-large-v2"
_PLAIN_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DOCS = [
    IRDocument(doc_id=stable_text_id("alpha"), text="alpha", metadata={}),
    IRDocument(doc_id=stable_text_id("beta"), text="beta", metadata={}),
]


def _write_index_dir(
    directory: Path,
    *,
    embedding_model: str,
    uses_e5_prefixes: "bool | None",
    documents: list = _DOCS,
    embedding_revision: "str | None" = None,
) -> dict:
    """Persist a meta.json + documents.jsonl + placeholder faiss.index.

    ``uses_e5_prefixes=None`` omits the key entirely, which is exactly what an
    index persisted before the prefix fix looks like on disk;
    ``embedding_revision=None`` likewise omits the A5 revision stamp (an index
    built before the pin was enforced).
    """
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "embedding_model": embedding_model,
        "normalize_embeddings": True,
        "num_documents": len(documents),
        "doc_ids_sha1": corpus_doc_ids_sha1(documents),
    }
    if uses_e5_prefixes is not None:
        meta["uses_e5_prefixes"] = uses_e5_prefixes
    if embedding_revision is not None:
        meta["embedding_revision"] = embedding_revision
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with (directory / "documents.jsonl").open("w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps({"doc_id": d.doc_id, "text": d.text, "metadata": d.metadata}) + "\n")
    (directory / "faiss.index").write_bytes(b"placeholder")
    return meta


class _FakeSentenceTransformer:
    #: every construction, so a test can prove which revision kwarg reached
    #: the encoder (A5: the pin is enforced, not merely recorded).
    constructed: list = []

    def __init__(self, model_name: str, device: str = "cpu", **kwargs) -> None:
        self.model_name = model_name
        self.device = device
        self.kwargs = dict(kwargs)
        _FakeSentenceTransformer.constructed.append((model_name, dict(kwargs)))

    def encode(self, texts, **_kwargs):
        return np.zeros((len(texts), 4), dtype="float32")


class _FakeFlatIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.added = 0

    def add(self, embeddings) -> None:
        self.added += int(embeddings.shape[0])


@pytest.fixture
def stub_ir_deps(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install fake ``sentence_transformers`` and ``faiss`` so load/build never
    touch a real model or a real FAISS binary."""
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)

    faiss_mod = types.ModuleType("faiss")
    faiss_mod.omp_set_num_threads = lambda n: None
    faiss_mod.IndexFlatIP = _FakeFlatIndex
    faiss_mod.IndexFlatL2 = _FakeFlatIndex
    faiss_mod.read_index = lambda path: ("loaded", path)
    faiss_mod.write_index = lambda index, path: Path(path).write_bytes(b"written")
    monkeypatch.setitem(sys.modules, "faiss", faiss_mod)
    return faiss_mod


@pytest.fixture
def forbid_ir_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any dependency import proves the refusal fired too late."""
    st_mod = types.ModuleType("sentence_transformers")

    def _boom(*_a, **_k):
        raise AssertionError("SentenceTransformer must not be constructed for a refused index")

    st_mod.SentenceTransformer = _boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)


def test_stale_index_error_is_a_typed_runtime_refusal() -> None:
    assert issubclass(StaleIndexError, RuntimeError)
    assert STALE_INDEX_OPT_IN_ENV == "CAGE_ALLOW_STALE_INDEX"


@pytest.mark.parametrize("flag_on_disk", [None, False], ids=["key-absent", "key-false"])
def test_load_refuses_stale_pre_prefix_index_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forbid_ir_deps: None, flag_on_disk
) -> None:
    monkeypatch.delenv(STALE_INDEX_OPT_IN_ENV, raising=False)
    index_dir = tmp_path / "ir_stale"
    _write_index_dir(index_dir, embedding_model=_E5_MODEL, uses_e5_prefixes=flag_on_disk)

    with pytest.raises(StaleIndexError) as excinfo:
        FaissIRIndex.load(index_dir)

    msg = str(excinfo.value)
    assert str(index_dir) in msg
    assert _E5_MODEL in msg
    assert STALE_INDEX_OPT_IN_ENV in msg
    assert "--rebuild-ir-index" in msg
    # The refusal leaves the archive untouched (no provenance stamp on a refused load).
    assert "stale_index_opt_in" not in json.loads((index_dir / "meta.json").read_text())


@pytest.mark.parametrize("flag_value", ["0", "", "no", "false"])
def test_load_stale_opt_in_requires_truthy_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forbid_ir_deps: None, flag_value: str
) -> None:
    monkeypatch.setenv(STALE_INDEX_OPT_IN_ENV, flag_value)
    index_dir = tmp_path / "ir_stale"
    _write_index_dir(index_dir, embedding_model=_E5_MODEL, uses_e5_prefixes=None)

    with pytest.raises(StaleIndexError):
        FaissIRIndex.load(index_dir)


@pytest.mark.parametrize("flag_value", ["1", "true", "TRUE", "yes"])
def test_load_stale_opt_in_downgrades_to_warning_and_stamps_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_ir_deps: types.ModuleType,
    capsys: pytest.CaptureFixture[str],
    flag_value: str,
) -> None:
    monkeypatch.setenv(STALE_INDEX_OPT_IN_ENV, flag_value)
    index_dir = tmp_path / "ir_pilot_archive"
    _write_index_dir(index_dir, embedding_model=_E5_MODEL, uses_e5_prefixes=None)

    inst = FaissIRIndex.load(index_dir)

    # Returned flag: the caller can see (and record) that this index is stale.
    assert inst.stale_index_opt_in is True
    # Queries keep matching the un-prefixed passages the archive was built with.
    assert inst.uses_e5_prefixes is False
    assert [d.doc_id for d in inst.documents] == [d.doc_id for d in _DOCS]
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert STALE_INDEX_OPT_IN_ENV in out
    # Stamp on the archive metadata so provenance survives beyond this process.
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["stale_index_opt_in"] is True
    assert meta["stale_index_opt_in_env"] == STALE_INDEX_OPT_IN_ENV
    assert meta["embedding_model"] == _E5_MODEL  # existing keys preserved
    assert meta["uses_e5_prefixes"] is False  # the stale build fact is now explicit


def test_load_fresh_prefixed_index_unaffected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_ir_deps: types.ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(STALE_INDEX_OPT_IN_ENV, raising=False)
    index_dir = tmp_path / "ir_fresh"
    _write_index_dir(index_dir, embedding_model=_E5_MODEL, uses_e5_prefixes=True)
    before = (index_dir / "meta.json").read_bytes()

    inst = FaissIRIndex.load(index_dir)

    assert inst.uses_e5_prefixes is True
    assert inst.stale_index_opt_in is False
    assert "WARNING" not in capsys.readouterr().out
    assert (index_dir / "meta.json").read_bytes() == before


def test_load_non_prefix_model_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_ir_deps: types.ModuleType
) -> None:
    """A model family that never needed prefixes is not stale for lacking them."""
    monkeypatch.delenv(STALE_INDEX_OPT_IN_ENV, raising=False)
    index_dir = tmp_path / "ir_plain"
    _write_index_dir(index_dir, embedding_model=_PLAIN_MODEL, uses_e5_prefixes=None)

    inst = FaissIRIndex.load(index_dir)

    assert inst.uses_e5_prefixes is False
    assert inst.stale_index_opt_in is False


def test_ensure_ir_index_propagates_stale_refusal_and_rebuild_clears_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_ir_deps: types.ModuleType
) -> None:
    """The driver seam (ensure_ir_index) neither silently loads nor silently
    rebuilds a stale-prefix index: it refuses, and only --rebuild-ir-index
    (rebuild=True) produces a fresh, prefixed index that then loads cleanly."""
    monkeypatch.delenv(STALE_INDEX_OPT_IN_ENV, raising=False)
    index_dir = tmp_path / "ir_stale"
    _write_index_dir(index_dir, embedding_model=_E5_MODEL, uses_e5_prefixes=None)

    with pytest.raises(StaleIndexError):
        ensure_ir_index(index_dir=index_dir, documents=_DOCS, embedding_model=_E5_MODEL)

    rebuilt = ensure_ir_index(
        index_dir=index_dir, documents=_DOCS, embedding_model=_E5_MODEL, rebuild=True
    )
    assert rebuilt.uses_e5_prefixes is True
    assert json.loads((index_dir / "meta.json").read_text())["uses_e5_prefixes"] is True

    reloaded = ensure_ir_index(index_dir=index_dir, documents=_DOCS, embedding_model=_E5_MODEL)
    assert reloaded.uses_e5_prefixes is True
    assert reloaded.stale_index_opt_in is False


# ---------------------------------------------------------------------------
# Backlog A5 (review 2026-09-17 defect 5): the dense retriever REVISION is
# enforced, not merely recorded. The pin reaches SentenceTransformer(revision=),
# is stamped in meta.json, refuses a mismatching load, and drives a rebuild.
# ---------------------------------------------------------------------------

_REV_A = "f169b11e22de13617baa190a028a32f3493550b6"
_REV_B = "0000000000000000000000000000000000000000"


def test_index_revision_mismatch_is_a_typed_runtime_refusal() -> None:
    assert issubclass(IndexRevisionMismatchError, RuntimeError)


def test_build_passes_the_revision_to_the_encoder_and_stamps_meta(
    tmp_path: Path, stub_ir_deps: types.ModuleType
) -> None:
    _FakeSentenceTransformer.constructed = []
    idx = FaissIRIndex(embedding_model=_E5_MODEL, embedding_revision=_REV_A)
    idx.build(_DOCS)
    idx.save(tmp_path / "ir_rev")
    assert _FakeSentenceTransformer.constructed == [(_E5_MODEL, {"revision": _REV_A})]
    meta = json.loads((tmp_path / "ir_rev" / "meta.json").read_text(encoding="utf-8"))
    assert meta["embedding_revision"] == _REV_A
    # Unpinned build: no revision kwarg (HF default), an explicit null stamp.
    _FakeSentenceTransformer.constructed = []
    unpinned = FaissIRIndex(embedding_model=_E5_MODEL)
    unpinned.build(_DOCS)
    unpinned.save(tmp_path / "ir_unpinned")
    assert _FakeSentenceTransformer.constructed == [(_E5_MODEL, {})]
    meta = json.loads((tmp_path / "ir_unpinned" / "meta.json").read_text(encoding="utf-8"))
    assert "embedding_revision" in meta and meta["embedding_revision"] is None


@pytest.mark.parametrize("on_disk", [None, _REV_B], ids=["unstamped", "other-revision"])
def test_load_refuses_a_revision_mismatch_before_any_dependency(
    tmp_path: Path, forbid_ir_deps: None, on_disk: "str | None"
) -> None:
    index_dir = tmp_path / "ir_rev"
    _write_index_dir(
        index_dir, embedding_model=_E5_MODEL, uses_e5_prefixes=True, embedding_revision=on_disk
    )
    with pytest.raises(IndexRevisionMismatchError) as excinfo:
        FaissIRIndex.load(index_dir, embedding_revision=_REV_A)
    msg = str(excinfo.value)
    assert _REV_A in msg and str(index_dir) in msg and "--rebuild-ir-index" in msg


def test_load_with_matching_revision_and_unpinned_load_carry_the_stamp(
    tmp_path: Path, stub_ir_deps: types.ModuleType
) -> None:
    index_dir = tmp_path / "ir_rev"
    _write_index_dir(
        index_dir, embedding_model=_E5_MODEL, uses_e5_prefixes=True, embedding_revision=_REV_A
    )
    _FakeSentenceTransformer.constructed = []
    inst = FaissIRIndex.load(index_dir, embedding_revision=_REV_A)
    assert inst.embedding_revision == _REV_A
    assert _FakeSentenceTransformer.constructed == [(_E5_MODEL, {"revision": _REV_A})]
    # An unpinned load serves the archive at ITS stamped revision (provenance
    # preserved), never at a silently different one.
    _FakeSentenceTransformer.constructed = []
    inst = FaissIRIndex.load(index_dir)
    assert inst.embedding_revision == _REV_A
    assert _FakeSentenceTransformer.constructed == [(_E5_MODEL, {"revision": _REV_A})]


def test_ensure_ir_index_rebuilds_on_revision_drift_and_reuses_on_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_ir_deps: types.ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(STALE_INDEX_OPT_IN_ENV, raising=False)
    index_dir = tmp_path / "ir_rev"
    _write_index_dir(index_dir, embedding_model=_E5_MODEL, uses_e5_prefixes=True)

    # Unstamped index + a pin: rebuilt AT the pin (never served under it).
    built = ensure_ir_index(
        index_dir=index_dir, documents=_DOCS, embedding_model=_E5_MODEL,
        embedding_revision=_REV_A,
    )
    assert built.embedding_revision == _REV_A
    assert "rebuilding" in capsys.readouterr().out
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["embedding_revision"] == _REV_A

    # Same pin: reused (loaded, no rebuild; the placeholder faiss bytes prove it).
    (index_dir / "faiss.index").write_bytes(b"untouched")
    again = ensure_ir_index(
        index_dir=index_dir, documents=_DOCS, embedding_model=_E5_MODEL,
        embedding_revision=_REV_A,
    )
    assert again.embedding_revision == _REV_A
    assert (index_dir / "faiss.index").read_bytes() == b"untouched"

    # A different pin: rebuilt at the new pin.
    drifted = ensure_ir_index(
        index_dir=index_dir, documents=_DOCS, embedding_model=_E5_MODEL,
        embedding_revision=_REV_B,
    )
    assert drifted.embedding_revision == _REV_B
    assert (index_dir / "faiss.index").read_bytes() != b"untouched"
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["embedding_revision"] == _REV_B
