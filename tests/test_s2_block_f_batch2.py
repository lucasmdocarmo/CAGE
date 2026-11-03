"""S2 block F batch 2, row F-6: ONE retrieval corpus per dataset.

S1 served two corpora per dataset because the corpus followed each
window's registered N (the measured prefix of the manifest trial): Qasper
3,587 documents at N 300 against 4,068 at N >= 500, MuSiQue 10,881 at 800
against 19,601 at 2,000 (S1 report section 14). A retrieval contrast across
rungs then compared searches over different document sets. Row F-6 builds
the corpus from the UNION of every manifest trial's ids plus the distractor
pool, so every cell of a dataset searches the same documents whatever its
N or trial; the measured set stays the trial prefix (A9).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.orchestration.ir import corpus_doc_ids_sha1  # noqa: E402

import test_run_experiment_campaign_layout as layout  # noqa: E402
from test_a14_prep_on_gpu import _FakeDenseIndex  # noqa: E402

DISTRACTORS = 4


def _rag_manifest_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    manifest_path: Path,
    num_queries: int,
    num_trials: int,
) -> list[list[Any]]:
    """Drive the real runner on a dense-retrieval cell under a query manifest
    and return the document set handed to the index builder, one per trial."""
    made: list[list[Any]] = []

    def fake_ensure(*, index_dir, documents, embedding_model, rebuild=False, device="cpu",
                    embedding_revision=None):
        made.append(list(documents))
        return _FakeDenseIndex(documents, device, [])

    monkeypatch.setattr(layout.runner, "ensure_ir_index", fake_ensure)
    layout._run_cell(
        monkeypatch, root, "rag", "squad_v2", ttft_base=100.0, num_trials=num_trials,
        loader_factory=layout._BigPoolLoader,
        engine_factory=lambda model: layout._AsyncCampaignEngine(model, "rag", 100.0),
        extra_argv=(
            "--top-k", "3", "--ir-index-dir", str(tmp_path / "ir"),
            "--query-manifest", str(manifest_path),
            # argparse keeps the LAST spelling of a repeated option, so this
            # overrides the harness's --num-queries N_QUERIES.
            "--num-queries", str(num_queries),
        ),
        extra_env={"CAGE_DISTRACTOR_DOCS": str(DISTRACTORS)},
    )
    return made


def _windows(root: Path) -> list[Path]:
    return sorted(root.glob("cells/*/window_squad_v2-*"))


def _experiment(wdir: Path) -> dict[str, Any]:
    return json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))["experiment"]


def test_select_manifest_union_covers_every_trial_once_in_manifest_order() -> None:
    pool = layout._BigPoolLoader("squad_v2").load()
    ids = [ex.id for ex in pool]
    manifest = {"trials": {"10": [ids[5]], "2": [ids[3], ids[4], ids[1]], "1": [ids[1], ids[0]]}}
    union = layout.runner.select_manifest_union(manifest, examples=pool)
    # trial 1 first, then 2, then 10 (NUMERIC trial order, not "1" < "10" < "2"),
    # each trial's ids in listed order, an id seen before skipped
    assert [ex.id for ex in union] == [ids[1], ids[0], ids[3], ids[4], ids[5]]
    with pytest.raises(layout.runner.QueryCountError, match="no trial"):
        layout.runner.select_manifest_union({"trials": {}}, examples=pool)
    with pytest.raises(layout.runner.QueryCountError, match="not found"):
        layout.runner.select_manifest_union(
            {"trials": {"1": [ids[0], "squad_v2-q999"]}}, examples=pool
        )
    with pytest.raises(layout.runner.QueryCountError, match="integer"):
        layout.runner.select_manifest_union({"trials": {"one": [ids[0]]}}, examples=pool)
    # review LOW 5: a trial that is not a list of ids is refused, as the prefix helper refuses it
    with pytest.raises(layout.runner.QueryCountError, match="list"):
        layout.runner.select_manifest_union({"trials": {"1": {ids[0]: 1}}}, examples=pool)


def test_manifest_corpus_is_the_trial_union_and_identical_across_n_and_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mpath, manifest = layout._prefix_manifest(tmp_path, layout.MANIFEST_TRIAL_IDS)
    union_ids = {i for ids in manifest["trials"].values() for i in ids}
    pool = {ex.id: ex for ex in layout._BigPoolLoader("squad_v2").load()}
    union_texts = {ex.context[0] for i, ex in pool.items() if i in union_ids}
    # outside EVERY trial, in load order: the distractor candidates
    outside = [ex.context[0] for i, ex in pool.items() if i not in union_ids]

    # Run A: the full N (8) over both trials in one invocation.
    root_a = tmp_path / "a" / "a" / layout.RUN_ID
    made_a = _rag_manifest_cell(
        tmp_path, monkeypatch, root_a, manifest_path=mpath,
        num_queries=layout.N_QUERIES, num_trials=2,
    )
    # Run B: a smaller N (2), trial 1 only, in a second run tree.
    root_b = tmp_path / "b" / "a" / layout.RUN_ID
    made_b = _rag_manifest_cell(
        tmp_path, monkeypatch, root_b, manifest_path=mpath, num_queries=2, num_trials=1,
    )
    assert len(made_a) == 2 and len(made_b) == 1
    hashes = {corpus_doc_ids_sha1(docs) for docs in made_a + made_b}
    assert len(hashes) == 1, "one corpus per dataset: N and trial must not change it"
    for docs in made_a + made_b:
        gold = {d.text for d in docs if d.metadata.get("source") == "dataset_context"}
        distractors = [d.text for d in docs if d.metadata.get("source") == "distractor"]
        assert gold == union_texts
        assert distractors == outside[:DISTRACTORS]  # the union's gold never a distractor
        assert len(docs) == len(union_texts) + DISTRACTORS
    # The MEASURED set stays the trial prefix (A9): N rows, the first N ids.
    for ordinal, wdir in enumerate(_windows(root_a), start=1):
        rows = layout._read_jsonl(wdir / "requests.jsonl")
        assert [r["example_id"] for r in rows] == manifest["trials"][str(ordinal)][:layout.N_QUERIES]
    (wdir_b,) = _windows(root_b)
    assert [r["example_id"] for r in layout._read_jsonl(wdir_b / "requests.jsonl")] == (
        manifest["trials"]["1"][:2]
    )
    # Provenance: metrics.json names the corpus source and its example count.
    (the_hash,) = hashes
    for wdir in _windows(root_a) + [wdir_b]:
        record = _experiment(wdir)["ir_index"]
        assert record["corpus_sha1"] == the_hash
        assert record["corpus_source"] == "manifest-union"
        assert record["corpus_example_count"] == len(union_ids) == 2 * layout.MANIFEST_TRIAL_IDS
        assert record["num_documents"] == len(union_texts) + DISTRACTORS


def test_without_a_manifest_the_corpus_is_the_measured_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pilot path (no manifest) keeps its corpus: the measured examples'
    contexts, no distractors (Decision 3B is manifest-mode only)."""
    made: list[list[Any]] = []

    def fake_ensure(*, index_dir, documents, embedding_model, rebuild=False, device="cpu",
                    embedding_revision=None):
        made.append(list(documents))
        return _FakeDenseIndex(documents, device, [])

    monkeypatch.setattr(layout.runner, "ensure_ir_index", fake_ensure)
    root = tmp_path / "camp" / "a" / layout.RUN_ID
    layout._run_cell(
        monkeypatch, root, "rag", "squad_v2", ttft_base=100.0, num_trials=1,
        engine_factory=lambda model: layout._AsyncCampaignEngine(model, "rag", 100.0),
        extra_argv=("--top-k", "3", "--ir-index-dir", str(tmp_path / "ir")),
        extra_env={"CAGE_DISTRACTOR_DOCS": str(DISTRACTORS)},
    )
    (docs,) = made
    measured = {ex.context[0] for ex in layout._FakeLoader("squad_v2").load()}
    assert {d.text for d in docs} == measured and len(docs) == layout.N_QUERIES
    (wdir,) = _windows(root)
    record = _experiment(wdir)["ir_index"]
    assert record["corpus_source"] == "measured-set"
    assert record["corpus_example_count"] == layout.N_QUERIES
    assert record["corpus_sha1"] == corpus_doc_ids_sha1(docs)
