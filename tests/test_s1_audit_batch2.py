"""S1 audit of 2026-10-17, batch 2 (owner: "Approved all with recommendations.
keep going lets close all open problems").

G4: the manifest builder sets aside whole context groups (QASPER papers, else
paragraphs) for the runner's warm-up and pre-load BEFORE the trial draws and
records them; the runner's candidates are exactly the recorded reserve; the
plan refuses a manifest that leaves fewer candidates than the registered
warm-up size, counting the reserve (exact) or the loaded remainder (an upper
bound). E6 (ADR-0173, option c): windows of the native-only datasets (RULER)
are scored by the model-free evaluator even under --full; every score row
carries its scoring_scope and the scoring manifest records the set. Each test
fails on the code before the change it names.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "3_run", REPO_ROOT / "scripts" / "4_analysis",
           Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rescore_quality as rq  # noqa: E402
import run_campaign as rc  # noqa: E402
from src.analysis.stats.ledger import hash_artifacts, write_ledger  # noqa: E402
from src.data.manifest import (  # noqa: E402
    MANIFEST_VERSION,
    ManifestError,
    build_manifest,
    draw_warmup_reserve,
)
from test_query_manifest import BUDGET, FakeExample, make_pool  # noqa: E402
from test_rescore_wiring import EVIDENCE_ROWS, _args, _fast_evaluator, _write_evidence  # noqa: E402
from test_run_campaign import (  # noqa: E402,F401  (fixtures ride the import)
    _rungs_for,
    _write_manifest,
    calibrations_a,
    floor_table,
)
from test_run_experiment_campaign_layout import (  # noqa: E402
    BIG_POOL_TOTAL,
    MANIFEST_TRIAL_IDS,
    RUN_ID,
    WARM_N,
    _base_id,
    _BigPoolLoader,
    _prefix_manifest,
    _run_cold_start_cell,
    _warm_ids,
)


# --------------------------------------------------------------------------- #
# G4: the manifest builder's warm-up reserve
# --------------------------------------------------------------------------- #


def _groups_of(pool: list[FakeExample]) -> dict[str, tuple]:
    return {ex.id: tuple(ex.context) for ex in pool}


def test_the_reserve_takes_whole_groups_outside_every_trial() -> None:
    pool = make_pool(8, 5)  # 40 questions in 8 paragraph groups of 5
    # pool_target packs the whole slate (the default max(3N, N T) would stop at 15)
    m = build_manifest(pool, num_queries=5, num_trials=3, seed=1, block_budget=BUDGET, warmup_reserve=7,
                       pool_target=1000)
    assert m["manifest_version"] == MANIFEST_VERSION == 4
    assert m["stats"]["pool_size"] == 40
    reserve = m["warmup_reserve"]
    reserved = set(reserve["ids"])
    assert reserve["requested"] == 7 and reserve["n"] == len(reserved) >= 7
    assert reserve["n"] == 10 and reserve["n_groups"] == 2  # whole groups of 5: the next whole group past 7
    assert reserve["group_key"] == "context" and reserve["seed"] == 1 + 3
    group = _groups_of(pool)
    # whole groups: every question sharing a reserved paragraph is reserved too
    reserved_groups = {group[i] for i in reserved}
    assert all(group[ex.id] not in reserved_groups for ex in pool if ex.id not in reserved)
    # no reserved id in any trial; the trials come from the rest
    every_trial = {i for ids in m["trials"].values() for i in ids}
    assert not (reserved & every_trial)
    assert m["stats"]["warmup_reserve_n"] == 10
    assert m["stats"]["trial_pool_size"] == m["stats"]["pool_size"] - 10
    assert m["stats"]["trials_disjoint"] is True  # 30 remain for 3 x 5


def test_the_reserve_groups_by_paper_id_when_the_loader_records_one() -> None:
    pool: list[FakeExample] = []
    for p in range(6):
        for q in range(3):
            # distinct contexts inside a paper: only paper_id can tie them together
            pool.append(FakeExample(id=f"p{p}_q{q}", question="?", context=[f"paper {p} para {q} " + "w " * 40],
                                    metadata={"paper_id": f"paper-{p}"}))
    m = build_manifest(pool, num_queries=4, num_trials=2, seed=3, block_budget=BUDGET, warmup_reserve=4,
                       pool_target=1000)
    assert m["stats"]["pool_size"] == 18
    reserve = m["warmup_reserve"]
    assert reserve["group_key"] == "paper_id"
    papers = {i.split("_")[0] for i in reserve["ids"]}
    assert reserve["n"] == 3 * len(papers) == 6 and reserve["n_groups"] == 2  # two whole papers
    every_trial = {i for ids in m["trials"].values() for i in ids}
    assert not ({i.split("_")[0] for i in every_trial} & papers)


def test_a_zero_reserve_reproduces_the_v3_trial_draws() -> None:
    pool = make_pool(8, 5)
    plain = build_manifest(pool, num_queries=5, num_trials=3, seed=1, block_budget=BUDGET)
    zero = build_manifest(pool, num_queries=5, num_trials=3, seed=1, block_budget=BUDGET, warmup_reserve=0)
    assert plain["trials"] == zero["trials"]
    assert plain["warmup_reserve"]["ids"] == [] and plain["warmup_reserve"]["n"] == 0
    assert plain["stats"]["warmup_reserve_n"] == 0 and plain["stats"]["trial_pool_size"] == plain["stats"]["pool_size"]


def test_the_reserve_is_deterministic_and_independent_of_num_queries() -> None:
    pool = make_pool(8, 5)
    a = build_manifest(pool, num_queries=5, num_trials=3, seed=1, block_budget=BUDGET, warmup_reserve=7,
                       pool_target=1000)
    b = build_manifest(pool, num_queries=9, num_trials=3, seed=1, block_budget=BUDGET, warmup_reserve=7,
                       pool_target=1000)
    assert a["warmup_reserve"] == b["warmup_reserve"]  # the two-run build recipe relies on this
    assert a["trials"] == build_manifest(pool, num_queries=5, num_trials=3, seed=1, block_budget=BUDGET,
                                         warmup_reserve=7, pool_target=1000)["trials"]


def test_reserve_refusals_name_the_achievable_n() -> None:
    pool = make_pool(8, 5)
    with pytest.raises(ManifestError, match="achievable n at this reserve is 30"):
        build_manifest(pool, num_queries=31, num_trials=1, seed=1, block_budget=BUDGET, warmup_reserve=7,
                       pool_target=1000)
    with pytest.raises(ManifestError, match="exceeds the in-corpus pool"):
        draw_warmup_reserve(pool, [ex.id for ex in pool], 41, seed=0)
    for bad in (-1, True, 2.5):
        with pytest.raises(ManifestError, match="int >= 0"):
            draw_warmup_reserve(pool, [ex.id for ex in pool], bad, seed=0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# G4: the plan counts the candidates and refuses a short manifest
# --------------------------------------------------------------------------- #


def test_the_plan_records_the_candidate_count_by_source(tmp_path: Path) -> None:
    grid = rc.SESSION_GRIDS["a"]
    path = _write_manifest(tmp_path, "qasper", name="plain.json")
    rec = rc._register_query_manifests(grid, {"qasper": path})["qasper"]
    assert rec["warmup_candidates"] == {"n": None, "source": "unknown"}
    doc = json.loads(path.read_text(encoding="utf-8"))
    every_trial = {i for ids in doc["trials"].values() for i in ids}
    doc["stats"] = {"source_examples_loaded": len(every_trial) + 705}
    path.write_text(json.dumps(doc), encoding="utf-8")
    rec = rc._register_query_manifests(grid, {"qasper": path})["qasper"]
    assert rec["warmup_candidates"] == {"n": 705, "source": "loaded_remainder"}
    doc["warmup_reserve"] = {"ids": [f"r{i}" for i in range(25)] + [sorted(every_trial)[0]], "n": 26}
    path.write_text(json.dumps(doc), encoding="utf-8")
    rec = rc._register_query_manifests(grid, {"qasper": path})["qasper"]
    assert rec["warmup_candidates"] == {"n": 25, "source": "reserve"}  # a trial id never counts


def test_the_plan_refuses_a_manifest_short_of_warm_up_candidates(
    floor_table: Path, calibrations_a: dict[str, Path], tmp_path: Path
) -> None:
    floor = rc.load_floor_table(floor_table)
    path = _write_manifest(tmp_path, "qasper")
    doc = json.loads(path.read_text(encoding="utf-8"))
    every_trial = {i for ids in doc["trials"].values() for i in ids}
    doc["stats"] = {"source_examples_loaded": len(every_trial) + rc.WARMUP_POOL_QUERIES - 1}
    path.write_text(json.dumps(doc), encoding="utf-8")

    def _plan() -> dict[str, Any]:
        return rc.build_plan(
            "a", floor, window_duration_s=300.0, query_manifests={"qasper": path},
            calibrations=calibrations_a, rung_calibrations=_rungs_for(calibrations_a, "a"),
        )

    with pytest.raises(rc.PlanError, match=r"leaves 19 warm-up candidate\(s\).*loaded_remainder.*--warmup-reserve"):
        _plan()
    doc["warmup_reserve"] = {"ids": [f"r{i}" for i in range(rc.WARMUP_POOL_QUERIES)], "n": rc.WARMUP_POOL_QUERIES}
    path.write_text(json.dumps(doc), encoding="utf-8")
    plan = _plan()
    assert plan["query_manifests"]["qasper"]["warmup_candidates"] == {"n": 20, "source": "reserve"}
    assert plan["warmup_candidates_caveat"] is None  # every registered manifest judged exactly


def test_the_header_names_the_manifests_judged_inexactly() -> None:
    # review 2026-10-17 batch 2 MEDIUM 1: a loaded_remainder or unknown count is a gap the review must see
    assert rc._warmup_candidates_caveat({}) is None
    assert rc._warmup_candidates_caveat({"qasper": {"warmup_candidates": {"n": 53, "source": "reserve"}}}) is None
    text = rc._warmup_candidates_caveat({
        "qasper": {"warmup_candidates": {"n": 53, "source": "reserve"}},
        "squad_v2": {"warmup_candidates": {"n": 700, "source": "loaded_remainder"}},
        "musique": {"warmup_candidates": {"n": None, "source": "unknown"}},
    })
    assert text is not None and "musique (unknown)" in text and "squad_v2 (loaded_remainder)" in text and "qasper" not in text


# --------------------------------------------------------------------------- #
# G4: the runner draws its warm-ups from the recorded reserve
# --------------------------------------------------------------------------- #


def test_the_runner_warms_up_from_the_recorded_reserve_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mpath, manifest = _prefix_manifest(tmp_path, MANIFEST_TRIAL_IDS)
    # 16 loaded ids sit outside both trials; the reserve names WARM_N of them
    ids = [f"squad_v2-q{i:03d}" for i in range(BIG_POOL_TOTAL)]
    reserve_ids = ids[2 * MANIFEST_TRIAL_IDS + 3: 2 * MANIFEST_TRIAL_IDS + 3 + WARM_N]
    manifest["manifest_version"] = 4
    manifest["warmup_reserve"] = {"requested": WARM_N, "ids": reserve_ids, "n": WARM_N}
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    engines = _run_cold_start_cell(
        monkeypatch, root, loader_factory=_BigPoolLoader, extra=("--query-manifest", str(mpath)),
    )
    assert len(engines) >= 1
    for eng in engines:
        warm = {_base_id(r) for r in _warm_ids(eng.served)}
        assert warm == set(reserve_ids), (warm, reserve_ids)


# --------------------------------------------------------------------------- #
# E6 (ADR-0173 c): RULER windows scored natively, recorded on every row
# --------------------------------------------------------------------------- #


def test_the_scoring_scope_follows_the_window_dataset() -> None:
    assert rq._native_only_set(_args()) == frozenset({"ruler"})
    assert rq._native_only_set(_args(native_only_datasets="")) == frozenset()
    assert rq._native_only_set(_args(native_only_datasets="ruler, squad_v2")) == frozenset({"ruler", "squad_v2"})
    ruler = Path("run/cells/B3/window_ruler-02/qa_evidence.jsonl")
    squad = Path("run/cells/B3/window_squad_v2-01/qa_evidence.jsonl")
    legacy = Path("run/trial_1/qa_evidence.jsonl")
    native = frozenset({"ruler"})
    assert rq._window_dataset(ruler) == "ruler" and rq._window_dataset(legacy) is None
    assert rq._scoring_scope(ruler, _args(full=True), native) == "native_only"
    assert rq._scoring_scope(squad, _args(full=True), native) == "full"
    assert rq._scoring_scope(legacy, _args(full=True), native) == "full"
    assert rq._scoring_scope(squad, _args(), native) == "fast"
    assert rq._scoring_scope(ruler, _args(full=True), frozenset()) == "full"
    ev = rq._build_native_evaluator(_args(full=True, device="cpu"))
    assert not any((ev.use_nli, ev.use_embeddings, ev.use_bertscore, ev.use_rouge, ev.use_lettucedetect))


def test_the_tree_pass_scores_ruler_windows_natively_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / RUN_ID
    (run / "cells").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": RUN_ID}), encoding="utf-8")
    _write_evidence(run / "cells" / "B3" / "window_squad_v2-01" / "qa_evidence.jsonl")
    _write_evidence(run / "cells" / "B3" / "window_ruler-01" / "qa_evidence.jsonl")
    sealed = [p for p in sorted(run.rglob("*")) if p.is_file()]
    write_ledger(hash_artifacts(sealed, base_dir=run), run / "ledger.json")

    built: list[str] = []

    def _native(args: Any) -> Any:
        built.append("native")
        return _fast_evaluator()

    monkeypatch.setattr(rq, "_build_native_evaluator", _native)
    assert rq.run_scoring_tree(run, "s01-fast", _args()) == 0
    assert built == ["native"]  # built once, on the first RULER window
    out = run / "scoring" / "s01-fast" / "cells" / "B3"
    ruler_rows = [json.loads(l) for l in (out / "window_ruler-01" / "qa_scores.jsonl").read_text().splitlines()]
    squad_rows = [json.loads(l) for l in (out / "window_squad_v2-01" / "qa_scores.jsonl").read_text().splitlines()]
    assert len(ruler_rows) == len(squad_rows) == len(EVIDENCE_ROWS)
    assert {r["scoring_scope"] for r in ruler_rows} == {"native_only"}
    assert {r["scoring_scope"] for r in squad_rows} == {"fast"}
    assert all(r["grounding_score"] is None for r in ruler_rows)
    man = json.loads((run / "scoring" / "s01-fast" / "scoring_manifest.json").read_text(encoding="utf-8"))
    assert man["native_only_datasets"] == ["ruler"] and man["n_windows_native_only"] == 1
    assert man["native_only_scope"] == "native_only"


def test_a_direct_scorer_call_without_a_scope_is_labeled_unlabeled(tmp_path: Path) -> None:
    # review 2026-10-17 batch 2 LOW 2: no default of "full" on a model-free call
    ev = tmp_path / "window_squad_v2-01" / "qa_evidence.jsonl"
    _write_evidence(ev)
    rows, _ = rq._score_evidence_file(ev, _fast_evaluator())
    assert {r["scoring_scope"] for r in rows} == {"unlabeled"}
    rows, _ = rq._score_evidence_file(ev, _fast_evaluator(), scoring_scope="fast")
    assert {r["scoring_scope"] for r in rows} == {"fast"}
