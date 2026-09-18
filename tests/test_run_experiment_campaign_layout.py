"""Task #116 / K-COV1 proof: the REAL run_experiment code path produces the
RESULTS_LAYOUT-v2 campaign tree the entire analysis chain consumes.

The walkthrough finding (K-COV1): campaign_layout.py was built + unit-tested
but had ZERO production callers — run_experiment.py wrote only the pilot tree.
These tests drive run_experiment.main() itself (importlib-loaded, the
test_integration_wiring pattern) with a stubbed engine adapter at the
setup_inference_engine seam and a stubbed dataset loader (no network, no GPU,
$0) for a tiny 2-dataset x 2-arm x 2-window campaign into a tmp root, then
assert END-TO-END:

(a) verify_results.py's v2 gate exits GREEN on the produced tree;
(b) organize_results.py builds index/cells_index.csv with the expected rows;
(c) run_campaign_analysis.py DESIGN-INPUT mode runs to completion on it
    (contrast #1, B1-vs-B2 — the two stub arms);
(d) every requests.jsonl row carries the #127 join triple
    (example_id / repeat_index / record_index) + the adapter honesty columns
    + the shared `ok` validity predicate;
(e) the §5 seal verifies — and a tampered byte FAILS the gate (negative arm);
(f) resume: re-invoking run_experiment on a half-written cell completes it
    without re-serving or duplicating a single row (metrics_json_valid
    semantics: a syntactically-invalid window metrics.json reads as
    incomplete and is reset + re-emitted).

Sealing goes through scripts/3_run/seal_campaign_run.py (the write-time-hash
journal cross-check, S0-15) — the same entry point run_full_sweep.sh's
campaign mode invokes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_4A = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_4A), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import organize_results as org  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
import verify_results as vr  # noqa: E402
from src.data.loader import CAGExample  # noqa: E402
from src.inference.engine import InferenceResponse  # noqa: E402


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_script(
    "run_experiment_campaign_it", REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"
)
sealer = _load_script(
    "seal_campaign_run_it", REPO_ROOT / "scripts" / "3_run" / "seal_campaign_run.py"
)

RUN_ID = "20260821-1200-a-qwen3-14b"
DATASETS = ("squad_v2", "hotpotqa")
#: (pilot baseline name, ttft base ms) — no_cache -> B1 (gold-fresh),
#: prefix_cache -> B2 (gold-reuse): contrast #1's registered pair.
ARMS = (("no_cache", 200.0), ("prefix_cache", 120.0))
ROW_KEY_OF = {
    "no_cache": "gold-fresh|none|none|single|vllm|qwen3-14b|F1",
    "prefix_cache": "gold-reuse|none|none|single|vllm|qwen3-14b|F1",
}
N_QUERIES = 8
N_TRIALS = 2


class StubEngine:
    """Canned streamed-response engine at the setup_inference_engine seam."""

    def __init__(self, model_name: str, baseline: str, ttft_base: float) -> None:
        self.model_name = model_name
        self.baseline = baseline
        self.ttft_base = ttft_base

    def is_ready(self) -> bool:
        return True

    def _respond(self, request: Any) -> InferenceResponse:
        rid = str(request.request_id or "")
        jitter = int(hashlib.sha1(rid.encode()).hexdigest(), 16) % 37
        ttft = self.ttft_base + float(jitter)
        response = InferenceResponse(
            request_id=rid,
            generated_text=f"stub answer for {rid}",
            ttft_ms=ttft,
            total_time_ms=ttft + 40.0,
            num_tokens=6,
            model_name=self.model_name,
            finish_reason="stop",
            error=None,
            prompt_tokens=120,
            cached_prompt_tokens=64 if self.baseline == "prefix_cache" else 0,
        )
        # ADR-0007 honesty/provenance attributes the adapters stamp.
        response.engine_id = f"stub-{self.baseline}"
        response.usage_telemetry_available = True
        response.cached_token_telemetry_available = True
        response.retries = 0
        response.reference_engine = False
        return response

    def generate(self, request: Any, stream: bool = False) -> InferenceResponse:
        return self._respond(request)

    def batch_generate(self, requests: list) -> list:
        return [self._respond(r) for r in requests]

    def shutdown(self) -> None:
        pass


class _FakeLoader:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset

    def load(self, max_examples: Optional[int] = None) -> list:
        n = N_QUERIES if max_examples is None else min(N_QUERIES, max_examples)
        return [
            CAGExample(
                id=f"{self.dataset}-q{i:03d}",
                question=f"What is fact {i} of {self.dataset}?",
                context=[f"The fact {i} of {self.dataset} is answer-{i}."],
                answer=f"answer-{i}",
                metadata={},
            )
            for i in range(n)
        ]


def _campaign_env(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("CAGE_CAMPAIGN_ROOT", str(root))
    monkeypatch.setenv("CAGE_PROVIDER", "test-local")
    monkeypatch.setenv("CAGE_HARDWARE", "stub-cpu x1")
    monkeypatch.setenv("CAGE_MODEL_SLUG", "qwen3-14b")
    monkeypatch.setenv("CAGE_ENGINE_VERSION", "0.0-stub")
    monkeypatch.setenv("CAGE_DATASET_MANIFESTS_SHA256", "0" * 64)
    monkeypatch.setenv("CAGE_CAMPAIGN_DATASETS", "squad_v2 hotpotqa")
    monkeypatch.setenv("CAGE_SKIP_QUALITY", "1")
    monkeypatch.setenv("CAGE_REQUEST_SETTLE_MS", "0")
    for var in (
        "CAGE_QUERY_MANIFEST",
        "CAGE_CELL_ARM",
        "CAGE_CELL_RETRIEVER",
        "CAGE_CELL_POLICY",
        "CAGE_CELL_FAMILY",
        "CAGE_CELL_TOPOLOGY",
        "CAGE_CELL_BUDGET_R",
        "CAGE_CELL_RATE_FRAC",
        "CAGE_CORPUS_PREFIX_BUDGET",
        "CAGE_CORPUS_RUNG",
        "CAGE_CELL_CORPUS_BUDGET",
        "CAGE_ORDER_BY_CONTEXT",
        "CAGE_MANIFEST_TRIAL",
        "VLLM_KV_CACHE_DTYPE",
    ):
        monkeypatch.delenv(var, raising=False)


def _run_cell(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    baseline: str,
    dataset: str,
    *,
    ttft_base: float,
    num_trials: int = N_TRIALS,
    extra_argv: tuple[str, ...] = (),
    loader_factory: Optional[Any] = None,
    engine_factory: Optional[Any] = None,
    extra_env: Optional[dict[str, str]] = None,
) -> None:
    """Drive the REAL runner main() for one cell with the stub seams patched.

    ``extra_argv`` / ``loader_factory`` / ``engine_factory`` are the ADR-0102
    cold-start test hooks; ``extra_env`` (ADR-0106) sets CAGE_* identity env
    AFTER the campaign env reset (defaults keep every pre-existing caller
    untouched).
    """
    _campaign_env(monkeypatch, root)
    for key, value in (extra_env or {}).items():
        monkeypatch.setenv(key, value)
    if engine_factory is None:
        engine_factory = lambda model: StubEngine(model, baseline, ttft_base)  # noqa: E731
    if loader_factory is None:
        loader_factory = _FakeLoader
    monkeypatch.setattr(
        runner,
        "setup_inference_engine",
        lambda model, cfg, *, backend, use_offline=False, strict=True: engine_factory(model),
    )
    monkeypatch.setattr(runner, "get_loader", lambda ds, split=None, seed=0: loader_factory(ds))
    monkeypatch.setattr(runner, "start_http_server", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_safe_get_json", lambda url, timeout=5: None)
    staging = root / ".staging" / "test" / baseline / dataset
    argv = [
        "run_experiment.py",
        "--baseline", baseline,
        "--baseline-label", baseline,
        "--model", "Qwen/Qwen3-14B",
        "--dataset", dataset,
        "--num-queries", str(N_QUERIES),
        "--num-trials", str(num_trials),
        "--seed", "7",
        "--max-tokens", "16",
        "--api-base", "http://127.0.0.1:9",
        "--reranker-model", "none",
        "--output-dir", str(staging),
        *extra_argv,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = runner.main()
    assert rc is None  # main() sys.exit(1)s on failure — reaching here means success


def _window_dirs(root: Path, baseline: str, dataset: str) -> list[Path]:
    cell = root / "cells" / ROW_KEY_OF[baseline]
    return sorted(cell.glob(f"window_{dataset}-*"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _file_hashes(paths: list[Path]) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


@pytest.fixture(scope="module")
def campaign_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2 arms x 2 datasets x 2 windows produced by the REAL runner, sealed via
    seal_campaign_run.py, organized via organize_results.py."""
    mp = pytest.MonkeyPatch()
    try:
        root = tmp_path_factory.mktemp("campaign") / "results" / "camp1" / "a" / RUN_ID
        for baseline, ttft_base in ARMS:
            for dataset in DATASETS:
                _run_cell(mp, root, baseline, dataset, ttft_base=ttft_base)
        assert sealer.main([str(root)]) == 0, "seal_campaign_run.py refused a clean tree"
        org.organize_run(root)
        return root
    finally:
        mp.undo()


# ---------------------------------------------------------------------------
# (a) verify_results v2 gate GREEN
# ---------------------------------------------------------------------------


def test_verify_results_v2_gate_green(campaign_tree: Path, tmp_path: Path) -> None:
    report = vr.verify_run(campaign_tree)
    fails = [f for f in report["findings"] if f["severity"] == "FAIL"]
    assert report["ok"], f"v2 gate FAILED on the runner-produced tree: {fails}"
    assert report["accounting"]["totals"]["n_windows"] == len(ARMS) * len(DATASETS) * N_TRIALS
    # The CLI gate agrees (exit 0), report written OUTSIDE the tree.
    rc = vr.main([str(campaign_tree), "--out", str(tmp_path / "verification")])
    assert rc == 0


def test_ledger_seal_present_and_verifies(campaign_tree: Path) -> None:
    from src.analysis.stats.ledger import verify_ledger

    ledger = campaign_tree / "ledger.json"
    assert ledger.is_file(), "seal_campaign_run.py did not write the §5 ledger"
    assert verify_ledger(ledger, campaign_tree) == []
    # The write-time journal exists and covers every sealed entry (S0-15).
    from src.orchestration.campaign_session import read_write_time_journal

    journal = read_write_time_journal(campaign_tree)
    sealed = json.loads(ledger.read_text(encoding="utf-8"))["entries"]
    missing = sorted(set(sealed) - set(journal))
    assert not missing, f"sealed artifacts absent from the write-time journal: {missing}"


# ---------------------------------------------------------------------------
# (b) organize_results index
# ---------------------------------------------------------------------------


def test_window_metrics_persist_stale_index_opt_in_false(campaign_tree: Path) -> None:
    # Backlog A6 / F6 (review 2026-09-17 defect 2): every window states whether
    # its retrieval served a stale (pre-prefix) index; False here, and
    # campaign_session refuses a True.
    for baseline, _ in ARMS:
        for dataset in DATASETS:
            for wdir in _window_dirs(campaign_tree, baseline, dataset):
                meta = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))
                assert meta["experiment"]["stale_index_opt_in"] is False


def test_organize_index_has_expected_rows(campaign_tree: Path) -> None:
    import pandas as pd

    index = pd.read_csv(campaign_tree / "index" / "cells_index.csv")
    assert len(index) == len(ARMS) * len(DATASETS) * N_TRIALS
    assert set(index["row_key"]) == set(ROW_KEY_OF.values())
    assert set(index["dataset"]) == set(DATASETS)
    assert set(index["baseline"]) == {"B1", "B2"}
    assert set(index["window"]) == {1, 2}
    # The per-window metrics.json sentinel is indexed as an auxiliary artifact.
    assert all("metrics.json" in a for a in index["artifacts"])


# ---------------------------------------------------------------------------
# (c) run_campaign_analysis DESIGN-INPUT mode
# ---------------------------------------------------------------------------


def test_campaign_analysis_design_input_completes(campaign_tree: Path) -> None:
    rc = rca.main([str(campaign_tree), "--contrasts", "1"])
    assert rc == 0, "DESIGN-INPUT analysis did not complete on the runner tree"
    analysis_root = campaign_tree / "analysis"
    stats_files = sorted(analysis_root.glob("*/stats.json"))
    assert stats_files, "no stats.json emitted"
    stats = json.loads(stats_files[-1].read_text(encoding="utf-8"))
    assert "DESIGN-INPUT" in stats["mode_stamp"]
    # The B1-vs-B2 contrast actually computed on per-query ttft_ms pairs.
    dumped = json.dumps(stats)
    assert "ttft_ms" in dumped


# ---------------------------------------------------------------------------
# (d) requests.jsonl rows: #127 join triple + honesty/provenance columns
# ---------------------------------------------------------------------------


def test_requests_rows_carry_join_triple_and_honesty_columns(campaign_tree: Path) -> None:
    checked = 0
    for baseline, _ in ARMS:
        for dataset in DATASETS:
            for wdir in _window_dirs(campaign_tree, baseline, dataset):
                rows = _read_jsonl(wdir / "requests.jsonl")
                assert len(rows) == N_QUERIES
                for row in rows:
                    for key in ("example_id", "repeat_index", "record_index"):
                        assert key in row, f"{wdir.name}: row lacks #127 key {key!r}"
                    assert row["ok"] is True  # shared validity predicate (#127)
                    for key in (
                        "engine_id",
                        "usage_telemetry_available",
                        "cached_token_telemetry_available",
                        "retries",
                        "reference_engine",
                    ):
                        assert key in row, f"{wdir.name}: honesty column {key!r} missing"
                    checked += 1
    assert checked == len(ARMS) * len(DATASETS) * N_TRIALS * N_QUERIES


def test_requests_and_evidence_reconcile_per_window(campaign_tree: Path) -> None:
    for baseline, _ in ARMS:
        for dataset in DATASETS:
            for wdir in _window_dirs(campaign_tree, baseline, dataset):
                n_req = len(_read_jsonl(wdir / "requests.jsonl"))
                n_ev = len(_read_jsonl(wdir / "qa_evidence.jsonl"))
                assert n_req == n_ev == N_QUERIES


def test_cell_json_windows_table_and_manifest_provenance(campaign_tree: Path) -> None:
    manifest = json.loads((campaign_tree / "manifest.json").read_text(encoding="utf-8"))
    for key in ("git_sha", "seed", "engine", "engine_version", "model", "run_id",
                "kv_cache_dtype", "datasets"):
        assert manifest.get(key) not in (None, ""), f"manifest {key} is null/absent"
    assert manifest["model"] == "qwen3-14b"
    assert manifest["datasets"] == list(DATASETS)
    for baseline, _ in ARMS:
        meta = json.loads(
            (campaign_tree / "cells" / ROW_KEY_OF[baseline] / "cell.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(meta["windows"]) == {
            f"{ds}-{t:02d}" for ds in DATASETS for t in range(1, N_TRIALS + 1)
        }
        for entry in meta["windows"].values():
            assert entry["t_end"] > entry["t_start"]


# ---------------------------------------------------------------------------
# (e) negative arm: a tampered byte FAILS the gate
# ---------------------------------------------------------------------------


def test_tampered_byte_fails_verify(campaign_tree: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "results" / "camp1" / "a" / RUN_ID
    shutil.copytree(campaign_tree, tampered)
    victim = next(iter(_window_dirs(tampered, "no_cache", "squad_v2"))) / "requests.jsonl"
    data = bytearray(victim.read_bytes())
    data[0] ^= 0x01
    victim.write_bytes(bytes(data))
    report = vr.verify_run(tampered)
    assert not report["ok"], "tampered tree still verified GREEN"
    assert any(
        f["check"] == "ledger" and "HASH-MISMATCH" in f["detail"]
        for f in report["findings"]
    ), report["findings"]


# ---------------------------------------------------------------------------
# (f) resume: a half-written cell completes without duplicating rows
# ---------------------------------------------------------------------------


def test_resume_completes_half_written_cell_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    # Half-written cell: only window 01 exists (as after a crash between trials).
    _run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0, num_trials=1)
    w1 = root / "cells" / ROW_KEY_OF["no_cache"] / "window_squad_v2-01"
    before = _file_hashes(sorted(p for p in w1.iterdir() if p.is_file()))

    # Re-invoke for the full 2 trials: window 01 must be SKIPPED byte-identically
    # (no re-serving), window 02 emitted, no duplicated rows anywhere.
    _run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0, num_trials=2)
    after = _file_hashes(sorted(p for p in w1.iterdir() if p.is_file()))
    assert after == before, "resume re-wrote the already-complete window 01"
    w2 = root / "cells" / ROW_KEY_OF["no_cache"] / "window_squad_v2-02"
    assert w2.is_dir(), "resume did not complete the missing window 02"
    for wdir in (w1, w2):
        assert len(_read_jsonl(wdir / "requests.jsonl")) == N_QUERIES
    meta = json.loads((w1.parent / "cell.json").read_text(encoding="utf-8"))
    assert set(meta["windows"]) == {"squad_v2-01", "squad_v2-02"}

    # metrics_json_valid semantics: a syntactically-INVALID window metrics.json
    # reads as incomplete — the window is reset and re-emitted, never trusted.
    (w2 / "metrics.json").write_text('{"truncated": ', encoding="utf-8")
    _run_cell(monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0, num_trials=2)
    assert json.loads((w2 / "metrics.json").read_text(encoding="utf-8"))  # parses again
    assert len(_read_jsonl(w2 / "requests.jsonl")) == N_QUERIES
    assert not (w1.parent / "window_squad_v2-03").exists(), (
        "reset window was appended as a NEW ordinal instead of re-emitted in place"
    )
    meta = json.loads((w1.parent / "cell.json").read_text(encoding="utf-8"))
    assert set(meta["windows"]) == {"squad_v2-01", "squad_v2-02"}

    # The recovered single-cell run seals + verifies green end-to-end.
    assert sealer.main([str(root)]) == 0
    report = vr.verify_run(root)
    assert report["ok"], [f for f in report["findings"] if f["severity"] == "FAIL"]


# ---------------------------------------------------------------------------
# (g) ADR-0102 (owner decision 2026-09-16): cold start per window with a
#     registered warm-up from a DISJOINT pool. A trial IS a window in campaign
#     mode; the warm-up must never touch a measured query (that would
#     cache-warm the measured set under a "cold" label) and its results are
#     discarded: never a results row, never a requests.jsonl row.
# ---------------------------------------------------------------------------

WARM_N = 4
POOL_TOTAL = 24  # measured 8 (N_QUERIES) + a disjoint pool of 16


class _PoolLoader(_FakeLoader):
    """A loader with MORE examples than the measured set (prefix-stable,
    like the HF shuffle(seed).select(range(k)) loaders)."""

    total = POOL_TOTAL

    def load(self, max_examples: Optional[int] = None) -> list:
        n = self.total if max_examples is None else min(self.total, max_examples)
        return [
            CAGExample(
                id=f"{self.dataset}-q{i:03d}",
                question=f"What is fact {i} of {self.dataset}?",
                context=[f"The fact {i} of {self.dataset} is answer-{i}."],
                answer=f"answer-{i}",
                metadata={},
            )
            for i in range(n)
        ]


class _ExactLoader(_PoolLoader):
    """Exactly the measured set: NO disjoint example exists."""

    total = N_QUERIES


class RecordingEngine(StubEngine):
    """StubEngine that records every request id it serves, in order."""

    served: list[str]

    def __init__(self, model_name: str, baseline: str, ttft_base: float) -> None:
        super().__init__(model_name, baseline, ttft_base)
        self.served = []

    def _respond(self, request: Any) -> InferenceResponse:
        self.served.append(str(request.request_id or ""))
        return super()._respond(request)


def _measured_ids(dataset: str) -> set[str]:
    return {f"{dataset}-q{i:03d}" for i in range(N_QUERIES)}


def _warm_ids(served: list[str]) -> list[str]:
    return [rid for rid in served if "__warmup_pool" in rid]


def _base_id(rid: str) -> str:
    return rid.split("__warmup_pool")[0]


def _run_cold_start_cell(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    dataset: str = "squad_v2",
    num_trials: int = N_TRIALS,
    warm_n: int = WARM_N,
    loader_factory: Any = _PoolLoader,
    extra: tuple[str, ...] = (),
) -> list[RecordingEngine]:
    engines: list[RecordingEngine] = []

    def _engine(model: str) -> RecordingEngine:
        eng = RecordingEngine(model, "no_cache", 200.0)
        engines.append(eng)
        return eng

    # The stub api_base is unreachable, so the real reset would refuse in
    # campaign mode; the reset itself has its own tests below.
    monkeypatch.setattr(runner, "_reset_prefix_cache", lambda *a, **k: None)
    _run_cell(
        monkeypatch,
        root,
        "no_cache",
        dataset,
        ttft_base=200.0,
        num_trials=num_trials,
        extra_argv=("--warmup-pool-queries", str(warm_n), *extra),
        loader_factory=loader_factory,
        engine_factory=_engine,
    )
    return engines


def test_warmup_pool_is_disjoint_and_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    engines = _run_cold_start_cell(monkeypatch, root)
    assert len(engines) == N_TRIALS  # one engine per trial (= per window)
    measured = _measured_ids("squad_v2")
    for eng in engines:
        warm = _warm_ids(eng.served)
        assert len(warm) == WARM_N, eng.served
        # Served BEFORE the measured requests (cold cache -> warm-up -> measure).
        assert eng.served[:WARM_N] == warm
        # DISJOINT from every measured id.
        assert not ({_base_id(r) for r in warm} & measured)
        assert len({_base_id(r) for r in warm}) == WARM_N
        # The measured set itself is untouched (still the first N_QUERIES).
        assert set(eng.served[WARM_N:]) == measured

    wdirs = _window_dirs(root, "no_cache", "squad_v2")
    assert len(wdirs) == N_TRIALS
    for ordinal, wdir in enumerate(wdirs, start=1):
        rows = _read_jsonl(wdir / "requests.jsonl")
        assert len(rows) == N_QUERIES
        assert {r["example_id"] for r in rows} == measured
        assert not any("__warmup_pool" in r["example_id"] for r in rows)
        if (wdir / "qa_evidence.jsonl").is_file():
            ev = _read_jsonl(wdir / "qa_evidence.jsonl")
            assert not any("__warmup_pool" in str(r.get("example_id")) for r in ev)
        # Window metadata: the summary line + the warmup_ids sha256, never rows.
        meta = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))
        wp = meta["warmup_pool"]
        assert wp["num_queries"] == WARM_N
        assert wp["num_requests"] == WARM_N
        assert wp["included_in_metrics"] is False
        assert wp["disjoint_from_measured"] is True
        assert wp["trial_ordinal"] == ordinal
        assert isinstance(wp["ids_sha256"], str) and len(wp["ids_sha256"]) == 64
        served_base = [_base_id(r) for r in _warm_ids(engines[ordinal - 1].served)]
        assert wp["ids_sha256"] == runner.warmup_pool_ids_sha256(served_base)
        # The legacy measured-set replay stays OFF.
        assert meta["warmup"]["num_queries"] == 0
        assert meta["experiment"]["num_warmup_requests"] == 0


def test_warmup_pool_draw_is_deterministic_and_trial_seeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_a = tmp_path / "a" / "results" / "camp1" / "a" / RUN_ID
    root_b = tmp_path / "b" / "results" / "camp1" / "a" / RUN_ID
    first = _run_cold_start_cell(monkeypatch, root_a)
    second = _run_cold_start_cell(monkeypatch, root_b)
    draws_first = [[_base_id(r) for r in _warm_ids(e.served)] for e in first]
    draws_second = [[_base_id(r) for r in _warm_ids(e.served)] for e in second]
    # Same --seed + same trial ordinal -> the same draw, run after run.
    assert draws_first == draws_second
    # Different trial ordinals -> different draws (seeded by seed AND ordinal).
    assert draws_first[0] != draws_first[1]
    # The pure draw function agrees with what the runner served. Without a
    # manifest the measured set is load(N_QUERIES) UNCHANGED and the runner
    # loads a separate candidate slate of N_QUERIES + WARM_N x the registered
    # multiplier, so the candidates are "the loaded examples beyond the
    # measured set" (over-provisioned for the context-disjointness filter).
    slate = _PoolLoader("squad_v2").load(
        max_examples=N_QUERIES + WARM_N * runner.WARMUP_POOL_CANDIDATE_MULTIPLIER
    )
    measured_examples = _PoolLoader("squad_v2").load(max_examples=N_QUERIES)
    measured = _measured_ids("squad_v2")
    candidates = [ex for ex in slate if ex.id not in measured]
    assert len(candidates) == WARM_N * runner.WARMUP_POOL_CANDIDATE_MULTIPLIER
    ctx = runner.warmup_context_keys(measured_examples)
    for trial in (1, 2):
        drawn = runner.draw_warmup_pool(
            candidates, measured_ids=measured, measured_context_keys=ctx,
            n=WARM_N, seed=7 + trial - 1, trial=trial,
        )
        assert [ex.id for ex in drawn] == draws_first[trial - 1]


def test_warmup_pool_refuses_when_no_disjoint_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    with pytest.raises(SystemExit) as exc:
        _run_cold_start_cell(monkeypatch, root, loader_factory=_ExactLoader)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "disjoint from the measured set" in out
    # Nothing was emitted: no window may exist on a refused cold-start.
    assert not _window_dirs(root, "no_cache", "squad_v2")


def test_draw_warmup_pool_typed_refusal_and_determinism() -> None:
    pool = _PoolLoader("squad_v2").load()
    measured = _measured_ids("squad_v2")
    ctx = runner.warmup_context_keys(pool[:N_QUERIES])
    candidates = [ex for ex in pool if ex.id not in measured]
    kw: dict[str, Any] = {"measured_ids": measured, "measured_context_keys": ctx}
    with pytest.raises(runner.WarmupPoolError, match="disjoint"):
        runner.draw_warmup_pool(candidates, n=len(candidates) + 1, seed=7, trial=1, **kw)
    # A measured id sneaking into the candidates is filtered, never served.
    drawn = runner.draw_warmup_pool(pool, n=WARM_N, seed=7, trial=1, **kw)
    assert not ({ex.id for ex in drawn} & measured)
    again = runner.draw_warmup_pool(pool, n=WARM_N, seed=7, trial=1, **kw)
    assert [ex.id for ex in drawn] == [ex.id for ex in again]
    other = runner.draw_warmup_pool(pool, n=WARM_N, seed=8, trial=1, **kw)
    assert [ex.id for ex in drawn] != [ex.id for ex in other]
    with pytest.raises(runner.WarmupPoolError):
        runner.draw_warmup_pool(candidates, n=-1, seed=7, trial=1, **kw)
    assert runner.draw_warmup_pool(candidates, n=0, seed=7, trial=1, **kw) == []


def test_warmup_pool_manifest_draws_outside_every_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With a query manifest the measured set is per-trial by id; the pool
    # must sit outside EVERY trial's ids (trial 2's measured queries are
    # off-limits for trial 1's warm-up too).
    dataset = "squad_v2"
    ids = [f"{dataset}-q{i:03d}" for i in range(POOL_TOTAL)]
    trial_ids = {"1": ids[:N_QUERIES], "2": ids[N_QUERIES:2 * N_QUERIES]}
    manifest = {
        "version": 2,
        "dataset": dataset,
        "trials": trial_ids,
        "blocks": [],
        "question_to_block": {},
        "stats": {"pool_size": POOL_TOTAL, "n_blocks": 0},
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    engines = _run_cold_start_cell(
        monkeypatch, root, extra=("--query-manifest", str(mpath))
    )
    every_trial = set(trial_ids["1"]) | set(trial_ids["2"])
    for ordinal, eng in enumerate(engines, start=1):
        warm = {_base_id(r) for r in _warm_ids(eng.served)}
        assert len(warm) == WARM_N
        assert not (warm & every_trial), (ordinal, warm)
        assert set(eng.served[WARM_N:]) == set(trial_ids[str(ordinal)])


def test_warmup_pool_refuses_legacy_measured_set_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --warmup-queries replays the MEASURED set (cache-warms the measured
    # queries): combining it with the disjoint-pool warm-up contradicts the
    # cold-start protocol and refuses.
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    with pytest.raises(SystemExit) as exc:
        _run_cold_start_cell(monkeypatch, root, extra=("--warmup-queries", "1"))
    assert exc.value.code == 1
    assert "warmup-queries" in capsys.readouterr().out


def test_campaign_mode_resets_before_every_window_and_refuses_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # (1) In campaign mode the reset runs before EVERY trial, trial 1
    # included (the server may still hold the previous cell's cache), and
    # in STRICT mode.
    calls: list[dict[str, Any]] = []

    def _fake_reset(api_base: str, *, backend: str = "vllm", model: str = "",
                    strict: bool = False) -> None:
        calls.append({"backend": backend, "strict": strict})

    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    monkeypatch.setattr(runner, "_reset_prefix_cache", _fake_reset)
    _run_cell(
        monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0,
        extra_argv=("--reset-cache-between-trials",),
    )
    assert len(calls) == N_TRIALS
    assert all(c["strict"] is True for c in calls)

    # (2) A failed reset is a REFUSAL in campaign mode: the real helper
    # against an unreachable server raises the typed error and main() exits
    # nonzero with NO window emitted (a window that starts warm when the
    # plan says cold would be a mislabeled row).
    monkeypatch.undo()
    root2 = tmp_path / "results" / "camp2" / "a" / RUN_ID
    with pytest.raises(SystemExit) as exc:
        _run_cell(
            monkeypatch, root2, "no_cache", "squad_v2", ttft_base=200.0,
            extra_argv=("--reset-cache-between-trials",),
        )
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "could not reset the vllm cache before this window" in out
    assert "ADR-0102" in out
    assert not _window_dirs(root2, "no_cache", "squad_v2")


def test_reset_prefix_cache_strict_raises_pilot_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(runner.CacheResetError, match="reset"):
        runner._reset_prefix_cache(
            "http://127.0.0.1:9", backend="vllm", model="stub", strict=True
        )
    # Pilot (non-campaign) path: the historical WARNING, never an abort.
    runner._reset_prefix_cache("http://127.0.0.1:9", backend="vllm", model="stub")
    assert "WARNING: could not reset prefix cache" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# (h) ADR-0102 repair (adversarial review 2026-09-16): the measured set must be
#     byte-identical with and without the warm-up pool on EVERY loader (QASPER
#     bounds PAPERS, not questions), and disjointness is by served CONTEXT as
#     well as by id (a warm-up query sharing a measured paragraph would
#     cache-warm the measured prefix under a cold label).
# ---------------------------------------------------------------------------

PAPERS_TOTAL = 12
QUESTIONS_PER_PAPER = 3


class _PaperLoader(_FakeLoader):
    """QASPER-style loader: ``max_examples`` bounds PAPERS (groups), each
    contributing QUESTIONS_PER_PAPER questions over the SAME full-text context
    (src/data/loader.py QasperLoader: "max_examples bounds PAPERS, each
    contributing all its questions")."""

    def load(self, max_examples: Optional[int] = None) -> list:
        n = PAPERS_TOTAL if max_examples is None else min(PAPERS_TOTAL, max_examples)
        out = []
        for p in range(n):
            ctx = [f"Paper {p} of {self.dataset}, section A.", f"Paper {p}, section B."]
            for q in range(QUESTIONS_PER_PAPER):
                out.append(
                    CAGExample(
                        id=f"{self.dataset}-p{p:02d}_q{q}",
                        question=f"Question {q} on paper {p}?",
                        context=ctx,
                        answer=f"answer-{p}-{q}",
                        metadata={"dataset": "qasper", "paper_id": f"p{p:02d}"},
                    )
                )
        return out


SHARED_CANDIDATES = 4  # candidates q008..q011 reuse the paragraphs of q000..q003


class _SharedParagraphLoader(_PoolLoader):
    """SQuAD-style one-to-one loader where several questions share ONE
    paragraph: the first SHARED_CANDIDATES examples beyond the measured set
    carry a measured example's paragraph (disjoint by id, NOT by context)."""

    def load(self, max_examples: Optional[int] = None) -> list:
        n = self.total if max_examples is None else min(self.total, max_examples)
        out = []
        for i in range(n):
            shared = N_QUERIES <= i < N_QUERIES + SHARED_CANDIDATES
            para = i - N_QUERIES if shared else i
            out.append(
                CAGExample(
                    id=f"{self.dataset}-q{i:03d}",
                    question=f"What is fact {i} of {self.dataset}?",
                    context=[f"The fact {para} of {self.dataset} is answer-{para}."],
                    answer=f"answer-{para}",
                    metadata={},
                )
            )
        return out


class _AllSharedLoader(_PoolLoader):
    """Every candidate beyond the measured set shares a measured paragraph:
    no context-disjoint warm-up exists (typed refusal expected)."""

    def load(self, max_examples: Optional[int] = None) -> list:
        n = self.total if max_examples is None else min(self.total, max_examples)
        return [
            CAGExample(
                id=f"{self.dataset}-q{i:03d}",
                question=f"What is fact {i} of {self.dataset}?",
                context=[f"The fact {i % N_QUERIES} of {self.dataset} is answer-{i % N_QUERIES}."],
                answer=f"answer-{i % N_QUERIES}",
                metadata={},
            )
            for i in range(n)
        ]


def _measured_ids_from_windows(root: Path, baseline: str, dataset: str) -> list[set[str]]:
    return [
        {r["example_id"] for r in _read_jsonl(wdir / "requests.jsonl")}
        for wdir in _window_dirs(root, baseline, dataset)
    ]


def test_warmup_pool_never_changes_the_measured_set_on_a_paper_bounded_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pre-ADR measured set on QASPER = ALL questions of the first N_QUERIES
    # papers (24 rows), NOT the first N_QUERIES questions of a larger load.
    # (The campaign roster of _campaign_env is squad_v2 + hotpotqa; the
    # paper-bounded stub is dataset-name agnostic, so it rides "hotpotqa".)
    ds = "hotpotqa"
    root_off = tmp_path / "off" / "results" / "camp1" / "a" / RUN_ID
    root_on = tmp_path / "on" / "results" / "camp1" / "a" / RUN_ID
    _run_cold_start_cell(monkeypatch, root_off, dataset=ds, warm_n=0,
                         loader_factory=_PaperLoader)
    engines = _run_cold_start_cell(monkeypatch, root_on, dataset=ds,
                                   loader_factory=_PaperLoader)
    expected = {
        f"{ds}-p{p:02d}_q{q}" for p in range(N_QUERIES) for q in range(QUESTIONS_PER_PAPER)
    }
    off = _measured_ids_from_windows(root_off, "no_cache", ds)
    on = _measured_ids_from_windows(root_on, "no_cache", ds)
    assert len(off) == len(on) == N_TRIALS
    for a, b in zip(off, on):
        assert a == b == expected
    # The warm-up draws from OTHER papers only (context-disjoint by paper).
    measured_papers = {f"p{p:02d}" for p in range(N_QUERIES)}
    for eng in engines:
        warm = [_base_id(r) for r in _warm_ids(eng.served)]
        assert len(warm) == WARM_N
        assert not ({w.split("-")[1].split("_")[0] for w in warm} & measured_papers), warm


def test_warmup_pool_excludes_candidates_sharing_a_measured_paragraph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    engines = _run_cold_start_cell(monkeypatch, root, loader_factory=_SharedParagraphLoader)
    shared_ids = {f"squad_v2-q{i:03d}" for i in range(N_QUERIES, N_QUERIES + SHARED_CANDIDATES)}
    for eng in engines:
        warm = {_base_id(r) for r in _warm_ids(eng.served)}
        assert len(warm) == WARM_N
        assert not (warm & shared_ids), warm
        assert not (warm & _measured_ids("squad_v2"))
    for wdir in _window_dirs(root, "no_cache", "squad_v2"):
        wp = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))["warmup_pool"]
        assert wp["disjoint_ids"] is True
        assert wp["disjoint_contexts"] is True
        assert wp["excluded_by_id"] == 0  # the slate is loaded beyond the measured set
        assert wp["excluded_by_context"] == SHARED_CANDIDATES
        assert wp["disjoint_from_measured"] is True


def test_warmup_pool_refuses_when_every_candidate_shares_a_measured_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    with pytest.raises(SystemExit) as exc:
        _run_cold_start_cell(monkeypatch, root, loader_factory=_AllSharedLoader)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "disjoint from the measured set" in out
    assert "context" in out
    assert not _window_dirs(root, "no_cache", "squad_v2")


def test_warmup_context_keys_fingerprint_paragraphs_and_paper_ids() -> None:
    examples = _PaperLoader("qasper").load(max_examples=2)
    keys = runner.warmup_context_keys(examples)
    # 2 papers x 2 distinct paragraphs + 2 paper ids.
    assert len(keys) == 6
    assert "paper:p00" in keys and "paper:p01" in keys
    plain = _PoolLoader("squad_v2").load(max_examples=3)
    assert len(runner.warmup_context_keys(plain)) == 3
    assert runner.warmup_context_keys([]) == set()


def test_draw_warmup_pool_refuses_context_overlap_and_partitions() -> None:
    pool = _SharedParagraphLoader("squad_v2").load()
    measured_examples = pool[:N_QUERIES]
    measured = {ex.id for ex in measured_examples}
    ctx = runner.warmup_context_keys(measured_examples)
    candidates = pool[N_QUERIES:]
    split = runner.partition_warmup_candidates(
        candidates, measured_ids=measured, measured_context_keys=ctx
    )
    assert split.excluded_by_id == 0
    assert split.excluded_by_context == SHARED_CANDIDATES
    assert len(split.disjoint) == len(candidates) - SHARED_CANDIDATES
    # A measured id sneaking into the candidates counts as an id exclusion.
    split2 = runner.partition_warmup_candidates(
        pool, measured_ids=measured, measured_context_keys=ctx
    )
    assert split2.excluded_by_id == N_QUERIES
    drawn = runner.draw_warmup_pool(
        candidates, measured_ids=measured, measured_context_keys=ctx,
        n=WARM_N, seed=7, trial=1,
    )
    assert not ({ex.id for ex in drawn} & {c.id for c in candidates[:SHARED_CANDIDATES]})
    with pytest.raises(runner.WarmupPoolError, match="context"):
        runner.draw_warmup_pool(
            candidates, measured_ids=measured, measured_context_keys=ctx,
            n=len(candidates) - SHARED_CANDIDATES + 1, seed=7, trial=1,
        )


# ---------------------------------------------------------------------------
# (i) ADR-0102 repair: the strict reset verifies the engine is QUIESCENT
#     (no in-flight requests holding KV blocks) before the flush, because
#     vLLM answers 200 to /reset_prefix_cache even when it declines to reset,
#     and records the verification in metrics.json['cold_start'].
# ---------------------------------------------------------------------------

import http.server  # noqa: E402
import threading  # noqa: E402


def test_parse_running_requests_sums_labeled_gauges() -> None:
    text = (
        "# HELP vllm:num_requests_running Number of requests currently running\n"
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{model_name="qwen"} 2.0\n'
        'vllm:num_requests_running{model_name="other"} 1\n'
        "vllm:num_requests_waiting 5\n"
    )
    assert runner.parse_running_requests(text, "vllm:num_requests_running") == 3
    assert runner.parse_running_requests(text, "sglang:num_running_reqs") is None
    assert runner.parse_running_requests("", "vllm:num_requests_running") is None


class _EngineStub(http.server.BaseHTTPRequestHandler):
    """A vLLM-shaped stub: /metrics reports a running-requests gauge that
    drains one request per probe; POST /reset_prefix_cache answers 200."""

    running: list[int] = [0]
    posts: list[str] = []
    drain: bool = True

    def log_message(self, *args: Any) -> None:  # silence
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        n = self.running[0]
        if self.drain and n > 0:
            self.running[0] = n - 1
        body = f'vllm:num_requests_running{{model_name="m"}} {n}\n'.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        self.posts.append(self.path)
        self.send_response(200)
        self.end_headers()


@pytest.fixture()
def engine_stub(monkeypatch: pytest.MonkeyPatch):
    _EngineStub.running = [0]
    _EngineStub.posts = []
    _EngineStub.drain = True
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EngineStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(runner, "COLD_START_QUIESCE_POLL_S", 0.01)
    monkeypatch.setattr(runner, "COLD_START_QUIESCE_TIMEOUT_S", 0.5)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_reset_prefix_cache_strict_waits_for_quiescence_and_records(engine_stub: str) -> None:
    _EngineStub.running = [2]  # two in-flight requests drain over two probes
    record = runner._reset_prefix_cache(engine_stub, backend="vllm", model="m", strict=True)
    assert _EngineStub.posts == ["/reset_prefix_cache"]
    assert record["backend"] == "vllm"
    assert record["endpoint"] == "/reset_prefix_cache"
    assert record["verified"] is True
    probe = record["quiescence_probe"]
    assert probe["gauge"] == "vllm:num_requests_running"
    assert probe["readable"] is True
    assert probe["running_at_first_probe"] == 2
    assert probe["running_before_flush"] == 0
    assert record["adr"] == "ADR-0102"


def test_reset_prefix_cache_strict_refuses_when_requests_never_drain(engine_stub: str) -> None:
    _EngineStub.running = [1]
    _EngineStub.drain = False
    with pytest.raises(runner.CacheResetError, match="in-flight"):
        runner._reset_prefix_cache(engine_stub, backend="vllm", model="m", strict=True)
    assert _EngineStub.posts == []  # never flushed a busy engine under a cold label


def test_reset_prefix_cache_pilot_path_does_not_probe(engine_stub: str) -> None:
    _EngineStub.running = [1]
    _EngineStub.drain = False
    record = runner._reset_prefix_cache(engine_stub, backend="vllm", model="m")
    assert _EngineStub.posts == ["/reset_prefix_cache"]
    assert record["verified"] is False
    assert record["quiescence_probe"] is None


def test_campaign_window_metrics_carry_the_cold_start_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = {"backend": "vllm", "endpoint": "/reset_prefix_cache", "verified": True,
            "adr": "ADR-0102"}
    monkeypatch.setattr(runner, "_reset_prefix_cache", lambda *a, **k: dict(fake))
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    _run_cell(
        monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0,
        extra_argv=("--reset-cache-between-trials",),
    )
    wdirs = _window_dirs(root, "no_cache", "squad_v2")
    assert len(wdirs) == N_TRIALS
    for wdir in wdirs:
        meta = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))
        assert meta["cold_start"] == fake


# ---------------------------------------------------------------------------
# ADR-0106 / backlog A4: B12 corpus-truncation rungs served from the manifest
# ---------------------------------------------------------------------------
#
# The runner serves EVERY measured query of a corpus-trunc cell against the
# rung block (out-of-corpus queries are served, expected abstention, never
# dropped), labels each row in_corpus / corpus_rung / corpus_tokens, and
# refuses (A4 guard) any budget the manifest does not carry.

from src.data.manifest import build_manifest  # noqa: E402

# words*4//3 under the fake contexts: header 7 words, each Document 9 words.
# 8 docs = 79 words -> 105 tokens (<= FULL 120: one block holds all 8);
# rung 60 keeps 4 docs (43 words -> 57) and drops 4.
TRUNC_FULL = 120
TRUNC_RUNG = 60
TRUNC_ENV = {
    "CAGE_CELL_ARM": "corpus-trunc",
    "CAGE_CELL_RETRIEVER": "none",
    "CAGE_CELL_CORPUS_BUDGET": str(TRUNC_RUNG),  # the identity rung (driver seam)
}
B3_ENV = {"CAGE_CELL_ARM": "corpus-reuse", "CAGE_CELL_RETRIEVER": "none"}


def _trunc_manifest(
    tmp_path: Path, dataset: str = "squad_v2", rungs: tuple[int, ...] = (TRUNC_RUNG,)
) -> tuple[Path, dict[str, Any]]:
    pool = _FakeLoader(dataset).load()
    manifest = build_manifest(
        pool, num_queries=N_QUERIES, num_trials=N_TRIALS, seed=7,
        block_budget=TRUNC_FULL, dataset=dataset, trunc_budgets=rungs,
    )
    path = tmp_path / "manifest_trunc.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def _run_corpus_cell(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    env: dict[str, str],
    extra: tuple[str, ...],
    dataset: str = "squad_v2",
) -> None:
    # The stub api_base is unreachable; the reset has its own tests above.
    monkeypatch.setattr(runner, "_reset_prefix_cache", lambda *a, **k: None)
    _run_cell(
        monkeypatch, root, "prefix_cache", dataset, ttft_base=120.0,
        extra_argv=extra, extra_env=env,
    )


def _window(root: Path, row_key: str, dataset: str, ordinal: int) -> Path:
    return root / "cells" / row_key / f"window_{dataset}-{ordinal:02d}"


def test_corpus_trunc_serves_every_query_and_labels_in_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mpath, manifest = _trunc_manifest(tmp_path)
    rung = manifest["trunc_rungs"][str(TRUNC_RUNG)]
    assert 0 < rung["n_in_corpus"] < N_QUERIES  # the fixture truly truncates
    rung_block = rung["blocks"][0]
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    _run_corpus_cell(
        monkeypatch, root, env=TRUNC_ENV,
        extra=(
            "--query-manifest", str(mpath),
            "--corpus-prefix-budget", str(TRUNC_RUNG),
            "--corpus-rung", str(TRUNC_RUNG),
        ),
    )
    row_key = f"corpus-trunc|none|none|single|vllm|qwen3-14b|F1|cb{TRUNC_RUNG}"
    in_ids = set(rung["in_corpus_ids"])
    for trial in (1, 2):
        wdir = _window(root, row_key, "squad_v2", trial)
        rows = _read_jsonl(wdir / "requests.jsonl")
        # Every measured id of the trial is served: nothing dropped.
        assert {r["example_id"] for r in rows} == set(manifest["trials"][str(trial)])
        assert len(rows) == N_QUERIES
        for r in rows:
            assert r["corpus_rung"] == TRUNC_RUNG
            assert r["corpus_tokens"] == rung_block["token_count"]
            assert r["in_corpus"] is (r["example_id"] in in_ids)
        assert sum(1 for r in rows if not r["in_corpus"]) == rung["n_out_of_corpus"]
        # The served context IS the rung block (a strict prefix of the full
        # block), for in- and out-of-corpus queries alike.
        evidence = _read_jsonl(wdir / "qa_evidence.jsonl")
        assert len(evidence) == N_QUERIES
        for e in evidence:
            assert e["used_contexts"] == [rung_block["text"]]
            assert e["corpus_rung"] == TRUNC_RUNG
            assert e["in_corpus"] is (e["example_id"] in in_ids)
        full_text = manifest["blocks"][0]["text"]
        assert full_text != rung_block["text"] and full_text.startswith(rung_block["text"])


def test_corpus_reuse_manifest_mode_labels_every_query_in_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # B3 at the manifest's full budget: in-corpus by construction, no rung.
    mpath, manifest = _trunc_manifest(tmp_path)
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    _run_corpus_cell(
        monkeypatch, root, env=B3_ENV,
        extra=("--query-manifest", str(mpath), "--corpus-prefix-budget", str(TRUNC_FULL)),
    )
    wdir = _window(root, "corpus-reuse|none|none|single|vllm|qwen3-14b|F1", "squad_v2", 1)
    rows = _read_jsonl(wdir / "requests.jsonl")
    assert len(rows) == N_QUERIES
    for r in rows:
        assert r["in_corpus"] is True
        assert r["corpus_rung"] is None
        assert r["corpus_tokens"] == manifest["blocks"][0]["token_count"]


def test_non_corpus_cells_carry_absent_corpus_labels(campaign_tree: Path) -> None:
    # Absence stays absence: gold arms carry null labels, never False/0.
    for baseline, _ in ARMS:
        for wdir in _window_dirs(campaign_tree, baseline, DATASETS[0]):
            for r in _read_jsonl(wdir / "requests.jsonl"):
                assert r["in_corpus"] is None
                assert r["corpus_rung"] is None
                assert r["corpus_tokens"] is None


# ---------------------------------------------------------------------------
# ADR-0114 (backlog A2): the loader's resolved answer_type + is_impossible
# reach BOTH per-row artifacts, so the three-clause Qasper predicate can key
# on them without re-reading the dataset.
# ---------------------------------------------------------------------------

#: The Qasper loader's answer_type vocabulary (src/data/loader.py
#: QasperLoader._resolve_annotator), cycled over the measured set.
ANSWER_TYPE_CYCLE = ("abstractive", "unanswerable", "yes_no", "extractive")


class _TypedLoader(_FakeLoader):
    """Examples carrying the Qasper loader's resolved labels in metadata."""

    def load(self, max_examples: Optional[int] = None) -> list:
        examples = super().load(max_examples)
        for i, ex in enumerate(examples):
            answer_type = ANSWER_TYPE_CYCLE[i % len(ANSWER_TYPE_CYCLE)]
            ex.metadata["answer_type"] = answer_type
            ex.metadata["is_impossible"] = answer_type == "unanswerable"
            if answer_type == "unanswerable":
                ex.answer = ""
            elif answer_type == "yes_no":
                ex.answer = "Yes"
        return examples


def test_rows_persist_loader_answer_type_and_is_impossible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    _run_cell(
        monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0,
        num_trials=1, loader_factory=_TypedLoader,
    )
    (wdir,) = _window_dirs(root, "no_cache", "squad_v2")
    expected = {
        f"squad_v2-q{i:03d}": ANSWER_TYPE_CYCLE[i % len(ANSWER_TYPE_CYCLE)]
        for i in range(N_QUERIES)
    }
    for name in ("requests.jsonl", "qa_evidence.jsonl"):
        rows = _read_jsonl(wdir / name)
        assert len(rows) == N_QUERIES
        for r in rows:
            assert "answer_type" in r and "is_impossible" in r, f"{name}: {r.keys()}"
            assert r["answer_type"] == expected[r["example_id"]]
            assert r["is_impossible"] is (r["answer_type"] == "unanswerable")
    # The unanswerable rows persisted the empty reference the loader emits.
    ev = _read_jsonl(wdir / "qa_evidence.jsonl")
    for r in ev:
        if r["answer_type"] == "unanswerable":
            assert r["reference_answer"] == ""


def test_rows_without_loader_labels_carry_null_answer_type(campaign_tree: Path) -> None:
    # Absence stays absence: a loader that emits neither label persists null
    # in both artifacts (never a guessed type, never False).
    for baseline, _ in ARMS:
        for wdir in _window_dirs(campaign_tree, baseline, DATASETS[0]):
            for name in ("requests.jsonl", "qa_evidence.jsonl"):
                for r in _read_jsonl(wdir / name):
                    assert "answer_type" in r and r["answer_type"] is None
                    assert "is_impossible" in r and r["is_impossible"] is None


def test_rows_persist_loader_labels_verbatim_without_coercion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The emission seam persists exactly what the loader resolved: an unknown
    # answer_type lands verbatim (never coerced to a known type) so the
    # predicate consumer refuses the mislabeled row loudly (ADR-0114) instead
    # of scoring a guess. The is_impossible flag is verified at the loader
    # boundary (backlog A8, below), so only a consistent bool reaches here.
    class _BadLabelLoader(_FakeLoader):
        def load(self, max_examples: Optional[int] = None) -> list:
            examples = super().load(max_examples)
            examples[0].metadata["is_impossible"] = False
            examples[0].metadata["answer_type"] = "boolean"
            return examples

    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    _run_cell(
        monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0,
        num_trials=1, loader_factory=_BadLabelLoader,
    )
    (wdir,) = _window_dirs(root, "no_cache", "squad_v2")
    for name in ("requests.jsonl", "qa_evidence.jsonl"):
        by_id = {r["example_id"]: r for r in _read_jsonl(wdir / name)}
        bad = by_id["squad_v2-q000"]
        assert bad["is_impossible"] is False and bad["answer_type"] == "boolean"
        assert by_id["squad_v2-q001"]["is_impossible"] is None


# ---------------------------------------------------------------------------
# Backlog A8 (review 2026-09-17 defect 1): the loader's is_impossible flag is
# verified against the gold at the LOADER BOUNDARY (before any engine work)
# and threaded into the scorer, so answerability_provenance is flag-verified
# on flagged rows and a mislabeled row never reaches serving.
# ---------------------------------------------------------------------------


class _RecordingEngine(StubEngine):
    calls: list = []

    def _respond(self, request: Any) -> InferenceResponse:
        _RecordingEngine.calls.append(str(request.request_id or ""))
        return super()._respond(request)


def _refusing_loader(mutate: Any) -> Any:
    class _Loader(_FakeLoader):
        def load(self, max_examples: Optional[int] = None) -> list:
            examples = super().load(max_examples)
            mutate(examples)
            return examples

    return _Loader


@pytest.mark.parametrize(
    "mutate, needle",
    [
        # Unanswerable gold (empty answer, empty all_answers) flagged answerable.
        (lambda exs: (exs[2].metadata.update({"is_impossible": False, "all_answers": []}),
                      setattr(exs[2], "answer", "")), "answerability mismatch"),
        # Answerable gold flagged unanswerable.
        (lambda exs: exs[0].metadata.update({"is_impossible": True}), "answerability mismatch"),
        # Non-bool stand-in: refused, never coerced.
        (lambda exs: exs[1].metadata.update({"is_impossible": "True"}), "must be a bool"),
    ],
    ids=["unanswerable-flagged-answerable", "answerable-flagged-unanswerable", "non-bool"],
)
def test_answerability_flag_mismatch_refuses_before_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    mutate: Any, needle: str,
) -> None:
    _RecordingEngine.calls = []
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    with pytest.raises(SystemExit) as exc:
        _run_cell(
            monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0, num_trials=1,
            loader_factory=_refusing_loader(mutate),
            engine_factory=lambda model: _RecordingEngine(model, "no_cache", 200.0),
        )
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert needle in out
    assert _RecordingEngine.calls == [], "a mislabeled row must refuse BEFORE serving"
    assert not _window_dirs(root, "no_cache", "squad_v2")


def test_verify_answerability_labels_is_typed_and_counts_flagged_rows() -> None:
    from src.data.loader import AnswerabilityFlagError
    from src.evaluation.quality import AnswerabilityMismatchError

    examples = _TypedLoader("squad_v2").load()
    assert runner.verify_answerability_labels(examples) == N_QUERIES
    assert runner.verify_answerability_labels(_FakeLoader("squad_v2").load()) == 0
    bad = _TypedLoader("squad_v2").load()
    bad[1].metadata["is_impossible"] = False  # the unanswerable row (cycle index 1)
    with pytest.raises(AnswerabilityMismatchError, match=bad[1].id):
        runner.verify_answerability_labels(bad)
    worse = _TypedLoader("squad_v2").load()
    worse[0].metadata["is_impossible"] = 1
    with pytest.raises(AnswerabilityFlagError):
        runner.verify_answerability_labels(worse)


def test_flagged_rows_score_flag_verified_and_unflagged_reference_derived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.evaluation.quality import (
        ANSWERABILITY_FLAG_VERIFIED, ANSWERABILITY_REFERENCE_DERIVED,
    )

    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    _run_cell(
        monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0,
        num_trials=1, loader_factory=_TypedLoader,
    )
    (wdir,) = _window_dirs(root, "no_cache", "squad_v2")
    rows = _read_jsonl(wdir / "requests.jsonl")
    assert rows and all(r["answerability_provenance"] == ANSWERABILITY_FLAG_VERIFIED for r in rows)
    # The unanswerable rows are scored as no-answer items (flag == empty gold).
    for r in rows:
        assert r["is_answerable"] == (0.0 if r["answer_type"] == "unanswerable" else 1.0)

    root2 = tmp_path / "results" / "camp2" / "a" / RUN_ID
    _run_cell(monkeypatch, root2, "no_cache", "squad_v2", ttft_base=200.0, num_trials=1)
    (wdir2,) = _window_dirs(root2, "no_cache", "squad_v2")
    rows2 = _read_jsonl(wdir2 / "requests.jsonl")
    assert rows2 and all(
        r["answerability_provenance"] == ANSWERABILITY_REFERENCE_DERIVED for r in rows2
    )


@pytest.mark.parametrize(
    "env, extra_flags, needle",
    [
        # A4: a rung the manifest's ladder does not carry.
        (TRUNC_ENV, ("--corpus-prefix-budget", "90", "--corpus-rung", "90"), "rung"),
        # The budget flag and the rung flag disagree.
        (TRUNC_ENV, ("--corpus-prefix-budget", str(TRUNC_FULL), "--corpus-rung", str(TRUNC_RUNG)), "--corpus-rung"),
        # A corpus-trunc cell without its explicit rung.
        (TRUNC_ENV, ("--corpus-prefix-budget", str(TRUNC_RUNG)), "--corpus-rung"),
        # A rung on a non-trunc arm.
        (B3_ENV, ("--corpus-prefix-budget", str(TRUNC_RUNG), "--corpus-rung", str(TRUNC_RUNG)), "corpus-trunc"),
        # A4 for B3/B4/B10: the served budget must equal the manifest's block budget.
        (B3_ENV, ("--corpus-prefix-budget", "100"), "block_budget"),
        # The identity rung (CAGE_CELL_CORPUS_BUDGET) and the served rung disagree.
        ({**TRUNC_ENV, "CAGE_CELL_CORPUS_BUDGET": "90"},
         ("--corpus-prefix-budget", str(TRUNC_RUNG), "--corpus-rung", str(TRUNC_RUNG)),
         "CAGE_CELL_CORPUS_BUDGET"),
        # A corpus-trunc identity without its rung coordinate.
        ({"CAGE_CELL_ARM": "corpus-trunc", "CAGE_CELL_RETRIEVER": "none"},
         ("--corpus-prefix-budget", str(TRUNC_RUNG), "--corpus-rung", str(TRUNC_RUNG)),
         "CAGE_CELL_CORPUS_BUDGET"),
    ],
)
def test_corpus_budget_guards_refuse_with_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    env: dict[str, str], extra_flags: tuple[str, ...], needle: str,
) -> None:
    mpath, _ = _trunc_manifest(tmp_path)
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    with pytest.raises(SystemExit) as exc:
        _run_corpus_cell(
            monkeypatch, root, env=env, extra=("--query-manifest", str(mpath), *extra_flags),
        )
    assert exc.value.code == 1
    assert needle in capsys.readouterr().out
    assert not (root / "cells").exists() or not any(
        p.name.startswith("window_") for p in (root / "cells").rglob("window_*")
    ), "a refused cell must never emit a window"


def test_corpus_trunc_refuses_the_non_manifest_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The pack-one-block fallback DROPS out-of-corpus queries: a pilot
    # convenience, never a registered path for the ladder arm.
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    with pytest.raises(SystemExit) as exc:
        _run_corpus_cell(
            monkeypatch, root, env=TRUNC_ENV,
            extra=("--corpus-prefix-budget", str(TRUNC_RUNG), "--corpus-rung", str(TRUNC_RUNG)),
        )
    assert exc.value.code == 1
    assert "manifest" in capsys.readouterr().out


def test_resolve_corpus_serving_unit_refusals() -> None:
    # The pure guard, exercised without the runner: every refusal is typed.
    manifest = {"block_budget": 120, "blocks": [{"block_id": 0, "text": "t", "token_count": 5}],
                "trunc_rungs": {"60": {"budget": 60, "blocks": [{"block_id": 0, "text": "u", "token_count": 3}],
                                       "in_corpus_ids": ["a"], "n_in_corpus": 1, "n_out_of_corpus": 1}}}
    def resolve(corpus_budget, corpus_rung, cell_arm, manifest, cell_corpus_budget="same"):
        if cell_corpus_budget == "same":
            cell_corpus_budget = corpus_rung if cell_arm is not None else None
        return runner.resolve_corpus_serving(
            corpus_budget=corpus_budget, corpus_rung=corpus_rung, cell_arm=cell_arm,
            cell_corpus_budget=cell_corpus_budget, manifest=manifest,
        )

    assert resolve(0, None, "gold-fresh", None) is None
    plan = resolve(60, 60, "corpus-trunc", manifest)
    assert plan.rung == 60 and plan.in_corpus_ids == frozenset({"a"})
    assert plan.blocks[0]["text"] == "u"
    # Pilot path (no identity env at all) may serve a rung without a coordinate.
    assert resolve(60, 60, None, manifest, cell_corpus_budget=None).rung == 60
    full = resolve(120, None, "corpus-reuse", manifest)
    assert full.rung is None and full.in_corpus_ids is None and full.blocks[0]["text"] == "t"
    fallback = resolve(120, None, None, None)
    assert fallback.blocks is None  # pilot fallback packs its own block
    for args in (
        (0, None, "corpus-trunc", manifest),          # trunc cell, no corpus mode at all
        (60, None, "corpus-trunc", manifest),         # trunc cell without its rung
        (60, 60, "corpus-trunc", None),               # rung without a manifest
        (60, 60, "corpus-reuse", manifest),           # rung on a non-trunc arm
        (120, 60, "corpus-trunc", manifest),          # budget != rung
        (90, 90, "corpus-trunc", manifest),           # rung absent from the ladder
        (100, None, "corpus-reuse", manifest),        # B3 budget != block_budget
        (100, None, None, {"blocks": []}),            # manifest without a block_budget
        (60, 0, "corpus-trunc", manifest),            # non-positive rung
    ):
        with pytest.raises(ValueError):
            resolve(*args)
    # Identity rung vs served rung: both must agree whenever an identity is present.
    with pytest.raises(ValueError, match="CAGE_CELL_CORPUS_BUDGET"):
        resolve(60, 60, "corpus-trunc", manifest, cell_corpus_budget=90)
    with pytest.raises(ValueError, match="CAGE_CELL_CORPUS_BUDGET"):
        resolve(60, 60, "corpus-trunc", manifest, cell_corpus_budget=None)
    with pytest.raises(ValueError, match="CAGE_CELL_CORPUS_BUDGET"):
        resolve(120, None, "corpus-reuse", manifest, cell_corpus_budget=60)


# ---------------------------------------------------------------------------
# Backlog A9 (--num-queries half, DECISION.md A1/A5 per-row N): with a query
# manifest, --num-queries selects the FIRST n ids of each trial in manifest
# order (nested prefix subsets: an 800-query cell is a tested prefix of the
# 2,000-query manifest); a trial with fewer than n ids refuses; the ADR-0102
# warm-up pool still draws outside EVERY trial's FULL id set, not the prefix.
# ---------------------------------------------------------------------------

MANIFEST_TRIAL_IDS = 16  # ids per manifest trial; the cell measures N_QUERIES = 8 of them
BIG_POOL_TOTAL = 48      # 2 trials x 16 ids + 16 candidates outside every trial


class _BigPoolLoader(_PoolLoader):
    total = BIG_POOL_TOTAL


def _prefix_manifest(tmp_path: Path, ids_per_trial: int, *, dataset: str = "squad_v2") -> tuple[Path, dict]:
    # Reversed id order inside each trial so "manifest order" is observable
    # (a sorted prefix would be indistinguishable from a manifest-order one).
    ids = [f"{dataset}-q{i:03d}" for i in range(BIG_POOL_TOTAL)]
    trial_ids = {
        "1": list(reversed(ids[:ids_per_trial])),
        "2": list(reversed(ids[MANIFEST_TRIAL_IDS:MANIFEST_TRIAL_IDS + ids_per_trial])),
    }
    manifest = {
        "manifest_version": 3,
        "dataset": dataset,
        "trials": trial_ids,
        "blocks": [],
        "question_to_block": {},
        "stats": {"pool_size": BIG_POOL_TOTAL, "n_blocks": 0},
    }
    mpath = tmp_path / "manifest_prefix.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return mpath, manifest


def test_manifest_num_queries_selects_the_trial_prefix_and_warmup_avoids_every_trial_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mpath, manifest = _prefix_manifest(tmp_path, MANIFEST_TRIAL_IDS)
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    engines = _run_cold_start_cell(
        monkeypatch, root, loader_factory=_BigPoolLoader,
        extra=("--query-manifest", str(mpath)),
    )
    every_trial = {i for ids in manifest["trials"].values() for i in ids}
    assert len(every_trial) == 2 * MANIFEST_TRIAL_IDS
    for ordinal, eng in enumerate(engines, start=1):
        trial = manifest["trials"][str(ordinal)]
        prefix, tail = trial[:N_QUERIES], trial[N_QUERIES:]
        # Measured = the FIRST N_QUERIES ids of the trial, in manifest order.
        measured_served = [r for r in eng.served if "__warmup_pool" not in r]
        assert measured_served == prefix, (ordinal, measured_served)
        # The un-measured tail of the trial is never served ...
        assert not (set(eng.served) & set(tail))
        # ... and the warm-up draws outside EVERY trial's FULL id set (the
        # tails of both trials included), never merely outside the prefix.
        warm = {_base_id(r) for r in _warm_ids(eng.served)}
        assert len(warm) == WARM_N
        assert not (warm & every_trial), (ordinal, warm)
    wdirs = _window_dirs(root, "no_cache", "squad_v2")
    assert len(wdirs) == N_TRIALS
    for ordinal, wdir in enumerate(wdirs, start=1):
        rows = _read_jsonl(wdir / "requests.jsonl")
        assert [r["example_id"] for r in rows] == manifest["trials"][str(ordinal)][:N_QUERIES]
        meta = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))
        assert meta["experiment"]["num_queries"] == N_QUERIES
        assert meta["experiment"]["query_manifest"]["trial_ids"] == MANIFEST_TRIAL_IDS
        assert meta["experiment"]["query_manifest"]["prefix"] == N_QUERIES


def test_manifest_trial_shorter_than_num_queries_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mpath, _ = _prefix_manifest(tmp_path, N_QUERIES - 2)  # 6 ids < --num-queries 8
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    with pytest.raises(SystemExit) as exc:
        _run_cold_start_cell(
            monkeypatch, root, loader_factory=_BigPoolLoader,
            extra=("--query-manifest", str(mpath)),
        )
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "--num-queries 8" in out and "trial 1" in out and "6 ids" in out
    assert not _window_dirs(root, "no_cache", "squad_v2")


def test_select_manifest_prefix_is_pure_ordered_and_fails_closed() -> None:
    pool = _BigPoolLoader("squad_v2").load()
    ids = [ex.id for ex in pool]
    manifest = {"trials": {"1": list(reversed(ids[:MANIFEST_TRIAL_IDS]))}}
    picked = runner.select_manifest_prefix(manifest, trial=1, examples=pool, num_queries=N_QUERIES)
    assert [ex.id for ex in picked] == manifest["trials"]["1"][:N_QUERIES]
    # Nested prefixes: a smaller n is a prefix of a larger n's selection.
    smaller = runner.select_manifest_prefix(manifest, trial=1, examples=pool, num_queries=3)
    assert [ex.id for ex in smaller] == [ex.id for ex in picked][:3]
    whole = runner.select_manifest_prefix(manifest, trial=1, examples=pool, num_queries=MANIFEST_TRIAL_IDS)
    assert [ex.id for ex in whole] == manifest["trials"]["1"]
    with pytest.raises(runner.QueryCountError, match="trial 1"):
        runner.select_manifest_prefix(manifest, trial=1, examples=pool, num_queries=MANIFEST_TRIAL_IDS + 1)
    with pytest.raises(runner.QueryCountError, match="num-queries"):
        runner.select_manifest_prefix(manifest, trial=1, examples=pool, num_queries=0)
    with pytest.raises(runner.QueryCountError, match="trial 2"):
        runner.select_manifest_prefix(manifest, trial=2, examples=pool, num_queries=1)


# ---------------------------------------------------------------------------
# Backlog A5 / F5a: the runner honors the driver's retrieval pins verbatim
# ---------------------------------------------------------------------------


def test_runner_cli_honors_the_driver_retrieval_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The campaign driver (scripts/3_run/run_campaign.py) pins --top-k,
    --embedding-model, --ir-index-dir, CAGE_DISTRACTOR_DOCS (A5) and, on B7,
    --redis-key-prefix + --flush-redis-namespace (F5a) on every retrieval
    cell. Those flags must exist on the runner CLI, reach run_baseline
    unchanged, and the runner's own defaults must equal the driver's
    registered values, so a cell that lost a pin can never drift silently
    to a different value than the plan header records."""
    driver = _load_script("run_campaign_a5_pin", REPO_ROOT / "scripts" / "3_run" / "run_campaign.py")
    src = (REPO_ROOT / "scripts" / "3_run" / "run_experiment.py").read_text(encoding="utf-8")
    for flag in (
        '"--top-k"', '"--embedding-model"', '"--embedding-revision"', '"--ir-index-dir"',
        '"--redis-key-prefix"', '"--flush-redis-namespace"',
    ):
        assert flag in src, flag
    for wiring in (
        "top_k=top_k_value",
        "embedding_model=embedding_model",
        "embedding_revision=args.embedding_revision",
        "ir_index_dir=args.ir_index_dir",
        "redis_key_prefix=args.redis_key_prefix",
        "flush_redis_namespace=args.flush_redis_namespace",
    ):
        assert wiring in src, wiring
    # The runner's defaults equal the registered driver values (a pin that
    # falls off a cell must not change the served behavior unnoticed; the
    # driver still refuses the stale plan at load time).
    assert f'os.getenv("CAGE_DISTRACTOR_DOCS", "{driver.DISTRACTOR_DOCS}")' in src
    assert f'default="{driver.DEFAULT_IR_INDEX_ROOT}"' in src
    # The freeze-slot id passes the runner's normalization unchanged.
    assert runner.normalize_embedding_model("intfloat/e5-large-v2") == "intfloat/e5-large-v2"

    # The parsed values, through the runner's REAL parser (no run): the
    # parser class is untouched (argparse's own __init__ resolves the module
    # global, so a class swap recurses); its parse_args is wrapped to capture
    # the namespace and stop right after parsing.
    captured: dict = {}
    parser_cls = runner.argparse.ArgumentParser
    real_parse_args = parser_cls.parse_args

    def _capture(self, *a, **k):
        ns = real_parse_args(self, *a, **k)
        captured["ns"] = ns
        raise SystemExit(0)

    monkeypatch.setattr(parser_cls, "parse_args", _capture)
    monkeypatch.setattr(sys, "argv", [
        "run_experiment.py", "--baseline", "hybrid", "--model", "m",
        "--dataset", "squad_v2",
        "--top-k", str(driver.RETRIEVAL_TOP_K),
        "--embedding-model", "intfloat/e5-large-v2",
        "--embedding-revision", "f169b11e22de13617baa190a028a32f3493550b6",
        "--ir-index-dir", driver.DEFAULT_IR_INDEX_ROOT,
        "--redis-key-prefix", driver.redis_key_prefix_for_row("k"),
        "--flush-redis-namespace",
    ])
    with pytest.raises(SystemExit):
        runner.main()
    ns = captured["ns"]
    assert ns.top_k == driver.RETRIEVAL_TOP_K == 3
    assert ns.embedding_model == "intfloat/e5-large-v2"
    assert ns.embedding_revision == "f169b11e22de13617baa190a028a32f3493550b6"
    assert ns.ir_index_dir == driver.DEFAULT_IR_INDEX_ROOT
    assert ns.redis_key_prefix == driver.redis_key_prefix_for_row("k")
    assert ns.flush_redis_namespace is True


def test_embedding_revision_pin_is_threaded_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 2026-09-17 defect 5 (backlog A5): --embedding-revision reaches
    ensure_ir_index (proven on the dense seam by tests/test_ir.py) and is
    persisted in metrics.json["experiment"]; absent stays null (a pilot run
    without the pin never records a revision it did not enforce)."""
    root = tmp_path / "results" / "camp1" / "a" / RUN_ID
    rev = "f169b11e22de13617baa190a028a32f3493550b6"
    _run_cell(
        monkeypatch, root, "no_cache", "squad_v2", ttft_base=200.0, num_trials=1,
        extra_argv=("--embedding-revision", rev),
    )
    (wdir,) = _window_dirs(root, "no_cache", "squad_v2")
    meta = json.loads((wdir / "metrics.json").read_text(encoding="utf-8"))
    assert meta["experiment"]["embedding_revision"] == rev

    root2 = tmp_path / "results" / "camp2" / "a" / RUN_ID
    _run_cell(monkeypatch, root2, "no_cache", "squad_v2", ttft_base=200.0, num_trials=1)
    (wdir2,) = _window_dirs(root2, "no_cache", "squad_v2")
    meta2 = json.loads((wdir2 / "metrics.json").read_text(encoding="utf-8"))
    assert "embedding_revision" in meta2["experiment"]
    assert meta2["experiment"]["embedding_revision"] is None
