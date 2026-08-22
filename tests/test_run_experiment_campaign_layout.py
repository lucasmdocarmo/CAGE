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
) -> None:
    """Drive the REAL runner main() for one cell with the stub seams patched."""
    _campaign_env(monkeypatch, root)
    monkeypatch.setattr(
        runner,
        "setup_inference_engine",
        lambda model, cfg, *, backend, use_offline=False, strict=True: StubEngine(
            model, baseline, ttft_base
        ),
    )
    monkeypatch.setattr(runner, "get_loader", lambda ds, split=None, seed=0: _FakeLoader(ds))
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
