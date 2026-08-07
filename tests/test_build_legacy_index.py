"""Tests for scripts/4_analysis/build_legacy_index.py (the pilot legacy bridge).

The bridge re-keys a legacy pilot tree (results/phase2 layout) into the v2
``index/cells_index.csv`` handoff. These tests build a tiny synthetic legacy
archive under tmp_path (the REAL archive is read-only and never touched by
tests) and verify:

- exact organize_results.INDEX_COLUMNS column contract (driver-consumable);
- from_legacy mapping + baseline attribution + merge-on-row-key semantics;
- the canonical validity rule (error / empty_generation / rep>0 rows dropped,
  counted in the provenance map);
- unmappable cells land in the skipped list with the from_legacy reason;
- the S2 policy_event mask derivation from ``compression_applied``;
- read-only guarantees (archive untouched; out_dir inside archive refused);
- dataset inference fails closed on an un-suffixed run name;
- run_campaign_analysis.load_index consumes the bridge unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_legacy_index as bli  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402
from organize_results import INDEX_COLUMNS  # noqa: E402

RUN_SQUAD = "pilot_run_squad_v2"
RUN_HOTPOT = "pilot_run_hotpotqa"


def _write_results_csv(
    trial_dir: Path,
    rows: list[dict[str, object]],
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "example_id", "ttft_ms", "latency_ms", "tpot_ms", "f1_score",
        "exact_match", "grounding_score", "error", "empty_generation",
        "repeat_index", "compression_applied",
    ]
    df = pd.DataFrame(rows, columns=columns).fillna("")
    df.to_csv(trial_dir / "results.csv", index=False)


def _row(
    example_id: str,
    ttft: float,
    f1: float = 0.5,
    *,
    error: str = "",
    empty: str = "False",
    rep: str = "0",
    compression: str = "",
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "ttft_ms": ttft,
        "latency_ms": ttft + 100.0,
        "tpot_ms": 10.0,
        "f1_score": f1,
        "exact_match": 1.0 if f1 >= 0.99 else 0.0,
        "grounding_score": 0.8,
        "error": error,
        "empty_generation": empty,
        "repeat_index": rep,
        "compression_applied": compression,
    }


def _default_rows(n: int = 6, base_ttft: float = 100.0) -> list[dict[str, object]]:
    return [_row(f"ex{i}", base_ttft + i) for i in range(n)]


@pytest.fixture()
def archive(tmp_path: Path) -> Path:
    """Synthetic legacy archive: two runs, mappable + unmappable cells."""
    root = tmp_path / "archive"
    for run, base in ((RUN_SQUAD, 100.0), (RUN_HOTPOT, 200.0)):
        run_dir = root / run
        for cell, tree in (
            ("no_cache", "baselines"),        # -> B1 gold-fresh
            ("rag", "baselines"),             # -> B6 retr-fresh|rerank
            ("cag_true_on", "envelope"),      # -> B3 corpus-reuse
            ("prefix_cache", "baselines"),    # -> B2 gold-reuse
            ("cag_full", "compression"),      # -> B2 too (merge on row_key)
        ):
            for trial in (1, 2):
                _write_results_csv(
                    run_dir / tree / cell / f"trial_{trial}", _default_rows(6, base)
                )
        # unmappable name (not in LEGACY_ALIASES)
        _write_results_csv(
            run_dir / "baselines" / "mystery_cell" / "trial_1", _default_rows(3, base)
        )
    return root


def _build(archive: Path, tmp_path: Path, runs: list[str] | None = None) -> bli.BridgeResult:
    return bli.build_bridge(
        archive,
        runs or [RUN_SQUAD, RUN_HOTPOT],
        tmp_path / "bridge",
    )


# ---------------------------------------------------------------------------
# Column contract + driver hand-off
# ---------------------------------------------------------------------------


class TestIndexContract:
    def test_columns_match_organize_results_exactly(
        self, archive: Path, tmp_path: Path
    ) -> None:
        result = _build(archive, tmp_path)
        index = pd.read_csv(result.index_csv)
        assert list(index.columns) == list(INDEX_COLUMNS)

    def test_driver_load_index_consumes_bridge_unchanged(
        self, archive: Path, tmp_path: Path
    ) -> None:
        result = _build(archive, tmp_path)
        index = rca.load_index(result.out_dir)  # raises on contract violation
        assert not index.empty
        required = set(rca._INDEX_REQUIRED_COLUMNS)  # noqa: SLF001
        assert required <= set(index.columns)

    def test_window_rows_and_dataset_axis(self, archive: Path, tmp_path: Path) -> None:
        result = _build(archive, tmp_path)
        index = pd.read_csv(result.index_csv)
        # 5 mappable cells/run x 2 trials, prefix_cache+cag_full merge into one
        # row_key but keep separate windows: 10 windows per run.
        assert len(index) == 20
        assert set(index["dataset"].unique()) == {"squad_v2", "hotpotqa"}
        assert set(index["family"].unique()) == {"F1"}
        assert (index["model"] == "qwen3-14b").all()
        assert (index["engine"] == "vllm").all()

    def test_baseline_attribution(self, archive: Path, tmp_path: Path) -> None:
        result = _build(archive, tmp_path)
        index = pd.read_csv(result.index_csv)
        by_arm = index.groupby("arm")["baseline"].unique().to_dict()
        assert list(by_arm["gold-fresh"]) == ["B1"]
        assert list(by_arm["gold-reuse"]) == ["B2"]
        assert list(by_arm["corpus-reuse"]) == ["B3"]
        assert list(by_arm["retr-fresh"]) == ["B6"]


# ---------------------------------------------------------------------------
# Mapping semantics
# ---------------------------------------------------------------------------


class TestMapping:
    def test_same_rowkey_cells_merge_with_sequential_windows(
        self, archive: Path, tmp_path: Path
    ) -> None:
        result = _build(archive, tmp_path)
        index = pd.read_csv(result.index_csv)
        gold_reuse = index[
            (index["arm"] == "gold-reuse") & (index["dataset"] == "squad_v2")
        ]
        # prefix_cache (2 trials) + cag_full (2 trials) -> 4 windows, ordinals 1-4.
        assert sorted(gold_reuse["window"]) == [1, 2, 3, 4]
        assert gold_reuse["row_key"].nunique() == 1
        provenance = pd.read_csv(result.out_dir / "index" / "legacy_windows_map.csv")
        merged = provenance[
            (provenance["row_key"] == gold_reuse["row_key"].iloc[0])
            & (provenance["dataset"] == "squad_v2")
        ]
        assert set(merged["legacy_cell"]) == {"prefix_cache", "cag_full"}

    def test_cell_json_records_spec_and_provenance(
        self, archive: Path, tmp_path: Path
    ) -> None:
        result = _build(archive, tmp_path)
        index = pd.read_csv(result.index_csv)
        cell_json = result.out_dir / index["cell_json"].iloc[0]
        payload = json.loads(cell_json.read_text(encoding="utf-8"))
        assert payload["stamp"] == bli.STAMP
        assert payload["cellspec"]["model"] == "qwen3-14b"
        assert payload["legacy_sources"]
        assert "qwen3-8b" in payload["actual_pilot_model"]

    def test_unmappable_cell_is_skipped_with_reason(
        self, archive: Path, tmp_path: Path
    ) -> None:
        result = _build(archive, tmp_path)
        skipped = pd.read_csv(result.out_dir / "index" / "skipped_cells.csv")
        mystery = skipped[skipped["legacy_cell"] == "mystery_cell"]
        assert len(mystery) == 2  # one per run
        assert mystery["reason"].str.contains("unknown legacy baseline").all()
        index = pd.read_csv(result.index_csv)
        assert "mystery_cell" not in " ".join(index["row_key"].unique())


# ---------------------------------------------------------------------------
# Validity rule + record content
# ---------------------------------------------------------------------------


class TestValidityAndRecords:
    def test_invalid_rows_dropped_and_counted(self, tmp_path: Path) -> None:
        root = tmp_path / "arch"
        rows = _default_rows(4) + [
            _row("bad1", 999.0, error="boom"),
            _row("bad2", 999.0, empty="True"),
            _row("ex0__rep1", 1.0, rep="1"),
        ]
        _write_results_csv(root / RUN_SQUAD / "baselines" / "no_cache" / "trial_1", rows)
        result = bli.build_bridge(root, [RUN_SQUAD], tmp_path / "bridge")
        provenance = pd.read_csv(result.out_dir / "index" / "legacy_windows_map.csv")
        rec = provenance.iloc[0]
        assert rec["n_rows_raw"] == 7
        assert rec["n_rows_valid_rep0"] == 4
        assert rec["n_error_rows"] == 1
        assert rec["n_empty_gen_rows"] == 1
        assert rec["n_rep_gt0_rows"] == 1
        index = pd.read_csv(result.index_csv)
        requests = (
            (result.out_dir / index["window_dir"].iloc[0] / "requests.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )
        ids = {json.loads(line)["example_id"] for line in requests}
        assert ids == {"ex0", "ex1", "ex2", "ex3"}

    def test_records_are_numeric_only_and_split_by_artifact(
        self, archive: Path, tmp_path: Path
    ) -> None:
        result = _build(archive, tmp_path)
        index = pd.read_csv(result.index_csv)
        window_dir = result.out_dir / index["window_dir"].iloc[0]
        req = json.loads(
            (window_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        qa = json.loads(
            (window_dir / "qa_evidence.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert isinstance(req["example_id"], str)
        assert "ttft_ms" in req and "f1_score" not in req
        assert "f1_score" in qa and "ttft_ms" not in qa
        for rec in (req, qa):
            for key, value in rec.items():
                if key != "example_id":
                    assert isinstance(value, float)

    def test_policy_event_mask_derived_from_compression_applied(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "arch"
        rows = [
            _row("e1", 10.0, compression="True"),
            _row("e2", 11.0, compression="False"),
        ]
        _write_results_csv(
            root / RUN_SQUAD / "compression" / "compressed_rag" / "trial_1", rows
        )
        result = bli.build_bridge(root, [RUN_SQUAD], tmp_path / "bridge")
        index = pd.read_csv(result.index_csv)
        qa_lines = (
            (result.out_dir / index["window_dir"].iloc[0] / "qa_evidence.jsonl")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )
        masks = {r["example_id"]: r[bli.POLICY_EVENT_COLUMN]
                 for r in map(json.loads, qa_lines)}
        assert masks == {"e1": 1.0, "e2": 0.0}


# ---------------------------------------------------------------------------
# Read-only + fail-closed guarantees
# ---------------------------------------------------------------------------


class TestGuards:
    def test_archive_is_never_written(self, archive: Path, tmp_path: Path) -> None:
        before = {p.relative_to(archive).as_posix() for p in archive.rglob("*")}
        _build(archive, tmp_path)
        after = {p.relative_to(archive).as_posix() for p in archive.rglob("*")}
        assert before == after

    def test_out_dir_inside_archive_refused(self, archive: Path, tmp_path: Path) -> None:
        with pytest.raises(bli.BridgeError, match="READ-ONLY"):
            bli.build_bridge(archive, [RUN_SQUAD], archive / "bridge")

    def test_dataset_inference_fails_closed(self, tmp_path: Path) -> None:
        root = tmp_path / "arch"
        _write_results_csv(
            root / "run_without_suffix" / "baselines" / "no_cache" / "trial_1",
            _default_rows(3),
        )
        with pytest.raises(bli.BridgeError, match="cannot infer the dataset"):
            bli.build_bridge(root, ["run_without_suffix"], tmp_path / "bridge")

    def test_missing_run_fails_closed(self, archive: Path, tmp_path: Path) -> None:
        with pytest.raises(bli.BridgeError, match="does not exist"):
            bli.build_bridge(archive, ["no_such_run_squad_v2"], tmp_path / "bridge")


# ---------------------------------------------------------------------------
# End-to-end: the driver runs design-input on the bridge
# ---------------------------------------------------------------------------


class TestDriverEndToEnd:
    def test_design_input_headline_contrast_on_bridge(
        self, archive: Path, tmp_path: Path
    ) -> None:
        result = _build(archive, tmp_path)
        analysis = rca.run_analysis(
            result.out_dir,
            contrast_ids=[4],
            metrics=["ttft_ms"],
            mode="design-input",
        )
        stats = json.loads(analysis.stats_path.read_text(encoding="utf-8"))
        assert stats["mode_stamp"] == rca.DESIGN_STAMP
        headline = [c for c in stats["contrasts"] if c["contrast_id"] == 4]
        assert headline, "headline contrast #4 must compute on the bridge"
        datasets = {row["dataset"] for row in headline[0]["per_dataset"]}
        assert datasets == {"squad_v2", "hotpotqa"}
        # no confirmatory lock may ever appear in a design-input dry-run
        assert not (result.out_dir / rca.LOCK_NAME).exists()
