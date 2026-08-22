"""Unit pins for src/orchestration/campaign_session.py (the task #116 seam).

Covers the pieces the end-to-end proof (tests/test_run_experiment_campaign_layout.py)
exercises only implicitly: cell-identity derivation (legacy labels, explicit
CAGE_CELL_* axes, fail-closed refusals), model-slug resolution, the write-time
hash journal + seal cross-check (S0-15), per-window resume reset semantics,
and the import-light cell-dir CLI the shell resume gates call.
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

from src.analysis.cellspec import CellSpec  # noqa: E402
from src.orchestration import campaign_layout as cl  # noqa: E402
from src.orchestration import campaign_session as cs  # noqa: E402

RUN_ID = "20260821-1300-a-qwen3-14b"


def _run_root(tmp_path: Path) -> Path:
    return tmp_path / "results" / "camp1" / "a" / RUN_ID


def _fake_git(_repo: Path) -> tuple[str, bool]:
    return "deadbeef" * 5, False


def _session(tmp_path: Path, **overrides: Any) -> cs.CampaignCellSession:
    kwargs: dict[str, Any] = {
        "run_root": _run_root(tmp_path),
        "dataset": "squad_v2",
        "spec": CellSpec.from_baseline("B1", model="qwen3-14b"),
        "num_trials": 2,
        "run_seed": 7,
    }
    kwargs.update(overrides)
    return cs.CampaignCellSession(**kwargs)


# ---------------------------------------------------------------------------
# Model slug + engine mapping
# ---------------------------------------------------------------------------


def test_model_slug_roster_and_override() -> None:
    assert cs.resolve_model_slug("Qwen/Qwen3-14B", {}) == "qwen3-14b"
    assert cs.resolve_model_slug("qwen3-14b", {}) == "qwen3-14b"  # already a slug
    assert (
        cs.resolve_model_slug("Qwen/Qwen3-8B", {"CAGE_MODEL_SLUG": "qwen3-14b"})
        == "qwen3-14b"
    )  # explicit stand-in labeling (design-input runs)


def test_model_slug_unknown_refuses() -> None:
    with pytest.raises(cs.CampaignSessionError, match="CAGE_MODEL_SLUG"):
        cs.resolve_model_slug("Qwen/Qwen3-8B", {})
    with pytest.raises(cs.CampaignSessionError):
        cs.resolve_model_slug("x", {"CAGE_MODEL_SLUG": "not-a-roster-model"})


def test_derive_from_legacy_labels() -> None:
    spec = cs.derive_cell_spec(
        baseline="redis",
        baseline_label="redis_retrieval_cache_cold",
        backend="vllm",
        model="qwen3-14b",
        env={},
    )
    # redis is §7.5-retired onto the ranked retr-fresh tuple (≈ B6).
    assert spec.to_row_key() == "retr-fresh|rerank|none|single|vllm|qwen3-14b|F1"
    spec2 = cs.derive_cell_spec(
        baseline="compressed_cag",
        baseline_label=None,
        backend="vllm",
        model="qwen3-14b",
        env={},
    )
    assert spec2.policy == "compress-fp8" and spec2.family == "F3"


def test_derive_explicit_axes_override_legacy() -> None:
    env = {
        "CAGE_CELL_ARM": "retr-fresh",
        "CAGE_CELL_RETRIEVER": "bm25",
        "CAGE_CELL_FAMILY": "F2",
        "CAGE_CELL_BUDGET_R": "0.5",
        "CAGE_CELL_RATE_FRAC": "0.8",
    }
    spec = cs.derive_cell_spec(
        baseline="no_cache", baseline_label=None, backend="sglang",
        model="qwen3-14b", env=env,
    )
    assert spec.to_row_key() == "retr-fresh|bm25|none|single|sglang|qwen3-14b|F2|r0.5|lam0.8"


def test_derive_refusals_are_fail_closed() -> None:
    # Arm without its retriever axis.
    with pytest.raises(cs.CampaignSessionError, match="must be set together"):
        cs.derive_cell_spec(
            baseline="no_cache", baseline_label=None, backend="vllm",
            model="qwen3-14b", env={"CAGE_CELL_ARM": "gold-fresh"},
        )
    # Legacy-unmapped label (pilot-only cell).
    with pytest.raises(cs.CampaignSessionError, match="LEGACY_ALIASES"):
        cs.derive_cell_spec(
            baseline="not_a_baseline", baseline_label="distributed_router_replicated",
            backend="vllm", model="qwen3-14b", env={},
        )
    # No charter engine for legacy backends.
    with pytest.raises(cs.CampaignSessionError, match="charter engine"):
        cs.derive_cell_spec(
            baseline="no_cache", baseline_label=None, backend="ollama",
            model="qwen3-14b", env={},
        )
    # Charter-illegal combination surfaces CellSpec's own gate.
    with pytest.raises(cs.CampaignSessionError, match="charter-illegal"):
        cs.derive_cell_spec(
            baseline="no_cache", baseline_label=None, backend="hf-oracle",
            model="qwen3-14b", env={"CAGE_CELL_FAMILY": "F2"},
        )


# ---------------------------------------------------------------------------
# Session construction refusals
# ---------------------------------------------------------------------------


def test_session_refuses_non_roster_dataset(tmp_path: Path) -> None:
    with pytest.raises(cs.CampaignSessionError, match="dataset id"):
        _session(tmp_path, dataset="trivia_qa")


def test_session_refuses_bad_root_shape(tmp_path: Path) -> None:
    with pytest.raises(cs.CampaignSessionError, match="session"):
        cs.CampaignCellSession(
            run_root=tmp_path / "results" / "camp1" / "nope" / RUN_ID,
            dataset="squad_v2",
            spec=CellSpec.from_baseline("B1", model="qwen3-14b"),
            num_trials=1,
            run_seed=7,
        )
    with pytest.raises(cs.CampaignSessionError, match="grammar"):
        cs.CampaignCellSession(
            run_root=tmp_path / "results" / "camp1" / "a" / "Bad_Run_Id",
            dataset="squad_v2",
            spec=CellSpec.from_baseline("B1", model="qwen3-14b"),
            num_trials=1,
            run_seed=7,
        )


def test_from_cli_inactive_returns_none() -> None:
    class _Args:
        campaign_root = None
        baseline = "no_cache"
        baseline_label = None
        backend = "vllm"
        model = "qwen3-14b"
        dataset = "squad_v2"
        num_trials = 1
        seed = 7
        kv_cache_dtype = None
        top_k_sweep = False

    assert cs.CampaignCellSession.from_cli(_Args(), env={}) is None


def test_from_cli_refuses_top_k_sweep(tmp_path: Path) -> None:
    class _Args:
        campaign_root = str(_run_root(tmp_path))
        baseline = "rag"
        baseline_label = None
        backend = "vllm"
        model = "qwen3-14b"
        dataset = "squad_v2"
        num_trials = 1
        seed = 7
        kv_cache_dtype = None
        top_k_sweep = True

    with pytest.raises(cs.CampaignSessionError, match="top-k-sweep"):
        cs.CampaignCellSession.from_cli(_Args(), env={})


# ---------------------------------------------------------------------------
# Write-time hash journal + seal cross-check (S0-15)
# ---------------------------------------------------------------------------


def _mini_tree(tmp_path: Path) -> Path:
    """One-cell one-window v2 tree via campaign_layout, fully journaled."""
    run_root = _run_root(tmp_path)
    run = cl.CampaignRun.create(
        run_root,
        campaign="camp1",
        session="a",
        run_id=RUN_ID,
        model="qwen3-14b",
        engine="vllm",
        engine_version="0.0-test",
        seed=1,
        provider="test",
        hardware="test-gpu",
        dataset_manifests_sha256="0" * 64,
        cellspec_schema_version=1,
        git_provenance=_fake_git,
    )
    cell = run.cell(CellSpec.from_baseline("B1", model="qwen3-14b"))
    handle = cell.add_window(
        "squad_v2",
        seed=1,
        rep=1,
        t_start=0.0,
        t_end=10.0,
        requests=[{"example_id": "e0", "ttft_ms": 1.0}],
        cage_stats=[],
        engine_metrics={"snapshot": "x"},
        qa_evidence=[{"example_id": "e0"}],
    )
    paths = [run_root / "manifest.json", handle.window_dir.parent / "cell.json"]
    paths += sorted(p for p in handle.window_dir.iterdir() if p.is_file())
    cs.append_write_time_hashes(run_root, paths)
    return run_root


def test_seal_green_on_journaled_tree(tmp_path: Path) -> None:
    run_root = _mini_tree(tmp_path)
    ledger = cs.seal_campaign_run(run_root)
    assert ledger.is_file()
    journal = cs.read_write_time_journal(run_root)
    sealed = json.loads(ledger.read_text(encoding="utf-8"))["entries"]
    assert set(sealed) <= set(journal)


def test_seal_refuses_post_write_mutation(tmp_path: Path) -> None:
    run_root = _mini_tree(tmp_path)
    victim = next(run_root.glob("cells/*/window_*/requests.jsonl"))
    victim.write_text('{"example_id": "e0", "ttft_ms": 999.0}\n', encoding="utf-8")
    with pytest.raises(cs.CampaignSessionError, match="write-time hash"):
        cs.seal_campaign_run(run_root)
    assert not (run_root / "ledger.json").exists(), "refusal must seal NOTHING"


def test_seal_refuses_unjournaled_file(tmp_path: Path) -> None:
    run_root = _mini_tree(tmp_path)
    stray = next(run_root.glob("cells/*/window_*")) / "extra.json"
    stray.write_text("{}", encoding="utf-8")
    with pytest.raises(cs.CampaignSessionError, match="not in "):
        cs.seal_campaign_run(run_root)


def test_seal_refuses_without_journal(tmp_path: Path) -> None:
    run_root = _mini_tree(tmp_path)
    (run_root / cs.JOURNAL_NAME).unlink()
    with pytest.raises(cs.CampaignSessionError, match=cs.JOURNAL_NAME):
        cs.seal_campaign_run(run_root)


def test_journal_superseded_entries_tolerated(tmp_path: Path) -> None:
    """A rewritten file (e.g. cell.json after every window) seals against its
    LAST journal entry; stale entries for deleted files are ignored."""
    run_root = _mini_tree(tmp_path)
    meta = next(run_root.glob("cells/*/cell.json"))
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["baseline"] = doc.get("baseline", "")
    meta.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cs.append_write_time_hashes(run_root, [meta])  # the rewrite is journaled
    assert cs.seal_campaign_run(run_root).is_file()


# ---------------------------------------------------------------------------
# Per-window resume reset
# ---------------------------------------------------------------------------


def test_reset_incomplete_windows_drops_dir_and_declaration(tmp_path: Path) -> None:
    session = _session(tmp_path)
    cell_dir = session.cell_dir
    # Window 01: complete (valid metrics.json). Window 02: half-written (no
    # metrics.json). Window 03 of ANOTHER dataset: must never be touched.
    w1 = session.window_dir(1)
    w1.mkdir(parents=True)
    (w1 / "metrics.json").write_text('{"ok": 1}', encoding="utf-8")
    w2 = session.window_dir(2)
    w2.mkdir(parents=True)
    (w2 / "requests.jsonl").write_text('{"example_id": "e0"}\n', encoding="utf-8")
    other = cell_dir / "window_hotpotqa-01"
    other.mkdir(parents=True)
    (cell_dir / "cell.json").write_text(
        json.dumps(
            {
                "cellspec": session.spec.to_flat_dict(),
                "baseline": "B1",
                "windows": {
                    "squad_v2-01": {"dataset": "squad_v2"},
                    "squad_v2-02": {"dataset": "squad_v2"},
                    "hotpotqa-01": {"dataset": "hotpotqa"},
                },
            }
        ),
        encoding="utf-8",
    )
    assert session.reset_incomplete_windows() == [2]
    assert w1.is_dir() and not w2.exists() and other.is_dir()
    meta = json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))
    assert set(meta["windows"]) == {"squad_v2-01", "hotpotqa-01"}
    # The cell.json rewrite was journaled at write time (S0-15).
    journal = cs.read_write_time_journal(session.run_root)
    assert f"cells/{session.row_key}/cell.json" in journal


def test_window_complete_is_parse_checked(tmp_path: Path) -> None:
    session = _session(tmp_path)
    w1 = session.window_dir(1)
    w1.mkdir(parents=True)
    assert not session.window_complete(1)  # absent
    (w1 / "metrics.json").write_text('{"trunc', encoding="utf-8")
    assert not session.window_complete(1)  # invalid JSON = incomplete (J2)
    (w1 / "metrics.json").write_text('{"ok": 1}', encoding="utf-8")
    assert session.window_complete(1)


def test_reset_refuses_corrupt_cell_json(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.cell_dir.mkdir(parents=True)
    (session.cell_dir / "cell.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(cs.CampaignSessionError, match="CAGE_FORCE_RERUN"):
        session.reset_incomplete_windows()


# ---------------------------------------------------------------------------
# The cell-dir CLI (what the shell gates call via campaign_cell_dir)
# ---------------------------------------------------------------------------


def test_cell_dir_cli_prints_row_key_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CAGE_CAMPAIGN_ROOT", str(_run_root(tmp_path)))
    rc = cs.main(["cell-dir", "--baseline", "prefix_cache", "--model", "qwen3-14b"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out.endswith("cells/gold-reuse|none|none|single|vllm|qwen3-14b|F1")


def test_cell_dir_cli_refuses_unmapped_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CAGE_CAMPAIGN_ROOT", str(_run_root(tmp_path)))
    rc = cs.main(
        ["cell-dir", "--baseline", "nope", "--baseline-label",
         "distributed_router_replicated", "--model", "qwen3-14b"]
    )
    assert rc == 1
    assert "LEGACY_ALIASES" in capsys.readouterr().err


def test_cell_dir_cli_requires_campaign_root(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CAGE_CAMPAIGN_ROOT", raising=False)
    rc = cs.main(["cell-dir", "--baseline", "no_cache", "--model", "qwen3-14b"])
    assert rc == 2
    assert "CAGE_CAMPAIGN_ROOT" in capsys.readouterr().err
