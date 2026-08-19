"""Offline tests for scripts/4_analysis/calibrate_instrument_a_tau.py (#146).

The Instrument-A (LettuceDetect) τ-calibration runner for the Qasper Y
predicate (owner decision #120/F7, ADR-0091). Everything here is offline and
runs in seconds: NO model download, NO network, NO real LettuceDetect load.
Layers covered:

- anchor validation arms: valid inventory (counts + sha256), malformed rows,
  count mismatch, duplicate ids, one-class components, missing files —
  all refusals typed (CalibrationError);
- τ-sweep math on synthetic scores with HAND-DERIVED operating tables,
  optimum, tie-break (smallest τ) and the 1-percentage-point band;
- manifest schema + the no-overwrite refusal (before any scoring happens);
- --plan mode never touches the network/model: proven by poisoning the model
  stack in sys.modules AND a raising build_scorer sentinel;
- the REAL build_scorer wiring through a sys.modules lettucedetect stub
  (pattern: tests/test_instrument_revision_provenance.py) including
  instrument_provenance capture;
- the DEFAULT_LETTUCEDETECT_MODEL constant is drift-guarded against the
  QualityEvaluator in-code default (the constant exists so --plan never
  imports the model stack).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "4_analysis" / "calibrate_instrument_a_tau.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "calibrate_instrument_a_tau", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # register BEFORE exec (dataclass-safe)
    spec.loader.exec_module(module)
    return module


cal = _load_module()


QUALITY_ENVS = (
    "CAGE_LETTUCEDETECT_MODEL",
    "CAGE_LETTUCEDETECT_REVISION",
    "CAGE_NLI_REVISION",
    "CAGE_EMBEDDING_REVISION",
    "CAGE_BERTSCORE_REVISION",
    "CAGE_DISABLE_LETTUCEDETECT",
    "CAGE_QUALITY_STRICT",
    "CAGE_CLAIM_CHECKER",
    "CAGE_NLI_THREE_CLASS",
    "CAGE_BERTSCORE_IDF",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in QUALITY_ENVS:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# Anchor fixtures: a hand-designed 11-row pool with hand-derivable optimum
# --------------------------------------------------------------------------- #
# label: 1 = grounded, 0 = hallucinated. The stub scorer parses "score=<v>"
# out of the claim, so the score distribution is fixed by construction:
# hallucinated scores {.05,.10,.15,.20,.25,.30}, grounded {.70,.80,.85,.90,.95}
# -> perfectly separated; unique optimum tau*=0.70 (BA 1.0), band = {0.70}.
_POOL: dict[str, list[tuple[str, int, float]]] = {
    "frank_test": [("f1", 0, 0.10), ("f2", 1, 0.90), ("f3", 0, 0.20)],
    "qags_cnndm": [("qc1", 0, 0.30), ("qc2", 1, 0.80)],
    "qags_xsum": [("qx1", 0, 0.15), ("qx2", 1, 0.95)],
    "ragtruth_test": [
        ("r1", 0, 0.25), ("r2", 1, 0.85), ("r3", 0, 0.05), ("r4", 1, 0.70),
    ],
}
_POOL_N = sum(len(v) for v in _POOL.values())  # 11


def _anchor_row(component: str, rid: str, label: int, score: float) -> dict[str, Any]:
    return {
        "id": f"{component}/{rid}",
        "component": component,
        "source_dataset": "TEST",
        "task_type": "QA",
        "question": None,
        "context": f"Context paragraph for {rid}.",
        "claim": f"Claim {rid} score={score}",
        "label": label,
    }


def _write_pool(dirpath: Path, pool: dict[str, list[tuple[str, int, float]]]) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    for component, rows in pool.items():
        path = dirpath / f"{component}.jsonl"
        path.write_text(
            "".join(
                json.dumps(_anchor_row(component, rid, label, score)) + "\n"
                for rid, label, score in rows
            ),
            encoding="utf-8",
        )
    return dirpath


@pytest.fixture()
def anchors_dir(tmp_path: Path) -> Path:
    return _write_pool(tmp_path / "anchors", _POOL)


def _stub_scorer(question: str, context: list[str], answer: str) -> float:
    """Deterministic stand-in: the intended score rides inside the claim."""
    m = re.search(r"score=([0-9.]+)", answer)
    assert m, f"stub scorer got a claim without a score marker: {answer!r}"
    return float(m.group(1))


def _stub_factory():
    def provenance() -> dict[str, Any]:
        return {
            "instrument_id": "stub-model@lettucedetect-0.0",
            "provenance": {"lettucedetect": {"model": "stub-model", "revision": None}},
            "env": {
                "CAGE_LETTUCEDETECT_MODEL": None,
                "CAGE_LETTUCEDETECT_REVISION": None,
            },
        }

    return _stub_scorer, provenance


# --------------------------------------------------------------------------- #
# (1) Anchor validation arms
# --------------------------------------------------------------------------- #
def test_load_anchors_valid_inventory(anchors_dir: Path) -> None:
    rows, inventory = cal.load_anchors(anchors_dir, _POOL_N)
    assert len(rows) == _POOL_N
    assert inventory["n_total"] == _POOL_N
    assert inventory["n_grounded"] == 5
    assert inventory["n_hallucinated"] == 6
    assert set(inventory["components"]) == set(cal.ANCHOR_COMPONENTS)
    frank = inventory["components"]["frank_test"]
    assert frank["n_rows"] == 3
    assert frank["n_grounded"] == 1
    assert frank["n_hallucinated"] == 2
    expected_sha = hashlib.sha256(
        (anchors_dir / "frank_test.jsonl").read_bytes()
    ).hexdigest()
    assert frank["sha256"] == expected_sha
    # Pool order is component-file order (the fixed ANCHOR_COMPONENTS tuple).
    assert rows[0]["id"] == "frank_test/f1"
    assert rows[-1]["id"] == "ragtruth_test/r4"


def test_load_anchors_count_mismatch_refuses(anchors_dir: Path) -> None:
    with pytest.raises(cal.CalibrationError, match="expected-anchors"):
        cal.load_anchors(anchors_dir, _POOL_N + 1)


@pytest.mark.parametrize("bad", [0, -3, True, "4724"])
def test_load_anchors_bad_expected_refuses(anchors_dir: Path, bad: Any) -> None:
    with pytest.raises(cal.CalibrationError):
        cal.load_anchors(anchors_dir, bad)


def test_load_anchors_missing_dir_refuses(tmp_path: Path) -> None:
    with pytest.raises(cal.CalibrationError, match="not found"):
        cal.load_anchors(tmp_path / "nope", _POOL_N)


def test_load_anchors_missing_component_file_refuses(anchors_dir: Path) -> None:
    (anchors_dir / "qags_xsum.jsonl").unlink()
    with pytest.raises(cal.CalibrationError, match="qags_xsum"):
        cal.load_anchors(anchors_dir, _POOL_N)


def test_load_anchors_empty_component_refuses(anchors_dir: Path) -> None:
    (anchors_dir / "qags_cnndm.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(cal.CalibrationError, match="empty"):
        cal.load_anchors(anchors_dir, _POOL_N)


def test_load_anchors_invalid_json_refuses(anchors_dir: Path) -> None:
    path = anchors_dir / "frank_test.jsonl"
    path.write_text(path.read_text() + "{not json\n", encoding="utf-8")
    with pytest.raises(cal.CalibrationError, match="invalid JSON"):
        cal.load_anchors(anchors_dir, _POOL_N)


def test_load_anchors_missing_field_refuses(anchors_dir: Path) -> None:
    row = _anchor_row("frank_test", "fX", 0, 0.4)
    del row["claim"]
    path = anchors_dir / "frank_test.jsonl"
    path.write_text(path.read_text() + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(cal.CalibrationError, match="missing required fields"):
        cal.load_anchors(anchors_dir, _POOL_N + 1)


@pytest.mark.parametrize("label", [2, -1, 0.5, True, None, "1"])
def test_load_anchors_bad_label_refuses(anchors_dir: Path, label: Any) -> None:
    row = _anchor_row("frank_test", "fX", 0, 0.4)
    row["label"] = label
    path = anchors_dir / "frank_test.jsonl"
    path.write_text(path.read_text() + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(cal.CalibrationError, match="label"):
        cal.load_anchors(anchors_dir, _POOL_N + 1)


def test_load_anchors_wrong_component_value_refuses(anchors_dir: Path) -> None:
    row = _anchor_row("ragtruth_test", "fX", 0, 0.4)  # wrong: lives in frank file
    path = anchors_dir / "frank_test.jsonl"
    path.write_text(path.read_text() + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(cal.CalibrationError, match="component"):
        cal.load_anchors(anchors_dir, _POOL_N + 1)


def test_load_anchors_duplicate_id_refuses(anchors_dir: Path) -> None:
    row = _anchor_row("qags_xsum", "qx1", 0, 0.4)  # id already in the pool
    path = anchors_dir / "qags_xsum.jsonl"
    path.write_text(path.read_text() + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(cal.CalibrationError, match="duplicate"):
        cal.load_anchors(anchors_dir, _POOL_N + 1)


def test_load_anchors_one_class_component_refuses(anchors_dir: Path) -> None:
    path = anchors_dir / "qags_cnndm.jsonl"
    path.write_text(
        json.dumps(_anchor_row("qags_cnndm", "qc1", 0, 0.3)) + "\n",
        encoding="utf-8",
    )  # drops the grounded row -> one-class component
    with pytest.raises(cal.CalibrationError, match="one class"):
        cal.load_anchors(anchors_dir, _POOL_N - 1)


def test_load_anchors_empty_context_refuses(anchors_dir: Path) -> None:
    row = _anchor_row("frank_test", "fX", 0, 0.4)
    row["context"] = "   "
    path = anchors_dir / "frank_test.jsonl"
    path.write_text(path.read_text() + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(cal.CalibrationError, match="context"):
        cal.load_anchors(anchors_dir, _POOL_N + 1)


# --------------------------------------------------------------------------- #
# (2) τ-sweep math: hand-derived operating tables, tie-break, band
# --------------------------------------------------------------------------- #
def test_sweep_tau_hand_derived_table() -> None:
    # hallucinated scores [.1, .2], grounded [.8, .9] (h=1 marks hallucinated)
    table = cal.sweep_tau([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0])
    assert [r["tau"] for r in table] == [0.1, 0.2, 0.8, 0.9]
    t01, t02, t08, t09 = table
    # tau=.1: nothing predicted hallucinated
    assert (t01["tp"], t01["fp"], t01["fn"], t01["tn"]) == (0, 0, 2, 2)
    assert t01["precision"] is None
    assert t01["recall"] == 0.0
    assert t01["f1"] == 0.0
    assert t01["balanced_accuracy"] == 0.5
    # tau=.2: {.1} predicted hallucinated
    assert (t02["tp"], t02["fp"], t02["fn"], t02["tn"]) == (1, 0, 1, 2)
    assert t02["precision"] == 1.0
    assert t02["recall"] == 0.5
    assert t02["balanced_accuracy"] == 0.75
    assert t02["f1"] == pytest.approx(2 / 3)
    # tau=.8: perfect split
    assert (t08["tp"], t08["fp"], t08["fn"], t08["tn"]) == (2, 0, 0, 2)
    assert t08["balanced_accuracy"] == 1.0
    assert t08["f1"] == 1.0
    # tau=.9: one grounded item swept in
    assert (t09["tp"], t09["fp"], t09["fn"], t09["tn"]) == (2, 1, 0, 1)
    assert t09["precision"] == pytest.approx(2 / 3)
    assert t09["specificity"] == 0.5
    assert t09["balanced_accuracy"] == 0.75

    point, band = cal.select_tau(table)
    assert point["tau"] == 0.8
    assert band["tau_low"] == 0.8
    assert band["tau_high"] == 0.8
    assert band["n_candidates_in_band"] == 1
    assert band["tolerance"] == 0.01
    assert band["balanced_accuracy_floor"] == pytest.approx(0.99)


def test_select_tau_tie_breaks_to_smallest_and_band_spans_ties() -> None:
    # hallucinated [.1, .7], grounded [.4, .9]:
    #   tau .1 -> BA .5 ; tau .4 -> BA .75 ; tau .7 -> BA .5 ; tau .9 -> BA .75
    # exact tie at max BA .75 between tau .4 and tau .9 -> smallest wins.
    table = cal.sweep_tau([0.1, 0.7, 0.4, 0.9], [1, 1, 0, 0])
    bas = {r["tau"]: r["balanced_accuracy"] for r in table}
    assert bas == {0.1: 0.5, 0.4: 0.75, 0.7: 0.5, 0.9: 0.75}
    point, band = cal.select_tau(table)
    assert point["tau"] == 0.4  # tie -> smallest
    assert band["tau_low"] == 0.4
    assert band["tau_high"] == 0.9
    assert band["n_candidates_in_band"] == 2  # .7 (BA .5) is NOT in the band


def test_band_includes_near_optimum_within_one_percentage_point() -> None:
    # 49 hallucinated at .1, 1 hallucinated at .3, 50 grounded at .9:
    #   tau .3 -> recall 49/50=.98, spec 1  -> BA .99
    #   tau .9 -> recall 1,          spec 1 -> BA 1.0 (optimum)
    scores = [0.1] * 49 + [0.3] + [0.9] * 50
    labels = [1] * 50 + [0] * 50
    table = cal.sweep_tau(scores, labels)
    point, band = cal.select_tau(table)
    assert point["tau"] == 0.9
    assert point["balanced_accuracy"] == 1.0
    assert band["tau_low"] == pytest.approx(0.3)  # BA .99 >= 1.0 - .01
    assert band["tau_high"] == pytest.approx(0.9)
    assert band["n_candidates_in_band"] == 2


@pytest.mark.parametrize(
    "scores, labels, match",
    [
        ([0.1, 0.2], [1], "equal-length"),
        ([], [], "equal-length"),
        ([0.1, 1.2], [1, 0], "within"),
        ([0.1, -0.2], [1, 0], "within"),
        ([0.1, float("nan")], [1, 0], "within"),
        ([0.1, 0.2], [1, 2], "binary"),
        ([0.1, 0.2], [1, 1], "BOTH classes"),
        ([0.1, 0.2], [0, 0], "BOTH classes"),
    ],
)
def test_sweep_tau_refusal_arms(scores: Any, labels: Any, match: str) -> None:
    with pytest.raises(cal.CalibrationError, match=match):
        cal.sweep_tau(scores, labels)


def test_select_tau_empty_table_refuses() -> None:
    with pytest.raises(cal.CalibrationError, match="empty"):
        cal.select_tau([])


# --------------------------------------------------------------------------- #
# (3) Manifest schema + no-overwrite refusal (stubbed scorer, no model)
# --------------------------------------------------------------------------- #
def test_run_calibration_manifest_schema(anchors_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out" / "2026-01-01"
    manifest_path = cal.run_calibration(
        anchors_dir=anchors_dir,
        expected_anchors=_POOL_N,
        out_dir=out_dir,
        seed=cal.DEFAULT_SEED,
        force=False,
        scorer_factory=_stub_factory,
    )
    assert manifest_path == out_dir / cal.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest) == set(cal._MANIFEST_REQUIRED_KEYS)
    assert manifest["schema_version"] == cal.SCHEMA_VERSION
    assert manifest["artifact"] == cal.ARTIFACT_KIND
    assert manifest["rule"] == cal.SELECTION_RULE
    assert manifest["seed"] == cal.DEFAULT_SEED
    assert manifest["expected_anchors"] == _POOL_N

    # Hand-derived optimum on the designed pool: perfect separation at .70.
    assert manifest["tau"] == pytest.approx(0.70)
    op = manifest["operating_point"]
    assert (op["tp"], op["fp"], op["fn"], op["tn"]) == (6, 0, 0, 5)
    assert op["balanced_accuracy"] == 1.0
    assert manifest["band"]["tau_low"] == pytest.approx(0.70)
    assert manifest["band"]["tau_high"] == pytest.approx(0.70)
    assert manifest["band"]["n_candidates_in_band"] == 1

    # Operating table covers every distinct observed score.
    assert len(manifest["operating_table"]) == 11

    # Supplementary RAGTruth-only block (NOT the registered value).
    supp = manifest["supplementary_ragtruth_only"]
    assert supp["component"] == "ragtruth_test"
    assert supp["n_rows"] == 4
    assert supp["tau"] == pytest.approx(0.70)
    assert "NOT the registered" in supp["note"]

    # Anchor inventory rides in the manifest with per-file hashes.
    inv = manifest["anchor_inventory"]
    assert inv["n_total"] == _POOL_N
    for component in cal.ANCHOR_COMPONENTS:
        assert len(inv["components"][component]["sha256"]) == 64

    # Instrument provenance from the (stubbed) factory.
    assert manifest["instrument"]["instrument_id"] == "stub-model@lettucedetect-0.0"
    assert "lettucedetect" in manifest["instrument"]["provenance"]

    # Score stash exists, hashes match, one row per anchor.
    scores_path = out_dir / cal.SCORES_NAME
    assert manifest["scores_file"] == cal.SCORES_NAME
    assert (
        manifest["scores_sha256"]
        == hashlib.sha256(scores_path.read_bytes()).hexdigest()
    )
    score_rows = [json.loads(l) for l in scores_path.read_text().splitlines()]
    assert len(score_rows) == _POOL_N
    assert set(score_rows[0]) == {"id", "component", "label", "score"}


def test_run_calibration_refuses_overwrite_then_force(
    anchors_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out" / "2026-01-01"
    cal.run_calibration(
        anchors_dir=anchors_dir, expected_anchors=_POOL_N, out_dir=out_dir,
        seed=1, force=False, scorer_factory=_stub_factory,
    )
    with pytest.raises(cal.CalibrationError, match="--force"):
        cal.run_calibration(
            anchors_dir=anchors_dir, expected_anchors=_POOL_N, out_dir=out_dir,
            seed=1, force=False, scorer_factory=_stub_factory,
        )
    # Deliberate re-run succeeds.
    cal.run_calibration(
        anchors_dir=anchors_dir, expected_anchors=_POOL_N, out_dir=out_dir,
        seed=1, force=True, scorer_factory=_stub_factory,
    )


def test_overwrite_refusal_fires_before_any_scoring(
    anchors_dir: Path, tmp_path: Path
) -> None:
    """The refusal must come BEFORE the multi-hour scoring pass."""
    out_dir = tmp_path / "out" / "2026-01-01"
    out_dir.mkdir(parents=True)
    (out_dir / cal.MANIFEST_NAME).write_text("{}", encoding="utf-8")

    def exploding_factory():
        raise AssertionError("scorer_factory must not be called on refusal")

    with pytest.raises(cal.CalibrationError, match="--force"):
        cal.run_calibration(
            anchors_dir=anchors_dir, expected_anchors=_POOL_N, out_dir=out_dir,
            seed=1, force=False, scorer_factory=exploding_factory,
        )


def test_score_anchors_refuses_none_and_out_of_range() -> None:
    rows = [_anchor_row("frank_test", "f1", 0, 0.1)]
    with pytest.raises(cal.CalibrationError, match="no grounding score"):
        cal.score_anchors(rows, lambda q, c, a: None, progress_every=0)
    with pytest.raises(cal.CalibrationError, match="invalid grounding score"):
        cal.score_anchors(rows, lambda q, c, a: 1.5, progress_every=0)


# --------------------------------------------------------------------------- #
# (4) main() CLI arms (stubbed build_scorer; no model)
# --------------------------------------------------------------------------- #
def test_main_end_to_end_and_no_overwrite(
    anchors_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(cal, "build_scorer", lambda device="cpu": _stub_factory())
    argv = [
        "--anchors-dir", str(anchors_dir),
        "--expected-anchors", str(_POOL_N),
        "--out-root", str(tmp_path / "reg"),
        "--date", "2026-01-02",
    ]
    assert cal.main(argv) == 0
    out = capsys.readouterr().out
    assert "tau = 0.7" in out
    manifest_path = tmp_path / "reg" / "2026-01-02" / cal.MANIFEST_NAME
    assert manifest_path.is_file()

    # Second run refuses (registered artifact), --force re-runs.
    assert cal.main(argv) == 1
    assert "--force" in capsys.readouterr().err
    assert cal.main(argv + ["--force"]) == 0


def test_main_count_mismatch_is_a_clean_refusal(
    anchors_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    rc = cal.main([
        "--anchors-dir", str(anchors_dir),
        "--expected-anchors", "4724",  # real pool size, wrong for the fixture
        "--out-root", str(tmp_path / "reg"),
        "--plan",
    ])
    assert rc == 1
    assert "expected-anchors" in capsys.readouterr().err


def test_main_bad_date_refuses(anchors_dir: Path, tmp_path: Path,
                               capsys: pytest.CaptureFixture) -> None:
    rc = cal.main([
        "--anchors-dir", str(anchors_dir),
        "--expected-anchors", str(_POOL_N),
        "--out-root", str(tmp_path / "reg"),
        "--date", "01-02-2026",
        "--plan",
    ])
    assert rc == 1
    assert "YYYY-MM-DD" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# (5) --plan mode: validates anchors, prints the plan, NEVER touches a model
# --------------------------------------------------------------------------- #
def test_plan_mode_never_touches_model_or_network(
    anchors_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    # Poison the entire model stack: ANY import attempt raises ImportError.
    for name in (
        "lettucedetect", "lettucedetect.models", "lettucedetect.models.inference",
        "transformers", "torch", "sentence_transformers", "bert_score",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    # Sentinel on the scorer seam: --plan must never build a scorer.
    def exploding_build_scorer(device: str = "cpu"):
        raise AssertionError("--plan must not call build_scorer")
    monkeypatch.setattr(cal, "build_scorer", exploding_build_scorer)
    # Sentinel on the network: --plan must never open a socket.
    import socket as socket_module

    def exploding_socket(*a: Any, **kw: Any):
        raise AssertionError("--plan must not touch the network")
    monkeypatch.setattr(socket_module, "socket", exploding_socket)

    rc = cal.main([
        "--anchors-dir", str(anchors_dir),
        "--expected-anchors", str(_POOL_N),
        "--out-root", str(tmp_path / "reg"),
        "--date", "2026-01-03",
        "--plan",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PLAN" in out
    assert f"n={_POOL_N}" in out
    assert "estimated time" in out
    assert cal.SELECTION_RULE["id"] in out
    # Plan writes NOTHING.
    assert not (tmp_path / "reg").exists()


def test_plan_mode_warns_on_manifest_collision(
    anchors_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    out_dir = tmp_path / "reg" / "2026-01-04"
    out_dir.mkdir(parents=True)
    (out_dir / cal.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    rc = cal.main([
        "--anchors-dir", str(anchors_dir),
        "--expected-anchors", str(_POOL_N),
        "--out-root", str(tmp_path / "reg"),
        "--date", "2026-01-04",
        "--plan",
    ])
    assert rc == 0
    assert "EXISTS" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# (6) REAL build_scorer wiring through a sys.modules lettucedetect stub
# --------------------------------------------------------------------------- #
_STUB_SHA = "c" * 40


def _stub_lettucedetect(monkeypatch: pytest.MonkeyPatch) -> None:
    """sys.modules stub exercising the REAL QualityEvaluator lazy-load path
    (pattern: tests/test_instrument_revision_provenance.py). The detector
    flags the whole answer iff it contains 'HALLUC'."""

    class _Tok:
        def __call__(self, texts: list[str], add_special_tokens: bool = False):
            return {"input_ids": [[1] * len(t.split()) for t in texts]}

    class _Cfg:
        _commit_hash = _STUB_SHA
        max_position_embeddings = 8192

    class HallucinationDetector:
        def __init__(self, method: str, model_path: str, device: str) -> None:
            self.detector = types.SimpleNamespace(
                model=types.SimpleNamespace(config=_Cfg()), tokenizer=_Tok(),
            )

        def predict(self, context: list[str], question: str, answer: str,
                    output_format: str) -> list[dict[str, Any]]:
            if "HALLUC" in answer:
                return [{"start": 0, "end": len(answer), "text": answer}]
            return []

    root = types.ModuleType("lettucedetect")
    models = types.ModuleType("lettucedetect.models")
    inference = types.ModuleType("lettucedetect.models.inference")
    inference.HallucinationDetector = HallucinationDetector  # type: ignore[attr-defined]
    models.inference = inference  # type: ignore[attr-defined]
    root.models = models  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lettucedetect", root)
    monkeypatch.setitem(sys.modules, "lettucedetect.models", models)
    monkeypatch.setitem(sys.modules, "lettucedetect.models.inference", inference)


def test_build_scorer_real_path_with_stubbed_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_lettucedetect(monkeypatch)
    scorer, provenance = cal.build_scorer(device="cpu")
    assert scorer("", ["Some context paragraph."], "A grounded claim.") == 1.0
    assert scorer("", ["Some context paragraph."], "HALLUC nonsense.") == 0.0
    prov = provenance()
    assert prov["instrument_id"].startswith(cal.DEFAULT_LETTUCEDETECT_MODEL + "@")
    assert prov["provenance"]["lettucedetect"]["model"] == (
        cal.DEFAULT_LETTUCEDETECT_MODEL
    )
    assert prov["provenance"]["lettucedetect"]["revision"] == _STUB_SHA
    assert prov["env"]["CAGE_LETTUCEDETECT_MODEL"] is None


def test_build_scorer_respects_model_and_revision_pin_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CAGE_LETTUCEDETECT_MODEL is honored; a mismatching
    CAGE_LETTUCEDETECT_REVISION pin fails closed through the REAL
    QualityEvaluator machinery (strict mode raises)."""
    from src.evaluation.quality import InstrumentUnavailableError

    _stub_lettucedetect(monkeypatch)
    monkeypatch.setenv("CAGE_LETTUCEDETECT_MODEL", "org/custom-model")
    monkeypatch.setenv("CAGE_LETTUCEDETECT_REVISION", "d" * 40)  # != stub SHA
    scorer, provenance = cal.build_scorer(device="cpu")
    with pytest.raises(InstrumentUnavailableError) as ei:
        scorer("", ["Some context paragraph."], "A grounded claim.")
    assert ei.value.instrument == "lettucedetect"
    assert ei.value.model == "org/custom-model"
    assert "CAGE_LETTUCEDETECT_REVISION" in ei.value.cause


# --------------------------------------------------------------------------- #
# (7) Registration-bound pins
# --------------------------------------------------------------------------- #
def test_default_model_constant_matches_quality_evaluator_default() -> None:
    """DEFAULT_LETTUCEDETECT_MODEL exists so --plan never imports the model
    stack; it must never drift from the QualityEvaluator in-code default."""
    from src.evaluation.quality import QualityEvaluator

    ev = QualityEvaluator(
        use_nli=False, use_embeddings=False, use_bertscore=False,
        use_rouge=False, use_lettucedetect=False, claim_checker="nli",
    )
    assert ev.lettucedetect_model_name == cal.DEFAULT_LETTUCEDETECT_MODEL


def test_selection_rule_is_registration_bound_in_docstring() -> None:
    doc = cal.__doc__
    assert "REGISTERED SELECTION RULE" in doc
    assert "MANIFEST SCHEMA" in doc
    assert "BALANCED ACCURACY" in doc
    assert "smallest" in doc.lower()
    assert "1" in cal.SELECTION_RULE["band"] and "percentage point" in (
        cal.SELECTION_RULE["band"]
    )
    assert cal.BAND_TOLERANCE == 0.01
    assert cal.SELECTION_RULE["id"] == "max_balanced_accuracy_pooled_v1"


def test_band_tolerance_and_rule_ride_in_manifest_constants() -> None:
    assert "tau" in cal._MANIFEST_REQUIRED_KEYS
    assert "rule" in cal._MANIFEST_REQUIRED_KEYS
    assert "band" in cal._MANIFEST_REQUIRED_KEYS
    assert "operating_table" in cal._MANIFEST_REQUIRED_KEYS
    assert "anchor_inventory" in cal._MANIFEST_REQUIRED_KEYS
    assert "instrument" in cal._MANIFEST_REQUIRED_KEYS
    assert "seed" in cal._MANIFEST_REQUIRED_KEYS
