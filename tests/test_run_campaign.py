"""Pins for scripts/3_run/run_campaign.py — THE campaign sweep driver (T1.2).

WHAT is pinned and WHY:

- **Session-'a' enumeration integers** (870 cells / 2610 windows / 36
  relaunches / 13 blocked; per-family 104+10 / 340+272 / 144 cells): the grid is
  the charter's registered design (§6.1 full 5×6 factorial, §6.8 reduced 3×3,
  §7.6.1 family × group matrix, 3 replications per grid point per D6 §6.3,
  and the ADR-0106 B12 ladder: one corpus-trunc cell PER RUNG (1400, 700)
  wherever B12 is carried).
  A silent count drift here is a silently changed experiment — the single
  worst failure mode of a sweep driver.
- **Cell BEHAVIOR realization** (the verified T1.2 blocker): the --baseline
  token selects only the base pipeline, so every arm's remaining behavior
  must be realized explicitly — corpus arms carry --corpus-prefix-budget
  (B12's budget TRUNCATED: that is the whole B12-vs-B3 slot), the reranker is
  pinned explicitly (B5 OFF vs B6 ON — the one pre-registered ablation),
  retr-comp compresses RETRIEVED context (the Phase-2 confound fix),
  retr-trunc rank-truncates, corpus-comp launches with the fp8 KV env and
  retr-store launches vLLM with the LMCache connector env verbatim from
  run_kv_store.sh. Within one grid slice, no two baselines may share the same
  (behavior argv, serving config) — identical pairs are mislabeled duplicate
  cells. Arms whose lever the FROZEN launcher fleet cannot realize
  (retr-store on sglang) must be enumerated BLOCKED, never served lever-free.
- **Ordering / relaunch minimality**: relaunch steps must equal the number of
  DISTINCT executable serving configs (engine, prefix_mode, model, budget_r,
  kv_dtype, connector), rate changes must never relaunch, blocked cells
  launch nothing, and every executable non-hf cell must run under the
  relaunch config that precedes it — a cell served under the wrong config is
  mislabeled data.
- **The identity seam**: each cell step's argv + CAGE_CELL_* env must
  round-trip through campaign_session.derive_cell_spec to the EXACT row key
  the plan claims — otherwise the driver and the runner would write different
  cells (the gap this task exists to close).
- **Launcher env exactness**: CAGE_KV_BUDGET_BYTES = floor(r×D) (vLLM) and
  CAGE_SGLANG_MAX_TOTAL_TOKENS = floor(r×D) // 163840 (SGLang; qwen3-14b
  bf16 KV = 2·40·8·128·2 B/token) — both pinned by independent arithmetic,
  never by re-calling the planner.
- **Uniform max_model_len (backlog A10)**: every relaunch of both engines
  carries VLLM_MAX_MODEL_LEN=32768 (RULER SHAPE-32K = 32,512 + 256, plus
  long Qasper papers; the pilot shell default 4096 must never reach a
  campaign relaunch), the relaunch record and the plan header carry the
  value, load_plan refuses a relaunch without it, and a grid registering
  RULER tasks refuses a value below SHAPE-32K.
- **Refusal paths** (fail-closed doctrine): unknown session; known-but-
  unregistered session; empty enumeration; missing/invalid/mismatched floor
  table; blocked cells at 'run' WITHOUT the explicit --skip-blocked consent;
  zero executable cells; hand-edited row keys. λ_compute=null must surface as
  the "kv-bound-only [pending calibration]" rate label, never a bare number
  (P6 honesty).
- **'run' semantics on stubs** (no GPU, no network, no real engines): resume
  skips complete windows, --force-rerun overrides, a failed cell writes a
  dot-named DATASET-SUFFIXED sentinel (F1 row keys are shared by 4 datasets)
  and execution CONTINUES with a nonzero exit, a later success/complete-skip
  CLEARS the stale sentinel, --skip-blocked runs the executable subset loudly
  and gates a plain --seal, --seal fires only on fully-passed runs unless
  --seal-partial (which also seals despite failures/blocked skips).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cellspec import CellSpec  # noqa: E402
from src.orchestration.campaign_session import derive_cell_spec  # noqa: E402

_SCRIPT_PATH = REPO_ROOT / "scripts" / "3_run" / "run_campaign.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_campaign", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # register BEFORE exec (dataclass-safe)
    spec.loader.exec_module(module)
    return module


rc = _load_module()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ANCHOR_DEMAND = 10_000_000_000  # arbitrary but realistic-scale demand bytes


def _floor_table_doc(
    *,
    model: str = "qwen3-14b",
    grid: str = "anchor-fine",
    # Session 'a' pre-resolves EVERY registered r — the §6.1 factorial's 5
    # levels PLUS the §6.4 fine additions {1.25, 0.375} (W4.3), so the
    # fixture default is the 7-level anchor-fine union.
    r_values=(1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25),
    lambda_compute: Optional[float] = None,
) -> Dict[str, Any]:
    rows = []
    for r in r_values:
        lam_kv = 2.0 * r  # deterministic, distinguishable per r
        lam_star = lam_kv if lambda_compute is None else min(lam_kv, lambda_compute)
        rows.append(
            {
                "r": r,
                "demand_bytes": _ANCHOR_DEMAND,
                "budget_bytes": int(r * _ANCHOR_DEMAND),
                "lambda_kv_rps": lam_kv,
                "lambda_compute_rps": lambda_compute,
                "lambda_star_pred_rps": lam_star,
                "lambda_star_basis": "test",
            }
        )
    return {
        "schema": "floor-table-v1",
        "generated_inputs": {
            "model": model,
            "engine": "vllm",
            "kv_dtype": "bf16",
            "grid": grid,
        },
        "rows": rows,
    }


@pytest.fixture()
def floor_table(tmp_path: Path) -> Path:
    path = tmp_path / "floor_table.json"
    path.write_text(json.dumps(_floor_table_doc()), encoding="utf-8")
    return path


#: The dense-retriever freeze slot as the registration artifact carries it
#: (INSTRUMENT_REVISIONS.dense_retriever, ADR-0099). Mirrored into a tmp
#: artifact so every plan built here is hermetic (no dependency on the
#: untracked MyDocs copy) while pinning the SAME values the driver reads.
_FREEZE_EMBEDDING_MODEL = "intfloat/e5-large-v2"
_FREEZE_EMBEDDING_REVISION = "f169b11e22de13617baa190a028a32f3493550b6"


def _freeze_doc(revisions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if revisions is None:
        revisions = {
            "dense_retriever": {
                "model": _FREEZE_EMBEDDING_MODEL,
                "revision": _FREEZE_EMBEDDING_REVISION,
                "resolved": "test mirror of the registration artifact",
            },
            # The QUALITY module's similarity embedder: a DIFFERENT
            # instrument's pin; the driver must never consume it.
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            },
        }
    return {"INSTRUMENT_REVISIONS": revisions}


def _write_freeze(path: Path, doc: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# The autouse freeze fixture lives in tests/conftest.py (suite-wide hermeticity,
# review 2026-09-17 A5): every module that builds a plan gets the tmp mirror,
# not only this one. The literals below must equal conftest's so the header
# assertions here double as the drift check between the two copies.


def test_conftest_freeze_mirror_pins_the_same_literals() -> None:
    import conftest as suite_conftest

    assert suite_conftest.FREEZE_FILE_ENV_VAR == rc.FREEZE_FILE_ENV_VAR
    assert suite_conftest.FREEZE_EMBEDDING_MODEL == _FREEZE_EMBEDDING_MODEL
    assert suite_conftest.FREEZE_EMBEDDING_REVISION == _FREEZE_EMBEDDING_REVISION
    assert suite_conftest.hermetic_freeze_doc() == _freeze_doc()


@pytest.fixture()
def plan_a(floor_table: Path) -> Dict[str, Any]:
    floor = rc.load_floor_table(floor_table)
    return rc.build_plan("a", floor, window_duration_s=300.0)


def _cells(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in plan["steps"] if s["kind"] == "cell"]


def _relaunches(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in plan["steps"] if s["kind"] == "relaunch"]


def _tiny_grid(**overrides: Any) -> Any:
    """A minimal legal SessionGrid (2 F1 cells on vLLM) for run/order tests."""
    kwargs: Dict[str, Any] = dict(
        session="a",
        group="A",
        model="qwen3-14b",
        f1_baselines=("B1", "B2"),
        f1_engines=("vllm",),
        f1_datasets=("squad_v2",),
        hf_oracle_cells=(),
        f2_baselines=(),
        f2_engines=("vllm",),
        f2_budgets=(),
        f2_rates=(),
        f2_dataset="qasper",
        f3_baselines=(),
        f3_engines=("vllm",),
        f3_budgets=(),
        f3_rates=(),
        f3_dataset="qasper",
        dist_cells=(),
    )
    kwargs.update(overrides)
    return rc.SessionGrid(**kwargs)


_STUB_SOURCE = """\
import json, os, sys
with open(os.environ["STUB_CALLS"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps({
        "argv": sys.argv[1:],
        "env": {k: v for k, v in os.environ.items() if k.startswith(("CAGE_", "VLLM_", "SGLANG_"))},
    }) + "\\n")
marker = os.environ.get("STUB_FAIL_MARKER", "")
if marker and marker in " ".join(sys.argv[1:]):
    sys.exit(1)
sys.exit(0)
"""


@pytest.fixture()
def stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A recording subprocess stub: every invocation appends a JSON line."""
    stub_path = tmp_path / "stub.py"
    stub_path.write_text(_STUB_SOURCE, encoding="utf-8")
    calls_path = tmp_path / "stub_calls.jsonl"
    monkeypatch.setenv("STUB_CALLS", str(calls_path))
    monkeypatch.delenv("STUB_FAIL_MARKER", raising=False)
    # Hermetic endpoint env (Batch 2 W2): 'run' refuses a runner override or
    # a differing launcher port, so the developer's shell must not leak in.
    for name in (*rc.API_BASE_OVERRIDE_ENVS, *rc.SHELL_PORT_ENVS):
        monkeypatch.delenv(name, raising=False)

    class Stub:
        cmd = (sys.executable, str(stub_path))

        @staticmethod
        def calls() -> List[Dict[str, Any]]:
            if not calls_path.exists():
                return []
            return [
                json.loads(line)
                for line in calls_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    return Stub


def _stub_plan(grid: Any, floor_path: Path, stub_cmd) -> Dict[str, Any]:
    """Build a plan against a synthetic grid with stubbed runner + launchers."""
    floor = rc.load_floor_table(floor_path)
    orig = rc.SESSION_GRIDS
    rc.SESSION_GRIDS = {grid.session: grid}
    try:
        return rc.build_plan(
            grid.session,
            floor,
            window_duration_s=60.0,
            runner_cmd=stub_cmd,
            launcher_cmds={"vllm": stub_cmd, "sglang": stub_cmd},
        )
    finally:
        rc.SESSION_GRIDS = orig


def _run_root(tmp_path: Path) -> Path:
    # results/<campaign>/<session>/<run_id> — session dir 'a' matches the plan.
    root = tmp_path / "results" / "camp" / "a" / "run-001"
    root.parent.mkdir(parents=True, exist_ok=True)
    return root


def _complete_cell(root: Path, step: Dict[str, Any]) -> None:
    """Materialize a cell as complete: windows[1..N] each with metrics.json."""
    cell_dir = root / "cells" / step["row_key"]
    for n in range(1, step["windows"] + 1):
        wdir = cell_dir / f"window_{step['dataset']}-{n:02d}"
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "metrics.json").write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------------------
# Plan enumeration — the registered session-'a' grid, integer-pinned
# ---------------------------------------------------------------------------


class TestPlanCountsSessionA:
    """§6.1/§6.4/§6.8/§7.6.1/D5#5 densities, derived independently and
    pinned exactly.

    F2 grid points per engine (W4.3): the §6.1 factorial 5r × 6λ = 30, plus
    the §6.4 fine overlay 7r × 2λ = 14, minus the overlap (the 5 factorial
    r's × the 2 fine rates {0.85, 1.05}, both already factorial members)
    = 10 ⇒ 30 + 14 − 10 = 34 coordinates.
    """

    def test_total_counts(self, plan_a):
        # F1: 12 baselines × 2 engines × 4 datasets              =  96
        #   + ADR-0106 B12 ladder: B12 is 2 rungs (1400, 700), so
        #     its 2 engines × 4 datasets = 8 cells become 16      =  +8
        # F1 HF oracle: B3×4 + {B1,B2,B6}×2                      =  10
        # F2 qasper: 5 FRESH × 2 engines × 34 coordinates        = 340
        #   (old pin 300 = 5 × 2 × 30, pre-§6.4-overlay)
        # F2 ruler (D5#5, W4.4): 1 (B1) × 2 engines × 34 × 4 tasks = 272
        # F3: 7 REUSE × 2 engines × 3 budgets × 3 rates          = 126
        #   + B12 ladder: 2 engines × 3 × 3 = 18 cells become 36 = +18
        # ⇒ 104 + 10 + 340 + 272 + 144 = 870
        #   (old pin 844, pre-ADR-0106; older 532)
        assert plan_a["counts"]["cells"] == 870
        # 3 windows per grid point (D6 §6.3) ⇒ 870 × 3 = 2610
        #   (old 2532 = 844 × 3; older 1596)
        assert plan_a["counts"]["windows"] == 2610
        # Relaunch boundaries = distinct EXECUTABLE serving configs
        # (engine, prefix, budget, kv_dtype, connector); hf is in-process (0):
        #   vllm:   F1 {plain, fp8·B10, lmcache·B8}                   =  3
        #           F3 3 budgets × {plain, fp8, lmcache}              =  9
        #           F2 7 budgets, all plain (FRESH set has no lever)  =  7
        #             (5 factorial + the 2 §6.4 fine-only levels; the
        #              ruler steps REUSE the qasper boundaries — dataset
        #              is not a serving-config dimension)
        #   sglang: F1 {plain, fp8}   (B8 BLOCKED: no connector knob) =  2
        #           F3 3 budgets × {plain, fp8}                       =  6
        #           F2 7 budgets, plain                               =  7
        #   ADR-0103 (corpus-fresh B4 served prefix OFF by relaunch):
        #           F1 B4 = the budget-free prefix-OFF plain config, new
        #           on EACH engine                                    = +1 ×2
        #           F3 B4 = prefix-OFF plain at r ∈ {1.0, 0.5, 0.25}, which
        #           is IDENTICAL to the F2 plain-OFF configs at those r
        #           (F3 budgets ⊂ F2 budgets; family is not a serving
        #           dimension), so they share F2's boundaries       = +0
        # ⇒ (19 + 1) + (15 + 1) = 36  (old pin 34 = 19 + 15, pre-ADR-0103;
        #    older 30 = 17 + 13, pre-fine-grid)
        assert plan_a["counts"]["relaunches"] == 36
        # Blocked: retr-store (B8) on sglang — the frozen launcher has no
        # KV-store connector knob: F1 4 datasets + F3 3×3 = 13. Serving them
        # connector-free would duplicate plain rag under a B8 label.
        #   + ADR-0106 B12 rung cells with NO query manifest registered for
        #     their dataset (this fixture passes none): a rung serves only
        #     from a manifest carrying the ladder, so all 52 B12 cells
        #     (F1 2 eng × 4 ds × 2 rungs = 16; F3 2 eng × 9 × 2 = 36) are
        #     blocked_on the missing manifest, never silently run through
        #     the non-manifest fallback that DROPS out-of-corpus queries.
        # ⇒ 13 + 52 = 65  (old pin 13, pre-manifest-registration)
        assert plan_a["counts"]["blocked"] == 65

    def test_blocked_cells_are_sglang_retr_store_plus_unmanifested_b12(self, plan_a):
        blocked = [s for s in _cells(plan_a) if s["blocked_on"]]
        assert len(blocked) == 65
        b8 = [s for s in blocked if s["baseline"] == "B8"]
        b12 = [s for s in blocked if s["baseline"] == "B12"]
        assert len(b8) == 13 and len(b12) == 52
        assert len(b8) + len(b12) == len(blocked)
        for s in b8:
            assert s["cellspec"]["arm"] == "retr-store"
            assert s["cellspec"]["engine"] == "sglang"
            assert s["blocked_on"] == rc.SGLANG_STORE_BLOCKED_ON
        for s in b12:
            assert s["cellspec"]["arm"] == "corpus-trunc"
            assert s["blocked_on"] == rc.trunc_manifest_blocked_on(s["dataset"])
            assert f"query-manifest:{s['dataset']}" in s["blocked_on"]
        assert sorted(plan_a["blocked_row_keys"]) == sorted(
            s["row_key"] for s in blocked
        )
        # and NO relaunch step ever launches the unrealizable config
        for s in _relaunches(plan_a):
            assert not (s["engine"] == "sglang" and s["connector"] is not None)

    def test_per_family_counts(self, plan_a):
        cells = _cells(plan_a)
        by = {}
        for s in cells:
            eng = s["cellspec"]["engine"]
            key = "F1-hf" if (s["family"] == "F1" and eng == "hf") else s["family"]
            by[key] = by.get(key, 0) + 1
        # F2 = 340 qasper (5 FRESH × 2 eng × 34 coords) + 272 ruler
        # (1 × 2 eng × 34 coords × 4 tasks) = 612.
        # ADR-0106 B12 ladder (2 rungs): F1 96 + 8 = 104 (B12's 2 eng × 4
        # datasets doubled), F3 126 + 18 = 144 (B12's 2 eng × 3 × 3 doubled);
        # the hf oracle slice and F2 carry no B12 (old pins F1 96, F3 126).
        assert by == {"F1": 104, "F1-hf": 10, "F2": 612, "F3": 144}
        windows = {k: 3 * v for k, v in by.items()}
        # Old window pins: F2 900 (= 300 cells × 3, pre-§6.4/pre-RULER),
        # F3 378 (= 126 × 3), F1 288 (= 96 × 3). New F2 = 612 × 3
        # = (340 qasper + 272 ruler) × 3 = 1020 + 816 = 1836; F1 = 104 × 3
        # = 312; F3 = 144 × 3 = 432 (ADR-0106 ladder).
        assert windows == {"F1": 312, "F1-hf": 30, "F2": 1836, "F3": 432}

    def test_hf_oracle_exact_reduced_set(self, plan_a):
        hf = [s for s in _cells(plan_a) if s["cellspec"]["engine"] == "hf"]
        got = {(s["baseline"], s["dataset"]) for s in hf}
        want = {("B3", d) for d in ("squad_v2", "hotpotqa", "musique", "qasper")}
        want |= {(b, d) for b in ("B1", "B2", "B6") for d in ("squad_v2", "qasper")}
        assert got == want
        assert len(hf) == 10

    def test_hf_never_under_pressure(self, plan_a):
        # Structural (§7.3): the oracle is F1-only; a pressure hf cell would
        # already be refused by CellSpec, and none may be enumerated.
        for s in _cells(plan_a):
            if s["cellspec"]["engine"] == "hf":
                assert s["family"] == "F1"
                assert s["serving"] is None  # in-process: no relaunch boundary
        for s in _relaunches(plan_a):
            assert s["engine"] in ("vllm", "sglang")

    def test_f2_grid_values(self, plan_a):
        f2 = [s for s in _cells(plan_a) if s["family"] == "F2"]
        # W4.3: the §6.4 fine overlay adds budget levels {1.25, 0.375}.
        assert {s["cellspec"]["budget_r"] for s in f2} == {
            1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25,
        }
        assert {s["cellspec"]["rate_frac"] for s in f2} == {0.5, 0.7, 0.85, 0.95, 1.05, 1.2}
        # The fine-ONLY levels run at EXACTLY the two §6.4 chassis rates —
        # never the other four factorial fractions.
        for s in f2:
            if s["cellspec"]["budget_r"] in (1.25, 0.375):
                assert s["cellspec"]["rate_frac"] in (0.85, 1.05)
        # qasper carries the FRESH set; ruler carries the registered B1-only
        # pairing (D5#5 conservative pin — see SESSION_GRIDS['a']).
        assert {s["baseline"] for s in f2 if s["dataset"] == "qasper"} == {
            "B1", "B5", "B6", "B9", "B11",
        }
        assert {s["baseline"] for s in f2 if s["dataset"] == "ruler"} == {"B1"}
        assert {s["dataset"] for s in f2} == {"qasper", "ruler"}
        # F2 is THE prefix-OFF family; everything else serves ON.
        assert all(s["serving"]["prefix_mode"] == "OFF" for s in f2)

    def test_f3_grid_values(self, plan_a):
        f3 = [s for s in _cells(plan_a) if s["family"] == "F3"]
        assert {s["cellspec"]["budget_r"] for s in f3} == {1.0, 0.5, 0.25}
        assert {s["cellspec"]["rate_frac"] for s in f3} == {0.85, 0.95, 1.05}
        assert {s["baseline"] for s in f3} == {"B2", "B3", "B4", "B7", "B8", "B10", "B12"}
        # F3 serves prefix ON, except corpus-fresh (B4), which ADR-0103
        # serves prefix OFF by relaunch in EVERY family (its REUSE-bit family
        # carriage beside B3 is unchanged).
        for s in f3:
            want = "OFF" if s["baseline"] == "B4" else "ON"
            assert s["serving"]["prefix_mode"] == want, s["row_key"]


# ---------------------------------------------------------------------------
# Ordering / relaunch minimality
# ---------------------------------------------------------------------------


class TestOrdering:
    @staticmethod
    def _config_of(serving):
        return (
            serving["engine"],
            serving["prefix_mode"],
            serving["budget_r"],
            serving["kv_dtype"],
            serving["connector"],
        )

    def test_relaunches_equal_distinct_serving_configs(self, plan_a):
        # Minimality: with cells grouped by config, relaunch count == the
        # number of distinct EXECUTABLE configs — the theoretical minimum
        # (blocked cells launch nothing; kv_dtype/connector ARE config dims).
        configs = set()
        for s in _cells(plan_a):
            if s["serving"] is not None and not s["blocked_on"]:
                configs.add(self._config_of(s["serving"]))
        # 36 = the 30 pre-fine configs + the 2 §6.4 fine-only F2 budget
        # levels × 2 engines + the ADR-0103 budget-free prefix-OFF plain
        # config for F1 B4 × 2 engines (F3 B4 rides the F2 plain-OFF
        # boundaries at the same r; see
        # TestPlanCountsSessionA.test_total_counts).
        assert len(_relaunches(plan_a)) == len(configs) == 36

    def test_every_cell_runs_under_its_preceding_relaunch(self, plan_a):
        current = None
        for s in plan_a["steps"]:
            if s["kind"] == "relaunch":
                current = (
                    s["engine"],
                    s["prefix_mode"],
                    s["budget_r"],
                    s["kv_dtype"],
                    s["connector"],
                )
                continue
            if s["serving"] is None:
                continue  # hf oracle: in-process
            if s["blocked_on"]:
                continue  # never runs — no serving-boundary claim to check
            assert current == self._config_of(
                s["serving"]
            ), f"cell {s['row_key']} would run under serving config {current}"

    def test_corpus_fresh_prefix_off_rides_f2_boundaries(self, plan_a):
        # ADR-0103 minimality: every B4 F3 cell's serving config is one of
        # the F2 plain prefix-OFF configs at the same r (F3 budgets are a
        # subset of F2 budgets), so B4 F3 adds NO relaunch; only the
        # budget-free F1 B4 config is new (one per engine).
        f2_configs = {
            self._config_of(s["serving"])
            for s in _cells(plan_a)
            if s["family"] == "F2"
        }
        f1_b4_configs = set()
        for s in _cells(plan_a):
            if s["baseline"] != "B4":
                continue
            cfg = self._config_of(s["serving"])
            if s["family"] == "F3":
                assert cfg in f2_configs, s["row_key"]
            else:
                assert s["family"] == "F1"
                assert cfg not in f2_configs, s["row_key"]
                f1_b4_configs.add(cfg)
        assert f1_b4_configs == {
            ("vllm", "OFF", None, None, None),
            ("sglang", "OFF", None, None, None),
        }

    def test_rate_changes_never_relaunch(self, floor_table):
        # Synthetic small grid: 1 F1 cell + F2 = 1 baseline × 2 budgets × 2
        # rates + F3 = 1 baseline × 1 budget × 2 rates, all on vLLM.
        grid = _tiny_grid(
            f1_baselines=("B1",),
            f2_baselines=("B1",),
            f2_budgets=(1.0, 0.5),
            f2_rates=(0.85, 1.05),
            f3_baselines=("B3",),
            f3_budgets=(0.5,),
            f3_rates=(0.85, 1.05),
        )
        plan = _stub_plan(grid, floor_table, ("stub",))
        # distinct configs: F1 plain + F3 r=0.5 + F2 r∈{0.5,1.0} = 4 — the 2
        # rate levels per budget must NOT add boundaries (rate is client-side).
        assert len(_relaunches(plan)) == 4
        assert len(_cells(plan)) == 1 + 4 + 2

    #: charter-derived KV arithmetic for the anchor model (cache_budget
    #: MODEL_KV: 2 (K,V) × 40 layers × 8 KV heads × 128 head_dim × 2 B BF16)
    #: — restated INDEPENDENTLY here so the pin never tautologically re-calls
    #: the planner it is checking.
    _QWEN3_14B_KV_BYTES_PER_TOKEN = 2 * 40 * 8 * 128 * 2  # = 163_840

    def test_budget_env_on_relaunch_steps(self, plan_a):
        # The launcher env carries plan_budget's PRIMARY knob per engine —
        # frozen contract (CAGE_KV_BUDGET_BYTES / CAGE_SGLANG_MAX_TOTAL_TOKENS)
        # plus the launch levers (KV dtype / connector), plus the uniform
        # per-session max_model_len (backlog A10; both engines read
        # VLLM_MAX_MODEL_LEN), plus the launcher port env (Batch 2 W2, from
        # the ONE table the cells' --api-base derives from), and nothing else.
        for s in _relaunches(plan_a):
            env = dict(s["env"])
            assert env.pop("VLLM_MAX_MODEL_LEN") == "32768"
            port_env, port = {
                "vllm": ("VLLM_PORT", "8000"), "sglang": ("SGLANG_PORT", "30000"),
            }[s["engine"]]
            assert env.pop(port_env) == port
            # launch levers ride the documented launcher env vars
            if s["kv_dtype"] == "fp8":
                if s["engine"] == "vllm":
                    assert env.pop("VLLM_KV_CACHE_DTYPE") == "fp8"
                else:
                    # the SGLang launcher's own documented fp8 token
                    assert env.pop("SGLANG_KV_CACHE_DTYPE") == "fp8_e5m2"
            if s["connector"] == "lmcache":
                assert s["engine"] == "vllm"  # sglang store cells are BLOCKED
                # verbatim run_kv_store.sh:123 — never re-derived
                assert env.pop("VLLM_KV_TRANSFER_CONFIG") == (
                    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
                )
            if s["budget_r"] is None:
                assert env == {}, "budget-free relaunch must carry no budget env"
                continue
            budget_bytes = int(s["budget_r"] * _ANCHOR_DEMAND)  # floor(r×D), exact here
            if s["engine"] == "vllm":
                # the bytes knob is dtype-independent: SAME byte budget for
                # every config at the same r (iso-bytes anchor, §6.5)
                assert env == {"CAGE_KV_BUDGET_BYTES": str(budget_bytes)}
            else:
                # SGLang's token dial: tokens = floor(r×D) // (KV bytes/token
                # AT THE SERVED DTYPE) — an fp8 server stores half the bytes
                # per token (charter P5 factor 0.5), so bf16 arithmetic would
                # realize only half the byte budget on B10's relaunches.
                per_token = self._QWEN3_14B_KV_BYTES_PER_TOKEN
                if s["kv_dtype"] == "fp8":
                    per_token //= 2
                assert env == {
                    "CAGE_SGLANG_MAX_TOTAL_TOKENS": str(budget_bytes // per_token)
                }
            assert "--no-prefix-cache" in s["argv"] or s["prefix_mode"] == "ON"


# ---------------------------------------------------------------------------
# The identity seam (plan → runner) and rate honesty
# ---------------------------------------------------------------------------


class TestCellSteps:
    def test_identity_env_roundtrips_through_derive_cell_spec(self, plan_a):
        # The runner derives its cell tuple from argv + CAGE_CELL_* env
        # (campaign_session seam). Every sampled step must re-mint the SAME
        # row key the plan claims — or driver and runner write different cells.
        cells = _cells(plan_a)
        for s in cells[::17] + [cells[0], cells[-1]]:
            argv = s["argv"]

            def val(flag: str, argv=argv) -> str:
                return argv[argv.index(flag) + 1]

            derived = derive_cell_spec(
                baseline=val("--baseline"),
                baseline_label=val("--baseline-label"),
                backend=val("--backend"),
                model=val("--model"),
                env=s["env"],
            )
            assert derived.to_row_key() == s["row_key"]

    def test_pressure_cells_carry_open_loop_flags_and_rate(self, plan_a):
        for s in _cells(plan_a):
            argv = s["argv"]
            if s["family"] in ("F2", "F3"):
                assert "--workload-mode" in argv and "open_loop" in argv
                assert "--rate" in argv and "--duration-s" in argv
                lam = s["lambda_star_pred_rps"]
                frac = s["cellspec"]["rate_frac"]
                assert s["offered_rate_rps"] == pytest.approx(lam * frac)
                assert float(argv[argv.index("--rate") + 1]) == pytest.approx(
                    s["offered_rate_rps"], rel=1e-5
                )
            else:
                # F1 is sub-pressure: no open-loop flags, no fabricated rate.
                assert "--workload-mode" not in argv
                assert s["offered_rate_rps"] is None
                assert s["rate_basis"] is None
                # absence stays absence: no budget/rate identity env keys
                assert "CAGE_CELL_BUDGET_R" not in s["env"]
                assert "CAGE_CELL_RATE_FRAC" not in s["env"]

    def test_lambda_compute_null_labels_rates_pending_calibration(self, plan_a):
        pressure = [s for s in _cells(plan_a) if s["family"] in ("F2", "F3")]
        assert pressure
        assert all(
            s["rate_basis"] == "kv-bound-only [pending calibration]" for s in pressure
        )

    def test_calibrated_lambda_compute_changes_basis(self, tmp_path):
        path = tmp_path / "ft_cal.json"
        path.write_text(
            json.dumps(_floor_table_doc(lambda_compute=1.0)), encoding="utf-8"
        )
        plan = rc.build_plan("a", rc.load_floor_table(path), window_duration_s=300.0)
        pressure = [s for s in _cells(plan) if s["family"] in ("F2", "F3")]
        assert all("pending calibration" not in s["rate_basis"] for s in pressure)


# ---------------------------------------------------------------------------
# Cell BEHAVIOR realization (the verified T1.2 blocker): argv + launch env
# must make every baseline BEHAVE as its arm, not merely be labeled as it
# ---------------------------------------------------------------------------


def _argv_value(step, flag):
    argv = step["argv"]
    assert flag in argv, f"{step['baseline']} argv lacks {flag}: {argv}"
    return argv[argv.index(flag) + 1]


def _by_baseline(plan, family, engine, dataset=None):
    out = {}
    for s in _cells(plan):
        if s["family"] != family or s["cellspec"]["engine"] != engine:
            continue
        if dataset is not None and s["dataset"] != dataset:
            continue
        out.setdefault(s["baseline"], []).append(s)
    return out


class TestCorpusFreshPrefixOff:
    """ADR-0103 (owner decision 2026-09-16): corpus-fresh (B4) is served with
    the engine prefix cache OFF through a per-arm RELAUNCH, uniformly on every
    engine, in every family. Its family carriage (REUSE bit, rides F3 beside
    B3) is unchanged. Before ADR-0103 the runner's ``no_cache`` token only
    labeled telemetry, so B4 and B3 were served by the SAME prefix-ON server:
    the mislabeled-duplicate failure class."""

    @staticmethod
    def _b3_b4(plan, family, engine):
        cells = _by_baseline(plan, family, engine)
        return cells["B3"], cells["B4"]

    @pytest.mark.parametrize("engine", ["vllm", "sglang"])
    @pytest.mark.parametrize("family", ["F1", "F3"])
    def test_b4_serves_prefix_off_and_b3_on(self, plan_a, family, engine):
        b3, b4 = self._b3_b4(plan_a, family, engine)
        assert b3 and b4
        assert all(s["serving"]["prefix_mode"] == "ON" for s in b3)
        assert all(s["serving"]["prefix_mode"] == "OFF" for s in b4)
        # Serving config differs ONLY by the prefix mode: same engine,
        # budget, and (absent) launch levers.
        for s in b4:
            assert s["serving"]["kv_dtype"] is None
            assert s["serving"]["connector"] is None
            assert s["serving"]["topology"] == "single"
        assert {s["serving"]["budget_r"] for s in b3} == {
            s["serving"]["budget_r"] for s in b4
        }

    def test_b4_family_carriage_unchanged(self, plan_a):
        # The REUSE bit still carries B4 in F3 beside B3, never in F2.
        f3 = {s["baseline"] for s in _cells(plan_a) if s["family"] == "F3"}
        f2 = {s["baseline"] for s in _cells(plan_a) if s["family"] == "F2"}
        assert {"B3", "B4"} <= f3
        assert "B4" not in f2
        for s in _cells(plan_a):
            if s["baseline"] == "B4":
                assert s["cellspec"]["arm"] == "corpus-fresh"
                assert s["family"] in ("F1", "F3")

    def test_b4_relaunch_argv_carries_no_prefix_cache(self, plan_a):
        # Every B4 cell runs under a relaunch whose argv disables the
        # engine prefix cache (launchers map --no-prefix-cache to
        # --no-enable-prefix-caching / --disable-radix-cache).
        current = None
        for s in plan_a["steps"]:
            if s["kind"] == "relaunch":
                current = s
                continue
            if s["baseline"] != "B4":
                continue
            assert current is not None
            assert current["prefix_mode"] == "OFF"
            assert "--no-prefix-cache" in current["argv"]
            assert current["engine"] == s["cellspec"]["engine"]

    def test_hf_cells_unaffected(self, plan_a):
        # The in-process oracle has no server: serving stays None and the
        # B4 rule never enumerates an hf cell (the reduced slice has none).
        hf = [s for s in _cells(plan_a) if s["cellspec"]["engine"] == "hf"]
        assert hf and all(s["serving"] is None for s in hf)
        assert all(s["baseline"] != "B4" for s in hf)

    def test_rule_is_the_named_constant(self):
        # Doctrine: the knob is a named module constant citing its ADR, and
        # _prefix_off is the ONE rule the sort key, the serving-config
        # identity, the relaunch step and the cell step all consult.
        assert rc.PREFIX_OFF_ARMS == frozenset({"corpus-fresh"})
        assert "ADR-0103" in rc._prefix_off.__doc__

    def test_rule_surfaces_in_the_plan_header(self, plan_a, plan_b):
        # Like the ADR-0102 constants: the operator reviews the registered
        # prefix-OFF arms in the header, not only inside the cell list.
        for plan in (plan_a, plan_b):
            knobs = plan["behavior_knobs"]
            assert knobs["prefix_off_arms"] == ["corpus-fresh"]
            assert knobs["prefix_off_adr"] == "ADR-0103"


class TestBehaviorRealization:
    def test_corpus_arms_carry_the_corpus_block(self, plan_a):
        # B3/B4/B10 serve the SHARED corpus-as-prefix block
        # (run_prefix_envelope.sh cag_true_*: --corpus-prefix-budget); B12
        # serves the TRUNCATED block at ITS rung (ADR-0106 ladder); the
        # budget delta IS the B12-vs-B3 contrast ("store less than you know").
        knobs = plan_a["behavior_knobs"]
        full = knobs["corpus_prefix_budget_tokens"]
        rungs = tuple(knobs["corpus_trunc_budgets"])
        assert all(0 < r < full for r in rungs)
        for s in _cells(plan_a):
            arm = s["cellspec"]["arm"]
            if arm in ("corpus-reuse", "corpus-fresh", "corpus-comp"):
                assert _argv_value(s, "--corpus-prefix-budget") == str(full)
                assert "--corpus-rung" not in s["argv"]
            elif arm == "corpus-trunc":
                rung = s["cellspec"]["corpus_budget_tokens"]
                assert rung in rungs
                assert _argv_value(s, "--corpus-prefix-budget") == str(rung)
                assert _argv_value(s, "--corpus-rung") == str(rung)
            else:
                # gold/retrieval arms never serve a corpus block
                assert "--corpus-prefix-budget" not in s["argv"]
                assert "--corpus-rung" not in s["argv"]

    def test_reranker_ablated_exactly_once(self, plan_a):
        # §7.1 ranking rule: B5 = dense WITHOUT the reranker, B6 = the pinned
        # cross-encoder; B7-B9/B11 inherit the RANKED pipeline. The reranker
        # is EXPLICIT in every retrieval argv — inheriting the runner default
        # silently is how the pre-registered ablation collapsed (B5 ≡ B6).
        for s in _cells(plan_a):
            retriever = s["cellspec"]["retriever"]
            if retriever == "none":
                assert "--reranker-model" not in s["argv"]
                continue
            assert _argv_value(s, "--retriever") == "dense"
            if s["baseline"] == "B5":
                assert retriever == "dense"
                assert _argv_value(s, "--reranker-model") == "none"
                # ADR-0104: B5 serves the dense top-k UNRANKED, no pool.
                assert "--rerank-pool" not in s["argv"]
            else:
                assert retriever == "rerank"
                assert _argv_value(s, "--reranker-model") == rc.RERANKER_MODEL
                # ADR-0104: the ranked pipeline reranks a POOL of RERANK_POOL
                # candidates and serves --top-k (the runner default, 3).
                assert _argv_value(s, "--rerank-pool") == str(rc.RERANK_POOL)

    def test_rerank_pool_is_the_named_constant(self, plan_a, plan_b):
        # ADR-0104 (owner decision 2026-09-16): pool 10, reranked whole, top 3
        # served. The knob is a named module constant citing its ADR, pinned
        # in the argv (never a runner default) and reviewable in the header.
        assert rc.RERANK_POOL == 10
        assert rc.RETRIEVER_ARGV["rerank"] == (
            "--retriever", "dense",
            "--reranker-model", rc.RERANKER_MODEL,
            "--rerank-pool", "10",
        )
        assert rc.RETRIEVER_ARGV["dense"] == (
            "--retriever", "dense", "--reranker-model", "none",
        )
        for plan in (plan_a, plan_b):
            knobs = plan["behavior_knobs"]
            assert knobs["rerank_pool"] == 10
            assert knobs["rerank_pool_adr"] == "ADR-0104"

    def test_retr_comp_compresses_retrieved_context(self, plan_a):
        # run_compression.sh: without --context-source retrieved the arm
        # compresses GOLD context (CAG+compression mislabeled as
        # RAG+compression) — the Phase-2 confound. Only retr-comp needs it.
        for s in _cells(plan_a):
            if s["cellspec"]["arm"] == "retr-comp":
                assert _argv_value(s, "--context-source") == "retrieved"
            else:
                assert "--context-source" not in s["argv"]

    def test_retr_trunc_rank_truncates(self, plan_a):
        kept = plan_a["behavior_knobs"]["retr_trunc_kept_docs"]
        for s in _cells(plan_a):
            if s["cellspec"]["arm"] == "retr-trunc":
                assert _argv_value(s, "--max-context-docs") == str(kept)
            else:
                assert "--max-context-docs" not in s["argv"]

    def test_corpus_comp_gets_fp8_at_launch_and_recorded(self, plan_a):
        # run_compression.sh: fp8 KV is a server LAUNCH lever
        # (VLLM_KV_CACHE_DTYPE=fp8); run_experiment --kv-cache-dtype is
        # record-only provenance. Both must be present, and the serving
        # config must carry the dtype so B10 rides its OWN relaunch.
        b10 = [s for s in _cells(plan_a) if s["baseline"] == "B10"]
        assert b10
        for s in b10:
            assert s["serving"]["kv_dtype"] == "fp8"
            assert _argv_value(s, "--kv-cache-dtype") == "fp8"
        others = [s for s in _cells(plan_a) if s["baseline"] != "B10"]
        assert all("--kv-cache-dtype" not in s["argv"] for s in others)
        assert all(
            s["serving"] is None or s["serving"]["kv_dtype"] is None for s in others
        )

    def test_retr_store_gets_lmcache_connector_serving(self, plan_a):
        # B8's ONLY serving delta vs B6 is the connector (run_kv_store.sh:
        # same retrieval, same prompts) — so the connector MUST be a serving
        # -config dimension or B8 is a byte-identical duplicate of B6.
        b8 = [s for s in _cells(plan_a) if s["baseline"] == "B8"]
        assert b8
        for s in b8:
            assert s["serving"]["connector"] == "lmcache"
            if s["cellspec"]["engine"] == "vllm":
                assert s["blocked_on"] is None
            else:
                assert s["blocked_on"] == rc.SGLANG_STORE_BLOCKED_ON
        assert all(
            s["serving"] is None or s["serving"]["connector"] is None
            for s in _cells(plan_a)
            if s["baseline"] != "B8"
        )

    def test_no_two_baselines_share_behavior_in_a_grid_slice(self, plan_a):
        # THE anti-mislabeling rule, stated pairwise: within one grid slice
        # (family, engine, dataset, budget, rate), two different baselines
        # sharing the same behavior argv (label stripped — labels are
        # identity, not behavior) AND the same serving config would be
        # byte-identical duplicate cells under different names — the exact
        # blocker this repair closes (B3≡B2, B5≡B6, B8≡B6, B10≡B3, B11≡B6,
        # B12≡B2 before the fix).
        def signature(step):
            argv = list(step["argv"])
            i = argv.index("--baseline-label")
            del argv[i : i + 2]
            serving = step["serving"]
            return (
                tuple(argv),
                None if serving is None else tuple(sorted(serving.items())),
            )

        slices = {}
        for s in _cells(plan_a):
            key = (
                s["family"],
                s["cellspec"]["engine"],
                s["dataset"],
                s["cellspec"]["budget_r"],
                s["cellspec"]["rate_frac"],
            )
            slices.setdefault(key, []).append(s)
        for key, steps in slices.items():
            seen = {}
            for s in steps:
                sig = signature(s)
                assert sig not in seen, (
                    f"slice {key}: {s['baseline']} and {seen[sig]} share argv+serving "
                    "— mislabeled duplicate cells"
                )
                seen[sig] = s["baseline"]


# ---------------------------------------------------------------------------
# ADR-0106 / backlog A4: the B12 corpus-truncation ladder (one cell per rung)
# ---------------------------------------------------------------------------


class TestCorpusTruncLadder:
    """ADR-0106 (owner decision 2026-09-16, charter §7.7(d)): B12 is a
    descending corpus-budget ladder whose 2,800 point is B3's own cell; the
    registered rungs (1400, 700) each enumerate ONE corpus-trunc cell
    wherever B12 is carried (F1 and F3), the rung rides the identity seam as
    CAGE_CELL_CORPUS_BUDGET and the argv as the explicit --corpus-rung."""

    LADDER = (1400, 700)

    @staticmethod
    def _b12(plan):
        return [s for s in _cells(plan) if s["cellspec"]["arm"] == "corpus-trunc"]

    def test_registered_ladder_and_header(self, plan_a, plan_b):
        for session in ("a", "b"):
            assert rc.SESSION_GRIDS[session].corpus_trunc_budgets == self.LADDER
        for plan in (plan_a, plan_b):
            knobs = plan["behavior_knobs"]
            assert "corpus_trunc_budget_tokens" not in knobs  # the single-rung knob is gone
            assert knobs["corpus_trunc_budgets"] == list(self.LADDER)
            # The full ladder as the operator reads it: B3's budget first.
            assert knobs["corpus_trunc_ladder"] == [
                knobs["corpus_prefix_budget_tokens"], *self.LADDER
            ]
            assert knobs["corpus_trunc_adr"] == "ADR-0106"

    @pytest.mark.parametrize("family", ["F1", "F3"])
    @pytest.mark.parametrize("engine", ["vllm", "sglang"])
    def test_one_cell_per_rung_in_every_slice(self, plan_a, family, engine):
        # Group the B12 cells of one (family, engine) by workload slice: each
        # slice carries exactly the registered rungs, each once.
        slices = {}
        for s in self._b12(plan_a):
            if s["family"] != family or s["cellspec"]["engine"] != engine:
                continue
            key = (s["dataset"], s["cellspec"]["budget_r"], s["cellspec"]["rate_frac"])
            slices.setdefault(key, []).append(s["cellspec"]["corpus_budget_tokens"])
        expected_slices = 4 if family == "F1" else 9  # 4 datasets; 3 r × 3 λ
        assert len(slices) == expected_slices
        for key, rungs in slices.items():
            assert tuple(rungs) == self.LADDER, key  # descending, no duplicates

    def test_total_b12_cells_session_a_and_b(self, plan_a, plan_b):
        # a: F1 2 eng × 4 ds × 2 rungs = 16; F3 2 eng × 9 × 2 = 36 ⇒ 52.
        # b: identical F1/F3 carriage (no B12 in the hf slice, F2 or DIST).
        assert len(self._b12(plan_a)) == 52
        assert len(self._b12(plan_b)) == 52
        for plan in (plan_a, plan_b):
            for s in self._b12(plan):
                assert s["cellspec"]["engine"] != "hf"
                assert s["family"] in ("F1", "F3")

    def test_rung_rides_identity_env_argv_and_row_key(self, plan_a):
        for s in _cells(plan_a):
            rung = s["cellspec"].get("corpus_budget_tokens")
            if s["cellspec"]["arm"] == "corpus-trunc":
                assert rung in self.LADDER
                assert s["env"]["CAGE_CELL_CORPUS_BUDGET"] == str(rung)
                assert s["row_key"].endswith(f"|cb{rung}")
                assert _argv_value(s, "--corpus-rung") == str(rung)
                assert _argv_value(s, "--corpus-prefix-budget") == str(rung)
            else:
                assert rung is None
                assert "CAGE_CELL_CORPUS_BUDGET" not in s["env"]
                assert "|cb" not in s["row_key"]
                assert "--corpus-rung" not in s["argv"]

    def test_rung_cells_round_trip_through_derive_cell_spec(self, plan_a):
        # The generic seam test samples every 17th cell; the rung coordinate
        # must round-trip on EVERY B12 cell or two rungs collapse into one row.
        keys = set()
        for s in self._b12(plan_a):
            argv = s["argv"]

            def val(flag: str, argv=argv) -> str:
                return argv[argv.index(flag) + 1]

            derived = derive_cell_spec(
                baseline=val("--baseline"),
                baseline_label=val("--baseline-label"),
                backend=val("--backend"),
                model=val("--model"),
                env=s["env"],
            )
            assert derived.to_row_key() == s["row_key"]
            keys.add((s["row_key"], s["dataset"]))
        # Every rung cell is its own (row, dataset) pair: 52 cells. Distinct
        # row keys = 40: F1 keys are shared by the 4 datasets (2 engines × 2
        # rungs = 4) and F3 keys are per coordinate (2 × 9 × 2 = 36).
        assert len(keys) == 52
        assert len({k for k, _ in keys}) == 40

    def test_rungs_share_one_serving_config(self, plan_a):
        # The rung is a RUNNER-side corpus fact (the served block), never a
        # server dial: both rungs of a slice run under the same relaunch
        # boundary (relaunch count pinned at 36 in test_total_counts).
        by_slice = {}
        for s in self._b12(plan_a):
            key = (s["family"], s["cellspec"]["engine"], s["dataset"],
                   s["cellspec"]["budget_r"], s["cellspec"]["rate_frac"])
            by_slice.setdefault(key, []).append(s["serving"])
        for key, servings in by_slice.items():
            assert len(servings) == 2 and servings[0] == servings[1], key

    def test_ladder_order_is_descending_in_the_plan(self, plan_a):
        # Within a slice the 1400 rung precedes the 700 rung (descending
        # budgets, as the charter reads the ladder), deterministically.
        seen = {}
        for s in self._b12(plan_a):
            key = (s["family"], s["cellspec"]["engine"], s["dataset"],
                   s["cellspec"]["budget_r"], s["cellspec"]["rate_frac"])
            seen.setdefault(key, []).append(s["index"])
        for key, indices in seen.items():
            assert indices == sorted(indices), key

    @pytest.mark.parametrize(
        "ladder",
        [(), (2800,), (2900,), (700, 1400), (1400, 1400), (1400, 0), (1400.0, 700)],
    )
    def test_grid_refuses_bad_ladders(self, ladder):
        with pytest.raises(rc.PlanError):
            _tiny_grid(f1_baselines=("B12",), corpus_trunc_budgets=ladder)

    def test_tiny_grid_enumerates_one_cell_per_rung(self, floor_table, stub):
        grid = _tiny_grid(f1_baselines=("B3", "B12"), corpus_trunc_budgets=(1000, 500))
        plan = _stub_plan(grid, floor_table, stub.cmd)
        cells = _cells(plan)
        assert [s["baseline"] for s in cells] == ["B3", "B12", "B12"]
        assert [s["cellspec"].get("corpus_budget_tokens") for s in cells] == [None, 1000, 500]
        assert plan["counts"]["cells"] == 3
        assert plan["behavior_knobs"]["corpus_trunc_ladder"] == [2800, 1000, 500]


# ---------------------------------------------------------------------------
# Plan schema roundtrip + CLI purity
# ---------------------------------------------------------------------------


class TestPlanSchema:
    def test_cli_plan_writes_loadable_plan_and_nothing_else(self, tmp_path, floor_table):
        out = tmp_path / "plan_a.json"
        before = {p for p in tmp_path.rglob("*")}
        code = rc.main(
            [
                "plan",
                "--session",
                "a",
                "--floor-table",
                str(floor_table),
                "--window-duration-s",
                "300",
                "--out",
                str(out),
            ]
        )
        assert code == 0
        after = {p for p in tmp_path.rglob("*")}
        assert after - before == {out}, "plan must write NOTHING except --out"
        plan = rc.load_plan(out)
        assert plan["counts"]["cells"] == 870  # see TestPlanCountsSessionA (ADR-0106; old 844)
        # every row key re-mints from its embedded cellspec (never hand-built)
        for s in _cells(plan)[::50]:
            assert CellSpec.from_flat_dict(s["cellspec"]).to_row_key() == s["row_key"]

    def test_load_plan_refuses_hand_edited_row_key(self, tmp_path, plan_a):
        plan = json.loads(json.dumps(plan_a))
        cell = next(s for s in plan["steps"] if s["kind"] == "cell")
        cell["row_key"] = "hand|built|key"
        path = tmp_path / "tampered.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        with pytest.raises(rc.RunError, match="minted"):
            rc.load_plan(path)

    def test_load_plan_refuses_wrong_schema_and_missing_file(self, tmp_path):
        with pytest.raises(rc.RunError, match="not found"):
            rc.load_plan(tmp_path / "nope.json")
        bad = tmp_path / "bad.json"
        bad.write_text('{"schema": "something-else", "steps": []}', encoding="utf-8")
        with pytest.raises(rc.RunError, match="schema"):
            rc.load_plan(bad)


# ---------------------------------------------------------------------------
# Fail-closed guards at plan time
# ---------------------------------------------------------------------------


class TestPlanRefusals:
    def test_unknown_session_refuses(self, floor_table):
        with pytest.raises(rc.PlanError, match="unknown session"):
            rc.build_plan("zz", rc.load_floor_table(floor_table), window_duration_s=60.0)

    def test_known_but_unregistered_session_refuses(self, floor_table):
        # 'cd-act1' is §1 vocabulary but its grid registration is Group-C/D
        # work ('b' registered with W4.6): the driver must refuse loudly,
        # never fabricate a grid.
        with pytest.raises(rc.PlanError, match="NOT yet registered"):
            rc.build_plan(
                "cd-act1", rc.load_floor_table(floor_table), window_duration_s=60.0
            )

    def test_empty_enumeration_refuses(self):
        grid = _tiny_grid(f1_baselines=(), f1_datasets=())
        with pytest.raises(rc.PlanError, match="ZERO cells"):
            rc.enumerate_cells(grid)

    def test_missing_floor_table_refuses(self, tmp_path):
        with pytest.raises(rc.PlanError, match="not found"):
            rc.load_floor_table(tmp_path / "absent.json")

    def test_invalid_json_floor_table_refuses(self, tmp_path):
        path = tmp_path / "ft.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(rc.PlanError, match="not valid JSON"):
            rc.load_floor_table(path)

    def test_wrong_schema_floor_table_refuses(self, tmp_path):
        path = tmp_path / "ft.json"
        doc = _floor_table_doc()
        doc["schema"] = "floor-table-v0"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(rc.PlanError, match="floor-table-v1"):
            rc.load_floor_table(path)

    def test_row_missing_demand_refuses(self, tmp_path):
        doc = _floor_table_doc()
        del doc["rows"][0]["demand_bytes"]
        path = tmp_path / "ft.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(rc.PlanError, match="demand_bytes"):
            rc.load_floor_table(path)

    def test_model_mismatch_refuses(self, tmp_path):
        path = tmp_path / "ft.json"
        path.write_text(
            json.dumps(_floor_table_doc(model="llama-3.3-70b")), encoding="utf-8"
        )
        with pytest.raises(rc.PlanError, match="wrong model|model"):
            rc.build_plan("a", rc.load_floor_table(path), window_duration_s=60.0)

    def test_reduced_grid_table_missing_full_r_refuses(self, tmp_path):
        # Session 'a' F2 needs r=1.5/0.75 (§6.1 full grid); a reduced-grid
        # floor table cannot honestly serve it.
        path = tmp_path / "ft.json"
        path.write_text(
            json.dumps(_floor_table_doc(grid="reduced", r_values=(1.0, 0.5, 0.25))),
            encoding="utf-8",
        )
        with pytest.raises(rc.PlanError, match="no row for"):
            rc.build_plan("a", rc.load_floor_table(path), window_duration_s=60.0)

    def test_bad_window_duration_refuses(self, floor_table):
        floor = rc.load_floor_table(floor_table)
        for bad in (0.0, -5.0, float("nan"), float("inf")):
            with pytest.raises(rc.PlanError, match="window_duration_s"):
                rc.build_plan("a", floor, window_duration_s=bad)

    def test_cli_refusal_exit_code(self, tmp_path):
        code = rc.main(
            [
                "plan",
                "--session",
                "a",
                "--floor-table",
                str(tmp_path / "absent.json"),
                "--window-duration-s",
                "300",
            ]
        )
        assert code == 2


# ---------------------------------------------------------------------------
# DIST enumeration + 'run' refusal of blocked cells
# ---------------------------------------------------------------------------


class TestDistBlocked:
    # NOTE (T3.2): vllm DIST/pd cells are EXECUTABLE now (gated behind
    # --allow-pd; pinned in tests/test_pd_launcher.py), so the blocked-cell
    # pins here ride the still-unregistered tp overlay instead.
    def test_dist_cells_enumerated_and_blocked(self, floor_table, stub):
        grid = _tiny_grid(dist_cells=(("B3", "vllm", "tp"),))
        plan = _stub_plan(grid, floor_table, stub.cmd)
        dist = [s for s in _cells(plan) if s["family"] == "DIST"]
        assert len(dist) == 1
        assert dist[0]["blocked_on"] == rc.TP_DIST_BLOCKED_ON
        assert plan["counts"]["blocked"] == 1
        assert plan["blocked_row_keys"] == [dist[0]["row_key"]]

    def test_run_refuses_blocked_cells_before_executing(
        self, tmp_path, floor_table, stub
    ):
        grid = _tiny_grid(dist_cells=(("B3", "vllm", "tp"),))
        plan = _stub_plan(grid, floor_table, stub.cmd)
        with pytest.raises(rc.RunError, match=r"blocked.*tp-topology"):
            rc.run_plan(plan, _run_root(tmp_path))
        assert stub.calls() == [], "nothing may execute when blocked cells exist"


# ---------------------------------------------------------------------------
# 'run' on stubs — end-to-end, resume, failure continuation, sealing
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.mark.parametrize("value", ["1", "0", ""])
    def test_run_refuses_when_the_stale_index_escape_hatch_is_set(
        self, tmp_path, floor_table, stub, monkeypatch, value
    ):
        # Backlog A6 / F6 (review 2026-09-17 defect 2): _exec inherits the
        # operator's shell, so CAGE_ALLOW_STALE_INDEX would reach every
        # retrieval cell and serve a pre-prefix index under a WARNING. The
        # driver refuses on PRESENCE (any value), before the first step.
        from src.orchestration.ir import STALE_INDEX_OPT_IN_ENV

        assert rc.STALE_INDEX_OPT_IN_ENV == STALE_INDEX_OPT_IN_ENV == "CAGE_ALLOW_STALE_INDEX"
        monkeypatch.setenv(rc.STALE_INDEX_OPT_IN_ENV, value)
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        with pytest.raises(rc.RunError, match="CAGE_ALLOW_STALE_INDEX"):
            rc.run_plan(plan, _run_root(tmp_path))
        assert stub.calls() == [], "nothing may execute under the escape hatch"

    def test_end_to_end_two_cell_plan(self, tmp_path, floor_table, stub):
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 0
        calls = stub.calls()
        # 1 relaunch (vllm plain config) + 2 cells, in plan order.
        assert len(calls) == 3
        assert calls[0]["argv"][0] == "restart"  # launcher stub sees its verb
        for call, step in zip(calls[1:], _cells(plan)):
            assert call["argv"] == step["argv"][2:] + ["--campaign-root", str(root)]
            # identity env reached the subprocess (the seam, not just the plan)
            for key, value in step["env"].items():
                assert call["env"][key] == value

    def test_resume_skips_complete_windows(self, tmp_path, floor_table, stub):
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        first, second = _cells(plan)
        _complete_cell(root, first)  # 3/3 windows with metrics.json present
        assert rc.run_plan(plan, root) == 0
        cell_calls = [c for c in stub.calls() if c["argv"][0] != "restart"]
        assert len(cell_calls) == 1
        assert "--baseline-label" in cell_calls[0]["argv"]
        label = cell_calls[0]["argv"][cell_calls[0]["argv"].index("--baseline-label") + 1]
        assert label == f"{second['baseline']}_{second['cellspec']['arm']}"

    def test_partial_windows_do_not_skip(self, tmp_path, floor_table, stub):
        # 2 of 3 windows complete => the cell still runs (the runner's own
        # per-window resume fills the gap; the driver must not call it done).
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        first, _ = _cells(plan)
        cell_dir = root / "cells" / first["row_key"]
        for n in (1, 2):
            wdir = cell_dir / f"window_{first['dataset']}-{n:02d}"
            wdir.mkdir(parents=True)
            (wdir / "metrics.json").write_text("{}", encoding="utf-8")
        # window 3: dir exists but NO metrics.json — incomplete, not counted
        (cell_dir / f"window_{first['dataset']}-03").mkdir()
        rc.run_plan(plan, root)
        cell_calls = [c for c in stub.calls() if c["argv"][0] != "restart"]
        assert len(cell_calls) == 2

    def test_force_rerun_overrides_resume(self, tmp_path, floor_table, stub):
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        for step in _cells(plan):
            _complete_cell(root, step)
        assert rc.run_plan(plan, root, force_rerun=True) == 0
        cell_calls = [c for c in stub.calls() if c["argv"][0] != "restart"]
        assert len(cell_calls) == 2

    def test_failed_cell_continues_and_exits_nonzero(
        self, tmp_path, floor_table, stub, monkeypatch
    ):
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        first, second = _cells(plan)
        monkeypatch.setenv("STUB_FAIL_MARKER", f"{first['baseline']}_{first['cellspec']['arm']}")
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 1
        cell_calls = [c for c in stub.calls() if c["argv"][0] != "restart"]
        assert len(cell_calls) == 2, "execution must CONTINUE past a failed cell"
        # dataset-suffixed: F1 row keys are shared by four datasets — a bare
        # per-cell sentinel could not say WHICH dataset's pass failed.
        sentinel = root / "cells" / first["row_key"] / f".STATUS-{first['dataset']}"
        assert sentinel.is_file()
        content = sentinel.read_text(encoding="utf-8")
        assert "STATUS=failed" in content
        assert f"dataset={first['dataset']}" in content
        # dot-named on purpose: the §5 seal scope skips dot entries, so a
        # sentinel can never poison a later --seal-partial seal.
        assert sentinel.name.startswith(".")
        assert not (
            root / "cells" / second["row_key"] / f".STATUS-{second['dataset']}"
        ).exists()

    def test_stale_failed_sentinel_cleared_after_later_success(
        self, tmp_path, floor_table, stub, monkeypatch
    ):
        # A resumed run that SUCCEEDS (or finds the cell already complete)
        # must remove the old failed sentinel — a stale STATUS=failed over
        # healthy data is a false forensic record.
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        first, second = _cells(plan)
        root = _run_root(tmp_path)
        marker = f"{first['baseline']}_{first['cellspec']['arm']}"
        monkeypatch.setenv("STUB_FAIL_MARKER", marker)
        assert rc.run_plan(plan, root) == 1
        sentinel = root / "cells" / first["row_key"] / f".STATUS-{first['dataset']}"
        assert sentinel.is_file()
        # rerun with the failure gone: the cell now succeeds -> sentinel gone
        monkeypatch.delenv("STUB_FAIL_MARKER")
        assert rc.run_plan(plan, root) == 0
        assert not sentinel.exists()
        # and the skipped-complete path clears too: refail is impossible to
        # re-record when the windows are all complete on disk
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("STATUS=failed dataset=x reason=stale\n", encoding="utf-8")
        _complete_cell(root, first)
        _complete_cell(root, second)
        assert rc.run_plan(plan, root) == 0
        assert not sentinel.exists()

    def test_failed_relaunch_fails_cells_until_next_boundary(
        self, tmp_path, floor_table, stub, monkeypatch
    ):
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        monkeypatch.setenv("STUB_FAIL_MARKER", "restart")  # the relaunch fails
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 1
        cell_calls = [c for c in stub.calls() if c["argv"][0] != "restart"]
        assert cell_calls == [], "cells must not run against a failed serving config"
        for step in _cells(plan):
            assert (
                root / "cells" / step["row_key"] / f".STATUS-{step['dataset']}"
            ).is_file()

    def test_zero_executable_cells_is_an_error(self, tmp_path, floor_table, stub):
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        plan = dict(plan, steps=[s for s in plan["steps"] if s["kind"] == "relaunch"])
        with pytest.raises(rc.RunError, match="zero executable cells"):
            rc.run_plan(plan, _run_root(tmp_path))

    def test_root_session_mismatch_refuses(self, tmp_path, floor_table, stub):
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        wrong = tmp_path / "results" / "camp" / "b" / "run-001"
        wrong.parent.mkdir(parents=True)
        with pytest.raises(rc.RunError, match="session"):
            rc.run_plan(plan, wrong)

    def test_seal_only_on_fully_passed_runs(self, tmp_path, floor_table, stub):
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        # all-pass + --seal: the sealer stub is invoked with the root
        assert rc.run_plan(plan, root, seal=True, seal_cmd=stub.cmd) == 0
        assert stub.calls()[-1]["argv"] == [str(root)]

    def test_seal_skipped_on_failures_without_seal_partial(
        self, tmp_path, floor_table, stub, monkeypatch
    ):
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        first, _ = _cells(plan)
        monkeypatch.setenv("STUB_FAIL_MARKER", f"{first['baseline']}_{first['cellspec']['arm']}")
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root, seal=True, seal_cmd=stub.cmd) == 1
        # the sealer must NOT have run: no call whose argv is just the root
        assert [str(root)] not in [c["argv"] for c in stub.calls()]

    def test_seal_partial_seals_despite_failures(
        self, tmp_path, floor_table, stub, monkeypatch
    ):
        # --seal-partial is the EXPLICIT operator consent to seal a tree with
        # failed cells (their sentinels are dot-named, outside the seal
        # scope); the exit code still reports the failure.
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        first, _ = _cells(plan)
        monkeypatch.setenv("STUB_FAIL_MARKER", f"{first['baseline']}_{first['cellspec']['arm']}")
        root = _run_root(tmp_path)
        assert (
            rc.run_plan(plan, root, seal=True, seal_partial=True, seal_cmd=stub.cmd)
            == 1
        )
        assert stub.calls()[-1]["argv"] == [str(root)], "sealer must have run"

    def test_ruler_task_steps_resume_independently(
        self, tmp_path, floor_table, stub
    ):
        # W4.4 driver-resume correctness: per-task RULER steps share one
        # (row_key, dataset='ruler') window space; each claims its own
        # ordinal range (base = task_index × replications). Completing task
        # 1's range (ruler-01..03) must NOT mark task 2 (ruler-04..06) done —
        # the old flat count would have silently skipped a registered cell.
        grid = _tiny_grid(
            f1_baselines=(),
            f1_datasets=("squad_v2",),
            f2_baselines=("B1",),
            f2_budgets=(1.0,),
            f2_rates=(0.85,),
            f2_ruler_baselines=("B1",),
            f2_ruler_tasks=("niah_multikey", "qa"),
        )
        plan = _stub_plan(grid, floor_table, stub.cmd)
        cells = _cells(plan)
        # 1 qasper step + 2 per-task ruler steps, one shared row key
        assert [(s["dataset"], s["ruler_task"]) for s in cells] == [
            ("qasper", None),
            ("ruler", "niah_multikey"),
            ("ruler", "qa"),
        ]
        assert len({s["row_key"] for s in cells}) == 1
        assert [s["window_ordinal_base"] for s in cells] == [0, 0, 3]
        root = _run_root(tmp_path)
        task1 = cells[1]
        # materialize ONLY task 1's claimed range: ruler-01..03
        cell_dir = root / "cells" / task1["row_key"]
        for n in (1, 2, 3):
            wdir = cell_dir / f"window_ruler-{n:02d}"
            wdir.mkdir(parents=True)
            (wdir / "metrics.json").write_text("{}", encoding="utf-8")
        assert rc.run_plan(plan, root) == 0
        cell_calls = [c for c in stub.calls() if c["argv"][0] != "restart"]
        # qasper + task 2 ran; task 1 was skipped-complete on ITS range only
        assert len(cell_calls) == 2
        tasks_run = [
            c["argv"][c["argv"].index("--ruler-task") + 1]
            for c in cell_calls
            if "--ruler-task" in c["argv"]
        ]
        assert tasks_run == ["qa"]
        # task 2's runner invocation carried its ordinal base env
        (qa_call,) = [c for c in cell_calls if "--ruler-task" in c["argv"]]
        assert qa_call["env"]["CAGE_WINDOW_ORDINAL_BASE"] == "3"

    def test_ruler_task_failure_sentinels_are_range_scoped(
        self, tmp_path, floor_table, stub, monkeypatch
    ):
        # A task-2 failure writes a sentinel that names ITS ordinal range;
        # task 1's later success must not clear it (distinct spellings).
        grid = _tiny_grid(
            f1_baselines=(),
            f1_datasets=("squad_v2",),
            f2_baselines=("B1",),
            f2_budgets=(1.0,),
            f2_rates=(0.85,),
            f2_ruler_baselines=("B1",),
            f2_ruler_tasks=("niah_multikey", "qa"),
        )
        plan = _stub_plan(grid, floor_table, stub.cmd)
        monkeypatch.setenv("STUB_FAIL_MARKER", "--ruler-task qa")
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 1
        row_key = _cells(plan)[0]["row_key"]
        cell_dir = root / "cells" / row_key
        # base 0 steps keep the pre-W4.4 sentinel spelling; the qa step
        # (base 3 => range starts at ordinal 04) gets its own suffix.
        assert not (cell_dir / ".STATUS-qasper").exists()
        assert not (cell_dir / ".STATUS-ruler").exists()
        assert (cell_dir / ".STATUS-ruler-from-04").is_file()

    def test_skip_blocked_runs_executable_subset_loudly(
        self, tmp_path, floor_table, stub
    ):
        # --skip-blocked = explicit consent: executable cells run, blocked
        # cells execute NOTHING and are reported; a plain --seal stays gated
        # (the tree is not the full registered session) until --seal-partial.
        # (tp overlay = the currently-blocked DIST cell; pd is executable
        # since T3.2 and pinned in tests/test_pd_launcher.py.)
        grid = _tiny_grid(dist_cells=(("B3", "vllm", "tp"),))
        plan = _stub_plan(grid, floor_table, stub.cmd)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root, skip_blocked=True, seal=True, seal_cmd=stub.cmd) == 0
        cell_calls = [c for c in stub.calls() if c["argv"][0] != "restart"]
        assert len(cell_calls) == 2, "only the 2 executable F1 cells may run"
        labels = {c["argv"][c["argv"].index("--baseline-label") + 1] for c in cell_calls}
        assert labels == {"B1_gold-fresh", "B2_gold-reuse"}
        # blocked cell: no sentinel (nothing failed), no seal (gated)
        blocked = next(s for s in _cells(plan) if s["blocked_on"])
        assert not (
            root / "cells" / blocked["row_key"] / f".STATUS-{blocked['dataset']}"
        ).exists()
        assert [str(root)] not in [c["argv"] for c in stub.calls()]
        # --seal-partial lifts the gate on the same run
        assert (
            rc.run_plan(
                plan, root, skip_blocked=True, seal=True, seal_partial=True,
                seal_cmd=stub.cmd, force_rerun=True,
            )
            == 0
        )
        assert stub.calls()[-1]["argv"] == [str(root)]


# ---------------------------------------------------------------------------
# W4.3 — §6.4 anchor fine r-grid (registration, dedup, membership markers)
# ---------------------------------------------------------------------------


class TestAnchorFineGrid:
    def test_registered_constants_are_the_charter_values(self):
        # §6.4 verbatim: r ∈ {1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25} at the
        # two chassis-validated rates 0.85·λ* and 1.05·λ*.
        assert rc.ANCHOR_FINE_BUDGET_LEVELS == (1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25)
        assert rc.ANCHOR_FINE_RATE_FRACTIONS == (0.85, 1.05)
        # both fine rates are registered dispatcher fractions (§6.1 grid) —
        # the fine grid can never offer a rate the D6 generator lacks
        assert set(rc.ANCHOR_FINE_RATE_FRACTIONS) <= set(rc.FULL_RATE_FRACTIONS)

    def test_membership_markers_partition_the_f2_grid(self, plan_a):
        # Coordinate classes (per engine, per baseline):
        #   factorial-only: 5r × the 4 non-fine rates {0.5, 0.7, 0.95, 1.2} = 20
        #   both grids:     5r × the 2 fine rates {0.85, 1.05}              = 10
        #   fine-only:      {1.25, 0.375} × {0.85, 1.05}                    =  4
        # 20 + 10 + 4 = 34 coordinates (the count pin's derivation).
        qasper = [
            s for s in _cells(plan_a)
            if s["family"] == "F2" and s["dataset"] == "qasper"
        ]
        by_marker = {}
        for s in qasper:
            by_marker.setdefault(tuple(s["grids"]), 0)
            by_marker[tuple(s["grids"])] += 1
        # 5 FRESH × 2 engines = 10 (baseline, engine) pairs per coordinate
        assert by_marker == {
            (rc.GRID_D6_FACTORIAL,): 20 * 10,
            (rc.GRID_D6_FACTORIAL, rc.GRID_ANCHOR_FINE): 10 * 10,
            (rc.GRID_ANCHOR_FINE,): 4 * 10,
        }
        for s in qasper:
            r, frac = s["cellspec"]["budget_r"], s["cellspec"]["rate_frac"]
            markers = tuple(s["grids"])
            if r in (1.25, 0.375):
                assert markers == (rc.GRID_ANCHOR_FINE,)
            elif frac in (0.85, 1.05):
                assert markers == (rc.GRID_D6_FACTORIAL, rc.GRID_ANCHOR_FINE)
            else:
                assert markers == (rc.GRID_D6_FACTORIAL,)

    def test_no_duplicate_cells_from_the_overlay(self, plan_a):
        # Dedup pin: a coordinate on both grids is ONE cell — the plan must
        # never enumerate the same (row_key, dataset, ruler_task) twice.
        seen = set()
        for s in _cells(plan_a):
            key = (s["row_key"], s["dataset"], s["ruler_task"])
            assert key not in seen, f"duplicate enumeration: {key}"
            seen.add(key)

    def test_non_f2_cells_carry_no_grid_marker(self, plan_a):
        # Absence stays absence: F1/F3 (and hf) cells sit on no registered
        # budget×rate grid, so their marker is null, never [].
        for s in _cells(plan_a):
            if s["family"] == "F2":
                assert s["grids"]
            else:
                assert s["grids"] is None

    def test_fine_grid_in_plan_header(self, plan_a):
        assert plan_a["fine_grid"] == {
            "budget_levels": [1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25],
            "rate_fractions": [0.85, 1.05],
            "membership_labels": [rc.GRID_D6_FACTORIAL, rc.GRID_ANCHOR_FINE],
        }

    def test_fine_axes_must_register_together(self):
        with pytest.raises(rc.PlanError, match="f2_fine_budgets and f2_fine_rates"):
            _tiny_grid(f2_fine_budgets=(1.25,))


# ---------------------------------------------------------------------------
# W4.4 grid half — D5#5 RULER-paired F2 cells (session a)
# ---------------------------------------------------------------------------


class TestRulerPairing:
    def test_task_literals_are_exactly_the_registered_charter_subset(self):
        # The RULER lane's registered literals (src/data/ruler._TASKS), the
        # D5#5 subset: NIAH-MK/MQ, VT, QA — niah_single is a loader default,
        # NOT a charter subset member, and must not be enumerated.
        assert rc.RULER_F2_TASKS == (
            "niah_multikey",
            "niah_multiquery",
            "variable_tracking",
            "qa",
        )

    def test_every_ruler_cell_has_its_matched_qasper_twin(self, plan_a):
        # D5#5: "every RULER cell PAIRED with a matched real-text Qasper
        # cell" — same row key (same baseline/engine/budget/rate coordinate).
        qasper_keys = {
            s["row_key"]
            for s in _cells(plan_a)
            if s["family"] == "F2" and s["dataset"] == "qasper"
        }
        ruler = [s for s in _cells(plan_a) if s["dataset"] == "ruler"]
        # 1 baseline (B1) × 2 engines × 34 coordinates × 4 tasks = 272
        assert len(ruler) == 272
        for s in ruler:
            assert s["row_key"] in qasper_keys, (
                f"ruler cell {s['row_key']} has no matched qasper twin"
            )

    def test_ruler_argv_pins_task_and_shape32k(self, plan_a):
        # SHAPE-32K (§5.1 item 1): 32,512-in + 256-out — EXPLICIT on every
        # step (the loader's 4096 default is a pilot convenience).
        ruler = [s for s in _cells(plan_a) if s["dataset"] == "ruler"]
        for s in ruler:
            assert _argv_value(s, "--ruler-context-tokens") == "32512"
            assert _argv_value(s, "--max-tokens") == "256"
            assert _argv_value(s, "--ruler-task") == s["ruler_task"]
            assert s["ruler_task"] in rc.RULER_F2_TASKS
        # and non-ruler steps never carry the instrument flags
        for s in _cells(plan_a):
            if s["dataset"] != "ruler":
                assert "--ruler-task" not in s["argv"]
                assert s["ruler_task"] is None

    def test_per_task_ordinal_ranges_are_disjoint(self, plan_a):
        # base = task_index × replications: 4 tasks × 3 reps ⇒ {0, 3, 6, 9};
        # each step claims (base, base+3] so the on-disk ranges are disjoint.
        ruler = [s for s in _cells(plan_a) if s["dataset"] == "ruler"]
        by_key = {}
        for s in ruler:
            by_key.setdefault(s["row_key"], []).append(s)
        for key, steps in by_key.items():
            bases = sorted(s["window_ordinal_base"] for s in steps)
            assert bases == [0, 3, 6, 9], f"{key}: bases {bases}"
            for s in steps:
                if s["window_ordinal_base"]:
                    assert s["env"]["CAGE_WINDOW_ORDINAL_BASE"] == str(
                        s["window_ordinal_base"]
                    )
                else:
                    # base 0 stays ABSENT from the env (default, not "0")
                    assert "CAGE_WINDOW_ORDINAL_BASE" not in s["env"]

    def test_unpaired_ruler_baseline_refuses(self):
        with pytest.raises(rc.PlanError, match="no twin"):
            _tiny_grid(
                f2_baselines=("B1",),
                f2_budgets=(1.0,),
                f2_rates=(0.85,),
                f2_ruler_baselines=("B5",),  # not in f2_baselines
                f2_ruler_tasks=("qa",),
            )

    def test_unregistered_task_literal_refuses(self):
        with pytest.raises(rc.PlanError, match="ruler._TASKS"):
            _tiny_grid(
                f2_baselines=("B1",),
                f2_budgets=(1.0,),
                f2_rates=(0.85,),
                f2_ruler_baselines=("B1",),
                f2_ruler_tasks=("needle_haystack",),  # drifted spelling
            )


# ---------------------------------------------------------------------------
# W4.2 — gpu_count producer (plan side; writer side in test_campaign_layout)
# ---------------------------------------------------------------------------


class TestGpuCountProducer:
    def test_every_executable_session_a_cell_carries_gpu_count_1(self, plan_a):
        # Session a is the single-GPU anchor (§7.6 A): serving_tp=1 ⇒ every
        # topology-'single' cell (hf oracle included — same box) counts 1.
        for s in _cells(plan_a):
            assert s["gpu_count"] == 1
            assert s["env"]["CAGE_GPU_COUNT"] == "1"

    def test_blocked_tp_cell_without_registration_has_null_gpu_count(
        self, floor_table, stub
    ):
        # A tp cell with no registered dist_tp_size is BLOCKED and its count
        # is an explicit null — the debt stays visible, never guessed.
        grid = _tiny_grid(dist_cells=(("B3", "vllm", "tp"),))
        plan = _stub_plan(grid, floor_table, stub.cmd)
        (dist,) = [s for s in _cells(plan) if s["family"] == "DIST"]
        assert dist["blocked_on"] == rc.TP_DIST_BLOCKED_ON
        assert dist["gpu_count"] is None
        assert "CAGE_GPU_COUNT" not in dist["env"]

    def test_pd_cell_gpu_count_is_role_sum(self, floor_table, stub):
        # pd = prefill + decode role GPU counts summed; the default (1, 1)
        # single-node dev shape counts 2.
        grid = _tiny_grid(dist_cells=(("B3", "vllm", "pd"),))
        floor = rc.load_floor_table(floor_table)
        orig = rc.SESSION_GRIDS
        rc.SESSION_GRIDS = {grid.session: grid}
        try:
            plan = rc.build_plan(
                grid.session,
                floor,
                window_duration_s=60.0,
                runner_cmd=stub.cmd,
                launcher_cmds={
                    "vllm": stub.cmd,
                    "sglang": stub.cmd,
                    rc.PD_LAUNCHER_KEY: stub.cmd,
                },
            )
        finally:
            rc.SESSION_GRIDS = orig
        (dist,) = [s for s in _cells(plan) if s["family"] == "DIST"]
        assert dist["blocked_on"] is None
        assert dist["gpu_count"] == 1 + 1
        assert dist["env"]["CAGE_GPU_COUNT"] == "2"


# ---------------------------------------------------------------------------
# W4.6 — SESSION_GRIDS['b'] (Run C-prime; §7.6.1 Group B + the DIST overlay)
# ---------------------------------------------------------------------------


@pytest.fixture()
def floor_table_b(tmp_path: Path) -> Path:
    # Group B floor table: llama-3.3-70b on the §6.8 reduced grid — covers
    # the session's every registered r ({1.0, 0.5, 0.25}; dist_budget_r=1.0).
    path = tmp_path / "floor_table_b.json"
    path.write_text(
        json.dumps(
            _floor_table_doc(
                model="llama-3.3-70b", grid="reduced", r_values=(1.0, 0.5, 0.25)
            )
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def plan_b(floor_table_b: Path) -> Dict[str, Any]:
    return rc.build_plan("b", rc.load_floor_table(floor_table_b), window_duration_s=300.0)


class TestSessionB:
    def test_total_counts(self, plan_b):
        # F1: 12 baselines × 2 engines × 4 QA datasets           =  96
        #   + ADR-0106 B12 ladder (2 rungs): 2 eng × 4 ds doubled = +8
        # F1 HF oracle: B3×4 + {B1,B2,B6}×2 (anchor slice reuse) =  10
        # F2: 5 FRESH × 2 engines × 3 budgets × 3 rates (§6.8)   =  90
        # F3: 7 REUSE × 2 engines × 3 budgets × 3 rates          = 126
        #   + B12 ladder: 2 eng × 3 × 3 = 18 doubled             = +18
        # DIST: {B1, B3} × vllm × {tp, pd}                       =   4
        # ⇒ 104 + 10 + 90 + 144 + 4 = 352; windows 352 × 3 = 1056
        #   (old pins 326 / 978, pre-ADR-0106)
        assert plan_b["counts"]["cells"] == 352
        assert plan_b["counts"]["windows"] == 1056
        # Blocked: sglang retr-store (B8), F1 4 + F3 3×3 = 13; the DIST
        # legs are all EXECUTABLE (tp registered, pd launcher exists).
        #   + ADR-0106 B12 rung cells with no query manifest registered
        #     (identical F1/F3 carriage to session a): 16 + 36 = 52
        # ⇒ 13 + 52 = 65  (old pin 13, pre-manifest-registration)
        assert plan_b["counts"]["blocked"] == 65
        # Relaunches = distinct executable configs:
        #   vllm:   F1 {plain, fp8, lmcache} 3 + F3 3×{plain,fp8,lmcache} 9
        #           + F2 3 plain + DIST {tp leg, pd leg} 2        = 17
        #   sglang: F1 {plain, fp8} 2 + F3 3×{plain,fp8} 6 + F2 3 = 11
        #   ADR-0103 (corpus-fresh B4 served prefix OFF by relaunch):
        #           F1 B4 = budget-free prefix-OFF plain, new per engine
        #                                                         = +1 ×2
        #           F3 B4 = prefix-OFF plain at r ∈ {1.0, 0.5, 0.25} =
        #           the F2 plain-OFF configs (same budgets)       = +0
        #           DIST carries {B1, B3} only (no B4 leg)         = +0
        # ⇒ (17 + 1) + (11 + 1) = 30  (old pin 28, pre-ADR-0103)
        assert plan_b["counts"]["relaunches"] == 30

    def test_no_fine_grid_and_no_ruler_on_group_b(self, plan_b):
        # §6.8: the fine r-grid runs on Group A ONLY; the D5#5 RULER pairing
        # is likewise an anchor-only registration.
        assert plan_b["fine_grid"] is None
        assert plan_b["ruler_f2"] is None
        assert all(s["dataset"] != "ruler" for s in _cells(plan_b))
        f2 = [s for s in _cells(plan_b) if s["family"] == "F2"]
        assert {s["cellspec"]["budget_r"] for s in f2} == {1.0, 0.5, 0.25}
        assert {s["cellspec"]["rate_frac"] for s in f2} == {0.85, 0.95, 1.05}

    def test_dist_cells_are_the_transfer_pair_on_vllm_only(self, plan_b):
        # Plan-B scope: {B1, B3} × {tp, pd} on vLLM ONLY (#18 pairs
        # topologies WITHIN one engine; SGLang PD is T3.4-gated — its cells
        # are NOT registered, not even as blocked).
        dist = [s for s in _cells(plan_b) if s["family"] == "DIST"]
        got = {
            (s["baseline"], s["cellspec"]["engine"], s["cellspec"]["topology"])
            for s in dist
        }
        assert got == {
            ("B1", "vllm", "tp"),
            ("B1", "vllm", "pd"),
            ("B3", "vllm", "tp"),
            ("B3", "vllm", "pd"),
        }
        for s in dist:
            assert s["blocked_on"] is None
            if s["cellspec"]["topology"] == "pd":
                assert s["gate"] == rc.PD_GATE
                assert s["gpu_count"] == 4 + 4  # role GPU counts summed
            else:
                assert s["gate"] is None
                assert s["gpu_count"] == 8  # the registered dist_tp_size

    def test_serving_tp_rides_every_single_topology_relaunch(self, plan_b):
        # §7.6 Group B e5: TP=4. Every single-topology relaunch (budget-free
        # F1 ones INCLUDED — a 70B F1 server launched without the degree
        # would silently serve TP=1) carries the engine's T3.1 env.
        singles = [s for s in _relaunches(plan_b) if s["topology"] == "single"]
        assert singles
        for s in singles:
            assert s["tp"] == 4
            if s["engine"] == "vllm":
                assert s["env"]["CAGE_VLLM_TENSOR_PARALLEL"] == "4"
            else:
                assert s["env"]["CAGE_SGLANG_TP"] == "4"

    def test_single_topology_cells_count_the_tp_ranks(self, plan_b):
        # W4.2 on Group B: a topology-'single' cell served TP-sharded counts
        # its ranks (gpu_count = serving_tp = 4), hf oracle included
        # (batch-1 device_map rides the same 4-GPU box, §7.7(e)).
        for s in _cells(plan_b):
            if s["cellspec"]["topology"] == "single":
                assert s["gpu_count"] == 4

    def test_tp_leg_launches_tp8_at_the_dist_budget(self, plan_b, floor_table_b):
        # The tp leg rides the SINGLE-instance launcher at dist_tp_size=8,
        # serving floor(dist_budget_r × D) = 10^10 bytes total — per-rank
        # slice (GQA shards) = 10^10 // 8 = 1_250_000_000.
        (tp_leg,) = [s for s in _relaunches(plan_b) if s["topology"] == "tp"]
        assert tp_leg["tp"] == 8
        assert tp_leg["env"]["CAGE_VLLM_TENSOR_PARALLEL"] == "8"
        assert tp_leg["env"]["CAGE_KV_BUDGET_BYTES"] == str(
            (1 * _ANCHOR_DEMAND) // 8
        )
        assert tp_leg["budget_bytes"] == 1 * _ANCHOR_DEMAND
        assert tp_leg["budget_r"] is None  # DIST overlay: not a pressure coord

    def test_pd_leg_splits_the_same_total_and_carries_role_tp(self, plan_b):
        # Iso-aggregate-bytes (§6.6a): the pd leg splits the SAME
        # floor(1.0 × D) total the tp leg serves — 0.5 split of 10^10 =
        # 5e9 + 5e9 role POOLS — and both role instances launch at TP=4
        # (one env, frozen launcher contract). The launcher passes each
        # budget env VERBATIM as --kv-cache-memory-bytes on that TP=4
        # instance, and the flag's registered convention is PER-RANK (the
        # SAME one the tp leg's env uses), so each role env carries
        # pool // 4 — handing the role TOTAL to 4 ranks would realize 4×
        # the §6.5 pools (2026-09-02 verifier major).
        (pd_leg,) = [s for s in _relaunches(plan_b) if s["topology"] == "pd"]
        assert pd_leg["tp"] == 4
        assert pd_leg["env"]["CAGE_VLLM_TENSOR_PARALLEL"] == "4"
        prefill_rank = int(pd_leg["env"]["CAGE_KV_BUDGET_BYTES_PREFILL"])
        decode_rank = int(pd_leg["env"]["CAGE_KV_BUDGET_BYTES_DECODE"])
        assert prefill_rank == 5_000_000_000 // 4 == 1_250_000_000
        assert decode_rank == 1_250_000_000
        # The plan record keeps the registered §6.5 exact-sum role pools AND
        # the per-rank env basis gate (j) closes against.
        assert pd_leg["pd"]["prefill_bytes"] == 5_000_000_000
        assert pd_leg["pd"]["decode_bytes"] == 5_000_000_000
        assert (
            pd_leg["pd"]["prefill_bytes"] + pd_leg["pd"]["decode_bytes"]
            == 1 * _ANCHOR_DEMAND
        )
        assert pd_leg["pd"]["prefill_bytes_per_rank"] == prefill_rank
        assert pd_leg["pd"]["decode_bytes_per_rank"] == decode_rank
        # Cross-leg §6.6a closure under the ONE registered convention:
        # pd realized = (pool // 4) × 4 ranks × 2 roles = 10^10 = tp
        # realized = (total // 8) × 8 — the #18 pair stays iso-aggregate.
        realized_pd = (prefill_rank + decode_rank) * 4
        assert pd_leg["pd"]["expected_bytes_total"] == realized_pd
        (tp_leg,) = [s for s in _relaunches(plan_b) if s["topology"] == "tp"]
        realized_tp = int(tp_leg["env"]["CAGE_KV_BUDGET_BYTES"]) * 8
        assert realized_pd == realized_tp == 1 * _ANCHOR_DEMAND

    def test_f2_budget_env_divides_per_rank_at_tp4(self, plan_b):
        # Independent arithmetic for the TP-sharded budget env: GQA shards ⇒
        # per-rank = floor(r × D) // 4. llama-3.3-70b bf16 KV/token =
        # 2 (K,V) × 80 layers × 8 KV heads × 128 head_dim × 2 B = 327_680;
        # SGLang tokens = per-rank // 327_680.
        per_token = 2 * 80 * 8 * 128 * 2
        assert per_token == 327_680
        for s in _relaunches(plan_b):
            if s["topology"] != "single" or s["budget_r"] is None:
                continue
            per_rank = int(s["budget_r"] * _ANCHOR_DEMAND) // 4
            if s["engine"] == "vllm" and s["kv_dtype"] is None and s["connector"] is None:
                assert s["env"]["CAGE_KV_BUDGET_BYTES"] == str(per_rank)
            if s["engine"] == "sglang" and s["kv_dtype"] is None:
                assert s["env"]["CAGE_SGLANG_MAX_TOTAL_TOKENS"] == str(
                    per_rank // per_token
                )

    def test_dist_registration_shape_refusals(self):
        # unequal pd role counts: unrealizable with the frozen pd launcher
        with pytest.raises(rc.PlanError, match="must be equal"):
            _tiny_grid(dist_pd_role_gpus=(4, 2))
        # a TP=1 'tp' leg is a topology/count contradiction
        with pytest.raises(rc.PlanError, match="dist_tp_size"):
            _tiny_grid(dist_tp_size=1)

    def test_plan_b_roundtrips_through_load_plan(self, tmp_path, plan_b):
        out = tmp_path / "plan_b.json"
        out.write_text(json.dumps(plan_b), encoding="utf-8")
        assert rc.load_plan(out)["counts"]["cells"] == 352  # TestSessionB pin (ADR-0106; old 326)


# ---------------------------------------------------------------------------
# Plan schema v3 (the v2-precedent bump)
# ---------------------------------------------------------------------------


class TestPlanSchemaV5:
    def test_schema_literal(self):
        assert rc.PLAN_SCHEMA == "cage-campaign-plan-v5"

    @pytest.mark.parametrize(
        "old_schema",
        ["cage-campaign-plan-v2", "cage-campaign-plan-v3", "cage-campaign-plan-v4"],
    )
    def test_older_plan_refuses(self, tmp_path, old_schema):
        # A v2 plan predates gpu_count / grids / ruler_task /
        # window_ordinal_base / relaunch tp; a v3 plan predates the
        # ADR-0102/0103/0104/0106 argv and the query-manifest registration;
        # a v4 plan predates the A9 per-row N cell keys (row_class,
        # num_queries) and the --num-queries argv. 'run' must refuse all
        # three and the operator re-plans (the v2 precedent).
        old = tmp_path / "old.json"
        old.write_text(
            json.dumps({"schema": old_schema, "steps": []}),
            encoding="utf-8",
        )
        with pytest.raises(rc.RunError, match="schema"):
            rc.load_plan(old)

    def test_cell_step_missing_v3_key_refuses(self, tmp_path, plan_a):
        plan = json.loads(json.dumps(plan_a))
        cell = next(s for s in plan["steps"] if s["kind"] == "cell")
        del cell["gpu_count"]
        path = tmp_path / "missing_key.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        with pytest.raises(rc.RunError, match="gpu_count"):
            rc.load_plan(path)

    def test_relaunch_step_missing_tp_refuses(self, tmp_path, plan_a):
        plan = json.loads(json.dumps(plan_a))
        step = next(s for s in plan["steps"] if s["kind"] == "relaunch")
        del step["tp"]
        path = tmp_path / "missing_tp.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        with pytest.raises(rc.RunError, match="tp"):
            rc.load_plan(path)

    @pytest.mark.parametrize("key", ["row_class", "num_queries"])
    def test_cell_step_missing_v5_key_refuses(self, tmp_path, plan_a, key):
        # A9: the per-row N keys are REQUIRED cell-step keys (v5).
        plan = json.loads(json.dumps(plan_a))
        cell = next(s for s in plan["steps"] if s["kind"] == "cell")
        del cell[key]
        path = tmp_path / f"missing_{key}.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        with pytest.raises(rc.RunError, match=key):
            rc.load_plan(path)


# ---------------------------------------------------------------------------
# ADR-0102 (owner decision 2026-09-16): cold start per window with a
# registered warm-up from a DISJOINT pool (W_warm = 20)
# ---------------------------------------------------------------------------


class TestColdStartPerWindow:
    def test_registered_constants(self):
        # The knobs are module constants (reviewable, ADR-cited), never a
        # silent default inside a step builder.
        assert rc.RESET_CACHE_PER_WINDOW is True
        assert rc.WARMUP_POOL_QUERIES == 20
        assert set(rc.SERVER_ENGINES) == {"vllm", "sglang", "lmdeploy"}
        assert "hf" not in rc.SERVER_ENGINES

    def test_every_server_engine_cell_carries_cold_start_argv(self, plan_a, plan_b):
        # A window that starts warm when the plan says cold is a mislabeled
        # row: EVERY server-engine cell resets the engine cache per trial
        # and warms up from the disjoint pool; the in-process hf oracle
        # gets neither (no server cache to flush).
        for plan in (plan_a, plan_b):
            for s in _cells(plan):
                engine = s["cellspec"]["engine"]
                if engine in rc.SERVER_ENGINES:
                    assert "--reset-cache-between-trials" in s["argv"], s["row_key"]
                    assert _argv_value(s, "--warmup-pool-queries") == str(
                        rc.WARMUP_POOL_QUERIES
                    )
                else:
                    assert engine == "hf"
                    assert "--reset-cache-between-trials" not in s["argv"]
                    assert "--warmup-pool-queries" not in s["argv"]
                # The legacy flag replays the MEASURED set (cache-warms the
                # measured queries) and must never ride a campaign cell.
                assert "--warmup-queries" not in s["argv"], s["row_key"]

    def test_hf_cells_exist_and_are_excluded(self, plan_a):
        hf = [s for s in _cells(plan_a) if s["cellspec"]["engine"] == "hf"]
        assert len(hf) == 10  # the reduced oracle set is the exclusion witness
        for s in hf:
            assert "--reset-cache-between-trials" not in s["argv"]
            assert "--warmup-pool-queries" not in s["argv"]

    def test_plan_header_records_both_constants(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            knobs = plan["behavior_knobs"]
            assert knobs["reset_cache_per_window"] is True
            assert knobs["warmup_pool_queries"] == rc.WARMUP_POOL_QUERIES
            assert knobs["cold_start_adr"] == "ADR-0102"

    def test_counts_unchanged_by_cold_start(self, plan_a, plan_b):
        # The window protocol adds argv, never cells or windows (the pins at
        # TestPlanCountsSessionA / TestSessionB stay: 870/2610 and 352/1056,
        # the ADR-0106 ladder pins; pre-ADR-0106 844/2532 and 326/978).
        assert (plan_a["counts"]["cells"], plan_a["counts"]["windows"]) == (870, 2610)
        assert (plan_b["counts"]["cells"], plan_b["counts"]["windows"]) == (352, 1056)

    def test_load_plan_refuses_server_cell_without_cold_start(self, tmp_path, plan_a):
        # A stale plan (pre-ADR-0102) whose server-engine cell lacks the
        # cold-start argv would run warm windows under a cold label: 'run'
        # refuses it at load time and the operator re-plans.
        plan = json.loads(json.dumps(plan_a))
        cell = next(
            s for s in plan["steps"]
            if s["kind"] == "cell" and s["cellspec"]["engine"] == "vllm"
        )
        cell["argv"] = [
            a for a in cell["argv"] if a != "--reset-cache-between-trials"
        ]
        path = tmp_path / "stale_cold_start.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        with pytest.raises(rc.RunError, match="reset-cache-between-trials"):
            rc.load_plan(path)

        plan = json.loads(json.dumps(plan_a))
        cell = next(
            s for s in plan["steps"]
            if s["kind"] == "cell" and s["cellspec"]["engine"] == "sglang"
        )
        i = cell["argv"].index("--warmup-pool-queries")
        del cell["argv"][i:i + 2]
        path.write_text(json.dumps(plan), encoding="utf-8")
        with pytest.raises(rc.RunError, match="warmup-pool-queries"):
            rc.load_plan(path)

    def test_load_plan_accepts_a_fresh_plan(self, tmp_path, plan_a):
        path = tmp_path / "fresh.json"
        path.write_text(json.dumps(plan_a), encoding="utf-8")
        assert rc.load_plan(path)["counts"]["cells"] == 870  # ADR-0106 pin (old 844)


# ---------------------------------------------------------------------------
# Adversarial review 2026-09-16, defect 5: query-manifest registration. A
# B12 rung serves ONLY from a manifest carrying the ladder (the runner's A4
# guard), so the plan must carry the manifest per dataset or the rung cells
# are blocked_on the missing registration (visible debt, never a silent
# fallback that DROPS out-of-corpus queries).
# ---------------------------------------------------------------------------


def _manifest_trials(dataset: str, *, trials: int, ids_per_trial: int) -> Dict[str, List[str]]:
    """Disjoint per-trial id lists in manifest (draw) order."""
    return {
        str(t): [
            f"{dataset}-q{(t - 1) * ids_per_trial + i:05d}" for i in range(ids_per_trial)
        ]
        for t in range(1, trials + 1)
    }


def _write_manifest(
    tmp_path: Path,
    dataset: str,
    *,
    block_budget: int = 2800,
    rungs: tuple = (1400, 700),
    name: Optional[str] = None,
    # A9 per-row N: the planner refuses a manifest whose trials carry fewer
    # ids than the dataset's most demanding cell (primary n = 2000, 3 reps),
    # so the default fixture carries exactly that; tests lower it to probe
    # the shortfall refusal.
    trials: int = 3,
    ids_per_trial: int = 2000,
    trial_ids: Optional[Dict[str, List[str]]] = None,
) -> Path:
    """A minimal manifest carrying exactly what the planner validates."""
    doc = {
        "manifest_version": 3,
        "dataset": dataset,
        "block_budget": block_budget,
        "trials": (
            _manifest_trials(dataset, trials=trials, ids_per_trial=ids_per_trial)
            if trial_ids is None
            else trial_ids
        ),
        "blocks": [],
        "question_to_block": {},
        "trunc_rungs": {
            str(r): {"budget": r, "blocks": [], "in_corpus_ids": []} for r in rungs
        },
    }
    path = tmp_path / (name or f"manifest_{dataset}.json")
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture()
def manifests_a(tmp_path: Path) -> Dict[str, Path]:
    return {ds: _write_manifest(tmp_path, ds) for ds in rc.QA_DATASETS}


@pytest.fixture()
def plan_a_manifests(floor_table: Path, manifests_a: Dict[str, Path]) -> Dict[str, Any]:
    floor = rc.load_floor_table(floor_table)
    return rc.build_plan("a", floor, window_duration_s=300.0, query_manifests=manifests_a)


class TestQueryManifestRegistration:
    @staticmethod
    def _b12(plan):
        return [s for s in _cells(plan) if s["cellspec"]["arm"] == "corpus-trunc"]

    def test_without_manifests_every_b12_cell_is_blocked(self, plan_a):
        b12 = self._b12(plan_a)
        assert len(b12) == 52
        for s in b12:
            assert s["blocked_on"] == rc.trunc_manifest_blocked_on(s["dataset"])
        for s in _cells(plan_a):
            assert "--query-manifest" not in s["argv"], s["row_key"]
        assert plan_a["query_manifests"] == {}

    def test_with_manifests_b12_is_executable_and_every_qa_cell_carries_the_flag(
        self, plan_a_manifests, manifests_a
    ):
        plan = plan_a_manifests
        for s in self._b12(plan):
            assert s["blocked_on"] is None
        # Back to the B8-only blocked set (13, see TestPlanCountsSessionA).
        assert plan["counts"]["blocked"] == 13
        for s in _cells(plan):
            if s["dataset"] in manifests_a:
                assert _argv_value(s, "--query-manifest") == str(
                    manifests_a[s["dataset"]].resolve()
                ), s["row_key"]
            else:
                assert s["dataset"] == "ruler"
                assert "--query-manifest" not in s["argv"]
        header = plan["query_manifests"]
        assert set(header) == set(rc.QA_DATASETS)
        for ds, rec in header.items():
            assert rec["path"] == str(manifests_a[ds].resolve())
            assert len(rec["sha256"]) == 64
            assert rec["block_budget"] == 2800
            assert rec["trunc_rungs"] == [1400, 700]

    def test_counts_unchanged_by_manifest_registration(self, plan_a, plan_a_manifests):
        # Registration changes executability, never the enumeration: the
        # TestPlanCountsSessionA pins (870 / 2610 / 36) hold on both plans.
        for key in ("cells", "windows", "relaunches"):
            assert plan_a_manifests["counts"][key] == plan_a["counts"][key] == {
                "cells": 870, "windows": 2610, "relaunches": 36
            }[key]

    def test_partial_registration_blocks_only_the_unmanifested_datasets(
        self, floor_table, tmp_path
    ):
        floor = rc.load_floor_table(floor_table)
        only = {"squad_v2": _write_manifest(tmp_path, "squad_v2")}
        plan = rc.build_plan("a", floor, window_duration_s=300.0, query_manifests=only)
        for s in self._b12(plan):
            if s["dataset"] == "squad_v2":
                assert s["blocked_on"] is None
                assert _argv_value(s, "--query-manifest") == str(only["squad_v2"].resolve())
            else:
                assert s["blocked_on"] == rc.trunc_manifest_blocked_on(s["dataset"])
        unblocked = [s for s in self._b12(plan) if s["dataset"] == "squad_v2"]
        assert len(unblocked) == 4  # F1 only: 2 engines x 2 rungs (F3 runs on qasper)
        assert plan["counts"]["blocked"] == 65 - 4

    @pytest.mark.parametrize(
        "spoil, match",
        [
            ("missing", "not found"),
            ("dataset", "for dataset"),
            ("budget", "block_budget"),
            ("rung", "rung"),
            ("unknown", "not a dataset"),
        ],
    )
    def test_manifest_registration_refusals(self, floor_table, tmp_path, spoil, match):
        floor = rc.load_floor_table(floor_table)
        manifests: Dict[str, Path] = {}
        if spoil == "missing":
            manifests["squad_v2"] = tmp_path / "absent.json"
        elif spoil == "dataset":
            manifests["squad_v2"] = _write_manifest(tmp_path, "hotpotqa", name="wrong.json")
        elif spoil == "budget":
            manifests["squad_v2"] = _write_manifest(tmp_path, "squad_v2", block_budget=2000)
        elif spoil == "rung":
            manifests["squad_v2"] = _write_manifest(tmp_path, "squad_v2", rungs=(1400,))
        elif spoil == "unknown":
            manifests["nq_open"] = _write_manifest(tmp_path, "nq_open")
        with pytest.raises(rc.PlanError, match=match):
            rc.build_plan("a", floor, window_duration_s=300.0, query_manifests=manifests)

    def test_real_build_manifest_artifact_registers(self, floor_table, stub, tmp_path):
        # The registration contract holds for the REAL artifact
        # build_query_manifest.py writes (src.data.manifest.build_manifest).
        from src.data.loader import CAGExample
        from src.data.manifest import build_manifest

        pool = [
            CAGExample(
                id=f"squad_v2-q{i:03d}",
                question=f"Question {i}?",
                context=[f"Paragraph {i} " + "word " * 30],
                answer=f"a{i}",
                metadata={},
            )
            for i in range(40)
        ]
        manifest = build_manifest(
            pool, num_queries=8, num_trials=3, seed=7, block_budget=2800,
            dataset="squad_v2", trunc_budgets=(1000, 500),
        )
        path = tmp_path / "squad_v2.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        floor = rc.load_floor_table(floor_table)

        def _plan(grid):
            orig = rc.SESSION_GRIDS
            rc.SESSION_GRIDS = {grid.session: grid}
            try:
                return rc.build_plan(
                    grid.session, floor, window_duration_s=60.0, runner_cmd=stub.cmd,
                    launcher_cmds={"vllm": stub.cmd, "sglang": stub.cmd},
                    query_manifests={"squad_v2": path},
                )
            finally:
                rc.SESSION_GRIDS = orig

        # A9: an 8-id-per-trial artifact cannot serve the registered primary
        # n (B3 on vllm = 2000) -> the plan refuses naming the shortfall ...
        grid = _tiny_grid(f1_baselines=("B3", "B12"), corpus_trunc_budgets=(1000, 500))
        with pytest.raises(rc.PlanError, match=r"squad_v2.*trial 1.*2000.*shortfall 1992"):
            _plan(grid)
        # ... unless the grid registers the dataset's achievable n (A5).
        grid = _tiny_grid(
            f1_baselines=("B3", "B12"), corpus_trunc_budgets=(1000, 500),
            achievable_n={"squad_v2": 8},
        )
        plan = _plan(grid)
        cells = _cells(plan)
        assert [s["baseline"] for s in cells] == ["B3", "B12", "B12"]
        assert all(s["blocked_on"] is None for s in cells)
        assert all(_argv_value(s, "--query-manifest") == str(path.resolve()) for s in cells)
        assert all(_argv_value(s, "--num-queries") == "8" for s in cells)
        assert [s["row_class"] for s in cells] == ["primary", "secondary", "secondary"]
        assert plan["counts"]["blocked"] == 0

    def test_cli_registers_manifests_and_refuses_malformed(self, tmp_path, floor_table):
        out = tmp_path / "plan.json"
        squad = _write_manifest(tmp_path, "squad_v2")
        base = ["plan", "--session", "a", "--floor-table", str(floor_table),
                "--window-duration-s", "300", "--out", str(out)]
        assert rc.main(base + ["--query-manifest", f"squad_v2={squad}"]) == 0
        plan = rc.load_plan(out)
        assert set(plan["query_manifests"]) == {"squad_v2"}
        # Malformed (no '=') and duplicate registrations refuse (exit 2).
        assert rc.main(base + ["--query-manifest", str(squad)]) == 2
        assert rc.main(
            base + ["--query-manifest", f"squad_v2={squad}", "--query-manifest", f"squad_v2={squad}"]
        ) == 2


# ---------------------------------------------------------------------------
# Adversarial review 2026-09-16, defect 4: load_plan refuses EVERY stale
# plan shape today's ADRs fail-close against, not only ADR-0102's.
# ---------------------------------------------------------------------------


def _dump(tmp_path: Path, plan: Dict[str, Any], name: str) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _preceding_relaunch(plan: Dict[str, Any], cell: Dict[str, Any]) -> Dict[str, Any]:
    idx = plan["steps"].index(cell)
    return next(s for s in reversed(plan["steps"][:idx]) if s["kind"] == "relaunch")


class TestStalePlanRefusals:
    def test_fresh_plan_with_manifests_loads(self, tmp_path, plan_a_manifests):
        path = _dump(tmp_path, plan_a_manifests, "fresh.json")
        assert rc.load_plan(path)["counts"]["cells"] == 870  # ADR-0106 pin

    def test_warmup_pool_value_drift_refuses(self, tmp_path, plan_a):
        plan = json.loads(json.dumps(plan_a))
        cell = next(s for s in _cells(plan) if s["cellspec"]["engine"] == "vllm")
        cell["argv"][cell["argv"].index("--warmup-pool-queries") + 1] = "5"
        with pytest.raises(rc.RunError, match="warmup-pool-queries"):
            rc.load_plan(_dump(tmp_path, plan, "warm5.json"))

    def test_prefix_off_arm_served_prefix_on_refuses(self, tmp_path, plan_a):
        # ADR-0103: a corpus-fresh (B4) cell must be served prefix OFF, in
        # its own serving record AND by the relaunch it runs under.
        plan = json.loads(json.dumps(plan_a))
        cell = next(
            s for s in _cells(plan)
            if s["cellspec"]["arm"] == "corpus-fresh" and s["cellspec"]["engine"] == "vllm"
        )
        cell["serving"]["prefix_mode"] = "ON"
        with pytest.raises(rc.RunError, match="prefix"):
            rc.load_plan(_dump(tmp_path, plan, "b4_on.json"))

        plan = json.loads(json.dumps(plan_a))
        cell = next(
            s for s in _cells(plan)
            if s["cellspec"]["arm"] == "corpus-fresh" and s["cellspec"]["engine"] == "vllm"
        )
        relaunch = _preceding_relaunch(plan, cell)
        assert "--no-prefix-cache" in relaunch["argv"]
        relaunch["argv"] = [a for a in relaunch["argv"] if a != "--no-prefix-cache"]
        with pytest.raises(rc.RunError, match="no-prefix-cache"):
            rc.load_plan(_dump(tmp_path, plan, "b4_relaunch_on.json"))

    def test_rerank_pool_drift_refuses_both_ways(self, tmp_path, plan_a):
        # ADR-0104: a ranked cell without the pool would run the legacy
        # rerank-exactly-top-k pipeline under the pooled row key; a dense
        # (B5) cell with a pool is the runner-refused leak.
        plan = json.loads(json.dumps(plan_a))
        cell = next(s for s in _cells(plan) if s["cellspec"]["retriever"] == "rerank")
        i = cell["argv"].index("--rerank-pool")
        del cell["argv"][i:i + 2]
        with pytest.raises(rc.RunError, match="rerank-pool"):
            rc.load_plan(_dump(tmp_path, plan, "no_pool.json"))

        plan = json.loads(json.dumps(plan_a))
        cell = next(s for s in _cells(plan) if s["cellspec"]["retriever"] == "rerank")
        i = cell["argv"].index("--rerank-pool")
        cell["argv"][i + 1] = "3"
        with pytest.raises(rc.RunError, match="rerank-pool"):
            rc.load_plan(_dump(tmp_path, plan, "pool3.json"))

        plan = json.loads(json.dumps(plan_a))
        cell = next(s for s in _cells(plan) if s["cellspec"]["retriever"] == "dense")
        cell["argv"] += ["--rerank-pool", str(rc.RERANK_POOL)]
        with pytest.raises(rc.RunError, match="rerank-pool"):
            rc.load_plan(_dump(tmp_path, plan, "dense_pool.json"))

    def test_corpus_trunc_cell_without_rung_or_manifest_refuses(
        self, tmp_path, plan_a_manifests
    ):
        # ADR-0106: an EXECUTABLE rung cell carries --corpus-rung equal to
        # its identity rung and the --query-manifest that serves it.
        def _b12(plan):
            return next(
                s for s in _cells(plan)
                if s["cellspec"]["arm"] == "corpus-trunc" and s["blocked_on"] is None
            )

        plan = json.loads(json.dumps(plan_a_manifests))
        cell = _b12(plan)
        i = cell["argv"].index("--corpus-rung")
        del cell["argv"][i:i + 2]
        with pytest.raises(rc.RunError, match="corpus-rung"):
            rc.load_plan(_dump(tmp_path, plan, "no_rung.json"))

        plan = json.loads(json.dumps(plan_a_manifests))
        cell = _b12(plan)
        cell["argv"][cell["argv"].index("--corpus-rung") + 1] = "9999"
        with pytest.raises(rc.RunError, match="corpus-rung"):
            rc.load_plan(_dump(tmp_path, plan, "wrong_rung.json"))

        plan = json.loads(json.dumps(plan_a_manifests))
        cell = _b12(plan)
        i = cell["argv"].index("--query-manifest")
        del cell["argv"][i:i + 2]
        with pytest.raises(rc.RunError, match="query-manifest"):
            rc.load_plan(_dump(tmp_path, plan, "no_manifest.json"))


# ---------------------------------------------------------------------------
# Backlog A9 (--num-queries half): registered per-row N
# (MyDocs/registration/power_decision_2026-08-07/DECISION.md, amendment A1
# table; A5 achievable-n branch). Every cell carries --num-queries <n> for
# its row class; the plan refuses a manifest that cannot supply n.
# ---------------------------------------------------------------------------

from dataclasses import replace as _dc_replace  # noqa: E402


def _expected_class(step: Dict[str, Any], grid: Any) -> str:
    """The A1 table restated INDEPENDENTLY of the driver's classifier."""
    cs = step["cellspec"]
    if cs["engine"] == "hf":
        return "identity"  # HF-oracle / T=0 identity cells
    if step["family"] in ("F2", "F3") or step["dataset"] == "ruler":
        return "window"  # loaded/window cells: W requests per window
    if step["family"] == "DIST":
        return "identity"  # TTFT-only topology contrast (#18)
    assert step["family"] == "F1"
    if step["baseline"] in grid.primary_baselines and cs["engine"] == grid.primary_engine:
        return "primary"  # the #4 contrast cells on the pinned engine
    return "secondary"


def _n_of_class(cls: str) -> int:
    return {"primary": 2000, "secondary": 800, "identity": 300, "window": 200}[cls]


class TestPerRowN:
    def test_registered_constants_and_grid_fields(self):
        # The A1 table lives on the SessionGrid (reviewable in the header),
        # seeded from named module constants citing the decision record.
        assert rc.N_PRIMARY == 2000
        assert rc.N_SECONDARY == 800
        assert rc.N_IDENTITY == 300
        assert rc.WINDOW_REQUESTS == 200
        assert rc.PRIMARY_ENGINE == "vllm"
        assert rc.PRIMARY_BASELINES == ("B3", "B6")
        assert set(rc.ROW_CLASSES) == {"primary", "secondary", "identity", "window"}
        for grid in rc.SESSION_GRIDS.values():
            assert grid.n_primary == 2000
            assert grid.n_secondary == 800
            assert grid.n_identity == 300
            assert grid.window_requests == 200
            assert grid.primary_engine == "vllm"
            assert grid.primary_baselines == ("B3", "B6")
            # Qasper's achievable n is an OWNER registration (A5): not
            # fabricated here, so the anchor grids register none.
            assert dict(grid.achievable_n) == {}

    def test_classifier_table_every_family_engine_baseline(self, plan_a, plan_b):
        # Every (family x engine x baseline) cell of both sessions lands in
        # the class the A1 table assigns, and the per-class cell counts are
        # pinned by independent arithmetic:
        #   a: primary = B3,B6 x vllm x 4 datasets = 8; secondary = the other
        #      F1 server cells 104 - 8 = 96; identity = 10 hf; window =
        #      F2 612 + F3 144 = 756  (sum 870 = the cells pin)
        #   b: primary 8; secondary 96; identity = 10 hf + 4 DIST = 14;
        #      window = F2 90 + F3 144 = 234  (sum 352)
        for plan, grid, want in (
            (plan_a, rc.SESSION_GRIDS["a"],
             {"primary": 8, "secondary": 96, "identity": 10, "window": 756}),
            (plan_b, rc.SESSION_GRIDS["b"],
             {"primary": 8, "secondary": 96, "identity": 14, "window": 234}),
        ):
            got: Dict[str, int] = {}
            for s in _cells(plan):
                cls = _expected_class(s, grid)
                assert s["row_class"] == cls, s["row_key"]
                got[cls] = got.get(cls, 0) + 1
            assert got == want
            assert sum(want.values()) == plan["counts"]["cells"]
            assert plan["per_row_n"]["cells_by_class"] == want
        # Direct table checks on the pure classifier.
        grid = rc.SESSION_GRIDS["a"]
        f1 = {(s["baseline"], s["cellspec"]["engine"]): s for s in _cells(plan_a) if s["family"] == "F1"}
        assert f1[("B3", "vllm")]["row_class"] == "primary"
        assert f1[("B6", "vllm")]["row_class"] == "primary"
        assert f1[("B3", "sglang")]["row_class"] == "secondary"
        assert f1[("B6", "sglang")]["row_class"] == "secondary"
        assert f1[("B1", "vllm")]["row_class"] == "secondary"
        assert f1[("B12", "vllm")]["row_class"] == "secondary"
        assert f1[("B3", "hf")]["row_class"] == "identity"
        for s in _cells(plan_a):
            if s["dataset"] == "ruler":
                assert s["row_class"] == "window"
        assert {s["row_class"] for s in _cells(plan_b) if s["family"] == "DIST"} == {"identity"}
        cell = next(c for c in rc.enumerate_cells(grid) if c.baseline_id == "B3" and c.spec.engine == "vllm" and c.spec.family == "F1")
        assert rc.row_class(grid, cell) == "primary"

    def test_every_cell_carries_num_queries_for_its_class(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            for s in _cells(plan):
                n = _n_of_class(s["row_class"])
                assert s["num_queries"] == n, s["row_key"]
                assert _argv_value(s, "--num-queries") == str(n), s["row_key"]
                # The window pool: the open-loop generator draws from the
                # measured set, so W IS the registered --num-queries.
                if s["family"] in ("F2", "F3"):
                    assert s["num_queries"] == rc.WINDOW_REQUESTS

    def test_plan_header_records_the_constants(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            head = plan["per_row_n"]
            assert head["n_primary"] == 2000
            assert head["n_secondary"] == 800
            assert head["n_identity"] == 300
            assert head["window_requests"] == 200
            assert head["primary_engine"] == "vllm"
            assert head["primary_baselines"] == ["B3", "B6"]
            assert head["achievable_n"] == {}
            assert head["achievable_n_caveat"] is None
            assert "power_decision_2026-08-07" in head["decision"]

    def test_counts_unchanged_by_per_row_n(self, plan_a, plan_b):
        # --num-queries is argv on existing cells: the enumeration pins
        # (TestPlanCountsSessionA / TestSessionB) are byte-identical.
        assert (
            plan_a["counts"]["cells"], plan_a["counts"]["windows"],
            plan_a["counts"]["relaunches"], plan_a["counts"]["blocked"],
        ) == (870, 2610, 36, 65)
        assert (
            plan_b["counts"]["cells"], plan_b["counts"]["windows"],
            plan_b["counts"]["relaunches"], plan_b["counts"]["blocked"],
        ) == (352, 1056, 30, 65)

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"n_primary": 0}, "n_primary"),
            ({"n_secondary": -1}, "n_secondary"),
            ({"n_identity": True}, "n_identity"),
            ({"window_requests": 0}, "window_requests"),
            ({"n_primary": 2.0}, "n_primary"),
            ({"primary_engine": "hf"}, "primary_engine"),
            ({"primary_engine": "nope"}, "primary_engine"),
            ({"primary_engine": "sglang"}, "primary_engine"),  # not an f1 engine of the tiny grid
            ({"primary_baselines": ()}, "primary_baselines"),
            ({"primary_baselines": ("B99",)}, "primary_baselines"),
            ({"achievable_n": {"qasper": 3000}}, "achievable_n"),  # above n_primary: not a lowering
            ({"achievable_n": {"nq_open": 100}}, "achievable_n"),  # not a dataset of the grid
            ({"achievable_n": {"qasper": 0}}, "achievable_n"),
            ({"achievable_n": {"qasper": 1.5}}, "achievable_n"),
        ],
    )
    def test_grid_registration_refusals(self, overrides, match):
        with pytest.raises(rc.PlanError, match=match):
            _tiny_grid(**overrides)

    def test_manifest_shortfall_refuses_naming_dataset_trial_n_and_shortfall(
        self, floor_table, tmp_path
    ):
        floor = rc.load_floor_table(floor_table)
        # squad_v2 carries primary cells (B3/B6 on vllm): trial 2 with 1999
        # ids is one short of the registered n = 2000.
        short = _write_manifest(
            tmp_path, "squad_v2",
            trial_ids={
                "1": [f"squad_v2-q{i:05d}" for i in range(2000)],
                "2": [f"squad_v2-q{i:05d}" for i in range(2000, 3999)],
                "3": [f"squad_v2-q{i:05d}" for i in range(4000, 6000)],
            },
        )
        with pytest.raises(rc.PlanError, match=r"squad_v2.*trial 2.*1999.*n=2000.*shortfall 1"):
            rc.build_plan("a", floor, window_duration_s=300.0, query_manifests={"squad_v2": short})
        # A manifest with fewer trials than the registered windows refuses too
        # (the runner would read trial 3 and find nothing).
        two = _write_manifest(tmp_path, "squad_v2", trials=2, name="two.json")
        with pytest.raises(rc.PlanError, match=r"squad_v2.*trial 3"):
            rc.build_plan("a", floor, window_duration_s=300.0, query_manifests={"squad_v2": two})
        # Malformed trials refuse rather than being read as empty.
        bad = _write_manifest(tmp_path, "squad_v2", trial_ids={"1": "not-a-list"}, name="bad.json")
        with pytest.raises(rc.PlanError, match=r"squad_v2.*trial"):
            rc.build_plan("a", floor, window_duration_s=300.0, query_manifests={"squad_v2": bad})
        # ruler (the F2 instrument) carries only window cells: a 200-id
        # manifest is enough for it and the header records the trial sizes.
        plan = rc.build_plan(
            "a", floor, window_duration_s=300.0,
            query_manifests={"squad_v2": _write_manifest(tmp_path, "squad_v2", name="ok.json")},
        )
        assert plan["query_manifests"]["squad_v2"]["trial_sizes"] == {"1": 2000, "2": 2000, "3": 2000}

    def test_achievable_n_lowers_the_class_n_with_a_header_caveat(self, floor_table, tmp_path):
        floor = rc.load_floor_table(floor_table)
        grid = _dc_replace(rc.SESSION_GRIDS["a"], achievable_n={"qasper": 900})
        orig = rc.SESSION_GRIDS
        rc.SESSION_GRIDS = {"a": grid}
        try:
            plan = rc.build_plan("a", floor, window_duration_s=300.0)
            for s in _cells(plan):
                base = _n_of_class(s["row_class"])
                want = min(base, 900) if s["dataset"] == "qasper" else base
                assert s["num_queries"] == want, s["row_key"]
                assert _argv_value(s, "--num-queries") == str(want)
            lowered = [s for s in _cells(plan) if s["dataset"] == "qasper" and s["row_class"] == "primary"]
            assert len(lowered) == 2 and all(s["num_queries"] == 900 for s in lowered)
            # secondary (800) < 900: the override only LOWERS, never raises.
            assert all(
                s["num_queries"] == 800
                for s in _cells(plan) if s["dataset"] == "qasper" and s["row_class"] == "secondary"
            )
            head = plan["per_row_n"]
            assert head["achievable_n"] == {"qasper": 900}
            assert "qasper" in head["achievable_n_caveat"] and "A5" in head["achievable_n_caveat"]
            # A 900-id qasper manifest now registers (it would refuse at 2000).
            m900 = _write_manifest(tmp_path, "qasper", ids_per_trial=900)
            plan = rc.build_plan("a", floor, window_duration_s=300.0, query_manifests={"qasper": m900})
            assert plan["query_manifests"]["qasper"]["trial_sizes"]["1"] == 900
        finally:
            rc.SESSION_GRIDS = orig
        with pytest.raises(rc.PlanError, match=r"qasper.*trial 1.*900.*n=2000"):
            rc.build_plan("a", floor, window_duration_s=300.0, query_manifests={"qasper": m900})
        # Counts are untouched by the override.
        assert plan["counts"]["cells"] == 870

    def test_load_plan_refuses_cells_without_or_with_drifted_num_queries(self, tmp_path, plan_a):
        def _strip(engine: str, name: str) -> Path:
            plan = json.loads(json.dumps(plan_a))
            cell = next(s for s in _cells(plan) if s["cellspec"]["engine"] == engine)
            i = cell["argv"].index("--num-queries")
            del cell["argv"][i:i + 2]
            return _dump(tmp_path, plan, name)

        # A server-engine cell AND an hf cell without --num-queries refuse.
        with pytest.raises(rc.RunError, match="num-queries"):
            rc.load_plan(_strip("vllm", "no_n_vllm.json"))
        with pytest.raises(rc.RunError, match="num-queries"):
            rc.load_plan(_strip("hf", "no_n_hf.json"))
        # argv value drifted from the step's own record.
        plan = json.loads(json.dumps(plan_a))
        cell = next(s for s in _cells(plan) if s["row_class"] == "primary")
        cell["argv"][cell["argv"].index("--num-queries") + 1] = "800"
        with pytest.raises(rc.RunError, match="num-queries"):
            rc.load_plan(_dump(tmp_path, plan, "drift.json"))
        # step record drifted from the registered class n (a primary cell
        # relabeled to 800 with a consistent argv is still a mislabeled row).
        plan = json.loads(json.dumps(plan_a))
        cell = next(s for s in _cells(plan) if s["row_class"] == "primary")
        cell["num_queries"] = 800
        cell["argv"][cell["argv"].index("--num-queries") + 1] = "800"
        with pytest.raises(rc.RunError, match="num_queries"):
            rc.load_plan(_dump(tmp_path, plan, "relabel.json"))
        # row_class drifted.
        plan = json.loads(json.dumps(plan_a))
        cell = next(s for s in _cells(plan) if s["row_class"] == "primary")
        cell["row_class"] = "secondary"
        with pytest.raises(rc.RunError, match="row_class"):
            rc.load_plan(_dump(tmp_path, plan, "reclass.json"))
        # header without the registration refuses.
        plan = json.loads(json.dumps(plan_a))
        del plan["per_row_n"]
        with pytest.raises(rc.RunError, match="per_row_n"):
            rc.load_plan(_dump(tmp_path, plan, "no_header.json"))
        # and the untouched plan loads.
        assert rc.load_plan(_dump(tmp_path, plan_a, "fresh_n.json"))["counts"]["cells"] == 870

    def test_run_passes_num_queries_to_the_runner(self, tmp_path, floor_table, stub):
        plan = _stub_plan(_tiny_grid(f1_baselines=("B1", "B3")), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 0
        cell_calls = [c for c in stub.calls() if "--baseline" in c["argv"]]
        by_label = {c["argv"][c["argv"].index("--baseline-label") + 1]: c["argv"] for c in cell_calls}
        assert by_label["B1_gold-fresh"][by_label["B1_gold-fresh"].index("--num-queries") + 1] == "800"
        assert by_label["B3_corpus-reuse"][by_label["B3_corpus-reuse"].index("--num-queries") + 1] == "2000"


# ---------------------------------------------------------------------------
# Backlog A5 (retrieval pins) + F5a (per-cell Redis namespaces)
# ---------------------------------------------------------------------------


def _retrieval_cells(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in _cells(plan) if s["cellspec"]["retriever"] != "none"]


def _b7_cells(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in _cells(plan) if s["cellspec"]["arm"] == "retr-reuse"]


class TestRetrievalPinsA5:
    """Backlog A5: the runner's retrieval knobs (--top-k, --embedding-model,
    --ir-index-dir, CAGE_DISTRACTOR_DOCS) are REGISTERED driver values pinned
    on every retrieval cell, never runner defaults; the embedding model is
    read from the ADR-0099 freeze slot (INSTRUMENT_REVISIONS.dense_retriever),
    never a second hard-coded copy."""

    def test_named_constants(self):
        assert rc.RETRIEVAL_TOP_K == 3
        assert rc.DISTRACTOR_DOCS == 1000
        assert rc.DEFAULT_IR_INDEX_ROOT == "./experiments/ir_index"
        assert rc.SessionGrid.__dataclass_fields__["ir_index_root"].default == rc.DEFAULT_IR_INDEX_ROOT
        assert rc.FREEZE_DENSE_RETRIEVER_SLOT == "dense_retriever"
        assert rc.FREEZE_FILE_ENV_VAR == "CAGE_FREEZE_RESOLUTIONS"
        assert rc.DEFAULT_FREEZE_FILE == (
            rc.REPO_ROOT / "MyDocs" / "registration" / "freeze_resolutions.json"
        )

    def test_pins_resolve_from_the_freeze_slot(self, freeze_file):
        pins = rc.resolve_retrieval_pins(freeze_file)
        assert pins.embedding_model == _FREEZE_EMBEDDING_MODEL
        assert pins.embedding_revision == _FREEZE_EMBEDDING_REVISION
        assert pins.freeze_file == str(freeze_file.resolve())
        # The env seam is the default source (no explicit path).
        assert rc.resolve_retrieval_pins() == pins

    def test_the_freeze_pin_is_read_not_hard_coded(self, tmp_path, floor_table, stub):
        # A different registered model flows through to every retrieval
        # cell: the driver carries no second copy of the id that could drift
        # from the freeze artifact.
        doc = _freeze_doc()
        doc["INSTRUMENT_REVISIONS"]["dense_retriever"]["model"] = "org/other-encoder"
        doc["INSTRUMENT_REVISIONS"]["dense_retriever"]["revision"] = "d" * 40
        path = _write_freeze(tmp_path / "other.json", doc)
        pins = rc.resolve_retrieval_pins(path)
        assert pins.embedding_model == "org/other-encoder"
        assert pins.embedding_revision == "d" * 40
        grid = _tiny_grid(f1_baselines=("B5", "B6"))
        floor = rc.load_floor_table(floor_table)
        orig = rc.SESSION_GRIDS
        rc.SESSION_GRIDS = {grid.session: grid}
        try:
            plan = rc.build_plan(
                grid.session, floor, window_duration_s=60.0, runner_cmd=stub.cmd,
                launcher_cmds={"vllm": stub.cmd}, freeze_file=path,
            )
        finally:
            rc.SESSION_GRIDS = orig
        assert plan["behavior_knobs"]["embedding_model"] == "org/other-encoder"
        for s in _cells(plan):
            assert _argv_value(s, "--embedding-model") == "org/other-encoder"

    @pytest.mark.parametrize(
        "spoil, match",
        [
            (lambda d: None, "missing"),  # file absent
            (lambda d: "not json", "not valid JSON"),
            (lambda d: {"no": "revisions"}, "INSTRUMENT_REVISIONS"),
            (lambda d: _freeze_doc({}), "dense_retriever"),
            # Only the QUALITY embedder is registered: never consumed as a
            # fallback for the retriever.
            (lambda d: _freeze_doc({"embedding": d["INSTRUMENT_REVISIONS"]["embedding"]}), "dense_retriever"),
            (lambda d: _freeze_doc({"dense_retriever": {"model": "", "revision": "x"}}), "model"),
            (lambda d: _freeze_doc({"dense_retriever": {"model": "m"}}), "revision"),
            (lambda d: _freeze_doc({"dense_retriever": "intfloat/e5-large-v2"}), "dense_retriever"),
        ],
    )
    def test_freeze_refusals_are_typed(self, tmp_path, spoil, match):
        path = tmp_path / "spoiled.json"
        doc = spoil(_freeze_doc())
        if doc is None:
            pass  # absent file
        elif isinstance(doc, str):
            path.write_text(doc, encoding="utf-8")
        else:
            _write_freeze(path, doc)
        with pytest.raises(rc.PlanError, match=match):
            rc.resolve_retrieval_pins(path)

    def test_build_plan_refuses_without_the_freeze_artifact(self, tmp_path, floor_table, monkeypatch):
        monkeypatch.setenv(rc.FREEZE_FILE_ENV_VAR, str(tmp_path / "absent.json"))
        floor = rc.load_floor_table(floor_table)
        with pytest.raises(rc.PlanError, match="freeze"):
            rc.build_plan("a", floor, window_duration_s=300.0)
        # An explicit path wins over the env seam.
        good = _write_freeze(tmp_path / "explicit.json", _freeze_doc())
        assert rc.build_plan("a", floor, window_duration_s=300.0, freeze_file=good)["counts"]["cells"] == 870

    def test_every_retrieval_cell_pins_top_k_model_and_index_root(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            root = plan["behavior_knobs"]["ir_index_root"]
            retrieval = _retrieval_cells(plan)
            assert retrieval
            for s in retrieval:
                assert _argv_value(s, "--top-k") == str(rc.RETRIEVAL_TOP_K)
                assert _argv_value(s, "--embedding-model") == _FREEZE_EMBEDDING_MODEL
                # Review 2026-09-17 defect 5: the revision is ENFORCED (runner
                # passes it to SentenceTransformer), not merely recorded.
                assert _argv_value(s, "--embedding-revision") == _FREEZE_EMBEDDING_REVISION
                assert _argv_value(s, "--ir-index-dir") == root
                assert s["env"]["CAGE_DISTRACTOR_DOCS"] == str(rc.DISTRACTOR_DOCS)
            for s in _cells(plan):
                if s["cellspec"]["retriever"] == "none":
                    for flag in ("--top-k", "--embedding-model", "--embedding-revision", "--ir-index-dir"):
                        assert flag not in s["argv"], (s["baseline"], flag)
                    assert "CAGE_DISTRACTOR_DOCS" not in s["env"]

    def test_ir_index_root_is_a_registered_grid_path(self, floor_table, stub):
        grid = _tiny_grid(f1_baselines=("B5", "B6"), ir_index_root="/data/ir_index")
        plan = _stub_plan(grid, floor_table, stub.cmd)
        assert plan["behavior_knobs"]["ir_index_root"] == "/data/ir_index"
        for s in _cells(plan):
            assert _argv_value(s, "--ir-index-dir") == "/data/ir_index"
        for bad in ("", "   ", None, 3):
            with pytest.raises(rc.PlanError, match="ir_index_root"):
                _tiny_grid(ir_index_root=bad)

    def test_distractor_docs_is_not_identity(self, plan_a):
        # CAGE_DISTRACTOR_DOCS rides the cell env as BEHAVIOR: derive_cell_spec
        # ignores it (identity rides the CAGE_CELL_* seam and nothing else).
        cell = _retrieval_cells(plan_a)[0]
        argv = cell["argv"]

        def val(flag: str) -> str:
            return argv[argv.index(flag) + 1]

        def derive(env: Dict[str, str]) -> str:
            return derive_cell_spec(
                baseline=val("--baseline"),
                baseline_label=val("--baseline-label"),
                backend=val("--backend"),
                model=val("--model"),
                env=env,
            ).to_row_key()

        with_env = dict(cell["env"])
        assert "CAGE_DISTRACTOR_DOCS" in with_env
        without = {k: v for k, v in with_env.items() if k != "CAGE_DISTRACTOR_DOCS"}
        altered = {**with_env, "CAGE_DISTRACTOR_DOCS": "7"}
        assert derive(with_env) == derive(without) == derive(altered) == cell["row_key"]

    def test_header_records_the_pins(self, plan_a, plan_b, freeze_file):
        for plan in (plan_a, plan_b):
            knobs = plan["behavior_knobs"]
            assert knobs["retrieval_top_k"] == 3
            assert knobs["distractor_docs"] == 1000
            assert knobs["embedding_model"] == _FREEZE_EMBEDDING_MODEL
            assert knobs["embedding_model_revision"] == _FREEZE_EMBEDDING_REVISION
            assert knobs["embedding_model_freeze_slot"] == "INSTRUMENT_REVISIONS.dense_retriever"
            assert knobs["embedding_model_freeze_file"] == str(freeze_file.resolve())
            assert knobs["embedding_model_adr"] == "ADR-0099"
            assert knobs["ir_index_root"] == rc.DEFAULT_IR_INDEX_ROOT
            assert knobs["retrieval_pins_backlog"] == "A5"

    def test_counts_unchanged_by_the_pins(self, plan_a, plan_b):
        # Argv/env pins never mint cells: the enumeration pins hold.
        c = plan_a["counts"]
        assert (c["cells"], c["windows"], c["relaunches"], c["blocked"]) == (870, 2610, 36, 65)
        c = plan_b["counts"]
        assert (c["cells"], c["windows"], c["relaunches"], c["blocked"]) == (352, 1056, 30, 65)

    def test_load_plan_refuses_stale_retrieval_pins(self, tmp_path, plan_a):
        def _cell(plan: Dict[str, Any]) -> Dict[str, Any]:
            return _retrieval_cells(plan)[0]

        # --top-k missing.
        plan = json.loads(json.dumps(plan_a))
        cell = _cell(plan)
        i = cell["argv"].index("--top-k")
        del cell["argv"][i:i + 2]
        with pytest.raises(rc.RunError, match="top-k"):
            rc.load_plan(_dump(tmp_path, plan, "no_top_k.json"))
        # --top-k drifted.
        plan = json.loads(json.dumps(plan_a))
        cell = _cell(plan)
        cell["argv"][cell["argv"].index("--top-k") + 1] = "5"
        with pytest.raises(rc.RunError, match="top-k"):
            rc.load_plan(_dump(tmp_path, plan, "top_k_5.json"))
        # --embedding-model drifted from the header's freeze pin.
        plan = json.loads(json.dumps(plan_a))
        cell = _cell(plan)
        cell["argv"][cell["argv"].index("--embedding-model") + 1] = "org/other"
        with pytest.raises(rc.RunError, match="embedding-model"):
            rc.load_plan(_dump(tmp_path, plan, "model_drift.json"))
        # --embedding-revision drifted from / missing against the header's pin.
        plan = json.loads(json.dumps(plan_a))
        cell = _cell(plan)
        cell["argv"][cell["argv"].index("--embedding-revision") + 1] = "deadbeef"
        with pytest.raises(rc.RunError, match="embedding-revision"):
            rc.load_plan(_dump(tmp_path, plan, "revision_drift.json"))
        plan = json.loads(json.dumps(plan_a))
        cell = _cell(plan)
        i = cell["argv"].index("--embedding-revision")
        del cell["argv"][i:i + 2]
        with pytest.raises(rc.RunError, match="embedding-revision"):
            rc.load_plan(_dump(tmp_path, plan, "revision_missing.json"))
        # Header without the revision pin: a pre-enforcement plan.
        plan = json.loads(json.dumps(plan_a))
        del plan["behavior_knobs"]["embedding_model_revision"]
        with pytest.raises(rc.RunError, match="embedding_model_revision"):
            rc.load_plan(_dump(tmp_path, plan, "no_header_revision.json"))
        # A retriever-none cell carrying the flag.
        plan = json.loads(json.dumps(plan_a))
        none_cell = next(s for s in _cells(plan) if s["cellspec"]["retriever"] == "none")
        none_cell["argv"] += ["--embedding-revision", _FREEZE_EMBEDDING_REVISION]
        with pytest.raises(rc.RunError, match="embedding-revision"):
            rc.load_plan(_dump(tmp_path, plan, "none_with_revision.json"))
        # --ir-index-dir drifted from the registered root.
        plan = json.loads(json.dumps(plan_a))
        cell = _cell(plan)
        cell["argv"][cell["argv"].index("--ir-index-dir") + 1] = "/elsewhere"
        with pytest.raises(rc.RunError, match="ir-index-dir"):
            rc.load_plan(_dump(tmp_path, plan, "root_drift.json"))
        # CAGE_DISTRACTOR_DOCS missing / drifted.
        plan = json.loads(json.dumps(plan_a))
        del _cell(plan)["env"]["CAGE_DISTRACTOR_DOCS"]
        with pytest.raises(rc.RunError, match="CAGE_DISTRACTOR_DOCS"):
            rc.load_plan(_dump(tmp_path, plan, "no_distractors.json"))
        plan = json.loads(json.dumps(plan_a))
        _cell(plan)["env"]["CAGE_DISTRACTOR_DOCS"] = "0"
        with pytest.raises(rc.RunError, match="CAGE_DISTRACTOR_DOCS"):
            rc.load_plan(_dump(tmp_path, plan, "zero_distractors.json"))
        # A non-retrieval cell carrying a retrieval pin is a mislabeled row.
        plan = json.loads(json.dumps(plan_a))
        gold = next(s for s in _cells(plan) if s["cellspec"]["retriever"] == "none")
        gold["argv"] += ["--top-k", "3"]
        with pytest.raises(rc.RunError, match="top-k"):
            rc.load_plan(_dump(tmp_path, plan, "gold_top_k.json"))
        # Header without the pins refuses (a pre-A5 plan).
        plan = json.loads(json.dumps(plan_a))
        del plan["behavior_knobs"]["embedding_model"]
        with pytest.raises(rc.RunError, match="embedding_model"):
            rc.load_plan(_dump(tmp_path, plan, "no_header_pin.json"))
        plan = json.loads(json.dumps(plan_a))
        del plan["behavior_knobs"]["ir_index_root"]
        with pytest.raises(rc.RunError, match="ir_index_root"):
            rc.load_plan(_dump(tmp_path, plan, "no_header_root.json"))
        # And the untouched plan loads.
        assert rc.load_plan(_dump(tmp_path, plan_a, "fresh_a5.json"))["counts"]["cells"] == 870

    def test_cli_plan_carries_freeze_file(self, tmp_path, floor_table):
        good = _write_freeze(tmp_path / "cli_freeze.json", _freeze_doc())
        out = tmp_path / "plan.json"
        code = rc.main([
            "plan", "--session", "a", "--floor-table", str(floor_table),
            "--window-duration-s", "300", "--freeze-file", str(good), "--out", str(out),
        ])
        assert code == 0
        plan = rc.load_plan(out)
        assert plan["behavior_knobs"]["embedding_model_freeze_file"] == str(good.resolve())


class TestRedisNamespacesF5a:
    """Backlog F5a: every B7 (retr-reuse) cell owns a Redis namespace minted
    from its row key (cage:<sha1(row_key)[:12]>) and flushes it at start, so
    no two cells ever share artifact-cache entries and each starts empty."""

    def test_prefix_rule(self):
        row = "retr-reuse|rerank|reuse|single|vllm|qwen3-14b|F1"
        import hashlib as _h
        expected = "cage:" + _h.sha1(row.encode("utf-8")).hexdigest()[:12]
        assert rc.redis_key_prefix_for_row(row) == expected
        assert rc.REDIS_CACHE_ARMS == frozenset({"retr-reuse"})
        assert rc.REDIS_KEY_PREFIX_ROOT == "cage"
        assert rc.REDIS_NAMESPACE_SHA_CHARS == 12
        with pytest.raises(rc.PlanError, match="row key"):
            rc.redis_key_prefix_for_row("")

    def test_every_b7_cell_owns_a_namespace_and_flushes_it(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            b7 = _b7_cells(plan)
            assert b7
            for s in b7:
                assert s["baseline"] == "B7"
                assert _argv_value(s, "--redis-key-prefix") == rc.redis_key_prefix_for_row(s["row_key"])
                assert "--flush-redis-namespace" in s["argv"]
            for s in _cells(plan):
                if s["cellspec"]["arm"] != "retr-reuse":
                    assert "--redis-key-prefix" not in s["argv"], s["baseline"]
                    assert "--flush-redis-namespace" not in s["argv"], s["baseline"]

    def test_no_two_cells_share_a_namespace(self, plan_a):
        b7 = _b7_cells(plan_a)
        by_row: Dict[str, str] = {}
        for s in b7:
            prefix = _argv_value(s, "--redis-key-prefix")
            by_row.setdefault(s["row_key"], prefix)
            assert by_row[s["row_key"]] == prefix
        prefixes = list(by_row.values())
        assert len(set(prefixes)) == len(prefixes) == len(by_row) >= 2
        # Two concrete cells, stated explicitly: different row keys, different
        # namespaces (the pre-F5a plan gave BOTH the runner's default 'cage').
        first, second = b7[0], next(s for s in b7 if s["row_key"] != b7[0]["row_key"])
        assert _argv_value(first, "--redis-key-prefix") != _argv_value(second, "--redis-key-prefix")
        assert _argv_value(first, "--redis-key-prefix") != "cage"

    def test_header_records_the_rule(self, plan_a):
        knobs = plan_a["behavior_knobs"]
        assert knobs["redis_cache_arms"] == ["retr-reuse"]
        assert knobs["redis_namespace_rule"] == "cage:<sha1(row_key)[:12]>"
        assert knobs["redis_namespace_backlog"] == "F5a"

    def test_load_plan_refuses_stale_namespaces(self, tmp_path, plan_a):
        # Prefix drifted (two cells would share entries).
        plan = json.loads(json.dumps(plan_a))
        cell = _b7_cells(plan)[0]
        cell["argv"][cell["argv"].index("--redis-key-prefix") + 1] = "cage"
        with pytest.raises(rc.RunError, match="redis-key-prefix"):
            rc.load_plan(_dump(tmp_path, plan, "shared_prefix.json"))
        # Prefix missing.
        plan = json.loads(json.dumps(plan_a))
        cell = _b7_cells(plan)[0]
        i = cell["argv"].index("--redis-key-prefix")
        del cell["argv"][i:i + 2]
        with pytest.raises(rc.RunError, match="redis-key-prefix"):
            rc.load_plan(_dump(tmp_path, plan, "no_prefix.json"))
        # Flush missing (the cell would start warm).
        plan = json.loads(json.dumps(plan_a))
        cell = _b7_cells(plan)[0]
        cell["argv"].remove("--flush-redis-namespace")
        with pytest.raises(rc.RunError, match="flush-redis-namespace"):
            rc.load_plan(_dump(tmp_path, plan, "no_flush.json"))
        # A non-B7 cell carrying a namespace is a mislabeled row.
        plan = json.loads(json.dumps(plan_a))
        b6 = next(s for s in _cells(plan) if s["baseline"] == "B6")
        b6["argv"] += ["--redis-key-prefix", "cage:deadbeef0000"]
        with pytest.raises(rc.RunError, match="redis-key-prefix"):
            rc.load_plan(_dump(tmp_path, plan, "b6_prefix.json"))

    def test_run_passes_the_namespace_to_the_runner(self, tmp_path, floor_table, stub):
        plan = _stub_plan(_tiny_grid(f1_baselines=("B6", "B7")), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 0
        cell_calls = [c for c in stub.calls() if "--baseline" in c["argv"]]
        by_label = {c["argv"][c["argv"].index("--baseline-label") + 1]: c for c in cell_calls}
        b7 = by_label["B7_retr-reuse"]
        row = next(s["row_key"] for s in _cells(plan) if s["baseline"] == "B7")
        assert b7["argv"][b7["argv"].index("--redis-key-prefix") + 1] == rc.redis_key_prefix_for_row(row)
        assert "--flush-redis-namespace" in b7["argv"]
        assert b7["env"]["CAGE_DISTRACTOR_DOCS"] == "1000"
        b6 = by_label["B6_retr-fresh"]
        assert "--redis-key-prefix" not in b6["argv"]
        assert b6["argv"][b6["argv"].index("--top-k") + 1] == "3"


# ---------------------------------------------------------------------------
# Backlog A10: uniform max_model_len per session (RULER SHAPE-32K + Qasper)
# ---------------------------------------------------------------------------


_SHAPE_32K_TOTAL = 32_512 + 256  # RULER input + output, restated independently


class TestMaxModelLenA10:
    """Backlog A10 (Tier A): RULER requests are 32,512 input + 256 output
    tokens and Qasper papers exceed 4,096 tokens, while the uniform-regime
    rule (_serving_config.sh) demands ONE max_model_len per session. The
    value is a registered SessionGrid knob (32768), rides EVERY relaunch of
    BOTH engines as VLLM_MAX_MODEL_LEN (vLLM --max-model-len; SGLang maps it
    to --context-length), is recorded on the relaunch step and in the plan
    header, and a stale plan lacking it is refused. The pilot shell default
    (4096) is untouched and must never reach a campaign relaunch."""

    def test_registered_constant_and_grid_field(self):
        assert rc.DEFAULT_MAX_MODEL_LEN == 32_768 == _SHAPE_32K_TOTAL
        assert rc.MAX_MODEL_LEN_ENV == "VLLM_MAX_MODEL_LEN"
        assert rc.RULER_CONTEXT_TOKENS + rc.RULER_OUTPUT_TOKENS == _SHAPE_32K_TOTAL
        for grid in rc.SESSION_GRIDS.values():
            assert grid.max_model_len == 32_768
            assert isinstance(grid.max_model_len, int)

    def test_env_on_every_relaunch_for_both_engines(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            relaunches = _relaunches(plan)
            assert relaunches
            assert {s["engine"] for s in relaunches} == {"vllm", "sglang"}
            for s in relaunches:
                assert s["max_model_len"] == 32_768, s
                assert s["env"]["VLLM_MAX_MODEL_LEN"] == "32768", s
                # the pilot default can never ride a campaign relaunch
                assert s["env"]["VLLM_MAX_MODEL_LEN"] != "4096"
        # Session b: the single, tp AND pd relaunch shapes all carry it
        # (the pd launcher reads the same env for both role instances).
        assert {s["topology"] for s in _relaunches(plan_b)} == {"single", "tp", "pd"}

    def test_header_records_it(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            shapes = plan["serving_shapes"]
            assert shapes["max_model_len"] == 32_768
            assert shapes["max_model_len_env"] == "VLLM_MAX_MODEL_LEN"
            assert shapes["max_model_len_backlog"] == "A10"

    def test_cell_steps_never_carry_it(self, plan_a):
        # The value is a SERVER dial (relaunch boundary), never cell argv or
        # cell env: identity rides the CAGE_CELL_* seam and nothing else.
        for s in _cells(plan_a):
            assert "VLLM_MAX_MODEL_LEN" not in s["env"]
            assert "--max-model-len" not in s["argv"]

    def test_ruler_grid_refuses_a_value_below_shape_32k(self):
        ruler_kwargs: Dict[str, Any] = dict(
            f2_baselines=("B1",),
            f2_budgets=(1.0,),
            f2_rates=(0.85,),
            f2_ruler_baselines=("B1",),
            f2_ruler_tasks=("qa",),
        )
        with pytest.raises(rc.PlanError, match="max_model_len=4096"):
            _tiny_grid(max_model_len=4096, **ruler_kwargs)
        with pytest.raises(rc.PlanError, match="32768"):
            _tiny_grid(max_model_len=_SHAPE_32K_TOTAL - 1, **ruler_kwargs)
        # exactly SHAPE-32K is the floor, not below it
        assert _tiny_grid(max_model_len=_SHAPE_32K_TOTAL, **ruler_kwargs).max_model_len == 32_768
        # a grid WITHOUT RULER tasks may register the pilot regime (no
        # RULER request would exceed it; Qasper coverage is the operator's
        # registered choice, reviewable in the header)
        assert _tiny_grid(max_model_len=4096).max_model_len == 4096

    @pytest.mark.parametrize("bad", [0, -1, True, 4096.0, "32768", None])
    def test_grid_refuses_non_integer_or_non_positive(self, bad):
        with pytest.raises(rc.PlanError, match="max_model_len"):
            _tiny_grid(max_model_len=bad)

    def test_registered_value_rides_the_relaunch(self, floor_table, stub):
        plan = _stub_plan(_tiny_grid(max_model_len=8192), floor_table, stub.cmd)
        (relaunch,) = _relaunches(plan)
        assert relaunch["max_model_len"] == 8192
        assert relaunch["env"]["VLLM_MAX_MODEL_LEN"] == "8192"
        assert plan["serving_shapes"]["max_model_len"] == 8192

    def test_load_plan_refuses_a_relaunch_without_it(self, tmp_path, plan_a):
        # record missing (a pre-A10 plan)
        plan = json.loads(json.dumps(plan_a))
        del _relaunches(plan)[0]["max_model_len"]
        with pytest.raises(rc.RunError, match="max_model_len"):
            rc.load_plan(_dump(tmp_path, plan, "no_record.json"))
        # env missing: the launcher would fall back to the pilot 4096
        plan = json.loads(json.dumps(plan_a))
        del _relaunches(plan)[0]["env"]["VLLM_MAX_MODEL_LEN"]
        with pytest.raises(rc.RunError, match="VLLM_MAX_MODEL_LEN"):
            rc.load_plan(_dump(tmp_path, plan, "no_env.json"))
        # env drifted from the record (the record would lie about the server)
        plan = json.loads(json.dumps(plan_a))
        _relaunches(plan)[-1]["env"]["VLLM_MAX_MODEL_LEN"] = "4096"
        with pytest.raises(rc.RunError, match="VLLM_MAX_MODEL_LEN"):
            rc.load_plan(_dump(tmp_path, plan, "env_drift.json"))
        # record disagrees with the header: a non-uniform session
        plan = json.loads(json.dumps(plan_a))
        step = _relaunches(plan)[5]
        step["max_model_len"] = 16_384
        step["env"]["VLLM_MAX_MODEL_LEN"] = "16384"
        with pytest.raises(rc.RunError, match="uniform"):
            rc.load_plan(_dump(tmp_path, plan, "non_uniform.json"))
        # record is not a positive integer
        plan = json.loads(json.dumps(plan_a))
        _relaunches(plan)[0]["max_model_len"] = "32768"
        with pytest.raises(rc.RunError, match="max_model_len"):
            rc.load_plan(_dump(tmp_path, plan, "str_record.json"))
        # header missing the value (a pre-A10 header)
        plan = json.loads(json.dumps(plan_a))
        del plan["serving_shapes"]["max_model_len"]
        with pytest.raises(rc.RunError, match="max_model_len"):
            rc.load_plan(_dump(tmp_path, plan, "no_header.json"))
        # a fresh plan still loads (the refusals above are the only change)
        assert rc.load_plan(_dump(tmp_path, plan_a, "fresh_a10.json"))["counts"]["cells"] == 870

    def test_run_passes_it_to_the_launcher(self, tmp_path, floor_table, stub):
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 0
        calls = stub.calls()
        launcher_call = calls[0]
        assert launcher_call["argv"][0] == "restart"
        assert launcher_call["env"]["VLLM_MAX_MODEL_LEN"] == "32768"

    def test_counts_unchanged_by_max_model_len(self, plan_a, plan_b):
        # A server dial on an existing boundary adds no cell, window,
        # relaunch or block (pins: TestPlanCountsSessionA / TestSessionB).
        c = plan_a["counts"]
        assert (c["cells"], c["windows"], c["relaunches"], c["blocked"]) == (870, 2610, 36, 65)
        c = plan_b["counts"]
        assert (c["cells"], c["windows"], c["relaunches"], c["blocked"]) == (352, 1056, 30, 65)


# ---------------------------------------------------------------------------
# Batch 2 finding W1 (ADR-0055 "serving writes, scoring reads"): every cell
# step pins CAGE_SKIP_QUALITY=1 so no campaign window is produced by inline
# model-based scoring inside the measured window
# ---------------------------------------------------------------------------


class TestDecoupledScoringW1:
    """ADR-0055 (accepted 2026-08-04) requires quality scoring as a separate
    post-serving pass. The runner honors it only under CAGE_SKIP_QUALITY=1
    and defaults to inline model scoring, so the driver pins the env on
    EVERY cell step (hf oracle and blocked cells included), records the
    rule in the header, and load_plan refuses a stale plan without it."""

    def test_registered_constants(self):
        assert rc.SKIP_QUALITY_ENV == "CAGE_SKIP_QUALITY"
        assert rc.SKIP_QUALITY_VALUE == "1"
        assert rc.DECOUPLED_SCORING_ADR == "ADR-0055"

    def test_every_cell_step_pins_the_env_and_no_relaunch_carries_it(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            cells = _cells(plan)
            assert cells
            for s in cells:
                assert s["env"]["CAGE_SKIP_QUALITY"] == "1", s["row_key"]
            # A server dial it is not: relaunch env stays exactly as pinned
            # by TestOrdering.test_budget_env_on_relaunch_steps.
            for s in _relaunches(plan):
                assert "CAGE_SKIP_QUALITY" not in s["env"]

    def test_pin_is_not_identity(self, plan_a):
        cell = _cells(plan_a)[0]
        argv = cell["argv"]

        def val(flag: str) -> str:
            return argv[argv.index(flag) + 1]

        def derive(env: Dict[str, str]) -> str:
            return derive_cell_spec(
                baseline=val("--baseline"),
                baseline_label=val("--baseline-label"),
                backend=val("--backend"),
                model=val("--model"),
                env=env,
            ).to_row_key()

        with_env = dict(cell["env"])
        assert with_env["CAGE_SKIP_QUALITY"] == "1"
        without = {k: v for k, v in with_env.items() if k != "CAGE_SKIP_QUALITY"}
        altered = {**with_env, "CAGE_SKIP_QUALITY": "0"}
        assert derive(with_env) == derive(without) == derive(altered) == cell["row_key"]

    def test_header_records_the_rule(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            knobs = plan["behavior_knobs"]
            assert knobs["quality_scoring"] == "decoupled"
            assert knobs["quality_scoring_env"] == "CAGE_SKIP_QUALITY"
            assert knobs["quality_scoring_adr"] == "ADR-0055"

    def test_load_plan_refuses_a_cell_without_the_pin(self, tmp_path, plan_a):
        # Missing: a pre-W1 plan would score inline inside the window.
        plan = json.loads(json.dumps(plan_a))
        del _cells(plan)[0]["env"]["CAGE_SKIP_QUALITY"]
        with pytest.raises(rc.RunError, match="CAGE_SKIP_QUALITY"):
            rc.load_plan(_dump(tmp_path, plan, "no_skip_quality.json"))
        # Drifted: an explicit inline switch is the same violation.
        plan = json.loads(json.dumps(plan_a))
        _cells(plan)[-1]["env"]["CAGE_SKIP_QUALITY"] = "0"
        with pytest.raises(rc.RunError, match="ADR-0055"):
            rc.load_plan(_dump(tmp_path, plan, "inline_scoring.json"))
        # The hf oracle cell is pinned too (the evaluator is CPU-side either way).
        plan = json.loads(json.dumps(plan_a))
        hf = next(s for s in _cells(plan) if s["cellspec"]["engine"] == "hf")
        del hf["env"]["CAGE_SKIP_QUALITY"]
        with pytest.raises(rc.RunError, match="CAGE_SKIP_QUALITY"):
            rc.load_plan(_dump(tmp_path, plan, "hf_no_pin.json"))
        # And the untouched plan loads.
        assert rc.load_plan(_dump(tmp_path, plan_a, "fresh_w1.json"))["counts"]["cells"] == 870

    def test_run_pin_wins_over_the_operator_shell(self, tmp_path, floor_table, stub, monkeypatch):
        # _exec copies the shell env and applies the step env on top: an
        # exported inline switch never reaches a cell.
        monkeypatch.setenv("CAGE_SKIP_QUALITY", "0")
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 0
        cell_calls = [c for c in stub.calls() if "--baseline" in c["argv"]]
        assert len(cell_calls) == 2
        for call in cell_calls:
            assert call["env"]["CAGE_SKIP_QUALITY"] == "1"

    def test_counts_unchanged_by_the_pin(self, plan_a, plan_b):
        c = plan_a["counts"]
        assert (c["cells"], c["windows"], c["relaunches"], c["blocked"]) == (870, 2610, 36, 65)
        c = plan_b["counts"]
        assert (c["cells"], c["windows"], c["relaunches"], c["blocked"]) == (352, 1056, 30, 65)


# ---------------------------------------------------------------------------
# Batch 2 finding W2 (owner picked option A): every server-engine cell pins
# --api-base and every relaunch exports the launcher port env, both from the
# ONE port table that mirrors the frozen launchers' defaults
# ---------------------------------------------------------------------------


#: (script, the exact default line the table mirrors): a launcher default
#: drift fails here, never at 3 a.m. on the pod.
_LAUNCHER_PORT_LINES = (
    ("scripts/2_serving/manage_vllm_server.sh", 'PORT="${VLLM_PORT:-8000}"'),
    ("scripts/2_serving/manage_sglang_server.sh", 'PORT="${SGLANG_PORT:-30000}"'),
    ("scripts/2_serving/manage_vllm_pd.sh", 'PROXY_PORT="${CAGE_PD_PROXY_PORT:-8000}"'),
    ("scripts/2_serving/manage_vllm_pd.sh", 'PREFILL_PORT="${CAGE_PD_PREFILL_PORT:-8100}"'),
    ("scripts/2_serving/manage_vllm_pd.sh", 'DECODE_PORT="${CAGE_PD_DECODE_PORT:-8200}"'),
)
_ENDPOINT_OF = {"vllm": "http://localhost:8000", "sglang": "http://localhost:30000"}
_PORT_ENV_OF = {"vllm": ("VLLM_PORT", "8000"), "sglang": ("SGLANG_PORT", "30000")}


class TestEngineEndpointsW2:
    """Before W2 no cell carried an endpoint: every cell rode the runner's
    --api-base default (http://localhost:8000, the vLLM port), so every SGLang
    cell was sent to a port no SGLang server listens on. The driver now
    derives BOTH the launcher port env and the cell endpoint from
    ENGINE_PORTS, load_plan refuses a stale plan per cell and per relaunch,
    and 'run' refuses while a runner override env is exported."""

    def test_port_table_mirrors_the_frozen_launchers(self):
        assert rc.ENGINE_PORTS == {"vllm": 8000, "sglang": 30000}
        assert rc.PORT_LAUNCH_ENV == {"vllm": "VLLM_PORT", "sglang": "SGLANG_PORT"}
        assert rc.PD_PROXY_PORT == 8000
        assert rc.PD_PROXY_PORT_ENV == "CAGE_PD_PROXY_PORT"
        assert rc.API_BASE_OVERRIDE_ENVS == ("CAGE_SGLANG_API_BASE", "CAGE_LMDEPLOY_API_BASE")
        assert rc.PD_ROLE_PORT_ENVS == {"CAGE_PD_PREFILL_PORT": 8100, "CAGE_PD_DECODE_PORT": 8200}
        assert rc.SHELL_PORT_ENVS == {
            "VLLM_PORT": 8000, "SGLANG_PORT": 30000, "CAGE_PD_PROXY_PORT": 8000,
            "CAGE_PD_PREFILL_PORT": 8100, "CAGE_PD_DECODE_PORT": 8200,
        }
        for rel, line in _LAUNCHER_PORT_LINES:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            assert line in text, f"{rel}: launcher default drifted from {line!r}"
        assert rc.engine_api_base("vllm") == "http://localhost:8000"
        assert rc.engine_api_base("sglang") == "http://localhost:30000"
        assert rc.engine_api_base("sglang", "tp") == "http://localhost:30000"
        assert rc.engine_api_base("vllm", "pd") == "http://localhost:8000"
        for engine in ("lmdeploy", "hf"):
            with pytest.raises(rc.PlanError, match="no registered port"):
                rc.engine_api_base(engine)
        # Review F4: the pd proxy is vLLM's; no other engine may be pinned to it.
        with pytest.raises(rc.PlanError, match="no registered pd proxy"):
            rc.engine_api_base("sglang", "pd")

    def test_every_server_cell_pins_its_engine_endpoint_and_hf_none(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            seen = set()
            for s in _cells(plan):
                engine = s["cellspec"]["engine"]
                if engine == "hf":
                    assert "--api-base" not in s["argv"], s["row_key"]
                    continue
                want = (
                    "http://localhost:8000"
                    if s["cellspec"]["topology"] == "pd"
                    else _ENDPOINT_OF[engine]
                )
                assert _argv_value(s, "--api-base") == want, s["row_key"]
                seen.add((engine, s["cellspec"]["topology"]))
            assert {("vllm", "single"), ("sglang", "single")} <= seen
        assert {("vllm", "tp"), ("vllm", "pd")} <= {
            (s["cellspec"]["engine"], s["cellspec"]["topology"]) for s in _cells(plan_b)
        }
        # The pre-W2 failure, stated: no SGLang cell dials the vLLM port.
        assert not [
            s for s in _cells(plan_a)
            if s["cellspec"]["engine"] == "sglang"
            and _argv_value(s, "--api-base") == "http://localhost:8000"
        ]

    def test_every_relaunch_exports_the_port_env_and_records_the_endpoint(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            for s in _relaunches(plan):
                if s["topology"] == "pd":
                    assert s["env"]["CAGE_PD_PROXY_PORT"] == "8000"
                    # Review F3: the role ports the telemetry endpoints name
                    # are exported from the same constants.
                    assert s["env"]["CAGE_PD_PREFILL_PORT"] == "8100"
                    assert s["env"]["CAGE_PD_DECODE_PORT"] == "8200"
                    assert s["env"]["CAGE_TELEMETRY_ENDPOINTS"] == (
                        "prefill=http://localhost:8100,decode=http://localhost:8200"
                    )
                    assert s["api_base"] == "http://localhost:8000"
                    assert "VLLM_PORT" not in s["env"]
                    continue
                port_env, port = _PORT_ENV_OF[s["engine"]]
                assert s["env"][port_env] == port, s
                assert s["api_base"] == f"http://localhost:{port}"
                # one engine, one port env: never the other engine's
                assert not ({"VLLM_PORT", "SGLANG_PORT"} - {port_env}) & set(s["env"]), s
        assert {s["topology"] for s in _relaunches(plan_b)} == {"single", "tp", "pd"}

    def test_cells_dial_the_port_their_relaunch_launched(self, plan_a, plan_b):
        # Agreement by construction, restated by walking the plan: each
        # executable server cell's --api-base names the port carried by the
        # env of the relaunch it runs under.
        for plan in (plan_a, plan_b):
            current = None
            checked = 0
            for s in plan["steps"]:
                if s["kind"] == "relaunch":
                    current = s
                    continue
                if s["serving"] is None or s["blocked_on"]:
                    continue
                assert current is not None
                api = _argv_value(s, "--api-base")
                assert api == current["api_base"], s["row_key"]
                port_env = (
                    "CAGE_PD_PROXY_PORT"
                    if current["topology"] == "pd"
                    else _PORT_ENV_OF[current["engine"]][0]
                )
                assert api.endswith(":" + current["env"][port_env]), s["row_key"]
                checked += 1
            assert checked > 0

    def test_header_records_the_table(self, plan_a, plan_b):
        for plan in (plan_a, plan_b):
            shapes = plan["serving_shapes"]
            assert shapes["engine_ports"] == {"vllm": 8000, "sglang": 30000}
            assert shapes["port_launch_env"] == {"vllm": "VLLM_PORT", "sglang": "SGLANG_PORT"}
            assert shapes["pd_proxy_port"] == 8000
            assert shapes["pd_proxy_port_env"] == "CAGE_PD_PROXY_PORT"
            assert shapes["api_base_override_envs"] == [
                "CAGE_SGLANG_API_BASE", "CAGE_LMDEPLOY_API_BASE",
            ]
            assert shapes["engine_ports_finding"] == "Batch 2 W2"

    def test_counts_and_row_keys_unchanged_by_the_endpoint(self, plan_a, plan_b):
        # Argv on existing cells: the enumeration pins hold, and the identity
        # seam never reads argv (test_identity_env_roundtrips_through_derive_cell_spec).
        c = plan_a["counts"]
        assert (c["cells"], c["windows"], c["relaunches"], c["blocked"]) == (870, 2610, 36, 65)
        c = plan_b["counts"]
        assert (c["cells"], c["windows"], c["relaunches"], c["blocked"]) == (352, 1056, 30, 65)

    def test_load_plan_refuses_stale_endpoints(self, tmp_path, plan_a, plan_b):
        def _sglang(plan):
            return next(s for s in _cells(plan) if s["cellspec"]["engine"] == "sglang")

        # A server cell without the flag (a pre-W2 plan).
        plan = json.loads(json.dumps(plan_a))
        cell = _sglang(plan)
        i = cell["argv"].index("--api-base")
        del cell["argv"][i:i + 2]
        with pytest.raises(rc.RunError, match="api-base"):
            rc.load_plan(_dump(tmp_path, plan, "no_api_base.json"))
        # The exact pre-W2 failure by hand: an SGLang cell pointed at the vLLM port.
        plan = json.loads(json.dumps(plan_a))
        cell = _sglang(plan)
        cell["argv"][cell["argv"].index("--api-base") + 1] = "http://localhost:8000"
        with pytest.raises(rc.RunError, match="30000"):
            rc.load_plan(_dump(tmp_path, plan, "sglang_on_8000.json"))
        # An hf cell carrying an endpoint.
        plan = json.loads(json.dumps(plan_a))
        hf = next(s for s in _cells(plan) if s["cellspec"]["engine"] == "hf")
        hf["argv"] += ["--api-base", "http://localhost:8000"]
        with pytest.raises(rc.RunError, match="hf cell"):
            rc.load_plan(_dump(tmp_path, plan, "hf_api_base.json"))
        # A relaunch without the port env (the launcher would serve on its shell default).
        plan = json.loads(json.dumps(plan_a))
        relaunch = next(s for s in _relaunches(plan) if s["engine"] == "sglang")
        del relaunch["env"]["SGLANG_PORT"]
        with pytest.raises(rc.RunError, match="SGLANG_PORT"):
            rc.load_plan(_dump(tmp_path, plan, "no_port_env.json"))
        # A relaunch whose port env drifted (a server on a port no cell dials).
        plan = json.loads(json.dumps(plan_a))
        relaunch = next(s for s in _relaunches(plan) if s["engine"] == "vllm")
        relaunch["env"]["VLLM_PORT"] = "8001"
        with pytest.raises(rc.RunError, match="VLLM_PORT"):
            rc.load_plan(_dump(tmp_path, plan, "port_drift.json"))
        # The pd relaunch without the proxy port env (session b).
        plan = json.loads(json.dumps(plan_b))
        pd = next(s for s in _relaunches(plan) if s["topology"] == "pd")
        del pd["env"]["CAGE_PD_PROXY_PORT"]
        with pytest.raises(rc.RunError, match="CAGE_PD_PROXY_PORT"):
            rc.load_plan(_dump(tmp_path, plan, "no_proxy_port.json"))
        # Review F3: the pd relaunch with a drifted role port env.
        plan = json.loads(json.dumps(plan_b))
        pd = next(s for s in _relaunches(plan) if s["topology"] == "pd")
        pd["env"]["CAGE_PD_PREFILL_PORT"] = "8300"
        with pytest.raises(rc.RunError, match="CAGE_PD_PREFILL_PORT"):
            rc.load_plan(_dump(tmp_path, plan, "role_port_drift.json"))
        # Review F1: a relaunch whose api_base record drifted or is missing.
        plan = json.loads(json.dumps(plan_a))
        _relaunches(plan)[0]["api_base"] = "http://localhost:8001"
        with pytest.raises(rc.RunError, match="api_base record"):
            rc.load_plan(_dump(tmp_path, plan, "record_drift.json"))
        plan = json.loads(json.dumps(plan_a))
        del _relaunches(plan)[0]["api_base"]
        with pytest.raises(rc.RunError, match="api_base"):
            rc.load_plan(_dump(tmp_path, plan, "no_record.json"))
        # Review F1: an SGLang cell moved under a vLLM relaunch dials 30000
        # while the boundary it sits under serves 8000; whatever SGLang server
        # survived an earlier boundary would serve it (mislabeled data).
        plan = json.loads(json.dumps(plan_a))
        cell = next(
            s for s in _cells(plan)
            if s["cellspec"]["engine"] == "sglang" and s["blocked_on"] is None
        )
        vllm_relaunch = next(s for s in _relaunches(plan) if s["engine"] == "vllm")
        plan["steps"].remove(cell)
        plan["steps"].insert(plan["steps"].index(vllm_relaunch) + 1, cell)
        with pytest.raises(rc.RunError, match="relaunch serving"):
            rc.load_plan(_dump(tmp_path, plan, "moved_cell.json"))
        # And the untouched plans load.
        assert rc.load_plan(_dump(tmp_path, plan_a, "fresh_w2_a.json"))["counts"]["cells"] == 870
        assert rc.load_plan(_dump(tmp_path, plan_b, "fresh_w2_b.json"))["counts"]["cells"] == 352

    def test_blocked_cell_without_a_registered_endpoint_carries_none(
        self, tmp_path, floor_table, stub
    ):
        # Review F4: an SGLang pd cell is enumerated BLOCKED (the pd launcher
        # is vLLM-only); it has no registered endpoint, so it carries no
        # --api-base (the gpu_count rule: visible debt, never a guess), the
        # plan still builds and loads, and giving it one is refused.
        grid = _tiny_grid(dist_cells=(("B3", "sglang", "pd"),))
        plan = _stub_plan(grid, floor_table, stub.cmd)
        (dist,) = [s for s in _cells(plan) if s["family"] == "DIST"]
        assert dist["blocked_on"] and "PD launcher" in dist["blocked_on"]
        assert "--api-base" not in dist["argv"]
        assert rc.load_plan(_dump(tmp_path, plan, "blocked_pd.json"))["counts"]["blocked"] == 1
        tampered = json.loads(json.dumps(plan))
        (dist,) = [s for s in _cells(tampered) if s["family"] == "DIST"]
        dist["argv"] += ["--api-base", "http://localhost:8000"]
        with pytest.raises(rc.RunError, match="no registered endpoint"):
            rc.load_plan(_dump(tmp_path, tampered, "blocked_pd_pinned.json"))

    @pytest.mark.parametrize("name", ["CAGE_SGLANG_API_BASE", "CAGE_LMDEPLOY_API_BASE"])
    @pytest.mark.parametrize("value", ["http://elsewhere:1", ""])
    def test_run_refuses_while_a_runner_override_env_is_exported(
        self, tmp_path, floor_table, stub, monkeypatch, name, value
    ):
        # The runner resolves the override BEFORE --api-base, and _exec
        # inherits the shell: refused on presence, before the first step.
        monkeypatch.setenv(name, value)
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        with pytest.raises(rc.RunError, match=name):
            rc.run_plan(plan, _run_root(tmp_path))
        assert stub.calls() == [], "nothing may execute under an endpoint override"

    def test_run_passes_the_port_to_the_launcher_and_the_endpoint_to_the_runner(
        self, tmp_path, floor_table, stub, monkeypatch
    ):
        for name in ("VLLM_PORT", "SGLANG_PORT"):
            monkeypatch.delenv(name, raising=False)
        plan = _stub_plan(_tiny_grid(f1_engines=("vllm", "sglang")), floor_table, stub.cmd)
        root = _run_root(tmp_path)
        assert rc.run_plan(plan, root) == 0
        calls = stub.calls()
        launchers = [c for c in calls if c["argv"][0] == "restart"]
        cells = [c for c in calls if "--baseline" in c["argv"]]
        assert len(launchers) == 2 and len(cells) == 4
        # One relaunch per engine, each carrying ITS port env and not the other's.
        assert {
            (c["env"].get("VLLM_PORT"), c["env"].get("SGLANG_PORT")) for c in launchers
        } == {(None, "30000"), ("8000", None)}
        for c in cells:
            backend = c["argv"][c["argv"].index("--backend") + 1]
            assert c["argv"][c["argv"].index("--api-base") + 1] == _ENDPOINT_OF[backend]

    @pytest.mark.parametrize(
        "name, value",
        [("VLLM_PORT", "8001"), ("SGLANG_PORT", " 30001 "), ("CAGE_PD_PREFILL_PORT", "8300")],
    )
    def test_run_refuses_a_shell_port_that_differs_from_the_table(
        self, tmp_path, floor_table, stub, monkeypatch, name, value
    ):
        # Review F5: the preflight dials the shell value while every relaunch
        # pins the registered port, so a differing shell value would make the
        # preflight evidence come from a port the campaign never serves on.
        monkeypatch.setenv(name, value)
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        with pytest.raises(rc.RunError, match=name):
            rc.run_plan(plan, _run_root(tmp_path))
        assert stub.calls() == [], "nothing may execute under a differing shell port"

    def test_run_accepts_a_shell_port_equal_to_the_table(
        self, tmp_path, floor_table, stub, monkeypatch
    ):
        # An equal shell value is fine, and the step env still carries the
        # registered port to the launcher (_exec applies it on top of the shell).
        monkeypatch.setenv("VLLM_PORT", "8000")
        monkeypatch.setenv("SGLANG_PORT", "30000")
        plan = _stub_plan(_tiny_grid(), floor_table, stub.cmd)
        assert rc.run_plan(plan, _run_root(tmp_path)) == 0
        (launcher,) = [c for c in stub.calls() if c["argv"][0] == "restart"]
        assert launcher["env"]["VLLM_PORT"] == "8000"
