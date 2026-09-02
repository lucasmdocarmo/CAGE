"""Pins for scripts/3_run/run_campaign.py — THE campaign sweep driver (T1.2).

WHAT is pinned and WHY:

- **Session-'a' enumeration integers** (532 cells / 1596 windows / 30
  relaunches / 13 blocked; per-family 96+10 / 300 / 126 cells): the grid is
  the charter's registered design (§6.1 full 5×6 factorial, §6.8 reduced 3×3,
  §7.6.1 family × group matrix, 3 replications per grid point per D6 §6.3).
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
    grid: str = "full",
    r_values=(1.5, 1.0, 0.75, 0.5, 0.25),
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
        "env": {k: v for k, v in os.environ.items() if k.startswith("CAGE_")},
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
    """§6.1/§6.8/§7.6.1 densities, derived independently and pinned exactly."""

    def test_total_counts(self, plan_a):
        # F1: 12 baselines × 2 engines × 4 datasets              =  96
        # F1 HF oracle: B3×4 + {B1,B2,B6}×2                      =  10
        # F2: 5 FRESH × 2 engines × 5 budgets × 6 rates          = 300
        # F3: 7 REUSE × 2 engines × 3 budgets × 3 rates          = 126
        assert plan_a["counts"]["cells"] == 532
        # 3 windows per grid point (D6 §6.3) ⇒ 532 × 3 = 1596
        assert plan_a["counts"]["windows"] == 1596
        # Relaunch boundaries = distinct EXECUTABLE serving configs
        # (engine, prefix, budget, kv_dtype, connector); hf is in-process (0):
        #   vllm:   F1 {plain, fp8·B10, lmcache·B8}                   =  3
        #           F3 3 budgets × {plain, fp8, lmcache}              =  9
        #           F2 5 budgets, all plain (FRESH set has no lever)  =  5
        #   sglang: F1 {plain, fp8}   (B8 BLOCKED: no connector knob) =  2
        #           F3 3 budgets × {plain, fp8}                       =  6
        #           F2 5 budgets, plain                               =  5
        # ⇒ 17 + 13 = 30
        assert plan_a["counts"]["relaunches"] == 30
        # Blocked: retr-store (B8) on sglang — the frozen launcher has no
        # KV-store connector knob: F1 4 datasets + F3 3×3 = 13. Serving them
        # connector-free would duplicate plain rag under a B8 label.
        assert plan_a["counts"]["blocked"] == 13

    def test_blocked_cells_are_exactly_sglang_retr_store(self, plan_a):
        blocked = [s for s in _cells(plan_a) if s["blocked_on"]]
        assert len(blocked) == 13
        for s in blocked:
            assert s["baseline"] == "B8"
            assert s["cellspec"]["arm"] == "retr-store"
            assert s["cellspec"]["engine"] == "sglang"
            assert s["blocked_on"] == rc.SGLANG_STORE_BLOCKED_ON
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
        assert by == {"F1": 96, "F1-hf": 10, "F2": 300, "F3": 126}
        windows = {k: 3 * v for k, v in by.items()}
        assert windows == {"F1": 288, "F1-hf": 30, "F2": 900, "F3": 378}

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
        assert {s["cellspec"]["budget_r"] for s in f2} == {1.5, 1.0, 0.75, 0.5, 0.25}
        assert {s["cellspec"]["rate_frac"] for s in f2} == {0.5, 0.7, 0.85, 0.95, 1.05, 1.2}
        assert {s["baseline"] for s in f2} == {"B1", "B5", "B6", "B9", "B11"}
        # F2 is THE prefix-OFF family; everything else serves ON.
        assert all(s["serving"]["prefix_mode"] == "OFF" for s in f2)

    def test_f3_grid_values(self, plan_a):
        f3 = [s for s in _cells(plan_a) if s["family"] == "F3"]
        assert {s["cellspec"]["budget_r"] for s in f3} == {1.0, 0.5, 0.25}
        assert {s["cellspec"]["rate_frac"] for s in f3} == {0.85, 0.95, 1.05}
        assert {s["baseline"] for s in f3} == {"B2", "B3", "B4", "B7", "B8", "B10", "B12"}
        assert all(s["serving"]["prefix_mode"] == "ON" for s in f3)


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
        assert len(_relaunches(plan_a)) == len(configs) == 30

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
        # — plus the launch levers (KV dtype / connector) and nothing else.
        for s in _relaunches(plan_a):
            env = dict(s["env"])
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


class TestBehaviorRealization:
    def test_corpus_arms_carry_the_corpus_block(self, plan_a):
        # B3/B4/B10 serve the SHARED corpus-as-prefix block
        # (run_prefix_envelope.sh cag_true_*: --corpus-prefix-budget); B12
        # serves the TRUNCATED block — the budget delta IS the whole
        # B12-vs-B3 one-slot contrast ("store less than you know").
        knobs = plan_a["behavior_knobs"]
        full, trunc = (
            knobs["corpus_prefix_budget_tokens"],
            knobs["corpus_trunc_budget_tokens"],
        )
        assert 0 < trunc < full
        for s in _cells(plan_a):
            arm = s["cellspec"]["arm"]
            if arm in ("corpus-reuse", "corpus-fresh", "corpus-comp"):
                assert _argv_value(s, "--corpus-prefix-budget") == str(full)
            elif arm == "corpus-trunc":
                assert _argv_value(s, "--corpus-prefix-budget") == str(trunc)
            else:
                # gold/retrieval arms never serve a corpus block
                assert "--corpus-prefix-budget" not in s["argv"]

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
            else:
                assert retriever == "rerank"
                assert _argv_value(s, "--reranker-model") == rc.RERANKER_MODEL

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
        assert plan["counts"]["cells"] == 532
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
        # 'b' is §1 vocabulary but its grid registration is Wave-3 work: the
        # driver must refuse loudly, never fabricate a grid.
        with pytest.raises(rc.PlanError, match="NOT yet registered"):
            rc.build_plan("b", rc.load_floor_table(floor_table), window_duration_s=60.0)

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
