"""S2 experiment B (session s2b): experiment A's grid on Qwen3-235B-A22B-Instruct-2507-FP8 and MuSiQue.

Owner rulings: 2026-11-21 ("ok drop B7 from both, 8 arms. lets go"; "Yes, all
three", decision D3 (a)) and 2026-11-22 ("Yes, all seven", decisions B-D1 to
B-D7). Spec Block, measurements and record:
MyDocs/RunPod/S2_PLAN_A_B_QWEN3_235B_2026-11-19.md, sections 20.2 (the twelve
acceptance tests, numbered in the section headers below) and 20.9.

- B-D1: the slug qwen3-235b-fp8. B-D7: group C, beside the 80B.
- B-D2: 846 questions per no-load reference, s2a's count. Owner 2026-11-22
  ("Rebuild on 846 questions", after the independent review): the manifest's
  trial pool is those 846, so every question a window serves under load has a
  no-load reference, as on s2a.
- B-D3: S2B.env names this session; S2R.env stays the record of the 80B
  rehearsals (tests/test_s2_rehearsal_profile.py pins its values on their own).
- D3 (a): A's gate list, one corpus class for B3, B4 and B10, B5 over BM25.
- The class token counts were measured on 2026-11-22 with the model's own
  tokenizer, the corpus arms on the rebuilt manifest's blocks
  (experiments/S1/2026-10-23_2350/plots/demand_tables_20261122_pool846/).
- Owner 2026-11-22: "180 min and 89h" (the scoring bound and the seatbelt).
- Owner 2026-11-22 ("Search whole split"; "Yes, all three", decisions W-D1 to
  W-D3; plan section 20.15): the retrieval arms search every passage of the
  MuSiQue split. The grid registers the session's own count of extra passages,
  21,100, where every other session keeps 1,000.
- Review 2026-11-22: the plan header states the grid's own reason for the 846
  (M2), the engine start timeouts are this landing's (the 80B's first vLLM
  start took 518 s of the common 600), and the bounds are read from the
  master's own plan print (L7).

Fixtures: tests/test_run_campaign.py (the S2 plan stack),
tests/test_s2_block_f_batch3.py (the shipped profile reader and the gate's
window tree), tests/test_s2a_e3_registration.py (the run id mint and the grid
digests of the other sessions), tests/test_s2a_eight_arms.py (what a plan step
does, without its position).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, get_args

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_kv_pool_capture_s0f24 as kpt  # noqa: E402
import test_run_campaign as trc  # noqa: E402
import test_s2_block_f_batch3 as f3  # noqa: E402
import test_s2a_e3_registration as e3  # noqa: E402
import test_s2a_eight_arms as e8  # noqa: E402
import run_campaign_analysis as rca  # noqa: E402  (scripts/4_analysis is on the path once trc is imported)
from src.analysis import cellspec  # noqa: E402
from src.analysis import regime_gate_rules as rules  # noqa: E402
from src.analysis.stats import families  # noqa: E402
from src.orchestration import cache_budget, campaign_session  # noqa: E402

from test_kv_pool_capture_s0f24 import stub_bin  # noqa: E402,F401  (fixture re-export: pkill is an inert stub)

rc = trc.rc
org = trc._org
layout = trc._layout

MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
MODEL_REVISION = "e156cb4efae43fbee1a1ab073f946a1377e6b969"
SLUG = "qwen3-235b-fp8"
#: owner 2026-11-22 ("Rebuild on 846 questions"): the pool is 898 questions (the first 180 blocks of the
#: 490-block packing), 52 of them the warm-up reserve, so every trial reorders the same 846, as on s2a
MANIFEST = "data/manifests/musique_846x3_seed42_wr52_tok-Qwen3-235B-A22B-Instruct-2507-FP8.json"
MANIFEST_SHA256 = "f3eab1ed9ae8432d437bc2feb5c1dabc97390710ceab7fd7769a56863d31fa54"
#: the 80B session's manifest: the same packing run to 490 blocks (the two tokenizers count alike)
FULL_PACKING = "data/manifests/musique_2000x3_seed42_wr50_tok-Qwen3-Next-80B-A3B-Instruct.json"
MASTER = REPO_ROOT / "scripts" / "6_experiments" / "cage_experiment.sh"
RESULT = (
    REPO_ROOT / "experiments" / "S1" / "2026-10-23_2350" / "plots" / "demand_tables_20261122_pool846"
    / "demand_tables_musique_Qwen3-235B-A22B-Instruct-2507-FP8_manifest.json"
)
PROFILES = REPO_ROOT / "scripts" / "6_experiments" / "profiles"
GPU = "NVIDIA H200"
#: the seven helper models every S2 landing prefetches beside its served model
HELPERS = (
    "intfloat/e5-large-v2", "BAAI/bge-reranker-large", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    "KRLabsOrg/lettucedect-base-modernbert-en-v1", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "sentence-transformers/all-MiniLM-L6-v2", "roberta-base",
)

ARMS = ("B1", "B2", "B3", "B4", "B5", "B6", "B9", "B10")
GATE_ARMS = "B1 B3 B4 B6"
ANCHOR = 2694
#: arm -> mean served tokens (prompt plus answer), the result file's rounded
#: means; the three corpus arms register B3's count (D3 (a))
ARM_TOKENS = {
    "gold-fresh": 2694, "gold-reuse": 2694,
    "corpus-fresh": 2395, "corpus-reuse": 2395, "corpus-comp": 2395,
    "retr-fresh": 506, "retr-reuse": 517, "retr-store": 506,
    "retr-comp": 320, "retr-trunc": 221,
}
#: class -> its largest request, the largest over the class's arms (2,777 on
#: each of the three corpus arms)
CLASS_MAX = {2694: 4939, 2395: 2777, 694: 804, 517: 1092, 506: 1196, 320: 589, 221: 587}
#: class -> request cap: the session cap on the anchor, the ADR-0158 rule elsewhere
CAPS = {2694: 16384, 2395: 8192, 694: 2048, 517: 4096, 506: 4096, 320: 2048, 221: 2048}
#: the classes the eight arms are served in
CLASSES = [320, 506, 2395, 2694]
BY_BASELINE = {"B1": 2694, "B2": 2694, "B3": 2395, "B4": 2395, "B10": 2395, "B5": 506, "B6": 506, "B9": 320}

#: Regression pins read on 2026-11-22 from commit eadfd50, before session s2b
#: was registered (the working tree under PYTHONHASHSEED 0 and 12345, and a
#: copy of the commit under seed 7: equal): sha256 of
#: repr(get_session_grid("s2a")), and of the fixture plans' steps as
#: ``_plan_digest`` reads them.
S2A_GRID_SHA256 = "88d86c52801bbb1c80723fc1cc2b1c4569a0c475ac476d63de98c111756550e2"
S2A_PLAN_SHA256 = "989b91806eff649a7eb584bbbf4428a53303a485c07e8889b95471a5828f468f"
CD_ACT1_PLAN_SHA256 = "a086916532511d7d388cdd57163f2e39dc8a773eb74482de1374474cc44a6bd3"

#: the grid fields in which s2b differs from s2a; every other field is equal
GRID_DIFFERS = {
    "session", "group", "model", "f1_datasets", "f2_dataset", "f3_dataset", "blocks", "achievable_n",
    "achievable_n_basis", "window_pool_n", "max_model_len", "demand_seq_tokens", "corpus_trunc_demand_seq_tokens",
    "demand_class_max_served_tokens", "demand_seq_tokens_source", "demand_class_request_cap_tokens",
    "distractor_docs",
}
#: owner 2026-11-22 ("Search whole split", decision W-D2 (a)): the number of
#: distinct passages in the MuSiQue validation split, registered as session
#: s2b's count of extra passages; from 11,450 up the runner indexes them all
WHOLE_SPLIT_DOCS = 21_100
DOCS_ENV = "CAGE_DISTRACTOR_DOCS"
#: the profile keys in which S2B.env differs from S2A.env; every other key
#: S2A.env states is equal in S2B.env
PROFILE_DIFFERS = {
    "EXP", "SESSION", "QUERY_MANIFESTS", "CALIBRATION_MANIFEST", "MODEL", "MODEL_SLUG", "PREFETCH_MODELS",
    "FLOOR_AVG_SEQ_TOKENS", "HF_TOKEN_FILE", "SETUP_BOUND_MIN", "DISK_GB", "SCORE_BOUND_MIN", "HOURS",
    "RUN_BUDGET_HOURS", "SEATBELT",
}
#: the keys S2B.env states and S2A.env leaves at the common value (600 s)
S2B_ONLY_KEYS = {"VLLM_START_TIMEOUT", "SGLANG_START_TIMEOUT"}


def _grid() -> Any:
    return rc.get_session_grid("s2b")


def _plan(session: str, tmp_path: Path, **overrides: Any) -> Dict[str, Any]:
    anchor = int(rc.get_session_grid(session).demand_seq_tokens[rc.DEMAND_ANCHOR_ARM])
    return trc._s2_plan(session, tmp_path, anchor=anchor, **overrides)


@pytest.fixture(scope="module")
def plan(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    return _plan("s2b", tmp_path_factory.mktemp("s2b"))


def _stated(name: str) -> Dict[str, str]:
    """Every KEY=VALUE line a profile states (comments left out, the last statement wins)."""
    out: Dict[str, str] = {}
    for line in (PROFILES / name).read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([A-Z_][A-Z0-9_]*)=(.*)", line)
        if m:
            out[m.group(1)] = m.group(2).strip('"')
    return out


def _canon(x: Any) -> Any:
    """Floats to six significant digits (a last-digit difference between the
    Mac's and the pod's arithmetic must not move the digest; counts, bytes and
    tokens are integers and stay exact); mappings by sorted key."""
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    raise TypeError(type(x))


def _plan_digest(plan: Dict[str, Any]) -> str:
    steps = [_canon(e8._essence(s)) for s in plan["steps"]]
    return hashlib.sha256(json.dumps(steps, sort_keys=True).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# acceptance test 1: the model's rows
# --------------------------------------------------------------------------- #


def test_the_models_kv_row_is_its_configs_arithmetic() -> None:
    """2 (keys and values) x 94 layers x 4 KV heads x 128 x 2 bytes per token
    [V config.json at the pinned revision]; a dense entry, no recurrent state."""
    row = cache_budget.MODEL_KV[SLUG]
    assert (row.kv_bytes_per_token, row.formula) == (192_512, "2*94*4*128*2")
    # the registered formula computes the registered number (the checkpoint's own config is read in the next test)
    assert math.prod(int(factor) for factor in row.formula.split("*")) == row.kv_bytes_per_token
    assert not row.hybrid and not row.mla_tp_replicated
    assert cache_budget.fixed_state_bytes_per_seq(SLUG, 4) == 0
    # what 50 requests of the anchor class hold: 25.93 GB on the pod
    assert cache_budget.demand_bytes(SLUG, concurrency=50, avg_seq_tokens=ANCHOR, tp=4) == 50 * ANCHOR * 192_512 == 25_931_366_400


def test_the_kv_row_is_the_arithmetic_of_the_checkpoints_own_config() -> None:
    """Review 2026-11-22 L7: the row read against config.json at the pinned
    revision, where this machine holds it (no network: the Hub cache only)."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(MODEL_ID, "config.json", revision=MODEL_REVISION)
    if not isinstance(path, str):
        pytest.skip(f"config.json of {MODEL_ID} at {MODEL_REVISION} is not in this machine's Hub cache")
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    # the head dimension is the config's own key: hidden_size / num_attention_heads is 64 on this model
    layers, kv_heads, head_dim = cfg["num_hidden_layers"], cfg["num_key_value_heads"], cfg["head_dim"]
    assert (layers, kv_heads, head_dim, cfg["torch_dtype"]) == (94, 4, 128, "bfloat16")
    row = cache_budget.MODEL_KV[SLUG]
    assert row.kv_bytes_per_token == 2 * layers * kv_heads * head_dim * 2
    assert row.formula == f"2*{layers}*{kv_heads}*{head_dim}*2"


def test_the_hub_id_and_the_slug_map_both_ways() -> None:
    assert campaign_session.HF_MODEL_SLUGS[MODEL_ID] == SLUG
    assert rc.HF_ID_OF_SLUG[SLUG] == MODEL_ID
    assert campaign_session.resolve_model_slug(MODEL_ID, env={}) == SLUG
    assert campaign_session.resolve_model_slug(SLUG, env={}) == SLUG
    assert SLUG in get_args(cellspec.Model)


def test_a_cell_of_the_model_has_a_row_key_that_reads_back() -> None:
    spec = cellspec.CellSpec.from_baseline(
        "B3", engine="sglang", model=SLUG, family="F3", topology="single", budget_r=0.25, rate_frac=1.05,
    )
    key = spec.to_row_key()
    assert key == "corpus-reuse|none|none|single|sglang|qwen3-235b-fp8|F3|r0.25|lam1.05"
    assert org.parse_row_key_dir(key) == spec
    # S2 serves no LMDeploy cell; the model is not on that engine's list
    with pytest.raises(cellspec.InvalidCellSpecError, match="lmdeploy"):
        cellspec.CellSpec.from_baseline("B3", engine="lmdeploy", model=SLUG)


def test_the_model_joins_group_c_beside_the_80b() -> None:
    """B-D7 (a): nothing reads the group map backward, so two models share C."""
    assert org.GROUP_OF_MODEL[SLUG] == "C" == org.GROUP_OF_MODEL["qwen3-next-80b"]
    assert _grid().group == "C"
    assert set(org.GROUP_OF_MODEL) == set(get_args(cellspec.Model)) == set(layout._MODELS)


def test_the_other_models_rows_are_untouched() -> None:
    """Regression pin (passes before and after)."""
    row = cache_budget.MODEL_KV["qwen3-next-80b"]
    assert (row.kv_bytes_per_token, row.attn_layers, row.kv_heads_total, row.head_dim, row.state_layers) == (24_576, 12, 2, 256, 36)
    assert cache_budget.MODEL_KV["llama-3.3-70b"].kv_bytes_per_token == 327_680
    assert campaign_session.HF_MODEL_SLUGS["Qwen/Qwen3-Next-80B-A3B-Instruct"] == "qwen3-next-80b"
    assert rc.HF_ID_OF_SLUG["qwen3-next-80b"] == "Qwen/Qwen3-Next-80B-A3B-Instruct"
    assert rc.FREE_BYTES_AFTER_WEIGHTS[("qwen3-next-80b", GPU)] == 70 * 1024 ** 3
    assert rc.FREE_BYTES_AFTER_WEIGHTS[("llama-3.3-70b", GPU)] == 75 * 1024 ** 3
    assert rc.VLLM_HYBRID_STORE_BLOCKED_MODELS == frozenset({"qwen3-next-80b"})


# --------------------------------------------------------------------------- #
# acceptance test 2: the session and its grid
# --------------------------------------------------------------------------- #


def test_the_session_is_registered_in_every_vocabulary() -> None:
    assert "s2b" in rc.SESSION_GRIDS and _grid().session == "s2b"
    assert "s2b" in layout.SESSIONS and layout.SESSIONS == org.SESSIONS == rc.SESSIONS
    # the analysis compiles the pressure families of this session on MuSiQue
    assert families.pressure_dataset("s2b") == "musique"
    roster = ("squad_v2", "musique", "qasper")
    for family in ("F2", "F3", "DIST"):
        assert families.registered_datasets(family, roster, session="s2b") == ("musique",)
    assert families.registered_datasets("F1", roster, session="s2b") == roster
    # the other sessions keep theirs
    assert families.pressure_dataset("cd-act1") == "musique"
    assert families.pressure_dataset("s2a") == families.pressure_dataset("a") == families.pressure_dataset(None) == "qasper"


def test_the_run_id_mints_under_the_run_id_grammar() -> None:
    rid = e3._mint("s2b", SLUG)
    assert org.RUN_ID_RE.match(rid), rid
    assert rid.endswith("-s2b-qwen3-235b-fp8") and len(rid) == 34   # the grammar holds 41


def test_the_grid_is_experiment_as_on_another_model_and_dataset() -> None:
    """Acceptance tests 2 and 12: every field of the two grids is equal except
    the listed ones, so the arms, the rungs, the loads, the references and the
    question count are A's by construction."""
    a, b = rc.get_session_grid("s2a"), _grid()
    for field in dataclasses.fields(a):
        same = getattr(a, field.name) == getattr(b, field.name)
        assert same != (field.name in GRID_DIFFERS), field.name
    assert b.f1_baselines == rc.S2A_ARMS == ARMS
    assert b.f2_budgets == b.f3_budgets == rc.S2A_RUNGS == (1.0, 0.25)
    assert b.f2_rates == b.f3_rates == (0.95, 1.05)

    def shape(p: Any) -> tuple:
        return (p.block, p.baselines, p.engines, p.rungs, p.loads, p.windows, p.topology, p.ruler_tasks)

    assert [shape(p) for p in b.blocks] == [shape(p) for p in a.blocks]
    assert [p.dataset for p in b.blocks] == ["musique", "musique"]
    assert [(p.loads[0].frac, p.windows) for p in b.blocks] == [(0.95, 2), (1.05, 3)]


def test_the_grids_own_values() -> None:
    b = _grid()
    assert (b.session, b.group, b.model) == ("s2b", "C", SLUG)
    assert b.f1_datasets == ("musique",) and b.f2_dataset == b.f3_dataset == "musique"
    # B-D2 (a): s2a's count, read from s2a's grid (review L9), and the whole trial pool of the rebuilt
    # manifest; the pressure pool is 500
    assert dict(b.achievable_n) == {"musique": rc.get_session_grid("s2a").achievable_n["qasper"]} == {"musique": 846}
    assert dict(b.window_pool_n) == {"musique": 500}
    assert b.n_secondary == rc.N_PRIMARY == 2000
    # the session cap is the rule's cap for the anchor's largest request (4,939 tokens), as on cd-act1 (ADR-0192)
    assert b.max_model_len == 16_384 == rc._rule_request_cap(rc.DEFAULT_MAX_MODEL_LEN, CLASS_MAX[ANCHOR])
    assert b.serving_tp == 4 and not b.identity_gated and b.helpers_on_gpu
    assert b.f1_retriever_cells == (("B5", "vllm", "bm25"), ("B5", "sglang", "bm25"))
    assert b.hf_oracle_cells == () and b.dist_cells == () and b.f1_variant_cells == ()
    assert (b.dist_tp_size, b.dist_pd_role_gpus) == (None, (1, 1))
    assert (b.corpus_prefix_budget_tokens, b.corpus_trunc_budgets) == (2640, (700,))
    assert b.floor_budget_r == 1.0 and b.replications == 1 and b.f2_ruler_tasks == ()


def test_the_plan_header_states_why_s2b_serves_846_questions(plan: Dict[str, Any]) -> None:
    """Review 2026-11-22 M2: the header's general reason (the dataset's split
    cannot supply the registered n) is false on s2b, where MuSiQue supplies the
    2,000 cd-act1 serves and the owner chose A's count. The header states the
    grid's own reason."""
    grid = _grid()
    caveat = plan["per_row_n"]["achievable_n_caveat"]
    assert plan["per_row_n"]["achievable_n"] == {"musique": 846}
    assert caveat == f"achievable_n lowers the registered class n for musique=846 ({grid.achievable_n_basis})"
    # second review L1: the sentence about the references holds for one manifest, which it names (the
    # planner does not check the property: the profile's manifest is pinned by name and sha256 below)
    for part in ("owner 2026-11-22", "B-D2 (a)", "Rebuild on 846 questions", "MuSiQue supplies the registered 2,000",
                 Path(MANIFEST).name, "not for the A5 reason"):
        assert part in caveat, part
    assert "cannot supply" not in caveat and "pre-declared" not in caveat
    assert {name for name, g in rc.SESSION_GRIDS.items() if g.achievable_n_basis is not None} == {"s2b"}
    # a session whose dataset cannot supply the registered n keeps the general reason
    a = rc.get_session_grid("s2a")
    assert a.achievable_n_basis is None
    general = rc._achievable_n_caveat(a)
    assert "qasper=846" in general and "cannot supply the registered n" in general
    # the basis stays out of the repr, which the registration pins of the other sessions hash
    assert "achievable_n_basis" not in repr(grid)


def test_the_cells_are_the_eight_arms_on_one_stack() -> None:
    cells = rc.enumerate_cells(_grid())
    assert len(cells) == 82 and all(c.blocked_on is None for c in cells)
    assert {c.baseline_id for c in cells} == set(ARMS)
    assert {c.spec.model for c in cells} == {SLUG} and {c.dataset for c in cells} == {"musique"}
    assert {c.spec.engine for c in cells} == {"vllm", "sglang"}
    # no PD cell, no B8 (so no LMCache), no engine variant, no B7
    assert {c.spec.topology for c in cells} == {"single"} and {c.spec.family for c in cells} == {"F1", "F2", "F3"}
    assert not any(c.spec.arm in ("retr-store", "retr-reuse") or c.spec.engine_variant for c in cells)
    assert {c.spec.retriever for c in cells if c.baseline_id == "B5"} == {"dense", "bm25"}
    assert {c.spec.retriever for c in cells if c.baseline_id == "B5" and c.spec.family != "F1"} == {"dense"}


# --------------------------------------------------------------------------- #
# acceptance test 3: the fixture plan's counts and the ladders
# --------------------------------------------------------------------------- #


def test_the_plan_counts(plan: Dict[str, Any]) -> None:
    c = plan["counts"]
    assert (plan["session"], plan["model"]) == ("s2b", SLUG)
    # 8 arms x 2 engines x 2 rungs x 2 loads = 64 pressure cells, 2 + 3 windows per
    # (arm, engine, rung); 8 arms x 2 engines of references plus B5 over BM25 twice
    assert (c["cells"], c["windows"], c["relaunches"], c["dry_windows"], c["blocked"]) == (82, 178, 34, 14, 0)
    assert c["by_family"] == {
        "F1": {"cells": 18, "windows": 18},
        "F2": {"cells": 32, "windows": 80},
        "F3": {"cells": 32, "windows": 80},
    }
    assert len(plan["steps"]) == 226 and plan["blocked_row_keys"] == []
    rows = trc._s2_rows(plan)
    assert Counter(s["row_class"] for s in rows) == {"window": 64, "secondary": 16, "primary": 2}
    assert sum(s["windows"] for s in trc._cells(plan) if s.get("block") == "X1") == 160
    assert {s["baseline"] for s in rows} == set(ARMS)
    budgeted = [s for s in plan["steps"] if s["kind"] == "relaunch" and s.get("budget_r") is not None]
    assert len(budgeted) == 7 * 2 * 2 and {s["budget_r"] for s in budgeted} == {1.0, 0.25}
    assert plan["rung_calibration"]["required_rungs"] == {"sglang": [1.0, 0.25], "vllm": [1.0, 0.25]}


def test_every_reference_serves_846_questions_and_every_window_draws_from_500(plan: Dict[str, Any]) -> None:
    rows = trc._s2_rows(plan)
    refs = [s for s in rows if s["family"] == "F1"]
    assert len(refs) == 18 and {s["num_queries"] for s in refs} == {846}
    assert all(rc._argv_flag_value(s["argv"], "--num-queries") == "846" for s in refs)
    assert sorted(s["row_key"] for s in rows if s["row_class"] == "primary") == [
        "corpus-reuse|none|none|single|vllm|qwen3-235b-fp8|F1",
        "retr-fresh|rerank|none|single|vllm|qwen3-235b-fp8|F1",
    ]
    assert {s["num_queries"] for s in rows if s["row_class"] == "window"} == {500}
    assert set(_grid().f1_datasets) <= set(_grid().achievable_n)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_the_ladders_run_eight_triples_per_engine_over_the_four_classes(engine: str) -> None:
    grid = _grid()
    assert rc.session_demand_classes(grid) == CLASSES
    selection = rc.ladder_cells(grid, engine, rc.budgeted_rungs(grid), {}, all_classes=True)
    triples = sorted((role, rc.demand_seq_tokens(grid, c.spec), float(c.spec.budget_r)) for role, c in selection.roles())
    assert triples == sorted(
        (role, tokens, r)
        for role, tokens in (("anchor", 2694), ("smallest", 320), ("measured", 506), ("measured", 2395))
        for r in (1.0, 0.25)
    )
    assert not any(c.spec.arm == "retr-reuse" for _role, c in selection.roles())


# --------------------------------------------------------------------------- #
# acceptance test 4: the class tables as measured
# --------------------------------------------------------------------------- #


def test_the_arms_register_the_measured_counts_with_one_corpus_class() -> None:
    grid = _grid()
    assert dict(grid.demand_seq_tokens) == ARM_TOKENS
    # the grid requires the 700 rung's class; no s2b arm serves it
    assert dict(grid.corpus_trunc_demand_seq_tokens) == {700: 694}
    assert dict(grid.demand_class_max_served_tokens) == CLASS_MAX
    by_baseline = {c.baseline_id: rc.demand_seq_tokens(grid, c.spec) for c in rc.enumerate_cells(grid)}
    assert by_baseline == BY_BASELINE
    assert rc.session_demand_classes(grid) == CLASSES
    # the 80B session keeps its two corpus classes, as measured on 2026-11-01
    cd = rc.get_session_grid("cd-act1")
    assert (cd.demand_seq_tokens["corpus-reuse"], cd.demand_seq_tokens["corpus-fresh"]) == (2387, 2386)


def test_the_source_names_the_tokenizer_the_manifest_and_the_result_folder(plan: Dict[str, Any]) -> None:
    source = _grid().demand_seq_tokens_source
    for part in (MODEL_ID, MODEL_REVISION, Path(MANIFEST).name, "2026-10-23_2350", "demand_tables_20261122_pool846"):
        assert part in source, part
    assert source not in (rc.DEMAND_SEQ_TOKENS_SOURCE, rc.get_session_grid("cd-act1").demand_seq_tokens_source)
    assert plan["demand_classes"]["source"] == source
    assert plan["demand_classes"]["anchor_seq_tokens"] == ANCHOR
    assert plan["demand_classes"]["seq_tokens"] == dict(sorted(ARM_TOKENS.items()))


def test_the_registered_values_are_the_result_files() -> None:
    if not RESULT.is_file():
        pytest.skip(f"{RESULT.relative_to(REPO_ROOT)} is not on this machine (experiments/ is not in git)")
    doc = json.loads(RESULT.read_text(encoding="utf-8"))
    assert doc["target_tokenizer"] == {"id": MODEL_ID, "revision": MODEL_REVISION}
    assert (doc["dataset"], doc["engine"], doc["manifest"]) == ("musique", "vllm", MANIFEST)
    arms = doc["arms"]
    measured = {arm: rec["target_served_mean_int"] for arm, rec in arms.items()}
    # D3 (a): B4 and B10 measured 2,393 on the 303 S1 rows whose question is in
    # the manifest's pool, B3 2,395 on 739 rows of the same prompt; the three
    # register B3's count
    assert (measured["corpus-fresh"], measured["corpus-comp"], measured["corpus-reuse"]) == (2393, 2393, 2395)
    assert arms["corpus-fresh"]["target_prompt_mean"] == arms["corpus-comp"]["target_prompt_mean"]
    assert (arms["corpus-fresh"]["rows"], arms["corpus-reuse"]["rows"]) == (303, 739)
    # the S1 rows of a question outside the pool have no block in the manifest and are left out
    assert (arms["corpus-fresh"]["rows_missing_in_manifest"], arms["corpus-reuse"]["rows_missing_in_manifest"]) == (497, 1261)
    # every gold and retrieval prompt was rebuilt exactly as S1 served it
    for arm in ("gold-fresh", "gold-reuse", "retr-fresh", "retr-reuse", "retr-comp", "retr-trunc"):
        assert arms[arm]["rebuild_exact_rate"] == 1.0, arm
    grid = _grid()
    for arm, tokens in grid.demand_seq_tokens.items():
        if arm == "retr-store":
            want = measured["retr-fresh"]   # B8 is not on s2b; it takes retr-fresh's shape [D]
        elif arm in ("corpus-fresh", "corpus-comp"):
            want = measured["corpus-reuse"]
        else:
            want = measured[arm]
        assert tokens == want, arm
    assert dict(grid.corpus_trunc_demand_seq_tokens) == {700: measured["corpus-trunc"]}
    # a class's largest request is the largest over its arms (the file's own
    # class table keeps one arm per class, so it is not the comparand)
    largest: Dict[int, int] = {}
    for arm, rec in arms.items():
        cls = measured["corpus-trunc"] if arm == "corpus-trunc" else grid.demand_seq_tokens[arm]
        largest[cls] = max(largest.get(cls, 0), rec["target_served_max"])
    assert dict(grid.demand_class_max_served_tokens) == largest
    assert {arms[a]["target_served_max"] for a in ("corpus-reuse", "corpus-fresh", "corpus-comp")} == {2777}


def test_the_request_caps_follow_the_rule_and_none_is_registered(plan: Dict[str, Any]) -> None:
    grid = _grid()
    assert {c: rc.request_cap_tokens(grid, c) for c in CAPS} == CAPS
    assert rc.request_cap_tokens(grid, None) == 16_384
    assert set(CAPS) == set(grid.demand_seq_tokens.values()) | set(grid.corpus_trunc_demand_seq_tokens.values())
    assert dict(grid.demand_class_request_cap_tokens) == {}
    shapes = plan["serving_shapes"]
    assert shapes["max_model_len"] == 16_384
    assert shapes["request_caps"] == {str(c): cap for c, cap in CAPS.items()}
    assert "request_cap_explicit" not in shapes and shapes["request_cap_rule"] == rc.REQUEST_CAP_RULE
    # every served request sits at or below 0.34 of its cap (the corpus class is the closest: 2,777 of 8,192)
    closest = max(CLASS_MAX[c] / CAPS[c] for c in CLASSES)
    assert closest == 2777 / 8192 and closest < 0.34


# --------------------------------------------------------------------------- #
# acceptance test 5: the caches and the memory row
# --------------------------------------------------------------------------- #


def test_every_cache_at_the_profiles_concurrency_holds_one_capped_request() -> None:
    v = f3._shipped("S2B.env", "FLOOR_CONCURRENCY")
    assert v == {"FLOOR_CONCURRENCY": "50"}
    grid = _grid()
    cells = rc.enumerate_cells(grid)
    records = rc._mac_relaunch_records(grid, cells, int(v["FLOOR_CONCURRENCY"]))
    # seven serving groups (B5 shares B6's) at two rungs on two engines
    assert len(records) == 28 and {r["demand_class"]["seq_tokens"] for r in records} == set(CLASSES)
    for rec in records:
        pool = min(tokens for _label, tokens in rc._pool_tokens_of(rec["budget_plan"]))
        assert pool >= rec["max_model_len"] == CAPS[rec["demand_class"]["seq_tokens"]], rec
    # the quarter caches in tokens: floor(0.25 x 50 x the class), twice that for B10's 8-bit cache
    quarter = {
        (r["demand_class"]["seq_tokens"], r["budget_plan"]["kv_dtype"]): rc._pool_tokens_of(r["budget_plan"])[0][1]
        for r in records if r["budget_r"] == 0.25
    }
    assert quarter == {(2694, "bf16"): 33_675, (2395, "bf16"): 29_937, (2395, "fp8"): 59_875, (506, "bf16"): 6_325, (320, "bf16"): 4_000}
    assert rc.mac_pool_shortfalls(grid, cells, int(v["FLOOR_CONCURRENCY"])) == []


def _h200_row_gib(weight_bytes: float, card_gib: float = 141e9 / 1024 ** 3) -> float:
    """The planner's rule for one H200 of a four-GPU pod (the comment above
    FREE_BYTES_AFTER_WEIGHTS): vLLM's 0.90 of the card (141 GB listed, 131.32
    GiB), less a quarter of the weights, less a 10 GiB cut for the engine's own
    working memory [A]."""
    return card_gib * 0.90 - weight_bytes / 4 / 1024 ** 3 - 10


def test_the_memory_row_is_the_planners_rule_for_the_h200() -> None:
    """Plan 20.6 [D], review 2026-11-22 L7: the row is the whole GiB of the
    rule that gives the two rows registered before it, applied to this
    model's weight files (236,426,193,880 bytes over 24 files)."""
    rows = rc.FREE_BYTES_AFTER_WEIGHTS
    # the rule on the two bf16 models (2 bytes per parameter; the parameter counts of the rows' comment)
    assert math.floor(_h200_row_gib(70_553.7e6 * 2)) * 1024 ** 3 == rows[("llama-3.3-70b", GPU)]
    assert math.floor(_h200_row_gib(81_324.9e6 * 2)) * 1024 ** 3 == rows[("qwen3-next-80b", GPU)]
    assert _h200_row_gib(236_426_193_880) == pytest.approx(53.14, abs=0.01)
    assert math.floor(_h200_row_gib(236_426_193_880)) * 1024 ** 3 == rows[(SLUG, GPU)] == 53 * 1024 ** 3
    # review L3: each H200 of the 2026-11-14 pod reports 143,771 MiB (140.40 GiB); on that size the rule
    # leaves 61 GiB, so the registered 53 is conservative
    assert math.floor(_h200_row_gib(236_426_193_880, card_gib=143_771 / 1024)) == 61
    # the largest planned cache is the anchor's at r = 1.0: 6.04 GiB per GPU, 11 percent of the row
    grid = _grid()
    records = rc._mac_relaunch_records(grid, rc.enumerate_cells(grid), 50)
    largest = max(r["budget_plan"]["per_rank_bytes"] for r in records)
    assert largest == 25_931_366_400 // 4 == 6_482_841_600
    assert largest / (53 * 1024 ** 3) == pytest.approx(0.114, abs=0.001)
    assert rc.pool_ceiling_overruns(records, GPU) == []


def test_the_pool_ceiling_refuses_without_the_models_row(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = _grid()
    records = rc._mac_relaunch_records(grid, rc.enumerate_cells(grid), 50)
    # another GPU has no row for this model: the check never passes blind
    other = rc.pool_ceiling_overruns(records, "NVIDIA H100 80GB HBM3")
    assert len(other) == 28 and all(o["free_bytes"] is None and o["model"] == SLUG for o in other)
    monkeypatch.delitem(rc.FREE_BYTES_AFTER_WEIGHTS, (SLUG, GPU))
    rep = rc.preflight_registration_check(
        "s2b", floor_avg_seq_tokens=str(ANCHOR), query_manifests={"musique": str(REPO_ROOT / MANIFEST)},
        charter_datasets=("squad_v2", "musique", "qasper"), floor_concurrency=50, gpu_id=GPU,
    )
    assert len(rep["pool_ceiling_overruns"]) == 28
    (problem,) = rep["problems"]
    assert "FREE_BYTES_AFTER_WEIGHTS has no measured value for" in problem and SLUG in problem and GPU in problem


# --------------------------------------------------------------------------- #
# acceptance tests 6, 7 and 12: the profile and stage 0's registration check
# --------------------------------------------------------------------------- #


def test_the_built_manifest_holds_one_pool_of_846_questions_that_every_trial_reorders() -> None:
    """Owner 2026-11-22 ("Rebuild on 846 questions"): a reference serves the
    whole trial pool, so every question a window serves under load has a
    no-load reference, as on s2a (565 of 1,210 loaded questions had none on
    the manifest built first, a trial pool of 2,357; review M1). The pool is a prefix of the packing
    the 80B session serves: the first 180 of its 490 blocks."""
    raw = (REPO_ROOT / MANIFEST).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256
    doc = json.loads(raw)
    assert doc["tokenizer"] == {"id": MODEL_ID, "revision": MODEL_REVISION}
    assert (doc["dataset"], doc["num_queries"], doc["num_trials"], doc["seed"], doc["block_budget"]) == ("musique", 846, 3, 42, 2640)
    stats = doc["stats"]
    assert (stats["n_blocks"], stats["pool_size"], stats["warmup_reserve_n"], stats["trial_pool_size"]) == (180, 898, 52, 846)
    assert sorted(doc["trunc_rungs"]) == ["700"] and doc["warmup_reserve"]["n"] == 52
    trials = doc["trials"]
    pool = set(trials["1"])
    # every trial is the same 846 questions in another order
    assert sorted(trials) == ["1", "2", "3"] and all(len(t) == 846 and set(t) == pool for t in trials.values())
    assert len({tuple(t) for t in trials.values()}) == 3 and stats["trials_disjoint"] is False
    # a reference serves the first 846 ids of trial 1, window k the first 500 ids of trial k
    grid = _grid()
    n_ref, n_window = grid.achievable_n["musique"], grid.window_pool_n["musique"]
    referenced = set(trials["1"][:n_ref])
    loaded = set().union(*(set(t[:n_window]) for t in trials.values()))
    assert referenced == pool and loaded <= referenced and len(loaded) == 797
    # the reserve is outside every trial and completes the pool
    reserve = set(doc["warmup_reserve"]["ids"])
    q2b = doc["question_to_block"]
    assert len(reserve) == 52 and not reserve & pool and set(q2b) == pool | reserve
    # the references touch every block, so 666 of their 846 requests arrive on a block already served
    assert len({q2b[q] for q in trials["1"]}) == 180
    # the same packing as the 80B session's manifest, stopped after 180 blocks (the two tokenizers count alike)
    full = json.loads((REPO_ROOT / FULL_PACKING).read_text(encoding="utf-8"))
    assert len(full["blocks"]) == 490 and doc["blocks"] == full["blocks"][:180]
    assert doc["trunc_rungs"]["700"]["blocks"] == full["trunc_rungs"]["700"]["blocks"][:180]
    assert all(full["question_to_block"][q] == b for q, b in q2b.items())
    # s2a's manifest has the same shape: one pool of 846 that each of its trials reorders
    a = json.loads((REPO_ROOT / e3.MANIFEST).read_text(encoding="utf-8"))
    assert a["stats"]["trial_pool_size"] == 846 and len({frozenset(t) for t in a["trials"].values()}) == 1


def test_the_shipped_profile_names_the_session_the_model_and_the_manifest() -> None:
    v = f3._shipped(
        "S2B.env", "EXP", "SESSION", "MODEL", "MODEL_SLUG", "PIN_MODEL_REVISION", "QUERY_MANIFESTS", "CALIBRATION_MANIFEST",
        "PREFETCH_MODELS", "FLOOR_AVG_SEQ_TOKENS", "GPU_ID", "GPU_COUNT", "ENGINES", "CALIBRATE", "TOPOLOGY", "HF_TOKEN_FILE",
        "DISK_GB", "VOLUME_GB", "FRONTIER_LOADS", "RUN_CUT_BY_BUDGET", "CALIBRATION_BUDGET_FRACTION", "RUNG_ALL_CLASSES",
    )
    assert v == {
        "EXP": "S2B", "SESSION": "s2b", "MODEL": MODEL_ID, "MODEL_SLUG": SLUG, "PIN_MODEL_REVISION": "1",
        "QUERY_MANIFESTS": f"musique={MANIFEST}", "CALIBRATION_MANIFEST": MANIFEST,
        # the served model plus the seven helper models, as on S2A
        "PREFETCH_MODELS": " ".join((MODEL_ID,) + HELPERS),
        "FLOOR_AVG_SEQ_TOKENS": str(ANCHOR), "GPU_ID": GPU, "GPU_COUNT": "4", "ENGINES": "vllm sglang",
        "CALIBRATE": "vllm sglang", "TOPOLOGY": "tp",
        # the model is not gated: no token travels to this landing's pod
        "HF_TOKEN_FILE": "",
        # owner 2026-11-22 ("500 GB on S2B"): the weights are 236 GB against S2A's 141 GB on its 400 GB disk
        "DISK_GB": "500", "VOLUME_GB": "300", "FRONTIER_LOADS": "0", "RUN_CUT_BY_BUDGET": "",
        "CALIBRATION_BUDGET_FRACTION": "1.0", "RUNG_ALL_CLASSES": "1",
    }
    assert float(v["CALIBRATION_BUDGET_FRACTION"]) == _grid().floor_budget_r
    assert (REPO_ROOT / MANIFEST).is_file()


def test_the_shipped_profile_carries_no_lmcache_no_skip_and_no_failure_accepted_in_advance() -> None:
    """The ruling of 2026-11-21 took B8 out, and s2b has neither an identity
    gate nor a PD cell: nothing is installed for a store, no blocked row is
    skipped, and no gate failure is accepted before it is seen."""
    v = f3._shipped("S2B.env", "INSTALL_LMCACHE", "LMCACHE_VERSION", "SKIP_BLOCKED", "IDENTITY_FAIL_ACCEPTED", "PD_FAIL_ACCEPTED")
    assert v == {"INSTALL_LMCACHE": "0", "LMCACHE_VERSION": "", "SKIP_BLOCKED": "0", "IDENTITY_FAIL_ACCEPTED": "0", "PD_FAIL_ACCEPTED": "0"}
    stated = _stated("S2B.env")
    assert "IDENTITY_FAIL_ACCEPTED" not in stated and "PD_FAIL_ACCEPTED" not in stated


def test_the_two_profiles_are_one_design() -> None:
    """Acceptance tests 6 and 12: S2B.env states the keys S2A.env states (and
    its own two start timeouts), with the same values except the model's, the
    dataset's and the limits'; so the gate keys, the comparison list and the
    measures are A's."""
    a, b = _stated("S2A.env"), _stated("S2B.env")
    assert (set(a) - set(b), set(b) - set(a)) == (set(), S2B_ONLY_KEYS)
    differing = {k for k in a if a[k] != b[k]}
    assert differing <= PROFILE_DIFFERS, {k: (a[k], b[k]) for k in sorted(differing - PROFILE_DIFFERS)}
    for key in ("EXP", "SESSION", "QUERY_MANIFESTS", "CALIBRATION_MANIFEST", "MODEL", "MODEL_SLUG", "PREFETCH_MODELS",
                "FLOOR_AVG_SEQ_TOKENS", "HF_TOKEN_FILE"):
        assert a[key] != b[key], key
    # the two bounds that follow the size of the weights (236 GB against 141 GB): the setup time
    # and the pod disk (owner 2026-11-22: "500 GB on S2B"; S2A stays at 400)
    assert (a["SETUP_BOUND_MIN"], b["SETUP_BOUND_MIN"]) == ("40", "50")
    assert (a["DISK_GB"], b["DISK_GB"]) == ("400", "500")
    # owner 2026-11-22 ("180 min and 89h"): the scoring bound, and the seatbelt that covers the sum with it
    assert (a["SCORE_BOUND_MIN"], b["SCORE_BOUND_MIN"]) == ("120", "180")
    assert (a["SEATBELT"], b["SEATBELT"]) == ("88h", "89h")
    # the run bound and the budget stop are equal today; they are listed as allowed to differ because each
    # follows its own projection
    assert {k: b[k] for k in ("HOURS", "RUN_BUDGET_HOURS")} == {k: a[k] for k in ("HOURS", "RUN_BUDGET_HOURS")}
    shared = ("REGIME_GATE_LOADS", "MIN_IN_REGIME_WINDOWS", "CONFIRMATORY_ARMS", "CONTRASTS", "ANALYSIS_METRICS")
    assert {k: b[k] for k in shared} == {k: a[k] for k in shared} == {
        "REGIME_GATE_LOADS": "1.05", "MIN_IN_REGIME_WINDOWS": "1.05:3", "CONFIRMATORY_ARMS": GATE_ARMS,
        "CONTRASTS": "1 2 3 4 5 7 8", "ANALYSIS_METRICS": "ttft_ms predicate",
    }


def test_the_profile_states_its_own_engine_start_timeouts() -> None:
    """The 80B's first vLLM start on the 2026-11-14 pod took 518 s of the
    common 600 [V its engine log]; this model's weights are 45 percent larger
    (236 GB against 163 GB), so S2B.env states its own timeouts. S2A keeps the
    common value. Whether 1,000 s holds the 235B's first start is [?] until
    stage 4; this test pins the values, not that."""
    keys = ("VLLM_START_TIMEOUT", "SGLANG_START_TIMEOUT")
    assert f3._shipped("S2B.env", *keys) == dict.fromkeys(keys, "1000")
    assert f3._shipped("S2A.env", *keys) == dict.fromkeys(keys, "600")
    assert set(keys) == S2B_ONLY_KEYS <= set(_stated("S2B.env"))


def test_every_value_the_profile_states_is_the_value_the_master_sources() -> None:
    """Review 2026-11-22 L8: the tests here read the file's own lines; the
    master sources the file after _common.env. No key is stated twice and no
    value carries an expansion, so the two readings are equal."""
    text = (PROFILES / "S2B.env").read_text(encoding="utf-8")
    keys = re.findall(r"^([A-Z_][A-Z0-9_]*)=", text, re.M)
    stated = _stated("S2B.env")
    assert len(keys) == len(set(keys)) == len(stated) == 60
    assert f3._shipped("S2B.env", *sorted(stated)) == stated
    # second review L8: no line sets a variable in another form (an `export KEY=value` line would reach
    # the master without being a stated key)
    other = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#") and not re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", ln)]
    assert other == []


def test_stage_0_finds_every_pool_wide_enough_and_no_problem() -> None:
    """Acceptance test 7: the registration report on the shipped concurrency,
    anchor, GPU and manifest."""
    v = f3._shipped("S2B.env", "FLOOR_CONCURRENCY", "FLOOR_AVG_SEQ_TOKENS", "GPU_ID")
    rep = rc.preflight_registration_check(
        "s2b", floor_avg_seq_tokens=v["FLOOR_AVG_SEQ_TOKENS"], query_manifests={"musique": str(REPO_ROOT / MANIFEST)},
        charter_datasets=("squad_v2", "musique", "qasper"), floor_concurrency=int(v["FLOOR_CONCURRENCY"]), gpu_id=v["GPU_ID"],
    )
    assert (rep["cells"], rep["executable_cells"], rep["anchor_seq_tokens"]) == (82, 82, ANCHOR)
    assert rep["pool_shortfalls"] == [] and rep["pool_ceiling_overruns"] == [] and rep["problems"] == []
    # what stage 4 does on this session: no LMCache, no identity gate, no PD smoke, no extra-buffer probe
    assert (rep["needs_lmcache"], rep["lmcache_store_gib"], rep["identity_gated"], rep["pd_cells"], rep["pd_env"],
            rep["sglang_xb_probe"]) == (False, None, False, False, None, False)
    assert rep["tp_env"] == {"sglang": {"CAGE_SGLANG_TP": "4"}, "vllm": {"CAGE_VLLM_TENSOR_PARALLEL": "4"}}
    assert (rep["serving_tp"], rep["gpu_count_needed"]) == (4, 4)
    # the start stage 4 proves on each engine: the anchor's cache at r = 1.0, and
    # beside the helper models the retrieval class's
    assert {e: (p["seq_tokens"], p["budget_r"], p["planned_tokens"], p["planned_bytes"]) for e, p in rep["largest_pools"].items()} == {
        "vllm": (2694, 1.0, 134_700, 25_931_366_400), "sglang": (2694, 1.0, 134_700, 25_931_366_400),
    }
    assert {e: (p["seq_tokens"], p["planned_tokens"]) for e, p in rep["largest_helper_pools"].items()} == {
        "vllm": (506, 25_300), "sglang": (506, 25_300),
    }
    # a floor table sized on another anchor is refused
    other = rc.preflight_registration_check(
        "s2b", floor_avg_seq_tokens="4978", query_manifests={"musique": str(REPO_ROOT / MANIFEST)},
        charter_datasets=("musique",), floor_concurrency=50, gpu_id=GPU,
    )
    assert any(str(ANCHOR) in p for p in other["problems"]), other["problems"]


def test_the_shipped_gate_blocks_on_four_arms_at_load_105() -> None:
    v = f3._shipped("S2B.env", "SESSION", "REGIME_GATE_LOADS", "MIN_IN_REGIME_WINDOWS", "CONFIRMATORY_ARMS")
    assert v == {"SESSION": "s2b", "REGIME_GATE_LOADS": "1.05", "MIN_IN_REGIME_WINDOWS": "1.05:3", "CONFIRMATORY_ARMS": GATE_ARMS}
    doc = rc.regime_gate_projection(
        "s2b", loads=v["REGIME_GATE_LOADS"], need=v["MIN_IN_REGIME_WINDOWS"], confirmatory_arms=v["CONFIRMATORY_ARMS"],
    )
    assert doc["loads_without_a_group"] == [] and doc["short_descriptive"] == {} and doc["unresolved_load_cells"] == 0
    groups = list(doc["groups"].values())
    assert all(g["load"] == "1.05" and g["need"] == g["registered"] == 3 for g in groups)
    blocking = [g for g in groups if g["mode"] == rules.BLOCKING]
    assert (len(groups), len(blocking), sum(g["mode"] == rules.DESCRIPTIVE for g in groups)) == (32, 16, 16)
    assert {g["baseline"] for g in blocking} == set(GATE_ARMS.split())
    assert sum(g["need"] for g in blocking) == 48      # the windows that must all qualify
    assert all(k.split("|")[5] == "musique" for k in doc["groups"])


def test_a_gate_list_that_names_an_arm_the_grid_does_not_serve_is_refused_at_stage_0() -> None:
    """The stage 14 gate forms its groups from the windows on disk, so the
    stage 0 projection is the only guard against a roster arm without cells."""
    for arm in ("B7", "B8", "B11"):
        with pytest.raises(rc.PlanError, match=rf"CONFIRMATORY_ARMS names \['{arm}'\]: not an executable pressure baseline of session 's2b'"):
            rc.regime_gate_projection("s2b", loads="1.05", need="1.05:3", confirmatory_arms=f"B1 B3 B4 B6 {arm}")


def _gate_tree(root: Path, label_of: Any) -> Path:
    """One regime.json per registered window of the 82 s2b cells."""
    grid = _grid()
    for c in rc.enumerate_cells(grid):
        key = c.spec.to_row_key()
        f3._write_cell(root, key, c.baseline_id)
        for k in range(1, (c.windows if c.windows is not None else grid.replications) + 1):
            f3._write_window(root, key, c.dataset, k, label_of(c), "stationary")
    return root


def test_the_stage_14_gate_judges_load_105_alone_with_the_shipped_values(tmp_path: Path) -> None:
    v = f3._shipped("S2B.env", "REGIME_GATE_LOADS", "MIN_IN_REGIME_WINDOWS", "CONFIRMATORY_ARMS")
    kw = dict(loads=v["REGIME_GATE_LOADS"], confirmatory_arms=v["CONFIRMATORY_ARMS"])
    need = v["MIN_IN_REGIME_WINDOWS"]
    low_out = _gate_tree(tmp_path / "low_out", lambda c: "IN_REGIME" if c.spec.rate_frac == 1.05 else "OVERLOADED")
    res = f3.regime_gate.evaluate(low_out, need, **kw)
    modes = [g["mode"] for g in res["groups"].values()]
    assert res["ok"] is True and res["short"] == {} and res["short_descriptive"] == {}
    assert (len(modes), modes.count(rules.BLOCKING), modes.count(rules.DESCRIPTIVE)) == (32, 16, 16)
    b9_short = _gate_tree(tmp_path / "b9", lambda c: "OVERLOADED" if (c.baseline_id, c.spec.rate_frac) == ("B9", 1.05) else "IN_REGIME")
    res = f3.regime_gate.evaluate(b9_short, need, **kw)
    assert res["ok"] is True and res["short"] == {} and len(res["short_descriptive"]) == 4   # 2 engines x 2 rungs
    b3_short = _gate_tree(tmp_path / "b3", lambda c: "OVERLOADED" if (c.baseline_id, c.spec.rate_frac) == ("B3", 1.05) else "IN_REGIME")
    res = f3.regime_gate.evaluate(b3_short, need, **kw)
    assert res["ok"] is False and len(res["short"]) == 4


# --------------------------------------------------------------------------- #
# acceptance test 6, the comparison list: every listed id resolves on s2b's cells
# --------------------------------------------------------------------------- #


def _index() -> pd.DataFrame:
    rows = []
    for cell in rc.enumerate_cells(_grid()):
        s = cell.spec
        rows.append({"row_key": s.to_row_key(), "family": s.family, "arm": s.arm, "retriever": s.retriever,
                     "policy": s.policy, "topology": s.topology, "engine": s.engine, "model": s.model,
                     "baseline": org.BASELINE_OF_CELL.get((s.arm, s.retriever), ""), "dataset": cell.dataset,
                     "budget_r": s.budget_r, "rate_frac": s.rate_frac, "engine_variant": ""})
    return pd.DataFrame(rows)


def test_every_listed_comparison_resolves_on_the_cells_s2b_plans() -> None:
    """B-D6 (a): A's seven comparisons. A listed id that refuses fails stage 14
    of the landing, so each must find one pair per engine on MuSiQue."""
    listed = _stated("S2B.env")["CONTRASTS"].split()
    assert listed == ["1", "2", "3", "4", "5", "7", "8"]
    index = _index()
    for cid in (int(x) for x in listed):
        contrast = rca.CONTRAST_BY_ID[cid]
        assert "C" in contrast.groups, cid
        assert rca.classify_contrast(contrast) is None and contrast.unit == "per_query", cid
        pairs = rca.select_contrast_pairs(index, contrast)   # raises on a refusal
        assert sorted(rca.fp.parse_row_key(p.cell_row_key)["engine"] for p in pairs) == ["sglang", "vllm"], cid
        assert all(p.datasets == ("musique",) for p in pairs)
    # the comparison of B8 is not listed: B8 is not on s2b and the driver refuses the whole look on it
    with pytest.raises(rca.AnalysisError, match="needs F1 cells for both B8 and B6"):
        rca.select_contrast_pairs(index, rca.CONTRAST_BY_ID[9])
    metrics = _stated("S2B.env")["ANALYSIS_METRICS"].split()
    assert tuple(metrics) == families.DEFAULT_METRICS


def _arm_pairs_on_one_class(session: str) -> List[tuple]:
    grid = rc.get_session_grid(session)
    cls = {org.BASELINE_OF_CELL.get((c.spec.arm, c.spec.retriever), ""): rc.demand_seq_tokens(grid, c.spec)
           for c in rc.enumerate_cells(grid) if c.spec.family != "F1"}
    return sorted((cid, a, b) for cid, a, b in rca.registered_arm_pairs() if a in cls and b in cls and cls[a] == cls[b])


def test_the_arm_pairs_that_share_a_class_are_as() -> None:
    """One corpus class keeps B4 with B3 and B10 with B3 on one class, so the
    arm pairs under load form as on s2a (read from both grids, review L7)."""
    assert _arm_pairs_on_one_class("s2b") == _arm_pairs_on_one_class("s2a") == [
        (1, "B1", "B2"), (2, "B4", "B3"), (5, "B5", "B6"), (8, "B10", "B3"),
    ]


# --------------------------------------------------------------------------- #
# acceptance test 8: the other sessions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("session, digest", sorted({**e3.OTHER_GRID_SHA256, "cd-act1": e3.CD_ACT1_GRID_SHA256, "s2a": S2A_GRID_SHA256}.items()))
def test_the_other_sessions_are_registered_exactly_as_before(session: str, digest: str) -> None:
    """Regression pin (passes before and after): s2b is derived from s2a's
    grid and shares cd-act1's dataset, and neither may move by one field."""
    assert hashlib.sha256(repr(rc.get_session_grid(session)).encode()).hexdigest() == digest


@pytest.mark.parametrize("session, digest, steps", [("s2a", S2A_PLAN_SHA256, 226), ("cd-act1", CD_ACT1_PLAN_SHA256, 321)])
def test_the_other_s2_plans_are_unchanged_step_by_step(session: str, digest: str, steps: int, tmp_path: Path) -> None:
    """Regression pin (passes before and after): what every step of the
    fixture plan does, in order (the kind, the cell, the rate, the window, the
    cache size, the request cap), read without positions and paths."""
    built = trc._s2_plan(session, tmp_path)
    assert len(built["steps"]) == steps
    assert _plan_digest(built) == digest


# --------------------------------------------------------------------------- #
# acceptance test 9: the plan on the built manifest
# --------------------------------------------------------------------------- #


def test_the_plan_builds_on_the_built_manifest_and_reloads(tmp_path: Path) -> None:
    """The registered n (846) is the manifest's trial size and passes the
    coverage rule: a reference serves the whole trial."""
    built = _plan("s2b", tmp_path, query_manifests={"musique": str(REPO_ROOT / MANIFEST)})
    assert (built["counts"]["cells"], built["counts"]["windows"], built["counts"]["blocked"]) == (82, 178, 0)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(built), encoding="utf-8")
    assert len(rc.load_plan(path)["steps"]) == len(built["steps"]) == 226
    # a manifest of another dataset is refused
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(rc.PlanError):
        _plan("s2b", other, query_manifests={"qasper": str(REPO_ROOT / e3.MANIFEST)})


# --------------------------------------------------------------------------- #
# acceptance test 12: the two experiments are one design, at the plan
# --------------------------------------------------------------------------- #


def test_the_two_plans_hold_the_same_cells_windows_and_serving_groups(plan: Dict[str, Any], tmp_path: Path) -> None:
    plan_a = trc._s2_plan("s2a", tmp_path)

    def cells(p: Dict[str, Any]) -> List[tuple]:
        return sorted(
            (s["baseline"], s["cellspec"]["engine"], s["family"], s["cellspec"]["retriever"], str(s["cellspec"]["budget_r"]),
             str(s["cellspec"]["rate_frac"]), s["row_class"], s["num_queries"], s.get("block") or "", s.get("cell_windows") or 0)
            for s in trc._s2_rows(p)
        )

    assert cells(plan) == cells(plan_a)
    for key in ("cells", "windows", "relaunches", "dry_windows", "blocked", "by_family"):
        assert plan["counts"][key] == plan_a["counts"][key], key
    assert Counter(s["kind"] for s in plan["steps"]) == Counter(s["kind"] for s in plan_a["steps"])
    # the same serving groups in the same order, each class in A's place
    place = dict(zip(sorted(e8.CLASSES), CLASSES))

    def groups(p: Dict[str, Any], rename: Dict[int, int]) -> List[tuple]:
        return [
            (s["engine"], s.get("prefix_mode"), s.get("kv_dtype"), s.get("budget_r"),
             rename.get((s.get("demand_class") or {}).get("seq_tokens")))
            for s in p["steps"] if s["kind"] == "relaunch"
        ]

    assert groups(plan, {c: c for c in CLASSES}) == groups(plan_a, place)


# --------------------------------------------------------------------------- #
# acceptance test 10: the limits
# --------------------------------------------------------------------------- #


def test_the_run_limits_are_as_over_the_same_projection(plan: Dict[str, Any]) -> None:
    """The run projection is the S0 step table by engine and retriever, scaled
    by the pool size; it reads neither the model nor the class length, so s2b
    projects what s2a projects and takes A's run bound and budget stop (owner
    2026-11-22 on A: "Keep 55/50/88"). Limitation: the fixture's rates and a
    table measured on another model; the pod's stage 6 projects again from its
    measured floors."""
    keys = ("HOURS", "RUN_BUDGET_HOURS")
    v = f3._shipped("S2B.env", *keys)
    assert v == {"HOURS": "55", "RUN_BUDGET_HOURS": "50"} == f3._shipped("S2A.env", *keys)
    rp = plan["run_projection"]
    assert rp["unmeasured_steps"] == []
    assert rp["total_h"] == pytest.approx(42.55, abs=0.01)
    assert float(v["HOURS"]) - rp["total_h"] == pytest.approx(12.45, abs=0.01)
    assert float(v["RUN_BUDGET_HOURS"]) - rp["total_h"] == pytest.approx(7.45, abs=0.01)


# --------------------------------------------------------------------------- #
# review 2026-11-22 L7: the bounds, the pod and the timeouts as the master prints them
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def master_plan(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """The real master on the shipped S2B.env under --plan (it prints every
    stage's commands, runs nothing and creates nothing), with this test's own
    landing root and freeze file."""
    tmp = tmp_path_factory.mktemp("s2b_master")
    freeze = tmp / "freeze_resolutions.json"
    freeze.write_text(json.dumps({"INSTRUMENT_REVISIONS": {
        "served_model_qwen3_235b_fp8": {"model": MODEL_ID, "revision": MODEL_REVISION},
    }}), encoding="utf-8")
    # only the process basics: a caller's shell may carry another profile's exported keys
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "USER", "LOGNAME", "SHELL")}
    env.update({"CAGE_EXP_ROOT": str(tmp / "experiments"), "CAGE_EXP_DATE": "2026-11-22",
                "CAGE_MAC_PYTHON": sys.executable, "FREEZE_FILE": str(freeze)})
    proc = subprocess.run(["bash", str(MASTER), "S2B", "--plan"], capture_output=True, text=True, env=env,
                          timeout=600, cwd=str(REPO_ROOT))
    return {"proc": proc, "root": tmp / "experiments"}


def test_the_master_plans_every_stage_of_s2b_and_creates_nothing(master_plan: Dict[str, Any]) -> None:
    proc = master_plan["proc"]
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert not master_plan["root"].exists()
    out = proc.stdout
    assert out.count("[S2B] stage ") == 30 and "[S2B] stage analyze passed" in out   # 15 stages, each opened and passed
    assert ("regime-gate projection session=s2b loads='1.05' floor='1.05:3' roster=B1 B3 B4 B6 gated groups=32 "
            "blocking=16 descriptive=16 unresolved_load_cells=0") in out
    # every engine job loads the served model offline at the pinned revision
    engine_jobs = [line for line in out.splitlines() if "VLLM_START_TIMEOUT=" in line]
    assert engine_jobs and all(f"CAGE_MODEL_REVISION={MODEL_REVISION}" in line for line in engine_jobs)


def test_the_masters_stage_bounds_sum_to_5185_minutes_under_the_89h_seatbelt(master_plan: Dict[str, Any]) -> None:
    """The master's own sum (cage_experiment.sh stage_bound_minutes), read from
    its stage 0 print: S2A's 5,115 min plus ten for the setup bound (50 against
    40) and sixty for the scoring bound (180 against 120; owner 2026-11-22,
    "180 min and 89h"). 89h keeps 155 min over the sum. Stage 0 asks for a
    balance of 86.4 h x 4 GPUs x the live price + 50 USD: 1,878.22 USD at the
    5.29 USD per GPU-hour read on 2026-11-21 [D]."""
    out = master_plan["proc"].stdout
    assert ("seatbelt 89h = 5340 min; the stage bounds need 5185 min (readiness 15, setup 50, validate 235, "
            "calibrate 1200 with calibrate-rungs 540 per engine, plan, run 3300, seal, score 180, collect, "
            "pull margin 150)") in out
    assert "test 5340 -ge 5185" in out
    # the pod stage 0 plans: the disk (owner 2026-11-22, "500 GB on S2B"), the volume, the seatbelt and the
    # hours the stage bounds bill (5,185 min printed to one decimal)
    (provision,) = [line for line in out.splitlines() if "provision plan print" in line]
    argv = shlex.split(provision.partition(" : ")[2])   # the master prints the command shell-quoted
    flags = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    want = {"--gpu-id": "NVIDIA H200", "--gpu-count": "4", "--disk-gb": "500", "--volume-gb": "300",
            "--cloud-type": "SECURE", "--terminate-after": "89h", "--hours": "86.4"}
    assert {k: flags.get(k) for k in want} == want


def test_every_engine_job_of_the_plan_carries_the_1000_s_start_timeouts(master_plan: Dict[str, Any]) -> None:
    out = master_plan["proc"].stdout
    engine_jobs = [line for line in out.splitlines() if "VLLM_START_TIMEOUT=" in line]
    assert len(engine_jobs) >= 8
    assert all("VLLM_START_TIMEOUT=1000" in line and "SGLANG_START_TIMEOUT=1000" in line for line in engine_jobs)
    # the stage 4 start jobs keep their own deadline above one full start wait (the 80B's validate_vllm job
    # took 913 s with a 518 s start, so 395 s after it [V state.json of landing 2026-11-14]). 1,500 s for
    # SGLang is the plan print's figure: plan mode has no stage 0 record, and on a pod that recorded a helper
    # pool SGLang's job gets 3,000 s for its two starts (cage_experiment.sh, review 2026-10-21 H1)
    assert "pod_job.sh wait validate_vllm 1500" in out and "pod_job.sh wait validate_sglang 1500" in out


# --------------------------------------------------------------------------- #
# review 2026-11-22 M4 (owner: "Add the guard now"): SGLang's default cache precision
# --------------------------------------------------------------------------- #


def _sglang_default_start(stub: Path, log_root: Path, dtype_line: str, **env: str) -> subprocess.CompletedProcess:
    """The real launcher on the stub engine, with no SGLANG_KV_CACHE_DTYPE requested."""
    python_stub = kpt._engine_stub(stub, "sglang-python")
    return kpt._run_launcher(
        "manage_sglang_server.sh", stub, log_root, "start", "fake/test-model",
        SGLANG_START_TIMEOUT="10", CAGE_SGLANG_PYTHON=str(python_stub),
        CAGE_TEST_POOL_LINES=kpt.SGLANG_LOG + dtype_line + kpt.SGLANG_READY_LINE,
        CAGE_TEST_READY_WHEN_LOGGED="KV Cache is allocated", **env,
    )


@pytest.mark.parametrize("dtype, cap", [
    ("torch.float8_e4m3fn", ""), ("torch.float8_e5m2", ""),
    # second review L8: 28 of the plan's 34 relaunches carry a token cap (the stub's pool holds 430,913 tokens)
    ("torch.float8_e4m3fn", "430913"),
])
def test_an_sglang_default_start_fails_when_the_engine_chose_an_8_bit_cache(dtype: str, cap: str, stub_bin: Path, tmp_path: Path) -> None:
    """SGLang's budget is a token cap, so the realized-pool check passes
    whatever precision holds the cache; an 8-bit cache the plan did not ask
    for would label every SGLang row with the wrong precision and make B10
    equal B3 on that engine. The start log names the type (0.5.10.post1 printed
    "Using KV cache dtype: torch.bfloat16" on the 80B's default starts and
    torch.float8_e4m3fn on its 8-bit start [V the 2026-11-14 engine logs])."""
    env = {"CAGE_SGLANG_MAX_TOTAL_TOKENS": cap} if cap else {}
    bad = _sglang_default_start(stub_bin, tmp_path / "bad", f"[2026-11-22 10:00:00 TP0] Using KV cache dtype: {dtype}\n", **env)
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert ("--max-total-tokens 430913" in bad.stdout) == bool(cap)
    assert f"no SGLANG_KV_CACHE_DTYPE was requested but the start log says 'Using KV cache dtype: {dtype}'" in bad.stdout
    # the start is torn down, never left serving
    assert bad.stdout.index("no SGLANG_KV_CACHE_DTYPE was requested") < bad.stdout.index("Stopping SGLang server")


@pytest.mark.parametrize("dtype_line", [
    "[2026-11-14 18:00:46 TP0] Using KV cache dtype: torch.bfloat16\n",
    # a version that prints no such line is not judged (the 16-bit default is [A] there)
    "",
])
def test_an_sglang_default_start_passes_on_a_16_bit_cache_or_without_the_line(dtype_line: str, stub_bin: Path, tmp_path: Path) -> None:
    ok = _sglang_default_start(stub_bin, tmp_path / "ok", dtype_line)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "Server ready with model: fake/test-model" in ok.stdout and "was requested but" not in ok.stdout


def _guard(tmp_path: Path, log: bytes | None, requested: str = "") -> subprocess.CompletedProcess:
    """The launcher's own function, cut from the script and run alone under
    its shell options on one start log (None: the log file does not exist)."""
    text = (REPO_ROOT / "scripts" / "2_serving" / "manage_sglang_server.sh").read_text(encoding="utf-8")
    start = text.index("_sglang_kv_dtype_realized() {")
    func = text[start:text.index("\n}\n", start) + 3]
    path = tmp_path / "start.log"
    if log is not None:
        path.write_bytes(log)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if requested:
        env["SGLANG_KV_CACHE_DTYPE"] = requested
    script = "set -euo pipefail\nRED=''; NC=''\n" + func + '_sglang_kv_dtype_realized "$1"\n'
    return subprocess.run(["bash", "-c", script, "guard", str(path)], capture_output=True, text=True, env=env)


def _dtype_line(dtype: str) -> bytes:
    return f"[2026-11-22 10:00:00 TP0] Using KV cache dtype: {dtype}\n".encode()


#: the five 8-bit float types torch names (torch 2.13.0: dir(torch), names beginning float8)
FLOAT8_TYPES = ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz", "float8_e8m0fnu")


@pytest.mark.parametrize("name", FLOAT8_TYPES)
def test_the_guard_fails_a_default_start_on_every_8_bit_type_torch_names(name: str, tmp_path: Path) -> None:
    proc = _guard(tmp_path, kpt.SGLANG_LOG.encode() + _dtype_line(f"torch.{name}") + kpt.SGLANG_READY_LINE.encode())
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert f"the start log says 'Using KV cache dtype: torch.{name}'" in proc.stdout


@pytest.mark.parametrize("log", [
    _dtype_line("torch.bfloat16"), _dtype_line("torch.float16"), _dtype_line("torch.float32"),
    kpt.SGLANG_LOG.encode(),   # no such line
    None,                      # no log file
], ids=["bfloat16", "float16", "float32", "no line", "no file"])
def test_the_guard_passes_a_default_start_on_a_wider_type_or_without_the_line(log: bytes | None, tmp_path: Path) -> None:
    proc = _guard(tmp_path, log)
    assert (proc.returncode, proc.stdout) == (0, ""), proc.stdout + proc.stderr


def test_the_guard_reads_a_start_log_that_holds_a_nul_byte(tmp_path: Path) -> None:
    """Second review L2: grep treats a file with a NUL byte as binary and
    prints "Binary file ... matches" in place of the match (to stderr on GNU
    grep 3.5 and later [A], where the guard would then pass an 8-bit start).
    The guard reads the log as text."""
    log = b"[2026-11-22 09:59:59 TP0] progress \x00 bar\n" + _dtype_line("torch.float8_e4m3fn") * 2
    proc = _guard(tmp_path, log)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "the start log says 'Using KV cache dtype: torch.float8_e4m3fn'" in proc.stdout
    assert "Binary file" not in proc.stdout and proc.stdout.count("Using KV cache dtype") == 1   # one quoted line
    # the same log passes when it names a 16-bit type
    assert _guard(tmp_path, log.replace(b"float8_e4m3fn", b"bfloat16")).returncode == 0


def test_the_guard_leaves_a_requested_format_to_its_own_check(tmp_path: Path) -> None:
    """A start that asked for fp8_e4m3 must log that type (S0F-88 U8, unchanged);
    another requested format is reported unchecked, as before."""
    ok = _guard(tmp_path, _dtype_line("torch.float8_e4m3fn"), requested="fp8_e4m3")
    assert ok.returncode == 0 and "KV cache dtype realized: torch.float8_e4m3fn" in ok.stdout
    bad = _guard(tmp_path, _dtype_line("torch.bfloat16"), requested="fp8_e4m3")
    assert bad.returncode == 1 and "never said 'Using KV cache dtype: torch.float8_e4m3fn'" in bad.stdout
    other = _guard(tmp_path, _dtype_line("torch.float8_e5m2"), requested="fp8_e5m2")
    assert other.returncode == 0 and "not checked" in other.stdout


# --------------------------------------------------------------------------- #
# owner 2026-11-22 ("Search whole split"; "Yes, all three"): the retrieval arms
# search every passage of the split (plan 20.15, acceptance tests 1 to 7)
# --------------------------------------------------------------------------- #


def _retrieval_counts(built: Dict[str, Any]) -> Dict[bool, set]:
    """retrieval step? -> the counts its cells, references and dry windows carry (None: no key)."""
    out: Dict[bool, set] = {True: set(), False: set()}
    for step in built["steps"]:
        if step["kind"] in ("cell", "dry_window"):
            out[step["cellspec"]["retriever"] != "none"].add((step.get("env") or {}).get(DOCS_ENV))
    return out


def test_the_grid_registers_the_whole_split_count_on_s2b_alone() -> None:
    """Acceptance test 1 (decisions W-D1 (a) and W-D2 (a)): a field of the
    grid, kept out of the repr so the registration digests of the other
    sessions hold (the digest test above pins them)."""
    assert _grid().distractor_docs == WHOLE_SPLIT_DOCS
    others = {name: g.distractor_docs for name, g in rc.SESSION_GRIDS.items() if name != "s2b"}
    assert others == dict.fromkeys(others, 1000) and rc.DISTRACTOR_DOCS == 1000 and len(others) == 5
    assert "distractor_docs" not in repr(_grid())


def test_every_retrieval_step_of_the_s2b_plan_carries_the_whole_split_count(plan: Dict[str, Any]) -> None:
    """Acceptance tests 2 and 4: the pressure cells, the no-load references
    (B5 over BM25 among them) and every retrieval fill-ceiling window (four:
    the rows of B5 and B9 on each engine; B6 shares B5's class and window);
    no step that retrieves nothing carries a count; the header records it."""
    counts = _retrieval_counts(plan)
    assert counts == {True: {str(WHOLE_SPLIT_DOCS)}, False: {None}}
    retrieving = [s for s in plan["steps"] if s["kind"] == "cell" and s["cellspec"]["retriever"] != "none"]
    assert {s["baseline"] for s in retrieving} == {"B5", "B6", "B9"}
    assert {s["cellspec"]["retriever"] for s in retrieving} == {"dense", "rerank", "bm25"}
    assert plan["behavior_knobs"]["distractor_docs"] == WHOLE_SPLIT_DOCS


def test_the_ladder_cells_of_the_retrieval_classes_carry_the_same_count(tmp_path: Path) -> None:
    """Acceptance test 2, the ladders: calibrate-rungs builds each ladder window
    with the step builder the run uses, so the rate limit of a retrieval class
    is measured on the document set its cells search. This test calls that
    builder on the ladder's own selection; it does not run calibrate-rungs."""
    grid = _grid()
    doc = trc._floor_table_doc(model=grid.model, grid="reduced", r_values=(1.0, 0.5, 0.25), tp=int(grid.serving_tp))
    doc["generated_inputs"]["avg_seq_tokens"] = ANCHOR
    path = tmp_path / "floor.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    floor, pins = rc.load_floor_table(path), rc.resolve_retrieval_pins()
    seen: Dict[bool, set] = {True: set(), False: set()}
    for engine in ("vllm", "sglang"):
        selection = rc.ladder_cells(grid, engine, rc.budgeted_rungs(grid), {}, all_classes=True)
        for _role, cell in selection.roles():
            step = rc._cell_step(cell, grid, floor, rc.DEFAULT_RUNNER_CMD, 42, pins, slo_floors_env=None, ladder=True)
            seen[cell.spec.retriever != "none"].add(step["env"].get(DOCS_ENV))
    assert seen == {True: {str(WHOLE_SPLIT_DOCS)}, False: {None}}


@pytest.mark.parametrize("session", ["s2a", "cd-act1"])
def test_the_other_s2_plans_keep_the_registered_1000(session: str, tmp_path: Path) -> None:
    """Acceptance tests 3 and 4. The step digests above leave the environment
    out, so they would not see the count leak to another session."""
    built = trc._s2_plan(session, tmp_path)
    assert _retrieval_counts(built) == {True: {"1000"}, False: {None}}
    assert built["behavior_knobs"]["distractor_docs"] == 1000


def test_a_loaded_plan_must_carry_the_count_its_session_registers(plan: Dict[str, Any], tmp_path: Path) -> None:
    """Acceptance test 5: the count is read from the session's registration,
    never from the plan, so neither a stale cell nor an edited header passes."""

    def load(doc: Dict[str, Any], name: str) -> Dict[str, Any]:
        path = tmp_path / name
        path.write_text(json.dumps(doc), encoding="utf-8")
        return rc.load_plan(path)

    def first_retrieval_cell(doc: Dict[str, Any]) -> Dict[str, Any]:
        return next(s for s in doc["steps"] if s["kind"] == "cell" and s["cellspec"]["retriever"] != "none")

    assert len(load(plan, "ok.json")["steps"]) == 226
    stale = copy.deepcopy(plan)
    first_retrieval_cell(stale)["env"][DOCS_ENV] = "1000"
    with pytest.raises(rc.RunError, match=rf"{DOCS_ENV} is '1000', the registered pin is {WHOLE_SPLIT_DOCS}"):
        load(stale, "stale_cell.json")
    header = copy.deepcopy(plan)
    header["behavior_knobs"]["distractor_docs"] = 1000
    with pytest.raises(rc.RunError, match=rf"distractor_docs.*{WHOLE_SPLIT_DOCS}"):
        load(header, "edited_header.json")
    (tmp_path / "a").mkdir()
    plan_a = trc._s2_plan("s2a", tmp_path / "a")
    assert len(load(plan_a, "a_ok.json")["steps"]) == 226
    leak = copy.deepcopy(plan_a)
    first_retrieval_cell(leak)["env"][DOCS_ENV] = str(WHOLE_SPLIT_DOCS)
    with pytest.raises(rc.RunError, match=rf"{DOCS_ENV} is '{WHOLE_SPLIT_DOCS}', the registered pin is 1000"):
        load(leak, "leaked_cell.json")
    # third review L2: the header of another session is checked too, and a plan that names no
    # registered session is refused (the count cannot be looked up)
    leak_header = copy.deepcopy(plan_a)
    leak_header["behavior_knobs"]["distractor_docs"] = WHOLE_SPLIT_DOCS
    with pytest.raises(rc.RunError, match=r"distractor_docs is 21100; session 's2a' registers 1000"):
        load(leak_header, "leaked_header.json")
    for name, edit in (("no_session.json", lambda d: d.pop("session")), ("other_session.json", lambda d: d.update(session="s9"))):
        orphan = copy.deepcopy(plan)
        edit(orphan)
        with pytest.raises(rc.RunError, match="names no registered session"):
            load(orphan, name)


def test_a_loaded_plan_checks_the_count_on_its_fill_ceiling_windows_too(plan: Dict[str, Any], tmp_path: Path) -> None:
    """Third review L1: a fill-ceiling window copies its cell's environment and
    builds the same document set, so the loader re-checks its count as on a
    cell. Before, a window edited to another count loaded and would have
    built a second index (10,650 documents) for its 75 seconds."""

    def load(doc: Dict[str, Any], name: str) -> Dict[str, Any]:
        path = tmp_path / name
        path.write_text(json.dumps(doc), encoding="utf-8")
        return rc.load_plan(path)

    def dry(doc: Dict[str, Any], retrieving: bool) -> Dict[str, Any]:
        return next(s for s in doc["steps"] if s["kind"] == "dry_window" and (s["cellspec"]["retriever"] != "none") == retrieving)

    # the fixture plan holds four retrieval windows: the dense retrieval row (B5's; class 506, which B6
    # shares) and the compressed retrieval row (B9's), on each engine
    windows = [s for s in plan["steps"] if s["kind"] == "dry_window" and s["cellspec"]["retriever"] != "none"]
    assert sorted((s["engine"], s["row_key"].split("|")[0], s["cellspec"]["retriever"]) for s in windows) == [
        ("sglang", "retr-comp", "rerank"), ("sglang", "retr-fresh", "dense"),
        ("vllm", "retr-comp", "rerank"), ("vllm", "retr-fresh", "dense"),
    ]
    stale = copy.deepcopy(plan)
    dry(stale, True)["env"][DOCS_ENV] = "1000"
    with pytest.raises(rc.RunError, match=rf"dry window env {DOCS_ENV} is '1000', the registered pin is {WHOLE_SPLIT_DOCS}"):
        load(stale, "stale_window.json")
    missing = copy.deepcopy(plan)
    del dry(missing, True)["env"][DOCS_ENV]
    with pytest.raises(rc.RunError, match=rf"dry window env {DOCS_ENV} is None, the registered pin is {WHOLE_SPLIT_DOCS}"):
        load(missing, "window_without_count.json")
    extra = copy.deepcopy(plan)
    dry(extra, False)["env"][DOCS_ENV] = str(WHOLE_SPLIT_DOCS)
    with pytest.raises(rc.RunError, match=rf"dry window \(retriever 'none'\) carries {DOCS_ENV}"):
        load(extra, "window_with_a_count_it_does_not_use.json")
    (tmp_path / "a").mkdir()
    leak = copy.deepcopy(trc._s2_plan("s2a", tmp_path / "a"))
    dry(leak, True)["env"][DOCS_ENV] = str(WHOLE_SPLIT_DOCS)
    with pytest.raises(rc.RunError, match=rf"dry window env {DOCS_ENV} is '{WHOLE_SPLIT_DOCS}', the registered pin is 1000"):
        load(leak, "leaked_window.json")


#: run in a child process so the dataset library reads the offline switches at
#: its import: the document set the runner's own builders give for a manifest
#: and each count of extra passages
_CORPUS_SCRIPT = r"""
import json, sys
from pathlib import Path
repo, manifest = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / "scripts" / "3_run"))
import run_experiment as rx
from src.orchestration.ir import build_corpus_from_contexts, stable_text_id
loader = rx.get_loader("musique", split=rx.default_dataset_split("musique"), seed=42)
try:
    pool = loader.load(max_examples=None)
except Exception as exc:
    print(json.dumps({"unavailable": repr(exc)[:300]}))
    sys.exit(0)
split = {stable_text_id(str(c)) for ex in pool for c in (ex.context or []) if c}
union = rx.select_manifest_union(json.loads((repo / manifest).read_text(encoding="utf-8")), examples=pool)
base = {d.doc_id for d in build_corpus_from_contexts(union, dataset_name="musique")}
gold = [c for ex in union for c in (ex.context or []) if c]
gold += [g for ex in union for g in ((ex.metadata or {}).get("gold_paragraphs") or []) if g]
out = {"questions": len(pool), "split": len(split), "own": len(base), "counts": {}}
for n in (int(x) for x in sys.argv[3:]):
    ids = base | {stable_text_id(t) for t in rx.select_distractor_texts(pool, gold, n)}
    out["counts"][str(n)] = [len(ids), ids == split]
print(json.dumps(out))
"""


def test_the_registered_count_indexes_every_passage_of_the_split() -> None:
    """Acceptance test 6: the runner's own builders on the profile's manifest.
    The runner takes "the first n passages not already in the set" and reports
    no error when n is too small, so the boundary is pinned here: one passage
    less than the remainder leaves the set short of the split."""
    registered = _grid().distractor_docs
    proc = subprocess.run(
        [sys.executable, "-c", _CORPUS_SCRIPT, str(REPO_ROOT), MANIFEST, "1000", "11449", "11450", str(registered)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=600,
        env={**os.environ, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    doc = json.loads(proc.stdout.strip().splitlines()[-1])
    if "unavailable" in doc:
        pytest.skip(f"the MuSiQue split is not in this machine's dataset cache: {doc['unavailable']}")
    assert (doc["questions"], doc["split"], doc["own"]) == (2417, WHOLE_SPLIT_DOCS, 9650)
    assert doc["counts"] == {
        "1000": [10_650, False],      # the count every other session registers
        "11449": [21_099, False],     # one passage short
        "11450": [21_100, True],      # the exact remainder
        str(registered): [WHOLE_SPLIT_DOCS, True],
    }
    assert registered == doc["split"]   # W-D2 (a): the count is the number of passages in the split


@pytest.mark.parametrize("bad", [-1, True, 2.5, "1000", None])
def test_a_count_that_is_not_a_whole_number_of_passages_is_refused_at_registration(bad: Any) -> None:
    """Acceptance test 7."""
    with pytest.raises(rc.PlanError, match="distractor_docs"):
        dataclasses.replace(rc.get_session_grid("s2a"), distractor_docs=bad)
    # 0 adds no passage (the runner's rule) and is a legal registration
    assert dataclasses.replace(rc.get_session_grid("s2a"), distractor_docs=0).distractor_docs == 0
