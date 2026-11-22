"""S2 register item A-5 (owner 2026-11-21: "Yes, all four"): the s2a demand
classes are the Qasper means measured in Llama-3.3-70B tokens.

Spec and record: MyDocs/RunPod/S2A_A5_DEMAND_TABLES_2026-11-21.md.
- A5-1 (a): the three corpus arms (B3, B4, B10) share one class, 2,240, the
  count of B3's 845 rows; B4 and B10 measured 2,236 on their 793 rows and
  serve the same prompt.
- A5-2 (a): the class of B9 (640 tokens) takes an explicit request cap of
  4,096. The rule (ADR-0158) gives 8,192 for its largest request (2,339) and
  its quarter-budget cache holds 8,000 tokens at the profile's concurrency.
- The registered values are the result file's of 2026-11-21, read here when
  the S1 plots folder is on the machine (experiments/ is not in git).

Fixtures: tests/test_run_campaign.py (the S2 plan stack),
tests/test_s2_block_f_batch3.py (the shipped profile reader).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_run_campaign as trc  # noqa: E402
import test_s2_block_f_batch3 as f3  # noqa: E402

rc = trc.rc

LLAMA = "meta-llama/Llama-3.3-70B-Instruct"
LLAMA_REVISION = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
MANIFEST = "data/manifests/qasper_846x12_seed42_wr50_tok-Llama-3.3-70B-Instruct.json"
RESULT = (
    REPO_ROOT / "experiments" / "S1" / "2026-10-23_2350" / "plots" / "demand_tables_20261121"
    / "demand_tables_qasper_Llama-3.3-70B-Instruct_manifest.json"
)
#: arm -> mean served tokens (prompt plus answer), the result file's rounded means
ARM_TOKENS = {
    "gold-fresh": 4978, "gold-reuse": 4978,
    "corpus-fresh": 2240, "corpus-reuse": 2240, "corpus-comp": 2240,
    "retr-fresh": 995, "retr-reuse": 1008, "retr-store": 995,
    "retr-comp": 640, "retr-trunc": 377,
}
#: class -> its largest request (the largest over the class's arms)
CLASS_MAX = {4978: 21715, 2240: 2818, 1008: 3814, 995: 3993, 640: 2339, 461: 887, 377: 1839}
#: class -> request cap: the session cap on the anchor, the ADR-0158 rule
#: elsewhere, 4,096 by registration on B9's class (the rule gives 8,192)
CAPS = {4978: 32768, 2240: 8192, 1008: 8192, 995: 8192, 640: 4096, 461: 2048, 377: 4096}
B9_CLASS = 640


def _grid() -> Any:
    return rc.get_session_grid("s2a")


def _plan(session: str, tmp_path: Path) -> Dict[str, Any]:
    anchor = int(rc.get_session_grid(session).demand_seq_tokens[rc.DEMAND_ANCHOR_ARM])
    return trc._s2_plan(session, tmp_path, anchor=anchor)


@pytest.fixture(scope="module")
def plan(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    return _plan("s2a", tmp_path_factory.mktemp("s2a_a5"))


# --------------------------------------------------------------------------- #
# the registered tables
# --------------------------------------------------------------------------- #


def test_the_arms_register_the_measured_llama_counts() -> None:
    grid = _grid()
    assert dict(grid.demand_seq_tokens) == ARM_TOKENS
    # the grid requires the 700 rung's class; no s2a arm serves it
    assert dict(grid.corpus_trunc_demand_seq_tokens) == {700: 461}
    assert dict(grid.demand_class_max_served_tokens) == CLASS_MAX


def test_the_corpus_arms_share_one_class_and_the_session_serves_four() -> None:
    """A5-1 (a): B4 with B3 and B10 with B3 stay on one class, so their arm
    pairs under load still form; B5 and B6 share the dense retrieval class.
    Since 2026-11-21 B7 and its class (1008) are off the session: four classes
    (five with B7)."""
    grid = _grid()
    by_baseline = {c.baseline_id: rc.demand_seq_tokens(grid, c.spec) for c in rc.enumerate_cells(grid)}
    assert by_baseline == {
        "B1": 4978, "B2": 4978, "B3": 2240, "B4": 2240, "B10": 2240,
        "B5": 995, "B6": 995, "B9": B9_CLASS,
    }
    assert rc.session_demand_classes(grid) == [640, 995, 2240, 4978]


def test_the_source_names_the_tokenizer_the_manifest_and_the_result_folder(plan: Dict[str, Any]) -> None:
    source = _grid().demand_seq_tokens_source
    for part in (LLAMA, LLAMA_REVISION, Path(MANIFEST).name, "2026-10-23_2350", "demand_tables_20261121"):
        assert part in source, part
    assert source != rc.DEMAND_SEQ_TOKENS_SOURCE
    # the plan header carries what the budgets were sized on
    assert plan["demand_classes"]["source"] == source
    assert plan["demand_classes"]["anchor_seq_tokens"] == 4978
    assert plan["demand_classes"]["seq_tokens"] == dict(sorted(ARM_TOKENS.items()))


def test_the_registered_values_are_the_result_files() -> None:
    if not RESULT.is_file():
        pytest.skip(f"{RESULT.relative_to(REPO_ROOT)} is not on this machine (experiments/ is not in git)")
    doc = json.loads(RESULT.read_text(encoding="utf-8"))
    assert doc["target_tokenizer"] == {"id": LLAMA, "revision": LLAMA_REVISION}
    assert (doc["dataset"], doc["engine"], doc["manifest"]) == ("qasper", "vllm", MANIFEST)
    arms = doc["arms"]
    measured = {arm: rec["target_served_mean_int"] for arm, rec in arms.items()}
    # A5-1 (a): B4 and B10 measured 2,236 on 793 rows of the same prompt as
    # B3's 2,240 on 845 rows; the three register B3's count
    assert (measured["corpus-fresh"], measured["corpus-comp"], measured["corpus-reuse"]) == (2236, 2236, 2240)
    assert arms["corpus-fresh"]["target_prompt_mean"] == arms["corpus-comp"]["target_prompt_mean"]
    assert (arms["corpus-fresh"]["rows"], arms["corpus-reuse"]["rows"]) == (793, 845)
    grid = _grid()
    for arm, tokens in grid.demand_seq_tokens.items():
        if arm == "retr-store":
            want = measured["retr-fresh"]   # B8 is not on s2a; it takes retr-fresh's shape [D]
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


# --------------------------------------------------------------------------- #
# the request caps and the pools
# --------------------------------------------------------------------------- #


def test_the_request_caps(plan: Dict[str, Any]) -> None:
    grid = _grid()
    assert {c: rc.request_cap_tokens(grid, c) for c in CAPS} == CAPS
    assert dict(grid.demand_class_request_cap_tokens) == {B9_CLASS: 4096}
    shapes = plan["serving_shapes"]
    assert shapes["request_caps"] == {str(c): cap for c, cap in CAPS.items()}
    # the header names the cap the rule did not set, and its rule text says so
    assert shapes["request_cap_explicit"] == {str(B9_CLASS): 4096}
    assert shapes["request_cap_rule"].startswith(rc.REQUEST_CAP_RULE)
    assert "request_cap_explicit" in shapes["request_cap_rule"] and "ADR-0213" in shapes["request_cap_rule"]


def test_b9s_servers_and_cells_carry_the_registered_cap(plan: Dict[str, Any]) -> None:
    b9 = [s for s in plan["steps"] if s["kind"] == "relaunch"
          and (s.get("demand_class") or {}).get("seq_tokens") == B9_CLASS]
    assert sorted((s["engine"], s["budget_r"]) for s in b9) == [
        ("sglang", 0.25), ("sglang", 1.0), ("vllm", 0.25), ("vllm", 1.0),
    ]
    assert all(s["max_model_len"] == 4096 and s["env"][rc.MAX_MODEL_LEN_ENV] == "4096" for s in b9)
    cells = [s for s in trc._s2_rows(plan) if s["baseline"] == "B9"]
    caps = {(s["family"], rc._argv_flag_value(s["argv"], "--request-cap-tokens")) for s in cells}
    # its loaded cells at the class cap, its no-load reference at the session cap
    assert caps == {("F2", "4096"), ("F1", "32768")}
    # the fill-ceiling check before B9's loaded cells runs under the same cap (one per engine)
    dry = [s for s in plan["steps"] if s["kind"] == "dry_window" and s["demand_class"]["seq_tokens"] == B9_CLASS]
    assert sorted(s["engine"] for s in dry) == ["sglang", "vllm"]
    assert {rc._argv_flag_value(s["argv"], "--request-cap-tokens") for s in dry} == {"4096"}


def test_every_cache_at_the_profiles_concurrency_holds_one_capped_request() -> None:
    v = f3._shipped("S2A.env", "FLOOR_CONCURRENCY")
    grid = _grid()
    cells = rc.enumerate_cells(grid)
    records = rc._mac_relaunch_records(grid, cells, int(v["FLOOR_CONCURRENCY"]))
    assert {r["demand_class"]["seq_tokens"] for r in records} == {4978, 2240, 995, 640}   # 1008 left with B7
    for rec in records:
        pool = min(tokens for _label, tokens in rc._pool_tokens_of(rec["budget_plan"]))
        assert pool >= rec["max_model_len"], rec
    b9_quarter = [r for r in records if r["demand_class"]["seq_tokens"] == B9_CLASS and r["budget_r"] == 0.25]
    assert sorted(r["engine"] for r in b9_quarter) == ["sglang", "vllm"]
    for rec in b9_quarter:
        # 0.25 x 50 requests x 640 tokens
        assert [tokens for _label, tokens in rc._pool_tokens_of(rec["budget_plan"])] == [8000]
        assert rec["max_model_len"] == 4096
    assert rc.mac_pool_shortfalls(grid, cells, int(v["FLOOR_CONCURRENCY"])) == []


def test_under_the_rule_alone_b9s_quarter_cache_is_192_tokens_short() -> None:
    """Why the cap is registered: the same grid without it fails the stage 0
    pool rule on both engines."""
    grid = replace(_grid(), demand_class_request_cap_tokens={})
    assert rc.request_cap_tokens(grid, B9_CLASS) == 8192
    short = rc.mac_pool_shortfalls(grid, rc.enumerate_cells(grid), 50)
    assert sorted((s["engine"], s["seq_tokens"], s["budget_r"], s["max_model_len"], s["pool_tokens"], s["shortfall_tokens"])
                  for s in short) == [
        ("sglang", B9_CLASS, 0.25, 8192, 8000, 192), ("vllm", B9_CLASS, 0.25, 8192, 8000, 192),
    ]


@pytest.mark.parametrize("bad, why", [
    # under the class's largest request plus the 256 answer tokens a request
    # reserves (2,339 + 256 = 2,595): the engines admit prompt + max_tokens
    (2048, "below"),
    (3000, "power of two"),
    (16384, "above"),           # over the rule's cap (8,192)
])
def test_an_explicit_cap_outside_its_bounds_is_refused_at_registration(bad: int, why: str) -> None:
    """Review 2026-11-21 M1: refused when the grid is built (at import for a
    registered session), never as late as the plan."""
    with pytest.raises(rc.PlanError, match=why) as exc:
        replace(_grid(), demand_class_request_cap_tokens={B9_CLASS: bad})
    assert "demand_class_request_cap_tokens" in str(exc.value)
    assert str(bad) in str(exc.value) and "640" in str(exc.value)


def test_an_explicit_cap_equal_to_the_rules_is_accepted() -> None:
    grid = replace(_grid(), demand_class_request_cap_tokens={B9_CLASS: 8192})
    assert rc.request_cap_tokens(grid, B9_CLASS) == 8192
    # another class is untouched by B9's registration
    assert rc.request_cap_tokens(grid, 995) == 8192


def test_the_lower_bound_leaves_room_for_the_answer_tokens() -> None:
    """Review 2026-11-21 M2: a cap equal to the largest request cannot serve
    it (the engines admit prompt + max_tokens, and a QA request reserves
    WINDOW_MAX_TOKENS = 256). The smallest cap accepted is the largest request
    plus 256; in this variant of the grid the rule alone gives 4,096."""
    assert rc.WINDOW_MAX_TOKENS == 256
    grid = _grid()

    def variant(largest: int, cap: int):
        maxima = {**grid.demand_class_max_served_tokens, B9_CLASS: largest}
        return replace(grid, demand_class_max_served_tokens=maxima, demand_class_request_cap_tokens={B9_CLASS: cap})

    held = variant(1792, 2048)                      # 1,792 + 256 = 2,048 exactly
    assert rc.request_cap_tokens(held, B9_CLASS) == 2048
    assert rc.request_cap_tokens(replace(held, demand_class_request_cap_tokens={}), B9_CLASS) == 4096
    with pytest.raises(rc.PlanError, match="below"):
        variant(1793, 2048)                         # one token more does not fit
    with pytest.raises(rc.PlanError, match="below"):
        variant(2048, 2048)                         # the cap equal to the largest request


@pytest.mark.parametrize("bad, why", [
    ({4978: 4096}, "below the anchor"),             # the anchor serves the session cap
    ({702: 4096}, "not a demand class"),            # a key left behind by a re-measure
    ({"640": 4096}, "not a demand class"),          # a string key is never looked up
    ({640: None}, "integer"),
    ({640: 4096.0}, "integer"),
    ({640: "4096"}, "integer"),
    ({640: True}, "integer"),
    ({640: 4096, 461: 3000}, "power of two"),       # a bad cap on a class no cell serves
    ([640, 4096], "mapping"),
    (None, "mapping"),
])
def test_a_malformed_explicit_cap_table_is_refused_at_registration(bad: Any, why: str) -> None:
    """Review 2026-11-21 M1: before the check an entry on the anchor, on an
    unknown class or with a null value was accepted and never read, and the
    plan header printed a cap no server had."""
    with pytest.raises(rc.PlanError, match=why) as exc:
        replace(_grid(), demand_class_request_cap_tokens=bad)
    assert "demand_class_request_cap_tokens" in str(exc.value)


# --------------------------------------------------------------------------- #
# the ladders, the profile, the other sessions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_the_ladders_run_eight_triples_per_engine(engine: str) -> None:
    grid = _grid()
    assert f3._shipped("S2A.env", "RUNG_ALL_CLASSES") == {"RUNG_ALL_CLASSES": "1"}
    selection = rc.ladder_cells(grid, engine, rc.budgeted_rungs(grid), {}, all_classes=True)
    triples = sorted((role, rc.demand_seq_tokens(grid, c.spec), float(c.spec.budget_r)) for role, c in selection.roles())
    assert triples == sorted(
        (role, tokens, r)
        # ten triples with ("measured", 1008), the class of B7, before 2026-11-21
        for role, tokens in (("anchor", 4978), ("smallest", B9_CLASS), ("measured", 995), ("measured", 2240))
        for r in (1.0, 0.25)
    )


def test_the_shipped_profile_states_the_measured_anchor() -> None:
    v = f3._shipped("S2A.env", "FLOOR_AVG_SEQ_TOKENS", "FLOOR_CONCURRENCY", "GPU_ID")
    anchor = _grid().demand_seq_tokens[rc.DEMAND_ANCHOR_ARM]
    assert v["FLOOR_AVG_SEQ_TOKENS"] == str(anchor) == "4978"
    kw: Dict[str, Any] = dict(
        query_manifests={"qasper": str(REPO_ROOT / MANIFEST)}, charter_datasets=("qasper",),
        floor_concurrency=int(v["FLOOR_CONCURRENCY"]), gpu_id=v["GPU_ID"],
    )
    # a floor table sized on the old anchor is refused at stage 0
    old = rc.preflight_registration_check("s2a", floor_avg_seq_tokens="4779", **kw)
    assert any("4978" in p for p in old["problems"]), old["problems"]


@pytest.mark.parametrize("session, caps", [
    ("a", {4779: 32768, 2336: 8192, 1205: 4096, 1127: 8192, 1126: 8192, 702: 4096, 480: 4096, 348: 4096}),
    ("b", {4779: 32768, 2336: 8192, 1205: 4096, 1127: 8192, 1126: 8192, 702: 4096, 480: 4096, 348: 4096}),
    ("s1", {4779: 32768, 2336: 8192, 1205: 4096, 1127: 8192, 1126: 8192, 702: 4096, 480: 4096, 348: 4096}),
    ("cd-act1", {2694: 16384, 2387: 8192, 2386: 8192, 690: 2048, 517: 4096, 506: 4096, 320: 2048, 221: 2048}),
])
def test_the_other_sessions_keep_their_caps_and_register_none(session: str, caps: Dict[int, int]) -> None:
    """Read on 2026-11-21 before the change. The explicit cap is not part of
    the grid's repr (the registration pins of test_s2a_e3_registration.py read
    the repr), so its absence on the other sessions is pinned here."""
    grid = rc.get_session_grid(session)
    assert {c: rc.request_cap_tokens(grid, c) for c in caps} == caps
    assert set(caps) == set(grid.demand_seq_tokens.values()) | set(grid.corpus_trunc_demand_seq_tokens.values())
    assert dict(grid.demand_class_request_cap_tokens) == {}


def test_a_plan_without_an_explicit_cap_carries_no_such_header_key(tmp_path: Path) -> None:
    """The other sessions' plan headers stay byte-identical."""
    shapes = _plan("cd-act1", tmp_path)["serving_shapes"]
    assert "request_cap_explicit" not in shapes
    assert shapes["request_cap_rule"] == rc.REQUEST_CAP_RULE
    assert shapes["request_caps"]["2694"] == 16384
