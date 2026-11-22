"""S2 block F batch 4, row F-5 (2026-11-03): the ladders' (engine, class, rung) triples on the Mac.

``ladder_cells`` is the one selection rule the pod's ladder (``calibrate_rungs``)
and the Mac verb ``run_campaign.py ladder-cells`` share: the anchor, the
smallest class, every class between them (under --all-classes) and the RULER
instrument, each at the rungs it carries executable pressure cells at. The
verb prints one line per triple and a count line per engine, before any pod
bills (stage 5 of the master runs it, in plan mode too).
Spec: MyDocs/RunPod/S2_BLOCK_F_BATCH2-3_SPEC_2026-11-02.md section 4; the
counts are finding 4's table.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RC_PATH = REPO_ROOT / "scripts" / "3_run" / "run_campaign.py"
MUSIQUE_80B = "data/manifests/musique_2000x3_seed42_wr50_tok-Qwen3-Next-80B-A3B-Instruct.json"


def _load_rc():
    found = sys.modules.get("run_campaign")
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location("run_campaign", RC_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rc = _load_rc()

TRIPLE_LINE = re.compile(r"^(vllm|sglang) (anchor|smallest|measured|ruler) \S+ B\d+ \S+ \d+ [0-9.]+$")
COUNT_LINE = re.compile(r"^engine=(vllm|sglang) triples=(\d+) classes=(\d+)$")


def _select(session: str, engine: str, *, manifests: Optional[Dict[str, dict]] = None,
            all_classes: bool = True, rungs=None):
    grid = rc.get_session_grid(session)
    rung_list = rc._resolve_rung_list(grid, rungs)
    return grid, rc.ladder_cells(grid, engine, rung_list, manifests or {},
                                 all_classes=all_classes, anchor_only=False, ceilings_only=False)


def _by_class(grid, cells) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for c in cells:
        out.setdefault(rc.demand_seq_tokens(grid, c.spec), []).append(float(c.spec.budget_r))
    return {seq: sorted(rs) for seq, rs in out.items()}


def _verb(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(RC_PATH), "ladder-cells", *args], capture_output=True, text=True,
                          cwd=str(REPO_ROOT), env={**os.environ, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})


# --------------------------------------------------------------------------- #
# the selection, pinned to finding 4's table
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_s2a_ladders_eight_triples_over_four_classes_per_engine_with_the_qasper_manifest(engine: str) -> None:
    # owner 2026-11-19 (experiment A as ruled, variant E3): nine arms on the rungs 1.0
    # and 0.25, no B11 and no B12, so five classes at two rungs each. Before that day
    # the grid laddered 15 triples over seven classes (three rungs, 348 and 480 at 0.25).
    # The class counts are the Llama ones since 2026-11-21 (register A-5, ADR-0213;
    # 4779, 2336, 1127, 1126 and 702 before): the same five classes, re-measured.
    # Owner 2026-11-21: B7 left the session, and its class (1008) with it: four classes
    # at two rungs, eight triples per engine (ten with B7).
    grid, sel = _select("s2a", engine, manifests={"qasper": {}})
    by_class = _by_class(grid, sel.all_cells)
    assert len(sel.all_cells) == 8
    assert by_class == {seq: [0.25, 1.0] for seq in (4978, 2240, 995, 640)}
    assert [rc.demand_seq_tokens(grid, c.spec) for c in sel.anchor_cells] == [4978] * 2
    assert [rc.demand_seq_tokens(grid, c.spec) for c in sel.small_cells] == [640] * 2
    assert sorted(sel.measured_cells) == [995, 2240]
    assert sel.ruler_cells == []
    # the cells are the ladder's: anchor gold-fresh on this engine, every cell executable
    assert all(c.spec.engine == engine and c.blocked_on is None for c in sel.all_cells)
    assert {c.spec.arm for c in sel.anchor_cells} == {"gold-fresh"}


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_s2a_without_the_qasper_manifest_selects_the_same_cells(engine: str) -> None:
    # the session carries no corpus-trunc class since 2026-11-19, so the manifest no
    # longer changes the selection (it dropped the 480-token class before, 14 triples)
    grid, sel = _select("s2a", engine)
    by_class = _by_class(grid, sel.all_cells)
    assert len(sel.all_cells) == 8
    assert sorted(by_class) == [640, 995, 2240, 4978]   # 702, 1126, 1127, 2336, 4779 before A-5; 1008 left with B7


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_s1_without_the_qasper_manifest_drops_the_corpus_trunc_classes(engine: str) -> None:
    # review 2026-11-20 M3: measured_pressure_classes counts a corpus-trunc class only when
    # its dataset has a registered manifest (the plan blocks B12 on a dataset without one).
    # The s2a tests above pinned that rule until 2026-11-19 and cannot see it since (no
    # corpus-trunc class there); s1 still carries B12 on Qasper at the rungs 1400 and 700
    grid, bare = _select("s1", engine)
    _, with_manifest = _select("s1", engine, manifests={"qasper": {}})
    assert (len(bare.all_cells), len(with_manifest.all_cells)) == (14, 18)
    trunc = set(grid.corpus_trunc_demand_seq_tokens.values())
    assert trunc == {480, 1205}
    assert set(_by_class(grid, with_manifest.all_cells)) - set(_by_class(grid, bare.all_cells)) == trunc
    assert _by_class(grid, with_manifest.all_cells)[480] == _by_class(grid, with_manifest.all_cells)[1205] == [0.25, 0.5]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_cd_act1_ladders_fifteen_triples_over_five_classes_per_engine(engine: str) -> None:
    grid, sel = _select("cd-act1", engine, manifests={"musique": {}})
    by_class = _by_class(grid, sel.all_cells)
    assert len(sel.all_cells) == 15
    assert by_class == {seq: [0.25, 0.5, 1.0] for seq in (2694, 2387, 2386, 517, 506)}
    assert sorted(sel.measured_cells) == [517, 2386, 2387]
    assert [rc.demand_seq_tokens(grid, c.spec) for c in sel.small_cells] == [506] * 3


def test_the_pairs_of_finding_4_eight_on_s2a_and_ten_on_cd_act1() -> None:
    pairs: Dict[str, Set[Tuple[str, int]]] = {}
    for session, name in (("s2a", "qasper"), ("cd-act1", "musique")):
        for engine in ("vllm", "sglang"):
            grid, sel = _select(session, engine, manifests={name: {}})
            pairs.setdefault(session, set()).update((engine, rc.demand_seq_tokens(grid, c.spec)) for c in sel.all_cells)
    # s2a: four classes per engine since 2026-11-21 (five with B7, 10 pairs; seven before 2026-11-19, 14 pairs)
    assert (len(pairs["s2a"]), len(pairs["cd-act1"])) == (8, 10)


def test_s1_without_the_all_classes_flag_ladders_the_two_anchors_six_per_engine() -> None:
    total = 0
    for engine in ("vllm", "sglang"):
        grid, sel = _select("s1", engine, all_classes=False)
        assert sel.measured_cells == {} and sel.ruler_cells == []
        assert len(sel.anchor_cells) == 3 and len(sel.small_cells) == 3
        total += len(sel.all_cells)
    assert total == 12  # the S1 ladder logs' count


def test_the_rung_list_keeps_the_ladders_refusal_of_an_unregistered_rung() -> None:
    grid = rc.get_session_grid("s1")
    assert rc._resolve_rung_list(grid, None) == rc.budgeted_rungs(grid)
    assert rc._resolve_rung_list(grid, (0.5, 1.25)) == (1.25, 0.5)  # s1 registers 1.25, 0.5, 0.25
    with pytest.raises(rc.PlanError, match=r"--rungs 0\.3 is not a budgeted rung of session 's1'"):
        rc._resolve_rung_list(grid, (0.3,))


def test_ceilings_only_with_all_classes_is_refused_by_the_selection() -> None:
    grid = rc.get_session_grid("s1")
    with pytest.raises(rc.PlanError, match="--ceilings-only ladders the interpolated classes alone"):
        rc.ladder_cells(grid, "vllm", rc.budgeted_rungs(grid), {}, all_classes=True, anchor_only=False, ceilings_only=True)


# --------------------------------------------------------------------------- #
# the verb
# --------------------------------------------------------------------------- #


def test_the_verb_prints_one_line_per_triple_and_a_count_line_s1() -> None:
    proc = _verb("--session", "s1", "--engine", "vllm")
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 7
    assert all(TRIPLE_LINE.match(line) for line in lines[:-1]), lines
    assert lines[-1] == "engine=vllm triples=6 classes=2"
    assert lines[0].startswith("vllm anchor gold-fresh B1 single 4779 1")
    assert any(line.startswith("vllm smallest ") for line in lines)


def test_the_verb_on_s2a_without_the_manifest_reads_eight_with_all_classes() -> None:
    proc = _verb("--session", "s2a", "--engine", "sglang", "--all-classes")
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[-1] == "engine=sglang triples=8 classes=4"   # 10 and 5 with B7; 14 and 6 before 2026-11-19
    roles = [line.split()[1] for line in lines[:-1]]
    assert roles.count("anchor") == 2 and roles.count("smallest") == 2 and roles.count("measured") == 4


@pytest.mark.skipif(not (REPO_ROOT / MUSIQUE_80B).exists(), reason="the 80B MuSiQue manifest is not on this checkout")
def test_the_verb_on_cd_act1_with_the_musique_manifest_reads_fifteen() -> None:
    proc = _verb("--session", "cd-act1", "--engine", "vllm", "--all-classes", "--query-manifest", f"musique={MUSIQUE_80B}",
                 "--rungs", "1,0.5,0.25")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "engine=vllm triples=15 classes=5"


def test_the_verb_refuses_an_engine_the_session_does_not_serve_and_a_bad_rung() -> None:
    proc = _verb("--session", "s1", "--engine", "hf")
    assert proc.returncode == 2 and "is not a server engine of session 's1'" in proc.stderr
    proc = _verb("--session", "s1", "--engine", "vllm", "--rungs", "0.3")
    assert proc.returncode == 2 and "--rungs 0.3 is not a budgeted rung of session 's1'" in proc.stderr


def test_the_print_order_is_the_ladders_run_order_anchor_ruler_smallest_measured() -> None:
    # review 2026-11-03 LOW 2: the ladder runs the anchor, then the RULER instrument, then the
    # smallest class, then each measured class (run_campaign.py, the four _ladder calls at the end
    # of calibrate_rungs); the verb prints in that order so the owner reads the pod's sequence
    rc = _load_rc()
    grid = rc.get_session_grid("a")
    sel = rc.ladder_cells(grid, "vllm", rc._resolve_rung_list(grid, None), {}, all_classes=True)
    assert sel.ruler_cells, "session a carries the RULER instrument"
    rank = {"anchor": 0, rc.RULER_ROLE: 1, "smallest": 2, "measured": 3}
    ranks = [rank[role] for role, _ in sel.roles()]
    assert ranks == sorted(ranks) and set(ranks) == {0, 1, 2, 3}, ranks
    assert len(sel.roles()) == len(sel.all_cells)
