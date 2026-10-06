"""S0F-21 (ADR-0141): the serving-config library marks the memory-utilization
default it sets, and the cluster manager treats a marked default as unset.

Cause [V, read 2026-10-06]: ``scripts/lib/_serving_config.sh:50`` exported
``VLLM_GPU_MEMORY_UTILIZATION=0.90`` whenever the variable was unset;
``scripts/3_run/cloud_run.sh:192`` sources the library before it calls
``run_baselines.sh`` (line 251), which sources it again at line 30; the
cluster manager's ``resolve_gpu_share`` (``manage_vllm_cluster.py:225-240``)
treats any non-empty value as the operator's and refuses a shared launch above
0.50. So the pilot's three-replica family could never start on one GPU: the
manager saw the library's 0.90 as an explicit request.

Fix (owner: "Follow recommendations", option A): the library exports
``CAGE_VLLM_MEM_UTIL_DEFAULTED=1`` exactly when it set the default itself and
never touches the marker on a later re-source; ``resolve_gpu_share`` treats
the value as unset when the marker is "1" and the value equals the library
default "0.90", on both the shared and the distinct path, and its ``source``
string names the rule. Any other explicit value is honored or refused as
before. The capture-before-sourcing alternative (the pd launcher's own
pattern, ``manage_vllm_pd.sh:107``) was rejected for the cluster path because
under ``cloud_run.sh`` the default is already exported before
``run_baselines.sh`` starts.

The library is sourced in a real bash with the two variables cleared; the
manager is imported from its file like tests/test_manage_vllm_cluster.py.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LIB = REPO_ROOT / "scripts" / "lib" / "_serving_config.sh"
CLUSTER_PY = REPO_ROOT / "scripts" / "2_serving" / "manage_vllm_cluster.py"
MARKER = "CAGE_VLLM_MEM_UTIL_DEFAULTED"
VALUE = "VLLM_GPU_MEMORY_UTILIZATION"


def _load_cluster():
    spec = importlib.util.spec_from_file_location("manage_vllm_cluster_s0f21", CLUSTER_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mc = _load_cluster()


def _clean_env(extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in (VALUE, MARKER)}
    env.update(extra)
    return env


def _source(extra: dict[str, str], *, times: int = 1) -> list[str]:
    """Source the library ``times`` times in one bash; return [value, marker]."""
    sources = "; ".join([f'source "{LIB}" >/dev/null'] * times)
    script = (
        f'{sources}; printf "%s|%s\\n" "${{{VALUE}:-unset}}" "${{{MARKER}:-unset}}"'
    )
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True,
        env=_clean_env(extra), timeout=30, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1].split("|")


# ---------------------------------------------------------------------------
# 1. the library: the marker follows the default it set, and only that
# ---------------------------------------------------------------------------


def test_lib_marks_the_default_it_sets() -> None:
    assert _source({}) == ["0.90", "1"]


def test_lib_does_not_mark_an_explicit_value() -> None:
    assert _source({VALUE: "0.45"}) == ["0.45", "unset"]
    # An operator's explicit 0.90 is indistinguishable by value, so the
    # marker is the only difference: absent here.
    assert _source({VALUE: "0.90"}) == ["0.90", "unset"]


def test_lib_sourced_twice_keeps_the_marker() -> None:
    # cloud_run.sh sources first, run_baselines.sh again: the second sourcing
    # sees the value set and must not clear the marker.
    assert _source({}, times=2) == ["0.90", "1"]


def test_lib_keeps_an_exported_marker_off_an_explicit_value() -> None:
    # A stale marker in the shell beside an operator value: the library
    # neither clears nor sets it (the manager decides by value equality).
    assert _source({VALUE: "0.45", MARKER: "1"}) == ["0.45", "1"]


# ---------------------------------------------------------------------------
# 2. the manager: a marked library default is unset
# ---------------------------------------------------------------------------


def test_manager_names_the_marker() -> None:
    assert mc.MEM_UTIL_DEFAULTED_ENV == MARKER


def test_shared_marked_default_resolves_to_0_45_and_names_the_rule() -> None:
    d = mc.resolve_gpu_share(
        replica_count=3, replica_gpus=(None, None, None),
        env={VALUE: "0.90", MARKER: "1"},
    )
    assert d.mode == "shared"
    assert d.mem_util == "0.45"
    assert d.source.startswith("default")
    assert "S0F-21" in d.source
    assert "S0F-21" in d.banner()


def test_distinct_marked_default_keeps_0_90_as_a_default() -> None:
    d = mc.resolve_gpu_share(
        replica_count=2, replica_gpus=("0", "1"), env={VALUE: "0.90", MARKER: "1"},
    )
    assert d.mode == "distinct"
    assert d.mem_util == "0.90"
    assert d.source.startswith("default")


def test_shared_explicit_value_beside_the_marker_is_honored() -> None:
    d = mc.resolve_gpu_share(
        replica_count=2, replica_gpus=(None, None), env={VALUE: "0.45", MARKER: "1"},
    )
    assert d.mem_util == "0.45" and d.source == "explicit"


def test_shared_0_90_without_the_marker_is_still_refused() -> None:
    with pytest.raises(mc.GpuShareError, match="vLLM startup check"):
        mc.resolve_gpu_share(
            replica_count=2, replica_gpus=(None, None), env={VALUE: "0.90"},
        )


def test_stale_marker_with_another_spelling_is_explicit() -> None:
    # "0.9" is not the library's "0.90": an operator typed it, so the shared
    # ceiling applies as before.
    with pytest.raises(mc.GpuShareError, match="vLLM startup check"):
        mc.resolve_gpu_share(
            replica_count=2, replica_gpus=(None, None), env={VALUE: "0.9", MARKER: "1"},
        )


def test_marker_other_than_1_is_ignored() -> None:
    with pytest.raises(mc.GpuShareError):
        mc.resolve_gpu_share(
            replica_count=2, replica_gpus=(None, None), env={VALUE: "0.90", MARKER: "yes"},
        )


# ---------------------------------------------------------------------------
# 3. end to end: the marker crosses from bash into the manager's environment
# ---------------------------------------------------------------------------


def test_marker_reaches_the_manager_through_the_environment(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import importlib.util, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('mc_probe', {str(CLUSTER_PY)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['mc_probe'] = m  # dataclasses resolve annotations through sys.modules\n"
        "spec.loader.exec_module(m)\n"
        "d = m.resolve_gpu_share(replica_count=3, replica_gpus=(None, None, None), env=os.environ)\n"
        "print('RESOLVED', d.mode, d.mem_util, d.source.split(' ')[0])\n",
        encoding="utf-8",
    )
    script = f'source "{LIB}" >/dev/null; source "{LIB}" >/dev/null; "{sys.executable}" "{probe}"'
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True,
        env=_clean_env({}), timeout=60, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "RESOLVED shared 0.45 default" in proc.stdout


def test_explicit_0_90_exported_after_sourcing_reads_as_the_default(tmp_path: Path) -> None:
    """The marker's known hole (review 2026-10-06, MEDIUM): in ONE shell, after
    the library defaulted, an operator's explicit `export ...=0.90` cannot be
    told from the default (same value, marker still set). The contract: the
    launch resolves to the shared default and the source string names the
    way out (unset the marker), rather than the pre-fix refusal."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import importlib.util, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('mc_probe', {str(CLUSTER_PY)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['mc_probe'] = m\n"
        "spec.loader.exec_module(m)\n"
        "d = m.resolve_gpu_share(replica_count=3, replica_gpus=(None, None, None), env=os.environ)\n"
        "print('RESOLVED', d.mode, d.mem_util, '|', d.source)\n",
        encoding="utf-8",
    )
    script = (
        f'source "{LIB}" >/dev/null; export {VALUE}=0.90; "{sys.executable}" "{probe}"; '
        f'unset {MARKER}; "{sys.executable}" "{probe}"'
    )
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True,
        env=_clean_env({}), timeout=60, cwd=str(REPO_ROOT),
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESOLVED")]
    assert len(lines) == 1, proc.stdout + proc.stderr  # the second probe refused
    assert lines[0].startswith("RESOLVED shared 0.45 |")
    assert f"unset {MARKER}" in lines[0] and "S0F-21" in lines[0]
    # With the marker unset the explicit 0.90 is the operator's again: refused.
    assert "REFUSING cluster launch" in proc.stderr or "vLLM startup check" in proc.stderr


# ---------------------------------------------------------------------------
# 4. static: the pilot path still sources the library on both entry points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel", ["scripts/3_run/cloud_run.sh", "scripts/3_run/run_baselines.sh"]
)
def test_pilot_entry_points_source_the_library(rel: str) -> None:
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    assert "_serving_config.sh" in text


def test_lib_region_and_manager_rule_carry_no_dashes() -> None:
    lib = LIB.read_text(encoding="utf-8")
    start = lib.index("S0F-21")
    region = lib[start:lib.index("export VLLM_GPU_MEMORY_UTILIZATION", start) + 40]
    em_dash, en_dash = chr(0x2014), chr(0x2013)
    assert em_dash not in region and en_dash not in region
    text = CLUSTER_PY.read_text(encoding="utf-8")
    region = text[text.index("MEM_UTIL_DEFAULTED_ENV"): text.index("def build_serve_args")]
    assert em_dash not in region and en_dash not in region
