"""Doctrine pins for the shell layer and the dependency manifest.

Two standards from CAGE_TECHNICAL_REVIEW_2026-08-04.md are enforced here so they
cannot silently regress:

1. _common.sh adoption (review §2/§5): every non-deprecated shell script under
   scripts/ sources scripts/lib/_common.sh (shared die/require_cmd/confirm --
   fail-loud error discipline), except a short, individually-justified exemption
   list. Library files under scripts/lib/ ARE the sourced layer, not consumers.

2. Pinning policy (review §5 S1/S17/S18; PUBLICATION.md D9 §9.13): no
   scientific-instrument line in requirements.txt may regress to a bare floor
   pin (">=" with no "==" and no upper bound) -- every instrument is either
   exact-pinned or tightly bounded, per the policy header in requirements.txt.

Pure static checks: no GPU, no server, no network.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Deliberate exemptions from the _common.sh sourcing doctrine. Each entry needs a
# reason strong enough to survive review -- "forgot" is not one.
SOURCING_EXEMPT: Dict[str, str] = {
    # Executed by GCP from instance METADATA (a standalone copy of the file's
    # content, run as root from /var/lib/google): no repo tree exists relative to
    # the script at run time, so sourcing would fail; and its contract forbids
    # exiting early inside the ~30s preemption budget, which die() would do.
    "scripts/5_observability/gcp_shutdown_hook.sh": "runs standalone from GCP metadata; must never exit early",
}

# Scientific instruments (review §5.1 tool table + statistical/analysis peers):
# packages whose version drift can change a reported number.
SCIENTIFIC_INSTRUMENTS = frozenset({
    "torch", "transformers", "datasets",
    "sentence-transformers", "bert-score", "rouge-score", "lettucedetect",
    "llmlingua", "ragas", "evaluate",
    "scipy", "numpy", "pandas", "matplotlib", "seaborn",
    "faiss-cpu", "ranx",
})


def _non_deprecated_scripts() -> List[Path]:
    out = [
        p for p in sorted(SCRIPTS_DIR.rglob("*.sh"))
        if "deprecated" not in p.parts and p.parent.name != "lib"
    ]
    assert out, "no shell scripts found under scripts/ -- wrong repo root?"
    return out


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def _sources_file(text: str, name: str) -> bool:
    """True only for an actual `source` / `.` STATEMENT naming the file.

    A bare substring check is satisfiable by a shellcheck directive or prose
    comment mentioning the file with the real source line deleted — the exact
    mirror-drift the doctrine exists to prevent (adversarial review
    2026-08-12 on the D1 launcher test). The path portion may contain spaces
    (e.g. `source "$(cd "$(dirname ...)" && pwd)/_common.sh"`), so match any
    non-comment run after the source keyword rather than a single \\S+ token."""
    return re.search(rf"^\s*(?:source|\.)\s+[^#\n]*{re.escape(name)}", text, re.M) is not None


# ---------------------------------------------------------------------------
# 1. _common.sh adoption
# ---------------------------------------------------------------------------

def test_every_nondeprecated_script_sources_common_sh() -> None:
    missing = []
    for script in _non_deprecated_scripts():
        rel = _rel(script)
        if rel in SOURCING_EXEMPT:
            continue
        if not _sources_file(script.read_text(encoding="utf-8"), "_common.sh"):
            missing.append(rel)
    assert not missing, (
        "scripts not sourcing scripts/lib/_common.sh (add the source line or a "
        f"justified SOURCING_EXEMPT entry): {missing}"
    )


def test_sourcing_exemptions_exist_and_are_documented() -> None:
    # An exemption for a deleted file is stale; an undocumented exemption is a hole.
    for rel, reason in SOURCING_EXEMPT.items():
        p = REPO_ROOT / rel
        assert p.is_file(), f"stale SOURCING_EXEMPT entry (file gone): {rel}"
        assert reason.strip(), f"SOURCING_EXEMPT entry without a reason: {rel}"
        text = p.read_text(encoding="utf-8")
        assert "EXEMPT" in text.upper(), (
            f"{rel} is exempt from _common.sh sourcing but carries no in-file "
            "comment explaining the exemption"
        )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
def test_all_nondeprecated_scripts_parse_with_bash_n() -> None:
    bad = []
    for script in _non_deprecated_scripts():
        proc = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            bad.append(f"{_rel(script)}: {proc.stderr.strip()}")
    assert not bad, "shell syntax errors:\n" + "\n".join(bad)


def test_common_sh_exists_and_defines_the_contract() -> None:
    common = SCRIPTS_DIR / "lib" / "_common.sh"
    assert common.is_file(), "scripts/lib/_common.sh is missing"
    text = common.read_text(encoding="utf-8")
    for fn in ("die()", "require_cmd()", "warn()", "confirm()"):
        assert fn in text, f"_common.sh no longer defines {fn}"


# ---------------------------------------------------------------------------
# 2. requirements.txt pinning policy
# ---------------------------------------------------------------------------

_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([<>=!~].*)?$")


def _requirement_lines() -> List[Tuple[str, str, str]]:
    """(package_name_lowercase, specifier, raw_line) for each requirement line."""
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    out: List[Tuple[str, str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "@" in line:   # skip comments + direct refs
            continue
        m = _REQ_LINE.match(line)
        if not m:
            continue
        name = m.group(1).lower()
        spec = (m.group(3) or "").replace(" ", "")
        out.append((name, spec, raw))
    assert out, "requirements.txt parsed to zero requirement lines"
    return out


def _is_bare_floor(spec: str) -> bool:
    """A bare floor pin constrains only from below: >= (or >) with no == / cap."""
    if "==" in spec:
        return False
    has_floor = ">=" in spec or bool(re.search(r"(?<!=)>(?!=)", spec))
    has_cap = "<" in spec or "~=" in spec
    return has_floor and not has_cap


# ---------------------------------------------------------------------------
# 3. Canonical interpreter (B1) + shipped-tree guard (B3) — code assertion
#    walkthrough 2026-08-07 (MyDocs/CODE_ASSERTION_2026-08.md).
# ---------------------------------------------------------------------------

# The ONE canonical CPython. Must equal CAGE_CANONICAL_PYTHON in _common.sh,
# the requirements.txt header, and the setup script's provisioning target.
CANONICAL_PYTHON = "3.13"


def test_canonical_python_single_source_of_truth() -> None:
    """B1: the interpreter is declared in exactly one place and echoed
    consistently — requirements header, _common.sh constant, and the GPU
    bootstrap must all agree, and the bootstrap must never use bare python3."""
    common = (SCRIPTS_DIR / "lib" / "_common.sh").read_text(encoding="utf-8")
    m = re.search(r'CAGE_CANONICAL_PYTHON="(\d+\.\d+)"', common)
    assert m, "_common.sh no longer defines CAGE_CANONICAL_PYTHON"
    assert m.group(1) == CANONICAL_PYTHON, (
        f"_common.sh says {m.group(1)}, test constant says {CANONICAL_PYTHON} "
        f"— update BOTH deliberately or neither"
    )

    req_header = "\n".join(
        (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()[:5]
    )
    assert f"Python {CANONICAL_PYTHON} required" in req_header, (
        "requirements.txt header must declare the canonical interpreter "
        f"(expected 'Python {CANONICAL_PYTHON} required')"
    )

    setup = (SCRIPTS_DIR / "1_setup" / "setup_gpu_cloud.sh").read_text(encoding="utf-8")
    assert 'PYBIN="python${CAGE_CANONICAL_PYTHON}"' in setup, (
        "setup_gpu_cloud.sh must derive its interpreter from CAGE_CANONICAL_PYTHON"
    )
    assert re.search(r"^\s*python3 -m venv", setup, re.M) is None, (
        "setup_gpu_cloud.sh creates a venv with bare `python3` — finding B1: "
        "the venv must be created with the canonical interpreter, fail-closed"
    )


def test_engine_launchers_source_uniform_serving_config() -> None:
    """D1 (walkthrough Topic 4, 2026-08-12): scripts/lib/_serving_config.sh is
    the serving-uniformity source of truth ALL engine launchers must source —
    an engine launched outside the uniform regime silently breaks §6.5
    cross-mechanism fairness. Pins every manage_*_server.sh (vllm, sglang,
    lmdeploy today; any future engine automatically) to source it."""
    launchers = sorted((SCRIPTS_DIR / "2_serving").glob("manage_*_server.sh"))
    assert len(launchers) >= 3, (
        f"expected the three engine launchers (vllm/sglang/lmdeploy), found: "
        f"{[p.name for p in launchers]}"
    )
    missing = [
        _rel(p) for p in launchers
        if not _sources_file(p.read_text(encoding="utf-8"), "_serving_config.sh")
    ]
    assert not missing, (
        "engine launchers not sourcing scripts/lib/_serving_config.sh "
        f"(uniform-serving doctrine, D1): {missing}"
    )


def _git_ls_files() -> set:
    """Tracked paths (index-inclusive), or skip when git/its metadata is absent
    — a tarball deploy on the VM has no .git and BUILD_INFO is the provenance."""
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout (tarball deploy)")
    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--",
         "scripts", "src", "configs", "requirements.txt", "pytest.ini"],
        capture_output=True, text=True, check=True,
    ).stdout
    return set(out.splitlines())


def test_deployable_code_is_tracked_by_git() -> None:
    """B3 (the A1 vaccine): worktree presence is NOT shipped-ness. The deploy
    tarball is `git archive HEAD`, so every deployable file must be TRACKED —
    an is_file() guard passed for three days while scripts/lib/_common.sh was
    silently absent from every tarball (gitignore `lib/` shadowing)."""
    tracked = _git_ls_files()
    wanted = [
        p for p in sorted((REPO_ROOT / "scripts").rglob("*"))
        if p.suffix in (".sh", ".py") and "__pycache__" not in p.parts
    ]
    wanted += [
        p for p in sorted((REPO_ROOT / "src").rglob("*.py"))
        if "__pycache__" not in p.parts
    ]
    wanted += [
        p for p in sorted((REPO_ROOT / "configs").rglob("*"))
        if p.is_file() and p.name != ".DS_Store"
    ]
    wanted += [REPO_ROOT / "requirements.txt", REPO_ROOT / "pytest.ini"]
    missing = [
        str(p.relative_to(REPO_ROOT))
        for p in wanted
        if str(p.relative_to(REPO_ROOT)) not in tracked
    ]
    assert not missing, (
        "on disk but NOT tracked by git — these will silently NOT ship in the "
        f"deploy tarball (git archive HEAD): {missing}"
    )


def test_no_scientific_instrument_is_bare_floor_pinned() -> None:
    offenders = []
    for name, spec, raw in _requirement_lines():
        if name in SCIENTIFIC_INSTRUMENTS and (_is_bare_floor(spec) or not spec):
            offenders.append(raw.strip())
    assert not offenders, (
        "scientific instruments must be exact-pinned or tightly bounded "
        "(requirements.txt policy header; review S1/S17): " + ", ".join(offenders)
    )


def test_statistics_and_retrieval_eval_instruments_are_declared() -> None:
    names = {name for name, _, _ in _requirement_lines()}
    # scipy computes every confirmatory p-value (review S10/S18); ranx is the
    # charter D8 §8.2 Layer-0 retrieval-eval tool (review §5.1: was undeclared).
    assert "scipy" in names, "scipy missing from requirements.txt (review S10/S18)"
    assert "ranx" in names, "ranx missing from requirements.txt (charter D8 §8.2)"


def test_exact_pins_match_environment_when_installed() -> None:
    """An exact pin must record the tested venv's reality, not an aspiration.

    Only enforced for packages actually installed in the running environment --
    GPU-host-only instruments are absent locally and are covered by tier-2 bounds.
    """
    from importlib import metadata

    mismatches = []
    for name, spec, _ in _requirement_lines():
        if name not in SCIENTIFIC_INSTRUMENTS or "==" not in spec:
            continue
        pinned = spec.split("==", 1)[1].split(",", 1)[0]
        installed: Optional[str]
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue  # not installed here (e.g. GPU-only stack) -- nothing to compare
        if installed != pinned:
            mismatches.append(f"{name}: pinned=={pinned} but installed {installed}")
    assert not mismatches, (
        "exact pins diverge from the installed environment (re-test then bump "
        "deliberately): " + "; ".join(mismatches)
    )
