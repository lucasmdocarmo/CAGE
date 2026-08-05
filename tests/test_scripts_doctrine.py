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


# ---------------------------------------------------------------------------
# 1. _common.sh adoption
# ---------------------------------------------------------------------------

def test_every_nondeprecated_script_sources_common_sh() -> None:
    missing = []
    for script in _non_deprecated_scripts():
        rel = _rel(script)
        if rel in SOURCING_EXEMPT:
            continue
        if "_common.sh" not in script.read_text(encoding="utf-8"):
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
