"""Header contract for scripts/ — Order / Objective / Cloud on every script.

Owner directive (2026-08-21): "all scripts must have their order, objective,
and most importantly for which cloud they applied, gcp or runpod". Enforced
here so a NEW script cannot land without declaring itself:

1. Every non-deprecated ``scripts/**/*.{sh,py}`` carries, near the top (first
   ``HEADER_WINDOW`` lines — the ``#``-comment block right after the shebang
   for .sh, the first lines of the module docstring for .py), three labeled
   lines::

       Order:     <where it sits in the lifecycle>
       Objective: <one line, what it does>
       Cloud:     gcp | runpod | both | local

2. The ``Cloud:`` value is one of exactly {gcp, runpod, both, local}
   (semantics: gcp/runpod = provider-only; both = used in cloud runs on
   either provider; local = operator machine / analysis only).

3. Provider-ONLY dirs are self-consistent: everything under ``scripts/gcp/``
   declares ``Cloud: gcp``; everything under ``scripts/runpod/`` declares
   ``Cloud: runpod``.

4. The ``scripts/README.md`` master catalog lists every script with the SAME
   Cloud value as its header (the table is generated from the headers; this
   pin keeps them from drifting apart).

The script list is globbed LIVE — no hardcoded file list — so an unstamped
new script fails the suite. Pure static checks: no GPU, no network.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
README = SCRIPTS_DIR / "README.md"

CLOUD_VALUES = frozenset({"gcp", "runpod", "both", "local"})

# The header must sit at the TOP of the file (shebang + comment block / module
# docstring), not buried mid-file.
HEADER_WINDOW = 30

# A label line is either a '# '-comment line (.sh) or a docstring-body line
# (.py) — where the first body line may share the line with the opening quotes
# (PEP-257 one-line-summary style: '"""Order: ...').
_PREFIX = r"(?:#\s*|\"\"\"|''')?"
ORDER_RE = re.compile(rf"^{_PREFIX}Order:\s+\S", re.MULTILINE)
OBJECTIVE_RE = re.compile(rf"^{_PREFIX}Objective:\s+\S", re.MULTILINE)
CLOUD_RE = re.compile(rf"^{_PREFIX}Cloud:\s+(\S+)\s*$", re.MULTILINE)


def _non_deprecated_scripts() -> list[Path]:
    out = sorted(
        p
        for p in SCRIPTS_DIR.rglob("*")
        if p.suffix in {".sh", ".py"}
        and p.is_file()
        and "deprecated" not in p.parts
        and "__pycache__" not in p.parts
    )
    assert out, "no scripts found under scripts/ — wrong repo root?"
    return out


def _rel(p: Path) -> str:
    return str(p.relative_to(SCRIPTS_DIR))


def _head(p: Path) -> str:
    lines = p.read_text(encoding="utf-8").splitlines()[:HEADER_WINDOW]
    return "\n".join(lines) + "\n"


def _cloud_of(p: Path) -> str:
    m = CLOUD_RE.search(_head(p))
    assert m, f"{_rel(p)}: no 'Cloud:' line in the first {HEADER_WINDOW} lines"
    return m.group(1)


_SCRIPTS = _non_deprecated_scripts()


@pytest.mark.parametrize("script", _SCRIPTS, ids=[_rel(p) for p in _SCRIPTS])
def test_script_declares_order_objective_cloud(script: Path) -> None:
    """Every script carries the three labeled header lines, near the top."""
    head = _head(script)
    assert ORDER_RE.search(head), (
        f"{_rel(script)}: missing 'Order:' header line in the first "
        f"{HEADER_WINDOW} lines (scripts/README.md documents the contract)"
    )
    assert OBJECTIVE_RE.search(head), (
        f"{_rel(script)}: missing 'Objective:' header line in the first "
        f"{HEADER_WINDOW} lines"
    )
    m = CLOUD_RE.search(head)
    assert m, (
        f"{_rel(script)}: missing 'Cloud:' header line in the first "
        f"{HEADER_WINDOW} lines"
    )
    assert m.group(1) in CLOUD_VALUES, (
        f"{_rel(script)}: Cloud value {m.group(1)!r} not in "
        f"{sorted(CLOUD_VALUES)}"
    )


def test_gcp_dir_scripts_declare_cloud_gcp() -> None:
    """scripts/gcp/ is provider-ONLY: every script there declares Cloud: gcp."""
    scripts = [p for p in _SCRIPTS if p.parent.name == "gcp"]
    assert scripts, "scripts/gcp/ is empty — provider split regressed?"
    offenders = {_rel(p): _cloud_of(p) for p in scripts if _cloud_of(p) != "gcp"}
    assert not offenders, f"scripts/gcp/ scripts must declare 'Cloud:     gcp': {offenders}"


def test_runpod_dir_scripts_declare_cloud_runpod() -> None:
    """scripts/runpod/ is provider-ONLY: every script there declares Cloud: runpod."""
    scripts = [p for p in _SCRIPTS if p.parent.name == "runpod"]
    assert scripts, "scripts/runpod/ is empty — provider split regressed?"
    offenders = {_rel(p): _cloud_of(p) for p in scripts if _cloud_of(p) != "runpod"}
    assert not offenders, f"scripts/runpod/ scripts must declare 'Cloud:     runpod': {offenders}"


def _readme_catalog_clouds() -> dict[str, str]:
    """Parse the README master catalog rows: | `<rel>` | order | objective | cloud |."""
    rows: dict[str, str] = {}
    row_re = re.compile(r"^\|\s*`([^`]+)`\s*\|.*\|\s*(\S+)\s*\|\s*$")
    for line in README.read_text(encoding="utf-8").splitlines():
        m = row_re.match(line)
        if m and m.group(2) in CLOUD_VALUES:
            rows[m.group(1)] = m.group(2)
    return rows


def test_readme_catalog_covers_every_script_and_matches_headers() -> None:
    """The scripts/README.md catalog lists every script, Cloud column == header."""
    catalog = _readme_catalog_clouds()
    assert catalog, "scripts/README.md: master catalog table not found/parseable"
    missing = [_rel(p) for p in _SCRIPTS if _rel(p) not in catalog]
    assert not missing, (
        "scripts/README.md master catalog is missing rows for: "
        f"{missing} (regenerate the table from the headers)"
    )
    mismatched = {
        _rel(p): (catalog[_rel(p)], _cloud_of(p))
        for p in _SCRIPTS
        if catalog[_rel(p)] != _cloud_of(p)
    }
    assert not mismatched, (
        "scripts/README.md catalog Cloud column disagrees with the script "
        f"headers (catalog, header): {mismatched}"
    )
    stale = sorted(set(catalog) - {_rel(p) for p in _SCRIPTS})
    assert not stale, f"scripts/README.md catalog lists scripts that no longer exist: {stale}"
