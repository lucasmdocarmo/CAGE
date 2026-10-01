"""ragas is gone from the pod venv (S0F-2, S0F-3; ADR-0129).

The S0 finding: ``ragas>=0.1.0,<0.5`` pulled ``instructor 0.4.0`` (``openai<2``,
``typer<0.10``) into cage-env; pip then downgraded typer to 0.9.4 and openai to
1.109.1 against vLLM 0.19.1's ``openai>=2`` and fastapi-cli's ``typer>=0.16``,
and the pod uninstalled ragas, instructor and langchain-openai by hand. Nothing
under src/, scripts/ or tests/ imports ragas: the §8.6(d) RAGAS-style judge in
scripts/4_analysis/ragas_head_to_head.py is a prompt sent over HTTP with
``requests``. With ragas out, typer stays at vLLM's version and the fastapi CLI
conflict (S0F-3) needs no code.

Pinned here: no ragas line in requirements.txt; no ragas entry in the
provenance scoring-stack list (it recorded None with a warning); the head-to-head
runner imports no ragas; and ``instrument_versions()`` carries no ragas key.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIREMENTS = REPO_ROOT / "requirements.txt"
HEAD_TO_HEAD = REPO_ROOT / "scripts" / "4_analysis" / "ragas_head_to_head.py"


def _requirement_names(text: str):
    names = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if m:
            names.append(m.group(1).lower())
    return names


def test_requirements_declare_no_ragas() -> None:
    names = _requirement_names(REQUIREMENTS.read_text(encoding="utf-8"))
    assert "ragas" not in names, "ragas drags instructor (openai<2, typer<0.10) into cage-env"
    # the packages the S0 pod had to uninstall by hand must not be declared either
    for gone in ("instructor", "langchain-openai", "langchain"):
        assert gone not in names, gone
    # requests, which the head-to-head judge needs, stays
    assert "requests" in names


def test_provenance_scoring_stack_has_no_ragas_entry() -> None:
    from src.observability import provenance as prov

    assert "ragas" not in prov.SCORING_STACK_PACKAGES
    assert "ragas" not in prov.instrument_versions()


def test_head_to_head_runner_imports_no_ragas() -> None:
    text = HEAD_TO_HEAD.read_text(encoding="utf-8")
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert not re.search(r"^\s*(import ragas|from ragas)", code, re.M)
    assert "import requests" in code  # the judge speaks HTTP, nothing else
