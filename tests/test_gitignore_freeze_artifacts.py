"""Pins for the .gitignore freeze-artifact negations (task #144, Topic 13 L-B(iii)).

The G1 registration binding (ADR-0089) points the confirmatory driver at the ONE
tracked registration document and its machine-readable margins sidecar:

- ``MyDocs/registration/PRE_REGISTRATION.md``   (PREREG_PATH)
- ``MyDocs/registration/registered_margins.json`` (REGISTERED_MARGINS_PATH)

The 2026-08-19 walkthrough (MyDocs/registration/CODE_ASSERTION_2026-08.md,
Topic 13 finding L-B(iii)) found .gitignore ignoring ``MyDocs/`` wholesale with
zero negations — making both freeze artifacts UNTRACKABLE, so G1 would bind the
one look to an uncommitted, silently-editable file and a reviewer at the
registered SHA would get no prereg at all. The fix restructures the block to
``MyDocs/*`` + dir/file negations for exactly those two artifacts while every
other MyDocs file stays ignored (ADR-0079 privacy boundary).

These pins ask git ITSELF (``git check-ignore``) rather than re-parsing the
pattern text, so any future .gitignore edit that regresses the trackability is
caught by behavior, not by grep. Pure local checks: no GPU, no network.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = REPO_ROOT / ".gitignore"

#: The two freeze artifacts the confirmatory driver binds to (paths mirror
#: PREREG_PATH / REGISTERED_MARGINS_PATH in scripts/4_analysis/run_campaign_analysis.py).
FREEZE_ARTIFACTS = (
    "MyDocs/registration/PRE_REGISTRATION.md",
    "MyDocs/registration/registered_margins.json",
)

#: Representative MyDocs paths that MUST remain ignored (ADR-0079): the charter,
#: the ADR register, living docs, and every OTHER registration/ file.
MUST_STAY_IGNORED = (
    "MyDocs/PUBLICATION.md",
    "MyDocs/DECISIONS.md",
    "MyDocs/BACKLOG.md",
    "MyDocs/LEDGER.md",
    "MyDocs/registration/CODE_ASSERTION_2026-08.md",
    "MyDocs/registration/AMENDMENT_LOG.md",
    "MyDocs/registration/CLAIM_LADDER.md",
    # Nested-below-registration paths stay ignored too (the negation is
    # file-exact, not a subtree re-open).
    "MyDocs/registration/power_sim_2026-08-07/power_tables.csv",
)


def _is_ignored(rel_path: str) -> bool:
    """True iff git's ignore rules would ignore rel_path (works for absent files)."""
    proc = subprocess.run(
        ["git", "check-ignore", "-q", "--", rel_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # 0 = ignored, 1 = not ignored, 128 = error (fail loud, never silently pass).
    if proc.returncode not in (0, 1):
        raise AssertionError(
            f"git check-ignore errored (rc={proc.returncode}) on {rel_path!r}: "
            f"{proc.stderr.strip()}"
        )
    return proc.returncode == 0


@pytest.mark.parametrize("artifact", FREEZE_ARTIFACTS)
def test_freeze_artifact_is_trackable(artifact: str) -> None:
    """Both #112 freeze artifacts must NOT be ignored -- else the G1 binding is
    hollow (the one look would bind to an untracked, silently-editable file)."""
    assert not _is_ignored(artifact), (
        f"{artifact} is gitignored -- the #112 freeze cannot track it and the "
        f"G1 registration binding (ADR-0089) is hollow. Restore the negation "
        f"block in .gitignore (Topic 13 L-B(iii), task #144)."
    )


@pytest.mark.parametrize("path", MUST_STAY_IGNORED)
def test_other_mydocs_paths_stay_ignored(path: str) -> None:
    """The negations are file-exact: everything else in MyDocs/ (charter, ADR
    register, living docs, other registration files) stays private (ADR-0079)."""
    assert _is_ignored(path), (
        f"{path} is NOT gitignored -- the MyDocs privacy boundary (ADR-0079) "
        f"leaked. Only the two freeze artifacts may be negated."
    )


def test_gitignore_carries_the_negation_block() -> None:
    """Cheap static companion: the wholesale `MyDocs/` ignore must not return
    (it would silently re-shadow the negations on some git versions' ordering),
    and the four structural lines of the fix must all be present."""
    text = GITIGNORE.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert "MyDocs/" not in lines, (
        "wholesale `MyDocs/` ignore is back in .gitignore -- it makes the "
        "freeze artifacts untrackable (a dir-level ignore beats file negations)"
    )
    for required in (
        "MyDocs/*",
        "!MyDocs/registration/",
        "MyDocs/registration/*",
        "!MyDocs/registration/PRE_REGISTRATION.md",
        "!MyDocs/registration/registered_margins.json",
    ):
        assert required in lines, f".gitignore lost the line {required!r} (task #144)"


def test_freeze_artifact_paths_match_the_driver() -> None:
    """The negated paths must be the SAME paths the confirmatory driver binds to
    -- a renamed PREREG_PATH with a stale negation would quietly re-open L-B(iii)."""
    driver = (
        REPO_ROOT / "scripts" / "4_analysis" / "run_campaign_analysis.py"
    ).read_text(encoding="utf-8")
    assert '"MyDocs" / "registration" / "PRE_REGISTRATION.md"' in driver, (
        "run_campaign_analysis.py no longer points PREREG_PATH at "
        "MyDocs/registration/PRE_REGISTRATION.md -- update the .gitignore "
        "negation AND this test together"
    )
    assert '"MyDocs" / "registration" / "registered_margins.json"' in driver, (
        "run_campaign_analysis.py no longer points REGISTERED_MARGINS_PATH at "
        "MyDocs/registration/registered_margins.json -- update the .gitignore "
        "negation AND this test together"
    )
