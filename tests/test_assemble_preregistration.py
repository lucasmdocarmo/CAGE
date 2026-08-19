"""Offline tests for scripts/4_analysis/assemble_preregistration.py (#112 prepare).

The assembler is the freeze instrument that mints the ONE tracked
registration document (ADR-0089/G1, ADR-0092). These tests pin its
fail-closed contract on synthetic skeletons + a tmp git repo — no network,
no models, no MyDocs dependency (the one real-draft pin skips off-owner
machines):

- unresolved placeholders REFUSE listing every missing name;
- missing EMBED-FILE / source_file sources REFUSE naming the path;
- unknown resolution keys (typo guard) REFUSE;
- --force overwrite semantics (existing output refuses without it);
- happy path produces byte-stable output (deterministic assembly);
- --require-clean refuses on a dirty synthetic git repo and passes on a
  clean one;
- FREEZE_SHA grammar + the 'Machinery SHA: `<sha>`' binding-line guard stay
  aligned with the G1 regex in run_campaign_analysis.py;
- tokens smuggled through resolution values are caught by the final scan;
- DRAFT-ONLY blocks are stripped, and unbalanced markers refuse.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import assemble_preregistration as ap  # noqa: E402

SHA_OK = "a" * 40

SKELETON = """\
# Synthetic prereg

Machinery SHA: `{{PLACEHOLDER:FREEZE_SHA}}`

tau = {{PLACEHOLDER:QASPER_TAU}}

<!-- DRAFT-ONLY-BEGIN -->
draft-editing note that must never reach the frozen document
<!-- DRAFT-ONLY-END -->

## Appendix

{{EMBED-FILE:embedded.md}}
"""

EMBEDDED_TEXT = "embedded appendix body\n"


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    (tmp_path / "skeleton.md").write_text(SKELETON, encoding="utf-8")
    (tmp_path / "embedded.md").write_text(EMBEDDED_TEXT, encoding="utf-8")
    return tmp_path


def _write_resolutions(root: Path, mapping: dict) -> Path:
    import json

    path = root / "resolutions.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return path


def _run(root: Path, resolutions: dict | None, *extra: str) -> int:
    args = [
        "--skeleton", str(root / "skeleton.md"),
        "--output", str(root / "out.md"),
        "--repo-root", str(root),
    ]
    if resolutions is not None:
        args += ["--resolutions", str(_write_resolutions(root, resolutions))]
    args += list(extra)
    return ap.main(args)


HAPPY = {"FREEZE_SHA": SHA_OK, "QASPER_TAU": "0.83"}


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_unresolved_placeholders_refuse_listing_every_name(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(workdir, {})
    err = capsys.readouterr().err
    assert rc == 2
    assert "REFUSED" in err and "unresolved placeholder" in err
    assert "FREEZE_SHA" in err and "QASPER_TAU" in err
    assert not (workdir / "out.md").exists()


def test_unknown_resolution_key_refuses(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(workdir, {**HAPPY, "TYPO_NAME": "x"})
    err = capsys.readouterr().err
    assert rc == 2
    assert "TYPO_NAME" in err and "not present in the skeleton" in err


def test_missing_embed_file_refuses_naming_path(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workdir / "embedded.md").unlink()
    rc = _run(workdir, HAPPY)
    err = capsys.readouterr().err
    assert rc == 2
    assert "missing EMBED-FILE source" in err and "embedded.md" in err


def test_missing_source_file_resolution_refuses(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(
        workdir,
        {**HAPPY, "QASPER_TAU": {"source_file": "no_such_manifest.json"}},
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "missing resolution source file" in err
    assert "no_such_manifest.json" in err


def test_malformed_resolution_value_refuses(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(workdir, {**HAPPY, "QASPER_TAU": 0.83})  # number, not string
    err = capsys.readouterr().err
    assert rc == 2
    assert "malformed resolutions" in err and "QASPER_TAU" in err


def test_freeze_sha_grammar_refuses(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(workdir, {**HAPPY, "FREEZE_SHA": "NOT-A-SHA"})
    err = capsys.readouterr().err
    assert rc == 2
    assert "FREEZE_SHA" in err and "hex" in err


def test_qasper_tau_range_refuses(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(workdir, {**HAPPY, "QASPER_TAU": "1.7"})
    err = capsys.readouterr().err
    assert rc == 2
    assert "QASPER_TAU" in err and "(0, 1)" in err


def test_machinery_sha_line_guard_refuses_when_line_absent(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    skel = workdir / "skeleton.md"
    skel.write_text(
        skel.read_text(encoding="utf-8").replace(
            "Machinery SHA: `{{PLACEHOLDER:FREEZE_SHA}}`",
            "machinery sha (broken drift): {{PLACEHOLDER:FREEZE_SHA}}",
        ),
        encoding="utf-8",
    )
    rc = _run(workdir, HAPPY)
    err = capsys.readouterr().err
    assert rc == 2
    assert "Machinery SHA" in err and "skeleton drift" in err


def test_token_smuggled_via_resolution_value_refuses(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(workdir, {**HAPPY, "QASPER_TAU": "{{PLACEHOLDER:SNEAKY}}"})
    err = capsys.readouterr().err
    assert rc == 2
    # Caught either as an unresolvable name or by the final leftover scan --
    # both are fail-closed; pin that SNEAKY is named.
    assert "SNEAKY" in err


def test_unbalanced_draft_only_markers_refuse(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    skel = workdir / "skeleton.md"
    skel.write_text(
        skel.read_text(encoding="utf-8").replace(ap.DRAFT_ONLY_END + "\n", ""),
        encoding="utf-8",
    )
    rc = _run(workdir, HAPPY)
    err = capsys.readouterr().err
    assert rc == 2
    assert "unterminated" in err


# ---------------------------------------------------------------------------
# happy path + overwrite semantics
# ---------------------------------------------------------------------------

def test_happy_path_assembles_byte_stable_output(workdir: Path) -> None:
    assert _run(workdir, HAPPY) == 0
    first = (workdir / "out.md").read_bytes()
    text = first.decode("utf-8")
    assert f"Machinery SHA: `{SHA_OK}`" in text
    assert "tau = 0.83" in text
    assert EMBEDDED_TEXT.rstrip("\n") in text
    assert "draft-editing note" not in text  # DRAFT-ONLY stripped
    assert "{{PLACEHOLDER:" not in text and "{{EMBED-FILE:" not in text
    # Deterministic: re-assembly over the same inputs is byte-identical.
    assert _run(workdir, HAPPY, "--force") == 0
    assert (workdir / "out.md").read_bytes() == first


def test_existing_output_refuses_without_force(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workdir / "out.md").write_text("already frozen", encoding="utf-8")
    rc = _run(workdir, HAPPY)
    err = capsys.readouterr().err
    assert rc == 2
    assert "already exists" in err and "--force" in err
    assert (workdir / "out.md").read_text(encoding="utf-8") == "already frozen"


def test_force_overwrites_existing_output(workdir: Path) -> None:
    (workdir / "out.md").write_text("already frozen", encoding="utf-8")
    assert _run(workdir, HAPPY, "--force") == 0
    assert "Machinery SHA" in (workdir / "out.md").read_text(encoding="utf-8")


def test_check_mode_writes_nothing(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(workdir, HAPPY, "--check")
    out = capsys.readouterr().out
    assert rc == 0
    assert "CHECK OK" in out and "sha256" in out
    assert not (workdir / "out.md").exists()


def test_list_placeholders_inventory(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(workdir, None, "--list-placeholders")
    out = capsys.readouterr().out
    assert rc == 0
    assert "FREEZE_SHA" in out and "QASPER_TAU" in out
    assert not (workdir / "out.md").exists()


def test_missing_resolutions_flag_refuses(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(workdir, None)
    err = capsys.readouterr().err
    assert rc == 2
    assert "--resolutions is required" in err


# ---------------------------------------------------------------------------
# --require-clean on a tmp git repo
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c", "user.email=cage@test",
            "-c", "user.name=cage-test",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def git_workdir(workdir: Path) -> Path:
    _git(workdir, "init", "-q")
    _write_resolutions(workdir, HAPPY)
    _git(workdir, "add", "-A")
    _git(workdir, "commit", "-q", "-m", "init")
    return workdir


def test_require_clean_passes_on_clean_repo(git_workdir: Path) -> None:
    assert _run(git_workdir, HAPPY, "--require-clean") == 0
    assert (git_workdir / "out.md").exists()


def test_require_clean_refuses_on_dirty_repo(
    git_workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (git_workdir / "uncommitted_edit.txt").write_text("dirt", encoding="utf-8")
    rc = _run(git_workdir, HAPPY, "--require-clean")
    err = capsys.readouterr().err
    assert rc == 2
    assert "DIRTY" in err
    assert not (git_workdir / "out.md").exists()


def test_require_clean_refuses_outside_a_git_repo(
    workdir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(workdir, HAPPY, "--require-clean")
    err = capsys.readouterr().err
    assert rc == 2
    assert "git status failed" in err or "cannot determine git state" in err


# ---------------------------------------------------------------------------
# cross-pins against the G1 binding + the real draft skeleton
# ---------------------------------------------------------------------------

def test_machinery_line_regex_equals_driver_g1_regex() -> None:
    """The assembler's binding-line pattern IS the driver's grep (ADR-0089)."""
    import run_campaign_analysis as rca

    assert (
        ap.MACHINERY_SHA_LINE_RE.pattern
        == rca._PREREG_EMBEDDED_SHA_RE.pattern
    )


def test_cli_subprocess_refusal_exit_code(tmp_path: Path) -> None:
    (tmp_path / "skeleton.md").write_text(
        "x {{PLACEHOLDER:ONLY_ONE}}\n", encoding="utf-8"
    )
    res = _write_resolutions(tmp_path, {})
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS_DIR / "assemble_preregistration.py"),
            "--skeleton", str(tmp_path / "skeleton.md"),
            "--resolutions", str(res),
            "--output", str(tmp_path / "out.md"),
            "--repo-root", str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "REFUSED" in proc.stderr and "ONLY_ONE" in proc.stderr


_REAL_DRAFT = REPO_ROOT / "MyDocs" / "registration" / "PRE_REGISTRATION_DRAFT.md"
#: The seven registered freeze-time values (#112 task definition).
_REGISTERED_PLACEHOLDERS = frozenset(
    {
        "QASPER_TAU",
        "TAU_BAND",
        "REGISTERED_MARGINS",
        "INSTRUMENT_REVISIONS",
        "FREEZE_SHA",
        "SUITE_AT_SHA",
        "OSF_ID",
    }
)


@pytest.mark.skipif(
    not _REAL_DRAFT.is_file(),
    reason="MyDocs draft skeleton absent (gitignored owner-machine artifact)",
)
def test_real_draft_skeleton_pin() -> None:
    """Owner-machine pin: the real draft's placeholder inventory is exactly
    the seven registered names, its G1 binding line survives stripping, and
    every EMBED-FILE source exists."""
    text = _REAL_DRAFT.read_text(encoding="utf-8")
    stripped = ap.strip_draft_only(text)
    names = set(ap.PLACEHOLDER_RE.findall(stripped))
    assert names == set(_REGISTERED_PLACEHOLDERS)
    assert "Machinery SHA: `{{PLACEHOLDER:FREEZE_SHA}}`" in stripped
    for rel in ap.EMBED_RE.findall(stripped):
        assert (REPO_ROOT / rel.strip()).is_file(), rel
