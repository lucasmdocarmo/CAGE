#!/usr/bin/env python3
"""Assemble the ONE tracked registration document (freeze task #112 — assembler).

Reads a draft skeleton (default: MyDocs/registration/PRE_REGISTRATION_DRAFT.md)
plus a resolutions JSON and produces the final, frozen
MyDocs/registration/PRE_REGISTRATION.md — the single tracked registration
artifact the confirmatory look is mechanically bound to (ADR-0089/G1;
run_campaign_analysis.py PREREG_PATH + _PREREG_EMBEDDED_SHA_RE). Charter
authority: MyDocs/PUBLICATION.md D9 §9.13; embedding doctrine: ADR-0092
(EMBED-AT-FREEZE) + ADR-0079.

Skeleton grammar (two token forms, nothing else is special):

- ``{{PLACEHOLDER:NAME}}``  — a value that exists only at freeze time (e.g.
  FREEZE_SHA, QASPER_TAU). Resolved from the resolutions JSON: a mapping of
  NAME -> value, where value is either a plain string or an object
  ``{"source_file": "<repo-root-relative path>"}`` whose file content (with
  the trailing newline stripped) is embedded verbatim.
- ``{{EMBED-FILE:<repo-root-relative path>}}`` — embeds the file's full text
  (trailing newline stripped) at assembly time. Used for the ADR-0092 full
  DECISIONS.md embed and the registration-draft appendices, so the frozen
  document carries whatever those sources hold AT the freeze SHA.

Additionally, any lines between ``<!-- DRAFT-ONLY-BEGIN -->`` and
``<!-- DRAFT-ONLY-END -->`` marker lines (inclusive) are STRIPPED at assembly
— skeleton-editing notes that must not appear in the frozen document.
Unbalanced or nested markers refuse (fail closed).

FAIL-CLOSED CONTRACT (every branch refuses with exit code 2, never degrades):

- every ``{{PLACEHOLDER:NAME}}`` must be resolved; otherwise REFUSE listing
  every unresolved name;
- every resolutions key must exist in the skeleton (a typo'd key would
  otherwise silently resolve nothing) — unknown keys REFUSE;
- every referenced source file (EMBED-FILE or ``source_file``) must exist;
- ``--require-clean`` REFUSES when ``git status --porcelain`` is non-empty
  (or when git state cannot be determined at all);
- an existing output REFUSES without ``--force``;
- after substitution, NO token of either form may survive in the output
  (a token smuggled in via an embedded file or a resolution value refuses);
- FREEZE_SHA, when present, must be 7-64 char lowercase hex AND the resolved
  output must carry the exact ``Machinery SHA: `<sha>``` line the G1 binding
  greps for (mirrors run_campaign_analysis.py:241) — skeleton drift that
  breaks the binding line refuses HERE, not at the one look;
- QASPER_TAU, when present, must parse as a float strictly inside (0, 1).

This script NEVER touches the network and NEVER writes to git (its only git
use is a read-only ``git status --porcelain`` under ``--require-clean``).
It writes exactly one file: the output document. Committing, tagging and the
OSF registration are OWNER steps — see
MyDocs/registration/FREEZE_RUNBOOK_2026-08-19.md.

Exit codes: 0 = success (assembled / --check passed / --list-placeholders);
2 = refusal (message on stderr, prefixed ``REFUSED:``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SKELETON = Path("MyDocs/registration/PRE_REGISTRATION_DRAFT.md")
DEFAULT_OUTPUT = Path("MyDocs/registration/PRE_REGISTRATION.md")

PLACEHOLDER_RE = re.compile(r"\{\{PLACEHOLDER:([A-Za-z0-9_]+)\}\}")
EMBED_RE = re.compile(r"\{\{EMBED-FILE:([^{}\n]+)\}\}")
#: MUST stay equal to run_campaign_analysis.py `_PREREG_EMBEDDED_SHA_RE`
#: (G1 binding, ADR-0089): the confirmatory look greps the assembled document
#: for exactly this line.
MACHINERY_SHA_LINE_RE = re.compile(r"Machinery SHA: `([0-9a-f]{7,64})`")
_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")


DRAFT_ONLY_BEGIN = "<!-- DRAFT-ONLY-BEGIN -->"
DRAFT_ONLY_END = "<!-- DRAFT-ONLY-END -->"


class AssemblyRefusal(Exception):
    """Fail-closed refusal; the message names every cause found."""


def strip_draft_only(text: str) -> str:
    """Remove every DRAFT-ONLY block (marker lines inclusive).

    Refuses on unbalanced or nested markers — a half-stripped freeze document
    is worse than no document.
    """
    out: List[str] = []
    inside = False
    for lineno, line in enumerate(text.splitlines(keepends=True), start=1):
        stripped = line.strip()
        if stripped == DRAFT_ONLY_BEGIN:
            if inside:
                raise AssemblyRefusal(
                    f"nested {DRAFT_ONLY_BEGIN} at line {lineno} — DRAFT-ONLY "
                    "blocks must not nest"
                )
            inside = True
            continue
        if stripped == DRAFT_ONLY_END:
            if not inside:
                raise AssemblyRefusal(
                    f"unmatched {DRAFT_ONLY_END} at line {lineno}"
                )
            inside = False
            continue
        if not inside:
            out.append(line)
    if inside:
        raise AssemblyRefusal(
            f"unterminated {DRAFT_ONLY_BEGIN} block — missing "
            f"{DRAFT_ONLY_END}"
        )
    return "".join(out)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_text(path: Path, role: str) -> str:
    if not path.is_file():
        raise AssemblyRefusal(f"{role} not found: {path}")
    return path.read_text(encoding="utf-8")


def load_resolutions(path: Path, repo_root: Path) -> Dict[str, str]:
    """Load the resolutions JSON into NAME -> literal replacement text.

    Values: plain string, or {"source_file": "<repo-root-relative path>"}
    (file text embedded, trailing newline stripped). All missing source
    files are collected and refused together.
    """
    raw = _read_text(path, "resolutions JSON")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssemblyRefusal(f"resolutions JSON {path} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise AssemblyRefusal(
            f"resolutions JSON {path} must be a JSON object of "
            "NAME -> string | {\"source_file\": path}"
        )
    resolved: Dict[str, str] = {}
    missing: List[str] = []
    bad: List[str] = []
    for name, value in data.items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            bad.append(f"{name!r} (not a legal placeholder name)")
            continue
        if isinstance(value, str):
            resolved[name] = value
        elif isinstance(value, dict) and set(value) == {"source_file"}:
            src = repo_root / str(value["source_file"])
            if not src.is_file():
                missing.append(f"{name} -> {src}")
            else:
                resolved[name] = src.read_text(encoding="utf-8").rstrip("\n")
        else:
            bad.append(
                f"{name!r} (value must be a string or "
                "{\"source_file\": path}, got: " + repr(value) + ")"
            )
    problems: List[str] = []
    if bad:
        problems.append("malformed resolutions: " + "; ".join(sorted(bad)))
    if missing:
        problems.append(
            "missing resolution source file(s): " + "; ".join(sorted(missing))
        )
    if problems:
        raise AssemblyRefusal(" | ".join(problems))
    return resolved


def expand_embeds(text: str, repo_root: Path) -> str:
    """Expand every {{EMBED-FILE:path}} token; refuse listing ALL missing files."""
    missing: List[str] = []

    def _sub(match: "re.Match[str]") -> str:
        rel = match.group(1).strip()
        src = repo_root / rel
        if not src.is_file():
            missing.append(rel)
            return match.group(0)
        return src.read_text(encoding="utf-8").rstrip("\n")

    expanded = EMBED_RE.sub(_sub, text)
    if missing:
        raise AssemblyRefusal(
            "missing EMBED-FILE source(s): "
            + ", ".join(sorted(set(missing)))
            + f" (paths are repo-root-relative to {repo_root})"
        )
    return expanded


def _validate_known_values(resolutions: Dict[str, str]) -> None:
    """Per-name validators for the load-bearing registered placeholders."""
    problems: List[str] = []
    sha = resolutions.get("FREEZE_SHA")
    if sha is not None and not _SHA_RE.fullmatch(sha):
        problems.append(
            f"FREEZE_SHA {sha!r} is not a 7-64 char lowercase hex git SHA "
            "(the G1 binding would refuse it — run_campaign_analysis.py)"
        )
    tau = resolutions.get("QASPER_TAU")
    if tau is not None:
        try:
            tau_f = float(tau)
        except ValueError:
            tau_f = float("nan")
        if not (0.0 < tau_f < 1.0):
            problems.append(
                f"QASPER_TAU {tau!r} must parse as a float strictly in (0, 1) "
                "(the #146 Instrument-A calibrated threshold)"
            )
    if problems:
        raise AssemblyRefusal(" | ".join(problems))


def assemble_text(
    skeleton_text: str, resolutions: Dict[str, str], repo_root: Path
) -> str:
    """The pure assembly: embeds, substitution, and every fail-closed check
    that does not involve the filesystem output. Returns the final document
    text; raises AssemblyRefusal otherwise. Deterministic: identical inputs
    produce identical bytes (no timestamps, no environment reads).
    """
    expanded = expand_embeds(strip_draft_only(skeleton_text), repo_root)

    names_in_text = set(PLACEHOLDER_RE.findall(expanded))
    unknown = sorted(set(resolutions) - names_in_text)
    if unknown:
        raise AssemblyRefusal(
            "resolutions name(s) not present in the skeleton (typo guard): "
            + ", ".join(unknown)
        )
    unresolved = sorted(names_in_text - set(resolutions))
    if unresolved:
        raise AssemblyRefusal(
            "unresolved placeholder(s) — the freeze cannot proceed with "
            "holes in the registration: " + ", ".join(unresolved)
        )
    _validate_known_values(resolutions)

    final = PLACEHOLDER_RE.sub(lambda m: resolutions[m.group(1)], expanded)

    leftovers = sorted(
        set(PLACEHOLDER_RE.findall(final))
        | {f"EMBED-FILE:{p.strip()}" for p in EMBED_RE.findall(final)}
    )
    if leftovers:
        raise AssemblyRefusal(
            "token(s) survive in the assembled output (smuggled via an "
            "embedded file or a resolution value — fail closed): "
            + ", ".join(leftovers)
        )

    sha = resolutions.get("FREEZE_SHA")
    if sha is not None:
        match = MACHINERY_SHA_LINE_RE.search(final)
        if match is None or match.group(1) != sha:
            raise AssemblyRefusal(
                "the assembled document does not carry the exact "
                "'Machinery SHA: `<FREEZE_SHA>`' line the G1 binding greps "
                "for (run_campaign_analysis.py _PREREG_EMBEDDED_SHA_RE) — "
                "skeleton drift; fix the draft, do not freeze"
            )
    return final


def git_is_dirty(repo_root: Path) -> bool:
    """True iff the worktree has ANY uncommitted state (tracked or untracked).

    Fail closed: an undeterminable git state (no git, not a repo) refuses —
    a freeze must PROVE cleanliness, never assume it.
    """
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssemblyRefusal(
            f"cannot determine git state of {repo_root} ({exc}) — "
            "--require-clean must PROVE a clean tree, not assume one"
        )
    if proc.returncode != 0:
        raise AssemblyRefusal(
            f"git status failed in {repo_root} "
            f"(rc={proc.returncode}: {proc.stderr.strip()}) — "
            "--require-clean must PROVE a clean tree, not assume one"
        )
    return bool(proc.stdout.strip())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assemble_preregistration.py",
        description=(
            "Assemble MyDocs/registration/PRE_REGISTRATION.md from the draft "
            "skeleton + a resolutions JSON (freeze task #112). Fail-closed; "
            "offline; never commits."
        ),
    )
    parser.add_argument(
        "--skeleton", type=Path, default=DEFAULT_SKELETON,
        help=f"draft skeleton path (default: {DEFAULT_SKELETON})",
    )
    parser.add_argument(
        "--resolutions", type=Path, default=None,
        help="resolutions JSON (NAME -> string | {'source_file': path}); "
        "required except with --list-placeholders",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"output document path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--repo-root", type=Path, default=_REPO_ROOT,
        help="repo root for relative paths, EMBED-FILE/source_file "
        "resolution, and the --require-clean git check (default: this repo)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing output (without it, an existing output "
        "REFUSES)",
    )
    parser.add_argument(
        "--require-clean", action="store_true",
        help="refuse unless `git status --porcelain` is empty at --repo-root "
        "(the freeze GO path always passes this)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="run EVERY check and print the output digest, but write nothing",
    )
    parser.add_argument(
        "--list-placeholders", action="store_true",
        help="expand embeds, list the placeholder inventory, write nothing "
        "(no resolutions needed)",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root: Path = args.repo_root.resolve()

    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else repo_root / p

    skeleton_path = _abs(args.skeleton)
    output_path = _abs(args.output)

    try:
        skeleton_text = _read_text(skeleton_path, "skeleton")

        if args.list_placeholders:
            expanded = expand_embeds(strip_draft_only(skeleton_text), repo_root)
            names = sorted(set(PLACEHOLDER_RE.findall(expanded)))
            print(f"placeholders ({len(names)}) in {skeleton_path}:")
            for name in names:
                print(f"  {{{{PLACEHOLDER:{name}}}}}")
            return 0

        if args.require_clean and git_is_dirty(repo_root):
            raise AssemblyRefusal(
                f"worktree at {repo_root} is DIRTY (git status --porcelain "
                "non-empty) — the freeze assembles from a committed machinery "
                "state only (--require-clean; ADR-0089/G1 discipline)"
            )

        if args.resolutions is None:
            raise AssemblyRefusal(
                "--resolutions is required (every placeholder must be "
                "resolved; use --list-placeholders for the inventory)"
            )
        resolutions = load_resolutions(_abs(args.resolutions), repo_root)
        final = assemble_text(skeleton_text, resolutions, repo_root)
        digest = hashlib.sha256(final.encode("utf-8")).hexdigest()

        if args.check:
            print(
                f"CHECK OK: {len(final.encode('utf-8'))} bytes, "
                f"sha256 {digest} (nothing written)"
            )
            return 0

        if output_path.exists() and not args.force:
            raise AssemblyRefusal(
                f"output already exists: {output_path} — re-assembling over a "
                "frozen registration requires an explicit --force (and, "
                "post-OSF, a logged §9.11 amendment)"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final, encoding="utf-8")
        print(
            f"assembled {output_path} "
            f"({len(final.encode('utf-8'))} bytes, sha256 {digest})"
        )
        return 0
    except AssemblyRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
