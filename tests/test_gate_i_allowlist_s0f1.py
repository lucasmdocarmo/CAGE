"""Gate (i) pip-check allow-list and the numba companion pin (S0F-1, ADR-0126).

The S0 finding: vLLM 0.19.1 declares ``numba==0.61.2`` (numpy<2.3) while the
Tier-1 pin is ``numpy==2.5.1``, so every pod resolves a newer numba and
``pip check`` prints one line; gate (i) failed on it at every S0 preflight
(results/s0/ops/preflight_s0-19_gate_j.log:43). vLLM imports numba only in the
ngram speculative proposer, which the charter retired, so the conflict is
metadata only for CAGE.

Pinned here:
1. BEHAVIOR (extract-and-execute, the tests/test_preflight_gates.py pattern):
   the gate body is sliced out of preflight_check.sh and run against a fake
   ``pip`` package on PYTHONPATH that feeds canned stdout/stderr/rc. The exact
   S0 line passes as an accepted deviation; the 0.68.0 variant, an extra line,
   a nonzero exit with empty stdout and a malformed allow-list entry all FAIL;
   a clean pip check still passes.
2. STATIC: every allow-list entry carries a reason and an ADR id; the one S0
   entry is the verbatim S0 line; requirements.txt pins the numba version that
   line names; the pin equals the numba installed in this venv when present.

No GPU, no network.
"""
from __future__ import annotations

import importlib.metadata as md
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO_ROOT / "scripts" / "checks" / "preflight_check.sh"
ALLOWLIST = REPO_ROOT / "scripts" / "checks" / "pip_check_allowlist.txt"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
MARKER = "CAGE-ENV-REGISTRATION-GATE"

#: The line every S0 preflight printed (results/s0/ops/preflight_s0-19_gate_j.log:43).
S0_LINE = "vllm 0.19.1 has requirement numba==0.61.2, but you have numba 0.67.0."
DRIFT_LINE = "vllm 0.19.1 has requirement numba==0.61.2, but you have numba 0.68.0."
OTHER_LINE = "lettucedetect 0.1.7 has requirement openai==1.66.3, but you have openai 2.3.0."

_FAKE_PIP = """\
import os
import sys

sys.stdout.write(os.environ.get("FAKE_PIP_STDOUT", ""))
sys.stderr.write(os.environ.get("FAKE_PIP_STDERR", ""))
sys.exit(int(os.environ.get("FAKE_PIP_RC", "0")))
"""


def _snippet() -> str:
    text = PREFLIGHT.read_text(encoding="utf-8")
    start = text.index(f"# {MARKER}")
    end = text.index("\nPY\n", start)
    return text[start:end]


def _root(tmp_path: Path, allowlist: Optional[str]) -> Path:
    """A throwaway CAGE_ROOT: a requirements.txt whose one pin is installed, and
    the allow-list text (None = no file)."""
    root = tmp_path / "root"
    (root / "scripts" / "checks").mkdir(parents=True)
    (root / "requirements.txt").write_text(
        f"pytest=={md.version('pytest')}\n", encoding="utf-8")
    if allowlist is not None:
        (root / "scripts" / "checks" / "pip_check_allowlist.txt").write_text(
            allowlist, encoding="utf-8")
    return root


def _run_gate(tmp_path: Path, *, stdout: str, rc: int, stderr: str = "",
              allowlist: Optional[str] = None) -> subprocess.CompletedProcess:
    fake = tmp_path / "fakepip" / "pip"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("", encoding="utf-8")
    (fake / "__main__.py").write_text(_FAKE_PIP, encoding="utf-8")
    env: Dict[str, str] = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path / "fakepip")
    env["FAKE_PIP_STDOUT"] = stdout
    env["FAKE_PIP_STDERR"] = stderr
    env["FAKE_PIP_RC"] = str(rc)
    canonical = f"{sys.version_info.major}.{sys.version_info.minor}"
    return subprocess.run(
        [sys.executable, "-", canonical, str(_root(tmp_path, allowlist))],
        input=_snippet(), capture_output=True, text=True, env=env,
        cwd=str(REPO_ROOT),
    )


REAL_ALLOWLIST = ALLOWLIST.read_text(encoding="utf-8") if ALLOWLIST.is_file() else ""


# ---------------------------------------------------------------------------
# behavior: the extracted gate against a fake pip
# ---------------------------------------------------------------------------


def test_gate_marker_declared() -> None:
    assert f"# {MARKER}" in PREFLIGHT.read_text(encoding="utf-8")


def test_fake_pip_harness_reaches_the_gate(tmp_path: Path) -> None:
    proc = _run_gate(tmp_path, stdout="No broken requirements found.\n", rc=0,
                     allowlist=REAL_ALLOWLIST)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[ok] pip check clean" in proc.stdout
    assert "1 exact pins verified" in proc.stdout


def test_accepted_s0_line_passes_and_is_printed_with_its_adr(tmp_path: Path) -> None:
    proc = _run_gate(tmp_path, stdout=S0_LINE + "\n", rc=1, allowlist=REAL_ALLOWLIST)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[accepted]" in proc.stdout and S0_LINE in proc.stdout
    assert "ADR-0126" in proc.stdout
    assert "[FAIL]" not in proc.stdout


def test_numba_drift_variant_fails(tmp_path: Path) -> None:
    proc = _run_gate(tmp_path, stdout=DRIFT_LINE + "\n", rc=1, allowlist=REAL_ALLOWLIST)
    assert proc.returncode == 1
    assert "[FAIL] pip check reports broken requirements" in proc.stdout
    assert DRIFT_LINE in proc.stdout
    assert "[accepted]" not in proc.stdout


def test_accepted_plus_unknown_line_fails_naming_the_unknown(tmp_path: Path) -> None:
    proc = _run_gate(tmp_path, stdout=S0_LINE + "\n" + OTHER_LINE + "\n", rc=1,
                     allowlist=REAL_ALLOWLIST)
    assert proc.returncode == 1
    assert "[accepted]" in proc.stdout and S0_LINE in proc.stdout
    fail = proc.stdout.split("[FAIL]", 1)[1]
    assert OTHER_LINE in fail
    assert S0_LINE not in fail, "an accepted line must not be reported as broken"


def test_nonzero_exit_with_empty_stdout_fails_and_shows_stderr(tmp_path: Path) -> None:
    proc = _run_gate(tmp_path, stdout="", rc=1,
                     stderr="ERROR: Exception: metadata parse failure\n",
                     allowlist=REAL_ALLOWLIST)
    assert proc.returncode == 1
    assert "[FAIL]" in proc.stdout
    assert "metadata parse failure" in proc.stdout


def test_missing_allowlist_accepts_nothing(tmp_path: Path) -> None:
    proc = _run_gate(tmp_path, stdout=S0_LINE + "\n", rc=1, allowlist=None)
    assert proc.returncode == 1
    assert S0_LINE in proc.stdout.split("[FAIL]", 1)[1]


@pytest.mark.parametrize("bad", [
    S0_LINE + "\n",                               # no reason, no ADR
    S0_LINE + " ## some reason\n",                # no ADR
    S0_LINE + " ## some reason ## ticket-12\n",   # not an ADR id
])
def test_malformed_allowlist_entry_fails_closed(tmp_path: Path, bad: str) -> None:
    proc = _run_gate(tmp_path, stdout=S0_LINE + "\n", rc=1, allowlist=bad)
    assert proc.returncode == 1
    assert "pip_check_allowlist.txt" in proc.stdout
    assert "[accepted]" not in proc.stdout


# ---------------------------------------------------------------------------
# static: the tracked allow-list and the companion pin
# ---------------------------------------------------------------------------


def _entries():
    rows = []
    for raw in REAL_ALLOWLIST.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([p.strip() for p in line.split(" ## ")])
    return rows


def test_allowlist_exists_with_exactly_the_s0_entry() -> None:
    assert ALLOWLIST.is_file(), "scripts/checks/pip_check_allowlist.txt is tracked"
    rows = _entries()
    assert len(rows) == 1, rows
    line, reason, adr = rows[0]
    assert line == S0_LINE
    assert len(reason) >= 20, "every accepted line carries a reason"
    assert re.fullmatch(r"ADR-\d{4}", adr), adr
    assert adr == "ADR-0126"


def test_requirements_pin_numba_at_the_allowlisted_version() -> None:
    text = REQUIREMENTS.read_text(encoding="utf-8")
    pins = re.findall(r"^numba==(\S+)$", text, re.M)
    assert pins == ["0.67.0"], pins
    m = re.search(r"but you have numba (\S+)\.$", S0_LINE)
    assert m and m.group(1) == pins[0], "the pin and the accepted line must agree"
    # the comment above the pin names the mechanism and the ADR
    at = re.search(r"^numba==0\.67\.0$", text, re.M).start()
    block = text[max(0, at - 1200):at]
    assert "numba==0.61.2" in block and "ADR-0126" in block


def test_companion_pin_matches_the_installed_numba_when_present() -> None:
    try:
        installed = md.version("numba")
    except md.PackageNotFoundError:
        pytest.skip("numba not installed in this venv")
    assert installed == "0.67.0"


def test_preflight_syntax_still_parses() -> None:
    proc = subprocess.run(["bash", "-n", str(PREFLIGHT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
