"""K-PIN1 (task #140): pin that every preflight gate VERDICT is consumed.

The Topic-12 finding: preflight_check.sh's ``gate_rc`` *definition* was pinned
(the ``0|3) : ;;`` string), but nothing pinned its CONSUMPTION — deleting a
single ``gate_rc $?`` line after a python sub-gate disconnected that gate from
the overall exit code with all tests green (the J5 defect class: gate runs,
verdict unconsumed). Three layers close it:

1. WIRING (re-reads the CURRENT script): every ``python3 ... <<'PY'`` heredoc
   sub-gate is verdict-consumed — either launched under ``if !`` with a
   ``FAILED=1`` branch, or immediately followed by ``gate_rc $?``.
2. EXTRACT-AND-EXECUTE: the CURRENT ``gate_rc`` function body runs under real
   bash — rc 0 and rc 3 (skip-with-reason) leave the accumulator untouched,
   any other rc sets ``FAILED=1``.
3. END-TO-END SUBPROCESS: the REAL, unmodified preflight_check.sh runs with
   ``python3``/``curl`` PATH stubs so every external dependency is green, and
   ONE targeted gate — (n), the CAGE-CALIBRATION-ARTIFACT-GATE named by the
   K row's pre-calibration-skip semantics — is driven to rc 0 / 3 / 1 / 7:
   all-green exits 0, a skip (rc 3) still exits 0, a poisoned/failing gate
   makes the overall preflight exit nonzero. A poison env var proves the
   in-shell ``fail()`` path reaches the exit code too.

No GPU, no server, no network: curl/python3 are stubbed at the PATH boundary.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO_ROOT / "scripts" / "checks" / "preflight_check.sh"

#: The default MODEL argument the script probes for at /v1/models.
DEFAULT_MODEL = "Qwen/Qwen3-8B"

#: The targeted sub-gate: gate (n) declares the pre-calibration skip (rc 3)
#: precondition the K row names, so it is the clean lever for pass/skip/fail.
TARGET_GATE_MARKER = "CAGE-CALIBRATION-ARTIFACT-GATE"

# Env vars the gates read: strip them from every subprocess so a developer
# shell can never leak poison/scope state into the end-to-end runs (same list
# discipline as tests/test_preflight_gates.py).
_GATE_ENV_VARS = (
    "CAGE_ISO_BYTES_TOL", "CAGE_ISO_BYTES_LOGS", "CAGE_ISO_BYTES_LOG_ROOT",
    "CAGE_CALIBRATION_MANIFESTS", "CAGE_DATASETS", "DATASET",
    "CAGE_REGIME_GATE_SAMPLES", "CAGE_REGIME_GATE_INTERVAL",
    "CAGE_REGIME_KV_METRIC", "CAGE_REGIME_PREEMPT_METRIC",
    "CAGE_TELEMETRY_MOCK", "CAGE_DISABLE_LETTUCEDETECT",
    "CAGE_DISABLE_COMPRESSION", "CAGE_ALLOW_NO_COMPRESSION",
    "CAGE_ALLOW_REPLAY", "CAGE_ALLOW_NO_BACKUP",
    "LMDEPLOY_CACHE_MAX_ENTRY_COUNT", "LMDEPLOY_QUANT_POLICY",
    "CAGE_QUALITY_STRICT", "CAGE_LMDEPLOY_BACKEND_CHECK", "CAGE_CLAIM_CHECKER",
    "CAGE_PREFLIGHT_BACKENDS", "CAGE_MIN_FREE_GB",
    "CAGE_TEST_CAL_GATE_RC",
)


def _text() -> str:
    return PREFLIGHT.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Wiring: every heredoc sub-gate's verdict is consumed
# ---------------------------------------------------------------------------


def _heredoc_gates(lines: List[str]) -> List[Tuple[int, int]]:
    """(launch_line_idx, terminator_line_idx) for every python heredoc gate."""
    gates = []
    for i, line in enumerate(lines):
        if re.search(r"python3\s+-.*<<'PY'\s*$", line):
            j = next(k for k in range(i + 1, len(lines)) if lines[k] == "PY")
            gates.append((i, j))
    return gates


def test_every_python_heredoc_gate_is_verdict_consumed() -> None:
    """Deleting one `gate_rc $?` (or one `FAILED=1` branch) must fail HERE."""
    lines = _text().splitlines()
    gates = _heredoc_gates(lines)
    # Gate-count floor: 10 python sub-gates today (b/c/d, h, i, j, l, m, n, o,
    # p, q minus the if-wrapped trio counted once each). A vanished gate is a
    # louder defect than an unconsumed one.
    assert len(gates) >= 10, (
        f"only {len(gates)} python heredoc sub-gates found in "
        f"{PREFLIGHT.name} — gates were deleted or the launch pattern drifted"
    )
    for i, j in gates:
        launch = lines[i]
        if re.match(r"\s*if\s+!\s+python3", launch):
            # `if ! python3 ... <<'PY' ... PY` / `then FAILED=1 fi`
            tail = "\n".join(lines[j + 1 : j + 6])
            assert "FAILED=1" in tail, (
                f"{PREFLIGHT.name}:{i + 1}: if-wrapped sub-gate no longer sets "
                f"FAILED=1 in its failure branch (K-PIN1):\n{launch}"
            )
        else:
            after = next((l for l in lines[j + 1 :] if l.strip()), "")
            assert after.strip() == "gate_rc $?", (
                f"{PREFLIGHT.name}:{i + 1}: sub-gate is not followed by "
                f"`gate_rc $?` — its verdict is disconnected from the exit "
                f"code (K-PIN1):\n{launch}\n(next line: {after!r})"
            )


def test_final_verdict_consumes_the_accumulator() -> None:
    text = _text()
    assert re.search(r'if \[ "\$FAILED" -eq 0 \]', text), (
        "the final verdict no longer reads the FAILED accumulator (K-PIN1)"
    )
    # Both terminal exits present: green -> 0, any accumulated failure -> 1.
    tail = text[text.rindex('if [ "$FAILED" -eq 0 ]') :]
    assert "exit 0" in tail and "exit 1" in tail


# ---------------------------------------------------------------------------
# 2. Extract-and-execute: gate_rc semantics from the CURRENT script
# ---------------------------------------------------------------------------


def test_gate_rc_semantics_executed() -> None:
    """rc 0 and rc 3 leave FAILED alone; any other rc sets it; FAILED is
    sticky (a later green gate cannot wash out an earlier failure)."""
    m = re.search(r"^gate_rc\(\) \{[^\n]*\n.*?^\}", _text(), re.M | re.S)
    assert m, "gate_rc() not found in preflight_check.sh"
    script = f"""
set -uo pipefail
FAILED=0
{m.group(0)}
gate_rc 0; echo "after0=$FAILED"
gate_rc 3; echo "after3=$FAILED"
gate_rc 2; echo "after2=$FAILED"
gate_rc 0; echo "sticky=$FAILED"
"""
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, proc.stderr
    for token in ("after0=0", "after3=0", "after2=1", "sticky=1"):
        assert token in proc.stdout, (
            f"gate_rc semantics drifted (expected {token}):\n{proc.stdout}"
        )


# ---------------------------------------------------------------------------
# 3. End-to-end: the REAL script under PATH stubs
# ---------------------------------------------------------------------------


def _stub_dir(tmp_path: Path) -> Path:
    """python3/curl stubs: everything green; the targeted gate's rc comes
    from CAGE_TEST_CAL_GATE_RC (matched by its stable CAGE-* marker in the
    heredoc body, so gate additions/reordering cannot break this test)."""
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    py = stubs / "python3"
    py.write_text(
        "#!/bin/bash\n"
        'body=""\n'
        'if [ ! -t 0 ]; then body="$(cat)"; fi\n'
        'case "$body" in\n'
        f"  *{TARGET_GATE_MARKER}*) exit \"${{CAGE_TEST_CAL_GATE_RC:-0}}\" ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    py.chmod(0o755)
    curl = stubs / "curl"
    curl.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' '{DEFAULT_MODEL}'\n"
        "exit 0\n",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    return stubs


def _run_preflight(tmp_path: Path, **extra_env: str) -> subprocess.CompletedProcess:
    stubs = _stub_dir(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in _GATE_ENV_VARS}
    env["PATH"] = f"{stubs}:{env['PATH']}"
    env["CAGE_MIN_FREE_GB"] = "1"  # gate (f) probes the real disk; keep it green
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(PREFLIGHT)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
    )


def test_e2e_all_gates_green_exits_zero(tmp_path: Path) -> None:
    """Harness soundness baseline: with every dependency stubbed green the
    REAL script exits 0 — required for the skip/poison cases to mean anything."""
    proc = _run_preflight(tmp_path)
    assert proc.returncode == 0, (
        f"all-green preflight exited {proc.returncode} — the end-to-end "
        f"harness lost its baseline:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "PREFLIGHT PASS" in proc.stdout
    assert "(n) calibration" in proc.stdout or "(n)" in proc.stdout, (
        "gate (n) no longer announces itself — retarget TARGET_GATE_MARKER"
    )


def test_e2e_rc3_skip_gate_does_not_fail_preflight(tmp_path: Path) -> None:
    """The K-row semantics: rc 3 = skip-with-reason (pre-calibration), NOT a
    failure — the overall preflight still passes."""
    proc = _run_preflight(tmp_path, CAGE_TEST_CAL_GATE_RC="3")
    assert proc.returncode == 0, (
        f"an rc=3 (declared skip) gate FAILED the preflight (exit "
        f"{proc.returncode}) — skip semantics lost:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "PREFLIGHT PASS" in proc.stdout


def test_e2e_poisoned_gate_fails_preflight(tmp_path: Path) -> None:
    """THE K-PIN1 pin: one failing sub-gate must poison the overall exit code.
    Deleting the `gate_rc $?` after gate (n) makes exactly this test fail."""
    proc = _run_preflight(tmp_path, CAGE_TEST_CAL_GATE_RC="1")
    assert proc.returncode != 0, (
        "a FAILING sub-gate did not fail the preflight — its verdict is "
        f"unconsumed (K-PIN1):\n{proc.stdout}"
    )
    assert "PREFLIGHT FAIL" in proc.stdout


def test_e2e_crash_rc_fails_preflight(tmp_path: Path) -> None:
    """Any rc outside {0, 3} — e.g. a python crash's 7 — is a failure, never
    a skip."""
    proc = _run_preflight(tmp_path, CAGE_TEST_CAL_GATE_RC="7")
    assert proc.returncode != 0, (
        f"rc=7 sub-gate did not fail the preflight (K-PIN1):\n{proc.stdout}"
    )
    assert "PREFLIGHT FAIL" in proc.stdout


def test_e2e_in_shell_fail_path_reaches_exit_code(tmp_path: Path) -> None:
    """The fail() accumulator path (in-shell gates) must also reach the exit
    code: one poison env var on an otherwise all-green run fails the preflight."""
    proc = _run_preflight(tmp_path, CAGE_TELEMETRY_MOCK="1")
    assert proc.returncode != 0, (
        f"poison env var did not fail the preflight:\n{proc.stdout}"
    )
    assert "CAGE_TELEMETRY_MOCK" in proc.stdout
    assert "PREFLIGHT FAIL" in proc.stdout
