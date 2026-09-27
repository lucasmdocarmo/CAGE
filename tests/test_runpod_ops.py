"""Pins for the RunPod ops suite (restructure phase 2: CLI v2 + pod ledger).

teardown_pod.sh migrated to the runpodctl 2.x command tree (the v1 verbs
`runpodctl get pod` / `runpodctl remove pod` NO LONGER EXIST — the reference is
MyDocs/RunPod/runpod-cli-reference.md §2), plus four ops scripts:

  provision_pod.sh  PLAN-by-default provisioning (creation only via --yes = the
                    owner GO, honoring the standing run-approval gate) with a
                    12h cost seatbelt and a create event appended to the pod
                    ledger (results/ops/pod_ledger.jsonl). The seatbelt is
                    CLIENT-SIDE: pod_watchdog.sh, a detached workstation loop
                    armed right after the create, deletes the pod at the
                    resolved deadline. RunPod has NO server-side auto-terminate
                    (live 2026-09-26: `runpodctl pod create` 2.11.0 and 2.14.0
                    reject --terminate-after with code usage_error; the v2
                    CreatePodRequest has no such field), so the flag must
                    NEVER reach the CLI again: every create failed while it did.
  pod_watchdog.sh   arm/run/status/disarm of that loop; state files beside the
                    ledger; a fired watchdog appends a delete event with
                    "by":"watchdog".
  pod_status.sh     read-only monitoring joined with the ledger; exits nonzero
                    when any pod's known age exceeds --max-age-hours (the
                    runaway-cost alarm).
  cost_report.sh    fully-OFFLINE cost table from the ledger create/delete
                    pairs; malformed lines are refused loudly by line number;
                    --billing shells `runpodctl billing pods` + `runpodctl
                    billing network-volume` as the account authority (the
                    bare `billing` group prints help and exits 0 on 2.11.0).

$0 DOCTRINE: everything here is offline. runpodctl is a PATH-shim FAKE written
into a tmp dir that records argv and serves canned outputs — the real CLI/API
is never touched (mutating verbs against the real API are forbidden).
"""
from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
TEARDOWN = SCRIPTS / "runpod" / "teardown_pod.sh"
PROVISION = SCRIPTS / "runpod" / "provision_pod.sh"
POD_STATUS = SCRIPTS / "runpod" / "pod_status.sh"
COST_REPORT = SCRIPTS / "runpod" / "cost_report.sh"
WATCHDOG = SCRIPTS / "runpod" / "pod_watchdog.sh"

GPU = "NVIDIA GeForce RTX 4090"
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# CLAUDE.md "Process safety (hard rule)": process-lifecycle tests run inside a
# container, never on the macOS host. Every test below that starts a detached
# watchdog loop (`arm`, every `--yes` create, the teardown disarm) or drives
# `run` with its sleep and timeout children carries this marker. On the pod
# (Linux) they execute through run_tests.sh (S0-11); on a Mac they need
# CAGE_PROCESS_TESTS=1 inside a container.
_LIFECYCLE = pytest.mark.skipif(
    sys.platform == "darwin" and os.environ.get("CAGE_PROCESS_TESTS") != "1",
    reason="process-lifecycle test: runs inside a container (CLAUDE.md process safety), never on the macOS host",
)

# `runpodctl gpu list` output as CLI v2 ACTUALLY emits it: a JSON document, with
# the price on a different line than the gpuId. The pre-2026-08-25 fixture here
# was a v1-style single table row, which let a real defect pass — the script's
# line-oriented price grep matched the `"gpuId": ...` line, found no number on
# it, and silently degraded every plan to price=unknown. Keep this shaped like
# the live CLI so the price path is exercised for real.
_GPU_LIST_JSON = json.dumps([
    {
        "available": True,
        "communityCloud": True,
        "communityPricePerHr": 0.34,
        "dataCenterAvailability": [{"dataCenterId": "US-IL-1", "stockStatus": "Low"}],
        "displayName": "RTX 4090",
        "gpuId": GPU,
        "memoryInGb": 24,
        "secureCloud": True,
        "securePricePerHr": 0.69,
        "stockStatus": "Low",
    },
    {
        "available": True,
        "communityCloud": False,
        "communityPricePerHr": None,
        "dataCenterAvailability": [{"dataCenterId": "EU-RO-1", "stockStatus": "Low"}],
        "displayName": "L40S",
        "gpuId": "NVIDIA L40S",
        "memoryInGb": 48,
        "secureCloud": True,
        "securePricePerHr": 0.99,
        "stockStatus": "Low",
    },
], indent=2) + "\n"

# Env vars that would leak state (a real ledger, a real pod SSH target, a
# non-interactive auto-yes) into these hermetic tests.
_LEAK_ENV = (
    "CAGE_POD_LEDGER", "CAGE_POD_SSH", "CAGE_ASSUME_YES", "CAGE_RUNPOD_REST",
    "RUNPOD_API_KEY", "CAGE_BACKUP_TARGET", "CAGE_SSH_OPTS", "CAGE_POD_IMAGE",
    "CAGE_WATCHDOG_TICK", "CAGE_WATCHDOG_RETRIES", "CAGE_WATCHDOG_CLI_TIMEOUT",
)

_FAKE = """#!/usr/bin/env bash
set -u
d="{d}"
printf '%s\\n' "$*" >> "$d/argv.log"
{fail_clause}case "$* " in
  "pod delete "*) [ -f "$d/hang_delete" ] && sleep 30; [ -f "$d/keep_listed" ] || printf '[]\\n' > "$d/pod_list.out"; exit 0 ;;
  "pod list --all "*) [ -f "$d/hang_list" ] && sleep 30; cat "$d/pod_list.out" ;;
  "pod get "*) printf 'id: %s\\nstatus: RUNNING\\n' "$3" ;;
  "network-volume list "*) cat "$d/nv_list.out" ;;
  "gpu list "*) cat "$d/gpu_list.out" ;;
  "pod create "*) cat "$d/pod_create.out" ;;
  "billing "*) cat "$d/billing.out" ;;
  "version "*) printf 'runpodctl 2.11.0 (fake)\\n' ;;
  *) printf 'fake-runpodctl: unhandled: %s\\n' "$*" >&2; exit 9 ;;
esac
"""


def _install_fake(
    tmp_path: Path,
    pod_list: str = "[]\n",
    nv_list: str = "[]\n",
    gpu_list: str = _GPU_LIST_JSON,
    pod_create: str = '{"id":"fakepod1234abcd"}\n',
    billing: str = "Account balance: $12.34\n",
    fail_all: bool = False,
) -> tuple[Path, Path]:
    """Write the PATH-shim fake runpodctl; returns (bin_dir, argv_log).

    fail_all=True makes every verb log argv then exit 1 — the offline stand-in
    for "CLI/network absent". (A genuinely-absent-CLI run is NOT exercised on
    purpose: the scripts' PATH guard appends /opt/homebrew/bin, which on a dev
    Mac would resurrect the REAL runpodctl — forbidden by the $0 doctrine. The
    shim always shadows it instead.)
    """
    d = tmp_path / "fakebin"
    d.mkdir(exist_ok=True)
    (d / "pod_list.out").write_text(pod_list, encoding="utf-8")
    (d / "nv_list.out").write_text(nv_list, encoding="utf-8")
    (d / "gpu_list.out").write_text(gpu_list, encoding="utf-8")
    (d / "pod_create.out").write_text(pod_create, encoding="utf-8")
    (d / "billing.out").write_text(billing, encoding="utf-8")
    fail_clause = '[ "$1" = version ] || exit 1\n' if fail_all else ""
    fake = d / "runpodctl"
    fake.write_text(_FAKE.format(d=d, fail_clause=fail_clause), encoding="utf-8")
    fake.chmod(0o755)
    log = d / "argv.log"
    log.write_text("", encoding="utf-8")
    return d, log


def _env(fake_bin: Path, **extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _LEAK_ENV}
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.update(extra)
    return env


def _env_nospawn(fake_bin: Path, tmp_path: Path) -> dict:
    """Env for HOST-runnable `--yes` tests: the ledger lands under tmp_path (never
    the real results/ops) and the watchdog refuses to arm (invalid tick), so even
    a regressed refusal could never write the real ledger or start a loop."""
    return _env(fake_bin, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl"), CAGE_WATCHDOG_TICK="not-a-number")


def _bash(script: str, env: dict, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=120,
        env=env, cwd=str(cwd) if cwd else None,
    )


def _argv_lines(log: Path) -> list[str]:
    return [l for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]


def _code_lines(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def _ts(delta_hours: float = 0.0) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return (now - datetime.timedelta(hours=delta_hours)).strftime(TS_FMT)


def _create_event(pod_id: str, ts: str, price: float | None = 0.5, **over: object) -> dict:
    e: dict = {
        "ts_utc": ts, "pod_id": pod_id, "name": "cage-s0", "gpu_id": GPU,
        "gpu_count": 1, "price_per_hour_usd": price,
        "terminate_after": "2026-08-21T00:00:00Z", "watchdog_pid": None,
        "purpose": "s0", "event": "create",
    }
    e.update(over)
    return e


def _write_ledger(path: Path, events: list[dict | str]) -> None:
    lines = [e if isinstance(e, str) else json.dumps(e) for e in events]
    path.write_text("".join(l + "\n" for l in lines), encoding="utf-8")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.1)
    raise AssertionError(f"watchdog pid {pid} still alive after {timeout}s")


def _rfc3339_in(seconds: float) -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=seconds)).strftime(TS_FMT)


def _is_our_loop(pid: int) -> bool:
    """True iff <pid> is a live `pod_watchdog.sh run` loop (the identity check
    pidfile_alive applies). A pidfile can name ANY pid (two tests write pid 1 on
    purpose), so no signal is ever sent without this check: `pkill -P 1` would
    SIGTERM every user process whose parent is launchd."""
    if pid < 2:
        return False
    proc = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "command="],
                          capture_output=True, text=True)
    return "pod_watchdog.sh run" in proc.stdout


@pytest.fixture(autouse=True)
def _reap_watchdogs(tmp_path: Path):
    """Every --yes create arms a REAL detached watchdog loop (against the fake
    CLI). Kill whatever a test left armed so no loop outlives the test run and
    ever calls a runpodctl outside the shim (the $0 doctrine). Identity-checked:
    a stale or fabricated pidfile is never signaled."""
    yield
    for pf in tmp_path.rglob("watchdog_*.pid"):
        try:
            pid = int(pf.read_text(encoding="utf-8").strip())
        except ValueError:
            continue
        if not _is_our_loop(pid):
            continue
        # Only the recorded, identity-checked pid; never a parent-pid or pattern
        # kill (CLAUDE.md process safety). The loop's orphaned tick `sleep` expires
        # on its own.
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


# ---------------------------------------------------------------------------
# teardown_pod.sh — CLI v2 migration pins
# ---------------------------------------------------------------------------

def test_teardown_source_pins_v2_rest_path_guard_and_ledger_hook() -> None:
    text = TEARDOWN.read_text(encoding="utf-8")
    code = _code_lines(text)
    assert 'RUNPOD_REST="${CAGE_RUNPOD_REST:-https://api.runpod.io/v2}"' in code, (
        "the REST fallback must default to the v2 base (v1 rest.runpod.io/v1 "
        "retires 2026-11-15) while keeping the CAGE_RUNPOD_REST override"
    )
    assert "rest.runpod.io/v1" not in code, "no code path may still target the v1 REST base"
    assert ('command -v runpodctl >/dev/null 2>&1 || '
            'PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"') in code, (
        "the PATH guard for non-interactive macOS shells (Homebrew dirs absent) is required"
    )
    # Ledger delete-event hook: closes provision_pod.sh's create event.
    assert "CAGE_POD_LEDGER" in code and "pod_ledger.jsonl" in code
    assert '"event":"delete"' in code.replace(" ", ""), (
        "a successful delete must append the {\"event\":\"delete\"} ledger line"
    )
    # $0 proof covers network volumes too (a surviving volume = clean-room violation).
    assert "runpodctl network-volume list" in code
    assert "STILL BILLING" in text, "fail-loud billing language must survive the migration"


def test_teardown_code_never_uses_v1_verbs() -> None:
    code = _code_lines(TEARDOWN.read_text(encoding="utf-8"))
    assert "runpodctl remove pod" not in code, "v1 verb `remove pod` no longer exists in runpodctl 2.x"
    assert "runpodctl get pod" not in code, "v1 verb `get pod` no longer exists in runpodctl 2.x"


@_LIFECYCLE
@pytest.mark.skipif(not (REPO_ROOT / ".venv" / "bin" / "python").exists(),
                    reason="repo venv required for the pull gate's ledger verification")
@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not on PATH")
def test_teardown_invokes_v2_delete_and_both_zero_listings(tmp_path: Path) -> None:
    """Full happy path against the PATH-shim fake: verified pull -> confirm ->
    `pod delete` -> `pod list --all` + `network-volume list` (both empty) ->
    TEARDOWN_COMPLETE, with the ledger delete event appended."""
    py = str(REPO_ROOT / ".venv" / "bin" / "python")
    remote = tmp_path / "remote_run"
    (remote / "cells").mkdir(parents=True)
    (remote / "manifest.json").write_text('{"run_id": "t"}', encoding="utf-8")
    (remote / "cells" / "a.csv").write_text("x\n1\n", encoding="utf-8")
    seal = (
        "import sys; from pathlib import Path\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from src.analysis.stats.ledger import hash_artifacts, write_ledger\n"
        f"run = Path({str(remote)!r})\n"
        "files = sorted(p for p in run.rglob('*') if p.is_file())\n"
        "write_ledger(hash_artifacts(files, base_dir=run), run / 'ledger.json')\n"
    )
    subprocess.run([py, "-c", seal], check=True, capture_output=True, text=True)

    fake, log = _install_fake(tmp_path)
    pod_ledger = tmp_path / "pod_ledger.jsonl"
    env = _env(fake, CAGE_ASSUME_YES="1", CAGE_POD_LEDGER=str(pod_ledger))
    # A live seatbelt guards the pod (as provision_pod.sh leaves it); teardown must disarm it.
    armed = _bash(f'bash "{WATCHDOG}" arm podtest99999999 2099-01-01T00:00:00Z', env=env)
    assert armed.returncode == 0, armed.stderr
    wd_pid = int((tmp_path / "watchdog_podtest99999999.pid").read_text(encoding="utf-8").strip())
    proc = _bash(
        f'bash "{TEARDOWN}" podtest99999999 "file://{remote}" "{tmp_path}/dest"', env=env,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "TEARDOWN_COMPLETE" in proc.stdout
    assert "watchdog DISARMED" in proc.stdout, "teardown must disarm the seatbelt after the delete"
    _wait_dead(wd_pid)
    assert not (tmp_path / "watchdog_podtest99999999.pid").exists()

    calls = _argv_lines(log)
    assert "pod delete podtest99999999" in calls
    assert "pod list --all" in calls
    assert "network-volume list" in calls
    assert calls.index("pod delete podtest99999999") < calls.index("pod list --all"), (
        "the $0 listings must come AFTER the delete"
    )
    # NEVER the v1 verbs, in any recorded invocation.
    for call in calls:
        assert not call.startswith("remove pod"), f"v1 verb invoked: {call}"
        assert not call.startswith("get pod"), f"v1 verb invoked: {call}"

    events = [json.loads(l) for l in pod_ledger.read_text(encoding="utf-8").splitlines()]
    assert events == [{
        "ts_utc": events[0]["ts_utc"], "pod_id": "podtest99999999", "event": "delete",
    }], f"unexpected ledger contents: {events}"
    datetime.datetime.strptime(events[0]["ts_utc"], TS_FMT)  # ISO-8601 Z or raise


# ---------------------------------------------------------------------------
# provision_pod.sh — plan-by-default, --yes gate, seatbelt, ledger
# ---------------------------------------------------------------------------

def test_provision_plan_mode_creates_nothing_and_exits_zero(tmp_path: Path) -> None:
    fake, log = _install_fake(tmp_path)
    proc = _bash(f'bash "{PROVISION}" --gpu-id "{GPU}" --hours 6', env=_env(fake))
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "PLAN ONLY" in proc.stdout and "--yes" in proc.stdout
    assert "--terminate-after 12h" in proc.stdout, "the plan must show the default seatbelt"
    assert "pod_watchdog.sh" in proc.stdout and "client-side" in proc.stdout, (
        "the plan must say the seatbelt is the client-side watchdog, not a RunPod feature"
    )
    creates = [c for c in _argv_lines(log) if c.startswith("pod create")]
    assert creates == [], f"PLAN mode must create NOTHING, but the fake saw: {creates}"
    assert not list(tmp_path.glob("watchdog_*.pid")), "PLAN mode must arm nothing"


def test_provision_plan_derives_price_from_gpu_list(tmp_path: Path) -> None:
    fake, log = _install_fake(tmp_path)
    proc = _bash(f'bash "{PROVISION}" --gpu-id "{GPU}" --hours 6', env=_env(fake))
    assert proc.returncode == 0
    assert "$0.69/h" in proc.stdout, "price must be parsed out of the (fake) `runpodctl gpu list` JSON"
    assert re.search(r"estimated total\s+: \$4\.14", proc.stdout), "6h x $0.69 x 1 GPU = $4.14"
    assert any(c.startswith("gpu list") for c in _argv_lines(log))


def test_provision_plan_price_follows_cloud_type(tmp_path: Path) -> None:
    """COMMUNITY must quote the community price, not the secure one.

    The two differ by ~2x, so picking the wrong field misstates the cost plan
    the owner approves at the run-approval gate.
    """
    fake, _ = _install_fake(tmp_path)
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --cloud-type COMMUNITY --hours 6',
        env=_env(fake),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "$0.34/h" in proc.stdout, "COMMUNITY must quote communityPricePerHr"
    assert "$0.69/h" not in proc.stdout, "the SECURE price must not leak into a COMMUNITY plan"
    assert re.search(r"estimated total\s+: \$2\.04", proc.stdout), "6h x $0.34 x 1 GPU = $2.04"


def test_provision_plan_degrades_to_null_price_when_cli_fails(tmp_path: Path) -> None:
    fake, log = _install_fake(tmp_path, fail_all=True)
    proc = _bash(f'bash "{PROVISION}" --gpu-id "{GPU}" --hours 6', env=_env(fake))
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "unknown" in proc.stdout and "--price-per-hour" in proc.stdout, (
        "with no derivable price the plan must degrade to unknown + a note"
    )
    assert not any(c.startswith("pod create") for c in _argv_lines(log))


@pytest.mark.parametrize("bad", ["twelve hours", "12", "12hh", "0h", "2026-08-26", "-5h",
                                 "2099-02-30T00:00:00Z"])
def test_provision_refuses_unparseable_seatbelt(tmp_path: Path, bad: str) -> None:
    """An unresolvable seatbelt must fail LOUDLY, before anything is created.

    Failing closed here is what keeps a malformed deadline from producing a pod
    with no seatbelt (the watchdog could only refuse to arm AFTER the create).
    The calendar-invalid instant (Feb 30) is the case BSD date normalizes silently;
    the pre-create check round-trips it through the watchdog's `check`.
    """
    fake, log = _install_fake(tmp_path)
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --terminate-after "{bad}" --yes',
        env=_env_nospawn(fake, tmp_path),
    )
    assert proc.returncode != 0, f"{bad!r} must be refused; stdout:\n{proc.stdout}"
    assert "--terminate-after" in proc.stderr
    assert not any(c.startswith("pod create") for c in _argv_lines(log)), (
        f"nothing may be created when the seatbelt is unresolvable ({bad!r})"
    )
    assert not (tmp_path / "l.jsonl").exists() and not list(tmp_path.glob("watchdog_*.pid"))


@_LIFECYCLE
def test_provision_accepts_absolute_seatbelt_verbatim(tmp_path: Path) -> None:
    """An operator may pass the RFC3339 form; it reaches the WATCHDOG unchanged
    (deadline file + ledger) and never the CLI, which has no such flag."""
    fake, log = _install_fake(tmp_path)
    ledger = tmp_path / "l.jsonl"
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --terminate-after 2099-01-01T00:00:00Z --yes',
        env=_env(fake, CAGE_POD_LEDGER=str(ledger)),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    creates = [c for c in _argv_lines(log) if c.startswith("pod create")]
    assert len(creates) == 1
    assert "--terminate-after" not in creates[0], "runpodctl has no such flag (usage_error, 2026-09-26)"
    assert (tmp_path / "watchdog_fakepod1234abcd.deadline").read_text(
        encoding="utf-8").strip() == "2099-01-01T00:00:00Z"
    e = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert e["terminate_after"] == "2099-01-01T00:00:00Z"
    assert isinstance(e["watchdog_pid"], int) and _alive(e["watchdog_pid"])


def test_provision_requires_gpu_id(tmp_path: Path) -> None:
    fake, _ = _install_fake(tmp_path)
    proc = _bash(f'bash "{PROVISION}"', env=_env(fake))
    assert proc.returncode == 2
    assert "usage:" in proc.stderr and "--gpu-id" in proc.stderr


def test_provision_refuses_past_absolute_deadline_before_create(tmp_path: Path) -> None:
    """A past RFC3339 --terminate-after passes the format check but can never be
    armed; the refusal must come BEFORE the cost-starting create, or the pod would
    come up unguarded with exit 0."""
    fake, log = _install_fake(tmp_path)
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --terminate-after 2020-01-01T00:00:00Z --yes',
        env=_env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl")),
    )
    assert proc.returncode != 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "not in the future" in proc.stderr
    assert not any(c.startswith("pod create") for c in _argv_lines(log)), "nothing may be created"
    assert not (tmp_path / "l.jsonl").exists()


@_LIFECYCLE
def test_provision_yes_arms_default_seatbelt_and_writes_ledger(tmp_path: Path) -> None:
    fake, log = _install_fake(tmp_path)
    ledger = tmp_path / "pod_ledger.jsonl"
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --name cage-s0 --purpose s0-gate '
        f'--price-per-hour 0.86 --hours 6 --yes',
        env=_env(fake, CAGE_POD_LEDGER=str(ledger)),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    creates = [c for c in _argv_lines(log) if c.startswith("pod create")]
    assert len(creates) == 1
    # The seatbelt is CLIENT-SIDE (pod_watchdog.sh): runpodctl has no
    # --terminate-after (2.11.0 and 2.14.0 answer usage_error; live 2026-09-26),
    # and while this script sent it EVERY create failed. Pin: the flag never
    # reaches the CLI; the resolved instant lands ~12h out in the plan print,
    # the deadline file and the ledger; a live watchdog guards the pod.
    assert "--terminate-after" not in creates[0], (
        f"--terminate-after must NEVER reach runpodctl (no such flag): {creates[0]}"
    )
    m = re.search(r"deletes the pod at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", proc.stdout)
    assert m, f"the plan must print the watchdog deadline instant:\n{proc.stdout}"
    passed = m.group(1)
    deadline = datetime.datetime.strptime(passed, TS_FMT).replace(tzinfo=datetime.timezone.utc)
    ahead = (deadline - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    assert 11.5 * 3600 < ahead < 12.5 * 3600, (
        f"the default seatbelt must land ~12h out, got {ahead / 3600:.2f}h ({passed})"
    )
    assert "watchdog ARMED" in proc.stdout and "client-side" in proc.stdout
    pidfile = tmp_path / "watchdog_fakepod1234abcd.pid"
    assert pidfile.is_file(), "arming must leave a pidfile beside the ledger"
    wd_pid = int(pidfile.read_text(encoding="utf-8").strip())
    assert _alive(wd_pid), "the armed watchdog loop must be alive"
    assert (tmp_path / "watchdog_fakepod1234abcd.deadline").read_text(
        encoding="utf-8").strip() == passed
    assert "fakepod1234abcd" in proc.stdout and "teardown_pod.sh" in proc.stdout \
        and "setup_runpod.sh" in proc.stdout, "the create must print the id + next steps"

    # Siting flags absent -> they must NOT reach the CLI (runpodctl would take an
    # empty pin as a real constraint) and must land as null in the ledger.
    assert "--data-center-ids" not in creates[0]
    assert "--network-volume-id" not in creates[0]

    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    e = json.loads(lines[0])
    assert set(e) == {"ts_utc", "pod_id", "name", "gpu_id", "gpu_count",
                      "price_per_hour_usd", "terminate_after", "watchdog_pid",
                      "data_center_ids", "network_volume_id", "purpose", "event"}
    assert e["event"] == "create" and e["pod_id"] == "fakepod1234abcd"
    assert e["data_center_ids"] is None and e["network_volume_id"] is None, (
        "siting fields must be recorded as null when the flags were not given "
        "(absent data is null, never a fabricated value)"
    )
    assert e["name"] == "cage-s0" and e["gpu_id"] == GPU and e["purpose"] == "s0-gate"
    assert isinstance(e["gpu_count"], int) and e["gpu_count"] == 1
    assert isinstance(e["price_per_hour_usd"], float) and e["price_per_hour_usd"] == 0.86
    # The ledger records the resolved instant (the watchdog's deadline) and the
    # watchdog's pid, so readers can compare against a wall clock and a process table.
    assert e["terminate_after"] == passed
    datetime.datetime.strptime(e["terminate_after"], TS_FMT)
    datetime.datetime.strptime(e["ts_utc"], TS_FMT)
    assert e["watchdog_pid"] == wd_pid


def test_provision_no_terminate_after_warns_loud(tmp_path: Path) -> None:
    fake, log = _install_fake(tmp_path)
    ledger = tmp_path / "pod_ledger.jsonl"
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --no-terminate-after --yes',
        env=_env(fake, CAGE_POD_LEDGER=str(ledger)),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "WARNING" in proc.stderr and "SEATBELT DISABLED" in proc.stderr, (
        "disabling the seatbelt must be announced LOUDLY"
    )
    creates = [c for c in _argv_lines(log) if c.startswith("pod create")]
    assert len(creates) == 1 and "--terminate-after" not in creates[0]
    e = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert e["terminate_after"] is None and e["watchdog_pid"] is None
    assert "no watchdog will be armed" in proc.stderr
    assert not list(tmp_path.glob("watchdog_*.pid")), "no watchdog may be armed when the seatbelt is disabled"


def test_provision_plan_shows_siting_pins(tmp_path: Path) -> None:
    """The siting pins must be visible in the PLAN block the owner approves.

    A network volume attaches ONLY at create time and ONLY in its own
    datacenter, so where the pod lands IS part of the GO decision — a plan
    that hides the pins would get an approval for a different pod.
    """
    fake, log = _install_fake(tmp_path)
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --data-center-ids US-IL-1,EU-RO-1 '
        f'--network-volume-id nvol1234abcd --hours 6',
        env=_env(fake),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "US-IL-1,EU-RO-1" in proc.stdout, "the datacenter pin must appear in the plan"
    assert "nvol1234abcd" in proc.stdout, "the network-volume pin must appear in the plan"
    assert not any(c.startswith("pod create") for c in _argv_lines(log)), (
        "PLAN mode must still create NOTHING when siting flags are given"
    )


@_LIFECYCLE
def test_provision_create_passes_siting_flags_and_ledgers_them(tmp_path: Path) -> None:
    """--yes must forward both siting flags verbatim to `pod create` and record
    them in the create event, so the ledger is the audit trail of where the pod
    (and its create-time-only volume attachment) were sited."""
    fake, log = _install_fake(tmp_path)
    ledger = tmp_path / "pod_ledger.jsonl"
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --data-center-ids US-IL-1 '
        f'--network-volume-id nvol1234abcd --yes',
        env=_env(fake, CAGE_POD_LEDGER=str(ledger)),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    creates = [c for c in _argv_lines(log) if c.startswith("pod create")]
    assert len(creates) == 1
    assert "--data-center-ids US-IL-1" in creates[0]
    assert "--network-volume-id nvol1234abcd" in creates[0]
    e = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert e["data_center_ids"] == "US-IL-1"
    assert e["network_volume_id"] == "nvol1234abcd"


@pytest.mark.parametrize("bad", ["", "US-IL-1,", ",US-IL-1", "US-IL-1,,EU-RO-1",
                                 "US_IL_1", "US-IL-1, EU-RO-1"])
def test_provision_refuses_bad_data_center_ids(tmp_path: Path, bad: str) -> None:
    """A malformed --data-center-ids must be refused BEFORE any create.

    Siting is a create-time-only lever: a typo'd pin that reached the CLI could
    site the pod away from its network volume, unfixably. Empty values count —
    an explicitly-given empty pin must refuse, never silently mean "no pin".
    """
    fake, log = _install_fake(tmp_path)
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --data-center-ids "{bad}" --yes',
        env=_env_nospawn(fake, tmp_path),
    )
    assert proc.returncode != 0, f"{bad!r} must be refused; stdout:\n{proc.stdout}"
    assert "--data-center-ids" in proc.stderr
    assert not any(c.startswith("pod create") for c in _argv_lines(log)), (
        f"nothing may be created on a refused --data-center-ids ({bad!r})"
    )
    assert not list(tmp_path.glob("watchdog_*.pid"))


def test_provision_default_image_is_a_published_tag(tmp_path: Path) -> None:
    """The default image must be a tag that exists on Docker Hub.

    The pre-2026-09-25 default (…cudnn-devel-ubuntu24.04) never existed (the docs
    example is ubuntu22.04), so every S0-1 create would have failed or stalled.
    A hermetic test cannot reach Docker Hub; it pins the literal so any future
    edit is reviewed against the registry (verified 2026-09-25).
    """
    fake, _ = _install_fake(tmp_path)
    proc = _bash(f'bash "{PROVISION}" --gpu-id "{GPU}" --hours 6', env=_env(fake))
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404" in proc.stdout


@_LIFECYCLE
def test_provision_network_volume_zeroes_the_volume_disk(tmp_path: Path) -> None:
    """A network volume REPLACES the volume disk at /workspace (RunPod docs,
    pods/storage/types), and runpodctl omits volumeInGb when it is 0, so the
    create must send --volume-in-gb 0 beside --network-volume-id and the PLAN
    print must say the volume disk is replaced; without a volume the 100 GB
    default still rides the create."""
    fake, log = _install_fake(tmp_path)
    ledger = tmp_path / "pod_ledger.jsonl"
    plan = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --network-volume-id nvol1234abcd --hours 6',
        env=_env(fake),
    )
    assert plan.returncode == 0, f"stdout:\n{plan.stdout}\nstderr:\n{plan.stderr}"
    assert "replaces the volume disk" in plan.stdout and "--volume-in-gb 0" in plan.stdout
    assert "/workspace volume: 100 GB" not in plan.stdout
    env = _env(fake, CAGE_POD_LEDGER=str(ledger))
    proc = _bash(f'bash "{PROVISION}" --gpu-id "{GPU}" --network-volume-id nvol1234abcd --yes', env=env)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "watchdog ARMED" in proc.stdout
    creates = [c for c in _argv_lines(log) if c.startswith("pod create")]
    assert len(creates) == 1 and "--volume-in-gb 0 " in creates[0] + " "
    assert "--volume-mount-path /workspace" in creates[0], "the mount path is the network volume's"
    # the fake returns the same pod id for every create: disarm before the second one
    assert _bash(f'bash "{WATCHDOG}" disarm fakepod1234abcd', env=env).returncode == 0
    proc2 = _bash(f'bash "{PROVISION}" --gpu-id "{GPU}" --yes', env=env)
    assert proc2.returncode == 0, f"stdout:\n{proc2.stdout}\nstderr:\n{proc2.stderr}"
    assert "watchdog ARMED" in proc2.stdout and "SEATBELT NOT ARMED" not in proc2.stderr
    creates = [c for c in _argv_lines(log) if c.startswith("pod create")]
    assert len(creates) == 2 and "--volume-in-gb 100 " in creates[1] + " "


def test_provision_refuses_volume_gb_beside_network_volume(tmp_path: Path) -> None:
    """An explicit --volume-gb next to --network-volume-id is refused, never
    silently zeroed: the network volume replaces the volume disk, and the script's
    rule for create-time-only levers is fail-closed."""
    fake, log = _install_fake(tmp_path)
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --volume-gb 200 --network-volume-id nvol1234abcd --yes',
        env=_env_nospawn(fake, tmp_path),
    )
    assert proc.returncode != 0, f"must refuse; stdout:\n{proc.stdout}"
    assert "--volume-gb" in proc.stderr and "--network-volume-id" in proc.stderr
    assert not any(c.startswith("pod create") for c in _argv_lines(log))
    assert not list(tmp_path.glob("watchdog_*.pid"))


def test_provision_refuses_empty_network_volume_id(tmp_path: Path) -> None:
    fake, log = _install_fake(tmp_path)
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --network-volume-id "" --yes',
        env=_env_nospawn(fake, tmp_path),
    )
    assert proc.returncode != 0, "an explicit empty volume id must refuse, not mean 'no volume'"
    assert "--network-volume-id" in proc.stderr
    assert not any(c.startswith("pod create") for c in _argv_lines(log))
    assert not list(tmp_path.glob("watchdog_*.pid"))


def test_provision_arm_failure_is_loud_but_ledger_still_lands(tmp_path: Path) -> None:
    """If the watchdog cannot be armed the pod IS billing: the create event must
    still land (watchdog_pid null) and the operator must be told to arm by hand."""
    fake, _ = _install_fake(tmp_path)
    ledger = tmp_path / "pod_ledger.jsonl"
    proc = _bash(
        f'bash "{PROVISION}" --gpu-id "{GPU}" --yes',
        env=_env(fake, CAGE_POD_LEDGER=str(ledger), CAGE_WATCHDOG_TICK="not-a-number"),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SEATBELT NOT ARMED" in proc.stderr and "pod_watchdog.sh arm fakepod1234abcd" in proc.stderr
    e = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert e["event"] == "create" and e["watchdog_pid"] is None
    assert e["terminate_after"] is not None, "the intended deadline is still recorded"
    assert not list(tmp_path.glob("watchdog_*.pid"))


def test_provision_create_argv_never_carries_terminate_after(tmp_path: Path) -> None:
    """Regression pin for the 2026-09-26 blocker: the phantom CLI flag made EVERY
    create fail with usage_error. Whatever seatbelt form the operator passes, the
    `pod create` argv must not carry it. Runs on the host: the watchdog is made
    to refuse arming (invalid tick), so no detached process is ever started."""
    fake, log = _install_fake(tmp_path)
    for i, extra in enumerate(["", "--terminate-after 90m", "--terminate-after 2099-01-01T00:00:00Z"]):
        env = _env(fake, CAGE_POD_LEDGER=str(tmp_path / f"l{i}.jsonl"), CAGE_WATCHDOG_TICK="not-a-number")
        proc = _bash(f'bash "{PROVISION}" --gpu-id "{GPU}" {extra} --yes', env=env)
        assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        assert "SEATBELT NOT ARMED" in proc.stderr
    creates = [c for c in _argv_lines(log) if c.startswith("pod create")]
    assert len(creates) == 3
    assert all("--terminate-after" not in c for c in creates), creates
    assert not list(tmp_path.glob("watchdog_*.pid")), "no loop may have been started on the host"


# ---------------------------------------------------------------------------
# pod_watchdog.sh: the client-side seatbelt (fire, no duplicate, fail loud,
# arm/status/duplicate/disarm)
# ---------------------------------------------------------------------------

@_LIFECYCLE
@pytest.mark.parametrize("listing", [
    '[{"id":"podaaaa1111bbbb"}]\n',                 # bare list, the CLI contract
    '{"pods":[{"id":"podaaaa1111bbbb"}]}\n',        # wrapped under "pods"
    '{"data":[{"id":"podaaaa1111bbbb"}]}\n',        # wrapped under "data"
], ids=["bare-list", "pods-wrapper", "data-wrapper"])
def test_watchdog_run_fires_deletes_and_ledgers_by_watchdog(tmp_path: Path, listing: str) -> None:
    fake, log = _install_fake(tmp_path, pod_list=listing)
    ledger = tmp_path / "pod_ledger.jsonl"
    proc = _bash(
        f'bash "{WATCHDOG}" run podaaaa1111bbbb {_rfc3339_in(1)}',
        env=_env(fake, CAGE_POD_LEDGER=str(ledger), CAGE_WATCHDOG_TICK="1", CAGE_WATCHDOG_RETRIES="3"),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    calls = _argv_lines(log)
    assert "pod delete podaaaa1111bbbb" in calls and "pod list --all" in calls
    assert calls.index("pod delete podaaaa1111bbbb") < calls.index("pod list --all"), (
        "the listing check must follow the delete"
    )
    events = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines()]
    assert events == [{"ts_utc": events[0]["ts_utc"], "pod_id": "podaaaa1111bbbb",
                       "event": "delete", "by": "watchdog"}], events
    datetime.datetime.strptime(events[0]["ts_utc"], TS_FMT)
    assert "DONE" in proc.stdout


@_LIFECYCLE
def test_watchdog_run_appends_no_duplicate_when_ledger_already_closed(tmp_path: Path) -> None:
    fake, _ = _install_fake(tmp_path, pod_list="[]\n")
    ledger = tmp_path / "pod_ledger.jsonl"
    _write_ledger(ledger, [{"ts_utc": "2026-08-20T01:00:00Z", "pod_id": "podaaaa1111bbbb",
                            "event": "delete"}])
    proc = _bash(
        f'bash "{WATCHDOG}" run podaaaa1111bbbb 2020-01-01T00:00:00Z',
        env=_env(fake, CAGE_POD_LEDGER=str(ledger), CAGE_WATCHDOG_TICK="1"),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1, "no duplicate delete event"
    assert "already closes" in proc.stdout


@_LIFECYCLE
def test_watchdog_run_fails_loud_when_pod_survives(tmp_path: Path) -> None:
    fake, log = _install_fake(tmp_path, pod_list='[{"id":"podaaaa1111bbbb"}]\n')
    (fake / "keep_listed").write_text("", encoding="utf-8")  # the fake keeps listing the pod
    proc = _bash(
        f'bash "{WATCHDOG}" run podaaaa1111bbbb 2020-01-01T00:00:00Z',
        env=_env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl"),
                 CAGE_WATCHDOG_TICK="1", CAGE_WATCHDOG_RETRIES="2"),
    )
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "FAILED" in proc.stdout and "STILL BE BILLING" in proc.stdout
    assert sum(1 for c in _argv_lines(log) if c == "pod delete podaaaa1111bbbb") == 2
    assert not (tmp_path / "l.jsonl").exists(), (
        "no delete event may be ledgered for a pod that is still listed"
    )


@_LIFECYCLE
@pytest.mark.parametrize("listing", [
    '{"error":"service unavailable","code":"server_error"}\n',   # an error object
    '',                                                           # empty output
    'ID              NAME  STATUS\npodaaaa1111bbbb x     RUNNING\n',  # a table, not JSON
    '[{"name":"x"}]\n',                                           # a list whose entries carry no id
], ids=["error-object", "empty", "table-text", "no-id-key"])
def test_watchdog_run_treats_unparseable_listing_as_unknown_never_gone(tmp_path: Path, listing: str) -> None:
    """Anything but a parsed JSON list of pods from `pod list --all` is UNKNOWN,
    never "the pod is gone": no DONE, no ledger delete event, loud FAILED after
    the retries (the pod may still bill)."""
    fake, log = _install_fake(tmp_path, pod_list=listing)
    (fake / "keep_listed").write_text("", encoding="utf-8")  # the fake keeps returning that shape
    proc = _bash(
        f'bash "{WATCHDOG}" run podaaaa1111bbbb 2020-01-01T00:00:00Z',
        env=_env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl"),
                 CAGE_WATCHDOG_TICK="1", CAGE_WATCHDOG_RETRIES="2"),
    )
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "FAILED" in proc.stdout and "DONE" not in proc.stdout
    assert "listing rc=2" in proc.stdout and "UNKNOWN" in proc.stdout, (
        "the unparseable listing must be reported as unknown (rc 2), never as 'still listed'"
    )
    assert not (tmp_path / "l.jsonl").exists(), "no delete event may be ledgered on an unknown listing"


def test_watchdog_run_exits_superseded_when_pidfile_names_another_pid(tmp_path: Path) -> None:
    """A pidfile naming another pid means this loop was re-armed or is a stale
    twin: exit without firing. Host-safe: the check runs before the first tick, so
    no sleep child and no CLI call is ever started."""
    fake, log = _install_fake(tmp_path)
    (tmp_path / "watchdog_podaaaa1111bbbb.pid").write_text("1\n", encoding="utf-8")
    proc = _bash(
        f'bash "{WATCHDOG}" run podaaaa1111bbbb 2099-01-01T00:00:00Z',
        env=_env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl"), CAGE_WATCHDOG_TICK="1"),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "superseded" in proc.stdout
    assert _argv_lines(log) == [], "exiting as superseded must not touch the CLI"
    assert not (tmp_path / "l.jsonl").exists()


def test_watchdog_disarm_never_signals_a_pid_that_is_not_our_loop(tmp_path: Path) -> None:
    """The owner rule made concrete: a pidfile naming pid 1 (launchd) is a STALE
    record, removed without any signal; pid 1 is still alive afterwards."""
    fake, _ = _install_fake(tmp_path)
    pf = tmp_path / "watchdog_podaaaa1111bbbb.pid"
    pf.write_text("1\n", encoding="utf-8")
    proc = _bash(f'bash "{WATCHDOG}" disarm podaaaa1111bbbb',
                 env=_env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl")))
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "stale pidfile removed" in proc.stderr and "DISARMED" not in proc.stdout
    assert not pf.exists()
    assert _alive(1), "pid 1 must be untouched (PermissionError on kill -0 counts as alive)"


@pytest.mark.parametrize("deadline, code, needle", [
    ("2099-01-01T00:00:00Z", 0, "OK "),
    ("2020-01-01T00:00:00Z", 1, "not in the future"),
    ("2099-02-30T00:00:00Z", 1, "deadline"),
    ("12h", 1, "RFC3339"),
])
def test_watchdog_check_validates_without_arming(tmp_path: Path, deadline: str, code: int, needle: str) -> None:
    """`check` is the validator provision_pod.sh calls before the plan print: format,
    calendar round-trip, future. It never spawns anything."""
    fake, log = _install_fake(tmp_path)
    proc = _bash(f'bash "{WATCHDOG}" check {deadline}', env=_env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl")))
    assert proc.returncode == code, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert needle in (proc.stdout if code == 0 else proc.stderr)
    assert _argv_lines(log) == [] and not list(tmp_path.glob("watchdog_*.pid"))


@_LIFECYCLE
def test_watchdog_run_exits_when_its_pidfile_is_removed(tmp_path: Path) -> None:
    """Removing the pidfile IS a disarm: a loop whose state was wiped (pytest
    pruning a tmp dir, an operator rm) exits without firing, so a leaked loop can
    never reach a runpodctl outside the shim hours later."""
    fake, log = _install_fake(tmp_path)
    env = _env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl"), CAGE_WATCHDOG_TICK="1")
    proc = _bash(f'bash "{WATCHDOG}" arm podaaaa1111bbbb 2099-01-01T00:00:00Z', env=env)
    assert proc.returncode == 0, proc.stderr
    pid = int((tmp_path / "watchdog_podaaaa1111bbbb.pid").read_text(encoding="utf-8").strip())
    time.sleep(2.5)  # the loop has seen its pidfile at least once
    (tmp_path / "watchdog_podaaaa1111bbbb.pid").unlink()
    _wait_dead(pid, timeout=8.0)
    assert not any(c.startswith("pod delete") for c in _argv_lines(log)), "exiting is not firing"
    assert not (tmp_path / "l.jsonl").exists()


@_LIFECYCLE
@pytest.mark.parametrize("knob", ["hang_delete", "hang_list"])
def test_watchdog_run_bounds_a_hung_cli_call(tmp_path: Path, knob: str) -> None:
    """A hung runpodctl at the deadline must not stall the seatbelt forever: each
    CLI call (the delete, and the listing captured through $(...)) is bounded by
    CAGE_WATCHDOG_CLI_TIMEOUT and counts as one failed attempt."""
    fake, log = _install_fake(tmp_path, pod_list='[{"id":"podaaaa1111bbbb"}]\n')
    (fake / knob).write_text("", encoding="utf-8")  # that CLI verb sleeps 30 s
    (fake / "keep_listed").write_text("", encoding="utf-8")
    t0 = time.monotonic()
    proc = _bash(
        f'bash "{WATCHDOG}" run podaaaa1111bbbb 2020-01-01T00:00:00Z',
        env=_env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl"), CAGE_WATCHDOG_TICK="1",
                 CAGE_WATCHDOG_RETRIES="2", CAGE_WATCHDOG_CLI_TIMEOUT="1"),
    )
    elapsed = time.monotonic() - t0
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert elapsed < 20, f"a 1 s CLI timeout must bound the run, took {elapsed:.1f}s"
    assert "FAILED" in proc.stdout


@_LIFECYCLE
def test_watchdog_term_during_a_hung_delete_stops_the_cli_killer(tmp_path: Path) -> None:
    """Disarm can land while the loop is inside a CLI call. The exit trap must then
    stop the CLI-timeout killer subshell (a recorded pid), so nothing the loop
    started can signal a stale pid after the loop is gone."""
    fake, _ = _install_fake(tmp_path, pod_list='[{"id":"podaaaa1111bbbb"}]\n')
    (fake / "hang_delete").write_text("", encoding="utf-8")
    env = _env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl"), CAGE_WATCHDOG_TICK="1",
               CAGE_WATCHDOG_RETRIES="1", CAGE_WATCHDOG_CLI_TIMEOUT="20")
    loop = subprocess.Popen(["bash", str(WATCHDOG), "run", "podaaaa1111bbbb", "2020-01-01T00:00:00Z"],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(1.5)                 # inside the hung `pod delete`, killer armed
        loop.send_signal(signal.SIGTERM)   # our own child, by its recorded pid
        out, _ = loop.communicate(timeout=10)
    finally:
        if loop.poll() is None:
            loop.kill()
    assert loop.returncode == 130, f"rc={loop.returncode}\n{out}"
    m = re.search(r"stopped the CLI killer \(pid (\d+)\)", out)
    assert m, f"the exit trap must report the killer it stopped:\n{out}"
    _wait_dead(int(m.group(1)), timeout=5.0)
    assert not (tmp_path / "l.jsonl").exists(), "a TERM'd loop must not ledger a delete"


@pytest.mark.parametrize("args, code, needle", [
    ("arm podaaaa1111bbbb 2020-01-01T00:00:00Z", 1, "not in the future"),
    ("arm BAD_ID 2099-01-01T00:00:00Z", 1, "pod_id must match"),
    ("arm podaaaa1111bbbb 12h", 1, "RFC3339"),
    ("arm podaaaa1111bbbb 2099-02-30T00:00:00Z", 1, "deadline"),
    ("bogus", 2, "usage:"),
    ("arm podaaaa1111bbbb", 2, "usage:"),
])
def test_watchdog_refuses_bad_input(tmp_path: Path, args: str, code: int, needle: str) -> None:
    fake, _ = _install_fake(tmp_path)
    proc = _bash(f'bash "{WATCHDOG}" {args}', env=_env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl")))
    assert proc.returncode == code, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert needle in proc.stderr
    assert not list(tmp_path.glob("watchdog_*.pid")), "a refused arm must leave no state"


@_LIFECYCLE
def test_watchdog_arm_status_duplicate_disarm_roundtrip(tmp_path: Path) -> None:
    fake, _ = _install_fake(tmp_path)
    env = _env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl"))
    proc = _bash(f'bash "{WATCHDOG}" arm podaaaa1111bbbb 2099-01-01T00:00:00Z', env=env)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    m = re.search(r"^WATCHDOG_PID=(\d+)$", proc.stdout, re.M)
    assert m and "watchdog ARMED" in proc.stdout
    pid = int(m.group(1))
    assert _alive(pid)
    assert (tmp_path / "watchdog_podaaaa1111bbbb.pid").read_text(encoding="utf-8").strip() == str(pid)
    status = _bash(f'bash "{WATCHDOG}" status', env=env)
    assert status.returncode == 0 and "watchdog=ALIVE" in status.stdout \
        and "2099-01-01T00:00:00Z" in status.stdout
    dup = _bash(f'bash "{WATCHDOG}" arm podaaaa1111bbbb 2099-01-01T00:00:00Z', env=env)
    assert dup.returncode == 1 and "already guards" in dup.stderr, "one watchdog per pod"
    dis = _bash(f'bash "{WATCHDOG}" disarm podaaaa1111bbbb', env=env)
    assert dis.returncode == 0 and "DISARMED" in dis.stdout
    _wait_dead(pid)
    assert not (tmp_path / "watchdog_podaaaa1111bbbb.pid").exists()
    after = _bash(f'bash "{WATCHDOG}" status', env=env)
    assert "no watchdog state" in after.stdout


def test_watchdog_status_reports_dead_for_stale_pidfile(tmp_path: Path) -> None:
    fake, _ = _install_fake(tmp_path)
    (tmp_path / "watchdog_podaaaa1111bbbb.pid").write_text("1\n", encoding="utf-8")  # pid 1 is not our loop
    proc = _bash(f'bash "{WATCHDOG}" status podaaaa1111bbbb',
                 env=_env(fake, CAGE_POD_LEDGER=str(tmp_path / "l.jsonl")))
    assert proc.returncode == 0 and "watchdog=DEAD" in proc.stdout and "UNGUARDED" in proc.stdout


# ---------------------------------------------------------------------------
# pod_status.sh — ledger join, spend estimate, runaway-cost alarm
# ---------------------------------------------------------------------------

def test_pod_status_reports_uptime_and_spend(tmp_path: Path) -> None:
    fake, _ = _install_fake(
        tmp_path,
        pod_list="ID              NAME     GPU          STATUS\n"
                 "podaaaa1111bbbb cage-s0  NVIDIA-L40S  RUNNING\n",
    )
    ledger = tmp_path / "pod_ledger.jsonl"
    _write_ledger(ledger, [_create_event("podaaaa1111bbbb", _ts(2.0), price=0.5)])
    proc = _bash(f'bash "{POD_STATUS}"', env=_env(fake, CAGE_POD_LEDGER=str(ledger)))
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert re.search(r"podaaaa1111bbbb\s+cage-s0\s+2\.0\dh", proc.stdout), (
        "uptime must be now − the ledger create ts (~2h)"
    )
    assert re.search(r"\$1\.0[01]", proc.stdout), "spend = 2h x $0.50/h x 1 GPU = ~$1.00"
    assert "watchdog: NONE" in proc.stdout, "an unguarded live pod must be reported as such"


@_LIFECYCLE
def test_pod_status_shows_watchdog_alive_and_dead(tmp_path: Path) -> None:
    fake, _ = _install_fake(tmp_path, pod_list='[{"id":"podaaaa1111bbbb"},{"id":"podcccc3333dddd"}]\n')
    ledger = tmp_path / "pod_ledger.jsonl"
    _write_ledger(ledger, [_create_event("podaaaa1111bbbb", _ts(1.0)),
                           _create_event("podcccc3333dddd", _ts(1.0))])
    env = _env(fake, CAGE_POD_LEDGER=str(ledger))
    armed = _bash(f'bash "{WATCHDOG}" arm podaaaa1111bbbb 2099-01-01T00:00:00Z', env=env)
    assert armed.returncode == 0, armed.stderr
    (tmp_path / "watchdog_podcccc3333dddd.pid").write_text("1\n", encoding="utf-8")  # stale: pid 1 is not ours
    proc = _bash(f'bash "{POD_STATUS}"', env=env)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    lines = proc.stdout.splitlines()

    def wd_line_after(pod: str) -> str:
        i = next(k for k, l in enumerate(lines) if f"pod get {pod}" in l)
        return next(l for l in lines[i:] if "watchdog:" in l)

    assert "ALIVE" in wd_line_after("podaaaa1111bbbb") and "2099-01-01T00:00:00Z" in wd_line_after("podaaaa1111bbbb")
    assert "DEAD" in wd_line_after("podcccc3333dddd") and "UNGUARDED" in wd_line_after("podcccc3333dddd")


def test_pod_status_alarm_exit_on_max_age(tmp_path: Path) -> None:
    fake, _ = _install_fake(
        tmp_path, pod_list="podaaaa1111bbbb cage-s0  NVIDIA-L40S  RUNNING\n",
    )
    ledger = tmp_path / "pod_ledger.jsonl"
    _write_ledger(ledger, [_create_event("podaaaa1111bbbb", _ts(2.0), price=0.5)])
    proc = _bash(
        f'bash "{POD_STATUS}" --max-age-hours 1',
        env=_env(fake, CAGE_POD_LEDGER=str(ledger)),
    )
    assert proc.returncode != 0, "a pod older than --max-age-hours must exit nonzero (watch-loop alarm)"
    assert "ALARM" in proc.stderr and "max-age-hours" in proc.stderr


def test_pod_status_degrades_without_ledger(tmp_path: Path) -> None:
    fake, _ = _install_fake(
        tmp_path, pod_list="podaaaa1111bbbb cage-s0  NVIDIA-L40S  RUNNING\n",
    )
    proc = _bash(
        f'bash "{POD_STATUS}"',
        env=_env(fake, CAGE_POD_LEDGER=str(tmp_path / "absent.jsonl")),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "no ledger" in proc.stderr, "the missing ledger must be announced, not silent"
    assert "unknown — pass --price-per-hour at provision" in proc.stdout
    assert "podaaaa1111bbbb" in proc.stdout


# ---------------------------------------------------------------------------
# cost_report.sh — offline pairing, LIVE flag, malformed refusal, --billing
# ---------------------------------------------------------------------------

def test_cost_report_pairs_events_offline_with_total(tmp_path: Path) -> None:
    fake, log = _install_fake(tmp_path)
    ledger = tmp_path / "pod_ledger.jsonl"
    _write_ledger(ledger, [
        _create_event("podclosed111111", "2026-08-20T00:00:00Z", price=2.0, name="cage-a"),
        {"ts_utc": "2026-08-20T03:00:00Z", "pod_id": "podclosed111111", "event": "delete",
         "by": "watchdog"},  # a seatbelt-fired delete pairs like any other
    ])
    proc = _bash(f'bash "{COST_REPORT}"', env=_env(fake, CAGE_POD_LEDGER=str(ledger)))
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    # Hand-computed: (03:00 − 00:00) = 3.00h x $2.00/h x 1 GPU = $6.00.
    assert re.search(r"podclosed111111.*3\.00\s+2\.00\s+6\.00", proc.stdout)
    assert "TOTAL (known prices): $6.00" in proc.stdout
    assert "LIVE/BILLING" not in proc.stdout.split("TOTAL")[0].replace(
        "FLAGS", ""), "a closed pod must not be flagged live"
    assert _argv_lines(log) == [], "the default report must be fully OFFLINE (no CLI calls)"


def test_cost_report_flags_live_billing_pod(tmp_path: Path) -> None:
    fake, _ = _install_fake(tmp_path)
    ledger = tmp_path / "pod_ledger.jsonl"
    _write_ledger(ledger, [_create_event("podlive22222222", _ts(1.0), price=1.0, name="cage-b")])
    proc = _bash(f'bash "{COST_REPORT}"', env=_env(fake, CAGE_POD_LEDGER=str(ledger)))
    assert proc.returncode == 0
    assert "LIVE/BILLING" in proc.stdout and "OPEN(now)" in proc.stdout
    assert re.search(r"podlive22222222.*1\.0[01]", proc.stdout), "open pod runtime uses NOW as the end (~1h)"


def test_cost_report_refuses_malformed_line_with_line_number(tmp_path: Path) -> None:
    fake, _ = _install_fake(tmp_path)
    ledger = tmp_path / "pod_ledger.jsonl"
    _write_ledger(ledger, [
        _create_event("podclosed111111", "2026-08-20T00:00:00Z", price=None),
        "THIS IS NOT JSON",
    ])
    proc = _bash(f'bash "{COST_REPORT}"', env=_env(fake, CAGE_POD_LEDGER=str(ledger)))
    assert proc.returncode == 2, "malformed ledger lines must refuse the whole report"
    assert "MALFORMED LEDGER LINE 2" in proc.stderr, "the refusal must name the line number"

    # A schema violation (create event missing its required keys) is equally refused.
    _write_ledger(ledger, [{"ts_utc": "2026-08-20T00:00:00Z", "pod_id": "x", "event": "create"}])
    proc2 = _bash(f'bash "{COST_REPORT}"', env=_env(fake, CAGE_POD_LEDGER=str(ledger)))
    assert proc2.returncode == 2 and "MALFORMED LEDGER LINE 1" in proc2.stderr

    # The seatbelt fields are typed too: watchdog_pid must be int|null (a JSON
    # boolean is an int to Python and is refused explicitly).
    for bad in ("12345", True):
        _write_ledger(ledger, [_create_event("podclosed111111", "2026-08-20T00:00:00Z", watchdog_pid=bad)])
        proc3 = _bash(f'bash "{COST_REPORT}"', env=_env(fake, CAGE_POD_LEDGER=str(ledger)))
        assert proc3.returncode == 2 and "watchdog_pid" in proc3.stderr, f"watchdog_pid={bad!r} must refuse"


def test_cost_report_billing_flag_is_labeled_authority(tmp_path: Path) -> None:
    fake, log = _install_fake(tmp_path)
    ledger = tmp_path / "pod_ledger.jsonl"
    _write_ledger(ledger, [_create_event("podclosed111111", "2026-08-20T00:00:00Z"),
                           {"ts_utc": "2026-08-20T01:00:00Z", "pod_id": "podclosed111111",
                            "event": "delete"}])
    proc = _bash(f'bash "{COST_REPORT}" --billing', env=_env(fake, CAGE_POD_LEDGER=str(ledger)))
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "AUTHORITY" in proc.stdout, "the account view must be labeled as the authority"
    assert "Account balance: $12.34" in proc.stdout
    argv = _argv_lines(log)
    assert "billing pods --grouping podId" in argv and "billing network-volume" in argv, (
        "the bare 'billing' group prints help and exits 0 on runpodctl 2.11.0; the "
        "subcommands carry the account view"
    )
    assert not any(l == "billing" or l.startswith("billing -") for l in argv), (
        "the bare group verb must never be called, with or without flags"
    )


# ---------------------------------------------------------------------------
# hygiene: all five scripts parse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", [TEARDOWN, PROVISION, POD_STATUS, COST_REPORT, WATCHDOG],
                         ids=lambda p: p.name)
def test_bash_n_parses(script: Path) -> None:
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"bash -n failed for {script}:\n{proc.stderr}"
