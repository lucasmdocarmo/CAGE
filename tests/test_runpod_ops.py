"""Pins for the RunPod ops suite (restructure phase 2: CLI v2 + pod ledger).

teardown_pod.sh migrated to the runpodctl 2.x command tree (the v1 verbs
`runpodctl get pod` / `runpodctl remove pod` NO LONGER EXIST — the reference is
MyDocs/runpod-cli-reference.md §2), plus three new ops scripts:

  provision_pod.sh  PLAN-by-default provisioning (creation only via --yes = the
                    owner GO, honoring the standing run-approval gate) with a
                    12h --terminate-after cost seatbelt and a create event
                    appended to the pod ledger (results/ops/pod_ledger.jsonl).
  pod_status.sh     read-only monitoring joined with the ledger; exits nonzero
                    when any pod's known age exceeds --max-age-hours (the
                    runaway-cost alarm).
  cost_report.sh    fully-OFFLINE cost table from the ledger create/delete
                    pairs; malformed lines are refused loudly by line number;
                    --billing shells `runpodctl billing` as the account
                    authority.

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
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
TEARDOWN = SCRIPTS / "runpod" / "teardown_pod.sh"
PROVISION = SCRIPTS / "runpod" / "provision_pod.sh"
POD_STATUS = SCRIPTS / "runpod" / "pod_status.sh"
COST_REPORT = SCRIPTS / "runpod" / "cost_report.sh"

GPU = "NVIDIA GeForce RTX 4090"
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Env vars that would leak state (a real ledger, a real pod SSH target, a
# non-interactive auto-yes) into these hermetic tests.
_LEAK_ENV = (
    "CAGE_POD_LEDGER", "CAGE_POD_SSH", "CAGE_ASSUME_YES", "CAGE_RUNPOD_REST",
    "RUNPOD_API_KEY", "CAGE_BACKUP_TARGET", "CAGE_SSH_OPTS", "CAGE_POD_IMAGE",
)

_FAKE = """#!/usr/bin/env bash
set -u
d="{d}"
printf '%s\\n' "$*" >> "$d/argv.log"
{fail_clause}case "$* " in
  "pod delete "*) exit 0 ;;
  "pod list --all "*) cat "$d/pod_list.out" ;;
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
    gpu_list: str = f'"{GPU}"  RTX4090  24  $0.69/hr\n',
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
        "gpu_count": 1, "price_per_hour_usd": price, "terminate_after": "12h",
        "purpose": "s0", "event": "create",
    }
    e.update(over)
    return e


def _write_ledger(path: Path, events: list[dict | str]) -> None:
    lines = [e if isinstance(e, str) else json.dumps(e) for e in events]
    path.write_text("".join(l + "\n" for l in lines), encoding="utf-8")


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
    proc = _bash(
        f'bash "{TEARDOWN}" podtest99999999 "file://{remote}" "{tmp_path}/dest"', env=env,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "TEARDOWN_COMPLETE" in proc.stdout

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
    creates = [c for c in _argv_lines(log) if c.startswith("pod create")]
    assert creates == [], f"PLAN mode must create NOTHING, but the fake saw: {creates}"


def test_provision_plan_derives_price_from_gpu_list(tmp_path: Path) -> None:
    fake, log = _install_fake(tmp_path)
    proc = _bash(f'bash "{PROVISION}" --gpu-id "{GPU}" --hours 6', env=_env(fake))
    assert proc.returncode == 0
    assert "$0.69/h" in proc.stdout, "price must be derived from the (fake) `runpodctl gpu list` row"
    assert re.search(r"estimated total\s+: \$4\.14", proc.stdout), "6h x $0.69 x 1 GPU = $4.14"
    assert any(c.startswith("gpu list") for c in _argv_lines(log))


def test_provision_plan_degrades_to_null_price_when_cli_fails(tmp_path: Path) -> None:
    fake, log = _install_fake(tmp_path, fail_all=True)
    proc = _bash(f'bash "{PROVISION}" --gpu-id "{GPU}" --hours 6', env=_env(fake))
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "unknown" in proc.stdout and "--price-per-hour" in proc.stdout, (
        "with no derivable price the plan must degrade to unknown + a note"
    )
    assert not any(c.startswith("pod create") for c in _argv_lines(log))


def test_provision_requires_gpu_id(tmp_path: Path) -> None:
    fake, _ = _install_fake(tmp_path)
    proc = _bash(f'bash "{PROVISION}"', env=_env(fake))
    assert proc.returncode == 2
    assert "usage:" in proc.stderr and "--gpu-id" in proc.stderr


def test_provision_yes_passes_default_seatbelt_and_writes_ledger(tmp_path: Path) -> None:
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
    assert "--terminate-after 12h" in creates[0], (
        "the 12h cost seatbelt must be passed by DEFAULT on every real create"
    )
    assert "fakepod1234abcd" in proc.stdout and "teardown_pod.sh" in proc.stdout \
        and "setup_runpod.sh" in proc.stdout, "the create must print the id + next steps"

    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    e = json.loads(lines[0])
    assert set(e) == {"ts_utc", "pod_id", "name", "gpu_id", "gpu_count",
                      "price_per_hour_usd", "terminate_after", "purpose", "event"}
    assert e["event"] == "create" and e["pod_id"] == "fakepod1234abcd"
    assert e["name"] == "cage-s0" and e["gpu_id"] == GPU and e["purpose"] == "s0-gate"
    assert isinstance(e["gpu_count"], int) and e["gpu_count"] == 1
    assert isinstance(e["price_per_hour_usd"], float) and e["price_per_hour_usd"] == 0.86
    assert e["terminate_after"] == "12h"
    datetime.datetime.strptime(e["ts_utc"], TS_FMT)


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
    assert e["terminate_after"] is None


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
        {"ts_utc": "2026-08-20T03:00:00Z", "pod_id": "podclosed111111", "event": "delete"},
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
    assert "billing" in _argv_lines(log)


# ---------------------------------------------------------------------------
# hygiene: all four scripts parse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", [TEARDOWN, PROVISION, POD_STATUS, COST_REPORT],
                         ids=lambda p: p.name)
def test_bash_n_parses(script: Path) -> None:
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"bash -n failed for {script}:\n{proc.stderr}"
