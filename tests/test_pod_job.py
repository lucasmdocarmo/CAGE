"""scripts/6_experiments/pod_job.sh (ADR-0143): detached pod jobs over ssh.

The runner is the RunPod port of scripts/gcp/remote_job.sh: submit writes a
local JSON handle and four remote files, status reads the status file first,
wait polls with backoff to a hard deadline (124), kill signals the recorded
pid's group only, fetch copies the remote log. Here `ssh` is a fake on PATH
that runs the command locally under a temporary HOME (the "pod"), `scp` copies
from that HOME, and `setsid` is a shim (the Mac has none; the remote snippet
needs it). No network, no pod.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
POD_JOB = REPO_ROOT / "scripts" / "6_experiments" / "pod_job.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")

FAKE_SSH = r'''#!/bin/bash
# fake ssh: skip option flags (and their values), the first bare word is the
# host, the rest is the remote command, run locally under the fake HOME.
set -u
host=""
while [ $# -gt 0 ]; do
  case "$1" in
    -p|-i|-o) shift 2 ;;
    -*) shift ;;
    *) if [ -z "$host" ]; then host="$1"; shift; else break; fi ;;
  esac
done
[ -n "$host" ] || { echo "fake ssh: no host" >&2; exit 255; }
# ADR-0161: an outage injector. The file .ssh_fail holds a count of round
# trips that fail like a timed-out channel (exit 255, nothing on stdout).
if [ -s "$CAGE_TEST_HOME/.ssh_fail" ]; then
  n=$(cat "$CAGE_TEST_HOME/.ssh_fail")
  if [ "$n" -gt 0 ]; then
    echo $(( n - 1 )) > "$CAGE_TEST_HOME/.ssh_fail"
    echo "Connection timed out during banner exchange" >&2
    exit 255
  fi
fi
echo "$host" >> "$CAGE_TEST_HOME/.ssh_hosts"
export HOME="$CAGE_TEST_HOME"
cd "$HOME"
exec bash -c "$*"
'''

FAKE_SCP = r'''#!/bin/bash
set -u
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -P|-i|-o) shift 2 ;;
    -*) shift ;;
    *) args+=("$1"); shift ;;
  esac
done
# ADR-0161: the same outage injector as the fake ssh (.scp_fail)
if [ -s "$CAGE_TEST_HOME/.scp_fail" ]; then
  n=$(cat "$CAGE_TEST_HOME/.scp_fail")
  if [ "$n" -gt 0 ]; then
    echo $(( n - 1 )) > "$CAGE_TEST_HOME/.scp_fail"
    echo "scp: Connection closed" >&2
    exit 255
  fi
fi
src="${args[0]}"; dst="${args[1]}"
path="${src#*:}"
cp "$CAGE_TEST_HOME/$path" "$dst"
'''

FAKE_SETSID = "#!/bin/sh\nexec \"$@\"\n"


def _stub(d: Path, name: str, body: str) -> None:
    p = d / name
    p.write_text(body, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture()
def pod(tmp_path: Path) -> Dict[str, Path]:
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    _stub(stub_bin, "ssh", FAKE_SSH)
    _stub(stub_bin, "scp", FAKE_SCP)
    _stub(stub_bin, "setsid", FAKE_SETSID)
    home = tmp_path / "podhome"
    home.mkdir()
    jobs = tmp_path / "jobs"
    return {"bin": stub_bin, "home": home, "jobs": jobs}


def _env(pod: Dict[str, Path], **extra: str) -> Dict[str, str]:
    env = {
        k: v for k, v in os.environ.items()
        # PYTHONUNBUFFERED stripped so the unbuffered-job test proves the wrapper,
        # never an inherited shell export (review 2026-10-10, LOW 6)
        if not k.startswith("CAGE_POD") and k not in ("CAGE_JOBS_DIR", "CAGE_SSH_KEY", "PYTHONUNBUFFERED")
    }
    env["PATH"] = f"{pod['bin']}:{env.get('PATH', '')}"
    env["CAGE_TEST_HOME"] = str(pod["home"])
    env["CAGE_POD_SSH"] = "root@pod.test"
    env["CAGE_POD_SSH_PORT"] = "2222"
    env["CAGE_JOBS_DIR"] = str(pod["jobs"])
    env["CAGE_POD_JOB_POLL_S"] = "1"
    env.update(extra)
    return env


def _run(pod: Dict[str, Path], *argv: str, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(POD_JOB), *argv], capture_output=True, text=True,
        env=_env(pod, **extra), timeout=120, cwd=str(REPO_ROOT),
    )


def _wait_status(pod: Dict[str, Path], name: str, want_prefix: str, tries: int = 20) -> str:
    out = ""
    for _ in range(tries):
        out = _run(pod, "status", name).stdout.strip()
        if out.startswith(want_prefix):
            return out
        time.sleep(0.25)
    return out


# ---------------------------------------------------------------------------
# submit, status, the handle and the remote files
# ---------------------------------------------------------------------------


def test_submit_writes_the_handle_and_the_remote_files(pod: Dict[str, Path]) -> None:
    proc = _run(pod, "submit", "hello", "echo hi from the pod; exit 0", "60")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "submitted hello -> root@pod.test (remote pid " in proc.stdout
    handle = json.loads((pod["jobs"] / "hello.json").read_text(encoding="utf-8"))
    assert handle["id"] == "hello" and handle["mode"] == "pod-ssh"
    assert handle["host"] == "root@pod.test" and handle["port"] == "2222"
    assert re.fullmatch(r"pid:\d+", handle["handle"])
    assert handle["remote_log"] == "~/.cage_jobs/hello.log"
    assert handle["remote_status"] == "~/.cage_jobs/hello.status"
    assert handle["remote_pid_file"] == "~/.cage_jobs/hello.pid"
    assert "pod_job.sh kill hello" in handle["cancel_cmd"]
    assert "-p 2222" in handle["poll_cmd"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", handle["submitted_at"])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", handle["deadline_at"])
    assert handle["deadline_s"] == 60 and handle["billable"] is True
    assert _wait_status(pod, "hello", "DONE") == "DONE(0)"
    remote = pod["home"] / ".cage_jobs"
    assert (remote / "hello.cmd").read_text(encoding="utf-8") == "echo hi from the pod; exit 0"
    assert (remote / "hello.status").read_text(encoding="utf-8").strip() == "0"
    assert (remote / "hello.pid").read_text(encoding="utf-8").strip() == handle["handle"][4:]
    assert "hi from the pod" in (remote / "hello.log").read_text(encoding="utf-8")
    # the ssh went to the configured host
    assert "root@pod.test" in (pod["home"] / ".ssh_hosts").read_text(encoding="utf-8")


def test_failed_job_reports_its_exit_code(pod: Dict[str, Path]) -> None:
    assert _run(pod, "submit", "boom", "echo failing >&2; exit 7").returncode == 0
    assert _wait_status(pod, "boom", "FAILED") == "FAILED(7)"
    proc = _run(pod, "status", "boom")
    assert proc.returncode == 1


def test_wait_returns_124_at_the_deadline_and_kill_reaps_the_recorded_pid(pod: Dict[str, Path]) -> None:
    assert _run(pod, "submit", "slow", "sleep 20; echo late").returncode == 0
    proc = _run(pod, "wait", "slow", "1")
    assert proc.returncode == 124, proc.stdout + proc.stderr
    assert "DEADLINE_EXCEEDED after 1s" in proc.stdout
    assert "pod_job.sh kill slow" in proc.stdout
    handle = json.loads((pod["jobs"] / "slow.json").read_text(encoding="utf-8"))
    pid = int(handle["handle"][4:])
    proc = _run(pod, "kill", "slow")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"killed slow (remote pid {pid} on root@pod.test)" in proc.stdout
    assert (pod["home"] / ".cage_jobs" / "slow.status").read_text(encoding="utf-8").strip() == "143"
    assert _run(pod, "status", "slow").stdout.strip() == "KILLED"
    # the recorded pid is gone (only that pid was signaled)
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.25)
    else:
        pytest.fail(f"recorded pid {pid} still alive after kill")


def test_wait_returns_0_on_done_and_1_on_failed(pod: Dict[str, Path]) -> None:
    assert _run(pod, "submit", "ok", "exit 0").returncode == 0
    proc = _run(pod, "wait", "ok", "30")
    assert proc.returncode == 0 and proc.stdout.strip().endswith("DONE(0)")
    assert _run(pod, "submit", "bad", "exit 3").returncode == 0
    proc = _run(pod, "wait", "bad", "30")
    assert proc.returncode == 1 and proc.stdout.strip().endswith("FAILED(3)")


def test_crashed_when_the_pid_is_dead_and_no_status_landed(pod: Dict[str, Path]) -> None:
    pod["jobs"].mkdir()
    (pod["jobs"] / "ghost.json").write_text(
        json.dumps({"id": "ghost", "handle": "pid:999999"}), encoding="utf-8")
    proc = _run(pod, "status", "ghost")
    assert proc.stdout.strip() == "CRASHED" and proc.returncode == 1


def test_unknown_without_a_handle(pod: Dict[str, Path]) -> None:
    proc = _run(pod, "status", "nothing")
    assert proc.stdout.strip() == "UNKNOWN" and proc.returncode == 1


def test_lost_when_the_pid_is_alive_but_not_our_job(pod: Dict[str, Path]) -> None:
    # Review 2026-10-06, LOW 15 (the J8 pid-reuse guard): this test process is
    # alive and is not the job, so the handle reads LOST, never RUNNING.
    pod["jobs"].mkdir()
    (pod["jobs"] / "reused.json").write_text(
        json.dumps({"id": "reused", "handle": f"pid:{os.getpid()}"}), encoding="utf-8")
    proc = _run(pod, "status", "reused")
    assert proc.stdout.strip() == "LOST" and proc.returncode == 1
    proc = _run(pod, "wait", "reused", "5")
    assert proc.returncode == 1 and proc.stdout.strip().endswith("LOST")


def test_status_reports_unreachable_when_the_ssh_channel_fails(pod: Dict[str, Path]) -> None:
    # ADR-0161 (S0 attempt 2, 2026-10-10 01:29Z): a timed-out ssh round trip is
    # the channel, not the job; it must not read as the handle-less UNKNOWN.
    assert _run(pod, "submit", "blink", "sleep 15").returncode == 0
    (pod["home"] / ".ssh_fail").write_text("1\n", encoding="utf-8")
    proc = _run(pod, "status", "blink")
    assert proc.stdout.strip() == "UNREACHABLE" and proc.returncode == 1
    assert "banner exchange" in proc.stderr
    assert _run(pod, "status", "blink").stdout.strip() == "RUNNING"  # the next round trip is fine
    _run(pod, "kill", "blink")


def test_wait_rides_out_a_short_ssh_outage_and_still_reads_the_verdict(pod: Dict[str, Path]) -> None:
    # Two failed polls inside the grace, then the job's own exit code decides.
    assert _run(pod, "submit", "outage", "sleep 3; exit 0").returncode == 0
    (pod["home"] / ".ssh_fail").write_text("2\n", encoding="utf-8")
    proc = _run(pod, "wait", "outage", "60", CAGE_POD_JOB_UNREACHABLE_GRACE_S="60")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[-1] == "DONE(0)"
    assert sum(1 for ln in lines if "unverified for" in ln and "ADR-0161" in ln) == 2
    assert "UNREACHABLE for" not in proc.stdout
    # the verdict of a job that FAILED during the outage is still its exit code
    assert _run(pod, "submit", "outage2", "exit 5").returncode == 0
    (pod["home"] / ".ssh_fail").write_text("1\n", encoding="utf-8")
    proc = _run(pod, "wait", "outage2", "60", CAGE_POD_JOB_UNREACHABLE_GRACE_S="60")
    assert proc.returncode == 1 and proc.stdout.strip().endswith("FAILED(5)")


def test_wait_fails_after_the_unreachable_grace_and_names_the_billing_risk(pod: Dict[str, Path]) -> None:
    assert _run(pod, "submit", "dark", "sleep 30").returncode == 0
    (pod["home"] / ".ssh_fail").write_text("1000\n", encoding="utf-8")
    proc = _run(pod, "wait", "dark", "60", CAGE_POD_JOB_UNREACHABLE_GRACE_S="2")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    last = proc.stdout.strip().splitlines()[-1]
    assert last.startswith("UNREACHABLE for ") and "may still be RUNNING and billing" in last
    assert "CAGE_POD_JOB_UNREACHABLE_GRACE_S=2" in last
    # grace 0 restores the fail-at-once behavior
    proc = _run(pod, "wait", "dark", "60", CAGE_POD_JOB_UNREACHABLE_GRACE_S="0")
    assert proc.returncode == 1 and proc.stdout.strip().splitlines()[-1].startswith("UNREACHABLE for 0s")
    (pod["home"] / ".ssh_fail").write_text("0\n", encoding="utf-8")
    _run(pod, "kill", "dark")


def test_fetch_retries_a_failed_channel_and_gives_up_after_three(pod: Dict[str, Path]) -> None:
    # ADR-0161: a copy is safe to repeat; the channel's 255 gets three attempts.
    assert _run(pod, "submit", "copyme", "echo payload").returncode == 0
    assert _wait_status(pod, "copyme", "DONE") == "DONE(0)"
    dest = pod["jobs"].parent / "fetched_retry"
    (pod["home"] / ".scp_fail").write_text("2\n", encoding="utf-8")
    proc = _run(pod, "fetch", "copyme", str(dest), CAGE_SSH_RETRY_S="0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # the retry lines go to stderr (review 2026-10-10, MEDIUM 1): stdout stays data
    assert proc.stderr.count("attempt") == 2 and "attempt 3 of 3" in proc.stderr
    assert "attempt" not in proc.stdout and proc.stdout.strip().startswith("fetched ->")
    assert (dest / "copyme.log").read_text(encoding="utf-8").startswith("payload")
    (pod["home"] / ".scp_fail").write_text("5\n", encoding="utf-8")
    proc = _run(pod, "fetch", "copyme", str(dest / "again"), CAGE_SSH_RETRY_S="0")
    assert proc.returncode == 1 and "fetch failed for 'copyme'" in proc.stderr
    assert proc.stderr.count("attempt") == 2
    (pod["home"] / ".scp_fail").write_text("0\n", encoding="utf-8")


def test_jobs_run_with_unbuffered_python_output(pod: Dict[str, Path]) -> None:
    # ADR-0162: a killed job's python lines must be in its log, not in a buffer.
    assert _run(pod, "submit", "buf", "echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED").returncode == 0
    assert _wait_status(pod, "buf", "DONE") == "DONE(0)"
    assert "PYTHONUNBUFFERED=1" in (pod["home"] / ".cage_jobs" / "buf.log").read_text(encoding="utf-8")


def test_a_landed_status_wins_over_the_pid_heuristics(pod: Dict[str, Path]) -> None:
    # S0F-32: the probe's checks are not atomic; a job that ended between them
    # has its exit code on disk, and that code is the verdict, never LOST or
    # CRASHED. Both pid states are covered: alive-and-not-ours, and dead.
    pod["jobs"].mkdir()
    (pod["home"] / ".cage_jobs").mkdir(parents=True, exist_ok=True)
    for name, pid in (("late", os.getpid()), ("gone", 999999)):
        (pod["jobs"] / f"{name}.json").write_text(json.dumps({"id": name, "handle": f"pid:{pid}"}), encoding="utf-8")
        (pod["home"] / ".cage_jobs" / f"{name}.status").write_text("0\n", encoding="utf-8")
        proc = _run(pod, "status", name)
        assert proc.stdout.strip() == "DONE(0)" and proc.returncode == 0, (name, proc.stdout)


def test_resubmit_starts_a_fresh_remote_log(pod: Dict[str, Path]) -> None:
    # Review 2026-10-06, MEDIUM 8: a marker from attempt 1 must not survive into attempt 2's log.
    assert _run(pod, "submit", "again", "echo ATTEMPT_ONE_MARKER").returncode == 0
    assert _wait_status(pod, "again", "DONE") == "DONE(0)"
    assert _run(pod, "submit", "again", "echo attempt two").returncode == 0
    assert _wait_status(pod, "again", "DONE") == "DONE(0)"
    log = (pod["home"] / ".cage_jobs" / "again.log").read_text(encoding="utf-8")
    assert "ATTEMPT_ONE_MARKER" not in log and "attempt two" in log


def test_double_submit_of_a_running_job_is_refused(pod: Dict[str, Path]) -> None:
    assert _run(pod, "submit", "twice", "sleep 15").returncode == 0
    proc = _run(pod, "submit", "twice", "echo again")
    assert proc.returncode == 1
    assert "already RUNNING" in proc.stderr
    _run(pod, "kill", "twice")


def test_invalid_job_names_are_refused(pod: Dict[str, Path]) -> None:
    for bad in ("a b", "x/y", "$(id)", "it;s"):
        proc = _run(pod, "submit", bad, "echo no")
        assert proc.returncode == 1, bad
        assert "invalid job name" in proc.stderr
    assert not (pod["home"] / ".cage_jobs").exists()


def test_tail_grep_fetch_and_list(pod: Dict[str, Path]) -> None:
    assert _run(pod, "submit", "chatty", "for i in 1 2 3; do echo line $i; done; echo 'Traceback: x'").returncode == 0
    assert _wait_status(pod, "chatty", "DONE") == "DONE(0)"
    tail = _run(pod, "tail", "chatty", "2").stdout
    assert tail.strip().splitlines() == ["line 3", "Traceback: x"]
    grep = _run(pod, "grep", "chatty").stdout
    assert "Traceback: x" in grep and "line 1" not in grep
    dest = pod["jobs"].parent / "fetched"
    proc = _run(pod, "fetch", "chatty", str(dest))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (dest / "chatty.log").read_text(encoding="utf-8").startswith("line 1")
    listing = _run(pod, "list").stdout
    assert "chatty" in listing and "DONE(0)" in listing


def test_missing_host_is_loud(pod: Dict[str, Path]) -> None:
    env = _env(pod)
    env.pop("CAGE_POD_SSH")
    proc = subprocess.run(["bash", str(POD_JOB), "submit", "x", "echo"], capture_output=True,
                          text=True, env=env, timeout=60, cwd=str(REPO_ROOT))
    assert proc.returncode == 1 and "CAGE_POD_SSH is unset" in proc.stderr


def test_usage_exits_2(pod: Dict[str, Path]) -> None:
    proc = _run(pod, "nonsense")
    assert proc.returncode == 2 and "USAGE" in proc.stderr


# ---------------------------------------------------------------------------
# static pins: header, sourcing, process safety, dashes
# ---------------------------------------------------------------------------


def test_static_contract() -> None:
    text = POD_JOB.read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:30])
    assert "# Order:" in head and "# Objective:" in head and "# Cloud:     runpod" in head
    assert re.search(r"^\s*source\s+[^#\n]*_common\.sh", text, re.M)
    for banned in ("pkill", "killall", "kill -1", "kill 0 ", "-P 1"):
        assert banned not in text, banned
    assert chr(0x2014) not in text and chr(0x2013) not in text
    proc = subprocess.run(["bash", "-n", str(POD_JOB)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    # kills name the recorded pid's group, never a pattern
    assert 'kill -- -$pid' in text
