"""scripts/6_experiments/monitor_pod.sh (ADR-0143): the Mac-side pod monitor.

Design section 9 test 5: fed canned `pod get` JSON, a watch_campaign exit code,
a .STATUS-* file, a stale journal with 0 percent GPU, log lines and sync
markers, one tick writes status.json, errors.jsonl (de-duplicated) and
alerts.log, and prints the PORTAL ACTION block only on an alert. Fakes:
runpodctl, ssh (runs the bundle locally under a fake HOME), nvidia-smi, stat,
and a fake scripts tree holding pod_watchdog.sh and watch_campaign.sh.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR = REPO_ROOT / "scripts" / "6_experiments" / "monitor_pod.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


def _w(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture()
def world(tmp_path: Path) -> Dict[str, Path]:
    b = tmp_path / "bin"
    home = tmp_path / "podhome"; home.mkdir()
    pod_repo = tmp_path / "podrepo"
    root = pod_repo / "results" / "camp1" / "a" / "run1"
    (root / "cells" / "k1" / "window_squad_v2-01").mkdir(parents=True)
    (root / "cells" / "k1" / "window_squad_v2-01" / "regime.json").write_text('{"label": "KV_PRESSURE_OK"}', encoding="utf-8")
    (root / "write_time_hashes.jsonl").write_text("{}\n", encoding="utf-8")
    (pod_repo / ".agent").mkdir()
    jobs = home / ".cage_jobs"; jobs.mkdir()
    (jobs / "run.log").write_text("step 1 ok\nstep 2 ok\n", encoding="utf-8")
    (jobs / "run.pid").write_text(str(os.getpid()), encoding="utf-8")   # a live pid: RUNNING
    _w(b / "ssh", r'''
        #!/bin/bash
        host=""
        while [ $# -gt 0 ]; do case "$1" in -p|-i|-o) shift 2 ;; -*) shift ;; *) if [ -z "$host" ]; then host="$1"; shift; else break; fi ;; esac; done
        export HOME="$CAGE_TEST_HOME"; cd "$HOME"
        [ -z "${CAGE_TEST_SSH_DOWN:-}" ] || exit 255
        exec bash -c "$*"
        ''')
    _w(b / "runpodctl", r'''
        #!/bin/bash
        case "$1" in
          pod)
            [ -z "${CAGE_TEST_POD_GET_EMPTY:-}" ] || exit 1
            extra=""
            [ -n "${CAGE_TEST_POD_MINIMAL:-}" ] || extra=", \"gpuCount\": ${CAGE_TEST_GPU_COUNT:-1}"
            [ -n "${CAGE_TEST_POD_MINIMAL:-}" ] || [ -n "${CAGE_TEST_NO_COST_PER_HR:-}" ] || extra="$extra, \"costPerHr\": ${CAGE_TEST_COST_PER_HR:-3.49}"
            echo "{\"id\": \"pod123\", \"runtimeStatus\": \"${CAGE_TEST_RUNTIME:-running}\", \"desiredStatus\": \"RUNNING\"$extra}" ;;
          user) [ -n "${CAGE_TEST_NO_USER:-}" ] || echo "{\"balance\": ${CAGE_TEST_BALANCE:-300.0}}" ;;
        esac
        ''')
    _w(b / "nvidia-smi", r'''
        #!/bin/bash
        case "$*" in
          *compute-apps*) echo 4242 ;;
          *query-gpu*) echo "${CAGE_TEST_GPU_UTIL:-87}, 60000, 81559" ;;
          *) echo "Xid 0 ECC 0" ;;
        esac
        ''')
    _w(b / "stat", r'''
        #!/bin/bash
        if [ "$1" = "-c" ]; then
          fmt="$2"; shift 2
          if /usr/bin/stat -c %Y / >/dev/null 2>&1; then exec /usr/bin/stat -c "$fmt" "$@"; fi
          [ "$fmt" = "%Y" ] && exec /usr/bin/stat -f %m "$@"
        fi
        exec /usr/bin/stat "$@"
        ''')
    scripts = tmp_path / "fake_scripts"
    _w(scripts / "runpod" / "pod_watchdog.sh", r'''
        #!/bin/bash
        echo "pod=$2 watchdog=${CAGE_TEST_WD:-ALIVE} pid=1 deadline=${CAGE_TEST_DEADLINE:-2099-01-01T00:00:00Z} log=/dev/null"
        ''')
    _w(pod_repo / "scripts" / "5_observability" / "watch_campaign.sh", r'''
        #!/bin/bash
        echo "cells 1/1 windows 1/1"; echo "${CAGE_TEST_VERDICT:-RUNNING-HEALTHY}"; exit "${CAGE_TEST_WATCH_RC:-0}"
        ''')
    out = tmp_path / "monitor"
    return {"bin": b, "home": home, "pod_repo": pod_repo, "root": root, "scripts": scripts, "out": out, "jobs": jobs}


def _tick(w: Dict[str, Path], *extra_args: str, **env_extra: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAGE_")}
    env["PATH"] = f"{w['bin']}:{env.get('PATH', '')}"
    env["CAGE_TEST_HOME"] = str(w["home"])
    env.update(env_extra)
    argv = ["bash", str(MONITOR), "--pod", "pod123", "--ssh", "root@pod.test", "--port", "2222",
            "--run-root", str(w["root"]), "--pod-repo", str(w["pod_repo"]), "--out", str(w["out"]),
            "--scripts", str(w["scripts"]), "--price", "3.49", "--created", "2026-10-06T00:00:00Z",
            "--balance-floor", "50", "--stall-min", "30", "--job", "run", "--backup-backend", "local",
            "--once", *extra_args]
    return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=120, cwd=str(REPO_ROOT))


def _status(w: Dict[str, Path]) -> dict:
    return json.loads((w["out"] / "status.json").read_text(encoding="utf-8"))


def test_healthy_tick_writes_status_and_log_and_no_alert(world: Dict[str, Path]) -> None:
    proc = _tick(world)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    st = _status(world)
    assert st["schema"] == "cage-monitor-status-v1" and st["tick"] == 1
    assert st["pod"]["runtime_status"] == "running" and st["pod"]["watchdog_alive"] is True
    assert st["pod"]["seatbelt_min_left"] > 1000 and st["pod"]["balance"] == 300.0
    assert st["pod"]["cost_usd"] > 0 and st["pod"]["hours"] > 0
    # ADR-0148 item 10: the pod's own rate (all GPUs) times the hours, named as such
    assert st["pod"]["gpu_count"] == 1 and st["pod"]["cost_basis"] == "pod costPerHr"
    assert st["pod"]["cost_per_hour_usd"] == 3.49
    assert st["pod"]["cost_usd"] == round(st["pod"]["hours"] * 3.49, 2)
    assert st["run"]["watch_rc"] == 0 and st["run"]["watch_verdict"] == "RUNNING-HEALTHY"
    assert st["run"]["sentinels"] == 0 and st["run"]["windows"] == 1
    assert st["run"]["journal_age_s"] >= 0 and st["run"]["gpu_util"] == 87.0
    assert st["run"]["compute_apps"] == 1 and st["run"]["job_status"] == "RUNNING"
    assert "KV_PRESSURE_OK" in st["run"]["regime"]
    assert st["alerts"] == []
    assert "PORTAL ACTION" not in proc.stdout
    assert not (world["out"] / "alerts.log").exists()
    log = (world["out"] / "monitor.log").read_text(encoding="utf-8")
    assert "tick=1 pod=running wd=ALIVE" in log and "job=RUNNING" in log
    assert (world["out"] / "nvidia_q_0001.txt").is_file()   # the 1st tick samples nvidia-smi -q


def test_pod_not_running_is_a_hard_alert_with_the_portal_block(world: Dict[str, Path]) -> None:
    proc = _tick(world, CAGE_TEST_RUNTIME="initializing")
    assert proc.returncode == 0
    st = _status(world)
    assert [a["level"] for a in st["alerts"]] == ["HARD"]
    assert "runtimeStatus=initializing" in st["alerts"][0]["text"]
    assert "PORTAL ACTION" in proc.stdout and "HARD: runtimeStatus=initializing" in proc.stdout
    alerts = (world["out"] / "alerts.log").read_text(encoding="utf-8")
    assert " HARD runtimeStatus=initializing" in alerts


def test_stop_on_alert_exits_3_on_a_hard_alert_only(world: Dict[str, Path]) -> None:
    assert _tick(world, "--stop-on-alert", CAGE_TEST_WD="DEAD (stale pidfile)").returncode == 3
    assert _tick(world, "--stop-on-alert", CAGE_TEST_WATCH_RC="3", CAGE_TEST_VERDICT="STALLED>10min").returncode == 0
    st = _status(world)
    assert [a["level"] for a in st["alerts"]] == ["SOFT"] and "STALLED" in st["alerts"][0]["text"]


def test_new_sentinel_and_stale_idle_journal_are_hard(world: Dict[str, Path]) -> None:
    assert _tick(world).returncode == 0
    (world["root"] / "cells" / "k1" / ".STATUS-squad_v2").write_text("failed", encoding="utf-8")
    old = time.time() - 3600
    os.utime(world["root"] / "write_time_hashes.jsonl", (old, old))
    _tick(world, CAGE_TEST_GPU_UTIL="0")
    st = _status(world)
    texts = [a["text"] for a in st["alerts"] if a["level"] == "HARD"]
    assert any("new failure sentinel" in t for t in texts)
    assert any("idle-cell shape" in t for t in texts)
    # the sentinel count is now the baseline: the next tick does not re-alert on it
    _tick(world, CAGE_TEST_GPU_UTIL="90")
    assert not any("new failure sentinel" in a["text"] for a in _status(world)["alerts"])


def test_seatbelt_under_the_floor_is_hard_whatever_the_job_state(world: Dict[str, Path]) -> None:
    # Review 2026-10-06, HIGH 4: the seatbelt deletes the pod mid-score or
    # mid-pull just the same; the alert no longer waits for a RUNNING job.
    soon = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 1800))
    _tick(world, CAGE_TEST_DEADLINE=soon)
    assert any("seatbelt fires in" in a["text"] and a["level"] == "HARD" for a in _status(world)["alerts"])
    (world["jobs"] / "run.pid").unlink()            # no job at all
    _tick(world, CAGE_TEST_DEADLINE=soon)
    st = _status(world)
    assert st["run"]["job_status"] == "NONE"
    assert any("seatbelt fires in" in a["text"] and "(job NONE)" in a["text"] for a in st["alerts"])
    # a custom floor
    far = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3 * 3600))
    _tick(world, "--seatbelt-floor-min", "240", CAGE_TEST_DEADLINE=far)
    assert any("seatbelt fires in" in a["text"] for a in _status(world)["alerts"])


def test_an_ssh_blip_does_not_fire_a_sentinel_alert_on_the_next_tick(world: Dict[str, Path]) -> None:
    # Review 2026-10-06, MEDIUM 5: a failed bundle reports sentinels -1; the
    # baseline must not move, so the healthy tick after it stays quiet.
    assert _tick(world).returncode == 0
    _tick(world, CAGE_TEST_SSH_DOWN="1")
    st = _status(world)
    assert st["run"]["sentinels"] == -1 and st["run"]["sentinels_baseline"] == 0 and st["run"]["ssh_ok"] is False
    _tick(world)
    st = _status(world)
    assert st["alerts"] == [], st["alerts"]
    assert st["run"]["sentinels_baseline"] == 0


def test_the_loop_ends_when_the_parent_is_gone(world: Dict[str, Path]) -> None:
    # Review 2026-10-06, LOW 17: no orphan monitor after a killed master.
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAGE_")}
    env["PATH"] = f"{world['bin']}:{env.get('PATH', '')}"
    env["CAGE_TEST_HOME"] = str(world["home"])
    argv = ["bash", str(MONITOR), "--pod", "pod123", "--ssh", "root@pod.test", "--run-root", str(world["root"]),
            "--pod-repo", str(world["pod_repo"]), "--out", str(world["out"]), "--scripts", str(world["scripts"]),
            "--interval", "1", "--parent", "999999"]
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=60, cwd=str(REPO_ROOT))
    assert proc.returncode == 0
    assert "parent 999999 is gone: exiting" in proc.stdout
    assert not (world["out"] / "status.json").exists()   # it left before the first tick


def test_balance_under_the_floor_and_crashed_job_are_hard(world: Dict[str, Path]) -> None:
    (world["jobs"] / "run.pid").write_text("999999", encoding="utf-8")
    _tick(world, CAGE_TEST_BALANCE="12.5")
    st = _status(world)
    texts = [a["text"] for a in st["alerts"] if a["level"] == "HARD"]
    assert any("balance 12.5 under the floor 50" in t for t in texts)
    assert any("CRASHED" in t for t in texts)


def test_error_lines_are_deduplicated_across_ticks_and_tracebacks_are_soft(world: Dict[str, Path]) -> None:
    (world["jobs"] / "run.log").write_text("ok\nTraceback (most recent call last):\n  boom\n[run_campaign] RELAUNCH FAILED (exit 1)\n", encoding="utf-8")
    _tick(world)
    st = _status(world)
    assert st["new_error_lines"] == 2
    assert any(a["level"] == "SOFT" and "Traceback" in a["text"] for a in st["alerts"])
    _tick(world)
    assert _status(world)["new_error_lines"] == 0
    rows = [json.loads(l) for l in (world["out"] / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and {r["source"] for r in rows} == {"job log"}
    assert all(r["ts_utc"].endswith("Z") and r["first_seen_utc"].endswith("Z") for r in rows)


def test_ssh_failure_is_hard_and_the_tick_still_lands(world: Dict[str, Path]) -> None:
    proc = _tick(world, CAGE_TEST_SSH_DOWN="1")
    assert proc.returncode == 0
    st = _status(world)
    assert any("ssh bundle failed" in a["text"] and a["level"] == "HARD" for a in st["alerts"])
    assert st["run"]["job_status"] == "NONE"


def test_the_balance_carries_forward_between_user_reads(world: Dict[str, Path]) -> None:
    # S0F-38: the balance is read every 10th tick; the ticks between carried None.
    _tick(world)
    st = _status(world)
    assert st["pod"]["balance"] == 300.0 and st["pod"]["balance_utc"].endswith("Z")
    first_utc = st["pod"]["balance_utc"]
    _tick(world, CAGE_TEST_NO_USER="1")
    st = _status(world)
    assert st["pod"]["balance"] == 300.0 and st["pod"]["balance_utc"] == first_utc
    _tick(world, CAGE_TEST_BALANCE="12.5")                      # a fresh read replaces the carried value
    st = _status(world)
    assert st["pod"]["balance"] == 12.5 and any("balance 12.5 under the floor" in a["text"] for a in st["alerts"])


def test_a_run_root_without_cells_reads_no_cell_written_yet(world: Dict[str, Path]) -> None:
    shutil.rmtree(world["root"] / "cells")
    proc = _tick(world, CAGE_TEST_WATCH_RC="1", CAGE_TEST_VERDICT="[watch_campaign] ERROR: run dir not found")
    assert proc.returncode == 0
    st = _status(world)
    assert st["run"]["run_dir"] is False and st["run"]["watch_verdict"].startswith("no cell written yet")
    assert st["run"]["windows"] == 0 and st["alerts"] == []
    assert "no cell written yet" in (world["out"] / "monitor.log").read_text(encoding="utf-8")


def test_cost_clock_uses_the_pods_rate_then_list_price_times_gpu_count(world: Dict[str, Path]) -> None:
    # ADR-0148 item 10: the monitor multiplied hours by the per-GPU list price
    # alone; the provisioner and the two cost scripts multiply by the GPU count,
    # so a 2-GPU pod read 1/2 of its spend here. The pod JSON carries both the
    # pod's own hourly rate and its GPU count.
    _tick(world, CAGE_TEST_GPU_COUNT="2", CAGE_TEST_COST_PER_HR="6.98")
    st = _status(world)["pod"]
    assert st["gpu_count"] == 2 and st["cost_basis"] == "pod costPerHr" and st["cost_per_hour_usd"] == 6.98
    assert st["cost_usd"] == round(st["hours"] * 6.98, 2)
    # without the pod's rate: the list price (3.49) times the pod's 2 GPUs
    _tick(world, CAGE_TEST_GPU_COUNT="2", CAGE_TEST_NO_COST_PER_HR="1")
    st = _status(world)["pod"]
    assert st["cost_basis"] == "list price x gpu_count" and st["cost_per_hour_usd"] == pytest.approx(6.98)
    assert st["cost_usd"] == round(st["hours"] * 6.98, 2)


def test_cost_clock_carries_the_count_and_rate_forward_on_a_failed_pod_read(world: Dict[str, Path]) -> None:
    _tick(world, CAGE_TEST_GPU_COUNT="2", CAGE_TEST_COST_PER_HR="6.98")
    proc = _tick(world, CAGE_TEST_POD_GET_EMPTY="1")
    assert proc.returncode == 0
    st = _status(world)
    assert st["pod"]["runtime_status"] == "unknown"          # the HARD alert still fires
    assert any("runtimeStatus=unknown" in a["text"] for a in st["alerts"])
    assert st["pod"]["gpu_count"] == 2 and st["pod"]["cost_per_hour_usd"] == 6.98
    assert st["pod"]["cost_basis"] == "pod costPerHr" and st["pod"]["cost_usd"] is not None


def test_cost_clock_is_none_when_no_rate_is_known(world: Dict[str, Path]) -> None:
    # a pod JSON without gpuCount or costPerHr and no earlier tick: nothing is
    # assumed (never "1 GPU"), the hours still run
    _tick(world, CAGE_TEST_POD_MINIMAL="1")
    st = _status(world)["pod"]
    assert st["hours"] > 0
    assert st["gpu_count"] is None and st["cost_per_hour_usd"] is None
    assert st["cost_basis"] is None and st["cost_usd"] is None


def test_usage_errors(world: Dict[str, Path]) -> None:
    proc = subprocess.run(["bash", str(MONITOR), "--pod", "x"], capture_output=True, text=True)
    assert proc.returncode == 2
    proc = subprocess.run(["bash", str(MONITOR), "--bogus"], capture_output=True, text=True)
    assert proc.returncode == 2 and "unknown argument" in proc.stderr
