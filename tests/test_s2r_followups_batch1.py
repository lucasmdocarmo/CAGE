"""S2R 2026-11-14 follow-ups, batch 1 (owner 2026-11-15, "Approve all three"): three checks
of the master that could not fail. Item 1: step_has read the step log's own "# cmd:" header,
so a probe whose success token is in its command text passed on a refusal (the ship stage's
"|" name probe, live on pod dxc7uct8ukwniv). Item 10: stage 12 looked for the dry-window
roots under the full run id, while the driver names them with the id's first 30 characters,
so a 38-character cd-act1 id pulled none and said nothing. G7: with POD_RESULTS_DIR set the
disk projection was judged against the whole DISK_GB, the weights included.
Spec: MyDocs/RunPod/S2R_2026-11-14_FOLLOWUPS.md items 1 and 10, and the G7 note.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import test_experiment_master as tm
from test_experiment_master import world  # noqa: F401  (the fake Mac and pod fixture)

#: a mkdir for the fake pod that refuses "|" in a name, as the US-CA-2 volume does (EINVAL 22)
MKDIR_REFUSING_PIPE = r'''
#!/bin/bash
for a in "$@"; do
  case "$a" in *"|"*) echo "mkdir: cannot create directory '$a': Invalid argument" >&2; exit 1 ;; esac
done
exec /bin/mkdir "$@"
'''


def _steps(w: Dict[str, Path]) -> Path:
    return w["exp_root"] / "S1" / tm.DATE / "extras" / "steps"


def test_a_results_root_that_refuses_the_pipe_fails_stage_2_item1(world: Dict[str, Path]) -> None:  # noqa: F811
    """The probe prints NAME_PROBE_REFUSED with exit 0 and its command text carries
    NAME_PROBE_OK: the check must read the output lines only, so the stage fails here,
    at stage 2, not at the first cell write of stage 5."""
    tm._w(world["bin"] / "mkdir", MKDIR_REFUSING_PIPE)
    proc = tm._master(world, "--yes", "provision", "--to", "ship")
    (log,) = list(_steps(world).glob("ship_*results_root_accepts*.log"))
    text = log.read_text(encoding="utf-8")
    assert "NAME_PROBE_REFUSED" in [ln for ln in text.splitlines() if not ln.startswith("#")], text
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert "[FAIL] the results root refuses the character | in file names" in proc.stdout
    assert tm._state(world)["stages"]["ship"]["status"] == "failed"


#: 38 characters, the length of a cd-act1 run id (S1's was 28, inside the driver's 30-character cut)
LONG_RID = "20261006-120000-cd-act1-qwen3-next-80b"
#: the name run_campaign.dry_window_root gives that run: the id's first 30 characters, "-dry-", six hex digits
DRY_NAME = LONG_RID[:30] + "-dry-abc123"


def _land(w: Dict[str, Path]) -> Path:
    return w["exp_root"] / "S1" / tm.DATE


def _to_collect_with_a_long_run_id(w: Dict[str, Path]) -> None:
    """Stage 0 mints the id; the test replaces it (a stage 0 that passed never re-mints),
    then the landing runs through collect, so `--only pull` is the stage under test."""
    proc = tm._master(w)
    assert proc.returncode == 10, proc.stdout[-2000:]      # stopped before provision (no --yes)
    path = _land(w) / "extras" / "state.json"
    st = json.loads(path.read_text(encoding="utf-8"))
    st["run_id"] = LONG_RID
    path.write_text(json.dumps(st, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    proc = tm._master(w, "--yes", "provision", "--to", "collect")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert tm._state(w)["run_id"] == LONG_RID


def _dry_root_on_the_pod(w: Dict[str, Path]) -> Path:
    root = w["pod_repo"] / "results" / "camp1" / "a" / DRY_NAME
    window = root / "cells" / "k1" / "window_squad_v2-01"
    window.mkdir(parents=True)
    (window / "regime.json").write_text('{"label": "IN_REGIME"}', encoding="utf-8")
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    return root


def _run_log_names_the_dry_root(w: Dict[str, Path]) -> None:
    log = _land(w) / "logs" / "runner" / "run.log"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"CAMPAIGN MODE (task #116): run root {w['pod_repo']}/results/camp1/a/{DRY_NAME}\n")


def test_stage_12_pulls_the_dry_root_of_a_run_id_longer_than_30_characters_item10(world: Dict[str, Path]) -> None:  # noqa: F811
    _to_collect_with_a_long_run_id(world)
    _dry_root_on_the_pod(world)
    _run_log_names_the_dry_root(world)
    proc = tm._master(world, "--only", "pull")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    local = _land(world) / "run" / "camp1" / "a" / DRY_NAME
    assert (local / "cells" / "k1" / "window_squad_v2-01" / "regime.json").is_file(), proc.stdout[-2000:]
    assert (local / "manifest.json").is_file()
    assert f"dry-window evidence pulled: 1 root(s) ({DRY_NAME})" in proc.stdout


def test_a_dry_root_the_run_log_names_but_the_pull_did_not_land_fails_stage_12_item10(world: Dict[str, Path]) -> None:  # noqa: F811
    """The driver's own log is the record of the dry roots it wrote; one that is not on
    the Mac after the pull fails the stage before teardown can delete the only copy."""
    _to_collect_with_a_long_run_id(world)
    _run_log_names_the_dry_root(world)          # named, but absent on the pod
    proc = tm._master(world, "--only", "pull")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert f"[FAIL] the run log names the dry root {DRY_NAME} but the pull did not land it" in proc.stdout
    assert tm._state(world)["stages"]["pull"]["status"] == "failed"


def test_a_run_without_a_dry_window_passes_stage_12_and_says_so_item10(world: Dict[str, Path]) -> None:  # noqa: F811
    _to_collect_with_a_long_run_id(world)
    proc = tm._master(world, "--only", "pull")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "no dry-window root on the pod and none named by the run log" in proc.stdout


def test_a_dead_channel_during_the_dry_root_listing_fails_stage_12_item10(world: Dict[str, Path]) -> None:  # noqa: F811
    """Before item 10 the listing's failure was swallowed (`|| true`), so a dead channel
    read as "no dry root"; nothing is judged absent on a dead channel."""
    _to_collect_with_a_long_run_id(world)
    _dry_root_on_the_pod(world)
    proc = tm._master(world, "--only", "pull")
    assert proc.returncode == 0, proc.stdout[-3000:]
    before = len(tm._calls(world))
    proc = tm._master(world, "--only", "pull", "--redo")
    assert proc.returncode == 0, proc.stdout[-3000:]
    ssh_calls = [c for c in tm._calls(world)[before:] if c.startswith("ssh")]
    probe = next(i for i, c in enumerate(ssh_calls) if "ls -d" in c and "-dry-" in c)
    (world["home"] / ".ssh_fail_skip").write_text(f"{probe}\n", encoding="utf-8")
    (world["home"] / ".ssh_fail").write_text("3\n", encoding="utf-8")
    proc = tm._master(world, "--only", "pull", "--redo", CAGE_SSH_RETRY_S="0")
    (world["home"] / ".ssh_fail").write_text("0\n", encoding="utf-8")
    (world["home"] / ".ssh_fail_skip").write_text("0\n", encoding="utf-8")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "the dry-root listing did not reach the pod (ssh rc=255)" in proc.stdout


def _g7_pod_gb(w: Dict[str, Path]) -> str:
    logs = sorted(_steps(w).glob("plan_*disk_projection*.log"), key=lambda q: q.stat().st_mtime)
    cmd = logs[-1].read_text(encoding="utf-8").split("# cmd:")[1].splitlines()[0]
    return cmd.split("--pod-gb ")[1].split()[0]


def test_the_disk_projection_bound_is_the_results_disk_free_space_read_at_stage_6_g7(world: Dict[str, Path], tmp_path: Path) -> None:  # noqa: F811
    """With POD_RESULTS_DIR set the run root and its mirror share the container disk with
    the weights and with whatever the stages before the plan wrote there (the S2R pod read
    91 GiB free at stage 3 and 65 GiB at stage 11), so the bound is the free space of that
    disk read live at stage 6 (owner 2026-11-15, option C), recorded in the state; a read
    without a number fails the stage. The world fixture's df answers CAGE_TEST_DF_FREE_KB."""
    real = tmp_path / "cage-results"
    prof = tmp_path / "S1_results.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8") + f"\nPOD_RESULTS_DIR={real}\n", encoding="utf-8")
    world["profile"] = prof
    proc = tm._master(world, "--yes", "provision", "--to", "plan")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert _g7_pod_gb(world) == "91"
    # the disk filled since stage 3 (whose record stays 91): the live read decides
    proc = tm._master(world, "--only", "plan", "--redo", CAGE_TEST_DF_FREE_KB=str(7 * 1024 * 1024))
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert _g7_pod_gb(world) == "7"
    st = tm._state(world)
    assert st["pod"]["results_disk_free_gb"] == "7" and st["pod"]["disk_free_gb"] == "91"
    assert f"disk projection (G7): the bound is the 7 GB free now on the disk holding {real} (DISK_GB=120 holds the weights too)" in proc.stdout
    (read,) = list(_steps(world).glob("plan_*results_disk_free_space*.log"))
    assert f"df -Pk {real}" in read.read_text(encoding="utf-8").replace("\\", "")     # the header is %q-quoted
    # a read without a number refuses: stage 6 has the pod, so nothing is guessed
    proc = tm._master(world, "--only", "plan", "--redo", CAGE_TEST_DF_FREE_KB="none")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert f"[FAIL] plan: the free-space read on {real} returned no number (G7)" in proc.stdout


def test_without_pod_results_dir_the_disk_projection_bound_stays_the_volume_g7(world: Dict[str, Path]) -> None:  # noqa: F811
    proc = tm._master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_DF_FREE_KB=str(21 * 1024 * 1024))
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert _g7_pod_gb(world) == "100"      # VOLUME_GB of _common.env: the network volume holds the run root
    assert not list(_steps(world).glob("plan_*results_disk_free_space*.log"))
    assert "free now on the disk holding" not in proc.stdout


# ---------------------------------------------------------------------------
# independent review 2026-11-15 (a fresh subagent on the session's model): M1, L1, L4, L6
# ---------------------------------------------------------------------------

#: a wrapper that prints one line with a NUL byte, then runs the fake it replaced
NUL_THEN_REAL = r'''
#!/bin/bash
printf 'progress \0 bar\n'
exec bash "$(dirname "$0")/__NAME___real.sh" "$@"
'''


def test_a_nul_byte_in_a_step_log_blinds_neither_step_has_nor_step_out_review_l1(world: Dict[str, Path]) -> None:  # noqa: F811
    """grep prints "Binary file ... matches" in place of the lines of a log that holds a NUL
    unless it reads it as text (-a): the pod id of a created, billing pod would not be parsed
    (step_out) and every marker check would read absent (step_has)."""
    for name in ("provision_pod", "pod_watchdog"):
        real = world["scripts"] / "runpod" / f"{name}.sh"
        real.rename(real.with_name(f"{name}_real.sh"))
        tm._w(real, NUL_THEN_REAL.replace("__NAME__", name))
    proc = tm._master(world, "--yes", "provision", "--to", "provision")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert tm._state(world)["pod"]["id"] == "pod123"
    (log,) = list(_steps(world).glob("provision_*provision_pod_in_*.log"))
    assert b"\x00" in log.read_bytes()            # the NUL reached the step log


def test_an_attempt_log_older_than_the_last_setup_does_not_fail_stage_12_review_m1(world: Dict[str, Path]) -> None:  # noqa: F811
    """A landing resumed on a re-provisioned pod (the container disk guard's instruction:
    --from setup --redo) keeps the earlier attempt's run_<stamp>.log on the Mac; the dry
    root it names died with that disk, so it is named and not required. run.log, the
    latest attempt, is always required."""
    _to_collect_with_a_long_run_id(world)
    gone = LONG_RID[:30] + "-dry-0ld999"
    old = _land(world) / "logs" / "runner" / "run_20200101T000000Z.log"
    old.write_text(f"CAMPAIGN MODE (task #116): run root /gone/results/camp1/a/{gone}\n", encoding="utf-8")
    proc = tm._master(world, "--only", "pull")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "attempt log run_20200101T000000Z.log predates the last setup" in proc.stdout
    assert gone not in [ln for ln in proc.stdout.splitlines() if "[FAIL]" in ln]


def test_stage_12_says_when_no_run_log_is_there_to_cross_check_review_l4(world: Dict[str, Path]) -> None:  # noqa: F811
    _to_collect_with_a_long_run_id(world)
    for log in (_land(world) / "logs" / "runner").glob("run*.log"):
        log.rename(log.with_suffix(".moved"))
    proc = tm._master(world, "--only", "pull")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "no run log on the Mac to cross-check" in proc.stdout
    assert "none named by the run log" not in proc.stdout


def test_stage_12_pulls_the_dry_and_drift_roots_of_a_short_run_id_review_l6(world: Dict[str, Path]) -> None:  # noqa: F811
    """S1's shape: a run id of 30 characters or fewer names its roots with the whole id."""
    proc = tm._master(world, "--yes", "provision", "--to", "collect")
    assert proc.returncode == 0, proc.stdout[-3000:]
    rid = tm._state(world)["run_id"]
    assert len(rid) <= 30
    for suffix in ("-dry-abc123", "-dry-drift"):
        window = world["pod_repo"] / "results" / "camp1" / "a" / (rid + suffix) / "cells" / "k1" / "window_squad_v2-01"
        window.mkdir(parents=True)
        (window / "regime.json").write_text("{}", encoding="utf-8")
    proc = tm._master(world, "--only", "pull")
    assert proc.returncode == 0, proc.stdout[-3000:]
    for suffix in ("-dry-abc123", "-dry-drift"):
        assert (_land(world) / "run" / "camp1" / "a" / (rid + suffix) / "cells" / "k1" / "window_squad_v2-01" / "regime.json").is_file()
    assert "dry-window evidence pulled: 2 root(s)" in proc.stdout


def test_stage_3_refuses_one_gb_under_the_default_floor_and_passes_at_it(world: Dict[str, Path]) -> None:  # noqa: F811
    """The 20 GB default of DISK_FREE_MIN_GB, pinned on purpose: until 2026-11-15 the whole
    suite exercised it by accident through the host's own disk (102 failures at 14 GiB free)."""
    proc = tm._master(world, "--yes", "provision", "--to", "setup", CAGE_TEST_DF_FREE_KB=str(20 * 1024 * 1024 - 1))
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] setup: 19 GB free on the disk holding the model cache" in proc.stdout and "below DISK_FREE_MIN_GB=20" in proc.stdout
    proc = tm._master(world, "--from", "setup", "--redo", "--to", "setup", CAGE_TEST_DF_FREE_KB=str(20 * 1024 * 1024))
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert tm._state(world)["pod"]["disk_free_gb"] == "20"
