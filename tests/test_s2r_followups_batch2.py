"""S2R 2026-11-14 follow-ups, batch 2 (owner 2026-11-15, "Approve all five").
Item 2: the monitor read a job that wrote its status and exited between the probe's two
checks as CRASHED. Item 8: teardown_pod.sh ran its final sync ON the pod against the Mac's
view of the backup (ssh:// to the pod's own public address), which cannot connect from
inside the pod. Item 11: stage 14 tested for a verify_results marker the script never
prints. Item 9: on a budget-cut tree the analysis driver's "no cells for this contrast"
refusal failed the stage before the panels and analysis.txt. Item 4: the pod shape
evidence carried no CPU model, socket layout, GPU topology, host load or disk breakdown.
Spec: MyDocs/RunPod/S2R_2026-11-14_FOLLOWUPS.md items 2, 4, 8, 9 and 11.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

import test_experiment_master as tm
import test_monitor_pod as tmon
from test_experiment_master import world  # noqa: F401  (the fake Mac and pod of the master)
from test_monitor_pod import world as mon_world  # noqa: F401  (the fake pod of the monitor)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEARDOWN = REPO_ROOT / "scripts" / "runpod" / "teardown_pod.sh"

for _p in (REPO_ROOT, REPO_ROOT / "scripts" / "4_analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _land(w: Dict[str, Path]) -> Path:
    return w["exp_root"] / "S1" / tm.DATE


def _steps(w: Dict[str, Path]) -> Path:
    return _land(w) / "extras" / "steps"


# ---------------------------------------------------------------------------
# item 2: the monitor's job-status probe
# ---------------------------------------------------------------------------

#: a cat that finishes the job at the instant the probe reads its pid: the status file
#: appears after the probe's first status check and the pid it returns is dead
CAT_THAT_ENDS_THE_JOB = r'''
#!/bin/bash
case "$*" in
  *run.pid) echo "DONE(0)" > "$HOME/.cage_jobs/run.status"; echo 999999; exit 0 ;;
esac
exec /bin/cat "$@"
'''


def test_a_job_that_ends_between_the_two_checks_is_not_crashed_item2(mon_world: Dict[str, Path]) -> None:  # noqa: F811
    """pod_job.sh status re-reads the status file after the pid check; the monitor's probe
    did not, so a clean exit inside that gap read as CRASHED, a HARD alert stage 8 fails on."""
    tmon._w(mon_world["bin"] / "cat", CAT_THAT_ENDS_THE_JOB)
    assert not (mon_world["jobs"] / "run.status").exists()
    proc = tmon._tick(mon_world)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    st = tmon._status(mon_world)
    assert st["run"]["job_status"] == "DONE(0)", st["run"]
    assert not any("CRASHED" in a["text"] for a in st["alerts"]), st["alerts"]


def test_a_dead_pid_with_no_status_file_is_still_crashed_item2(mon_world: Dict[str, Path]) -> None:  # noqa: F811
    (mon_world["jobs"] / "run.pid").write_text("999999", encoding="utf-8")
    tmon._tick(mon_world)
    st = tmon._status(mon_world)
    assert st["run"]["job_status"] == "CRASHED"
    assert any("CRASHED" in a["text"] for a in st["alerts"] if a["level"] == "HARD")


# ---------------------------------------------------------------------------
# item 8: teardown's final on-pod sync
# ---------------------------------------------------------------------------

def _teardown(tmp_path: Path, target: str) -> tuple[subprocess.CompletedProcess, str]:
    """teardown_pod.sh with a logging ssh that always fails and a logging runpodctl: the
    pull gate cannot pass, so the script aborts before the confirm, the delete and the
    watchdog (no process is touched); step 1 is what these tests read."""
    b = tmp_path / "bin"
    log = tmp_path / "calls.log"
    log.write_text("", encoding="utf-8")
    tm._w(b / "ssh", f'#!/bin/bash\necho "ssh $*" >> "{log}"\nexit 255\n')
    tm._w(b / "runpodctl", f'#!/bin/bash\necho "runpodctl $*" >> "{log}"\necho "[]"\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CAGE_", "RUNPOD_"))}
    env.update({"PATH": f"{b}:{env.get('PATH', '')}", "CAGE_POD_SSH": "root@pod.test", "CAGE_SSH_OPTS": "-p 2222",
                "CAGE_ASSUME_YES": "1", "CAGE_POD_LEDGER": str(tmp_path / "ledger.jsonl")})
    proc = subprocess.run(["bash", str(TEARDOWN), "podtest1", target, str(tmp_path / "dest")],
                          capture_output=True, text=True, env=env, timeout=120, cwd=str(REPO_ROOT))
    return proc, log.read_text(encoding="utf-8")


def test_teardown_skips_the_on_pod_sync_when_the_target_is_the_pod_itself_item8(tmp_path: Path) -> None:
    proc, calls = _teardown(tmp_path, "ssh://root@pod.test/root/backup/results/camp1/a/run1")
    assert "[1/5] final on-pod sync" in proc.stdout
    assert "SKIPPED (the backup target is a path on the pod itself" in proc.stdout, proc.stdout + proc.stderr
    assert "sync_results.sh" not in calls, calls            # no sync was sent to the pod
    assert "WARNING: final on-pod sync FAILED" not in proc.stderr
    # the gate still decides: the pull cannot pass here, so nothing is deleted
    assert proc.returncode == 1 and "ABORT (fail-closed)" in proc.stderr
    assert "pod delete" not in calls


def test_teardown_still_syncs_on_the_pod_for_an_off_box_target_item8(tmp_path: Path) -> None:
    proc, calls = _teardown(tmp_path, f"file://{tmp_path}/no_such_backup")
    assert "SKIPPED (the backup target is a path on the pod itself" not in proc.stdout
    assert "sync_results.sh results" in calls and "-p 2222" in calls, calls
    assert proc.returncode == 1 and "pod delete" not in calls


# ---------------------------------------------------------------------------
# items 11 and 9: stage 14 on a partial tree
# ---------------------------------------------------------------------------

#: verify_results.py's real closing line (scripts/4_analysis/verify_results.py, the
#: "[verify_results] <verdict>: n FAIL, n WARN, n SKIP" print), verdict from the env
FAKE_VERIFY = r'''
import os, sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
(out / "verify_report.md").write_text("# verify\n", encoding="utf-8")
fail = os.environ.get("CAGE_TEST_VERIFY_FAIL") == "1"
if os.environ.get("CAGE_TEST_VERIFY_CRASH") == "1":
    print("Traceback (most recent call last):\n  boom", file=sys.stderr); sys.exit(1)
print(f"[verify_results] {'FAIL' if fail else 'PASS'}: {2 if fail else 0} FAIL, 0 WARN, 57 SKIP")
sys.exit(1 if fail else 0)
'''

#: the analysis driver: CAGE_TEST_ANALYSIS=refuse prints the named refusal of
#: select_contrast_pairs and exits 1, =crash exits 1 with a traceback, else it writes a stamp
FAKE_ANALYSIS = r'''
import json, os, sys
from pathlib import Path
mode = os.environ.get("CAGE_TEST_ANALYSIS", "")
if mode == "refuse":
    print("ERROR: contrast #4 (RAG vs CAG (headline)) needs F1 cells for both B6 and B3; this run's F1 baselines: []", file=sys.stderr); sys.exit(1)
if mode == "no_dataset":
    print("ERROR: contrast #4: 'k1' and 'k2' share no dataset \u2014 nothing to pair", file=sys.stderr); sys.exit(1)
if mode == "no_pair":
    print("ERROR: contrast #4 (RAG vs CAG (headline)): B6 and B3 cells never share the same engine/model axes in this run \u2014 no legal pair", file=sys.stderr); sys.exit(1)
if mode == "ambiguous":
    print("ERROR: contrast #4: axes {} select multiple cells per side (cells=['a', 'b'], refs=['c']) \u2014 the pair is ambiguous; refusing to guess", file=sys.stderr); sys.exit(1)
if mode == "crash":
    print("Traceback (most recent call last):\n  boom", file=sys.stderr); sys.exit(1)
root = Path(sys.argv[1]); stamp = root / "analysis" / "20261006-120000"
stamp.mkdir(parents=True, exist_ok=True)
(stamp / "stats.json").write_text(json.dumps({"mode": "DESIGN-INPUT-ONLY", "contrasts": []}), encoding="utf-8")
(stamp / "summary.md").write_text("# summary\n", encoding="utf-8")
print("analysis done"); sys.exit(0)
'''


def _partial_landing_to_teardown(w: Dict[str, Path], **env: str) -> subprocess.CompletedProcess:
    tm._w(w["scripts"] / "4_analysis" / "verify_results.py", FAKE_VERIFY, exe=False)
    tm._w(w["scripts"] / "4_analysis" / "run_campaign_analysis.py", FAKE_ANALYSIS, exe=False)
    return tm._master(w, "--yes", "provision", "--yes", "teardown", "--to", "teardown", CAGE_TEST_RUN_BUDGET_STOP="1", **env)


def test_a_clean_verify_on_a_partial_tree_is_not_noted_as_fail_findings_item11(world: Dict[str, Path]) -> None:  # noqa: F811
    proc = _partial_landing_to_teardown(world)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    proc = tm._master(world, "--only", "analyze")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "[verify_results] PASS: 0 FAIL" in proc.stdout
    assert "verify_results reports FAIL findings" not in proc.stdout
    assert not any("verify_results" in n for n in tm._state(world)["stages"]["analyze"]["notes"])


def test_fail_findings_on_a_partial_tree_are_noted_and_the_stage_goes_on_item11(world: Dict[str, Path]) -> None:  # noqa: F811
    proc = _partial_landing_to_teardown(world)
    assert proc.returncode == 0, proc.stdout[-3000:]
    proc = tm._master(world, "--only", "analyze", CAGE_TEST_VERIFY_FAIL="1")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "verify_results reports FAIL findings on the PARTIAL tree" in proc.stdout
    assert any("verify_results reported FAIL findings" in n for n in tm._state(world)["stages"]["analyze"]["notes"])


def test_a_contrast_without_cells_on_a_partial_tree_is_a_named_result_item9(world: Dict[str, Path]) -> None:  # noqa: F811
    """The budget stop cut the run before the contrast's cells; the driver's refusal is
    printed and noted, and the panels and analysis.txt are still written."""
    proc = _partial_landing_to_teardown(world)
    assert proc.returncode == 0, proc.stdout[-3000:]
    proc = tm._master(world, "--only", "analyze", CAGE_TEST_ANALYSIS="refuse")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "contrast #4 (RAG vs CAG (headline)) needs F1 cells for both B6 and B3" in proc.stdout
    assert "the PARTIAL tree holds no cells for a registered contrast" in proc.stdout
    st = tm._state(world)
    assert st["stages"]["analyze"]["status"] == "passed"
    assert any("no cells for a registered contrast" in n for n in st["stages"]["analyze"]["notes"])
    assert "stamp" not in st.get("analysis", {})                       # no look was taken
    land = _land(world)
    assert list((land / "plots").glob("panels_*/panels_index.csv"))
    assert (land / "plots" / "analysis.txt").is_file()


def test_any_other_analysis_failure_on_a_partial_tree_fails_the_stage_item9(world: Dict[str, Path]) -> None:  # noqa: F811
    proc = _partial_landing_to_teardown(world)
    assert proc.returncode == 0, proc.stdout[-3000:]
    proc = tm._master(world, "--only", "analyze", CAGE_TEST_ANALYSIS="crash")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] run_campaign_analysis (design input): rc=1 without the named no-cells refusal" in proc.stdout
    assert tm._state(world)["stages"]["analyze"]["status"] == "failed"


def test_the_same_refusal_on_a_complete_tree_fails_the_stage_item9(world: Dict[str, Path]) -> None:  # noqa: F811
    tm._w(world["scripts"] / "4_analysis" / "verify_results.py", FAKE_VERIFY, exe=False)
    tm._w(world["scripts"] / "4_analysis" / "run_campaign_analysis.py", FAKE_ANALYSIS, exe=False)
    proc = tm._master(world, "--yes", "provision", "--yes", "teardown", "--to", "teardown")
    assert proc.returncode == 0, proc.stdout[-3000:]
    proc = tm._master(world, "--only", "analyze", CAGE_TEST_ANALYSIS="refuse")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] run_campaign_analysis (design input): rc=1, expected 0" in proc.stdout


def test_the_driver_still_words_the_no_cells_refusal_the_master_matches_item9() -> None:
    """The master matches the text "needs F1 cells for both"; this pins the driver's side."""
    import run_campaign_analysis as rca
    contrast = rca.CONTRAST_BY_ID[4]                 # the profile default, CONTRASTS="4" in _common.env
    assert contrast.baseline_a and contrast.baseline_b
    index = pd.DataFrame(columns=["row_key", "family", "baseline", "engine", "model", "topology", "policy"])
    with pytest.raises(rca.AnalysisError, match="needs F1 cells for both"):
        rca.select_contrast_pairs(index, contrast)


# ---------------------------------------------------------------------------
# item 4: the pod shape and disk evidence
# ---------------------------------------------------------------------------

def test_the_shape_step_records_cpu_topology_load_and_disk_usage_item4(world: Dict[str, Path]) -> None:  # noqa: F811
    """What comparing two pods needed and the S2R shape file lacked: the CPU model and
    socket layout, the GPU to CPU map, the host load, and where the container disk's bytes
    are, at stage 3 and again at stage 6. A pod without lscpu still passes."""
    proc = tm._master(world, "--yes", "provision", "--to", "plan")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    shape = (_land(world) / "logs" / "system" / "pod_shape.txt").read_text(encoding="utf-8")
    cmd = shape.split("# cmd:")[1].splitlines()[0].replace("\\", "")
    for probe in ("lscpu", "nvidia-smi topo -m", "uptime"):
        assert probe in cmd, (probe, cmd)
    out = [ln for ln in shape.splitlines() if not ln.startswith("#")]
    assert any(ln.startswith("uptime:") for ln in out), out
    assert any("lscpu" in ln for ln in out), out          # its lines, or "lscpu: unavailable" on this host
    system = _land(world) / "logs" / "system"
    for name in ("disk_usage_setup.txt", "disk_usage_plan.txt"):
        text = (system / name).read_text(encoding="utf-8")
        assert "du -x -k -d 2 /" in text.replace("\\", "") and "podhome-usage" in text, text


# ---------------------------------------------------------------------------
# independent review 2026-11-15 (a fresh subagent on the session's model): L1 to L8
# ---------------------------------------------------------------------------

def test_teardown_still_syncs_on_the_pod_for_an_ssh_target_on_another_host_review_l3(tmp_path: Path) -> None:
    """The skip is for the pod's own host only: a mirror kept on another machine is reachable from the pod."""
    proc, calls = _teardown(tmp_path, "ssh://root@other.test/backup/results/camp1/a/run1")
    assert "SKIPPED (the backup target is a path on the pod itself" not in proc.stdout
    assert "sync_results.sh results" in calls, calls


@pytest.mark.parametrize("mode", ["no_dataset", "no_pair"])
def test_the_sibling_no_pair_refusals_on_a_partial_tree_are_named_results_too_review_l1(world: Dict[str, Path], mode: str) -> None:  # noqa: F811
    proc = _partial_landing_to_teardown(world)
    assert proc.returncode == 0, proc.stdout[-3000:]
    proc = tm._master(world, "--only", "analyze", CAGE_TEST_ANALYSIS=mode)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "the PARTIAL tree holds no cells for a registered contrast" in proc.stdout
    assert (_land(world) / "plots" / "analysis.txt").is_file()


def test_an_ambiguous_pair_on_a_partial_tree_still_fails_the_stage_review_l1(world: Dict[str, Path]) -> None:  # noqa: F811
    """A grid defect, not a cut run: two cells on one side of a pair is never a named result."""
    proc = _partial_landing_to_teardown(world)
    assert proc.returncode == 0, proc.stdout[-3000:]
    proc = tm._master(world, "--only", "analyze", CAGE_TEST_ANALYSIS="ambiguous")
    assert proc.returncode == 1 and "without the named no-cells refusal" in proc.stdout, proc.stdout[-3000:]


def test_the_driver_still_words_the_three_refusals_the_master_matches_review_l1() -> None:
    src = (REPO_ROOT / "scripts" / "4_analysis" / "run_campaign_analysis.py").read_text(encoding="utf-8")
    body = src.split("def select_contrast_pairs(")[1].split("\ndef ")[0]
    for token in ("needs F1 cells for both", "share no dataset", "cells never share the same"):
        assert token in body, token


def test_a_verify_crash_on_a_partial_tree_fails_the_stage_review_l6(world: Dict[str, Path]) -> None:  # noqa: F811
    """Exit 1 is expected on a partial tree only with the script's own FAIL verdict line;
    an exit 1 without it (an uncaught exception) is not "the failed cells"."""
    proc = _partial_landing_to_teardown(world)
    assert proc.returncode == 0, proc.stdout[-3000:]
    proc = tm._master(world, "--only", "analyze", CAGE_TEST_VERIFY_CRASH="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] verify_results: rc=1 without its FAIL verdict line on a PARTIAL tree" in proc.stdout
    assert "verify_results reports FAIL findings" not in proc.stdout


#: an lscpu whose summary mixes flat and indented lines (util-linux prints either, by version)
FAKE_LSCPU = r'''
#!/bin/bash
printf 'Architecture:            x86_64\nCPU(s):                  192\nModel name:              INTEL(R) XEON(R) PLATINUM 8558\n  Thread(s) per core:    2\n  Core(s) per socket:    48\n  Socket(s):             2\nNUMA node(s):            2\nVulnerability Meltdown:  Not affected\n'
'''


def test_the_shape_step_filters_lscpu_and_keeps_the_original_probes_exit_review_l4_l5(world: Dict[str, Path]) -> None:  # noqa: F811
    tm._w(world["bin"] / "lscpu", FAKE_LSCPU)
    proc = tm._master(world, "--yes", "provision", "--to", "setup")
    assert proc.returncode == 0, proc.stdout[-3000:]
    out = [ln for ln in (_land(world) / "logs" / "system" / "pod_shape.txt").read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    got = [" ".join(ln.split()) for ln in out if ln.startswith("lscpu:")]
    assert got == ["lscpu: CPU(s): 192", "lscpu: Model name: INTEL(R) XEON(R) PLATINUM 8558", "lscpu: Thread(s) per core: 2",
                   "lscpu: Core(s) per socket: 48", "lscpu: Socket(s): 2", "lscpu: NUMA node(s): 2"], got
    # one of the four original probes failing fails the step with ITS exit code, whatever the added probes return
    proc = tm._master(world, "--from", "setup", "--redo", "--to", "setup", CAGE_TEST_SHAPE_RC="7")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] pod shape evidence: rc=7, expected 0" in proc.stdout


def test_a_du_cut_by_its_timeout_is_said_in_the_record_review_l2(world: Dict[str, Path]) -> None:  # noqa: F811
    proc = tm._master(world, "--yes", "provision", "--to", "setup")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "du_rc=0 " in (_land(world) / "logs" / "system" / "disk_usage_setup.txt").read_text(encoding="utf-8")
    tm._w(world["bin"] / "du", "#!/bin/bash\nprintf '5\\t/partial\\n'\nexit 124\n")
    proc = tm._master(world, "--from", "setup", "--redo", "--to", "setup")
    assert proc.returncode == 0, proc.stdout[-3000:]
    text = (_land(world) / "logs" / "system" / "disk_usage_setup.txt").read_text(encoding="utf-8")
    assert "du_rc=124 (124 = cut at 120 s: the listing is incomplete)" in text, text


def test_a_dead_channel_at_the_disk_usage_record_does_not_fail_the_plan_stage_review_l8(world: Dict[str, Path]) -> None:  # noqa: F811
    """The record is evidence only: stage 6 had no ssh round trip at that point before item 4."""
    proc = tm._master(world, "--yes", "provision", "--to", "plan")
    assert proc.returncode == 0, proc.stdout[-3000:]
    before = len(tm._calls(world))
    proc = tm._master(world, "--only", "plan", "--redo")
    assert proc.returncode == 0, proc.stdout[-3000:]
    ssh_calls = [c for c in tm._calls(world)[before:] if c.startswith("ssh")]
    probe = next(i for i, c in enumerate(ssh_calls) if "du -x -k -d 2" in c)
    (world["home"] / ".ssh_fail_skip").write_text(f"{probe}\n", encoding="utf-8")
    (world["home"] / ".ssh_fail").write_text("3\n", encoding="utf-8")
    proc = tm._master(world, "--only", "plan", "--redo", CAGE_SSH_RETRY_S="0")
    (world["home"] / ".ssh_fail").write_text("0\n", encoding="utf-8")
    (world["home"] / ".ssh_fail_skip").write_text("0\n", encoding="utf-8")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "container disk usage (plan) not recorded (evidence only; the stage goes on)" in proc.stdout
    assert tm._state(world)["stages"]["plan"]["status"] == "passed"
