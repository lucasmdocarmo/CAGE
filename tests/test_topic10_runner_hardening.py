"""Pins for the Topic-10 pre-S0 runner-hardening batch (task #136).

Findings J1-J3, J8-J10 + J12 hygiene from the 2026-08-18 code walkthrough
(MyDocs/registration/CODE_ASSERTION_2026-08.md, "Topic 10"). Same doctrine style
as tests/test_scripts_doctrine.py: static content pins so the fixes cannot
silently regress, plus cheap behavioral checks of the shared helpers under real
bash. Pure local checks: no GPU, no server, no network, no cloud.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
COMMON = SCRIPTS / "lib" / "_common.sh"

RUNNERS = {
    "baselines": SCRIPTS / "3_run" / "run_baselines.sh",
    "compression": SCRIPTS / "3_run" / "run_compression.sh",
    "memory_sweep": SCRIPTS / "3_run" / "run_memory_sweep.sh",
    "kv_store": SCRIPTS / "3_run" / "run_kv_store.sh",
    "prefix_envelope": SCRIPTS / "3_run" / "run_prefix_envelope.sh",
}
ORCHESTRATORS = {
    "full_sweep": SCRIPTS / "3_run" / "run_full_sweep.sh",
    "cloud_run": SCRIPTS / "3_run" / "cloud_run.sh",
}


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _bash(script: str, env: dict | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )


def _func_body(text: str, name: str) -> str:
    # The `{` line may carry a trailing usage comment (repo idiom), hence [^\n]*.
    m = re.search(rf"^{re.escape(name)}\(\) \{{[^\n]*\n(.*?)^\}}", text, re.M | re.S)
    assert m, f"function {name}() not found"
    return m.group(1)


def _code_lines(text: str) -> list[str]:
    """Non-comment lines (leading-# stripped out) -- for pins that must ignore prose."""
    return [l for l in text.splitlines() if not l.lstrip().startswith("#")]


# ---------------------------------------------------------------------------
# J1 (CRITICAL): server-start helpers must propagate a restart failure.
# ---------------------------------------------------------------------------

def test_j1_baselines_helpers_propagate_restart_failure_statically() -> None:
    """Both helpers used to end with `sleep 10`, returning sleep's 0 even when
    the restart FAILED -- making all four SERVER-FAIL -> mark_cells_failed
    branches unreachable dead code (a dead engine proceeded into cells)."""
    text = _text(RUNNERS["baselines"])
    for fn in ("start_server_without_prefix_cache", "start_server_with_prefix_cache"):
        body = _func_body(text, fn)
        restart_lines = [l for l in body.splitlines() if "manage_vllm_server.sh restart" in l]
        assert restart_lines, f"{fn}: no restart line found"
        for line in restart_lines:
            assert "|| return 1" in line, (
                f"{fn}: the restart must propagate failure (`|| return 1`), or the "
                f"trailing settle-wait makes the helper return 0 on a DEAD engine (J1): {line!r}"
            )
        assert "sleep" in body, f"{fn}: the settle wait after a SUCCESSFUL restart must stay"


def test_j1_helper_pattern_returns_nonzero_when_restart_fails(tmp_path: Path) -> None:
    """Behavioral: run the ACTUAL helper functions (extracted verbatim, sleep
    shortened) against a stub manage_vllm_server.sh -- restart rc must become
    the helper's rc, so the SERVER-FAIL branches are reachable again."""
    text = _text(RUNNERS["baselines"])
    fns = "\n".join(
        re.search(rf"^{fn}\(\) \{{\n.*?^\}}", text, re.M | re.S).group(0)  # type: ignore[union-attr]
        for fn in ("start_server_without_prefix_cache", "start_server_with_prefix_cache")
    ).replace("sleep 10", "sleep 0")
    stub_dir = tmp_path / "scripts" / "2_serving"
    stub_dir.mkdir(parents=True)
    stub = stub_dir / "manage_vllm_server.sh"
    stub.write_text('#!/bin/bash\nexit "${STUB_RC:-0}"\n', encoding="utf-8")
    stub.chmod(0o755)
    harness = f"""
set -uo pipefail
cd "{tmp_path}"
MODEL=test-model
{fns}
start_server_without_prefix_cache >/dev/null && echo A_OK || echo A_FAIL
start_server_with_prefix_cache banner >/dev/null && echo B_OK || echo B_FAIL
"""
    ok = _bash(harness, env={"STUB_RC": "0"})
    assert ok.returncode == 0 and "A_OK" in ok.stdout and "B_OK" in ok.stdout, ok.stdout + ok.stderr
    bad = _bash(harness, env={"STUB_RC": "1"})
    assert "A_FAIL" in bad.stdout and "B_FAIL" in bad.stdout, (
        "helpers returned 0 for a FAILED restart -- the J1 dead-engine bug is back:\n"
        + bad.stdout + bad.stderr
    )


# ---------------------------------------------------------------------------
# J2 (MAJOR): resume gates must JSON-validate metrics.json, not just stat it.
# ---------------------------------------------------------------------------

def test_j2_all_five_runners_validate_metrics_json_in_cell_complete() -> None:
    for name, path in RUNNERS.items():
        body = _func_body(_text(path), "cell_complete")
        assert "metrics_json_valid" in body, (
            f"{name}: cell_complete must call metrics_json_valid -- a bare -f check "
            "makes a corrupt/truncated/foreign metrics.json resume-proof 'complete' (J2)"
        )
        assert '[ -f "' not in body, (
            f"{name}: cell_complete still contains a bare existence check (J2)"
        )


def test_j2_metrics_json_valid_behavior(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text('{"ok": 1}', encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text('{"truncated": ', encoding="utf-8")
    script = f"""
set -uo pipefail
source "{COMMON}"
metrics_json_valid "{good}" && echo GOOD_OK
metrics_json_valid "{bad}" 2>/dev/null && echo BAD_OK || echo BAD_REJECTED
metrics_json_valid "{tmp_path}/missing.json" && echo MISSING_OK || echo MISSING_REJECTED
_loud="$(metrics_json_valid "{bad}" 2>&1 >/dev/null || true)"
case "$_loud" in *WARNING*) echo LOUD ;; esac
"""
    proc = _bash(script)
    for token in ("GOOD_OK", "BAD_REJECTED", "MISSING_REJECTED", "LOUD"):
        assert token in proc.stdout, f"expected {token}: {proc.stdout}\n{proc.stderr}"
    assert "BAD_OK" not in proc.stdout and "MISSING_OK" not in proc.stdout


# ---------------------------------------------------------------------------
# J3 (MAJOR): no shared static root; flock per run root; unique run-ids;
#             dataset in cell identity.
# ---------------------------------------------------------------------------

def test_j3_no_shared_static_run_root_anywhere() -> None:
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for p in sorted(SCRIPTS.rglob("*.sh"))
        if "deprecated" not in p.parts
        and any("results/phase2/local" in l for l in _code_lines(_text(p)))
    ]
    assert not offenders, (
        "shared static run root results/phase2/local is back (J3: two standalone "
        f"runs interleave/wipe each other's cells): {offenders}"
    )


def test_j3_every_runner_and_orchestrator_acquires_the_run_lock() -> None:
    for name, path in {**RUNNERS, **ORCHESTRATORS}.items():
        assert re.search(r"^\s*acquire_run_lock ", _text(path), re.M), (
            f"{name}: must acquire_run_lock on its run root (J3: a second resume "
            "instance can rm -rf a live cell)"
        )


def test_j3_acquire_run_lock_contract_in_common_sh() -> None:
    body = _func_body(_text(COMMON), "acquire_run_lock")
    assert "flock -n" in body, "the lock must be NON-blocking (exit loudly, never queue)"
    assert ".cage_run.lock" in body
    assert "CAGE_RUN_LOCK_HELD" in body, (
        "re-entrancy sentinel gone: run_full_sweep -> cloud_run -> run_baselines "
        "would deadlock-by-refusal on their shared root"
    )
    assert "die " in body, "a held lock must fail LOUD (die), not warn-and-proceed"


def test_j3_lock_reentrancy_under_parent(tmp_path: Path) -> None:
    """With CAGE_RUN_LOCK_HELD naming the same lockfile (the orchestrator-child
    chain), acquisition is a no-op success -- no flock needed for this path."""
    script = f"""
set -uo pipefail
source "{COMMON}"
export CAGE_RUN_LOCK_HELD="{tmp_path}/root/.cage_run.lock"
acquire_run_lock "{tmp_path}/root" >/dev/null 2>&1 && echo REENTRANT_OK
"""
    proc = _bash(script)
    assert "REENTRANT_OK" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(shutil.which("flock") is None, reason="flock(1) not on PATH (macOS)")
def test_j3_second_instance_is_refused_while_lock_held(tmp_path: Path) -> None:
    script = f"""
set -uo pipefail
source "{COMMON}"
acquire_run_lock "{tmp_path}/root" >/dev/null 2>&1 || exit 90
( unset CAGE_RUN_LOCK_HELD
  acquire_run_lock "{tmp_path}/root" >/dev/null 2>&1 ) && echo SECOND_ACQUIRED || echo SECOND_REFUSED
"""
    proc = _bash(script)
    assert "SECOND_REFUSED" in proc.stdout, (
        "a second instance acquired the held run lock (J3): " + proc.stdout + proc.stderr
    )


def test_j3_mint_run_id_seconds_random_and_dataset(tmp_path: Path) -> None:
    script = f"""
set -uo pipefail
source "{COMMON}"
for i in 1 2 3 4 5; do mint_run_id qwen3-8b 500 3 squad_v2; echo; done
"""
    proc = _bash(script)
    ids = [l for l in proc.stdout.splitlines() if l.strip()]
    assert len(ids) == 5, proc.stdout + proc.stderr
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_qwen3-8b_500x3_[0-9a-f]{4}_squad_v2$")
    for rid in ids:
        assert pat.match(rid), f"run-id format drifted (need seconds + 4-hex random + dataset LAST): {rid}"
    assert len(set(ids)) > 1, "5 mints collided -- the random suffix is gone (J3 converge hazard)"


def test_j3_no_minute_granular_minting_left_in_runners() -> None:
    for p in sorted((SCRIPTS / "3_run").glob("*.sh")):
        assert "date +%Y-%m-%d_%H%M)" not in _text(p), (
            f"{p.name}: minute-granular run-id minting is back (J3 fragment/converge "
            "hazard) -- use mint_run_id"
        )


def test_j3_minting_scripts_use_mint_run_id() -> None:
    for name, path in {**RUNNERS, **ORCHESTRATORS}.items():
        if name in ("memory_sweep", "kv_store", "prefix_envelope", "baselines", "compression",
                    "full_sweep", "cloud_run"):
            assert "mint_run_id " in _text(path), f"{name}: fresh roots must come from mint_run_id"


def test_j3_status_sentinels_carry_dataset_identity() -> None:
    for name, path in RUNNERS.items():
        for line in _text(path).splitlines():
            if 'echo "STATUS=' in line and "model=$MODEL" in line:
                assert "dataset=$DATASET" in line, (
                    f"{name}: STATUS sentinel omits the dataset (J3 cell identity): {line.strip()}"
                )


# ---------------------------------------------------------------------------
# J8 (MINOR): trap/orphan cluster.
# ---------------------------------------------------------------------------

def test_j8_compression_has_exit_trap_stopping_server_and_chaining_log_guard() -> None:
    text = _text(RUNNERS["compression"])
    traps = re.findall(r"^\s*trap\s+(.+?)\s+EXIT\s*$", text, re.M)
    assert traps and traps[-1] == "cleanup", (
        "run_compression.sh needs `trap cleanup EXIT` as its LAST EXIT trap (J8: "
        f"a signal left the vLLM server holding the GPU); got {traps}"
    )
    body = _func_body(text, "cleanup")
    stop_at = body.find("manage_vllm_server.sh stop")
    lg_at = body.find("__lg_cleanup")
    assert stop_at != -1, "cleanup() must stop the vLLM server"
    assert lg_at != -1, (
        "cleanup() must chain __lg_cleanup: `trap cleanup EXIT` REPLACED the "
        "log-guard's EXIT trap (bash keeps ONE handler per signal)"
    )
    assert stop_at < lg_at, "server stop must precede the log-guard chain"


def test_j8_memory_sweep_interim_trap_chains_both_daemon_stops() -> None:
    text = _text(RUNNERS["memory_sweep"])
    interim = [
        t for t in re.findall(r"^\s*trap\s+'(.+?)'\s+EXIT\s*$", text, re.M)
        if "gcs_backup_daemon.sh stop" in t
    ]
    assert interim, "interim gcs-daemon stop trap missing from run_memory_sweep.sh"
    for t in interim:
        assert "__lg_cleanup" in t, (
            "the interim trap must ALSO chain __lg_cleanup (J8): registering it "
            "already replaced the log-guard's trap, so an exit in that window "
            "orphaned the log_sync daemon"
        )


def test_j8_cloud_run_kills_the_sync_process_group_and_waits() -> None:
    text = _text(ORCHESTRATORS["cloud_run"])
    assert 'setsid bash -c "$_SYNC_LOOP"' in text, (
        "the periodic syncer must run in its OWN process group (setsid) so cleanup "
        "can kill the in-flight grandchild gsutil/gcloud (J8)"
    )
    body = _func_body(text, "cleanup")
    assert re.search(r'kill\s+-TERM\s+--\s+"-\$SYNC_PID"', body), (
        "cleanup() must kill the syncer's WHOLE process group (kill -TERM -- -$SYNC_PID)"
    )
    assert 'wait "$SYNC_PID"' in body, "cleanup() must wait the syncer before the final sync"
    assert "pkill -TERM -P" in body, "the no-setsid fallback must still reap the loop's children"


def test_j8_detached_helpers_close_the_run_lock_fd() -> None:
    for rel in (
        "scripts/3_run/cloud_run.sh",
        "scripts/5_observability/gcs_backup_daemon.sh",
        "scripts/lib/_log_guard.sh",
    ):
        assert "200>&-" in _text(REPO_ROOT / rel), (
            f"{rel}: detached daemon launches must close fd 200 (the J3 run-lock fd) "
            "-- a surviving daemon would otherwise hold the run lock forever"
        )


def test_j8_daemon_liveness_is_identity_checked() -> None:
    cases = {
        "scripts/5_observability/gcs_backup_daemon.sh": "pidfile_alive",
        "scripts/5_observability/log_sync_daemon.sh": 'pidfile_alive "$PIDF" "log_sync_daemon.sh"',
        "scripts/5_observability/run_status_logger.sh": 'pidfile_alive "$PIDF" "run_status_logger.sh"',
    }
    for rel, needle in cases.items():
        assert needle in _text(REPO_ROOT / rel), (
            f"{rel}: pid_alive must delegate to the identity-checked pidfile_alive "
            "(J8: a recycled pid must never be treated as -- or KILLED as -- our daemon)"
        )
    # the stop path of the gcs daemon must consult the identity check before killing
    stop_body = _func_body(_text(REPO_ROOT / "scripts/5_observability/gcs_backup_daemon.sh"), "stop_pidfile")
    assert "pid_alive" in stop_body, "stop_pidfile must identity-check before kill (PID reuse)"


def test_j8_pidfile_alive_behavior(tmp_path: Path) -> None:
    pf = tmp_path / "d.pid"
    script = f"""
set -uo pipefail
source "{COMMON}"
sleep 30 & sp=$!
echo "$sp" > "{pf}"
pidfile_alive "{pf}" "sleep" && echo MATCH_OK || echo MATCH_FAIL
pidfile_alive "{pf}" "definitely_not_this_daemon.sh" && echo REUSE_OK || echo REUSE_REJECTED
kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
echo 99999999 > "{pf}"
pidfile_alive "{pf}" "sleep" && echo DEAD_OK || echo DEAD_REJECTED
rm -f "{pf}"
pidfile_alive "{pf}" "sleep" && echo GONE_OK || echo GONE_REJECTED
"""
    proc = _bash(script)
    for token in ("MATCH_OK", "REUSE_REJECTED", "DEAD_REJECTED", "GONE_REJECTED"):
        assert token in proc.stdout, f"expected {token}: {proc.stdout}\n{proc.stderr}"


# ---------------------------------------------------------------------------
# J9 (MINOR): forensics env dump must redact secret VALUES, keep NAMES.
# ---------------------------------------------------------------------------

def test_j9_collect_logs_env_dump_is_redacted_statically() -> None:
    text = _text(REPO_ROOT / "scripts/5_observability/collect_logs.sh")
    assert "***REDACTED***" in text and "TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL" in text, (
        "the env forensics dump must pipe through the secret-redaction awk (J9: "
        "HUGGING_FACE_HUB_TOKEN / VLLM_API_KEY were uploaded VERBATIM to the bucket)"
    )
    dump = re.search(r"env \| grep -iE \"VLLM\|CAGE\|CUDA\|REDIS\|HF_HOME\|HUGGING\"(.*?)cage_env\.txt", text, re.S)
    assert dump and "***REDACTED***" in dump.group(1), (
        "the redaction must sit BETWEEN the env grep and cage_env.txt -- a redactor "
        "elsewhere in the file does not protect this dump"
    )


def test_j9_redaction_pipeline_behavior() -> None:
    text = _text(REPO_ROOT / "scripts/5_observability/collect_logs.sh")
    m = re.search(r"awk -F= '([^']+)'", text)
    assert m, "cannot extract the redaction awk program"
    awk_prog = m.group(1)
    proc = _bash(
        f"env | grep -iE 'VLLM|CAGE|CUDA|REDIS|HF_HOME|HUGGING' | sort | awk -F= '{awk_prog}'",
        env={
            "HUGGING_FACE_HUB_TOKEN": "hf_supersecret123",
            "VLLM_API_KEY": "sk-alsoverysecret",
            "CAGE_PHASE": "phase2",
        },
    )
    out = proc.stdout
    assert "hf_supersecret123" not in out and "sk-alsoverysecret" not in out, (
        "secret VALUES leaked through the redaction pipeline: " + out
    )
    assert "HUGGING_FACE_HUB_TOKEN=***REDACTED***" in out
    assert "VLLM_API_KEY=***REDACTED***" in out
    assert "CAGE_PHASE=phase2" in out, "non-secret values must survive unredacted"


# ---------------------------------------------------------------------------
# J10 (MINOR): teardown skip ceremony + extended read-only $0 sweep.
# ---------------------------------------------------------------------------

def test_j10_skip_local_pull_needs_typed_ceremony_and_marker() -> None:
    text = _text(REPO_ROOT / "scripts/6_teardown/teardown_vm.sh")
    branch = re.search(r'if \[ "\$\{CAGE_SKIP_LOCAL_PULL:-0\}" = "1" \]; then(.*?)\nelse', text, re.S)
    assert branch, "the CAGE_SKIP_LOCAL_PULL branch is gone"
    body = branch.group(1)
    assert 'CAGE_SKIP_LOCAL_PULL_CONFIRM' in body and "I-ACCEPT-DATA-LOSS" in body, (
        "skipping the pull gate must require the SECOND ceremony "
        "CAGE_SKIP_LOCAL_PULL_CONFIRM=I-ACCEPT-DATA-LOSS (J10: =1 alone silently "
        "bypassed the fail-closed gate)"
    )
    assert "exit 1" in body, "an unconfirmed skip must ABORT (exit 1), fail-closed"
    assert "PULL_BYPASSED_" in body, "a confirmed skip must record a bypass marker file"


def test_j10_zero_dollar_sweep_covers_disks_addresses_buckets_readonly() -> None:
    text = _text(REPO_ROOT / "scripts/6_teardown/teardown_vm.sh")
    tail = text[text.index("[6/6]"):]
    for needle in ("instances list", "disks list", "addresses list", "buckets list"):
        assert needle in tail, f"$0 sweep must include read-only `{needle}` (J10)"
    gcloud_cmds = [l for l in tail.splitlines() if l.strip().startswith("gcloud")]
    assert gcloud_cmds, "expected actual gcloud listing commands in the [6/6] sweep"
    for cmd in gcloud_cmds:
        assert "delete" not in cmd, f"the [6/6] sweep must stay REPORT-ONLY: {cmd.strip()}"


# ---------------------------------------------------------------------------
# J12 (hygiene): .DS_Store untracked+ignored; run_tests decoupled; retirement.
# ---------------------------------------------------------------------------

def _tracked_files() -> list[str]:
    if not (REPO_ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout / git unavailable")
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        capture_output=True, text=True, check=True,
    ).stdout
    return out.splitlines()


def test_j12_no_tracked_ds_store_and_ignore_rule_exists() -> None:
    tracked = [l for l in _tracked_files() if l.endswith(".DS_Store")]
    assert not tracked, f".DS_Store files are tracked again (J12): {tracked}"
    gitignore = _text(REPO_ROOT / ".gitignore")
    assert re.search(r"^\.DS_Store\s*$", gitignore, re.M), (
        ".gitignore must carry an UNanchored .DS_Store rule"
    )


def test_j12_run_tests_default_path_is_local_pytest_no_cluster() -> None:
    text = _text(SCRIPTS / "checks" / "run_tests.sh")
    assert "--with-cluster" in text and "CAGE_TESTS_WITH_CLUSTER" in text, (
        "GPU/cluster mode must be an explicit opt-in flag (J12)"
    )
    assert "-m pytest tests/" in text, "the default path must run pytest"
    starts = [m.start() for m in re.finditer(r"manage_vllm_cluster\.py start", text)]
    assert len(starts) == 1, "expected exactly one gated cluster start"
    gate = text.index('if [ "$WITH_CLUSTER" = "1" ]')
    assert starts[0] > gate, (
        "the cluster start must live INSIDE the --with-cluster gate -- the old "
        "version hard-gated the whole suite behind it and ZERO tests ran locally"
    )


def test_j12_simulate_network_retired_to_deprecated() -> None:
    old = SCRIPTS / "checks" / "simulate_network.sh"
    new = SCRIPTS / "deprecated" / "simulate_network.sh"
    assert not old.exists(), "simulate_network.sh must not live in scripts/checks/ anymore"
    assert new.is_file(), "simulate_network.sh must be preserved under scripts/deprecated/"
    text = _text(new)
    assert "DEPRECATED" in text, "the deprecation stamp is missing"
    assert "scripts/deprecated/README.md" in text
    assert "simulate_network.sh" in _text(SCRIPTS / "deprecated" / "README.md"), (
        "scripts/deprecated/README.md must list the retirement"
    )
    tracked = _tracked_files()
    assert "scripts/deprecated/simulate_network.sh" in tracked, (
        "the retired script must stay TRACKED at its new path (git mv, not plain mv)"
    )
