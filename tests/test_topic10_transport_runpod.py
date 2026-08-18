"""Pins for the Topic-10 RunPod-primary transport batch (task #137, J4 + J7).

Findings J4 (gcloud/gsutil/metadata-hard transports degrading SILENTLY off-GCP;
shared last_gcs_sync_ok marker masking failures) and J7 (GCP-DLVM-shaped setup,
half-landed HF-timeout fix, pilot dataset/model rosters) from the 2026-08-18
code walkthrough (MyDocs/registration/CODE_ASSERTION_2026-08.md, "Topic 10").
Owner decision (verbatim): "runpod is the cloud now" — RunPod is PRIMARY, GCP
is a retained portability backend behind scripts/lib/transport.sh.

Same doctrine style as tests/test_scripts_doctrine.py and
tests/test_topic10_runner_hardening.py: static content pins so the fixes cannot
silently regress, plus behavioral checks under real bash. The gcs/s3/ssh
backends are tested via CAGE_TRANSPORT_DRYRUN=1 (exact command construction,
offline); the file:// backend is tested for REAL against tmpdirs. No GPU, no
server, no network, no cloud.
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
TRANSPORT = SCRIPTS / "lib" / "transport.sh"
SYNC = SCRIPTS / "5_observability" / "sync_results.sh"
SYNC_SHIM = SCRIPTS / "5_observability" / "sync_results_to_gcs.sh"
PULL = SCRIPTS / "5_observability" / "pull_run.sh"
COLLECT = SCRIPTS / "5_observability" / "collect_logs.sh"
BACKUP_DAEMON = SCRIPTS / "5_observability" / "gcs_backup_daemon.sh"
LOG_SYNC_DAEMON = SCRIPTS / "5_observability" / "log_sync_daemon.sh"
LOG_GUARD = SCRIPTS / "lib" / "_log_guard.sh"
WATCH_CAMPAIGN = SCRIPTS / "5_observability" / "watch_campaign.sh"
CLOUD_RUN = SCRIPTS / "3_run" / "cloud_run.sh"
MEMORY_SWEEP = SCRIPTS / "3_run" / "run_memory_sweep.sh"
STATS_SH = SCRIPTS / "4_analysis" / "run_phase2_stats.sh"
TEARDOWN_POD = SCRIPTS / "6_teardown" / "teardown_pod.sh"
TEARDOWN_VM = SCRIPTS / "6_teardown" / "teardown_vm.sh"
SETUP_RUNPOD = SCRIPTS / "1_setup" / "setup_runpod.sh"
SETUP_GCP = SCRIPTS / "1_setup" / "setup_gpu_cloud.sh"
DOWNLOAD_DATASETS = SCRIPTS / "1_setup" / "download_datasets.py"

# Charter D5 staging roster (MyDocs/PUBLICATION.md, DECIDED 2026-07-27):
# 3 locality sets + qasper + the SCBench 2-subset slice + the ShareGPT load
# donor. RULER is GENERATED (never downloaded); CRAG is CITE-ONLY (D5#8).
CHARTER_STAGED_KEYS = {"squad_v2", "hotpotqa", "musique", "qasper", "scbench", "sharegpt"}

# Env vars that would let a target leak into "no target configured" tests.
_TARGET_ENV_VARS = (
    "CAGE_BACKUP_TARGET", "CAGE_RESULTS_BUCKET", "CAGE_ALLOW_NO_BACKUP",
    "CAGE_TRANSPORT_DRYRUN", "CAGE_TRANSPORT_GCS_TOOL", "GOOGLE_CLOUD_PROJECT",
    "CAGE_AGENT_DIR",
)


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _code_lines(text: str) -> list[str]:
    """Non-comment lines — for pins that must ignore prose."""
    return [l for l in text.splitlines() if not l.lstrip().startswith("#")]


def _clean_env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _TARGET_ENV_VARS}
    env.update(extra)
    return env


def _bash(script: str, env: dict | None = None, cwd: Path | None = None,
          timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True, text=True, timeout=timeout,
        env=env if env is not None else dict(os.environ),
        cwd=str(cwd) if cwd else None,
    )


def _transport(snippet: str, env: dict | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return _bash(f'set -uo pipefail; source "{TRANSPORT}"; {snippet}', env=env, cwd=cwd)


# ---------------------------------------------------------------------------
# transport.sh — resolution table (scheme -> backend)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,backend", [
    ("gs://bucket", "gcs"),
    ("gs://bucket/prefix/run", "gcs"),
    ("s3://volume/prefix", "s3"),
    ("ssh://root@1.2.3.4/workspace/backup", "ssh"),
    ("ssh://host.example/data", "ssh"),
    ("file:///mnt/volume/backup", "local"),
    ("/mnt/volume/backup", "local"),
])
def test_transport_resolution_table(target: str, backend: str) -> None:
    proc = _transport(f'transport_resolve "{target}"', env=_clean_env())
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == backend


@pytest.mark.parametrize("bad", ["http://x/y", "gcs://bucket", "bucket-name", ""])
def test_transport_unknown_scheme_dies_loud(bad: str) -> None:
    proc = _transport(f'transport_resolve "{bad}"', env=_clean_env())
    assert proc.returncode != 0, f"'{bad}' must be REFUSED, got: {proc.stdout}"
    assert "FATAL" in proc.stderr, proc.stderr


def test_transport_defines_full_api() -> None:
    text = _text(TRANSPORT)
    for fn in ("transport_resolve()", "transport_join()", "transport_push()",
               "transport_pull()", "transport_ls()", "transport_exists()",
               "transport_ensure()", "transport_default_target()",
               "require_backup_target()"):
        assert fn in text, f"transport.sh no longer defines {fn}"


# ---------------------------------------------------------------------------
# Dry-run command construction (gcs / s3 / ssh — offline)
# ---------------------------------------------------------------------------

def test_dryrun_gcs_gcloud_command_construction() -> None:
    env = _clean_env(CAGE_TRANSPORT_DRYRUN="1", CAGE_TRANSPORT_GCS_TOOL="gcloud")
    push = _transport('transport_push /tmp/src gs://bucket/run', env=env)
    assert push.returncode == 0, push.stderr
    assert "DRYRUN: gcloud storage rsync -r /tmp/src gs://bucket/run" in push.stdout
    pull = _transport('transport_pull gs://bucket/run /tmp/dst', env=env)
    assert "DRYRUN: gcloud storage rsync -r gs://bucket/run /tmp/dst" in pull.stdout
    ls = _transport('transport_ls gs://bucket/run', env=env)
    assert "DRYRUN: gcloud storage ls -r gs://bucket/run" in ls.stdout
    exists = _transport('transport_exists gs://bucket/run', env=env)
    assert "DRYRUN: gcloud storage ls gs://bucket/run" in exists.stdout


def test_dryrun_gcs_gsutil_fallback_keeps_checksum_flag() -> None:
    env = _clean_env(CAGE_TRANSPORT_DRYRUN="1", CAGE_TRANSPORT_GCS_TOOL="gsutil")
    push = _transport('transport_push /tmp/src gs://bucket/run', env=env)
    # -c: checksum compare (a truncated partial upload must never become permanent).
    assert "DRYRUN: gsutil -m rsync -c -r /tmp/src gs://bucket/run" in push.stdout
    pull = _transport('transport_pull gs://bucket/run /tmp/dst', env=env)
    assert "DRYRUN: gsutil -m rsync -r gs://bucket/run /tmp/dst" in pull.stdout


def test_dryrun_s3_endpoint_construction() -> None:
    env = _clean_env(CAGE_TRANSPORT_DRYRUN="1",
                     CAGE_S3_ENDPOINT="https://s3api-eu-ro-1.runpod.io")
    push = _transport('transport_push /tmp/src s3://vol/run', env=env)
    assert ("DRYRUN: aws s3 sync /tmp/src s3://vol/run "
            "--endpoint-url https://s3api-eu-ro-1.runpod.io") in push.stdout
    pull = _transport('transport_pull s3://vol/run /tmp/dst', env=env)
    assert ("DRYRUN: aws s3 sync s3://vol/run /tmp/dst "
            "--endpoint-url https://s3api-eu-ro-1.runpod.io") in pull.stdout
    # Without an endpoint the flag must vanish entirely (plain AWS S3).
    env_no = _clean_env(CAGE_TRANSPORT_DRYRUN="1")
    env_no.pop("CAGE_S3_ENDPOINT", None)
    push_no = _transport('transport_push /tmp/src s3://vol/run', env=env_no)
    assert "DRYRUN: aws s3 sync /tmp/src s3://vol/run" in push_no.stdout
    assert "--endpoint-url" not in push_no.stdout


def test_dryrun_ssh_command_construction() -> None:
    env = _clean_env(CAGE_TRANSPORT_DRYRUN="1", CAGE_SSH_OPTS="-p 2222")
    push = _transport('transport_push /tmp/src ssh://root@1.2.3.4/workspace/backup', env=env)
    assert push.returncode == 0, push.stderr
    out = push.stdout
    assert "rsync -az" in out
    assert "root@1.2.3.4:/workspace/backup/" in out
    assert "ssh -p 2222" in out, "CAGE_SSH_OPTS must reach the rsync -e transport (RunPod non-standard ports)"
    pull = _transport('transport_pull ssh://root@1.2.3.4/workspace/backup /tmp/dst', env=env)
    assert "root@1.2.3.4:/workspace/backup/ /tmp/dst/" in pull.stdout
    # A host-only ssh target (no path) must die, not silently rsync to $HOME.
    bad = _transport('transport_push /tmp/src ssh://root@1.2.3.4', env=env)
    assert bad.returncode != 0
    assert "no path component" in bad.stderr


# ---------------------------------------------------------------------------
# file:// backend — REAL round-trip against tmpdirs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not on PATH")
def test_local_backend_real_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "cells").mkdir(parents=True)
    (src / "manifest.json").write_text('{"run": 1}', encoding="utf-8")
    (src / "cells" / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    dst = tmp_path / "backup"

    env = _clean_env()
    push = _transport(f'transport_push "{src}" "file://{dst}"', env=env)
    assert push.returncode == 0, push.stderr
    assert (dst / "manifest.json").read_text(encoding="utf-8") == '{"run": 1}'
    assert (dst / "cells" / "a.csv").is_file()

    # Incremental second push picks up a new file (mirror, not one-shot).
    (src / "cells" / "b.csv").write_text("x,y\n3,4\n", encoding="utf-8")
    assert _transport(f'transport_push "{src}" "file://{dst}"', env=env).returncode == 0
    assert (dst / "cells" / "b.csv").is_file()

    ls = _transport(f'transport_ls "file://{dst}"', env=env)
    assert ls.returncode == 0
    for name in ("manifest.json", "a.csv", "b.csv"):
        assert name in ls.stdout, f"transport_ls must list {name}: {ls.stdout}"

    assert _transport(f'transport_exists "file://{dst}"', env=env).returncode == 0
    assert _transport(f'transport_exists "file://{tmp_path}/nope"', env=env).returncode != 0

    pulled = tmp_path / "pulled"
    pull = _transport(f'transport_pull "file://{dst}" "{pulled}"', env=env)
    assert pull.returncode == 0, pull.stderr
    assert (pulled / "cells" / "b.csv").read_text(encoding="utf-8") == "x,y\n3,4\n"

    # Plain absolute path (no file:// scheme) is the same backend.
    dst2 = tmp_path / "backup2"
    assert _transport(f'transport_push "{src}" "{dst2}"', env=env).returncode == 0
    assert (dst2 / "manifest.json").is_file()


# ---------------------------------------------------------------------------
# require_backup_target — the J4 refusal gate (build item (c))
# ---------------------------------------------------------------------------

def test_require_backup_target_refuses_without_target(tmp_path: Path) -> None:
    proc = _transport(f'require_backup_target "{tmp_path}/run"', env=_clean_env())
    assert proc.returncode != 0, "a run with NO backup target must REFUSE to start (J4)"
    assert "no backup target" in proc.stderr
    assert "CAGE_ALLOW_NO_BACKUP" in proc.stderr, "the refusal must name the explicit override"
    assert not (tmp_path / "run" / "NO_BACKUP_OVERRIDE").exists()


def test_require_backup_target_override_records_marker(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    proc = _transport(f'require_backup_target "{run_root}"',
                      env=_clean_env(CAGE_ALLOW_NO_BACKUP="1"))
    assert proc.returncode == 0, proc.stderr
    marker = run_root / "NO_BACKUP_OVERRIDE"
    assert marker.is_file(), "the override must be recorded durably in the run root"
    body = marker.read_text(encoding="utf-8")
    assert "CAGE_ALLOW_NO_BACKUP=1" in body
    assert "epoch=" in body
    assert "NO off-box persistence" in proc.stderr, "the override must be LOUD, not silent"


def test_require_backup_target_resolves_and_normalizes(tmp_path: Path) -> None:
    ok = _transport(f'require_backup_target "{tmp_path}/run"',
                    env=_clean_env(CAGE_BACKUP_TARGET="file:///mnt/vol/backup"))
    assert ok.returncode == 0 and ok.stdout.strip() == "file:///mnt/vol/backup"
    legacy = _transport(f'require_backup_target "{tmp_path}/run"',
                        env=_clean_env(CAGE_RESULTS_BUCKET="my-bucket"))
    assert legacy.returncode == 0 and legacy.stdout.strip() == "gs://my-bucket"
    # With a real target, no override marker appears.
    assert not (tmp_path / "run" / "NO_BACKUP_OVERRIDE").exists()


def test_cloud_run_wired_through_refusal_gate() -> None:
    text = _text(CLOUD_RUN)
    assert re.search(r'^\s*(?:source|\.)\s+[^#\n]*transport\.sh', text, re.M), (
        "cloud_run.sh must source scripts/lib/transport.sh"
    )
    gate = text.find('require_backup_target "$CAGE_RUN_ROOT"')
    assert gate != -1, "cloud_run.sh must call require_backup_target on its run root (J4 build item (c))"
    loop = text.find("_SYNC_LOOP=")
    assert loop != -1 and gate < loop, (
        "the refusal gate must run BEFORE the background sync loop is launched"
    )
    assert 'export CAGE_BACKUP_TARGET=' in text, (
        "cloud_run.sh must export the resolved target so children (sync/collect/manifest) inherit it"
    )


def test_run_manifest_echoes_backup_override() -> None:
    text = _text(SCRIPTS / "5_observability" / "observe_run.py")
    assert '"backup_target"' in text and '"no_backup_override"' in text, (
        "observe_run.py must echo the backup target + CAGE_ALLOW_NO_BACKUP override "
        "into run_manifest.json (J4 build item (c))"
    )


# ---------------------------------------------------------------------------
# sync_results.sh — loud degradation + per-backend markers (J4)
# ---------------------------------------------------------------------------

def test_sync_results_dies_without_target(tmp_path: Path) -> None:
    proc = _bash(f'bash "{SYNC}" data', env=_clean_env(), cwd=tmp_path)
    assert proc.returncode != 0, (
        "sync with NO resolvable target must EXIT NONZERO (J4: the old script "
        "could exit 0 with nothing synced)"
    )
    assert "no backup target" in proc.stderr


def test_sync_results_override_skips_loudly(tmp_path: Path) -> None:
    proc = _bash(f'bash "{SYNC}" data', env=_clean_env(CAGE_ALLOW_NO_BACKUP="1"), cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "SKIPPING off-box sync" in proc.stderr, "the skip must be announced, never silent"


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not on PATH")
def test_sync_results_local_real_sync_and_per_backend_markers(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "metrics.json").write_text("{}", encoding="utf-8")
    agent = tmp_path / "agent"
    env = _clean_env(CAGE_BACKUP_TARGET=f"file://{tmp_path}/backup",
                     CAGE_AGENT_DIR=str(agent))
    proc = _bash(f'bash "{SYNC}" data', env=env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "backup" / "data" / "metrics.json").is_file(), (
        "the REMOTE_SUBPATH default must mirror LOCAL_DIR under the target"
    )
    ok_marker = agent / "last_sync_ok_local"
    assert ok_marker.is_file(), "success marker must be PER-BACKEND (J4 shared-marker masking)"
    assert "backend=local" in ok_marker.read_text(encoding="utf-8")
    assert not (agent / "last_gcs_sync_ok").exists(), (
        "a non-gcs sync must NOT refresh the gcs marker — that is exactly the "
        "masking J4 flagged"
    )


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not on PATH")
def test_sync_results_failure_writes_failure_marker_and_propagates(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "metrics.json").write_text("{}", encoding="utf-8")
    blocker = tmp_path / "blockfile"
    blocker.write_text("not a directory", encoding="utf-8")
    agent = tmp_path / "agent"
    env = _clean_env(CAGE_BACKUP_TARGET=f"file://{blocker}/sub",
                     CAGE_AGENT_DIR=str(agent))
    proc = _bash(f'bash "{SYNC}" data', env=env, cwd=tmp_path)
    assert proc.returncode != 0, "a FAILED transfer must exit nonzero (no '|| true' masking)"
    assert "sync FAILED" in proc.stderr
    fail_marker = agent / "last_sync_fail_local"
    assert fail_marker.is_file(), "failure marker must be written, distinct from success"
    assert "rc=" in fail_marker.read_text(encoding="utf-8")
    assert not (agent / "last_sync_ok_local").exists()


def test_sync_shim_forwards_to_canonical() -> None:
    text = _text(SYNC_SHIM)
    assert "DEPRECATED" in text
    assert re.search(r'^exec bash "\$SCRIPT_DIR/sync_results\.sh" "\$@"$', text, re.M), (
        "the old name must FORWARD (exec) to sync_results.sh with identical args"
    )
    # Behavioral: the shim really execs the canonical script (override path, no cloud).
    proc = _bash(f'bash "{SYNC_SHIM}" data', env=_clean_env(CAGE_ALLOW_NO_BACKUP="1"))
    assert proc.returncode == 0, proc.stderr
    assert "DEPRECATED name" in proc.stderr
    assert "SKIPPING off-box sync" in proc.stderr, "canonical behavior must shine through the shim"


# ---------------------------------------------------------------------------
# No-silent-degradation pins (J4): no '|| true' on any sync path
# ---------------------------------------------------------------------------

def test_sync_results_itself_has_no_or_true() -> None:
    for line in _code_lines(_text(SYNC)):
        assert "|| true" not in line, f"sync_results.sh must never mask a failure: {line!r}"


def test_callers_announce_sync_failures_instead_of_or_true() -> None:
    """Every sync/collect invocation on the persistence path must either
    propagate the failure or ANNOUNCE it — `|| true` is the J4 bug."""
    offenders = []
    for path in (CLOUD_RUN, LOG_SYNC_DAEMON, LOG_GUARD, STATS_SH, COLLECT):
        for line in _code_lines(_text(path)):
            if ("sync_results.sh" in line or "collect_logs.sh" in line) and "|| true" in line:
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, "sync-path failures silently swallowed again (J4):\n" + "\n".join(offenders)


def test_no_caller_still_uses_the_deprecated_sync_name() -> None:
    """All live callers go through the canonical sync_results.sh; only the shim
    itself and the standalone GCP metadata hook may keep the old name."""
    allowed = {SYNC_SHIM.name, "gcp_shutdown_hook.sh"}
    offenders = []
    for path in sorted(SCRIPTS.rglob("*.sh")):
        if "deprecated" in path.parts or path.name in allowed:
            continue
        for line in _code_lines(_text(path)):
            if "sync_results_to_gcs.sh" in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {line.strip()}")
    assert not offenders, "callers must use sync_results.sh (shim is compat-only):\n" + "\n".join(offenders)


def test_memory_sweep_daemon_start_is_fail_closed() -> None:
    text = _text(MEMORY_SWEEP)
    line = next((l for l in text.splitlines() if "gcs_backup_daemon.sh start" in l), None)
    assert line is not None
    assert "|| true" not in line, (
        "the daemon start must not be `|| true`-swallowed (J4: a sweep could run "
        "with zero off-box persistence silently)"
    )
    start_at = text.find("gcs_backup_daemon.sh start")
    assert "die" in text[start_at:start_at + 400], "a failed daemon start must abort the sweep"


def test_backup_daemon_start_refuses_without_target() -> None:
    text = _text(BACKUP_DAEMON)
    assert "CAGE_BACKUP_TARGET" in text, "the daemon must accept the provider-neutral target"
    # Behavioral: unset target -> die (nonzero), unless the explicit override.
    refuse = _bash(f'bash "{BACKUP_DAEMON}" start', env=_clean_env())
    assert refuse.returncode != 0, (
        "daemon start with NO target must FAIL (J4: warn+exit-0 let sweeps run bare)"
    )
    assert "no backup target" in refuse.stderr
    allow = _bash(f'bash "{BACKUP_DAEMON}" start', env=_clean_env(CAGE_ALLOW_NO_BACKUP="1"))
    assert allow.returncode == 0, allow.stderr
    assert "NO off-box backup" in allow.stderr


def test_watch_campaign_reads_per_backend_markers() -> None:
    text = _text(WATCH_CAMPAIGN)
    assert "last_sync_ok_" in text, (
        "watch_campaign.sh must include the per-backend markers in its sync-lag scan "
        "(non-GCS backends were invisible to it)"
    )


# ---------------------------------------------------------------------------
# pull_run.sh — provider-neutral pull, same ledger gate (J4)
# ---------------------------------------------------------------------------

def test_pull_run_is_provider_neutral_statically() -> None:
    text = _text(PULL)
    assert re.search(r'^\s*(?:source|\.)\s+[^#\n]*transport\.sh', text, re.M)
    assert "transport_pull" in text, "the transfer must go through the transport library"
    assert "s3://*" in text and "ssh://*" in text and "file://*" in text, (
        "pull_run.sh must accept s3:// / ssh:// / file:// targets (J4: gsutil-only "
        "meant a RunPod campaign had NO pull path)"
    )
    # The discipline gates stay EXACTLY: DO-NOT-TEARDOWN trap + ledger verify.
    assert "DO NOT TEARDOWN" in text
    assert "SAFE TO TEARDOWN" in text
    assert "verify_ledger" in text


@pytest.mark.skipif(not (REPO_ROOT / ".venv" / "bin" / "python").exists(),
                    reason="repo venv required for ledger verification")
@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not on PATH")
def test_pull_run_file_backend_end_to_end_with_ledger(tmp_path: Path) -> None:
    """REAL pull over the file:// backend: sealed tree pulls clean -> SAFE TO
    TEARDOWN; a tampered artifact -> DO NOT TEARDOWN, nonzero."""
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

    dest = tmp_path / "local_copy"
    ok = _bash(f'bash "{PULL}" "file://{remote}" "{dest}"', env=_clean_env())
    assert ok.returncode == 0, f"stdout:\n{ok.stdout}\nstderr:\n{ok.stderr}"
    assert "SAFE TO TEARDOWN" in ok.stdout
    assert (dest / "cells" / "a.csv").is_file()

    (remote / "cells" / "a.csv").write_text("TAMPERED\n", encoding="utf-8")
    bad = _bash(f'bash "{PULL}" "file://{remote}" "{tmp_path}/copy2"', env=_clean_env())
    assert bad.returncode != 0
    assert "DO NOT TEARDOWN" in bad.stderr
    assert "SAFE TO TEARDOWN" not in bad.stdout


# ---------------------------------------------------------------------------
# teardown_pod.sh — RunPod teardown, verified-pull-first (build item (d))
# ---------------------------------------------------------------------------

def test_teardown_pod_ordering_and_fail_closed_doctrine() -> None:
    text = _text(TEARDOWN_POD)
    code = "\n".join(_code_lines(text))
    pull_at = code.find("pull_run.sh")
    confirm_at = code.find('confirm "Delete RunPod pod')
    delete_at = code.find("runpodctl remove pod")
    listing_at = code.find("runpodctl get pod")
    assert -1 not in (pull_at, confirm_at, delete_at, listing_at), (
        f"missing step: pull={pull_at} confirm={confirm_at} delete={delete_at} list={listing_at}"
    )
    assert pull_at < confirm_at < delete_at < listing_at, (
        "fail-closed ordering (same as teardown_vm.sh): verified pull FIRST, then "
        "confirm ceremony, then delete, then read-only $0 listing"
    )
    assert "SAFE TO TEARDOWN" in text, "the gate must check pull_run's literal authorization line"
    assert "ABORT (fail-closed)" in text
    assert "DATA MAY BE LOST" in text, "--force must announce possible data loss"
    assert "rm -rf" not in text, "teardown must not delete local data"
    # The [5/5] $0 proof is READ-ONLY: no destructive call after it.
    tail = text[text.index("[5/5]"):]
    assert "remove pod" not in tail and "-X DELETE" not in tail, (
        "the $0 sweep must stay report-only"
    )


def test_teardown_pod_usage_and_pull_gate_behavior(tmp_path: Path) -> None:
    no_args = _bash(f'bash "{TEARDOWN_POD}"', env=_clean_env())
    assert no_args.returncode == 2
    assert "usage:" in no_args.stderr
    # A failing pull gate must ABORT before any deletion is attempted (no
    # runpodctl/API is touched: the abort happens first). file:// target that
    # does not exist -> pull_run fails -> fail-closed abort.
    env = _clean_env()
    env.pop("CAGE_POD_SSH", None)
    proc = _bash(
        f'bash "{TEARDOWN_POD}" pod-123 "file://{tmp_path}/does-not-exist" "{tmp_path}/dest"',
        env=env,
    )
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "ABORT (fail-closed)" in proc.stderr
    assert "TEARDOWN_COMPLETE" not in proc.stdout


def test_teardown_vm_retained_as_gcp_port() -> None:
    text = _text(TEARDOWN_VM)
    assert "GCP PORT" in text and "teardown_pod.sh" in text, (
        "teardown_vm.sh must be labeled as the retained GCP port pointing at the "
        "RunPod-primary teardown"
    )


# ---------------------------------------------------------------------------
# setup_runpod.sh — container-shaped bootstrap (J7, build item (e))
# ---------------------------------------------------------------------------

def _line_no(text: str, pattern: str) -> int:
    for i, line in enumerate(text.splitlines(), start=1):
        if re.search(pattern, line):
            return i
    return -1


@pytest.mark.parametrize("setup", [SETUP_RUNPOD, SETUP_GCP], ids=["runpod", "gcp_port"])
def test_hf_timeout_exported_before_dataset_staging(setup: Path) -> None:
    """J7: the dataset-hang fix was half-landed — HF_HUB_DOWNLOAD_TIMEOUT was
    exported AFTER the dataset step. It must precede BOTH staging and prefetch."""
    text = _text(setup)
    export_at = _line_no(text, r"^export HF_HUB_DOWNLOAD_TIMEOUT")
    stage_at = _line_no(text, r"^\s*python\s+scripts/1_setup/download_datasets\.py")
    # Match the actual heredoc invocation, not prose comments about the hang.
    prefetch_at = _line_no(text, r"^\s*from huggingface_hub import snapshot_download")
    assert export_at != -1, f"{setup.name}: HF_HUB_DOWNLOAD_TIMEOUT export missing"
    assert stage_at != -1, f"{setup.name}: dataset staging step missing"
    assert prefetch_at != -1, f"{setup.name}: model prefetch missing"
    assert export_at < stage_at, (
        f"{setup.name}: HF_HUB_DOWNLOAD_TIMEOUT (line {export_at}) must be exported "
        f"BEFORE dataset staging (line {stage_at}) — finding J7"
    )
    assert export_at < prefetch_at


def test_setup_runpod_stages_full_charter_roster() -> None:
    text = _text(SETUP_RUNPOD)
    m = re.search(r'CHARTER_DATASETS="\$\{CHARTER_DATASETS:-([^}]+)\}"', text)
    assert m, "setup_runpod.sh must declare an overridable CHARTER_DATASETS roster"
    staged = set(m.group(1).split())
    assert staged == CHARTER_STAGED_KEYS, (
        f"staged roster {sorted(staged)} != charter D5 stageable set "
        f"{sorted(CHARTER_STAGED_KEYS)} (RULER is generated; CRAG is cite-only)"
    )
    # Every staged key must be a real download_datasets.py key.
    dl = _text(DOWNLOAD_DATASETS)
    for key in staged:
        assert f'"{key}"' in dl, f"staged key {key} unknown to download_datasets.py"
    # The charter exclusions stay excluded from the staging loop.
    assert "crag" not in staged and "ruler" not in staged


def test_setup_runpod_prefetches_final_scope_models() -> None:
    text = _text(SETUP_RUNPOD)
    m = re.search(r'PREFETCH_MODELS="\$\{PREFETCH_MODELS:-([^}]+)\}"', text)
    assert m, "setup_runpod.sh must declare an overridable PREFETCH_MODELS roster"
    roster = set(m.group(1).split())
    assert roster == {"Qwen/Qwen3-14B", "meta-llama/Llama-3.3-70B-Instruct"}, (
        f"FINAL SCOPE v2 roster is Qwen3-14B (anchor, Session A + PD overlay) + "
        f"Llama-3.3-70B (Session B scale); got {sorted(roster)} — the pilot-era "
        "Qwen3-8B/MiMo/EAGLE roster must be gone (J7)"
    )


def test_setup_runpod_is_container_shaped() -> None:
    """J7: no DLVM ceremony inside a RunPod container — no sudo, no systemd,
    no deadsnakes PPA; canonical interpreter still fail-closed (B1)."""
    text = _text(SETUP_RUNPOD)
    code = _code_lines(text)
    for forbidden in ("sudo ", "systemctl", "add-apt-repository", "deadsnakes"):
        hits = [l for l in code if forbidden in l]
        assert not hits, f"container-shaped setup must not use {forbidden!r}: {hits}"
    assert 'PYBIN="python${CAGE_CANONICAL_PYTHON}"' in text, (
        "setup_runpod.sh must derive its interpreter from CAGE_CANONICAL_PYTHON (B1)"
    )
    assert re.search(r"^\s*python3 -m venv", text, re.M) is None, (
        "the venv must be created with the canonical interpreter, never bare python3"
    )
    assert re.search(r'^\s*"\$PYBIN" -m venv cage-env', text, re.M), (
        "the cage-env venv must come from $PYBIN"
    )
    assert "pip install -r requirements.txt" in text, "the repo's pinned manifest must be installed"
    assert "cage_stats.api" in text, "the standard verify step must close the bootstrap"


def test_setup_gcp_labeled_as_port_and_runpod_primary() -> None:
    text = _text(SETUP_GCP)
    assert "setup_runpod.sh" in text and "PRIMARY" in text, (
        "setup_gpu_cloud.sh must carry the header note: RunPod is primary, this "
        "script is the retained GCP port (owner decision 2026-08-16)"
    )


# ---------------------------------------------------------------------------
# #139 — charter dataset roster reconciliation (download_datasets.py)
# ---------------------------------------------------------------------------

def _load_download_datasets_module():
    import importlib.util
    import sys
    import types

    # download_datasets imports `datasets` at module top; stub it so the pin
    # runs offline (the loader function bodies are not executed here).
    if "datasets" not in sys.modules:
        stub = types.ModuleType("datasets")
        stub.load_dataset = lambda *a, **k: None  # noqa: ARG005
        sys.modules["datasets"] = stub
    spec = importlib.util.spec_from_file_location(
        "cage_download_datasets_pin",
        REPO_ROOT / "scripts" / "1_setup" / "download_datasets.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_campaign_keys_match_charter_roster_exactly() -> None:
    """#139: `--dataset all` stages EXACTLY the charter D5 downloadable set —
    the same six keys setup_runpod.sh stages (RULER generated, CRAG cite-only,
    trivia_qa/natural_questions pilot-era)."""
    mod = _load_download_datasets_module()
    assert set(mod.CAMPAIGN_KEYS) == CHARTER_STAGED_KEYS, (
        f"CAMPAIGN_KEYS {sorted(mod.CAMPAIGN_KEYS)} must equal the charter "
        f"roster {sorted(CHARTER_STAGED_KEYS)} (#139)"
    )


def test_pilot_extras_are_separate_and_still_stageable() -> None:
    """#139: pilot-era extras live in PILOT_EXTRA_KEYS — disjoint from the
    charter roster, excluded from `all`, but each still has a specs entry so
    explicit `--dataset <key>` staging keeps working for pilot rescores."""
    mod = _load_download_datasets_module()
    extras = set(mod.PILOT_EXTRA_KEYS)
    assert extras == {"crag", "natural_questions", "trivia_qa"}
    assert not (extras & set(mod.CAMPAIGN_KEYS)), "extras must not leak into `all`"
    specs = mod.dataset_specs()
    for key in extras:
        assert key in specs, f"pilot extra {key!r} must remain explicitly stageable"


def test_gate_p_known_roster_tightens_to_charter() -> None:
    """#139: preflight gate (p) derives its known set from CAMPAIGN_KEYS —
    after the reconciliation a campaign run requesting a pilot extra refuses."""
    mod = _load_download_datasets_module()
    known = set(mod.CAMPAIGN_KEYS) | set(mod.CALIBRATION_KEYS) | {"ruler"}
    for key in mod.PILOT_EXTRA_KEYS:
        assert key not in known, (
            f"{key!r} must be OUTSIDE gate (p)'s charter roster so a campaign "
            "request for it refuses loudly"
        )
