"""scripts/6_experiments/cage_experiment.sh (ADR-0143): the staged master.

Design section 9, tests 1, 2, 4 and 7: stage order and state, the money gate,
the landing layout, the header contract. The master runs against FAKES:
runpodctl, ssh, scp, setsid, nvidia-smi, curl and stat on a temporary PATH; a
fake scripts tree under CAGE_SCRIPTS_DIR for the Mac-side scripts; a fake pod
repo (the profile's POD_REPO) whose scripts the fake ssh runs locally under a
temporary HOME. Every fake records its argv, prints the marker lines the real
script prints, and produces the files the next stage reads, so the whole
sequence 0 to 14 runs on the Mac with no network, no pod and no GPU.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPDIR = REPO_ROOT / "scripts" / "6_experiments"
MASTER = EXPDIR / "cage_experiment.sh"
STAGES = ["preflight-mac", "provision", "ship", "setup", "validate", "calibrate", "plan", "run",
          "monitor", "seal", "score", "collect", "pull", "analyze", "teardown"]
DATE = "2026-10-06"
# The child shells that source the shipped profiles get PATH only. Under the
# master the whole profile is exported (set -a), so an inherited EXP made the
# _common.env case print "S0" and failed stage 0 on the S0 day (2026-10-07).
# The files are the subject of those tests, never the caller's environment.
PROFILE_ENV = {"PATH": os.environ.get("PATH", "")}

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


def _w(path: Path, body: str, exe: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    if exe:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


FAKE_SSH = r'''
#!/bin/bash
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
echo "ssh $host :: $*" >> "$CAGE_TEST_LOG"
export HOME="$CAGE_TEST_HOME"
cd "$HOME"
exec bash -c "$*"
'''

FAKE_SCP = r'''
#!/bin/bash
set -u
args=(); rec=0
while [ $# -gt 0 ]; do
  case "$1" in
    -P|-i|-o) shift 2 ;;
    -r) rec=1; shift ;;
    -*) shift ;;
    *) args+=("$1"); shift ;;
  esac
done
echo "scp ${args[*]}" >> "$CAGE_TEST_LOG"
resolve() { p="${1#*:}"; case "$p" in /*) echo "$p" ;; *) echo "$CAGE_TEST_HOME/$p" ;; esac; }
src="${args[0]}"; dst="${args[1]}"
case "$src" in *:*) src="$(resolve "$src")" ;; esac
case "$dst" in *:*) dst="$(resolve "$dst")" ;; esac
if [ -d "$src" ]; then mkdir -p "$dst"; cp -R "$src"/. "$dst"/; else mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; fi
'''

FAKE_RUNPODCTL = r'''
#!/bin/bash
echo "runpodctl $*" >> "$CAGE_TEST_LOG"
cmd="${1:-}"; sub="${2:-}"
case "$cmd" in
  version) echo "runpodctl 2.14.0-test" ;;
  user) echo '{"email": "owner@example.test", "balance": 324.5}' ;;
  pod)
    case "$sub" in
      list) echo "${CAGE_TEST_PODS:-[]}" ;;
      get) echo "{\"id\": \"$3\", \"runtimeStatus\": \"${CAGE_TEST_RUNTIME:-running}\", \"desiredStatus\": \"RUNNING\"}" ;;
      *) echo "{}" ;;
    esac ;;
  network-volume)
    case "$sub" in
      list) echo "${CAGE_TEST_VOLUMES:-[]}" ;;
      create) echo '{"id": "vol123", "name": "cage-test", "dataCenterId": "US-NE-1"}' ;;
      delete) echo '{"deleted": true}' ;;
    esac ;;
  gpu) echo '[{"gpuId": "NVIDIA H100 80GB HBM3", "securePricePerHr": 3.49, "dataCenterAvailability": [{"dataCenterId": "US-NE-1", "stockStatus": "High"}, {"dataCenterId": "EU-RO-1", "stockStatus": "none"}, {"dataCenterId": "US-KS-2", "stockStatus": "none"}]}]' ;;
  ssh) if [ -n "${CAGE_TEST_SSH_INFO_BAD:-}" ]; then echo "ssh info: command + key"; else echo '{"ip": "pod.test", "port": 2222, "user": "root"}'; fi ;;
  billing) echo "[]" ;;
  *) echo "{}" ;;
esac
exit 0
'''


@pytest.fixture()
def world(tmp_path: Path) -> Dict[str, Path]:
    """The fake Mac + pod world the master runs in."""
    log = tmp_path / "calls.log"
    log.write_text("", encoding="utf-8")
    home = tmp_path / "podhome"; home.mkdir()
    pod_repo = tmp_path / "podrepo"
    backup = tmp_path / "backup"
    exp_root = tmp_path / "experiments"
    fake_scripts = tmp_path / "fake_scripts"
    ledger = tmp_path / "pod_ledger.jsonl"
    freeze = tmp_path / "freeze_resolutions.json"
    freeze.write_text(json.dumps({"QASPER_TAU": "0.5", "INSTRUMENT_REVISIONS": {}}), encoding="utf-8")

    # --- stub binaries ------------------------------------------------------
    b = tmp_path / "bin"
    _w(b / "ssh", FAKE_SSH)
    _w(b / "scp", FAKE_SCP)
    _w(b / "runpodctl", FAKE_RUNPODCTL)
    _w(b / "setsid", "#!/bin/sh\nexec \"$@\"\n")
    _w(b / "curl", "#!/bin/sh\nexit 0\n")
    _w(b / "nvidia-smi", r'''
        #!/bin/bash
        case "$*" in
          *compute-apps*) : ;;
          *query-gpu*) echo "3, 1000, 81559" ;;
          *) echo "fake nvidia-smi -q" ;;
        esac
        ''')
    # GNU stat -c %Y on the fake pod (the Mac's stat is BSD)
    _w(b / "stat", r'''
        #!/bin/bash
        if [ "$1" = "-c" ]; then
          fmt="$2"; shift 2
          if /usr/bin/stat -c %Y / >/dev/null 2>&1; then exec /usr/bin/stat -c "$fmt" "$@"; fi
          [ "$fmt" = "%Y" ] && exec /usr/bin/stat -f %m "$@"
        fi
        exec /usr/bin/stat "$@"
        ''')

    # --- the fake Mac-side scripts tree -----------------------------------
    _w(fake_scripts / "checks" / "run_tests.sh", "#!/bin/bash\necho \"run_tests $*\" >> \"$CAGE_TEST_LOG\"; echo '4000 passed'; exit 0\n")
    _w(fake_scripts / "ops" / "package_repo.sh", r'''
        #!/bin/bash
        echo "package_repo $*" >> "$CAGE_TEST_LOG"
        out="${1:-/tmp/x.tar.gz}"; d="$(mktemp -d)"
        printf 'sha=%s\ndirty=%s\npackaged_at=2026-10-06T00:00:00Z\n' "${CAGE_TEST_SHA:-0123456789abcdef0123456789abcdef01234567}" "${CAGE_TEST_DIRTY:-0}" > "$d/BUILD_INFO"
        tar czf "$out" -C "$d" BUILD_INFO
        echo "PACKAGED  $out"
        ''')
    _w(fake_scripts / "runpod" / "provision_pod.sh", r'''
        #!/bin/bash
        echo "provision_pod $*" >> "$CAGE_TEST_LOG"
        [ -z "${CAGE_TEST_PROVISION_RC:-}" ] || exit "$CAGE_TEST_PROVISION_RC"
        case " $* " in
          *" --yes "*)
            echo '{"id": "pod123"}'
            printf '{"ts_utc":"2026-10-06T10:00:00Z","pod_id":"pod123","price_per_hour_usd": 3.49,"event":"create"}\n' >> "$CAGE_POD_LEDGER"
            [ -z "${CAGE_TEST_NO_SEATBELT:-}" ] || echo "[cage] WARNING: SEATBELT NOT ARMED: pod_watchdog.sh arm failed"
            echo "[cage] pod CREATED: pod123   (seatbelt: 24h)" ;;
          *) echo "[cage] =================== RunPod provisioning PLAN ==================="; echo "  price             : \$3.49/h"; echo "[cage] PLAN ONLY: nothing was created and nothing is billing (run-approval gate)." ;;
        esac
        ''')
    _w(fake_scripts / "runpod" / "pod_watchdog.sh", "#!/bin/bash\necho \"pod_watchdog $*\" >> \"$CAGE_TEST_LOG\"; echo \"pod=$2 watchdog=ALIVE pid=1 deadline=2099-01-01T00:00:00Z log=/dev/null\"\n")
    _w(fake_scripts / "runpod" / "teardown_pod.sh", r'''
        #!/bin/bash
        echo "teardown_pod $* ASSUME_YES=${CAGE_ASSUME_YES:-} SSH=${CAGE_POD_SSH:-}" >> "$CAGE_TEST_LOG"
        echo "=== SAFE TEARDOWN (RunPod): pod=$1 ==="
        echo "[1/5] final on-pod sync"; echo "[2/5] verified pull"; echo "SAFE TO TEARDOWN"; echo "[3/5] confirm ceremony"
        echo "[4/5] deleting pod $1 ... (cost-stopping action)"
        echo "[5/5] confirming \$0 (read-only pod + network-volume listings) ..."
        echo "[teardown_pod] WARNING: the network-volume listing above is NOT empty" >&2
        exit "${CAGE_TEST_TEARDOWN_RC:-1}"
        ''')
    _w(fake_scripts / "runpod" / "cost_report.sh", "#!/bin/bash\necho \"cost_report $*\" >> \"$CAGE_TEST_LOG\"; echo 'POD_ID  HOURS  COST'; echo 'pod123 1.0 3.49'; exit 0\n")
    _w(fake_scripts / "5_observability" / "pull_run.sh", r'''
        #!/bin/bash
        echo "pull_run $* SSH_OPTS=${CAGE_SSH_OPTS:-}" >> "$CAGE_TEST_LOG"
        target="$1"; dest="$2"
        path="${target#ssh://}"; path="/${path#*/}"
        [ -d "$path" ] || { echo "LEDGER-MISSING: $path" >&2; exit 2; }
        mkdir -p "$dest"; cp -R "$path"/. "$dest"/
        echo "[pull_run] [3/3] ledger intact"
        echo "SAFE TO TEARDOWN"
        ''')
    _w(fake_scripts / "3_run" / "run_campaign.py", r'''
        class PlanError(Exception):
            pass
        def get_session_grid(session):
            if session not in ("a", "b"):
                raise PlanError(f"session {session!r} is not a registered grid")
            return {"session": session}
        ''', exe=False)
    _w(fake_scripts / "4_analysis" / "build_floor_table.py", r'''
        import json, sys
        from pathlib import Path
        args = sys.argv[1:]
        p = Path(args[args.index("--out") + 1])
        if p.exists() and "--force" not in args:
            print("REFUSED: exists", file=sys.stderr); sys.exit(2)
        p.write_text(json.dumps({"schema": "floor-table-v1", "argv": args}), encoding="utf-8")
        ''', exe=False)
    _w(fake_scripts / "4_analysis" / "verify_results.py", r'''
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        out = Path(args[args.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
        (out / "verify_report.md").write_text("# verify\n0 FAIL, 1 WARN\nWARN: a mini manifest\n", encoding="utf-8")
        print("verify PASS"); sys.exit(0)
        ''', exe=False)
    _w(fake_scripts / "4_analysis" / "organize_results.py", r'''
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        root = Path(args[0]); idx = root / "index"
        if idx.exists() and "--force" not in args:
            print("refusing to overwrite", file=sys.stderr); sys.exit(1)
        idx.mkdir(parents=True, exist_ok=True)
        (idx / "cells_index.csv").write_text("row_key,engine,model,dataset,family,window\nk1,vllm,qwen3-14b,squad_v2,F1,window_squad_v2-01\nk2,hf,qwen3-14b,squad_v2,F1,window_squad_v2-01\n", encoding="utf-8")
        (idx / "coverage_report.md").write_text("# coverage\nMISSING: B3 vllm (floor)\n", encoding="utf-8")
        (idx / "provenance.json").write_text("{}", encoding="utf-8")
        print("organized"); sys.exit(0)
        ''', exe=False)
    _w(fake_scripts / "4_analysis" / "run_campaign_analysis.py", r'''
        import json, sys
        from pathlib import Path
        args = sys.argv[1:]
        root = Path(args[0]); stamp = root / "analysis" / "20261006-120000"
        stamp.mkdir(parents=True, exist_ok=True)
        (stamp / "stats.json").write_text(json.dumps({"mode": "DESIGN-INPUT-ONLY", "contrasts": [{"id": 4, "metric": "ttft_ms", "n": 50}], "figures": [{"kind": "forest", "file": "forest_ttft_ms.png"}]}), encoding="utf-8")
        (stamp / "summary.md").write_text("# summary\ncontrast 4: ttft_ms, n=50\n", encoding="utf-8")
        (stamp / "forest_ttft_ms.png").write_bytes(b"\x89PNG fake")
        print("analysis done", args); sys.exit(0)
        ''', exe=False)
    _w(fake_scripts / "4_analysis" / "render_window_panels.py", r'''
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        out = Path(args[args.index("--out") + 1]); out.mkdir(parents=True)
        (out / "panels_index.csv").write_text("cell,window\nk1,window_squad_v2-01\n", encoding="utf-8")
        print("panels", args); sys.exit(0)
        ''', exe=False)

    # --- the fake pod repo (run by the fake ssh, locally) -------------------
    pr = pod_repo
    (pr / "logs" / "vllm").mkdir(parents=True)
    (pr / "logs" / "vllm" / "vllm_x.log").write_text("fake engine log\n", encoding="utf-8")
    (pr / "results" / "calibration").mkdir(parents=True)
    (pr / "data" / "manifests").mkdir(parents=True)
    # the pod's venv python: answers the master's torch/vllm shape probe itself
    # (the Mac venv has no vllm), runs everything else with the real interpreter
    _w(pr / "cage-env" / "bin" / "python", f'''
        #!/bin/bash
        case "$*" in
          *"import torch, vllm"*) echo "torch 2.10.0 cuda 12.8 vllm 0.19.1"; exit "${{CAGE_TEST_SHAPE_RC:-0}}" ;;
        esac
        exec {sys.executable} "$@"
        ''')
    _w(pr / "scripts" / "runpod" / "setup_runpod.sh", r'''
        #!/bin/bash
        echo "setup_runpod CHARTER_DATASETS=$CHARTER_DATASETS PREFETCH_MODELS=$PREFETCH_MODELS" >> "$CAGE_TEST_LOG"
        echo "[cage]   cage-env: link"
        echo "[cage]   all charter datasets staged: $CHARTER_DATASETS"
        for m in $PREFETCH_MODELS; do echo "[cage]   $m: cached"; done
        echo "[cage]   pynvml OK -> GPU memory-pressure telemetry WILL be captured"
        echo "[cage]   cage_stats.api import OK -> serving telemetry available"
        [ -z "${CAGE_TEST_SETUP_WARN:-}" ] || echo "[cage] WARNING: dataset stage FAILED: qasper"
        [ -z "${CAGE_TEST_SETUP_NOTE_BAD:-}" ] || echo "[cage]   NOTE: cage_stats.api not importable (No module named httpx); set CAGE_STATS_HOME"
        echo "[cage]  RunPod bootstrap complete. Next (docs/RUNBOOK.md lifecycle):"
        echo "[cage]  NOTE: harness trees carry no ledger.json until the campaign driver seals them"
        ''')
    for eng in ("vllm", "sglang", "lmdeploy"):
        _w(pr / "scripts" / "2_serving" / f"manage_{eng}_server.sh", f"#!/bin/bash\necho \"launcher {eng} $*\" >> \"$CAGE_TEST_LOG\"; echo \"{eng} $1 ok\"; exit 0\n")
    _w(pr / "scripts" / "checks" / "preflight_check.sh", "#!/bin/bash\necho \"preflight $* BACKENDS=$CAGE_PREFLIGHT_BACKENDS\" >> \"$CAGE_TEST_LOG\"; echo 'PREFLIGHT PASS -- all Gate-2 components green.'; exit \"${CAGE_TEST_PREFLIGHT_RC:-0}\"\n")
    _w(pr / "scripts" / "3_run" / "calibrate_cell.py", r'''
        import json, sys
        from pathlib import Path
        a = sys.argv[1:]; out = Path(a[a.index("--output") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"procedure_version": "cal-v2 (2026-09-30)", "start_qps_source": "floor-service-rate", "lambda_star": 2.5, "argv": a}), encoding="utf-8")
        print("[calibrate] wrote", out); sys.exit(0)
        ''', exe=False)
    _w(pr / "scripts" / "3_run" / "run_campaign.py", r'''
        import json, os, sys
        from pathlib import Path
        a = sys.argv[1:]
        if a[0] == "plan":
            out = Path(a[a.index("--out") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
            plan = {"schema": "cage-campaign-plan-v5", "session": a[a.index("--session") + 1], "argv": a,
                    "steps": [{"kind": "relaunch", "engine": "vllm", "argv": ["manage_vllm_server.sh", "start"]},
                              {"kind": "cell", "engine": "vllm", "row_key": "k1", "argv": ["run_experiment.py", "--vllm-telemetry"]},
                              {"kind": "cell", "engine": "hf", "row_key": "k2", "argv": ["run_cag_reference.py"]}],
                    "blocked_row_keys": ["k9"] if os.environ.get("CAGE_TEST_BLOCKED") else []}
            if os.environ.get("CAGE_TEST_BAD_PLAN"):
                plan["steps"][1]["argv"] = ["run_experiment.py"]
            out.write_text(json.dumps(plan), encoding="utf-8"); print("plan written"); sys.exit(0)
        if a[0] == "run":
            root = Path(a[a.index("--campaign-root") + 1])
            for k in ("k1", "k2"):
                w = root / "cells" / k / "window_squad_v2-01"; w.mkdir(parents=True, exist_ok=True)
                (w / "regime.json").write_text('{"label": "KV_PRESSURE_OK"}', encoding="utf-8")
                (w / "metrics.json").write_text("{}", encoding="utf-8")
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "write_time_hashes.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "observability" / "serving_configs").mkdir(parents=True, exist_ok=True)
            (root / "observability" / "serving_configs" / "x_vllm.json").write_text(json.dumps({"engine": "vllm", "gpu_memory_utilization": 0.9, "kv_pool_bytes_realized": 5713920000}), encoding="utf-8")
            print("env CAGE_RUN_ROOT=", os.environ.get("CAGE_RUN_ROOT"), "VLLM_START_TIMEOUT=", os.environ.get("VLLM_START_TIMEOUT"))
            print("run argv:", " ".join(a))
            for bad in ("CAGE_SLO_FLOORS_JSON", "CAGE_ALLOW_STALE_INDEX", "VLLM_PORT"):
                if bad in os.environ:
                    print(f"REFUSED: {bad} is set", file=sys.stderr); sys.exit(2)
            if os.environ.get("CAGE_TEST_RUN_STOP_FAILED"):
                print("[run_campaign] STOP FAILED (exit 1, timeout): launcher=vllm")
                if "--seal" in a: (root / "ledger.json").write_text("{}", encoding="utf-8")
                sys.exit(2)
            if "--seal" in a:
                (root / "ledger.json").write_text("{}", encoding="utf-8")
            print("[run_campaign] sealed."); sys.exit(0)
        ''', exe=False)
    _w(pr / "scripts" / "3_run" / "seal_campaign_run.py", "import sys\nfrom pathlib import Path\nPath(sys.argv[1], 'ledger.json').write_text('{}')\nprint('sealed')\n", exe=False)
    _w(pr / "scripts" / "4_analysis" / "rescore_quality.py", r'''
        import sys
        from pathlib import Path
        a = sys.argv[1:]; root = Path(a[a.index("--run-root") + 1]); rid = a[a.index("--scoring-run-id") + 1]
        if (root / "scoring" / rid).exists():
            print(f"ERROR: {root / 'scoring' / rid} already exists: scoring passes are append-only", file=sys.stderr); sys.exit(2)
        (root / "scoring" / rid).mkdir(parents=True); print("scored", a); sys.exit(0)
        ''', exe=False)
    _w(pr / "scripts" / "4_analysis" / "build_predicate_table.py", r'''
        import sys
        from pathlib import Path
        a = sys.argv[1:]; root = Path(a[0]); rid = a[a.index("--scoring-run-id") + 1]
        assert "--max-null-fraction" in a and "--freeze-file" in a
        if (root / "predicate" / rid).exists() and "--force" not in a:
            print("already exists; rebuild deliberately with --force", file=sys.stderr); sys.exit(2)
        (root / "predicate" / rid).mkdir(parents=True, exist_ok=True); print("predicate", a); sys.exit(0)
        ''', exe=False)
    _w(pr / "scripts" / "5_observability" / "gcs_backup_daemon.sh", r'''
        #!/bin/bash
        echo "backup_daemon $* TARGET=${CAGE_BACKUP_TARGET:-}" >> "$CAGE_TEST_LOG"
        case "$1" in
          start) echo "[gcs-backup] started" ;;
          status) echo "[gcs-backup] RUNNING (pid 1) scope=x tree=$2"; exit 0 ;;
          stop) mkdir -p "$CAGE_TEST_BACKUP/$2" .agent; cp -R "$2"/. "$CAGE_TEST_BACKUP/$2"/ 2>/dev/null; touch .agent/last_sync_ok_local; echo "[gcs-backup] daemon stopped" ;;
        esac
        ''')
    _w(pr / "scripts" / "5_observability" / "collect_logs.sh", r'''
        #!/bin/bash
        echo "collect_logs TOKEN=$CAGE_COLLECT_TOKEN TARGET=$CAGE_BACKUP_TARGET" >> "$CAGE_TEST_LOG"
        mkdir -p "$CAGE_TEST_BACKUP/vm_logs/pod"; echo forensic > "$CAGE_TEST_BACKUP/vm_logs/pod/nvidia.txt"
        echo "COLLECT_LOGS_DONE host=pod sentinel=COLLECT_OK_$CAGE_COLLECT_TOKEN"
        ''')
    _w(pr / "scripts" / "5_observability" / "watch_campaign.sh", "#!/bin/bash\necho 'cells 2/2 windows 2/2'; echo 'RUNNING-HEALTHY'; exit 0\n")

    # --- the profile under test ---------------------------------------------
    profile = tmp_path / "S1.env"
    profile.write_text(textwrap.dedent(f'''
        EXP=S1
        SESSION=a
        CAMPAIGN=camp1
        MODEL=Qwen/Qwen3-14B
        MODEL_SLUG=qwen3-14b
        GPU_ID="NVIDIA H100 80GB HBM3"
        GPU_COUNT=1
        ENGINES="vllm hf"
        CALIBRATE="vllm"
        BUDGET_RATIOS="1.0"
        HOURS=1
        SEATBELT=8h
        SETUP_BOUND_MIN=1
        DC_PREFS="EU-RO-1 US-NE-1"
        PREFETCH_MODELS="Qwen/Qwen3-14B"
        MAX_NULL_FRACTION=0.2
        MONITOR_INTERVAL_S=1
        POD_REPO={pr}
        POD_BACKUP_DIR={backup}
        POD_TARBALL={tmp_path}/cage_repo.tar.gz
        FREEZE_FILE={freeze}
        QUERY_MANIFESTS="squad_v2=data/manifests/squad_v2_50x3_seed42.json"
        '''), encoding="utf-8")

    return {"tmp": tmp_path, "bin": b, "home": home, "pod_repo": pr, "backup": backup,
            "exp_root": exp_root, "scripts": fake_scripts, "ledger": ledger, "log": log,
            "profile": profile}


def _env(w: Dict[str, Path], **extra: str) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CAGE_") and k not in ("VLLM_PORT", "SGLANG_PORT")}
    env["PATH"] = f"{w['bin']}:{env.get('PATH', '')}"
    env.update({
        "CAGE_TEST_LOG": str(w["log"]), "CAGE_TEST_HOME": str(w["home"]),
        "CAGE_TEST_BACKUP": str(w["backup"]), "CAGE_POD_LEDGER": str(w["ledger"]),
        "CAGE_SCRIPTS_DIR": str(w["scripts"]), "CAGE_EXP_ROOT": str(w["exp_root"]),
        "CAGE_EXP_DATE": DATE, "CAGE_MAC_PYTHON": sys.executable,
        "CAGE_POD_READY_TIMEOUT_S": "5", "CAGE_POD_READY_POLL_S": "1", "CAGE_POD_JOB_POLL_S": "1",
    })
    env.update(extra)
    return env


def _master(w: Dict[str, Path], *argv: str, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(MASTER), "S1", "--profile", str(w["profile"]), *argv],
        capture_output=True, text=True, env=_env(w, **extra), timeout=600, cwd=str(REPO_ROOT))


def _state(w: Dict[str, Path]) -> dict:
    return json.loads((w["exp_root"] / "S1" / DATE / "extras" / "state.json").read_text(encoding="utf-8"))


def _calls(w: Dict[str, Path]) -> List[str]:
    return w["log"].read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# 1. plan mode: every stage printed, nothing run, nothing created
# ---------------------------------------------------------------------------


def test_plan_mode_prints_every_stage_and_runs_nothing(world: Dict[str, Path]) -> None:
    proc = _master(world, "--plan")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for i, s in enumerate(STAGES):
        assert f"stage {i}: {s} (plan)" in proc.stdout, s
    assert "plan complete" in proc.stdout
    assert "[plan] provision pod in EU-RO-1 (--yes)" in proc.stdout
    # run_step prints commands %q-escaped (spaces become "\ "), so match space-free parts
    assert "run_campaign.py" in proc.stdout and "--campaign-root" in proc.stdout and "--seal" in proc.stdout
    assert "teardown_pod.sh" in proc.stdout and "network-volume" in proc.stdout
    assert _calls(world) == []                       # no fake was called
    assert not (world["exp_root"] / "S1").exists()   # no landing folder
    # an empty MAX_NULL_FRACTION refuses the live stage 10 but the plan still shows every stage
    empty = world["tmp"] / "S1_nomax.env"
    empty.write_text(world["profile"].read_text(encoding="utf-8").replace("MAX_NULL_FRACTION=0.2", "MAX_NULL_FRACTION="), encoding="utf-8")
    world["profile"] = empty
    proc = _master(world, "--plan")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[plan] NOTE: MAX_NULL_FRACTION is empty" in proc.stdout and "stage 14: teardown (plan)" in proc.stdout
    assert _calls(world) == []


def test_list_stages_and_usage(world: Dict[str, Path]) -> None:
    proc = subprocess.run(["bash", str(MASTER), "S1", "--list-stages"], capture_output=True, text=True, env=_env(world))
    assert proc.stdout.split() == STAGES
    proc = subprocess.run(["bash", str(MASTER)], capture_output=True, text=True, env=_env(world))
    assert proc.returncode == 2 and "stages (in order)" in proc.stderr
    proc = _master(world, "--from", "nowhere")
    assert proc.returncode == 1 and "unknown stage 'nowhere'" in proc.stderr


# ---------------------------------------------------------------------------
# 2. the money gate
# ---------------------------------------------------------------------------


def test_without_yes_the_master_stops_before_provision_after_stage_0(world: Dict[str, Path]) -> None:
    proc = _master(world)
    assert proc.returncode == 10, proc.stdout + proc.stderr
    assert "stage provision is BILLABLE or IRREVERSIBLE" in proc.stdout
    assert "stopped before provision (no --yes provision)" in proc.stdout
    st = _state(world)
    assert st["stages"]["preflight-mac"]["status"] == "passed"
    assert "provision" not in st["stages"]
    assert st["go"] == []
    calls = _calls(world)
    assert not any("network-volume create" in c or "--yes" in c for c in calls)
    assert any(c.startswith("runpodctl gpu list --include-unavailable") for c in calls)
    assert any("provision_pod" in c and "--yes" not in c for c in calls)   # the plan print ran
    # stage 0 facts landed
    assert st["run_id"].startswith("2") and re.fullmatch(r"[a-z0-9][a-z0-9-]{2,40}", st["run_id"])
    assert st["build"]["sha"] == "0123456789abcdef0123456789abcdef01234567"
    assert st["siting"]["available_dcs"] == "US-NE-1"
    assert (world["exp_root"] / "S1" / DATE / "extras" / "profile.env").is_file()


# ---------------------------------------------------------------------------
# 3. the full sequence against the fakes
# ---------------------------------------------------------------------------


def test_full_sequence_with_both_gos_lands_the_run(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--yes", "teardown")
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]
    assert "all requested stages passed" in proc.stdout
    st = _state(world)
    assert [s for s in STAGES if st["stages"][s]["status"] == "passed"] == STAGES
    assert [g["stage"] for g in st["go"]] == ["provision", "teardown"]
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", g["instant_utc"]) for g in st["go"])
    assert st["pod"]["id"] == "pod123" and st["pod"]["volume_id"] == "vol123" and st["pod"]["dc"] == "US-NE-1"
    assert st["pod"]["ssh_host"] == "root@pod.test" and st["pod"]["ssh_port"] == "2222"
    assert st["pod"]["price_per_hour_usd"] == "3.49"
    assert st["cost"]["true_zero_utc"]
    rid = st["run_id"]

    calls = _calls(world)
    # siting: EU-RO-1 has no stock, so only US-NE-1 is tried, volume first then the pod
    assert not any("data-center-id EU-RO-1" in c for c in calls)
    i_vol = next(i for i, c in enumerate(calls) if "network-volume create" in c)
    i_pod = next(i for i, c in enumerate(calls) if "provision_pod" in c and "--yes" in c)
    assert i_vol < i_pod and "--data-center-ids US-NE-1 --network-volume-id vol123" in calls[i_pod]
    # order of the pod-side jobs (the submit snippet is multi-line: scan the whole log)
    names: List[str] = []
    for n in re.findall(r"\.cage_jobs/([A-Za-z0-9_.-]+)\.cmd", world["log"].read_text(encoding="utf-8")):
        if not names or names[-1] != n:
            names.append(n)
    assert names == ["setup", "validate_vllm", "calibrate_vllm", "plan", "run", "score", "collect"]
    assert any("setup_runpod CHARTER_DATASETS=squad_v2 musique qasper PREFETCH_MODELS=Qwen/Qwen3-14B" in c for c in calls), \
        [c for c in calls if "setup" in c.lower()]
    assert any("preflight Qwen/Qwen3-14B http://localhost:8000 BACKENDS=vllm" in c for c in calls), \
        [c for c in calls if "preflight" in c]
    assert any("teardown_pod pod123" in c and "ASSUME_YES=1" in c and "SSH=root@pod.test" in c for c in calls), \
        [c for c in calls if "teardown" in c]
    assert calls.index(next(c for c in calls if "network-volume delete vol123" in c)) > calls.index(next(c for c in calls if "teardown_pod" in c))
    assert calls[-1].startswith("runpodctl user")
    # every ssh/scp went to the host the state file holds, never to the placeholder
    assert not any("<user@host>" in c for c in calls), [c[:120] for c in calls if "<user@host>" in c][:5]
    assert sum(1 for c in calls if c.startswith("ssh root@pod.test :: ")) >= 10

    land = world["exp_root"] / "S1" / DATE
    run_local = land / "run" / "camp1" / "a" / rid
    assert (run_local / "ledger.json").is_file() and (run_local / "cells" / "k1" / "window_squad_v2-01" / "regime.json").is_file()
    # byte-identical to the backup copy the pod synced
    backup_copy = world["backup"] / "results" / "camp1" / "a" / rid
    for p in sorted(backup_copy.rglob("*")):
        if p.is_file():
            assert (run_local / p.relative_to(backup_copy)).read_bytes() == p.read_bytes(), p
    assert (land / "logs" / "setup" / "setup.log").is_file()
    assert (land / "logs" / "runner" / "run.log").is_file()
    assert "CAGE_RUN_ROOT=" in (land / "logs" / "runner" / "run.log").read_text(encoding="utf-8")
    assert (land / "logs" / "engines" / "vllm" / "vllm_x.log").is_file()
    assert (land / "logs" / "system" / "vm_logs" / "pod" / "nvidia.txt").is_file()
    assert (land / "extras" / "plan.json").is_file()
    assert (land / "extras" / "calibration" / "S1_vllm.json").is_file()
    assert list((land / "extras" / "calibration").glob("floor_table_*.json"))
    assert json.loads((land / "extras" / "calibration" / "budget_vllm_r1.0.json").read_text(encoding="utf-8"))  # the REAL cache_budget module
    assert (land / "plots" / "20261006-120000" / "stats.json").is_file()
    assert (land / "plots" / "20261006-120000" / "forest_ttft_ms.png").is_file()
    assert list((land / "plots").glob("panels_*/panels_index.csv"))
    analysis = (land / "plots" / "analysis.txt").read_text(encoding="utf-8")
    assert "to be written by the main session after reading the run" in analysis
    assert f"run id: {rid}" in analysis and "MISSING" in analysis
    assert (land / "extras" / "monitor" / "status.json").is_file()
    assert list((land / "extras").glob("verify_*/verify_report.md"))   # stamped per attempt
    assert st["monitor"]["pid"] == ""   # stopped by stage 11
    assert "POD_REPO=" in (land / "extras" / "profile.env").read_text(encoding="utf-8")


def test_resume_skips_passed_stages_and_redo_repeats_one(world: Dict[str, Path]) -> None:
    assert _master(world, "--yes", "provision", "--yes", "teardown").returncode == 0
    n_calls = len(_calls(world))
    proc = _master(world)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.count("already passed") == len(STAGES)
    assert len(_calls(world)) == n_calls                    # nothing ran again
    proc = _master(world, "--only", "analyze", "--redo")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "stage 13: analyze" in proc.stdout and "stage 12: pull" not in proc.stdout
    # the driver's one-look lock: the recorded stamp is reused, the driver is not re-run
    assert "reusing analysis stamp 20261006-120000" in proc.stdout
    assert _state(world)["stages"]["analyze"]["status"] == "passed"
    # the scoring pass is append-only: --redo reuses scoring/<id> and rebuilds the predicate with --force
    proc = _master(world, "--only", "score", "--redo")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    log = (world["exp_root"] / "S1" / DATE / "logs" / "runner" / "score.log").read_text(encoding="utf-8")
    assert "exists: reused (append-only)" in log and "--force" in log


def test_only_refuses_when_the_predecessor_has_not_passed(world: Dict[str, Path]) -> None:
    proc = _master(world, "--only", "teardown", "--yes", "teardown")
    assert proc.returncode == 1
    assert "predecessor 'analyze' has not passed" in proc.stderr
    # --redo never waives the predecessor rule, and a bare --redo is refused
    proc = _master(world, "--only", "teardown", "--redo", "--yes", "teardown")
    assert proc.returncode == 1 and "predecessor 'analyze' has not passed" in proc.stderr
    proc = _master(world, "--redo")
    assert proc.returncode == 1 and "--redo needs --from" in proc.stderr


# ---------------------------------------------------------------------------
# 4. fail-closed: expected exit codes, PORTAL ACTION, state takeover
# ---------------------------------------------------------------------------


def test_a_step_with_an_unexpected_exit_code_fails_the_stage_and_prints_the_portal_block(world: Dict[str, Path]) -> None:
    proc = _master(world, CAGE_TEST_PROVISION_RC="2")   # the plan print in stage 0 returns 2
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "[FAIL] provision plan print: rc=2, expected 0" in proc.stdout
    assert "PORTAL ACTION" in proc.stdout and "resume:" in proc.stdout
    assert "--from preflight-mac" in proc.stdout
    st = _state(world)
    assert st["stages"]["preflight-mac"]["status"] == "failed" and st["stages"]["preflight-mac"]["rc"] == 1


def test_setup_warning_lines_fail_stage_3_even_at_exit_0(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", CAGE_TEST_SETUP_WARN="1")
    assert proc.returncode == 1
    assert "[FAIL] setup: warn-only steps left the lines above" in proc.stdout
    st = _state(world)
    assert st["stages"]["setup"]["status"] == "failed" and st["pod"]["id"] == "pod123"
    assert "id: pod123" in proc.stdout and "NOT YET" in proc.stdout


def test_seatbelt_not_armed_fails_provision_but_records_the_billing_pod(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", CAGE_TEST_NO_SEATBELT="1")
    assert proc.returncode == 1
    assert "BILLING with NO seatbelt" in proc.stdout
    assert _state(world)["pod"]["id"] == "pod123"


def test_resume_after_a_partial_provision_creates_no_second_pod(world: Dict[str, Path]) -> None:
    # Review 2026-10-06, CRITICAL 1: the pod was CREATED, the ssh-info parse
    # failed, the printed resume must continue with the recorded pod.
    proc = _master(world, "--yes", "provision", CAGE_TEST_SSH_INFO_BAD="1")
    assert proc.returncode == 1
    assert "could not parse host/port" in proc.stdout and "CAGE_POD_SSH_OVERRIDE" in proc.stdout
    assert "pod pod123 is RECORDED; the resume continues with it and creates NO second pod" in proc.stdout
    st = _state(world)
    assert st["pod"]["id"] == "pod123" and st["pod"]["volume_id"] == "vol123"
    creates_before = sum(1 for c in _calls(world) if "network-volume create" in c or ("provision_pod" in c and "--yes" in c))
    assert creates_before == 2
    proc = _master(world, "--only", "provision", "--yes", "provision",
                   CAGE_POD_SSH_OVERRIDE="root@pod.test", CAGE_POD_SSH_PORT_OVERRIDE="2222")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "already recorded (resume after a partial provision)" in proc.stdout
    creates_after = sum(1 for c in _calls(world) if "network-volume create" in c or ("provision_pod" in c and "--yes" in c))
    assert creates_after == creates_before                       # nothing was created again
    st = _state(world)
    assert st["pod"]["ssh_host"] == "root@pod.test" and st["stages"]["provision"]["status"] == "passed"
    assert [g["stage"] for g in st["go"]] == ["provision"]         # the GO was recorded once


def test_unknown_pod_id_after_a_create_refuses_until_the_override(world: Dict[str, Path]) -> None:
    assert _master(world).returncode == 10
    sp = world["exp_root"] / "S1" / DATE / "extras" / "state.json"
    st = json.loads(sp.read_text(encoding="utf-8")); st["pod"]["id"] = "unknown-see-portal"
    sp.write_text(json.dumps(st), encoding="utf-8")
    proc = _master(world, "--only", "provision", "--yes", "provision")
    assert proc.returncode == 1 and "its id was never parsed" in proc.stdout
    assert not any("network-volume create" in c for c in _calls(world))
    proc = _master(world, "--only", "provision", "--yes", "provision", CAGE_POD_ID_OVERRIDE="pod999")
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert _state(world)["pod"]["id"] == "pod999"
    assert not any("network-volume create" in c for c in _calls(world))


def test_setup_benign_note_passes_and_the_import_note_fails(world: Dict[str, Path]) -> None:
    # Review 2026-10-06, HIGH 2: setup_runpod.sh:450 prints a NOTE on every
    # bootstrap; only the telemetry import NOTE (line 424) is a failure.
    proc = _master(world, "--yes", "provision", CAGE_TEST_SETUP_NOTE_BAD="1")
    assert proc.returncode == 1 and "not importable" in proc.stdout
    assert _state(world)["stages"]["setup"]["status"] == "failed"
    # the same pod, setup repeated without the bad note: the benign closing NOTE passes
    proc = _master(world, "--from", "setup", "--redo", "--only", "setup")
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert _state(world)["stages"]["setup"]["status"] == "passed"
    log = (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "setup.log").read_text(encoding="utf-8")
    assert "NOTE: harness trees carry no ledger.json" in log


def test_short_seatbelt_fails_stage_0_before_anything_bills(world: Dict[str, Path], tmp_path: Path) -> None:
    # Review 2026-10-06, HIGH 4: the seatbelt must cover every pod-side bound.
    short = tmp_path / "S1_short.env"
    short.write_text(world["profile"].read_text(encoding="utf-8").replace("SEATBELT=8h", "SEATBELT=2h"), encoding="utf-8")
    world["profile"] = short
    proc = _master(world)
    assert proc.returncode == 1
    assert "SEATBELT=2h (120 min) is shorter than the 431 min the stages need" in proc.stdout
    assert not any("provision_pod" in c for c in _calls(world))


def test_shipped_profiles_cover_their_stage_bounds_and_name_registered_models() -> None:
    import importlib
    cb = importlib.import_module("src.orchestration.cache_budget")
    for p in sorted((EXPDIR / "profiles").glob("S*.env")):
        proc = subprocess.run(["bash", "-c", f'set -a; source "{EXPDIR}/profiles/_common.env"; source "{p}"; '
                               'printf "%s|%s|%s|%s|%s|%s|%s" "$MODEL_SLUG" "$SEATBELT" "$HOURS" "$SETUP_BOUND_MIN" "$ENGINES" "$CALIBRATE" "${SCORE_BOUND_MIN:-180}"'],
                              capture_output=True, text=True, env=PROFILE_ENV)
        slug, seatbelt, hours, setup_min, engines, calibrate, score_min = proc.stdout.split("|")
        assert slug in cb.MODEL_KV, f"{p.name}: MODEL_SLUG={slug} is not in cache_budget.MODEL_KV {sorted(cb.MODEL_KV)}"
        n_srv = len([e for e in engines.split() if e != "hf"]); n_cal = len(calibrate.split())
        need = int(setup_min) + n_srv * 25 + n_cal * 60 + 15 + int(hours) * 60 + int(score_min) + 30 + 60
        unit = seatbelt[-1]; n = int(seatbelt[:-1])
        have = n * 60 if unit == "h" else n if unit == "m" else n * 1440
        assert have >= need, f"{p.name}: SEATBELT={seatbelt} ({have} min) < {need} min needed"


def test_a_second_master_on_the_same_landing_is_refused(world: Dict[str, Path]) -> None:
    # Review 2026-10-06, MEDIUM 9: the lock holds a pid whose command line is
    # this script; an impostor process with that name holds it for the test.
    assert _master(world).returncode == 10
    lock = world["exp_root"] / "S1" / DATE / "extras" / ".master.lock"
    impostor = subprocess.Popen(["bash", "-c", "exec -a cage_experiment.sh sleep 30"])
    try:
        lock.write_text(f"{impostor.pid}\n", encoding="utf-8")
        proc = _master(world)
        assert proc.returncode == 1 and f"another master (pid {impostor.pid}) is running" in proc.stderr
    finally:
        impostor.kill(); impostor.wait()          # our own child, by its recorded pid
    lock.write_text("999999\n", encoding="utf-8")   # a stale lock from a dead master is taken over
    proc = _master(world)
    assert proc.returncode == 10 and "stale master lock" in proc.stdout
    assert not lock.exists()                        # released on exit


def test_plan_audit_refuses_a_cell_without_telemetry(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", CAGE_TEST_BAD_PLAN="1")
    assert proc.returncode == 1
    assert "vllm cell carries --vllm-telemetry 0 times: k1" in proc.stdout
    assert _state(world)["stages"]["plan"]["status"] == "failed"


def test_an_unexpected_teardown_exit_code_fails_stage_14(world: Dict[str, Path]) -> None:
    # expected "0|1": a 2 (the delete itself failed) fails the stage, nothing is swept
    proc = _master(world, "--yes", "provision", "--yes", "teardown", CAGE_TEST_TEARDOWN_RC="2")
    assert proc.returncode == 1
    assert "rc=2, expected 0|1" in proc.stdout
    assert not any("network-volume delete" in c for c in _calls(world))
    assert _state(world)["stages"]["teardown"]["status"] == "failed"


def test_run_exit_2_with_stop_failed_only_is_accepted_and_noted(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--yes", "teardown", CAGE_TEST_RUN_STOP_FAILED="1")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    st = _state(world)
    assert st["stages"]["run"]["status"] == "passed"
    assert any("STOP FAILED" in n for n in st["stages"]["run"]["notes"])


def test_running_stage_in_state_refuses_without_redo(world: Dict[str, Path]) -> None:
    assert _master(world).returncode == 10
    sp = world["exp_root"] / "S1" / DATE / "extras" / "state.json"
    st = json.loads(sp.read_text(encoding="utf-8"))
    st["stages"]["preflight-mac"]["status"] = "running"
    sp.write_text(json.dumps(st), encoding="utf-8")
    proc = _master(world)
    assert proc.returncode == 1 and "recorded as RUNNING" in proc.stderr
    sp.write_text("{not json", encoding="utf-8")
    proc = _master(world)
    assert proc.returncode == 1 and "not valid JSON" in proc.stderr


def test_to_stops_after_the_named_stage_with_the_pod_billing(world: Dict[str, Path]) -> None:
    # a smoke: bootstrap, validate, calibrate and plan, never the grid
    proc = _master(world, "--yes", "provision", "--to", "plan")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "stopped after plan (--to)" in proc.stdout and "PORTAL ACTION" in proc.stdout
    assert "--from run" in proc.stdout and "NOT YET" in proc.stdout
    st = _state(world)
    assert st["stages"]["plan"]["status"] == "passed" and "run" not in st["stages"]
    assert not any(".cage_jobs/run.cmd" in c for c in _calls(world))
    assert not any("teardown_pod" in c for c in _calls(world))


def test_a_missing_manifest_fails_stage_0_before_anything_bills(world: Dict[str, Path], tmp_path: Path) -> None:
    bad = tmp_path / "S1_badmanifest.env"
    bad.write_text(world["profile"].read_text(encoding="utf-8") + '\nQUERY_MANIFESTS="squad_v2=data/manifests/squad_v2_50x3_seed42.json musique=data/manifests/musique_50x3_seed42.json"\n', encoding="utf-8")
    world["profile"] = bad
    proc = _master(world)
    assert proc.returncode == 1
    assert "manifest(s) named by the profile do not exist in the repo: data/manifests/musique_50x3_seed42.json" in proc.stdout
    assert not any("provision_pod" in c for c in _calls(world))


def test_clean_room_violation_fails_stage_0(world: Dict[str, Path]) -> None:
    proc = _master(world, CAGE_TEST_VOLUMES='[{"id": "leftover"}]')
    assert proc.returncode == 1
    assert "clean room: network volumes exist" in proc.stdout


def test_dirty_build_fails_stage_0(world: Dict[str, Path]) -> None:
    proc = _master(world, CAGE_TEST_DIRTY="1")
    assert proc.returncode == 1 and "BUILD_INFO is not dirty=0" in proc.stdout


def test_rehearsal_n_reaches_the_planner_and_blocked_cells_need_the_profile_consent(world: Dict[str, Path], tmp_path: Path) -> None:
    # ADR-0144: REHEARSAL_N rides the plan argv; a plan with blocked cells
    # refuses stage 7 unless SKIP_BLOCKED=1, which adds --skip-blocked loudly.
    reh = tmp_path / "S1_rehearsal.env"
    reh.write_text(world["profile"].read_text(encoding="utf-8") + "\nREHEARSAL_N=50\n", encoding="utf-8")
    world["profile"] = reh
    proc = _master(world, "--yes", "provision", "--to", "run", CAGE_TEST_BLOCKED="1")
    assert proc.returncode == 1, proc.stdout[-3000:] + proc.stderr[-1000:]
    plan = json.loads((world["exp_root"] / "S1" / DATE / "extras" / "plan.json").read_text(encoding="utf-8"))
    assert "--rehearsal-n" in plan["argv"] and plan["argv"][plan["argv"].index("--rehearsal-n") + 1] == "50"
    assert "the plan carries 1 blocked cell(s)" in proc.stdout and "SKIP_BLOCKED is not 1" in proc.stdout
    assert not any(".cage_jobs/run.cmd" in c for c in _calls(world))      # nothing was submitted
    assert _state(world)["stages"]["run"]["status"] == "failed"
    # with the consent: the run is submitted with --skip-blocked and the note lands in the state
    consent = tmp_path / "S1_rehearsal_consent.env"
    consent.write_text(reh.read_text(encoding="utf-8") + "SKIP_BLOCKED=1\n", encoding="utf-8")
    world["profile"] = consent
    proc = _master(world, "--from", "run", "--redo", "--to", "run", CAGE_TEST_BLOCKED="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "running the executable subset loudly (SKIP_BLOCKED=1)" in proc.stdout
    log = (world["exp_root"] / "S1" / DATE / "logs" / "runner" / "run.log").read_text(encoding="utf-8")
    assert "run argv:" in log and "--skip-blocked" in log and "--seal-partial" not in log
    assert any("blocked cell(s) skipped loudly" in n for n in _state(world)["stages"]["run"]["notes"])


def test_without_rehearsal_n_and_without_blocked_cells_the_argv_is_unchanged(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "run")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    plan = json.loads((world["exp_root"] / "S1" / DATE / "extras" / "plan.json").read_text(encoding="utf-8"))
    assert "--rehearsal-n" not in plan["argv"]
    log = (world["exp_root"] / "S1" / DATE / "logs" / "runner" / "run.log").read_text(encoding="utf-8")
    assert "--skip-blocked" not in log and "blocked cell" not in proc.stdout


# ---------------------------------------------------------------------------
# 5. static contract: headers, sourcing, bash 3.2, process safety, profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cage_experiment.sh", "pod_job.sh", "monitor_pod.sh"])
def test_scripts_parse_under_the_macos_bash_and_carry_the_contract(name: str) -> None:
    path = EXPDIR / name
    text = path.read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:30])
    assert "# Order:" in head and "# Objective:" in head and "# Cloud:     runpod" in head
    assert re.search(r"^\s*source\s+[^#\n]*_common\.sh", text, re.M)
    for banned in ("pkill", "killall", "kill -1 ", "kill 0 ", " -P 1", "declare -A", "mapfile", "readarray"):
        assert banned not in text, banned
    assert chr(0x2014) not in text and chr(0x2013) not in text
    bash32 = "/bin/bash" if Path("/bin/bash").exists() else "bash"
    proc = subprocess.run([bash32, "-n", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_write_analysis_header_and_profiles_are_plain() -> None:
    text = (EXPDIR / "write_analysis.py").read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:30])
    assert "Order:" in head and "Objective:" in head and "Cloud:     local" in head
    assert chr(0x2014) not in text
    for p in sorted((EXPDIR / "profiles").glob("*.env")):
        body = p.read_text(encoding="utf-8")
        assert chr(0x2014) not in body
        for line in body.splitlines():
            if line.strip() and not line.startswith("#"):
                assert re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", line), f"{p.name}: {line}"
                assert "$(" not in line and "`" not in line, f"{p.name}: {line}"
        proc = subprocess.run(["bash", "-c", f'set -a; source "{EXPDIR}/profiles/_common.env"; source "{p}"; printf "%s" "$EXP"'],
                              capture_output=True, text=True, env=PROFILE_ENV)
        assert proc.returncode == 0 and proc.stdout == (p.stem if p.stem != "_common" else ""), p.name
