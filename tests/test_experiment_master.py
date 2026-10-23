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
# G18 (2026-10-17): teardown right after the verified pull; analyze (Mac only) last
STAGES = ["preflight-mac", "provision", "ship", "setup", "validate", "calibrate", "plan", "run",
          "monitor", "seal", "score", "collect", "pull", "teardown", "analyze"]
#: G15: every registered profile prefetches the served model and these (the
#: retriever, the reranker, the compressor, the four scoring instruments)
PREFETCH_FULL = ("Qwen/Qwen3-14B intfloat/e5-large-v2 BAAI/bge-reranker-large "
                 "microsoft/llmlingua-2-xlm-roberta-large-meetingbank KRLabsOrg/lettucedect-base-modernbert-en-v1 "
                 "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli sentence-transformers/all-MiniLM-L6-v2 roberta-base")
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
# ADR-0161: an outage injector; .ssh_fail holds a count of round trips that
# fail like a timed-out channel (exit 255, nothing run).
if [ -s "$CAGE_TEST_HOME/.ssh_fail" ]; then
  n=$(cat "$CAGE_TEST_HOME/.ssh_fail")
  if [ "$n" -gt 0 ]; then
    echo $(( n - 1 )) > "$CAGE_TEST_HOME/.ssh_fail"
    echo "ssh: connect to host $host port 2222: Operation timed out" >&2
    exit 255
  fi
fi
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
# ADR-0161: the same injector for scp (.scp_fail)
if [ -s "$CAGE_TEST_HOME/.scp_fail" ]; then
  n=$(cat "$CAGE_TEST_HOME/.scp_fail")
  if [ "$n" -gt 0 ]; then
    echo $(( n - 1 )) > "$CAGE_TEST_HOME/.scp_fail"
    echo "scp: Connection closed" >&2
    exit 255
  fi
fi
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
      get) echo "{\"id\": \"$3\", \"runtimeStatus\": \"${CAGE_TEST_RUNTIME:-running}\", \"desiredStatus\": \"RUNNING\", \"memoryInGb\": 172}" ;;
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
    # P-2 (S0F-88): stage 2 compares the Mac HEAD with the packaged sha; the
    # fake HEAD is the fake packager's sha unless a test moves it.
    _w(b / "git", "#!/bin/bash\ncase \"$*\" in *\"rev-parse HEAD\"*) echo \"${CAGE_TEST_HEAD_SHA:-${CAGE_TEST_SHA:-0123456789abcdef0123456789abcdef01234567}}\" ;; *) exit 1 ;; esac\n")
    # /metrics answers with the live names of 2026-10-07 (S0F-37), /server_info
    # with SGLang's launch arguments (S0F-88 U7); everything else is a silent 200
    _w(b / "curl", "#!/bin/bash\ncase \"$*\" in *\"/metrics\"*) printf 'sglang:token_usage 0.1\\nsglang:num_retracted_reqs 0\\n# TYPE sglang:num_retracted_requests_total counter\\nsglang:num_running_reqs 0\\nsglang:num_queue_reqs 0\\nvllm:kv_cache_usage_perc 0.1\\n' ;; *\"/server_info\"*) printf '{\"version\":\"0.5.10.post1\",\"enable_cache_report\":%s}\\n' \"${CAGE_TEST_CACHE_REPORT:-true}\" ;; esac\nexit 0\n")
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
    _w(fake_scripts / "checks" / "run_tests.sh", "#!/bin/bash\necho \"run_tests CAGE_LOAD_GATE=${CAGE_LOAD_GATE:-} $*\" >> \"$CAGE_TEST_LOG\"; case \"$*\" in *test_load_gate_p1*) echo \"${CAGE_TEST_LOAD_GATE_OUT:-1 passed}\" ;; *) echo '4000 passed' ;; esac; exit 0\n")
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
            # ADR-0176 Group B: a datacenter that refuses the pod (the volume was created)
            [ -z "${CAGE_TEST_PROVISION_YES_RC:-}" ] || { echo "[cage] no stock for the pod in this datacenter"; exit "$CAGE_TEST_PROVISION_YES_RC"; }
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
        echo "  -- network volume(s) listed: KEPT by design (owner rule 2026-10-22)"
        echo "TEARDOWN_COMPLETE pod=$1"
        exit "${CAGE_TEST_TEARDOWN_RC:-0}"
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
        import sys
        class PlanError(Exception):
            pass
        # ADR-0155: the stage 0 probe reads the anchor arm's registered shape
        DEMAND_ANCHOR_ARM = "gold-fresh"
        class _Grid:
            def __init__(self, session):
                self.session = session
                self.demand_seq_tokens = {"gold-fresh": 4779}
        def get_session_grid(session):
            if session not in ("a", "b"):
                raise PlanError(f"session {session!r} is not a registered grid")
            return _Grid(session)
        def preflight_registration_check(session, *, floor_avg_seq_tokens, rehearsal_n=None, query_manifests=None, charter_datasets=(), floor_concurrency=None, gpu_id=None):
            # gap triage 2026-10-09: the stage 0 registration preflight; two test knobs
            import os
            get_session_grid(session)
            problems = []
            if not gpu_id:  # S0F-88 B-7: the master passes the profile's GPU_ID
                problems.append("the master passed no GPU_ID to the pool ceiling check (S0F-88 B-7)")
            if str(floor_avg_seq_tokens).strip() != "4779":
                problems.append(f"FLOOR_AVG_SEQ_TOKENS={floor_avg_seq_tokens} but session {session!r} registers 4779 served tokens for the anchor arm 'gold-fresh' (ADR-0155)")
            if os.environ.get("CAGE_TEST_PREFLIGHT_PROBLEMS"):
                problems.append(os.environ["CAGE_TEST_PREFLIGHT_PROBLEMS"])
            # A11 item 4: the largest planned pool per engine, on request
            largest = {} if os.environ.get("CAGE_TEST_LARGEST_POOL") != "1" else {
                "vllm": {"planned_tokens": 298687, "planned_bytes": 48936960000,
                         "env": {"CAGE_KV_BUDGET_BYTES": "48936960000", "VLLM_MAX_MODEL_LEN": "32768"}}}
            # review 2026-10-21 H1: the largest pool a GPU-helper cell plans, on request
            helpers = {} if os.environ.get("CAGE_TEST_HELPER_POOL") != "1" else {
                "sglang": {"planned_tokens": 70437, "planned_bytes": 11540480000,
                           "env": {"CAGE_SGLANG_MAX_TOTAL_TOKENS": "70437", "VLLM_MAX_MODEL_LEN": "8192"}}}
            return {"session": session, "anchor_seq_tokens": 4779, "cells": 3, "executable_cells": 3,
                    "needs_lmcache": os.environ.get("CAGE_TEST_NEEDS_LMCACHE") == "1", "largest_pools": largest,
                    "largest_helper_pools": helpers, "problems": problems}
        if __name__ == "__main__":
            a = sys.argv[1:]
            if a and a[0] == "rungs":  # ADR-0154: stage 5 lists the budgeted rungs on the Mac
                print("1 0.5"); sys.exit(0)
            print("REFUSED: fake Mac planner knows only 'rungs'", file=sys.stderr); sys.exit(2)
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
    # Spec Block A1 (2026-10-16): the fake stands in so that the master's "0|3"
    # contract is exercised by a real exit 0, not by Python's exit 2 on a missing file.
    _w(fake_scripts / "4_analysis" / "yield_table.py", r'''
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        out = Path(args[args.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
        (out / "yield_table.csv").write_text("row_key,window_dir,goodput_frac\nk1,window_squad_v2-01,1.0\n", encoding="utf-8")
        (out / "yield_summary.md").write_text("# yield\n", encoding="utf-8")
        print("[yield_table] windows=1 records=1 skips=0", args); sys.exit(0)
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
          # S0F-57: the stage 4 lmcache probe; the fake venv has none unless a
          # test says so (the pod suite of 2026-10-09 ran where lmcache imports)
          *"import lmcache"*) exit "${{CAGE_TEST_LMCACHE_IMPORT_RC:-1}}" ;;
        esac
        exec {sys.executable} "$@"
        ''')
    # the pod suite job runs `python -m pytest` inside the fake repo: one passing test (SUITE_ON_POD=1 worlds)
    _w(pr / "tests" / "test_pod_suite_ok.py", "def test_ok():\n    assert True\n", exe=False)
    # P-1: the stage 4 live-load gate names its own file; a passing stand-in here
    _w(pr / "tests" / "test_load_gate_p1.py", "def test_gate():\n    assert True\n", exe=False)
    _w(pr / "scripts" / "runpod" / "setup_runpod.sh", r'''
        #!/bin/bash
        echo "setup_runpod CHARTER_DATASETS=$CHARTER_DATASETS PREFETCH_MODELS=$PREFETCH_MODELS" >> "$CAGE_TEST_LOG"
        echo "[cage]   cage-env: link"
        echo "[cage]   all charter datasets staged: $CHARTER_DATASETS"
        if [ "${INSTALL_LMCACHE:-0}" = "1" ]; then
          if [ -n "${CAGE_TEST_LMCACHE_FAIL:-}" ]; then echo "[cage] FATAL: lmcache install or import FAILED (S0F-57, ADR-0158)"; exit 1; fi
          echo "[cage]   lmcache: 0.3.x installed beside vllm 0.19.1"
        fi
        for m in $PREFETCH_MODELS; do echo "[cage]   $m: cached"; done
        echo "[cage]   pynvml OK -> GPU memory-pressure telemetry WILL be captured"
        echo "[cage]   cage_stats.api import OK -> serving telemetry available"
        [ -z "${CAGE_TEST_SETUP_WARN:-}" ] || echo "[cage] WARNING: dataset stage FAILED: qasper"
        [ -z "${CAGE_TEST_SETUP_NOTE_BAD:-}" ] || echo "[cage]   NOTE: cage_stats.api not importable (No module named httpx); set CAGE_STATS_HOME"
        echo "[cage]  RunPod bootstrap complete. Next (docs/RUNBOOK.md lifecycle):"
        echo "[cage]  NOTE: harness trees carry no ledger.json until the campaign driver seals them"
        ''')
    for eng in ("vllm", "sglang", "lmdeploy"):
        # a start writes the realized-pool record the validate probes read (ADR-0142)
        _w(pr / "scripts" / "2_serving" / f"manage_{eng}_server.sh", f"#!/bin/bash\nextra=\"\"; [ -z \"${{VLLM_KV_CACHE_DTYPE:-}}${{SGLANG_KV_CACHE_DTYPE:-}}\" ] || extra=\" kv=${{VLLM_KV_CACHE_DTYPE:-}}${{SGLANG_KV_CACHE_DTYPE:-}}\"; [ -z \"${{LMCACHE_MAX_LOCAL_CPU_SIZE:-}}\" ] || extra=\"$extra store=${{LMCACHE_MAX_LOCAL_CPU_SIZE}}\"; echo \"launcher {eng} $*$extra\" >> \"$CAGE_TEST_LOG\"; case \"$1\" in start|restart) mkdir -p logs/{eng}; echo '{{}}' > logs/{eng}/CURRENT.kvpool.json; [ -n \"${{LMCACHE_MAX_LOCAL_CPU_SIZE:-}}\" ] && echo \"'max_local_cpu_size': ${{LMCACHE_MAX_LOCAL_CPU_SIZE}}.0,\" >> logs/{eng}/vllm_fake.log ;; esac; echo \"{eng} $1 ok\"; exit 0\n")
    _w(pr / "scripts" / "checks" / "preflight_check.sh", "#!/bin/bash\necho \"preflight $* BACKENDS=$CAGE_PREFLIGHT_BACKENDS\" >> \"$CAGE_TEST_LOG\"; echo 'PREFLIGHT PASS -- all Gate-2 components green.'; exit \"${CAGE_TEST_PREFLIGHT_RC:-0}\"\n")
    # S0F-59 (ADR-0151): the thinking pin probe, one per server engine at stage 4
    _w(pr / "scripts" / "checks" / "probe_thinking_pin.py", r'''
        import json, os, sys
        from pathlib import Path
        a = sys.argv[1:]
        backend = a[a.index("--backend") + 1]
        out = Path(a[a.index("--out") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
        ok = os.environ.get("CAGE_TEST_THINKING_RC", "0") != "1"
        # review 2026-10-13 A1: the stage 4 start check calls this probe with a
        # start_check_<name> record; CAGE_TEST_START_CHECK_RC=1 fails those calls only
        if "start_check_" in out.name and os.environ.get("CAGE_TEST_START_CHECK_RC") == "1":
            ok = False
        out.write_text(json.dumps({"backend": backend, "ok": ok}), encoding="utf-8")
        with open(os.environ["CAGE_TEST_LOG"], "a", encoding="utf-8") as fh:
            fh.write(f"thinking_probe {backend} {a[a.index('--api-base') + 1]}\n")
        print(f"THINKING_PIN_OK backend={backend} num_tokens=3" if ok else f"THINKING_PIN_FAILED backend={backend} reason=test")
        sys.exit(0 if ok else 1)
        ''', exe=False)
    # S0F-88 U2 (review H1): the pod memory probe on the LMCache path
    _w(pr / "scripts" / "checks" / "probe_pod_memory.py", r'''
        import os, sys
        ok = os.environ.get("CAGE_TEST_POD_MEMORY_RC", "0") != "1"
        print("POD_MEMORY_OK need_gib=22 available_gib=150.0 source=cgroup-v2" if ok else "POD_MEMORY_FAILED need_gib=172 available_gib=150.0 source=cgroup-v2")
        sys.exit(0 if ok else 1)
        ''', exe=False)
    # S0F-88 B-6: the reranker device probe, once at stage 4 after the vLLM stop
    _w(pr / "scripts" / "checks" / "probe_reranker_device.py", r'''
        import os, sys
        with open(os.environ["CAGE_TEST_LOG"], "a", encoding="utf-8") as fh:
            fh.write("rerank_device_probe\n")
        ok = os.environ.get("CAGE_TEST_RERANK_RC", "0") != "1"
        print("RERANK_DEVICE_OK queries=50 identical=50 max_abs_score_diff=1e-06" if ok else "RERANK_DEVICE_FAILED reason=test")
        sys.exit(0 if ok else 1)
        ''', exe=False)
    # A14: the compressor and dense-encoder device probes, beside the reranker probe
    for name, marker in (("probe_compressor_device.py", "COMPRESS_DEVICE_OK"), ("probe_encoder_device.py", "ENCODER_DEVICE_OK")):
        _w(pr / "scripts" / "checks" / name, f'print("{marker} queries=50 identical=50")\n', exe=False)
    # A14: the compressor and dense-encoder device probes beside the reranker probe
    for name, marker in (("probe_compressor_device.py", "COMPRESSOR_DEVICE_OK"), ("probe_encoder_device.py", "ENCODER_DEVICE_OK")):
        _w(pr / "scripts" / "checks" / name, f'print("{marker} queries=50 identical=50")\n', exe=False)
    # ADR-0176 Group D: the stage 10 scoring batch probe; CAGE_TEST_SCORING_PROBE
    # selects its verdict (OK default, FAILED, or CRASH for no verdict line)
    _w(pr / "scripts" / "checks" / "probe_scoring_batch.py", r'''
        import json, os, sys
        from pathlib import Path
        a = sys.argv[1:]
        with open(os.environ["CAGE_TEST_LOG"], "a", encoding="utf-8") as fh:
            fh.write("scoring_probe " + " ".join(a) + "\n")
        verdict = os.environ.get("CAGE_TEST_SCORING_PROBE", "OK")
        if verdict == "CRASH":
            print("Traceback: boom", file=sys.stderr); sys.exit(1)
        out = Path(a[a.index("--out") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ok": verdict == "OK"}), encoding="utf-8")
        if verdict == "OK":
            print("SCORING_BATCH_OK rows=50 batch_size=%s row_by_row_s_per_row=0.3100 batched_s_per_row=0.1000 speedup=3.10 window=cells/k1/window_squad_v2-01" % a[a.index("--batch-size") + 1]); sys.exit(0)
        print("SCORING_BATCH_FAILED rows=50 batch_size=32 reason=1 of 50 rows differ; first: row 3 (e3) field faithfulness: 0.5 row by row vs 0.6 batched"); sys.exit(1)
        ''', exe=False)
    _w(pr / "scripts" / "3_run" / "calibrate_cell.py", r'''
        import json, sys
        from pathlib import Path
        a = sys.argv[1:]; out = Path(a[a.index("--output") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"procedure_version": "cal-v3 (2026-10-12)", "start_qps_source": "floor-service-rate", "lambda_star": 2.5, "argv": a}), encoding="utf-8")
        print("[calibrate] wrote", out); sys.exit(0)
        ''', exe=False)
    _w(pr / "scripts" / "3_run" / "run_campaign.py", r'''
        import json, os, sys
        from pathlib import Path
        a = sys.argv[1:]
        if a[0] == "rungs":
            print("1 0.5"); sys.exit(0)
        if a[0] == "pool-check":  # A11 item 4
            with open(os.environ["CAGE_TEST_LOG"], "a", encoding="utf-8") as fh:
                fh.write("pool_check " + " ".join(a[1:]) + "\n")
            ok = os.environ.get("CAGE_TEST_POOL_CHECK_RC", "0") != "1"
            print("LARGEST_POOL_OK engine=vllm realized_tokens=298687" if ok else "LARGEST_POOL_FAILED engine=vllm reason=test")
            sys.exit(0 if ok else 1)
        if a[0] == "calibrate-rungs":
            # ADR-0160 (S0 attempt 2): the ladder windows run the campaign seam, which
            # refuses without the provider and hardware env (campaign_session.py)
            if not os.environ.get("CAGE_PROVIDER") or not os.environ.get("CAGE_HARDWARE"):
                print("campaign mode refused: CAGE_PROVIDER / CAGE_HARDWARE unset", file=sys.stderr); sys.exit(1)
            out = Path(a[a.index("--out") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
            fail = os.environ.get("CAGE_TEST_RUNGS_FAIL") == "1" and (
                not os.environ.get("CAGE_TEST_RUNGS_FAIL_ENGINE")
                or os.environ.get("CAGE_TEST_RUNGS_FAIL_ENGINE") == a[a.index("--engine") + 1])
            label = "LADDER_EXHAUSTED" if fail else "ESTIMATED"
            rungs = {r: {"r": float(r), "label": label, "lambda_star_qps": None if fail else 3.0 * float(r)}
                     for r in a[a.index("--rungs") + 1].split(",")}
            small_fail = os.environ.get("CAGE_TEST_SMALL_RUNGS_FAIL") == "1"
            small = {r: {"r": float(r), "label": "LADDER_EXHAUSTED" if small_fail else "ESTIMATED",
                         "lambda_star_qps": None if small_fail else 12.0 * float(r)}
                     for r in a[a.index("--rungs") + 1].split(",")}
            # ADR-0156: the v2 artifact carries both anchors under classes
            doc = {"schema": "cage-rung-calibration-v2", "engine": a[a.index("--engine") + 1], "argv": a,
                   "anchor_seq_tokens": 4779, "rungs": rungs,
                   "classes": {"4779": {"role": "anchor", "arm": "gold-fresh", "seq_tokens": 4779, "rungs": rungs},
                               "348": {"role": "smallest", "arm": "retr-trunc", "seq_tokens": 348, "prefix_mode": "OFF", "rungs": small}}}
            out.write_text(json.dumps(doc), encoding="utf-8")
            print("[calibrate-rungs] artifact written:", out); sys.exit(1 if fail else 0)
        if a[0] == "plan":
            out = Path(a[a.index("--out") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
            plan = {"schema": "cage-campaign-plan-v6", "session": a[a.index("--session") + 1], "argv": a,
                    "steps": [{"kind": "relaunch", "engine": "vllm", "argv": ["manage_vllm_server.sh", "start"]},
                              # cell steps carry engine/topology inside cellspec (the real plan shape; gap triage 2026-10-09 L2)
                              {"kind": "cell", "cellspec": {"engine": "vllm", "topology": "single"}, "row_key": "k1", "argv": ["run_experiment.py", "--vllm-telemetry"]},
                              {"kind": "cell", "cellspec": {"engine": "hf", "topology": "single"}, "row_key": "k2", "argv": ["run_cag_reference.py"]}],
                    "blocked_row_keys": ["k9"] if os.environ.get("CAGE_TEST_BLOCKED") else []}
            # S0F-88 B-5: the measured run projection the plan stage compares with HOURS
            total = float(os.environ.get("CAGE_TEST_PROJECTION_S", "1200"))
            plan["run_projection"] = {
                "total_s": total, "total_h": round(total / 3600, 2), "source": "test",
                "by_kind_s": {"relaunch": 60.0, "dry_window": 0.0, "cell": total - 60.0},
                "unmeasured_steps": ["cell k3"] if os.environ.get("CAGE_TEST_UNMEASURED") else [],
                "note": "test"}
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
            for bad in ("CAGE_SLO_FLOORS_JSON", "CAGE_ALLOW_STALE_INDEX", "VLLM_PORT", "CAGE_KV_CEILING"):
                if bad in os.environ:
                    print(f"REFUSED: {bad} is set", file=sys.stderr); sys.exit(2)
            if os.environ.get("CAGE_TEST_RUN_DRY_FAILED"):
                print("[run_campaign] DRY WINDOW FAILED: engine=vllm class=4779 tokens r=0.5 rate=2.85 rps expected IN_REGIME, label='UNPRESSURED'")
                print("[run_campaign] totals: {'ok': 79, 'dry-window-failed': 14, 'skipped-dry-window': 28}")
                sys.exit(3)
            if os.environ.get("CAGE_TEST_RUN_CELL_FAILED"):
                # decision 3 (2026-10-13): the run finished (totals printed) with a failed cell
                print("  [               failed] gold-fresh|none|none|single|vllm|qwen3-14b|F1 (qasper)")
                print("[run_campaign] totals: {'ok': 109, 'failed': 1}")
                sys.exit(1)
            if os.environ.get("CAGE_TEST_RUN_CRASH"):
                print("Traceback (most recent call last):\n  boom", file=sys.stderr)
                sys.exit(1)
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
        # review 2026-10-10 (R2): a --full pass carries the freeze file that pins the instruments
        assert "--full" in a and "--freeze-file" in a, a
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
        SEATBELT=12h
        SETUP_BOUND_MIN=1
        CALIBRATE_RUNGS_BOUND_S=9000
        DC_PREFS="EU-RO-1 US-NE-1"
        MAX_NULL_FRACTION=0.2
        STEP_TIMEOUT_S=5400
        LADDER_FAIL_FAST=1
        BACKUP_ARCHIVE_DIR=/workspace/backup-archives
        CELL_STALL_MIN=60
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


# The master exports whole profiles (set -a), so a caller that sourced S0.env
# (the master running stage 0's suite, or a dev shell) carries every profile
# key; the fake world keeps only the process basics (2026-10-07: an inherited
# POD_RESULTS_DIR=/root/... failed the unpack step under the exported profile).
_ENV_KEEP = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "USER", "LOGNAME", "SHELL", "CAGE_LOG_ROOT")


def _env(w: Dict[str, Path], **extra: str) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in _ENV_KEEP}
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
    assert "[plan] NOTE: MAX_NULL_FRACTION is empty" in proc.stdout and "stage 13: teardown (plan)" in proc.stdout
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
    # P-1 (S0F-88): stage 4 runs the live-load gate as its own pod job
    # review 2026-10-13 A1: the stage 4 start check is one more pod job after the engine probes
    assert names == ["setup", "load_gate", "validate_vllm", "start_check", "calibrate_vllm", "calibrate_rungs_vllm", "plan", "run", "score", "collect"]
    assert any(f"setup_runpod CHARTER_DATASETS=squad_v2 musique qasper PREFETCH_MODELS={PREFETCH_FULL}" in c for c in calls), \
        [c for c in calls if "setup" in c.lower()]
    assert any("preflight Qwen/Qwen3-14B http://localhost:8000 BACKENDS=vllm" in c for c in calls), \
        [c for c in calls if "preflight" in c]
    assert any("teardown_pod pod123" in c and "ASSUME_YES=1" in c and "SSH=root@pod.test" in c for c in calls), \
        [c for c in calls if "teardown" in c]
    assert not any("network-volume delete" in c for c in calls)  # owner rule 2026-10-22: the volume is never deleted
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
    assert list((land / "plots").glob("yield_*/yield_table.csv"))   # Spec Block A1 step ran
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
    assert "stage 14: analyze" in proc.stdout and "stage 12: pull" not in proc.stdout  # G18: analyze last
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
    assert "predecessor 'pull' has not passed" in proc.stderr  # G18: teardown follows the pull
    # --redo never waives the predecessor rule, and a bare --redo is refused
    proc = _master(world, "--only", "teardown", "--redo", "--yes", "teardown")
    assert proc.returncode == 1 and "predecessor 'pull' has not passed" in proc.stderr
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
    short.write_text(world["profile"].read_text(encoding="utf-8").replace("SEATBELT=12h", "SEATBELT=2h"), encoding="utf-8")
    world["profile"] = short
    proc = _master(world)
    assert proc.returncode == 1
    # readiness 0 (5 s) + setup 1 + validate (10 load gate + 25 + 40 start check) + calibrate (60 + 150, ADR-0154
    # rung calibration 9,000 s) + plan 15 + run 60 + seal 10 + score 180 + collect 30 + pull margin 90 (G16)
    assert "SEATBELT=2h (120 min) is shorter than the 671 min the stages need" in proc.stdout
    assert not any("provision_pod" in c for c in _calls(world))


def test_shipped_profiles_cover_their_stage_bounds_and_name_registered_models() -> None:
    import importlib
    cb = importlib.import_module("src.orchestration.cache_budget")
    for p in sorted((EXPDIR / "profiles").glob("S*.env")):
        proc = subprocess.run(["bash", "-c", f'set -a; source "{EXPDIR}/profiles/_common.env"; source "{p}"; '
                               'printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s" "$MODEL_SLUG" "$SEATBELT" "$HOURS" "$SETUP_BOUND_MIN" "$ENGINES" "$CALIBRATE" "${SCORE_BOUND_MIN:-180}" "$CALIBRATE_RUNGS_BOUND_S" "$FLOOR_AVG_SEQ_TOKENS" "${SUITE_ON_POD:-0}" "${START_CHECK:-1}" "${PULL_MARGIN_MIN:-90}"'],
                              capture_output=True, text=True, env=PROFILE_ENV)
        slug, seatbelt, hours, setup_min, engines, calibrate, score_min, rungs_bound_s, floor_tokens, suite_on_pod, start_check, pull_min = proc.stdout.split("|")
        assert slug in cb.MODEL_KV, f"{p.name}: MODEL_SLUG={slug} is not in cache_budget.MODEL_KV {sorted(cb.MODEL_KV)}"
        # ADR-0155: every shipped profile sizes the floor table on the anchor shape
        assert floor_tokens == "4779", f"{p.name}: FLOOR_AVG_SEQ_TOKENS={floor_tokens} is not the registered anchor shape"
        n_srv = len([e for e in engines.split() if e != "hf"]); n_cal = len(calibrate.split())
        # S0F-88 review L4: plus the stage 4 load gate (10 min) and the pod suite (30 min when SUITE_ON_POD=1);
        # G16 (2026-10-17): plus the readiness wait (900 s), the start check (2,400 s), the seal (600 s) and a
        # the profile's pull margin (ADR-0176 Group C; the common default 90 = S0 attempt 5's 68 min)
        need = (15 + int(setup_min) + 10 + (30 if suite_on_pod == "1" else 0) + n_srv * 25 + (40 if start_check == "1" else 0)
                + n_cal * 60 + n_cal * int(rungs_bound_s) // 60 + 15 + int(hours) * 60 + 10 + int(score_min) + 30 + int(pull_min))
        unit = seatbelt[-1]; n = int(seatbelt.split()[0][:-1])
        assert unit in ("h", "m"), f"{p.name}: SEATBELT={seatbelt} must be <n>h or <n>m (G30)"
        have = n * 60 if unit == "h" else n
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


def test_the_plan_stage_refuses_a_run_projected_past_hours_and_prices_the_whole_pod_b5_p4(
    world: Dict[str, Path],
) -> None:
    # S0F-88 B-5 (R9): the measured projection (1.5 h) exceeds HOURS=1
    proc = _master(world, "--yes", "provision", CAGE_TEST_PROJECTION_S="5400")
    assert proc.returncode == 1
    assert "run projection (S0F-88 B-5): 1.50 h against the 1 h bound" in proc.stdout
    assert "the projected run (1.50 h) exceeds HOURS=1" in proc.stdout
    assert _state(world)["stages"]["plan"]["status"] == "failed"
    # S0F-88 P-4: the cost estimate covers every stage bound (671 min with the G16 terms), not HOURS alone
    provision = [c for c in _calls(world) if "provision_pod" in c]
    assert provision and all("--hours 11.2 " in c for c in provision)


def test_the_plan_stage_refuses_a_step_without_a_measured_duration_b5(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", CAGE_TEST_UNMEASURED="1")
    assert proc.returncode == 1
    assert "1 plan step(s) have no measured duration (e.g. cell k3)" in proc.stdout


def test_an_unexpected_teardown_exit_code_fails_stage_14(world: Dict[str, Path]) -> None:
    # expected 0: a 2 (the delete itself failed) fails the stage, nothing is swept
    proc = _master(world, "--yes", "provision", "--yes", "teardown", CAGE_TEST_TEARDOWN_RC="2")
    assert proc.returncode == 1
    assert "rc=2, expected 0 (log:" in proc.stdout  # review 2026-10-22 M2: HEAD printed "expected 0|1"
    assert not any("network-volume delete" in c for c in _calls(world))
    assert _state(world)["stages"]["teardown"]["status"] == "failed"


def test_run_exit_2_with_stop_failed_only_is_accepted_and_noted(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--yes", "teardown", CAGE_TEST_RUN_STOP_FAILED="1")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    st = _state(world)
    assert st["stages"]["run"]["status"] == "passed"
    assert any("STOP FAILED" in n for n in st["stages"]["run"]["notes"])
    # review 2026-10-17 batch 4 (H1): a passing run copies no unsealed tree
    assert "custody (G6)" not in proc.stdout and not any("run_unsealed" in c for c in _calls(world))


def test_rung_calibration_job_feeds_the_plan(world: Dict[str, Path]) -> None:
    # ADR-0154: stage 5 runs calibrate-rungs per engine AFTER the floor table is
    # on the pod, with the session's rungs, and stage 6 registers the artifact.
    proc = _master(world, "--yes", "provision", "--to", "plan")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    cmd = (world["home"] / ".cage_jobs" / "calibrate_rungs_vllm.cmd").read_text(encoding="utf-8")
    assert "run_campaign.py calibrate-rungs --session a --engine vllm --floor-table results/calibration/floor_table.json" in cmd
    assert "--calibration results/calibration/S1_vllm.json --rungs 1,0.5" in cmd
    assert "--out results/calibration/rungs_S1_vllm.json --out-root results/rung-calibration" in cmd
    assert "--query-manifest squad_v2=data/manifests/squad_v2_50x3_seed42.json" in cmd
    assert " start " not in cmd, "the job's relaunches are the plan's own; no engine start before it"
    plan_cmd = (world["home"] / ".cage_jobs" / "plan.cmd").read_text(encoding="utf-8")
    assert "--rung-calibration vllm=results/calibration/rungs_S1_vllm.json" in plan_cmd
    # ADR-0169: the RULER instrument's ladder is named beside the two anchors
    assert (
        "rung calibration (ADR-0154, ADR-0156, ADR-0169): rungs 1,0.5 per engine on the anchor, "
        "the RULER instrument (where registered) and the smallest class"
    ) in proc.stdout
    fetched = world["exp_root"] / "S1" / DATE / "extras" / "calibration" / "rungs_S1_vllm.json"
    assert json.loads(fetched.read_text(encoding="utf-8"))["rungs"]["1"]["label"] == "ESTIMATED"
    calls = _calls(world)
    # the scp of the floor table to the pod is the only call naming its pod path in clear (job bodies are base64)
    i_floor = next(i for i, c in enumerate(calls) if "results/calibration/floor_table.json" in c)
    i_rungs = next(i for i, c in enumerate(calls) if "calibrate_rungs_vllm.cmd" in c)
    assert i_floor < i_rungs, "the rung job needs the floor table on the pod first"


def test_a_rung_that_is_not_estimated_fails_stage_5_before_the_plan(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_RUNGS_FAIL="1")
    assert proc.returncode == 1
    assert "stage calibrate FAILED" in proc.stdout
    assert not (world["home"] / ".cage_jobs" / "plan.cmd").exists(), "stage 6 must not run"
    st = _state(world)
    assert st["stages"]["calibrate"]["status"] == "failed" and "plan" not in st["stages"]


def test_stage_0_refuses_a_registration_problem_before_anything_bills(world: Dict[str, Path]) -> None:
    # gap triage 2026-10-09: the manifest shortfall and the unstaged dataset the
    # registered S1 profile hits at stage 6 are refused at stage 0, on the Mac
    proc = _master(world, "--yes", "provision",
                   CAGE_TEST_PREFLIGHT_PROBLEMS="--query-manifest qasper: trial 1 carries 50 ids but the primary cells register n=2000: shortfall 1950")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "REFUSED: --query-manifest qasper" in proc.stdout and "stage preflight-mac FAILED" in proc.stdout
    assert not any("provision_pod" in c for c in _calls(world)), "nothing may bill after a stage 0 refusal"


def test_stage_4_proves_the_lmcache_import_when_the_plan_needs_it(world: Dict[str, Path]) -> None:
    # S0F-57 (gap triage C1): stage 0 found an executable vLLM retr-store cell, so
    # the validate job imports lmcache in the pod venv; the fake venv has none
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_NEEDS_LMCACHE="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "NEEDS_LMCACHE=1" in proc.stdout and "stage validate FAILED" in proc.stdout
    assert "lmcache module does not import" in proc.stdout
    assert _state(world)["registration"]["needs_lmcache"] in ("1", 1)
    cmd = (world["home"] / ".cage_jobs" / "validate_vllm.cmd").read_text(encoding="utf-8")
    assert "import lmcache" in cmd
    log = (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "validate_vllm.log").read_text(encoding="utf-8")
    assert "LMCACHE_IMPORT_MISSING" in log and "PREFLIGHT PASS" in log
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists(), "stage 5 must not run"


def test_stage_4_refuses_a_pod_whose_own_memory_cannot_hold_the_lmcache_store(world: Dict[str, Path]) -> None:
    # S0F-88 U2 (review H1): /proc/meminfo shows the host's memory inside a pod;
    # the probe reads the container's cgroup, and its failure fails stage 4
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_NEEDS_LMCACHE="1",
                   CAGE_TEST_LMCACHE_IMPORT_RC="0", CAGE_TEST_POD_MEMORY_RC="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "memory left to the pod container" in proc.stdout and "POD_MEMORY_FAILED" in proc.stdout
    cmd = (world["home"] / ".cage_jobs" / "validate_vllm.cmd").read_text(encoding="utf-8")
    assert "probe_pod_memory.py --need-gib" in cmd and "/proc/meminfo" not in cmd
    ok = _master(world, "--from", "validate", "--redo", "--to", "validate", CAGE_TEST_NEEDS_LMCACHE="1",
                 CAGE_TEST_LMCACHE_IMPORT_RC="0", CAGE_TEST_POD_MEMORY_RC="0")
    assert ok.returncode == 0, ok.stdout[-3000:]
    assert "POD_MEMORY_OK" in ok.stdout


def test_stage_3_installs_lmcache_when_the_profile_opts_in(world: Dict[str, Path], tmp_path: Path) -> None:
    # S0F-57 / ADR-0158: INSTALL_LMCACHE=1 (a profile knob; _common.env defaults it to 0)
    # reaches the bootstrap and its proof line is required
    prof = tmp_path / "S1_lmcache.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8") + "\nINSTALL_LMCACHE=1\n", encoding="utf-8")
    world["profile"] = prof
    proc = _master(world, "--yes", "provision", "--to", "setup")
    assert proc.returncode == 0, proc.stdout[-3000:]
    cmd = (world["home"] / ".cage_jobs" / "setup.cmd").read_text(encoding="utf-8")
    assert "INSTALL_LMCACHE='1'" in cmd
    assert "installed beside vllm" in (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "setup.log").read_text(encoding="utf-8")
    proc = _master(world, "--from", "setup", "--redo", "--to", "setup", CAGE_TEST_LMCACHE_FAIL="1")
    assert proc.returncode == 1 and "stage setup FAILED" in proc.stdout


def test_stage_4_skips_the_lmcache_probe_when_no_cell_needs_it(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "validate")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "NEEDS_LMCACHE=0" in proc.stdout
    assert "import lmcache" not in (world["home"] / ".cage_jobs" / "validate_vllm.cmd").read_text(encoding="utf-8")


def test_a_smallest_class_rung_that_is_not_estimated_fails_stage_5_too(world: Dict[str, Path]) -> None:
    # ADR-0156: the fake job exits 0 with the anchor ESTIMATED but the smallest
    # class LADDER_EXHAUSTED; the master's own check over the classes block fails
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_SMALL_RUNGS_FAIL="1")
    assert proc.returncode == 1
    assert "stage calibrate FAILED" in proc.stdout
    assert "smallest 348" in proc.stdout and "LADDER_EXHAUSTED" in proc.stdout
    assert not (world["home"] / ".cage_jobs" / "plan.cmd").exists()


def test_run_exit_3_dry_window_failed_fails_the_stage_loudly(world: Dict[str, Path]) -> None:
    # ADR-0153: a dry window off the regime skips a whole (engine, class); the
    # tree is partial, the master names the pairs and stops for the owner.
    proc = _master(world, "--yes", "provision", "--yes", "teardown", CAGE_TEST_RUN_DRY_FAILED="1")
    assert proc.returncode == 1
    assert "exit 3 = DRY WINDOW FAILED (ADR-0153)" in proc.stdout
    assert "DRY WINDOW FAILED: engine=vllm class=4779 tokens r=0.5" in proc.stdout
    st = _state(world)
    assert st["stages"]["run"]["status"] == "failed"
    assert any("exit 3: DRY WINDOW FAILED" in n for n in st["stages"]["run"]["notes"])


def test_stage_4_starts_every_first_time_serving_configuration_once(world: Dict[str, Path], tmp_path: Path) -> None:
    # Review 2026-10-13 A1 (items 29, 32): the fp8 starts, the LMCache start at
    # the plan's store and the GPU reranker beside vLLM run once at stage 4.
    two = tmp_path / "S1_two.env"
    two.write_text(world["profile"].read_text(encoding="utf-8").replace('ENGINES="vllm hf"', 'ENGINES="vllm sglang hf"'), encoding="utf-8")
    world["profile"] = two
    proc = _master(world, "--yes", "provision", "--to", "validate")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    cmd = (world["home"] / ".cage_jobs" / "start_check.cmd").read_text(encoding="utf-8")
    assert "SGLANG_KV_CACHE_DTYPE=fp8_e4m3 bash scripts/2_serving/manage_sglang_server.sh restart" not in cmd  # through one()
    assert "one sglang_fp8_e4m3 sglang http://localhost:30000 manage_sglang_server.sh SGLANG_KV_CACHE_DTYPE=fp8_e4m3" in cmd
    assert "one vllm_fp8 vllm http://localhost:8000 manage_vllm_server.sh VLLM_KV_CACHE_DTYPE=fp8" in cmd
    assert "probe_reranker_device.py --out logs/rerank_device_probe_beside_vllm.json" in cmd
    assert cmd.index("manage_vllm_server.sh restart") < cmd.index("rerank_device_probe_beside_vllm")
    log = (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "start_check.log").read_text(encoding="utf-8")
    for marker in ("START_OK:sglang_fp8_e4m3", "START_OK:vllm_fp8", "START_OK:rerank_beside_vllm", "COMPUTE_APPS=0"):
        assert marker in log, marker
    assert "START_FAILED" not in log and _state(world)["stages"]["validate"]["status"] == "passed"
    calls = _calls(world)
    assert any("launcher sglang restart" in c and "kv=fp8_e4m3" in c for c in calls)
    assert any("launcher vllm restart" in c and "kv=fp8" in c for c in calls)
    if "LMCACHE_MAX_LOCAL_CPU_SIZE=" in cmd:  # the fake registration reports an LMCache plan
        assert "START_OK:lmcache_store_" in log and any("store=22" in c for c in calls)


def test_stage_4_start_check_fails_the_stage_when_a_first_time_start_fails(world: Dict[str, Path]) -> None:
    # a first-time configuration that starts but does not serve fails stage 4,
    # named, before calibration bills (the per-engine probes still pass)
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_START_CHECK_RC="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "START_FAILED:vllm_fp8" in proc.stdout
    assert "did not start or serve (review 2026-10-13 A1)" in proc.stdout
    assert _state(world)["stages"]["validate"]["status"] == "failed"
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists()


def test_stage_5_resumes_per_engine_and_carries_the_fail_fast_and_timeout_knobs(
    world: Dict[str, Path], tmp_path: Path
) -> None:
    # Review 2026-10-13 item 3 and item 21: an engine whose calibration and
    # ladder passed on this pod and build is not rerun on the resume; the
    # ladder job carries --fail-fast and CAGE_STEP_TIMEOUT_S, the run job the
    # timeout and the dry-window rerun flag, when the profile sets them.
    prof = tmp_path / "S1_knobs.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8").replace('ENGINES="vllm hf"', 'ENGINES="vllm sglang hf"').replace('CALIBRATE="vllm"', 'CALIBRATE="vllm sglang"')
                    + "\nLADDER_FAIL_FAST=1\nSTEP_TIMEOUT_S=5400\nDRY_TELEMETRY_RERUN=1\nSEATBELT=72h\nREHEARSAL_N=50\n", encoding="utf-8")
    # (G15: DRY_TELEMETRY_RERUN is an S0 key, so this profile is a rehearsal)
    world["profile"] = prof
    proc = _master(world, "--yes", "provision", "--to", "calibrate", CAGE_TEST_RUNGS_FAIL="1", CAGE_TEST_RUNGS_FAIL_ENGINE="sglang")
    assert proc.returncode == 1, proc.stdout[-4000:]
    st = _state(world)
    assert st["stages"].get("calibrate", {}).get("status") == "failed", proc.stdout[-3000:] + proc.stderr[-1000:]
    done_key = f"{st['pod']['id']}:{st['build']['sha']}"
    assert st["calibration"]["cell_done"]["vllm"] == done_key and st["calibration"]["rungs_done"]["vllm"] == done_key
    assert "rungs_done" not in st["calibration"] or "sglang" not in st["calibration"]["rungs_done"]
    rungs_cmd = (world["home"] / ".cage_jobs" / "calibrate_rungs_vllm.cmd").read_text(encoding="utf-8")
    assert " --fail-fast" in rungs_cmd and "CAGE_STEP_TIMEOUT_S=5400" in rungs_cmd
    before = sum(1 for c in _calls(world) if "calibrate_rungs_vllm.cmd" in c or "calibrate_vllm.cmd" in c)
    proc = _master(world, "--from", "calibrate", "--redo", "--to", "run")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    assert "calibrate_cell vllm already passed on this pod and build" in proc.stdout
    assert "rung calibration vllm already passed on this pod and build" in proc.stdout
    after = sum(1 for c in _calls(world) if "calibrate_rungs_vllm.cmd" in c or "calibrate_vllm.cmd" in c)
    assert after == before  # vLLM's two jobs were not resubmitted
    assert _state(world)["calibration"]["rungs_done"]["sglang"] == done_key
    run_cmd = (world["home"] / ".cage_jobs" / "run.cmd").read_text(encoding="utf-8")
    assert "CAGE_STEP_TIMEOUT_S=5400" in run_cmd and run_cmd.rstrip().endswith("--dry-window-telemetry-rerun")
    assert "--fail-fast" not in run_cmd  # RUN_FAIL_FAST unset on this profile
    plan_cmd = (world["home"] / ".cage_jobs" / "plan.cmd").read_text(encoding="utf-8")
    assert "--front-load-first-contacts" not in plan_cmd  # FRONT_LOAD unset


def test_stage_4_fails_when_the_sglang_server_does_not_report_cached_tokens(world: Dict[str, Path], tmp_path: Path) -> None:
    # Review 2026-10-13 item 8 (U7 was pinned as text): /server_info answers
    # enable_cache_report=false, stage 4 fails naming it, before calibration bills
    two = tmp_path / "S1_two.env"
    two.write_text(world["profile"].read_text(encoding="utf-8").replace('ENGINES="vllm hf"', 'ENGINES="vllm sglang hf"'), encoding="utf-8")
    world["profile"] = two
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_CACHE_REPORT="false")
    assert proc.returncode == 1, proc.stdout[-3000:]
    # the probe sets rc=1 on CACHE_REPORT_OFF, so the job itself fails the
    # stage and job_run prints the job's tail, which names the marker
    assert "CACHE_REPORT_OFF" in proc.stdout and "stage validate FAILED" in proc.stdout
    assert _state(world)["stages"]["validate"]["status"] == "failed"
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists()  # nothing billed past stage 4
    log = (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "failed")
    assert any("CACHE_REPORT_OFF" in p.read_text(encoding="utf-8") for p in log.glob("validate_sglang_*.log"))


def test_the_run_job_strips_an_exported_kv_ceiling_pin(world: Dict[str, Path]) -> None:
    # Review 2026-10-13 item 8 (ADR-0166 L2 had no test): CAGE_KV_CEILING in
    # the master's environment never reaches the run job (RUN_ENV_UNSET); the
    # fake run refuses when it does
    proc = _master(world, "--yes", "provision", "--to", "run", CAGE_KV_CEILING="0.9")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    run_cmd = (world["home"] / ".cage_jobs" / "run.cmd").read_text(encoding="utf-8")
    assert "-u CAGE_KV_CEILING" in run_cmd
    log = (world["exp_root"] / "S1" / DATE / "logs" / "runner" / "run.log").read_text(encoding="utf-8")
    assert "REFUSED: CAGE_KV_CEILING is set" not in log and "run argv:" in log


def test_the_front_load_and_run_fail_fast_keys_reach_the_plan_and_run_jobs(world: Dict[str, Path], tmp_path: Path) -> None:
    # Review 2026-10-13 A2 and B1: S0 profile keys, carried as flags
    prof = tmp_path / "S1_speed.env"
    # G15: both are S0 keys, refused on a registered profile, so this one is a rehearsal
    prof.write_text(world["profile"].read_text(encoding="utf-8") + "\nFRONT_LOAD=1\nRUN_FAIL_FAST=infra\nREHEARSAL_N=50\n", encoding="utf-8")
    world["profile"] = prof
    proc = _master(world, "--yes", "provision", "--to", "run")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    plan_cmd = (world["home"] / ".cage_jobs" / "plan.cmd").read_text(encoding="utf-8")
    assert " --front-load-first-contacts --out results/calibration/plan_" in plan_cmd
    run_cmd = (world["home"] / ".cage_jobs" / "run.cmd").read_text(encoding="utf-8")
    assert run_cmd.rstrip().endswith("--seal --fail-fast infra")


def test_the_pod_memory_cloud_type_and_archive_dir_reach_their_consumers(world: Dict[str, Path], tmp_path: Path) -> None:
    # Review 2026-10-13: item 2 (the pod's allocated memory recorded at readiness
    # and passed to the stage 4 memory probe), item 24 (the profile's CLOUD_TYPE
    # and PORTS reach the provisioner), item 17 (the archive dir reaches the
    # backup daemon's start and stop).
    prof = tmp_path / "S1_arch.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8") + "\nBACKUP_ARCHIVE_DIR=/workspace/backup-archives\nBACKUP_ARCHIVE_EVERY=3\n", encoding="utf-8")
    world["profile"] = prof
    proc = _master(world, "--yes", "provision", "--to", "run")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    st = _state(world)
    assert st["pod"]["memory_gb"] == "172"
    steps = world["exp_root"] / "S1" / DATE / "extras" / "steps"
    (prov,) = [p for p in steps.glob("provision_*provision_pod_in*.log")]
    assert "--cloud-type SECURE --ports 22/tcp" in prov.read_text(encoding="utf-8")
    vllm_cmd = (world["home"] / ".cage_jobs" / "validate_vllm.cmd").read_text(encoding="utf-8")
    if "probe_pod_memory.py" in vllm_cmd:  # the fake registration reports an LMCache plan
        assert "probe_pod_memory.py --need-gib" in vllm_cmd and "--pod-gib 172" in vllm_cmd
    calls = _calls(world)
    assert any("gcs_backup_daemon.sh start" in c and "CAGE_BACKUP_ARCHIVE_DIR=/workspace/backup-archives CAGE_BACKUP_ARCHIVE_EVERY=3" in c for c in calls)


def _partial_profile(world: Dict[str, Path], tmp_path: Path, *, rehearsal: bool = True) -> None:
    text = world["profile"].read_text(encoding="utf-8") + "\nPARTIAL_TREE_OK=1\n"
    if rehearsal:
        text += "REHEARSAL_N=50\n"
    prof = tmp_path / "S1_partial.env"
    prof.write_text(text, encoding="utf-8")
    world["profile"] = prof


def test_a_failed_dry_window_passes_as_partial_and_is_sealed_under_the_s0_consent(
    world: Dict[str, Path], tmp_path: Path
) -> None:
    # Decision 3 (2026-10-13): under PARTIAL_TREE_OK=1 on a rehearsal profile
    # the run stage passes as PARTIAL, the seal runs, and the state records it.
    _partial_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "seal", CAGE_TEST_RUN_DRY_FAILED="1")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    assert "[PARTIAL] run ended with exit 3 = DRY WINDOW FAILED" in proc.stdout
    assert "PARTIAL tree accepted on the profile's PARTIAL_TREE_OK=1" in proc.stdout
    assert "sealing a PARTIAL tree (exit 3: dry window(s) failed" in proc.stdout
    st = _state(world)
    assert st["stages"]["run"]["status"] == "passed" and st["stages"]["seal"]["status"] == "passed"
    assert st["run"]["partial"].startswith("exit 3")
    assert any("PARTIAL tree accepted" in n for n in st["stages"]["run"]["notes"])
    assert any(".cage_jobs/seal.cmd" in c for c in _calls(world))


def test_a_failed_cell_passes_as_partial_but_a_crash_still_fails(world: Dict[str, Path], tmp_path: Path) -> None:
    # exit 1 with the run's totals line = failed cells (PARTIAL); exit 1 with
    # no totals line = a crash, which fails the stage even with the consent.
    _partial_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "run", CAGE_TEST_RUN_CELL_FAILED="1")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    assert "[PARTIAL] run ended with exit 1 = failed cell(s) or relaunch(es)" in proc.stdout
    assert _state(world)["run"]["partial"].startswith("exit 1")
    proc = _master(world, "--from", "run", "--redo", "--to", "run", CAGE_TEST_RUN_CRASH="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] run ended" in proc.stdout and _state(world)["stages"]["run"]["status"] == "failed"


def test_the_partial_consent_is_refused_on_a_registered_profile_before_anything_bills(
    world: Dict[str, Path], tmp_path: Path
) -> None:
    _partial_profile(world, tmp_path, rehearsal=False)
    proc = _master(world, "--yes", "provision", "--to", "run")
    assert proc.returncode == 1
    assert "PARTIAL_TREE_OK=1 on a registered run (REHEARSAL_N empty)" in proc.stdout
    assert _state(world)["stages"]["preflight-mac"]["status"] == "failed"
    assert not any("pod create" in c for c in _calls(world))


def test_floor_avg_seq_tokens_must_be_the_anchor_shape(world: Dict[str, Path], tmp_path: Path) -> None:
    # ADR-0155: stage 0 refuses a profile sized on another shape before anything bills
    wrong = tmp_path / "S1_shape.env"
    wrong.write_text(world["profile"].read_text(encoding="utf-8") + "\nFLOOR_AVG_SEQ_TOKENS=3100\n", encoding="utf-8")
    world["profile"] = wrong
    proc = _master(world)
    assert proc.returncode == 1
    assert "FLOOR_AVG_SEQ_TOKENS=3100 but session 'a' registers 4779 served tokens" in proc.stdout
    assert not any("provision_pod" in c for c in _calls(world))


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


def test_clean_room_pods_refuse_and_a_kept_volume_is_noted_at_stage_0(world: Dict[str, Path]) -> None:
    # owner rule 2026-10-22 (ADR-0176 Group B): a volume kept from an earlier run
    # is a note at stage 0, never a refusal; a foreign pod still refuses
    proc = _master(world, CAGE_TEST_VOLUMES='[{"id": "leftover"}]')
    assert proc.returncode == 10, proc.stdout[-2500:]
    assert "NOTE: network volume(s) exist before this experiment" in proc.stdout
    assert "clean room: network volumes exist" not in proc.stdout
    st = _state(world)
    assert st["stages"]["preflight-mac"]["status"] == "passed"
    assert any("kept, billing storage" in n for n in st["stages"]["preflight-mac"]["notes"])
    proc = _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_PODS='[{"id": "stranger"}]')
    assert proc.returncode == 1 and "clean room: pods exist before this experiment" in proc.stdout


def test_dirty_build_fails_stage_0(world: Dict[str, Path]) -> None:
    proc = _master(world, CAGE_TEST_DIRTY="1")
    assert proc.returncode == 1 and "BUILD_INFO is not dirty=0" in proc.stdout


def test_mac_suite_zero_skips_only_the_suite_loudly_and_records_it(world: Dict[str, Path], tmp_path: Path) -> None:
    # ADR-0145: the default runs the suite gate; MAC_SUITE=0 skips that one step,
    # says so, notes it in the state, and every other stage 0 check still runs.
    assert _master(world).returncode == 10
    # P-1: the suite is one run_tests call; the live-load gate is its own call
    suite = lambda calls: sum(1 for c in calls if c.startswith("run_tests") and "test_load_gate_p1" not in c)
    gate = lambda calls: sum(1 for c in calls if c.startswith("run_tests") and "test_load_gate_p1" in c)
    assert suite(_calls(world)) == 1 and gate(_calls(world)) == 1
    skip = tmp_path / "S1_nosuite.env"
    # G15 / ADR-0145: MAC_SUITE=0 is refused on a registered profile, so this one is a rehearsal
    skip.write_text(world["profile"].read_text(encoding="utf-8") + "\nMAC_SUITE=0\nREHEARSAL_N=50\n", encoding="utf-8")
    world["profile"] = skip
    proc = _master(world, "--plan")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Mac test suite SKIPPED (MAC_SUITE=0" in proc.stdout and "[plan] Mac test suite" not in proc.stdout
    proc = _master(world, "--from", "preflight-mac", "--redo", "--to", "preflight-mac")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Mac test suite SKIPPED (MAC_SUITE=0" in proc.stdout
    calls = _calls(world)
    assert suite(calls) == 1          # the suite did not run again
    assert gate(calls) == 2           # the P-1 gate runs even under MAC_SUITE=0
    assert sum(1 for c in calls if c.startswith("package_repo")) == 2        # the other checks did
    assert sum(1 for c in calls if c.startswith("runpodctl gpu list")) == 2
    st = _state(world)
    assert st["stages"]["preflight-mac"]["status"] == "passed"
    assert any("MAC_SUITE=0" in n for n in st["stages"]["preflight-mac"]["notes"])


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


def test_pod_jobs_that_start_engines_put_the_driver_venv_first_on_path(world: Dict[str, Path]) -> None:
    # Live 2026-10-07: the validate job's `vllm serve` died with "failed to run
    # command 'vllm'" because nothing put cage-env/bin on the pod job's PATH.
    proc = _master(world, "--yes", "provision", "--to", "run")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    prefix = f"export PATH={world['pod_repo']}/cage-env/bin:$PATH"
    for name in ("validate_vllm", "calibrate_vllm", "calibrate_rungs_vllm", "run"):
        cmd = (world["home"] / ".cage_jobs" / f"{name}.cmd").read_text(encoding="utf-8")
        assert cmd.startswith(f"cd {world['pod_repo']} && {prefix}"), (name, cmd[:160])
    for name in ("validate_vllm", "calibrate_vllm", "calibrate_rungs_vllm"):
        assert "VLLM_START_TIMEOUT=600" in (world["home"] / ".cage_jobs" / f"{name}.cmd").read_text(encoding="utf-8")
    # ADR-0162: the SGLang start timeout rides every engine job beside the vLLM one
    for name in ("validate_vllm", "calibrate_vllm", "calibrate_rungs_vllm", "run"):
        assert " SGLANG_START_TIMEOUT=600" in (world["home"] / ".cage_jobs" / f"{name}.cmd").read_text(encoding="utf-8"), name
    # preflight gate (p) judges exactly the datasets stage 3 staged (the profile's CHARTER_DATASETS)
    assert "CAGE_DATASETS=squad_v2,musique,qasper" in (world["home"] / ".cage_jobs" / "validate_vllm.cmd").read_text(encoding="utf-8")
    assert "cage-env/bin" not in (world["home"] / ".cage_jobs" / "setup.cmd").read_text(encoding="utf-8")
    # the profile's engine levers are not in the job commands unless set
    for name in ("validate_vllm", "calibrate_vllm", "calibrate_rungs_vllm", "run"):
        assert "CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH" not in (world["home"] / ".cage_jobs" / f"{name}.cmd").read_text(encoding="utf-8")


def test_the_sglang_piecewise_lever_is_forwarded_to_every_engine_job(world: Dict[str, Path], tmp_path: Path) -> None:
    # S0F-35: the profile sets the lever; the pod sees it only through the job command.
    lever = tmp_path / "S1_lever.env"
    lever.write_text(world["profile"].read_text(encoding="utf-8") + "\nCAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH=1\n", encoding="utf-8")
    world["profile"] = lever
    proc = _master(world, "--yes", "provision", "--to", "run")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    for name in ("validate_vllm", "calibrate_vllm", "calibrate_rungs_vllm", "run"):
        cmd = (world["home"] / ".cage_jobs" / f"{name}.cmd").read_text(encoding="utf-8")
        assert " CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH=1 && " in cmd, (name, cmd[:200])
    assert "CAGE_SGLANG_DISABLE_PIECEWISE_CUDA_GRAPH" not in (world["home"] / ".cage_jobs" / "setup.cmd").read_text(encoding="utf-8")


def test_a_stage_0_redo_mid_run_keeps_the_run_id_and_tolerates_its_own_pod_and_volume(world: Dict[str, Path]) -> None:
    # S0F-36: a code fix mid-run must re-ship through stage 0 and stage 2 while
    # this landing's own pod and volume exist; a foreign resource still refuses.
    proc = _master(world, "--yes", "provision", "--to", "ship")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-800:]
    rid = _state(world)["run_id"]
    ours = {"CAGE_TEST_PODS": '[{"id": "pod123"}]', "CAGE_TEST_VOLUMES": '[{"id": "vol123"}]'}
    proc = _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_SHA="fedcba9876543210fedcba9876543210fedcba98", **ours)
    assert proc.returncode == 0, proc.stdout[-2500:] + proc.stderr[-800:]
    assert "run id kept" in proc.stdout
    st = _state(world)
    assert st["run_id"] == rid and st["build"]["sha"].startswith("fedcba")
    # P-2: after the fix the Mac HEAD is the new commit the tarball carries.
    proc = _master(world, "--only", "ship", "--redo", CAGE_TEST_SHA="fedcba9876543210fedcba9876543210fedcba98", **ours)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-800:]
    assert _state(world)["stages"]["ship"]["status"] == "passed"
    proc = _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_PODS='[{"id": "pod123"}, {"id": "stranger"}]', CAGE_TEST_VOLUMES='[{"id": "vol123"}]')
    assert proc.returncode == 1 and "clean room: pods exist" in proc.stdout


def test_ship_refuses_a_tarball_older_than_the_mac_head_p2(world: Dict[str, Path]) -> None:
    # P-2 (S0F-88, 2026-10-11): a fix committed after stage 0, then
    # `--only ship --redo`, shipped the tarball stage 0 built (4df4da9, not
    # cef4cde). Ship now compares the Mac HEAD with the packaged sha.
    proc = _master(world, "--yes", "provision", "--to", "ship")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-800:]
    ours = {"CAGE_TEST_PODS": '[{"id": "pod123"}]', "CAGE_TEST_VOLUMES": '[{"id": "vol123"}]'}
    moved = "1111111111111111111111111111111111111111"
    proc = _master(world, "--only", "ship", "--redo", CAGE_TEST_HEAD_SHA=moved, **ours)
    assert proc.returncode == 1, proc.stdout[-2000:]
    assert f"Mac HEAD is {moved}" in proc.stdout
    assert "--only preflight-mac --redo" in proc.stdout
    # the documented remedy: repackage at the new HEAD, then ship passes
    assert _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_SHA=moved, **ours).returncode == 0
    proc = _master(world, "--only", "ship", "--redo", CAGE_TEST_SHA=moved, **ours)
    assert proc.returncode == 0, proc.stdout[-2000:]


def test_stage_0_runs_the_live_load_gate_and_refuses_a_skip_p1(world: Dict[str, Path]) -> None:
    # P-1 (S0F-88, R17): the gate runs at stage 0 with CAGE_LOAD_GATE=1 even
    # under MAC_SUITE=0, and a gate that skipped (env lost) is not a pass.
    proc = _master(world, "--only", "preflight-mac")
    assert proc.returncode == 0, proc.stdout[-2000:]
    gate_calls = [c for c in _calls(world) if "test_load_gate_p1.py" in c]
    assert gate_calls and "CAGE_LOAD_GATE=1" in gate_calls[0]
    proc = _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_LOAD_GATE_OUT="1 skipped")
    assert proc.returncode == 1 and "live-load gate did not run and pass" in proc.stdout


def test_validate_runs_the_preflight_on_vllm_only_and_engine_probes_on_sglang(world: Dict[str, Path], tmp_path: Path) -> None:
    # S0F-37: preflight_check.sh is vLLM's gate set; SGLang gets its own probes.
    two = tmp_path / "S1_two.env"
    two.write_text(world["profile"].read_text(encoding="utf-8").replace('ENGINES="vllm hf"', 'ENGINES="vllm sglang hf"'), encoding="utf-8")
    world["profile"] = two
    proc = _master(world, "--yes", "provision", "--to", "validate")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    vllm_cmd = (world["home"] / ".cage_jobs" / "validate_vllm.cmd").read_text(encoding="utf-8")
    sgl_cmd = (world["home"] / ".cage_jobs" / "validate_sglang.cmd").read_text(encoding="utf-8")
    assert "preflight_check.sh" in vllm_cmd and "/flush_cache" not in vllm_cmd
    # S0F-54 (ADR-0149): the deferred flush, quoted so the remote shell never globs the "?"
    assert "preflight_check.sh" not in sgl_cmd and 'POST "http://localhost:30000/flush_cache?timeout=20"' in sgl_cmd
    # S0F-59 (ADR-0151): the thinking pin probe runs on BOTH engines, before the stop
    assert "probe_thinking_pin.py --backend vllm --api-base http://localhost:8000" in vllm_cmd
    assert "probe_thinking_pin.py --backend sglang --api-base http://localhost:30000" in sgl_cmd
    assert vllm_cmd.index("probe_thinking_pin.py") < vllm_cmd.index("manage_vllm_server.sh stop")
    # S0F-88 B-6: the device probe runs once, on the free GPU after the vLLM stop
    assert vllm_cmd.index("manage_vllm_server.sh stop") < vllm_cmd.index("probe_reranker_device.py")
    assert "probe_reranker_device.py" not in sgl_cmd
    log = (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "validate_sglang.log").read_text(encoding="utf-8")
    for marker in ("VALIDATE_API_OK", "RUNNING_REQS=0", "ENGINE_FLUSH_OK", "METRIC_OK sglang:token_usage", "METRIC_OK sglang:num_retracted_reqs",
                   "METRIC_OK sglang:num_running_reqs", "METRIC_OK sglang:num_queue_reqs", "POOL_RECORD_OK", "THINKING_PIN_OK backend=sglang", "COMPUTE_APPS=0",
                   "CACHE_REPORT_ON"):  # S0F-88 U7
        assert marker in log, marker
    vllm_log = (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "validate_vllm.log").read_text(encoding="utf-8")
    assert "THINKING_PIN_OK backend=vllm" in vllm_log and "RERANK_DEVICE_OK" in vllm_log
    assert "thinking_probe vllm http://localhost:8000" in _calls(world)
    assert "thinking_probe sglang http://localhost:30000" in _calls(world)
    assert _state(world)["stages"]["validate"]["status"] == "passed"


def test_validate_starts_at_the_largest_planned_pool_and_checks_it_a11_item4(world: Dict[str, Path]) -> None:
    # A11 item 4 (owner 2026-10-21): stage 0's largest pool rides the stage 4 start,
    # and the relaunch rule checks it before calibration bills
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_LARGEST_POOL="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    cmd = (world["home"] / ".cage_jobs" / "validate_vllm.cmd").read_text(encoding="utf-8")
    assert "env CAGE_KV_BUDGET_BYTES=48936960000 VLLM_MAX_MODEL_LEN=32768 bash scripts/2_serving/manage_vllm_server.sh start" in cmd
    assert cmd.index("pool-check --engine vllm --planned-tokens 298687 --planned-bytes 48936960000") < cmd.index("manage_vllm_server.sh stop")
    assert any(c.startswith("pool_check --engine vllm --planned-tokens 298687") for c in _calls(world))
    assert "LARGEST_POOL_OK engine=vllm" in proc.stdout
    # a pool outside the rule fails stage 4, and nothing calibrates
    proc = _master(world, "--only", "validate", "--redo", CAGE_TEST_LARGEST_POOL="1", CAGE_TEST_POOL_CHECK_RC="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "LARGEST_POOL_FAILED engine=vllm reason=test" in proc.stdout and "stage validate FAILED" in proc.stdout
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists()


def test_validate_skips_the_largest_pool_start_when_the_plan_has_none_a11_item4(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "validate")
    assert proc.returncode == 0, proc.stdout[-3000:]
    cmd = (world["home"] / ".cage_jobs" / "validate_vllm.cmd").read_text(encoding="utf-8")
    assert "pool-check" not in cmd and "env bash scripts/2_serving/manage_vllm_server.sh start" in cmd
    assert "no budgeted single-GPU pool in the plan; the largest-pool start is skipped (A11 item 4)" in proc.stdout


def test_validate_fails_when_the_thinking_pin_probe_fails(world: Dict[str, Path]) -> None:
    # S0F-59 (ADR-0151): the served text still carries the thinking marker (the
    # 2026-10-08 landing's SGLang shape): stage 4 fails and names the probe.
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_THINKING_RC="1")
    assert proc.returncode == 1, proc.stdout[-3000:] + proc.stderr[-1000:]
    # the probe's exit code fails the job (rc=1) and the job step fails the
    # stage; the job log tail the master prints names the probe's verdict
    assert "[FAIL] job validate_vllm: wait: rc=1" in proc.stdout
    assert "THINKING_PIN_FAILED backend=vllm reason=test" in proc.stdout
    assert "stage validate FAILED" in proc.stdout
    assert "thinking_probe vllm http://localhost:8000" in _calls(world)
    # the engine was still stopped after the failed probe (the stop follows the probe)
    calls = _calls(world)
    assert calls.index("thinking_probe vllm http://localhost:8000") < calls.index("launcher vllm stop")
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists(), "stage 5 must not run"


def test_validate_refuses_an_engine_without_a_registered_probe_set(world: Dict[str, Path], tmp_path: Path) -> None:
    lm = tmp_path / "S1_lm.env"
    lm.write_text(world["profile"].read_text(encoding="utf-8").replace('ENGINES="vllm hf"', 'ENGINES="vllm lmdeploy hf"'), encoding="utf-8")
    world["profile"] = lm
    proc = _master(world, "--yes", "provision", "--to", "validate")
    assert proc.returncode == 1
    assert "lmdeploy: no validation probe set is registered" in proc.stdout
    assert not (world["home"] / ".cage_jobs" / "validate_lmdeploy.cmd").exists()


def test_the_run_job_states_provider_and_hardware_for_the_campaign_seam(world: Dict[str, Path]) -> None:
    # S0F-40: campaign mode refuses without CAGE_PROVIDER and CAGE_HARDWARE.
    proc = _master(world, "--yes", "provision", "--to", "run")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    cmd = (world["home"] / ".cage_jobs" / "run.cmd").read_text(encoding="utf-8")
    assert "CAGE_PROVIDER=runpod CAGE_HARDWARE='NVIDIA H100 80GB HBM3 x 1' " in cmd, cmd[:400]
    # ADR-0160 (S0 attempt 2, 2026-10-10): the rung ladder's windows run the same
    # seam; the job carries the same env and the same poison strip as the run job
    rungs = (world["home"] / ".cage_jobs" / "calibrate_rungs_vllm.cmd").read_text(encoding="utf-8")
    assert "CAGE_PROVIDER=runpod CAGE_HARDWARE='NVIDIA H100 80GB HBM3 x 1' " in rungs, rungs[:400]
    assert "env -u CAGE_SLO_FLOORS_JSON" in rungs


def test_pod_results_dir_relinks_the_results_root_and_the_cells_land_there(world: Dict[str, Path], tmp_path: Path) -> None:
    # S0F-39: with POD_RESULTS_DIR set, stage 2 moves <repo>/results onto that
    # directory and leaves a symlink, so the run's cells land there unchanged.
    real = tmp_path / "cage-results"
    prof = tmp_path / "S1_results.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8") + f"\nPOD_RESULTS_DIR={real}\n", encoding="utf-8")
    world["profile"] = prof
    (world["pod_repo"] / "results" / "calibration" / "seed.txt").write_text("seed", encoding="utf-8")
    proc = _master(world, "--yes", "provision", "--to", "run")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert f"results root on the pod: {real}" in proc.stdout
    link = world["pod_repo"] / "results"
    assert link.is_symlink() and link.resolve() == real.resolve()
    assert (real / "calibration" / "seed.txt").read_text(encoding="utf-8") == "seed"      # moved, not lost
    rid = _state(world)["run_id"]
    assert (real / "camp1" / "a" / rid / "cells" / "k1" / "window_squad_v2-01" / "regime.json").is_file()
    # a second ship (a mid-run re-ship) keeps the link and the data
    proc = _master(world, "--only", "ship", "--redo", CAGE_TEST_PODS='[{"id": "pod123"}]', CAGE_TEST_VOLUMES='[{"id": "vol123"}]')
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert link.is_symlink() and (real / "camp1" / "a" / rid / "cells" / "k1" / "window_squad_v2-01" / "regime.json").is_file()
    # the default leaves the repo's own results directory alone
    world["profile"] = tmp_path / "S1.env"


def test_the_pod_suite_runs_once_per_shipped_sha_and_again_after_a_reship(world: Dict[str, Path], tmp_path: Path) -> None:
    # 2026-10-07: six stage 4 retries reran the 7-minute pod suite every time.
    prof = tmp_path / "S1_suite.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8") + "\nSUITE_ON_POD=1\n", encoding="utf-8")
    world["profile"] = prof
    proc = _master(world, "--yes", "provision", "--to", "validate")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    # one submit = one decode line writing the remote .cmd (status probes name the file too)
    submits = lambda: sum(1 for c in _calls(world) if c.rstrip().endswith("base64 -d > ~/.cage_jobs/suite_pod.cmd"))
    assert submits() == 1 and _state(world)["suite_pod"]["sha"] == _state(world)["build"]["sha"]
    proc = _master(world, "--only", "validate", "--redo")
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert "pod suite already passed on sha" in proc.stdout and submits() == 1
    ours = {"CAGE_TEST_PODS": '[{"id": "pod123"}]', "CAGE_TEST_VOLUMES": '[{"id": "vol123"}]'}
    assert _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_SHA="abcdefabcdefabcdefabcdefabcdefabcdefabcd", **ours).returncode == 0
    assert _master(world, "--only", "ship", "--redo", CAGE_TEST_SHA="abcdefabcdefabcdefabcdefabcdefabcdefabcd", **ours).returncode == 0
    proc = _master(world, "--only", "validate", "--redo")
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert "already passed" not in proc.stdout and submits() == 2


def test_a_transient_ssh_failure_is_retried_and_a_persistent_one_fails_the_step(world: Dict[str, Path]) -> None:
    # ADR-0161 (S0 attempt 2, 2026-10-10 01:29Z): one timed-out channel ended the
    # run while the pod job ran on. Idempotent round trips (pssh, pscp_to,
    # pscp_from) get three attempts on exit 255; a remote command's own failure
    # is still reported after one.
    assert _master(world, "--yes", "provision", "--to", "provision").returncode == 0
    (world["home"] / ".scp_fail").write_text("1\n", encoding="utf-8")   # the tarball copy times out once
    (world["home"] / ".ssh_fail").write_text("1\n", encoding="utf-8")   # the unpack round trip times out once
    proc = _master(world, "--from", "ship", "--to", "ship", CAGE_SSH_RETRY_S="0")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert proc.stdout.count("[retry] ssh channel to the pod failed (exit 255)") == 2
    assert "attempt 2 of 3" in proc.stdout
    assert _state(world)["stages"]["ship"]["status"] == "passed"
    # three consecutive channel failures fail the step and the stage, naming rc 255
    (world["home"] / ".ssh_fail").write_text("3\n", encoding="utf-8")
    proc = _master(world, "--from", "ship", "--redo", "--to", "ship", CAGE_SSH_RETRY_S="0")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] unpack on the pod: rc=255" in proc.stdout
    assert proc.stdout.count("[retry] ssh channel to the pod failed (exit 255)") == 2
    (world["home"] / ".ssh_fail").write_text("0\n", encoding="utf-8")
    # a remote command that fails on its own merits is never retried
    proc = _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_PODS='[{"id": "pod123"}, {"id": "stranger"}]',
                   CAGE_TEST_VOLUMES='[{"id": "vol123"}]')
    assert proc.returncode == 1 and "[retry]" not in proc.stdout


def test_stage_2_probes_that_the_results_root_accepts_the_pipe_character(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "ship")
    assert proc.returncode == 0, proc.stdout[-2000:]
    logs = list((world["exp_root"] / "S1" / DATE / "extras" / "steps").glob("ship_*results_root_accepts*.log"))
    assert len(logs) == 1 and "NAME_PROBE_OK" in logs[0].read_text(encoding="utf-8")
    assert not list((world["pod_repo"] / "results").glob(".cage_name_probe_*"))     # the probe cleans up


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


# ---------------------------------------------------------------------------
# 6. audit 2026-10-17, the shells batch (G5, G6, G8, G14 to G18, G30)
# ---------------------------------------------------------------------------


def test_registered_profile_s0_keys_and_unset_guards_are_refused_at_stage_0_g14_g15(world: Dict[str, Path], tmp_path: Path) -> None:
    bad = tmp_path / "S1_reg.env"
    bad.write_text(world["profile"].read_text(encoding="utf-8").replace("STEP_TIMEOUT_S=5400", "STEP_TIMEOUT_S=")
                   + "\nMAX_NULL_FRACTION=1.0\nFRONT_LOAD=1\nPREFETCH_MODELS=\"Qwen/Qwen3-14B\"\n", encoding="utf-8")
    world["profile"] = bad
    proc = _master(world)
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "registered profile (REHEARSAL_N empty) carries S0-only or unset keys (G14, G15" in proc.stdout
    for frag in ("STEP_TIMEOUT_S=empty", "MAX_NULL_FRACTION=1.0", "FRONT_LOAD=1", "PREFETCH_MODELS-lacks:intfloat/e5-large-v2"):
        assert frag in proc.stdout, frag
    assert not any("provision_pod" in c for c in _calls(world))
    # the same keys are legal on a rehearsal (S0): stage 0 passes and stops before the GO
    reh = tmp_path / "S1_reh.env"
    reh.write_text(bad.read_text(encoding="utf-8") + "\nREHEARSAL_N=50\n", encoding="utf-8")
    world["profile"] = reh
    assert _master(world).returncode == 10


def test_a_day_seatbelt_is_refused_at_stage_0_g30(world: Dict[str, Path], tmp_path: Path) -> None:
    prof = tmp_path / "S1_days.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8").replace("SEATBELT=12h", "SEATBELT=2d"), encoding="utf-8")
    world["profile"] = prof
    proc = _master(world)
    assert proc.returncode == 1 and "SEATBELT=2d is not <n>h or <n>m" in proc.stdout
    assert not any("provision_pod" in c for c in _calls(world))


def test_a_balance_below_the_pod_bound_fails_stage_0_g17(world: Dict[str, Path], tmp_path: Path) -> None:
    # the fake account holds 324.5 USD; 100 run hours at the 3.49 USD list price plus the 50 USD floor exceed it
    prof = tmp_path / "S1_poor.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8").replace("HOURS=1\n", "HOURS=100\n").replace("SEATBELT=12h", "SEATBELT=200h"), encoding="utf-8")
    world["profile"] = prof
    proc = _master(world)
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "balance 324.5 USD is below the" in proc.stdout and "the pod bound needs" in proc.stdout
    assert not any("provision_pod" in c and "--yes" in c for c in _calls(world))
    # the passing path records the figures
    world["profile"] = world["tmp"] / "S1.env"
    assert _master(world).returncode == 10
    st = _state(world)
    assert st["cost"]["balance_before_usd"] == "324.5" and st["cost"]["price_per_hour_list_usd"] == "3.49"
    assert float(st["cost"]["pod_bound_need_usd"]) < 324.5


def test_the_seatbelt_guard_refuses_a_pod_stage_when_the_deadline_is_short_g16(world: Dict[str, Path]) -> None:
    # the fake watchdog reports a deadline 30 min away: ship refuses before anything is copied to the pod
    _w(world["scripts"] / "runpod" / "pod_watchdog.sh",
       "#!/bin/bash\necho \"pod_watchdog $*\" >> \"$CAGE_TEST_LOG\"\n"
       "dl=$(python3 -c 'import datetime; print((datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30)).strftime(\"%Y-%m-%dT%H:%M:%SZ\"))')\n"
       "echo \"pod=$2 watchdog=ALIVE pid=1 deadline=$dl log=/dev/null\"\n")
    proc = _master(world, "--yes", "provision")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "the seatbelt fires in" in proc.stdout and "through pull need" in proc.stdout and "re-arm the watchdog" in proc.stdout
    st = _state(world)
    assert st["stages"]["provision"]["status"] == "passed" and st["stages"]["ship"]["status"] == "failed"
    assert not any(c.startswith("scp") and "cage_repo.tar.gz" in c for c in _calls(world))
    # a dead watchdog refuses too, naming the re-arm command
    _w(world["scripts"] / "runpod" / "pod_watchdog.sh", "#!/bin/bash\necho \"pod=$2 watchdog=DEAD pid=? deadline=? log=/dev/null\"\n")
    proc = _master(world, "--from", "ship", "--redo")
    assert proc.returncode == 1 and "the seatbelt is not ALIVE for pod123 before stage ship" in proc.stdout


def test_a_resume_reuses_the_job_that_finished_while_the_master_was_gone_g5(world: Dict[str, Path]) -> None:
    assert _master(world, "--yes", "provision", "--to", "ship").returncode == 0
    land = world["exp_root"] / "S1" / DATE
    state_path = land / "extras" / "state.json"
    st = json.loads(state_path.read_text(encoding="utf-8"))
    # the master died during setup; its job ran on and finished DONE(0) on the pod
    st["stages"]["setup"] = {"status": "running", "rc": None, "expected_rc": 0, "started_utc": "2026-10-06T10:00:00Z",
                             "ended_utc": None, "evidence": [], "notes": []}
    state_path.write_text(json.dumps(st), encoding="utf-8")
    remote = world["home"] / ".cage_jobs"; remote.mkdir(exist_ok=True)
    (remote / "setup.cmd").write_text("bash scripts/runpod/setup_runpod.sh\n", encoding="utf-8")
    log_lines = ["[cage]   all charter datasets staged: squad_v2 musique qasper"]
    log_lines += [f"[cage]   {m}: cached" for m in PREFETCH_FULL.split()]
    log_lines += ["[cage]   pynvml OK", "[cage]   cage_stats.api import OK", "[cage]  RunPod bootstrap complete."]
    (remote / "setup.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (remote / "setup.status").write_text("0\n", encoding="utf-8")
    (land / "extras" / "jobs" / "setup.json").write_text(json.dumps({
        "id": "setup", "mode": "pod-ssh", "host": "root@pod.test", "port": "2222", "handle": "pid:424242",
        "remote_log": "~/.cage_jobs/setup.log", "remote_status": "~/.cage_jobs/setup.status",
        "submitted_at": "2026-10-06T10:00:00Z", "deadline_s": 60, "billable": True}), encoding="utf-8")
    submits = lambda: sum(1 for c in _calls(world) if "base64 -d" in c and "setup.cmd" in c)
    before = submits()
    proc = _master(world, "--from", "setup", "--redo", "--to", "setup")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    assert "job setup finished DONE(0) during the interrupted attempt" in proc.stdout
    assert submits() == before                       # no second submit
    assert (land / "logs" / "setup" / "setup.log").read_text(encoding="utf-8").endswith("RunPod bootstrap complete.\n")
    st = _state(world)
    assert st["stages"]["setup"]["status"] == "passed" and any("G5" in n for n in st["stages"]["setup"]["notes"])
    # a deliberate --redo of a PASSED stage still submits a fresh job (the takeover rule is for the interrupted stage only)
    proc = _master(world, "--only", "setup", "--redo")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert submits() == before + 1
    # review batch 4 (M4): a handle whose pod is UNREACHABLE never gets a second submit
    (world["home"] / ".ssh_fail").write_text("9", encoding="utf-8")
    proc = _master(world, "--only", "setup", "--redo")
    assert proc.returncode == 1 and "the pod is UNREACHABLE: its state cannot be read, so nothing is submitted" in proc.stdout
    assert submits() == before + 1
    (world["home"] / ".ssh_fail").write_text("0", encoding="utf-8")


def test_a_failed_registered_run_copies_the_unsealed_tree_off_the_pod_first_g6(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", CAGE_TEST_RUN_CRASH="1")
    assert proc.returncode == 1
    assert "custody (G6): copying the pod's mirror of the UNSEALED tree" in proc.stdout
    calls = _calls(world)
    assert any(c.startswith("scp") and "run_unsealed" in c for c in calls), [c for c in calls if c.startswith("scp")][-3:]
    st = _state(world)
    assert st["stages"]["run"]["status"] == "failed" and any("G6 custody" in n for n in st["stages"]["run"]["notes"])


def test_the_scoring_batch_size_reaches_the_pass_only_when_the_profile_sets_it_g19(world: Dict[str, Path], tmp_path: Path) -> None:
    proc = _master(world, "--yes", "provision", "--to", "score")
    assert proc.returncode == 0, proc.stdout[-3000:]
    cmd = (world["home"] / ".cage_jobs" / "score.cmd").read_text(encoding="utf-8")
    assert "--batch-size" not in cmd            # empty = the sequential path
    prof = tmp_path / "S1_batched.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8") + "\nSCORING_BATCH_SIZE=32\n", encoding="utf-8")
    world["profile"] = prof
    proc = _master(world, "--only", "score", "--redo")
    assert proc.returncode == 0, proc.stdout[-3000:]
    cmd = (world["home"] / ".cage_jobs" / "score.cmd").read_text(encoding="utf-8")
    # A15 (2026-10-21): the run-level score cache sits between the freeze file and the batch size
    assert re.search(r"--freeze-file MyDocs/registration/freeze_resolutions\.json --score-cache-dir logs/score_cache/\S+ --batch-size 32; \}", cmd), cmd[-600:]


def test_a_recorded_conductor_pod_passes_the_clean_room_and_the_zero_proof_g8(world: Dict[str, Path]) -> None:
    # without the record a listed pod is a clean-room refusal ...
    proc = _master(world, CAGE_TEST_PODS='[{"id": "cond1"}]')
    assert proc.returncode == 1 and "clean room: pods exist before this experiment" in proc.stdout
    # ... with it, stage 0 and the stage 13 zero proof tolerate exactly that pod
    state_path = world["exp_root"] / "S1" / DATE / "extras" / "state.json"
    st = json.loads(state_path.read_text(encoding="utf-8"))
    st["conductor"] = {"id": "cond1"}
    state_path.write_text(json.dumps(st), encoding="utf-8")
    proc = _master(world, "--yes", "provision", "--yes", "teardown", CAGE_TEST_PODS='[{"id": "cond1"}]')
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    assert _state(world)["stages"]["teardown"]["status"] == "passed"


def test_the_plan_job_carries_the_profiles_last_block_owner_2026_10_21(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "plan", LAST_BLOCK="L5")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    plan_cmd = (world["home"] / ".cage_jobs" / "plan.cmd").read_text(encoding="utf-8")
    assert " --last-block L5 --out results/calibration/plan_" in plan_cmd


def test_the_budget_deadline_is_fixed_at_the_first_submit_review_2026_10_21_m1(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "run", RUN_BUDGET_HOURS="5")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    epoch = _state(world)["run"]["stop_at_epoch"]
    assert f"--stop-at-epoch {epoch}" in (world["home"] / ".cage_jobs" / "run.cmd").read_text(encoding="utf-8")
    # a resubmit (here, a redone run stage) keeps the first deadline, never re-arms the cap
    proc = _master(world, "--only", "run", "--redo", RUN_BUDGET_HOURS="5")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert f"--stop-at-epoch {epoch}" in (world["home"] / ".cage_jobs" / "run.cmd").read_text(encoding="utf-8")


def test_a_pinned_profile_without_a_served_model_revision_is_refused_review_2026_10_21_l2(world: Dict[str, Path], tmp_path: Path) -> None:
    pinned = tmp_path / "S1_pin.env"
    pinned.write_text(world["profile"].read_text(encoding="utf-8") + "\nPIN_MODEL_REVISION=1\n", encoding="utf-8")
    world["profile"] = pinned
    proc = _master(world)  # the world freeze file has no served_model slot
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "PIN_MODEL_REVISION=1-but-no-served_model-revision-for-" in proc.stdout
    assert not any("provision_pod" in c for c in _calls(world))


def test_validate_runs_the_helpers_beside_sglang_at_the_largest_helper_pool_review_2026_10_21_h1(world: Dict[str, Path], tmp_path: Path) -> None:
    two = tmp_path / "S1_two.env"
    two.write_text(world["profile"].read_text(encoding="utf-8").replace('ENGINES="vllm hf"', 'ENGINES="vllm sglang hf"'), encoding="utf-8")
    world["profile"] = two
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_HELPER_POOL="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    cmd = (world["home"] / ".cage_jobs" / "validate_sglang.cmd").read_text(encoding="utf-8")
    second = cmd.index("env CAGE_SGLANG_MAX_TOTAL_TOKENS=70437 VLLM_MAX_MODEL_LEN=8192 bash scripts/2_serving/manage_sglang_server.sh start")
    first_stop = cmd.index("manage_sglang_server.sh stop")
    assert first_stop < second  # the second start follows the first stop
    for probe in ("pool-check --engine sglang --planned-tokens 70437", "probe_reranker_device.py", "probe_compressor_device.py", "probe_encoder_device.py"):
        assert second < cmd.index(probe, second) < cmd.rindex("manage_sglang_server.sh stop"), probe
    assert "the GPU helpers fit beside SGLang at the largest helper-cell pool (70437 tokens)" in proc.stdout
    assert "RETRACTION_COUNTER_DECLARED" in (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "validate_sglang.log").read_text(encoding="utf-8")
    assert "env CAGE_SGLANG_MAX_TOTAL_TOKENS=70437" not in (world["home"] / ".cage_jobs" / "validate_vllm.cmd").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. ADR-0176 Group B (owner 2026-10-22): the volume is never deleted, the
#    teardown deletes the pod only, the S1 profile matches the approved plan
# ---------------------------------------------------------------------------


def test_teardown_deletes_the_pod_only_and_records_the_kept_volume(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--yes", "teardown", CAGE_TEST_VOLUMES='[{"id": "vol123"}]')
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    calls = _calls(world)
    assert not any("network-volume delete" in c for c in calls)
    assert any("teardown_pod pod123" in c for c in calls)
    assert "network volume(s) vol123 KEPT with the run tree (owner rule 2026-10-22)" in proc.stdout
    assert "volumes STILL LISTED" not in proc.stdout
    st = _state(world)
    assert st["stages"]["teardown"]["status"] == "passed" and st["pod"]["volume_kept"] == "vol123"
    assert st["cost"]["true_zero_utc"]  # the zero proof is zero pods


def test_a_refused_pod_keeps_the_volume_and_the_resume_reuses_it(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", CAGE_TEST_PROVISION_YES_RC="3")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "volume vol123 is KEPT there (never deleted by a script" in proc.stdout
    assert "PORTAL ACTION" in proc.stdout and "(KEPT: never deleted by a script, owner rule 2026-10-22)" in proc.stdout
    calls = _calls(world)
    assert not any("network-volume delete" in c for c in calls)
    assert sum(1 for c in calls if "network-volume create" in c) == 1
    st = _state(world)
    assert st["pod"]["volume_id"] == "vol123" and st["pod"]["dc"] == "US-NE-1" and not st["pod"].get("id")
    assert st["stages"]["provision"]["status"] == "failed"
    proc = _master(world, "--from", "provision", "--yes", "provision", "--to", "provision", CAGE_TEST_VOLUMES='[{"id": "vol123"}]')
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "volume vol123 in US-NE-1 is already recorded (kept after a refused pod): reused" in proc.stdout
    calls = _calls(world)
    assert sum(1 for c in calls if "network-volume create" in c) == 1          # no second volume
    pods = [c for c in calls if "provision_pod" in c and "--yes" in c]
    assert len(pods) == 2 and "--data-center-ids US-NE-1 --network-volume-id vol123" in pods[-1]
    st = _state(world)
    assert st["pod"]["id"] == "pod123" and st["stages"]["provision"]["status"] == "passed"


def test_a_recorded_volume_deleted_by_hand_is_not_reused(world: Dict[str, Path]) -> None:
    # review 2026-10-22 M1: the owner deleted the kept volume in the console; the
    # resume sees it unlisted, clears the record and creates a new one
    proc = _master(world, "--yes", "provision", CAGE_TEST_PROVISION_YES_RC="3")
    assert proc.returncode == 1 and _state(world)["pod"]["volume_id"] == "vol123"
    proc = _master(world, "--from", "provision", "--yes", "provision", "--to", "provision")  # the fake listing is empty
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "recorded volume vol123 is no longer listed (deleted by hand): the record is cleared" in proc.stdout
    assert sum(1 for c in _calls(world) if "network-volume create" in c) == 2
    assert not any("network-volume delete" in c for c in _calls(world))
    st = _state(world)
    assert st["pod"]["id"] == "pod123" and st["pod"]["volume_id"] == "vol123" and st["pod"]["dc"] == "US-NE-1"


def test_the_real_teardown_script_keeps_a_listed_volume_and_exits_0(world: Dict[str, Path], tmp_path: Path) -> None:
    # the shipped teardown_pod.sh against the fake runpodctl: --force skips the
    # pull gate loudly (no pod here); the pod delete runs, the volume listing is
    # read-only and a listed volume is a note, never a delete and never exit 1
    real = REPO_ROOT / "scripts" / "runpod" / "teardown_pod.sh"
    env = _env(world, CAGE_ASSUME_YES="1", CAGE_TEST_VOLUMES='[{"id": "vol123"}]')
    env.pop("CAGE_POD_SSH", None)
    proc = subprocess.run(["bash", str(real), "pod123", f"file://{tmp_path}/absent", str(tmp_path / "dest"), "--force"],
                          capture_output=True, text=True, env=env, timeout=120, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "network volume(s) listed: KEPT by design" in proc.stdout and "TEARDOWN_COMPLETE" in proc.stdout
    calls = _calls(world)
    assert "runpodctl pod delete pod123" in calls and "runpodctl network-volume list" in calls
    assert not any("network-volume delete" in c for c in calls)
    # a pod still listed after the delete is still a failure
    proc = subprocess.run(["bash", str(real), "pod123", f"file://{tmp_path}/absent", str(tmp_path / "dest2"), "--force"],
                          capture_output=True, text=True, env=_env(world, CAGE_ASSUME_YES="1", CAGE_TEST_PODS='[{"id": "pod123"}]'),
                          timeout=120, cwd=str(REPO_ROOT))
    assert proc.returncode == 1 and "still appears in the listing" in proc.stderr


def test_the_shipped_s1_profile_matches_adr0176() -> None:
    keys = ("LAST_BLOCK", "MIN_IN_REGIME_WINDOWS", "CHARTER_DATASETS", "RUN_BUDGET_HOURS", "SCORE_BOUND_MIN",
            "CALIBRATE_RUNGS_BOUND_S", "SEATBELT", "HOURS", "SESSION", "PULL_MARGIN_MIN", "QUERY_MANIFESTS",
            "INSTALL_LMCACHE", "SKIP_BLOCKED")
    fmt = "|".join("${%s:-}" % k for k in keys)
    proc = subprocess.run(["bash", "-c", f'set -a; source "{EXPDIR}/profiles/_common.env"; source "{EXPDIR}/profiles/S1.env"; printf "%s" "{fmt}"'],
                          capture_output=True, text=True, env=PROFILE_ENV)
    assert proc.returncode == 0, proc.stderr
    got = dict(zip(keys, proc.stdout.split("|")))
    assert got == {"LAST_BLOCK": "", "MIN_IN_REGIME_WINDOWS": "", "CHARTER_DATASETS": "squad_v2 musique qasper",
                   "RUN_BUDGET_HOURS": "115", "SCORE_BOUND_MIN": "1200", "CALIBRATE_RUNGS_BOUND_S": "18000",
                   "SEATBELT": "164h", "HOURS": "120", "SESSION": "s1", "PULL_MARGIN_MIN": "300",
                   # Group E (2026-10-23): the three registered question files, one per S1 dataset
                   "QUERY_MANIFESTS": "squad_v2=data/manifests/squad_v2_2000x3_seed42_wr50.json "
                                      "qasper=data/manifests/qasper_852x12_seed42_wr50.json "
                                      "musique=data/manifests/musique_2000x1_seed42_wr50.json",
                   # B8 left S1 whole on 2026-10-23: no LMCache install, no blocked cell to skip
                   "INSTALL_LMCACHE": "0", "SKIP_BLOCKED": "0"}
    for item in got["QUERY_MANIFESTS"].split():
        assert (REPO_ROOT / item.split("=", 1)[1]).is_file(), item
    body = (EXPDIR / "profiles" / "S1.env").read_text(encoding="utf-8")
    assert "hotpotqa" not in body.split("# ADR-0176: the three datasets")[1].splitlines()[0]
    assert "L5" not in [l for l in body.splitlines() if l.startswith("LAST_BLOCK=")][0]


# ---------------------------------------------------------------------------
# 8. ADR-0176 Group C (owner 2026-10-22, option 2): the ssh pull before the pod
#    delete also brings the A/A report, and the pull margin is a profile key
# ---------------------------------------------------------------------------


def test_the_pull_brings_the_aa_report_when_present_and_notes_its_absence(world: Dict[str, Path]) -> None:
    # absent: the fake score job has no run_calibration.py, so the A6 line prints and nothing is written
    proc = _master(world, "--yes", "provision", "--to", "pull")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "no A/A report beside the run root" in proc.stdout
    st = _state(world)
    assert st["pull"]["aa_report"] == "absent"
    rid = st["run_id"]
    land = world["exp_root"] / "S1" / DATE
    assert not (land / "extras" / f"aa_repeats_{rid}.json").exists()
    # present: the report beside the run root on the pod is pulled into the extras
    (world["pod_repo"] / "results" / "camp1" / "a" / f"aa_repeats_{rid}.json").write_text('{"aa": "ok"}', encoding="utf-8")
    proc = _master(world, "--only", "pull", "--redo")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert (land / "extras" / f"aa_repeats_{rid}.json").read_text(encoding="utf-8") == '{"aa": "ok"}'
    assert _state(world)["pull"]["aa_report"].endswith(f"aa_repeats_{rid}.json")
    assert any("aa_repeats" in c and c.startswith("scp") for c in _calls(world))


def test_the_pull_margin_is_read_from_the_profile_in_the_seatbelt_sum(world: Dict[str, Path], tmp_path: Path) -> None:
    # the world profile: 671 min with the default 90; a 300 min margin adds 210
    prof = tmp_path / "S1_pull.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8").replace("SEATBELT=12h", "SEATBELT=2h") + "\nPULL_MARGIN_MIN=300\n", encoding="utf-8")
    world["profile"] = prof
    proc = _master(world)
    assert proc.returncode == 1
    assert "SEATBELT=2h (120 min) is shorter than the 881 min the stages need" in proc.stdout
    assert "pull margin 300" in proc.stdout


def test_a_dead_channel_during_the_aa_probe_fails_the_pull_instead_of_recording_absent(world: Dict[str, Path]) -> None:
    # review 2026-10-22 M1: ssh rc 255 three times is not "no such file"
    proc = _master(world, "--yes", "provision", "--to", "collect")
    assert proc.returncode == 0, proc.stdout[-3000:]
    rid = _state(world)["run_id"]
    (world["pod_repo"] / "results" / "camp1" / "a" / f"aa_repeats_{rid}.json").write_text('{"aa": "ok"}', encoding="utf-8")
    (world["home"] / ".ssh_fail").write_text("3\n", encoding="utf-8")
    proc = _master(world, "--from", "pull", "--to", "pull", CAGE_SSH_RETRY_S="0")
    (world["home"] / ".ssh_fail").write_text("0\n", encoding="utf-8")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "did not reach the pod (ssh rc=255)" in proc.stdout
    st = _state(world)
    assert st["stages"]["pull"]["status"] == "failed" and "aa_report" not in st.get("pull", {})


# ---------------------------------------------------------------------------
# 9. ADR-0176 Group D (owner 2026-10-22, step 1): the batched scoring path is
#    used only after the stage 10 probe proves it on 50 real answers
# ---------------------------------------------------------------------------


def _batched_profile(world: Dict[str, Path], tmp_path: Path) -> None:
    prof = tmp_path / "S1_batched.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8") + "\nSCORING_BATCH_SIZE=32\n", encoding="utf-8")
    world["profile"] = prof


def test_the_scoring_probe_passes_and_the_pass_runs_batched(world: Dict[str, Path], tmp_path: Path) -> None:
    _batched_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "score")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    calls = _calls(world)
    probe = [c for c in calls if c.startswith("scoring_probe ")]
    assert len(probe) == 1 and "--batch-size 32 --rows 50 --device cuda --freeze-file MyDocs/registration/freeze_resolutions.json" in probe[0]
    cmd = (world["home"] / ".cage_jobs" / "score.cmd").read_text(encoding="utf-8")
    assert "--batch-size 32" in cmd
    names = [n for n in re.findall(r"\.cage_jobs/([A-Za-z0-9_.-]+)\.cmd", world["log"].read_text(encoding="utf-8"))]
    assert names.index("score_probe") < names.index("score")
    st = _state(world)
    assert st["scoring"]["batch_proof"].startswith("SCORING_BATCH_OK rows=50 batch_size=32")
    assert "scoring runs batched at 32 (the proof passed)" in proc.stdout


@pytest.mark.parametrize("verdict", ["FAILED", "CRASH"])
def test_a_failed_or_crashed_scoring_probe_runs_the_pass_row_by_row(world: Dict[str, Path], tmp_path: Path, verdict: str) -> None:
    _batched_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "score", CAGE_TEST_SCORING_PROBE=verdict)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    cmd = (world["home"] / ".cage_jobs" / "score.cmd").read_text(encoding="utf-8")
    assert "--batch-size" not in cmd
    assert "the pass runs row by row" in proc.stdout
    st = _state(world)
    assert st["stages"]["score"]["status"] == "passed"
    assert st["scoring"]["batch_proof"].startswith("SCORING_BATCH_FAILED")
    if verdict == "CRASH":
        assert "no verdict line" in st["scoring"]["batch_proof"]
    assert any("batched scoring NOT used" in n for n in st["stages"]["score"]["notes"])


def test_without_a_batch_size_no_probe_runs(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "score")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert not any(c.startswith("scoring_probe ") for c in _calls(world))
    assert not (world["home"] / ".cage_jobs" / "score_probe.cmd").exists()
    assert "scoring" not in _state(world)
