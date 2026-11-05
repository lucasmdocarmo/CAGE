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
# ADR-0178 block 2 (review M3): .ssh_fail_skip lets that many round trips
# through first, so an outage can start at a later probe of one stage.
if [ -s "$CAGE_TEST_HOME/.ssh_fail_skip" ]; then
  k=$(cat "$CAGE_TEST_HOME/.ssh_fail_skip")
  if [ "$k" -gt 0 ]; then
    echo $(( k - 1 )) > "$CAGE_TEST_HOME/.ssh_fail_skip"
    echo "ssh $host :: $*" >> "$CAGE_TEST_LOG"
    export HOME="$CAGE_TEST_HOME"
    cd "$HOME"
    exec bash -c "$*"
  fi
fi
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
    _w(b / "curl", "#!/bin/bash\ncase \"$*\" in *\"/metrics\"*) case \"${CAGE_TEST_RETRACTION:-lazy}\" in declared) r='sglang:num_retracted_reqs 0\\n# TYPE sglang:num_retracted_requests_total counter\\n' ;; none) r='' ;; prefix) r='sglang:num_retracted_reqs_created 1\\n' ;; *) r='sglang:num_retracted_reqs{model_name=\"m\"} 0.0\\n' ;; esac; printf \"sglang:token_usage 0.1\\n${r}sglang:num_running_reqs 0\\nsglang:num_queue_reqs 0\\nvllm:kv_cache_usage_perc 0.1\\n\" ;; *\"/server_info\"*) printf '{\"version\":\"0.5.10.post1\",\"enable_cache_report\":%s}\\n' \"${CAGE_TEST_CACHE_REPORT:-true}\" ;; esac\nexit 0\n")
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
            # S2 Block B-1: CAGE_TEST_TP_ENV=1 shards the single instance over four ranks
            tp4 = os.environ.get("CAGE_TEST_TP_ENV") == "1"
            # A11 item 4: the largest planned pool per engine, on request (its env carries the TP variable at tp 4)
            largest = {} if os.environ.get("CAGE_TEST_LARGEST_POOL") != "1" else {
                "vllm": {"planned_tokens": 298687, "planned_bytes": 48936960000, "tp": 4 if tp4 else 1,
                         "env": dict({"CAGE_KV_BUDGET_BYTES": "48936960000", "VLLM_MAX_MODEL_LEN": "32768"},
                                     **({"CAGE_VLLM_TENSOR_PARALLEL": "4"} if tp4 else {}))}}
            # review 2026-10-21 H1: the largest pool a GPU-helper cell plans, on request
            helpers = {} if os.environ.get("CAGE_TEST_HELPER_POOL") != "1" else {
                "sglang": {"planned_tokens": 70437, "planned_bytes": 11540480000, "tp": 4 if tp4 else 1,
                           "env": dict({"CAGE_SGLANG_MAX_TOTAL_TOKENS": "70437", "VLLM_MAX_MODEL_LEN": "8192"},
                                       **({"CAGE_SGLANG_TP": "4"} if tp4 else {}))}}
            # review 2026-10-31 (G8 finding 2): only an engine with executable single cells gets a TP env
            tp_engines = os.environ.get("CAGE_TEST_TP_ENV_ENGINES", "vllm sglang").split()
            tp_env = {e: v for e, v in {"vllm": {"CAGE_VLLM_TENSOR_PARALLEL": "4"}, "sglang": {"CAGE_SGLANG_TP": "4"}}.items()
                      if tp4 and e in tp_engines}
            gc = os.environ.get("CAGE_TEST_GPU_COUNT_NEEDED", "4" if tp4 else "1")  # "" = no count (an older preflight)
            return {"session": session, "anchor_seq_tokens": 4779, "cells": 3, "executable_cells": 3,
                    "needs_lmcache": os.environ.get("CAGE_TEST_NEEDS_LMCACHE") == "1", "largest_pools": largest,
                    "largest_helper_pools": helpers, "tp_env": tp_env, "gpu_count_needed": int(gc) if gc else None,
                    # S2 block C (A-7): the identity gate rides the report like NEEDS_LMCACHE
                    "identity_gated": os.environ.get("CAGE_TEST_IDENTITY_GATED") == "1",
                    # S2 block D (DEC-D2, DEC-D3): the pd cells flag and the control-rung PD stack env
                    "pd_cells": os.environ.get("CAGE_TEST_PD_CELLS") == "1",
                    "pd_env": None if os.environ.get("CAGE_TEST_PD_CELLS") != "1" or os.environ.get("CAGE_TEST_PD_ENV", "1") != "1" else {
                        "engine": "vllm", "budget_r": 1.0, "seq_tokens": 4779, "tp": 2, "planned_bytes": 2651340800,
                        "env": __import__("json").loads('__PD_ENV_JSON__')},
                    "problems": problems}
        def run_projection(steps):
            # ADR-0190 (review 2026-11-05 MEDIUM 2): the stage 6 first-segment check imports this;
            # CAGE_TEST_STEP_S seconds per step, so a test sizes the first segment against the stop.
            # ADR-0191: a pd step is unmeasured (charged nothing, named as the real function names it)
            import os
            pd = [f"{s['kind']} {s.get('row_key') or s.get('engine')}" + ("" if s["kind"] == "relaunch" else " on topology pd")
                  for s in steps if s.get("topology") == "pd" or (s.get("cellspec") or {}).get("topology") == "pd"]
            total = float(os.environ.get("CAGE_TEST_STEP_S", "400")) * (len(steps) - len(pd))
            return {"total_s": total, "total_h": round(total / 3600, 2), "unmeasured_steps": pd}
        if __name__ == "__main__":
            a = sys.argv[1:]
            if a and a[0] == "rungs":  # ADR-0154: stage 5 lists the budgeted rungs on the Mac
                print("1 0.5"); sys.exit(0)
            if a and a[0] == "classes":  # B4: the demand classes the class floors run on
                print("348,4779"); sys.exit(0)
            if a and a[0] == "regime-gate-projection":  # S2 block F row F-10: stage 0's $0 projection, under --plan too
                import os as _os
                def _opt(flag, default=""):
                    return a[a.index(flag) + 1] if flag in a else default
                if _os.environ.get("CAGE_TEST_PROJECTION_REFUSE"):
                    print("REFUSED: MIN_IN_REGIME_WINDOWS=6 exceeds the 2 registered windows of vllm|gold-fresh|none|none|single|qasper|1.0|0.95 "
                          "(BLOCKING; 30 such group(s)): the gate would fail the landing by construction", file=sys.stderr); sys.exit(2)
                print(f"regime-gate projection session={_opt('--session')} loads={_opt('--loads', '1.0+')!r} floor={_opt('--min')!r} "
                      f"roster={_opt('--confirmatory-arms') or '(empty: every group blocks)'} gated groups=0 blocking=0 descriptive=0 unresolved_load_cells=0")
                sys.exit(0)
            if a and a[0] == "ladder-cells":  # S2 block F row F-5: the (engine, class, rung) triples, on the Mac before the pod bills
                import os as _os
                from pathlib import Path as _Path
                eng = a[a.index("--engine") + 1]
                with open(_Path(_os.environ["CAGE_TEST_HOME"]) / ".cage_ladder_cells_argv", "a", encoding="utf-8") as fh:
                    fh.write(" ".join(a) + "\n")
                knob = int(_os.environ.get("CAGE_TEST_LADDER_CELLS_RC", "0"))
                if knob:
                    print(f"REFUSED: fake ladder-cells rc {knob} for engine {eng}", file=sys.stderr); sys.exit(knob)
                for r in ("1", "0.5"):
                    print(f"{eng} anchor gold-fresh B1 single 4779 {r}")
                    print(f"{eng} smallest retr-trunc B5 single 348 {r}")
                print(f"engine={eng} triples=4 classes=2"); sys.exit(0)
            print("REFUSED: fake Mac planner knows only 'rungs', 'classes', 'regime-gate-projection' and 'ladder-cells'", file=sys.stderr); sys.exit(2)
        '''.replace("__PD_ENV_JSON__", json.dumps(_PD_ENV, sort_keys=True)), exe=False)
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
    # S2 block F row F-7 (2026-11-03): the retrieval-stage sidebar, the same "0|3"
    # contract as the yield table; the fake writes the two files and exits 0.
    _w(fake_scripts / "4_analysis" / "retrieval_stage.py", r'''
        import os, sys
        from pathlib import Path
        args = sys.argv[1:]
        knob = int(os.environ.get("CAGE_TEST_RETRIEVAL_STAGE_RC", "0"))   # review L8: drive the "0|3" contract
        if knob:
            print(f"[retrieval_stage] {'REFUSED' if knob == 3 else 'ERROR'}: fake rc {knob}", file=sys.stderr); sys.exit(knob)
        out = Path(args[args.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
        (out / "retrieval_stage.md").write_text("# Retrieval stage\n", encoding="utf-8")
        (out / "retrieval_stage.json").write_text("{}", encoding="utf-8")
        print("[retrieval_stage] groups=1 pairs=0 skipped_cells=0", args); sys.exit(0)
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
        _w(pr / "scripts" / "2_serving" / f"manage_{eng}_server.sh", f"#!/bin/bash\nextra=\"\"; [ -z \"${{VLLM_KV_CACHE_DTYPE:-}}${{SGLANG_KV_CACHE_DTYPE:-}}\" ] || extra=\" kv=${{VLLM_KV_CACHE_DTYPE:-}}${{SGLANG_KV_CACHE_DTYPE:-}}\"; [ -z \"${{LMCACHE_MAX_LOCAL_CPU_SIZE:-}}\" ] || extra=\"$extra store=${{LMCACHE_MAX_LOCAL_CPU_SIZE}}\"; [ -z \"${{CAGE_VLLM_TENSOR_PARALLEL:-}}${{CAGE_SGLANG_TP:-}}\" ] || extra=\"$extra tp=${{CAGE_VLLM_TENSOR_PARALLEL:-}}${{CAGE_SGLANG_TP:-}}\"; echo \"launcher {eng} $*$extra\" >> \"$CAGE_TEST_LOG\"; case \"$1\" in start|restart) mkdir -p logs/{eng}; echo '{{}}' > logs/{eng}/CURRENT.kvpool.json; [ -n \"${{LMCACHE_MAX_LOCAL_CPU_SIZE:-}}\" ] && echo \"'max_local_cpu_size': ${{LMCACHE_MAX_LOCAL_CPU_SIZE}}.0,\" >> logs/{eng}/vllm_fake.log ;; esac; echo \"{eng} $1 ok\"; exit 0\n")
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
    # S2 block C (C-1): the identity gate script; prompts and run record their argv and
    # write --out, verdict prints the one marker from CAGE_TEST_IDENTITY_VERDICT (PASS default);
    # a run whose --phase equals CAGE_TEST_IDENTITY_RUN_FAIL writes an error phase file and exits 1
    _w(pr / "scripts" / "checks" / "identity_gate.py", r'''
        import json, os, sys
        from pathlib import Path
        a = sys.argv[1:]
        with open(os.environ["CAGE_TEST_LOG"], "a", encoding="utf-8") as fh:
            fh.write("identity_gate " + " ".join(a) + "\n")
        out = Path(a[a.index("--out") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
        if a[0] == "run" and a[a.index("--phase") + 1] == os.environ.get("CAGE_TEST_IDENTITY_RUN_FAIL", ""):
            out.write_text(json.dumps({"phase": a[a.index("--phase") + 1], "error": "HTTP 500 at prompt 3"}), encoding="utf-8")
            print("IDENTITY_GATE_RUN_ERROR phase=" + a[a.index("--phase") + 1] + " reason=HTTP 500 at prompt 3", file=sys.stderr); sys.exit(1)
        if a[0] in ("prompts", "run"):
            out.write_text(json.dumps({"phase": a[0], "argv": a}), encoding="utf-8"); sys.exit(0)
        if os.environ.get("CAGE_TEST_IDENTITY_NO_WRITE") == "1":
            print("IDENTITY_GATE CRASH before the verdict"); sys.exit(1)
        verdict = os.environ.get("CAGE_TEST_IDENTITY_VERDICT", "PASS")
        out.write_text(json.dumps({"schema": "cage.identity_gate/1", "verdict": verdict, "session": "a", "argv": a}), encoding="utf-8")
        print(f"IDENTITY_GATE {verdict}")
        sys.exit(1 if verdict == "ERROR" else 0)
        ''', exe=False)
    # S2 block D (DEC-D3): the PD stack launcher and the smoke; the launcher logs the pins,
    # the degree and the role budgets its start carried (CAGE_TEST_PD_START_RC=1 fails the
    # start), the smoke writes --out with the verdict from CAGE_TEST_PD_SMOKE_VERDICT (PASS
    # default) and prints the one marker; its exit is 1 on ERROR only
    _w(pr / "scripts" / "2_serving" / "manage_vllm_pd.sh", r'''
        #!/bin/bash
        echo "launcher pd $* pins=${CAGE_PD_PREFILL_GPUS:-}/${CAGE_PD_DECODE_GPUS:-} tp=${CAGE_VLLM_TENSOR_PARALLEL:-} budgets=${CAGE_KV_BUDGET_BYTES_PREFILL:-}/${CAGE_KV_BUDGET_BYTES_DECODE:-}" >> "$CAGE_TEST_LOG"
        if [ "$1" = "start" ] && [ "${CAGE_TEST_PD_START_RC:-0}" = "1" ]; then echo "pd start failed"; exit 1; fi
        echo "pd $1 ok"; exit 0
        ''')
    _w(pr / "scripts" / "checks" / "pd_smoke.py", r'''
        import json, os, sys
        from pathlib import Path
        a = sys.argv[1:]
        with open(os.environ["CAGE_TEST_LOG"], "a", encoding="utf-8") as fh:
            fh.write("pd_smoke " + " ".join(a) + "\n")
        out = Path(a[a.index("--out") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
        if os.environ.get("CAGE_TEST_PD_SMOKE_NO_WRITE") == "1":
            print("PD_SMOKE CRASH before the verdict"); sys.exit(1)
        verdict = os.environ.get("CAGE_TEST_PD_SMOKE_VERDICT", "PASS")
        out.write_text(json.dumps({"schema": "cage.pd_smoke/1", "verdict": verdict, "session": a[a.index("--session") + 1],
                                   "model": a[a.index("--model") + 1], "argv": a}), encoding="utf-8")
        print(f"PD_SMOKE {verdict} requests=8")
        sys.exit(1 if verdict == "ERROR" else 0)
        ''', exe=False)
    # S1 audit 2026-10-23 (MEDIUM 4): the offline dataset probe, before any engine starts
    _w(pr / "scripts" / "checks" / "probe_datasets_offline.py", r'''
        import json, os, sys
        from pathlib import Path
        a = sys.argv[1:]
        out = Path(a[a.index("--out") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
        ok = os.environ.get("CAGE_TEST_DATASETS_OFFLINE_RC", "0") != "1"
        with open(os.environ["CAGE_TEST_LOG"], "a", encoding="utf-8") as fh:
            fh.write(f"datasets_offline {a[a.index('--datasets') + 1]} require_offline={'--require-offline' in a} HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '')}\n")
        out.write_text(json.dumps({"ok": ok}), encoding="utf-8")
        print("DATASETS_OFFLINE_OK hf_hub_offline=1 loaded=squad_v2=3,musique=3,qasper=3" if ok else "DATASETS_OFFLINE_FAILED reason=dataset musique did not load: FileNotFoundError")
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
                              {"kind": "cell", "cellspec": {"engine": "vllm", "topology": "single"}, "row_key": "k1", "dataset": "squad_v2", "argv": ["run_experiment.py", "--vllm-telemetry"]},
                              {"kind": "cell", "cellspec": {"engine": "hf", "topology": "single"}, "row_key": "k2", "dataset": "ruler", "argv": ["run_cag_reference.py"]}],
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
            if os.environ.get("CAGE_TEST_NO_RELAUNCH"):  # ADR-0190 review MEDIUM 2: a plan the budget stop never checks
                plan["steps"] = plan["steps"][1:]
            # the planner's pd shape (run_campaign.py _cell_step, PD_TELEMETRY_ENDPOINTS_ENV; block D, ADR-0182): the
            # role pins (CAGE_PD_*) ride the pd relaunch only; a pd cell or dry window carries the telemetry pair
            pd_tel = {"CAGE_TELEMETRY_ENDPOINTS": "prefill=http://localhost:8100,decode=http://localhost:8200"}
            if os.environ.get("CAGE_TEST_PD_TAIL"):  # ADR-0191: a measured segment, then a pd segment the table cannot project
                plan["steps"] += [
                    {"kind": "relaunch", "engine": "vllm", "topology": "single", "argv": ["manage_vllm_server.sh", "start"]},
                    {"kind": "cell", "cellspec": {"engine": "vllm", "topology": "single"}, "row_key": "k3", "dataset": "squad_v2", "argv": ["run_experiment.py", "--vllm-telemetry"]},
                    {"kind": "relaunch", "engine": "vllm", "topology": "pd", "argv": ["manage_vllm_pd.sh", "start"], "env": {"CAGE_PD_PREFILL_GPUS": "0,1"}},
                    {"kind": "cell", "cellspec": {"engine": "vllm", "topology": "pd"}, "row_key": "k4", "dataset": "squad_v2", "argv": ["run_experiment.py", "--vllm-telemetry"],
                     "env": {} if os.environ.get("CAGE_TEST_PD_NO_TELEMETRY") else dict(pd_tel)}]
                plan["run_projection"]["unmeasured_steps"] = ["relaunch vllm", "cell k4 on topology pd"]
            if os.environ.get("CAGE_TEST_PD_FIRST"):  # ADR-0191 review LOW 4: a pd cell inside the first segment (step 0 under NO_RELAUNCH)
                plan["steps"].insert(1 if plan["steps"][0]["kind"] == "relaunch" else 0,
                                     {"kind": "cell", "cellspec": {"engine": "vllm", "topology": "pd"}, "row_key": "k0", "dataset": "squad_v2", "argv": ["run_experiment.py", "--vllm-telemetry"], "env": dict(pd_tel)})
                plan["run_projection"]["unmeasured_steps"] = ["cell k0 on topology pd"]
            if os.environ.get("CAGE_TEST_PD_NULL_LIST"):  # six-fix review LOW 1: a hand-edited projection with no list
                plan["run_projection"]["unmeasured_steps"] = None
            if os.environ.get("CAGE_TEST_PD_WRONG_NAMES"):  # six-fix review LOW 2: the pod's code names other steps
                plan["run_projection"]["unmeasured_steps"] = ["x", "y"]
            if os.environ.get("CAGE_TEST_NO_PROJECTION"):  # six-fix review LOW 1: a plan from a planner without B-5
                del plan["run_projection"]
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
            if os.environ.get("CAGE_TEST_RUN_BUDGET_STOP"):
                print("[run_campaign] BUDGET STOP (ADR-0175): the segment at step 7 projects 9000 s and would end 120 s past the deadline; the remaining steps are not run")
                if os.environ.get("CAGE_TEST_RUN_BUDGET_STOP_DRY"):
                    print("[run_campaign] DRY WINDOW FAILED: engine=vllm class=4779 tokens r=0.5 rate=2.85 rps expected IN_REGIME, label='UNPRESSURED'")
                    print("[run_campaign] --seal SKIPPED: run has failures and/or blocked-skipped cells and --seal-partial was not given (a seal marks the tree done)")
                    print("[run_campaign] exit 4 (budget stop) also carries failed dry windows (review 2026-10-20 L7)")
                elif "--seal" in a:
                    (root / "ledger.json").write_text("{}", encoding="utf-8")
                print("[run_campaign] totals: {'ok': 60, 'skipped-budget-stop': 40}")
                sys.exit(4)
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
        POD_HF_CACHE={tmp_path}/podhome/.cache/huggingface
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

#: S2 block D (DEC-D3): the control-rung PD stack env the fake Mac planner reports
#: under CAGE_TEST_PD_CELLS=1 and stage 0 records for the stage 4 smoke (the pins,
#: the degree, the role budgets, the ports, the telemetry pair; values carry commas)
_PD_ENV = {
    "CAGE_KV_BUDGET_BYTES_PREFILL": "1325670400", "CAGE_KV_BUDGET_BYTES_DECODE": "1325670400",
    "CAGE_PD_PREFILL_GPUS": "0,1", "CAGE_PD_DECODE_GPUS": "2,3", "CAGE_VLLM_TENSOR_PARALLEL": "2",
    "VLLM_MAX_MODEL_LEN": "32768", "CAGE_PD_PROXY_PORT": "8000", "CAGE_PD_PREFILL_PORT": "8100",
    "CAGE_PD_DECODE_PORT": "8200", "CAGE_TELEMETRY_ENDPOINTS": "prefill=http://localhost:8100,decode=http://localhost:8200",
}
_PD_ENV_SORTED = " ".join(f"{k}={v}" for k, v in sorted(_PD_ENV.items()))


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
    assert names == ["setup", "load_gate", "datasets_offline", "validate_vllm", "start_check", "calibrate_vllm", "calibrate_rungs_vllm", "plan", "run", "score", "collect"]
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
    # readiness 0 (5 s) + setup 1 + validate (20: load gate + offline dataset probe, + 25 + 40 start check) + calibrate (60 + 150, ADR-0154
    # rung calibration 9,000 s) + plan 15 + run 60 + seal 10 + score 180 + collect 30 + pull margin 90 (G16)
    assert "SEATBELT=2h (120 min) is shorter than the 681 min the stages need" in proc.stdout
    assert not any("provision_pod" in c for c in _calls(world))


def _run_campaign_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cage_run_campaign_for_profile_tests", REPO_ROOT / "scripts" / "3_run" / "run_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_shipped_profiles_cover_their_stage_bounds_and_name_registered_models() -> None:
    import importlib
    cb = importlib.import_module("src.orchestration.cache_budget")
    rc = _run_campaign_module()
    for p in sorted((EXPDIR / "profiles").glob("S*.env")):
        if p.read_text(encoding="utf-8").startswith("# SUPERSEDED"):
            continue  # a kept record (S2.env since block E-1), not a landing profile; the master's preflight refuses it on the anchor shape
        proc = subprocess.run(["bash", "-c", f'set -a; source "{EXPDIR}/profiles/_common.env"; source "{p}"; '
                               'printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s" "$MODEL_SLUG" "$SEATBELT" "$HOURS" "$SETUP_BOUND_MIN" "$ENGINES" "$CALIBRATE" "${SCORE_BOUND_MIN:-180}" "$CALIBRATE_RUNGS_BOUND_S" "$FLOOR_AVG_SEQ_TOKENS" "${SUITE_ON_POD:-0}" "${START_CHECK:-1}" "${PULL_MARGIN_MIN:-90}" "$SESSION" "$IDENTITY_GATE_BOUND_S"'],
                              capture_output=True, text=True, env=PROFILE_ENV)
        slug, seatbelt, hours, setup_min, engines, calibrate, score_min, rungs_bound_s, floor_tokens, suite_on_pod, start_check, pull_min, session, ig_bound_s = proc.stdout.split("|")
        assert slug in cb.MODEL_KV, f"{p.name}: MODEL_SLUG={slug} is not in cache_budget.MODEL_KV {sorted(cb.MODEL_KV)}"
        # ADR-0155, S2 block A-5 (2026-11-01): every shipped profile sizes the floor table on ITS
        # session's anchor shape (s2a keeps the S0 shape, cd-act1 registers 2694); a placeholder
        # profile for a session the driver does not register yet (S3.env) keeps the common anchor
        try:
            grid = rc.get_session_grid(session)
            anchor, gated = str(grid.demand_seq_tokens[rc.DEMAND_ANCHOR_ARM]), bool(grid.identity_gated)
        except rc.PlanError:
            anchor, gated = "4779", False
        assert floor_tokens == anchor, f"{p.name}: FLOOR_AVG_SEQ_TOKENS={floor_tokens} is not session {session}'s anchor shape {anchor}"
        n_srv = len([e for e in engines.split() if e != "hf"]); n_cal = len(calibrate.split())
        # S0F-88 review L4: plus the stage 4 load gate and the offline dataset probe (20 min) and the pod suite (30 min when SUITE_ON_POD=1);
        # G16 (2026-10-17): plus the readiness wait (900 s), the start check (2,400 s), the seal (600 s) and a
        # the profile's pull margin (ADR-0176 Group C; the common default 90 = S0 attempt 5's 68 min)
        # S2 block C (2026-11-01): the identity gate (IDENTITY_GATE_BOUND_S) bills at stage 4 on a gated session
        need = (15 + int(setup_min) + 20 + (30 if suite_on_pod == "1" else 0) + n_srv * 25 + (40 if start_check == "1" else 0)
                + (int(ig_bound_s) // 60 if gated else 0)
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
    # S0F-88 P-4: the cost estimate covers every stage bound (681 min with the G16 terms), not HOURS alone
    provision = [c for c in _calls(world) if "provision_pod" in c]
    assert provision and all("--hours 11.3 " in c for c in provision)


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
    # S2 Block B-2: the degree the start used is recorded with the check (1 on a single GPU)
    assert "--planned-bytes 48936960000 --tp 1 --out logs/vllm/largest_pool_check.json" in cmd
    assert _state(world)["registration"]["largest_pool"]["vllm"]["tp"] == "1"
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
    assert "no budgeted single-instance pool in the plan; the largest-pool start is skipped (A11 item 4)" in proc.stdout


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
    # A15 (2026-10-21): the run-level score cache sits between the freeze file and the batch size;
    # S1 audit 2026-10-23 (H1): under results/, which no stage copies file by file (logs/ is)
    assert re.search(r"--freeze-file MyDocs/registration/freeze_resolutions\.json --score-cache-dir results/score_cache/\S+ --batch-size 32; \}", cmd), cmd[-600:]
    assert "logs/score_cache" not in cmd


# ---------------------------------------------------------------------------
# S1 audit 2026-10-23: the fixes the three readers found before the S1 relaunch
# ---------------------------------------------------------------------------

def test_calibrate_starts_each_engine_at_the_registered_largest_pool_s1_audit_c1(world: Dict[str, Path]) -> None:
    """CRITICAL 1: the class floor prompts reach the anchor class (4,779 tokens) and
    the launcher default context is 4,096; the start carries stage 0's record."""
    proc = _master(world, "--yes", "provision", "--to", "calibrate", CAGE_TEST_LARGEST_POOL="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    cmd = (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").read_text(encoding="utf-8")
    assert "env CAGE_KV_BUDGET_BYTES=48936960000 VLLM_MAX_MODEL_LEN=32768 bash scripts/2_serving/manage_vllm_server.sh start" in cmd
    assert cmd.index("VLLM_MAX_MODEL_LEN=32768") < cmd.index("calibrate_cell.py")


def test_class_floors_without_a_recorded_largest_pool_refuse_stage_5_s1_audit_c1(world: Dict[str, Path], tmp_path: Path) -> None:
    prof = tmp_path / "S1_floors.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8") + "\nCLASS_FLOORS=1\n", encoding="utf-8")
    world["profile"] = prof
    proc = _master(world, "--yes", "provision", "--to", "calibrate")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "CLASS_FLOORS=1 but stage 0 recorded no largest pool for this engine" in proc.stdout
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists()
    # with the record, the same profile starts at the pool and passes the class floors on
    proc = _master(world, "--from", "calibrate", "--redo", "--to", "calibrate", CAGE_TEST_LARGEST_POOL="1")
    assert proc.returncode == 1  # stage 0 was not redone, so the record is still absent: refused again, nothing submitted
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists()


def test_a_budget_stop_passes_as_partial_but_not_over_failed_dry_windows_s1_audit_m3(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "run", CAGE_TEST_RUN_BUDGET_STOP="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "exit 4 = BUDGET STOP (ADR-0175)" in proc.stdout
    st = _state(world)
    assert st["stages"]["run"]["status"] == "passed" and st["run"]["partial"].startswith("exit 4")
    # the same exit over failed dry windows is the exit 3 outcome on a registered profile
    proc = _master(world, "--from", "run", "--redo", "--to", "run", CAGE_TEST_RUN_BUDGET_STOP="1", CAGE_TEST_RUN_BUDGET_STOP_DRY="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] the budget stop also carries failed dry windows (ADR-0153)" in proc.stdout
    assert "custody (G6): copying the pod's mirror of the UNSEALED tree" in proc.stdout
    st = _state(world)
    assert st["stages"]["run"]["status"] == "failed"
    assert any("BUDGET STOP with failed dry windows" in n for n in st["stages"]["run"]["notes"])


def test_a_takeover_resume_reuses_a_run_that_ended_with_exit_4_s1_audit_h2(world: Dict[str, Path]) -> None:
    """HIGH 2: a run that ended (here a budget stop) while the master was gone is
    never resubmitted: a resubmit truncates its log and the reseal is refused."""
    proc = _master(world, "--yes", "provision", "--to", "run", CAGE_TEST_RUN_BUDGET_STOP="1")
    assert proc.returncode == 0, proc.stdout[-3000:]
    land = world["exp_root"] / "S1" / DATE
    state_path = land / "extras" / "state.json"
    st = json.loads(state_path.read_text(encoding="utf-8"))
    st["stages"]["run"]["status"] = "running"; st["stages"]["run"]["rc"] = None
    st["run"].pop("partial", None)
    state_path.write_text(json.dumps(st), encoding="utf-8")
    submits = lambda: sum(1 for c in _calls(world) if "base64 -d" in c and "run.cmd" in c)
    before = submits()
    proc = _master(world, "--from", "run", "--redo", "--to", "run", CAGE_TEST_RUN_BUDGET_STOP="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "job run finished FAILED(4) during the interrupted attempt: its log is fetched, nothing is resubmitted" in proc.stdout
    assert submits() == before
    st = _state(world)
    assert st["stages"]["run"]["status"] == "passed" and st["run"]["partial"].startswith("exit 4")
    # review 2026-10-23 M5: a handle submitted BEFORE the interrupted attempt started is
    # an older attempt's job, never reused: the stage submits afresh
    st = json.loads(state_path.read_text(encoding="utf-8"))
    st["stages"]["run"]["status"] = "running"; st["stages"]["run"]["rc"] = None
    st["stages"]["run"]["started_utc"] = "2099-01-01T00:00:00Z"
    st["run"].pop("partial", None)
    state_path.write_text(json.dumps(st), encoding="utf-8")
    proc = _master(world, "--from", "run", "--redo", "--to", "run", CAGE_TEST_RUN_BUDGET_STOP="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "predates the interrupted attempt (2099-01-01T00:00:00Z): a fresh job is submitted" in proc.stdout
    assert submits() == before + 1


def test_the_run_job_declares_the_dataset_roster_for_the_sealed_manifest_s1_audit_m2(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "run")
    assert proc.returncode == 0, proc.stdout[-3000:]
    cmd = (world["home"] / ".cage_jobs" / "run.cmd").read_text(encoding="utf-8")
    # review 2026-10-23 M3: the roster is the plan's own cell datasets (the fake plan
    # carries a RULER cell the charter roster would refuse), sorted
    assert "CAGE_CAMPAIGN_DATASETS='ruler squad_v2'" in cmd
    assert cmd.index("CAGE_CAMPAIGN_DATASETS=") < cmd.index("run_campaign.py run")

def test_stage_4_loads_every_charter_dataset_offline_before_any_engine_starts_s1_audit_m4(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "validate")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    calls = _calls(world)
    probe = [c for c in calls if c.startswith("datasets_offline ")]
    assert probe == ["datasets_offline squad_v2,musique,qasper require_offline=False HF_HUB_OFFLINE="]  # the world pins no revision
    # the probe runs after the load gate and before the first engine start
    order = [c for c in calls if c.startswith("datasets_offline ") or c.startswith("launcher ")]
    assert order and order[0] == probe[0]  # before the first launcher start of stage 4
    assert "DATASETS_OFFLINE_OK hf_hub_offline=1 loaded=squad_v2=3,musique=3,qasper=3" in proc.stdout
    # a dataset that does not load fails stage 4 before any engine bills
    proc = _master(world, "--from", "validate", "--redo", "--to", "validate", CAGE_TEST_DATASETS_OFFLINE_RC="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    # the probe exits 1, so the job wait fails first (the marker gate is the second layer)
    assert "[FAIL] job datasets_offline: wait: rc=1" in proc.stdout
    assert "DATASETS_OFFLINE_FAILED reason=dataset musique did not load" in proc.stdout
    assert _state(world)["stages"]["validate"]["status"] == "failed"
    assert not [c for c in _calls(world) if c.startswith("launcher ")][len([c for c in calls if c.startswith("launcher ")]):]  # no engine started on the failed attempt


def test_a_pinned_profile_requires_the_offline_load_s1_audit_m4(world: Dict[str, Path], tmp_path: Path) -> None:
    pinned = tmp_path / "S1_pin_off.env"
    pinned.write_text(world["profile"].read_text(encoding="utf-8") + "\nPIN_MODEL_REVISION=1\n", encoding="utf-8")
    world["profile"] = pinned
    freeze_path = world["tmp"] / "freeze_resolutions.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze.setdefault("INSTRUMENT_REVISIONS", {})["served_model"] = {"model": "Qwen/Qwen3-14B", "revision": "abc123"}
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    proc = _master(world, "--yes", "provision", "--to", "validate")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    probe = [c for c in _calls(world) if c.startswith("datasets_offline ")]
    assert probe == ["datasets_offline squad_v2,musique,qasper require_offline=True HF_HUB_OFFLINE=1"]

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
    # S1 2026-10-23: the fake serves the live shape, the gauge with no counter yet
    assert "RETRACTION_COUNTER_LAZY" in (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "validate_sglang.log").read_text(encoding="utf-8")
    assert "sglang: no retraction yet, so the retraction counter is not created" in proc.stdout
    assert "env CAGE_SGLANG_MAX_TOTAL_TOKENS=70437" not in (world["home"] / ".cage_jobs" / "validate_vllm.cmd").read_text(encoding="utf-8")


def _two_engine_profile(world: Dict[str, Path], tmp_path: Path) -> None:
    two = tmp_path / "S1_two_retraction.env"
    two.write_text(world["profile"].read_text(encoding="utf-8").replace('ENGINES="vllm hf"', 'ENGINES="vllm sglang hf"'), encoding="utf-8")
    world["profile"] = two


def test_a_declared_retraction_counter_passes_without_the_lazy_note_s1_2026_10_23(world: Dict[str, Path], tmp_path: Path) -> None:
    _two_engine_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_RETRACTION="declared")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    log = (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "validate_sglang.log").read_text(encoding="utf-8")
    assert "RETRACTION_COUNTER_DECLARED" in log and "RETRACTION_COUNTER_LAZY" not in log
    assert "no retraction yet" not in proc.stdout


def test_a_names_check_pass_without_counter_or_gauge_sample_fails_on_the_retraction_gate_review_m1(world: Dict[str, Path], tmp_path: Path) -> None:
    """Review 2026-10-23 M1: a line that starts with the gauge name passes the names
    check but is no gauge sample, so only the retraction check can stop it. The
    probe exits 1 on RETRACTION_COUNTER_MISSING, so the job fails before the
    master's marker gates run (the same two layers as the other stage 4 checks)."""
    _two_engine_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_RETRACTION="prefix")
    assert proc.returncode != 0, proc.stdout[-3000:]
    assert "METRIC_OK sglang:num_retracted_reqs" in proc.stdout  # the names check passed
    assert "RETRACTION_COUNTER_MISSING" in proc.stdout
    assert "RETRACTION_COUNTER_LAZY" not in proc.stdout and "RETRACTION_COUNTER_DECLARED" not in proc.stdout
    assert "lacks a name the regime bridge" not in proc.stdout
    assert _state(world)["stages"]["validate"]["status"] == "failed"


def test_neither_the_retraction_counter_nor_its_gauge_fails_validate_s1_2026_10_23(world: Dict[str, Path], tmp_path: Path) -> None:
    _two_engine_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_RETRACTION="none")
    assert proc.returncode != 0, proc.stdout[-3000:]
    assert "RETRACTION_COUNTER_MISSING" in proc.stdout + "".join(
        p.read_text(encoding="utf-8") for p in (world["exp_root"] / "S1" / DATE / "logs" / "setup").rglob("validate_sglang*.log"))
    assert _state(world)["stages"]["validate"]["status"] == "failed"


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
            "INSTALL_LMCACHE", "SKIP_BLOCKED", "CLASS_CEILINGS", "CLASS_CEILINGS_BOUND_S")
    fmt = "|".join("${%s:-}" % k for k in keys)
    proc = subprocess.run(["bash", "-c", f'set -a; source "{EXPDIR}/profiles/_common.env"; source "{EXPDIR}/profiles/S1.env"; printf "%s" "{fmt}"'],
                          capture_output=True, text=True, env=PROFILE_ENV)
    assert proc.returncode == 0, proc.stderr
    got = dict(zip(keys, proc.stdout.split("|")))
    assert got == {"LAST_BLOCK": "", "MIN_IN_REGIME_WINDOWS": "", "CHARTER_DATASETS": "squad_v2 musique qasper",
                   "RUN_BUDGET_HOURS": "115", "SCORE_BOUND_MIN": "1200", "CALIBRATE_RUNGS_BOUND_S": "18000",
                   # ADR-0177 (2026-10-25, review H1): the plan stage's ceiling ladders
                   # (36,000 s per engine), re-label and drift window add 1,260 min
                   "SEATBELT": "182h", "CLASS_CEILINGS": "1", "CLASS_CEILINGS_BOUND_S": "36000",
                   "HOURS": "120", "SESSION": "s1", "PULL_MARGIN_MIN": "300",
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
    # the world profile: 681 min with the default 90; a 300 min margin adds 210
    prof = tmp_path / "S1_pull.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8").replace("SEATBELT=12h", "SEATBELT=2h") + "\nPULL_MARGIN_MIN=300\n", encoding="utf-8")
    world["profile"] = prof
    proc = _master(world)
    assert proc.returncode == 1
    assert "SEATBELT=2h (120 min) is shorter than the 891 min the stages need" in proc.stdout
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


def test_the_pull_brings_the_driver_shape_calibration_report_beside_the_aa_report(world: Dict[str, Path]) -> None:
    # ADR-0178 block 2 (D4): the A6 step writes calibration_report_<run>.json
    # beside aa_repeats_<run>.json; the pull brings both, and notes the sibling
    # absent on a pod build that predates block 2
    proc = _master(world, "--yes", "provision", "--to", "pull")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    st = _state(world)
    rid = st["run_id"]
    land = world["exp_root"] / "S1" / DATE
    pod_dir = world["pod_repo"] / "results" / "camp1" / "a"
    # the A/A file alone: the sibling is noted absent, the stage passes
    (pod_dir / f"aa_repeats_{rid}.json").write_text('{"aa": "ok"}', encoding="utf-8")
    proc = _master(world, "--only", "pull", "--redo")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "no driver-shape calibration report beside the run root" in proc.stdout
    assert _state(world)["pull"]["calibration_report"] == "absent"
    # both files: both pulled, both recorded
    (pod_dir / f"calibration_report_{rid}.json").write_text('{"seed": 1}', encoding="utf-8")
    proc = _master(world, "--only", "pull", "--redo")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert (land / "extras" / f"calibration_report_{rid}.json").read_text(encoding="utf-8") == '{"seed": 1}'
    assert _state(world)["pull"]["calibration_report"].endswith(f"calibration_report_{rid}.json")
    assert any("calibration_report" in c and c.startswith("scp") for c in _calls(world))
    # the plan prints the sibling pull beside the A/A pull
    proc = _master(world, "--plan", "--only", "pull")
    assert "pull calibration report" in proc.stdout and "pull A/A report (A6)" in proc.stdout


def test_a_dead_channel_during_the_calibration_report_probe_fails_the_pull(world: Dict[str, Path]) -> None:
    # review M3: the second probe fails the stage on a dead channel exactly
    # like the first; the outage starts at that probe (.ssh_fail_skip lets the
    # earlier round trips of the stage through)
    proc = _master(world, "--yes", "provision", "--to", "pull")
    assert proc.returncode == 0, proc.stdout[-3000:]
    rid = _state(world)["run_id"]
    pod_dir = world["pod_repo"] / "results" / "camp1" / "a"
    (pod_dir / f"aa_repeats_{rid}.json").write_text('{"aa": "ok"}', encoding="utf-8")
    (pod_dir / f"calibration_report_{rid}.json").write_text('{"seed": 1}', encoding="utf-8")
    before = len(_calls(world))
    proc = _master(world, "--only", "pull", "--redo")
    assert proc.returncode == 0, proc.stdout[-3000:]
    stage_calls = _calls(world)[before:]
    ssh_calls = [c for c in stage_calls if c.startswith("ssh")]
    probe = next(i for i, c in enumerate(ssh_calls) if "calibration_report" in c and "test -f" in c)
    (world["home"] / ".ssh_fail_skip").write_text(f"{probe}\n", encoding="utf-8")
    (world["home"] / ".ssh_fail").write_text("3\n", encoding="utf-8")
    proc = _master(world, "--only", "pull", "--redo", CAGE_SSH_RETRY_S="0")
    (world["home"] / ".ssh_fail").write_text("0\n", encoding="utf-8")
    (world["home"] / ".ssh_fail_skip").write_text("0\n", encoding="utf-8")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "the calibration report probe did not reach the pod (ssh rc=255)" in proc.stdout
    st = _state(world)
    assert st["stages"]["pull"]["status"] == "failed"
    # the A/A pull of this run completed before the dead probe; the sibling key
    # is the earlier successful pull's record (state persists across runs), so
    # the proof of the refusal is the stage status and the probe message above
    assert st["pull"]["aa_report"].endswith(f"aa_repeats_{rid}.json")


# ---------------------------------------------------------------------------
# 9. ADR-0176 Group D (owner 2026-10-22, step 1): the batched scoring path is
#    used only after the stage 10 probe proves it on 50 real answers
# ---------------------------------------------------------------------------


def _batched_profile(world: Dict[str, Path], tmp_path: Path) -> None:
    # S2 block F row F-11 (owner 2026-11-04): a batch size comes with the two stated per-row bounds
    prof = tmp_path / "S1_batched.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8")
                    + "\nSCORING_BATCH_SIZE=32\nSCORING_TOL_FAITHFULNESS=0.005\nSCORING_TOL_BERTSCORE=1e-5\n", encoding="utf-8")
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


def test_a_batch_size_without_both_stated_bounds_is_refused_at_stage_0_s2_f11(world: Dict[str, Path], tmp_path: Path) -> None:
    """S2 block F row F-11 (owner 2026-11-04, option (a)): a profile that batches the
    scoring pass states the two per-row bounds the stage 10 proof allows (0 = exact);
    an empty or non-numeric one is refused at stage 0, under --plan too, before the
    suite or any pod, naming the bad key."""
    base = world["profile"].read_text(encoding="utf-8")
    both = ("SCORING_TOL_FAITHFULNESS", "SCORING_TOL_BERTSCORE")
    cases = {"empty": ("SCORING_BATCH_SIZE=32\n", both),
             "text": ("SCORING_BATCH_SIZE=32\nSCORING_TOL_FAITHFULNESS=abc\nSCORING_TOL_BERTSCORE=1e-5\n", both[:1]),
             "negative": ("SCORING_BATCH_SIZE=32\nSCORING_TOL_FAITHFULNESS=0.005\nSCORING_TOL_BERTSCORE=-1e-5\n", both[1:]),
             # review LOW 1: an overflow literal reads as inf on the pod; review gap 6: a signed form is not stated plainly
             "overflow": ("SCORING_BATCH_SIZE=32\nSCORING_TOL_FAITHFULNESS=0.005\nSCORING_TOL_BERTSCORE=1e400\n", both[1:]),
             "plus": ("SCORING_BATCH_SIZE=32\nSCORING_TOL_FAITHFULNESS=+0.5\nSCORING_TOL_BERTSCORE=1e-5\n", both[:1])}
    marker = "needs the stage 10 proof bounds stated (S2 block F row F-11"
    for case, (extra, bad_keys) in cases.items():
        prof = tmp_path / f"S1_{case}.env"
        prof.write_text(base + "\n" + extra, encoding="utf-8")
        world["profile"] = prof
        for argv in ((), ("--plan",)):
            proc = _master(world, *argv)
            assert proc.returncode == 1, (case, argv, proc.stdout[-2000:])
            assert marker in proc.stdout, (case, argv, proc.stdout[-2000:])
            named = proc.stdout.split(marker)[1].splitlines()[0]
            for key in both:
                assert (key in named) is (key in bad_keys), (case, argv, key, named)
            assert not any("provision_pod" in c for c in _calls(world))
    # both stated (0 is a statement: exact): stage 0 passes under --plan and prints them
    good = tmp_path / "S1_bounds.env"
    good.write_text(base + "\nSCORING_BATCH_SIZE=32\nSCORING_TOL_FAITHFULNESS=0\nSCORING_TOL_BERTSCORE=1e-5\n", encoding="utf-8")
    world["profile"] = good
    proc = _master(world, "--plan")
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert "scoring proof bounds (F-11): faithfulness 0, completeness_bertscore 1e-5 (batch size 32)" in proc.stdout


def test_the_stage_10_probe_carries_the_profiles_bounds_after_the_freeze_file_s2_f11(world: Dict[str, Path], tmp_path: Path) -> None:
    """The master passes SCORING_TOL_FAITHFULNESS and SCORING_TOL_BERTSCORE to the probe as
    --tol-faithfulness=X and --tol-bertscore=Y (the = form, review MEDIUM 1: a negative in any
    notation then reaches argparse as a value and the probe answers with a verdict), after
    the freeze file and before --out; the pass keeps its batch size on OK. A stated 0 is sent
    (S1.env). An empty key sends no flag at all, so the probe refuses it by name with the
    full instruments (a live run never gets there: stage 0 refuses the profile)."""
    _batched_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "score")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    probe = [c for c in _calls(world) if c.startswith("scoring_probe ")]
    assert len(probe) == 1
    assert (" --device cuda --freeze-file MyDocs/registration/freeze_resolutions.json --tol-faithfulness=0.005"
            " --tol-bertscore=1e-5 --out logs/scoring_batch_probe.json" in probe[0]), probe[0]
    cmd = (world["home"] / ".cage_jobs" / "score_probe.cmd").read_text(encoding="utf-8")
    assert "--tol-faithfulness=0.005 --tol-bertscore=1e-5" in cmd
    assert "--batch-size 32" in (world["home"] / ".cage_jobs" / "score.cmd").read_text(encoding="utf-8")
    batched = world["profile"].read_text(encoding="utf-8")
    zero = tmp_path / "S1_zerobound.env"
    zero.write_text(batched.replace("SCORING_TOL_FAITHFULNESS=0.005", "SCORING_TOL_FAITHFULNESS=0"), encoding="utf-8")
    world["profile"] = zero
    proc = _master(world, "--only", "score", "--redo")
    assert proc.returncode == 0, proc.stdout[-3000:]
    probe = [c for c in _calls(world) if c.startswith("scoring_probe ")]
    assert len(probe) == 2 and "--tol-faithfulness=0 --tol-bertscore=1e-5 --out" in probe[1], probe[1]
    prof = tmp_path / "S1_onebound.env"
    prof.write_text(batched.replace("SCORING_TOL_BERTSCORE=1e-5", "SCORING_TOL_BERTSCORE="), encoding="utf-8")
    world["profile"] = prof
    proc = _master(world, "--only", "score", "--redo")
    assert proc.returncode == 0, proc.stdout[-3000:]
    probe = [c for c in _calls(world) if c.startswith("scoring_probe ")]
    assert len(probe) == 3 and "--tol-faithfulness=0.005 --out" in probe[2] and "--tol-bertscore" not in probe[2], probe[2]


# ---------------------------------------------------------------------------
# 10. S2 Block B (owner 2026-10-31): the master on four GPUs. B-1 the TP env
#     and the GPU count at stage 0, the stage 4 gate and every engine start;
#     B-2 the degree on the pool checks; B-4 the container disk free-space
#     floor; B-5 the stock note; B-6 the console volume override
# ---------------------------------------------------------------------------


def _four_gpu_profile(world: Dict[str, Path], tmp_path: Path, topology: str = "tp") -> None:
    """The world profile on a four-GPU tensor-parallel pod with both server engines."""
    prof = tmp_path / f"S1_{topology}4.env"
    prof.write_text(world["profile"].read_text(encoding="utf-8").replace("GPU_COUNT=1", "GPU_COUNT=4")
                    .replace('ENGINES="vllm hf"', 'ENGINES="vllm sglang hf"') + f"\nTOPOLOGY={topology}\n", encoding="utf-8")
    world["profile"] = prof


def test_stage_0_records_the_tp_env_and_the_gpu_count_and_refuses_a_mismatch_s2_b1(world: Dict[str, Path], tmp_path: Path) -> None:
    # a single-GPU session: no TP env, one GPU needed, no stock note
    proc = _master(world)
    assert proc.returncode == 10, proc.stdout[-2500:]
    st = _state(world)
    assert st["registration"]["gpu_count_needed"] == "1" and st["registration"]["tp_env"] == {"vllm": "", "sglang": ""}
    assert "S2 Block B-5" not in proc.stdout

    def plans() -> int:
        return sum(1 for c in _calls(world) if "provision_pod" in c)
    before = plans()
    # the grid shards each stack over four GPUs but the profile provisions one: refused before the plan print
    proc = _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_TP_ENV="1")
    assert proc.returncode == 1, proc.stdout[-2500:]
    assert "the profile provisions GPU_COUNT=1 but session a needs 4 GPU(s) per serving stack" in proc.stdout
    assert plans() == before
    st = _state(world)
    assert st["registration"]["gpu_count_needed"] == "4"
    assert st["registration"]["tp_env"] == {"vllm": "CAGE_VLLM_TENSOR_PARALLEL=4", "sglang": "CAGE_SGLANG_TP=4"}
    # an empty count (an older preflight) refuses too
    proc = _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_TP_ENV="1", CAGE_TEST_GPU_COUNT_NEEDED="")
    assert proc.returncode == 1 and "GPU_COUNT_NEEDED empty" in proc.stdout, proc.stdout[-2500:]
    assert plans() == before
    # the four-GPU profile passes, notes that the stock read is per GPU id, and prices four GPUs
    _four_gpu_profile(world, tmp_path)
    proc = _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_TP_ENV="1")
    assert proc.returncode == 0, proc.stdout[-2500:]
    assert "NOTE: a 4-GPU pod: the stock read is per GPU id, so 4 GPUs together in one datacenter are proven only by the create at stage 1 (S2 Block B-5)" in proc.stdout
    assert "--gpu-count 4" in [c for c in _calls(world) if "provision_pod" in c][-1]
    assert _state(world)["stages"]["preflight-mac"]["status"] == "passed"
    # review 2026-10-31 (G8 finding 2): an engine the profile starts with no executable
    # single cell gets no TP env and would start at TP=1 at stage 4: refused at stage 0
    before = plans()
    proc = _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_TP_ENV="1", CAGE_TEST_TP_ENV_ENGINES="vllm")
    assert proc.returncode == 1, proc.stdout[-2500:]
    assert "the profile starts engine sglang (ENGINES or CALIBRATE) on a 4-GPU pod but stage 0 recorded no TP variable for it" in proc.stdout
    assert plans() == before
    assert _state(world)["registration"]["tp_env"] == {"vllm": "CAGE_VLLM_TENSOR_PARALLEL=4", "sglang": ""}
    # the same grid on a profile that starts vLLM alone passes
    world["profile"].write_text(world["profile"].read_text(encoding="utf-8").replace('ENGINES="vllm sglang hf"', 'ENGINES="vllm hf"'), encoding="utf-8")
    proc = _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_TP_ENV="1", CAGE_TEST_TP_ENV_ENGINES="vllm")
    assert proc.returncode == 0, proc.stdout[-2500:]
    assert _state(world)["stages"]["preflight-mac"]["status"] == "passed"


def test_every_engine_start_carries_the_tp_env_and_the_gate_accepts_tp_s2_b1(world: Dict[str, Path], tmp_path: Path) -> None:
    _four_gpu_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "calibrate", CAGE_TEST_TP_ENV="1", CAGE_TEST_HELPER_POOL="1")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    jobs = world["home"] / ".cage_jobs"
    # the budget-free validate starts (no largest pool in this plan)
    vllm_cmd = (jobs / "validate_vllm.cmd").read_text(encoding="utf-8")
    assert "env CAGE_VLLM_TENSOR_PARALLEL=4 bash scripts/2_serving/manage_vllm_server.sh start" in vllm_cmd and "pool-check" not in vllm_cmd
    sgl_cmd = (jobs / "validate_sglang.cmd").read_text(encoding="utf-8")
    assert "env CAGE_SGLANG_TP=4 bash scripts/2_serving/manage_sglang_server.sh start" in sgl_cmd
    # review 2026-10-31 (G8 finding 4): the helper start beside SGLang carries the degree and its pool check records it
    assert "env CAGE_SGLANG_MAX_TOTAL_TOKENS=70437 CAGE_SGLANG_TP=4 VLLM_MAX_MODEL_LEN=8192 bash scripts/2_serving/manage_sglang_server.sh start" in sgl_cmd
    assert "pool-check --engine sglang --planned-tokens 70437 --planned-bytes 11540480000 --tp 4 --out logs/sglang/helper_pool_check.json" in sgl_cmd
    assert any(c.startswith("pool_check --engine sglang") and " --tp 4 " in c for c in _calls(world))
    # the four start-check starts, the reranker start among them
    sc = (jobs / "start_check.cmd").read_text(encoding="utf-8")
    assert "manage_sglang_server.sh SGLANG_KV_CACHE_DTYPE=fp8_e4m3 CAGE_SGLANG_TP=4; stopall" in sc
    assert "manage_vllm_server.sh VLLM_KV_CACHE_DTYPE=fp8 CAGE_VLLM_TENSOR_PARALLEL=4; stopall" in sc
    assert "env CAGE_VLLM_TENSOR_PARALLEL=4 bash scripts/2_serving/manage_vllm_server.sh restart" in sc
    # the floor calibration start, and the seam names the four GPUs
    assert "env CAGE_VLLM_TENSOR_PARALLEL=4 bash scripts/2_serving/manage_vllm_server.sh start" in (jobs / "calibrate_vllm.cmd").read_text(encoding="utf-8")
    assert "CAGE_HARDWARE='NVIDIA H100 80GB HBM3 x 4'" in (jobs / "calibrate_rungs_vllm.cmd").read_text(encoding="utf-8")
    # every launcher start the fakes saw carried the degree
    starts = [c for c in _calls(world) if c.startswith("launcher ") and (" start " in c or " restart " in c)]
    assert len(starts) >= 7 and all(" tp=4" in c for c in starts), starts
    # S2 block D: the pd stack validation is built (the stage 4 smoke runs on a session
    # whose stage 0 recorded pd cells); a TOPOLOGY=pd profile is no longer refused by name
    _four_gpu_profile(world, tmp_path, topology="pd")
    proc = _master(world, "--from", "validate", "--redo", "--to", "validate", CAGE_TEST_TP_ENV="1")
    assert proc.returncode == 0, proc.stdout[-2500:]
    assert "the pd stack validation is a separate Spec" not in proc.stdout and "TOPOLOGY=pd" not in proc.stdout


def test_the_largest_pool_start_carries_the_tp_env_and_the_check_records_the_degree_s2_b2(world: Dict[str, Path], tmp_path: Path) -> None:
    _four_gpu_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_TP_ENV="1", CAGE_TEST_LARGEST_POOL="1")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    cmd = (world["home"] / ".cage_jobs" / "validate_vllm.cmd").read_text(encoding="utf-8")
    assert "env CAGE_KV_BUDGET_BYTES=48936960000 CAGE_VLLM_TENSOR_PARALLEL=4 VLLM_MAX_MODEL_LEN=32768 bash scripts/2_serving/manage_vllm_server.sh start" in cmd
    assert "pool-check --engine vllm --planned-tokens 298687 --planned-bytes 48936960000 --tp 4 --out logs/vllm/largest_pool_check.json" in cmd
    assert any(c.startswith("pool_check --engine vllm") and " --tp 4 " in c for c in _calls(world))
    assert _state(world)["registration"]["largest_pool"]["vllm"]["tp"] == "4"
    assert "LARGEST_POOL_OK engine=vllm" in proc.stdout


def test_stage_1_records_a_console_volume_from_the_override_s2_b6(world: Dict[str, Path]) -> None:
    # the id without its datacenter is refused before anything is created
    proc = _master(world, "--yes", "provision", "--to", "provision", CAGE_VOLUME_ID_OVERRIDE="volx")
    assert proc.returncode == 1, proc.stdout[-2500:]
    assert "CAGE_VOLUME_ID_OVERRIDE=volx needs CAGE_VOLUME_DC_OVERRIDE=<datacenter id>" in proc.stdout
    assert not any("network-volume create" in c or ("provision_pod" in c and "--yes" in c) for c in _calls(world))
    st = _state(world)
    assert st["stages"]["provision"]["status"] == "failed" and st["go"] == []
    # an id that is not listed (a typo, a deleted volume) refuses: no second volume bills on it
    proc = _master(world, "--from", "provision", "--yes", "provision", "--to", "provision",
                   CAGE_VOLUME_ID_OVERRIDE="volx", CAGE_VOLUME_DC_OVERRIDE="US-NE-1")
    assert proc.returncode == 1, proc.stdout[-2500:]
    assert "CAGE_VOLUME_ID_OVERRIDE=volx is not in the network-volume listing: check the id in the console; nothing is created" in proc.stdout
    assert not any("network-volume create" in c or ("provision_pod" in c and "--yes" in c) for c in _calls(world))
    st = _state(world)
    assert st["pod"]["volume_id"] == "" and st["stages"]["provision"]["status"] == "failed" and st["go"] == []
    # review 2026-10-31 (G8 finding 3): the listing sites the volume elsewhere: refused, record cleared, nothing created
    proc = _master(world, "--from", "provision", "--yes", "provision", "--to", "provision",
                   CAGE_VOLUME_ID_OVERRIDE="volx", CAGE_VOLUME_DC_OVERRIDE="US-CA-2", CAGE_TEST_VOLUMES='[{"id": "volx", "dataCenterId": "US-NE-1"}]')
    assert proc.returncode == 1, proc.stdout[-2500:]
    assert "CAGE_VOLUME_DC_OVERRIDE=US-CA-2 but the listing sites volume volx in US-NE-1: set CAGE_VOLUME_DC_OVERRIDE=US-NE-1; nothing is created" in proc.stdout
    assert not any("network-volume create" in c or ("provision_pod" in c and "--yes" in c) for c in _calls(world))
    st = _state(world)
    assert st["pod"]["volume_id"] == "" and st["stages"]["provision"]["status"] == "failed" and st["go"] == []
    # with the datacenter the listing confirms: recorded, found in the listing, reused; no volume is created
    proc = _master(world, "--from", "provision", "--yes", "provision", "--to", "provision",
                   CAGE_VOLUME_ID_OVERRIDE="volx", CAGE_VOLUME_DC_OVERRIDE="US-NE-1", CAGE_TEST_VOLUMES='[{"id": "volx", "dataCenterId": "US-NE-1"}]')
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "volume volx in US-NE-1 recorded from CAGE_VOLUME_ID_OVERRIDE (S2 Block B-6)" in proc.stdout
    assert "volume volx in US-NE-1 is already recorded (kept after a refused pod): reused, no second volume is created" in proc.stdout
    calls = _calls(world)
    assert not any("network-volume create" in c for c in calls)
    pods = [c for c in calls if "provision_pod" in c and "--yes" in c]
    assert len(pods) == 1 and "--data-center-ids US-NE-1 --network-volume-id volx" in pods[0]
    st = _state(world)
    assert st["pod"]["id"] == "pod123" and st["pod"]["volume_id"] == "volx" and st["pod"]["dc"] == "US-NE-1"
    assert st["stages"]["provision"]["status"] == "passed" and [g["stage"] for g in st["go"]] == ["provision"]


def test_the_volume_override_never_drops_a_volume_this_landing_recorded_s2_b6(world: Dict[str, Path]) -> None:
    # review 2026-10-31 (G8 finding 3): a refused pod leaves the created volume recorded
    # (ADR-0176 Group B, billing storage); the override must not replace it in the state
    proc = _master(world, "--yes", "provision", "--to", "provision", CAGE_TEST_PROVISION_YES_RC="1")
    assert proc.returncode == 1, proc.stdout[-2500:]
    assert _state(world)["pod"]["volume_id"] == "vol123"
    creates = sum(1 for c in _calls(world) if "network-volume create" in c or ("provision_pod" in c and "--yes" in c))
    proc = _master(world, "--from", "provision", "--yes", "provision", "--to", "provision",
                   CAGE_VOLUME_ID_OVERRIDE="volx", CAGE_VOLUME_DC_OVERRIDE="US-NE-1", CAGE_TEST_VOLUMES='[{"id": "volx", "dataCenterId": "US-NE-1"}]')
    assert proc.returncode == 1, proc.stdout[-2500:]
    assert "this landing already recorded volume vol123 (kept after a refused pod, billing storage); CAGE_VOLUME_ID_OVERRIDE=volx would drop it from the state" in proc.stdout
    assert sum(1 for c in _calls(world) if "network-volume create" in c or ("provision_pod" in c and "--yes" in c)) == creates
    st = _state(world)
    assert st["pod"]["volume_id"] == "vol123" and st["pod"]["dc"] == "US-NE-1" and st["stages"]["provision"]["status"] == "failed"
    # the recorded id itself is a no-op override: reused as before, nothing created
    proc = _master(world, "--from", "provision", "--yes", "provision", "--to", "provision",
                   CAGE_VOLUME_ID_OVERRIDE="vol123", CAGE_VOLUME_DC_OVERRIDE="US-NE-1", CAGE_TEST_VOLUMES='[{"id": "vol123", "dataCenterId": "US-NE-1"}]')
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "volume vol123 in US-NE-1 is already recorded (kept after a refused pod): reused, no second volume is created" in proc.stdout
    assert sum(1 for c in _calls(world) if "network-volume create" in c) == 1
    assert _state(world)["pod"]["id"] == "pod123" and _state(world)["pod"]["volume_id"] == "vol123"


def test_stage_3_records_the_weights_disk_free_space_and_refuses_below_the_floor_s2_b4(world: Dict[str, Path], tmp_path: Path) -> None:
    # review 2026-10-31 (G8 finding 1): the read is of the disk holding the model cache
    # (POD_HF_CACHE, the pod user's cache by default), never of POD_REPO on the network
    # volume; the fake profile names a local cache, absent until stage 3 creates it
    cache = world["home"] / ".cache" / "huggingface"
    assert not cache.exists()
    proc = _master(world, "--yes", "provision", "--to", "setup")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    free = _state(world)["pod"]["disk_free_gb"]
    assert re.fullmatch(r"[0-9]+", free), free
    assert f"weights disk ({cache}): {free} GB free after the prefetch (DISK_GB=120, floor 20 GB)" in proc.stdout
    steps = world["exp_root"] / "S1" / DATE / "extras" / "steps"
    (log,) = list(steps.glob("setup_*weights_disk_free_space*.log"))
    cmd = log.read_text(encoding="utf-8").split("# cmd:")[1].splitlines()[0]  # the command is %q-escaped in the step log
    assert "df\\ -Pk" in cmd and str(cache) in cmd and str(world["pod_repo"]) not in cmd and "awk" not in cmd, cmd
    assert cache.is_dir()  # the pod-side mkdir -p ran first: a cache the prefetch has not created never fails the read
    # the recorded figure is the POSIX "Available" column of that path's disk (within 1 GB of a second read)
    mac = subprocess.run(["df", "-Pk", str(cache)], capture_output=True, text=True, check=True).stdout.splitlines()[-1].split()
    assert abs(int(mac[3]) // 1024 // 1024 - int(free)) <= 1, (mac, free)
    # a floor above the disk refuses, names the profile key to raise, and no engine starts
    proc = _master(world, "--from", "setup", "--redo", "--to", "setup", DISK_FREE_MIN_GB="999999999")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert f"[FAIL] setup: {free} GB free on the disk holding the model cache ({cache}) after the prefetch, below DISK_FREE_MIN_GB=999999999 (DISK_GB=120 holds the weights, S2 Block B-4 option a): raise DISK_GB in the profile" in proc.stdout
    assert _state(world)["stages"]["setup"]["status"] == "failed"
    assert not (world["home"] / ".cage_jobs" / "validate_vllm.cmd").exists()
    # without the profile key the command names the pod user's cache on the container disk, whatever the Mac answers
    bare = tmp_path / "S1_nocache.env"
    bare.write_text("\n".join(ln for ln in world["profile"].read_text(encoding="utf-8").splitlines() if not ln.strip().startswith("POD_HF_CACHE=")) + "\n", encoding="utf-8")
    world["profile"] = bare
    _master(world, "--from", "setup", "--redo", "--to", "setup")
    (log,) = list(steps.glob("setup_*weights_disk_free_space*.log"))
    cmd = log.read_text(encoding="utf-8").split("# cmd:")[1].splitlines()[0]
    assert "/root/.cache/huggingface" in cmd and str(world["pod_repo"]) not in cmd, cmd


def test_the_shipped_s2_profile_holds_the_weights_on_a_300_gb_container_disk_s2_b4() -> None:
    keys = ("DISK_GB", "GPU_COUNT", "TOPOLOGY", "VOLUME_GB", "DISK_FREE_MIN_GB", "ENGINES")
    fmt = "|".join("${%s:-}" % k for k in keys)
    proc = subprocess.run(["bash", "-c", f'set -a; source "{EXPDIR}/profiles/_common.env"; source "{EXPDIR}/profiles/S2.env"; printf "%s" "{fmt}"'],
                          capture_output=True, text=True, env=PROFILE_ENV)
    assert proc.returncode == 0, proc.stderr
    got = dict(zip(keys, proc.stdout.split("|")))
    # B-4 option a: 300 GB holds the 80B (163 GB bf16) or the 70B (141 GB); the master's 20 GB floor applies
    assert got == {"DISK_GB": "300", "GPU_COUNT": "4", "TOPOLOGY": "tp", "VOLUME_GB": "300", "DISK_FREE_MIN_GB": "", "ENGINES": "vllm sglang"}


# ---------------------------------------------------------------------------
# 11. S2 block E (2026-11-01): the two landing profiles, the Hub token path (E-2)
#     and the served-model pins (E-4)
# ---------------------------------------------------------------------------

S2_HELPERS = ("intfloat/e5-large-v2", "BAAI/bge-reranker-large", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
              "KRLabsOrg/lettucedect-base-modernbert-en-v1", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
              "sentence-transformers/all-MiniLM-L6-v2", "roberta-base")


def _profile_values(name: str, keys: tuple) -> Dict[str, str]:
    fmt = "|".join("${%s:-}" % k for k in keys)
    proc = subprocess.run(["bash", "-c", f'set -a; source "{EXPDIR}/profiles/_common.env"; source "{EXPDIR}/profiles/{name}"; printf "%s" "{fmt}"'],
                          capture_output=True, text=True, env=PROFILE_ENV)
    assert proc.returncode == 0, proc.stderr
    return dict(zip(keys, proc.stdout.split("|")))


def _profile_keys(name: str) -> set:
    return set(re.findall(r"^([A-Z_][A-Z0-9_]*)=", (EXPDIR / "profiles" / name).read_text(encoding="utf-8"), re.M))


def test_the_shipped_s2_landing_profiles_carry_every_s1_key_and_the_block_e_picks_s2_e1() -> None:
    s1_keys = _profile_keys("S1.env")
    for name in ("S2A.env", "S2B.env"):
        missing = s1_keys - _profile_keys(name)
        assert not missing, f"{name}: S1.env keys without a value: {sorted(missing)}"
    keys = ("EXP", "SESSION", "MODEL", "MODEL_SLUG", "GPU_ID", "GPU_COUNT", "ENGINES", "CALIBRATE", "TOPOLOGY", "DC_PREFS",
            "HOURS", "RUN_BUDGET_HOURS", "SEATBELT", "SETUP_BOUND_MIN", "CALIBRATE_RUNGS_BOUND_S", "SCORE_BOUND_MIN", "PULL_MARGIN_MIN",
            "FLOOR_AVG_SEQ_TOKENS", "INSTALL_LMCACHE", "LMCACHE_VERSION", "PIN_MODEL_REVISION", "RUNG_ALL_CLASSES", "CLASS_CEILINGS",
            "MIN_IN_REGIME_WINDOWS", "MAX_NULL_FRACTION", "SUITE_ON_POD", "MAC_SUITE", "FRONT_LOAD", "VOLUME_GB", "DISK_GB",
            "QUERY_MANIFESTS", "CALIBRATION_MANIFEST", "PREFETCH_MODELS", "HF_TOKEN_FILE", "SKIP_BLOCKED",
            "CALIBRATION_BUDGET_FRACTION", "REGIME_GATE_LOADS", "CONFIRMATORY_ARMS",
            "SCORING_TOL_FAITHFULNESS", "SCORING_TOL_BERTSCORE")
    a, b = _profile_values("S2A.env", keys), _profile_values("S2B.env", keys)
    shared = {"GPU_ID": "NVIDIA H200", "GPU_COUNT": "4", "ENGINES": "vllm sglang", "CALIBRATE": "vllm sglang", "TOPOLOGY": "tp",
              "DC_PREFS": "US-CA-2 US-CO-1 CA-MTL-3 CA-MTL-4 EU-FR-1 AP-JP-1", "CALIBRATE_RUNGS_BOUND_S": "32400",
              "PIN_MODEL_REVISION": "1", "RUNG_ALL_CLASSES": "1", "CLASS_CEILINGS": "0",
              # decisions 1 to 3 of the batch 2/3 spec (owner 2026-11-03): the floor at the grids' floor rung,
              # the gate at the registered count per load, the ladder bound that holds 15 triples per engine
              "CALIBRATION_BUDGET_FRACTION": "1.0", "REGIME_GATE_LOADS": "0.95 1.05", "MIN_IN_REGIME_WINDOWS": "0.95:2 1.05:3",
              "MAX_NULL_FRACTION": "0.06", "SUITE_ON_POD": "1", "MAC_SUITE": "1", "FRONT_LOAD": "0", "VOLUME_GB": "300",
              "DISK_GB": "300", "SKIP_BLOCKED": "0",
              # S2 block F row F-11 (owner 2026-11-04, option (a)): the per-row bounds the stage 10 proof allows
              "SCORING_TOL_FAITHFULNESS": "0.005", "SCORING_TOL_BERTSCORE": "1e-5"}
    for k, v in shared.items():
        assert a[k] == v and b[k] == v, (k, a[k], b[k])
    own = ("EXP", "SESSION", "MODEL", "MODEL_SLUG", "HOURS", "RUN_BUDGET_HOURS", "SEATBELT", "SETUP_BOUND_MIN",
           "SCORE_BOUND_MIN", "PULL_MARGIN_MIN", "FLOOR_AVG_SEQ_TOKENS", "INSTALL_LMCACHE", "LMCACHE_VERSION", "CONFIRMATORY_ARMS")
    # the seatbelts grew with the calibrate term (2 x 240 min) under decision 3; the rosters are decision 2's
    assert {k: a[k] for k in own} == {
        "EXP": "S2A", "SESSION": "s2a", "MODEL": "meta-llama/Llama-3.3-70B-Instruct", "MODEL_SLUG": "llama-3.3-70b",
        "HOURS": "50", "RUN_BUDGET_HOURS": "45", "SEATBELT": "83h", "SETUP_BOUND_MIN": "40", "SCORE_BOUND_MIN": "120",
        "PULL_MARGIN_MIN": "150", "FLOOR_AVG_SEQ_TOKENS": "4779", "INSTALL_LMCACHE": "0", "LMCACHE_VERSION": "",
        "CONFIRMATORY_ARMS": "B1 B3 B4 B6 B7 B11 B12"}
    assert {k: b[k] for k in own} == {
        "EXP": "S2B", "SESSION": "cd-act1", "MODEL": "Qwen/Qwen3-Next-80B-A3B-Instruct", "MODEL_SLUG": "qwen3-next-80b",
        "HOURS": "60", "RUN_BUDGET_HOURS": "55", "SEATBELT": "98h", "SETUP_BOUND_MIN": "50", "SCORE_BOUND_MIN": "150",
        "PULL_MARGIN_MIN": "180", "FLOOR_AVG_SEQ_TOKENS": "2694", "INSTALL_LMCACHE": "1", "LMCACHE_VERSION": "0.4.4",
        "CONFIRMATORY_ARMS": "B1 B3 B4 B6 B7 B8"}
    # the served model plus the seven helpers, and no Qwen3-14B (no pod reader defaults to it)
    for p in (a, b):
        assert p["PREFETCH_MODELS"].split() == [p["MODEL"], *S2_HELPERS], p["PREFETCH_MODELS"]
    # S2B serves its MuSiQue questions packed in the 80B's tokens (the file exists and calibrates too);
    # S2A names the 70B-token Qasper file, which stage 0 refuses until the gated tokenizer can be loaded (A-6)
    assert b["QUERY_MANIFESTS"] == "musique=data/manifests/musique_2000x3_seed42_wr50_tok-Qwen3-Next-80B-A3B-Instruct.json"
    assert b["CALIBRATION_MANIFEST"] == b["QUERY_MANIFESTS"].partition("=")[2] and (REPO_ROOT / b["CALIBRATION_MANIFEST"]).is_file()
    assert a["QUERY_MANIFESTS"] == "qasper=data/manifests/qasper_852x12_seed42_wr50_tok-Llama-3.3-70B-Instruct.json"
    assert a["CALIBRATION_MANIFEST"] == a["QUERY_MANIFESTS"].partition("=")[2]
    # E-2: the token file key is set on the gated 70B only, as an expanded path outside the repo
    assert a["HF_TOKEN_FILE"].startswith("/") and not a["HF_TOKEN_FILE"].startswith(str(REPO_ROOT)), a["HF_TOKEN_FILE"]
    assert b["HF_TOKEN_FILE"] == ""


def test_the_block_b_s2_profile_is_marked_superseded_by_the_landing_profiles_s2_e1() -> None:
    head = (EXPDIR / "profiles" / "S2.env").read_text(encoding="utf-8").splitlines()[:2]
    assert head[0].startswith("# SUPERSEDED 2026-11-01") and "S2A.env" in head[0] and "S2B.env" in head[1], head


def _profile_with(world: Dict[str, Path], name: str, extra: str) -> None:
    # a profile variant (the master reads profile keys from the profile; _common.env's empty
    # HF_TOKEN_FILE default means a value in the environment never reaches it, review 2026-11-01 L2)
    p = world["tmp"] / name
    p.write_text(world["profile"].read_text(encoding="utf-8") + "\n" + extra + "\n", encoding="utf-8")
    world["profile"] = p


def _pod_hub_dir(world: Dict[str, Path]) -> Path:
    # the fake ssh runs every remote command with HOME = the pod home, so the pod's hub reads
    # HOME/.cache/huggingface/token there; the world profile's POD_HF_CACHE names that directory
    return world["home"] / ".cache" / "huggingface"


def test_a_mac_token_file_ships_to_the_pod_cache_as_a_file_and_never_enters_a_log_s2_e2(world: Dict[str, Path], tmp_path: Path) -> None:
    token = "hf_FAKEtokenFORtheTESTonly0123456789"
    tf = tmp_path / "hf_token"  # outside the repo
    tf.write_text(token + "\n", encoding="utf-8")
    tf.chmod(0o600)
    _profile_with(world, "S1_token.env", f"HF_TOKEN_FILE={tf}")
    proc = _master(world, "--yes", "provision", "--to", "ship")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "HF_TOKEN_FILE_OK path=" + str(tf) in proc.stdout
    landed = _pod_hub_dir(world) / "token"
    assert landed.read_text(encoding="utf-8") == token + "\n" and stat.S_IMODE(landed.stat().st_mode) == 0o600
    calls = _calls(world)
    assert any(c.startswith("scp ") and str(tf) in c and c.rstrip().endswith("/token") for c in calls), calls
    assert any("chmod 600 " in c and c.rstrip().endswith("echo HF_TOKEN_OK") for c in calls), calls
    # the token's bytes exist in the Mac file and the pod file only: no step log, job command,
    # state, call line or ledger line carries them (the paths may appear, the contents never)
    carriers = [world["log"], world["ledger"], *world["exp_root"].rglob("*"), *(world["home"] / ".cage_jobs").glob("*")]
    for pth in carriers:
        if pth.is_file():
            assert token not in pth.read_text(encoding="utf-8", errors="replace"), pth
    assert token not in proc.stdout and token not in proc.stderr
    steps = world["exp_root"] / "S1" / DATE / "extras" / "steps"
    (log,) = list(steps.glob("ship_*scp_HF_token_file*.log"))
    assert str(tf) in log.read_text(encoding="utf-8")
    # review 2026-11-01 M2: the pod named its token path first, and it is the landed file
    (log,) = list(steps.glob("ship_*the_pod_s_token_path*.log"))
    assert f"HF_TOKEN_PATH={landed};" in log.read_text(encoding="utf-8")


def test_a_cache_dir_the_pod_hub_does_not_read_is_refused_before_the_token_is_copied_s2_e2(world: Dict[str, Path], tmp_path: Path) -> None:
    # review 2026-11-01 (block E, M2): a profile override of POD_HF_CACHE away from the pod's
    # hub directory would land the token where nothing reads it; ship refuses before the copy
    tf = tmp_path / "hf_token"
    tf.write_text("hf_FAKEtokenFORtheTESTonly0123456789\n", encoding="utf-8")
    tf.chmod(0o600)
    elsewhere = tmp_path / "elsewhere" / "huggingface"
    _profile_with(world, "S1_token_elsewhere.env", f"HF_TOKEN_FILE={tf}\nPOD_HF_CACHE={elsewhere}")
    proc = _master(world, "--yes", "provision", "--to", "ship")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert f"[FAIL] the pod reads its Hub token at another path than {elsewhere}/token" in proc.stdout
    assert not (elsewhere / "token").exists() and not (_pod_hub_dir(world) / "token").exists()
    assert not any(c.startswith("scp ") and str(tf) in c for c in _calls(world))
    assert _state(world)["stages"]["ship"]["status"] == "failed"


@pytest.mark.parametrize("case", ["missing", "mode644", "in_repo", "tilde"])
def test_a_bad_token_file_is_refused_at_stage_0_before_the_suite_or_any_pod_s2_e2(world: Dict[str, Path], tmp_path: Path, case: str) -> None:
    probe = REPO_ROOT / "tests" / f".hf_token_probe_{os.getpid()}"
    try:
        if case == "missing":
            path, reason = str(tmp_path / "absent"), "not readable"
        elif case == "mode644":
            tf = tmp_path / "hf_token"
            tf.write_text("hf_x\n", encoding="utf-8")
            tf.chmod(0o644)
            path, reason = str(tf), "mode 0644, not 0600"
        elif case == "in_repo":
            probe.write_text("hf_x\n", encoding="utf-8")
            probe.chmod(0o600)
            path, reason = str(probe), "inside the repo"
        else:
            path, reason = "~/absent_token", "starts with a literal ~"
        _profile_with(world, "S1_badtoken.env", f'HF_TOKEN_FILE="{path}"')  # quoted: bash keeps a literal ~
        proc = _master(world)
        assert proc.returncode == 1, proc.stdout[-3000:]
        assert f"[FAIL] HF_TOKEN_FILE={path}: " in proc.stdout and reason in proc.stdout, proc.stdout[-1500:]
        calls = _calls(world)
        assert not any(c.startswith("provision_pod") or c.startswith("run_tests") for c in calls), calls
    finally:
        if probe.exists():
            probe.unlink()


def test_an_empty_token_file_key_leaves_the_flow_unchanged_and_the_shell_cannot_set_it_s2_e2(world: Dict[str, Path], tmp_path: Path) -> None:
    tf = tmp_path / "hf_token"
    tf.write_text("hf_x\n", encoding="utf-8")
    tf.chmod(0o600)
    # review 2026-11-01 L2: a value exported in the operator's shell never ships a token on a
    # profile that does not set the key (_common.env's empty default wins over the environment)
    proc = _master(world, "--yes", "provision", "--to", "ship", HF_TOKEN_FILE=str(tf))
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert not list((world["exp_root"] / "S1" / DATE / "extras" / "steps").glob("*HF_token*"))
    assert not (_pod_hub_dir(world) / "token").exists()
    assert "HF_TOKEN" not in proc.stdout


def test_a_served_model_slot_named_for_the_model_pins_it_and_only_prefetched_models_carry_pins_s2_e4(world: Dict[str, Path], tmp_path: Path) -> None:
    _profile_with(world, "S1_pin_slot.env", "PIN_MODEL_REVISION=1")
    freeze_path = world["tmp"] / "freeze_resolutions.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    # no slot named exactly served_model: the pin sits in a slot named for the model, as the S2 freeze file does
    freeze["INSTRUMENT_REVISIONS"] = {"served_model_other": {"model": "Other/Model-X", "revision": "def456"},
                                      "served_model_qwen3_14b": {"model": "Qwen/Qwen3-14B", "revision": "abc123"}}
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    proc = _master(world, "--yes", "provision", "--to", "setup")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    cmd = (world["home"] / ".cage_jobs" / "setup.cmd").read_text(encoding="utf-8")
    assert "CAGE_HUB_CHECK_MODEL='Qwen/Qwen3-14B=abc123'" in cmd, cmd
    # review 2026-11-01 M1: a pin for a model this profile does not prefetch stays out of the
    # setup command (the S1 command keeps its eight pairs with the two S2 slots in the file)
    prefetch = cmd.split("PREFETCH_REVISIONS='")[1].split("'")[0].split()
    assert prefetch == ["Qwen/Qwen3-14B=abc123"] and "Other/Model-X" not in cmd, prefetch


def test_a_served_model_slot_for_another_model_is_no_pin_and_two_revisions_for_one_model_are_none_s2_e4(world: Dict[str, Path], tmp_path: Path) -> None:
    _profile_with(world, "S1_pin_other.env", "PIN_MODEL_REVISION=1")
    freeze_path = world["tmp"] / "freeze_resolutions.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["INSTRUMENT_REVISIONS"] = {"served_model_other": {"model": "Other/Model-X", "revision": "def456"}}
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    proc = _master(world)
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "PIN_MODEL_REVISION=1-but-no-served_model-revision-for-Qwen/Qwen3-14B" in proc.stdout
    assert not any("provision_pod" in c for c in _calls(world))
    # review 2026-11-01 L3: two slots naming this model with different revisions refuse the same
    # way (no silent first match); two slots agreeing on one revision are that pin
    freeze["INSTRUMENT_REVISIONS"] = {"served_model": {"model": "Qwen/Qwen3-14B", "revision": "abc123"},
                                      "served_model_again": {"model": "Qwen/Qwen3-14B", "revision": "zzz999"}}
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    proc = _master(world)
    assert proc.returncode == 1 and "PIN_MODEL_REVISION=1-but-no-served_model-revision-for-Qwen/Qwen3-14B" in proc.stdout, proc.stdout[-2000:]
    freeze["INSTRUMENT_REVISIONS"]["served_model_again"]["revision"] = "abc123"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    proc = _master(world, "--yes", "provision", "--to", "setup")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "CAGE_HUB_CHECK_MODEL='Qwen/Qwen3-14B=abc123'" in (world["home"] / ".cage_jobs" / "setup.cmd").read_text(encoding="utf-8")


def test_the_live_freeze_file_pins_both_s2_served_models_and_every_prefetched_model_s2_e4() -> None:
    freeze = REPO_ROOT / "MyDocs" / "registration" / "freeze_resolutions.json"
    if not freeze.is_file():
        pytest.skip("the registration drafts live in MyDocs (not in every checkout)")
    slots = json.loads(freeze.read_text(encoding="utf-8"))["INSTRUMENT_REVISIONS"]
    pins = {v["model"]: v["revision"] for v in slots.values() if isinstance(v, dict) and v.get("model")}
    for name, model in (("served_model_llama33_70b", "meta-llama/Llama-3.3-70B-Instruct"),
                        ("served_model_qwen3_next_80b", "Qwen/Qwen3-Next-80B-A3B-Instruct")):
        assert slots[name]["model"] == model and re.fullmatch(r"[0-9a-f]{40}", slots[name]["revision"]), name
    for p in ("S2A.env", "S2B.env"):
        for model in _profile_values(p, ("PREFETCH_MODELS",))["PREFETCH_MODELS"].split():
            assert re.fullmatch(r"[0-9a-f]{40}", pins.get(model, "")), f"{p}: {model} has no 40-hex pin in the freeze file"


# ---------------------------------------------------------------------------
# S2 block C (2026-11-01, register C-1 / A-7, charter 5.2): the identity gate
# runs once at stage 4 on a gated session (stage 0's IDENTITY_GATED=1), after
# the engine probes and before the start check: one prompt build, then three
# launcher starts (vLLM prefix on, vLLM prefix off, SGLang) each with its run
# phase, then one verdict. PASS and FAIL are results the plan reads at stage
# 6; ERROR fails stage 4 before anything bills at stage 5.
# ---------------------------------------------------------------------------


def _gated_profile(world: Dict[str, Path], tmp_path: Path, *, engines: str = "vllm sglang hf", extra: str = "") -> None:
    """A profile that can carry the gate: the engines, the musique manifest the
    gate draws its prompts from, and the served-model revision pin."""
    p = tmp_path / "S1_gated.env"
    text = world["profile"].read_text(encoding="utf-8").replace('ENGINES="vllm hf"', f'ENGINES="{engines}"')
    # the gate adds 60 min to the validate bound: the world's 12 h seatbelt (720 min) no longer covers the 766 min needed
    text = text.replace("SEATBELT=12h", "SEATBELT=14h")
    text = text.replace('QUERY_MANIFESTS="squad_v2=data/manifests/squad_v2_50x3_seed42.json"',
                        'QUERY_MANIFESTS="squad_v2=data/manifests/squad_v2_50x3_seed42.json musique=data/manifests/musique_2000x1_seed42_wr50.json"')
    p.write_text(text + "\nPIN_MODEL_REVISION=1\n" + extra + "\n", encoding="utf-8")
    freeze_path = world["tmp"] / "freeze_resolutions.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["INSTRUMENT_REVISIONS"] = {"served_model": {"model": "Qwen/Qwen3-14B", "revision": "abc123"}}
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    world["profile"] = p


def _job_names(world: Dict[str, Path]) -> List[str]:
    names: List[str] = []
    for n in re.findall(r"\.cage_jobs/([A-Za-z0-9_.-]+)\.cmd", world["log"].read_text(encoding="utf-8")):
        if not names or names[-1] != n:
            names.append(n)
    return names


def _job_cmd(world: Dict[str, Path], name: str) -> str:
    return (world["home"] / ".cage_jobs" / f"{name}.cmd").read_text(encoding="utf-8")


def test_stage_4_runs_the_identity_gate_between_the_engine_probes_and_the_start_check_s2_c1(world: Dict[str, Path], tmp_path: Path) -> None:
    _gated_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_LARGEST_POOL="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    # stage 0 prints and records the gate like NEEDS_LMCACHE, and the stage 4 bound grows by the gate bound
    assert "IDENTITY_GATED=1" in proc.stdout and _state(world)["registration"]["identity_gated"] in ("1", 1)
    assert "validate 170," in proc.stdout, [ln for ln in proc.stdout.splitlines() if "seatbelt" in ln]
    assert "three engine starts at stage 4" in proc.stdout, "vLLM on, vLLM off and SGLang (review 2026-11-01 LOW 5)"
    assert _job_names(world) == ["setup", "load_gate", "datasets_offline", "validate_vllm", "validate_sglang", "identity_gate",
                                 "start_check", "calibrate_vllm", "calibrate_rungs_vllm", "plan"]
    cmd = _job_cmd(world, "identity_gate")
    pr = world["pod_repo"]
    assert cmd.startswith(f"cd {pr} && export PATH={pr}/cage-env/bin:$PATH VLLM_START_TIMEOUT=600"), cmd[:200]
    assert " SGLANG_START_TIMEOUT=600" in cmd and " HF_HUB_OFFLINE=1 CAGE_MODEL_REVISION=abc123" in cmd
    # the prompt build: the profile's musique manifest, trial 1, the pinned revision, the session
    assert ("cage-env/bin/python scripts/checks/identity_gate.py prompts --manifest data/manifests/musique_2000x1_seed42_wr50.json "
            "--dataset musique --trial 1 --count 200 --model 'Qwen/Qwen3-14B' --revision abc123 --session a "
            "--out logs/identity_gate/prompts.json") in cmd
    # three starts in order: vLLM prefix on, vLLM prefix off, SGLang; each at the engine's largest-pool env
    on = "env CAGE_KV_BUDGET_BYTES=48936960000 VLLM_MAX_MODEL_LEN=32768 bash scripts/2_serving/manage_vllm_server.sh start 'Qwen/Qwen3-14B' && "
    assert cmd.count(on[:-len(" && ")]) == 2
    i_on = cmd.index(on)
    i_off = cmd.index(on.replace("' && ", "' --no-prefix-cache && "))
    i_sgl = cmd.index("bash scripts/2_serving/manage_sglang_server.sh start 'Qwen/Qwen3-14B' && ")
    assert i_on < i_off < i_sgl
    assert cmd.count("bash scripts/2_serving/manage_vllm_server.sh stop") == 2
    assert cmd.count("bash scripts/2_serving/manage_sglang_server.sh stop") == 1
    run = "cage-env/bin/python scripts/checks/identity_gate.py run --engine {e} --phase {ph} --api-base {api} --model 'Qwen/Qwen3-14B' --prompts logs/identity_gate/prompts.json --passes {n} --out logs/identity_gate/{ph}.json"
    assert run.format(e="vllm", ph="vllm_on", api="http://localhost:8000", n=2) in cmd
    assert run.format(e="vllm", ph="vllm_off", api="http://localhost:8000", n=1) in cmd
    assert run.format(e="sglang", ph="sglang", api="http://localhost:30000", n=2) in cmd
    assert cmd.count("identity_gate.py run ") == 3
    assert "cage-env/bin/python scripts/checks/identity_gate.py verdict --phases logs/identity_gate --out logs/identity_gate.json" in cmd
    assert "--no-sglang" not in cmd and cmd.rstrip().endswith("echo COMPUTE_APPS=$n; exit $rc")
    # the fake launchers saw the three starts, the control one prefix-off
    starts = [c for c in _calls(world) if c.startswith("launcher ") and " start " in c]
    assert "launcher vllm start Qwen/Qwen3-14B --no-prefix-cache" in starts
    # every gate file is pulled beside the landing's extras, the verdict is recorded, the plan reads it
    extras = world["exp_root"] / "S1" / DATE / "extras" / "identity_gate"
    assert sorted(p.name for p in extras.iterdir()) == ["identity_gate.json", "prompts.json", "sglang.json", "vllm_off.json", "vllm_on.json"]
    assert json.loads((extras / "identity_gate.json").read_text(encoding="utf-8"))["verdict"] == "PASS"
    assert _state(world)["stages"]["validate"]["identity_gate"] == "PASS"
    assert "IDENTITY_GATE PASS" in proc.stdout
    plan_cmd = _job_cmd(world, "plan")
    assert " --identity-gate logs/identity_gate.json" in plan_cmd and "--accept-identity-fail" not in plan_cmd


def test_a_vllm_only_gated_profile_runs_two_starts_and_records_sglang_absent(world: Dict[str, Path], tmp_path: Path) -> None:
    _gated_profile(world, tmp_path, engines="vllm hf")
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_IDENTITY_GATED="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    cmd = _job_cmd(world, "identity_gate")
    assert "manage_sglang_server.sh" not in cmd and cmd.count("identity_gate.py run ") == 2
    assert "--out logs/identity_gate.json --no-sglang" in cmd
    assert "validate 145," in proc.stdout, [ln for ln in proc.stdout.splitlines() if "seatbelt" in ln]
    assert "two engine starts at stage 4" in proc.stdout, "no SGLang on this profile (review 2026-11-01 LOW 5)"
    extras = world["exp_root"] / "S1" / DATE / "extras" / "identity_gate"
    assert sorted(p.name for p in extras.iterdir()) == ["identity_gate.json", "prompts.json", "vllm_off.json", "vllm_on.json"]


def test_an_identity_fail_stops_stage_4_before_stage_5_bills_until_the_owner_accepts_dec_d7(world: Dict[str, Path], tmp_path: Path) -> None:
    """S2 block D DEC-D7 (a) (block C review MEDIUM 3): a FAIL is recorded as a
    result at stage 4 and the stage fails right after, so stage 5 never bills
    before the owner's written acceptance (DEC-C7 let it calibrate first)."""
    _gated_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_IDENTITY_VERDICT="FAIL")
    assert proc.returncode == 1, proc.stdout[-3000:]
    st = _state(world)
    assert st["stages"]["validate"]["status"] == "failed" and st["stages"]["validate"]["identity_gate"] == "FAIL"
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists(), "DEC-D7: stage 5 never bills on an unaccepted FAIL"
    assert not (world["home"] / ".cage_jobs" / "plan.cmd").exists()
    assert "IDENTITY_GATE FAIL" in proc.stdout and "IDENTITY_FAIL_ACCEPTED" in proc.stdout and "stage validate FAILED" in proc.stdout
    extras = world["exp_root"] / "S1" / DATE / "extras" / "identity_gate"
    assert json.loads((extras / "identity_gate.json").read_text(encoding="utf-8"))["verdict"] == "FAIL", "the verdict is on the Mac"


def test_the_rerun_under_the_acceptance_skips_the_gate_job_and_plans_with_both_flags_dec_d7(world: Dict[str, Path], tmp_path: Path) -> None:
    _gated_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_IDENTITY_VERDICT="FAIL")
    assert proc.returncode == 1, proc.stdout[-3000:]
    # the owner's acceptance in the profile: the resumed master re-records the fetched
    # verdict, skips the gate job, calibrates, and plans the rest with both flags
    _profile_with(world, "S1_gated_accept.env", "IDENTITY_FAIL_ACCEPTED=1")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_IDENTITY_VERDICT="FAIL")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert sum(1 for c in _calls(world) if c.startswith("identity_gate verdict")) == 1, "the gate job is not rerun once accepted"
    assert "not rerun" in proc.stdout
    st = _state(world)
    assert st["stages"]["validate"]["identity_gate"] == "FAIL" and st["stages"]["calibrate"]["status"] == "passed"
    assert " --identity-gate logs/identity_gate.json --accept-identity-fail" in _job_cmd(world, "plan")
    assert st["stages"]["plan"]["status"] == "passed"
    # without the acceptance a rerun of stage 4 runs the gate again (the owner may have changed the pin)
    _profile_with(world, "S1_gated_unaccept.env", "IDENTITY_FAIL_ACCEPTED=0")
    proc = _master(world, "--only", "validate", "--redo", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_IDENTITY_VERDICT="PASS")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert sum(1 for c in _calls(world) if c.startswith("identity_gate verdict")) == 2
    assert _state(world)["stages"]["validate"]["identity_gate"] == "PASS"


def test_an_error_verdict_fails_stage_4_before_anything_bills_at_stage_5(world: Dict[str, Path], tmp_path: Path) -> None:
    _gated_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_IDENTITY_VERDICT="ERROR")
    assert proc.returncode == 1 and "stage validate FAILED" in proc.stdout, proc.stdout[-3000:]
    assert "identity_gate" not in _state(world)["stages"]["validate"]
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists(), "stage 5 must not run"
    log = (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "identity_gate.log").read_text(encoding="utf-8")
    assert "IDENTITY_GATE ERROR" in log


def test_a_run_phase_failure_fetches_the_gate_record_before_failing_stage_4_review_l1(world: Dict[str, Path], tmp_path: Path) -> None:
    """A phase that ends in an error leaves prompts.json and the phase files on
    the pod with the reason and every request recorded up to it: they are
    fetched before the stage fails, not only the job log tail."""
    _gated_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_IDENTITY_RUN_FAIL="vllm_off")
    assert proc.returncode == 1 and "stage validate FAILED" in proc.stdout, proc.stdout[-3000:]
    assert "the identity gate job ended before a verdict" in proc.stdout
    extras = world["exp_root"] / "S1" / DATE / "extras" / "identity_gate"
    assert sorted(p.name for p in extras.iterdir()) == ["prompts.json", "vllm_off.json", "vllm_on.json"], "the record of the failed job"
    assert json.loads((extras / "vllm_off.json").read_text(encoding="utf-8"))["error"] == "HTTP 500 at prompt 3"
    assert "identity_gate" not in _state(world)["stages"]["validate"]
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists(), "stage 5 must not run"


def test_a_failed_redo_of_stage_4_wipes_the_stale_verdict_so_the_plan_refuses_review_l2(world: Dict[str, Path], tmp_path: Path) -> None:
    """Run 1 ends PASS. A --redo of stage 4 that dies inside a run phase never
    reaches the verdict step: `state begin` opens the stage with a fresh record,
    so no verdict key survives, and --from plan --redo (which applies no
    predecessor rule) refuses instead of planning on run 1's verdict (review
    2026-11-01 LOW 2: the scenario it names cannot occur; this pins why)."""
    _gated_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_IDENTITY_GATED="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert _state(world)["stages"]["validate"]["identity_gate"] == "PASS"
    proc = _master(world, "--only", "validate", "--redo", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_IDENTITY_RUN_FAIL="vllm_on")
    assert proc.returncode == 1 and "stage validate FAILED" in proc.stdout, proc.stdout[-3000:]
    assert "identity_gate" not in _state(world)["stages"]["validate"], "state begin resets the stage record"
    proc = _master(world, "--from", "plan", "--to", "plan", "--redo", CAGE_TEST_IDENTITY_GATED="1")
    assert proc.returncode == 1 and "recorded no identity verdict" in proc.stdout, proc.stdout[-3000:]
    assert _state(world)["stages"]["plan"]["status"] == "failed"


def test_an_ungated_profile_runs_no_identity_gate_and_passes_no_flag(world: Dict[str, Path], tmp_path: Path) -> None:
    _gated_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "IDENTITY_GATED=0" in proc.stdout and _state(world)["registration"]["identity_gated"] in ("0", 0)
    assert "validate 110," in proc.stdout
    assert "identity_gate" not in _job_names(world) and not (world["home"] / ".cage_jobs" / "identity_gate.cmd").exists()
    assert "--identity-gate" not in _job_cmd(world, "plan") and "identity_gate" not in _state(world)["stages"]["validate"]
    assert not (world["exp_root"] / "S1" / DATE / "extras" / "identity_gate").exists()


def test_a_gated_session_without_the_gate_manifest_or_the_revision_pin_refuses_at_stage_0(world: Dict[str, Path], tmp_path: Path) -> None:
    # the world profile registers squad_v2 only and pins no served-model revision
    _two_engine_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_IDENTITY_GATED="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "IDENTITY_GATE_DATASET=musique" in proc.stdout and "MODEL_REVISION" in proc.stdout
    assert not any("provision_pod" in c and "--yes" in c for c in _calls(world)), "nothing bills on a refused stage 0"



# ---------------------------------------------------------------------------
# S2 block D (MyDocs/RunPod/S2_BLOCK_D_SPEC_2026-11-01.md, ADR-0182): the PD
# preflight smoke. Stage 0 records whether the session plans pd cells and the
# control-rung PD stack env (DEC-D3); stage 4 starts that stack, runs
# scripts/checks/pd_smoke.py and stops it, after the identity gate and before
# the start check; PASS and FAIL are results the plan reads (DEC-D6), an
# unaccepted FAIL stops the stage (DEC-D7), ERROR fails it; the run consents
# to the pd cells (--allow-pd) only on PASS (DEC-D2).
# ---------------------------------------------------------------------------


def _pd_profile(world: Dict[str, Path], tmp_path: Path) -> None:
    # the smoke adds 40 min to the validate bound: the world's 12 h seatbelt no longer covers it
    p = tmp_path / "S1_pd.env"
    p.write_text(world["profile"].read_text(encoding="utf-8").replace("SEATBELT=12h", "SEATBELT=14h"), encoding="utf-8")
    world["profile"] = p


def test_stage_4_runs_the_pd_smoke_after_the_identity_gate_and_the_plan_reads_it_s2_d3(world: Dict[str, Path], tmp_path: Path) -> None:
    _gated_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_LARGEST_POOL="1", CAGE_TEST_PD_CELLS="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    st = _state(world)
    # stage 0 prints and records the pd cells like IDENTITY_GATED, with the stack env, and the stage 4 bound grows by the smoke bound
    assert "PD_CELLS=1" in proc.stdout and st["registration"]["pd_cells"] in ("1", 1)
    assert st["registration"]["pd_env"] == ";".join(f"{k}={v}" for k, v in sorted(_PD_ENV.items()))
    assert "validate 210," in proc.stdout, [ln for ln in proc.stdout.splitlines() if "seatbelt" in ln]
    assert "PD smoke (S2 block D)" in proc.stdout
    assert _job_names(world) == ["setup", "load_gate", "datasets_offline", "validate_vllm", "validate_sglang", "identity_gate",
                                 "pd_smoke", "start_check", "calibrate_vllm", "calibrate_rungs_vllm", "plan"]
    cmd = _job_cmd(world, "pd_smoke")
    pr = world["pod_repo"]
    assert cmd.startswith(f"cd {pr} && export PATH={pr}/cage-env/bin:$PATH VLLM_START_TIMEOUT=600"), cmd[:200]
    # the stack starts at the recorded env, the smoke runs against it, the stack stops, the GPUs are checked free
    start = f"env {_PD_ENV_SORTED} bash scripts/2_serving/manage_vllm_pd.sh start 'Qwen/Qwen3-14B' && "
    assert start in cmd, cmd
    smoke = "cage-env/bin/python scripts/checks/pd_smoke.py --session a --model 'Qwen/Qwen3-14B' --out logs/pd_smoke.json"
    assert smoke in cmd and cmd.count("bash scripts/2_serving/manage_vllm_pd.sh stop") == 1
    assert cmd.index(start) < cmd.index(smoke) < cmd.index("manage_vllm_pd.sh stop")
    assert cmd.rstrip().endswith("echo COMPUTE_APPS=$n; exit $rc")
    assert "launcher pd start Qwen/Qwen3-14B pins=0,1/2,3 tp=2 budgets=1325670400/1325670400" in _calls(world)
    assert "launcher pd stop pins=/ tp= budgets=/" in _calls(world)
    # the verdict is pulled beside the landing's extras, recorded, and the plan reads it after the identity verdict
    extras = world["exp_root"] / "S1" / DATE / "extras" / "pd_smoke"
    assert json.loads((extras / "pd_smoke.json").read_text(encoding="utf-8"))["verdict"] == "PASS"
    assert st["stages"]["validate"]["pd_smoke"] == "PASS" and "PD_SMOKE PASS" in proc.stdout
    plan_cmd = _job_cmd(world, "plan")
    assert " --identity-gate logs/identity_gate.json --pd-smoke logs/pd_smoke.json" in plan_cmd and "--accept-pd-fail" not in plan_cmd


def test_the_run_consents_to_the_pd_cells_on_a_smoke_pass_s2_d2(world: Dict[str, Path], tmp_path: Path) -> None:
    _pd_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "run", CAGE_TEST_PD_CELLS="1")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "validate 125," in proc.stdout, [ln for ln in proc.stdout.splitlines() if "seatbelt" in ln]
    assert "pd_smoke" in _job_names(world) and _state(world)["stages"]["validate"]["pd_smoke"] == "PASS"
    assert " --pd-smoke logs/pd_smoke.json" in _job_cmd(world, "plan") and "--identity-gate" not in _job_cmd(world, "plan")
    log = (world["exp_root"] / "S1" / DATE / "logs" / "runner" / "run.log").read_text(encoding="utf-8")
    assert "run argv:" in log and " --allow-pd" in log
    # the run job's env unsets the role pins beside the other per-step launcher variables
    run_cmd = _job_cmd(world, "run")
    assert "-u CAGE_PD_PREFILL_GPUS -u CAGE_PD_DECODE_GPUS" in run_cmd


def test_a_pd_smoke_fail_stops_stage_4_until_the_owner_accepts_and_withholds_the_run_consent_s2_d6(world: Dict[str, Path], tmp_path: Path) -> None:
    _pd_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "run", CAGE_TEST_PD_CELLS="1", CAGE_TEST_PD_SMOKE_VERDICT="FAIL")
    assert proc.returncode == 1, proc.stdout[-3000:]
    st = _state(world)
    assert st["stages"]["validate"]["status"] == "failed" and st["stages"]["validate"]["pd_smoke"] == "FAIL"
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists(), "stage 5 never bills before the acceptance"
    assert "PD_SMOKE FAIL" in proc.stdout and "PD_FAIL_ACCEPTED" in proc.stdout and "stage validate FAILED" in proc.stdout
    extras = world["exp_root"] / "S1" / DATE / "extras" / "pd_smoke"
    assert json.loads((extras / "pd_smoke.json").read_text(encoding="utf-8"))["verdict"] == "FAIL"
    # the owner's acceptance: the rerun skips the smoke job, plans with --accept-pd-fail, runs without --allow-pd
    _profile_with(world, "S1_pd_accept.env", "PD_FAIL_ACCEPTED=1")
    proc = _master(world, "--yes", "provision", "--to", "run", CAGE_TEST_PD_CELLS="1", CAGE_TEST_PD_SMOKE_VERDICT="FAIL")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert sum(1 for c in _calls(world) if c.startswith("pd_smoke ")) == 1, "the smoke job is not rerun once accepted (DEC-D7)"
    assert "not rerun" in proc.stdout
    assert _state(world)["stages"]["validate"]["pd_smoke"] == "FAIL"
    assert " --pd-smoke logs/pd_smoke.json --accept-pd-fail" in _job_cmd(world, "plan")
    log = (world["exp_root"] / "S1" / DATE / "logs" / "runner" / "run.log").read_text(encoding="utf-8")
    assert "run argv:" in log and "--allow-pd" not in log


def test_a_pd_smoke_error_fails_stage_4_before_anything_bills_at_stage_5(world: Dict[str, Path], tmp_path: Path) -> None:
    _pd_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PD_CELLS="1", CAGE_TEST_PD_SMOKE_VERDICT="ERROR")
    assert proc.returncode == 1 and "stage validate FAILED" in proc.stdout, proc.stdout[-3000:]
    assert "pd_smoke" not in _state(world)["stages"]["validate"]
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists(), "stage 5 must not run"
    log = (world["exp_root"] / "S1" / DATE / "logs" / "setup" / "pd_smoke.log").read_text(encoding="utf-8")
    assert "PD_SMOKE ERROR" in log
    assert "launcher pd stop pins=/ tp= budgets=/" in _calls(world), "the stack is stopped whatever the verdict"


def test_a_pd_stack_that_does_not_come_up_fails_stage_4_before_any_request(world: Dict[str, Path], tmp_path: Path) -> None:
    _pd_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PD_CELLS="1", CAGE_TEST_PD_START_RC="1")
    assert proc.returncode == 1 and "the PD smoke job ended before a verdict" in proc.stdout, proc.stdout[-3000:]
    assert "pd_smoke" not in _state(world)["stages"]["validate"]
    assert not any(c.startswith("pd_smoke ") for c in _calls(world)), "no request is sent to a stack that did not start"
    assert "launcher pd stop pins=/ tp= budgets=/" in _calls(world)
    assert not (world["home"] / ".cage_jobs" / "calibrate_vllm.cmd").exists()


def test_a_session_without_pd_cells_runs_no_smoke_and_passes_no_flag(world: Dict[str, Path], tmp_path: Path) -> None:
    proc = _master(world, "--yes", "provision", "--to", "run")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "PD_CELLS=0" in proc.stdout and _state(world)["registration"]["pd_cells"] in ("0", 0)
    assert "pd_smoke" not in _job_names(world) and "pd_smoke" not in _state(world)["stages"]["validate"]
    assert "--pd-smoke" not in _job_cmd(world, "plan")
    log = (world["exp_root"] / "S1" / DATE / "logs" / "runner" / "run.log").read_text(encoding="utf-8")
    assert "run argv:" in log and "--allow-pd" not in log
    assert not (world["exp_root"] / "S1" / DATE / "extras" / "pd_smoke").exists()


def test_stage_0_refuses_pd_cells_without_a_recorded_stack_env(world: Dict[str, Path], tmp_path: Path) -> None:
    proc = _master(world)
    assert proc.returncode == 10, proc.stdout[-2500:]
    proc = _master(world, "--only", "preflight-mac", "--redo", CAGE_TEST_PD_CELLS="1", CAGE_TEST_PD_ENV="0")
    assert proc.returncode == 1, proc.stdout[-2500:]
    assert "no PD_ENV line" in proc.stdout
    assert not any("provision_pod" in c and "--yes" in c for c in _calls(world)), "nothing bills on a refused stage 0"


# review 2026-11-01 (fresh Fable 5.1) MEDIUM 1 and MEDIUM 4: the job removes the
# previous attempt's pod-side verdict before the start, so a gate or smoke that
# dies without writing leaves nothing to fetch; and the accepted-FAIL skip puts
# the Mac copy of the verdict back on the pod, so a landing resumed on a new pod
# still plans against it.


def test_a_stale_pod_verdict_is_not_judged_after_a_smoke_crash(world: Dict[str, Path], tmp_path: Path) -> None:
    _pd_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_PD_CELLS="1")
    assert proc.returncode == 0 and _state(world)["stages"]["validate"]["pd_smoke"] == "PASS", proc.stdout[-2000:]
    pod_verdict = world["pod_repo"] / "logs" / "pd_smoke.json"
    assert pod_verdict.exists(), "attempt 1 left its verdict on the pod"
    # attempt 2 dies before writing: attempt 1's PASS must not be fetched and recorded
    proc = _master(world, "--only", "validate", "--redo", CAGE_TEST_PD_CELLS="1", CAGE_TEST_PD_SMOKE_NO_WRITE="1")
    assert proc.returncode == 1 and "stage validate FAILED" in proc.stdout, proc.stdout[-3000:]
    assert "pd_smoke" not in _state(world)["stages"]["validate"]
    assert not pod_verdict.exists(), "the job removed the stale verdict before the start"
    assert "launcher pd stop pins=/ tp= budgets=/" in _calls(world)


def test_a_stale_pod_identity_verdict_is_not_judged_after_a_gate_crash(world: Dict[str, Path], tmp_path: Path) -> None:
    _gated_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "validate", CAGE_TEST_IDENTITY_GATED="1")
    assert proc.returncode == 0 and _state(world)["stages"]["validate"]["identity_gate"] == "PASS", proc.stdout[-2000:]
    pod_verdict = world["pod_repo"] / "logs" / "identity_gate.json"
    assert pod_verdict.exists()
    proc = _master(world, "--only", "validate", "--redo", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_IDENTITY_NO_WRITE="1")
    assert proc.returncode == 1 and "stage validate FAILED" in proc.stdout, proc.stdout[-3000:]
    assert "identity_gate" not in _state(world)["stages"]["validate"]
    assert not pod_verdict.exists(), "the job removed the stale verdict before the prompts step"


def test_the_accepted_pd_fail_skip_restores_the_verdict_on_the_pod(world: Dict[str, Path], tmp_path: Path) -> None:
    _pd_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "run", CAGE_TEST_PD_CELLS="1", CAGE_TEST_PD_SMOKE_VERDICT="FAIL")
    assert proc.returncode == 1, proc.stdout[-2000:]
    pod_verdict = world["pod_repo"] / "logs" / "pd_smoke.json"
    pod_verdict.unlink()  # the owner deleted the pod while deciding; the resumed landing provisions a new one
    _profile_with(world, "S1_pd_accept.env", "PD_FAIL_ACCEPTED=1")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PD_CELLS="1", CAGE_TEST_PD_SMOKE_VERDICT="FAIL")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert sum(1 for c in _calls(world) if c.startswith("pd_smoke ")) == 1, "the smoke is still not rerun"
    assert json.loads(pod_verdict.read_text(encoding="utf-8"))["verdict"] == "FAIL", "the Mac copy is back where the plan job reads it"
    assert " --pd-smoke logs/pd_smoke.json --accept-pd-fail" in _job_cmd(world, "plan")


def test_the_accepted_identity_fail_skip_restores_the_verdict_on_the_pod(world: Dict[str, Path], tmp_path: Path) -> None:
    _gated_profile(world, tmp_path)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_IDENTITY_VERDICT="FAIL")
    assert proc.returncode == 1, proc.stdout[-2000:]
    pod_verdict = world["pod_repo"] / "logs" / "identity_gate.json"
    pod_verdict.unlink()
    _profile_with(world, "S1_gated_accept.env", "IDENTITY_FAIL_ACCEPTED=1")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_IDENTITY_GATED="1", CAGE_TEST_IDENTITY_VERDICT="FAIL")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert sum(1 for c in _calls(world) if c.startswith("identity_gate verdict")) == 1
    assert json.loads(pod_verdict.read_text(encoding="utf-8"))["verdict"] == "FAIL"
    assert " --identity-gate logs/identity_gate.json --accept-identity-fail" in _job_cmd(world, "plan")



def test_a_redo_keeps_every_attempts_job_log_on_the_pod_and_on_the_mac(world: Dict[str, Path]) -> None:
    """Block F row F-2 (S2 register T3, S0F-98): on the S1 resume the second
    attempt's submit truncated the first attempt's remote log and its fetch
    overwrote the Mac copy (run.log, 84,218 lines). Every attempt's log now
    survives on both sides under <name>_<submitted_at>.log; <name>.log stays
    the latest attempt for every reader."""
    proc = _master(world, "--yes", "provision", "--to", "setup")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    land = world["exp_root"] / "S1" / DATE
    logdir = land / "logs" / "setup"
    first = (logdir / "setup.log").read_text(encoding="utf-8")
    assert first
    handle_path = land / "extras" / "jobs" / "setup.json"
    stamp1 = json.loads(handle_path.read_text(encoding="utf-8"))["submitted_at"].replace(":", "").replace("-", "")
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp1)
    # the Mac keeps attempt 1 under its own stamp as soon as it is fetched
    assert (logdir / f"setup_{stamp1}.log").read_text(encoding="utf-8") == first
    # attempt 2 (a deliberate --redo) carries a distinct submit instant
    h = json.loads(handle_path.read_text(encoding="utf-8"))
    h["submitted_at"] = "2026-10-26T00:41:00Z"
    handle_path.write_text(json.dumps(h), encoding="utf-8")
    proc = _master(world, "--from", "setup", "--redo", "--only", "setup")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    stamp2 = json.loads(handle_path.read_text(encoding="utf-8"))["submitted_at"].replace(":", "").replace("-", "")
    assert stamp2 != "20261026T004100Z"
    remote = world["home"] / ".cage_jobs"
    # the pod kept attempt 1's log beside the fresh one, stamped with the handle it read
    assert (remote / "setup_20261026T004100Z.log").read_text(encoding="utf-8") == first
    assert (remote / "setup.log").is_file()
    # the Mac holds attempt 1 (its first stamp) and attempt 2 (its own), and the latest copy
    assert (logdir / f"setup_{stamp1}.log").read_text(encoding="utf-8") == first
    assert (logdir / f"setup_{stamp2}.log").read_text(encoding="utf-8") == (logdir / "setup.log").read_text(encoding="utf-8")
    assert (logdir / "setup.log").read_text(encoding="utf-8") == (remote / "setup.log").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Block F row F-9 (S2 register decision 7): the pre-authorized teardown. The
# owner's words go to extras/go_teardown through --authorize teardown "<words>";
# stage 13 honors the file only after the pull gate passed; the GO record
# carries the file's line; teardown_pod.sh is unchanged and no path deletes
# the network volume.
# ---------------------------------------------------------------------------


def _go_file(w: Dict[str, Path]) -> Path:
    return w["exp_root"] / "S1" / DATE / "extras" / "go_teardown"


def test_teardown_without_yes_or_pre_authorization_still_stops_before_it(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", CAGE_TEST_VOLUMES='[{"id": "vol123"}]')
    assert proc.returncode == 10, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "stopped before teardown (no --yes teardown)" in proc.stdout
    assert not _go_file(world).exists()
    calls = _calls(world)
    assert not any("teardown_pod pod123" in c for c in calls)
    assert not any("network-volume delete" in c for c in calls)
    st = _state(world)
    assert st["stages"]["pull"]["status"] == "passed" and "teardown" not in st["stages"]
    assert not any(g["stage"] == "teardown" for g in st["go"])


def test_the_authorize_verb_writes_the_go_file_and_runs_nothing(world: Dict[str, Path]) -> None:
    import re

    # no landing yet: refused, nothing created
    proc = _master(world, "--authorize", "teardown", "delete the pod once the pull is verified")
    assert proc.returncode == 1 and "no landing" in proc.stderr
    assert not (world["exp_root"] / "S1").exists()
    proc = _master(world, "--to", "preflight-mac")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-1000:]
    n_calls = len(_calls(world))
    proc = _master(world, "--authorize", "teardown", "delete the pod once the pull is verified")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    assert "teardown pre-authorized" in proc.stdout
    line = _go_file(world).read_text(encoding="utf-8")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z delete the pod once the pull is verified\n", line)
    assert len(_calls(world)) == n_calls                                      # no fake was called
    assert not any(g["stage"] == "teardown" for g in _state(world)["go"])    # the GO is recorded at stage 13
    # the words are the GO: empty words are refused; only teardown can be pre-authorized
    proc = _master(world, "--authorize", "teardown", "   ")
    assert proc.returncode == 1 and "words are empty" in proc.stderr
    proc = _master(world, "--authorize", "provision", "go")
    assert proc.returncode == 1 and "only teardown" in proc.stderr
    assert _go_file(world).read_text(encoding="utf-8") == line               # the refusals left the file alone


def test_a_pre_authorized_teardown_runs_after_the_pull_gate_without_a_prompt_and_keeps_the_volume(world: Dict[str, Path]) -> None:
    assert _master(world, "--to", "preflight-mac").returncode == 0
    assert _master(world, "--authorize", "teardown", "go: delete the pod after the verified pull").returncode == 0
    proc = _master(world, "--from", "provision", "--yes", "provision", CAGE_TEST_VOLUMES='[{"id": "vol123"}]')
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-1500:]
    assert "stopped before teardown" not in proc.stdout
    assert "pre-authorized" in proc.stdout and "[4/5] deleting pod pod123" in proc.stdout
    calls = _calls(world)
    assert any("teardown_pod pod123" in c and "ASSUME_YES=1" in c for c in calls)
    assert not any("network-volume delete" in c for c in calls)
    st = _state(world)
    assert st["stages"]["pull"]["status"] == "passed" and st["stages"]["teardown"]["status"] == "passed"
    assert st["stages"]["analyze"]["status"] == "passed"
    (go,) = [g for g in st["go"] if g["stage"] == "teardown"]
    line = _go_file(world).read_text(encoding="utf-8").strip()
    assert line in go["words"] and "go: delete the pod after the verified pull" in go["words"]
    assert "--yes teardown" not in go["words"]
    assert st["pod"]["volume_kept"] == "vol123" and st["cost"]["true_zero_utc"]


def test_a_pre_authorized_teardown_waits_for_the_pull_gate(world: Dict[str, Path]) -> None:
    assert _master(world, "--to", "preflight-mac").returncode == 0
    assert _master(world, "--authorize", "teardown", "go").returncode == 0
    # the file alone opens nothing: with the pull not passed the master stops as without it
    proc = _master(world, "--from", "teardown")
    assert proc.returncode == 10, proc.stdout[-3000:] + proc.stderr[-1000:]
    assert "the pull gate has not passed" in proc.stdout and "stopped before teardown" in proc.stdout
    calls = _calls(world)
    assert not any("teardown_pod" in c for c in calls)
    assert not any("network-volume delete" in c for c in calls)
    assert "teardown" not in _state(world)["stages"]
    # --plan with the file present still runs nothing
    proc = _master(world, "--plan")
    assert proc.returncode == 0 and not any("teardown_pod" in c for c in _calls(world))


def test_a_pre_authorized_teardown_is_not_honored_on_a_pod_provisioned_after_the_pull(world: Dict[str, Path]) -> None:
    """Review 2026-11-02 HIGH 1 (block F row F-9): the pull gate must belong to
    the LIVE pod. After an honored teardown the file stays; a new pod on the
    same landing (provision redone, nothing pulled from it) must not be deleted
    on the old pull: the pull's end instant has to be later than the provision's."""
    assert _master(world, "--to", "preflight-mac").returncode == 0
    assert _master(world, "--authorize", "teardown", "go: delete the pod after the verified pull").returncode == 0
    proc = _master(world, "--from", "provision", "--yes", "provision", CAGE_TEST_VOLUMES='[{"id": "vol123"}]')
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    assert sum("teardown_pod pod123" in c for c in _calls(world)) == 1
    # a new pod on the same landing: provision, ship and setup redone, nothing pulled from it
    proc = _master(world, "--from", "provision", "--yes", "provision", "--redo", "--to", "setup", CAGE_TEST_VOLUMES='[{"id": "vol123"}]')
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    st = _state(world)["stages"]
    assert st["pull"]["status"] == "passed" and st["pull"]["ended_utc"] < st["provision"]["ended_utc"], (st["pull"], st["provision"])
    proc = _master(world, "--from", "teardown", "--redo")
    assert proc.returncode == 10, proc.stdout[-2500:] + proc.stderr[-1000:]
    assert "stopped before teardown (no --yes teardown)" in proc.stdout
    assert "the pull gate has not passed for the live pod" in proc.stdout
    assert st["pull"]["ended_utc"] in proc.stdout and st["provision"]["ended_utc"] in proc.stdout
    assert sum("teardown_pod pod123" in c for c in _calls(world)) == 1
    assert _state(world)["stages"]["teardown"]["status"] == "passed"   # the first, honored attempt's record is untouched


def test_the_run_jobs_log_is_kept_under_its_submit_stamp_on_the_mac(world: Dict[str, Path]) -> None:
    """Review 2026-11-02 MEDIUM 1 (block F row F-2): the run job is fetched by
    the run stage itself, not by job_run, so the log the row was written for
    (run.log) must get the same stamped copy on the Mac."""
    proc = _master(world, "--yes", "provision", "--to", "run")
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]
    land = world["exp_root"] / "S1" / DATE
    stamp = json.loads((land / "extras" / "jobs" / "run.json").read_text(encoding="utf-8"))["submitted_at"].replace(":", "").replace("-", "")
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp)
    latest = (land / "logs" / "runner" / "run.log").read_text(encoding="utf-8")
    assert latest
    assert (land / "logs" / "runner" / f"run_{stamp}.log").read_text(encoding="utf-8") == latest


def test_the_authorize_verb_is_refused_under_plan(world: Dict[str, Path]) -> None:
    """Review 2026-11-02 LOW 1 (block F row F-9): the plan banner promises that
    nothing is created, so --plan --authorize writes no file."""
    assert _master(world, "--to", "preflight-mac").returncode == 0
    n_calls = len(_calls(world))
    proc = _master(world, "--plan", "--authorize", "teardown", "go")
    assert proc.returncode == 1, proc.stdout[-1500:] + proc.stderr[-1000:]
    assert "refused under --plan" in proc.stderr
    assert not _go_file(world).exists()
    assert len(_calls(world)) == n_calls   # the refusal ran nothing (the preflight's calls precede it)


def test_no_teardown_path_deletes_the_network_volume_static_contract() -> None:
    """Review 2026-11-02 LOW 4 (block F row F-9): the fake teardown of this
    world never calls runpodctl, so the "no volume delete" assertions of the
    F-9 tests cannot fail there; the design is carried by the real scripts'
    text (owner rule 2026-10-22): the pod delete is the one destructive step
    and the volume is only ever listed."""
    root = Path(__file__).resolve().parents[1]
    for rel in ("scripts/runpod/teardown_pod.sh", "scripts/6_experiments/cage_experiment.sh", "scripts/runpod/pod_watchdog.sh"):
        text = (root / rel).read_text(encoding="utf-8")
        assert not re.search(r"network-volumes?\s+(delete|remove|rm)\b", text), rel
        for line in text.splitlines():
            if "network-volume" in line:
                assert "DELETE" not in line and "-X " not in line, (rel, line)
    teardown = (root / "scripts/runpod/teardown_pod.sh").read_text(encoding="utf-8")
    assert 'runpodctl pod delete "$POD_ID"' in teardown
    assert "network-volume list" in teardown


def test_stage_14_runs_the_retrieval_stage_sidebar_after_the_yield_table(world: Dict[str, Path]) -> None:
    """S2 block F row F-7 (2026-11-03): stage 14 writes the retrieval-stage
    sidebar right after the yield table, with the yield table's "0|3" contract
    (the named refusal of a tree without an F1 retrieval window is printed and
    does not fail the stage; rc 2 does)."""
    proc = _master(world, "--plan")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[plan] retrieval stage (F-7)" in proc.stdout
    line = next(ln for ln in proc.stdout.splitlines() if "[plan] retrieval stage (F-7)" in ln)
    assert "expect rc=0|3" in line and "4_analysis/retrieval_stage.py" in line and "--out" in line
    assert proc.stdout.index("yield table (A1)") < proc.stdout.index("retrieval stage (F-7)") < proc.stdout.index("[plan] organize_results")
    assert _master(world, "--yes", "provision", "--yes", "teardown").returncode == 0
    land = world["exp_root"] / "S1" / DATE
    outs = list(land.glob("plots/retrieval_stage_*/retrieval_stage.md"))
    assert len(outs) == 1, outs
    assert _state(world)["stages"]["analyze"]["status"] == "passed"


def test_stage_0_runs_the_regime_gate_projection_under_plan_and_refuses_an_unmeetable_floor(world: Dict[str, Path]) -> None:
    """S2 block F row F-10 (2026-11-03): the projection is pure, so stage 0 runs
    it under --plan too (the dry proof carries the projected windows per gated
    group) and a floor the registered grid cannot meet fails the plan at $0."""
    proc = _master(world, "--plan")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[plan] regime-gate projection (F-10)" in proc.stdout
    assert "regime-gate projection session=a loads='1.0+' floor='' roster=(empty: every group blocks)" in proc.stdout
    proc = _master(world, "--plan", CAGE_TEST_PROJECTION_REFUSE="1")
    assert proc.returncode == 1, proc.stdout[-1500:] + proc.stderr[-800:]
    assert "REFUSED: MIN_IN_REGIME_WINDOWS=6 exceeds the 2 registered windows of" in proc.stdout
    assert "[FAIL] regime-gate projection (F-10)" in proc.stdout and "stage preflight-mac FAILED" in proc.stdout
    proc = _master(world, "--to", "preflight-mac")
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-800:]
    assert "[step] regime-gate projection (F-10)" in proc.stdout
    assert _state(world)["stages"]["preflight-mac"]["status"] == "passed"


def test_stage_14_passes_the_loads_and_the_roster_to_the_gate(world: Dict[str, Path]) -> None:
    """S2 block F row F-10 (2026-11-03): the gate and the projection read
    REGIME_GATE_LOADS and CONFIRMATORY_ARMS from the profile (the defaults keep
    the rule before F-10: loads "1.0+", every group blocking)."""
    proc = _master(world, "--plan")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if "[plan] regime label table (item 18, E12 rule)" in ln)
    assert "regime_gate.py" in line and "--loads 1.0+ --confirmatory-arms ''" in line
    with world["profile"].open("a", encoding="utf-8") as fh:
        fh.write('\nREGIME_GATE_LOADS="0.95 1.05"\nMIN_IN_REGIME_WINDOWS="0.95:2 1.05:3"\nCONFIRMATORY_ARMS="B1 B6"\n')
    proc = _master(world, "--plan")
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-800:]
    line = next(ln for ln in proc.stdout.splitlines() if "[plan] regime label table (item 18, E12 rule)" in ln)
    assert "0.95:2\\ 1.05:3 --loads 0.95\\ 1.05 --confirmatory-arms B1\\ B6" in line
    proj = next(ln for ln in proc.stdout.splitlines() if "regime-gate projection session=a" in ln)
    assert "loads='0.95 1.05' floor='0.95:2 1.05:3' roster=B1 B6" in proj


def test_stage_14_passes_the_retrieval_stage_refusal_and_fails_its_rc_2(world: Dict[str, Path]) -> None:
    """Review L8 (2026-11-03): the "0|3" contract of the F-7 step, driven: rc 3
    (the named refusal) leaves stage 14 passed; rc 2 fails it by name."""
    proc = _master(world, "--yes", "provision", "--yes", "teardown", CAGE_TEST_RETRIEVAL_STAGE_RC="3")
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-800:]
    assert "[retrieval_stage] REFUSED: fake rc 3" in proc.stdout
    assert _state(world)["stages"]["analyze"]["status"] == "passed"
    proc = _master(world, "--only", "analyze", "--redo", CAGE_TEST_RETRIEVAL_STAGE_RC="2")
    assert proc.returncode == 1, proc.stdout[-1500:] + proc.stderr[-800:]
    assert "[FAIL] retrieval stage (F-7): rc=2, expected 0|3" in proc.stdout
    assert _state(world)["stages"]["analyze"]["status"] == "failed"


# ---------------------------------------------------------------------------
# S2 block F row F-5 (2026-11-03): the ladder triples on the Mac before the pod bills
# ---------------------------------------------------------------------------


def test_stage_5_prints_the_ladder_triples_per_engine_on_the_mac_f5(world: Dict[str, Path]) -> None:
    # before the ladder jobs, stage 5 runs ladder-cells on the Mac for every CALIBRATE engine with the
    # stage's own rung list and manifests, in plan mode too, and says the count per engine
    world["profile"].write_text(world["profile"].read_text(encoding="utf-8") + "\nRUNG_ALL_CLASSES=1\n", encoding="utf-8")
    proc = _master(world, "--plan")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "vllm anchor gold-fresh B1 single 4779 1" in proc.stdout
    assert "ladder vllm: engine=vllm triples=4 classes=2 (F-5)" in proc.stdout
    argv = (world["home"] / ".cage_ladder_cells_argv").read_text(encoding="utf-8").splitlines()
    assert argv == ["ladder-cells --session a --engine vllm --rungs 1,0.5 "
                    f"--query-manifest squad_v2={REPO_ROOT}/data/manifests/squad_v2_50x3_seed42.json --all-classes"]


def test_a_ladder_cells_refusal_fails_stage_5_before_the_ladder_jobs_f5(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "calibrate", CAGE_TEST_LADDER_CELLS_RC="2")
    assert proc.returncode == 1, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "REFUSED: fake ladder-cells rc 2 for engine vllm" in proc.stdout
    assert "[FAIL] ladder-cells (F-5) refused for engine vllm" in proc.stdout
    assert not (world["home"] / ".cage_jobs" / "calibrate_rungs_vllm.cmd").exists()
    assert _state(world)["stages"]["calibrate"]["status"] == "failed"


def test_stage_5_plan_leaves_an_absent_manifest_out_of_the_ladder_print_and_says_so_f5(world: Dict[str, Path]) -> None:
    # S2A names a Qasper manifest the Mac does not hold yet (gated tokenizer); stage 0 refuses the
    # live run until it exists, and --plan still prints the rest of the ladder with the absence named
    world["profile"].write_text(world["profile"].read_text(encoding="utf-8") + '\nRUNG_ALL_CLASSES=1\n'
                                'QUERY_MANIFESTS="squad_v2=data/manifests/squad_v2_50x3_seed42.json '
                                'musique=data/manifests/musique_absent_f5_fixture.json"\n', encoding="utf-8")
    proc = _master(world, "--plan")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    note = [ln for ln in proc.stdout.splitlines() if "ladder-cells (F-5): manifest(s) absent on the Mac" in ln]
    assert len(note) == 1 and note[0].endswith(" musique"), proc.stdout[-4000:]
    assert "ladder vllm: engine=vllm triples=4 classes=2 (F-5)" in proc.stdout
    argv = (world["home"] / ".cage_ladder_cells_argv").read_text(encoding="utf-8").splitlines()
    assert argv == ["ladder-cells --session a --engine vllm --rungs 1,0.5 "
                    f"--query-manifest squad_v2={REPO_ROOT}/data/manifests/squad_v2_50x3_seed42.json --all-classes"]


def test_a_live_stage_5_passes_an_absent_manifest_through_to_the_verb_f5(world: Dict[str, Path]) -> None:
    # only --plan leaves an absent manifest out: a live stage 5 reached by a resume hands the
    # profile's full list to the verb, whose refusal fails the stage before the ladder bills
    ok = _master(world, "--yes", "provision", "--to", "validate")
    assert ok.returncode == 0, ok.stdout[-4000:] + ok.stderr[-2000:]
    world["profile"].write_text(world["profile"].read_text(encoding="utf-8") + '\nRUNG_ALL_CLASSES=1\n'
                                'QUERY_MANIFESTS="squad_v2=data/manifests/squad_v2_50x3_seed42.json '
                                'musique=data/manifests/musique_absent_f5_fixture.json"\n', encoding="utf-8")
    proc = _master(world, "--from", "calibrate", "--to", "calibrate")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "absent on the Mac" not in proc.stdout
    argv = (world["home"] / ".cage_ladder_cells_argv").read_text(encoding="utf-8").splitlines()
    assert argv[-1] == ("ladder-cells --session a --engine vllm --rungs 1,0.5 "
                        f"--query-manifest squad_v2={REPO_ROOT}/data/manifests/squad_v2_50x3_seed42.json "
                        f"--query-manifest musique={REPO_ROOT}/data/manifests/musique_absent_f5_fixture.json --all-classes")


def test_stage_5_hands_the_verb_absolute_manifest_paths_from_any_cwd_f5(world: Dict[str, Path], tmp_path: Path) -> None:
    # review 2026-11-03 MEDIUM 1: the verb opens the manifest files and the master never changes
    # directory, so a master launched outside the repo root must still name the Mac's own copies
    world["profile"].write_text(world["profile"].read_text(encoding="utf-8") + "\nRUNG_ALL_CLASSES=1\n", encoding="utf-8")
    proc = subprocess.run(["bash", str(MASTER), "S1", "--profile", str(world["profile"]), "--plan"],
                          capture_output=True, text=True, env=_env(world), timeout=600, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "ladder vllm: engine=vllm triples=4 classes=2 (F-5)" in proc.stdout
    argv = (world["home"] / ".cage_ladder_cells_argv").read_text(encoding="utf-8").splitlines()
    assert argv == ["ladder-cells --session a --engine vllm --rungs 1,0.5 "
                    f"--query-manifest squad_v2={REPO_ROOT}/data/manifests/squad_v2_50x3_seed42.json --all-classes"]


# ---------------------------------------------------------------------------
# S2 block F batch 5 (2026-11-04, review 2026-11-03 LOW 3): the F-5 check runs
# before the floor jobs
# ---------------------------------------------------------------------------


def test_stage_5_runs_the_ladder_check_before_the_floor_jobs_low3(world: Dict[str, Path]) -> None:
    # the check needs only the session, the rungs and the manifests, none of which the floor
    # jobs produce, so its print comes before the first calibrate_<engine> job is submitted
    proc = _master(world, "--yes", "provision", "--to", "calibrate")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    out = proc.stdout
    i_ladder = out.index("ladder vllm: engine=vllm triples=4 classes=2 (F-5)")
    i_floor = out.index("[step] job calibrate_vllm: wait")
    i_rungs = out.index("[step] job calibrate_rungs_vllm: wait")
    assert i_ladder < i_floor < i_rungs


def test_a_ladder_cells_refusal_fails_stage_5_before_the_floor_jobs_low3(world: Dict[str, Path]) -> None:
    proc = _master(world, "--yes", "provision", "--to", "calibrate", CAGE_TEST_LADDER_CELLS_RC="2")
    assert proc.returncode == 1, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "[FAIL] ladder-cells (F-5) refused for engine vllm" in proc.stdout
    jobs = world["home"] / ".cage_jobs"
    assert not (jobs / "calibrate_vllm.cmd").exists(), "the floor job must not bill before the check"
    assert not (jobs / "calibrate_rungs_vllm.cmd").exists()
    assert _state(world)["stages"]["calibrate"]["status"] == "failed"


# ---------------------------------------------------------------------------
# ADR-0190 (owner 2026-11-05, option D): RUN_CUT_BY_BUDGET, the rehearsal cut
# ---------------------------------------------------------------------------


def test_run_cut_by_budget_turns_the_stage_6_projection_refusal_into_a_warning_adr0190(world: Dict[str, Path]) -> None:
    """ADR-0190: a profile that states RUN_CUT_BY_BUDGET=1 with a budget stop below HOURS
    (stage 0 proved it) passes a plan projected past HOURS at stage 6 with a WARN line that
    names the stop, because the budget stop ends the run with sentinels before the bound
    (ADR-0175 item 1); the refusal text of review 2026-10-21 M2 never prints."""
    _profile_with(world, "S1_cut.env", "RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="5400")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "run projection (S0F-88 B-5): 1.50 h against the 1 h bound" in proc.stdout
    assert ("[WARN] the projected run (1.50 h) exceeds HOURS=1; RUN_CUT_BY_BUDGET=1: the budget stop at "
            "RUN_BUDGET_HOURS=0.5 h ends the run with sentinels before the bound") in proc.stdout
    assert "exceeds HOURS=1: the job would be killed" not in proc.stdout
    assert _state(world)["stages"]["plan"]["status"] == "passed"
    # stage 0 said so too, before the pod billed
    assert "RUN_CUT_BY_BUDGET=1 (ADR-0190): stage 6 accepts a plan projected past HOURS=1" in proc.stdout


def test_run_cut_by_budget_needs_a_budget_stop_below_hours_and_the_refusal_stands_without_it_adr0190(
    world: Dict[str, Path],
) -> None:
    """ADR-0190: the key is 1 or empty; 1 needs RUN_BUDGET_HOURS set to a number above 0 and
    below HOURS, refused at stage 0 (under --plan too) before any pod and before the Mac suite
    (review 2026-11-05 LOW 7); HOURS is converted first and named when it is not a number (LOW 3)."""
    marker = "RUN_CUT_BY_BUDGET=1 needs RUN_BUDGET_HOURS set to a number above 0 and below HOURS=1"
    cases = {"no stop": ("RUN_CUT_BY_BUDGET=1", marker),
             "stop equals HOURS": ("RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=1", marker),
             "stop past HOURS": ("RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=2", marker),
             "stop at zero": ("RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0", marker),
             "stop not a number": ("RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=abc", marker),
             "HOURS not a number": ("RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5\nHOURS=abc",
                                    "RUN_CUT_BY_BUDGET=1 needs HOURS set to a number (got abc)"),
             "key not 1": ("RUN_CUT_BY_BUDGET=yes\nRUN_BUDGET_HOURS=0.5", "RUN_CUT_BY_BUDGET=yes: must be 1 or empty")}
    base = world["profile"]
    for case, (extra, text) in cases.items():
        world["profile"] = base  # review 2026-11-05 LOW 6: each case is the base profile plus its own lines
        _profile_with(world, f"S1_cut_{case.replace(' ', '_')}.env", extra)
        assert world["profile"].read_text(encoding="utf-8").count("RUN_CUT_BY_BUDGET=") == 1, case
        for argv in ((), ("--plan",)):
            proc = _master(world, *argv)
            assert proc.returncode == 1, (case, argv, proc.stdout[-2000:])
            assert text in proc.stdout, (case, argv, proc.stdout[-2000:])
            assert not any(c.startswith("provision_pod") or c.startswith("run_tests") for c in _calls(world)), (case, argv)


def test_a_budget_stop_without_the_key_keeps_the_stage_6_refusal_adr0190(world: Dict[str, Path]) -> None:
    """ADR-0190: the budget stop alone (no key) is no exemption; the M2 refusal stands at stage 6."""
    _profile_with(world, "S1_stop_only.env", "RUN_BUDGET_HOURS=0.5")
    proc = _master(world, "--yes", "provision", CAGE_TEST_PROJECTION_S="5400")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "the projected run (1.50 h) exceeds HOURS=1: the job would be killed" in proc.stdout
    assert _state(world)["stages"]["plan"]["status"] == "failed"


def test_run_cut_by_budget_proves_the_first_segment_fits_the_budget_stop_at_stage_6_adr0190(world: Dict[str, Path]) -> None:
    """Review 2026-11-05 MEDIUM 2: the runner fixes the stop epoch at the submit and admits a
    segment (a relaunch and its steps up to the next relaunch) only when its whole table
    projection ends before the stop (run_campaign.py _past_budget), so a first segment past
    RUN_BUDGET_HOURS x 3600 leaves a run that ends at its first relaunch with sentinels and
    measures no cell. Stage 6 projects the first segment with the repo's run_projection and
    refuses when it does not fit, or when the plan carries no relaunch for the stop to check."""
    _profile_with(world, "S1_cut.env", "RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5")
    # the fake run_projection charges CAGE_TEST_STEP_S per step; the fake plan's three steps are one segment
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert ("first budget segment (ADR-0190): steps 0 to 2 (1 relaunch, 0 dry windows, 2 cells) project 1500 s "
            "against the RUN_BUDGET_HOURS=0.5 h stop (1800 s): the runner admits it") in proc.stdout
    proc = _master(world, "--from", "plan", "--to", "plan", "--redo", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="700")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert ("[FAIL] first budget segment (ADR-0190): steps 0 to 2 (1 relaunch, 0 dry windows, 2 cells) project 2100 s "
            "against the RUN_BUDGET_HOURS=0.5 h stop (1800 s): the runner would admit no segment") in proc.stdout
    assert _state(world)["stages"]["plan"]["status"] == "failed"
    proc = _master(world, "--from", "plan", "--to", "plan", "--redo", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_NO_RELAUNCH="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] RUN_CUT_BY_BUDGET=1 but the plan carries no relaunch step: the budget stop is checked before relaunches only" in proc.stdout


def test_the_budget_stop_proof_is_repeated_at_stages_6_and_7_after_a_profile_edit_adr0190(world: Dict[str, Path]) -> None:
    """Review 2026-11-05 LOW 4: stage 0's proof that the stop lies in (0, HOURS) does not survive
    a profile edit under --from; stage 6 repeats it before the plan audit and stage 7 before
    arming the stop, so an edited RUN_BUDGET_HOURS at or past HOURS never reaches the audit
    or arms a stop past the bound."""
    _profile_with(world, "S1_cut.env", "RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="5400")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    _profile_with(world, "S1_cut_edited.env", "RUN_BUDGET_HOURS=5")  # the last statement wins: 5 is past HOURS=1
    marker = "RUN_CUT_BY_BUDGET=1 needs RUN_BUDGET_HOURS set to a number above 0 and below HOURS=1 (got 5)"
    for argv in (("--from", "plan", "--to", "plan", "--redo"), ("--from", "run", "--to", "run", "--redo")):
        proc = _master(world, *argv, CAGE_TEST_PROJECTION_S="5400")
        assert proc.returncode == 1, (argv, proc.stdout[-3000:])
        assert marker in proc.stdout, (argv, proc.stdout[-2000:])
    assert "stop_at_epoch" not in _state(world).get("run", {})
    assert not (world["home"] / ".cage_jobs" / "run.cmd").exists(), "stage 7 must not submit"


def test_a_stray_environment_value_of_the_key_never_reaches_the_master_adr0190(world: Dict[str, Path]) -> None:
    """Review 2026-11-05 LOW 5: _common.env states RUN_CUT_BY_BUDGET= (empty), so a value in the
    environment never reaches the master (the HF_TOKEN_FILE precedent, review 2026-11-01 L2):
    the M2 refusal stands on a profile that states only the stop."""
    _profile_with(world, "S1_stop_only.env", "RUN_BUDGET_HOURS=0.5")
    proc = _master(world, "--yes", "provision", CAGE_TEST_PROJECTION_S="5400", RUN_CUT_BY_BUDGET="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "the projected run (1.50 h) exceeds HOURS=1: the job would be killed" in proc.stdout
    assert "RUN_CUT_BY_BUDGET=1 (ADR-0190)" not in proc.stdout


def test_run_cut_by_budget_tolerates_unmeasured_steps_only_past_the_stop_adr0191(world: Dict[str, Path]) -> None:
    """ADR-0191 (owner 2026-11-05, "(a) now, and (b) later"): cd-act1's pd steps have no row in the
    projection table and stage 6 refuses unmeasured steps. Under RUN_CUT_BY_BUDGET=1 they are
    tolerated only when the table projection of the plan prefix before the relaunch that starts
    the first unmeasured segment exceeds the stop (the runner then never admits that segment,
    unless the pod outruns the table by the printed ratio); otherwise the refusal stands."""
    _profile_with(world, "S1_cut.env", "RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5")
    # seven steps: the measured first segment (3), a measured segment (2), then a pd segment (2) the table cannot
    # project; at 500 s per measured step the pd relaunch at step 5 sits behind a 2,500 s prefix, past the 1,800 s stop
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500", CAGE_TEST_PD_TAIL="1")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert ("[WARN] 2 plan step(s) have no measured duration (e.g. relaunch vllm), the first at step 5; the plan prefix "
            "before its segment (the relaunch at step 5) projects 0.69 h against the RUN_BUDGET_HOURS=0.5 h stop, so the "
            "stop ends the run before any of them unless the prefix finishes under the stop: a pod 1.39 times faster than "
            "the table, or its 2 relaunches failing within about 900 s each (a failed relaunch is skipped with sentinels, "
            "not fatal, with RUN_FAIL_FAST empty), or cells and dry windows of the prefix failing early (each costs its "
            "attempt, not its table row) (RUN_CUT_BY_BUDGET=1, ADR-0191; a landing of record refuses them)") in proc.stdout
    assert "the run time cannot be projected" not in proc.stdout
    assert _state(world)["stages"]["plan"]["status"] == "passed"
    # at 300 s per step the prefix is 1,500 s, inside the stop: the pd segment is reachable, refused
    proc = _master(world, "--from", "plan", "--to", "plan", "--redo", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="300", CAGE_TEST_PD_TAIL="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert ("[FAIL] 2 plan step(s) have no measured duration (e.g. relaunch vllm): the first at step 5 is reachable inside "
            "the RUN_BUDGET_HOURS=0.5 h stop (the plan prefix before its segment projects 0.42 h)") in proc.stdout
    assert _state(world)["stages"]["plan"]["status"] == "failed"


def test_without_the_key_unmeasured_steps_keep_the_refusal_adr0191(world: Dict[str, Path]) -> None:
    """ADR-0191: a landing of record refuses unmeasured steps as before, whatever the budget stop says."""
    _profile_with(world, "S1_stop_only.env", "RUN_BUDGET_HOURS=0.5")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_STEP_S="500", CAGE_TEST_PD_TAIL="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert ("[FAIL] 2 plan step(s) have no measured duration (e.g. relaunch vllm): the run time cannot be projected; "
            "measure them or extend RUN_*_SECONDS with an ADR") in proc.stdout
    assert "ADR-0191" not in proc.stdout


def test_a_pd_cell_without_the_telemetry_endpoints_fails_the_plan_audit_2026_11_05(world: Dict[str, Path]) -> None:
    """S2R stage 6 blocker (2026-11-05): the audit's pd check (81680b0) demanded a CAGE_PD_* key on
    every pd cell, a key the planner writes on the pd relaunch only (block D, ADR-0182); a pd cell
    or dry window carries CAGE_TELEMETRY_ENDPOINTS (PD_TELEMETRY_ENDPOINTS_ENV), so the check failed
    all 85 pd steps of the Mac-built cd-act1 plan and no test covered it. The check asks for the
    key the planner writes: the faithful fake pd cell passes, one without the key is named."""
    _profile_with(world, "S1_cut.env", "RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500",
                   CAGE_TEST_PD_TAIL="1", CAGE_TEST_PD_NO_TELEMETRY="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] pd cell without CAGE_TELEMETRY_ENDPOINTS env: k4" in proc.stdout
    assert "CAGE_PD_*" not in proc.stdout
    assert _state(world)["stages"]["plan"]["status"] == "failed"
    proc = _master(world, "--from", "plan", "--to", "plan", "--redo", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500", CAGE_TEST_PD_TAIL="1")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "pd cell without" not in proc.stdout


def test_a_budget_stop_edited_after_the_plan_audit_is_refused_at_stage_7_adr0191(world: Dict[str, Path]) -> None:
    """Review 2026-11-05 LOW 1 (ADR-0191): the tolerance of unmeasured steps holds for the
    RUN_BUDGET_HOURS the stage 6 audit judged; stage 6 records that value and stage 7 refuses a
    different one (inside the range budget_cut_check proves) until the plan is re-audited."""
    _profile_with(world, "S1_cut.env", "RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert _state(world)["plan"]["audit_budget_hours"] == "0.5"
    _profile_with(world, "S1_cut_edited.env", "RUN_BUDGET_HOURS=0.75")  # the last statement wins; 0.75 is inside (0, HOURS=1)
    proc = _master(world, "--from", "run", "--to", "run", "--redo", CAGE_TEST_PROJECTION_S="5400")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] RUN_BUDGET_HOURS=0.75 differs from the 0.5 h the stage 6 audit judged the plan against (ADR-0191): rerun --from plan" in proc.stdout
    assert "stop_at_epoch" not in _state(world).get("run", {})
    assert not (world["home"] / ".cage_jobs" / "run.cmd").exists(), "stage 7 must not submit"
    proc = _master(world, "--from", "plan", "--to", "plan", "--redo", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500")
    assert proc.returncode == 0, proc.stdout[-4000:]
    assert _state(world)["plan"]["audit_budget_hours"] == "0.75"


def test_the_mac_table_and_the_pod_plan_must_agree_on_the_unmeasured_count_adr0191(world: Dict[str, Path]) -> None:
    """Review 2026-11-05 LOW 2 (ADR-0191): under the key the tolerance is computed on the Mac with
    the Mac's run_campaign.py while the plan carries the pod's list; a count mismatch (the table
    edited on one side) is refused instead of judged on one side's table."""
    _profile_with(world, "S1_cut.env", "RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5")
    # the fake pod plan lists one unmeasured step; the fake Mac table finds none (no pd step in the plan)
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500", CAGE_TEST_UNMEASURED="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert ("[FAIL] the pod's plan lists 1 unmeasured step(s) and the Mac's table 0: the two run_campaign.py differ; "
            "re-plan with one code (ADR-0191)") in proc.stdout
    assert _state(world)["stages"]["plan"]["status"] == "failed"


def test_an_unmeasured_step_inside_the_first_segment_or_before_any_relaunch_is_refused_adr0191(world: Dict[str, Path]) -> None:
    """Review 2026-11-05 LOW 4 (ADR-0191): the three branches the first test does not reach, pinned.
    A pd cell inside the first segment has an empty prefix (0.00 h) and is reachable; one before
    any relaunch (no relaunch at all) draws the no-relaunch refusal and the same reachable line
    with no traceback; a plan with no unmeasured step prints nothing of ADR-0191."""
    _profile_with(world, "S1_cut.env", "RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500", CAGE_TEST_PD_FIRST="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert ("[FAIL] 1 plan step(s) have no measured duration (e.g. cell k0 on topology pd): the first at step 1 is reachable "
            "inside the RUN_BUDGET_HOURS=0.5 h stop (the plan prefix before its segment projects 0.00 h)") in proc.stdout
    proc = _master(world, "--from", "plan", "--to", "plan", "--redo", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500",
                   CAGE_TEST_PD_FIRST="1", CAGE_TEST_NO_RELAUNCH="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] RUN_CUT_BY_BUDGET=1 but the plan carries no relaunch step" in proc.stdout
    assert "the first at step 0 is reachable inside the RUN_BUDGET_HOURS=0.5 h stop (the plan prefix before its segment projects 0.00 h)" in proc.stdout
    assert "Traceback" not in proc.stdout + proc.stderr
    proc = _master(world, "--from", "plan", "--to", "plan", "--redo", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500")
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "first budget segment (ADR-0190): steps 0 to 2" in proc.stdout
    assert "no measured duration" not in proc.stdout and "ADR-0191" not in proc.stdout


def test_a_projection_without_a_list_or_without_a_projection_is_named_not_tracebacked_2026_11_05(world: Dict[str, Path]) -> None:
    """Six-fix review LOW 1 (ADR-0191): the pod-list comparison guards the projection as the rest of
    the audit does. A projection whose unmeasured_steps is null draws a named FAIL and no traceback;
    a plan with no projection draws only the no-projection FAIL, never the mismatch line."""
    _profile_with(world, "S1_cut.env", "RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500",
                   CAGE_TEST_PD_TAIL="1", CAGE_TEST_PD_NULL_LIST="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] the plan's run_projection carries no unmeasured_steps list (S0F-88 B-5): re-plan with the current code" in proc.stdout
    assert "Traceback" not in proc.stdout + proc.stderr
    proc = _master(world, "--from", "plan", "--to", "plan", "--redo", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500",
                   CAGE_TEST_PD_TAIL="1", CAGE_TEST_NO_PROJECTION="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] the plan carries no run_projection (S0F-88 B-5): re-plan with the current code" in proc.stdout
    assert "the two run_campaign.py differ" not in proc.stdout
    assert "Traceback" not in proc.stdout + proc.stderr


def test_the_mac_table_and_the_pod_plan_must_name_the_same_unmeasured_steps_2026_11_05(world: Dict[str, Path]) -> None:
    """Six-fix review LOW 2 (ADR-0191): equal counts with different step names are a mismatch too;
    the first differing entry is named."""
    _profile_with(world, "S1_cut.env", "RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500",
                   CAGE_TEST_PD_TAIL="1", CAGE_TEST_PD_WRONG_NAMES="1")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert ("[FAIL] the pod's plan and the Mac's table list 2 unmeasured step(s) each but differ at entry 0: pod 'x', "
            "Mac 'relaunch vllm': the two run_campaign.py differ; re-plan with one code (ADR-0191)") in proc.stdout
    assert _state(world)["stages"]["plan"]["status"] == "failed"


def test_stage_7_refuses_a_cut_landing_whose_audit_recorded_no_stop_2026_11_05(world: Dict[str, Path]) -> None:
    """Six-fix review LOW 4 (ADR-0191): a state without plan.audit_budget_hours under the switch
    (a stage 6 from before the key, or a failed put) is refused by name at stage 7, outside plan
    mode, with no stop armed and no job submitted."""
    _profile_with(world, "S1_cut.env", "RUN_CUT_BY_BUDGET=1\nRUN_BUDGET_HOURS=0.5")
    proc = _master(world, "--yes", "provision", "--to", "plan", CAGE_TEST_PROJECTION_S="5400", CAGE_TEST_STEP_S="500")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    sp = world["exp_root"] / "S1" / DATE / "extras" / "state.json"
    st = json.loads(sp.read_text(encoding="utf-8"))
    del st["plan"]["audit_budget_hours"]
    sp.write_text(json.dumps(st), encoding="utf-8")
    proc = _master(world, "--from", "run", "--to", "run", "--redo", CAGE_TEST_PROJECTION_S="5400")
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "[FAIL] stage 6 recorded no plan.audit_budget_hours under RUN_CUT_BY_BUDGET=1 (ADR-0191): rerun --from plan" in proc.stdout
    assert "stop_at_epoch" not in _state(world).get("run", {})
    assert not (world["home"] / ".cage_jobs" / "run.cmd").exists(), "stage 7 must not submit"
    proc = _master(world, "--plan", "--from", "run", "--to", "run", CAGE_TEST_PROJECTION_S="5400")
    assert "recorded no plan.audit_budget_hours" not in proc.stdout, "plan mode has no state to read and stays exempt"
