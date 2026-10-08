"""Backlog Tier A item A1 (S0-9 / S0-20): shared-GPU memory-utilization rule
for the multi-replica cluster launcher (scripts/2_serving/manage_vllm_cluster.py).

WHAT is pinned and WHY: the launcher used to hand EVERY replica
--gpu-memory-utilization 0.90 (VLLM_GPU_MEMORY_UTILIZATION or 0.90). vLLM's
startup check requests that fraction of the device unconditionally, so on ONE
shared GPU the second replica is refused at startup and the S0-9 cluster proof
(start -> status -> routed traffic -> stop) cannot even begin. The rule:

1. replicas > 1 without distinct per-replica CUDA_VISIBLE_DEVICES pins
   (--replica-gpus) = SHARED GPU: the per-instance default becomes
   SHARED_GPU_MEM_UTIL (0.45, a named constant citing A1 / S0-9 / S0-20) and
   an explicit VLLM_GPU_MEMORY_UTILIZATION above 0.50 is a typed refusal
   (GpuShareError) naming the vLLM startup check and both fixes.
2. replicas pinned to pairwise-disjoint GPUs (or a single replica) = DISTINCT:
   the 0.90 uniform operating point stands; VLLM_GPU_MEMORY_UTILIZATION still
   overrides it.
3. The decision (mode + value + source) is PRINTED on launch and recorded in
   the cluster state so the S0 evidence shows which regime served.
4. The pin rides CUDA_VISIBLE_DEVICES in each replica's process env and
   nowhere else; the refusal fires BEFORE any process is stopped or started.

No GPU / engine / network: start_cluster is driven with monkeypatched process
and readiness seams.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLUSTER_PY = REPO_ROOT / "scripts" / "2_serving" / "manage_vllm_cluster.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mc = _load("manage_vllm_cluster_tests", CLUSTER_PY)


# ---------------------------------------------------------------------------
# 0. the named constants and their provenance docstrings
# ---------------------------------------------------------------------------


def test_shared_gpu_constants_are_named_and_cite_backlog_rows() -> None:
    assert mc.SHARED_GPU_MEM_UTIL == 0.45
    assert mc.SHARED_GPU_MEM_UTIL_CEILING == 0.50
    assert mc.DISTINCT_GPU_MEM_UTIL == 0.90
    text = CLUSTER_PY.read_text(encoding="utf-8")
    # The constant must be immediately followed by a docstring citing the
    # backlog item and BOTH S0 rows it unblocks (doctrine: every knob names
    # its ADR or backlog item next to its value).
    m = re.search(
        r'^SHARED_GPU_MEM_UTIL\s*(?::\s*float)?\s*=\s*0\.45\s*\n"""(.*?)"""',
        text, re.M | re.S,
    )
    assert m, "SHARED_GPU_MEM_UTIL = 0.45 must carry a docstring"
    doc = m.group(1)
    assert "A1" in doc and "S0-9" in doc and "S0-20" in doc
    # the A1 region (constants through the resolver) carries no em/en dashes
    region = text[text.index("SHARED_GPU_MEM_UTIL"): text.index("def build_serve_args")]
    assert "\u2014" not in region and "\u2013" not in region, "no em/en dashes in the A1 region"


def test_gpu_share_error_is_a_typed_refusal() -> None:
    assert issubclass(mc.GpuShareError, ValueError)


# ---------------------------------------------------------------------------
# 1. parse_replica_gpus: the per-replica CUDA_VISIBLE_DEVICES pin spec
# ---------------------------------------------------------------------------


def test_parse_replica_gpus_none_means_no_pins() -> None:
    assert mc.parse_replica_gpus(None, 3) == (None, None, None)
    assert mc.parse_replica_gpus("", 3) == (None, None, None)
    assert mc.parse_replica_gpus("   ", 2) == (None, None)


def test_parse_replica_gpus_one_gpu_per_replica() -> None:
    assert mc.parse_replica_gpus("0,1,2", 3) == ("0", "1", "2")


def test_parse_replica_gpus_tp_groups_use_plus_and_become_csv_pins() -> None:
    # A replica that spans several GPUs (TP) lists them with '+'; the pin
    # handed to CUDA_VISIBLE_DEVICES is the comma form vLLM expects.
    assert mc.parse_replica_gpus("0+1,2+3", 2) == ("0,1", "2,3")


@pytest.mark.parametrize(
    "spec, count",
    [
        ("0,1", 3),        # too few
        ("0,1,2,3", 3),    # too many
        ("0,0,1", 3),      # duplicate device
        ("0+1,1+2", 2),    # overlapping TP groups
        ("a,b", 2),        # non-numeric
        ("0,,1", 3),       # empty entry
        ("0+,1", 2),       # empty device inside a group
        ("00,0", 2),       # leading zero: one device under two spellings
    ],
    ids=["too-few", "too-many", "duplicate", "overlap", "non-numeric", "empty-entry", "empty-group", "leading-zero"],
)
def test_parse_replica_gpus_refuses_malformed_or_overlapping(spec: str, count: int) -> None:
    with pytest.raises(mc.GpuShareError) as excinfo:
        mc.parse_replica_gpus(spec, count)
    assert "--replica-gpus" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2. resolve_gpu_share: the rule itself
# ---------------------------------------------------------------------------


def test_shared_default_is_0_45(monkeypatch: pytest.MonkeyPatch) -> None:
    d = mc.resolve_gpu_share(replica_count=3, replica_gpus=(None, None, None), env={})
    assert d.mode == "shared"
    assert d.mem_util == "0.45"
    assert d.source == "default"


def test_distinct_pins_keep_0_90_default() -> None:
    d = mc.resolve_gpu_share(replica_count=3, replica_gpus=("0", "1", "2"), env={})
    assert d.mode == "distinct"
    assert d.mem_util == "0.90"
    assert d.source == "default"


def test_single_replica_is_distinct_at_0_90() -> None:
    d = mc.resolve_gpu_share(replica_count=1, replica_gpus=(None,), env={})
    assert d.mode == "distinct"
    assert d.mem_util == "0.90"


def test_shared_explicit_above_ceiling_is_refused_naming_check_and_fixes() -> None:
    with pytest.raises(mc.GpuShareError) as excinfo:
        mc.resolve_gpu_share(
            replica_count=2, replica_gpus=(None, None),
            env={"VLLM_GPU_MEMORY_UTILIZATION": "0.90"},
        )
    msg = str(excinfo.value)
    assert "vLLM startup check" in msg
    assert "--gpu-memory-utilization" in msg
    assert "0.90" in msg and "0.50" in msg
    # both fixes named: lower the value, or pin distinct GPUs
    assert "VLLM_GPU_MEMORY_UTILIZATION" in msg
    assert "--replica-gpus" in msg


def test_shared_explicit_at_or_below_ceiling_is_honored() -> None:
    d = mc.resolve_gpu_share(
        replica_count=2, replica_gpus=(None, None),
        env={"VLLM_GPU_MEMORY_UTILIZATION": "0.40"},
    )
    assert d.mode == "shared" and d.mem_util == "0.40" and d.source == "explicit"
    # the boundary itself is allowed ("above 0.50" refuses, 0.50 does not)
    d = mc.resolve_gpu_share(
        replica_count=2, replica_gpus=(None, None),
        env={"VLLM_GPU_MEMORY_UTILIZATION": "0.50"},
    )
    assert d.mem_util == "0.50"


def test_distinct_explicit_override_is_honored_even_above_ceiling() -> None:
    d = mc.resolve_gpu_share(
        replica_count=2, replica_gpus=("0", "1"),
        env={"VLLM_GPU_MEMORY_UTILIZATION": "0.95"},
    )
    assert d.mode == "distinct" and d.mem_util == "0.95" and d.source == "explicit"


@pytest.mark.parametrize("bad", ["abc", "0", "-0.5", "1.5", "0,9", "nan", "inf"])
def test_malformed_explicit_value_is_refused(bad: str) -> None:
    with pytest.raises(mc.GpuShareError) as excinfo:
        mc.resolve_gpu_share(
            replica_count=2, replica_gpus=("0", "1"),
            env={"VLLM_GPU_MEMORY_UTILIZATION": bad},
        )
    assert "VLLM_GPU_MEMORY_UTILIZATION" in str(excinfo.value)


def test_empty_explicit_value_means_unset_on_both_launchers() -> None:
    # bash cannot tell '' from unset, so the python resolver agrees: default.
    d = mc.resolve_gpu_share(
        replica_count=2, replica_gpus=(None, None),
        env={"VLLM_GPU_MEMORY_UTILIZATION": "  "},
    )
    assert d.mem_util == "0.45" and d.source == "default"


def test_partial_pins_are_shared_never_silently_distinct() -> None:
    # parse_replica_gpus never produces a partial tuple, but the resolver
    # must still treat any unpinned replica as sharing (fail closed).
    d = mc.resolve_gpu_share(replica_count=2, replica_gpus=("0", None), env={})
    assert d.mode == "shared" and d.mem_util == "0.45"


def test_decision_banner_names_mode_value_and_source() -> None:
    d = mc.resolve_gpu_share(replica_count=3, replica_gpus=(None, None, None), env={})
    banner = d.banner()
    assert banner.startswith("[cage] gpu-share decision:")
    assert "shared" in banner and "0.45" in banner and "default" in banner
    assert "SHARED_GPU_MEM_UTIL" in banner
    d2 = mc.resolve_gpu_share(replica_count=2, replica_gpus=("0", "1"), env={})
    assert "distinct" in d2.banner() and "0.90" in d2.banner()


# ---------------------------------------------------------------------------
# 3. build_serve_args carries the RESOLVED value, never a hidden default
# ---------------------------------------------------------------------------


def test_build_serve_args_takes_resolved_mem_util(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
    args = mc.build_serve_args("fake/model", 8001, gpu_memory_utilization="0.45")
    i = args.index("--gpu-memory-utilization")
    assert args[i + 1] == "0.45"
    assert args.count("--gpu-memory-utilization") == 1


def test_build_serve_args_has_no_positional_default_for_mem_util() -> None:
    with pytest.raises(TypeError):
        mc.build_serve_args("fake/model", 8001)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 4. start_cluster wiring: env seam, banner, state, refusal-before-anything
# ---------------------------------------------------------------------------


class _Launches:
    def __init__(self) -> None:
        self.calls: List[Tuple[List[str], Dict[str, str]]] = []
        self._pid = 40000

    def __call__(self, cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
        self.calls.append((list(cmd), dict(env or {})))
        self._pid += 1
        return self._pid


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Launches:
    launches = _Launches()
    monkeypatch.setattr(mc, "LOG_DIR", tmp_path / "cluster")
    monkeypatch.setattr(mc, "STATE_FILE", tmp_path / "cluster" / "cluster_state.json")
    monkeypatch.setattr(mc, "launch_process", launches)
    monkeypatch.setattr(mc, "wait_for", lambda predicate, timeout, label, **kw: 0.0)
    monkeypatch.setattr(mc, "is_pid_running", lambda pid: True)
    monkeypatch.setattr(mc, "replica_ready", lambda api_base, model: True)
    monkeypatch.setattr(mc, "terminate_process_group", lambda *a, **k: None)
    monkeypatch.setattr(mc, "refuse_bound_ports", lambda ports: None)
    monkeypatch.setattr(
        mc, "fetch_router_stats",
        lambda url: {"num_replicas": mc._EXPECTED_REPLICAS, "distinct_api_bases": mc._EXPECTED_REPLICAS},
    )
    monkeypatch.setattr(mc.importlib.util, "find_spec", lambda name: object())
    monkeypatch.delenv("VLLM_GPU_MEMORY_UTILIZATION", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # ADR-0124 (S0F-19): shared mode REQUIRES the per-replica KV pin; the wired
    # starts below carry the S0 pd value so the shared cases keep launching.
    monkeypatch.setenv(mc.KV_BUDGET_ENV, "5713920000")
    monkeypatch.delenv("CAGE_KV_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("CAGE_VLLM_GPU_BLOCKS_OVERRIDE", raising=False)
    return launches


def _start(replicas: int, replica_gpus: Optional[str]) -> int:
    mc._EXPECTED_REPLICAS = replicas
    return mc.start_cluster(
        model="fake/model",
        replica_count=replicas,
        base_port=8001,
        router_port=9000,
        router_strategy="hash",
        replica_timeout=1,
        router_timeout=1,
        replica_gpus=replica_gpus,
    )


def test_start_shared_prints_decision_and_serves_0_45(
    wired: _Launches, capsys: pytest.CaptureFixture[str]
) -> None:
    _start(3, None)
    out = capsys.readouterr().out
    assert "[cage] gpu-share decision: shared" in out
    assert "0.45" in out
    replica_cmds = [cmd for cmd, _ in wired.calls if cmd[:2] == ["vllm", "serve"]]
    assert len(replica_cmds) == 3
    for cmd in replica_cmds:
        assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.45"
    # no pin: CUDA_VISIBLE_DEVICES is NOT injected (ambient visibility rules)
    for cmd, env in wired.calls:
        if cmd[:2] == ["vllm", "serve"]:
            assert "CUDA_VISIBLE_DEVICES" not in env
    state = mc.load_state()
    assert state["gpu_share"]["mode"] == "shared"
    assert state["gpu_share"]["mem_util"] == "0.45"


def test_start_distinct_pins_ride_cuda_visible_devices_and_serve_0_90(
    wired: _Launches, capsys: pytest.CaptureFixture[str]
) -> None:
    _start(2, "0+1,2+3")
    out = capsys.readouterr().out
    assert "[cage] gpu-share decision: distinct" in out
    assert "0.90" in out
    replica_envs = [env for cmd, env in wired.calls if cmd[:2] == ["vllm", "serve"]]
    assert [e["CUDA_VISIBLE_DEVICES"] for e in replica_envs] == ["0,1", "2,3"]
    replica_cmds = [cmd for cmd, _ in wired.calls if cmd[:2] == ["vllm", "serve"]]
    for cmd in replica_cmds:
        assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.90"
    state = mc.load_state()
    assert state["gpu_share"]["mode"] == "distinct"
    assert state["replicas"][0]["cuda_visible_devices"] == "0,1"
    assert state["replicas"][1]["cuda_visible_devices"] == "2,3"
    assert state["replica_gpus"] == "0+1,2+3"


def test_start_refuses_shared_0_90_before_touching_any_process(
    wired: _Launches, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
    stops: List[bool] = []
    monkeypatch.setattr(mc, "stop_cluster", lambda *a, **k: stops.append(True) or 0)
    with pytest.raises(mc.GpuShareError):
        _start(2, None)
    assert wired.calls == [], "refusal must precede every launch"
    assert stops == [], "refusal must precede the self-cleaning teardown"
    assert not mc.STATE_FILE.exists()


def test_start_shared_with_the_lib_defaulted_0_90_serves_0_45(
    wired: _Launches, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """S0F-21 (ADR-0141): scripts/lib/_serving_config.sh exports 0.90 plus the
    marker CAGE_VLLM_MEM_UTIL_DEFAULTED=1 when nothing was set; the manager
    must read that pair as "unset" and serve the shared default, where the
    pre-fix code refused the launch as an explicit 0.90 above the ceiling."""
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
    monkeypatch.setenv(mc.MEM_UTIL_DEFAULTED_ENV, "1")
    _start(3, None)
    out = capsys.readouterr().out
    assert "[cage] gpu-share decision: shared" in out
    assert "S0F-21" in out
    replica_cmds = [cmd for cmd, _ in wired.calls if cmd[:2] == ["vllm", "serve"]]
    assert len(replica_cmds) == 3
    for cmd in replica_cmds:
        assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.45"
    state = mc.load_state()
    assert state["gpu_share"]["mem_util"] == "0.45"
    assert "S0F-21" in state["gpu_share"]["source"]


def test_start_refuses_pin_count_mismatch_before_anything(wired: _Launches) -> None:
    with pytest.raises(mc.GpuShareError):
        _start(3, "0,1")
    assert wired.calls == []


def test_existing_state_reuse_requires_matching_pins(wired: _Launches, capsys: pytest.CaptureFixture[str]) -> None:
    _start(2, "0,1")
    capsys.readouterr()
    # same config -> reuse (no new launches)
    n = len(wired.calls)
    _start(2, "0,1")
    assert len(wired.calls) == n
    assert "already running" in capsys.readouterr().out
    # different pins -> a relaunch, never a silent reuse under other dials
    _start(2, None)
    assert len(wired.calls) > n


# ---------------------------------------------------------------------------
# 5. CLI surface
# ---------------------------------------------------------------------------


def test_parser_exposes_replica_gpus_on_start_and_restart() -> None:
    parser = mc.build_parser()
    for verb in ("start", "restart"):
        ns = parser.parse_args([verb, "--model", "m", "--replicas", "2", "--replica-gpus", "0,1"])
        assert ns.replica_gpus == "0,1"
        ns = parser.parse_args([verb, "--model", "m"])
        assert ns.replica_gpus is None


def test_main_refuses_start_and_restart_since_the_router_left(
    wired: _Launches, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR-0147 (2026-10-07): the prefix router left src, so the CLI launch
    refuses before any process is touched; the GPU-share and KV-pin rules
    stay tested through start_cluster directly (sections 4 and 6)."""
    for verb in ("start", "restart"):
        monkeypatch.setattr(sys, "argv", ["manage_vllm_cluster.py", verb, "--model", "m", "--replicas", "2"])
        rc = mc.main()
        assert rc == 1
        err = capsys.readouterr().err
        assert "Error:" in err and "left src on 2026-10-07" in err and "manage_vllm_pd.sh" in err
        assert wired.calls == []
    # stop and status, the cleanup traps' commands, are untouched
    monkeypatch.setattr(sys, "argv", ["manage_vllm_cluster.py", "stop"])
    assert mc.main() == 0
    monkeypatch.setattr(sys, "argv", ["manage_vllm_cluster.py", "status"])
    assert mc.main() == 1  # "Cluster is not running."


# ---------------------------------------------------------------------------
# 6. ADR-0124 (S0F-19): the per-replica KV byte pin and the sequential start
# ---------------------------------------------------------------------------


def test_kv_budget_env_and_typed_refusal() -> None:
    assert mc.KV_BUDGET_ENV == "CAGE_KV_BUDGET_BYTES_REPLICA"
    assert issubclass(mc.KvBudgetError, ValueError)


def test_resolve_kv_budget_shared_requires_the_pin() -> None:
    with pytest.raises(mc.KvBudgetError) as excinfo:
        mc.resolve_kv_budget(mode="shared", env={})
    msg = str(excinfo.value)
    assert "CAGE_KV_BUDGET_BYTES_REPLICA" in msg and "--kv-cache-memory-bytes" in msg
    assert "pd" in msg.lower()  # the precedent the operator already knows
    assert mc.resolve_kv_budget(mode="shared", env={"CAGE_KV_BUDGET_BYTES_REPLICA": "5713920000"}) == 5713920000
    assert mc.resolve_kv_budget(mode="distinct", env={}) is None
    assert mc.resolve_kv_budget(mode="distinct", env={"CAGE_KV_BUDGET_BYTES_REPLICA": " 42 "}) == 42
    # empty is unset, like the bash launchers
    assert mc.resolve_kv_budget(mode="distinct", env={"CAGE_KV_BUDGET_BYTES_REPLICA": "  "}) is None


@pytest.mark.parametrize("bad", ["0", "-1", "1.5", "1e9", "abc", "5_000"])
def test_resolve_kv_budget_refuses_bad_values(bad: str) -> None:
    with pytest.raises(mc.KvBudgetError) as excinfo:
        mc.resolve_kv_budget(mode="shared", env={"CAGE_KV_BUDGET_BYTES_REPLICA": bad})
    assert "positive integer" in str(excinfo.value)


@pytest.mark.parametrize("clash", ["CAGE_KV_BUDGET_BYTES", "CAGE_VLLM_GPU_BLOCKS_OVERRIDE"])
def test_resolve_kv_budget_refuses_single_instance_knobs_alongside(clash: str) -> None:
    # Mirrors manage_vllm_pd.sh: two caps for the same pools is a refusal, never
    # a precedence rule.
    env = {"CAGE_KV_BUDGET_BYTES_REPLICA": "100", clash: "7"}
    with pytest.raises(mc.KvBudgetError) as excinfo:
        mc.resolve_kv_budget(mode="distinct", env=env)
    assert clash in str(excinfo.value)


def test_build_serve_args_places_the_kv_pin_before_the_log_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_DISABLE_LOG_REQUESTS", "0")
    args = mc.build_serve_args("fake/model", 8101, gpu_memory_utilization="0.45", kv_budget_bytes=123)
    i = args.index("--kv-cache-memory-bytes")
    assert args[i + 1] == "123"
    assert i > args.index("--gpu-memory-utilization")
    assert args[-1] == "--enable-log-requests"  # the W29 contract: the log flag stays last
    assert "--kv-cache-memory-bytes" not in mc.build_serve_args(
        "fake/model", 8101, gpu_memory_utilization="0.90", kv_budget_bytes=None
    )


def test_start_shared_without_pin_refuses_before_touching_any_process(
    wired: _Launches, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(mc.KV_BUDGET_ENV, raising=False)
    stops: List[bool] = []
    monkeypatch.setattr(mc, "stop_cluster", lambda *a, **k: stops.append(True) or 0)
    with pytest.raises(mc.KvBudgetError):
        _start(2, None)
    assert wired.calls == [] and stops == []
    assert not mc.STATE_FILE.exists()


def test_start_shared_pins_every_replica_and_records_it(
    wired: _Launches, capsys: pytest.CaptureFixture[str]
) -> None:
    _start(2, None)
    out = capsys.readouterr().out
    assert "kv-cache-memory-bytes 5713920000" in out or "5713920000" in out
    replica_cmds = [cmd for cmd, _ in wired.calls if cmd[:2] == ["vllm", "serve"]]
    assert len(replica_cmds) == 2
    for cmd in replica_cmds:
        assert cmd[cmd.index("--kv-cache-memory-bytes") + 1] == "5713920000"
    state = mc.load_state()
    assert state["kv_budget_bytes"] == 5713920000
    assert state["start_order"] == "sequential"


def test_start_distinct_without_pin_keeps_the_legacy_argv(wired: _Launches, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(mc.KV_BUDGET_ENV, raising=False)
    _start(2, "0,1")
    replica_cmds = [cmd for cmd, _ in wired.calls if cmd[:2] == ["vllm", "serve"]]
    assert all("--kv-cache-memory-bytes" not in cmd for cmd in replica_cmds)
    state = mc.load_state()
    assert state["kv_budget_bytes"] is None
    assert state["start_order"] == "concurrent"


def _events_start(monkeypatch: pytest.MonkeyPatch, wired: _Launches, replicas: int, pins: Optional[str]) -> List[Tuple[str, str]]:
    events: List[Tuple[str, str]] = []

    def _launch(cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
        label = cmd[cmd.index("--port") + 1] if "--port" in cmd else "router"
        events.append(("launch", label))
        return wired(cmd, log_path, env)

    def _wait(predicate, timeout, label, **kw) -> float:
        events.append(("wait", label.split(" on ")[0]))
        return 0.0

    monkeypatch.setattr(mc, "launch_process", _launch)
    monkeypatch.setattr(mc, "wait_for", _wait)
    _start(replicas, pins)
    return events


def test_start_shared_launches_replicas_one_at_a_time(wired: _Launches, monkeypatch: pytest.MonkeyPatch) -> None:
    """Spec 9 option C: on a shared GPU replica k+1 launches only after replica k
    answered, so vLLM's init-time fraction check meets a settled device."""
    events = _events_start(monkeypatch, wired, 3, None)
    assert events[:6] == [
        ("launch", "8001"), ("wait", "replica-1"),
        ("launch", "8002"), ("wait", "replica-2"),
        ("launch", "8003"), ("wait", "replica-3"),
    ]
    assert events[6] == ("launch", "router")


def test_start_distinct_keeps_the_concurrent_launch(wired: _Launches, monkeypatch: pytest.MonkeyPatch) -> None:
    events = _events_start(monkeypatch, wired, 2, "0,1")
    assert events[:4] == [
        ("launch", "8001"), ("launch", "8002"),
        ("wait", "replica-1"), ("wait", "replica-2"),
    ]


# ---------------------------------------------------------------------------
# 7. S0F-18: port probe, fail-fast, env-backed timeouts, default port
# ---------------------------------------------------------------------------


def test_default_base_port_is_off_the_image_nginx_on_start_and_restart() -> None:
    parser = mc.build_parser()
    for verb in ("start", "restart"):
        ns = parser.parse_args([verb, "--model", "m"])
        assert ns.base_port == 8101, "8001 is the RunPod image's nginx (S0F-18); 8101 was proven live"
    text = (REPO_ROOT / "scripts" / "3_run" / "run_baselines.sh").read_text(encoding="utf-8")
    assert "CLUSTER_BASE_PORT=${CLUSTER_BASE_PORT:-8101}" in text


def test_port_probe_refuses_a_bound_port_naming_the_holder() -> None:
    import socket

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("0.0.0.0", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        free, reason = mc.port_is_free(port)
        assert free is False and reason
        with pytest.raises(mc.PortInUseError) as excinfo:
            mc.refuse_bound_ports({"replica-1": port})
        msg = str(excinfo.value)
        assert f"replica-1 port {port}" in msg and "already bound" in msg
        assert "holder:" in msg
        assert "--base-port" in msg and "nginx" in msg
    finally:
        holder.close()
    free, _ = mc.port_is_free(port)
    assert free is True  # SO_REUSEADDR: the closed listener does not false-refuse


def test_start_probes_every_port_after_the_stale_stop_and_before_any_launch(
    wired: _Launches, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: List[Dict[str, int]] = []

    def _probe(ports: Dict[str, int]) -> None:
        seen.append(dict(ports))
        raise mc.PortInUseError("replica-1 port 8001 is already bound")

    monkeypatch.setattr(mc, "refuse_bound_ports", _probe)
    with pytest.raises(mc.PortInUseError):
        _start(2, "0,1")
    assert wired.calls == [], "the probe must precede every launch"
    assert seen == [{"replica-1": 8001, "replica-2": 8002, "router": 9000}]
    assert not mc.STATE_FILE.exists()


def _parse(argv: List[str]):
    return mc.build_parser().parse_args(argv)


def test_timeout_defaults_come_from_the_launchers_env_knobs() -> None:
    assert mc.REPLICA_TIMEOUT_ENV == "VLLM_START_TIMEOUT"
    assert mc.ROUTER_TIMEOUT_ENV == "ROUTER_START_TIMEOUT"
    ns = _parse(["start", "--model", "m"])
    assert (ns.replica_timeout, ns.router_timeout) == (None, None)  # resolved in main, not argparse
    assert mc.resolve_timeouts(ns, {}) == (300, 60)
    assert mc.resolve_timeouts(ns, {"VLLM_START_TIMEOUT": "900", "ROUTER_START_TIMEOUT": "120"}) == (900, 120)
    # explicit flags still win
    ns = _parse(["start", "--model", "m", "--replica-timeout", "5", "--router-timeout", "7"])
    assert mc.resolve_timeouts(ns, {"VLLM_START_TIMEOUT": "900"}) == (5, 7)
    # empty is unset (the manager's rule for VLLM_GPU_MEMORY_UTILIZATION too)
    ns = _parse(["start", "--model", "m"])
    assert mc.resolve_timeouts(ns, {"VLLM_START_TIMEOUT": "  "}) == (300, 60)
    # 0 is the bash launchers' and the suite's fail-fast idiom: accepted
    assert mc.resolve_timeouts(ns, {"VLLM_START_TIMEOUT": "0"}) == (0, 60)


@pytest.mark.parametrize("bad", ["abc", "-5", "1.5"])
def test_malformed_timeout_env_exits_2_on_start_only(
    wired: _Launches, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    monkeypatch.setenv("VLLM_START_TIMEOUT", bad)
    monkeypatch.setattr(sys, "argv", ["manage_vllm_cluster.py", "start", "--model", "m", "--replicas", "1"])
    with pytest.raises(SystemExit) as excinfo:
        mc.main()
    assert excinfo.value.code == 2
    assert wired.calls == []
    assert "usage:" in capsys.readouterr().err  # the start refusal, consumed here
    # Review 2026-09-30 (MEDIUM 2): stop and status, the cleanup traps' commands,
    # never read the knob and never exit 2 because of it.
    monkeypatch.setattr(sys, "argv", ["manage_vllm_cluster.py", "stop"])
    assert mc.main() == 0
    monkeypatch.setattr(sys, "argv", ["manage_vllm_cluster.py", "status"])
    assert mc.main() == 1  # "Cluster is not running."
    assert "usage:" not in capsys.readouterr().err




class _FakeChild:
    def __init__(self, pid: int, rc: Optional[int]) -> None:
        self.pid = pid
        self._rc = rc

    def poll(self) -> Optional[int]:
        return self._rc


def test_wait_for_fails_fast_when_the_child_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S0F-18 run 2: both replicas died at 18:51:13 and the manager waited until
    19:03:05 (11 min 54 s of billed idle H100) because it never looked at its
    own children. The wait must raise on the first poll after the exit."""
    log = tmp_path / "replica-1.log"
    log.write_text("".join(f"line {i}\n" for i in range(40)), encoding="utf-8")
    monkeypatch.setitem(mc._CHILDREN, 777001, _FakeChild(777001, 1))
    with pytest.raises(mc.ChildExitedError) as excinfo:
        mc.wait_for(lambda: False, 30, "replica-1 on http://localhost:8101", pid=777001, log_path=log)
    msg = str(excinfo.value)
    assert "replica-1" in msg and "exit code 1" in msg and str(log) in msg
    assert "line 39" in msg and "line 10" not in msg  # the tail, not the whole log


def test_wait_for_returns_the_elapsed_seconds_when_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(mc._CHILDREN, 777002, _FakeChild(777002, None))
    elapsed = mc.wait_for(lambda: True, 30, "router on http://localhost:9000", pid=777002, log_path=None)
    assert isinstance(elapsed, float) and 0.0 <= elapsed < 2.0


def test_is_pid_running_reports_a_reaped_child_as_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    import os as _os

    me = _os.getpid()  # os.kill(me, 0) succeeds, so only the child table can say "dead"
    monkeypatch.setitem(mc._CHILDREN, me, _FakeChild(me, 0))
    assert mc.is_pid_running(me) is False


def test_terminate_process_group_signals_the_group_of_an_exited_leader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review 2026-09-30 (MEDIUM 1): a replica's `vllm serve` leader can exit while
    its EngineCore keeps the GPU. The cleanup must signal the GROUP (pgid == pid
    under start_new_session) even though the leader is dead and reaped."""
    import os as _os
    import signal as _signal

    sent: List[Tuple[int, int]] = []
    alive = {"group": True}

    def _killpg(pgid: int, sig: int) -> None:
        sent.append((pgid, sig))
        if sig == 0 and not alive["group"]:
            raise ProcessLookupError
        if sig == _signal.SIGTERM:
            alive["group"] = False

    def _getpgid(pid: int) -> int:
        raise ProcessLookupError  # the leader is reaped: no pgid lookup possible

    monkeypatch.setattr(_os, "killpg", _killpg)
    monkeypatch.setattr(_os, "getpgid", _getpgid)
    monkeypatch.setitem(mc._CHILDREN, 777003, _FakeChild(777003, 3))
    mc.terminate_process_group(777003, "replica-1", silent=True)
    assert (777003, _signal.SIGTERM) in sent, "the group must be signaled through pgid == pid"
    assert (777003, _signal.SIGKILL) not in sent  # the group was gone after SIGTERM


def test_terminate_process_group_does_nothing_when_the_group_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    import os as _os

    sent: List[Tuple[int, int]] = []

    def _killpg(pgid: int, sig: int) -> None:
        sent.append((pgid, sig))
        raise ProcessLookupError

    monkeypatch.setattr(_os, "killpg", _killpg)
    monkeypatch.setattr(_os, "getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError))
    monkeypatch.setitem(mc._CHILDREN, 777004, _FakeChild(777004, 0))
    mc.terminate_process_group(777004, "replica-2", silent=True)
    assert sent == [(777004, 0)]  # one liveness probe, no signal


@pytest.mark.skipif(
    os.environ.get("CAGE_PROCESS_TESTS") != "1",
    reason="process-lifecycle test: runs only inside a container (CAGE_PROCESS_TESTS=1), never on the macOS host",
)
def test_terminate_process_group_reaches_a_survivor_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Live counterpart (container only): a leader that exits leaves a child in
    its session; the manager's own cleanup must reach that child by group."""
    import os as _os
    import time as _time

    monkeypatch.setattr(mc, "LOG_DIR", tmp_path)
    pid = mc.launch_process(["sh", "-c", "sleep 30 & exit 3"], tmp_path / "x.log")
    for _ in range(50):
        if mc.child_exit_code(pid) is not None:
            break
        _time.sleep(0.1)
    assert mc.child_exit_code(pid) == 3
    mc.terminate_process_group(pid, "leader-exited", silent=True)
    with pytest.raises(ProcessLookupError):
        _os.killpg(pid, 0)


def test_refuse_bound_ports_sees_a_bound_but_not_listening_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review 2026-09-30 (MEDIUM 3): a vLLM still loading weights holds its port
    bound without listening; on Linux the SO_REUSEADDR probe would call it free."""
    monkeypatch.setattr(mc, "_psutil_sockets_on", lambda port: [("CLOSE", 4242)] if port == 48123 else [])
    with pytest.raises(mc.PortInUseError) as excinfo:
        mc.refuse_bound_ports({"replica-1": 48123})
    assert "state CLOSE" in str(excinfo.value) and "4242" in str(excinfo.value)
    # TIME_WAIT alone is a dead server's leftover: not a refusal
    monkeypatch.setattr(mc, "_psutil_sockets_on", lambda port: [("TIME_WAIT", None)])
    mc.refuse_bound_ports({"replica-1": 48124})


def test_refuse_bound_ports_refuses_colliding_planned_ports() -> None:
    with pytest.raises(mc.PortInUseError, match="collide"):
        mc.refuse_bound_ports({"replica-1": 8998, "replica-2": 8999, "replica-3": 9000, "router": 9000})


def test_existing_state_reuse_requires_the_same_kv_pin(
    wired: _Launches, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review 2026-09-30 (LOW 5): the KV byte pin is a dial like the GPU pins; a
    running cluster under another pin is relaunched, never reused."""
    _start(2, None)
    n = len(wired.calls)
    capsys.readouterr()
    _start(2, None)
    assert len(wired.calls) == n and "already running" in capsys.readouterr().out
    monkeypatch.setenv(mc.KV_BUDGET_ENV, "100")
    _start(2, None)
    assert len(wired.calls) > n


def test_start_prints_ready_after_seconds(wired: _Launches, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(mc, "wait_for", lambda predicate, timeout, label, **kw: 12.4)
    _start(1, None)
    out = capsys.readouterr().out
    assert "replica-1 ready after 12 s" in out
    assert "router ready after 12 s" in out


# ---------------------------------------------------------------------------
# 8. S0F-18: run_tests.sh passes the cluster knobs through, RUNBOOK names the listeners
# ---------------------------------------------------------------------------


RUN_TESTS = REPO_ROOT / "scripts" / "checks" / "run_tests.sh"


def test_run_tests_cluster_knobs_are_env_backed_and_the_start_line_stays_single() -> None:
    text = RUN_TESTS.read_text(encoding="utf-8")
    for knob in ("CAGE_CLUSTER_BASE_PORT", "CAGE_CLUSTER_ROUTER_PORT", "ROUTER_TEST_API_BASE"):
        assert knob in text, f"run_tests.sh must honor {knob}"
    assert text.count("manage_vllm_cluster.py start") == 1


def _run_tests_with_stub_python(tmp_path: Path, env_extra: Dict[str, str]) -> str:
    import os as _os
    import subprocess

    stub_bin = tmp_path / "bin"
    stub_bin.mkdir(parents=True)
    argv_log = tmp_path / "argv.log"
    stub = stub_bin / "python3"
    stub.write_text(f'#!/bin/sh\necho "$@" >> "{argv_log}"\nexit 0\n', encoding="utf-8")
    stub.chmod(0o755)
    env = dict(_os.environ)
    env.update({
        "PATH": f"{stub_bin}:{env.get('PATH', '')}",
        "VIRTUAL_ENV": str(tmp_path),  # keeps PYTHON=python3 (the stub)
        "CAGE_TESTS_WITH_CLUSTER": "1",
        "VLLM_TEST_MODEL": "fake/model",
    })
    for k in ("CAGE_CLUSTER_BASE_PORT", "CAGE_CLUSTER_ROUTER_PORT", "ROUTER_TEST_API_BASE"):
        env.pop(k, None)
    env.update(env_extra)
    subprocess.run(["bash", str(RUN_TESTS)], env=env, capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT))
    return argv_log.read_text(encoding="utf-8") if argv_log.exists() else ""


def test_run_tests_appends_the_cluster_flags_only_when_the_knobs_are_set(tmp_path: Path) -> None:
    plain = _run_tests_with_stub_python(tmp_path / "plain", {})
    start = [l for l in plain.splitlines() if "manage_vllm_cluster.py start" in l]
    assert len(start) == 1 and "--base-port" not in start[0] and "--router-port" not in start[0]

    knobbed = _run_tests_with_stub_python(
        tmp_path / "knobbed", {"CAGE_CLUSTER_BASE_PORT": "8301", "CAGE_CLUSTER_ROUTER_PORT": "9100"}
    )
    start = [l for l in knobbed.splitlines() if "manage_vllm_cluster.py start" in l]
    assert len(start) == 1
    assert "--base-port 8301" in start[0] and "--router-port 9100" in start[0]


def test_runbook_names_the_image_listeners() -> None:
    text = (REPO_ROOT / "docs" / "RUNBOOK.md").read_text(encoding="utf-8")
    for port in ("3001", "7270", "7861", "8001", "8081", "9091"):
        assert port in text, f"docs/RUNBOOK.md must list the RunPod image's nginx listener {port} (S0F-18)"
    assert "CAGE_CLUSTER_BASE_PORT" in text
