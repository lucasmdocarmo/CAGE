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
    monkeypatch.setattr(mc, "wait_for", lambda predicate, timeout, label: None)
    monkeypatch.setattr(mc, "is_pid_running", lambda pid: True)
    monkeypatch.setattr(mc, "replica_ready", lambda api_base, model: True)
    monkeypatch.setattr(mc, "terminate_process_group", lambda *a, **k: None)
    monkeypatch.setattr(
        mc, "fetch_router_stats",
        lambda url: {"num_replicas": mc._EXPECTED_REPLICAS, "distinct_api_bases": mc._EXPECTED_REPLICAS},
    )
    monkeypatch.setattr(mc.importlib.util, "find_spec", lambda name: object())
    monkeypatch.delenv("VLLM_GPU_MEMORY_UTILIZATION", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
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


def test_main_reports_gpu_share_refusal_as_error_exit_1(
    wired: _Launches, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
    monkeypatch.setattr(sys, "argv", ["manage_vllm_cluster.py", "start", "--model", "m", "--replicas", "2"])
    rc = mc.main()
    assert rc == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "vLLM startup check" in err
    assert wired.calls == []
