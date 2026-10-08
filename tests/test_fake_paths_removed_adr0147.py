"""ADR-0147 (2026-10-07): the fake paths left src and the scripts that named
them follow.

Pinned here, on the runner (loaded by path, the tests/test_wave1 pattern):

1. ``require_served_engine``: a campaign cell on the in-process vLLM engine
   (``--offline``) refuses, in ``run_experiment`` and in ``main()`` before any
   engine contact, with the same exit code as the telemetry refusal. The
   in-process adapter's ttft_ms is the full response time, a stand-in the
   campaign must never record as a streamed first-token time (charter D2).
2. The legacy ``distributed`` baseline and the hand-written ``hpc_code``
   dataset are no longer CLI choices, and the simulated-transfer label is
   gone: ``default_experiment_label`` falls through for the old token.

No GPU, no engine, no network.
"""
from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"
DEPLOY = REPO_ROOT / "scripts" / "2_serving" / "deploy_cluster.sh"


def _load_runner():
    spec = importlib.util.spec_from_file_location("cage_run_experiment_adr0147", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


# --------------------------------------------------------------------------
# 1. the --offline refusal on the campaign path
# --------------------------------------------------------------------------


def test_require_served_engine_refuses_offline_in_campaign_mode() -> None:
    with pytest.raises(ValueError, match="offline") as excinfo:
        runner.require_served_engine(True, True)
    assert "ADR-0147" in str(excinfo.value)
    assert "full response time" in str(excinfo.value)


def test_require_served_engine_leaves_the_pilot_path_and_served_engines_alone() -> None:
    runner.require_served_engine(False, True)   # pilot --offline: unchanged
    runner.require_served_engine(True, False)   # campaign on a served engine
    runner.require_served_engine(False, False)


def test_offline_refusal_is_wired_in_both_slots() -> None:
    run_src = inspect.getsource(runner.run_experiment)
    # run_experiment: beside the telemetry refusal, before any dataset or engine work
    call = run_src.index("require_served_engine(campaign_session is not None, use_offline)")
    assert run_src.index("require_campaign_telemetry(campaign_session is not None") < call
    assert call < run_src.index("Loading dataset")
    assert call < run_src.index("setup_inference_engine(model, baseline_config")
    # main(): inside the campaign activation block, same exit code as the telemetry refusal
    main_src = inspect.getsource(runner.main)
    activation = main_src.index("CAMPAIGN MODE (task #116)")
    telemetry = main_src.index("require_campaign_telemetry(True, args.backend, args.vllm_telemetry)")
    served = main_src.index("require_served_engine(True, args.offline)")
    first_run = main_src.index("def _run_with_top_k")
    assert activation < telemetry < served < first_run
    assert "sys.exit(2)" in main_src[served:first_run]


# --------------------------------------------------------------------------
# 2. the stale names are gone from the runner
# --------------------------------------------------------------------------


def test_simulated_transfer_label_is_gone() -> None:
    assert runner.default_experiment_label(
        "distributed", sharding_policy="sharded_context", warmup_queries=0
    ) == "distributed"  # falls through: no "distributed_sharded_sim", no router label
    assert "distributed_sharded_sim" not in inspect.getsource(runner.default_experiment_label)


def test_hpc_code_is_no_longer_a_code_dataset_or_a_test_split() -> None:
    assert runner.default_dataset_split("hpc_code") == "validation"
    assert runner.is_code_dataset("hpc_code") is False
    assert runner.default_dataset_split("humaneval") == "test"
    assert runner.is_code_dataset("mbpp") is True


def test_cli_choices_no_longer_offer_the_removed_paths() -> None:
    main_src = inspect.getsource(runner.main)
    assert '"hpc_code"' not in main_src
    assert '"distributed", "hybrid"' not in main_src
    assert '"rag", "hybrid"' in main_src  # the baseline choices still parse


# --------------------------------------------------------------------------
# 3. the deploy script's router-container paths refuse
# --------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["local", "k8s"])
def test_deploy_cluster_router_paths_refuse(command: str) -> None:
    proc = subprocess.run(["bash", str(DEPLOY), command, "--dry-run"],
                          capture_output=True, text=True, timeout=30, cwd=str(REPO_ROOT))
    assert proc.returncode == 1
    assert "REFUSING" in proc.stderr and "ADR-0147" in proc.stderr
    assert "manage_vllm_server.sh" in proc.stderr
