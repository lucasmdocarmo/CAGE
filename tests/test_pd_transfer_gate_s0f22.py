"""Campaign pd windows carry the engine's KV transfer ticket on every served row (S0F-22, Batch 1).

Facts the pins rest on (vLLM v0.19.1 source, read 2026-10-01):
- the prefill's ``NixlConnector.request_finished`` returns a ticket of eight
  keys (``do_remote_prefill``, ``do_remote_decode``, ``remote_block_ids``
  nested per KV group, ``remote_engine_id``, ``remote_request_id``,
  ``remote_host``, ``remote_port``, ``tp_size``; nixl_connector.py:989-998),
  only when the request carried ``do_remote_decode`` (:958), and never a
  ``source`` key;
- the decode consumes it only with the five address keys present and a
  non-empty ``remote_block_ids`` (:818-847), and an empty list with
  ``do_remote_prefill`` true kills the engine (:855-856);
- the decode's response carries no ticket at all, so the only engine-written
  ticket a client can see is the prefill's, relayed verbatim by the proxy in
  the ``x-kv-transfer-params`` response header the adapter already parses.

Before this gate, the driver's pd cells (``--baseline no_cache`` /
``prefix_cache``, run_campaign.ARM_RUNNER_BASELINE) never reached the T3.3
provenance gate, which keys on the ``distributed`` baseline token, so a pd
window with zero transfers was emitted with no refusal. This gate keys on
the cell's topology, not on the baseline token, and checks the ticket's
SHAPE: engine-written fields the proxy cannot invent. Proof that the decode
PULLED the blocks is Batch 2 (the per-window decode counters).
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUN_EXPERIMENT_PY = REPO_ROOT / "scripts" / "3_run" / "run_experiment.py"
PD_PROXY_PY = REPO_ROOT / "scripts" / "2_serving" / "pd_proxy.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load(RUN_EXPERIMENT_PY, "run_experiment_s0f22")
pd_proxy = _load(PD_PROXY_PY, "pd_proxy_s0f22")

TICKET: Dict[str, Any] = {
    "do_remote_prefill": True,
    "do_remote_decode": False,
    "remote_block_ids": [[7, 8, 9]],
    "remote_engine_id": "engine-a",
    "remote_request_id": "cmpl-1",
    "remote_host": "localhost",
    "remote_port": 5600,
    "tp_size": 4,
}


def _row(ticket: Any, *, error: Any = None, example_id: str = "q1") -> Dict[str, Any]:
    """A closed-loop results row as run_experiment writes it: ``error`` and
    ``empty_generation`` are the row's validity fields; there is no ``ok`` key
    (that derived key is written to qa_evidence.jsonl only)."""
    if isinstance(ticket, dict):
        ticket = json.dumps(ticket, sort_keys=True)  # the results writer's spelling
    return {
        "example_id": example_id,
        "error": error,
        "empty_generation": False,
        "kv_transfer_params": ticket,
    }


def test_gate_accepts_engine_shaped_tickets_as_string_or_dict() -> None:
    rows = [
        _row(TICKET),
        {"example_id": "q2", "error": None, "empty_generation": False,
         "kv_transfer_params": dict(TICKET)},
        # a refused or timed-out request legitimately carries no ticket: the
        # adapter's error response has none, so the results writer stores ""
        _row("", error="HTTP 502: prefill returned no usable KV transfer ticket", example_id="q3"),
        # the open-loop dropped-by-cap / no_response stub row has no
        # kv_transfer_params key at all (run_experiment, open-loop stage)
        {"example_id": "q4", "baseline": "no_cache", "baseline_family": "no_cache",
         "workload_mode": "open_loop", "error": "dropped_by_cap"},
    ]
    runner.enforce_pd_transfer_tickets(rows)  # must not raise


def test_gate_checks_an_empty_generation_row_that_was_served() -> None:
    # The skip rule is ``error``, not the derived ``ok``: an empty generation
    # is a 2xx the proxy answered, and the proxy answers 2xx only with a
    # ticket, so the row must carry one.
    served_empty = {**_row(TICKET, example_id="q5"), "empty_generation": True}
    runner.enforce_pd_transfer_tickets([served_empty])  # must not raise
    served_empty_no_ticket = {**_row("", example_id="q6"), "empty_generation": True}
    with pytest.raises(RuntimeError, match="CAMPAIGN PD TICKET.*q6"):
        runner.enforce_pd_transfer_tickets([served_empty_no_ticket])


@pytest.mark.parametrize("value", ["", None, "missing"], ids=["empty", "none", "absent"])
def test_gate_refuses_a_served_row_without_a_ticket(value: Any) -> None:
    row = _row(TICKET)
    if value == "missing":
        del row["kv_transfer_params"]
    else:
        row["kv_transfer_params"] = value
    with pytest.raises(RuntimeError, match="CAMPAIGN PD TICKET.*q1"):
        runner.enforce_pd_transfer_tickets([row])


@pytest.mark.parametrize("ticket,needle", [
    ({k: v for k, v in TICKET.items() if k != "remote_host"}, "remote_host"),
    ({k: v for k, v in TICKET.items() if k != "remote_engine_id"}, "remote_engine_id"),
    ({k: v for k, v in TICKET.items() if k != "remote_request_id"}, "remote_request_id"),
    ({k: v for k, v in TICKET.items() if k != "remote_port"}, "remote_port"),
    ({**TICKET, "remote_block_ids": []}, "remote_block_ids"),
    ({**TICKET, "remote_block_ids": [[]]}, "remote_block_ids"),
    ({**TICKET, "remote_block_ids": None}, "remote_block_ids"),
    ({**TICKET, "remote_block_ids": "1,2,3"}, "remote_block_ids"),
    ({**TICKET, "do_remote_prefill": False}, "do_remote_prefill"),
    ({"source": "nixl", "transfer_bytes": 10}, "do_remote_prefill"),
], ids=[
    "no-host", "no-engine", "no-request-id", "no-port", "ids-empty",
    "ids-empty-group", "ids-null", "ids-string", "prefill-flag-false",
    "old-fabricated-shape",
])
def test_gate_refuses_a_malformed_ticket(ticket: Dict[str, Any], needle: str) -> None:
    with pytest.raises(RuntimeError, match=f"CAMPAIGN PD TICKET.*{needle}"):
        runner.enforce_pd_transfer_tickets([_row(ticket)])


def test_gate_refuses_unparseable_and_non_object_tickets() -> None:
    with pytest.raises(RuntimeError, match="unparseable"):
        runner.enforce_pd_transfer_tickets([_row("not json{")])
    with pytest.raises(RuntimeError, match="not an object"):
        runner.enforce_pd_transfer_tickets([_row(json.dumps([1, 2]))])


def test_gate_is_wired_on_the_pd_topology_outside_the_distributed_block() -> None:
    src = inspect.getsource(runner.run_experiment)
    call = src.index("enforce_pd_transfer_tickets(results)")
    guard = src.rindex("campaign_session is not None", 0, call)
    assert call - guard < 400, "the campaign guard must govern the call"
    guard_text = src[guard:call]
    # the hard attribute path: a missing ``spec`` or ``topology`` raises on
    # every campaign cell instead of silently skipping the gate (fail-closed)
    assert 'campaign_session.spec.topology == "pd"' in guard_text, (
        "keyed on the cell topology, never the baseline token, never via getattr defaults")
    assert "getattr(" not in guard_text
    # after the distributed block, and not inside it: the two gates are independent
    dist = src.index('== "distributed"')
    assert dist < guard
    assert '== "distributed"' not in src[guard:call]
    assert src.index("validate_distributed_artifacts(", dist) < call


def test_proxy_header_is_the_one_the_adapter_parses() -> None:
    # The proxy relays the engine's ticket in the header the vLLM adapter
    # already reads; a drift of either name would silently empty every row.
    from src.inference.openai_chat_adapter import OpenAIChatAdapter

    assert pd_proxy.TICKET_HEADER == "x-kv-transfer-params"
    parsed = OpenAIChatAdapter._extract_header_kv_transfer_params(
        None, {"x-kv-transfer-params": json.dumps(TICKET)}
    )
    assert parsed == TICKET
    # the adapter's header channel is gated on the per-engine flag, and vllm is on
    from src.inference.vllm_adapter import VLLMAdapter

    assert VLLMAdapter._kv_transfer_telemetry is True
    # the gate's required keys are the ones the decode reads (nixl_connector.py:818-828)
    assert set(runner.PD_TICKET_REQUIRED_KEYS) == {
        "remote_block_ids", "remote_engine_id", "remote_request_id", "remote_host", "remote_port",
    }
    assert set(pd_proxy.TICKET_REQUIRED_KEYS) == set(runner.PD_TICKET_REQUIRED_KEYS)
