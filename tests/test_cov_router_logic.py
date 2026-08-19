"""Offline coverage for the router's pure decision logic (K-COV4, #142).

src/orchestration/router.py (0/357) and cache_manager.py (0/76) are wired into
the serving path but every existing test is a live-gated skip — silently, not
declaredly, unproven. These tests exercise the PURE routing/parsing logic
offline: prefix hashing, replica selection, ROUTER_REPLICAS parsing, the
simulated KV-transfer arithmetic, and stats accounting.

The lean analysis venv has no fastapi/pydantic/transformers, so minimal stub
modules are injected into sys.modules BEFORE importing the router (the
sys.modules-stub pattern of tests/test_compression_ops.py). The stubs are
import-surface only: FastAPI decorators are pass-through, HTTPException is a
plain exception carrying status_code/detail. No serving endpoint is invoked —
endpoint handlers stay out of scope for the offline layer (they are the
VERIFY-LIVE surface).

No network: aiohttp is imported by the module but never used — every test
drives router methods whose code paths make no HTTP call (tokenizer state is
pre-set so initialize_tokenizer's replica probe never runs).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
import types


def _missing(name: str) -> bool:
    """True when `name` is neither imported nor installed (stub needed)."""
    return name not in sys.modules and importlib.util.find_spec(name) is None


def _install_router_import_stubs() -> None:
    """Stub fastapi/pydantic so the router module imports in the lean venv.

    A venv that really has the packages keeps them (find_spec wins) — the
    stubs exist only so the LEAN analysis venv can import the module at all.
    """
    if _missing("fastapi"):
        fastapi = types.ModuleType("fastapi")

        class HTTPException(Exception):
            def __init__(self, status_code: int, detail: str = ""):
                super().__init__(f"{status_code}: {detail}")
                self.status_code = status_code
                self.detail = detail

        class FastAPI:
            def __init__(self, *args, **kwargs):
                pass

            def _passthrough(self, *args, **kwargs):
                def decorator(fn):
                    return fn
                return decorator

            get = post = on_event = _passthrough

        fastapi.FastAPI = FastAPI
        fastapi.HTTPException = HTTPException
        responses = types.ModuleType("fastapi.responses")

        class _Response:
            def __init__(self, *args, **kwargs):
                pass

        responses.PlainTextResponse = _Response
        responses.StreamingResponse = _Response
        responses.JSONResponse = _Response
        fastapi.responses = responses
        sys.modules["fastapi"] = fastapi
        sys.modules["fastapi.responses"] = responses
    if _missing("pydantic"):
        pydantic = types.ModuleType("pydantic")

        class BaseModel:
            pass

        pydantic.BaseModel = BaseModel
        pydantic.ConfigDict = dict
        sys.modules["pydantic"] = pydantic


_install_router_import_stubs()

from src.orchestration.cache_manager import (  # noqa: E402
    CacheNode,
    SimulatedKVCacheManager,
)
from src.orchestration.router import (  # noqa: E402
    PrefixAwareRouter,
    ReplicaConfig,
    _parse_router_replicas_env,
)
from src.utils.prompting import extract_cacheable_prefix_text  # noqa: E402

HTTPException = sys.modules["fastapi"].HTTPException


def _replicas(n: int = 3) -> list[ReplicaConfig]:
    return [
        ReplicaConfig(replica_id=f"replica-{i + 1}", api_base=f"http://host{i + 1}:8000")
        for i in range(n)
    ]


def _router(n: int = 3, strategy: str = "hash") -> PrefixAwareRouter:
    r = PrefixAwareRouter(_replicas(n), strategy=strategy)
    # Pre-resolve tokenizer state so route_request_with_simulation never
    # probes a replica for a model name (the only network arm in the path).
    r.tokenization_mode = "utf8_fallback"
    r.tokenizer_name = "offline-test"
    return r


# --------------------------------------------------------------------------- #
# ROUTER_REPLICAS env parsing
# --------------------------------------------------------------------------- #


class TestParseRouterReplicasEnv:
    def test_empty_and_whitespace_give_no_replicas(self):
        assert _parse_router_replicas_env("") == []
        assert _parse_router_replicas_env("   ") == []

    def test_json_list_of_strings(self):
        got = _parse_router_replicas_env('["http://a:8001/", "http://b:8002"]')
        assert [(r.replica_id, r.api_base) for r in got] == [
            ("replica-1", "http://a:8001"),  # trailing slash stripped
            ("replica-2", "http://b:8002"),
        ]

    def test_json_list_of_objects_with_defaults_and_skips(self):
        value = json.dumps([
            {"replica_id": "gpu-a", "api_base": "http://a:8001/", "weight": 2},
            {"api_base": "http://b:8002"},        # id defaults by position
            {"replica_id": "no-base"},             # no api_base -> skipped
            "not-a-dict",                          # non-dict -> skipped
            {"api_base": "http://c:8003", "weight": "heavy"},  # bad weight -> 1.0
        ])
        got = _parse_router_replicas_env(value)
        assert [(r.replica_id, r.api_base, r.weight) for r in got] == [
            ("gpu-a", "http://a:8001", 2.0),
            ("replica-2", "http://b:8002", 1.0),
            ("replica-5", "http://c:8003", 1.0),
        ]

    def test_single_json_object_is_wrapped(self):
        got = _parse_router_replicas_env(
            '{"replica_id": "solo", "api_base": "http://a:8001"}'
        )
        assert [(r.replica_id, r.api_base) for r in got] == [("solo", "http://a:8001")]

    def test_comma_separated_name_equals_url(self):
        got = _parse_router_replicas_env(
            "gpu-a=http://a:8001/, gpu-b=http://b:8002 ,http://c:8003"
        )
        assert [(r.replica_id, r.api_base) for r in got] == [
            ("gpu-a", "http://a:8001"),
            ("gpu-b", "http://b:8002"),
            ("replica-3", "http://c:8003"),
        ]

    def test_mixed_type_json_list_yields_only_dict_entries(self):
        # Mixed list is NOT treated as all-strings; string members are skipped.
        got = _parse_router_replicas_env('["http://a:8001", {"api_base": "http://b:8002"}]')
        assert [(r.replica_id, r.api_base) for r in got] == [
            ("replica-2", "http://b:8002")
        ]

    def test_empty_name_falls_back_to_positional_id(self):
        got = _parse_router_replicas_env("=http://a:8001")
        assert [(r.replica_id, r.api_base) for r in got] == [
            ("replica-1", "http://a:8001")
        ]


# --------------------------------------------------------------------------- #
# Prefix hashing + replica selection
# --------------------------------------------------------------------------- #


class TestPrefixHashingAndSelection:
    def test_compute_prefix_hash_is_sha1_of_canonical_token_json(self):
        r = _router()
        tokens = [1, 2, 3]
        expected = hashlib.sha1(
            json.dumps(tokens, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert r.compute_prefix_hash(tokens) == expected
        assert r.compute_prefix_hash(tokens) == r.compute_prefix_hash([1, 2, 3])
        assert r.compute_prefix_hash([1, 2, 4]) != expected

    def test_prefix_tokens_utf8_fallback_uses_cacheable_prefix(self):
        r = _router()
        prompt = "System preamble.\n\nQuestion: what?\nAnswer:"
        expected = list(extract_cacheable_prefix_text(prompt).encode("utf-8"))
        assert r._prefix_tokens(prompt) == expected

    def test_prefix_tokens_uses_model_tokenizer_when_present(self):
        r = _router()

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [len(text), 7]

        r._tokenizer = FakeTokenizer()
        prompt = "hello world"
        assert r._prefix_tokens(prompt) == [
            len(extract_cacheable_prefix_text(prompt)), 7
        ]

    def test_prefix_tokens_falls_back_when_tokenizer_raises(self):
        r = _router()

        class BrokenTokenizer:
            def encode(self, text, add_special_tokens=False):
                raise RuntimeError("tokenizer exploded")

        r._tokenizer = BrokenTokenizer()
        prompt = "hello"
        assert r._prefix_tokens(prompt) == list(
            extract_cacheable_prefix_text(prompt).encode("utf-8")
        )
        assert r.tokenization_mode == "utf8_fallback"
        assert "exploded" in r.tokenizer_error

    def test_hash_strategy_selects_by_modulo(self):
        r = _router(3)
        prefix_hash = "ff"  # int("ff", 16) = 255; 255 % 3 = 0
        assert r._select_replicated_replica(prefix_hash).replica_id == "replica-1"
        assert r._select_replicated_replica("01").replica_id == "replica-2"

    def test_round_robin_cycles_with_request_count(self):
        r = _router(3, strategy="round_robin")
        ids = []
        for _ in range(4):
            ids.append(r._select_replicated_replica("00").replica_id)
            r.request_count += 1
        assert ids == ["replica-1", "replica-2", "replica-3", "replica-1"]

    def test_no_replicas_is_a_503_refusal(self):
        r = PrefixAwareRouter([], strategy="hash")
        try:
            r._select_replicated_replica("00")
            raise AssertionError("expected HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 503

    def test_set_strategy_validates(self):
        r = _router()
        r.set_strategy("round_robin")
        assert r.strategy == "round_robin"
        try:
            r.set_strategy("weighted")
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "round_robin" in str(exc)

    def test_replica_config_hashes_by_id(self):
        a = ReplicaConfig(replica_id="r1", api_base="http://a")
        b = ReplicaConfig(replica_id="r1", api_base="http://b")
        assert hash(a) == hash(b)
        assert len({a.replica_id, b.replica_id}) == 1


# --------------------------------------------------------------------------- #
# Simulated KV cache manager (cache_manager.py)
# --------------------------------------------------------------------------- #


def _nodes(n: int) -> list[CacheNode]:
    return [
        CacheNode(node_id=f"replica-{i + 1}", host=f"http://host{i + 1}", port=0,
                  vram_total=0, vram_used=0, cache_blocks_capacity=0,
                  cache_blocks_used=0)
        for i in range(n)
    ]


class TestSimulatedKVCacheManager:
    def test_replicated_policy_has_zero_transfer_cost(self):
        mgr = SimulatedKVCacheManager(_nodes(3), policy="replicated")
        out = mgr.resolve_prefix([1, 2, 3, 4])
        assert out["cached_tokens"] == 4
        assert out["transfer_required"] is False
        assert out["transfer_bytes"] == 0
        assert out["transfer_latency_ms"] == 0.0
        assert out["target_node_id"] in {n.node_id for n in _nodes(3)}

    def test_target_node_is_stable_md5_modulo(self):
        mgr = SimulatedKVCacheManager(_nodes(3), policy="replicated")
        tokens = [5, 6, 7]
        prefix_hash = int(
            hashlib.md5(",".join(map(str, tokens)).encode()).hexdigest(), 16
        )
        expected = _nodes(3)[prefix_hash % 3].node_id
        assert mgr.resolve_prefix(tokens)["target_node_id"] == expected
        assert mgr.resolve_prefix(tokens)["target_node_id"] == expected  # stable

    def test_sharded_context_transfer_arithmetic(self):
        mgr = SimulatedKVCacheManager(_nodes(3), policy="sharded_context")
        num_tokens = 300
        out = mgr.resolve_prefix(list(range(num_tokens)))
        # (N-1)/N of the context fetched remotely at bytes_per_token each.
        expected_bytes = num_tokens * (2 / 3) * mgr.bytes_per_token
        assert out["transfer_bytes"] == int(expected_bytes)
        assert out["transfer_required"] is True
        bw_bytes_sec = mgr.network_bandwidth_gbps * 1e9 / 8
        assert out["transfer_latency_ms"] == (
            expected_bytes / bw_bytes_sec * 1000.0
        )
        # bytes_per_token = hidden 2048 x 16 layers x 2 (K+V) x 2 (fp16).
        assert mgr.bytes_per_token == 2048 * 16 * 2 * 2

    def test_sharded_context_single_node_has_no_remote_fetch(self):
        mgr = SimulatedKVCacheManager(_nodes(1), policy="sharded_context")
        out = mgr.resolve_prefix([1, 2, 3])
        assert out["transfer_bytes"] == 0
        assert out["transfer_required"] is False

    def test_initialize_replaces_cluster_and_policy(self):
        mgr = SimulatedKVCacheManager(_nodes(2), policy="replicated")
        mgr.initialize(_nodes(3), policy="sharded_context")
        assert mgr.policy == "sharded_context"
        assert len(mgr.node_list) == 3
        assert mgr.get_cluster_stats() == {"policy": "sharded_context", "nodes": 3}

    def test_allocate_and_invalidate_are_simulation_noops(self):
        mgr = SimulatedKVCacheManager(_nodes(2))
        assert mgr.allocate_context([1, 2]) == []
        mgr.invalidate(["block-1"])  # must not raise


# --------------------------------------------------------------------------- #
# route_request_with_simulation (offline: pre-resolved tokenizer state)
# --------------------------------------------------------------------------- #


class TestRouteRequestWithSimulation:
    def test_replicated_policy_metadata_contract(self):
        r = _router(3)
        replica, meta = asyncio.run(r.route_request_with_simulation("some prompt"))
        assert meta["selected_replica_id"] == replica.replica_id
        assert meta["target_node_id"] == replica.replica_id
        assert meta["cached_tokens"] == 0
        assert meta["transfer_required"] is False
        assert meta["transfer_bytes"] == 0
        assert meta["transfer_latency_ms"] == 0.0
        assert meta["policy"] == "replicated"
        assert meta["routing_mode"] == "prefix_hash"
        assert meta["tokenization_mode"] == "utf8_fallback"
        assert meta["prefix_token_count"] == len(
            extract_cacheable_prefix_text("some prompt").encode("utf-8")
        )
        # Deterministic: the same prompt routes to the same replica.
        replica2, meta2 = asyncio.run(r.route_request_with_simulation("some prompt"))
        assert replica2.replica_id == replica.replica_id
        assert meta2["prefix_hash"] == meta["prefix_hash"]

    def test_round_robin_reports_its_routing_mode(self):
        r = _router(2, strategy="round_robin")
        seen = []
        for _ in range(3):
            replica, meta = asyncio.run(r.route_request_with_simulation("p"))
            assert meta["routing_mode"] == "round_robin"
            seen.append(replica.replica_id)
        assert seen == ["replica-1", "replica-2", "replica-1"]

    def test_sharded_policy_uses_simulation_and_reports_transfer(self):
        r = _router(3)
        r.set_sharding_policy("sharded_context")
        prompt = "x" * 400
        replica, meta = asyncio.run(r.route_request_with_simulation(prompt))
        assert meta["routing_mode"] == "simulated_transfer"
        assert meta["policy"] == "sharded_context"
        assert meta["transfer_bytes"] > 0
        assert meta["transfer_required"] is True
        assert meta["transfer_latency_ms"] > 0.0
        assert meta["cached_tokens"] == meta["prefix_token_count"]
        # The selected replica is the simulation's target node.
        assert replica.replica_id == meta["target_node_id"]

    def test_stats_accounting_tracks_requests_per_replica(self):
        r = _router(3)
        for prompt in ("a", "b", "c", "a"):
            asyncio.run(r.route_request_with_simulation(prompt))
        stats = r.get_stats()
        assert stats["total_requests"] == 4
        assert sum(stats["replica_distribution"].values()) == 4
        assert stats["num_replicas"] == 3
        assert stats["strategy"] == "hash"
        assert stats["sharding_policy"] == "replicated"
        assert stats["tokenization_mode"] == "utf8_fallback"
        assert stats["distinct_api_bases"] == 3
        assert [rep["replica_id"] for rep in stats["replicas"]] == [
            "replica-1", "replica-2", "replica-3"
        ]

    def test_route_request_returns_replica_only(self):
        r = _router(3)
        replica = asyncio.run(r.route_request("prompt"))
        assert replica.replica_id in {"replica-1", "replica-2", "replica-3"}


# --------------------------------------------------------------------------- #
# initialize_tokenizer: the transformers-missing fallback arm (offline)
# --------------------------------------------------------------------------- #


class TestInitializeTokenizer:
    def test_named_tokenizer_without_transformers_falls_back(self, monkeypatch):
        # ROUTER_TOKENIZER names a tokenizer, transformers is unavailable:
        # the router must degrade to utf8_fallback with the error recorded,
        # never crash the serving path.
        monkeypatch.setitem(sys.modules, "transformers", None)  # ImportError
        monkeypatch.setenv("ROUTER_TOKENIZER", "some/model")
        r = PrefixAwareRouter(_replicas(1))
        assert r.tokenization_mode == "uninitialized"
        asyncio.run(r.initialize_tokenizer())
        assert r.tokenization_mode == "utf8_fallback"
        assert r._tokenizer is None
        assert r.tokenizer_name == "some/model"
        assert r.tokenizer_error

    def test_already_initialized_tokenizer_is_kept(self):
        r = _router()
        sentinel = object()
        r._tokenizer = sentinel
        asyncio.run(r.initialize_tokenizer())
        assert r._tokenizer is sentinel
