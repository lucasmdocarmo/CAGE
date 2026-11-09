"""DEC-8 (2026-11-08): the pool capture reads, beside the bytes and tokens
channels, the lines a hybrid start prints and the driver's geometry check
needs: vLLM's "Maximum concurrency" line (the block count is recovered from
it) and its "blocks available" warning, SGLang's Mamba cache size and its
running-request ceiling. The fixtures are the S2R lines of 2026-11-07
verbatim (experiments/S2R/2026-11-07_1536). The bytes and tokens channels,
the gate and the record schema are unchanged."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE = REPO_ROOT / "scripts" / "checks" / "kv_pool_log.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("kv_pool_log_dec8", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kp = _load_module()
GIB = 1024 ** 3

VLLM_S2R_LOG = (
    "(EngineCore pid=328892) INFO 11-07 23:00:01 [core.py:105] Initializing a V1 LLM engine (v0.19.1) with config: "
    "model='Qwen/Qwen3-Next-80B-A3B-Instruct'\n"
    "(EngineCore pid=328892) INFO 11-07 23:02:57 [gpu_worker.py:361] Initial free memory 139.81 GiB, reserved 1.23 GiB "
    "memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling.\n"
    "(EngineCore pid=328892) INFO 11-07 23:02:57 [kv_cache_utils.py:1319] GPU KV cache size: 26,928 tokens\n"
    "(EngineCore pid=328892) INFO 11-07 23:02:57 [kv_cache_utils.py:1324] Maximum concurrency for 16,384 tokens per request: 5.91x\n"
    "(Worker_TP0 pid=329181) WARNING 11-07 23:02:57 [compilation.py:1323] Capping cudagraph capture sizes from max 512 to 384 "
    "to fit Mamba cache blocks (396 blocks available). This limits the maximum batch size that can use CUDA graphs.\n"
    "(Worker_TP1 pid=329182) WARNING 11-07 23:02:57 [compilation.py:1323] Capping cudagraph capture sizes from max 512 to 384 "
    "to fit Mamba cache blocks (396 blocks available). This limits the maximum batch size that can use CUDA graphs.\n"
)
SGLANG_S2R_LOG = (
    "[2026-11-07 23:18:13] server_args=ServerArgs(model_path='Qwen/Qwen3-Next-80B-A3B-Instruct', max_running_requests=None, "
    "max_total_tokens=215766, max_mamba_cache_size=None, mamba_ssm_dtype=None)\n"
    "[2026-11-07 23:19:23 TP0] KV Cache is allocated. #tokens: 215766, K size: 1.23 GB, V size: 1.23 GB\n"
    "[2026-11-07 23:19:23 TP0] Mamba Cache is allocated. max_mamba_cache_size: 2286, conv_state size: 0.94GB, ssm_state size: 40.20GB \n"
    "[2026-11-07 23:19:23 TP1] KV Cache is allocated. #tokens: 215766, K size: 1.23 GB, V size: 1.23 GB\n"
    "[2026-11-07 23:19:23 TP1] Mamba Cache is allocated. max_mamba_cache_size: 2286, conv_state size: 0.94GB, ssm_state size: 40.20GB \n"
    "[2026-11-07 23:20:40 TP0] max_total_num_tokens=215766, chunked_prefill_size=8192, max_prefill_tokens=16384, "
    "max_running_requests=762, context_len=16384, available_gpu_mem=55.73 GB\n"
)
# the S0 2026-09-30 dense start: no hybrid lines
VLLM_DENSE_LOG = (
    "(EngineCore pid=29655) INFO 09-30 16:06:26 [core.py:105] Initializing a V1 LLM engine (v0.19.1) with config: model='Qwen/Qwen3-8B'\n"
    "(EngineCore pid=29655) INFO 09-30 16:07:24 [gpu_worker.py:361] Initial free memory 77.91 GiB, reserved 5.32 GiB memory for KV Cache "
    "as specified by kv_cache_memory_bytes config and skipped memory profiling.\n"
    "(EngineCore pid=29655) INFO 09-30 16:07:24 [kv_cache_utils.py:1319] GPU KV cache size: 38,736 tokens\n"
)


def test_the_vllm_hybrid_lines_are_read_beside_the_pool() -> None:
    r = kp.parse_engine_log("vllm", VLLM_S2R_LOG)
    assert (r["tokens"], r["bytes"]) == (26_928, int(1.23 * GIB))
    assert r["extra"] == {"max_concurrency": 5.91, "max_concurrency_request_tokens": 16_384, "blocks": 396}
    assert not any("Maximum concurrency" in e or "blocks available" in e for e in r["evidence"])  # not pool lines
    assert sum("Maximum concurrency" in e for e in r["extra_evidence"]) == 1
    assert sum("blocks available" in e for e in r["extra_evidence"]) == 2  # one per rank, the last wins


def test_the_sglang_hybrid_lines_are_read_beside_the_pool() -> None:
    r = kp.parse_engine_log("sglang", SGLANG_S2R_LOG)
    assert (r["tokens"], r["bytes"]) == (215_766, int(2.46 * GIB))
    assert r["extra"] == {"max_mamba_cache_size": 2286, "max_running_requests": 762}


def test_a_dense_start_carries_an_empty_extra() -> None:
    assert kp.parse_engine_log("vllm", VLLM_DENSE_LOG)["extra"] == {}
    dense_sglang = (
        "[2026-09-30 14:27:37] server_args=ServerArgs(model_path='Qwen/Qwen3-8B', max_running_requests=None)\n"
        "[2026-08-18 10:00:00] KV Cache is allocated. #tokens: 430913, K size: 13.15 GB, V size: 13.15 GB\n"
        "[2026-08-18 10:00:01] max_total_num_tokens=430913, chunked_prefill_size=8192\n"
    )
    assert kp.parse_engine_log("sglang", dense_sglang)["extra"] == {}


def test_a_corrupted_hybrid_line_is_loud() -> None:
    with pytest.raises(kp.IsoBytesError, match="corrupted"):
        kp.parse_engine_log("vllm", VLLM_DENSE_LOG + "x [kv_cache_utils.py:1324] Maximum concurrency for tokens per request: lots\n")
    with pytest.raises(kp.IsoBytesError, match="corrupted"):
        kp.parse_engine_log("sglang", SGLANG_S2R_LOG.replace("max_mamba_cache_size: 2286", "max_mamba_cache_size: many"))


def test_the_tail_rule_covers_the_hybrid_lines() -> None:
    older = VLLM_S2R_LOG.replace("5.91x", "9.63x").replace("26,928", "43,792").replace("(396 blocks", "(645 blocks")
    r = kp.parse_engine_log("vllm", older + VLLM_S2R_LOG)
    assert r["starts"] == 2 and r["tokens"] == 26_928 and r["extra"]["max_concurrency"] == 5.91 and r["extra"]["blocks"] == 396


def test_the_capture_and_pd_records_carry_the_extra(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(VLLM_S2R_LOG, encoding="utf-8")
    rec = kp.capture_record("vllm", log, role="prefill", now=lambda: "2026-11-08T00:00:00Z")
    assert rec["extra"] == {"max_concurrency": 5.91, "max_concurrency_request_tokens": 16_384, "blocks": 396}
    assert (rec["tokens"], rec["schema"]) == (26_928, kp.CAPTURE_SCHEMA)
    pd = kp.pd_record("vllm", {"prefill": rec, "decode": rec}, now=lambda: "2026-11-08T00:00:00Z")
    assert pd["roles"]["decode"]["extra"]["blocks"] == 396
    sg = tmp_path / "sglang.log"
    sg.write_text(SGLANG_S2R_LOG, encoding="utf-8")
    assert kp.capture_record("sglang", sg)["extra"] == {"max_mamba_cache_size": 2286, "max_running_requests": 762}
    dense = tmp_path / "dense.log"
    dense.write_text(VLLM_DENSE_LOG, encoding="utf-8")
    assert kp.capture_record("vllm", dense)["extra"] == {}
