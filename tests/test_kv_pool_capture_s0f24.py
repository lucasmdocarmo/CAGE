"""S0F-24 (ADR-0142): launcher-side capture of the REALIZED KV pool.

Before this change gate (j) (``scripts/checks/preflight_check.sh``,
CAGE-ISO-BYTES-GATE) found each engine's newest startup log by modification
time and parsed it; the parser lived inside the gate's heredoc, so no launcher
could reuse it, and a stale or multi-start log was a discovery hazard (ADR-0130
tail rule, S0F-23 empty logs). Now:

1. the parser is the importable module ``scripts/checks/kv_pool_log.py`` and
   the heredoc imports it (the gate's behavior is unchanged: the existing
   tests/test_preflight_gates.py suite still exec's the snippet);
2. the module's ``capture`` command parses ONE log at ready time (tail rule,
   bounded retries for a line still being written) and writes an atomic
   record ``CURRENT.kvpool.json`` under the engine's log directory, merging
   the realized pool into the serving-config JSON when that file exists;
3. every launcher calls it at its ready line through
   ``cage_kv_pool_capture`` (scripts/lib/_serving_config.sh), never fatal to
   the start, and ``stop`` removes the record, so a current record always
   means a running engine; the pd launcher writes one record with both roles;
4. gate (j) discovery is: explicit pins, then the current record (exact file,
   no mtime), then the newest log exactly as before (byte-identical output).

Owner: "Follow recommendations" (design A, 2026-10-06). No GPU, no network;
the launcher runs use stub engines that print the S0 pool lines into the log.
"""
from __future__ import annotations

import importlib.util
import json
import re
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKS = REPO_ROOT / "scripts" / "checks"
MODULE = CHECKS / "kv_pool_log.py"
PREFLIGHT = CHECKS / "preflight_check.sh"
SERVING = REPO_ROOT / "scripts" / "2_serving"
LIB = REPO_ROOT / "scripts" / "lib" / "_serving_config.sh"
RUNBOOK = REPO_ROOT / "docs" / "RUNBOOK.md"
README = REPO_ROOT / "scripts" / "README.md"

GIB = 1024 ** 3
MIB = 1024 ** 2

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")


def _load_module():
    spec = importlib.util.spec_from_file_location("kv_pool_log_s0f24", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kp = _load_module()

# --- S0 2026-09-30 log shapes (verbatim from tests/test_preflight_gates.py) ---
VLLM_019_ECHO_LINE = (
    "(APIServer pid=28971) INFO 09-30 16:04:44 [utils.py:233] non-default args: "
    "{'model_tag': 'Qwen/Qwen3-8B', 'max_model_len': 32768, 'gpu_memory_utilization': 0.45, "
    "'kv_cache_memory_bytes': 5713920000, 'enable_prefix_caching': True}\n"
)
VLLM_019_START_LINE = (
    "(EngineCore pid=29655) INFO 09-30 16:06:26 [core.py:105] Initializing a V1 LLM engine "
    "(v0.19.1) with config: model='Qwen/Qwen3-8B', speculative_config=None\n"
)
VLLM_019_RESERVED_LINE = (
    "(EngineCore pid=29655) INFO 09-30 16:07:24 [gpu_worker.py:361] Initial free memory 77.91 GiB, "
    "reserved 5.32 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped "
    "memory profiling. This does not respect the gpu_memory_utilization config.\n"
)
VLLM_019_TOKENS_LINE = (
    "(EngineCore pid=29655) INFO 09-30 16:07:24 [kv_cache_utils.py:1319] GPU KV cache size: 38,736 tokens\n"
)
VLLM_019_BUDGETED_LOG = (
    VLLM_019_ECHO_LINE + VLLM_019_START_LINE + VLLM_019_RESERVED_LINE + VLLM_019_TOKENS_LINE
)
VLLM_019_PROFILED_LOG = (
    "(EngineCore pid=22150) INFO 09-30 15:27:10 [core.py:105] Initializing a V1 LLM engine (v0.19.1) with config: model='Qwen/Qwen3-8B'\n"
    "(EngineCore pid=22150) INFO 09-30 15:28:52 [gpu_worker.py:436] Available KV cache memory: 53.54 GiB\n"
    "(EngineCore pid=22150) INFO 09-30 15:28:52 [kv_cache_utils.py:1319] GPU KV cache size: 389,888 tokens\n"
)
VLLM_V0_LOG = (
    "INFO 07-14 worker.py:267] model weights take 15.27GiB; non_torch_memory takes "
    "0.06GiB; PyTorch activation peak memory takes 1.40GiB; the rest of the memory "
    "reserved for KV Cache is 5.33GiB.\n"
)
SGLANG_LOG = (
    "[2026-09-30 14:27:37] server_args=ServerArgs(model_path='Qwen/Qwen3-8B')\n"
    "[2026-08-18 10:00:00] KV Cache is allocated. #tokens: 430913, K size: 13.15 GB, V size: 13.15 GB\n"
    "[2026-08-18 10:00:01] max_total_num_tokens=430913, chunked_prefill_size=8192\n"
)
# SGLang logs this line after its startup warm-up generation (0.5.10.post1,
# sglang/srt/entrypoints/http_server.py); the launcher's "Server ready" waits
# for it besides /v1/models (S0F-54, live 2026-10-08).
SGLANG_READY_LINE = "[2026-08-18 10:00:02] The server is fired up and ready to roll!\n"
LMDEPLOY_017_LOG = (
    "2026-09-30 14:36:57,224 - lmdeploy - INFO - async_engine.py:130 - input backend=turbomind, "
    "backend_config=TurbomindEngineConfig(dtype='auto', session_len=32768, cache_max_entry_count=0.8763)\n"
    "Loading:   0%|          | 0/36 [00:00<?, ?it/s]\rLoading: 100%|██████████| 36/36 [00:02<00:00, 15.2it/s]\r"
    "[TM][INFO][0930.14:37:15.641964][turbomind.cc:319] Object cache budget: 55634.70 MB from free 63488.19 MB and ratio 0.876\n"
)

VLLM_BYTES = int(5.32 * GIB)
VLLM_TOKENS = 38736


def _snippet(marker: str) -> str:
    text = PREFLIGHT.read_text(encoding="utf-8")
    start = text.index(f"# {marker}")
    end = text.index("\nPY\n", start)
    return text[start:end]


# ---------------------------------------------------------------------------
# 1. the parser moved; the gate heredoc imports it (one source of truth)
# ---------------------------------------------------------------------------


def test_gate_heredoc_imports_the_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    ns: dict = {"__name__": "cage_iso_bytes_gate_under_test"}
    exec(compile(_snippet("CAGE-ISO-BYTES-GATE"), "CAGE-ISO-BYTES-GATE", "exec"), ns)
    for name in ("parse_engine_log", "compare_pair", "relative_gap", "newest_log",
                 "IsoBytesError", "main", "last_start_tail"):
        assert name in ns, name
    # The same source file, whatever module name the import used.
    assert ns["parse_engine_log"].__code__.co_filename == str(MODULE)
    assert ns["main"].__code__.co_filename == str(MODULE)
    assert "kv_pool_log" in _snippet("CAGE-ISO-BYTES-GATE")


def test_gate_heredoc_resolves_the_module_through_the_env_from_any_cwd(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Review 2026-10-06, LOW-3: exec'd in-process from a non-repo cwd the
    # resolver has only CAGE_CHECKS_DIR (argv[2] is a pytest flag).
    monkeypatch.chdir(tmp_path)
    code = compile(_snippet("CAGE-ISO-BYTES-GATE"), "CAGE-ISO-BYTES-GATE", "exec")
    monkeypatch.delenv("CAGE_CHECKS_DIR", raising=False)
    with pytest.raises(SystemExit, match="kv_pool_log.py not found"):
        exec(code, {"__name__": "cage_iso_bytes_gate_under_test"})
    monkeypatch.setenv("CAGE_CHECKS_DIR", str(CHECKS))
    ns: dict = {"__name__": "cage_iso_bytes_gate_under_test"}
    exec(code, ns)
    assert ns["main"].__code__.co_filename == str(MODULE)


def test_module_has_a_doc_header_and_no_dashes() -> None:
    text = MODULE.read_text(encoding="utf-8")
    assert "Order:" in text and "Objective:" in text and "Cloud:" in text
    assert "S0F-24" in text
    assert chr(0x2014) not in text and chr(0x2013) not in text


# ---------------------------------------------------------------------------
# 2. capture_record: the parser plus the record, with bounded retries
# ---------------------------------------------------------------------------


def test_capture_record_vllm_budgeted(tmp_path: Path) -> None:
    log = tmp_path / "vllm_Qwen_Qwen3-8B_20260930_160444.log"
    log.write_text(VLLM_019_BUDGETED_LOG, encoding="utf-8")
    rec = kp.capture_record("vllm", log, now=lambda: "2026-10-06T12:00:00Z")
    assert rec["schema"] == kp.CAPTURE_SCHEMA
    assert rec["engine"] == "vllm" and rec["role"] is None
    assert rec["log"] == str(log)
    assert rec["bytes"] == VLLM_BYTES and rec["tokens"] == VLLM_TOKENS
    assert rec["starts"] == 1 and len(rec["evidence"]) == 2
    assert rec["captured_utc"] == "2026-10-06T12:00:00Z"
    assert rec["log_mtime"] == pytest.approx(log.stat().st_mtime)


@pytest.mark.parametrize("engine,text,b,t", [
    ("sglang", SGLANG_LOG, int((13.15 + 13.15) * GIB), 430913),
    ("lmdeploy", LMDEPLOY_017_LOG, int(55634.70 * MIB), None),
])
def test_capture_record_other_engines(tmp_path: Path, engine: str, text: str, b: int, t) -> None:
    log = tmp_path / f"{engine}.log"
    log.write_bytes(text.encode("utf-8"))
    rec = kp.capture_record(engine, log, role="prefill")
    assert rec["bytes"] == b and rec["tokens"] == t
    assert rec["role"] == "prefill"
    assert rec["captured_utc"].endswith("Z")


def test_capture_record_retries_until_the_line_appears(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(VLLM_019_ECHO_LINE + VLLM_019_START_LINE, encoding="utf-8")
    sleeps: List[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:  # the engine writes its pool line during the wait
            with log.open("a", encoding="utf-8") as fh:
                fh.write(VLLM_019_RESERVED_LINE + VLLM_019_TOKENS_LINE)

    rec = kp.capture_record("vllm", log, retries=5, interval_s=0.25, sleep=sleep)
    assert rec["bytes"] == VLLM_BYTES
    assert sleeps == [0.25, 0.25]


def test_capture_record_gives_up_after_the_retries(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(VLLM_019_ECHO_LINE, encoding="utf-8")
    sleeps: List[float] = []
    with pytest.raises(kp.IsoBytesError, match="NO recognizable KV-pool line"):
        kp.capture_record("vllm", log, retries=3, interval_s=0.5, sleep=sleeps.append)
    assert sleeps == [0.5, 0.5, 0.5]


def test_capture_record_corrupted_line_never_retries(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(VLLM_019_RESERVED_LINE.replace("5.32 GiB", "??? GiB"), encoding="utf-8")
    sleeps: List[float] = []
    with pytest.raises(kp.IsoBytesError, match="corrupted"):
        kp.capture_record("vllm", log, retries=3, interval_s=0.5, sleep=sleeps.append)
    assert sleeps == []


def test_capture_record_missing_log_is_loud(tmp_path: Path) -> None:
    with pytest.raises(kp.IsoBytesError, match="does not exist"):
        kp.capture_record("vllm", tmp_path / "absent.log", retries=0)


# ---------------------------------------------------------------------------
# 3. the record files: atomic write, current-record resolution, the merge
# ---------------------------------------------------------------------------


def _vllm_record(tmp_path: Path, utc: str, **over) -> dict:
    log = tmp_path / "vllm_single.log"
    log.write_text(VLLM_019_BUDGETED_LOG, encoding="utf-8")
    rec = kp.capture_record("vllm", log, now=lambda: utc)
    rec.update(over)
    return rec


def test_write_record_is_atomic_and_round_trips(tmp_path: Path) -> None:
    rec = _vllm_record(tmp_path, "2026-10-06T12:00:00Z")
    out = tmp_path / "vllm" / kp.CURRENT_NAME
    assert kp.write_record(out, rec) == out
    assert json.loads(out.read_text(encoding="utf-8")) == rec
    assert not list((tmp_path / "vllm").glob("*.tmp*"))


def test_read_current_none_single_pd_and_newest_wins(tmp_path: Path) -> None:
    root = tmp_path / "logs"
    assert kp.read_current(root, "vllm") is None
    single = _vllm_record(tmp_path, "2026-10-06T12:00:00Z")
    kp.write_record(root / "vllm" / kp.CURRENT_NAME, single)
    got = kp.read_current(root, "vllm")
    assert got is not None and got["record"] == single
    assert got["path"] == root / "vllm" / kp.CURRENT_NAME
    # a pd record captured LATER wins
    prefill = _vllm_record(tmp_path, "2026-10-06T12:05:00Z", role="prefill")
    decode = _vllm_record(tmp_path, "2026-10-06T12:05:00Z", role="decode")
    pd = kp.pd_record("vllm", {"prefill": prefill, "decode": decode},
                      now=lambda: "2026-10-06T12:05:01Z")
    assert pd["schema"] == kp.CAPTURE_SCHEMA and pd["topology"] == "pd"
    kp.write_record(root / "vllm" / kp.CURRENT_PD_NAME, pd)
    got = kp.read_current(root, "vllm")
    assert got["record"]["topology"] == "pd"
    assert set(got["record"]["roles"]) == {"prefill", "decode"}
    # a single record captured later than the pd one wins again
    newer = _vllm_record(tmp_path, "2026-10-06T12:09:00Z")
    kp.write_record(root / "vllm" / kp.CURRENT_NAME, newer)
    assert kp.read_current(root, "vllm")["record"] == newer


def test_read_current_equal_timestamps_break_the_tie_by_mtime(tmp_path: Path) -> None:
    # Review 2026-10-06, LOW-6: captured_utc has second resolution.
    root = tmp_path / "logs"
    single = _vllm_record(tmp_path, "2026-10-06T12:00:00Z")
    kp.write_record(root / "vllm" / kp.CURRENT_NAME, single)
    pd = kp.pd_record("vllm", {"prefill": _vllm_record(tmp_path, "2026-10-06T12:00:00Z", role="prefill")},
                      now=lambda: "2026-10-06T12:00:00Z")
    kp.write_record(root / "vllm" / kp.CURRENT_PD_NAME, pd)
    os.utime(root / "vllm" / kp.CURRENT_NAME, (1_700_000_000, 1_700_000_000))
    os.utime(root / "vllm" / kp.CURRENT_PD_NAME, (1_700_000_005, 1_700_000_005))
    assert kp.read_current(root, "vllm")["record"]["topology"] == "pd"
    os.utime(root / "vllm" / kp.CURRENT_NAME, (1_700_000_009, 1_700_000_009))
    assert kp.read_current(root, "vllm")["record"] == single


def test_read_current_corrupt_or_foreign_record_is_loud(tmp_path: Path) -> None:
    root = tmp_path / "logs"
    path = root / "vllm" / kp.CURRENT_NAME
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(kp.IsoBytesError, match=str(path)):
        kp.read_current(root, "vllm")
    path.write_text(json.dumps({"schema": "something-else", "engine": "vllm"}), encoding="utf-8")
    with pytest.raises(kp.IsoBytesError, match="schema"):
        kp.read_current(root, "vllm")
    rec = _vllm_record(tmp_path, "2026-10-06T12:00:00Z", engine="sglang")
    kp.write_record(path, rec)
    with pytest.raises(kp.IsoBytesError, match="engine"):
        kp.read_current(root, "vllm")


def test_merge_into_serving_config_adds_the_realized_pool(tmp_path: Path) -> None:
    cfg = tmp_path / "20261006T120000Z_qwen3-8b.json"
    cfg.write_text(json.dumps({"model": "Qwen/Qwen3-8B", "kv_budget_bytes": 5713920000}),
                   encoding="utf-8")
    rec = _vllm_record(tmp_path, "2026-10-06T12:00:30Z")
    assert kp.merge_into_serving_config(cfg, rec) is True
    doc = json.loads(cfg.read_text(encoding="utf-8"))
    assert doc["model"] == "Qwen/Qwen3-8B" and doc["kv_budget_bytes"] == 5713920000
    assert doc["kv_pool_bytes_realized"] == VLLM_BYTES
    assert doc["kv_pool_tokens_realized"] == VLLM_TOKENS
    assert doc["kv_pool_evidence"] == rec["evidence"]
    assert doc["kv_pool_log"] == rec["log"]
    assert doc["kv_pool_captured_utc"] == "2026-10-06T12:00:30Z"
    assert not list(tmp_path.glob("*.tmp*"))


def test_merge_into_serving_config_missing_or_invalid_file_is_false(tmp_path: Path, capsys) -> None:
    rec = _vllm_record(tmp_path, "2026-10-06T12:00:30Z")
    assert kp.merge_into_serving_config(tmp_path / "absent.json", rec) is False
    assert not (tmp_path / "absent.json").exists()
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert kp.merge_into_serving_config(bad, rec) is False
    assert bad.read_text(encoding="utf-8") == "{"
    out = capsys.readouterr().out
    assert "absent.json" in out and "bad.json" in out


# ---------------------------------------------------------------------------
# 4. the CLI the launchers call
# ---------------------------------------------------------------------------


def _cli(*argv: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MODULE), *argv], capture_output=True, text=True,
        cwd=str(cwd or REPO_ROOT), timeout=60,
    )


def test_cli_capture_single_writes_the_record_and_merges(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(VLLM_019_BUDGETED_LOG, encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"model": "m"}), encoding="utf-8")
    out = tmp_path / "logs" / "vllm" / kp.CURRENT_NAME
    proc = _cli("capture", "--engine", "vllm", "--log", str(log), "--out", str(out),
                "--merge-into", str(cfg), "--retries", "0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "KV pool captured" in proc.stdout and "5.320 GiB" in proc.stdout
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["bytes"] == VLLM_BYTES and rec["role"] is None
    assert json.loads(cfg.read_text(encoding="utf-8"))["kv_pool_bytes_realized"] == VLLM_BYTES


def test_cli_capture_no_pool_line_exits_3_and_writes_nothing(tmp_path: Path) -> None:
    log = tmp_path / "vllm.log"
    log.write_text(VLLM_019_ECHO_LINE, encoding="utf-8")
    out = tmp_path / kp.CURRENT_NAME
    proc = _cli("capture", "--engine", "vllm", "--log", str(log), "--out", str(out),
                "--retries", "1", "--interval", "0")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "NO recognizable KV-pool line" in proc.stdout + proc.stderr
    assert not out.exists()


def test_cli_capture_pd_roles_write_one_record(tmp_path: Path) -> None:
    prefill = tmp_path / "vllm_pd_prefill.log"
    decode = tmp_path / "vllm_pd_decode.log"
    prefill.write_text(VLLM_019_BUDGETED_LOG, encoding="utf-8")
    decode.write_text(VLLM_019_BUDGETED_LOG.replace("pid=29655", "pid=29656"), encoding="utf-8")
    cfg_p = tmp_path / "p.json"
    cfg_d = tmp_path / "d.json"
    for p in (cfg_p, cfg_d):
        p.write_text(json.dumps({"topology": "pd"}), encoding="utf-8")
    out = tmp_path / kp.CURRENT_PD_NAME
    proc = _cli("capture", "--engine", "vllm", "--role", f"prefill={prefill}",
                "--role", f"decode={decode}", "--out", str(out),
                "--merge-into", f"prefill={cfg_p}", "--merge-into", f"decode={cfg_d}",
                "--retries", "0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["topology"] == "pd" and rec["engine"] == "vllm"
    assert rec["roles"]["prefill"]["bytes"] == VLLM_BYTES
    assert rec["roles"]["decode"]["log"] == str(decode)
    for p in (cfg_p, cfg_d):
        assert json.loads(p.read_text(encoding="utf-8"))["kv_pool_bytes_realized"] == VLLM_BYTES


def test_cli_capture_pd_partial_failure_exits_3_and_writes_nothing(tmp_path: Path) -> None:
    prefill = tmp_path / "p.log"
    decode = tmp_path / "d.log"
    prefill.write_text(VLLM_019_BUDGETED_LOG, encoding="utf-8")
    decode.write_text(VLLM_019_ECHO_LINE, encoding="utf-8")  # no pool line
    out = tmp_path / kp.CURRENT_PD_NAME
    proc = _cli("capture", "--engine", "vllm", "--role", f"prefill={prefill}",
                "--role", f"decode={decode}", "--out", str(out), "--retries", "0")
    assert proc.returncode == 3
    assert "decode" in proc.stdout + proc.stderr
    assert not out.exists()


@pytest.mark.parametrize("argv", [
    ("capture", "--engine", "vllm", "--log", "a.log"),                       # no --out
    ("capture", "--engine", "vllm", "--out", "o.json"),                      # no log, no role
    ("capture", "--engine", "vllm", "--log", "a.log", "--role", "p=b.log", "--out", "o.json"),
    ("capture", "--engine", "vllm", "--role", "noequals", "--out", "o.json"),
    ("capture", "--engine", "vllm", "--role", "p=a.log", "--role", "p=b.log", "--out", "o.json"),
    ("nonsense",),
])
def test_cli_usage_errors_exit_2(argv) -> None:
    proc = _cli(*argv)
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_cli_single_merge_into_path_may_contain_equals(tmp_path: Path) -> None:
    # Review 2026-10-06, LOW-5: a hand-typed CAGE_RUN_ROOT can carry "=".
    log = tmp_path / "vllm.log"
    log.write_text(VLLM_019_BUDGETED_LOG, encoding="utf-8")
    cfg_dir = tmp_path / "run=root" / "observability"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "CURRENT.json"
    cfg.write_text(json.dumps({"engine": "vllm"}), encoding="utf-8")
    proc = _cli("capture", "--engine", "vllm", "--log", str(log),
                "--out", str(tmp_path / "rec.json"), "--merge-into", str(cfg), "--retries", "0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(cfg.read_text(encoding="utf-8"))["kv_pool_bytes_realized"] == VLLM_BYTES


def test_cli_failed_capture_removes_a_stale_record(tmp_path: Path) -> None:
    # Review 2026-10-06, HIGH-1: a record from an earlier start must not
    # survive a capture that failed, or gate (j) certifies the old pool.
    out = tmp_path / "rec.json"
    kp.write_record(out, _vllm_record(tmp_path, "2026-10-06T11:00:00Z"))
    log = tmp_path / "vllm.log"
    log.write_text("no pool line\n", encoding="utf-8")
    proc = _cli("capture", "--engine", "vllm", "--log", str(log), "--out", str(out), "--retries", "0")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert not out.exists()
    assert "removed the stale launcher capture" in proc.stdout


# ---------------------------------------------------------------------------
# 5. gate (j): pins, then the current record, then the newest log
# ---------------------------------------------------------------------------


def _gate_env(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("CAGE_ISO_BYTES_LOG_ROOT", str(root))
    for var in ("CAGE_ISO_BYTES_TOL", "CAGE_ISO_BYTES_LOGS", "CAGE_ISO_POOL_SUM_BYTES"):
        monkeypatch.delenv(var, raising=False)


def _write_log(root: Path, engine: str, name: str, text: str, mtime: int) -> Path:
    d = root / engine
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(text, encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


def test_gate_prefers_the_current_record_over_a_newer_log(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    root = tmp_path / "logs"
    budgeted = _write_log(root, "vllm", "vllm_a.log", VLLM_019_BUDGETED_LOG, 1_700_000_000)
    _write_log(root, "vllm", "vllm_b.log", VLLM_019_PROFILED_LOG, 1_700_000_900)  # newer, other pool
    rec = kp.capture_record("vllm", budgeted, now=lambda: "2026-10-06T12:00:00Z")
    kp.write_record(root / "vllm" / kp.CURRENT_NAME, rec)
    _gate_env(monkeypatch, root)
    rc = kp.main(["gate", "vllm"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "[pool] vllm: bytes=5.320 GiB tokens=38736" in out
    assert f"capture={root / 'vllm' / kp.CURRENT_NAME}" in out
    assert "captured=2026-10-06T12:00:00Z" in out and f"log={budgeted}" in out
    assert "53.540" not in out


def test_gate_explicit_pin_wins_over_the_record(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    root = tmp_path / "logs"
    budgeted = _write_log(root, "vllm", "vllm_a.log", VLLM_019_BUDGETED_LOG, 1_700_000_000)
    kp.write_record(root / "vllm" / kp.CURRENT_NAME, kp.capture_record("vllm", budgeted))
    pinned = tmp_path / "pinned.log"
    pinned.write_text(VLLM_V0_LOG, encoding="utf-8")
    _gate_env(monkeypatch, root)
    monkeypatch.setenv("CAGE_ISO_BYTES_LOGS", f"vllm={pinned}")
    rc = kp.main(["gate", "vllm"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"log={pinned}" in out and "bytes=5.330 GiB" in out
    assert "capture=" not in out


def test_gate_without_a_record_reads_the_newest_log_as_before(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    root = tmp_path / "logs"
    newest = _write_log(root, "vllm", "vllm_b.log", VLLM_019_PROFILED_LOG, 1_700_000_900)
    _gate_env(monkeypatch, root)
    rc = kp.main(["gate", "vllm"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out == (
        f"  [pool] vllm: bytes=53.540 GiB tokens=389888 "
        f"log={newest} mtime={newest.stat().st_mtime:.0f}\n"
        "  [note] single-engine scope ('vllm'): pairwise parity is "
        "vacuous; realized pool recorded above. The FULL §6.5 gate "
        "needs all final-scope engines in CAGE_PREFLIGHT_BACKENDS.\n"
        "  [PASS] iso-bytes gate (vacuous parity; realized pool recorded)\n"
    )


def test_gate_pd_record_enters_as_the_role_sum(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    root = tmp_path / "logs"
    p = _write_log(root, "vllm", "vllm_pd_prefill.log", VLLM_019_BUDGETED_LOG, 1_700_000_000)
    d = _write_log(root, "vllm", "vllm_pd_decode.log",
                   VLLM_019_BUDGETED_LOG.replace("pid=29655", "pid=29656"), 1_700_000_001)
    pd = kp.pd_record("vllm", {
        "prefill": kp.capture_record("vllm", p, role="prefill"),
        "decode": kp.capture_record("vllm", d, role="decode"),
    })
    kp.write_record(root / "vllm" / kp.CURRENT_PD_NAME, pd)
    _gate_env(monkeypatch, root)
    monkeypatch.setenv("CAGE_ISO_POOL_SUM_BYTES", "11427840000")
    rc = kp.main(["gate", "vllm"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "[pool] vllm:prefill: bytes=5.320 GiB" in out
    assert "[pool] vllm:decode: bytes=5.320 GiB" in out
    assert "[pool] vllm: SUM of 2 role pools" in out
    assert "[PASS] vllm: pool SUM" in out


POOL_SUM_OPERATOR_ERROR = (
    "  [FAIL] CAGE_ISO_POOL_SUM_BYTES is set but CAGE_ISO_BYTES_LOGS pins no "
    "role-split (engine:role) log in scope -- a declared pool sum with nothing "
    "to sum is operator error; drop the env or pin the role logs"
)


def test_gate_pool_sum_without_roles_or_record_fails_before_any_log_as_before(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # Review 2026-10-06, MEDIUM-2: the legacy operator-error path keeps its
    # one-line stdout (no [pool] lines first) and the legacy wording.
    root = tmp_path / "logs"
    _write_log(root, "vllm", "vllm_a.log", VLLM_019_BUDGETED_LOG, 1_700_000_000)
    _gate_env(monkeypatch, root)
    monkeypatch.setenv("CAGE_ISO_POOL_SUM_BYTES", "11427840000")
    rc = kp.main(["gate", "vllm"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.rstrip("\n").splitlines()[-1] == POOL_SUM_OPERATOR_ERROR
    assert "[pool]" not in out


def test_gate_pool_sum_with_role_pins_and_a_bad_role_log_names_the_log_only(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # Review 2026-10-06, MEDIUM-2: role pins ARE in scope; a failed role parse
    # must not be reported as "pins no role-split log".
    root = tmp_path / "logs"
    p = _write_log(root, "vllm", "vllm_pd_prefill.log", VLLM_019_BUDGETED_LOG, 1_700_000_000)
    d = _write_log(root, "vllm", "vllm_pd_decode.log", VLLM_019_START_LINE, 1_700_000_001)
    _gate_env(monkeypatch, root)
    monkeypatch.setenv("CAGE_ISO_BYTES_LOGS", f"vllm:prefill={p},vllm:decode={d}")
    monkeypatch.setenv("CAGE_ISO_POOL_SUM_BYTES", "11427840000")
    rc = kp.main(["gate", "vllm"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "pins no role-split" not in out
    assert "NO recognizable KV-pool line" in out and str(d) in out


def test_gate_corrupt_record_fails_naming_the_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    root = tmp_path / "logs"
    _write_log(root, "vllm", "vllm_a.log", VLLM_019_BUDGETED_LOG, 1_700_000_000)
    bad = root / "vllm" / kp.CURRENT_NAME
    bad.write_text("{oops", encoding="utf-8")
    _gate_env(monkeypatch, root)
    rc = kp.main(["gate", "vllm"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] vllm:" in out and str(bad) in out


def test_gate_record_parity_across_engines(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    root = tmp_path / "logs"
    v = _write_log(root, "vllm", "vllm.log", VLLM_019_PROFILED_LOG, 1_700_000_000)
    (root / "lmdeploy").mkdir(parents=True)
    lm = root / "lmdeploy" / "lmdeploy.log"
    lm.write_bytes(LMDEPLOY_017_LOG.encode("utf-8"))
    kp.write_record(root / "vllm" / kp.CURRENT_NAME, kp.capture_record("vllm", v))
    kp.write_record(root / "lmdeploy" / kp.CURRENT_NAME, kp.capture_record("lmdeploy", lm))
    _gate_env(monkeypatch, root)
    rc = kp.main(["gate", "vllm,lmdeploy"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.count("capture=") == 2
    assert "[PASS] vllm vs lmdeploy: bytes gap 0.01" in out


# ---------------------------------------------------------------------------
# 6. the launchers: capture at the ready line, remove on stop
# ---------------------------------------------------------------------------


def _clean_env(**extra: str) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CAGE_", "VLLM_", "SGLANG_", "LMDEPLOY_", "PD_"))}
    env.update(extra)
    return env


def _stub(d: Path, name: str, body: str) -> Path:
    p = d / name
    p.write_text(body, encoding="utf-8")
    p.chmod(0o755)
    return p


@pytest.fixture()
def stub_bin(tmp_path: Path) -> Path:
    """pgrep finds nothing, pkill is inert, curl answers /v1/models with the
    requested model (only once CAGE_TEST_READY_WHEN_LOGGED appears in a log
    under CAGE_LOG_ROOT, so readiness never races the engine stub's write)
    and 0 elsewhere, nvidia-smi answers only the total-memory query (the
    LMDeploy dial mapping); every engine stub prints CAGE_TEST_POOL_LINES,
    which the launcher redirects into the start log."""
    d = tmp_path / "stub_bin"
    d.mkdir()
    _stub(d, "pgrep", "#!/bin/sh\nexit 1\n")
    _stub(d, "pkill", "#!/bin/sh\nexit 0\n")
    _stub(d, "curl", (
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *v1/models*)\n"
        '    if [ -n "${CAGE_TEST_READY_WHEN_LOGGED:-}" ]; then\n'
        '      grep -q "$CAGE_TEST_READY_WHEN_LOGGED" "$CAGE_LOG_ROOT"/*/*.log 2>/dev/null || exit 1\n'
        "    fi\n"
        '    printf \'{"data":[{"id":"%s"}]}\\n\' "${CAGE_TEST_MODEL:-fake/test-model}" ;;\n'
        "  *) exit 0 ;;\n"
        "esac\n"
    ))
    _stub(d, "nvidia-smi", (
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *memory.total*) echo 81559 ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n"
    ))
    _stub(d, "vllm", "#!/bin/sh\nprintf '%b' \"${CAGE_TEST_POOL_LINES:-}\"\nexit 0\n")
    return d


def _engine_stub(d: Path, name: str) -> Path:
    """An interpreter or entry point that answers version probes, import
    probes, and prints the pool lines when it is asked to serve."""
    return _stub(d, name, (
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *launch_server*|*api_server*) printf '%b' \"${CAGE_TEST_POOL_LINES:-}\"; exit 0 ;;\n"
        "  *__version__*) echo 0.0-stub ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    ))


def _run_launcher(script: str, stub_bin: Path, log_root: Path, *args: str, **env_extra: str) -> subprocess.CompletedProcess:
    env = _clean_env(**env_extra)
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    env["CAGE_LOG_ROOT"] = str(log_root)
    return subprocess.run(["bash", str(SERVING / script), *args],
                          capture_output=True, text=True, env=env, timeout=120)


def test_vllm_launcher_captures_at_ready_and_removes_on_stop(stub_bin: Path, tmp_path: Path) -> None:
    log_root = tmp_path / "logroot"
    proc = _run_launcher(
        "manage_vllm_server.sh", stub_bin, log_root, "start", "fake/test-model",
        VLLM_START_TIMEOUT="10", CAGE_TEST_POOL_LINES=VLLM_019_BUDGETED_LOG,
        CAGE_TEST_READY_WHEN_LOGGED="GPU KV cache size",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Server ready with model: fake/test-model" in proc.stdout
    assert "KV pool captured" in proc.stdout
    record = log_root / "vllm" / kp.CURRENT_NAME
    rec = json.loads(record.read_text(encoding="utf-8"))
    assert rec["engine"] == "vllm" and rec["bytes"] == VLLM_BYTES
    assert rec["log"].startswith(str(log_root / "vllm" / "vllm_fake_test-model_"))
    # The single launcher's stop sweeps every vllm serve, so it removes the pd
    # record too (missing test 5 of the review).
    pd_record = log_root / "vllm" / kp.CURRENT_PD_NAME
    pd_record.write_text("{}", encoding="utf-8")
    proc = _run_launcher("manage_vllm_server.sh", stub_bin, log_root, "stop")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not record.exists() and not pd_record.exists()


def test_vllm_restart_after_a_crash_never_certifies_the_earlier_start(
        stub_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # Review 2026-10-06, HIGH-1 (the reviewer's counterexample): start with a
    # pool line, "crash" (no stop), start again with no pool line and a
    # capture that fails. The old record must be gone and gate (j) must fall
    # back to the newest log and FAIL, as it did before S0F-24.
    log_root = tmp_path / "logroot"
    stale_pd = log_root / "vllm" / kp.CURRENT_PD_NAME
    stale_pd.parent.mkdir(parents=True)
    stale_pd.write_text("{}", encoding="utf-8")
    proc = _run_launcher(
        "manage_vllm_server.sh", stub_bin, log_root, "start", "fake/test-model",
        VLLM_START_TIMEOUT="10", CAGE_TEST_POOL_LINES=VLLM_019_BUDGETED_LOG,
        CAGE_TEST_READY_WHEN_LOGGED="GPU KV cache size",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = log_root / "vllm" / kp.CURRENT_NAME
    assert record.exists() and not stale_pd.exists()  # a start removes records before it spawns
    proc = _run_launcher(
        "manage_vllm_server.sh", stub_bin, log_root, "start", "fake/test-model",
        VLLM_START_TIMEOUT="10", CAGE_TEST_POOL_LINES="no pool line here\n",
        CAGE_KV_POOL_CAPTURE_RETRIES="0",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not record.exists()
    _gate_env(monkeypatch, log_root)
    rc = kp.main(["gate", "vllm"])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "capture=" not in out
    assert "NO recognizable KV-pool line" in out


def test_vllm_launcher_start_survives_a_failed_capture(stub_bin: Path, tmp_path: Path) -> None:
    log_root = tmp_path / "logroot"
    proc = _run_launcher(
        "manage_vllm_server.sh", stub_bin, log_root, "start", "fake/test-model",
        VLLM_START_TIMEOUT="10", CAGE_TEST_POOL_LINES="no pool line here\n",
        CAGE_KV_POOL_CAPTURE_RETRIES="0",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Server ready" in proc.stdout
    assert "KV pool capture failed" in proc.stdout + proc.stderr
    assert not (log_root / "vllm" / kp.CURRENT_NAME).exists()


def test_vllm_launcher_merges_into_the_serving_config(stub_bin: Path, tmp_path: Path) -> None:
    log_root = tmp_path / "logroot"
    run_root = tmp_path / "run"
    proc = _run_launcher(
        "manage_vllm_server.sh", stub_bin, log_root, "start", "fake/test-model",
        VLLM_START_TIMEOUT="10", CAGE_TEST_POOL_LINES=VLLM_019_BUDGETED_LOG,
        CAGE_TEST_READY_WHEN_LOGGED="GPU KV cache size", CAGE_RUN_ROOT=str(run_root),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cfgs = sorted((run_root / "observability" / "serving_configs").glob("*.json"))
    assert len(cfgs) == 1
    doc = json.loads(cfgs[0].read_text(encoding="utf-8"))
    assert doc["kv_pool_bytes_realized"] == VLLM_BYTES
    assert doc["args"].startswith("vllm serve fake/test-model")


def test_sglang_launcher_captures_at_ready_and_removes_on_stop(stub_bin: Path, tmp_path: Path) -> None:
    log_root = tmp_path / "logroot"
    python_stub = _engine_stub(stub_bin, "sglang-python")
    proc = _run_launcher(
        "manage_sglang_server.sh", stub_bin, log_root, "start", "fake/test-model",
        SGLANG_START_TIMEOUT="10", CAGE_SGLANG_PYTHON=str(python_stub),
        CAGE_TEST_POOL_LINES=SGLANG_LOG + SGLANG_READY_LINE, CAGE_TEST_READY_WHEN_LOGGED="KV Cache is allocated",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Server ready with model: fake/test-model" in proc.stdout
    assert "KV pool captured" in proc.stdout
    record = log_root / "sglang" / kp.CURRENT_NAME
    rec = json.loads(record.read_text(encoding="utf-8"))
    assert rec["engine"] == "sglang" and rec["tokens"] == 430913
    proc = _run_launcher("manage_sglang_server.sh", stub_bin, log_root, "stop")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not record.exists()


def _sg_log(tokens: int) -> str:
    return (
        f"[2026-08-18 10:00:00] KV Cache is allocated. #tokens: {tokens}, K size: 13.15 GB, V size: 13.15 GB\n"
        f"[2026-08-18 10:00:01] max_total_num_tokens={tokens}, chunked_prefill_size=8192\n"
    )


def _fraction_aware_sglang(d: Path) -> Path:
    """A fake SGLang that prints CAGE_TEST_SHORT_LINES at the base fraction 0.90
    and CAGE_TEST_FULL_LINES at any other fraction (the raised one)."""
    return _stub(d, "sglang-python-frac", (
        "#!/bin/sh\n"
        'case "$*" in\n'
        '  *launch_server*) case "$*" in *"--mem-fraction-static 0.90"*) printf "%b" "${CAGE_TEST_SHORT_LINES:-}" ;; *) printf "%b" "${CAGE_TEST_FULL_LINES:-}" ;; esac; exit 0 ;;\n'
        "  *__version__*) echo 0.0-stub ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n"
    ))


def test_sglang_launcher_raises_the_fraction_once_when_the_cap_exceeds_the_pool(stub_bin: Path, tmp_path: Path) -> None:
    # ADR-0162 (S0 attempt 2): --max-total-tokens 298,687 against a 282,583-token
    # pool at 0.90 [V 2026-10-09]. The launcher reads the realized pool, restarts
    # once at a raised fraction and the second start holds the cap.
    log_root = tmp_path / "logroot"
    python_stub = _fraction_aware_sglang(stub_bin)
    proc = _run_launcher(
        "manage_sglang_server.sh", stub_bin, log_root, "start", "fake/test-model",
        SGLANG_START_TIMEOUT="10", CAGE_SGLANG_PYTHON=str(python_stub),
        CAGE_SGLANG_MAX_TOTAL_TOKENS="298687",
        CAGE_TEST_SHORT_LINES=_sg_log(282583) + SGLANG_READY_LINE,
        CAGE_TEST_FULL_LINES=_sg_log(300000) + SGLANG_READY_LINE,
        CAGE_TEST_READY_WHEN_LOGGED="KV Cache is allocated",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "KV pool short of --max-total-tokens 298687 at --mem-fraction-static 0.90" in proc.stdout
    assert "realized 282583 tokens" in proc.stdout
    m = re.search(r"restarting once at (0\.\d+) \(ADR-0162\)", proc.stdout)
    assert m, proc.stdout
    # 0.90 + 16,104 tokens x (26.3 GiB / 282,583) / 81,559 MiB + 0.005 = 0.924 [D]
    assert 0.92 <= float(m.group(1)) <= 0.93, m.group(1)
    assert proc.stdout.count("Server ready with model: fake/test-model") == 2
    assert proc.stdout.count("Server stopped") == 1  # the short start was stopped before the retry
    rec = json.loads((log_root / "sglang" / kp.CURRENT_NAME).read_text(encoding="utf-8"))
    assert rec["engine"] == "sglang" and rec["tokens"] == 300000
    # the start that serves was composed with the raised fraction
    assert f"--mem-fraction-static {m.group(1)}" in proc.stdout


def test_sglang_launcher_refuses_when_the_raised_fraction_is_still_short(stub_bin: Path, tmp_path: Path) -> None:
    log_root = tmp_path / "logroot"
    python_stub = _fraction_aware_sglang(stub_bin)
    proc = _run_launcher(
        "manage_sglang_server.sh", stub_bin, log_root, "start", "fake/test-model",
        SGLANG_START_TIMEOUT="10", CAGE_SGLANG_PYTHON=str(python_stub),
        CAGE_SGLANG_MAX_TOTAL_TOKENS="298687",
        CAGE_TEST_SHORT_LINES=_sg_log(282583) + SGLANG_READY_LINE,
        CAGE_TEST_FULL_LINES=_sg_log(290000) + SGLANG_READY_LINE,
        CAGE_TEST_READY_WHEN_LOGGED="KV Cache is allocated",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "KV pool still short of --max-total-tokens 298687" in proc.stdout
    assert "refusing to serve a smaller pool than planned" in proc.stdout
    assert proc.stdout.count("Server stopped") == 2
    assert not (log_root / "sglang" / kp.CURRENT_NAME).exists()
    assert not (log_root / "sglang" / "sglang_server.pid").exists()


def test_sglang_launcher_refuses_at_once_when_no_higher_fraction_exists(stub_bin: Path, tmp_path: Path) -> None:
    # Review 2026-10-10 (LOW 3): at the fraction cap already, a retry would
    # repeat the identical start; the launcher refuses and stops the short server.
    log_root = tmp_path / "logroot"
    python_stub = _fraction_aware_sglang(stub_bin)
    proc = _run_launcher(
        "manage_sglang_server.sh", stub_bin, log_root, "start", "fake/test-model",
        SGLANG_START_TIMEOUT="10", CAGE_SGLANG_PYTHON=str(python_stub),
        CAGE_SGLANG_MAX_TOTAL_TOKENS="298687", VLLM_GPU_MEMORY_UTILIZATION="0.95",
        CAGE_TEST_SHORT_LINES=_sg_log(282583) + SGLANG_READY_LINE,
        CAGE_TEST_FULL_LINES=_sg_log(282583) + SGLANG_READY_LINE,
        CAGE_TEST_READY_WHEN_LOGGED="KV Cache is allocated",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "no higher fraction is available" in proc.stdout
    assert "restarting once" not in proc.stdout
    assert proc.stdout.count("Server args:") == 1 and proc.stdout.count("Server stopped") == 1
    assert not (log_root / "sglang" / "sglang_server.pid").exists()
    # a malformed cap knob is reported and the computation falls back instead of aborting
    proc = _run_launcher(
        "manage_sglang_server.sh", stub_bin, log_root, "start", "fake/test-model",
        SGLANG_START_TIMEOUT="10", CAGE_SGLANG_PYTHON=str(python_stub),
        CAGE_SGLANG_MAX_TOTAL_TOKENS="298687", CAGE_SGLANG_MEM_FRACTION_MAX="lots",
        CAGE_TEST_SHORT_LINES=_sg_log(282583) + SGLANG_READY_LINE,
        CAGE_TEST_FULL_LINES=_sg_log(300000) + SGLANG_READY_LINE,
        CAGE_TEST_READY_WHEN_LOGGED="KV Cache is allocated",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "is not a fraction in (0, 1]; using 0.95" in proc.stderr
    assert "restarting once at" in proc.stdout


def test_sglang_launcher_does_not_retry_a_start_that_never_got_ready(stub_bin: Path, tmp_path: Path) -> None:
    # A start that never gets ready has an unknown cause; it fails as before
    # (one timeout, one stop), never a blind retry at a raised fraction.
    log_root = tmp_path / "logroot"
    python_stub = _fraction_aware_sglang(stub_bin)
    proc = _run_launcher(
        "manage_sglang_server.sh", stub_bin, log_root, "start", "fake/test-model",
        SGLANG_START_TIMEOUT="4", CAGE_SGLANG_PYTHON=str(python_stub),
        CAGE_SGLANG_MAX_TOTAL_TOKENS="298687",
        CAGE_TEST_SHORT_LINES="no pool, no ready line\n",
        CAGE_TEST_FULL_LINES=_sg_log(300000) + SGLANG_READY_LINE,
        CAGE_TEST_READY_WHEN_LOGGED="KV Cache is allocated",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "failed to start within 4s" in proc.stdout
    assert "restarting once" not in proc.stdout
    assert proc.stdout.count("Server args:") == 1 and proc.stdout.count("Server stopped") == 1


def test_sglang_launcher_does_not_restart_when_the_pool_holds_the_cap(stub_bin: Path, tmp_path: Path) -> None:
    log_root = tmp_path / "logroot"
    python_stub = _engine_stub(stub_bin, "sglang-python")
    proc = _run_launcher(
        "manage_sglang_server.sh", stub_bin, log_root, "start", "fake/test-model",
        SGLANG_START_TIMEOUT="10", CAGE_SGLANG_PYTHON=str(python_stub),
        CAGE_SGLANG_MAX_TOTAL_TOKENS="400000",
        CAGE_TEST_POOL_LINES=SGLANG_LOG + SGLANG_READY_LINE, CAGE_TEST_READY_WHEN_LOGGED="KV Cache is allocated",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "restarting once" not in proc.stdout and "KV pool short" not in proc.stdout
    assert proc.stdout.count("Server ready with model: fake/test-model") == 1


def test_sglang_launcher_is_not_ready_before_sglangs_own_ready_line(stub_bin: Path, tmp_path: Path) -> None:
    # S0F-54 (live 2026-10-08): /v1/models answered two seconds after uvicorn
    # came up while SGLang's startup warm-up generation still ran, and the
    # flush posted in that window was refused. "Server ready" needs SGLang's
    # own ready line in the start log too; a log without it times out loudly,
    # names the missing line, captures nothing, and stops the server it spawned
    # (review LOW 1: a later start would otherwise reuse it unchecked).
    log_root = tmp_path / "logroot"
    python_stub = _engine_stub(stub_bin, "sglang-python")
    proc = _run_launcher(
        "manage_sglang_server.sh", stub_bin, log_root, "start", "fake/test-model",
        SGLANG_START_TIMEOUT="6", CAGE_SGLANG_PYTHON=str(python_stub),
        CAGE_TEST_POOL_LINES=SGLANG_LOG, CAGE_TEST_READY_WHEN_LOGGED="KV Cache is allocated",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Server ready" not in proc.stdout
    assert "failed to start within 6s" in proc.stdout
    assert "The server is fired up and ready to roll!" in proc.stdout
    assert not (log_root / "sglang" / kp.CURRENT_NAME).exists()
    assert "Server stopped" in proc.stdout
    assert not (log_root / "sglang" / "sglang_server.pid").exists()


def test_lmdeploy_launcher_captures_at_ready_and_removes_on_stop(stub_bin: Path, tmp_path: Path) -> None:
    log_root = tmp_path / "logroot"
    entry = _engine_stub(stub_bin, "lmdeploy-stub")
    proc = _run_launcher(
        "manage_lmdeploy_server.sh", stub_bin, log_root, "start", "fake/test-model",
        LMDEPLOY_START_TIMEOUT="10", CAGE_LMDEPLOY_BIN=str(entry),
        CAGE_MODEL_WEIGHTS_GIB="16", CAGE_TEST_POOL_LINES=LMDEPLOY_017_LOG,
        CAGE_TEST_READY_WHEN_LOGGED="Object cache budget",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TurboMind backend confirmed" in proc.stdout
    assert "KV pool captured" in proc.stdout
    record = log_root / "lmdeploy" / kp.CURRENT_NAME
    rec = json.loads(record.read_text(encoding="utf-8"))
    assert rec["engine"] == "lmdeploy" and rec["bytes"] == int(55634.70 * MIB)
    proc = _run_launcher("manage_lmdeploy_server.sh", stub_bin, log_root, "stop")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not record.exists()


def test_pd_launcher_capture_function_and_stop(stub_bin: Path, tmp_path: Path) -> None:
    """The pd launcher is sourced (define only, never dispatch) and its
    capture call is driven directly: both roles enter one record, and
    stop_stack removes the pd record AND the single-instance one (it kills
    every vllm serve)."""
    log_root = tmp_path / "logroot"
    (log_root / "vllm").mkdir(parents=True)
    prefill = log_root / "vllm" / "vllm_pd_prefill_x.log"
    decode = log_root / "vllm" / "vllm_pd_decode_x.log"
    prefill.write_text(VLLM_019_BUDGETED_LOG, encoding="utf-8")
    decode.write_text(VLLM_019_BUDGETED_LOG.replace("pid=29655", "pid=29656"), encoding="utf-8")
    single = log_root / "vllm" / kp.CURRENT_NAME
    single.write_text("{}", encoding="utf-8")
    env = _clean_env(CAGE_LOG_ROOT=str(log_root))
    env["PATH"] = f"{stub_bin}:{env.get('PATH', '/usr/bin:/bin')}"
    script = (
        f'source "{SERVING / "manage_vllm_pd.sh"}" && '
        'cage_kv_pool_capture_pd "$LOG_DIR" "' + str(prefill) + '" "' + str(decode) + '" "" "" && '
        'echo CAPTURED_OK && stop_stack && echo STOPPED_OK'
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CAPTURED_OK" in proc.stdout and "STOPPED_OK" in proc.stdout
    assert "KV pool captured" in proc.stdout
    # the record existed between the two calls: the capture line names it
    assert str(log_root / "vllm" / kp.CURRENT_PD_NAME) in proc.stdout
    assert not (log_root / "vllm" / kp.CURRENT_PD_NAME).exists()
    assert not single.exists()


# ---------------------------------------------------------------------------
# 7. static pins: every launcher calls the hook, stops remove, docs name it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script,engine", [
    ("manage_vllm_server.sh", "vllm"),
    ("manage_sglang_server.sh", "sglang"),
    ("manage_lmdeploy_server.sh", "lmdeploy"),
])
def test_single_launchers_call_the_capture_and_remove_on_stop(script: str, engine: str) -> None:
    text = (SERVING / script).read_text(encoding="utf-8")
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert f'cage_kv_pool_capture --engine {engine} --log "$log_file" --out "$LOG_DIR/CURRENT.kvpool.json"' in code
    assert 'rm -f "$LOG_DIR/CURRENT.kvpool.json"' in code
    assert "S0F-24" in text


def test_pd_launcher_calls_the_pd_capture_and_removes_both_on_stop() -> None:
    text = (SERVING / "manage_vllm_pd.sh").read_text(encoding="utf-8")
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert 'cage_kv_pool_capture_pd "$LOG_DIR" "$prefill_log" "$decode_log"' in code
    assert 'rm -f "$LOG_DIR/CURRENT.kvpool.json" "$LOG_DIR/CURRENT.pd.kvpool.json"' in code
    vllm = (SERVING / "manage_vllm_server.sh").read_text(encoding="utf-8")
    assert 'rm -f "$LOG_DIR/CURRENT.kvpool.json" "$LOG_DIR/CURRENT.pd.kvpool.json"' in vllm


def test_lib_defines_the_hook_and_docs_name_the_record() -> None:
    lib = LIB.read_text(encoding="utf-8")
    assert "cage_kv_pool_capture()" in lib and "cage_kv_pool_capture_pd()" in lib
    assert "kv_pool_log.py" in lib
    assert "CURRENT.kvpool.json" in RUNBOOK.read_text(encoding="utf-8")
    assert "`checks/kv_pool_log.py`" in README.read_text(encoding="utf-8")
