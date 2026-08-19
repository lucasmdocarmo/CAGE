"""Preflight gate build-out pins (task #138: J5 + J11; merges #118's S0 items).

The J5 finding: three launch-path scripts deferred charter §6.5 budget-parity
correctness to a "preflight iso-bytes gate" that DID NOT EXIST. This file pins
the now-real gate — and the J11 coverage gates — at three levels:

1. EXTRACT-AND-EXECUTE (the tests/test_integration_wiring.py telemetry-parity
   pattern): each python sub-gate is embedded in
   scripts/checks/preflight_check.sh as a heredoc under a stable
   ``CAGE-*-GATE`` marker; tests slice the snippet out and run it for real —
   content pins alone cannot prove a gate runs. The iso-bytes parsers and
   parity math are additionally exec'd into a namespace for granular
   fixture-level unit tests (realistic startup-log variants per engine,
   corrupted lines, tolerance math).
2. BEHAVIOR: the poison-env gate (e) block is extracted and run under bash
   with poisoned/clean environments; the campaign-layout and open-loop gates
   ACTUALLY RUN locally (they are pure-python); the regime-bridge gate runs
   against a loopback Prometheus stub; the dataset-staleness gate runs against
   a synthetic HF cache.
3. WIRING PINS: the shell file declares every new lettered gate, the backends
   default covers all three final-scope engines, and the three formerly
   deferring scripts now name the real gate.

No GPU, no network beyond 127.0.0.1, no cloud.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO_ROOT / "scripts" / "checks" / "preflight_check.sh"
SERVING_CONFIG = REPO_ROOT / "scripts" / "lib" / "_serving_config.sh"
SGLANG_LAUNCHER = REPO_ROOT / "scripts" / "2_serving" / "manage_sglang_server.sh"
LMDEPLOY_LAUNCHER = REPO_ROOT / "scripts" / "2_serving" / "manage_lmdeploy_server.sh"

GIB = 1024 ** 3
MIB = 1024 ** 2

# Env vars the gates read: strip them from every subprocess so a developer
# shell can never leak state into a test.
_GATE_ENV_VARS = (
    "CAGE_ISO_BYTES_TOL", "CAGE_ISO_BYTES_LOGS", "CAGE_ISO_BYTES_LOG_ROOT",
    "CAGE_CALIBRATION_MANIFESTS", "CAGE_DATASETS", "DATASET",
    "CAGE_REGIME_GATE_SAMPLES", "CAGE_REGIME_GATE_INTERVAL",
    "CAGE_REGIME_KV_METRIC", "CAGE_REGIME_PREEMPT_METRIC",
    "HF_DATASETS_CACHE", "HF_HOME",
    "CAGE_SCBENCH_HF_PATH", "CAGE_SHAREGPT_HF_PATH",
    "CAGE_TELEMETRY_MOCK", "CAGE_DISABLE_LETTUCEDETECT",
    "CAGE_DISABLE_COMPRESSION", "CAGE_ALLOW_NO_COMPRESSION",
    "CAGE_ALLOW_REPLAY", "CAGE_ALLOW_NO_BACKUP",
    "LMDEPLOY_CACHE_MAX_ENTRY_COUNT", "LMDEPLOY_QUANT_POLICY",
    "CAGE_QUALITY_STRICT", "CAGE_LMDEPLOY_BACKEND_CHECK", "CAGE_CLAIM_CHECKER",
)


def _clean_env(**extra: str) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _GATE_ENV_VARS}
    env.update(extra)
    return env


def _snippet(marker: str) -> str:
    """Slice one python heredoc gate out of preflight_check.sh by its marker."""
    text = PREFLIGHT.read_text(encoding="utf-8")
    start = text.index(f"# {marker}")
    end = text.index("\nPY\n", start)
    return text[start:end]


def _run_gate(marker: str, *argv: str, env: Optional[Dict[str, str]] = None
              ) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-", *argv],
        input=_snippet(marker),
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env if env is not None else _clean_env(),
    )


# ---------------------------------------------------------------------------
# Wiring pins
# ---------------------------------------------------------------------------


def test_preflight_bash_syntax_ok() -> None:
    proc = subprocess.run(["bash", "-n", str(PREFLIGHT)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_preflight_declares_every_new_lettered_gate() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")
    for marker in (
        "CAGE-POISON-ENV-GATE-BEGIN", "CAGE-POISON-ENV-GATE-END",
        "CAGE-ISO-BYTES-GATE", "CAGE-BACKEND-ENDPOINTS-GATE",
        "CAGE-CAMPAIGN-LAYOUT-GATE", "CAGE-OPENLOOP-GATE",
        "CAGE-CALIBRATION-ARTIFACT-GATE", "CAGE-REGIME-BRIDGE-GATE",
        "CAGE-DATASET-STALENESS-GATE",
    ):
        assert marker in text, f"preflight lost the {marker} gate marker"
    for letter in ("(j)", "(k)", "(l)", "(m)", "(n)", "(o)", "(p)"):
        assert f'echo "{letter}' in text, f"gate {letter} echo line missing"
    # Skip-with-reason plumbing: exit code 3 must not count as a failure.
    assert "gate_rc() {" in text and "0|3) : ;;" in text


def test_preflight_backends_default_is_all_three_final_scope_engines() -> None:
    """J11: CAGE_PREFLIGHT_BACKENDS defaulted to 'vllm' only, silently skipping
    SGLang/LMDeploy telemetry parity. Pin the exact default expansion."""
    text = PREFLIGHT.read_text(encoding="utf-8")
    assert 'PREFLIGHT_BACKENDS="${CAGE_PREFLIGHT_BACKENDS:-vllm,sglang,lmdeploy}"' in text
    assert ':-vllm}"' not in text  # the old vllm-only default must be gone


def test_poison_env_list_contains_every_j11_addition() -> None:
    begin = PREFLIGHT.read_text(encoding="utf-8")
    block = begin[begin.index("CAGE-POISON-ENV-GATE-BEGIN"):
                  begin.index("CAGE-POISON-ENV-GATE-END")]
    for var in (
        "CAGE_TELEMETRY_MOCK", "CAGE_DISABLE_LETTUCEDETECT",
        "CAGE_DISABLE_COMPRESSION", "CAGE_ALLOW_NO_COMPRESSION",
        "CAGE_ALLOW_REPLAY", "CAGE_ALLOW_NO_BACKUP",
        "LMDEPLOY_CACHE_MAX_ENTRY_COUNT", "LMDEPLOY_QUANT_POLICY",
        "CAGE_QUALITY_STRICT", "CAGE_LMDEPLOY_BACKEND_CHECK",
        "CAGE_CLAIM_CHECKER",
    ):
        assert var in block, f"poison-env gate (e) lost {var}"


def test_deferring_scripts_now_name_the_real_gate() -> None:
    """J5 closure: the three launch-path scripts deferred §6.5 to a phantom
    gate; each must now name the concrete CAGE-ISO-BYTES-GATE."""
    for path in (SERVING_CONFIG, SGLANG_LAUNCHER, LMDEPLOY_LAUNCHER):
        assert "CAGE-ISO-BYTES-GATE" in path.read_text(encoding="utf-8"), (
            f"{path.name} no longer names the real iso-bytes gate")


# ---------------------------------------------------------------------------
# Poison-env gate (e): extracted bash, behavior-tested
# ---------------------------------------------------------------------------


def _run_poison_gate(env: Dict[str, str]) -> subprocess.CompletedProcess:
    text = PREFLIGHT.read_text(encoding="utf-8")
    block = text[text.index("# CAGE-POISON-ENV-GATE-BEGIN"):
                 text.index("# CAGE-POISON-ENV-GATE-END")]
    script = (
        "set -uo pipefail\nFAILED=0\n"
        'pass() { echo "  [PASS] $1"; }\n'
        'fail() { echo "  [FAIL] $1"; FAILED=1; }\n'
        + block + "\nexit $FAILED\n"
    )
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, env=env)


def test_poison_gate_clean_env_passes_and_flags_claim_checker() -> None:
    proc = _run_poison_gate(_clean_env())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[FAIL]" not in proc.stdout
    # J6: the in-code claim-checker default must be flagged LOUDLY.
    assert "CAGE_CLAIM_CHECKER unset -> in-code default 'nli'" in proc.stdout
    assert "#120" in proc.stdout


@pytest.mark.parametrize("var,value", [
    ("CAGE_ALLOW_REPLAY", "1"),
    ("CAGE_ALLOW_NO_BACKUP", "1"),
    ("LMDEPLOY_CACHE_MAX_ENTRY_COUNT", "0.7"),
    ("LMDEPLOY_QUANT_POLICY", "8"),
    ("CAGE_TELEMETRY_MOCK", "1"),
])
def test_poison_gate_fails_on_each_new_poison(var: str, value: str) -> None:
    proc = _run_poison_gate(_clean_env(**{var: value}))
    assert proc.returncode == 1, f"{var}={value} must poison preflight"
    assert f"[FAIL] {var} is set" in proc.stdout


def test_poison_gate_quality_strict_weakening_fails() -> None:
    proc = _run_poison_gate(_clean_env(CAGE_QUALITY_STRICT="0"))
    assert proc.returncode == 1
    assert "CAGE_QUALITY_STRICT=0 DISABLES" in proc.stdout
    # An explicit strict value passes and is echoed.
    proc = _run_poison_gate(_clean_env(CAGE_QUALITY_STRICT="1"))
    assert proc.returncode == 0
    assert "CAGE_QUALITY_STRICT strict" in proc.stdout


def test_poison_gate_lmdeploy_backend_check_warn_fails_strict_passes() -> None:
    proc = _run_poison_gate(_clean_env(CAGE_LMDEPLOY_BACKEND_CHECK="warn"))
    assert proc.returncode == 1
    assert "bypasses the P7 TurboMind backend assertion" in proc.stdout
    proc = _run_poison_gate(_clean_env(CAGE_LMDEPLOY_BACKEND_CHECK="strict"))
    assert proc.returncode == 0


def test_poison_gate_explicit_claim_checker_is_echoed() -> None:
    proc = _run_poison_gate(_clean_env(CAGE_CLAIM_CHECKER="minicheck"))
    assert proc.returncode == 0
    assert "CAGE_CLAIM_CHECKER=minicheck" in proc.stdout


# ---------------------------------------------------------------------------
# (j) iso-bytes gate: parser fixtures + parity math (exec'd namespace)
# ---------------------------------------------------------------------------

VLLM_V1_LOG = """\
INFO 08-18 09:59:58 [gpu_worker.py:276] Available KV cache memory: 20.93 GiB
INFO 08-18 10:00:00 [kv_cache_utils.py:716] GPU KV cache size: 457,343 tokens
INFO 08-18 10:00:00 [kv_cache_utils.py:720] Maximum concurrency for 4,096 tokens per request: 111.66x
"""

VLLM_V0_LOG = """\
INFO 07-14 worker.py:267] model weights take 15.27GiB; non_torch_memory takes \
0.06GiB; PyTorch activation peak memory takes 1.40GiB; the rest of the memory \
reserved for KV Cache is 5.33GiB.
"""

VLLM_LEGACY_LOG = """\
INFO: # GPU blocks: 27392, # CPU blocks: 2048
"""

SGLANG_LOG = """\
[2026-08-18 10:00:00] KV Cache is allocated. #tokens: 430913, K size: 13.15 GB, V size: 13.15 GB
[2026-08-18 10:00:01] max_total_num_tokens=430913, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=2049, context_len=4096
"""

SGLANG_SCHEDULER_ONLY_LOG = """\
[2026-08-18 10:00:01] max_total_num_tokens=430913, chunked_prefill_size=8192
"""

LMDEPLOY_LOG = """\
[TM][INFO] [BlockManager] block_size = 6 MB
[TM][INFO] [BlockManager] max_block_count = 1274
[TM][INFO] [BlockManager] chunk_size = 1274
"""


@pytest.fixture(scope="module")
def iso() -> dict:
    ns: dict = {"__name__": "cage_iso_bytes_gate_under_test"}
    exec(compile(_snippet("CAGE-ISO-BYTES-GATE"), "CAGE-ISO-BYTES-GATE", "exec"), ns)
    return ns


def test_iso_parses_vllm_v1_tokens_and_bytes(iso: dict) -> None:
    r = iso["parse_engine_log"]("vllm", VLLM_V1_LOG)
    assert r["tokens"] == 457343
    assert r["bytes"] == int(20.93 * GIB)
    assert len(r["evidence"]) == 2  # concurrency line NOT matched as a pool line


def test_iso_parses_vllm_v0_reserved_line(iso: dict) -> None:
    r = iso["parse_engine_log"]("vllm", VLLM_V0_LOG)
    assert r["bytes"] == int(5.33 * GIB)
    assert r["tokens"] is None


def test_iso_parses_vllm_legacy_blocks_as_tokens(iso: dict) -> None:
    r = iso["parse_engine_log"]("vllm", VLLM_LEGACY_LOG)
    assert r["tokens"] == 27392 * 16
    assert r["bytes"] is None


def test_iso_parses_sglang_alloc_and_tokens(iso: dict) -> None:
    r = iso["parse_engine_log"]("sglang", SGLANG_LOG)
    assert r["tokens"] == 430913
    assert r["bytes"] == int((13.15 + 13.15) * GIB)


def test_iso_parses_sglang_scheduler_fallback_tokens_only(iso: dict) -> None:
    r = iso["parse_engine_log"]("sglang", SGLANG_SCHEDULER_ONLY_LOG)
    assert r["tokens"] == 430913
    assert r["bytes"] is None


def test_iso_parses_lmdeploy_blockmanager_product(iso: dict) -> None:
    r = iso["parse_engine_log"]("lmdeploy", LMDEPLOY_LOG)
    assert r["bytes"] == 6 * MIB * 1274
    assert r["tokens"] is None


def test_iso_lmdeploy_half_pair_is_loud(iso: dict) -> None:
    half = "[TM][INFO] [BlockManager] block_size = 6 MB\n"
    with pytest.raises(iso["IsoBytesError"], match="half a product"):
        iso["parse_engine_log"]("lmdeploy", half)


def test_iso_corrupted_line_is_loud(iso: dict) -> None:
    corrupted = "INFO [kv_cache_utils.py] GPU KV cache size: ??? tokens\n"
    with pytest.raises(iso["IsoBytesError"], match="corrupted"):
        iso["parse_engine_log"]("vllm", corrupted)


def test_iso_no_recognizable_line_is_loud(iso: dict) -> None:
    with pytest.raises(iso["IsoBytesError"], match="NO recognizable KV-pool line"):
        iso["parse_engine_log"]("vllm", "INFO server started on port 8000\n")


def test_iso_unknown_engine_is_loud(iso: dict) -> None:
    with pytest.raises(iso["IsoBytesError"], match="no KV-pool parser"):
        iso["parse_engine_log"]("triton", "anything")


def test_iso_negative_quantity_is_loud(iso: dict) -> None:
    with pytest.raises(iso["IsoBytesError"], match="negative"):
        iso["parse_engine_log"]("vllm", "INFO: # GPU blocks: -5\n")


def _reading(engine: str, *, b: Optional[int] = None,
             t: Optional[int] = None) -> dict:
    return {"engine": engine, "bytes": b, "tokens": t, "evidence": []}


def test_iso_parity_bytes_within_tolerance(iso: dict) -> None:
    basis, gap, within = iso["compare_pair"](
        _reading("vllm", b=int(20.0 * GIB)), _reading("sglang", b=int(20.9 * GIB)), 0.05)
    assert basis == "bytes" and within and gap == pytest.approx(0.9 / 20.9, rel=1e-6)


def test_iso_parity_bytes_outside_tolerance(iso: dict) -> None:
    basis, gap, within = iso["compare_pair"](
        _reading("vllm", b=int(20.0 * GIB)), _reading("lmdeploy", b=int(22.0 * GIB)), 0.05)
    assert basis == "bytes" and not within and gap > 0.05


def test_iso_parity_token_proxy_basis(iso: dict) -> None:
    ra, rb = _reading("vllm", t=457343), _reading("sglang", t=430913)
    basis, gap, within = iso["compare_pair"](ra, rb, 0.05)
    assert basis == "tokens" and not within  # gap ~0.0578 > 0.05
    _, _, within_loose = iso["compare_pair"](ra, rb, 0.06)
    assert within_loose


def test_iso_parity_mixed_basis_is_loud(iso: dict) -> None:
    with pytest.raises(iso["IsoBytesError"], match="no common basis"):
        iso["compare_pair"](_reading("vllm", b=GIB), _reading("sglang", t=1000), 0.05)


def test_iso_parity_zero_pool_is_loud(iso: dict) -> None:
    with pytest.raises(iso["IsoBytesError"], match="zero"):
        iso["relative_gap"](0, 0)


# ---------------------------------------------------------------------------
# (j) iso-bytes gate: CLI end-to-end against a synthetic logs/ tree
# ---------------------------------------------------------------------------


def _write_logs(root: Path, **texts: str) -> None:
    for engine, text in texts.items():
        d = root / engine
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{engine}_probe_20260818.log").write_text(text, encoding="utf-8")


def test_iso_cli_three_engine_parity_passes(iso: dict, tmp_path: Path,
                                            monkeypatch, capsys) -> None:
    # 20.93 / 20.5 / 20.93 GiB pools: every pairwise gap under 5%.
    lmdeploy = ("[TM][INFO] [BlockManager] block_size = 20 MB\n"
                "[TM][INFO] [BlockManager] max_block_count = 1049\n")  # 20.49 GiB
    sglang = ("[x] KV Cache is allocated. #tokens: 430913, "
              "K size: 10.25 GB, V size: 10.25 GB\n")  # 20.5 GiB
    _write_logs(tmp_path, vllm=VLLM_V1_LOG, sglang=sglang, lmdeploy=lmdeploy)
    monkeypatch.setenv("CAGE_ISO_BYTES_LOG_ROOT", str(tmp_path))
    monkeypatch.delenv("CAGE_ISO_BYTES_TOL", raising=False)
    monkeypatch.delenv("CAGE_ISO_BYTES_LOGS", raising=False)
    rc = iso["main"](["gate", "vllm,sglang,lmdeploy"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert out.count("[pool]") == 3
    assert out.count("[PASS]") == 3  # three pairwise comparisons
    assert "bytes gap" in out


def test_iso_cli_out_of_tolerance_fails(iso: dict, tmp_path: Path,
                                        monkeypatch, capsys) -> None:
    lmdeploy = ("[TM][INFO] [BlockManager] block_size = 10 MB\n"
                "[TM][INFO] [BlockManager] max_block_count = 1000\n")  # ~9.8 GiB
    _write_logs(tmp_path, vllm=VLLM_V1_LOG, lmdeploy=lmdeploy)
    monkeypatch.setenv("CAGE_ISO_BYTES_LOG_ROOT", str(tmp_path))
    monkeypatch.delenv("CAGE_ISO_BYTES_TOL", raising=False)
    monkeypatch.delenv("CAGE_ISO_BYTES_LOGS", raising=False)
    rc = iso["main"](["gate", "vllm,lmdeploy"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NOT iso-bytes" in out


def test_iso_cli_missing_scoped_engine_log_fails(iso: dict, tmp_path: Path,
                                                 monkeypatch, capsys) -> None:
    _write_logs(tmp_path, vllm=VLLM_V1_LOG)
    monkeypatch.setenv("CAGE_ISO_BYTES_LOG_ROOT", str(tmp_path))
    monkeypatch.delenv("CAGE_ISO_BYTES_LOGS", raising=False)
    rc = iso["main"](["gate", "vllm,sglang,lmdeploy"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "sglang: no startup log" in out and "lmdeploy: no startup log" in out


def test_iso_cli_single_engine_scope_is_vacuous_pass(iso: dict, tmp_path: Path,
                                                     monkeypatch, capsys) -> None:
    _write_logs(tmp_path, vllm=VLLM_V1_LOG)
    monkeypatch.setenv("CAGE_ISO_BYTES_LOG_ROOT", str(tmp_path))
    monkeypatch.delenv("CAGE_ISO_BYTES_LOGS", raising=False)
    rc = iso["main"](["gate", "vllm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "vacuous" in out and "[pool] vllm" in out


def test_iso_cli_explicit_log_pins_override_discovery(iso: dict, tmp_path: Path,
                                                      monkeypatch, capsys) -> None:
    pinned = tmp_path / "pinned_vllm.log"
    pinned.write_text(VLLM_V0_LOG, encoding="utf-8")
    monkeypatch.setenv("CAGE_ISO_BYTES_LOG_ROOT", str(tmp_path / "empty"))
    monkeypatch.setenv("CAGE_ISO_BYTES_LOGS", f"vllm={pinned}")
    rc = iso["main"](["gate", "vllm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert str(pinned) in out


def test_iso_cli_bad_tolerance_fails(iso: dict, tmp_path: Path,
                                     monkeypatch, capsys) -> None:
    monkeypatch.setenv("CAGE_ISO_BYTES_LOG_ROOT", str(tmp_path))
    monkeypatch.setenv("CAGE_ISO_BYTES_TOL", "1.5")
    monkeypatch.delenv("CAGE_ISO_BYTES_LOGS", raising=False)
    rc = iso["main"](["gate", "vllm"])
    assert rc == 1
    assert "outside (0, 1)" in capsys.readouterr().out


def test_iso_cli_lmdeploy_turbomind_alias_normalizes(iso: dict, tmp_path: Path,
                                                     monkeypatch, capsys) -> None:
    _write_logs(tmp_path, lmdeploy=LMDEPLOY_LOG)
    monkeypatch.setenv("CAGE_ISO_BYTES_LOG_ROOT", str(tmp_path))
    monkeypatch.delenv("CAGE_ISO_BYTES_LOGS", raising=False)
    rc = iso["main"](["gate", "lmdeploy-turbomind"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[pool] lmdeploy" in out


# ---------------------------------------------------------------------------
# (l) campaign-layout round-trip gate: ACTUALLY RUNS locally
# ---------------------------------------------------------------------------


def test_layout_gate_runs_the_producer_organizer_roundtrip() -> None:
    proc = _run_gate("CAGE-CAMPAIGN-LAYOUT-GATE")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[PASS] campaign_layout produced" in proc.stdout


# ---------------------------------------------------------------------------
# (m) open-loop schedule + replay guard gate: ACTUALLY RUNS locally
# ---------------------------------------------------------------------------


def test_openloop_gate_runs_schedule_and_replay_guard() -> None:
    proc = _run_gate("CAGE-OPENLOOP-GATE")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "replay guard engages" in proc.stdout


# ---------------------------------------------------------------------------
# (n) calibration artifact gate
# ---------------------------------------------------------------------------

_CAL_V1 = {
    "procedure_version": "cal-v1 (2026-08-12)",
    "model": "qwen3-14b",
    "engine": "vllm",
    "budget_fraction": 0.9,
    "procedure": {"floor_n_requests": 30},
    "confirmatory": False,
    "floor": {"ttft_s": 0.21, "tpot_s": 0.011, "n_requests": 30,
              "statistic": "median"},
    "lambda_star": {"label": "ESTIMATED", "lambda_star_qps": 6.5},
}


def test_calibration_gate_skips_with_reason_pre_calibration() -> None:
    proc = _run_gate("CAGE-CALIBRATION-ARTIFACT-GATE", env=_clean_env())
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "[SKIP] pre-calibration" in proc.stdout


def test_calibration_gate_accepts_a_cal_v1_manifest(tmp_path: Path) -> None:
    p = tmp_path / "calibration_vllm.json"
    p.write_text(json.dumps(_CAL_V1), encoding="utf-8")
    proc = _run_gate("CAGE-CALIBRATION-ARTIFACT-GATE",
                     env=_clean_env(CAGE_CALIBRATION_MANIFESTS=str(p)))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cal-v1 shape OK" in proc.stdout


def test_calibration_gate_declared_but_missing_fails(tmp_path: Path) -> None:
    proc = _run_gate(
        "CAGE-CALIBRATION-ARTIFACT-GATE",
        env=_clean_env(CAGE_CALIBRATION_MANIFESTS=str(tmp_path / "absent*.json")))
    assert proc.returncode == 1
    assert "matched NO files" in proc.stdout


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.pop("floor"), "missing required cal-v1 keys"),
    (lambda d: d.update(procedure_version="cal-v2"), "is not cal-v1"),
    (lambda d: d.update(confirmatory=True), "confirmatory must be False"),
    (lambda d: d.update(budget_fraction=-1), "not positive finite"),
    (lambda d: d.update(lambda_star={}), "not a non-empty object"),
])
def test_calibration_gate_rejects_malformed_manifests(tmp_path: Path, mutate,
                                                      needle: str) -> None:
    doc = json.loads(json.dumps(_CAL_V1))
    mutate(doc)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    proc = _run_gate("CAGE-CALIBRATION-ARTIFACT-GATE",
                     env=_clean_env(CAGE_CALIBRATION_MANIFESTS=str(p)))
    assert proc.returncode == 1, proc.stdout
    assert needle in proc.stdout


# ---------------------------------------------------------------------------
# (o) regime-inputs bridge gate: loopback Prometheus stub
# ---------------------------------------------------------------------------


class _MetricsStub(BaseHTTPRequestHandler):
    kv_line = "vllm:gpu_cache_usage_perc{model_name=\"m\"} 0.42\n"
    preempt = True
    hits = 0

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        type(self).hits += 1
        body = "# HELP test stub\n" + self.kv_line
        if self.preempt:
            body += f"vllm:num_preemptions_total{{model_name=\"m\"}} {float(type(self).hits)}\n"
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:  # silence the test log
        pass


@pytest.fixture()
def metrics_server():
    _MetricsStub.hits = 0
    _MetricsStub.preempt = True
    server = HTTPServer(("127.0.0.1", 0), _MetricsStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _regime_env(**extra: str) -> Dict[str, str]:
    return _clean_env(CAGE_REGIME_GATE_SAMPLES="3",
                      CAGE_REGIME_GATE_INTERVAL="0.05", **extra)


def test_regime_gate_skips_with_reason_when_no_server() -> None:
    proc = _run_gate("CAGE-REGIME-BRIDGE-GATE", "http://127.0.0.1:9",
                     env=_regime_env())
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "[SKIP] live-only gate: no server" in proc.stdout


def test_regime_gate_certifies_live_telemetry(metrics_server: str) -> None:
    proc = _run_gate("CAGE-REGIME-BRIDGE-GATE", metrics_server,
                     env=_regime_env())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[PASS] regime bridge certified live telemetry" in proc.stdout
    assert "rho_kv_time_avg" in proc.stdout


def test_regime_gate_missing_metric_fails_loud(metrics_server: str) -> None:
    _MetricsStub.preempt = False
    proc = _run_gate("CAGE-REGIME-BRIDGE-GATE", metrics_server,
                     env=_regime_env())
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "exposes no 'vllm:num_preemptions_total'" in proc.stdout


# ---------------------------------------------------------------------------
# (p) dataset staleness gate: synthetic HF cache + fake `datasets` module
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_datasets_path(tmp_path: Path) -> Path:
    """A PYTHONPATH dir with a stub `datasets` module so importing the staging
    roster (scripts/1_setup/download_datasets.py) never needs the HF package
    (same fake-module technique as tests/test_dataset_download_staging.py)."""
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "datasets.py").write_text(
        "def load_dataset(*a, **k):\n"
        "    raise RuntimeError('stub: staleness gate must never download')\n",
        encoding="utf-8",
    )
    return stub_dir


def _stage(cache: Path, dirname: str) -> None:
    d = cache / dirname
    d.mkdir(parents=True, exist_ok=True)
    (d / "dataset_info.json").write_text("{}", encoding="utf-8")


def _datasets_env(stub: Path, cache: Path, **extra: str) -> Dict[str, str]:
    return _clean_env(PYTHONPATH=str(stub), HF_DATASETS_CACHE=str(cache), **extra)


# The default campaign roster's HF-cache directory names ("/" -> "___"), from
# download_datasets.dataset_specs() at the default (un-overridden) HF paths.
_DEFAULT_ROSTER_CACHE_DIRS = (
    "hotpot_qa",                # hotpotqa
    "dgslibisey___MuSiQue",     # musique
    "allenai___qasper",         # qasper
    "microsoft___SCBench",      # scbench (both configs share the dir)
    "RyokoAI___ShareGPT52K",    # sharegpt
    "squad_v2",                 # squad_v2
)


def test_datasets_gate_unset_env_evaluates_default_roster_and_refuses(
        fake_datasets_path: Path, tmp_path: Path) -> None:
    """K-led4 fix (task #141): with CAGE_DATASETS and DATASET both unset the
    gate must NOT silently skip -- it evaluates the full default campaign
    roster and, with nothing staged, REFUSES loudly."""
    proc = _run_gate("CAGE-DATASET-STALENESS-GATE",
                     env=_datasets_env(fake_datasets_path, tmp_path / "cache"))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "[INFO] no explicit dataset request" in proc.stdout
    assert "DEFAULT campaign roster" in proc.stdout
    assert "REFUSING" in proc.stdout and "No silent subset" in proc.stdout
    # Every charter campaign key must be named in the refusal.
    for key in ("hotpotqa", "musique", "qasper", "scbench", "sharegpt",
                "squad_v2"):
        assert key in proc.stdout, f"default-roster refusal lost {key}"
    assert "[SKIP]" not in proc.stdout


def test_datasets_gate_unset_env_passes_when_default_roster_staged(
        fake_datasets_path: Path, tmp_path: Path) -> None:
    """K-led4 fix (task #141): unset env + fully staged default roster PASSES
    (loud INFO line, one PASS per key, never a skip)."""
    cache = tmp_path / "cache"
    for dirname in _DEFAULT_ROSTER_CACHE_DIRS:
        _stage(cache, dirname)
    proc = _run_gate("CAGE-DATASET-STALENESS-GATE",
                     env=_datasets_env(fake_datasets_path, cache))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[INFO] no explicit dataset request" in proc.stdout
    assert proc.stdout.count("[PASS] dataset") == len(_DEFAULT_ROSTER_CACHE_DIRS)
    assert "[SKIP]" not in proc.stdout


def test_datasets_gate_has_no_silent_skip_path() -> None:
    """K-led4 wiring pin: the staleness gate source must carry NO skip exit --
    every path is PASS/INFO or FAIL (exit 3 = the gate_rc skip code)."""
    snippet = _snippet("CAGE-DATASET-STALENESS-GATE")
    assert "[SKIP]" not in snippet
    assert "exit(3)" not in snippet


def test_datasets_gate_passes_when_requested_sets_are_staged(
        fake_datasets_path: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _stage(cache, "squad_v2")
    _stage(cache, "allenai___qasper")
    proc = _run_gate(
        "CAGE-DATASET-STALENESS-GATE",
        env=_datasets_env(fake_datasets_path, cache,
                          CAGE_DATASETS="squad_v2,qasper"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.count("[PASS] dataset") == 2


def test_datasets_gate_refuses_unstaged_charter_dataset(
        fake_datasets_path: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _stage(cache, "squad_v2")  # qasper deliberately NOT staged
    proc = _run_gate(
        "CAGE-DATASET-STALENESS-GATE",
        env=_datasets_env(fake_datasets_path, cache,
                          CAGE_DATASETS="squad_v2,qasper"))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "REFUSING" in proc.stdout and "qasper" in proc.stdout
    assert "No silent subset" in proc.stdout


def test_datasets_gate_refuses_unknown_key(fake_datasets_path: Path,
                                           tmp_path: Path) -> None:
    proc = _run_gate(
        "CAGE-DATASET-STALENESS-GATE",
        env=_datasets_env(fake_datasets_path, tmp_path / "cache",
                          CAGE_DATASETS="sqaud_v2"))  # typo on purpose
    assert proc.returncode == 1
    assert "not in the charter roster" in proc.stdout


def test_datasets_gate_ruler_is_synthetic_and_always_ok(
        fake_datasets_path: Path, tmp_path: Path) -> None:
    proc = _run_gate(
        "CAGE-DATASET-STALENESS-GATE",
        env=_datasets_env(fake_datasets_path, tmp_path / "cache",
                          CAGE_DATASETS="ruler"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "synthetic" in proc.stdout


def test_datasets_gate_honors_runner_dataset_fallback(
        fake_datasets_path: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _stage(cache, "squad_v2")
    proc = _run_gate(
        "CAGE-DATASET-STALENESS-GATE",
        env=_datasets_env(fake_datasets_path, cache, DATASET="squad_v2"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dataset 'squad_v2' staged" in proc.stdout


# ---------------------------------------------------------------------------
# (q) cage-stats pin parity gate (task #143, finding L-A CRITICAL): the
# installed cage-stats must be EXACTLY the requirements.txt pinned commit --
# one commit of lag fabricated rho_KV=0.0 from an absent occupancy gauge (E2b).
# Parse/channel fixtures via the exec'd namespace (iso-gate pattern) plus
# end-to-end runs against the LIVE venv and synthetic requirements files.
# ---------------------------------------------------------------------------

_PIN_SHA = "df0eab4697aff133ff9dc76a7d45d8be706d89c0"


def test_preflight_declares_pin_parity_gate() -> None:
    text = PREFLIGHT.read_text(encoding="utf-8")
    assert "CAGE-STATS-PIN-PARITY-GATE" in text, "preflight lost gate (q)"
    assert 'echo "(q)' in text, "gate (q) echo line missing"
    # (q) is a hard gate on required infra: it must NEVER use the skip code 3.
    assert "sys.exit(3)" not in _snippet("CAGE-STATS-PIN-PARITY-GATE")


@pytest.fixture(scope="module")
def pinq() -> dict:
    ns: dict = {"__name__": "cage_pin_parity_gate_under_test"}
    exec(compile(_snippet("CAGE-STATS-PIN-PARITY-GATE"),
                 "CAGE-STATS-PIN-PARITY-GATE", "exec"), ns)
    return ns


def _req(sha_or_line: str) -> str:
    line = (f"cage-stats @ git+https://github.com/lucasmdocarmo/"
            f"cage-stats.git@{sha_or_line}"
            if re.fullmatch(r"[0-9A-Fa-f]+", sha_or_line) else sha_or_line)
    return f"# comment line\nnumpy==1.26.4\n{line}\n"


def test_pinq_parses_full_sha_and_normalizes_case(pinq: dict) -> None:
    assert pinq["pinned_sha"](_req(_PIN_SHA)) == _PIN_SHA
    assert pinq["pinned_sha"](_req(_PIN_SHA.upper())) == _PIN_SHA


def test_pinq_short_sha_is_refused(pinq: dict) -> None:
    with pytest.raises(pinq["PinParityError"], match="FULL 40-char"):
        pinq["pinned_sha"](_req("df0eab4"))


def test_pinq_missing_pin_is_refused(pinq: dict) -> None:
    with pytest.raises(pinq["PinParityError"], match="NO cage-stats pin"):
        pinq["pinned_sha"]("numpy==1.26.4\n")


def test_pinq_commented_out_pin_does_not_count(pinq: dict) -> None:
    text = f"# cage-stats @ git+https://github.com/x/cage-stats.git@{_PIN_SHA}\n"
    with pytest.raises(pinq["PinParityError"], match="NO cage-stats pin"):
        pinq["pinned_sha"](text)


def test_pinq_duplicate_pins_are_refused(pinq: dict) -> None:
    with pytest.raises(pinq["PinParityError"], match="exactly one"):
        pinq["pinned_sha"](_req(_PIN_SHA) + _req("0" * 40))


def test_pinq_non_git_pin_is_refused(pinq: dict) -> None:
    with pytest.raises(pinq["PinParityError"], match="not a parseable"):
        pinq["pinned_sha"](_req("cage-stats==0.1.0"))


def test_pinq_channel_extractors(pinq: dict) -> None:
    assert pinq["commit_from_direct_url"](
        {"url": "x", "vcs_info": {"vcs": "git", "commit_id": _PIN_SHA.upper()}}
    ) == _PIN_SHA
    assert pinq["commit_from_direct_url"](
        {"url": "x", "archive_info": {"hash": "sha256=aa"}}) is None
    assert pinq["commit_from_direct_url"](None) is None
    assert pinq["commit_from_version"](f"0.1.0+g{_PIN_SHA}") == _PIN_SHA
    assert pinq["commit_from_version"]("0.1.0") is None
    assert pinq["commit_from_version"]("2.0.0+cu118") is None
    assert pinq["commit_from_freeze_line"](
        f"cage-stats @ git+https://github.com/x/cage-stats.git@{_PIN_SHA}"
    ) == _PIN_SHA
    assert pinq["commit_from_freeze_line"]("cage-stats==0.1.0") is None


class _StubDist:
    """Minimal importlib.metadata.Distribution stand-in for channel tests."""

    def __init__(self, version: str, direct_url: Optional[str]) -> None:
        self.version = version
        self._direct_url = direct_url

    def read_text(self, name: str) -> Optional[str]:
        return self._direct_url if name == "direct_url.json" else None


def test_pinq_channels_prefer_direct_url_and_skip_freeze(pinq: dict) -> None:
    dist = _StubDist(
        f"0.1.0+g{_PIN_SHA}",
        json.dumps({"url": "x", "vcs_info": {"commit_id": _PIN_SHA}}))

    def _freeze_must_not_run():
        raise AssertionError("freeze fallback consulted despite direct_url")
        yield  # pragma: no cover -- generator: raises only if iterated

    channels = pinq["installed_commits"](dist, freeze_lines=_freeze_must_not_run())
    assert channels == {"direct_url.vcs_info": _PIN_SHA, "version.+g": _PIN_SHA}


def test_pinq_freeze_fallback_engages_without_direct_url(pinq: dict) -> None:
    dist = _StubDist("0.5.0", None)
    channels = pinq["installed_commits"](
        dist,
        freeze_lines=[f"cage-stats @ git+https://github.com/x/y.git@{_PIN_SHA}"])
    assert channels == {"pip.freeze": _PIN_SHA}


def test_pinq_no_channel_yields_empty(pinq: dict) -> None:
    assert pinq["installed_commits"](_StubDist("0.5.0", None),
                                     freeze_lines=[]) == {}


def test_pinq_corrupt_direct_url_is_loud(pinq: dict) -> None:
    with pytest.raises(pinq["PinParityError"], match="corrupt"):
        pinq["installed_commits"](_StubDist("0.5.0", "{not json"),
                                  freeze_lines=[])


def test_pinq_main_disagreeing_channels_fail(pinq: dict, monkeypatch,
                                             capsys) -> None:
    dist = _StubDist(
        "0.1.0+g" + "0" * 40,
        json.dumps({"url": "x", "vcs_info": {"commit_id": _PIN_SHA}}))
    monkeypatch.setattr(pinq["md"], "distribution", lambda name: dist)
    rc = pinq["main"](["gate", str(REPO_ROOT / "requirements.txt")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "DISAGREE" in out


def test_pinq_main_missing_install_fails(pinq: dict, monkeypatch,
                                         capsys) -> None:
    def _absent(name: str):
        raise pinq["md"].PackageNotFoundError(name)

    monkeypatch.setattr(pinq["md"], "distribution", _absent)
    rc = pinq["main"](["gate", str(REPO_ROOT / "requirements.txt")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NOT installed" in out and "required infra" in out


def test_pinq_end_to_end_parity_passes_on_this_venv() -> None:
    """The real deal: the ACTUAL requirements.txt pin against the ACTUAL venv
    install. This is the assertion the L-A finding said no test made."""
    proc = _run_gate("CAGE-STATS-PIN-PARITY-GATE",
                     str(REPO_ROOT / "requirements.txt"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[PASS] cage-stats installed commit == requirements.txt pin" in proc.stdout


def test_pinq_end_to_end_mismatch_fails(tmp_path: Path) -> None:
    req = tmp_path / "requirements.txt"
    req.write_text(_req("0" * 40), encoding="utf-8")
    proc = _run_gate("CAGE-STATS-PIN-PARITY-GATE", str(req))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "cage-stats pin parity" in proc.stdout
    assert "E2b" in proc.stdout  # the WHY travels with the failure


def test_pinq_end_to_end_unparseable_pin_fails(tmp_path: Path) -> None:
    req = tmp_path / "requirements.txt"
    req.write_text(_req("df0eab4"), encoding="utf-8")
    proc = _run_gate("CAGE-STATS-PIN-PARITY-GATE", str(req))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FULL 40-char" in proc.stdout


def test_pinq_end_to_end_missing_pin_fails(tmp_path: Path) -> None:
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.26.4\n", encoding="utf-8")
    proc = _run_gate("CAGE-STATS-PIN-PARITY-GATE", str(req))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "NO cage-stats pin" in proc.stdout
