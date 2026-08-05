"""Regression tests for the 2026-08 code-review fixes in the evaluation area.

Each test targets one verified finding and fails against the pre-fix behavior:
- src/evaluation/compression.py: analytical_kv_footprint() never reached the MLA
  branch of kv_cache_bytes() (DeepSeek-V3 silently scored with the wrong GQA formula).
- scripts/4_analysis/verify_results.py: the primary discovery glob used Path.rglob,
  which does not traverse directory symlinks (the exact tree run_phase2_stats.sh
  builds under stats/all_results), so it silently found zero files there.
- src/evaluation/performance.py: TPOT included non-streaming rows where
  ttft_ms == total_time_ms by construction (TTFT unobservable), folding a spurious
  ~0 interval into the distribution instead of excluding it as unmeasurable.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.compression import analytical_kv_footprint  # noqa: E402
from src.evaluation.performance import PerformanceEvaluator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "4_analysis"))
from verify_results import verify_dir  # noqa: E402


# --------------------------------------------------------------------------- #
# compression.py: analytical_kv_footprint() MLA branch (DeepSeek-V2/V3)
# --------------------------------------------------------------------------- #
def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, cfg_attrs: dict) -> None:
    """Install a fake `transformers` module in sys.modules so AutoConfig.from_pretrained
    returns a stand-in HF config object, without requiring the real package or network."""
    cfg = types.SimpleNamespace(**cfg_attrs)

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(_name):
            return cfg

    fake_mod = types.ModuleType("transformers")
    fake_mod.AutoConfig = _FakeAutoConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)


# DeepSeek-V3 public config values relevant to KV footprint.
_DEEPSEEK_V3_CFG = {
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "hidden_size": 7168,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
}


def test_analytical_kv_footprint_auto_derives_mla_latent_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch, _DEEPSEEK_V3_CFG)
    result = analytical_kv_footprint(
        "deepseek-ai/DeepSeek-V3", 1000, dtype="bf16", baseline_dtype="bf16"
    )
    assert result is not None
    # 512 (kv_lora_rank) + 64 (qk_rope_head_dim) = 576, matching the charter's own
    # 61L*576*2 = 68.6 KiB/token figure (MyDocs/PUBLICATION.md D4).
    assert result["mla_latent_dim"] == 576
    assert result["kv_bytes_per_token"] == pytest.approx(61 * 576 * 2.0)
    assert result["kv_bytes_per_token"] / 1024 == pytest.approx(68.625)
    assert result["kv_cache_bytes"] == pytest.approx(61 * 576 * 2.0 * 1000)


def test_analytical_kv_footprint_mla_latent_dim_not_kv_lora_rank_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # kv_lora_rank alone (512) would silently undercount by ~11% vs. the charter figure;
    # the decoupled RoPE key (qk_rope_head_dim) must be included too.
    _install_fake_transformers(monkeypatch, _DEEPSEEK_V3_CFG)
    result = analytical_kv_footprint("deepseek-ai/DeepSeek-V3", 1000)
    assert result is not None
    assert result["mla_latent_dim"] != 512
    assert result["mla_latent_dim"] == 576


def test_analytical_kv_footprint_non_mla_model_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    # A regular GQA model (no kv_lora_rank in its config) must NOT get an MLA latent dim
    # and must keep using the standard 2*layers*kv_heads*head_dim formula.
    _install_fake_transformers(
        monkeypatch,
        {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
        },
    )
    result = analytical_kv_footprint("meta-llama/Llama-3-8B", 1000)
    assert result is not None
    assert result["mla_latent_dim"] is None
    head_dim = 4096 // 32
    expected_bytes_per_token = 2.0 * 32 * 8 * head_dim * 2.0  # bf16 = 2 bytes/elem
    assert result["kv_bytes_per_token"] == pytest.approx(expected_bytes_per_token)


def test_analytical_kv_footprint_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_transformers(monkeypatch, _DEEPSEEK_V3_CFG)
    result = analytical_kv_footprint("deepseek-ai/DeepSeek-V3", 1000, mla_latent_dim=100)
    assert result is not None
    assert result["mla_latent_dim"] == 100


# --------------------------------------------------------------------------- #
# verify_results.py: symlink-safe metrics discovery
# --------------------------------------------------------------------------- #
def _write_metrics_and_csv(trial_dir: Path, *, baseline: str, dataset: str,
                           n_rows: int) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{baseline}_{dataset}_20260101"
    metrics = {
        "experiment": {"baseline": baseline, "dataset": dataset, "model": "m"},
        "performance": {"total_requests": n_rows},
    }
    (trial_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    rows = "\n".join(str(i) for i in range(n_rows))
    (trial_dir / f"{stem}_results.csv").write_text(f"example_id\n{rows}\n", encoding="utf-8")


def test_verify_dir_finds_metrics_through_symlinked_tree(tmp_path: Path) -> None:
    # Real cell dir, as a normal run-root layout would produce.
    real_run = tmp_path / "real_run"
    _write_metrics_and_csv(
        real_run / "baselines" / "no_cache" / "trial_1",
        baseline="no_cache", dataset="squad", n_rows=2,
    )

    # The flat symlink tree run_phase2_stats.sh builds under stats/all_results
    # (`ln -sfn "$(cd "$d" && pwd)" "$ALL_Q/$_b"`).
    flat = tmp_path / "stats" / "all_results"
    flat.mkdir(parents=True)
    (flat / "no_cache").symlink_to(real_run / "baselines" / "no_cache")

    report = verify_dir(flat)
    assert report["ok"] is True
    assert "errors" not in report  # no_per_trial_metrics_found must NOT fire
    assert len(report["checks"]) == 1
    assert report["checks"][0]["actual_rows"] == 2
    assert report["checks"][0]["ok"] is True


def test_verify_dir_direct_dir_still_works_without_symlinks(tmp_path: Path) -> None:
    # Non-regression: the plain (non-symlinked) case must keep working exactly as before.
    real_run = tmp_path / "real_run"
    _write_metrics_and_csv(
        real_run / "baselines" / "no_cache" / "trial_1",
        baseline="no_cache", dataset="squad", n_rows=3,
    )
    report = verify_dir(real_run)
    assert report["ok"] is True
    assert len(report["checks"]) == 1
    assert report["checks"][0]["actual_rows"] == 3


# --------------------------------------------------------------------------- #
# performance.py: TPOT excludes unmeasurable (non-streaming) generation intervals
# --------------------------------------------------------------------------- #
def test_tpot_excludes_nonstreaming_zero_generation_interval() -> None:
    ev = PerformanceEvaluator(monitor_resources=False)
    ev.start()
    # A real streaming request: TTFT genuinely observed, generation interval > 0.
    ev.record_request(request_id="r1", ttft_ms=100.0, total_time_ms=600.0, num_tokens=51)
    # A non-streaming (e.g. --offline debug engine) row: ttft_ms == total_time_ms by
    # construction because TTFT is unobservable there -- must be EXCLUDED from TPOT,
    # not folded in as a spurious ~0 interval.
    ev.record_request(request_id="r2", ttft_ms=500.0, total_time_ms=500.0, num_tokens=50)
    ev.stop()

    metrics = ev.compute_metrics()
    # Only r1 contributes: generation_time=500ms / 50 intervals = 10.0 ms/token.
    # Before the fix, r2 contributed a spurious tpot=0.0, pulling the mean down to 5.0.
    assert metrics.avg_tpot_ms == pytest.approx(10.0)


def test_tpot_all_nonstreaming_falls_back_to_zero_not_negative_or_nan() -> None:
    ev = PerformanceEvaluator(monitor_resources=False)
    ev.start()
    ev.record_request(request_id="r1", ttft_ms=500.0, total_time_ms=500.0, num_tokens=50)
    ev.stop()
    metrics = ev.compute_metrics()
    # No measurable TPOT rows at all -> the existing empty-list fallback (0.0), not NaN.
    assert metrics.avg_tpot_ms == 0.0
