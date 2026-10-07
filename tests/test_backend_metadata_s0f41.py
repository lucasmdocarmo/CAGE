"""S0F-41: engine provenance for the SGLang and HF-oracle backends.

Live H100 (2026-10-07): the campaign seam refuses any window whose backend
reports neither server_version nor client_library_version, and
capture_backend_metadata covered vllm and ollama only, so every SGLang and
HF-oracle cell was refused after running its window. SGLang's version comes
from GET /get_server_info (its own venv is not importable here); the HF
oracle's from the transformers library it runs in-process.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("run_experiment_s0f41", REPO_ROOT / "scripts" / "3_run" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sglang_reports_the_server_version_from_get_server_info(monkeypatch) -> None:
    rx = _load()
    seen = []

    def fake_get(url, *, timeout=5):
        seen.append(url)
        if url.endswith("/get_server_info"):
            return {"version": "0.5.10.post1", "model_path": "Qwen/Qwen3-14B"}
        if url.endswith("/v1/models"):
            return {"data": [{"id": "Qwen/Qwen3-14B"}]}
        return None

    monkeypatch.setattr(rx, "_safe_get_json", fake_get)
    md = rx.capture_backend_metadata(api_base="http://localhost:30000/", backend="sglang", model_name="Qwen/Qwen3-14B", use_offline=False)
    assert md["server_version"] == "0.5.10.post1"
    assert md["loaded_model"] == "Qwen/Qwen3-14B" and md["loaded_models"] == ["Qwen/Qwen3-14B"]
    assert "http://localhost:30000/get_server_info" in seen


def test_hf_oracle_reports_the_transformers_version(monkeypatch) -> None:
    rx = _load()
    monkeypatch.setattr(rx, "_safe_get_json", lambda url, *, timeout=5: None)
    md = rx.capture_backend_metadata(api_base="http://localhost:8000", backend="hf-oracle", model_name="Qwen/Qwen3-14B", use_offline=False)
    assert md["client_library_version"] and md["client_library_version"].startswith("transformers ")
    assert md["loaded_model"] == "Qwen/Qwen3-14B"


def test_an_unknown_backend_still_reports_nothing(monkeypatch) -> None:
    rx = _load()
    monkeypatch.setattr(rx, "_safe_get_json", lambda url, *, timeout=5: None)
    md = rx.capture_backend_metadata(api_base="http://x", backend="lmdeploy", model_name="m", use_offline=False)
    assert md["server_version"] is None and md["client_library_version"] is None
