"""scripts/checks/probe_datasets_offline.py (S1 audit 2026-10-23, MEDIUM 4): the
stage 4 proof that every charter dataset loads through the runner's loader under
the run job's offline mode. The loader is faked here; the live pod is the proof.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "checks" / "probe_datasets_offline.py"


def _probe_module() -> Any:
    spec = importlib.util.spec_from_file_location("probe_datasets_offline_under_test", PROBE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_dataset_loads_and_the_verdict_names_the_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    probe = _probe_module()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(probe, "_load", lambda name, k: {"squad_v2": 3, "musique": 3, "qasper": 2}[name])
    out = tmp_path / "rec.json"
    assert probe.main(["--datasets", "squad_v2,musique,qasper", "--max-examples", "3", "--require-offline", "--out", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["ok"] is True and rec["hf_hub_offline"] is True
    assert {k: v["examples"] for k, v in rec["datasets"].items()} == {"squad_v2": 3, "musique": 3, "qasper": 2}
    assert probe.verdict_line(rec) == "DATASETS_OFFLINE_OK hf_hub_offline=1 loaded=squad_v2=3,musique=3,qasper=2"


def test_a_loader_error_or_an_empty_load_fails_with_the_dataset_named(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _probe_module()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def _boom(name: str, k: int) -> int:
        if name == "musique":
            raise FileNotFoundError("dgslibisey/MuSiQue not in the cache (offline mode is enabled)")
        return 3

    monkeypatch.setattr(probe, "_load", _boom)
    rec = probe.probe(["squad_v2", "musique", "qasper"], 3, True)
    assert rec["ok"] is False and rec["reason"] == "dataset musique did not load: FileNotFoundError"
    assert "qasper" not in rec["datasets"]  # stopped at the first failure
    assert probe.verdict_line(rec).startswith("DATASETS_OFFLINE_FAILED reason=dataset musique did not load")
    monkeypatch.setattr(probe, "_load", lambda name, k: 0)
    rec = probe.probe(["squad_v2"], 3, True)
    assert rec["ok"] is False and rec["reason"] == "dataset squad_v2 loaded zero examples"


def test_require_offline_refuses_an_online_environment_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _probe_module()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    called = []
    monkeypatch.setattr(probe, "_load", lambda name, k: called.append(name) or 3)
    rec = probe.probe(["squad_v2"], 3, True)
    assert rec["ok"] is False and "HF_HUB_OFFLINE is not 1" in rec["reason"] and called == []
    # without the flag an online load is allowed and recorded as such
    rec = probe.probe(["squad_v2"], 3, False)
    assert rec["ok"] is True and rec["hf_hub_offline"] is False and called == ["squad_v2"]


def test_the_script_runs_end_to_end_and_refuses_bad_arguments(tmp_path: Path) -> None:
    proc = subprocess.run([sys.executable, str(PROBE), "--datasets", "", "--out", str(tmp_path / "x.json")],
                          capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120)
    assert proc.returncode == 1 and "DATASETS_OFFLINE_FAILED reason=no dataset named" in proc.stdout
    proc = subprocess.run([sys.executable, str(PROBE), "--datasets", "squad_v2", "--require-offline"],
                          capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120,
                          env={k: v for k, v in __import__("os").environ.items() if k != "HF_HUB_OFFLINE"})
    assert proc.returncode == 1 and "HF_HUB_OFFLINE is not 1" in proc.stdout and "Traceback" not in proc.stderr
