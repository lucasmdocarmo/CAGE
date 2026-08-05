"""Unit tests for the dataset-staging additions in
scripts/1_setup/download_datasets.py (charter D5 + D8 §8.6(a)).

Covers:
- SCBench staged as BOTH charter subsets (scbench_kv + scbench_qa_eng)
- RAGTruth + TRUE-benchmark anchors staged behind explicit selection
  ('calibration' or per-name) and NOT inside the default 'all' roster
- 'all' = the charter campaign roster including scbench
- env-var overrides (CAGE_SCBENCH_HF_PATH / CAGE_RAGTRUTH_HF_PATH /
  CAGE_TRUE_HF_SPECS) resolved at call time
- non-empty-split assertion + exit-code discipline preserved for the new
  multi-spec entries

Same load-by-path + fake-`datasets`-module technique as
tests/test_review_fixes_data.py (the HF `datasets` package is NOT required).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "1_setup" / "download_datasets.py"


def _install_fake_datasets(monkeypatch, load_dataset_fn):
    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset_fn
    monkeypatch.setitem(sys.modules, "datasets", fake)


def _load_module():
    spec = importlib.util.spec_from_file_location("cage_download_datasets", DOWNLOAD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeDatasetDict:
    def __init__(self, split_data):
        self._split_data = split_data

    def keys(self):
        return self._split_data.keys()

    def __getitem__(self, key):
        return self._split_data[key]


def _recording_success(calls):
    def load_dataset(*args, **kwargs):
        calls.append(args)
        return _FakeDatasetDict({"test": list(range(3))})
    return load_dataset


# ---------------------------------------------------------------------------
# Spec map
# ---------------------------------------------------------------------------


def test_scbench_staged_as_both_charter_subsets(monkeypatch):
    _install_fake_datasets(monkeypatch, lambda *a, **k: None)
    module = _load_module()

    assert module.dataset_specs()["scbench"] == [
        ("microsoft/SCBench", "scbench_kv"),
        ("microsoft/SCBench", "scbench_qa_eng"),
    ]


def test_true_anchor_default_specs_and_custom_parse(monkeypatch):
    _install_fake_datasets(monkeypatch, lambda *a, **k: None)
    module = _load_module()

    # Default: three HF-hosted TRUE-benchmark constituent tasks.
    assert module.dataset_specs()["true"] == [
        ("tals/vitaminc", None),
        ("paws", "labeled_final"),
        ("fever", "v1.0"),
    ]

    monkeypatch.setenv("CAGE_TRUE_HF_SPECS", "org/one, org/two:cfg ,")
    assert module.dataset_specs()["true"] == [
        ("org/one", None),
        ("org/two", "cfg"),
    ]


def test_env_overrides_resolved_at_call_time(monkeypatch):
    _install_fake_datasets(monkeypatch, lambda *a, **k: None)
    module = _load_module()

    monkeypatch.setenv("CAGE_SCBENCH_HF_PATH", "mirror/SCBench")
    monkeypatch.setenv("CAGE_RAGTRUTH_HF_PATH", "mirror/ragtruth")
    specs = module.dataset_specs()
    assert specs["scbench"][0][0] == "mirror/SCBench"
    assert specs["ragtruth"] == [("mirror/ragtruth", None)]


def test_existing_single_spec_entries_unchanged(monkeypatch):
    """The pre-existing roster keeps its exact (path, config) pairs."""
    _install_fake_datasets(monkeypatch, lambda *a, **k: None)
    specs = _load_module().dataset_specs()

    assert specs["hotpotqa"] == [("hotpot_qa", "distractor")]
    assert specs["qasper"] == [("allenai/qasper", None)]
    assert specs["squad_v2"] == [("squad_v2", None)]
    assert specs["trivia_qa"] == [("trivia_qa", "rc")]
    assert specs["musique"] == [("dgslibisey/MuSiQue", None)]


# ---------------------------------------------------------------------------
# main() selection semantics
# ---------------------------------------------------------------------------


def test_dataset_scbench_downloads_both_configs(monkeypatch):
    calls = []
    _install_fake_datasets(monkeypatch, _recording_success(calls))
    module = _load_module()

    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "scbench"])
    assert module.main() == 0
    assert calls == [
        ("microsoft/SCBench", "scbench_kv"),
        ("microsoft/SCBench", "scbench_qa_eng"),
    ]


def test_dataset_calibration_downloads_ragtruth_and_true_anchors(monkeypatch):
    calls = []
    _install_fake_datasets(monkeypatch, _recording_success(calls))
    module = _load_module()

    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "calibration"])
    assert module.main() == 0

    paths = [c[0] for c in calls]
    assert "KRLabsOrg/ragtruth" in paths
    assert "tals/vitaminc" in paths
    assert ("paws", "labeled_final") in calls
    assert ("fever", "v1.0") in calls


def test_all_includes_scbench_but_not_calibration_anchors(monkeypatch):
    calls = []
    _install_fake_datasets(monkeypatch, _recording_success(calls))
    module = _load_module()

    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "all"])
    assert module.main() == 0

    paths = [c[0] for c in calls]
    assert ("microsoft/SCBench", "scbench_kv") in calls
    assert ("microsoft/SCBench", "scbench_qa_eng") in calls
    assert "allenai/qasper" in paths  # Qasper staged in the roster
    # Calibration anchors stay behind explicit selection.
    assert "KRLabsOrg/ragtruth" not in paths
    assert "tals/vitaminc" not in paths


# ---------------------------------------------------------------------------
# Fail-closed exit-code discipline extends to the new entries
# ---------------------------------------------------------------------------


def test_one_failing_scbench_config_flips_exit_code(monkeypatch):
    def flaky(name, config=None, split=None):
        if config == "scbench_qa_eng":
            raise RuntimeError("simulated failure for the second subset only")
        return _FakeDatasetDict({"test": list(range(3))})

    _install_fake_datasets(monkeypatch, flaky)
    module = _load_module()

    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "scbench"])
    assert module.main() == 1


def test_empty_split_on_new_entry_flips_exit_code(monkeypatch):
    _install_fake_datasets(monkeypatch, lambda *a, **k: _FakeDatasetDict({"test": []}))
    module = _load_module()

    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "ragtruth"])
    assert module.main() == 1


def test_calibration_failure_flips_exit_code(monkeypatch):
    def always_fails(*args, **kwargs):
        raise RuntimeError("simulated network failure")

    _install_fake_datasets(monkeypatch, always_fails)
    module = _load_module()

    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "calibration"])
    assert module.main() == 1
