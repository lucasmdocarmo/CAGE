"""Regression tests for review-fix findings in scripts/1_setup/download_datasets.py.

Covers:
- main() must return a non-zero exit code when every requested dataset fails to
  download, instead of silently printing "All requested datasets downloaded!" and
  exiting 0 (finding: BUG major, scripts/1_setup/download_datasets.py:76).
- download_dataset() must treat a successfully-returned-but-EMPTY split as a
  failure (silent/corrupt download), since no such check existed before.

The `datasets` package is not installed in the analysis venv (download_datasets.py
imports it at module top-level, unlike the lazy-import loaders in src/data/loader.py),
so these tests inject a fake `datasets` module into sys.modules before loading the
script by path -- the same technique tests/test_dataset_loaders.py and
tests/test_cag_reference_runner.py use for their own not-a-package targets.
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
    """Install a fake `datasets` module whose load_dataset() is `load_dataset_fn`."""
    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset_fn
    monkeypatch.setitem(sys.modules, "datasets", fake)


def _load_download_datasets_module():
    """Load scripts/1_setup/download_datasets.py by path (it's not a package)."""
    spec = importlib.util.spec_from_file_location("cage_download_datasets", DOWNLOAD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeDatasetDict:
    """Minimal stand-in for a HF DatasetDict (no-split load_dataset() return)."""

    def __init__(self, split_data):
        self._split_data = split_data

    def keys(self):
        return self._split_data.keys()

    def __getitem__(self, key):
        return self._split_data[key]


def test_download_datasets_main_exits_nonzero_when_every_dataset_fails(monkeypatch):
    """Regression: main() used to catch every per-dataset exception, print a
    'Warning: Skipping...' line, and keep going -- ending with 'All requested
    datasets downloaded!' and an implicit exit code 0 even when EVERY dataset
    failed. It must now return 1.
    """
    def always_fails(*args, **kwargs):
        raise RuntimeError("simulated network failure")

    _install_fake_datasets(monkeypatch, always_fails)
    module = _load_download_datasets_module()

    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "all"])
    rc = module.main()

    assert rc == 1


def test_download_datasets_main_exits_zero_on_full_success(monkeypatch):
    """Control case: every dataset succeeding (with non-empty splits) still exits 0."""
    def always_succeeds(*args, **kwargs):
        return _FakeDatasetDict({"validation": list(range(5))})

    _install_fake_datasets(monkeypatch, always_succeeds)
    module = _load_download_datasets_module()

    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "squad_v2"])
    rc = module.main()

    assert rc == 0


def test_download_datasets_main_exits_nonzero_on_partial_failure(monkeypatch):
    """One dataset failing among several must still flip the overall exit code,
    not just print a warning and report blanket success.
    """
    def flaky(name, config=None, split=None):
        if name == "squad_v2":
            raise RuntimeError("simulated failure for squad_v2 only")
        return _FakeDatasetDict({"validation": list(range(3))})

    _install_fake_datasets(monkeypatch, flaky)
    module = _load_download_datasets_module()

    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "all"])
    rc = module.main()

    assert rc == 1


def test_download_dataset_raises_on_empty_split(monkeypatch):
    """Regression: a load_dataset() call that returns without raising but with an
    EMPTY split (silent/corrupt download) must be treated as a failure -- there was
    previously no size/count validation anywhere in the script.
    """
    def returns_empty(*args, **kwargs):
        return _FakeDatasetDict({"validation": []})

    _install_fake_datasets(monkeypatch, returns_empty)
    module = _load_download_datasets_module()

    with pytest.raises(AssertionError, match="empty"):
        module.download_dataset("squad_v2")


def test_download_datasets_main_exits_nonzero_when_split_downloads_empty(monkeypatch):
    """The empty-split check must actually flip main()'s exit code, not just raise
    in isolation.
    """
    def returns_empty(*args, **kwargs):
        return _FakeDatasetDict({"validation": []})

    _install_fake_datasets(monkeypatch, returns_empty)
    module = _load_download_datasets_module()

    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "squad_v2"])
    rc = module.main()

    assert rc == 1
