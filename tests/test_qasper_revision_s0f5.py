"""Qasper through the Hub's parquet export at a pinned commit (S0F-5, ADR-0127).

The S0 finding: ``datasets`` 4.8.5 refuses the script-based ``allenai/qasper``;
S0 loaded it only because the Mac's legacy cache was copied to the pod. The fix
passes ``revision=QASPER_REVISION`` (the parquet-converter commit on the Hub,
no config name) in the two places that call ``load_dataset`` for qasper: the
loader at run time (src/data/loader.py) and the staging script at pod setup
(scripts/1_setup/download_datasets.py). One shared pin, because in offline mode
``datasets`` ignores ``revision`` and serves whatever is cached.

Pinned here:
1. OFFLINE (fake ``datasets`` module, the tests/test_dataset_loaders.py
   pattern): the loader call carries the pinned revision and no config; the
   staging script carries the same pin for qasper and nothing new for the
   other datasets; both pins are one value; the tracked 50x3 manifest still
   has the digest the live test compares against.
2. LIVE, opt-in (``CAGE_HF_LIVE=1``, marker ``integration``): the real
   ``datasets`` library loads the validation split through the pinned route
   and the rebuilt 50x3 manifest hashes to the tracked digest. This is the
   acceptance check; setup_runpod.sh runs it on every pod after staging.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "1_setup" / "download_datasets.py"
MANIFEST = REPO_ROOT / "data" / "manifests" / "qasper_50x3_seed42.json"
SETUP_RUNPOD = REPO_ROOT / "scripts" / "runpod" / "setup_runpod.sh"

#: The Hub's parquet-converter commit for allenai/qasper (refs/convert/parquet,
#: read 2026-09-30); the loader and the staging script must both carry it.
QASPER_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
#: sha256 of data/manifests/qasper_50x3_seed42.json (the S0 manifest; the
#: parquet route reproduced it end to end on 2026-09-30).
MANIFEST_SHA256 = "04171d4f992f1ce97bddabae6d40554af1b4ee9f63713be3882490be4599fb2f"


def _install_fake_datasets(monkeypatch, rows):
    """A ``datasets`` stand-in whose load_dataset records (args, kwargs)."""
    calls = []

    class _FakeDataset(list):
        def shuffle(self, seed=None):
            return self

        def select(self, indices):
            return _FakeDataset(self[i] for i in indices)

        def keys(self):
            return ["validation"]

        def __getitem__(self, key):
            if isinstance(key, str):
                return list(self)
            return super().__getitem__(key)

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeDataset(rows)

    fake = types.ModuleType("datasets")
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return calls


def _load_download_module():
    spec = importlib.util.spec_from_file_location("cage_download_datasets_s0f5", DOWNLOAD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# offline pins
# ---------------------------------------------------------------------------


def test_loader_exports_the_pinned_revision() -> None:
    from src.data.loader import QASPER_REVISION as pinned

    assert pinned == QASPER_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", pinned), "a full commit sha, never a moving ref"


def test_qasper_loader_passes_revision_and_no_config(monkeypatch) -> None:
    from src.data.loader import QasperLoader

    calls = _install_fake_datasets(monkeypatch, [])
    QasperLoader(split="validation", seed=42).load()
    assert calls == [(("allenai/qasper",), {"split": "validation", "revision": QASPER_REVISION})]


def test_download_script_pins_the_same_revision_for_qasper_only(monkeypatch) -> None:
    calls = _install_fake_datasets(monkeypatch, [1, 2, 3])
    module = _load_download_module()
    assert module.DATASET_REVISIONS == {"allenai/qasper": QASPER_REVISION}
    # the (hf_path, config) tuple gate (p) and the staging tests unpack is unchanged
    assert module.dataset_specs()["qasper"] == [("allenai/qasper", None)]

    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "qasper"])
    assert module.main() == 0
    assert calls == [(("allenai/qasper",), {"split": None, "revision": QASPER_REVISION})]

    calls.clear()
    monkeypatch.setattr(sys, "argv", ["download_datasets.py", "--dataset", "hotpotqa"])
    assert module.main() == 0
    assert calls == [(("hotpotqa/hotpot_qa", "distractor"), {"split": None})]


def test_both_call_sites_carry_one_pin(monkeypatch) -> None:
    from src.data.loader import QASPER_REVISION as loader_pin

    _install_fake_datasets(monkeypatch, [])
    module = _load_download_module()
    assert module.DATASET_REVISIONS["allenai/qasper"] == loader_pin


def test_tracked_manifest_has_the_acceptance_digest() -> None:
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert digest == MANIFEST_SHA256


def test_setup_runpod_runs_the_live_check_after_staging() -> None:
    text = SETUP_RUNPOD.read_text(encoding="utf-8")
    stage = text.index("staging charter datasets")
    live = text.index("CAGE_HF_LIVE=1")
    assert stage < live, "the live digest check runs after the dataset stage"
    assert "tests/test_qasper_revision_s0f5.py" in text


# ---------------------------------------------------------------------------
# live, opt-in: the acceptance check (the Hub route reproduces the S0 manifest)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("CAGE_HF_LIVE") != "1",
                    reason="opt-in: set CAGE_HF_LIVE=1 to load allenai/qasper from the Hub")
def test_live_hub_route_reproduces_the_50x3_manifest_digest() -> None:
    pytest.importorskip("datasets", reason="the real datasets library is the thing under test")
    from src.data.loader import get_loader, gold_only
    from src.data.manifest import build_manifest

    loader = get_loader("qasper", seed=42)
    examples = loader.load()
    manifest = build_manifest(
        examples, num_queries=50, num_trials=3, seed=42, block_budget=2800,
        dataset="qasper", split=loader.split, context_selector=gold_only,
        trunc_budgets=(1400, 700),
    )
    rebuilt = json.dumps(manifest, indent=1).encode("utf-8")
    assert hashlib.sha256(rebuilt).hexdigest() == MANIFEST_SHA256, (
        f"the pinned Hub route no longer reproduces {MANIFEST.name}: "
        f"{manifest['stats']}"
    )
