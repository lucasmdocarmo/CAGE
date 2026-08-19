"""Offline coverage for src/observability/provenance.py (K-COV10, #142).

The provenance module was at 0% — its default/fallback arms would have fired
FIRST on an expensive GPU box. Every collector's fail-soft contract (record
None + warn, never raise) is pinned here offline:

- BUILD_INFO parsing + the tarball-deploy git_sha/git_dirty fallbacks
- installed_package_commit's direct_url.json arms
- vllm/torch version probes (import stubbed both ways)
- instrument_versions None-for-missing recording
- sha256_file / collect_dataset_fingerprints byte-exact hashing + None arms
- gpu_info on a fake pynvml (bytes-decode + per-field failure arms)
- GCP metadata parsing (stubbed urlopen) and the off-cloud None shape
- build_manifest end-to-end off-cloud + write_manifest atomicity + the
  dirty-tree warning; write_provenance pattern-scoped hashing

Device layer and network are stubbed via sys.modules / monkeypatch: no GPU,
no metadata server, no git repo required (tmp dirs are outside the CAGE
repo, so `git rev-parse` legitimately fails there and the fallback fires).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import types
import urllib.error
from pathlib import Path

import pytest

import src.observability.provenance as prov


# --------------------------------------------------------------------------- #
# BUILD_INFO parsing + git fallbacks (tarball deploys)
# --------------------------------------------------------------------------- #


class TestBuildInfoAndGitFallbacks:
    def test_build_info_parses_key_value_lines(self, tmp_path: Path):
        (tmp_path / "BUILD_INFO").write_text(
            "sha=abc123\ndirty=1\npackaged_at=2026-08-18T00:00:00Z\n"
            "malformed line without equals\n",
            encoding="utf-8",
        )
        info = prov._build_info(str(tmp_path))
        assert info == {
            "sha": "abc123", "dirty": "1", "packaged_at": "2026-08-18T00:00:00Z"
        }

    def test_absent_build_info_is_empty(self, tmp_path: Path):
        assert prov._build_info(str(tmp_path)) == {}

    def test_git_sha_falls_back_to_build_info_off_repo(self, tmp_path: Path):
        # tmp_path is not a git repo: rev-parse fails, BUILD_INFO travels
        # with the tarball (the 2026-07-15 smoke-manifest lesson).
        (tmp_path / "BUILD_INFO").write_text("sha=deadbeef\n", encoding="utf-8")
        assert prov.git_sha(str(tmp_path)) == "deadbeef"

    def test_git_sha_none_when_nothing_available(self, tmp_path: Path):
        assert prov.git_sha(str(tmp_path)) is None

    @pytest.mark.parametrize("flag,expected", [("1", True), ("0", False)])
    def test_git_dirty_falls_back_to_build_info(self, tmp_path, flag, expected):
        (tmp_path / "BUILD_INFO").write_text(f"dirty={flag}\n", encoding="utf-8")
        assert prov.git_dirty(str(tmp_path)) is expected

    def test_git_dirty_none_when_unknowable(self, tmp_path: Path):
        assert prov.git_dirty(str(tmp_path)) is None

    def test_git_dirty_from_porcelain_output(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(prov, "_run_cmd", lambda *a, **k: " M src/foo.py")
        assert prov.git_dirty(str(tmp_path)) is True
        monkeypatch.setattr(prov, "_run_cmd", lambda *a, **k: "")
        assert prov.git_dirty(str(tmp_path)) is False

    def test_run_cmd_failure_records_none(self):
        assert prov._run_cmd(["false"]) is None
        assert prov._run_cmd(["/no/such/binary-xyz"]) is None


# --------------------------------------------------------------------------- #
# Package/version probes
# --------------------------------------------------------------------------- #


class TestVersionProbes:
    def test_installed_package_commit_none_for_absent_dist(self):
        assert prov.installed_package_commit("definitely-not-a-package-xyz") is None

    def test_installed_package_commit_reads_direct_url(self, monkeypatch):
        class FakeDist:
            def read_text(self, name):
                assert name == "direct_url.json"
                return json.dumps({
                    "url": "https://example.invalid/repo.git",
                    "vcs_info": {"vcs": "git", "commit_id": "1cb902e2"},
                })

        import importlib.metadata as md

        monkeypatch.setattr(md, "distribution", lambda name: FakeDist())
        assert prov.installed_package_commit("cage-stats") == "1cb902e2"

    def test_installed_package_commit_none_for_non_vcs_install(self, monkeypatch):
        class FakeDist:
            def read_text(self, name):
                return None  # wheel install: no direct_url.json

        import importlib.metadata as md

        monkeypatch.setattr(md, "distribution", lambda name: FakeDist())
        assert prov.installed_package_commit("cage-stats") is None

    def test_vllm_version_from_stub_and_none_when_missing(self, monkeypatch):
        fake = types.ModuleType("vllm")
        fake.__version__ = "0.19.1"
        monkeypatch.setitem(sys.modules, "vllm", fake)
        assert prov.vllm_version() == "0.19.1"
        monkeypatch.setitem(sys.modules, "vllm", None)  # ImportError arm
        assert prov.vllm_version() is None

    def test_torch_version_stub_and_missing(self, monkeypatch):
        fake = types.ModuleType("torch")
        fake.__version__ = "2.9.0"
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert prov._torch_version() == "2.9.0"
        monkeypatch.setitem(sys.modules, "torch", None)
        assert prov._torch_version() is None

    def test_instrument_versions_records_none_for_missing(self):
        out = prov.instrument_versions(["numpy", "definitely-not-a-package-xyz"])
        assert out["numpy"]  # installed in the analysis venv
        assert out["definitely-not-a-package-xyz"] is None  # honest null


# --------------------------------------------------------------------------- #
# Hashing: files tied to exact bytes on disk
# --------------------------------------------------------------------------- #


class TestHashing:
    def test_sha256_file_matches_hashlib(self, tmp_path: Path):
        payload = b"cage provenance bytes\n" * 100
        path = tmp_path / "results.csv"
        path.write_bytes(payload)
        assert prov.sha256_file(str(path)) == hashlib.sha256(payload).hexdigest()

    def test_sha256_file_none_when_unreadable(self, tmp_path: Path):
        assert prov.sha256_file(str(tmp_path / "missing.csv")) is None

    def test_collect_dataset_fingerprints_shape(self, tmp_path: Path):
        good = tmp_path / "manifest.json"
        good.write_bytes(b"{}")
        out = prov.collect_dataset_fingerprints({
            "query_manifest": str(good),
            "missing_corpus": str(tmp_path / "gone.json"),
        })
        assert out["query_manifest"] == {
            "path": str(good),
            "sha256": hashlib.sha256(b"{}").hexdigest(),
            "size_bytes": 2,
        }
        assert out["missing_corpus"]["sha256"] is None
        assert out["missing_corpus"]["size_bytes"] is None


# --------------------------------------------------------------------------- #
# GPU info on a fake device layer
# --------------------------------------------------------------------------- #


def _fake_pynvml(*, init_raises=False, cuda_raises=False):
    mod = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    class _Mem:
        total = 24 * 1024 * 1024 * 1024  # 24576 MB

    mod.NVMLError = NVMLError

    def nvmlInit():
        if init_raises:
            raise NVMLError("no driver")

    mod.nvmlInit = nvmlInit
    mod.nvmlShutdown = lambda: None
    mod.nvmlDeviceGetCount = lambda: 1
    mod.nvmlDeviceGetHandleByIndex = lambda i: "h0"
    mod.nvmlDeviceGetName = lambda h: b"NVIDIA L4"  # bytes-decode arm
    mod.nvmlDeviceGetMemoryInfo = lambda h: _Mem()
    mod.nvmlSystemGetDriverVersion = lambda: "555.42.02"

    def nvmlSystemGetCudaDriverVersion():
        if cuda_raises:
            raise NVMLError("old driver")
        return 12040

    mod.nvmlSystemGetCudaDriverVersion = nvmlSystemGetCudaDriverVersion
    return mod


class TestGpuInfo:
    def test_full_population_with_bytes_decode(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml())
        assert prov.gpu_info() == {
            "name": "NVIDIA L4",
            "memory_total_mb": 24576,
            "driver_version": "555.42.02",
            "cuda_version": 12040,
            "device_count": 1,
        }

    def test_cuda_probe_failure_records_none_only_there(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(cuda_raises=True))
        info = prov.gpu_info()
        assert info["cuda_version"] is None
        assert info["name"] == "NVIDIA L4"  # rest of the record survives

    def test_nvml_init_failure_records_all_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(init_raises=True))
        assert prov.gpu_info() == {
            "name": None, "memory_total_mb": None, "driver_version": None,
            "cuda_version": None, "device_count": None,
        }


# --------------------------------------------------------------------------- #
# GCP metadata (stubbed urlopen — no metadata server, no network)
# --------------------------------------------------------------------------- #


class _FakeMetaResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestGcpMetadata:
    def test_on_gce_parses_zone_and_machine_type_tails(self, monkeypatch):
        answers = {
            "instance/zone": b"projects/123/zones/us-east1-b",
            "instance/machine-type": b"projects/123/machineTypes/g2-standard-8",
            "instance/name": b"cage-l4-1",
            "instance/scheduling/preemptible": b"TRUE",
            "project/project-id": b"cage-project",
        }

        def fake_urlopen(req, timeout=None):
            path = req.full_url.replace(f"{prov._GCP_METADATA_ROOT}/", "")
            assert req.headers.get("Metadata-flavor") == "Google"
            return _FakeMetaResponse(answers[path])

        monkeypatch.setattr(prov.urllib.request, "urlopen", fake_urlopen)
        meta = prov.gcp_instance_metadata()
        assert meta == {
            "on_gce": True,
            "name": "cage-l4-1",
            "zone": "us-east1-b",
            "machine_type": "g2-standard-8",
            "preemptible": "TRUE",
            "project": "cage-project",
        }

    def test_off_cloud_is_a_none_shaped_noop(self, monkeypatch):
        def fake_urlopen(req, timeout=None):
            raise urllib.error.URLError("no metadata server here")

        monkeypatch.setattr(prov.urllib.request, "urlopen", fake_urlopen)
        meta = prov.gcp_instance_metadata()
        assert meta["on_gce"] is False
        assert all(
            meta[k] is None
            for k in ("name", "zone", "machine_type", "preemptible", "project")
        )


# --------------------------------------------------------------------------- #
# build_manifest / write_manifest / write_provenance
# --------------------------------------------------------------------------- #


def _offline(monkeypatch, tmp_path: Path) -> Path:
    """Deterministic off-cloud environment for build_manifest."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "BUILD_INFO").write_text("sha=cafebabe\ndirty=1\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml(init_raises=True))
    monkeypatch.setattr(
        prov.urllib.request, "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("off")),
    )
    return repo


class TestManifestEndToEnd:
    def test_build_manifest_off_cloud_defaults(self, monkeypatch, tmp_path: Path):
        repo = _offline(monkeypatch, tmp_path)
        manifest = prov.build_manifest(
            run_id="run-1",
            created_at="2026-08-18T12:00:00+00:00",
            cage_repo_dir=str(repo),
            model="Qwen/Qwen3-8B",
            dataset="squad_v2",
            seed=42,
            gpu_memory_utilization=0.85,
            extra={"note": "offline test"},
        )
        assert manifest.run_id == "run-1"
        assert manifest.created_at == "2026-08-18T12:00:00+00:00"  # caller's clock
        assert manifest.cage_git_sha == "cafebabe"      # BUILD_INFO fallback
        assert manifest.cage_git_dirty is True
        assert manifest.vllm_version is None            # fail-soft nulls
        assert manifest.torch_version is None
        assert manifest.gpu["name"] is None
        assert manifest.gcp_instance["on_gce"] is False
        assert manifest.gpu_memory_utilization == 0.85
        assert manifest.instrument_versions["numpy"]    # auto-collected
        assert manifest.extra == {"note": "offline test"}
        d = manifest.to_dict()
        assert d["model"] == "Qwen/Qwen3-8B" and d["seed"] == 42

    def test_write_manifest_round_trips_and_warns_on_dirty(
        self, monkeypatch, tmp_path: Path, caplog
    ):
        repo = _offline(monkeypatch, tmp_path)
        manifest = prov.build_manifest(
            run_id="run-1", created_at="2026-08-18T12:00:00+00:00",
            cage_repo_dir=str(repo),
        )
        out = tmp_path / "obs" / "run_manifest.json"
        with caplog.at_level(logging.WARNING, logger="cage.observability.provenance"):
            prov.write_manifest(manifest, str(out))
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc == manifest.to_dict()
        assert not out.with_suffix(out.suffix + ".tmp").exists()  # atomic
        assert any("DIRTY" in rec.message for rec in caplog.records)

    def test_write_provenance_hashes_only_result_patterns(self, tmp_path: Path):
        run = tmp_path / "run"
        (run / "baselines" / "no_cache").mkdir(parents=True)
        csv = run / "baselines" / "no_cache" / "results.csv"
        csv.write_bytes(b"a,b\n1,2\n")
        metrics = run / "baselines" / "no_cache" / "trial_metrics.json"
        metrics.write_bytes(b"{}")
        (run / "notes.txt").write_bytes(b"not a result artifact")
        out = tmp_path / "provenance.json"
        payload = prov.write_provenance(
            str(run), str(out), created_at="2026-08-18T12:00:00+00:00"
        )
        assert payload["file_count"] == 2
        assert payload["generated_at"] == "2026-08-18T12:00:00+00:00"
        rel = "baselines/no_cache/results.csv"
        assert payload["files"][rel]["sha256"] == (
            hashlib.sha256(b"a,b\n1,2\n").hexdigest()
        )
        assert payload["files"][rel]["size_bytes"] == 8
        assert "notes.txt" not in payload["files"]
        # The written JSON is the returned payload, byte-parseable.
        assert json.loads(out.read_text(encoding="utf-8")) == payload
