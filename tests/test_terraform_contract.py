"""Regression pins for the terraform <-> docs contracts (2026-08-02 findings).

These are STATIC contract checks (no cloud, no terraform binary required except
for the optional validate test): they pin the exact invariants whose drift
caused two CONFIRMED campaign-blocking findings:

1. run_id grammar: terraform/variables.tf must validate run_id against the
   SAME RESULTS_LAYOUT.md §1 bucket-name grammar the analysis layer uses
   (organize_results.RUN_ID_RE), and main.tf must use run_id VERBATIM as the
   slug — otherwise the bucket terraform creates (cage-<slug>) and the
   gs://cage-<run_id> the RUNBOOK exports on the node diverge, and the sync
   daemon writes a whole session into a nonexistent bucket.

2. Boot-disk labels: GCE does NOT propagate instance labels to boot disks, so
   modules/gpu_session must set labels inside boot_disk.initialize_params or
   the TRUE-$0 disk orphan sweep (`gcloud compute disks list
   --filter='labels.agent-run=<run_id>'`) can never match a surviving disk.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = REPO_ROOT / "scripts" / "4_analysis"
for _p in (str(_SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import organize_results as org  # noqa: E402

TF_DIR = REPO_ROOT / "terraform"
VARIABLES_TF = TF_DIR / "variables.tf"
MAIN_TF = TF_DIR / "main.tf"
TFVARS_EXAMPLE = TF_DIR / "terraform.tfvars.example"
GPU_SESSION_TF = TF_DIR / "modules" / "gpu_session" / "main.tf"


def _run_id_validation_pattern() -> str:
    """Extract the regex inside the run_id variable's validation block."""
    text = VARIABLES_TF.read_text(encoding="utf-8")
    var_block = re.search(r'variable\s+"run_id"\s*\{.*?\n\}', text, flags=re.DOTALL)
    assert var_block, "variable \"run_id\" block not found in variables.tf"
    match = re.search(r'can\(regex\("([^"]+)",\s*var\.run_id\)\)', var_block.group(0))
    assert match, "run_id validation regex not found in variables.tf"
    return match.group(1)


# ---------------------------------------------------------------------------
# Finding 2 — run_id grammar drift (bucket-name divergence)
# ---------------------------------------------------------------------------


def test_terraform_run_id_grammar_matches_results_layout() -> None:
    """The terraform validation regex IS the §1 grammar the analysis layer pins."""
    assert _run_id_validation_pattern() == org.RUN_ID_RE.pattern


def test_terraform_run_id_grammar_rejects_underscore_run_ids() -> None:
    """The exact run_id the old tfvars.example shipped must be rejected."""
    pattern = re.compile(_run_id_validation_pattern())
    assert pattern.match("2026-08-02_0000_smoke") is None  # the shipped bug
    assert pattern.match("Run-1") is None  # uppercase
    assert pattern.match("a.b.c") is None  # dots
    assert pattern.match("20260815-0230-a-qwen3-14b")  # §1 example form


def test_tfvars_example_run_id_is_bucket_name_safe() -> None:
    """terraform.tfvars.example must ship a run_id that passes the grammar
    (an operator copies it as documented; RUNBOOK then exports
    gs://cage-<run_id> verbatim)."""
    text = TFVARS_EXAMPLE.read_text(encoding="utf-8")
    match = re.search(r'^run_id\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match, "run_id assignment not found in terraform.tfvars.example"
    run_id = match.group(1)
    assert org.RUN_ID_RE.match(run_id), (
        f"terraform.tfvars.example run_id {run_id!r} violates the §1 bucket "
        f"grammar {org.RUN_ID_RE.pattern} — bucket cage-<slug> would diverge "
        "from the documented gs://cage-<run_id>"
    )


def test_main_tf_uses_run_id_verbatim_as_slug() -> None:
    """No lossy re-slugging of run_id: slug == run_id by construction, so the
    bucket name cage-<run_id> can never diverge from the docs."""
    text = MAIN_TF.read_text(encoding="utf-8")
    assert re.search(r"^\s*run_slug\s*=\s*var\.run_id\s*$", text, flags=re.MULTILINE), (
        "main.tf must set run_slug = var.run_id (validation already enforces "
        "the bucket-safe grammar); a replace()-style slug reintroduces the "
        "bucket-name divergence class"
    )
    assert 'bucket_name = "cage-${local.run_slug}"' in text


# ---------------------------------------------------------------------------
# Finding 3 — boot disks must carry the orphan-sweep labels
# ---------------------------------------------------------------------------


def test_boot_disk_initialize_params_carries_labels() -> None:
    """boot_disk.initialize_params must set labels = var.labels: instance
    labels do NOT propagate to disks, and the TRUE-$0 disk sweep keys on
    labels.agent-run."""
    text = GPU_SESSION_TF.read_text(encoding="utf-8")
    block = re.search(r"boot_disk\s*\{.*?initialize_params\s*\{(.*?)\n\s*\}", text, flags=re.DOTALL)
    assert block, "boot_disk.initialize_params block not found in gpu_session/main.tf"
    assert re.search(r"^\s*labels\s*=\s*var\.labels\s*$", block.group(1), flags=re.MULTILINE), (
        "boot_disk.initialize_params must carry labels = var.labels — without "
        "it the disk orphan sweep (labels.agent-run) reads 'clean' while a "
        "surviving disk keeps billing"
    )


# ---------------------------------------------------------------------------
# Whole-stack syntax gate (runs only where terraform is installed + init'd)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("terraform") is None or not (TF_DIR / ".terraform").is_dir(),
    reason="terraform binary or .terraform providers not available",
)
def test_terraform_validate_passes() -> None:
    proc = subprocess.run(
        ["terraform", "validate", "-no-color"],
        cwd=TF_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"terraform validate failed:\n{proc.stdout}\n{proc.stderr}"
