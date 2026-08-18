"""Regression pins for run_memory_sweep.sh's EXIT-trap lifecycle (2026-08-02 finding).

CONFIRMED finding: `trap cleanup EXIT` silently REPLACED the earlier EXIT trap that
stopped the gcs_backup_daemon started just before it (bash keeps ONE handler per
signal), and cleanup() chained only __lg_cleanup -- so the daemon's detached setsid
sync loop was orphaned on every memory-sweep exit and its stop-time "final
authoritative sync" (gcs_backup_daemon.sh header contract) never ran.

These are STATIC contract checks (no GPU, no GCS, no vLLM binary): they pin the trap
structure of the script itself, plus one behavioral check that the fixed structure --
a later trap whose handler folds in the daemon stop -- fires the stop exactly once
under real bash trap-replacement semantics.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SH = REPO_ROOT / "scripts" / "3_run" / "run_memory_sweep.sh"


def _sweep_text() -> str:
    return SWEEP_SH.read_text(encoding="utf-8")


def _cleanup_body() -> str:
    m = re.search(r"^cleanup\(\) \{\n(.*?)^\}", _sweep_text(), flags=re.M | re.S)
    assert m, "run_memory_sweep.sh must define a cleanup() function"
    return m.group(1)


def test_sweep_script_parses() -> None:
    subprocess.run(["bash", "-n", str(SWEEP_SH)], check=True)


def test_last_exit_trap_is_cleanup() -> None:
    """The EXIT handler that actually fires is the LAST one registered: it must be
    cleanup, and any earlier EXIT trap (the interim daemon-stop trap) is therefore
    dead on the normal path -- cleanup() must carry its work forward."""
    traps = re.findall(r"^\s*trap\s+(.+?)\s+EXIT\s*$", _sweep_text(), flags=re.M)
    assert traps, "expected at least one `trap ... EXIT` in run_memory_sweep.sh"
    assert traps[-1] == "cleanup", (
        "the final registered EXIT trap must be `trap cleanup EXIT`; a later trap "
        f"would orphan cleanup() exactly like the original bug (got: {traps[-1]!r})"
    )


def test_cleanup_stops_gcs_daemon_before_log_guard_chain() -> None:
    """cleanup() must (a) stop the gcs backup daemon on the sweep's own phase tree
    and (b) do so BEFORE chaining __lg_cleanup, per the confirmed-finding fix."""
    body = _cleanup_body()
    stop_at = body.find("gcs_backup_daemon.sh stop")
    lg_at = body.find("__lg_cleanup")
    assert stop_at != -1, (
        "cleanup() must stop gcs_backup_daemon.sh: `trap cleanup EXIT` replaces the "
        "interim stop trap, so without this the setsid sync loop is orphaned and the "
        "final authoritative sync never runs"
    )
    assert lg_at != -1, "cleanup() must still chain the log-guard's __lg_cleanup"
    assert stop_at < lg_at, "daemon stop must precede the __lg_cleanup chain"
    assert '"results/${_MS_PHASE}"' in body, (
        "daemon stop must target the same quoted results/<phase> tree the start used, "
        "so stop addresses the same per-scope daemon instance"
    )


def test_daemon_start_still_paired_with_interim_trap() -> None:
    """The start call keeps an immediately-following interim EXIT trap so an exit in
    the window before cleanup() is registered still stops the daemon."""
    text = _sweep_text()
    start_at = text.find("gcs_backup_daemon.sh start")
    assert start_at != -1, "run_memory_sweep.sh must start the gcs backup daemon"
    # 1200 (was 600): the interim trap's comment grew when task #136 (Topic-10 J8)
    # made it chain __lg_cleanup too; the pinned contract itself is unchanged.
    window = text[start_at : start_at + 1200]
    assert re.search(r"trap 'bash scripts/5_observability/gcs_backup_daemon\.sh stop ", window), (
        "the daemon start must be followed by the interim EXIT stop trap covering the "
        "gap before `trap cleanup EXIT` replaces it"
    )


def test_bash_trap_replacement_fold_in_fires_stop_exactly_once(tmp_path: Path) -> None:
    """Behavioral pin of the fix's mechanism under real bash: a later `trap cleanup
    EXIT` REPLACES the interim daemon-stop trap (the confirmed bug), and folding the
    stop into cleanup() fires it exactly once, before the __lg_cleanup chain."""
    marker = tmp_path / "markers"
    script = r"""
set -uo pipefail
daemon_stop() { printf 'daemon_stop\n' >> "$MARKER"; }
trap 'daemon_stop' EXIT                       # interim trap (daemon-start analogue)
__lg_cleanup() { printf 'lg_cleanup\n' >> "$MARKER"; }
cleanup() {
  printf 'vllm_stop\n' >> "$MARKER"
  daemon_stop
  type __lg_cleanup >/dev/null 2>&1 && __lg_cleanup
}
trap cleanup EXIT                             # REPLACES the interim trap
exit 0
"""
    subprocess.run(
        ["bash", "-c", script],
        check=True,
        env={**os.environ, "MARKER": str(marker)},
    )
    fired = marker.read_text(encoding="utf-8").split()
    # The interim trap did NOT fire on its own (it was replaced -- the bug mechanism);
    # the fold-in fired the stop exactly once, and before the log-guard chain.
    assert fired == ["vllm_stop", "daemon_stop", "lg_cleanup"], fired
