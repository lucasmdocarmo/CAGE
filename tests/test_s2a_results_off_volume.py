"""S2A keeps the results root and the backup mirror off the network volume, on a
400 GB container disk (owner 2026-11-21, final review questions 1 (a) and 4 (b):
"approve all as recomended"; MyDocs/RunPod/S2_FINAL_REVIEW_2026-11-21.md section 12.1).

Every cell directory is its row key, which carries "|". The US-CA-2 network volume
(NFS, vast backend) refused that character on landing S2R 2026-11-14 (follow-ups
items 1 and 6), US-CA-2 is first in S2A's DC_PREFS, and stage 2 stops a landing whose
results root refuses it. S2R.env and S2B.env carry the three keys since 2026-11-15
(tests/test_s2_rehearsal_profile.py); these tests hold S2A.env to the same values.
"""
from __future__ import annotations

from test_s2_rehearsal_profile import _shipped, _stated

#: the values S2B.env and S2R.env state (follow-ups item 6, option B)
OFF_VOLUME = {
    "POD_RESULTS_DIR": "/root/cage-results",
    "POD_BACKUP_DIR": "/root/backup",
    "BACKUP_TARGET": "file:///root/backup",
}


def test_s2a_keeps_the_results_and_the_backup_off_the_volume() -> None:
    """The sourced values are read too: _common.env hard-sets BACKUP_TARGET to the
    volume, so a profile line that did not win would leave the mirror there."""
    assert _shipped("S2A.env", *OFF_VOLUME) == OFF_VOLUME
    stated = _stated("S2A.env")
    assert {k: stated.get(k) for k in OFF_VOLUME} == OFF_VOLUME


def test_s2a_states_the_same_off_volume_values_as_s2b() -> None:
    """One value per key across the two landing profiles: a later edit of one side
    that the other does not mirror fails here."""
    a, b = _stated("S2A.env"), _stated("S2B.env")
    assert {k: a.get(k) for k in OFF_VOLUME} == {k: b.get(k) for k in OFF_VOLUME}


def test_s2a_states_a_400_gb_container_disk_and_keeps_the_300_gb_volume() -> None:
    """The results and the mirror now share the disk with the 141 GB of weights, and
    the S2R pod's 300 GB disk lost 26 GB over 18.6 h with 2.7 GB explained."""
    assert _shipped("S2A.env", "DISK_GB", "VOLUME_GB") == {"DISK_GB": "400", "VOLUME_GB": "300"}
    assert _stated("S2A.env")["DISK_GB"] == "400"


def test_s2a_keeps_the_archive_of_the_run_tree_on_the_volume() -> None:
    """The one copy that survives the pod's own disk: an archive's name carries no
    "|", so it stays on the volume (true before and after the change)."""
    assert _shipped("S2A.env", "BACKUP_ARCHIVE_DIR") == {"BACKUP_ARCHIVE_DIR": "/workspace/backup-archives"}
