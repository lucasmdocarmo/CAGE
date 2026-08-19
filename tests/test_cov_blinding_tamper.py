"""Tamper-guard coverage for the §9.8 sealed arm-label map (K-COV7, #142).

src/analysis/stats/blinding.py's refusal/tamper arms were the under-covered
part of the module: test_stats_proof.py pins the happy path, the double-seal /
double-unblind refusals and ONE hash-mismatch mutation, but the ``_load_sealed``
guard's other arms (missing file, invalid JSON, missing seal keys, non-object
mapping, added/removed mapping entries, forged unblind stamp) had never been
exercised. Every mutated blinded artifact below must be DETECTED and REFUSED.

All offline: tmp_path files only, no GPU, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.stats.blinding import (
    AlreadyUnblindedError,
    BlindingError,
    SealedMapTamperError,
    scramble_labels,
    unblind,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "arm": ["B3", "B6", "B3", "B6", "B9", "B9"],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })


def _seal(tmp_path: Path, seed: int = 11) -> Path:
    sealed = tmp_path / "blinding" / "sealed_map.json"
    scramble_labels(_frame(), "arm", seed=seed, sealed_map_path=sealed)
    return sealed


def _mutate(sealed: Path, fn) -> None:
    doc = json.loads(sealed.read_text(encoding="utf-8"))
    fn(doc)
    sealed.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# _load_sealed refusal arms (via unblind)
# --------------------------------------------------------------------------- #


class TestSealedMapTamperGuards:
    def test_missing_sealed_map_refused(self, tmp_path: Path):
        with pytest.raises(BlindingError, match="not found"):
            unblind(tmp_path / "nope.json", tmp_path / "unblind.log")

    def test_invalid_json_is_a_tamper_refusal(self, tmp_path: Path):
        sealed = _seal(tmp_path)
        sealed.write_text('{"seal_version": 1, TRUNCATED', encoding="utf-8")
        with pytest.raises(SealedMapTamperError, match="not valid JSON"):
            unblind(sealed, tmp_path / "unblind.log")

    @pytest.mark.parametrize("key", ["map_sha256", "mapping", "arm_col"])
    def test_missing_seal_key_refused(self, tmp_path: Path, key: str):
        sealed = _seal(tmp_path)
        _mutate(sealed, lambda doc: doc.pop(key))
        with pytest.raises(SealedMapTamperError, match=f"missing key {key!r}"):
            unblind(sealed, tmp_path / "unblind.log")

    def test_non_object_mapping_refused(self, tmp_path: Path):
        sealed = _seal(tmp_path)
        _mutate(sealed, lambda doc: doc.update(mapping=["B3", "ARM-01"]))
        with pytest.raises(SealedMapTamperError, match="not an object"):
            unblind(sealed, tmp_path / "unblind.log")

    def test_swapped_code_assignment_detected(self, tmp_path: Path):
        # Re-pointing one real label at a different blind code changes the
        # canonical mapping bytes -> sha mismatch.
        sealed = _seal(tmp_path)

        def swap(doc):
            labels = sorted(doc["mapping"])
            a, b = labels[0], labels[1]
            doc["mapping"][a], doc["mapping"][b] = (
                doc["mapping"][b], doc["mapping"][a]
            )

        _mutate(sealed, swap)
        with pytest.raises(SealedMapTamperError, match="hash mismatch"):
            unblind(sealed, tmp_path / "unblind.log")

    def test_added_mapping_entry_detected(self, tmp_path: Path):
        sealed = _seal(tmp_path)
        _mutate(sealed, lambda doc: doc["mapping"].update({"B99": "ARM-99"}))
        with pytest.raises(SealedMapTamperError, match="hash mismatch"):
            unblind(sealed, tmp_path / "unblind.log")

    def test_removed_mapping_entry_detected(self, tmp_path: Path):
        sealed = _seal(tmp_path)
        _mutate(sealed, lambda doc: doc["mapping"].pop(sorted(doc["mapping"])[0]))
        with pytest.raises(SealedMapTamperError, match="hash mismatch"):
            unblind(sealed, tmp_path / "unblind.log")

    def test_forged_unblind_stamp_still_refuses_reuse(self, tmp_path: Path):
        # Writing a fake unblinded_utc directly into the seal cannot re-open
        # it: the one-time guard fires on ANY non-null stamp.
        sealed = _seal(tmp_path)
        _mutate(sealed, lambda doc: doc.update(unblinded_utc="2020-01-01T00:00:00+00:00"))
        with pytest.raises(AlreadyUnblindedError, match="2020-01-01"):
            unblind(sealed, tmp_path / "unblind.log")


# --------------------------------------------------------------------------- #
# scramble_labels refusal arms not covered elsewhere
# --------------------------------------------------------------------------- #


class TestScrambleRefusals:
    def test_missing_labels_in_input_refused(self, tmp_path: Path):
        df = _frame()
        df.loc[2, "arm"] = None  # 3 distinct labels remain -> the NaN gate fires
        with pytest.raises(BlindingError, match="must be complete"):
            scramble_labels(df, "arm", seed=1, sealed_map_path=tmp_path / "s.json")
        assert not (tmp_path / "s.json").exists()  # refusal leaves no seal

    def test_blind_codes_are_opaque_and_mapping_never_returned(self, tmp_path: Path):
        sealed = tmp_path / "s.json"
        blinded, returned_path = scramble_labels(
            _frame(), "arm", seed=5, sealed_map_path=sealed
        )
        # The return value must not unblind anyone: a frame + the path only.
        assert returned_path == sealed
        assert set(blinded["arm"]) == {"ARM-01", "ARM-02", "ARM-03"}
        assert not set(blinded["arm"]) & {"B3", "B6", "B9"}
        # Non-label columns ride along untouched.
        assert list(blinded["value"]) == list(_frame()["value"])

    def test_same_seed_same_assignment_different_seed_may_differ(self, tmp_path: Path):
        a_path = tmp_path / "a.json"
        b_path = tmp_path / "b.json"
        a, _ = scramble_labels(_frame(), "arm", seed=5, sealed_map_path=a_path)
        b, _ = scramble_labels(_frame(), "arm", seed=5, sealed_map_path=b_path)
        assert list(a["arm"]) == list(b["arm"])
        map_a = json.loads(a_path.read_text(encoding="utf-8"))["mapping"]
        map_b = json.loads(b_path.read_text(encoding="utf-8"))["mapping"]
        assert map_a == map_b


# --------------------------------------------------------------------------- #
# The one-time unblind event log
# --------------------------------------------------------------------------- #


class TestUnblindEventLog:
    def test_unblind_appends_dated_event_and_stamps_seal(self, tmp_path: Path):
        sealed = _seal(tmp_path)
        log = tmp_path / "unblind_log.jsonl"
        sealed_before = json.loads(sealed.read_text(encoding="utf-8"))
        mapping = unblind(sealed, log)
        # Returned map is blind -> real (inverse of the sealed real -> blind).
        assert set(mapping) == set(sealed_before["mapping"].values())
        assert set(mapping.values()) == set(sealed_before["mapping"])
        # Exactly one dated JSONL event carrying the seal identity.
        lines = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(lines) == 1
        event = lines[0]
        assert event["event"] == "UNBLIND"
        assert event["map_sha256"] == sealed_before["map_sha256"]
        assert event["arm_col"] == "arm"
        assert event["sealed_map"] == str(sealed)
        assert event["utc"]
        # The seal itself is stamped with the same moment.
        sealed_after = json.loads(sealed.read_text(encoding="utf-8"))
        assert sealed_after["unblinded_utc"] == event["utc"]
