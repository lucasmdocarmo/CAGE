"""Per-query win/loss/tie counts — mandatory beside every paired test (§8.13).

An average that hides "helps 10 queries, destroys 8" is a lie of aggregation;
the W/L/T triple is binding on the §7.8 recipe and compiled by D9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WinLossTie:
    """W/L/T for arm A vs arm B over paired per-query values."""

    wins: int
    losses: int
    ties: int

    @property
    def n_pairs(self) -> int:
        return self.wins + self.losses + self.ties


def _as_float_1d(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} contains non-finite values (NaN/inf) — clean upstream, "
            f"no silent dropping here"
        )
    return arr


def win_loss_tie(
    a: Any, b: Any, *, higher_is_better: bool = True
) -> WinLossTie:
    """Count queries where arm A beats / loses to / ties arm B (paired)."""
    arr_a = _as_float_1d("a", a)
    arr_b = _as_float_1d("b", b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"paired arrays differ in length: {arr_a.size} vs {arr_b.size}")
    diffs = arr_a - arr_b
    n_pos = int(np.count_nonzero(diffs > 0))
    n_neg = int(np.count_nonzero(diffs < 0))
    ties = diffs.size - n_pos - n_neg
    if higher_is_better:
        return WinLossTie(wins=n_pos, losses=n_neg, ties=ties)
    return WinLossTie(wins=n_neg, losses=n_pos, ties=ties)
