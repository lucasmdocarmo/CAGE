"""Multiplicity corrections (§9.3): Holm within family, BH-FDR exploratory.

Pure functions over p-value vectors; adjusted p-values returned in the input
order. Tested against scipy / statsmodels reference values in
tests/test_stats_engine.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_pvals(pvals: Any) -> np.ndarray:
    arr = np.asarray(pvals, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"p-values must be 1-D, got shape {arr.shape}")
    if arr.size and (not np.all(np.isfinite(arr)) or arr.min() < 0.0 or arr.max() > 1.0):
        raise ValueError(f"p-values must all lie in [0, 1] and be finite; got {arr!r}")
    return arr


def holm(pvals: Any) -> np.ndarray:
    """Holm step-down adjusted p-values (Holm 1979). FWER control."""
    p = _as_pvals(pvals)
    m = p.size
    if m == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(p, kind="stable")
    factors = m - np.arange(m)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(factors * p[order]))
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def benjamini_hochberg(pvals: Any) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (BH 1995). FDR control."""
    p = _as_pvals(pvals)
    m = p.size
    if m == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, m + 1)
    scaled = p[order] * m / ranks
    adjusted_sorted = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted
