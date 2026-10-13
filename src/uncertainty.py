"""Uncertainty measures.

Phase 4 contains ONLY the scalar measures, so validation results can *record* model uncertainty.
They are informational: **uncertainty is never a failure criterion** (see src/validator.py).
The Classical Controller (probing, region bounds) is NOT implemented yet.
"""
from __future__ import annotations

import numpy as np


def prediction_entropy(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """H(x) = -sum_c p_c ln p_c (natural log; maximum ln C). Accepts (C,) or (n, C)."""
    p = np.atleast_2d(np.asarray(proba, dtype=np.float64))
    return -(p * np.log(p + eps)).sum(axis=1)


def prediction_margin(proba: np.ndarray) -> np.ndarray:
    """Margin = p_(1) - p_(2) (top-two probabilities). Lower margin = more uncertain."""
    p = np.atleast_2d(np.asarray(proba, dtype=np.float64))
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]
