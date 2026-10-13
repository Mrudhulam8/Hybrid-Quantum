"""Common search score (Phase 6). The score RANKS candidates for an optimiser; it never decides failure.

search_score != verified_failure.  Only `FailureValidator` establishes `verified_failure`. This module
reads a `ValidationResult` but NEVER reads its `verified_failure` / `failure_reason` fields (asserted
in tests), so no optimiser can be rewarded for - or can mark - a verdict directly.

Valid, queried candidate (value in [0, 1]):
    score = ( w_u * U + w_f * F + w_n * N ) / (w_u + w_f + w_n)
    U = target-model entropy / ln(C)                       (uncertainty; informational, NOT a failure criterion)
    F = 1 - p_target(oracle_label)  if the oracle decided with confidence >= the validator's
        min_oracle_confidence, else 0                      (surrogate failure likelihood: target mass off the oracle label)
    N = 1 - exp(-d_min / novelty_length)                   (novelty: RMS-z distance to earlier queried candidates of THIS run)
Invalid candidate (never queried):
    score = -(invalid_base + invalid_slope * min(excess, invalid_excess_cap))
    excess = max(0, M/tau_M - 1, nn/tau_nn - 1)            (how far outside the validity gates; reference-only, query-free)
Every valid score is therefore >= 0 and every invalid score is <= -invalid_base < 0.

Defaults (w_u=0.5, w_f=0.5, w_n=0) are a-priori placeholders fixed BEFORE any experiment; they must
not be tuned on failure outcomes.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from .validator import ValidationResult


@dataclass(frozen=True)
class SearchScoreConfig:
    w_uncertainty: float = 0.5
    w_failure_likelihood: float = 0.5
    w_novelty: float = 0.0
    novelty_length: float = 0.1
    invalid_base: float = 0.1
    invalid_slope: float = 0.1
    invalid_excess_cap: float = 10.0

    @classmethod
    def from_cfg(cls, d: Dict[str, Any]) -> "SearchScoreConfig":
        c = cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
        c.validate()
        return c

    def validate(self) -> None:
        ok = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        bad = []
        for k in ("w_uncertainty", "w_failure_likelihood", "w_novelty"):
            if not (ok(getattr(self, k)) and getattr(self, k) >= 0): bad.append(f"{k} must be a finite number >= 0")
        if not bad and self.w_uncertainty + self.w_failure_likelihood + self.w_novelty <= 0:
            bad.append("at least one score weight must be > 0")
        for k in ("novelty_length", "invalid_base"):
            if not (ok(getattr(self, k)) and getattr(self, k) > 0): bad.append(f"{k} must be > 0")
        for k in ("invalid_slope", "invalid_excess_cap"):
            if not (ok(getattr(self, k)) and getattr(self, k) >= 0): bad.append(f"{k} must be >= 0")
        if bad:
            raise ValueError("Invalid search objective config: " + "; ".join(bad))


class SearchScore:
    """Stateful only through the novelty history of ONE run (create a fresh instance per run)."""

    def __init__(self, cfg: SearchScoreConfig, min_oracle_confidence: float, scale: Optional[np.ndarray] = None):
        cfg.validate()
        self.cfg, self.min_oracle_confidence = cfg, float(min_oracle_confidence)
        self.scale = None if scale is None else np.where(np.asarray(scale, float) > 0, scale, 1.0)
        self._hist: list = []

    def describe(self) -> Dict[str, Any]:
        return {**asdict(self.cfg), "min_oracle_confidence": self.min_oracle_confidence}

    # ------------------------------------------------------------------
    def components(self, r: ValidationResult, x: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Uses ONLY: candidate_valid, queried, target_proba/entropy, oracle_label/confidence and the
        validity distances/thresholds. Never `verified_failure` or `failure_reason`."""
        if not (r.candidate_valid and r.queried and r.target_proba is not None):
            return {"valid": 0.0, "uncertainty": 0.0, "failure_likelihood": 0.0, "novelty": 0.0,
                    "excess": self._excess(r)}
        p = np.asarray(r.target_proba, float)
        U = float(r.target_entropy) / math.log(len(p)) if len(p) > 1 else 0.0
        F = 0.0
        if r.oracle_label is not None and r.oracle_confidence is not None \
                and r.oracle_confidence >= self.min_oracle_confidence:
            F = float(1.0 - p[int(r.oracle_label)])
        N = 0.0
        if self.cfg.w_novelty > 0 and x is not None and self._hist:
            H = np.stack(self._hist)
            s = self.scale if self.scale is not None else 1.0
            d = np.sqrt((((H - x) / s) ** 2).mean(axis=1)).min()
            N = float(1.0 - math.exp(-d / self.cfg.novelty_length))
        elif self.cfg.w_novelty > 0 and x is not None:
            N = 1.0
        return {"valid": 1.0, "uncertainty": min(max(U, 0.0), 1.0), "failure_likelihood": min(max(F, 0.0), 1.0),
                "novelty": N, "excess": 0.0}

    def _excess(self, r: ValidationResult) -> float:
        cap = self.cfg.invalid_excess_cap
        ex = []
        if r.distance_to_reference is not None and r.mahalanobis_threshold:
            ex.append(r.distance_to_reference / r.mahalanobis_threshold - 1.0)
        if r.nearest_neighbor_distance is not None and r.nn_distance_threshold \
                and math.isfinite(r.nn_distance_threshold):
            ex.append(r.nearest_neighbor_distance / r.nn_distance_threshold - 1.0)
        if r.distance_to_reference is None and r.nearest_neighbor_distance is None:
            return cap                                          # malformed candidate: worst penalty
        return float(min(max([0.0] + ex), cap))

    def score(self, r: ValidationResult, x: Optional[np.ndarray] = None) -> float:
        c = self.components(r, x)
        if c["valid"] == 0.0:
            return -(self.cfg.invalid_base + self.cfg.invalid_slope * c["excess"])
        w = self.cfg
        tot = w.w_uncertainty + w.w_failure_likelihood + w.w_novelty
        s = (w.w_uncertainty * c["uncertainty"] + w.w_failure_likelihood * c["failure_likelihood"]
             + w.w_novelty * c["novelty"]) / tot
        if w.w_novelty > 0 and x is not None:
            self._hist.append(np.asarray(x, float))
        return float(s)
