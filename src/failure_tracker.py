"""Candidate -> valid -> verified failure -> unique failure bookkeeping (Phase 6).

Records only what the search itself produced: ids, method/seed, query number, validity, search score,
the validator's verdict and discovery order. It stores NO test labels and no held-out data.

* `verified_failures`        = number of DISTINCT candidate ids the validator verified as failures.
* `verified_failure_events`  = raw count incl. re-evaluations (only differs with duplicate_policy='allow').
* `unique_verified_failures` = greedy leader clustering, in discovery order, at Euclidean resolution
                               `validator.dedup_resolution` (scaled units): a new failure is unique if it is
                               >= resolution from every earlier unique failure. Deterministic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .validator import ValidationResult


@dataclass
class EvalRecord:
    eval_index: int                  # 0-based order in which candidates reached the validator
    candidate_id: str
    method: str
    seed: int
    query_number: Optional[int]      # 1-based global target-model query index (None if never queried)
    candidate_valid: bool
    validity_reason: str
    search_score: float
    verified_failure: bool           # the VALIDATOR's verdict, copied verbatim
    failure_reason: str
    discovery_order: Optional[int]   # 1-based rank among distinct verified failures (None otherwise)
    re_evaluation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FailureTracker:
    def __init__(self, method: str, seed: int, resolution: float):
        self.method, self.seed, self.resolution = method, int(seed), float(resolution)
        self.records: List[EvalRecord] = []
        self.verified_ids: List[str] = []             # discovery order, distinct
        self._verified_set = set()
        self._leaders: List[np.ndarray] = []
        self.n_events = 0
        self.first_failure_query: Optional[int] = None

    def record(self, r: ValidationResult, score: float, x: np.ndarray, re_evaluation: bool = False) -> EvalRecord:
        order = None
        if r.verified_failure:                           # verdict comes from the validator ONLY
            self.n_events += 1
            if r.candidate_id not in self._verified_set:
                self._verified_set.add(r.candidate_id)
                self.verified_ids.append(r.candidate_id)
                order = len(self.verified_ids)
                if self.first_failure_query is None:
                    self.first_failure_query = r.target_query_index
                x = np.asarray(x, float)
                if not self._leaders or min(float(np.linalg.norm(x - L)) for L in self._leaders) >= self.resolution:
                    self._leaders.append(x.copy())
        rec = EvalRecord(len(self.records), r.candidate_id, self.method, self.seed, r.target_query_index,
                         bool(r.candidate_valid), r.validity_reason, float(score), bool(r.verified_failure),
                         r.failure_reason, order, re_evaluation)
        self.records.append(rec)
        return rec

    @property
    def n_verified(self) -> int:
        return len(self.verified_ids)

    @property
    def n_unique(self) -> int:
        return len(self._leaders)

    @property
    def n_queried(self) -> int:
        return sum(1 for r in self.records if r.query_number is not None)

    def rows(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.records]
