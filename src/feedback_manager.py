"""Evaluation gateway shared by EVERY search method (Phase 6).

`FeedbackManager.evaluate(candidates)` is the only place search code turns candidates into feedback:

    candidates -> duplicate policy -> FailureValidator.validate_batch (-> QueryBudgetedModel.query)
               -> SearchScore -> FailureTracker -> `Feedback(score, valid, ...)` for the optimiser

QUERY ACCOUNTING (identical for all methods)
* Only `FailureValidator` touches the target model, and only through `QueryBudgetedModel.query`.
* Candidates are validated in order, in chunks no larger than the REMAINING budget. A chunk of m
  candidates can contain at most m valid ones, so a chunk never exceeds the budget and
  `QueryBudgetExceeded` cannot fire; invalid candidates consume zero queries, so a population/batch
  that does not fit is processed as a deterministic PREFIX and the search then stops cleanly.
* The budget is never reset or replaced; the black-box counter is checked after every chunk
  (monotone, and increase == number of queried records), otherwise `BudgetAccountingError`.
* Duplicates (same `candidate_id`): policy 'skip' returns the cached feedback (zero queries);
  'allow' re-evaluates (one more query if valid).
* Shared proposal cap: at most `max_candidates` candidates may be submitted per run (the Phase 5
  "open design question": invalid proposals are free, so something must bound them, identically for
  every method).
Optimisers receive `Feedback` objects that carry a search score, NOT the verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .candidate import Candidate
from .failure_tracker import FailureTracker
from .search_score import SearchScore
from .target_model import QueryBudgetExceeded
from .validator import FailureValidator

STOP_BUDGET = "budget_exhausted"
STOP_CAP = "candidate_cap_reached"


class BudgetAccountingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Feedback:
    candidate: Candidate
    score: float                    # search score (ranking value only)
    valid: bool
    queried: bool
    duplicate: bool                 # served from cache (zero queries)
    query_number: Optional[int]


class FeedbackManager:
    def __init__(self, validator: FailureValidator, black_box, score: SearchScore, tracker: FailureTracker,
                 *, seed: int, duplicate_policy: str, max_candidates: int):
        if duplicate_policy not in ("skip", "allow"):
            raise ValueError("duplicate_policy must be 'skip' or 'allow'")
        if black_box.budget is None:
            raise ValueError("search requires a finite query budget (got an unbudgeted black box)")
        self.validator, self.bb, self.score, self.tracker = validator, black_box, score, tracker
        self.seed, self.policy, self.max_candidates = int(seed), duplicate_policy, int(max_candidates)
        self.start_queries = black_box.n_queries
        self._last = black_box.n_queries
        self._cache: Dict[str, Feedback] = {}
        self.n_submitted = 0            # proposals handed to evaluate() (new + duplicate)
        self.n_duplicates = 0
        self.n_generated = 0
        self.stop_reason: Optional[str] = None
        self.n_valid = 0
        self.n_invalid = 0
        self.scores_valid: List[float] = []
        self.scores_all: List[float] = []
        self.candidate_ids: List[str] = []       # every submitted candidate, submission order
        self.valid_flags: List[bool] = []

    # ---- state -----------------------------------------------------------------------------
    @property
    def queries_used(self) -> int:
        return self.bb.n_queries - self.start_queries

    @property
    def proposals_left(self) -> int:
        return max(0, self.max_candidates - self.n_submitted)

    @property
    def stopped(self) -> bool:
        if self.bb.exhausted:
            self.stop_reason = self.stop_reason or STOP_BUDGET
        elif self.proposals_left <= 0:
            self.stop_reason = self.stop_reason or STOP_CAP
        return self.stop_reason is not None

    def has_seen(self, candidate_id: str) -> bool:
        return candidate_id in self._cache

    def note_generated(self, n: int) -> None:
        self.n_generated += int(n)

    # ---- evaluation --------------------------------------------------------------------------
    def evaluate(self, candidates: Sequence[Candidate]) -> List[Feedback]:
        """Evaluate in order. Returns feedback for the evaluated PREFIX (shorter than the input only
        when the budget or the proposal cap stopped the search)."""
        out: List[Feedback] = []
        i, n = 0, len(candidates)
        while i < n:
            if self.stopped:
                break
            take = min(n - i, int(self.bb.remaining), self.proposals_left)
            chunk = list(candidates[i:i + take])
            out.extend(self._evaluate_chunk(chunk))
            i += take
        return out

    def _evaluate_chunk(self, chunk: List[Candidate]) -> List[Feedback]:
        fresh: List[Candidate] = []
        plan = []
        local = set()
        for c in chunk:
            cid = c.candidate_id
            if self.policy == "skip" and (cid in self._cache or cid in local):
                plan.append(("dup", cid))
            else:
                plan.append(("eval", len(fresh)))
                fresh.append(c)
                local.add(cid)
        results = []
        if fresh:
            before = self.bb.n_queries
            try:
                results = self.validator.validate_batch(fresh, self.bb, self.seed)
            except QueryBudgetExceeded:                    # defensive: cannot happen (chunk <= remaining)
                self.stop_reason = STOP_BUDGET
                raise
            consumed = self.bb.n_queries - before
            if before < self._last or consumed != sum(1 for r in results if r.queried) or self.bb.n_queries > self.bb.budget:
                raise BudgetAccountingError(
                    f"query accounting violated: counter {self._last}->{self.bb.n_queries}, "
                    f"queried rows {sum(1 for r in results if r.queried)}, budget {self.bb.budget}")
            self._last = self.bb.n_queries
        fb_new: List[Feedback] = []
        for c, r in zip(fresh, results):
            s = self.score.score(r, c.x)
            re_eval = c.candidate_id in self._cache
            self.tracker.record(r, s, c.x, re_evaluation=re_eval)
            fb = Feedback(c, s, bool(r.candidate_valid), bool(r.queried), False, r.target_query_index)
            self._cache[c.candidate_id] = fb
            fb_new.append(fb)
            self.n_valid += int(r.candidate_valid)
            self.n_invalid += int(not r.candidate_valid)
            self.scores_all.append(s)
            if r.candidate_valid:
                self.scores_valid.append(s)
        out: List[Feedback] = []
        for c, (kind, ref) in zip(chunk, plan):
            self.n_submitted += 1
            self.candidate_ids.append(c.candidate_id)
            if kind == "eval":
                fb = fb_new[ref]
                out.append(fb)
                self.valid_flags.append(fb.valid)
            else:
                cached = self._cache[ref]
                self.n_duplicates += 1
                out.append(Feedback(c, cached.score, cached.valid, cached.queried, True, cached.query_number))
                self.valid_flags.append(cached.valid)
        return out
