"""Common search-engine interface, configuration and result schema (Phase 6).

Every method (random / bayesian / cmaes) is a `SearchEngine`: it receives the SAME `SearchSpace`,
`FailureValidator`, query budget, explicit seed and a method config, and is run against a FRESH
`QueryBudgetedModel` (`target_model.black_box(budget)`). All candidate evaluation goes through
`FeedbackManager.evaluate` -> `FailureValidator` -> `QueryBudgetedModel.query`. Engines never see
labels, the test split, target-model internals, or the verdict of the validator (they receive search
scores only).
"""
from __future__ import annotations

import math
import platform
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .failure_tracker import FailureTracker
from .feedback_manager import FeedbackManager
from .search_score import SearchScore, SearchScoreConfig
from .search_space import SearchSpace
from .utils import config_hash, save_json
from .validator import FailureValidator

# RNG streams (Phase 5 uses 1 = uniform, 2 = local perturbation)
STREAM_BO_INIT, STREAM_BO_PROPOSE, STREAM_BO_GP = 11, 12, 13
STREAM_CMA, STREAM_CMA_X0 = 21, 22

RESULT_FORMAT = 1


def stream_rng(seed: int, stream: int, *index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream), *[int(i) for i in index]]))


# ==========================================================================
# Common configuration
# ==========================================================================
@dataclass(frozen=True)
class CommonSearchConfig:
    query_budgets: Tuple[int, ...] = (100, 250, 500, 1000, 2500, 5000)
    seeds: Tuple[int, ...] = tuple(range(1, 11))
    duplicate_policy: str = "skip"                  # skip | allow
    max_candidates_per_query: float = 50.0          # shared proposal cap = ceil(factor * budget)
    store_trajectory: bool = True
    low_validity_warning_rate: float = 0.01
    unique_resolution: float = 0.05                 # Euclidean (scaled units); from validator.dedup_resolution
    objective: SearchScoreConfig = field(default_factory=SearchScoreConfig)

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "CommonSearchConfig":
        c = cfg["search"]["common"]
        seeds = c.get("seeds") or cfg["reproducibility"]["seeds"]
        out = cls(tuple(int(b) for b in c["query_budgets"]), tuple(int(s) for s in seeds), c["duplicate_policy"],
                  float(c["max_candidates_per_query"]), bool(c["store_trajectory"]),
                  float(c["low_validity_warning_rate"]), float(cfg["validator"]["dedup_resolution"]),
                  SearchScoreConfig.from_cfg(c["objective"]))
        out.validate()
        return out

    def validate(self) -> None:
        bad = []
        if not self.query_budgets or any((isinstance(b, bool) or not isinstance(b, int) or b < 1) for b in self.query_budgets):
            bad.append("query_budgets must be a non-empty list of integers >= 1")
        if not self.seeds or any((isinstance(s, bool) or not isinstance(s, int) or s < 0) for s in self.seeds):
            bad.append("seeds must be a non-empty list of integers >= 0")
        if self.duplicate_policy not in ("skip", "allow"): bad.append("duplicate_policy must be skip | allow")
        if not (math.isfinite(self.max_candidates_per_query) and self.max_candidates_per_query >= 1):
            bad.append("max_candidates_per_query must be >= 1")
        if not 0 <= self.low_validity_warning_rate <= 1: bad.append("low_validity_warning_rate must be in [0, 1]")
        if not (math.isfinite(self.unique_resolution) and self.unique_resolution > 0): bad.append("unique_resolution must be > 0")
        self.objective.validate()
        if bad:
            raise ValueError("Invalid search.common config: " + "; ".join(bad))

    def max_candidates(self, budget: int) -> int:
        return max(int(math.ceil(self.max_candidates_per_query * budget)), int(budget))


# ==========================================================================
# Result schema
# ==========================================================================
@dataclass
class SearchResult:
    method: str
    seed: int
    query_budget: int
    queries_used: int
    candidates_generated: int            # created by the algorithm
    candidates_evaluated: int            # submitted to the validator (new + duplicate)
    candidates_valid: int                # distinct evaluated candidates passing validity
    candidates_invalid: int
    duplicates_skipped: int
    verified_failures: int               # distinct candidate ids verified by the validator
    unique_verified_failures: int        # leader clusters at validator.dedup_resolution
    verified_failure_events: int
    first_failure_query: Optional[int]
    candidate_ids: List[str]             # every evaluated candidate, submission order
    verified_failure_ids: List[str]      # discovery order
    objective_stats: Dict[str, Any]
    termination_reason: str
    reproducibility: Dict[str, Any]
    paper_eligible: bool
    warnings: List[str]
    black_box: Dict[str, Any]            # final QueryBudgetedModel.snapshot()
    trajectory: Optional[List[Dict[str, Any]]] = None

    def to_dict(self, include_trajectory: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        d["format"] = RESULT_FORMAT
        if not include_trajectory:
            d["trajectory"] = None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SearchResult":
        d = dict(d)
        if d.pop("format", None) != RESULT_FORMAT:
            raise ValueError("unsupported search-result format")
        return cls(**d)

    def save(self, path, include_trajectory: bool = True):
        return save_json(self.to_dict(include_trajectory), path)


def _stats(v: Sequence[float]) -> Dict[str, Any]:
    if not len(v):
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    a = np.asarray(v, float)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}


# ==========================================================================
# Engine
# ==========================================================================
class SearchEngine(ABC):
    method = "abstract"

    def __init__(self, space: SearchSpace, validator: FailureValidator, query_budget: int, seed: int,
                 common: CommonSearchConfig, method_cfg: Any, *, paper_eligible_context: bool = False):
        if isinstance(query_budget, bool) or not isinstance(query_budget, (int, np.integer)) or query_budget < 1:
            raise ValueError(f"query_budget must be an integer >= 1, got {query_budget!r}")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(f"seed must be an integer >= 0, got {seed!r}")
        if tuple(space.feature_names) != tuple(validator.feature_names):
            raise ValueError("SearchSpace and FailureValidator disagree on the feature schema")
        common.validate()
        self.space, self.validator = space, validator
        self.query_budget, self.seed = int(query_budget), int(seed)
        self.common, self.cfg = common, method_cfg
        self.paper_eligible_context = bool(paper_eligible_context)
        self.warnings: List[str] = []
        self._used = False

    # subclasses implement the search loop and return a termination reason
    @abstractmethod
    def _search(self, session: FeedbackManager) -> str: ...

    def method_config_dict(self) -> Dict[str, Any]:
        return asdict(self.cfg) if hasattr(self.cfg, "__dataclass_fields__") else dict(self.cfg or {})

    # ------------------------------------------------------------------
    def run(self, black_box) -> SearchResult:
        if self._used:
            raise RuntimeError("a SearchEngine instance is single-use; construct a new one for each run")
        self._used = True
        if black_box.budget is None or black_box.budget != self.query_budget:
            raise ValueError(f"black box budget {black_box.budget!r} != search query_budget {self.query_budget}")
        if black_box.n_queries != 0:
            raise ValueError("search must start on a FRESH black box (n_queries == 0); budgets are never reset or shared")
        score = SearchScore(self.common.objective, self.validator.cfg.rule_min_oracle_confidence,
                            scale=self.space.ref_std)
        tracker = FailureTracker(self.method, self.seed, self.common.unique_resolution)
        session = FeedbackManager(self.validator, black_box, score, tracker, seed=self.seed,
                                  duplicate_policy=self.common.duplicate_policy,
                                  max_candidates=self.common.max_candidates(self.query_budget))
        reason = self._search(session)
        if session.stopped:
            reason = session.stop_reason
        return self._finish(session, tracker, black_box, reason)

    def _finish(self, session: FeedbackManager, tracker: FailureTracker, bb, reason: str) -> SearchResult:
        warnings = list(self.warnings)
        accounting_ok = (session.queries_used == tracker.n_queried == bb.n_queries - session.start_queries
                         and bb.n_queries <= bb.budget)
        if not accounting_ok:
            warnings.append("query accounting inconsistent")
        if session.n_submitted and session.n_valid / max(session.n_valid + session.n_invalid, 1) < self.common.low_validity_warning_rate:
            warnings.append(f"low validity rate: {session.n_valid} of {session.n_valid + session.n_invalid} "
                            f"evaluated candidates were valid (< {self.common.low_validity_warning_rate})")
        if reason == "candidate_cap_reached":
            warnings.append(f"proposal cap reached before the query budget was used ({session.queries_used}/{bb.budget} queries)")
        if not self.paper_eligible_context:
            warnings.append("paper_eligible_context not supplied/true: result is not paper-eligible")
        dropped = session.n_generated - session.n_submitted
        if dropped > 0:
            warnings.append(f"{dropped} generated candidates were never evaluated (budget/cap reached)")
        import scipy, sklearn
        repro = {
            "method": self.method, "seed": self.seed, "space_id": self.space.space_id,
            "reference_fingerprint": self.validator.ref.fingerprint,
            "validator_config_hash": config_hash(asdict(self.validator.cfg)),
            "common_config_hash": config_hash(asdict(self.common)),
            "method_config": self.method_config_dict(), "objective": session.score.describe(),
            "trajectory_sha256": _traj_hash(session.candidate_ids),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "max_candidates": session.max_candidates,
            "duplicate_policy": self.common.duplicate_policy, **self.extra_reproducibility()}
        stats = {"valid": _stats(session.scores_valid), "all": _stats(session.scores_all),
                 "best_score": max(session.scores_all) if session.scores_all else None}
        return SearchResult(
            method=self.method, seed=self.seed, query_budget=self.query_budget, queries_used=session.queries_used,
            candidates_generated=session.n_generated, candidates_evaluated=session.n_submitted,
            candidates_valid=session.n_valid, candidates_invalid=session.n_invalid,
            duplicates_skipped=session.n_duplicates, verified_failures=tracker.n_verified,
            unique_verified_failures=tracker.n_unique, verified_failure_events=tracker.n_events,
            first_failure_query=tracker.first_failure_query, candidate_ids=list(session.candidate_ids),
            verified_failure_ids=list(tracker.verified_ids), objective_stats=stats, termination_reason=reason,
            reproducibility=repro, paper_eligible=bool(accounting_ok and self.paper_eligible_context),
            warnings=warnings, black_box=bb.snapshot(),
            trajectory=tracker.rows() if self.common.store_trajectory else None)

    def extra_reproducibility(self) -> Dict[str, Any]:
        return {}


def _traj_hash(ids: Sequence[str]) -> str:
    import hashlib
    return hashlib.sha256("\x1f".join(ids).encode()).hexdigest()[:16]
