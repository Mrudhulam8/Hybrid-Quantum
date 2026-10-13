"""Experiment runner infrastructure (Phase 6; `quantum` added in Phase 7): builds the SHARED components once and runs one search
method on a FRESH query-budgeted black box, returning the common `SearchResult` schema.

This module only provides the mechanics. It is not invoked by any test or script at campaign scale;
the full CIC-IDS2017 benchmark is deliberately NOT run in Phase 6.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .bayesian_search import BayesianSearch, BayesianSearchConfig
from .cmaes_search import CMAESConfig, CMAESSearch
from .quantum_search import QuantumSearch, QuantumSearchConfig
from .random_search import RandomSearch, RandomSearchConfig
from .search_base import CommonSearchConfig, SearchEngine, SearchResult
from .search_space import SearchSpace, SearchSpaceConfig
from .validator import FailureValidator

METHODS = {"random": RandomSearch, "bayesian": BayesianSearch, "cmaes": CMAESSearch, "quantum": QuantumSearch}
_METHOD_CFG = {"random": RandomSearchConfig, "bayesian": BayesianSearchConfig, "cmaes": CMAESConfig,
               "quantum": QuantumSearchConfig}


def build_search_components(cfg: Dict[str, Any], prepared) -> Tuple[SearchSpace, FailureValidator]:
    """SearchSpace and FailureValidator from the TRAINING split only (no test data, no labels in the space)."""
    space = SearchSpace.from_prepared(prepared, SearchSpaceConfig.from_cfg(cfg))
    validator = FailureValidator.from_prepared(prepared, cfg)
    return space, validator


def make_engine(method: str, cfg: Dict[str, Any], space: SearchSpace, validator: FailureValidator,
                budget: int, seed: int, *, paper_eligible_context: bool = False,
                method_cfg: Optional[Any] = None, run_context: Optional[Dict[str, Any]] = None) -> SearchEngine:
    if method not in METHODS:
        raise ValueError(f"unknown search method {method!r}; choose from {tuple(METHODS)}")
    common = CommonSearchConfig.from_cfg(cfg)
    mcfg = method_cfg if method_cfg is not None else _METHOD_CFG[method].from_cfg(cfg)
    kw = {"paper_eligible_context": paper_eligible_context}
    if method == "random":
        kw["space_config"] = SearchSpaceConfig.from_cfg(cfg)
    if method == "quantum":
        kw["run_context"] = run_context
    return METHODS[method](space, validator, budget, seed, common, mcfg, **kw)


def run_search_method(method: str, cfg: Dict[str, Any], space: SearchSpace, validator: FailureValidator,
                      target_model, budget: int, seed: int, *, paper_eligible_context: Optional[bool] = None,
                      method_cfg: Optional[Any] = None) -> SearchResult:
    """One run = one fresh black box with exactly `budget` queries. The budget is never reset or shared."""
    if paper_eligible_context is None:
        paper_eligible_context = not getattr(target_model, "is_dev_mock", True)
    ctx = {"dataset": getattr(validator.ref, "source", ""),            # names only; no labels, no test data
           "model_under_test": {"kind": getattr(target_model, "kind", None),
                                "is_dev_mock": bool(getattr(target_model, "is_dev_mock", True))}}
    engine = make_engine(method, cfg, space, validator, budget, seed,
                         paper_eligible_context=bool(paper_eligible_context), method_cfg=method_cfg, run_context=ctx)
    bb = target_model.black_box(budget, name=f"{method}-seed{seed}-budget{budget}")
    return engine.run(bb)


def run_grid(methods: Sequence[str], budgets: Iterable[int], seeds: Iterable[int], cfg: Dict[str, Any],
             space: SearchSpace, validator: FailureValidator, target_model, out_dir: Optional[Path] = None,
             paper_eligible_context: Optional[bool] = None) -> List[SearchResult]:
    """Runs every (method, budget, seed). Provided as infrastructure; NOT executed in Phase 6 at benchmark scale."""
    out: List[SearchResult] = []
    for m in methods:
        for b in budgets:
            for s in seeds:
                r = run_search_method(m, cfg, space, validator, target_model, int(b), int(s),
                                      paper_eligible_context=paper_eligible_context)
                if out_dir is not None:
                    r.save(Path(out_dir) / f"{m}_b{b}_s{s}.json")
                out.append(r)
    return out
