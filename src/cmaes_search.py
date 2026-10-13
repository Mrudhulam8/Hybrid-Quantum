"""Baseline 3: CMA-ES (Phase 6) - lightweight, self-contained, deterministic.

The `cma` package is NOT used (not required; avoids a dependency and keeps accounting transparent).
Implementation follows Hansen's "The CMA Evolution Strategy: A Tutorial" (2016): weighted
recombination (mu = lambda/2, log weights), cumulative step-size adaptation, rank-one + rank-mu
covariance update, eigendecomposition every generation.

COORDINATES. The optimiser works in a `Representation` (default 'full': standardised features
v = (x - ref_mean)/ref_std, the same units as the Phase 4/5 perturbation scales; 'pca' optional).
sigma0 is in those units (default 0.1, matching the Phase 5 local-perturbation default; an a-priori
placeholder, not tuned on outcomes).

BOUNDS. Sampled points are repaired by `representation.decode` (clip to the SearchSpace box, then the
space's optional constraints, via SearchSpace.project); the strategy is updated with the repaired
coordinates actually evaluated (`encode(decoded)`), so it adapts to the feasible region.

SCORE. CMA-ES minimises f = -search_score. Invalid candidates receive the common graded penalty, so
the ranking is defined for every individual; they cost zero queries.

BUDGET. A population is evaluated in sample order through the shared FeedbackManager, in chunks no
larger than the remaining budget. If the budget runs out inside a population, the evaluated PREFIX is
kept, the rest is dropped (counted as generated-but-not-evaluated), no update is made, and the search
terminates cleanly ('budget_exhausted'); a population is never silently over-evaluated.

RESTARTS. If the step size collapses, the covariance becomes ill-conditioned, or fitness is flat for
`flat_generations` generations, the run restarts from a new (seeded) start point so that remaining budget
is not wasted (`restart_on_convergence`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .candidate import Candidate
from .search_base import STREAM_CMA, STREAM_CMA_X0, SearchEngine, stream_rng
from .search_representation import REPRESENTATIONS, make_representation

X0_POLICIES = ("reference_anchor", "reference_mean", "random_uniform")
STOP_MAX_GEN = "max_generations_reached"


@dataclass(frozen=True)
class CMAESConfig:
    representation: str = "full"           # full | pca
    n_components: int = 10                 # pca only
    latent_bound: float = 2.0              # pca only
    sigma0: float = 0.1
    popsize: Optional[int] = None          # null = 4 + floor(3 ln n)
    max_generations: Optional[int] = None  # null = until budget / proposal cap
    x0: str = "reference_anchor"
    restart_on_convergence: bool = True
    sigma_min: float = 1.0e-4
    max_condition: float = 1.0e12
    flat_generations: int = 3

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "CMAESConfig":
        b = cfg["search"]["cmaes"]
        c = cls(**{k: b[k] for k in cls.__dataclass_fields__ if k in b})
        c.validate()
        return c

    def validate(self) -> None:
        bad = []
        ok = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        pint = lambda v, m=1: isinstance(v, int) and not isinstance(v, bool) and v >= m
        if self.representation not in REPRESENTATIONS: bad.append(f"representation must be one of {REPRESENTATIONS}")
        if not pint(self.n_components): bad.append("n_components must be an integer >= 1")
        if not (ok(self.latent_bound) and self.latent_bound > 0): bad.append("latent_bound must be > 0")
        if not (ok(self.sigma0) and self.sigma0 > 0): bad.append("sigma0 must be > 0")
        if self.popsize is not None and not pint(self.popsize, 4): bad.append("popsize must be null or an integer >= 4")
        if self.max_generations is not None and not pint(self.max_generations): bad.append("max_generations must be null or an integer >= 1")
        if self.x0 not in X0_POLICIES: bad.append(f"x0 must be one of {X0_POLICIES}")
        if not (ok(self.sigma_min) and self.sigma_min > 0): bad.append("sigma_min must be > 0")
        if not (ok(self.max_condition) and self.max_condition > 1): bad.append("max_condition must be > 1")
        if not pint(self.flat_generations): bad.append("flat_generations must be an integer >= 1")
        if bad:
            raise ValueError("Invalid search.cmaes config: " + "; ".join(bad))


class CMAESState:
    """Plain (mu/mu_w, lambda)-CMA-ES in R^n. Minimisation. No I/O, no RNG of its own."""

    def __init__(self, mean: np.ndarray, sigma: float, popsize: Optional[int] = None):
        n = len(mean)
        self.n = n
        self.lam = int(popsize) if popsize else 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        w = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / float((self.w ** 2).sum())
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.m = np.array(mean, float)
        self.sigma = float(sigma)
        self.C = np.eye(n)
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.B, self.D = np.eye(n), np.ones(n)
        self.generation = 0

    def ask(self, Z: np.ndarray) -> np.ndarray:
        """Z: (lam, n) standard-normal draws supplied by the caller (keeps RNG policy outside)."""
        return self.m + self.sigma * (Z * self.D) @ self.B.T

    def tell(self, Y: np.ndarray, f: np.ndarray) -> None:
        """Y: (lam, n) points actually evaluated, f: (lam,) values to MINIMISE."""
        order = np.argsort(f, kind="stable")
        sel = Y[order[:self.mu]]
        m_old = self.m.copy()
        self.m = self.w @ sel
        step = (self.m - m_old) / self.sigma
        invsqrtC = (self.B / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (invsqrtC @ step)
        g = self.generation + 1
        hsig = float(np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * g)) / self.chiN < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * step
        art = (sel - m_old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (art.T * self.w) @ art)
        self.sigma *= math.exp(min(1.0, (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)))
        self.C = (self.C + self.C.T) / 2
        eig, B = np.linalg.eigh(self.C)
        eig = np.maximum(eig, 1e-300)
        self.D, self.B = np.sqrt(eig), B
        self.generation = g

    @property
    def condition(self) -> float:
        return float((self.D.max() / self.D.min()) ** 2)


class CMAESSearch(SearchEngine):
    method = "cmaes"

    def __init__(self, space, validator, query_budget, seed, common, method_cfg: Optional[CMAESConfig] = None, *,
                 paper_eligible_context: bool = False):
        method_cfg = method_cfg or CMAESConfig()
        method_cfg.validate()
        super().__init__(space, validator, query_budget, seed, common, method_cfg,
                         paper_eligible_context=paper_eligible_context)
        self.rep = make_representation(method_cfg.representation, space, validator.ref.X,   # features only
                                       method_cfg.n_components, method_cfg.latent_bound)
        self.n_restarts = 0
        self.generations = 0
        self.population_sizes: List[int] = []

    def _start_point(self, restart: int) -> np.ndarray:
        pol, rep = self.cfg.x0, self.rep
        if pol == "reference_mean":
            v = rep.encode(self.space.ref_mean[None])[0]
        elif pol == "reference_anchor":
            ref = self.validator.ref.X                                  # feature rows only
            i = int(stream_rng(self.seed, STREAM_CMA_X0, restart).integers(0, ref.shape[0]))
            v = rep.encode(self.space.project(ref[i][None]))[0]
        else:
            v = rep.lower + (rep.upper - rep.lower) * stream_rng(self.seed, STREAM_CMA_X0, restart).random(rep.dim)
        return np.clip(v, rep.lower, rep.upper)

    def _search(self, session) -> str:
        c, rep = self.cfg, self.rep
        restart, flat = 0, 0
        st = CMAESState(self._start_point(0), c.sigma0, c.popsize)
        while not session.stopped:
            if c.max_generations is not None and self.generations >= c.max_generations:
                return STOP_MAX_GEN
            Z = stream_rng(self.seed, STREAM_CMA, restart, st.generation).standard_normal((st.lam, st.n))
            V = np.clip(st.ask(Z), rep.lower, rep.upper)
            X = rep.decode(V)
            cands = [Candidate.from_array(X[i], self.space.feature_names, "cmaes",
                                          {"generator": "cmaes", "seed": self.seed, "restart": restart,
                                           "generation": st.generation, "index": i}) for i in range(st.lam)]
            session.note_generated(len(cands))
            fb = session.evaluate(cands)
            self.generations += 1
            self.population_sizes.append(len(fb))
            if len(fb) < st.lam:                                   # budget/cap hit inside the population
                self.warnings.append(f"partial population discarded: {len(fb)}/{st.lam} evaluated, no final update")
                break
            f = -np.array([x.score for x in fb])
            st.tell(rep.encode(X), f)                              # update with the REPAIRED points evaluated
            flat = flat + 1 if float(f.max() - f.min()) < 1e-12 else 0
            if c.restart_on_convergence and (st.sigma * float(st.D.max()) < c.sigma_min
                                             or st.condition > c.max_condition or flat >= c.flat_generations):
                restart += 1
                self.n_restarts += 1
                flat = 0
                st = CMAESState(self._start_point(restart), c.sigma0, c.popsize)
        return session.stop_reason or "budget_exhausted"

    def extra_reproducibility(self) -> Dict[str, Any]:
        return {"representation": self.rep.describe(), "generations": self.generations, "restarts": self.n_restarts,
                "popsize": self.population_sizes[0] if self.population_sizes else None}
