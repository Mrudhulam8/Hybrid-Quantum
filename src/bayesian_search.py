"""Baseline 2: Bayesian optimisation with a Gaussian-process surrogate (Phase 6).

Loop (strictly sequential unless `batch_size` > 1):
  1. initial design: `n_initial` points drawn in the optimiser's coordinate box with the run seed
     (they COUNT against the budget like any other proposal; there are no free warm-up queries);
  2. each point is evaluated through the shared FeedbackManager -> FailureValidator -> QueryBudgetedModel
     and yields a SEARCH SCORE (valid: uncertainty/failure-likelihood mix; invalid: graded penalty);
  3. a scikit-learn `GaussianProcessRegressor` (ARD Matern-5/2 or RBF + white noise) is fitted ONLY on
     (coordinates, score) pairs obtained from counted evaluations of this run;
  4. proposals = uniform points in the box + Gaussian jitter around the best observed points; the next
     candidate(s) maximise Expected Improvement (or UCB) over the proposals; already-seen candidate ids
     are excluded;
  5. repeat until the budget (or the shared proposal cap) is exhausted.

DIMENSIONALITY. A GP on ~77 raw dimensions is unreliable (length-scale fitting, O(n^3) cost,
concentration of distances). Default representation = 'pca': top-k principal axes of the TRAINING
reference features (deterministic, label-free; see search_representation.py), k = 10. 'full' is
available but emits a warning for d > 20. The surrogate sees at most `max_train_points` observations
(top scorers + most recent), selected deterministically.

The surrogate is a ranking tool. Whether a candidate is a failure is decided only by the validator.
"""
from __future__ import annotations

import math
import warnings as pywarnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import norm

from .candidate import Candidate
from .search_base import STREAM_BO_GP, STREAM_BO_INIT, STREAM_BO_PROPOSE, SearchEngine, stream_rng
from .search_representation import REPRESENTATIONS, make_representation


@dataclass(frozen=True)
class BayesianSearchConfig:
    representation: str = "pca"            # pca | full
    n_components: int = 10
    latent_bound: float = 2.0              # PCA box half-width in units of each axis' std (a-priori placeholder)
    init_fraction: float = 0.1             # of the query budget (counted, never free)
    min_initial: int = 5
    n_initial: Optional[int] = None        # overrides init_fraction when set
    kernel: str = "matern52"               # matern52 | rbf
    ard: bool = True
    acquisition: str = "ei"                # ei | ucb
    xi: float = 0.01                       # EI exploration offset
    ucb_beta: float = 2.0
    n_random_proposals: int = 512
    n_local_proposals: int = 256
    local_top_k: int = 5
    local_sigma: float = 0.05              # in normalised [0,1] coordinates
    batch_size: int = 1
    min_separation: float = 0.01           # min normalised distance between points chosen in one batch
    max_train_points: int = 500
    n_restarts_optimizer: int = 0
    refit_hyperparams_every: int = 10      # posterior-only refit in between (fixed kernel)
    noise_floor: float = 1.0e-6
    include_invalid_observations: bool = True

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "BayesianSearchConfig":
        b = cfg["search"]["bayesian"]
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
        if not (ok(self.init_fraction) and 0 < self.init_fraction <= 1): bad.append("init_fraction must be in (0, 1]")
        if not pint(self.min_initial): bad.append("min_initial must be an integer >= 1")
        if self.n_initial is not None and not pint(self.n_initial): bad.append("n_initial must be null or an integer >= 1")
        if self.kernel not in ("matern52", "rbf"): bad.append("kernel must be matern52 | rbf")
        if self.acquisition not in ("ei", "ucb"): bad.append("acquisition must be ei | ucb")
        if not (ok(self.xi) and self.xi >= 0): bad.append("xi must be >= 0")
        if not (ok(self.ucb_beta) and self.ucb_beta >= 0): bad.append("ucb_beta must be >= 0")
        if not pint(self.n_random_proposals, 1): bad.append("n_random_proposals must be an integer >= 1")
        if not pint(self.n_local_proposals, 0): bad.append("n_local_proposals must be an integer >= 0")
        if not pint(self.local_top_k): bad.append("local_top_k must be an integer >= 1")
        if not (ok(self.local_sigma) and self.local_sigma > 0): bad.append("local_sigma must be > 0")
        if not pint(self.batch_size): bad.append("batch_size must be an integer >= 1")
        if not (ok(self.min_separation) and self.min_separation >= 0): bad.append("min_separation must be >= 0")
        if not pint(self.max_train_points, 2): bad.append("max_train_points must be an integer >= 2")
        if not pint(self.n_restarts_optimizer, 0): bad.append("n_restarts_optimizer must be an integer >= 0")
        if not pint(self.refit_hyperparams_every): bad.append("refit_hyperparams_every must be an integer >= 1")
        if not (ok(self.noise_floor) and self.noise_floor > 0): bad.append("noise_floor must be > 0")
        if bad:
            raise ValueError("Invalid search.bayesian config: " + "; ".join(bad))


def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    """EI for MAXIMISATION. Zero where the predictive std is ~0."""
    mu, sd = np.asarray(mu, float), np.asarray(sd, float)
    imp = mu - best - xi
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 1e-12, imp / sd, 0.0)
    ei = imp * norm.cdf(z) + sd * norm.pdf(z)
    return np.where(sd > 1e-12, np.maximum(ei, 0.0), 0.0)


def upper_confidence_bound(mu: np.ndarray, sd: np.ndarray, beta: float) -> np.ndarray:
    return np.asarray(mu, float) + beta * np.asarray(sd, float)


class BayesianSearch(SearchEngine):
    method = "bayesian"

    def __init__(self, space, validator, query_budget, seed, common, method_cfg: Optional[BayesianSearchConfig] = None, *,
                 paper_eligible_context: bool = False):
        method_cfg = method_cfg or BayesianSearchConfig()
        method_cfg.validate()
        super().__init__(space, validator, query_budget, seed, common, method_cfg,
                         paper_eligible_context=paper_eligible_context)
        # reference FEATURES only (ReferenceData.X); labels are never read
        self.rep = make_representation(method_cfg.representation, space, validator.ref.X,
                                       method_cfg.n_components, method_cfg.latent_bound)
        if self.rep.dim > 20:
            self.warnings.append(f"GP fitted on {self.rep.dim} dimensions (> 20): surrogate is unreliable; use representation='pca'")
        self._span = np.where(self.rep.upper > self.rep.lower, self.rep.upper - self.rep.lower, 1.0)
        # observations (all from counted evaluations of THIS run)
        self._T: List[np.ndarray] = []
        self._y: List[float] = []
        self._gp = None
        self._gp_fits = 0
        self._gp_warn = 0
        self._gp_failures = 0
        self.last_acquisition: Optional[Dict[str, Any]] = None       # for tests/diagnostics

    # ---- coordinates -------------------------------------------------------------------------
    def _to_latent(self, T):
        return self.rep.lower + np.atleast_2d(T) * self._span

    def _candidates(self, T: np.ndarray, origin: str, meta: List[Dict[str, Any]]) -> List[Candidate]:
        X = self.rep.decode(self._to_latent(T))
        return [Candidate.from_array(X[i], self.space.feature_names, origin, meta[i]) for i in range(len(X))]

    def _n_initial(self) -> int:
        c = self.cfg
        n = c.n_initial if c.n_initial is not None else max(c.min_initial, int(math.ceil(c.init_fraction * self.query_budget)))
        return int(min(n, self.query_budget))

    def initial_design(self, n: int) -> np.ndarray:
        """n points in the normalised box [0,1]^k; point i depends only on (seed, i)."""
        k = self.rep.dim
        return np.stack([stream_rng(self.seed, STREAM_BO_INIT, i).random(k) for i in range(n)])

    # ---- surrogate ---------------------------------------------------------------------------
    def _training_set(self):
        T, y = np.stack(self._T), np.asarray(self._y)
        cap = self.cfg.max_train_points
        idx = np.arange(len(y))
        if not self.cfg.include_invalid_observations:
            idx = idx[y >= 0]
        if len(idx) > cap:
            n_top = cap // 4
            top = idx[np.argsort(-y[idx], kind="stable")[:n_top]]
            rest = np.setdiff1d(idx, top)[-(cap - n_top):]
            idx = np.sort(np.concatenate([top, rest]))
        return T[idx], y[idx]

    def _fit(self, T: np.ndarray, y: np.ndarray, reoptimise: bool):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
        k = T.shape[1]
        ls = np.full(k, 0.5) if self.cfg.ard else 0.5
        base = Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e1), nu=2.5) if self.cfg.kernel == "matern52" \
            else RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e1))
        if reoptimise or self._gp is None:
            kernel = ConstantKernel(1.0, (1e-2, 1e2)) * base + WhiteKernel(1e-2, (1e-6, 1e0))
            optimizer = "fmin_l_bfgs_b"
        else:
            kernel, optimizer = self._gp.kernel_, None
        rs = int(np.random.SeedSequence([self.seed, STREAM_BO_GP, self._gp_fits]).generate_state(1)[0] % (2 ** 31 - 1))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=self.cfg.noise_floor, normalize_y=True, optimizer=optimizer,
                                      n_restarts_optimizer=self.cfg.n_restarts_optimizer if optimizer else 0,
                                      random_state=rs)
        with pywarnings.catch_warnings(record=True) as caught:
            pywarnings.simplefilter("always")
            gp.fit(T, y)
        self._gp_warn += len(caught)
        self._gp_fits += 1
        return gp

    # ---- proposals / acquisition -----------------------------------------------------------------
    def _proposals(self, iteration: int) -> np.ndarray:
        c, k = self.cfg, self.rep.dim
        rng = stream_rng(self.seed, STREAM_BO_PROPOSE, iteration)
        rand = rng.random((c.n_random_proposals, k))
        parts = [rand]
        if c.n_local_proposals and self._y:
            y = np.asarray(self._y)
            top = np.argsort(-y, kind="stable")[:c.local_top_k]
            T = np.stack(self._T)
            centres = T[top][rng.integers(0, len(top), c.n_local_proposals)]
            parts.append(np.clip(centres + c.local_sigma * rng.standard_normal((c.n_local_proposals, k)), 0.0, 1.0))
        return np.vstack(parts)

    def acquire(self, iteration: int, session) -> List[Candidate]:
        """Fit the surrogate on counted observations and return up to `batch_size` unseen candidates."""
        c = self.cfg
        P = self._proposals(iteration)
        T, y = self._training_set()
        acq = None
        try:
            self._gp = self._fit(T, y, reoptimise=(self._gp is None or iteration % c.refit_hyperparams_every == 0))
            mu, sd = self._gp.predict(P, return_std=True)
            acq = expected_improvement(mu, sd, float(y.max()), c.xi) if c.acquisition == "ei" \
                else upper_confidence_bound(mu, sd, c.ucb_beta)
            if not np.isfinite(acq).all():
                raise FloatingPointError("non-finite acquisition")
        except Exception as e:                       # numerical failure: deterministic random fallback
            self._gp_failures += 1
            self._gp = None
            if self._gp_failures == 1:
                self.warnings.append(f"GP fit/predict failed ({type(e).__name__}); random fallback used for affected iterations")
            acq = np.zeros(len(P))
        meta = [{"generator": "bayesian", "seed": self.seed, "iteration": iteration, "index": i,
                 "acquisition": c.acquisition} for i in range(len(P))]
        # rank (stable => deterministic); drop proposals whose decoded candidate was already evaluated
        order = np.argsort(-acq, kind="stable")
        cands = self._candidates(P[order], "bayesian", [meta[i] for i in order])
        chosen: List[int] = []
        for j, cand in enumerate(cands):
            if session.has_seen(cand.candidate_id) and self.common.duplicate_policy == "skip":
                continue
            if any(np.linalg.norm(P[order][j] - P[order][q]) < c.min_separation for q in chosen):
                continue
            chosen.append(j)
            if len(chosen) >= c.batch_size:
                break
        self.last_acquisition = {"iteration": iteration, "n_train": int(len(y)), "max_acq": float(acq.max()),
                                 "chosen_acq": [float(acq[order][j]) for j in chosen]}
        self._pending_T = [P[order][j] for j in chosen]
        return [cands[j] for j in chosen]

    # ---- main loop ------------------------------------------------------------------------------
    def _observe(self, T_list, feedback) -> None:
        for t, fb in zip(T_list, feedback):
            if fb.duplicate:                          # cached result already observed once
                continue
            self._T.append(np.asarray(t, float))
            self._y.append(float(fb.score))

    def _search(self, session) -> str:
        n0 = min(self._n_initial(), session.proposals_left)
        T0 = self.initial_design(n0)
        meta = [{"generator": "bayesian_init", "seed": self.seed, "iteration": 0, "index": i} for i in range(n0)]
        init = self._candidates(T0, "bayesian_init", meta)
        session.note_generated(len(init))
        fb = session.evaluate(init)
        self._observe(T0[:len(fb)], fb)
        iteration = 0
        while not session.stopped:
            iteration += 1
            cands = self.acquire(iteration, session)
            if not cands:                             # every proposal already seen: fresh random point, new RNG index
                k = self.rep.dim
                t = stream_rng(self.seed, STREAM_BO_INIT, 10 ** 6 + iteration).random(k)
                cands, self._pending_T = self._candidates(t[None], "bayesian_fallback",
                                                          [{"generator": "bayesian_fallback", "seed": self.seed,
                                                            "iteration": iteration, "index": 0}]), [t]
            session.note_generated(len(cands))
            fb = session.evaluate(cands)
            self._observe(self._pending_T[:len(fb)], fb)
        if self._gp_warn:
            self.warnings.append(f"scikit-learn emitted {self._gp_warn} warnings during GP fitting (e.g. ConvergenceWarning)")
        return session.stop_reason

    def extra_reproducibility(self) -> Dict[str, Any]:
        return {"representation": self.rep.describe(), "gp_fits": self._gp_fits, "gp_failures": self._gp_failures,
                "n_observations": len(self._y)}
