"""Reference oracle: decides (a) whether a candidate is a REALISTIC input and (b) what label the
reference data supports for it. It is built from the TRAINING split only.

INDEPENDENCE GUARANTEES (enforced by structure and by tests/test_phase4.py)
* The oracle never receives, imports or calls the target model.
* It is constructed from a `ReferenceData` view that contains X/y/category of the TRAINING split
  only. `ReferenceData.from_prepared` reads only `X_train, y_train, cat_train, feature_names,
  class_names, provenance` - never any `*_test` field.
* The held-out test split is therefore unreachable from here.

STATISTICAL DESIGN
* Validity  = finite & right schema (checked by Candidate) AND inside the reference feature range
  AND global Mahalanobis distance <= tau_M AND 1-NN distance <= tau_NN.
  Mahalanobis alone is weak here (flow features are heavy-tailed, the reference is a multi-modal
  mixture, so a single Gaussian is a crude model), hence the extra NN-distance gate.
* Thresholds are calibrated on REFERENCE statistics only (empirical quantiles of the reference's
  own distances), never on discovery results. They must be frozen before comparing search methods.
* Mahalanobis stability: covariance = Ledoit-Wolf (or fixed) shrinkage + optional ridge, then an
  eigen-decomposition with an eigenvalue floor. Collinear/near-constant directions therefore
  cannot blow distances up. Distance = ||(x-mu) V diag(1/sqrt(w_floored))||.
* Label oracle = k-NN vote in the scaled Euclidean space. Purity = fraction of the k neighbours
  carrying the majority label. If purity < min_purity the oracle ABSTAINS (label = None).
  Ties: fewer summed distance wins, then lowest label index. Neighbour order: (distance, index).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .candidate import OK
from .utils import get_logger

log = get_logger("oracle")

OUTSIDE_RANGE = "outside_feature_range"
MAHA_EXCEEDED = "mahalanobis_exceeded"
NN_EXCEEDED = "nn_distance_exceeded"
NO_LABEL = -1


# ==========================================================================
# Reference data (TRAIN ONLY)
# ==========================================================================
@dataclass(frozen=True)
class ReferenceData:
    X: np.ndarray                      # (n, d) scaled features, TRAINING split
    y: np.ndarray                      # (n,) task labels (same label space as the target model)
    categories: Optional[np.ndarray]   # (n,) attack-type strings (training split) or None
    feature_names: Tuple[str, ...]
    class_names: Tuple[str, ...]
    source: str = ""
    fingerprint: str = field(init=False, default="")

    def __post_init__(self):
        import hashlib
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.X, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(self.y, dtype="<i8").tobytes())
        h.update("\x1f".join(self.feature_names).encode())
        object.__setattr__(self, "fingerprint", h.hexdigest()[:16])

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @classmethod
    def from_prepared(cls, prepared, max_rows: Optional[int] = None, seed: int = 0) -> "ReferenceData":
        """TRAINING-split view only. Touches no `*_test` attribute (asserted in tests)."""
        X = np.asarray(prepared.X_train, dtype=np.float64)
        y = np.asarray(prepared.y_train)
        cats = np.asarray(prepared.cat_train)
        if max_rows is not None and len(X) > max_rows:       # deterministic uniform cap
            idx = np.sort(np.random.default_rng(seed).choice(len(X), size=max_rows, replace=False))
            X, y, cats = X[idx], y[idx], cats[idx]
        return cls(X, y, cats, tuple(prepared.feature_names), tuple(prepared.class_names),
                   str(prepared.provenance.get("source", "")))


# ==========================================================================
# Mahalanobis
# ==========================================================================
class MahalanobisValidity:
    def __init__(self, shrinkage: str = "ledoit_wolf", shrinkage_alpha: float = 0.1,
                 ridge: float = 0.0, eigenvalue_floor_rel: float = 1e-8):
        if shrinkage not in ("ledoit_wolf", "fixed", "none"):
            raise ValueError(f"unknown shrinkage {shrinkage!r}")
        self.shrinkage, self.alpha, self.ridge, self.floor_rel = shrinkage, shrinkage_alpha, ridge, eigenvalue_floor_rel

    def fit(self, X: np.ndarray) -> "MahalanobisValidity":
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        if self.shrinkage == "ledoit_wolf":
            from sklearn.covariance import ledoit_wolf
            S, self.shrinkage_coef_ = ledoit_wolf(X)
        else:
            Xc = X - self.mean_
            S = Xc.T @ Xc / max(n - 1, 1)
            if self.shrinkage == "fixed":
                S = (1 - self.alpha) * S + self.alpha * (np.trace(S) / d) * np.eye(d)
                self.shrinkage_coef_ = float(self.alpha)
            else:
                self.shrinkage_coef_ = 0.0
        if self.ridge:
            S = S + self.ridge * np.eye(d)
        self.cov_ = 0.5 * (S + S.T)                                    # exactly symmetric
        w, V = np.linalg.eigh(self.cov_)
        floor = max(self.floor_rel * float(w.max()), np.finfo(np.float64).tiny)
        self.n_floored_ = int((w < floor).sum())
        wf = np.maximum(w, floor)
        self.eigvals_, self.eigvecs_ = wf, V
        self.whiten_ = V / np.sqrt(wf)
        self.precision_ = (V / wf) @ V.T
        p = wf / wf.sum()
        self.diagnostics_ = {"n": int(n), "d": int(d), "shrinkage": self.shrinkage,
                             "shrinkage_coef": float(self.shrinkage_coef_),
                             "condition_number": float(wf.max() / wf.min()),
                             "n_eigenvalues_floored": self.n_floored_,
                             "effective_rank": float(np.exp(-(p * np.log(p)).sum()))}
        return self

    def distance(self, X: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(np.asarray(X, dtype=np.float64)) - self.mean_) @ self.whiten_
        return np.sqrt((Z * Z).sum(axis=1))


# ==========================================================================
# k-NN reference
# ==========================================================================
class KNNReference:
    """Brute-force, chunked, DETERMINISTIC k-NN (no tree/ANN nondeterminism)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, categories: Optional[np.ndarray],
                 n_classes: int, chunk_size: int = 64):
        self.X, self.y, self.cats, self.C, self.chunk = X, np.asarray(y, dtype=np.int64), categories, n_classes, chunk_size
        self._sq = (X * X).sum(axis=1)

    def neighbors(self, Q: np.ndarray, k: int, self_idx: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Indices and exact Euclidean distances (m,k), ordered by (distance, index).
        `self_idx[i]` excludes reference row self_idx[i] for query i (leave-one-out)."""
        Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
        k = min(k, self.X.shape[0] - (1 if self_idx is not None else 0))
        m = Q.shape[0]
        out_i = np.empty((m, k), dtype=np.int64)
        out_d = np.empty((m, k))
        for s in range(0, m, self.chunk):
            q = Q[s:s + self.chunk]
            d2 = (q * q).sum(1)[:, None] - 2.0 * q @ self.X.T + self._sq[None, :]
            np.maximum(d2, 0.0, out=d2)
            if self_idx is not None:
                d2[np.arange(len(q)), self_idx[s:s + self.chunk]] = np.inf
            kth = np.partition(d2, k - 1, axis=1)[:, k - 1]
            for r in range(len(q)):
                cand = np.flatnonzero(d2[r] <= kth[r])                 # includes ALL ties at the k-th distance
                exact = np.sqrt(((self.X[cand] - q[r]) ** 2).sum(1))
                order = np.lexsort((cand, exact))[:k]
                out_i[s + r], out_d[s + r] = cand[order], exact[order]
        return out_i, out_d

    def _vote(self, labels: np.ndarray, dists: np.ndarray, n_labels: int) -> Tuple[int, float]:
        counts = np.bincount(labels, minlength=n_labels)
        top = np.flatnonzero(counts == counts.max())
        if len(top) > 1:
            sums = [dists[labels == t].sum() for t in top]
            top = top[[int(np.argmin(sums))]]                          # argmin -> lowest label on exact ties
        return int(top[0]), float(counts[top[0]] / len(labels))

    def vote(self, idx: np.ndarray, dist: np.ndarray, min_purity: float):
        m = idx.shape[0]
        label = np.full(m, NO_LABEL, dtype=np.int64)
        purity = np.zeros(m)
        category: List[Optional[str]] = [None] * m
        for i in range(m):
            lab, pur = self._vote(self.y[idx[i]], dist[i], self.C)
            purity[i] = pur
            if pur >= min_purity:
                label[i] = lab
            if self.cats is not None:
                cs = self.cats[idx[i]]
                u, inv = np.unique(cs, return_inverse=True)
                cnt = np.bincount(inv)
                top = np.flatnonzero(cnt == cnt.max())
                if len(top) > 1:
                    top = top[[int(np.argmin([dist[i][inv == t].sum() for t in top]))]]
                category[i] = str(u[top[0]])
        return label, purity, category


# ==========================================================================
# Oracle
# ==========================================================================
@dataclass
class OracleBatch:
    valid: np.ndarray                  # (m,) bool
    reason: List[str]
    mahalanobis: np.ndarray
    nn_distance: np.ndarray
    purity: np.ndarray                 # k-NN purity of the majority label
    label: np.ndarray                  # oracle label, NO_LABEL (-1) if the oracle abstains
    category: List[Optional[str]]      # majority attack type among neighbours (reporting only)


class ReferenceOracle:
    def __init__(self, reference: ReferenceData, vcfg, seed: Optional[int] = None):
        """`vcfg` is a validator.ValidatorConfig. Fits everything on `reference` (TRAIN only)."""
        self.ref, self.cfg = reference, vcfg
        self.seed = vcfg.seed if seed is None else seed
        X = reference.X
        self.lo, self.hi = X.min(0), X.max(0)
        span = self.hi - self.lo
        self.lo_m, self.hi_m = self.lo - vcfg.range_margin * span, self.hi + vcfg.range_margin * span

        self.maha = MahalanobisValidity(vcfg.maha_shrinkage, vcfg.maha_shrinkage_alpha, vcfg.maha_ridge,
                                        vcfg.maha_eig_floor_rel).fit(X)
        ref_m = self.maha.distance(X)
        self.knn = KNNReference(X, reference.y, reference.categories, len(reference.class_names), vcfg.knn_chunk_size)

        if vcfg.maha_mode == "empirical_quantile":
            self.tau_maha = float(np.quantile(ref_m, vcfg.maha_quantile))
        elif vcfg.maha_mode == "chi2_quantile":
            from scipy.stats import chi2
            self.tau_maha = float(np.sqrt(chi2.ppf(vcfg.maha_quantile, df=self.maha.diagnostics_["effective_rank"])))
        else:
            self.tau_maha = float(vcfg.maha_absolute)

        n_cal = min(vcfg.calibration_sample_size, reference.n)
        self.cal_idx = np.sort(np.random.default_rng(self.seed).choice(reference.n, size=n_cal, replace=False))
        _, loo = self.knn.neighbors(X[self.cal_idx], 1, self_idx=self.cal_idx)
        self.loo_nn_dist = loo[:, 0]
        if not vcfg.nn_gate_enabled:
            self.tau_nn = float("inf")
        elif vcfg.nn_gate_mode == "empirical_quantile":
            self.tau_nn = float(np.quantile(self.loo_nn_dist, vcfg.nn_gate_quantile))
        else:
            self.tau_nn = float(vcfg.nn_gate_absolute)

        self.diagnostics = {
            "reference_n": reference.n, "reference_d": reference.d, "reference_fingerprint": reference.fingerprint,
            "reference_source": reference.source, "mahalanobis": self.maha.diagnostics_,
            "tau_mahalanobis": self.tau_maha, "tau_nn": self.tau_nn,
            "reference_mahalanobis_quantiles": {str(q): float(np.quantile(ref_m, q)) for q in (.5, .9, .99, .995, .999, 1.0)},
            "loo_nn_distance_quantiles": {str(q): float(np.quantile(self.loo_nn_dist, q)) for q in (.5, .9, .99, .995, .999, 1.0)},
            "calibration_sample_size": int(n_cal), "seed": int(self.seed)}

    # ------------------------------------------------------------------
    def assess(self, X: np.ndarray) -> OracleBatch:
        """X: (m, d) FINITE candidates already schema-checked. Uses no target model."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.ref.d or not np.isfinite(X).all():
            raise ValueError("assess() requires finite (m, d) input; use FailureValidator for raw candidates")
        tol = self.cfg.numerical_tol
        in_range = ((X >= self.lo_m - tol) & (X <= self.hi_m + tol)).all(axis=1) if self.cfg.range_enabled \
            else np.ones(len(X), bool)
        M = self.maha.distance(X)
        idx, dist = self.knn.neighbors(X, self.cfg.knn_k)
        label, purity, cat = self.knn.vote(idx, dist, self.cfg.knn_min_purity)
        nn = dist[:, 0]
        ok_m, ok_nn = M <= self.tau_maha + tol, nn <= self.tau_nn + tol
        valid = in_range & ok_m & ok_nn
        reason = [OK if valid[i] else (OUTSIDE_RANGE if not in_range[i] else MAHA_EXCEEDED if not ok_m[i] else NN_EXCEEDED)
                  for i in range(len(X))]
        return OracleBatch(valid, reason, M, nn, purity, label, cat)

    # ------------------------------------------------------------------
    # Reference-only diagnostics (no target model, no test data, NOT research results)
    # ------------------------------------------------------------------
    def self_consistency(self, sample_size: int = 5000, seed: int = 0) -> Dict[str, Any]:
        """Leave-one-out: how often does the k-NN oracle reproduce a reference point's own label?"""
        n = min(sample_size, self.ref.n)
        idx = np.sort(np.random.default_rng(seed).choice(self.ref.n, size=n, replace=False))
        nb, dist = self.knn.neighbors(self.ref.X[idx], self.cfg.knn_k, self_idx=idx)
        label, purity, _ = self.knn.vote(nb, dist, self.cfg.knn_min_purity)
        own = self.ref.y[idx]
        decided = label != NO_LABEL
        out = {"n": int(n), "abstain_rate": float(1 - decided.mean()),
               "agreement_when_decided": float((label[decided] == own[decided]).mean()) if decided.any() else None,
               "agreement_overall_abstain_counts_as_disagree": float((label == own).mean()),
               "mean_purity": float(purity.mean())}
        if self.ref.categories is not None:
            cats = self.ref.categories[idx]
            out["per_category"] = {str(c): {"n": int((cats == c).sum()),
                                            "agreement": float((label[cats == c] == own[cats == c]).mean()),
                                            "abstain_rate": float((label[cats == c] == NO_LABEL).mean())}
                                   for c in sorted(set(cats))}
        return out

    def perturbation_validity_profile(self, sigmas: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1.0),
                                      n: int = 2000, seed: int = 0) -> Dict[str, Any]:
        """Validity/abstention rate of reference points + isotropic Gaussian noise. Helps choose
        thresholds from REFERENCE data alone, before any search method is run."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(self.ref.n, size=min(n, self.ref.n), replace=False)
        prof = {}
        for s in sigmas:
            Q = self.ref.X[idx] + s * rng.standard_normal((len(idx), self.ref.d))
            b = self.assess(Q)
            reasons = {r: int(sum(1 for x in b.reason if x == r)) for r in set(b.reason)}
            prof[str(s)] = {"valid_rate": float(b.valid.mean()), "oracle_abstain_rate": float((b.label == NO_LABEL).mean()),
                            "reasons": reasons}
        return prof
