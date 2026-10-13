"""Search space: bounds, reference statistics, projection and reproducible sampling.

This module is QUERY-FREE and LABEL-FREE. It is built from a feature matrix (the TRAINING split's
X) and feature names - nothing else. It has no access to labels, categories, the target model, its
predictions, or any failure outcome. Deciding whether a point is a failure is the Phase 4
validator's job, never this module's.

CONSTRAINTS (all optional, all derived from the reference matrix only)
* bounds      : per-feature [lower, upper] (reference min/max, reference quantiles, or absolute).
* radius      : RMS z-score ball around the reference mean.
* mahalanobis : Mahalanobis ellipsoid around the reference mean (reuses Phase 4's
                `MahalanobisValidity` for the covariance fit; labels are not involved).
`project()` = clip to the box, then shrink radially toward the reference mean until each enabled
constraint holds, then clip again. Radial shrinking toward a point inside the box keeps the result
in the box, and shrinking only reduces norms around that same centre, so the constraints are
mutually consistent and `project` is idempotent. If a constraint is enabled and the reference mean
lies outside a (quantile) bound, that bound is widened to include the mean (recorded in provenance).

REPRODUCIBILITY
Candidate `i` for seed `s` and stream `k` is drawn from its OWN generator
`default_rng(SeedSequence([s, k, i]))`, so a candidate never depends on batch size or on other
candidates: generate(10, seed=s) == generate(4, seed=s) + generate(6, seed=s, start_index=4).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .oracle import MahalanobisValidity          # Phase 4: covariance statistics only
from .utils import resolve_path, save_json

STREAM_UNIFORM = 1
STREAM_LOCAL = 2
EPS_SHRINK = 1e-9                                 # shrink to R*(1-EPS): makes project() exactly idempotent
_FORMAT = 1


# ==========================================================================
# Config
# ==========================================================================
@dataclass(frozen=True)
class SearchSpaceConfig:
    seed: int = 0
    numerical_tol: float = 1e-9
    bounds_mode: str = "reference_quantile"
    lower_quantile: float = 0.001
    upper_quantile: float = 0.999
    margin: float = 0.0
    absolute_lower: Optional[float] = None
    absolute_upper: Optional[float] = None
    radius_enabled: bool = False
    radius_max: float = 3.0
    maha_enabled: bool = False
    maha_quantile: float = 0.99
    maha_shrinkage: str = "ledoit_wolf"
    maha_shrinkage_alpha: float = 0.1
    maha_ridge: float = 0.0
    maha_eig_floor_rel: float = 1e-8
    generator_default: str = "uniform_random"
    local_scale: float = 0.1
    local_distribution: str = "gaussian"
    local_truncation: float = 3.0

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "SearchSpaceConfig":
        s = cfg["search_space"]
        c = cls(seed=s["seed"], numerical_tol=s["numerical_tol"],
                bounds_mode=s["bounds"]["mode"], lower_quantile=s["bounds"]["lower_quantile"],
                upper_quantile=s["bounds"]["upper_quantile"], margin=s["bounds"]["margin"],
                absolute_lower=s["bounds"]["absolute_lower"], absolute_upper=s["bounds"]["absolute_upper"],
                radius_enabled=s["constraints"]["radius"]["enabled"], radius_max=s["constraints"]["radius"]["max_radius"],
                maha_enabled=s["constraints"]["mahalanobis"]["enabled"],
                maha_quantile=s["constraints"]["mahalanobis"]["quantile"],
                maha_shrinkage=s["constraints"]["mahalanobis"]["shrinkage"],
                maha_shrinkage_alpha=s["constraints"]["mahalanobis"]["shrinkage_alpha"],
                maha_ridge=s["constraints"]["mahalanobis"]["ridge"],
                maha_eig_floor_rel=s["constraints"]["mahalanobis"]["eigenvalue_floor_rel"],
                generator_default=s["generators"]["default"],
                local_scale=s["generators"]["local_perturbation"]["scale"],
                local_distribution=s["generators"]["local_perturbation"]["distribution"],
                local_truncation=s["generators"]["local_perturbation"]["truncation"])
        c.validate()
        return c

    def validate(self) -> None:
        def num(v): return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
        bad = []
        if not (isinstance(self.seed, int) and not isinstance(self.seed, bool) and self.seed >= 0): bad.append("seed must be a non-negative integer")
        if not (num(self.numerical_tol) and self.numerical_tol >= 0): bad.append("numerical_tol must be a finite number >= 0")
        if self.bounds_mode not in ("reference_minmax", "reference_quantile", "absolute"): bad.append("bounds.mode must be reference_minmax | reference_quantile | absolute")
        if not (num(self.margin) and self.margin >= 0): bad.append("bounds.margin must be a finite number >= 0")
        if self.bounds_mode == "reference_quantile" and not (num(self.lower_quantile) and num(self.upper_quantile) and 0 <= self.lower_quantile < self.upper_quantile <= 1):
            bad.append("bounds quantiles must satisfy 0 <= lower_quantile < upper_quantile <= 1")
        if self.bounds_mode == "absolute" and not (num(self.absolute_lower) and num(self.absolute_upper) and self.absolute_lower < self.absolute_upper):
            bad.append("bounds.absolute_lower/upper must be finite with lower < upper")
        if self.radius_enabled and not (num(self.radius_max) and self.radius_max > 0): bad.append("constraints.radius.max_radius must be > 0")
        if self.maha_enabled:
            if not (num(self.maha_quantile) and 0 < self.maha_quantile < 1): bad.append("constraints.mahalanobis.quantile must be in (0, 1)")
            if self.maha_shrinkage not in ("ledoit_wolf", "fixed", "none"): bad.append("constraints.mahalanobis.shrinkage invalid")
            if not (num(self.maha_shrinkage_alpha) and 0 <= self.maha_shrinkage_alpha <= 1): bad.append("constraints.mahalanobis.shrinkage_alpha must be in [0, 1]")
            if not (num(self.maha_ridge) and self.maha_ridge >= 0): bad.append("constraints.mahalanobis.ridge must be >= 0")
            if not (num(self.maha_eig_floor_rel) and self.maha_eig_floor_rel >= 0): bad.append("constraints.mahalanobis.eigenvalue_floor_rel must be >= 0")
        if self.generator_default not in ("uniform_random", "local_perturbation"): bad.append("generators.default must be uniform_random | local_perturbation")
        if not (num(self.local_scale) and self.local_scale > 0): bad.append("generators.local_perturbation.scale must be > 0")
        if self.local_distribution not in ("gaussian", "uniform"): bad.append("generators.local_perturbation.distribution must be gaussian | uniform")
        if not (num(self.local_truncation) and self.local_truncation > 0): bad.append("generators.local_perturbation.truncation must be > 0")
        if bad:
            raise ValueError("Invalid search_space config: " + "; ".join(bad))


def _ro(a) -> np.ndarray:
    a = np.array(a, dtype=np.float64, copy=True)
    a.setflags(write=False)
    return a


# ==========================================================================
# Search space
# ==========================================================================
@dataclass(frozen=True, eq=False)
class SearchSpace:
    feature_names: Tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    ref_mean: np.ndarray                        # reference statistics (optional use)
    ref_std: np.ndarray
    radius_max: Optional[float] = None
    maha_whiten: Optional[np.ndarray] = None    # (d, d); distance = ||(x-ref_mean) @ W||
    maha_tau: Optional[float] = None
    tol: float = 1e-9
    provenance: Dict[str, Any] = field(default_factory=dict)
    space_id: str = field(init=False, default="")

    def __post_init__(self):
        names = tuple(str(n) for n in self.feature_names)
        d = len(names)
        if d == 0 or len(set(names)) != d:
            raise ValueError("feature_names must be non-empty and unique")
        arrs = {k: _ro(getattr(self, k)) for k in ("lower", "upper", "ref_mean", "ref_std")}
        for k, a in arrs.items():
            if a.shape != (d,) or not np.isfinite(a).all():
                raise ValueError(f"{k} must be finite with shape ({d},)")
        if (arrs["lower"] > arrs["upper"]).any():
            raise ValueError("lower must be <= upper for every feature")
        if (arrs["ref_std"] < 0).any():
            raise ValueError("ref_std must be >= 0")
        W = None
        if self.maha_whiten is not None:
            W = _ro(self.maha_whiten)
            if W.shape != (d, d) or not np.isfinite(W).all() or self.maha_tau is None or not self.maha_tau > 0:
                raise ValueError("invalid Mahalanobis constraint state")
        if self.radius_max is not None and not (np.isfinite(self.radius_max) and self.radius_max > 0):
            raise ValueError("radius_max must be > 0")
        object.__setattr__(self, "feature_names", names)
        for k, a in arrs.items():
            object.__setattr__(self, k, a)
        object.__setattr__(self, "maha_whiten", W)
        h = hashlib.sha256()
        h.update("\x1f".join(names).encode())
        for a in (arrs["lower"], arrs["upper"], arrs["ref_mean"], arrs["ref_std"]):
            h.update(a.astype("<f8").tobytes())
        h.update(repr((self.radius_max, self.maha_tau, self.tol)).encode())
        if W is not None:
            h.update(W.astype("<f8").tobytes())
        object.__setattr__(self, "space_id", h.hexdigest()[:16])

    # ---- basic properties ------------------------------------------------------
    @property
    def dim(self) -> int:
        return len(self.feature_names)

    @property
    def has_radius(self) -> bool:
        return self.radius_max is not None

    @property
    def has_mahalanobis(self) -> bool:
        return self.maha_whiten is not None

    # ---- construction (feature matrix + names ONLY) ------------------------------
    @classmethod
    def from_reference(cls, X: np.ndarray, feature_names: Sequence[str], config: SearchSpaceConfig) -> "SearchSpace":
        config.validate()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("reference matrix must be 2-D with at least 2 rows")
        if X.shape[1] != len(feature_names):
            raise ValueError("reference matrix and feature_names disagree on dimension")
        if not np.isfinite(X).all():
            raise ValueError("reference matrix contains NaN/inf")
        d = X.shape[1]
        if config.bounds_mode == "reference_minmax":
            lo, hi = X.min(axis=0), X.max(axis=0)
        elif config.bounds_mode == "reference_quantile":
            lo, hi = np.quantile(X, config.lower_quantile, axis=0), np.quantile(X, config.upper_quantile, axis=0)
        else:
            lo, hi = np.full(d, float(config.absolute_lower)), np.full(d, float(config.absolute_upper))
        span = hi - lo
        lo, hi = lo - config.margin * span, hi + config.margin * span
        mean, std = X.mean(axis=0), X.std(axis=0, ddof=1)
        widened = 0
        if config.radius_enabled or config.maha_enabled:      # keep the shrink centre inside the box
            nlo, nhi = np.minimum(lo, mean), np.maximum(hi, mean)
            widened = int(((nlo < lo) | (nhi > hi)).sum())
            lo, hi = nlo, nhi
        W = tau = None
        if config.maha_enabled:
            m = MahalanobisValidity(config.maha_shrinkage, config.maha_shrinkage_alpha, config.maha_ridge,
                                    config.maha_eig_floor_rel).fit(X)
            tau = float(np.quantile(m.distance(X), config.maha_quantile))
            W = m.whiten_
        ref_fp = hashlib.sha256(np.ascontiguousarray(X, dtype="<f8").tobytes() + "\x1f".join(feature_names).encode()).hexdigest()[:16]
        prov = {"config": asdict(config), "reference_fingerprint": ref_fp, "n_reference": int(X.shape[0]),
                "n_bounds_widened_to_include_mean": widened, "bounds_mode": config.bounds_mode,
                "n_zero_range_features": int((hi == lo).sum())}
        return cls(tuple(feature_names), lo, hi, mean, std,
                   config.radius_max if config.radius_enabled else None, W, tau, config.numerical_tol, prov)

    @classmethod
    def from_prepared(cls, prepared, config: SearchSpaceConfig) -> "SearchSpace":
        """Reads ONLY `X_train` and `feature_names` (no labels, no test split, no model)."""
        return cls.from_reference(prepared.X_train, prepared.feature_names, config)

    # ---- validation / measures ---------------------------------------------------
    def validate_matrix(self, X: Any, name: str = "X") -> np.ndarray:
        A = np.asarray(X, dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if A.ndim != 2 or A.shape[1] != self.dim:
            raise ValueError(f"{name} must have shape (n, {self.dim}), got {A.shape}")
        if not np.isfinite(A).all():
            raise ValueError(f"{name} contains NaN/inf")
        return A

    def radius(self, X: Any) -> np.ndarray:
        A = self.validate_matrix(X)
        z = (A - self.ref_mean) / np.where(self.ref_std > 0, self.ref_std, 1.0)
        return np.sqrt((z * z).mean(axis=1))

    def mahalanobis(self, X: Any) -> np.ndarray:
        if not self.has_mahalanobis:
            raise ValueError("this search space has no Mahalanobis constraint")
        Z = (self.validate_matrix(X) - self.ref_mean) @ self.maha_whiten
        return np.sqrt((Z * Z).sum(axis=1))

    def contains(self, X: Any) -> np.ndarray:
        """Boolean mask: inside the box and every enabled constraint (small numerical tolerance)."""
        A = self.validate_matrix(X)
        ok = ((A >= self.lower - self.tol) & (A <= self.upper + self.tol)).all(axis=1)
        if self.has_radius:
            ok &= self.radius(A) <= self.radius_max * (1 + self.tol) + self.tol
        if self.has_mahalanobis:
            ok &= self.mahalanobis(A) <= self.maha_tau * (1 + self.tol) + self.tol
        return ok

    def project(self, X: Any) -> np.ndarray:
        """Clip to the box, shrink radially toward the reference mean to satisfy enabled
        constraints, clip again. Finite input required; idempotent; never uses a model."""
        A = np.clip(self.validate_matrix(X), self.lower, self.upper)
        for dist, limit in ((self.radius, self.radius_max if self.has_radius else None),
                            (self.mahalanobis if self.has_mahalanobis else None, self.maha_tau)):
            if limit is None or dist is None or A.shape[0] == 0:
                continue
            r = dist(A)
            f = np.where(r > limit, limit * (1.0 - EPS_SHRINK) / np.maximum(r, 1e-300), 1.0)
            A = self.ref_mean + (A - self.ref_mean) * f[:, None]
        return np.clip(A, self.lower, self.upper)

    # ---- reproducible sampling -----------------------------------------------------
    @staticmethod
    def candidate_rng(seed: int, stream: int, index: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream), int(index)]))

    def sample_uniform_matrix(self, n: int, seed: int, start_index: int = 0) -> np.ndarray:
        """n points uniform in the box (then projected if constraints are enabled)."""
        X = np.empty((n, self.dim))
        span = self.upper - self.lower
        for i in range(n):
            X[i] = self.lower + span * self.candidate_rng(seed, STREAM_UNIFORM, start_index + i).random(self.dim)
        return self.project(X) if n else X

    # ---- serialisation ----------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"format": _FORMAT, "space_id": self.space_id, "feature_names": list(self.feature_names),
                "lower": self.lower.tolist(), "upper": self.upper.tolist(),
                "ref_mean": self.ref_mean.tolist(), "ref_std": self.ref_std.tolist(),
                "radius_max": self.radius_max,
                "maha_whiten": None if self.maha_whiten is None else self.maha_whiten.tolist(),
                "maha_tau": self.maha_tau, "tol": self.tol, "provenance": self.provenance}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SearchSpace":
        if d.get("format") != _FORMAT:
            raise ValueError(f"unsupported search-space format {d.get('format')!r}")
        s = cls(tuple(d["feature_names"]), np.asarray(d["lower"]), np.asarray(d["upper"]), np.asarray(d["ref_mean"]),
                np.asarray(d["ref_std"]), d["radius_max"],
                None if d["maha_whiten"] is None else np.asarray(d["maha_whiten"]), d["maha_tau"], d["tol"],
                dict(d.get("provenance", {})))
        if d.get("space_id") != s.space_id:
            raise ValueError("space_id mismatch: corrupted or edited search-space file")
        return s

    def to_json(self, path) -> Path:
        return save_json(self.to_dict(), path)

    @classmethod
    def from_json(cls, path) -> "SearchSpace":
        return cls.from_dict(json.load(open(resolve_path(path))))
