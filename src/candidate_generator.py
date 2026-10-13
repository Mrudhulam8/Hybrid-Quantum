"""Candidate generators (QUERY-FREE, LABEL-FREE). No optimisation happens here.

A generator turns (search space, explicit seed, index range) into candidate feature vectors and
wraps them as Phase 4 `Candidate` objects. It receives only a `SearchSpace` and, for local
perturbation, a matrix of reference ANCHOR FEATURES. It never receives labels, categories, the
target model, predictions or failure outcomes, and it never decides what a failure is - that is
the Phase 4 validator's job. Generating a candidate costs zero target-model queries.

Candidate metadata: {generator, seed, index[, anchor_index]}; `Candidate.origin` = generator name.
The candidate id depends only on the feature vector and schema (Phase 4), not on this metadata.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .candidate import OK, Candidate
from .search_space import STREAM_LOCAL, STREAM_UNIFORM, SearchSpace, SearchSpaceConfig


def _check_int(name: str, v: Any, minimum: int = 0) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {v!r}")
    return int(v)


class CandidateGenerator(ABC):
    name: str = "abstract"
    stream: int = 0

    def __init__(self, space: SearchSpace):
        self.space = space

    @abstractmethod
    def _generate(self, n: int, seed: int, start_index: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Return an (n, d) matrix and one metadata dict per row."""

    def generate_matrix(self, n: int, *, seed: int, start_index: int = 0) -> np.ndarray:
        n, seed, start = _check_int("n", n), _check_int("seed", seed), _check_int("start_index", start_index)
        if n == 0:
            return np.empty((0, self.space.dim))
        X, _ = self._generate(n, seed, start)
        return self.space.validate_matrix(X, "generated candidates")

    def generate(self, n: int, *, seed: int, start_index: int = 0) -> List[Candidate]:
        """Exactly `n` Phase 4 Candidates. `seed` is mandatory and explicit (no hidden global RNG)."""
        n, seed, start = _check_int("n", n), _check_int("seed", seed), _check_int("start_index", start_index)
        if n == 0:
            return []
        X, metas = self._generate(n, seed, start)
        X = self.space.validate_matrix(X, "generated candidates")
        out = [Candidate.from_array(X[i], self.space.feature_names, origin=self.name, meta=metas[i]) for i in range(n)]
        if not all(c.check(self.space.feature_names) == OK for c in out):      # explicit (survives python -O)
            raise RuntimeError("generator produced a malformed candidate (internal invariant violated)")
        return out

    def describe(self) -> Dict[str, Any]:
        return {"generator": self.name, "stream": self.stream, "space_id": self.space.space_id}


# ==========================================================================
# A. Uniform random
# ==========================================================================
class UniformRandomGenerator(CandidateGenerator):
    """Uniform samples inside the configured bounds (projected if constraints are enabled)."""
    name = "uniform_random"
    stream = STREAM_UNIFORM

    def _generate(self, n, seed, start_index):
        X = self.space.sample_uniform_matrix(n, seed, start_index)
        return X, [{"generator": self.name, "seed": seed, "index": start_index + i} for i in range(n)]


# ==========================================================================
# B. Local perturbation of real reference samples
# ==========================================================================
@dataclass(frozen=True)
class LocalPerturbationParams:
    scale: float = 0.1                 # fraction of each feature's reference std
    distribution: str = "gaussian"     # gaussian | uniform
    truncation: float = 3.0            # gaussian clipped to +-truncation (units of scale*std)

    def validate(self) -> None:
        ok = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
        bad = []
        if not (ok(self.scale) and self.scale > 0): bad.append("scale must be > 0")
        if self.distribution not in ("gaussian", "uniform"): bad.append("distribution must be gaussian | uniform")
        if not (ok(self.truncation) and self.truncation > 0): bad.append("truncation must be > 0")
        if bad:
            raise ValueError("Invalid local-perturbation params: " + "; ".join(bad))

    @classmethod
    def from_config(cls, c: SearchSpaceConfig) -> "LocalPerturbationParams":
        p = cls(c.local_scale, c.local_distribution, c.local_truncation)
        p.validate()
        return p

    @property
    def max_abs_multiple(self) -> float:
        """Largest possible |raw perturbation| in units of (scale * std)."""
        return self.truncation if self.distribution == "gaussian" else 1.0


class LocalPerturbationGenerator(CandidateGenerator):
    """anchor (a real reference row, drawn uniformly) + bounded noise, then projection.

    Raw perturbation: delta_j = scale * std_j * xi_j, with xi_j ~ N(0,1) clipped to +-truncation
    (gaussian) or U(-1, 1) (uniform), so |delta_j| <= max_abs_multiple * scale * std_j always.
    Projection can move the final vector further (see SearchSpace.project): with only the box (or
    only the radius ball) active it never moves it farther from the projected anchor than delta.
    """
    name = "local_perturbation"
    stream = STREAM_LOCAL

    def __init__(self, space: SearchSpace, anchors: np.ndarray, params: Optional[LocalPerturbationParams] = None):
        super().__init__(space)
        self.params = params or LocalPerturbationParams()
        self.params.validate()
        A = np.asarray(anchors, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] < 1:
            raise ValueError("anchors must be a non-empty 2-D feature matrix")
        self.anchors = space.validate_matrix(A, "anchors")           # finite + right dimension
        self.anchors.setflags(write=False)
        self.anchor_fingerprint = hashlib.sha256(np.ascontiguousarray(self.anchors, dtype="<f8").tobytes()).hexdigest()[:16]

    def _generate(self, n, seed, start_index):
        sp, p, d = self.space, self.params, self.space.dim
        idx = np.empty(n, dtype=np.int64)
        noise = np.empty((n, d))
        for i in range(n):
            rng = sp.candidate_rng(seed, self.stream, start_index + i)
            idx[i] = rng.integers(0, self.anchors.shape[0])
            if p.distribution == "gaussian":
                noise[i] = np.clip(rng.standard_normal(d), -p.truncation, p.truncation)
            else:
                noise[i] = rng.uniform(-1.0, 1.0, d)
        base = sp.project(self.anchors[idx])
        X = sp.project(base + p.scale * sp.ref_std * noise)
        return X, [{"generator": self.name, "seed": seed, "index": start_index + i, "anchor_index": int(idx[i])}
                   for i in range(n)]

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "params": asdict(self.params), "n_anchors": int(self.anchors.shape[0]),
                "anchor_fingerprint": self.anchor_fingerprint}


# ==========================================================================
# Factory
# ==========================================================================
GENERATOR_NAMES = ("uniform_random", "local_perturbation")


def make_generator(name: str, space: SearchSpace, config: SearchSpaceConfig,
                   anchors: Optional[np.ndarray] = None) -> CandidateGenerator:
    if name == "uniform_random":
        return UniformRandomGenerator(space)
    if name == "local_perturbation":
        if anchors is None:
            raise ValueError("local_perturbation needs an `anchors` feature matrix (reference rows)")
        return LocalPerturbationGenerator(space, anchors, LocalPerturbationParams.from_config(config))
    raise ValueError(f"unknown generator {name!r}; choose from {GENERATOR_NAMES}")
