"""Optimiser coordinates <-> candidate feature vectors (Phase 6). Query-free and LABEL-FREE.

Built from the TRAINING reference FEATURE matrix only (`ReferenceData.X`) and the `SearchSpace`.
No labels, no test split, no target model, no search outcome.

* `StandardizedFullRepresentation` ('full'): v = (x - ref_mean) / ref_std (zero std -> 1). Same units
  as the Phase 4/5 perturbation scales ("fraction of a feature's reference std"). Bounds = search-space box.
* `PCARepresentation` ('pca'): top-k principal axes of the reference features; v = coordinates in
  units of each axis' standard deviation, box [-B, B]^k. decode(v) = project(mean + (v*sd) @ V^T).
  Used by default for the Gaussian process because a GP on ~77 raw dimensions is unreliable
  (Phase 4 measured an effective rank of ~9 of 77 on CIC-IDS2017, so a ~10-D subspace is natural).
  Deterministic: eigendecomposition of the reference covariance, eigenvector sign fixed so the
  largest-|entry| is positive, components ordered by decreasing eigenvalue.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict

import numpy as np

from .search_space import SearchSpace

REPRESENTATIONS = ("full", "pca")


class Representation:
    kind = "abstract"
    dim: int
    lower: np.ndarray       # coordinate bounds
    upper: np.ndarray

    def decode(self, V: np.ndarray) -> np.ndarray:           # (n, dim) -> (n, d) feature vectors in the space
        raise NotImplementedError

    def encode(self, X: np.ndarray) -> np.ndarray:           # (n, d) -> (n, dim) coordinates (clipped to bounds)
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError


class StandardizedFullRepresentation(Representation):
    kind = "full"

    def __init__(self, space: SearchSpace):
        self.space = space
        self.scale = np.where(space.ref_std > 0, space.ref_std, 1.0)
        self.dim = space.dim
        self.lower = (space.lower - space.ref_mean) / self.scale
        self.upper = (space.upper - space.ref_mean) / self.scale

    def decode(self, V):
        V = np.clip(np.atleast_2d(np.asarray(V, float)), self.lower, self.upper)
        return self.space.project(self.space.ref_mean + V * self.scale)

    def encode(self, X):
        X = self.space.validate_matrix(X)
        return np.clip((X - self.space.ref_mean) / self.scale, self.lower, self.upper)

    def describe(self):
        return {"kind": "full", "dim": self.dim, "space_id": self.space.space_id}


class PCARepresentation(Representation):
    kind = "pca"

    def __init__(self, space: SearchSpace, reference_X: np.ndarray, n_components: int = 10, latent_bound: float = 2.0):
        if not (isinstance(n_components, int) and n_components >= 1):
            raise ValueError("n_components must be an integer >= 1")
        if not (np.isfinite(latent_bound) and latent_bound > 0):
            raise ValueError("latent_bound must be > 0")
        X = space.validate_matrix(reference_X, "reference_X")
        if X.shape[0] < 2:
            raise ValueError("need >= 2 reference rows")
        self.space = space
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        cov = (Xc.T @ Xc) / (X.shape[0] - 1)
        w, V = np.linalg.eigh((cov + cov.T) / 2.0)
        order = np.argsort(-w, kind="stable")
        w, V = np.clip(w[order], 0.0, None), V[:, order]
        keep = int(min(n_components, space.dim, int((w > 1e-12 * max(w[0], 1e-300)).sum()) or 1))
        V = V[:, :keep].copy()
        for j in range(keep):                                   # deterministic sign convention
            if V[np.argmax(np.abs(V[:, j])), j] < 0:
                V[:, j] = -V[:, j]
        self.V, self.sd = V, np.sqrt(w[:keep])
        self.dim = keep
        self.bound = float(latent_bound)
        self.lower, self.upper = np.full(keep, -self.bound), np.full(keep, self.bound)
        self.explained_variance_ratio = (w[:keep] / max(w.sum(), 1e-300)).tolist()
        self.fingerprint = hashlib.sha256(self.V.astype("<f8").tobytes() + self.sd.astype("<f8").tobytes()).hexdigest()[:16]

    def decode(self, V):
        V = np.clip(np.atleast_2d(np.asarray(V, float)), self.lower, self.upper)
        return self.space.project(self.mean + (V * self.sd) @ self.V.T)

    def encode(self, X):
        X = self.space.validate_matrix(X)
        return np.clip(((X - self.mean) @ self.V) / np.where(self.sd > 0, self.sd, 1.0), self.lower, self.upper)

    def describe(self):
        return {"kind": "pca", "dim": self.dim, "latent_bound": self.bound, "fingerprint": self.fingerprint,
                "explained_variance_ratio_total": float(sum(self.explained_variance_ratio)),
                "space_id": self.space.space_id}


def make_representation(kind: str, space: SearchSpace, reference_X: np.ndarray, n_components: int = 10,
                        latent_bound: float = 2.0) -> Representation:
    if kind == "full":
        return StandardizedFullRepresentation(space)
    if kind == "pca":
        return PCARepresentation(space, reference_X, n_components, latent_bound)
    raise ValueError(f"unknown representation {kind!r}; choose from {REPRESENTATIONS}")
