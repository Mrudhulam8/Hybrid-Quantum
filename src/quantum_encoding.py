"""Deterministic mapping between the classical latent search region and the quantum circuit (Phase 7).

Label-free, target-free, query-free: it only does geometry on latent coordinates.

LATENT SPACE. Coordinates of the Phase 6 `PCARepresentation` (units of each principal axis' std, box
[-B, B]^k). A "search region" is a latent box  [c - r, c + r] intersected with [-B, B]^k  around a centre c.

CELLS. The n qubits are assigned to the k latent axes round-robin (qubit q -> axis q mod k; needs n >= k),
so axis j owns m_j = |{q : q mod k == j}| qubits. A measured bitstring therefore names ONE cell of the
region: along axis j the region is split into 2**m_j equal slabs, and the m_j bits of that axis (first
owned qubit = most significant) give the slab index. 2**n basis states <-> 2**n cells (a bijection).
A candidate is a point drawn uniformly INSIDE the measured cell with a seeded generator (so repeated
measurements of one cell give distinct, reproducible candidates).

ANGLE ENCODING (region conditioning). Every qubit of axis j receives RY(phi_j),
    phi_j = (pi / 2) * clip(c_j / B, -1, 1)          in [-pi/2, pi/2],
applied after the Hadamard layer. It is deterministic, bounded (numerically stable) and conditions the
initial state on where the region sits in the latent box. The circuit parameters are then trained against
the surrogate cell scores, so the proposal distribution does not rely on this layer carrying information
by itself (see docs/quantum_search_policy.md).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np


class LatentCellEncoding:
    def __init__(self, latent_dim: int, n_qubits: int, latent_bound: float):
        if not (isinstance(latent_dim, (int, np.integer)) and latent_dim >= 1):
            raise ValueError("latent_dim must be an integer >= 1")
        if not (isinstance(n_qubits, (int, np.integer)) and n_qubits >= 1):
            raise ValueError("n_qubits must be an integer >= 1")
        if n_qubits < latent_dim:
            raise ValueError(f"n_qubits ({n_qubits}) must be >= latent_dim ({latent_dim}): every latent axis needs >= 1 qubit")
        if not (math.isfinite(latent_bound) and latent_bound > 0):
            raise ValueError("latent_bound must be > 0")
        self.k, self.n, self.B = int(latent_dim), int(n_qubits), float(latent_bound)
        self.axis_of_qubit = np.arange(self.n) % self.k
        self.bits_per_axis = np.array([int((self.axis_of_qubit == j).sum()) for j in range(self.k)])
        self.n_cells = 2 ** self.n
        # (2**n, n) bit matrix, qubit 0 = most-significant bit of the basis index
        self._bits = (np.arange(self.n_cells)[:, None] >> (self.n - 1 - np.arange(self.n))) & 1
        self._slab = np.zeros((self.n_cells, self.k), dtype=np.int64)          # slab index of each cell along each axis
        for j in range(self.k):
            qs = np.where(self.axis_of_qubit == j)[0]
            m = len(qs)
            for pos, q in enumerate(qs):
                self._slab[:, j] += self._bits[:, q] << (m - 1 - pos)

    # ---- circuit-side ----------------------------------------------------------------------
    def angles(self, centre: np.ndarray) -> np.ndarray:
        c = np.asarray(centre, dtype=np.float64).reshape(-1)
        if c.shape[0] != self.k or not np.isfinite(c).all():
            raise ValueError(f"centre must be {self.k} finite numbers")
        phi = 0.5 * math.pi * np.clip(c / self.B, -1.0, 1.0)
        return phi[self.axis_of_qubit]

    # ---- geometry -----------------------------------------------------------------------------
    def region(self, centre: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
        if not (math.isfinite(radius) and radius > 0):
            raise ValueError("radius must be > 0")
        c = np.clip(np.asarray(centre, dtype=np.float64).reshape(-1), -self.B, self.B)
        lo, hi = np.clip(c - radius, -self.B, self.B), np.clip(c + radius, -self.B, self.B)
        if not (hi > lo).all():
            raise ValueError("degenerate region")
        return lo, hi

    def cell_bounds(self, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(n_cells, k) lower and upper latent corners of every cell of the region [lo, hi]."""
        width = (hi - lo) / (2.0 ** self.bits_per_axis)
        cl = lo + self._slab * width
        return cl, cl + width

    def cell_centres(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        cl, cu = self.cell_bounds(lo, hi)
        return 0.5 * (cl + cu)

    def sample_in_cells(self, cells: np.ndarray, lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One uniform point per entry of `cells` (basis indices); shape (len(cells), k); always inside [-B, B]^k."""
        cells = np.asarray(cells, dtype=np.int64).reshape(-1)
        if cells.size and (cells.min() < 0 or cells.max() >= self.n_cells):
            raise ValueError("cell index out of range")
        cl, cu = self.cell_bounds(lo, hi)
        u = rng.random((len(cells), self.k))
        return cl[cells] + u * (cu[cells] - cl[cells])

    def describe(self) -> Dict[str, Any]:
        return {"latent_dim": self.k, "n_qubits": self.n, "latent_bound": self.B, "n_cells": self.n_cells,
                "bits_per_axis": self.bits_per_axis.tolist(), "angle_map": "RY((pi/2)*clip(c_j/B,-1,1)) after H"}
