"""Hybrid classical-quantum search (Phase 7). A SEARCH MECHANISM, not a failure detector.



    SearchSpace -> PCA latent representation (TRAINING reference features only; Phase 6 code)

        -> classical kernel surrogate of the SEARCH SCORE over latent space (counted observations only)

        -> choose a promising latent region (a box around an elite observed point)

        -> encode the region into a small variational circuit (quantum_encoding.py)

        -> train the circuit's parameters so that its measurement distribution favours cells whose

           SURROGATE score is high (diagonal cost, parameter-shift gradients; quantum_circuit.py)

        -> measure; every measured bitstring names one cell; draw a candidate inside it

        -> decode latent -> 77-D features -> SearchSpace.project (inside PCARepresentation.decode)

        -> drop duplicates -> FeedbackManager.evaluate -> FailureValidator -> QueryBudgetedModel

        -> search scores come back -> surrogate / region / radius update -> next iteration



WHAT THE QUANTUM PART SEES: only numbers derived from surrogate search scores of candidates that were

evaluated through the shared, counted gateway. It never sees labels, oracle output, the test split, the

target-model object, or a failure verdict (`Feedback` carries a search score, not the verdict).

`search_score != verified_failure`: only `FailureValidator` establishes a verified failure.



QUERY ACCOUNTING. Identical to Phase 6: all evaluation goes through `FeedbackManager.evaluate`.

Circuit executions and shots cost ZERO target-model queries and are tallied separately

(`reproducibility["quantum"]["cost"]`). The initial design is counted against the budget (no free warm-up).



DEFAULTS are a-priori placeholders; do not tune them on failure outcomes. No quantum advantage is

assumed. With the default 4 qubits there are only 16 cells per iteration, so classically enumerating the

cell scores is an obvious comparison that may match or beat the circuit (not tested here); this prototype

exists to test the *hybrid mechanism and its accounting*, not to claim a benefit.

"""

from __future__ import annotations



import math

import time

from dataclasses import dataclass

from typing import Any, Dict, List, Optional



import numpy as np



from .candidate import Candidate

from .quantum_circuit import (BACKENDS, ENTANGLEMENTS, QuantumCircuitConfig, VariationalCircuit, VQAOptimizer,

                              make_backend)

from .quantum_encoding import LatentCellEncoding

from .search_base import SearchEngine, stream_rng

from .search_representation import make_representation



STREAM_Q_INIT, STREAM_Q_SAMPLE, STREAM_Q_JIT, STREAM_Q_FALLBACK = 31, 32, 33, 34

STOP_MAX_ITER = "max_iterations_reached"



LIMITATIONS = (

    "classical state-vector simulation of an ideal, noise-free circuit; no quantum hardware was used and no hardware claim is made",

    "no quantum advantage is assumed or claimed; the circuit is evaluated only as a search mechanism under a matched target-model query budget",

    "with n qubits the circuit proposes among only 2**n cells per iteration; classically enumerating the same surrogate scores is an obvious comparison and may match or beat it for small n (not tested)",

    "the search score is a ranking proxy; the ReferenceOracle is a proxy validation mechanism based on reference-data consensus and is not ground-truth labeling",

    "the latent representation (PCA of the training reference features) is low-dimensional and cannot reach candidates outside its affine subspace",

)





# ==========================================================================

# Config

# ==========================================================================

@dataclass(frozen=True)

class QuantumSearchConfig:

    backend: str = "auto"                  # auto | numpy_statevector | pennylane

    n_qubits: int = 4

    depth: int = 2

    shots: int = 256

    latent_dimension: int = 4              # PCA axes; must be <= n_qubits

    latent_bound: float = 2.0              # latent box half-width in axis-std units (as Phase 6 PCA)

    initialization: str = "uniform"         # uniform | reference_guided

    seed: int = 0                          # salt combined with the run seed to derive the circuit seed

    iterations: Optional[int] = None       # cap on quantum iterations (null = until budget / proposal cap)

    entanglement: str = "ring"

    init_fraction: float = 0.1             # initial design as a fraction of the budget (counted, never free)

    min_initial: int = 5

    n_initial: Optional[int] = None

    batch_size: int = 8                    # candidates proposed per quantum iteration

    opt_steps: int = 10                    # Adam steps on the circuit parameters per iteration

    learning_rate: float = 0.2

    region_top_k: int = 3                  # region centres rotate over the k best observed points

    radius_init: float = 0.5               # region half-width, as a fraction of latent_bound

    radius_min: float = 0.05

    radius_max: float = 1.0

    radius_expand: float = 1.3             # on improvement of the best observed search score

    radius_shrink: float = 0.7             # otherwise

    kernel_length: float = 0.5             # Gaussian-kernel length of the surrogate (latent units)

    exploration_weight: float = 0.2        # weight of the novelty bonus (1 - kernel density) in the cell score

    max_surrogate_points: int = 500



    @classmethod

    def from_cfg(cls, cfg: Dict[str, Any]) -> "QuantumSearchConfig":

        q = cfg["search"]["quantum"]

        c = cls(**{k: q[k] for k in cls.__dataclass_fields__ if k in q})

        c.validate()

        return c



    def circuit_config(self, seed: int) -> QuantumCircuitConfig:

        return QuantumCircuitConfig(self.backend, self.n_qubits, self.depth, self.shots, seed, self.entanglement)



    def validate(self) -> None:

        ok = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

        pint = lambda v, m=1: isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v >= m

        bad = []

        if self.backend not in BACKENDS: bad.append(f"backend must be one of {BACKENDS}")

        if not pint(self.n_qubits): bad.append("n_qubits must be an integer >= 1")

        if not pint(self.depth): bad.append("depth must be an integer >= 1")

        if not pint(self.shots): bad.append("shots must be an integer >= 1")

        if not pint(self.latent_dimension): bad.append("latent_dimension must be an integer >= 1")

        if pint(self.n_qubits) and pint(self.latent_dimension) and self.latent_dimension > self.n_qubits:

            bad.append("latent_dimension must be <= n_qubits (every latent axis needs a qubit)")

        if not (ok(self.latent_bound) and self.latent_bound > 0): bad.append("latent_bound must be > 0")

        if self.initialization not in ("uniform", "reference_guided"):

            bad.append("initialization must be 'uniform' or 'reference_guided'")

        if not pint(self.seed, 0): bad.append("seed must be an integer >= 0")

        if self.iterations is not None and not pint(self.iterations): bad.append("iterations must be null or an integer >= 1")

        if self.entanglement not in ENTANGLEMENTS: bad.append(f"entanglement must be one of {ENTANGLEMENTS}")

        if not (ok(self.init_fraction) and 0 < self.init_fraction <= 1): bad.append("init_fraction must be in (0, 1]")

        if not pint(self.min_initial): bad.append("min_initial must be an integer >= 1")

        if self.n_initial is not None and not pint(self.n_initial): bad.append("n_initial must be null or an integer >= 1")

        if not pint(self.batch_size): bad.append("batch_size must be an integer >= 1")

        if not pint(self.opt_steps, 0): bad.append("opt_steps must be an integer >= 0")

        if not (ok(self.learning_rate) and self.learning_rate > 0): bad.append("learning_rate must be > 0")

        if not pint(self.region_top_k): bad.append("region_top_k must be an integer >= 1")

        for k in ("radius_init", "radius_min", "radius_max"):

            if not (ok(getattr(self, k)) and getattr(self, k) > 0): bad.append(f"{k} must be > 0")

        if not bad and not (self.radius_min <= self.radius_init <= self.radius_max):

            bad.append("need radius_min <= radius_init <= radius_max")

        if not (ok(self.radius_expand) and self.radius_expand >= 1): bad.append("radius_expand must be >= 1")

        if not (ok(self.radius_shrink) and 0 < self.radius_shrink <= 1): bad.append("radius_shrink must be in (0, 1]")

        if not (ok(self.kernel_length) and self.kernel_length > 0): bad.append("kernel_length must be > 0")

        if not (ok(self.exploration_weight) and self.exploration_weight >= 0): bad.append("exploration_weight must be >= 0")

        if not pint(self.max_surrogate_points, 2): bad.append("max_surrogate_points must be an integer >= 2")

        if bad:

            raise ValueError("Invalid search.quantum config: " + "; ".join(bad))





# ==========================================================================

# Classical surrogate of the search score (label-free; counted observations only)

# ==========================================================================

class KernelSurrogate:

    """Nadaraya-Watson (Gaussian-kernel) regression of the SEARCH SCORE over latent coordinates, plus a

    kernel-density novelty term. Deterministic, no hyper-parameter fitting, no RNG."""



    def __init__(self, length: float, exploration_weight: float):

        self.length, self.kappa = float(length), float(exploration_weight)

        self.Z = np.zeros((0, 0))

        self.y = np.zeros(0)



    def fit(self, Z: np.ndarray, y: np.ndarray) -> None:

        self.Z, self.y = np.asarray(Z, float), np.asarray(y, float)



    def predict(self, P: np.ndarray):

        """-> (mean score, density in [0, 1]) at the points P (m, k)."""

        P = np.atleast_2d(np.asarray(P, float))

        if len(self.y) == 0:

            return np.zeros(len(P)), np.zeros(len(P))

        d2 = ((P[:, None, :] - self.Z[None, :, :]) ** 2).sum(-1)

        d2min = d2.min(axis=1, keepdims=True)

        w = np.exp(-(d2 - d2min) / (2.0 * self.length ** 2))

        mu = (w @ self.y) / w.sum(axis=1)

        density = np.exp(-d2min[:, 0] / (2.0 * self.length ** 2))

        return mu, density



    def cell_scores(self, P: np.ndarray) -> np.ndarray:

        mu, dens = self.predict(P)

        return mu + self.kappa * (1.0 - dens)





def _normalised_energies(scores: np.ndarray) -> np.ndarray:

    """Cost values e_b in [-1, 0]: the best cell gets -1. All-equal scores -> zeros (no preference)."""

    s = np.asarray(scores, float)

    span = float(s.max() - s.min())

    if not np.isfinite(span) or span < 1e-12:

        return np.zeros_like(s)

    return -(s - s.min()) / span





def allocate_by_counts(counts: np.ndarray, n: int) -> np.ndarray:

    """Largest-remainder allocation of `n` proposals to cells in proportion to measured counts.

    Deterministic. Returns the cell index of each proposal, ordered by decreasing count (ties: lower index)."""

    counts = np.asarray(counts, dtype=np.int64)

    total = int(counts.sum())

    if n <= 0 or total <= 0:

        return np.zeros(0, dtype=np.int64)

    exp = n * counts / total

    base = np.floor(exp).astype(np.int64)

    rem = int(n - base.sum())

    if rem > 0:

        order = np.lexsort((np.arange(len(counts)), -(exp - base)))

        base[order[:rem]] += 1

    cells: List[int] = []

    for c in np.lexsort((np.arange(len(counts)), -counts)):

        cells.extend([int(c)] * int(base[c]))

    return np.asarray(cells, dtype=np.int64)





# ==========================================================================

# Engine

# ==========================================================================

class QuantumSearch(SearchEngine):

    method = "quantum"



    def __init__(self, space, validator, query_budget, seed, common, method_cfg: Optional[QuantumSearchConfig] = None, *,

                 paper_eligible_context: bool = False, run_context: Optional[Dict[str, Any]] = None):

        method_cfg = method_cfg or QuantumSearchConfig()

        method_cfg.validate()

        super().__init__(space, validator, query_budget, seed, common, method_cfg,

                         paper_eligible_context=paper_eligible_context)

        c = method_cfg

        # TRAINING reference FEATURES only (ReferenceData.X); no labels, no test data

        self.rep = make_representation("pca", space, validator.ref.X, c.latent_dimension, c.latent_bound)

        if self.rep.dim < c.latent_dimension:

            self.warnings.append(f"latent dimension reduced from {c.latent_dimension} to {self.rep.dim} "

                                 f"(reference covariance rank)")

        self.enc = LatentCellEncoding(self.rep.dim, c.n_qubits, c.latent_bound)

        circuit_seed = int(np.random.SeedSequence([self.seed, c.seed]).generate_state(1)[0] % (2 ** 31 - 1))

        backend, notes = make_backend(c.backend)

        self.warnings.extend(notes)

        self.warnings.append("quantum search runs on a classical simulator of an ideal circuit; no quantum advantage is claimed")

        self.circuit = VariationalCircuit(c.circuit_config(circuit_seed), backend)

        self.opt = VQAOptimizer(c.learning_rate)

        self.surrogate = KernelSurrogate(c.kernel_length, c.exploration_weight)

        self.run_context = dict(run_context or {})

        self._circuit_seed = circuit_seed

        self._theta = self.circuit.initial_params()

        self._radius = c.radius_init * c.latent_bound

        self._Z: List[np.ndarray] = []

        self._y: List[float] = []

        self.iterations = 0

        self.fallbacks = 0

        self.duplicates_dropped = 0

        self.opt_steps_total = 0

        self.seconds_quantum_total = 0.0

        self._log: List[Dict[str, float]] = []

        self.last_iteration: Optional[Dict[str, Any]] = None       # for tests/diagnostics

        self._t_start = time.perf_counter()



    # ---- helpers ---------------------------------------------------------------------------

    def _n_initial(self) -> int:

        c = self.cfg

        n = c.n_initial if c.n_initial is not None else max(c.min_initial, int(math.ceil(c.init_fraction * self.query_budget)))

        return int(min(n, self.query_budget))



    def initial_design(self, n: int) -> np.ndarray:

        """Generate deterministic uniform or reference-guided latent points; query-free."""

        if n <= 0:

            return np.zeros((0, self.rep.dim))



        B, k = self.cfg.latent_bound, self.rep.dim

        if self.cfg.initialization == "uniform":

            return np.stack([

                B * (2.0 * stream_rng(self.seed, STREAM_Q_INIT, i).random(k) - 1.0)

                for i in range(n)

            ])



        reference_X = np.asarray(self.validator.ref.X, dtype=float)

        reference_X = reference_X[np.all(np.isfinite(reference_X), axis=1)]

        if len(reference_X) == 0:

            raise ValueError(

                "Reference-guided initialization requires finite reference rows."

            )



        reference_Z = self.rep.encode(reference_X)

        latent_std = np.maximum(np.std(reference_Z, axis=0), 1e-6)

        points = []

        for i in range(n):

            rng = stream_rng(self.seed, STREAM_Q_INIT, i)

            anchor = reference_Z[int(rng.integers(len(reference_Z)))]

            point = anchor + rng.normal(size=k) * 0.10 * latent_std

            points.append(np.clip(point, -B, B))

        return np.stack(points)



    def _candidates(self, V: np.ndarray, origin: str, metas: List[Dict[str, Any]]) -> List[Candidate]:

        X = self.rep.decode(V)                                   # latent -> features -> SearchSpace.project

        return [Candidate.from_array(X[i], self.space.feature_names, origin, metas[i]) for i in range(len(X))]



    def _observe(self, feedback) -> float:

        best_new = -np.inf

        for fb in feedback:

            if fb.duplicate:

                continue

            self._Z.append(self.rep.encode(fb.candidate.x[None])[0])      # coordinates actually evaluated (repaired)

            self._y.append(float(fb.score))

            best_new = max(best_new, float(fb.score))

        return best_new



    def _surrogate_fit(self) -> None:

        Z, y = np.stack(self._Z), np.asarray(self._y)

        cap = self.cfg.max_surrogate_points

        if len(y) > cap:                                        # top quarter + most recent (deterministic)

            n_top = cap // 4

            idx = np.arange(len(y))

            top = idx[np.argsort(-y, kind="stable")[:n_top]]

            idx = np.sort(np.concatenate([top, np.setdiff1d(idx, top)[-(cap - n_top):]]))

            Z, y = Z[idx], y[idx]

        self.surrogate.fit(Z, y)



    # ---- one quantum iteration ---------------------------------------------------------------------

    def _propose(self, it: int, session) -> List[Candidate]:

        c = self.cfg

        t0 = time.perf_counter()

        order = np.argsort(-np.asarray(self._y), kind="stable")

        top = order[:min(c.region_top_k, len(order))]

        centre = self._Z[int(top[(it - 1) % len(top)])]

        lo, hi = self.enc.region(centre, self._radius)

        self._surrogate_fit()

        cell_pts = self.enc.cell_centres(lo, hi)

        energies = _normalised_energies(self.surrogate.cell_scores(cell_pts))

        angles = self.enc.angles(centre)

        exec0 = self.circuit.executions

        theta, hist = self.opt.train(self.circuit, self._theta, angles, energies, c.opt_steps)

        self._theta = theta                                     # warm start for the next iteration

        probs = self.circuit.probabilities(theta, angles)

        counts = self.circuit.sample_counts(probs, stream_rng(self.seed, STREAM_Q_SAMPLE, it))

        self.opt_steps_total += c.opt_steps

        n = min(c.batch_size, session.proposals_left, max(int(session.bb.remaining), 1))   # read-only counter; never over-generates the last batch

        cells = allocate_by_counts(counts, n)

        V = self.enc.sample_in_cells(cells, lo, hi, stream_rng(self.seed, STREAM_Q_JIT, it))

        metas = [{"generator": "quantum", "seed": self.seed, "iteration": it, "index": j, "cell": int(cells[j]),

                  "bitstring": format(int(cells[j]), f"0{self.circuit.n}b")} for j in range(len(cells))]

        cands = self._candidates(V, "quantum", metas)

        keep, seen_here = [], set()

        for cand in cands:                                      # duplicate policy (same as the shared gateway)

            dup = cand.candidate_id in seen_here or (self.common.duplicate_policy == "skip" and session.has_seen(cand.candidate_id))

            if dup:

                self.duplicates_dropped += 1

                continue

            seen_here.add(cand.candidate_id)

            keep.append(cand)

        if not keep:                                            # nothing new: fresh uniform latent point

            self.fallbacks += 1

            for j in range(20):

                v = self.cfg.latent_bound * (2.0 * stream_rng(self.seed, STREAM_Q_FALLBACK, it, j).random(self.rep.dim) - 1.0)

                fb_c = self._candidates(v[None], "quantum_fallback",

                                        [{"generator": "quantum_fallback", "seed": self.seed, "iteration": it, "index": j}])[0]

                if not session.has_seen(fb_c.candidate_id) or self.common.duplicate_policy == "allow":

                    break

            keep = [fb_c]

        p = np.clip(probs, 1e-300, None)

        self._log.append({"energy_before": float(hist[0]), "energy_after": float(hist[-1]),

                          "entropy": float(-(probs * np.log(p)).sum()), "max_prob": float(probs.max()),

                          "radius": float(self._radius), "distinct_cells": int((counts > 0).sum())})

        self.seconds_quantum_total += time.perf_counter() - t0

        self.last_iteration = {"iteration": it, "centre": centre.copy(), "radius": float(self._radius), "lo": lo, "hi": hi,

                               "energies": energies, "probs": probs, "counts": counts, "cells": cells,

                               "executions": self.circuit.executions - exec0, "n_proposed": len(keep)}

        return keep



    # ---- main loop -----------------------------------------------------------------------------------

    def _search(self, session) -> str:

        c = self.cfg

        self._t_start = time.perf_counter()

        n0 = min(self._n_initial(), session.proposals_left)

        V0 = self.initial_design(n0)

        init = self._candidates(

            V0,

            "quantum_init",

            [

                {

                    "generator": "quantum_init",

                    "seed": self.seed,

                    "iteration": 0,

                    "index": i,

                }

                for i in range(n0)

            ],

        )

        session.note_generated(len(init))

        self._observe(session.evaluate(init))

        it = 0



        while not session.stopped:

            if not self._y:

                it += 1

                # Generate a fresh deterministic fallback point on every attempt.

                rng = stream_rng(self.seed, STREAM_Q_FALLBACK, it)

                if c.initialization == "uniform":
                    fallback_points = c.latent_bound * (
                        2.0 * rng.random(self.rep.dim) - 1.0
                    )
                else:
                    reference_X = np.asarray(self.validator.ref.X, dtype=float)
                    reference_X = reference_X[
                        np.all(np.isfinite(reference_X), axis=1)
                    ]
                    if len(reference_X) == 0:
                        raise ValueError(
                            "Reference-guided initialization requires finite reference rows."
                        )

                    reference_Z = self.rep.encode(reference_X)
                    latent_std = np.maximum(np.std(reference_Z, axis=0), 1e-6)
                    anchor = reference_Z[int(rng.integers(len(reference_Z)))]
                    fallback_points = np.clip(
                        anchor + rng.normal(size=self.rep.dim) * 0.10 * latent_std,
                        -c.latent_bound,
                        c.latent_bound,
                    )

                fallback_points = fallback_points[None, :]

                cands = self._candidates(

                    fallback_points,

                    "quantum_fallback",

                    [

                        {

                            "generator": "quantum_fallback",

                            "initialization": c.initialization,

                            "seed": self.seed,

                            "iteration": it,

                            "index": 0,

                        }

                    ],

                )

                self.fallbacks += 1

                session.note_generated(len(cands))

                self._observe(session.evaluate(cands))

                continue



            if c.iterations is not None and self.iterations >= c.iterations:

                return STOP_MAX_ITER



            it += 1

            self.iterations += 1

            cands = self._propose(it, session)

            best_before = max(self._y)

            session.note_generated(len(cands))

            best_new = self._observe(session.evaluate(cands))



            if best_new > best_before:

                self._radius = min(

                    self._radius * c.radius_expand,

                    c.radius_max * c.latent_bound,

                )

            else:

                self._radius = max(

                    self._radius * c.radius_shrink,

                    c.radius_min * c.latent_bound,

                )



        return session.stop_reason





    # ---- reporting ---------------------------------------------------------------------------------------

    def extra_reproducibility(self) -> Dict[str, Any]:

        c, log = self.cfg, self._log

        mean = lambda k: float(np.mean([r[k] for r in log])) if log else None

        q = {

            **self.circuit.describe(),

            "backend_requested": c.backend, "latent_dimension": self.rep.dim, "latent_dimension_requested": c.latent_dimension,

            "latent_bound": c.latent_bound, "quantum_seed_salt": c.seed, "circuit_seed": self._circuit_seed,

            "encoding": self.enc.describe(),

            "objective": "diagonal cost sum_b p(b)*e_b with e_b = -normalised SURROGATE search score of cell b "

                         "(search score from counted observations; never oracle labels or verdicts)",

            "classical_surrogate": {"kind": "gaussian_kernel_regression", "kernel_length": c.kernel_length,

                                    "exploration_weight": c.exploration_weight, "n_observations": len(self._y)},

            "iterations": self.iterations, "fallback_points": self.fallbacks, "duplicates_dropped_before_evaluation": self.duplicates_dropped,

            "cost": {"circuit_executions": self.circuit.executions, "shots_sampled": self.circuit.shots_sampled,

                     "optimisation_steps": self.opt_steps_total, "target_queries_charged": 0},

            # wall-clock only; the one non-deterministic block (compare results with this key removed)

            "runtime": {"search_seconds": float(time.perf_counter() - self._t_start),

                        "circuit_seconds": float(self.circuit.seconds),

                        "quantum_iteration_seconds": float(self.seconds_quantum_total)},

            "circuit_training": {"mean_energy_before": mean("energy_before"), "mean_energy_after": mean("energy_after"),

                                 "mean_entropy": mean("entropy"), "mean_max_probability": mean("max_prob"),

                                 "mean_distinct_cells_per_iteration": mean("distinct_cells")},

            "limitations": list(LIMITATIONS),

        }

        out = {"representation": self.rep.describe(), "quantum": q, "n_observations": len(self._y)}

        if self.run_context:

            out["run_context"] = self.run_context

        return out
