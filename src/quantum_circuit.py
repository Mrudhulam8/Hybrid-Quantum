"""Small variational quantum circuit + simulator backends (Phase 7).

This module knows NOTHING about targets, validators, labels or failures. It maps
(parameters, encoding angles) -> measurement distribution over n-qubit bitstrings, and provides a
parameter-shift gradient of a DIAGONAL cost  E(theta) = sum_b p_theta(b) * e_b  (the numbers `e_b` are
supplied by the caller; the circuit never sees where they come from).

CIRCUIT (qubit 0 is the most-significant bit of the bitstring index, as in PennyLane)
    |0...0>  ->  H on every qubit                      (uniform superposition: QAOA-style start)
             ->  RY(enc_q) on every qubit              (angle encoding of the search region; quantum_encoding.py)
             ->  `depth` x [ RY(t_{l,q,0}) RZ(t_{l,q,1}) on every qubit ; CNOT entangling layer ]
             ->  RY(t_final_q) on every qubit
             ->  measurement in the computational basis
    n_params = depth * 2 * n_qubits + n_qubits.   Entangler: 'ring' (q -> q+1 mod n) or 'linear'.

BACKENDS
* `numpy_statevector` : built-in exact state-vector simulator (always available; real amplitudes/gates,
  not a classical surrogate of the algorithm).
* `pennylane`         : PennyLane `default.qubit` adapter. NOT installed in the environment where this file was
  written, so the adapter is untested there; at construction it cross-checks itself against the NumPy
  reference simulator and refuses to run if the two disagree.
* `auto`              : pennylane if importable AND its self-check passes, else numpy_statevector (the fallback
  reason is recorded and surfaced as a warning in search results).
Measurement "shots" are emulated by multinomial sampling from the exact, noise-free probabilities with a
seeded NumPy generator for BOTH backends (ideal simulator; no hardware noise; no hardware claims).
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

BACKENDS = ("auto", "numpy_statevector", "pennylane")
ENTANGLEMENTS = ("ring", "linear")
MAX_QUBITS = 12
STREAM_Q_PARAM_INIT = 35

Op = Tuple[str, Tuple[int, ...], Optional[float]]        # (name, wires, angle or None)


class QuantumBackendUnavailable(ImportError):
    pass


# ==========================================================================
# Config
# ==========================================================================
@dataclass(frozen=True)
class QuantumCircuitConfig:
    backend: str = "auto"
    n_qubits: int = 4
    depth: int = 2
    shots: int = 256
    seed: int = 0
    entanglement: str = "ring"

    def validate(self) -> None:
        pint = lambda v, lo, hi=None: (isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v >= lo
                                       and (hi is None or v <= hi))
        bad = []
        if self.backend not in BACKENDS: bad.append(f"backend must be one of {BACKENDS}")
        if not pint(self.n_qubits, 1, MAX_QUBITS): bad.append(f"n_qubits must be an integer in [1, {MAX_QUBITS}]")
        if not pint(self.depth, 1): bad.append("depth must be an integer >= 1")
        if not pint(self.shots, 1): bad.append("shots must be an integer >= 1")
        if not pint(self.seed, 0): bad.append("seed must be an integer >= 0")
        if self.entanglement not in ENTANGLEMENTS: bad.append(f"entanglement must be one of {ENTANGLEMENTS}")
        if bad:
            raise ValueError("Invalid quantum circuit config: " + "; ".join(bad))


# ==========================================================================
# Backends
# ==========================================================================
def _h():
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)


def _ry(a: float):
    c, s = math.cos(a / 2.0), math.sin(a / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(a: float):
    return np.array([[np.exp(-0.5j * a), 0], [0, np.exp(0.5j * a)]], dtype=np.complex128)


class QuantumBackend(ABC):
    name = "abstract"
    framework = "abstract"
    framework_version = ""

    @abstractmethod
    def probabilities(self, n_qubits: int, ops: Sequence[Op]) -> np.ndarray:
        """Exact measurement distribution, shape (2**n,), index = bitstring with qubit 0 as MSB."""

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "framework": self.framework, "framework_version": self.framework_version}


class NumpyStatevectorBackend(QuantumBackend):
    name = "numpy_statevector"
    framework = "numpy (built-in reference state-vector simulator)"

    def __init__(self):
        self.framework_version = np.__version__

    def probabilities(self, n_qubits: int, ops: Sequence[Op]) -> np.ndarray:
        n = int(n_qubits)
        psi = np.zeros((2,) * n, dtype=np.complex128)
        psi[(0,) * n] = 1.0
        for name, wires, angle in ops:
            if name == "CNOT":
                c, t = wires
                if c == t:
                    raise ValueError("CNOT control == target")
                idx = [slice(None)] * n
                idx[c] = 1
                sub = psi[tuple(idx)]                       # a view; axis c removed
                ta = t if t < c else t - 1
                psi[tuple(idx)] = np.flip(sub, axis=ta).copy()
                continue
            if name == "H":
                U = _h()
            elif name == "RY":
                U = _ry(float(angle))
            elif name == "RZ":
                U = _rz(float(angle))
            else:
                raise ValueError(f"unknown gate {name!r}")
            q = wires[0]
            psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
        p = (psi.real ** 2 + psi.imag ** 2).reshape(-1)
        s = float(p.sum())
        if not (np.isfinite(s) and abs(s - 1.0) < 1e-9):
            raise FloatingPointError(f"state vector lost normalisation (norm^2 = {s})")
        return p


class PennyLaneBackend(QuantumBackend):
    name = "pennylane"
    framework = "pennylane (default.qubit, analytic probabilities)"

    def __init__(self):
        try:
            import pennylane as qml
        except Exception as e:                              # ImportError or a broken install
            raise QuantumBackendUnavailable(f"pennylane could not be imported: {type(e).__name__}: {e}") from e
        self._qml = qml
        self.framework_version = getattr(qml, "__version__", "unknown")
        self._cache: Dict[Any, Any] = {}

    def _qnode(self, n: int, structure: Tuple[Tuple[str, Tuple[int, ...]], ...]):
        key = (n, structure)
        if key not in self._cache:
            qml = self._qml
            dev = qml.device("default.qubit", wires=n)

            @qml.qnode(dev)
            def circuit(angles):
                k = 0
                for name, wires in structure:
                    if name == "H":
                        qml.Hadamard(wires=wires[0])
                    elif name == "RY":
                        qml.RY(angles[k], wires=wires[0]); k += 1
                    elif name == "RZ":
                        qml.RZ(angles[k], wires=wires[0]); k += 1
                    elif name == "CNOT":
                        qml.CNOT(wires=list(wires))
                    else:
                        raise ValueError(f"unknown gate {name!r}")
                return qml.probs(wires=list(range(n)))

            self._cache[key] = circuit
        return self._cache[key]

    def probabilities(self, n_qubits: int, ops: Sequence[Op]) -> np.ndarray:
        structure = tuple((name, tuple(w)) for name, w, _ in ops)
        angles = np.array([a for name, _, a in ops if a is not None], dtype=np.float64)
        return np.asarray(self._qnode(int(n_qubits), structure)(angles), dtype=np.float64).reshape(-1)


def pennylane_available() -> Tuple[bool, str]:
    try:
        import pennylane  # noqa: F401
        return True, getattr(pennylane, "__version__", "unknown")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def reference_self_check(backend: QuantumBackend, n_qubits: int = 3, tol: float = 1e-8) -> float:
    """Max |p_backend - p_numpy| on a fixed, deterministic circuit. Raises if it exceeds `tol`."""
    cfg = QuantumCircuitConfig(backend="numpy_statevector", n_qubits=n_qubits, depth=2, shots=1, seed=7)
    ref = VariationalCircuit(cfg, NumpyStatevectorBackend())
    theta = np.random.default_rng(123).uniform(-math.pi, math.pi, ref.n_params)
    enc = np.linspace(-0.6, 0.9, n_qubits)
    ops = ref.ops(theta, enc)
    dev = float(np.abs(backend.probabilities(n_qubits, ops) - NumpyStatevectorBackend().probabilities(n_qubits, ops)).max())
    if not dev <= tol:
        raise AssertionError(f"backend {backend.name!r} disagrees with the NumPy reference simulator (max dev {dev:.3e})")
    return dev


def make_backend(requested: str = "auto") -> Tuple[QuantumBackend, List[str]]:
    """Returns (backend, notes). `notes` are human-readable messages (e.g. why 'auto' fell back)."""
    if requested not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if requested == "numpy_statevector":
        return NumpyStatevectorBackend(), []
    if requested == "pennylane":
        b = PennyLaneBackend()                                # raises QuantumBackendUnavailable
        reference_self_check(b)                               # raises on disagreement
        return b, []
    try:
        b = PennyLaneBackend()
        reference_self_check(b)
        return b, []
    except Exception as e:
        return NumpyStatevectorBackend(), [f"backend 'auto': PennyLane unavailable or failed its self-check "
                                           f"({type(e).__name__}: {e}); using the built-in NumPy state-vector simulator"]


# ==========================================================================
# Variational circuit
# ==========================================================================
class VariationalCircuit:
    def __init__(self, cfg: QuantumCircuitConfig, backend: Optional[QuantumBackend] = None):
        cfg.validate()
        self.cfg = cfg
        self.n, self.depth = int(cfg.n_qubits), int(cfg.depth)
        self.backend = backend if backend is not None else make_backend(cfg.backend)[0]
        self.n_params = self.depth * 2 * self.n + self.n
        self.executions = 0                  # circuit executions (exact-probability evaluations)
        self.shots_sampled = 0
        self.seconds = 0.0

    # ---- structure --------------------------------------------------------------------------
    def entangling_pairs(self) -> List[Tuple[int, int]]:
        n = self.n
        if n < 2:
            return []
        if self.cfg.entanglement == "linear":
            return [(q, q + 1) for q in range(n - 1)]
        if n == 2:
            return [(0, 1)]
        return [(q, (q + 1) % n) for q in range(n)]

    def initial_params(self) -> np.ndarray:
        """Deterministic small random start (std 0.1) from (seed, stream)."""
        rng = np.random.default_rng(np.random.SeedSequence([int(self.cfg.seed), STREAM_Q_PARAM_INIT]))
        return 0.1 * rng.standard_normal(self.n_params)

    def ops(self, theta: Sequence[float], enc_angles: Optional[Sequence[float]] = None) -> List[Op]:
        th = np.asarray(theta, dtype=np.float64).reshape(-1)
        if th.shape[0] != self.n_params or not np.isfinite(th).all():
            raise ValueError(f"theta must be {self.n_params} finite numbers")
        enc = np.zeros(self.n) if enc_angles is None else np.asarray(enc_angles, dtype=np.float64).reshape(-1)
        if enc.shape[0] != self.n or not np.isfinite(enc).all():
            raise ValueError(f"enc_angles must be {self.n} finite numbers")
        ops: List[Op] = [("H", (q,), None) for q in range(self.n)]
        ops += [("RY", (q,), float(enc[q])) for q in range(self.n)]
        k = 0
        for _ in range(self.depth):
            for q in range(self.n):
                ops.append(("RY", (q,), float(th[k]))); ops.append(("RZ", (q,), float(th[k + 1]))); k += 2
            ops += [("CNOT", pair, None) for pair in self.entangling_pairs()]
        for q in range(self.n):
            ops.append(("RY", (q,), float(th[k]))); k += 1
        return ops

    def describe(self) -> Dict[str, Any]:
        return {"n_qubits": self.n, "depth": self.depth, "n_params": self.n_params, "shots": int(self.cfg.shots),
                "entanglement": self.cfg.entanglement, "n_entangling_gates_per_layer": len(self.entangling_pairs()),
                **self.backend.describe()}

    # ---- execution -----------------------------------------------------------------------------
    def probabilities(self, theta, enc_angles=None) -> np.ndarray:
        t0 = time.perf_counter()
        p = np.asarray(self.backend.probabilities(self.n, self.ops(theta, enc_angles)), dtype=np.float64)
        self.seconds += time.perf_counter() - t0
        self.executions += 1
        if p.shape != (2 ** self.n,) or not np.isfinite(p).all():
            raise FloatingPointError("backend returned an invalid distribution")
        return p

    def sample_counts(self, probs: np.ndarray, rng: np.random.Generator, shots: Optional[int] = None) -> np.ndarray:
        """Multinomial measurement counts, shape (2**n,), sum == shots (ideal, noise-free sampling)."""
        s = int(self.cfg.shots if shots is None else shots)
        p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
        p = p / p.sum()
        self.shots_sampled += s
        return rng.multinomial(s, p)

    def counts_to_bitstrings(self, counts: np.ndarray) -> np.ndarray:
        """(shots, n) 0/1 matrix, ordered by basis index (a permutation-free, deterministic expansion)."""
        idx = np.repeat(np.arange(len(counts)), counts)
        return ((idx[:, None] >> (self.n - 1 - np.arange(self.n))) & 1).astype(np.int8)

    # ---- cost / gradient -------------------------------------------------------------------------
    def expectation(self, theta, enc_angles, energies: np.ndarray) -> float:
        e = np.asarray(energies, dtype=np.float64)
        if e.shape != (2 ** self.n,):
            raise ValueError("energies must have one entry per basis state")
        return float(self.probabilities(theta, enc_angles) @ e)

    def gradient(self, theta, enc_angles, energies: np.ndarray) -> np.ndarray:
        """Exact parameter-shift gradient (each parameter drives one RY/RZ gate with a Pauli generator)."""
        th = np.asarray(theta, dtype=np.float64).copy()
        g = np.zeros(self.n_params)
        for i in range(self.n_params):
            old = th[i]
            th[i] = old + math.pi / 2; fp = self.expectation(th, enc_angles, energies)
            th[i] = old - math.pi / 2; fm = self.expectation(th, enc_angles, energies)
            th[i] = old
            g[i] = 0.5 * (fp - fm)
        return g


class VQAOptimizer:
    """Standard variational-quantum-algorithm outer loop: Adam on the CIRCUIT parameters using the
    parameter-shift gradient of the diagonal cost. It tunes theta only; it never picks candidates."""

    def __init__(self, lr: float = 0.2, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = float(lr), beta1, beta2, eps

    def train(self, circuit: VariationalCircuit, theta0, enc_angles, energies, steps: int) -> Tuple[np.ndarray, List[float]]:
        th = np.asarray(theta0, dtype=np.float64).copy()
        m, v = np.zeros_like(th), np.zeros_like(th)
        hist = [circuit.expectation(th, enc_angles, energies)]
        for t in range(1, int(steps) + 1):
            g = circuit.gradient(th, enc_angles, energies)
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g * g
            th -= self.lr * (m / (1 - self.b1 ** t)) / (np.sqrt(v / (1 - self.b2 ** t)) + self.eps)
            hist.append(circuit.expectation(th, enc_angles, energies))
        return th, hist
