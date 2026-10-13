"""Phase 7 tests: hybrid classical-quantum search (mechanics only).

Tiny synthetic fixtures (6 features; one 77-feature fixture for representation shape checks). No
CIC-IDS2017, no XGBoost, no performance comparison. Assertions check DETERMINISM, ACCOUNTING, BOUNDS,
LABEL-/TEST-FREENESS and SCHEMA COMPATIBILITY; none is research evidence.

Quantum backend: tests run on the built-in NumPy state-vector simulator (a real circuit simulator, always
available). Tests that need PennyLane are SKIPPED with the exact reason when it is not installed; they are
never faked.
"""
import dataclasses
import json
import math
import re
import unittest
from pathlib import Path

import numpy as np

from src import quantum_circuit as qc_mod, quantum_encoding as qe_mod, quantum_search as qs_mod
from src.candidate import Candidate
from src.experiment_runner import METHODS, make_engine, run_grid, run_search_method
from src.failure_tracker import FailureTracker
from src.feedback_manager import BudgetAccountingError, FeedbackManager
from src.oracle import ReferenceData
from src.quantum_circuit import (NumpyStatevectorBackend, QuantumBackendUnavailable, QuantumCircuitConfig,
                                 VariationalCircuit, VQAOptimizer, make_backend, pennylane_available)
from src.quantum_encoding import LatentCellEncoding
from src.quantum_search import KernelSurrogate, QuantumSearch, QuantumSearchConfig, allocate_by_counts
from src.random_search import RandomSearch, RandomSearchConfig
from src.bayesian_search import BayesianSearch, BayesianSearchConfig
from src.cmaes_search import CMAESConfig, CMAESSearch
from src.search_base import CommonSearchConfig, SearchResult
from src.search_representation import PCARepresentation
from src.search_score import SearchScore, SearchScoreConfig
from src.search_space import SearchSpace, SearchSpaceConfig
from src.target_model import QueryBudgetedModel
from src.utils import load_config
from src.validator import FailureValidator, ValidatorConfig

D = 6
NAMES = tuple(f"f{i}" for i in range(D))
NP = "numpy_statevector"


# ------------------------------------------------------------------ fixtures
def make_ref(n=400) -> ReferenceData:
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 0.5, (n, D)); X0[:, 0] -= 1.0
    X1 = rng.normal(0, 0.5, (n, D)); X1[:, 0] += 1.0
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    return ReferenceData(np.vstack([X0, X1]), y, None, NAMES, ("BENIGN", "ATTACK"), "phase7-fixture")


def vcfg(**over) -> ValidatorConfig:
    return dataclasses.replace(ValidatorConfig(calibration_sample_size=200, knn_k=10), **over)


class Backend:
    """Counting target backend (decision boundary x0 = 0.4 differs from the reference labelling x0 = 0)."""
    def __init__(self):
        self.rows = self.calls = 0

    def predict_proba(self, X):
        self.rows += len(X); self.calls += 1
        p1 = 1 / (1 + np.exp(-4 * (X[:, 0] - 0.4)))
        return np.column_stack([1 - p1, p1])


class FakeTarget:
    kind, is_dev_mock = "dev-mock", True

    def __init__(self):
        self.backend = Backend()

    def black_box(self, budget, name=None):
        return QueryBudgetedModel(self.backend, D, 2, budget, name=name or "fake")


def qcfg(**over) -> QuantumSearchConfig:
    base = dict(backend=NP, latent_dimension=3, n_qubits=4, depth=2, shots=64, opt_steps=3, batch_size=4)
    base.update(over)
    return QuantumSearchConfig(**base)


class Env:
    def __init__(self, radius=1.5, cap_factor=50, policy="skip", validator_over=None):
        self.ref = make_ref()
        self.validator = FailureValidator.from_reference(self.ref, vcfg(**(validator_over or {})))
        sc = dataclasses.replace(SearchSpaceConfig(), radius_enabled=radius is not None, radius_max=radius or 3.0)
        self.space = SearchSpace.from_reference(self.ref.X, NAMES, sc)
        self.space_cfg = sc
        self.common = CommonSearchConfig(max_candidates_per_query=cap_factor, duplicate_policy=policy)

    def box(self, budget):
        b = Backend()
        return QueryBudgetedModel(b, D, 2, budget), b

    def engine(self, budget, seed=1, validator=None, **mcfg):
        return QuantumSearch(self.space, validator or self.validator, budget, seed, self.common, qcfg(**mcfg))

    def run(self, budget, seed=1, **mcfg):
        bb, backend = self.box(budget)
        eng = self.engine(budget, seed, **mcfg)
        return eng.run(bb), bb, backend, eng

    def classical(self, method, budget, seed=1):
        if method == "random":
            return RandomSearch(self.space, self.validator, budget, seed, self.common, RandomSearchConfig(), space_config=self.space_cfg)
        if method == "bayesian":
            return BayesianSearch(self.space, self.validator, budget, seed, self.common, BayesianSearchConfig(n_components=3))
        return CMAESSearch(self.space, self.validator, budget, seed, self.common, CMAESConfig())


def strip_runtime(d):
    """Wall-clock timings are the only non-deterministic content of a quantum result."""
    d = json.loads(json.dumps(d))
    d["reproducibility"]["quantum"].pop("runtime", None)
    return d


def _code(mod):
    text = Path(mod.__file__).read_text()
    code = re.sub(r'"""(?:.|\n)*?"""', "", text)
    return re.sub(r"#.*", "", code)


# ==================================================================== 1-3, 6. circuit construction
def test_circuit_qubit_count_and_parameter_count():
    for n in (1, 2, 3, 4, 6):
        c = VariationalCircuit(QuantumCircuitConfig(backend=NP, n_qubits=n, depth=2))
        assert c.n == n and c.n_params == 2 * 2 * n + n
        ops = c.ops(c.initial_params(), np.zeros(n))
        used = {w for _, ws, _ in ops for w in ws}
        assert used == set(range(n))                                         # every qubit is used, none beyond n
        assert len(c.probabilities(c.initial_params(), np.zeros(n))) == 2 ** n


def test_circuit_depth_configuration_controls_gate_counts():
    for n in (2, 3, 4, 5):
        for depth in (1, 2, 3, 4):
            c = VariationalCircuit(QuantumCircuitConfig(backend=NP, n_qubits=n, depth=depth))
            ops = c.ops(c.initial_params(), np.zeros(n))
            cnt = lambda g: sum(1 for o in ops if o[0] == g)
            n_ent = 1 if n == 2 else n                                       # ring
            assert cnt("H") == n and cnt("CNOT") == depth * n_ent
            assert cnt("RZ") == depth * n and cnt("RY") == n + depth * n + n  # encoding + layers + final
            assert c.n_params == depth * 2 * n + n
    lin = VariationalCircuit(QuantumCircuitConfig(backend=NP, n_qubits=4, depth=2, entanglement="linear"))
    assert lin.entangling_pairs() == [(0, 1), (1, 2), (2, 3)]
    ring = VariationalCircuit(QuantumCircuitConfig(backend=NP, n_qubits=4, depth=2))
    assert ring.entangling_pairs() == [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_circuit_config_validation():
    for bad in (dict(n_qubits=0), dict(n_qubits=13), dict(depth=0), dict(shots=0), dict(seed=-1),
                dict(backend="ibm"), dict(entanglement="star"), dict(n_qubits=True)):
        try:
            QuantumCircuitConfig(**bad).validate(); assert False, bad
        except ValueError:
            pass
    QuantumCircuitConfig().validate()
    d = QuantumCircuitConfig()
    assert (d.n_qubits, d.depth, d.shots) == (4, 2, 256)                     # development defaults


def test_simulator_is_a_correct_circuit_simulator():
    b = NumpyStatevectorBackend()
    assert np.allclose(b.probabilities(2, [("H", (0,), None), ("CNOT", (0, 1), None)]), [.5, 0, 0, .5])      # Bell
    assert np.allclose(b.probabilities(2, [("RY", (0,), math.pi)]), [0, 0, 1, 0])                           # qubit 0 = MSB
    assert np.allclose(b.probabilities(2, [("RY", (1,), math.pi)]), [0, 1, 0, 0])
    p = b.probabilities(3, [("RY", (2,), math.pi), ("CNOT", (2, 0), None)])                                  # |001> -> |101>
    assert np.allclose(p, np.eye(8)[5])
    assert np.allclose(b.probabilities(1, [("H", (0,), None), ("RZ", (0,), 1.234), ("H", (0,), None)]),
                       [math.cos(1.234 / 2) ** 2, math.sin(1.234 / 2) ** 2])                                 # interference
    c = VariationalCircuit(QuantumCircuitConfig(backend=NP, n_qubits=4, depth=3))
    th = np.random.default_rng(0).uniform(-3, 3, c.n_params)
    p = c.probabilities(th, np.linspace(-1, 1, 4))
    assert abs(p.sum() - 1) < 1e-12 and (p >= 0).all()


def test_entangling_layers_create_entanglement():
    b = NumpyStatevectorBackend()
    p = b.probabilities(2, [("RY", (0,), math.pi / 2), ("CNOT", (0, 1), None)])        # correlated bits
    assert np.allclose(p, [.5, 0, 0, .5])
    q = b.probabilities(2, [("RY", (0,), math.pi / 2)])                                 # product state
    assert np.allclose(q, [.5, 0, .5, 0])


def test_parameter_shift_gradient_matches_finite_differences_and_training_lowers_cost():
    c = VariationalCircuit(QuantumCircuitConfig(backend=NP, n_qubits=4, depth=2, seed=3))
    th, enc = c.initial_params() + 0.3, np.array([.1, -.2, .3, .4])
    e = np.random.default_rng(1).random(16)
    g = c.gradient(th, enc, e)
    for i in (0, 5, 11, 19):
        h = 1e-6
        tp, tm = th.copy(), th.copy(); tp[i] += h; tm[i] -= h
        fd = (c.expectation(tp, enc, e) - c.expectation(tm, enc, e)) / (2 * h)
        assert abs(g[i] - fd) < 1e-6
    th2, hist = VQAOptimizer(0.2).train(c, th, enc, e, 15)
    assert hist[-1] < hist[0] and len(hist) == 16 and th2.shape == th.shape


# ==================================================================== 4. deterministic seeds
def test_circuit_is_deterministic_per_seed():
    mk = lambda s: VariationalCircuit(QuantumCircuitConfig(backend=NP, seed=s))
    assert np.array_equal(mk(5).initial_params(), mk(5).initial_params())
    assert not np.array_equal(mk(5).initial_params(), mk(6).initial_params())
    c = mk(5)
    th, enc = c.initial_params(), np.zeros(4)
    assert np.array_equal(c.probabilities(th, enc), c.probabilities(th, enc))
    p = c.probabilities(th, enc)
    a = c.sample_counts(p, np.random.default_rng(9)); b = c.sample_counts(p, np.random.default_rng(9))
    d = c.sample_counts(p, np.random.default_rng(10))
    assert np.array_equal(a, b) and not np.array_equal(a, d)


# ==================================================================== 5. deterministic latent encoding
def test_latent_encoding_is_deterministic_bounded_and_documented():
    enc = LatentCellEncoding(3, 4, 2.0)
    c = np.array([-5.0, 0.3, 1.9])
    a1, a2 = enc.angles(c), enc.angles(c.copy())
    assert np.array_equal(a1, a2) and a1.shape == (4,)
    assert (np.abs(a1) <= math.pi / 2 + 1e-12).all() and np.isfinite(a1).all()   # clipped: numerically stable for outliers
    assert abs(enc.angles(np.array([2.0, 2.0, 2.0]))[0] - math.pi / 2) < 1e-12
    assert np.array_equal(enc.angles(np.zeros(3)), np.zeros(4))
    assert enc.axis_of_qubit.tolist() == [0, 1, 2, 0] and enc.bits_per_axis.tolist() == [2, 1, 1]
    assert "angle_map" in enc.describe()
    for bad in (lambda: LatentCellEncoding(5, 4, 2.0), lambda: LatentCellEncoding(2, 4, 0.0), lambda: enc.angles(np.zeros(2))):
        try:
            bad(); assert False
        except ValueError:
            pass


def test_cells_tile_the_region_and_bitstrings_map_bijectively():
    enc = LatentCellEncoding(3, 5, 2.0)
    lo, hi = enc.region(np.array([0.5, -1.5, 1.9]), 0.8)
    assert (lo >= -2).all() and (hi <= 2).all() and (hi > lo).all()
    cl, cu = enc.cell_bounds(lo, hi)
    assert cl.shape == (32, 3) and (cl >= lo - 1e-12).all() and (cu <= hi + 1e-12).all()
    vol = ((cu - cl).prod(axis=1)).sum()
    assert abs(vol - (hi - lo).prod()) < 1e-9                                  # cells tile the region exactly
    assert len({tuple(r) for r in np.round(cl, 9)}) == 32                      # distinct cells
    pts = enc.sample_in_cells(np.arange(32), lo, hi, np.random.default_rng(0))
    assert ((pts >= cl) & (pts <= cu)).all()
    assert np.array_equal(pts, enc.sample_in_cells(np.arange(32), lo, hi, np.random.default_rng(0)))


# ==================================================================== 6. measurement output shape
def test_measurement_output_shape():
    c = VariationalCircuit(QuantumCircuitConfig(backend=NP, n_qubits=4, shots=100, seed=1))
    p = c.probabilities(c.initial_params(), np.zeros(4))
    counts = c.sample_counts(p, np.random.default_rng(0))
    assert counts.shape == (16,) and counts.sum() == 100 and (counts >= 0).all()
    bits = c.counts_to_bitstrings(counts)
    assert bits.shape == (100, 4) and set(np.unique(bits)) <= {0, 1}
    assert c.shots_sampled == 100
    assert c.sample_counts(p, np.random.default_rng(0), shots=7).sum() == 7


# ==================================================================== 7-9. latent -> original space
def test_latent_to_original_space_77_features_finite_and_inside_bounds():
    rng = np.random.default_rng(0)
    d = 77
    A = rng.normal(size=(d, 9)); X = rng.normal(size=(1500, 9)) @ A.T + 0.1 * rng.normal(size=(1500, d))
    names = tuple(f"c{i}" for i in range(d))
    space = SearchSpace.from_reference(X, names, SearchSpaceConfig())
    rep = PCARepresentation(space, X, n_components=4, latent_bound=2.0)         # TRAINING features only
    assert rep.dim == 4
    enc = LatentCellEncoding(4, 4, 2.0)
    lo, hi = enc.region(np.zeros(4), 1.0)
    V = enc.sample_in_cells(np.arange(16), lo, hi, np.random.default_rng(1))
    Xc = rep.decode(V)
    assert Xc.shape == (16, 77) and np.isfinite(Xc).all()
    assert (Xc >= space.lower - 1e-9).all() and (Xc <= space.upper + 1e-9).all()
    Vw = rep.decode(np.full((2, 4), 99.0))                                     # wildly out-of-box latent: clipped + projected
    assert np.isfinite(Vw).all() and space.contains(Vw).all()
    assert np.allclose(rep.encode(Xc), rep.encode(rep.decode(rep.encode(Xc))), atol=1e-8)


def test_reconstructed_candidates_are_finite_and_in_bounds_during_a_run():
    for radius in (None, 1.5):
        e = Env(radius=radius)
        seen = []

        class Rec(FailureValidator):
            def validate_batch(self, candidates, black_box, seed=0):
                seen.extend(candidates)
                return super().validate_batch(candidates, black_box, seed)

        v = Rec(e.validator.oracle, e.validator.cfg)
        bb, _ = e.box(25)
        r = e.engine(25, 2, validator=v).run(bb)
        X = np.stack([c.x for c in seen])
        assert len(X) == r.candidates_evaluated and np.isfinite(X).all()
        assert (X >= e.space.lower - 1e-9).all() and (X <= e.space.upper + 1e-9).all()
        assert e.space.contains(X).all()
        assert all(c.check(NAMES) == "ok" for c in seen)


# ==================================================================== 10. duplicates
def test_duplicate_handling_never_reevaluates_and_falls_back_when_all_seen():
    e = Env()
    eng = e.engine(30, 1)
    orig = eng.enc.sample_in_cells
    eng.enc.sample_in_cells = lambda cells, lo, hi, rng: np.zeros((len(cells), eng.rep.dim))   # every sample identical
    bb, backend = e.box(30)
    r = eng.run(bb)
    assert len(set(r.candidate_ids)) == len(r.candidate_ids) and r.duplicates_skipped == 0
    assert eng.duplicates_dropped > 0 and eng.fallbacks > 0
    assert r.queries_used == backend.rows <= 30
    eng.enc.sample_in_cells = orig
    assert r.reproducibility["quantum"]["duplicates_dropped_before_evaluation"] == eng.duplicates_dropped


def test_allocate_by_counts_is_exact_and_deterministic():
    counts = np.array([0, 50, 30, 20, 0, 0, 0, 0])
    for n in (1, 3, 8, 10, 33):
        a = allocate_by_counts(counts, n)
        assert len(a) == n and np.array_equal(a, allocate_by_counts(counts.copy(), n))
        assert set(a.tolist()) <= {1, 2, 3}
    assert allocate_by_counts(counts, 10).tolist() == [1] * 5 + [2] * 3 + [3] * 2
    assert allocate_by_counts(np.zeros(4, int), 3).size == 0 and allocate_by_counts(counts, 0).size == 0


# ==================================================================== 11-13. query budget
def test_query_budget_enforced_exactly():
    e = Env()
    for budget in (1, 7, 20, 33):
        r, bb, backend, _ = e.run(budget, seed=1)
        assert r.queries_used == bb.n_queries == backend.rows <= budget, budget
        assert r.black_box["budget"] == budget and r.query_budget == budget
        if r.termination_reason == "budget_exhausted":
            assert r.queries_used == budget
        assert r.termination_reason in ("budget_exhausted", "candidate_cap_reached", "max_iterations_reached")


def test_initial_design_is_counted_and_never_free():
    e = Env()
    eng = e.engine(20, 3, n_initial=7)
    V = eng.initial_design(7)
    assert V.shape == (7, 3) and (np.abs(V) <= 2.0).all() and np.array_equal(V[:4], eng.initial_design(4))
    assert np.array_equal(V, e.engine(20, 3, n_initial=7).initial_design(7))
    assert eng.initial_design(0).shape == (0, 3)
    bb, backend = e.box(20)
    r = eng.run(bb)
    expected = [c.candidate_id for c in eng._candidates(V, "quantum_init", [{}] * 7)]
    assert r.candidate_ids[:7] == expected and backend.rows == r.queries_used == 20
    assert e.engine(100)._n_initial() == 10 and e.engine(3)._n_initial() == 3     # never above the budget


def test_reference_guided_initial_design_is_deterministic_and_bounded():
    e = Env()
    eng = e.engine(20, seed=3, n_initial=7, initialization="reference_guided")
    V = eng.initial_design(7)
    assert V.shape == (7, 3)
    assert np.isfinite(V).all() and (np.abs(V) <= eng.cfg.latent_bound).all()
    assert np.array_equal(V, e.engine(
        20, seed=3, n_initial=7, initialization="reference_guided"
    ).initial_design(7))
    assert not np.array_equal(V, e.engine(20, seed=3, n_initial=7).initial_design(7))


def test_budget_cannot_be_reset_or_reused():
    e = Env()
    bb, _ = e.box(10)
    e.engine(10).run(bb)
    assert bb.n_queries == 10
    for bad in (lambda: e.engine(10).run(bb),                                   # used box
                lambda: e.engine(10).run(e.box(11)[0]),                         # budget mismatch
                lambda: e.engine(10).run(QueryBudgetedModel(Backend(), D, 2, None))):    # unbudgeted
        try:
            bad(); assert False
        except ValueError:
            pass
    eng = e.engine(10)
    eng.run(e.box(10)[0])
    try:
        eng.run(e.box(10)[0]); assert False
    except RuntimeError:
        pass
    assert bb.n_queries == 10


def test_a_budget_reset_during_a_quantum_run_is_detected():
    e = Env()

    class Tamper(Backend):
        bb = None

        def predict_proba(self, X):
            P = super().predict_proba(X)
            if self.calls == 3:
                self.bb._n_queries = 0                                          # simulate a budget reset
            return P

    t = Tamper()
    bb = QueryBudgetedModel(t, D, 2, 40)
    t.bb = bb
    try:
        e.engine(40).run(bb); assert False, "tampering must be detected"
    except BudgetAccountingError:
        pass


def test_all_target_calls_go_through_the_wrapper_and_circuit_runs_are_query_free():
    e = Env()
    eng = e.engine(18, 1)
    bb, backend = e.box(18)
    for _ in range(25):                                                          # circuit work alone: zero queries
        eng.circuit.probabilities(eng.circuit.initial_params(), np.zeros(4))
    eng.initial_design(5)                                                        # initial design is query-free too
    assert backend.rows == backend.calls == 0 and bb.n_queries == 0
    r = eng.run(bb)
    assert backend.rows == bb.n_queries == r.queries_used and backend.calls == bb.n_calls
    q = r.reproducibility["quantum"]
    assert q["cost"]["circuit_executions"] > 0 and q["cost"]["target_queries_charged"] == 0
    assert r.queries_used <= 18                                                  # circuit executions are not queries


def test_quantum_cost_is_recorded_separately_and_consistent():
    r, bb, backend, eng = Env().run(25, seed=2, opt_steps=2)
    q = r.reproducibility["quantum"]
    n_params = q["n_params"]
    # per iteration: (opt_steps * (2*n_params + 1)) gradient+cost evals + 1 initial expectation + 1 sampling distribution
    expected = q["iterations"] * (1 + 2 * (2 * n_params + 1) + 1) if q["iterations"] else 0
    assert q["cost"]["circuit_executions"] == expected
    assert q["cost"]["shots_sampled"] == q["iterations"] * 64 and q["cost"]["optimisation_steps"] == q["iterations"] * 2
    assert set(q["runtime"]) == {"search_seconds", "circuit_seconds", "quantum_iteration_seconds"}


# ==================================================================== 14-15. no labels / no test data
class BlindRef:
    """Reference view that raises if anything reads labels/categories through `validator.ref`."""
    def __init__(self, ref):
        object.__setattr__(self, "_r", ref)

    def __getattr__(self, name):
        if name in ("y", "categories"):
            raise AssertionError(f"label data accessed through validator.ref.{name}")
        return getattr(self._r, name)


def test_quantum_search_never_reads_reference_labels():
    e = Env()

    class Blind(FailureValidator):
        def __init__(self, base):
            super().__init__(base.oracle, base.cfg)
            self.ref = BlindRef(self.ref)

    v = Blind(e.validator)
    bb, backend = e.box(20)
    r = e.engine(20, 1, validator=v).run(bb)                                    # would raise AssertionError on any label read
    assert r.queries_used == 20
    ref = make_ref(); ref2 = ReferenceData(ref.X, ref.y, np.array(["A"] * len(ref.y)), NAMES, ref.class_names, "x")
    assert ref2.categories is not None                                          # categories exist but are unused: see static test


class PoisonedPrepared:
    def __init__(self, ref):
        self.X_train, self.y_train, self.cat_train = ref.X, ref.y, np.array(["BENIGN"] * len(ref.y))
        self.feature_names, self.class_names = list(ref.feature_names), list(ref.class_names)
        self.provenance = {"source": "poison-test"}

    def __getattr__(self, name):
        if "test" in name:
            raise AssertionError(f"held-out test data accessed: {name}")
        raise AttributeError(name)


def test_no_test_split_accessed_by_quantum_construction_or_run():
    cfg = load_config()
    cfg["validator"]["calibration"]["sample_size"] = 200
    cfg["validator"]["knn"]["k"] = 10
    cfg["search"]["quantum"].update(backend=NP, latent_dimension=3, shots=32, opt_steps=2)
    prep = PoisonedPrepared(make_ref())
    space = SearchSpace.from_prepared(prep, SearchSpaceConfig.from_cfg(cfg))
    validator = FailureValidator.from_prepared(prep, cfg)
    eng = make_engine("quantum", cfg, space, validator, 12, 1)
    r = eng.run(QueryBudgetedModel(Backend(), D, 2, 12))
    assert r.queries_used <= 12 and r.method == "quantum"


def test_static_quantum_modules_have_no_labels_no_internals_no_direct_queries_no_verdict():
    for mod in (qs_mod, qc_mod, qe_mod):
        code = _code(mod)
        assert not re.search(r"\b(X|y|cat)_test\b", code), mod.__name__
        assert not re.search(r"import xgboost|\bTargetModel\b|\.clf\b|booster|predict_proba_unbudgeted", code), mod.__name__
        assert not re.search(r"\.ref\.y\b|\.ref\.categories|\.categories\b|\by_train\b|\bcat_train\b", code), mod.__name__
        assert not re.search(r"\.query\(|\.predict_proba\(|_QueryBudgetedModel", code), mod.__name__
        assert not re.search(r"\bverified_failure\b|\bfailure_reason\b|oracle_label|oracle_confidence", code), mod.__name__
        assert "target_model" not in code, mod.__name__
    for mod in (qc_mod, qe_mod):                                                # circuit/encoding cannot even reach the validator
        assert not re.search(r"from \.(validator|oracle|target_model|feedback_manager|search_base|candidate)\b", _code(mod)), mod.__name__
    assert not re.search(r"\bbb\b|black_box|\.budget\b|n_queries", _code(qc_mod) + _code(qe_mod))


def test_search_does_not_depend_on_the_validator_verdict():
    e = Env()

    class Flipped(FailureValidator):
        def validate_batch(self, candidates, black_box, seed=0):
            return [dataclasses.replace(r, verified_failure=not r.verified_failure, failure_reason="x")
                    for r in super().validate_batch(candidates, black_box, seed)]

    base = e.run(30, seed=4)[0]
    bb, _ = e.box(30)
    flipped = e.engine(30, 4, validator=Flipped(e.validator.oracle, e.validator.cfg)).run(bb)
    assert base.candidate_ids == flipped.candidate_ids                          # trajectory independent of the verdict
    assert base.reproducibility["trajectory_sha256"] == flipped.reproducibility["trajectory_sha256"]


# ==================================================================== 16-17. tiny synthetic search, reproducibility
def test_tiny_synthetic_quantum_search_executes_and_records_everything():
    r, bb, backend, eng = Env().run(30, seed=1)
    assert r.method == "quantum" and r.seed == 1 and r.query_budget == 30 and r.queries_used == 30
    assert r.candidates_generated >= r.candidates_evaluated >= r.candidates_valid > 0
    assert r.verified_failures >= r.unique_verified_failures >= 0
    assert len(r.candidate_ids) == r.candidates_evaluated and r.objective_stats["valid"]["n"] == r.candidates_valid
    assert r.termination_reason == "budget_exhausted" and not r.paper_eligible          # fixture is not a paper context
    assert eng.iterations >= 1 and eng.last_iteration is not None
    it = eng.last_iteration
    assert abs(it["probs"].sum() - 1) < 1e-9 and it["counts"].sum() == 64 and len(it["energies"]) == 16
    q = r.reproducibility["quantum"]
    for k in ("backend", "framework", "framework_version", "n_qubits", "depth", "shots", "latent_dimension", "n_params",
              "cost", "runtime", "circuit_training", "limitations", "encoding", "objective", "classical_surrogate", "circuit_seed"):
        assert k in q, k
    assert (q["backend"], q["n_qubits"], q["depth"], q["shots"], q["latent_dimension"]) == (NP, 4, 2, 64, 3)
    assert q["circuit_training"]["mean_energy_after"] <= q["circuit_training"]["mean_energy_before"] + 1e-9
    assert len(q["limitations"]) >= 4 and any("no quantum advantage" in w for w in r.warnings)


def test_same_seed_reproducible_different_seed_differs():
    e = Env()
    a, b, c = e.run(30, seed=3)[0], e.run(30, seed=3)[0], e.run(30, seed=4)[0]
    assert a.candidate_ids == b.candidate_ids and a.verified_failure_ids == b.verified_failure_ids
    assert a.reproducibility["trajectory_sha256"] == b.reproducibility["trajectory_sha256"]
    assert strip_runtime(a.to_dict()) == strip_runtime(b.to_dict())
    assert a.candidate_ids != c.candidate_ids
    salted = QuantumSearch(e.space, e.validator, 30, 3, e.common, qcfg(seed=7)).run(e.box(30)[0])
    assert salted.reproducibility["quantum"]["circuit_seed"] != a.reproducibility["quantum"]["circuit_seed"]


def test_iteration_cap_terminates_cleanly():
    r, *_ = Env().run(500, seed=1, iterations=2, n_initial=5)
    assert r.termination_reason == "max_iterations_reached" and r.reproducibility["quantum"]["iterations"] == 2
    assert r.queries_used <= 5 + 2 * 4


# ==================================================================== 18-19. schema + runner
def test_quantum_result_schema_matches_phase6_schema():
    e = Env()
    ref_keys = None
    for m in ("random", "bayesian", "cmaes"):
        bb, _ = e.box(15)
        d = e.classical(m, 15).run(bb).to_dict()
        ref_keys = set(d) if ref_keys is None else ref_keys
        assert set(d) == ref_keys, m
    r = e.run(15)[0]
    d = r.to_dict()
    assert set(d) == ref_keys
    assert set(d["reproducibility"]) >= {"method", "seed", "space_id", "reference_fingerprint", "validator_config_hash",
                                           "common_config_hash", "method_config", "objective", "trajectory_sha256",
                                           "duplicate_policy", "max_candidates"}
    assert d["reproducibility"]["method"] == "quantum" and d["reproducibility"]["method_config"]["n_qubits"] == 4
    back = SearchResult.from_dict(json.loads(json.dumps(d)))                     # JSON round trip
    assert back.method == "quantum" and back.candidate_ids == r.candidate_ids
    assert not any("label" in k or "test" in k for k in r.trajectory[0])          # no hidden labels stored


def test_quantum_is_a_supported_runner_method():
    assert tuple(METHODS) == ("random", "bayesian", "cmaes", "quantum")
    cfg = load_config()
    cfg["validator"]["calibration"]["sample_size"] = 200
    cfg["validator"]["knn"]["k"] = 10
    cfg["search"]["bayesian"]["n_components"] = 3
    cfg["search"]["quantum"].update(backend=NP, latent_dimension=3, shots=32, opt_steps=2, batch_size=4)
    cfg["search"]["common"]["max_candidates_per_query"] = 20
    e = Env()
    space = SearchSpace.from_reference(e.ref.X, NAMES, dataclasses.replace(SearchSpaceConfig(), radius_enabled=True, radius_max=1.5))
    validator = FailureValidator.from_reference(e.ref, vcfg())
    eng = make_engine("quantum", cfg, space, validator, 10, 2)
    assert isinstance(eng, QuantumSearch)
    for m in METHODS:                                                            # same call for all four methods
        t = FakeTarget()
        r = run_search_method(m, cfg, space, validator, t, 12, 2)
        assert r.method == m and r.queries_used == t.backend.rows <= 12 and r.query_budget == 12
    t = FakeTarget()
    r = run_search_method("quantum", cfg, space, validator, t, 10, 1)
    ctx = r.reproducibility["run_context"]
    assert ctx["dataset"] == "phase7-fixture"                                    # names only; never labels/test data
    assert ctx["model_under_test"] == {"kind": "dev-mock", "is_dev_mock": True}
    out = run_grid(["quantum"], [8], [1, 2], cfg, space, validator, FakeTarget())
    assert [x.seed for x in out] == [1, 2] and all(x.queries_used <= 8 for x in out)
    try:
        make_engine("annealing", cfg, space, validator, 10, 1); assert False
    except ValueError:
        pass


def test_quantum_config_loads_validates_and_phase4_phase6_config_is_unchanged():
    cfg = load_config()
    q = QuantumSearchConfig.from_cfg(cfg)
    assert (q.n_qubits, q.depth, q.shots, q.latent_dimension, q.backend) == (4, 2, 256, 4, "auto")
    for key in ("backend", "n_qubits", "depth", "shots", "latent_dimension", "initialization", "seed", "iterations"):
        assert key in cfg["search"]["quantum"], key
    v = cfg["validator"]
    assert v["failure_rule"]["min_oracle_confidence"] == 0.8 and v["mahalanobis"]["quantile"] == 0.995
    assert v["nn_gate"]["quantile"] == 0.995 and v["knn"]["k"] == 15 and v["knn"]["min_purity"] == 0.8
    for k in ("random", "bayesian", "cmaes", "common"):
        assert k in cfg["search"]
    for bad in (dict(n_qubits=3, latent_dimension=4), dict(shots=0), dict(batch_size=0), dict(radius_min=0.9, radius_init=0.5),
                dict(radius_shrink=0), dict(backend="ibm"), dict(latent_bound=float("nan")), dict(iterations=0),
                dict(initialization="unknown")):
        try:
            dataclasses.replace(QuantumSearchConfig(), **bad).validate(); assert False, bad
        except ValueError:
            pass


# ==================================================================== surrogate / objective
def test_surrogate_and_cell_objective_use_scores_only():
    s = KernelSurrogate(0.5, 0.2)
    Z = np.array([[0.0, 0.0], [1.0, 1.0]]); y = np.array([0.1, 0.9])
    s.fit(Z, y)
    mu, dens = s.predict(np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]]))
    assert mu[0] < mu[1] and dens[0] == 1.0 and dens[2] < 1e-6 and np.isfinite(mu).all()
    assert s.cell_scores(np.array([[5.0, 5.0]]))[0] > mu[2]                       # unexplored point gets the novelty bonus
    assert np.array_equal(qs_mod._normalised_energies(np.array([3., 3., 3.])), np.zeros(3))
    e = qs_mod._normalised_energies(np.array([1., 2., 3.]))
    assert e.min() == -1.0 and e.max() == 0.0
    empty = KernelSurrogate(0.5, 0.2)
    assert np.array_equal(empty.predict(np.zeros((2, 3)))[0], np.zeros(2))


def test_quantum_objective_is_surrogate_search_score_not_failure_label():
    r, _, _, eng = Env().run(30, seed=1)
    assert "SURROGATE search score" in r.reproducibility["quantum"]["objective"]
    it = eng.last_iteration
    assert (it["energies"] >= -1 - 1e-12).all() and (it["energies"] <= 1e-12).all()
    sc = SearchScore(SearchScoreConfig(), 0.8)                                    # the score object never reads the verdict
    assert "verified_failure" not in _code(qs_mod)
    assert sc.cfg.w_uncertainty + sc.cfg.w_failure_likelihood > 0


# ==================================================================== backends (honest skips)
def test_backend_selection_and_auto_fallback_note():
    b, notes = make_backend("numpy_statevector")
    assert b.name == NP and notes == []
    ok, info = pennylane_available()
    auto, notes = make_backend("auto")
    if ok:
        assert auto.name in ("pennylane", NP)
    else:
        assert auto.name == NP and len(notes) == 1 and "PennyLane unavailable" in notes[0]
        r = QuantumSearch(Env().space, Env().validator, 8, 1, Env().common, qcfg(backend="auto")).run(Env().box(8)[0])
        assert any("PennyLane unavailable" in w for w in r.warnings)
        assert r.reproducibility["quantum"]["backend"] == NP and r.reproducibility["quantum"]["backend_requested"] == "auto"
    try:
        make_backend("ibm"); assert False
    except ValueError:
        pass


def test_explicit_pennylane_backend_fails_loudly_when_missing():
    ok, info = pennylane_available()
    if ok:
        raise unittest.SkipTest("pennylane is installed; this guard is only testable without it")
    try:
        make_backend("pennylane"); assert False
    except QuantumBackendUnavailable as e:
        assert "pennylane" in str(e)
    try:
        QuantumSearch(Env().space, Env().validator, 8, 1, Env().common, qcfg(backend="pennylane"))
        assert False
    except QuantumBackendUnavailable:
        pass


def test_pennylane_backend_agrees_with_numpy_reference():
    ok, info = pennylane_available()
    if not ok:
        raise unittest.SkipTest(f"pennylane not installed ({info}); PennyLane adapter NOT exercised")
    pl = qc_mod.PennyLaneBackend()
    ref = NumpyStatevectorBackend()
    for n, depth, seed in ((2, 1, 0), (3, 2, 1), (4, 2, 2), (5, 3, 3)):
        c = VariationalCircuit(QuantumCircuitConfig(backend=NP, n_qubits=n, depth=depth, seed=seed))
        th = np.random.default_rng(seed).uniform(-3, 3, c.n_params)
        ops = c.ops(th, np.linspace(-1, 1, n))
        assert np.abs(pl.probabilities(n, ops) - ref.probabilities(n, ops)).max() < 1e-8


def test_quantum_search_runs_with_pennylane_backend():
    ok, info = pennylane_available()
    if not ok:
        raise unittest.SkipTest(f"pennylane not installed ({info}); PennyLane search path NOT exercised")
    e = Env()
    r, bb, backend, _ = e.run(15, seed=1, backend="pennylane")
    assert r.reproducibility["quantum"]["backend"] == "pennylane" and r.queries_used == backend.rows <= 15
