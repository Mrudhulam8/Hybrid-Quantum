"""Phase 6 tests: classical search baselines (random / Bayesian-GP / CMA-ES).

Tiny synthetic fixtures only (6 features, no CIC-IDS2017, no XGBoost). Assertions check MECHANICS -
determinism, bounds, budget/query accounting, label-freeness, serialisation - and are NOT research
evidence. No test compares method performance.
"""
import dataclasses
import json
import re
from pathlib import Path

import numpy as np

from src import (bayesian_search as bo_mod, cmaes_search as cma_mod, failure_tracker as ft_mod,
                 feedback_manager as fm_mod, random_search as rs_mod, search_base as sb_mod,
                 search_representation as rep_mod, search_score as sc_mod)
from src.bayesian_search import BayesianSearch, BayesianSearchConfig, expected_improvement, upper_confidence_bound
from src.candidate import Candidate
from src.cmaes_search import CMAESConfig, CMAESSearch, CMAESState
from src.experiment_runner import METHODS, make_engine, run_search_method
from src.failure_tracker import FailureTracker
from src.feedback_manager import BudgetAccountingError, FeedbackManager
from src.oracle import ReferenceData
from src.random_search import RandomSearch, RandomSearchConfig
from src.search_base import CommonSearchConfig, SearchResult
from src.search_representation import PCARepresentation, StandardizedFullRepresentation
from src.search_score import SearchScore, SearchScoreConfig
from src.search_space import SearchSpace, SearchSpaceConfig
from src.target_model import QueryBudgetedModel
from src.utils import load_config
from src.validator import FailureValidator, ValidatorConfig

D = 6
NAMES = tuple(f"f{i}" for i in range(D))
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ fixtures
def make_ref(n=400) -> ReferenceData:
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 0.5, (n, D)); X0[:, 0] -= 1.0
    X1 = rng.normal(0, 0.5, (n, D)); X1[:, 0] += 1.0
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    return ReferenceData(np.vstack([X0, X1]), y, None, NAMES, ("BENIGN", "ATTACK"), "phase6-fixture")


def vcfg(**over) -> ValidatorConfig:
    return dataclasses.replace(ValidatorConfig(calibration_sample_size=200, knn_k=10), **over)


class Backend:
    """Counting target backend; its decision boundary (x0 = 0.4) differs from the reference labelling (x0 = 0)."""
    def __init__(self):
        self.rows = self.calls = 0

    def predict_proba(self, X):
        self.rows += len(X); self.calls += 1
        p1 = 1 / (1 + np.exp(-4 * (X[:, 0] - 0.4)))
        return np.column_stack([1 - p1, p1])


class Env:
    """Shared components for one test."""
    def __init__(self, radius=None, cap_factor=50, validator_over=None, objective=None, policy="skip"):
        self.ref = make_ref()
        self.validator = FailureValidator.from_reference(self.ref, vcfg(**(validator_over or {})))
        sc = dataclasses.replace(SearchSpaceConfig(), radius_enabled=radius is not None, radius_max=radius or 3.0)
        self.space = SearchSpace.from_reference(self.ref.X, NAMES, sc)
        self.space_cfg = sc
        self.common = CommonSearchConfig(max_candidates_per_query=cap_factor, duplicate_policy=policy,
                                         objective=objective or SearchScoreConfig())

    def box(self, budget):
        b = Backend()
        return QueryBudgetedModel(b, D, 2, budget), b

    def engine(self, method, budget, seed=1, **mcfg):
        if method == "random":
            return RandomSearch(self.space, self.validator, budget, seed, self.common, RandomSearchConfig(**mcfg),
                                space_config=self.space_cfg)
        if method == "bayesian":
            mcfg.setdefault("n_components", 3)
            return BayesianSearch(self.space, self.validator, budget, seed, self.common, BayesianSearchConfig(**mcfg))
        return CMAESSearch(self.space, self.validator, budget, seed, self.common, CMAESConfig(**mcfg))

    def run(self, method, budget, seed=1, **mcfg):
        bb, backend = self.box(budget)
        eng = self.engine(method, budget, seed, **mcfg)
        return eng.run(bb), bb, backend, eng


ALL = ("random", "bayesian", "cmaes")


class RecordingValidator(FailureValidator):
    """Records every candidate that reaches the validator (valid or not)."""
    def __init__(self, base: FailureValidator):
        super().__init__(base.oracle, base.cfg)
        self.seen = []

    def validate_batch(self, candidates, black_box, seed=0):
        self.seen.extend(candidates)
        return super().validate_batch(candidates, black_box, seed)


# ------------------------------------------------------------------ determinism / reproducibility
def test_random_search_deterministic_same_seed_and_different_seeds_differ():
    e = Env()
    a, _, _, _ = e.run("random", 12, seed=3)
    b, _, _, _ = e.run("random", 12, seed=3)
    c, _, _, _ = e.run("random", 12, seed=4)
    assert a.candidate_ids == b.candidate_ids and a.verified_failure_ids == b.verified_failure_ids
    assert a.reproducibility["trajectory_sha256"] == b.reproducibility["trajectory_sha256"]
    assert a.candidate_ids != c.candidate_ids
    assert a.to_dict() == b.to_dict()


def test_random_search_trajectory_independent_of_batch_size():
    e = Env(radius=1.5)
    a, *_ = e.run("random", 20, seed=2, batch_size=1)
    b, *_ = e.run("random", 20, seed=2, batch_size=7)
    c, *_ = e.run("random", 20, seed=2, batch_size=64)
    assert a.candidate_ids == b.candidate_ids == c.candidate_ids


def test_random_search_uses_phase5_generator_sequence():
    from src.candidate_generator import UniformRandomGenerator
    e = Env(radius=1.5)
    r, *_ = e.run("random", 10, seed=5)
    ref = [c.candidate_id for c in UniformRandomGenerator(e.space).generate(len(r.candidate_ids), seed=5)]
    assert r.candidate_ids == ref


def test_cmaes_deterministic_and_seed_sensitive():
    e = Env(radius=1.5)
    a, *_ = e.run("cmaes", 30, seed=1, popsize=8)
    b, *_ = e.run("cmaes", 30, seed=1, popsize=8)
    c, *_ = e.run("cmaes", 30, seed=2, popsize=8)
    assert a.candidate_ids == b.candidate_ids and a.to_dict() == b.to_dict()
    assert a.candidate_ids != c.candidate_ids


def test_bayesian_deterministic_and_seed_sensitive():
    e = Env(radius=1.5)
    a, *_ = e.run("bayesian", 24, seed=1)
    b, *_ = e.run("bayesian", 24, seed=1)
    c, *_ = e.run("bayesian", 24, seed=2)
    assert a.candidate_ids == b.candidate_ids and a.verified_failure_ids == b.verified_failure_ids
    assert a.candidate_ids != c.candidate_ids


def test_reproducibility_metadata_present():
    e = Env(radius=1.5)
    for m in ALL:
        r, *_ = e.run(m, 10, seed=7)
        rp = r.reproducibility
        for k in ("method", "seed", "space_id", "reference_fingerprint", "validator_config_hash", "common_config_hash",
                  "method_config", "objective", "trajectory_sha256", "numpy", "sklearn", "duplicate_policy"):
            assert k in rp, (m, k)
        assert rp["method"] == m and rp["seed"] == 7 and rp["space_id"] == e.space.space_id
        assert rp["reference_fingerprint"] == e.ref.fingerprint


# ------------------------------------------------------------------ Bayesian specifics
def test_bayesian_initial_design_counted_deterministic_and_in_box():
    e = Env(radius=1.5)
    eng = e.engine("bayesian", 7, seed=3, n_initial=7)
    T = eng.initial_design(7)
    assert T.shape == (7, eng.rep.dim) and (T >= 0).all() and (T <= 1).all()
    assert np.array_equal(T, e.engine("bayesian", 7, seed=3, n_initial=7).initial_design(7))
    assert not np.array_equal(T, e.engine("bayesian", 7, seed=4, n_initial=7).initial_design(7))
    assert np.array_equal(T[:4], eng.initial_design(4))                       # point i depends only on (seed, i)
    bb, backend = e.box(7)
    r = eng.run(bb)
    expected = [c.candidate_id for c in eng._candidates(T, "bayesian_init", [{}] * 7)]
    assert r.candidate_ids[:7] == expected
    assert r.queries_used == backend.rows == 7 and len(eng._y) == 7             # init is counted, never free


def test_bayesian_initial_size_rule_and_budget_cap():
    e = Env()
    assert e.engine("bayesian", 100)._n_initial() == 10 and e.engine("bayesian", 20)._n_initial() == 5
    assert e.engine("bayesian", 3)._n_initial() == 3                              # never above the budget
    assert e.engine("bayesian", 1000, n_initial=11)._n_initial() == 11


def test_expected_improvement_and_ucb_properties():
    from scipy.stats import norm
    ei = expected_improvement(np.array([1.0]), np.array([1.0]), best=0.0, xi=0.0)[0]
    assert abs(ei - (norm.cdf(1) + norm.pdf(1))) < 1e-9
    assert expected_improvement(np.array([5.0]), np.array([0.0]), 0.0)[0] == 0.0           # no uncertainty -> 0
    mu = np.linspace(-1, 1, 9)
    v = expected_improvement(mu, np.full(9, 0.3), 0.0, 0.01)
    assert (v >= 0).all() and (np.diff(v) >= -1e-12).all()                          # monotone in the mean
    s = expected_improvement(np.zeros(5), np.linspace(0.1, 1, 5), 0.0, 0.0)
    assert (np.diff(s) > 0).all()                                                    # more uncertainty -> more EI at mu=best
    assert np.allclose(upper_confidence_bound(np.array([1.0]), np.array([2.0]), 1.5), 4.0)


def test_bayesian_acquisition_uses_only_counted_observations_and_returns_unseen_candidates():
    e = Env(radius=1.5)
    r, bb, backend, eng = e.run("bayesian", 30, seed=1, n_initial=6)
    assert eng.last_acquisition is not None and np.isfinite(eng.last_acquisition["max_acq"])
    assert eng._gp_fits > 0
    # every surrogate observation is a (coordinates, score) pair from a candidate that went through the gateway
    assert len(eng._y) == r.candidates_evaluated - r.duplicates_skipped
    assert len(eng._T) == len(eng._y) and all(np.isfinite(eng._y))
    assert len(set(r.candidate_ids)) == len(r.candidate_ids)                         # acquisition never re-proposes
    assert r.queries_used == backend.rows == bb.n_queries


def test_bayesian_acquire_excludes_seen_ids_and_respects_batch():
    e = Env(radius=1.5)
    bb, backend = e.box(40)
    eng = e.engine("bayesian", 40, seed=1, n_initial=6, batch_size=3)
    eng.run(bb)
    assert eng.last_acquisition is not None
    assert len(eng.last_acquisition["chosen_acq"]) <= 3


def test_bayesian_training_set_cap_keeps_best_and_recent():
    e = Env()
    eng = e.engine("bayesian", 100, max_train_points=8)
    rng = np.random.default_rng(0)
    for i in range(20):
        eng._T.append(rng.random(eng.rep.dim)); eng._y.append(float(i == 3) * 5.0 + 0.01 * i)
    T, y = eng._training_set()
    assert len(y) == 8 and y.max() == max(eng._y) and abs(y[-1] - eng._y[-1]) < 1e-12


def test_bayesian_default_representation_is_reduced_and_gp_dim_warning():
    e = Env()
    eng = e.engine("bayesian", 20, n_components=3)
    assert eng.rep.kind == "pca" and eng.rep.dim == 3
    big = rep_mod.PCARepresentation(e.space, e.ref.X, n_components=50)
    assert big.dim <= D                                                               # cannot exceed feature count
    assert BayesianSearchConfig().representation == "pca"


# ------------------------------------------------------------------ CMA-ES specifics
def test_cmaes_state_minimises_a_sphere():
    st = CMAESState(np.full(5, 3.0), 1.0)
    rng = np.random.default_rng(0)
    for _ in range(160):
        Z = rng.standard_normal((st.lam, st.n))
        Y = st.ask(Z)
        st.tell(Y, (Y ** 2).sum(axis=1))
    assert (st.m ** 2).sum() < 1e-6 and st.generation == 160


def test_cmaes_default_population_and_partial_population_terminates_cleanly():
    assert CMAESState(np.zeros(77), 0.1).lam == 4 + int(3 * np.log(77))
    e = Env(radius=1.5)
    r, bb, backend, eng = e.run("cmaes", 13, seed=1, popsize=9)
    assert r.queries_used == 13 == bb.n_queries == backend.rows and bb.exhausted
    assert r.termination_reason == "budget_exhausted"
    assert r.candidates_evaluated <= r.candidates_generated
    if r.candidates_generated > r.candidates_evaluated:
        assert any("partial population" in w for w in r.warnings)


def test_cmaes_max_generations_and_restarts():
    e = Env(radius=1.5)
    r, *_ = e.run("cmaes", 500, seed=1, popsize=8, max_generations=3)
    assert r.termination_reason == "max_generations_reached" and r.reproducibility["generations"] == 3
    assert r.queries_used <= 24
    r2, *_ = e.run("cmaes", 60, seed=1, popsize=8, sigma0=1e-6, sigma_min=1e-3)       # collapses -> restarts
    assert r2.reproducibility["restarts"] >= 1 and r2.queries_used <= 60


def test_cmaes_x0_policies_and_representations_run():
    e = Env(radius=1.5)
    for x0 in ("reference_anchor", "reference_mean", "random_uniform"):
        r, *_ = e.run("cmaes", 12, popsize=6, x0=x0)
        assert r.queries_used <= 12 and r.termination_reason in ("budget_exhausted", "candidate_cap_reached")
    r, *_ = e.run("cmaes", 12, popsize=6, representation="pca", n_components=3)
    assert r.reproducibility["representation"]["kind"] == "pca"


# ------------------------------------------------------------------ bounds / uniqueness / validity
def test_bounds_respected_and_ids_unique_for_all_methods():
    for radius in (None, 1.5):
        for m in ALL:
            env2 = Env(radius=radius)
            rv = RecordingValidator(env2.validator)
            env2.validator = rv
            bb, _ = env2.box(15)
            eng = env2.engine(m, 15, seed=2)
            r = eng.run(bb)
            X = np.stack([c.x for c in rv.seen])
            assert len(X) == r.candidates_evaluated
            assert (X >= env2.space.lower - 1e-9).all() and (X <= env2.space.upper + 1e-9).all(), m
            assert env2.space.contains(X).all(), m                                   # also radius constraint
            ids = [c.candidate_id for c in rv.seen]
            assert len(set(ids)) == len(ids), m                                      # skip policy: no re-evaluation
            assert r.candidates_valid + r.candidates_invalid == len(set(r.candidate_ids))


def test_invalid_candidates_rejected_and_consume_zero_queries():
    e = Env()
    bb, backend = e.box(5)
    sess = FeedbackManager(e.validator, bb, SearchScore(SearchScoreConfig(), 0.8), FailureTracker("t", 0, 0.05),
                           seed=0, duplicate_policy="skip", max_candidates=100)
    far = [Candidate.from_array(np.full(D, 50.0 + i), NAMES, "t") for i in range(10)]
    bad = [Candidate.from_array(np.r_[np.nan, np.zeros(D - 1)], NAMES, "t"), Candidate.from_array(np.zeros(D + 1), NAMES, "t")]
    fb = sess.evaluate(far + bad)
    assert len(fb) == 12 and not any(f.valid or f.queried for f in fb)
    assert bb.n_queries == 0 and backend.rows == 0 and backend.calls == 0 and not bb.exhausted
    assert all(f.score < 0 for f in fb)


def test_invalid_penalty_is_graded_and_below_every_valid_score():
    e = Env()
    sc = SearchScore(SearchScoreConfig(), 0.8)
    bb, _ = e.box(None)
    pts = [np.zeros(D) + np.r_[k, np.zeros(D - 1)] for k in (-1.0, 0.5, 8.0, 30.0, 300.0)]
    res = e.validator.validate_matrix(np.stack(pts), bb)
    scores = [sc.score(r) for r in res]
    valid = [s for s, r in zip(scores, res) if r.candidate_valid]
    inval = [s for s, r in zip(scores, res) if not r.candidate_valid]
    assert valid and inval and min(valid) >= 0 and max(inval) <= -0.1
    far = [sc.score(r) for r in e.validator.validate_matrix(np.array([[10.0] + [0] * 5, [100.0] + [0] * 5]), bb)]
    assert far[1] <= far[0] < 0


# ------------------------------------------------------------------ budget enforcement
def test_query_budget_enforced_exactly_for_all_methods_and_budgets():
    e = Env(radius=1.5)
    for m in ALL:
        for budget in (1, 7, 20, 33):
            r, bb, backend, _ = e.run(m, budget, seed=1)
            assert r.queries_used == bb.n_queries == backend.rows <= budget, (m, budget)
            assert r.black_box["budget"] == budget and r.query_budget == budget
            if r.termination_reason == "budget_exhausted":
                assert r.queries_used == budget


def test_budget_cannot_be_reset_or_reused():
    e = Env(radius=1.5)
    bb, _ = e.box(10)
    r = e.engine("random", 10).run(bb)
    assert bb.n_queries == 10
    for bad in (lambda: e.engine("random", 10).run(bb),                      # used box
                lambda: e.engine("random", 10).run(e.box(11)[0]),            # wrong budget
                lambda: e.engine("random", 10).run(QueryBudgetedModel(Backend(), D, 2, None))):   # unbudgeted
        try:
            bad(); assert False
        except ValueError:
            pass
    eng = e.engine("random", 10)
    eng.run(e.box(10)[0])
    try:
        eng.run(e.box(10)[0]); assert False
    except RuntimeError:
        pass
    assert bb.n_queries == 10                                               # untouched by the rejected attempts


def test_feedback_manager_detects_counter_tampering():
    e = Env(radius=1.5)

    class Tamper(Backend):
        bb = None
        def predict_proba(self, X):
            P = super().predict_proba(X)
            if self.calls == 2:
                self.bb._n_queries = 0                                       # simulate a budget reset
            return P

    t = Tamper()
    bb = QueryBudgetedModel(t, D, 2, 50)
    t.bb = bb
    sess = FeedbackManager(e.validator, bb, SearchScore(SearchScoreConfig(), 0.8), FailureTracker("t", 0, 0.05),
                           seed=0, duplicate_policy="skip", max_candidates=100)
    pts = e.space.project(e.ref.X[:6] * 0.2)
    c = [Candidate.from_array(p, NAMES, "t") for p in pts]
    sess.evaluate(c[:3])
    try:
        sess.evaluate(c[3:]); assert False
    except BudgetAccountingError:
        pass


def test_all_target_calls_go_through_the_wrapper_and_generation_is_query_free():
    e = Env(radius=1.5)
    for m in ALL:
        r, bb, backend, eng = e.run(m, 18, seed=1)
        assert backend.rows == bb.n_queries == r.queries_used
        assert backend.calls == bb.n_calls
    # generation alone never touches the model
    eng = e.engine("bayesian", 18, seed=1)
    bb, backend = e.box(18)
    eng.initial_design(5); e.engine("random", 18).generator.generate(10, seed=1)
    assert backend.rows == backend.calls == 0 and bb.n_queries == 0


# ------------------------------------------------------------------ no labels / no test data / no verdict leakage
class PoisonedPrepared:
    def __init__(self, ref: ReferenceData, X_test):
        self.X_train, self.y_train, self.cat_train = ref.X, ref.y, np.array(["BENIGN"] * len(ref.y))
        self.feature_names, self.class_names = list(ref.feature_names), list(ref.class_names)
        self.provenance = {"source": "poison-test"}
        self._X_test = X_test

    def __getattr__(self, name):
        if "test" in name:
            raise AssertionError(f"held-out test data accessed: {name}")
        raise AttributeError(name)


def test_no_test_split_accessed_by_search_construction_or_run():
    cfg = load_config()
    cfg["validator"]["calibration"]["sample_size"] = 200
    cfg["validator"]["knn"]["k"] = 10
    cfg["search"]["bayesian"]["n_components"] = 3
    prep = PoisonedPrepared(make_ref(), None)
    space = SearchSpace.from_prepared(prep, SearchSpaceConfig.from_cfg(cfg))
    validator = FailureValidator.from_prepared(prep, cfg)
    for m in ALL:
        eng = make_engine(m, cfg, space, validator, 12, 1)
        bb, backend = QueryBudgetedModel(Backend(), D, 2, 12), None
        r = eng.run(bb)
        assert r.queries_used <= 12


def test_engines_work_without_labels_in_the_reference_categories():
    e = Env(radius=1.5)
    assert e.ref.categories is None
    for m in ALL:
        assert e.run(m, 8)[0].queries_used <= 8


def _code(mod):
    text = Path(mod.__file__).read_text()
    code = re.sub(r'"""(?:.|\n)*?"""', "", text)
    return re.sub(r"#.*", "", code)


def test_static_no_labels_no_internals_no_direct_queries_no_verdict_in_optimisers():
    optimisers = (rs_mod, bo_mod, cma_mod, sb_mod, sc_mod, ft_mod, rep_mod)
    for mod in optimisers + (fm_mod,):
        code = _code(mod)
        assert not re.search(r"\b(X|y|cat)_test\b", code), mod.__name__
        assert not re.search(r"import xgboost|\bTargetModel\b|\.clf\b|booster|predict_proba_unbudgeted", code), mod.__name__
        assert not re.search(r"\.ref\.y\b|\.ref\.categories|\.categories\b|\by_train\b|\bcat_train\b", code), mod.__name__
        assert not re.search(r"\.query\(|\.predict_proba\(|_QueryBudgetedModel", code), mod.__name__   # validator is the only path
    for mod in (rs_mod, bo_mod, cma_mod):
        assert not re.search(r"\bverified_failure\b|\bfailure_reason\b", _code(mod)), mod.__name__    # optimisers never see the verdict
        assert "target_model" not in _code(mod)
    assert not re.search(r"\bverified_failure\b|\bfailure_reason\b", _code(sc_mod))


def test_search_score_ignores_the_verdict_and_is_not_a_failure_flag():
    e = Env()
    bb, _ = e.box(None)
    sc = SearchScore(SearchScoreConfig(), 0.8)
    res = e.validator.validate_matrix(np.array([[0.2, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0]]), bb)
    for r in res:
        flipped = dataclasses.replace(r, verified_failure=not r.verified_failure,
                                      failure_reason="whatever")
        assert sc.score(r) == sc.score(flipped)
    # uncertain but agreeing => high-ish uncertainty term, still NOT a verified failure
    unc = dataclasses.replace(res[1], target_proba=[0.5, 0.5], target_entropy=float(np.log(2)),
                              target_prediction=res[1].oracle_label, verified_failure=False)
    comp = sc.components(unc)
    assert comp["uncertainty"] > 0.99 and unc.verified_failure is False


def test_search_score_components_and_validation():
    e = Env()
    bb, _ = e.box(None)
    r = [x for x in e.validator.validate_matrix(np.array([[0.2, 0, 0, 0, 0, 0]]), bb)][0]
    assert r.candidate_valid and r.oracle_label == 1 and r.target_prediction == 0 and r.verified_failure
    sc = SearchScore(SearchScoreConfig(), 0.8)
    c = sc.components(r)
    assert abs(c["failure_likelihood"] - (1 - r.target_proba[r.oracle_label])) < 1e-12 and 0 <= sc.score(r) <= 1
    for bad in (dict(w_uncertainty=-1), dict(w_uncertainty=0, w_failure_likelihood=0, w_novelty=0), dict(invalid_base=0),
                dict(w_novelty=float("nan"))):
        try:
            SearchScoreConfig(**bad).validate(); assert False, bad
        except ValueError:
            pass
    nov = SearchScore(SearchScoreConfig(w_novelty=1.0, w_uncertainty=0, w_failure_likelihood=0), 0.8)
    assert nov.score(r, np.zeros(D)) == 1.0                                       # first point is maximally novel
    assert nov.score(r, np.zeros(D)) == 0.0                                       # identical point has zero novelty


# ------------------------------------------------------------------ termination / duplicates / tracking
def test_clean_termination_when_every_candidate_is_invalid():
    e = Env(cap_factor=3, validator_over=dict(maha_mode="absolute", maha_absolute=1e-6))
    for m in ALL:
        r, bb, backend, _ = e.run(m, 5)
        assert r.termination_reason == "candidate_cap_reached", m
        assert r.queries_used == 0 == backend.rows and r.candidates_evaluated == 15 and r.candidates_valid == 0
        assert any("low validity" in w for w in r.warnings) and any("cap reached" in w for w in r.warnings)
        assert r.verified_failures == 0 and not r.paper_eligible


def test_termination_reasons_are_from_the_documented_set():
    e = Env(radius=1.5)
    for m in ALL:
        r, *_ = e.run(m, 9)
        assert r.termination_reason in ("budget_exhausted", "candidate_cap_reached", "max_generations_reached")


def test_duplicate_policy_skip_vs_allow_and_cache():
    for policy, expected_queries in (("skip", 1), ("allow", 2)):
        e = Env(policy=policy)
        bb, backend = e.box(10)
        sess = FeedbackManager(e.validator, bb, SearchScore(SearchScoreConfig(), 0.8), FailureTracker("t", 0, 0.05),
                               seed=0, duplicate_policy=policy, max_candidates=50)
        c = Candidate.from_array(np.array([0.2, 0, 0, 0, 0, 0]), NAMES, "t")
        f1 = sess.evaluate([c])[0]
        f2 = sess.evaluate([Candidate.from_array(c.x, NAMES, "other-origin")])[0]
        assert f1.valid and f1.queried and bb.n_queries == expected_queries
        assert f2.duplicate == (policy == "skip") and f2.score == f1.score
        assert sess.tracker.n_verified == 1                                  # same id is one failure either way
        assert sess.tracker.n_events == expected_queries
    e = Env()                                                                # duplicates inside ONE chunk
    bb, _ = e.box(10)
    sess = FeedbackManager(e.validator, bb, SearchScore(SearchScoreConfig(), 0.8), FailureTracker("t", 0, 0.05),
                           seed=0, duplicate_policy="skip", max_candidates=50)
    c = Candidate.from_array(np.array([0.2, 0, 0, 0, 0, 0]), NAMES, "t")
    out = sess.evaluate([c, c, c])
    assert bb.n_queries == 1 and [f.duplicate for f in out] == [False, True, True] and sess.n_duplicates == 2


def test_failure_tracker_records_required_fields_and_no_label_columns():
    e = Env()
    r, *_ = e.run("random", 20, seed=1)
    assert r.trajectory is not None and len(r.trajectory) == r.candidates_evaluated - r.duplicates_skipped
    need = {"candidate_id", "method", "seed", "query_number", "candidate_valid", "search_score", "verified_failure",
            "discovery_order"}
    assert need <= set(r.trajectory[0])
    assert not any("label" in k or "test" in k for k in r.trajectory[0])
    orders = [t["discovery_order"] for t in r.trajectory if t["discovery_order"] is not None]
    assert orders == list(range(1, len(orders) + 1)) and len(orders) == r.verified_failures
    q = [t["query_number"] for t in r.trajectory if t["query_number"] is not None]
    assert q == list(range(1, len(q) + 1))                                    # query numbers are 1..queries_used in order
    assert all(t["method"] == "random" and t["seed"] == 1 for t in r.trajectory)
    assert all(t["query_number"] is None for t in r.trajectory if not t["candidate_valid"])


def test_tracker_verdict_comes_from_validator_and_unique_failures_cluster():
    e = Env()
    bb, _ = e.box(None)
    tr = FailureTracker("t", 0, 0.5)
    sc = SearchScore(SearchScoreConfig(), 0.8)
    pts = [np.array([0.2, 0, 0, 0, 0, 0]), np.array([0.21, 0, 0, 0, 0, 0]), np.array([0.2, 1.0, 0, 0, 0, 0])]
    res = e.validator.validate_matrix(np.stack(pts), bb)
    for r, x in zip(res, pts):
        tr.record(r, sc.score(r), x)
    assert tr.verified_ids == [r.candidate_id for r in res if r.verified_failure]
    assert tr.n_verified == sum(r.verified_failure for r in res)
    assert tr.n_unique <= tr.n_verified and tr.n_unique >= 1


def test_all_three_algorithms_run_and_find_validator_verified_failures_only():
    e = Env()
    for m in ALL:
        rv = RecordingValidator(e.validator)
        e2 = Env()
        e2.validator = rv
        bb, _ = e2.box(25)
        r = e2.engine(m, 25, seed=1).run(bb)
        assert r.queries_used <= 25 and r.candidates_evaluated >= 1
        assert r.verified_failures <= r.candidates_valid <= r.queries_used
        assert set(r.verified_failure_ids) <= set(r.candidate_ids)
        assert r.unique_verified_failures <= r.verified_failures
        # re-check each reported failure with an independent validator pass: the validator's verdict must agree
        by_id = {c.candidate_id: c for c in rv.seen}
        chk, _ = e.box(None)
        for cid in r.verified_failure_ids:
            assert e.validator.validate_batch([by_id[cid]], chk)[0].verified_failure


# ------------------------------------------------------------------ results / serialisation / runner / config
def test_results_are_serialisable_and_round_trip(tmp_path):
    e = Env(radius=1.5)
    for m in ALL:
        r, *_ = e.run(m, 10)
        d = r.to_dict()
        blob = json.dumps(d)
        r2 = SearchResult.from_dict(json.loads(blob))
        assert r2.to_dict() == d
        for k in ("method", "seed", "query_budget", "queries_used", "candidates_generated", "candidates_valid",
                  "verified_failures", "unique_verified_failures", "candidate_ids", "objective_stats",
                  "termination_reason", "reproducibility", "paper_eligible", "warnings"):
            assert k in d
        p = r.save(tmp_path / f"{m}.json")
        assert json.load(open(p))["method"] == m
        assert r.to_dict(include_trajectory=False)["trajectory"] is None


def test_paper_eligible_requires_context_and_consistent_accounting():
    e = Env(radius=1.5)
    bb, _ = e.box(8)
    r = RandomSearch(e.space, e.validator, 8, 1, e.common, paper_eligible_context=True, space_config=e.space_cfg).run(bb)
    assert r.paper_eligible is True and not any("accounting" in w for w in r.warnings)
    assert e.run("random", 8)[0].paper_eligible is False


def test_experiment_runner_selects_methods_with_one_schema():
    cfg = load_config()
    cfg["search"]["bayesian"]["n_components"] = 3
    e = Env(radius=1.5)

    class FakeTarget:
        is_dev_mock = True
        def __init__(self): self.backend = Backend()
        def black_box(self, budget, name=None): return QueryBudgetedModel(self.backend, D, 2, budget, name or "x")

    assert set(ALL) <= set(METHODS)                         # Phase 7 adds 'quantum'; the three Phase 6 baselines remain
    keys = None
    for m in ALL:
        r = run_search_method(m, cfg, e.space, e.validator, FakeTarget(), 9, 1)
        assert r.method == m and r.query_budget == 9 and r.paper_eligible is False        # dev-mock => not eligible
        keys = keys or set(r.to_dict())
        assert set(r.to_dict()) == keys
    t = FakeTarget(); t.is_dev_mock = False
    assert run_search_method("random", cfg, e.space, e.validator, t, 9, 1).paper_eligible is True
    try:
        make_engine("nope", cfg, e.space, e.validator, 9, 1); assert False
    except ValueError:
        pass


def test_config_blocks_load_and_invalid_values_are_rejected():
    cfg = load_config()
    common = CommonSearchConfig.from_cfg(cfg)
    assert common.query_budgets == (100, 250, 500, 1000, 2500, 5000) and common.seeds == tuple(range(1, 11))
    assert common.unique_resolution == cfg["validator"]["dedup_resolution"]
    RandomSearchConfig.from_cfg(cfg); BayesianSearchConfig.from_cfg(cfg); CMAESConfig.from_cfg(cfg)
    for cls, bad in ((RandomSearchConfig, dict(batch_size=0)), (RandomSearchConfig, dict(generator="x")),
                     (BayesianSearchConfig, dict(acquisition="pi")), (BayesianSearchConfig, dict(max_train_points=1)),
                     (BayesianSearchConfig, dict(representation="tsne")), (CMAESConfig, dict(popsize=2)),
                     (CMAESConfig, dict(sigma0=0)), (CMAESConfig, dict(x0="test_set"))):
        try:
            cls(**bad).validate(); assert False, bad
        except ValueError:
            pass
    for bad in (dict(duplicate_policy="drop"), dict(query_budgets=()), dict(max_candidates_per_query=0.5), dict(seeds=(-1,))):
        try:
            dataclasses.replace(common, **bad).validate(); assert False, bad
        except ValueError:
            pass


def test_phase4_validator_thresholds_and_phase5_search_space_config_unchanged():
    cfg = load_config()
    v = cfg["validator"]
    assert v["failure_rule"]["min_oracle_confidence"] == 0.8 and v["mahalanobis"]["quantile"] == 0.995
    assert v["nn_gate"]["quantile"] == 0.995 and v["knn"]["k"] == 15 and v["knn"]["min_purity"] == 0.8
    assert cfg["search_space"]["bounds"]["mode"] == "reference_quantile"
    SearchSpaceConfig.from_cfg(cfg)


def test_representations_are_deterministic_label_free_and_inside_bounds():
    e = Env()
    a, b = PCARepresentation(e.space, e.ref.X, 3), PCARepresentation(e.space, e.ref.X, 3)
    assert a.fingerprint == b.fingerprint and a.dim == 3 and np.allclose(a.V.T @ a.V, np.eye(3), atol=1e-9)
    assert (np.diff(a.sd) <= 1e-12).all()
    V = np.random.default_rng(0).uniform(-5, 5, (50, 3))
    X = a.decode(V)
    assert e.space.contains(X).all()
    inside = np.random.default_rng(1).uniform(-1, 1, (20, 3))
    z1 = a.encode(a.decode(inside))
    assert np.allclose(a.encode(a.decode(z1)), z1, atol=1e-8)                    # encode o decode is idempotent
    assert (z1 >= a.lower).all() and (z1 <= a.upper).all()
    f = StandardizedFullRepresentation(e.space)
    Z = np.random.default_rng(2).uniform(f.lower, f.upper, (30, D))
    assert e.space.contains(f.decode(Z)).all()
    assert np.allclose(f.encode(f.decode(Z)), Z, atol=1e-8)
    import inspect
    assert "y" not in inspect.signature(PCARepresentation.__init__).parameters
