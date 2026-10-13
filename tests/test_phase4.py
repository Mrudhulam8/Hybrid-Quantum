"""Phase 4 tests: candidate, oracle, validator, query accounting, label-leakage guards.

All fixtures are SYNTHETIC with hand-constructed geometry. Assertions check mechanics (thresholds,
counting, determinism, independence) - they are NOT research evidence about any target model.
"""
import dataclasses
import inspect
import json
import math
import os
import re
from pathlib import Path

import numpy as np

from src import candidate as cand_mod, oracle as oracle_mod, validator as val_mod
from src.candidate import Candidate, candidates_from_matrix
from src.oracle import (KNNReference, MahalanobisValidity, ReferenceData, ReferenceOracle, NO_LABEL)
from src.preprocessing import prepare_data
from src.target_model import (DEV_MOCK_ENV, QueryBudgetExceeded, QueryBudgetedModel, train_target_model)
from src.uncertainty import prediction_entropy, prediction_margin
from src.utils import load_config
from src.validator import (RESULT_COLUMNS, F_ABSTAIN, F_AGREES, F_INVALID, F_LOW_ORACLE, F_LOW_TARGET,
                           F_VERIFIED, FailureValidator, ValidatorConfig, load_results_json,
                           results_to_dataframe, save_results)

D = 5
NAMES = tuple(f"f{i}" for i in range(D))
CENTER1 = np.array([2.0, 0, 0, 0, 0])           # centre of class-1 blob
CENTER0 = np.array([-2.0, 0, 0, 0, 0])


# ------------------------------------------------------------------ fixtures
def make_ref(seed=0, n=600) -> ReferenceData:
    rng = np.random.default_rng(seed)
    X0 = rng.normal(0, 0.5, (n, D)); X0[:, 0] -= 2
    X1 = rng.normal(0, 0.5, (n, D)); X1[:, 0] += 2
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    cats = np.where(y == 0, "BENIGN", "ATTACK-A")
    return ReferenceData(np.vstack([X0, X1]), y, cats, NAMES, ("BENIGN", "ATTACK"), "unit-test-fixture")


def vcfg(**over) -> ValidatorConfig:
    base = ValidatorConfig(calibration_sample_size=200)
    return dataclasses.replace(base, **over)


class Backend:
    """Deterministic logistic target on feature 0 with boundary c. Counts rows really evaluated."""
    def __init__(self, c=0.0, s=4.0):
        self.c, self.s, self.rows, self.calls = c, s, 0, 0

    def predict_proba(self, X):
        self.rows += len(X); self.calls += 1
        p1 = 1 / (1 + np.exp(-self.s * (X[:, 0] - self.c)))
        return np.column_stack([1 - p1, p1])


def bb(c=0.0, s=4.0, budget=None):
    b = Backend(c, s)
    return QueryBudgetedModel(b, D, 2, budget), b


def validator(ref=None, **over) -> FailureValidator:
    return FailureValidator.from_reference(ref or make_ref(), vcfg(**over))


def C(x, origin="t"):
    return Candidate.from_array(x, NAMES, origin)


# ------------------------------------------------------------------ candidate
def test_candidate_id_deterministic_and_origin_independent():
    a, b = C([1.0, -0.0, 3, 4, 5], "random"), C([1.0, 0.0, 3, 4, 5], "quantum")
    assert a.candidate_id == b.candidate_id and a == b and hash(a) == hash(b)
    assert a.candidate_id != C([1.0, 0.0, 3, 4, 5.0000001]).candidate_id
    other_schema = Candidate.from_array([1.0, 0.0, 3, 4, 5], tuple("abcde"))
    assert other_schema.candidate_id != a.candidate_id


def test_candidate_roundtrip_readonly_and_model_input():
    c = C([0.1, 0.2, 0.30000000000000004, 4, 5], "r")
    assert not c.x.flags.writeable
    d = json.loads(json.dumps(c.to_dict()))
    assert Candidate.from_dict(d, NAMES) == c and np.array_equal(Candidate.from_dict(d, NAMES).x, c.x)
    m = c.to_model_input()
    assert m.shape == (1, D) and m.dtype == np.float64
    m[0, 0] = 99
    assert c.x[0] == 0.1                                  # returned array is a copy
    try:
        Candidate.from_dict(d, tuple("abcde")); assert False
    except ValueError:
        pass


def test_candidate_dimensionality_and_schema_validation():
    assert C(np.zeros(D)).check(NAMES) == "ok"
    assert Candidate.from_array(np.zeros(D + 1), NAMES + ("extra",)).check(NAMES) == "bad_dimension"
    assert Candidate.from_array(np.zeros(D - 1), NAMES[:-1]).check(NAMES) == "bad_dimension"
    assert Candidate.from_array(np.zeros(D), tuple("vwxyz")).check(NAMES) == "bad_feature_schema"
    try:
        Candidate.from_array(np.zeros((2, D)), NAMES); assert False
    except ValueError:
        pass
    try:
        Candidate.from_array(np.zeros(D + 1), NAMES, strict=True); assert False
    except ValueError:
        pass


def test_candidate_nan_inf_detection():
    for bad in (np.nan, np.inf, -np.inf):
        x = np.zeros(D); x[2] = bad
        assert C(x).check(NAMES) == "non_finite"


def test_candidate_to_original_units_uses_the_phase2_preprocessor():
    cfg = load_config(overrides={"data": {"source": "synthetic-cic-schema", "max_rows": None,
                                          "synthetic": {"n_rows": 3000, "data_seed": 0}}})
    p = prepare_data(cfg)
    c = Candidate.from_array(p.X_test[0], p.feature_names)
    raw = c.to_original_units(p.preprocessor)
    assert list(raw) == list(p.feature_names)
    assert np.allclose(list(raw.values()), p.preprocessor.inverse_transform(p.X_test[:1])[0])


# ------------------------------------------------------------------ uncertainty measures
def test_uncertainty_measures():
    assert np.isclose(prediction_entropy([0.5, 0.5])[0], math.log(2), atol=1e-9)
    assert prediction_entropy([1.0, 0.0])[0] < 1e-9
    assert np.isclose(prediction_margin([0.7, 0.3])[0], 0.4) and np.isclose(prediction_margin([0.5, 0.5])[0], 0.0)


# ------------------------------------------------------------------ validity
def test_valid_candidate_is_valid_and_queried_once():
    v, (box, spy) = validator(), bb(0.0)
    r = v.validate_batch([C(CENTER1)], box, seed=7)[0]
    assert r.candidate_valid and r.validity_reason == "ok" and r.queried
    assert r.distance_to_reference < r.mahalanobis_threshold
    assert r.nearest_neighbor_distance < r.nn_distance_threshold
    assert r.seed == 7 and r.target_query_index == 1 and r.target_query_count == 1 == spy.rows


def test_ood_candidates_are_invalid_and_never_queried():
    far = np.array([50.0, 0, 0, 0, 0])
    box, spy = bb(0.0)
    # 1) outside feature range
    r = validator(range_enabled=True).validate_batch([C(far)], box)[0]
    assert not r.candidate_valid and r.validity_reason == "outside_feature_range"
    # 2) range gate off -> Mahalanobis gate catches it
    r = validator(range_enabled=False).validate_batch([C([12.0, 0, 0, 0, 0])], box)[0]
    assert r.validity_reason == "mahalanobis_exceeded"
    # 3) range + Mahalanobis off -> NN-distance gate catches the empty gap between the blobs
    r = validator(range_enabled=False, maha_mode="absolute", maha_absolute=1e9).validate_batch([C(np.zeros(D))], box)[0]
    assert r.validity_reason == "nn_distance_exceeded"
    assert spy.rows == 0 and box.n_queries == 0                       # nothing was charged
    assert not r.queried and r.target_prediction is None and r.failure_reason == F_INVALID
    assert r.verified_failure is False


def test_invalid_candidate_cannot_be_a_failure_even_if_target_is_wrong():
    box, spy = bb(c=-50.0)                                            # target says "1" everywhere
    r = validator().validate_batch([C(CENTER0 * 30)], box)[0]
    assert not r.candidate_valid and not r.verified_failure and spy.rows == 0


def test_range_gate_regression_floor_features_need_margin_or_gate_off():
    """Finding from validator-diagnostics: many real features sit exactly at their minimum, so a
    zero-margin range gate rejects any downward perturbation. Keep this behaviour documented."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, D)); X[:200, 1] = X[:, 1].min()          # half the rows at the floor of f1
    ref = ReferenceData(X, (X[:, 0] > 0).astype(int), None, NAMES, ("BENIGN", "ATTACK"), "floor-fixture")
    x = X[0].copy(); x[1] = X[:, 1].min() - 0.01                          # tiny nudge below the floor
    kw = dict(maha_mode="absolute", maha_absolute=1e9, nn_gate_enabled=False)
    box, _ = bb(0.0)
    strict = FailureValidator.from_reference(ref, vcfg(range_enabled=True, range_margin=0.0, **kw)).validate_batch([C(x)], box)[0]
    margin = FailureValidator.from_reference(ref, vcfg(range_enabled=True, range_margin=0.05, **kw)).validate_batch([C(x)], box)[0]
    off = FailureValidator.from_reference(ref, vcfg(range_enabled=False, **kw)).validate_batch([C(x)], box)[0]
    assert strict.validity_reason == "outside_feature_range" and margin.candidate_valid and off.candidate_valid
    assert ValidatorConfig.from_cfg(load_config()).range_enabled is False


# ------------------------------------------------------------------ uncertainty != failure
def test_uncertain_but_agreeing_candidate_is_not_a_failure():
    box, _ = bb(c=1.99, s=1.0)                                        # p1(CENTER1) ~ 0.5025: max uncertainty, argmax=1
    r = validator().validate_batch([C(CENTER1)], box)[0]
    assert r.target_entropy > 0.69 and r.target_margin < 0.01 and r.target_prediction == 1
    assert r.oracle_label == 1 and not r.verified_failure and r.failure_reason == F_AGREES


def test_confident_disagreement_is_a_verified_failure():
    box, _ = bb(c=3.0, s=4.0)                                         # target confidently says 0 at CENTER1
    r = validator().validate_batch([C(CENTER1)], box)[0]
    assert r.target_prediction == 0 and r.target_confidence > 0.95 and r.target_entropy < 0.2
    assert r.oracle_label == 1 and r.oracle_confidence >= 0.8
    assert r.verified_failure and r.failure_reason == F_VERIFIED


def test_valid_agreement_and_disagreement_both_covered():
    v = validator()
    ok_box, _ = bb(0.0); bad_box, _ = bb(3.0)
    agree = v.validate_batch([C(CENTER1), C(CENTER0)], ok_box)
    assert [r.verified_failure for r in agree] == [False, False]
    dis = v.validate_batch([C(CENTER1), C(CENTER0)], bad_box)
    assert dis[0].verified_failure and not dis[1].verified_failure     # only the class-1 blob is mislabelled


def test_target_confidence_filter_only_restricts():
    box, _ = bb(c=2.02, s=1.0)                                        # slightly wrong (p0 ~ 0.505) at CENTER1
    assert validator().validate_batch([C(CENTER1)], box)[0].verified_failure
    box2, _ = bb(c=2.02, s=1.0)
    r = validator(rule_min_target_confidence=0.9).validate_batch([C(CENTER1)], box2)[0]
    assert not r.verified_failure and r.failure_reason == F_LOW_TARGET


def test_oracle_is_independent_of_the_target_model():
    v = validator()
    r1 = v.validate_batch([C(CENTER1)], bb(0.0)[0])[0].to_dict()
    r2 = v.validate_batch([C(CENTER1)], bb(3.0)[0])[0].to_dict()
    target_keys = {"queried", "target_prediction", "target_confidence", "target_proba", "target_entropy",
                   "target_margin", "verified_failure", "failure_reason", "target_query_index", "target_query_count"}
    assert {k: v_ for k, v_ in r1.items() if k not in target_keys} == {k: v_ for k, v_ in r2.items() if k not in target_keys}
    assert r1["target_prediction"] != r2["target_prediction"]
    assert list(inspect.signature(ReferenceOracle.__init__).parameters) == ["self", "reference", "vcfg", "seed"]


# ------------------------------------------------------------------ Mahalanobis
def test_mahalanobis_matches_scipy_when_unregularised():
    from scipy.spatial.distance import mahalanobis
    X = make_ref().X
    m = MahalanobisValidity(shrinkage="none", ridge=0.0, eigenvalue_floor_rel=1e-14).fit(X)
    P = np.linalg.inv(np.cov(X.T))
    Q = np.random.default_rng(1).normal(0, 2, (10, D))
    ref = [mahalanobis(q, X.mean(0), P) for q in Q]
    assert np.allclose(m.distance(Q), ref, rtol=1e-8, atol=1e-8)


def test_mahalanobis_threshold_behaviour_absolute_and_quantile():
    ref = make_ref()
    v = validator(ref, range_enabled=False, nn_gate_enabled=False, maha_mode="absolute", maha_absolute=3.0)
    e = np.zeros(D); e[0] = 1.0
    m1 = float(v.oracle.maha.distance(v.oracle.maha.mean_ + e)[0])    # distance is linear along a ray
    inside, outside = v.oracle.maha.mean_ + e * (2.9 / m1), v.oracle.maha.mean_ + e * (3.1 / m1)
    box, _ = bb(0.0)
    r_in, r_out = v.validate_batch([C(inside), C(outside)], box)
    assert r_in.candidate_valid and not r_out.candidate_valid and r_out.validity_reason == "mahalanobis_exceeded"
    assert abs(r_in.distance_to_reference - 2.9) < 1e-6 and abs(r_out.distance_to_reference - 3.1) < 1e-6
    vq = validator(ref, maha_mode="empirical_quantile", maha_quantile=0.9)
    assert np.isclose(vq.oracle.tau_maha, np.quantile(vq.oracle.maha.distance(ref.X), 0.9))
    assert validator(ref, maha_quantile=0.999).oracle.tau_maha > vq.oracle.tau_maha
    assert validator(ref, maha_mode="chi2_quantile").oracle.tau_maha > 0


def test_mahalanobis_numerically_stable_with_collinear_and_constant_columns():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(300, 3))
    X = np.column_stack([A[:, 0], A[:, 1], A[:, 0] + A[:, 1], np.zeros(300), A[:, 2]])   # exact collinearity + constant
    for shrink in ("none", "fixed", "ledoit_wolf"):
        m = MahalanobisValidity(shrinkage=shrink, eigenvalue_floor_rel=1e-8).fit(X)
        d = m.distance(np.vstack([X[:5], X[0] + 1e-3]))
        assert np.isfinite(d).all() and np.isfinite(m.diagnostics_["condition_number"])
        assert m.diagnostics_["effective_rank"] <= 5
    assert MahalanobisValidity(shrinkage="none").fit(X).n_floored_ >= 1


# ------------------------------------------------------------------ k-NN purity / ties
def _tiny_ref(points, labels, d=1):
    X = np.asarray(points, float).reshape(len(points), d)
    return X, np.asarray(labels, int)


def test_knn_tie_handling_is_deterministic():
    # k=4 tie 2-2: summed distance decides
    X, y = _tiny_ref([1, 2, -1.5, -3], [0, 0, 1, 1])
    knn = KNNReference(X, y, None, 2)
    idx, dist = knn.neighbors(np.array([[0.0]]), 4)
    assert knn.vote(idx, dist, 0.5)[0][0] == 0                         # class0 sum 3.0 < class1 sum 4.5
    X, y = _tiny_ref([1, 3, -1.5, -2], [0, 0, 1, 1])
    knn = KNNReference(X, y, None, 2)
    idx, dist = knn.neighbors(np.array([[0.0]]), 4)
    assert knn.vote(idx, dist, 0.5)[0][0] == 1                         # class1 sum 3.5 < class0 sum 4.0
    # exact tie in votes and summed distance -> lowest label; neighbour order by (distance, index)
    X, y = _tiny_ref([1, 3, -1, -3], [1, 1, 0, 0])
    knn = KNNReference(X, y, None, 2)
    idx, dist = knn.neighbors(np.array([[0.0]]), 4)
    assert idx[0].tolist() == [0, 2, 1, 3] and knn.vote(idx, dist, 0.5)[0][0] == 0


def test_knn_independent_of_chunk_size():
    ref = make_ref(); Q = np.random.default_rng(3).normal(0, 2, (50, D))
    a = KNNReference(ref.X, ref.y, None, 2, chunk_size=1).neighbors(Q, 15)
    b = KNNReference(ref.X, ref.y, None, 2, chunk_size=64).neighbors(Q, 15)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def _two_region_ref():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 0.01, (10, 2)); B = rng.normal(0, 0.01, (10, 2)) + [10, 0]
    y = np.r_[[0] * 9 + [1], [1] * 6 + [0] * 4]
    return ReferenceData(np.vstack([A, B]), y, None, ("u", "w"), ("BENIGN", "ATTACK"), "tiny")


def _tiny_validator(**over):
    base = dict(knn_k=10, range_enabled=False, nn_gate_enabled=False, maha_mode="absolute", maha_absolute=1e9,
                calibration_sample_size=5, knn_min_purity=0.8, rule_min_oracle_confidence=0.8)
    base.update(over)
    return FailureValidator.from_reference(_two_region_ref(), vcfg(**base))


class Flat:                                                           # target that always answers "1" with p=.9
    def predict_proba(self, X): return np.tile([0.1, 0.9], (len(X), 1))


def test_knn_purity_threshold_behaviour():
    A, B = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    box = lambda: QueryBudgetedModel(Flat(), 2, 2, None)
    rA, rB = _tiny_validator().validate_batch([Candidate.from_array(A, ("u", "w")), Candidate.from_array(B, ("u", "w"))], box())
    assert np.isclose(rA.nearest_neighbor_purity, 0.9) and rA.oracle_label == 0 and np.isclose(rA.oracle_confidence, 0.9)
    assert np.isclose(rB.nearest_neighbor_purity, 0.6) and rB.oracle_label is None and rB.oracle_confidence == 0.0
    assert rB.failure_reason == F_ABSTAIN and not rB.verified_failure
    # lower min_purity: oracle now labels region B (=1) and the always-1 target agrees
    rB2 = _tiny_validator(knn_min_purity=0.5, rule_min_oracle_confidence=0.5).validate_batch(
        [Candidate.from_array(B, ("u", "w"))], box())[0]
    assert rB2.oracle_label == 1 and np.isclose(rB2.oracle_confidence, 0.6) and rB2.failure_reason == F_AGREES
    # region A: purity .9, oracle says 0, target says 1 -> failure; raising the rule threshold above .9 blocks it
    assert rA.verified_failure and rA.failure_reason == F_VERIFIED
    rA2 = _tiny_validator(rule_min_oracle_confidence=0.95).validate_batch([Candidate.from_array(A, ("u", "w"))], box())[0]
    assert not rA2.verified_failure and rA2.failure_reason == F_LOW_ORACLE


# ------------------------------------------------------------------ query accounting
def test_exactly_one_query_per_valid_candidate_and_none_for_invalid():
    v, (box, spy) = validator(), bb(0.0)
    X = [C(CENTER1), C(np.full(D, np.nan)), C(CENTER0), C(np.zeros(D + 1)), C([99, 0, 0, 0, 0]), C(CENTER1 * 1.01)]
    r = v.validate_batch(X, box, seed=1)
    queried = [x for x in r if x.queried]
    assert len(queried) == 3 == box.n_queries == spy.rows and spy.calls == 1
    assert [x.target_query_index for x in queried] == [1, 2, 3]
    assert all(x.target_query_count == 3 for x in r)
    r2 = v.validate_batch([C(CENTER0)], box)
    assert r2[0].target_query_index == 4 and box.n_queries == spy.rows == 4 and spy.calls == 2


def test_budget_exhaustion_raises_and_charges_nothing():
    v, (box, spy) = validator(), bb(0.0, budget=5)
    batch = [C(CENTER1)] * 6
    try:
        v.validate_batch(batch, box); assert False
    except QueryBudgetExceeded:
        pass
    assert box.n_queries == 0 == spy.rows
    r = v.validate_batch([C(CENTER1)] * 4 + [C(np.full(D, np.inf))] * 3, box)    # 4 valid fit in budget 5
    assert box.n_queries == 4 == spy.rows and box.remaining == 1
    try:
        v.validate_batch([C(CENTER1)] * 2, box); assert False
    except QueryBudgetExceeded:
        pass
    assert box.n_queries == 4 == spy.rows


def test_all_invalid_batch_makes_no_black_box_call_even_with_zero_budget():
    box, spy = bb(0.0, budget=0)
    r = validator().validate_batch([C(np.full(D, np.nan)), C([99, 0, 0, 0, 0])], box)
    assert spy.calls == 0 and box.n_queries == 0 and not any(x.queried for x in r)
    assert validator().validate_batch([], box) == []


def test_nan_inf_rejected_without_crashing_the_batch():
    box, spy = bb(0.0)
    r = validator().validate_batch([C([np.nan, 0, 0, 0, 0]), C([np.inf, 0, 0, 0, 0]), C(CENTER1)], box)
    assert [x.validity_reason for x in r] == ["non_finite", "non_finite", "ok"]
    assert r[0].oracle_label is None and r[0].distance_to_reference is None
    assert spy.rows == 1


def test_validator_rejects_mismatched_black_box():
    v = validator()
    wrong = QueryBudgetedModel(Backend(), D + 1, 2, None)
    try:
        v.validate_batch([C(CENTER1)], wrong); assert False
    except ValueError:
        pass


# ------------------------------------------------------------------ no label leakage
class PoisonedPrepared:
    """Exposes ONLY training fields; any held-out/test access fails the test."""
    def __init__(self, p):
        self.X_train, self.y_train, self.cat_train = p.X_train, p.y_train, p.cat_train
        self.feature_names, self.class_names, self.provenance = p.feature_names, p.class_names, p.provenance

    def __getattr__(self, name):
        if "test" in name:
            raise AssertionError(f"held-out test field accessed: {name}")
        raise AttributeError(name)


def _synthetic_prepared(n=5000):
    cfg = load_config(overrides={"data": {"source": "synthetic-cic-schema", "max_rows": None,
                                          "synthetic": {"n_rows": n, "data_seed": 0}}})
    cfg["validator"]["calibration"]["sample_size"] = 300
    return cfg, prepare_data(cfg)


def test_no_test_label_dependency_runtime_and_static():
    cfg, prep = _synthetic_prepared()
    v = FailureValidator.from_prepared(PoisonedPrepared(prep), cfg)      # raises if any *_test field is read
    ref = v.oracle.ref
    assert ref.n == len(prep.X_train) and ref.fingerprint == ReferenceData.from_prepared(prep).fingerprint
    assert not any("test" in f.name for f in dataclasses.fields(ReferenceData))
    X = prep.X_train[:20] + 0.01
    spy_box = QueryBudgetedModel(Backend(), prep.X_train.shape[1], 2, None)
    v.validate_batch(candidates_from_matrix(X, ref.feature_names, "t"), spy_box)
    # static guard on the discovery-side modules
    src_dir = Path(inspect.getfile(val_mod)).parent
    for mod in ("oracle.py", "validator.py", "candidate.py", "uncertainty.py"):
        text = (src_dir / mod).read_text()
        code = re.sub(r'""".*?"""', "", text, flags=re.S)
        code = re.sub(r"#.*", "", code)
        assert not re.search(r"\b(X|y|cat)_test\b", code), f"{mod} references held-out test data"
        assert not re.search(r"import xgboost|\bTargetModel\b|\.clf\b|booster|predict_proba_unbudgeted|target_model import",
                             code), f"{mod} reaches target-model internals"


# ------------------------------------------------------------------ config
def test_validator_config_loads_from_project_config_and_is_overridable():
    cfg = load_config()
    c = ValidatorConfig.from_cfg(cfg)
    v = cfg["validator"]
    assert (c.knn_k, c.knn_min_purity) == (v["knn"]["k"], v["knn"]["min_purity"])
    assert c.maha_quantile == v["mahalanobis"]["quantile"] and c.maha_shrinkage == v["mahalanobis"]["shrinkage"]
    assert c.rule_min_oracle_confidence == v["failure_rule"]["min_oracle_confidence"] and c.numerical_tol == v["numerical_tol"]
    c2 = ValidatorConfig.from_cfg(load_config(overrides={"validator": {"knn": {"k": 7}, "mahalanobis": {"quantile": 0.9}}}))
    assert c2.knn_k == 7 and c2.maha_quantile == 0.9 and c2.knn_min_purity == c.knn_min_purity


def test_invalid_validator_config_is_rejected():
    bad_overrides = [{"knn": {"k": 0}}, {"knn": {"min_purity": 1.5}}, {"failure_rule": {"min_oracle_confidence": 0.5}},
                     {"mahalanobis": {"threshold_mode": "absolute", "absolute_threshold": None}},
                     {"mahalanobis": {"shrinkage": "bogus"}}, {"nn_gate": {"quantile": 1.0}},
                     {"failure_rule": {"min_target_confidence": 2.0}}]
    for ov in bad_overrides:
        try:
            ValidatorConfig.from_cfg(load_config(overrides={"validator": ov})); assert False, ov
        except ValueError:
            pass


# ------------------------------------------------------------------ serialisation / reproducibility
def _mixed_results():
    v, (box, _) = validator(), bb(3.0)
    return v, v.validate_batch([C(CENTER1), C(CENTER0), C(np.full(D, np.nan)), C([99, 0, 0, 0, 0])], box, seed=5)


def test_result_schema_contains_all_required_fields_and_serialises(tmp_path):
    v, res = _mixed_results()
    required = ["candidate_id", "seed", "target_prediction", "target_confidence", "oracle_label", "oracle_confidence",
                "candidate_valid", "validity_reason", "distance_to_reference", "nearest_neighbor_distance",
                "nearest_neighbor_purity", "verified_failure", "target_query_count"]
    assert set(required) <= set(RESULT_COLUMNS)
    paths = save_results(res, tmp_path / "val", meta=v.describe())
    back = load_results_json(paths["json"])
    assert [r.to_dict() for r in back] == [r.to_dict() for r in res]
    import pandas as pd
    df = pd.read_csv(paths["csv"])
    assert list(df.columns) == RESULT_COLUMNS and len(df) == 4
    assert df["verified_failure"].tolist() == [True, False, False, False]
    meta = json.load(open(paths["json"]))["meta"]
    assert "validator_config" in meta and "oracle" in meta
    assert results_to_dataframe(res).shape == (4, len(RESULT_COLUMNS))


def test_reproducibility_same_inputs_same_everything():
    ref = make_ref()
    X = np.random.default_rng(9).normal(0, 1.5, (60, D))
    out = []
    for chunk in (64, 1):                                              # chunk size must not matter either
        v = FailureValidator.from_reference(ref, vcfg(knn_chunk_size=chunk))
        box, _ = bb(1.0)
        out.append(([r.to_dict() for r in v.validate_matrix(X, box, seed=3)], v.describe()))
    assert out[0][0] == out[1][0]
    d0, d1 = out[0][1]["oracle"], out[1][1]["oracle"]
    assert d0["tau_mahalanobis"] == d1["tau_mahalanobis"] and d0["tau_nn"] == d1["tau_nn"]
    assert d0["reference_fingerprint"] == make_ref().fingerprint
    assert make_ref(seed=1).fingerprint != make_ref(seed=0).fingerprint
    # calibration seed changes the sub-sample deterministically
    a = validator(make_ref(), seed=1).oracle.tau_nn; b = validator(make_ref(), seed=1).oracle.tau_nn
    assert a == b


# ------------------------------------------------------------------ diagnostics + pipeline integration
def test_reference_only_diagnostics_run_and_are_sane():
    v = validator()
    sc = v.oracle.self_consistency(sample_size=300, seed=0)
    assert sc["n"] == 300 and 0 <= sc["abstain_rate"] <= 1 and set(sc["per_category"]) == {"BENIGN", "ATTACK-A"}
    prof = v.oracle.perturbation_validity_profile(sigmas=(0.05, 3.0), n=200, seed=0)
    assert prof["0.05"]["valid_rate"] >= prof["3.0"]["valid_rate"]


def test_integration_with_phase2_data_and_phase3_black_box():
    cfg, prep = _synthetic_prepared()
    os.environ[DEV_MOCK_ENV] = "1"
    try:
        tm = train_target_model(cfg, prep, kind="dev-mock")          # DEV MOCK: plumbing only, no result is reported
    finally:
        os.environ.pop(DEV_MOCK_ENV)
    v = FailureValidator.from_prepared(prep, cfg)
    box = tm.black_box(500)
    rng = np.random.default_rng(0)
    Q = prep.X_train[rng.choice(len(prep.X_train), 300, replace=False)] + rng.normal(0, 0.02, (300, prep.X_train.shape[1]))
    res = v.validate_matrix(Q, box, seed=1, origin="fixture")
    assert box.n_queries == sum(r.queried for r in res) and box.n_queries <= 300
    assert all(r.verified_failure <= r.candidate_valid for r in res)
    assert all(not r.queried for r in res if not r.candidate_valid)
