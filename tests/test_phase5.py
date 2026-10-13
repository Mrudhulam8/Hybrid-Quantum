"""Phase 5 tests: search space + query-free candidate generation.

Synthetic fixtures only (no CIC-IDS2017, no XGBoost). Assertions check mechanics - determinism,
bounds, limits, query-freeness, label-freeness - and are NOT research evidence.
"""
import dataclasses
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from src import candidate_generator as gen_mod, search_space as space_mod
from src.candidate import Candidate
from src.candidate_generator import (GENERATOR_NAMES, LocalPerturbationGenerator, LocalPerturbationParams,
                                     UniformRandomGenerator, make_generator)
from src.oracle import MahalanobisValidity, ReferenceData
from src.search_space import SearchSpace, SearchSpaceConfig
from src.target_model import QueryBudgetedModel, TargetModel
from src.utils import load_config
from src.validator import FailureValidator, ValidatorConfig

D = 6
NAMES = tuple(f"f{i}" for i in range(D))
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ fixtures
def ref_matrix(seed=0, n=800):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, D))
    A[:, 1] = 0.6 * A[:, 0] + 0.8 * A[:, 1]            # correlation
    A[:, 2] = A[:, 2] ** 3                              # heavy tail
    A[: n // 2, 3] = A[:, 3].min()                      # clipped floor (like many CIC features)
    return A


def scfg(**over) -> SearchSpaceConfig:
    return dataclasses.replace(SearchSpaceConfig(), **over)


def space(X=None, **over) -> SearchSpace:
    return SearchSpace.from_reference(ref_matrix() if X is None else X, NAMES, scfg(**over))


class Spy:
    def __init__(self):
        self.rows = self.calls = 0

    def predict_proba(self, X):
        self.rows += len(X); self.calls += 1
        p = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack([1 - p, p])


def box(budget=10_000):
    s = Spy()
    return QueryBudgetedModel(s, D, 2, budget), s


def ids(cands):
    return [c.candidate_id for c in cands]


# ------------------------------------------------------------------ search space basics
def test_bounds_modes_match_reference_statistics():
    X = ref_matrix()
    s = space(X, bounds_mode="reference_minmax")
    assert np.array_equal(s.lower, X.min(0)) and np.array_equal(s.upper, X.max(0))
    s = space(X, bounds_mode="reference_quantile", lower_quantile=0.05, upper_quantile=0.95)
    assert np.allclose(s.lower, np.quantile(X, 0.05, axis=0)) and np.allclose(s.upper, np.quantile(X, 0.95, axis=0))
    s = space(X, bounds_mode="reference_minmax", margin=0.1)
    span = X.max(0) - X.min(0)
    assert np.allclose(s.lower, X.min(0) - 0.1 * span) and np.allclose(s.upper, X.max(0) + 0.1 * span)
    s = space(X, bounds_mode="absolute", absolute_lower=-2.0, absolute_upper=3.0)
    assert (s.lower == -2.0).all() and (s.upper == 3.0).all()
    assert s.feature_names == NAMES and s.dim == D and (s.lower <= s.upper).all()
    assert np.allclose(s.ref_mean, X.mean(0)) and np.allclose(s.ref_std, X.std(0, ddof=1))
    assert not s.lower.flags.writeable


def test_finite_value_and_shape_validation():
    s = space()
    assert s.validate_matrix(np.zeros(D)).shape == (1, D)
    for bad in (np.full((2, D), np.nan), np.full((1, D), np.inf), np.zeros((2, D + 1)), np.zeros((2, 2, D))):
        try:
            s.validate_matrix(bad); assert False
        except ValueError:
            pass
        try:
            s.project(bad); assert False
        except ValueError:
            pass
    X = ref_matrix()
    for args in ((np.where(np.eye(len(X), D, dtype=bool), np.nan, X), NAMES), (X[:1], NAMES), (X, NAMES[:-1]), (X, NAMES[:-1] + ("f0",))):
        try:
            SearchSpace.from_reference(args[0], args[1], scfg()); assert False
        except ValueError:
            pass


def test_project_clips_to_box_and_is_idempotent():
    s = space()
    wild = np.random.default_rng(1).normal(0, 50, (500, D))
    P = s.project(wild)
    assert (P >= s.lower).all() and (P <= s.upper).all() and s.contains(P).all()
    assert np.array_equal(s.project(P), P)
    inside = np.clip(ref_matrix()[:50], s.lower, s.upper)
    assert np.array_equal(s.project(inside), inside)                    # points already inside are untouched
    assert s.project(np.empty((0, D))).shape == (0, D)


def test_radius_constraint_projection():
    X = ref_matrix()
    s = space(X, radius_enabled=True, radius_max=1.5)
    wild = np.random.default_rng(2).normal(0, 20, (400, D))
    P = s.project(wild)
    assert (s.radius(P) <= 1.5 + 1e-9).all() and s.contains(P).all()
    assert (P >= s.lower).all() and (P <= s.upper).all()
    assert np.array_equal(s.project(P), P)
    small = s.ref_mean + 0.1 * (X[:30] - s.ref_mean)                      # well inside the ball
    assert np.array_equal(s.project(small), np.clip(small, s.lower, s.upper))
    assert (s.radius(wild) > 1.5).any()                                    # the constraint actually bound


def test_mahalanobis_constraint_projection_reuses_phase4_statistics():
    X = ref_matrix()
    s = space(X, maha_enabled=True, maha_quantile=0.9, maha_shrinkage="ledoit_wolf")
    m = MahalanobisValidity("ledoit_wolf", 0.1, 0.0, 1e-8).fit(X)
    assert np.isclose(s.maha_tau, np.quantile(m.distance(X), 0.9))
    assert np.allclose(s.mahalanobis(X[:20]), m.distance(X[:20]))
    wild = np.random.default_rng(3).normal(0, 20, (400, D))
    P = s.project(wild)
    assert (s.mahalanobis(P) <= s.maha_tau + 1e-9).all() and s.contains(P).all()
    assert (P >= s.lower).all() and (P <= s.upper).all() and np.array_equal(s.project(P), P)
    assert 0.05 < (s.mahalanobis(X) > s.maha_tau).mean() < 0.15            # ~10% of the reference is outside
    try:
        space(X).mahalanobis(X[:2]); assert False
    except ValueError:
        pass


def test_both_constraints_are_mutually_consistent_and_idempotent():
    s = space(radius_enabled=True, radius_max=2.0, maha_enabled=True, maha_quantile=0.95)
    P = s.project(np.random.default_rng(4).normal(0, 30, (500, D)))
    assert s.contains(P).all() and np.array_equal(s.project(P), P)


def test_bounds_widened_to_include_mean_when_constraints_enabled():
    X = ref_matrix().copy()
    X[:, 4] = 0.0
    X[:5, 4] = 1000.0                                                       # mean 6.25 >> 99th percentile (0)
    plain = space(X, upper_quantile=0.99)
    assert plain.upper[4] < plain.ref_mean[4] and plain.provenance["n_bounds_widened_to_include_mean"] == 0
    wide = space(X, upper_quantile=0.99, radius_enabled=True, radius_max=3.0)
    assert wide.upper[4] >= wide.ref_mean[4] and wide.provenance["n_bounds_widened_to_include_mean"] >= 1
    assert wide.contains(wide.project(np.random.default_rng(0).normal(0, 9, (200, D)))).all()


def test_serialization_round_trip_dict_and_json(tmp_path):
    s = space(radius_enabled=True, radius_max=2.0, maha_enabled=True, maha_quantile=0.95)
    for t in (SearchSpace.from_dict(json.loads(json.dumps(s.to_dict()))), SearchSpace.from_json(s.to_json(tmp_path / "ss.json"))):
        assert t.space_id == s.space_id and t.feature_names == s.feature_names
        for a in ("lower", "upper", "ref_mean", "ref_std", "maha_whiten"):
            assert np.array_equal(getattr(t, a), getattr(s, a))
        assert t.radius_max == s.radius_max and t.maha_tau == s.maha_tau and t.provenance == s.provenance
        w = np.random.default_rng(0).normal(0, 9, (50, D))
        assert np.array_equal(t.project(w), s.project(w))
        assert np.array_equal(UniformRandomGenerator(t).generate_matrix(20, seed=5), UniformRandomGenerator(s).generate_matrix(20, seed=5))
    d = s.to_dict()
    d["lower"][0] -= 1.0
    for bad in (d, {**s.to_dict(), "format": 99}):
        try:
            SearchSpace.from_dict(bad); assert False
        except ValueError:
            pass


# ------------------------------------------------------------------ uniform generator
def test_uniform_count_dimension_finite_and_bounds():
    s = space()
    g = UniformRandomGenerator(s)
    for n in (1, 7, 100):
        c = g.generate(n, seed=1)
        M = np.stack([x.x for x in c])
        assert len(c) == n and M.shape == (n, D) and np.isfinite(M).all()
        assert (M >= s.lower).all() and (M <= s.upper).all()
        assert all(x.check(NAMES) == "ok" for x in c)
    assert g.generate(0, seed=1) == [] and g.generate_matrix(0, seed=1).shape == (0, D)
    for bad_n in (-1, 1.5, True, "3"):
        try:
            g.generate(bad_n, seed=1); assert False
        except ValueError:
            pass


def test_uniform_determinism_batch_independence_and_seed_behaviour():
    g = UniformRandomGenerator(space())
    a = g.generate_matrix(30, seed=7)
    assert np.array_equal(a, g.generate_matrix(30, seed=7))
    assert np.array_equal(a, np.vstack([g.generate_matrix(11, seed=7), g.generate_matrix(19, seed=7, start_index=11)]))
    assert ids(g.generate(30, seed=7)) == ids(g.generate(30, seed=7))
    b = g.generate_matrix(30, seed=8)
    assert not np.allclose(a, b) and not (a == b).any(axis=1).any()          # different seed -> different rows
    assert not np.allclose(a, g.generate_matrix(30, seed=7, start_index=1000))
    assert len(set(ids(g.generate(30, seed=7)))) == 30
    try:
        g.generate(5); assert False                                         # seed is mandatory
    except TypeError:
        pass
    for bad in (-1, 1.5, True):
        try:
            g.generate(5, seed=bad); assert False
        except ValueError:
            pass


def test_uniform_is_roughly_uniform_inside_absolute_bounds():
    s = space(bounds_mode="absolute", absolute_lower=-1.0, absolute_upper=1.0)
    M = UniformRandomGenerator(s).generate_matrix(4000, seed=0)
    assert (np.abs(M.mean(0)) < 0.05).all() and (M.min(0) < -0.99).all() and (M.max(0) > 0.99).all()


# ------------------------------------------------------------------ local perturbation
def test_local_perturbation_per_feature_limit_when_only_bounds_active():
    X = ref_matrix()
    s = space(X)
    for dist, mult in (("gaussian", 2.5), ("uniform", 1.0)):
        g = LocalPerturbationGenerator(s, X, LocalPerturbationParams(scale=0.2, distribution=dist, truncation=2.5))
        c = g.generate(300, seed=3)
        M = np.stack([x.x for x in c])
        a = s.project(X[[x.meta["anchor_index"] for x in c]])
        assert (np.abs(M - a) <= mult * 0.2 * s.ref_std + 1e-9).all() and np.abs(M - a).max() > 0
        assert np.isfinite(M).all() and M.shape == (300, D) and s.contains(M).all()
    tiny = LocalPerturbationGenerator(s, X, LocalPerturbationParams(scale=1e-9)).generate_matrix(20, seed=1)
    assert np.allclose(tiny, s.project(X[[c.meta["anchor_index"] for c in LocalPerturbationGenerator(s, X, LocalPerturbationParams(scale=1e-9)).generate(20, seed=1)]]), atol=1e-6)


def test_local_perturbation_norm_limit_with_radius_constraint():
    X = ref_matrix()
    s = space(X, radius_enabled=True, radius_max=2.0)
    p = LocalPerturbationParams(scale=0.3, distribution="gaussian", truncation=3.0)
    c = LocalPerturbationGenerator(s, X, p).generate(300, seed=2)
    M = np.stack([x.x for x in c])
    a = s.project(X[[x.meta["anchor_index"] for x in c]])
    zdist = np.sqrt((((M - a) / s.ref_std) ** 2).sum(1))
    assert (zdist <= p.truncation * p.scale * np.sqrt(D) + 1e-6).all()          # projection is non-expansive here
    assert s.contains(M).all()


def test_local_perturbation_uses_real_anchors_and_respects_constraints():
    X = ref_matrix()
    s = space(X, maha_enabled=True, maha_quantile=0.99, radius_enabled=True, radius_max=3.0)
    g = LocalPerturbationGenerator(s, X, LocalPerturbationParams(scale=0.01))
    c = g.generate(100, seed=9)
    M = np.stack([x.x for x in c])
    idx = [x.meta["anchor_index"] for x in c]
    assert all(0 <= i < len(X) for i in idx) and len(set(idx)) > 20           # anchors spread over the reference
    a = s.project(X[idx])
    assert (np.sqrt((((M - a) / s.ref_std) ** 2).sum(1)) < 0.5).all()            # tiny scale -> stays at its anchor
    assert s.contains(M).all()


def test_local_determinism_batch_independence_and_dependence_on_inputs():
    X = ref_matrix()
    s = space(X)
    g = LocalPerturbationGenerator(s, X)
    a = g.generate_matrix(25, seed=4)
    assert np.array_equal(a, g.generate_matrix(25, seed=4))
    assert np.array_equal(a, np.vstack([g.generate_matrix(10, seed=4), g.generate_matrix(15, seed=4, start_index=10)]))
    assert not np.allclose(a, g.generate_matrix(25, seed=5))
    assert not np.allclose(a, LocalPerturbationGenerator(s, X, LocalPerturbationParams(scale=0.5)).generate_matrix(25, seed=4))
    assert not np.allclose(a, LocalPerturbationGenerator(s, ref_matrix(seed=1), None).generate_matrix(25, seed=4))
    assert g.describe()["anchor_fingerprint"] != LocalPerturbationGenerator(s, ref_matrix(seed=1)).describe()["anchor_fingerprint"]
    assert not np.allclose(UniformRandomGenerator(s).generate_matrix(25, seed=4), a)   # separate RNG streams per generator


def test_anchor_and_factory_validation():
    X = ref_matrix()
    s = space(X)
    for bad in (np.zeros((3, D + 1)), np.full((3, D), np.nan), np.empty((0, D)), np.zeros(D)[None, :][0, :2]):
        try:
            LocalPerturbationGenerator(s, bad); assert False
        except ValueError:
            pass
    cfg = scfg()
    assert isinstance(make_generator("uniform_random", s, cfg), UniformRandomGenerator)
    assert isinstance(make_generator("local_perturbation", s, cfg, anchors=X), LocalPerturbationGenerator)
    for args in (("local_perturbation", s, cfg), ("nope", s, cfg)):
        try:
            make_generator(*args); assert False
        except ValueError:
            pass
    assert set(GENERATOR_NAMES) == {"uniform_random", "local_perturbation"}
    for bad in (LocalPerturbationParams(scale=0), LocalPerturbationParams(distribution="x"), LocalPerturbationParams(truncation=-1)):
        try:
            bad.validate(); assert False
        except ValueError:
            pass


# ------------------------------------------------------------------ Phase 4 integration
def test_generated_candidates_are_phase4_candidates_with_deterministic_ids():
    X = ref_matrix()
    s = space(X)
    for g in (UniformRandomGenerator(s), LocalPerturbationGenerator(s, X)):
        c = g.generate(40, seed=3)
        assert all(isinstance(x, Candidate) for x in c) and all(x.origin == g.name for x in c)
        assert ids(c) == ids(g.generate(40, seed=3)) and len(set(ids(c))) == 40
        for i, x in enumerate(c):
            assert x.candidate_id == Candidate.from_array(x.x, NAMES, "anything").candidate_id   # id = f(vector, schema)
            assert x.meta["generator"] == g.name and x.meta["seed"] == 3 and x.meta["index"] == i
            assert set(x.meta) <= {"generator", "seed", "index", "anchor_index"}              # no model outputs
            assert not x.x.flags.writeable and x.to_model_input().shape == (1, D)
        assert Candidate.from_dict(json.loads(json.dumps(c[0].to_dict())), NAMES) == c[0]
        assert c[5].meta["index"] == 5 and g.generate(1, seed=3, start_index=5)[0] == c[5]


def test_phase4_validator_is_the_only_thing_that_queries_generated_candidates():
    X = ref_matrix()
    y = (X[:, 0] > 0).astype(int)
    ref = ReferenceData(X, y, None, NAMES, ("BENIGN", "ATTACK"), "unit-test-fixture")
    v = FailureValidator.from_reference(ref, ValidatorConfig(calibration_sample_size=200))
    s = space(X)
    bb, spy = box()
    cands = UniformRandomGenerator(s).generate(60, seed=1) + LocalPerturbationGenerator(s, X).generate(60, seed=1)
    assert bb.n_queries == 0 == spy.rows                                       # generation consumed nothing
    res = v.validate_batch(cands, bb, seed=1)
    n_valid = sum(r.candidate_valid for r in res)
    assert 0 < n_valid <= 120 and bb.n_queries == n_valid == spy.rows == sum(r.queried for r in res)
    assert [r.candidate_id for r in res] == ids(cands)


# ------------------------------------------------------------------ query-freeness / label-freeness
def test_zero_target_model_queries_even_if_every_model_entry_point_is_booby_trapped():
    X = ref_matrix()
    bb, spy = box()
    trapped = [(QueryBudgetedModel, n) for n in ("query", "predict_proba", "predict")] + [(TargetModel, "predict_proba_unbudgeted")]
    saved = {(c, n): getattr(c, n) for c, n in trapped}

    def boom(*a, **k):
        raise AssertionError("candidate generation touched the target model")
    try:
        for c, n in trapped:
            setattr(c, n, boom)
        s = space(X, radius_enabled=True, maha_enabled=True)
        u, l = UniformRandomGenerator(s), LocalPerturbationGenerator(s, X)
        u.generate(50, seed=1); l.generate(50, seed=1); u.generate_matrix(10, seed=2); l.generate_matrix(10, seed=2)
        s.project(np.random.default_rng(0).normal(size=(20, D))); s.to_dict()
    finally:
        for (c, n), f in saved.items():
            setattr(c, n, f)
    assert bb.n_queries == 0 == spy.rows == spy.calls


def test_generation_layer_does_not_import_target_model_or_xgboost():
    code = ("import sys; import src.search_space, src.candidate_generator; "
            "bad=[m for m in sys.modules if m in ('src.target_model','xgboost','src.validator')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


class PoisonedPrepared:
    """Exposes ONLY X_train and feature_names; touching anything else fails the test."""
    def __init__(self, X, names):
        self.X_train, self.feature_names = X, names

    def __getattr__(self, name):
        raise AssertionError(f"search space touched forbidden field: {name}")


def test_no_labels_available_runtime_signatures_and_static_scan():
    X = ref_matrix()
    s = SearchSpace.from_prepared(PoisonedPrepared(X, list(NAMES)), scfg())      # raises if y_*/cat_*/X_test/provenance read
    assert s.dim == D
    forbidden_params = {"y", "labels", "label", "categories", "cat", "model", "black_box", "target", "y_train", "predictions"}
    for fn in (SearchSpace.from_reference, SearchSpace.from_prepared, UniformRandomGenerator.__init__,
               LocalPerturbationGenerator.__init__, gen_mod.CandidateGenerator.generate,
               make_generator):
        assert not (set(inspect.signature(fn).parameters) & forbidden_params), fn
    assert not any(re.search(r"label|categor|^y|model|pred", f.name) for f in dataclasses.fields(SearchSpace))
    for mod in (space_mod, gen_mod):
        text = Path(inspect.getfile(mod)).read_text()
        code = re.sub(r"#.*", "", re.sub(r'""".*?"""', "", text, flags=re.S))
        assert not re.search(r"\b(y_train|cat_train|X_test|y_test|cat_test|ReferenceData|TargetModel|QueryBudgetedModel|"
                             r"target_model|xgboost|predict_proba|predict|FailureValidator)\b|\.clf\b|categories|labels?\b", code), mod.__name__


# ------------------------------------------------------------------ configuration
def test_config_loading_and_required_keys():
    cfg = load_config()
    s = cfg["search_space"]
    for key in ("seed", "bounds", "constraints", "generators", "numerical_tol"):
        assert key in s
    assert "radius" in s["constraints"] and "mahalanobis" in s["constraints"]
    assert "scale" in s["generators"]["local_perturbation"] and "default" in s["generators"]
    c = SearchSpaceConfig.from_cfg(cfg)
    assert c.seed == s["seed"] and c.bounds_mode == s["bounds"]["mode"] and c.local_scale == s["generators"]["local_perturbation"]["scale"]
    assert c.radius_enabled is False and c.maha_enabled is False and c.generator_default == "local_perturbation"
    o = SearchSpaceConfig.from_cfg(load_config(overrides={"search_space": {"generators": {"local_perturbation": {"scale": 0.25}}}}))
    assert o.local_scale == 0.25 and o.local_truncation == c.local_truncation


def test_invalid_config_values_are_rejected():
    nan = float("nan")
    bad = [{"seed": -1}, {"seed": 1.5}, {"numerical_tol": -1.0},
           {"bounds": {"mode": "bogus"}}, {"bounds": {"lower_quantile": 0.9, "upper_quantile": 0.1}},
           {"bounds": {"upper_quantile": 1.5}}, {"bounds": {"margin": -0.1}}, {"bounds": {"margin": nan}},
           {"bounds": {"mode": "absolute"}}, {"bounds": {"mode": "absolute", "absolute_lower": 2.0, "absolute_upper": 1.0}},
           {"constraints": {"radius": {"enabled": True, "max_radius": 0.0}}},
           {"constraints": {"mahalanobis": {"enabled": True, "quantile": 1.0}}},
           {"constraints": {"mahalanobis": {"enabled": True, "shrinkage": "bogus"}}},
           {"generators": {"default": "bogus"}}, {"generators": {"local_perturbation": {"scale": 0.0}}},
           {"generators": {"local_perturbation": {"scale": nan}}}, {"generators": {"local_perturbation": {"distribution": "x"}}},
           {"generators": {"local_perturbation": {"truncation": -1.0}}}]
    for ov in bad:
        try:
            SearchSpaceConfig.from_cfg(load_config(overrides={"search_space": ov})); assert False, ov
        except ValueError:
            pass
    try:
        SearchSpace.from_reference(ref_matrix(), NAMES, scfg(bounds_mode="bogus")); assert False
    except ValueError:
        pass


# ------------------------------------------------------------------ reproducibility
def test_reproducibility_for_fixed_seed_space_config_and_reference():
    kw = dict(radius_enabled=True, radius_max=2.5, maha_enabled=True, maha_quantile=0.97)
    X1, X2 = ref_matrix(), ref_matrix()
    s1, s2 = space(X1, **kw), space(X2, **kw)
    assert s1.space_id == s2.space_id and s1.provenance["reference_fingerprint"] == s2.provenance["reference_fingerprint"]
    for name in GENERATOR_NAMES:
        m1 = make_generator(name, s1, scfg(**kw), X1).generate_matrix(40, seed=11)
        m2 = make_generator(name, s2, scfg(**kw), X2).generate_matrix(40, seed=11)
        assert np.array_equal(m1, m2)
        assert not np.array_equal(m1, make_generator(name, s1, scfg(**kw), X1).generate_matrix(40, seed=12))
    s3 = space(ref_matrix(seed=1), **kw)
    assert s3.space_id != s1.space_id and s3.provenance["reference_fingerprint"] != s1.provenance["reference_fingerprint"]
    assert space(X1, **{**kw, "radius_max": 2.6}).space_id != s1.space_id


# ------------------------------------------------------------------ added during the Phase 5 audit
def _correlated_ref(seed, corr, n=600):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, D))
    for j in range(1, D):
        A[:, j] = corr * A[:, 0] + np.sqrt(1 - corr ** 2) * A[:, j]      # strongly correlated features
    A[:, 2] = A[:, 2] ** 3
    return A


def test_project_clips_BEFORE_shrinking_so_box_and_ellipsoid_hold_together():
    """Regression found by mutation testing: 'shrink toward the mean, THEN clip' leaves the Mahalanobis
    ellipsoid on correlated data (a coordinate-wise clip is not radial). Clip-first keeps the segment
    mean->point inside the (convex) box, so shrinking afterwards cannot leave it. This fixture
    (correlated features, box narrower than the ellipsoid) violated the constraint 265 times in
    32,400 points under the wrong order and 0 times under the correct one."""
    names_kw = [dict(maha_enabled=True, maha_quantile=0.9),
                dict(maha_enabled=True, maha_quantile=0.99, bounds_mode="reference_quantile", lower_quantile=0.2, upper_quantile=0.8),
                dict(maha_enabled=True, maha_quantile=0.5, bounds_mode="reference_quantile", lower_quantile=0.3,
                     upper_quantile=0.7, radius_enabled=True, radius_max=1.0)]
    n_box_bound = 0
    for seed in range(3):
        for corr in (0.0, 0.6, 0.95):
            X = _correlated_ref(seed, corr)
            for kw in names_kw:
                s = space(X, **kw)
                W = np.random.default_rng(seed + 100).normal(0, 25, (300, D))
                P = s.project(W)
                assert s.contains(P).all(), (seed, corr, kw)
                assert (P >= s.lower).all() and (P <= s.upper).all() and np.array_equal(s.project(P), P)
                n_box_bound += int((W != np.clip(W, s.lower, s.upper)).any(axis=1).sum())
    assert n_box_bound > 0                                   # the box really did bind in this fixture


def test_generator_validity_probe_script_runs_on_a_synthetic_fixture_and_is_model_and_test_free():
    import importlib.util
    from src.preprocessing import prepare_data
    spec = importlib.util.spec_from_file_location("probe_gv", ROOT / "scripts" / "probe_generator_validity.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg = load_config(overrides={"data": {"source": "synthetic-cic-schema", "max_rows": None,
                                          "synthetic": {"n_rows": 3000, "data_seed": 0}}})
    out = mod.probe(prepare_data(cfg), cfg, n=60, seed=1, cal_sample=200)
    assert len(out["rows"]) == 7 and "Not an experimental result" in out["NOTE"] and out["paper_eligible"] is False
    assert all(0.0 <= r["valid_rate"] <= 1.0 and r["n"] == 60 for r in out["rows"])
    text = (ROOT / "scripts" / "probe_generator_validity.py").read_text()
    code = re.sub(r"#.*", "", re.sub(r'""".*?"""', "", text, flags=re.S))
    assert not re.search(r"\b(X_test|y_test|cat_test|y_train|TargetModel|QueryBudgetedModel|train_target_model|xgboost)\b", code)
