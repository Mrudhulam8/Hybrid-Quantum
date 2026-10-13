"""Phase 3 tests. Query-accounting tests use a SPY backend (independent of the wrapper's own
counter) and a DEV-MOCK model. No research result is produced or asserted from the mock.
Tests marked XGB run only where xgboost is installed (your machine / Colab)."""
import json
import os
import threading
import unittest

import numpy as np

from src.metrics import classification_report_dict
from src.preprocessing import prepare_data
from src.target_model import (BlackBoxModel, DEV_MOCK_ENV, QueryBudgetExceeded, QueryBudgetedModel,
                              QueryResult, TargetModel, proba_hash, train_and_evaluate,
                              train_target_model)
from src.utils import load_config


def _has_xgb():
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


class Spy:
    """Backend that records how many rows it was REALLY asked to evaluate."""
    def __init__(self, C=2):
        self.rows, self.calls, self.C, self._l = 0, 0, C, threading.Lock()

    def predict_proba(self, X):
        with self._l:
            self.rows += len(X); self.calls += 1
        p = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack([1 - p, p]) if self.C == 2 else np.full((len(X), self.C), 1 / self.C)


def _bb(budget, d=4, spy=None):
    spy = spy or Spy()
    return QueryBudgetedModel(spy, d, 2, budget), spy


def _prep(n=6000):
    cfg = load_config(overrides={"data": {"source": "synthetic-cic-schema", "max_rows": None,
                                          "synthetic": {"n_rows": n, "data_seed": 0}}})
    return cfg, prepare_data(cfg)


# ---------------------------------------------------------------- query accounting
def test_every_row_is_counted_exactly_once_against_independent_spy():
    bb, spy = _bb(1000)
    bb.query(np.zeros((5, 4)))
    bb.query(np.ones((1, 4)))
    bb.predict(np.zeros((3, 4)))
    bb.predict_proba(np.zeros((2, 4)))
    assert bb.n_queries == 11 == spy.rows          # wrapper counter == rows the model really saw
    assert bb.n_calls == 4 == spy.calls
    assert bb.history == [(1, 5, 5), (2, 1, 6), (3, 3, 9), (4, 2, 11)]
    assert bb.remaining == 989


def test_query_returns_proba_and_pred_for_a_single_charge():
    bb, spy = _bb(10)
    r = bb.query(np.array([[2.0, 0, 0, 0], [-2.0, 0, 0, 0]]))
    assert isinstance(r, QueryResult) and r.n == 2 and bb.n_queries == 2 == spy.rows
    assert r.pred.tolist() == [1, 0] and np.allclose(r.proba.sum(1), 1)
    assert np.allclose(r.confidence, r.proba.max(1))


def test_1d_input_is_one_query():
    bb, spy = _bb(10)
    bb.query(np.zeros(4))
    assert bb.n_queries == 1 == spy.rows


def test_exceeding_budget_raises_and_nothing_is_evaluated_or_charged():
    bb, spy = _bb(10)
    bb.query(np.zeros((7, 4)))
    try:
        bb.query(np.zeros((4, 4)))               # 7 + 4 > 10
        assert False, "should have raised"
    except QueryBudgetExceeded as e:
        assert e.requested == 4 and e.remaining == 3 and e.budget == 10
    assert bb.n_queries == 7 == spy.rows          # no partial evaluation, no charge
    assert spy.calls == 1 and not bb.exhausted
    bb.query(np.zeros((3, 4)))                    # exactly fills the budget
    assert bb.n_queries == 10 and bb.exhausted and bb.remaining == 0


def test_budget_boundary_and_post_exhaustion():
    bb, spy = _bb(10)
    bb.query(np.zeros((10, 4)))                   # batch == budget is allowed
    assert bb.exhausted
    for fn in (bb.query, bb.predict, bb.predict_proba):
        try:
            fn(np.zeros((1, 4)))
            assert False
        except QueryBudgetExceeded:
            pass
    assert bb.n_queries == 10 == spy.rows


def test_zero_budget_and_unlimited_budget():
    bb, spy = _bb(0)
    try:
        bb.query(np.zeros((1, 4))); assert False
    except QueryBudgetExceeded:
        pass
    assert spy.rows == 0
    ub, spy2 = _bb(None)
    ub.query(np.zeros((500, 4)))
    assert ub.n_queries == 500 == spy2.rows and ub.remaining == float("inf") and not ub.exhausted


def test_invalid_budget_rejected():
    for bad in (-1, 1.5, True, "10"):
        try:
            QueryBudgetedModel(Spy(), 4, 2, bad); assert False, bad
        except (ValueError, TypeError):
            pass


def test_invalid_inputs_are_rejected_and_not_charged():
    bb, spy = _bb(100)
    bad = [np.zeros((2, 3)), np.zeros((0, 4)), np.full((2, 4), np.nan), np.full((1, 4), np.inf),
           np.zeros((2, 2, 4))]
    for X in bad:
        try:
            bb.query(X); assert False
        except ValueError:
            pass
    assert bb.n_queries == 0 == spy.rows == bb.n_calls


def test_failed_backend_call_is_not_charged():
    class Boom:
        def predict_proba(self, X): raise RuntimeError("model crashed")
    bb = QueryBudgetedModel(Boom(), 4, 2, 10)
    try:
        bb.query(np.zeros((2, 4))); assert False
    except RuntimeError:
        pass
    assert bb.n_queries == 0


def test_invalid_backend_output_is_rejected():
    class Bad:
        def predict_proba(self, X): return np.full((len(X), 2), 0.9)   # rows sum to 1.8
    bb = QueryBudgetedModel(Bad(), 4, 2, 10)
    try:
        bb.query(np.zeros((1, 4))); assert False
    except RuntimeError:
        pass
    assert bb.n_queries == 0


def test_independent_counters_per_wrapper():
    cfg, prep = _prep()
    os.environ[DEV_MOCK_ENV] = "1"
    try:
        tm = train_target_model(cfg, prep, kind="dev-mock")
    finally:
        os.environ.pop(DEV_MOCK_ENV)
    a, b = tm.black_box(50), tm.black_box(50)
    a.query(prep.X_test[:20])
    assert a.n_queries == 20 and b.n_queries == 0


def test_thread_safe_exact_accounting_under_contention():
    bb, spy = _bb(500)
    ok, rej = [0], [0]
    lk = threading.Lock()

    def worker():
        for _ in range(100):
            try:
                bb.query(np.zeros((1, 4)))
                with lk: ok[0] += 1
            except QueryBudgetExceeded:
                with lk: rej[0] += 1
    ts = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert ok[0] == 500 and rej[0] == 300 and bb.n_queries == 500 == spy.rows


def test_black_box_exposes_no_model_or_label_access():
    bb, _ = _bb(10)
    public = [a for a in dir(bb) if not a.startswith("_")]
    assert not any(w in a.lower() for a in public for w in ("backend", "label", "model", "clf", "tree", "booster", "gradient"))
    assert isinstance(bb, BlackBoxModel)          # conforms to the shared SearchEngine interface
    r = bb.query(np.zeros((1, 4)))
    r.proba[:] = 0                                # mutating the result must not affect the model
    assert np.allclose(bb.query(np.zeros((1, 4))).proba.sum(1), 1)


# ---------------------------------------------------------------- metrics
def test_metrics_match_hand_computation():
    y = np.array([0, 0, 1, 1, 1])
    P = np.array([[.9, .1], [.4, .6], [.2, .8], [.7, .3], [.05, .95]])
    r = classification_report_dict(y, P, ["BENIGN", "ATTACK"], categories=np.array(list("aabbb")))
    assert r["confusion_matrix"] == [[1, 1], [1, 2]]                 # rows = true, cols = predicted
    assert np.isclose(r["accuracy"], 0.6) and np.isclose(r["precision"], 2 / 3)
    assert np.isclose(r["recall"], 2 / 3) and np.isclose(r["f1"], 2 / 3)
    assert np.isclose(r["roc_auc"], 5 / 6)
    assert r["n_errors"] == 2 and r["per_category_accuracy"]["a"]["accuracy"] == 0.5


def test_high_confidence_errors_and_single_class_auc():
    y = np.array([1, 1, 0])
    P = np.array([[.99, .01], [.6, .4], [.5, .5]])                   # wrong@.99, wrong@.6, argmax->0 ok
    r = classification_report_dict(y, P, ["B", "A"])
    assert r["high_confidence_errors"][">=0.9"] == 1 and r["n_errors"] == 2
    r1 = classification_report_dict(np.zeros(3, int), P, ["B", "A"])
    assert r1["roc_auc"] is None


# ---------------------------------------------------------------- training / saving (DEV MOCK)
def test_dev_mock_is_blocked_without_explicit_env_flag():
    cfg, prep = _prep(3000)
    os.environ.pop(DEV_MOCK_ENV, None)
    try:
        train_target_model(cfg, prep, kind="dev-mock"); assert False
    except PermissionError:
        pass


def test_xgboost_missing_raises_instead_of_silently_falling_back():
    if _has_xgb():
        raise unittest.SkipTest("xgboost installed; this guard is only testable without it")
    cfg, prep = _prep(3000)
    try:
        train_target_model(cfg, prep, kind="xgboost"); assert False
    except ImportError as e:
        assert "No fallback" in str(e)


def test_train_evaluate_save_pipeline_with_dev_mock_is_flagged_and_reproducible(tmp_path):
    cfg, prep = _prep()
    os.environ[DEV_MOCK_ENV] = "1"
    try:
        r1 = train_and_evaluate(cfg, prep, kind="dev-mock", models_dir=tmp_path / "m",
                                results_dir=tmp_path / "r", check_reproducibility=True)
    finally:
        os.environ.pop(DEV_MOCK_ENV)
    assert r1["tag"].startswith("DEV_MOCK_") and r1["is_dev_mock"] and r1["paper_eligible"] is False
    assert r1["saved_model_roundtrip_verified"] and r1["reproducibility_check"]["identical"]
    for p in [tmp_path / "m" / f"{r1['tag']}.joblib", tmp_path / "m" / f"{r1['tag']}_meta.json",
              tmp_path / "r" / "raw" / f"target_model_{r1['tag']}.json",
              tmp_path / "r" / "tables" / f"target_model_{r1['tag']}_metrics.csv",
              tmp_path / "r" / "tables" / f"target_model_{r1['tag']}_confusion_matrix.csv",
              tmp_path / "r" / "tables" / f"target_model_{r1['tag']}_per_category.csv",
              tmp_path / "r" / "figures" / f"target_model_{r1['tag']}_confusion_matrix.png"]:
        assert p.exists() and p.stat().st_size > 0, p
    saved = json.load(open(tmp_path / "r" / "raw" / f"target_model_{r1['tag']}.json"))
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]:
        assert k in saved["metrics"]
    assert saved["data"]["provenance"]["source"] == "synthetic-cic-schema"
    assert saved["data"]["split_seed"] == 42 and "config_hash" in saved
    # loaded model reproduces predictions and works through the budgeted wrapper
    tm = TargetModel.load(tmp_path / "m", r1["tag"])
    bb = tm.black_box(25)
    out = bb.query(prep.X_test[:25])
    assert proba_hash(tm.predict_proba_unbudgeted(prep.X_test)) == r1["test_proba_sha256"]
    assert bb.n_queries == 25 and out.proba.shape == (25, 2) and bb.exhausted


def test_fixed_split_makes_training_data_identical_across_runs():
    cfg, _ = _prep()
    a, b = prepare_data(cfg), prepare_data(cfg)
    assert np.array_equal(a.X_train, b.X_train) and np.array_equal(a.y_test, b.y_test)


# ---------------------------------------------------------------- XGB-only tests (your machine)
def _small_xgb_cfg():
    cfg, prep = _prep(8000)
    cfg["target_model"]["params"]["n_estimators"] = 40
    cfg["target_model"]["params"]["n_jobs"] = 1
    return cfg, prep


def test_XGB_train_predict_budget_roundtrip(tmp_path):
    if not _has_xgb():
        raise unittest.SkipTest("xgboost not installed here")
    cfg, prep = _small_xgb_cfg()
    r = train_and_evaluate(cfg, prep, kind="xgboost", models_dir=tmp_path / "m", results_dir=tmp_path / "r")
    assert r["kind"] == "xgboost" and not r["is_dev_mock"] and r["saved_model_roundtrip_verified"]
    assert r["paper_eligible"] is False                   # synthetic data -> never paper-eligible
    assert r["metrics"]["accuracy"] > 0.8
    tm = TargetModel.load(tmp_path / "m", r["tag"])
    bb = tm.black_box(10)
    res = bb.query(prep.X_test[:10])
    assert res.proba.shape == (10, 2) and bb.exhausted
    try:
        bb.query(prep.X_test[:1]); assert False
    except QueryBudgetExceeded:
        pass


def test_XGB_training_is_reproducible():
    if not _has_xgb():
        raise unittest.SkipTest("xgboost not installed here")
    cfg, prep = _small_xgb_cfg()
    p1 = train_target_model(cfg, prep).predict_proba_unbudgeted(prep.X_test)
    p2 = train_target_model(cfg, prep).predict_proba_unbudgeted(prep.X_test)
    assert proba_hash(p1) == proba_hash(p2)
