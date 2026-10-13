"""Target model (XGBoost) and the QueryBudgetedModel black-box wrapper.

TWO SEPARATE ACCESS PATHS
-------------------------
* `TargetModel.predict_proba_unbudgeted` : training-side evaluation (test-set metrics). Never
  handed to a search engine.
* `TargetModel.black_box(budget)` -> `QueryBudgetedModel` : the ONLY object a search engine
  (Random, GP-BO, CMA-ES, Quantum) ever receives. It exposes inputs -> probabilities/predictions
  and counters; no labels, no trees, no gradients, no raw classifier.

QUERY ACCOUNTING (strict)
-------------------------
1 query = 1 input vector (1 row) evaluated by the target model.
* A batch of n rows costs exactly n queries.
* The budget is checked BEFORE the model is evaluated. If n > remaining, nothing is evaluated,
  nothing is charged, and `QueryBudgetExceeded` is raised (no partial batches).
* Invalid inputs (wrong shape, NaN/inf, empty) raise ValueError and are not charged.
* A query is charged only after the backend returned successfully.
* `query()` returns probabilities AND the argmax prediction for the same charge.
  `predict()` / `predict_proba()` are convenience wrappers and EACH charge their own call.
* There is deliberately no reset(): create a fresh wrapper for every search run.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from .utils import (config_hash, environment_report, get_logger, resolve_path, save_json)

log = get_logger("target_model")

DEV_MOCK_ENV = "HQFD_ALLOW_DEV_MOCK"


# ==========================================================================
# Query-budgeted black box
# ==========================================================================
class QueryBudgetExceeded(RuntimeError):
    def __init__(self, requested: int, remaining: float, budget: int):
        self.requested, self.remaining, self.budget = requested, remaining, budget
        super().__init__(f"Query budget exceeded: requested {requested} queries but only "
                         f"{remaining} of {budget} remain. Nothing was evaluated or charged.")


@dataclass(frozen=True)
class QueryResult:
    proba: np.ndarray      # (n, C) class probabilities
    pred: np.ndarray       # (n,) argmax class

    @property
    def n(self) -> int:
        return int(self.proba.shape[0])

    @property
    def confidence(self) -> np.ndarray:
        return self.proba.max(axis=1)


@runtime_checkable
class BlackBoxModel(Protocol):
    """Interface every SearchEngine codes against (Random, GP-BO, CMA-ES, Quantum)."""
    n_features: int
    n_classes: int
    budget: Optional[int]
    n_queries: int
    remaining: float
    exhausted: bool

    def query(self, X: np.ndarray) -> QueryResult: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def can_afford(self, n: int) -> bool: ...


class QueryBudgetedModel:
    def __init__(self, backend: Any, n_features: int, n_classes: int,
                 budget: Optional[int], name: str = "target"):
        if budget is not None and (isinstance(budget, bool) or int(budget) != budget or budget < 0):
            raise ValueError(f"budget must be a non-negative integer or None, got {budget!r}")
        self.__backend = backend              # name-mangled; not part of the public interface
        self.n_features, self.n_classes, self.name = int(n_features), int(n_classes), name
        self._budget = None if budget is None else int(budget)
        self._n_queries = 0
        self._n_calls = 0
        self._history: List[Tuple[int, int, int]] = []     # (call_id, rows, cumulative_queries)
        self._lock = threading.RLock()

    # ---- read-only counters -------------------------------------------------
    @property
    def budget(self) -> Optional[int]:
        return self._budget

    @property
    def n_queries(self) -> int:
        return self._n_queries

    @property
    def n_calls(self) -> int:
        return self._n_calls

    @property
    def remaining(self) -> float:
        return math.inf if self._budget is None else self._budget - self._n_queries

    @property
    def exhausted(self) -> bool:
        return self._budget is not None and self._n_queries >= self._budget

    @property
    def history(self) -> List[Tuple[int, int, int]]:
        return list(self._history)

    def can_afford(self, n: int) -> bool:
        return self.remaining >= n

    def snapshot(self) -> Dict[str, Any]:
        return {"name": self.name, "budget": self._budget, "n_queries": self._n_queries,
                "n_calls": self._n_calls, "remaining": None if self._budget is None else self.remaining,
                "exhausted": self.exhausted}

    # ---- queries ------------------------------------------------------------
    def _validate(self, X: Any) -> np.ndarray:
        A = np.asarray(X, dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if A.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got shape {A.shape}")
        if A.shape[0] == 0:
            raise ValueError("Empty query batch (0 rows).")
        if A.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {A.shape[1]}")
        if not np.isfinite(A).all():
            raise ValueError("Query contains NaN/inf; not evaluated and not charged.")
        return np.ascontiguousarray(A)

    def query(self, X: Any) -> QueryResult:
        A = self._validate(X)
        n = A.shape[0]
        with self._lock:
            if self._budget is not None and self._n_queries + n > self._budget:
                raise QueryBudgetExceeded(n, self.remaining, self._budget)
            P = np.asarray(self.__backend.predict_proba(A.copy()), dtype=np.float64)
            if P.shape != (n, self.n_classes) or not np.isfinite(P).all() \
                    or not np.allclose(P.sum(axis=1), 1.0, atol=1e-4):
                raise RuntimeError(f"Backend returned invalid probabilities (shape {P.shape}).")
            self._n_queries += n
            self._n_calls += 1
            self._history.append((self._n_calls, n, self._n_queries))
        return QueryResult(proba=P.copy(), pred=P.argmax(axis=1))

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.query(X).proba

    def predict(self, X: Any) -> np.ndarray:
        return self.query(X).pred


# ==========================================================================
# Target model container
# ==========================================================================
class DevMockClassifier:
    """DEVELOPMENT / UNIT-TEST ONLY (sklearn logistic regression). NOT a research model.
    Usable only when the env var HQFD_ALLOW_DEV_MOCK=1 is set; all its outputs are prefixed
    DEV_MOCK_ and flagged paper_eligible=False."""
    is_dev_mock = True

    def __init__(self, seed: int = 0):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(max_iter=300, random_state=seed)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


@dataclass
class TargetModel:
    clf: Any
    kind: str                      # 'xgboost' | 'dev-mock'
    n_features: int
    n_classes: int
    class_names: List[str]
    feature_names: List[str]
    params: Dict[str, Any]
    train_info: Dict[str, Any]

    @property
    def is_dev_mock(self) -> bool:
        return self.kind == "dev-mock"

    def predict_proba_unbudgeted(self, X: np.ndarray) -> np.ndarray:
        """TRAINING-SIDE EVALUATION ONLY (test metrics). Search engines must use black_box()."""
        return np.asarray(self.clf.predict_proba(np.asarray(X, dtype=np.float64)), dtype=np.float64)

    def black_box(self, budget: Optional[int], name: Optional[str] = None) -> QueryBudgetedModel:
        return QueryBudgetedModel(self.clf, self.n_features, self.n_classes, budget,
                                  name=name or f"{self.kind}-blackbox")

    # ---- persistence ----------------------------------------------------------
    def save(self, models_dir, tag: str) -> Dict[str, Path]:
        d = resolve_path(models_dir)
        d.mkdir(parents=True, exist_ok=True)
        if self.kind == "xgboost":
            model_path = d / f"{tag}.json"
            self.clf.save_model(str(model_path))
        else:
            model_path = d / f"{tag}.joblib"
            joblib.dump(self.clf, model_path)
        meta_path = save_json({"kind": self.kind, "n_features": self.n_features,
                               "n_classes": self.n_classes, "class_names": self.class_names,
                               "feature_names": self.feature_names, "params": self.params,
                               "train_info": self.train_info, "model_file": model_path.name},
                              d / f"{tag}_meta.json")
        return {"model": model_path, "meta": meta_path}

    @classmethod
    def load(cls, models_dir, tag: str) -> "TargetModel":
        import json
        d = resolve_path(models_dir)
        meta = json.load(open(d / f"{tag}_meta.json"))
        if meta["kind"] == "xgboost":
            XGBClassifier = _import_xgboost()
            clf = XGBClassifier()
            clf.load_model(str(d / meta["model_file"]))
        else:
            clf = joblib.load(d / meta["model_file"])
        return cls(clf, meta["kind"], meta["n_features"], meta["n_classes"], meta["class_names"],
                   meta["feature_names"], meta["params"], meta["train_info"])


# ==========================================================================
# Training
# ==========================================================================
def _import_xgboost():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except ImportError as e:
        raise ImportError("XGBoost is REQUIRED for the research target model. Install it with "
                          "`pip install xgboost` (or `pip install -r requirements.txt`). "
                          "No fallback model will be substituted.") from e


def train_target_model(cfg: Dict[str, Any], prepared, kind: Optional[str] = None,
                       seed: Optional[int] = None) -> TargetModel:
    """Train the target model on prepared.X_train. The test set is never used for training or
    early stopping; a validation slice is carved out of TRAIN (stratified, seeded)."""
    tm = cfg["target_model"]
    kind = kind or tm.get("type", "xgboost")
    seed = tm.get("seed", 42) if seed is None else seed
    Xtr, ytr = prepared.X_train, prepared.y_train
    C = len(prepared.class_names)

    if kind == "dev-mock":
        if os.environ.get(DEV_MOCK_ENV) != "1":
            raise PermissionError(f"dev-mock model is for unit tests only; set {DEV_MOCK_ENV}=1 to "
                                  "acknowledge. It must never be used for research results.")
        log.warning("*** DEV-MOCK MODEL (sklearn). NOT A RESEARCH MODEL. Outputs are not paper-eligible. ***")
        t0 = time.time()
        clf = DevMockClassifier(seed).fit(Xtr, ytr)
        info = {"seed": seed, "n_train_used": int(len(Xtr)), "train_seconds": time.time() - t0}
        return TargetModel(clf, kind, Xtr.shape[1], C, list(prepared.class_names),
                           list(prepared.feature_names), {"model": "LogisticRegression"}, info)
    if kind != "xgboost":
        raise ValueError(f"unknown target model type {kind!r}")

    XGBClassifier = _import_xgboost()
    params = dict(tm.get("params", {}))
    spw = params.pop("scale_pos_weight", None)
    if spw is not None and C == 2:
        params["scale_pos_weight"] = float(spw)
    params.update(random_state=int(seed),
                  eval_metric="logloss" if C == 2 else "mlogloss",
                  objective="binary:logistic" if C == 2 else "multi:softprob")
    esr = tm.get("early_stopping_rounds")
    vf = float(tm.get("val_fraction", 0.1))

    if esr and vf > 0:
        counts = np.bincount(ytr, minlength=C)
        strat = ytr if counts[counts > 0].min() >= 2 else None
        tr_i, va_i = train_test_split(np.arange(len(Xtr)), test_size=vf, random_state=seed, stratify=strat)
        clf = XGBClassifier(early_stopping_rounds=int(esr), **params)
        t0 = time.time()
        clf.fit(Xtr[tr_i], ytr[tr_i], eval_set=[(Xtr[va_i], ytr[va_i])], verbose=False)
        n_used, n_val = len(tr_i), len(va_i)
    else:
        clf = XGBClassifier(**params)
        t0 = time.time()
        clf.fit(Xtr, ytr, verbose=False)
        n_used, n_val = len(Xtr), 0
    info = {"seed": int(seed), "n_train_used": int(n_used), "n_val_used_for_early_stopping": int(n_val),
            "train_seconds": time.time() - t0,
            "best_iteration": int(getattr(clf, "best_iteration", clf.n_estimators - 1)),
            "early_stopping_rounds": esr, "val_fraction": vf if esr else 0.0,
            "xgboost_version": __import__("xgboost").__version__}
    params_out = {**params, "early_stopping_rounds": esr}
    log.info("XGBoost trained: best_iteration=%s, %.1fs", info["best_iteration"], info["train_seconds"])
    return TargetModel(clf, kind, Xtr.shape[1], C, list(prepared.class_names),
                       list(prepared.feature_names), params_out, info)


def proba_hash(P: np.ndarray) -> str:
    """Fingerprint of test-set probabilities (rounded to 6 d.p.) for reproducibility checks."""
    return hashlib.sha256(np.round(np.asarray(P, dtype=np.float64), 6).tobytes()).hexdigest()[:16]


# ==========================================================================
# Train + evaluate + save
# ==========================================================================
def train_and_evaluate(cfg: Dict[str, Any], prepared, kind: Optional[str] = None,
                       models_dir=None, results_dir=None,
                       check_reproducibility: bool = False) -> Dict[str, Any]:
    from .metrics import classification_report_dict
    from .visualization import plot_confusion_matrix
    import pandas as pd

    models_dir = models_dir or cfg["project"]["models_dir"]
    rdir = resolve_path(results_dir or cfg["project"]["results_dir"])
    for sub in ("raw", "tables", "figures"):
        (rdir / sub).mkdir(parents=True, exist_ok=True)

    model = train_target_model(cfg, prepared, kind)
    prov = prepared.provenance
    prefix = "DEV_MOCK_" if model.is_dev_mock else ""
    tag = f"{prefix}{model.kind}_{prov['source']}_{cfg['data']['task']}"
    paper_eligible = bool(prov.get("paper_eligible")) and not model.is_dev_mock
    if not paper_eligible:
        log.warning("Results of this run are NOT paper-eligible (dataset=%s, model=%s).",
                    prov["source"], model.kind)

    P = model.predict_proba_unbudgeted(prepared.X_test)
    report = classification_report_dict(prepared.y_test, P, prepared.class_names,
                                        categories=prepared.cat_test)
    phash = proba_hash(P)

    # persist + verify the saved artefact reproduces the in-memory model
    saved = model.save(models_dir, tag)
    reloaded = TargetModel.load(models_dir, tag)
    n_chk = min(2000, len(prepared.X_test))
    roundtrip_ok = bool(np.allclose(reloaded.predict_proba_unbudgeted(prepared.X_test[:n_chk]),
                                    P[:n_chk], atol=1e-6))
    if not roundtrip_ok:
        raise RuntimeError("Saved model does not reproduce in-memory predictions.")

    repro = None
    if check_reproducibility:
        P2 = train_target_model(cfg, prepared, kind).predict_proba_unbudgeted(prepared.X_test)
        repro = {"hash_run1": phash, "hash_run2": proba_hash(P2),
                 "max_abs_diff": float(np.abs(P - P2).max()), "identical": phash == proba_hash(P2)}
        log.info("Reproducibility check: identical=%s (max|diff|=%.2e)", repro["identical"], repro["max_abs_diff"])

    # tables / figure
    names = prepared.class_names
    cm = np.array(report["confusion_matrix"])
    pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names]) \
        .to_csv(rdir / "tables" / f"target_model_{tag}_confusion_matrix.csv")
    summary_row = {k: report.get(k) for k in ["accuracy", "precision", "recall", "f1", "roc_auc",
                                              "n_errors", "error_rate", "average"]}
    summary_row.update({f"high_conf_errors_{k}": v for k, v in report["high_confidence_errors"].items()})
    summary_row.update(tag=tag, n_test=report["n_samples"], paper_eligible=paper_eligible)
    pd.DataFrame([summary_row]).to_csv(rdir / "tables" / f"target_model_{tag}_metrics.csv", index=False)
    pd.DataFrame([{"category": c, **v} for c, v in report["per_category_accuracy"].items()]) \
        .to_csv(rdir / "tables" / f"target_model_{tag}_per_category.csv", index=False)
    fig_path = plot_confusion_matrix(cm, names, rdir / "figures" / f"target_model_{tag}_confusion_matrix.png",
                                     title=f"Target model confusion matrix ({model.kind}, {prov['source']})",
                                     dev_watermark=not paper_eligible)

    result = {
        "tag": tag, "kind": model.kind, "is_dev_mock": model.is_dev_mock,
        "paper_eligible": paper_eligible,
        "data": {"provenance": prov, "report": prepared.report, "task": cfg["data"]["task"],
                 "n_features": model.n_features, "split_seed": prepared.report.get("split_seed")},
        "model": {"params": model.params, "train_info": model.train_info},
        "metrics": report, "test_proba_sha256": phash, "saved_model_roundtrip_verified": roundtrip_ok,
        "reproducibility_check": repro, "config_hash": config_hash(cfg), "config": cfg,
        "environment": environment_report(),
        "files": {"model": str(saved["model"]), "meta": str(saved["meta"]), "figure": str(fig_path)},
    }
    save_json(result, rdir / "raw" / f"target_model_{tag}.json")
    return result
