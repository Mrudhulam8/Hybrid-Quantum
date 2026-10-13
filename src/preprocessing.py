"""Cleaning, subsampling, splitting, and an INVERTIBLE, train-only-fitted preprocessor.

Design points that matter later in the project
----------------------------------------------
* Duplicates are removed BEFORE the split, so no identical flow sits in both train and test.
* Imputation medians, constant-column removal and scaler statistics are fitted on TRAIN only.
* `Preprocessor.inverse_transform` maps scaled vectors back to original feature units, so a
  candidate produced by a search engine can be reported as a real flow-feature vector.
* The data split is controlled by `data.split_seed` (fixed), NOT by the search seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler

from .data_loader import BENIGN, RawDataset, load_dataset
from .utils import config_hash, get_logger, resolve_path, save_json

log = get_logger("preprocessing")


# --------------------------------------------------------------------------
# Cleaning / subsampling
# --------------------------------------------------------------------------
def clean_frame(X: pd.DataFrame, category: np.ndarray, drop_duplicates: bool = True):
    """Replace +/-inf by NaN, drop all-NaN columns and (optionally) exact duplicate rows."""
    rep: Dict[str, Any] = {"rows_in": int(len(X))}
    X = X.copy()
    arr = X.to_numpy()
    n_inf = int(np.isinf(arr).sum())
    X = X.replace([np.inf, -np.inf], np.nan)
    rep["inf_replaced"] = n_inf
    rep["nan_cells"] = int(X.isna().sum().sum())
    allnan = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=allnan)
    rep["all_nan_columns_dropped"] = allnan
    if drop_duplicates:
        key = X.copy()
        key["__cat__"] = category
        keep = ~key.duplicated(keep="first").to_numpy()
        X, category = X[keep].reset_index(drop=True), category[keep]
    rep["duplicates_removed"] = int(rep["rows_in"] - len(X))
    rep["rows_out"] = int(len(X))
    return X, np.asarray(category), rep


def stratified_subsample(category: np.ndarray, max_rows: Optional[int],
                         rng: np.random.Generator, min_per_class: int = 100) -> np.ndarray:
    """Row indices of a subsample that (1) keeps up to `min_per_class` rows of every class
    (so rare attacks survive) and (2) fills the rest proportionally. Caveat: rare classes end
    up slightly over-represented relative to the natural class prior."""
    n = len(category)
    if max_rows is None or n <= max_rows:
        return np.arange(n)
    cats, inv, counts = np.unique(category, return_inverse=True, return_counts=True)
    keep = np.minimum(counts, min_per_class)
    remaining = max_rows - int(keep.sum())
    if remaining > 0:
        extra = counts - keep
        add = np.minimum(np.floor(remaining * extra / extra.sum()).astype(int), extra)
        keep = keep + add
    idx = [rng.choice(np.flatnonzero(inv == k), size=int(keep[k]), replace=False)
           for k in range(len(cats))]
    return np.sort(np.concatenate(idx))


# --------------------------------------------------------------------------
# Preprocessor
# --------------------------------------------------------------------------
def _slog1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _inv_slog1p(z):
    return np.sign(z) * np.expm1(np.abs(z))


class Preprocessor:
    """impute(median) -> signed log1p (optional) -> drop constant cols -> scale."""

    def __init__(self, scaler: str = "standard", log_transform: bool = True):
        self.scaler_name, self.log_transform = scaler, log_transform

    def _make_scaler(self):
        if self.scaler_name == "standard":
            return StandardScaler()
        if self.scaler_name == "minmax":
            return MinMaxScaler()
        if self.scaler_name == "none":
            return FunctionTransformer(None, inverse_func=None, validate=False)
        raise ValueError(f"unknown scaler {self.scaler_name}")

    def _prep(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.reindex(columns=self.columns_in_)
            A = X.to_numpy(dtype=np.float64, copy=True)
        else:
            A = np.array(X, dtype=np.float64, copy=True)
        A[~np.isfinite(A)] = np.nan
        miss = np.isnan(A)
        if miss.any():
            A[miss] = np.take(self.medians_, np.where(miss)[1])
        return _slog1p(A) if self.log_transform else A

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.columns_in_ = list(X.columns)
        A = X.to_numpy(dtype=np.float64, copy=True)
        A[~np.isfinite(A)] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(A, axis=0)
        self.medians_ = np.where(np.isnan(med), 0.0, med)
        B = self._prep(X)
        self.keep_mask_ = B.std(axis=0) > 1e-12
        self.feature_names_ = [c for c, k in zip(self.columns_in_, self.keep_mask_) if k]
        self.scaler_ = self._make_scaler().fit(B[:, self.keep_mask_])
        return self

    def transform(self, X) -> np.ndarray:
        return self.scaler_.transform(self._prep(X)[:, self.keep_mask_])

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Scaled -> original units (kept columns only)."""
        B = self.scaler_.inverse_transform(np.asarray(Z, dtype=np.float64))
        return _inv_slog1p(B) if self.log_transform else B

    def summary(self) -> Dict[str, Any]:
        return {"scaler": self.scaler_name, "log_transform": self.log_transform,
                "n_features_in": len(self.columns_in_), "n_features_kept": len(self.feature_names_),
                "constant_columns_dropped": [c for c, k in zip(self.columns_in_, self.keep_mask_) if not k]}


# --------------------------------------------------------------------------
# End-to-end preparation
# --------------------------------------------------------------------------
@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray            # per cfg task (binary 0/1, or class index)
    y_test: np.ndarray
    cat_train: np.ndarray          # string category per row (attack type / BENIGN)
    cat_test: np.ndarray
    feature_names: List[str]
    class_names: List[str]
    preprocessor: Preprocessor
    provenance: Dict[str, Any]
    report: Dict[str, Any]

    @property
    def y_train_binary(self) -> np.ndarray:
        return (self.cat_train != BENIGN).astype(np.int8)

    @property
    def y_test_binary(self) -> np.ndarray:
        return (self.cat_test != BENIGN).astype(np.int8)

    def summary_table(self) -> pd.DataFrame:
        names = sorted(set(self.cat_train) | set(self.cat_test))
        rows = [{"category": c, "train": int((self.cat_train == c).sum()),
                 "test": int((self.cat_test == c).sum())} for c in names]
        df = pd.DataFrame(rows)
        df["total"] = df["train"] + df["test"]
        df["fraction"] = (df["total"] / df["total"].sum()).round(5)
        return df.sort_values("total", ascending=False).reset_index(drop=True)


def prepare_data(cfg: Dict[str, Any], raw: Optional[RawDataset] = None,
                 split_seed: Optional[int] = None) -> PreparedData:
    d = cfg["data"]
    seed = d.get("split_seed", 42) if split_seed is None else split_seed
    rng = np.random.default_rng(seed)
    raw = raw if raw is not None else load_dataset(cfg)

    X, cat, rep = clean_frame(raw.X, raw.category, d.get("drop_duplicates", True))
    idx = stratified_subsample(cat, d.get("max_rows"), rng, d.get("min_rows_per_class", 100))
    X, cat = X.iloc[idx].reset_index(drop=True), cat[idx]
    rep["rows_after_subsample"] = int(len(X))

    counts = pd.Series(cat).value_counts()
    strat = np.where(pd.Series(cat).map(counts).to_numpy() >= 2, cat, "__RARE__")
    if (strat == "__RARE__").sum() < 2:
        strat = None
    tr, te = train_test_split(np.arange(len(X)), test_size=d.get("test_size", 0.3),
                              random_state=seed, stratify=strat)
    pre = Preprocessor(d.get("scaler", "standard"), d.get("log_transform", True)).fit(X.iloc[tr])
    Xtr, Xte = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te])
    ctr, cte = cat[tr], cat[te]

    if d.get("task", "binary") == "binary":
        class_names = [BENIGN, "ATTACK"]
        ytr, yte = (ctr != BENIGN).astype(np.int64), (cte != BENIGN).astype(np.int64)
    else:
        class_names = sorted(set(cat))
        m = {c: i for i, c in enumerate(class_names)}
        ytr = np.array([m[c] for c in ctr]); yte = np.array([m[c] for c in cte])

    rep.update({"split_seed": seed, "n_train": int(len(tr)), "n_test": int(len(te)),
                "task": d.get("task", "binary"), "preprocessor": pre.summary(),
                "data_config_hash": config_hash(d)})
    prov = dict(raw.provenance)
    if prov.get("paper_eligible") is False:
        log.warning("Prepared data comes from a NON-paper-eligible source (%s).", prov["source"])
    log.info("Prepared %s: train=%d test=%d features=%d attack_rate_train=%.4f",
             prov["source"], len(tr), len(te), Xtr.shape[1], float((ctr != BENIGN).mean()))
    return PreparedData(Xtr, Xte, ytr, yte, ctr, cte, pre.feature_names_, class_names, pre, prov, rep)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_prepared(p: PreparedData, out_dir, tag: str) -> Path:
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{tag}.npz", X_train=p.X_train, X_test=p.X_test,
                        y_train=p.y_train, y_test=p.y_test,
                        cat_train=p.cat_train.astype(str), cat_test=p.cat_test.astype(str))
    joblib.dump(p.preprocessor, out / f"{tag}_preprocessor.joblib")
    save_json({"provenance": p.provenance, "report": p.report,
               "feature_names": p.feature_names, "class_names": p.class_names}, out / f"{tag}_meta.json")
    return out / f"{tag}.npz"


def load_prepared(out_dir, tag: str) -> PreparedData:
    out = resolve_path(out_dir)
    z = np.load(out / f"{tag}.npz", allow_pickle=False)
    import json
    meta = json.load(open(out / f"{tag}_meta.json"))
    return PreparedData(z["X_train"], z["X_test"], z["y_train"], z["y_test"],
                        z["cat_train"], z["cat_test"], meta["feature_names"], meta["class_names"],
                        joblib.load(out / f"{tag}_preprocessor.joblib"),
                        meta["provenance"], meta["report"])
