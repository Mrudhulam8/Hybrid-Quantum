"""Metrics.

Section 1 (Phase 3): classification metrics for the target model.
Section 2 (Phase 11): Failure Discovery Efficiency and search metrics.   [not yet implemented]
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)


def classification_report_dict(y_true: np.ndarray, proba: np.ndarray, class_names: Sequence[str],
                               categories: Optional[np.ndarray] = None,
                               high_conf: Sequence[float] = (0.9, 0.99)) -> Dict[str, Any]:
    """Accuracy / precision / recall / F1 / ROC-AUC / confusion matrix (+ error diagnostics).

    Binary: precision/recall/F1 are for the positive class (index 1, 'ATTACK').
    Multiclass: macro and weighted averages; ROC-AUC is one-vs-rest macro.
    Confusion matrix: rows = true class, columns = predicted class.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    C = proba.shape[1]
    assert C == len(class_names), "proba columns must match class_names"
    pred = proba.argmax(axis=1)
    labels = list(range(C))
    p, r, f, sup = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    out: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "class_names": list(class_names),
        "accuracy": float(accuracy_score(y_true, pred)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
        "per_class": {class_names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                       "f1": float(f[i]), "support": int(sup[i])} for i in labels},
    }
    if C == 2:
        out.update(precision=float(p[1]), recall=float(r[1]), f1=float(f[1]), average="binary(pos=1)")
        out["roc_auc"] = (float(roc_auc_score(y_true, proba[:, 1]))
                          if len(np.unique(y_true)) == 2 else None)
    else:
        pm, rm, fm, _ = precision_recall_fscore_support(y_true, pred, labels=labels, average="macro",
                                                        zero_division=0)
        out.update(precision=float(pm), recall=float(rm), f1=float(fm), average="macro")
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                                 labels=labels))
        except ValueError:
            out["roc_auc"] = None
    wrong, conf = pred != y_true, proba.max(axis=1)
    out["n_errors"] = int(wrong.sum())
    out["error_rate"] = float(wrong.mean())
    out["high_confidence_errors"] = {f">={t}": int((wrong & (conf >= t)).sum()) for t in high_conf}
    if categories is not None:
        cats = np.asarray(categories)
        out["per_category_accuracy"] = {
            str(c): {"n": int((cats == c).sum()),
                     "accuracy": float((pred[cats == c] == y_true[cats == c]).mean())}
            for c in sorted(set(cats))}
    return out
