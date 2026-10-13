"""Candidate: one input vector in the EXACT feature space (scaled, ordered) the target model uses.

* feature names/order are preserved and checked against the expected schema,
* `candidate_id` is deterministic (hash of schema + exact float64 values; independent of
  `origin`/`meta`, so the same vector proposed by two methods gets the same id),
* malformed candidates (wrong length, NaN/inf) can be CONSTRUCTED (a search method may propose
  them) but `check()` reports them, so the validator rejects them without querying the model.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

OK = "ok"
BAD_DIMENSION = "bad_dimension"
BAD_SCHEMA = "bad_feature_schema"
NON_FINITE = "non_finite"


def schema_hash(feature_names: Sequence[str]) -> str:
    return hashlib.sha256("\x1f".join(feature_names).encode()).hexdigest()[:8]


@dataclass(frozen=True, eq=False)
class Candidate:
    x: np.ndarray
    feature_names: Tuple[str, ...]
    origin: str = ""                       # e.g. 'random', 'cmaes', 'quantum' (not part of the id)
    meta: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: str = field(init=False)

    def __post_init__(self):
        a = np.array(self.x, dtype=np.float64, copy=True)
        if a.ndim != 1:
            raise ValueError(f"Candidate.x must be 1-D, got shape {a.shape}")
        a = a + 0.0                                    # -0.0 -> 0.0
        a[np.isnan(a)] = np.nan                        # canonical NaN bit pattern
        a.setflags(write=False)
        object.__setattr__(self, "x", a)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        digest = hashlib.sha256(schema_hash(self.feature_names).encode()
                                + a.astype("<f8").tobytes()).hexdigest()[:16]
        object.__setattr__(self, "candidate_id", digest)

    # identity = id
    def __eq__(self, other):
        return isinstance(other, Candidate) and self.candidate_id == other.candidate_id

    def __hash__(self):
        return hash(self.candidate_id)

    # ---- construction ---------------------------------------------------------
    @classmethod
    def from_array(cls, x, feature_names: Sequence[str], origin: str = "",
                   meta: Optional[Mapping[str, Any]] = None, strict: bool = False) -> "Candidate":
        c = cls(np.asarray(x, dtype=np.float64), tuple(feature_names), origin, dict(meta or {}))
        if strict:
            reason = c.check(feature_names)
            if reason != OK:
                raise ValueError(f"Invalid candidate: {reason}")
        return c

    # ---- validation -----------------------------------------------------------
    def check(self, expected_feature_names: Sequence[str]) -> str:
        """'ok' | 'bad_dimension' | 'bad_feature_schema' | 'non_finite' (cheap; no model use)."""
        expected = tuple(expected_feature_names)
        if self.x.shape[0] != len(expected):
            return BAD_DIMENSION
        if self.feature_names != expected:
            return BAD_SCHEMA
        if not np.isfinite(self.x).all():
            return NON_FINITE
        return OK

    # ---- conversion -----------------------------------------------------------
    def to_model_input(self) -> np.ndarray:
        """(1, d) float64 array: the exact format accepted by QueryBudgetedModel."""
        return self.x.reshape(1, -1).copy()

    def to_original_units(self, preprocessor) -> Dict[str, float]:
        """Map back to original CIC feature units (for reporting only)."""
        raw = preprocessor.inverse_transform(self.x.reshape(1, -1))[0]
        return dict(zip(self.feature_names, map(float, raw)))

    # ---- serialisation --------------------------------------------------------
    def to_dict(self, include_names: bool = False) -> Dict[str, Any]:
        d = {"candidate_id": self.candidate_id, "schema_hash": schema_hash(self.feature_names),
             "origin": self.origin, "meta": dict(self.meta), "x": self.x.tolist()}
        if include_names:
            d["feature_names"] = list(self.feature_names)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], feature_names: Optional[Sequence[str]] = None) -> "Candidate":
        names = feature_names if feature_names is not None else d.get("feature_names")
        if names is None:
            raise ValueError("feature_names required (not stored in the dict)")
        c = cls(np.asarray(d["x"], dtype=np.float64), tuple(names), d.get("origin", ""), dict(d.get("meta", {})))
        if "candidate_id" in d and d["candidate_id"] != c.candidate_id:
            raise ValueError("candidate_id mismatch: wrong feature_names or corrupted data")
        return c


def candidates_from_matrix(X: np.ndarray, feature_names: Sequence[str], origin: str = "",
                           metas: Optional[Sequence[Mapping[str, Any]]] = None) -> List[Candidate]:
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return [Candidate.from_array(X[i], feature_names, origin, metas[i] if metas else None)
            for i in range(X.shape[0])]
