"""Failure Critic. The ONLY gateway through which discovery code evaluates candidates.

UNCERTAINTY != FAILURE. Four separate facts are recorded for every candidate:
  A. model uncertainty  - entropy / margin / confidence of the target model (informational only)
  B. candidate validity - realistic input? (schema, finite, range, Mahalanobis, NN distance)
  C. oracle evidence    - label + purity from the TRAINING-split reference (never the target model)
  D. verified failure   = valid AND oracle_confidence >= min_oracle_confidence
                          AND target_prediction != oracle_label
                          [AND target_confidence >= min_target_confidence, default 0 = off]
Uncertainty appears nowhere in rule D except the optional target-confidence filter, which can only
make the rule stricter.

QUERY ACCOUNTING
* One valid candidate = exactly one query, through `QueryBudgetedModel.query` (no other path).
* Malformed/invalid candidates are rejected BEFORE the model is touched -> never charged.
* A whole batch of valid candidates is ONE `black_box.query` call, so a batch that does not fit the
  remaining budget raises `QueryBudgetExceeded` with nothing evaluated (Phase 3 semantics).
* Consequence for later phases: search methods get target-model feedback only for VALID candidates.
  The same policy applies to every method.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .candidate import OK, Candidate, candidates_from_matrix
from .oracle import NO_LABEL, ReferenceData, ReferenceOracle
from .uncertainty import prediction_entropy, prediction_margin
from .utils import config_hash, resolve_path, save_json

# failure_reason vocabulary
F_VERIFIED = "verified_failure"
F_INVALID = "invalid_candidate"
F_ABSTAIN = "oracle_abstained"
F_LOW_ORACLE = "oracle_confidence_below_threshold"
F_AGREES = "target_agrees_with_oracle"
F_LOW_TARGET = "target_confidence_below_threshold"


# ==========================================================================
# Config
# ==========================================================================
@dataclass(frozen=True)
class ValidatorConfig:
    numerical_tol: float = 1e-9
    seed: int = 0
    reference_max_rows: Optional[int] = None
    range_enabled: bool = False
    range_margin: float = 0.0
    maha_mode: str = "empirical_quantile"
    maha_quantile: float = 0.995
    maha_absolute: Optional[float] = None
    maha_shrinkage: str = "ledoit_wolf"
    maha_shrinkage_alpha: float = 0.1
    maha_ridge: float = 0.0
    maha_eig_floor_rel: float = 1e-8
    nn_gate_enabled: bool = True
    nn_gate_mode: str = "empirical_quantile"
    nn_gate_quantile: float = 0.995
    nn_gate_absolute: Optional[float] = None
    calibration_sample_size: int = 5000
    knn_k: int = 15
    knn_min_purity: float = 0.8
    knn_chunk_size: int = 64
    rule_min_oracle_confidence: float = 0.8
    rule_min_target_confidence: float = 0.0

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "ValidatorConfig":
        v = cfg["validator"]
        c = cls(numerical_tol=v["numerical_tol"], seed=v["seed"], reference_max_rows=v["reference"]["max_rows"],
                range_enabled=v["feature_range"]["enabled"], range_margin=v["feature_range"]["margin"],
                maha_mode=v["mahalanobis"]["threshold_mode"], maha_quantile=v["mahalanobis"]["quantile"],
                maha_absolute=v["mahalanobis"]["absolute_threshold"], maha_shrinkage=v["mahalanobis"]["shrinkage"],
                maha_shrinkage_alpha=v["mahalanobis"]["shrinkage_alpha"], maha_ridge=v["mahalanobis"]["ridge"],
                maha_eig_floor_rel=v["mahalanobis"]["eigenvalue_floor_rel"],
                nn_gate_enabled=v["nn_gate"]["enabled"], nn_gate_mode=v["nn_gate"]["mode"],
                nn_gate_quantile=v["nn_gate"]["quantile"], nn_gate_absolute=v["nn_gate"]["absolute_threshold"],
                calibration_sample_size=v["calibration"]["sample_size"], knn_k=v["knn"]["k"],
                knn_min_purity=v["knn"]["min_purity"], knn_chunk_size=v["knn"]["chunk_size"],
                rule_min_oracle_confidence=v["failure_rule"]["min_oracle_confidence"],
                rule_min_target_confidence=v["failure_rule"]["min_target_confidence"])
        c.validate()
        return c

    def validate(self) -> None:
        bad = []
        if self.knn_k < 1: bad.append("knn.k must be >= 1")
        if not 0 < self.knn_min_purity <= 1: bad.append("knn.min_purity must be in (0, 1]")
        if not 0 < self.rule_min_oracle_confidence <= 1: bad.append("failure_rule.min_oracle_confidence must be in (0, 1]")
        if not 0 <= self.rule_min_target_confidence <= 1: bad.append("failure_rule.min_target_confidence must be in [0, 1]")
        if self.maha_mode not in ("empirical_quantile", "chi2_quantile", "absolute"): bad.append("bad mahalanobis.threshold_mode")
        if self.maha_mode != "absolute" and not 0 < self.maha_quantile < 1: bad.append("mahalanobis.quantile must be in (0, 1)")
        if self.maha_mode == "absolute" and not (self.maha_absolute and self.maha_absolute > 0): bad.append("mahalanobis.absolute_threshold must be > 0")
        if self.maha_shrinkage not in ("ledoit_wolf", "fixed", "none"): bad.append("bad mahalanobis.shrinkage")
        if not 0 <= self.maha_shrinkage_alpha <= 1: bad.append("mahalanobis.shrinkage_alpha must be in [0, 1]")
        if self.nn_gate_mode not in ("empirical_quantile", "absolute"): bad.append("bad nn_gate.mode")
        if self.nn_gate_enabled and self.nn_gate_mode == "empirical_quantile" and not 0 < self.nn_gate_quantile < 1: bad.append("nn_gate.quantile must be in (0, 1)")
        if self.nn_gate_enabled and self.nn_gate_mode == "absolute" and not (self.nn_gate_absolute and self.nn_gate_absolute > 0): bad.append("nn_gate.absolute_threshold must be > 0")
        if self.range_margin < 0: bad.append("feature_range.margin must be >= 0")
        if self.numerical_tol < 0: bad.append("numerical_tol must be >= 0")
        if self.calibration_sample_size < 2: bad.append("calibration.sample_size must be >= 2")
        if self.rule_min_oracle_confidence < self.knn_min_purity:
            bad.append("failure_rule.min_oracle_confidence < knn.min_purity has no effect (oracle abstains first); set it >= knn.min_purity")
        if bad:
            raise ValueError("Invalid validator config: " + "; ".join(bad))


# ==========================================================================
# Result schema
# ==========================================================================
@dataclass
class ValidationResult:
    # identity / provenance
    candidate_id: str
    seed: int
    origin: str
    # B. validity
    candidate_valid: bool
    validity_reason: str
    distance_to_reference: Optional[float]          # global Mahalanobis distance
    mahalanobis_threshold: Optional[float]
    nearest_neighbor_distance: Optional[float]      # Euclidean 1-NN distance in the reference
    nn_distance_threshold: Optional[float]
    # C. oracle evidence (reference/training data only)
    oracle_label: Optional[int]
    oracle_confidence: Optional[float]              # purity if the oracle assigned a label, else 0.0
    nearest_neighbor_purity: Optional[float]        # k-NN purity of the majority label (always computed)
    oracle_category: Optional[str]                  # majority attack type among neighbours (reporting only)
    # A. target model (only for valid, queried candidates)
    queried: bool
    target_prediction: Optional[int]
    target_confidence: Optional[float]
    target_proba: Optional[List[float]]
    target_entropy: Optional[float]                 # informational; NOT used by the failure rule
    target_margin: Optional[float]                  # informational; NOT used by the failure rule
    target_query_index: Optional[int]               # 1-based global index of this candidate's query
    target_query_count: int                         # cumulative queries consumed on the black box
    # D. decision
    verified_failure: bool
    failure_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


RESULT_COLUMNS = [f.name for f in fields(ValidationResult)]


def results_to_dataframe(results: Sequence[ValidationResult]):
    import pandas as pd
    df = pd.DataFrame([r.to_dict() for r in results], columns=RESULT_COLUMNS)
    df["target_proba"] = df["target_proba"].map(lambda p: None if p is None else json.dumps(p))
    return df


def save_results(results: Sequence[ValidationResult], path_stem, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    stem = resolve_path(path_stem)
    j = save_json({"meta": meta or {}, "columns": RESULT_COLUMNS, "results": [r.to_dict() for r in results]},
                  stem.with_suffix(".json"))
    c = stem.with_suffix(".csv")
    results_to_dataframe(results).to_csv(c, index=False)
    return {"json": j, "csv": c}


def load_results_json(path) -> List[ValidationResult]:
    d = json.load(open(resolve_path(path)))
    return [ValidationResult(**r) for r in d["results"]]


# ==========================================================================
# Validator
# ==========================================================================
class FailureValidator:
    def __init__(self, oracle: ReferenceOracle, cfg: ValidatorConfig):
        self.oracle, self.cfg = oracle, cfg
        self.ref = oracle.ref
        self.feature_names = self.ref.feature_names
        self.class_names = self.ref.class_names

    @classmethod
    def from_reference(cls, reference: ReferenceData, cfg) -> "FailureValidator":
        vcfg = cfg if isinstance(cfg, ValidatorConfig) else ValidatorConfig.from_cfg(cfg)
        vcfg.validate()
        return cls(ReferenceOracle(reference, vcfg), vcfg)

    @classmethod
    def from_prepared(cls, prepared, cfg: Dict[str, Any]) -> "FailureValidator":
        """Convenience: builds the TRAIN-only reference view, then the validator."""
        vcfg = ValidatorConfig.from_cfg(cfg)
        ref = ReferenceData.from_prepared(prepared, vcfg.reference_max_rows, vcfg.seed)
        return cls.from_reference(ref, vcfg)

    def describe(self, cfg_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = {"validator_config": asdict(self.cfg), "oracle": self.oracle.diagnostics}
        if cfg_dict is not None:
            out["validator_config_hash"] = config_hash(cfg_dict["validator"])
        return out

    # ------------------------------------------------------------------
    def validate_matrix(self, X: np.ndarray, black_box, seed: int = 0, origin: str = "") -> List[ValidationResult]:
        return self.validate_batch(candidates_from_matrix(X, self.feature_names, origin), black_box, seed)

    def validate_batch(self, candidates: Sequence[Candidate], black_box, seed: int = 0) -> List[ValidationResult]:
        """Validate candidates; query the target model once for the valid ones."""
        if black_box.n_features != self.ref.d:
            raise ValueError(f"black box expects {black_box.n_features} features, reference has {self.ref.d}")
        if black_box.n_classes != len(self.class_names):
            raise ValueError("black box and reference disagree on the number of classes")
        n = len(candidates)
        if n == 0:
            return []
        cfg = self.cfg
        # ---- 1. cheap schema checks (no model, no reference) -------------------------------
        pre = [c.check(self.feature_names) for c in candidates]
        well = [i for i in range(n) if pre[i] == OK]
        # ---- 2. oracle assessment on well-formed candidates (never touches the target model) -
        assess = self.oracle.assess(np.stack([candidates[i].x for i in well])) if well else None
        pos = {i: j for j, i in enumerate(well)}
        valid_idx = [i for i in well if assess.valid[pos[i]]]
        # ---- 3. ONE query call for the valid candidates (raises QueryBudgetExceeded atomically) -
        before = black_box.n_queries
        qres = black_box.query(np.stack([candidates[i].x for i in valid_idx])) if valid_idx else None
        qpos = {i: j for j, i in enumerate(valid_idx)}
        # ---- 4. assemble results -----------------------------------------------------------
        out: List[ValidationResult] = []
        for i, cand in enumerate(candidates):
            r = dict(candidate_id=cand.candidate_id, seed=int(seed), origin=cand.origin,
                     candidate_valid=False, validity_reason=pre[i], distance_to_reference=None,
                     mahalanobis_threshold=None, nearest_neighbor_distance=None, nn_distance_threshold=None,
                     oracle_label=None, oracle_confidence=None, nearest_neighbor_purity=None, oracle_category=None,
                     queried=False, target_prediction=None, target_confidence=None, target_proba=None,
                     target_entropy=None, target_margin=None, target_query_index=None,
                     target_query_count=0, verified_failure=False, failure_reason=F_INVALID)
            if i in pos:
                j = pos[i]
                lab = int(assess.label[j])
                r.update(candidate_valid=bool(assess.valid[j]), validity_reason=assess.reason[j],
                         distance_to_reference=float(assess.mahalanobis[j]), mahalanobis_threshold=self.oracle.tau_maha,
                         nearest_neighbor_distance=float(assess.nn_distance[j]), nn_distance_threshold=self.oracle.tau_nn,
                         oracle_label=None if lab == NO_LABEL else lab,
                         oracle_confidence=float(assess.purity[j]) if lab != NO_LABEL else 0.0,
                         nearest_neighbor_purity=float(assess.purity[j]), oracle_category=assess.category[j])
            if i in qpos:
                q = qpos[i]
                p = qres.proba[q]
                r.update(queried=True, target_prediction=int(qres.pred[q]), target_confidence=float(p.max()),
                         target_proba=p.tolist(), target_entropy=float(prediction_entropy(p)[0]),
                         target_margin=float(prediction_margin(p)[0]), target_query_index=before + q + 1)
                r["verified_failure"], r["failure_reason"] = self._rule(r)
            out.append(ValidationResult(**r))
        total = black_box.n_queries
        for res in out:                       # cumulative count AFTER the batch (same for every row)
            res.target_query_count = total
        return out

    def _rule(self, r: Dict[str, Any]):
        """D. The explicit failure rule. Uncertainty (entropy/margin) is deliberately absent."""
        c = self.cfg
        if not r["candidate_valid"]:
            return False, F_INVALID
        if r["oracle_label"] is None:
            return False, F_ABSTAIN
        if r["oracle_confidence"] < c.rule_min_oracle_confidence:
            return False, F_LOW_ORACLE
        if r["target_prediction"] == r["oracle_label"]:
            return False, F_AGREES
        if r["target_confidence"] < c.rule_min_target_confidence:
            return False, F_LOW_TARGET
        return True, F_VERIFIED
