"""Phase 8 environment setup and audits (real-data loader, leakage audit, accounting and reproducibility checks).

Used by `scripts/phase8_preflight.py` and `scripts/phase8_run.py`. Nothing here alters an algorithm; it only loads the
existing artefacts and verifies them. The real-data loader NEVER falls back to KDD99/synthetic data or to the dev-mock model.
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .experiment_runner import METHODS, build_search_components, make_engine, run_search_method
from .phase8 import (DEFAULT_MODEL_TAG, EXPECTED_DATASET, _sha256_file, campaign_fingerprint)
from .utils import deep_update, resolve_path

SEARCH_MODULES = ("search_base", "random_search", "bayesian_search", "cmaes_search", "quantum_search", "quantum_circuit",
                  "quantum_encoding", "search_score", "search_representation", "candidate_generator", "search_space")
QUANTUM_MODULES = ("quantum_search", "quantum_circuit", "quantum_encoding")
# any of these tokens inside a search module means it could reach labels / test data / model internals
FORBIDDEN_COMMON = (r"import xgboost", r"\bTargetModel\b", r"\.clf\b", r"booster", r"predict_proba_unbudgeted",
                    r"\b(X|y|cat)_test\b", r"\.ref\.y\b", r"\.categories\b", r"\by_train\b", r"\bcat_train\b", r"\bload_prepared\b")
FORBIDDEN_QUANTUM_EXTRA = (r"\.query\(", r"\.predict_proba\(", r"\bverified_failure\b", r"target_model")


class PreflightFailure(RuntimeError):
    pass


def _strip(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r'"""(?:.|\n)*?"""', "", code)
    code = re.sub(r"'''(?:.|\n)*?'''", "", code)
    return code


def static_leakage_audit(src_dir: Path) -> List[str]:
    """Returns violations (empty = pass)."""
    bad = []
    for name in SEARCH_MODULES:
        p = Path(src_dir) / f"{name}.py"
        if not p.exists():
            bad.append(f"{name}.py missing")
            continue
        code = _strip(p.read_text())
        pats = FORBIDDEN_COMMON + (FORBIDDEN_QUANTUM_EXTRA if name in QUANTUM_MODULES else ())
        for pat in pats:
            m = re.search(pat, code)
            if m:
                bad.append(f"{name}.py matches /{pat}/ at {m.group(0)!r}")
    return bad


class BlindRef:
    """Proxy for `validator.ref` that raises if a search engine reads reference labels / categories."""
    def __init__(self, ref):
        object.__setattr__(self, "_r", ref)

    def __getattr__(self, name):
        if name in ("y", "categories"):
            raise AssertionError(f"labels read through validator.ref.{name}")
        return getattr(self._r, name)


def blind_validator(validator):
    from .validator import FailureValidator

    class _BV(FailureValidator):
        def __init__(self, base):
            super().__init__(base.oracle, base.cfg)
            self.ref = BlindRef(self.ref)
    return _BV(validator)


class CountingClf:
    """Wraps the real classifier to count rows reaching it; the ONLY path to it is QueryBudgetedModel."""
    def __init__(self, clf):
        self._clf = clf
        self.rows = 0
        self.calls = 0

    def predict_proba(self, X):
        self.rows += len(X)
        self.calls += 1
        return self._clf.predict_proba(X)


def counting_target(model):
    c = CountingClf(model.clf)
    return dataclasses.replace(model, clf=c), c


# ==========================================================================
# Real-data component loader
# ==========================================================================
def force_real_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return deep_update(cfg, {"data": {"source": EXPECTED_DATASET}})          # no 'auto' fallback, ever


def load_real_components(cfg: Dict[str, Any], model_tag: str = DEFAULT_MODEL_TAG) -> Dict[str, Any]:
    """Loads prepared real CIC-IDS2017 data, the saved XGBoost artefact, SearchSpace and FailureValidator.
    Raises PreflightFailure with a clear message for anything missing or ineligible."""
    from .preprocessing import load_prepared, prepare_data
    from .target_model import TargetModel
    from .utils import config_hash
    cfg = force_real_cfg(cfg)
    d = cfg["data"]
    raw_dir = resolve_path(d["raw_dir"]) / d["cic_subdir"]
    tag = f"{EXPECTED_DATASET}_{d['task']}"
    cache = resolve_path(d["processed_dir"]) / f"{tag}.npz"
    if cache.exists():
        prepared = load_prepared(d["processed_dir"], tag)
        if prepared.report.get("data_config_hash") != config_hash(d):
            raise PreflightFailure(f"processed cache {cache} was built with a different data config "
                                   f"({prepared.report.get('data_config_hash')} != {config_hash(d)}); re-run prepare-data")
    elif raw_dir.exists() and list(raw_dir.glob("*.csv")):
        prepared = prepare_data(cfg)
    else:
        raise PreflightFailure(f"real CIC-IDS2017 data not found: neither {cache} nor *.csv in {raw_dir}")
    prov = prepared.provenance
    if prov.get("source") != EXPECTED_DATASET or prov.get("is_synthetic") or not prov.get("paper_eligible"):
        raise PreflightFailure(f"dataset is not real paper-eligible CIC-IDS2017: {prov}")
    mdir = resolve_path(cfg["project"]["models_dir"])
    if not (mdir / f"{model_tag}_meta.json").exists():
        raise PreflightFailure(f"target-model artefact not found: {mdir / (model_tag + '_meta.json')}")
    model = TargetModel.load(cfg["project"]["models_dir"], model_tag)
    if model.kind != "xgboost" or model.is_dev_mock:
        raise PreflightFailure(f"target model must be xgboost (got {model.kind})")
    if list(model.feature_names) != list(prepared.feature_names):
        raise PreflightFailure("model feature schema differs from the prepared data schema")
    space, validator = build_search_components(cfg, prepared)     # TRAINING split only
    meta = json.loads((mdir / f"{model_tag}_meta.json").read_text())
    model_file = mdir / meta["model_file"]
    return {"cfg": cfg, "prepared": prepared, "model": model, "space": space, "validator": validator,
            "model_tag": model_tag, "model_file": model_file, "model_sha256": _sha256_file(model_file),
            "model_meta_sha256": _sha256_file(mdir / f"{model_tag}_meta.json"), "provenance": prov}


def verify_model_artifact(comp: Dict[str, Any]) -> Dict[str, Any]:
    """Confirms the loaded artefact is the Phase 3 one: re-hash its test-set probabilities (features only, no labels)
    and compare to the stored Phase 3 report. Also recomputes (does not overwrite) the headline metrics for the record."""
    from .target_model import proba_hash
    cfg, prepared, model = comp["cfg"], comp["prepared"], comp["model"]
    rep_path = resolve_path(cfg["project"]["results_dir"]) / "raw" / f"target_model_{comp['model_tag']}.json"
    if not rep_path.exists():
        raise PreflightFailure(f"Phase 3 report {rep_path} not found; cannot verify the artefact")
    rep = json.loads(rep_path.read_text())
    P = model.predict_proba_unbudgeted(prepared.X_test)       # TRAINING-SIDE verification, outside any search
    h = proba_hash(P)
    if h != rep.get("test_proba_sha256"):
        raise PreflightFailure(f"model/data mismatch: test proba hash {h} != Phase 3 report {rep.get('test_proba_sha256')}")
    if not rep.get("paper_eligible"):
        raise PreflightFailure("Phase 3 report says the target model is NOT paper eligible")
    return {"report_path": str(rep_path), "test_proba_sha256": h, "report": rep}


def make_fingerprint(comp: Dict[str, Any]) -> str:
    return campaign_fingerprint(comp["cfg"], reference_fingerprint=comp["validator"].ref.fingerprint,
                                model_sha256=comp["model_sha256"], dataset=comp["provenance"]["source"])


# ==========================================================================
# Dynamic audits (work on any (space, validator, target_model) triple, real or synthetic)
# ==========================================================================
def strip_runtime(d: Dict[str, Any]) -> Dict[str, Any]:
    d = json.loads(json.dumps(d, default=str))
    d.get("reproducibility", {}).get("quantum", {}).pop("runtime", None)
    return d


def accounting_audit(cfg, space, validator, target_model, budget: int, seed: int,
                     methods=tuple(METHODS)) -> Dict[str, Any]:
    """Every method on a counting wrapper: rows reaching the classifier == counted queries == result.queries_used <= budget,
    and no call bypasses the budget wrapper. Also runs each method behind a label-blind validator."""
    out = {}
    tm, cnt = counting_target(target_model)
    for m in methods:
        c0, k0 = cnt.rows, cnt.calls
        res = run_search_method(m, cfg, space, blind_validator(validator), tm, budget, seed, paper_eligible_context=False)
        rows, calls = cnt.rows - c0, cnt.calls - k0
        ok = rows == res.queries_used == res.black_box["n_queries"] <= budget and calls == res.black_box["n_calls"]
        out[m] = {"queries_used": res.queries_used, "backend_rows": rows, "backend_calls": calls,
                  "wrapper_calls": res.black_box["n_calls"], "ok": bool(ok), "unique_verified_failures": res.unique_verified_failures,
                  "verified_failures": res.verified_failures}
    return out


def reproducibility_audit(cfg, space, validator, target_model, method: str, budget: int, seed: int) -> Dict[str, Any]:
    ra = run_search_method(method, cfg, space, validator, target_model, budget, seed, paper_eligible_context=False)
    rb = run_search_method(method, cfg, space, validator, target_model, budget, seed, paper_eligible_context=False)
    da, db = strip_runtime(ra.to_dict()), strip_runtime(rb.to_dict())
    same = {"candidate_sequence": ra.candidate_ids == rb.candidate_ids, "queries_used": ra.queries_used == rb.queries_used,
            "verified_failure_ids": ra.verified_failure_ids == rb.verified_failure_ids,
            "scores": [t["search_score"] for t in (ra.trajectory or [])] == [t["search_score"] for t in (rb.trajectory or [])],
            "summary": da == db}
    return {"method": method, "budget": budget, "seed": seed, "identical": all(same.values()), "details": same}
