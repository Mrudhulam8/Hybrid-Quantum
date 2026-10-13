"""Phase 7 smoke test on a TINY SYNTHETIC fixture (not CIC-IDS2017, not a research result).

    python scripts/phase7_smoke.py [--budget 40] [--seed 1] [--backend numpy_statevector]

Runs random / bayesian / cmaes / quantum once each against a counting backend and CHECKS MECHANICS ONLY:
exact query accounting, no budget reset, no target bypass, no label / test-data access, same-seed
determinism, and result-schema compatibility. It prints counts, NOT a method comparison, and makes no claim
about which method is better. The budget is limited to 20-50 on purpose.
"""
import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from phase6_smoke import CountingBackend, D, NAMES, fixture                              # noqa: E402
from src import quantum_circuit, quantum_encoding, quantum_search                          # noqa: E402
from src.experiment_runner import METHODS, make_engine                                    # noqa: E402
from src.oracle import ReferenceData                                                      # noqa: E402
from src.search_space import SearchSpace, SearchSpaceConfig                               # noqa: E402
from src.target_model import QueryBudgetedModel                                           # noqa: E402
from src.utils import load_config                                                         # noqa: E402
from src.validator import FailureValidator                                                # noqa: E402


class BlindRef:
    def __init__(self, ref):
        object.__setattr__(self, "_r", ref)

    def __getattr__(self, name):
        if name in ("y", "categories"):
            raise AssertionError(f"labels read through validator.ref.{name}")
        return getattr(self._r, name)


class BlindValidator(FailureValidator):
    def __init__(self, base):
        super().__init__(base.oracle, base.cfg)
        self.ref = BlindRef(self.ref)


def strip_runtime(d):
    d = json.loads(json.dumps(d))
    d["reproducibility"].get("quantum", {}).pop("runtime", None)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", default="auto", choices=["auto", "numpy_statevector", "pennylane"])
    a = ap.parse_args()
    if not 20 <= a.budget <= 50:
        ap.error("--budget must be in 20..50 (smoke test only; the full benchmark is NOT run in Phase 7)")
    cfg = load_config()
    cfg["search"]["bayesian"].update(n_components=3)
    cfg["search"]["quantum"].update(latent_dimension=3, backend=a.backend)             # 6 synthetic features
    cfg["search"]["common"]["max_candidates_per_query"] = 20
    ref, validator = fixture()
    space = SearchSpace.from_reference(ref.X, NAMES, dataclasses.replace(SearchSpaceConfig(), radius_enabled=True, radius_max=1.5))
    checks = {}

    # ---- 1/2. all four methods: exact counts + compatible schema -----------------------------------------
    rows, key_sets = [], {}
    for m in METHODS:
        backend = CountingBackend()
        bb = QueryBudgetedModel(backend, D, 2, a.budget)
        r = make_engine(m, cfg, space, validator, a.budget, a.seed).run(bb)
        assert backend.rows == bb.n_queries == r.queries_used <= a.budget, f"{m}: query accounting"
        assert backend.calls == bb.n_calls, f"{m}: a backend call bypassed the wrapper"
        key_sets[m] = set(r.to_dict())
        json.dumps(r.to_dict())
        rows.append({"method": m, "budget": r.query_budget, "queries_used": r.queries_used, "backend_rows": backend.rows,
                     "evaluated": r.candidates_evaluated, "valid": r.candidates_valid,
                     "verified_failures": r.verified_failures, "unique_verified_failures": r.unique_verified_failures,
                     "termination": r.termination_reason})
        if m == "quantum":
            q = r.reproducibility["quantum"]
            quantum_result = r
    assert len({frozenset(k) for k in key_sets.values()}) == 1, "result schemas differ between methods"
    checks["exact_query_counts_all_methods"] = all(x["queries_used"] == x["backend_rows"] <= a.budget for x in rows)
    checks["result_schema_identical_all_methods"] = True
    checks["no_target_bypass_backend_calls_equal_wrapper_calls"] = True

    # ---- 3. determinism --------------------------------------------------------------------------------------
    r2 = make_engine("quantum", cfg, space, validator, a.budget, a.seed).run(QueryBudgetedModel(CountingBackend(), D, 2, a.budget))
    assert r2.candidate_ids == quantum_result.candidate_ids, "quantum search not reproducible"
    assert strip_runtime(r2.to_dict()) == strip_runtime(quantum_result.to_dict()), "quantum result dict differs for same seed"
    r3 = make_engine("quantum", cfg, space, validator, a.budget, a.seed + 1).run(QueryBudgetedModel(CountingBackend(), D, 2, a.budget))
    assert r3.candidate_ids != quantum_result.candidate_ids, "different seeds gave identical trajectories"
    checks["same_seed_deterministic"] = True
    checks["different_seed_differs"] = True

    # ---- 4. budget cannot be reset / reused ----------------------------------------------------------------------
    spent = QueryBudgetedModel(CountingBackend(), D, 2, a.budget)
    make_engine("quantum", cfg, space, validator, a.budget, a.seed).run(spent)
    for label, fn in (("reuse spent box", lambda: make_engine("quantum", cfg, space, validator, a.budget, a.seed).run(spent)),
                      ("wrong budget", lambda: make_engine("quantum", cfg, space, validator, a.budget, a.seed).run(
                          QueryBudgetedModel(CountingBackend(), D, 2, a.budget + 1)))):
        try:
            fn(); raise SystemExit(f"budget guard failed: {label}")
        except ValueError:
            pass
    assert spent.n_queries <= a.budget
    checks["budget_cannot_be_reset_or_reused"] = True

    # ---- 5. circuit executions are query-free ---------------------------------------------------------------------------
    backend = CountingBackend()
    bb = QueryBudgetedModel(backend, D, 2, a.budget)
    eng = make_engine("quantum", cfg, space, validator, a.budget, a.seed)
    for _ in range(200):
        eng.circuit.probabilities(eng.circuit.initial_params(), np.zeros(eng.circuit.n))
    assert backend.rows == backend.calls == bb.n_queries == 0, "circuit execution touched the target"
    checks["circuit_execution_costs_zero_target_queries"] = True

    # ---- 6. no label access / no test data ------------------------------------------------------------------------------------
    bv = BlindValidator(validator)
    make_engine("quantum", cfg, space, bv, a.budget, a.seed).run(QueryBudgetedModel(CountingBackend(), D, 2, a.budget))
    checks["no_reference_label_access_dynamic"] = True
    for mod in (quantum_search, quantum_circuit, quantum_encoding):
        code = re.sub(r"#.*", "", re.sub(r'"""(?:.|\n)*?"""', "", Path(mod.__file__).read_text()))
        assert not re.search(r"\b(X|y|cat)_test\b|\.ref\.y\b|\.categories\b|\by_train\b|\.query\(|\.predict_proba\(|"
                             r"\bverified_failure\b|target_model|import xgboost", code), mod.__name__
    checks["no_label_or_test_access_static"] = True

    q = quantum_result.reproducibility["quantum"]
    print(json.dumps({
        "note": "mechanics-only synthetic smoke test; counts below are NOT a performance comparison",
        "methods": rows,
        "quantum": {k: q[k] for k in ("backend", "framework", "framework_version", "backend_requested", "n_qubits", "depth",
                                      "shots", "latent_dimension", "iterations", "cost", "runtime")},
        "quantum_warnings": quantum_result.warnings,
        "checks": checks}, indent=1))
    assert all(checks.values())
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
