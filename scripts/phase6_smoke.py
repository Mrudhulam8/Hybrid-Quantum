"""Phase 6 smoke test on a TINY SYNTHETIC fixture (not CIC-IDS2017, not a research result).

    python scripts/phase6_smoke.py [--budget 40] [--seed 1]

Runs random / bayesian / cmaes once each against a counting backend and checks exact query accounting,
determinism (second identical run) and that every backend row went through the QueryBudgetedModel.
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_runner import METHODS, make_engine                                   # noqa: E402
from src.oracle import ReferenceData                                                    # noqa: E402
from src.search_space import SearchSpace, SearchSpaceConfig                             # noqa: E402
from src.target_model import QueryBudgetedModel                                         # noqa: E402
from src.utils import load_config                                                       # noqa: E402
from src.validator import FailureValidator, ValidatorConfig                             # noqa: E402

D = 6
NAMES = tuple(f"f{i}" for i in range(D))


class CountingBackend:
    def __init__(self):
        self.rows = self.calls = 0

    def predict_proba(self, X):
        self.rows += len(X); self.calls += 1
        p1 = 1 / (1 + np.exp(-4 * (X[:, 0] - 0.4)))          # boundary shifted vs the reference labelling (x0 > 0)
        return np.column_stack([1 - p1, p1])


def fixture(n=500):
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 0.5, (n, D)); X0[:, 0] -= 1.0
    X1 = rng.normal(0, 0.5, (n, D)); X1[:, 0] += 1.0
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    ref = ReferenceData(np.vstack([X0, X1]), y, None, NAMES, ("BENIGN", "ATTACK"), "smoke-fixture")
    vcfg = dataclasses.replace(ValidatorConfig(), calibration_sample_size=200, knn_k=10)
    return ref, FailureValidator.from_reference(ref, vcfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config()
    cfg["search"]["bayesian"].update(n_components=3)
    cfg["search"]["common"]["max_candidates_per_query"] = 20
    ref, validator = fixture()
    space = SearchSpace.from_reference(ref.X, NAMES, dataclasses.replace(SearchSpaceConfig(), radius_enabled=True, radius_max=1.5))
    rows = []
    for m in METHODS:
        runs = []
        for _ in range(2):
            backend = CountingBackend()
            bb = QueryBudgetedModel(backend, D, 2, a.budget)
            r = make_engine(m, cfg, space, validator, a.budget, a.seed).run(bb)
            assert backend.rows == bb.n_queries == r.queries_used, "backend rows != counted queries"
            assert r.queries_used <= a.budget and r.black_box["n_queries"] == r.queries_used
            runs.append((r, backend))
        (r1, _), (r2, _) = runs
        assert r1.candidate_ids == r2.candidate_ids and r1.verified_failure_ids == r2.verified_failure_ids, "not reproducible"
        json.dumps(r1.to_dict())
        rows.append({"method": m, "queries_used": r1.queries_used, "budget": r1.query_budget,
                     "proposed": r1.candidates_evaluated, "valid": r1.candidates_valid,
                     "verified_failures": r1.verified_failures, "unique": r1.unique_verified_failures,
                     "termination": r1.termination_reason, "reproducible": True,
                     "traj": r1.reproducibility["trajectory_sha256"]})
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
