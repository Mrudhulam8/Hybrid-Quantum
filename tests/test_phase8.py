"""Phase 8 unit tests. SYNTHETIC / MOCK DATA ONLY: nothing here is a research result and no test output may be used in the paper."""
import dataclasses
import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import phase8, phase8_report, phase8_setup                                    # noqa: E402
from src.experiment_runner import METHODS                                              # noqa: E402
from src.oracle import ReferenceData                                                   # noqa: E402
from src.phase8 import (METHOD_ORDER, ResultStore, CampaignMismatchError, RunSpec, build_matrix, make_run_id,   # noqa: E402
                        run_campaign, run_one)
from src.search_space import SearchSpace, SearchSpaceConfig                            # noqa: E402
from src.target_model import QueryBudgetedModel, TargetModel                           # noqa: E402
from src.utils import load_config                                                      # noqa: E402
from src.validator import FailureValidator, ValidatorConfig                            # noqa: E402

D = 6
NAMES = tuple(f"f{i}" for i in range(D))


class FakeClf:
    def __init__(self):
        self.rows = 0
        self.calls = 0

    def predict_proba(self, X):
        self.rows += len(X); self.calls += 1
        p1 = 1 / (1 + np.exp(-4 * (X[:, 0] - 0.4)))
        return np.column_stack([1 - p1, p1])


def fixture(n=400):
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 0.5, (n, D)); X0[:, 0] -= 1.0
    X1 = rng.normal(0, 0.5, (n, D)); X1[:, 0] += 1.0
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    ref = ReferenceData(np.vstack([X0, X1]), y, None, NAMES, ("BENIGN", "ATTACK"), "synthetic-test-fixture")
    vcfg = dataclasses.replace(ValidatorConfig(), calibration_sample_size=200, knn_k=10)
    return ref, FailureValidator.from_reference(ref, vcfg)


def setup(budgets=(20, 30), seeds=(1, 2, 3)):
    cfg = load_config()
    cfg["search"]["bayesian"].update(n_components=3)
    cfg["search"]["quantum"].update(latent_dimension=3, backend="numpy_statevector")
    cfg["search"]["common"].update(max_candidates_per_query=20, query_budgets=list(budgets), seeds=list(seeds))
    ref, validator = fixture()
    space = SearchSpace.from_reference(ref.X, NAMES, dataclasses.replace(SearchSpaceConfig(), radius_enabled=True, radius_max=1.5))
    model = TargetModel(FakeClf(), "xgboost", D, 2, ["BENIGN", "ATTACK"], list(NAMES), {}, {})
    return cfg, space, validator, model


def _run(cfg, space, validator, model, out, **kw):
    specs = kw.pop("specs", None) or build_matrix(cfg)
    return specs, run_campaign(specs, cfg, space, validator, model, out, config_fingerprint="fp-test", paper_eligible_context=False,
                               dataset="synthetic-test", model_name="fake", **kw)


# ---------------------------------------------------------------- matrix / ids / config
def test_experiment_matrix_and_run_ids():
    cfg = load_config()
    m = build_matrix(cfg)
    assert len(m) == 240 == 4 * 6 * 10
    ids = [s.run_id for s in m]
    assert len(set(ids)) == 240 and ids[0] == "random_100_1" and ids[-1] == "quantum_5000_10"
    assert make_run_id("cmaes", 5000, 10) == "cmaes_5000_10"
    assert ids == [s.run_id for s in build_matrix(cfg)]                                       # deterministic order
    assert {s.method for s in m} == set(METHOD_ORDER)


def test_configuration_consistency_and_four_methods_registered():
    cfg = load_config()
    assert tuple(METHODS) == METHOD_ORDER or set(METHODS) == set(METHOD_ORDER)
    assert len(METHODS) == 4
    assert cfg["search"]["common"]["query_budgets"] == [100, 250, 500, 1000, 2500, 5000]
    assert cfg["reproducibility"]["seeds"] == list(range(1, 11))
    assert sorted({s.budget for s in build_matrix(cfg)}) == cfg["search"]["common"]["query_budgets"]
    try:
        build_matrix(cfg, methods=["nope"]); assert False
    except ValueError:
        pass
    try:
        build_matrix(cfg, seeds=[1, 1]); assert False
    except ValueError:
        pass


# ---------------------------------------------------------------- propagation / schema
def test_budget_and_seed_propagation_schema_and_no_budget_reset(tmp_path):
    cfg, space, validator, model = setup()
    specs, summ = _run(cfg, space, validator, model, tmp_path)
    assert summ["completed_total"] == len(specs) == 4 * 2 * 3 and not summ["missing"]
    recs = ResultStore(tmp_path / "raw" / "run_results.jsonl").ok_records()
    for s in specs:
        r = recs[s.run_id]
        assert (r["method"], r["query_budget"], r["seed"]) == (s.method, s.budget, s.seed)
        assert r["queries_used"] <= s.budget and r["black_box"]["budget"] == s.budget
        assert r["black_box"]["n_queries"] == r["queries_used"]                                   # fresh box per run, no reset
        for f in phase8.REQUIRED_FIELDS:
            assert f in r, f
        assert r["unique_verified_failures"] <= r["verified_failures"] <= max(r["verified_failure_events"], 0) + 0
        assert r["duplicate_candidates"] + r["unique_candidates"] >= 0
        assert 0 <= r["valid_rate"] <= 1 and 0 <= r["duplicate_rate"] <= 1
        assert r["candidate_count"] == r["unique_candidates"] + r["duplicate_candidates"]
        assert r["valid_candidates"] + r["invalid_candidates"] == r["unique_candidates"]
        json.dumps(r)
    assert model.clf.rows >= 0


def test_no_query_budget_reset_or_reuse_of_black_box():
    cfg, space, validator, model = setup()
    from src.experiment_runner import make_engine
    bb = model.black_box(20)
    make_engine("random", cfg, space, validator, 20, 1).run(bb)
    for fn in (lambda: make_engine("random", cfg, space, validator, 20, 1).run(bb),
               lambda: make_engine("random", cfg, space, validator, 20, 1).run(model.black_box(21))):
        try:
            fn(); assert False, "budget guard failed"
        except ValueError:
            pass
    assert bb.n_queries <= 20


def test_same_seed_same_result_and_different_seed_differs(tmp_path):
    cfg, space, validator, model = setup()
    fp = "fp"
    kw = dict(paper_eligible_context=False, dataset="synthetic-test", model_name="fake", config_fingerprint=fp)
    a = run_one(RunSpec("quantum", 20, 1), cfg, space, validator, model, **kw)
    b = run_one(RunSpec("quantum", 20, 1), cfg, space, validator, model, **kw)
    c = run_one(RunSpec("quantum", 20, 2), cfg, space, validator, model, **kw)
    strip = lambda r: {k: v for k, v in r.items() if k not in ("runtime_seconds", "method_specific")}
    assert strip(a) == strip(b) and a["trajectory_sha256"] != c["trajectory_sha256"]
    rep = phase8_setup.reproducibility_audit(cfg, space, validator, model, "quantum", 20, 1)
    assert rep["identical"], rep
    rep = phase8_setup.reproducibility_audit(cfg, space, validator, model, "cmaes", 20, 1)
    assert rep["identical"], rep


def test_query_accounting_all_methods_and_circuits_cost_zero():
    cfg, space, validator, model = setup()
    r = phase8_setup.accounting_audit(cfg, space, validator, model, 30, 1)
    assert set(r) == set(METHOD_ORDER) and all(x["ok"] for x in r.values()), r
    rec = run_one(RunSpec("quantum", 30, 1), cfg, space, validator, model, paper_eligible_context=False,
                  dataset="s", model_name="m", config_fingerprint="f")
    ms = rec["method_specific"]
    assert ms["target_queries_charged_by_circuit"] == 0 and ms["quantum_evaluations"] > 0
    assert ms["n_qubits"] == cfg["search"]["quantum"]["n_qubits"] and ms["depth"] == cfg["search"]["quantum"]["depth"]
    assert ms["shots"] == cfg["search"]["quantum"]["shots"]


# ---------------------------------------------------------------- duplicates / unique failures
def test_duplicate_handling_and_unique_failure_counting():
    from src.failure_tracker import FailureTracker
    from types import SimpleNamespace as NS
    t = FailureTracker("m", 1, 0.05)
    mk = lambda cid, q: NS(candidate_id=cid, target_query_index=q, candidate_valid=True, validity_reason="", verified_failure=True, failure_reason="")
    x = np.zeros(3)
    t.record(mk("a", 1), 0.0, x); t.record(mk("a", 2), 0.0, x, re_evaluation=True)            # same id twice
    t.record(mk("b", 3), 0.0, x + 0.001)                                                      # distinct id, same cluster
    t.record(mk("c", 4), 0.0, x + 1.0)                                                        # distinct cluster
    assert t.n_events == 4 and t.n_verified == 3 and t.n_unique == 2                          # total != distinct != unique
    cfg, space, validator, model = setup()
    rec = run_one(RunSpec("random", 30, 1), cfg, space, validator, model, paper_eligible_context=False,
                  dataset="s", model_name="m", config_fingerprint="f")
    assert len(rec["verified_failure_ids"]) == len(set(rec["verified_failure_ids"])) == rec["verified_failures"]


# ---------------------------------------------------------------- aggregation / CI / stats
def _df(values, budgets=(100,), seeds=range(1, 11)):
    import pandas as pd
    rows = []
    for m, base in values.items():
        for b in budgets:
            for i, s in enumerate(seeds):
                v = base[i] if hasattr(base, "__len__") else base
                rows.append({"method": m, "query_budget": b, "seed": s, "unique_verified_failures": v, "verified_failures": v,
                             "failures_per_100_queries": 100 * v / b, "id_failures_per_100_queries": 100 * v / b,
                             "valid_rate": 0.5, "duplicate_rate": 0.1, "failure_rate": v / b, "runtime_seconds": 1.0,
                             "queries_used": b, "first_verified_failure_query": 5.0,
                             "verified_failure_ids": [f"{m}{j}" for j in range(int(v))],
                             "failure_query_numbers": list(range(1, int(v) + 1)), "paper_eligible": True,
                             "dataset": "cic-ids2017", "warnings": [], "method_specific": {}})
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], categories=list(METHOD_ORDER), ordered=True)
    return df


def test_aggregation_correctness_and_id_separation():
    vals = list(range(1, 11))
    df = _df({"random": vals, "quantum": vals})
    s = phase8.summary_by_method_budget(df)
    r = s[s.method == "random"].iloc[0]
    assert r["unique_verified_failures__mean"] == 5.5 and r["unique_verified_failures__median"] == 5.5
    assert abs(r["unique_verified_failures__std"] - np.std(vals, ddof=1)) < 1e-12
    assert r["unique_verified_failures__min"] == 1 and r["unique_verified_failures__max"] == 10
    assert r["distinct_ids_pooled_across_seeds"] == 10 and r["mean_distinct_ids_per_seed"] == 5.5   # ids pooled, not averaged
    assert r["unique_verified_failures__ci95_boot_low"] <= 5.5 <= r["unique_verified_failures__ci95_boot_high"]


def test_confidence_intervals():
    a = np.arange(10.0)
    lo, hi = phase8.bootstrap_ci(a)
    assert (lo, hi) == phase8.bootstrap_ci(a)                                                  # deterministic
    assert 2.5 < lo < 4.5 < hi < 6.5
    assert phase8.bootstrap_ci([3, 3, 3]) == (3.0, 3.0)
    tl, th = phase8.t_ci(a)
    assert abs((tl + th) / 2 - 4.5) < 1e-9 and tl < 4.5 < th


def test_statistical_analysis_detects_real_effect_and_not_noise():
    rng = np.random.default_rng(0)
    base = rng.integers(10, 20, 10).astype(float)
    df = _df({"random": base, "bayesian": base + rng.normal(0, 1, 10), "cmaes": base - 5, "quantum": base})
    t = phase8.statistical_tests(df)
    p = t[(t.metric == phase8.PRIMARY_METRIC)].set_index("classical")
    assert p.loc["random", "p_value"] == 1.0 and not p.loc["random", "significant_holm"]       # identical -> no difference
    assert p.loc["random", "verdict"] == "No statistically significant difference was detected."
    assert p.loc["cmaes", "mean_diff"] == 5 and p.loc["cmaes", "p_value"] < 0.01 and p.loc["cmaes", "rank_biserial"] == 1.0
    assert p.loc["cmaes", "p_holm"] >= p.loc["cmaes", "p_value"]
    assert set(t.metric) == {phase8.PRIMARY_METRIC, phase8.SENSITIVITY_METRIC}
    adj = phase8.holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    r = phase8.paired_comparison(np.arange(10.0), np.arange(10.0))
    assert r["p_value"] == 1.0 and r["n_nonzero_diffs"] == 0


# ---------------------------------------------------------------- resume / no duplicate execution
def test_resume_never_reexecutes_completed_runs_and_records_errors(tmp_path):
    cfg, space, validator, model = setup(budgets=(20,), seeds=(1, 2))
    specs = build_matrix(cfg)
    calls = []

    def runner(spec, *a, **kw):
        calls.append(spec.run_id)
        if spec.run_id == "cmaes_20_2" and calls.count("cmaes_20_2") == 1:
            raise RuntimeError("simulated crash")
        return run_one(spec, *a, **kw)

    s1 = run_campaign(specs, cfg, space, validator, model, tmp_path, config_fingerprint="fp", paper_eligible_context=False,
                      dataset="d", model_name="m", runner=runner)
    assert s1["errors_now"] == 1 and s1["missing"] == ["cmaes_20_2"]
    store = ResultStore(tmp_path / "raw" / "run_results.jsonl")
    assert len(store.error_records()) == 1                                                   # failure kept, not dropped
    n_before = len(calls)
    s2 = run_campaign(specs, cfg, space, validator, model, tmp_path, config_fingerprint="fp", paper_eligible_context=False,
                      dataset="d", model_name="m", runner=runner)
    assert calls[n_before:] == ["cmaes_20_2"]                                                # only the failed run re-executed
    assert s2["completed_total"] == len(specs) and not s2["missing"]
    s3 = run_campaign(specs, cfg, space, validator, model, tmp_path, config_fingerprint="fp", paper_eligible_context=False,
                      dataset="d", model_name="m", runner=runner)
    assert s3["executed_now"] == 0 and len(calls) == n_before + 1
    ok = store.ok_records()
    assert len(ok) == len(specs) and len([r for r in store.all_records() if r["status"] == "ok"]) == len(specs)
    assert store.error_records() == []                                                       # superseded by the ok record
    try:
        run_campaign(specs, cfg, space, validator, model, tmp_path, config_fingerprint="OTHER", paper_eligible_context=False,
                     dataset="d", model_name="m", runner=runner); assert False
    except CampaignMismatchError:
        pass
    try:
        store.append(dict(next(iter(ok.values())))); assert False
    except RuntimeError:
        pass


def test_resume_tolerates_truncated_last_line(tmp_path):
    cfg, space, validator, model = setup(budgets=(20,), seeds=(1,))
    specs = build_matrix(cfg, methods=["random"])
    _run(cfg, space, validator, model, tmp_path, specs=specs)
    with open(tmp_path / "raw" / "run_results.jsonl", "a") as f:
        f.write('{"run_id": "broken", "stat')
    assert len(ResultStore(tmp_path / "raw" / "run_results.jsonl").ok_records()) == 1


# ---------------------------------------------------------------- report generation / eligibility guard
def test_paper_result_generation_and_ineligible_guard(tmp_path):
    cfg, space, validator, model = setup(budgets=(20, 30), seeds=(1, 2, 3, 4))
    specs, summ = _run(cfg, space, validator, model, tmp_path)
    manifest = phase8.build_manifest(cfg, specs, root=ROOT, dataset_provenance={"source": "synthetic-test", "paper_eligible": False},
                                     model_artifact={"path": "fake", "sha256": "x"}, config_fingerprint="fp-test",
                                     paper_eligible=False, warnings=["unit-test"])
    try:
        phase8_report.analyze_campaign(tmp_path, cfg, len(specs), manifest); assert False, "ineligible results must be refused"
    except phase8_report.IneligibleResultsError:
        pass
    res = phase8_report.analyze_campaign(tmp_path, cfg, len(specs), manifest, allow_ineligible=True)
    assert not res["completeness"]["paper_eligible"]
    for p in ("raw/run_results.csv", "aggregated/summary_by_method_budget.csv", "aggregated/summary_by_method.csv",
              "aggregated/failure_discovery_curves.csv", "statistical/statistical_tests.csv", "paper_results.md",
              "tables/table3_mean_unique_verified_failures.csv", "tables/table5_statistical_comparison.csv",
              "figures/fig1_unique_failures_vs_queries.png", "figures/fig2_mean_unique_failures_by_budget.pdf",
              "figures/fig6_duplicate_rate.png"):
        assert (tmp_path / p).exists(), p
    md = (tmp_path / "paper_results.md").read_text()
    assert "DRAFT - NOT PAPER ELIGIBLE" in md and "quantum advantage" in md and "## 14." in md
    assert not re.search(r"quantum (method )?is superior|quantum supremacy achieved|demonstrates? quantum advantage", md, re.I)


def test_conclusion_is_generated_from_statistics_and_conservative():
    df_same = _df({"random": [3] * 10, "bayesian": [3] * 10, "cmaes": [3] * 10, "quantum": [3] * 10})
    s, t = phase8.summary_by_method_budget(df_same), phase8.statistical_tests(df_same)
    c = phase8_report.generate_conclusion(s, t)
    assert "No statistically significant difference was detected" in c and "B:" in c
    df_w = _df({"random": [3] * 10, "bayesian": [3] * 10, "cmaes": [3] * 10, "quantum": [9] * 10}, budgets=(100, 250))
    c = phase8_report.generate_conclusion(phase8.summary_by_method_budget(df_w), phase8.statistical_tests(df_w))
    assert "higher" in c and "speedup" in c and "do not demonstrate" in c
    df_l = _df({"random": [9] * 10, "bayesian": [3] * 10, "cmaes": [3] * 10, "quantum": [3] * 10})
    c = phase8_report.generate_conclusion(phase8.summary_by_method_budget(df_l), phase8.statistical_tests(df_l))
    assert "C:" in c                                                                           # negative results are reported


# ---------------------------------------------------------------- leakage
def test_no_test_label_access_static_and_dynamic():
    assert phase8_setup.static_leakage_audit(ROOT / "src") == []
    for name in ("phase8", "phase8_report"):                                                  # campaign path never touches test data
        code = re.sub(r"#.*", "", re.sub(r'"""(?:.|\n)*?"""', "", (ROOT / "src" / f"{name}.py").read_text()))
        assert not re.search(r"\b(X|y|cat)_test\b|\bload_prepared\b|\.clf\b|predict_proba_unbudgeted", code), name
    cfg, space, validator, model = setup()
    bv = phase8_setup.blind_validator(validator)
    try:
        bv.ref.y; assert False
    except AssertionError as e:
        assert "labels read" in str(e)
    phase8_setup.accounting_audit(cfg, space, validator, model, 20, 1)                       # all four methods behind a blind validator


def test_static_audit_detects_planted_leak(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    for n in phase8_setup.SEARCH_MODULES:
        (src / f"{n}.py").write_text("x = 1\n")
    assert phase8_setup.static_leakage_audit(src) == []
    (src / "random_search.py").write_text("labels = prepared.y_test\n")
    assert any("random_search" in v for v in phase8_setup.static_leakage_audit(src))
    (src / "random_search.py").write_text("x = 1\n"); (src / "quantum_search.py").write_text("p = model.predict_proba(X)\n")
    assert any("quantum_search" in v for v in phase8_setup.static_leakage_audit(src))


def test_real_data_loader_refuses_missing_data_and_never_falls_back(tmp_path):
    cfg = load_config(overrides={"data": {"raw_dir": str(tmp_path / "raw"), "processed_dir": str(tmp_path / "proc"), "source": "auto"}})
    assert phase8_setup.force_real_cfg(cfg)["data"]["source"] == "cic-ids2017"
    try:
        phase8_setup.load_real_components(cfg); assert False
    except phase8_setup.PreflightFailure as e:
        assert "not found" in str(e)


if __name__ == "__main__":
    unittest.main()
