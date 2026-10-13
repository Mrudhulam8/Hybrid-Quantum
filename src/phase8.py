"""Phase 8: full experimental evaluation infrastructure (matrix, resumable runner, aggregation, statistics).

Protocol (documented BEFORE any campaign run):

* BUDGETS ARE INDEPENDENT RUNS, not cumulative checkpoints. Phase 6/7 define one run as one fresh
  `QueryBudgetedModel` with exactly `budget` queries (`run_search_method`); the initial-design size and the
  proposal cap depend on the budget, so a budget-B run is NOT a prefix of a larger run. Phase 8 therefore runs
  every (method, budget, seed) cell independently: 4 x 6 x 10 = 240 runs, no budget reset, no reuse.
* Nothing in this module changes an algorithm, the validator, the oracle, the failure definition or the budgets.
  It only schedules the existing `run_search_method`, records results and analyses them.
* PRIMARY METRIC = `unique_verified_failures` (Phase 6 definition: greedy leader clustering of distinct verified
  failure candidates at `validator.dedup_resolution`). SENSITIVITY METRIC = `verified_failures` (distinct candidate
  ids). `verified_failure_events` (raw count incl. re-evaluations) is also stored. Never average ids.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .experiment_runner import METHODS, build_search_components, run_search_method
from .search_base import CommonSearchConfig
from .utils import config_hash, resolve_path, save_json

METHOD_ORDER: Tuple[str, ...] = ("random", "bayesian", "cmaes", "quantum")
PRIMARY_METRIC = "unique_verified_failures"
SENSITIVITY_METRIC = "verified_failures"
RECORD_FORMAT = 1
BUDGET_SEMANTICS = "independent_runs"          # one fresh black box per (method, budget, seed); never cumulative
EXPECTED_DATASET = "cic-ids2017"
DEFAULT_MODEL_TAG = "xgboost_cic-ids2017_binary"

REQUIRED_FIELDS = (
    "dataset", "target_model", "method", "seed", "query_budget", "queries_used", "candidate_count", "valid_candidates",
    "invalid_candidates", "unique_candidates", "duplicate_candidates", "verified_failures", "unique_verified_failures",
    "verified_failure_ids", "first_verified_failure_query", "valid_rate", "duplicate_rate", "failure_rate",
    "failures_per_100_queries", "runtime_seconds", "paper_eligible", "warnings")


class CampaignMismatchError(RuntimeError):
    """Existing results were produced under a different configuration; refusing to mix them."""


# ==========================================================================
# Experiment matrix
# ==========================================================================
@dataclass(frozen=True)
class RunSpec:
    method: str
    budget: int
    seed: int

    @property
    def run_id(self) -> str:
        return make_run_id(self.method, self.budget, self.seed)


def make_run_id(method: str, budget: int, seed: int) -> str:
    return f"{method}_{int(budget)}_{int(seed)}"


def build_matrix(cfg: Dict[str, Any], methods: Optional[Sequence[str]] = None, budgets: Optional[Sequence[int]] = None,
                 seeds: Optional[Sequence[int]] = None) -> List[RunSpec]:
    """Deterministic ordering: method (METHOD_ORDER) -> budget -> seed. Budgets/seeds come from the repository config
    (`search.common.query_budgets`, `search.common.seeds` or `reproducibility.seeds`) unless explicitly narrowed."""
    common = CommonSearchConfig.from_cfg(cfg)
    methods = tuple(methods) if methods is not None else METHOD_ORDER
    budgets = tuple(int(b) for b in (budgets if budgets is not None else common.query_budgets))
    seeds = tuple(int(s) for s in (seeds if seeds is not None else common.seeds))
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}; registered: {tuple(METHODS)}")
    if len(set(methods)) != len(methods) or len(set(budgets)) != len(budgets) or len(set(seeds)) != len(seeds):
        raise ValueError("methods, budgets and seeds must not contain duplicates")
    specs = [RunSpec(m, b, s) for m in methods for b in budgets for s in seeds]
    ids = [x.run_id for x in specs]
    assert len(ids) == len(set(ids))
    return specs


# ==========================================================================
# Result record
# ==========================================================================
def _rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def method_specific(result) -> Dict[str, Any]:
    r = result.reproducibility
    m = result.method
    if m == "bayesian":
        rep = r.get("representation", {})
        return {"surrogate_config": r.get("method_config"), "latent_dimension": rep.get("dim"),
                "n_observations": r.get("n_observations"), "gp_fits": r.get("gp_fits"), "gp_failures": r.get("gp_failures")}
    if m == "cmaes":
        rep = r.get("representation", {})
        return {"population_size": r.get("popsize"), "generation_count": r.get("generations"),
                "restarts": r.get("restarts"), "latent_dimension": rep.get("dim")}
    if m == "quantum":
        q = r.get("quantum", {})
        cost = q.get("cost", {})
        return {"backend": q.get("backend"), "backend_requested": q.get("backend_requested"),
                "framework": q.get("framework"), "framework_version": q.get("framework_version"),
                "n_qubits": q.get("n_qubits"), "depth": q.get("depth"), "shots": q.get("shots"),
                "latent_dimension": q.get("latent_dimension"), "iterations": q.get("iterations"),
                "quantum_evaluations": cost.get("circuit_executions"), "shots_sampled": cost.get("shots_sampled"),
                "target_queries_charged_by_circuit": cost.get("target_queries_charged"),
                "quantum_runtime": q.get("runtime")}
    if m == "random":
        return {"generator": r.get("generator")}
    return {}


def result_to_record(spec: RunSpec, result, *, dataset: str, target_model: str, runtime_seconds: float,
                     config_fingerprint: str) -> Dict[str, Any]:
    """Flatten a `SearchResult` into the Phase 8 record. Rates are derived here from counters the engine produced."""
    if result.method != spec.method or result.seed != spec.seed or result.query_budget != spec.budget:
        raise ValueError(f"result ({result.method}, {result.query_budget}, {result.seed}) does not match spec {spec.run_id}")
    if result.queries_used > spec.budget:
        raise ValueError(f"{spec.run_id}: queries_used {result.queries_used} exceeds budget {spec.budget}")
    q_nums = [t["query_number"] for t in (result.trajectory or [])
              if t.get("discovery_order") is not None and t.get("query_number") is not None]
    unique_cands = len(set(result.candidate_ids))
    rec = {
        "format": RECORD_FORMAT, "run_id": spec.run_id, "status": "ok", "config_fingerprint": config_fingerprint,
        "dataset": dataset, "target_model": target_model, "method": spec.method, "seed": spec.seed,
        "query_budget": spec.budget, "queries_used": result.queries_used,
        "candidate_count": result.candidates_evaluated,            # submitted to the validator (new + duplicates)
        "valid_candidates": result.candidates_valid, "invalid_candidates": result.candidates_invalid,
        "unique_candidates": unique_cands, "duplicate_candidates": result.duplicates_skipped,
        "verified_failures": result.verified_failures,             # distinct candidate ids
        "unique_verified_failures": result.unique_verified_failures,   # PRIMARY (cluster-unique)
        "verified_failure_events": result.verified_failure_events,
        "verified_failure_ids": list(result.verified_failure_ids),
        "failure_query_numbers": q_nums,                            # query index of each distinct id's discovery
        "first_verified_failure_query": result.first_failure_query,
        "valid_rate": _rate(result.candidates_valid, result.candidates_valid + result.candidates_invalid),
        "duplicate_rate": _rate(result.duplicates_skipped, result.candidates_evaluated),
        "failure_rate": _rate(result.verified_failures, result.queries_used),
        "failures_per_100_queries": 100.0 * _rate(result.unique_verified_failures, result.queries_used),
        "id_failures_per_100_queries": 100.0 * _rate(result.verified_failures, result.queries_used),
        "runtime_seconds": float(runtime_seconds), "paper_eligible": bool(result.paper_eligible),
        "warnings": list(result.warnings), "termination_reason": result.termination_reason,
        "trajectory_sha256": result.reproducibility.get("trajectory_sha256"),
        "black_box": result.black_box, "method_specific": method_specific(result),
    }
    missing = [f for f in REQUIRED_FIELDS if f not in rec]
    assert not missing, missing
    return rec


def error_record(spec: RunSpec, exc: BaseException, config_fingerprint: str, runtime_seconds: float) -> Dict[str, Any]:
    return {"format": RECORD_FORMAT, "run_id": spec.run_id, "status": "error", "config_fingerprint": config_fingerprint,
            "method": spec.method, "seed": spec.seed, "query_budget": spec.budget,
            "error_type": type(exc).__name__, "error": str(exc)[:2000], "runtime_seconds": float(runtime_seconds),
            "paper_eligible": False}


# ==========================================================================
# Resumable store (JSONL, append-only)
# ==========================================================================
class ResultStore:
    """Append-only JSONL. A completed (`status == "ok"`) run is never overwritten or re-executed. Error records are kept
    (never silently dropped) and the run is retried on resume; the later `ok` record supersedes the error for analysis."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _lines(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:        # truncated final line from an interrupted write: ignore it
                    continue
        return out

    def all_records(self) -> List[Dict[str, Any]]:
        return self._lines()

    def ok_records(self) -> Dict[str, Dict[str, Any]]:
        d: Dict[str, Dict[str, Any]] = {}
        for r in self._lines():
            if r.get("status") == "ok":
                if r["run_id"] in d:
                    raise RuntimeError(f"duplicate completed run in {self.path}: {r['run_id']}")
                d[r["run_id"]] = r
        return d

    def error_records(self) -> List[Dict[str, Any]]:
        done = set(self.ok_records())
        return [r for r in self._lines() if r.get("status") == "error" and r["run_id"] not in done]

    def append(self, rec: Dict[str, Any]) -> None:
        if rec.get("status") == "ok" and rec["run_id"] in self.ok_records():
            raise RuntimeError(f"refusing to append a second completed record for {rec['run_id']}")
        line = json.dumps(rec, sort_keys=True, default=_json_default)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ==========================================================================
# Directory layout
# ==========================================================================
SUBDIRS = ("raw", "aggregated", "tables", "figures", "statistical", "logs")


def ensure_layout(out_dir) -> Path:
    out = Path(out_dir)
    for s in SUBDIRS:
        (out / s).mkdir(parents=True, exist_ok=True)
    return out


def campaign_fingerprint(cfg: Dict[str, Any], *, reference_fingerprint: str, model_sha256: str, dataset: str) -> str:
    """Everything that must be identical for two results to be comparable/resumable together."""
    keys = ("search", "validator", "search_space", "data", "target_model", "reproducibility")
    blob = {k: cfg.get(k) for k in keys}
    blob.update(reference_fingerprint=reference_fingerprint, model_sha256=model_sha256, dataset=dataset,
                budget_semantics=BUDGET_SEMANTICS)
    return config_hash(blob)


# ==========================================================================
# Campaign execution
# ==========================================================================
def run_one(spec: RunSpec, cfg, space, validator, target_model, *, paper_eligible_context: bool, dataset: str,
            model_name: str, config_fingerprint: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = run_search_method(spec.method, cfg, space, validator, target_model, spec.budget, spec.seed,
                            paper_eligible_context=paper_eligible_context)
    return result_to_record(spec, res, dataset=dataset, target_model=model_name,
                            runtime_seconds=time.perf_counter() - t0, config_fingerprint=config_fingerprint)


def run_campaign(specs: Sequence[RunSpec], cfg, space, validator, target_model, out_dir, *, config_fingerprint: str,
                 paper_eligible_context: bool, dataset: str, model_name: str,
                 runner: Callable[..., Dict[str, Any]] = run_one, results_file: str = "run_results.jsonl",
                 progress: Optional[Callable[[str], None]] = None, stop_on_error: bool = False) -> Dict[str, Any]:
    """Execute missing runs only. Returns counts. Safe to interrupt and call again."""
    out = ensure_layout(out_dir)
    store = ResultStore(out / "raw" / results_file)
    for r in store.all_records():
        if r.get("config_fingerprint") != config_fingerprint:
            raise CampaignMismatchError(
                f"{store.path} contains runs with config_fingerprint {r.get('config_fingerprint')} != current "
                f"{config_fingerprint}. Use a new output directory; mixing configurations is not allowed.")
    say = progress or (lambda s: None)
    done = store.ok_records()
    n_run = n_err = 0
    for i, spec in enumerate(specs, 1):
        if spec.run_id in done:
            say(f"[{i}/{len(specs)}] skip {spec.run_id} (completed)")
            continue
        t0 = time.perf_counter()
        try:
            rec = runner(spec, cfg, space, validator, target_model, paper_eligible_context=paper_eligible_context,
                         dataset=dataset, model_name=model_name, config_fingerprint=config_fingerprint)
            store.append(rec)
            n_run += 1
            say(f"[{i}/{len(specs)}] done {spec.run_id}: queries={rec['queries_used']} "
                f"unique_failures={rec['unique_verified_failures']} t={rec['runtime_seconds']:.1f}s")
        except Exception as e:                                    # recorded, never silently dropped
            n_err += 1
            store.append(error_record(spec, e, config_fingerprint, time.perf_counter() - t0))
            say(f"[{i}/{len(specs)}] ERROR {spec.run_id}: {type(e).__name__}: {e}")
            if stop_on_error:
                raise
    ok = store.ok_records()
    wanted = {s.run_id for s in specs}
    return {"requested": len(specs), "completed_total": len(wanted & set(ok)), "executed_now": n_run,
            "skipped_already_complete": len(wanted & set(done)), "errors_now": n_err,
            "missing": sorted(wanted - set(ok)), "results_path": str(store.path)}


# ==========================================================================
# Loading / aggregation
# ==========================================================================
def load_run_table(path):
    """DataFrame of completed runs (one row per run_id); nested fields kept as objects."""
    import pandas as pd
    recs = list(ResultStore(path).ok_records().values())
    df = pd.DataFrame(recs)
    if len(df):
        df["method"] = pd.Categorical(df["method"], categories=list(METHOD_ORDER), ordered=True)
        df = df.sort_values(["method", "query_budget", "seed"]).reset_index(drop=True)
    return df


def flat_run_csv(df, path) -> None:
    flat = df.copy()
    for c in ("verified_failure_ids", "failure_query_numbers", "warnings"):
        if c in flat:
            flat[c] = flat[c].map(lambda v: json.dumps(v))
    for c in ("black_box", "method_specific"):
        if c in flat:
            flat[c] = flat[c].map(lambda v: json.dumps(v, sort_keys=True, default=_json_default))
    flat.to_csv(path, index=False)


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 10000, level: float = 0.95, seed: int = 0,
                 stat: Callable[[np.ndarray], float] = np.mean) -> Tuple[float, float]:
    """Percentile bootstrap CI of `stat` (default mean) over the seeds. Deterministic (fixed RNG seed).
    With n = 10 seeds this is an approximation; it describes sampling variability of the search, not causality."""
    a = np.asarray(values, float)
    if a.size == 0:
        return (math.nan, math.nan)
    if a.size == 1 or np.all(a == a[0]):
        return (float(stat(a)), float(stat(a)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = np.apply_along_axis(stat, 1, a[idx]) if stat is not np.mean else a[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return (float(lo), float(hi))


def t_ci(values: Sequence[float], level: float = 0.95) -> Tuple[float, float]:
    from scipy import stats
    a = np.asarray(values, float)
    if a.size < 2 or np.all(a == a[0]):
        m = float(a.mean()) if a.size else math.nan
        return (m, m)
    se = a.std(ddof=1) / math.sqrt(a.size)
    h = stats.t.ppf(0.5 + level / 2, a.size - 1) * se
    return (float(a.mean() - h), float(a.mean() + h))


def describe_values(values: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(values, float)
    lo, hi = bootstrap_ci(a)
    tlo, thi = t_ci(a)
    return {"n": int(a.size), "mean": float(a.mean()) if a.size else math.nan,
            "median": float(np.median(a)) if a.size else math.nan,
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()) if a.size else math.nan, "max": float(a.max()) if a.size else math.nan,
            "ci95_boot_low": lo, "ci95_boot_high": hi, "ci95_t_low": tlo, "ci95_t_high": thi}


SUMMARY_METRICS = ("unique_verified_failures", "verified_failures", "failures_per_100_queries", "id_failures_per_100_queries",
                   "valid_rate", "duplicate_rate", "failure_rate", "runtime_seconds", "queries_used",
                   "first_verified_failure_query")


def summary_by_method_budget(df):
    import pandas as pd
    rows = []
    for (m, b), g in df.groupby(["method", "query_budget"], observed=True, sort=True):
        row = {"method": str(m), "query_budget": int(b), "n_seeds": int(len(g))}
        for metric in SUMMARY_METRICS:
            vals = g[metric].dropna().to_numpy(float)
            if metric == "first_verified_failure_query":
                row["runs_without_failure"] = int(g[metric].isna().sum())
            if len(vals) == 0:
                continue
            for k, v in describe_values(vals).items():
                if k != "n":
                    row[f"{metric}__{k}"] = v
        ids = [set(x) for x in g["verified_failure_ids"]]
        row["distinct_ids_pooled_across_seeds"] = int(len(set().union(*ids))) if ids else 0     # reported separately
        row["mean_distinct_ids_per_seed"] = float(np.mean([len(x) for x in ids])) if ids else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summary_by_method(df):
    """Pooled over budgets: mean of per-run values (descriptive only; budgets are not exchangeable)."""
    import pandas as pd
    rows = []
    for m, g in df.groupby("method", observed=True, sort=True):
        row = {"method": str(m), "n_runs": int(len(g))}
        for metric in ("unique_verified_failures", "failures_per_100_queries", "valid_rate", "duplicate_rate", "runtime_seconds"):
            row[f"{metric}__mean"] = float(g[metric].mean())
            row[f"{metric}__median"] = float(g[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def discovery_curves(df, *, budget: Optional[int] = None, grid: Optional[Sequence[int]] = None):
    """Cumulative distinct verified failures vs target-model queries from the runs of ONE budget (default: the largest).
    Uses the per-run query index of every distinct failure (exact). The cluster-unique count is only known at the end of a
    run (the tracker keeps no per-step cluster history), so the curve uses distinct ids; the final point of each run
    is reported alongside in the budget summary."""
    import pandas as pd
    if budget is None:
        budget = int(df["query_budget"].max())
    sub = df[df["query_budget"] == budget]
    if grid is None:
        step = max(1, budget // 100)
        grid = sorted(set(list(range(0, budget + 1, step)) + [budget]))
    rows = []
    for m, g in sub.groupby("method", observed=True, sort=True):
        mat = np.array([[np.searchsorted(np.sort(np.asarray(q, float)), x, side="right") for x in grid]
                        for q in g["failure_query_numbers"]], float)
        for j, x in enumerate(grid):
            col = mat[:, j]
            lo, hi = bootstrap_ci(col, n_boot=2000)
            rows.append({"method": str(m), "budget": budget, "queries": int(x), "n_seeds": int(len(col)),
                         "mean_distinct_failures": float(col.mean()), "std": float(col.std(ddof=1)) if len(col) > 1 else 0.0,
                         "ci95_boot_low": lo, "ci95_boot_high": hi})
    return pd.DataFrame(rows)


# ==========================================================================
# Statistics
# ==========================================================================
def holm_adjust(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def rank_biserial_paired(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-), zero differences dropped."""
    from scipy import stats
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    wp, wm = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def paired_comparison(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """x = quantum, y = classical, paired by seed. Wilcoxon signed-rank (two-sided) is primary. Assumptions:
    paired observations (same seeds), symmetric distribution of differences; zero differences are dropped (Wilcoxon
    zero_method) and, if ALL differences are zero, p = 1. Shapiro-Wilk on the differences and a paired t-test are
    reported as sensitivity only."""
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("paired samples must be non-empty and equally long")
    d = x - y
    nz = int(np.count_nonzero(d))
    out: Dict[str, Any] = {"n_pairs": int(d.size), "n_nonzero_diffs": nz, "mean_quantum": float(x.mean()),
                           "mean_classical": float(y.mean()), "median_quantum": float(np.median(x)),
                           "median_classical": float(np.median(y)), "mean_diff": float(d.mean()),
                           "median_diff": float(np.median(d))}
    lo, hi = bootstrap_ci(d)
    out.update(mean_diff_ci95_low=lo, mean_diff_ci95_high=hi)
    if nz == 0:
        out.update(wilcoxon_stat=0.0, p_value=1.0, test="wilcoxon (all differences zero)")
    else:
        try:
            r = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="auto")
            out.update(wilcoxon_stat=float(r.statistic), p_value=float(r.pvalue), test="wilcoxon signed-rank (two-sided)")
        except ValueError as e:
            out.update(wilcoxon_stat=math.nan, p_value=math.nan, test=f"wilcoxon failed: {e}")
    out["rank_biserial"] = rank_biserial_paired(d)
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    out["cohens_dz"] = float(d.mean() / sd) if sd > 0 else (0.0 if d.mean() == 0 else math.inf)
    if d.size >= 3 and np.ptp(d) > 0:
        out["shapiro_p_of_differences"] = float(stats.shapiro(d).pvalue)
    else:
        out["shapiro_p_of_differences"] = math.nan
    if d.size >= 2 and sd > 0:
        out["paired_t_p_sensitivity"] = float(stats.ttest_rel(x, y).pvalue)
    else:
        out["paired_t_p_sensitivity"] = math.nan
    return out


def statistical_tests(df, *, metrics: Sequence[str] = (PRIMARY_METRIC, SENSITIVITY_METRIC), alpha: float = 0.05,
                      focal: str = "quantum"):
    """Quantum vs each classical method at each budget, paired by seed. Holm correction is applied per metric over the
    whole family (3 comparisons x n_budgets). Only seeds present for BOTH methods are paired."""
    import pandas as pd
    rows = []
    classical = [m for m in METHOD_ORDER if m != focal]
    budgets = sorted(df["query_budget"].unique())
    for metric in metrics:
        fam = []
        for b in budgets:
            for c in classical:
                a = df[(df.method == focal) & (df.query_budget == b)].set_index("seed")[metric]
                z = df[(df.method == c) & (df.query_budget == b)].set_index("seed")[metric]
                seeds = sorted(set(a.index) & set(z.index))
                if not seeds:
                    continue
                res = paired_comparison(a.loc[seeds].to_numpy(float), z.loc[seeds].to_numpy(float), alpha)
                res.update(metric=metric, comparison=f"{focal}_vs_{c}", classical=c, query_budget=int(b))
                fam.append(res)
        padj = holm_adjust([r["p_value"] if not math.isnan(r["p_value"]) else 1.0 for r in fam])
        for r, p in zip(fam, padj):
            r["p_holm"] = p
            r["significant_holm"] = bool(p < alpha)
            if r["significant_holm"]:
                r["verdict"] = f"{focal} higher" if r["mean_diff"] > 0 else f"{focal} lower"
            else:
                r["verdict"] = "No statistically significant difference was detected."
            rows.append(r)
    cols = ["metric", "comparison", "classical", "query_budget"]
    out = pd.DataFrame(rows)
    if len(out):
        out = out[cols + [c for c in out.columns if c not in cols]]
    return out


# ==========================================================================
# Manifest
# ==========================================================================
def _sha256_file(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted((root / "src").glob("*.py")) + [root / "configs" / "default.yaml"]:
        if p.exists():
            h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def git_version(root: Path) -> Dict[str, Any]:
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if sha.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=10)
            return {"git_commit": sha.stdout.strip(), "git_dirty": bool(dirty.stdout.strip())}
    except Exception:
        pass
    return {"git_commit": None, "note": "not a git repository"}


def software_versions() -> Dict[str, Optional[str]]:
    import importlib
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("numpy", "scipy", "sklearn", "pandas", "xgboost", "pennylane", "matplotlib", "cma"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "unknown")
        except Exception:
            out[m] = None
    return out


def build_manifest(cfg, specs: Sequence[RunSpec], *, root: Path, dataset_provenance: Dict[str, Any],
                   model_artifact: Dict[str, Any], config_fingerprint: str, paper_eligible: bool,
                   warnings: Sequence[str], quantum_runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    q = dict(cfg["search"]["quantum"])
    d = cfg["data"]
    return {
        "phase": 8, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_version": cfg["project"]["version"], **git_version(root), "source_tree_sha256_16": source_tree_hash(root),
        "dataset": {"identifier": dataset_provenance.get("source"), "provenance": dataset_provenance,
                    "split_policy": {"split_seed": d["split_seed"], "test_size": d["test_size"], "max_rows": d["max_rows"],
                                     "drop_duplicates": d["drop_duplicates"], "task": d["task"], "scaler": d["scaler"],
                                     "log_transform": d["log_transform"],
                                     "note": "fixed across search seeds; oracle/validator/search space use the TRAINING split only"}},
        "target_model_artifact": model_artifact,
        "methods": list(dict.fromkeys(s.method for s in specs)),
        "budgets": sorted({s.budget for s in specs}), "seeds": sorted({s.seed for s in specs}),
        "budget_semantics": BUDGET_SEMANTICS, "n_runs_planned": len(specs),
        "primary_metric": PRIMARY_METRIC, "sensitivity_metric": SENSITIVITY_METRIC,
        "failure_definition": "verified_failure = candidate_valid AND oracle_confidence >= threshold AND target_prediction != oracle_label (unchanged)",
        "configuration": cfg, "config_hash": config_hash(cfg), "campaign_fingerprint": config_fingerprint,
        "software": software_versions(),
        "quantum": {"backend_requested": q.get("backend"), "backend_resolved_in_runs": (quantum_runtime or {}).get("backend"),
                    "framework": (quantum_runtime or {}).get("framework"),
                    "framework_version": (quantum_runtime or {}).get("framework_version"),
                    "n_qubits": q.get("n_qubits"), "depth": q.get("depth"), "shots": q.get("shots"),
                    "latent_dimension": q.get("latent_dimension"),
                    "characterisation": "hybrid quantum-assisted search on a classical simulator; no quantum hardware; "
                                        "no claim of quantum advantage"},
        "paper_eligible": bool(paper_eligible), "warnings": list(warnings)}
