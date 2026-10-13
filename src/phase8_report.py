"""Phase 8 reporting: tables, figures and the data-driven `paper_results.md`.

Every sentence in the generated conclusion is produced from the measured statistics; nothing is written before the
data exists. If the runs are incomplete or not paper-eligible the generators REFUSE (unless `allow_ineligible=True`,
in which case the output is stamped as a non-paper draft)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .phase8 import (METHOD_ORDER, PRIMARY_METRIC, SENSITIVITY_METRIC, BUDGET_SEMANTICS, discovery_curves,
                     flat_run_csv, load_run_table, statistical_tests, summary_by_method, summary_by_method_budget)

LABELS = {"random": "Random", "bayesian": "Bayesian opt.", "cmaes": "CMA-ES", "quantum": "Quantum-assisted"}
COLORS = {"random": "#2a78d6", "bayesian": "#eb6834", "cmaes": "#1baf7a", "quantum": "#4a3aa7"}   # validated palette slots 1,2,3,7
MARKERS = {"random": "o", "bayesian": "s", "cmaes": "^", "quantum": "D"}
LINESTYLES = {"random": "-", "bayesian": "--", "cmaes": "-.", "quantum": ":"}
HIST_MODEL_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


class IneligibleResultsError(RuntimeError):
    pass


def completeness(df, expected: int) -> Dict[str, Any]:
    n = int(len(df))
    elig = bool(n and df["paper_eligible"].all()) if n else False
    datasets = sorted(set(df["dataset"])) if n else []
    return {"n_completed": n, "n_expected": expected, "complete": n == expected,
            "all_runs_paper_eligible": elig, "datasets": datasets,
            "real_cic": datasets == ["cic-ids2017"], "n_runs_with_warnings": int(sum(bool(w) for w in df["warnings"])) if n else 0}


def check_eligible(df, expected: int, allow_ineligible: bool) -> Dict[str, Any]:
    c = completeness(df, expected)
    ok = c["complete"] and c["all_runs_paper_eligible"] and c["real_cic"]
    c["paper_eligible"] = bool(ok)
    if not ok and not allow_ineligible:
        raise IneligibleResultsError(
            f"results are not paper-eligible/complete ({c}); refusing to write paper artefacts. "
            f"Pass allow_ineligible=True only for a clearly-marked draft/mechanics check.")
    return c


def _fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _md_table(df) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _save_table(df, tables_dir: Path, name: str) -> None:
    df.to_csv(tables_dir / f"{name}.csv", index=False)
    (tables_dir / f"{name}.md").write_text(_md_table(df) + "\n", encoding="utf-8")


# ==========================================================================
# Tables
# ==========================================================================
def make_tables(df, summ, tests, cfg, tables_dir: Path, target_model_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    import pandas as pd
    tables_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    # Table 1: target model (historical training-time test-split metrics; NOT search results)
    if target_model_report:
        m = target_model_report["metrics"]
        t1 = pd.DataFrame([{"metric": k, "value": m.get(k)} for k in HIST_MODEL_METRICS] +
                          [{"metric": "paper_eligible", "value": target_model_report.get("paper_eligible")},
                           {"metric": "test_proba_sha256", "value": target_model_report.get("test_proba_sha256")}])
    else:
        t1 = pd.DataFrame([{"metric": "target-model report", "value": "UNAVAILABLE (Phase 3 report json not found)"}])
    _save_table(t1, tables_dir, "table1_target_model_performance"); out["table1"] = t1
    # Table 2: search configuration
    s = cfg["search"]
    rows = [{"method": "all", "parameter": "query_budgets (independent runs)", "value": s["common"]["query_budgets"]},
            {"method": "all", "parameter": "seeds", "value": s["common"].get("seeds") or cfg["reproducibility"]["seeds"]},
            {"method": "all", "parameter": "duplicate_policy", "value": s["common"]["duplicate_policy"]},
            {"method": "all", "parameter": "max_candidates_per_query", "value": s["common"]["max_candidates_per_query"]},
            {"method": "all", "parameter": "failure rule", "value": cfg["validator"]["failure_rule"]}]
    for m in METHOD_ORDER:
        for k, v in s[m].items():
            rows.append({"method": m, "parameter": k, "value": v})
    t2 = pd.DataFrame(rows); _save_table(t2, tables_dir, "table2_search_configuration"); out["table2"] = t2
    # Table 3 / 4
    def pivot(col_mean, col_std, col_lo, col_hi, nd=2):
        rows = []
        for b in sorted(summ["query_budget"].unique()):
            r = {"budget": int(b)}
            for m in METHOD_ORDER:
                x = summ[(summ.method == m) & (summ.query_budget == b)]
                if len(x):
                    x = x.iloc[0]
                    r[LABELS[m]] = (f"{_fmt(x[col_mean], nd)} ± {_fmt(x[col_std], nd)} "
                                    f"[{_fmt(x[col_lo], nd)}, {_fmt(x[col_hi], nd)}]")
            rows.append(r)
        return pd.DataFrame(rows)
    t3 = pivot("unique_verified_failures__mean", "unique_verified_failures__std",
               "unique_verified_failures__ci95_boot_low", "unique_verified_failures__ci95_boot_high")
    _save_table(t3, tables_dir, "table3_mean_unique_verified_failures"); out["table3"] = t3
    t4 = pivot("failures_per_100_queries__mean", "failures_per_100_queries__std",
               "failures_per_100_queries__ci95_boot_low", "failures_per_100_queries__ci95_boot_high", 3)
    _save_table(t4, tables_dir, "table4_failures_per_100_queries"); out["table4"] = t4
    # Table 5
    p = tests[tests.metric == PRIMARY_METRIC]
    t5 = pd.DataFrame({"budget": p.query_budget, "comparison": p.comparison, "mean_quantum": p.mean_quantum.round(2),
                       "mean_classical": p.mean_classical.round(2), "mean_diff": p.mean_diff.round(2),
                       "median_diff": p.median_diff.round(2), "rank_biserial": p.rank_biserial.round(3),
                       "p_raw": p.p_value.round(4), "p_holm": p.p_holm.round(4), "verdict": p.verdict})
    _save_table(t5, tables_dir, "table5_statistical_comparison"); out["table5"] = t5
    # Table 6
    rows = []
    for m in METHOD_ORDER:
        g = df[df.method == m]
        if not len(g):
            continue
        ms = g["method_specific"].iloc[0]
        rows.append({"method": LABELS[m], "mean_runtime_s": round(g.runtime_seconds.mean(), 2),
                     "median_runtime_s": round(g.runtime_seconds.median(), 2), "max_runtime_s": round(g.runtime_seconds.max(), 2),
                     "latent_dimension": ms.get("latent_dimension"),
                     "details": json.dumps({k: ms[k] for k in ("backend", "n_qubits", "depth", "shots", "framework", "framework_version")
                                            if k in ms}) if m == "quantum" else ""})
    t6 = pd.DataFrame(rows); _save_table(t6, tables_dir, "table6_runtime_and_computational_configuration"); out["table6"] = t6
    return out


# ==========================================================================
# Figures
# ==========================================================================
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.5, "legend.frameon": False, "lines.linewidth": 1.6, "font.family": "DejaVu Sans"})
    return plt


def _save(fig, fig_dir: Path, name: str, plt, watermark: Optional[str]) -> List[str]:
    if watermark:
        fig.text(0.5, 0.5, watermark, fontsize=26, color="grey", alpha=0.18, ha="center", va="center", rotation=25)
    paths = []
    for ext in ("png", "pdf"):
        p = fig_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(str(p))
    plt.close(fig)
    return paths


def make_figures(df, summ, curves, fig_dir: Path, *, watermark: Optional[str] = None) -> List[str]:
    plt = _style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: List[str] = []
    methods = [m for m in METHOD_ORDER if m in set(df.method.astype(str))]
    budgets = sorted(df.query_budget.unique())
    W = 3.4 * 2   # ~ two-column width in inches

    # 1. discovery curves
    fig, ax = plt.subplots(figsize=(W * 0.6, 3.0))
    for m in methods:
        c = curves[curves.method == m]
        ax.plot(c.queries, c.mean_distinct_failures, color=COLORS[m], ls=LINESTYLES[m], label=LABELS[m])
        ax.fill_between(c.queries, c.ci95_boot_low, c.ci95_boot_high, color=COLORS[m], alpha=0.12, linewidth=0)
    ax.set_xlabel("Target-model queries"); ax.set_ylabel("Distinct verified failures (mean of seeds)")
    ax.set_title(f"Failure discovery vs queries (budget {int(curves.budget.iloc[0])}; band = 95% bootstrap CI)")
    ax.legend(); made += _save(fig, fig_dir, "fig1_unique_failures_vs_queries", plt, watermark)

    # 2. grouped bars at each budget (primary metric)
    fig, ax = plt.subplots(figsize=(W * 0.8, 3.0))
    n, wd = len(methods), 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        xs, ys, lo, hi = [], [], [], []
        for j, b in enumerate(budgets):
            r = summ[(summ.method == m) & (summ.query_budget == b)]
            if len(r):
                r = r.iloc[0]; xs.append(j + (i - (n - 1) / 2) * wd); ys.append(r["unique_verified_failures__mean"])
                lo.append(r["unique_verified_failures__mean"] - r["unique_verified_failures__ci95_boot_low"])
                hi.append(r["unique_verified_failures__ci95_boot_high"] - r["unique_verified_failures__mean"])
        ax.bar(xs, ys, width=wd * 0.92, color=COLORS[m], label=LABELS[m], yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(range(len(budgets))); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Target-model query budget (independent runs)"); ax.set_ylabel("Unique verified failures (mean ± 95% CI)")
    ax.set_ylim(bottom=0); ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0)); made += _save(fig, fig_dir, "fig2_mean_unique_failures_by_budget", plt, watermark)

    # 3. efficiency
    fig, ax = plt.subplots(figsize=(W * 0.6, 3.0))
    for m in methods:
        r = summ[summ.method == m].sort_values("query_budget")
        ax.errorbar(range(len(r)), r["failures_per_100_queries__mean"],
                    yerr=[r["failures_per_100_queries__mean"] - r["failures_per_100_queries__ci95_boot_low"],
                          r["failures_per_100_queries__ci95_boot_high"] - r["failures_per_100_queries__mean"]],
                    color=COLORS[m], marker=MARKERS[m], ls=LINESTYLES[m], capsize=2, label=LABELS[m], ms=4)
    ax.set_xticks(range(len(budgets))); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Target-model query budget"); ax.set_ylabel("Unique verified failures per 100 queries")
    ax.set_ylim(bottom=0); ax.legend(); made += _save(fig, fig_dir, "fig3_failures_per_100_queries", plt, watermark)

    # 4. first-failure query (largest budget), box + points; runs with no failure are counted in the label
    bmax = budgets[-1]
    fig, ax = plt.subplots(figsize=(W * 0.8, 3.2))
    data, labs = [], []
    for m in methods:
        g = df[(df.method == m) & (df.query_budget == bmax)]
        v = g.first_verified_failure_query.dropna().to_numpy(float)
        data.append(v); labs.append(f"{LABELS[m]}\n(no failure: {int(g.first_verified_failure_query.isna().sum())}/{len(g)})")
    bp = ax.boxplot([d if len(d) else [np.nan] for d in data], tick_labels=labs, patch_artist=True, widths=0.5,
                    medianprops={"color": "black"})
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.35)
    for i, (d, m) in enumerate(zip(data, methods), 1):
        ax.scatter(np.full(len(d), i) + np.linspace(-0.12, 0.12, max(len(d), 1))[:len(d)], d, s=10, color=COLORS[m], zorder=3)
    ax.set_ylabel("Query index of first verified failure"); ax.set_title(f"First-failure query (budget {bmax})")
    ax.tick_params(axis="x", labelsize=7); made += _save(fig, fig_dir, "fig4_first_failure_query", plt, watermark)

    # 5/6. valid and duplicate rate
    for key, ylab, name in (("valid_rate", "Valid-candidate rate", "fig5_valid_candidate_rate"),
                            ("duplicate_rate", "Duplicate rate", "fig6_duplicate_rate")):
        fig, ax = plt.subplots(figsize=(W * 0.6, 3.0))
        for m in methods:
            r = summ[summ.method == m].sort_values("query_budget")
            ax.errorbar(range(len(r)), r[f"{key}__mean"],
                        yerr=[r[f"{key}__mean"] - r[f"{key}__ci95_boot_low"], r[f"{key}__ci95_boot_high"] - r[f"{key}__mean"]],
                        color=COLORS[m], marker=MARKERS[m], ls=LINESTYLES[m], capsize=2, label=LABELS[m], ms=4)
        ax.set_xticks(range(len(budgets))); ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel("Target-model query budget"); ax.set_ylabel(ylab); ax.set_ylim(0, 1.02); ax.legend()
        made += _save(fig, fig_dir, name, plt, watermark)
    return made


# ==========================================================================
# paper_results.md (conclusion generated from the statistics)
# ==========================================================================
def generate_conclusion(summ, tests) -> str:
    if tests is None:
        p = None
    else:
        p = tests[tests.metric == PRIMARY_METRIC] if len(tests) else tests.iloc[:0]
    budgets = sorted(p.query_budget.unique()) if p is not None and len(p) else []
    lines: List[str] = []
    sig = p[p.significant_holm] if p is not None and len(p) else (p.iloc[:0] if p is not None else None)
    q = summ[summ.method == "quantum"].set_index("query_budget") if summ is not None and len(summ) else None
    for b in budgets:
        parts = []
        for m in ("random", "bayesian", "cmaes"):
            r = summ[(summ.method == m) & (summ.query_budget == b)] if summ is not None and len(summ) else []
            if len(r):
                parts.append(f"{r.iloc[0]['unique_verified_failures__mean']:.2f} for {LABELS[m]}")
        if q is not None and b in q.index:
            lines.append(f"At a query budget of {b}, the quantum-assisted method discovered a mean of "
                         f"{q.loc[b, 'unique_verified_failures__mean']:.2f} unique verified failures per seed, compared with "
                         + ", ".join(parts) + ".")
    if sig is None or len(sig) == 0:
        lines.append("No statistically significant difference was detected between the hybrid quantum-assisted search and any "
                     "classical baseline at any budget after Holm correction (alpha = 0.05, paired Wilcoxon signed-rank over seeds).")
    else:
        for _, r in sig.iterrows():
            lines.append(f"Budget {int(r.query_budget)}, {r.comparison}: {r.verdict} (mean difference {r.mean_diff:+.2f}, "
                         f"Holm-adjusted p = {r.p_holm:.4f}, rank-biserial r = {r.rank_biserial:+.2f}).")
        n_hi = int((sig["mean_diff"] > 0).sum()); n_lo = int((sig["mean_diff"] < 0).sum())
        lines.append(f"Of {len(p)} primary comparisons, {n_hi} were significantly in favour of the quantum-assisted method and "
                     f"{n_lo} significantly against it (Holm-adjusted); the remaining {len(p) - len(sig)} showed no statistically "
                     f"significant difference.")
    # classification A/B/C
    pos = sig[sig.mean_diff > 0] if sig is not None and len(sig) else sig
    neg = sig[sig.mean_diff < 0] if sig is not None and len(sig) else sig
    if len(neg):
        cls = "C: the quantum-assisted search performed worse than at least one classical baseline at one or more budgets."
    elif len(pos) and all(((pos.classical == c).any()) for c in ("random", "bayesian", "cmaes")):
        cls = "A (conditional): the quantum-assisted search was significantly higher than every classical baseline at one or more budgets."
    elif len(pos):
        cls = "Mixed: significantly higher than some baselines at some budgets, with no significant deficit."
    else:
        cls = "B: the quantum-assisted search performed similarly to the classical baselines (no significant differences detected)."
    lines.append(f"Outcome category: {cls}")
    lines.append("These results describe search effectiveness per target-model query on a classical simulator. They do not "
                 "demonstrate quantum speedup, quantum advantage or any complexity benefit.")
    return "\n\n".join(lines)


def write_paper_results(path: Path, *, df, summ, summ_method, tests, tables, cfg, comp: Dict[str, Any], manifest: Dict[str, Any],
                        target_model_report: Optional[Dict[str, Any]], figures: List[str], curves_budget: int) -> Path:
    banner = ""
    if not comp["paper_eligible"]:
        banner = ("> **DRAFT - NOT PAPER ELIGIBLE.** " + f"completed={comp['n_completed']}/{comp['n_expected']}, "
                  f"datasets={comp['datasets']}, all_runs_paper_eligible={comp['all_runs_paper_eligible']}. "
                  "Do not cite any number below.\n\n")
    q = df[df.method == "quantum"]
    qs = q["method_specific"].iloc[0] if len(q) else {}
    t3, t4, t5 = tables["table3"], tables["table4"], tables["table5"]
    sens = tests[tests.metric == SENSITIVITY_METRIC]
    n_err = comp.get("n_error_records", 0)
    md = f"""# Phase 8 - Experimental Results

{banner}## 1. Experimental setup
Research question: under an identical target-model query budget, does hybrid quantum-assisted search discover rare verified
failure modes of a fixed model more effectively than Random Search, Bayesian Optimization or CMA-ES? Four methods x
{len(summ.query_budget.unique())} budgets x {len(df.seed.unique())} seeds = {comp['n_expected']} runs
({comp['n_completed']} completed). The XGBoost model is a fixed target; the result is search efficiency, not model accuracy.

## 2. Dataset
{manifest['dataset']['identifier']} (paper_eligible={manifest['dataset']['provenance'].get('paper_eligible')}); split seed
{manifest['dataset']['split_policy']['split_seed']}, test_size {manifest['dataset']['split_policy']['test_size']},
max_rows {manifest['dataset']['split_policy']['max_rows']}. The split is fixed across search seeds; the oracle, validator and search space use the training split only.

## 3. Target model
Historical Phase 3 training-time metrics on the held-out test split (not search results):

{_md_table(tables['table1'])}

Artifact: `{manifest['target_model_artifact'].get('path')}` (sha256 `{manifest['target_model_artifact'].get('sha256')}`).

## 4. Reference oracle
Reference k-NN oracle with Mahalanobis and nearest-neighbour validity gates, fitted on the training split only (unchanged from Phase 4).
A verified failure = candidate_valid AND oracle_confidence >= {cfg['validator']['failure_rule']['min_oracle_confidence']} AND target_prediction != oracle_label. Uncertainty alone is never a failure.

## 5. Search methods
Random (reference-anchored local perturbation), Bayesian optimization (GP, PCA latent dim {cfg['search']['bayesian']['n_components']}, EI), CMA-ES (full space), and a
hybrid quantum-assisted search (latent dim {cfg['search']['quantum']['latent_dimension']}). Parameters in Table 2 (`tables/table2_search_configuration.csv`).
Note the methods use different latent representations as defined in Phases 6/7; this is a design property, not tuned here.

## 6. Query-budget protocol
Budgets {sorted(int(b) for b in summ.query_budget.unique())} are **independent runs** (`{BUDGET_SEMANTICS}`): each run gets a fresh
query-budgeted black box; there is no reset, reuse or cumulative checkpointing. One target-model prediction = one query; invalid
candidates rejected before evaluation, circuit executions and surrogate computations cost zero queries.

## 7. Seed protocol
Seeds {sorted(int(s) for s in df.seed.unique())} (repository configuration), identical for all methods.

## 8. Evaluation metrics
Primary: `unique_verified_failures` (distinct verified failures clustered at validator.dedup_resolution = {cfg['validator']['dedup_resolution']}), and per 100 queries.
Sensitivity: `verified_failures` (distinct candidate ids). Secondary: first-failure query, valid rate, duplicate rate, runtime.
Failure-category diversity is **not reported**: per-failure category metadata is not recorded by the Phase 6/7 tracker.

## 9. Main results
Mean ± std [95% bootstrap CI] of unique verified failures per seed:

{_md_table(t3)}

Unique verified failures per 100 queries:

{_md_table(t4)}

Distinct failure ids pooled across seeds are in `aggregated/summary_by_method_budget.csv` (`distinct_ids_pooled_across_seeds`), kept separate from per-seed means.
Figures: {', '.join(Path(f).name for f in figures if f.endswith('.png'))} (curves use runs at budget {curves_budget}).

## 10. Statistical analysis
Paired by seed, Wilcoxon signed-rank (two-sided), Holm correction over the {len(t5)} primary comparisons; effect size = matched-pairs rank-biserial r;
Shapiro-Wilk on differences and paired t-test reported as sensitivity in `statistical/statistical_tests.csv`.

{_md_table(t5)}

Sensitivity metric (distinct ids): {int(sens.significant_holm.sum())} of {len(sens)} comparisons significant after Holm correction.

## 11. Quantum-specific analysis
Backend `{qs.get('backend')}` ({qs.get('framework')} {qs.get('framework_version')}), qubits {qs.get('n_qubits')}, depth {qs.get('depth')}, shots {qs.get('shots')},
latent dimension {qs.get('latent_dimension')}. Mean quantum iterations {q['method_specific'].map(lambda m: m.get('iterations') or 0).mean():.1f}, mean circuit evaluations
{q['method_specific'].map(lambda m: m.get('quantum_evaluations') or 0).mean():.1f} per run; target queries charged by circuits: {int(q['method_specific'].map(lambda m: m.get('target_queries_charged_by_circuit') or 0).sum())}.
This is a classical simulation of a quantum-assisted candidate-proposal mechanism, not execution on quantum hardware.

## 12. Limitations
* Single dataset, single target model, single fixed split; 10 seeds (limited power; exact Wilcoxon minimum two-sided p = 2/2^10 before correction).
* Hyper-parameters are a-priori placeholders, not tuned; methods differ in latent dimension (quantum {cfg['search']['quantum']['latent_dimension']}, Bayesian {cfg['search']['bayesian']['n_components']}, CMA-ES full).
* The failure verdict depends on the k-NN reference oracle and validity gates, not on ground truth.
* Discovery curves use distinct ids; cluster-unique counts are final-only.
* XGBoost multithreaded prediction is reproducible only for fixed hardware/thread count. Runtime comparisons mix simulator and classical overheads and are not a primary result.
* Failed/errored run records: {n_err}.

## 13. Reproducibility information
Campaign fingerprint `{manifest['campaign_fingerprint']}`, config hash `{manifest['config_hash']}`, source hash `{manifest['source_tree_sha256_16']}`,
git `{manifest.get('git_commit')}`, software {json.dumps(manifest['software'])}. See `experiment_manifest.json`.

## 14. Scientifically justified conclusion
{generate_conclusion(summ, tests)}
"""
    path.write_text(md, encoding="utf-8")
    return path


def analyze_campaign(out_dir, cfg, specs_n: int, manifest: Dict[str, Any], *, allow_ineligible: bool = False,
                     target_model_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """STEPS 10-14: aggregate -> statistics -> tables -> figures -> paper_results.md."""
    out = Path(out_dir)
    df = load_run_table(out / "raw" / "run_results.jsonl")
    if not len(df):
        raise IneligibleResultsError("no completed runs")
    comp = check_eligible(df, specs_n, allow_ineligible)
    from .phase8 import ResultStore
    comp["n_error_records"] = len(ResultStore(out / "raw" / "run_results.jsonl").error_records())
    flat_run_csv(df, out / "raw" / "run_results.csv")
    summ = summary_by_method_budget(df)
    summ.to_csv(out / "aggregated" / "summary_by_method_budget.csv", index=False)
    sm = summary_by_method(df); sm.to_csv(out / "aggregated" / "summary_by_method.csv", index=False)
    curves = discovery_curves(df); curves.to_csv(out / "aggregated" / "failure_discovery_curves.csv", index=False)
    tests = statistical_tests(df); tests.to_csv(out / "statistical" / "statistical_tests.csv", index=False)
    tables = make_tables(df, summ, tests, cfg, out / "tables", target_model_report)
    wm = None if comp["paper_eligible"] else "NOT PAPER ELIGIBLE"
    figs = make_figures(df, summ, curves, out / "figures", watermark=wm)
    p = write_paper_results(out / "paper_results.md", df=df, summ=summ, summ_method=sm, tests=tests, tables=tables, cfg=cfg,
                            comp=comp, manifest=manifest, target_model_report=target_model_report, figures=figs,
                            curves_budget=int(curves.budget.iloc[0]))
    return {"completeness": comp, "paper_results": str(p), "n_figures": len(figs), "n_tests": int(len(tests))}
