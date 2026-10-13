# Phase 8 - Experimental Results

## 1. Experimental setup
Research question: under an identical target-model query budget, does hybrid quantum-assisted search discover rare verified
failure modes of a fixed model more effectively than Random Search, Bayesian Optimization or CMA-ES? Four methods x
6 budgets x 10 seeds = 240 runs
(240 completed). The XGBoost model is a fixed target; the result is search efficiency, not model accuracy.

## 2. Dataset
cic-ids2017 (paper_eligible=True); split seed
42, test_size 0.3,
max_rows 200000. The split is fixed across search seeds; the oracle, validator and search space use the training split only.

## 3. Target model
Historical Phase 3 training-time metrics on the held-out test split (not search results):

| metric | value |
|---|---|
| accuracy | 0.9987999599986667 |
| precision | 0.9953760498254223 |
| recall | 0.9978242361176805 |
| f1 | 0.9965986394557823 |
| roc_auc | 0.999960118112016 |
| paper_eligible | True |
| test_proba_sha256 | 7cf6382981729f1d |

Artifact: `C:\Users\mrudhula\Downloads\Hybrid Quantum\hqfd_phase8_fixed\hybrid_quantum_failure_discovery\models\xgboost_cic-ids2017_binary.json` (sha256 `a66ca7de31df986b1cce4bc6b13c8857c63c2475983bd7f6d2a3edadd928b635`).

## 4. Reference oracle
Reference k-NN oracle with Mahalanobis and nearest-neighbour validity gates, fitted on the training split only (unchanged from Phase 4).
A verified failure = candidate_valid AND oracle_confidence >= 0.8 AND target_prediction != oracle_label. Uncertainty alone is never a failure.

## 5. Search methods
Random (reference-anchored local perturbation), Bayesian optimization (GP, PCA latent dim 10, EI), CMA-ES (full space), and a
hybrid quantum-assisted search (latent dim 4). Parameters in Table 2 (`tables/table2_search_configuration.csv`).
Note the methods use different latent representations as defined in Phases 6/7; this is a design property, not tuned here.

## 6. Query-budget protocol
Budgets [100, 250, 500, 1000, 2500, 5000] are **independent runs** (`independent_runs`): each run gets a fresh
query-budgeted black box; there is no reset, reuse or cumulative checkpointing. One target-model prediction = one query; invalid
candidates rejected before evaluation, circuit executions and surrogate computations cost zero queries.

## 7. Seed protocol
Seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (repository configuration), identical for all methods.

## 8. Evaluation metrics
Primary: `unique_verified_failures` (distinct verified failures clustered at validator.dedup_resolution = 0.05), and per 100 queries.
Sensitivity: `verified_failures` (distinct candidate ids). Secondary: first-failure query, valid rate, duplicate rate, runtime.
Failure-category diversity is **not reported**: per-failure category metadata is not recorded by the Phase 6/7 tracker.

## 9. Main results
Mean ± std [95% bootstrap CI] of unique verified failures per seed:

| budget | Random | Bayesian opt. | CMA-ES | Quantum-assisted |
|---|---|---|---|---|
| 100 | 6.00 ± 2.62 [4.50, 7.50] | 14.50 ± 33.20 [0.00, 38.00] | 5.10 ± 16.13 [0.00, 15.30] | 0.00 ± 0.00 [0.00, 0.00] |
| 250 | 13.00 ± 3.40 [11.10, 15.00] | 16.90 ± 49.37 [0.00, 49.50] | 18.20 ± 51.86 [0.00, 52.90] | 0.00 ± 0.00 [0.00, 0.00] |
| 500 | 25.80 ± 5.05 [22.80, 28.80] | 81.50 ± 174.58 [0.20, 196.20] | 50.50 ± 115.33 [0.00, 135.30] | 0.00 ± 0.00 [0.00, 0.00] |
| 1000 | 51.00 ± 6.82 [46.80, 54.80] | 86.20 ± 152.96 [3.90, 183.30] | 148.10 ± 304.93 [0.00, 346.60] | 0.00 ± 0.00 [0.00, 0.00] |
| 2500 | 131.40 ± 10.74 [125.40, 138.00] | 482.90 ± 784.46 [101.40, 984.20] | 471.00 ± 905.51 [0.00, 1086.60] | 0.00 ± 0.00 [0.00, 0.00] |
| 5000 | 265.70 ± 10.49 [259.60, 271.90] | 2231.10 ± 1603.08 [1312.79, 3177.00] | 1243.50 ± 1854.86 [253.70, 2452.90] | 0.00 ± 0.00 [0.00, 0.00] |

Unique verified failures per 100 queries:

| budget | Random | Bayesian opt. | CMA-ES | Quantum-assisted |
|---|---|---|---|---|
| 100 | 6.000 ± 2.625 [4.500, 7.500] | 14.500 ± 33.204 [0.000, 38.000] | 5.100 ± 16.128 [0.000, 15.300] | 0.000 ± 0.000 [0.000, 0.000] |
| 250 | 5.200 ± 1.360 [4.440, 6.000] | 6.760 ± 19.748 [0.000, 19.800] | 7.280 ± 20.742 [0.000, 21.160] | 0.000 ± 0.000 [0.000, 0.000] |
| 500 | 5.160 ± 1.010 [4.560, 5.760] | 16.300 ± 34.915 [0.040, 39.240] | 10.100 ± 23.067 [0.000, 27.060] | 0.000 ± 0.000 [0.000, 0.000] |
| 1000 | 5.100 ± 0.682 [4.680, 5.480] | 8.620 ± 15.296 [0.390, 18.330] | 14.810 ± 30.493 [0.000, 34.660] | 0.000 ± 0.000 [0.000, 0.000] |
| 2500 | 5.256 ± 0.430 [5.016, 5.520] | 19.316 ± 31.379 [4.056, 39.368] | 18.840 ± 36.220 [0.000, 43.464] | 0.000 ± 0.000 [0.000, 0.000] |
| 5000 | 5.314 ± 0.210 [5.192, 5.438] | 44.622 ± 32.062 [26.256, 63.540] | 24.870 ± 37.097 [5.074, 49.058] | 0.000 ± 0.000 [0.000, 0.000] |

Distinct failure ids pooled across seeds are in `aggregated/summary_by_method_budget.csv` (`distinct_ids_pooled_across_seeds`), kept separate from per-seed means.
Figures: fig1_unique_failures_vs_queries.png, fig2_mean_unique_failures_by_budget.png, fig3_failures_per_100_queries.png, fig4_first_failure_query.png, fig5_valid_candidate_rate.png, fig6_duplicate_rate.png (curves use runs at budget 5000).

## 10. Statistical analysis
Paired by seed, Wilcoxon signed-rank (two-sided), Holm correction over the 18 primary comparisons; effect size = matched-pairs rank-biserial r;
Shapiro-Wilk on differences and paired t-test reported as sensitivity in `statistical/statistical_tests.csv`.

| budget | comparison | mean_quantum | mean_classical | mean_diff | median_diff | rank_biserial | p_raw | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 100 | quantum_vs_random | 0.0 | 6.0 | -6.0 | -5.5 | -1.0 | 0.002 | 0.0352 | quantum lower |
| 100 | quantum_vs_bayesian | 0.0 | 14.5 | -14.5 | 0.0 | -1.0 | 0.5 | 1.0 | No statistically significant difference was detected. |
| 100 | quantum_vs_cmaes | 0.0 | 5.1 | -5.1 | 0.0 | -1.0 | 1.0 | 1.0 | No statistically significant difference was detected. |
| 250 | quantum_vs_random | 0.0 | 13.0 | -13.0 | -12.0 | -1.0 | 0.002 | 0.0352 | quantum lower |
| 250 | quantum_vs_bayesian | 0.0 | 16.9 | -16.9 | 0.0 | -1.0 | 0.5 | 1.0 | No statistically significant difference was detected. |
| 250 | quantum_vs_cmaes | 0.0 | 18.2 | -18.2 | 0.0 | -1.0 | 0.5 | 1.0 | No statistically significant difference was detected. |
| 500 | quantum_vs_random | 0.0 | 25.8 | -25.8 | -26.0 | -1.0 | 0.002 | 0.0352 | quantum lower |
| 500 | quantum_vs_bayesian | 0.0 | 81.5 | -81.5 | 0.0 | -1.0 | 0.125 | 1.0 | No statistically significant difference was detected. |
| 500 | quantum_vs_cmaes | 0.0 | 50.5 | -50.5 | 0.0 | -1.0 | 0.25 | 1.0 | No statistically significant difference was detected. |
| 1000 | quantum_vs_random | 0.0 | 51.0 | -51.0 | -51.5 | -1.0 | 0.002 | 0.0352 | quantum lower |
| 1000 | quantum_vs_bayesian | 0.0 | 86.2 | -86.2 | 0.0 | -1.0 | 0.125 | 1.0 | No statistically significant difference was detected. |
| 1000 | quantum_vs_cmaes | 0.0 | 148.1 | -148.1 | 0.0 | -1.0 | 0.25 | 1.0 | No statistically significant difference was detected. |
| 2500 | quantum_vs_random | 0.0 | 131.4 | -131.4 | -128.5 | -1.0 | 0.002 | 0.0352 | quantum lower |
| 2500 | quantum_vs_bayesian | 0.0 | 482.9 | -482.9 | 0.0 | -1.0 | 0.125 | 1.0 | No statistically significant difference was detected. |
| 2500 | quantum_vs_cmaes | 0.0 | 471.0 | -471.0 | 0.0 | -1.0 | 0.25 | 1.0 | No statistically significant difference was detected. |
| 5000 | quantum_vs_random | 0.0 | 265.7 | -265.7 | -266.0 | -1.0 | 0.002 | 0.0352 | quantum lower |
| 5000 | quantum_vs_bayesian | 0.0 | 2231.1 | -2231.1 | -2288.5 | -1.0 | 0.002 | 0.0352 | quantum lower |
| 5000 | quantum_vs_cmaes | 0.0 | 1243.5 | -1243.5 | -407.0 | -1.0 | 0.0625 | 0.6875 | No statistically significant difference was detected. |

Sensitivity metric (distinct ids): 7 of 18 comparisons significant after Holm correction.

## 11. Quantum-specific analysis
Backend `pennylane` (pennylane (default.qubit, analytic probabilities) 0.45.1), qubits 4, depth 2, shots 256,
latent dimension 4. Mean quantum iterations 260.3, mean circuit evaluations
107257.3 per run; target queries charged by circuits: 0.
This is a classical simulation of a quantum-assisted candidate-proposal mechanism, not execution on quantum hardware.

## 12. Limitations
* Single dataset, single target model, single fixed split; 10 seeds (limited power; exact Wilcoxon minimum two-sided p = 2/2^10 before correction).
* Hyper-parameters are a-priori placeholders, not tuned; methods differ in latent dimension (quantum 4, Bayesian 10, CMA-ES full).
* The failure verdict depends on the k-NN reference oracle and validity gates, not on ground truth.
* Discovery curves use distinct ids; cluster-unique counts are final-only.
* XGBoost multithreaded prediction is reproducible only for fixed hardware/thread count. Runtime comparisons mix simulator and classical overheads and are not a primary result.
* Failed/errored run records: 0.

## 13. Reproducibility information
Campaign fingerprint `bab7dcaeda7b`, config hash `b2a50186eabb`, source hash `a63fffeb21f12597`,
git `None`, software {"python": "3.11.4", "platform": "Windows-10-10.0.26300-SP0", "numpy": "2.4.6", "scipy": "1.17.1", "sklearn": "1.9.1", "pandas": "3.0.6", "xgboost": "3.2.0", "pennylane": "0.45.1", "matplotlib": "3.11.2", "cma": "4.5.0"}. See `experiment_manifest.json`.

## 14. Scientifically justified conclusion
At a query budget of 100, the quantum-assisted method discovered a mean of 0.00 unique verified failures per seed, compared with 6.00 for Random, 14.50 for Bayesian opt., 5.10 for CMA-ES.

At a query budget of 250, the quantum-assisted method discovered a mean of 0.00 unique verified failures per seed, compared with 13.00 for Random, 16.90 for Bayesian opt., 18.20 for CMA-ES.

At a query budget of 500, the quantum-assisted method discovered a mean of 0.00 unique verified failures per seed, compared with 25.80 for Random, 81.50 for Bayesian opt., 50.50 for CMA-ES.

At a query budget of 1000, the quantum-assisted method discovered a mean of 0.00 unique verified failures per seed, compared with 51.00 for Random, 86.20 for Bayesian opt., 148.10 for CMA-ES.

At a query budget of 2500, the quantum-assisted method discovered a mean of 0.00 unique verified failures per seed, compared with 131.40 for Random, 482.90 for Bayesian opt., 471.00 for CMA-ES.

At a query budget of 5000, the quantum-assisted method discovered a mean of 0.00 unique verified failures per seed, compared with 265.70 for Random, 2231.10 for Bayesian opt., 1243.50 for CMA-ES.

Budget 100, quantum_vs_random: quantum lower (mean difference -6.00, Holm-adjusted p = 0.0352, rank-biserial r = -1.00).

Budget 250, quantum_vs_random: quantum lower (mean difference -13.00, Holm-adjusted p = 0.0352, rank-biserial r = -1.00).

Budget 500, quantum_vs_random: quantum lower (mean difference -25.80, Holm-adjusted p = 0.0352, rank-biserial r = -1.00).

Budget 1000, quantum_vs_random: quantum lower (mean difference -51.00, Holm-adjusted p = 0.0352, rank-biserial r = -1.00).

Budget 2500, quantum_vs_random: quantum lower (mean difference -131.40, Holm-adjusted p = 0.0352, rank-biserial r = -1.00).

Budget 5000, quantum_vs_random: quantum lower (mean difference -265.70, Holm-adjusted p = 0.0352, rank-biserial r = -1.00).

Budget 5000, quantum_vs_bayesian: quantum lower (mean difference -2231.10, Holm-adjusted p = 0.0352, rank-biserial r = -1.00).

Of 18 primary comparisons, 0 were significantly in favour of the quantum-assisted method and 7 significantly against it (Holm-adjusted); the remaining 11 showed no statistically significant difference.

Outcome category: C: the quantum-assisted search performed worse than at least one classical baseline at one or more budgets.

These results describe search effectiveness per target-model query on a classical simulator. They do not demonstrate quantum speedup, quantum advantage or any complexity benefit.

## Experimental Limitations and Incomplete Runs

Three of the ten quantum-method runs at the query budget of 100 terminated with `candidate_cap_reached`. Each generated 5,000 candidates, but none passed candidate validation. Consequently, these runs used 0 of the 100 permitted target-model queries and discovered 0 verified failures. They are retained in the campaign records and are not query-budget overruns; however, they did not complete the intended query budget. Therefore, the zero-failure result for these runs reflects a candidate-generation and validity limitation, not evidence that 100 target-model evaluations were performed without finding a failure.

The quantum method was evaluated using a classical simulator. The results do not demonstrate quantum speedup or quantum advantage. The reference-consensus oracle is a proxy rather than ground truth, and findings should be interpreted as disagreements with that oracle under the stated validation and confidence criteria.
