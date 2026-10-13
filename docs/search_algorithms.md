# Classical search baselines (Phase 6)

Phase 6 implements and mechanically validates three classical baselines: **Uniform Random Search**,
**Bayesian / Gaussian-process search** and **CMA-ES**. It contains **no quantum search, no CIC-IDS2017
benchmark run, no paper results and no claim that any method is better than another.** The synthetic
smoke runs in `scripts/phase6_smoke.py` and the unit tests check mechanics only.

## Common interface (`src/search_base.py`)
Every method is a `SearchEngine(space, validator, query_budget, seed, common_cfg, method_cfg)` and is run
once on a **fresh** `QueryBudgetedModel` (`TargetModel.black_box(budget)`); an engine instance is
single-use. `run()` returns a `SearchResult` with: `method, seed, query_budget, queries_used,
candidates_generated, candidates_evaluated, candidates_valid/invalid, duplicates_skipped,
verified_failures, unique_verified_failures, verified_failure_events, first_failure_query, candidate_ids,
verified_failure_ids, objective_stats, termination_reason, reproducibility, paper_eligible, warnings,
black_box (final counter snapshot), trajectory`. `to_dict()/from_dict()/save()` are JSON round-trippable.
`src/experiment_runner.py` selects `random | bayesian | cmaes` and produces this one schema
(`run_search_method`, `run_grid`; the grid is infrastructure only and is not run in Phase 6).

## Shared evaluation gateway (`src/feedback_manager.py`)
All methods evaluate candidates through `FeedbackManager.evaluate`:
`candidates -> duplicate policy -> FailureValidator.validate_batch -> QueryBudgetedModel.query`.
Optimisers receive `Feedback(score, valid, queried, duplicate, query_number)`; they never receive the
validator's verdict, labels or the model.

### Query-budget accounting
* One target-model prediction = one query, via `QueryBudgetedModel.query` only (static test: no search
  module calls `.query(` / `.predict_proba(` or touches model internals).
* Candidates are validated in order in chunks **no larger than the remaining budget**. A chunk of m
  candidates holds at most m valid ones, so it can never exceed the budget; invalid candidates are
  rejected before the model and consume **zero** queries. A batch/population that does not fit is
  therefore processed as a deterministic prefix and the search stops cleanly.
* The budget is never reset or replaced. `run()` refuses a used or mismatched black box; after every
  chunk the counter is checked (monotone; increase == number of queried records), otherwise
  `BudgetAccountingError`.
* Candidate generation (Phase 5 generators, initial designs, CMA-ES sampling, GP acquisition) consumes
  zero queries.
* **Shared proposal cap** (`search.common.max_candidates_per_query`, default 50): invalid proposals are
  free, so a method that proposes only invalid candidates could run forever. At most
  `ceil(factor * budget)` candidates may be submitted per run, identically for every method; the run then
  ends with `termination_reason = "candidate_cap_reached"` and a warning. This resolves the Phase 5 open
  design question; the value 50 is an a-priori placeholder, fixed without looking at failure outcomes.

### Duplicate handling
`candidate_id` is the Phase 4 hash of schema + exact float64 vector. `duplicate_policy: skip` (default)
returns the cached feedback for an already-evaluated id (zero queries; counted in `duplicates_skipped`);
`allow` re-evaluates (one more query if valid). Either way an id is one verified failure.
`unique_verified_failures` = greedy leader clustering in discovery order at Euclidean resolution
`validator.dedup_resolution` (scaled units).

## Common search score (`src/search_score.py`)
`search_score != verified_failure`. The score ranks candidates; only `FailureValidator` establishes
`verified_failure`. The score code never reads `verified_failure`/`failure_reason` (tested, flipping the
flag leaves the score unchanged), and optimisers cannot see the verdict (static test).
* valid, queried: `(w_u*U + w_f*F + w_n*N) / sum(w)` in [0, 1]. `U` = normalised target entropy
  (uncertainty, informational), `F` = `1 - p_target(oracle_label)` when the oracle decided with confidence
  >= the validator threshold (surrogate failure likelihood), `N` = novelty w.r.t. earlier queried
  candidates of the same run (off by default). Uncertainty alone never makes a failure.
* invalid: `-(invalid_base + invalid_slope * min(excess, cap))`, `excess` = overshoot of the Mahalanobis /
  NN gates. This is query-free reference information and gives optimisers a gradient toward validity.
* Defaults `w_u = w_f = 0.5, w_n = 0` are placeholders, not tuned.
`F` uses the **reference-oracle** label (training split k-NN consensus), which is a proxy, not ground truth.

## Methods
### Uniform Random Search (`src/random_search.py`)
Phase 5 generator (`local_perturbation` is selected for the Phase 8 Random baseline; `uniform_random` remains available as a separate generator). The local generator samples a real reference-feature row as an anchor and adds bounded feature-wise noise; it uses no labels or target-model feedback. This avoids spending the proposal cap on points outside the high-density reference manifold, which can occur when uniform sampling is performed in the full 77-dimensional box.
explicit seed, no feedback used. Candidate *i* of a seed is independent of batch size, so the evaluated
sequence is too (tested). Not handicapped: same budget, validator, duplicate policy and cap as the others.

### Bayesian / GP search (`src/bayesian_search.py`)
Initial design of `max(min_initial, ceil(init_fraction*budget))` points (counted against the budget; no
free warm-up) -> scikit-learn `GaussianProcessRegressor` (ARD Matern-5/2 or RBF + white noise,
`normalize_y`) fitted **only** on (coordinates, score) pairs from this run's gateway evaluations ->
Expected Improvement (or UCB) maximised over 512 uniform + 256 local proposals -> evaluate -> update,
until the budget/cap is exhausted. Hyper-parameters are re-optimised every `refit_hyperparams_every`
iterations (posterior-only refit in between); at most `max_train_points` observations (top scorers + most
recent, deterministic). `batch_size > 1` selects the top-q with a minimum separation. A numerical GP
failure falls back deterministically to random proposals and is reported as a warning.

**High-dimensional limitation.** A GP on ~77 raw dimensions is unreliable (distance concentration,
length-scale fitting, O(n^3)). Default representation = `pca`: top-k principal axes of the TRAINING
reference **features** (`src/search_representation.py`; deterministic eigendecomposition, fixed sign
convention, box `[-B, B]^k` in std units, `decode = SearchSpace.project(mean + (v*sd) @ V^T)`), k = 10.
It uses no labels, no test data, no model. Phase 4 measured an effective rank of ~9 of 77 on CIC-IDS2017,
which motivates k = 10 but does not prove the subspace contains failures: failure regions orthogonal to
the leading axes are unreachable for this method (the other methods do not share that restriction when
run in the full space). `representation: full` exists and warns for d > 20.

### CMA-ES (`src/cmaes_search.py`)
Self-contained, deterministic (mu/mu_w, lambda)-CMA-ES (Hansen 2016 tutorial); the `cma` package is not
used. Default coordinates = standardised features (`(x - ref_mean)/ref_std`), `sigma0 = 0.1`, population
`4 + floor(3 ln n)` (configurable), `max_generations` configurable. Sample draws come from
`SeedSequence([seed, stream, restart, generation])`. Bounds: samples are repaired by
`SearchSpace.project` and the strategy is updated with the repaired points. It minimises `-search_score`.
Start point: a seeded reference **feature** row (`x0: reference_anchor`), the reference mean, or uniform.
Restarts on step-size collapse / ill-conditioning / flat fitness. If the budget ends inside a population
the evaluated prefix is kept, the rest is dropped (reported), no update is made and the run ends with
`budget_exhausted`.

## Reproducibility
Explicit integer seed everywhere; every random draw comes from its own
`default_rng(SeedSequence([seed, stream, ...]))` (streams: Phase 5 uniform = 1, local = 2; Bayesian init /
proposals / GP = 11 / 12 / 13; CMA-ES sampling / start = 21 / 22). No global RNG. Results carry
`space_id`, reference fingerprint, validator/common config hashes, method config, objective, library
versions and `trajectory_sha256` (hash of the evaluated id sequence). Same inputs -> identical trajectories
on the same machine (BLAS/LAPACK differences across machines can in principle perturb GP/CMA-ES floating
point; not tested across machines).

## Data policy
**No labels:** search code receives a `SearchSpace` (from `X_train` + feature names), a `FailureValidator`
and a query-budgeted black box. It never reads `y`, categories or oracle internals; only
`ReferenceData.X` (features) is used for PCA axes and CMA-ES/local anchors. **No test data:** nothing in
Phase 6 reads a `*_test` field (poisoned-object test + static scan). The oracle's labels come from the
TRAINING split only, and only inside `FailureValidator`.

## Known limitations
* Uniform sampling in the full box is almost surely invalid in high dimension (Phase 5 note). With the
  shared proposal cap, random search may terminate by cap with unused budget on real data; the policy
  (cap, generator constraints) should be reviewed against reference-only validity diagnostics
  (`scripts/probe_generator_validity.py`) **before** the benchmark, never against failure outcomes.
* The Bayesian method searches a 10-D subspace by default; CMA-ES searches the full space. This is a
  design difference that any later comparison must state.
* All placeholders (score weights, `sigma0`, `latent_bound`, cap factor, population sizes) are unvalidated.
* `paper_eligible` is true only if the caller passes `paper_eligible_context=True` (the runner derives it
  from "not a dev-mock target") and accounting is consistent; synthetic-data results are never eligible.
* The `cma` and `xgboost` packages were not installed in the development sandbox: XGBoost-dependent
  Phase 3 tests were skipped there; Phase 6 tests use a counting backend.
