# Search-space and candidate-generation policy (Phase 5)

Phase 5 builds **infrastructure only**. No search algorithm, no optimisation, no experimental
comparison and no research result exists at this stage.

## Binding rules
1. **Candidate generation is query-free.** `SearchSpace` and every `CandidateGenerator` consume
   **zero** target-model queries. They are never handed a model, a black box, or a prediction.
   (Tests booby-trap every model entry point, assert the generation modules do not even import
   `target_model`/`xgboost`/`validator`, and check a `QueryBudgetedModel` counter stays at 0.)
2. **Search algorithms cannot access labels.** Generators and the search space are built from the
   training split's feature matrix and feature names only (`SearchSpace.from_reference(X, names, cfg)`;
   `from_prepared` reads only `X_train` and `feature_names`). Labels, attack categories, the held-out
   test split and failure outcomes are unreachable (poisoned-object test + static scan).
3. **Search algorithms cannot directly access target-model internals** (trees, parameters, gradients,
   hidden state). The only model access in the whole project is `QueryBudgetedModel.query` (Phase 3),
   reached during discovery solely through the Phase 4 validator.
4. **Validation is handled exclusively by Phase 4.** `FailureValidator` decides validity, oracle
   evidence and `verified_failure`. Generator-side constraints are *shape constraints on proposals*,
   not validity verdicts; they never replace the validator's independent checks.
5. **Invalid candidates may be rejected before target-model evaluation.** The validator never queries
   candidates that fail its gates, so they cost no queries (see "Open design question" below).
6. **Generating a candidate is not evidence of a failure.** A `Candidate` carries only its feature
   vector, id, origin (generator name) and provenance metadata `{generator, seed, index[, anchor_index]}`.
   It never carries a model prediction, label or failure flag.
7. **No experimental comparison is performed in Phase 5.** Numbers produced by
   `scripts/probe_generator_validity.py` are descriptive diagnostics of generator output against the
   validity gates, not results.

## What exists
| Component | Purpose |
|---|---|
| `SearchSpace` | feature names/order, bounds, reference mean/std, optional radius and Mahalanobis constraints, `project()`, finite-value validation, deterministic JSON round trip with an integrity id |
| `UniformRandomGenerator` | uniform in the box (then projected if constraints are enabled) |
| `LocalPerturbationGenerator` | real reference row + bounded noise (`scale` x per-feature reference std; gaussian truncated, or uniform), then projected |
| `SearchSpaceConfig` | validated `configs/default.yaml -> search_space` block |

**Reproducibility.** Candidate `i` of stream `k` under seed `s` comes from its own
`default_rng(SeedSequence([s, k, i]))`, so it does not depend on batch size, on other candidates, or on
any global RNG state. `generate(10, seed=s)` equals `generate(4, seed=s)` followed by
`generate(6, seed=s, start_index=4)`. The seed is mandatory and explicit.

## Known properties and limitations (stated, not hidden)
* **Uniform sampling in the full box is almost surely invalid in high dimension.** On the *synthetic
  fallback* (77 features; a code check, NOT evidence about CIC-IDS2017) 0 of 800 box-uniform
  candidates passed the Phase 4 Mahalanobis gate. Local perturbation of real rows stayed mostly valid
  at small scales (~99 % at scale 0.05-0.1) and fell sharply by scale 0.5 (~30 %). Real-data numbers
  must be obtained with `scripts/probe_generator_validity.py` on the laptop.
* **Constraint projection creates a boundary shell.** With the Mahalanobis (or radius) constraint on,
  uniform proposals are far outside the constraint set, so projection moves essentially all of them
  onto its boundary (100 % on the boundary in the synthetic probe). Candidates are then valid but
  *not* uniform inside the constrained region. Any baseline described as "uniform" must state this.
* **Anchors may be projected.** `LocalPerturbationGenerator` projects the sampled anchor into the space
  first (relevant only for rows outside a quantile box or constraint). `anchor_index` always refers to
  the original reference row.
* **Perturbation limits hold exactly only when just the box and/or radius constraint is active.** With
  the Mahalanobis constraint the radial projection is non-expansive in the Mahalanobis metric, not in
  the per-feature metric.
* **Bounds are placeholders** (default: 0.1 %-99.9 % reference quantiles). They must be fixed from
  reference-only diagnostics before any method comparison, never from failure outcomes.
* **No active-feature subsets yet.** Every feature is searched. `search.active_dims` (Phase 1) is an
  unused placeholder; any feature-selection scheme must be label-free and is deferred.

## Open design question for Phase 6 - RESOLVED in Phase 6 (see `docs/search_algorithms.md`: shared proposal cap `search.common.max_candidates_per_query`)
_Original text kept for the record:_
Invalid candidates cost zero queries, so a method that mostly proposes invalid candidates receives no
target-model feedback and could generate indefinitely. Before building the baselines, fix **one**
policy applied identically to every method, e.g. a cap on proposed candidates per budgeted query, or a
shared choice of generator-side constraints. This choice is a methodological design decision and must
be made without looking at failure outcomes.
