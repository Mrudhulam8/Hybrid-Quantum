# Data-split policy (Phase 4)

| Role | Data | Used by | Never used by |
|---|---|---|---|
| **Target-model training** | `X_train` / `y_train` minus a seeded 10 % slice | XGBoost trees (Phase 3) | oracle statistics (see note) |
| **Early-stopping validation** | seeded 10 % slice carved from the *training* split | XGBoost early stopping (Phase 3) | search, oracle |
| **Reference / oracle fitting** | the **entire training split** (`X_train`, `y_train`, `cat_train`) | Mahalanobis statistics, feature ranges, k-NN labels, threshold calibration (`src/oracle.py`) | the target model, any test field |
| **Held-out final test** | `X_test` / `y_test` / `cat_test` (59,998 rows on the real run) | final evaluation only (Phase 3 metrics, later reporting) | **discovery, validator, oracle, search objective** |

## Why the oracle reuses the training split
The Phase 3 split gives no third, disjoint partition, and the target model is already trained on
the training split. A disjoint reference would require retraining the target model, which is
out of scope (Phase 3 behaviour must not change). The held-out test split stays untouched.

## Consequences (stated as limitations, not hidden)
1. **Shared source of truth.** The oracle and the target model are both derived from the same
   training labels. Wherever the training labels are noisy, both inherit the noise, so a "verified
   failure" is *disagreement with the local reference consensus*, **not** ground truth.
2. **Likely under-counting near training points.** Close to training points the model usually
   agrees with its own training labels; disagreements will concentrate in sparsely covered
   regions - exactly where the oracle is least certain. The purity/abstention thresholds
   control that trade-off.
3. **Oracle is not the target model's judge of reality.** It is an independent, reference-based
   proxy. Any paper claim must be phrased as "disagreement with a reference-consensus oracle".

## Enforcement in code
* `ReferenceData.from_prepared` reads only `X_train, y_train, cat_train, feature_names,
  class_names, provenance`.
* `tests/test_phase4.py::test_no_test_label_dependency_runtime_and_static` passes the validator a
  `PoisonedPrepared` object whose `*_test` attributes raise, and statically scans `oracle.py`,
  `validator.py`, `candidate.py`, `uncertainty.py` for `X_test/y_test/cat_test` and for any
  target-model internals. (Mutation-checked: injecting `prepared.y_test` makes it fail.)
* Thresholds are calibrated from reference statistics only and must be **frozen before any search
  method is compared**; tuning them on discovery outcomes would bias the study.
