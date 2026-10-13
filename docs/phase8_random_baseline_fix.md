# Phase 8 Random Baseline Proposal Fix

## Problem
The previous Phase 8 campaign used `uniform_random` over the full 77-dimensional reference-quantile box. The Phase 4 validator then rejected essentially all proposals using the reference Mahalanobis and nearest-neighbour validity gates. The observed campaign records reached the proposal cap with zero target-model queries.

## Fixed protocol
Phase 8 Random Search now uses the existing Phase 5 `local_perturbation` generator. Each proposal selects a training/reference feature row as an anchor and adds bounded feature-wise noise using the configured `scale=0.1`, Gaussian distribution, and truncation=3.0. The generator receives no labels, test data, target-model predictions, or failure outcomes.

`uniform_random` remains implemented and may be used for reference-only generator-validity diagnostics, but it is no longer the Phase 8 Random baseline.

## Important reproducibility rule
The old `results/phase8/raw/run_results.jsonl` belongs to the old configuration and must not be mixed with the fixed campaign. Preserve it as diagnostic evidence and run the fixed campaign in a fresh output directory, for example `results/phase8_fixed`.

Run the Phase 8 preflight against the fixed configuration before starting the campaign. The reference-only generator-validity probe should also be run and saved before the paper campaign.

## Scientific scope
This change does not alter query budgets, the validator, the verified-failure definition, target-model artifact, data split, seeds, duplicate policy, or query accounting. It changes only the predeclared proposal distribution so that Random Search can actually reach the shared target-model query budget rather than exhausting proposals on off-manifold candidates.
