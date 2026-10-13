# Phase 8 protocol

Run order: `python main.py prepare-data --source cic-ids2017` -> `python main.py train-model --source cic-ids2017 --check-reproducibility`
-> `python scripts/phase8_preflight.py` -> `python scripts/phase8_run.py smoke` -> `python scripts/phase8_run.py campaign`
(resumable; repeat until 240/240) -> `python scripts/phase8_run.py analyze`.

* Budgets are INDEPENDENT runs (fresh query-budgeted black box per method x budget x seed); never cumulative, never reset.
* Primary metric: `unique_verified_failures` (Phase 6 leader clustering at validator.dedup_resolution). Sensitivity: `verified_failures` (distinct ids).
* Stats: paired by seed, Wilcoxon signed-rank (two-sided), Holm over 3 comparisons x 6 budgets per metric, rank-biserial effect size, bootstrap CIs.
* `analyze` refuses incomplete or non-paper-eligible results. Failure-category diversity is not available from the Phase 6/7 tracker.
