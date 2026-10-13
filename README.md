# Hybrid Classical-Quantum Framework for Rare Failure Discovery

Research prototype that compares **Random Search, GP-BO, CMA-ES and a
Quantum-Assisted Search** at discovering rare, validated failure cases of a
trained black-box classifier (XGBoost on CIC-IDS2017) under a fixed
target-model query budget.

> **Scope statement.** This project measures *empirical failure-discovery
> efficiency under a query budget* using a classical simulator. It does **not**
> claim asymptotic quantum speedup or practical quantum advantage. Negative
> results will be reported as-is.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py check-env
pytest -q
```
Google Colab: `!pip install -r requirements.txt` then run the same commands.

## Status
| Phase | Description | Status |
|---|---|---|
| 1 | Structure, config, env, reproducibility utilities | done |
| 2 | Dataset loading (CIC-IDS2017 + labelled fallbacks), cleaning, split, invertible preprocessing | done |
| 3 | XGBoost target model, QueryBudgetedModel (strict query counting), metrics, saving | done (XGBoost run must be executed locally/Colab) |
| 4 | Failure validation infrastructure: Candidate, reference oracle (Mahalanobis + k-NN), FailureValidator, result schema | done (infrastructure only; no experiments) |
| 5 | Search space and query-free candidate generation (uniform + local perturbation) | done (infrastructure only; no experiments) |
| 6 | Classical search baselines: Uniform Random, Bayesian/GP, CMA-ES on one shared evaluation gateway | done (mechanics validated on synthetic fixtures only; no benchmark run, no results) |
| 7 | Hybrid quantum-assisted search (simulator) | done |
| 8 | Full evaluation infrastructure (`src/phase8*.py`, `scripts/phase8_*.py`): matrix, resumable runner, stats, figures, paper_results | infrastructure + tests done; **campaign NOT run** (needs real CIC-IDS2017 + trained XGBoost model; see docs/phase8_protocol.md) |
| 9-17 | see roadmap | pending |

## Layout
See `src/` (one module per framework component), `experiments/`, `configs/`,
`results/{figures,tables,raw}`. Stub modules state the phase that implements them.

## Data
Place the CIC-IDS2017 *MachineLearningCSV* files in `data/raw/cic-ids2017/`, then:
```bash
python main.py prepare-data                      # source=auto: CIC-IDS2017 -> KDD99 -> synthetic
python main.py prepare-data --source cic-ids2017 --max-rows 200000
```
`auto` falls back loudly (banner + `provenance.paper_eligible` flag). **The synthetic fallback is for
pipeline tests only and must never be used for reported results.** The data split uses a fixed
`data.split_seed`, so variance across the 10 experiment seeds reflects search variance only.

Tests: `pytest -q` (or `python tests/_mini_runner.py tests.test_phase1 tests.test_phase2`).

## Target model (Phase 3)
```bash
python main.py train-model                          # needs xgboost; uses data.source from the config
python main.py train-model --source cic-ids2017 --check-reproducibility
```
Search engines never see the model: they receive a `QueryBudgetedModel` from
`TargetModel.black_box(budget)`. One query = one input row; budgets are enforced *before*
evaluation (no partial batches); exceeding raises `QueryBudgetExceeded`. XGBoost is the only research
model. A sklearn `dev-mock` exists for unit tests only (needs `HQFD_ALLOW_DEV_MOCK=1`, outputs are
prefixed `DEV_MOCK_` and flagged `paper_eligible=false`); the CLI never falls back to it.

## Failure validation (Phase 4) - infrastructure only
`FailureValidator` is the single gateway through which discovery code evaluates candidates.
**Uncertainty is not failure.** Each `ValidationResult` keeps four facts apart: model uncertainty
(entropy/margin; informational), candidate validity (Mahalanobis + NN-distance gates), oracle evidence
(k-NN label/purity from the *training split only*), and `verified_failure`
= valid AND oracle_confidence >= min AND target_prediction != oracle_label.
Valid candidates cost exactly one query via `QueryBudgetedModel.query`; invalid ones are never sent
to the model. Thresholds live in `configs/default.yaml -> validator`. Data roles are in
`docs/data_split_policy.md`. Reference-only diagnostics (no model, no test data, not a result):
```bash
python main.py validator-diagnostics --source cic-ids2017
```

## Search space and candidate generation (Phase 5) - infrastructure only
**Implemented:** `src/search_space.py` (feature order, bounds, reference statistics, optional radius /
Mahalanobis constraints, `project()`, deterministic JSON round trip), `src/candidate_generator.py`
(`UniformRandomGenerator`, `LocalPerturbationGenerator`, both producing Phase 4 `Candidate` objects with
deterministic ids and `{generator, seed, index}` metadata), a validated `search_space` config block,
`tests/test_phase5.py` (synthetic fixtures only; CIC-IDS2017 is not a test dependency) and
`docs/search_space_policy.md`. Generation is **query-free and label-free**; only the Phase 4
validator ever queries the target model.

**Not implemented (later phases):** any search or optimisation algorithm (Random Search baseline logic,
GP-BO, CMA-ES, quantum), the feedback loop, unique-failure tracking, experiments, figures.

**No experiments were run and no results were produced in Phase 5.** The real-data checks happen on
your laptop later. A reference-only, query-free probe of how many generated candidates pass the Phase 4
validity gates (not a result) is provided:
```bash
python scripts/probe_generator_validity.py --source cic-ids2017
```

## Classical search baselines (Phase 6) - infrastructure only
**Implemented:** `src/search_base.py` (common `SearchEngine` + `SearchResult`), `src/feedback_manager.py`
(shared evaluation gateway: duplicate policy, chunked query accounting, proposal cap),
`src/search_score.py` (common search score), `src/failure_tracker.py`, `src/search_representation.py`
(label-free PCA / standardised coordinates), `src/random_search.py`, `src/bayesian_search.py`
(scikit-learn GP + EI/UCB), `src/cmaes_search.py` (self-contained CMA-ES), `src/experiment_runner.py`,
`tests/test_phase6.py`, `docs/search_algorithms.md`, config `search.common|random|bayesian|cmaes`.
All three methods share the SearchSpace, FailureValidator, query-budget wrapper, duplicate policy and seed
policy; budgets are arbitrary positive integers and are never reset. **The search score ranks candidates;
only the Phase 4 validator establishes a verified failure.** No quantum search, no CIC-IDS2017 benchmark, no
results and no performance claims exist at this stage. Mechanics-only smoke test (tiny synthetic fixture):
```bash
python scripts/phase6_smoke.py --budget 40 --seed 1
python tests/_mini_runner.py tests.test_phase6      # or: pytest -q tests/test_phase6.py
```

## Hybrid quantum-assisted search (Phase 7) - mechanics only
**Implemented:** `src/quantum_search.py` (`QuantumSearch`, same `SearchEngine` interface/result schema as the
classical methods), `src/quantum_circuit.py` (small variational circuit, backend abstraction, parameter-shift
gradients), `src/quantum_encoding.py` (latent region <-> qubits), config `search.quantum`, `quantum` registered in
`src/experiment_runner.py`, `tests/test_phase7.py`, `scripts/phase7_smoke.py`, `docs/quantum_search_policy.md`.

*Architecture.* The Phase 6 PCA representation of the **training reference features** reduces ~77 features to a
small latent space (default 4-D; never 77 qubits). A classical kernel surrogate of the common **search score** picks a
promising latent region; the region is angle-encoded into a 4-qubit, depth-2 variational circuit whose parameters are
trained so its measurement distribution favours cells with high surrogate score; measured cells are decoded to latent
points, mapped back to 77 features, projected into the `SearchSpace`, de-duplicated and evaluated through the shared
`FeedbackManager -> FailureValidator -> QueryBudgetedModel`. New scores feed the next iteration. **The quantum
component only proposes; only the Phase 4 validator establishes a verified failure.**

*Fairness.* One target-model prediction = one query, identical to the other three methods; the initial design is
counted; circuit executions cost zero target queries and are reported separately in `reproducibility.quantum.cost`.
No labels, test data or verdicts reach the circuit.

*Simulator limits.* Ideal, noise-free classical simulation; no hardware claims; no quantum advantage assumed. PennyLane
is used if installed and self-checked, otherwise a built-in NumPy state-vector simulator (the backend used is recorded in
every result). No results or performance numbers exist at this stage.

```bash
python scripts/phase7_smoke.py --budget 40 --seed 1      # tiny synthetic fixture, budget 20-50 only
python tests/_mini_runner.py tests.test_phase7           # or: pytest -q tests/test_phase7.py
```
