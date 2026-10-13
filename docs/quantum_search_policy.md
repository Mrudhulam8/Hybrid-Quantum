# Hybrid quantum-assisted search policy (Phase 7)

> **The quantum component is evaluated as a search mechanism under a matched target-model query budget. No quantum advantage is assumed or claimed without empirical evidence.**

> **The ReferenceOracle is a proxy validation mechanism based on reference-data consensus and is not ground-truth labeling.**

Phase 7 implements and mechanically validates the method. It contains **no CIC-IDS2017 benchmark run, no
paper results, no performance comparison and no quantum-advantage claim.** Everything here describes
mechanics, accounting and limits.

## 1. Hybrid architecture

```
SearchSpace
   -> PCA latent representation  (Phase 6 PCARepresentation; TRAINING reference features only)
   -> classical kernel surrogate of the SEARCH SCORE over latent space (counted observations only)
   -> promising latent region  (box around an elite observed point)
   -> region encoded into a small variational circuit        (src/quantum_encoding.py)
   -> circuit parameters trained against surrogate cell scores (src/quantum_circuit.py)
   -> measurement -> bitstring -> latent cell -> latent point   (src/quantum_search.py)
   -> inverse PCA -> 77-D candidate -> SearchSpace.project
   -> duplicate removal
   -> FeedbackManager -> FailureValidator -> QueryBudgetedModel -> ReferenceOracle comparison
   -> search score (+ validator-recorded verdict, which the search never sees)
   -> surrogate / region / radius update -> next iteration
```

The quantum component **proposes** latent regions/points. It never decides what a failure is. Only
`FailureValidator` establishes `verified_failure`:

```
verified_failure = candidate_valid AND oracle_confidence >= configured threshold AND target_prediction != oracle_label
```

Uncertainty is not failure. `search_score != verified_failure`. Phase 4 thresholds and semantics are unchanged.

## 2. Framework and backend

| | |
|---|---|
| Intended framework | PennyLane (`default.qubit`), already listed in `requirements.txt` |
| Always-available backend | `numpy_statevector`: a built-in exact state-vector simulator (real amplitudes and gates, in `src/quantum_circuit.py`) |
| `backend: auto` (default) | PennyLane if importable **and** its self-check against the NumPy reference passes, otherwise `numpy_statevector`; the fallback reason is stored as a warning in every result |
| `backend: pennylane` | PennyLane only; raises `QuantumBackendUnavailable` if missing, `AssertionError` if its output disagrees with the reference |

Only one quantum framework is used (no Qiskit). The NumPy simulator exists because PennyLane was **not
installed and could not be installed** in the environment where Phase 7 was written (no network). It is a
genuine circuit simulator, not a classical algorithm presented as quantum. **The PennyLane adapter has not
been executed in that environment**; its tests skip with the exact reason, and its self-check refuses to run
if it ever disagrees with the reference. The backend actually used is recorded in every result
(`reproducibility.quantum.backend`, `.framework`, `.framework_version`).

## 3. Circuit design

Qubit 0 is the most-significant bit of the measured basis index (PennyLane's convention).

```
|0..0> -> H on all qubits
       -> RY(phi_q)  on all qubits                   (region encoding, section 5)
       -> depth x [ RY(theta) RZ(theta) on all qubits ; CNOT entangling layer ]
       -> RY(theta) on all qubits
       -> measure in the computational basis
```

* `n_params = depth * 2 * n_qubits + n_qubits`; entangler `ring` (q -> q+1 mod n; one CNOT for n = 2) or `linear`.
* Development defaults: **n_qubits = 4, depth = 2, shots = 256**, latent dimension 4 (config `search.quantum`).
  `n_qubits` is capped at 12; `latent_dimension <= n_qubits`.
* Gradients are exact **parameter-shift** gradients of the diagonal cost (verified against finite differences in the tests).
* **Shots** are emulated by multinomial sampling of the exact, noise-free probabilities with a seeded NumPy
  generator (both backends). There is no hardware noise model and **no statement about quantum hardware**.

## 4. Latent representation and reconstruction

CIC-IDS2017 has ~77 features; they are **never** loaded into 77 qubits. The search uses the Phase 6
`PCARepresentation` (`src/search_representation.py`), fitted by `PCARepresentation(space, validator.ref.X, ...)` on
the **training reference feature matrix only**: no test data, no labels, no hidden labels, no experimental
results. Axes are principal directions in units of their standard deviation, box `[-B, B]^k` (`B = 2` by
default, `k = latent_dimension = 4`). The recorded `explained_variance_ratio_total` shows how little of the
variance a 4-D latent space keeps; candidates are confined to that affine subspace.

Reconstruction: `latent -> mean + (v * sd) @ V^T -> SearchSpace.project` (clip to the box, then optional
radius / Mahalanobis constraints). The projected vector is what the validator sees, and the surrogate is fed
its re-encoded (repaired) latent coordinates.

## 5. Encoding and decoding (`src/quantum_encoding.py`)

* A **region** is the latent box `[c - r, c + r] ∩ [-B, B]^k` around an elite observed point `c`.
* The `n` qubits are assigned to the `k` latent axes round-robin (`qubit q -> axis q mod k`). A bitstring selects
  one of `2^n` cells that tile the region exactly (axis `j` is split into `2^(m_j)` slabs, `m_j` = its qubit count).
* **Angle encoding**: every qubit of axis `j` gets `RY(phi_j)`, `phi_j = (pi/2) * clip(c_j / B, -1, 1)`. Deterministic,
  bounded in `[-pi/2, pi/2]`, stable for outliers.
* **Decoding**: a measured cell becomes a candidate by drawing a point uniformly inside that cell with a seeded
  generator (stream per `(seed, iteration)`), so repeated measurements of one cell give distinct, reproducible points.
* The encoding layer only conditions the initial state on where the region sits. The proposal distribution is shaped by
  the **trained circuit parameters**, not by this layer; it is not claimed to carry independent information.

## 6. Quantum objective

The circuit minimises the diagonal cost `E(theta) = sum_b p_theta(b) * e_b`, where `e_b` is the **negative
normalised surrogate search score** of cell `b` (best cell = -1; all-equal scores = 0 = no preference).

The surrogate is a Gaussian-kernel regression of the **common Phase 6 search score** (uncertainty and
surrogate failure-likelihood mix, with the graded penalty for invalid candidates that encodes feasibility) over
the latent coordinates of candidates already evaluated in this run, plus a novelty/diversity bonus
`1 - kernel density`. It is fitted from `(latent, score)` pairs only. **The circuit receives 2^n numbers; it never
sees oracle labels, the oracle's output, target predictions, the test split, the validator's verdict or any
precomputed failure result.** Circuit parameters are tuned with Adam on parameter-shift gradients, the standard
variational-algorithm outer loop; that optimiser tunes `theta` only and never selects candidates. After each
iteration the parameters warm-start the next one.

## 7. Classical feedback loop

1. Counted initial design (uniform in the latent box; `max(5, ceil(0.1 * budget))` points, **never free**).
2. Evaluate through `FeedbackManager -> FailureValidator -> QueryBudgetedModel`; record search scores.
3. Fit the surrogate on the counted observations (top quarter + most recent when above 500 points).
4. Pick a region centre (rotating over the best 3 observed points) and a radius (grows x1.3 when the best observed
   score improved, shrinks x0.7 otherwise, within `[0.05, 1.0] * B`).
5. Encode, train, measure (`shots` shots), allocate `batch_size` proposals to cells by largest remainder of the counts.
6. Decode, project, drop duplicates (a fresh uniform latent point is used if every proposal was already seen).
7. Evaluate; feed new scores back; repeat until the budget or the shared proposal cap is exhausted.

## 8. Query-budget accounting and fairness

* Exactly the Phase 6 accounting: **one target-model prediction = one query**, only via `FailureValidator` ->
  `QueryBudgetedModel.query`, through the shared `FeedbackManager`. Invalid candidates are rejected before the model
  and cost zero queries. The budget is never reset or shared; a spent or mismatched black box is refused; a counter
  reset during a run raises `BudgetAccountingError`.
* **Circuit executions, shots and optimisation steps cost zero target-model queries** and are tallied separately
  (`reproducibility.quantum.cost`; wall-clock time in `reproducibility.quantum.runtime`, the only
  non-deterministic block). Every observation used by the surrogate or the cost came from the counted search process.
* Same target model, `SearchSpace`, `Candidate`, `FailureValidator`, `ReferenceOracle`, validity rules, duplicate
  policy (`skip`), proposal cap, stopping rules, seed protocol and result schema as Random / Bayesian / CMA-ES.
  The last batch is sized to the remaining budget.
* The quantum method uses a different initial-design RNG stream from Bayesian search but the same rule for its size.
  Its extra internal computation is **not** a target-model query and must not be read as an advantage.
* Initialisation defaults to `uniform`, preserving the baseline proposal distribution. The opt-in
  `search.quantum.initialization: reference_guided` samples training-reference feature rows, encodes them in the
  existing latent representation and adds deterministic, bounded local perturbations. It uses no labels or target
  queries; results from the two initialization modes are distinct experimental conditions and should not be pooled
  as though their proposal distributions were the same.
* Circuit seed = `SeedSequence([run_seed, search.quantum.seed])`; `search.quantum.seed` is a salt, the experiment seed
  protocol (`reproducibility.seeds`) is unchanged.

## 9. Reproducibility

Same seed => identical candidate ids, verified failures and result dict (excluding the `runtime` block). Per-iteration
randomness uses independent `SeedSequence([seed, stream, iteration])` streams (31 init, 32 shots, 33 in-cell sampling,
34 fallback, 35 circuit-parameter init). Results store the backend, framework and version, circuit/encoding
descriptions, the PCA fingerprint, config hashes and a trajectory hash. No hidden test labels are stored.

## 10. Policies

* **No-label-access policy.** Quantum modules never read `ReferenceData.y`, categories or any oracle/label output; they read
  only `ReferenceData.X` (via `PCARepresentation`) and `Feedback.score`. Enforced by a dynamic test (a reference view that
  raises on label access) and by static source checks.
* **No-test-data policy.** Representation, surrogate and circuit see the training reference features and counted
  search observations only. Enforced by a poisoned-`prepared` test that raises if any `*test*` attribute is touched.
* **No verdict leakage.** The search trajectory is invariant to a validator that flips every verdict (tested).
* **No bypass.** Static checks forbid `.query(` / `.predict_proba(` / model internals in the quantum modules.

## 11. Limitations

* Classical simulation of an ideal circuit; no hardware, no noise; simulator cost grows as `2^n` (n <= 12 enforced).
* **With only 2^n cells per iteration, directly enumerating the same surrogate scores classically is an obvious
  comparison and may match or beat the circuit as a proposer for small n (not tested here).** The prototype tests the hybrid
  mechanism and its accounting; any benefit over the classical baselines is an empirical question that Phase 7 does not answer.
* The latent space is a low-dimensional PCA subspace (default 4-D) of the training features; failures outside it are unreachable.
* The surrogate is a simple kernel smoother whose hyper-parameters are a-priori placeholders; the search score is a proxy.
* Failures are verified against a reference-consensus proxy, not ground truth.
* The PennyLane adapter is unexercised where this was developed (see section 2).
* Defaults (batch size, radius schedule, kernel length, learning rate) were fixed before any experiment and must not be
  tuned on failure outcomes.
