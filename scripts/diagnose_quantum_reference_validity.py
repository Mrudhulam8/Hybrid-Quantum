"""Compare uniform and reference-guided quantum latent initialization."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.candidate import OK
from src.phase8_setup import load_real_components
from src.quantum_search import QuantumSearch, QuantumSearchConfig
from src.search_base import CommonSearchConfig


def assess(candidates, space, validator) -> dict:
    """Assess candidates with the reference oracle only; never query the target model."""
    schema_rejections = Counter()
    valid_rows = []
    for candidate in candidates:
        reason = candidate.check(space.feature_names)
        if reason == OK:
            valid_rows.append(candidate.x)
        else:
            schema_rejections[str(reason)] += 1

    oracle_reasons = Counter()
    valid_count = 0
    if valid_rows:
        result = validator.oracle.assess(np.asarray(valid_rows, dtype=np.float64))
        oracle_reasons.update(str(reason) for reason in result.reason)
        valid_count = int(np.asarray(result.valid, dtype=bool).sum())

    total = len(candidates)
    return {
        "candidates": total,
        "valid": valid_count,
        "invalid": total - valid_count,
        "validity_rate": valid_count / total if total else None,
        "schema_rejections": dict(schema_rejections),
        "oracle_reasons": dict(oracle_reasons),
    }


def main() -> None:
    with (PROJECT / "configs" / "default.yaml").open("r", encoding="utf-8-sig") as stream:
        cfg = yaml.safe_load(stream)

    components = load_real_components(cfg)
    cfg = components["cfg"]
    space, validator = components["space"], components["validator"]
    quantum_config = QuantumSearchConfig.from_cfg(cfg)
    seed, n_candidates = 20261012, 500

    engine = QuantumSearch(
        space=space,
        validator=validator,
        query_budget=1,
        seed=seed,
        common=CommonSearchConfig.from_cfg(cfg),
        method_cfg=quantum_config,
        paper_eligible_context=True,
    )

    uniform_candidates = engine._candidates(
        engine.initial_design(n_candidates),
        origin="uniform_initialization_diagnostic",
        metas=[{"diagnostic_index": i} for i in range(n_candidates)],
    )
    baseline = assess(uniform_candidates, space, validator)

    reference_X = np.asarray(validator.ref.X, dtype=np.float64)
    finite_reference = reference_X[np.isfinite(reference_X).all(axis=1)]
    if len(finite_reference) == 0:
        raise ValueError("validator reference contains no finite feature rows")

    rng = np.random.default_rng(seed)
    chosen_indices = rng.integers(0, len(finite_reference), size=n_candidates)
    anchors = engine.rep.encode(finite_reference[chosen_indices])
    sample_size = min(5000, len(finite_reference))
    sample_indices = rng.choice(len(finite_reference), size=sample_size, replace=False)
    latent_std = np.maximum(np.std(engine.rep.encode(finite_reference[sample_indices]), axis=0), 1e-6)

    guided_latent = anchors + rng.normal(size=anchors.shape) * (0.10 * latent_std)
    guided_latent = np.clip(
        guided_latent, -quantum_config.latent_bound, quantum_config.latent_bound
    )
    guided_candidates = engine._candidates(
        guided_latent,
        origin="reference_guided_initialization_diagnostic",
        metas=[{"diagnostic_index": i} for i in range(n_candidates)],
    )
    reference_guided = assess(guided_candidates, space, validator)

    summary = {
        "diagnostic_only": True,
        "seed": seed,
        "candidate_count_per_method": n_candidates,
        "baseline_uniform_latent": baseline,
        "experimental_reference_guided": reference_guided,
        "target_model_queries": 0,
        "thresholds_changed": False,
        "campaign_results_modified": False,
        "note": (
            "Validity diagnostic only. Reference-guided proposals use training reference rows; "
            "the proposal distributions differ by design and this does not establish quantum advantage."
        ),
    }

    output_path = PROJECT / "results" / "diagnostics" / "quantum_reference_validity_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
