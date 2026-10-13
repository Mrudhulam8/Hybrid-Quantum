"""Ten-seed validity-only comparison of quantum latent initializations."""
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
from src.quantum_search import QuantumSearch, QuantumSearchConfig
from src.search_base import CommonSearchConfig
from src.search_representation import PCARepresentation


def assess(candidates, space, validator):
    schema = [c.check(space.feature_names) for c in candidates]
    indices = [i for i, reason in enumerate(schema) if reason == OK]
    reasons = Counter(str(reason) for reason in schema if reason != OK)

    valid = 0
    if indices:
        X = np.stack([candidates[i].x for i in indices])
        results = validator.oracle.assess(X)
        oracle_reasons = list(results.reason)
        reasons.update(str(r) for r in oracle_reasons)
        valid = sum(r == OK for r in oracle_reasons)

    return {
        "candidates": len(candidates),
        "valid": int(valid),
        "validity_rate": float(valid / len(candidates)),
        "rejection_reasons": dict(reasons),
    }


def main():
    cfg_path = PROJECT / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    from src.phase8_setup import load_real_components
    components = load_real_components(cfg)
    prepared = components["prepared"]
    space, validator = components["space"], components["validator"]
    qcfg = QuantumSearchConfig.from_cfg(cfg)
    reference_X = np.asarray(validator.ref.X, dtype=float)
    reference_X = reference_X[np.all(np.isfinite(reference_X), axis=1)]

    n = 500
    seeds = list(range(20261012, 20261022))
    per_seed = []

    for seed in seeds:
        engine = QuantumSearch(
            space=space,
            validator=validator,
            query_budget=1,
            seed=seed,
            common=CommonSearchConfig.from_cfg(cfg),
            method_cfg=qcfg,
            paper_eligible_context=True,
        )

        uniform_latent = engine.initial_design(n)
        uniform_candidates = engine._candidates(
            uniform_latent,
            origin="uniform_latent_10seed_diagnostic",
            metas=[{"diagnostic_index": i} for i in range(n)],
        )
        uniform = assess(uniform_candidates, space, validator)

        rng = np.random.default_rng(seed)
        chosen = rng.integers(0, len(reference_X), size=n)
        representation = PCARepresentation(
            space=space,
            reference_X=reference_X,
            n_components=qcfg.latent_dimension,
            latent_bound=qcfg.latent_bound,
        )
        anchors = representation.encode(reference_X[chosen])

        sample_size = min(5000, len(reference_X))
        sample_idx = rng.choice(len(reference_X), size=sample_size, replace=False)
        sample_latent = representation.encode(reference_X[sample_idx])
        latent_std = np.maximum(np.std(sample_latent, axis=0), 1e-6)
        guided_latent = anchors + rng.normal(size=anchors.shape) * 0.10 * latent_std
        guided_latent = np.clip(
            guided_latent, -qcfg.latent_bound, qcfg.latent_bound
        )
        guided_candidates = engine._candidates(
            guided_latent,
            origin="reference_guided_10seed_diagnostic",
            metas=[{"diagnostic_index": i} for i in range(n)],
        )
        guided = assess(guided_candidates, space, validator)

        per_seed.append({
            "seed": seed,
            "uniform": uniform,
            "reference_guided": guided,
            "difference_percentage_points": 100 * (
                guided["validity_rate"] - uniform["validity_rate"]
            ),
        })
        print(
            f"seed={seed}: uniform={uniform['valid']}/{n} "
            f"({100*uniform['validity_rate']:.1f}%), "
            f"guided={guided['valid']}/{n} "
            f"({100*guided['validity_rate']:.1f}%)"
        )

    uniform_rates = np.array([r["uniform"]["validity_rate"] for r in per_seed])
    guided_rates = np.array([r["reference_guided"]["validity_rate"] for r in per_seed])
    differences = guided_rates - uniform_rates

    summary = {
        "diagnostic_only": True,
        "seeds": seeds,
        "candidates_per_method_per_seed": n,
        "total_candidate_assessments": len(seeds) * n * 2,
        "uniform_mean_validity_rate": float(uniform_rates.mean()),
        "uniform_std_validity_rate": float(uniform_rates.std(ddof=1)),
        "reference_guided_mean_validity_rate": float(guided_rates.mean()),
        "reference_guided_std_validity_rate": float(guided_rates.std(ddof=1)),
        "paired_difference_mean_percentage_points": float(differences.mean() * 100),
        "paired_difference_std_percentage_points": float(differences.std(ddof=1) * 100),
        "seeds_guided_better": int(np.sum(differences > 0)),
        "seeds_tied": int(np.sum(differences == 0)),
        "seeds_guided_worse": int(np.sum(differences < 0)),
        "per_seed": per_seed,
        "target_model_queries": 0,
        "thresholds_changed": False,
        "campaign_results_modified": False,
        "note": "Validity-only diagnostic; not evidence of improved failure discovery or quantum advantage.",
    }

    out = PROJECT / "results" / "diagnostics" / "quantum_reference_validity_10seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2))
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    main()
