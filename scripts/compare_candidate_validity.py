"""Compare candidate validity for quantum initialization and local perturbation."""
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
from src.candidate_generator import LocalPerturbationGenerator, LocalPerturbationParams
from src.phase8_setup import load_real_components
from src.quantum_search import QuantumSearch, QuantumSearchConfig
from src.search_base import CommonSearchConfig
from src.search_space import SearchSpaceConfig


def assess(label: str, candidates, space, validator) -> dict:
    schema_rejections = Counter()
    well_formed = []
    for candidate in candidates:
        reason = candidate.check(space.feature_names)
        if reason == OK:
            well_formed.append(candidate.x)
        else:
            schema_rejections[str(reason)] += 1

    oracle_counts = Counter()
    valid_count = 0
    if well_formed:
        result = validator.oracle.assess(np.asarray(well_formed, dtype=np.float64))
        valid_count = int(np.asarray(result.valid, dtype=bool).sum())
        oracle_counts.update(str(reason) for reason in result.reason)

    total = len(candidates)
    return {
        "method": label,
        "generated": total,
        "schema_rejections": dict(schema_rejections),
        "oracle_reasons": dict(oracle_counts),
        "valid": valid_count,
        "validity_rate": valid_count / total if total else None,
        "target_model_queries": 0,
    }


def main() -> None:
    outdir = PROJECT / "results" / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    with (PROJECT / "configs" / "default.yaml").open("r", encoding="utf-8-sig") as stream:
        cfg = yaml.safe_load(stream)

    components = load_real_components(cfg)
    cfg = components["cfg"]
    space, validator = components["space"], components["validator"]

    n_candidates = 500
    seed = 20261012
    quantum_config = QuantumSearchConfig.from_cfg(cfg)
    quantum_engine = QuantumSearch(
        space=space,
        validator=validator,
        query_budget=1,
        seed=seed,
        common=CommonSearchConfig.from_cfg(cfg),
        method_cfg=quantum_config,
        paper_eligible_context=True,
    )
    quantum_candidates = quantum_engine._candidates(
        quantum_engine.initial_design(n_candidates),
        "quantum_validity_comparison",
        [{"index": i} for i in range(n_candidates)],
    )

    space_config = SearchSpaceConfig.from_cfg(cfg)
    generator = LocalPerturbationGenerator(
        space,
        np.asarray(validator.ref.X, dtype=np.float64),
        LocalPerturbationParams.from_config(space_config),
    )
    classical_candidates = generator.generate(n_candidates, seed=seed)

    summary = {
        "diagnostic_only": True,
        "seed": seed,
        "candidates_per_method": n_candidates,
        "quantum": assess("quantum_initial_design", quantum_candidates, space, validator),
        "classical": assess("classical_local_perturbation", classical_candidates, space, validator),
        "thresholds_changed": False,
        "original_campaign_modified": False,
        "interpretation_note": (
            "Diagnostic comparison only. Proposal distributions differ by design; "
            "this measures validity, not overall search performance or quantum advantage."
        ),
    }

    output_path = outdir / "quantum_vs_classical_validity.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
