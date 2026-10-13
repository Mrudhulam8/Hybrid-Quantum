"""Diagnose validity of a reproducible quantum initial design without target queries."""
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


def main() -> None:
    project = PROJECT
    outdir = project / "results" / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    with (project / "configs" / "default.yaml").open("r", encoding="utf-8-sig") as stream:
        cfg = yaml.safe_load(stream)

    from src.phase8_setup import load_real_components

    components = load_real_components(cfg)
    space, validator = components["space"], components["validator"]
    qcfg = QuantumSearchConfig.from_cfg(cfg)
    seed = 20261012
    n_candidates = 500

    engine = QuantumSearch(
        space=space,
        validator=validator,
        query_budget=1,
        seed=seed,
        common=CommonSearchConfig.from_cfg(cfg),
        method_cfg=qcfg,
        paper_eligible_context=True,
    )
    latent = engine.initial_design(n_candidates)
    candidates = engine._candidates(
        latent,
        origin="quantum_validity_diagnostic",
        metas=[{"diagnostic_index": i} for i in range(n_candidates)],
    )

    schema_reasons = [candidate.check(space.feature_names) for candidate in candidates]
    well_formed_indices = [i for i, reason in enumerate(schema_reasons) if reason == OK]
    oracle_reasons = validator.oracle.assess(
        np.stack([candidates[i].x for i in well_formed_indices])
    ).reason if well_formed_indices else []

    reasons = Counter(str(reason) for reason in schema_reasons if reason != OK)
    reasons.update(str(reason) for reason in oracle_reasons)
    valid_count = sum(reason == OK for reason in oracle_reasons)

    summary = {
        "diagnostic_only": True,
        "method": "quantum_initial_design",
        "seed": seed,
        "candidates_generated": len(candidates),
        "valid_candidates": valid_count,
        "invalid_candidates": len(candidates) - valid_count,
        "validity_rate": valid_count / len(candidates) if candidates else None,
        "rejection_reasons": dict(reasons),
        "target_model_queries": 0,
        "thresholds_changed": False,
        "campaign_results_modified": False,
    }

    outpath = outdir / "quantum_candidate_validity.json"
    outpath.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved diagnostic to: {outpath}")


if __name__ == "__main__":
    main()
