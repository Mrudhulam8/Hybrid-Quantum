"""Reference-only, query-free probe: what fraction of GENERATED candidates would pass the Phase 4
validity gates (Mahalanobis / NN-distance)?

  * No target model is built, loaded or queried.   * No held-out test field is read.
  * No failure is evaluated.                       * NOT an experiment and NOT a result.

It exists to inform design decisions (e.g. how invalid candidates are accounted for) BEFORE any
search method is compared. Usage (from the repository root):
    python scripts/probe_generator_validity.py --source cic-ids2017
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate_generator import make_generator                     # noqa: E402
from src.oracle import ReferenceData, ReferenceOracle                  # noqa: E402
from src.search_space import SearchSpace, SearchSpaceConfig            # noqa: E402
from src.utils import load_config, resolve_path, save_json             # noqa: E402
from src.validator import ValidatorConfig                              # noqa: E402

NOTE = ("Reference-only descriptive probe. No target model queried, no test data read, no failure "
        "evaluated. Not an experimental result.")


def probe(prepared, cfg: Dict[str, Any], n: int = 1500, seed: int = 1, cal_sample: int = 2000) -> Dict[str, Any]:
    """Validity-gate pass rates of generator output. Reads prepared.X_train/y_train/cat_train/names only."""
    ref = ReferenceData.from_prepared(prepared)                            # TRAIN split only
    oracle = ReferenceOracle(ref, dataclasses.replace(ValidatorConfig.from_cfg(cfg),
                                                      calibration_sample_size=min(cal_sample, ref.n)))
    X, names = prepared.X_train, prepared.feature_names
    base = SearchSpaceConfig.from_cfg(cfg)
    settings = [("uniform_random", "box only", {}),
                ("uniform_random", "box + radius<=1.0", dict(radius_enabled=True, radius_max=1.0)),
                ("uniform_random", "box + mahalanobis q=0.5", dict(maha_enabled=True, maha_quantile=0.5))]
    settings += [("local_perturbation", f"scale={s}", dict(local_scale=s)) for s in (0.05, 0.1, 0.25, 0.5)]
    rows = []
    for gen, label, over in settings:
        sc = dataclasses.replace(base, **over)
        space = SearchSpace.from_reference(X, names, sc)
        M = make_generator(gen, space, sc, anchors=X).generate_matrix(n, seed=seed)
        b = oracle.assess(M)
        row = {"generator": gen, "setting": label, "n": int(n), "valid_rate": float(b.valid.mean()),
               "reasons": {r: int(sum(1 for x in b.reason if x == r)) for r in sorted(set(b.reason))}}
        if space.has_mahalanobis:   # share of candidates sitting ON the constraint boundary (projected shell)
            row["share_on_generator_mahalanobis_boundary"] = float((space.mahalanobis(M) >= space.maha_tau * (1 - 1e-6)).mean())
        rows.append(row)
    return {"NOTE": NOTE, "dataset": prepared.provenance.get("source"),
            "paper_eligible": prepared.provenance.get("paper_eligible"), "d": int(X.shape[1]),
            "n_reference": int(X.shape[0]), "tau_mahalanobis_validator": oracle.tau_maha,
            "tau_nn_validator": oracle.tau_nn, "seed": int(seed), "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, help="auto|cic-ids2017|kdd99|synthetic-cic-schema")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, overrides={"data": {"source": a.source}} if a.source else None)
    from src.preprocessing import load_prepared, prepare_data
    d = cfg["data"]
    tag = f"{d['source']}_{d['task']}"
    cached = resolve_path(d["processed_dir"]) / f"{tag}.npz"
    prepared = load_prepared(d["processed_dir"], tag) if d["source"] != "auto" and cached.exists() else prepare_data(cfg)
    out = probe(prepared, cfg, n=a.n, seed=a.seed)
    path = save_json(out, Path(cfg["project"]["results_dir"]) / "raw" / f"generator_validity_probe_{out['dataset']}.json")
    print(f"dataset={out['dataset']} paper_eligible={out['paper_eligible']} d={out['d']} "
          f"tau_maha={out['tau_mahalanobis_validator']:.3f} tau_nn={out['tau_nn_validator']:.3f}")
    for r in out["rows"]:
        extra = f"  on-boundary={r['share_on_generator_mahalanobis_boundary']:.3f}" if "share_on_generator_mahalanobis_boundary" in r else ""
        print(f"  {r['generator']:<18s} {r['setting']:<26s} valid_rate={r['valid_rate']:.3f}  {r['reasons']}{extra}")
    print(f"saved: {path}\n{NOTE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
