"""Entry point.  Usage:

    python main.py check-env           # verify installed packages & directories
    python main.py show-config         # print resolved configuration
    python main.py prepare-data        # load -> clean -> split -> scale; saves to data/processed/
    python main.py prepare-data --source synthetic-cic-schema --max-rows 50000
    python main.py train-model         # XGBoost target model: train, evaluate, save (needs xgboost)
    python main.py train-model --source cic-ids2017 --check-reproducibility
    python main.py validator-diagnostics --source cic-ids2017   # reference-only oracle/threshold diagnostics
    python main.py check-env --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils import (environment_report, ensure_dirs, get_logger, load_config,
                       config_hash, set_global_seed)

REQUIRED = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "pyyaml", "matplotlib"]
NEEDED_LATER = {"cma": "Phase 8 (CMA-ES)", "pennylane": "Phase 7 (quantum search; optional, NumPy simulator fallback)",
                "torch": "Phase 5 optional autoencoder", "seaborn": "Phase 15 plots"}


def cmd_check_env(cfg) -> int:
    log = get_logger("check-env")
    ensure_dirs(cfg)
    env = environment_report()
    log.info("Python %s on %s", env["python"], env["platform"])
    missing_required = []
    for pkg, ver in env["packages"].items():
        if ver is None:
            tag = "REQUIRED" if pkg in REQUIRED else NEEDED_LATER.get(pkg, "optional")
            log.info("  [missing] %-20s (%s)", pkg, tag)
            if pkg in REQUIRED:
                missing_required.append(pkg)
        else:
            log.info("  [ok]      %-20s %s", pkg, ver)
    rng = set_global_seed(cfg["reproducibility"]["global_seed"])
    log.info("Seed check (first draw, should be identical every run): %.6f", rng.random())
    log.info("Config hash: %s", config_hash(cfg))
    if missing_required:
        log.error("Missing required packages: %s  ->  pip install -r requirements.txt",
                  ", ".join(missing_required))
        return 1
    log.info("Phase 1 environment OK.")
    return 0


def cmd_prepare_data(cfg) -> int:
    from src.preprocessing import prepare_data, save_prepared
    from src.utils import resolve_path, save_json
    ensure_dirs(cfg)
    p = prepare_data(cfg)
    tag = f"{p.provenance['source']}_{cfg['data']['task']}"
    path = save_prepared(p, cfg["data"]["processed_dir"], tag)
    tbl = p.summary_table()
    out = resolve_path(cfg["project"]["results_dir"]) / "tables" / f"dataset_summary_{tag}.csv"
    tbl.to_csv(out, index=False)
    print(tbl.to_string(index=False))
    print(f"\nsource={p.provenance['source']}  paper_eligible={p.provenance['paper_eligible']}  "
          f"features={len(p.feature_names)}  train={len(p.X_train)}  test={len(p.X_test)}")
    print(f"saved: {path}\n       {out}")
    return 0


def cmd_train_model(cfg, check_repro: bool) -> int:
    from src.preprocessing import prepare_data
    from src.target_model import train_and_evaluate
    ensure_dirs(cfg)
    kind = cfg["target_model"]["type"]
    try:
        prepared = prepare_data(cfg)
        res = train_and_evaluate(cfg, prepared, kind=kind, check_reproducibility=check_repro)
    except ImportError as e:
        print(f"\nERROR: {e}")
        return 2
    m = res["metrics"]
    print("\n" + "=" * 70)
    print(f"model={res['kind']}  dataset={res['data']['provenance']['source']}  "
          f"paper_eligible={res['paper_eligible']}")
    print(f"accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
          f"f1={m['f1']:.4f}  roc_auc={m['roc_auc']}")
    print(f"confusion_matrix (rows=true, cols=pred): {m['confusion_matrix']}")
    print(f"errors={m['n_errors']} / {m['n_samples']}   high-confidence errors: {m['high_confidence_errors']}")
    print(f"best_iteration={res['model']['train_info'].get('best_iteration')}  "
          f"test_proba_sha256={res['test_proba_sha256']}  roundtrip_ok={res['saved_model_roundtrip_verified']}")
    if res["reproducibility_check"]:
        print(f"reproducibility: {res['reproducibility_check']}")
    if not res["paper_eligible"]:
        print("NOTE: this run is NOT paper-eligible (fallback/synthetic dataset or mock model).")
    print("files:"); [print("   ", v) for v in res["files"].values()]
    print(f"    results/raw/target_model_{res['tag']}.json  (+ tables/*.csv)")
    return 0


def cmd_validator_diagnostics(cfg) -> int:
    """Descriptive statistics of the TRAINING-split reference + validator thresholds.
    Queries no target model, reads no test labels, runs no search: NOT an experiment result."""
    from src.oracle import ReferenceData, ReferenceOracle
    from src.preprocessing import load_prepared, prepare_data
    from src.utils import resolve_path, save_json
    from src.validator import ValidatorConfig
    ensure_dirs(cfg)
    d = cfg["data"]
    tag = f"{d['source']}_{d['task']}"
    cache = resolve_path(d["processed_dir"]) / f"{tag}.npz"
    prepared = load_prepared(d["processed_dir"], tag) if d["source"] != "auto" and cache.exists() \
        else prepare_data(cfg)
    vcfg = ValidatorConfig.from_cfg(cfg)
    ref = ReferenceData.from_prepared(prepared, vcfg.reference_max_rows, vcfg.seed)   # TRAIN split only
    oracle = ReferenceOracle(ref, vcfg)
    out = {"NOTE": "Reference-only descriptive diagnostics. No target model queried, no test labels used, "
                   "no search run. Not an experimental result.",
           "dataset": prepared.provenance.get("source"), "paper_eligible": prepared.provenance.get("paper_eligible"),
           "oracle": oracle.diagnostics, "oracle_self_consistency_leave_one_out": oracle.self_consistency(),
           "perturbation_validity_profile": oracle.perturbation_validity_profile(),
           "validator_config": cfg["validator"]}
    path = save_json(out, Path(cfg["project"]["results_dir"]) / "raw" / f"validator_diagnostics_{prepared.provenance.get('source')}.json")
    o = out["oracle"]
    print(f"reference: n={o['reference_n']} d={o['reference_d']} fingerprint={o['reference_fingerprint']}")
    print(f"mahalanobis: tau={o['tau_mahalanobis']:.3f}  cond={o['mahalanobis']['condition_number']:.3g}  "
          f"eff_rank={o['mahalanobis']['effective_rank']:.1f}  floored_eigs={o['mahalanobis']['n_eigenvalues_floored']}")
    print(f"nn-gate tau={o['tau_nn']:.3f}")
    sc = out["oracle_self_consistency_leave_one_out"]
    print(f"oracle LOO self-consistency: abstain={sc['abstain_rate']:.4f}  agreement_when_decided={sc['agreement_when_decided']}")
    print("perturbation validity profile (sigma -> valid_rate):",
          {k: round(v['valid_rate'], 3) for k, v in out['perturbation_validity_profile'].items()})
    print(f"saved: {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["check-env", "show-config", "prepare-data", "train-model", "validator-diagnostics"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, help="auto|cic-ids2017|kdd99|synthetic-cic-schema")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--check-reproducibility", action="store_true",
                    help="train twice and compare test-probability hashes")
    ap.add_argument("--task", default=None, choices=["binary", "multiclass"])
    args = ap.parse_args()
    ov = {k: v for k, v in {"source": args.source, "max_rows": args.max_rows, "task": args.task}.items()
          if v is not None}
    cfg = load_config(args.config, overrides={"data": ov} if ov else None)
    if args.command == "show-config":
        print(json.dumps(cfg, indent=2))
        return 0
    if args.command == "validator-diagnostics":
        return cmd_validator_diagnostics(cfg)
    if args.command == "train-model":
        return cmd_train_model(cfg, args.check_reproducibility)
    if args.command == "prepare-data":
        return cmd_prepare_data(cfg)
    return cmd_check_env(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
