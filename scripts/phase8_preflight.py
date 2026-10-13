"""Phase 8 preflight. Fails loudly (exit 1) unless every requirement for the 240-run campaign is met.

    python scripts/phase8_preflight.py [--config configs/default.yaml] [--model-tag xgboost_cic-ids2017_binary]
                                       [--budget 50] [--seed 1] [--out results/phase8]

Checks that need real data are marked BLOCKED (and count as failures) when an upstream check failed.
Runs NO campaign. Writes results/phase8/logs/preflight.json.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_runner import METHODS                                           # noqa: E402
from src.phase8 import (METHOD_ORDER, build_matrix, ensure_layout, software_versions)   # noqa: E402
from src.phase8_setup import (PreflightFailure, accounting_audit, load_real_components,   # noqa: E402
                              reproducibility_audit, static_leakage_audit, verify_model_artifact)
from src.search_base import CommonSearchConfig                                       # noqa: E402
from src.utils import load_config, resolve_path                                      # noqa: E402

EXPECTED_BUDGETS = [100, 250, 500, 1000, 2500, 5000]
EXPECTED_SEEDS = list(range(1, 11))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-tag", default="xgboost_cic-ids2017_binary")
    ap.add_argument("--budget", type=int, default=50, help="small budget for accounting/reproducibility audits")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/phase8")
    a = ap.parse_args()
    cfg = load_config(a.config)
    results, ctx = [], {}

    def check(name, fn, needs=()):
        blocked = [n for n in needs if not next((r["ok"] for r in results if r["check"] == n), False)]
        if blocked:
            results.append({"check": name, "ok": False, "status": "BLOCKED", "detail": f"depends on failed check(s): {blocked}"})
            return
        try:
            detail = fn()
            results.append({"check": name, "ok": True, "status": "PASS", "detail": detail})
        except Exception as e:                                          # noqa: BLE001 - report everything
            tb = traceback.format_exc(limit=3) if not isinstance(e, PreflightFailure) else ""
            results.append({"check": name, "ok": False, "status": "FAIL", "detail": f"{type(e).__name__}: {e} {tb}".strip()})

    def c_deps():
        v = software_versions()
        missing = [k for k in ("numpy", "scipy", "sklearn", "pandas", "xgboost", "matplotlib") if not v.get(k)]
        if missing:
            raise PreflightFailure(f"missing packages: {missing}")
        return v

    def c_budgets_seeds():
        common = CommonSearchConfig.from_cfg(cfg)
        if list(common.query_budgets) != EXPECTED_BUDGETS or list(common.seeds) != EXPECTED_SEEDS:
            raise PreflightFailure(f"configured budgets/seeds {common.query_budgets}/{common.seeds} differ from the expected protocol")
        n = len(build_matrix(cfg))
        if n != 240:
            raise PreflightFailure(f"matrix has {n} runs, expected 240")
        return {"budgets": list(common.query_budgets), "seeds": list(common.seeds), "n_runs": n}

    def c_methods():
        if set(METHODS) != set(METHOD_ORDER) or len(METHODS) != 4:
            raise PreflightFailure(f"registered methods {tuple(METHODS)} != {METHOD_ORDER}")
        return list(METHODS)

    def c_static():
        bad = static_leakage_audit(ROOT / "src")
        if bad:
            raise PreflightFailure("static leakage audit violations: " + "; ".join(bad))
        return "no label/test/model-internal tokens in search modules"

    def c_resultdir():
        out = ensure_layout(resolve_path(a.out))
        t = out / "logs" / ".write_test"
        t.write_text("ok"); t.unlink()
        return str(out)

    def c_data_model():
        comp = load_real_components(cfg, a.model_tag)
        ctx["comp"] = comp
        p = comp["provenance"]
        return {"dataset": p["source"], "paper_eligible": p["paper_eligible"], "n_train": int(len(comp["prepared"].X_train)),
                "n_features": len(comp["prepared"].feature_names), "model_kind": comp["model"].kind,
                "model_sha256": comp["model_sha256"], "reference_fingerprint": comp["validator"].ref.fingerprint}

    def c_artifact():
        v = verify_model_artifact(ctx["comp"])
        m = v["report"]["metrics"]
        ctx["target_report"] = v["report"]
        return {"test_proba_sha256": v["test_proba_sha256"], "paper_eligible": v["report"]["paper_eligible"],
                "historical_metrics": {k: m.get(k) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}}

    def c_validator_oracle():
        c = ctx["comp"]
        return {"oracle": type(c["validator"].oracle).__name__, "validator": type(c["validator"]).__name__,
                "min_oracle_confidence": c["cfg"]["validator"]["failure_rule"]["min_oracle_confidence"]}

    def c_accounting():
        c = ctx["comp"]
        r = accounting_audit(c["cfg"], c["space"], c["validator"], c["model"], a.budget, a.seed)
        bad = [m for m, x in r.items() if not x["ok"]]
        if bad:
            raise PreflightFailure(f"query accounting failed for {bad}: {r}")
        return r

    def c_quantum():
        c = ctx["comp"]
        from src.experiment_runner import make_engine
        eng = make_engine("quantum", c["cfg"], c["space"], c["validator"], a.budget, a.seed)
        d = eng.circuit.describe()
        if eng.circuit.executions != 0:
            raise PreflightFailure("fresh circuit already executed")
        return d

    def c_repro():
        c = ctx["comp"]
        out = [reproducibility_audit(c["cfg"], c["space"], c["validator"], c["model"], m, a.budget, a.seed)
               for m in ("quantum", "cmaes")]
        bad = [o for o in out if not o["identical"]]
        if bad:
            raise PreflightFailure(f"not reproducible: {bad}")
        return out

    check("dependencies", c_deps)
    check("configured_budgets_seeds_matrix", c_budgets_seeds)
    check("four_search_methods_registered", c_methods)
    check("static_no_label_or_model_internal_access", c_static)
    check("result_directory", c_resultdir)
    check("real_cic_ids2017_and_target_artifact", c_data_model)
    check("target_model_artifact_matches_phase3_report_and_paper_eligible", c_artifact, ["real_cic_ids2017_and_target_artifact"])
    check("validator_and_oracle", c_validator_oracle, ["real_cic_ids2017_and_target_artifact"])
    check("query_budget_wrapper_accounting_and_dynamic_no_label_access_all_methods", c_accounting,
          ["real_cic_ids2017_and_target_artifact"])
    check("quantum_backend", c_quantum, ["real_cic_ids2017_and_target_artifact"])
    check("reproducibility_quantum_and_classical", c_repro, ["query_budget_wrapper_accounting_and_dynamic_no_label_access_all_methods"])

    ok = all(r["ok"] for r in results)
    print("=" * 78)
    for r in results:
        print(f"[{r['status']:7s}] {r['check']}")
        if not r["ok"] or "-v" in sys.argv:
            print(f"          {str(r['detail'])[:600]}")
    print("=" * 78)
    print("PREFLIGHT " + ("PASSED - campaign may start" if ok else "FAILED - DO NOT START THE 240-RUN CAMPAIGN"))
    log_dir = resolve_path(a.out) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "preflight.json").write_text(json.dumps({"passed": ok, "checks": results}, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
