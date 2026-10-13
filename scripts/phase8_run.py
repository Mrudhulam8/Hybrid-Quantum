"""Phase 8 driver.

    python scripts/phase8_run.py smoke     [--budget 50] [--seed 1]      # STEP 7: all 4 methods on REAL data, own dir
    python scripts/phase8_run.py campaign  [--limit N]                    # STEP 9: 240 runs, resumable
    python scripts/phase8_run.py analyze                                  # STEPS 10-14 (refuses ineligible/incomplete)

Requires `python scripts/phase8_preflight.py` to have passed (checked via logs/preflight.json).
Never uses synthetic data or a fallback dataset.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase8 import (METHOD_ORDER, ResultStore, build_manifest, build_matrix, ensure_layout,     # noqa: E402
                        run_campaign)
from src.phase8_report import analyze_campaign                                                      # noqa: E402
from src.phase8_setup import accounting_audit, load_real_components, make_fingerprint, verify_model_artifact   # noqa: E402
from src.utils import load_config, resolve_path, save_json                                          # noqa: E402


def require_preflight(out: Path):
    p = out / "logs" / "preflight.json"
    if not p.exists() or not json.loads(p.read_text()).get("passed"):
        raise SystemExit("preflight has not passed (results/phase8/logs/preflight.json). Refusing to run.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["smoke", "campaign", "analyze"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-tag", default="xgboost_cic-ids2017_binary")
    ap.add_argument("--out", default="results/phase8")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="execute at most N missing runs (resumable testing)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = resolve_path(a.out)
    ensure_layout(out)
    require_preflight(out)
    comp = load_real_components(cfg, a.model_tag)
    cfg = comp["cfg"]
    ver = verify_model_artifact(comp)
    fp = make_fingerprint(comp)
    dataset, model_name = comp["provenance"]["source"], comp["model_tag"]

    if a.command == "smoke":
        sdir = out / "smoke"
        specs = build_matrix(cfg, budgets=[a.budget], seeds=[a.seed])
        r = run_campaign(specs, cfg, comp["space"], comp["validator"], comp["model"], sdir, config_fingerprint=fp,
                         paper_eligible_context=True, dataset=dataset, model_name=model_name, progress=print)
        acc = accounting_audit(cfg, comp["space"], comp["validator"], comp["model"], a.budget, a.seed)
        print(json.dumps({"smoke": r, "accounting": acc}, indent=1))
        return 0 if r["completed_total"] == len(specs) and all(x["ok"] for x in acc.values()) else 1

    specs = build_matrix(cfg)
    if a.command == "campaign":
        done = ResultStore(out / "raw" / "run_results.jsonl").ok_records()
        todo = [s for s in specs if s.run_id not in done]
        r = run_campaign(specs if a.limit is None else todo[:a.limit], cfg, comp["space"], comp["validator"], comp["model"],
                         out, config_fingerprint=fp, paper_eligible_context=True, dataset=dataset, model_name=model_name,
                         progress=print)
        ok = ResultStore(out / "raw" / "run_results.jsonl").ok_records()
        print(json.dumps({**r, "completed_of_240": sum(s.run_id in ok for s in specs)}, indent=1))
        return 0

    # analyze
    ok = ResultStore(out / "raw" / "run_results.jsonl").ok_records()
    qrt = next((r["method_specific"] for r in ok.values() if r["method"] == "quantum"), {})
    complete = all(s.run_id in ok for s in specs)
    warnings = [] if complete else [f"campaign incomplete: {sum(s.run_id in ok for s in specs)}/{len(specs)} runs"]
    manifest = build_manifest(cfg, specs, root=ROOT, dataset_provenance=comp["provenance"],
                              model_artifact={"path": str(comp["model_file"]), "sha256": comp["model_sha256"],
                                              "meta_sha256": comp["model_meta_sha256"], "tag": comp["model_tag"],
                                              "test_proba_sha256": ver["test_proba_sha256"]},
                              config_fingerprint=fp, paper_eligible=complete and all(r["paper_eligible"] for r in ok.values()),
                              warnings=warnings, quantum_runtime=qrt)
    save_json(manifest, out / "experiment_manifest.json")
    res = analyze_campaign(out, cfg, len(specs), manifest, target_model_report=ver["report"])
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
