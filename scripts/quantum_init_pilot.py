
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, save_json
from src.phase8_setup import load_real_components
from src.experiment_runner import run_search_method


def main() -> int:
    config_path = ROOT / "configs" / "default.yaml"
    output_dir = ROOT / "results" / "quantum_init_pilot_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "pilot_results.jsonl"
    summary_file = output_dir / "summary.json"

    # Never overwrite previous pilot results.
    if output_file.exists() or summary_file.exists():
        raise SystemExit(
            f"Pilot output already exists in {output_dir}. "
            "Rename the folder before running again."
        )

    print("Loading project configuration and real model components...", flush=True)
    cfg = load_config(str(config_path))
    components = load_real_components(
        cfg, "xgboost_cic-ids2017_binary"
    )

    if not isinstance(components, dict):
        raise SystemExit(
            "Expected load_real_components() to return a dictionary."
        )

    space = components.get("space")
    validator = components.get("validator")
    target_model = components.get("model")

    if space is None or validator is None or target_model is None:
        raise SystemExit(
            "Missing space, validator, or model in loaded components."
        )

    records = []

    for initialization in ("uniform", "reference_guided"):
        for seed in (1, 2):
            print(
                f"\nStarting quantum run: initialization={initialization}, "
                f"seed={seed}, budget=100",
                flush=True,
            )

            started = time.time()
            record = {
                "method": "quantum",
                "initialization": initialization,
                "seed": seed,
                "budget": 100,
            }

            try:
                # Reload config for each run, changing only initialization.
                run_cfg = load_config(str(config_path))
                run_cfg.setdefault("search", {}).setdefault("quantum", {})
                run_cfg["search"]["quantum"]["initialization"] = initialization

                # Confirm the setting that will be used.
                actual_init = run_cfg["search"]["quantum"]["initialization"]
                if actual_init != initialization:
                    raise RuntimeError(
                        f"Initialization mismatch: expected {initialization}, "
                        f"got {actual_init}"
                    )

                result = run_search_method(
                    method="quantum",
                    cfg=run_cfg,
                    space=space,
                    validator=validator,
                    target_model=target_model,
                    budget=100,
                    seed=seed,
                )

                if not hasattr(result, "to_dict"):
                    raise TypeError(
                        "Search result does not provide to_dict(); "
                        "refusing to record an incomplete result."
                    )

                result_data = result.to_dict()
                record["result"] = result_data
                record["status"] = "ok"

                # Record key metrics when present in the result schema.
                for field in (
                    "queries_used",
                    "candidate_count",
                    "verified_failures",
                    "unique_verified_failures",
                    "verified_failure_ids",
                    "termination_reason",
                    "warnings",
                ):
                    if field in result_data:
                        record[field] = result_data[field]

                record["config_initialization"] = actual_init

            except Exception as exc:
                record["status"] = "error"
                record["error"] = repr(exc)
                record["traceback"] = traceback.format_exc()
                print(f"ERROR: {exc}", flush=True)

            record["runtime_seconds"] = round(time.time() - started, 3)
            records.append(record)

            with output_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

            print(
                f"Finished: status={record['status']}, "
                f"runtime={record['runtime_seconds']}s",
                flush=True,
            )

    save_json({"runs": records}, summary_file)

    successful = sum(r["status"] == "ok" for r in records)
    failed = len(records) - successful

    print(f"\nPilot completed: {successful}/4 successful, {failed} failed.")
    print(f"Detailed results: {output_file}")
    print(f"Summary: {summary_file}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
