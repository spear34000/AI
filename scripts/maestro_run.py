from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from maestro.common import append_jsonl, load_json, now_iso, repo_root, save_json
from maestro.dataset_agent import build_datasets
from maestro.evaluator_agent import run_evaluation
from maestro.planner_agent import build_plan
from maestro.promoter_agent import decide
from maestro.trainer_agent import build_train_plan


def log_phase(path: Path, phase: str, status: str, payload: Dict[str, Any]) -> None:
    append_jsonl(
        path,
        [
            {
                "ts": now_iso(),
                "phase": phase,
                "status": status,
                "payload": payload,
            }
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--phases", default="planner,builder,trainer,evaluator,promoter")
    args = ap.parse_args()

    root = repo_root()
    spec = load_json(Path(args.spec), default={})
    if not isinstance(spec, dict):
        raise RuntimeError("spec must be a JSON object")

    experiment_id = str(spec.get("experiment_id", "mainline_single_v1"))
    out_dir = root / "artifacts_maestro" / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_history = out_dir / "run_history.jsonl"

    phases = [x.strip() for x in str(args.phases).split(",") if x.strip()]

    planner: Dict[str, Any] = {}
    if "planner" in phases:
        planner = build_plan(spec, root)
        save_json(out_dir / "planner_decision.json", planner)
        log_phase(run_history, "planner", "ok", {"planner_decision": str(out_dir / "planner_decision.json")})
    else:
        planner = load_json(out_dir / "planner_decision.json", default={}) or {}

    if "builder" in phases:
        report = build_datasets(spec, root, out_dir)
        log_phase(run_history, "builder", "ok", {"dataset_report": str(out_dir / "dataset_report.json"), "summary": {"general_train_rows": report["general"]["train_rows"], "logic_train_rows": report["logic"]["train_rows"]}})

    if "trainer" in phases:
        report = build_train_plan(spec, planner, root, out_dir)
        log_phase(run_history, "trainer", "ok", {"train_report": str(out_dir / "train_report.json"), "execute_train": report["execute_train"]})

    if "evaluator" in phases:
        report = run_evaluation(spec, planner, root, out_dir)
        log_phase(run_history, "evaluator", "ok", {"eval_report": str(out_dir / "eval_report.json"), "checkpoints": report["checkpoints"]})

    if "promoter" in phases:
        report = decide(spec, root, out_dir)
        log_phase(run_history, "promoter", "ok", {"promotion_report": str(out_dir / "promotion_report.json"), "winner": report.get("winner", "")})


if __name__ == "__main__":
    main()
