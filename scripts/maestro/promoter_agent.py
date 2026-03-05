from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .common import load_json, save_json


def _extract_exact(path: Path) -> Dict[str, float]:
    payload = load_json(path, default={})
    out: Dict[str, float] = {}
    for row in payload.get("results", []):
        ckpt = str(row.get("checkpoint", ""))
        try:
            out[ckpt] = float(row.get("exact_match", 0.0))
        except Exception:
            out[ckpt] = 0.0
    return out


def _extract_prompt_metrics(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for bench_name, bench_payload in payload.items():
        for row in bench_payload.get("results", []):
            ckpt = str(row.get("checkpoint", ""))
            out.setdefault(ckpt, {})
            out[ckpt][f"{bench_name}.general_prompt_suite_pass_rate"] = float(row.get("general_prompt_suite_pass_rate", 0.0))
            out[ckpt][f"{bench_name}.collapse_rate"] = float(row.get("collapse_rate", 1.0))
            out[ckpt][f"{bench_name}.unknown_rate"] = float(row.get("unknown_rate", 1.0))
            out[ckpt][f"{bench_name}.definition_relevance_score"] = float(row.get("definition_relevance_score", 0.0))
            out[ckpt][f"{bench_name}.logic_reasoning_readout_rate"] = float(row.get("logic_reasoning_readout_rate", 0.0))
    return out


def decide(spec: Dict[str, Any], root: Path, out_dir: Path) -> Dict[str, Any]:
    eval_report = load_json(out_dir / "eval_report.json", default={})
    exact_scores: Dict[str, float] = {}
    for row in eval_report.get("exact_runs", []):
        out_json = Path(row["out_json"])
        if out_json.exists():
            exact_scores.update(_extract_exact(out_json))
    prompt_metrics = _extract_prompt_metrics(eval_report.get("scored_prompts", {}))

    thresholds = spec.get("promotion", {})
    decisions: List[Dict[str, Any]] = []
    for ckpt, logic_exact in sorted(exact_scores.items()):
        metrics = prompt_metrics.get(ckpt, {})
        general_pass = max([v for k, v in metrics.items() if k.endswith("general_prompt_suite_pass_rate")] or [0.0])
        collapse_rate = min([v for k, v in metrics.items() if k.endswith("collapse_rate")] or [1.0])
        unknown_rate = min([v for k, v in metrics.items() if k.endswith("unknown_rate")] or [1.0])
        definition_score = max([v for k, v in metrics.items() if k.endswith("definition_relevance_score")] or [0.0])
        decision = "hold"
        reasons: List[str] = []
        if logic_exact < float(thresholds.get("logic_exact_min", 0.10)):
            reasons.append("logic_exact below threshold")
        if general_pass < float(thresholds.get("general_prompt_suite_pass_rate_min", 0.50)):
            reasons.append("general prompt suite pass too low")
        if collapse_rate > float(thresholds.get("collapse_rate_max", 0.20)):
            reasons.append("collapse rate too high")
        if definition_score < float(thresholds.get("definition_relevance_score_min", 0.25)):
            reasons.append("definition relevance too low")
        if not reasons:
            decision = "promote"
        decisions.append(
            {
                "checkpoint": ckpt,
                "logic_exact_match": logic_exact,
                "general_prompt_suite_pass_rate": general_pass,
                "collapse_rate": collapse_rate,
                "unknown_rate": unknown_rate,
                "definition_relevance_score": definition_score,
                "decision": decision,
                "reasons": reasons,
            }
        )
    report = {"decisions": decisions, "winner": next((x["checkpoint"] for x in decisions if x["decision"] == "promote"), "")}
    save_json(out_dir / "promotion_report.json", report)
    return report
