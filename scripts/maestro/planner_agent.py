from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .common import load_json, pick_first_existing


def _safe_exact(path: Path) -> float | None:
    payload = load_json(path, default=None)
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            try:
                return float(first.get("exact_match"))
            except Exception:
                return None
    try:
        return float(payload.get("exact_match"))
    except Exception:
        return None


def build_plan(spec: Dict[str, Any], root: Path) -> Dict[str, Any]:
    candidates = spec.get("base_candidates", {})
    general_base = pick_first_existing(candidates.get("general_logic", []))
    logic_reference = pick_first_existing(candidates.get("logic_reference", []))
    arithmetic_reference = pick_first_existing(candidates.get("arithmetic_reference", []))

    baseline_logic_eval = root / "artifacts_teacher_arith_pure_mix_fromscratch_v3_runA" / "eval_v3_512_compare.json"
    stage3_logic_eval = root / "artifacts_bigslm_50m_safe_fromscratch_v1_runC_stage3" / "eval_fixed_logic_compare.json"

    return {
        "experiment_id": str(spec.get("experiment_id", "mainline_single_v1")),
        "goal": str(spec.get("goal", "single_pure_model")),
        "priority_capability": str(spec.get("priority_capability", "logic")),
        "selected": {
            "mainline_base_checkpoint": general_base,
            "logic_reference_checkpoint": logic_reference,
            "arithmetic_reference_checkpoint": arithmetic_reference,
        },
        "known_failures": [
            "loss != exact_match",
            "narrow fine-tune causes catastrophic forgetting",
            "arith-only branch contaminates general prompts",
        ],
        "baseline_metrics": {
            "arithmetic_branch_exact_v3": _safe_exact(baseline_logic_eval),
            "logic_stage3_compare_exact": _safe_exact(stage3_logic_eval),
        },
        "phases": {
            "phase1_general": {
                "dataset_id": "mainline_general_v3_clean",
                "resume_from": general_base,
                "response_loss_only": False,
            },
            "phase2_logic": {
                "dataset_id": "mainline_logic_mix_v3",
                "resume_from": "artifacts_mainline_general_v3_runA/slm_best.pt",
                "mix_ratio": {"general": 0.85, "logic": 0.15},
                "response_loss_only": True,
            },
            "phase3_general_repair": {
                "enabled": True,
                "dataset_id": "mainline_logic_repair_mix_v3",
                "resume_from": "artifacts_mainline_logic_v3_runA/slm_best.pt",
                "response_loss_only": False,
            },
        },
        "promotion_gates": spec.get("promotion", {}),
    }
