from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from .common import resolve_python, run_subprocess, save_json


def _spm_model(root: Path) -> str:
    return str(root / "artifacts_tokenizer_spm_from_stage2_v1" / "spm16k.model")


def _base_train_args(root: Path) -> List[str]:
    return [
        "--batch_size", "1",
        "--grad_accum_steps", "16",
        "--seq_len", "384",
        "--weight_decay", "0.1",
        "--warmup_steps", "200",
        "--grad_clip", "1.0",
        "--val_ratio", "0.02",
        "--eval_interval", "200",
        "--save_interval", "400",
        "--eval_batches", "12",
        "--num_workers", "0",
        "--d_model", "576",
        "--n_heads", "9",
        "--n_layers", "12",
        "--mlp_mult", "4",
        "--tokenizer_type", "spm",
        "--tokenizer_model", _spm_model(root),
        "--activation_checkpointing",
        "--skip_sample",
        "--ema_decay", "0.0",
        "--consistency_lambda", "0.0",
        "--plain_sft",
    ]


def _checkpoint_step(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Fallback for older torch versions without the `weights_only` argument.
            ckpt = torch.load(path, map_location="cpu")
        if not isinstance(ckpt, dict):
            return 0
        return int(ckpt.get("step", 0))
    except Exception:
        return 0


def build_train_plan(spec: Dict[str, Any], planner: Dict[str, Any], root: Path, out_dir: Path) -> Dict[str, Any]:
    py = resolve_python(root)
    base_ckpt = str(planner["selected"]["mainline_base_checkpoint"])
    base_step = _checkpoint_step(root / base_ckpt)
    execution = spec.get("execution", {})
    run_tag = str(execution.get("run_tag", "runA"))
    phase1_from_scratch = bool(execution.get("phase1_from_scratch", False))
    reset_step_on_resume = bool(execution.get("reset_step_on_resume", True))

    phase1_out = root / f"artifacts_mainline_general_v3_{run_tag}"
    phase2_out = root / f"artifacts_mainline_logic_v3_{run_tag}"
    phase3_out = root / f"artifacts_mainline_general_repair_v3_{run_tag}"
    phase1_steps = int(execution.get("phase1_steps", 18000 if phase1_from_scratch else base_step + 6000))
    phase2_steps = int(execution.get("phase2_steps", phase1_steps + 6000))
    phase3_steps = int(execution.get("phase3_steps", 2200))
    phase3_data_path = str(root / "data" / "mainline_logic_repair_mix_v3_train.jsonl")

    phase1 = [
        py, "scripts/train_slm.py",
        "--data_path", "data/mainline_general_v3_clean_train.jsonl",
        "--output_dir", str(phase1_out),
        "--steps", str(phase1_steps),
        "--lr", "8e-6",
    ] + _base_train_args(root)
    if not phase1_from_scratch:
        phase1.extend([
            "--resume_from", base_ckpt,
            "--resume_weights_only",
        ])
        if reset_step_on_resume:
            phase1.append("--reset_step_on_resume")
        phase1.extend([
            "--reset_best_on_resume",
        ])

    phase2 = [
        py, "scripts/train_slm.py",
        "--data_path", "data/mainline_logic_mix_v3_train.jsonl",
        "--output_dir", str(phase2_out),
        "--steps", str(phase2_steps),
        "--lr", "3e-6",
        "--resume_from", str(phase1_out / "slm_best.pt"),
        "--resume_weights_only",
        "--reset_step_on_resume",
        "--reset_best_on_resume",
        "--response_loss_only",
    ] + _base_train_args(root)

    phase3 = [
        py, "scripts/train_slm.py",
        "--data_path", phase3_data_path,
        "--output_dir", str(phase3_out),
        "--steps", str(phase3_steps),
        "--lr", "8e-7",
        "--resume_from", str(phase2_out / "slm_best.pt"),
        "--resume_weights_only",
        "--reset_step_on_resume",
        "--reset_best_on_resume",
    ] + _base_train_args(root)

    execute_train = bool(spec.get("execution", {}).get("run_train", False))
    allowed_phases = spec.get("execution", {}).get("train_phases", [])
    if isinstance(allowed_phases, list) and allowed_phases:
        allowed_set = {str(x) for x in allowed_phases}
    else:
        allowed_set = set()
    runs: List[Dict[str, Any]] = []
    for phase_name, cmd in [
        ("phase1_general", phase1),
        ("phase2_logic", phase2),
        ("phase3_general_repair", phase3),
    ]:
        row = {"phase": phase_name, "command": cmd, "started": False}
        should_run = execute_train and (not allowed_set or phase_name in allowed_set)
        if should_run:
            row["started"] = True
            row["result"] = run_subprocess(cmd, cwd=root, timeout=86400)
        runs.append(row)

    report = {
        "python_exe": py,
        "base_checkpoint": base_ckpt,
        "base_checkpoint_step": base_step,
        "run_tag": run_tag,
        "phase1_from_scratch": phase1_from_scratch,
        "reset_step_on_resume": reset_step_on_resume,
        "execute_train": execute_train,
        "train_phases": sorted(allowed_set) if allowed_set else ["phase1_general", "phase2_logic", "phase3_general_repair"],
        "phases": runs,
    }
    save_json(out_dir / "train_report.json", report)
    return report
