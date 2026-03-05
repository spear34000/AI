from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from train_ccl_compile import build_model_from_checkpoint, normalize_tier, safe_json_print
from omega2_agentic_core import (
    PatchDB,
    PatchRecord,
    build_distill_rows,
    build_runtime_prompt,
    choose_best_rule_text,
    extract_keywords,
    generate_with_search,
    hard_verify_response,
    load_specs,
    parse_search_list,
    soft_judge_score,
)


@dataclass
class WorkItem:
    spec: Any
    runtime_prompt: str
    response: str
    retrieved_patch_ids: List[str]
    search_meta: Dict[str, Any]


def choose_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable")
        return torch.device("cuda")
    return torch.device("cpu")


def choose_specs(
    specs: Sequence[Any],
    spec_by_id: Dict[str, Any],
    batch_size: int,
    rnd: random.Random,
    focus_scores: Dict[str, int],
    focus_ratio: float,
) -> List[Any]:
    n = max(1, int(batch_size))
    out: List[Any] = []

    focus_ids = sorted(focus_scores.keys(), key=lambda k: focus_scores.get(k, 0), reverse=True)
    n_focus = min(len(focus_ids), int(round(float(focus_ratio) * n)))
    for sid in focus_ids[:n_focus]:
        spec = spec_by_id.get(str(sid))
        if spec is not None:
            out.append(spec)

    remaining = [s for s in specs if str(s.spec_id) not in {str(x.spec_id) for x in out}]
    rnd.shuffle(remaining)
    out.extend(remaining[: max(0, n - len(out))])
    rnd.shuffle(out)
    return out[:n]


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_branches(args: argparse.Namespace) -> List[Tuple[float, int, float]]:
    temps = parse_search_list(args.search_temps, float, [0.0, 0.6, 0.9])
    topks = parse_search_list(args.search_top_k, int, [0, 60, 120])
    topps = parse_search_list(args.search_top_p, float, [1.0, 0.92, 0.97])
    m = max(len(temps), len(topks), len(topps))
    while len(temps) < m:
        temps.append(temps[-1])
    while len(topks) < m:
        topks.append(topks[-1])
    while len(topps) < m:
        topps.append(topps[-1])
    branches = [(float(temps[i]), int(topks[i]), float(topps[i])) for i in range(m)]
    return branches[: max(1, int(args.search_branches))]


def maybe_distill(
    args: argparse.Namespace,
    step: int,
    patch_db: PatchDB,
    out_dir: Path,
    base_checkpoint: Path,
) -> Dict[str, Any]:
    distill_dir = out_dir / "distill"
    distill_dir.mkdir(parents=True, exist_ok=True)

    rows = build_distill_rows(patch_db.active_records(), max_rows=int(args.distill_max_rows))
    ds_path = distill_dir / f"distill_dataset_step_{step:05d}.jsonl"
    with ds_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    plan = {
        "step": int(step),
        "dataset_path": str(ds_path),
        "rows": int(len(rows)),
        "recommended_command": [
            "python",
            "scripts/train_slm.py",
            "--data_path",
            str(ds_path),
            "--output_dir",
            str(Path(args.distill_output_dir)),
            "--resume_from",
            str(base_checkpoint),
            "--steps",
            str(int(args.distill_steps)),
            "--batch_size",
            str(int(args.distill_batch_size)),
            "--eval_interval",
            "25",
            "--save_interval",
            "50",
            "--skip_sample",
        ],
        "started": False,
        "returncode": None,
    }

    if bool(args.auto_distill):
        plan["started"] = True
        cmd = list(plan["recommended_command"])
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent), check=False)
        plan["returncode"] = int(proc.returncode)

    plan_path = distill_dir / f"distill_plan_step_{step:05d}.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Omega Compiler 2.0 runtime loop (Agentic OS)")
    parser.add_argument("--base_checkpoint", default="artifacts_ondevice_best/slm_ondevice_fp16.pt")
    parser.add_argument("--spec_path", default="data/ccl_specbook_v1.jsonl")
    parser.add_argument("--data_path", default="data/slm_mit_unified_v4.jsonl")
    parser.add_argument("--output_dir", default="artifacts_omega2_agentic")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--max_specs", type=int, default=1200)
    parser.add_argument("--generate_batch", type=int, default=24)
    parser.add_argument("--focus_ratio", type=float, default=0.6)
    parser.add_argument("--patch_top_k", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=120)

    parser.add_argument("--search_branches", type=int, default=3)
    parser.add_argument("--search_temps", type=str, default="0.0,0.6,0.9")
    parser.add_argument("--search_top_k", type=str, default="0,60,120")
    parser.add_argument("--search_top_p", type=str, default="1.0,0.92,0.97")

    parser.add_argument("--soft_threshold", type=float, default=0.42)
    parser.add_argument("--novelty_threshold", type=float, default=0.90)
    parser.add_argument("--focus_decay", type=float, default=0.90)
    parser.add_argument("--focus_max", type=int, default=1200)

    parser.add_argument("--distill_trigger_patches", type=int, default=1000)
    parser.add_argument("--distill_cooldown_steps", type=int, default=24)
    parser.add_argument("--distill_max_rows", type=int, default=2000)
    parser.add_argument("--auto_distill", action="store_true")
    parser.add_argument("--distill_output_dir", default="artifacts_omega2_distill")
    parser.add_argument("--distill_steps", type=int, default=160)
    parser.add_argument("--distill_batch_size", type=int, default=24)

    parser.add_argument("--flush_after_distill", action="store_true")
    parser.add_argument("--save_interval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    rnd = random.Random(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = choose_device(args.device)
    base_ckpt = Path(args.base_checkpoint)
    if not base_ckpt.exists():
        raise RuntimeError(f"base_checkpoint not found: {base_ckpt}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_db_path = out_dir / "patch_db.jsonl"
    history_path = out_dir / "omega2_history.jsonl"
    cex_path = out_dir / "counterexample_ledger.jsonl"

    model, tokenizer, model_args = build_model_from_checkpoint(base_ckpt, device=device)
    seq_len = int(args.seq_len) if int(args.seq_len) > 0 else int(model.seq_len)

    specs, spec_source = load_specs(
        spec_path=Path(args.spec_path),
        data_path=Path(args.data_path),
        max_specs=int(args.max_specs),
        seed=int(args.seed),
    )
    if not specs:
        raise RuntimeError("empty specs")
    spec_by_id = {str(s.spec_id): s for s in specs}
    hard_count = sum(1 for s in specs if normalize_tier(s.tier) == "hard")

    patch_db = PatchDB(path=patch_db_path, novelty_threshold=float(args.novelty_threshold))
    patch_db.load()

    branches = build_branches(args)
    focus_scores: Dict[str, int] = {}
    last_distill_step = -10_000
    start_ts = time.time()

    safe_json_print(
        {
            "mode": "omega2_agentic_os",
            "device": str(device),
            "base_checkpoint": str(base_ckpt),
            "spec_source": spec_source,
            "specs_total": int(len(specs)),
            "specs_hard": int(hard_count),
            "steps": int(args.steps),
            "generate_batch": int(args.generate_batch),
            "patch_top_k": int(args.patch_top_k),
            "search_branches": branches,
            "active_patches_start": int(patch_db.active_count()),
        }
    )

    for step in range(1, int(args.steps) + 1):
        t0 = time.time()
        batch = choose_specs(
            specs=specs,
            spec_by_id=spec_by_id,
            batch_size=int(args.generate_batch),
            rnd=rnd,
            focus_scores=focus_scores,
            focus_ratio=float(args.focus_ratio),
        )

        work_q: deque[WorkItem] = deque()
        for spec in batch:
            retrieved = patch_db.retrieve(spec.prompt, top_k=int(args.patch_top_k))
            patch_ids = [r.patch_id for r in retrieved]
            patch_db.mark_used(patch_ids)
            runtime_prompt = build_runtime_prompt(spec.prompt, retrieved)

            response, search_meta = generate_with_search(
                model=model,
                tokenizer=tokenizer,
                prompt=runtime_prompt,
                target=spec.target,
                device=device,
                max_new_tokens=int(args.max_new_tokens),
                branches=branches,
            )
            work_q.append(
                WorkItem(
                    spec=spec,
                    runtime_prompt=runtime_prompt,
                    response=response,
                    retrieved_patch_ids=patch_ids,
                    search_meta=search_meta,
                )
            )

        hard_pass_n = 0
        soft_pass_n = 0
        fail_n = 0
        patches_added_n = 0
        pass_hard_spec_ids: set[str] = set()
        cex_rows: List[Dict[str, Any]] = []
        failures_for_focus: List[str] = []

        while work_q:
            item = work_q.popleft()
            spec = item.spec

            hard_ok, hard_reason, hard_metric = hard_verify_response(
                model=model,
                tokenizer=tokenizer,
                spec=spec,
                prompt_for_eval=item.runtime_prompt,
                response=item.response,
                seq_len=seq_len,
                device=device,
            )
            soft_score = soft_judge_score(target=spec.target, response=item.response)
            soft_ok = soft_score >= float(args.soft_threshold)

            if hard_ok:
                hard_pass_n += 1
            if soft_ok:
                soft_pass_n += 1

            tier = normalize_tier(spec.tier)
            passed = hard_ok if tier == "hard" else (hard_ok or soft_ok)
            if passed and tier == "hard":
                pass_hard_spec_ids.add(str(spec.spec_id))

            if not passed:
                fail_n += 1
                failures_for_focus.append(str(spec.spec_id))
                failure_reason = hard_reason if not hard_ok else f"soft<{float(args.soft_threshold):.2f}"
                rule_text = choose_best_rule_text(spec=spec, failure_reason=failure_reason, response=item.response)
                kws = extract_keywords(spec.prompt, spec.target)
                added, rec = patch_db.add_patch(
                    spec=spec,
                    rule_text=rule_text,
                    keywords=kws,
                    created_step=int(step),
                    evidence={
                        "failure_reason": failure_reason,
                        "hard_metric": hard_metric,
                        "soft_score": soft_score,
                        "retrieved_patch_ids": item.retrieved_patch_ids,
                    },
                )
                if added:
                    patches_added_n += 1

                cex_rows.append(
                    {
                        "step": int(step),
                        "spec_id": str(spec.spec_id),
                        "tier": str(spec.tier),
                        "failure_reason": str(failure_reason),
                        "hard_ok": bool(hard_ok),
                        "soft_score": float(soft_score),
                        "response": str(item.response),
                        "rule_patch_added": bool(added),
                        "rule_patch_id": rec.patch_id if rec is not None else "",
                    }
                )

        append_jsonl(cex_path, cex_rows)

        # Goal-directed focus update.
        decay = float(args.focus_decay)
        for sid in list(focus_scores.keys()):
            focus_scores[sid] = int(max(0, round(focus_scores[sid] * decay)))
            if focus_scores[sid] <= 0:
                focus_scores.pop(sid, None)
        for sid in failures_for_focus:
            focus_scores[sid] = int(focus_scores.get(sid, 0) + 3)
        if len(focus_scores) > int(args.focus_max):
            keep = sorted(focus_scores.items(), key=lambda kv: kv[1], reverse=True)[: int(args.focus_max)]
            focus_scores = {k: int(v) for k, v in keep}

        distill_event: Dict[str, Any] | None = None
        need_distill = (
            patch_db.active_count() >= int(args.distill_trigger_patches)
            and (step - last_distill_step) >= int(args.distill_cooldown_steps)
        )
        if need_distill:
            distill_event = maybe_distill(
                args=args,
                step=step,
                patch_db=patch_db,
                out_dir=out_dir,
                base_checkpoint=base_ckpt,
            )
            last_distill_step = int(step)
            if bool(args.flush_after_distill):
                flushed = patch_db.flush_specs(pass_hard_spec_ids)
                if distill_event is not None:
                    distill_event["flushed_patches"] = int(flushed)

        if step % int(args.save_interval) == 0 or step == int(args.steps) or patches_added_n > 0 or (distill_event is not None):
            patch_db.save()

        checked = max(1, len(batch))
        step_row = {
            "step": int(step),
            "checked": int(len(batch)),
            "hard_pass_rate": float(hard_pass_n / checked),
            "soft_pass_rate": float(soft_pass_n / checked),
            "failures": int(fail_n),
            "patches_added": int(patches_added_n),
            "active_patches": int(patch_db.active_count()),
            "focus_specs": int(len(focus_scores)),
            "elapsed_step_sec": float(time.time() - t0),
        }
        if distill_event is not None:
            step_row["distill"] = distill_event
        append_jsonl(history_path, [step_row])
        safe_json_print(step_row)

    patch_db.save()
    summary = {
        "finished": True,
        "elapsed_sec": float(time.time() - start_ts),
        "base_checkpoint": str(base_ckpt),
        "spec_source": spec_source,
        "steps": int(args.steps),
        "active_patches_end": int(patch_db.active_count()),
        "patch_db": str(patch_db_path),
        "history_jsonl": str(history_path),
        "counterexample_ledger": str(cex_path),
        "model_args": model_args,
    }
    (out_dir / "omega2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    safe_json_print(summary)


if __name__ == "__main__":
    main()

