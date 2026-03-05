from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from train_ccl_compile import build_model_from_checkpoint, normalize_tier, safe_json_print
from omega2_agentic_core import extract_keywords, hard_verify_response, load_specs, soft_judge_score
from omega3_holographic_core import (
    HolographicVectorMemory,
    build_patch_vectors_from_failure,
    build_skill_vectors_from_success,
    choose_default_layers,
    generate_with_steering,
    mezo_tune_scalars,
    parse_layer_list,
)


def choose_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable")
        return torch.device("cuda")
    return torch.device("cpu")


def sample_specs(
    specs: Sequence[Any],
    batch_size: int,
    rnd: random.Random,
    fail_focus: Dict[str, int],
    focus_ratio: float,
) -> List[Any]:
    n = max(1, int(batch_size))
    by_id = {str(s.spec_id): s for s in specs}

    focus_ids = sorted(fail_focus.keys(), key=lambda k: fail_focus.get(k, 0), reverse=True)
    n_focus = min(len(focus_ids), int(round(float(focus_ratio) * n)))
    out: List[Any] = []
    for sid in focus_ids[:n_focus]:
        s = by_id.get(str(sid))
        if s is not None:
            out.append(s)

    remain = [s for s in specs if str(s.spec_id) not in {str(x.spec_id) for x in out}]
    rnd.shuffle(remain)
    out.extend(remain[: max(0, n - len(out))])
    rnd.shuffle(out)
    return out[:n]


def decay_focus(focus: Dict[str, int], decay: float, max_size: int) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in focus.items():
        nv = int(round(float(v) * float(decay)))
        if nv > 0:
            out[str(k)] = nv
    if len(out) > int(max_size):
        keep = sorted(out.items(), key=lambda kv: kv[1], reverse=True)[: int(max_size)]
        out = {k: int(v) for k, v in keep}
    return out


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Omega Compiler 3.0 holographic steering loop")
    parser.add_argument("--base_checkpoint", default="artifacts_ondevice_best/slm_ondevice_fp16.pt")
    parser.add_argument("--spec_path", default="data/ccl_specbook_v1.jsonl")
    parser.add_argument("--data_path", default="data/slm_mit_unified_v4.jsonl")
    parser.add_argument("--output_dir", default="artifacts_omega3_holographic")

    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--max_specs", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--focus_ratio", type=float, default=0.6)
    parser.add_argument("--focus_decay", type=float, default=0.90)
    parser.add_argument("--focus_max", type=int, default=1200)

    parser.add_argument("--layers", type=str, default="")
    parser.add_argument("--memory_top_k", type=int, default=6)
    parser.add_argument("--kind_weights", type=str, default="skill:1.0,patch:1.3")
    parser.add_argument("--base_alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=80)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=120)

    parser.add_argument("--soft_threshold", type=float, default=0.42)
    parser.add_argument("--merge_cosine", type=float, default=0.985)
    parser.add_argument("--memory_max_norm", type=float, default=2.0)

    parser.add_argument("--mezo_iters", type=int, default=3)
    parser.add_argument("--mezo_sigma", type=float, default=0.12)
    parser.add_argument("--mezo_lr", type=float, default=0.25)
    parser.add_argument("--mezo_scalar_clip", type=float, default=2.5)

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
        raise RuntimeError(f"base checkpoint not found: {base_ckpt}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_path = out_dir / "holo_memory.pt"
    history_path = out_dir / "omega3_history.jsonl"
    fail_path = out_dir / "failure_ledger.jsonl"

    model, tokenizer, model_args = build_model_from_checkpoint(base_ckpt, device=device)
    model.eval()
    seq_len = int(args.seq_len) if int(args.seq_len) > 0 else int(model.seq_len)

    layers = parse_layer_list(args.layers, model=model, default_max_layers=4)
    if not layers:
        layers = choose_default_layers(model, max_layers=4)

    specs, spec_source = load_specs(
        spec_path=Path(args.spec_path),
        data_path=Path(args.data_path),
        max_specs=int(args.max_specs),
        seed=int(args.seed),
    )
    if not specs:
        raise RuntimeError("no specs loaded")
    hard_specs = sum(1 for s in specs if normalize_tier(s.tier) == "hard")

    kind_weights: Dict[str, float] = {"skill": 1.0, "patch": 1.3}
    for token in str(args.kind_weights or "").split(","):
        t = token.strip()
        if ":" not in t:
            continue
        k, v = t.split(":", 1)
        try:
            kind_weights[str(k).strip().lower()] = float(v.strip())
        except Exception:
            continue

    memory = HolographicVectorMemory(path=memory_path, merge_cosine=float(args.merge_cosine))
    memory.load()

    fail_focus: Dict[str, int] = {}
    start_ts = time.time()
    safe_json_print(
        {
            "mode": "omega3_holographic_kernel",
            "device": str(device),
            "base_checkpoint": str(base_ckpt),
            "spec_source": spec_source,
            "specs_total": int(len(specs)),
            "specs_hard": int(hard_specs),
            "layers": layers,
            "steps": int(args.steps),
            "memory_start": {"skill": int(memory.count("skill")), "patch": int(memory.count("patch"))},
        }
    )

    for step in range(1, int(args.steps) + 1):
        t0 = time.time()
        batch = sample_specs(
            specs=specs,
            batch_size=int(args.batch_size),
            rnd=rnd,
            fail_focus=fail_focus,
            focus_ratio=float(args.focus_ratio),
        )

        step_fail_rows: List[Dict[str, Any]] = []
        pass_hard = 0
        pass_soft = 0
        fail_n = 0
        added_skill = 0
        added_patch = 0

        for spec in batch:
            # Retrieval: no token concatenation, only vector injection.
            retrieved = memory.retrieve(spec.prompt, top_k=int(args.memory_top_k))
            retrieved_ids = [rec.vector_id for _, rec in retrieved]
            memory.mark_used(retrieved_ids)
            layer_vectors = memory.compose_layer_vectors(
                retrieved=retrieved,
                kind_weights=kind_weights,
                max_norm=float(args.memory_max_norm),
            )
            if not layer_vectors:
                layer_vectors = {int(l): torch.zeros(model.head.in_features, dtype=torch.float32) for l in layers}

            base_scalars = {int(l): float(args.base_alpha) for l in layer_vectors.keys()}
            response = generate_with_steering(
                model=model,
                tokenizer=tokenizer,
                prompt=spec.prompt,
                device=device,
                max_new_tokens=int(args.max_new_tokens),
                layer_vectors=layer_vectors,
                alpha_by_layer=base_scalars,
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
            )

            hard_ok, hard_reason, _nll = hard_verify_response(
                model=model,
                tokenizer=tokenizer,
                spec=spec,
                prompt_for_eval=spec.prompt,
                response=response,
                seq_len=seq_len,
                device=device,
            )
            soft = soft_judge_score(spec.target, response)
            tier = normalize_tier(spec.tier)
            passed = hard_ok if tier == "hard" else (hard_ok or soft >= float(args.soft_threshold))

            mezo_meta: Dict[str, Any] | None = None
            if (not passed) and int(args.mezo_iters) > 0 and layer_vectors:
                best_scalars, tuned_response, mezo_meta = mezo_tune_scalars(
                    model=model,
                    tokenizer=tokenizer,
                    spec=spec,
                    device=device,
                    max_new_tokens=int(args.max_new_tokens),
                    base_layer_vectors=layer_vectors,
                    init_scalars=base_scalars,
                    seq_len=seq_len,
                    iters=int(args.mezo_iters),
                    sigma=float(args.mezo_sigma),
                    lr=float(args.mezo_lr),
                    temperature=max(0.0, float(args.temperature)),
                    top_k=max(0, int(args.top_k)),
                    top_p=min(1.0, max(0.5, float(args.top_p))),
                    scalar_clip=float(args.mezo_scalar_clip),
                    rnd=rnd,
                )
                response = tuned_response
                hard_ok, hard_reason, _nll = hard_verify_response(
                    model=model,
                    tokenizer=tokenizer,
                    spec=spec,
                    prompt_for_eval=spec.prompt,
                    response=response,
                    seq_len=seq_len,
                    device=device,
                )
                soft = soft_judge_score(spec.target, response)
                passed = hard_ok if tier == "hard" else (hard_ok or soft >= float(args.soft_threshold))

            if hard_ok:
                pass_hard += 1
            if soft >= float(args.soft_threshold):
                pass_soft += 1

            if passed:
                skill_vecs = build_skill_vectors_from_success(
                    model=model,
                    tokenizer=tokenizer,
                    spec=spec,
                    layers=layers,
                    device=device,
                )
                kws = extract_keywords(spec.prompt, spec.target, max_k=14)
                for layer_idx, vec in skill_vecs.items():
                    added, _rec = memory.add_or_merge(
                        kind="skill",
                        name=f"skill_{spec.spec_id}",
                        layer=int(layer_idx),
                        vector=vec,
                        keywords=kws,
                        score=1.0 if hard_ok else max(0.0, min(1.0, soft)),
                        created_step=int(step),
                        meta={"spec_id": str(spec.spec_id), "tier": str(spec.tier)},
                    )
                    if added:
                        added_skill += 1
            else:
                fail_n += 1
                fail_focus[str(spec.spec_id)] = int(fail_focus.get(str(spec.spec_id), 0) + 3)
                patch_vecs = build_patch_vectors_from_failure(
                    model=model,
                    tokenizer=tokenizer,
                    spec=spec,
                    response=response,
                    layers=layers,
                    device=device,
                )
                kws = extract_keywords(spec.prompt, spec.target, max_k=14)
                for layer_idx, vec in patch_vecs.items():
                    added, _rec = memory.add_or_merge(
                        kind="patch",
                        name=f"patch_{spec.spec_id}",
                        layer=int(layer_idx),
                        vector=vec,
                        keywords=kws,
                        score=max(0.0, min(1.0, 1.0 - soft)),
                        created_step=int(step),
                        meta={"spec_id": str(spec.spec_id), "tier": str(spec.tier), "hard_reason": hard_reason},
                    )
                    if added:
                        added_patch += 1

                step_fail_rows.append(
                    {
                        "step": int(step),
                        "spec_id": str(spec.spec_id),
                        "tier": str(spec.tier),
                        "hard_ok": bool(hard_ok),
                        "hard_reason": str(hard_reason),
                        "soft_score": float(soft),
                        "response": str(response),
                        "retrieved_vectors": retrieved_ids,
                        "mezo": mezo_meta,
                    }
                )

        fail_focus = decay_focus(fail_focus, decay=float(args.focus_decay), max_size=int(args.focus_max))
        append_jsonl(fail_path, step_fail_rows)

        if step % int(args.save_interval) == 0 or step == int(args.steps) or added_skill > 0 or added_patch > 0:
            memory.save()

        checked = max(1, len(batch))
        row = {
            "step": int(step),
            "checked": int(len(batch)),
            "hard_pass_rate": float(pass_hard / checked),
            "soft_pass_rate": float(pass_soft / checked),
            "failures": int(fail_n),
            "added_skill_vectors": int(added_skill),
            "added_patch_vectors": int(added_patch),
            "memory_skill": int(memory.count("skill")),
            "memory_patch": int(memory.count("patch")),
            "focus_specs": int(len(fail_focus)),
            "elapsed_step_sec": float(time.time() - t0),
        }
        append_jsonl(history_path, [row])
        safe_json_print(row)

    memory.save()
    summary = {
        "finished": True,
        "elapsed_sec": float(time.time() - start_ts),
        "base_checkpoint": str(base_ckpt),
        "spec_source": spec_source,
        "steps": int(args.steps),
        "layers": layers,
        "memory_path": str(memory_path),
        "history_jsonl": str(history_path),
        "failure_ledger": str(fail_path),
        "memory_end": {"skill": int(memory.count("skill")), "patch": int(memory.count("patch"))},
        "model_args": model_args,
    }
    (out_dir / "omega3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    safe_json_print(summary)


if __name__ == "__main__":
    main()

