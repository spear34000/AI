from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch

from omega2_agentic_core import hard_verify_response, load_specs, soft_judge_score
from omega3_holographic_core import choose_default_layers, generate_with_steering, parse_layer_list
from train_ccl_compile import build_model_from_checkpoint, normalize_tier


def choose_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable")
        return torch.device("cuda")
    return torch.device("cpu")


def as_rate(num: int, den: int) -> float:
    if int(den) <= 0:
        return 0.0
    return float(num) / float(den)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Omega3 holographic steering runtime")
    p.add_argument("--base_checkpoint", default="artifacts_pure_ko_fresh/slm_best.pt")
    p.add_argument("--memory_path", default="artifacts_omega3_holographic_run1/holo_memory.pt")
    p.add_argument("--spec_path", default="data/ccl_specbook_v1.jsonl")
    p.add_argument("--data_path", default="data/slm_mit_unified_v4.jsonl")
    p.add_argument("--max_specs", type=int, default=120)
    p.add_argument("--layers", default="")
    p.add_argument("--memory_top_k", type=int, default=6)
    p.add_argument("--base_alpha", type=float, default=1.0)
    p.add_argument("--soft_threshold", type=float, default=0.42)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=80)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--output_json", default="artifacts_omega3_holographic_run1/eval_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    ckpt = Path(args.base_checkpoint)
    memory_path = Path(args.memory_path)
    if not ckpt.exists():
        raise RuntimeError(f"base checkpoint not found: {ckpt}")
    if not memory_path.exists():
        raise RuntimeError(f"memory path not found: {memory_path}")

    from omega3_holographic_core import HolographicVectorMemory

    model, tokenizer, model_args = build_model_from_checkpoint(ckpt, device=device)
    model.eval()
    seq_len = int(getattr(model, "seq_len", 384))
    d_model = int(model.head.in_features)
    layers = parse_layer_list(args.layers, model=model, default_max_layers=4)
    if not layers:
        layers = choose_default_layers(model, max_layers=4)

    memory = HolographicVectorMemory(path=memory_path)
    memory.load()

    specs, spec_source = load_specs(
        spec_path=Path(args.spec_path),
        data_path=Path(args.data_path),
        max_specs=int(args.max_specs),
        seed=int(args.seed),
    )
    if not specs:
        raise RuntimeError("no specs loaded")

    total = 0
    hard_pass = 0
    soft_pass = 0
    passed = 0
    tier_total: Dict[str, int] = defaultdict(int)
    tier_pass: Dict[str, int] = defaultdict(int)
    reason_counter: Counter[str] = Counter()
    fail_rows: List[Dict[str, Any]] = []

    for spec in specs:
        total += 1
        tier = normalize_tier(spec.tier)
        tier_total[tier] += 1

        retrieved = memory.retrieve(spec.prompt, top_k=int(args.memory_top_k))
        layer_vectors = memory.compose_layer_vectors(
            retrieved=retrieved,
            kind_weights={"skill": 1.0, "patch": 1.3},
            max_norm=2.0,
        )
        if not layer_vectors:
            layer_vectors = {int(l): torch.zeros(d_model, dtype=torch.float32) for l in layers}
        scalars = {int(l): float(args.base_alpha) for l in layer_vectors.keys()}

        response = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=spec.prompt,
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            layer_vectors=layer_vectors,
            alpha_by_layer=scalars,
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
        )

        hard_ok, reason, _nll = hard_verify_response(
            model=model,
            tokenizer=tokenizer,
            spec=spec,
            prompt_for_eval=spec.prompt,
            response=response,
            seq_len=seq_len,
            device=device,
        )
        soft = soft_judge_score(spec.target, response)
        ok = bool(hard_ok) if tier == "hard" else (bool(hard_ok) or float(soft) >= float(args.soft_threshold))

        if hard_ok:
            hard_pass += 1
        if float(soft) >= float(args.soft_threshold):
            soft_pass += 1
        if ok:
            passed += 1
            tier_pass[tier] += 1
        else:
            reason_counter[str(reason)] += 1
            fail_rows.append(
                {
                    "spec_id": str(spec.spec_id),
                    "tier": tier,
                    "hard_reason": str(reason),
                    "soft_score": float(soft),
                }
            )

    out = {
        "mode": "omega3_eval",
        "device": str(device),
        "base_checkpoint": str(ckpt),
        "memory_path": str(memory_path),
        "spec_source": spec_source,
        "max_specs": int(args.max_specs),
        "total": int(total),
        "hard_pass_rate": as_rate(hard_pass, total),
        "soft_pass_rate": as_rate(soft_pass, total),
        "pass_rate": as_rate(passed, total),
        "soft_threshold": float(args.soft_threshold),
        "layers": layers,
        "tier": {
            k: {
                "total": int(tier_total.get(k, 0)),
                "passed": int(tier_pass.get(k, 0)),
                "pass_rate": as_rate(int(tier_pass.get(k, 0)), int(tier_total.get(k, 0))),
            }
            for k in sorted(set(tier_total.keys()).union(tier_pass.keys()))
        },
        "top_fail_reasons": reason_counter.most_common(10),
        "failed_examples": fail_rows[:20],
        "memory": {
            "skill": int(memory.count("skill")),
            "patch": int(memory.count("patch")),
        },
        "model_args": model_args,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

