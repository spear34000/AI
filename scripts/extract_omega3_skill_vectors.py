from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from train_ccl_compile import build_model_from_checkpoint, pick_first_non_empty
from omega3_holographic_core import (
    HolographicVectorMemory,
    capture_layer_signature,
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


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            yield row


def group_key(row: Dict[str, Any]) -> str:
    task = str(row.get("task_type", "")).strip().lower()
    seg = str(row.get("segment_tag", "")).strip().lower()
    lang = str(row.get("language", "")).strip().lower()
    parts = [x for x in [task, seg, lang] if x]
    if not parts:
        return "default"
    return "|".join(parts)


def normalize_vector(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = v.detach().to(dtype=torch.float32, device="cpu")
    n = float(torch.norm(x).item())
    if n <= float(eps):
        return x
    return x / float(n)


def compose_group_vector(vectors: List[torch.Tensor], mode: str) -> torch.Tensor:
    if not vectors:
        raise ValueError("vectors must not be empty")
    stack = torch.stack([normalize_vector(v) for v in vectors], dim=0)
    mean_vec = normalize_vector(stack.mean(dim=0))

    if str(mode).strip().lower() != "pca" or int(stack.shape[0]) < 3:
        return mean_vec

    centered = stack - stack.mean(dim=0, keepdim=True)
    if float(torch.norm(centered).item()) <= 1e-8:
        return mean_vec

    try:
        _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        return mean_vec
    if vh.dim() != 2 or vh.shape[0] <= 0:
        return mean_vec

    pc1 = vh[0]
    if float(torch.dot(pc1, mean_vec).item()) < 0.0:
        pc1 = -pc1
    return normalize_vector(pc1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract omega3 skill vectors from successful samples")
    parser.add_argument("--checkpoint", default="artifacts_ondevice_best/slm_ondevice_fp16.pt")
    parser.add_argument("--data_path", default="data/pure_ko_seed_v1.jsonl")
    parser.add_argument("--memory_path", default="artifacts_omega3_holographic/holo_memory.pt")
    parser.add_argument("--max_rows", type=int, default=800)
    parser.add_argument("--max_per_group", type=int, default=120)
    parser.add_argument("--layers", type=str, default="")
    parser.add_argument("--vector_mode", choices=["mean", "pca"], default="pca")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    rnd = random.Random(int(args.seed))
    device = choose_device(args.device)
    ckpt = Path(args.checkpoint)
    data_path = Path(args.data_path)
    if not ckpt.exists():
        raise RuntimeError(f"checkpoint not found: {ckpt}")
    if not data_path.exists():
        raise RuntimeError(f"data_path not found: {data_path}")

    model, tokenizer, _ = build_model_from_checkpoint(ckpt, device=device)
    model.eval()
    layers = parse_layer_list(args.layers, model=model, default_max_layers=4)

    rows = list(iter_rows(data_path))
    rnd.shuffle(rows)
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    vector_mode = str(args.vector_mode).strip().lower()

    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for row in rows:
        prompt = pick_first_non_empty(row, ["input", "instruction", "prompt", "question", "context"])
        target = pick_first_non_empty(row, ["output", "response", "answer", "completion", "target"])
        if len(prompt) < 2 or len(target) < 2:
            continue
        key = group_key(row)
        if len(grouped[key]) >= int(args.max_per_group):
            continue
        grouped[key].append((prompt, target))

    memory = HolographicVectorMemory(path=Path(args.memory_path))
    memory.load()

    added = 0
    merged = 0
    for key, pairs in grouped.items():
        # Skill vectors are extracted per layer from successful hidden-state signatures.
        layer_samples: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for prompt, target in pairs:
            sig = capture_layer_signature(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                response=target,
                layers=layers,
                device=device,
            )
            for layer_idx, vec in sig.items():
                layer_samples[int(layer_idx)].append(vec.detach().to(dtype=torch.float32, device="cpu"))

        for layer_idx, samples in layer_samples.items():
            cnt = max(1, len(samples))
            rep_vec = compose_group_vector(samples, mode=vector_mode)
            kws = [k for k in key.split("|") if k]
            is_added, _rec = memory.add_or_merge(
                kind="skill",
                name=f"group_{key}_l{int(layer_idx)}",
                layer=int(layer_idx),
                vector=rep_vec,
                keywords=kws,
                score=0.9,
                created_step=0,
                meta={
                    "source": str(data_path),
                    "group": key,
                    "count": int(cnt),
                    "vector_mode": vector_mode,
                },
            )
            if is_added:
                added += 1
            else:
                merged += 1

    memory.save()
    print(
        json.dumps(
            {
                "memory_path": str(args.memory_path),
                "groups": int(len(grouped)),
                "layers": layers,
                "vector_mode": vector_mode,
                "added": int(added),
                "merged": int(merged),
                "skill_total": int(memory.count("skill")),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
