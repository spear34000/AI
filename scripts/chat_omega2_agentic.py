from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from train_ccl_compile import build_model_from_checkpoint, generate_response
from omega2_agentic_core import PatchDB, build_runtime_prompt


def safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        enc = getattr(__import__("sys").stdout, "encoding", None) or "utf-8"
        fixed = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(fixed, flush=True)


def choose_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable")
        return torch.device("cuda")
    return torch.device("cpu")


def apply_repetition_penalty(logits: torch.Tensor, token_ids: List[int], penalty: float) -> torch.Tensor:
    if float(penalty) <= 1.0 or not token_ids:
        return logits
    out = logits.clone()
    for idx in set(int(t) for t in token_ids):
        if 0 <= idx < out.numel():
            val = out[idx]
            out[idx] = val * float(penalty) if val < 0 else val / float(penalty)
    return out


@torch.no_grad()
def generate_with_sampling(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> str:
    prefix = f"### Instruction\n{str(prompt).strip()}\n\n### Response\n"
    tokens = tokenizer.encode(prefix, add_bos=True, add_eos=False)
    seq_len = int(getattr(model, "seq_len", 384))

    for _ in range(max(1, int(max_new_tokens))):
        x = torch.tensor([tokens[-seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(x, targets=None)
        next_logits = logits[0, -1]
        next_logits = apply_repetition_penalty(next_logits, tokens, penalty=float(repetition_penalty))
        next_logits = next_logits / max(1e-5, float(temperature))

        if int(top_k) > 0:
            k = min(int(top_k), int(next_logits.numel()))
            kth = torch.topk(next_logits, k=k).values[-1]
            next_logits = torch.where(next_logits < kth, torch.full_like(next_logits, -float("inf")), next_logits)

        probs = torch.softmax(next_logits, dim=-1)
        if float(top_p) < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            mask = cdf > float(top_p)
            if torch.any(mask):
                first_idx = int(torch.where(mask)[0][0].item())
                sorted_probs[first_idx + 1 :] = 0.0
                probs = torch.zeros_like(probs)
                probs.scatter_(0, sorted_idx, sorted_probs)
                probs = probs / probs.sum().clamp_min(1e-9)

        next_id = int(torch.multinomial(probs, num_samples=1).item())
        tokens.append(next_id)
        if next_id == tokenizer.eos_id:
            break

    decoded = tokenizer.decode(tokens)
    if decoded.startswith(prefix):
        return decoded[len(prefix) :].strip()
    marker = "### Response"
    idx = decoded.rfind(marker)
    if idx >= 0:
        return decoded[idx + len(marker) :].lstrip(": \n\t").strip()
    return decoded.strip()


def run_one_turn(
    model: torch.nn.Module,
    tokenizer: Any,
    patch_db: PatchDB,
    user_prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    patch_top_k: int,
) -> Tuple[str, Dict[str, Any]]:
    retrieved = patch_db.retrieve(user_prompt, top_k=int(patch_top_k))
    patch_ids = [r.patch_id for r in retrieved]
    patch_db.mark_used(patch_ids)
    runtime_prompt = build_runtime_prompt(user_prompt, retrieved)

    if float(temperature) <= 0.0:
        answer = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=runtime_prompt,
            device=device,
            max_new_tokens=int(max_new_tokens),
        )
    else:
        answer = generate_with_sampling(
            model=model,
            tokenizer=tokenizer,
            prompt=runtime_prompt,
            device=device,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
        )

    meta = {
        "used_patches": len(retrieved),
        "used_patch_ids": patch_ids,
    }
    return answer, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Omega2 Agentic runtime patches")
    parser.add_argument("--base_checkpoint", default="artifacts_ondevice_best/slm_ondevice_fp16.pt")
    parser.add_argument("--patch_db", default="artifacts_omega2_agentic/patch_db.jsonl")
    parser.add_argument("--patch_top_k", type=int, default=4)
    parser.add_argument("--prompt", default="", help="single-turn prompt; if empty, interactive mode")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=60)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--show_meta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    base_ckpt = Path(args.base_checkpoint)
    if not base_ckpt.exists():
        raise RuntimeError(f"base checkpoint not found: {base_ckpt}")

    patch_db = PatchDB(path=Path(args.patch_db), novelty_threshold=0.9)
    patch_db.load()

    model, tokenizer, _ = build_model_from_checkpoint(base_ckpt, device=device)

    if str(args.prompt or "").strip():
        answer, meta = run_one_turn(
            model=model,
            tokenizer=tokenizer,
            patch_db=patch_db,
            user_prompt=str(args.prompt),
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            patch_top_k=int(args.patch_top_k),
        )
        safe_print(answer)
        if args.show_meta:
            safe_print(json.dumps(meta, ensure_ascii=False))
        patch_db.save()
        return

    safe_print("Interactive Omega2 mode. Type /exit to quit.")
    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            safe_print("")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "/exit", "/quit"}:
            break

        answer, meta = run_one_turn(
            model=model,
            tokenizer=tokenizer,
            patch_db=patch_db,
            user_prompt=q,
            device=device,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            repetition_penalty=float(args.repetition_penalty),
            patch_top_k=int(args.patch_top_k),
        )
        safe_print(answer)
        if args.show_meta:
            safe_print(json.dumps(meta, ensure_ascii=False))

    patch_db.save()


if __name__ == "__main__":
    main()

