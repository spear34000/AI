from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from train_ccl_compile import build_model_from_checkpoint
from omega3_holographic_core import (
    HolographicVectorMemory,
    capture_layer_signature,
    choose_default_layers,
    compress_prompt_to_latent_state,
    generate_with_steering,
    parse_layer_list,
)


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


def merge_session_state(
    state: Dict[int, torch.Tensor],
    new_sig: Dict[int, torch.Tensor],
    decay: float,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    keys = set(state.keys()).union(new_sig.keys())
    for k in keys:
        a = state.get(k)
        b = new_sig.get(k)
        if a is None and b is not None:
            out[k] = b.detach().to(dtype=torch.float32, device="cpu")
            continue
        if b is None and a is not None:
            out[k] = a.detach().to(dtype=torch.float32, device="cpu")
            continue
        assert a is not None and b is not None
        v = float(decay) * a + (1.0 - float(decay)) * b
        n = float(torch.norm(v).item())
        if n > 1e-8:
            v = v / n
        out[k] = v.to(dtype=torch.float32, device="cpu")
    return out


def load_session_state(path: Path) -> Dict[int, torch.Tensor]:
    if not path.exists():
        return {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}

    raw: Any = payload
    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        raw = payload.get("state")
    if not isinstance(raw, dict):
        return {}

    out: Dict[int, torch.Tensor] = {}
    for k, v in raw.items():
        if not torch.is_tensor(v):
            continue
        try:
            idx = int(k)
        except Exception:
            continue
        out[idx] = v.detach().to(dtype=torch.float32, device="cpu")
    return out


def save_session_state(path: Path, state: Dict[int, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "omega3_session_state_v1",
        "state": {str(int(k)): v.detach().to(dtype=torch.float16, device="cpu") for k, v in state.items()},
    }
    torch.save(payload, path)


def run_one_turn(
    model: torch.nn.Module,
    tokenizer: Any,
    memory: HolographicVectorMemory,
    prompt: str,
    device: torch.device,
    layers: Sequence[int],
    session_state: Dict[int, torch.Tensor],
    session_alpha: float,
    prompt_state_alpha: float,
    memory_top_k: int,
    base_alpha: float,
    max_input_tokens: int,
    max_output_tokens: int,
    long_context_chunk_tokens: int,
    long_context_overlap_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tuple[str, Dict[str, Any], Dict[int, torch.Tensor]]:
    prompt_src = str(prompt or "").strip()
    prompt_for_generation, prompt_state, prompt_meta = compress_prompt_to_latent_state(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_src,
        layers=layers,
        device=device,
        max_input_tokens=max(64, int(max_input_tokens)),
        chunk_tokens=max(8, int(long_context_chunk_tokens)),
        overlap_tokens=max(0, int(long_context_overlap_tokens)),
        tail_tokens=48,
    )
    retrieved = memory.retrieve(prompt_src, top_k=int(memory_top_k))
    retrieved_ids = [rec.vector_id for _, rec in retrieved]
    memory.mark_used(retrieved_ids)

    layer_vectors = memory.compose_layer_vectors(
        retrieved=retrieved,
        kind_weights={"skill": 1.0, "patch": 1.3},
        max_norm=2.0,
    )

    if prompt_state:
        for layer_idx, pv in prompt_state.items():
            if layer_idx in layer_vectors:
                layer_vectors[layer_idx] = layer_vectors[layer_idx] + float(prompt_state_alpha) * pv
            else:
                layer_vectors[layer_idx] = float(prompt_state_alpha) * pv

    if session_state:
        for layer_idx, sv in session_state.items():
            if layer_idx in layer_vectors:
                layer_vectors[layer_idx] = layer_vectors[layer_idx] + float(session_alpha) * sv
            else:
                layer_vectors[layer_idx] = float(session_alpha) * sv

    if not layer_vectors:
        # Keep path stable even if memory is empty.
        d_model = int(model.head.in_features)
        layer_vectors = {int(l): torch.zeros(d_model, dtype=torch.float32) for l in layers}

    output_limit = max(1, min(int(max_new_tokens), int(max_output_tokens)))
    alpha_by_layer = {int(l): float(base_alpha) for l in layer_vectors.keys()}
    answer = generate_with_steering(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_for_generation,
        device=device,
        max_new_tokens=int(output_limit),
        layer_vectors=layer_vectors,
        alpha_by_layer=alpha_by_layer,
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )

    sig = capture_layer_signature(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        response=answer,
        layers=layers,
        device=device,
        tail_tokens=48,
    )

    meta = {
        "used_vectors": len(retrieved),
        "used_vector_ids": retrieved_ids,
        "session_layers": sorted(int(k) for k in session_state.keys()),
        "prompt_state_alpha": float(prompt_state_alpha),
        "max_output_tokens": int(max_output_tokens),
        "output_limit_used": int(output_limit),
    }
    meta.update(prompt_meta)
    return answer, meta, sig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with Omega 3.0 holographic steering runtime")
    parser.add_argument("--base_checkpoint", default="artifacts_ondevice_best/slm_ondevice_fp16.pt")
    parser.add_argument("--memory_path", default="artifacts_omega3_holographic/holo_memory.pt")
    parser.add_argument(
        "--session_state_path",
        default="",
        help="optional .pt path to persist compressed latent session state",
    )
    parser.add_argument("--no_state_persist", action="store_true")
    parser.add_argument("--state_reset", action="store_true")
    parser.add_argument("--prompt", default="", help="single-turn prompt; if empty, interactive mode")
    parser.add_argument("--layers", default="")
    parser.add_argument("--memory_top_k", type=int, default=6)
    parser.add_argument("--base_alpha", type=float, default=1.0)
    parser.add_argument("--prompt_state_alpha", type=float, default=0.65)
    parser.add_argument("--session_alpha", type=float, default=0.45)
    parser.add_argument("--session_decay", type=float, default=0.75)
    parser.add_argument("--disable_session_state", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max_input_tokens", type=int, default=39768)
    parser.add_argument("--max_output_tokens", type=int, default=38000)
    parser.add_argument("--long_context_chunk_tokens", type=int, default=256)
    parser.add_argument("--long_context_overlap_tokens", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=80)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--show_meta", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    ckpt = Path(args.base_checkpoint)
    if not ckpt.exists():
        raise RuntimeError(f"base checkpoint not found: {ckpt}")

    memory = HolographicVectorMemory(path=Path(args.memory_path))
    memory.load()

    model, tokenizer, _ = build_model_from_checkpoint(ckpt, device=device)
    model.eval()
    layers = parse_layer_list(args.layers, model=model, default_max_layers=4)
    if not layers:
        layers = choose_default_layers(model, max_layers=4)

    state_path = Path(str(args.session_state_path)) if str(args.session_state_path).strip() else None
    session_state: Dict[int, torch.Tensor] = {}
    if state_path is not None and not bool(args.state_reset):
        session_state = load_session_state(state_path)

    if str(args.prompt or "").strip():
        answer, meta, sig = run_one_turn(
            model=model,
            tokenizer=tokenizer,
            memory=memory,
            prompt=str(args.prompt),
            device=device,
            layers=layers,
            session_state=session_state if not bool(args.disable_session_state) else {},
            session_alpha=float(args.session_alpha),
            prompt_state_alpha=float(args.prompt_state_alpha),
            memory_top_k=int(args.memory_top_k),
            base_alpha=float(args.base_alpha),
            max_input_tokens=int(args.max_input_tokens),
            max_output_tokens=int(args.max_output_tokens),
            long_context_chunk_tokens=int(args.long_context_chunk_tokens),
            long_context_overlap_tokens=int(args.long_context_overlap_tokens),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
        )
        if not bool(args.disable_session_state):
            session_state = merge_session_state(session_state, sig, decay=float(args.session_decay))
            if state_path is not None and not bool(args.no_state_persist):
                save_session_state(state_path, session_state)
        safe_print(answer)
        if args.show_meta:
            safe_print(json.dumps(meta, ensure_ascii=False))
        memory.save()
        return

    safe_print("Interactive Omega3 mode. Type /exit to quit.")
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

        answer, meta, sig = run_one_turn(
            model=model,
            tokenizer=tokenizer,
            memory=memory,
            prompt=q,
            device=device,
            layers=layers,
            session_state=session_state if not bool(args.disable_session_state) else {},
            session_alpha=float(args.session_alpha),
            prompt_state_alpha=float(args.prompt_state_alpha),
            memory_top_k=int(args.memory_top_k),
            base_alpha=float(args.base_alpha),
            max_input_tokens=int(args.max_input_tokens),
            max_output_tokens=int(args.max_output_tokens),
            long_context_chunk_tokens=int(args.long_context_chunk_tokens),
            long_context_overlap_tokens=int(args.long_context_overlap_tokens),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
        )
        if not bool(args.disable_session_state):
            session_state = merge_session_state(session_state, sig, decay=float(args.session_decay))
            if state_path is not None and not bool(args.no_state_persist):
                save_session_state(state_path, session_state)
        safe_print(answer)
        if args.show_meta:
            safe_print(json.dumps(meta, ensure_ascii=False))

    memory.save()
    if state_path is not None and not bool(args.disable_session_state) and not bool(args.no_state_persist):
        save_session_state(state_path, session_state)


if __name__ == "__main__":
    main()
