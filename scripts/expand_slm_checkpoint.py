from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import torch

from train_slm import ByteTokenizer, TinyGPT, tokenizer_from_state


def copy_1d(dst: torch.Tensor, src: torch.Tensor) -> None:
    n = min(dst.size(0), src.size(0))
    dst[:n].copy_(src[:n])


def copy_2d(dst: torch.Tensor, src: torch.Tensor) -> None:
    r = min(dst.size(0), src.size(0))
    c = min(dst.size(1), src.size(1))
    dst[:r, :c].copy_(src[:r, :c])


def copy_qkv_weight(dst: torch.Tensor, src: torch.Tensor, old_d: int, new_d: int) -> None:
    copy_cols = min(old_d, new_d)
    for seg in range(3):
        src_r0 = seg * old_d
        src_r1 = src_r0 + old_d
        dst_r0 = seg * new_d
        dst_r1 = dst_r0 + min(old_d, new_d)
        dst[dst_r0:dst_r1, :copy_cols].copy_(src[src_r0:src_r1, :copy_cols])


def infer_n_layers(state: Dict[str, torch.Tensor]) -> int:
    idxs = []
    pat = re.compile(r"^blocks\.(\d+)\.")
    for k in state.keys():
        m = pat.match(k)
        if m:
            idxs.append(int(m.group(1)))
    return (max(idxs) + 1) if idxs else 0


def map_source_key(key: str, src_state: Dict[str, torch.Tensor], old_layers: int) -> str | None:
    if key in src_state:
        return key
    m = re.match(r"^blocks\.(\d+)\.(.+)$", key)
    if not m:
        return None
    idx = int(m.group(1))
    tail = m.group(2)
    if idx < old_layers:
        return None
    fallback = f"blocks.{max(old_layers - 1, 0)}.{tail}"
    if fallback in src_state:
        return fallback
    return None


def expand_state(
    src_state: Dict[str, torch.Tensor],
    new_state: Dict[str, torch.Tensor],
    old_d: int,
    new_d: int,
    old_layers: int,
) -> Dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in new_state.items()}

    for key, dst_t in out.items():
        src_key = map_source_key(key, src_state, old_layers)
        if src_key is None:
            continue
        src_t = src_state[src_key]
        if dst_t.ndim == 1 and src_t.ndim == 1:
            copy_1d(dst_t, src_t)
        elif dst_t.ndim == 2 and src_t.ndim == 2:
            if key.endswith("attn.qkv.weight"):
                copy_qkv_weight(dst_t, src_t, old_d=old_d, new_d=new_d)
            else:
                copy_2d(dst_t, src_t)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--new_d_model", type=int, default=400)
    parser.add_argument("--new_n_heads", type=int, default=5)
    parser.add_argument("--new_n_layers", type=int, default=8)
    parser.add_argument("--new_seq_len", type=int, default=384)
    parser.add_argument("--mlp_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        raise RuntimeError(f"source checkpoint not found: {src_path}")

    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    src_state = ckpt["model_state"]

    old_d = int(src_state["token_emb.weight"].shape[1])
    old_seq = int(src_state["pos_emb.weight"].shape[0])
    old_layers = infer_n_layers(src_state)

    tok_state = ckpt.get("tokenizer", {})
    tokenizer = tokenizer_from_state(tok_state) if isinstance(tok_state, dict) and tok_state else ByteTokenizer()
    new_model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        seq_len=int(args.new_seq_len),
        d_model=int(args.new_d_model),
        n_heads=int(args.new_n_heads),
        n_layers=int(args.new_n_layers),
        mlp_mult=int(args.mlp_mult),
        dropout=float(args.dropout),
    )

    new_state = expand_state(
        src_state=src_state,
        new_state=new_model.state_dict(),
        old_d=old_d,
        new_d=int(args.new_d_model),
        old_layers=old_layers,
    )
    new_model.load_state_dict(new_state)

    ema_out = None
    src_ema = ckpt.get("ema_model_state")
    if isinstance(src_ema, dict):
        ema_state = expand_state(
            src_state=src_ema,
            new_state=new_model.state_dict(),
            old_d=old_d,
            new_d=int(args.new_d_model),
            old_layers=old_layers,
        )
        ema_out = ema_state

    old_args = dict(ckpt.get("args", {}))
    old_args["d_model"] = int(args.new_d_model)
    old_args["n_heads"] = int(args.new_n_heads)
    old_args["n_layers"] = int(args.new_n_layers)
    old_args["seq_len"] = int(args.new_seq_len)
    old_args["mlp_mult"] = int(args.mlp_mult)
    old_args["dropout"] = float(args.dropout)
    old_args["resume_from"] = ""

    out_ckpt = {
        "model_state": new_model.state_dict(),
        "ema_model_state": ema_out,
        "step": 0,
        "best_val_loss": float("inf"),
        "args": old_args,
        "tokenizer": tok_state if isinstance(tok_state, dict) and tok_state else {"type": "byte"},
        "source_checkpoint": str(src_path),
    }

    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, dst_path)

    payload = {
        "src": str(src_path),
        "dst": str(dst_path),
        "old_d_model": old_d,
        "new_d_model": int(args.new_d_model),
        "old_layers": old_layers,
        "new_layers": int(args.new_n_layers),
        "old_seq_len": old_seq,
        "new_seq_len": int(args.new_seq_len),
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
