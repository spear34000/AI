from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def cast_state_dict(state: dict, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            t = v.detach().cpu()
            if t.is_floating_point():
                t = t.to(dtype=dtype)
            out[k] = t
        else:
            out[k] = v
    return out


def parse_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a compact inference-only checkpoint.")
    parser.add_argument("--input", required=True, help="Path to training checkpoint (*.pt)")
    parser.add_argument("--output", default="", help="Path to output checkpoint")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--use_ema", action="store_true", help="Export ema_model_state when available")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"input checkpoint not found: {in_path}")

    out_path = Path(args.output) if str(args.output).strip() else in_path.with_name("slm_infer_fp16.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    raw_state = ckpt.get("model_state")
    if bool(args.use_ema) and ckpt.get("ema_model_state") is not None:
        raw_state = ckpt.get("ema_model_state")
    if raw_state is None:
        raise RuntimeError("model_state not found in checkpoint")

    dtype = parse_dtype(args.dtype)
    payload = {
        "format": "slm_infer_only_v1",
        "source_checkpoint": str(in_path),
        "step": int(ckpt.get("step", 0)),
        "best_val_loss": float(ckpt.get("best_val_loss", 0.0)),
        "args": ckpt.get("args", {}),
        "tokenizer": ckpt.get("tokenizer", {"type": "byte", "vocab_size": 259}),
        "model_state": cast_state_dict(raw_state, dtype=dtype),
    }
    torch.save(payload, out_path)

    src_bytes = in_path.stat().st_size
    out_bytes = out_path.stat().st_size
    ratio = float(out_bytes) / float(max(1, src_bytes))
    print(
        json.dumps(
            {
                "input": str(in_path),
                "output": str(out_path),
                "input_bytes": int(src_bytes),
                "output_bytes": int(out_bytes),
                "size_ratio": ratio,
                "dtype": str(args.dtype),
                "use_ema": bool(args.use_ema),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
