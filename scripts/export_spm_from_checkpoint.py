from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export embedded SentencePiece model from checkpoint.")
    p.add_argument("--checkpoint", required=True, help="checkpoint path")
    p.add_argument("--out_model", required=True, help="output .model path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out_model)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tok = ckpt.get("tokenizer", {})
    if not isinstance(tok, dict):
        raise RuntimeError("checkpoint tokenizer payload is missing")
    if str(tok.get("type", "")).strip().lower() != "spm":
        raise RuntimeError("checkpoint does not contain an SPM tokenizer")
    proto = tok.get("model_proto")
    if not isinstance(proto, (bytes, bytearray)):
        raise RuntimeError("SPM tokenizer model_proto is missing")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(proto))
    print(str(out_path))


if __name__ == "__main__":
    main()
