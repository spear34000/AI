from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from train_slm import (
    JsonlCausalDataset,
    TinyGPT,
    dataset_file_fingerprint,
    evaluate,
    safe_json_print,
    tokenizer_fingerprint,
    tokenizer_from_state,
)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model_from_ckpt(ckpt: Dict[str, Any], tokenizer: Any, device: torch.device) -> TinyGPT:
    args = ckpt.get("args", {})
    model = TinyGPT(
        vocab_size=int(tokenizer.vocab_size),
        seq_len=int(args["seq_len"]),
        d_model=int(args["d_model"]),
        n_heads=int(args["n_heads"]),
        n_layers=int(args["n_layers"]),
        mlp_mult=int(args.get("mlp_mult", 4)),
        dropout=float(args.get("dropout", 0.1)),
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--max_records", type=int, default=0)
    ap.add_argument("--fixed_val_count", type=int, default=0, help="0 means use all val records")
    ap.add_argument("--eval_batches", type=int, default=0, help="0 means full validation loader")
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise RuntimeError(f"data_path not found: {data_path}")

    results: List[Dict[str, Any]] = []

    for ckpt_path_str in args.checkpoints:
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.exists():
            raise RuntimeError(f"checkpoint not found: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path)
        tokenizer = tokenizer_from_state(ckpt.get("tokenizer", {}))
        tok_fp = tokenizer_fingerprint(tokenizer)
        train_args = ckpt.get("args", {})

        val_ds = JsonlCausalDataset(
            path=data_path,
            tokenizer=tokenizer,
            seq_len=int(train_args["seq_len"]),
            split=str(args.split),
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
            max_records=int(args.max_records),
        )
        if int(args.fixed_val_count) > 0 and len(val_ds.refs) > int(args.fixed_val_count):
            val_ds.refs = list(val_ds.refs[: int(args.fixed_val_count)])

        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=(device.type == "cuda"),
        )

        model = build_model_from_ckpt(ckpt=ckpt, tokenizer=tokenizer, device=device)
        max_batches = int(args.eval_batches) if int(args.eval_batches) > 0 else len(val_loader)
        val_loss, batches_used = evaluate(model, val_loader, device=device, max_batches=max_batches)
        ppl = float(math.exp(val_loss)) if math.isfinite(val_loss) and val_loss < 20.0 else float("inf")

        row = {
            "checkpoint": str(ckpt_path),
            "step": int(ckpt.get("step", 0)),
            "trainer_best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
            "eval_val_loss": float(val_loss),
            "eval_ppl": float(ppl),
            "val_records": int(len(val_ds)),
            "val_batches_used": int(batches_used),
            "val_signature": val_ds.signature(),
            "val_preview": val_ds.preview_stats(max_items=256),
            "dataset": dataset_file_fingerprint(data_path),
            "tokenizer": tok_fp,
            "device": str(device),
        }
        results.append(row)
        safe_json_print(row)

    summary = {
        "finished": True,
        "count": len(results),
        "results": results,
    }
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    safe_json_print(summary)


if __name__ == "__main__":
    main()
